// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <ips4o.hpp>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kassert/kassert.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <tlx/algorithm/multiway_merge.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/multi_level.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace bloomfilter2 {

using bloomfilter::hash_less;
using bloomfilter::hash_t;
using bloomfilter::HashRange;
using bloomfilter::HashRank;
using bloomfilter::kamping_agg;
using bloomfilter::SipHasher;
using bloomfilter::XXHasher;

using bloomfilter::HashStringIndex;

namespace _internal = bloomfilter::_internal;


namespace _debug {

template <typename StringSet>
std::vector<std::string> strings_of(StringSet const& ss) {
    std::vector<std::string> out;
    out.reserve(ss.size());
    for (auto it = ss.begin(); it != ss.end(); ++it) {
        auto const& str = ss[it];
        auto const* chars = reinterpret_cast<char const*>(ss.get_chars(str, 0));
        out.emplace_back(chars, ss.get_length(str));
    }
    return out;
}

inline std::vector<std::string> hashes_of(std::vector<HashStringIndex> const& pairs) {
    std::vector<std::string> out;
    out.reserve(pairs.size());
    for (auto const& pair: pairs) {
        out.push_back(
            fmt::format(
                "{}:{:016x}{}",
                pair.string_index,
                pair.hash_value,
                pair.is_lcp_root ? "*" : ""
            )
        );
    }
    return out;
}

} // namespace _debug


template <typename BloomFilterV1>
struct v1_traits;

template <bool reuse, typename HashPolicy>
struct v1_traits<bloomfilter::SingleLevel<reuse, HashPolicy>> {
    using hash_policy = HashPolicy;
    static constexpr bool is_grid = false;
};

template <bool reuse, typename HashPolicy>
struct v1_traits<bloomfilter::MultiLevel<reuse, HashPolicy>> {
    using hash_policy = HashPolicy;
    static constexpr bool is_grid = true;
};


template <typename HashPolicy>
class HashGenerator {
public:
    struct Result {
        std::vector<HashStringIndex>
            representatives; // strings participating in duplicate detection
        std::vector<size_t>
            lcp_duplicates; // strings sharing a `depth`-prefix with their predecessor, ascending
        std::vector<size_t>
            eos_candidates; // strings shorter than `depth`, already distinguished, ascending
    };

    explicit HashGenerator(size_t const num_strings) : hash_values_(num_strings) {}

    // First round: all strings are candidates, hashes computed from scratch.
    template <typename StringSet, typename LcpIter>
    Result generate(StringSet const& ss, size_t const depth, LcpIter const lcps) {
        KASSERT(depth > 0u);

        Result result;
        if (ss.empty()) {
            return result;
        }
        result.representatives.reserve(ss.size());

        hash_t curr_hash = 0;
        size_t candidate = 0;
        for (auto it = ss.begin(); it != ss.end(); ++it, ++candidate) {
            auto const& curr_str = ss[it];

            if (depth > ss.get_length(curr_str)) {
                result.eos_candidates.push_back(candidate);
                continue;
            }

            if (lcps[candidate] >= depth) {
                // non-empty: `lcps[0] == 0 < depth`, and `lcps[k] <= len(s_{k-1}) < depth`
                // after an EOS string, so the string following one always hashes
                KASSERT(!result.representatives.empty());

                // `curr_hash` still applies: same first `depth` characters
                result.lcp_duplicates.push_back(candidate);
                mark_lcp_root(result.representatives, candidate);
            } else {
                curr_hash = HashPolicy::hash(ss.get_chars(curr_str, 0), depth);
                result.representatives.emplace_back(curr_hash, candidate);
            }
            hash_values_[candidate] = curr_hash;
        }
        KASSERT(std::is_sorted(result.lcp_duplicates.begin(), result.lcp_duplicates.end()));
        return result;
    }

    // Later rounds: combines the cached `depth / 2` hash with the second half.
    template <typename StringSet, typename LcpIter>
    Result generate(
        StringSet const& ss,
        std::span<size_t const> candidates,
        size_t const depth,
        LcpIter const lcps
    ) {
        KASSERT(depth > 0u);

        Result result;
        if (candidates.empty()) {
            return result;
        }
        size_t const half_depth = depth / 2;
        result.representatives.reserve(candidates.size());

        hash_t curr_hash = 0;
        for (auto prev = candidates.front(); auto const& curr: candidates) {
            auto const& curr_str = ss.at(curr);

            if (depth > ss.get_length(curr_str)) {
                result.eos_candidates.push_back(curr); // `prev` deliberately not advanced
                continue;
            }

            // `lcps[curr]` relates `curr` to its neighbour, not to `prev`
            if (prev + 1 == curr && lcps[curr] >= depth) {
                KASSERT(!result.representatives.empty());

                result.lcp_duplicates.push_back(curr);
                mark_lcp_root(result.representatives, curr);
            } else {
                auto const chars = ss.get_chars(curr_str, half_depth);
                auto const suffix_hash = HashPolicy::hash(chars, half_depth);
                curr_hash = HashPolicy::combine(hash_values_[curr], suffix_hash);
                result.representatives.emplace_back(curr_hash, curr);
            }
            hash_values_[curr] = curr_hash;
            prev = curr;
        }
        KASSERT(std::is_sorted(result.lcp_duplicates.begin(), result.lcp_duplicates.end()));
        return result;
    }

private:
    std::vector<hash_t> hash_values_; // cache for hash values computed in previous rounds

    static void
    mark_lcp_root(std::vector<HashStringIndex>& representatives, size_t const candidate) {
        if (representatives.back().string_index + 1 == candidate) {
            representatives.back().is_lcp_root = true;
        }
    }
};


class RemoteDuplicateDetector {
public:
    virtual ~RemoteDuplicateDetector() = default;

    virtual Communicator const& comm_root() const = 0;

    virtual std::optional<std::vector<int>>
    find(std::vector<HashStringIndex> const& representatives, HashRange hash_range) = 0;
};


class AlltoallDuplicateDetector final : public RemoteDuplicateDetector {
public:
    AlltoallDuplicateDetector(Communicator const& comm_root, std::vector<Communicator> comms)
        : comm_root_{comm_root},
          comms_{std::move(comms)} {
        KASSERT(!comms_.empty());
    }

    Communicator const& comm_root() const override { return comm_root_; }

    std::optional<std::vector<int>>
    find(std::vector<HashStringIndex> const& hash_str_pairs, HashRange const hash_range) override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("bloomfilter_send_hashes");
        kamping::measurements::timer().start("bloomfilter_send_hashes");
        auto duplicates = find_recursive(comms_.begin(), comms_.end(), hash_str_pairs, hash_range);
        measuring_tool.stop("bloomfilter_send_indices");
        kamping::measurements::timer().stop_and_append();

        return duplicates;
    }

private:
    Communicator const& comm_root_;
    std::vector<Communicator> comms_;

    template <typename CommIt, typename T>
    std::optional<std::vector<int>> find_recursive(
        CommIt const comm_first,
        CommIt const comm_last,
        std::vector<T> const& hash_pairs,
        HashRange const hash_range
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        KASSERT(comm_first != comm_last);
        auto const& comm = *comm_first;

        auto hash_values = _internal::extract_hash_values(hash_pairs);
        auto recv_data = _internal::send_hash_values(hash_values, hash_range, comm);
        auto hash_rank_pairs = _internal::merge_intervals(
            recv_data.compute_hash_rank_pairs(),
            recv_data.local_offsets,
            recv_data.interval_sizes
        );

        if (comm_first + 1 == comm_last) {
            measuring_tool.add(hash_rank_pairs.size(), "bloomfilter_recv_hash_values");
            kamping::measurements::counter().append(
                "bloomfilter_recv_hash_values",
                static_cast<std::int64_t>(hash_rank_pairs.size()),
                kamping_agg
            );
            measuring_tool.stop("bloomfilter_send_hashes");
            kamping::measurements::timer().stop_and_append();

            measuring_tool.start("bloomfilter_compute_remote_duplicates");
            kamping::measurements::timer().start("bloomfilter_compute_remote_duplicates");
            auto result = _internal::compute_duplicates(
                hash_rank_pairs,
                recv_data.interval_sizes,
                recv_data.global_offsets
            );
            measuring_tool.add(result.duplicates.size(), "bloomfilter_remote_duplicates");
            kamping::measurements::counter().append(
                "bloomfilter_remote_duplicates",
                static_cast<std::int64_t>(result.duplicates.size()),
                kamping_agg
            );
            measuring_tool.stop("bloomfilter_compute_remote_duplicates");
            kamping::measurements::timer().stop_and_append();

            measuring_tool.start("bloomfilter_send_indices");
            kamping::measurements::timer().start("bloomfilter_send_indices");
            return _internal::send_duplicates(
                result.duplicates,
                result.send_counts,
                result.send_displs,
                comm,
                comm_root_
            );

        } else {
            auto bucket = hash_range.bucket(comm.rank(), comm.size());
            auto duplicates = find_recursive(comm_first + 1, comm_last, hash_rank_pairs, bucket);
            if (duplicates) {
                auto& global_offsets = recv_data.global_offsets;
                return send_dups_recursive(hash_rank_pairs, *duplicates, global_offsets, comm);
            } else {
                return {};
            }
        }
    }

    static std::vector<int> send_dups_recursive(
        std::vector<HashRank> const& hash_rank_pairs,
        std::vector<int> const& duplicates,
        std::vector<int>& global_offsets,
        Communicator const& comm
    ) {
        std::vector<int> send_counts(global_offsets.size());
        for (auto const& duplicate: duplicates) {
            send_counts[hash_rank_pairs[duplicate].rank]++;
        }

        std::vector<int> offsets{send_counts};
        std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), 0);
        std::vector<int> send_displs{offsets};

        std::vector<int> remote_idxs(duplicates.size());
        auto counters = std::move(global_offsets);
        for (int i = 0; auto const& duplicate: duplicates) {
            for (; i < duplicate; ++i) {
                counters[hash_rank_pairs[i].rank]++;
            }
            auto rank = hash_rank_pairs[i].rank;
            remote_idxs[offsets[rank]++] = counters[rank];
        }

        kamping::measurements::timer().start("bloomfilter_alltoall_indices");
        auto result = comm.alltoallv(
            kamping::send_buf(remote_idxs),
            kamping::send_counts(send_counts),
            kamping::send_displs(send_displs)
        );
        kamping::measurements::timer().stop_and_append();
        return result;
    }
};


// Short-circuits `inner` when globally every PE holds at most one hash value: the
// alltoall machinery then collapses to a single allgather plus a local check.
class BaseCaseDetector final : public RemoteDuplicateDetector {
public:
    explicit BaseCaseDetector(std::unique_ptr<RemoteDuplicateDetector> inner)
        : inner_{std::move(inner)} {}

    Communicator const& comm_root() const override { return inner_->comm_root(); }

    std::optional<std::vector<int>>
    find(std::vector<HashStringIndex> const& hash_str_pairs, HashRange const hash_range) override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("bloomfilter_base_case");
        kamping::measurements::timer().start("bloomfilter_base_case");
        // `nullopt` here means "does not apply", unlike the `nullopt` we return
        auto base_case = _internal::find_remote_duplicates_base_case(hash_str_pairs, comm_root());
        measuring_tool.stop("bloomfilter_base_case");
        kamping::measurements::timer().stop_and_append();

        if (base_case) {
            return base_case;
        }
        return inner_->find(hash_str_pairs, hash_range);
    }

private:
    std::unique_ptr<RemoteDuplicateDetector> inner_;
};


template <typename Subcommunicators>
inline std::unique_ptr<RemoteDuplicateDetector> make_remote_duplicate_detector(
    Subcommunicators const& comms, bool const grid, bool const enable_base_case
) {
    // single-level == a grid whose only level is the root communicator
    std::vector<Communicator> levels;
    if (grid) {
        levels = multi_level::GridCommunicators<Communicator>{comms}.comms;
    } else {
        levels.push_back(comms.comm_root());
    }

    std::unique_ptr<RemoteDuplicateDetector> remote_duplicate_detector =
        std::make_unique<AlltoallDuplicateDetector>(comms.comm_root(), std::move(levels));

    if (enable_base_case) {
        remote_duplicate_detector =
            std::make_unique<BaseCaseDetector>(std::move(remote_duplicate_detector));
    }
    return remote_duplicate_detector;
}


class BloomFilter {
public:
    struct Duplicates {
        std::vector<size_t> local;  // string indices with a collioson with another local string
        std::vector<size_t> remote; // string indices with a collision on another PE
    };

    explicit BloomFilter(RemoteDuplicateDetector& remote_duplicate_detector)
        : remote_duplicate_detector_{remote_duplicate_detector} {}

    Duplicates find_duplicates(std::vector<HashStringIndex>&& representatives) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        auto hash_idx_pairs = std::move(representatives);

        measuring_tool.start("bloomfilter_sort_local_hashes");
        kamping::measurements::timer().start("bloomfilter_sort_local_hashes");
        ips4o::sort(hash_idx_pairs.begin(), hash_idx_pairs.end(), hash_less<HashStringIndex>{});
        measuring_tool.stop("bloomfilter_sort_local_hashes");
        kamping::measurements::timer().stop_and_append();

        measuring_tool.start("bloomfilter_find_local_duplicates");
        kamping::measurements::timer().start("bloomfilter_find_local_duplicates");
        auto local_dups = get_local_duplicates(hash_idx_pairs);
        std::erase_if(hash_idx_pairs, std::not_fn(should_send));
        measuring_tool.stop("bloomfilter_find_local_duplicates");
        kamping::measurements::timer().stop_and_append();

        SPDLOG_DEBUG("  local dups  = {}", local_dups);
        SPDLOG_DEBUG("  hashes sent = {}", _debug::hashes_of(hash_idx_pairs));

        // the returned `int`s index the *post-erase* vector, so erase and prune must stay together
        measuring_tool.start("bloomfilter_find_remote_duplicates");
        kamping::measurements::timer().start("bloomfilter_find_remote_duplicates");
        auto const positions =
            remote_duplicate_detector_.find(hash_idx_pairs, universe).value_or(std::vector<int>{});
        auto remote_dups = prune_remote_duplicates(positions, hash_idx_pairs);
        measuring_tool.stop("bloomfilter_find_remote_duplicates");
        kamping::measurements::timer().stop_and_append();

        SPDLOG_DEBUG("  remote positions = {}", positions);
        SPDLOG_DEBUG("  remote dups      = {}", remote_dups);

        return {std::move(local_dups), std::move(remote_dups)};
    }

private:
    static constexpr HashRange universe{0, std::numeric_limits<hash_t>::max()};

    RemoteDuplicateDetector& remote_duplicate_detector_;

    static bool should_send(HashStringIndex const& v) noexcept {
        return !v.is_local_dup || v.send_anyway;
    }

    static std::vector<size_t> get_local_duplicates(std::vector<HashStringIndex>& local_values) {
        std::vector<size_t> local_duplicates;
        if (!local_values.empty()) {
            for (auto it = local_values.begin(); it < local_values.end() - 1;) {
                auto& pivot = *it++;
                if (it->hash_value == pivot.hash_value) {
                    pivot.is_local_dup = true;
                    pivot.send_anyway = true;
                    local_duplicates.push_back(pivot.string_index);

                    do {
                        it->is_local_dup = true;
                        local_duplicates.push_back(it->string_index);
                    } while (++it != local_values.end() && it->hash_value == pivot.hash_value);

                } else if (pivot.is_lcp_root) {
                    pivot.is_local_dup = true;
                    pivot.send_anyway = true;
                    local_duplicates.push_back(pivot.string_index);
                }
            }
            // the last element never is a pivot; skip it if an equal-hash run already recored it
            auto& last = local_values.back();
            if (last.is_lcp_root && !last.is_local_dup) {
                last.is_local_dup = true;
                last.send_anyway = true;
                local_duplicates.push_back(last.string_index);
            }
        }
        return local_duplicates;
    }

    static std::vector<size_t> prune_remote_duplicates(
        std::vector<int> const& duplicates, std::vector<HashStringIndex> const& hash_idx_pairs
    ) {
        std::vector<size_t> pruned_duplicates;
        pruned_duplicates.reserve(duplicates.size());

        for (auto const& duplicate: duplicates) {
            if (auto const& orig_pair = hash_idx_pairs[duplicate]; !orig_pair.send_anyway) {
                pruned_duplicates.push_back(orig_pair.string_index);
            }
        }
        return pruned_duplicates;
    }
};


inline std::vector<size_t> merge_duplicates(
    std::vector<size_t>& local_hash_dups,
    std::vector<size_t>& local_lcp_dups,
    std::vector<size_t>& remote_dups
) {
    KASSERT(std::is_sorted(local_lcp_dups.begin(), local_lcp_dups.end()));
    ips4o::sort(remote_dups.begin(), remote_dups.end());
    ips4o::sort(local_hash_dups.begin(), local_hash_dups.end());

    using Iterator = std::vector<size_t>::iterator;
    using IteratorPair = std::pair<Iterator, Iterator>;
    std::array<IteratorPair, 3> iter_pairs{
        {{local_hash_dups.begin(), local_hash_dups.end()},
         {local_lcp_dups.begin(), local_lcp_dups.end()},
         {remote_dups.begin(), remote_dups.end()}}
    };
    size_t const num_merged_elems =
        local_hash_dups.size() + local_lcp_dups.size() + remote_dups.size();

    std::vector<size_t> merged_elems(num_merged_elems);
    tlx::multiway_merge(
        iter_pairs.begin(),
        iter_pairs.end(),
        merged_elems.begin(),
        num_merged_elems
    );
    // the three inputs need to be pairwise disjoint sets
    KASSERT(std::adjacent_find(merged_elems.begin(), merged_elems.end()) == merged_elems.end());
    return merged_elems;
}

template <typename StringSet>
void set_depth(
    StringSet const& ss,
    size_t const depth,
    std::optional<std::span<size_t const>> candidates,
    std::span<size_t const> eos_candidates,
    std::span<size_t> results
) {
    if (candidates.has_value()) {
        for (auto const& candidate: candidates.value()) {
            results[candidate] = depth;
        }
    } else {
        std::fill(results.begin(), results.end(), depth);
    }


    for (auto const& candidate: eos_candidates) {
        results[candidate] = ss.get_length(ss.at(candidate));
    }
}

namespace internal {
// One round with at a the given depth.
// Returns the candidates detect as duplicates are returned for the next round.
template <typename StringPtr, typename HashPolicy>
std::vector<size_t> detect_duplicates_at_depth(
    StringPtr const& strptr,
    size_t const depth,
    size_t const round,
    HashGenerator<HashPolicy>& generator,
    BloomFilter& filter,
    std::vector<size_t>& results,
    std::optional<std::span<size_t const>> candidates
) {
    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    measuring_tool.setRound(round);

    auto const& ss = strptr.active();

    measuring_tool.start("bloomfilter_generate_hash_pairs");
    kamping::measurements::timer().start("bloomfilter_generate_hash_pairs");
    auto hashes = [&]() {
        if (candidates.has_value()) {
            return generator.generate(ss, candidates.value(), depth, strptr.lcp());
        } else {
            return generator.generate(ss, depth, strptr.lcp());
        }
    }();
    measuring_tool.stop("bloomfilter_generate_hash_pairs");
    kamping::measurements::timer().stop_and_append();

    SPDLOG_DEBUG("=== round {}, depth {} ===", round, depth);
    SPDLOG_DEBUG("  candidates      = {}", candidates.value_or(std::span<size_t const>{}));
    SPDLOG_DEBUG("  representatives = {}", _debug::hashes_of(hashes.representatives));
    SPDLOG_DEBUG("  lcp dups        = {}", hashes.lcp_duplicates);
    SPDLOG_DEBUG("  eos candidates  = {}", hashes.eos_candidates);

    auto dups = filter.find_duplicates(std::move(hashes.representatives));

    measuring_tool.start("bloomfilter_merge_duplicates");
    kamping::measurements::timer().start("bloomfilter_merge_duplicates");
    auto const final_duplicates = merge_duplicates(dups.local, hashes.lcp_duplicates, dups.remote);
    measuring_tool.stop("bloomfilter_merge_duplicates");
    kamping::measurements::timer().stop_and_append();

    SPDLOG_DEBUG("  --> duplicates ({}) = {}", final_duplicates.size(), final_duplicates);

    measuring_tool.start("bloomfilter_write_depth");
    kamping::measurements::timer().start("bloomfilter_write_depth");


    set_depth(ss, depth, candidates, hashes.eos_candidates, results);
    measuring_tool.stop("bloomfilter_write_depth");
    kamping::measurements::timer().stop_and_append();

    return final_duplicates;
}
} // namespace internal


// Prefix doubling: repeatedly double `depth` until no PE holds a candidate whose
// `depth`-prefix is shared with another string. Belongs in `prefix_doubling.hpp`.
template <typename HashPolicy, typename StringPtr, typename Subcommunicators>
std::vector<size_t> compute_distinguishing_prefixes(
    StringPtr const& strptr,
    Subcommunicators const& comms,
    size_t const start_depth,
    RemoteDuplicateDetector& remote_duplicate_detector
) {
    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    auto const& ss = strptr.active();
    std::vector<size_t> results(ss.size());

    SPDLOG_DEBUG("bloomfilter input: {} strings = {}", ss.size(), _debug::strings_of(ss));
    SPDLOG_DEBUG("bloomfilter input: lcps = {}", std::span<size_t const>{strptr.lcp(), ss.size()});

    // the generator outlives the rounds: its cache is what makes the halved rehash work
    HashGenerator<HashPolicy> generator{ss.size()};
    BloomFilter filter{remote_duplicate_detector};

    size_t round = 0;
    auto candidates = internal::detect_duplicates_at_depth(
        strptr,
        start_depth,
        round,
        generator,
        filter,
        results,
        std::nullopt
    );

    for (size_t i = start_depth * 2; i < std::numeric_limits<size_t>::max(); i *= 2) {
        measuring_tool.add(candidates.size(), "bloomfilter_num_candidates");
        kamping::measurements::counter().append(
            "bloomfilter_num_candidates",
            static_cast<std::int64_t>(candidates.size()),
            kamping_agg
        );

        measuring_tool.start("bloomfilter_allreduce");
        kamping::measurements::timer().start("bloomfilter_allreduce");
        auto const all_empty = comms.comm_root().allreduce_single(
            kamping::send_buf(candidates.empty()),
            kamping::op(std::logical_and<>{})
        );
        measuring_tool.stop("bloomfilter_allreduce");
        kamping::measurements::timer().stop_and_append();

        if (all_empty) {
            break;
        }

        candidates = internal::detect_duplicates_at_depth(
            strptr,
            i,
            ++round,
            generator,
            filter,
            results,
            std::span<std::size_t const>(candidates)
        );
    }

    measuring_tool.setRound(0);
    return results;
}

} // namespace bloomfilter2
} // namespace dss_mehnert
