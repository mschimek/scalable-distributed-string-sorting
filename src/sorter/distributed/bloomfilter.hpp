// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include <ips4o.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/algorithm/multiway_merge.hpp>
#include <tlx/die.hpp>
#include <tlx/siphash.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "hash/xxhash.hpp"
#include "mpi/communicator.hpp"
#include "sorter/distributed/multi_level.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace bloomfilter {

using hash_t = std::uint64_t;

// todo use uint64_t for hash_values
struct HashStringIndex {
    hash_t hash_value;
    size_t string_index;
    bool is_local_dup = false;
    bool send_anyway = false;
    bool is_lcp_root = false;

    bool operator<(HashStringIndex const& rhs) const { return hash_value < rhs.hash_value; }

    friend std::ostream& operator<<(std::ostream& stream, HashStringIndex const& hash_pair) {
        return stream << "[" << hash_pair.hash_value << ", " << hash_pair.string_index
                      << (hash_pair.is_local_dup ? ", is_local_dup: " : "")
                      << (hash_pair.send_anyway ? ", send_anyway" : "")
                      << (hash_pair.is_lcp_root ? ", is_lcp_root" : "") << "]";
    }
};

struct HashRank {
    hash_t hash_value;
    // todo update PEIndex to use int
    int rank;

    bool operator<(HashRank const& rhs) const { return hash_value < rhs.hash_value; }

    friend std::ostream& operator<<(std::ostream& stream, HashRank const& hash_pair) {
        return stream << "[" << hash_pair.hash_value << ", " << hash_pair.rank << "]";
    }
};

struct SipHasher {
    static inline hash_t hash(unsigned char const* str, size_t length, size_t filter_size) {
        return tlx::siphash(str, length) % filter_size;
    }
};

struct XXHasher {
    static inline hash_t hash(unsigned char const* str, size_t length, size_t filter_size) {
        xxh::hash_state_t<64> hash_stream;
        hash_stream.update(str, length);
        xxh::hash_t<64> hashV = hash_stream.digest();
        return hashV % filter_size;
    }

    static inline hash_t
    hash(unsigned char const* str, size_t length, size_t filter_size, hash_t old_hash) {
        xxh::hash_state_t<64> hash_stream;
        hash_stream.update(str, length);
        xxh::hash_t<64> new_hash = hash_stream.digest();
        return (old_hash ^ new_hash) % filter_size;
    }
};

struct HashRange {
    hash_t lower;
    hash_t upper;

    HashRange bucket(size_t idx, size_t num_buckets) const {
        auto bucket_size = this->bucket_size(num_buckets);
        auto bucket_lower = lower + idx * bucket_size;
        auto bucket_upper = lower + (idx + 1) * bucket_size - 1;

        if (idx + 1 == num_buckets) {
            return {bucket_lower, upper};
        } else {
            return {bucket_lower, bucket_upper};
        }
    }

    size_t bucket_size(size_t num_buckets) const { return (upper - lower) / num_buckets; }
};

struct RecvData {
    std::vector<hash_t> hashes;
    std::vector<int> interval_sizes;
    std::vector<int> local_offsets;
    std::vector<int> global_offsets;

    std::vector<HashRank> compute_hash_rank_pairs() const {
        std::vector<HashRank> hash_pairs(hashes.size());

        auto hash_it = hashes.begin();
        auto dest_it = hash_pairs.begin();
        for (int rank = 0; auto const& interval: interval_sizes) {
            for (auto const end = hash_it + interval; hash_it != end; ++hash_it) {
                *dest_it++ = HashRank{*hash_it, rank};
            }
            ++rank;
        }
        return hash_pairs;
    }
};

namespace _internal {

inline std::vector<int> compute_interval_sizes(
    std::vector<hash_t> const& hashes, HashRange const hash_range, size_t const num_intervals
) {
    std::vector<int> intervals;
    intervals.reserve(num_intervals);

    auto bucket_size = hash_range.bucket_size(num_intervals);

    auto current_pos = hashes.begin();
    for (size_t i = 0; i + 1 < num_intervals; ++i) {
        hash_t upper_limit = hash_range.lower + (i + 1) * bucket_size - 1;
        auto pos = std::upper_bound(current_pos, hashes.end(), upper_limit);
        intervals.push_back(pos - current_pos);
        current_pos = pos;
    }
    intervals.push_back(hashes.end() - current_pos);

    return intervals;
}

inline std::vector<HashRank> merge_intervals(
    std::vector<HashRank>&& values,
    std::vector<int> const& local_offsets,
    std::vector<int> const& interval_sizes
) {
    using Iterator = std::vector<HashRank>::iterator;
    using IteratorPair = std::pair<Iterator, Iterator>;

    assert_equal(local_offsets.size(), interval_sizes.size());
    std::vector<IteratorPair> iter_pairs(local_offsets.size());

    auto op = [begin = values.begin()](auto const& offset, auto const& size) {
        return IteratorPair{begin + offset, begin + offset + size};
    };
    std::transform(
        local_offsets.begin(),
        local_offsets.end(),
        interval_sizes.begin(),
        iter_pairs.begin(),
        op
    );

    std::vector<HashRank> merged_values(values.size());
    tlx::multiway_merge(iter_pairs.begin(), iter_pairs.end(), merged_values.begin(), values.size());
    return merged_values;
}


template <typename T>
inline std::vector<hash_t> extract_hash_values(std::vector<T> const& values) {
    std::vector<hash_t> hash_values(values.size());

    auto get_hash = [](auto const& x) { return x.hash_value; };
    std::transform(values.begin(), values.end(), hash_values.begin(), get_hash);

    return hash_values;
}

inline RecvData send_hash_values(
    std::vector<hash_t> const& hashes, HashRange const hash_range, Communicator const& comm
) {
    namespace kmp = kamping;

    auto interval_sizes = _internal::compute_interval_sizes(hashes, hash_range, comm.size());

    std::vector<int> offsets{interval_sizes};
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), 0);
    assert_equal(offsets.back() + interval_sizes.back(), std::ssize(hashes));

    auto offset_result = comm.alltoall(kmp::send_buf(offsets));
    auto result = comm.alltoallv(
        kmp::send_buf(hashes),
        kmp::send_counts(interval_sizes),
        kmp::send_displs(offsets)
    );

    return {
        result.extract_recv_buffer(),
        result.extract_recv_counts(),
        result.extract_recv_displs(),
        offset_result.extract_recv_buffer()};
}

inline std::pair<std::vector<int>, std::vector<int>> compute_duplicates(
    std::vector<HashRank> const& hash_rank_pairs,
    std::vector<int> const& interval_sizes,
    std::vector<int>&& global_offsets
) {
    // todo rewrite with unique
    std::vector<std::vector<int>> result_sets(interval_sizes.size());
    auto counters = std::move(global_offsets);

    if (!hash_rank_pairs.empty()) {
        // todo improve this loop
        bool duplicate = false;
        auto prev = hash_rank_pairs.begin();
        for (auto curr = prev + 1; curr != hash_rank_pairs.end(); ++prev, ++curr) {
            auto idx = counters[prev->rank]++;
            if (prev->hash_value == curr->hash_value) {
                result_sets[prev->rank].push_back(idx);
                duplicate = true;
            } else if (duplicate) {
                result_sets[prev->rank].push_back(idx);
                duplicate = false;
            }
        }
        if (duplicate) {
            result_sets[prev->rank].push_back(counters[prev->rank]++);
        }
    }

    std::vector<int> send_counts(result_sets.size());
    auto get_size = [](auto const& set) { return set.size(); };
    std::transform(result_sets.begin(), result_sets.end(), send_counts.begin(), get_size);

    auto num_duplicates = std::accumulate(send_counts.begin(), send_counts.end(), 0);

    std::vector<int> send_buf(num_duplicates);
    for (auto dest_it = send_buf.begin(); auto const& set: result_sets) {
        dest_it = std::copy(set.begin(), set.end(), dest_it);
    }

    return {std::move(send_buf), std::move(send_counts)};
}

inline std::optional<std::vector<int>> send_duplicates(
    std::vector<int> const& duplicates,
    std::vector<int> const& interval_sizes,
    Communicator const& comm_send,
    Communicator const& comm_global
) {
    namespace kmp = kamping;

    bool any_dups = !duplicates.empty();
    auto result = comm_global.allreduce(kmp::send_buf({any_dups}), kmp::op(std::logical_or<>{}));
    auto any_global_dups = result.extract_recv_buffer()[0];

    if (any_global_dups) {
        return comm_send.alltoallv(kmp::send_buf(duplicates), kmp::send_counts(interval_sizes))
            .extract_recv_buffer();
    } else {
        return {};
    }
}

} // namespace _internal


template <typename HashPolicy, typename Derived>
class BloomFilter {
    template <typename StringSet>
    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;
    using GridCommunicator = multi_level::GridCommunicators<Communicator>;

public:
    static constexpr size_t filter_size = std::numeric_limits<uint64_t>::max();

    BloomFilter(Communicator const& comm_global, GridCommunicator const& comm_grid)
        : comm_global_(comm_global),
          comm_grid_(comm_grid) {}

    template <typename StringSet, typename... Candidates>
    std::vector<size_t> filter(
        StringLcpPtr<StringSet> const strptr,
        size_t const depth,
        std::vector<size_t>& results,
        Candidates const&... candidates
    ) {
        auto& measuringTool = measurement::MeasuringTool::measuringTool();
        auto const& ss = strptr.active();

        measuringTool.start("bloomfilter_find_local_duplicates");
        auto hash_pairs = generate_hash_pairs(ss, candidates..., depth, strptr.lcp());
        auto& hash_idx_pairs = hash_pairs.hash_idx_pairs;

        ips4o::sort(hash_idx_pairs.begin(), hash_idx_pairs.end());
        auto local_hash_dups = get_local_duplicates(hash_idx_pairs);
        std::erase_if(hash_idx_pairs, std::not_fn(should_send));
        measuringTool.stop("bloomfilter_find_local_duplicates");

        measuringTool.start("bloomfilter_find_remote_duplicates");
        auto derived = static_cast<Derived*>(this);
        auto remote_dups = derived->find_remote_duplicates(
            comm_grid_.comms.begin(),
            comm_grid_.comms.end(),
            hash_idx_pairs,
            HashRange{0, filter_size}
        );
        measuringTool.stop("bloomfilter_find_remote_duplicates");

        measuringTool.start("bloomfilter_merge_duplicates");
        auto& local_lcp_dups = hash_pairs.lcp_duplicates;
        auto remote_dups_ =
            prune_remote_duplicates(remote_dups.value_or(std::vector<int>{}), hash_idx_pairs);
        auto final_duplicates = merge_duplicates(local_hash_dups, local_lcp_dups, remote_dups_);
        measuringTool.stop("bloomfilter_merge_duplicates");

        measuringTool.start("bloomfilter_write_depth");
        set_depth(ss, depth, candidates..., hash_pairs.eos_candidates, results);
        measuringTool.stop("bloomfilter_write_depth");

        return final_duplicates;
    }

protected:
    Communicator const& comm_global_;
    GridCommunicator const& comm_grid_;

private:
    struct GeneratedHashPairs {
        std::vector<HashStringIndex> hash_idx_pairs;
        std::vector<size_t> lcp_duplicates;
        std::vector<size_t> eos_candidates;
    };

    template <typename StringSet, typename LcpIter>
    static GeneratedHashPairs generate_hash_pairs(
        StringSet const& ss,
        std::vector<size_t> const& candidates,
        size_t const depth,
        LcpIter const lcps
    ) {
        using std::begin;

        if (candidates.empty()) {
            return {};
        }

        std::vector<HashStringIndex> hash_idx_pairs;
        std::vector<size_t> eos_candidates;
        std::vector<size_t> lcp_dups;
        eos_candidates.reserve(candidates.size());
        hash_idx_pairs.reserve(candidates.size());
        lcp_dups.reserve(candidates.size());

        for (auto prev = candidates.front(); auto const& curr: candidates) {
            auto const& curr_str = ss[begin(ss) + curr];

            if (depth > ss.get_length(curr_str)) {
                eos_candidates.push_back(curr);
            } else if (prev + 1 == curr && lcps[curr] >= depth) {
                lcp_dups.push_back(curr);
                if (hash_idx_pairs.back().string_index + 1 == curr) {
                    hash_idx_pairs.back().is_lcp_root = true;
                }
            } else {
                hash_t hash = HashPolicy::hash(ss.get_chars(curr_str, 0), depth, filter_size);
                hash_idx_pairs.emplace_back(hash, curr);
            }
            prev = curr;
        }
        return {std::move(hash_idx_pairs), std::move(lcp_dups), std::move(eos_candidates)};
    }

    template <typename StringSet, typename LcpIter>
    static GeneratedHashPairs
    generate_hash_pairs(StringSet const& ss, size_t const depth, LcpIter const lcps) {
        using std::begin;

        if (ss.empty()) {
            return {};
        }

        std::vector<HashStringIndex> hash_idx_pairs;
        std::vector<size_t> lcp_dups;
        std::vector<size_t> eos_candidates;
        eos_candidates.reserve(ss.size());
        hash_idx_pairs.reserve(ss.size());
        lcp_dups.reserve(ss.size());

        for (size_t candidate = 0; candidate != ss.size(); ++candidate) {
            auto const& str = ss[begin(ss) + candidate];

            if (depth > ss.get_length(str)) {
                eos_candidates.push_back(candidate);
            } else if (lcps[candidate] >= depth) {
                lcp_dups.push_back(candidate);
                if (hash_idx_pairs.back().string_index + 1 == candidate) {
                    hash_idx_pairs.back().is_lcp_root = true;
                }
            } else {
                auto hash = HashPolicy::hash(ss.get_chars(str, 0), depth, filter_size);
                hash_idx_pairs.emplace_back(hash, candidate);
            }
        }
        return {std::move(hash_idx_pairs), std::move(lcp_dups), {std::move(eos_candidates)}};
    }

    static std::vector<size_t> get_local_duplicates(std::vector<HashStringIndex>& local_values) {
        if (local_values.empty()) {
            return {};
        }

        std::vector<size_t> local_duplicates;
        for (auto it = local_values.begin(); it != local_values.end() - 1;) {
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
        if (local_values.back().is_lcp_root) {
            auto& pivot = local_values.back();
            pivot.is_local_dup = true;
            pivot.send_anyway = true;
            local_duplicates.push_back(pivot.string_index);
        }
        return local_duplicates;
    }

    static std::vector<size_t> prune_remote_duplicates(
        std::vector<int> const& duplicates, std::vector<HashStringIndex> const& hash_idx_pairs
    ) {
        std::vector<size_t> pruned_duplicates;
        pruned_duplicates.reserve(duplicates.size());

        for (auto const& duplicate: duplicates) {
            auto const& orig_pair = hash_idx_pairs[duplicate];
            if (!orig_pair.send_anyway) {
                pruned_duplicates.push_back(orig_pair.string_index);
            }
        }
        return pruned_duplicates;
    }

    static std::vector<size_t> merge_duplicates(
        std::vector<size_t>& local_hash_dups,
        std::vector<size_t>& local_lcp_dups,
        std::vector<size_t>& remote_dups
    ) {
        // todo could this use multiway merge instead?
        ips4o::sort(remote_dups.begin(), remote_dups.end());
        ips4o::sort(local_hash_dups.begin(), local_hash_dups.end());

        using Iterator = std::vector<size_t>::iterator;
        using IteratorPair = std::pair<Iterator, Iterator>;
        std::array<IteratorPair, 3> iter_pairs{
            {{local_hash_dups.begin(), local_hash_dups.end()},
             {local_lcp_dups.begin(), local_lcp_dups.end()},
             {remote_dups.begin(), remote_dups.end()}}};
        size_t total_elems = local_hash_dups.size() + local_lcp_dups.size() + remote_dups.size();

        std::vector<size_t> merged_elems(total_elems);
        tlx::multiway_merge(
            iter_pairs.begin(),
            iter_pairs.end(),
            merged_elems.begin(),
            total_elems
        );
        return merged_elems;
    }

    template <typename StringSet>
    static void set_depth(
        StringSet const& ss,
        size_t const depth,
        std::vector<size_t> const& eos_candidates,
        std::vector<size_t>& results
    ) {
        using std::begin;

        std::fill(results.begin(), results.end(), depth);

        for (auto const& candidate: eos_candidates) {
            results[candidate] = ss.get_length(ss[begin(ss) + candidate]);
        }
    }

    template <typename StringSet>
    static void set_depth(
        StringSet const& ss,
        size_t const depth,
        std::vector<size_t> const& candidates,
        std::vector<size_t> const& eos_candidates,
        std::vector<size_t>& results
    ) {
        using std::begin;

        for (auto const& candidate: candidates) {
            results[candidate] = depth;
        }

        for (auto const& candidate: eos_candidates) {
            results[candidate] = ss.get_length(ss[begin(ss) + candidate]);
        }
    }

    static bool should_send(HashStringIndex const& v) { return !v.is_local_dup || v.send_anyway; }
};

template <typename HashPolicy>
class SingleLevel : public BloomFilter<HashPolicy, SingleLevel<HashPolicy>> {
    friend BloomFilter<HashPolicy, SingleLevel<HashPolicy>>;

public:
    using BloomFilter<HashPolicy, SingleLevel<HashPolicy>>::BloomFilter;

private:
    template <typename CommIt>
    std::optional<std::vector<int>> find_remote_duplicates(
        CommIt const comm_first,
        CommIt const comm_last,
        std::vector<HashStringIndex> const& hash_str_pairs,
        HashRange const hash_range
    ) {
        assert(comm_first != comm_last);
        auto const& comm = *(comm_last - 1);

        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("bloomfilter_send_hashes");
        auto hash_values = _internal::extract_hash_values(hash_str_pairs);
        auto recv_data = _internal::send_hash_values(hash_values, hash_range, comm);
        auto hash_rank_pairs = _internal::merge_intervals(
            recv_data.compute_hash_rank_pairs(),
            recv_data.local_offsets,
            recv_data.interval_sizes
        );
        measuring_tool.add(hash_rank_pairs.size(), "bloomfilter_recv_hash_values");
        measuring_tool.stop("bloomfilter_send_hashes");

        measuring_tool.start("bloomfilter_compute_remote_duplicates");
        auto [duplicates, interval_sizes] = _internal::compute_duplicates(
            hash_rank_pairs,
            recv_data.interval_sizes,
            std::move(recv_data.global_offsets)
        );
        measuring_tool.add(duplicates.size(), "bloomfilter_remote_duplicates");
        measuring_tool.stop("bloomfilter_compute_remote_duplicates");

        measuring_tool.start("bloomfilter_send_indices");
        auto remote_dups =
            _internal::send_duplicates(duplicates, interval_sizes, comm, this->comm_global_);
        measuring_tool.stop("bloomfilter_send_indices");

        return remote_dups;
    }
};

template <typename HashPolicy>
class MultiLevel : public BloomFilter<HashPolicy, MultiLevel<HashPolicy>> {
    friend BloomFilter<HashPolicy, MultiLevel<HashPolicy>>;

public:
    using BloomFilter<HashPolicy, MultiLevel<HashPolicy>>::BloomFilter;

private:
    template <typename... Args>
    std::optional<std::vector<int>> find_remote_duplicates(Args&&... args) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        measuring_tool.start("bloomfilter_send_hashes");
        auto duplicates = find_remote_duplicates_(std::forward<Args>(args)...);
        measuring_tool.stop("bloomfilter_send_indices");

        return duplicates;
    }

    template <typename CommIt, typename T>
    std::optional<std::vector<int>> find_remote_duplicates_(
        CommIt const comm_first,
        CommIt const comm_last,
        std::vector<T> const& hash_pairs,
        HashRange const hash_range
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        assert(comm_first != comm_last);
        auto const& comm = *comm_first;

        // todo add correct timing
        auto hash_values = _internal::extract_hash_values(hash_pairs);
        auto recv_data = _internal::send_hash_values(hash_values, hash_range, comm);
        auto hash_rank_pairs = _internal::merge_intervals(
            recv_data.compute_hash_rank_pairs(),
            recv_data.local_offsets,
            recv_data.interval_sizes
        );

        if (comm_first + 1 == comm_last) {
            measuring_tool.add(hash_rank_pairs.size(), "bloomfilter_recv_hash_values");
            measuring_tool.stop("bloomfilter_send_hashes");

            measuring_tool.start("bloomfilter_compute_remote_duplicates");
            auto [duplicates, interval_sizes] = _internal::compute_duplicates(
                hash_rank_pairs,
                recv_data.interval_sizes,
                std::move(recv_data.global_offsets)
            );
            measuring_tool.add(duplicates.size(), "bloomfilter_remote_duplicates");
            measuring_tool.stop("bloomfilter_compute_remote_duplicates");

            measuring_tool.start("bloomfilter_send_indices");
            return _internal::send_duplicates(duplicates, interval_sizes, comm, this->comm_global_);

        } else {
            auto bucket = hash_range.bucket(comm.rank(), comm.size());
            auto duplicates =
                find_remote_duplicates_(comm_first + 1, comm_last, hash_rank_pairs, bucket);
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
        namespace kmp = kamping;

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
            // todo this loop should probably use and iterator
            for (; i < duplicate; ++i) {
                counters[hash_rank_pairs[i].rank]++;
            }
            auto rank = hash_rank_pairs[i].rank;
            remote_idxs[offsets[rank]++] = counters[rank];
        }

        auto result = comm.alltoallv(
            kmp::send_buf(remote_idxs),
            kmp::send_counts(send_counts),
            kmp::send_displs(send_displs)
        );
        return result.extract_recv_buffer();
    }
};

} // namespace bloomfilter
} // namespace dss_mehnert
