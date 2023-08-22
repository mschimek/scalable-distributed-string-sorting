// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include <bits/iterator_concepts.h>
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

// todo use uint64_t for hash_values
struct HashStringIndex {
    size_t hashValue;
    size_t stringIndex;
    bool isLocalDuplicate = false;
    bool isLocalDuplicateButSendAnyway = false;
    bool isLcpLocalRoot = false;

    bool operator<(HashStringIndex const& rhs) const { return hashValue < rhs.hashValue; }

    friend std::ostream& operator<<(std::ostream& stream, HashStringIndex const& hashStringIndex) {
        return stream << "[" << hashStringIndex.hashValue << ", " << hashStringIndex.stringIndex
                      << ", localDup: " << hashStringIndex.isLocalDuplicate
                      << ", sendAnyway: " << hashStringIndex.isLocalDuplicateButSendAnyway << "]";
    }
};

struct HashPEIndex {
    // todo update member naming scheme
    size_t hashValue;
    size_t PEIndex;

    bool operator<(HashPEIndex const& rhs) const { return hashValue < rhs.hashValue; }

    friend std::ostream& operator<<(std::ostream& stream, HashPEIndex const& hashPEIndex) {
        return stream << "[" << hashPEIndex.hashValue << ", " << hashPEIndex.PEIndex << "]";
    }
};

struct SipHasher {
    static inline size_t hash(unsigned char const* str, size_t length, size_t filter_size) {
        return tlx::siphash(str, length) % filter_size;
    }
};

struct XXHasher {
    static inline size_t hash(unsigned char const* str, size_t length, size_t filter_size) {
        xxh::hash_state_t<64> hash_stream;
        hash_stream.update(str, length);
        xxh::hash_t<64> hashV = hash_stream.digest();
        return hashV % filter_size;
    }

    static inline size_t
    hash(unsigned char const* str, size_t length, size_t filter_size, size_t old_hash) {
        xxh::hash_state_t<64> hash_stream;
        hash_stream.update(str, length);
        xxh::hash_t<64> new_hash = hash_stream.digest();
        return (old_hash ^ new_hash) % filter_size;
    }
};

struct HashRange {
    size_t lower;
    size_t upper;

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

namespace _internal {

inline std::vector<int> compute_interval_sizes(
    std::vector<size_t> const& hashes, HashRange hash_range, size_t num_intervals
) {
    // todo get rid of push_back
    std::vector<int> intervals;
    intervals.reserve(num_intervals);

    auto current_pos = hashes.begin();

    auto bucket_size = hash_range.bucket_size(num_intervals);
    for (size_t i = 0; i + 1 < num_intervals; ++i) {
        size_t upper_limit = hash_range.lower + (i + 1) * bucket_size - 1;
        auto pos = std::upper_bound(current_pos, hashes.end(), upper_limit);
        intervals.push_back(pos - current_pos);
        current_pos = pos;
    }
    intervals.push_back(hashes.end() - current_pos);

    return intervals;
}

template <typename T>
inline std::vector<T>
merge_intervals(std::vector<T>&& values, std::vector<int> const& interval_sizes) {
    using Iterator = std::vector<T>::iterator;
    using IteratorPair = std::pair<Iterator, Iterator>;

    std::vector<IteratorPair> iter_pairs;
    iter_pairs.reserve(interval_sizes.size());
    for (auto it = values.begin(); auto const& interval: interval_sizes) {
        iter_pairs.emplace_back(it, it + interval);
        it += interval;
    }

    std::vector<T> merged_values(values.size());
    tlx::multiway_merge(iter_pairs.begin(), iter_pairs.end(), merged_values.begin(), values.size());
    return merged_values;
}

template <typename T>
inline std::vector<size_t> extract_hash_values(std::vector<T> const& values) {
    std::vector<size_t> hash_values(values.size());

    auto get_hash = [](auto const& x) { return x.hashValue; };
    std::transform(values.begin(), values.end(), hash_values.begin(), get_hash);

    return hash_values;
}

} // namespace _internal


struct RecvData {
    std::vector<size_t> hashes;
    std::vector<int> interval_sizes;
    std::vector<int> global_offsets;

    std::vector<HashPEIndex> compute_hash_rank_pairs() const {
        std::vector<HashPEIndex> hash_pairs(hashes.size());

        auto hash_it = hashes.begin();
        auto dest = hash_pairs.begin();
        for (size_t rank = 0; auto const& interval: interval_sizes) {
            auto hash_pair = [rank_ = rank++](auto const& hash) {
                return HashPEIndex{hash, rank_};
            };
            dest = std::transform(hash_it, hash_it + interval, dest, hash_pair);
            hash_it += interval;
        }
        return hash_pairs;
    }
};

class SendOnlyHashesToFilter {
public:
    static RecvData
    send_to_filter(std::vector<size_t>& hashes, HashRange hash_range, Communicator const& comm) {
        namespace kmp = kamping;
        // auto& measuringTool = measurement::MeasuringTool::measuringTool();

        // measuringTool.start("bloomfilter_send_to_filterSetup");
        auto interval_sizes = _internal::compute_interval_sizes(hashes, hash_range, comm.size());

        std::vector<int> offsets{interval_sizes};
        std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), 0);
        assert_equal(offsets.back() + interval_sizes.back(), std::ssize(hashes));
        // measuringTool.stop("bloomfilter_send_to_filterSetup");

        // measuringTool.start("bloomfilter_sendEncodedValuesOverall");
        auto offset_result = comm.alltoall(kmp::send_buf(offsets));
        auto result = comm.alltoallv(
            kmp::send_buf(hashes),
            kmp::send_counts(interval_sizes),
            kmp::send_displs(offsets)
        );
        // measuringTool.stop("bloomfilter_sendEncodedValuesOverall");

        return {
            result.extract_recv_buffer(),
            result.extract_recv_counts(),
            offset_result.extract_recv_buffer()};
    }
};

// todo return an optional here
struct FindDuplicates {
    static inline std::vector<size_t> find_duplicates(
        std::vector<HashPEIndex> const& hash_rank_pairs, RecvData&& data, Communicator const& comm
    ) {
        namespace kmp = kamping;

        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        measuring_tool.add(hash_rank_pairs.size(), "bloomfilter_recvHashValues");

        measuring_tool.start("bloomfilter_findDuplicatesOverallIntern");

        measuring_tool.start("bloomfilter_findDuplicatesFind");
        auto [duplicates_idxs, interval_sizes] = compute_duplicate_indices(
            hash_rank_pairs,
            data.interval_sizes,
            std::move(data.global_offsets) // offsets are invalidated here
        );
        measuring_tool.stop("bloomfilter_findDuplicatesFind");

        measuring_tool.start("bloomfilter_findDuplicatesSendDups");
        bool any_dups = !duplicates_idxs.empty();
        auto result = comm.allreduce(kmp::send_buf({any_dups}), kmp::op(std::logical_or<>{}));
        auto any_global_dups = result.extract_recv_buffer()[0];

        std::vector<size_t> duplicates;
        if (any_global_dups) {
            duplicates =
                comm.alltoallv(kmp::send_buf(duplicates_idxs), kmp::send_counts(interval_sizes))
                    .extract_recv_buffer();
        }

        measuring_tool.stop("bloomfilter_findDuplicatesSendDups");
        measuring_tool.stop("bloomfilter_findDuplicatesOverallIntern");

        return duplicates;
    }

private:
    // todo use int for indices
    static std::pair<std::vector<size_t>, std::vector<int>> compute_duplicate_indices(
        std::vector<HashPEIndex> const& hash_rank_pairs,
        std::vector<int> const& interval_sizes,
        std::vector<int>&& global_offsets
    ) {
        std::vector<std::vector<size_t>> result_sets(interval_sizes.size());
        auto counters = std::move(global_offsets);

        if (!hash_rank_pairs.empty()) {
            // todo improve this loop
            // todo maybe only send first n - 1 duplicates
            bool duplicate = false;
            auto prev = hash_rank_pairs.begin();
            for (auto curr = prev + 1; curr != hash_rank_pairs.end(); ++prev, ++curr) {
                auto idx = counters[prev->PEIndex]++;
                if (prev->hashValue == curr->hashValue) {
                    result_sets[prev->PEIndex].push_back(idx);
                    duplicate = true;
                } else if (duplicate) {
                    result_sets[prev->PEIndex].push_back(idx);
                    duplicate = false;
                }
            }
            if (duplicate) {
                result_sets[prev->PEIndex].push_back(counters[prev->PEIndex]++);
            }
        }

        std::vector<int> send_counts(result_sets.size());
        auto get_size = std::size<std::vector<size_t>>;
        std::transform(result_sets.begin(), result_sets.end(), send_counts.begin(), get_size);

        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        auto num_duplicates = std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
        measuring_tool.add(num_duplicates, "bloomfilter_findDuplicatesSendDups");

        std::vector<size_t> send_buf(num_duplicates);
        for (auto dest_it = send_buf.begin(); auto const& set: result_sets) {
            dest_it = std::copy(set.begin(), set.end(), dest_it);
        }

        return {std::move(send_buf), std::move(send_counts)};
    }
};

template <typename HashPolicy, typename Derived>
class BloomFilter {
    template <typename StringSet>
    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

public:
    static constexpr size_t filter_size = std::numeric_limits<uint64_t>::max();

    BloomFilter(size_t size, size_t startDepth) : hash_values_(size, 0) {}

    template <typename StringSet>
    std::vector<size_t> filter(
        StringLcpPtr<StringSet> strptr,
        size_t depth,
        std::vector<size_t>& results,
        multi_level::GridCommunicators<Communicator> const& grid
    ) {
        auto& measuringTool = measurement::MeasuringTool::measuringTool();
        auto const& ss = strptr.active();

        measuringTool.start("bloomfilter_prepare");
        auto hash_pairs = generate_hash_pairs(ss, depth, strptr.lcp());
        auto& hash_idx_pairs = hash_pairs.hash_idx_pairs;

        ips4o::sort(hash_idx_pairs.begin(), hash_idx_pairs.end());
        auto local_hash_dups = get_local_duplicates(hash_idx_pairs);
        std::erase_if(hash_idx_pairs, std::not_fn(should_send));
        measuringTool.stop("bloomfilter_prepare");

        auto remote_dups = Derived::find_remote_duplicates(
            grid.comms.begin(),
            grid.comms.end(),
            hash_idx_pairs,
            {0, filter_size}
        );

        measuringTool.start("bloomfilter_getIndices");
        auto duplicate_idxs = merge_duplicate_indices(
            strptr.active().size(),
            local_hash_dups,
            hash_pairs.lcp_duplicates,
            remote_dups,
            hash_idx_pairs
        );
        measuringTool.stop("bloomfilter_getIndices");

        measuringTool.start("bloomfilter_setDepth");
        set_depth(strptr.active(), depth, hash_pairs.eos_candidates, results);
        measuringTool.stop("bloomfilter_setDepth");

        return duplicate_idxs;
    }

    template <typename StringSet>
    std::vector<size_t> filter(
        StringLcpPtr<StringSet> strptr,
        size_t const depth,
        std::vector<size_t> const& candidates,
        std::vector<size_t>& results,
        multi_level::GridCommunicators<Communicator> const& grid
    ) {
        auto& measuringTool = measurement::MeasuringTool::measuringTool();
        auto const& ss = strptr.active();

        measuringTool.start("bloomfilter_prepare");
        auto hash_pairs = generate_hash_pairs(ss, candidates, depth, strptr.lcp());
        auto& hash_idx_pairs = hash_pairs.hash_idx_pairs;

        ips4o::sort(hash_idx_pairs.begin(), hash_idx_pairs.end());
        auto local_hash_dups = get_local_duplicates(hash_idx_pairs);
        std::erase_if(hash_idx_pairs, std::not_fn(should_send));
        measuringTool.stop("bloomfilter_prepare");

        auto remote_dups = Derived::find_remote_duplicates(
            grid.comms.begin(),
            grid.comms.end(),
            hash_idx_pairs,
            {0, filter_size}
        );

        measuringTool.start("bloomfilter_getIndices");
        auto duplicate_idxs = merge_duplicate_indices(
            ss.size(),
            local_hash_dups,
            hash_pairs.lcp_duplicates,
            remote_dups,
            hash_idx_pairs
        );
        measuringTool.stop("bloomfilter_getIndices");

        measuringTool.start("bloomfilter_setDepth");
        set_depth(ss, depth, candidates, hash_pairs.eos_candidates, results);
        measuringTool.stop("bloomfilter_setDepth");

        return duplicate_idxs;
    }

private:
    std::vector<size_t> hash_values_;

    struct GeneratedHashPairs {
        std::vector<HashStringIndex> hash_idx_pairs;
        std::vector<size_t> lcp_duplicates;
        std::vector<size_t> eos_candidates;
    };

    template <typename StringSet, typename LcpIter>
    GeneratedHashPairs generate_hash_pairs(
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
                if (hash_idx_pairs.back().stringIndex + 1 == curr) {
                    hash_idx_pairs.back().isLcpLocalRoot = true;
                }
            } else {
                // todo could reuse the old hash value
                size_t hash = HashPolicy::hash(ss.get_chars(curr_str, 0), depth, filter_size);
                hash_idx_pairs.emplace_back(hash, curr);
                hash_values_[curr] = hash;
            }
            prev = curr;
        }
        return {std::move(hash_idx_pairs), std::move(lcp_dups), std::move(eos_candidates)};
    }

    template <typename StringSet, typename LcpIter>
    GeneratedHashPairs
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
                if (hash_idx_pairs.back().stringIndex + 1 == candidate) {
                    hash_idx_pairs.back().isLcpLocalRoot = true;
                }
            } else {
                auto hash = HashPolicy::hash(ss.get_chars(str, 0), depth, filter_size);
                hash_idx_pairs.emplace_back(hash, candidate);
                hash_values_[candidate] = hash;
            }
        }
        return {std::move(hash_idx_pairs), std::move(lcp_dups), {std::move(eos_candidates)}};
    }

    std::vector<size_t> get_local_duplicates(std::vector<HashStringIndex>& local_values) {
        if (local_values.empty()) {
            return {};
        }

        std::vector<size_t> local_duplicates;
        for (auto it = local_values.begin(); it != local_values.end() - 1;) {
            auto& pivot = *it++;
            if (it->hashValue == pivot.hashValue) {
                pivot.isLocalDuplicate = true;
                pivot.isLocalDuplicateButSendAnyway = true;
                it->isLocalDuplicate = true;
                local_duplicates.push_back(pivot.stringIndex);
                local_duplicates.push_back(it->stringIndex);

                while (++it != local_values.end() && it->hashValue == pivot.hashValue) {
                    it->isLocalDuplicate = true;
                    local_duplicates.push_back(it->stringIndex);
                }
            } else if (pivot.isLcpLocalRoot) {
                pivot.isLocalDuplicate = true;
                pivot.isLocalDuplicateButSendAnyway = true;
                local_duplicates.push_back(pivot.stringIndex);
            }
        }
        if (local_values.back().isLcpLocalRoot) {
            auto& pivot = local_values.back();
            pivot.isLocalDuplicate = true;
            pivot.isLocalDuplicateButSendAnyway = true;
            local_duplicates.push_back(pivot.stringIndex);
        }
        return local_duplicates;
    }

    template <typename StringSet>
    void set_depth(
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
    void set_depth(
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

    static std::vector<size_t> merge_duplicate_indices(
        size_t const size,
        std::vector<size_t>& local_hash_dups,
        std::vector<size_t>& local_lcp_dups,
        std::vector<size_t>& remote_dups,
        std::vector<HashStringIndex> const& orig_string_idxs
    ) {
        // todo consider adding all and removing duplicate indices later
        std::vector<size_t> sorted_remote_dups;
        sorted_remote_dups.reserve(remote_dups.size());
        for (auto const& idx: remote_dups) {
            auto const& original = orig_string_idxs[idx];
            if (!original.isLocalDuplicateButSendAnyway) {
                sorted_remote_dups.push_back(original.stringIndex);
            }
        }

        // todo could this use multiway merge instead?
        ips4o::sort(sorted_remote_dups.begin(), sorted_remote_dups.end());
        ips4o::sort(local_hash_dups.begin(), local_hash_dups.end());

        using Iterator = std::vector<size_t>::iterator;
        std::array<std::pair<Iterator, Iterator>, 3> iter_pairs{
            {{local_hash_dups.begin(), local_hash_dups.end()},
             {local_lcp_dups.begin(), local_lcp_dups.end()},
             {sorted_remote_dups.begin(), sorted_remote_dups.end()}}};
        size_t total_elems =
            local_hash_dups.size() + local_lcp_dups.size() + sorted_remote_dups.size();

        std::vector<size_t> merged_elems(total_elems);
        tlx::multiway_merge(
            iter_pairs.begin(),
            iter_pairs.end(),
            merged_elems.begin(),
            total_elems
        );
        return merged_elems;
    }

    static bool should_send(HashStringIndex const& v) {
        return !v.isLocalDuplicate || v.isLocalDuplicateButSendAnyway;
    }
};

template <typename HashPolicy>
class SingleLevel : public BloomFilter<HashPolicy, SingleLevel<HashPolicy>> {
    friend BloomFilter<HashPolicy, SingleLevel<HashPolicy>>;

public:
    using BloomFilter<HashPolicy, SingleLevel<HashPolicy>>::BloomFilter;

private:
    template <typename CommIt>
    static std::vector<size_t> find_remote_duplicates(
        CommIt const comm_first,
        CommIt const comm_last,
        std::vector<HashStringIndex> const& hash_str_pairs,
        HashRange const hash_range
    ) {
        assert(comm_first != comm_last);
        auto const& comm = *(comm_last - 1);

        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        measuring_tool.start("bloomfilter_sendHashStringIndices");
        auto hash_values = _internal::extract_hash_values(hash_str_pairs);
        auto recv_data = SendOnlyHashesToFilter::send_to_filter(hash_values, hash_range, comm);
        auto hash_rank_pairs = _internal::merge_intervals(
            recv_data.compute_hash_rank_pairs(),
            recv_data.interval_sizes
        );
        measuring_tool.stop("bloomfilter_sendHashStringIndices");

        return FindDuplicates::find_duplicates(hash_rank_pairs, std::move(recv_data), comm);
    }
};

template <typename HashPolicy>
class MultiLevel : public BloomFilter<HashPolicy, MultiLevel<HashPolicy>> {
    friend BloomFilter<HashPolicy, MultiLevel<HashPolicy>>;

public:
    using BloomFilter<HashPolicy, MultiLevel<HashPolicy>>::BloomFilter;

private:
    // todo hash_pairs is overallocating
    template <typename CommIt, typename T>
    static std::vector<size_t> find_remote_duplicates(
        CommIt const comm_first,
        CommIt const comm_last,
        std::vector<T> const& hash_pairs,
        HashRange const hash_range
    ) {
        namespace kmp = kamping;

        assert(comm_first != comm_last);
        auto const& comm = *comm_first;

        // todo add correct timing
        auto hash_values = _internal::extract_hash_values(hash_pairs);
        auto recv_data = SendOnlyHashesToFilter::send_to_filter(hash_values, hash_range, comm);
        auto hash_rank_pairs = _internal::merge_intervals(
            recv_data.compute_hash_rank_pairs(),
            recv_data.interval_sizes
        );

        if (comm_first + 1 == comm_last) {
            return FindDuplicates::find_duplicates(hash_rank_pairs, std::move(recv_data), comm);
        } else {
            // todo could discard hash_value during merge?!
            auto sub_range = hash_range.bucket(comm.rank(), comm.size());
            auto duplicates =
                find_remote_duplicates(comm_first + 1, comm_last, hash_rank_pairs, sub_range);

            // todo resolve usage of size_t for send_counts
            std::vector<int> send_counts(recv_data.global_offsets.size());
            for (auto const& duplicate: duplicates) {
                send_counts[hash_rank_pairs[duplicate].PEIndex]++;
            }

            std::vector<int> offsets{send_counts};
            std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), 0);
            std::vector<int> send_displs{offsets};

            std::vector<size_t> remote_idxs(duplicates.size());
            {
                auto counters = std::move(recv_data.global_offsets);
                for (size_t i = 0; auto const& duplicate: duplicates) {
                    // todo use iterator
                    for (; i < duplicate; ++i) {
                        counters[hash_rank_pairs[i].PEIndex]++;
                    }
                    auto rank = hash_rank_pairs[i].PEIndex;
                    remote_idxs[offsets[rank]++] = counters[rank]++;
                }
            }

            auto result = comm.alltoallv(
                kmp::send_buf(remote_idxs),
                kmp::send_counts(send_counts),
                kmp::send_displs(send_displs)
            );
            return result.extract_recv_buffer();
        }
    }
};

} // namespace bloomfilter
} // namespace dss_mehnert
