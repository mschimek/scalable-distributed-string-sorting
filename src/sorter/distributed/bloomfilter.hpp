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
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace bloomfilter {

// todo remove
using namespace dss_schimek;

struct Duplicate {
    size_t index;
    bool hasReachedEOS;
};

// todo use uint64_t for hash_values
struct HashTriple {
    size_t hashValue;
    size_t stringIndex;
    size_t PEIndex;

    bool operator<(HashTriple const& rhs) const { return hashValue < rhs.hashValue; }

    friend std::ostream& operator<<(std::ostream& stream, HashTriple const& hashTriple) {
        return stream << "[" << hashTriple.hashValue << ", " << hashTriple.stringIndex << ", "
                      << hashTriple.PEIndex << "]";
    }
};

struct StringTriple {
    unsigned char const* string;
    size_t stringIndex;
    size_t PEIndex;

    bool operator<(StringTriple const& rhs) const {
        size_t i = 0;
        while (string[i] != 0 && string[i] == rhs.string[i])
            ++i;
        return string[i] < rhs.string[i];
    }

    friend std::ostream& operator<<(std::ostream& stream, StringTriple const& stringTriple) {
        return stream << "[" << stringTriple.string << ", " << stringTriple.stringIndex << ", "
                      << stringTriple.PEIndex << "]" << std::endl;
    }
};

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
    size_t hashValue;
    size_t PEIndex;

    bool operator<(HashPEIndex const& rhs) const { return hashValue < rhs.hashValue; }

    friend std::ostream& operator<<(std::ostream& stream, HashPEIndex const& hashPEIndex) {
        return stream << "[" << hashPEIndex.hashValue << ", " << hashPEIndex.PEIndex << "]";
    }
};

struct AllToAllHashesNaive {
    template <typename DataType>
    static std::vector<DataType> alltoallv(
        std::vector<DataType>& send_data,
        std::vector<size_t> const& interval_sizes,
        Communicator const& comm
    ) {
        namespace kmp = kamping;

        // todo use vector<int> everywhere in bloomfilter
        std::vector<int> send_counts{interval_sizes.begin(), interval_sizes.end()};
        auto result = comm.alltoallv(kmp::send_buf(send_data), kmp::send_counts(send_counts));
        return result.extract_recv_buffer();
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

struct RecvData {
    std::vector<size_t> data;
    std::vector<size_t> interval_sizes;
    std::vector<size_t> global_offsets;
};

template <typename SendPolicy>
class SendOnlyHashesToFilter : private SendPolicy {
public:
    static RecvData sendToFilter(
        std::vector<HashStringIndex> const& hashes, size_t filter_size, Communicator const& comm
    ) {
        namespace kmp = kamping;
        auto& measuringTool = measurement::MeasuringTool::measuringTool();

        measuringTool.start("bloomfilter_sendToFilterSetup");
        std::vector<size_t> send_values(hashes.size());
        auto get_hash = [](auto const& x) { return x.hashValue; };
        std::transform(hashes.begin(), hashes.end(), send_values.begin(), get_hash);

        auto interval_sizes = computeIntervalSizes(send_values, filter_size, comm.size());

        std::vector<size_t> offsets(interval_sizes.size());
        std::exclusive_scan(
            interval_sizes.begin(),
            interval_sizes.end(),
            offsets.begin(),
            size_t{0}
        );
        assert_equal(offsets.back() + interval_sizes.back(), hashes.size());

        auto offset_result = comm.alltoall(kmp::send_buf(offsets));
        auto global_offsets = offset_result.extract_recv_buffer();

        auto interval_resut = comm.alltoall(kmp::send_buf(interval_sizes));
        auto recv_interval_sizes = interval_resut.extract_recv_buffer();
        measuringTool.stop("bloomfilter_sendToFilterSetup");

        measuringTool.start("bloomfilter_sendEncodedValuesOverall");
        auto result = SendPolicy::alltoallv(send_values, interval_sizes, filter_size, comm);
        measuringTool.stop("bloomfilter_sendEncodedValuesOverall");

        return {std::move(result), std::move(recv_interval_sizes), std::move(offsets)};
    }

    static std::vector<HashPEIndex> addPEIndex(RecvData const& recv_data) {
        std::vector<HashPEIndex> hash_pairs(recv_data.data.size());

        auto hashes = recv_data.data.begin();
        auto dest = hash_pairs.begin();
        for (size_t PE_idx = 0; auto const& interval: recv_data.interval_sizes) {
            auto hash_pair = [idx = PE_idx++](auto const& hash) { return HashPEIndex{hash, idx}; };
            dest = std::transform(hashes, hashes + interval, dest, hash_pair);
            hashes += interval;
        }
        return hash_pairs;
    }

private:
    static std::vector<size_t> computeIntervalSizes(
        std::vector<size_t> const& hashes, size_t bloomFilterSize, size_t num_intervals
    ) {
        std::vector<size_t> intervals;
        intervals.reserve(num_intervals);

        auto current_pos = hashes.begin();
        for (size_t i = 0; i + 1 < num_intervals; ++i) {
            size_t upper_limit = (i + 1) * (bloomFilterSize / num_intervals) - 1;
            auto pos = std::upper_bound(current_pos, hashes.end(), upper_limit);
            intervals.push_back(pos - current_pos);
            current_pos = pos;
        }
        intervals.push_back(hashes.end() - current_pos);

        return intervals;
    }
};

template <typename GolombPolicy>
struct FindDuplicates {
    static inline std::vector<size_t> findDuplicates(
        std::vector<HashPEIndex>& hash_triples, RecvData const& data, Communicator const& comm
    ) {
        namespace kmp = kamping;

        using Iterator = std::vector<HashPEIndex>::iterator;
        using IteratorPair = std::pair<Iterator, Iterator>;

        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        auto const& interval_sizes = data.interval_sizes;
        auto const& global_offsets = data.global_offsets;

        measuring_tool.add(hash_triples.size(), "bloomfilter_recvHashValues");
        measuring_tool.start("bloomfilter_findDuplicatesOverallIntern");
        measuring_tool.start("bloomfilter_findDuplicatesSetup");

        auto num_elems = std::accumulate(interval_sizes.begin(), interval_sizes.end(), size_t{0});

        std::vector<IteratorPair> iter_pairs;
        for (auto it = hash_triples.begin(); auto const& interval: interval_sizes) {
            iter_pairs.emplace_back(it, it + interval);
            it += interval;
        }
        measuring_tool.stop("bloomfilter_findDuplicatesSetup");

        measuring_tool.start("bloomfilter_findDuplicatesMerge");
        std::vector<HashPEIndex> merged_triples(num_elems);
        tlx::multiway_merge(
            iter_pairs.begin(),
            iter_pairs.end(),
            merged_triples.begin(),
            num_elems
        );
        measuring_tool.stop("bloomfilter_findDuplicatesMerge");

        measuring_tool.start("bloomfilter_findDuplicatesFind");
        std::vector<std::vector<size_t>> result_sets(interval_sizes.size());
        std::vector<size_t> counters(global_offsets.begin(), global_offsets.end());

        if (!merged_triples.empty()) {
            bool duplicate = false;
            auto prev = merged_triples.begin();
            for (auto curr = prev + 1; curr != merged_triples.end(); ++prev, ++curr) {
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

        std::vector<size_t> send_counts(result_sets.size());
        auto get_size = std::size<std::vector<size_t>>;
        std::transform(result_sets.begin(), result_sets.end(), send_counts.begin(), get_size);

        auto num_duplicates = std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
        measuring_tool.add(num_duplicates * sizeof(size_t), "bloomfilter_findDuplicatesSendDups");

        std::vector<size_t> send_buf(num_duplicates);
        for (auto dest_it = send_buf.begin(); auto const& set: result_sets) {
            dest_it = std::copy(set.begin(), set.end(), dest_it);
        }
        measuring_tool.stop("bloomfilter_findDuplicatesFind");

        measuring_tool.start("bloomfilter_findDuplicatesSendDups");
        bool any_dups = num_duplicates > 0;
        auto result = comm.allreduce(kmp::send_buf({any_dups}), kmp::op(std::logical_or<>{}));
        auto any_global_dups = result.extract_recv_buffer()[0];

        std::vector<size_t> duplicates;
        if (any_global_dups) {
            duplicates = GolombPolicy::alltoallv(send_buf, send_counts, interval_sizes, comm);
        }

        measuring_tool.stop("bloomfilter_findDuplicatesSendDups");
        measuring_tool.stop("bloomfilter_findDuplicatesOverallIntern");

        return duplicates;
    }
};

template <typename SendPolicy, typename FindDuplicatesPolicy>
class SingleLevelPolicy {
public:
    static std::vector<size_t> findRemoteDuplicates(
        std::vector<HashStringIndex> const& hash_idx_pairs,
        size_t filter_size,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("bloomfilter_sendHashStringIndices");
        RecvData recv_data = SendPolicy::sendToFilter(hash_idx_pairs, filter_size, comm);
        measuring_tool.stop("bloomfilter_sendHashStringIndices");

        measuring_tool.start("bloomfilter_addPEIndex");
        auto recvHashPEIndices = SendPolicy::addPEIndex(recv_data);
        measuring_tool.stop("bloomfilter_addPEIndex");

        return FindDuplicatesPolicy::findDuplicates(recvHashPEIndices, recv_data, comm);
    }
};

// template <typename Communiuator>
// struct GridCommunicators {
//     std::vector<Communicator> comms;
// };
//
// template <typename SendPolicy, typename FindDuplicatesPolicy>
// class MultiLevelPolicy {
//     std::vector<size_t> findRemoteDuplicates(
//         std::vector<HashStringIndex> const& hash_idx_pairs,
//         size_t filter_size,
//         GridCommunicators<Communicator> const& grid
//     ) {
//         for (auto const& comm: grid.comms) {
//         }
//     }
// };

template <
    typename StringSet,
    typename FindDuplicatesPolicy,
    typename SendPolicy,
    typename HashPolicy>
class BloomFilter {
    using StringLcpPtr = typename tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    // todo bikeshed name
    using BloomFilterPolicy = SingleLevelPolicy<SendPolicy, FindDuplicatesPolicy>;

    size_t size;
    std::vector<size_t> hashValues;
    size_t startDepth;

public:
    BloomFilter(size_t size, size_t startDepth)
        : size(size),
          hashValues(size, 0),
          startDepth(startDepth) {}

    size_t const filter_size = std::numeric_limits<uint64_t>::max();

    template <typename T>
    struct GeneratedHashStructuresEOSCandidates {
        std::vector<T> data;
        std::vector<size_t> eosCandidates;
    };

    template <typename T>
    struct GeneratedHashesLocalDupsEOSCandidates {
        std::vector<T> data;
        std::vector<size_t> localDups;
        std::vector<size_t> eosCandidates;
    };

    std::vector<size_t> filter(
        StringLcpPtr strptr, size_t depth, std::vector<size_t>& results, Communicator const& comm
    ) {
        auto& measuringTool = measurement::MeasuringTool::measuringTool();

        measuringTool.start("bloomfilter_generateHashStringIndices");
        auto [hash_idx_pairs, local_lcp_dups, eos_candidates] =
            generateHashStringIndices(strptr.active(), depth, strptr.lcp());
        measuringTool.stop("bloomfilter_generateHashStringIndices");

        measuringTool.start("bloomfilter_sortHashStringIndices");
        ips4o::sort(hash_idx_pairs.begin(), hash_idx_pairs.end());
        measuringTool.stop("bloomfilter_sortHashStringIndices");

        measuringTool.start("bloomfilter_indicesOfLocalDuplicates");
        auto local_hash_dups = localDuplicateIndices(hash_idx_pairs);
        measuringTool.stop("bloomfilter_indicesOfLocalDuplicates");

        measuringTool.start("bloomfilter_ReducedHashStringIndices");
        std::erase_if(hash_idx_pairs, std::not_fn(shouldSend));
        measuringTool.stop("bloomfilter_ReducedHashStringIndices");

        auto remote_dups =
            BloomFilterPolicy::findRemoteDuplicates(hash_idx_pairs, filter_size, comm);

        measuringTool.start("bloomfilter_getIndices");
        auto indicesOfAllDuplicates = getSortedIndicesOfDuplicates(
            strptr.active().size(),
            local_hash_dups,
            local_lcp_dups,
            remote_dups,
            hash_idx_pairs
        );
        measuringTool.stop("bloomfilter_getIndices");

        measuringTool.start("bloomfilter_setDepth");
        setDepth(strptr.active(), depth, eos_candidates, results);
        measuringTool.stop("bloomfilter_setDepth");

        return indicesOfAllDuplicates;
    }

    std::vector<size_t> filter(
        StringLcpPtr strptr,
        size_t depth,
        std::vector<size_t> const& candidates,
        std::vector<size_t>& results,
        Communicator const& comm
    ) {
        auto& measuringTool = measurement::MeasuringTool::measuringTool();

        measuringTool.start("bloomfilter_generateHashStringIndices");
        auto [hash_idx_pairs, local_lcp_dups, eos_candidates] =
            generateHashStringIndices(strptr.active(), candidates, depth, strptr.lcp());
        measuringTool.stop("bloomfilter_generateHashStringIndices");

        measuringTool.start("bloomfilter_sortHashStringIndices");
        ips4o::sort(hash_idx_pairs.begin(), hash_idx_pairs.end());
        measuringTool.stop("bloomfilter_sortHashStringIndices");

        measuringTool.start("bloomfilter_indicesOfLocalDuplicates");
        auto local_hash_dups = localDuplicateIndices(hash_idx_pairs);
        measuringTool.stop("bloomfilter_indicesOfLocalDuplicates");

        measuringTool.start("bloomfilter_ReducedHashStringIndices");
        std::erase_if(hash_idx_pairs, std::not_fn(shouldSend));
        measuringTool.stop("bloomfilter_ReducedHashStringIndices");

        auto remote_dups =
            BloomFilterPolicy::findRemoteDuplicates(hash_idx_pairs, filter_size, comm);

        measuringTool.start("bloomfilter_getIndices");
        auto indicesOfAllDuplicates = getSortedIndicesOfDuplicates(
            strptr.active().size(),
            local_hash_dups,
            local_lcp_dups,
            remote_dups,
            hash_idx_pairs
        );
        measuringTool.stop("bloomfilter_getIndices");

        measuringTool.start("bloomfilter_setDepth");
        setDepth(strptr.active(), depth, candidates, eos_candidates, results);
        measuringTool.stop("bloomfilter_setDepth");

        return indicesOfAllDuplicates;
    }

private:
    std::vector<size_t> localDuplicateIndices(std::vector<HashStringIndex>& local_values) {
        std::vector<size_t> local_duplicates;
        if (local_values.empty()) {
            return local_duplicates;
        }

        for (auto it = local_values.begin(); it < local_values.end() - 1;) {
            auto& pivot = *it++;
            if (it->hashValue == pivot.hashValue) {
                pivot.isLocalDuplicate = true;
                pivot.isLocalDuplicateButSendAnyway = true;
                it->isLocalDuplicate = true;
                local_duplicates.push_back(pivot.stringIndex);
                local_duplicates.push_back(it->stringIndex);

                while (++it < local_values.end() && it->hashValue == pivot.hashValue) {
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

    void setDepth(
        StringSet const& ss,
        size_t depth,
        std::vector<size_t> const& eosCandidates,
        std::vector<size_t>& results
    ) {
        using std::begin;

        std::fill(results.begin(), results.end(), depth);

        for (auto const& candidate: eosCandidates) {
            results[candidate] = ss.get_length(ss[begin(ss) + candidate]);
        }
    }

    void setDepth(
        StringSet const& ss,
        size_t depth,
        std::vector<size_t> const& candidates,
        std::vector<size_t> const& eosCandidates,
        std::vector<size_t>& results
    ) {
        using std::begin;

        for (auto const& candidate: candidates) {
            results[candidate] = depth;
        }

        for (auto const& candidate: eosCandidates) {
            results[candidate] = ss.get_length(ss[begin(ss) + candidate]);
        }
    }

    template <typename LcpIterator>
    GeneratedHashesLocalDupsEOSCandidates<HashStringIndex> generateHashStringIndices(
        StringSet ss, std::vector<size_t> const& candidates, size_t depth, LcpIterator lcps
    ) {
        using std::begin;

        if (candidates.empty()) {
            return {};
        }

        std::vector<HashStringIndex> hash_idx_pairs;
        std::vector<size_t> eos_candidates;
        std::vector<size_t> local_lcp_dups;
        eos_candidates.reserve(candidates.size());
        hash_idx_pairs.reserve(candidates.size());
        local_lcp_dups.reserve(candidates.size());

        for (auto prev = candidates.front(); auto const& curr: candidates) {
            auto const& curr_str = ss[begin(ss) + curr];

            if (depth > ss.get_length(curr_str)) {
                eos_candidates.push_back(curr);
            } else if (prev + 1 == curr && lcps[curr] >= depth) {
                local_lcp_dups.push_back(curr);
                if (hash_idx_pairs.back().stringIndex + 1 == curr) {
                    hash_idx_pairs.back().isLcpLocalRoot = true;
                }
            } else {
                size_t hash = HashPolicy::hash(ss.get_chars(curr_str, 0), depth, filter_size);
                hash_idx_pairs.emplace_back(hash, curr);
                hashValues[curr] = hash;
            }
            prev = curr;
        }
        return {std::move(hash_idx_pairs), std::move(local_lcp_dups), std::move(eos_candidates)};
    }

    template <typename LcpIterator>
    GeneratedHashesLocalDupsEOSCandidates<HashStringIndex>
    generateHashStringIndices(StringSet const& ss, size_t depth, LcpIterator lcps) {
        using std::begin;

        if (ss.empty()) {
            return {};
        }

        std::vector<HashStringIndex> hash_idx_pairs;
        std::vector<size_t> local_lcp_dups;
        std::vector<size_t> eos_candidates;
        eos_candidates.reserve(ss.size());
        hash_idx_pairs.reserve(ss.size());
        local_lcp_dups.reserve(ss.size());

        for (size_t candidate = 0; candidate != ss.size(); ++candidate) {
            auto const& str = ss[begin(ss) + candidate];

            if (depth > ss.get_length(str)) {
                eos_candidates.push_back(candidate);
            } else if (lcps[candidate] >= depth) {
                local_lcp_dups.push_back(candidate);
                if (hash_idx_pairs.back().stringIndex + 1 == candidate) {
                    hash_idx_pairs.back().isLcpLocalRoot = true;
                }
            } else {
                auto hash = HashPolicy::hash(ss.get_chars(str, 0), depth, filter_size);
                hash_idx_pairs.emplace_back(hash, candidate);
                hashValues[candidate] = hash;
            }
        }
        return {std::move(hash_idx_pairs), std::move(local_lcp_dups), std::move(eos_candidates)};
    }

    static std::vector<size_t> getSortedIndicesOfDuplicates(
        const size_t size,
        std::vector<size_t>& local_hash_dups,
        std::vector<size_t>& local_lcp_dups,
        std::vector<size_t>& remote_dups,
        std::vector<HashStringIndex> const& orig_string_idxs
    ) {
        std::vector<size_t> sorted_remote_dups;
        sorted_remote_dups.reserve(remote_dups.size());
        for (auto const& idx: remote_dups) {
            auto const& original = orig_string_idxs[idx];
            if (!original.isLocalDuplicateButSendAnyway) {
                sorted_remote_dups.push_back(original.stringIndex);
            }
        }

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

    static bool shouldSend(HashStringIndex const& v) {
        return !v.isLocalDuplicate || v.isLocalDuplicateButSendAnyway;
    }
};

} // namespace bloomfilter
} // namespace dss_mehnert
