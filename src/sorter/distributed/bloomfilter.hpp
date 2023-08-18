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

#include "encoding/golomb_encoding.hpp"
#include "hash/xxhash.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/communicator.hpp"
#include "mpi/environment.hpp"
#include "strings/stringtools.hpp"
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
        std::vector<DataType>& sendData,
        std::vector<size_t> const& intervalSizes,
        std::vector<size_t> const&,
        mpi::environment env
    ) {
        return alltoallv(sendData, intervalSizes, env);
    }

    template <typename DataType>
    static std::vector<DataType> alltoallv(
        std::vector<DataType>& sendData,
        std::vector<size_t> const& intervalSizes,
        mpi::environment env
    ) {
        using AllToAllv = mpi::AllToAllvCombined<mpi::AllToAllvSmall>;
        return AllToAllv::alltoallv(sendData.data(), intervalSizes, env);
    }

    static std::string getName() { return "noGolombEncoding"; }
};

struct AllToAllHashesGolomb {
    template <typename DataType>
    static inline std::vector<DataType> alltoallv(
        std::vector<DataType>& sendData,
        std::vector<size_t> const& intervalSizes,
        const size_t bloomFilterSize,
        mpi::environment env,
        const size_t b = 1048576
    ) {
        using AllToAllv = mpi::AllToAllvCombined<mpi::AllToAllvSmall>;

        auto& measuringTool = measurement::MeasuringTool::measuringTool();

        measuringTool.start("bloomfilter_golombEncoding");
        std::vector<size_t> encodedValuesSizes;
        std::vector<size_t> encodedValues;
        encodedValues.reserve(sendData.size() + 2 + env.size());

        auto begin = sendData.begin();

        if (intervalSizes.size() != env.size()) {
            std::cout << "not same size" << std::endl;
            std::abort();
        }
        for (size_t j = 0; j < intervalSizes.size(); ++j) {
            auto const intervalSize = intervalSizes[j];
            if (intervalSize == 0) {
                encodedValuesSizes.push_back(0);
                continue;
            }
            auto const end = begin + intervalSize;
            auto const encodedValuesSize = encodedValues.size();
            size_t bFromBook = getB(bloomFilterSize / env.size(), intervalSize);
            if (bFromBook == 1)
                bFromBook = 20000000000;
            encodedValues.push_back(0); // dummy value
            auto refToSize = encodedValues.size() - 1;
            encodedValues.push_back(bFromBook);

            getDeltaEncoding(begin, end, std::back_inserter(encodedValues), bFromBook);
            const size_t sizeEncodedValues = encodedValues.size() - encodedValuesSize;
            encodedValues[refToSize] = sizeEncodedValues - 1;
            encodedValuesSizes.push_back(sizeEncodedValues);
            begin = end;
        }
        measuringTool.stop("bloomfilter_golombEncoding");

        measuringTool.start("bloomfilter_sendEncodedValues");
        std::vector<size_t> recvEncodedValues =
            AllToAllv::alltoallv(encodedValues.data(), encodedValuesSizes, env);
        measuringTool.stop("bloomfilter_sendEncodedValues");
        measuringTool.add(
            std::accumulate(
                encodedValuesSizes.begin(),
                encodedValuesSizes.end(),
                static_cast<uint64_t>(0)
            ) * sizeof(size_t),
            "bloomfilter_sentEncodedValues"
        );
        measuringTool.start("bloomfilter_golombDecoding");
        std::vector<size_t> decodedValues;

        decodedValues.reserve(recvEncodedValues.size());
        auto curDecodeIt = recvEncodedValues.begin();

        for (size_t i = 0; i < env.size() && curDecodeIt != recvEncodedValues.end(); ++i) {
            size_t const encodedIntervalSizes = *(curDecodeIt++);
            auto const end = curDecodeIt + encodedIntervalSizes;
            size_t const bFromBook = *(curDecodeIt++);
            getDeltaDecoding(curDecodeIt, end, std::back_inserter(decodedValues), bFromBook);
            curDecodeIt = end;
        }
        measuringTool.stop("bloomfilter_golombDecoding");
        return decodedValues;
    }

    template <typename DataType>
    static inline std::vector<DataType> alltoallv(
        std::vector<DataType>& sendData,
        std::vector<size_t> const& intervalSizes,
        std::vector<size_t> const& intervalRange,
        mpi::environment env
    ) {
        using AllToAllv = mpi::AllToAllvCombined<mpi::AllToAllvSmall>;
        std::vector<size_t> encodedValuesSizes;
        std::vector<size_t> encodedValues;
        encodedValues.reserve(sendData.size() + 2 + env.size());

        auto begin = sendData.begin();

        if (intervalSizes.size() != env.size()) {
            std::cout << "not same size" << std::endl;
            std::abort();
        }
        for (size_t j = 0; j < intervalSizes.size(); ++j) {
            auto const intervalSize = intervalSizes[j];
            auto const intervalMax = intervalRange[j];
            if (intervalSize == 0) {
                encodedValuesSizes.push_back(0);
                continue;
            }
            auto const end = begin + intervalSize;
            auto const encodedValuesSize = encodedValues.size();

            size_t bFromBook = getB(intervalMax, intervalSize);
            if (bFromBook < 8)
                bFromBook = 8;
            encodedValues.push_back(0); // dummy value
            auto refToSize = encodedValues.size() - 1;
            encodedValues.push_back(bFromBook);

            getDeltaEncoding(begin, end, std::back_inserter(encodedValues), bFromBook);
            const size_t sizeEncodedValues = encodedValues.size() - encodedValuesSize;
            encodedValues[refToSize] = sizeEncodedValues - 1;
            encodedValuesSizes.push_back(sizeEncodedValues);
            begin = end;
        }

        std::vector<size_t> recvEncodedValues =
            AllToAllv::alltoallv(encodedValues.data(), encodedValuesSizes, env);
        std::vector<size_t> decodedValues;

        decodedValues.reserve(recvEncodedValues.size());
        auto curDecodeIt = recvEncodedValues.begin();

        for (size_t i = 0; i < env.size() && curDecodeIt != recvEncodedValues.end(); ++i) {
            const size_t encodedIntervalSizes = *(curDecodeIt++);
            auto const end = curDecodeIt + encodedIntervalSizes;
            const size_t bFromBook = *(curDecodeIt++);
            getDeltaDecoding(curDecodeIt, end, std::back_inserter(decodedValues), bFromBook);
            curDecodeIt = end;
        }

        return decodedValues;
    }
    static std::string getName() { return "sequentialGolombEncoding"; }
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
        std::vector<HashStringIndex> const& hashes,
        size_t filter_size,
        dss_mehnert::Communicator const& comm
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

        if constexpr (std::is_same_v<SendPolicy, AllToAllHashesNaive>) {
            measuringTool.start("bloomfilter_sendEncodedValuesOverall");
            auto result = SendPolicy::alltoallv(send_values, interval_sizes, comm);
            measuringTool.stop("bloomfilter_sendEncodedValuesOverall");
            return {std::move(result), std::move(recv_interval_sizes), std::move(offsets)};

        } else if constexpr (std::is_same_v<SendPolicy, AllToAllHashesGolomb>) {
            measuringTool.start("bloomfilter_sendEncodedValuesOverall");
            auto result = SendPolicy::alltoallv(send_values, interval_sizes, filter_size, comm);
            measuringTool.stop("bloomfilter_sendEncodedValuesOverall");
            return {std::move(result), std::move(recv_interval_sizes), std::move(offsets)};

        } else {
            []<bool flag = false> { static_assert(flag, "unexpected SendPolicy"); }
            ();
        }
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
        std::vector<HashPEIndex>& hash_triples,
        RecvData const& data,
        dss_mehnert::Communicator const& comm
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

    static std::vector<size_t> getSortedIndicesOfDuplicates(
        const size_t size,
        std::vector<size_t>& local_hash_dups,
        std::vector<size_t>& local_dups,
        std::vector<size_t>& remote_dups,
        std::vector<HashStringIndex> const& originalMapping
    ) {
        std::vector<size_t> sorted_remote_dups;
        sorted_remote_dups.reserve(remote_dups.size());
        for (auto const& idx: remote_dups) {
            auto const& original = originalMapping[idx];
            if (!original.isLocalDuplicateButSendAnyway) {
                sorted_remote_dups.push_back(original.stringIndex);
            }
        }

        ips4o::sort(sorted_remote_dups.begin(), sorted_remote_dups.end());
        ips4o::sort(local_hash_dups.begin(), local_hash_dups.end());

        using Iterator = std::vector<size_t>::iterator;
        std::array<std::pair<Iterator, Iterator>, 3> iter_pairs{
            {{local_hash_dups.begin(), local_hash_dups.end()},
             {local_dups.begin(), local_dups.end()},
             {sorted_remote_dups.begin(), sorted_remote_dups.end()}}};
        size_t total_elems = local_hash_dups.size() + local_dups.size() + sorted_remote_dups.size();

        std::vector<size_t> merged_elems(total_elems);
        tlx::multiway_merge(
            iter_pairs.begin(),
            iter_pairs.end(),
            merged_elems.begin(),
            total_elems
        );
        return merged_elems;
    }
};

template <typename StringSet>
class ExcatDistinguishingPrefix {
    using String = typename StringSet::String;
    using Iterator = typename StringSet::Iterator;
    using CharIt = typename StringSet::CharIterator;
    using StringLcpPtr = typename tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    void computeExactDistPrefixLengths(
        std::vector<StringTriple>& stringTriples, std::vector<size_t>& distinguishingPrefixLength
    ) {
        mpi::environment env;

        if (stringTriples.empty())
            return;

        std::stable_sort(stringTriples.begin(), stringTriples.end());

        for (size_t i = 1; i < stringTriples.size(); ++i) {
            const StringTriple prevStringTriple = stringTriples[i - 1];
            const StringTriple curStringTriple = stringTriples[i];

            unsigned char const* s1 = prevStringTriple.string;
            unsigned char const* s2 = curStringTriple.string;

            const size_t distValuePrevCur = 1 + dss_schimek::calc_lcp(s1, s2);
            if (prevStringTriple.PEIndex == env.rank()) {
                const size_t oldValue = distinguishingPrefixLength[prevStringTriple.stringIndex];
                distinguishingPrefixLength[prevStringTriple.stringIndex] =
                    distValuePrevCur > oldValue ? distValuePrevCur : oldValue;
            }
            if (curStringTriple.PEIndex == env.rank()) {
                const size_t oldValue = distinguishingPrefixLength[curStringTriple.stringIndex];
                distinguishingPrefixLength[curStringTriple.stringIndex] =
                    distValuePrevCur > oldValue ? distValuePrevCur : oldValue;
            }
        }
    }

    struct ContainerSizesIndices {
        StringLcpContainer<StringSet> container;
        std::vector<size_t> intervalSizes;
        std::vector<size_t> stringIndices;
    };

    std::vector<StringTriple> generateStringTriples(ContainerSizesIndices& containerSizesIndices) {
        mpi::environment env;

        std::vector<size_t> const& intervalSizes = containerSizesIndices.intervalSizes;
        const StringSet globalSet = containerSizesIndices.container.make_string_set();
        std::vector<size_t>& stringIndices = containerSizesIndices.stringIndices;

        size_t totalNumSentStrings =
            std::accumulate(intervalSizes.begin(), intervalSizes.end(), size_t{0});

        std::vector<StringTriple> stringTriples;
        if (totalNumSentStrings == 0)
            return stringTriples;

        stringTriples.reserve(totalNumSentStrings);
        size_t curOffset = 0;
        auto begin = globalSet.begin();

        for (size_t curRank = 0; curRank < env.size(); ++curRank) {
            for (size_t i = 0; i < intervalSizes[curRank]; ++i) {
                String curString = globalSet[begin + curOffset + i];
                stringTriples.emplace_back(
                    globalSet.get_chars(curString, 0),
                    stringIndices[curOffset + i],
                    curRank
                );
            }
            curOffset += intervalSizes[curRank];
        }
        return stringTriples;
    }

    ContainerSizesIndices allgatherStrings(StringLcpPtr strptr, std::vector<size_t>& candidates) {
        mpi::environment env;

        StringSet ss = strptr.active();
        std::vector<unsigned char> send_buffer;

        for (size_t j = 0; j < candidates.size(); ++j) {
            String str = ss[ss.begin() + candidates[j]];
            size_t string_length = ss.get_length(str) + 1;
            std::copy_n(ss.get_chars(str, 0), string_length, std::back_inserter(send_buffer));
        }
        size_t numStrings = candidates.size();

        std::vector<size_t> recvCounts = mpi::allgather(numStrings, env);
        std::vector<size_t> stringIndices = mpi::allgatherv(candidates, env);
        std::vector<unsigned char> recvBuffer = mpi::allgatherv(send_buffer, env);
        return {std::move(recvBuffer), std::move(recvCounts), std::move(stringIndices)};
    }

public:
    void filter_exact(
        StringLcpPtr strptr, std::vector<size_t>& candidates, std::vector<size_t>& results
    ) {
        ContainerSizesIndices containerSizesIndices = allgatherStrings(strptr, candidates);
        std::vector<StringTriple> globalStringTriples =
            generateStringTriples(containerSizesIndices);
        computeExactDistPrefixLengths(globalStringTriples, results);
    }
};

template <typename StringSet>
class BloomfilterTest {
    using String = typename StringSet::String;
    using Iterator = typename StringSet::Iterator;
    using CharIt = typename StringSet::CharIterator;
    using StringLcpPtr = typename tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    mpi::environment env;

    void computeExactDistPrefixLengths(
        std::vector<StringTriple>& stringTriples, std::vector<size_t>& distinguishingPrefixLength
    ) {
        mpi::environment env;
        if (stringTriples.empty())
            return;

        std::stable_sort(stringTriples.begin(), stringTriples.end());

        for (size_t i = 1; i < stringTriples.size(); ++i) {
            const StringTriple prevStringTriple = stringTriples[i - 1];
            const StringTriple curStringTriple = stringTriples[i];

            unsigned char const* s1 = prevStringTriple.string;
            unsigned char const* s2 = curStringTriple.string;

            const size_t distValuePrevCur = 1 + dss_schimek::calc_lcp(s1, s2);
            if (prevStringTriple.PEIndex == env.rank()) {
                const size_t oldValue = distinguishingPrefixLength[prevStringTriple.stringIndex];
                distinguishingPrefixLength[prevStringTriple.stringIndex] =
                    distValuePrevCur > oldValue ? distValuePrevCur : oldValue;
            }
            if (curStringTriple.PEIndex == env.rank()) {
                const size_t oldValue = distinguishingPrefixLength[curStringTriple.stringIndex];
                distinguishingPrefixLength[curStringTriple.stringIndex] =
                    distValuePrevCur > oldValue ? distValuePrevCur : oldValue;
            }
        }
    }

    struct ContainerSizesIndices {
        StringContainer<StringSet> container;
        std::vector<size_t> intervalSizes;
        std::vector<size_t> stringIndices;
    };

    std::vector<StringTriple> generateStringTriples(ContainerSizesIndices& containerSizesIndices) {
        std::vector<size_t> const& intervalSizes = containerSizesIndices.intervalSizes;
        const StringSet globalSet = containerSizesIndices.container.make_string_set();
        std::vector<size_t>& stringIndices = containerSizesIndices.stringIndices;

        size_t totalNumSentStrings =
            std::accumulate(intervalSizes.begin(), intervalSizes.end(), size_t{0});

        std::vector<StringTriple> stringTriples;
        if (totalNumSentStrings == 0)
            return stringTriples;

        stringTriples.reserve(totalNumSentStrings);
        size_t curOffset = 0;
        auto begin = globalSet.begin();

        for (size_t curRank = 0; curRank < env.size(); ++curRank) {
            for (size_t i = 0; i < intervalSizes[curRank]; ++i) {
                String curString = globalSet[begin + curOffset + i];
                stringTriples.emplace_back(
                    globalSet.get_chars(curString, 0),
                    stringIndices[curOffset + i],
                    curRank
                );
            }
            curOffset += intervalSizes[curRank];
        }
        return stringTriples;
    }

    ContainerSizesIndices allgatherStrings(StringLcpPtr strptr, std::vector<size_t>& candidates) {
        auto const& ss = strptr.active();
        std::vector<unsigned char> send_buffer;

        for (size_t j = 0; j < candidates.size(); ++j) {
            String str = ss[ss.begin() + candidates[j]];
            size_t string_length = ss.get_length(str) + 1;
            std::copy_n(ss.get_chars(str, 0), string_length, std::back_inserter(send_buffer));
        }
        size_t numStrings = candidates.size();

        return {
            mpi::allgatherv(send_buffer, env),
            mpi::allgather(numStrings, env),
            mpi::allgatherv(candidates, env)};
    }

public:
    void filter_exact(
        StringLcpPtr strptr, std::vector<size_t>& candidates, std::vector<size_t>& results
    ) {
        ContainerSizesIndices containerSizesIndices = allgatherStrings(strptr, candidates);
        std::vector<StringTriple> globalStringTriples =
            generateStringTriples(containerSizesIndices);
        computeExactDistPrefixLengths(globalStringTriples, results);
    }
};

struct SipHasher {
    static inline size_t hash(unsigned char const* str, size_t length, size_t bloomFilterSize) {
        return tlx::siphash(str, length) % bloomFilterSize;
    }
};

struct XXHasher {
    static inline size_t hash(unsigned char const* str, size_t length, size_t bloomFilterSize) {
        xxh::hash_state_t<64> hash_stream;
        hash_stream.update(str, length);
        xxh::hash_t<64> hashV = hash_stream.digest();
        return hashV % bloomFilterSize;
    }

    static inline size_t hash(
        unsigned char const* str, size_t length, size_t bloomFilterSize, const size_t oldHashValue
    ) {
        xxh::hash_state_t<64> hash_stream;
        hash_stream.update(str, length);
        xxh::hash_t<64> hashV = hash_stream.digest();
        return (oldHashValue ^ hashV) % bloomFilterSize;
    }
};

template <
    typename StringSet,
    typename FindDuplicatesPolicy,
    typename SendPolicy,
    typename HashPolicy>
class BloomFilter {
    using String = typename StringSet::String;
    using Iterator = typename StringSet::Iterator;
    using CharIt = typename StringSet::CharIterator;
    using StringLcpPtr = typename tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

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
        StringLcpPtr strptr,
        size_t depth,
        std::vector<size_t>& results,
        dss_mehnert::Communicator const& comm
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
        auto local_dups = localDuplicateIndices(hash_idx_pairs);
        measuringTool.stop("bloomfilter_indicesOfLocalDuplicates");

        measuringTool.start("bloomfilter_ReducedHashStringIndices");
        std::erase_if(hash_idx_pairs, std::not_fn(should_send));
        measuringTool.stop("bloomfilter_ReducedHashStringIndices");

        measuringTool.start("bloomfilter_sendHashStringIndices");
        RecvData recvData = SendPolicy::sendToFilter(hash_idx_pairs, filter_size, comm);
        measuringTool.stop("bloomfilter_sendHashStringIndices");

        measuringTool.start("bloomfilter_addPEIndex");
        auto recvHashPEIndices = SendPolicy::addPEIndex(recvData);
        measuringTool.stop("bloomfilter_addPEIndex");

        auto indicesOfRemoteDuplicates =
            FindDuplicatesPolicy::findDuplicates(recvHashPEIndices, recvData, comm);

        measuringTool.start("bloomfilter_getIndices");
        auto indicesOfAllDuplicates = FindDuplicatesPolicy::getSortedIndicesOfDuplicates(
            strptr.active().size(),
            local_dups,
            local_lcp_dups,
            indicesOfRemoteDuplicates,
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
        dss_mehnert::Communicator const& comm
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
        auto local_dups = localDuplicateIndices(hash_idx_pairs);
        measuringTool.stop("bloomfilter_indicesOfLocalDuplicates");

        measuringTool.start("bloomfilter_ReducedHashStringIndices");
        std::erase_if(hash_idx_pairs, std::not_fn(should_send));
        measuringTool.stop("bloomfilter_ReducedHashStringIndices");

        measuringTool.start("bloomfilter_sendHashStringIndices");
        RecvData recvData = SendPolicy::sendToFilter(hash_idx_pairs, filter_size, comm);
        measuringTool.stop("bloomfilter_sendHashStringIndices");

        measuringTool.start("bloomfilter_addPEIndex");
        auto recvHashPEIndices = SendPolicy::addPEIndex(recvData);
        measuringTool.stop("bloomfilter_addPEIndex");

        auto indicesOfRemoteDuplicates =
            FindDuplicatesPolicy::findDuplicates(recvHashPEIndices, recvData, comm);

        measuringTool.start("bloomfilter_getIndices");
        auto indicesOfAllDuplicates = FindDuplicatesPolicy::getSortedIndicesOfDuplicates(
            strptr.active().size(),
            local_dups,
            local_lcp_dups,
            indicesOfRemoteDuplicates,
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

    static bool should_send(HashStringIndex const& v) {
        return !v.isLocalDuplicate || v.isLocalDuplicateButSendAnyway;
    }
};

} // namespace bloomfilter
} // namespace dss_mehnert
