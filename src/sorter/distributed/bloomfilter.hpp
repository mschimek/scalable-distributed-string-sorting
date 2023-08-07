// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <bitset>
#include <numeric>
#include <random>
#include <type_traits>

#include <ips4o.hpp>
#include <tlx/algorithm/multiway_merge.hpp>
#include <tlx/siphash.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "encoding/golomb_encoding.hpp"
#include "hash/xxhash.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/environment.hpp"
#include "strings/stringtools.hpp"
#include "util/measuringTool.hpp"

namespace dss_schimek {

struct Duplicate {
    size_t index;
    bool hasReachedEOS;
    Duplicate(size_t index, bool hasReachedEOS) : index(index), hasReachedEOS(hasReachedEOS) {}
};

struct HashTriple {
    size_t hashValue;
    size_t stringIndex;
    size_t PEIndex;
    HashTriple() = default;

    HashTriple(size_t hashValue, size_t stringIndex, size_t PEIndex)
        : hashValue(hashValue),
          stringIndex(stringIndex),
          PEIndex(PEIndex) {}

    bool operator<(HashTriple const& rhs) const { return hashValue < rhs.hashValue; }

    friend std::ostream& operator<<(std::ostream& stream, HashTriple const& hashTriple) {
        return stream << "[" << hashTriple.hashValue << ", " << hashTriple.stringIndex << ", "
                      << hashTriple.PEIndex << "]";
    }
    operator std::string() const {
        return "(" + std::to_string(hashValue) + ", " + std::to_string(stringIndex) + ", "
               + std::to_string(PEIndex) + ")";
    }
};

struct StringTriple {
    unsigned char const* string;
    size_t stringIndex;
    size_t PEIndex;
    StringTriple() = default;

    StringTriple(unsigned char const* string_, size_t stringIndex, size_t PEIndex)
        : string(string_),
          stringIndex(stringIndex),
          PEIndex(PEIndex) {}

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
    HashStringIndex(
        const size_t hashValue,
        const size_t stringIndex,
        bool isLocalDuplicate,
        bool isLocalDuplicateButSendAnyway
    )
        : hashValue(hashValue),
          stringIndex(stringIndex),
          isLocalDuplicate(isLocalDuplicate),
          isLocalDuplicateButSendAnyway(isLocalDuplicateButSendAnyway) {}
    HashStringIndex(const size_t hashValue, const size_t stringIndex)
        : hashValue(hashValue),
          stringIndex(stringIndex),
          isLocalDuplicate(false) {}

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
    HashPEIndex() : hashValue(0), PEIndex(0) {}
    HashPEIndex(const size_t hashValue, const size_t PEIndex)
        : hashValue(hashValue),
          PEIndex(PEIndex) {}

    bool operator<(HashPEIndex const& rhs) const { return hashValue < rhs.hashValue; }

    friend std::ostream& operator<<(std::ostream& stream, HashPEIndex const& hashPEIndex) {
        return stream << "[" << hashPEIndex.hashValue << ", " << hashPEIndex.PEIndex << "]";
    }
};

struct AllToAllHashesNaive {
    template <typename DataType>
    static inline std::vector<DataType> alltoallv(
        std::vector<DataType>& sendData,
        std::vector<size_t> const& intervalSizes,
        std::vector<size_t> const&,
        mpi::environment env
    ) {
        return alltoallv(sendData, intervalSizes, env);
    }

    template <typename DataType>
    static inline std::vector<DataType> alltoallv(
        std::vector<DataType>& sendData,
        std::vector<size_t> const& intervalSizes,
        mpi::environment env
    ) {
        using AllToAllv = dss_schimek::mpi::AllToAllvCombined<dss_schimek::mpi::AllToAllvSmall>;

        auto result = AllToAllv::alltoallv(sendData.data(), intervalSizes, env);
        return result;
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
        using AllToAllv = mpi::AllToAllvCombined<dss_schimek::mpi::AllToAllvSmall>;
        using namespace dss_schimek::measurement;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();
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
        // std::vector<size_t> recvEncodedValuesSizes =
        //    dss_schimek::mpi::alltoall(encodedValuesSizes);
        measuringTool.stop("bloomfilter_sendEncodedValues");
        measuringTool.add(
            std::accumulate(
                encodedValuesSizes.begin(),
                encodedValuesSizes.end(),
                static_cast<uint64_t>(0)
            ) * sizeof(size_t),
            "bloomfilter_sentEncodedValues",
            false
        );
        // measuringTool.add(std::accumulate(intervalSizes.begin(),
        // intervalSizes.end(), static_cast<uint64_t>(0)) * sizeof(size_t),
        //    "bloomfilter_unencodedValues", false);
        measuringTool.start("bloomfilter_golombDecoding");
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
        using AllToAllv = dss_schimek::mpi::AllToAllvCombined<dss_schimek::mpi::AllToAllvSmall>;
        using namespace dss_schimek::measurement;
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
            // std::cout << "rank: " << env.rank() << "intevalMax: " << intervalMax << "
            // intervalSize: " << intervalSize << " b = " << bFromBook << " encodedValues.size() "
            // << sizeEncodedValues << std::endl;
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
            // std::cout << "rank: " << env.rank() << " encodedIntervalSizes " <<
            // encodedIntervalSizes << " bFromBook " << bFromBook << std::endl;
            curDecodeIt = end;
        }
        // std::cout << "rank: " << env.rank() << " finished " << "decodedValues: " <<
        // decodedValues.size() << std::endl;
        return decodedValues;
    }
    static std::string getName() { return "sequentialGolombEncoding"; }
};

struct AllToAllHashValuesPipeline {
    static inline auto
    getStart(size_t partnerId, std::vector<size_t> const& startIndices, std::vector<size_t>& data) {
        return data.begin() + startIndices[partnerId];
    }

    template <typename InputIterator>
    static inline void getEncoding(
        const InputIterator begin,
        const InputIterator end,
        const size_t b,
        std::vector<size_t>& encoding
    ) {
        encoding.clear();
        encoding.reserve(1 + end - begin);
        encoding.push_back(0);
        getDeltaEncoding(begin, end, std::back_inserter(encoding), b);
        size_t encodingSize = encoding.size();
        encoding[0] = encodingSize - 1;
    }

private:
    template <typename InputIterator>
    static inline void pointToPoint(
        const InputIterator begin,
        const InputIterator end,
        const size_t partnerId,
        const size_t b,
        size_t* encoding,
        size_t* outputBuffer,
        MPI_Request* mpi_request
    ) {
        dss_schimek::mpi::environment env;
        using dss_schimek::measurement::MeasuringTool;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        // const size_t gigabyte = 1000;
        // const size_t maxSize = 5 * gigabyte;

        if (*encoding > env.mpi_max_int()) {
            std::cout << " too many values to send in 1-factor algorithm" << std::endl;
            std::abort();
        }
        int32_t sendSize = *encoding + 1;
        dss_schimek::mpi::data_type_mapper<size_t> dtm;
        measuringTool.addRawCommunication((*encoding + 1) * sizeof(size_t), "1factor");
        MPI_Isend(
            encoding,
            sendSize,
            dtm.get_mpi_type(),
            partnerId,
            42,
            env.communicator(),
            mpi_request
        );
        MPI_Irecv(
            outputBuffer,
            env.mpi_max_int(),
            dtm.get_mpi_type(),
            partnerId,
            42,
            env.communicator(),
            mpi_request + 1
        );
    }

public:
    static inline std::vector<std::vector<size_t>> alltoallv(
        std::vector<size_t>& sendData,
        std::vector<size_t>& intervalSizes,
        mpi::environment env,
        const size_t b = 1048576
    ) {
        using namespace dss_schimek::measurement;

        const size_t maxRecvSize = 1000000000u;
        std::vector<size_t*> recvBuffers(env.size(), nullptr);
        for (size_t i = 0; i < env.size(); ++i) {
            recvBuffers[i] = new size_t[maxRecvSize];
        }
        //
        //      std::vector<size_t> recvIntervalSizes =
        //      dss_schimek::mpi::alltoall(intervalSizes); std::vector<size_t>
        //      recvStartIndices; recvStartIndices.reserve(env.size());
        //      recvStartIndices.push_back(0);
        //      std::partial_sum(recvIntervalSizes.begin(),
        //      recvIntervalSizes.end(), std::back_inserter(recvStartIndices));

        std::vector<size_t> startIndices;
        startIndices.reserve(env.size());
        startIndices.push_back(0);
        std::partial_sum(
            intervalSizes.begin(),
            intervalSizes.end(),
            std::back_inserter(startIndices)
        );

        std::vector<std::vector<size_t>> encodings(env.size());
        std::vector<MPI_Request> requests((env.size() - 1) * 2);

        for (size_t j = 0; j < env.size() - 1; ++j) {
            size_t idlePE = (env.size() / 2 * j) % (env.size() - 1);
            if (env.rank() == env.size() - 1) {
                const size_t partnerId = idlePE;
                auto const curIt = sendData.begin() + startIndices[partnerId];
                auto const curEnd = curIt + intervalSizes[partnerId];
                getEncoding(curIt, curEnd, b, encodings[partnerId]);
                pointToPoint(
                    curIt,
                    curEnd,
                    partnerId,
                    b,
                    encodings[partnerId].data(),
                    recvBuffers[partnerId],
                    requests.data() + 2 * j
                );

                // exchange with PE idle
            } else if (env.rank() == idlePE) {
                const size_t partnerId = env.size() - 1;
                auto const curIt = sendData.begin() + startIndices[partnerId];
                auto const curEnd = curIt + intervalSizes[partnerId];
                getEncoding(curIt, curEnd, b, encodings[partnerId]);
                pointToPoint(
                    curIt,
                    curEnd,
                    partnerId,
                    b,
                    encodings[partnerId].data(),
                    recvBuffers[partnerId],
                    requests.data() + 2 * j
                );
                // exchange with PE env.size() - 1
            } else {
                const size_t partnerId = ((j + env.size()) - env.rank() - 1) % (env.size() - 1);
                auto const curIt = sendData.begin() + startIndices[partnerId];
                auto const curEnd = curIt + intervalSizes[partnerId];
                getEncoding(curIt, curEnd, b, encodings[partnerId]);
                pointToPoint(
                    curIt,
                    curEnd,
                    partnerId,
                    b,
                    encodings[partnerId].data(),
                    recvBuffers[partnerId],
                    requests.data() + 2 * j
                );

                // exchange with PE   ((j - i) % env.size() - 1
            }
        }

        std::vector<std::vector<size_t>> decodedVectors(env.size());
        std::copy_n(
            getStart(env.rank(), startIndices, sendData),
            intervalSizes[env.rank()],
            std::back_inserter(decodedVectors[env.rank()])
        );
        std::vector<bool> alreadyRecv(env.size() - 1, false);
        while (true) {
            size_t counter = 0;

            for (size_t j = 0; j < env.size() - 1; ++j) {
                if (alreadyRecv[j]) {
                    ++counter;
                    continue;
                }
                size_t idlePE = (env.size() / 2 * j) % (env.size() - 1);
                size_t partnerId = 0;
                if (env.rank() == env.size() - 1) {
                    partnerId = idlePE;
                } else if (env.rank() == idlePE) {
                    partnerId = env.size() - 1;
                } else {
                    partnerId = ((j + env.size()) - env.rank() - 1) % (env.size() - 1);
                }

                int32_t flag = 0;
                MPI_Test(requests.data() + 2 * j + 1, &flag, MPI_STATUSES_IGNORE);
                if (flag != 0) {
                    size_t* start = recvBuffers[partnerId];
                    const size_t recvEncodedValuesSize = *start;
                    alreadyRecv[j] = true;
                    getDeltaDecoding(
                        start + 1,
                        start + 1 + recvEncodedValuesSize,
                        std::back_inserter(decodedVectors[partnerId]),
                        b
                    );
                }
            }
            if (counter == env.size() - 1)
                break;
        }
        //      env.barrier();
        //      for (size_t i = 0; i < env.size(); ++i) {
        //	      auto it = recvValuesRef.begin();
        //	      std::vector<size_t> ref(it + recvStartIndices[i], it +
        // recvStartIndices[i] + recvSizesRef[i]); 	      if (ref !=
        // decodedVectors[i]) { 	              for (size_t j = 0; j <
        // decodedVectors[i].size() ; ++j) { 	        	      std::cout
        //<< "rank: " << env.rank() << " " << j << " " << ref[j] << " " <<
        // decodedVectors[i][j] << std::endl; 	        	      if
        // (decodedVectors[i][j] != ref[j]) break;
        //
        //	              }
        //	              std::cout << "rank: " << env.rank() << i << " not
        // recv what i sended "  << std::endl;
        //	      }
        //	      if (decodedVectors[i].size() != recvSizesRef[i]) {
        //
        //		std::cout << "rank: " << env.rank() << " from: " << i <<
        //" decoded size" << decodedVectors[i].size() << " != " <<
        // recvSizesRef[i]
        //<< std::endl; 		std::cout << "rank: " << env.rank() << "
        // from: " << i << " recvBufferSize " << *recvBuffers[i]  << std::endl;
        //		decodedVectors[i].pop_back();
        //		}
        //      }
        //      if (env.rank() == 0)
        //	std::cout << "reached end of sending" << std::endl;
        MPI_Waitall(2 * (env.size() - 1), requests.data(), MPI_STATUSES_IGNORE);
        for (size_t* ptr: recvBuffers) {
            delete[] ptr;
        }
        return decodedVectors;
    }

    static std::string getName() { return "GolombPipelined"; }
};

inline std::vector<size_t> computeIntervalSizes(
    std::vector<size_t> const& hashes, const size_t bloomFilterSize, mpi::environment env
) {
    std::vector<size_t> indices;

    indices.reserve(env.size());
    auto curPosWithinVector = hashes.begin();
    for (size_t i = 0; i + 1 < env.size(); ++i) {
        const size_t upperPartitionLimit = (i + 1) * (bloomFilterSize / env.size()) - 1;
        auto pos = std::upper_bound(curPosWithinVector, hashes.end(), upperPartitionLimit);
        indices.push_back(pos - curPosWithinVector);
        curPosWithinVector = pos;
    }
    indices.push_back(hashes.size() - (curPosWithinVector - hashes.begin()));
    return indices;
}

struct RecvData {
    std::vector<size_t> data;
    std::vector<size_t> intervalSizes;
    std::vector<size_t> globalOffsets;
    RecvData(
        std::vector<size_t>&& data,
        std::vector<size_t>&& intervalSizes,
        std::vector<size_t>&& globalOffsets
    )
        : data(std::move(data)),
          intervalSizes(std::move(intervalSizes)),
          globalOffsets(std::move(globalOffsets)) {}
};

template <typename SendPolicy>
struct SendOnlyHashesToFilter : private SendPolicy {
    using SendType = size_t;

    static inline std::vector<SendType>
    extractSendValues(std::vector<HashStringIndex> const& hashStringIndices) {
        dss_schimek::mpi::environment env;
        std::vector<size_t> hashValues;
        hashValues.reserve(hashStringIndices.size());
        for (auto const& hashStringIndex: hashStringIndices) {
            hashValues.push_back(hashStringIndex.hashValue);
        }

        return hashValues;
    }

    static inline RecvData
    sendToFilter(std::vector<HashStringIndex> const& hashes, size_t bloomfilterSize) {
        using namespace dss_schimek::measurement;
        dss_schimek::mpi::environment env;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("bloomfilter_sendToFilterSetup");
        std::vector<size_t> sendValues = extractSendValues(hashes);
        std::vector<size_t> intervalSizes = computeIntervalSizes(sendValues, bloomfilterSize, env);

        // assert TODO normal assert
        const size_t sumIntervalSizes =
            std::accumulate(intervalSizes.begin(), intervalSizes.end(), static_cast<size_t>(0u));
        if (hashes.size() != sumIntervalSizes) {
            std::cout << "rank: " << env.rank() << " sendValues.size() " << sendValues.size() << " "
                      << sumIntervalSizes << std::endl;
            std::abort();
        }

        std::vector<size_t> offsets;
        offsets.reserve(intervalSizes.size());
        offsets.push_back(0);
        std::partial_sum(
            intervalSizes.begin(),
            intervalSizes.end() - 1,
            std::back_inserter(offsets)
        );
        offsets = dss_schimek::mpi::alltoall(offsets, env);

        std::vector<size_t> recvIntervalSizes = dss_schimek::mpi::alltoall(intervalSizes, env);
        measuringTool.stop("bloomfilter_sendToFilterSetup");
        if constexpr (std::is_same<SendPolicy, dss_schimek::AllToAllHashValuesPipeline>::value) {
            measuringTool.start("bloomfilter_sendEncodedValuesOverall");
            auto const& unflattenedResult = SendPolicy::alltoallv(sendValues, intervalSizes, env);
            measuringTool.stop("bloomfilter_sendEncodedValuesOverall");
            std::vector<size_t> result = flatten(unflattenedResult);
            return RecvData(std::move(result), std::move(recvIntervalSizes), std::move(offsets));
        }
        if constexpr (std::is_same<SendPolicy, dss_schimek::AllToAllHashesGolomb>::value) {
            measuringTool.start("bloomfilter_sendEncodedValuesOverall");
            std::vector<size_t> result =
                SendPolicy::alltoallv(sendValues, intervalSizes, bloomfilterSize, env);
            measuringTool.stop("bloomfilter_sendEncodedValuesOverall");
            return RecvData(std::move(result), std::move(recvIntervalSizes), std::move(offsets));
        }
        if constexpr (std::is_same<SendPolicy, dss_schimek::AllToAllHashesNaive>::value) {
            measuringTool.start("bloomfilter_sendEncodedValuesOverall");
            std::vector<size_t> result = SendPolicy::alltoallv(sendValues, intervalSizes, env);
            measuringTool.stop("bloomfilter_sendEncodedValuesOverall");
            return RecvData(std::move(result), std::move(recvIntervalSizes), std::move(offsets));
        }
    }

    static inline std::vector<HashPEIndex> addPEIndex(RecvData const& recvData) {
        std::vector<HashPEIndex> hashesPEIndex;
        hashesPEIndex.reserve(recvData.data.size());

        size_t curPE = 0;
        size_t curBoundary = recvData.intervalSizes[0];
        for (size_t i = 0; i < recvData.data.size(); ++i) {
            while (i == curBoundary)
                curBoundary += recvData.intervalSizes[++curPE];
            hashesPEIndex.emplace_back(recvData.data[i], curPE);
        }
        return hashesPEIndex;
    }
};

template <typename GolombPolicy>
struct FindDuplicates {
    using DataType = HashPEIndex;

    static inline std::vector<size_t>
    findDuplicates(std::vector<HashPEIndex>& hashPEIndices, RecvData const& recvData) {
        using Iterator = std::vector<HashPEIndex>::iterator;
        using IteratorPair = std::pair<Iterator, Iterator>;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();
        dss_schimek::mpi::environment env;

        measuringTool.add(hashPEIndices.size(), "bloomfilter_recvHashValues", false);
        measuringTool.start("bloomfilter_findDuplicatesOverallIntern");
        measuringTool.start("bloomfilter_findDuplicatesSetup");
        std::vector<IteratorPair> iteratorPairs;
        size_t elementsToMerge = std::accumulate(
            recvData.intervalSizes.begin(),
            recvData.intervalSizes.end(),
            static_cast<size_t>(0u)
        );
        std::vector<HashPEIndex> mergedElements(elementsToMerge);
        Iterator it = hashPEIndices.begin();

        for (size_t i = 0; i < recvData.intervalSizes.size(); ++i) {
            iteratorPairs.emplace_back(it, it + recvData.intervalSizes[i]);
            it += recvData.intervalSizes[i];
        }
        measuringTool.stop("bloomfilter_findDuplicatesSetup");
        //
        //++++++++++++++++++++++++++++++++++++++
        //
        measuringTool.start("bloomfilter_findDuplicatesMerge");
        tlx::multiway_merge(
            iteratorPairs.begin(),
            iteratorPairs.end(),
            mergedElements.begin(),
            elementsToMerge
        );
        measuringTool.stop("bloomfilter_findDuplicatesMerge");
        //
        //++++++++++++++++++++++++++++++++++++++
        //

        measuringTool.start("bloomfilter_findDuplicatesFind");
        std::vector<std::vector<size_t>> result_sets(recvData.intervalSizes.size());
        std::vector<size_t> counters(recvData.intervalSizes.size(), 0);

        HashPEIndex prevHashTriple = mergedElements.empty() ? HashPEIndex{0, 0} : mergedElements[0];
        bool duplicate = false;

        for (size_t i = 1; i < mergedElements.size(); ++i) {
            const HashPEIndex curHashTriple = mergedElements[i];
            if (prevHashTriple.hashValue == curHashTriple.hashValue) {
                result_sets[prevHashTriple.PEIndex].push_back(counters[prevHashTriple.PEIndex]++);
                duplicate = true;
            } else if (duplicate) {
                result_sets[prevHashTriple.PEIndex].push_back(counters[prevHashTriple.PEIndex]++);
                duplicate = false;
            } else {
                ++counters[prevHashTriple.PEIndex];
            }
            prevHashTriple = curHashTriple;
        }

        if (duplicate)
            result_sets[prevHashTriple.PEIndex].push_back(counters[prevHashTriple.PEIndex]++);

        std::vector<size_t> sendBuffer;
        sendBuffer.reserve(elementsToMerge);
        std::vector<size_t> sendCounts_;
        // write back to PEs
        for (size_t i = 0; i < result_sets.size(); ++i) {
            sendCounts_.push_back(result_sets[i].size());
            for (size_t j = 0; j < result_sets[i].size(); ++j) {
                sendBuffer.push_back(result_sets[i][j] + recvData.globalOffsets[i]);
            }
        }
        measuringTool.stop("bloomfilter_findDuplicatesFind");
        //
        //++++++++++++++++++++++++++++
        //
        size_t totalNumSendDuplicates =
            std::accumulate(sendCounts_.begin(), sendCounts_.end(), static_cast<size_t>(0u));
        measuringTool.add(
            totalNumSendDuplicates * sizeof(size_t),
            "bloomfilter_findDuplicatesSendDups"
        );
        measuringTool.start("bloomfilter_findDuplicatesSendDups");
        int mpiSmallTypes = 0;
        if (totalNumSendDuplicates > 0)
            mpiSmallTypes = 1;
        bool dupsToSend = (0 != dss_schimek::mpi::allreduce_max(mpiSmallTypes, env));
        std::vector<size_t> duplicates;
        // if (dupsToSend)
        //    duplicates = dss_schimek::mpi::AllToAllvSmall::alltoallv(
        //        sendBuffer.data(), sendCounts_);
        if (dupsToSend) {
            duplicates =
                GolombPolicy::alltoallv(sendBuffer, sendCounts_, recvData.intervalSizes, env);
            // duplicates = dss_schimek::mpi::AllToAllvSmall::alltoallv(
            //     sendBuffer.data(), sendCounts_);
        }
        measuringTool.stop("bloomfilter_findDuplicatesSendDups");
        measuringTool.stop("bloomfilter_findDuplicatesOverallIntern");

        return duplicates;
    }

    static std::vector<size_t> getSortedIndicesOfDuplicates(
        const size_t size,
        std::vector<size_t>& localHashDuplicates,
        std::vector<size_t>& localDuplicates,
        std::vector<size_t>& remoteDuplicates,
        std::vector<HashStringIndex> const& originalMapping
    ) {
        std::vector<size_t> indicesOfAllDuplicates(localDuplicates);
        std::vector<size_t> sortedIndicesOfRemoteDuplicates;
        sortedIndicesOfRemoteDuplicates.reserve(remoteDuplicates.size());
        dss_schimek::mpi::environment env;

        // std::cout << "rank: " << env.rank() << " localhashDuplicates: " <<
        // localHashDuplicates.size() << " localDuplicates: " <<
        // localDuplicates.size() << " remoteDuplicates: " <<
        // remoteDuplicates.size() << " originalMapping: " <<
        // originalMapping.size() << std::endl;
        // Assumption localHashDuplicates and localDuplicates are sorted

        // indicesOfAllDuplicates.reserve(localDuplicates.size() +
        // remoteDuplicates.size());
        for (size_t i = 0; i < remoteDuplicates.size(); ++i) {
            const size_t curIndex = remoteDuplicates[i];

            bool isAlsoLocalDuplicate = originalMapping[curIndex].isLocalDuplicateButSendAnyway;
            if (!isAlsoLocalDuplicate) {
                const size_t stringIndex = originalMapping[curIndex].stringIndex;
                sortedIndicesOfRemoteDuplicates.push_back(stringIndex);
            }
        }

        ips4o::sort(sortedIndicesOfRemoteDuplicates.begin(), sortedIndicesOfRemoteDuplicates.end());
        ips4o::sort(localHashDuplicates.begin(), localHashDuplicates.end());
        using Iterator = std::vector<size_t>::iterator;
        std::vector<std::pair<Iterator, Iterator>> iteratorPairs;
        iteratorPairs.emplace_back(localHashDuplicates.begin(), localHashDuplicates.end());
        iteratorPairs.emplace_back(localDuplicates.begin(), localDuplicates.end());
        iteratorPairs.emplace_back(
            sortedIndicesOfRemoteDuplicates.begin(),
            sortedIndicesOfRemoteDuplicates.end()
        );
        const size_t elementsToMerge = localHashDuplicates.size() + localDuplicates.size()
                                       + sortedIndicesOfRemoteDuplicates.size();
        std::vector<size_t> mergedElements(elementsToMerge);
        tlx::multiway_merge(
            iteratorPairs.begin(),
            iteratorPairs.end(),
            mergedElements.begin(),
            elementsToMerge
        );

        return mergedElements;
    }
    static std::vector<size_t> getIndicesOfDuplicates(
        const size_t size,
        std::vector<size_t>& localDuplicates,
        std::vector<size_t>& remoteDuplicates,
        std::vector<HashStringIndex> const& originalMapping
    ) {
        std::vector<size_t> indicesOfAllDuplicates(localDuplicates);
        dss_schimek::mpi::environment env;
        // indicesOfAllDuplicates.reserve(localDuplicates.size() +
        // remoteDuplicates.size());
        for (size_t i = 0; i < remoteDuplicates.size(); ++i) {
            const size_t curIndex = remoteDuplicates[i];
            bool isAlsoLocalDuplicate = originalMapping[curIndex].isLocalDuplicateButSendAnyway;
            if (!isAlsoLocalDuplicate) {
                const size_t stringIndex = originalMapping[curIndex].stringIndex;
                indicesOfAllDuplicates.push_back(stringIndex);
                if (indicesOfAllDuplicates.back() >= size) {
                    std::cout << "rank: " << env.rank()
                              << " elem: " << indicesOfAllDuplicates.back() << " map.size() "
                              << originalMapping.size() << " curIndex " << curIndex << std::endl;
                }
            }
        }

        return indicesOfAllDuplicates;
    }
};

template <typename StringSet>
class ExcatDistinguishingPrefix {
    using String = typename StringSet::String;
    using Iterator = typename StringSet::Iterator;
    using CharIt = typename StringSet::CharIterator;
    using StringLcpPtr = typename tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    dss_schimek::mpi::environment env;

    void computeExactDistPrefixLengths(
        std::vector<StringTriple>& stringTriples, std::vector<size_t>& distinguishingPrefixLength
    ) {
        dss_schimek::mpi::environment env;
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
        using Container = dss_schimek::StringLcpContainer<StringSet>;
        Container container;
        std::vector<size_t> intervalSizes;
        std::vector<size_t> stringIndices;
        ContainerSizesIndices(
            Container&& container,
            std::vector<size_t>&& intervalSizes,
            std::vector<size_t>&& stringIndices
        )
            : container(std::move(container)),
              intervalSizes(std::move(intervalSizes)),
              stringIndices(std::move(stringIndices)) {}
    };

    std::vector<StringTriple> generateStringTriples(ContainerSizesIndices& containerSizesIndices) {
        std::vector<size_t> const& intervalSizes = containerSizesIndices.intervalSizes;
        const StringSet globalSet = containerSizesIndices.container.make_string_set();
        std::vector<size_t>& stringIndices = containerSizesIndices.stringIndices;

        size_t totalNumSentStrings =
            std::accumulate(intervalSizes.begin(), intervalSizes.end(), static_cast<size_t>(0u));

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
        StringSet ss = strptr.active();
        std::vector<unsigned char> send_buffer;

        for (size_t j = 0; j < candidates.size(); ++j) {
            String str = ss[ss.begin() + candidates[j]];
            size_t string_length = ss.get_length(str) + 1;
            std::copy_n(ss.get_chars(str, 0), string_length, std::back_inserter(send_buffer));
        }
        size_t numStrings = candidates.size();

        std::vector<size_t> recvCounts = dss_schimek::mpi::allgather(numStrings, env);
        std::vector<size_t> stringIndices = dss_schimek::mpi::allgatherv(candidates, env);
        std::vector<unsigned char> recvBuffer = dss_schimek::mpi::allgatherv(send_buffer, env);
        return ContainerSizesIndices(
            std::move(recvBuffer),
            std::move(recvCounts),
            std::move(stringIndices)
        );
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

    dss_schimek::mpi::environment env;

    void computeExactDistPrefixLengths(
        std::vector<StringTriple>& stringTriples, std::vector<size_t>& distinguishingPrefixLength
    ) {
        dss_schimek::mpi::environment env;
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
        using Container = dss_schimek::StringContainer<StringSet>;
        Container container;
        std::vector<size_t> intervalSizes;
        std::vector<size_t> stringIndices;
        ContainerSizesIndices(
            Container&& container,
            std::vector<size_t>&& intervalSizes,
            std::vector<size_t>&& stringIndices
        )
            : container(std::move(container)),
              intervalSizes(std::move(intervalSizes)),
              stringIndices(std::move(stringIndices)) {}
    };

    std::vector<StringTriple> generateStringTriples(ContainerSizesIndices& containerSizesIndices) {
        std::vector<size_t> const& intervalSizes = containerSizesIndices.intervalSizes;
        const StringSet globalSet = containerSizesIndices.container.make_string_set();
        std::vector<size_t>& stringIndices = containerSizesIndices.stringIndices;

        size_t totalNumSentStrings =
            std::accumulate(intervalSizes.begin(), intervalSizes.end(), static_cast<size_t>(0u));

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
        StringSet ss = strptr.active();
        std::vector<unsigned char> send_buffer;

        for (size_t j = 0; j < candidates.size(); ++j) {
            String str = ss[ss.begin() + candidates[j]];
            size_t string_length = ss.get_length(str) + 1;
            std::copy_n(ss.get_chars(str, 0), string_length, std::back_inserter(send_buffer));
        }
        size_t numStrings = candidates.size();

        std::vector<size_t> recvCounts = dss_schimek::mpi::allgather(numStrings, env);
        std::vector<size_t> stringIndices = dss_schimek::mpi::allgatherv(candidates, env);
        std::vector<unsigned char> recvBuffer = dss_schimek::mpi::allgatherv(send_buffer, env);
        return ContainerSizesIndices(
            std::move(recvBuffer),
            std::move(recvCounts),
            std::move(stringIndices)
        );
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

    dss_schimek::mpi::environment env;

    const size_t size;
    std::vector<size_t> hashValues;
    const size_t startDepth;
    static constexpr bool hashValueOptimization = true;

public:
    BloomFilter(const size_t size, const size_t startDepth)
        : size(size),
          hashValues(size, 0),
          startDepth(startDepth) {}
    //~BloomFilter() { delete[] hashValues; }
    const size_t bloomFilterSize = std::numeric_limits<uint64_t>::max(
    ); // set to this size because distribution/load balancing was not
       // good enough TODO Discuss Multisequence Selection?

    template <typename T>
    struct GeneratedHashStructuresEOSCandidates {
        std::vector<T> data;
        std::vector<size_t> eosCandidates;
        GeneratedHashStructuresEOSCandidates(

            std::vector<T>&& data, std::vector<size_t>&& eosCandidates
        )
            : data(std::move(data)),
              eosCandidates(std::move(eosCandidates)) {}
    };

    template <typename T>
    struct GeneratedHashesLocalDupsEOSCandidates {
        std::vector<T> data;
        std::vector<size_t> localDups;
        std::vector<size_t> eosCandidates;

        GeneratedHashesLocalDupsEOSCandidates(
            std::vector<T>&& data,
            std::vector<size_t>&& localDups,
            std::vector<size_t>&& eosCandidates
        )
            : data(std::move(data)),
              localDups(std::move(localDups)),
              eosCandidates(std::move(eosCandidates)) {}
    };

    template <typename LcpIterator>
    GeneratedHashesLocalDupsEOSCandidates<HashStringIndex> generateHashStringIndices(
        StringSet ss, std::vector<size_t> const& candidates, const size_t depth, LcpIterator lcps
    ) {
        std::vector<HashStringIndex> hashStringIndices;
        std::vector<size_t> eosCandidates;
        std::vector<size_t> localDups;
        if (candidates.size() == 0) {
            return {std::move(hashStringIndices), std::move(localDups), std::move(eosCandidates)};
        }
        eosCandidates.reserve(candidates.size());
        hashStringIndices.reserve(candidates.size());
        localDups.reserve(candidates.size());
        const Iterator begin = ss.begin();

        // special case for first element
        {
            size_t curCandidate = candidates.front();
            String curString = ss[begin + curCandidate];
            const size_t length = ss.get_length(curString);
            if (depth > length) {
                eosCandidates.push_back(curCandidate);
            } else {
                const size_t curHash =
                    HashPolicy::hash(ss.get_chars(curString, 0), depth, bloomFilterSize);
                hashStringIndices.emplace_back(curHash, curCandidate);
                hashValues[curCandidate] = curHash;
            }
        }

        for (size_t i = 1; i < candidates.size(); ++i) {
            size_t curCandidate = candidates[i];
            size_t prevCandidate = candidates[i - 1];

            String curString = ss[begin + curCandidate];
            const size_t length = ss.get_length(curString);
            if (depth > length) {
                eosCandidates.push_back(curCandidate);
            } else if (prevCandidate + 1 == curCandidate && lcps[curCandidate] >= depth) {
                localDups.push_back(curCandidate);
                if (hashStringIndices.back().stringIndex + 1 == curCandidate) {
                    hashStringIndices.back().isLcpLocalRoot = true;
                    if (env.rank() == 25 && curCandidate == 197869) {
                        std::cout << "generation: " << hashStringIndices.back().stringIndex
                                  << std::endl;
                    }
                }
            } else {
                const size_t curHash =
                    HashPolicy::hash(ss.get_chars(curString, 0), depth, bloomFilterSize);
                hashStringIndices.emplace_back(curHash, curCandidate);
                hashValues[curCandidate] = curHash;
            }
        }
        return {std::move(hashStringIndices), std::move(localDups), std::move(eosCandidates)};
    }
    GeneratedHashStructuresEOSCandidates<HashStringIndex> generateHashStringIndices(
        StringSet ss, std::vector<size_t> const& candidates, const size_t depth
    ) {
        std::vector<HashStringIndex> hashStringIndices;
        std::vector<size_t> eosCandidates;
        eosCandidates.reserve(candidates.size());
        hashStringIndices.reserve(candidates.size());
        const Iterator begin = ss.begin();
        if (!hashValueOptimization) {
            for (const size_t curCandidate: candidates) {
                String curString = ss[begin + curCandidate];
                const size_t length = ss.get_length(curString);
                if (depth > length) {
                    eosCandidates.push_back(curCandidate);
                } else {
                    const size_t curHash =
                        HashPolicy::hash(ss.get_chars(curString, 0), depth, bloomFilterSize);
                    hashStringIndices.emplace_back(curHash, curCandidate);
                    hashValues[curCandidate] = curHash;
                }
            }
        } else {
            for (const size_t curCandidate: candidates) {
                String curString = ss[begin + curCandidate];
                const size_t length = ss.get_length(curString);
                if (depth > length) {
                    eosCandidates.push_back(curCandidate);
                } else {
                    const size_t halfDepth = depth / 2;
                    const size_t curHash = HashPolicy::hash(
                        ss.get_chars(curString, halfDepth),
                        halfDepth,
                        bloomFilterSize,
                        hashValues[curCandidate]
                    );
                    hashStringIndices.emplace_back(curHash, curCandidate);
                    hashValues[curCandidate] = curHash;
                }
            }
        }

        return GeneratedHashStructuresEOSCandidates(
            std::move(hashStringIndices),
            std::move(eosCandidates)
        );
    }

    template <typename LcpIterator>
    GeneratedHashesLocalDupsEOSCandidates<HashStringIndex>
    generateHashStringIndices(StringSet ss, const size_t depth, LcpIterator lcps) {
        std::vector<HashStringIndex> hashStringIndices;
        std::vector<size_t> eosCandidates;
        hashStringIndices.reserve(ss.size());
        const Iterator begin = ss.begin();

        std::vector<size_t> localDups;
        if (ss.size() == 0) {
            return {std::move(hashStringIndices), std::move(localDups), std::move(eosCandidates)};
        }
        eosCandidates.reserve(ss.size());
        hashStringIndices.reserve(ss.size());
        localDups.reserve(ss.size());

        // special case for first element
        {
            size_t curCandidate = 0;
            String curString = ss[begin + curCandidate];
            const size_t length = ss.get_length(curString);
            if (depth > length) {
                eosCandidates.push_back(curCandidate);
            } else {
                const size_t curHash =
                    HashPolicy::hash(ss.get_chars(curString, 0), depth, bloomFilterSize);
                hashStringIndices.emplace_back(curHash, curCandidate);
                hashValues[curCandidate] = curHash;
            }
        }

        for (size_t curCandidate = 1; curCandidate < ss.size(); ++curCandidate) {
            String curString = ss[begin + curCandidate];
            const size_t length = ss.get_length(curString);
            if (depth > length) {
                eosCandidates.push_back(curCandidate);
            } else if (lcps[curCandidate] >= depth) {
                localDups.push_back(curCandidate);
                if (hashStringIndices.back().stringIndex + 1 == curCandidate)
                    hashStringIndices.back().isLcpLocalRoot = true;
            } else {
                const size_t curHash =
                    HashPolicy::hash(ss.get_chars(curString, 0), depth, bloomFilterSize);
                hashStringIndices.emplace_back(curHash, curCandidate);
                hashValues[curCandidate] = curHash;
            }
        }

        return {std::move(hashStringIndices), std::move(localDups), std::move(eosCandidates)};
    }

    std::vector<size_t> getIndicesOfLocalDuplicates(std::vector<HashStringIndex>& hashStringIndices
    ) {
        std::vector<size_t> indicesOfLocalDuplicates;
        if (hashStringIndices.empty())
            return indicesOfLocalDuplicates;

        for (size_t i = 0; i < hashStringIndices.size() - 1;) {
            HashStringIndex& pivotHashStringIndex = hashStringIndices[i];
            size_t j = i + 1;
            HashStringIndex& curHashStringIndex = hashStringIndices[j];
            if (curHashStringIndex.hashValue == pivotHashStringIndex.hashValue) {
                pivotHashStringIndex.isLocalDuplicate = true;
                pivotHashStringIndex.isLocalDuplicateButSendAnyway = true;
                curHashStringIndex.isLocalDuplicate = true;
                indicesOfLocalDuplicates.push_back(pivotHashStringIndex.stringIndex);
                indicesOfLocalDuplicates.push_back(curHashStringIndex.stringIndex);

                ++j;
                while (j < hashStringIndices.size()
                       && hashStringIndices[j].hashValue == pivotHashStringIndex.hashValue) {
                    hashStringIndices[j].isLocalDuplicate = true;
                    indicesOfLocalDuplicates.push_back(hashStringIndices[j].stringIndex);
                    ++j;
                }
            } else if (pivotHashStringIndex.isLcpLocalRoot) {
                pivotHashStringIndex.isLocalDuplicate = true;
                pivotHashStringIndex.isLocalDuplicateButSendAnyway = true;
                indicesOfLocalDuplicates.push_back(pivotHashStringIndex.stringIndex);
            }
            i = j;
        }
        if (hashStringIndices.back().isLcpLocalRoot) {
            auto& pivotHashStringIndex = hashStringIndices.back();
            pivotHashStringIndex.isLocalDuplicate = true;
            pivotHashStringIndex.isLocalDuplicateButSendAnyway = true;
            indicesOfLocalDuplicates.push_back(pivotHashStringIndex.stringIndex);
        }
        return indicesOfLocalDuplicates;
    }

    void setDepth(
        StringLcpPtr strptr,
        const size_t depth,
        std::vector<size_t> const& eosCandidates,
        std::vector<size_t>& results
    ) {
        // eosCandidates is subset of candidates whose length is <= depth
        StringSet ss = strptr.active();
        for (size_t candidate = 0; candidate < ss.size(); ++candidate)
            results[candidate] = depth;

        for (const size_t curEOSCandidate: eosCandidates) {
            String str = ss[ss.begin() + curEOSCandidate];
            size_t length = ss.get_length(str);
            results[curEOSCandidate] = length;
        }
    }

    void setDepth(
        StringLcpPtr strptr,
        const size_t depth,
        std::vector<size_t> const& candidates,
        std::vector<size_t> const& eosCandidates,
        std::vector<size_t>& results
    ) {
        // eosCandidates is subset of candidates whose length is <= depth
        StringSet ss = strptr.active();
        for (const size_t curCandidate: candidates)
            results[curCandidate] = depth;

        for (const size_t curEOSCandidate: eosCandidates) {
            String str = ss[ss.begin() + curEOSCandidate];
            size_t length = ss.get_length(str);
            results[curEOSCandidate] = length;
        }
    }

    // Don't need candidates list in first iteration -> all strings are
    // candidates
    std::vector<size_t>
    filter(StringLcpPtr strptr, const size_t depth, std::vector<size_t>& results) {
        dss_schimek::mpi::environment env;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("bloomfilter_generateHashStringIndices");
        // GeneratedHashStructuresEOSCandidates<HashStringIndex>
        //    hashStringIndicesEOSCandidates =
        //        generateHashStringIndices(strptr.active(), depth);
        GeneratedHashesLocalDupsEOSCandidates<HashStringIndex> hashStringIndicesEOSCandidates =
            generateHashStringIndices(strptr.active(), depth, strptr.lcp());

        std::vector<size_t>& localLcpDuplicates = hashStringIndicesEOSCandidates.localDups;
        std::vector<HashStringIndex>& hashStringIndices = hashStringIndicesEOSCandidates.data;
        std::vector<size_t> const& eosCandidates = hashStringIndicesEOSCandidates.eosCandidates;
        measuringTool.stop("bloomfilter_generateHashStringIndices");

        measuringTool.start("bloomfilter_sortHashStringIndices");
        ips4o::sort(hashStringIndices.begin(), hashStringIndices.end());
        // std::sort(hashStringIndices.begin(), hashStringIndices.end());
        measuringTool.stop("bloomfilter_sortHashStringIndices");

        measuringTool.start("bloomfilter_indicesOfLocalDuplicates");
        std::vector<size_t> indicesOfLocalDuplicates =
            getIndicesOfLocalDuplicates(hashStringIndices);
        measuringTool.stop("bloomfilter_indicesOfLocalDuplicates");

        measuringTool.start("bloomfilter_ReducedHashStringIndices");
        std::vector<HashStringIndex> reducedHashStringIndices;
        reducedHashStringIndices.reserve(hashStringIndices.size());
        std::copy_if(
            hashStringIndices.begin(),
            hashStringIndices.end(),
            std::back_inserter(reducedHashStringIndices),
            [&](HashStringIndex const& v) {
                return !v.isLocalDuplicate || v.isLocalDuplicateButSendAnyway;
            }
        );
        measuringTool.stop("bloomfilter_ReducedHashStringIndices");

        measuringTool.start("bloomfilter_sendHashStringIndices");
        RecvData recvData = SendPolicy::sendToFilter(reducedHashStringIndices, bloomFilterSize);
        measuringTool.stop("bloomfilter_sendHashStringIndices");

        measuringTool.start("bloomfilter_addPEIndex");
        std::vector<HashPEIndex> recvHashPEIndices = SendPolicy::addPEIndex(recvData);
        measuringTool.stop("bloomfilter_addPEIndex");

        // measuringTool.start(std::string("bloomfilter_findDuplicatesOverall"),
        // curIteration);
        std::vector<size_t> indicesOfRemoteDuplicates =
            FindDuplicatesPolicy::findDuplicates(recvHashPEIndices, recvData);
        // measuringTool.end(std::string("bloomfilter_findDuplicatesOverall"),
        // curIteration);

        measuringTool.start("bloomfilter_getIndices");

        std::vector<size_t> indicesOfAllDuplicates =
            FindDuplicatesPolicy::getSortedIndicesOfDuplicates(
                strptr.active().size(),
                indicesOfLocalDuplicates,
                localLcpDuplicates,
                indicesOfRemoteDuplicates,
                reducedHashStringIndices
            );
        measuringTool.stop("bloomfilter_getIndices");

        measuringTool.start("bloomfilter_setDepth");
        setDepth(strptr, depth, eosCandidates, results);
        measuringTool.stop("bloomfilter_setDepth");

        return indicesOfAllDuplicates;
    }

    std::vector<size_t> filter(
        StringLcpPtr strptr,
        const size_t depth,
        std::vector<size_t> const& candidates,
        std::vector<size_t>& results
    ) {
        dss_schimek::mpi::environment env;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("bloomfilter_generateHashStringIndices");
        GeneratedHashesLocalDupsEOSCandidates<HashStringIndex> hashStringIndicesEOSCandidates =
            generateHashStringIndices(strptr.active(), candidates, depth, strptr.lcp());

        std::vector<size_t>& localLcpDuplicates = hashStringIndicesEOSCandidates.localDups;
        std::vector<HashStringIndex>& hashStringIndices = hashStringIndicesEOSCandidates.data;
        std::vector<size_t> const& eosCandidates = hashStringIndicesEOSCandidates.eosCandidates;
        measuringTool.stop("bloomfilter_generateHashStringIndices");

        measuringTool.start("bloomfilter_sortHashStringIndices");
        ips4o::sort(hashStringIndices.begin(), hashStringIndices.end());
        measuringTool.stop("bloomfilter_sortHashStringIndices");

        measuringTool.start("bloomfilter_indicesOfLocalDuplicates");
        std::vector<size_t> indicesOfLocalDuplicates =
            getIndicesOfLocalDuplicates(hashStringIndices);
        measuringTool.stop("bloomfilter_indicesOfLocalDuplicates");

        measuringTool.start("bloomfilter_ReducedHashStringIndices");
        std::vector<HashStringIndex> reducedHashStringIndices;
        reducedHashStringIndices.reserve(hashStringIndices.size());
        std::copy_if(
            hashStringIndices.begin(),
            hashStringIndices.end(),
            std::back_inserter(reducedHashStringIndices),
            [&](HashStringIndex const& v) {
                return !v.isLocalDuplicate || v.isLocalDuplicateButSendAnyway;
            }
        );
        measuringTool.stop("bloomfilter_ReducedHashStringIndices");

        measuringTool.start("bloomfilter_sendHashStringIndices");
        RecvData recvData = SendPolicy::sendToFilter(reducedHashStringIndices, bloomFilterSize);
        measuringTool.stop("bloomfilter_sendHashStringIndices");

        measuringTool.start("bloomfilter_addPEIndex");
        std::vector<HashPEIndex> recvHashPEIndices = SendPolicy::addPEIndex(recvData);
        measuringTool.stop("bloomfilter_addPEIndex");

        std::vector<size_t> indicesOfRemoteDuplicates =
            FindDuplicatesPolicy::findDuplicates(recvHashPEIndices, recvData);

        measuringTool.start("bloomfilter_getIndices");
        std::vector<size_t> indicesOfAllDuplicates =
            FindDuplicatesPolicy::getSortedIndicesOfDuplicates(
                strptr.active().size(),
                indicesOfLocalDuplicates,
                localLcpDuplicates,
                indicesOfRemoteDuplicates,
                reducedHashStringIndices
            );
        measuringTool.stop("bloomfilter_getIndices");

        measuringTool.start("bloomfilter_setDepth");
        setDepth(strptr, depth, candidates, eosCandidates, results);
        measuringTool.stop("bloomfilter_setDepth");

        return indicesOfAllDuplicates;
    }
};
} // namespace dss_schimek
