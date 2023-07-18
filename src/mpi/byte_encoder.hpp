// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "encoding/integer_compression.hpp"

namespace dss_schimek {
class IntegerCompression {
    struct CompressedIntegersCounts {
        std::vector<uint8_t> integers;
        std::vector<uint64_t> counts;
    };

public:
    // return number bytes written
    template <typename InputIterator, typename OutputIterator>
    static size_t write(InputIterator input, OutputIterator output, size_t numElemsToWrite) {
        dss_schimek::Writer writer(output);
        for (size_t i = 0; i < numElemsToWrite; ++i) {
            writer.PutVarint(*(input + i));
        }
        return writer.getNumPutBytes();
    }

    template <typename CountIterator, typename InputIterator>
    static CompressedIntegersCounts
    writeRanges(CountIterator beginCounts, CountIterator endCounts, InputIterator dataInputBegin) {
        const uint64_t sumOfLcpsToEncode = std::accumulate(beginCounts, endCounts, 0ull);
        CompressedIntegersCounts returnData;
        returnData.integers.resize(sumOfLcpsToEncode * sizeof(uint64_t));
        // just a guess that is far too conservative in most cases
        returnData.counts.reserve(endCounts - beginCounts);

        uint64_t curNumEncodedLcps = 0u;
        uint64_t curNumWrittenBytes = 0u;
        for (CountIterator it = beginCounts; it != endCounts; ++it) {
            const uint64_t usedBytes = write(
                dataInputBegin + curNumEncodedLcps,
                returnData.integers.begin() + curNumWrittenBytes,
                *it
            );
            returnData.counts.push_back(usedBytes);
            curNumWrittenBytes += usedBytes;
            curNumEncodedLcps += *it;
        }
        return returnData;
    }

    template <typename InputIterator, typename OutputIterator>
    static uint64_t read(InputIterator input, OutputIterator begin, OutputIterator end) {
        dss_schimek::Reader reader(input, begin, end);
        reader.decode();
        return reader.getNumReadBytes();
    }

    template <typename CountIterator, typename InputIterator>
    static std::vector<uint64_t>
    readRanges(CountIterator beginCounts, CountIterator endCounts, InputIterator dataInputBegin) {
        const uint64_t sumOfRecvLcpValues = std::accumulate(beginCounts, endCounts, 0ull);
        std::vector<uint64_t> output(sumOfRecvLcpValues);
        uint64_t curNumWrittenLcps = 0u;
        uint64_t curNumConsumedBytes = 0u;
        for (CountIterator it = beginCounts; it != endCounts; ++it) {
            curNumConsumedBytes += read(
                dataInputBegin + curNumConsumedBytes,
                output.begin() + curNumWrittenLcps,
                output.begin() + curNumWrittenLcps + *it
            );
            curNumWrittenLcps += *it;
        }
        return output;
    }
};

class EmptyByteEncoderMemCpy {
public:
    static std::string getName() { return "EmptyByteEncoderMemCpy"; }

    template <typename StringSet>
    std::pair<unsigned char*, size_t> write(unsigned char* buffer, const StringSet ss) const {
        using String = typename StringSet::String;

        const size_t size = ss.size();
        size_t chars_written = 0;

        auto begin = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[begin + i];
            size_t str_length = ss.get_length(str) + 1;

            chars_written += str_length;
            memcpy(buffer, ss.get_chars(str, 0), str_length);
            buffer += str_length;
        }
        return {buffer, chars_written};
    }
};

class EmptyLcpByteEncoderMemCpy {
public:
    static std::string getName() { return "EmptyLcpByteEncoderMemCpy"; }

    template <typename StringSet>
    std::pair<unsigned char*, size_t>
    write(unsigned char* buffer, const StringSet ss, size_t const* lcps) const {
        using String = typename StringSet::String;

        const size_t size = ss.size();
        size_t chars_written = 0;

        auto begin = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[begin + i];
            size_t str_length = ss.get_length(str) + 1;
            size_t str_lcp = lcps[i];
            size_t unique_chars = str_length - str_lcp;

            chars_written += unique_chars;
            memcpy(buffer, ss.get_chars(str, str_lcp), unique_chars);
            buffer += unique_chars;
        }
        return {buffer, chars_written};
    }
};

class EmptyPrefixDoublingLcpByteEncoderMemCpy {
public:
    static std::string getName() { return "EmptyPrefixDoublingLcpByteEncoderMemCpy"; }

    template <typename StringSet>
    std::pair<unsigned char*, size_t> write(
        unsigned char* buffer,
        const StringSet ss,
        size_t const* lcps,
        size_t const* distinguishingPrefixes
    ) const {
        using String = typename StringSet::String;

        const size_t size = ss.size();
        size_t chars_written = 0;

        auto begin = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[begin + i];
            size_t str_length = distinguishingPrefixes[i];
            size_t str_lcp = lcps[i];
            size_t unqiue_chars = str_length - str_lcp;

            chars_written += unqiue_chars + 1;
            memcpy(buffer, ss.get_chars(str, str_lcp), unqiue_chars);
            buffer += unqiue_chars;
            *buffer = 0;
            ++buffer;
        }
        return {buffer, chars_written};
    }
};

} // namespace dss_schimek
