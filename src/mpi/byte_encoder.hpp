// (c) 2019 Matthias Schimek
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
        returnData.integers.resize(
            sumOfLcpsToEncode * sizeof(uint64_t)
        ); // just a guess that is far too conservative in
           // most cases
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
    /*
     * No endcoding
     */
public:
    static std::string getName() { return "EmptyByteEncoderMemCpy"; }

    template <typename StringSet>
    std::pair<unsigned char*, size_t> write(unsigned char* buffer, const StringSet ss) const {
        using String = typename StringSet::String;

        const size_t size = ss.size();
        size_t numCharsWritten = 0;

        auto beginOfSet = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[beginOfSet + i];
            size_t stringLength = ss.get_length(str) + 1;
            numCharsWritten += stringLength;
            memcpy(buffer, ss.get_chars(str, 0), stringLength);
            buffer += stringLength;
        }
        return std::make_pair(buffer, numCharsWritten);
    }
};

class EmptyLcpByteEncoderMemCpy {
    /*
     * No endcoding
     */
public:
    static std::string getName() { return "EmptyLcpByteEncoderMemCpy"; }

    template <typename StringSet>
    std::pair<unsigned char*, size_t>
    write(unsigned char* buffer, const StringSet ss, size_t const* lcps) const {
        using String = typename StringSet::String;

        const size_t size = ss.size();
        size_t numCharsWritten = 0;

        auto beginOfSet = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[beginOfSet + i];
            size_t stringLength = ss.get_length(str) + 1;
            size_t stringLcp = lcps[i];
            size_t actuallyWrittenChars = stringLength - stringLcp;
            numCharsWritten += actuallyWrittenChars;
            memcpy(buffer, ss.get_chars(str, stringLcp), actuallyWrittenChars);
            buffer += actuallyWrittenChars;
        }
        return std::make_pair(buffer, numCharsWritten);
    }
};

class EmptyPrefixDoublingLcpByteEncoderMemCpy {
    /*
     * No endcoding
     */
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
        size_t numCharsWritten = 0;

        auto beginOfSet = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[beginOfSet + i];
            size_t stringLength = distinguishingPrefixes[i];
            size_t stringLcp = lcps[i];
            size_t actuallyWrittenChars = stringLength - stringLcp;
            memcpy(buffer, ss.get_chars(str, stringLcp), actuallyWrittenChars);
            buffer += actuallyWrittenChars;
            *buffer = 0;
            ++buffer;
            numCharsWritten += actuallyWrittenChars + 1;
        }
        return std::make_pair(buffer, numCharsWritten);
    }
};
class EmptyByteEncoderCopy {
    /*
     * No endcoding
     */
public:
    static std::string getName() { return "EmptyByteEncoderCopy"; }
};

class SequentialDelayedByteEncoder {
    /*
     * Encoding:
     * (|Number Chars : size_t | Number Strings: size_t | [char : unsigned char]
     * | [number : size_t] |)*
     *
     */
public:
    static std::string getName() { return "SequentialDelayedByteEncoder"; }
    size_t computeNumberOfSendBytes(size_t charsToSend, size_t numbersToSend) const {
        return charsToSend + sizeof(size_t) * numbersToSend + 2 * sizeof(size_t);
    }

    size_t computeNumberOfSendBytes(
        std::vector<size_t> const& charsToSend, std::vector<size_t> const& numbersToSend
    ) const {
        assert(charsToSend.size() == numbersToSend.size());
        size_t numberOfSendBytes = 0;
        for (size_t i = 0; i < charsToSend.size(); ++i) {
            numberOfSendBytes += computeNumberOfSendBytes(charsToSend[i], numbersToSend[i]);
        }
        return numberOfSendBytes;
    }

    std::pair<size_t, size_t>
    computeNumberOfRecvData(char unsigned const* buffer, size_t size) const {
        size_t recvChars = 0, recvNumbers = 0;
        size_t i = 0;
        while (i < size) {
            size_t curRecvChars = 0;
            size_t curRecvNumbers = 0;
            memcpy(&curRecvChars, buffer + i, sizeof(size_t));
            recvChars += curRecvChars;
            i += sizeof(size_t);
            memcpy(&curRecvNumbers, buffer + i, sizeof(size_t));
            recvNumbers += curRecvNumbers;
            i += sizeof(size_t);
            i += curRecvChars + sizeof(size_t) * curRecvNumbers;
        }
        return std::make_pair(recvChars, recvNumbers);
    }

    // think about common interface
    unsigned char* write(
        unsigned char* buffer,
        unsigned char const* charsToWrite,
        size_t numChars,
        size_t const* numbersToWrite,
        size_t numNumbers
    ) const {
        const size_t sizeOfSizeT = sizeof(size_t);

        memcpy(buffer, &numChars, sizeOfSizeT);
        buffer += sizeOfSizeT;
        memcpy(buffer, &numNumbers, sizeOfSizeT);
        buffer += sizeOfSizeT;
        memcpy(buffer, charsToWrite, numChars);
        buffer += numChars;
        const size_t bytesToWrite = sizeof(size_t) * numNumbers;
        memcpy(buffer, numbersToWrite, bytesToWrite);
        buffer += bytesToWrite;
        return buffer;
    }

    void read_(
        unsigned char* buffer,
        std::vector<unsigned char>& charsToRead,
        std::vector<size_t>& numbersToRead
    ) const {
        size_t numCharsToRead = 0;
        size_t numNumbersToRead = 0;
        memcpy(&numCharsToRead, buffer, sizeof(size_t));
        buffer += sizeof(size_t);
        memcpy(&numNumbersToRead, buffer, sizeof(size_t));
        buffer += sizeof(size_t);
        charsToRead.clear();
        charsToRead.reserve(numCharsToRead);
        numbersToRead.clear();
        numbersToRead.reserve(numNumbersToRead);

        for (size_t i = 0; i < numCharsToRead; ++i)
            charsToRead.emplace_back(buffer[i]);
        for (size_t i = 0; i < numNumbersToRead; ++i) {
            size_t tmp;
            std::memcpy(&tmp, buffer + numCharsToRead + i * sizeof(size_t), sizeof(size_t));
            numbersToRead.emplace_back(tmp);
        }
    }
    std::pair<std::vector<unsigned char>, std::vector<size_t>>
    read(unsigned char* buffer, size_t size) const {
        auto recvData = computeNumberOfRecvData(buffer, size);
        size_t numCharsToRead = recvData.first;
        size_t numNumbersToRead = recvData.second;
        std::vector<unsigned char> charsToRead;
        std::vector<size_t> numbersToRead;
        charsToRead.reserve(numCharsToRead);
        numbersToRead.reserve(numNumbersToRead);

        size_t curPos = 0;
        while (curPos < size) {
            memcpy(&numCharsToRead, buffer + curPos, sizeof(size_t));
            curPos += sizeof(size_t);
            memcpy(&numNumbersToRead, buffer + curPos, sizeof(size_t));
            curPos += sizeof(size_t);

            for (size_t i = 0; i < numCharsToRead; ++i)
                charsToRead.emplace_back(*(buffer + curPos + i));
            curPos += numCharsToRead;
            for (size_t i = 0; i < numNumbersToRead; ++i) {
                size_t tmp;
                std::memcpy(&tmp, buffer + curPos + i * sizeof(size_t), sizeof(size_t));
                numbersToRead.emplace_back(tmp);
            }
            curPos += numNumbersToRead * sizeof(size_t);
        }
        return make_pair(std::move(charsToRead), std::move(numbersToRead));
    }
};

class SequentialByteEncoder {
    /*
     * Encoding:
     * (|Number Chars : size_t | Number Strings: size_t | [char : unsigned char]
     * | [number : size_t] |)*
     *
     */
public:
    static std::string getName() { return "SequentialByteEncoder"; }
    size_t computeNumberOfSendBytes(size_t charsToSend, size_t numbersToSend) const {
        return charsToSend + sizeof(size_t) * numbersToSend + 2 * sizeof(size_t);
    }

    size_t computeNumberOfSendBytes(
        std::vector<size_t> const& charsToSend, std::vector<size_t> const& numbersToSend
    ) const {
        assert(charsToSend.size() == numbersToSend.size());
        size_t numberOfSendBytes = 0;
        for (size_t i = 0; i < charsToSend.size(); ++i) {
            numberOfSendBytes += computeNumberOfSendBytes(charsToSend[i], numbersToSend[i]);
        }
        return numberOfSendBytes;
    }

    std::pair<size_t, size_t>
    computeNumberOfRecvData(char unsigned const* buffer, size_t size) const {
        size_t recvChars = 0, recvNumbers = 0;
        size_t i = 0;
        while (i < size) {
            size_t curRecvChars = 0;
            size_t curRecvNumbers = 0;
            memcpy(&curRecvChars, buffer + i, sizeof(size_t));
            recvChars += curRecvChars;
            i += sizeof(size_t);
            memcpy(&curRecvNumbers, buffer + i, sizeof(size_t));
            recvNumbers += curRecvNumbers;
            i += sizeof(size_t);
            i += curRecvChars + sizeof(size_t) * curRecvNumbers;
        }
        return std::make_pair(recvChars, recvNumbers);
    }

    // start work here
    template <typename StringSet>
    unsigned char*
    write(unsigned char* buffer, const StringSet ss, size_t const* numbersToWrite) const {
        using String = typename StringSet::String;

        unsigned char* const startOfBuffer = buffer;
        size_t numChars = 0;
        const size_t size = ss.size();

        buffer += sizeof(size_t);
        memcpy(buffer, &size, sizeof(size_t));
        buffer += sizeof(size_t);
        auto beginOfSet = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[beginOfSet + i];
            size_t stringLength = ss.get_length(str) + 1;
            numChars += stringLength;
            memcpy(buffer, ss.get_chars(str, 0), stringLength);
            buffer += stringLength;
        }
        memcpy(startOfBuffer, &numChars, sizeof(size_t));
        memcpy(buffer, numbersToWrite, size * sizeof(size_t));
        buffer += size * sizeof(size_t);

        return buffer;
    }

    std::pair<std::vector<unsigned char>, std::vector<size_t>>
    read(unsigned char* buffer, size_t size) const {
        auto recvData = computeNumberOfRecvData(buffer, size);
        size_t numCharsToRead = recvData.first;
        size_t numNumbersToRead = recvData.second;
        std::vector<unsigned char> charsToRead;
        std::vector<size_t> numbersToRead;
        charsToRead.reserve(numCharsToRead);
        numbersToRead.reserve(numNumbersToRead);

        size_t curPos = 0;
        while (curPos < size) {
            memcpy(&numCharsToRead, buffer + curPos, sizeof(size_t));
            curPos += sizeof(size_t);
            memcpy(&numNumbersToRead, buffer + curPos, sizeof(size_t));
            curPos += sizeof(size_t);

            for (size_t i = 0; i < numCharsToRead; ++i)
                charsToRead.emplace_back(*(buffer + curPos + i));
            curPos += numCharsToRead;
            for (size_t i = 0; i < numNumbersToRead; ++i) {
                size_t tmp;
                std::memcpy(&tmp, buffer + curPos + i * sizeof(size_t), sizeof(size_t));
                numbersToRead.emplace_back(tmp);
            }
            curPos += numNumbersToRead * sizeof(size_t);
        }
        return make_pair(std::move(charsToRead), std::move(numbersToRead));
    }
};

class NoLcps {
public:
    static std::string getName() { return "NoLcps"; }
};

class InterleavedByteEncoder {
    /*
     * Encoding:
     * (|Number Chars : size_t | Number Strings : size_t | (char : unsigned char
     * | number : size_t)^NumberStrings )*
     */
public:
    static std::string getName() { return "InterleavedByteEncoder"; }
    size_t computeNumberOfSendBytes(size_t charsToSend, size_t numbersToSend) const {
        return charsToSend + sizeof(size_t) * numbersToSend + 2 * sizeof(size_t);
    }

    size_t computeNumberOfSendBytes(
        std::vector<size_t> const& charsToSend, std::vector<size_t> const& numbersToSend
    ) const {
        assert(charsToSend.size() == numbersToSend.size());
        size_t numberOfSendBytes = 0;
        for (size_t i = 0; i < charsToSend.size(); ++i) {
            numberOfSendBytes += computeNumberOfSendBytes(charsToSend[i], numbersToSend[i]);
        }
        return numberOfSendBytes;
    }

    std::pair<size_t, size_t>
    computeNumberOfRecvData(char unsigned const* buffer, size_t size) const {
        size_t recvChars = 0, recvNumbers = 0;
        size_t i = 0;
        while (i < size) {
            size_t curRecvChars = 0;
            size_t curRecvNumbers = 0;
            memcpy(&curRecvChars, buffer + i, sizeof(size_t));
            recvChars += curRecvChars;
            i += sizeof(size_t);
            memcpy(&curRecvNumbers, buffer + i, sizeof(size_t));
            recvNumbers += curRecvNumbers;
            i += sizeof(size_t);
            i += curRecvChars + sizeof(size_t) * curRecvNumbers;
        }
        return std::make_pair(recvChars, recvNumbers);
    }

    template <typename StringSet>
    unsigned char*
    write(unsigned char* buffer, const StringSet ss, size_t const* numbersToWrite) const {
        using String = typename StringSet::String;

        unsigned char* const startOfBuffer = buffer;
        size_t numChars = 0;
        const size_t size = ss.size();

        buffer += sizeof(size_t);
        memcpy(buffer, &size, sizeof(size_t));
        buffer += sizeof(size_t);
        auto beginOfSet = ss.begin();
        for (size_t i = 0; i < size; ++i) {
            String str = ss[beginOfSet + i];
            size_t stringLength = ss.get_length(str) + 1;
            numChars += stringLength;
            memcpy(buffer, ss.get_chars(str, 0), stringLength);
            buffer += stringLength;
            memcpy(buffer, numbersToWrite + i, sizeof(size_t));
            buffer += sizeof(size_t);
        }
        memcpy(startOfBuffer, &numChars, sizeof(size_t));

        return buffer;
    }

    std::pair<std::vector<unsigned char>, std::vector<size_t>>
    read(unsigned char* buffer, size_t size) const {
        auto const [numTotalCharsToRead, numTotalNumbersToRead] =
            computeNumberOfRecvData(buffer, size);
        std::vector<unsigned char> charsToRead;
        std::vector<size_t> numbersToRead;
        charsToRead.reserve(numTotalCharsToRead);
        numbersToRead.reserve(numTotalNumbersToRead);

        size_t curPos = 0;
        while (curPos < size) { // loop over all interval (= pieces received from other PEs)
            size_t numCharsToRead = 0, numNumbersToRead = 0;
            memcpy(&numCharsToRead, buffer + curPos, sizeof(size_t));
            curPos += sizeof(size_t);
            memcpy(&numNumbersToRead, buffer + curPos, sizeof(size_t));
            curPos += sizeof(size_t);

            for (size_t i = 0; i < numNumbersToRead; ++i) {
                while (true) {
                    unsigned char const curChar = *(buffer + curPos);
                    ++curPos;
                    charsToRead.emplace_back(curChar);
                    if (curChar == 0)
                        break;
                }
                size_t curNumber = 0;
                memcpy(&curNumber, buffer + curPos, sizeof(size_t));
                curPos += sizeof(size_t);
                numbersToRead.emplace_back(curNumber);
            }
        }
        return make_pair(std::move(charsToRead), std::move(numbersToRead));
    }
};
} // namespace dss_schimek
