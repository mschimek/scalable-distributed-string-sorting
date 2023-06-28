// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once
#include <bitset>
#include <iostream>
#include <vector>

#include "delta_stream.hpp"
#include "golomb_bit_stream.hpp"

struct BlockWriter {
    std::vector<size_t> data;

    void PutRaw(const size_t value) { data.push_back(value); }
    size_t block_size() { return sizeof(size_t); };
};

template <typename T>
struct PrintMe;

template <typename OutputIterator>
struct BlockWriterIt {
    using outItValueType = typename OutputIterator::container_type::value_type;
    using type = typename std::enable_if<std::is_same<outItValueType, size_t>::value, size_t>::type;

    OutputIterator outIt;
    BlockWriterIt(OutputIterator outIt) : outIt(outIt) {}

    void PutRaw(const type value) { outIt = value; }
    size_t block_size() { return sizeof(size_t); };
};

template <typename Iterator>
struct BlockReader {
    Iterator curPos;
    Iterator end;
    std::vector<size_t> decoded_data;

    BlockReader(Iterator begin, Iterator end) : curPos(begin), end(end) {}
    bool HasNext() { return curPos != end; }
    template <typename T>
    T GetRaw() {
        return *(curPos++);
    }
};

template <typename T>
void printBits(T const& value) {
    std::cout << std::bitset<sizeof(T) * 8>(value) << std::endl;
}

// see  Moffat, Turpin, Compression And Coding Algorithms page. 40
size_t getB(const size_t maxValue, const size_t numberValues) {
    if (numberValues == 0)
        return 1;
    const size_t avgGap = maxValue / numberValues;
    if (avgGap < 1)
        return 1;
    const size_t k = static_cast<size_t>(std::floor(std::log2(avgGap)));
    size_t exponent = 1u;
    return exponent << k;
}

template <typename InputIterator, typename OutputIterator>
inline void getDeltaEncoding(
    InputIterator begin, InputIterator end, BlockWriterIt<OutputIterator>& blockWriter, size_t b
) {
    using BlockWriter = BlockWriterIt<OutputIterator>;
    using GolombWriter = thrill::core::GolombBitStreamWriter<BlockWriter>;
    using DeltaStreamWriter = thrill::core::DeltaStreamWriter<GolombWriter, size_t>;

    GolombWriter golombWriter(blockWriter, b);
    DeltaStreamWriter deltaStreamWriter(golombWriter);
    InputIterator it = begin;
    while (it != end) {
        deltaStreamWriter.Put(*it);
        ++it;
    }
}

template <typename InputIterator, typename OutputIterator>
inline void getDeltaEncoding(InputIterator begin, InputIterator end, OutputIterator out, size_t b) {
    BlockWriterIt<OutputIterator> blockWriter(out);
    getDeltaEncoding(begin, end, blockWriter, b);
}

// std::vector<size_t> getDecoding(const std::vector<size_t>& values) {
//   using GolombReader = thrill::core::GolombBitStreamReader<BlockReader>;
//
//   BlockReader reader(values.begin(), values.end());
//   std::vector<size_t> decodedValues;
//   const size_t b = 8;
//   GolombReader golombReader(reader, b);
//
//   while(golombReader.HasNext())
//     decodedValues.push_back(golombReader.Next<size_t>());
//
//   return decodedValues;
// }

template <typename InputIterator, typename OutputIterator>
void getDeltaDecoding(
    const InputIterator begin, const InputIterator end, OutputIterator out, size_t b
) {
    using GolombReader = typename thrill::core::GolombBitStreamReader<BlockReader<InputIterator>>;
    using DeltaReader = thrill::core::DeltaStreamReader<GolombReader, size_t>;

    BlockReader reader(begin, end);
    GolombReader golombReader(reader, b);
    DeltaReader deltaReader(golombReader);

    while (golombReader.HasNext()) {
        out = deltaReader.template Next<size_t>();
    }
}
