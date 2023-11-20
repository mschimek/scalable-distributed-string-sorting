// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "encoding/integer_compression.hpp"

namespace dss_mehnert {

class IntegerCompression {
    struct CompressedIntegersCounts {
        std::vector<uint8_t> integers;
        std::vector<size_t> counts;
    };

public:
    template <typename InputIterator>
    static CompressedIntegersCounts
    writeRanges(std::span<size_t const> intervals, InputIterator input_it) {
        size_t const num_integers = std::accumulate(intervals.begin(), intervals.end(), size_t{});

        CompressedIntegersCounts result;
        result.counts.reserve(intervals.size());
        result.integers.resize(num_integers * sizeof(uint64_t));


        for (auto output_it = result.integers.begin(); auto const interval: intervals) {
            size_t const bytes_written = write(input_it, output_it, interval);
            result.counts.push_back(bytes_written);
            output_it += bytes_written;
            input_it += interval;
        }
        return result;
    }

    template <typename InputIterator>
    static std::vector<uint64_t>
    readRanges(std::span<size_t const> intervals, InputIterator input_it) {
        size_t const num_integers = std::accumulate(intervals.begin(), intervals.end(), size_t{0});

        std::vector<uint64_t> output(num_integers);
        for (auto output_it = output.begin(); auto const interval: intervals) {
            input_it += read(input_it, output_it, output_it + interval);
            output_it += interval;
        }
        return output;
    }

private:
    template <typename InputIterator, typename OutputIterator>
    static size_t write(InputIterator input, OutputIterator output, size_t count) {
        dss_schimek::Writer writer(output);
        for (size_t i = 0; i < count; ++i) {
            writer.PutVarint(*(input + i));
        }
        return writer.getNumPutBytes();
    }

    template <typename InputIterator, typename OutputIterator>
    static size_t read(InputIterator first, OutputIterator d_first, OutputIterator d_last) {
        dss_schimek::Reader reader(first, d_first, d_last);
        reader.decode();
        return reader.getNumReadBytes();
    }
};

} // namespace dss_mehnert
