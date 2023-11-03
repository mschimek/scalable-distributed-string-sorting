// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>

#include <kamping/collectives/bcast.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die/core.hpp>
#include <tlx/math/div_ceil.hpp>

#include "mpi/communicator.hpp"
#include "mpi/read_input.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_mehnert {

namespace _internal {

inline size_t get_global_seed(Communicator const& comm) {
    size_t seed = 0;
    if (comm.is_root()) {
        std::random_device rand_seed;
        seed = rand_seed();
    }
    comm.bcast_single(kamping::send_recv_buf(seed));
    return seed;
}

} // namespace _internal


template <typename StringSet>
class FileDistributer : public StringLcpContainer<StringSet> {
public:
    FileDistributer(std::string const& path, Communicator const& comm)
        : StringLcpContainer<StringSet>{distribute_lines(path, 0, comm)} {}

    static std::string getName() { return "FileDistributer"; }
};

template <typename StringSet>
class SuffixGenerator : public StringLcpContainer<StringSet> {
    using String = typename StringSet::String;

private:
    std::vector<unsigned char> readFile(std::string const& path) {
        RawStringsLines data;
        size_t const fileSize = get_file_size(path);
        std::ifstream in(path);
        std::vector<unsigned char>& rawStrings = data.rawStrings;
        rawStrings.reserve(1.5 * fileSize);

        std::string line;
        data.lines = 0u;
        while (std::getline(in, line)) {
            ++data.lines;
            for (unsigned char curChar: line)
                rawStrings.push_back(curChar);
        }
        rawStrings.push_back(0);
        in.close();
        return rawStrings;
    }

    auto distributeSuffixes(std::vector<unsigned char> const& text, Communicator const& comm) {
        size_t const textSize = text.size();
        size_t const estimatedTotalCharCount = textSize * (textSize + 1) / 2 + textSize;
        size_t const estimatedCharCount = estimatedTotalCharCount / comm.size();
        size_t const globalSeed = 0;
        std::mt19937 randGen(globalSeed);
        std::uniform_int_distribution<size_t> dist(0, comm.size() - 1);
        std::vector<unsigned char> rawStrings;
        rawStrings.reserve(estimatedCharCount);

        size_t numGenStrings = 0;
        for (size_t i = 0; i < textSize; ++i) {
            size_t PEIndex = dist(randGen);
            if (PEIndex == comm.rank()) {
                // only create your own strings
                ++numGenStrings;
                std::copy(text.begin() + i, text.end(), std::back_inserter(rawStrings));
                // Assume that text is zero terminated
            }
        }
        rawStrings.shrink_to_fit();
        return std::make_pair(std::move(rawStrings), numGenStrings);
    }

public:
    SuffixGenerator(std::string const& path, Communicator const& comm) {
        std::vector<unsigned char> text = readFile(path);
        auto [rawStrings, genStrings] = distributeSuffixes(text, comm);
        this->update(std::move(rawStrings));
        String* begin = this->strings();
        std::random_device rand;
        std::mt19937 gen{rand()};
        std::shuffle(begin, begin + genStrings, gen);
    }

    static std::string getName() { return "SuffixGenerator"; }
};

template <typename StringSet>
class DNRatioGenerator : public StringLcpContainer<StringSet> {
public:
    DNRatioGenerator(size_t const num_strings, size_t const len_strings, double const DN_ratio)
        : StringLcpContainer<StringSet>{
            get_raw_strings(num_strings, len_strings, DN_ratio, kamping::comm_world())} {
        std::random_device rand;
        std::mt19937 gen{rand()};

        auto& strings = this->get_strings();
        std::shuffle(strings.begin(), strings.end(), gen);
        this->make_contiguous();
    }

    static std::string getName() { return "DNRatioGenerator"; }

private:
    static constexpr unsigned char min_char = 'A';
    static constexpr unsigned char max_char = 'Z';
    static constexpr unsigned char char_range = max_char - min_char + 1;

    std::vector<unsigned char> get_raw_strings(
        size_t const num_strings,
        size_t const len_strings_requested,
        double const DN_ratio,
        Communicator const& comm
    ) {
        size_t const k = std::max(
            len_strings_requested * std::clamp(DN_ratio, 0.0, 1.0),
            std::ceil(std::log(num_strings) / std::log(char_range))
        );
        size_t const len_strings = std::max(len_strings_requested, k);

        std::vector<unsigned char> raw_strings;
        raw_strings.reserve((1.005 * num_strings) * (len_strings + 1) / comm.size());

        std::mt19937 gen{_internal::get_global_seed(comm)};
        std::uniform_int_distribution<size_t> dist{0, comm.size() - 1};

        size_t const rand_char = min_char + (gen() % char_range);
        for (size_t i = 0; i < num_strings; ++i) {
            size_t PEIndex = dist(gen);
            if (PEIndex == comm.rank()) {
                // only create your own strings
                raw_strings.resize(raw_strings.size() + k, min_char);

                size_t current = i;
                for (auto it = raw_strings.rbegin(); it != raw_strings.rbegin() + k; ++it) {
                    if (current == 0)
                        break;
                    *it = min_char + (current % char_range);
                    current /= char_range;
                }

                raw_strings.resize(raw_strings.size() + (len_strings - k), rand_char);
                raw_strings.push_back(0);
            }
        }
        return raw_strings;
    }
};

template <typename StringSet>
class RandomStringLcpContainer : public StringLcpContainer<StringSet> {
    using Char = typename StringSet::Char;

public:
    RandomStringLcpContainer(
        size_t const size, size_t const min_length = 10, size_t const max_length = 20
    ) {
        Communicator comm;
        std::vector<Char> random_raw_string_data;
        std::random_device rand_seed;
        std::mt19937 rand_gen(rand_seed());
        std::uniform_int_distribution<Char> char_dis(65, 90);

        size_t effectiveSize = size / comm.size();
        std::cout << "effective size: " << effectiveSize << std::endl;
        std::uniform_int_distribution<size_t> length_dis(min_length, max_length);
        random_raw_string_data.reserve(effectiveSize + 1);
        for (size_t i = 0; i < effectiveSize; ++i) {
            size_t length = length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j)
                random_raw_string_data.emplace_back(char_dis(rand_gen));
            random_raw_string_data.emplace_back(Char(0));
        }
        this->update(std::move(random_raw_string_data));
    }

    static std::string getName() { return "RandomStringGenerator"; }
};

template <typename StringSet>
class SkewedRandomStringLcpContainer : public StringLcpContainer<StringSet> {
    using Char = typename StringSet::Char;

public:
    SkewedRandomStringLcpContainer(
        size_t const size, size_t const min_length = 100, size_t const max_length = 200
    ) {
        std::vector<Char> random_raw_string_data;
        std::random_device rand_seed;
        Communicator const& comm{kamping::comm_world()};
        std::mt19937 rand_gen(_internal::get_global_seed(comm));
        std::uniform_int_distribution<Char> small_char_dis(65, 70);
        std::uniform_int_distribution<Char> char_dis(65, 90);

        std::uniform_int_distribution<size_t> dist(0, comm.size() - 1);
        std::uniform_int_distribution<size_t> normal_length_dis(min_length, max_length);
        std::uniform_int_distribution<size_t> large_length_dis(min_length + 100, max_length + 100);

        size_t const numLongStrings = size / 4;
        size_t const numSmallStrings = size - numLongStrings;
        std::size_t curChars = 0;

        random_raw_string_data.reserve(size + 1);
        for (size_t i = 0; i < numLongStrings; ++i) {
            size_t const PEIndex = dist(rand_gen);
            bool const takeValue = (PEIndex == comm.rank());
            size_t length = large_length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j) {
                unsigned char generatedChar = small_char_dis(rand_gen);
                if (takeValue) {
                    random_raw_string_data.push_back(generatedChar);
                }
            }
            if (takeValue) {
                random_raw_string_data.emplace_back(Char(0));
                curChars += length + 1;
            }
        }

        for (size_t i = 0; i < numSmallStrings; ++i) {
            size_t const PEIndex = dist(rand_gen);
            bool const takeValue = (PEIndex == comm.rank());
            size_t length = normal_length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j) {
                unsigned char const generatedChar = char_dis(rand_gen);
                if (takeValue) {
                    random_raw_string_data.push_back(generatedChar);
                }
            }
            if (takeValue) {
                random_raw_string_data.push_back(Char(0));
                curChars += length + 1;
            }
        }
        random_raw_string_data.resize(curChars);
        this->update(std::move(random_raw_string_data));
    }

    static std::string getName() { return "SkewedStringGenerator"; }
};

template <typename StringSet>
class SkewedDNRatioGenerator : public StringLcpContainer<StringSet> {
    using String = typename StringSet::String;
    std::tuple<std::vector<unsigned char>, size_t, size_t>

    getRawStringsTimoStyle(
        size_t numStrings, size_t desiredStringLength, double dToN, Communicator const& comm = {}
    ) {
        size_t const minInternChar = 65;
        size_t const maxInternChar = 90;

        size_t const numberInternChars = maxInternChar - minInternChar + 1;
        size_t const k = std::max(
            desiredStringLength * dToN,
            std::ceil(std::log(numStrings) / std::log(numberInternChars))
        );
        size_t const stringLength = std::max(desiredStringLength, k);
        std::vector<unsigned char> rawStrings;
        rawStrings.reserve(numStrings * (stringLength + 1) / comm.size());

        size_t const globalSeed = 0;
        std::mt19937 randGen(globalSeed);
        size_t const randomChar = minInternChar + (randGen() % numberInternChars);
        std::uniform_int_distribution<size_t> dist(0, comm.size() - 1);

        size_t numGenStrings = 0;
        size_t curOffset = 0;
        size_t const longStringMaxIndex = 0.2 * numStrings;
        size_t const longStringLength = stringLength * 3;
        for (size_t i = 0; i < numStrings; ++i) {
            size_t PEIndex = dist(randGen);
            if (PEIndex == comm.rank()) {
                // only create your own strings
                ++numGenStrings;
                size_t curIndex = i;
                for (size_t j = 0; j < k; ++j) {
                    rawStrings.push_back(minInternChar);
                }
                for (size_t j = 0; j < k; ++j) {
                    if (curIndex == 0)
                        break;
                    rawStrings[curOffset + k - 1 - j] =
                        minInternChar + (curIndex % numberInternChars);
                    curIndex /= numberInternChars;
                }
                for (size_t j = k; j < stringLength; ++j)
                    rawStrings.push_back(randomChar);
                if (i < longStringMaxIndex) {
                    for (size_t j = k; j < longStringLength + k; ++j)
                        rawStrings.push_back(randomChar);
                    curOffset += longStringLength;
                }
                rawStrings.push_back(0);
                curOffset += stringLength + 1;
            }
        }
        rawStrings.resize(curOffset);

        return make_tuple(rawStrings, numGenStrings, stringLength);
    }

public:
    SkewedDNRatioGenerator(
        size_t const size, size_t const stringLength = 40, double const dToN = 0.5
    ) {
        size_t genStrings = 0;
        size_t genStringLength = 0;
        std::vector<unsigned char> rawStrings;
        std::tie(rawStrings, genStrings, genStringLength) =
            getRawStringsTimoStyle(size, stringLength, dToN);
        this->update(std::move(rawStrings));
        String* begin = this->strings();
        std::random_device rand;
        std::mt19937 gen(rand());
        std::shuffle(begin, begin + genStrings, gen);
        this->make_contiguous();
    }

    static std::string getName() { return "SkewedDNRatioGenerator"; }
};

template <typename StringSet>
struct RandomCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    RandomCharGenerator(size_t const num_chars) : std::vector<Char>(num_chars) {
        std::random_device rand_seed;
        std::mt19937 gen(rand_seed());
        std::uniform_int_distribution<Char> dist{'A', 'Z'};
        std::generate(this->begin(), this->end(), [&] { return dist(gen); });
    }
};

template <typename StringSet>
struct FileCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    FileCharGenerator(std::string const& path, Communicator const& comm)
        : std::vector<Char>{distribute_file(path, 0, comm)} {
        std::replace(this->begin(), this->end(), static_cast<Char>(0), static_cast<Char>('A'));
    }
};

template <typename StringSet>
struct FileSegmentCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    FileSegmentCharGenerator(
        std::string const& path, size_t const segment_size, Communicator const& comm
    )
        : std::vector<Char>{distribute_file_segments(path, segment_size, false, comm)} {
        std::replace(this->begin(), this->end(), static_cast<Char>(0), static_cast<Char>('A'));
    }
};

template <typename StringSet>
struct CompressedSuffixGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedSuffixGenerator(std::vector<Char>& chars, size_t const step = 1) {
        assert(step > 0);

        for (size_t offset = 0; offset < chars.size(); offset += step) {
            size_t const length = static_cast<size_t>(chars.size() - offset);
            this->emplace_back(chars.data() + offset, length);
        }
    }
};

template <typename StringSet>
struct CompressedWindowGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedWindowGenerator(std::vector<Char>& chars, size_t const length, size_t const step) {
        assert(length > 0 && step > 0);

        for (size_t offset = 0; offset + length <= chars.size(); offset += step) {
            this->emplace_back(chars.data() + offset, length);
        }
    }
};

template <typename StringSet>
struct CompressedDifferenceCoverGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedDifferenceCoverGenerator(
        std::vector<Char>& chars, size_t const size, bool const full_cover, Communicator const& comm
    ) {
        size_t const chars_size = chars.size();
        if (full_cover) {
            shift_chars_left(chars, size, comm);
        }

        auto const difference_cover = get_difference_cover(size);
        this->reserve((chars.size() / size + 1) * difference_cover.size());
        for (auto const& k: difference_cover) {
            for (size_t offset = k; offset < chars_size; offset += size) {
                auto const length = std::min<size_t>(size, chars.size() - offset);
                this->emplace_back(chars.data() + offset, length);
            }
        }
    }

private:
    static std::vector<size_t> get_difference_cover(size_t const size) {
        // clang-format off
        switch (size) {
            case 3:  { return {0, 1}; }
            case 7:  { return {1, 2, 4}; }
            case 13: { return {1, 2, 4, 10}; }
            case 21: { return {1, 2, 5, 15, 17}; }
            case 31: { return {1, 2, 4, 9, 13, 19}; }
            case 32: { return {1, 2, 3, 4, 8, 12, 20}; }
            case 64: { return {1, 2, 3, 6, 15, 17, 35, 43, 60}; }
            // todo case 128: { return {}; }
            // todo case 256: { return {}; } 
            // todo maybe generate these on the fly?
            case 512: {
                return {0, 1, 2, 3, 4, 9, 18, 27, 36, 45, 64, 83, 102, 121, 140, 159, 178, 197, 216,
                    226, 236, 246, 256, 266, 267, 268, 269, 270 };
            }
            case 1024: {
                return {0, 1, 2, 3, 4, 5, 6, 13, 26, 39, 52, 65, 78, 91, 118, 145, 172, 199, 226,
                    253, 280, 307, 334, 361, 388, 415, 442, 456, 470, 484, 498, 512, 526, 540, 541,
                    542, 543, 544, 545, 546};
            }
            case 2048: {
                return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 38, 57, 76, 95, 114, 133, 152, 171, 190,
                    229, 268, 307, 346, 385, 424, 463, 502, 541, 580, 619, 658, 697, 736, 775, 814,
                    853, 892, 931, 951, 971, 991, 1011, 1031, 1051, 1071, 1091, 1111, 1131, 1132,
                    1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140 };
            }
            case 4096: {
                return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 27, 54, 81, 108, 135, 162,
                    189, 216, 243, 270, 297, 324, 351, 378, 433, 488, 543, 598, 653, 708, 763, 818,
                    873, 928, 983, 1038, 1093, 1148, 1203, 1258, 1313, 1368, 1423, 1478, 1533,
                    1588, 1643, 1698, 1753, 1808, 1863, 1891, 1919, 1947, 1975, 2003, 2031, 2059,
                    2087, 2115, 2143, 2171, 2199, 2227, 2255, 2256, 2257, 2258, 2259, 2260, 2261,
                    2262, 2263, 2264, 2265, 2266, 2267, 2268 };
            }
            case 8192: {
                return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 37, 74,
                    111, 148, 185, 222, 259, 296, 333, 370, 407, 444, 481, 518, 555, 592, 629, 666,
                    703, 778, 853, 928, 1003, 1078, 1153, 1228, 1303, 1378, 1453, 1528, 1603, 1678,
                    1753, 1828, 1903, 1978, 2053, 2128, 2203, 2278, 2353, 2428, 2503, 2578, 2653,
                    2728, 2803, 2878, 2953, 3028, 3103, 3178, 3253, 3328, 3403, 3478, 3516, 3554,
                    3592, 3630, 3668, 3706, 3744, 3782, 3820, 3858, 3896, 3934, 3972, 4010, 4048,
                    4086, 4124, 4162, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209,
                    4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218};
            }
            default: {
                tlx_die("no difference cover available for X=" << size);
            }
        }
        // clang-format on
    }

    static void
    shift_chars_left(std::vector<Char>& chars, size_t const size, Communicator const& comm) {
        using namespace kamping;

        // NOTE this only works correctly if ever PE has at least `size` characters
        chars.reserve(chars.size() + size - 1);

        Request req;
        if (comm.rank() > 0) {
            std::span const send_chars{chars.begin(), std::min(size - 1, chars.size())};
            comm.issend(send_buf(send_chars), destination(comm.rank() - 1), request(req));
        }
        if (comm.rank() < comm.size() - 1) {
            std::vector<Char> recv_chars;
            comm.recv(recv_buf(recv_chars));
            chars.insert(chars.end(), recv_chars.begin(), recv_chars.end());
        }
        req.wait();
    }
};

template <typename StringSet>
struct CompressedDNRatioGenerator : public StringLcpContainer<StringSet> {
    using Char = StringSet::Char;

    CompressedDNRatioGenerator(
        double const dn_ratio,
        size_t const len_strings,
        size_t const num_strings,
        Communicator const& comm
    ) {
        tlx_die_verbose_if(dn_ratio > 0.5, "D/N ratios greater than 1/2 are not suppported");

        assert(0 <= dn_ratio && dn_ratio <= 1);
        size_t const strings_per_chunk = std::max<size_t>(1, 2 * len_strings * dn_ratio);
        size_t const chars_per_chunk = len_strings + strings_per_chunk - 1;

        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<Char> dist{'A', 'Z'};

        Char padding_char = dist(gen);
        comm.bcast_single(kamping::send_recv_buf(padding_char));

        auto& raw_strings = *this->raw_strings_;
        auto& strings = this->strings_;

        size_t const num_chunks = tlx::div_ceil(num_strings, strings_per_chunk);
        raw_strings.reserve(num_chunks * chars_per_chunk);

        for (size_t n = 0; n < num_strings; n += strings_per_chunk) {
            raw_strings.resize(raw_strings.size() + chars_per_chunk, padding_char);
            std::generate_n(raw_strings.rbegin(), len_strings, [&] { return dist(gen); });

            auto const chunk_size = std::min<size_t>(strings_per_chunk, num_strings - n);
            auto const begin = raw_strings.end() - chars_per_chunk;
            for (size_t i = 0; i < chunk_size; ++i) {
                strings.emplace_back(&*(begin + i), len_strings);
            }
        }
        this->lcps_.resize(strings.size());
    }
};

} // namespace dss_mehnert
