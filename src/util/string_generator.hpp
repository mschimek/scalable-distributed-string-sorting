// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include "mpi/allgather.hpp"
#include "mpi/environment.hpp"
#include "mpi/read_input.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_schimek {

template <typename StringSet>
class FileDistributer : public StringLcpContainer<StringSet> {
public:
    FileDistributer(std::string const& path)
        : StringLcpContainer<StringSet>{distribute_lines(path)} {}

    static std::string getName() { return "FileDistributer"; }
};

template <typename StringSet>
class SuffixGenerator : public StringLcpContainer<StringSet> {
    using String = typename StringSet::String;

private:
    std::vector<unsigned char> readFile(std::string const& path) {
        using dss_schimek::RawStringsLines;
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

    auto distributeSuffixes(std::vector<unsigned char> const& text) {
        dss_schimek::mpi::environment env;

        size_t const textSize = text.size();
        size_t const estimatedTotalCharCount = textSize * (textSize + 1) / 2 + textSize;
        size_t const estimatedCharCount = estimatedTotalCharCount / env.size();
        size_t const globalSeed = 0;
        std::mt19937 randGen(globalSeed);
        std::uniform_int_distribution<size_t> dist(0, env.size() - 1);
        std::vector<unsigned char> rawStrings;
        rawStrings.reserve(estimatedCharCount);

        size_t numGenStrings = 0;
        for (size_t i = 0; i < textSize; ++i) {
            size_t PEIndex = dist(randGen);
            if (PEIndex == env.rank()) {
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
    SuffixGenerator(std::string const& path) {
        std::vector<unsigned char> text = readFile(path);
        auto [rawStrings, genStrings] = distributeSuffixes(text);
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
    DNRatioGenerator(size_t size, size_t stringLength, double DN_ratio)
        : StringLcpContainer<StringSet>{get_raw_strings(size, stringLength, DN_ratio, {})} {
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
        size_t num_strings, size_t len_strings_requested, double DN_ratio, mpi::environment env
    ) {
        size_t const k = std::max(
            len_strings_requested * std::clamp(DN_ratio, 0.0, 1.0),
            std::ceil(std::log(num_strings) / std::log(char_range))
        );
        size_t const len_strings = std::max(len_strings_requested, k);

        std::vector<unsigned char> raw_strings;
        raw_strings.reserve((1.005 * num_strings) * (len_strings + 1) / env.size());

        std::mt19937 gen{get_global_seed(env)};
        std::uniform_int_distribution<size_t> dist{0, env.size() - 1};

        size_t const rand_char = min_char + (gen() % char_range);
        for (size_t i = 0; i < num_strings; ++i) {
            size_t PEIndex = dist(gen);
            if (PEIndex == env.rank()) {
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

    size_t get_global_seed(mpi::environment env) {
        size_t seed = 0;
        if (env.rank() == 0) {
            std::random_device rand_seed;
            seed = rand_seed();
        }
        return mpi::broadcast(seed, env);
    }
};

template <typename StringSet>
class RandomStringLcpContainer : public StringLcpContainer<StringSet> {
    using Char = typename StringSet::Char;

public:
    RandomStringLcpContainer(
        size_t const size, size_t const min_length = 10, size_t const max_length = 20
    ) {
        dss_schimek::mpi::environment env;
        std::vector<Char> random_raw_string_data;
        std::random_device rand_seed;
        std::mt19937 rand_gen(rand_seed());
        std::uniform_int_distribution<Char> char_dis(65, 90);

        size_t effectiveSize = size / env.size();
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
    size_t getSameSeedGlobally(dss_schimek::mpi::environment env) {
        size_t seed = 0;
        if (env.rank() == 0) {
            std::random_device rand_seed;
            seed = rand_seed();
        }
        return dss_schimek::mpi::broadcast(seed, env);
    }
    SkewedRandomStringLcpContainer(
        size_t const size, size_t const min_length = 100, size_t const max_length = 200
    ) {
        std::vector<Char> random_raw_string_data;
        std::random_device rand_seed;
        dss_schimek::mpi::environment env;
        size_t const globalSeed = 0;       // getSameSeedGlobally();
        std::mt19937 rand_gen(globalSeed); // rand_seed());
        std::uniform_int_distribution<Char> small_char_dis(65, 70);
        std::uniform_int_distribution<Char> char_dis(65, 90);

        std::uniform_int_distribution<size_t> dist(0, env.size() - 1);
        std::uniform_int_distribution<size_t> normal_length_dis(min_length, max_length);
        std::uniform_int_distribution<size_t> large_length_dis(min_length + 100, max_length + 100);

        size_t const numLongStrings = size / 4;
        size_t const numSmallStrings = size - numLongStrings;
        std::size_t curChars = 0;

        random_raw_string_data.reserve(size + 1);
        for (size_t i = 0; i < numLongStrings; ++i) {
            size_t const PEIndex = dist(rand_gen);
            // std::cout << "rank: " << env.rank() << " PEIndex: " << PEIndex <<
            // std::endl;
            bool const takeValue = (PEIndex == env.rank());
            size_t length = large_length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j) {
                unsigned char generatedChar = small_char_dis(rand_gen);
                if (takeValue) {
                    random_raw_string_data.push_back(generatedChar);
                }
            }
            if (takeValue) {
                // std::cout << "taken" << std::endl;
                random_raw_string_data.emplace_back(Char(0));
                curChars += length + 1;
            }
        }

        for (size_t i = 0; i < numSmallStrings; ++i) {
            size_t const PEIndex = dist(rand_gen);
            // std::cout << "rank: " << env.rank() << " PEIndex: " << PEIndex <<
            // std::endl;
            bool const takeValue = (PEIndex == env.rank());
            size_t length = normal_length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j) {
                unsigned char const generatedChar = char_dis(rand_gen);
                if (takeValue) {
                    random_raw_string_data.push_back(generatedChar);
                }
            }
            if (takeValue) {
                // std::cout << "taken" << std::endl;
                random_raw_string_data.push_back(Char(0));
                curChars += length + 1;
            }
        }
        //});
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
        size_t numStrings,
        size_t desiredStringLength,
        double dToN,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
    ) {
        size_t const minInternChar = 65;
        size_t const maxInternChar = 90;

        size_t const numberInternChars = maxInternChar - minInternChar + 1;
        size_t const k = std::max(
            desiredStringLength * dToN,
            std::ceil(std::log(numStrings) / std::log(numberInternChars))
        );
        size_t const stringLength = std::max(desiredStringLength, k);
        std::vector<unsigned char> rawStrings; //(numStrings * (stringLength + 1), minInternChar);
        rawStrings.reserve(numStrings * (stringLength + 1) / env.size());

        size_t const globalSeed = 0;
        std::mt19937 randGen(globalSeed);
        size_t const randomChar = minInternChar + (randGen() % numberInternChars);
        std::uniform_int_distribution<size_t> dist(0, env.size() - 1);

        size_t numGenStrings = 0;
        size_t curOffset = 0;
        size_t const longStringMaxIndex = 0.2 * numStrings;
        size_t const longStringLength = stringLength * 3;
        for (size_t i = 0; i < numStrings; ++i) {
            size_t PEIndex = dist(randGen);
            if (PEIndex == env.rank()) {
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

} // namespace dss_schimek

namespace dss_mehnert {

template <typename StringSet>
struct RandomCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    RandomCharGenerator(size_t const num_chars) : std::vector<Char>(num_chars) {
        std::random_device rand_seed;
        std::mt19937 gen(rand_seed());
        std::uniform_int_distribution<Char> dist{65, 90};
        std::generate(this->begin(), this->end(), [&] { return dist(gen); });
    }
};

template <typename StringSet>
struct FileCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    FileCharGenerator(std::string const& path)
        : std::vector<Char>{dss_schimek::distribute_file(path)} {}
};

template <typename StringSet>
struct CompressedSuffixGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedSuffixGenerator(std::vector<Char>& chars, size_t const step = 1) {
        assert(step > 0);

        for (auto it = chars.begin(); it < chars.end(); it += step) {
            this->emplace_back(&*it, Length{static_cast<size_t>(chars.end() - it)});
        }
    }
};

template <typename StringSet>
struct CompressedStringGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedStringGenerator(
        std::vector<Char>& chars, size_t const length, size_t const step = 1
    ) {
        assert(length > 0 && step > 0);

        for (auto it = chars.begin(); it + length <= chars.end(); it += step) {
            this->emplace_back(&*it, Length{length});
        }
    }
};

template <typename StringSet>
struct CompressedDifferenceCoverGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedDifferenceCoverGenerator(std::vector<Char>& chars, size_t const size) {
        // pad with sentinels to ensure that all strings are created
        chars.resize(chars.size() + size - 1, 0);

        for (auto const& k: get_difference_cover(size)) {
            for (auto it = chars.begin() + k; it + size <= chars.size(); it += size) {
                this->emplace_back(&*it, Length{size});
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
            case 32: { return {1, 2, 3, 4, 8, 12, 20}; }
            case 64: { return {1, 2, 3, 6, 15, 17, 35, 43, 60}; }
            // case 128: {
            //     return {};
            // }
            // case 256: {
            //     return {};
            // }
            // case 512: {
            //     return {};
            // }
            // case 1024: {
            //     return {};
            // }
            // case 2048: {
            //     return {};
            // }
            default: {
                tlx_die("no difference cover available for N=" << size);
            }
        }
        // clang-format on
    }
};

} // namespace dss_mehnert
