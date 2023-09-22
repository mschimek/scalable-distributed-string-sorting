// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>

#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace sample {

template <typename T, typename... Args>
struct is_present {
    static constexpr bool value{(std::is_same_v<T, Args> || ...)};
};

struct MaxLength {
    size_t max_length;
};

struct DistPrefixes {
    std::vector<size_t> const& prefixes;
};

template <typename... Args>
struct SampleParams : public Args... {
    static constexpr bool has_max_length{is_present<MaxLength, Args...>::value};
    static constexpr bool has_dist_prefixes{is_present<DistPrefixes, Args...>::value};

    // todo is this constructor really necessary
    SampleParams(uint64_t num_partitions_, uint64_t sampling_factor_, Args... args)
        : Args(args)...,
          num_partitions(num_partitions_),
          sampling_factor(sampling_factor_) {}

    uint64_t num_partitions;
    uint64_t sampling_factor;

    constexpr uint64_t get_num_splitters(uint64_t local_size) const noexcept {
        return std::min(sampling_factor * (num_partitions - 1), local_size);
    }
};

inline size_t get_local_offset(size_t local_size, Communicator const& comm) {
    return comm.exscan_single(kamping::send_buf(local_size), kamping::op(std::plus<>{}));
}

template <typename StringSet, typename Params>
static size_t get_num_chars(StringSet const& ss, Params const& params) {
    if constexpr (Params::has_dist_prefixes) {
        return std::accumulate(std::begin(params.prefixes), std::end(params.prefixes), size_t{0});
    } else {
        auto op = [&ss](auto sum, auto const& str) { return sum + ss.get_length(str); };
        return std::accumulate(std::begin(ss), std::end(ss), size_t{0}, op);
    }
}

template <typename StringSet, typename Params>
static size_t get_splitter_len(
    StringSet const& ss, typename StringSet::String const& str, size_t index, Params const& params
) {
    size_t splitter_len;
    if constexpr (Params::has_dist_prefixes) {
        splitter_len = params.prefixes[index];
    } else {
        splitter_len = ss.get_length(str);
    }

    if constexpr (Params::has_max_length) {
        return std::min(params.max_length, splitter_len);
    } else {
        return splitter_len;
    }
}

template <typename Char, bool is_indexed>
struct SampleResult;

template <typename Char>
struct SampleResult<Char, false> {
    std::vector<Char> sample;
};

template <typename Char>
struct SampleResult<Char, true> {
    std::vector<Char> sample;
    std::vector<uint64_t> indices;
};

namespace _internal {

template <bool is_random>
class StringIndexSample;

template <>
class StringIndexSample<false> {
public:
    StringIndexSample() = delete;

    StringIndexSample(size_t strings, size_t splitters)
        : splitter_dist_{static_cast<double>(strings) / static_cast<double>(splitters + 1)} {}

    size_t get_splitter(size_t splitter) {
        return static_cast<size_t>(static_cast<double>(splitter) * splitter_dist_);
    }

private:
    double splitter_dist_;
};

template <>
class StringIndexSample<true> {
public:
    StringIndexSample() = delete;

    StringIndexSample(size_t strings, size_t)
        : gen_{}, // todo what to do about seed here?
          dist_{0, strings == 0 ? 0 : strings - 1} {}

    size_t get_splitter(size_t) { return dist_(gen_); }

private:
    std::mt19937_64 gen_;
    std::uniform_int_distribution<size_t> dist_;
};

} // namespace _internal

template <bool is_indexed_ = false, bool is_random_ = false>
class StringBasedSampling {
public:
    static constexpr bool is_indexed = is_indexed_;
    static constexpr bool is_random = is_random_;

    static constexpr std::string_view getName() { return "NumStrings"; }

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, is_indexed>;

    template <typename StringSet, typename Params>
    static Result<StringSet>
    sample_splitters(StringSet const& ss, Params const& params, Communicator const& comm) {
        size_t const num_splitters = params.get_num_splitters(ss.size());
        size_t const local_offset = is_indexed ? get_local_offset(ss.size(), comm) : 0;

        _internal::StringIndexSample<is_random> sample{ss.size(), num_splitters};

        Result<StringSet> result;
        result.sample.reserve(num_splitters * (100 + 1u)); // todo
        if constexpr (is_indexed) {
            result.indices.resize(num_splitters);
        }

        for (size_t i = 0; i < num_splitters; ++i) {
            auto const splitter_index = sample.get_splitter(i + 1);
            auto const splitter = ss[ss.begin() + splitter_index];
            auto const splitter_len = get_splitter_len(ss, splitter, splitter_index, params);
            auto const splitter_chars = ss.get_chars(splitter, 0);

            std::copy_n(splitter_chars, splitter_len, std::back_inserter(result.sample));
            result.sample.push_back(0);

            if constexpr (is_indexed) {
                result.indices[i] = local_offset + splitter_index;
            }
        }
        return result;
    }
};

template <bool is_indexed_ = false, bool is_random_ = false>
class CharBasedSampling {
public:
    static constexpr bool is_indexed = is_indexed_;
    static constexpr bool is_random = is_random_;

    static constexpr std::string_view getName() { return "NumChars"; }

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, is_indexed>;

    template <typename StringSet, typename Params>
    static Result<StringSet>
    sample_splitters(StringSet const& ss, Params const& params, Communicator const& comm) {
        size_t const num_chars = get_num_chars(ss, params);
        size_t const num_splitters = params.get_num_splitters(ss.size());
        size_t const splitter_dist = num_chars / (num_splitters + 1);
        size_t const local_offset = is_indexed ? get_local_offset(ss.size(), comm) : 0;

        Result<StringSet> result;
        // todo better estimate for length of splitters
        result.sample.reserve(num_splitters * (100 + 1));
        if constexpr (is_indexed) {
            result.indices.reserve(num_splitters);
        }

        auto string = ss.begin(), index = 0;
        size_t next_boundary = splitter_dist, current_chars = 0;
        for (size_t i = 0; i < num_splitters && string != ss.end(); ++i) {
            for (; current_chars < next_boundary && string != ss.end(); ++string, ++index) {
                current_chars += get_splitter_len(ss, ss[string], index, params);
            }
            next_boundary += splitter_dist;

            assert_unequal(string, ss.begin());

            auto const splitter = ss[string - 1];
            auto const splitter_index = index - 1;
            auto const splitter_len = get_splitter_len(ss, splitter, splitter_index, params);
            auto const splitter_chars = ss.get_chars(splitter, 0);

            std::copy_n(splitter_chars, splitter_len, std::back_inserter(result.sample));
            result.sample.push_back(0);

            if constexpr (is_indexed) {
                result.indices.emplace_back(local_offset + splitter_index);
            }
        }
        return result;
    }
};

} // namespace sample
} // namespace dss_mehnert
