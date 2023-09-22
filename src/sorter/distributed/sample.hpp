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

#include "ips4o/ips4o.hpp"
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

    // todo rename get_sample_size
    constexpr uint64_t get_num_samples(uint64_t const local_size) const noexcept {
        return std::min(sampling_factor * (num_partitions - 1), local_size);
    }
};

inline size_t get_local_offset(size_t const local_size, Communicator const& comm) {
    return comm.exscan_single(kamping::send_buf(local_size), kamping::op(std::plus<>{}));
}

template <typename StringSet, typename Params>
static size_t accumulate_chars(StringSet const& ss, Params const& params) {
    if constexpr (Params::has_dist_prefixes) {
        return std::accumulate(std::begin(params.prefixes), std::end(params.prefixes), size_t{0});
    } else {
        auto op = [&ss](auto sum, auto const& str) { return sum + ss.get_length(str); };
        return std::accumulate(std::begin(ss), std::end(ss), size_t{0}, op);
    }
}

template <typename StringSet, typename Params>
static size_t get_string_len(
    StringSet const& ss, typename StringSet::String const& str, size_t index, Params const& params
) {
    size_t string_len;
    if constexpr (Params::has_dist_prefixes) {
        string_len = params.prefixes[index];
    } else {
        string_len = ss.get_length(str);
    }

    if constexpr (Params::has_max_length) {
        return std::min(params.max_length, string_len);
    } else {
        return string_len;
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
class StringIndexSampler;

template <>
class StringIndexSampler<false> {
public:
    StringIndexSampler() = delete;

    StringIndexSampler(size_t const strings, size_t const samples)
        : sample_dist_{static_cast<double>(strings) / static_cast<double>(samples + 1)} {}

    size_t get_sample(size_t const index) {
        return static_cast<size_t>(static_cast<double>(index) * sample_dist_);
    }

private:
    double sample_dist_;
};

template <>
class StringIndexSampler<true> {
public:
    StringIndexSampler() = delete;

    StringIndexSampler(size_t const strings, size_t)
        : gen_{}, // todo what to do about seed here?
          dist_{0, std::max<size_t>(1, strings) - 1} {}

    size_t get_sample(size_t) { return dist_(gen_); }

private:
    std::mt19937_64 gen_;
    std::uniform_int_distribution<size_t> dist_;
};

template <bool is_random>
class CharIndexSampler;

template <>
class CharIndexSampler<false> {
public:
    CharIndexSampler() = delete;

    CharIndexSampler(size_t const num_chars, size_t const num_samples)
        : sample_dist_{num_chars / (num_samples + 1)},
          current_boundary_{0} {}

    size_t next() { return current_boundary_ += sample_dist_; }

private:
    // todo this could also use double instead of size_t
    size_t sample_dist_;
    size_t current_boundary_;
};

template <>
class CharIndexSampler<true> {
public:
    CharIndexSampler() = delete;

    CharIndexSampler(size_t const num_chars, size_t const num_samples)
        : sample_(num_samples),
          current_{0} {
        // todo again, what about seed here
        std::mt19937_64 gen;
        std::uniform_int_distribution<size_t> dist{0, std::max<size_t>(1, num_chars) - 1};
        std::generate(sample_.begin(), sample_.end(), [&] { return dist(gen); });
        ips4o::sort(sample_.begin(), sample_.end());
    }

    size_t next() { return sample_[current_++]; }

private:
    std::vector<size_t> sample_;
    size_t current_;
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
        size_t const num_samples = params.get_num_samples(ss.size());
        size_t const local_offset = is_indexed ? get_local_offset(ss.size(), comm) : 0;

        _internal::StringIndexSampler<is_random> sampler{ss.size(), num_samples};

        Result<StringSet> result;
        result.sample.reserve(num_samples * (100 + 1u)); // todo
        if constexpr (is_indexed) {
            result.indices.resize(num_samples);
        }

        for (size_t i = 0; i < num_samples; ++i) {
            auto const sample_index = sampler.get_sample(i + 1); // todo ????
            auto const sample = ss[ss.begin() + sample_index];
            auto const sample_len = get_string_len(ss, sample, sample_index, params);
            auto const sample_chars = ss.get_chars(sample, 0);

            std::copy_n(sample_chars, sample_len, std::back_inserter(result.sample));
            result.sample.push_back(0);

            if constexpr (is_indexed) {
                result.indices[i] = local_offset + sample_index;
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
        size_t const num_chars = accumulate_chars(ss, params);
        size_t const num_samples = params.get_num_samples(ss.size());
        size_t const local_offset = is_indexed ? get_local_offset(ss.size(), comm) : 0;

        _internal::CharIndexSampler<is_random> sampler{num_chars, num_samples};

        Result<StringSet> result;
        // todo better estimate for length of strings
        result.sample.reserve(num_samples * (100 + 1));
        if constexpr (is_indexed) {
            result.indices.reserve(num_samples);
        }

        auto string = ss.begin(), index = 0;
        size_t current_chars = 0;
        for (size_t i = 0; i < num_samples && string != ss.end(); ++i) {
            auto const next_boundary = sampler.next();
            for (; current_chars < next_boundary && string != ss.end(); ++string, ++index) {
                current_chars += get_string_len(ss, ss[string], index, params);
            }

            assert_unequal(string, ss.begin());

            auto const sample = ss[string - 1];
            auto const sample_index = index - 1;
            auto const sample_len = get_string_len(ss, sample, sample_index, params);
            auto const sample_chars = ss.get_chars(sample, 0);

            std::copy_n(sample_chars, sample_len, std::back_inserter(result.sample));
            result.sample.push_back(0);

            if constexpr (is_indexed) {
                result.indices.emplace_back(local_offset + sample_index);
            }
        }
        return result;
    }
};

} // namespace sample
} // namespace dss_mehnert
