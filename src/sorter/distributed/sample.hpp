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

#include <ips4o.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>

#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace sample {

inline size_t get_sample_size(
    size_t const local_size, size_t const num_partitions, size_t const sampling_factor
) {
    return std::min(sampling_factor * (num_partitions - 1), local_size);
}

inline size_t get_local_offset(size_t const local_size, Communicator const& comm) {
    return comm.exscan_single(kamping::send_buf(local_size), kamping::op(std::plus<>{}));
}

struct NoExtraArg {};

struct MaxLength {
    size_t max_length;
};

struct DistPrefixes {
    std::span<size_t const> prefixes;
};

template <typename StringSet>
size_t get_string_len(
    StringSet const& ss,
    typename StringSet::String const& str,
    size_t const index,
    NoExtraArg const arg
) {
    return ss.get_length(str);
}

template <typename StringSet>
size_t get_string_len(
    StringSet const& ss,
    typename StringSet::String const& str,
    size_t const index,
    MaxLength const arg
) {
    return std::min(arg.max_length, ss.get_length(str));
}

template <typename StringSet>
size_t get_string_len(
    StringSet const& ss,
    typename StringSet::String const& str,
    size_t const index,
    DistPrefixes const arg
) {
    return arg.prefixes[index];
}

template <typename StringSet, typename ExtraArg>
size_t accumulate_chars(StringSet const& ss, ExtraArg const arg) {
    auto op = [&ss, arg](auto const sum, auto const& str) { return sum + ss.get_length(str); };
    return std::accumulate(ss.begin(), ss.end(), size_t{0}, op);
}

template <typename StringSet>
size_t accumulate_chars(StringSet const& ss, DistPrefixes const arg) {
    return std::accumulate(arg.prefixes.begin(), arg.prefixes.end(), size_t{0});
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
    size_t local_offset;
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

    // todo make stateless
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

template <bool is_indexed_, bool is_random_>
class StringBasedSampling {
public:
    static constexpr bool is_indexed = is_indexed_;
    static constexpr bool is_random = is_random_;

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, is_indexed>;

    explicit StringBasedSampling(size_t const sampling_factor)
        : sampling_factor_{sampling_factor} {}

    template <typename StringSet, typename ExtraArg>
    Result<StringSet> sample_splitters(
        StringSet const& ss,
        size_t const num_partitions,
        ExtraArg const arg,
        Communicator const& comm
    ) const {
        size_t const sample_size = get_sample_size(ss.size(), num_partitions, sampling_factor_);

        _internal::StringIndexSampler<is_random> sampler{ss.size(), sample_size};

        Result<StringSet> result;
        result.sample.reserve(sample_size * (100 + 1u)); // todo
        if constexpr (is_indexed) {
            result.local_offset = get_local_offset(ss.size(), comm);
            result.indices.resize(sample_size);
        }

        for (size_t i = 0; i < sample_size; ++i) {
            auto const sample_index = sampler.get_sample(i + 1); // todo ???? convert to next
            auto const& sample = ss.at(sample_index);
            auto const sample_len = get_string_len(ss, sample, sample_index, arg);
            auto const sample_chars = ss.get_chars(sample, 0);

            auto const begin = sample_chars, end = begin + sample_len;
            result.sample.insert(result.sample.end(), begin, end);
            result.sample.push_back(0);

            if constexpr (is_indexed) {
                result.indices[i] = result.local_offset + sample_index;
            }
        }
        return result;
    }

private:
    size_t sampling_factor_;
};

template <bool is_indexed_, bool is_random_>
class CharBasedSampling {
public:
    static constexpr bool is_indexed = is_indexed_;
    static constexpr bool is_random = is_random_;

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, is_indexed>;

    explicit CharBasedSampling(size_t const sampling_factor) : sampling_factor_{sampling_factor} {}

    template <typename StringSet, typename ExtraArg>
    Result<StringSet> sample_splitters(
        StringSet const& ss,
        size_t const num_partitions,
        ExtraArg const arg,
        Communicator const& comm
    ) const {
        size_t const num_chars = accumulate_chars(ss, arg);
        size_t const sample_size = get_sample_size(ss.size(), num_partitions, sampling_factor_);

        _internal::CharIndexSampler<is_random> sampler{num_chars, sample_size};

        Result<StringSet> result;
        result.sample.reserve((num_chars / std::max<size_t>(1, ss.size()) + 1) * sample_size);
        if constexpr (is_indexed) {
            result.local_offset = get_local_offset(ss.size(), comm);
            result.indices.reserve(sample_size);
        }

        auto string = ss.begin();
        size_t current_chars = 0, index = 0;
        for (size_t i = 0; i < sample_size && string != ss.end(); ++i) {
            auto const next_boundary = sampler.next();
            for (; current_chars < next_boundary && string != ss.end(); ++string, ++index) {
                current_chars += get_string_len(ss, ss[string], index, arg);
            }

            assert_unequal(string, ss.begin());

            auto const& sample = ss[string - 1];
            auto const sample_index = index - 1;
            auto const sample_len = get_string_len(ss, sample, sample_index, arg);
            auto const sample_chars = ss.get_chars(sample, 0);

            auto const begin = sample_chars, end = begin + sample_len;
            result.sample.insert(result.sample.end(), begin, end);
            result.sample.push_back(0);

            if constexpr (is_indexed) {
                result.indices.emplace_back(result.local_offset + sample_index);
            }
        }
        return result;
    }

private:
    size_t sampling_factor_;
};

} // namespace sample
} // namespace dss_mehnert
