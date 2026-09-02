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
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>

#include "dss/mpi/communicator.hpp"

namespace dss_mehnert {
namespace sample {

// Sizes are measured in strings for string-based and in characters for character-based
// sampling.
struct SampleParams {
    size_t local_size;
    size_t max_num_samples;  // a sample is a string, so there can be no more than ss.size()
    double sample_distance;  // the elements controlled by a single sample
    size_t seed;             // PE-dependent seed, used by the randomized samplers
};

// Configuration of the samplers below; implicitly convertible from a plain sampling factor,
// which is all the string-based sampler needs.
struct SamplingConfig {
    size_t sampling_factor = 2;
    // Character-based sampling only: replace each sampled string by one of its two neighbors
    // (fair coin flip). The sampler picks the string that a sampled character position falls
    // into, which is biased towards the longest strings and makes sorting the sample expensive.
    // A neighbor covers about as many characters of the input but is picked by position rather
    // than by length, so it tends to be shorter. Purely a heuristic; nothing guarantees it.
    bool shift_to_neighbor = false;

    SamplingConfig() = default;

    SamplingConfig(size_t const sampling_factor) : sampling_factor{sampling_factor} {}

    SamplingConfig(size_t const sampling_factor, bool const shift_to_neighbor)
        : sampling_factor{sampling_factor},
          shift_to_neighbor{shift_to_neighbor} {}
};

// Deterministic sampling: on a balanced input this amounts to the sampling_factor *
// (num_partitions - 1) samples per PE that regular sampling classically draws.
inline size_t get_total_num_samples(
    size_t const num_partitions, size_t const sampling_factor, size_t const num_processes
) {
    if (num_partitions < 2) {
        return 0; // nothing to split
    }
    return num_processes * std::max<size_t>(1, sampling_factor) * (num_partitions - 1);
}

// Randomized sampling: sampling_factor * log2(P) samples per PE on a balanced input. P is the
// world size, not the size of the communicator being partitioned, because the bucket size
// guarantee holds with high probability in P and the groups get small on the lower levels.
inline size_t
get_total_num_random_samples(size_t const sampling_factor, size_t const num_processes) {
    double const log_p = std::log2(std::max(2.0, static_cast<double>(kamping::world_size())));
    double const num_samples =
        static_cast<double>(num_processes * std::max<size_t>(1, sampling_factor)) * log_p;
    return static_cast<size_t>(std::ceil(num_samples));
}

// omega = |S| / (total number of samples)
inline double get_sample_distance(size_t const global_size, size_t const total_num_samples) {
    if (total_num_samples == 0) {
        return 0.0;
    }
    return static_cast<double>(global_size) / static_cast<double>(total_num_samples);
}

// The number of samples drawn by this PE: one per omega strings/characters that it holds. If
// more samples than strings are requested (omega < 1, or omega < avg. string length for
// character-based sampling), the count is capped: the PE can not contribute more distinct
// samples than it holds strings.
inline size_t get_num_samples(SampleParams const& params) {
    if (params.local_size == 0 || !(params.sample_distance > 0.0)) {
        return 0; // nothing to sample from on this PE
    }
    double const num_samples = static_cast<double>(params.local_size) / params.sample_distance;
    return std::min(params.max_num_samples, static_cast<size_t>(std::llround(num_samples)));
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
    // upper bound on the length of a splitter drawn from this input; only the sample is
    // truncated, the string exchange still uses the full distinguishing prefixes
    size_t max_length = std::numeric_limits<size_t>::max();
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
    return std::min(arg.max_length, arg.prefixes[index]);
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

// The characters seen by the character-based sampler; uses the same (possibly truncated) lengths
// as its consumer loop. With full lengths the boundaries are spread over more characters than the
// loop can traverse and the trailing samples are lost.
template <typename StringSet, typename ExtraArg>
size_t accumulate_sample_chars(StringSet const& ss, ExtraArg const arg) {
    size_t num_chars = 0, index = 0;
    for (auto it = ss.begin(); it != ss.end(); ++it, ++index) {
        num_chars += get_string_len(ss, ss[it], index, arg);
    }
    return num_chars;
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

// The distance between two samples of this PE. This is omega, unless get_num_samples had to
// cap the number of samples at the local size: the samples must still cover the whole local
// input, so the distance is derived from the number of samples actually drawn.
inline double get_local_sample_distance(SampleParams const& params, size_t const num_samples) {
    if (num_samples == 0) {
        return 0.0;
    }
    return static_cast<double>(params.local_size) / static_cast<double>(num_samples);
}

// Shift the samples left by half of the sample distance.
// A technique proposed by Claude to reduce the overloading of the first and last bucket - however,
// this does not change the worst case bounds.
inline size_t
get_sample_position(size_t const index, size_t const local_size, double const sample_distance) {
    if (local_size == 0) {
        return 0; // no samples are drawn on an empty PE; guards the local_size - 1 below
    }
    double const position = (static_cast<double>(index) + 0.5) * sample_distance;
    return std::min(local_size - 1, static_cast<size_t>(position));
}

// Picks one of the two neighbors of a string, see SamplingConfig::shift_to_neighbor.
class NeighborShift {
public:
    explicit NeighborShift(size_t const seed) : gen_{seed} {}

    // a local string index adjacent to `index`, clamped to the local input
    size_t operator()(size_t const index, size_t const local_size) {
        if (local_size < 2) {
            return index;
        }
        bool const forward = index == 0 || (index + 1 < local_size && coin_(gen_));
        return forward ? index + 1 : index - 1;
    }

private:
    std::mt19937_64 gen_;
    std::bernoulli_distribution coin_{0.5};
};

template <bool is_random>
class StringIndexSampler;

template <>
class StringIndexSampler<false> {
public:
    StringIndexSampler() = delete;

    explicit StringIndexSampler(SampleParams const& params)
        : local_size_{params.local_size},
          num_samples_{get_num_samples(params)},
          sample_distance_{get_local_sample_distance(params, num_samples_)} {}

    size_t size() const { return num_samples_; }

    // a local string index
    size_t get_sample(size_t const index) {
        return get_sample_position(index, local_size_, sample_distance_);
    }

private:
    size_t local_size_;
    size_t num_samples_;
    double sample_distance_;
};

template <>
class StringIndexSampler<true> {
public:
    StringIndexSampler() = delete;

    // the generator is seeded with the PE's rank so each PE draws an independent
    // sample rather than the same sequence everywhere
    explicit StringIndexSampler(SampleParams const& params)
        : num_samples_{get_num_samples(params)},
          gen_{params.seed},
          dist_{0, std::max<size_t>(1, params.local_size) - 1} {}

    size_t size() const { return num_samples_; }

    size_t get_sample(size_t) { return dist_(gen_); }

private:
    size_t num_samples_;
    std::mt19937_64 gen_;
    std::uniform_int_distribution<size_t> dist_;
};

template <bool is_random>
class CharIndexSampler;

template <>
class CharIndexSampler<false> {
public:
    CharIndexSampler() = delete;

    explicit CharIndexSampler(SampleParams const& params)
        : local_size_{params.local_size},
          num_samples_{get_num_samples(params)},
          sample_distance_{get_local_sample_distance(params, num_samples_)},
          current_{0} {}

    size_t size() const { return num_samples_; }

    // boundaries are 1-based local character positions, see CharIndexSampler<true>
    size_t next() {
        return get_sample_position(current_++, local_size_, sample_distance_) + 1;
    }

private:
    size_t local_size_;
    size_t num_samples_;
    double sample_distance_;
    size_t current_;
};

template <>
class CharIndexSampler<true> {
public:
    CharIndexSampler() = delete;

    // the generator is seeded with the PE's rank so each PE draws an independent
    // sample rather than the same sequence everywhere
    explicit CharIndexSampler(SampleParams const& params)
        : sample_(get_num_samples(params)),
          current_{0} {
        // boundaries are drawn in [1, num_chars] (a 1-based character position):
        // a boundary of 0 would leave `string` at ss.begin() in the consumer loop
        // and make it read ss[string - 1] (out of bounds), so 0 must be excluded.
        std::mt19937_64 gen{params.seed};
        std::uniform_int_distribution<size_t> dist{1, std::max<size_t>(1, params.local_size)};
        std::generate(sample_.begin(), sample_.end(), [&] { return dist(gen); });
        ips4o::sort(sample_.begin(), sample_.end());
    }

    size_t size() const { return sample_.size(); }

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

    StringBasedSampling() = default;

    explicit StringBasedSampling(SamplingConfig const config) : config_{config} {}

    template <typename StringSet, typename ExtraArg>
    Result<StringSet> sample_splitters(
        StringSet const& ss,
        size_t const num_partitions,
        ExtraArg const arg,
        Communicator const& comm
    ) const {
        // n = global number of strings
        size_t const global_strings =
            comm.allreduce_single(kamping::send_buf(ss.size()), kamping::op(std::plus<>{}));
        size_t const total_num_samples =
            is_random ? get_total_num_random_samples(config_.sampling_factor, comm.size())
                      : get_total_num_samples(num_partitions, config_.sampling_factor, comm.size());

        SampleParams const params{
            .local_size = ss.size(),
            .max_num_samples = ss.size(),
            .sample_distance = get_sample_distance(global_strings, total_num_samples),
            .seed = comm.rank(),
        };

        _internal::StringIndexSampler<is_random> sampler{params};
        size_t const sample_size = sampler.size();

        Result<StringSet> result;
        result.sample.reserve(sample_size * (100 + 1u)); // todo
        if constexpr (is_indexed) {
            result.local_offset = get_local_offset(ss.size(), comm);
            result.indices.resize(sample_size);
        }

        for (size_t i = 0; i < sample_size; ++i) {
            auto const sample_index = sampler.get_sample(i);
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
    SamplingConfig config_;
};

template <bool is_indexed_, bool is_random_>
class CharBasedSampling {
public:
    static constexpr bool is_indexed = is_indexed_;
    static constexpr bool is_random = is_random_;

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, is_indexed>;

    CharBasedSampling() = default;

    explicit CharBasedSampling(SamplingConfig const config) : config_{config} {}

    template <typename StringSet, typename ExtraArg>
    Result<StringSet> sample_splitters(
        StringSet const& ss,
        size_t const num_partitions,
        ExtraArg const arg,
        Communicator const& comm
    ) const {
        // n = global number of characters
        size_t const num_chars = accumulate_sample_chars(ss, arg);
        size_t const global_chars =
            comm.allreduce_single(kamping::send_buf(num_chars), kamping::op(std::plus<>{}));
        size_t const total_num_samples =
            is_random ? get_total_num_random_samples(config_.sampling_factor, comm.size())
                      : get_total_num_samples(num_partitions, config_.sampling_factor, comm.size());

        SampleParams const params{
            .local_size = num_chars,
            .max_num_samples = ss.size(),
            .sample_distance = get_sample_distance(global_chars, total_num_samples),
            .seed = comm.rank(),
        };

        _internal::CharIndexSampler<is_random> sampler{params};
        size_t const sample_size = sampler.size();

        Result<StringSet> result;
        result.sample.reserve((num_chars / std::max<size_t>(1, ss.size()) + 1) * sample_size);
        if constexpr (is_indexed) {
            result.local_offset = get_local_offset(ss.size(), comm);
            result.indices.reserve(sample_size);
        }

        // seeded independently of the position sampler above, so that the two draws do not
        // share a random sequence
        _internal::NeighborShift neighbor_shift{params.seed + 0x9e3779b97f4a7c15};

        auto string = ss.begin();
        size_t current_chars = 0, index = 0;
        for (size_t i = 0; i < sample_size && string != ss.end(); ++i) {
            auto const next_boundary = sampler.next();
            for (; current_chars < next_boundary && string != ss.end(); ++string, ++index) {
                current_chars += get_string_len(ss, ss[string], index, arg);
            }

            assert_unequal(string, ss.begin());

            auto const sample_index = config_.shift_to_neighbor
                                          ? neighbor_shift(index - 1, ss.size())
                                          : index - 1;
            auto const& sample = ss.at(sample_index);
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
    SamplingConfig config_;
};

} // namespace sample
} // namespace dss_mehnert
