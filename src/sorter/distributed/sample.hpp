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
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>

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

    constexpr uint64_t get_number_splitters(uint64_t local_size) const noexcept {
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

class NumStringsPolicy {
public:
    static constexpr bool isIndexed = false;
    static constexpr std::string_view getName() { return "NumStrings"; }

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, isIndexed>;

    template <typename StringSet, typename Params>
    static Result<StringSet>
    sample_splitters(StringSet const& ss, Params const& params, Communicator const& comm) {
        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(ss.size());
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);

        Result<StringSet> result;

        auto& raw_splitters = result.sample;
        raw_splitters.reserve(nr_splitters * (100 + 1u));

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const size_t splitter_index = i * static_cast<size_t>(splitter_dist);
            auto const splitter = ss[ss.begin() + splitter_index];
            const size_t splitter_len = get_splitter_len(ss, splitter, splitter_index, params);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }
        return result;
    }
};

class IndexedNumStringsPolicy {
public:
    static constexpr bool isIndexed = true;
    static constexpr std::string_view getName() { return "IndexedNumStrings"; }

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, isIndexed>;

    template <typename StringSet, typename Params>
    static Result<StringSet>
    sample_splitters(StringSet const& ss, Params const& params, Communicator const& comm) {
        size_t const local_offset = get_local_offset(ss.size(), comm);
        size_t const local_num_strings = ss.size();
        size_t const nr_splitters = params.get_number_splitters(local_num_strings);
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);

        Result<StringSet> sample_indices;
        auto& [raw_splitters, splitter_idxs] = sample_indices;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));
        splitter_idxs.resize(nr_splitters, local_offset);

        // todo this could be improved
        for (size_t i = 1; i <= nr_splitters; ++i) {
            size_t const splitter_index = static_cast<size_t>(i * splitter_dist);
            splitter_idxs[i - 1] += splitter_index;
            auto const splitter = ss[ss.begin() + splitter_index];
            size_t const splitter_len = get_splitter_len(ss, splitter, splitter_index, params);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }

        return sample_indices;
    }
};

class NumCharsPolicy {
public:
    static constexpr bool isIndexed = false;
    static constexpr std::string_view getName() { return "NumChars"; }

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, isIndexed>;

    template <typename StringSet, typename Params>
    static Result<StringSet>
    sample_splitters(StringSet const& ss, Params const& params, Communicator const& comm) {
        size_t const num_chars = get_num_chars(ss, params);
        size_t const local_num_strings = ss.size();
        size_t const nr_splitters = params.get_number_splitters(local_num_strings);
        size_t const splitter_dist = num_chars / (nr_splitters + 1);

        Result<StringSet> result;

        auto& raw_splitters = result.sample;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1));
        raw_splitters.reserve(nr_splitters * (100 + 1));

        size_t string_index = 0;
        for (size_t i = 1; i <= nr_splitters && string_index < local_num_strings; ++i) {
            size_t num_chars_seen = 0;
            while (num_chars_seen < splitter_dist && string_index < local_num_strings) {
                num_chars_seen += ss.get_length(ss[ss.begin() + string_index]);
                ++string_index;
            }

            auto const splitter = ss[ss.begin() + string_index - 1];
            const size_t splitter_len = get_splitter_len(ss, splitter, string_index - 1, params);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }
        return result;
    }
};

class IndexedNumCharsPolicy {
public:
    static constexpr bool isIndexed = true;
    static constexpr std::string_view getName() { return "IndexedNumChars"; }

    template <typename StringSet>
    using Result = SampleResult<typename StringSet::Char, isIndexed>;

    template <typename StringSet, typename Params>
    static Result<StringSet>
    sample_splitters(StringSet const& ss, Params const& params, Communicator const& comm) {
        const size_t local_offset = get_local_offset(ss.size(), comm);
        const size_t num_chars = get_num_chars(ss, params);
        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        const size_t splitter_dist = num_chars / (nr_splitters + 1);

        Result<StringSet> sample_indices;
        auto& [raw_splitters, splitter_idxs] = sample_indices;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));
        splitter_idxs.resize(nr_splitters, local_offset);

        size_t i = 1;
        for (size_t string_idx = 0; i <= nr_splitters && string_idx < local_num_strings; ++i) {
            size_t num_chars_seen = 0;
            while (num_chars_seen < splitter_dist && string_idx < local_num_strings) {
                num_chars_seen += ss.get_length(ss[ss.begin() + string_idx]);
                ++string_idx;
            }
            splitter_idxs[i - 1] += string_idx - 1;

            auto const splitter = ss[ss.begin() + string_idx - 1];
            const size_t splitter_len = get_splitter_len(ss, splitter, string_idx - 1, params);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }
        if (i < nr_splitters) {
            splitter_idxs.resize(i);
        }
        return sample_indices;
    }
};

} // namespace sample
} // namespace dss_mehnert
