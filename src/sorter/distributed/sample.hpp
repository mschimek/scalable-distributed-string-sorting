// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <vector>

#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>

#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace sample {

struct SampleParams {
    uint64_t num_partitions;
    uint64_t sampling_factor;

    constexpr uint64_t get_number_splitters(uint64_t local_size) const noexcept {
        return std::min(sampling_factor * (num_partitions - 1), local_size);
    }
};

inline size_t get_local_offset(size_t local_size, Communicator const& comm) {
    using namespace kamping;
    return comm.exscan_single(send_buf(local_size), op(ops::plus<>{}));
}

template <typename Char>
struct SampleIndices {
    std::vector<Char> sample;
    std::vector<uint64_t> indices;
};

template <typename StringSet, typename Policy>
class Sampler {
public:
    static auto sample_splitters(
        StringSet const& ss, SampleParams const& params, size_t max_length, Communicator const& comm
    ) {
        auto num_chars = [&ss] {
            auto op = [&ss](auto sum, auto const& str) { return sum + ss.get_length(str); };
            return std::accumulate(std::begin(ss), std::end(ss), 0ull, op);
        };
        auto max_len = [=, &ss](auto const& str, size_t idx) {
            return std::min(ss.get_length(str), max_length);
        };
        return Policy::sample_splitters_(ss, params, num_chars, max_len, comm);
    }

    static auto sample_splitters(
        StringSet const& ss,
        SampleParams const& params,
        std::vector<size_t> const& dist,
        Communicator const& comm
    ) {
        auto num_chars = [&dist] {
            return std::accumulate(std::begin(dist), std::end(dist), 0ull);
        };
        auto max_len = [&dist](auto const&, size_t idx) { return dist[idx]; };
        return Policy::sample_splitters_(ss, params, num_chars, max_len, comm);
    }
};

template <typename StringSet>
class NumStringsPolicy : public Sampler<StringSet, NumStringsPolicy<StringSet>> {
    friend class Sampler<StringSet, NumStringsPolicy<StringSet>>;

public:
    static constexpr bool isIndexed = false;
    static constexpr std::string_view getName() { return "NumStrings"; }

private:
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static std::vector<Char> sample_splitters_(
        StringSet const& ss,
        SampleParams const& params,
        auto get_num_chars,
        auto get_splitter_len,
        Communicator const& comm
    ) {
        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(ss.size());
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);

        std::vector<Char> raw_splitters;
        raw_splitters.reserve(nr_splitters * (100 + 1u));

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const size_t splitter_index = i * static_cast<size_t>(splitter_dist);
            const String splitter = ss[ss.begin() + splitter_index];
            const size_t splitter_len = get_splitter_len(splitter, splitter_index);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }
        return raw_splitters;
    }
};

template <typename StringSet>
class IndexedNumStringPolicy : public Sampler<StringSet, IndexedNumStringPolicy<StringSet>> {
    friend class Sampler<StringSet, IndexedNumStringPolicy<StringSet>>;

public:
    static constexpr bool isIndexed = true;
    static constexpr std::string_view getName() { return "IndexedNumStrings"; }

private:
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static SampleIndices<Char> sample_splitters_(
        StringSet const& ss,
        SampleParams const& params,
        auto get_num_chars,
        auto get_splitter_len,
        Communicator const& comm
    ) {
        const size_t local_offset = get_local_offset(ss.size(), comm);
        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);

        SampleIndices<Char> sample_indices;
        auto& [raw_splitters, splitter_idxs] = sample_indices;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));
        splitter_idxs.resize(nr_splitters, local_offset);

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const size_t splitter_index = static_cast<size_t>(i * splitter_dist);
            splitter_idxs[i - 1] += splitter_index;
            const String splitter = ss[ss.begin() + splitter_index];
            const size_t splitter_len = get_splitter_len(splitter, splitter_index);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }
        return sample_indices;
    }
};

template <typename StringSet>
class NumCharsPolicy : public Sampler<StringSet, NumCharsPolicy<StringSet>> {
    friend class Sampler<StringSet, NumCharsPolicy<StringSet>>;

public:
    static constexpr bool isIndexed = false;
    static constexpr std::string_view getName() { return "NumChars"; }

private:
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static std::vector<Char> sample_splitters_(
        StringSet const& ss,
        SampleParams const& params,
        auto get_num_chars,
        auto get_splitter_len,
        Communicator const& comm
    ) {
        const size_t num_chars = get_num_chars();
        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        const size_t splitter_dist = num_chars / (nr_splitters + 1);

        std::vector<Char> raw_splitters;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1));
        raw_splitters.reserve(nr_splitters * (100 + 1));

        size_t string_index = 0;
        for (size_t i = 1; i <= nr_splitters && string_index < local_num_strings; ++i) {
            size_t num_chars_seen = 0;
            while (num_chars_seen < splitter_dist && string_index < local_num_strings) {
                num_chars_seen += ss.get_length(ss[ss.begin() + string_index]);
                ++string_index;
            }

            const String splitter = ss[ss.begin() + string_index - 1];
            const size_t splitter_len = get_splitter_len(splitter, string_index - 1);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }
        return raw_splitters;
    }
};

template <typename StringSet>
class IndexedNumCharsPolicy : public Sampler<StringSet, IndexedNumCharsPolicy<StringSet>> {
    friend class Sampler<StringSet, IndexedNumCharsPolicy<StringSet>>;

public:
    static constexpr bool isIndexed = true;
    static constexpr std::string_view getName() { return "IndexedNumChars"; }

private:
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static SampleIndices<Char> sample_splitters_(
        StringSet const& ss,
        SampleParams const& params,
        auto get_num_chars,
        auto get_splitter_len,
        Communicator const& comm
    ) {
        const size_t local_offset = get_local_offset(ss.size(), comm);
        const size_t num_chars = get_num_chars();
        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        const size_t splitter_dist = num_chars / (nr_splitters + 1);

        SampleIndices<Char> sample_indices;
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

            const String splitter = ss[ss.begin() + string_idx - 1];
            const size_t splitter_len = get_splitter_len(splitter, string_idx - 1);
            std::copy_n(ss.get_chars(splitter, 0), splitter_len, std::back_inserter(raw_splitters));
            raw_splitters.push_back(0);
        }
        if (i < nr_splitters)
            splitter_idxs.resize(i);
        return sample_indices;
    }
};

} // namespace sample
} // namespace dss_mehnert
