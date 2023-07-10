// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string_view>
#include <vector>

#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>

#include "mpi/communicator.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace sample {

struct SampleParams {
    uint64_t num_partitions;
    uint64_t sampling_factor;

    constexpr uint64_t get_number_splitters(uint64_t local_size) const noexcept {
        return std::min(sampling_factor * (num_partitions - 1), local_size);
    }
};

inline uint64_t getLocalOffset(uint64_t localStringSize, Communicator const& comm) {
    return comm.exscan_single(
        kamping::send_buf(localStringSize),
        kamping::op(kamping::ops::plus<>{})
    );
}

struct SampleIndices {
    std::vector<unsigned char> sample;
    std::vector<uint64_t> indices;
};

template <typename StringSet>
class SampleSplittersNumStringsPolicy {
public:
    static constexpr bool isIndexed = false;
    static constexpr std::string_view getName() { return "NumStrings"; }

    static std::vector<typename StringSet::Char> sample_splitters(
        StringSet const& ss,
        size_t maxLength,
        SampleParams const& params,
        [[maybe_unused]] Communicator const& comm
    ) {
        using dss_schimek::measurement::MeasuringTool;
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(ss.size());
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);
        std::vector<Char> raw_splitters;
        raw_splitters.reserve(nr_splitters * (100 + 1u));

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const String splitter = ss[ss.begin() + static_cast<size_t>(i * splitter_dist)];
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength = std::min(splitterLength, maxLength);
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        measuringTool.add(nr_splitters, "nrSplittes", false);
        return raw_splitters;
    }

    static std::vector<typename StringSet::Char> sample_splitters(
        StringSet const& ss,
        SampleParams const& params,
        std::vector<size_t> const& dist,
        [[maybe_unused]] Communicator const& comm
    ) {
        using dss_schimek::measurement::MeasuringTool;
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(ss.size());
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);
        std::vector<Char> raw_splitters;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const size_t splitterIndex = i * splitter_dist;
            const size_t maxLength = dist[splitterIndex];
            const String splitter = ss[ss.begin() + static_cast<size_t>(splitterIndex)];
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength =
                splitterLength > (maxLength) ? (maxLength) : splitterLength;
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        measuringTool.add(nr_splitters, "nrSplittes", false);
        return raw_splitters;
    }
};

template <typename StringSet>
class SampleIndexedSplittersNumStringsPolicy {
public:
    static constexpr bool isIndexed = true;
    static constexpr std::string_view getName() { return "IndexedNumStrings"; }

    static SampleIndices sample_splitters(
        StringSet const& ss, size_t maxLength, SampleParams const& params, Communicator const& comm
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        SampleIndices sampleIndices;

        uint64_t localOffset = getLocalOffset(ss.size(), comm);

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);
        std::vector<Char>& raw_splitters = sampleIndices.sample;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));
        std::vector<uint64_t>& splitterIndices = sampleIndices.indices;
        splitterIndices.resize(nr_splitters, localOffset);

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const uint64_t splitterIndex = static_cast<uint64_t>(i * splitter_dist);
            splitterIndices[i - 1] += splitterIndex;
            const String splitter = ss[ss.begin() + splitterIndex];
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength =
                splitterLength > (maxLength) ? (maxLength) : splitterLength;
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        return sampleIndices;
    }

    static SampleIndices sample_splitters(
        StringSet const& ss,
        SampleParams const& params,
        std::vector<uint64_t> const& dist,
        Communicator const& comm
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        SampleIndices sampleIndices;

        uint64_t localOffset = getLocalOffset(ss.size(), comm);

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);
        std::vector<Char>& raw_splitters = sampleIndices.sample;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));
        std::vector<uint64_t>& splitterIndices = sampleIndices.indices;
        splitterIndices.resize(nr_splitters, localOffset);

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const uint64_t splitterIndex = static_cast<uint64_t>(i * splitter_dist);
            const uint64_t maxLength = dist[splitterIndex];
            splitterIndices[i - 1] += splitterIndex;
            const String splitter = ss[ss.begin() + splitterIndex];
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength =
                splitterLength > (maxLength) ? (maxLength) : splitterLength;
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        return sampleIndices;
    }
};

template <typename StringSet>
class SampleIndexedSplittersNumCharsPolicy {
public:
    static constexpr bool isIndexed = true;
    static constexpr std::string_view getName() { return "IndexedNumChars"; }

    static SampleIndices sample_splitters(
        StringSet const& ss,
        const size_t maxLength,
        SampleParams const& params,
        Communicator const& comm
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;

        SampleIndices sampleIndices;
        uint64_t localOffset = getLocalOffset(ss.size(), comm);
        const size_t num_chars = std::accumulate(
            ss.begin(),
            ss.end(),
            static_cast<size_t>(0u),
            [&ss](size_t const& sum, String const& str) { return sum + ss.get_length(str); }
        );

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        const size_t splitter_dist = num_chars / (nr_splitters + 1);

        std::vector<Char>& raw_splitters = sampleIndices.sample;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));
        std::vector<uint64_t>& splitterIndices = sampleIndices.indices;
        splitterIndices.resize(nr_splitters, localOffset);

        size_t string_index = 0;
        size_t i = 1;
        for (; i <= nr_splitters && string_index < local_num_strings; ++i) {
            size_t num_chars_seen = 0;
            while (num_chars_seen < splitter_dist && string_index < local_num_strings) {
                num_chars_seen += ss.get_length(ss[ss.begin() + string_index]);
                ++string_index;
            }
            splitterIndices[i - 1] += string_index - 1;

            const String splitter = ss[ss.begin() + string_index - 1];
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength =
                splitterLength > (maxLength) ? (maxLength) : splitterLength;
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        if (i < nr_splitters)
            splitterIndices.resize(i);
        return sampleIndices;
    }

    static SampleIndices sample_splitters(
        StringSet const& ss,
        SampleParams const& params,
        std::vector<uint64_t> const& dist,
        Communicator const& comm
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;

        SampleIndices sampleIndices;
        uint64_t localOffset = getLocalOffset(ss.size(), comm);
        // const size_t num_chars =
        //     std::accumulate(ss.begin(), ss.end(), static_cast<size_t>(0u),
        //         [&ss](const size_t& sum, const String& str) {
        //             return sum + ss.get_length(str);
        //         });
        const size_t num_distPref =
            std::accumulate(dist.begin(), dist.end(), static_cast<size_t>(0u));

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = params.get_number_splitters(local_num_strings);
        const size_t splitter_dist = num_distPref / (nr_splitters + 1);

        std::vector<Char>& raw_splitters = sampleIndices.sample;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));
        std::vector<uint64_t>& splitterIndices = sampleIndices.indices;
        splitterIndices.resize(nr_splitters, localOffset);

        size_t string_index = 0;
        size_t i = 1;
        for (; i <= nr_splitters && string_index < local_num_strings; ++i) {
            size_t num_chars_seen = 0;
            while (num_chars_seen < splitter_dist && string_index < local_num_strings) {
                num_chars_seen += dist[string_index];
                ++string_index;
            }
            splitterIndices[i - 1] += string_index - 1;

            const String splitter = ss[ss.begin() + string_index - 1];
            const uint64_t maxLength = dist[string_index - 1];
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength =
                splitterLength > (maxLength) ? (maxLength) : splitterLength;
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        if (i < nr_splitters)
            splitterIndices.resize(i);
        return sampleIndices;
    }
};

template <typename StringSet>
class SampleSplittersNumCharsPolicy {
public:
    static constexpr bool isIndexed = false;
    static constexpr std::string_view getName() { return "NumChars"; }

    static std::vector<typename StringSet::Char> sample_splitters(
        StringSet const& ss,
        size_t maxLength,
        SampleParams const& params,
        [[maybe_unused]] Communicator const& comm
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;

        const size_t num_chars = std::accumulate(
            ss.begin(),
            ss.end(),
            static_cast<size_t>(0u),
            [&ss](size_t const& sum, String const& str) { return sum + ss.get_length(str); }
        );

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
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength =
                splitterLength > (maxLength) ? (maxLength) : splitterLength;
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        return raw_splitters;
    }

    static std::vector<typename StringSet::Char> sample_splitters(
        StringSet const& ss,
        SampleParams const& params,
        std::vector<uint64_t> const& dist,
        [[maybe_unused]] Communicator const& comm
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;

        const size_t num_chars = std::accumulate(
            ss.begin(),
            ss.end(),
            static_cast<size_t>(0u),
            [&ss](size_t const& sum, String const& str) { return sum + ss.get_length(str); }
        );

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
            const uint64_t maxLength = dist[string_index - 1];
            const size_t splitterLength = ss.get_length(splitter);
            const size_t usedSplitterLength =
                splitterLength > (maxLength) ? (maxLength) : splitterLength;
            std::copy_n(
                ss.get_chars(splitter, 0),
                usedSplitterLength,
                std::back_inserter(raw_splitters)
            );
            raw_splitters.push_back(0);
        }
        return raw_splitters;
    }
};

} // namespace sample
} // namespace dss_mehnert
