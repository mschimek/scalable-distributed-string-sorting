#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "mpi/allgather.hpp"
#include "mpi/environment.hpp"
#include "util/measuringTool.hpp"

namespace dss_schimek {

uint64_t getNumberSplitter(uint64_t PESize, uint64_t localSetSize, uint64_t overSamplingFactor) {
    return std::min<uint64_t>(overSamplingFactor * (PESize - 1), localSetSize);
}

template <typename StringSet>
class SampleSplittersNumStringsPolicy {
public:
    static constexpr bool isIndexed = false;
    static std::string getName() { return "NumStrings"; }

    static std::vector<typename StringSet::Char> sample_splitters(
        StringSet const& ss,
        size_t maxLength,
        uint64_t samplingFactor,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
    ) {
        using measurement::MeasuringTool;
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = getNumberSplitter(env.size(), ss.size(), samplingFactor);
        double const splitter_dist =
            static_cast<double>(local_num_strings) / static_cast<double>(nr_splitters + 1);
        std::vector<Char> raw_splitters;
        // raw_splitters.reserve(nr_splitters * (maxLength + 1u));
        raw_splitters.reserve(nr_splitters * (100 + 1u));

        for (size_t i = 1; i <= nr_splitters; ++i) {
            const String splitter = ss[ss.begin() + static_cast<size_t>(i * splitter_dist)];
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

    static std::vector<typename StringSet::Char> sample_splitters(
        StringSet const& ss,
        uint64_t samplingFactor,
        std::vector<size_t> const& dist,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
    ) {
        using measurement::MeasuringTool;
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters = getNumberSplitter(env.size(), ss.size(), samplingFactor);
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

uint64_t getLocalOffset(uint64_t localStringSize) {
    dss_schimek::mpi::environment env;
    auto allOffsets = dss_schimek::mpi::allgather(localStringSize);
    uint64_t localOffset = 0;
    for (size_t i = 0; i < env.rank(); ++i)
        localOffset += allOffsets[i];
    return localOffset;
}

struct SampleIndices {
    std::vector<unsigned char> sample;
    std::vector<uint64_t> indices;
};

template <typename StringSet>
class SampleIndexedSplittersNumStringsPolicy {
public:
    static constexpr bool isIndexed = true;
    static std::string getName() { return "IndexedNumStrings"; }

    static SampleIndices sample_splitters(
        StringSet const& ss,
        size_t maxLength,
        uint64_t samplingFactor,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        SampleIndices sampleIndices;

        uint64_t localOffset = getLocalOffset(ss.size());

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters =
            getNumberSplitter(env.size(), local_num_strings, samplingFactor);
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
        uint64_t samplingFactor,
        std::vector<uint64_t> const& dist,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;
        SampleIndices sampleIndices;

        uint64_t localOffset = getLocalOffset(ss.size());

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters =
            getNumberSplitter(env.size(), local_num_strings, samplingFactor);
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
    static std::string getName() { return "IndexedNumChars"; }

    static SampleIndices sample_splitters(
        StringSet const& ss,
        const size_t maxLength,
        const uint64_t samplingFactor,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;

        SampleIndices sampleIndices;
        uint64_t localOffset = getLocalOffset(ss.size());
        const size_t num_chars = std::accumulate(
            ss.begin(),
            ss.end(),
            static_cast<size_t>(0u),
            [&ss](size_t const& sum, String const& str) { return sum + ss.get_length(str); }
        );

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters =
            getNumberSplitter(env.size(), local_num_strings, samplingFactor);
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
        const uint64_t samplingFactor,
        std::vector<uint64_t> const& dist,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
    ) {
        using Char = typename StringSet::Char;
        using String = typename StringSet::String;

        SampleIndices sampleIndices;
        uint64_t localOffset = getLocalOffset(ss.size());
        // const size_t num_chars =
        //     std::accumulate(ss.begin(), ss.end(), static_cast<size_t>(0u),
        //         [&ss](const size_t& sum, const String& str) {
        //             return sum + ss.get_length(str);
        //         });
        const size_t num_distPref =
            std::accumulate(dist.begin(), dist.end(), static_cast<size_t>(0u));

        const size_t local_num_strings = ss.size();
        const size_t nr_splitters =
            getNumberSplitter(env.size(), local_num_strings, samplingFactor);
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
    static std::string getName() { return "NumChars"; }

    static std::vector<typename StringSet::Char> sample_splitters(
        StringSet const& ss,
        size_t maxLength,
        uint64_t samplingFactor,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
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
        const size_t nr_splitters =
            getNumberSplitter(env.size(), local_num_strings, samplingFactor);
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
        uint64_t samplingFactor,
        std::vector<uint64_t> const& dist,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
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
        const size_t nr_splitters =
            getNumberSplitter(env.size(), local_num_strings, samplingFactor);
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
} // namespace dss_schimek
