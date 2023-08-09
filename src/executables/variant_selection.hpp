#pragma once

#include <cstdlib>

#include "util/random_string_generator.hpp"

struct GeneratedStringsArgs {
    size_t numOfStrings = 0;
    size_t stringLength = 0;
    size_t minStringLength = 0;
    size_t maxStringLength = 0;
    double dToNRatio = 0.5;
    std::string path = "";
};

template <typename StringGenerator, typename StringSet>
StringGenerator getGeneratedStringContainer(GeneratedStringsArgs const& args) {
    if constexpr (std::is_same_v<StringGenerator, dss_schimek::DNRatioGenerator<StringSet>>) {
        return StringGenerator(args.numOfStrings, args.stringLength, args.dToNRatio);
    } else if constexpr (std::is_same_v<StringGenerator, dss_schimek::FileDistributer<StringSet>>) {
        return dss_schimek::FileDistributer<StringSet>(args.path);
    } else if constexpr (std::is_same_v<
                             StringGenerator,
                             dss_schimek::SkewedDNRatioGenerator<StringSet>>) {
        return dss_schimek::SkewedDNRatioGenerator<StringSet>(
            args.numOfStrings,
            args.stringLength,
            args.dToNRatio
        );
    } else if constexpr (std::is_same_v<StringGenerator, dss_schimek::SuffixGenerator<StringSet>>) {
        return dss_schimek::SuffixGenerator<StringSet>(args.path);
    } else {
        return StringGenerator(args.numOfStrings, args.minStringLength, args.maxStringLength);
    }
}

namespace PolicyEnums {

enum class StringGenerator {
    skewedRandomStringLcpContainer = 0,
    DNRatioGenerator = 1,
    File = 2,
    SkewedDNRatioGenerator = 3,
    SuffixGenerator = 4
};

inline StringGenerator getStringGenerator(size_t i) {
    switch (i) {
        case 0:
            return StringGenerator::skewedRandomStringLcpContainer;
        case 1:
            return StringGenerator::DNRatioGenerator;
        case 2:
            return StringGenerator::File;
        case 3:
            return StringGenerator::SkewedDNRatioGenerator;
        case 4:
            return StringGenerator::SuffixGenerator;
        default:
            std::abort();
    }
}

enum class SampleString {
    numStrings = 0,
    numChars = 1,
    indexedNumStrings = 2,
    indexedNumChars = 3
};

inline SampleString getSampleString(size_t i) {
    switch (i) {
        case 0:
            return SampleString::numStrings;
        case 1:
            return SampleString::numChars;
        case 2:
            return SampleString::indexedNumStrings;
        case 3:
            return SampleString::indexedNumChars;
        default:
            std::abort();
    }
}

enum class MPIRoutineAllToAll { small = 0, directMessages = 1, combined = 2 };

inline MPIRoutineAllToAll getMPIRoutineAllToAll(size_t i) {
    switch (i) {
        case 0:
            return MPIRoutineAllToAll::small;
        case 1:
            return MPIRoutineAllToAll::directMessages;
        case 2:
            return MPIRoutineAllToAll::combined;
        default:
            std::abort();
    }
}

enum class Subcommunicators { none = 0, naive = 1, grid = 2 };

inline Subcommunicators getSubcommunicators(size_t i) {
    switch (i) {
        case 0:
            return Subcommunicators::none;
        case 1:
            return Subcommunicators::naive;
        case 2:
            return Subcommunicators::grid;
        default:
            std::abort();
    }
};

struct CombinationKey {
    StringGenerator string_generator;
    SampleString sample_policy;
    MPIRoutineAllToAll alltoall_routine;
    Subcommunicators subcomms;
    bool prefix_compression;
    bool lcp_compression;
    bool prefix_doubling;
    bool golomb_encoding;

    auto operator<=>(CombinationKey const&) const = default;
};

} // namespace PolicyEnums
