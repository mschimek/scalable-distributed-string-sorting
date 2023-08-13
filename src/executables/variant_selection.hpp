#pragma once

#include <cstdlib>
#include <filesystem>

#include <tlx/die/core.hpp>

#include "util/string_generator.hpp"

struct GeneratedStringsArgs {
    size_t num_strings;
    size_t len_strings;
    size_t len_strings_min;
    size_t len_strings_max;
    double DN_ratio = 0.5;
    std::string path = "";
};

template <typename StringGenerator, typename StringSet>
StringGenerator getGeneratedStringContainer(GeneratedStringsArgs const& args) {
    using namespace dss_schimek;

    auto check_path_exists = [](std::string const& path) {
        tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
    };

    if constexpr (std::is_same_v<StringGenerator, DNRatioGenerator<StringSet>>) {
        return StringGenerator(args.num_strings, args.len_strings, args.DN_ratio);
    } else if constexpr (std::is_same_v<StringGenerator, FileDistributer<StringSet>>) {
        check_path_exists(args.path);
        return FileDistributer<StringSet>(args.path);
    } else if constexpr (std::is_same_v<StringGenerator, SkewedDNRatioGenerator<StringSet>>) {
        return SkewedDNRatioGenerator<StringSet>(args.num_strings, args.len_strings, args.DN_ratio);
    } else if constexpr (std::is_same_v<StringGenerator, SuffixGenerator<StringSet>>) {
        check_path_exists(args.path);
        return SuffixGenerator<StringSet>(args.path);
    } else {
        return StringGenerator(args.num_strings, args.len_strings_min, args.len_strings_max);
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
