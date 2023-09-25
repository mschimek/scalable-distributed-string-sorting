#pragma once

#include <cstdlib>
#include <filesystem>

#include <tlx/die/core.hpp>

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

// todo merge with sorterargs
enum class Redistribution { naive = 0, simple_strings = 1, simple_chars = 2 };

inline Redistribution getRedistribution(size_t i) {
    switch (i) {
        case 0:
            return Redistribution::naive;
        case 1:
            return Redistribution::simple_strings;
        case 2:
            return Redistribution::simple_chars;
        default:
            std::abort();
    }
};

} // namespace PolicyEnums
