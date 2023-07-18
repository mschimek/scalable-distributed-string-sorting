#pragma once

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

enum class GolombEncoding {
    noGolombEncoding = 0,
    sequentialGolombEncoding = 1,
    pipelinedGolombEncoding = 2
};

inline GolombEncoding getGolombEncoding(size_t i) {
    switch (i) {
        case 0:
            return GolombEncoding::noGolombEncoding;
        case 1:
            return GolombEncoding::sequentialGolombEncoding;
        case 2:
            return GolombEncoding::pipelinedGolombEncoding;
        default:
            std::abort();
    }
}

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

enum class ByteEncoder {
    emptyByteEncoderCopy = 0,
    emptyByteEncoderMemCpy = 1,
    emptyLcpByteEncoderMemCpy = 2,
};

inline ByteEncoder getByteEncoder(size_t i) {
    switch (i) {
        case 0:
            return ByteEncoder::emptyByteEncoderCopy;
        case 1:
            return ByteEncoder::emptyByteEncoderMemCpy;
        case 2:
            return ByteEncoder::emptyLcpByteEncoderMemCpy;
        default:
            std::cout << "Enum ByteEncoder not defined" << std::endl;
            std::abort();
    }
}

struct CombinationKey {
    GolombEncoding golombEncoding_;
    StringGenerator stringGenerator_;
    SampleString sampleStringPolicy_;
    MPIRoutineAllToAll mpiRoutineAllToAll_;
    ByteEncoder byteEncoder_;
    bool compressLcps_;

    auto operator<=>(CombinationKey const&) const = default;
};

} // namespace PolicyEnums
