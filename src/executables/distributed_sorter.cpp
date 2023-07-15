// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>

#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <tlx/cmdline_parser.hpp>
#include <tlx/die.hpp>
#include <tlx/die/core.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/is_sorted.hpp"
#include "mpi/warmup.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "util/measuringTool.hpp"
#include "util/random_string_generator.hpp"
#include "variant_selection.hpp"

template <
    typename StringSet,
    typename StringGenerator,
    typename SamplePolicy,
    typename MPIAllToAllRoutine,
    typename ByteEncoder>
void execute_sorter(
    size_t numOfStrings,
    bool const check,
    bool const check_exhaustive,
    size_t iteration,
    bool const strongScaling,
    GeneratedStringsArgs genStringArgs,
    std::vector<size_t> levels,
    dss_mehnert::Communicator comm = {}
) {
    using namespace dss_mehnert;
    using namespace dss_schimek;
    using namespace dss_schimek::measurement;

    using StringLcpPtr = typename tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    using dss_schimek::mpi::AllToAllStringImpl;
    using AllToAllPolicy = AllToAllStringImpl<true, StringSet, MPIAllToAllRoutine, ByteEncoder>;
    using Sorter = sorter::DistributedMergeSort<StringLcpPtr, AllToAllPolicy, SamplePolicy>;

    // clang-format off
    std::string prefix = std::string("RESULT")
        + " numberProcessors=" + std::to_string(comm.size()) 
        + " samplePolicy=" + std::string(SamplePolicy::getName())
        // + " mergeSort=" + std::string(Sorter::getName())
        + " StringGenerator=" + StringGenerator::getName()
        + " dToNRatio=" + std::to_string(genStringArgs.dToNRatio)
        + " stringLength=" + std::to_string(genStringArgs.stringLength)
        + " MPIAllToAllRoutine=" + MPIAllToAllRoutine::getName()
        + " ByteEncoder=" + ByteEncoder::getName()
        + " StringSet=" + StringSet::getName()
        + " iteration=" + std::to_string(iteration)
        + " size=" + std::to_string(numOfStrings)
        + " strongScaling=" + std::to_string(strongScaling)
        + " num_levels=" + std::to_string(levels.size());
    // clang-format on

    MeasuringTool& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(false);

    CheckerWithCompleteExchange<StringLcpPtr> checker;

    if (!strongScaling) {
        genStringArgs.numOfStrings *= comm.size();
    }
    if (comm.is_root()) {
        std::cout << "string generation started " << std::endl;
    }

    comm.barrier();
    StringGenerator generatedContainer =
        getGeneratedStringContainer<StringGenerator, StringSet>(genStringArgs);
    if (check || check_exhaustive) {
        checker.storeLocalInput(generatedContainer.raw_strings());
    }

    comm.barrier();
    if (comm.is_root()) {
        std::cout << "string generation completed " << std::endl;
    }
    StringLcpPtr rand_string_ptr = generatedContainer.make_string_lcp_ptr();
    const size_t numGeneratedChars = generatedContainer.char_size();
    const size_t numGeneratedStrings = generatedContainer.size();

    measuring_tool.add(numGeneratedChars - numGeneratedStrings, "InputChars", false);
    measuring_tool.add(numGeneratedStrings, "InputStrings", false);

    // MPI warmup
    auto num_bytes = std::min<size_t>(numOfStrings * 5, 100000u);
    dss_schimek::mpi::randomDataAllToAllExchange(num_bytes, comm);

    comm.barrier();

    measuring_tool.start("sorting_overall");

    Sorter merge_sort;
    StringLcpContainer<StringSet> sorted_string_cont =
        merge_sort.template sort<>(rand_string_ptr, std::move(generatedContainer), levels, comm);

    // todo investigate this
    measuring_tool.start("prefix_decompression");
    if (AllToAllPolicy::PrefixCompression && comm.size() > 1) {
        sorted_string_cont.extendPrefix(
            sorted_string_cont.make_string_set(),
            sorted_string_cont.savedLcps()
        );
    }
    measuring_tool.stop("prefix_decompression");

    measuring_tool.stop("sorting_overall");

    if (check || check_exhaustive) {
        const StringLcpPtr sorted_strptr = sorted_string_cont.make_string_lcp_ptr();
        bool const is_complete_and_sorted = dss_schimek::is_complete_and_sorted(
            sorted_strptr,
            numGeneratedChars,
            sorted_string_cont.char_size(),
            numGeneratedStrings,
            sorted_string_cont.size(),
            comm
        );

        die_verbose_unless(is_complete_and_sorted, "not sorted");
        if (check_exhaustive) {
            bool const isSorted = checker.check(sorted_strptr, true);
            die_verbose_unless(isSorted, "not sorted completely");
        }
    }

    std::stringstream buffer;
    measuring_tool.writeToStream(buffer);
    if (comm.is_root()) {
        std::cout << buffer.str() << std::endl;
    }
    measuring_tool.reset();
}

using namespace dss_schimek;

struct SorterArgs {
    size_t size;
    bool check;
    bool check_exhaustive;
    size_t iteration;
    bool strong_scaling;
    GeneratedStringsArgs generator_args;
    std::vector<size_t> levels;
};

template <
    typename StringSet,
    typename StringGenerator,
    typename SamplePolicy,
    typename MPIRoutineAllToAll,
    typename ByteEncoder>
void arg6(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    execute_sorter<StringSet, StringGenerator, SamplePolicy, MPIRoutineAllToAll, ByteEncoder>(
        args.size,
        args.check,
        args.check_exhaustive,
        args.iteration,
        args.strong_scaling,
        args.generator_args,
        args.levels
    );
}

template <
    typename StringSet,
    typename StringGenerator,
    typename SamplePolicy,
    typename MPIRoutineAllToAll>
void arg5(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    switch (key.byteEncoder_) {
        case PolicyEnums::ByteEncoder::emptyByteEncoderCopy: {
            // using ByteEncoder = dss_schimek::EmptyByteEncoderCopy;
            // arg6<StringSet, StringGenerator, SampleString,
            // MPIRoutineAllToAll,
            //    ByteEncoder>(key, args);
            break;
        }
        case PolicyEnums::ByteEncoder::emptyByteEncoderMemCpy: {
            using ByteEncoder = dss_schimek::EmptyByteEncoderMemCpy;
            arg6<StringSet, StringGenerator, SamplePolicy, MPIRoutineAllToAll, ByteEncoder>(
                key,
                args
            );
            break;
        }
        case PolicyEnums::ByteEncoder::sequentialDelayedByteEncoder: {
            // using ByteEncoder = dss_schimek::SequentialDelayedByteEncoder;
            // arg6<StringSet, StringGenerator, SampleString,
            // MPIRoutineAllToAll,
            //    ByteEncoder>(key, args);
            break;
        }
        case PolicyEnums::ByteEncoder::sequentialByteEncoder: {
            using ByteEncoder = dss_schimek::SequentialByteEncoder;
            arg6<StringSet, StringGenerator, SamplePolicy, MPIRoutineAllToAll, ByteEncoder>(
                key,
                args
            );
            break;
        }
        case PolicyEnums::ByteEncoder::interleavedByteEncoder: {
            using ByteEncoder = dss_schimek::InterleavedByteEncoder;
            arg6<StringSet, StringGenerator, SamplePolicy, MPIRoutineAllToAll, ByteEncoder>(
                key,
                args
            );
            break;
        }
        case PolicyEnums::ByteEncoder::emptyLcpByteEncoderMemCpy: {
            using ByteEncoder = dss_schimek::EmptyLcpByteEncoderMemCpy;
            arg6<StringSet, StringGenerator, SamplePolicy, MPIRoutineAllToAll, ByteEncoder>(
                key,
                args
            );
            break;
        }
    };
}
template <typename StringSet, typename StringGenerator, typename SamplePolicy>
void arg4(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    // todo get rid of this
    switch (key.mpiRoutineAllToAll_) {
        case PolicyEnums::MPIRoutineAllToAll::small: {
            // using MPIRoutineAllToAll = dss_schimek::mpi::AllToAllvSmall;
            // arg5<StringSet, StringGenerator, SampleString,
            // MPIRoutineAllToAll>(
            //    key, args);
            tlx_die("not implemented");
            break;
        }
        case PolicyEnums::MPIRoutineAllToAll::directMessages: {
            // using MPIRoutineAllToAll = dss_schimek::mpi::AllToAllvDirectMessages;
            // arg5<StringSet, StringGenerator, SampleString,
            // MPIRoutineAllToAll>(
            //    key, args);
            tlx_die("not implemented");
            break;
        }
        case PolicyEnums::MPIRoutineAllToAll::combined: {
            using MPIRoutineAllToAll =
                dss_schimek::mpi::AllToAllvCombined<dss_schimek::mpi::AllToAllvSmall>;
            arg5<StringSet, StringGenerator, SamplePolicy, MPIRoutineAllToAll>(key, args);
            break;
        }
    }
}
template <typename StringSet, typename StringGenerator>
void arg3(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_mehnert::sample;
    switch (key.sampleStringPolicy_) {
        case PolicyEnums::SampleString::numStrings: {
            using SampleString = NumStringsPolicy<StringSet>;
            arg4<StringSet, StringGenerator, SampleString>(key, args);
            break;
        }
        case PolicyEnums::SampleString::numChars: {
            using SampleString = NumCharsPolicy<StringSet>;
            arg4<StringSet, StringGenerator, SampleString>(key, args);
            break;
        }
        case PolicyEnums::SampleString::indexedNumStrings: {
            using SampleString = IndexedNumStringPolicy<StringSet>;
            arg4<StringSet, StringGenerator, SampleString>(key, args);
            break;
        }
        case PolicyEnums::SampleString::indexedNumChars: {
            using SampleString = IndexedNumCharsPolicy<StringSet>;
            arg4<StringSet, StringGenerator, SampleString>(key, args);
            break;
        }
    };
}

template <typename StringSet>
void arg2(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    switch (key.stringGenerator_) {
        case PolicyEnums::StringGenerator::skewedRandomStringLcpContainer: {
            // using StringGenerator =
            //    dss_schimek::SkewedRandomStringLcpContainer<StringSet>;
            // arg3<StringSet, StringGenerator>(key, args);
            break;
        }
        case PolicyEnums::StringGenerator::DNRatioGenerator: {
            using StringGenerator = dss_schimek::DNRatioGenerator<StringSet>;
            arg3<StringSet, StringGenerator>(key, args);
            break;
        }
        case PolicyEnums::StringGenerator::File: {
            using StringGenerator = dss_schimek::FileDistributer<StringSet>;
            arg3<StringSet, StringGenerator>(key, args);
            break;
        }
        case PolicyEnums::StringGenerator::SkewedDNRatioGenerator: {
            using StringGenerator = dss_schimek::SkewedDNRatioGenerator<StringSet>;
            arg3<StringSet, StringGenerator>(key, args);
            break;
        }
        case PolicyEnums::StringGenerator::SuffixGenerator: {
            using StringGenerator = dss_schimek::SuffixGenerator<StringSet>;
            arg3<StringSet, StringGenerator>(key, args);
            break;
        }
    };
}

void arg1(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    switch (key.stringSet_) {
        case PolicyEnums::StringSet::UCharLengthStringSet:
            arg2<UCharLengthStringSet>(key, args);
            break;
        case PolicyEnums::StringSet::UCharStringSet:
            // arg2<UCharStringSet>(key, args);
            break;
    };
}

int main(int argc, char* argv[]) {
    using namespace dss_schimek;

    kamping::Environment env{argc, argv};

    bool check = false;
    bool checkExhaustive = false;
    unsigned int generator = 0;
    bool strongScaling = false;
    unsigned int sampleStringsPolicy = static_cast<int>(PolicyEnums::SampleString::numStrings);
    unsigned int byteEncoder = static_cast<int>(PolicyEnums::ByteEncoder::emptyByteEncoderMemCpy);
    unsigned int mpiRoutineAllToAll = static_cast<int>(PolicyEnums::MPIRoutineAllToAll::combined);
    unsigned int numberOfStrings = 100000;
    unsigned int numberOfIterations = 5;
    unsigned int stringLength = 50;
    double dToNRatio = 0.5;
    std::string path = "";
    std::vector<std::string> levelsParam;

    tlx::CmdlineParser cp;
    cp.set_description("a distributed string sorter");
    cp.set_author("Matthias Schimek, Pascal Mehnert");
    cp.add_string('y', "path", path, " path to file");
    cp.add_double('r', "dToNRatio", dToNRatio, "D/N ratio");
    cp.add_unsigned('s', "size", numberOfStrings, " number of strings to be generated");
    cp.add_unsigned(
        'p',
        "sampleStringsPolicy",
        sampleStringsPolicy,
        "0 = NumStrings, 1 = NumChars, 2 = IndexedNumStrings, 3 = "
        "IndexedNumChars"
    );
    cp.add_unsigned(
        'b',
        "byteEncoder",
        byteEncoder,
        "emptyByteEncoderCopy = 0, emptyByteEncoderMemCpy = 1, "
        "sequentialDelayedByteEncoder = 2, sequentialByteEncoder = 3, "
        "interleavedByteEncoder = 4, emptyLcpByteEncoderMemCpy = 5"
    );
    cp.add_unsigned(
        'm',
        "MPIRoutineAllToAll",
        mpiRoutineAllToAll,
        "small = 0, directMessages = 1, combined = 2"
    );
    cp.add_unsigned('i', "numberOfIterations", numberOfIterations, "");
    cp.add_flag('c', "check", check, " ");
    cp.add_flag('w', "exhaustiveCheck", checkExhaustive, " ");
    cp.add_unsigned('k', "generator", generator, " 0 = skewed, 1 = DNGen ");
    cp.add_flag('x', "strongScaling", strongScaling, " ");
    cp.add_unsigned('a', "stringLength", stringLength, " string Length ");
    cp.add_opt_param_stringlist(
        "num_group",
        levelsParam,
        "no. groups during each level of merge sort"
    );

    if (!cp.process(argc, argv)) {
        return EXIT_FAILURE;
    }

    PolicyEnums::CombinationKey key{
        .stringSet_ = PolicyEnums::StringSet::UCharLengthStringSet,
        .golombEncoding_ = PolicyEnums::GolombEncoding::noGolombEncoding,
        .stringGenerator_ = PolicyEnums::getStringGenerator(generator),
        .sampleStringPolicy_ = PolicyEnums::getSampleString(sampleStringsPolicy),
        .mpiRoutineAllToAll_ = PolicyEnums::getMPIRoutineAllToAll(mpiRoutineAllToAll),
        .byteEncoder_ = PolicyEnums::getByteEncoder(byteEncoder),
        .compressLcps_ = true,
    };

    GeneratedStringsArgs generatorArgs{
        .numOfStrings = numberOfStrings,
        .stringLength = stringLength,
        .minStringLength = stringLength,
        .maxStringLength = stringLength + 10,
        .dToNRatio = dToNRatio,
        .path = path,
    };

    std::vector<size_t> levels;
    for (auto const& level: levelsParam) {
        levels.push_back(std::stoi(level));
    }

    for (size_t i = 0; i < numberOfIterations; ++i) {
        SorterArgs
            args{numberOfStrings, check, checkExhaustive, i, strongScaling, generatorArgs, levels};
        arg1(key, args);
    }

    return EXIT_SUCCESS;
}
