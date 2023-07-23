// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>

#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <tlx/cmdline_parser.hpp>
#include <tlx/die.hpp>
#include <tlx/die/core.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/alltoall.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/warmup.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "util/measuringTool.hpp"
#include "util/random_string_generator.hpp"
#include "variant_selection.hpp"

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
    typename MPIAllToAllRoutine,
    typename GolombEncoding,
    typename ByteEncoder,
    typename LcpCompression>
void run_merge_sort(SorterArgs args) {
    using namespace dss_mehnert;
    using namespace dss_schimek;

    using StringLcpPtr = typename tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    // todo implement this variable
    constexpr bool lcp_compression = LcpCompression();

    using AllToAllPolicy =
        mpi::AllToAllStringImpl<lcp_compression, StringSet, MPIAllToAllRoutine, ByteEncoder>;
    using Sorter = sorter::DistributedMergeSort<StringLcpPtr, AllToAllPolicy, SamplePolicy>;

    dss_mehnert::Communicator comm;

    // clang-format off
    std::string prefix = std::string("RESULT")
        + " numberProcessors=" + std::to_string(comm.size()) 
        + " samplePolicy=" + std::string(SamplePolicy::getName())
        + " StringGenerator=" + StringGenerator::getName()
        + " dToNRatio=" + std::to_string(args.generator_args.dToNRatio)
        + " stringLength=" + std::to_string(args.generator_args.stringLength)
        + " MPIAllToAllRoutine=" + MPIAllToAllRoutine::getName()
        + " ByteEncoder=" + ByteEncoder::getName()
        + " StringSet=" + StringSet::getName()
        + " iteration=" + std::to_string(args.iteration)
        + " numStrings=" + std::to_string(args.size)
        + " strongScaling=" + std::to_string(args.strong_scaling)
        + " numLevels=" + std::to_string(args.levels.size());
    // clang-format on

    auto& measuring_tool = dss_mehnert::measurement::MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(false);

    CheckerWithCompleteExchange<StringLcpPtr> checker;

    if (!args.strong_scaling) {
        args.generator_args.numOfStrings *= comm.size();
    }
    if (comm.is_root()) {
        std::cout << "string generation started " << std::endl;
    }

    comm.barrier();
    StringGenerator rand_container =
        getGeneratedStringContainer<StringGenerator, StringSet>(args.generator_args);
    if (args.check || args.check_exhaustive) {
        checker.storeLocalInput(rand_container.raw_strings());
    }

    comm.barrier();
    if (comm.is_root()) {
        std::cout << "string generation completed " << std::endl;
    }
    StringLcpPtr rand_string_ptr = rand_container.make_string_lcp_ptr();
    const size_t num_gen_chars = rand_container.char_size();
    const size_t num_gen_strs = rand_container.size();

    measuring_tool.add(num_gen_chars - num_gen_strs, "input_chars");
    measuring_tool.add(num_gen_strs, "input_strings");

    auto num_bytes = std::min<size_t>(args.size * 5, 100000u);
    dss_schimek::mpi::randomDataAllToAllExchange(num_bytes, comm);

    comm.barrier();

    measuring_tool.start("sorting_overall");

    Sorter merge_sort;
    StringLcpContainer<StringSet> sorted_string_cont =
        merge_sort.template sort<>(rand_string_ptr, std::move(rand_container), args.levels, comm);

    measuring_tool.stop("sorting_overall");

    if (args.check || args.check_exhaustive) {
        const StringLcpPtr sorted_strptr = sorted_string_cont.make_string_lcp_ptr();
        bool const is_complete_and_sorted = dss_schimek::is_complete_and_sorted(
            sorted_strptr,
            num_gen_chars,
            sorted_string_cont.char_size(),
            num_gen_strs,
            sorted_string_cont.size(),
            comm
        );
        die_verbose_unless(is_complete_and_sorted, "not sorted");

        if (args.check_exhaustive) {
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

template <typename... Args>
void arg7(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    if (key.prefix_doubling) {
        // todo implement golomb encoding option
        tlx_die("not yet implemented");
    } else {
        run_merge_sort<Args...>(args);
    }
}

template <typename... Args>
void arg6(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    if (key.lcp_compression) {
        arg7<Args..., std::true_type>(key, args);
    } else {
        arg7<Args..., std::false_type>(key, args);
    }
}

template <typename... Args>
void arg5(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_schimek;

    if (key.prefix_compression) {
        arg6<Args..., EmptyLcpByteEncoderMemCpy>(key, args);
    } else {
        arg6<Args..., EmptyByteEncoderMemCpy>(key, args);
    }
}

template <typename... Args>
void arg4(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_schimek;

    switch (key.golomb_encoding) {
        case PolicyEnums::GolombEncoding::noGolombEncoding: {
            arg5<Args..., AllToAllHashesNaive>(key, args);
            break;
        }
        case PolicyEnums::GolombEncoding::sequentialGolombEncoding: {
            arg5<Args..., AllToAllHashesGolomb>(key, args);
            break;
        }
        case PolicyEnums::GolombEncoding::pipelinedGolombEncoding: {
            // arg5<Args..., AllToAllHashValuesPipeline>(key, args);
            tlx_die("not implemented");
            break;
        }
    }
}

template <typename... Args>
void arg3(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_schimek::mpi;

    switch (key.alltoall_routine) {
        case PolicyEnums::MPIRoutineAllToAll::small: {
            // arg4<Args..., AllToAllvSmall>(key, args);
            tlx_die("not implemented");
            break;
        }
        case PolicyEnums::MPIRoutineAllToAll::directMessages: {
            // arg4<Args..., AllToAllvDirectMessages>(key, args);
            tlx_die("not implemented");
            break;
        }
        case PolicyEnums::MPIRoutineAllToAll::combined: {
            arg4<Args..., AllToAllvCombined<AllToAllvSmall>>(key, args);
            break;
        }
    }
}

template <typename... Args>
void arg2(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_mehnert::sample;

    switch (key.sample_policy) {
        case PolicyEnums::SampleString::numStrings: {
            arg3<Args..., NumStringsPolicy>(key, args);
            break;
        }
        case PolicyEnums::SampleString::numChars: {
            arg3<Args..., NumCharsPolicy>(key, args);
            break;
        }
        case PolicyEnums::SampleString::indexedNumStrings: {
            arg3<Args..., IndexedNumStringPolicy>(key, args);
            break;
        }
        case PolicyEnums::SampleString::indexedNumChars: {
            arg3<Args..., IndexedNumCharsPolicy>(key, args);
            break;
        }
    };
}

template <typename StringSet>
void arg1(PolicyEnums::CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_schimek;

    switch (key.string_generator) {
        case PolicyEnums::StringGenerator::skewedRandomStringLcpContainer: {
            // arg2<SkewedRandomStringLcpContainer<StringSet>();
            tlx_die("not implemented");
            break;
        }
        case PolicyEnums::StringGenerator::DNRatioGenerator: {
            arg2<StringSet, DNRatioGenerator<StringSet>>(key, args);
            break;
        }
        case PolicyEnums::StringGenerator::File: {
            arg2<StringSet, FileDistributer<StringSet>>(key, args);
            break;
        }
        case PolicyEnums::StringGenerator::SkewedDNRatioGenerator: {
            arg2<StringSet, SkewedDNRatioGenerator<StringSet>>(key, args);
            break;
        }
        case PolicyEnums::StringGenerator::SuffixGenerator: {
            arg2<StringSet, SuffixGenerator<StringSet>>(key, args);
            break;
        }
    };
}

int main(int argc, char* argv[]) {
    using namespace dss_schimek;

    kamping::Environment env{argc, argv};

    bool check = false;
    bool check_exhaustive = false;
    bool strong_scaling = false;
    bool prefix_compression = false;
    bool lcp_compression = false;
    bool prefix_doubling = false;
    unsigned int generator = static_cast<int>(PolicyEnums::StringGenerator::DNRatioGenerator);
    unsigned int sample_policy = static_cast<int>(PolicyEnums::SampleString::numStrings);
    unsigned int alltoall_routine = static_cast<int>(PolicyEnums::MPIRoutineAllToAll::combined);
    unsigned int golomb_encoding = static_cast<int>(PolicyEnums::GolombEncoding::noGolombEncoding);
    size_t num_strings = 100000;
    size_t num_iterations = 5;
    size_t string_length = 50;
    double DN_ratio = 0.5;
    std::string path = "";
    std::vector<std::string> levels_param;

    tlx::CmdlineParser cp;
    cp.set_description("a distributed string sorter");
    cp.set_author("Matthias Schimek, Pascal Mehnert");
    cp.add_string('y', "path", path, "path to file");
    cp.add_unsigned(
        'k',
        "generator",
        generator,
        "type of string generation to use (0 = skewed, 1 = DNGen)"
    );
    cp.add_double('r', "DN-ratio", DN_ratio, "D/N ratio of generated strings");
    cp.add_size_t('n', "num-strings", num_strings, "number of strings to be generated");
    cp.add_size_t('m', "len-strings", string_length, "length of generated strings");
    cp.add_size_t('i', "num-iterations", num_iterations, "numer of sorting iterations to run");
    cp.add_flag('x', "strong-scaling", strong_scaling, "perform a strong scaling experiment");
    cp.add_unsigned(
        's',
        "sample-policy",
        sample_policy,
        "strategy to use for splitter sampling "
        "(0 = NumStrings, 1 = NumChars, 2 = IndexedNumStrings, 3 = IndexedNumChars)"
    );
    cp.add_flag(
        'l',
        "lcp-compression",
        lcp_compression,
        "compress LCP values during string exchange"
    );
    cp.add_flag(
        'p',
        "prefix-compression",
        prefix_compression,
        "use LCP compression during string exchange"
    );
    cp.add_flag('d', "prefix-doubling", prefix_doubling, "use prefix doubling merge sort");
    cp.add_unsigned(
        'a',
        "alltoall-routine",
        alltoall_routine,
        "All-To-All routine to use during string exchange "
        "(0 = small, 1 = directMessages, 2 = combined)"
    );
    cp.add_unsigned(
        'g',
        "golomb",
        golomb_encoding,
        "type of golomb encoding to use during prefix doubling"
    );
    cp.add_flag(
        'c',
        "check",
        check,
        "check if strings/chars were lost and that the result is sorted"
    );
    cp.add_flag(
        'C',
        "check-exhaustive",
        check_exhaustive,
        "check that the the output exactly matches the input"
    );
    cp.add_opt_param_stringlist(
        "num-groups",
        levels_param,
        "no. groups for each level of multi-level merge sort"
    );

    if (!cp.process(argc, argv)) {
        return EXIT_FAILURE;
    }

    PolicyEnums::CombinationKey key{
        // todo add parameter
        .golomb_encoding = PolicyEnums::getGolombEncoding(golomb_encoding),
        .string_generator = PolicyEnums::getStringGenerator(generator),
        .sample_policy = PolicyEnums::getSampleString(sample_policy),
        .alltoall_routine = PolicyEnums::getMPIRoutineAllToAll(alltoall_routine),
        .prefix_compression = prefix_compression,
        .lcp_compression = lcp_compression,
        .prefix_doubling = prefix_doubling,
    };

    GeneratedStringsArgs generator_args{
        .numOfStrings = num_strings,
        .stringLength = string_length,
        .minStringLength = string_length,
        .maxStringLength = string_length + 10,
        .dToNRatio = DN_ratio,
        .path = path,
    };

    std::vector<size_t> levels(levels_param.size());
    auto stoi = [](auto& str) { return std::stoi(str); };
    std::transform(levels_param.begin(), levels_param.end(), levels.begin(), stoi);

    for (size_t i = 0; i < num_iterations; ++i) {
        SorterArgs
            args{num_strings, check, check_exhaustive, i, strong_scaling, generator_args, levels};
        arg1<dss_schimek::UCharLengthStringSet>(key, args);
    }

    return EXIT_SUCCESS;
}
