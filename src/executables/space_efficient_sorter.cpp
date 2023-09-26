// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
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
#include "mpi/warmup.hpp"
#include "options.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/multi_level.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "sorter/distributed/sample.hpp"
#include "sorter/distributed/space_efficient.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"
#include "util/string_generator.hpp"
#include "variant_selection.hpp"

enum class CharGenerator { file = 0, random = 1, sentinel = 2 };

enum class StringGenerator { suffix = 0, window = 1, difference_cover = 2, sentinel = 3 };

template <typename T>
T get_enum_value(size_t const i) {
    if (i < static_cast<size_t>(T::sentinel)) {
        return static_cast<T>(i);
    } else {
        tlx_die("enum value out of range");
    }
}

struct GeneratedStringsArgs {
    CharGenerator char_gen;
    StringGenerator string_gen;
    size_t num_chars;
    size_t step;
    size_t len_strings;
    size_t difference_cover;
    std::string path;
};

struct SorterArgs {
    std::string experiment;
    size_t sampling_factor;
    size_t quantile_size;
    bool check;
    bool check_exhaustive;
    size_t iteration;
    GeneratedStringsArgs generator_args;
    std::vector<size_t> levels;
};

struct CombinationKey {
    bool sample_chars;
    bool sample_indexed;
    bool sample_random;
    PolicyEnums::MPIRoutineAllToAll alltoall_routine;
    PolicyEnums::Subcommunicators subcomms;
    bool rquick_v1;
    bool rquick_lcp;
    PolicyEnums::Redistribution redistribution;
    bool prefix_compression;
    bool lcp_compression;
    bool prefix_doubling;
    bool grid_bloomfilter;
};

void check_path_exists(std::string const& path) {
    tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
};

template <typename StringSet>
auto generate_compressed_strings(SorterArgs const& args, dss_mehnert::Communicator const& comm) {
    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto const& gen_args = args.generator_args;

    auto input_chars = [=]() -> std::vector<typename StringSet::Char> {
        using namespace PolicyEnums;
        using namespace dss_mehnert;

        switch (gen_args.char_gen) {
            case CharGenerator::random: {
                return RandomCharGenerator<StringSet>{gen_args.num_chars};
            }
            case CharGenerator::file: {
                return FileCharGenerator<StringSet>{gen_args.path};
            }
            case CharGenerator::sentinel: {
                break;
            }
        }
        tlx_die("invalid chararcter generator");
    }();

    auto input_strings = [=, &input_chars]() -> std::vector<typename StringSet::String> {
        using namespace dss_mehnert;

        auto const step = gen_args.step, length = gen_args.len_strings;
        auto const dc = gen_args.difference_cover;

        switch (gen_args.string_gen) {
            case StringGenerator::suffix: {
                return CompressedSuffixGenerator<StringSet>{input_chars, step};
            }
            case StringGenerator::window: {
                return CompressedWindowGenerator<StringSet>{input_chars, length, step};
            }
            case StringGenerator::difference_cover: {
                return CompressedDifferenceCoverGenerator<StringSet>{input_chars, dc};
            }
            case StringGenerator::sentinel: {
                break;
            }
        }
        tlx_die("invalid string generator");
    }();

    dss_mehnert::StringLcpContainer<StringSet> input_container{
        std::move(input_chars),
        std::move(input_strings)};
    measuring_tool.stop("generate_strings");

    comm.barrier();

    auto const num_gen_chars = input_container.char_size();
    auto const num_gen_strs = input_container.size();
    auto const num_uncompressed_chars = input_container.make_string_set().get_sum_length();
    measuring_tool.add(num_gen_chars - num_gen_strs, "input_chars");
    measuring_tool.add(num_gen_strs, "input_strings");
    measuring_tool.add(num_uncompressed_chars, "uncompressed_input_chars");

    return input_container;
}

template <
    typename CharType,
    typename SamplePolicy,
    typename PartitionPolicy,
    typename MPIAllToAllRoutine,
    typename Subcommunicators,
    typename RedistributionPolicy,
    typename LcpCompression,
    typename PrefixCompression,
    typename BloomFilterPolicy>
void run_space_efficient_sort(
    CombinationKey const& key,
    SorterArgs const& args,
    std::string prefix,
    dss_mehnert::Communicator const& comm
) {
    using namespace dss_schimek;

    constexpr bool lcp_compression = LcpCompression();
    constexpr bool prefix_compression = PrefixCompression();

    using MaterializedStringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using AllToAllPolicy = mpi::AllToAllStringImplPrefixDoubling<
        lcp_compression,
        prefix_compression,
        MaterializedStringSet,
        MPIAllToAllRoutine>;
    using BaseSorter = dss_mehnert::sorter::PrefixDoublingMergeSort<
        tlx::sort_strings_detail::StringLcpPtr<MaterializedStringSet, size_t>,
        Subcommunicators,
        RedistributionPolicy,
        AllToAllPolicy,
        SamplePolicy,
        PartitionPolicy,
        BloomFilterPolicy>;

    using StringSet = dss_mehnert::CompressedStringSet<CharType>;
    // todo maybe use different sample policy
    using Sorter = dss_mehnert::sorter::
        SpaceEfficientSort<CharType, Subcommunicators, SamplePolicy, PartitionPolicy, BaseSorter>;

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(false);

    measuring_tool.disableCommVolume();

    auto input_container = generate_compressed_strings<StringSet>(args, comm);
    // auto num_input_chars = input_container.char_size();
    // auto num_input_strs = input_container.size();

    // skip unused levels for multi-level merge sort
    auto pred = [&](auto const& group_size) { return group_size < comm.size(); };
    auto first_level = std::find_if(args.levels.begin(), args.levels.end(), pred);

    measuring_tool.enableCommVolume();

    comm.barrier();
    measuring_tool.start("none", "sorting_overall");

    measuring_tool.start("none", "create_communicators");
    Subcommunicators comms{first_level, args.levels.end(), comm};
    measuring_tool.stop("none", "create_communicators", comm);

    Sorter merge_sort{SamplePolicy{args.sampling_factor}, args.quantile_size};
    auto permutation = merge_sort.sort(std::move(input_container), comms);

    measuring_tool.stop("none", "sorting_overall", comm);
    measuring_tool.disable();
    measuring_tool.disableCommVolume();

    measuring_tool.write_on_root(std::cout, comm);
    measuring_tool.reset();
}

std::string get_result_prefix(
    SorterArgs const& args, CombinationKey const& key, dss_mehnert::Communicator const& comm
) {
    // clang-format off
    return std::string("RESULT")
           + (args.experiment.empty() ? "" : (" experiment=" + args.experiment))
           + " num_procs="          + std::to_string(comm.size())
           + " num_levels="         + std::to_string(args.levels.size())
           + " iteration="          + std::to_string(args.iteration)
           + " sample_chars="       + std::to_string(key.sample_chars)
           + " sample_indexed="     + std::to_string(key.sample_indexed)
           + " sample_random="      + std::to_string(key.sample_random)
           + " sampling_factor="    + std::to_string(args.sampling_factor)
           + " rquick_v1="          + std::to_string(key.rquick_v1)
           + " rquick_lcp="         + std::to_string(key.rquick_lcp)
           + " lcp_compression="    + std::to_string(key.lcp_compression)
           + " prefix_compression=" + std::to_string(key.prefix_compression)
           + " prefix_doubling="    + std::to_string(key.prefix_doubling)
           + " grid_bloomfilter="   + std::to_string(key.grid_bloomfilter);
    // clang-format on
}

template <
    typename CharType,
    typename SamplePolicy,
    typename PartitionPolicy,
    typename MPIAllToAllRoutine,
    typename Subcommunicators,
    typename RedistributionPolicy,
    typename LcpCompression,
    typename PrefixCompression,
    typename BloomFilterPolicy>
void print_config(std::string_view prefix, SorterArgs const& args, CombinationKey const& key) {
    // todo print string generator arguments
    // todo partition policy
    // todo std::cout << prefix << " key=string_generator name=" << StringGenerator::getName() <<
    // "\n";
    std::cout << prefix << " key=alltoall_routine name=" << MPIAllToAllRoutine::getName() << "\n";
    std::cout << prefix << " key=subcomms name=" << Subcommunicators::get_name() << "\n";
    std::cout << prefix << " key=redistribution name=" << RedistributionPolicy::get_name() << "\n";
}

void die_with_feature(std::string_view feature) {
    tlx_die("feature disabled for compile time; enable with '-D" << feature << "=On'");
}

template <typename... Args>
void arg8(CombinationKey const& key, SorterArgs const& args) {
    dss_mehnert::Communicator comm;
    auto prefix = get_result_prefix(args, key, comm);
    if (comm.is_root()) {
        print_config<Args...>(prefix, args, key);
    }

    run_space_efficient_sort<Args...>(key, args, prefix, comm);
}

template <typename... Args>
void arg7(CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_mehnert::bloomfilter;

    if (key.grid_bloomfilter) {
        arg8<Args..., MultiLevel<XXHasher>>(key, args);
    } else {
        arg8<Args..., SingleLevel<XXHasher>>(key, args);
    }
}

template <typename... Args>
void arg6(CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_schimek;

    if (key.prefix_compression) {
        arg7<Args..., std::true_type>(key, args);
    } else {
        arg7<Args..., std::false_type>(key, args);
    }
}

template <typename... Args>
void arg5(CombinationKey const& key, SorterArgs const& args) {
    if (key.lcp_compression) {
        arg6<Args..., std::true_type>(key, args);
    } else {
        arg6<Args..., std::false_type>(key, args);
    }
}

template <typename... Args>
void arg4(CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_mehnert::multi_level;
    using namespace dss_mehnert::redistribution;
    using dss_mehnert::Communicator;

    switch (key.subcomms) {
        case PolicyEnums::Subcommunicators::none: {
            if constexpr (CliOptions::enable_split) {
                using Split = NoSplit<Communicator>;
                using Redistribution = NoRedistribution<Communicator>;
                arg5<Args..., Split, Redistribution>(key, args);
            } else {
                die_with_feature("CLI_ENABLE_SPLIT");
            }
            break;
        }
        case PolicyEnums::Subcommunicators::naive: {
            if constexpr (CliOptions::enable_split) {
                using Split = RowwiseSplit<Communicator>;
                switch (key.redistribution) {
                    case PolicyEnums::Redistribution::naive: {
                        using Redistribution = NaiveRedistribution<Communicator>;
                        arg5<Args..., Split, Redistribution>(key, args);
                        break;
                    };
                    case PolicyEnums::Redistribution::simple_strings: {
                        using Redistribution = SimpleStringRedistribution<Communicator>;
                        arg5<Args..., Split, Redistribution>(key, args);
                        break;
                    };
                    case PolicyEnums::Redistribution::simple_chars: {
                        using Redistribution = SimpleCharRedistribution<Communicator>;
                        arg5<Args..., Split, Redistribution>(key, args);
                        break;
                    };
                };
            } else {
                die_with_feature("CLI_ENABLE_SPLIT");
            }
            break;
        }
        case PolicyEnums::Subcommunicators::grid: {
            using Split = GridwiseSplit<Communicator>;
            using Redistribution = GridwiseRedistribution<Communicator>;
            arg5<Args..., Split, Redistribution>(key, args);
            break;
        }
    }
}

// todo remove this
template <typename... Args>
void arg3(CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_schimek::mpi;

    switch (key.alltoall_routine) {
        case PolicyEnums::MPIRoutineAllToAll::small: {
            if constexpr (CliOptions::enable_alltoall) {
                arg4<Args..., AllToAllvSmall>(key, args);
            } else {
                die_with_feature("CLI_ENABLE_ALLTOALL");
            }
            break;
        }
        case PolicyEnums::MPIRoutineAllToAll::directMessages: {
            if constexpr (CliOptions::enable_alltoall) {
                arg4<Args..., AllToAllvDirectMessages>(key, args);
            } else {
                die_with_feature("CLI_ENABLE_ALLTOALL");
            }
            break;
        }
        case PolicyEnums::MPIRoutineAllToAll::combined: {
            arg4<Args..., AllToAllvCombined<AllToAllvSmall>>(key, args);
            break;
        }
    }
}

template <typename CharType, typename Sampler>
void arg2(CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_mehnert::partition;

    constexpr bool is_indexed = Sampler::is_indexed;

    tlx_die_verbose_if(
        key.rquick_v1 && key.rquick_lcp,
        "RQuick v1 does not support using LCP values"
    );

    if (key.rquick_v1) {
        if constexpr (CliOptions::enable_rquick_v1) {
            arg3<CharType, Sampler, RQuickV1<CharType, is_indexed>>(key, args);
        } else {
            die_with_feature("CLI_ENABLE_RQUICK_V1");
        }
    } else {
        if (key.rquick_lcp) {
            if constexpr (CliOptions::enable_rquick_lcp) {
                arg3<CharType, Sampler, RQuickV2<CharType, is_indexed, true>>(key, args);
            } else {
                die_with_feature("CLI_ENABLE_RQUICK_LCP");
            }
        } else {
            arg3<CharType, Sampler, RQuickV2<CharType, is_indexed, false>>(key, args);
        }
    }
}

template <typename... Args>
void arg1(CombinationKey const& key, SorterArgs const& args) {
    using namespace dss_mehnert::sample;

    // todo this could/should use type erasure

    auto with_random = [=]<bool indexed, bool random> {
        if (key.sample_chars) {
            arg2<Args..., CharBasedSampling<indexed, random>>(key, args);
        } else {
            arg2<Args..., StringBasedSampling<indexed, random>>(key, args);
        }
    };

    auto with_indexed = [=]<bool indexed> {
        if (key.sample_random) {
            with_random.template operator()<indexed, true>();
        } else {
            with_random.template operator()<indexed, false>();
        }
    };

    if (key.sample_chars) {
        with_indexed.template operator()<true>();
    } else {
        with_indexed.template operator()<false>();
    }
}

int main(int argc, char* argv[]) {
    using namespace dss_schimek;

    kamping::Environment env{argc, argv};

    bool check = false;
    bool check_exhaustive = false;
    bool sample_chars = false;
    bool sample_indexed = false;
    bool sample_random = false;
    size_t sampling_factor = 2;
    bool rquick_v1 = false;
    bool rquick_lcp = false;
    bool prefix_compression = false;
    bool lcp_compression = false;
    bool prefix_doubling = false;
    bool grid_bloomfilter = true;
    size_t quantile_size = 100 * 1024 * 1024;
    size_t num_chars = 100000;
    size_t step = 1;
    size_t len_strings = 500;
    size_t difference_cover = 3;
    unsigned int char_generator = static_cast<unsigned int>(CharGenerator::random);
    unsigned int string_generator = static_cast<unsigned int>(StringGenerator::suffix);
    unsigned int alltoall_routine = static_cast<int>(PolicyEnums::MPIRoutineAllToAll::combined);
    unsigned int comm_split = static_cast<int>(PolicyEnums::Subcommunicators::grid);
    unsigned int redistribution = static_cast<int>(PolicyEnums::Redistribution::naive);
    size_t num_iterations = 5;
    std::string path;
    std::string experiment;
    std::vector<std::string> levels_param;

    tlx::CmdlineParser cp;
    cp.set_description("a space efficient distributed string sorter");
    cp.set_author("Pascal Mehnert");
    cp.add_string('e', "experiment", experiment, "name to identify the experiment being run");
    cp.add_unsigned(
        'c',
        "char-generator",
        char_generator,
        "char generator to use "
        "([0]=random, 1=file)"
    );
    cp.add_unsigned(
        's',
        "string-generator",
        string_generator,
        "string generator to use "
        "([0]=suffix, 1=window, 2=difference_cover)"
    );
    cp.add_size_t('m', "len-strings", len_strings, "number of characters per string");
    cp.add_size_t('N', "num-chars", num_chars, "number of chars per rank");
    cp.add_size_t('T', "step", step, "characters to skip between strings");
    cp.add_size_t('D', "difference-cover", difference_cover, "size of difference cover");
    cp.add_string('y', "path", path, "path to input file");
    cp.add_size_t('i', "num-iterations", num_iterations, "number of sorting iterations to run");
    cp.add_flag('C', "sample-chars", sample_chars, "use character based sampling");
    cp.add_flag('I', "sample-indexed", sample_indexed, "use indexed sampling");
    cp.add_flag('R', "sample-random", sample_random, "use random sampling");
    cp.add_size_t('S', "sampling-factor", sampling_factor, "use the given oversampling factor");
    cp.add_flag('Q', "rquick-v1", rquick_v1, "use version 1 of RQuick (defaults to v2)");
    cp.add_flag('L', "rquick-lcp", rquick_lcp, "use LCP values in RQuick (only with v2)");
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
    cp.add_flag(
        'g',
        "grid-bloomfilter",
        grid_bloomfilter,
        "use gridwise bloom filter (requires prefix doubling) [default]"
    );
    cp.add_bytes(
        'q',
        "quantile-size",
        quantile_size,
        "work on quantiles of the given size [default: 100MiB]"
    );
    cp.add_unsigned(
        'a',
        "alltoall-routine",
        alltoall_routine,
        "All-To-All routine to use during string exchange "
        "(0=small, 1=direct, [2]=combined)"
    );
    cp.add_unsigned(
        'u',
        "subcomms",
        comm_split,
        "whether and how to create subcommunicators during merge sort "
        "(0=none, 1=naive, [2]=grid)"
    );
    cp.add_unsigned(
        't',
        "redistribution",
        redistribution,
        "redistribution scheme to use for multi-level sort"
        "([0]=naive, 1=simple-strings, 2=simple-chars)"
    );
    cp.add_flag(
        'v',
        "verify",
        check,
        "check if strings/chars were lost and that the result is sorted"
    );
    cp.add_flag(
        'V',
        "verify-full",
        check_exhaustive,
        "check that the the output exactly matches the input"
    );
    cp.add_opt_param_stringlist(
        "group-size",
        levels_param,
        "size of groups for multi-level merge sort"
    );

    if (!cp.process(argc, argv)) {
        return EXIT_FAILURE;
    }

    CombinationKey key{
        .sample_chars = sample_chars,
        .sample_indexed = sample_indexed,
        .sample_random = sample_random,
        .alltoall_routine = PolicyEnums::getMPIRoutineAllToAll(alltoall_routine),
        .subcomms = PolicyEnums::getSubcommunicators(comm_split),
        .rquick_v1 = rquick_v1,
        .rquick_lcp = rquick_lcp,
        .redistribution = PolicyEnums::getRedistribution(redistribution),
        .prefix_compression = prefix_compression,
        .lcp_compression = lcp_compression,
        .prefix_doubling = prefix_doubling,
        .grid_bloomfilter = grid_bloomfilter,
    };

    GeneratedStringsArgs generator_args{
        .char_gen = get_enum_value<CharGenerator>(char_generator),
        .string_gen = get_enum_value<StringGenerator>(string_generator),
        .num_chars = num_chars,
        .step = step,
        .len_strings = len_strings,
        .difference_cover = difference_cover,
        .path = path,
    };

    std::vector<size_t> levels(levels_param.size());
    auto stoi = [](auto& str) { return std::stoi(str); };
    std::transform(levels_param.begin(), levels_param.end(), levels.begin(), stoi);

    tlx_die_verbose_unless(
        std::is_sorted(levels.begin(), levels.end(), std::greater_equal<>{}),
        "the given group sizes must be decreasing"
    );

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    measuring_tool.disableCommVolume();

    // todo does this make sense? maybe discard first round instead?
    dss_schimek::mpi::randomDataAllToAllExchange(100000u);

    measuring_tool.enableCommVolume();

    for (size_t i = 0; i < num_iterations; ++i) {
        SorterArgs args{
            .experiment = experiment,
            .sampling_factor = sampling_factor,
            .quantile_size = quantile_size,
            .check = check,
            .check_exhaustive = check_exhaustive,
            .iteration = i,
            .generator_args = generator_args,
            .levels = levels};
        arg1<unsigned char>(key, args);
    }

    return EXIT_SUCCESS;
}
