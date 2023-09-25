// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
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
#include "options.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/multi_level.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"
#include "util/string_generator.hpp"
#include "variant_selection.hpp"

struct GeneratedStringsArgs {
    size_t num_strings;
    size_t len_strings;
    size_t len_strings_min;
    size_t len_strings_max;
    double DN_ratio;
    std::string path;
};

struct SorterArgs {
    std::string experiment;
    size_t num_strings;
    size_t sampling_factor;
    bool check;
    bool check_exhaustive;
    size_t iteration;
    bool strong_scaling;
    GeneratedStringsArgs generator_args;
    std::vector<size_t> levels;
};

struct CombinationKey {
    PolicyEnums::StringGenerator string_generator;
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
    bool space_efficient;
};

template <typename StringSet>
auto generate_strings(
    CombinationKey const& key, SorterArgs const& args, dss_mehnert::Communicator const& comm
) {
    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();

    if (comm.is_root()) {
        std::cout << "string generation started " << std::endl;
    }

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto check_path_exists = [](std::string const& path) {
        tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
    };

    auto input_container = [=]() -> dss_mehnert::StringLcpContainer<StringSet> {
        using namespace dss_schimek;

        auto const& gen_args = args.generator_args;
        auto const num_strings = (args.strong_scaling ? 1 : comm.size()) * gen_args.num_strings;
        auto const len_strings = gen_args.len_strings;
        auto const DN_ratio = gen_args.DN_ratio;
        auto const& path = gen_args.path;

        switch (key.string_generator) {
            case PolicyEnums::StringGenerator::skewedRandomStringLcpContainer: {
                tlx_die("not implemented");
            }
            case PolicyEnums::StringGenerator::DNRatioGenerator: {
                return DNRatioGenerator<StringSet>{num_strings, len_strings, DN_ratio};
            }
            case PolicyEnums::StringGenerator::File: {
                check_path_exists(path);
                return FileDistributer<StringSet>{path};
            }
            case PolicyEnums::StringGenerator::SkewedDNRatioGenerator: {
                return SkewedDNRatioGenerator<StringSet>{num_strings, len_strings, DN_ratio};
            }
            case PolicyEnums::StringGenerator::SuffixGenerator: {
                check_path_exists(path);
                return SuffixGenerator<StringSet>{path};
            }
        };
        tlx_die("invalid string generator");
    }();
    measuring_tool.stop("generate_strings");

    comm.barrier();
    if (comm.is_root()) {
        std::cout << "string generation completed " << std::endl;
    }

    auto const num_gen_chars = input_container.char_size();
    auto const num_gen_strs = input_container.size();
    measuring_tool.add(num_gen_chars - num_gen_strs, "input_chars");
    measuring_tool.add(num_gen_strs, "input_strings");

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
void run_merge_sort(
    CombinationKey const& key,
    SorterArgs const& args,
    std::string prefix,
    dss_mehnert::Communicator const& comm
) {
    using namespace dss_schimek;

    constexpr bool lcp_compression = LcpCompression();
    constexpr bool prefix_compression = PrefixCompression();

    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;
    using AllToAllPolicy =
        mpi::AllToAllStringImpl<lcp_compression, prefix_compression, StringSet, MPIAllToAllRoutine>;

    using MergeSort = dss_mehnert::sorter::DistributedMergeSort<
        StringLcpPtr,
        Subcommunicators,
        RedistributionPolicy,
        AllToAllPolicy,
        SamplePolicy,
        PartitionPolicy>;

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(false);
    measuring_tool.disableCommVolume();

    auto input_container = generate_strings<StringSet>(key, args, comm);
    auto num_input_chars = input_container.char_size();
    auto num_input_strs = input_container.size();

    CheckerWithCompleteExchange<StringLcpPtr> checker;
    if (args.check || args.check_exhaustive) {
        checker.storeLocalInput(input_container.raw_strings());
    }

    // skip unused levels for multi-level merge sort
    auto pred = [&](auto const& group_size) { return group_size < comm.size(); };
    auto first_level = std::find_if(args.levels.begin(), args.levels.end(), pred);

    measuring_tool.enableCommVolume();
    comm.barrier();

    measuring_tool.start("none", "sorting_overall");

    measuring_tool.start("none", "create_communicators");
    Subcommunicators comms{first_level, args.levels.end(), comm};
    measuring_tool.stop("none", "create_communicators", comm);

    MergeSort merge_sort{SamplePolicy{args.sampling_factor}};
    auto sorted_container = merge_sort.sort(std::move(input_container), comms);

    measuring_tool.stop("none", "sorting_overall", comm);

    if (args.check || args.check_exhaustive) {
        auto sorted_str_ptr = sorted_container.make_string_lcp_ptr();
        auto num_sorted_chars = sorted_container.char_size();
        auto num_sorted_strs = sorted_container.size();

        bool is_sorted = is_complete_and_sorted(
            sorted_str_ptr,
            num_input_chars,
            num_sorted_chars,
            num_input_strs,
            num_sorted_strs,
            comm
        );
        die_verbose_unless(is_sorted, "output is not sorted or missing characters");

        if (args.check_exhaustive) {
            bool is_complete = checker.check(sorted_str_ptr, true);
            die_verbose_unless(is_complete, "output is not a permutation of the input");
        }
    }

    measuring_tool.write_on_root(std::cout, comm);
    measuring_tool.reset();
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
void run_prefix_doubling(
    CombinationKey const& key,
    SorterArgs const& args,
    std::string prefix,
    dss_mehnert::Communicator const& comm
) {
    using namespace dss_schimek;

    constexpr bool lcp_compression = LcpCompression();
    constexpr bool prefix_compression = PrefixCompression();

    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;
    using AllToAllPolicy = mpi::AllToAllStringImplPrefixDoubling<
        lcp_compression,
        prefix_compression,
        StringSet,
        MPIAllToAllRoutine>;

    using MergeSort = dss_mehnert::sorter::PrefixDoublingMergeSort<
        StringLcpPtr,
        Subcommunicators,
        RedistributionPolicy,
        AllToAllPolicy,
        SamplePolicy,
        PartitionPolicy,
        BloomFilterPolicy>;

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(false);

    measuring_tool.disableCommVolume();

    auto input_container = generate_strings<StringSet>(key, args, comm);
    auto num_input_chars = input_container.char_size();
    auto num_input_strs = input_container.size();

    CheckerWithCompleteExchange<StringLcpPtr> checker;
    if (args.check || args.check_exhaustive) {
        checker.storeLocalInput(input_container.raw_strings());
    }

    // skip unused levels for multi-level merge sort
    auto pred = [&](auto const& group_size) { return group_size < comm.size(); };
    auto first_level = std::find_if(args.levels.begin(), args.levels.end(), pred);

    measuring_tool.enableCommVolume();

    comm.barrier();
    measuring_tool.start("none", "sorting_overall");

    measuring_tool.start("none", "create_communicators");
    Subcommunicators comms{first_level, args.levels.end(), comm};
    measuring_tool.stop("none", "create_communicators", comm);

    MergeSort merge_sort{SamplePolicy{args.sampling_factor}};
    auto permutation = merge_sort.sort(std::move(input_container), comms);

    measuring_tool.stop("none", "sorting_overall", comm);
    measuring_tool.disable();
    measuring_tool.disableCommVolume();

    if ((args.check || args.check_exhaustive) && comm.size() > 1) {
        auto complete_strings_cont = dss_mehnert::sorter::apply_permutation(
            StringLcpContainer<StringSet>{checker.getLocalInput()},
            permutation,
            comm
        );

        auto sorted_str_ptr = complete_strings_cont.make_string_lcp_ptr();
        auto num_sorted_chars = complete_strings_cont.char_size();
        auto num_sorted_strs = complete_strings_cont.size();

        bool is_complete_and_sorted = dss_schimek::is_complete_and_sorted(
            sorted_str_ptr,
            num_input_chars,
            num_sorted_chars,
            num_input_strs,
            num_sorted_strs,
            comm
        );

        die_verbose_unless(is_complete_and_sorted, "output is not sorted or missing characters");

        if (args.check_exhaustive) {
            bool const isSorted = checker.check(sorted_str_ptr, false);
            die_verbose_unless(isSorted, "output is not a permutation of the input");
        }
    }

    measuring_tool.write_on_root(std::cout, comm);
    measuring_tool.reset();
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
    tlx_die("not yet implemented");
}

std::string get_result_prefix(
    SorterArgs const& args, CombinationKey const& key, dss_mehnert::Communicator const& comm
) {
    // clang-format off
    return std::string("RESULT")
           + (args.experiment.empty() ? "" : (" experiment=" + args.experiment))
           + " num_procs="          + std::to_string(comm.size())
           + " num_strings="        + std::to_string(args.num_strings)
           + " len_strings="        + std::to_string(args.generator_args.len_strings)
           + " num_levels="         + std::to_string(args.levels.size())
           + " iteration="          + std::to_string(args.iteration)
           + " strong_scaling="     + std::to_string(args.strong_scaling)
           + " sample_chars="       + std::to_string(key.sample_chars)
           + " sample_indexed="     + std::to_string(key.sample_indexed)
           + " sample_random="      + std::to_string(key.sample_random)
           + " sampling_factor="    + std::to_string(args.sampling_factor)
           + " rquick_v1="          + std::to_string(key.rquick_v1)
           + " rquick_lcp="         + std::to_string(key.rquick_lcp)
           + " lcp_compression="    + std::to_string(key.lcp_compression)
           + " prefix_compression=" + std::to_string(key.prefix_compression)
           + " prefix_doubling="    + std::to_string(key.prefix_doubling)
           + " grid_bloomfilter="   + std::to_string(key.grid_bloomfilter)
           + " space_efficient="    + std::to_string(key.space_efficient)
           + " dn_ratio="           + std::to_string(args.generator_args.DN_ratio);
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

    if (key.space_efficient) {
        // todo this currently ignores the prefix doubling flag
        if constexpr (CliOptions::enable_space_efficient) {
            run_space_efficient_sort<Args...>(key, args, prefix, comm);
        } else {
            die_with_feature("CLI_ENABLE_SPACE_EFFICIENT");
        }
    } else if (key.prefix_doubling) {
        if constexpr (CliOptions::enable_prefix_doubling) {
            run_prefix_doubling<Args...>(key, args, prefix, comm);
        } else {
            die_with_feature("CLI_ENABLE_PREFIX_DOUBLING");
        }
    } else {
        run_merge_sort<Args...>(key, args, prefix, comm);
    }
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
    bool strong_scaling = false;
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
    bool space_efficient = false;
    unsigned int generator = static_cast<int>(PolicyEnums::StringGenerator::DNRatioGenerator);
    unsigned int alltoall_routine = static_cast<int>(PolicyEnums::MPIRoutineAllToAll::combined);
    unsigned int comm_split = static_cast<int>(PolicyEnums::Subcommunicators::grid);
    unsigned int redistribution = static_cast<int>(PolicyEnums::Redistribution::naive);
    size_t num_strings = 100000;
    size_t num_iterations = 5;
    size_t string_length = 50;
    size_t len_strings_min = string_length;
    size_t len_strings_max = string_length + 10;
    double DN_ratio = 0.5;
    std::string path;
    std::string experiment;
    std::vector<std::string> levels_param;

    tlx::CmdlineParser cp;
    cp.set_description("a distributed string sorter");
    cp.set_author("Matthias Schimek, Pascal Mehnert");
    cp.add_string('e', "experiment", experiment, "name to identify the experiment being run");
    cp.add_unsigned(
        'k',
        "generator",
        generator,
        "type of string generation to use "
        "(0=skewed, [1]=DNGen, 2=file, 3=skewedDNGen, 4=suffixGen)"
    );
    cp.add_string('y', "path", path, "path to input file");
    cp.add_double('r', "DN-ratio", DN_ratio, "D/N ratio of generated strings");
    cp.add_size_t('n', "num-strings", num_strings, "number of strings to be generated");
    cp.add_size_t('m', "len-strings", string_length, "length of generated strings");
    cp.add_size_t('b', "min-len-strings", len_strings_min, "minimum length of generated strings");
    cp.add_size_t('B', "max-len-strings", len_strings_max, "maximum length of generated strings");
    cp.add_size_t('i', "num-iterations", num_iterations, "number of sorting iterations to run");
    cp.add_flag('x', "strong-scaling", strong_scaling, "perform a strong scaling experiment");
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
    cp.add_flag('s', "space-efficient", space_efficient, "use space efficient sorting");
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
        .string_generator = PolicyEnums::getStringGenerator(generator),
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
        .space_efficient = space_efficient,
    };

    GeneratedStringsArgs generator_args{
        .num_strings = num_strings,
        .len_strings = string_length,
        .len_strings_min = len_strings_min,
        .len_strings_max = len_strings_max,
        .DN_ratio = DN_ratio,
        .path = path,
    };

    std::vector<size_t> levels(levels_param.size());
    auto stoi = [](auto& str) { return std::stoi(str); };
    std::transform(levels_param.begin(), levels_param.end(), levels.begin(), stoi);

    if (!std::is_sorted(levels.begin(), levels.end(), std::greater_equal<>{})) {
        tlx_die("the given group sizes must be decreasing");
    }

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    measuring_tool.disableCommVolume();

    // todo does this make sense? maybe discard first round instead?
    auto num_bytes = std::min<size_t>(num_strings * 5, 100000u);
    dss_schimek::mpi::randomDataAllToAllExchange(num_bytes);

    measuring_tool.enableCommVolume();

    for (size_t i = 0; i < num_iterations; ++i) {
        SorterArgs args{
            .experiment = experiment,
            .num_strings = num_strings,
            .sampling_factor = sampling_factor,
            .check = check,
            .check_exhaustive = check_exhaustive,
            .iteration = i,
            .strong_scaling = strong_scaling,
            .generator_args = generator_args,
            .levels = levels};
        arg1<unsigned char>(key, args);
    }

    return EXIT_SUCCESS;
}
