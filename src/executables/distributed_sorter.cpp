// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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

#include "executables/common_cli.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "options.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"
#include "util/string_generator.hpp"

enum class StringGenerator {
    skewed_random = 0,
    dn_ratio,
    file,
    skewed_dn_ratio,
    suffix,
    sentinel,
};

enum class Permutation { simple = 0, multi_level, sentinel };

struct SorterArgs : public CommonArgs {
    size_t string_generator = static_cast<size_t>(StringGenerator::dn_ratio);
    size_t permutation = static_cast<size_t>(Permutation::simple);
    size_t num_strings = 100000;
    size_t len_strings = 100;
    size_t len_strings_min = len_strings;
    size_t len_strings_max = len_strings + 10;
    std::string path;
    double dn_ratio = 0.5;
    size_t iteration = 0;
    bool strong_scaling = false;
    std::vector<size_t> levels;

    std::string get_prefix(dss_mehnert::Communicator const& comm) const {
        // clang-format off
        return CommonArgs::get_prefix(comm) 
               + " num_strings="    + std::to_string(num_strings)
               + " len_strings="    + std::to_string(len_strings)
               + " num_levels="     + std::to_string(levels.size())
               + " iteration="      + std::to_string(iteration)
               + " strong_scaling=" + std::to_string(strong_scaling)
               + " dn_ratio="       + std::to_string(dn_ratio);
        // clang-format on
    }

    size_t scaled_strings(dss_mehnert::Communicator const& comm) const {
        return (strong_scaling ? 1 : comm.size()) * num_strings;
    }
};

template <typename StringSet>
auto generate_strings(SorterArgs const& args, dss_mehnert::Communicator const& comm) {
    using namespace dss_mehnert;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto input_container = [&]() -> StringLcpContainer<StringSet> {
        switch (clamp_enum_value<StringGenerator>(args.string_generator)) {
            case StringGenerator::skewed_random: {
                tlx_die("not implemented");
            }
            case StringGenerator::dn_ratio: {
                return DNRatioGenerator<StringSet>{
                    args.scaled_strings(comm),
                    args.len_strings,
                    args.dn_ratio,
                    comm
                };
            }
            case StringGenerator::file: {
                check_path_exists(args.path);
                return FileDistributer<StringSet>{args.path, comm};
            }
            case StringGenerator::skewed_dn_ratio: {
                return SkewedDNRatioGenerator<StringSet>{
                    args.scaled_strings(comm),
                    args.len_strings,
                    args.dn_ratio,
                    comm
                };
            }
            case StringGenerator::suffix: {
                check_path_exists(args.path);
                return SuffixGenerator<StringSet>{args.path, comm};
            }
            case StringGenerator::sentinel: {
                break;
            }
        };
        tlx_die("invalid string generator");
    }();
    measuring_tool.stop("generate_strings");

    comm.barrier();

    auto const num_gen_chars = input_container.char_size();
    auto const num_gen_strs = input_container.size();
    measuring_tool.add(num_gen_chars - num_gen_strs, "input_chars");
    measuring_tool.add(num_gen_strs, "input_strings");

    return input_container;
}

template <typename CharType, typename AlltoallConfig, typename BloomFilterPolicy>
void run_merge_sort(
    SorterArgs const& args, std::string prefix, dss_mehnert::Communicator const& comm
) {
    constexpr auto alltoall_config = AlltoallConfig();
    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using PartitionPolicy = dss_mehnert::MergeSortPartitionPolicy<CharType>;

    auto dispatch = [&]<typename RedistributionPolicy>(RedistributionPolicy redistribution) {
        using Subcommunicators = RedistributionPolicy::Subcommunicators;
        using MergeSort = dss_mehnert::sorter::
            DistributedMergeSort<alltoall_config, RedistributionPolicy, PartitionPolicy>;

        using dss_mehnert::measurement::MeasuringTool;
        auto& measuring_tool = MeasuringTool::measuringTool();
        measuring_tool.setPrefix(prefix);
        measuring_tool.setVerbose(args.verbose);

        measuring_tool.disableCommVolume();
        auto input_container = generate_strings<StringSet>(args, comm);

        dss_mehnert::MergeSortChecker<StringSet> checker;
        if (args.check_sorted || args.check_complete) {
            checker.store_container(input_container);
        }
        measuring_tool.enableCommVolume();

        comm.barrier();

        measuring_tool.start("none", "create_communicators");
        auto const first_level = get_first_level(args.levels, comm);
        Subcommunicators comms{first_level, args.levels.end(), comm};
        measuring_tool.stop("none", "create_communicators", comm);

        measuring_tool.start("none", "sorting_overall");
        MergeSort merge_sort{
            dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
                args.sampler,
                args.get_splitter_sorter()
            ),
            std::move(redistribution)
        };
        merge_sort.sort(input_container, comms);
        measuring_tool.stop("none", "sorting_overall", comm);

        measuring_tool.disableCommVolume();

        if (args.count_prefixes) {
            count_prefix_lengths(input_container, comm);
        }

        measuring_tool.disable();

        if (args.check_sorted) {
            auto const is_sorted = checker.is_sorted(input_container.make_string_set(), comm);
            die_verbose_unless(is_sorted, "output is not sorted");
            auto const is_complete = checker.is_complete(input_container, comm);
            die_verbose_unless(is_complete, "output is missing chars or strings");
        }
        if (args.check_complete) {
            auto const is_exact = checker.check_exhaustive(input_container, comm);
            die_verbose_unless(is_exact, "output is not a permutation of the input");
        }

        measuring_tool.write_on_root(std::cout, comm);
        measuring_tool.reset();
    };

    dss_mehnert::dispatch_redistribution<StringSet>(dispatch, args);
}

template <
    typename CharType,
    typename AlltoallConfig,
    typename BloomFilterPolicy,
    typename Permutation>
void run_prefix_doubling(
    SorterArgs const& args, std::string prefix, dss_mehnert::Communicator const& comm
) {
    constexpr auto alltoall_config = AlltoallConfig();
    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::IntLength>;
    using PartitionPolicy =
        dss_mehnert::PrefixDoublingPartitionPolicy<CharType, dss_mehnert::IntLength, Permutation>;

    auto dispatch = [&]<typename RedistributionPolicy>(RedistributionPolicy redistribution) {
        using Subcommunicators = RedistributionPolicy::Subcommunicators;
        using MergeSort = dss_mehnert::sorter::prefix_doubling::PrefixDoublingMergeSort<
            alltoall_config,
            RedistributionPolicy,
            PartitionPolicy,
            BloomFilterPolicy,
            Permutation>;

        using dss_mehnert::measurement::MeasuringTool;
        auto& measuring_tool = MeasuringTool::measuringTool();
        measuring_tool.setPrefix(prefix);
        measuring_tool.setVerbose(args.verbose);

        measuring_tool.disableCommVolume();

        auto input_container = generate_strings<StringSet>(args, comm);

        dss_mehnert::PrefixDoublingChecker<StringSet> checker;
        if (args.check_sorted || args.check_complete) {
            checker.store_container(input_container);
        }
        measuring_tool.enableCommVolume();

        comm.barrier();

        measuring_tool.start("none", "create_communicators");
        auto const first_level = get_first_level(args.levels, comm);
        Subcommunicators comms{first_level, args.levels.end(), comm};
        measuring_tool.stop("none", "create_communicators", comm);

        measuring_tool.start("none", "sorting_overall");
        MergeSort merge_sort{
            dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
                args.sampler,
                args.get_splitter_sorter()
            ),
            std::move(redistribution)
        };
        auto permutation = merge_sort.sort(std::move(input_container), comms);
        measuring_tool.stop("none", "sorting_overall", comm);

        measuring_tool.disableCommVolume();

        measuring_tool.disable();

        if (args.check_sorted) {
            auto const is_sorted = checker.is_sorted(permutation, comms);
            die_verbose_unless(is_sorted, "output permutation is not sorted");
        }
        if (args.check_complete) {
            auto const is_complete = checker.is_complete(permutation, comms);
            die_verbose_unless(is_complete, "output permutation is not complete");
        }

        measuring_tool.write_on_root(std::cout, comm);
        measuring_tool.reset();
    };

    using AugmentedStringSet = dss_mehnert::sorter::AugmentedStringSet<StringSet, Permutation>;
    dss_mehnert::dispatch_redistribution<AugmentedStringSet>(dispatch, args);
}

template <typename... Args>
void dispatch_permutation(
    SorterArgs const& args, std::string prefix, dss_mehnert::Communicator const& comm
) {
    using namespace dss_mehnert;

    switch (clamp_enum_value<Permutation>(args.permutation)) {
        case Permutation::simple: {
            run_prefix_doubling<Args..., SimplePermutation>(args, prefix, comm);
            return;
        }
        case Permutation::multi_level: {
            run_prefix_doubling<Args..., MultiLevelPermutation>(args, prefix, comm);
            return;
        }
        case Permutation::sentinel: {
            break;
        }
    }
    tlx_die("invalid permutation");
}

template <typename CharType, typename... Args>
void dispatch_sorter(SorterArgs const& args) {
    static_assert(!CliOptions::use_shared_memory_sort);
    dss_mehnert::Communicator comm;

    // todo print config
    auto prefix = args.get_prefix(comm);

    if constexpr (CliOptions::use_rquick_sort) {
        using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
        run_rquick<StringSet>(args, prefix, comm, generate_strings<StringSet>);
    } else if (args.prefix_doubling) {
        if constexpr (CliOptions::enable_prefix_doubling) {
            dispatch_permutation<CharType, Args...>(args, prefix, comm);
        } else {
            die_with_feature("CLI_ENABLE_PREFIX_DOUBLING");
        }
    } else {
        run_merge_sort<CharType, Args...>(args, prefix, comm);
    }
}

// hack to integrate experiment flags in kaval
void set_experiment(SorterArgs& args, size_t num_levels) {
    std::vector<std::string> const template_values{"np", "dn"};
    auto it = std::find(template_values.begin(), template_values.end(), args.experiment);
    if (it == template_values.end()) {
        // return if no or custom name is given
        return;
    }
    std::string const prefix = args.experiment;
    if (CliOptions::use_rquick_sort) {
        args.experiment = prefix + "_ratio_rquick";
    } else {
        switch (num_levels) {
            case 1:
                args.experiment = prefix + "_ratio_single";
                break;
            case 2:
                args.experiment = prefix + "_ratio_double";
                break;
            case 3:
                args.experiment = prefix + "_ratio_triple_optimal";
                break;
            default:
                break;
        }
    }
}


int main(int argc, char* argv[]) {
    SorterArgs args;

    tlx::CmdlineParser cp;
    cp.set_description("a distributed string sorter");
    cp.set_author("Matthias Schimek, Pascal Mehnert");

    add_common_args(args, cp);

    cp.add_size_t(
        'k',
        "string-generator",
        args.string_generator,
        "type of string generation to use "
        "(0=skewed, [1]=DNGen, 2=file, 3=skewedDNGen, 4=suffixGen)"
    );
    cp.add_size_t(
        'o',
        "permutation",
        args.permutation,
        "type of permutation to use for PDMS "
        "([0]=simple, 1=multi-level)"
    );
    cp.add_string('y', "path", args.path, "path to input file");
    std::string output_path;
    cp.add_string("json_output_path", output_path, "path to output file");
    cp.add_double('r', "DN-ratio", args.dn_ratio, "D/N ratio of generated strings");
    cp.add_size_t('n', "num-strings", args.num_strings, "number of strings to be generated");
    cp.add_size_t('m', "len-strings", args.len_strings, "length of generated strings");
    cp.add_size_t(
        'b',
        "min-len-strings",
        args.len_strings_min,
        "minimum length of generated strings"
    );
    cp.add_size_t(
        'B',
        "max-len-strings",
        args.len_strings_max,
        "maximum length of generated strings"
    );
    cp.add_flag('x', "strong-scaling", args.strong_scaling, "perform a strong scaling experiment");

    std::vector<std::string> levels_param;
    cp.add_stringlist('Y', "group-size", levels_param, "size of groups for multi-level merge sort");

    size_t cpus_per_node = 48;
    cp.add_size_t("cpus-per-node", cpus_per_node, "number of cpus per node (default 48)");
    size_t num_levels = 1;
    cp.add_size_t("num-levels", num_levels, "number of levels (default 1)");


    if (!cp.process(argc, argv)) {
        return EXIT_FAILURE;
    }

    kamping::Environment env{argc, argv};

    if (levels_param.size() == 1 && levels_param.front() == "") {
        levels_param.clear();
    }
    parse_level_arg(cpus_per_node, num_levels, levels_param, args.levels);
    set_experiment(args, num_levels);

    auto run_algo = [&]() {
        if constexpr (CliOptions::use_shared_memory_sort) {
            using CharType = unsigned char;
            using String = dss_mehnert::SimpleString<CharType, CharType*>;
            using StringSet = dss_mehnert::GenericStringSet<String>;
            run_shared_memory(args, kamping::comm_world(), generate_strings<StringSet>);
        } else {
            for (size_t i = 0; i < args.num_iterations; ++i) {
                args.iteration = i;
                dispatch_common_args([&]<typename... T> { dispatch_sorter<T...>(args); }, args);
            }
        }
    };
    // redirect
    if (kamping::comm_world().is_root()) {
        std::ofstream out(output_path);
        std::streambuf* coutbuf = std::cout.rdbuf();
        std::cout.rdbuf(out.rdbuf());
        run_algo();
        std::cout.rdbuf(coutbuf);
    } else {
        run_algo();
    }

    return EXIT_SUCCESS;
}
