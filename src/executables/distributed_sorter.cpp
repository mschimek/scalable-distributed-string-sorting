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
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/spdlog_adapter/logging.hpp>
#include <spdlog/cfg/env.h>
#include <tlx/cmdline_parser.hpp>
#include <tlx/die.hpp>
#include <tlx/die/core.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "bench/reporting.hpp"
#include "executables/common_cli.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/print_strings.hpp"
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
    skewed_dn_length,
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
    bool dn_encode_padding = false;
    // skewed_dn_length: the fraction of the smallest strings whose length is drawn from an
    // interval that is skew_factor times longer, and which PE a string is generated on
    double skew_fraction = 0.0;
    double skew_factor = 1.0;
    // skewed_dn_length: pad the distinguishing prefix with a single constant character instead of
    // the tiled per-group encoding
    bool use_uniform_prefix = false;
    size_t id_placement = static_cast<size_t>(dss_mehnert::IdPlacement::random);
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
                    comm,
                    args.dn_encode_padding
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
            case StringGenerator::skewed_dn_length: {
                return SkewedDNRatioLengthGenerator<StringSet>{
                    {
                        .global_strings = args.scaled_strings(comm),
                        .min_length = args.len_strings_min,
                        .max_length = args.len_strings_max,
                        .use_uniform_prefix = args.use_uniform_prefix,
                        .dn_ratio = args.dn_ratio,
                        .skew_fraction = args.skew_fraction,
                        .skew_factor = args.skew_factor,
                        .placement = clamp_enum_value<IdPlacement>(args.id_placement),
                        .seed = args.seed,
                    },
                    comm
                };
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

    using kamping::measurements::GlobalAggregationMode;
    std::vector<GlobalAggregationMode> const agg{
        GlobalAggregationMode::min,
        GlobalAggregationMode::max,
        GlobalAggregationMode::sum,
    };
    kamping::measurements::counter()
        .add("input_chars", static_cast<std::int64_t>(num_gen_chars - num_gen_strs), agg);
    kamping::measurements::counter()
        .add("input_strings", static_cast<std::int64_t>(num_gen_strs), agg);

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
        kamping::measurements::timer().synchronize_and_start("create_communicators");
        auto const first_level = get_first_level(args.levels, comm);
        Subcommunicators comms{first_level, args.levels.end(), comm};
        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("none", "create_communicators", comm);

        measuring_tool.start("none", "sorting_overall");
        kamping::measurements::timer().synchronize_and_start("sorting_overall");
        MergeSort merge_sort{
            dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
                args.sampler.scaled_to_levels(get_num_levels(args.levels, comm)),
                args.get_splitter_sorter(),
                args.get_local_sorter()
            ),
            std::move(redistribution),
            args.onefactor_params(),
            args.get_local_sorter()
        };
        merge_sort.sort(input_container, comms, args.sampler.splitter_length_factor);
        kamping::measurements::timer().stop_and_append();
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
        if (args.print_sorted) {
            dss_mehnert::gather_and_print_strings(input_container, comm);
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

        // prefix doubling only returns a permutation; keep a copy of the local
        // input so we can materialize the globally sorted strings for printing
        dss_mehnert::StringLcpContainer<StringSet> input_copy;
        if (args.print_sorted) {
            dss_mehnert::copy_container(input_container, input_copy);
        }
        measuring_tool.enableCommVolume();

        comm.barrier();

        measuring_tool.start("none", "create_communicators");
        kamping::measurements::timer().synchronize_and_start("create_communicators");
        auto const first_level = get_first_level(args.levels, comm);
        Subcommunicators comms{first_level, args.levels.end(), comm};
        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("none", "create_communicators", comm);

        measuring_tool.start("none", "sorting_overall");
        kamping::measurements::timer().synchronize_and_start("sorting_overall");
        MergeSort merge_sort{
            dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
                args.sampler.scaled_to_levels(get_num_levels(args.levels, comm)),
                args.get_splitter_sorter(),
                args.get_local_sorter()
            ),
            std::move(redistribution),
            args.bloomfilter_base_case,
            args.bloomfilter_level_dedup,
            args.onefactor_params(),
            args.get_local_sorter()
        };
        auto permutation = merge_sort.sort(std::move(input_container), comms);
        kamping::measurements::timer().stop_and_append();
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
        if (args.print_sorted) {
            if constexpr (std::is_same_v<Permutation, dss_mehnert::SimplePermutation>) {
                auto sorted_container = dss_mehnert::sorter::prefix_doubling::apply_permutation(
                    input_copy.make_string_set(),
                    permutation,
                    comm
                );
                dss_mehnert::gather_and_print_strings(sorted_container, comm);
            } else if (comm.is_root()) {
                std::cout << "--print-sorted is only supported for the simple permutation\n";
            }
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
            dss_mehnert::die_with_feature("CLI_ENABLE_PREFIX_DOUBLING");
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


// CLI11 counterpart to the tlx parsing in main(), kept in parallel so it can be
// verified before switching over. Registers the shared options via the CLI11
// add_common_args overload and adds the distributed_sorter-specific options in
// named groups. To switch main() over: build a CLI::App, call this, then
// CLI11_PARSE(app, argc, argv) instead of cp.process().
void add_sorter_args(
    SorterArgs& args,
    CLI::App& app,
    std::string& output_path,
    std::string& timer_json_path,
    std::vector<std::string>& levels_param,
    size_t& cpus_per_node,
    size_t& num_levels
) {
    add_common_args(args, app);

    // -- Input ----------------------------------------------------------------
    app.add_option(
           "--string-generator",
           args.string_generator,
           "type of string generation to use "
           "(0=skewed, [1]=DNGen, 2=file, 3=skewedDNGen, 4=suffixGen, 5=skewedDNLenGen)"
    )
        ->group("Input");
    app.add_option(
           "--permutation",
           args.permutation,
           "type of permutation to use for PDMS ([0]=simple, 1=multi-level)"
    )
        ->group("Input");
    app.add_option("--path", args.path, "path to input file")->group("Input");
    app.add_option("--DN-ratio", args.dn_ratio, "D/N ratio of generated strings")->group("Input");
    app.add_flag(
           "--dn-encode-padding",
           args.dn_encode_padding,
           "for DNGen, fill the padding with repeated blocks encoding (string-id / 3) instead of a "
           "constant character; keeps the distinguishing prefix but varies the bloom filter hashes"
    )
        ->group("Input");
    app.add_option("--num-strings", args.num_strings, "number of strings to be generated")
        ->group("Input");
    app.add_option("--len-strings", args.len_strings, "length of generated strings")
        ->group("Input");
    app.add_option("--min-len-strings", args.len_strings_min, "minimum length of generated strings")
        ->group("Input");
    app.add_option("--max-len-strings", args.len_strings_max, "maximum length of generated strings")
        ->group("Input");
    app.add_option(
           "--skew-fraction",
           args.skew_fraction,
           "for skewedDNLenGen, the fraction of the smallest strings that are stretched; their "
           "length is drawn from [min-len-strings, skew-factor * max-len-strings]"
    )
        ->group("Input");
    app.add_option(
           "--skew-factor",
           args.skew_factor,
           "for skewedDNLenGen, the factor by which the stretched strings may be longer"
    )
        ->group("Input");
    app.add_flag(
           "--input-use-uniform-prefix",
           args.use_uniform_prefix,
           "for skewedDNLenGen, pad the distinguishing prefix with a single constant character "
           "instead of the tiled per-group encoding"
    )
        ->group("Input");
    app.add_option(
           "--placement",
           args.id_placement,
           "for skewedDNLenGen, which PE a string is generated on ([0]=random, 1=contiguous); "
           "with contiguous placement the stretched (smallest) strings all land on the low ranks, "
           "so the input itself is imbalanced in characters"
    )
        ->group("Input");
    app.add_flag("--strong-scaling", args.strong_scaling, "perform a strong scaling experiment")
        ->group("Input");

    // -- Multi-level ----------------------------------------------------------
    app.add_option("--group-size", levels_param, "size of groups for multi-level merge sort")
        ->group("Multi-level");
    app.add_option("--cpus-per-node", cpus_per_node, "number of cpus per node (default 48)")
        ->group("Multi-level");
    app.add_option("--num-levels", num_levels, "number of levels (default 1)")
        ->group("Multi-level");

    // -- Output ---------------------------------------------------------------
    app.add_option("--json_output_path", output_path, "path to output file")->group("Output");
    app.add_option(
           "--timer-json-path",
           timer_json_path,
           "path for the kamping timer JSON report (empty = disabled)"
    )
        ->group("Output");
}

int main(int argc, char* argv[]) {
    SorterArgs args;

    CLI::App app{"a distributed string sorter"};
    app.option_defaults()->always_capture_default();

    std::string output_path;
    std::string timer_json_path;
    std::vector<std::string> levels_param;
    size_t cpus_per_node = 48;
    size_t num_levels = 1;

    add_sorter_args(
        args,
        app,
        output_path,
        timer_json_path,
        levels_param,
        cpus_per_node,
        num_levels
    );

    CLI11_PARSE(app, argc, argv);

    kamping::Environment env{argc, argv};

    // log level comes from the SPDLOG_LEVEL env var, e.g. SPDLOG_LEVEL=debug
    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();

    if (levels_param.size() == 1 && levels_param.front() == "") {
        levels_param.clear();
    }
    parse_level_arg(cpus_per_node, num_levels, levels_param, args.levels);
    set_experiment(args, num_levels);

    dss_mehnert::Report report;
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
                // aggregate this iteration's kamping timer tree and reset it
                report.step_iteration();
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

    if (!timer_json_path.empty() && kamping::comm_world().is_root()) {
        nlohmann::ordered_json config;
        config["p"] = kamping::comm_world().size();
        config["experiment"] = args.experiment;
        config["i_mpi_adjust_alltoallv"] = [] {
            char* val = std::getenv("I_MPI_ADJUST_ALLTOALLV");
            if (val == nullptr) {
                return std::string{};
            }
            return std::string{val};
        }();
        config["i_mpi_adjust_allgatherv"] = [] {
            char* val = std::getenv("I_MPI_ADJUST_ALLGATHERV");
            if (val == nullptr) {
                return std::string{};
            }
            return std::string{val};
        }();

        config["input"]["string-generator"] = args.string_generator;
        config["input"]["path"] = args.path;
        config["input"]["num-strings"] = args.num_strings;
        config["input"]["length-strings"] = args.len_strings;
        config["input"]["min-len-strings"] = args.len_strings_min;
        config["input"]["max-len-strings"] = args.len_strings_max;
        config["input"]["DN-ratio"] = args.dn_ratio;
        config["input"]["dn-encode-padding"] = args.dn_encode_padding;
        config["input"]["use-uniform-prefix"] = args.use_uniform_prefix;

        config["num-iterations"] = args.num_iterations;
        config["permutation"] = args.permutation;
        config["num-levels"] = num_levels;
        config["cpus-per-node"] = cpus_per_node;
        config["group-size"] = args.levels;

        config["sample-chars"] = args.sampler.sample_chars;
        config["sample-indexed"] = args.sampler.sample_indexed;
        config["sample-random"] = args.sampler.sample_random;
        config["sampling-factor"] = args.sampler.sampling_factor;
        config["splitter-length-factor"] = args.sampler.splitter_length_factor;
        config["redistribute-sample"] = args.sampler.redistribute_sample;
        config["level-adjusted-scaling"] = args.sampler.level_adjusted_scaling;
        config["local-sorter"] = args.local_sorter;
        config["splitter-sequential"] = args.splitter_sequential;

        config["rquick-v1"] = args.rquick_v1;
        config["rquick-lcp"] = args.rquick_lcp;
        config["long-filter"] = args.long_filter;
        config["prefix-doubling"] = args.prefix_doubling;
        config["grid-bloomfilter"] = args.grid_bloomfilter;
        config["bloomfilter-base-case"] = args.bloomfilter_base_case;
        config["bloomfilter-level-dedup"] = args.bloomfilter_level_dedup;
        config["lcp-compression"] = args.lcp_compression;
        config["prefix-compression"] = args.prefix_compression;
        config["alltoall"] = args.alltoall_routine;
        config["alltoall_onefactor_num_slots"] = args.onefactor_num_slots;
        config["alltoall_onefactor_synchronized"] = args.onefactor_synchronized;
        config["alltoall_onefactor_use_issend"] = args.onefactor_use_issend;
        config["redistribution"] = args.redistribution;
        config["strong-scaling"] = args.strong_scaling;

        config["check-sorted"] = args.check_sorted;
        config["check-complete"] = args.check_complete;
        config["count-prefixes"] = args.count_prefixes;
        config["print-sorted"] = args.print_sorted;
        config["verbose"] = args.verbose;

        report.push_config(config);

        auto out = dss_mehnert::make_output_stream(timer_json_path);
        report.print(*out);
    }

    return EXIT_SUCCESS;
}
