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
#include "executables/serialization.hpp"
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
    dn_ratio,
    dn_ratio_random,
    file,
    sentinel,
};

EnumNames<StringGenerator> const string_generator_names{
    {"dn-ratio", StringGenerator::dn_ratio},
    {"dn-ratio-random", StringGenerator::dn_ratio_random},
    {"file", StringGenerator::file},
};

template <typename Json>
void to_json(Json& json, StringGenerator const value) {
    json = enum_name(string_generator_names, value);
}

enum class Permutation { simple = 0, multi_level, sentinel };

EnumNames<Permutation> const permutation_names{
    {"simple", Permutation::simple},
    {"multi-level", Permutation::multi_level},
};

template <typename Json>
void to_json(Json& json, Permutation const value) {
    json = enum_name(permutation_names, value);
}

struct SorterArgs : public CommonArgs {
    StringGenerator string_generator = StringGenerator::dn_ratio;
    Permutation permutation = Permutation::simple;
    size_t num_strings = 100000;
    size_t len_strings = 100;
    size_t len_strings_min = len_strings;
    size_t len_strings_max = len_strings + 10;
    std::string path;
    // for the file generator: cap the number of bytes read from the file (0 = read the whole file)
    size_t max_num_bytes = 0;
    double dn_ratio = 0.5;
    bool dn_encode_padding = false;
    // skewed_dn_length: the fraction of the smallest strings whose length is drawn from an
    // interval that is skew_factor times longer, and which PE a string is generated on
    double skew_fraction = 0.0;
    double skew_factor = 1.0;
    // skewed_dn_length: pad the distinguishing prefix with a single constant character instead of
    // the tiled per-group encoding
    bool use_uniform_prefix = false;
    dss_mehnert::IdPlacement id_placement = dss_mehnert::IdPlacement::random;
    // skewed_dn_length: reproduce on a single PE the input a run with this many PEs would produce,
    // PE by PE in rank order (0 = generate normally)
    size_t simulate_num_pes = 0;
    size_t iteration = 0;
    bool strong_scaling = false;
    // number of irregular alltoallv warmup rounds run before each sort (0 = no warmup)
    size_t mpi_warmup_rounds = 0;
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

    // the number of PEs the input is generated for: the simulated count when a run is being
    // reproduced on a single PE, the actual one otherwise
    size_t generating_pes(dss_mehnert::Communicator const& comm) const {
        return simulate_num_pes != 0 ? simulate_num_pes : comm.size();
    }

    size_t scaled_strings(dss_mehnert::Communicator const& comm) const {
        return (strong_scaling ? 1 : generating_pes(comm)) * num_strings;
    }
};

template <typename StringSet>
auto generate_strings(SorterArgs const& args, dss_mehnert::Communicator const& comm) {
    using namespace dss_mehnert;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto input_container = [&]() -> StringLcpContainer<StringSet> {
        switch (args.string_generator) {
            case StringGenerator::dn_ratio: {
                return DNRatioGenerator<StringSet>{
                    args.scaled_strings(comm),
                    args.len_strings,
                    args.dn_ratio,
                    comm,
                    args.dn_encode_padding
                };
            }
            case StringGenerator::dn_ratio_random: {
                SkewedDNArgs const gen_args{
                    .global_strings = args.scaled_strings(comm),
                    .min_length = args.len_strings_min,
                    .max_length = args.len_strings_max,
                    .use_uniform_prefix = args.use_uniform_prefix,
                    .dn_ratio = args.dn_ratio,
                    .skew_fraction = args.skew_fraction,
                    .skew_factor = args.skew_factor,
                    .placement = args.id_placement,
                    .seed = args.seed,
                };
                if (args.simulate_num_pes != 0) {
                    tlx_die_verbose_unless(
                        comm.size() == 1,
                        "--input-simulate-num-pes reproduces a distributed run on a "
                        "single PE, so "
                        "it has to be run with a single MPI rank"
                    );
                    return SkewedDNRatioLengthGenerator<StringSet>::simulate(
                        gen_args,
                        args.simulate_num_pes
                    );
                }
                return SkewedDNRatioLengthGenerator<StringSet>{gen_args, comm};
            }
            case StringGenerator::file: {
                check_path_exists(args.path);
                return FileDistributer<StringSet>{args.path, comm, args.max_num_bytes};
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
        [[maybe_unused]] std::size_t volatile warmup_sink = 0;
        for (std::size_t i = 0; i < args.mpi_warmup_rounds; ++i) {
            kamping::measurements::timer().synchronize_and_start("warmup-round");
            warmup_sink += mpi_irregular_warmup(50000, 50500, comms.comm_root());
            kamping::measurements::timer().stop_and_append();
        }

        comm.barrier();
        measuring_tool.start("none", "sorting_overall");
        kamping::measurements::timer().synchronize_and_start("sorting_overall");
        MergeSort merge_sort{
            dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
                args.sampler.scaled_to_levels(get_num_levels(args.levels, comm)),
                args.get_splitter_sorter(),
                args.get_local_sorter(),
                args.alltoallv_params()
            ),
            std::move(redistribution),
            args.alltoallv_params(),
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

        measuring_tool.write_on_root(*args.measurement_output, comm);
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
                args.get_local_sorter(),
                args.alltoallv_params()
            ),
            std::move(redistribution),
            args.bloomfilter_base_case,
            args.bloomfilter_level_dedup,
            args.alltoallv_params(),
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

        measuring_tool.write_on_root(*args.measurement_output, comm);
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

    switch (args.permutation) {
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
    std::string& timer_json_path,
    std::vector<std::string>& levels_param,
    size_t& cpus_per_node,
    size_t& num_levels
) {
    add_common_args(args, app);

    // -- Input ----------------------------------------------------------------
    app.add_option("--input-generator", args.string_generator, "type of string generation to use")
        ->transform(
            CLI::CheckedTransformer(string_generator_names, CLI::ignore_case)
                .description(enum_value_list(string_generator_names))
        )
        ->default_str(enum_name(string_generator_names, args.string_generator))
        ->group("Input");
    app.add_option("--permutation", args.permutation, "type of permutation to use for PDMS")
        ->transform(
            CLI::CheckedTransformer(permutation_names, CLI::ignore_case)
                .description(enum_value_list(permutation_names))
        )
        ->default_str(enum_name(permutation_names, args.permutation))
        ->group("Input");
    app.add_option("--input-path", args.path, "path to input file")->group("Input");
    app.add_option(
           "--input-max-num-bytes",
           args.max_num_bytes,
           "for the file generator, truncate the input to at most this many bytes (0 = whole file)"
    )
        ->group("Input");
    app.add_option("--input-generator-DN-ratio", args.dn_ratio, "D/N ratio of generated strings")
        ->group("Input");
    app.add_flag(
           "--input-dn-encode-padding",
           args.dn_encode_padding,
           "for DNGen, fill the padding with repeated blocks encoding (string-id / 3) instead of a "
           "constant character; keeps the distinguishing prefix but varies the bloom filter hashes"
    )
        ->group("Input");
    app.add_option(
           "--input-generator-num-strings",
           args.num_strings,
           "number of strings to be generated"
    )
        ->group("Input");
    app.add_option(
           "--input-generator-length-strings",
           args.len_strings,
           "length of generated strings"
    )
        ->group("Input");
    app.add_option(
           "--input-generator-min-length-strings",
           args.len_strings_min,
           "minimum length of generated strings"
    )
        ->group("Input");
    app.add_option(
           "--input-generator-max-length-strings",
           args.len_strings_max,
           "maximum length of generated strings"
    )
        ->group("Input");
    app.add_option(
           "--input-generator-skew-fraction",
           args.skew_fraction,
           "for skewedDNLenGen, the fraction of the smallest strings that are stretched; their "
           "length is drawn from [min-len-strings, skew-factor * max-len-strings]"
    )
        ->group("Input");
    app.add_option(
           "--input-generator-skew-factor",
           args.skew_factor,
           "for skewedDNLenGen, the factor by which the stretched strings may be longer"
    )
        ->group("Input");
    app.add_flag(
           "--input-generator-use-uniform-prefix",
           args.use_uniform_prefix,
           "for skewedDNLenGen, pad the distinguishing prefix with a single constant character "
           "instead of the tiled per-group encoding"
    )
        ->group("Input");
    app.add_option(
           "--input-generator-placement",
           args.id_placement,
           "for skewedDNLenGen, which PE a string is generated on; with contiguous placement the "
           "stretched (smallest) strings all land on the low ranks, so the input itself is "
           "imbalanced in characters"
    )
        ->transform(
            CLI::CheckedTransformer(dss_mehnert::id_placement_names, CLI::ignore_case)
                .description(enum_value_list(dss_mehnert::id_placement_names))
        )
        ->default_str(enum_name(dss_mehnert::id_placement_names, args.id_placement))
        ->group("Input");
    app.add_option(
           "--input-generator-simulate-num-pes",
           args.simulate_num_pes,
           "for skewedDNLenGen, generate on a single PE the input a run with this many PEs would "
           "produce, PE by PE in rank order (0 = generate normally). Pass the command line of that "
           "run with this option added to sort the very same input, e.g. as a shared memory "
           "baseline; requires a single MPI rank"
    )
        ->group("Input");
    app.add_flag("--strong-scaling", args.strong_scaling, "perform a strong scaling experiment")
        ->group("General");
    app.add_option(
           "--mpi-warmup-rounds",
           args.mpi_warmup_rounds,
           "number of irregular alltoallv warmup rounds to run before each sort (0 = no warmup)"
    )
        ->group("General");

    // -- Multi-level ----------------------------------------------------------
    app.add_option("--group-size", levels_param, "size of groups for multi-level merge sort")
        ->group("Multi-level");
    app.add_option("--cpus-per-node", cpus_per_node, "number of cpus per node (default 48)")
        ->group("Multi-level");
    app.add_option("--num-levels", num_levels, "number of levels (default 1)")
        ->group("Multi-level");

    // -- Output ---------------------------------------------------------------
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

    std::string timer_json_path;
    std::vector<std::string> levels_param;
    size_t cpus_per_node = 48;
    size_t num_levels = 1;

    add_sorter_args(
        args,
        app,
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
    // the RESULT records go next to the timer JSON, rather than into a redirected stdout
    std::ofstream measurement_file;
    if (!timer_json_path.empty() && kamping::comm_world().is_root()) {
        auto path = std::filesystem::path(timer_json_path);
        if (path.extension() == ".json") {
            path.replace_extension();
        }
        path += "_additional_measurements.txt";

        measurement_file.open(path);
        tlx_die_verbose_unless(measurement_file, "could not open '" << path.string() << "'");
        args.measurement_output = &measurement_file;
    }
    run_algo();

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
        config["input"]["max-num-bytes"] = args.max_num_bytes;
        config["input"]["num-strings"] = args.num_strings;
        config["input"]["length-strings"] = args.len_strings;
        config["input"]["min-len-strings"] = args.len_strings_min;
        config["input"]["max-len-strings"] = args.len_strings_max;
        config["input"]["DN-ratio"] = args.dn_ratio;
        config["input"]["dn-encode-padding"] = args.dn_encode_padding;
        config["input"]["use-uniform-prefix"] = args.use_uniform_prefix;
        config["input"]["skew-fraction"] = args.skew_fraction;
        config["input"]["skew-factor"] = args.skew_factor;
        config["input"]["placement"] = args.id_placement;
        // the run being reproduced; `p` above is 1 for a simulated run, so record it separately
        config["input"]["simulate-num-pes"] = args.simulate_num_pes;

        config["num-iterations"] = args.num_iterations;
        config["mpi-warmup-rounds"] = args.mpi_warmup_rounds;
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
        config["local-sorter"] = args.get_local_sorter();
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
        config["alltoall"] = enum_name(alltoall_names, args.get_alltoall_routine());
        config["alltoall_large_counts"] = args.alltoall_large_counts;
        config["alltoall_onefactor_num_slots"] = args.onefactor_num_slots;
        config["alltoall_onefactor_synchronized"] = args.onefactor_synchronized;
        config["alltoall_onefactor_use_issend"] = args.onefactor_use_issend;
        config["redistribution"] = enum_name(redistribution_names, args.get_redistribution());
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
