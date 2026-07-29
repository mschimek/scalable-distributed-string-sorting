// (c) 2023 Pascal Mehnert
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The command line of distributed_sorter: the arguments it parses into, and the CLI11 options
// that fill them.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <kamping/communicator.hpp>
#include <tlx/die/core.hpp>

#include "executables/serialization.hpp"
#include "mpi/alltoallv/params.hpp"
#include "mpi/communicator.hpp"
#include "options.hpp"
#include "sorter/distributed/partition_policy_factory.hpp"
#include "sorter/local_sorter.hpp"
#include "util/string_generator.hpp"

enum class MPIRoutineAllToAll { native = 0, direct, onefactor, pairwise, sentinel };

inline EnumNames<MPIRoutineAllToAll> const alltoall_names{
    {"native", MPIRoutineAllToAll::native},
    {"direct", MPIRoutineAllToAll::direct},
    {"onefactor", MPIRoutineAllToAll::onefactor},
    {"pairwise", MPIRoutineAllToAll::pairwise},
};

// clang-format off
enum class Redistribution { none = 0, naive, simple_strings, simple_chars,
                            det_strings, det_chars, grid, sentinel };
// clang-format on

inline EnumNames<Redistribution> const redistribution_names{
    {"none", Redistribution::none},
    {"naive", Redistribution::naive},
    {"simple-strings", Redistribution::simple_strings},
    {"simple-chars", Redistribution::simple_chars},
    {"det-strings", Redistribution::det_strings},
    {"det-chars", Redistribution::det_chars},
    {"grid", Redistribution::grid},
};

struct CommonArgs {
    std::string experiment;
    MPIRoutineAllToAll alltoall_routine = MPIRoutineAllToAll::native;
    bool alltoall_large_counts = true;
    size_t onefactor_num_slots = 16;
    bool onefactor_use_issend = false;
    bool onefactor_synchronized = false;
    dss_mehnert::SamplerArgs sampler;
    bool rquick_v1 = false;
    bool rquick_lcp = false;
    bool long_filter = false;
    bool splitter_sequential = false;
    Redistribution redistribution = Redistribution::grid;
    dss_mehnert::LocalSorter local_sorter = dss_mehnert::LocalSorter::multikey_quicksort;
    bool prefix_compression = false;
    bool lcp_compression = false;
    bool prefix_doubling = false;
    bool grid_bloomfilter = true;
    bool bloomfilter_base_case = false;
    bool bloomfilter_level_dedup = true;
    size_t num_iterations = 5;
    bool check_sorted = false;
    bool check_complete = false;
    bool verbose = false;
    bool count_prefixes = false;
    bool print_sorted = false;
    // base seed for input generation; the same seed reproduces the same input
    size_t seed = 42;
    // where MeasuringTool writes its RESULT records; main points this at a file when one is
    // requested, and only on the root PE, which is the only one that writes
    std::ostream* measurement_output = &std::cout;

    std::string get_prefix(dss_mehnert::Communicator const& comm) const {
        // clang-format off
        return std::string("RESULT")
               + (experiment.empty() ? "" : (" experiment=" + experiment))
               + " num_procs="          + std::to_string(comm.size())
               + " sample_chars="       + std::to_string(sampler.sample_chars)
               + " sample_indexed="     + std::to_string(sampler.sample_indexed)
               + " sample_random="      + std::to_string(sampler.sample_random)
               + " sampling_factor="    + std::to_string(sampler.sampling_factor)
               + " splitter_length_factor=" + std::to_string(sampler.splitter_length_factor)
               + " redistribute_sample=" + std::to_string(sampler.redistribute_sample)
               + " level_adjusted_scaling=" + std::to_string(sampler.level_adjusted_scaling)
               + " local_sorter="       + enum_name(dss_mehnert::local_sorter_names, local_sorter)
               + " rquick_v1="          + std::to_string(rquick_v1)
               + " rquick_lcp="         + std::to_string(rquick_lcp)
               + " long_filter="        + std::to_string(long_filter)
               + " lcp_compression="    + std::to_string(lcp_compression)
               + " prefix_compression=" + std::to_string(prefix_compression)
               + " prefix_doubling="    + std::to_string(prefix_doubling)
               + " grid_bloomfilter="   + std::to_string(grid_bloomfilter)
               + " bloomfilter_base_case=" + std::to_string(bloomfilter_base_case)
               + " bloomfilter_level_dedup=" + std::to_string(bloomfilter_level_dedup)
               + " alltoall="            + enum_name(alltoall_names, alltoall_routine)
               + " alltoall_large_counts=" + std::to_string(alltoall_large_counts)
               + " onefactor_num_slots=" + std::to_string(onefactor_num_slots)
               + " onefactor_use_issend=" + std::to_string(onefactor_use_issend)
               + " onefactor_synchronized=" + std::to_string(onefactor_synchronized);
        // clang-format on
    }

    dss_mehnert::mpi::AlltoallvAlgorithm get_alltoall_algorithm() const {
        using dss_mehnert::mpi::AlltoallvAlgorithm;

        switch (alltoall_routine) {
            case MPIRoutineAllToAll::native:
                return AlltoallvAlgorithm::native;
            case MPIRoutineAllToAll::direct:
                return AlltoallvAlgorithm::direct;
            case MPIRoutineAllToAll::onefactor:
                return AlltoallvAlgorithm::onefactor;
            case MPIRoutineAllToAll::pairwise:
                return AlltoallvAlgorithm::pairwise;
            case MPIRoutineAllToAll::sentinel:
                break;
        }
        tlx_die("unknown MPI routine");
    }

    dss_mehnert::mpi::AlltoallvParams alltoallv_params() const {
        using dss_mehnert::mpi::AlltoallvAlgorithm;
        using dss_mehnert::mpi::OneFactorMode;

        auto const algorithm = get_alltoall_algorithm();
        tlx_die_verbose_if(
            algorithm != AlltoallvAlgorithm::native && !CliOptions::enable_alltoall,
            "this alltoallv algorithm requires the CLI_ENABLE_ALLTOALL feature"
        );

        return {
            .algorithm = algorithm,
            .large_counts = alltoall_large_counts,
            .onefactor = {
                .mode =
                    onefactor_synchronized ? OneFactorMode::synchronized : OneFactorMode::windowed,
                .num_slots = onefactor_num_slots,
                .use_issend = onefactor_use_issend,
            },
        };
    }

    dss_mehnert::SplitterSorter get_splitter_sorter() const {
        using dss_mehnert::SplitterSorter;
        tlx_die_verbose_if(rquick_v1 && rquick_lcp, "RQuick v1 does not support using LCP values");
        tlx_die_verbose_if(
            splitter_sequential && (rquick_v1 || rquick_lcp),
            "can't use both RQuick and sequential sorting"
        );
        tlx_die_verbose_if(
            long_filter && (rquick_v1 || rquick_lcp || splitter_sequential),
            "the long filter can't be combined with another splitter sorter"
        );
        tlx_die_verbose_if(
            long_filter && !sampler.sample_indexed,
            "the long filter requires indexed sampling ('-I')"
        );

        if (splitter_sequential) {
            return SplitterSorter::Sequential;
        } else if (rquick_v1) {
            return SplitterSorter::RQuickV1;
        } else if (rquick_lcp) {
            return SplitterSorter::RQuickLcp;
        } else if (long_filter) {
            return SplitterSorter::RQuickLongFilter;
        } else {
            return SplitterSorter::RQuickV2;
        }
    }
};

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

inline void parse_level_arg(std::vector<std::string> const& param, std::vector<size_t>& levels) {
    std::transform(param.begin(), param.end(), std::back_inserter(levels), [](auto& str) {
        return std::stoi(str);
    });


    tlx_die_verbose_unless(
        std::is_sorted(levels.begin(), levels.end(), std::greater_equal<>{}),
        "the given group sizes must be decreasing"
    );
}

inline void parse_level_arg(
    int cpus_per_level,
    int num_levels,
    std::vector<std::string> const& param,
    std::vector<size_t>& levels
) {
    if (!param.empty()) {
        std::transform(param.begin(), param.end(), std::back_inserter(levels), [](auto& str) {
            return std::stoi(str);
        });

    } else {
        // compute levels:
        switch (num_levels) {
            case 1:
                break;
            case 2:
                levels.push_back(cpus_per_level);
                break;
            case 3: {
                size_t size = kamping::comm_world().size() / cpus_per_level;
                size_t const log = std::log2(size);
                tlx_die_verbose_unless(
                    static_cast<size_t>(std::pow(2, log)) == size,
                    "Num procs divided by number cpus per Node must be power of two"
                );
                if (static_cast<size_t>(std::sqrt(size)) >= 2) {
                    size_t const log = std::log2(size);
                    if (log % 2 == 1) {
                        size *= 2;
                    }
                    levels.push_back(static_cast<size_t>(std::sqrt(size)) * cpus_per_level);
                }
                levels.push_back(cpus_per_level);
                break;
            }
            default:
                tlx_die("not implemented yet");
                break;
        }
    }
    tlx_die_verbose_unless(
        std::is_sorted(levels.begin(), levels.end(), std::greater_equal<>{}),
        "the given group sizes must be decreasing"
    );
    if (kamping::comm_world().is_root() == 1) {
        std::cout << "levels: " << std::endl;
        for (auto const& level: levels) {
            std::cout << level << ", ";
        }
    }
}

inline void add_common_args(CommonArgs& args, CLI::App& app) {
    // -- General --------------------------------------------------------------
    app.add_option("--experiment", args.experiment, "name to identify the experiment being run")
        ->group("General");
    app.add_option("--num-iterations", args.num_iterations, "number of sorting iterations to run")
        ->group("General");
    app.add_option("--seed", args.seed, "base seed for input generation (default 42)")
        ->group("General");

    // -- Sampling -------------------------------------------------------------
    app.add_flag("--sample-chars", args.sampler.sample_chars, "use character based sampling")
        ->group("Sampling");
    app.add_flag("--sample-indexed", args.sampler.sample_indexed, "use indexed sampling")
        ->group("Sampling");
    app.add_flag("--sample-random", args.sampler.sample_random, "use random sampling")
        ->group("Sampling");
    app.add_option(
           "--sampling-factor",
           args.sampler.sampling_factor,
           "use the given oversampling factor"
    )
        ->group("Sampling");
    app.add_option(
           "--splitter-length-factor",
           args.sampler.splitter_length_factor,
           "maximum splitter length as a multiple of (avg_lcp + 5)"
    )
        ->group("Sampling");
    app.add_flag(
           "--redistribute-sample",
           args.sampler.redistribute_sample,
           "pseudorandomly redistribute the splitter sample across PEs before sorting it"
    )
        ->group("Sampling");
    app.add_flag(
           "--level-adjusted-scaling",
           args.sampler.level_adjusted_scaling,
           "scale the sampling factor with the number of levels"
    )
        ->group("Sampling");

    // -- Splitter sorting -----------------------------------------------------
    app.add_flag("--rquick-v1", args.rquick_v1, "use version 1 of RQuick (defaults to v2)")
        ->group("Splitter Sorting");
    app.add_flag("--rquick-lcp", args.rquick_lcp, "use LCP values in RQuick (only with v2)")
        ->group("Splitter Sorting");
    app.add_flag(
           "--long-filter",
           args.long_filter,
           "sort the splitter sample with the long-string filter (RQuick v2, indexed only)"
    )
        ->group("Splitter Sorting");
    app.add_flag(
           "--splitter-sequential",
           args.splitter_sequential,
           "use sequential splitter sorting"
    )
        ->group("Splitter Sorting");

    // -- Bloom filter ---------------------------------------------------------
    app.add_flag("--prefix-doubling", args.prefix_doubling, "use prefix doubling merge sort")
        ->group("Bloom Filter");
    app.add_flag(
           "--grid-bloomfilter",
           args.grid_bloomfilter,
           "use gridwise bloom filter (requires prefix doubling) [default]"
    )
        ->group("Bloom Filter");
    app.add_flag(
           "--bloomfilter-base-case",
           args.bloomfilter_base_case,
           "enable the allgather-based bloom filter base case when every PE holds "
           "at most one hash value"
    )
        ->group("Bloom Filter");
    app.add_flag(
           "--bloomfilter-level-dedup,--no-bloomfilter-level-dedup{false}",
           args.bloomfilter_level_dedup,
           "forward only one entry per distinct hash at each intermediate grid level [default]"
    )
        ->group("Bloom Filter");

    // -- Communication --------------------------------------------------------
    app.add_flag(
           "--lcp-compression",
           args.lcp_compression,
           "compress LCP values during string exchange"
    )
        ->group("Communication");
    app.add_flag(
           "--prefix-compression",
           args.prefix_compression,
           "use LCP compression during string exchange"
    )
        ->group("Communication");
    app.add_option("--local-sorter", args.local_sorter, "sequential sorter for the base case")
        ->transform(
            CLI::CheckedTransformer(dss_mehnert::local_sorter_names, CLI::ignore_case)
                .description(enum_value_list(dss_mehnert::local_sorter_names))
        )
        ->default_str(enum_name(dss_mehnert::local_sorter_names, args.local_sorter))
        ->group("General");
    app.add_option(
           "--redistribution",
           args.redistribution,
           "redistribution scheme to use for multi-level sort"
    )
        ->transform(
            CLI::CheckedTransformer(redistribution_names, CLI::ignore_case)
                .description(enum_value_list(redistribution_names))
        )
        ->default_str(enum_name(redistribution_names, args.redistribution))
        ->group("Communication");

    // -- All-to-All -----------------------------------------------------------
    app.add_option(
           "--alltoallv",
           args.alltoall_routine,
           "All-To-All routine to use during string exchange"
    )
        ->transform(
            CLI::CheckedTransformer(alltoall_names, CLI::ignore_case)
                .description(enum_value_list(alltoall_names))
        )
        ->default_str(enum_name(alltoall_names, args.alltoall_routine))
        ->group("All-to-All");
    app.add_flag(
           "--enable-large-counts-handling,!--disable-large-counts-handling",
           args.alltoall_large_counts,
           "guard every alltoallv against exceeding the int32 count limit, falling back "
           "to the big-datatype exchange when it would [default]"
    )
        ->group("All-to-All");
    app.add_option(
           "--alltoallv-onefactor-num-slots",
           args.onefactor_num_slots,
           "number of outstanding isend/irecv pairs for the one_factor routine"
    )
        ->group("All-to-All");
    app.add_flag(
           "--alltoallv-onefactor-issend",
           args.onefactor_use_issend,
           "use synchronous (rendezvous) sends instead of standard sends in the "
           "one_factor routine"
    )
        ->group("All-to-All");
    app.add_flag(
           "--alltoallv-onefactor-synchronized",
           args.onefactor_synchronized,
           "run the one_factor routine as p lock-step Sendrecv rounds instead of "
           "the pipelined window"
    )
        ->group("All-to-All");

    // -- Checking / debugging -------------------------------------------------
    app.add_flag("--check-sorted", args.check_sorted, "check that the result is sorted")
        ->group("Checking");
    app.add_flag("--check-complete", args.check_complete, "check that the result is complete")
        ->group("Checking");
    app.add_flag("--verbose", args.verbose, "print some debug output")->group("Checking");
    app.add_flag("--count-prefixes", args.count_prefixes, "count LCPs and dist prefixes")
        ->group("Checking");
    app.add_flag(
           "--print-sorted",
           args.print_sorted,
           "gather the sorted strings on the root PE and print them (debug only)"
    )
        ->group("Checking");
}

inline void add_sorter_args(
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

// hack to integrate experiment flags in kaval
inline void set_experiment(SorterArgs& args, size_t num_levels) {
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
