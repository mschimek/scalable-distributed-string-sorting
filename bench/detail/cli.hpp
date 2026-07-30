// (c) 2023 Pascal Mehnert
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The command line of distributed_sorter: the CLI11 options that fill the argument structs
// in args.hpp, and the post-processing of the parsed values.

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

#include "detail/args.hpp"
#include "detail/enum_names.hpp"

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
    app.add_option("--algorithm", args.algorithm, "top-level sorting algorithm to run")
        ->transform(
            CLI::CheckedTransformer(algorithm_names, CLI::ignore_case)
                .description(enum_value_list(algorithm_names))
        )
        ->default_str(enum_name(algorithm_names, args.algorithm))
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
           args.alltoall_algorithm,
           "All-To-All routine to use during string exchange"
    )
        ->transform(
            CLI::CheckedTransformer(dss_mehnert::mpi::alltoall_names, CLI::ignore_case)
                .description(enum_value_list(dss_mehnert::mpi::alltoall_names))
        )
        ->default_str(enum_name(dss_mehnert::mpi::alltoall_names, args.alltoall_algorithm))
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
    if (args.algorithm == Algorithm::rquick) {
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
