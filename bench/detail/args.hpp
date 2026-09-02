// (c) 2023 Pascal Mehnert
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The arguments distributed_sorter's command line parses into. Kept free of CLI11 so that the
// rest of the benchmark harness can see the arguments without paying for the parser; the CLI11
// options that fill these structs live in cli.hpp, which only the executable includes.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <tlx/die/core.hpp>

#include "detail/enum_names.hpp"
#include "dss/mpi/alltoallv/params.hpp"
#include "dss/mpi/communicator.hpp"
#include "dss/options.hpp"
#include "dss/sorter/distributed/partition_policy_factory.hpp"
#include "dss/sorter/local_sorter.hpp"
#include "input/string_generator.hpp"

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

enum class Algorithm { merge_sort = 0, prefix_doubling, rquick, shared_memory, sentinel };

inline EnumNames<Algorithm> const algorithm_names{
    {"merge-sort", Algorithm::merge_sort},
    {"prefix-doubling", Algorithm::prefix_doubling},
    {"rquick", Algorithm::rquick},
    {"shared-memory", Algorithm::shared_memory},
};

template <typename Json>
void to_json(Json& json, Algorithm const value) {
    json = enum_name(algorithm_names, value);
}

struct CommonArgs {
    std::string experiment;
    dss_mehnert::mpi::AlltoallvAlgorithm alltoall_algorithm =
        dss_mehnert::mpi::AlltoallvAlgorithm::native;
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
    Algorithm algorithm = Algorithm::merge_sort;
    bool grid_bloomfilter = true;
    bool bloomfilter_base_case = false;
    bool bloomfilter_level_dedup = true;
    size_t num_iterations = 5;
    bool check_sorted = false;
    bool check_complete = false;
    bool verbose = false;
    bool gather_counters = false;
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
               + " shift_sample_to_neighbor=" + std::to_string(sampler.shift_sample_to_neighbor)
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
               + " prefix_doubling="    + std::to_string(algorithm == Algorithm::prefix_doubling)
               + " grid_bloomfilter="   + std::to_string(grid_bloomfilter)
               + " bloomfilter_base_case=" + std::to_string(bloomfilter_base_case)
               + " bloomfilter_level_dedup=" + std::to_string(bloomfilter_level_dedup)
               + " alltoall="            + enum_name(dss_mehnert::mpi::alltoall_names, alltoall_algorithm)
               + " alltoall_large_counts=" + std::to_string(alltoall_large_counts)
               + " onefactor_num_slots=" + std::to_string(onefactor_num_slots)
               + " onefactor_use_issend=" + std::to_string(onefactor_use_issend)
               + " onefactor_synchronized=" + std::to_string(onefactor_synchronized);
        // clang-format on
    }

    dss_mehnert::mpi::AlltoallvParams alltoallv_params() const {
        using dss_mehnert::mpi::AlltoallvAlgorithm;
        using dss_mehnert::mpi::OneFactorMode;

        tlx_die_verbose_if(
            alltoall_algorithm != AlltoallvAlgorithm::native && !CliOptions::enable_alltoall,
            "this alltoallv algorithm requires the CLI_ENABLE_ALLTOALL feature"
        );

        return {
            .algorithm = alltoall_algorithm,
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
    dss_mehnert::bench::input::StringGenerator string_generator =
        dss_mehnert::bench::input::StringGenerator::dn_ratio;
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

    // the parsed command line as bench/input/ wants to see it: everything the generators need and
    // nothing about how the sort itself is configured
    dss_mehnert::bench::input::Config input_config(dss_mehnert::Communicator const& comm) const {
        return {
            .generator = string_generator,
            .num_strings = scaled_strings(comm),
            .len_strings = len_strings,
            .len_strings_min = len_strings_min,
            .len_strings_max = len_strings_max,
            .dn_ratio = dn_ratio,
            .dn_encode_padding = dn_encode_padding,
            .skew_fraction = skew_fraction,
            .skew_factor = skew_factor,
            .use_uniform_prefix = use_uniform_prefix,
            .id_placement = id_placement,
            .simulate_num_pes = simulate_num_pes,
            .seed = seed,
            .path = path,
            .max_num_bytes = max_num_bytes,
        };
    }
};
