// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>

#include <CLI/CLI.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <tlx/die/core.hpp>
#include <tlx/sort/strings/parallel_sample_sort.hpp>

#include "executables/serialization.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/alltoallv/dispatch.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/print_strings.hpp"
#include "options.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/partition_policy_factory.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "sorter/distributed/sample.hpp"
#include "sorter/local_sorter.hpp"
#include "strings/stringset.hpp"

inline void check_path_exists(std::string const& path) {
    tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
};

// The old `combined` kind is gone: it is now the orthogonal `--alltoall-large-counts` flag,
// which applies to every algorithm rather than being one of them.
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

template <typename T>
T clamp_enum_value(size_t const i) {
    return static_cast<T>(std::min(i, static_cast<size_t>(T::sentinel)));
}

struct CommonArgs {
    std::string experiment;
    size_t alltoall_algorithm = static_cast<size_t>(MPIRoutineAllToAll::native);
    bool alltoall_large_counts = true;
    size_t onefactor_num_slots = 16;
    bool onefactor_use_issend = false;
    bool onefactor_synchronized = false;
    dss_mehnert::SamplerArgs sampler;
    bool rquick_v1 = false;
    bool rquick_lcp = false;
    bool long_filter = false;
    bool splitter_sequential = false;
    size_t redistribution = static_cast<size_t>(Redistribution::grid);
    size_t local_sorter = static_cast<size_t>(dss_mehnert::LocalSorter::multikey_quicksort);
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
               + " local_sorter="       + enum_name(dss_mehnert::local_sorter_names, get_local_sorter())
               + " rquick_v1="          + std::to_string(rquick_v1)
               + " rquick_lcp="         + std::to_string(rquick_lcp)
               + " long_filter="        + std::to_string(long_filter)
               + " lcp_compression="    + std::to_string(lcp_compression)
               + " prefix_compression=" + std::to_string(prefix_compression)
               + " prefix_doubling="    + std::to_string(prefix_doubling)
               + " grid_bloomfilter="   + std::to_string(grid_bloomfilter)
               + " bloomfilter_base_case=" + std::to_string(bloomfilter_base_case)
               + " bloomfilter_level_dedup=" + std::to_string(bloomfilter_level_dedup)
               + " alltoall="            + enum_name(alltoall_names, get_alltoall_routine())
               + " alltoall_large_counts=" + std::to_string(alltoall_large_counts)
               + " onefactor_num_slots=" + std::to_string(onefactor_num_slots)
               + " onefactor_use_issend=" + std::to_string(onefactor_use_issend)
               + " onefactor_synchronized=" + std::to_string(onefactor_synchronized);
        // clang-format on
    }

    MPIRoutineAllToAll get_alltoall_routine() const {
        return clamp_enum_value<MPIRoutineAllToAll>(alltoall_algorithm);
    }

    Redistribution get_redistribution() const {
        return clamp_enum_value<Redistribution>(redistribution);
    }

    dss_mehnert::mpi::AlltoallvAlgorithm get_alltoall_algorithm() const {
        using dss_mehnert::mpi::AlltoallvAlgorithm;

        switch (get_alltoall_routine()) {
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

    dss_mehnert::LocalSorter get_local_sorter() const {
        return clamp_enum_value<dss_mehnert::LocalSorter>(local_sorter);
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

inline auto
get_first_level(std::vector<size_t> const& levels, dss_mehnert::Communicator const& comm) {
    return std::find_if(levels.begin(), levels.end(), [&](auto const& group_size) {
        return group_size < comm.size();
    });
}

// the input is partitioned once per level and once more in the final round
inline size_t
get_num_levels(std::vector<size_t> const& levels, dss_mehnert::Communicator const& comm) {
    return static_cast<size_t>(std::distance(get_first_level(levels, comm), levels.end())) + 1;
}

template <typename Callback, typename... Args>
void dispatch_bloomfilter(Callback cb, CommonArgs const& args) {
    using namespace dss_mehnert::bloomfilter;

    if (args.grid_bloomfilter) {
        cb.template operator()<Args..., MultiLevel<true, XXHasher>>();
    } else {
        cb.template operator()<Args..., SingleLevel<true, XXHasher>>();
    }
}

// The alltoallv algorithm is a runtime parameter (see CommonArgs::alltoallv_params), so only
// the two compression flags are still peeled into template parameters here.
template <typename Callback, typename CharType>
void dispatch_alltoall_strings(Callback cb, CommonArgs const& args) {
    auto dispatch_config = [&]<bool compress_lcps, bool compress_prefixes> {
        using dss_mehnert::mpi::AlltoallStringsConfig;
        constexpr AlltoallStringsConfig config{
            .compress_lcps = compress_lcps,
            .compress_prefixes = compress_prefixes,
        };
        using Config = std::integral_constant<AlltoallStringsConfig, config>;
        dispatch_bloomfilter<Callback, CharType, Config>(cb, args);
    };

    auto disptach_prefix_compression = [&]<bool compress_lcps> {
        if (args.prefix_compression) {
            dispatch_config.template operator()<compress_lcps, true>();
        } else {
            dispatch_config.template operator()<compress_lcps, false>();
        }
    };

    // validates the selected algorithm against the enabled features
    (void)args.alltoallv_params();

    if (args.lcp_compression) {
        disptach_prefix_compression.template operator()<true>();
    } else {
        disptach_prefix_compression.template operator()<false>();
    }
}

template <typename Callback>
inline void dispatch_common_args(Callback cb, CommonArgs const& args) {
    dispatch_alltoall_strings<Callback, unsigned char>(cb, args);
}

// CLI11 equivalent of add_common_args. Kept in parallel with the tlx-based
// add_common_args above so the two can be compared before switching over; the
// options are identical (same names, short flags, defaults) but organised into
// named groups so `--help` is easier to read. Once verified this should replace
// the tlx variant and the tlx variant can be removed.
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
        ->default_str(enum_name(dss_mehnert::local_sorter_names, args.get_local_sorter()))
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
        ->default_str(enum_name(redistribution_names, args.get_redistribution()))
        ->group("Communication");

    // -- All-to-All -----------------------------------------------------------
    app.add_option(
           "--alltoallv",
           args.alltoall_algorithm,
           "All-To-All routine to use during string exchange"
    )
        ->transform(
            CLI::CheckedTransformer(alltoall_names, CLI::ignore_case)
                .description(enum_value_list(alltoall_names))
        )
        ->default_str(enum_name(alltoall_names, args.get_alltoall_routine()))
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

template <typename Container>
inline void count_prefix_lengths(Container& container, dss_mehnert::Communicator const& comm) {
    using namespace kamping;

    using Char = Container::Char;
    std::vector<Char> const last_string =
        container.empty() ? std::vector<Char>{0} : container.get_raw_string(container.size() - 1);
    auto const pred_string = dss_mehnert::get_predecessor(last_string, container.empty(), comm);

    size_t first_lcp = 0;
    if (pred_string && !container.empty()) {
        auto const first_string = container.get_raw_string(0);
        first_lcp = dss_schimek::calc_lcp(pred_string->data(), first_string.data());
    }

    size_t const last_lcp = container.empty() ? 0 : container.lcps().back();
    auto const pred_lcp = container.size() == 1
                              ? first_lcp
                              : dss_mehnert::get_predecessor(last_lcp, container.empty(), comm);

    auto const begin = container.lcps().begin(), end = container.lcps().end();
    auto const local_lcp = first_lcp + std::accumulate(begin, end, size_t{0});

    auto const dist = [](auto const lcp1, auto const lcp2) { return std::max(lcp1, lcp2) + 1; };

    auto local_dist = container.empty() || !pred_lcp ? 0 : dist(*pred_lcp, first_lcp);
    if (!container.empty()) {
        local_dist =
            std::transform_reduce(std::next(begin), end, begin, size_t{0}, std::plus<>{}, dist);
    }

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();

    measuring_tool.add(local_lcp, "global_lcp_sum");
    measuring_tool.add(local_dist, "global_dist_prefix");
}

template <typename SorterArgs, typename GenerateStrings>
void run_shared_memory(
    SorterArgs args, dss_mehnert::Communicator const& comm, GenerateStrings generate_strings
) {
    tlx_die_unequal(comm.size_signed(), 1);

    auto input_container = generate_strings(args, comm);
    auto input_strings = input_container.get_strings();

    for (size_t i = 0; i != args.num_iterations; ++i) {
        args.iteration = i;
        auto const prefix = args.get_prefix(comm);

        // restore original order of input strings
        input_container.set(std::vector{input_strings});

        auto const before = std::chrono::high_resolution_clock::now();
        tlx::sort_strings_detail::parallel_sample_sort(input_container.make_string_ptr(), 0, 0);
        auto const after = std::chrono::high_resolution_clock::now();
        auto const delta = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);
        size_t const elapsed = delta.count();

        std::cout << prefix << " key=sorting_overall max_time=" << elapsed << std::endl;

        if (args.check_sorted) {
            auto const is_sorted = input_container.make_string_set().check_order();
            die_verbose_unless(is_sorted, "output is not sorted");
        }
    }
}

template <typename StringSet, typename SorterArgs, typename GenerateStrings>
void run_rquick(
    SorterArgs const& args,
    std::string prefix,
    dss_mehnert::Communicator const& comm,
    GenerateStrings generate_strings
) {
    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(args.verbose);

    measuring_tool.disableCommVolume();
    auto input_container = generate_strings(args, comm);

    dss_mehnert::MergeSortChecker<StringSet> checker;
    if (args.check_sorted || args.check_complete) {
        checker.store_container(input_container);
    }
    measuring_tool.enableCommVolume();

    comm.barrier();

    std::random_device rd;
    std::mt19937_64 gen{rd()};

    auto const tag = comm.default_tag();
    auto const& mpi_comm = comm.mpi_communicator();

    if (args.rquick_lcp) {
        using StringPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;
        measuring_tool.start("none", "sorting_overall");
        RQuick2::Data<StringPtr> data{input_container.release_raw_strings()};
        auto sorted_container =
            RQuick2::sort(std::move(data), tag, gen, mpi_comm, args.get_local_sorter());
        measuring_tool.stop("none", "sorting_overall", comm);

        measuring_tool.disable();
        measuring_tool.disableCommVolume();

        if (args.check_sorted) {
            auto const is_sorted = checker.is_sorted(sorted_container.make_string_set(), comm);
            die_verbose_unless(is_sorted, "output is not sorted");
            auto const is_complete = checker.is_complete(sorted_container, comm);
            die_verbose_unless(is_complete, "output is missing chars or strings");
        }
        if (args.check_complete) {
            auto const is_exact = checker.check_exhaustive(sorted_container, comm);
            die_verbose_unless(is_exact, "output is not a permutation of the input");
        }
        if (args.print_sorted) {
            dss_mehnert::gather_and_print_strings(sorted_container, comm);
        }
    } else {
        using StringPtr = tlx::sort_strings_detail::StringPtr<StringSet>;
        measuring_tool.start("none", "sorting_overall");
        RQuick2::Data<StringPtr> data{input_container.release_raw_strings()};
        auto sorted_container =
            RQuick2::sort(std::move(data), tag, gen, mpi_comm, args.get_local_sorter());
        measuring_tool.stop("none", "sorting_overall", comm);

        if (args.print_sorted) {
            dss_mehnert::gather_and_print_strings(sorted_container, comm);
        }
    }

    measuring_tool.write_on_root(*args.measurement_output, comm);
    measuring_tool.reset();
}

inline size_t mpi_warmup(size_t const bytes_per_PE, dss_mehnert::Communicator const& comm) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> dist{'A', 'Z'};

    std::vector<unsigned char> random_data(bytes_per_PE * comm.size());
    std::generate(random_data.begin(), random_data.end(), [&] { return dist(gen); });

    auto recv_data = comm.alltoall(kamping::send_buf(random_data));

    auto volatile sum = std::accumulate(recv_data.begin(), recv_data.end(), size_t{0});
    return sum;
}

// Irregular counterpart to mpi_warmup: every PE sends a different, random number of bytes to every
// other PE, drawn uniformly from [min_bytes, max_bytes).
inline size_t mpi_irregular_warmup(
    size_t const min_bytes, size_t const max_bytes, dss_mehnert::Communicator const& comm
) {
    tlx_die_unless(min_bytes < max_bytes);

    std::mt19937_64 gen{comm.rank()};
    // uniform_int_distribution is inclusive on both ends, so draw from [min_bytes, max_bytes - 1]
    std::uniform_int_distribution<size_t> count_dist{min_bytes, max_bytes - 1};
    std::uniform_int_distribution<unsigned char> byte_dist{'A', 'Z'};

    std::vector<int> send_counts(comm.size());
    std::generate(send_counts.begin(), send_counts.end(), [&] {
        return static_cast<int>(count_dist(gen));
    });

    size_t const send_total = std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
    std::vector<unsigned char> send_data(send_total);
    std::generate(send_data.begin(), send_data.end(), [&] { return byte_dist(gen); });

    // recv_counts are exchanged internally by kamping from the send_counts
    auto recv_data =
        comm.alltoallv(kamping::send_buf(send_data), kamping::send_counts(send_counts));

    auto volatile sum = std::accumulate(recv_data.begin(), recv_data.end(), size_t{0});
    return sum;
}

namespace dss_mehnert {

template <typename StringSet, typename Callback>
void dispatch_redistribution(Callback cb, CommonArgs const& args) {
    using namespace dss_mehnert::redistribution;
    using dss_mehnert::Communicator;

    auto const redistribution = args.get_redistribution();
    if constexpr (CliOptions::enable_redistribution) {
        using PolymorphicPolicy =
            PolymorphicRedistributionPolicy<StringSet, RowwiseSplit<Communicator>>;

        switch (redistribution) {
            case Redistribution::none: {
                // cb(NoRedistribution<Communicator>{});
                tlx_die("disabled for compile-time");
                return;
            }
            case Redistribution::naive: {
                cb(PolymorphicPolicy{NaiveRedistribution<Communicator>{}});
                return;
            };
            case Redistribution::simple_strings: {
                cb(PolymorphicPolicy{SimpleStringRedistribution<Communicator>{}});
                return;
            };
            case Redistribution::simple_chars: {
                cb(PolymorphicPolicy{SimpleCharRedistribution<Communicator>{}});
                return;
            };
            case Redistribution::det_strings: {
                cb(PolymorphicPolicy{DeterministicStringRedistribution<Communicator>{}});
                return;
            };
            case Redistribution::det_chars: {
                cb(PolymorphicPolicy{DeterministicCharRedistribution<Communicator>{}});
                return;
            };
            case Redistribution::grid: {
                cb(GridwiseRedistribution<Communicator>{});
                return;
            }
            case Redistribution::sentinel: {
                break;
            }
        };
        tlx_die("unknown redistribution policy");
    } else {
        if (redistribution == Redistribution::grid) {
            cb(GridwiseRedistribution<Communicator>{});
        } else {
            dss_mehnert::die_with_feature("CLI_ENABLE_REDISTRIBUTION");
        }
    }
}

} // namespace dss_mehnert
