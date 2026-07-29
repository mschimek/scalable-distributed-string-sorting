// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
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
#include <tlx/die.hpp>
#include <tlx/die/core.hpp>
#include <tlx/sort/strings/parallel_sample_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "bench/reporting.hpp"
#include "executables/cli.hpp"
#include "executables/serialization.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/print_strings.hpp"
#include "options.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"
#include "util/string_generator.hpp"


inline void check_path_exists(std::string const& path) {
    tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
};

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
            RQuick2::sort(std::move(data), tag, gen, mpi_comm, args.local_sorter);
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
            RQuick2::sort(std::move(data), tag, gen, mpi_comm, args.local_sorter);
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

    auto const redistribution = args.redistribution;
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
                args.local_sorter,
                args.alltoallv_params()
            ),
            std::move(redistribution),
            args.alltoallv_params(),
            args.local_sorter
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
                args.local_sorter,
                args.alltoallv_params()
            ),
            std::move(redistribution),
            args.bloomfilter_base_case,
            args.bloomfilter_level_dedup,
            args.alltoallv_params(),
            args.local_sorter
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

int main(int argc, char* argv[]) {
    SorterArgs args;

    CLI::App app{"a distributed string sorter"};
    app.option_defaults()->always_capture_default();

    std::string timer_json_path;
    std::vector<std::string> levels_param;
    size_t cpus_per_node = 48;
    size_t num_levels = 1;

    add_sorter_args(args, app, timer_json_path, levels_param, cpus_per_node, num_levels);

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
        config["alltoall"] = args.alltoall_algorithm;
        config["alltoall_large_counts"] = args.alltoall_large_counts;
        config["alltoall_onefactor_num_slots"] = args.onefactor_num_slots;
        config["alltoall_onefactor_synchronized"] = args.onefactor_synchronized;
        config["alltoall_onefactor_use_issend"] = args.onefactor_use_issend;
        config["redistribution"] = enum_name(redistribution_names, args.redistribution);
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
