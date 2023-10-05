// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <filesystem>
#include <type_traits>
#include <utility>

#include <kamping/collectives/alltoall.hpp>
#include <tlx/cmdline_parser.hpp>
#include <tlx/die/core.hpp>

#include "options.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/redistribution.hpp"

inline void check_path_exists(std::string const& path) {
    tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
};

enum class MPIRoutineAllToAll { native = 0, direct, combined, sentinel };

enum class Redistribution { none = 0, naive, simple_strings, simple_chars, grid, sentinel };

template <typename T>
T clamp_enum_value(size_t const i) {
    return static_cast<T>(std::min(i, static_cast<size_t>(T::sentinel)));
}

struct CommonArgs {
    std::string experiment;
    bool sample_chars = false;
    bool sample_indexed = false;
    bool sample_random = false;
    size_t sampling_factor = 2;
    size_t alltoall_routine = static_cast<size_t>(MPIRoutineAllToAll::native);
    bool rquick_v1 = false;
    bool rquick_lcp = false;
    size_t redistribution = static_cast<size_t>(Redistribution::grid);
    bool prefix_compression = false;
    bool lcp_compression = false;
    bool prefix_doubling = false;
    bool grid_bloomfilter = false;
    size_t num_iterations = 5;
    bool check_sorted = false;
    bool check_complete = false;
    bool verbose = false;

    std::string get_prefix(dss_mehnert::Communicator const& comm) const {
        // clang-format off
        return std::string("RESULT")
               + (experiment.empty() ? "" : (" experiment=" + experiment))
               + " num_procs="          + std::to_string(comm.size())
               + " sample_chars="       + std::to_string(sample_chars)
               + " sample_indexed="     + std::to_string(sample_indexed)
               + " sample_random="      + std::to_string(sample_random)
               + " sampling_factor="    + std::to_string(sampling_factor)
               + " rquick_v1="          + std::to_string(rquick_v1)
               + " rquick_lcp="         + std::to_string(rquick_lcp)
               + " lcp_compression="    + std::to_string(lcp_compression)
               + " prefix_compression=" + std::to_string(prefix_compression)
               + " prefix_doubling="    + std::to_string(prefix_doubling)
               + " grid_bloomfilter="   + std::to_string(grid_bloomfilter);
        // clang-format on
    }
};

inline void die_with_feature(std::string_view feature) {
    tlx_die("feature disabled for compile time; enable with '-D" << feature << "=On'");
}

inline void parse_level_arg(std::vector<std::string> const& param, std::vector<size_t>& levels) {
    std::transform(param.begin(), param.end(), std::back_inserter(levels), [](auto& str) {
        return std::stoi(str);
    });

    tlx_die_verbose_unless(
        std::is_sorted(levels.begin(), levels.end(), std::greater_equal<>{}),
        "the given group sizes must be decreasing"
    );
}

inline auto
get_first_level(std::vector<size_t> const& levels, dss_mehnert::Communicator const& comm) {
    return std::find_if(levels.begin(), levels.end(), [&](auto const& group_size) {
        return group_size < comm.size();
    });
}

template <typename Callback, typename... Args>
void arg7(Callback cb, CommonArgs const& args) {
    namespace bloomfilter = dss_mehnert::bloomfilter;

    if (args.grid_bloomfilter) {
        cb.template operator()<Args..., bloomfilter::MultiLevel<bloomfilter::XXHasher>>();
    } else {
        cb.template operator()<Args..., bloomfilter::SingleLevel<bloomfilter::XXHasher>>();
    }
}

template <typename Callback, typename... Args>
void arg6(Callback cb, CommonArgs const& args) {
    using namespace dss_mehnert::redistribution;
    using dss_mehnert::Communicator;

    auto const redistribution = clamp_enum_value<Redistribution>(args.redistribution);
    if constexpr (CliOptions::enable_redistribution) {
        switch (redistribution) {
            case Redistribution::none: {
                arg7<Callback, Args..., NoRedistribution<Communicator>>(cb, args);
                return;
            }
            case Redistribution::naive: {
                arg7<Callback, Args..., NaiveRedistribution<Communicator>>(cb, args);
                return;
            };
            case Redistribution::simple_strings: {
                arg7<Callback, Args..., SimpleStringRedistribution<Communicator>>(cb, args);
                return;
            };
            case Redistribution::simple_chars: {
                arg7<Callback, Args..., SimpleCharRedistribution<Communicator>>(cb, args);
                return;
            };
            case Redistribution::grid: {
                arg7<Callback, Args..., GridwiseRedistribution<Communicator>>(cb, args);
                return;
            }
            case Redistribution::sentinel: {
                break;
            }
        };
        tlx_die("unknown redistribution policy");
    } else {
        if (redistribution == Redistribution::grid) {
            arg7<Callback, Args..., GridwiseRedistribution<Communicator>>(cb, args);
        } else {
            die_with_feature("CLI_ENABLE_REDISTRIBUTION");
        }
    }
}

template <typename Callback, typename CharType, typename AlltoallConfig, typename Sampler>
void arg5(Callback cb, CommonArgs const& args) {
    using namespace dss_mehnert::partition;

    constexpr bool is_indexed = Sampler::is_indexed;

    tlx_die_verbose_if(
        args.rquick_v1 && args.rquick_lcp,
        "RQuick v1 does not support using LCP values"
    );

    if (args.rquick_v1) {
        if constexpr (CliOptions::enable_rquick_v1) {
            using SplitterPolicy = RQuickV1<CharType, is_indexed>;
            using PartitionPolicy = PartitionPolicy<Sampler, SplitterPolicy>;
            arg6<Callback, CharType, AlltoallConfig, PartitionPolicy>(cb, args);
        } else {
            die_with_feature("CLI_ENABLE_RQUICK_V1");
        }
    } else {
        if (args.rquick_lcp) {
            if constexpr (CliOptions::enable_rquick_lcp) {
                using SplitterPolicy = RQuickV2<CharType, is_indexed, true>;
                using PartitionPolicy = PartitionPolicy<Sampler, SplitterPolicy>;
                arg6<Callback, CharType, AlltoallConfig, PartitionPolicy>(cb, args);
            } else {
                die_with_feature("CLI_ENABLE_RQUICK_LCP");
            }
        } else {
            using SplitterPolicy = RQuickV2<CharType, is_indexed, false>;
            using PartitionPolicy = PartitionPolicy<Sampler, SplitterPolicy>;
            arg6<Callback, CharType, AlltoallConfig, PartitionPolicy>(cb, args);
        }
    }
}

template <typename Callback, typename... Args>
void arg4(Callback cb, CommonArgs const& args) {
    using namespace dss_mehnert::sample;

    // todo this could/should use type erasure

    auto with_random = [=]<bool indexed, bool random> {
        if (args.sample_chars) {
            arg5<Callback, Args..., CharBasedSampling<indexed, random>>(cb, args);
        } else {
            arg5<Callback, Args..., StringBasedSampling<indexed, random>>(cb, args);
        }
    };

    auto with_indexed = [=]<bool indexed> {
        if (args.sample_random) {
            with_random.template operator()<indexed, true>();
        } else {
            with_random.template operator()<indexed, false>();
        }
    };

    if (args.sample_chars) {
        with_indexed.template operator()<true>();
    } else {
        with_indexed.template operator()<false>();
    }
}

template <
    typename Callback,
    typename CharType,
    typename AlltoallvCombinedKind,
    typename LcpCompression,
    typename PrefixCompression>
void arg3b(Callback cb, CommonArgs const& args) {
    using dss_mehnert::mpi::AlltoallStringsConfig;
    constexpr AlltoallStringsConfig config{
        .alltoall_kind = AlltoallvCombinedKind(),
        .compress_lcps = LcpCompression(),
        .compress_prefixes = PrefixCompression(),
    };
    arg4<Callback, CharType, std::integral_constant<AlltoallStringsConfig, config>>(cb, args);
}

template <typename Callback, typename... Args>
void arg3(Callback cb, CommonArgs const& args) {
    using namespace dss_schimek;

    if (args.prefix_compression) {
        arg3b<Callback, Args..., std::true_type>(cb, args);
    } else {
        arg3b<Callback, Args..., std::false_type>(cb, args);
    }
}

template <typename Callback, typename... Args>
void arg2(Callback cb, CommonArgs const& args) {
    if (args.lcp_compression) {
        arg3<Callback, Args..., std::true_type>(cb, args);
    } else {
        arg3<Callback, Args..., std::false_type>(cb, args);
    }
}

// todo remove this or make small the default
template <typename Callback, typename... Args>
void arg1(Callback cb, CommonArgs const& args) {
    switch (clamp_enum_value<MPIRoutineAllToAll>(args.alltoall_routine)) {
        using Kind = dss_mehnert::mpi::AlltoallvCombinedKind;

        case MPIRoutineAllToAll::native: {
            arg2<Callback, Args..., std::integral_constant<Kind, Kind::native>>(cb, args);
            return;
        }
        case MPIRoutineAllToAll::direct: {
            if constexpr (CliOptions::enable_alltoall) {
                arg2<Callback, Args..., std::integral_constant<Kind, Kind::direct>>(cb, args);
            } else {
                die_with_feature("CLI_ENABLE_ALLTOALL");
            }
            return;
        }
        case MPIRoutineAllToAll::combined: {
            if constexpr (CliOptions::enable_alltoall) {
                arg2<Callback, Args..., std::integral_constant<Kind, Kind::combined>>(cb, args);
            } else {
                die_with_feature("CLI_ENABLE_ALLTOALL");
            }
            return;
        }
        case MPIRoutineAllToAll::sentinel: {
            break;
        }
    }
    tlx_die("unknown MPI routine");
}

template <typename Callback>
inline void dispatch_common_args(Callback cb, CommonArgs const& args) {
    arg1<Callback, unsigned char>(cb, args);
}

inline void add_common_args(CommonArgs& args, tlx::CmdlineParser& cp) {
    cp.add_string('e', "experiment", args.experiment, "name to identify the experiment being run");
    cp.add_size_t(
        'i',
        "num-iterations",
        args.num_iterations,
        "number of sorting iterations to run"
    );
    cp.add_flag('C', "sample-chars", args.sample_chars, "use character based sampling");
    cp.add_flag('I', "sample-indexed", args.sample_indexed, "use indexed sampling");
    cp.add_flag('R', "sample-random", args.sample_random, "use random sampling");
    cp.add_size_t(
        'S',
        "sampling-factor",
        args.sampling_factor,
        "use the given oversampling factor"
    );
    cp.add_flag('Q', "rquick-v1", args.rquick_v1, "use version 1 of RQuick (defaults to v2)");
    cp.add_flag('L', "rquick-lcp", args.rquick_lcp, "use LCP values in RQuick (only with v2)");
    cp.add_flag(
        'l',
        "lcp-compression",
        args.lcp_compression,
        "compress LCP values during string exchange"
    );
    cp.add_flag(
        'p',
        "prefix-compression",
        args.prefix_compression,
        "use LCP compression during string exchange"
    );
    cp.add_flag('d', "prefix-doubling", args.prefix_doubling, "use prefix doubling merge sort");
    cp.add_flag(
        'g',
        "grid-bloomfilter",
        args.grid_bloomfilter,
        "use gridwise bloom filter (requires prefix doubling) [default]"
    );
    cp.add_size_t(
        'a',
        "alltoall-routine",
        args.alltoall_routine,
        "All-To-All routine to use during string exchange "
        "(0=small, 1=direct, [2]=combined)"
    );
    cp.add_size_t(
        't',
        "redistribution",
        args.redistribution,
        "redistribution scheme to use for multi-level sort "
        "(0=none, 1=naive, 2=simple-strings, 3=simple-chars, [4]=grid)"
    );
    cp.add_flag('v', "check-sorted", args.check_sorted, "check that the result is sorted");
    cp.add_flag('V', "check-complete", args.check_complete, "check that the result is complete");
    cp.add_flag("verbose", args.verbose, "print some debug output");
}

inline size_t mpi_warmup(size_t const bytes_per_PE, dss_mehnert::Communicator const& comm) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> dist{'A', 'Z'};

    std::vector<unsigned char> random_data(bytes_per_PE * comm.size());
    std::generate(random_data.begin(), random_data.end(), [&] { return dist(gen); });

    auto recv_data = comm.alltoall(kamping::send_buf(random_data)).extract_recv_buffer();

    auto volatile sum = std::accumulate(recv_data.begin(), recv_data.end(), size_t{0});
    return sum;
}
