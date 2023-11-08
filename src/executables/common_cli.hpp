// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <type_traits>
#include <utility>

#include <kamping/collectives/alltoall.hpp>
#include <tlx/cmdline_parser.hpp>
#include <tlx/die/core.hpp>
#include <tlx/sort/strings/parallel_sample_sort.hpp>

#include "mpi/alltoall_combined.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "options.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringset.hpp"

inline void check_path_exists(std::string const& path) {
    tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
};

enum class MPIRoutineAllToAll { native = 0, direct, combined, sentinel };

// clang-format off
enum class Redistribution { none = 0, naive, simple_strings, simple_chars,
                            det_strings, det_chars, grid, sentinel };
// clang-format on

enum class SplitterSorter { RQuickV1, RQuickV2, RQuickLcp, Sequential };

template <typename T>
T clamp_enum_value(size_t const i) {
    return static_cast<T>(std::min(i, static_cast<size_t>(T::sentinel)));
}

struct SamplerArgs {
    bool sample_chars = false;
    bool sample_indexed = false;
    bool sample_random = false;
    size_t sampling_factor = 2;
};

struct CommonArgs {
    std::string experiment;
    size_t alltoall_routine = static_cast<size_t>(MPIRoutineAllToAll::native);
    SamplerArgs sampler;
    bool rquick_v1 = false;
    bool rquick_lcp = false;
    bool splitter_sequential = false;
    size_t redistribution = static_cast<size_t>(Redistribution::grid);
    bool prefix_compression = false;
    bool lcp_compression = false;
    bool prefix_doubling = false;
    bool grid_bloomfilter = true;
    size_t num_iterations = 5;
    bool check_sorted = false;
    bool check_complete = false;
    bool verbose = false;

    std::string get_prefix(dss_mehnert::Communicator const& comm) const {
        // clang-format off
        return std::string("RESULT")
               + (experiment.empty() ? "" : (" experiment=" + experiment))
               + " num_procs="          + std::to_string(comm.size())
               + " sample_chars="       + std::to_string(sampler.sample_chars)
               + " sample_indexed="     + std::to_string(sampler.sample_indexed)
               + " sample_random="      + std::to_string(sampler.sample_random)
               + " sampling_factor="    + std::to_string(sampler.sampling_factor)
               + " rquick_v1="          + std::to_string(rquick_v1)
               + " rquick_lcp="         + std::to_string(rquick_lcp)
               + " lcp_compression="    + std::to_string(lcp_compression)
               + " prefix_compression=" + std::to_string(prefix_compression)
               + " prefix_doubling="    + std::to_string(prefix_doubling)
               + " grid_bloomfilter="   + std::to_string(grid_bloomfilter);
        // clang-format on
    }

    SplitterSorter get_splitter_sorter() const {
        tlx_die_verbose_if(rquick_v1 && rquick_lcp, "RQuick v1 does not support using LCP values");
        tlx_die_verbose_if(
            splitter_sequential && (rquick_v1 || rquick_lcp),
            "can't use both RQuick and sequential sorting"
        );

        if (splitter_sequential) {
            return SplitterSorter::Sequential;
        } else if (rquick_v1) {
            return SplitterSorter::RQuickV1;
        } else if (rquick_lcp) {
            return SplitterSorter::RQuickLcp;
        } else {
            return SplitterSorter::RQuickV2;
        }
    }
};

inline void die_with_feature [[noreturn]] (std::string_view feature) {
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
void dispatch_bloomfilter(Callback cb, CommonArgs const& args) {
    using namespace dss_mehnert::bloomfilter;

    if (args.grid_bloomfilter) {
        cb.template operator()<Args..., MultiLevel<true, XXHasher>>();
    } else {
        cb.template operator()<Args..., SingleLevel<true, XXHasher>>();
    }
}

template <typename Callback, typename CharType>
void dispatch_alltoall_strings(Callback cb, CommonArgs const& args) {
    using AlltoallKind = dss_mehnert::mpi::AlltoallvCombinedKind;

    auto dispatch_config = [&]<AlltoallKind kind, bool compress_lcps, bool compress_prefixes> {
        using dss_mehnert::mpi::AlltoallStringsConfig;
        constexpr AlltoallStringsConfig config{
            .alltoall_kind = kind,
            .compress_lcps = compress_lcps,
            .compress_prefixes = compress_prefixes,
        };
        using Config = std::integral_constant<AlltoallStringsConfig, config>;
        dispatch_bloomfilter<Callback, CharType, Config>(cb, args);
    };

    auto disptach_prefix_compression = [&]<AlltoallKind kind, bool compress_lcps> {
        if (args.prefix_compression) {
            dispatch_config.template operator()<kind, compress_lcps, true>();
        } else {
            dispatch_config.template operator()<kind, compress_lcps, false>();
        }
    };

    auto dispatch_lcp_compression = [&]<AlltoallKind Kind> {
        if (args.lcp_compression) {
            disptach_prefix_compression.template operator()<Kind, true>();
        } else {
            disptach_prefix_compression.template operator()<Kind, false>();
        }
    };

    switch (clamp_enum_value<MPIRoutineAllToAll>(args.alltoall_routine)) {
        case MPIRoutineAllToAll::native: {
            dispatch_lcp_compression.template operator()<AlltoallKind::native>();
            return;
        }
        case MPIRoutineAllToAll::direct: {
            if constexpr (CliOptions::enable_alltoall) {
                dispatch_lcp_compression.template operator()<AlltoallKind::direct>();
            } else {
                die_with_feature("CLI_ENABLE_ALLTOALL");
            }
            return;
        }
        case MPIRoutineAllToAll::combined: {
            if constexpr (CliOptions::enable_alltoall) {
                dispatch_lcp_compression.template operator()<AlltoallKind::combined>();
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
    dispatch_alltoall_strings<Callback, unsigned char>(cb, args);
}

inline void add_common_args(CommonArgs& args, tlx::CmdlineParser& cp) {
    cp.add_string('e', "experiment", args.experiment, "name to identify the experiment being run");
    cp.add_size_t(
        'i',
        "num-iterations",
        args.num_iterations,
        "number of sorting iterations to run"
    );
    cp.add_flag('C', "sample-chars", args.sampler.sample_chars, "use character based sampling");
    cp.add_flag('I', "sample-indexed", args.sampler.sample_indexed, "use indexed sampling");
    cp.add_flag('R', "sample-random", args.sampler.sample_random, "use random sampling");
    cp.add_size_t(
        'S',
        "sampling-factor",
        args.sampler.sampling_factor,
        "use the given oversampling factor"
    );
    cp.add_flag('Q', "rquick-v1", args.rquick_v1, "use version 1 of RQuick (defaults to v2)");
    cp.add_flag('L', "rquick-lcp", args.rquick_lcp, "use LCP values in RQuick (only with v2)");
    cp.add_flag("splitter-sequential", args.splitter_sequential, "use sequential splitter sorting");
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
        "alltoall",
        args.alltoall_routine,
        "All-To-All routine to use during string exchange "
        "([0]=native, 1=direct, 2=combined)"
    );
    cp.add_size_t(
        't',
        "redistribution",
        args.redistribution,
        "redistribution scheme to use for multi-level sort "
        "(0=none, 1=naive, 2=simple-strings, 3=simple-chars, "
        " 4=det-strings, 5=det-chars, [6]=grid)"
    );
    cp.add_flag('v', "check-sorted", args.check_sorted, "check that the result is sorted");
    cp.add_flag('V', "check-complete", args.check_complete, "check that the result is complete");
    cp.add_flag("verbose", args.verbose, "print some debug output");
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

        std::cout << prefix << " max=" << elapsed << std::endl;

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

    measuring_tool.start("none", "sorting_overall");
    std::random_device rd;
    std::mt19937_64 gen{rd()};

    using StringPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    auto const tag = comm.default_tag();
    auto const& mpi_comm = comm.mpi_communicator();
    RQuick2::Data<StringPtr> data{input_container.release_raw_strings()};
    auto sorted_container = RQuick2::sort(std::move(data), tag, gen, mpi_comm);
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

    measuring_tool.write_on_root(std::cout, comm);
    measuring_tool.reset();
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

namespace dss_mehnert {
namespace redistribution {

template <typename StringSet, typename Subcommunicators>
class PolymorphicRedistributionPolicy
    : public RedistributionBase<
          Subcommunicators,
          PolymorphicRedistributionPolicy<StringSet, Subcommunicators>> {
public:
    using Communicator = Subcommunicators::Communicator;

    template <typename RedistributionPolicy>
    explicit PolymorphicRedistributionPolicy(RedistributionPolicy policy)
        : self_{new RedistributionObject<RedistributionPolicy>{std::move(policy)}} {}

    template <typename Strings>
    std::vector<size_t> impl(
        Strings const& strings,
        std::vector<size_t> const& intervals,
        Level<Communicator> const& level
    ) const {
        return self_->impl(strings, intervals, level);
    }

private:
    struct RedistributionConcept {
        virtual ~RedistributionConcept() = default;

        virtual std::vector<size_t> impl(
            FullStrings<StringSet> const& strings,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const = 0;

        virtual std::vector<size_t> impl(
            Prefixes const& prefixes,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const = 0;
    };

    template <typename RedistributionPolicy>
    struct RedistributionObject : public RedistributionConcept, private RedistributionPolicy {
        explicit RedistributionObject(RedistributionPolicy policy)
            : RedistributionPolicy{std::move(policy)} {}

        virtual std::vector<size_t> impl(
            FullStrings<StringSet> const& strings,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const override {
            return RedistributionPolicy::impl(strings, intervals, level);
        }

        virtual std::vector<size_t> impl(
            Prefixes const& prefixes,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const override {
            return RedistributionPolicy::impl(prefixes, intervals, level);
        }
    };

    std::unique_ptr<RedistributionConcept> self_;
};

} // namespace redistribution

template <typename StringSet, typename Callback>
void dispatch_redistribution(Callback cb, CommonArgs const& args) {
    using namespace dss_mehnert::redistribution;
    using dss_mehnert::Communicator;

    auto const redistribution = clamp_enum_value<Redistribution>(args.redistribution);
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
            die_with_feature("CLI_ENABLE_REDISTRIBUTION");
        }
    }
}

template <typename StringSet, typename... SamplerArgs>
class PolymorphicPartitionPolicy {
public:
    using This = PolymorphicPartitionPolicy<StringSet, SamplerArgs...>;
    using StringPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    template <typename PartitionPolicy>
    explicit PolymorphicPartitionPolicy(PartitionPolicy policy)
        : self_{new PartitionObject<PartitionPolicy>{std::move(policy)}} {}

    template <typename SamplerArg>
    std::vector<size_t> compute_partition(
        StringPtr const& strptr,
        size_t const num_partitions,
        SamplerArg const arg,
        Communicator const& comm
    ) const {
        return self_->compute_partition(strptr, num_partitions, arg, comm);
    }

private:
    template <typename SamplerArg>
    struct PartitionConcept_ {
        virtual ~PartitionConcept_() = default;

        virtual std::vector<size_t> compute_partition(
            StringPtr const& strptr,
            size_t const num_partitions,
            SamplerArg const arg,
            Communicator const& comm
        ) const = 0;
    };

    struct PartitionConcept : public PartitionConcept_<SamplerArgs>... {
        using PartitionConcept_<SamplerArgs>::compute_partition...;
    };

    template <typename PartitionPolicy, typename SamplerArg>
    struct PartitionObject_ : public virtual PartitionConcept, private virtual PartitionPolicy {
        virtual std::vector<size_t> compute_partition(
            StringPtr const& strptr,
            size_t const num_partitions,
            SamplerArg const arg,
            Communicator const& comm
        ) const override {
            return PartitionPolicy::compute_partition(strptr, num_partitions, arg, comm);
        }
    };

    template <typename PartitionPolicy>
    struct PartitionObject final : public PartitionObject_<PartitionPolicy, SamplerArgs>... {
        explicit PartitionObject(PartitionPolicy policy) : PartitionPolicy{std::move(policy)} {}
    };

    std::unique_ptr<PartitionConcept> self_;
};

template <typename Char>
using MergeSortPartitionPolicy =
    PolymorphicPartitionPolicy<StringSet<Char, Length>, sample::MaxLength>;

template <typename Char, typename LengthType, typename Permutation>
using PrefixDoublingPartitionPolicy = PolymorphicPartitionPolicy<
    sorter::AugmentedStringSet<StringSet<Char, LengthType>, Permutation>,
    sample::NoExtraArg,
    sample::DistPrefixes>;

template <typename Char, typename LengthType, typename Permutation>
using SpaceEfficientPartitionPolicy = PolymorphicPartitionPolicy<
    sorter::AugmentedStringSet<CompressedStringSet<Char, LengthType>, Permutation>,
    sample::NoExtraArg,
    sample::MaxLength,
    sample::DistPrefixes>;

template <typename Char, typename PolymorphicPolicy>
PolymorphicPolicy
init_partition_policy(SamplerArgs const& sampler, SplitterSorter splitter_sorter) {
    auto disptach_policy = [&]<typename PartitionPolicy> {
        return PolymorphicPolicy{PartitionPolicy{sampler.sampling_factor}};
    };

    auto dispatch_sorter = [&]<typename SamplePolicy> {
        using namespace dss_mehnert::partition;

        constexpr bool indexed = SamplePolicy::is_indexed;

        switch (splitter_sorter) {
            case SplitterSorter::RQuickV1: {
                if constexpr (CliOptions::enable_rquick_v1) {
                    using SplitterPolicy = RQuickV1<Char, indexed>;
                    using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                    return disptach_policy.template operator()<PartitionPolicy>();
                } else {
                    die_with_feature("CLI_ENABLE_RQUICK_V1");
                }
            }
            case SplitterSorter::RQuickV2: {
                using SplitterPolicy = RQuickV2<Char, indexed, false>;
                using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                return disptach_policy.template operator()<PartitionPolicy>();
            }
            case SplitterSorter::RQuickLcp: {
                if constexpr (CliOptions::enable_rquick_lcp) {
                    using SplitterPolicy = RQuickV2<Char, indexed, true>;
                    using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                    return disptach_policy.template operator()<PartitionPolicy>();
                } else {
                    die_with_feature("CLI_ENABLE_RQUICK_LCP");
                }
            }
            case SplitterSorter::Sequential: {
                using SplitterPolicy = Sequential<Char, indexed>;
                using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                return disptach_policy.template operator()<PartitionPolicy>();
            }
        }
        tlx_die("unknown splitter sorter");
    };

    auto dispatch_sampler = [&]<bool indexed, bool random> {
        using namespace dss_mehnert::sample;

        if (sampler.sample_chars) {
            using SamplePolicy = CharBasedSampling<indexed, random>;
            return dispatch_sorter.template operator()<SamplePolicy>();
        } else {
            using SamplePolicy = StringBasedSampling<indexed, random>;
            return dispatch_sorter.template operator()<SamplePolicy>();
        }
    };

    auto dispatch_random = [&]<bool indexed> {
        if (sampler.sample_random) {
            return dispatch_sampler.template operator()<indexed, true>();
        } else {
            return dispatch_sampler.template operator()<indexed, false>();
        }
    };

    if (sampler.sample_indexed) {
        return dispatch_random.template operator()<true>();
    } else {
        return dispatch_random.template operator()<false>();
    }
}

} // namespace dss_mehnert
