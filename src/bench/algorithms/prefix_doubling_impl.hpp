// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The prefix-doubling algorithm, parameterized over the permutation type. Included by exactly
// one translation unit per permutation (prefix_doubling_simple.cpp,
// prefix_doubling_multi_level.cpp), each of which instantiates `make_prefix_doubling` for its
// permutation and exposes it as a plain function. Do not include this anywhere else.

#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <kamping/collectives/barrier.hpp>
#include <kamping/measurements/timer.hpp>
#include <tlx/die.hpp>

#include "bench/algorithm.hpp"
#include "bench/algorithms/prefix_doubling.hpp"
#include "bench/dispatch.hpp"
#include "bench/input.hpp"
#include "executables/args.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/print_strings.hpp"
#include "options.hpp"
#include "sorter/distributed/partition_policy_factory.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace bench {
namespace {

template <
    typename CharType,
    typename AlltoallConfig,
    typename BloomFilterPolicy,
    typename Permutation,
    typename RedistributionPolicy>
class PrefixDoublingAlgorithm : public AlgorithmBase {
    static constexpr auto alltoall_config = AlltoallConfig();

    using StringSet = dss_mehnert::StringSet<CharType, IntLength>;
    using PartitionPolicy = PrefixDoublingPartitionPolicy<CharType, IntLength, Permutation>;
    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using MergeSort = sorter::prefix_doubling::PrefixDoublingMergeSort<
        alltoall_config,
        RedistributionPolicy,
        PartitionPolicy,
        BloomFilterPolicy,
        Permutation>;

public:
    PrefixDoublingAlgorithm(
        SorterArgs const& args, Communicator const& comm, RedistributionPolicy redistribution
    )
        : AlgorithmBase{args, comm},
          merge_sort_{
              init_partition_policy<CharType, PartitionPolicy>(
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
          } {}

    void prepare() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.disableCommVolume();

        input_container_ = generate_strings<StringSet>(args_, comm_);

        if (args_.check_sorted || args_.check_complete) {
            checker_.store_container(input_container_);
        }

        // prefix doubling only returns a permutation; keep a copy of the local
        // input so we can materialize the globally sorted strings for printing
        if (args_.print_sorted) {
            copy_container(input_container_, input_copy_);
        }
        measuring_tool.enableCommVolume();

        comm_.barrier();

        measuring_tool.start("none", "create_communicators");
        kamping::measurements::timer().synchronize_and_start("create_communicators");
        auto const first_level = get_first_level(args_.levels, comm_);
        comms_.emplace(first_level, args_.levels.end(), comm_);
        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("none", "create_communicators", comm_);
    }

    void run() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("none", "sorting_overall");
        kamping::measurements::timer().synchronize_and_start("sorting_overall");
        permutation_.emplace(merge_sort_.sort(std::move(input_container_), *comms_));
        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("none", "sorting_overall", comm_);
    }

    void verify() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.disableCommVolume();
        measuring_tool.disable();

        if (args_.check_sorted) {
            auto const is_sorted = checker_.is_sorted(*permutation_, *comms_);
            die_verbose_unless(is_sorted, "output permutation is not sorted");
        }
        if (args_.check_complete) {
            auto const is_complete = checker_.is_complete(*permutation_, *comms_);
            die_verbose_unless(is_complete, "output permutation is not complete");
        }
        if (args_.print_sorted) {
            if constexpr (std::is_same_v<Permutation, SimplePermutation>) {
                auto sorted_container = sorter::prefix_doubling::apply_permutation(
                    input_copy_.make_string_set(),
                    *permutation_,
                    comm_
                );
                gather_and_print_strings(sorted_container, comm_);
            } else if (comm_.is_root()) {
                std::cout << "--print-sorted is only supported for the simple permutation\n";
            }
        }
    }

private:
    MergeSort merge_sort_;
    StringLcpContainer<StringSet> input_container_;
    StringLcpContainer<StringSet> input_copy_;
    PrefixDoublingChecker<StringSet> checker_;
    std::optional<Subcommunicators> comms_;
    std::optional<Permutation> permutation_;
};

// Instantiated once per permutation, by the corresponding translation unit.
template <typename Permutation>
std::unique_ptr<AbstractAlgorithm>
make_prefix_doubling(SorterArgs const& args, Communicator const& comm) {
    using CharType = unsigned char;

    if constexpr (CliOptions::enable_prefix_doubling) {
        return dispatch_alltoall_config(args, [&]<typename AlltoallConfig> {
            return dispatch_bloomfilter(args, [&]<typename BloomFilterPolicy> {
                using StringSet = dss_mehnert::StringSet<CharType, IntLength>;
                using AugmentedStringSet = sorter::AugmentedStringSet<StringSet, Permutation>;

                return dispatch_redistribution<AugmentedStringSet>(
                    args,
                    [&]<typename RedistributionPolicy>(RedistributionPolicy redistribution) {
                        using Algorithm = PrefixDoublingAlgorithm<
                            CharType,
                            AlltoallConfig,
                            BloomFilterPolicy,
                            Permutation,
                            RedistributionPolicy>;
                        return std::unique_ptr<AbstractAlgorithm>{
                            new Algorithm{args, comm, std::move(redistribution)}
                        };
                    }
                );
            });
        });
    } else {
        die_with_feature("CLI_ENABLE_PREFIX_DOUBLING");
    }
}

} // namespace
} // namespace bench
} // namespace dss_mehnert
