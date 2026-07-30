// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include "bench/algorithms/merge_sort.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <kamping/collectives/barrier.hpp>
#include <kamping/measurements/timer.hpp>
#include <tlx/die.hpp>

#include "bench/dispatch.hpp"
#include "bench/input.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/print_strings.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/partition_policy_factory.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace bench {
namespace {

template <typename CharType, typename AlltoallConfig, typename RedistributionPolicy>
class MergeSortAlgorithm : public AlgorithmBase {
    static constexpr auto alltoall_config = AlltoallConfig();

    using StringSet = dss_mehnert::StringSet<CharType, Length>;
    using PartitionPolicy = MergeSortPartitionPolicy<CharType>;
    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using MergeSort =
        sorter::DistributedMergeSort<alltoall_config, RedistributionPolicy, PartitionPolicy>;

public:
    MergeSortAlgorithm(
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
        measuring_tool.enableCommVolume();

        comm_.barrier();

        measuring_tool.start("none", "create_communicators");
        kamping::measurements::timer().synchronize_and_start("create_communicators");
        auto const first_level = get_first_level(args_.levels, comm_);
        comms_.emplace(first_level, args_.levels.end(), comm_);
        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("none", "create_communicators", comm_);

        [[maybe_unused]] std::size_t volatile warmup_sink = 0;
        for (std::size_t i = 0; i < args_.mpi_warmup_rounds; ++i) {
            kamping::measurements::timer().synchronize_and_start("warmup-round");
            warmup_sink += mpi_irregular_warmup(50000, 50500, comms_->comm_root());
            kamping::measurements::timer().stop_and_append();
        }
    }

    void run() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        comm_.barrier();
        measuring_tool.start("none", "sorting_overall");
        kamping::measurements::timer().synchronize_and_start("sorting_overall");
        merge_sort_.sort(input_container_, *comms_, args_.sampler.splitter_length_factor);
        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("none", "sorting_overall", comm_);
    }

    void verify() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.disableCommVolume();

        if (args_.count_prefixes) {
            count_prefix_lengths(input_container_, comm_);
        }

        measuring_tool.disable();

        if (args_.check_sorted) {
            auto const is_sorted = checker_.is_sorted(input_container_.make_string_set(), comm_);
            die_verbose_unless(is_sorted, "output is not sorted");
            auto const is_complete = checker_.is_complete(input_container_, comm_);
            die_verbose_unless(is_complete, "output is missing chars or strings");
        }
        if (args_.check_complete) {
            auto const is_exact = checker_.check_exhaustive(input_container_, comm_);
            die_verbose_unless(is_exact, "output is not a permutation of the input");
        }
        if (args_.print_sorted) {
            gather_and_print_strings(input_container_, comm_);
        }
    }

private:
    MergeSort merge_sort_;
    StringLcpContainer<StringSet> input_container_;
    MergeSortChecker<StringSet> checker_;
    std::optional<Subcommunicators> comms_;
};

} // namespace

std::unique_ptr<AbstractAlgorithm>
make_merge_sort(SorterArgs const& args, Communicator const& comm) {
    using CharType = unsigned char;

    return dispatch_alltoall_config(args, [&]<typename AlltoallConfig> {
        using StringSet = dss_mehnert::StringSet<CharType, Length>;

        return dispatch_redistribution<StringSet>(
            args,
            [&]<typename RedistributionPolicy>(RedistributionPolicy redistribution) {
                using Algorithm =
                    MergeSortAlgorithm<CharType, AlltoallConfig, RedistributionPolicy>;
                return std::unique_ptr<AbstractAlgorithm>{
                    new Algorithm{args, comm, std::move(redistribution)}
                };
            }
        );
    });
}

} // namespace bench
} // namespace dss_mehnert
