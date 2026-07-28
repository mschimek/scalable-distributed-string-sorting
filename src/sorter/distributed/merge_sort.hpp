// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/bcast.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/alltoall_strings.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/misc.hpp"
#include "sorter/local_sorter.hpp"
#include "sorter/distributed/multi_level.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/sample.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace sorter {

using mpi::AlltoallStringsConfig;

template <AlltoallStringsConfig config, typename RedistributionPolicy, typename PartitionPolicy>
class BaseDistributedMergeSort : protected PartitionPolicy, protected RedistributionPolicy {
public:
    explicit BaseDistributedMergeSort(
        PartitionPolicy partition,
        RedistributionPolicy redistribution,
        mpi::AlltoallvParams alltoallv_params = {},
        LocalSorter local_sorter = LocalSorter::radixsort_CI3
    )
        : PartitionPolicy{std::move(partition)},
          RedistributionPolicy{std::move(redistribution)},
          alltoallv_params_{alltoallv_params},
          local_sorter_{local_sorter} {}

protected:
    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using Communicator = Subcommunicators::Communicator;

    using MeasuringTool = measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    // which alltoallv algorithm the string exchange uses, and how it is tuned
    mpi::AlltoallvParams alltoallv_params_;

    // the sequential sorter used for the local base case
    LocalSorter local_sorter_;

    template <typename StringSet, typename PermutationBuilder>
        requires(StringSet::has_length)
    void sort(
        StringLcpContainer<StringSet>& container,
        Subcommunicators const& comms,
        size_t const splitter_max_length,
        PermutationBuilder& builder
    ) {
        auto const& comm_root = comms.comm_root();

        sample::MaxLength arg{splitter_max_length};

        if constexpr (!Subcommunicators::is_single_level) {
            for (size_t round = 0; auto level: comms) {
                this->measuring_tool_.start("sort_globally", "partial_sorting");
                kamping::measurements::timer().synchronize_and_start("partial_sorting");
                auto const& comm = level.comm_exchange;
                auto const strptr = container.make_string_lcp_ptr();
                auto const send_counts = compute_sorted_send_counts(strptr, arg, level);
                exchange_and_merge(container, send_counts, arg, builder, comm);
                kamping::measurements::timer().stop_and_append();
                this->measuring_tool_.stop("sort_globally", "partial_sorting", comm_root);
                this->measuring_tool_.setRound(++round);
            }
        }

        {
            this->measuring_tool_.start("sort_globally", "final_sorting");
            kamping::measurements::timer().synchronize_and_start("final_sorting");
            auto const strptr = container.make_string_lcp_ptr();
            auto const& comm = comms.comm_final();
            auto const send_counts = compute_sorted_send_counts(strptr, arg, comm);
            exchange_and_merge(container, send_counts, arg, builder, comm);
            kamping::measurements::timer().stop_and_append();
            this->measuring_tool_.stop("sort_globally", "final_sorting", comm_root);
        }

        this->measuring_tool_.setRound(0);
    }

    template <typename StringPtr, typename ExtraArg>
    std::vector<size_t> compute_sorted_send_counts(
        StringPtr const& strptr,
        ExtraArg const extra_arg,
        multi_level::Level<Communicator> const& level
    ) {
        measuring_tool_.add(level.num_groups(), "num_groups");
        measuring_tool_.add(level.group_size(), "group_size");

        measuring_tool_.start("sort_globally", "compute_partition");
        kamping::measurements::timer().start("compute_partition");
        auto const interval_sizes = PartitionPolicy::compute_partition(
            strptr,
            level.num_groups(),
            extra_arg,
            level.comm_orig
        );
        kamping::measurements::timer().stop_and_append();
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "redistribute_strings");
        kamping::measurements::timer().start("redistribute_strings");
        auto const send_counts = RedistributionPolicy::compute_send_counts(
            strptr.active(),
            interval_sizes,
            extra_arg,
            level
        );
        assert_equal(send_counts.size(), level.comm_exchange.size());
        kamping::measurements::timer().stop_and_append();
        measuring_tool_.stop("sort_globally", "redistribute_strings");

        return send_counts;
    }

    template <typename StringPtr, typename ExtraArg>
    std::vector<size_t> compute_sorted_send_counts(
        StringPtr const& strptr, ExtraArg const extra_arg, Communicator const& comm
    ) {
        measuring_tool_.add(1, "num_groups");
        measuring_tool_.add(comm.size(), "group_size");

        measuring_tool_.start("sort_globally", "compute_partition");
        auto const send_counts =
            PartitionPolicy::compute_partition(strptr, comm.size(), extra_arg, comm);
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "redistribute_strings");
        measuring_tool_.stop("sort_globally", "redistribute_strings");

        return send_counts;
    }

    template <typename StringSet, typename ExtraArg, typename PermutationBuilder>
    void exchange_and_merge(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        ExtraArg const extra_arg,
        PermutationBuilder& builder,
        Communicator const& comm
    ) {
        using Permutation = PermutationBuilder::Permutation;

        assert_equal(send_counts.size(), comm.size());
        measuring_tool_.start("sort_globally", "exchange_and_merge");
        kamping::measurements::timer().start("exchange_and_merge");

        measuring_tool_.setPhase("string_exchange");
        measuring_tool_.start("all_to_all_strings");
        comm.barrier();
        kamping::measurements::timer().start("all_to_all_strings");
        kamping::measurements::timer().start("all_to_all_strings_recv_counts");

        std::vector<size_t> recv_counts;
        comm.alltoall(
            kamping::send_buf(send_counts),
            kamping::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(recv_counts)
        );
        kamping::measurements::timer().stop_and_append();

        {
            // strings this PE receives in this round; in the final round that is exactly the
            // size of its bucket, so min/max/sum show how well the splitters balanced them.
            // One value per PE (never one per bucket: that would make the counter's gather
            // quadratic in the number of PEs).
            using kamping::measurements::GlobalAggregationMode;
            std::vector<GlobalAggregationMode> const agg{
                GlobalAggregationMode::min,
                GlobalAggregationMode::max,
                GlobalAggregationMode::sum,
            };
            auto const num_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{0});
            kamping::measurements::counter()
                .append("bucket_size", static_cast<std::int64_t>(num_recv), agg);
        }

        comm.barrier();
        kamping::measurements::timer().start("all_to_all_strings_data");

        if constexpr (std::is_same_v<sample::DistPrefixes, ExtraArg>) {
            auto const& prefixes = extra_arg.prefixes;
            comm.template alltoall_strings<config, Permutation>(
                container,
                send_counts,
                recv_counts,
                prefixes,
                alltoallv_params_
            );
        } else {
            comm.template alltoall_strings<config, Permutation>(
                container,
                send_counts,
                recv_counts,
                alltoallv_params_
            );
        };
        kamping::measurements::timer().stop_and_append();
        kamping::measurements::timer().stop_and_append();
        measuring_tool_.stop("all_to_all_strings");

        measuring_tool_.setPhase("merging");
        measuring_tool_.start("merge_strings");
        kamping::measurements::timer().start("merge_strings");
        measuring_tool_.start("merge_ranges");
        kamping::measurements::timer().start("merge_ranges");
        std::vector<size_t> merge_counts = recv_counts;
        std::erase(merge_counts, 0);

        if (auto& lcps = container.lcps(); !container.empty()) {
            for (size_t i = 0, offset = 0; i != merge_counts.size(); ++i) {
                lcps[offset] = 0;
                offset += merge_counts[i];
            }
        }

        constexpr bool is_compressed = config.compress_prefixes;
        auto const result = merge::choose_merge<is_compressed>(container, merge_counts);
        builder.push(container.make_string_set(), std::move(recv_counts));
        kamping::measurements::timer().stop_and_append();
        measuring_tool_.stop("merge_ranges");

        measuring_tool_.start("prefix_decompression");
        kamping::measurements::timer().start("prefix_decompression");
        if constexpr (is_compressed) {
            container.extend_prefix(result.saved_lcps);
        }
        kamping::measurements::timer().stop_and_append();
        measuring_tool_.stop("prefix_decompression");
        kamping::measurements::timer().stop_and_append();
        measuring_tool_.stop("merge_strings");

        measuring_tool_.add(container.size(), "local_num_strings");
        measuring_tool_.add(container.char_size() - container.size(), "local_num_chars");

        {
            using kamping::measurements::GlobalAggregationMode;
            std::vector<GlobalAggregationMode> const agg{
                GlobalAggregationMode::min,
                GlobalAggregationMode::max,
                GlobalAggregationMode::sum,
                GlobalAggregationMode::gather,
            };
            kamping::measurements::counter()
                .append("local_num_strings", static_cast<std::int64_t>(container.size()), agg);
            kamping::measurements::counter().append(
                "local_num_chars",
                static_cast<std::int64_t>(container.char_size() - container.size()),
                agg
            );
        }

        kamping::measurements::timer().stop_and_append();
        measuring_tool_.stop("sort_globally", "exchange_and_merge");
    }
};

namespace _internal {

class DummyPermutationBuilder {
public:
    using Permutation = NoPermutation;

    template <typename StringSet>
    void push(StringSet const& ss, std::vector<size_t>) {}
};

} // namespace _internal

template <AlltoallStringsConfig config, typename RedistributionPolicy, typename PartitionPolicy>
class DistributedMergeSort
    : private BaseDistributedMergeSort<config, RedistributionPolicy, PartitionPolicy> {
public:
    using Base = BaseDistributedMergeSort<config, RedistributionPolicy, PartitionPolicy>;

    using Base::Base;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;

    template <typename StringSet>
    void sort(
        StringLcpContainer<StringSet>& container,
        Subcommunicators const& comms,
        size_t const splitter_length_factor = 100
    )
        requires(StringSet::has_length)
    {
        auto const& comm_root = comms.comm_root();

        this->measuring_tool_.setPhase("local_sorting");
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        {
            this->measuring_tool_.start("local_sorting", "sort_locally");
            kamping::measurements::timer().synchronize_and_start("sort_locally");
            auto const strptr = container.make_string_lcp_ptr();
            sort_strings_locally(strptr, this->local_sorter_);
            kamping::measurements::timer().stop_and_append();
            this->measuring_tool_.stop("local_sorting", "sort_locally", comm_root);
        }

        if (comm_root.size() > 1) {
            _internal::DummyPermutationBuilder builder;

            this->measuring_tool_.start("avg_lcp");
            auto const avg_lcp = compute_global_lcp_average(container.lcps(), comm_root);
            this->measuring_tool_.stop("avg_lcp");

            Base::sort(container, comms, splitter_length_factor * (avg_lcp + 5), builder);
        }
    }
};

} // namespace sorter
} // namespace dss_mehnert
