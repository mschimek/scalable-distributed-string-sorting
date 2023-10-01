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
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/multi_level.hpp"
#include "sorter/distributed/sample.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace sorter {

template <typename RedistributionPolicy, typename AllToAllStringPolicy, typename PartitionPolicy>
class BaseDistributedMergeSort : protected AllToAllStringPolicy, protected PartitionPolicy {
public:
    explicit BaseDistributedMergeSort(PartitionPolicy partition)
        : PartitionPolicy{std::move(partition)} {}

protected:
    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using Communicator = Subcommunicators::Communicator;

    using MeasuringTool = measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    template <typename StringPtr, typename ExtraArg>
    std::vector<size_t> compute_sorted_send_counts(
        StringPtr const& strptr,
        ExtraArg const extra_arg,
        multi_level::Level<Communicator> const& level
    ) {
        measuring_tool_.add(level.num_groups(), "num_groups");
        measuring_tool_.add(level.group_size(), "group_size");

        measuring_tool_.start("sort_globally", "compute_partition");
        auto const interval_sizes = PartitionPolicy::compute_partition(
            strptr,
            level.num_groups(),
            extra_arg,
            level.comm_orig
        );
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "redistribute_strings");
        auto const send_counts =
            RedistributionPolicy::compute_send_counts(strptr.active(), interval_sizes, level);
        assert_equal(send_counts.size(), level.comm_exchange.size());
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

    template <typename StringSet, typename ExtraArg>
    void exchange_and_merge(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        ExtraArg const extra_arg,
        Communicator const& comm
    ) {
        measuring_tool_.start("sort_globally", "exchange_and_merge");

        measuring_tool_.setPhase("string_exchange");
        measuring_tool_.start("all_to_all_strings");
        auto recv_counts = comm.alltoall(kamping::send_buf(send_counts)).extract_recv_buffer();

        if constexpr (std::is_same_v<sample::DistPrefixes, ExtraArg>) {
            AllToAllStringPolicy::alltoallv(container, send_counts, extra_arg.prefixes, comm);
        } else {
            AllToAllStringPolicy::alltoallv(container, send_counts, comm);
        };
        measuring_tool_.stop("all_to_all_strings");

        auto size_strings = container.size();
        auto size_chars = container.char_size() - size_strings;
        measuring_tool_.add(size_strings, "local_num_strings");
        measuring_tool_.add(size_chars, "local_num_chars");

        measuring_tool_.setPhase("merging");
        measuring_tool_.start("merge_strings");

        measuring_tool_.start("compute_ranges");
        std::erase(recv_counts, 0);
        std::vector<size_t> offsets(recv_counts.size());
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), offsets.begin(), size_t{0});

        if (auto& lcps = container.lcps(); !container.empty()) {
            for (auto const offset: offsets) {
                lcps[offset] = 0;
            }
        }
        measuring_tool_.stop("compute_ranges");

        measuring_tool_.start("merge_ranges");
        constexpr bool is_compressed = AllToAllStringPolicy::PrefixCompression;
        auto const result = merge::choose_merge<is_compressed>(container, offsets, recv_counts);
        measuring_tool_.stop("merge_ranges");

        measuring_tool_.start("prefix_decompression");
        if constexpr (is_compressed) {
            container.extend_prefix(result.saved_lcps);
        }
        measuring_tool_.stop("prefix_decompression");
        measuring_tool_.stop("merge_strings");

        measuring_tool_.stop("sort_globally", "exchange_and_merge");
    }
};

template <typename RedistributionPolicy, typename AllToAllStringPolicy, typename PartitionPolicy>
class DistributedMergeSort : private BaseDistributedMergeSort<
                                 RedistributionPolicy,
                                 AllToAllStringPolicy,
                                 PartitionPolicy> {
public:
    using Base =
        BaseDistributedMergeSort<RedistributionPolicy, AllToAllStringPolicy, PartitionPolicy>;

    using Base::Base;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;

    template <typename StringSet>
    void sort(StringLcpContainer<StringSet>& container, Subcommunicators const& comms)
        requires(has_member<typename StringSet::String, Length>)
    {
        auto const& comm_root = comms.comm_root();

        // todo make phase names consistent
        this->measuring_tool_.setPhase("local_sorting");
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        {
            this->measuring_tool_.start("local_sorting", "sort_locally");
            auto const strptr = container.make_string_lcp_ptr();
            tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
            this->measuring_tool_.stop("local_sorting", "sort_locally", comm_root);
        }

        if (comm_root.size() > 1) {
            this->measuring_tool_.start("avg_lcp");
            auto const avg_lcp = compute_global_lcp_average(container.lcps(), comm_root);

            // todo this formula is somewhat questionable
            sample::MaxLength arg{2 * avg_lcp + 8};
            this->measuring_tool_.stop("avg_lcp");

            if constexpr (!Subcommunicators::is_single_level) {
                for (size_t round = 0; auto level: comms) {
                    this->measuring_tool_.start("sort_globally", "partial_sorting");
                    auto const strptr = container.make_string_lcp_ptr();
                    auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, level);
                    Base::exchange_and_merge(container, send_counts, arg, level.comm_exchange);
                    this->measuring_tool_.stop("sort_globally", "partial_sorting", comm_root);
                    this->measuring_tool_.setRound(++round);
                }
            }

            {
                this->measuring_tool_.start("sort_globally", "final_sorting");
                auto const strptr = container.make_string_lcp_ptr();
                auto const& comm = comms.comm_final();
                auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, comm);
                Base::exchange_and_merge(container, send_counts, arg, comm);
                this->measuring_tool_.stop("sort_globally", "final_sorting", comm_root);
            }

            this->measuring_tool_.setRound(0);
        }
    }
};

} // namespace sorter
} // namespace dss_mehnert
