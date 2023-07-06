// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <random>
#include <string_view>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/partition.hpp"
#include "strings/stringtools.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace sorter {

// todo remove defaulted environments (dss_schimek::mpi::environment env;)
// todo investigate wheter RQuick is working correctly (in particular power oft two issue)
template <typename StringPtr, typename AllToAllStringPolicy, typename SamplePolicy>
class DistributedMergeSort : private AllToAllStringPolicy, private SamplePolicy {
public:
    using StringSet = typename StringPtr::StringSet;
    using StringLcpContainer = dss_schimek::StringLcpContainer<StringSet>;

    // todo get rid of NoLcps in AllToAll
    // todo need to consider AllToAllStringPolicy::PrefixCompression?
    // todo why pass both string_ptr and container?
    template <typename Levels>
    StringLcpContainer sort(
        StringPtr& string_ptr,
        StringLcpContainer&& container_,
        Levels&& intermediate_levels,
        Communicator const& comm
    ) {
        measuring_tool_.setPhase("local_sorting");

        StringSet const& ss = string_ptr.active();

        size_t charactersInSet = 0;
        for (auto const& str: ss) {
            charactersInSet += ss.get_length(str) + 1;
        }

        measuring_tool_.add(charactersInSet, "charactersInSet");

        measuring_tool_.start("sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
        measuring_tool_.stop("sort_locally");

        if (comm.size() == 1) {
            measuring_tool_.setPhase("none");
            return container_;
        }

        // todo address imbalance in partitions
        size_t round{0};
        Communicator comm_group{comm};
        // std::vector<size_t> intermediate_levels = {2, 2};
        StringLcpContainer container = std::move(container_);

        for (auto num_groups: intermediate_levels) {
            tlx_die_unless(1 < num_groups && num_groups < comm_group.size());

            auto [color, sorted_container] =
                sort_partial(std::move(container), num_groups, comm_group);
            container = std::move(sorted_container);
            comm_group = comm_group.split(color);
            measuring_tool_.setRound(++round);
        }

        auto sorted_container = sort_exhaustive(std::move(container), comm_group);

        // todo this is a horrible bodge, pls remove
        measuring_tool_.setRound(0);
        measuring_tool_.setPhase("none");
        return sorted_container;
    }

private:
    using MeasuringTool = dss_schimek::measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    std::pair<int, StringLcpContainer>
    sort_partial(StringLcpContainer&& container, size_t num_groups, Communicator const& comm) {
        tlx_die_unless(comm.size() % num_groups == 0);

        auto string_ptr = container.make_string_lcp_ptr();
        auto group_size{comm.size() / num_groups};

        measuring_tool_.setPhase("bucket_computation");
        auto global_lcp_avg = get_avg_lcp(string_ptr, comm);

        constexpr auto compute_partition = partition::compute_partition<StringPtr, SamplePolicy>;
        // todo why the *100 here
        // todo make sampling_factor variable
        auto interval_sizes =
            compute_partition(string_ptr, 100 * global_lcp_avg, num_groups, 2, comm);

        auto group_offset{comm.rank() / num_groups};
        std::vector<uint64_t> send_counts(comm.size());

        // todo replace with non-naive strategy
        auto dst_iter = send_counts.begin() + group_offset;
        for (auto const interval: interval_sizes) {
            *dst_iter = interval;
            dst_iter += group_size;
        }

        comm.barrier(); // todo maybe remove this barrier
        measuring_tool_.setPhase("string_exchange");
        auto [recv_container, recv_intervals] =
            string_exchange(std::move(container), send_counts, comm);

        // todo prune empty ranges before merging
        measuring_tool_.setPhase("merging");
        auto sorted_container = merge_ranges(std::move(recv_container), recv_intervals, comm);
        auto color = comm.rank() / group_size;

        return {color, std::move(sorted_container)};
    }

    StringLcpContainer sort_exhaustive(StringLcpContainer&& container, Communicator const& comm) {
        auto string_ptr = container.make_string_lcp_ptr();

        measuring_tool_.setPhase("bucket_computation");
        auto global_lcp_avg = get_avg_lcp(string_ptr, comm);

        constexpr auto compute_partition = partition::compute_partition<StringPtr, SamplePolicy>;
        // todo why the *100 here
        // todo make sampling_factor variable
        auto interval_sizes =
            compute_partition(string_ptr, 100 * global_lcp_avg, comm.size(), 2, comm);

        comm.barrier();
        measuring_tool_.setPhase("string_exchange");
        auto [recv_container, recv_intervals] =
            string_exchange(std::move(container), interval_sizes, comm);

        measuring_tool_.setPhase("merging");
        auto sorted_container = merge_ranges(std::move(recv_container), recv_intervals, comm);

        return sorted_container;
    }

    size_t get_avg_lcp(StringPtr string_ptr, Communicator const& comm) {
        measuring_tool_.start("avg_lcp");
        const size_t lcp_summand = 5u;
        auto avg_lcp = getAvgLcp(string_ptr, comm) + lcp_summand;
        measuring_tool_.stop("avg_lcp");
        return avg_lcp;
    }

    std::pair<StringLcpContainer, std::vector<size_t>> string_exchange(
        StringLcpContainer&& container,
        std::vector<size_t> const& interval_sizes,
        Communicator const& comm
    ) {
        measuring_tool_.start("all_to_all_strings");

        std::vector<size_t> recv_interval_sizes =
            comm.alltoall(kamping::send_buf(interval_sizes)).extract_recv_buffer();

        StringLcpContainer recv_string_container =
            AllToAllStringPolicy::alltoallv(container, interval_sizes, comm);
        measuring_tool_.stop("all_to_all_strings");

        auto num_received_chars = recv_string_container.char_size() - recv_string_container.size();
        measuring_tool_.add(num_received_chars, "num_received_chars", false);
        measuring_tool_.add(recv_string_container.size(), "num_recv_strings", false);

        return {std::move(recv_string_container), std::move(recv_interval_sizes)};
    }

    StringLcpContainer merge_ranges(
        StringLcpContainer&& container,
        std::vector<size_t>& interval_sizes,
        Communicator const& comm
    ) {
        measuring_tool_.start("compute_ranges");
        std::vector<std::pair<size_t, size_t>> ranges =
            compute_ranges_and_set_lcp_at_start_of_range(container, interval_sizes, comm);
        measuring_tool_.stop("compute_ranges");

        measuring_tool_.start("merge_ranges");
        size_t num_recv_elems =
            std::accumulate(interval_sizes.begin(), interval_sizes.end(), static_cast<size_t>(0u));
        assert(num_recv_elems == recv_string_cont.size());

        auto sorted_container =
            choose_merge<AllToAllStringPolicy>(std::move(container), ranges, num_recv_elems, comm);
        measuring_tool_.stop("merge_ranges");

        return sorted_container;
    }
};

} // namespace sorter
} // namespace dss_mehnert
