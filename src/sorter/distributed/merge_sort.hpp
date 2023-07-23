// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/communicator.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/sample.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace sorter {

// todo remove defaulted environments (dss_schimek::mpi::environment env;)
template <typename StringPtr, typename AllToAllStringPolicy, typename SamplePolicy>
class DistributedMergeSort : private AllToAllStringPolicy, private SamplePolicy {
public:
    static constexpr bool debug = false;
    static constexpr bool barrier = false;

    using StringLcpContainer = dss_schimek::StringLcpContainer<typename StringPtr::StringSet>;

    // todo why pass both string_ptr and container?
    template <typename Levels>
    StringLcpContainer sort(
        StringPtr& string_ptr,
        StringLcpContainer&& container,
        Levels&& intermediate_levels,
        Communicator const& comm
    ) {
        using namespace kamping;

        auto const& ss = string_ptr.active();
        measuring_tool_.setPhase("local_sorting");

        if constexpr (debug) {
            size_t chars_in_set = 0;
            for (auto const& str: ss) {
                chars_in_set += ss.get_length(str) + 1;
            }
            measuring_tool_.add(chars_in_set, "chars_in_set");
            assert_equal(chars_in_set, container.char_size());
        }

        measuring_tool_.start("sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
        measuring_tool_.stop("sort_locally");

        measuring_tool_.start("avg_lcp");
        const size_t lcp_summand = 5u;
        auto global_lcp_avg = getAvgLcp(string_ptr, comm) + lcp_summand;
        measuring_tool_.stop("avg_lcp");

        if (comm.size() == 1) {
            measuring_tool_.setPhase("none");
            return container;
        }

        size_t round{0};
        Communicator comm_group{comm};

        for (auto num_groups: intermediate_levels) {
            tlx_die_unless(1 < num_groups && num_groups < comm_group.size());

            size_t group_size;
            std::tie(group_size, container) =
                sort_partial(std::move(container), num_groups, global_lcp_avg, comm_group);

            auto local_color = comm_group.rank() / group_size;
            auto global_color = comm.rank() / group_size;

            // todo use create_subcommunicators here
            measuring_tool_.start("split_communicator");
            comm_group = comm_group.split(local_color);
            measuring_tool_.stop("split_communicator");

            if constexpr (debug) {
                size_t size_strings = container.size();
                size_t size_chars = container.char_size();
                auto group_strings = comm_group.reduce(send_buf(size_strings), op(ops::plus<>{}));
                auto group_chars = comm_group.reduce(send_buf(size_chars), op(ops::plus<>{}));
                if (comm_group.is_root()) {
                    std::cout << "iter=" << round << " group_world=" << global_color
                              << " group=" << local_color
                              << " group_strings=" << group_strings.extract_recv_buffer()[0]
                              << " group_chars=" << group_chars.extract_recv_buffer()[0] << "\n";
                }
            }

            measuring_tool_.setRound(++round);
        }

        auto sorted_container = sort_exhaustive(std::move(container), global_lcp_avg, comm_group);

        measuring_tool_.setRound(0);
        measuring_tool_.setPhase("none");
        return sorted_container;
    }

private:
    using MeasuringTool = measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    using SampleParams = sample::SampleParams<sample::MaxLength>;

    static constexpr auto compute_partition =
        partition::compute_partition<StringPtr, SamplePolicy, SampleParams>;

    std::pair<size_t, StringLcpContainer> sort_partial(
        StringLcpContainer&& container,
        size_t num_groups,
        size_t global_lcp_avg,
        Communicator const& comm
    ) {
        tlx_die_unless(comm.size() % num_groups == 0);

        auto string_ptr{container.make_string_lcp_ptr()};
        auto group_size{comm.size() / num_groups};

        measuring_tool_.setPhase("sort_globally");
        measuring_tool_.start("partial_sorting");
        measuring_tool_.add(num_groups, "num_groups");
        measuring_tool_.add(group_size, "group_size");

        measuring_tool_.setPhase("bucket_computation");

        // todo why the *100 here
        // todo make sampling_factor variable
        SampleParams params{num_groups, 2, {2 * 100 * global_lcp_avg}};
        auto interval_sizes{compute_partition(string_ptr, params, comm)};

        auto group_offset{comm.rank() / num_groups};
        std::vector<uint64_t> send_counts(comm.size());

        // todo replace with non-naive strategy
        auto dst_iter = send_counts.begin() + group_offset;
        for (auto const interval: interval_sizes) {
            *dst_iter = interval;
            dst_iter += group_size;
        }

        auto sorted_container = exchange_and_merge(std::move(container), send_counts, comm);

        measuring_tool_.setPhase("sort_globally");
        measuring_tool_.stop("partial_sorting");

        return {group_size, std::move(sorted_container)};
    }

    StringLcpContainer sort_exhaustive(
        StringLcpContainer&& container, size_t global_lcp_avg, Communicator const& comm
    ) {
        auto string_ptr = container.make_string_lcp_ptr();

        measuring_tool_.setPhase("sort_globally");
        measuring_tool_.start("final_sorting");
        measuring_tool_.add(container.size(), "num_strings");

        measuring_tool_.setPhase("bucket_computation");

        // todo why the *100 here
        // todo make sampling_factor variable
        SampleParams params{comm.size(), 2, sample::MaxLength{2 * 100 * global_lcp_avg}};
        auto interval_sizes = compute_partition(string_ptr, params, comm);
        auto sorted_container = exchange_and_merge(std::move(container), interval_sizes, comm);

        measuring_tool_.setPhase("sort_globally");
        measuring_tool_.stop("final_sorting");

        return std::move(sorted_container);
    }

    StringLcpContainer exchange_and_merge(
        StringLcpContainer&& container,
        std::vector<size_t>& interval_sizes,
        Communicator const& comm
    ) {
        measuring_tool_.setPhase("string_exchange");
        measuring_tool_.start("all_to_all_strings");

        auto recv_interval_sizes =
            comm.alltoall(kamping::send_buf(interval_sizes)).extract_recv_buffer();

        StringLcpContainer recv_string_container =
            AllToAllStringPolicy::alltoallv(container, interval_sizes, comm);
        container.deleteAll();
        measuring_tool_.stop("all_to_all_strings");

        auto size_strings = recv_string_container.size();
        auto size_chars = recv_string_container.char_size() - size_strings;
        measuring_tool_.add(size_strings, "local_num_strings", false);
        measuring_tool_.add(size_chars, "local_num_chars", false);

        measuring_tool_.setPhase("merging");
        measuring_tool_.start("pruning_ranges");
        // todo maybe add option to not prune for exhaustive sort
        std::erase(interval_sizes, 0);
        measuring_tool_.stop("pruning_ranges");

        measuring_tool_.start("compute_ranges");
        std::vector<std::pair<size_t, size_t>> ranges =
            compute_ranges_and_set_lcp_at_start_of_range(
                recv_string_container,
                recv_interval_sizes
            );
        measuring_tool_.stop("compute_ranges");

        measuring_tool_.start("merge_ranges");
        size_t num_recv_elems =
            std::accumulate(recv_interval_sizes.begin(), recv_interval_sizes.end(), size_t{0});

        auto sorted_container = choose_merge<AllToAllStringPolicy>(
            std::move(recv_string_container),
            ranges,
            num_recv_elems
        );
        measuring_tool_.stop("merge_ranges");

        measuring_tool_.start("prefix_decompression");
        if (AllToAllStringPolicy::PrefixCompression && comm.size() > 1) {
            std::vector<size_t> saved_lcps = std::move(sorted_container.savedLcps());
            sorted_container.extendPrefix(
                sorted_container.make_string_set(),
                saved_lcps.cbegin(),
                saved_lcps.cend()
            );
        }
        measuring_tool_.stop("prefix_decompression");

        return sorted_container;
    }
};

} // namespace sorter
} // namespace dss_mehnert
