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
#include <utility>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/multi_level.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/sample.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace sorter {

// todo remove defaulted environments (dss_schimek::mpi::environment env;)
template <
    typename StringPtr,
    typename Subcommunicators,
    typename AllToAllStringPolicy,
    typename SamplePolicy>
class DistributedMergeSort : private AllToAllStringPolicy {
public:
    static constexpr bool debug = false;

    using StringLcpContainer = dss_schimek::StringLcpContainer<typename StringPtr::StringSet>;

    StringLcpContainer
    sort(StringPtr& string_ptr, StringLcpContainer&& container, Subcommunicators const& comms) {
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

        if (comms.comm_root().size() == 1)
            return container;

        measuring_tool_.start("avg_lcp");
        const size_t lcp_summand = 5u;
        auto avg_lcp = getAvgLcp(string_ptr, comms.comm_root()) + lcp_summand;
        measuring_tool_.stop("avg_lcp");

        for (size_t round = 0; auto level: comms) {
            measuring_tool_.start("sort_globally", "partial_sorting");

            container = sort_partial(std::move(container), level, avg_lcp);

            measuring_tool_.stop("sort_globally", "partial_sorting");
            measuring_tool_.setRound(++round);

            if constexpr (debug) {
                auto const& comm_group = level.comm_group;
                auto local_color = comm_group.rank() / level.group_size();
                auto global_color = comms.comm_root().rank() / level.group_size();

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
        }

        measuring_tool_.start("sort_globally", "final_sorting");
        auto sorted_container = sort_exhaustive(std::move(container), avg_lcp, comms.comm_final());
        measuring_tool_.stop("sort_globally", "final_sorting");

        measuring_tool_.setRound(0);
        return sorted_container;
    }

private:
    using SampleParams = sample::SampleParams<sample::MaxLength>;
    using MeasuringTool = measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    static constexpr auto compute_partition =
        partition::compute_partition<StringPtr, SamplePolicy, SampleParams>;

    StringLcpContainer sort_partial(
        StringLcpContainer&& container,
        multi_level::Level<Communicator> const& level,
        size_t global_lcp_avg
    ) {
        auto string_ptr{container.make_string_lcp_ptr()};
        auto num_groups{level.num_groups()};
        auto group_size{level.group_size()};

        measuring_tool_.add(num_groups, "num_groups");
        measuring_tool_.add(group_size, "group_size");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        SampleParams params{num_groups, 2, {2 * 100 * global_lcp_avg}};
        auto interval_sizes{compute_partition(string_ptr, params, level.comm_orig)};
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "exchange_and_merge");
        auto send_counts{Subcommunicators::send_counts(interval_sizes, level)};
        auto const& comm_exchange{level.comm_exchange};
        auto sorted_container{exchange_and_merge(std::move(container), send_counts, comm_exchange)};
        measuring_tool_.stop("sort_globally", "exchange_and_merge");

        return sorted_container;
    }

    StringLcpContainer sort_exhaustive(
        StringLcpContainer&& container, size_t global_lcp_avg, Communicator const& comm
    ) {
        auto string_ptr = container.make_string_lcp_ptr();
        measuring_tool_.add(container.size(), "num_strings");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        SampleParams params{comm.size(), 2, sample::MaxLength{2 * 100 * global_lcp_avg}};
        auto interval_sizes = compute_partition(string_ptr, params, comm);
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "exchange_and_merge");
        auto sorted_container = exchange_and_merge(std::move(container), interval_sizes, comm);
        measuring_tool_.stop("sort_globally", "exchange_and_merge");

        return sorted_container;
    }

    StringLcpContainer exchange_and_merge(
        StringLcpContainer&& container, std::vector<size_t>& send_counts, Communicator const& comm
    ) {
        measuring_tool_.setPhase("string_exchange");
        measuring_tool_.start("all_to_all_strings");

        auto alltoall_result = comm.alltoall(kamping::send_buf(send_counts));
        auto recv_counts = alltoall_result.extract_recv_buffer();

        auto recv_string_container = AllToAllStringPolicy::alltoallv(container, send_counts, comm);
        container.deleteAll();
        measuring_tool_.stop("all_to_all_strings");

        auto size_strings = recv_string_container.size();
        auto size_chars = recv_string_container.char_size() - size_strings;
        measuring_tool_.add(size_strings, "local_num_strings", false);
        measuring_tool_.add(size_chars, "local_num_chars", false);

        measuring_tool_.setPhase("merging");
        measuring_tool_.start("merge_strings");
        measuring_tool_.start("compute_ranges");
        // prune empty ranges, particularly useful for multi-level
        std::erase(recv_counts, 0);

        auto ranges =
            compute_ranges_and_set_lcp_at_start_of_range(recv_string_container, recv_counts);
        measuring_tool_.stop("compute_ranges");

        measuring_tool_.start("merge_ranges");
        auto num_recv_elems = std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{0});

        auto sorted_container = choose_merge<AllToAllStringPolicy>(
            std::move(recv_string_container),
            ranges,
            num_recv_elems
        );
        measuring_tool_.stop("merge_ranges");

        measuring_tool_.start("prefix_decompression");
        if constexpr (AllToAllStringPolicy::PrefixCompression) {
            auto saved_lcps = std::move(sorted_container.savedLcps());
            sorted_container.extendPrefix(
                sorted_container.make_string_set(),
                saved_lcps.cbegin(),
                saved_lcps.cend()
            );
        }
        measuring_tool_.stop("prefix_decompression");
        measuring_tool_.stop("merge_strings");

        return sorted_container;
    }
};

} // namespace sorter
} // namespace dss_mehnert
