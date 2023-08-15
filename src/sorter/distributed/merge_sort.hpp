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
#include <tuple>
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
template <typename Subcommunicators, typename AllToAllStringPolicy, typename SamplePolicy>
class BaseDistributedMergeSort : private AllToAllStringPolicy {
protected:
    using MeasuringTool = measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    template <typename... SampleArgs, typename... AllToAllArgs>
    auto sort_partial(
        auto&& container,
        multi_level::Level<Communicator> const& level,
        std::tuple<SampleArgs...> extra_sample_args,
        std::tuple<AllToAllArgs...> extra_all_to_all_args
    ) {
        using partition::compute_partition;

        auto const& comm_orig = level.comm_orig;
        auto const& comm_exchange = level.comm_exchange;

        auto string_ptr = container.make_string_lcp_ptr();
        auto num_groups = level.num_groups();
        auto group_size = level.group_size();

        measuring_tool_.add(num_groups, "num_groups");
        measuring_tool_.add(group_size, "group_size");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        auto get_params = [=](auto... args) {
            return sample::SampleParams{num_groups, 2, args...};
        };
        auto params = std::apply(get_params, extra_sample_args);
        auto interval_sizes = compute_partition<SamplePolicy>(string_ptr, params, comm_orig);
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "exchange_and_merge");
        auto send_counts = Subcommunicators::send_counts(interval_sizes, level);
        auto sorted_container = exchange_and_merge(
            std::move(container),
            send_counts,
            comm_exchange,
            extra_all_to_all_args
        );
        measuring_tool_.stop("sort_globally", "exchange_and_merge");

        return sorted_container;
    }

    template <typename... SampleArgs, typename... AllToAllArgs>
    auto sort_exhaustive(
        auto&& container,
        Communicator const& comm,
        std::tuple<SampleArgs...> extra_sample_args,
        std::tuple<AllToAllArgs...> extra_all_to_all_args
    ) {
        using partition::compute_partition;

        auto string_ptr = container.make_string_lcp_ptr();
        measuring_tool_.add(container.size(), "num_strings");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        auto get_params = [&](auto... args) {
            return sample::SampleParams{comm.size(), 2, args...};
        };
        auto params = std::apply(get_params, extra_sample_args);
        auto interval_sizes = compute_partition<SamplePolicy>(string_ptr, params, comm);
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "exchange_and_merge");
        auto sorted_container =
            exchange_and_merge(std::move(container), interval_sizes, comm, extra_all_to_all_args);
        measuring_tool_.stop("sort_globally", "exchange_and_merge");

        return sorted_container;
    }

    template <typename... AllToAllArgs>
    auto exchange_and_merge(
        auto&& container,
        std::vector<size_t>& send_counts,
        Communicator const& comm,
        std::tuple<AllToAllArgs...> extra_all_to_all_args
    ) {
        measuring_tool_.setPhase("string_exchange");
        measuring_tool_.start("all_to_all_strings");

        auto alltoall_result = comm.alltoall(kamping::send_buf(send_counts));
        auto recv_counts = alltoall_result.extract_recv_buffer();

        auto alltoallv = [&, this](auto... args) {
            return this->AllToAllStringPolicy::alltoallv(container, send_counts, args..., comm);
        };
        auto recv_string_container = std::apply(alltoallv, extra_all_to_all_args);
        container.deleteAll();
        measuring_tool_.stop("all_to_all_strings");

        auto size_strings = recv_string_container.size();
        auto size_chars = recv_string_container.char_size() - size_strings;
        measuring_tool_.add(size_strings, "local_num_strings");
        measuring_tool_.add(size_chars, "local_num_chars");

        measuring_tool_.setPhase("merging");
        measuring_tool_.start("merge_strings");
        measuring_tool_.start("compute_ranges");
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

private:
    static constexpr bool debug = false;
};

template <
    typename StringPtr,
    typename Subcommunicators,
    typename AllToAllStringPolicy,
    typename SamplePolicy>
class DistributedMergeSort
    : private BaseDistributedMergeSort<Subcommunicators, AllToAllStringPolicy, SamplePolicy> {
public:
    using StringLcpContainer = dss_schimek::StringLcpContainer<typename StringPtr::StringSet>;

    // todo could simply pass Container by auto here
    StringLcpContainer sort(StringLcpContainer&& container, Subcommunicators const& comms) {
        using namespace kamping;

        // todo make phase names consistent
        auto string_ptr = container.make_string_lcp_ptr();
        auto const& ss = string_ptr.active();
        this->measuring_tool_.setPhase("local_sorting");

        if constexpr (debug) {
            auto op = [&ss](auto sum, auto const& str) { return sum + ss.get_length(str) + 1; };
            size_t chars_in_set = std::accumulate(std::begin(ss), std::end(ss), size_t{0}, op);
            this->measuring_tool_.add(chars_in_set, "chars_in_set");
        }

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
        this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

        if (comms.comm_root().size() == 1)
            return container;

        this->measuring_tool_.start("avg_lcp");
        const size_t lcp_summand = 5u;
        auto avg_lcp = getAvgLcp(string_ptr, comms.comm_root()) + lcp_summand;
        // todo what is the justification for this value
        sample::MaxLength max_length{2 * 100 * avg_lcp};
        std::tuple extra_sample_args{max_length};
        this->measuring_tool_.stop("avg_lcp");

        for (size_t round = 0; auto level: comms) {
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container = this->sort_partial(std::move(container), level, extra_sample_args, {});
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);

            if constexpr (debug) {
                auto local_color = level.comm_group.rank() / level.group_size();
                auto global_color = comms.comm_root().rank() / level.group_size();

                if (level.comm_group.is_root()) {
                    std::cout << "iter=" << round << " group_world=" << global_color
                              << " group=" << local_color << "\n";
                }
            }
        }

        this->measuring_tool_.start("sort_globally", "final_sorting");
        auto sorted_container =
            this->sort_exhaustive(std::move(container), comms.comm_final(), extra_sample_args, {});
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        this->measuring_tool_.setRound(0);
        return sorted_container;
    }

private:
    static constexpr bool debug = false;
};

} // namespace sorter
} // namespace dss_mehnert
