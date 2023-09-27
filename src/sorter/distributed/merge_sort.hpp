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

// todo this could also be an _internal namespace
// todo remove defaulted environments (dss_schimek::mpi::environment env;)
template <
    typename Subcommunicators,
    typename RedistributionPolicy,
    typename AllToAllStringPolicy,
    typename SamplePolicy,
    typename PartitionPolicy>
class BaseDistributedMergeSort : private AllToAllStringPolicy, private SamplePolicy {
public:
    explicit BaseDistributedMergeSort(SamplePolicy sampler) : SamplePolicy{std::move(sampler)} {}

protected:
    using Communicator = Subcommunicators::Communicator;

    using MeasuringTool = measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    template <
        typename Container,
        typename ExtraArg,
        typename Subcommunicators_ = Subcommunicators,
        std::enable_if_t<Subcommunicators_::is_multi_level, bool> = true>
    auto sort_partial(
        Container&& container,
        multi_level::Level<Communicator> const& level,
        ExtraArg const extra_arg
    ) {
        auto const& comm_orig = level.comm_orig;
        auto const& comm_exchange = level.comm_exchange;

        auto strptr = container.make_string_lcp_ptr();
        measuring_tool_.add(container.size(), "num_strings");

        auto num_groups = level.num_groups();
        auto group_size = level.group_size();

        measuring_tool_.add(num_groups, "num_groups");
        measuring_tool_.add(group_size, "group_size");

        measuring_tool_.start("sample_splitters");
        auto sample =
            SamplePolicy::sample_splitters(strptr.active(), num_groups, extra_arg, comm_orig);
        measuring_tool_.stop("sample_splitters");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        auto interval_sizes =
            PartitionPolicy::compute_partition(strptr, std::move(sample), num_groups, comm_orig);
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "exchange_and_merge");
        auto const& ss = strptr.active();
        auto send_counts = RedistributionPolicy::compute_send_counts(ss, interval_sizes, level);
        assert_equal(send_counts.size(), comm_exchange.size());
        auto sorted_container =
            exchange_and_merge(std::move(container), send_counts, comm_exchange, extra_arg);
        measuring_tool_.stop("sort_globally", "exchange_and_merge");

        return sorted_container;
    }

    template <typename Container, typename ExtraArg>
    auto
    sort_exhaustive(Container&& container, Communicator const& comm, ExtraArg const extra_arg) {
        auto strptr = container.make_string_lcp_ptr();
        measuring_tool_.add(container.size(), "num_strings");

        measuring_tool_.start("sample_splitters");
        auto sample = SamplePolicy::sample_splitters(strptr.active(), comm.size(), extra_arg, comm);
        measuring_tool_.stop("sample_splitters");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        auto interval_sizes =
            PartitionPolicy::compute_partition(strptr, std::move(sample), comm.size(), comm);
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "exchange_and_merge");
        auto sorted_container =
            exchange_and_merge(std::move(container), interval_sizes, comm, extra_arg);
        measuring_tool_.stop("sort_globally", "exchange_and_merge");

        return sorted_container;
    }

    template <typename Container, typename ExtraArg>
    auto exchange_and_merge(
        Container&& container,
        std::vector<size_t>& send_counts,
        Communicator const& comm,
        ExtraArg const extra_arg
    ) {
        measuring_tool_.setPhase("string_exchange");
        measuring_tool_.start("all_to_all_strings");

        auto recv_counts = comm.alltoall(kamping::send_buf(send_counts)).extract_recv_buffer();

        auto recv_string_cont = [&, this] {
            if constexpr (std::is_same_v<sample::DistPrefixes, ExtraArg>) {
                return this->AllToAllStringPolicy::alltoallv(
                    container,
                    send_counts,
                    extra_arg.prefixes,
                    comm
                );
            } else {
                return this->AllToAllStringPolicy::alltoallv(container, send_counts, comm);
            };
        }();
        container.delete_all();
        measuring_tool_.stop("all_to_all_strings");

        auto size_strings = recv_string_cont.size();
        auto size_chars = recv_string_cont.char_size() - size_strings;
        measuring_tool_.add(size_strings, "local_num_strings");
        measuring_tool_.add(size_chars, "local_num_chars");

        measuring_tool_.setPhase("merging");
        measuring_tool_.start("merge_strings");

        measuring_tool_.start("compute_ranges");
        std::erase(recv_counts, 0);
        std::vector<size_t> offsets(recv_counts.size());
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), offsets.begin(), size_t{0});

        auto const strptr = recv_string_cont.make_string_lcp_ptr();
        if (!recv_string_cont.empty()) {
            for (auto const offset: offsets) {
                strptr.set_lcp(offset, 0);
            }
        }
        measuring_tool_.stop("compute_ranges");

        measuring_tool_.start("merge_ranges");
        constexpr bool is_compressed = AllToAllStringPolicy::PrefixCompression;
        auto result = choose_merge<is_compressed>(strptr, offsets, recv_counts);
        result.container.set(recv_string_cont.release_raw_strings());
        recv_string_cont.delete_all();
        measuring_tool_.stop("merge_ranges");

        measuring_tool_.start("prefix_decompression");
        if constexpr (is_compressed) {
            auto const& lcps = result.saved_lcps;
            result.container.extend_prefix(lcps.begin(), lcps.end());
        }
        measuring_tool_.stop("prefix_decompression");
        measuring_tool_.stop("merge_strings");

        return std::move(result.container);
    }
};

template <
    typename StringPtr,
    typename Subcommunicators,
    typename RedistributionPolicy,
    typename AllToAllStringPolicy,
    typename SamplePolicy,
    typename PartitionPolicy>
class DistributedMergeSort : private BaseDistributedMergeSort<
                                 Subcommunicators,
                                 RedistributionPolicy,
                                 AllToAllStringPolicy,
                                 SamplePolicy,
                                 PartitionPolicy> {
public:
    using Base = BaseDistributedMergeSort<
        Subcommunicators,
        RedistributionPolicy,
        AllToAllStringPolicy,
        SamplePolicy,
        PartitionPolicy>;

    using Base::Base;

    using StringSet = StringPtr::StringSet;

    StringLcpContainer<StringSet>
    sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms) {
        using namespace kamping;

        // todo make phase names consistent
        auto string_ptr = container.make_string_lcp_ptr();
        this->measuring_tool_.setPhase("local_sorting");

        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
        this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

        if (comms.comm_root().size() == 1) {
            return container;
        }

        this->measuring_tool_.start("avg_lcp");
        auto lcp_begin = string_ptr.lcp();
        auto lcp_end = string_ptr.lcp() + string_ptr.size();
        auto avg_lcp = compute_global_lcp_average(lcp_begin, lcp_end, comms.comm_root());

        // todo this formula is somewhat questionable
        sample::MaxLength max_length{2 * avg_lcp + 8};
        this->measuring_tool_.stop("avg_lcp");

        if constexpr (Subcommunicators::is_multi_level) {
            for (size_t round = 0; auto level: comms) {
                this->measuring_tool_.start("sort_globally", "partial_sorting");
                container = this->sort_partial(std::move(container), level, max_length);
                this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
                this->measuring_tool_.setRound(++round);
            }
        }

        this->measuring_tool_.start("sort_globally", "final_sorting");
        auto sorted_container =
            this->sort_exhaustive(std::move(container), comms.comm_final(), max_length);
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        this->measuring_tool_.setRound(0);
        return sorted_container;
    }
};

} // namespace sorter
} // namespace dss_mehnert
