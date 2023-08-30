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
namespace redistribution {

template <typename Communicator>
class NoRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "no_redistribution"; }

    template <bool false_ = false>
    static std::vector<size_t> compute_send_counts(std::vector<size_t> const&, Level const&) {
        static_assert(false_, "redistribution policy does not support multi level sort");
        return {};
    }
};

template <typename Communicator>
class GridwiseRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "gridwise_redistribution"; }

    static std::vector<size_t>
    compute_send_counts(std::vector<size_t> const& interval_sizes, Level const& level) {
        // nothing to do here, intervals are same shape as column communicator
        tlx_assert_equal(interval_sizes.size(), level.comm_exchange.size());
        return interval_sizes;
    }
};

template <typename Communicator>
class NaiveRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "naive_redistribution"; }

    static std::vector<size_t>
    compute_send_counts(std::vector<size_t> const& interval_sizes, Level const& level) {
        auto const& comm = level.comm_orig;
        auto const group_size = level.group_size();
        auto const num_groups = comm.size() / group_size;
        auto const group_offset = comm.rank() / num_groups;

        // PE `i` sends strings to the `i * p' / p`th member of each group on the next level
        std::vector<size_t> send_counts(comm.size());
        auto dst_iter = send_counts.begin() + group_offset;
        for (auto const interval: interval_sizes) {
            *dst_iter = interval;
            dst_iter += group_size;
        }

        return send_counts;
    }
};

namespace _internal {

inline std::pair<std::vector<size_t>, std::vector<size_t>>
exscan_and_bcast(std::vector<size_t> const& values, Communicator const& comm) {
    std::vector<size_t> global_prefixes;
    comm.exscan(
        kamping::send_buf(values),
        kamping::recv_buf(global_prefixes),
        kamping::op(kamping::ops::plus<>{})
    );

    size_t const sentinel = comm.size() - 1;
    std::vector<size_t> global_sizes;
    if (comm.rank() == sentinel) {
        global_sizes.resize(values.size());
        std::transform(
            global_prefixes.begin(),
            global_prefixes.end(),
            values.begin(),
            global_sizes.begin(),
            std::plus<>{}
        );
    }
    comm.bcast(
        kamping::send_recv_buf(global_sizes),
        kamping::recv_counts(static_cast<int>(values.size())),
        kamping::root(sentinel)
    );

    return {std::move(global_prefixes), std::move(global_sizes)};
}

} // namespace _internal

template <typename Communicator>
class SimpleRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "simple_redistribution"; }

    // todo what about chars
    static std::vector<size_t>
    compute_send_counts(std::vector<size_t> const& interval_sizes, Level const& level) {
        auto [global_prefixes, global_sizes] =
            _internal::exscan_and_bcast(interval_sizes, level.comm_orig);

        std::vector<size_t> send_counts(level.comm_orig.size());

        for (size_t group = 0; group < interval_sizes.size(); ++group) {
            auto const prefix = global_prefixes[group];
            auto const elems_per_rank = tlx::div_ceil(global_sizes[group], level.group_size());
            auto const rank_in_group = prefix / elems_per_rank;

            auto dest = send_counts.begin() + level.group_size() * group + rank_in_group;

            auto const prev_boundary = rank_in_group * elems_per_rank;
            auto const end = prefix + interval_sizes[group];
            for (auto begin = prev_boundary; begin < end; begin += elems_per_rank) {
                auto const lower = std::max(begin, prefix);
                auto const upper = std::min(begin + elems_per_rank, end);
                *dest++ = upper - lower;
            }
        }

        return send_counts;
    }
};

} // namespace redistribution


namespace sorter {

// todo this could also be an _internal namespace
// todo remove defaulted environments (dss_schimek::mpi::environment env;)
template <
    typename Subcommunicators,
    typename RedistributionPolicy,
    typename AllToAllStringPolicy,
    typename SamplePolicy,
    typename PartitionPolicy>
class BaseDistributedMergeSort : private AllToAllStringPolicy {
protected:
    using Communicator = Subcommunicators::Communicator;

    using MeasuringTool = measurement::MeasuringTool;
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    template <
        typename Container,
        typename... SampleArgs,
        typename... AllToAllArgs,
        typename Subcommunicators_ = Subcommunicators,
        std::enable_if_t<Subcommunicators_::is_multi_level, bool> = true>
    auto sort_partial(
        Container&& container,
        multi_level::Level<Communicator> const& level,
        std::tuple<SampleArgs...> extra_sample_args,
        std::tuple<AllToAllArgs...> extra_all_to_all_args
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
        auto get_params = [=](auto... args) {
            return sample::SampleParams{num_groups, 2, args...};
        };
        auto params = std::apply(get_params, extra_sample_args);
        auto sample = SamplePolicy::sample_splitters(strptr.active(), params, comm_orig);
        measuring_tool_.stop("sample_splitters");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        auto interval_sizes =
            PartitionPolicy::compute_partition(strptr, std::move(sample), num_groups, comm_orig);
        measuring_tool_.stop("sort_globally", "compute_partition");

        measuring_tool_.start("sort_globally", "exchange_and_merge");
        auto send_counts = RedistributionPolicy::compute_send_counts(interval_sizes, level);
        assert_equal(send_counts.size(), comm_exchange.size());
        auto sorted_container = exchange_and_merge(
            std::move(container),
            send_counts,
            comm_exchange,
            extra_all_to_all_args
        );
        measuring_tool_.stop("sort_globally", "exchange_and_merge");

        return sorted_container;
    }

    template <typename Container, typename... SampleArgs, typename... AllToAllArgs>
    auto sort_exhaustive(
        Container&& container,
        Communicator const& comm,
        std::tuple<SampleArgs...> extra_sample_args,
        std::tuple<AllToAllArgs...> extra_all_to_all_args
    ) {
        auto strptr = container.make_string_lcp_ptr();
        measuring_tool_.add(container.size(), "num_strings");

        measuring_tool_.start("sample_splitters");
        auto get_params = [&](auto... args) {
            return sample::SampleParams{comm.size(), 2, args...};
        };
        auto params = std::apply(get_params, extra_sample_args);
        auto sample = SamplePolicy::sample_splitters(strptr.active(), params, comm);
        measuring_tool_.stop("sample_splitters");

        measuring_tool_.start("sort_globally", "compute_partition");
        measuring_tool_.setPhase("bucket_computation");
        auto interval_sizes =
            PartitionPolicy::compute_partition(strptr, std::move(sample), comm.size(), comm);
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
        auto recv_string_cont = std::apply(alltoallv, extra_all_to_all_args);
        container.deleteAll();
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

        if (recv_string_cont.size() > 0) {
            auto& lcps = recv_string_cont.lcps();
            auto set_lcp = [&lcps](auto const offset) {
                lcps[offset] = 0;
            };
            std::for_each(offsets.begin(), offsets.end(), set_lcp);
        }
        measuring_tool_.stop("compute_ranges");

        measuring_tool_.start("merge_ranges");
        auto result = choose_merge<AllToAllStringPolicy::PrefixCompression>(
            recv_string_cont.make_string_lcp_ptr(),
            offsets,
            recv_counts
        );
        result.container.set(recv_string_cont.release_raw_strings());
        recv_string_cont.deleteAll();
        measuring_tool_.stop("merge_ranges");

        measuring_tool_.start("prefix_decompression");
        if constexpr (AllToAllStringPolicy::PrefixCompression) {
            result.container.extend_prefix(
                result.container.make_string_set(),
                result.saved_lcps.cbegin(),
                result.saved_lcps.cend()
            );
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
    using StringLcpContainer = dss_schimek::StringLcpContainer<typename StringPtr::StringSet>;

    StringLcpContainer sort(StringLcpContainer&& container, Subcommunicators const& comms) {
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
        size_t const lcp_summand = 5u;
        auto lcp_begin = string_ptr.lcp();
        auto lcp_end = string_ptr.lcp() + string_ptr.size();
        auto avg_lcp = compute_global_lcp_average(lcp_begin, lcp_end, comms.comm_root());
        // todo what is the justification for this value
        sample::MaxLength max_length{2 * 100 * avg_lcp + lcp_summand};
        std::tuple extra_sample_args{max_length};
        this->measuring_tool_.stop("avg_lcp");

        if constexpr (Subcommunicators::is_multi_level) {
            for (size_t round = 0; auto level: comms) {
                this->measuring_tool_.start("sort_globally", "partial_sorting");
                container = this->sort_partial(std::move(container), level, extra_sample_args, {});
                this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
                this->measuring_tool_.setRound(++round);
            }
        }

        this->measuring_tool_.start("sort_globally", "final_sorting");
        auto sorted_container =
            this->sort_exhaustive(std::move(container), comms.comm_final(), extra_sample_args, {});
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        this->measuring_tool_.setRound(0);
        return sorted_container;
    }
};

} // namespace sorter
} // namespace dss_mehnert
