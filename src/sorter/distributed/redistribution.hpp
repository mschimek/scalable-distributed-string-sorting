// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <numeric>
#include <string_view>
#include <vector>

#include <kamping/collectives/bcast.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/math.hpp>

#include "sorter/distributed/multi_level.hpp"

namespace dss_mehnert {
namespace redistribution {

namespace _internal {

template <typename Communicator>
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
class NoRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "no_redistribution"; }

    template <typename StringSet, bool false_ = false>
    static std::vector<size_t>
    compute_send_counts(StringSet const&, std::vector<size_t> const&, Level const&) {
        static_assert(false_, "redistribution policy does not support multi-level sort");
        return {};
    }
};

template <typename Communicator>
class GridwiseRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "gridwise_redistribution"; }

    template <typename StringSet>
    static std::vector<size_t> compute_send_counts(
        StringSet const&, std::vector<size_t> const& interval_sizes, Level const& level
    ) {
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

    template <typename StringSet>
    static std::vector<size_t> compute_send_counts(
        StringSet const&, std::vector<size_t> const& interval_sizes, Level const& level
    ) {
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

template <typename Communicator>
class SimpleStringRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "simple_string_redistribution"; }

    template <typename StringSet>
    static std::vector<size_t> compute_send_counts(
        StringSet const&, std::vector<size_t> const& interval_sizes, Level const& level
    ) {
        auto [global_prefixes, global_sizes] =
            _internal::exscan_and_bcast(interval_sizes, level.comm_orig);

        std::vector<size_t> send_counts(level.comm_orig.size());

        for (size_t group = 0; group < interval_sizes.size(); ++group) {
            auto const prefix = global_prefixes[group];
            auto const strs_per_rank = tlx::div_ceil(global_sizes[group], level.group_size());
            auto const rank_in_group = prefix / strs_per_rank;

            auto dest = send_counts.begin() + level.group_size() * group + rank_in_group;

            auto const prev_boundary = rank_in_group * strs_per_rank;
            auto const end = prefix + interval_sizes[group];
            for (auto begin = prev_boundary; begin < end; begin += strs_per_rank) {
                auto const lower = std::max(begin, prefix);
                auto const upper = std::min(begin + strs_per_rank, end);
                *dest++ = upper - lower;
            }
        }
        return send_counts;
    }
};

template <typename Communicator>
class SimpleCharRedistribution {
public:
    using Level = multi_level::Level<Communicator>;

    static constexpr std::string_view get_name() { return "simple_char_redistribution"; }

    template <typename StringSet>
    static std::vector<size_t> compute_send_counts(
        StringSet const& ss, std::vector<size_t> const& interval_sizes, Level const& level
    ) {
        // todo maybe static assert has length member
        auto char_interval_sizes = compute_char_sizes(ss, interval_sizes);
        auto [global_prefixes, global_sizes] =
            _internal::exscan_and_bcast(char_interval_sizes, level.comm_orig);

        std::vector<size_t> send_counts(level.comm_orig.size());

        auto str = ss.begin();
        for (size_t group = 0; group < interval_sizes.size(); ++group) {
            auto const prefix = global_prefixes[group];
            auto const strs_per_rank = tlx::div_ceil(global_sizes[group], level.group_size());
            auto const rank_in_group = prefix / strs_per_rank;

            auto dest = send_counts.begin() + level.group_size() * group + rank_in_group;

            size_t sum_chars = prefix, num_strs = 0;
            auto next_boundary = (rank_in_group + 1) * strs_per_rank;
            for (auto const end = str + interval_sizes[group]; str != end; ++str, ++num_strs) {
                if (sum_chars > next_boundary) {
                    *dest++ = std::exchange(num_strs, 0);
                    next_boundary += strs_per_rank;
                }
                sum_chars += ss.get_length(ss[str]);
            }
            *dest = num_strs;
        }
        return send_counts;
    }

private:
    static std::vector<size_t>
    compute_char_sizes(auto const& ss, std::vector<size_t> const& interval_sizes) {
        std::vector<size_t> char_interval_sizes(interval_sizes);
        auto dest = char_interval_sizes.begin();

        for (auto strs = ss.begin(); auto const& interval: interval_sizes) {
            auto get_length = [&ss](auto const& acc, auto const& str) {
                return acc + ss.get_length(str) + 1;
            };
            *dest++ = std::accumulate(strs, strs + interval, size_t{0}, get_length);
            strs += interval;
        }
        return char_interval_sizes;
    }
};

} // namespace redistribution
} // namespace dss_mehnert
