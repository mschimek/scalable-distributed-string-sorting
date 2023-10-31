// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <numeric>
#include <string_view>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/bcast.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/math.hpp>

#include "sorter/distributed/multi_level.hpp"
#include "sorter/distributed/sample.hpp"

namespace dss_mehnert {
namespace redistribution {

using dss_mehnert::multi_level::GridwiseSplit;
using dss_mehnert::multi_level::Level;
using dss_mehnert::multi_level::NoSplit;
using dss_mehnert::multi_level::RowwiseSplit;

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

template <typename StringSet>
std::vector<size_t>
compute_char_intervals(StringSet const& ss, std::vector<size_t> const& interval_sizes) {
    std::vector<size_t> char_interval_sizes(interval_sizes.size());

    auto it = ss.begin();
    for (size_t i = 0; i != interval_sizes.size(); it += interval_sizes[i++]) {
        auto get_length = [](auto const& acc, auto const& str) { return acc + str.getLength(); };
        char_interval_sizes[i] = std::accumulate(it, it + interval_sizes[i], size_t{0}, get_length);
    }
    return char_interval_sizes;
}

template <typename StringSet>
std::vector<size_t> compute_char_intervals(
    StringSet const& ss, std::vector<size_t> const& interval_sizes, std::span<size_t const> prefixes
) {
    std::vector<size_t> char_interval_sizes(interval_sizes.size());

    auto it = prefixes.begin();
    for (size_t i = 0; i != interval_sizes.size(); it += interval_sizes[i++]) {
        char_interval_sizes[i] = std::accumulate(it, it + interval_sizes[i], size_t{0});
    }
    return char_interval_sizes;
}


} // namespace _internal

template <typename Subcommunicators_, typename Derived>
class RedistributionBase {
public:
    using Subcommunicators = Subcommunicators_;
    using Communicator = Subcommunicators::Communicator;

    template <typename StringSet>
    std::vector<size_t> compute_send_counts(
        StringSet const& ss,
        std::vector<size_t> const& send_counts,
        sample::NoExtraArg arg,
        Level<Communicator> const& level
    ) const {
        auto const derived = static_cast<Derived const*>(this);
        return derived->impl(ss, send_counts, level);
    }

    template <typename StringSet>
    std::vector<size_t> compute_send_counts(
        StringSet const& ss,
        std::vector<size_t> const& send_counts,
        sample::MaxLength arg,
        Level<Communicator> const& level
    ) const {
        auto const derived = static_cast<Derived const*>(this);
        return derived->impl(ss, send_counts, level);
    }

    template <typename StringSet>
    std::vector<size_t> compute_send_counts(
        StringSet const& ss,
        std::vector<size_t> const& send_counts,
        sample::DistPrefixes arg,
        Level<Communicator> const& level
    ) const {
        auto const derived = static_cast<Derived const*>(this);
        return derived->impl(ss, send_counts, level, arg.prefixes);
    }
};

template <typename Communicator>
class NoRedistribution
    : public RedistributionBase<NoSplit<Communicator>, NoRedistribution<Communicator>> {
public:
    template <typename StringSet, typename... Prefixes>
    static std::vector<size_t> impl(
        StringSet const&,
        std::vector<size_t> const&,
        Level<Communicator> const&,
        Prefixes const&... prefixes
    ) {
        tlx_die("redistribution policy does not support multi-level sort");
    }
};

template <typename Communicator>
class GridwiseRedistribution
    : public RedistributionBase<GridwiseSplit<Communicator>, GridwiseRedistribution<Communicator>> {
public:
    template <typename StringSet, typename... Prefixes>
    static std::vector<size_t> impl(
        StringSet const&,
        std::vector<size_t> const& interval_sizes,
        Level<Communicator> const& level,
        Prefixes const&... prefixes
    ) {
        // nothing to do here, intervals are same shape as column communicator
        tlx_assert_equal(interval_sizes.size(), level.comm_exchange.size());
        return interval_sizes;
    }
};

template <typename Communicator>
class NaiveRedistribution
    : public RedistributionBase<RowwiseSplit<Communicator>, NaiveRedistribution<Communicator>> {
public:
    template <typename StringSet, typename... Prefixes>
    static std::vector<size_t> impl(
        StringSet const&,
        std::vector<size_t> const& interval_sizes,
        Level<Communicator> const& level,
        Prefixes const&... prefixes
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
class SimpleStringRedistribution : public RedistributionBase<
                                       RowwiseSplit<Communicator>,
                                       SimpleStringRedistribution<Communicator>> {
public:
    template <typename StringSet, typename... Prefixes>
    static std::vector<size_t> impl(
        StringSet const&,
        std::vector<size_t> const& interval_sizes,
        Level<Communicator> const& level,
        Prefixes const&... prefixes
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
class SimpleCharRedistribution : public RedistributionBase<
                                     RowwiseSplit<Communicator>,
                                     SimpleCharRedistribution<Communicator>> {
public:
    template <typename StringSet, typename... Prefixes>
    static std::vector<size_t> impl(
        StringSet const& ss,
        std::vector<size_t> const& interval_sizes,
        Level<Communicator> const& level,
        Prefixes const&... prefixes
    ) {
        auto const char_intervals =
            _internal::compute_char_intervals(ss, interval_sizes, prefixes...);
        auto [global_prefixes, global_sizes] =
            _internal::exscan_and_bcast(char_intervals, level.comm_orig);

        std::vector<size_t> send_counts(level.comm_orig.size());

        auto str = ss.begin();
        for (size_t group = 0; group < interval_sizes.size(); ++group) {
            auto const prefix = global_prefixes[group];
            auto const strs_per_rank = tlx::div_ceil(global_sizes[group], level.group_size());
            auto const rank_in_group = prefix / strs_per_rank;

            // todo this loop needs reworking and does not respect prefixes
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
};

namespace _internal {

template <typename Communicator>
std::vector<size_t> compute_deterministic_redistribution(
    size_t const local_size,
    std::vector<size_t> const& interval_sizes,
    Level<Communicator> const& level
) {
    namespace kmp = kamping;

    auto const& comm = level.comm_orig;
    size_t const group_size = level.group_size(), num_groups = level.num_groups();
    size_t const global_size =
        comm.allreduce_single(kmp::send_buf(local_size), kmp::op(std::plus<>{}));

    size_t const small_piece = global_size / (2 * comm.size() * num_groups);
    auto is_small_piece = [small_piece](auto const size) { return size <= small_piece; };

    std::vector<size_t> send_counts(comm.size()), recv_counts;
    for (auto dest = send_counts.begin(); auto const interval: interval_sizes) {
        dest = std::fill_n(dest, group_size, interval);
    }
    comm.alltoall(kmp::send_buf(send_counts), kmp::recv_buf(recv_counts));

    size_t const capacity = tlx::div_ceil(
        std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{0}),
        group_size
    );
    std::vector<size_t> capacities(level.group_size(), capacity);

    // assign small pieces and update capacities
    for (size_t i = 0; i != recv_counts.size(); ++i) {
        if (is_small_piece(recv_counts[i])) {
            capacities[i / num_groups] -= recv_counts[i];
        }
    }

    std::fill(send_counts.begin(), send_counts.end(), size_t{0});
    auto add_request = [&](auto const receiver, auto const sender, auto const count) {
        if (comm.rank() % group_size == receiver) {
            send_counts[sender] = count;
        }
    };

    // create requests for small and large pieces
    for (size_t i = 0, rank = 0; i != recv_counts.size(); ++i) {
        auto recv_count = recv_counts[i];
        if (is_small_piece(recv_count)) {
            add_request(i / num_groups, i, recv_count);
        } else {
            while (capacities[rank] < recv_count) {
                add_request(rank, i, capacities[rank]);
                recv_count -= capacities[rank++];
            }

            if (recv_count > 0) {
                add_request(rank, i, recv_count);
                capacities[rank] -= recv_count;
            }
        }
    }

    comm.alltoall(kmp::send_buf(send_counts), kmp::recv_buf(recv_counts));
    assert_equal(std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{0}), local_size);
    return recv_counts;
}

} // namespace _internal

template <typename Communicator>
class DeterministicStringRedistribution : public RedistributionBase<
                                              RowwiseSplit<Communicator>,
                                              DeterministicStringRedistribution<Communicator>> {
public:
    template <typename StringSet, typename... Prefixes>
    static std::vector<size_t> impl(
        StringSet const& ss,
        std::vector<size_t> const& interval_sizes,
        Level<Communicator> const& level,
        Prefixes const&... prefixes
    ) {
        return _internal::compute_deterministic_redistribution(ss.size(), interval_sizes, level);
    }
};

template <typename Communicator>
class DeterministicCharRedistribution : public RedistributionBase<
                                            RowwiseSplit<Communicator>,
                                            DeterministicCharRedistribution<Communicator>> {
public:
    template <typename StringSet, typename... Prefixes>
    static std::vector<size_t> impl(
        StringSet const& ss,
        std::vector<size_t> const& interval_sizes,
        Level<Communicator> const& level,
        Prefixes const&... prefixes
    ) {
        auto const char_intervals =
            _internal::compute_char_intervals(ss, interval_sizes, prefixes...);
        auto const char_size =
            std::accumulate(char_intervals.begin(), char_intervals.end(), size_t{0});
        auto const send_counts_char =
            _internal::compute_deterministic_redistribution(char_size, char_intervals, level);

        return assign_intervals(ss, send_counts_char, prefixes...);
    }

private:
    template <typename StringSet>
    static std::vector<size_t>
    assign_intervals(StringSet const& ss, std::span<size_t const> send_counts_char) {
        std::vector<size_t> send_counts(send_counts_char.size());
        auto it = ss.begin();
        for (size_t i = 0, sum = 0, next = 0; i != send_counts.size(); ++i) {
            size_t count = 0;
            next += send_counts_char[i];

            for (; sum < next; ++it, ++count) {
                sum += it->length;
            }
            if (sum < next) {
                // todo maybe randomize this decision
                sum += it->length;
                ++it, ++count;
            }
            send_counts[i] = count;
            assert(it <= ss.end());
        }
        return send_counts;
    }

    template <typename StringSet>
    static std::vector<size_t> assign_intervals(
        StringSet const& ss,
        std::span<size_t const> send_counts_char,
        std::span<size_t const> prefixes
    ) {
        std::vector<size_t> send_counts(send_counts_char.size());
        auto it = prefixes.begin();
        for (size_t i = 0, sum = 0, next = 0; i != send_counts.size(); ++i) {
            size_t count = 0;
            next += send_counts_char[i];

            for (; sum < next; ++count) {
                sum += *it++;
            }
            if (sum < next) {
                sum += *it++;
                ++count;
            }
            send_counts[i] = count;
            assert(it <= prefixes.end());
        }
        return send_counts;
    }
};

} // namespace redistribution
} // namespace dss_mehnert
