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

template <typename StringSet>
struct FullStrings : public StringSet {
    static size_t length(typename StringSet::String const& str) { return str.length; }
};

struct Prefixes : public std::span<size_t const> {
    static size_t length(size_t const& length) { return length; }
};

namespace _internal {

template <typename Communicator>
inline std::pair<std::vector<size_t>, std::vector<size_t>>
exscan_and_bcast(std::vector<size_t> const& values, Communicator const& comm) {
    std::vector<size_t> global_prefixes;
    comm.exscan(
        kamping::send_buf(values),
        kamping::recv_buf(global_prefixes),
        kamping::op(std::plus<>{})
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

template <typename Strings>
std::vector<size_t>
compute_char_intervals(Strings const& string, std::vector<size_t> const& interval_sizes) {
    std::vector<size_t> char_interval_sizes(interval_sizes.size());

    auto it = string.begin();
    for (size_t i = 0; i != interval_sizes.size(); it += interval_sizes[i++]) {
        auto op = [](auto const& acc, auto const& str) { return acc + Strings::length(str); };
        char_interval_sizes[i] = std::accumulate(it, it + interval_sizes[i], size_t{0}, op);
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
        return derived->impl(FullStrings<StringSet>{ss}, send_counts, level);
    }

    template <typename StringSet>
    std::vector<size_t> compute_send_counts(
        StringSet const& ss,
        std::vector<size_t> const& send_counts,
        sample::MaxLength arg,
        Level<Communicator> const& level
    ) const {
        auto const derived = static_cast<Derived const*>(this);
        return derived->impl(FullStrings<StringSet>{ss}, send_counts, level);
    }

    template <typename StringSet>
    std::vector<size_t> compute_send_counts(
        StringSet const& ss,
        std::vector<size_t> const& send_counts,
        sample::DistPrefixes arg,
        Level<Communicator> const& level
    ) const {
        auto const derived = static_cast<Derived const*>(this);
        return derived->impl(Prefixes{arg.prefixes}, send_counts, level);
    }
};

template <typename Communicator>
class NoRedistribution
    : public RedistributionBase<NoSplit<Communicator>, NoRedistribution<Communicator>> {
public:
    template <typename Strings>
    static std::vector<size_t>
    impl(Strings const&, std::vector<size_t> const&, Level<Communicator> const&) {
        tlx_die("redistribution policy does not support multi-level sort");
    }
};

template <typename Communicator>
class GridwiseRedistribution
    : public RedistributionBase<GridwiseSplit<Communicator>, GridwiseRedistribution<Communicator>> {
public:
    template <typename Strings>
    static std::vector<size_t>
    impl(Strings const&, std::vector<size_t> const& intervals, Level<Communicator> const& level) {
        // nothing to do here, intervals are same shape as column communicator
        tlx_assert_equal(intervals.size(), level.comm_exchange.size());
        return intervals;
    }
};

template <typename Communicator>
class NaiveRedistribution
    : public RedistributionBase<RowwiseSplit<Communicator>, NaiveRedistribution<Communicator>> {
public:
    template <typename Strings>
    static std::vector<size_t>
    impl(Strings const&, std::vector<size_t> const& intervals, Level<Communicator> const& level) {
        auto const group_size = level.group_size(), num_groups = level.num_groups();
        auto const group_offset = level.comm_orig.rank() / num_groups;

        // PE `i` sends strings to the `i * p' / p`th member of each group on the next level
        std::vector<size_t> send_counts(level.comm_orig.size());

        for (size_t rank = group_offset; auto const interval: intervals) {
            send_counts[rank] = interval;
            rank += group_size;
        }

        return send_counts;
    }
};

template <typename Communicator>
class SimpleStringRedistribution : public RedistributionBase<
                                       RowwiseSplit<Communicator>,
                                       SimpleStringRedistribution<Communicator>> {
public:
    template <typename Strings>
    static std::vector<size_t>
    impl(Strings const&, std::vector<size_t> const& intervals, Level<Communicator> const& level) {
        auto [global_prefixes, global_sizes] =
            _internal::exscan_and_bcast(intervals, level.comm_orig);

        std::vector<size_t> send_counts(level.comm_orig.size());

        for (size_t group = 0; group != intervals.size(); ++group) {
            auto const prefix = global_prefixes[group];
            auto const capacity = tlx::div_ceil(global_sizes[group], level.group_size());
            auto const rank_in_group = prefix / capacity;

            auto rank = level.group_size() * group + rank_in_group;

            auto const end = prefix + intervals[group];
            for (auto begin = rank_in_group * capacity; begin < end; begin += capacity) {
                auto const lower = std::max(prefix, begin);
                auto const upper = std::min(begin + capacity, end);
                send_counts[rank++] = upper - lower;
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
    template <typename Strings>
    static std::vector<size_t> impl(
        Strings const& strings,
        std::vector<size_t> const& intervals,
        Level<Communicator> const& level
    ) {
        auto const char_intervals = _internal::compute_char_intervals(strings, intervals);
        auto const [global_prefixes, global_sizes] =
            _internal::exscan_and_bcast(char_intervals, level.comm_orig);

        std::vector<size_t> send_counts(level.comm_orig.size());

        auto string_offset = strings.begin();
        for (size_t group = 0; group < intervals.size(); ++group) {
            auto const prefix = global_prefixes[group];
            auto const capacity = tlx::div_ceil(global_sizes[group], level.group_size());
            auto const rank_in_group = prefix / capacity;

            auto rank = level.group_size() * group + rank_in_group;

            size_t string_count = 0, char_count = prefix;
            size_t threshold = (rank_in_group + 1) * capacity;

            for (auto const& str: std::span{string_offset, intervals[group]}) {
                if (char_count > threshold) {
                    send_counts[rank++] = string_count;
                    threshold += capacity;
                    string_count = 0;
                }
                char_count += Strings::length(str);
                ++string_count;
            }
            if (string_count > 0) {
                send_counts[rank] = string_count;
            }
            string_offset += intervals[group];
        }
        return send_counts;
    }
};

namespace _internal {

template <typename Communicator>
std::vector<size_t> compute_deterministic_redistribution(
    size_t const local_size, std::vector<size_t> const& intervals, Level<Communicator> const& level
) {
    // this implementation has latency `O(p)`, completely defeating the original
    // point of the algorithm, and is therefore only useful as a proof of concept.
    namespace kmp = kamping;

    auto const& comm = level.comm_orig;
    size_t const group_size = level.group_size(), num_groups = level.num_groups();
    size_t const global_size =
        comm.allreduce_single(kmp::send_buf(local_size), kmp::op(std::plus<>{}));

    size_t const small_piece = global_size / (2 * comm.size() * num_groups);
    auto is_small_piece = [small_piece](auto const size) { return size <= small_piece; };

    std::vector<size_t> send_counts(comm.size()), recv_counts;
    for (auto dest = send_counts.begin(); auto const interval: intervals) {
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
    template <typename Strings>
    static std::vector<size_t> impl(
        Strings const& strings,
        std::vector<size_t> const& intervals,
        Level<Communicator> const& level
    ) {
        return _internal::compute_deterministic_redistribution(strings.size(), intervals, level);
    }
};

template <typename Communicator>
class DeterministicCharRedistribution : public RedistributionBase<
                                            RowwiseSplit<Communicator>,
                                            DeterministicCharRedistribution<Communicator>> {
public:
    template <typename Strings>
    static std::vector<size_t> impl(
        Strings const& strings,
        std::vector<size_t> const& intervals,
        Level<Communicator> const& level
    ) {
        auto const char_intervals = _internal::compute_char_intervals(strings, intervals);
        auto const sum_chars =
            std::accumulate(char_intervals.begin(), char_intervals.end(), size_t{0});
        auto const char_send_counts =
            _internal::compute_deterministic_redistribution(sum_chars, char_intervals, level);

        std::vector<size_t> send_counts(char_send_counts.size());
        auto it = strings.begin();

        // this loop assumes that the character send counts match up exactly
        // with the string intervals
        size_t char_count = 0, threshold = 0;
        for (size_t i = 0; i != send_counts.size(); ++i) {
            size_t string_count = 0;
            threshold += char_send_counts[i];

            for (; char_count < threshold; ++it, ++string_count) {
                char_count += Strings::length(*it);
            }
            if (char_count < threshold) {
                char_count += Strings::length(*it);
                ++it, ++string_count;
            }
            send_counts[i] = string_count;
        }
        assert(it == strings.end());
        return send_counts;
    }
};

} // namespace redistribution
} // namespace dss_mehnert
