// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/bcast.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringtools.hpp"

namespace dss_mehnert {
namespace _internal {

struct LocalSplitterInterval {
    size_t local_sample_size;
    size_t global_prefix;
    size_t num_splitters;
    size_t splitter_dist;

    bool contains(size_t const global_index) const {
        return minimum() <= global_index && global_index < maximum();
    }

    size_t minimum() const { return global_prefix; }
    size_t maximum() const { return global_prefix + local_sample_size; }

    size_t local_index(size_t const global_index) const { return global_index - global_prefix; }

    template <typename StringSet>
    size_t accumulate_lengths(StringSet const& ss) const {
        using std::begin;

        // todo this is not great
        size_t splitter_size = 0;
        for (size_t i = 1; i <= num_splitters; ++i) {
            size_t const global_index = i * splitter_dist;
            if (contains(global_index)) {
                auto const str = ss[begin(ss) + local_index(global_index)];
                splitter_size += ss.get_length(str) + 1;
            }
        }
        return splitter_size;
    }
};

inline LocalSplitterInterval compute_splitter_interval(
    size_t const local_sample_size, size_t const num_partitions, Communicator const& comm
) {
    size_t const global_prefix = comm.exscan_single(
        kamping::send_buf(local_sample_size),
        kamping::op(kamping::ops::plus<>{})
    );

    size_t total_size = global_prefix + local_sample_size;
    comm.bcast_single(kamping::send_recv_buf(total_size), kamping::root(comm.size() - 1));

    size_t const num_splitters = std::min(num_partitions - 1, total_size);
    size_t const splitter_dist = total_size / (num_splitters + 1);

    return {local_sample_size, global_prefix, num_splitters, splitter_dist};
}

} // namespace _internal

template <typename StringSet>
dss_schimek::StringContainer<StringSet>
choose_splitters(StringSet const& samples, size_t const num_partitions) {
    assert(samples.check_order());

    size_t const num_splitters = std::min(num_partitions - 1, samples.size());
    size_t const splitter_dist = samples.size() / (num_splitters + 1);

    dss_schimek::StringContainer<StringSet> container;
    container.strings().reserve(num_splitters);

    for (size_t i = 0; i < num_splitters; ++i) {
        auto const& splitter = samples.at((i + 1) * splitter_dist);
        container.strings().emplace_back(splitter);
    }

    container.orderRawStrings();
    return container;
}

template <typename StringSet>
dss_schimek::StringContainer<StringSet> distributed_choose_splitters(
    StringSet const& ss, size_t const num_partitions, Communicator const& comm
) {
    auto const splitter_interval =
        _internal::compute_splitter_interval(ss.size(), num_partitions, comm);
    auto const splitter_size = splitter_interval.accumulate_lengths(ss);
    std::vector<unsigned char> splitter_chars(splitter_size);

    auto dest = splitter_chars.begin();
    for (size_t i = 1; i <= splitter_interval.num_splitters; ++i) {
        size_t const global_idx = i * splitter_interval.splitter_dist;

        if (splitter_interval.contains(global_idx)) {
            auto const local_idx = splitter_interval.local_index(global_idx);
            auto const& str = ss.at(local_idx);
            auto const length = ss.get_length(str) + 1;
            dest = std::copy_n(ss.get_chars(str, 0), length, dest);
        }
    }

    auto result = comm.allgatherv(kamping::send_buf(splitter_chars));
    return dss_schimek::StringContainer<StringSet>{result.extract_recv_buffer()};
}

template <typename StringSet>
dss_schimek::StringContainer<StringSet> distributed_choose_splitters_indexed(
    StringSet const& ss, size_t const num_partitions, Communicator const& comm
) {
    auto const interval = _internal::compute_splitter_interval(ss.size(), num_partitions, comm);
    auto const splitter_size = interval.accumulate_lengths(ss);
    std::vector<unsigned char> splitter_chars(splitter_size);
    std::vector<uint64_t> splitter_idxs;

    auto dest = splitter_chars.begin();
    for (size_t i = 1; i <= interval.num_splitters; ++i) {
        size_t const global_idx = i * interval.splitter_dist;

        // todo don't assume null termination
        if (interval.contains(global_idx)) {
            auto const local_idx = interval.local_index(global_idx);
            auto const& str = ss.at(local_idx);
            auto const length = ss.get_length(str) + 1;
            dest = std::copy_n(ss.get_chars(str, 0), length, dest);
            splitter_idxs.push_back(str.index);
        }
    }

    return dss_schimek::StringContainer<StringSet>{
        comm.allgatherv(kamping::send_buf(splitter_chars)).extract_recv_buffer(),
        comm.allgatherv(kamping::send_buf(splitter_idxs)).extract_recv_buffer()};
}

template <typename LcpIt>
size_t compute_global_lcp_average(LcpIt const first, LcpIt const last, Communicator const& comm) {
    size_t const local_lcp_sum = std::accumulate(first, last, size_t{0});
    size_t const local_num_strs = std::distance(first, last);

    auto result = comm.allreduce(
        kamping::send_buf({local_lcp_sum, local_num_strs}),
        kamping::op(kamping::ops::plus<>{})
    );

    auto const global_sum = result.extract_recv_buffer();
    return global_sum[0] / std::max(size_t{1}, global_sum[1]);
}

template <typename StringSet>
inline size_t binary_search(StringSet const& ss, typename StringSet::CharIterator const value) {
    auto left = ss.begin(), right = ss.end();

    while (left != right) {
        size_t const dist = (right - left) / 2;
        auto const& current = ss[left + dist];
        int ord = dss_schimek::scmp(ss.get_chars(current, 0), value);
        if (ord < 0) {
            left = left + dist + 1;
        } else if (ord == 0) {
            return left + dist - ss.begin();
        } else {
            right = left + dist;
        }
    }
    return left - ss.begin();
}

template <typename StringSet>
inline size_t binary_search_indexed(
    StringSet const& ss,
    typename StringSet::CharIterator const value,
    uint64_t const index,
    uint64_t const local_offset
) {
    auto scmp_indexed = [=, &ss](auto const other) {
        auto const other_index = other - ss.begin() + local_offset;
        auto const ord = dss_schimek::scmp(ss.get_chars(ss[other], 0), value);
        return ord == 0 ? other_index - index : ord;
    };

    auto left = ss.begin(), right = ss.end();

    while (left != right) {
        size_t const dist = (right - left) / 2;
        int const ord = scmp_indexed(left + dist);
        if (ord < 0) {
            left = left + dist + 1;
        } else if (ord == 0) {
            return left + dist - ss.begin();
        } else {
            right = left + dist;
        }
    }
    return left - ss.begin();
}

template <typename StringSet, typename SplitterSet>
inline std::vector<size_t> compute_interval_sizes(
    StringSet const& ss, SplitterSet const& splitters, size_t const num_partitions
) {
    std::vector<size_t> intervals;
    intervals.reserve(splitters.size());

    size_t const num_splitters = std::min(num_partitions - 1, ss.size());
    size_t const splitter_dist = ss.size() / (num_splitters + 1);

    for (size_t i = 0; i < splitters.size(); ++i) {
        auto it = ss.begin() + (i + 1) * splitter_dist;

        auto const splitter = splitters.get_chars(splitters.at(i), 0);

        while (it != ss.begin() && !dss_schimek::leq(ss.get_chars(ss[it], 0), splitter)) {
            --it;
        }
        while (it < ss.end() && dss_schimek::leq(ss.get_chars(ss[it], 0), splitter)) {
            ++it;
        }
        intervals.emplace_back(it - ss.begin());
    }
    intervals.emplace_back(ss.size());

    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

template <typename StringSet, typename SplitterSet>
inline std::vector<size_t>
compute_interval_binary(StringSet const& ss, SplitterSet const& splitters) {
    std::vector<size_t> intervals;
    intervals.reserve(splitters.size() + 1);

    for (auto const& splitter: splitters) {
        intervals.emplace_back(binary_search(ss, splitters.get_chars(splitter, 0)));
    }
    intervals.emplace_back(ss.size());

    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

template <typename StringSet, typename SplitterSet>
inline std::vector<size_t> compute_interval_binary_index(
    StringSet const& ss, SplitterSet const& splitters, uint64_t const local_offset
) {
    std::vector<size_t> intervals;
    intervals.reserve(splitters.size() + 1);

    for (auto const& splitter: splitters) {
        auto const chars = splitters.get_chars(splitter, 0);
        auto const index = splitter.getIndex();
        intervals.emplace_back(binary_search_indexed(ss, chars, index, local_offset));
    }
    intervals.emplace_back(ss.size());

    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

} // namespace dss_mehnert
