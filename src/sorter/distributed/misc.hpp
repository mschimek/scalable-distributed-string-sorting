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
#include <tlx/math/div_ceil.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringtools.hpp"

namespace dss_mehnert {
namespace _internal {

struct LocalSplitterInterval {
    size_t const global_prefix;
    size_t const splitter_dist;
    size_t const local_begin;
    size_t const local_end;

    size_t size() const { return local_end - local_begin; }

    size_t to_local_index(size_t const global_index) const { return global_index - global_prefix; }
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
    size_t const splitter_dist = std::max<size_t>(1, total_size / (num_splitters + 1));

    auto clamp = [=](auto const x) { return std::clamp<size_t>(x, 1, num_partitions); };
    size_t const local_begin = clamp(tlx::div_ceil(global_prefix, splitter_dist));
    size_t const local_end = clamp(tlx::div_ceil(global_prefix + local_sample_size, splitter_dist));

    assert(comm.is_same_on_all_ranks(splitter_dist));
    return {global_prefix, splitter_dist, local_begin, local_end};
}

} // namespace _internal

template <typename StringSet>
StringContainer<StringSet> choose_splitters(StringSet const& samples, size_t const num_partitions) {
    assert(samples.check_order());

    size_t const num_splitters = std::min(num_partitions - 1, samples.size());
    size_t const splitter_dist = samples.size() / (num_splitters + 1);

    StringContainer<StringSet> container;
    container.get_strings().reserve(num_splitters);

    for (size_t i = 0; i < num_splitters; ++i) {
        auto const& splitter = samples.at((i + 1) * splitter_dist);
        container.get_strings().emplace_back(splitter);
    }

    container.make_contiguous();
    return container;
}

template <typename StringSet>
StringContainer<StringSet> choose_splitters_distributed(
    StringSet const& ss, size_t const num_partitions, Communicator const& comm
) {
    constexpr size_t length_guess = 256;

    auto const splitter_interval =
        _internal::compute_splitter_interval(ss.size(), num_partitions, comm);
    std::vector<typename StringSet::Char> splitter_chars;
    splitter_chars.reserve(splitter_interval.size() * length_guess);

    std::vector<uint64_t> splitter_idxs;
    if constexpr (StringSet::is_indexed) {
        splitter_idxs.reserve(splitter_interval.size());
    }

    for (size_t i = splitter_interval.local_begin; i < splitter_interval.local_end; ++i) {
        size_t const global_idx = i * splitter_interval.splitter_dist;
        size_t const local_idx = splitter_interval.to_local_index(global_idx);
        assert(local_idx < ss.size());

        auto const& str = ss.at(local_idx);
        auto const length = ss.get_length(str);
        auto const chars = ss.get_chars(str, 0);
        splitter_chars.insert(splitter_chars.end(), chars, chars + length);
        splitter_chars.emplace_back(0);

        if constexpr (StringSet::is_indexed) {
            splitter_idxs.emplace_back(str.index);
        }
    }

    auto char_result = comm.allgatherv(kamping::send_buf(splitter_chars));

    if constexpr (StringSet::is_indexed) {
        auto idx_result = comm.allgatherv(kamping::send_buf(splitter_idxs));
        return StringContainer<StringSet>{
            char_result.extract_recv_buffer(),
            make_initializer<Index>(idx_result.extract_recv_buffer())
        };
    } else {
        return StringContainer<StringSet>{char_result.extract_recv_buffer()};
    }
}

inline size_t compute_global_lcp_average(std::span<size_t const> lcps, Communicator const& comm) {
    size_t const local_lcp_sum = std::accumulate(lcps.begin(), lcps.end(), size_t{0});
    auto result = comm.allreduce(
        kamping::send_buf({local_lcp_sum, lcps.size()}),
        kamping::op(kamping::ops::plus<>{})
    );

    auto const global_sum = result.extract_recv_buffer();
    return global_sum[0] / std::max(size_t{1}, global_sum[1]);
}

template <typename StringSet>
inline size_t binary_search(StringSet const& ss, typename StringSet::String const& value) {
    auto left = ss.begin(), right = ss.end();

    while (left != right) {
        size_t const dist = (right - left) / 2;
        auto const ord = ss.scmp(ss[left + dist], value);
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
    typename StringSet::String const& value,
    uint64_t const index,
    uint64_t const local_offset
) {
    auto scmp_indexed = [&](auto const other) {
        auto const other_index = other - ss.begin() + local_offset;
        auto const ord = ss.scmp(ss[other], value);
        return ord == 0 ? other_index <=> index : ord;
    };

    auto left = ss.begin(), right = ss.end();

    while (left != right) {
        size_t const dist = (right - left) / 2;
        auto const ord = scmp_indexed(left + dist);
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
    using dss_schimek::leq;

    std::vector<size_t> intervals;
    intervals.reserve(splitters.size() + 1);

    size_t const num_splitters = std::min(num_partitions - 1, ss.size());
    size_t const splitter_dist = ss.size() / (num_splitters + 1);

    for (size_t i = 0; i < splitters.size(); ++i) {
        auto it = ss.begin() + (i + 1) * splitter_dist;

        auto const splitter = splitters.get_chars(splitters.at(i), 0);

        while (it != ss.begin() && !leq(ss.get_chars(ss[it], 0), splitter)) {
            --it;
        }
        while (it < ss.end() && leq(ss.get_chars(ss[it], 0), splitter)) {
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
compute_interval_binary(StringSet const& ss, SplitterSet const& splitters)
    requires(StringSet::has_length)
{
    using String = StringSet::String;

    std::vector<size_t> intervals;
    intervals.reserve(splitters.size() + 1);

    for (auto const& splitter: splitters) {
        String const splitter_{splitter.string, splitter.length};
        intervals.emplace_back(binary_search(ss, splitter_));
    }
    intervals.emplace_back(ss.size());

    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

template <typename StringSet, typename SplitterSet>
inline std::vector<size_t> compute_interval_binary_index(
    StringSet const& ss, SplitterSet const& splitters, uint64_t const local_offset
) {
    static_assert(StringSet::has_length);

    using String = StringSet::String;

    std::vector<size_t> intervals;
    intervals.reserve(splitters.size() + 1);

    for (auto const& splitter: splitters) {
        String const splitter_{splitter.string, splitter.length};
        intervals.emplace_back(binary_search_indexed(ss, splitter_, splitter.index, local_offset));
    }
    intervals.emplace_back(ss.size());

    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

} // namespace dss_mehnert
