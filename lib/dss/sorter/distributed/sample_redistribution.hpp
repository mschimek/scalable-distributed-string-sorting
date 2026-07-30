// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/data_buffer.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>

#include "dss/mpi/communicator.hpp"
#include "dss/sorter/distributed/sample.hpp"

namespace dss_mehnert {
namespace sample {

// Seed used by default for the pseudorandom sample redistribution. It only has to
// be identical on all PEs (each PE mixes in its own rank), so a fixed constant is
// enough; the golden-ratio constant just spreads the low bits well.
inline constexpr uint64_t kDefaultRedistributionSeed = 0x9E3779B97F4A7C15ull;

// Pseudorandomly redistributes a sample across the PEs with a pair of alltoallv
// exchanges (chars, and -- for indexed samples -- indices). Each sampled string
// is sent to a uniformly random destination PE; a string's chars and its index
// stay paired, so the global multiset of sample strings -- and hence every
// downstream splitter -- is unchanged, only its distribution over PEs is
// balanced. This keeps RQuick's per-string-count balancing from being skewed by
// PEs that happen to sample unusually long (or unusually many) strings.
template <typename Char, bool is_indexed>
SampleResult<Char, is_indexed> redistribute_random(
    SampleResult<Char, is_indexed>&& sample,
    Communicator const& comm,
    mpi::AlltoallvParams const& alltoallv_params = {},
    uint64_t const seed = kDefaultRedistributionSeed
) {
    int const p = static_cast<int>(comm.size());
    if (p <= 1) {
        return std::move(sample);
    }

    size_t const num_strings =
        static_cast<size_t>(std::count(sample.sample.begin(), sample.sample.end(), Char{0}));

    std::mt19937_64 gen{seed + comm.rank()};
    std::uniform_int_distribution<int> pe_dist{0, p - 1};

    // pass 1: pick a uniformly random destination for each string and count the
    // chars (incl. terminator) and strings destined for each PE.
    std::vector<int> targets(num_strings);
    std::vector<int> char_counts(p, 0), str_counts(p, 0);
    {
        size_t pos = 0;
        for (size_t i = 0; i != num_strings; ++i) {
            size_t const begin = pos;
            while (sample.sample[pos] != Char{0}) {
                ++pos;
            }
            ++pos; // include terminator
            int const t = pe_dist(gen);
            targets[i] = t;
            char_counts[t] += static_cast<int>(pos - begin);
            ++str_counts[t];
        }
    }

    std::vector<int> char_displs(p), str_displs(p);
    std::exclusive_scan(char_counts.begin(), char_counts.end(), char_displs.begin(), 0);
    std::exclusive_scan(str_counts.begin(), str_counts.end(), str_displs.begin(), 0);

    // pass 2: group chars (and indices) by destination, keeping a string's chars
    // and its index in the same relative order so the two exchanges stay aligned.
    std::vector<Char> send_chars(sample.sample.size());
    std::vector<int> char_pos = char_displs;
    std::vector<uint64_t> send_indices;
    std::vector<int> str_pos;
    if constexpr (is_indexed) {
        send_indices.resize(num_strings);
        str_pos = str_displs;
    }
    {
        size_t pos = 0;
        for (size_t i = 0; i != num_strings; ++i) {
            size_t const begin = pos;
            while (sample.sample[pos] != Char{0}) {
                ++pos;
            }
            ++pos; // include terminator
            int const t = targets[i];
            std::copy(
                sample.sample.begin() + begin,
                sample.sample.begin() + pos,
                send_chars.begin() + char_pos[t]
            );
            char_pos[t] += static_cast<int>(pos - begin);
            if constexpr (is_indexed) {
                send_indices[str_pos[t]++] = sample.indices[i];
            }
        }
    }

    // alltoallv the chars (keyed by per-PE char counts) and, if present, the
    // indices (keyed by per-PE string counts). recv counts are derived internally.
    std::vector<size_t> const char_counts_sz{char_counts.begin(), char_counts.end()};

    SampleResult<Char, is_indexed> result;
    result.sample = comm.alltoallv_dispatch(send_chars, char_counts_sz, alltoallv_params);
    if constexpr (is_indexed) {
        std::vector<size_t> const str_counts_sz{str_counts.begin(), str_counts.end()};
        result.local_offset = sample.local_offset;
        result.indices = comm.alltoallv_dispatch(send_indices, str_counts_sz, alltoallv_params);
    }
    return result;
}

// redistribute_random wrapped with a "redistribute_sample" timer and (chars,
// strings) counters after redistribution. Callers already record the pre-
// redistribution size as sample_num_* (before Derived::sort_samples runs), so
// together these give the before/after balancing picture in the harness. The
// timer nests under whatever region the caller started.
template <typename Char, bool is_indexed>
SampleResult<Char, is_indexed> redistribute_random_timed(
    SampleResult<Char, is_indexed>&& sample,
    Communicator const& comm,
    mpi::AlltoallvParams const& alltoallv_params = {},
    uint64_t const seed = kDefaultRedistributionSeed
) {
    using kamping::measurements::GlobalAggregationMode;

    kamping::measurements::timer().start("redistribute_sample");
    auto result = redistribute_random(std::move(sample), comm, alltoallv_params, seed);
    kamping::measurements::timer().stop_and_append();

    auto const num_strings =
        static_cast<std::int64_t>(std::count(result.sample.begin(), result.sample.end(), Char{0}));
    auto const num_chars = static_cast<std::int64_t>(result.sample.size()) - num_strings;
    kamping::measurements::counter().append(
        "redist_sample_num_strings",
        num_strings,
        {GlobalAggregationMode::min, GlobalAggregationMode::max, GlobalAggregationMode::sum}
    );
    kamping::measurements::counter().append(
        "redist_sample_num_chars",
        num_chars,
        {GlobalAggregationMode::min, GlobalAggregationMode::max, GlobalAggregationMode::sum}
    );
    return result;
}

} // namespace sample
} // namespace dss_mehnert
