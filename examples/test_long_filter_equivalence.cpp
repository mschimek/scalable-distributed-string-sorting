// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

// Increment-1 equivalence test for the long-string splitter filter
// (PLAN_long_splitter_filter.md). On a length-skewed input (many short strings
// plus a few very long ones) it computes the merge-sort partition twice with the
// same (deterministic) sampling policy -- once with the stock indexed RQuickV2
// splitter sorter and once with RQuickV2LongFilter -- and asserts that both
// produce the *identical* interval sizes on every PE. That is the strong
// correctness property from the plan: the filter must not change which splitters
// get chosen, only how balanced the sample sort is.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <kamping/collectives/allreduce.hpp>
#include <kamping/environment.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>

#include "dss/mpi/communicator.hpp"
#include "dss/sorter/distributed/partition.hpp"
#include "dss/sorter/distributed/partition_long_filter.hpp"
#include "dss/sorter/distributed/sample.hpp"
#include "dss/strings/stringcontainer.hpp"
#include "dss/strings/stringset.hpp"
#include "dss/util/measuringTool.hpp"

namespace {

using Char = unsigned char;
using StringSet = dss_mehnert::StringSet<Char, dss_mehnert::Length>;
using Container = dss_mehnert::StringLcpContainer<StringSet>;

// many short strings (len in [min_short, max_short]) plus a few very long ones
// (len in [min_long, max_long]) -- exactly the case char-based sampling handles
// badly and the filter is meant to fix.
std::vector<Char> make_skewed_input(int rank, std::uint32_t seed) {
    std::mt19937 rng{seed ^ static_cast<std::uint32_t>(rank)};
    std::uniform_int_distribution<std::size_t> short_len{3, 12};
    std::uniform_int_distribution<std::size_t> long_len{200, 5000};
    std::uniform_int_distribution<int> char_dist{'a', 'z'};
    std::bernoulli_distribution is_long{0.03};

    std::vector<Char> bytes;
    for (std::size_t i = 0; i < 2000; ++i) {
        std::size_t const len = is_long(rng) ? long_len(rng) : short_len(rng);
        for (std::size_t j = 0; j < len; ++j) {
            bytes.push_back(static_cast<Char>(char_dist(rng)));
        }
        bytes.push_back(0);
    }
    return bytes;
}

} // namespace

int main(int argc, char** argv) {
    using namespace dss_mehnert;

    kamping::Environment env{argc, argv};
    Communicator comm{};

    // We run compute_partition twice in this single process; the MeasuringTool
    // singleton would otherwise abort on the repeated round-0 timer keys (the
    // real multi-level sorter bumps the round between calls).
    measurement::MeasuringTool::measuringTool().disable();

    constexpr std::size_t sampling_factor = 2;
    std::size_t const num_partitions = comm.size();

    using SamplePolicy = sample::CharBasedSampling</*is_indexed=*/true, /*is_random=*/false>;
    using PlainSplitter = partition::RQuickV2<Char, /*is_indexed=*/true, /*use_lcps=*/false>;
    using FilterSplitter = partition::RQuickV2LongFilter<Char, /*is_indexed=*/true, /*use_lcps=*/false>;
    using PlainPartition = partition::PartitionPolicy<SamplePolicy, PlainSplitter>;
    using FilterPartition = partition::PartitionPolicy<SamplePolicy, FilterSplitter>;

    auto run = [&](auto const& partitioner) {
        auto bytes = make_skewed_input(comm.rank(), 0xF117);
        Container container{std::move(bytes)};
        auto strptr = container.make_string_lcp_ptr();
        return partitioner.compute_partition(
            strptr, num_partitions, sample::NoExtraArg{}, comm
        );
    };

    auto const intervals_plain = run(PlainPartition{sampling_factor});
    auto const intervals_filter = run(FilterPartition{sampling_factor});

    bool const local_ok = intervals_plain == intervals_filter;
    auto const global_ok = comm.allreduce_single(
        kamping::send_buf(local_ok ? 1 : 0),
        kamping::op(kamping::ops::min<>{})
    );

    if (comm.rank() == 0) {
        std::cout << "[long-filter-equivalence] num_procs=" << comm.size()
                  << " num_partitions=" << num_partitions
                  << " result=" << (global_ok ? "OK" : "FAIL") << '\n';
    }

    if (!local_ok) {
        std::cout << "  rank " << comm.rank() << " mismatch:\n";
        std::size_t const n = std::max(intervals_plain.size(), intervals_filter.size());
        for (std::size_t i = 0; i < n; ++i) {
            auto const a = i < intervals_plain.size() ? intervals_plain[i] : 0;
            auto const b = i < intervals_filter.size() ? intervals_filter[i] : 0;
            if (a != b) {
                std::cout << "    [" << i << "] plain=" << a << " filter=" << b << '\n';
            }
        }
    }

    return global_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
