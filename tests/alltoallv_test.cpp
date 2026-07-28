// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Tests for the alltoallv layer. The choice of alltoallv algorithm is a pure implementation
// detail of the string exchange: every algorithm has to move exactly the same data, so a sort
// must produce bit-identical output no matter which one is selected, and enabling the
// large-count guard must not change anything either.

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "dss/dss.hpp"
#include "mpi/alltoallv/dispatch.hpp"
#include "mpi/communicator.hpp"
#include "test_util.hpp"

namespace {

using dss_mehnert::mpi::AlltoallvAlgorithm;
using dss_mehnert::mpi::AlltoallvParams;
using dss_mehnert::mpi::_internal::max_int32_critical_count;
using dss_test::Char;
using dss_test::Communicator;

constexpr size_t kIntMax = static_cast<size_t>(std::numeric_limits<int>::max());

std::string_view name_of(AlltoallvAlgorithm const algorithm) {
    switch (algorithm) {
        case AlltoallvAlgorithm::native: return "native";
        case AlltoallvAlgorithm::direct: return "direct";
        case AlltoallvAlgorithm::onefactor: return "onefactor";
        case AlltoallvAlgorithm::pairwise: return "pairwise";
    }
    return "unknown";
}

std::vector<AlltoallvAlgorithm> all_algorithms() {
    return {
        AlltoallvAlgorithm::native,
        AlltoallvAlgorithm::direct,
        AlltoallvAlgorithm::onefactor,
        AlltoallvAlgorithm::pairwise,
    };
}

// ---------------------------------------------------------------------------
// The int32 overflow guard that decides whether `large_counts` falls back to `direct`.
// The algorithms have different pinch points, which is the whole reason the check is
// algorithm-aware: MPI_Alltoallv takes int displacements, so its totals must fit, while the
// point-to-point schedules only ever cast an individual count.
// ---------------------------------------------------------------------------

TEST(AlltoallvOverflowGuard, NativeIsLimitedByTheTotalsButP2PIsNot) {
    // each count is comfortably inside int32, but the total is not
    std::vector<size_t> const counts{1'000'000'000, 1'000'000'000, 1'000'000'000, 0};
    std::vector<size_t> const small{1, 2, 3, 4};
    std::span<size_t const> const big{counts}, sm{small};

    EXPECT_EQ(max_int32_critical_count(AlltoallvAlgorithm::native, big, sm), 3'000'000'000ul);
    EXPECT_GE(max_int32_critical_count(AlltoallvAlgorithm::native, big, sm), kIntMax)
        << "native must fall back once the displacements overflow";

    for (auto const algorithm: {AlltoallvAlgorithm::onefactor, AlltoallvAlgorithm::pairwise}) {
        EXPECT_EQ(max_int32_critical_count(algorithm, big, sm), 1'000'000'000ul)
            << name_of(algorithm);
        EXPECT_LT(max_int32_critical_count(algorithm, big, sm), kIntMax)
            << name_of(algorithm) << " sends per partner, so the total does not matter";
    }
}

TEST(AlltoallvOverflowGuard, ASingleOversizedCountTripsEveryAlgorithm) {
    std::vector<size_t> const huge{kIntMax + 1, 0, 0, 0};
    std::vector<size_t> const small{1, 2, 3, 4};
    std::span<size_t const> const hg{huge}, sm{small};

    for (auto const algorithm:
         {AlltoallvAlgorithm::native, AlltoallvAlgorithm::onefactor, AlltoallvAlgorithm::pairwise}) {
        EXPECT_GE(max_int32_critical_count(algorithm, hg, sm), kIntMax)
            << name_of(algorithm) << " (send side)";
        EXPECT_GE(max_int32_critical_count(algorithm, sm, hg), kIntMax)
            << name_of(algorithm) << " (recv side)";
    }
}

TEST(AlltoallvOverflowGuard, DirectIsExemptAndSmallCountsNeverTrip) {
    std::vector<size_t> const huge{kIntMax + 1, kIntMax + 1};
    std::vector<size_t> const small{1, 2, 3, 4};
    std::span<size_t const> const hg{huge}, sm{small};

    // `direct` uses derived big datatypes, so it is never the thing that overflows
    EXPECT_EQ(max_int32_critical_count(AlltoallvAlgorithm::direct, hg, hg), 0ul);

    for (auto const algorithm: all_algorithms()) {
        EXPECT_LT(max_int32_critical_count(algorithm, sm, sm), kIntMax) << name_of(algorithm);
    }
}

// ---------------------------------------------------------------------------
// End-to-end: the selected algorithm must not be observable in the output.
// ---------------------------------------------------------------------------

struct Instance {
    std::string_view name;
    std::vector<std::string> (*make)(Communicator const&);
};

std::vector<std::string> random_input(Communicator const& comm) {
    return dss_test::random_strings(200, 1, 20, 0xC0FFEE + comm.rank());
}

std::vector<std::string> duplicate_input(Communicator const& comm) {
    return dss_test::duplicate_heavy_strings(200, 0xC0FFEE + comm.rank());
}

std::vector<std::string> skewed_input(Communicator const& comm) {
    // very uneven message sizes, which is where the schedules differ most
    return dss_test::length_skewed_strings(200, 0xC0FFEE + comm.rank());
}

std::vector<std::string> empty_ranks_input(Communicator const& comm) {
    // zero-sized messages: every schedule has to skip them consistently
    return comm.rank() % 2 == 0 ? dss_test::random_strings(200, 1, 20, comm.rank())
                                : std::vector<std::string>{};
}

std::vector<Instance> all_instances() {
    return {
        {"random", &random_input},
        {"duplicates", &duplicate_input},
        {"length_skewed", &skewed_input},
        {"empty_ranks", &empty_ranks_input},
    };
}

std::vector<Char> sort_with(
    Communicator const& comm,
    std::vector<std::string> const& input,
    AlltoallvAlgorithm const algorithm,
    bool const large_counts
) {
    dss_test::reset_measurements();
    auto packed = dss_test::pack(input); // run_sorter consumes its argument
    return dss::run_sorter<dss::kDefaultAlltoallConfig>(
        packed,
        comm,
        dss::kDefaultSamplerArgs,
        dss::SplitterSorter::RQuickV2,
        dss_mehnert::LocalSorter::radixsort_CI3,
        AlltoallvParams{.algorithm = algorithm, .large_counts = large_counts}
    );
}

class AlltoallvAlgorithmsAgree : public ::testing::TestWithParam<size_t> {};

TEST_P(AlltoallvAlgorithmsAgree, EveryAlgorithmProducesTheSameSortedOutput) {
    Communicator comm;
    auto const instance = all_instances()[GetParam()];
    auto const input = instance.make(comm);

    // `native` without the guard is the reference: it is what the sorter used before the
    // alltoallv algorithm became selectable
    auto const reference = sort_with(comm, input, AlltoallvAlgorithm::native, false);

    auto const gathered = dss_test::gather_in_rank_order(comm, reference);
    EXPECT_TRUE(std::is_sorted(gathered.begin(), gathered.end()))
        << "reference output is not globally sorted";

    for (auto const algorithm: all_algorithms()) {
        for (bool const large_counts: {false, true}) {
            auto const output = sort_with(comm, input, algorithm, large_counts);
            EXPECT_EQ(output, reference)
                << "instance=" << instance.name << " algorithm=" << name_of(algorithm)
                << " large_counts=" << large_counts;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Instances,
    AlltoallvAlgorithmsAgree,
    ::testing::Range(size_t{0}, all_instances().size()),
    [](::testing::TestParamInfo<size_t> const& info) {
        return std::string{all_instances()[info.param].name};
    }
);

} // namespace
