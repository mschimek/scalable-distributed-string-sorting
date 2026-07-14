// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// End-to-end tests for the distributed sorters. Every configuration must, on every input,
// produce a globally sorted output (the concatenation of the PEs' outputs in rank order) that
// is a permutation of the input. The sample-sort based merge sort must additionally keep the
// output balanced across the PEs.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "dss/dss.hpp"
#include "mpi/communicator.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/partition_policy_factory.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "sorter/local_sorter.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "test_util.hpp"

namespace {

using dss_mehnert::LocalSorter;
using dss_mehnert::SamplerArgs;
using dss_mehnert::SplitterSorter;
using dss_test::Char;
using dss_test::Communicator;

// The inputs the sorters are exercised on; each is a hard case for a different reason.
enum class Instance {
    random,          // distinct strings over a large alphabet: the well-behaved base case
    duplicates,      // many equal strings: buckets can not be split between PEs
    all_equal,       // every string identical: every splitter is the same string
    length_skewed,   // a few very long strings among short ones
    common_prefix,   // all strings share a long prefix, so the LCPs matter
    empty_ranks,     // half of the PEs hold no strings at all
    single_string,   // fewer strings in total than there are PEs
    globally_empty,  // no strings anywhere
};

std::string_view name_of(Instance const instance) {
    switch (instance) {
        case Instance::random: return "random";
        case Instance::duplicates: return "duplicates";
        case Instance::all_equal: return "all_equal";
        case Instance::length_skewed: return "length_skewed";
        case Instance::common_prefix: return "common_prefix";
        case Instance::empty_ranks: return "empty_ranks";
        case Instance::single_string: return "single_string";
        case Instance::globally_empty: return "globally_empty";
    }
    return "unknown";
}

std::vector<std::string> make_input(Instance const instance, Communicator const& comm) {
    size_t const rank = comm.rank(), seed = 0xC0FFEE + rank;
    switch (instance) {
        case Instance::random: return dss_test::random_strings(200, 1, 20, seed);
        case Instance::duplicates: return dss_test::duplicate_heavy_strings(200, seed);
        case Instance::all_equal: return std::vector<std::string>(200, "the same string");
        case Instance::length_skewed: return dss_test::length_skewed_strings(200, seed);
        case Instance::common_prefix: return dss_test::common_prefix_strings(200, seed);
        case Instance::empty_ranks:
            return rank % 2 == 0 ? dss_test::random_strings(200, 1, 20, seed)
                                 : std::vector<std::string>{};
        case Instance::single_string:
            return rank == 0 ? std::vector<std::string>{"only"} : std::vector<std::string>{};
        case Instance::globally_empty: return {};
    }
    return {};
}

// The whole correctness contract: sorted across PEs in rank order, and no string invented,
// dropped or corrupted.
void expect_sorted_permutation_of(
    Communicator const& comm,
    std::vector<Char> const& output,
    std::vector<std::string> const& local_input
) {
    auto const sorted = dss_test::gather_in_rank_order(comm, output);
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end())) << "output is not globally sorted";

    auto expected = dss_test::gather_in_rank_order(comm, dss_test::pack(local_input));
    std::sort(expected.begin(), expected.end());
    auto actual = sorted;
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected) << "output is not a permutation of the input";
}

struct Config {
    Instance instance;
    SamplerArgs sampler;
    SplitterSorter splitter_sorter;
    LocalSorter local_sorter;
};

std::string config_name(::testing::TestParamInfo<Config> const& info) {
    auto const& [instance, sampler, splitter_sorter, local_sorter] = info.param;
    std::string name{name_of(instance)};
    name += sampler.sample_chars ? "_chars" : "_strings";
    if (sampler.sample_indexed) {
        name += "_indexed";
    }
    if (sampler.sample_random) {
        name += "_random";
    }
    name += splitter_sorter == SplitterSorter::Sequential ? "_seq" : "_rquick";
    name += local_sorter == LocalSorter::radixsort_CI3 ? "_radix" : "_mkqs";
    return name;
}

std::vector<Config> all_configs() {
    std::vector<Config> configs;
    for (auto const instance:
         {Instance::random,
          Instance::duplicates,
          Instance::all_equal,
          Instance::length_skewed,
          Instance::common_prefix,
          Instance::empty_ranks,
          Instance::single_string,
          Instance::globally_empty}) {
        for (bool const chars: {false, true}) {
            for (bool const indexed: {false, true}) {
                for (bool const random: {false, true}) {
                    // the local sorter is a plain drop-in; alternate it rather than doubling
                    // the number of tests
                    auto const local_sorter = (chars != indexed) ? LocalSorter::multikey_quicksort
                                                                 : LocalSorter::radixsort_CI3;
                    SamplerArgs const sampler{
                        .sample_chars = chars,
                        .sample_indexed = indexed,
                        .sample_random = random,
                        .sampling_factor = 2,
                    };
                    for (auto const splitter_sorter:
                         {SplitterSorter::Sequential, SplitterSorter::RQuickV2}) {
                        configs.push_back({instance, sampler, splitter_sorter, local_sorter});
                    }
                }
            }
        }
    }
    return configs;
}

class MergeSortEndToEnd : public ::testing::TestWithParam<Config> {
    void SetUp() override { dss_test::reset_measurements(); }
};

TEST_P(MergeSortEndToEnd, SortsCorrectly) {
    Communicator comm;
    auto const& config = GetParam();

    auto const input = make_input(config.instance, comm);
    auto packed = dss_test::pack(input); // run_sorter consumes its argument

    auto const output = dss::run_sorter<dss::kDefaultAlltoallConfig>(
        packed,
        comm,
        config.sampler,
        config.splitter_sorter,
        config.local_sorter
    );

    expect_sorted_permutation_of(comm, output, input);
}

INSTANTIATE_TEST_SUITE_P(
    Configs, MergeSortEndToEnd, ::testing::ValuesIn(all_configs()), config_name
);

// The point of sample sort: the buckets, and hence the output, are balanced. A sampler that
// draws its samples from only part of the local input (the omega < 1 and the truncated splitter
// cases) picks splitters that are far too small and piles the input onto a single PE.
TEST(MergeSortBalance, OutputIsBalancedOnABalancedInput) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    size_t const local_size = 2000;

    for (bool const chars: {false, true}) {
        dss_test::reset_measurements();
        SamplerArgs const sampler{
            .sample_chars = chars,
            .sample_indexed = true,
            .sample_random = false,
            .sampling_factor = 2,
        };

        auto packed = dss_test::pack(dss_test::random_strings(local_size, 10, 30, comm.rank()));
        auto const output = dss::run_sorter<dss::kDefaultAlltoallConfig>(
            packed,
            comm,
            sampler,
            SplitterSorter::Sequential,
            LocalSorter::radixsort_CI3
        );

        size_t const num_strings =
            static_cast<size_t>(std::count(output.begin(), output.end(), Char{0}));
        auto const counts = dss_test::allgather_size(comm, num_strings);
        size_t const total = std::accumulate(counts.begin(), counts.end(), size_t{0});
        size_t const max = *std::max_element(counts.begin(), counts.end());

        EXPECT_EQ(total, local_size * comm.size());
        // Theorem 13 bounds the bucket at (1 + r/v) times the average, with r = p buckets and
        // v = f * (p - 1) samples per PE; 2x the average is a comfortable margin for that.
        EXPECT_LE(max, 2 * local_size) << (chars ? "character" : "string") << "-based sampling";
    }
}

// The multi-level sort partitions once per level; each level samples from an input that is
// already locally sorted, so a sampler that only looks at the front of the local array biases
// every splitter towards the small keys.
TEST(MergeSortMultiLevel, SortsCorrectlyWithTwoLevels) {
    Communicator comm;
    if (comm.size() != 4) {
        GTEST_SKIP() << "the level configuration below needs exactly four ranks";
    }

    using StringSet = dss_mehnert::StringSet<Char, dss_mehnert::Length>;
    using PartitionPolicy = dss_mehnert::MergeSortPartitionPolicy<Char>;
    using RedistributionPolicy =
        dss_mehnert::redistribution::NaiveRedistribution<dss_mehnert::Communicator>;
    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using MergeSort = dss_mehnert::sorter::
        DistributedMergeSort<dss::kDefaultAlltoallConfig, RedistributionPolicy, PartitionPolicy>;

    for (bool const chars: {false, true}) {
        dss_test::reset_measurements();
        auto const input = dss_test::random_strings(500, 5, 25, comm.rank());

        std::vector<size_t> const levels{2}; // one intermediate level of two groups of two PEs
        Subcommunicators comms{levels.begin(), levels.end(), comm};

        // two levels plus the final round; the sampling factor is scaled accordingly so the
        // imbalance bounds of the levels do not compound
        SamplerArgs sampler{
            .sample_chars = chars,
            .sample_indexed = true,
            .sample_random = false,
            .sampling_factor = 2,
            .level_adjusted_scaling = true,
        };
        sampler = sampler.scaled_to_levels(levels.size() + 1);

        MergeSort sorter{
            dss_mehnert::init_partition_policy<Char, PartitionPolicy>(
                sampler,
                SplitterSorter::Sequential,
                LocalSorter::radixsort_CI3
            ),
            RedistributionPolicy{},
            {},
            LocalSorter::radixsort_CI3,
        };

        dss_mehnert::StringLcpContainer<StringSet> container{dss_test::pack(input)};
        sorter.sort(container, comms, sampler.splitter_length_factor);

        std::vector<Char> output;
        for (auto const& str: container.get_strings()) {
            output.insert(output.end(), str.getChars(), str.getChars() + str.getLength());
            output.push_back(Char{0});
        }
        expect_sorted_permutation_of(comm, output, input);
    }
}

// RQuick as the top-level sorter (rather than as the splitter sorter). Its output is sorted
// across PEs but not balanced, so only the order and the permutation property are checked.
class RQuickEndToEnd : public ::testing::TestWithParam<Instance> {
    void SetUp() override { dss_test::reset_measurements(); }
};

TEST_P(RQuickEndToEnd, SortsCorrectly) {
    Communicator comm;
    auto const input = make_input(GetParam(), comm);
    auto packed = dss_test::pack(input); // run_rquick consumes its argument

    auto const output = dss::run_rquick(packed, comm, LocalSorter::radixsort_CI3);

    expect_sorted_permutation_of(comm, output, input);
}

INSTANTIATE_TEST_SUITE_P(
    Instances,
    RQuickEndToEnd,
    ::testing::Values(
        Instance::random,
        Instance::duplicates,
        Instance::all_equal,
        Instance::length_skewed,
        Instance::common_prefix,
        Instance::empty_ranks,
        Instance::single_string,
        Instance::globally_empty
    ),
    [](::testing::TestParamInfo<Instance> const& info) { return std::string{name_of(info.param)}; }
);

// The indexed RQuick orders equal strings by their index, which is what makes the splitter sort
// deterministic in the presence of duplicates.
TEST(RQuickIndexed, OrdersEqualStringsByIndex) {
    Communicator comm;
    dss_test::reset_measurements();
    size_t const local_size = 100;

    auto const input = dss_test::duplicate_heavy_strings(local_size, comm.rank());
    auto packed = dss_test::pack(input);
    std::vector<std::uint64_t> indices(local_size);
    std::iota(indices.begin(), indices.end(), comm.rank() * local_size);

    auto const [output, out_indices] =
        dss::run_rquick(packed, indices, comm, LocalSorter::radixsort_CI3);

    expect_sorted_permutation_of(comm, output, input);

    // gather (string, index) pairs in rank order and check the tie-break
    auto const strings = dss_test::gather_in_rank_order(comm, output);
    auto const all_indices = comm.allgatherv(kamping::send_buf(out_indices));
    ASSERT_EQ(strings.size(), all_indices.size());
    for (size_t i = 1; i != strings.size(); ++i) {
        if (strings[i - 1] == strings[i]) {
            EXPECT_LT(all_indices[i - 1], all_indices[i]) << "equal strings are not index-ordered";
        }
    }
}

} // namespace
