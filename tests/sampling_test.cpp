// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Tests for the splitter sampling strategies (src/sorter/distributed/sample.hpp).
//
// The invariant the whole partitioning rests on: a PE holding a share x of the global input
// must contribute a share x of the global sample, and its samples must be spread over its
// *entire* local input. Several of the tests below pin down edge cases where that used to
// break (imbalanced inputs, more samples requested than strings held, truncated splitters).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mpi/communicator.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "test_util.hpp"

namespace {

using namespace dss_mehnert::sample;
using dss_test::Char;
using dss_test::Communicator;

using StringSet = dss_mehnert::StringSet<Char, dss_mehnert::Length>;
using Container = dss_mehnert::StringLcpContainer<StringSet>;

Container make_container(std::vector<std::string> const& strings) {
    return Container{dss_test::pack(strings)};
}

size_t num_samples_in(std::vector<Char> const& sample) {
    return static_cast<size_t>(std::count(sample.begin(), sample.end(), Char{0}));
}

// The number of samples the PE *should* draw: one per omega elements it holds, where omega is
// the global input size divided by the total number of samples. Recomputed here from the
// definition rather than taken from the code under test.
size_t expected_num_samples(size_t const local_size, size_t const global_size, size_t const total) {
    if (local_size == 0 || total == 0 || global_size == 0) {
        return 0;
    }
    double const omega = static_cast<double>(global_size) / static_cast<double>(total);
    return static_cast<size_t>(std::llround(static_cast<double>(local_size) / omega));
}

std::vector<std::string> fixed_length_strings(size_t const count, size_t const len, char const c) {
    return std::vector<std::string>(count, std::string(len, c));
}

// ---------------------------------------------------------------------------------------------
// the sample size computation, in isolation
// ---------------------------------------------------------------------------------------------

TEST(SampleMath, TotalNumSamplesIsRegularSampling) {
    // p * f * (r - 1) samples in total, i.e. f * (r - 1) per PE on a balanced input
    EXPECT_EQ(get_total_num_samples(/*num_partitions=*/4, /*factor=*/2, /*num_pes=*/8), 8 * 2 * 3);
}

TEST(SampleMath, TotalNumSamplesHandlesDegenerateArguments) {
    // a single partition needs no splitters, and must not underflow (num_partitions - 1)
    EXPECT_EQ(get_total_num_samples(1, 2, 8), 0u);
    EXPECT_EQ(get_total_num_samples(0, 2, 8), 0u);
    // a sampling factor of 0 would ask for an empty sample; it is treated as 1
    EXPECT_EQ(get_total_num_samples(4, 0, 8), get_total_num_samples(4, 1, 8));
    EXPECT_EQ(get_total_num_random_samples(0, 8), get_total_num_random_samples(1, 8));
}

TEST(SampleMath, SampleDistanceIsOmega) {
    EXPECT_DOUBLE_EQ(get_sample_distance(1000, 10), 100.0);
    // no samples at all: omega is undefined and reported as 0, not as a division by zero
    EXPECT_DOUBLE_EQ(get_sample_distance(1000, 0), 0.0);
}

TEST(SampleMath, NumSamplesScalesWithLocalSize) {
    auto params = [](size_t const local) {
        return SampleParams{.local_size = local, .max_num_samples = local, .sample_distance = 10.0,
                            .seed = 0};
    };
    // a PE holding four times as much input draws four times as many samples
    EXPECT_EQ(get_num_samples(params(100)), 10u);
    EXPECT_EQ(get_num_samples(params(400)), 40u);
    // a PE holding less than half an omega rounds down to no sample at all
    EXPECT_EQ(get_num_samples(params(4)), 0u);
}

TEST(SampleMath, NumSamplesIsZeroOnEmptyPeAndWithoutSamples) {
    EXPECT_EQ(
        get_num_samples({.local_size = 0, .max_num_samples = 0, .sample_distance = 10.0, .seed = 0}),
        0u
    );
    EXPECT_EQ(
        get_num_samples({.local_size = 100, .max_num_samples = 100, .sample_distance = 0.0, .seed = 0}),
        0u
    );
}

TEST(SampleMath, NumSamplesIsCappedAtTheNumberOfStrings) {
    // character-based sampling: 100 strings of 10 characters each, and omega below one
    // character. A sample is a string, so at most 100 of them can be drawn -- not 1000.
    SampleParams const params{
        .local_size = 1000, // characters
        .max_num_samples = 100, // strings
        .sample_distance = 0.5,
        .seed = 0,
    };
    EXPECT_EQ(get_num_samples(params), 100u);
}

TEST(SampleMath, SamplesCoverTheWholeLocalInputWhenTheCountIsCapped) {
    // omega < 1 asks for more samples than there are strings; the count is capped, and the
    // *effective* distance must be recomputed from the capped count. Using omega directly would
    // squeeze all samples into the first fraction of the local input (and pick splitters that
    // are far too small).
    SampleParams const params{
        .local_size = 10,
        .max_num_samples = 10,
        .sample_distance = 0.4,
        .seed = 0,
    };
    size_t const num_samples = get_num_samples(params);
    ASSERT_EQ(num_samples, 10u);

    double const distance = _internal::get_local_sample_distance(params, num_samples);
    EXPECT_DOUBLE_EQ(distance, 1.0);

    std::vector<size_t> positions;
    for (size_t i = 0; i != num_samples; ++i) {
        positions.push_back(_internal::get_sample_position(i, params.local_size, distance));
    }
    EXPECT_TRUE(std::is_sorted(positions.begin(), positions.end()));
    EXPECT_EQ(positions.front(), 0u);
    EXPECT_EQ(positions.back(), params.local_size - 1); // the last string is reachable
    EXPECT_TRUE(std::adjacent_find(positions.begin(), positions.end()) == positions.end());
}

TEST(SampleMath, SamplePositionsStayInRange) {
    size_t const local_size = 7;
    double const distance = _internal::get_local_sample_distance(
        {.local_size = local_size, .max_num_samples = local_size, .sample_distance = 2.5, .seed = 0},
        3
    );
    for (size_t i = 0; i != 3; ++i) {
        EXPECT_LT(_internal::get_sample_position(i, local_size, distance), local_size);
    }
    // an empty PE draws no samples; the position must not underflow if it is asked anyway
    EXPECT_EQ(_internal::get_sample_position(0, 0, 1.0), 0u);
}

// ---------------------------------------------------------------------------------------------
// string-based sampling
// ---------------------------------------------------------------------------------------------

TEST(StringSampling, BalancedInputDrawsTheRegularSampleCount) {
    Communicator comm;
    size_t const num_partitions = 4, factor = 2;

    auto container = make_container(dss_test::random_strings(100, 5, 15, 0xC0FFEE + comm.rank()));
    auto const sample = StringBasedSampling<false, false>{factor}.sample_splitters(
        container.make_string_set(),
        num_partitions,
        NoExtraArg{},
        comm
    );

    // on a balanced input the omega rule reduces to classic regular sampling
    EXPECT_EQ(num_samples_in(sample.sample), factor * (num_partitions - 1));
}

TEST(StringSampling, ImbalancedInputScalesTheSampleCount) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    size_t const num_partitions = 4, factor = 2;

    // rank r holds (r + 1) * 50 strings, i.e. the input is imbalanced by a factor of p
    size_t const local_size = (comm.rank() + 1) * 50;
    auto container = make_container(dss_test::random_strings(local_size, 5, 15, comm.rank()));
    auto const sample = StringBasedSampling<false, false>{factor}.sample_splitters(
        container.make_string_set(),
        num_partitions,
        NoExtraArg{},
        comm
    );

    auto const sizes = dss_test::allgather_size(comm, local_size);
    size_t const global_size = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
    size_t const total = get_total_num_samples(num_partitions, factor, comm.size());

    // the PE's share of the sample equals its share of the input -- drawing the same number of
    // samples everywhere (as regular sampling classically does) would skew the splitters towards
    // the keys of the small PEs
    EXPECT_EQ(num_samples_in(sample.sample), expected_num_samples(local_size, global_size, total));

    auto const counts = dss_test::allgather_size(comm, num_samples_in(sample.sample));
    EXPECT_TRUE(std::is_sorted(counts.begin(), counts.end()));
    EXPECT_GT(counts.back(), counts.front());
}

TEST(StringSampling, EmptyPeDrawsNoSamples) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }

    bool const is_empty = comm.rank() == 0;
    auto container =
        make_container(is_empty ? std::vector<std::string>{}
                                : dss_test::random_strings(100, 5, 15, comm.rank()));
    auto const sample = StringBasedSampling<true, false>{2}.sample_splitters(
        container.make_string_set(),
        4,
        NoExtraArg{},
        comm
    );

    if (is_empty) {
        EXPECT_EQ(num_samples_in(sample.sample), 0u);
        EXPECT_TRUE(sample.indices.empty());
    } else {
        EXPECT_GT(num_samples_in(sample.sample), 0u);
    }
}

TEST(StringSampling, GloballyEmptyInputIsHandled) {
    Communicator comm;
    Container container{std::vector<Char>{}};
    auto const sample = StringBasedSampling<true, false>{2}.sample_splitters(
        container.make_string_set(),
        4,
        NoExtraArg{},
        comm
    );
    EXPECT_EQ(num_samples_in(sample.sample), 0u);
    EXPECT_TRUE(sample.indices.empty());
}

TEST(StringSampling, MoreSamplesRequestedThanStringsCoversTheWholeInput) {
    Communicator comm;
    // 8 strings per PE but a sample of 8 * 63 per PE requested: omega drops far below one
    size_t const local_size = 8, num_partitions = 64, factor = 8;

    auto container = make_container(dss_test::random_strings(local_size, 5, 15, comm.rank()));
    auto const sample = StringBasedSampling<true, false>{factor}.sample_splitters(
        container.make_string_set(),
        num_partitions,
        NoExtraArg{},
        comm
    );

    // every string is sampled exactly once: the count is capped at the number of strings, and
    // the samples still span the whole local input. Scaling the positions with the (uncapped)
    // omega instead would confine every sample to the first few strings -- which, once the input
    // is locally sorted, means all splitters come from the bottom of the key range.
    ASSERT_EQ(sample.indices.size(), local_size);
    std::vector<size_t> local_positions;
    for (auto const index: sample.indices) {
        local_positions.push_back(index - sample.local_offset);
    }
    std::vector<size_t> expected(local_size);
    std::iota(expected.begin(), expected.end(), size_t{0});
    EXPECT_EQ(local_positions, expected);
}

TEST(StringSampling, IndexedSamplesAreGlobalPositions) {
    Communicator comm;
    size_t const local_size = 100;

    auto container = make_container(dss_test::random_strings(local_size, 5, 15, comm.rank()));
    auto const sample = StringBasedSampling<true, false>{2}.sample_splitters(
        container.make_string_set(),
        4,
        NoExtraArg{},
        comm
    );

    EXPECT_EQ(sample.local_offset, comm.rank() * local_size);
    ASSERT_EQ(sample.indices.size(), num_samples_in(sample.sample));
    for (auto const index: sample.indices) {
        EXPECT_GE(index, sample.local_offset);
        EXPECT_LT(index, sample.local_offset + local_size);
    }
    EXPECT_TRUE(std::is_sorted(sample.indices.begin(), sample.indices.end()));
}

// ---------------------------------------------------------------------------------------------
// character-based sampling
// ---------------------------------------------------------------------------------------------

TEST(CharSampling, BalancedInputDrawsTheRegularSampleCount) {
    Communicator comm;
    size_t const num_partitions = 4, factor = 2;

    auto container = make_container(fixed_length_strings(100, 10, 'a'));
    auto const sample = CharBasedSampling<false, false>{factor}.sample_splitters(
        container.make_string_set(),
        num_partitions,
        NoExtraArg{},
        comm
    );
    EXPECT_EQ(num_samples_in(sample.sample), factor * (num_partitions - 1));
}

TEST(CharSampling, SampleCountNeverExceedsTheNumberOfStrings) {
    Communicator comm;
    // 8 strings of 20 characters, but a sample of 8 * 63 per PE requested: omega drops below a
    // single character. A sample is a string, so at most 8 can be drawn -- capping at the number
    // of *characters* instead would emit the same strings over and over and blow up the sample.
    size_t const local_size = 8;

    auto container = make_container(fixed_length_strings(local_size, 20, 'a'));
    auto const sample = CharBasedSampling<false, false>{8}.sample_splitters(
        container.make_string_set(),
        64,
        NoExtraArg{},
        comm
    );
    EXPECT_LE(num_samples_in(sample.sample), local_size);
}

TEST(CharSampling, TruncatedSplittersStillDrawTheFullSample) {
    Communicator comm;
    size_t const local_size = 100, str_len = 100, max_length = 10;
    size_t const num_partitions = 4, factor = 2;

    auto container = make_container(fixed_length_strings(local_size, str_len, 'a'));
    auto const sample = CharBasedSampling<true, false>{factor}.sample_splitters(
        container.make_string_set(),
        num_partitions,
        MaxLength{max_length},
        comm
    );

    // The sample boundaries must be laid out over the characters the sampler actually walks,
    // i.e. the truncated ones. Spreading them over the full string lengths (10x more characters
    // here) makes the walk run off the end of the local input: most of the sample is then lost,
    // and what remains comes from the front of the array only.
    ASSERT_EQ(num_samples_in(sample.sample), factor * (num_partitions - 1));

    std::vector<size_t> local_positions;
    for (auto const index: sample.indices) {
        local_positions.push_back(index - sample.local_offset);
    }
    EXPECT_TRUE(std::is_sorted(local_positions.begin(), local_positions.end()));
    EXPECT_GT(local_positions.back(), local_size / 2); // the sample reaches the back of the input
    // no sample is longer than the truncation length
    EXPECT_EQ(sample.sample.size(), num_samples_in(sample.sample) * (max_length + 1));
}

TEST(CharSampling, ImbalancedCharsScaleTheSampleCount) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    size_t const num_partitions = 4, factor = 2;

    // every PE holds the same number of strings, but rank r holds (r + 1) times the characters
    size_t const num_strings = 50, str_len = (comm.rank() + 1) * 10;
    auto container = make_container(fixed_length_strings(num_strings, str_len, 'a'));
    auto const sample = CharBasedSampling<false, false>{factor}.sample_splitters(
        container.make_string_set(),
        num_partitions,
        NoExtraArg{},
        comm
    );

    size_t const local_chars = num_strings * str_len;
    auto const chars = dss_test::allgather_size(comm, local_chars);
    size_t const global_chars = std::accumulate(chars.begin(), chars.end(), size_t{0});
    size_t const total = get_total_num_samples(num_partitions, factor, comm.size());

    // character-based sampling scales with the characters held, not with the strings held
    EXPECT_EQ(num_samples_in(sample.sample), expected_num_samples(local_chars, global_chars, total));
}

TEST(CharSampling, EmptyPeDrawsNoSamples) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }

    bool const is_empty = comm.rank() == 0;
    auto container = make_container(
        is_empty ? std::vector<std::string>{} : fixed_length_strings(100, 10, 'a')
    );
    auto const sample = CharBasedSampling<false, false>{2}.sample_splitters(
        container.make_string_set(),
        4,
        NoExtraArg{},
        comm
    );

    EXPECT_EQ(num_samples_in(sample.sample) == 0, is_empty);
}

// ---------------------------------------------------------------------------------------------
// randomized sampling
// ---------------------------------------------------------------------------------------------

TEST(RandomSampling, DrawsLogarithmicallyManySamples) {
    Communicator comm;
    size_t const local_size = 1000, factor = 3;

    auto container = make_container(dss_test::random_strings(local_size, 5, 15, comm.rank()));
    auto const sample = StringBasedSampling<false, true>{factor}.sample_splitters(
        container.make_string_set(),
        /*num_partitions=*/4,
        NoExtraArg{},
        comm
    );

    // unlike the deterministic sampler the budget does not depend on the number of partitions,
    // it is factor * log2(P) per PE
    size_t const total = get_total_num_random_samples(factor, comm.size());
    EXPECT_EQ(
        num_samples_in(sample.sample),
        expected_num_samples(local_size, local_size * comm.size(), total)
    );
}

TEST(RandomSampling, ScalesWithLocalSize) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    size_t const factor = 4;
    size_t const local_size = (comm.rank() + 1) * 500;

    auto container = make_container(dss_test::random_strings(local_size, 5, 15, comm.rank()));
    auto const sample = StringBasedSampling<true, true>{factor}.sample_splitters(
        container.make_string_set(),
        /*num_partitions=*/4,
        NoExtraArg{},
        comm
    );

    auto const sizes = dss_test::allgather_size(comm, local_size);
    size_t const global_size = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
    size_t const total = get_total_num_random_samples(factor, comm.size());
    EXPECT_EQ(sample.indices.size(), expected_num_samples(local_size, global_size, total));

    // the samples are drawn from this PE's strings (with replacement, so they need not be sorted)
    for (auto const index: sample.indices) {
        EXPECT_GE(index, sample.local_offset);
        EXPECT_LT(index, sample.local_offset + local_size);
    }
}

TEST(RandomSampling, EmptyPeDrawsNoSamples) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }

    bool const is_empty = comm.rank() == 0;
    auto container =
        make_container(is_empty ? std::vector<std::string>{}
                                : dss_test::random_strings(500, 5, 15, comm.rank()));
    auto const sample = CharBasedSampling<false, true>{4}.sample_splitters(
        container.make_string_set(),
        4,
        NoExtraArg{},
        comm
    );

    EXPECT_EQ(num_samples_in(sample.sample) == 0, is_empty);
}

} // namespace
