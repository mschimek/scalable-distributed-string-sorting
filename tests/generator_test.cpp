// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Tests for SkewedDNRatioLengthGenerator (src/util/string_generator.hpp).
//
// The generator is only useful if it keeps the promises its layout is built on: the lexicographic
// order is the id order, every string's distinguishing prefix is exactly dn_ratio of its length,
// and the length skew lands on the lexicographically smallest strings. The tests below check each
// of those directly, because every downstream test that uses the generator as an oracle depends
// on them.
//
// Every test runs in both prefix modes (use_uniform_prefix off/on). The tiled per-group prefix and
// the uniform (constant-character) prefix share every property except one: only the tiled encoding
// makes the lexicographic order the id order across different lengths, so that one test is skipped
// for the uniform mode.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/mpi_ops.hpp>

#include "dss/mpi/communicator.hpp"
#include "dss/strings/stringcontainer.hpp"
#include "dss/strings/stringset.hpp"
#include "input/string_generator.hpp"
#include "test_util.hpp"

namespace {

using dss_mehnert::IdPlacement;
using dss_mehnert::SkewedDNArgs;
using dss_test::Char;
using dss_test::Communicator;

using StringSet = dss_mehnert::StringSet<Char, dss_mehnert::Length>;
using Generator = dss_mehnert::SkewedDNRatioLengthGenerator<StringSet>;

SkewedDNArgs default_args(bool const use_uniform_prefix = false) {
    return {
        .global_strings = 2000,
        .min_length = 100,
        .max_length = 200,
        .use_uniform_prefix = use_uniform_prefix,
        .dn_ratio = 0.5,
        .skew_fraction = 0.0,
        .skew_factor = 1.0,
        .placement = IdPlacement::random,
    };
}

struct LocalString {
    size_t id;
    size_t length;
    std::string chars;
};

// the generator may raise the lengths to support the requested D/N ratio, so decoding has to use
// the arguments it actually generated with
std::vector<LocalString> generate(SkewedDNArgs& args, Communicator const& comm) {
    Generator container{args, comm};
    args = container.args();
    auto const ss = container.make_string_set();

    std::vector<LocalString> strings;
    strings.reserve(ss.size());
    for (auto const& str: ss) {
        auto const length = ss.get_length(str);
        auto const chars = ss.get_chars(str, 0);
        strings.push_back(
            {Generator::decode_id(chars, length, args), length, std::string{chars, chars + length}}
        );
    }
    return strings;
}

size_t allreduce_sum(Communicator const& comm, size_t const value) {
    return comm.allreduce_single(kamping::send_buf(value), kamping::op(std::plus<>{}));
}

// GetParam() is use_uniform_prefix: false = tiled per-group prefix, true = constant-character
// prefix.
class SkewedDNGenerator : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(
    PrefixEncoding,
    SkewedDNGenerator,
    ::testing::Bool(),
    [](testing::TestParamInfo<bool> const& info) { return info.param ? "uniform" : "tiled"; }
);

TEST_P(SkewedDNGenerator, EveryStringIsGeneratedExactlyOnce) {
    Communicator comm;
    auto args = default_args(GetParam());
    auto const local = generate(args, comm);

    std::vector<size_t> ids;
    for (auto const& str: local) {
        ids.push_back(str.id);
    }
    auto all_ids = comm.allgatherv(kamping::send_buf(ids));
    std::sort(all_ids.begin(), all_ids.end());

    std::vector<size_t> expected(args.global_strings);
    std::iota(expected.begin(), expected.end(), size_t{0});
    EXPECT_EQ(all_ids, expected);
}

// The property the tiled encoding rests on: the id block sits at the same offset in every string of
// a group, and the region before it is a whole number of group blocks, so comparing two strings
// always compares group id against group id (or, within a group, id against id). The uniform prefix
// gives that up -- it pads with a constant character whose run length depends on the string's own
// length, so a short string and a longer one are no longer ordered by id -- hence the skip.
TEST_P(SkewedDNGenerator, LexicographicOrderIsIdOrder) {
    if (GetParam()) {
        GTEST_SKIP() << "uniform prefix does not preserve id order across different lengths";
    }

    Communicator comm;
    for (auto const placement: {IdPlacement::random, IdPlacement::contiguous}) {
        for (double const skew_fraction: {0.0, 0.1}) {
            auto args = default_args();
            args.placement = placement;
            args.skew_fraction = skew_fraction;
            args.skew_factor = 8.0;

            auto local = generate(args, comm);
            std::vector<std::string> chars;
            for (auto const& str: local) {
                chars.push_back(str.chars);
            }
            std::sort(chars.begin(), chars.end());

            auto const all_chars = comm.allgatherv(kamping::send_buf(dss_test::pack(chars)));
            auto sorted = dss_test::unpack(all_chars);
            std::sort(sorted.begin(), sorted.end());

            ASSERT_EQ(sorted.size(), args.global_strings);
            for (size_t i = 0; i != sorted.size(); ++i) {
                auto const* str = reinterpret_cast<Char const*>(sorted[i].data());
                EXPECT_EQ(Generator::decode_id(str, sorted[i].size(), args), i)
                    << "string at rank " << i << " is not the string with id " << i;
            }
        }
    }
}

// D characters have to be inspected to tell a string apart from every other, and D is dn_ratio of
// the string's own length -- including the skewed strings, whose distinguishing prefix grows with
// them.
TEST_P(SkewedDNGenerator, DistinguishingPrefixMatchesTheDNRatio) {
    Communicator comm;
    for (double const dn_ratio: {0.25, 0.5, 1.0}) {
        auto args = default_args(GetParam());
        args.dn_ratio = dn_ratio;
        args.skew_fraction = 0.2;
        args.skew_factor = 4.0;

        double const w = Generator::id_width(args.global_strings);
        auto const local = generate(args, comm);
        size_t local_dist_chars = 0, local_chars = 0;

        for (auto const& str: local) {
            auto const prefix = static_cast<double>(
                Generator::distinguishing_prefix(str.length, args.global_strings, dn_ratio)
            );
            // D is dn_ratio of the string's *own* length -- the skewed strings get longer
            // distinguishing prefixes, they are not simply padded. Rounding the region down to
            // whole blocks can lose at most w characters, and D can never drop below the w
            // characters it takes to name one of the strings.
            double const requested = dn_ratio * static_cast<double>(str.length);
            EXPECT_LE(prefix, std::max(w, requested + 1.0)) << "id " << str.id;
            EXPECT_GT(prefix, requested - w - 1.0) << "id " << str.id;

            local_dist_chars += static_cast<size_t>(prefix);
            local_chars += str.length;
        }

        double const realized = static_cast<double>(allreduce_sum(comm, local_dist_chars))
                                / static_cast<double>(allreduce_sum(comm, local_chars));
        EXPECT_NEAR(realized, dn_ratio, 0.05) << "for a requested D/N ratio of " << dn_ratio;
    }
}

// Naming one of `global_strings` strings takes w characters, so a string shorter than w / dn_ratio
// cannot realize the requested ratio -- no layout can. The generator pins D at w for those strings
// and reports the shortfall, rather than lengthening the strings and silently handing back an
// instance nobody asked for.
TEST_P(SkewedDNGenerator, ShortStringsKeepTheSmallestUsableDistinguishingPrefix) {
    Communicator comm;
    auto args = default_args(GetParam());
    args.min_length = 20;
    args.max_length = 400;
    args.dn_ratio = 0.05; // w / dn_ratio = 60, so the short strings are below the floor

    size_t const w = Generator::id_width(args.global_strings);
    auto const local = generate(args, comm);

    EXPECT_EQ(args.min_length, 20u) << "the lengths should not have been raised to meet the ratio";
    EXPECT_EQ(args.max_length, 400u);

    for (auto const& str: local) {
        auto const prefix =
            Generator::distinguishing_prefix(str.length, args.global_strings, args.dn_ratio);
        auto const requested = static_cast<size_t>(args.dn_ratio * static_cast<double>(str.length));

        EXPECT_GE(prefix, w) << "id " << str.id << " cannot be named in fewer than w characters";
        EXPECT_LE(prefix, std::max(w, requested + 1)) << "id " << str.id;
        if (requested < w) {
            EXPECT_EQ(prefix, w) << "id " << str.id << " should carry exactly the id block";
        }
    }
}

// The regime the id-scale tiled block exists for: strings whose whole distinguishing prefix is a
// single block, and so carry no tiled region at all, coexisting with strings that carry several.
// Tiling the group index instead of the group's first id puts the two on different scales, and the
// order comes out wrong -- but only when both kinds are present, which is why the test above, whose
// strings all carry a tiled region, does not catch it.
TEST_P(SkewedDNGenerator, LexicographicOrderIsIdOrderAcrossOneAndManyBlockPrefixes) {
    if (GetParam()) {
        GTEST_SKIP() << "uniform prefix does not preserve id order across different lengths";
    }

    Communicator comm;
    auto args = default_args();
    args.min_length = 20;
    args.max_length = 400;
    args.dn_ratio = 0.05;

    auto const prefix_of = [&args](size_t const length) {
        return Generator::distinguishing_prefix(length, args.global_strings, args.dn_ratio);
    };
    size_t const w = Generator::id_width(args.global_strings);
    ASSERT_EQ(prefix_of(args.min_length), w) << "the short strings should carry only the id block";
    ASSERT_GT(prefix_of(args.max_length), w) << "the long strings should carry a tiled region";

    auto const local = generate(args, comm);
    std::vector<std::string> chars;
    for (auto const& str: local) {
        chars.push_back(str.chars);
    }

    auto const all_chars = comm.allgatherv(kamping::send_buf(dss_test::pack(chars)));
    auto sorted = dss_test::unpack(all_chars);
    std::sort(sorted.begin(), sorted.end());

    ASSERT_EQ(sorted.size(), args.global_strings);
    for (size_t i = 0; i != sorted.size(); ++i) {
        auto const* str = reinterpret_cast<Char const*>(sorted[i].data());
        EXPECT_EQ(Generator::decode_id(str, sorted[i].size(), args), i)
            << "string at rank " << i << " is not the string with id " << i;
    }
}

// The strings of a prefix group are identical up to their id block and differ in its last
// character, so exactly D characters have to be inspected to tell them apart. This is what makes
// D/N mean what it says: nothing shorter than D distinguishes a string from its group partner. The
// strings are paired by decoded id rather than by lexicographic order, because the uniform prefix
// is not laid out in id order -- but the group property holds either way.
TEST_P(SkewedDNGenerator, GroupMembersDifferOnlyInTheLastCharacterOfTheirPrefix) {
    Communicator comm;
    auto args = default_args(GetParam());
    args.skew_fraction = 0.2;
    args.skew_factor = 4.0;

    auto const local = generate(args, comm);
    std::vector<size_t> local_ids;
    std::vector<std::string> local_chars;
    for (auto const& str: local) {
        local_ids.push_back(str.id);
        local_chars.push_back(str.chars);
    }

    auto const ids = comm.allgatherv(kamping::send_buf(local_ids));
    auto const strings =
        dss_test::unpack(comm.allgatherv(kamping::send_buf(dss_test::pack(local_chars))));
    ASSERT_EQ(ids.size(), args.global_strings);
    ASSERT_EQ(strings.size(), ids.size());

    std::vector<std::string> by_id(strings.size());
    for (size_t i = 0; i != ids.size(); ++i) {
        by_id[ids[i]] = strings[i];
    }

    for (size_t id = 0; id + 1 < by_id.size(); id += 2) {
        auto const& lhs = by_id[id];
        auto const& rhs = by_id[id + 1];
        auto const prefix =
            Generator::distinguishing_prefix(lhs.size(), args.global_strings, args.dn_ratio);

        ASSERT_EQ(lhs.size(), rhs.size()) << "the strings of a group share their length";
        EXPECT_TRUE(std::equal(lhs.begin(), lhs.begin() + (prefix - 1), rhs.begin()))
            << "the strings of group " << id / 2 << " differ before their id block";
        EXPECT_NE(lhs[prefix - 1], rhs[prefix - 1])
            << "the strings of group " << id / 2 << " do not differ at the end of their prefix";
    }
}

TEST_P(SkewedDNGenerator, SkewLengthensTheSmallestStrings) {
    Communicator comm;
    auto args = default_args(GetParam());
    args.skew_fraction = 0.1;
    args.skew_factor = 10.0;

    auto const local = generate(args, comm);
    size_t const num_skewed = static_cast<size_t>(args.skew_fraction * args.global_strings);

    size_t local_skewed_chars = 0, local_rest_chars = 0;
    for (auto const& str: local) {
        (str.id < num_skewed ? local_skewed_chars : local_rest_chars) += str.length;
    }
    size_t const skewed_chars = allreduce_sum(comm, local_skewed_chars);
    size_t const rest_chars = allreduce_sum(comm, local_rest_chars);

    // 10% of the strings (the smallest ids) hold a large share of all characters
    EXPECT_GT(skewed_chars, rest_chars / 4)
        << "the skewed strings do not carry a disproportionate share of the characters";
}

// The two placements are the same instance, distributed differently: contiguous placement puts the
// long strings on the low ranks, so the input itself is imbalanced in characters, while the string
// counts stay balanced. That is the input the imbalance-aware (omega) sampling has to cope with.
TEST_P(SkewedDNGenerator, ContiguousPlacementMakesTheInputCharacterImbalanced) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }

    auto args = default_args(GetParam());
    args.placement = IdPlacement::contiguous;
    args.skew_fraction = 1.0 / comm.size(); // the strings of the first rank
    args.skew_factor = 10.0;

    auto const local = generate(args, comm);
    size_t local_chars = 0;
    for (auto const& str: local) {
        local_chars += str.length;
    }

    auto const chars = dss_test::allgather_size(comm, local_chars);
    auto const counts = dss_test::allgather_size(comm, local.size());
    auto const max_count = *std::max_element(counts.begin(), counts.end());
    auto const min_count = *std::min_element(counts.begin(), counts.end());

    // the ids are handed out in whole prefix groups, so the counts differ by at most a group or
    // two -- the imbalance is in the characters, not in the strings
    EXPECT_LT(max_count, min_count + min_count / 10) << "the string counts should stay balanced";
    EXPECT_GT(chars.front(), 2 * chars.back()) << "rank 0 should hold the long strings";
}

// The claim `simulate` makes: on a single PE it produces the exact bytes the PEs of a real run
// hold, concatenated in rank order. Byte equality is the strongest check available here -- it
// covers the length every prefix group drew, which PE each string was scattered to, and the order
// the strings ended up in within a PE. Anything weaker would also pass on an input that is merely
// drawn from the same distribution, which is the thing the simulation exists to rule out.
TEST_P(SkewedDNGenerator, SimulationReproducesTheDistributedInputByteForByte) {
    Communicator comm;
    for (auto const placement: {IdPlacement::random, IdPlacement::contiguous}) {
        // 1001 leaves the last prefix group with a single string (the id < global_strings guard),
        // 4 leaves the high ranks with nothing to generate at all
        for (size_t const global_strings: {size_t{2000}, size_t{1001}, size_t{4}}) {
            auto args = default_args(GetParam());
            args.global_strings = global_strings;
            args.placement = placement;
            args.skew_fraction = 0.1;
            args.skew_factor = 8.0;

            Generator distributed{args, comm};
            auto const gathered = comm.allgatherv(kamping::send_buf(distributed.raw_strings()));

            auto simulated = Generator::simulate(args, comm.size());
            auto const& raw = simulated.raw_strings();

            ASSERT_EQ(raw.size(), gathered.size())
                << "simulating " << comm.size() << " PEs, " << global_strings << " strings";
            EXPECT_TRUE(raw == gathered)
                << "the simulated input differs from the distributed one, simulating "
                << comm.size() << " PEs, " << global_strings << " strings";
        }
    }
}

// The actual use case is a single PE standing in for a run far wider than itself, so the simulated
// PE count is unrelated to the number of ranks the simulation runs on. The instance still has to be
// the whole instance: every id exactly once.
TEST_P(SkewedDNGenerator, SimulationForAWiderRunIsAWellFormedInstance) {
    auto const args = Generator::adjust_args([&] {
        auto args = default_args(GetParam());
        args.skew_fraction = 0.1;
        args.skew_factor = 8.0;
        return args;
    }());

    auto simulated = Generator::simulate(args, 16);
    auto const ss = simulated.make_string_set();

    std::vector<size_t> ids;
    ids.reserve(ss.size());
    for (auto const& str: ss) {
        ids.push_back(Generator::decode_id(ss.get_chars(str, 0), ss.get_length(str), args));
    }
    std::sort(ids.begin(), ids.end());

    std::vector<size_t> expected(args.global_strings);
    std::iota(expected.begin(), expected.end(), size_t{0});
    EXPECT_EQ(ids, expected);
}

TEST_P(SkewedDNGenerator, RandomPlacementKeepsTheInputBalanced) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }

    auto args = default_args(GetParam());
    args.global_strings = 20000;
    args.placement = IdPlacement::random;
    args.skew_fraction = 0.1;
    args.skew_factor = 10.0;

    auto const local = generate(args, comm);
    size_t local_chars = 0;
    for (auto const& str: local) {
        local_chars += str.length;
    }

    auto const chars = dss_test::allgather_size(comm, local_chars);
    auto const max = *std::max_element(chars.begin(), chars.end());
    auto const min = *std::min_element(chars.begin(), chars.end());

    // the long strings are spread over all PEs, so the skew shows up in the buckets, not in the
    // input
    EXPECT_LT(max, 2 * min);
}

} // namespace
