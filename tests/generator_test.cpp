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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/mpi_ops.hpp>

#include "mpi/communicator.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "test_util.hpp"
#include "util/string_generator.hpp"

namespace {

using dss_mehnert::IdPlacement;
using dss_mehnert::SkewedDNArgs;
using dss_test::Char;
using dss_test::Communicator;

using StringSet = dss_mehnert::StringSet<Char, dss_mehnert::Length>;
using Generator = dss_mehnert::SkewedDNRatioLengthGenerator<StringSet>;

SkewedDNArgs default_args() {
    return {
        .global_strings = 2000,
        .min_length = 100,
        .max_length = 200,
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

TEST(SkewedDNGenerator, EveryStringIsGeneratedExactlyOnce) {
    Communicator comm;
    auto args = default_args();
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

// The property the whole design rests on: the id block sits at the same offset in every string of
// a group, and the region before it is a whole number of group blocks, so comparing two strings
// always compares group id against group id (or, within a group, id against id).
TEST(SkewedDNGenerator, LexicographicOrderIsIdOrder) {
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
TEST(SkewedDNGenerator, DistinguishingPrefixMatchesTheDNRatio) {
    Communicator comm;
    for (double const dn_ratio: {0.25, 0.5, 1.0}) {
        auto args = default_args();
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
            // whole blocks can lose at most w characters.
            double const requested = dn_ratio * static_cast<double>(str.length);
            EXPECT_LE(prefix, requested + 1.0) << "id " << str.id;
            EXPECT_GT(prefix, requested - w - 1.0) << "id " << str.id;

            local_dist_chars += static_cast<size_t>(prefix);
            local_chars += str.length;
        }

        double const realized = static_cast<double>(allreduce_sum(comm, local_dist_chars))
                                / static_cast<double>(allreduce_sum(comm, local_chars));
        EXPECT_NEAR(realized, dn_ratio, 0.05) << "for a requested D/N ratio of " << dn_ratio;
    }
}

// The strings of a prefix group are identical up to their id block and differ in its last
// character, so exactly D characters have to be inspected to tell them apart. This is what makes
// D/N mean what it says: nothing shorter than D distinguishes a string from its group partner.
TEST(SkewedDNGenerator, GroupMembersDifferOnlyInTheLastCharacterOfTheirPrefix) {
    Communicator comm;
    auto args = default_args();
    args.skew_fraction = 0.2;
    args.skew_factor = 4.0;

    auto const local = generate(args, comm);
    std::vector<std::string> local_chars;
    for (auto const& str: local) {
        local_chars.push_back(str.chars);
    }

    auto all_strings =
        dss_test::unpack(comm.allgatherv(kamping::send_buf(dss_test::pack(local_chars))));
    std::sort(all_strings.begin(), all_strings.end()); // == sorted by id
    ASSERT_EQ(all_strings.size(), args.global_strings);

    for (size_t id = 0; id + 1 < all_strings.size(); id += 2) {
        auto const& lhs = all_strings[id];
        auto const& rhs = all_strings[id + 1];
        auto const prefix =
            Generator::distinguishing_prefix(lhs.size(), args.global_strings, args.dn_ratio);

        ASSERT_EQ(lhs.size(), rhs.size()) << "the strings of a group share their length";
        EXPECT_TRUE(std::equal(lhs.begin(), lhs.begin() + (prefix - 1), rhs.begin()))
            << "the strings of group " << id / 2 << " differ before their id block";
        EXPECT_NE(lhs[prefix - 1], rhs[prefix - 1])
            << "the strings of group " << id / 2 << " do not differ at the end of their prefix";
    }
}

TEST(SkewedDNGenerator, SkewLengthensTheSmallestStrings) {
    Communicator comm;
    auto args = default_args();
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
TEST(SkewedDNGenerator, ContiguousPlacementMakesTheInputCharacterImbalanced) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }

    auto args = default_args();
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

TEST(SkewedDNGenerator, RandomPlacementKeepsTheInputBalanced) {
    Communicator comm;
    if (comm.size() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }

    auto args = default_args();
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
