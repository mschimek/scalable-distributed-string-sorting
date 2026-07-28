// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/bcast.hpp>
#include <kamping/named_parameters.hpp>
#include <spdlog/spdlog.h>
#include <tlx/die/core.hpp>
#include <tlx/math/div_ceil.hpp>
#include <tlx/vector_free.hpp>

#include "mpi/communicator.hpp"
#include "mpi/read_input.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_mehnert {

namespace _internal {

// The default base seed used when the caller does not override it via --seed.
inline constexpr size_t default_seed = 42;

// Broadcasts `seed` from the root PE so every PE uses the same base seed. The
// seed is supplied by the caller (see the --seed CLI option) rather than drawn
// from std::random_device, so a run is reproducible: the same seed always
// produces the same input.
inline size_t get_global_seed(size_t seed, Communicator const& comm) {
    comm.bcast_single(kamping::send_recv_buf(seed));
    return seed;
}

} // namespace _internal


template <typename StringSet>
class FileDistributer : public StringLcpContainer<StringSet> {
public:
    // max_size caps the number of bytes read from the file (0 = read the whole file). The cut is
    // made at a byte boundary, so the last line may be truncated.
    FileDistributer(std::string const& path, Communicator const& comm, size_t const max_size = 0)
        : StringLcpContainer<StringSet>{distribute_lines(path, max_size, comm)} {}

    static std::string getName() { return "FileDistributer"; }
};

template <typename StringSet>
class SuffixGenerator : public StringLcpContainer<StringSet> {
    using String = typename StringSet::String;

private:
    std::vector<unsigned char> readFile(std::string const& path) {
        RawStringsLines data;
        size_t const fileSize = get_file_size(path);
        std::ifstream in(path);
        std::vector<unsigned char>& rawStrings = data.rawStrings;
        rawStrings.reserve(1.5 * fileSize);

        std::string line;
        data.lines = 0u;
        while (std::getline(in, line)) {
            ++data.lines;
            for (unsigned char curChar: line)
                rawStrings.push_back(curChar);
        }
        rawStrings.push_back(0);
        in.close();
        return rawStrings;
    }

    auto distributeSuffixes(std::vector<unsigned char> const& text, Communicator const& comm) {
        size_t const textSize = text.size();
        size_t const estimatedTotalCharCount = textSize * (textSize + 1) / 2 + textSize;
        size_t const estimatedCharCount = estimatedTotalCharCount / comm.size();
        size_t const globalSeed = 0;
        std::mt19937 randGen(globalSeed);
        std::uniform_int_distribution<size_t> dist(0, comm.size() - 1);
        std::vector<unsigned char> rawStrings;
        rawStrings.reserve(estimatedCharCount);

        size_t numGenStrings = 0;
        for (size_t i = 0; i < textSize; ++i) {
            size_t PEIndex = dist(randGen);
            if (PEIndex == comm.rank()) {
                // only create your own strings
                ++numGenStrings;
                std::copy(text.begin() + i, text.end(), std::back_inserter(rawStrings));
                // Assume that text is zero terminated
            }
        }
        rawStrings.shrink_to_fit();
        return std::make_pair(std::move(rawStrings), numGenStrings);
    }

public:
    SuffixGenerator(std::string const& path, Communicator const& comm) {
        std::vector<unsigned char> text = readFile(path);
        auto [rawStrings, genStrings] = distributeSuffixes(text, comm);
        this->update(std::move(rawStrings));
        String* begin = this->strings();
        std::random_device rand;
        std::mt19937 gen{rand()};
        std::shuffle(begin, begin + genStrings, gen);
    }

    static std::string getName() { return "SuffixGenerator"; }
};

template <typename StringSet>
class DNRatioGenerator : public StringLcpContainer<StringSet> {
public:
    using CharType = StringSet::Char;

    DNRatioGenerator(
        size_t const global_strings,
        size_t const length,
        double const dn_ratio,
        Communicator const& comm,
        bool const encode_padding = false
    ) {
        std::mt19937_64 gen{comm.rank()};

        this->update(get_raw_strings(global_strings, length, dn_ratio, encode_padding, gen, comm));
        std::shuffle(this->get_strings().begin(), this->get_strings().end(), gen);
        this->make_contiguous();
    }

    static std::string getName() { return "DNRatioGenerator"; }

private:
    static constexpr CharType char_min = 'A', char_max = 'Z';
    static constexpr size_t char_range = char_max - char_min + 1;

    // Number of consecutive string ids that share the same leading blocks
    // when `encode_padding` is enabled: strings x with the same value of x /
    // padding_group get an identical distinguishing prefix except for the
    // final id block, so they collide during the bloom filter's duplicate
    // detection while their distinguishing prefix length stays the same.
    static constexpr size_t padding_group = 3;

    // Encode `value` in base-`char_range`, right-aligned, ending just before
    // `end` (least significant digit at `end - 1`). Positions that are not
    // written must already contain the padding character (`char_min`).
    template <typename It>
    static void encode_number(It end, size_t value) {
        for (; value != 0; value /= char_range) {
            *(--end) = char_min + (value % char_range);
        }
    }

    static std::vector<CharType> get_raw_strings(
        size_t const global_strings,
        size_t const req_length,
        double const dn_ratio,
        bool const encode_padding,
        std::mt19937_64& gen,
        Communicator const& comm
    ) {
        auto const local_strings = distribute_strings(global_strings, gen, comm);

        // number of characters needed to encode any string id in base char_range
        size_t const w =
            std::max<size_t>(1, std::ceil(std::log(global_strings) / std::log(char_range)));
        size_t const k = std::max<size_t>(req_length * std::clamp(dn_ratio, 0.0, 1.0), w);
        size_t const length = std::max(req_length, k);

        std::uniform_int_distribution<CharType> char_dist{char_min, char_max};
        CharType rand_char = char_dist(gen);
        comm.bcast_single(kamping::send_recv_buf(rand_char));

        size_t const raw_size = local_strings.size() * (length + 1);
        std::vector<CharType> raw_strings(raw_size);

        // reusable buffer for the w-char id / (x / padding_group) block
        std::vector<CharType> block(encode_padding ? w : 0);

        for (auto str_offset = raw_strings.begin(); size_t const x: local_strings) {
            std::fill_n(str_offset, k, char_min);

            if (encode_padding) {
                // Tile the leading [0, k - w) region with repeated w-char blocks
                // encoding x / padding_group, then place the unique id in the
                // final block [k - w, k). `padding_group` consecutive ids share
                // the leading blocks and only differ in the id block, so the
                // distinguishing prefix length stays k, while different groups
                // differ early and produce many distinct bloom filter hashes.
                std::fill(block.begin(), block.end(), char_min);
                encode_number(block.end(), x / padding_group);

                // fill [0, k) with the block tiled right-aligned to k
                for (auto e = str_offset + k; e != str_offset;) {
                    auto const n = std::min<size_t>(w, e - str_offset);
                    e -= n;
                    std::copy(block.end() - n, block.end(), e);
                }
                // overwrite the final block with the unique string id
                std::fill_n(str_offset + (k - w), w, char_min);
                encode_number(str_offset + k, x);

                std::fill_n(str_offset + k, length - k, rand_char);
            } else {
                // distinguishing prefix: encode the unique string id in [0, k)
                encode_number(str_offset + k, x);
                std::fill_n(str_offset + k, length - k, rand_char);
            }
            str_offset += length + 1;
        }
        return raw_strings;
    }

    static std::vector<size_t> distribute_strings(
        size_t const global_strings, std::mt19937_64& gen, Communicator const& comm
    ) {
        size_t const chunk_size = tlx::div_ceil(global_strings, comm.size());
        size_t const lower = std::min(global_strings, comm.rank() * chunk_size);
        size_t const upper = std::min(global_strings, lower + chunk_size);
        size_t const local_size = upper - lower;

        std::uniform_int_distribution<int> rank_dist{0, comm.size_signed() - 1};

        std::vector<int> dest(local_size), counts(comm.size()), offsets(comm.size());
        std::generate(dest.begin(), dest.end(), [&] { return rank_dist(gen); });
        std::for_each(dest.begin(), dest.end(), [&](auto const& n) { ++counts[n]; });
        std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), size_t{0});

        std::vector<size_t> strings(local_size);
        for (size_t i = lower; auto const& rank: dest) {
            strings[offsets[rank]++] = i++;
        }

        return comm.alltoallv(kamping::send_buf(strings), kamping::send_counts(counts));
    }
};

// Which PE a string id is generated on. Only meaningful together with a length skew, which makes
// the character mass a function of the key rank: with `random` placement the input stays balanced
// and the skew shows up in the buckets, with `contiguous` placement the low ranks hold the long
// strings and the input itself is imbalanced in characters.
enum class IdPlacement { random = 0, contiguous, sentinel };

struct SkewedDNArgs {
    size_t global_strings;
    size_t min_length;
    size_t max_length;
    bool use_uniform_prefix;
    // the fraction of a string that is its distinguishing prefix
    double dn_ratio = 0.5;
    // the fraction of the smallest prefix groups whose length is drawn from a longer interval
    double skew_fraction = 0.0;
    // the upper end of that interval, as a multiple of max_length
    double skew_factor = 1.0;
    IdPlacement placement = IdPlacement::random;
    // base seed for the RNG; the same seed reproduces the same input
    size_t seed = _internal::default_seed;
};

// A generator whose character mass is a function of the key rank.
//
// String `x` belongs to prefix group `g = x / 2` and looks like this:
//
//     [ enc(2g + 2, w) repeated (D - w) / w times ][ enc(x + 2, w) ][ fill character ... ]
//       shared with the other string of g            the id           no information
//
// where `w` is the number of base-sigma digits of the largest encoded id and `D = dn_ratio *
// length`, rounded so that the region before the id block is a whole number of `w`-character
// blocks. Both strings of a group draw the same length, so they are identical up to the id block,
// and `D` characters have to be inspected to tell them apart -- exactly the D/N ratio that was
// asked for, for every string. Ids are encoded with a `+2` offset (see `id_offset`) so that no
// string is all padding characters.
//
// The tiled block holds `2g + 2`, the encoding of the group's *first id*, rather than `g` itself.
// That puts it on the same scale as the id block, which is what lets a string carry no tiled region
// at all: a string whose D is only one block wide starts with `enc(x + 2)`, and comparing it
// against a longer string's leading `enc(2g + 2)` still orders the two by id. Encoding `g` would
// compare `x + 2` against `g'` -- two different scales -- and the order would depend on the lengths.
//
// So every string starts with a fixed-width encoding that is monotone in its id: the lexicographic
// order is the id order, whatever the lengths are. The skew therefore lengthens the
// lexicographically *smallest* strings, which is what makes the instance hard: the character mass
// sits at the bottom of the key range, where character-based sampling, character-based
// redistribution and the splitter length limit all have to cope with it.
//
// `D` can never be smaller than `w`, because `w` characters are what it takes to name one of
// `global_strings` strings -- a string shorter than `w / dn_ratio` therefore realizes a larger D/N
// ratio than was asked for, and `adjust_args` says so rather than lengthening it.
template <typename StringSet>
class SkewedDNRatioLengthGenerator : public StringLcpContainer<StringSet> {
public:
    using CharType = StringSet::Char;

    SkewedDNRatioLengthGenerator(SkewedDNArgs const& args, Communicator const& comm)
        : args_{adjust_args(args)} {
        size_t const seed = _internal::get_global_seed(args_.seed, comm);
        std::mt19937_64 gen{seed + comm.rank()};

        auto [ids, lengths] = generate_ids(args_, gen, comm.rank(), comm.size());
        if (args_.placement == IdPlacement::random) {
            std::tie(ids, lengths) = scatter_ids(ids, lengths, gen, comm);
        }

        this->update(get_raw_strings(args_, ids, lengths, get_fill_char(seed)));
        std::shuffle(this->get_strings().begin(), this->get_strings().end(), gen);
        this->make_contiguous();
    }

    static std::string getName() { return "SkewedDNRatioLengthGenerator"; }

    // the arguments the strings were actually generated with; the lengths may have been raised to
    // meet the requested D/N ratio (see adjust_args)
    SkewedDNArgs const& args() const { return args_; }

    // A string has to be able to hold its id block, so the lengths are raised to `w` if they are
    // below it. Nothing else is required: a string too short to spend `dn_ratio` of itself on `w`
    // characters simply carries a single block and realizes a larger ratio, which is reported
    // rather than corrected -- no layout can name one of `global_strings` strings in fewer than `w`
    // characters, so lengthening the strings to meet the request would silently replace the
    // instance that was asked for.
    static SkewedDNArgs adjust_args(SkewedDNArgs args) {
        tlx_die_unless(args.global_strings >= group_size);
        tlx_die_unless(0.0 < args.dn_ratio && args.dn_ratio <= 1.0);
        tlx_die_unless(0.0 <= args.skew_fraction && args.skew_fraction <= 1.0);
        tlx_die_unless(args.skew_factor >= 1.0);

        auto const min_length = min_admissible_length(args.global_strings, args.dn_ratio);
        if (args.min_length < min_length) {
            log_info(
                "min length {} cannot hold the {} characters it takes to name one of {} strings, "
                "raising it to {}",
                args.min_length,
                min_length,
                args.global_strings,
                min_length
            );
            args.min_length = min_length;
        }

        // the shortest length that realizes the requested ratio; below it D is pinned at w
        auto const w = id_width(args.global_strings);
        if (auto const exact = static_cast<size_t>(std::ceil(w / args.dn_ratio));
            args.min_length < exact) {
            log_info(
                "strings shorter than {} cannot realize a D/N ratio of {} with {} strings: their "
                "distinguishing prefix stays at {} characters, so a string of length {} realizes {}",
                exact,
                args.dn_ratio,
                args.global_strings,
                w,
                args.min_length,
                static_cast<double>(w) / static_cast<double>(args.min_length)
            );
        }
        if (args.max_length < args.min_length) {
            log_info(
                "max length {} is below the min length, raising it to {}",
                args.max_length,
                args.min_length
            );
            args.max_length = args.min_length;
        }
        return args;
    }

    // number of characters needed to encode any string id in base char_range. The ids are encoded
    // with an offset (see id_offset), so the widest value is (global_strings - 1) + id_offset.
    static size_t id_width(size_t const global_strings) {
        return num_digits((global_strings - 1) + id_offset);
    }

    // the characters that have to be inspected to tell a string of the given length apart from
    // every other string: the shared prefix region plus the id block
    static size_t
    distinguishing_prefix(size_t const length, size_t const global_strings, double const dn_ratio) {
        size_t const w = id_width(global_strings);
        // at least the id block: w characters are what it takes to name one of the strings
        size_t const k = std::clamp<size_t>(std::llround(dn_ratio * length), w, length);
        return w * ((k - w) / w) + w; // the region holds a whole number of blocks
    }

    // the id a generated string encodes; the inverse of the layout above
    static size_t decode_id(CharType const* string, size_t const length, SkewedDNArgs const& args) {
        size_t const w = id_width(args.global_strings);
        auto const prefix = distinguishing_prefix(length, args.global_strings, args.dn_ratio);
        return decode_number(string + prefix - w, string + prefix) - id_offset;
    }

    // The smallest admissible min_length: a string has to hold its id block. The requested D/N
    // ratio does not enter into it -- a shorter string is not inadmissible, it just realizes a
    // larger ratio than was asked for (see adjust_args).
    static size_t min_admissible_length(size_t const global_strings, double const) {
        return id_width(global_strings);
    }

    // The input a run with `num_pes` PEs produces, gathered into a single container on one PE: the
    // strings of PE 0, then those of PE 1, and so on, byte for byte as those PEs would hold them.
    // This is what makes a shared memory baseline comparable to a distributed run -- it sorts the
    // very same input, in the very same order, rather than another draw from the same distribution.
    static StringLcpContainer<StringSet>
    simulate(SkewedDNArgs const& requested_args, size_t const num_pes) {
        tlx_die_unless(num_pes > 0);
        auto const args = adjust_args(requested_args);

        // get_global_seed only broadcasts the root's seed, and every PE of a run is given the same
        // one on the command line
        size_t const seed = args.seed;

        // the engines are kept across both passes: a PE draws its destinations before it knows
        // what it receives, so its shuffle continues from the state pass one leaves behind
        std::vector<std::mt19937_64> gens;
        gens.reserve(num_pes);
        std::vector<std::vector<size_t>> recv_ids(num_pes), recv_lengths(num_pes);

        // Pass one: the PEs draw their ids and lengths, and hand each string to its destination.
        for (size_t rank = 0; rank != num_pes; ++rank) {
            auto& gen = gens.emplace_back(seed + rank);
            auto [ids, lengths] = generate_ids(args, gen, rank, num_pes);

            if (args.placement == IdPlacement::random) {
                auto const dest = draw_destinations(ids.size(), num_pes, gen);
                for (size_t i = 0; i != ids.size(); ++i) {
                    recv_ids[dest[i]].push_back(ids[i]);
                    recv_lengths[dest[i]].push_back(lengths[i]);
                }
            } else {
                recv_ids[rank] = std::move(ids);
                recv_lengths[rank] = std::move(lengths);
            }
        }

        // Pass two: the steps the constructor takes, PE by PE.
        std::vector<CharType> raw_strings;
        for (size_t rank = 0; rank != num_pes; ++rank) {
            StringLcpContainer<StringSet> local;
            local.update(
                get_raw_strings(args, recv_ids[rank], recv_lengths[rank], get_fill_char(seed))
            );
            tlx::vector_free(recv_ids[rank]);
            tlx::vector_free(recv_lengths[rank]);

            std::shuffle(local.get_strings().begin(), local.get_strings().end(), gens[rank]);
            local.make_contiguous();

            auto const& chars = local.raw_strings();
            raw_strings.insert(raw_strings.end(), chars.begin(), chars.end());
        }

        StringLcpContainer<StringSet> container;
        container.update(std::move(raw_strings));
        return container;
    }

private:
    static constexpr CharType char_min = 'A', char_max = 'Z';
    static constexpr size_t char_range = char_max - char_min + 1;

    // the number of strings that share a distinguishing prefix; two consecutive ids never carry
    // in the last digit of their encoding, so their distinguishing prefix is exactly D
    static constexpr size_t group_size = 2;

    // Ids are encoded with this offset added, so the smallest encoded id is id_offset, not 0. An
    // all-char_min string (id 0) would share an arbitrarily long prefix with the char_min padding
    // of longer strings, inflating its distinguishing prefix up to its full length; offsetting
    // past 0 removes that one degenerate string. The offset must be even so the first member of
    // every group (2 * group) stays even after the shift and its partner (+1) never carries out of
    // the last digit of its encoding.
    static constexpr size_t id_offset = 2;

    // on the root PE only; a no-op if the application has not set up the loggers
    template <typename... Args>
    static void log_info(fmt::format_string<Args...> fmt, Args&&... args) {
        if (auto const logger = spdlog::get("root")) {
            SPDLOG_LOGGER_INFO(logger, fmt, std::forward<Args>(args)...);
        }
    }

    // The character the region after the id block is padded with. It carries no information, but it
    // has to be the same on every PE, so it is derived from the seed alone -- nothing about the PE
    // may enter into it, which is exactly why a simulated run can reproduce it.
    static CharType get_fill_char(size_t const seed) {
        return static_cast<CharType>(char_min + seed % char_range);
    }

    // the number of base-char_range digits of value (at least one). An exact integer count -- a
    // floating-point logarithm would misround at exact powers of char_range.
    static size_t num_digits(size_t value) {
        size_t digits = 1;
        for (; value >= char_range; value /= char_range) {
            ++digits;
        }
        return digits;
    }

    // Encode `value` in base-`char_range`, right-aligned, ending just before `end`. Positions
    // that are not written must already contain the padding character (`char_min`).
    template <typename It>
    static void encode_number(It end, size_t value) {
        for (; value != 0; value /= char_range) {
            *(--end) = char_min + (value % char_range);
        }
    }

    template <typename It>
    static size_t decode_number(It begin, It end) {
        size_t value = 0;
        for (; begin != end; ++begin) {
            value = value * char_range + static_cast<size_t>(*begin - char_min);
        }
        return value;
    }

    // The ids PE `rank` generates, together with their lengths. The origin ranges are chunked by
    // prefix group rather than by string id, so a group is always drawn by a single PE and the
    // two strings of a group can not end up with different lengths. Takes the rank explicitly
    // rather than a communicator, so `simulate` can replay it for a PE that is not this one.
    static std::pair<std::vector<size_t>, std::vector<size_t>> generate_ids(
        SkewedDNArgs const& args, std::mt19937_64& gen, size_t const rank, size_t const num_pes
    ) {
        size_t const num_groups = tlx::div_ceil(args.global_strings, group_size);
        size_t const chunk = tlx::div_ceil(num_groups, num_pes);
        size_t const first_group = std::min(num_groups, rank * chunk);
        size_t const last_group = std::min(num_groups, first_group + chunk);
        size_t const num_skewed = static_cast<size_t>(args.skew_fraction * num_groups);

        std::uniform_int_distribution<size_t> length{args.min_length, args.max_length};
        std::uniform_int_distribution<size_t> skewed_length{
            args.min_length,
            std::max<size_t>(args.min_length, args.skew_factor * args.max_length)
        };

        std::vector<size_t> ids, lengths;
        ids.reserve(group_size * (last_group - first_group));
        lengths.reserve(ids.capacity());

        for (size_t group = first_group; group != last_group; ++group) {
            // one length per group: the strings of a group have to share their distinguishing
            // prefix, so they have to share their length
            size_t const len = group < num_skewed ? skewed_length(gen) : length(gen);

            for (size_t i = 0; i != group_size; ++i) {
                size_t const id = group_size * group + i;
                if (id < args.global_strings) {
                    ids.push_back(id);
                    lengths.push_back(len);
                }
            }
        }
        return {std::move(ids), std::move(lengths)};
    }

    // The PE each local string is sent to: one uniform draw per string, in local index order.
    // Separate from the exchange so that `simulate` can replay the draws -- and thereby advance the
    // engine exactly as the real run does -- without an alltoallv.
    static std::vector<int>
    draw_destinations(size_t const count, size_t const num_pes, std::mt19937_64& gen) {
        std::uniform_int_distribution<int> rank_dist{0, static_cast<int>(num_pes) - 1};

        std::vector<int> dest(count);
        std::generate(dest.begin(), dest.end(), [&] { return rank_dist(gen); });
        return dest;
    }

    // send each string to a uniformly random PE, carrying its length along with its id
    static std::pair<std::vector<size_t>, std::vector<size_t>> scatter_ids(
        std::vector<size_t> const& ids,
        std::vector<size_t> const& lengths,
        std::mt19937_64& gen,
        Communicator const& comm
    ) {
        auto const dest = draw_destinations(ids.size(), comm.size(), gen);

        std::vector<int> counts(comm.size());
        std::vector<int> offsets(comm.size());
        std::for_each(dest.begin(), dest.end(), [&](auto const& n) { ++counts[n]; });
        std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), size_t{0});

        std::vector<size_t> send_ids(ids.size()), send_lengths(lengths.size());
        for (size_t i = 0; i != dest.size(); ++i) {
            size_t const pos = offsets[dest[i]]++;
            send_ids[pos] = ids[i];
            send_lengths[pos] = lengths[i];
        }

        return {
            comm.alltoallv(kamping::send_buf(send_ids), kamping::send_counts(counts)),
            comm.alltoallv(kamping::send_buf(send_lengths), kamping::send_counts(counts))
        };
    }

    static std::vector<CharType> get_raw_strings(
        SkewedDNArgs const& args,
        std::vector<size_t> const& ids,
        std::vector<size_t> const& lengths,
        CharType const fill_char
    ) {
        size_t const w = id_width(args.global_strings);
        size_t const num_chars = std::accumulate(lengths.begin(), lengths.end(), size_t{0});

        // zero initialized, so the terminators are already in place
        std::vector<CharType> raw_strings(num_chars + lengths.size());
        std::vector<CharType> block(w);

        auto dest = raw_strings.begin();
        for (size_t i = 0; i != ids.size(); ++i) {
            size_t const id = ids[i], len = lengths[i];
            auto const prefix = distinguishing_prefix(len, args.global_strings, args.dn_ratio);

            if (args.use_uniform_prefix) {
                std::fill_n(dest, prefix, char_min);
            } else {
                // The region before the id block: the group's first id, tiled block by block. Not
                // the group index itself -- the block has to be on the same scale as the id block
                // so that a string with no tiled region at all still compares correctly against
                // one that has some (see the layout description above).
                std::fill(block.begin(), block.end(), char_min);
                encode_number(block.end(), group_size * (id / group_size) + id_offset);
                for (auto out = dest; out != dest + (prefix - w); out += w) {
                    std::copy(block.begin(), block.end(), out);
                }

                std::fill_n(dest + (prefix - w), w, char_min);
            }
            encode_number(dest + prefix, id + id_offset);
            std::fill_n(dest + prefix, len - prefix, fill_char);

            dest += len + 1;
        }
        return raw_strings;
    }

    SkewedDNArgs args_;
};

template <typename StringSet>
class RandomStringLcpContainer : public StringLcpContainer<StringSet> {
    using Char = typename StringSet::Char;

public:
    RandomStringLcpContainer(
        size_t const size, size_t const min_length = 10, size_t const max_length = 20
    ) {
        Communicator comm;
        std::vector<Char> random_raw_string_data;
        std::random_device rand_seed;
        std::mt19937 rand_gen(rand_seed());
        std::uniform_int_distribution<Char> char_dis(65, 90);

        size_t effectiveSize = size / comm.size();
        std::cout << "effective size: " << effectiveSize << std::endl;
        std::uniform_int_distribution<size_t> length_dis(min_length, max_length);
        random_raw_string_data.reserve(effectiveSize + 1);
        for (size_t i = 0; i < effectiveSize; ++i) {
            size_t length = length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j)
                random_raw_string_data.emplace_back(char_dis(rand_gen));
            random_raw_string_data.emplace_back(Char(0));
        }
        this->update(std::move(random_raw_string_data));
    }

    static std::string getName() { return "RandomStringGenerator"; }
};

template <typename StringSet>
class SkewedRandomStringLcpContainer : public StringLcpContainer<StringSet> {
    using Char = typename StringSet::Char;

public:
    SkewedRandomStringLcpContainer(
        size_t const size,
        size_t const min_length,
        size_t const max_length,
        Communicator const& comm
    ) {
        std::vector<Char> random_raw_string_data;
        std::mt19937 rand_gen(_internal::get_global_seed(_internal::default_seed, comm));
        std::uniform_int_distribution<Char> small_char_dis(65, 70);
        std::uniform_int_distribution<Char> char_dis(65, 90);

        std::uniform_int_distribution<size_t> dist(0, comm.size() - 1);
        std::uniform_int_distribution<size_t> normal_length_dis(min_length, max_length);
        std::uniform_int_distribution<size_t> large_length_dis(min_length + 100, max_length + 100);

        size_t const numLongStrings = size / 4;
        size_t const numSmallStrings = size - numLongStrings;
        std::size_t curChars = 0;

        random_raw_string_data.reserve(size + 1);
        for (size_t i = 0; i < numLongStrings; ++i) {
            size_t const PEIndex = dist(rand_gen);
            bool const takeValue = (PEIndex == comm.rank());
            size_t length = large_length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j) {
                unsigned char generatedChar = small_char_dis(rand_gen);
                if (takeValue) {
                    random_raw_string_data.push_back(generatedChar);
                }
            }
            if (takeValue) {
                random_raw_string_data.emplace_back(Char(0));
                curChars += length + 1;
            }
        }

        for (size_t i = 0; i < numSmallStrings; ++i) {
            size_t const PEIndex = dist(rand_gen);
            bool const takeValue = (PEIndex == comm.rank());
            size_t length = normal_length_dis(rand_gen);
            for (size_t j = 0; j < length; ++j) {
                unsigned char const generatedChar = char_dis(rand_gen);
                if (takeValue) {
                    random_raw_string_data.push_back(generatedChar);
                }
            }
            if (takeValue) {
                random_raw_string_data.push_back(Char(0));
                curChars += length + 1;
            }
        }
        random_raw_string_data.resize(curChars);
        this->update(std::move(random_raw_string_data));
    }

    static std::string getName() { return "SkewedStringGenerator"; }
};

template <typename StringSet>
class SkewedDNRatioGenerator : public StringLcpContainer<StringSet> {
    using String = typename StringSet::String;
    std::tuple<std::vector<unsigned char>, size_t, size_t>

    getRawStringsTimoStyle(
        size_t numStrings, size_t desiredStringLength, double dToN, Communicator const& comm
    ) {
        size_t const minInternChar = 65;
        size_t const maxInternChar = 90;

        size_t const numberInternChars = maxInternChar - minInternChar + 1;
        size_t const k = std::max(
            desiredStringLength * dToN,
            std::ceil(std::log(numStrings) / std::log(numberInternChars))
        );
        size_t const stringLength = std::max(desiredStringLength, k);
        std::vector<unsigned char> rawStrings;
        rawStrings.reserve(numStrings * (stringLength + 1) / comm.size());

        size_t const globalSeed = 0;
        std::mt19937 randGen(globalSeed);
        size_t const randomChar = minInternChar + (randGen() % numberInternChars);
        std::uniform_int_distribution<size_t> dist(0, comm.size() - 1);

        size_t numGenStrings = 0;
        size_t curOffset = 0;
        size_t const longStringMaxIndex = 0.2 * numStrings;
        size_t const longStringLength = stringLength * 3;
        for (size_t i = 0; i < numStrings; ++i) {
            size_t PEIndex = dist(randGen);
            if (PEIndex == comm.rank()) {
                // only create your own strings
                ++numGenStrings;
                size_t curIndex = i;
                for (size_t j = 0; j < k; ++j) {
                    rawStrings.push_back(minInternChar);
                }
                for (size_t j = 0; j < k; ++j) {
                    if (curIndex == 0)
                        break;
                    rawStrings[curOffset + k - 1 - j] =
                        minInternChar + (curIndex % numberInternChars);
                    curIndex /= numberInternChars;
                }
                for (size_t j = k; j < stringLength; ++j)
                    rawStrings.push_back(randomChar);
                if (i < longStringMaxIndex) {
                    for (size_t j = k; j < longStringLength + k; ++j)
                        rawStrings.push_back(randomChar);
                    curOffset += longStringLength;
                }
                rawStrings.push_back(0);
                curOffset += stringLength + 1;
            }
        }
        rawStrings.resize(curOffset);

        return make_tuple(rawStrings, numGenStrings, stringLength);
    }

public:
    SkewedDNRatioGenerator(
        size_t const size, size_t const stringLength, double const dToN, Communicator const& comm
    ) {
        size_t genStrings = 0;
        size_t genStringLength = 0;
        std::vector<unsigned char> rawStrings;
        std::tie(rawStrings, genStrings, genStringLength) =
            getRawStringsTimoStyle(size, stringLength, dToN, comm);
        this->update(std::move(rawStrings));
        String* begin = this->strings();
        std::random_device rand;
        std::mt19937 gen(rand());
        std::shuffle(begin, begin + genStrings, gen);
        this->make_contiguous();
    }

    static std::string getName() { return "SkewedDNRatioGenerator"; }
};

template <typename StringSet>
struct RandomCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    RandomCharGenerator(size_t const num_chars) : std::vector<Char>(num_chars) {
        std::random_device rand_seed;
        std::mt19937 gen(rand_seed());
        std::uniform_int_distribution<Char> dist{'A', 'Z'};
        std::generate(this->begin(), this->end(), [&] { return dist(gen); });
    }
};

template <typename StringSet>
struct FileCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    FileCharGenerator(std::string const& path, Communicator const& comm)
        : std::vector<Char>{distribute_file(path, 0, comm)} {
        std::replace(this->begin(), this->end(), static_cast<Char>(0), static_cast<Char>('A'));
    }
};

template <typename StringSet>
struct FileSegmentCharGenerator : public std::vector<typename StringSet::Char> {
    using Char = StringSet::Char;

    FileSegmentCharGenerator(
        std::string const& path, size_t const segment_size, Communicator const& comm
    )
        : std::vector<Char>{distribute_file_segments(path, segment_size, false, comm)} {
        std::replace(this->begin(), this->end(), static_cast<Char>(0), static_cast<Char>('A'));
    }
};

template <typename StringSet>
struct CompressedSuffixGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedSuffixGenerator(std::vector<Char>& chars, size_t const step = 1) {
        assert(step > 0);

        for (size_t offset = 0; offset < chars.size(); offset += step) {
            size_t const length = static_cast<size_t>(chars.size() - offset);
            this->emplace_back(chars.data() + offset, length);
        }
    }
};

template <typename StringSet>
struct CompressedWindowGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedWindowGenerator(std::vector<Char>& chars, size_t const length, size_t const step) {
        assert(length > 0 && step > 0);

        for (size_t offset = 0; offset + length <= chars.size(); offset += step) {
            this->emplace_back(chars.data() + offset, length);
        }
    }
};

template <typename StringSet>
struct CompressedDifferenceCoverGenerator : public std::vector<typename StringSet::String> {
    using Char = StringSet::Char;

    CompressedDifferenceCoverGenerator(
        std::vector<Char>& chars, size_t const size, bool const full_cover, Communicator const& comm
    ) {
        size_t const chars_size = chars.size();
        if (full_cover) {
            shift_chars_left(chars, size, comm);
        }

        auto const difference_cover = get_difference_cover(size);
        this->reserve((chars.size() / size + 1) * difference_cover.size());
        for (auto const& k: difference_cover) {
            for (size_t offset = k; offset < chars_size; offset += size) {
                auto const length = std::min<size_t>(size, chars.size() - offset);
                this->emplace_back(chars.data() + offset, length);
            }
        }
    }

private:
    static std::vector<size_t> get_difference_cover(size_t const size) {
        // clang-format off
        switch (size) {
            case 3:  { return {0, 1}; }
            case 7:  { return {1, 2, 4}; }
            case 13: { return {1, 2, 4, 10}; }
            case 21: { return {1, 2, 5, 15, 17}; }
            case 31: { return {1, 2, 4, 9, 13, 19}; }
            case 32: { return {1, 2, 3, 4, 8, 12, 20}; }
            case 64: { return {1, 2, 3, 6, 15, 17, 35, 43, 60}; }
            case 512: {
                return {0, 1, 2, 3, 4, 9, 18, 27, 36, 45, 64, 83, 102, 121, 140, 159, 178, 197, 216,
                    226, 236, 246, 256, 266, 267, 268, 269, 270 };
            }
            case 1024: {
                return {0, 1, 2, 3, 4, 5, 6, 13, 26, 39, 52, 65, 78, 91, 118, 145, 172, 199, 226,
                    253, 280, 307, 334, 361, 388, 415, 442, 456, 470, 484, 498, 512, 526, 540, 541,
                    542, 543, 544, 545, 546};
            }
            case 2048: {
                return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 38, 57, 76, 95, 114, 133, 152, 171, 190,
                    229, 268, 307, 346, 385, 424, 463, 502, 541, 580, 619, 658, 697, 736, 775, 814,
                    853, 892, 931, 951, 971, 991, 1011, 1031, 1051, 1071, 1091, 1111, 1131, 1132,
                    1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140 };
            }
            case 4096: {
                return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 27, 54, 81, 108, 135, 162,
                    189, 216, 243, 270, 297, 324, 351, 378, 433, 488, 543, 598, 653, 708, 763, 818,
                    873, 928, 983, 1038, 1093, 1148, 1203, 1258, 1313, 1368, 1423, 1478, 1533, 1588,
                    1643, 1698, 1753, 1808, 1863, 1891, 1919, 1947, 1975, 2003, 2031, 2059, 2087,
                    2115, 2143, 2171, 2199, 2227, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262,
                    2263, 2264, 2265, 2266, 2267, 2268};
            }
            case 8192: {
                return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 37, 74,
                    111, 148, 185, 222, 259, 296, 333, 370, 407, 444, 481, 518, 555, 592, 629, 666,
                    703, 778, 853, 928, 1003, 1078, 1153, 1228, 1303, 1378, 1453, 1528, 1603, 1678,
                    1753, 1828, 1903, 1978, 2053, 2128, 2203, 2278, 2353, 2428, 2503, 2578, 2653,
                    2728, 2803, 2878, 2953, 3028, 3103, 3178, 3253, 3328, 3403, 3478, 3516, 3554,
                    3592, 3630, 3668, 3706, 3744, 3782, 3820, 3858, 3896, 3934, 3972, 4010, 4048,
                    4086, 4124, 4162, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209,
                    4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218};
            }
            default: {
                tlx_die("no difference cover available for X=" << size);
            }
        }
        // clang-format on
    }

    static void
    shift_chars_left(std::vector<Char>& chars, size_t const size, Communicator const& comm) {
        using namespace kamping;

        // NOTE this only works correctly if every PE has at least `size` characters
        chars.reserve(chars.size() + size - 1);

        Request req;
        if (comm.rank() > 0) {
            std::span const send_chars{chars.begin(), std::min(size - 1, chars.size())};
            comm.issend(send_buf(send_chars), destination(comm.rank() - 1), request(req));
        }
        if (comm.rank() < comm.size() - 1) {
            std::vector<Char> recv_chars;
            comm.recv(recv_buf(recv_chars));
            chars.insert(chars.end(), recv_chars.begin(), recv_chars.end());
        }
        req.wait();
    }
};

template <typename StringSet>
struct CompressedDNRatioGenerator : public StringLcpContainer<StringSet> {
    using Char = StringSet::Char;

    CompressedDNRatioGenerator(
        size_t const local_strings,
        size_t const length,
        double const dn_ratio,
        Communicator const& comm
    ) {
        tlx_die_verbose_if(dn_ratio < 0.0, "negative D/N ratios are not supported");
        tlx_die_verbose_if(dn_ratio > 0.5, "D/N ratios greater than 1/2 are not supported");

        size_t const strings_per_chunk = std::max<size_t>(1, 2 * length * dn_ratio);
        size_t const chars_per_chunk = length + strings_per_chunk - 1;

        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<Char> char_dist{'A', 'Z'};

        Char padding_char = char_dist(gen);
        comm.bcast_single(kamping::send_recv_buf(padding_char));

        auto& raw_strings = *this->raw_strings_;
        auto& strings = this->strings_;

        size_t const num_chunks = tlx::div_ceil(local_strings, strings_per_chunk);
        raw_strings.reserve(num_chunks * chars_per_chunk);

        for (size_t n = 0; n < local_strings; n += strings_per_chunk) {
            raw_strings.resize(raw_strings.size() + chars_per_chunk, padding_char);
            std::generate_n(raw_strings.rbegin(), length, [&] { return char_dist(gen); });

            auto const chunk_size = std::min<size_t>(strings_per_chunk, local_strings - n);
            auto const begin = raw_strings.end() - chars_per_chunk;
            for (size_t i = 0; i < chunk_size; ++i) {
                strings.emplace_back(&*(begin + i), length);
            }
        }
        this->lcps_.resize(strings.size());
    }
};

} // namespace dss_mehnert
