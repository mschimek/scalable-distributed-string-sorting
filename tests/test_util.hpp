// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Shared helpers for the MPI tests: building packed string inputs, gathering distributed
// results and the random instances the sorters are exercised on.

#pragma once

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <kamping/collectives/allgather.hpp>
#include <kamping/named_parameters.hpp>

#include "dss/mpi/communicator.hpp"
#include "dss/util/measuringTool.hpp"

namespace dss_test {

using Char = unsigned char;
using Communicator = dss_mehnert::Communicator;

// The MeasuringTool is a singleton that accumulates its records; adding the same key twice is a
// fatal error. The executables reset it once per iteration, the tests do so before every sort.
inline void reset_measurements() {
    dss_mehnert::measurement::MeasuringTool::measuringTool().reset();
}

// The sorters take and return strings packed into a single buffer of '\0'-terminated strings.
inline std::vector<Char> pack(std::vector<std::string> const& strings) {
    std::vector<Char> raw;
    for (auto const& str: strings) {
        raw.insert(raw.end(), str.begin(), str.end());
        raw.push_back(Char{0});
    }
    return raw;
}

inline std::vector<std::string> unpack(std::vector<Char> const& raw) {
    std::vector<std::string> strings;
    std::string current;
    for (auto const c: raw) {
        if (c == 0) {
            strings.push_back(current);
            current.clear();
        } else {
            current.push_back(static_cast<char>(c));
        }
    }
    return strings;
}

// The strings of all PEs in rank order; the concatenation is what "globally sorted" refers to.
inline std::vector<std::string>
gather_in_rank_order(Communicator const& comm, std::vector<Char> const& raw) {
    return unpack(comm.allgatherv(kamping::send_buf(raw)));
}

inline std::vector<size_t> allgather_size(Communicator const& comm, size_t const value) {
    return comm.allgather(kamping::send_buf(value));
}

// Distinct-ish strings over a large alphabet;
inline std::vector<std::string>
random_strings(size_t const count, size_t const min_len, size_t const max_len, size_t const seed) {
    std::mt19937_64 gen{seed};
    std::uniform_int_distribution<size_t> len_dist{min_len, max_len};
    std::uniform_int_distribution<int> char_dist{'a', 'z'};

    std::vector<std::string> strings;
    strings.reserve(count);
    for (size_t i = 0; i != count; ++i) {
        std::string str(len_dist(gen), '\0');
        std::generate(str.begin(), str.end(), [&] { return static_cast<char>(char_dist(gen)); });
        strings.push_back(std::move(str));
    }
    return strings;
}

// A two letter alphabet and short strings, so the input holds many duplicates.
inline std::vector<std::string> duplicate_heavy_strings(size_t const count, size_t const seed) {
    std::mt19937_64 gen{seed};
    std::uniform_int_distribution<int> char_dist{'a', 'b'};

    std::vector<std::string> strings;
    strings.reserve(count);
    for (size_t i = 0; i != count; ++i) {
        std::string str(4, '\0');
        std::generate(str.begin(), str.end(), [&] { return static_cast<char>(char_dist(gen)); });
        strings.push_back(std::move(str));
    }
    return strings;
}

// Many short strings and a few very long ones
inline std::vector<std::string> length_skewed_strings(size_t const count, size_t const seed) {
    std::mt19937_64 gen{seed};
    std::uniform_int_distribution<size_t> short_len{2, 8};
    std::uniform_int_distribution<size_t> long_len{200, 1000};
    std::uniform_int_distribution<int> char_dist{'a', 'z'};
    std::bernoulli_distribution is_long{0.05};

    std::vector<std::string> strings;
    strings.reserve(count);
    for (size_t i = 0; i != count; ++i) {
        std::string str(is_long(gen) ? long_len(gen) : short_len(gen), '\0');
        std::generate(str.begin(), str.end(), [&] { return static_cast<char>(char_dist(gen)); });
        strings.push_back(std::move(str));
    }
    return strings;
}

// A long common prefix on every string; all the work happens beyond the LCP.
inline std::vector<std::string> common_prefix_strings(size_t const count, size_t const seed) {
    std::string const prefix(64, 'x');
    auto suffixes = random_strings(count, 1, 6, seed);
    for (auto& str: suffixes) {
        str.insert(0, prefix);
    }
    return suffixes;
}

} // namespace dss_test
