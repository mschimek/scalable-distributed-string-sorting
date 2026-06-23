// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <kamping/collectives/gather.hpp>
#include <kamping/named_parameters.hpp>

#include "mpi/communicator.hpp"

namespace dss::examples {

struct InputConfig {
    std::size_t strings_per_rank = 1000;
    std::size_t min_len          = 3;
    std::size_t max_len          = 50;
    std::uint32_t seed           = 0xC0FFEE;
};

inline std::vector<unsigned char> make_local_input(int rank, InputConfig const& cfg) {
    std::mt19937 rng{cfg.seed ^ static_cast<std::uint32_t>(rank)};
    std::uniform_int_distribution<std::size_t> len_dist{cfg.min_len, cfg.max_len};
    std::uniform_int_distribution<int> char_dist{'a', 'z'};

    std::vector<unsigned char> bytes;
    for (std::size_t i = 0; i < cfg.strings_per_rank; ++i) {
        std::size_t const len = len_dist(rng);
        for (std::size_t j = 0; j < len; ++j) {
            bytes.push_back(static_cast<unsigned char>(char_dist(rng)));
        }
        bytes.push_back(0);
    }
    return bytes;
}

inline std::vector<unsigned char> make_all_a_local_input(int rank, InputConfig const& cfg) {
    std::mt19937 rng{cfg.seed ^ static_cast<std::uint32_t>(rank)};
    std::uniform_int_distribution<std::size_t> len_dist{cfg.min_len, cfg.max_len};
    std::uniform_int_distribution<int> char_dist{'a', 'z'};

    std::vector<unsigned char> bytes;
    for (std::size_t i = 0; i < cfg.strings_per_rank; ++i) {
        std::size_t const len = len_dist(rng);
        for (std::size_t j = 0; j < len; ++j) {
            bytes.push_back(static_cast<unsigned char>('a'));
        }
        bytes.push_back(0);
    }
    return bytes;
}

// Assigns each local string a globally unique 64-bit index. With one block of
// `strings_per_rank` consecutive indices per rank, the (string, index) order is
// total, which makes the indexed result deterministic and easy to verify.
inline std::vector<std::uint64_t> make_local_indices(int rank, InputConfig const& cfg) {
    std::vector<std::uint64_t> indices(cfg.strings_per_rank);
    std::uint64_t const base = static_cast<std::uint64_t>(rank) * cfg.strings_per_rank;
    std::iota(indices.begin(), indices.end(), base);
    return indices;
}

inline std::vector<std::string> split_nul(std::vector<unsigned char> const& bytes) {
    std::vector<std::string> out;
    auto begin = bytes.begin();
    for (auto it = bytes.begin(); it != bytes.end(); ++it) {
        if (*it == 0) {
            out.emplace_back(begin, it);
            begin = it + 1;
        }
    }
    return out;
}

// Pairs each NUL-terminated string in `bytes` with its index, formatted as
// "string@index". Strings without a matching index are paired with "?".
inline std::vector<std::string>
zip_nul(std::vector<unsigned char> const& bytes, std::vector<std::uint64_t> const& indices) {
    auto strings = split_nul(bytes);
    std::vector<std::string> out;
    out.reserve(strings.size());
    for (std::size_t i = 0; i < strings.size(); ++i) {
        if (i < indices.size()) {
            out.emplace_back(strings[i] + '@' + std::to_string(indices[i]));
        } else {
            out.emplace_back(strings[i] + "@?");
        }
    }
    return out;
}

// Gathers `sorted_local` and `input_copy` to rank 0, sorts the gathered input
// via std::sort, and reports OK/FAIL. Returns an exit code suitable for `main`.
inline int verify_and_report(
    dss_mehnert::Communicator const& comm,
    std::vector<unsigned char> const& sorted_local,
    std::vector<unsigned char> const& input_copy,
    std::string_view label)
{
    auto gathered_sorted = comm.gatherv(kamping::send_buf(sorted_local));
    auto gathered_input  = comm.gatherv(kamping::send_buf(input_copy));

    if (comm.rank() != 0) {
        return EXIT_SUCCESS;
    }

    auto sorted_strings   = split_nul(gathered_sorted);
    auto expected_strings = split_nul(gathered_input);
    std::sort(expected_strings.begin(), expected_strings.end());

    bool const ok = sorted_strings == expected_strings;
    std::cout << '[' << label << "] num_procs=" << comm.size()
              << " total_strings=" << expected_strings.size()
              << " result=" << (ok ? "OK" : "FAIL") << '\n';

    if (!ok) {
        std::size_t const limit = std::min<std::size_t>(8, expected_strings.size());
        for (std::size_t i = 0; i < limit; ++i) {
            std::string const& got =
                i < sorted_strings.size() ? sorted_strings[i] : std::string{"<missing>"};
            std::cout << "  [" << i << "] expected=\"" << expected_strings[i]
                      << "\" got=\"" << got << "\"\n";
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// Indexed counterpart of `verify_and_report`: checks that the gathered result is
// the input globally sorted by (string, index), with indices kept in sync with
// their strings. Returns an exit code suitable for `main`.
inline int verify_and_report_indexed(
    dss_mehnert::Communicator const& comm,
    std::vector<unsigned char> const& sorted_chars,
    std::vector<std::uint64_t> const& sorted_indices,
    std::vector<unsigned char> const& input_chars,
    std::vector<std::uint64_t> const& input_indices,
    std::string_view label)
{
    auto gathered_sorted_chars   = comm.gatherv(kamping::send_buf(sorted_chars));
    auto gathered_sorted_indices = comm.gatherv(kamping::send_buf(sorted_indices));
    auto gathered_input_chars    = comm.gatherv(kamping::send_buf(input_chars));
    auto gathered_input_indices  = comm.gatherv(kamping::send_buf(input_indices));

    if (comm.rank() != 0) {
        return EXIT_SUCCESS;
    }

    using Entry = std::pair<std::string, std::uint64_t>;
    auto zip = [](std::vector<std::string> const& strings,
                  std::vector<std::uint64_t> const& indices) {
        std::vector<Entry> entries;
        entries.reserve(strings.size());
        for (std::size_t i = 0; i < strings.size(); ++i) {
            entries.emplace_back(strings[i], i < indices.size() ? indices[i] : 0);
        }
        return entries;
    };

    auto got      = zip(split_nul(gathered_sorted_chars), gathered_sorted_indices);
    auto expected = zip(split_nul(gathered_input_chars), gathered_input_indices);
    std::sort(expected.begin(), expected.end());

    bool const ok = got.size() == expected.size()
                    && gathered_sorted_indices.size() == got.size() && got == expected;
    std::cout << '[' << label << "] num_procs=" << comm.size()
              << " total_strings=" << expected.size()
              << " result=" << (ok ? "OK" : "FAIL") << '\n';

    if (!ok) {
        std::size_t const limit = std::min<std::size_t>(8, expected.size());
        for (std::size_t i = 0; i < limit; ++i) {
            Entry const& g = i < got.size() ? got[i] : Entry{"<missing>", 0};
            std::cout << "  [" << i << "] expected=(\"" << expected[i].first << "\","
                      << expected[i].second << ") got=(\"" << g.first << "\"," << g.second
                      << ")\n";
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

}  // namespace dss::examples
