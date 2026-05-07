// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
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

}  // namespace dss::examples
