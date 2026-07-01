// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstddef>
#include <iostream>
#include <ostream>
#include <vector>

#include <kamping/collectives/gather.hpp>
#include <kamping/named_parameters.hpp>

#include "mpi/communicator.hpp"

namespace dss_mehnert {

// Debug utility: gather the (raw) strings of every PE on the root PE and print
// them, inserting a marker line before the strings originating from each PE.
template <typename Container>
void gather_and_print_strings(
    Container& container, dss_mehnert::Communicator const& comm, std::ostream& out = std::cout
) {
    using namespace kamping;
    using Char = typename Container::Char;

    // ensure the local strings are densely packed in the raw buffer
    container.make_contiguous();

    std::vector<Char> global_chars;
    auto result = comm.gatherv(
        send_buf(container.raw_strings()),
        recv_buf<BufferResizePolicy::resize_to_fit>(global_chars),
        recv_counts_out()
    );

    if (!comm.is_root()) {
        return;
    }

    // per-PE character counts let us find where each PE's strings begin
    auto const char_counts = result.extract_recv_counts();

    auto char_it = global_chars.begin();
    for (int rank = 0; rank != comm.size_signed(); ++rank) {
        out << "===== PE " << rank << " =====\n";

        auto const pe_end = char_it + char_counts[rank];
        while (char_it != pe_end) {
            auto const str_begin = char_it;
            while (char_it != pe_end && *char_it != 0) {
                ++char_it;
            }

            out.write(
                reinterpret_cast<char const*>(&*str_begin),
                static_cast<std::streamsize>((char_it - str_begin) * sizeof(Char))
            );
            out << '\n';

            if (char_it != pe_end) {
                ++char_it; // skip the null terminator
            }
        }
    }
    out.flush();
}

} // namespace dss_mehnert
