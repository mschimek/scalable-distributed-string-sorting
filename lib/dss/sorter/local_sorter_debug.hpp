// (c) 2025 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// TEMPORARY DEBUGGING AID -- remove once the local sorters are known to agree.
//
// Runs both local sorters on the same input and compares the resulting order and LCP values.
// Enabled at run time (no rebuild) by setting the environment variable:
//
//     DSS_COMPARE_LOCAL_SORTERS=1 mpiexec -n 4 ./build/distributed_sorter ...
//
// The sorters only permute the string array and fill the LCP array; the character buffer is
// read-only. Snapshotting the string and LCP arrays therefore gives both sorters the exact
// same input, and restoring the snapshot afterwards leaves the container untouched for the
// real (measured) sort that follows.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

#include <tlx/sort/strings/multikey_quicksort.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

namespace dss_mehnert {

inline bool debug_compare_local_sorters_enabled() {
    static bool const enabled = std::getenv("DSS_COMPARE_LOCAL_SORTERS") != nullptr;
    return enabled;
}

template <typename StringLcpContainer, typename Communicator>
void debug_compare_local_sorters(
    StringLcpContainer& container,
    Communicator const& comm,
    std::string const& where,
    size_t const depth = 0,
    size_t const memory = 0
) {
    if (!debug_compare_local_sorters_enabled()) {
        return;
    }

    auto const input_strings = container.get_strings();
    auto const input_lcps = container.lcps();

    tlx::sort_strings_detail::radixsort_CI3(container.make_string_lcp_ptr(), depth, memory);
    auto const radix_strings = container.get_strings();
    auto const radix_lcps = container.lcps();

    container.get_strings() = input_strings;
    container.lcps() = input_lcps;
    tlx::sort_strings_detail::multikey_quicksort(container.make_string_lcp_ptr(), depth, memory);
    auto const mkqs_strings = container.get_strings();
    auto const mkqs_lcps = container.lcps();

    

    auto ss = container.make_string_set();
    auto const same_string = [&ss](auto const& lhs, auto const& rhs) {
        auto const len = ss.get_length(lhs);
        auto const chars = ss.get_chars(lhs, 0);
        return len == ss.get_length(rhs)
               && std::equal(chars, chars + len, ss.get_chars(rhs, 0));
    };

    size_t const n = container.size();
    size_t order_mismatches = 0, lcp_mismatches = 0;
    std::string first_diff;

    for (size_t i = 0; i != n; ++i) {
        bool const order_differs = !same_string(radix_strings[i], mkqs_strings[i]);
        // lcp[0] is not written by either sorter
        bool const lcp_differs = i > 0 && radix_lcps[i] != mkqs_lcps[i];

        order_mismatches += order_differs;
        lcp_mismatches += lcp_differs;

        if ((order_differs || lcp_differs) && first_diff.empty()) {
            first_diff = " first at index " + std::to_string(i) + ": lcp CI3="
                         + std::to_string(radix_lcps[i]) + " mkqs=" + std::to_string(mkqs_lcps[i]);
        }
    }

    if (order_mismatches != 0 || lcp_mismatches != 0) {
        std::cerr << "COMPARE_LOCAL_SORTERS rank=" << comm.rank() << " " << where << " n=" << n
                  << " MISMATCH order=" << order_mismatches << " lcp=" << lcp_mismatches
                  << first_diff << std::endl;
    } else {
        std::cerr << "COMPARE_LOCAL_SORTERS rank=" << comm.rank() << " " << where << " n=" << n
                  << " ok (order and lcps identical)" << std::endl;
    }

    // hand the untouched input back to the real sort
    container.get_strings() = input_strings;
    container.lcps() = input_lcps;
}

} // namespace dss_mehnert
