// (c) 2025 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstddef>

#include <tlx/sort/strings/multikey_quicksort.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

namespace dss_mehnert {

// sequential sorter used for the base case and for sorting the splitter sample
enum class LocalSorter { radixsort_CI3 = 0, multikey_quicksort, sentinel };

// both sorters fill the LCP array of a StringLcpPtr and use memory as a recursion budget
template <typename StringPtr>
void sort_strings_locally(
    StringPtr const& strptr,
    LocalSorter const sorter,
    size_t const depth = 0,
    size_t const memory = 0
) {
    switch (sorter) {
        case LocalSorter::multikey_quicksort:
            tlx::sort_strings_detail::multikey_quicksort(strptr, depth, memory);
            return;
        default:
            tlx::sort_strings_detail::radixsort_CI3(strptr, depth, memory);
            return;
    }
}

} // namespace dss_mehnert
