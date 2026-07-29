// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Names for the enum-valued command line options. The name table of an enum is its single source
// of truth: CLI11 parses against it and lists the accepted values in --help, and the JSON report
// writes the name back through it.

#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "mpi/alltoallv/params.hpp"
#include "sorter/local_sorter.hpp"
#include "util/string_generator.hpp"

// a vector rather than a map, so that --help lists the values in the order they are declared
template <typename Enum>
using EnumNames = std::vector<std::pair<std::string, Enum>>;

template <typename Enum>
std::string enum_name(EnumNames<Enum> const& names, Enum const value) {
    auto const it = std::find_if(names.begin(), names.end(), [&](auto const& name) {
        return name.second == value;
    });
    return it != names.end() ? it->first : "invalid";
}

// the accepted values, as CLI11 should show them instead of its own "{name->0,...} OR {0,...}"
template <typename Enum>
std::string enum_value_list(EnumNames<Enum> const& names) {
    std::string list;
    for (auto const& [name, value]: names) {
        list += (list.empty() ? "{" : ",") + name;
    }
    return list + "}";
}

namespace dss_mehnert {

inline EnumNames<LocalSorter> const local_sorter_names{
    {"radixsort-ci3", LocalSorter::radixsort_CI3},
    {"multikey-quicksort", LocalSorter::multikey_quicksort},
};

template <typename Json>
void to_json(Json& json, LocalSorter const value) {
    json = enum_name(local_sorter_names, value);
}

namespace mpi {

inline EnumNames<AlltoallvAlgorithm> const alltoall_names{
    {"native", AlltoallvAlgorithm::native},
    {"direct", AlltoallvAlgorithm::direct},
    {"onefactor", AlltoallvAlgorithm::onefactor},
    {"pairwise", AlltoallvAlgorithm::pairwise},
};

template <typename Json>
void to_json(Json& json, AlltoallvAlgorithm const value) {
    json = enum_name(alltoall_names, value);
}

} // namespace mpi

inline EnumNames<IdPlacement> const id_placement_names{
    {"random", IdPlacement::random},
    {"contiguous", IdPlacement::contiguous},
};

template <typename Json>
void to_json(Json& json, IdPlacement const value) {
    json = enum_name(id_placement_names, value);
}

} // namespace dss_mehnert
