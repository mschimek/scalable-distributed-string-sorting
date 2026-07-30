// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The multi-level group sizes: which of the configured levels a run actually uses depends on the
// number of PEs, since a level only applies while its group size is smaller than the communicator.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

#include "dss/mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

inline auto get_first_level(std::vector<size_t> const& levels, Communicator const& comm) {
    return std::find_if(levels.begin(), levels.end(), [&](auto const& group_size) {
        return group_size < comm.size();
    });
}

// the input is partitioned once per level and once more in the final round
inline size_t get_num_levels(std::vector<size_t> const& levels, Communicator const& comm) {
    return static_cast<size_t>(std::distance(get_first_level(levels, comm), levels.end())) + 1;
}

} // namespace bench
} // namespace dss_mehnert
