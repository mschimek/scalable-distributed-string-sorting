// (c) 2024 Matthias Schimek
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Selection and tuning parameters for the alltoallv implementations in this directory.
// Kept separate from dispatch.hpp so that callers which only need to name an algorithm
// don't have to pull in every implementation.

#pragma once

#include <cstddef>

namespace dss_mehnert {
namespace mpi {

enum class AlltoallvCombinedKind { combined, native, direct, one_factor, pairwise };

// Selects how the 1-factor exchanges:
//   * windowed:     pipeline the exchanges over a fixed window of outstanding
//                   isend/irecv pairs.
//   * synchronized: perform the standard full 1-factor schedule as p (p-1 for even p)
//                   pairwise MPI_Sendrecv exchanges
enum class OneFactorMode { windowed, synchronized };

struct OneFactorParams {
    OneFactorMode mode = OneFactorMode::windowed;
    size_t num_slots = 16;
    bool use_issend = false;
};

} // namespace mpi
} // namespace dss_mehnert
