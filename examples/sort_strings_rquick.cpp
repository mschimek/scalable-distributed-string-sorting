// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

// Distributed string sorting with RQuick (V2) as the top-level sorter: the
// LCP-aware hypercube quicksort sorts all strings directly, rather than being
// used only to sort splitter candidates (see sort_strings_rquick_splitter.cpp).
//
// Note: RQuick partitions by pivot value, not by count, so the sorted output is
// distributed across PEs but may be unbalanced (some PEs can end up empty). The
// global order across PEs (in rank order) is what `verify_and_report` checks.

#include <kamping/environment.hpp>

#include "dss/dss.hpp"
#include "dss/mpi/communicator.hpp"
#include "example_common.hpp"

int main(int argc, char** argv) {
    kamping::Environment env{argc, argv};
    dss_mehnert::Communicator comm{};

    auto local_input = dss::examples::make_local_input(
        comm.rank(), dss::examples::InputConfig{.seed = 0xBADCAFE});
    auto input_copy  = local_input;  // run_rquick consumes its argument

    auto sorted_local = dss::run_rquick(local_input, comm);

    return dss::examples::verify_and_report(
        comm, sorted_local, input_copy, "example-rquick");
}
