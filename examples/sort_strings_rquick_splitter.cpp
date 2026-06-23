// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

// Distributed string sorting with sample-sort-based merge sort, using RQuick
// (V2) only as the splitter sorter (i.e. to globally sort the small sample of
// splitter candidates). The bulk of the strings are sorted by merge sort, not
// by RQuick. For a top-level RQuick string sort, see sort_strings_rquick.cpp.

#include <kamping/environment.hpp>

#include "dss/dss.hpp"
#include "example_common.hpp"
#include "mpi/communicator.hpp"

int main(int argc, char** argv) {
    kamping::Environment env{argc, argv};
    dss_mehnert::Communicator comm{};

    auto local_input = dss::examples::make_local_input(
        comm.rank(), dss::examples::InputConfig{.seed = 0xBADCAFE});
    auto input_copy  = local_input;

    auto sorted_local =
        dss::run_sorter(local_input, comm, dss::SplitterSorter::RQuickV2);

    return dss::examples::verify_and_report(
        comm, sorted_local, input_copy, "example-rquick-splitter");
}
