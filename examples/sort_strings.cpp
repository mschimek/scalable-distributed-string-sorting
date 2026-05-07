// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <kamping/environment.hpp>

#include "dss/dss.hpp"
#include "example_common.hpp"
#include "mpi/communicator.hpp"

int main(int argc, char** argv) {
    kamping::Environment env{argc, argv};
    dss_mehnert::Communicator comm{};

    auto local_input =
        dss::examples::make_local_input(comm.rank(), dss::examples::InputConfig{});
    auto input_copy  = local_input;  // run_sorter consumes its argument

    auto sorted_local = dss::run_sorter(local_input, comm);

    return dss::examples::verify_and_report(comm, sorted_local, input_copy, "example");
}
