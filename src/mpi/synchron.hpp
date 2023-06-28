// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include "mpi/environment.hpp"

namespace dss_schimek::mpi {
template <typename Functor>
void execute_in_order(
    Functor const& functor, dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
) {
    for (size_t i = 0; i < env.size(); ++i) {
        env.barrier();
        if (env.rank() == i)
            functor();
        volatile size_t j = 0;
        for (j = 0; j < 100000; j += 1)
            j += 1;
        env.barrier();
    }

    env.barrier();
}
} // namespace dss_schimek::mpi
