/*******************************************************************************
 * mpi/big_type.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Based on
 *     https://github.com/jeffhammond/BigMPI/blob/master/src/type_contiguous_x.c
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <limits>

#include <kamping/mpi_datatype.hpp>
#include <mpi.h>

namespace dss_schimek::mpi {

template <typename DataType>
MPI_Datatype get_big_type(const size_t size) {
    size_t mpi_max_int = std::numeric_limits<int>::max();
    size_t num_blocks = size / mpi_max_int;
    size_t remainder = size % mpi_max_int;

    MPI_Datatype result_type, block_type, blocks_type;
    MPI_Type_contiguous(mpi_max_int, kamping::mpi_datatype<DataType>(), &block_type);
    MPI_Type_contiguous(num_blocks, block_type, &blocks_type);

    if (remainder > 0) {
        MPI_Datatype leftover_type;
        MPI_Type_contiguous(remainder, kamping::mpi_datatype<DataType>(), &leftover_type);

        MPI_Aint lb, extent;
        MPI_Type_get_extent(kamping::mpi_datatype<DataType>(), &lb, &extent);
        MPI_Aint displ = num_blocks * mpi_max_int * extent;
        MPI_Aint displs[2] = {0, displ};
        std::int32_t blocklen[2] = {1, 1};
        MPI_Datatype mpitypes[2] = {blocks_type, leftover_type};
        MPI_Type_create_struct(2, blocklen, displs, mpitypes, &result_type);
        MPI_Type_commit(&result_type);
        MPI_Type_free(&leftover_type);
        MPI_Type_free(&blocks_type);
    } else {
        result_type = blocks_type;
        MPI_Type_commit(&result_type);
    }
    return result_type;
}

} // namespace dss_schimek::mpi

/******************************************************************************/
