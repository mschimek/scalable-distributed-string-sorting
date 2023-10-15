// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <kamping/mpi_datatype.hpp>
#include <mpi.h>
#include <tlx/die/core.hpp>

#include "mpi/communicator.hpp"

namespace dss_mehnert {

template <typename T>
static inline std::vector<T>
shift_left(T* send_data, size_t const send_count, dss_mehnert::Communicator const& comm) {
    int const source = comm.rank() + 1 < comm.size() ? comm.rank() + 1 : 0;
    int const dest = comm.rank() > 0 ? comm.rank() - 1 : comm.size() - 1;

    size_t recv_count = 0;

    // clang-format off
    MPI_Sendrecv(&send_count, 1, kamping::mpi_datatype<size_t>(), dest,   0,
                 &recv_count, 1, kamping::mpi_datatype<size_t>(), source, MPI_ANY_TAG,
                 comm.mpi_communicator(), MPI_STATUS_IGNORE);

    std::vector<T> recv_data(recv_count);
    MPI_Sendrecv(send_data,        send_count, kamping::mpi_datatype<T>(), dest,   0,
                 recv_data.data(), recv_count, kamping::mpi_datatype<T>(), source, MPI_ANY_TAG,
                 comm.mpi_communicator(), MPI_STATUS_IGNORE);
    // clang-format on

    return recv_data;
}

inline size_t get_file_size(std::string const& path) {
    std::ifstream in{path, std::ifstream::ate | std::ifstream::binary};
    tlx_die_verbose_unless(in.good(), "file not good");
    return in.tellg();
}

inline std::vector<unsigned char> distribute_file_segments(
    std::string const& input_path,
    size_t const requested_size,
    bool const random,
    Communicator const& comm
) {
    MPI_File mpi_file;
    MPI_File_open(
        comm.mpi_communicator(),
        input_path.c_str(),
        MPI_MODE_RDONLY,
        MPI_INFO_NULL,
        &mpi_file
    );

    MPI_Offset file_size = 0;
    MPI_File_get_size(mpi_file, &file_size);
    size_t const segment_size = std::min(requested_size, static_cast<size_t>(file_size));
    size_t offset = 0;

    if (random) {
        std::random_device rd;
        std::mt19937_64 gen{rd()};
        std::uniform_int_distribution<size_t> dist{0, file_size - segment_size};
        offset = dist(gen);
    } else {
        auto const max_step = (file_size - segment_size) / std::max<size_t>(comm.size() - 1, 1);
        offset = comm.rank() * std::min<size_t>(segment_size, max_step);
    }

    assert(offset + segment_size <= static_cast<size_t>(file_size));
    MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

    std::vector<unsigned char> result(segment_size);
    MPI_File_read(
        mpi_file,
        result.data(),
        segment_size,
        kamping::mpi_datatype<unsigned char>(),
        MPI_STATUS_IGNORE
    );
    return result;
}

inline std::vector<unsigned char>
distribute_file(std::string const& input_path, size_t const max_size, Communicator const& comm) {
    MPI_File mpi_file;
    MPI_File_open(
        comm.mpi_communicator(),
        input_path.c_str(),
        MPI_MODE_RDONLY,
        MPI_INFO_NULL,
        &mpi_file
    );

    MPI_Offset global_file_size = 0;
    MPI_File_get_size(mpi_file, &global_file_size);
    if (max_size > 0) {
        global_file_size = std::min(max_size, static_cast<size_t>(global_file_size));
    }

    size_t local_slice_size = global_file_size / comm.size();

    // leftover characters are distributed to the first `remaining` PEs
    size_t const remaining = global_file_size % comm.size();
    size_t const offset = comm.rank() * local_slice_size + std::min(comm.rank(), remaining);
    local_slice_size += (comm.rank() < remaining);

    MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

    std::vector<unsigned char> result(local_slice_size);
    MPI_File_read(
        mpi_file,
        result.data(),
        local_slice_size,
        kamping::mpi_datatype<unsigned char>(),
        MPI_STATUS_IGNORE
    );
    return result;
}

inline std::vector<unsigned char>
distribute_lines(std::string const& input_path, size_t const max_size, Communicator const& comm) {
    auto result = distribute_file(input_path, max_size, comm);
    auto first_newline = std::find(result.begin(), result.end(), '\n') - result.begin();

    std::replace(result.begin(), result.end(), '\n', static_cast<char>(0));

    // Shift the first string to the previous PE, so that each PE ends with complete string.
    // This is still not perfect, e.g. if a string spans multiple PEs ¯\_(ツ)_/¯.
    auto shifted_string = shift_left(result.data(), first_newline + 1, comm);

    if (comm.rank() + 1 < comm.size()) {
        std::copy(shifted_string.begin(), shifted_string.end(), std::back_inserter(result));
    }
    if (!result.empty() && result.back() != 0) {
        result.emplace_back(0);
    }

    if (comm.rank() > 0) {
        result.erase(result.begin(), result.begin() + first_newline + 1);
    }

    return result;
}

struct RawStringsLines {
    std::vector<unsigned char> rawStrings;
    size_t lines;
};

inline RawStringsLines read_file(std::string const& path) {
    RawStringsLines data;
    auto& [raw_strings, lines] = data;
    raw_strings.reserve(1.5 * get_file_size(path));

    std::ifstream in{path};
    for (std::string line; std::getline(in, line); ++lines) {
        std::copy(line.begin(), line.end(), std::back_inserter(raw_strings));
        raw_strings.push_back(0);
    }

    std::cout << "lines: " << lines << std::endl;
    return data;
}

} // namespace dss_mehnert
