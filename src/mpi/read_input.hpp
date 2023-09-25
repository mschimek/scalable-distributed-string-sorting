// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <tlx/die/core.hpp>

#include "mpi/environment.hpp"
#include "mpi/shift.hpp"
#include "mpi/type_mapper.hpp"

namespace dss_schimek {

inline size_t get_file_size(std::string const& path) {
    std::ifstream in{path, std::ifstream::ate | std::ifstream::binary};
    tlx_die_verbose_unless(in.good(), "file not good");
    return in.tellg();
}

inline std::vector<unsigned char>
distribute_file(std::string const& input_path, size_t max_size = 0, mpi::environment env = {}) {
    MPI_File mpi_file;

    MPI_File_open(
        env.communicator(),
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

    size_t local_slice_size = global_file_size / env.size();
    uint32_t remaining = global_file_size % env.size();

    // leftover characters are distributed to the first `remaining` PEs
    size_t offset = env.rank() * local_slice_size + std::min(env.rank(), remaining);
    local_slice_size += (env.rank() < remaining);

    MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

    std::vector<unsigned char> result(local_slice_size);
    MPI_File_read(
        mpi_file,
        result.data(),
        local_slice_size,
        mpi::type_mapper<unsigned char>::type(),
        MPI_STATUS_IGNORE
    );
    return result;
}

inline std::vector<unsigned char>
distribute_lines(std::string const& input_path, size_t max_size = 0, mpi::environment env = {}) {
    auto result = distribute_file(input_path, max_size, env);
    auto first_newline = std::find(result.begin(), result.end(), '\n') - result.begin();

    std::replace(result.begin(), result.end(), '\n', static_cast<char>(0));

    // Shift the first string to the previous PE, so that each PE ends with complete string.
    // This is still not perfect, e.g. if a string spans multiple PEs ¯\_(ツ)_/¯.
    auto shifted_string = mpi::shift_left(result.data(), first_newline + 1, env);

    if (env.rank() + 1 < env.size()) {
        std::copy(shifted_string.begin(), shifted_string.end(), std::back_inserter(result));
    }
    if (!result.empty() && result.back() != 0) {
        result.emplace_back(0);
    }

    if (env.rank() > 0) {
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

} // namespace dss_schimek
