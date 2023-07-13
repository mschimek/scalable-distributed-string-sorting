// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "mpi/big_type.hpp"
#include "mpi/environment.hpp"
#include "mpi/shift.hpp"
#include "mpi/type_mapper.hpp"

namespace dss_schimek {

size_t getFileSize(std::string const& path) {
    std::ifstream in(path, std::ifstream::ate | std::ifstream::binary);
    if (!in.good()) {
        std::cout << " file not good" << std::endl;
        std::abort();
    }
    const size_t fileSize = in.tellg();
    in.close();
    return fileSize;
}

struct RawStringsLines {
    std::vector<unsigned char> rawStrings;
    size_t lines;
};

template <typename InputIterator>
bool containsDuplicate(InputIterator begin, InputIterator end) {
    auto newEnd = std::unique(begin, end);
    return newEnd != end;
}

std::vector<unsigned char> distribute_strings(
    std::string const& input_path,
    size_t max_size = 0,
    dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
) {
    MPI_File mpi_file;

    MPI_File_open(
        env.communicator(),
        (char*)input_path.c_str(), // ugly cast to use old C interface
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
    int64_t larger_slices = global_file_size % env.size();

    size_t offset;
    if (env.rank() < larger_slices) {
        ++local_slice_size;
        offset = local_slice_size * env.rank();
    } else {
        offset = larger_slices * (local_slice_size + 1);
        offset += (env.rank() - larger_slices) * local_slice_size;
    }

    MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

    std::vector<unsigned char> result(local_slice_size);

    MPI_File_read(
        mpi_file,
        result.data(),
        local_slice_size,
        mpi::type_mapper<unsigned char>::type(),
        MPI_STATUS_IGNORE
    );

    size_t first_end = 0;
    while (first_end < result.size() && result[first_end] != '\n') {
        ++first_end;
    }
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] == '\n')
            result[i] = 0;
    }

    std::vector<unsigned char> end_of_last_string =
        dss_schimek::mpi::shift_left(result.data(), first_end + 1, env);
    // We copy this string, even if it's not the end of the last one on the
    // previous PE, but a new string. This way, we can delete it on the sending
    // PE without checking if it was the end.
    if (env.rank() + 1 < env.size()) {
        std::copy_n(
            end_of_last_string.begin(),
            end_of_last_string.size(),
            std::back_inserter(result)
        );
    } else {
        if (result.back() != 0) {
            result.emplace_back(0);
        } // Make last string end
    }
    if (env.rank() > 0) { // Delete the sent string
        result.erase(result.begin(), result.begin() + first_end + 1);
    }

    return result;
}

std::vector<unsigned char> readFilePerPE(std::string const& path, mpi::environment env) {
    std::ifstream in(path, std::ifstream::binary);
    if (!in.good()) {
        std::cout << "file not good on rank: " << env.rank() << std::endl;
        std::abort();
    }

    const uint64_t fileSize = getFileSize(path);
    std::vector<unsigned char> content;
    for (size_t i = 0; i < fileSize; ++i) {
        unsigned char curChar = in.get();
        content.push_back(curChar);
    }
    return content;
}

dss_schimek::RawStringsLines readFile(std::string const& path) {
    using dss_schimek::RawStringsLines;
    RawStringsLines data;
    const size_t fileSize = getFileSize(path);
    std::ifstream in(path);
    std::vector<unsigned char>& rawStrings = data.rawStrings;
    rawStrings.reserve(1.5 * fileSize);

    std::string line;
    data.lines = 0u;
    while (std::getline(in, line)) {
        ++data.lines;
        for (unsigned char curChar: line)
            rawStrings.push_back(curChar);
        rawStrings.push_back(0);
    }
    std::cout << "lines: " << data.lines << std::endl;
    in.close();
    return data;
}

std::vector<unsigned char> readFileInParallel(std::string const& path, mpi::environment env) {
    std::ifstream in(path, std::ifstream::binary);
    if (!in.good()) {
        std::cout << "file not good on rank: " << env.rank() << std::endl;
        std::abort();
    }

    const uint64_t fileSize = getFileSize(path);
    uint64_t localSliceSize = fileSize / env.size();
    uint64_t largerSlices = fileSize % env.size();
    uint64_t localOffset = localSliceSize * env.rank();
    if (env.rank() < largerSlices) {
        ++localSliceSize;
        localOffset = localSliceSize * env.rank();
    } else {
        localOffset = largerSlices * (localSliceSize + 1);
        localOffset += (env.rank() - largerSlices) * localSliceSize;
    }
    if (localOffset > 0u)
        in.seekg(localOffset - 1);

    std::vector<unsigned char> slice;
    slice.reserve(localSliceSize + 1);
    uint64_t startOffset = 0u;
    if (localOffset > 0u) {
        for (size_t i = 0; i < localSliceSize + 1u; ++i) {
            unsigned char curChar = in.get();
            slice.push_back(curChar);
        }
        auto it = slice.begin();
        for (; it < slice.end() && *it != '\n'; ++it)
            ;
        if (it == slice.end()) {
            std::cout << "rank: " << env.rank() << " reached end of slice without a starting string"
                      << std::endl;
            std::abort();
        }
        startOffset = it - slice.begin() + 1u;
    } else {
        for (size_t i = 0; i < localSliceSize; ++i) {
            unsigned char curChar = in.get();
            slice.push_back(curChar);
        }
    }

    while (slice.back() != '\n' && localOffset + slice.size() < fileSize)
        slice.push_back(in.get());

    std::vector<unsigned char> rawStrings;
    rawStrings.reserve(slice.size() - startOffset);
    for (size_t i = startOffset; i < slice.size(); ++i) {
        rawStrings.push_back(slice[i]);
        if (rawStrings.back() == '\n')
            rawStrings.back() = 0u;
    }
    if (rawStrings.back() != 0u)
        rawStrings.push_back(0u);
    return rawStrings;
}

std::vector<unsigned char> readFileAndDistribute(std::string const& path, mpi::environment env) {
    std::vector<unsigned char> rawStrings;
    size_t recvCount = 0;
    dss_schimek::mpi::data_type_mapper<size_t> dtm;
    std::vector<size_t> sendCounts;
    if (env.rank() == 0) {
        dss_schimek::RawStringsLines data = readFile(path);
        rawStrings = std::move(data.rawStrings);
        std::cout << "lines in readFileAndDistribute: " << data.lines << std::endl;
        const size_t smallPacketSize = data.lines / env.size();
        const size_t bigPacketSize = smallPacketSize + (data.lines % env.size());
        size_t curPos = 0;
        size_t lastPos = 0;
        for (size_t i = 0; i + 1 < env.size(); ++i) {
            for (size_t numStrings = 0; numStrings < smallPacketSize; ++numStrings) {
                while (rawStrings[curPos] != 0)
                    ++curPos;
                ++curPos;
            }
            sendCounts.push_back(curPos - lastPos);
            lastPos = curPos;
        }
        std::cout << "rawStringsize: " << rawStrings.size() << " lastPos: " << lastPos << std::endl;
        sendCounts.push_back(rawStrings.size() - lastPos);
        std::cout << "smallPacketSize: " << smallPacketSize << " bigPacketSize: " << bigPacketSize
                  << std::endl;
    }
    MPI_Scatter(
        sendCounts.data(),
        1,
        dtm.get_mpi_type(),
        &recvCount,
        1,
        dtm.get_mpi_type(),
        0,
        env.communicator()
    );

    std::vector<unsigned char> localRawStrings(recvCount);
    if (env.rank() == 0) {
        std::vector<size_t> offsets;
        offsets.push_back(0);
        std::partial_sum(sendCounts.begin(), sendCounts.end(), std::back_inserter(offsets));

        std::copy(rawStrings.begin(), rawStrings.begin() + sendCounts[0], localRawStrings.begin());
        for (int32_t i = 1; i < static_cast<int32_t>(env.size()); ++i) {
            auto receive_type = dss_schimek::mpi::get_big_type<unsigned char>(sendCounts[i]);
            MPI_Send(rawStrings.data() + offsets[i], 1, receive_type, i, 42, env.communicator());
        }
    } else {
        auto receive_type = dss_schimek::mpi::get_big_type<unsigned char>(recvCount);
        MPI_Recv(
            localRawStrings.data(),
            1,
            receive_type,
            0,
            42,
            env.communicator(),
            MPI_STATUSES_IGNORE
        );
    }

    std::cout << "rank: " << env.rank() << " recvCount: " << recvCount << std::endl;
    return localRawStrings;
}

} // namespace dss_schimek
