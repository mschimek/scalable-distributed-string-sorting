/*******************************************************************************
 * mpi/allgather.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "mpi/big_type.hpp"
#include "mpi/environment.hpp"
#include "mpi/type_mapper.hpp"
#include "util/measuringTool.hpp"

namespace dss_schimek::mpi {
template <typename DataType>
inline DataType broadcast(DataType& send_data, environment env = environment()) {
    data_type_mapper<DataType> dtm;
    DataType recvElem = send_data;
    MPI_Bcast(&recvElem, 1, dtm.get_mpi_type(), 0, env.communicator());
    return recvElem;
}
template <typename DataType>
inline std::vector<DataType> allgather(DataType& send_data, environment env = environment()) {
    using dss_schimek::measurement::MeasuringTool;

    MeasuringTool& measuringTool = MeasuringTool::measuringTool();
    measuringTool.addRawCommunication(sizeof(DataType), "allgather");

    data_type_mapper<DataType> dtm;
    std::vector<DataType> receive_data(env.size());
    MPI_Allgather(
        &send_data,
        1,
        dtm.get_mpi_type(),
        receive_data.data(),
        1,
        dtm.get_mpi_type(),
        env.communicator()
    );
    return receive_data;
}

// template<typename DataType>
// std::vector<DataType> allgatherv_my(DataType* data, size_t size,
//                dss_schimek::mpi::environment env = dss_schimek::mpi::environment()) {
//
//  std::vector<size_t> sendCounts = dss_schimek::mpi::allgather(size);
//  std::vector<size_t> offsets;
//  offsets.reserve(env.size());
//  offsets.emplace_back(0);
//  for (size_t i = 1; i < env.size(); ++i) {
//    offsets.emplace_back(offsets.back() + sendCounts[i -  1]);
//  }
//  //if (env.rank() == 0)
//  //for (size_t i = 0; i < offsets.size(); ++i) {
//  //  std::cout << i <<  "offset: " << offsets[i] << std::endl;
//  //}
//
//  std::vector<DataType> recvBuffer(offsets.back() + sendCounts.back());
//  std::copy_n(data, size, recvBuffer.data() + offsets[env.rank()]);
//
//  dss_schimek::mpi::data_type_mapper<DataType> dtm;
//  std::vector<MPI_Request> mpiRequest(2 * (env.size() - 1));
//  for (size_t i = 1; i < env.size(); ++i) {
//    int partner = (env.rank() + i) % env.size();
//    MPI_Isend(data,
//        size,
//        dtm.get_mpi_type(),
//        partner,
//        42,
//        env.communicator(),
//        &mpiRequest[2*(i - 1)]);
//
//    MPI_Irecv(
//        recvBuffer.data() + offsets[partner],
//        sendCounts[partner],
//        dtm.get_mpi_type(),
//        partner,
//        42,
//        env.communicator(),
//        &mpiRequest[2*(i -1) + 1]);
//  }
//  MPI_Waitall(2 * env.size() - 2, mpiRequest.data(), MPI_STATUSES_IGNORE);
//  return recvBuffer;
//}

template <typename DataType>
static inline std::vector<DataType>
allgatherv_small(std::vector<DataType>& send_data, environment env = environment()) {
    // measurement in overload that sends

    int32_t local_size = send_data.size();
    std::vector<int32_t> receiving_sizes = allgather(local_size);
    return allgatherv_small(send_data, receiving_sizes, env);
}

template <typename DataType>
static inline std::vector<DataType> allgatherv_small(
    std::vector<DataType>& send_data,
    std::vector<int32_t>& receiving_sizes,
    environment env = environment()
) {
    using dss_schimek::measurement::MeasuringTool;
    MeasuringTool& measuringTool = MeasuringTool::measuringTool();

    int32_t local_size = send_data.size();

    std::vector<int32_t> receiving_offsets(env.size(), 0);
    for (size_t i = 1; i < receiving_sizes.size(); ++i) {
        receiving_offsets[i] = receiving_offsets[i - 1] + receiving_sizes[i - 1];
    }

    std::vector<DataType> receiving_data(receiving_sizes.back() + receiving_offsets.back());

    measuringTool.addRawCommunication(
        send_data.size() * sizeof(DataType) * env.size(),
        "allgatherv_small"
    );

    data_type_mapper<DataType> dtm;
    MPI_Allgatherv(
        send_data.data(),
        local_size,
        dtm.get_mpi_type(),
        receiving_data.data(),
        receiving_sizes.data(),
        receiving_offsets.data(),
        dtm.get_mpi_type(),
        env.communicator()
    );

    return receiving_data;
}
template <typename DataType>
static inline std::vector<DataType>
allgatherv(std::vector<DataType>& send_data, environment env = environment()) {
    using dss_schimek::measurement::MeasuringTool;
    MeasuringTool& measuringTool = MeasuringTool::measuringTool();

    size_t local_size = send_data.size();
    std::vector<size_t> receiving_sizes = allgather(local_size);

    std::vector<size_t> receiving_offsets(env.size(), 0);
    for (size_t i = 1; i < receiving_sizes.size(); ++i) {
        receiving_offsets[i] = receiving_offsets[i - 1] + receiving_sizes[i - 1];
    }

    if (receiving_sizes.back() + receiving_offsets.back() < env.mpi_max_int()) {
        std::vector<int32_t> receiving_sizes_small(receiving_sizes.size());
        for (size_t i = 0; i < receiving_sizes.size(); ++i)
            receiving_sizes_small[i] = receiving_sizes[i];
        return allgatherv_small(send_data, receiving_sizes_small, env);
    } else {
        measuringTool.addRawCommunication(
            send_data.size() * sizeof(DataType) * env.size(),
            "allgatherv_small"
        );

        std::vector<MPI_Request> mpi_requests(2 * env.size());
        std::vector<DataType> receiving_data(receiving_sizes.back() + receiving_offsets.back());

        for (uint32_t i = 0; i < env.size(); ++i) {
            auto receive_type = get_big_type<DataType>(receiving_sizes[i]);
            MPI_Irecv(
                receiving_data.data() + receiving_offsets[i],
                1,
                receive_type,
                i,
                44227,
                env.communicator(),
                &mpi_requests[i]
            );
        }
        auto send_type = get_big_type<DataType>(local_size);
        for (uint32_t i = env.rank(); i < env.rank() + env.size(); ++i) {
            int32_t target = i % env.size();
            MPI_Isend(
                send_data.data(),
                1,
                send_type,
                target,
                44227,
                env.communicator(),
                &mpi_requests[env.size() + target]
            );
        }
        MPI_Waitall(2 * env.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);
        return receiving_data;
    }
}

} // namespace dss_schimek::mpi

namespace dss_schimek::mpi {
template <typename Char>
static inline std::vector<Char> allgather_strings(
    std::vector<Char>& raw_string_data,
    dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
) {
    // measure is downward the call stack
    auto receiving_data = dss_schimek::mpi::allgatherv(raw_string_data, env);
    return std::vector(std::move(receiving_data));
}
} // namespace dss_schimek::mpi
/******************************************************************************/
