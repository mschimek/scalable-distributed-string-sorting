// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include "mpi/big_type.hpp"
#include "mpi/environment.hpp"
#include "mpi/type_mapper.hpp"
#include "util/measuringTool.hpp"

namespace dss_schimek {
namespace mpi {

template <typename DataType>
inline std::vector<DataType> gather(DataType const& send_data, size_t root, environment env) {
    using dss_schimek::measurement::MeasuringTool;

    MeasuringTool& measuringTool = MeasuringTool::measuringTool();
    measuringTool.addRawCommunication(sizeof(DataType), "gather");

    data_type_mapper<DataType> dtm;
    std::vector<DataType> receive_data;
    if (env.rank() == root)
        receive_data.resize(env.size());
    MPI_Gather(
        &send_data,
        1,
        dtm.get_mpi_type(),
        receive_data.data(),
        1,
        dtm.get_mpi_type(),
        static_cast<int32_t>(root),
        env.communicator()
    );
    return receive_data;
}

template <typename DataType>
inline std::vector<DataType>
gatherv(std::vector<DataType>& send_data, size_t root, dss_schimek::mpi::environment env) {
    using namespace dss_schimek;
    using dss_schimek::measurement::MeasuringTool;

    MeasuringTool& measuringTool = MeasuringTool::measuringTool();
    measuringTool.addRawCommunication(sizeof(DataType), "gatherv");

    std::vector<size_t> receiveCounts = gather(send_data.size(), root, env);
    std::vector<size_t> offsets;
    offsets.reserve(receiveCounts.size() + 1);
    offsets.push_back(0);
    std::partial_sum(receiveCounts.begin(), receiveCounts.end(), std::back_inserter(offsets));

    data_type_mapper<DataType> dtm;
    std::vector<DataType> receive_data;
    if (env.rank() == root) {
        receive_data.resize(offsets.back());
        std::copy(send_data.begin(), send_data.end(), receive_data.begin() + offsets[root]);
        for (size_t i = root + 1; i < root + env.size(); ++i) {
            int32_t partner = static_cast<int32_t>(i % env.size());
            auto receiveType = dss_schimek::mpi::get_big_type<DataType>(receiveCounts[partner]);
            MPI_Recv(
                receive_data.data() + offsets[partner],
                1,
                receiveType,
                partner,
                42,
                env.communicator(),
                MPI_STATUSES_IGNORE
            );
        }
    } else {
        auto sendType = dss_schimek::mpi::get_big_type<DataType>(send_data.size());
        MPI_Send(send_data.data(), 1, sendType, root, 42, env.communicator());
    }
    return receive_data;
}

} // namespace mpi
} // namespace dss_schimek
