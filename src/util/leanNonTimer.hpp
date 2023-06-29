// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

#include "mpi/environment.hpp"
#include "mpi/type_mapper.hpp"

namespace dss_schimek {

template <typename DataType>
inline std::vector<DataType> allgather_(
    DataType& send_data, dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
) {
    dss_schimek::mpi::data_type_mapper<DataType> dtm;
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

template <typename Record>
class LeanNonTimer {
    using PseudoKey = typename Record::PseudoKey;
    static const size_t allocationSize = 10000;

public:
    LeanNonTimer() { records.reserve(allocationSize); }

    template <typename OutIt>
    void collect(OutIt out) {
        for (Record& record: records) {
            collect(out, record);
        }
    }

    void add(Record const& record) {
        auto it = std::find_if(records.begin(), records.end(), [&](Record const& storedRecord) {
            return storedRecord.pseudoKey() == record.pseudoKey();
        });
        if (it == records.end())
            records.push_back(record);
        else
            it->setValue(it->getValue() + record.getValue());
    }

private:
    dss_schimek::mpi::environment env;
    std::vector<Record> records;

    template <typename OutIt>
    inline void collect(OutIt out, Record& record) {
        // Cannot allgather Record as a whole since it is not trivially_copyable
        // -> workaround send only values

        auto ownValue = record.getValue();
        auto globalValues = allgather_(ownValue);
        const size_t sum =
            std::accumulate(globalValues.begin(), globalValues.end(), static_cast<size_t>(0u));
        record.setValue(sum);
        out = record;
    }
};
} // namespace dss_schimek
