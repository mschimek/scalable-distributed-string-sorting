// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

#include <kamping/collectives/reduce.hpp>
#include <kamping/named_parameters.hpp>

#include "util/measurements.hpp"

namespace dss_mehnert {
namespace measurement {

template <typename Record>
class LeanNonTimer {
public:
    struct OutputFormat {
        std::string description;
        Record record;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormat const& output) {
            return stream << "key=" << output.description << " " << Result << output.record;
        }
    };

    LeanNonTimer(std::string description_) : description(description_) {
        records.reserve(allocationSize);
    }

    template <typename Communicator>
    std::vector<OutputFormat> collect_on_root(Communicator const& comm) const {
        using namespace kamping;

        std::vector<OutputFormat> output;
        output.reserve(records.size());
        for (Record record: records) {
            auto local_value = record.getValue();
            auto result = comm.reduce(send_buf(local_value), op(ops::plus<>{}));

            if (comm.is_root()) {
                record.setValue(result.extract_recv_buffer()[0]);
                output.emplace_back(description, record);
            }
        }
        return output;
    }

    void add(Record const& record) {
        auto comp = [&](Record const& other) { return other.pseudoKey() == record.pseudoKey(); };
        auto it = std::find_if(records.begin(), records.end(), comp);
        if (it == records.end()) {
            records.push_back(record);
        } else {
            it->setValue(it->getValue() + record.getValue());
        }
    }

private:
    static const size_t allocationSize = 64;

    std::string description;
    std::vector<Record> records;
};

} // namespace measurement
} // namespace dss_mehnert
