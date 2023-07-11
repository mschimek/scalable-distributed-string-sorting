// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <numeric>
#include <vector>

#include <kamping/collectives/allgather.hpp>
#include <kamping/named_parameters.hpp>

#include "mpi/communicator.hpp"
#include "util/measurements.hpp"

namespace dss_mehnert {
namespace measurement {

template <typename Record, typename Value>
class NonTimer {
public:
    struct OutputFormat {
        Record record;
        Value min, max, avg, sum;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormat const& output) {
            return stream << Result << output.record << " min=" << output.min
                          << " max=" << output.max << " avg=" << output.avg
                          << " sum=" << output.sum;
        }
    };

    struct OutputFormatSingle {
        Record record;
        size_t process;
        Value value;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormatSingle const& output) {
            return stream << Result << output.record << " process=" << output.process
                          << " value=" << output.value;
        }
    };

    NonTimer() { entries.reserve(allocationSize); }

    std::pair<std::vector<OutputFormat>, std::vector<OutputFormatSingle>>
    collect(Communicator const& comm) {
        std::vector<OutputFormat> output;
        std::vector<OutputFormatSingle> output_single;
        for (auto const& [record, value, collect]: entries) {
            auto values = comm.allgather(kamping::send_buf(value)).extract_recv_buffer();

            if (collect) {
                auto min = std::min_element(std::begin(values), std::end(values));
                auto max = std::max_element(std::begin(values), std::end(values));
                auto sum = std::accumulate(std::begin(values), std::end(values), 0);
                output.emplace_back(record, *min, *max, sum / comm.size(), sum);
            } else {
                for (auto proc = 0; auto const& proc_value: values) {
                    output_single.emplace_back(record, proc++, proc_value);
                }
            }
        }
        return {std::move(output), std::move(output_single)};
    }

    void add(Record const& record, Value value, bool collect) {
        auto& [entry, v, c] = entries.emplace_back(record, value, collect);
        auto counter = pseudoKeyToCounter.try_emplace(entry.pseudoKey(), 0);
        entry.setPseudoKeyCounter(counter.first->second++);
    }

private:
    using PseudoKey = typename Record::PseudoKey;
    static const size_t allocationSize = 128;

    std::vector<std::tuple<Record, Value, bool>> entries;
    std::map<PseudoKey, size_t> pseudoKeyToCounter;
};

} // namespace measurement
} // namespace dss_mehnert
