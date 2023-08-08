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

#include <kamping/collectives/gather.hpp>
#include <kamping/named_parameters.hpp>

#include "util/measurements.hpp"

namespace dss_mehnert {
namespace measurement {

template <typename Record, typename Value>
class NonTimer {
public:
    struct OutputFormat {
        Record record;
        Summary<size_t> summary;
        size_t median;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormat const& output) {
            return stream << Result << output.record << " min=" << output.summary.min
                          << " max=" << output.summary.max << " median=" << output.median
                          << " avg=" << output.summary.avg << " sum=" << output.summary.sum;
        }
    };

    struct OutputFormatSingle {
        Record record;
        size_t process;
        Value value;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormatSingle const& output) {
            return stream << Result << output.record << " process=" << output.process << " "
                          << Result << output.value;
        }
    };

    NonTimer() { entries.reserve(allocationSize); }

    template <typename Communicator>
    std::pair<std::vector<OutputFormat>, std::vector<OutputFormatSingle>>
    collect_on_root(Communicator const& comm) {
        using namespace kamping;
        std::vector<OutputFormat> output;
        std::vector<OutputFormatSingle> output_single;

        for (auto const& [record, value, collect]: entries) {
            std::vector<size_t> values;
            comm.gather(send_buf(value.getValue()), root(comm.root()), recv_buf(values));

            if (comm.is_root()) {
                if (collect) {
                    auto summary = describe<size_t>(values.begin(), values.end());
                    auto median = get_median<size_t>(values.begin(), values.end());
                    output.emplace_back(record, summary, median);
                } else {
                    Value value_{value};
                    for (size_t proc = 0; auto const& proc_value: values) {
                        value_.setValue(proc_value);
                        output_single.emplace_back(record, proc++, value_);
                    }
                }
            }
        }
        return {std::move(output), std::move(output_single)};
    }

    void add(Record const& record, Value value, bool collect) {
        auto& [entry, v, c] = entries.emplace_back(record, value, collect);
        auto counter = key_counter_.try_emplace(entry.pseudoKey(), 0);
        entry.setPseudoKeyCounter(counter.first->second++);
    }

private:
    using PseudoKey = typename Record::PseudoKey;
    static const size_t allocationSize = 128;

    std::vector<std::tuple<Record, Value, bool>> entries;
    std::map<PseudoKey, size_t> key_counter_;
};

} // namespace measurement
} // namespace dss_mehnert
