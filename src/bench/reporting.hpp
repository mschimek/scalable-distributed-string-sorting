// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Lightweight JSON reporting for kamping::measurements timers, modelled after
// the `Report` mechanism in the kascade list-ranking benchmark.

#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <kamping/communicator.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/nlohmann_json_adapter/printer.hpp>
#include <nlohmann/json.hpp>

namespace dss_mehnert {

// Open an output stream for the given path. "stdout"/"stderr"/"" map to the
// respective standard streams, everything else to a file at exactly that path.
inline std::unique_ptr<std::ostream> make_output_stream(std::string const& output_file) {
    if (output_file.empty() || output_file == "stdout") {
        return std::make_unique<std::ostream>(std::cout.rdbuf());
    }
    if (output_file == "stderr") {
        return std::make_unique<std::ostream>(std::cerr.rdbuf());
    }

    auto file_stream = std::make_unique<std::ofstream>(output_file);
    if (!file_stream->is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }
    return file_stream;
}

// Collects the aggregated kamping timer trees of all iterations and emits them,
// together with an arbitrary config object, as a single JSON document.
class Report {
public:
    // Aggregate the current kamping timer tree and counters across all ranks
    // (collective!), append their JSON on the root PE, and reset both for the
    // next round.
    void step_iteration() {
        kamping::measurements::NLohmannJsonPrinter time_printer;
        kamping::measurements::timer().aggregate_and_print(time_printer);
        if (kamping::comm_world().is_root()) {
            times_.emplace_back(time_printer.json());
        }
        kamping::measurements::timer().clear();

        using counter_type = typename std::remove_reference_t<
            decltype(kamping::measurements::counter())>::DataType;
        kamping::measurements::NLohmannJsonPrinter<counter_type> counter_printer;
        kamping::measurements::counter().aggregate_and_print(counter_printer);
        if (kamping::comm_world().is_root()) {
            counters_.emplace_back(counter_printer.json());
        }
        kamping::measurements::counter().clear();
    }

    template <typename T>
    void push_config(T const& config) {
        config_ = config;
    }

    void print(std::ostream& out) {
        if (kamping::comm_world().is_root()) {
            nlohmann::ordered_json json;
            json["config"] = config_;
            json["timer"] = times_;
            json["counters"] = counters_;
            out << json.dump(2) << '\n';
        }
    }

private:
    nlohmann::ordered_json config_;
    std::vector<nlohmann::ordered_json> times_;
    std::vector<nlohmann::ordered_json> counters_;
};

} // namespace dss_mehnert
