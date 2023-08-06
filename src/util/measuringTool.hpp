// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <ostream>
#include <vector>

#include "mpi/communicator.hpp"
#include "util/lean_non_timer.hpp"
#include "util/measurements.hpp"
#include "util/non_timer.hpp"
#include "util/timer.hpp"

namespace dss_mehnert {
namespace measurement {

class MeasuringTool {
public:
    using NonTimerRecord = PhaseCounterRoundDescription;
    using TimerKey = PhaseRoundDescription;
    using TimerValue = CounterPerPhase;

    static MeasuringTool& measuringTool() {
        static MeasuringTool measuringTool;
        return measuringTool;
    }

    void reset() { state = {}; }

    // todo add back option to print value on every PE
    void add(size_t value, std::string_view description, bool collect = true) {
        if (!state.disabled) {
            if (state.verbose && comm.is_root()) {
                std::cout << description << std::endl;
            }
            NonTimerRecord record{state.phase, 0u, state.round, std::string{description}};
            state.non_timer.add(record, {value}, collect);
        }
    }

    void addRawCommunication(size_t value, std::string_view description) {
        // todo should this respect `state.disabled`?
        if (state.verbose && comm.is_root()) {
            std::cout << description << std::endl;
        }
        state.comm_volume.add({state.phase, value});
    }

    void start(std::string_view description) {
        if (!state.disabled) {
            if (state.verbose && comm.is_root()) {
                std::cout << description << std::endl;
            }
            state.timer.start({state.phase, state.round, std::string{description}}, {});
        }
    }

    void start(std::string_view phase, std::string_view desc) {
        setPhase(phase);
        start(desc);
    }

    void stop(std::string_view description) {
        if (!state.disabled) {
            if (state.verbose && comm.is_root()) {
                std::cout << description << std::endl;
            }
            state.timer.stop({state.phase, state.round, std::string{description}}, false, comm);
        }
    }

    void stop(std::string_view phase, std::string_view desc) {
        setPhase(phase);
        stop(desc);
    }

    void stop(std::string_view phase, std::string_view desc, Communicator const& comm) {
        setPhase(phase);
        if (!state.disabled) {
            if (state.verbose && comm.is_root()) {
                std::cout << desc << std::endl;
            }
            state.timer.stop({state.phase, state.round, std::string{desc}}, true, comm);
        }
    }

    void writeToStream(std::ostream& stream) {
        disable();
        auto records = state.non_timer.collect(comm);
        write_records(stream, records.first);
        write_records(stream, records.second);
        write_records(stream, state.comm_volume.collect(comm));
        write_records(stream, state.timer.collect(comm));
        enable();
    }

    bool isEnabled() { return !state.disabled; }
    void setEnabled(bool enable) { state.disabled = !enable; }
    void enable() { state.disabled = false; }
    void disable() { state.disabled = true; }
    void setVerbose(bool value) { state.verbose = value; }
    void setPrefix(std::string_view prefix_) { state.prefix = prefix_; }
    void setPhase(std::string_view phase) { state.phase = phase; }
    void setRound(size_t round) { state.round = round; }

private:
    struct State {
        bool disabled = false;
        bool verbose = false;
        std::string prefix = "";
        std::string phase = "none";
        size_t round = 0;
        NonTimer<NonTimerRecord, SimpleValue> non_timer;
        LeanNonTimer<PhaseValue> comm_volume = {"comm_volume"};
        Timer<TimerKey, TimerValue> timer;
    };

    Communicator comm;
    State state;

    void write_records(std::ostream& stream, auto const& records) {
        for (auto const& record: records) {
            stream << state.prefix << " " << record << "\n";
        }
    }
};

} // namespace measurement
} // namespace dss_mehnert

namespace dss_schimek {
namespace measurement {

using dss_mehnert::measurement::MeasuringTool;


} // namespace measurement
} // namespace dss_schimek
