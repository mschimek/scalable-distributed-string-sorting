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
    // Columns:
    // Prefix | Phase | CounterPerPhase | Round | Description | Type |
    // RawCommunication | SumUp |  Value
    //
    // NonTimer: Duplicates are allowed (but since counterPerPhase is
    // incremented there are no duplicates if all fields are taken into account)
    // Timer: Key = (Phase, Round, Description) Value = (CounterPerPhase, Type,
    // RawCommunication, SumUp, Value)

    // NonTimerRecord must contain type PseudoKey and functions pseudoKey() and
    // setPseudoKeyCounter, setValue(), getValue()
    using NonTimerRecord = PhaseCounterRoundDescription;
    // TimerKey must contain type PseudoKey and function pseudoKey()
    using TimerKey = PhaseRoundDescription;
    // TimerValue must contain functions setPseudoKeyCounter()
    using TimerValue = CounterPerPhase;

public:
    static MeasuringTool& measuringTool() {
        static MeasuringTool measuringTool;
        return measuringTool;
    }

    void reset() { state = {}; }

    void add(size_t value) {
        if (state.disabled)
            return;
        add(value, "unkown");
    }

    // todo add back option to print value on every PE
    void add(size_t value, std::string_view description, bool collect = true) {
        if (state.disabled)
            return;
        if (state.verbose && comm.is_root())
            std::cout << description << std::endl;
        state.nonTimer
            .add({state.curPhase, 0u, state.curRound, std::string{description}}, value, collect);
    }

    void addRawCommunication(size_t value, std::string_view description) {
        if (state.disabled)
            return;
        if (state.verbose && comm.is_root())
            std::cout << description << std::endl;

        state.communicationVolume.add({state.curPhase, value});
    }

    void start(std::string_view description) {
        if (state.disabled)
            return;
        if (state.verbose && comm.is_root())
            std::cout << description << std::endl;

        state.timer.start({state.curPhase, state.curRound, std::string{description}}, {});
    }

    void stop(std::string_view description) {
        if (state.disabled)
            return;
        if (state.verbose && comm.is_root())
            std::cout << description << std::endl;

        state.timer.stop({state.curPhase, state.curRound, std::string{description}});
    }

    void writeToStream(std::ostream& stream) {
        disable();
        auto records = state.nonTimer.collect(comm);
        write_records(stream, records.first);
        write_records(stream, records.second);
        write_records(stream, state.communicationVolume.collect(comm));
        write_records(stream, state.timer.collect(comm));
        enable();
    }

    void disableBarrier(bool value) { state.timer.setDisableBarrier(value); }

    void enable() { state.disabled = false; }

    void disable() { state.disabled = true; }

    void setVerbose(bool const value) { state.verbose = value; }

    void setPrefix(std::string_view prefix_) { state.prefix = prefix_; }

    void setPhase(std::string_view phase) { state.curPhase = phase; }

    void setRound(size_t round) { state.curRound = round; }

private:
    struct State {
        bool disabled = false;
        bool verbose = false;
        std::string prefix = "";
        std::string curPhase = "none";
        size_t curRound = 0;
        NonTimer<NonTimerRecord, size_t> nonTimer;
        LeanNonTimer<PhaseValue> communicationVolume = {"comm_volume"};
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
