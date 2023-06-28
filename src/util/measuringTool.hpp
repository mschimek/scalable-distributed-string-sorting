// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <ostream>
#include <vector>

#include "mpi/environment.hpp"
#include "util/leanNonTimer.hpp"
#include "util/nonTimer.hpp"
#include "util/timer.hpp"

namespace dss_schimek {
namespace measurement {

struct PhaseValue {
    using PseudoKey = std::string;

    std::string phase;
    size_t value;

    PhaseValue(std::string const& phase, const size_t value) : phase(phase), value(value) {}
    std::string const& pseudoKey() const {
        return phase;
    }
    void setValue(size_t value_) {
        value = value_;
    }
    size_t getValue() const {
        return value;
    }
};

struct PhaseCounterPerPhaseRoundDescriptionTypeRawCommunicationSumUpValue {
    using PseudoKey = std::string;

    std::string phase;
    size_t counterPerPhase;
    size_t round;
    std::string description;
    std::string type;
    bool rawCommunication;
    bool sumUp;
    size_t value;

    PhaseCounterPerPhaseRoundDescriptionTypeRawCommunicationSumUpValue(
        std::string const& phase,
        const size_t counterPerPhase,
        const size_t round,
        std::string const& description,
        std::string const& type,
        bool const rawCommunication,
        bool const sumUp,
        const size_t value
    )
        : phase(phase),
          counterPerPhase(counterPerPhase),
          round(round),
          description(description),
          type(type),
          rawCommunication(rawCommunication),
          sumUp(sumUp),
          value(value) {}

    std::string const& pseudoKey() const {
        return phase;
    }

    void setPseudoKeyCounter(size_t counter) {
        counterPerPhase = counter;
    }

    void setType(std::string const& type_) {
        type = type_;
    }

    void setValue(size_t value_) {
        value = value_;
    }

    size_t getValue() const {
        return value;
    }

    bool getSumUp() {
        return sumUp;
    }

    friend std::ostream& operator<<(
        std::ostream& stream,
        PhaseCounterPerPhaseRoundDescriptionTypeRawCommunicationSumUpValue& elem
    ) {
        return stream << "[" << elem.phase << ", " << elem.counterPerPhase << ", " << elem.round
                      << ", " << elem.description << ", " << elem.rawCommunication << ", "
                      << elem.sumUp << elem.value << "]";
    }
};

struct PhaseRoundDescription {
    using PseudoKey = std::string;

    std::string phase;
    size_t round;
    std::string description;

    PhaseRoundDescription(
        std::string const& phase, const size_t round, std::string const& description
    )
        : phase(phase),
          round(round),
          description(description) {}

    std::string const& pseudoKey() const {
        return phase;
    }

    bool operator<(PhaseRoundDescription const& rhs) const {
        return std::tie(phase, round, description)
               < std::tie(rhs.phase, rhs.round, rhs.description);
    }

    friend std::ostream& operator<<(std::ostream& stream, PhaseRoundDescription const& elem) {
        return stream << "[" << elem.phase << ", " << elem.round << ", " << elem.description << "]";
    }
};

struct CounterPerPhaseTypeRawCommunicationSumUpValue {
    size_t counterPerPhase;
    std::string type;
    bool rawCommunication;
    bool sumUp;
    size_t value;
    CounterPerPhaseTypeRawCommunicationSumUpValue(
        const size_t counterPerPhase,
        std::string const& type,
        bool const rawCommunication,
        bool const sumUp,
        const size_t value
    )
        : counterPerPhase(counterPerPhase),
          type(type),
          rawCommunication(rawCommunication),
          sumUp(sumUp),
          value(value) {}

    void setPseudoKeyCounter(size_t counter) {
        counterPerPhase = counter;
    }

    void setType(std::string const& type_) {
        type = type_;
    }

    void setValue(size_t value_) {
        value = value_;
    }
};

class MeasuringTool {
    // Columns:
    // Prefix | Phase | CounterPerPhase | Round | Description | Type |
    // RawCommunication | SumUp |  Value
    //
    // NonTimer: Duplicates are allowed (but since counterPerPhase is
    // incremented there are no duplicates if all fields are taken into account)
    // Timer: Key = (Phase, Round, Description) Value = (CounterPerPhase, Type,
    // RawCommunication, SumUp, Value)

    using NonTimerRecord = PhaseCounterPerPhaseRoundDescriptionTypeRawCommunicationSumUpValue;
    // NonTimerRecord must contain type PseudoKey and functions pseudoKey() and
    // setPseudoKeyCounter, setValue(), getValue()
    using TimerKey = PhaseRoundDescription;
    // TimerKey must contain type PseudoKey and function pseudoKey()
    using TimerValue = CounterPerPhaseTypeRawCommunicationSumUpValue;
    // TimerValue must contain functions setType() and setPseudoKeyCounter
    //
    dss_schimek::mpi::environment env;

    struct OutputFormat {
        std::string prefix;
        std::string phase;
        size_t counterPerPhase;
        size_t round;
        std::string description;
        std::string type;
        bool rawCommunication;
        bool sumUp;
        size_t value;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormat const& outputFormat) {
            return stream << outputFormat.prefix << " phase=" << outputFormat.phase
                          << " counterPerPhase=" << outputFormat.counterPerPhase
                          << " round=" << outputFormat.round
                          << " operation=" << outputFormat.description
                          << " type=" << outputFormat.type
                          << " rawCommunication=" << outputFormat.rawCommunication
                          << " sumUp=" << outputFormat.sumUp << " value=" << outputFormat.value;
        }
    };

public:
    static MeasuringTool& measuringTool() {
        static MeasuringTool measuringTool;
        return measuringTool;
    }

    void reset() {
        timer = Timer<TimerKey, TimerValue>();
        nonTimer = NonTimer<NonTimerRecord>();
        leanNonTimer = LeanNonTimer<PhaseValue>();
        disabled = false;
        verbose = false;
        prefix = "";
        curPhase = "none";
        curRound = 0;
    }

    void add(size_t value) {
        // if (disabled) return;
        // add(value, "unkown");
    }

    void add(size_t value, std::string const& description, bool const sumUp = true) {
        // if (disabled) return;
        // if (verbose && env.rank() == 0) std::cout << description << std::endl;
        // nonTimer.add(NonTimerRecord(curPhase, 0u, curRound, description,
        //     "number", false, sumUp, value));
    }

    void addRawCommunication(size_t value, std::string const& description) {
        // if (disabled) return;
        if (verbose && env.rank() == 0)
            std::cout << description << std::endl;
        leanNonTimer.add(PhaseValue(curPhase, value));
    }

    void start(std::string const& description) {
        if (disabled)
            return;
        if (verbose && env.rank() == 0)
            std::cout << description << std::endl;
        timer.start(
            TimerKey(curPhase, curRound, description),
            TimerValue(0u, "", false, false, 0u)
        );
    }

    void stop(std::string const& description) {
        if (disabled)
            return;
        if (verbose && env.rank() == 0)
            std::cout << description << std::endl;
        timer.stop(TimerKey(curPhase, curRound, description));
    }

    std::vector<OutputFormat> collect() {
        disable();
        const size_t numMeasurements = 1000;

        std::vector<OutputFormat> data;
        data.reserve(10 * numMeasurements);
        std::vector<NonTimerRecord> nonTimerRecords;
        std::vector<PhaseValue> leanNonTimerRecords;
        std::vector<std::pair<TimerKey, TimerValue>> timerRecords;
        nonTimerRecords.reserve(numMeasurements);
        leanNonTimerRecords.reserve(numMeasurements);
        timerRecords.reserve(6 * numMeasurements);

        nonTimer.collect(std::back_inserter(nonTimerRecords));
        leanNonTimer.collect(std::back_inserter(leanNonTimerRecords));
        timer.collect(std::back_inserter(timerRecords));

        for (auto const& nonTimerRecord: nonTimerRecords)
            data.push_back(
                {prefix,
                 nonTimerRecord.phase,
                 nonTimerRecord.counterPerPhase,
                 nonTimerRecord.round,
                 nonTimerRecord.description,
                 nonTimerRecord.type,
                 nonTimerRecord.rawCommunication,
                 nonTimerRecord.sumUp,
                 nonTimerRecord.value}
            );

        for (auto const& leanNonTimerRecord: leanNonTimerRecords)
            data.push_back(
                {prefix,
                 leanNonTimerRecord.phase,
                 0,
                 0,
                 "commVolume",
                 "number",
                 true,
                 false,
                 leanNonTimerRecord.value}
            );

        for (auto const& [timerKey, timerValue]: timerRecords)
            data.push_back(
                {prefix,
                 timerKey.phase,
                 timerValue.counterPerPhase,
                 timerKey.round,
                 timerKey.description,
                 timerValue.type,
                 timerValue.rawCommunication,
                 timerValue.sumUp,
                 timerValue.value}
            );
        enable();
        return data;
    }

    void writeToStream(std::ostream& stream) {
        for (auto const& data: collect())
            stream << data << std::endl;
    }
    void disableBarrier(bool value) {
        timer.setDisableBarrier(value);
    }

    void enable() {
        disabled = false;
    }

    void disable() {
        disabled = true;
    }

    void setVerbose(bool const value) {
        verbose = value;
    }

    void setPrefix(std::string const& prefix_) {
        prefix = prefix_;
    }

    void setPhase(std::string const& phase) {
        curPhase = phase;
    }

    void setRound(size_t round) {
        curRound = round;
    }

private:
    bool disabled = false;
    bool verbose = false;
    std::string prefix = "";
    std::string curPhase = "none";
    size_t curRound = 0;
    NonTimer<NonTimerRecord> nonTimer;
    LeanNonTimer<PhaseValue> leanNonTimer;
    Timer<TimerKey, TimerValue> timer;
};
} // namespace measurement
} // namespace dss_schimek
