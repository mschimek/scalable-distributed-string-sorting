// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>

namespace dss_mehnert {
namespace measurement {

struct ostream_wrapper {
    std::ostream& stream;
};

static constexpr struct result_ {
} Result;

inline ostream_wrapper operator<<(std::ostream& stream, result_) { return {stream}; }

struct PhaseValue {
    using PseudoKey = std::string;

    std::string phase;
    std::size_t value;

    std::string const& pseudoKey() const { return phase; }

    void setValue(std::size_t value_) { value = value_; }

    std::size_t getValue() const { return value; }

    friend std::ostream& operator<<(ostream_wrapper out, PhaseValue const& data) {
        return out.stream << "phase=" << data.phase << " value=" << data.value;
    }

    friend std::ostream& operator<<(std::ostream& stream, PhaseValue const& record) {
        return stream << "{" << Result << record << "}";
    }
};

struct PhaseRoundDescription {
    using PseudoKey = std::string;

    std::string phase;
    std::size_t round;
    std::string description;

    std::string const& pseudoKey() const { return phase; }

    friend auto operator<=>(PhaseRoundDescription const& lhs, PhaseRoundDescription const& rhs) {
        return std::tie(lhs.phase, lhs.round, lhs.description)
               <=> std::tie(rhs.phase, rhs.round, rhs.description);
    }

    friend std::ostream& operator<<(ostream_wrapper out, PhaseRoundDescription const& data) {
        return out.stream << "phase=" << data.phase << " round=" << data.round
                          << " key=" << data.description;
    }

    friend std::ostream& operator<<(std::ostream& stream, PhaseRoundDescription const& record) {
        return stream << "{" << Result << record << "}";
    }
};

struct PhaseCounterRoundDescription {
    using PseudoKey = std::string;

    std::string phase;
    std::size_t counterPerPhase;
    std::size_t round;
    std::string description;

    std::string const& pseudoKey() const { return phase; }

    void setPseudoKeyCounter(std::size_t counter) { counterPerPhase = counter; }

    friend std::ostream&
    operator<<(ostream_wrapper out, PhaseCounterRoundDescription const& record) {
        return out.stream << "phase=" << record.phase
                          << " counter_per_phase=" << record.counterPerPhase
                          << " round=" << record.round << " key=" << record.description;
    }

    friend std::ostream&
    operator<<(std::ostream& stream, PhaseCounterRoundDescription const& record) {
        return stream << "{" << Result << record << "}";
    }
};

struct CounterPerPhase {
    std::size_t counterPerPhase;

    void setPseudoKeyCounter(std::size_t counter) { counterPerPhase = counter; }

    friend std::ostream& operator<<(ostream_wrapper out, CounterPerPhase const& record) {
        return out.stream << "counter_per_phase=" << record.counterPerPhase;
    }

    friend std::ostream& operator<<(std::ostream& stream, CounterPerPhase const& record) {
        return stream << "{" << Result << record << "}";
    }
};

struct SimpleValue {
    std::size_t value;

    size_t getValue() const { return value; }

    void setValue(std::size_t value_) { value = value_; }

    friend std::ostream& operator<<(ostream_wrapper out, SimpleValue const& record) {
        return out.stream << "value=" << record.value;
    }

    friend std::ostream& operator<<(std::ostream& stream, SimpleValue const& record) {
        return stream << "{" << Result << record << "}";
    }
};

} // namespace measurement
} // namespace dss_mehnert
