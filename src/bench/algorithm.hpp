// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The interface distributed_sorter drives one measured run through.

#pragma once

#include "executables/args.hpp"
#include "mpi/communicator.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace bench {

class AbstractAlgorithm {
public:
    AbstractAlgorithm() = default;
    AbstractAlgorithm(AbstractAlgorithm const&) = delete;
    AbstractAlgorithm(AbstractAlgorithm&&) = delete;
    AbstractAlgorithm& operator=(AbstractAlgorithm const&) = delete;
    AbstractAlgorithm& operator=(AbstractAlgorithm&&) = delete;
    virtual ~AbstractAlgorithm() = default;

    // (re)create the input for one measured run, snapshot it for the checkers.
    virtual void prepare() = 0;

    // the measured sort; the algorithm owns its MeasuringTool and kamping timer phases
    virtual void run() = 0;

    // --check-sorted / --check-complete / --count-prefixes / --print-sorted
    virtual void verify() = 0;

    // emit this run's measurements
    virtual void report() = 0;
};

// The state and the reporting every algorithm shares.
class AlgorithmBase : public AbstractAlgorithm {
public:
    AlgorithmBase(SorterArgs const& args, Communicator const& comm) : args_{args}, comm_{comm} {}

    void report() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        measuring_tool.write_on_root(*args_.measurement_output, comm_);
        measuring_tool.reset();
    }

protected:
    SorterArgs const& args_;
    Communicator const& comm_;
};

} // namespace bench
} // namespace dss_mehnert
