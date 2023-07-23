// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <kamping/collectives/allreduce.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>

#include "mpi/communicator.hpp"
#include "util/measurements.hpp"

namespace dss_mehnert {
namespace measurement {


template <typename Key, typename Value>
class Timer {
public:
    struct OutputFormat {
        Key key;
        Value value;
        uint64_t min_time, max_time, avg_time;
        uint64_t min_loss, max_loss, avg_loss;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormat const& output) {
            return stream << Result << output.key << " " << Result << output.value
                          << " min_time=" << output.min_time << " max_time=" << output.max_time
                          << " avg_time=" << output.avg_time << " min_loss=" << output.min_loss
                          << " max_loss=" << output.max_loss << " avg_loss=" << output.avg_loss;
        }
    };

    Timer() = default;

    void start(Key const& key, Value const& value) {
        if (!measurementEnabled)
            return;
        if (keyToValue.contains(key)) {
            std::cout << keyToValue.find(key)->first;
            std::cout << "Key: " << key << " already added" << std::endl;
            std::abort();
        }
        auto const& itToValue = keyToValue.emplace(key, value);
        itToValue.first->second.setPseudoKeyCounter(incrementCounterPerPseudoKey(key));
        auto const& [itToStart, _] = keyToStart.emplace(key, PointInTime());

        // Start measurement
        itToStart->second = Clock::now();
    }

    void stop(Key const& key, Communicator const& comm) {
        if (!measurementEnabled) {
            return;
        }

        const PointInTime before_barrier = Clock::now();
        if (!disableBarrier) {
            comm.barrier();
        }
        const PointInTime after_barrier = Clock::now();

        // check whether key is present
        auto it = keyToStart.find(key);
        if (it == keyToStart.end()) {
            std::cout << "Key: " << key << " has no corresponding start" << std::endl;
            std::abort();
        }
        const PointInTime start_time = it->second;

        TimeIntervalDataType elapsed_active_time =
            std::chrono::duration_cast<TimeUnit>(before_barrier - start_time).count();
        TimeIntervalDataType elapsed_total_time =
            std::chrono::duration_cast<TimeUnit>(after_barrier - start_time).count();

        active_time_.emplace(key, elapsed_active_time);
        total_time_.emplace(key, elapsed_total_time);
    }

    std::vector<OutputFormat> collect(Communicator const& comm) {
        using namespace kamping;

        std::vector<OutputFormat> output;
        output.reserve(keyToValue.size());
        for (auto& [key, value]: keyToValue) {
            auto time = expect_key(active_time_, key)->second;
            // todo loss is currently alwas 0
            auto loss = getLoss(key);
            auto size = comm.size();

            output.push_back({
                .key = key,
                .value = value,
                .min_time = comm.allreduce_single(send_buf(time), op(ops::min<>{})),
                .max_time = comm.allreduce_single(send_buf(time), op(ops::max<>{})),
                .avg_time = comm.allreduce_single(send_buf(time), op(ops::plus<>{})) / size,
                .min_loss = comm.allreduce_single(send_buf(loss), op(ops::min<>{})),
                .max_loss = comm.allreduce_single(send_buf(loss), op(ops::max<>{})),
                .avg_loss = comm.allreduce_single(send_buf(loss), op(ops::plus<>{})) / size,
            });
        }
        return output;
    }

    void setDisableBarrier(bool value) { disableBarrier = value; }

private:
    using Clock = std::chrono::high_resolution_clock;
    using PointInTime = std::chrono::time_point<Clock>;
    using TimeUnit = std::chrono::nanoseconds;
    using TimeIntervalDataType = uint64_t;
    using PseudoKey = typename Key::PseudoKey;

    bool measurementEnabled = true;
    bool disableBarrier = false;

    size_t incrementCounterPerPseudoKey(Key const& key) {
        auto itCounter = pseudoKeyToCounter.find(key.pseudoKey());
        if (itCounter == pseudoKeyToCounter.end()) {
            pseudoKeyToCounter.emplace(key.pseudoKey(), 1u);
            return 0;
        } else {
            return itCounter->second++;
        }
    }

    TimeIntervalDataType getLoss(Key const& key) {
        auto active_time = expect_key(active_time_, key);
        auto total_time = expect_key(total_time_, key);
        return total_time->second - active_time->second;
    }

    auto expect_key(auto const& map, auto const& key) {
        auto iter = map.find(key);
        if (iter == std::end(map)) {
            std::cout << "Key " << key << " not present" << std::endl;
            std::abort();
        }
        return iter;
    }

    std::map<Key, Value> keyToValue;
    std::map<PseudoKey, size_t> pseudoKeyToCounter;
    std::map<Key, PointInTime> keyToStart;
    std::map<Key, TimeIntervalDataType> active_time_;
    std::map<Key, TimeIntervalDataType> total_time_;
};

} // namespace measurement
} // namespace dss_mehnert
