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
        if (!enabled)
            return;
        if (key_value_.contains(key)) {
            std::cout << key_value_.find(key)->first;
            std::cout << "Key: " << key << " already added" << std::endl;
            std::abort();
        }
        auto const& itToValue = key_value_.emplace(key, value);
        itToValue.first->second.setPseudoKeyCounter(incrementCounterPerPseudoKey(key));
        auto const& [itToStart, _] = key_start_.emplace(key, PointInTime());

        // Start measurement
        itToStart->second = Clock::now();
    }

    void stop(Key const& key, bool barrier = false, Communicator const& comm = {}) {
        if (!enabled) {
            return;
        }

        const PointInTime before_barrier = Clock::now();
        if (barrier) {
            comm.barrier();
        }
        const PointInTime after_barrier = Clock::now();

        // check whether key is present
        auto it = key_start_.find(key);
        if (it == key_start_.end()) {
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
        output.reserve(key_value_.size());
        for (auto& [key, value]: key_value_) {
            auto time = expect_key(active_time_, key)->second;
            // todo loss is currently alwas 0
            auto loss = get_loss(key);
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

private:
    using Clock = std::chrono::high_resolution_clock;
    using PointInTime = std::chrono::time_point<Clock>;
    using TimeUnit = std::chrono::nanoseconds;
    using TimeIntervalDataType = uint64_t;
    using PseudoKey = typename Key::PseudoKey;

    static constexpr bool enabled = true;

    size_t incrementCounterPerPseudoKey(Key const& key) {
        auto itCounter = key_counter_.find(key.pseudoKey());
        if (itCounter == key_counter_.end()) {
            key_counter_.emplace(key.pseudoKey(), 1u);
            return 0;
        } else {
            return itCounter->second++;
        }
    }

    TimeIntervalDataType get_loss(Key const& key) {
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

    std::map<Key, Value> key_value_;
    std::map<PseudoKey, size_t> key_counter_;
    std::map<Key, PointInTime> key_start_;
    std::map<Key, TimeIntervalDataType> active_time_;
    std::map<Key, TimeIntervalDataType> total_time_;
};

} // namespace measurement
} // namespace dss_mehnert
