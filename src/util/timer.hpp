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

#include <kamping/collectives/gather.hpp>
#include <kamping/environment.hpp>
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
    class Clock {
    public:
        using rep = double;
        using period = std::ratio<1>;
        using duration = std::chrono::duration<rep, period>;
        using time_point = std::chrono::time_point<Clock>;

        static constexpr bool is_steady = true;

        static time_point now() { return time_point{duration{kamping::Environment<>::wtime()}}; }
    };

    using PointInTime = std::chrono::time_point<Clock>;
    using TimeUnit = std::chrono::nanoseconds;
    using TimeIntervalDataType = uint64_t;
    using PseudoKey = typename Key::PseudoKey;

    struct OutputFormat {
        Key key;
        Value value;
        Summary<size_t> summary_time;
        Summary<size_t> summary_loss;

        friend std::ostream& operator<<(std::ostream& stream, OutputFormat const& output) {
            return stream << Result << output.key << " " << Result << output.value
                          << " min_time=" << output.summary_time.min
                          << " max_time=" << output.summary_time.max
                          << " avg_time=" << output.summary_time.avg
                          << " sum_time=" << output.summary_time.sum
                          << " min_loss=" << output.summary_loss.min
                          << " max_loss=" << output.summary_loss.max
                          << " avg_loss=" << output.summary_loss.avg
                          << " sum_loss=" << output.summary_loss.sum;
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

        PointInTime before_barrier = Clock::now();
        if (barrier) {
            comm.barrier();
        }
        PointInTime after_barrier = Clock::now();

        // check whether key is present
        auto it = key_start_.find(key);
        if (it == key_start_.end()) {
            std::cout << "Key: " << key << " has no corresponding start" << std::endl;
            std::abort();
        }
        PointInTime start_time = it->second;

        using std::chrono::duration_cast;
        auto elapsed_active_time = duration_cast<TimeUnit>(before_barrier - start_time);
        auto elapsed_total_time = duration_cast<TimeUnit>(after_barrier - start_time);

        active_time_.emplace(key, elapsed_active_time.count());
        total_time_.emplace(key, elapsed_total_time.count());
    }

    std::vector<OutputFormat> collect_on_root(Communicator const& comm) {
        using namespace kamping;

        std::vector<OutputFormat> output;
        output.reserve(key_value_.size());
        for (auto& [key, value]: key_value_) {
            auto time = expect_key(active_time_, key)->second;
            auto loss = get_loss(key);

            std::vector<size_t> recv_time, recv_loss;
            comm.gather(send_buf(time), root(comm.root()), recv_buf(recv_time));
            comm.gather(send_buf(loss), root(comm.root()), recv_buf(recv_loss));

            if (comm.is_root()) {
                auto summary_time = describe<size_t>(recv_time.begin(), recv_time.end());
                auto summary_loss = describe<size_t>(recv_loss.begin(), recv_loss.end());
                output.emplace_back(key, value, summary_time, summary_loss);
            }
        }
        return output;
    }

private:
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
