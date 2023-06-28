// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "mpi/allreduce.hpp"

namespace dss_schimek {

template <typename Key, typename Value>
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    using PointInTime = std::chrono::time_point<Clock>;
    using TimeUnit = std::chrono::nanoseconds;
    using TimeIntervalDataType = size_t;
    using PseudoKey = typename Key::PseudoKey;

public:
    Timer() = default;

    void start(Key const& key, Value const& value) {
        if (!measurementEnabled)
            return;
        if (keyToValue.find(key) != keyToValue.end()) {
            std::cout << "Key: " << key << " already added" << std::endl;
            std::abort();
        }
        auto const& itToValue = keyToValue.emplace(key, value);
        itToValue.first->second.setPseudoKeyCounter(incrementCounterPerPseudoKey(key));
        auto const& [itToStart, blabla] = keyToStart.emplace(key, PointInTime());

        // Start measurement
        const PointInTime start = Clock::now();
        itToStart->second = start;
    }

    void stop(Key const& key) {
        if (!measurementEnabled)
            return;

        const PointInTime endPoint = Clock::now();
        // if (!disableBarrier)
        // env.barrier();
        const PointInTime endPointAfterBarrier = Clock::now();
        // measurement stopped

        // check whether key is present
        auto it = keyToStart.find(key);
        if (it == keyToStart.end()) {
            std::cout << "Key: " << key << " has no corresponding start" << std::endl;
            std::abort();
        }
        const PointInTime startPoint = it->second;

        TimeIntervalDataType elapsedActiveTime =
            std::chrono::duration_cast<std::chrono::nanoseconds>(endPoint - startPoint).count();
        TimeIntervalDataType elapsedTotalTime =
            std::chrono::duration_cast<TimeUnit>(endPointAfterBarrier - startPoint).count();

        keyToActiveTime.emplace(key, elapsedActiveTime);
        keyToTotalTime.emplace(key, elapsedTotalTime);
    }

    template <typename OutputIterator>
    void collect(OutputIterator out) {
        for (auto& [key, value]: keyToValue) {
            value.setType("avgTime");
            value.setValue(avgTime(key));
            out = std::make_pair(key, value);

            value.setType("maxTime");
            value.setValue(maxTime(key));
            out = std::make_pair(key, value);

            value.setType("minTime");
            value.setValue(minTime(key));
            out = std::make_pair(key, value);

            value.setType("avgLoss");
            value.setValue(avgLoss(key));
            out = std::make_pair(key, value);

            value.setType("maxLoss");
            value.setValue(maxLoss(key));
            out = std::make_pair(key, value);

            value.setType("minLoss");
            value.setValue(minLoss(key));
            out = std::make_pair(key, value);
        }
    }

    void setDisableBarrier(bool value) {
        disableBarrier = value;
    }

private:
    bool measurementEnabled = true;
    bool disableBarrier = false;
    dss_schimek::mpi::environment env;

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
        auto itKeyActiveTime = keyToActiveTime.find(key);
        auto itKeyTotalTime = keyToTotalTime.find(key);
        if (itKeyActiveTime == keyToActiveTime.end() || itKeyTotalTime == keyToTotalTime.end()) {
            std::cout << key << std::endl;
            std::cout << "Key " << key << " not present" << std::endl;
            std::abort();
        }
        return (*itKeyTotalTime).second - (*itKeyActiveTime).second;
    }

    TimeIntervalDataType
    avgLoss(Key const& key, dss_schimek::mpi::environment env = dss_schimek::mpi::environment()) {
        TimeIntervalDataType localLoss = getLoss(key);
        TimeIntervalDataType sum = dss_schimek::mpi::allreduce_sum(localLoss);
        return sum / env.size();
    }

    TimeIntervalDataType maxLoss(Key const& key) {
        TimeIntervalDataType localLoss = getLoss(key);
        return dss_schimek::mpi::allreduce_max(localLoss);
    }

    TimeIntervalDataType minLoss(Key const& key) {
        TimeIntervalDataType localLoss = getLoss(key);
        return dss_schimek::mpi::allreduce_min(localLoss);
    }

    TimeIntervalDataType avgTime(Key const& key) {
        auto itKeyTime = keyToActiveTime.find(key);
        if (itKeyTime == keyToActiveTime.end()) {
            std::cout << "Key " << key << "not present" << std::endl;
            std::abort();
        }
        TimeIntervalDataType sum = dss_schimek::mpi::allreduce_sum((*itKeyTime).second);
        return sum / env.size();
    }

    TimeIntervalDataType maxTime(Key const& key) {
        auto itKeyTime = keyToActiveTime.find(key);
        if (itKeyTime == keyToActiveTime.end()) {
            std::cout << key << std::endl;
            std::cout << "Key " << key << " not present" << std::endl;
            std::abort();
        }
        return dss_schimek::mpi::allreduce_max((*itKeyTime).second);
    }

    TimeIntervalDataType minTime(Key const& key) {
        auto itKeyTime = keyToActiveTime.find(key);
        if (itKeyTime == keyToActiveTime.end()) {
            std::cout << key << std::endl;
            std::cout << "Key " << key << " not present" << std::endl;
            std::abort();
        }
        return dss_schimek::mpi::allreduce_min((*itKeyTime).second);
    }

    std::map<Key, Value> keyToValue;
    std::map<PseudoKey, size_t> pseudoKeyToCounter;
    std::map<Key, PointInTime> keyToStart;
    std::map<Key, TimeIntervalDataType> keyToActiveTime;
    std::map<Key, TimeIntervalDataType> keyToTotalTime;
};

} // namespace dss_schimek
