// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include <tlx/die.hpp>

namespace dss_mehnert {
namespace multi_level {

template <typename Communicator>
struct Level {
    Communicator const& comm_orig;
    Communicator const& comm_exchange;
    Communicator const& comm_group;

    size_t num_groups() const { return comm_orig.size() / comm_group.size(); }
    size_t group_size() const { return comm_group.size(); }
};

template <typename Impl, typename Communicator>
struct LevelIter {
    using iterator = LevelIter<Impl, Communicator>;

    Impl const& impl;
    size_t level;

    iterator& operator++() {
        ++level;
        return *this;
    }

    Level<Communicator> operator*() const { return impl.level(level); }

    friend void swap(iterator& lhs, iterator& rhs) {
        std::swap(lhs.impl, rhs.impl);
        std::swap(lhs.level, rhs.level);
    }

    bool operator==(iterator const& rhs) { return level == rhs.level; }
    bool operator!=(iterator const& rhs) { return level != rhs.level; }
};

template <typename Communicator>
class NoSplit {
public:
    using iterator = LevelIter<NoSplit<Communicator>, Communicator>;

    NoSplit(Communicator const& comm) : comm_(comm) {}

    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, 0}; }

    Level<Communicator> level(size_t level) const { tlx_die("not implemented"); };

    static std::vector<size_t>
    send_counts(std::vector<size_t> const& interval_sizes, Level<Communicator> const& level) {
        tlx_die("not implemented");
    };

    Communicator const& comm_root() const { return comm_; }
    Communicator const& comm_final() const { return comm_; }

private:
    Communicator comm_;
};

template <typename Communicator>
class NaiveSplit {
public:
    using iterator = LevelIter<NaiveSplit<Communicator>, Communicator>;

    NaiveSplit(auto first_level, auto last_level, Communicator const& root) {
        communicators_.reserve(std::distance(first_level, last_level) + 1);

        // todo could use ranges here
        Communicator comm{root};
        for (auto level = first_level; level != last_level; ++level) {
            auto group_size = *level;
            tlx_die_unless(comm.size() % group_size == 0);
            tlx_die_unless(1 < group_size && group_size < comm.size());

            int color = static_cast<int>(comm.rank() / group_size);
            communicators_.push_back(std::exchange(comm, comm.split(color)));
        }
        communicators_.push_back(std::move(comm));
    }

    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, communicators_.size() - 1}; }

    Communicator const& comm_root() const { return communicators_.front(); }
    Communicator const& comm_final() const { return communicators_.back(); }

    Level<Communicator> level(size_t level) const {
        return {communicators_[level], communicators_[level], communicators_[level + 1]};
    }

    static std::vector<size_t>
    send_counts(std::vector<size_t> const& interval_sizes, Level<Communicator> const& level) {
        auto const& comm = level.comm_orig;
        auto const group_size = level.group_size();
        auto const num_groups{comm.size() / group_size};
        auto const group_offset{comm.rank() / num_groups};

        // PE `i` sends strings to the `i * p' / p`th member of each group on the next level
        std::vector<size_t> send_counts(comm.size());
        auto dst_iter = send_counts.begin() + group_offset;
        for (auto const interval: interval_sizes) {
            *dst_iter = interval;
            dst_iter += group_size;
        }

        return send_counts;
    }

private:
    std::vector<Communicator> communicators_;
};


template <typename Communicator>
class GridwiseSplit {
public:
    using iterator = LevelIter<GridwiseSplit<Communicator>, Communicator>;

    GridwiseSplit(auto first_level, auto last_level, Communicator const& root) {
        comms_row_.reserve(std::distance(first_level, last_level) + 1);
        comms_col_.reserve(std::distance(first_level, last_level));

        Communicator comm{root};
        for (auto level = first_level; level != last_level; ++level) {
            auto group_size = *level;
            tlx_die_unless(comm.size() % group_size == 0);
            tlx_die_unless(1 < group_size && group_size < comm.size());

            // do NOT change the order of these two splits!
            int col = static_cast<int>(comm.rank() % group_size);
            comms_col_.push_back(comm.split(col));

            // todo again, could use range based split
            int row = static_cast<int>(comm.rank() / group_size);
            comms_row_.push_back(std::exchange(comm, comm.split(row)));
        }
        comms_row_.push_back(std::move(comm));
    }

    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, comms_col_.size()}; }

    Communicator const& comm_root() const { return comms_row_.front(); }
    Communicator const& comm_final() const { return comms_row_.back(); }

    Level<Communicator> level(size_t level) const {
        return {comms_row_[level], comms_col_[level], comms_row_[level + 1]};
    }

    static std::vector<size_t>
    send_counts(std::vector<size_t> const& interval_sizes, Level<Communicator> const& level) {
        // nothing to do here, intervals are same shape as column communicator
        tlx_assert_equal(interval_sizes.size(), level.comm_exchange.size());
        return interval_sizes;
    }

private:
    std::vector<Communicator> comms_row_;
    std::vector<Communicator> comms_col_;
};

} // namespace multi_level
} // namespace dss_mehnert
