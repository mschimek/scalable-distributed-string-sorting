// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
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

    size_t num_groups() const { return comm_exchange.size() / comm_group.size(); }
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

    NaiveSplit(auto first_level, auto last_level, Communicator const& root)
        : communicators_(create_communicators(first_level, last_level, root)) {}

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

    static std::vector<Communicator>
    create_communicators(auto first_level, auto last_level, Communicator const& root) {
        std::vector<Communicator> comms;
        comms.reserve(std::distance(first_level, last_level) + 1);

        // todo could use ranges here
        Communicator comm{root};
        for (auto level = first_level; level != last_level; ++level) {
            auto group_size = *level;
            tlx_die_unless(comm.size() % group_size == 0);
            tlx_die_unless(1 < group_size && group_size < comm.size());

            int color = static_cast<int>(comm.rank() / group_size);
            comms.push_back(std::exchange(comm, comm.split(color)));
        }
        comms.push_back(std::move(comm));

        return comms;
    }
};


        return send_counts;
    }
};

} // namespace multi_level
} // namespace dss_mehnert
