// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include <bits/iterator_concepts.h>
#include <kamping/checking_casts.hpp>
#include <kamping/rank_ranges.hpp>
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
    using iterator_category = std::forward_iterator_tag;
    using value_type = Level<Communicator>;
    using difference_type = std::ptrdiff_t;
    using pointer = Level<Communicator>*;
    using reference = Level<Communicator>;

    LevelIter() = default;

    LevelIter(Impl const& impl, size_t level) : impl_(&impl), level_(level) {}

    iterator& operator++() {
        ++level_;
        return *this;
    }

    iterator operator++(int) {
        iterator tmp{*this};
        ++level_;
        return tmp;
    }

    Level<Communicator> operator*() const { return impl_->level(level_); }

    friend void swap(iterator& lhs, iterator& rhs) {
        std::swap(lhs.impl_, rhs.impl_);
        std::swap(lhs.level_, rhs.level_);
    }
    friend bool operator==(iterator const& lhs, iterator const& rhs) {
        return lhs.level_ == rhs.level_;
    }
    friend bool operator!=(iterator const& lhs, iterator const& rhs) {
        return lhs.level_ != rhs.level_;
    }

private:
    Impl const* impl_;
    size_t level_;
};

template <typename Communicator>
class NoSplit {
public:
    using iterator = LevelIter<NoSplit<Communicator>, Communicator>;

    static constexpr std::string_view get_name() { return "no_split"; }

    NoSplit(auto first_level, auto last_level, Communicator const& comm) : comm_(comm) {
        tlx_die_verbose_unless(
            first_level == last_level,
            "you probably meant to use multi-level merge sort"
        );
    }

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

    static constexpr std::string_view get_name() { return "naive_split"; }

    NaiveSplit(auto first_level, auto last_level, Communicator const& root) {
        communicators_.reserve(std::distance(first_level, last_level) + 1);

        Communicator comm{root};
        for (auto level = first_level; level != last_level; ++level) {
            auto group_size = kamping::asserting_cast<int>(*level);
            tlx_die_unless(comm.size_signed() % group_size == 0);
            tlx_die_unless(1 < group_size && group_size < comm.size_signed());

            int first = (comm.rank_signed() / group_size) * group_size;
            int last = first + group_size - 1;
            kamping::RankRange range{first, last, 1};

            auto comm_group = comm.create_subcommunicators({std::array{range}});
            communicators_.push_back(std::exchange(comm, std::move(comm_group)));
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
        auto const num_groups = comm.size() / group_size;
        auto const group_offset = comm.rank() / num_groups;

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

    static constexpr std::string_view get_name() { return "grid_split"; }

    GridwiseSplit(auto first_level, auto last_level, Communicator const& root) {
        comms_row_.reserve(std::distance(first_level, last_level) + 1);
        comms_col_.reserve(std::distance(first_level, last_level));

        Communicator comm{root};
        for (auto level = first_level; level != last_level; ++level) {
            auto group_size = kamping::asserting_cast<int>(*level);
            tlx_die_unless(comm.size_signed() % group_size == 0);
            tlx_die_unless(1 < group_size && group_size < comm.size_signed());

            int col_first = comm.rank_signed() % group_size;
            int col_last = comm.size_signed() - group_size + col_first;
            kamping::RankRange col_range{col_first, col_last, group_size};

            auto comm_col = comm.create_subcommunicators({std::array{col_range}});
            comms_col_.push_back(std::move(comm_col));

            int row_first = (comm.rank_signed() / group_size) * group_size;
            int row_last = row_first + group_size - 1;
            kamping::RankRange row_range{row_first, row_last, 1};

            auto comm_row = comm.create_subcommunicators({std::array{row_range}});
            comms_row_.push_back(std::exchange(comm, std::move(comm_row)));
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
