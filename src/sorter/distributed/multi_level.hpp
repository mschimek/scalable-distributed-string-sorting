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
#include <kamping/communicator.hpp>
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
struct RowCommunicators {
    std::vector<Communicator> comms;

    template <typename LevelIt>
    RowCommunicators(LevelIt first_level, LevelIt last_level, Communicator const& root) {
        comms.reserve(std::distance(first_level, last_level) + 1);

        Communicator comm{root};
        for (auto level = first_level; level != last_level; ++level) {
            auto group_size = kamping::asserting_cast<int>(*level);
            tlx_die_unless(comm.size_signed() % group_size == 0);
            tlx_die_unless(1 < group_size && group_size < comm.size_signed());

            int first = (comm.rank_signed() / group_size) * group_size;
            int last = first + group_size - 1;
            kamping::RankRange range{first, last, 1};

            auto comm_group = comm.create_subcommunicators({std::array{range}});
            comms.emplace_back(std::exchange(comm, std::move(comm_group)));
        }
        comms.emplace_back(std::move(comm));
    }
};

template <typename Communicator>
struct ColumnCommunicators {
    std::vector<Communicator> comms;

    ColumnCommunicators(RowCommunicators<Communicator> const& rows_) {
        assert(!rows_.comms.empty());

        auto const& rows = rows_.comms;
        comms.reserve(rows.size() - 1);

        for (auto curr = rows.begin(), next = curr + 1; next != rows.end(); ++curr, ++next) {
            auto group_size = kamping::asserting_cast<int>(next->size());

            int col_first = curr->rank_signed() % group_size;
            int col_last = curr->size_signed() - group_size + col_first;
            kamping::RankRange col_range{col_first, col_last, group_size};

            auto comm = curr->create_subcommunicators({std::array{col_range}});
            comms.emplace_back(std::move(comm));
        }
    }
};

template <typename Communicator_>
class NoSplit {
public:
    using Communicator = Communicator_;
    using iterator = LevelIter<NoSplit<Communicator>, Communicator>;

    static constexpr bool is_single_level = true;

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

    Communicator const& comm_root() const { return comm_; }
    Communicator const& comm_final() const { return comm_; }

private:
    Communicator comm_;
};

template <typename Communicator_>
class RowwiseSplit {
public:
    using Communicator = Communicator_;
    using iterator = LevelIter<RowwiseSplit<Communicator>, Communicator>;

    static constexpr bool is_single_level = false;

    static constexpr std::string_view get_name() { return "naive_split"; }

    RowwiseSplit(auto first_level, auto last_level, Communicator const& root)
        : rows_{first_level, last_level, root} {}

    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, rows_.comms.size() - 1}; }

    Communicator const& comm_root() const { return rows_.comms.front(); }
    Communicator const& comm_final() const { return rows_.comms.back(); }
    RowCommunicators<Communicator> const& comms_row() const { return rows_; };

    Level<Communicator> level(size_t level) const {
        return {rows_.comms[level], rows_.comms[level], rows_.comms[level + 1]};
    }

private:
    RowCommunicators<Communicator> rows_;
};


template <typename Communicator_>
class GridwiseSplit {
public:
    using Communicator = Communicator_;
    using iterator = LevelIter<GridwiseSplit<Communicator>, Communicator>;

    static constexpr bool is_single_level = false;
    static constexpr std::string_view get_name() { return "grid_split"; }

    GridwiseSplit(auto first_level, auto last_level, Communicator const& root)
        : rows_{first_level, last_level, root},
          cols_{rows_} {}

    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, cols_.comms.size()}; }

    Communicator const& comm_root() const { return rows_.comms.front(); }
    Communicator const& comm_final() const { return rows_.comms.back(); }
    RowCommunicators<Communicator> const& comms_row() const { return rows_; }
    ColumnCommunicators<Communicator> const& comms_col() const { return cols_; }

    Level<Communicator> level(size_t level) const {
        return {rows_.comms[level], cols_.comms[level], rows_.comms[level + 1]};
    }

private:
    RowCommunicators<Communicator> rows_;
    ColumnCommunicators<Communicator> cols_;
};


template <typename Communicator>
struct GridCommunicators {
    std::vector<Communicator> comms;

    explicit GridCommunicators(NoSplit<Communicator> const& no_split)
        : comms{no_split.comm_final()} {}

    explicit GridCommunicators(RowwiseSplit<Communicator> const& naive_split)
        : comms{naive_split.comms_row().comms} {
        comms.emplace_back(naive_split.comm_final());
    }

    explicit GridCommunicators(GridwiseSplit<Communicator> const& grid_split)
        : comms{grid_split.comms_col().comms} {
        comms.emplace_back(grid_split.comm_final());
    }
};

} // namespace multi_level
} // namespace dss_mehnert
