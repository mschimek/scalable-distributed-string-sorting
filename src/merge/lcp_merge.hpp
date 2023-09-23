// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>

#include "strings/stringtools.hpp"
#include "tlx/die.hpp"

namespace dss_mehnert {
namespace merge {

namespace _internal {

template <typename StringSet, typename LcpType>
class MergeAdapter {
public:
    MergeAdapter() = default;

    MergeAdapter(StringSet const& ss, LcpType* lcp_begin) : active_(ss), lcp_(lcp_begin) {}

    StringSet const& active() const { return active_; }
    LcpType* lcp() const { return lcp_; }
    size_t size() const { return active_.size(); }
    bool empty() const { return active_.empty(); }

    MergeAdapter<StringSet, LcpType>& operator++() {
        assert(!empty());
        active_ = active_.sub(active_.begin() + 1, active_.end());
        ++lcp_;
        return *this;
    }

    typename StringSet::String first_string() const {
        assert(!empty());
        return active_[active_.begin()];
    }

    typename StringSet::CharIterator first_chars(size_t const depth) const {
        assert(!empty());
        return active_.get_chars(first_string(), depth);
    }

    LcpType first_lcp() const {
        assert(!empty());
        return *lcp_;
    }

    void swap(MergeAdapter<StringSet, LcpType>& rhs) {
        using std::swap;
        swap(active_, rhs.active_);
        swap(lcp_, rhs.lcp_);
    }

private:
    StringSet active_;
    LcpType* lcp_;
};

}; // namespace _internal

template <typename StringLcpPtr>
void lcp_merge(StringLcpPtr const& lhs, StringLcpPtr const& rhs, StringLcpPtr const& dest) {
    static_assert(StringLcpPtr::with_lcp, "LCP merge requries an LCP array");
    assert_equal(lhs.size() + rhs.size(), dest.size());
    assert(lhs.active().check_order());
    assert(rhs.active().check_order());

    using StringSet = typename StringLcpPtr::StringSet;
    using LcpType = typename StringLcpPtr::LcpType;
    using dss_schimek::calc_lcp;

    _internal::MergeAdapter<StringSet, LcpType> defender{lhs.active(), lhs.lcp()};
    _internal::MergeAdapter<StringSet, LcpType> contender{rhs.active(), rhs.lcp()};

    auto d_ss = dest.active();
    auto d_str = d_ss.begin();
    auto d_lcp = dest.lcp();

    LcpType curr_lcp = 0;

    if (!defender.empty() && !contender.empty()) {
        auto const& defender_str = defender.first_string();
        auto const& contender_str = contender.first_string();

        curr_lcp = calc_lcp(d_ss, defender_str, contender_str);

        auto const defender_char = defender.first_chars(curr_lcp);
        auto const contender_char = contender.first_chars(curr_lcp);
        if (d_ss.is_less(contender_str, contender_char, defender_str, defender_char)) {
            defender.swap(contender);
        }
    }
    if (!defender.empty()) {
        *d_str++ = defender.first_string();
        *d_lcp++ = 0;
        ++defender;
    }

    for (; !defender.empty() && !contender.empty(); ++defender) {
        assert_unequal(d_str, dest.active().end());

        if (defender.first_lcp() < curr_lcp) {
            *d_str++ = contender.first_string();
            *d_lcp++ = curr_lcp;
            curr_lcp = defender.first_lcp();
            defender.swap(contender);

        } else if (defender.first_lcp() > curr_lcp) {
            *d_str++ = defender.first_string();
            *d_lcp++ = defender.first_lcp();

        } else {
            auto const& defender_str = defender.first_string();
            auto const& contender_str = contender.first_string();
            auto defender_chars = defender.first_chars(curr_lcp);
            auto contender_chars = contender.first_chars(curr_lcp);

            auto const old_lcp = curr_lcp;
            while (d_ss.is_equal(defender_str, defender_chars, contender_str, contender_chars)) {
                ++defender_chars, ++contender_chars, ++curr_lcp;
            }

            if (d_ss.is_less(defender_str, defender_chars, contender_str, contender_chars)) {
                *d_str++ = defender.first_string();
                *d_lcp++ = defender.first_lcp();
            } else {
                *d_str++ = contender.first_string();
                *d_lcp++ = old_lcp;
                defender.swap(contender);
            }
        }

        assert_equal(calc_lcp(d_ss, d_ss[d_str - 2], d_ss[d_str - 1]), *(d_lcp - 1));
    }

    // copy remaining strings and LCP values
    if (!defender.empty()) {
        d_str = std::copy(defender.active().begin(), defender.active().end(), d_str);
        d_lcp = std::copy_n(defender.lcp(), defender.size(), d_lcp);
    } else if (!contender.empty()) {
        d_str = std::copy(contender.active().begin(), contender.active().end(), d_str);

        // special case for first contender string
        *d_lcp++ = curr_lcp;
        d_lcp = std::copy_n(contender.lcp() + 1, contender.size() - 1, d_lcp);
    }

    assert_equal(d_str, dest.active().end());
    assert_equal(std::distance(dest.lcp(), d_lcp), std::ssize(dest));
    assert(d_ss.check_order());
}

} // namespace merge
} // namespace dss_mehnert
