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

    typename StringSet::CharIterator first_chars(size_t const depth) const {
        assert(!empty());
        return active_.get_chars(first_string(), depth);
    }

    typename StringSet::String first_string() const {
        assert(!empty());
        return active_[active_.begin()];
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

    auto d_str = dest.active().begin();
    auto d_lcp = dest.lcp();

    LcpType curr_lcp = 0;

    if (!defender.empty() && !contender.empty()) {
        auto defender_chars = defender.first_chars(0);
        auto contender_chars = contender.first_chars(0);

        curr_lcp = calc_lcp(defender_chars, contender_chars);
        if (defender_chars[curr_lcp] > contender_chars[curr_lcp]) {
            defender.swap(contender);
        }

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
            auto defender_chars = defender.first_chars(curr_lcp);
            auto contender_chars = contender.first_chars(curr_lcp);

            auto const old_lcp = curr_lcp;
            while (*defender_chars != 0 && *defender_chars == *contender_chars) {
                ++defender_chars, ++contender_chars, ++curr_lcp;
            }

            if (*defender_chars < *contender_chars) {
                *d_str++ = defender.first_string();
                *d_lcp++ = defender.first_lcp();
            } else {
                *d_str++ = contender.first_string();
                *d_lcp++ = old_lcp;
                defender.swap(contender);
            }
        }

        assert_equal(calc_lcp(dest.active(), *(d_str - 2), *(d_str - 1)), *(d_lcp - 1));
    }

    // copy remaining strings
    d_str = std::copy(defender.active().begin(), defender.active().end(), d_str);
    d_str = std::copy(contender.active().begin(), contender.active().end(), d_str);
    d_lcp = std::copy_n(defender.lcp(), defender.size(), d_lcp);
    d_lcp = std::copy_n(contender.lcp(), contender.size(), d_lcp);

    assert_equal(d_str, dest.active().end());
    assert_equal(std::distance(dest.lcp(), d_lcp), std::ssize(dest));
    assert(dest.active().check_order());
}

} // namespace merge
} // namespace dss_mehnert
