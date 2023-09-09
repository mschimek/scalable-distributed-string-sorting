// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstddef>

#include "strings/stringtools.hpp"

namespace dss_mehnert {
namespace _internal {

template <typename StringPtr, typename CharCompare>
typename StringPtr::StringSet::Iterator lcp_bound_impl(
    StringPtr const& strptr, typename StringPtr::StringSet::String const& value, CharCompare comp
) {
    static_assert(StringPtr::with_lcp);
    assert(strptr.active().check_order());

    auto const ss = strptr.active();
    auto const begin = ss.begin(), end = ss.end();
    if (begin == end) {
        return end;
    }

    auto const value_chars = value.getChars();
    auto const first_chars = ss.get_chars(ss[begin], 0);

    auto curr_lcp = dss_schimek::calc_lcp(value_chars, first_chars);
    if (comp(value_chars[curr_lcp], first_chars[curr_lcp])) {
        return begin;
    }

    size_t i = 1;
    for (auto it = begin + 1; it != end; ++it, ++i) {
        if (curr_lcp > strptr.get_lcp(i)) {
            return it;
        } else if (curr_lcp == strptr.get_lcp(i)) {
            auto lhs = value_chars + curr_lcp;
            auto rhs = ss.get_chars(ss[it], curr_lcp);

            while (*lhs != 0 && *lhs == *rhs) {
                ++lhs, ++rhs, ++curr_lcp;
            }

            if (comp(*lhs, *rhs)) {
                return it;
            }
        } else {
            // nothing to do
        }
    }
    return end;
}

} // namespace _internal

// todo return lcp value
template <typename StringPtr>
typename StringPtr::StringSet::Iterator
lcp_lower_bound(StringPtr const& strptr, typename StringPtr::StringSet::String const& value) {
    return _internal::lcp_bound_impl(strptr, value, std::less_equal<>{});
}

template <typename StringPtr>
typename StringPtr::StringSet::Iterator
lcp_upper_bound(StringPtr const& strptr, typename StringPtr::StringSet::String const& value) {
    return _internal::lcp_bound_impl(strptr, value, std::less<>{});
}

} // namespace dss_mehnert
