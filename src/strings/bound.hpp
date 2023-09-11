// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstddef>

#include "strings/stringtools.hpp"
#include "tlx/die.hpp"

namespace dss_mehnert {

namespace _internal {

template <typename StringPtr>
using IteratorLcpPair =
    std::pair<typename StringPtr::StringSet::Iterator, typename StringPtr::LcpType>;

// note this relies heavily on null terminated strings
template <typename StringPtr, typename CharCompare>
IteratorLcpPair<StringPtr> lcp_bound_impl(
    StringPtr const& strptr,
    typename StringPtr::StringSet::String const& value,
    CharCompare comp,
    typename StringPtr::LcpType const known_prefix
) {
    static_assert(StringPtr::with_lcp);
    assert(strptr.active().check_order());

    using dss_schimek::calc_lcp;
    auto const ss = strptr.active();
    auto const begin = ss.begin(), end = ss.end();
    if (begin == end) {
        return {end, 0};
    }

    assert(known_prefix <= calc_lcp(ss, value, ss[begin]));
    auto const value_chars = ss.get_chars(value, 0);
    auto const first_chars = ss.get_chars(ss[begin], 0);

    auto curr_lcp = known_prefix + calc_lcp(value_chars + known_prefix, first_chars + known_prefix);
    if (comp(value_chars[curr_lcp], first_chars[curr_lcp])) {
        return {begin, curr_lcp};
    }

    size_t i = 1;
    for (auto it = begin + 1; it != end; ++it, ++i) {
        if (curr_lcp > strptr.get_lcp(i)) {
            assert_equal(calc_lcp(ss, ss[it], value), strptr.get_lcp(i));
            return {it, strptr.get_lcp(i)};
        } else if (curr_lcp == strptr.get_lcp(i)) {
            auto lhs = value_chars + curr_lcp;
            auto rhs = ss.get_chars(ss[it], curr_lcp);

            while (*lhs != 0 && *lhs == *rhs) {
                ++lhs, ++rhs, ++curr_lcp;
            }

            if (comp(*lhs, *rhs)) {
                assert_equal(calc_lcp(ss, ss[it], value), curr_lcp);
                return {it, curr_lcp};
            }
        } else {
            // nothing to do
        }
    }
    return {end, 0};
}

} // namespace _internal

template <typename StringPtr>
_internal::IteratorLcpPair<StringPtr> lcp_lower_bound(
    StringPtr const& strptr,
    typename StringPtr::StringSet::String const& value,
    typename StringPtr::LcpType const known_prefix = 0
) {
    return _internal::lcp_bound_impl(strptr, value, std::less_equal<>{}, known_prefix);
}

template <typename StringPtr>
_internal::IteratorLcpPair<StringPtr> lcp_upper_bound(
    StringPtr const& strptr,
    typename StringPtr::StringSet::String const& value,
    typename StringPtr::LcpType const known_prefix = 0
) {
    return _internal::lcp_bound_impl(strptr, value, std::less<>{}, known_prefix);
}

} // namespace dss_mehnert
