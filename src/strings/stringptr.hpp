/*******************************************************************************
 * src/tools/stringptr.hpp
 *
 * StringPtr, StringPtrOut, and NoLcpCalc specializations. Encapsulate string
 * and shadow array pointers.
 *
 * Additionally: LcpStringPtr encapsulates string and lcp arrays, which may be
 * interleaved or separate.
 *
 * StringLcpPtr               -> (string,lcp,size)
 * StringLcpCachePtr          -> (string,lcp,charcache,size)
 *
 * StringShadowPtr            -> (string,shadow,size,flip)
 * StringShadowOutPtr         -> (string,shadow,output,size,flip)
 * StringShadowLcpPtr         -> (string,shadow=lcp,size,flip)
 * StringShadowLcpOutPtr      -> (string,shadow=lcp,output,size,flip)
 * StringShadowLcpCacheOutPtr -> (string,shadow=lcp,charcache,output,size,flip)
 *
 *******************************************************************************
 * Copyright (C) 2013-2014 Timo Bingmann <tb@panthema.net>
 * Copyright (C) 2013-2014 Andreas Eberle <email@andreas-eberle.com>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>

#include <numa.h>
#include <stdint.h>
#include <tlx/logger.hpp>

#include "stringset.hpp"

namespace dss_schimek {

using LcpType = size_t;

//! Objectified string array pointer and shadow pointer array for out-of-place
//! swapping of pointers.
template <typename _StringSet>
class StringPtr {
public:
    typedef _StringSet StringSet;
    typedef typename StringSet::String String;

protected:
    //! strings (front) and temporary shadow (back) array
    StringSet active_;

public:
    //! constructor specifying all attributes
    StringPtr(StringSet const& ss) : active_(ss) {}

    //! return currently active array
    StringSet const& active() const { return active_; }

    //! return valid length
    size_t size() const { return active_.size(); }

    //! Advance (both) pointers by given offset, return sub-array
    StringPtr sub(size_t offset, size_t _size) const {
        assert(offset + _size <= size());
        return StringPtr(active_.subi(offset, offset + _size));
    }

    //! check sorted order of strings
    bool check() const {
        assert(output().check_order());
        return true;
    }

    //! Return i-th string pointer from active_
    String& str(size_t i) const {
        assert(i < size());
        return active_.at(i);
    }

    //! if we want to save the LCPs
    static bool const with_lcp = false;

    //! return reference to the i-th lcp
    LcpType& lcp(size_t /* i */) const { abort(); }

    //! set the i-th lcp to v and check its value
    void set_lcp(size_t /* i */, LcpType const& /* v */) const {}

    //! Fill whole LCP array with n times the value v, ! excluding the first
    //! LCP[0] position
    void fill_lcp(LcpType /* v */) {}

    //! set the i-th distinguishing cache charater to c

    //! Return pointer to LCP array
    LcpType* lcparray() const { abort(); }
};

template <typename _StringSet>
class StringLcpPtrMergeAdapter {
public:
    typedef _StringSet StringSet;
    typedef typename StringSet::String String;
    using Iterator = typename StringSet::Iterator;
    using CharIt = typename StringSet::CharIterator;

public:
    StringSet active_;
    LcpType* lcp_;
    size_t offset;
    Iterator strings_;

public:
    StringLcpPtrMergeAdapter()
        : active_(StringSet(nullptr, nullptr)),
          lcp_(nullptr),
          offset(0),
          strings_(nullptr) {}
    //! constructor specifying all attributes
    StringLcpPtrMergeAdapter(StringSet const& ss, LcpType* lcp_begin)
        : active_(ss),
          lcp_(lcp_begin),
          offset(0),
          strings_(ss.begin()) {}

    //! return currently active array
    StringSet const& active() const { return active_; }

    ////! return valid length
    size_t size() const { return active_.size() - offset; }

    ////! Advance (both) pointers by given offset, return sub-array
    StringLcpPtrMergeAdapter sub(size_t _offset, size_t _size) const {
        return StringLcpPtrMergeAdapter(
            active_.subi(offset + _offset, offset + _offset + _size),
            lcp_ + _offset + offset
        );
    }

    ////! check sorted order of strings
    // bool check() const
    //{
    //   assert(output().check_order());
    //   return true;
    // }

    ////! Return i-th string pointer from active_
    // String & str(size_t i) const
    //{
    //   assert(i < size());
    //   return active_.at(i);
    // }

    ////! return reference to the i-th lcp
    // LcpType& lcp(size_t i) const
    //{
    //   return lcp_[i];
    // }

    ////! set the i-th lcp to v and check its value
    // void set_lcp(size_t i, const LcpType& v) const
    //{
    //   assert(i < size());
    //   if (i >= size())
    //     std::cout << " lcp out of bounds " << std::endl;
    //   lcp_[i] = v;
    // }

    ////! Fill whole LCP array with n times the value v, ! excluding the first
    ////! LCP[0] position
    // void fill_lcp(LcpType v)
    //{
    //   std::fill_n(lcp_ + 1, size() - 1, v);
    // }

    // LcpType* lcp_array() const
    //{
    //   return lcp_;
    // }

    bool empty() const { return (active().size() - offset <= 0); }

    void setFirst(String str, LcpType lcp) {
        *(strings_ + offset) = str;
        *(lcp_ + offset) = lcp;
    }

    String& firstString() const { return *(strings_ + offset); }

    CharIt firstStringChars() const { return active().get_chars(*(strings_ + offset), 0); }

    LcpType& firstLcp() const { return *(lcp_ + offset); }

    StringLcpPtrMergeAdapter& operator++() {
        ++offset;
        return *this;
    }

    bool operator<(StringLcpPtrMergeAdapter const& rhs) const {
        return strings_ + offset < rhs.strings_ + rhs.offset;
    }
};

template <typename _StringSet>
class StringLcpPtr {
public:
    typedef _StringSet StringSet;
    typedef typename StringSet::String String;

protected:
    StringSet active_;
    LcpType* lcp_;

public:
    //! constructor specifying all attributes
    StringLcpPtr(StringSet const& ss, LcpType* lcp_begin) : active_(ss), lcp_(lcp_begin) {}

    //! return currently active array
    StringSet const& active() const { return active_; }

    //! return valid length
    size_t size() const { return active_.size(); }

    //! Advance (both) pointers by given offset, return sub-array
    StringLcpPtr sub(size_t offset, size_t _size) const {
        assert(offset + _size <= size());
        return StringLcpPtr(active_.subi(offset, offset + _size), lcp_ + offset);
    }

    //! check sorted order of strings
    bool check() const {
        assert(output().check_order());
        return true;
    }

    //! Return i-th string pointer from active_
    String& str(size_t i) const {
        assert(i < size());
        return active_.at(i);
    }

    //! return reference to the i-th lcp
    LcpType& lcp(size_t i) const { return lcp_[i]; }

    //! set the i-th lcp to v and check its value
    void set_lcp(size_t i, LcpType const& v) const {
        assert(i < size());
        if (i >= size())
            std::cout << " lcp out of bounds " << std::endl;
        lcp_[i] = v;
    }

    //! Fill whole LCP array with n times the value v, ! excluding the first
    //! LCP[0] position
    void fill_lcp(LcpType v) { std::fill_n(lcp_ + 1, size() - 1, v); }

    LcpType* lcp_array() const { return lcp_; }
};
/******************************************************************************/

//! Objectified string array pointer and shadow pointer array for out-of-place
//! swapping of pointers.
template <typename _StringSet>
class StringShadowPtr {
public:
    typedef _StringSet StringSet;
    typedef typename StringSet::String String;

protected:
    //! strings (front) and temporary shadow (back) array
    StringSet active_, shadow_;

    //! false if active_ is original, true if shadow_ is original
    bool flipped_;

public:
    //! constructor specifying all attributes
    StringShadowPtr(StringSet const& original, StringSet const& shadow, bool flipped = false)
        : active_(original),
          shadow_(shadow),
          flipped_(flipped) {}

    //! true if flipped to back array
    bool flipped() const { return flipped_; }

    //! return currently active array
    StringSet const& active() const { return active_; }

    //! return current shadow array
    StringSet const& shadow() const { return shadow_; }

    //! return valid length
    size_t size() const { return active_.size(); }

    //! Advance (both) pointers by given offset, return sub-array
    StringShadowPtr sub(size_t offset, size_t _size) const {
        assert(offset + _size <= size());
        return StringShadowPtr(
            active_.subi(offset, offset + _size),
            shadow_.subi(offset, offset + _size),
            flipped_
        );
    }

    //! construct a StringShadowPtr object specifying a sub-array with flipping
    //! to other array.
    StringShadowPtr flip(size_t offset, size_t _size) const {
        assert(offset + _size <= size());
        return StringShadowPtr(
            shadow_.subi(offset, offset + _size),
            active_.subi(offset, offset + _size),
            !flipped_
        );
    }

    //! construct a StringShadowPtr object specifying a sub-array with flipping
    //! to other array.
    StringShadowPtr flip() const { return StringShadowPtr(shadow_, active_, !flipped_); }

    //! Return the original for this StringShadowPtr for LCP calculation
    StringShadowPtr original() const { return flipped_ ? flip() : *this; }

    //! return subarray pointer to n strings in original array, might copy from
    //! shadow before returning.
    StringShadowPtr copy_back() const {
        if (!flipped_) {
            return *this;
        } else {
            std::move(active_.begin(), active_.end(), shadow_.begin());
            return flip();
        }
    }

    //! check sorted order of strings
    bool check() const { return true; }

    //! Return i-th string pointer from active_
    String& str(size_t i) const {
        assert(i < size());
        return active_.at(i);
    }

    //! return reference to the i-th lcp
    LcpType& lcp(size_t /* i */) const {
        static LcpType default_lcp_value = 0;
        return default_lcp_value;
    }

    //! set the i-th lcp to v and check its value
    void set_lcp(size_t /* i */, LcpType const& /* v */) const {}

    //! Fill whole LCP array with n times the value v, ! excluding the first
    //! LCP[0] position
    void fill_lcp(LcpType /* v */) {}
};

//! Objectified string array pointer and shadow pointer array for out-of-place
//! swapping of pointers.
template <typename _StringSet>
class StringShadowLcpPtr {
public:
    typedef _StringSet StringSet;
    typedef typename StringSet::String String;

protected:
    //! strings (front) and temporary shadow (back) array
    StringSet active_, shadow_;
    LcpType* lcp_;

    //! false if active_ is original, true if shadow_ is original
    bool flipped_;

public:
    //! constructor specifying all attributes
    StringShadowLcpPtr(
        StringSet const& original, StringSet const& shadow, LcpType* lcp, bool flipped = false
    )
        : active_(original),
          shadow_(shadow),
          lcp_(lcp),
          flipped_(flipped) {}

    //! true if flipped to back array
    bool flipped() const { return flipped_; }

    //! return currently active array
    StringSet const& active() const { return active_; }

    //! return current shadow array
    StringSet const& shadow() const { return shadow_; }

    //! return valid length
    size_t size() const { return active_.size(); }

    //! Advance (both) pointers by given offset, return sub-array
    StringShadowLcpPtr sub(size_t offset, size_t _size) const {
        assert(offset + _size <= size());
        return StringShadowLcpPtr(
            active_.subi(offset, offset + _size),
            shadow_.subi(offset, offset + _size),
            lcp_ + offset,
            flipped_
        );
    }

    //! construct a StringShadowPtr object specifying a sub-array with flipping
    //! to other array.
    StringShadowLcpPtr flip(size_t offset, size_t _size) const {
        assert(offset + _size <= size());
        return StringShadowLcpPtr(
            shadow_.subi(offset, offset + _size),
            active_.subi(offset, offset + _size),
            lcp_ + offset,
            !flipped_
        );
    }

    //! construct a StringShadowPtr object specifying a sub-array with flipping
    //! to other array.
    StringShadowLcpPtr flip() const {
        return StringShadowLcpPtr(shadow_, active_, lcp_, !flipped_);
    }

    //! Return the original for this StringShadowPtr for LCP calculation
    StringShadowLcpPtr original() const { return flipped_ ? flip() : *this; }

    //! return subarray pointer to n strings in original array, might copy from
    //! shadow before returning.
    StringShadowLcpPtr copy_back() const {
        if (!flipped_) {
            return *this;
        } else {
            std::move(active_.begin(), active_.end(), shadow_.begin());
            return flip();
        }
    }

    //! check sorted order of strings
    bool check() const { return true; }

    //! Return i-th string pointer from active_
    String& str(size_t i) const {
        assert(i < size());
        return active_.at(i);
    }

    //! return reference to the i-th lcp
    LcpType& lcp(size_t i) const { return lcp_[i]; }

    //! set the i-th lcp to v and check its value
    void set_lcp(size_t i, LcpType const& v) const { lcp_[i] = v; }

    //! Fill whole LCP array with n times the value v, ! excluding the first
    //! LCP[0] position
    void fill_lcp(LcpType v) { std::fill_n(lcp_ + 1, size() - 1, v); }
    LcpType* lcp_array() const { return lcp_; }
};
} // namespace dss_schimek
/******************************************************************************/
