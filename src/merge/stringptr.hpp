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

#include <cassert>

#include <numa.h>
#include <stdint.h>

#include "merge/stringtools.hpp"

namespace stringtools {

typedef uintptr_t lcp_t;

/******************************************************************************/

class LcpCacheStringPtr;

class LcpStringPtr {
public:
    string* strings;
    lcp_t* lcps;
    size_t size;

public:
    LcpStringPtr() : strings(NULL), lcps(NULL), size(0) {}

    LcpStringPtr(string* _strings, lcp_t* _lcps, size_t _size)
        : strings(_strings),
          lcps(_lcps),
          size(_size) {}

    bool empty() const { return (size == 0); }

    void setFirst(string s, lcp_t lcp) const {
        assert(size > 0);
        *strings = s;
        *lcps = lcp;
    }

    void setFirst(LcpStringPtr& ptr) {
        *strings = *ptr.strings;
        *lcps = *ptr.lcps;
    }

    string& firstString() const {
        assert(size > 0);
        return *strings;
    }

    lcp_t& firstLcp() const {
        assert(size > 0);
        return *lcps;
    }

    void copyFrom(LcpStringPtr& other, size_t length) const {
        memcpy(strings, other.strings, length * sizeof(string));
        memcpy(lcps, other.lcps, length * sizeof(lcp_t));
    }

    void copyStringsTo(string* destination, size_t length) const {
        memcpy(destination, strings, length * sizeof(string));
    }

    void setLcp(size_t position, lcp_t value) const {
        assert(position < size);
        lcps[position] = value;
    }

    // preincrement
    LcpStringPtr& operator++() {
        ++strings;
        ++lcps;
        --size;
        return *this;
    }

    //! return sub-array of (string,lcp) with offset and size
    LcpStringPtr sub(size_t offset, size_t n) const {
        assert(offset + n <= size);
        return LcpStringPtr(strings + offset, lcps + offset, n);
    }

    //! return empty end array.
    LcpStringPtr end() const { return sub(size, 0); }

    size_t operator-(LcpStringPtr const& rhs) const { return strings - rhs.strings; }

    bool operator<(LcpStringPtr const& rhs) const { return strings < rhs.strings; }
};

} // namespace stringtools

/******************************************************************************/
