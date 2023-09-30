/*******************************************************************************
 * src/tools/stringset.hpp
 *
 * Implementations of StringSet concept: UCharStringSet, VectorStringSet,
 * StringSuffixSet.
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
 * Copyright (C) 2015 Timo Bingmann <tb@panthema.net>
 * Copyright (C) 2019 Matthias Schimek
 * Copyright (C) 2023 Pascal Mehnert
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
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include <stdint.h>
#include <tlx/logger.hpp>

namespace dss_schimek {

/******************************************************************************/
// CharIterator -> character group functions

template <typename CharIterator>
inline uint32_t get_char_uint32_bswap32(CharIterator str, size_t depth) {
    uint32_t v = __builtin_bswap32(*(uint32_t*)(str + depth));
    if ((v & 0xFF000000LU) == 0)
        return 0;
    else if ((v & 0x00FF0000LU) == 0)
        return (v & 0xFFFF0000LU);
    else if ((v & 0x0000FF00LU) == 0)
        return (v & 0xFFFFFF00LU);
    return v;
}

template <typename CharIterator>
inline uint64_t get_char_uint64_bswap64(CharIterator str, size_t depth) {
    uint64_t v = __builtin_bswap64(*(uint64_t*)(str + depth));
    if ((v & 0xFF00000000000000LLU) == 0)
        return 0;
    else if ((v & 0x00FF000000000000LLU) == 0)
        return (v & 0xFFFF000000000000LLU);
    else if ((v & 0x0000FF0000000000LLU) == 0)
        return (v & 0xFFFFFF0000000000LLU);
    else if ((v & 0x000000FF00000000LLU) == 0)
        return (v & 0xFFFFFFFF00000000LLU);
    else if ((v & 0x00000000FF000000LLU) == 0)
        return (v & 0xFFFFFFFFFF000000LLU);
    else if ((v & 0x0000000000FF0000LLU) == 0)
        return (v & 0xFFFFFFFFFFFF0000LLU);
    else if ((v & 0x000000000000FF00LLU) == 0)
        return (v & 0xFFFFFFFFFFFFFF00LLU);
    return v;
}

/******************************************************************************/

/*!
 * Base class for common string set functions, included via CRTP.
 */
template <typename StringSet, typename Traits>
class StringSetBase {
public:
    //! index-based array access (readable and writable) to String objects.
    typename Traits::String& at(size_t i) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return *(ss.begin() + i);
    }

    //! \name CharIterator Comparisons
    //! \{

    //! check equality of two strings a and b at char iterators ai and bi.
    bool is_equal(
        typename Traits::String const& a,
        typename Traits::CharIterator const& ai,
        typename Traits::String const& b,
        typename Traits::CharIterator const& bi
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return !ss.is_end(a, ai) && !ss.is_end(b, bi) && (*ai == *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool is_less(
        typename Traits::String const& a,
        typename Traits::CharIterator const& ai,
        typename Traits::String const& b,
        typename Traits::CharIterator const& bi
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.is_end(a, ai) || (!ss.is_end(a, ai) && !ss.is_end(b, bi) && *ai < *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool is_leq(
        typename Traits::String const& a,
        typename Traits::CharIterator const& ai,
        typename Traits::String const& b,
        typename Traits::CharIterator const& bi
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.is_end(a, ai) || (!ss.is_end(a, ai) && !ss.is_end(b, bi) && *ai <= *bi);
    }

    //! \}

    size_t get_sum_length() const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        auto acc = [](auto const& n, auto const& str) { return n + str.getLength(); };
        return std::accumulate(ss.begin(), ss.end(), size_t{0}, acc);
    }

    //! \name Character Extractors
    //! \{

    typename Traits::Char get_char(typename Traits::String const& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return *ss.get_chars(s, depth);
    }

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t
    get_char_uint8_simple(typename Traits::String const& s, typename Traits::CharIterator i) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);

        if (ss.is_end(s, i))
            return 0;
        return uint8_t(*i);
    }

    //! Return up to 2 characters of string s at iterator i packed into a uint16
    //! (only works correctly for 8-bit characters)
    uint16_t get_char_uint16_simple(
        typename Traits::String const& s, typename Traits::CharIterator i
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);

        uint16_t v = 0;
        if (ss.is_end(s, i))
            return v;
        v = (uint16_t(*i) << 8);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint16_t(*i) << 0);
        return v;
    }

    //! Return up to 4 characters of string s at iterator i packed into a uint32
    //! (only works correctly for 8-bit characters)
    uint32_t get_char_uint32_simple(
        typename Traits::String const& s, typename Traits::CharIterator i
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);

        uint32_t v = 0;
        if (ss.is_end(s, i))
            return v;
        v = (uint32_t(*i) << 24);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint32_t(*i) << 16);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint32_t(*i) << 8);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint32_t(*i) << 0);
        return v;
    }

    //! Return up to 8 characters of string s at iterator i packed into a uint64
    //! (only works correctly for 8-bit characters)
    uint64_t get_char_uint64_simple(
        typename Traits::String const& s, typename Traits::CharIterator i
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);

        uint64_t v = 0;
        if (ss.is_end(s, i))
            return v;
        v = (uint64_t(*i) << 56);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint64_t(*i) << 48);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint64_t(*i) << 40);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint64_t(*i) << 32);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint64_t(*i) << 24);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint64_t(*i) << 16);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint64_t(*i) << 8);
        ++i;
        if (ss.is_end(s, i))
            return v;
        v |= (uint64_t(*i) << 0);
        return v;
    }

    uint8_t get_uint8(typename Traits::String const& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint8_simple(s, ss.get_chars(s, depth));
    }

    uint16_t get_uint16(typename Traits::String const& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint16_simple(s, ss.get_chars(s, depth));
    }

    uint32_t get_uint32(typename Traits::String const& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint32_simple(s, ss.get_chars(s, depth));
    }

    uint64_t get_uint64(typename Traits::String const& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint64_simple(s, ss.get_chars(s, depth));
    }

    //! \}

    //! Subset this string set using begin and size range.
    StringSet subr(typename Traits::Iterator const& begin, size_t size) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.sub(begin, begin + size);
    }

    //! Subset this string set using index range.
    StringSet subi(size_t begin, size_t end) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.sub(ss.begin() + begin, ss.begin() + end);
    }

    bool check_order(typename Traits::String const& s1, typename Traits::String const& s2) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);

        typename StringSet::CharIterator c1 = ss.get_chars(s1, 0);
        typename StringSet::CharIterator c2 = ss.get_chars(s2, 0);

        while (ss.is_equal(s1, c1, s2, c2))
            ++c1, ++c2;

        if (!ss.is_leq(s1, c1, s2, c2))
            return false;

        return true;
    }

    bool check_order() const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        if (ss.size() == 0)
            return true;

        for (typename Traits::Iterator pi = ss.begin() + 1; pi != ss.end(); ++pi) {
            if (!check_order(ss[pi - 1], ss[pi]))
                return false;
        }

        return true;
    }
};

/*----------------------------------------------------------------------------*/

template <typename Type>
struct StringSetGetKeyHelper {
    template <typename StringSet>
    static Type get_key(StringSet const& ss, typename StringSet::String const& s, size_t depth);
};

template <>
struct StringSetGetKeyHelper<uint8_t> {
    template <typename StringSet>
    static uint8_t get_key(StringSet const& ss, typename StringSet::String const& s, size_t depth) {
        return ss.get_uint8(s, depth);
    }
};

template <>
struct StringSetGetKeyHelper<uint16_t> {
    template <typename StringSet>
    static uint16_t
    get_key(StringSet const& ss, typename StringSet::String const& s, size_t depth) {
        return ss.get_uint16(s, depth);
    }
};

template <>
struct StringSetGetKeyHelper<uint32_t> {
    template <typename StringSet>
    static uint32_t
    get_key(StringSet const& ss, typename StringSet::String const& s, size_t depth) {
        return ss.get_uint32(s, depth);
    }
};

template <>
struct StringSetGetKeyHelper<uint64_t> {
    template <typename StringSet>
    static uint64_t
    get_key(StringSet const& ss, typename StringSet::String const& s, size_t depth) {
        return ss.get_uint64(s, depth);
    }
};

template <typename Type, typename StringSet>
Type get_key(StringSet const& ss, typename StringSet::String const& s, size_t depth) {
    return StringSetGetKeyHelper<Type>::get_key(ss, s, depth);
}

/******************************************************************************/

struct Length {
    using underlying_t = size_t;

    static constexpr std::string_view name{"length"};

    size_t length;
    size_t value() const { return length; }
    inline size_t getLength() const { return length; }
};

struct Index {
    using underlying_t = uint64_t;

    static constexpr std::string_view name{"index"};

    uint64_t index;
    uint64_t value() const { return index; }
    inline uint64_t getIndex() const { return index; }
};

struct PEIndex {
    using underlying_t = size_t;

    static constexpr std::string_view name{"rank"};

    size_t PEIndex;
    size_t value() const { return PEIndex; }
    inline size_t getPEIndex() const { return PEIndex; }
};

struct StringIndex {
    using underlying_t = size_t;

    static constexpr std::string_view name{"str_index"};

    size_t stringIndex;
    size_t value() const { return stringIndex; }
    inline size_t getStringIndex() const { return stringIndex; }
};

template <typename Char_, typename String, typename... Members>
struct StringData : public Members... {
    using Char = Char_;

    template <typename T>
    static constexpr bool has_member = (std::is_same_v<T, Members> || ...);

    template <typename... NewMembers>
    using with_members_t = StringData<Char, String, Members..., NewMembers...>;

    StringData() = default;

    template <typename... Init>
    StringData(String string, Init... args) noexcept : Init{args}...,
                                                       string{string} {}

    String string;

    inline void setChars(String string_) { string = string_; }

    inline String getChars() const { return string; }

    template <typename... NewMembers>
    inline with_members_t<NewMembers...> with_members(NewMembers&&... args) const noexcept {
        return {
            this->string,
            static_cast<Members const&>(*this)...,
            std::forward<NewMembers>(args)...};
    }
};

template <typename Data, typename T>
inline constexpr bool has_member = Data::template has_member<T>;

template <typename Char, typename String, typename... Args>
std::ostream& operator<<(std::ostream& out, StringData<Char, String, Args...> const& str) {
    out << "{";
    ((out << Args::name << "=" << static_cast<Args>(str).value() << ", "), ...);
    return out << "}";
}

/******************************************************************************/

/*!
 * Traits class implementing StringSet concept for char* and unsigned char*
 * strings with additional length attribute.
 */
template <typename String_, template <typename> typename StringSet_>
class GenericStringSetTraits {
public:
    //! exported alias for character type
    using Char = String_::Char;

    //! String reference: pointer to first character
    using String = String_;

    //! Iterator over string references: pointer over pointers
    using Iterator = String*;

    //! iterator of characters in a string
    using CharIterator = Char const*;

    template <typename String>
    using StringSet = StringSet_<String>;

    static constexpr bool is_indexed = has_member<String, Index>;
    static constexpr bool has_length = has_member<String, Length>;
};

/*!
 * Class implementing StringSet concept for char* and unsigned char* strings
 * with additional length attribute.
 */
template <typename String_>
class GenericStringSet : public GenericStringSetTraits<String_, GenericStringSet>,
                         public StringSetBase<
                             GenericStringSet<String_>,
                             GenericStringSetTraits<String_, GenericStringSet>> {
public:
    using Traits = GenericStringSetTraits<String_, GenericStringSet>;

    using Char = Traits::Char;
    using String = Traits::String;
    using Iterator = Traits::Iterator;
    using CharIterator = Traits::CharIterator;

    static constexpr bool is_compressed = false;

    //! Construct from begin and end string pointers
    GenericStringSet() = default;

    //! Construct from begin and end string pointers
    GenericStringSet(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }

    //! Check if the set contains no elements
    bool empty() const { return begin_ == end_; }

    //! Iterator representing first String position
    Iterator begin() const { return begin_; }

    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Iterator-based array access (readable and writable) to String objects.
    String& operator[](Iterator i) const { return *i; }

    //! Return CharIterator for referenced string, which belong to this set.
    CharIterator get_chars(String const& s, size_t depth) const { return s.string + depth; }

    //! Returns true if CharIterator is at end of the given String
    bool is_end(String const&, CharIterator const& i) const { return (*i == 0); }

    //! Return complete string (for debugging purposes)
    std::string get_string(String const& s, size_t const depth = 0) const {
        return {reinterpret_cast<char const*>(s.string) + depth};
    }

    // todo is this actually returning This?
    //! Subset this string set using iterator range.
    GenericStringSet sub(Iterator const begin, Iterator const end) const { return {begin, end}; }

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t get_char_uint8_simple(String const&, CharIterator i) const { return uint8_t(*i); }

    //! \}

    void print(std::string_view prefix = "") const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << prefix << "[" << i++ << "] = " << (*pi) << " = " << get_string(*pi, 0);
        }
    }

    static String empty_string() {
        static Char zero_char = 0;
        return {&zero_char};
    }

    // todo move
    size_t get_length(String const& str) const {
        if constexpr (has_member<String, Length>) {
            return str.length;
        } else {
            auto begin = get_chars(str, 0), end = begin;
            while (!is_end(str, end)) {
                ++end;
            }
            return end - begin;
        }
    }

private:
    //! array of string pointers
    Iterator begin_, end_;
};

template <typename Char, typename... Members>
using StringSet = GenericStringSet<StringData<Char, Char*, Members...>>;

} // namespace dss_schimek

/******************************************************************************/

namespace dss_mehnert {

using dss_schimek::GenericStringSet;
using dss_schimek::GenericStringSetTraits;
using dss_schimek::has_member;
using dss_schimek::Index;
using dss_schimek::Length;
using dss_schimek::PEIndex;
using dss_schimek::StringData;
using dss_schimek::StringIndex;
using dss_schimek::StringSet;
using dss_schimek::StringSetBase;

/*!
 * Class implementing StringSet concept for compressed representations.
 */
template <typename String_>
class GenericCompressedStringSet
    : public GenericStringSetTraits<String_, GenericCompressedStringSet>,
      public StringSetBase<
          GenericCompressedStringSet<String_>,
          GenericStringSetTraits<String_, GenericCompressedStringSet>> {
public:
    using Traits = dss_schimek::GenericStringSetTraits<String_, GenericCompressedStringSet>;

    using Char = Traits::Char;
    using String = Traits::String;
    using Iterator = Traits::Iterator;
    using CharIterator = Traits::CharIterator;

    static_assert(has_member<String, Length>);

    static constexpr bool is_compressed = true;

    //! Construct from begin and end string pointers
    GenericCompressedStringSet() = default;

    //! Construct from begin and end string pointers
    GenericCompressedStringSet(Iterator const begin, Iterator const end)
        : begin_(begin),
          end_(end) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }

    //! Check if the set contains no elements
    bool empty() const { return begin_ == end_; }

    //! Iterator representing first String position
    Iterator begin() const { return begin_; }

    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Iterator-based array access (readable and writable) to String objects.
    String& operator[](Iterator const i) const { return *i; }

    //! Return length of the referenced string.
    size_t get_length(String const& str) const { return str.length; }

    //! Return CharIterator for referenced string, which belong to this set.
    CharIterator get_chars(String const& s, size_t depth) const { return s.string + depth; }

    //! Returns true if CharIterator is at end of the given String
    bool is_end(String const& s, CharIterator const& i) const { return i == (s.string + s.length); }

    //! Return complete string (for debugging purposes)
    std::string get_string(String const& s, size_t depth = 0) const {
        return {reinterpret_cast<char const*>(s.string) + depth, s.length - depth};
    }

    //! Subset this string set using iterator range.
    GenericCompressedStringSet sub(Iterator const begin, Iterator const end) const {
        return {begin, end};
    }

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t get_char_uint8_simple(String const&, CharIterator i) const { return uint8_t(*i); }

    //! \}

    void print(std::string_view prefix = "") const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << prefix << "[" << i++ << "] = " << (*pi) << " = " << get_string(*pi, 0);
        }
    }

    static String empty_string() { return {nullptr, Length{0}}; }

private:
    //! array of string pointers
    Iterator begin_, end_;
};

template <typename Char, typename... Members>
using CompressedStringSet = GenericCompressedStringSet<StringData<Char, Char*, Length, Members...>>;

} // namespace dss_mehnert
