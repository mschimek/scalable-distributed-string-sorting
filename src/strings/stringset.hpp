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

#ifndef PSS_SRC_TOOLS_STRINGSET_HEADER
#define PSS_SRC_TOOLS_STRINGSET_HEADER

#include <cassert>
#include <memory>
#include <vector>

#include <stdint.h>
#include <tlx/logger.hpp>

namespace dss_schimek {

typedef uintptr_t lcp_t;

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
        const typename Traits::String& a,
        const typename Traits::CharIterator& ai,
        const typename Traits::String& b,
        const typename Traits::CharIterator& bi
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return !ss.is_end(a, ai) && !ss.is_end(b, bi) && (*ai == *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool is_less(
        const typename Traits::String& a,
        const typename Traits::CharIterator& ai,
        const typename Traits::String& b,
        const typename Traits::CharIterator& bi
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.is_end(a, ai) || (!ss.is_end(a, ai) && !ss.is_end(b, bi) && *ai < *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool is_leq(
        const typename Traits::String& a,
        const typename Traits::CharIterator& ai,
        const typename Traits::String& b,
        const typename Traits::CharIterator& bi
    ) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.is_end(a, ai) || (!ss.is_end(a, ai) && !ss.is_end(b, bi) && *ai <= *bi);
    }

    //! \}

    //! \name Character Extractors
    //! \{

    typename Traits::Char get_char(const typename Traits::String& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return *ss.get_chars(s, depth);
    }

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t
    get_char_uint8_simple(const typename Traits::String& s, typename Traits::CharIterator i) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);

        if (ss.is_end(s, i))
            return 0;
        return uint8_t(*i);
    }

    //! Return up to 2 characters of string s at iterator i packed into a uint16
    //! (only works correctly for 8-bit characters)
    uint16_t get_char_uint16_simple(
        const typename Traits::String& s, typename Traits::CharIterator i
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
        const typename Traits::String& s, typename Traits::CharIterator i
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
        const typename Traits::String& s, typename Traits::CharIterator i
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

    uint8_t get_uint8(const typename Traits::String& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint8_simple(s, ss.get_chars(s, depth));
    }

    uint16_t get_uint16(const typename Traits::String& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint16_simple(s, ss.get_chars(s, depth));
    }

    uint32_t get_uint32(const typename Traits::String& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint32_simple(s, ss.get_chars(s, depth));
    }

    uint64_t get_uint64(const typename Traits::String& s, size_t depth) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return get_char_uint64_simple(s, ss.get_chars(s, depth));
    }

    //! \}

    //! Subset this string set using begin and size range.
    StringSet subr(const typename Traits::Iterator& begin, size_t size) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.sub(begin, begin + size);
    }

    //! Subset this string set using index range.
    StringSet subi(size_t begin, size_t end) const {
        StringSet const& ss = *static_cast<StringSet const*>(this);
        return ss.sub(ss.begin() + begin, ss.begin() + end);
    }

    bool check_order(const typename Traits::String& s1, const typename Traits::String& s2) const {
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
    static Type get_key(StringSet const& ss, const typename StringSet::String& s, size_t depth);
};

template <>
struct StringSetGetKeyHelper<uint8_t> {
    template <typename StringSet>
    static uint8_t get_key(StringSet const& ss, const typename StringSet::String& s, size_t depth) {
        return ss.get_uint8(s, depth);
    }
};

template <>
struct StringSetGetKeyHelper<uint16_t> {
    template <typename StringSet>
    static uint16_t
    get_key(StringSet const& ss, const typename StringSet::String& s, size_t depth) {
        return ss.get_uint16(s, depth);
    }
};

template <>
struct StringSetGetKeyHelper<uint32_t> {
    template <typename StringSet>
    static uint32_t
    get_key(StringSet const& ss, const typename StringSet::String& s, size_t depth) {
        return ss.get_uint32(s, depth);
    }
};

template <>
struct StringSetGetKeyHelper<uint64_t> {
    template <typename StringSet>
    static uint64_t
    get_key(StringSet const& ss, const typename StringSet::String& s, size_t depth) {
        return ss.get_uint64(s, depth);
    }
};

template <typename Type, typename StringSet>
Type get_key(StringSet const& ss, const typename StringSet::String& s, size_t depth) {
    return StringSetGetKeyHelper<Type>::get_key(ss, s, depth);
}

/******************************************************************************/

/*!
 * Traits class implementing StringSet concept for char* and unsigned char*
 * strings.
 */
template <typename CharType>
class GenericCharStringSetTraits {
public:
    //! exported alias for character type
    typedef CharType Char;

    //! String reference: pointer to first character
    typedef Char* String;

    //! Iterator over string references: pointer over pointers
    typedef String* Iterator;

    //! iterator of characters in a string
    typedef Char const* CharIterator;

    //! exported alias for assumed string container
    typedef std::pair<Iterator, size_t> Container;
};

/*!
 * Class implementing StringSet concept for char* and unsigned char* strings.
 */
template <typename CharType>
class GenericCharStringSet
    : public GenericCharStringSetTraits<CharType>,
      public StringSetBase<GenericCharStringSet<CharType>, GenericCharStringSetTraits<CharType>> {
public:
    typedef GenericCharStringSetTraits<CharType> Traits;

    typedef typename Traits::Char Char;
    typedef typename Traits::String String;
    typedef typename Traits::Iterator Iterator;
    typedef typename Traits::CharIterator CharIterator;
    typedef typename Traits::Container Container;

    //! Construct from begin and end string pointers
    GenericCharStringSet(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    //! Construct from a string container
    explicit GenericCharStringSet(Container const& c) : begin_(c.first), end_(c.first + c.second) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
    //! Iterator representing first String position
    Iterator begin() const { return begin_; }
    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Iterator-based array access (readable and writable) to String objects.
    String& operator[](Iterator i) const { return *i; }

    //! Return CharIterator for referenced string, which belong to this set.
    CharIterator get_chars(String const& s, size_t depth) const { return s + depth; }

    //! Returns true if CharIterator is at end of the given String
    bool is_end(String const&, CharIterator const& i) const { return (*i == 0); }

    //! Return complete string (for debugging purposes)
    std::string get_string(String const& s, size_t depth = 0) const {
        return std::string(reinterpret_cast<char const*>(s) + depth);
    }

    //! Subset this string set using iterator range.
    GenericCharStringSet sub(Iterator begin, Iterator end) const {
        return GenericCharStringSet(begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    static Container allocate(size_t n) { return std::make_pair(new String[n], n); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        delete[] c.first;
        c.first = NULL;
    }

    //! \name CharIterator Comparisons
    //! \{

    //! check equality of two strings a and b at char iterators ai and bi.
    bool
    is_equal(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai == *bi) && (*ai != 0);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool
    is_less(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai < *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool
    is_leq(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai <= *bi);
    }

    //! \}

    //! \name Character Extractors
    //! \{

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t get_char_uint8_simple(String const&, CharIterator i) const { return uint8_t(*i); }

    //! \}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << *pi << " = " << get_string(*pi, 0);
        }
    }

    size_t get_length(String const& str) const {
        size_t length = 0;
        CharIterator it = get_chars(str, 0);
        while (*it != 0) {
            ++length;
            ++it;
        }
        return length;
    }

    static String empty_string() {
        static Char zero = 0;
        return &zero;
    }

    static std::string getName() { return "GenericCharStringSet"; }

protected:
    //! array of string pointers
    Iterator begin_, end_;
};

typedef GenericCharStringSet<char> CharStringSet;
typedef GenericCharStringSet<unsigned char> UCharStringSet;
/******************************************************************************/

template <typename String, typename Length = size_t>
struct StringLengthIndex {
    StringLengthIndex() : string(nullptr), length(0), index(0) {}
    StringLengthIndex(const String string, const Length length, const uint64_t index)
        : string(string),
          length(length),
          index(index) {}
    StringLengthIndex(const String string, const Length length)
        : string(string),
          length(length),
          index(0) {}
    String string;
    Length length;
    uint64_t index;
    inline void setChars(String newString) { string = newString; }
    inline String getChars() { return string; }
    inline Length getLength() { return length; }
};

template <typename String, typename Length>
std::ostream& operator<<(std::ostream& out, StringLengthIndex<String, Length> const& str_length) {
    return out << "[" << str_length.string << "," << str_length.length << "," << str_length.index
               << "]";
}
/*!
 * Traits class implementing StringSet concept for char* and unsigned char*
 * strings with additional length attribute.
 */
template <typename CharType>
class GenericCharLengthIndexStringSetTraits {
public:
    //! exported alias for character type
    using Char = CharType;

    //! String reference: pointer to first character
    using String = StringLengthIndex<Char*>;

    //! Iterator over string references: pointer over pointers
    typedef String* Iterator;

    //! iterator of characters in a string
    typedef Char const* CharIterator;

    //! exported alias for assumed string container
    typedef std::pair<Iterator, size_t> Container;
};

/*!
 * Class implementing StringSet concept for char* and unsigned char* strings
 * with additional length attribute.
 */
template <typename CharType>
class GenericCharLengthIndexStringSet : public GenericCharLengthIndexStringSetTraits<CharType>,
                                        public StringSetBase<
                                            GenericCharLengthIndexStringSet<CharType>,
                                            GenericCharLengthIndexStringSetTraits<CharType>> {
public:
    typedef GenericCharLengthIndexStringSetTraits<CharType> Traits;

    typedef typename Traits::Char Char;
    typedef typename Traits::String String;
    typedef typename Traits::Iterator Iterator;
    typedef typename Traits::CharIterator CharIterator;
    typedef typename Traits::Container Container;

    //! Construct from begin and end string pointers
    GenericCharLengthIndexStringSet(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    //! Construct from a string container
    explicit GenericCharLengthIndexStringSet(Container const& c)
        : begin_(c.first),
          end_(c.first + c.second) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
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
    std::string get_string(String const& s, size_t depth = 0) const {
        return std::string(reinterpret_cast<char const*>(s.string) + depth);
    }

    //! Subset this string set using iterator range.
    GenericCharLengthIndexStringSet sub(Iterator begin, Iterator end) const {
        return GenericCharLengthIndexStringSet(begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    static Container allocate(size_t n) { return std::make_pair(new String[n], n); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        delete[] c.first;
        c.first = NULL;
    }

    //! \name CharIterator Comparisons
    //! \{

    //! check equality of two strings a and b at char iterators ai and bi.
    bool is_equal(String const& a, CharIterator const& ai, String const& b, CharIterator const& bi)
        const {
        if (*ai == *bi && *ai == 0) {
            return a.index == b.index;
        }
        return (*ai == *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool is_less(String const& a, CharIterator const& ai, String const& b, CharIterator const& bi)
        const {
        if (*ai == 0 && *bi == 0)
            return a.index < b.index;
        return (*ai < *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool
    is_leq(String const& a, CharIterator const& ai, String const& b, CharIterator const& bi) const {
        if (*ai == 0 && *bi == 0)
            return a.index <= b.index;
        return (*ai <= *bi);
    }

    //! \}

    //! \name Character Extractors
    //! \{

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t get_char_uint8_simple(String const&, CharIterator i) const { return uint8_t(*i); }

    //! \}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << (*pi) << " = " << get_string(*pi, 0);
        }
    }

    static String empty_string() {
        static Char zeroChar = 0;
        static String zero(&zeroChar, 0);
        return zero;
    }

    size_t get_length(String const& str) const { return str.length; }

    uint64_t get_index(String const& str) const { return str.index; }

    static std::string getName() { return "GenericCharLengthIndexStringSet"; }

protected:
    //! array of string pointers
    Iterator begin_, end_;
};
typedef GenericCharLengthIndexStringSet<char> CharLengthIndexStringSet;
typedef GenericCharLengthIndexStringSet<unsigned char> UCharLengthIndexStringSet;

/******************************************************************************/

template <typename String, typename Length = size_t>
struct StringLength {
    StringLength() : string(nullptr), length(0) {}
    StringLength(const String string, const Length length) : string(string), length(length) {}
    String string;
    Length length;
    inline void setChars(String newString) { string = newString; }
    inline String getChars() { return string; }
    inline Length getLength() { return length; }
};

template <typename String, typename Length>
std::ostream& operator<<(std::ostream& out, StringLength<String, Length> const& str_length) {
    return out << "[" << str_length.string << "," << str_length.length << "]";
}
/*!
 * Traits class implementing StringSet concept for char* and unsigned char*
 * strings with additional length attribute.
 */
template <typename CharType>
class GenericCharLengthStringSetTraits {
public:
    //! exported alias for character type
    using Char = CharType;

    //! String reference: pointer to first character
    using String = StringLength<Char*>;

    //! Iterator over string references: pointer over pointers
    typedef String* Iterator;

    //! iterator of characters in a string
    typedef Char const* CharIterator;

    //! exported alias for assumed string container
    typedef std::pair<Iterator, size_t> Container;
};

/*!
 * Class implementing StringSet concept for char* and unsigned char* strings
 * with additional length attribute.
 */
template <typename CharType>
class GenericCharLengthStringSet : public GenericCharLengthStringSetTraits<CharType>,
                                   public StringSetBase<
                                       GenericCharLengthStringSet<CharType>,
                                       GenericCharLengthStringSetTraits<CharType>> {
public:
    typedef GenericCharLengthStringSetTraits<CharType> Traits;

    typedef typename Traits::Char Char;
    typedef typename Traits::String String;
    typedef typename Traits::Iterator Iterator;
    typedef typename Traits::CharIterator CharIterator;
    typedef typename Traits::Container Container;

    //! Construct from begin and end string pointers
    GenericCharLengthStringSet(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    //! Construct from a string container
    explicit GenericCharLengthStringSet(Container const& c)
        : begin_(c.first),
          end_(c.first + c.second) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
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
    std::string get_string(String const& s, size_t depth = 0) const {
        return std::string(reinterpret_cast<char const*>(s.string) + depth);
    }

    //! Subset this string set using iterator range.
    GenericCharLengthStringSet sub(Iterator begin, Iterator end) const {
        return GenericCharLengthStringSet(begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    static Container allocate(size_t n) { return std::make_pair(new String[n], n); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        delete[] c.first;
        c.first = NULL;
    }

    //! \name CharIterator Comparisons
    //! \{

    //! check equality of two strings a and b at char iterators ai and bi.
    bool
    is_equal(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai == *bi) && (*ai != 0);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool
    is_less(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai < *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool
    is_leq(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai <= *bi);
    }

    //! \}

    //! \name Character Extractors
    //! \{

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t get_char_uint8_simple(String const&, CharIterator i) const { return uint8_t(*i); }

    //! \}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << (*pi) << " = " << get_string(*pi, 0);
        }
    }
    void print(std::string const& prefix) const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << prefix << "[" << i++ << "] = " << (*pi) << " = " << get_string(*pi, 0);
        }
    }

    static String empty_string() {
        static Char zeroChar = 0;
        static String zero(&zeroChar, 0);
        return zero;
    }

    size_t get_length(String const& str) const { return str.length; }

    static std::string getName() { return "GenericCharLengthStringSet"; }

protected:
    //! array of string pointers
    Iterator begin_, end_;
};

typedef GenericCharLengthStringSet<char> CharLengthStringSet;
typedef GenericCharLengthStringSet<unsigned char> UCharLengthStringSet;

/******************************************************************************/

template <typename String>
struct StringStringIndexPEIndex {
    StringStringIndexPEIndex() : string(nullptr), stringIndex(0), PEIndex(0) {}
    StringStringIndexPEIndex(const String string, const size_t stringIndex, const size_t PEIndex)
        : string(string),
          stringIndex(stringIndex),
          PEIndex(PEIndex) {}
    String string;
    size_t stringIndex;
    size_t PEIndex;
};

template <typename String>
std::ostream& operator<<(std::ostream& out, StringStringIndexPEIndex<String> const& str) {
    return out << "[" << str.string << "," << str.stringIndex << ", " << str.PEIndex << "]";
}
/*!
 * Traits class implementing StringSet concept for char* and unsigned char*
 * strings with additional length attribute.
 */
template <typename CharType>
class GenericCharIndexPEIndexStringSetTraits {
public:
    //! exported alias for character type
    using Char = CharType;

    //! String reference: pointer to first character
    using String = StringStringIndexPEIndex<Char*>;

    //! Iterator over string references: pointer over pointers
    typedef String* Iterator;

    //! iterator of characters in a string
    typedef Char const* CharIterator;

    //! exported alias for assumed string container
    typedef std::pair<Iterator, size_t> Container;
};

/*!
 * Class implementing StringSet concept for char* and unsigned char* strings
 * with additional length attribute.
 */
template <typename CharType>
class GenericCharIndexPEIndexStringSet : public GenericCharIndexPEIndexStringSetTraits<CharType>,
                                         public StringSetBase<
                                             GenericCharIndexPEIndexStringSet<CharType>,
                                             GenericCharIndexPEIndexStringSetTraits<CharType>> {
public:
    typedef GenericCharIndexPEIndexStringSetTraits<CharType> Traits;

    typedef typename Traits::Char Char;
    typedef typename Traits::String String;
    typedef typename Traits::Iterator Iterator;
    typedef typename Traits::CharIterator CharIterator;
    typedef typename Traits::Container Container;

    //! Construct from begin and end string pointers
    GenericCharIndexPEIndexStringSet(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    //! Construct from a string container
    explicit GenericCharIndexPEIndexStringSet(Container const& c)
        : begin_(c.first),
          end_(c.first + c.second) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
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
    std::string get_string(String const& s, size_t depth = 0) const {
        return std::string(reinterpret_cast<char const*>(s.string) + depth);
    }

    //! Subset this string set using iterator range.
    GenericCharIndexPEIndexStringSet sub(Iterator begin, Iterator end) const {
        return GenericCharIndexPEIndexStringSet(begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    static Container allocate(size_t n) { return std::make_pair(new String[n], n); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        delete[] c.first;
        c.first = NULL;
    }

    //! \name CharIterator Comparisons
    //! \{

    //! check equality of two strings a and b at char iterators ai and bi.
    bool
    is_equal(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai == *bi) && (*ai != 0);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool
    is_less(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai < *bi);
    }

    //! check if string a is less or equal to string b at iterators ai and bi.
    bool
    is_leq(String const&, CharIterator const& ai, String const&, CharIterator const& bi) const {
        return (*ai <= *bi);
    }

    //! \}

    //! \name Character Extractors
    //! \{

    //! Return up to 1 characters of string s at iterator i packed into a uint8
    //! (only works correctly for 8-bit characters)
    uint8_t get_char_uint8_simple(String const&, CharIterator i) const { return uint8_t(*i); }

    //! \}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << (*pi) << " = " << get_string(*pi, 0);
        }
    }

    static String empty_string() {
        static String zero(0, 0, 0);
        return zero;
    }

    size_t get_length(String const& str) const {
        size_t length = 0;
        CharIterator it = get_chars(str, 0);
        while (*it != 0) {
            ++length;
            ++it;
        }
        return length;
    }

    size_t getIndex(String const& str) const { return str.stringIndex; }

    size_t getPEIndex(String const& str) const { return str.PEIndex; }

    static std::string getName() { return "GenericCharIndexPEIndexStringSet"; }

protected:
    //! array of string pointers
    Iterator begin_, end_;
};

typedef GenericCharIndexPEIndexStringSet<char> CharIndexPEIndexStringSet;
typedef GenericCharIndexPEIndexStringSet<unsigned char> UCharIndexPEIndexStringSet;

/*!
 * Class implementing StringSet concept for a std::vector containing std::string
 * objects.
 */
class VectorStringSetTraits {
public:
    //! exported alias for assumed string container
    typedef std::vector<std::string> Container;

    //! exported alias for character type
    typedef std::string::value_type Char;

    //! String reference: std::string, which should be reference counted.
    typedef typename Container::value_type String;

    //! Iterator over string references: using std::vector's iterator
    typedef typename Container::iterator Iterator;

    //! iterator of characters in a string
    typedef std::string::const_iterator CharIterator;
};

/*!
 * Class implementing StringSet concept for a std::vector containing std::string
 * objects.
 */
class VectorStringSet : public VectorStringSetTraits,
                        public StringSetBase<VectorStringSet, VectorStringSetTraits> {
public:
    //! Construct from begin and end string pointers
    VectorStringSet(Iterator const& begin, Iterator const& end) : begin_(begin), end_(end) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
    //! Iterator representing first String position
    Iterator begin() const { return begin_; }
    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Array access (readable and writable) to String objects.
    String& operator[](Iterator const& i) const { return *i; }

    //! Return CharIterator for referenced string, which belongs to this set.
    CharIterator get_chars(String const& s, size_t depth) const { return s.begin() + depth; }

    //! Returns true if CharIterator is at end of the given String
    bool is_end(String const& s, CharIterator const& i) const { return (i >= s.end()); }

    //! Return complete string (for debugging purposes)
    std::string get_string(String const& s, size_t depth = 0) const { return s.substr(depth); }

    //! Subset this string set using iterator range.
    VectorStringSet sub(Iterator begin, Iterator end) const { return VectorStringSet(begin, end); }

    //! Allocate a new temporary string container with n empty Strings
    static Container allocate(size_t n) { return Container(n); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        Container v;
        v.swap(c);
    }

    //! Construct from a string container
    explicit VectorStringSet(Container& c) : begin_(c.begin()), end_(c.end()) {}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << *pi << " = " << get_string(*pi, 0);
        }
    }

protected:
    //! vector of std::string objects
    Iterator begin_, end_;
};

/******************************************************************************/

/*!
 * Class implementing StringSet concept for a std::vector containing
 * std::unique_ptr<std::string> objects, which are non-copyable.
 */
class VectorPtrStringSetTraits {
public:
    //! exported alias for assumed string container
    typedef std::vector<std::unique_ptr<std::string>> Container;

    //! exported alias for character type
    typedef std::string::value_type Char;

    //! String reference: std::string, which should be reference counted.
    typedef typename Container::value_type String;

    //! Iterator over string references: using std::vector's iterator
    typedef typename Container::iterator Iterator;

    //! iterator of characters in a string
    typedef std::string::const_iterator CharIterator;
};

/*!
 * Class implementing StringSet concept for a std::vector containing std::string
 * objects.
 */
class VectorPtrStringSet : public VectorPtrStringSetTraits,
                           public StringSetBase<VectorPtrStringSet, VectorPtrStringSetTraits> {
public:
    //! Construct from begin and end string pointers
    VectorPtrStringSet(Iterator const& begin, Iterator const& end) : begin_(begin), end_(end) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
    //! Iterator representing first String position
    Iterator begin() const { return begin_; }
    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Array access (readable and writable) to String objects.
    String& operator[](Iterator const& i) const { return *i; }

    //! Return CharIterator for referenced string, which belongs to this set.
    CharIterator get_chars(String const& s, size_t depth) const { return s->begin() + depth; }

    //! Returns true if CharIterator is at end of the given String
    bool is_end(String const& s, CharIterator const& i) const { return (i >= s->end()); }

    //! Return complete string (for debugging purposes)
    std::string get_string(String const& s, size_t depth = 0) const { return s->substr(depth); }

    //! Subset this string set using iterator range.
    VectorPtrStringSet sub(Iterator begin, Iterator end) const {
        return VectorPtrStringSet(begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    static Container allocate(size_t n) { return Container(n); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        Container v;
        v.swap(c);
    }

    //! Construct from a string container
    explicit VectorPtrStringSet(Container& c) : begin_(c.begin()), end_(c.end()) {}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << pi->get() << " = " << get_string(*pi, 0);
        }
    }

protected:
    //! vector of std::string objects
    Iterator begin_, end_;
};

/******************************************************************************/

/*!
 * Class implementing StringSet concept for suffix sorting indexes of an
 * unsigned char* text object.
 */
class UCharSuffixSetTraits {
public:
    //! exported alias for assumed text container
    typedef unsigned char const* Text;

    //! exported alias for character type
    typedef unsigned char Char;

    //! String reference: suffix index of the text.
    typedef int String;

    //! Iterator over string references: using std::vector's iterator over
    //! suffix array vector
    typedef String* Iterator;

    //! iterator of characters in a string
    typedef Char const* CharIterator;

    //! exported alias for assumed string container
    typedef std::tuple<Text, Text, Iterator, size_t> Container;
};

/*!
 * Class implementing StringSet concept for suffix sorting indexes of a
 * std::string text object.
 */
class UCharSuffixSet : public UCharSuffixSetTraits,
                       public StringSetBase<UCharSuffixSet, UCharSuffixSetTraits> {
public:
    //! Construct from begin and end string pointers
    UCharSuffixSet(
        Text const& text, Text const& text_end, Iterator const& begin, Iterator const& end
    )
        : text_(text),
          text_end_(text_end),
          begin_(begin),
          end_(end) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
    //! Iterator representing first String position
    Iterator begin() const { return begin_; }
    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Array access (readable and writable) to String objects.
    String& operator[](Iterator const& i) const { return *i; }

    //! Return CharIterator for referenced string, which belongs to this set.
    CharIterator get_chars(String const& s, size_t depth) const { return text_ + s + depth; }

    //! Returns true if CharIterator is at end of the given String
    bool is_end(String const&, CharIterator const& i) const { return (i >= text_end_); }

    //! Return complete string (for debugging purposes)
    std::string get_string(String const& s, size_t depth = 0) const {
        return std::string(
            reinterpret_cast<char const*>(text_ + s + depth),
            reinterpret_cast<char const*>(text_end_)
        );
    }

    //! Subset this string set using iterator range.
    UCharSuffixSet sub(Iterator begin, Iterator end) const {
        return UCharSuffixSet(text_, text_end_, begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    Container allocate(size_t n) const {
        return std::make_tuple(text_, text_end_, new String[n], n);
    }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) { delete[] std::get<2>(c); }

    //! Construct from a string container
    explicit UCharSuffixSet(Container& c)
        : text_(std::get<0>(c)),
          text_end_(std::get<1>(c)),
          begin_(std::get<2>(c)),
          end_(std::get<2>(c) + std::get<3>(c)) {}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << *pi << " = " << get_string(*pi, 0);
        }
    }

protected:
    //! reference to base text
    Text text_, text_end_;

    //! iterators inside the output suffix array.
    Iterator begin_, end_;
};

/******************************************************************************/

/*!
 * Class implementing StringSet concept for suffix sorting indexes of a
 * std::string text object.
 */
class StringSuffixSetTraits {
public:
    //! exported alias for assumed string container
    typedef std::string Text;

    //! exported alias for character type
    typedef std::string::value_type Char;

    //! String reference: suffix index of the text.
    typedef typename Text::size_type String;

    //! Iterator over string references: using std::vector's iterator over
    //! suffix array vector
    typedef typename std::vector<String>::iterator Iterator;

    //! iterator of characters in a string
    typedef std::string::const_iterator CharIterator;

    //! exported alias for assumed string container
    typedef std::tuple<Text, std::vector<String>> Container;
};

/*!
 * Class implementing StringSet concept for suffix sorting indexes of a
 * std::string text object.
 */
class StringSuffixSet : public StringSuffixSetTraits,
                        public StringSetBase<StringSuffixSet, StringSuffixSetTraits> {
public:
    //! Construct from begin and end string pointers
    StringSuffixSet(Text const& text, Iterator const& begin, Iterator const& end)
        : text_(&text),
          begin_(begin),
          end_(end) {}

    //! Initializing constructor which fills output vector sa with indices.
    static StringSuffixSet Initialize(Text const& text, std::vector<String>& sa) {
        sa.resize(text.size());
        for (size_t i = 0; i < text.size(); ++i)
            sa[i] = i;
        return StringSuffixSet(text, sa.begin(), sa.end());
    }

    //! Return size of string array
    size_t size() const { return end_ - begin_; }
    //! Iterator representing first String position
    Iterator begin() const { return begin_; }
    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Array access (readable and writable) to String objects.
    String& operator[](Iterator const& i) const { return *i; }

    //! Return CharIterator for referenced string, which belongs to this set.
    CharIterator get_chars(String const& s, size_t depth) const {
        return text_->begin() + s + depth;
    }

    //! Returns true if CharIterator is at end of the given String
    bool is_end(String const&, CharIterator const& i) const { return (i >= text_->end()); }

    //! Return complete string (for debugging purposes)
    std::string get_string(String const& s, size_t depth = 0) const {
        return text_->substr(s + depth);
    }

    //! Subset this string set using iterator range.
    StringSuffixSet sub(Iterator begin, Iterator end) const {
        return StringSuffixSet(*text_, begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    Container allocate(size_t n) const { return std::make_tuple(*text_, std::vector<String>(n)); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        std::vector<String> v;
        v.swap(std::get<1>(c));
    }

    //! Construct from a string container
    explicit StringSuffixSet(Container& c)
        : text_(&std::get<0>(c)),
          begin_(std::get<1>(c).begin()),
          end_(std::get<1>(c).end()) {}

    void print() const {
        size_t i = 0;
        for (Iterator pi = begin(); pi != end(); ++pi) {
            LOG1 << "[" << i++ << "] = " << *pi << " = " << get_string(*pi, 0);
        }
    }

protected:
    //! reference to base text
    Text const* text_;

    //! iterators inside the output suffix array.
    Iterator begin_, end_;
};

} // namespace dss_schimek

#endif // !PSS_SRC_TOOLS_STRINGSET_HEADER

/******************************************************************************/
