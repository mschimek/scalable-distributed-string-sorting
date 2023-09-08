// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

#include <stdint.h>
#include <tlx/logger.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "strings/stringset.hpp"

namespace dss_schimek {

namespace _internal {

template <typename OutIt, typename Char, typename Init>
void init_str_len(OutIt out, std::vector<Char>& raw_strings, Init init) {
    for (auto it = raw_strings.begin(); it != raw_strings.end(); ++it, ++out) {
        auto begin = it;
        while (*it != 0)
            ++it;

        *out = init(&(*begin), static_cast<size_t>(std::distance(begin, it)));
    }
}

template <typename StringSet>
class InitPolicy {
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

public:
    std::vector<String> init_strings(std::vector<Char>& raw_strings) {
        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100);

        auto init = [](auto str, auto) { return String{str}; };
        init_str_len(std::back_inserter(strings), raw_strings, init);
        return strings;
    }
};

template <typename CharType>
class InitPolicy<GenericCharLengthStringSet<CharType>> {
    using StringSet = GenericCharLengthStringSet<CharType>;
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

public:
    std::vector<String> init_strings(std::vector<Char>& raw_strings) {
        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100);

        auto init = [](auto str, auto len) { return String{str, Length{len}}; };
        init_str_len(std::back_inserter(strings), raw_strings, init);
        return strings;
    }
};

template <typename CharType>
class InitPolicy<GenericCharLengthIndexStringSet<CharType>> {
    using StringSet = GenericCharLengthIndexStringSet<CharType>;
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

public:
    std::vector<String> init_strings(std::vector<Char>& raw_strings) {
        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100);

        auto init = [index = size_t{0}](auto str, auto len) mutable {
            return String{str, Length{len}, Index{index++}};
        };
        init_str_len(std::back_inserter(strings), raw_strings, init);
        return strings;
    }

    std::vector<String>
    init_strings(std::vector<Char>& raw_strings, std::vector<uint64_t> const& indices) {
        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100);

        auto init = [index_it = indices.cbegin()](auto str, auto len) mutable {
            return String{str, Length{len}, Index{*(index_it++)}};
        };
        init_str_len(std::back_inserter(strings), raw_strings, init);
        return strings;
    }
};

template <typename CharType>
class InitPolicy<GenericCharLengthIndexPEIndexStringSet<CharType>> {
    using StringSet = GenericCharLengthIndexPEIndexStringSet<CharType>;
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

public:
    // todo
    template <typename String_>
    std::vector<String>
    init_strings(std::vector<Char>& raw_strings, std::vector<String_> const& ss, size_t rank) {
        std::vector<String> strings(ss.size());
        auto op = [=, idx = size_t{0}](auto const& str) mutable {
            return String{str.string, Length{str.length}, StringIndex{idx++}, PEIndex{rank}};
        };
        std::transform(std::cbegin(ss), std::cend(ss), std::begin(strings), op);
        return strings;
    }

    std::vector<String> init_strings(
        std::vector<Char>& raw_strings,
        std::vector<size_t> const& intervals,
        std::vector<size_t> const& offsets
    ) {
        auto num_strs = std::accumulate(intervals.begin(), intervals.end(), size_t{0});
        std::vector<String> strings(num_strs);

        auto init = [](auto str, auto len) { return String{str, Length{len}}; };
        init_str_len(strings.begin(), raw_strings, init);

        auto str = strings.begin();
        for (size_t PE_idx = 0; PE_idx < intervals.size(); ++PE_idx) {
            for (size_t str_idx = 0; str_idx < intervals[PE_idx]; ++str_idx, ++str) {
                str->stringIndex = offsets[PE_idx] + str_idx;
                str->PEIndex = PE_idx;
            }
        }
        return strings;
    }

    // todo this overload is still pretty poor
    std::vector<String> init_strings(
        std::vector<Char>& raw_strings,
        std::vector<size_t>&& PE_indices,
        std::vector<size_t>&& str_indices
    ) {
        std::vector<String> strings(PE_indices.size());

        auto str_it = str_indices.cbegin();
        auto PE_it = PE_indices.cbegin();
        auto init = [&](auto str, auto len) -> String {
            return {str, Length{len}, StringIndex{*str_it++}, PEIndex{*PE_it++}};
        };
        init_str_len(strings.begin(), raw_strings, init);
        return strings;
    }
};

} // namespace _internal


template <typename StringSet_, typename Container>
class BaseStringContainer : private _internal::InitPolicy<StringSet_> {
public:
    using StringSet = StringSet_;
    using Char = StringSet::Char;
    using CharIterator = StringSet::CharIterator;
    using String = StringSet::String;

    using StringPtr = tlx::sort_strings_detail::StringPtr<StringSet>;
    using AutoStringPtr = StringPtr;

    static constexpr bool isIndexed = StringSet::is_indexed;
    static constexpr bool has_lcps = false;

    BaseStringContainer() : raw_strings_{std::make_unique<std::vector<Char>>()} {}

    String operator[](size_t i) { return strings_[i]; }
    String front() { return strings_.front(); }
    String back() { return strings_.back(); }
    String* strings() { return strings_.data(); }
    size_t size() const { return strings_.size(); }
    bool empty() const { return strings_.empty(); }
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<String>& getStrings() { return strings_; }
    std::vector<String> const& getStrings() const { return strings_; }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }
    std::vector<Char>&& release_raw_strings() { return std::move(*raw_strings_); }
    std::vector<String>&& release_strings() { return std::move(strings_); }

    friend void swap(
        BaseStringContainer<StringSet, Container>& lhs,
        BaseStringContainer<StringSet, Container>& rhs
    ) {
        std::swap(lhs.raw_strings_, rhs.raw_strings_);
        std::swap(rhs.strings_, rhs.strings_);
    }

    std::vector<unsigned char> getRawString(int64_t i) {
        if (i < 0 || static_cast<uint64_t>(i) > size())
            return std::vector<unsigned char>(1, 0);

        auto const length = strings_[i].length + 1;
        std::vector<unsigned char> rawString(length);
        std::copy(strings_[i].string, strings_[i].string + length, rawString.begin());
        return rawString;
    }

    StringSet make_string_set() { return {strings(), strings() + size()}; }
    StringPtr make_string_ptr() { return {make_string_set()}; }

    auto make_auto_ptr() { return static_cast<Container*>(this)->make_auto_ptr_(); }

    void resize_strings(size_t const count) { strings_.resize(count, String{}); }

    void deleteRawStrings() {
        raw_strings_->clear();
        raw_strings_->shrink_to_fit();
    }

    void deleteStrings() {
        strings_.clear();
        strings_.shrink_to_fit();
    }

    void deleteAll() {
        deleteRawStrings();
        deleteStrings();
    }

    void set(std::vector<Char>&& raw_strings) { *raw_strings_ = std::move(raw_strings); }
    void set(std::vector<String>&& strings) { strings_ = std::move(strings); }

    void update(std::vector<Char>&& raw_strings) {
        set(std::move(raw_strings));
        update_strings();
    }

    void orderRawStrings() {
        std::vector<Char> new_buffer;
        orderRawStrings(new_buffer);
    }

    void orderRawStrings(std::vector<Char>& char_buffer) {
        char_buffer.resize(make_string_set().get_sum_length() + size());

        for (auto dest = char_buffer.begin(); auto& string: strings_) {
            auto const chars = string.getChars();
            string.setChars(&*dest);
            dest = std::copy_n(chars, string.getLength(), dest);
            *dest++ = 0;
        }

        std::swap(*raw_strings_, char_buffer);
    }

    bool isConsistent() {
        for (size_t i = 0; i < strings_.size(); ++i) {
            auto const adressEndByteOfString = strings_[i].getChars() + strings_[i].getLength();
            if (adressEndByteOfString < raw_strings_->data()
                || adressEndByteOfString >= raw_strings_->data() + raw_strings_->size())
                return false;
            if (*adressEndByteOfString != 0)
                return false;
        }
        return true;
    }

protected:
    static constexpr size_t approx_string_length = 10;

    std::unique_ptr<std::vector<Char>> raw_strings_;
    std::vector<String> strings_;

    template <typename... Args>
    explicit BaseStringContainer(std::vector<Char>&& raw_strings, Args&&... args)
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))),
          strings_(_internal::InitPolicy<StringSet>::init_strings(
              *raw_strings_, std::forward<Args>(args)...
          )) {}

    template <typename... Args>
    void update_strings(Args&&... args) {
        strings_ = _internal::InitPolicy<StringSet>::init_strings(
            *raw_strings_,
            std::forward<Args>(args)...
        );
    }

private:
    StringPtr make_auto_ptr_() { return make_string_ptr(); }
};

template <typename StringSet_, typename Container>
class BaseStringLcpContainer : public BaseStringContainer<StringSet_, Container> {
    friend BaseStringContainer<StringSet_, Container>;

public:
    using Base = BaseStringContainer<StringSet_, Container>;
    using Char = Base::Char;
    using String = Base::String;

    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet_, size_t>;
    using AutoStringPtr = StringLcpPtr;

    static constexpr bool has_lcps = true;

    BaseStringLcpContainer() = default;

    size_t* lcp_array() { return lcps_.data(); }
    std::vector<size_t>& lcps() { return lcps_; }
    std::vector<size_t> const& lcps() const { return lcps_; }
    std::vector<size_t>&& release_lcps() { return std::move(lcps_); }

    friend void swap(
        BaseStringLcpContainer<StringSet_, Container>& lhs,
        BaseStringLcpContainer<StringSet_, Container>& rhs
    ) {
        std::swap(lhs.raw_strings_, rhs.raw_strings_);
        std::swap(lhs.strings_, rhs.strings_);
        std::swap(lhs.lcps_, rhs.lcps_);
    }

    StringLcpPtr make_string_lcp_ptr() { return {this->make_string_set(), this->lcp_array()}; }

    void resize_strings(size_t const count) {
        this->strings_.resize(count);
        this->lcps_.resize(count);
    }

    // pull in `set` overlaods from `BaseStringcontainer`
    using Base::set;

    void set(std::vector<size_t>&& lcps) { lcps_ = std::move(lcps); }

    void update(std::vector<Char>&& raw_strings) {
        set(std::move(raw_strings));
        this->update_strings();
        if (lcps_.size() != this->size())
            lcps_.resize(this->size(), 0);
    }

    void deleteLcps() {
        lcps_.clear();
        lcps_.shrink_to_fit();
    }

    void deleteAll() {
        this->deleteRawStrings();
        this->deleteStrings();
        deleteLcps();
    }

    template <typename StringSet, typename LcpIt>
    void extend_prefix(StringSet const& ss, const LcpIt first_lcp, LcpIt const last_lcp) {
        using std::begin;
        using std::end;

        assert_equal(std::distance(first_lcp, last_lcp), std::ssize(ss));
        assert(first_lcp == last_lcp || *first_lcp == 0);
        if (ss.empty()) {
            return;
        }

        size_t const L = std::accumulate(first_lcp, last_lcp, size_t{0});
        std::vector<typename StringSet::Char> raw_strings(this->char_size() + L);
        auto prev_chars = raw_strings.begin();
        auto curr_chars = raw_strings.begin();

        auto lcp_it = first_lcp;
        for (auto it = begin(ss); it != end(ss); ++it, ++lcp_it) {
            auto& curr_str = ss[it];
            auto curr_str_begin = ss.get_chars(curr_str, 0);
            auto curr_str_len = ss.get_length(curr_str) + 1;

            curr_str.string = &(*curr_chars);
            curr_str.length = curr_str_len + *lcp_it - 1;

            // copy common prefix from previous string
            auto lcp_chars = std::exchange(prev_chars, curr_chars);
            curr_chars = std::copy_n(lcp_chars, *lcp_it, curr_chars);

            // copy remaining (distinct) characters
            curr_chars = std::copy_n(curr_str_begin, curr_str_len, curr_chars);
        }

        *this->raw_strings_ = std::move(raw_strings);
    }

    bool operator==(Container const& other) {
        return this->raw_strings() == other.raw_strings() && lcps() == other.lcps();
    }

protected:
    std::vector<size_t> lcps_;

    template <typename... Args>
    explicit BaseStringLcpContainer(
        std::vector<Char>&& raw_strings, std::vector<size_t>&& lcps, Args&&... args
    )
        : Base{std::move(raw_strings), std::forward<Args>(args)...},
          lcps_(std::move(lcps)) {}

private:
    StringLcpPtr make_auto_ptr_() { return make_string_lcp_ptr(); }
};


template <typename StringSet_>
class StringContainer : public BaseStringContainer<StringSet_, StringContainer<StringSet_>> {
public:
    using Base = BaseStringContainer<StringSet_, StringContainer<StringSet_>>;
    using Char = Base::Char;

    StringContainer() = default;

    template <typename... Args>
    explicit StringContainer(std::vector<Char>&& raw_strings, Args&&... args)
        : Base{std::move(raw_strings), std::forward<Args>(args)...} {}

    template <typename Container_ = StringContainer<StringSet_>>
    std::enable_if_t<Container_::isIndexed>
    update(std::vector<Char>&& raw_strings, std::vector<uint64_t> const& indices) {
        this->set(std::move(raw_strings));
        this->update_strings(indices);
    }
};

template <typename StringSet_>
class StringLcpContainer
    : public BaseStringLcpContainer<StringSet_, StringLcpContainer<StringSet_>> {
public:
    using Base = BaseStringLcpContainer<StringSet_, StringLcpContainer<StringSet_>>;
    using Char = Base::Char;

    static constexpr bool isIndexed = false;

    StringLcpContainer() = default;

    explicit StringLcpContainer(std::vector<Char>&& raw_strings)
        : Base{std::move(raw_strings), {}} {
        this->lcps_.resize(this->size(), 0);
    }

    explicit StringLcpContainer(std::vector<Char>&& raw_strings, std::vector<size_t>&& lcps)
        : Base{std::move(raw_strings), std::move(lcps)} {}

    // To be used with GenericCharIndexPEIndexStringSet
    // todo maybe convert this to a named static method?
    explicit StringLcpContainer(
        std::vector<Char>&& raw_strings,
        std::vector<size_t>&& lcps,
        std::vector<size_t> const& interval_sizes,
        std::vector<size_t> const& offsets
    )
        : Base{std::move(raw_strings), std::move(lcps), interval_sizes, offsets} {}

    template <typename StringSet>
    explicit StringLcpContainer(StringLcpContainer<StringSet>&& cont, size_t rank)
        : Base{cont.release_raw_strings(), cont.release_lcps(), cont.getStrings(), rank} {}

    // todo this overload is pretty horrendous
    explicit StringLcpContainer(
        std::vector<Char>&& raw_strings,
        std::vector<size_t>&& lcps,
        std::vector<size_t>&& PE_indices,
        std::vector<size_t>&& str_indices
    )
        : Base{
            std::move(raw_strings),
            std::move(lcps),
            std::move(PE_indices),
            std::move(str_indices)} {}
};

// todo same here
template <typename StringSet_>
class IndexStringLcpContainer
    : public BaseStringLcpContainer<StringSet_, IndexStringLcpContainer<StringSet_>> {
public:
    using Base = BaseStringLcpContainer<StringSet_, IndexStringLcpContainer<StringSet_>>;
    using Char = Base::Char;

    static constexpr bool isIndexed = true;

    IndexStringLcpContainer() = default;

    explicit IndexStringLcpContainer(
        std::vector<Char>&& raw_strings, std::vector<uint64_t> const& indices
    )
        : Base{std::move(raw_strings), std::vector(indices.size(), 0), indices} {}
};

} // namespace dss_schimek
