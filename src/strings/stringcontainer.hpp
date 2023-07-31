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

namespace internal {

template <typename OutIt, typename Char, typename Init>
void init_str_len(OutIt out, std::vector<Char>& raw_strings, Init init) {
    for (auto it = raw_strings.begin(); it != raw_strings.end(); ++it, ++out) {
        auto begin = it;
        while (*it != 0)
            ++it;

        *out = init(&(*begin), static_cast<size_t>(std::distance(begin, it)));
    }
}

} // namespace internal

template <typename StringSet>
class InitPolicy {
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

public:
    std::vector<String> init_strings(std::vector<Char>& raw_strings) {
        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100);

        auto init = [](auto str, auto) { return String{str}; };
        internal::init_str_len(std::back_inserter(strings), raw_strings, init);
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
        internal::init_str_len(std::back_inserter(strings), raw_strings, init);
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
        internal::init_str_len(std::back_inserter(strings), raw_strings, init);
        return strings;
    }

    std::vector<String>
    init_strings(std::vector<Char>& raw_strings, std::vector<uint64_t> const& string_idxs) {
        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100);

        auto init = [index_it = string_idxs.cbegin()](auto str, auto len) mutable {
            return String{str, Length{len}, Index{*(index_it++)}};
        };
        internal::init_str_len(std::back_inserter(strings), raw_strings, init);
        return strings;
    }
};

// just a helper for the template specialization below
template <typename... Length_>
struct StringIndexPEIndex_ {
    template <typename String>
    using type = StringData<String, Length_..., StringIndex, PEIndex>;
};

// this specialization may be going a bit far in the pursuit of avoiding code duplication ^^
template <typename CharType, typename... Length_>
class InitPolicy<GenericStringSet<CharType, StringIndexPEIndex_<Length_...>::template type>> {
    using StringSet = GenericStringSet<CharType, StringIndexPEIndex_<Length_...>::template type>;
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

public:
    std::vector<String> init_strings(
        std::vector<Char>& raw_strings,
        std::vector<size_t> const& intervals,
        std::vector<size_t> const& offsets
    ) {
        auto num_strs = std::accumulate(intervals.begin(), intervals.end(), size_t{0});
        std::vector<String> strings(num_strs);

        auto init = [](auto str, auto len) { return String{str, Length_{len}...}; };
        internal::init_str_len(strings.begin(), raw_strings, init);

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

        auto PE_it = PE_indices.cbegin(), str_it = str_indices.cbegin();
        auto init = [&](auto str, auto len) mutable {
            return String{str, Length_{len}..., PEIndex{*(PE_it++)}, StringIndex{*(str_it++)}};
        };
        internal::init_str_len(strings.begin(), raw_strings, init);
        return strings;
    }
};

// TODO Think about better design (subclasses?) for different StringSets +
// additional data like the saved LCP values when doing prefix compression
template <typename StringSet_>
class StringLcpContainer : private InitPolicy<StringSet_> {
public:
    using StringSet = StringSet_;
    using Char = typename StringSet::Char;
    using CharIterator = typename StringSet::CharIterator;
    using String = typename StringSet::String;

    StringLcpContainer()
        : raw_strings_(std::make_unique<std::vector<Char>>()),
          strings_(),
          lcps_(),
          savedLcps_() {}

    StringLcpContainer(std::vector<Char>&& raw_strings)
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))) {
        update_strings();
        lcps_.resize(size(), 0);
    }

    explicit StringLcpContainer(std::vector<Char>&& raw_strings, std::vector<size_t>&& lcp)
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))),
          savedLcps_() {
        update_strings();
        lcps_ = std::move(lcp);
    }

    // To be used with GenericCharIndexPEIndexStringSet
    explicit StringLcpContainer(
        std::vector<Char>&& raw_strings,
        std::vector<size_t>&& lcp,
        std::vector<size_t> const& intervalSizes,
        std::vector<size_t> const& offsets
    )
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))),
          savedLcps_() {
        update_strings(intervalSizes, offsets);
        lcps_ = std::move(lcp);
    }

    // todo this overload is pretty horrendous
    explicit StringLcpContainer(
        std::vector<Char>&& raw_strings,
        std::vector<size_t>&& lcp,
        std::vector<size_t>&& PE_indices,
        std::vector<size_t>&& str_indices
    )
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))),
          savedLcps_() {
        update_strings(std::move(PE_indices), std::move(str_indices));
        lcps_ = std::move(lcp);
    }

    void orderRawStrings() {
        auto orderedRawStrings = new std::vector<unsigned char>(char_size());
        uint64_t curPos = 0;
        for (size_t i = 0; i < size(); ++i) {
            auto chars = strings_[i].getChars();
            auto length = strings_[i].getLength();
            auto curAdress = orderedRawStrings->data() + curPos;
            std::copy_n(chars, length + 1, curAdress);
            strings_[i].setChars(curAdress);
            curPos += length + 1;
        }
        raw_strings_.reset(orderedRawStrings);
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

    String operator[](size_t i) { return strings_[i]; }
    String front() { return strings_.front(); }
    String back() { return strings_.back(); }
    String* strings() { return strings_.data(); }
    std::vector<String>& getStrings() { return strings_; }
    size_t size() const { return strings_.size(); }
    bool empty() const { return strings_.empty(); }
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<size_t>& lcps() { return lcps_; }
    std::vector<size_t>& savedLcps() { return savedLcps_; }
    std::vector<size_t> const& lcps() const { return lcps_; }
    size_t* lcp_array() { return lcps_.data(); }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }

    void saveLcps() { savedLcps_ = lcps_; }

    template <typename StringSet, typename LcpIt>
    void extendPrefix(StringSet ss, LcpIt first_lcp, LcpIt last_lcp) {
        assert(std::distance(first_lcp, last_lcp) == std::ssize(ss));

        using Char = typename StringSet::Char;

        if (ss.size() == 0u)
            return;

        size_t L = std::accumulate(first_lcp, last_lcp, size_t{0});
        std::vector<Char> raw_strings(char_size() + L);

        auto& fst_str = ss[ss.begin()];
        auto const fst_str_begin = ss.get_chars(fst_str, 0);
        auto const fst_str_len = ss.get_length(fst_str) + 1;
        std::copy(fst_str_begin, fst_str_begin + fst_str_len, raw_strings.begin());

        fst_str.string = raw_strings.data();
        fst_str.length = fst_str_len - 1;

        auto prev_chars = raw_strings.begin();
        auto curr_chars = raw_strings.begin() + fst_str_len;
        for (size_t i = 1; i < ss.size(); ++i) {
            // copy common prefix from previous string
            size_t lcp = *(++first_lcp);
            std::copy(prev_chars, prev_chars + lcp, curr_chars);

            // copy remaining (distinct) characters
            auto& curr_str = ss[ss.begin() + i];
            auto const curr_str_begin = ss.get_chars(curr_str, 0);
            auto const curr_str_len = ss.get_length(curr_str) + 1;
            std::copy(curr_str_begin, curr_str_begin + curr_str_len, curr_chars + lcp);

            curr_str.string = &(*curr_chars);
            curr_str.length = curr_str_len + lcp - 1;

            prev_chars = curr_chars;
            curr_chars += curr_str_len + lcp;
        }

        *raw_strings_ = std::move(raw_strings);
    }

    StringSet make_string_set() { return StringSet(strings(), strings() + size()); }

    tlx::sort_strings_detail::StringPtr<StringSet> make_string_ptr() {
        return tlx::sort_strings_detail::StringPtr(make_string_set());
    }

    tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t> make_string_lcp_ptr() {
        return tlx::sort_strings_detail::StringLcpPtr(make_string_set(), lcp_array());
    }

    void deleteRawStrings() {
        raw_strings_->clear();
        raw_strings_->shrink_to_fit();
    }

    void deleteStrings() {
        strings_.clear();
        strings_.shrink_to_fit();
    }

    void deleteLcps() {
        lcps_.clear();
        lcps_.shrink_to_fit();
    }

    void deleteSavedLcps() {
        savedLcps_.clear();
        savedLcps_.shrink_to_fit();
    }

    void deleteAll() {
        deleteRawStrings();
        deleteStrings();
        deleteLcps();
        deleteSavedLcps();
    }

    void set(std::vector<Char>&& raw_strings) { *raw_strings_ = std::move(raw_strings); }
    void set(std::vector<String>&& strings) { strings_ = std::move(strings); }
    void set(std::vector<size_t>&& lcps) { lcps_ = std::move(lcps); }
    void setSavedLcps(std::vector<size_t>&& savedLcps) { savedLcps_ = std::move(savedLcps); }

    bool operator==(StringLcpContainer<StringSet_> const& other) {
        return (raw_strings() == other.raw_strings()) && (lcps() == other.lcps());
    }

    void update(std::vector<Char>&& raw_strings) {
        set(std::move(raw_strings));
        update_strings();
        if (lcps_.size() != size())
            lcps_.resize(size(), 0);
    }

    bool is_consistent() {
        if (lcps_.size() != strings_.size()) {
            LOG1 << "lcps.size() = " << lcps_.size() << " != " << strings_.size()
                 << " = strings.size()";
            return false;
        }

        return std::all_of(strings_.begin(), strings_.end(), [this](const String str) -> bool {
            if (str < raw_strings_.data() || str > raw_strings_.data() + raw_strings_.size())
                return false;
            if (str == raw_strings_.data())
                return true;
            return *(str - 1) == 0;
        });
    }

public:
    size_t sumOfCapacities() {
        return raw_strings_->capacity() * sizeof(Char) + strings_.capacity() * sizeof(String)
               + lcps_.capacity() * sizeof(size_t) + savedLcps_.capacity() * sizeof(size_t);
    }
    size_t sumOfSizes() {
        return raw_strings_->size() * sizeof(Char) + strings_.size() * sizeof(String)
               + lcps_.size() * sizeof(size_t) + savedLcps_.size() * sizeof(size_t);
    }

protected:
    static constexpr size_t approx_string_length = 10;
    std::unique_ptr<std::vector<Char>> raw_strings_;
    std::vector<String> strings_;   // strings
    std::vector<size_t> lcps_;      // lcp-values
    std::vector<size_t> savedLcps_; // only used for prefix compression, ->
                                    // lcp-values received from other PEs before
                                    // merging TODO think about better structure

    void update_strings() { strings_ = InitPolicy<StringSet>::init_strings(*raw_strings_); }

    void
    update_strings(std::vector<size_t> const& intervalSizes, std::vector<size_t> const& offsets) {
        strings_ = InitPolicy<StringSet>::init_strings(*raw_strings_, intervalSizes, offsets);
    }

    void update_strings(std::vector<size_t>&& PE_indices, std::vector<size_t>&& str_indices) {
        strings_ = InitPolicy<StringSet>::init_strings(
            *raw_strings_,
            std::move(PE_indices),
            std::move(str_indices)
        );
    }
};

template <typename StringSet_>
class IndexStringLcpContainer : private InitPolicy<StringSet_> {
public:
    using StringSet = StringSet_;
    using Char = typename StringSet::Char;
    using CharIterator = typename StringSet::CharIterator;
    using String = typename StringSet::String;

    IndexStringLcpContainer()
        : raw_strings_(std::make_unique<std::vector<Char>>()),
          strings_(),
          lcps_() {}

    explicit IndexStringLcpContainer(
        std::vector<Char>&& raw_strings, std::vector<uint64_t>& indices
    )
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))) {
        update_strings(indices);
        lcps_.resize(size(), 0);
    }

    String operator[](size_t i) { return strings_[i]; }
    String front() { return strings_.front(); }
    String back() { return strings_.back(); }
    String* strings() { return strings_.data(); }
    std::vector<String>& getStrings() { return strings_; }
    size_t size() const { return strings_.size(); }
    bool empty() const { return strings_.empty(); }
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<size_t>& lcps() { return lcps_; }
    std::vector<size_t> const& lcps() const { return lcps_; }
    size_t* lcp_array() { return lcps_.data(); }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }

    StringSet make_string_set() { return StringSet(strings(), strings() + size()); }

    tlx::sort_strings_detail::StringPtr<StringSet> make_string_ptr() {
        return tlx::sort_strings_detail::StringPtr(make_string_set());
    }

    tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t> make_string_lcp_ptr() {
        return tlx::sort_strings_detail::StringLcpPtr(make_string_set(), lcp_array());
    }

    void deleteRawStrings() {
        raw_strings_->clear();
        raw_strings_->shrink_to_fit();
    }

    void deleteStrings() {
        strings_.clear();
        strings_.shrink_to_fit();
    }

    void deleteLcps() {
        lcps_.clear();
        lcps_.shrink_to_fit();
    }

    void deleteAll() {
        deleteRawStrings();
        deleteStrings();
        deleteLcps();
    }

    void set(std::vector<Char>&& raw_strings) { *raw_strings_ = std::move(raw_strings); }
    void set(std::vector<String>&& strings) { strings_ = std::move(strings); }
    void set(std::vector<size_t>&& lcps) { lcps_ = std::move(lcps); }

    bool operator==(StringLcpContainer<StringSet_> const& other) {
        return (raw_strings() == other.raw_strings()) && (lcps() == other.lcps());
    }

    void update(std::vector<Char>&& raw_strings) {
        set(std::move(raw_strings));
        update_strings();
        if (lcps_.size() != size())
            lcps_.resize(size(), 0);
    }

public:
    size_t sumOfCapacities() {
        return raw_strings_->capacity() * sizeof(Char) + strings_.capacity() * sizeof(String)
               + lcps_.capacity() * sizeof(size_t);
    }
    size_t sumOfSizes() {
        return raw_strings_->size() * sizeof(Char) + strings_.size() * sizeof(String)
               + lcps_.size() * sizeof(size_t);
    }

protected:
    static constexpr size_t approx_string_length = 10;
    std::unique_ptr<std::vector<Char>> raw_strings_;
    std::vector<String> strings_; // strings
    std::vector<size_t> lcps_;    // lcp-values

    void update_strings() { strings_ = InitPolicy<StringSet>::init_strings(*raw_strings_); }

    void update_strings(std::vector<uint64_t> const& indices) {
        strings_ = InitPolicy<StringSet>::init_strings(*raw_strings_, indices);
    }
};

template <typename StringSet_>
class BaseStringContainer : private InitPolicy<StringSet_> {
public:
    using StringSet = StringSet_;
    using Char = typename StringSet::Char;
    using CharIterator = typename StringSet::CharIterator;
    using String = typename StringSet::String;
    static constexpr bool isIndexed = false;

    BaseStringContainer() : raw_strings_{std::make_unique<std::vector<Char>>()} {}

    String operator[](size_t i) { return strings_[i]; }
    String front() { return strings_.front(); }
    String back() { return strings_.back(); }
    String* strings() { return strings_.data(); }
    std::vector<String>& getStrings() { return strings_; }
    size_t size() const { return strings_.size(); }
    bool empty() const { return strings_.empty(); }
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }
    std::vector<Char>&& releaseRawStrings() { return std::move(*raw_strings_); }

    std::vector<unsigned char> getRawString(int64_t i) {
        if (i < 0 || static_cast<uint64_t>(i) > size())
            return std::vector<unsigned char>(1, 0);

        // todo should use StringSet::getLength here
        auto const length = strings_[i].length + 1;
        std::vector<unsigned char> rawString(length);
        std::copy(strings_[i].string, strings_[i].string + length, rawString.begin());
        return rawString;
    }

    StringSet make_string_set() { return StringSet(strings(), strings() + size()); }

    tlx::sort_strings_detail::StringPtr<StringSet> make_string_ptr() {
        return tlx::sort_strings_detail::StringPtr(make_string_set());
    }

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

    // todo make this private??
    void set(std::vector<Char>&& raw_strings) { *raw_strings_ = std::move(raw_strings); }
    void set(std::vector<String>&& strings) { strings_ = std::move(strings); }

    bool operator==(StringLcpContainer<StringSet_> const& other) {
        return raw_strings() == other.raw_strings();
    }

    void update(std::vector<Char>&& raw_strings) {
        set(std::move(raw_strings));
        update_strings();
    }

    void orderRawStrings() {
        auto orderedRawStrings = new std::vector<unsigned char>(char_size());
        uint64_t curPos = 0;
        for (size_t i = 0; i < size(); ++i) {
            auto chars = strings_[i].getChars();
            auto length = strings_[i].getLength();
            auto curAdress = orderedRawStrings->data() + curPos;
            std::copy_n(chars, length + 1, curAdress);
            strings_[i].setChars(curAdress);
            curPos += length + 1;
        }
        raw_strings_.reset(orderedRawStrings);
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

    size_t sumOfCapacities() {
        return raw_strings_->capacity() * sizeof(Char) + strings_.capacity() * sizeof(String);
    }

    size_t sumOfSizes() {
        return raw_strings_->size() * sizeof(Char) + strings_.size() * sizeof(String);
    }

protected:
    static constexpr size_t approx_string_length = 10;
    std::unique_ptr<std::vector<Char>> raw_strings_;
    std::vector<String> strings_;

    explicit BaseStringContainer(std::vector<Char>&& raw_strings, auto&&... args)
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))),
          strings_(InitPolicy<StringSet>::init_strings(
              *raw_strings_, std::forward<decltype(args)>(args)...
          )) {}

    void update_strings(auto&&... args) {
        strings_ = InitPolicy<StringSet>::init_strings(
            *raw_strings_,
            std::forward<decltype(args)>(args)...
        );
    }
};

template <typename StringSet_>
class StringContainer : public BaseStringContainer<StringSet_> {
public:
    using StringSet = BaseStringContainer<StringSet_>::StringSet;
    using Char = BaseStringContainer<StringSet_>::Char;
    using CharIterator = BaseStringContainer<StringSet_>::CharIterator;
    using String = BaseStringContainer<StringSet_>::String;

    static constexpr bool isIndexed = false;

    StringContainer() = default;

    explicit StringContainer(std::vector<Char>&& raw_strings)
        : BaseStringContainer<StringSet_>{std::move(raw_strings)} {};
};

template <typename StringSet_>
class IndexStringContainer : public BaseStringContainer<StringSet_> {
public:
    using StringSet = BaseStringContainer<StringSet_>::StringSet;
    using Char = BaseStringContainer<StringSet_>::Char;
    using CharIterator = BaseStringContainer<StringSet_>::CharIterator;
    using String = BaseStringContainer<StringSet_>::String;

    static constexpr bool isIndexed = true;

    IndexStringContainer() = default;

    explicit IndexStringContainer(
        std::vector<Char>&& raw_strings, std::vector<uint64_t> const& indices
    )
        : BaseStringContainer<StringSet_>{std::move(raw_strings), indices} {};

    void update(std::vector<Char>&& raw_strings, std::vector<uint64_t> indices) {
        this->set(std::move(raw_strings));
        this->update_strings(indices);
    }
};

} // namespace dss_schimek
