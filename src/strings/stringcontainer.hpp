// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <stdint.h>
#include <tlx/logger.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "strings/stringset.hpp"

namespace dss_schimek {

template <typename StringSet>
class InitPolicy {
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static constexpr size_t approx_string_length = 10;

public:
    std::vector<String> init_strings(std::vector<Char>& raw_strings) {
        std::vector<String> strings;
        size_t approx_string_size = raw_strings.size() / approx_string_length;
        strings.reserve(approx_string_size);
        for (size_t i = 0; i < raw_strings.size(); ++i) {
            strings.emplace_back(raw_strings.data() + i);
            while (raw_strings[i] != 0) ++i;
        }
        return strings;
    }
};

template <typename CharType>
class InitPolicy<GenericCharLengthStringSet<CharType>> {
    using StringSet = GenericCharLengthStringSet<CharType>;
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static constexpr size_t approx_string_length = 10;

public:
    std::vector<String> init_strings(std::vector<Char>& raw_strings) {
        // size_t stringNum = 0;
        // for (size_t i = 0; i < raw_strings.size(); ++i) {
        //    while (raw_strings[i] != 0)
        //        ++i;
        //    ++stringNum;
        //}

        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100); // just a guess
        for (size_t i = 0; i < raw_strings.size(); ++i) {
            strings.emplace_back(raw_strings.data() + i, i);
            while (raw_strings[i] != 0) ++i;
            strings.back().length = i - strings.back().length;
        }
        return strings;
    }
};

template <typename CharType>
class InitPolicy<GenericCharLengthIndexStringSet<CharType>> {
    using StringSet = GenericCharLengthIndexStringSet<CharType>;
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static constexpr size_t approx_string_length = 10;

public:
    std::vector<String> init_strings(std::vector<Char>& raw_strings) {
        // size_t stringNum = 0;
        // for (size_t i = 0; i < raw_strings.size(); ++i) {
        //    while (raw_strings[i] != 0)
        //        ++i;
        //    ++stringNum;
        //}

        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100); // just a guess
        size_t curString = 0;
        for (size_t i = 0; i < raw_strings.size(); ++i) {
            strings.emplace_back(raw_strings.data() + i, i, curString);
            while (raw_strings[i] != 0) ++i;
            strings.back().length = i - strings.back().length;
            ++curString;
        }
        return strings;
    }
    std::vector<String>
    init_strings(std::vector<Char>& raw_strings, std::vector<uint64_t> const& stringIndices) {
        // size_t stringNum = 0;
        // for (size_t i = 0; i < raw_strings.size(); ++i) {
        //    while (raw_strings[i] != 0)
        //        ++i;
        //    ++stringNum;
        //}

        std::vector<String> strings;
        strings.reserve(raw_strings.size() / 100); // just a guess
        size_t curString = 0;
        for (size_t i = 0; i < raw_strings.size(); ++i) {
            strings.emplace_back(raw_strings.data() + i, i, stringIndices[curString]);
            while (raw_strings[i] != 0) ++i;
            strings.back().length = i - strings.back().length;
            ++curString;
        }
        return strings;
    }
};

template <typename CharType>
class InitPolicy<GenericCharIndexPEIndexStringSet<CharType>> {
    using StringSet = GenericCharIndexPEIndexStringSet<CharType>;
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    static constexpr size_t approx_string_length = 10;

public:
    std::vector<String> init_strings(
        std::vector<Char>& raw_strings,
        std::vector<size_t> const& intervalSizes,
        std::vector<size_t> const& offsets
    ) {
        std::vector<String> strings;
        size_t numStrings =
            std::accumulate(intervalSizes.begin(), intervalSizes.end(), static_cast<size_t>(0u));
        strings.reserve(numStrings);
        size_t curOffset = 0;

        for (size_t curPEIndex = 0; curPEIndex < intervalSizes.size(); ++curPEIndex) {
            for (size_t stringIndex = 0; stringIndex < intervalSizes[curPEIndex]; ++stringIndex) {
                strings.emplace_back(
                    raw_strings.data() + curOffset,
                    offsets[curPEIndex] + stringIndex,
                    curPEIndex
                );
                while (raw_strings[curOffset] != 0) ++curOffset;
                ++curOffset;
            }
        }
        return strings;
    }

    std::vector<String> init_strings(
        std::vector<Char>& raw_strings,
        std::vector<size_t>&& PE_indices,
        std::vector<size_t>&& str_indices
    ) {
        std::vector<String> strings(PE_indices.size());
        for (size_t offset = 0, idx = 0; auto& str: strings) {
            str = {raw_strings.data() + offset, str_indices[idx], PE_indices[idx]};

            while (raw_strings[offset] != 0) ++offset;
            ++offset, ++idx;
        }
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
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<size_t>& lcps() { return lcps_; }
    std::vector<size_t>& savedLcps() { return savedLcps_; }
    std::vector<size_t> const& lcps() const { return lcps_; }
    size_t* lcp_array() { return lcps_.data(); }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }

    void saveLcps() { savedLcps_ = lcps_; }

    template <typename StringSet>
    void extendPrefix(StringSet ss, std::vector<size_t> const& lcps) {
        using String = typename StringSet::String;
        using Char = typename StringSet::Char;
        using CharIt = typename StringSet::CharIterator;
        const size_t initialPrefixLengthGuess = 10000u;

        if (ss.size() == 0u)
            return;

        const size_t L = std::accumulate(lcps.begin(), lcps.end(), static_cast<size_t>(0u));
        std::vector<Char> extendedRawStrings(char_size() + L);
        // extendedRawStrings.reserve(char_size() + L);

        std::vector<Char> curPrefix;
        curPrefix.reserve(initialPrefixLengthGuess);

        String curString = ss[ss.begin()];
        CharIt startCurString = ss.get_chars(curString, 0);
        size_t stringLength = ss.get_length(curString) + 1;
        size_t curPos = 0u;
        std::copy(
            startCurString,
            startCurString + stringLength,
            extendedRawStrings.begin() + curPos
        );
        curPos += stringLength;
        strings_[0].length = stringLength - 1;
        // std::copy_n(startCurString, stringLength,
        //    std::back_inserter(extendedRawStrings));
        std::vector<size_t> offset(ss.size(), 0);
        for (size_t i = 1; i < ss.size(); ++i) {
            offset[i] = curPos;
            int64_t lcp_diff = lcps[i] - lcps[i - 1];
            uint64_t totalStringLength = 0;
            if (lcp_diff < 0) {
                curPrefix.erase(curPrefix.end() + lcp_diff, curPrefix.end());
                lcp_diff = 0;

                // while (lcp_diff++ < 0)
                //    curPrefix.pop_back();
            } else {
                String prevString = ss[ss.begin() + i - 1];
                CharIt commonPrefix = ss.get_chars(prevString, 0);
                for (size_t j = 0; j < static_cast<size_t>(lcp_diff);
                     ++j) // lcp_diff > 0 see if-branch
                    curPrefix.push_back(*(commonPrefix + j));
            }
            std::copy(curPrefix.begin(), curPrefix.end(), extendedRawStrings.begin() + curPos);
            curPos += curPrefix.size();
            totalStringLength += curPrefix.size();
            // std::copy_n(curPrefix.begin(), curPrefix.size(),
            //    std::back_inserter(extendedRawStrings));

            String curString = ss[ss.begin() + i];
            CharIt startCurString = ss.get_chars(curString, 0);
            size_t stringLength = ss.get_length(curString) + 1;
            std::copy(
                startCurString,
                startCurString + stringLength,
                extendedRawStrings.begin() + curPos
            );
            curPos += stringLength;
            totalStringLength += stringLength;
            strings_[i].length = totalStringLength - 1;

            // std::copy_n(startCurString, stringLength,
            //    std::back_inserter(extendedRawStrings));
            // curPos += stringLength;
        }
        *raw_strings_ = std::move(extendedRawStrings);
        for (size_t i = 0; i < ss.size(); ++i) {
            strings_[i].string = raw_strings_->data() + offset[i];
        }

        // update(std::move(extendedRawStrings));
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

using StringLcpContainerUChar = StringLcpContainer<UCharStringSet>;

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

    // IndexStringLcpContainer(std::vector<Char>&& raw_strings)
    //    : raw_strings_(
    //          std::make_unique<std::vector<Char>>(std::move(raw_strings))) {
    //    update_strings();
    //    lcps_.resize(size(), 0);
    //}

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
class StringContainer : private InitPolicy<StringSet_> {
public:
    using StringSet = StringSet_;
    using Char = typename StringSet::Char;
    using CharIterator = typename StringSet::CharIterator;
    using String = typename StringSet::String;
    static constexpr bool isIndexed = false;

    StringContainer() : raw_strings_(std::make_unique<std::vector<Char>>()), strings_() {}

    explicit StringContainer(std::vector<Char>&& raw_strings)
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))) {
        update_strings();
    }

    String operator[](size_t i) { return strings_[i]; }
    String front() { return strings_.front(); }
    String back() { return strings_.back(); }
    String* strings() { return strings_.data(); }
    std::vector<String>& getStrings() { return strings_; }
    size_t size() const { return strings_.size(); }
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }
    std::vector<Char>&& releaseRawStrings() { return std::move(*raw_strings_); }

    std::vector<unsigned char> getRawString(int64_t i) {
        if (i < 0 || static_cast<uint64_t>(i) > size())
            return std::vector<unsigned char>(1, 0);

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

    void set(std::vector<Char>&& raw_strings) { *raw_strings_ = std::move(raw_strings); }
    void set(std::vector<String>&& strings) { strings_ = std::move(strings); }

    bool operator==(StringLcpContainer<StringSet_> const& other) {
        return (raw_strings() == other.raw_strings());
    }

    void update(std::vector<Char>&& raw_strings) {
        set(std::move(raw_strings));
        update_strings();
    }

public:
    size_t sumOfCapacities() {
        return raw_strings_->capacity() * sizeof(Char) + strings_.capacity() * sizeof(String);
    }
    size_t sumOfSizes() {
        return raw_strings_->size() * sizeof(Char) + strings_.size() * sizeof(String);
    }

protected:
    static constexpr size_t approx_string_length = 10;
    std::unique_ptr<std::vector<Char>> raw_strings_;
    std::vector<String> strings_; // strings

    void update_strings() { strings_ = InitPolicy<StringSet>::init_strings(*raw_strings_); }
};

template <typename StringSet_>
class IndexStringContainer : private InitPolicy<StringSet_> {
public:
    using StringSet = StringSet_;
    using Char = typename StringSet::Char;
    using CharIterator = typename StringSet::CharIterator;
    using String = typename StringSet::String;
    static constexpr bool isIndexed = true;

    IndexStringContainer() : raw_strings_(std::make_unique<std::vector<Char>>()), strings_() {}

    // IndexStringLcpContainer(std::vector<Char>&& raw_strings)
    //    : raw_strings_(
    //          std::make_unique<std::vector<Char>>(std::move(raw_strings))) {
    //    update_strings();
    //    lcps_.resize(size(), 0);
    //}

    explicit IndexStringContainer(std::vector<Char>&& raw_strings, std::vector<uint64_t>& indices)
        : raw_strings_(std::make_unique<std::vector<Char>>(std::move(raw_strings))) {
        update_strings(indices);
    }

    String operator[](size_t i) { return strings_[i]; }
    String front() { return strings_.front(); }
    String back() { return strings_.back(); }
    String* strings() { return strings_.data(); }
    std::vector<String>& getStrings() { return strings_; }
    size_t size() const { return strings_.size(); }
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }
    std::vector<unsigned char> getRawString(int64_t i) {
        if (i < 0 || static_cast<uint64_t>(i) > size())
            return std::vector<unsigned char>(1, 0);

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

    void set(std::vector<Char>&& raw_strings) { *raw_strings_ = std::move(raw_strings); }
    void set(std::vector<String>&& strings) { strings_ = std::move(strings); }

    bool operator==(StringLcpContainer<StringSet_> const& other) {
        return (raw_strings() == other.raw_strings());
    }

    void update(std::vector<Char>&& raw_strings) {
        set(std::move(raw_strings));
        update_strings();
    }
    void update(std::vector<Char>&& raw_strings, std::vector<uint64_t> const& indices) {
        set(std::move(raw_strings));
        update_strings(indices);
    }

public:
    size_t sumOfCapacities() {
        return raw_strings_->capacity() * sizeof(Char) + strings_.capacity() * sizeof(String);
    }
    size_t sumOfSizes() {
        return raw_strings_->size() * sizeof(Char) + strings_.size() * sizeof(String);
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

protected:
    static constexpr size_t approx_string_length = 10;
    std::unique_ptr<std::vector<Char>> raw_strings_;
    std::vector<String> strings_; // strings

    void update_strings() { strings_ = InitPolicy<StringSet>::init_strings(*raw_strings_); }

    void update_strings(std::vector<uint64_t> const& indices) {
        strings_ = InitPolicy<StringSet>::init_strings(*raw_strings_, indices);
    }
};

} // namespace dss_schimek
