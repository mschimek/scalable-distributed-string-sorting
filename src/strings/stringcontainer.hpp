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

template <typename Member, typename InputIt>
struct Initializer {
    InputIt begin, end;
};

template <typename Member>
auto make_initializer(std::vector<typename Member::underlying_t> const& data) {
    using iterator = std::vector<typename Member::underlying_t>::const_iterator;
    return Initializer<Member, iterator>{data.begin(), data.end()};
}

namespace _internal {

template <typename StringSet, typename... Member, typename... InputIt, typename OutputIt>
void init_strings_impl(
    std::vector<typename StringSet::Char>& raw_strings,
    std::tuple<Initializer<Member, InputIt>...> initializers,
    OutputIt dest
) {
    using String = StringSet::String;

    auto const begin = raw_strings.begin(), end = raw_strings.end();

    size_t i = 0;
    for (auto char_it = begin; char_it != end; ++char_it, ++i, ++dest) {
        auto const str_begin = char_it;
        while (*char_it != 0) {
            ++char_it;
        }
        auto const str_end = char_it;

        if constexpr (StringSet::has_length) {
            size_t const str_len = std::distance(str_begin, str_end);
            *dest = std::apply(
                [=](Initializer<Member, InputIt> const&... init) {
                    return String{&*str_begin, Length{str_len}, Member{init.begin[i]}...};
                },
                initializers
            );
        } else {
            *dest = std::apply(
                [=](Initializer<Member, InputIt> const&... init) {
                    return String{&*str_begin, Member{init.begin[i]}...};
                },
                initializers
            );
        }
    }
}

template <typename StringSet, typename... Member, typename... InputIt>
void init_strings(
    std::vector<typename StringSet::Char>& raw_strings,
    std::vector<typename StringSet::String>& strings,
    Initializer<Member, InputIt>... initializers_
) {
    constexpr size_t string_length_guess = 256;

    auto size = [](auto const& initializer) -> size_t {
        return std::distance(initializer.begin, initializer.end);
    };

    std::tuple<Initializer<Member, InputIt>...> initializers{initializers_...};
    if constexpr (sizeof...(Member) == 0) {
        strings.clear();
        strings.reserve(raw_strings.size() / string_length_guess);
        init_strings_impl<StringSet>(raw_strings, initializers, std::back_inserter(strings));

    } else {
        assert(std::apply([size](auto const&... x) { return (size(x) == ...); }, initializers));
        // todo I'm unhappy with the usage of back_inserter here, but resize
        // todo gives weird potential null-dereference warnings again
        strings.clear();
        strings.reserve(size(std::get<0>(initializers)));

        assert_equal(std::ssize(strings), std::count(raw_strings.begin(), raw_strings.end(), 0));
        init_strings_impl<StringSet>(raw_strings, initializers, std::back_inserter(strings));
    }
}

} // namespace _internal


template <typename StringSet_>
class StringContainer {
public:
    using StringSet = StringSet_;
    using Char = StringSet::Char;
    using CharIterator = StringSet::CharIterator;
    using String = StringSet::String;

    using StringPtr = tlx::sort_strings_detail::StringPtr<StringSet>;

    static constexpr bool is_indexed = StringSet::is_indexed;
    static constexpr bool has_lcps = false;

    StringContainer() : raw_strings_{std::make_unique<std::vector<Char>>()} {}

    explicit StringContainer(size_t const count)
        : raw_strings_{std::make_unique<std::vector<Char>>()},
          strings_(count) {}

    StringContainer(std::vector<Char>&& raw_strings, std::vector<String>&& strings)
        : raw_strings_{std::make_unique<std::vector<Char>>(std::move(raw_strings))},
          strings_{std::move(strings)} {}

    template <typename... Member, typename... InputIt>
    explicit StringContainer(
        std::vector<Char>&& raw_strings, Initializer<Member, InputIt>... initalizers
    )
        : raw_strings_{std::make_unique<std::vector<Char>>(std::move(raw_strings))} {
        _internal::init_strings<StringSet>(*raw_strings_, strings_, initalizers...);
    }

    String operator[](size_t i) { return strings_[i]; }
    String front() { return strings_.front(); }
    String back() { return strings_.back(); }
    String* strings() { return strings_.data(); }
    size_t size() const { return strings_.size(); }
    bool empty() const { return strings_.empty(); }
    size_t char_size() const { return raw_strings_->size(); }
    std::vector<String>& get_strings() { return strings_; }
    std::vector<String> const& get_strings() const { return strings_; }
    std::vector<Char>& raw_strings() { return *raw_strings_; }
    std::vector<Char> const& raw_strings() const { return *raw_strings_; }
    std::vector<Char>&& release_raw_strings() { return std::move(*raw_strings_); }
    std::vector<String>&& release_strings() { return std::move(strings_); }

    friend void swap(StringContainer<StringSet>& lhs, StringContainer<StringSet>& rhs) {
        std::swap(lhs.raw_strings_, rhs.raw_strings_);
        std::swap(lhs.strings_, rhs.strings_);
    }

    std::vector<Char> get_raw_string(int64_t const i) {
        if (0 <= i && i < std::ssize(this)) {
            auto const& str = strings_[i];
            std::vector<Char> buf(str.length + 1, 0);
            std::copy_n(str.string, str.length, buf.begin());
            return buf;
        } else {
            return std::vector<Char>(1, 0);
        }
    }

    StringSet make_string_set() { return {strings(), strings() + size()}; }
    StringPtr make_string_ptr() { return {make_string_set()}; }
    StringPtr make_auto_ptr() { return make_string_ptr(); }

    void resize_strings(size_t const count) {
        // for some reason GCC detects a potential null dereference here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
        strings_.resize(count);
#pragma GCC diagnostic pop
    }

    void delete_raw_strings() {
        raw_strings_->clear();
        raw_strings_->shrink_to_fit();
    }

    void delete_strings() {
        strings_.clear();
        strings_.shrink_to_fit();
    }

    void delete_all() {
        delete_raw_strings();
        delete_strings();
    }

    void set(std::vector<Char>&& raw_strings) { *raw_strings_ = std::move(raw_strings); }
    void set(std::vector<String>&& strings) { strings_ = std::move(strings); }

    template <typename... Member, typename... InputIt>
    void update(std::vector<Char>&& raw_strings, Initializer<Member, InputIt>... initializers) {
        set(std::move(raw_strings));
        _internal::init_strings<StringSet>(*raw_strings_, strings_, initializers...);
    }

    void make_contiguous() {
        std::vector<Char> new_buffer;
        make_contiguous(new_buffer);
    }

    void make_contiguous(std::vector<Char>& char_buffer) {
        char_buffer.resize(make_string_set().get_sum_length() + size());

        for (auto dest = char_buffer.begin(); auto& str: strings_) {
            auto const chars = str.getChars();
            str.setChars(&*dest);
            dest = std::copy_n(chars, str.length, dest);
            *dest++ = 0;
        }

        std::swap(*raw_strings_, char_buffer);
    }

    bool is_consistent() const {
        auto const begin = &*raw_strings_->begin(), end = &*raw_strings_->end();

        auto const ss = make_string_set();
        return std::all_of(strings_.begin(), strings_.end(), [=](auto const& str) {
            return begin <= str.getChars() && str.getChars() + ss.get_length(str) < end;
        });
    }

protected:
    std::unique_ptr<std::vector<Char>> raw_strings_;
    std::vector<String> strings_;
};

template <typename StringSet_>
class StringLcpContainer : public StringContainer<StringSet_> {
public:
    using Base = StringContainer<StringSet_>;
    using Char = Base::Char;
    using String = Base::String;

    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet_, size_t>;

    static constexpr bool has_lcps = true;

    StringLcpContainer() = default;

    explicit StringLcpContainer(size_t const count) : Base{count}, lcps_(count) {}

    StringLcpContainer(
        std::vector<Char>&& raw_strings, std::vector<String>&& strings, std::vector<size_t>&& lcps
    )
        : Base{std::move(raw_strings), std::move(strings)},
          lcps_{std::move(lcps)} {}

    template <typename... Member, typename... InputIt>
    explicit StringLcpContainer(
        std::vector<Char>&& raw_strings, Initializer<Member, InputIt>... initializer
    )
        : Base{std::move(raw_strings), initializer...},
          lcps_(this->size(), 0) {}

    template <typename... Member, typename... InputIt>
    explicit StringLcpContainer(
        std::vector<Char>&& raw_strings,
        std::vector<size_t>&& lcps,
        Initializer<Member, InputIt>... initializer
    )
        : Base{std::move(raw_strings), initializer...},
          lcps_{std::move(lcps)} {
        assert_equal(this->strings_.size(), lcps_.size());
    }

    size_t* lcp_array() { return lcps_.data(); }
    std::vector<size_t>& lcps() { return lcps_; }
    std::vector<size_t> const& lcps() const { return lcps_; }
    std::vector<size_t>&& release_lcps() { return std::move(lcps_); }

    friend void swap(StringLcpContainer<StringSet_>& lhs, StringLcpContainer<StringSet_>& rhs) {
        std::swap(lhs.raw_strings_, rhs.raw_strings_);
        std::swap(lhs.strings_, rhs.strings_);
        std::swap(lhs.lcps_, rhs.lcps_);
    }

    StringLcpPtr make_string_lcp_ptr() { return {this->make_string_set(), this->lcp_array()}; }
    StringLcpPtr make_auto_ptr() { return make_string_lcp_ptr(); }

    void resize_strings(size_t const count) {
        Base::resize_strings(count);
        this->lcps_.resize(count);
    }

    // todo rename to set_lcps, set_strings, ...
    using Base::set;
    void set(std::vector<size_t>&& lcps) { lcps_ = std::move(lcps); }

    template <typename... Member, typename... InputIt>
    void update(std::vector<Char>&& raw_strings, Initializer<Member, InputIt>... initializers) {
        Base::update(std::move(raw_strings), initializers...);
        lcps_.resize(this->size(), 0);
    }

    void delete_lcps() {
        lcps_.clear();
        lcps_.shrink_to_fit();
    }

    void delete_all() {
        this->delete_raw_strings();
        this->delete_strings();
        delete_lcps();
    }

    template <typename StringSet, typename LcpIt>
    void extend_prefix(StringSet const& ss, LcpIt const first_lcp, LcpIt const last_lcp) {
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
        for (auto it = ss.begin(); it != ss.end(); ++it, ++lcp_it) {
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

private:
    std::vector<size_t> lcps_;
};

} // namespace dss_schimek


namespace dss_mehnert {

using dss_schimek::Initializer;
using dss_schimek::make_initializer;
using dss_schimek::StringContainer;
using dss_schimek::StringLcpContainer;

} // namespace dss_mehnert
