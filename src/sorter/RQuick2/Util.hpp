#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

#include <RBC.hpp>

#include "kamping/mpi_datatype.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "strings/stringtools.hpp"

namespace RQuick {
namespace _internal {

template <typename StringPtr>
struct RawStrings {
    std::vector<typename StringPtr::StringSet::Char> raw_strs;

    template <typename Container>
    RawStrings(Container& container) : raw_strs{container.release_raw_strings()} {}
};

template <typename StringPtr>
struct Indices {
    std::vector<uint64_t> indices;

    template <typename Container>
    Indices(Container& container) : indices(container.size()) {
        auto const begin = container.strings().begin(), end = container.strings().end();
        std::transform(begin, end, [](auto const& str) { return str.index; });
    }
};

template <typename StringPtr>
struct LcpValues {
    std::vector<uint64_t> lcps;

    template <typename Container>
    LcpValues(Container& container) : lcps{container.release_lcps()} {}
};

template <typename StringPtr_, template <typename> typename... Members>
class DataMembers : private Members<StringPtr_>... {
public:
    using StringPtr = StringPtr_;
    using StringSet = StringPtr::StringSet;
    using String = StringPtr::StringSet::String;
    using Char = StringPtr::StringSet::Char;

    template <typename Container>
    DataMembers(Container&& container) : Members<StringPtr>{container}... {}

    size_t get_num_strings() const {
        if constexpr (has_index) {
            return this->indices.size();
        } else {
            return std::count(this->raw_strs.begin(), this->raw_strs.end(), '\0');
        }
    }

    void write(StringPtr& strptr) {
        auto const& ss = strptr.active();
        this->raw_strs.resize(ss.get_sum_length());
        if constexpr (has_index) {
            this->indices.resize(strptr.size());
        }

        auto char_dest = this->raw_strs.begin();
        for (size_t i = 0; auto const& str: ss) {
            auto const old_chars = str.getChars();
            str.setChars(&*char_dest);
            char_dest = std::copy_n(old_chars, str.getLength(), char_dest);
            *char_dest++ = 0;

            if constexpr (has_index) {
                this->indices[i] = str.getIndex();
            }
        }
    }

    void read_into(StringPtr& strptr) const {
        auto& ss = strptr.active();

        auto str_begin = this->raw_strs.begin();
        for (size_t i = 0; auto& str: std::span{ss.begin(), ss.end()}) {
            auto const str_end = std::find(str_begin, this->raw_strs.end(), '\0');
            auto const str_len = std::distance(str_begin, str_end) + 1;

            // todo consider lcp values if present
            if constexpr (has_index) {
                auto const str_idx = this->indices[i];
                str = {&*str_begin, dss_schimek::Length{str_len}, dss_schimek::Index{str_idx}};
            } else {
                str = {&*str_begin, dss_schimek::Length{str_len}};
            }

            assert(str_end != this->raw_strs.end());
            str_begin = str_end + 1, ++i;
        }
    }

    void send(int const dest, int const tag, RBC::Comm const& comm) const {
        auto mpi_type = kamping::mpi_datatype<Char>();
        RBC::Send(this->raw_strs.data(), this->raw_strs.size(), mpi_type, dest, tag, comm);

        if constexpr (has_index) {
            auto mpi_type = kamping::mpi_datatype<uint64_t>();
            RBC::Send(this->indices.data(), this->indices.size(), mpi_type, dest, tag, comm);
        }
    }

    void recv(int const src, int const tag, RBC::Comm const& comm) {
        auto char_type = kamping::mpi_datatype<Char>();
        MPI_Status status;
        RBC::Probe(src, tag, comm, &status);

        int count_chars = 0;
        MPI_Get_count(&status, char_type, &count_chars);
        this->raw_strs.resize(count_chars);
        RBC::Recv(this->raw_strs.data(), count_chars, char_type, src, tag, comm, MPI_STATUS_IGNORE);

        if constexpr (has_index) {
            auto idx_type = kamping::mpi_datatype<uint64_t>();
            RBC::Probe(src, tag, comm, &status);

            int count = 0;
            MPI_Get_count(&status, idx_type, &count_chars);

            auto& indices = this->indices;
            indices.resize(count);
            RBC::Recv(indices.data(), count_chars, idx_type, src, tag, comm, MPI_STATUS_IGNORE);
        }
    }

    void sendrecv(
        Indices<StringPtr>& recv,
        int const recv_cnt,
        int const partner,
        int const tag,
        RBC::Comm const& comm
    ) {
        auto size_type = kamping::mpi_datatype<size_t>();
        auto send_cnt = this->raw_strs.size(), char_cnt = 0;
        // clang-format off
        RBC::Sendrecv(&send_cnt, 1, size_type, partner, tag,
                      &char_cnt, 1, size_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);
        // clang-format on

        auto char_type = kamping::mpi_datatype<Char>();
        recv.raw_strs.resize(char_cnt);
        // clang-format off
        RBC::Sendrecv(this->raw_strs.data(), this->raw_strs.size(), char_type, partner, tag,
                      recv.indices.data(),   recv_cnt,              char_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);
        // clang-format on

        if constexpr (has_index) {
            recv.indices.resize(recv_cnt);
            auto mpi_type = kamping::mpi_datatype<uint64_t>();
            // clang-format off
            RBC::Sendrecv(this->indices.data(), this->indices.size(), mpi_type, partner, tag,
                          recv.indices.data(),  recv_cnt,             mpi_type, partner, tag,
                          comm, MPI_STATUS_IGNORE);
            // clang-format on
        }
    }

    String bcast_single(int const root, RBC::Comm const& comm) {
        auto const size_type = kamping::mpi_datatype<size_t>();
        size_t size = this->raw_strs.size();
        RBC::Bcast(&size, 1, size_type, root, comm);

        this->raw_strs.resize(size);
        auto const char_type = kamping::mpi_datatype<Char>();
        RBC::Bcast(this->raw_strs.data(), size, char_type, root, comm);

        if constexpr (has_index) {
            this->indices.resize(1);
            auto const idx_type = kamping::mpi_datatype<uint64_t>();
            RBC::Bcast(this->indices.data(), 1, idx_type, root, comm);
        }

        assert(std::count(raw_strs.begin(), raw_strs.end(), '\0') == 1);
        assert(raw_strs.back() == '\0');

        if constexpr (has_index) {
            dss_schimek::Index idx{this->indices.front()};
            return {this->raw_strs.data(), dss_schimek::Length{size - 1}, idx};
        } else {
            return {this->raw_strs.data(), dss_schimek::Length{size - 1}};
        }
    }

private:
    static_assert(
        std::is_same_v<
            std::tuple_element_t<0, std::tuple<Members<StringPtr>...>>,
            RawStrings<StringSet>>,
        "RawStrings must be the first Member"
    );
    static_assert(
        dss_schimek::has_member<String, dss_schimek::Length>,
        "the string set must have a length member"
    );

    static constexpr bool has_index =
        (std::is_same_v<Members<StringPtr>, Indices<StringPtr>> || ...);
    static constexpr bool has_lcp =
        (std::is_same_v<Members<StringPtr>, LcpValues<StringPtr>> || ...);
};

} // namespace _internal

// todo allow for LCP values
template <typename StringPtr>
using Container = dss_schimek::StringContainer<typename StringPtr::StringSet>;

template <typename StringPtr>
using StringT = typename StringPtr::StringSet::String;

template <typename StringPtr>
using CharT = typename StringPtr::StringSet::String;

template <typename StringPtr, typename Enable = void>
struct Comparator;

template <typename StringPtr>
struct Comparator<StringPtr, std::enable_if_t<!StringPtr::StringSet::is_indexed>> {
    bool operator()(StringT<StringPtr> const& lhs, StringT<StringPtr> const& rhs) {
        return dss_schimek::scmp(lhs.string, rhs.string) < 0;
    }
};

template <typename StringPtr>
struct Comparator<StringPtr, std::enable_if_t<StringPtr::StringSet::is_indexed>> {
    bool operator()(StringT<StringPtr> const& lhs, StringT<StringPtr> const& rhs) {
        auto ord = dss_schimek::scmp(lhs.string, rhs.string);
        return ord == 0 ? lhs.index < rhs.index : ord < 0;
    }
};

template <class StringPtr>
StringPtr make_str_ptr(Container<StringPtr>& container) {
    if constexpr (StringPtr::with_lcp) {
        static_assert(std::is_same_v<StringPtr, typename Container<StringPtr>::StringLcpPtr>);
        return container.make_str_lcp_ptr();
    } else {
        static_assert(std::is_same_v<StringPtr, typename Container<StringPtr>::StringPtr>);
        return container.make_str_ptr();
    }
}

// todo replace with function on stringcontainer
template <class StringPtr>
void resize(Container<StringPtr>& container, size_t const count) {
    container.strings().resize(count);
    if constexpr (StringPtr::with_lcp) {
        container.lcps().resize(count);
    }
}

template <typename StringPtr>
void copy_into(StringPtr const& strptr, Container<StringPtr>& container) {
    auto const& ss = strptr.active();

    container.strings().resize(strptr.size());
    std::copy(ss.begin(), ss.end(), container.strings());

    // todo consider lcp values
}

template <typename StringPtr>
using Data = std::conditional_t<
    StringPtr::StringSet::is_indexed,
    _internal::DataMembers<StringPtr, _internal::RawStrings, _internal::Indices>,
    _internal::DataMembers<StringPtr, _internal::RawStrings>>;

template <typename StringPtr>
struct TemporaryBuffers {
    Data<StringPtr>& send_data;
    Data<StringPtr>& recv_data;
    Container<StringPtr>& recv_strings;
    Container<StringPtr>& merge_strings;
    Container<StringPtr>& median_strings;
    std::vector<CharT<StringPtr>>& char_buffer;
};

} // namespace RQuick
