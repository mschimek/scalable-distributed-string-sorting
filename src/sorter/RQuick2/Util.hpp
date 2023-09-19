// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <RBC.hpp>
#include <kamping/mpi_datatype.hpp>

#include "merge/lcp_merge.hpp"
#include "strings/bound.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "strings/stringtools.hpp"
#include "util/measuringTool.hpp"

namespace RQuick2 {

template <typename StringPtr>
using StringT = typename StringPtr::StringSet::String;

template <typename StringPtr>
using IteratorT = typename StringPtr::StringSet::Iterator;

template <typename StringPtr>
using CharT = typename StringPtr::StringSet::Char;

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
        auto const ord = dss_schimek::scmp(lhs.string, rhs.string);
        return ord == 0 ? lhs.index < rhs.index : ord < 0;
    }
};

namespace _internal {

template <typename T>
void add_comm_volume(size_t const count) {
    using dss_mehnert::measurement::MeasuringTool;
    MeasuringTool::measuringTool().addRawCommunication(sizeof(T) * count, "RQuick");
}

template <typename StringPtr>
struct RawStrings {
    std::vector<typename StringPtr::StringSet::Char> raw_strs;
};

template <typename StringPtr>
struct Indices {
    std::vector<uint64_t> indices;
};

template <typename StringPtr>
struct LcpValues {
    std::vector<typename StringPtr::LcpType> lcps;
};

template <typename StringPtr_, template <typename> typename... Members>
class DataMembers : public Members<StringPtr_>... {
public:
    using StringPtr = StringPtr_;
    using StringSet = typename StringPtr::StringSet;
    using String = typename StringPtr::StringSet::String;

    using CharType = typename StringPtr::StringSet::Char;
    using IndexType = uint64_t;

    template <typename _StringPtr = StringPtr>
    using LcpType = _StringPtr::LcpType;

    static constexpr bool has_index =
        (std::is_same_v<Members<StringPtr>, Indices<StringPtr>> || ...);
    static constexpr bool has_lcp =
        (std::is_same_v<Members<StringPtr>, LcpValues<StringPtr>> || ...);

    DataMembers() = default;

    DataMembers(std::vector<CharType> raw_strs) : RawStrings<StringPtr>{std::move(raw_strs)} {}

    size_t get_num_strings(bool const ignore_lcps = false) const {
        if constexpr (has_index) {
            return this->indices.size();
        } else if constexpr (has_lcp) {
            if (!ignore_lcps) {
                return this->lcps.size();
            }
        }
        return std::count(this->raw_strs.begin(), this->raw_strs.end(), '\0');
    }

    void write(StringPtr const& strptr) {
        auto const& ss = strptr.active();
        this->raw_strs.resize(ss.get_sum_length() + ss.size());
        if constexpr (has_index) {
            this->indices.resize(ss.size());
        }

        auto char_dest = this->raw_strs.begin();
        for (size_t i = 0; auto& str: ss) {
            char_dest = std::copy_n(str.getChars(), str.getLength(), char_dest);
            *char_dest++ = 0;

            if constexpr (has_index) {
                this->indices[i++] = str.getIndex();
            }
        }

        if constexpr (has_lcp) {
            this->lcps.resize(strptr.size());
            std::copy_n(strptr.lcp(), strptr.size(), this->lcps.begin());
        }
    }

    void read_into(StringPtr const& strptr, bool const copy_lcps = true) {
        using dss_schimek::Index;
        using dss_schimek::Length;

        auto char_it = this->raw_strs.begin();
        auto const end = this->raw_strs.end();
        for (size_t i = 0; auto& str: strptr.active()) {
            auto const str_begin = char_it;
            while (*char_it != 0) {
                ++char_it;
            }
            auto const str_end = char_it;
            size_t const str_len = std::distance(str_begin, str_end);

            if constexpr (has_index) {
                str = {&*str_begin, Length{str_len}, Index{this->indices[i]}};
            } else {
                str = {&*str_begin, Length{str_len}};
            }

            assert(char_it != end);
            ++char_it, ++i;
        }

        if constexpr (has_lcp) {
            if (copy_lcps) {
                assert(this->lcps.size() == strptr.size());
                std::copy(this->lcps.begin(), this->lcps.end(), strptr.lcp());
            }
        }
    }

    void send(int const dest, int const tag, RBC::Comm const& comm, bool const send_lcps = true) {
        std::array<MPI_Request, 3> requests;
        std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);

        // clang-format off
        {
            int const char_tag = tag;
            RBC::Isend(this->raw_strs.data(), this->raw_strs.size(), char_type(),
                       dest, char_tag, comm, requests.data());
            add_comm_volume<CharType>(this->raw_strs.size());
        }

        if constexpr (has_index) {
            int const idx_tag = tag + 1;
            RBC::Isend(this->indices.data(), this->indices.size(), index_type(),
                       dest, idx_tag, comm, requests.data() + 1);
            add_comm_volume<IndexType>(this->indices.size());
        }

        if constexpr (has_lcp) {
            if (send_lcps) {
                int const lcp_tag = tag + 2;
                RBC::Isend(this->lcps.data(), this->lcps.size(), lcp_type(),
                           dest, lcp_tag, comm, requests.data() + 2);
                add_comm_volume<LcpType<>>(this->lcps.size());
            }
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    void recv(
        int const src,
        int const tag,
        RBC::Comm const& comm,
        bool const append = false,
        bool const recv_lcps = true
    ) {
        assert(!recv_lcps || !append);

        std::array<MPI_Request, 3> requests;
        std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);
        MPI_Status status;

        // clang-format off
        {
            int const char_tag = tag;

            int char_count = 0;
            RBC::Probe(src, char_tag, comm, &status);
            MPI_Get_count(&status, char_type(), &char_count);

            size_t const char_offset = append ? this->raw_strs.size() : 0;
            this->raw_strs.resize(char_offset + char_count);
            RBC::Irecv(this->raw_strs.data() + char_offset, char_count, char_type(),
                       src, char_tag, comm, requests.data());
        }

        if constexpr (has_index || has_lcp) {
            int const idx_tag = tag + 1, lcp_tag = tag + 2;
            int count = 0;

            if constexpr (has_index) {
                RBC::Probe(src, idx_tag, comm, &status);
                MPI_Get_count(&status, index_type(), &count);
            } else if constexpr (has_lcp) {
                if (recv_lcps) {
                    RBC::Probe(src, lcp_tag, comm, &status);
                    MPI_Get_count(&status, lcp_type(), &count);
                }
            }

            if constexpr (has_index) {
                size_t const idx_offset = append ? this->indices.size() : 0;
                this->indices.resize(idx_offset + count);
                RBC::Irecv(this->indices.data() + idx_offset, count, index_type(),
                           src, idx_tag, comm, requests.data() + 1);
            }
            if constexpr (has_lcp) {
                if (recv_lcps) {
                    size_t const lcp_offset = append ? this->lcps.size() : 0;
                    this->lcps.resize(lcp_offset + count);
                    RBC::Irecv(this->lcps.data() + lcp_offset, count, lcp_type(),
                               src, lcp_tag, comm, requests.data() + 2);
                }
            }
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // slight bodge to get the `send_lcps == false` case right
        if constexpr (has_lcp) {
            if (!recv_lcps) {
                auto const& chars = this->raw_strs;
                auto const num_strings = std::count(chars.begin(), chars.end(), 0);
                this->lcps.resize(num_strings);
            }
        }
    }

    void sendrecv(
        DataMembers<StringPtr, Members...>& recv,
        int const recv_cnt,
        int const partner,
        int const tag,
        RBC::Comm const& comm
    ) {
        std::array<MPI_Request, 6> requests;
        std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);

        // clang-format off
        if constexpr (has_index) {
            int const idx_tag = tag + 1;

            recv.indices.resize(recv_cnt);
            RBC::Irecv(recv.indices.data(), recv_cnt, index_type(),
                       partner, idx_tag, comm, requests.data());
            RBC::Isend(this->indices.data(), this->indices.size(), index_type(),
                       partner, idx_tag, comm, requests.data() + 1);
            add_comm_volume<IndexType>(this->indices.size());
        }

        if constexpr (has_lcp) {
            int const lcp_tag = tag + 2;

            recv.lcps.resize(recv_cnt);
            RBC::Irecv(recv.lcps.data(), recv_cnt, lcp_type(),
                       partner, lcp_tag, comm, requests.data() + 2);
            RBC::Isend(this->lcps.data(), this->lcps.size(), lcp_type(),
                       partner, lcp_tag, comm, requests.data() + 3);
            add_comm_volume<LcpType<>>(this->lcps.size());
        }

        {
            int const char_tag = tag;

            int send_cnt_char = this->raw_strs.size(), recv_cnt_char = 0;
            RBC::Isend(this->raw_strs.data(), send_cnt_char, char_type(),
                       partner, char_tag, comm, requests.data() + 4);
            add_comm_volume<CharType>(this->raw_strs.size());

            MPI_Status status;
            RBC::Probe(partner, char_tag, comm, &status);
            MPI_Get_count(&status, char_type(), &recv_cnt_char);

            recv.raw_strs.resize(recv_cnt_char);
            RBC::Irecv(recv.raw_strs.data(), recv_cnt_char, char_type(),
                       partner, char_tag, comm, requests.data() + 5);
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    void bcast_single(int const root, RBC::Comm const& comm) {
        using dss_schimek::Index;
        using dss_schimek::Length;

        auto const size_type = kamping::mpi_datatype<size_t>();

        size_t char_size = this->raw_strs.size();

        if constexpr (has_index && std::is_same_v<size_t, IndexType>) {
            // combine broadcast of raw string size and index
            this->indices.resize(1);
            std::array<uint64_t, 2> send_recv_buf = {char_size, this->indices[0]};
            RBC::Bcast(send_recv_buf.data(), 2, index_type(), root, comm);
            add_comm_volume<IndexType>(2);

            std::tie(char_size, this->indices.front()) = std::tuple_cat(send_recv_buf);
        } else {
            RBC::Bcast(&char_size, 1, size_type, root, comm);
            add_comm_volume<size_t>(1);

            if constexpr (has_index) {
                this->indices.resize(1);
                RBC::Bcast(this->indices.data(), 1, index_type(), root, comm);
                add_comm_volume<IndexType>(1);
            }
        }

        {
            this->raw_strs.resize(char_size);
            RBC::Bcast(this->raw_strs.data(), char_size, char_type(), root, comm);
            add_comm_volume<CharType>(this->raw_strs.size());
        }

        if constexpr (has_lcp) {
            // there is never a common prefix for a single string
            this->lcps.resize(1);
            this->lcps.front() = 0;
        }

        assert(std::count(this->raw_strs.begin(), this->raw_strs.end(), '\0') == 1);
        assert(this->raw_strs.back() == '\0');
    }

private:
    static_assert(
        std::is_same_v<
            std::tuple_element_t<0, std::tuple<Members<StringPtr>...>>,
            RawStrings<StringPtr>>,
        "RawStrings must be the first Member"
    );
    static_assert(
        dss_schimek::has_member<String, dss_schimek::Length>,
        "the string set must have a length member"
    );

    MPI_Datatype char_type() const { return kamping::mpi_datatype<CharType>(); }

    MPI_Datatype index_type() const { return kamping::mpi_datatype<IndexType>(); }

    template <typename _StringPtr = StringPtr>
    MPI_Datatype lcp_type() const {
        return kamping::mpi_datatype<LcpType<>>();
    }
};

// todo not a massive fan of this typedef
template <typename StringPtr>
using Data_ = std::conditional_t<
    StringPtr::StringSet::is_indexed,
    std::conditional_t<
        StringPtr::with_lcp,
        DataMembers<StringPtr, RawStrings, Indices, LcpValues>,
        DataMembers<StringPtr, RawStrings, Indices>>,
    std::conditional_t<
        StringPtr::with_lcp,
        DataMembers<StringPtr, RawStrings, LcpValues>,
        DataMembers<StringPtr, RawStrings>>>;

template <typename StringPtr>
std::pair<IteratorT<StringPtr>, size_t> lower_bound_index(
    StringPtr const& strptr,
    typename StringPtr::StringSet::String const& value,
    size_t const initial_lcp
) {
    auto const& ss = strptr.active();
    auto const begin = ss.begin(), end = ss.end();

    auto const value_length = value.getLength();
    auto const value_index = value.getIndex();

    if (!ss.empty()) {
        assert(initial_lcp <= dss_schimek::calc_lcp(ss, ss[begin], value));

        if (ss[begin].getLength() != value_length || value_length != initial_lcp
            || value_index <= ss[begin].getIndex()) {
            // value is already less than or equal to begin
            return {begin, initial_lcp};
        }

        size_t i = 1;
        for (auto it = begin + 1; it != ss.end(); ++it, ++i) {
            if (strptr.get_lcp(i) != value_length || value_index <= ss[it].getIndex()) {
                return {it, initial_lcp};
            }
        }
        return {end, 0};
    } else {
        return {begin, initial_lcp};
    }
}

template <typename StringPtr>
std::pair<IteratorT<StringPtr>, size_t>
lower_bound(StringPtr const& strptr, StringT<StringPtr> const& value) {
    assert(strptr.active().check_order());
    Comparator<StringPtr> const comp;

    auto const begin = strptr.active().begin();
    auto const end = strptr.active().end();
    if constexpr (StringPtr::with_lcp) {
        auto [result, lcp] = dss_mehnert::lcp_lower_bound(strptr, value);

        if constexpr (StringPtr::StringSet::is_indexed) {
            auto const bound_ptr = strptr.sub(result - begin, end - result);
            std::tie(result, lcp) = lower_bound_index(bound_ptr, value, lcp);
        }

        assert_equal(std::lower_bound(begin, end, value, comp), result);
        return {result, lcp};

    } else {
        return {std::lower_bound(begin, end, value, comp), 0};
    }
}

template <typename StringPtr>
std::pair<IteratorT<StringPtr>, size_t>
upper_bound(StringPtr const& strptr, StringT<StringPtr> const& value) {
    assert(strptr.active().check_order());
    Comparator<StringPtr> const comp;

    auto const begin = strptr.active().begin();
    auto const end = strptr.active().end();
    if constexpr (StringPtr::with_lcp) {
        auto [result, lcp] = dss_mehnert::lcp_upper_bound(strptr, value);
        assert_equal(std::upper_bound(begin, end, value, comp), result);
        return {result, lcp};
    } else {
        return {std::upper_bound(begin, end, value, comp), 0};
    }
}

template <typename StringPtr>
std::pair<IteratorT<StringPtr>, IteratorT<StringPtr>>
lower_upper_bound(StringPtr const& strptr, StringT<StringPtr> const& value) {
    auto const begin = strptr.active().begin();
    auto const end = strptr.active().end();

    auto const [lower, lower_lcp] = lower_bound(strptr, value);
    auto const strptr_geq = strptr.sub(lower - begin, end - lower);

    if constexpr (StringPtr::with_lcp) {
        auto const [upper, _] = dss_mehnert::lcp_upper_bound(strptr_geq, value, lower_lcp);
        return {lower, upper};
    } else {
        auto const [upper, _] = upper_bound(strptr_geq, value);
        return {lower, upper};
    }
}

} // namespace _internal

template <typename StringPtr>
using Container = std::conditional_t<
    StringPtr::with_lcp,
    dss_schimek::StringLcpContainer<typename StringPtr::StringSet>,
    dss_schimek::StringContainer<typename StringPtr::StringSet>>;

template <class StringPtr>
void merge(StringPtr const& strptr1, StringPtr const& strptr2, Container<StringPtr>& dest) {
    assert(strptr1.active().check_order());
    assert(strptr2.active().check_order());

    dest.resize_strings(strptr1.size() + strptr2.size());

    if constexpr (StringPtr::with_lcp) {
        dss_mehnert::merge::lcp_merge(strptr1, strptr2, dest.make_auto_ptr());
    } else {
        Comparator<StringPtr> const comp;
        auto const dest_set = dest.make_string_set();
        auto const ss1 = strptr1.active(), ss2 = strptr2.active();
        std::merge(ss1.begin(), ss1.end(), ss2.begin(), ss2.end(), dest_set.begin(), comp);
    }

    assert(dest.make_string_set().check_order());
}


template <typename StringPtr>
struct Data : public _internal::Data_<StringPtr> {
    using _internal::Data_<StringPtr>::Data_;
};

template <typename StringPtr>
struct TemporaryBuffers {
    Data<StringPtr> send_data;
    Data<StringPtr> recv_data;
    Container<StringPtr> recv_strings;
    Container<StringPtr> merge_strings;
    Container<StringPtr> median_strings;
    std::vector<CharT<StringPtr>> char_buffer;
};

} // namespace RQuick2
