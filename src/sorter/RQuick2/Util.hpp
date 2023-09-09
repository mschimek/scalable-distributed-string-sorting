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

#include "kamping/mpi_datatype.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "strings/stringtools.hpp"
#include "util/measuringTool.hpp"

namespace RQuick2 {
namespace _internal {

template <typename T>
void add_comm_volume(size_t const count) {
    using dss_mehnert::measurement::MeasuringTool;
    MeasuringTool::measuringTool().addRawCommunication(sizeof(T) * count, "RQuick");
}

template <typename StringPtr>
struct RawStrings {
    using CharType = typename StringPtr::StringSet::Char;

    std::vector<CharType> raw_strs;

    MPI_Datatype char_type() const { return kamping::mpi_datatype<CharType>(); }
};

template <typename StringPtr>
struct Indices {
    using IndexType = uint64_t;

    std::vector<IndexType> indices;

    MPI_Datatype index_type() const { return kamping::mpi_datatype<IndexType>(); }
};

template <typename StringPtr>
struct LcpValues {
    using LcpType = typename StringPtr::LcpType;

    std::vector<LcpType> lcps;

    MPI_Datatype lcp_type() const { return kamping::mpi_datatype<LcpType>(); }
};

template <typename StringPtr_, template <typename> typename... Members>
class DataMembers : public Members<StringPtr_>... {
    using This = DataMembers<StringPtr_, Members...>;

public:
    using StringPtr = StringPtr_;
    using StringSet = typename StringPtr::StringSet;
    using String = typename StringPtr::StringSet::String;
    using Char = typename StringPtr::StringSet::Char;

    static constexpr bool has_index =
        (std::is_same_v<Members<StringPtr>, Indices<StringPtr>> || ...);
    static constexpr bool has_lcp =
        (std::is_same_v<Members<StringPtr>, LcpValues<StringPtr>> || ...);

    DataMembers() = default;

    template <typename... Args>
    DataMembers(Args&&... args) : Members<StringPtr>{std::forward<Args>(args)}... {}

    size_t get_num_strings() const {
        if constexpr (has_index) {
            return this->indices.size();
        } else if constexpr (has_lcp) {
            return this->lcps.size();
        } else {
            return std::count(this->raw_strs.begin(), this->raw_strs.end(), '\0');
        }
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

    void read_into(StringPtr const& strptr) {
        using dss_schimek::Index;
        using dss_schimek::Length;

        auto str_begin = this->raw_strs.begin();
        auto const end = this->raw_strs.end();
        for (size_t i = 0; auto& str: strptr.active()) {
            auto const str_end = std::find(str_begin, end, '\0');
            size_t const str_len = std::distance(str_begin, str_end);

            // todo consider lcp values if present
            if constexpr (has_index) {
                str = {&*str_begin, Length{str_len}, Index{this->indices[i]}};
            } else {
                str = {&*str_begin, Length{str_len}};
            }

            assert(str_end != end);
            str_begin = str_end + 1, ++i;
        }

        if constexpr (has_lcp) {
            assert(this->lcps.size() == strptr.size());
            std::copy(this->lcps.begin(), this->lcps.end(), strptr.lcp());
        }
    }

    void send(int const dest, int const tag, RBC::Comm const& comm) const {
        std::array<MPI_Request, 3> requests;
        std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);

        // clang-format off
        {
            int const char_tag = tag;
            RBC::Isend(this->raw_strs.data(), this->raw_strs.size(), this->char_type(),
                       dest, char_tag, comm, requests.data());
            add_comm_volume<This::CharType>(this->raw_strs.size());
        }

        if constexpr (has_index) {
            int const idx_tag = tag + 1;
            RBC::Isend(this->indices.data(), this->indices.size(), this->index_type(),
                       dest, idx_tag, comm, requests.data() + 1);
            add_comm_volume<This::IndexType>(this->indices.size());
        }

        if constexpr (has_lcp) {
            int const lcp_tag = tag + 2;
            RBC::Isend(this->lcps.data(), this->lcps.size(), this->lcp_type(),
                       dest, lcp_tag, comm, requests.data() + 2);
            add_comm_volume<This::LcpType>(this->lcps.size());
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    void recv(int const src, int const tag, RBC::Comm const& comm, bool append = false) {
        std::array<MPI_Request, 3> requests = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        MPI_Status status;

        // clang-format off
        {
            int const char_tag = tag;

            int char_count = 0;
            RBC::Probe(src, char_tag, comm, &status);
            MPI_Get_count(&status, this->char_type(), &char_count);

            size_t const char_offset = append ? this->raw_strs.size() : 0;
            this->raw_strs.resize(char_offset + char_count);
            RBC::Irecv(this->raw_strs.data() + char_offset, char_count, this->char_type(),
                       src, char_tag, comm, requests.data());
        }

        if constexpr (has_index || has_lcp) {
            int const idx_tag = tag + 1, lcp_tag = tag + 2;
            int count = 0;

            if constexpr (has_index) {
                RBC::Probe(src, idx_tag, comm, &status);
                MPI_Get_count(&status, this->index_type(), &count);
            } else if constexpr (has_lcp){
                RBC::Probe(src, lcp_tag, comm, &status);
                MPI_Get_count(&status, this->lcp_type(), &count);
            }

            if constexpr (has_index) {
                size_t const idx_offset = append ? this->indices.size() : 0;
                this->indices.resize(idx_offset + count);
                RBC::Irecv(this->indices.data() + idx_offset, count, this->index_type(),
                           src, idx_tag, comm, requests.data() + 1);
            }
            if constexpr (has_lcp) {
                size_t const lcp_offset = append ? this->lcps.size() : 0;
                this->lcps.resize(lcp_offset + count);
                RBC::Irecv(this->lcps.data() + lcp_offset, count, this->lcp_type(),
                           src, lcp_tag, comm, requests.data() + 2);
            }
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
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
            RBC::Irecv(recv.indices.data(), recv_cnt, this->index_type(),
                       partner, idx_tag, comm, requests.data());
            RBC::Isend(this->indices.data(), this->indices.size(), this->index_type(),
                       partner, idx_tag, comm, requests.data() + 1);
            add_comm_volume<This::IndexType>(this->indices.size());
        }

        if constexpr (has_lcp) {
            int const lcp_tag = tag + 2;

            recv.lcps.resize(recv_cnt);
            RBC::Irecv(recv.lcps.data(), recv_cnt, this->lcp_type(),
                       partner, lcp_tag, comm, requests.data() + 2);
            RBC::Isend(this->lcps.data(), this->lcps.size(), this->lcp_type(),
                       partner, lcp_tag, comm, requests.data() + 3);
            add_comm_volume<This::LcpType>(this->lcps.size());
        }

        {
            int const char_tag = tag;

            int send_cnt_char = this->raw_strs.size(), recv_cnt_char = 0;
            RBC::Isend(this->raw_strs.data(), send_cnt_char, this->char_type(),
                       partner, char_tag, comm, requests.data() + 4);
            add_comm_volume<This::CharType>(this->raw_strs.size());

            MPI_Status status;
            RBC::Probe(partner, char_tag, comm, &status);
            MPI_Get_count(&status, this->char_type(), &recv_cnt_char);

            recv.raw_strs.resize(recv_cnt_char);
            RBC::Irecv(recv.raw_strs.data(), recv_cnt_char, this->char_type(),
                       partner, char_tag, comm, requests.data() + 5);
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    String bcast_single(int const root, RBC::Comm const& comm) {
        using dss_schimek::Index;
        using dss_schimek::Length;

        auto const size_type = kamping::mpi_datatype<size_t>();

        size_t char_size = this->raw_strs.size();

        if constexpr (has_index && std::is_same_v<size_t, typename This::IndexType>) {
            // combine broadcast of raw string size and index
            this->indices.resize(1);
            std::array<uint64_t, 2> send_recv_buf = {char_size, this->indices[0]};
            RBC::Bcast(send_recv_buf.data(), 2, this->index_type(), root, comm);
            add_comm_volume<This::IndexType>(2);

            std::tie(char_size, this->indices.front()) = std::tuple_cat(send_recv_buf);
        } else {
            RBC::Bcast(&char_size, 1, size_type, root, comm);
            add_comm_volume<size_t>(1);

            if constexpr (has_index) {
                this->indices.resize(1);
                RBC::Bcast(this->indices.data(), 1, this->index_type(), root, comm);
                add_comm_volume<This::IndexType>(1);
            }
        }

        {
            this->raw_strs.resize(char_size);
            RBC::Bcast(this->raw_strs.data(), char_size, this->char_type(), root, comm);
            add_comm_volume<This::CharType>(this->raw_strs.size());
        }

        if constexpr (has_lcp) {
            // there is never a common prefix for a single string
            this->lcps.resize(1);
            this->lcps.front() = 0;
        }

        assert(std::count(this->raw_strs.begin(), this->raw_strs.end(), '\0') == 1);
        assert(this->raw_strs.back() == '\0');
        if constexpr (has_index) {
            return {this->raw_strs.data(), Length{char_size - 1}, Index{this->indices.front()}};
        } else {
            return {this->raw_strs.data(), Length{char_size - 1}};
        }
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
};

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

} // namespace _internal


// todo allow for LCP values
template <typename StringPtr>
using Container = std::conditional_t<
    StringPtr::with_lcp,
    dss_schimek::StringLcpContainer<typename StringPtr::StringSet>,
    dss_schimek::StringContainer<typename StringPtr::StringSet>>;

template <typename StringPtr>
using StringT = typename StringPtr::StringSet::String;

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
        auto ord = dss_schimek::scmp(lhs.string, rhs.string);
        return ord == 0 ? lhs.index < rhs.index : ord < 0;
    }
};

template <typename StringPtr>
struct Data : public _internal::Data_<StringPtr> {
    using _internal::Data_<StringPtr>::Data_;

    static_assert(std::is_same_v<typename Container<StringPtr>::AutoStringPtr, StringPtr>);
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
