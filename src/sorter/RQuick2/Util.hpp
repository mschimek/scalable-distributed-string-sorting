#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <span>
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
    std::vector<typename StringPtr::StringSet::Char> raw_strs;
};

template <typename StringPtr>
struct Indices {
    std::vector<uint64_t> indices;
};

template <typename StringPtr>
struct LcpValues {
    std::vector<uint64_t> lcps;
};

template <typename StringPtr_, template <typename> typename... Members>
class DataMembers : public Members<StringPtr_>... {
public:
    using StringPtr = StringPtr_;
    using StringSet = typename StringPtr::StringSet;
    using String = typename StringPtr::StringSet::String;
    using Char = typename StringPtr::StringSet::Char;

    DataMembers() = default;

    template <typename... Args>
    DataMembers(Args&&... args) : Members<StringPtr>{std::forward<Args>(args)}... {}

    size_t get_num_strings() const {
        if constexpr (has_index) {
            return this->indices.size();
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
    }

    void send(int const dest, int const tag, RBC::Comm const& comm) const {
        int const char_tag = tag, idx_tag = tag + 1;
        std::array<MPI_Request, 2> requests = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        // clang-format off
        {
            auto const char_type = kamping::mpi_datatype<Char>();
            RBC::Isend(this->raw_strs.data(), this->raw_strs.size(), char_type, dest, char_tag, 
                       comm, requests.data());
            add_comm_volume<Char>(this->raw_strs.size());
        }

        if constexpr (has_index) {
            auto const idx_type = kamping::mpi_datatype<uint64_t>();
            RBC::Isend(this->indices.data(), this->indices.size(), idx_type, dest, idx_tag,
                       comm, requests.data() + 1);
            add_comm_volume<uint64_t>(this->indices.size());
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    void recv(int const src, int const tag, RBC::Comm const& comm, bool append = false) {
        int const char_tag = tag, idx_tag = tag + 1;
        std::array<MPI_Request, 2> requests = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        MPI_Status status;

        // clang-format off
        {
            auto const char_type = kamping::mpi_datatype<Char>();
            int char_count = 0;
            RBC::Probe(src, char_tag, comm, &status);
            MPI_Get_count(&status, char_type, &char_count);

            size_t const char_offset = append ? this->raw_strs.size() : 0;
            this->raw_strs.resize(char_offset + char_count);
            RBC::Irecv(this->raw_strs.data() + char_offset, char_count, char_type,
                       src, char_tag, comm, requests.data());
        }

        if constexpr (has_index) {
            auto const idx_type = kamping::mpi_datatype<uint64_t>();
            int idx_count = 0;
            RBC::Probe(src, idx_tag, comm, &status);
            MPI_Get_count(&status, idx_type, &idx_count);

            size_t const idx_offset = append ? this->indices.size() : 0;
            this->indices.resize(idx_offset + idx_count);
            RBC::Irecv(this->indices.data() + idx_offset, idx_count, idx_type,
                       src, idx_tag, comm, requests.data() + 1);
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
        int const char_tag = tag, idx_tag = tag + 1;
        std::array<MPI_Request, 4> requests;
        std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);

        // clang-format off
        if constexpr (has_index) {
            auto const idx_type = kamping::mpi_datatype<uint64_t>();
            recv.indices.resize(recv_cnt);
            RBC::Irecv(recv.indices.data(), recv_cnt, idx_type, partner, idx_tag,
                       comm, requests.data());
            RBC::Isend(this->indices.data(), this->indices.size(), idx_type, partner, idx_tag,
                       comm, requests.data() + 1);
            add_comm_volume<uint64_t>(this->indices.size());
        }

        {
            auto const char_type = kamping::mpi_datatype<Char>();
            int send_cnt_char = this->raw_strs.size(), recv_cnt_char = 0;

            RBC::Isend(this->raw_strs.data(), send_cnt_char, char_type, partner, char_tag,
                       comm, requests.data() + 2);
            add_comm_volume<Char>(this->raw_strs.size());

            MPI_Status status;
            RBC::Probe(partner, char_tag, comm, &status);
            MPI_Get_count(&status, char_type, &recv_cnt_char);

            recv.raw_strs.resize(recv_cnt_char);
            RBC::Irecv(recv.raw_strs.data(), recv_cnt_char, char_type, partner, char_tag,
                       comm, requests.data() + 3);
        }
        // clang-format on
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    String bcast_single(int const root, RBC::Comm const& comm) {
        using dss_schimek::Index;
        using dss_schimek::Length;

        auto const char_type = kamping::mpi_datatype<Char>();
        auto const size_type = kamping::mpi_datatype<size_t>();
        auto const idx_type = kamping::mpi_datatype<uint64_t>();

        size_t char_size = this->raw_strs.size();

        if constexpr (has_index && std::is_same_v<size_t, uint64_t>) {
            // combine broadcast of raw string size and index
            this->indices.resize(1);
            std::array<uint64_t, 2> send_recv_buf = {char_size, this->indices[0]};
            RBC::Bcast(send_recv_buf.data(), 2, idx_type, root, comm);
            add_comm_volume<uint64_t>(2);

            std::tie(char_size, this->indices.front()) = std::tuple_cat(send_recv_buf);
        } else {
            RBC::Bcast(&char_size, 1, size_type, root, comm);
            add_comm_volume<size_t>(1);

            if constexpr (has_index) {
                this->indices.resize(1);
                RBC::Bcast(this->indices.data(), 1, idx_type, root, comm);
                add_comm_volume<uint64_t>(1);
            }
        }

        this->raw_strs.resize(char_size);
        RBC::Bcast(this->raw_strs.data(), char_size, char_type, root, comm);
        add_comm_volume<Char>(this->raw_strs.size());

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

    static constexpr bool has_index =
        (std::is_same_v<Members<StringPtr>, Indices<StringPtr>> || ...);
    static constexpr bool has_lcp =
        (std::is_same_v<Members<StringPtr>, LcpValues<StringPtr>> || ...);
};

template <typename StringPtr>
using Data_ = std::conditional_t<
    StringPtr::StringSet::is_indexed,
    DataMembers<StringPtr, RawStrings, Indices>,
    DataMembers<StringPtr, RawStrings>>;

} // namespace _internal


// todo allow for LCP values
template <typename StringPtr>
using Container = std::conditional_t<
    StringPtr::StringSet::is_indexed,
    dss_schimek::StringContainer<typename StringPtr::StringSet>,
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
