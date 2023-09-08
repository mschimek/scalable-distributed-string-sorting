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

namespace RQuick2 {
namespace _internal {

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
            assert(str_len == 50);

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
        auto mpi_type = kamping::mpi_datatype<Char>();
        RBC::Send(this->raw_strs.data(), this->raw_strs.size(), mpi_type, dest, tag, comm);

        if constexpr (has_index) {
            auto mpi_type = kamping::mpi_datatype<uint64_t>();
            RBC::Send(this->indices.data(), this->indices.size(), mpi_type, dest, tag, comm);
        }
    }

    void recv(int const src, int const tag, RBC::Comm const& comm, bool append = false) {
        // clang-format off
        auto char_type = kamping::mpi_datatype<Char>();
        MPI_Status status;
        RBC::Probe(src, tag, comm, &status);

        int count_chars = 0;
        MPI_Get_count(&status, char_type, &count_chars);

        size_t offset = append ? this->raw_strs.size() : 0;
        this->raw_strs.resize(offset + count_chars);
        RBC::Recv(this->raw_strs.data() + offset, count_chars, char_type,
                  src, tag, comm, MPI_STATUS_IGNORE);

        if constexpr (has_index) {
            auto idx_type = kamping::mpi_datatype<uint64_t>();
            RBC::Probe(src, tag, comm, &status);

            int count = 0;
            MPI_Get_count(&status, idx_type, &count);

            size_t idx_offset = append ? this->indices.size() : 0;
            this->indices.resize(idx_offset + count);
            RBC::Recv(this->indices.data() + idx_offset, count, idx_type,
                      src, tag, comm, MPI_STATUS_IGNORE);
        }
        // clang-format on
    }

    void sendrecv(
        DataMembers<StringPtr, Members...>& recv,
        int const recv_cnt,
        int const partner,
        int const tag,
        RBC::Comm const& comm
    ) {
        // clang-format off
        auto const size_type = kamping::mpi_datatype<size_t>();
        size_t char_send_cnt = this->raw_strs.size(), char_recv_cnt = 0;
        RBC::Sendrecv(&char_send_cnt, 1, size_type, partner, tag,
                      &char_recv_cnt, 1, size_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);

        auto const char_type = kamping::mpi_datatype<Char>();
        recv.raw_strs.resize(char_recv_cnt);
        RBC::Sendrecv(this->raw_strs.data(), char_send_cnt, char_type, partner, tag,
                      recv.raw_strs.data(),  char_recv_cnt, char_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);

        if constexpr (has_index) {
            auto const mpi_type = kamping::mpi_datatype<uint64_t>();
            recv.indices.resize(recv_cnt);
            RBC::Sendrecv(this->indices.data(), this->indices.size(), mpi_type, partner, tag,
                          recv.indices.data(),  recv_cnt,             mpi_type, partner, tag,
                          comm, MPI_STATUS_IGNORE);
        }
        // clang-format on
    }

    String bcast_single(int const root, RBC::Comm const& comm) {
        using dss_schimek::Index;
        using dss_schimek::Length;

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

        assert(std::count(this->raw_strs.begin(), this->raw_strs.end(), '\0') == 1);
        assert(this->raw_strs.back() == '\0');

        if constexpr (has_index) {
            return {this->raw_strs.data(), Length{size - 1}, Index{this->indices.front()}};
        } else {
            return {this->raw_strs.data(), Length{size - 1}};
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
// todo get rid of index string container
template <typename StringPtr>
using Container = std::conditional_t<
    StringPtr::StringSet::is_indexed,
    dss_schimek::IndexStringContainer<typename StringPtr::StringSet>,
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

template <class StringPtr>
StringPtr make_str_ptr(Container<StringPtr>& container) {
    if constexpr (StringPtr::with_lcp) {
        static_assert(std::is_same_v<StringPtr, typename Container<StringPtr>::StringLcpPtr>);
        return container.make_string_lcp_ptr();
    } else {
        static_assert(std::is_same_v<StringPtr, typename Container<StringPtr>::StringPtr>);
        return container.make_string_ptr();
    }
}

template <typename StringPtr>
void copy_into(StringPtr const& strptr, Container<StringPtr>& container) {
    auto const& ss = strptr.active();

    container.resize_strings(strptr.size());
    std::copy(ss.begin(), ss.end(), container.getStrings().begin());

    // todo consider lcp values
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
