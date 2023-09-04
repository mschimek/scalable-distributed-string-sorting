#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include "RBC.hpp"
#include "kamping/mpi_datatype.hpp"
#include "strings/stringset.hpp"

namespace RQuick {
namespace _internal {

template <typename StringSet>
inline size_t count_chars(StringSet const& ss) {
    auto acc = [](auto const& n, auto const& str) { return n + str.getLength(); };
    return std::accumulate(ss.begin(), ss.end(), acc) + ss.size();
}

template <typename StringSet>
struct RawStrings {
    std::vector<typename StringSet::Char> raw_strs;

    void write(StringSet& ss) {
        raw_strs.resize(count_chars(ss));

        auto dest = raw_strs.begin();
        for (auto it = ss.begin(); it != ss.end(); ++it) {
            auto const old_chars = ss[it].getChars();
            ss[it].setChars(&*dest);
            dest = std::copy_n(old_chars, ss[it].getLength(), dest);
            *dest++ = 0;
        }
    }

    void send(int const dest, int const tag, RBC::Comm const& comm) const {
        auto mpi_type = kamping::mpi_datatype<typename StringSet::Char>();
        RBC::Send(raw_strs.data(), raw_strs.size(), mpi_type, dest, tag, comm);
    }

    void recv(int const src, int const tag, RBC::Comm const& comm) {
        auto mpi_type = kamping::mpi_datatype<typename StringSet::Char>();

        MPI_Status status;
        RBC::Probe(src, tag, comm, &status);

        int count = 0;
        MPI_Get_count(&status, mpi_type, &count);

        raw_strs.resize(count);
        RBC::Recv(raw_strs.data(), count, mpi_type, src, tag, comm, MPI_STATUS_IGNORE);
    }

    void bcast_single(int root, RBC::Comm const& comm) {
        assert(std::count(raw_strs.begin(), raw_strs.end(), '\0') == 1);

        auto size_type = kamping::mpi_datatype<size_t>();
        size_t size = raw_strs.size();
        RBC::Bcast(&size, 1, size_type, root, comm);

        raw_strs.resize(size);
        auto char_type = kamping::mpi_datatype<typename StringSet::Char>();
        RBC::Bcast(raw_strs.data(), size, char_type, root, comm);
    }
};

template <typename StringSet>
struct Indices {
    std::vector<uint64_t> indices;

    void write(StringSet& ss) {
        indices.resize(ss.size());

        auto dest = indices.begin();
        for (auto it = ss.begin(); it != ss.end(); ++it) {
            *dest++ = ss[it].getIndex();
        }
    }

    void send(int const dest, int const tag, RBC::Comm const& comm) const {
        auto mpi_type = kamping::mpi_datatype<uint64_t>();
        RBC::Send(indices.data(), indices.size(), mpi_type, dest, tag, comm);
    }

    void recv(int const src, int const tag, RBC::Comm const& comm) {
        auto mpi_type = kamping::mpi_datatype<uint64_t>();

        MPI_Status status;
        RBC::Probe(src, tag, comm, &status);

        int count = 0;
        MPI_Get_count(&status, mpi_type, &count);

        indices.resize(count);
        RBC::Recv(indices.data(), count, mpi_type, src, tag, comm, MPI_STATUS_IGNORE);
    }

    void bcast_single(int root, RBC::Comm const& comm) {
        assert(indicies.size() == 1);

        indices.resize(1);
        auto mpi_type = kamping::mpi_datatype<uint64_t>();
        RBC::Bcast(indices.data(), 1, mpi_type, root, comm);
    }
};

template <typename StringSet, template <typename> typename... Members>
class DataMembers : private Members<StringSet>... {
public:
    void write(StringSet& ss) { (Members<StringSet>::write(ss), ...); }

    void read_into(std::vector<typename StringSet::String>& strings) const {
        if constexpr (has_index) {
            strings.resize(this->indices.size());
        } else {
            auto& chars = this->raw_strs;
            strings.resize(std::count(chars.begin(), chars.end(), "\0"));
        }

        auto str_begin = this->raw_strs.begin();
        for (size_t i = 0; auto& str: strings) {
            auto const str_end = std::find(str_begin, this->raw_strs.end(), '\0');
            auto const str_len = std::distance(str_begin, str_end) + 1;

            if constexpr (has_index) {
                auto const str_idx = this->indices[i];
                str = {&*str_begin, dss_schimek::Length{str_len}, dss_schimek::Index{str_idx}};
            } else {
                str = {&*str_begin, dss_schimek::Length{str_len}};
            }

            assert(str_end != this->raw_strs.end());
            str_begin = str_end + 1, ++i;
        }
    };

    void send(int const dest, int const tag, RBC::Comm const& comm) const {
        (Members<StringSet>::send(dest, tag, comm), ...);
    }

    void recv(int const src, int const tag, RBC::Comm const& comm) {
        (Members<StringSet>::recv(src, tag, comm), ...);
    }

    void bcast_single(int const root, RBC::Comm const& comm) {
        (Members<StringSet>::bcast_single(root, comm), ...);
    }

private:
    static_assert(
        std::is_same_v<
            std::tuple_element_t<0, std::tuple<Members<StringSet>...>>,
            RawStrings<StringSet>>,
        "RawStrings must be the first Member"
    );
    static_assert(
        dss_schimek::has_member<typename StringSet::String, dss_schimek::Length>,
        "the string set must have a length member"
    );

    static constexpr bool has_index =
        (std::is_same_v<Members<StringSet>, Indices<StringSet>> || ...);
};

} // namespace _internal


template <typename StringSet>
using Data = std::conditional_t<
    StringSet::is_indexed,
    _internal::DataMembers<StringSet, _internal::RawStrings, _internal::Indices>,
    _internal::DataMembers<StringSet, _internal::RawStrings>>;

} // namespace RQuick
