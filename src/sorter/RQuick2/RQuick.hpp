/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2023, Pascal Mehnert <pascal.mehnert@student.kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/


#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <RBC.hpp>
#include <ips4o.hpp>
#include <kamping/mpi_datatype.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "./BinTreeMedianSelection.hpp"
#include "./RandomBitStore.hpp"
#include "sorter/RQuick2/Util.hpp"
#include "sorter/distributed/duplicate_sorting.hpp"

namespace Tools {

class DummyTimer {
public:
    DummyTimer() {}

    void start(RBC::Comm const&) {}
    void stop() {}
};

} // namespace Tools


namespace RQuick2 {
namespace _internal {

template <typename StringPtr>
using PairItSizeT = std::pair<typename StringPtr::StringSet::Iterator, size_t>;

inline void split(RBC::Comm const& comm, RBC::Comm* subcomm) {
    auto const nprocs = comm.getSize();
    auto const myrank = comm.getRank();

    bool const is_left_group = myrank < (nprocs / 2);

    int first = 0, last = 0;
    if (is_left_group) {
        first = 0;
        last = nprocs / 2 - 1;
    } else {
        first = nprocs / 2;
        last = nprocs - 1;
    }

    RBC::Comm_create_group(comm, subcomm, first, last);
}

/*
 * @brief Returns the k middle most elements.
 *
 * Returns the k middle most elements of ss if ss contains at least k
 * elements. Otherwise, ss is returned. This function uses
 * randomization to decide which elements are the 'middle most'
 * elements.
 */
template <class StringPtr>
StringPtr middleMostElements(
    StringPtr const& strptr, size_t const k, std::mt19937_64& async_gen, RandomBitStore& bit_gen
) {
    if (strptr.size() <= k) {
        return strptr;
    }

    auto const offset = (strptr.size() - k) / 2;

    if (strptr.size() % 2 == k % 2) {
        return strptr.sub(offset, k);
    } else {
        // Round down or round up randomly.
        auto const shift = bit_gen.getNextBit(async_gen);
        return strptr.sub(offset + shift, k);
    }
}

/*
 * @brief Distributed splitter selection with a binary reduction tree.
 *
 * @param ss Local input. The input must be sorted.
 */
template <class StringPtr>
StringT<StringPtr> selectSplitter(
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    StringPtr const& strptr,
    TemporaryBuffers<StringPtr>& buffers,
    int const tag,
    RBC::Comm const& comm
) {
    assert(strptr.active().check_order());

    auto const local_medians = middleMostElements(strptr, 2, async_gen, bit_gen);
    return BinTreeMedianSelection::select(local_medians, buffers, 2, async_gen, bit_gen, tag, comm);
}

/*
 * @brief Split vector according to a given splitter with or without tie-breaking.
 *
 * No tie-breaking: Returns a pointer to the first element which is
 * larger or equal to the splitter.
 *
 * With tie-breaking: Splits the vector according to a given splitter
 * such that the split is as close to the middle of the vector as
 * possible.
 *
 * @param v Local input. The input must be sorted.
 */
template <class StringPtr>
PairItSizeT<StringPtr> locateSplitterLeft(
    StringPtr const& strptr,
    StringT<StringPtr> const& splitter,
    int const partner,
    int const tag,
    RBC::Comm const& comm
) {
    auto const mpi_type = kamping::mpi_datatype<ptrdiff_t>();

    auto const& ss = strptr.active();
    auto const begin = ss.begin(), end = ss.end();

    if constexpr (StringPtr::StringSet::is_indexed) {
        auto const [begin_geq, _] = _internal::lower_bound(strptr, splitter);

        ptrdiff_t send_cnt = end - begin_geq;
        ptrdiff_t recv_cnt = -1;
        // clang-format off
        RBC::Sendrecv(&send_cnt, 1, mpi_type, partner, tag,
                      &recv_cnt, 1, mpi_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);
        // clang-format on
        _internal::add_comm_volume<ptrdiff_t>(1);
        return {begin_geq, recv_cnt};
    } else {
        auto const [begin_geq, end_leq] = _internal::lower_upper_bound(strptr, splitter);

        // m: my; t: theirs
        // Number of elemements < or == or > than the splitter
        ptrdiff_t const mleft = begin_geq - begin;
        ptrdiff_t const mright = end - end_leq;
        ptrdiff_t const mmiddle = ss.size() - mleft - mright;

        std::array<ptrdiff_t, 3> send = {mleft, mright, mmiddle};
        std::array<ptrdiff_t, 3> recv;
        // clang-format off
        RBC::Sendrecv(send.data(), 3, mpi_type, partner, tag,
                      recv.data(), 3, mpi_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);
        // clang-format on
        _internal::add_comm_volume<ptrdiff_t>(3);
        auto [tleft, tright, tmiddle] = recv;

        ptrdiff_t const split = (ss.size() + tleft + tright + tmiddle) / 2;
        ptrdiff_t const mkeep = std::min(mmiddle, std::max(split - mleft - tleft, ptrdiff_t{0}));
        ptrdiff_t const tkeep = std::min(tmiddle, std::max(split - tright - mright, ptrdiff_t{0}));

        return {begin + mleft + mkeep, tleft + tmiddle - tkeep};
    }
}

/*
 * @brief Split vector according to a given splitter with or without tie-breaking.
 *
 * No tie-breaking: Returns a pointer to the first element which is
 * larger or equal to the splitter.
 *
 * With tie-breaking: Splits the vector according to a given splitter
 * such that the split is as close to the middle of the vector as
 * possible.
 *
 * @param v Local input. The input must be sorted.
 */
template <class StringPtr>
PairItSizeT<StringPtr> locateSplitterRight(
    StringPtr const& strptr,
    StringT<StringPtr> const& splitter,
    int const partner,
    int const tag,
    RBC::Comm const& comm
) {
    auto const& ss = strptr.active();
    auto const begin = ss.begin(), end = ss.end();

    auto const mpi_type = kamping::mpi_datatype<ptrdiff_t>();
    if constexpr (StringPtr::StringSet::is_indexed) {
        auto const [begin_geq, _] = _internal::lower_bound(strptr, splitter);

        ptrdiff_t send_cnt = begin_geq - begin;
        ptrdiff_t recv_cnt = -1;
        // clang-format off
        RBC::Sendrecv(&send_cnt, 1, mpi_type, partner, tag,
                      &recv_cnt, 1, mpi_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);
        // clang-format on
        _internal::add_comm_volume<ptrdiff_t>(1);
        return {begin_geq, recv_cnt};
    } else {
        auto const [begin_geq, end_leq] = _internal::lower_upper_bound(strptr, splitter);

        // m: my; t: theirs
        ptrdiff_t const mleft = begin_geq - begin;
        ptrdiff_t const mright = end - end_leq;
        ptrdiff_t const mmiddle = ss.size() - mleft - mright;

        std::array<ptrdiff_t, 3> send = {mleft, mright, mmiddle};
        std::array<ptrdiff_t, 3> recv;
        // clang-format off
        RBC::Sendrecv(send.data(), 3, mpi_type, partner, tag,
                      recv.data(), 3, mpi_type, partner, tag,
                      comm, MPI_STATUS_IGNORE);
        // clang-format on
        _internal::add_comm_volume<ptrdiff_t>(3);
        auto [tleft, tright, tmiddle] = recv;

        ptrdiff_t const split = (ss.size() + tleft + tright + tmiddle) / 2;
        ptrdiff_t const mkeep = std::min(mmiddle, std::max(split - mright - tright, ptrdiff_t{0}));
        ptrdiff_t const tkeep = std::min(tmiddle, std::max(split - tleft - mleft, ptrdiff_t{0}));

        return {end - mright - mkeep, tright + tmiddle - tkeep};
    }
}

template <class StringPtr, class Tracker>
void sortRec(
    std::mt19937_64& gen,
    RandomBitStore& bit_store,
    Container<StringPtr>& local_strings,
    TemporaryBuffers<StringPtr>& buffers,
    Tracker&& tracker,
    int const tag,
    const RBC::Comm& comm
) {
    tracker.median_select_t.start(comm);

    auto const nprocs = comm.getSize();
    auto const myrank = comm.getRank();

    assert(nprocs >= 2);
    assert(local_strings.make_string_set().check_order());
    assert(tlx::is_power_of_two(nprocs));

    auto const is_left_group = myrank < nprocs / 2;

    // Select pivot globally with binary tree median selection.
    auto const strptr = local_strings.make_auto_ptr();
    auto const pivot = selectSplitter(gen, bit_store, strptr, buffers, tag, comm);
    tracker.median_select_t.stop();

    // Partition data into small elements and large elements.

    tracker.partition_t.start(comm);
    int const partner = (myrank + (nprocs / 2)) % nprocs;
    auto const [own_ptr, send_ptr, separator, recv_cnt] = [&] {
        if (is_left_group) {
            auto const [separator, recv_cnt] =
                locateSplitterLeft(strptr, pivot, partner, tag, comm);

            auto own_ptr = strptr.sub(0, separator - strptr.active().begin());
            auto send_ptr = strptr.sub(own_ptr.size(), strptr.size() - own_ptr.size());
            return std::tuple{own_ptr, send_ptr, separator, recv_cnt};
        } else {
            auto const [separator, recv_cnt] =
                locateSplitterRight(strptr, pivot, partner, tag, comm);

            auto send_ptr = strptr.sub(0, separator - strptr.active().begin());
            auto own_ptr = strptr.sub(send_ptr.size(), strptr.size() - send_ptr.size());
            return std::tuple{own_ptr, send_ptr, separator, recv_cnt};
        }
    }();
    tracker.partition_t.stop();

    // Move elements to partner and receive elements for own group.
    tracker.exchange_t.start(comm);
    buffers.send_data.write(send_ptr);
    buffers.send_data.sendrecv(buffers.recv_data, recv_cnt, partner, tag, comm);

    buffers.recv_strings.resize_strings(buffers.recv_data.get_num_strings());
    buffers.recv_data.read_into(buffers.recv_strings.make_auto_ptr());
    tracker.exchange_t.stop();

    // Merge received elements with own elements.
    tracker.merge_t.start(comm);
    merge(own_ptr, buffers.recv_strings.make_auto_ptr(), buffers.merge_strings);
    buffers.merge_strings.orderRawStrings(buffers.char_buffer);

    using std::swap;
    swap(local_strings, buffers.merge_strings);
    assert(local_strings.make_string_set().check_order());

    if (!local_strings.make_string_set().check_order()) {
        local_strings.make_string_set().print();
    }

    tracker.merge_t.stop();

    if (nprocs >= 4) {
        // Split communicator and solve subproblems.
        tracker.comm_split_t.start(comm);

        RBC::Comm subcomm;
        split(comm, &subcomm);

        tracker.comm_split_t.stop();

        sortRec(
            gen,
            bit_store,
            local_strings,
            buffers,
            std::forward<Tracker>(tracker),
            tag,
            subcomm
        );
    }
}

template <class StringPtr>
void sortLocally(StringPtr const& strptr) {
    if constexpr (StringPtr::StringSet::is_indexed) {
        if constexpr (StringPtr::with_lcp) {
            tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
            dss_mehnert::sort_duplicates(strptr);
        } else {
            using StringSet = typename StringPtr::StringSet;
            using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

            std::vector<size_t> lcps(strptr.size());
            StringLcpPtr augmented_ptr{strptr.active(), lcps.data()};
            tlx::sort_strings_detail::radixsort_CI3(augmented_ptr, 0, 0);
            dss_mehnert::sort_duplicates(augmented_ptr);
        }
    } else {
        tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
    }
}

template <class Tracker, class StringPtr>
Container<StringPtr> sort(
    std::mt19937_64& async_gen,
    Data<StringPtr>&& local_data,
    int const tag,
    Tracker&& tracker,
    RBC::Comm comm
) {
    if (comm.getSize() == 1) {
        tracker.local_sort_t.start(comm);
        Container<StringPtr> local_strings;
        local_strings.resize_strings(local_data.get_num_strings());
        local_data.read_into(local_strings.make_auto_ptr());
        std::swap(local_strings.raw_strings(), local_data.raw_strs);
        sortLocally(local_strings.make_auto_ptr());
        tracker.local_sort_t.stop();

        return local_strings;
    }

    tracker.move_to_pow_of_two_t.start(comm);

    auto const pow = tlx::round_down_to_power_of_two(comm.getSize());

    // Send data to a smaller hypercube if the number of processes is
    // not a power of two.
    if (comm.getRank() < comm.getSize() - pow) {
        // Not a power of two but we are part of the smaller hypercube
        // and receive elements.

        auto const source = pow + comm.getRank();
        local_data.recv(source, tag, comm, true, false);

        RBC::Comm sub_comm;
        RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
        comm = sub_comm;
    } else if (comm.getRank() >= pow) {
        // Not a power of two and we are not part of the smaller
        // hypercube.

        auto const target = comm.getRank() - pow;
        local_data.send(target, tag, comm, false);

        // This process is not part of 'sub_comm'. We call
        // this function to support MPI implementations
        // without MPI_Comm_create_group.
        RBC::Comm sub_comm;
        RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
        comm = sub_comm;

        tracker.move_to_pow_of_two_t.stop();

        return {};
    } else if (pow != comm.getSize()) {
        // Not a power of two but we are part of the smaller hypercube
        // and do not receive elements.

        RBC::Comm sub_comm;
        RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
        comm = sub_comm;
    } else {
        // The number of processes is a power of two.
    }

    tracker.move_to_pow_of_two_t.stop();

    assert(tlx::is_power_of_two(comm.getSize()));

    Container<StringPtr> local_strings;
    local_strings.resize_strings(local_data.get_num_strings());
    local_data.read_into(local_strings.make_auto_ptr());
    std::swap(local_strings.raw_strings(), local_data.raw_strs);

    TemporaryBuffers<StringPtr> buffers;
    buffers.recv_data = std::move(local_data);

    tracker.local_sort_t.start(comm);
    sortLocally(local_strings.make_auto_ptr());
    tracker.local_sort_t.stop();

    RandomBitStore bit_store;
    _internal::sortRec(
        async_gen,
        bit_store,
        local_strings,
        buffers,
        std::forward<Tracker>(tracker),
        tag,
        comm
    );
    return local_strings;
}

class DummyTracker {
public:
    Tools::DummyTimer local_sort_t;
    Tools::DummyTimer exchange_t;
    Tools::DummyTimer merge_t;
    Tools::DummyTimer median_select_t;
    Tools::DummyTimer partition_t;
    Tools::DummyTimer comm_split_t;
    Tools::DummyTimer move_to_pow_of_two_t;
};

} // namespace _internal

template <class Tracker, class StringPtr>
Container<StringPtr> sort(
    Tracker&& tracker,
    Data<StringPtr>&& data,
    int const tag,
    std::mt19937_64& async_gen,
    MPI_Comm const mpi_comm
) {
    RBC::Comm comm;
    RBC::Create_Comm_from_MPI(mpi_comm, &comm);
    return _internal::sort(
        async_gen,
        std::move(data),
        tag,
        std::forward<Tracker>(tracker),
        std::move(comm)
    );
}

template <class StringPtr>
Container<StringPtr>
sort(Data<StringPtr>&& data, int const tag, std::mt19937_64& async_gen, MPI_Comm const mpi_comm) {
    _internal::DummyTracker tracker;
    return sort(tracker, std::move(data), tag, async_gen, mpi_comm);
}

} // namespace RQuick2
