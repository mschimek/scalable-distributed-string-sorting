/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <ips4o.hpp>
#include <mpi.h>
#include <tlx/sort/strings/radix_sort.hpp>

#include "./BinTreeMedianSelection.hpp"
#include "./RandomBitStore.hpp"
#include "sorter/distributed/duplicate_sorting.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"

namespace RQuick {

template <class StringContainer_, bool isIndexed>
struct Data {
    using StringContainer = StringContainer_;
    std::vector<unsigned char> rawStrings;
    std::vector<uint64_t> indices;
    static constexpr bool isIndexed_ = isIndexed;

    void clear() {
        rawStrings.clear();
        indices.clear();
        rawStrings.shrink_to_fit();
        indices.shrink_to_fit();
    }
    StringContainer moveToContainer() {
        if constexpr (isIndexed) {
            using dss_schimek::Index;
            using dss_schimek::make_initializer;
            return StringContainer{std::move(rawStrings), make_initializer<Index>(indices)};
        } else {
            return StringContainer{std::move(rawStrings)};
        }
    }

    Data exchange(MPI_Comm comm, int32_t target, int32_t tag) {
        using namespace dss_schimek::measurement;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();
        if constexpr (!isIndexed) {
            MPI_Request requests[2];
            measuringTool.addRawCommunication(rawStrings.size(), "");
            MPI_Isend(rawStrings.data(), rawStrings.size(), MPI_BYTE, target, tag, comm, requests);

            int recv_size = 0;
            MPI_Status status;
            MPI_Probe(target, tag, comm, &status);
            MPI_Get_count(&status, MPI_BYTE, &recv_size);

            Data returnData;
            returnData.rawStrings.resize(recv_size);
            MPI_Irecv(
                returnData.rawStrings.data(),
                recv_size,
                MPI_BYTE,
                target,
                tag,
                comm,
                requests + 1
            );
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            return returnData;
        } else {
            int32_t tag_idx = tag + 1;
            MPI_Request requests[4];

            int rank;
            MPI_Comm_rank(comm, &rank);

            int send_size_strs = rawStrings.size();
            int send_size_idxs = indices.size() * sizeof(uint64_t);

            measuringTool.addRawCommunication(send_size_strs, "");
            measuringTool.addRawCommunication(send_size_idxs, "");
            MPI_Isend(rawStrings.data(), send_size_strs, MPI_BYTE, target, tag, comm, requests);
            MPI_Isend(
                indices.data(),
                send_size_idxs,
                MPI_BYTE,
                target,
                tag_idx,
                comm,
                requests + 1
            );

            Data returnData;
            MPI_Status status[2];

            int recv_size_strs = 0;
            MPI_Probe(target, tag, comm, status);
            MPI_Get_count(status, MPI_BYTE, &recv_size_strs);
            returnData.rawStrings.resize(recv_size_strs);
            auto recv_buf_strs = returnData.rawStrings.data();
            MPI_Irecv(recv_buf_strs, recv_size_strs, MPI_BYTE, target, tag, comm, requests + 2);

            int recv_size_idxs = 0;
            MPI_Probe(target, tag_idx, comm, status + 1);
            MPI_Get_count(status + 1, MPI_BYTE, &recv_size_idxs);
            returnData.indices.resize(recv_size_idxs);
            auto recv_buf_idxs = returnData.indices.data();
            MPI_Irecv(recv_buf_idxs, recv_size_idxs, MPI_BYTE, target, tag_idx, comm, requests + 3);

            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            return returnData;
        }
    }

    void IReceiveAppend(MPI_Comm comm, int32_t source, int32_t tag) {
        if constexpr (!isIndexed) {
            MPI_Status status;

            MPI_Probe(source, tag, comm, &status);
            int recv_cnt = 0;
            MPI_Get_count(&status, MPI_BYTE, &recv_cnt);

            // Avoid reallocations later.
            rawStrings.reserve(2 * (rawStrings.size() + recv_cnt));

            rawStrings.resize(rawStrings.size() + recv_cnt);
            MPI_Request request;
            MPI_Irecv(
                rawStrings.data() + rawStrings.size() - recv_cnt,
                recv_cnt,
                MPI_BYTE,
                source,
                tag,
                comm,
                &request
            );
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        } else {
            int32_t tagIndices = tag + 1;
            MPI_Status status[2];
            MPI_Request request[2];

            MPI_Probe(source, tag, comm, status);
            int recv_cnt = 0;
            MPI_Get_count(status, MPI_BYTE, &recv_cnt);

            // Avoid reallocations later.
            rawStrings.reserve(2 * (rawStrings.size() + recv_cnt));
            rawStrings.resize(rawStrings.size() + recv_cnt);

            MPI_Irecv(
                rawStrings.data() + rawStrings.size() - recv_cnt,
                recv_cnt,
                MPI_BYTE,
                source,
                tag,
                comm,
                request
            );

            MPI_Probe(source, tagIndices, comm, status + 1);
            int recv_indices_cnt = 0;
            MPI_Get_count(status + 1, MPI_BYTE, &recv_indices_cnt);

            // Avoid reallocations later.
            indices.reserve(2 * (indices.size() + recv_indices_cnt));
            indices.resize(indices.size() + recv_indices_cnt);

            MPI_Irecv(
                indices.data() + indices.size() - recv_indices_cnt,
                recv_indices_cnt,
                MPI_BYTE,
                source,
                tagIndices,
                comm,
                request + 1
            );

            MPI_Waitall(2, request, MPI_STATUSES_IGNORE);
        }
    }
    Data Recv(MPI_Comm comm, int32_t source, int32_t tag) {
        if constexpr (!isIndexed) {
            Data<StringContainer, isIndexed> returnData;
            MPI_Status status;
            MPI_Probe(source, tag, comm, &status);
            int count = 0;
            MPI_Get_count(&status, MPI_BYTE, &count);
            returnData.rawStrings.resize(count);

            MPI_Recv(
                returnData.rawStrings.data(),
                count,
                MPI_BYTE,
                source,
                tag,
                comm,
                MPI_STATUS_IGNORE
            );
            return returnData;
        } else {
            int32_t tagIndices = tag + 1;
            Data<StringContainer, isIndexed> returnData;
            MPI_Status status[2];
            MPI_Probe(source, tag, comm, status);
            int charCount = 0;
            MPI_Get_count(status, MPI_BYTE, &charCount);
            returnData.rawStrings.resize(charCount);

            MPI_Recv(
                returnData.rawStrings.data(),
                charCount,
                MPI_BYTE,
                source,
                tag,
                comm,
                MPI_STATUS_IGNORE
            );

            MPI_Probe(source, tagIndices, comm, status + 1);
            int indexCount = 0;
            MPI_Get_count(status + 1, MPI_BYTE, &indexCount);
            returnData.indices.resize(indexCount);

            MPI_Recv(
                returnData.indices.data(),
                indexCount,
                MPI_BYTE,
                source,
                tagIndices,
                comm,
                MPI_STATUS_IGNORE
            );
            return returnData;
        }
    }
    void Send(MPI_Comm comm, int32_t target, int32_t tag) {
        using namespace dss_schimek::measurement;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();
        measuringTool.addRawCommunication(rawStrings.size(), "");
        MPI_Send(rawStrings.data(), rawStrings.size(), MPI_BYTE, target, tag, comm);
        if constexpr (isIndexed) {
            int32_t tagIndices = tag + 1;
            measuringTool.addRawCommunication(indices.size() * sizeof(uint64_t), "");
            MPI_Send(
                indices.data(),
                indices.size() * sizeof(uint64_t),
                MPI_BYTE,
                target,
                tagIndices,
                comm
            );
        }
    }
}; // namespace RQuick

namespace _internal {
constexpr bool debugQuicksort = false;
constexpr bool barrierActive = false;

template <class Communicator>
inline void split(Communicator& comm, Communicator* subcomm) {
    int32_t myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    bool is_left_group = myrank < (nprocs / 2);
    int32_t colour = is_left_group;

    MPI_Comm_split(comm, colour, myrank, subcomm);
}

/*
 * @brief Returns the k middle most elements.
 *
 * Returns the k middle most elements of v if v contains at least k
 * elements. Otherwise, v is returned. This function uses
 * randomization to decide which elements are the 'middle most'
 * elements if v contains an odd number of elements and k is an even
 * number or if v contains an even number of elements and k is an odd
 * number.
 */
template <class StringContainer>
Data<StringContainer, StringContainer::is_indexed> middleMostElements(
    StringContainer& cont, size_t k, std::mt19937_64& async_gen, RandomBitStore& bit_gen
) {
    Data<StringContainer, StringContainer::is_indexed> data;
    if (cont.size() <= k) {
        data.rawStrings = cont.raw_strings();
        if constexpr (StringContainer::is_indexed) {
            data.indices.reserve(cont.size());
            for (auto str: cont.make_string_set())
                data.indices.push_back(str.index);
        }
        return data;
    }

    auto const offset = (cont.size() - k) / 2;
    bool const shiftCondition =
        ((cont.size() % 2) == 0 && ((k % 2) == 0)) || (cont.size() % 2 == 1 && ((k % 2) == 1));
    int64_t shift = shiftCondition ? 0 : bit_gen.getNextBit(async_gen);
    const uint64_t begin = offset + shift;
    auto ss = cont.make_string_set();

    for (size_t i = begin; i < begin + k; ++i) {
        auto const str = ss[ss.begin() + i];
        auto const length = ss.get_length(str) + 1;
        auto chars = ss.get_chars(str, 0);
        std::copy_n(chars, length, std::back_inserter(data.rawStrings));
        if constexpr (StringContainer::is_indexed) {
            data.indices.push_back(str.index);
        }
    }
    return data;
}

/*
 * @brief Distributed splitter selection with a binary reduction tree.
 *
 * @param v Local input. The input must be sorted.
 */
template <class Comp, class StringContainer, class Communicator>
Data<StringContainer, StringContainer::is_indexed> selectSplitter(
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    StringContainer& stringContainer,
    MPI_Datatype mpi_type,
    Comp&& comp,
    int tag,
    Communicator& comm
) {
    Data<StringContainer, StringContainer::is_indexed> local_medians =
        middleMostElements(stringContainer, 2, async_gen, bit_gen);

    auto res = BinTreeMedianSelection::select(
        std::move(local_medians),
        2,
        std::forward<Comp>(comp),
        mpi_type,
        async_gen,
        bit_gen,
        tag,
        comm
    );
    if constexpr (debugQuicksort) {
        if (res.rawStrings.size() < 1 || res.rawStrings.back() != 0) {
            std::cout << "error in final median" << std::endl;
            std::abort();
        }
    }
    return res;
}

/*
 * @brief Split vector according to a given splitter with or without
 * tie-breaking.
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
template <class T, class Comp>
T const* locateSplitter(
    std::vector<T> const& v,
    Comp&& comp,
    T const& splitter,
    std::mt19937_64& gen,
    RandomBitStore& bit_store,
    bool is_robust
) {
    auto const begin_equal_els =
        std::lower_bound(v.data(), v.data() + v.size(), splitter, std::forward<Comp>(comp));
    if (!is_robust) {
        return begin_equal_els;
    }

    auto const end_equal_els =
        std::upper_bound(begin_equal_els, v.data() + v.size(), splitter, std::forward<Comp>(comp));

    // Round down or round up randomly.
    auto const opt_split =
        v.data() + v.size() / 2 + ((v.size() % 2) == 1 && bit_store.getNextBit(gen));

    if (begin_equal_els < opt_split) {
        return std::min(opt_split, end_equal_els);
    } else {
        return begin_equal_els;
    }
}

template <class T, class Communicator>
void exchange(
    T const* send_begin,
    T const* send_end,
    std::vector<T>& v_recv,
    int target,
    MPI_Datatype mpi_type,
    int tag,
    Communicator& comm
) {
    auto const send_size = send_end - send_begin;

    MPI_Request requests[2];
    MPI_Isend(send_begin, send_size, mpi_type, target, tag, comm, requests);

    int recv_size = 0;
    MPI_Status status;
    MPI_Probe(target, tag, comm, &status);
    MPI_Get_count(&status, mpi_type, &recv_size);

    v_recv.resize(recv_size);
    MPI_Irecv(v_recv.data(), recv_size, mpi_type, target, tag, comm, requests + 1);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
}

template <class T, class Comp>
void merge(T const* begin1, T const* end1, T const* begin2, T const* end2, T* tbegin, Comp&& comp) {
    assert(std::is_sorted(begin1, end1, std::forward<Comp>(comp)));
    assert(std::is_sorted(begin2, end2, std::forward<Comp>(comp)));

    std::merge(begin1, end1, begin2, end2, tbegin, std::forward<Comp>(comp));
}

template <class Tracker, class Comp, class StringContainer, class Communicator>
StringContainer sortRec(
    std::mt19937_64& gen,
    RandomBitStore& bit_store,
    StringContainer&& stringContainer,
    Comp&& comp,
    MPI_Datatype mpi_type,
    bool is_robust,
    Tracker&& tracker,
    int tag,
    Communicator& comm
) {
    using StringSet = typename StringContainer::StringSet;
    using String = typename StringSet::String;
    tracker.median_select_t.start(comm);

    int32_t nprocs;
    int32_t myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    assert(nprocs >= 2);
    assert(tlx::integer_log2_floor(nprocs));

    auto const is_left_group = myrank < nprocs / 2;

    // Select pivot globally with binary tree median selection.
    auto pivot = selectSplitter(
        gen,
        bit_store,
        stringContainer,
        mpi_type,
        std::forward<Comp>(comp),
        tag,
        comm
    );

    if constexpr (debugQuicksort) {
        if (StringContainer::is_indexed && pivot.indices.size() != 1) {
            std::cout << "pivot does not contain exactly one string!";
            std::abort();
        }
    }

    String pivotString;
    dss_schimek::Length len{pivot.rawStrings.size() - 1};
    if constexpr (StringContainer::is_indexed) {
        pivotString = {pivot.rawStrings.data(), len, dss_schimek::Index{pivot.indices.front()}};
    } else {
        pivotString = {pivot.rawStrings.data(), len};
    }
    tracker.median_select_t.stop();

    // Partition data into small elements and large elements.

    tracker.partition_t.start(comm);

    auto const* separator = locateSplitter(
        stringContainer.get_strings(),
        std::forward<Comp>(comp),
        pivotString,
        gen,
        bit_store,
        is_robust
    );

    String* send_begin = stringContainer.get_strings().data();
    String* send_end = const_cast<String*>(separator);
    String* own_begin = const_cast<String*>(separator);
    String* own_end = stringContainer.get_strings().data() + stringContainer.get_strings().size();
    if (is_left_group) {
        std::swap(send_begin, own_begin);
        std::swap(send_end, own_end);
    }

    uint64_t sendCounts = 0;
    for (auto i = send_begin; i < send_end; ++i) {
        sendCounts += i->getLength() + 1;
    }
    Data<StringContainer, StringContainer::is_indexed> exchangeData;
    exchangeData.rawStrings.resize(sendCounts);
    if constexpr (StringContainer::is_indexed)
        exchangeData.indices.resize(send_end - send_begin);
    uint64_t curPos = 0;
    for (auto i = send_begin; i < send_end; ++i) {
        auto length = i->getLength() + 1;
        auto chars = i->getChars();
        std::copy_n(chars, length, exchangeData.rawStrings.begin() + curPos);
        curPos += length;
        if constexpr (StringContainer::is_indexed)
            exchangeData.indices[i - send_begin] = i->index;
    }
    const uint64_t ownCharsSize = stringContainer.char_size() - exchangeData.rawStrings.size();

    tracker.partition_t.stop();

    // Move elements to partner and receive elements for own group.
    tracker.exchange_t.start(comm);

    auto const partner = (myrank + (nprocs / 2)) % nprocs;

    Data<StringContainer, StringContainer::is_indexed> recvData =
        exchangeData.exchange(comm, partner, tag);
    exchangeData.clear();

    StringContainer recvStrings = recvData.moveToContainer();
    recvData.clear();

    tracker.exchange_t.stop();

    // Merge received elements with own elements.
    tracker.merge_t.start(comm);

    auto const num_elements = recvStrings.size() + (own_end - own_begin);
    std::vector<String> mergedStrings(num_elements);
    merge(
        own_begin,
        own_end,
        recvStrings.get_strings().begin(),
        recvStrings.get_strings().end(),
        mergedStrings.begin(),
        std::forward<Comp>(comp)
    );
    std::vector<unsigned char> mergedRawStrings(recvStrings.char_size() + ownCharsSize);

    curPos = 0;
    std::vector<uint64_t> mergedStringsIndices;
    if constexpr (StringContainer::is_indexed) {
        mergedStringsIndices.reserve(mergedStrings.size());
    }
    for (auto str: mergedStrings) {
        auto length = str.getLength() + 1;
        auto chars = str.getChars();
        std::copy_n(chars, length, mergedRawStrings.begin() + curPos);
        curPos += length;
        if constexpr (StringContainer::is_indexed)
            mergedStringsIndices.push_back(str.index);
    }

    if constexpr (StringContainer::is_indexed) {
        using dss_schimek::Index;
        using dss_schimek::make_initializer;
        stringContainer.update(
            std::move(mergedRawStrings),
            make_initializer<Index>(mergedStringsIndices)
        );
        mergedStringsIndices.clear();
        mergedStringsIndices.shrink_to_fit();
    } else {
        stringContainer.update(std::move(mergedRawStrings));
    }

    mergedStrings.clear();
    mergedStrings.shrink_to_fit();
    recvStrings.delete_all();

    tracker.merge_t.stop();

    if (nprocs >= 4) {
        // Split communicator and solve subproblems.
        tracker.comm_split_t.start(comm);

        MPI_Comm subcomm;
        split(comm, &subcomm);

        tracker.comm_split_t.stop();

        return sortRec<StringContainer::is_indexed>(
            gen,
            bit_store,
            std::move(stringContainer),
            std::forward<Comp>(comp),
            mpi_type,
            is_robust,
            std::forward<Tracker>(tracker),
            tag,
            subcomm
        );
    }

    return std::move(stringContainer);
}

template <typename StringContainer>
void sortLocally(StringContainer& container) {
    if constexpr (StringContainer::is_indexed) {
        std::vector<uint64_t> lcp(container.size(), 0);
        auto strptr =
            tlx::sort_strings_detail::StringLcpPtr(container.make_string_set(), lcp.data());
        tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
        dss_mehnert::sort_duplicates(strptr);
    } else {
        tlx::sort_strings_detail::radixsort_CI3(container.make_string_ptr(), 0, 0);
    }
}

template <class Tracker, class Data, class Comp, class Communicator>
typename Data::StringContainer sort(
    std::mt19937_64& async_gen,
    Data&& data,
    MPI_Datatype mpi_type,
    int tag,
    Communicator comm,
    Tracker&& tracker,
    Comp&& comp,
    bool is_robust
) {
    using StringContainer = typename Data::StringContainer;

    int32_t nprocs;
    int32_t myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    if (nprocs == 1) {
        StringContainer container = data.moveToContainer();
        tracker.local_sort_t.start(comm);
        sortLocally(container);
        tracker.local_sort_t.stop();

        return container;
    }

    tracker.move_to_pow_of_two_t.start(comm);

    auto const pow = tlx::round_down_to_power_of_two(nprocs);

    // Send data to a smaller hypercube if the number of processes is
    // not a power of two.
    if (myrank < nprocs - pow) {
        // Not a power of two but we are part of the smaller hypercube
        // and receive elements.

        auto const source = pow + myrank;

        data.IReceiveAppend(comm, source, tag);
        MPI_Comm sub_comm;
        // MPI_Comm_create_group(comm, &sub_comm, 0, pow - 1);
        MPI_Comm_split(comm, 0, myrank, &sub_comm);
        comm = sub_comm;
    } else if (myrank >= pow) {
        // Not a power of two and we are not part of the smaller
        // hypercube.

        auto const target = myrank - pow;
        data.Send(comm, target, tag);
        data.clear();

        // This process is not part of 'sub_comm'. We call
        // this function to support MPI implementations
        // without MPI_Comm_create_group.
        MPI_Comm sub_comm;
        MPI_Comm_split(comm, 1, myrank, &sub_comm);
        comm = sub_comm;

        tracker.move_to_pow_of_two_t.stop();

        return StringContainer();
    } else if (pow != nprocs) {
        // Not a power of two but we are part of the smaller hypercube
        // and do not receive elements.

        MPI_Comm sub_comm;
        MPI_Comm_split(comm, 0, myrank, &sub_comm);
        comm = sub_comm;

        // Avoid reallocations later.
        // v.reserve(3 * v.size());
    } else {
        // The number of processes is a power of two.

        // Avoid reallocations later.
        // v.reserve(2 * v.size());
    }

    StringContainer container = data.moveToContainer();
    data.clear();
    tracker.move_to_pow_of_two_t.stop();

    tracker.parallel_shuffle_t.start(comm);
    tracker.parallel_shuffle_t.stop();

    tracker.local_sort_t.start(comm);
    sortLocally(container);
    tracker.local_sort_t.stop();

    RandomBitStore bit_store;
    return _internal::sortRec<Data::isIndexed_>(
        async_gen,
        bit_store,
        std::move(container),
        std::forward<Comp>(comp),
        mpi_type,
        is_robust,
        std::forward<Tracker>(tracker),
        tag,
        comm
    );
}

class DummyTracker {
public:
    class DummyTimer {
    public:
        DummyTimer() {}

        void start(MPI_Comm&) {}
        void stop() {}
    };

    DummyTimer local_sort_t;
    DummyTimer exchange_t;
    DummyTimer parallel_shuffle_t;
    DummyTimer merge_t;
    DummyTimer median_select_t;
    DummyTimer partition_t;
    DummyTimer comm_split_t;
    DummyTimer move_to_pow_of_two_t;
};

} // namespace _internal

template <class Tracker, class Comp, class Data>
typename Data::StringContainer sort(
    Tracker&& tracker,
    std::mt19937_64& async_gen,
    Data&& data,
    MPI_Datatype mpi_type,
    int tag,
    MPI_Comm mpi_comm,
    Comp&& comp,
    bool is_robust
) {
    return RQuick::_internal::sort(
        async_gen,
        std::forward<Data>(data),
        mpi_type,
        tag,
        mpi_comm,
        std::forward<Tracker>(tracker),
        comp,
        is_robust
    );
}

template <class Comp, class Data>
typename Data::StringContainer sort(
    std::mt19937_64& async_gen,
    Data&& data,
    MPI_Datatype mpi_type,
    int tag,
    MPI_Comm mpi_comm,
    Comp&& comp,
    bool is_robust
) {
    _internal::DummyTracker tracker;
    return RQuick::_internal::sort(
        async_gen,
        std::forward<Data>(data),
        mpi_type,
        tag,
        mpi_comm,
        tracker,
        comp,
        is_robust
    );
}

} // namespace RQuick
