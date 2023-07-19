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
#include "sorter/distributed/duplicateSorting.hpp"
#include "util/measuringTool.hpp"

namespace Tools {
class DummyTimer {
public:
    DummyTimer() {}

    void start(MPI_Comm&) {}
    void stop() {}
};

} // namespace Tools

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
            return StringContainer(std::move(rawStrings), indices);
        } else {
            return StringContainer(std::move(rawStrings));
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
Data<StringContainer, StringContainer::isIndexed> middleMostElements(
    StringContainer& cont, size_t k, std::mt19937_64& async_gen, RandomBitStore& bit_gen
) {
    Data<StringContainer, StringContainer::isIndexed> data;
    if (cont.size() <= k) {
        data.rawStrings = cont.raw_strings();
        if constexpr (StringContainer::isIndexed) {
            data.indices.reserve(cont.size());
            for (auto str: cont.make_string_set()) data.indices.push_back(str.index);
        }
        return data;
    }

    auto const offset = (cont.size() - k) / 2;
    bool const shiftCondition =
        ((cont.size() % 2) == 0 && ((k % 2) == 0)) || (cont.size() % 2 == 1 && ((k % 2) == 1));
    int64_t shift = shiftCondition ? 0 : bit_gen.getNextBit(async_gen);
    const uint64_t begin = offset + shift;
    auto ss = cont.make_string_set();

    if constexpr (debugQuicksort) {
        if (begin + k > cont.size()) {
            std::cout << "out of bounds error in middleMostElements, container size: "
                      << cont.size() << " offset: " << offset << " k: " << k << " shift " << shift
                      << std::endl;
            std::abort();
        }
    }
    for (size_t i = begin; i < begin + k; ++i) {
        auto const str = ss[ss.begin() + i];
        auto const length = ss.get_length(str) + 1;
        auto chars = ss.get_chars(str, 0);
        std::copy_n(chars, length, std::back_inserter(data.rawStrings));
        if constexpr (StringContainer::isIndexed) {
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
Data<StringContainer, StringContainer::isIndexed> selectSplitter(
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    StringContainer& stringContainer,
    MPI_Datatype mpi_type,
    Comp&& comp,
    int tag,
    Communicator& comm
) {
    // assert StringContainer is sorted
    if constexpr (debugQuicksort) {
        if (!stringContainer.isConsistent()) {
            std::cout << "corrupt string cont" << std::endl;
            std::abort();
        }
    }
    Data<StringContainer, StringContainer::isIndexed> local_medians =
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

template <bool isIndexed, class Tracker, class Comp, class StringContainer, class Communicator>
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
    using dss_schimek::measurement::MeasuringTool;
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
        if (StringContainer::isIndexed && pivot.indices.size() != 1) {
            std::cout << "pivot does not contain exactly one string!";
            std::abort();
        }
    }

    tracker.median_select_t.stop();
    String pivotString(pivot.rawStrings.data(), pivot.rawStrings.size() - 1);
    if constexpr (StringContainer::isIndexed)
        pivotString.index = pivot.indices.front();
    // measuringTool.stop("Splitter_median_select");

    // Partition data into small elements and large elements.

    // measuringTool.start("Splitter_partition");
    tracker.partition_t.start(comm);

    auto const* separator = locateSplitter(
        stringContainer.getStrings(),
        std::forward<Comp>(comp),
        pivotString,
        gen,
        bit_store,
        is_robust
    );
    if constexpr (debugQuicksort) {
        if (separator < stringContainer.getStrings().data()
            || separator > stringContainer.getStrings().data() + stringContainer.size()) {
            std::cout << "error in locate splitter" << std::endl;
            std::abort();
        }
    }

    if constexpr (debugQuicksort) {
        const uint64_t partitionSize = separator - stringContainer.getStrings().data();
        std::cout << "rank: " << myrank << " size: " << partitionSize << " "
                  << stringContainer.size() - partitionSize << std::endl;
    }

    String* send_begin = stringContainer.getStrings().data();
    String* send_end = const_cast<String*>(separator);
    String* own_begin = const_cast<String*>(separator);
    String* own_end = stringContainer.getStrings().data() + stringContainer.getStrings().size();
    if (is_left_group) {
        std::swap(send_begin, own_begin);
        std::swap(send_end, own_end);
    }

    uint64_t sendCounts = 0;
    for (auto i = send_begin; i < send_end; ++i) {
        sendCounts += i->getLength() + 1;
    }
    Data<StringContainer, StringContainer::isIndexed> exchangeData;
    exchangeData.rawStrings.resize(sendCounts);
    if constexpr (StringContainer::isIndexed)
        exchangeData.indices.resize(send_end - send_begin);
    uint64_t curPos = 0;
    for (auto i = send_begin; i < send_end; ++i) {
        auto length = i->getLength() + 1;
        auto chars = i->getChars();
        std::copy_n(chars, length, exchangeData.rawStrings.begin() + curPos);
        curPos += length;
        if constexpr (StringContainer::isIndexed)
            exchangeData.indices[i - send_begin] = i->index;
    }
    const uint64_t ownCharsSize = stringContainer.char_size() - exchangeData.rawStrings.size();

    //    uint64_t inbalance = std::abs(
    //        static_cast<int64_t>(stringContainer.size()) - (send_end -
    //        send_begin));
    // measuringTool.add(inbalance, "inbalance", false);

    tracker.partition_t.stop();
    // measuringTool.stop("Splitter_partition");

    // Move elements to partner and receive elements for own group.
    tracker.exchange_t.start(comm);
    // measuringTool.start("Splitter_exchange");

    auto const partner = (myrank + (nprocs / 2)) % nprocs;

    Data<StringContainer, StringContainer::isIndexed> recvData =
        exchangeData.exchange(comm, partner, tag);
    exchangeData.clear();

    StringContainer recvStrings = recvData.moveToContainer();
    recvData.clear();

    tracker.exchange_t.stop();
    // measuringTool.stop("Splitter_exchange");
    // Merge received elements with own elements.
    tracker.merge_t.start(comm);
    // measuringTool.start("Splitter_merge");


    auto const num_elements = recvStrings.size() + (own_end - own_begin);
    std::vector<String> mergedStrings(num_elements);
    merge(
        own_begin,
        own_end,
        recvStrings.getStrings().begin(),
        recvStrings.getStrings().end(),
        mergedStrings.begin(),
        std::forward<Comp>(comp)
    );
    std::vector<unsigned char> mergedRawStrings(recvStrings.char_size() + ownCharsSize);

    curPos = 0;
    std::vector<uint64_t> mergedStringsIndices;
    if constexpr (StringContainer::isIndexed) {
        mergedStringsIndices.reserve(mergedStrings.size());
    }
    for (auto str: mergedStrings) {
        auto length = str.getLength() + 1;
        auto chars = str.getChars();
        std::copy_n(chars, length, mergedRawStrings.begin() + curPos);
        curPos += length;
        if constexpr (StringContainer::isIndexed)
            mergedStringsIndices.push_back(str.index);
    }
    if constexpr (StringContainer::isIndexed) {
        stringContainer.update(std::move(mergedRawStrings), mergedStringsIndices);
        mergedStringsIndices.clear();
        mergedStringsIndices.shrink_to_fit();
    } else {
        stringContainer.update(std::move(mergedRawStrings));
    }

    if constexpr (debugQuicksort) {
        if (!stringContainer.isConsistent()) {
            std::cout << "merged string cont not consistent " << std::endl;
            std::abort();
        }
    }

    mergedStrings.clear();
    mergedStrings.shrink_to_fit();
    recvStrings.deleteAll();

    if (nprocs >= 4) {
        // Split communicator and solve subproblems.
        tracker.comm_split_t.start(comm);

        MPI_Comm subcomm;
        split(comm, &subcomm);

        tracker.comm_split_t.stop();

        auto res = sortRec<StringContainer::isIndexed>(
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
        return res;
    }

    return std::move(stringContainer);
}

template <class T, class Ignore, class Communicator>
void shuffle(
    std::mt19937_64& async_gen,
    std::vector<T>& v,
    std::vector<Ignore>& v_tmp,
    MPI_Datatype mpi_type,
    int tag,
    Communicator& comm
) {
    // Just used for OpenMP
    std::ignore = v_tmp;

    int32_t nprocs;
    int32_t myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    // Generate a random bit generator for each thread. Using pointers
    // is faster than storing the generators in one vector due to
    // cache-line sharing.
#ifdef _OPENMP
    int num_threads = 1;
    std::vector<std::unique_ptr<std::mt19937_64>> omp_async_gens;
    std::vector<size_t> seeds;
#pragma omp parallel
    {
#pragma omp single
        {
            num_threads = omp_get_num_threads();
            omp_async_gens.resize(num_threads);
            for (int i = 0; i != num_threads; ++i) {
                seeds.push_back(async_gen());
            }
        }

        int const id = omp_get_thread_num();

        omp_async_gens[id].reset(new std::mt19937_64{seeds[id]});
    }

    std::vector<size_t> sizes(2 * num_threads);
    std::vector<size_t> prefix_sum(2 * num_threads + 1);

#endif

    const size_t comm_phases = tlx::integer_log2_floor(nprocs);

    for (size_t phase = 0; phase != comm_phases; ++phase) {
        const size_t mask = 1 << phase;
        const size_t partner = myrank ^ mask;

#ifdef _OPENMP

        int send_cnt = 0;
        int own_cnt = 0;
        int recv_cnt = 0;

#pragma omp parallel
        {
            int const id = omp_get_thread_num();

            const size_t stripe = tlx::div_ceil(v.size(), num_threads);
            std::unique_ptr<T[]> partitions[2] = {
                std::unique_ptr<T[]>(new T[stripe]),
                std::unique_ptr<T[]>(new T[stripe])};

            auto const begin = v.data() + id * stripe;
            auto const end = v.data() + std::min((id + 1) * stripe, v.size());

            auto& mygen = *omp_async_gens[id].get();
            T* partptrs[2] = {partitions[0].get(), partitions[1].get()};

            auto const size = end - begin;
            auto const remaining = size % (8 * sizeof(std::mt19937_64::result_type));
            auto ptr = begin;
            while (ptr < end - remaining) {
                auto rand = mygen();
                for (size_t j = 0; j < 8 * sizeof(std::mt19937_64::result_type); ++j) {
                    auto const bit = rand & 1;
                    rand = rand >> 1;
                    *partptrs[bit] = *ptr;
                    ++partptrs[bit];
                    ++ptr;
                }
            }

            auto rand = mygen();
            while (ptr < end) {
                // for (size_t i = 0; i != remaining; ++i) {
                auto const bit = rand & 1;
                rand = rand >> 1;
                *partptrs[bit] = *ptr;
                ++partptrs[bit];
                ++ptr;
            }

            sizes[id] = partptrs[0] - partitions[0].get();
            sizes[num_threads + id] = partptrs[1] - partitions[1].get();

#pragma omp barrier
#pragma omp master
            {
                tlx::exclusive_scan(sizes.begin(), sizes.end(), prefix_sum.begin(), size_t{0});

                send_cnt = prefix_sum[2 * num_threads] - prefix_sum[num_threads];
                own_cnt = prefix_sum[num_threads];
                recv_cnt = 0;
                RBC::Sendrecv(
                    &send_cnt,
                    1,
                    MPI_INT,
                    partner,
                    tag,
                    &recv_cnt,
                    1,
                    MPI_INT,
                    partner,
                    tag,
                    comm,
                    MPI_STATUS_IGNORE
                );

                v.resize(own_cnt + recv_cnt);
                v_tmp.resize(send_cnt);
            }
#pragma omp barrier

            std::copy(
                partitions[0].get(),
                partitions[0].get() + sizes[id],
                v.data() + prefix_sum[id]
            );
            std::copy(
                partitions[1].get(),
                partitions[1].get() + sizes[num_threads + id],
                v_tmp.data() + prefix_sum[id + num_threads] - prefix_sum[num_threads]
            );
        }

        RBC::Sendrecv(
            v_tmp.data(),
            send_cnt,
            mpi_type,
            partner,
            tag,
            v.data() + own_cnt,
            recv_cnt,
            mpi_type,
            partner,
            tag,
            comm,
            MPI_STATUS_IGNORE
        );

#else
        // v contains stringIndices
        std::unique_ptr<T[]> partition(new T[v.size()]);
        T* begins[2] = {v.data(), partition.get()};

        auto const size = v.size();
        auto const remaining = size % (8 * sizeof(std::mt19937_64::result_type));
        auto ptr = v.begin();
        auto const end = v.end();
        while (ptr < end - remaining) {
            auto rand = async_gen();
            for (size_t j = 0; j < 8 * sizeof(std::mt19937_64::result_type); ++j) {
                auto const bit = rand & 1;
                rand = rand >> 1;
                *begins[bit] = *ptr;
                ++begins[bit];
                ++ptr;
            }
        }

        auto rand = async_gen();
        while (ptr < end) {
            auto const bit = rand & 1;
            rand = rand >> 1;
            *begins[bit] = *ptr;
            ++begins[bit];
            ++ptr;
        }

        MPI_Request requests[2];
        // send sizes
        MPI_Isend(
            partition.get(),
            (begins[1] - partition.get()) * sizeof(T),
            mpi_type,
            partner,
            tag,
            comm,
            requests
        );

        int count = 0;
        MPI_Status status;
        MPI_Probe(partner, tag, comm, &status);
        MPI_Get_count(&status, mpi_type, &count);
        count /= sizeof(T);

        v.resize(begins[0] - v.data() + count); // begins[0] end of string indices

        MPI_Irecv(
            v.data() + v.size() - count,
            count * sizeof(T),
            mpi_type,
            partner,
            tag,
            comm,
            requests + 1
        );
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

#endif
    }
}

template <typename StringContainer>
void sortLocally(StringContainer& stringContainer) {
    if constexpr (StringContainer::isIndexed) {
        std::vector<uint64_t> lcp(stringContainer.size(), 0);
        auto strptr = stringContainer.make_string_ptr();
        auto augmentedStringPtr =
            tlx::sort_strings_detail::StringLcpPtr(strptr.active(), lcp.data());
        tlx::sort_strings_detail::radixsort_CI3(augmentedStringPtr, 0, 0);
        auto ranges = getDuplicateRanges(augmentedStringPtr);
        sortRanges(stringContainer, ranges);
    } else {
        tlx::sort_strings_detail::radixsort_CI3(stringContainer.make_string_ptr(), 0, 0);
    }
}

template <class Iterator, class Comp>
void sortLocally(Iterator begin, Iterator end, Comp&& comp) {
#ifdef _OPENMP
    ips4o::parallel::sort(begin, end, std::forward<Comp>(comp));
#else
    ips4o::sort(begin, end, std::forward<Comp>(comp));
#endif
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
    using dss_schimek::measurement::MeasuringTool;

    // MeasuringTool& measuringTool = MeasuringTool::measuringTool();
    // measuringTool.disableBarrier(true);
    // measuringTool.start("Splitter_baseCase");
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

    // measuringTool.start("Splitter_move_to_pow_of_two_t");
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
        // measuringTool.stop("Splitter_move_to_pow_of_two_t");

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
    // measuringTool.stop("Splitter_move_to_pow_of_two_t");
    tracker.move_to_pow_of_two_t.stop();

    // assert(tlx::is_power_of_two(nprocs));

    tracker.parallel_shuffle_t.start(comm);
    // measuringTool.start("Splitter_shuffle");
    // measuringTool.stop("Splitter_shuffle");
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
    Tools::DummyTimer local_sort_t;
    Tools::DummyTimer exchange_t;
    Tools::DummyTimer parallel_shuffle_t;
    Tools::DummyTimer merge_t;
    Tools::DummyTimer median_select_t;
    Tools::DummyTimer partition_t;
    Tools::DummyTimer comm_split_t;
    Tools::DummyTimer move_to_pow_of_two_t;
};
} // namespace _internal

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
