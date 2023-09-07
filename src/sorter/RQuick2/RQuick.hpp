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
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <RBC.hpp>
#include <ips4o.hpp>
#include <kamping/mpi_datatype.hpp>

#include "./BinTreeMedianSelection.hpp"
#include "./RandomBitStore.hpp"

namespace Tools {

class DummyTimer {
public:
    DummyTimer() {}

    void start(MPI_Comm&) {}
    void start(const RBC::Comm&) {}
    void stop() {}
};

} // namespace Tools

namespace RQuick {
namespace _internal {
template <class T>
using PairVectorItSizeT = std::pair<typename std::vector<T>::iterator, size_t>;

inline void split(const RBC::Comm& comm, RBC::Comm* subcomm) {
    auto const nprocs = comm.getSize();
    auto const myrank = comm.getRank();

    bool is_left_group = myrank < (nprocs / 2);

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
 * Returns the k middle most elements of v if v contains at least k
 * elements. Otherwise, v is returned. This function uses
 * randomization to decide which elements are the 'middle most'
 * elements if v contains an odd number of elements and k is an even
 * number or if v contains an even number of elements and k is an odd
 * number.
 */
template <class T>
std::vector<T> middleMostElements(
    std::vector<T> const& v, size_t k, std::mt19937_64& async_gen, RandomBitStore& bit_gen
) {
    if (v.size() <= k) {
        return std::vector<T>(v.begin(), v.end());
    }

    auto const offset = (v.size() - k) / 2;

    if (((v.size() % 2) == 0 && ((k % 2) == 0)) || (v.size() % 2 == 1 && ((k % 2) == 1))) {
        return std::vector<T>(v.data() + offset, v.data() + offset + k);
    } else {
        // Round down or round up randomly.

        auto const shift = bit_gen.getNextBit(async_gen);
        return std::vector<T>(v.data() + offset + shift, v.data() + offset + k + shift);
    }
}

/*
 * @brief Distributed splitter selection with a binary reduction tree.
 *
 * @param v Local input. The input must be sorted.
 */
template <class T, class Comp>
T selectSplitter(
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    std::vector<T> const& v,
    MPI_Datatype mpi_type,
    Comp comp,
    int tag,
    const RBC::Comm& comm
) {
    assert(std::is_sorted(v.begin(), v.end(), comp));

    auto const local_medians = middleMostElements(v, 2, async_gen, bit_gen);

    auto const s = BinTreeMedianSelection::select(
        local_medians.data(),
        local_medians.data() + local_medians.size(),
        2,
        std::forward<Comp>(comp),
        mpi_type,
        async_gen,
        bit_gen,
        tag,
        comm
    );

    return s;
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
template <class T, class Comp>
PairVectorItSizeT<T> const locateSplitterLeft(
    std::vector<T>& v,
    Comp comp,
    T const& splitter,
    bool is_robust,
    int partner,
    int tag,
    const RBC::Comm& comm
) {
    using It = typename std::vector<T>::iterator;

    if (!is_robust) {
        It begin_equal_els =
            std::lower_bound(v.begin(), v.end(), splitter, std::forward<Comp>(comp));

        int64_t send_cnt = v.end() - begin_equal_els;
        int64_t recv_cnt = -1;
        RBC::Sendrecv(
            &send_cnt,
            1,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            &recv_cnt,
            1,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            comm,
            MPI_STATUS_IGNORE
        );

        return {begin_equal_els, recv_cnt};
    }

    auto const begin_equal_els =
        std::lower_bound(v.begin(), v.end(), splitter, std::forward<Comp>(comp));
    auto const end_equal_els =
        std::upper_bound(begin_equal_els, v.end(), splitter, std::forward<Comp>(comp));

    // m: my; t: theirs
    // Number of elemements < or == or > than the splitter
    const int64_t mleft = begin_equal_els - v.begin();
    const int64_t mright = v.end() - end_equal_els;
    const int64_t mmiddle = v.size() - mleft - mright;

    auto [tleft, tright, tmiddle] = [&]() {
        int64_t send[3] = {mleft, mright, mmiddle};
        int64_t recv[3];
        RBC::Sendrecv(
            send,
            3,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            recv,
            3,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            comm,
            MPI_STATUS_IGNORE
        );
        return std::tuple(recv[0], recv[1], recv[2]);
    }();

    const int64_t perf_split = (v.size() + tleft + tright + tmiddle) / 2;
    const int64_t mkeep =
        std::min<int64_t>(mmiddle, std::max<int64_t>(perf_split - mleft - tleft, 0));
    const int64_t tkeep =
        std::min<int64_t>(tmiddle, std::max<int64_t>(perf_split - tright - mright, 0));

    return {v.begin() + mleft + mkeep, tleft + tmiddle - tkeep};
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
template <class T, class Comp>
PairVectorItSizeT<T> const locateSplitterRight(
    std::vector<T>& v,
    Comp comp,
    T const& splitter,
    bool is_robust,
    int partner,
    int tag,
    const RBC::Comm& comm
) {
    using It = typename std::vector<T>::iterator;

    if (!is_robust) {
        It begin_equal_els =
            std::lower_bound(v.begin(), v.end(), splitter, std::forward<Comp>(comp));

        int64_t send_cnt = begin_equal_els - v.begin();
        int64_t recv_cnt = -1;
        RBC::Sendrecv(
            &send_cnt,
            1,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            &recv_cnt,
            1,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            comm,
            MPI_STATUS_IGNORE
        );

        return {begin_equal_els, recv_cnt};
    }

    auto const begin_equal_els =
        std::lower_bound(v.begin(), v.end(), splitter, std::forward<Comp>(comp));
    auto const end_equal_els =
        std::upper_bound(begin_equal_els, v.end(), splitter, std::forward<Comp>(comp));

    // m: my; t: theirs
    const int64_t mleft = begin_equal_els - v.begin();
    const int64_t mright = v.end() - end_equal_els;
    const int64_t mmiddle = v.size() - mleft - mright;

    auto [tleft, tright, tmiddle] = [&]() {
        int64_t send[3] = {mleft, mright, mmiddle};
        int64_t recv[3];
        RBC::Sendrecv(
            send,
            3,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            recv,
            3,
            kamping::mpi_datatype<int64_t>(),
            partner,
            tag,
            comm,
            MPI_STATUS_IGNORE
        );
        return std::tuple(recv[0], recv[1], recv[2]);
    }();

    const int64_t perf_split = (v.size() + tleft + tright + tmiddle) / 2;
    const int64_t mkeep =
        std::min<int64_t>(mmiddle, std::max<int64_t>(perf_split - mright - tright, 0));
    const int64_t tkeep =
        std::min<int64_t>(tmiddle, std::max<int64_t>(perf_split - tleft - mleft, 0));

    return {v.end() - mright - mkeep, tright + tmiddle - tkeep};
}

/*
 * @brief Partitions two sequences according to a specific rank.
 *
 * Split two sequences into a left and a right part each. The left
 * parts contain 'rank' elements in total and the left parts only
 * contain elements which are smaller than elements of the right
 * parts. Tie-breaking is used meaning that elements of the first
 * sequence are smaller than the same elements of the second sequence
 * and equal elements of the same sequence use their index as
 * tie-breaker.
 *
 * @param begin1 Begin of the first sorted sequence.
 * @param end1 End of the first sorted sequence.
 * @param begin2 Begin of the second sorted sequence.
 * @param end2 End of the second sorted sequence.
 * @param rank Total size of the left part. Rank must be a positive
 *  number smaller of equal to the total size of both input sequences.
 * @param comp The comparator.
 */
template <class T, class Comp>
std::pair<T const*, T const*> twoSequenceSelection(
    T const* begin1, T const* end1, T const* begin2, T const* end2, int64_t rank, Comp comp
) {
    assert(std::is_sorted(begin1, end1, std::forward<Comp>(comp)));
    assert(std::is_sorted(begin2, end2, std::forward<Comp>(comp)));

    assert(static_cast<size_t>((end1 - begin1) + (end2 - begin2)) >= rank);

    // Shrink sequences by taking splitters from the first
    // sequence until first sequence does not contain any
    // elements.
    while (end1 != begin1) {
        assert(end1 > begin1);

        auto const offset1 = (end1 - begin1) / 2;
        auto const splitter1 = begin1 + offset1;
        assert(begin1 <= splitter1);
        assert(splitter1 < end1);

        // We use tie-breaking. Meaning than an element 'a' in
        // the first sequence is smaller than the same element
        // 'a' in the second sequence. Thus, the function
        // 'lower_bound' breaks ties implicitly.
        auto const splitter2 = std::lower_bound(begin2, end2, *splitter1, std::forward<Comp>(comp));
        auto const offset2 = splitter2 - begin2;

        auto const new_rank = offset1 + offset2;

        if (rank < new_rank) {
            // Continue on left part.

            // Size of first sequence decreases by at least
            // one as 'splitter1' is now excluded.

            end1 = splitter1;
            end2 = splitter2;
        } else if (rank > new_rank) {
            // Continue on right part.

            // Size of the first sequence decreases by at least one
            // (even if the frist element is selected as 'splitter1')
            // as 'splitter1' is now excluded. We can exclude
            // 'splitter1' from the first sequence, as all elements in
            // [splitter2, end2) as they are larger than 'splitter1'
            // (ties are already broken). However, as 'rank >
            // new_rank', we can remove the smallest element.

            rank -= new_rank + 1;
            begin1 = splitter1 + 1;
            begin2 = splitter2;
        } else {
            return {splitter1, splitter2};
        }
    }

    // Position in first sequence has been found. We still
    // need 'rank' elements. Those 'rank' elements must be the
    // first elements of the second sequence.
    return {begin1, begin2 + rank};
}

template <class T, class Comp>
void merge(T const* begin1, T const* end1, T const* begin2, T const* end2, T* tbegin, Comp comp) {
    assert(std::is_sorted(begin1, end1, std::forward<Comp>(comp)));
    assert(std::is_sorted(begin2, end2, std::forward<Comp>(comp)));

    std::merge(begin1, end1, begin2, end2, tbegin, std::forward<Comp>(comp));
}

template <class T, class Tracker, class Comp>
void sortRec(
    std::mt19937_64& gen,
    RandomBitStore& bit_store,
    std::vector<T>& v,
    std::vector<T>& v_merge,
    std::vector<T>& v_recv,
    Comp comp,
    MPI_Datatype mpi_type,
    bool is_robust,
    Tracker&& tracker,
    int tag,
    const RBC::Comm& comm
) {
    using It = typename std::vector<T>::iterator;

    tracker.median_select_t.start(comm);

    auto const nprocs = comm.getSize();
    auto const myrank = comm.getRank();

    assert(nprocs >= 2);
    assert(std::is_sorted(v.begin(), v.end(), std::forward<Comp>(comp)));
    assert(tlx::integer_log2_floor(nprocs));

    auto const is_left_group = myrank < nprocs / 2;

    // Select pivot globally with binary tree median selection.
    auto const pivot =
        selectSplitter(gen, bit_store, v, mpi_type, std::forward<Comp>(comp), tag, comm);

    tracker.median_select_t.stop();

    // Partition data into small elements and large elements.

    tracker.partition_t.start(comm);
    auto const partner = (myrank + (nprocs / 2)) % nprocs;
    It separator, send_begin, send_end, own_begin, own_end;
    int64_t recv_cnt = -1;
    if (is_left_group) {
        std::tie(separator, recv_cnt) =
            locateSplitterLeft(v, std::forward<Comp>(comp), pivot, is_robust, partner, tag, comm);

        send_begin = v.begin();
        send_end = separator;
        own_begin = separator;
        own_end = v.end();
        std::swap(send_begin, own_begin);
        std::swap(send_end, own_end);
    } else {
        std::tie(separator, recv_cnt) =
            locateSplitterRight(v, std::forward<Comp>(comp), pivot, is_robust, partner, tag, comm);

        send_begin = v.begin();
        send_end = separator;
        own_begin = separator;
        own_end = v.end();
    }

    tracker.partition_t.stop();

    // Move elements to partner and receive elements for own group.
    tracker.exchange_t.start(comm);

    v_recv.resize(recv_cnt);
    auto const send_size = send_end - send_begin;
    RBC::Sendrecv(
        v.data() + (send_begin - v.begin()),
        send_size,
        mpi_type,
        partner,
        tag,
        v_recv.data(),
        v_recv.size(),
        mpi_type,
        partner,
        tag,
        comm,
        MPI_STATUS_IGNORE
    );

    tracker.exchange_t.stop();

    // Merge received elements with own elements.
    tracker.merge_t.start(comm);

    auto const num_elements = v_recv.size() + (own_end - own_begin);
    v_merge.resize(num_elements);
    merge(
        own_begin,
        own_end,
        v_recv.begin(),
        v_recv.end(),
        v_merge.begin(),
        std::forward<Comp>(comp)
    );
    v.swap(v_merge);
    assert(std::is_sorted(v.begin(), v.end(), std::forward<Comp>(comp)));

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
            v,
            v_merge,
            v_recv,
            std::forward<Comp>(comp),
            mpi_type,
            is_robust,
            std::forward<Tracker>(tracker),
            tag,
            subcomm
        );
    }
}

template <class T>
void shuffle(
    std::mt19937_64& async_gen,
    std::vector<T>& v,
    std::vector<T>& v_half,
    MPI_Datatype mpi_type,
    int tag,
    const RBC::Comm& comm
) {
    // Just used for OpenMP
    std::ignore = v_half;

    const size_t nprocs = comm.getSize();
    const size_t myrank = comm.getRank();

    // Generate a random bit generator for each thread. Using pointers
    // is faster than storing the generators in one vector due to
    // cache-line sharing.

    const size_t comm_phases = tlx::integer_log2_floor(nprocs);

    for (size_t phase = 0; phase != comm_phases; ++phase) {
        const size_t mask = 1 << phase;
        const size_t partner = myrank ^ mask;

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

        RBC::Request requests[2];

        RBC::Isend(
            partition.get(),
            (begins[1] - partition.get()),
            mpi_type,
            partner,
            tag,
            comm,
            requests
        );

        int count = 0;
        MPI_Status status;
        RBC::Probe(partner, tag, comm, &status);
        MPI_Get_count(&status, mpi_type, &count);

        v.resize(begins[0] - v.data() + count);

        RBC::Irecv(v.data() + v.size() - count, count, mpi_type, partner, tag, comm, requests + 1);
        RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);
    }
}

template <class It, class Comp>
void sortLocally(It begin, It end, Comp comp) {
    ips4o::sort(begin, end, std::forward<Comp>(comp));
}

template <class Tracker, class T, class Comp>
void sort(
    std::mt19937_64& async_gen,
    std::vector<T>& v,
    MPI_Datatype mpi_type,
    int tag,
    Tracker&& tracker,
    Comp comp,
    bool is_robust,
    RBC::Comm comm
) {
    if (comm.getSize() == 1) {
        tracker.local_sort_t.start(comm);
        sortLocally(v.begin(), v.end(), std::forward<Comp>(comp));
        tracker.local_sort_t.stop();

        return;
    }


    tracker.move_to_pow_of_two_t.start(comm);

    auto const pow = tlx::round_down_to_power_of_two(comm.getSize());

    // Send data to a smaller hypercube if the number of processes is
    // not a power of two.
    if (comm.getRank() < comm.getSize() - pow) {
        // Not a power of two but we are part of the smaller hypercube
        // and receive elements.

        MPI_Status status;
        auto const source = pow + comm.getRank();

        RBC::Probe(source, tag, comm, &status);
        int recv_cnt = 0;
        MPI_Get_count(&status, mpi_type, &recv_cnt);

        // Avoid reallocations later.
        v.reserve(2 * (v.size() + recv_cnt));

        v.resize(v.size() + recv_cnt);
        MPI_Request request;
        RBC::Irecv(v.data() + v.size() - recv_cnt, recv_cnt, mpi_type, source, tag, comm, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        RBC::Comm sub_comm;
        RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
        comm = sub_comm;
    } else if (comm.getRank() >= pow) {
        // Not a power of two and we are not part of the smaller
        // hypercube.

        auto const target = comm.getRank() - pow;
        RBC::Send(v.data(), v.size(), mpi_type, target, tag, comm);
        v.clear();

        // This process is not part of 'sub_comm'. We call
        // this function to support MPI implementations
        // without MPI_Comm_create_group.
        RBC::Comm sub_comm;
        RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
        comm = sub_comm;

        tracker.move_to_pow_of_two_t.stop();

        return;
    } else if (pow != comm.getSize()) {
        // Not a power of two but we are part of the smaller hypercube
        // and do not receive elements.

        RBC::Comm sub_comm;
        RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
        comm = sub_comm;

        // Avoid reallocations later.
        v.reserve(3 * v.size());
    } else {
        // The number of processes is a power of two.

        // Avoid reallocations later.
        v.reserve(2 * v.size());
    }

    tracker.move_to_pow_of_two_t.stop();

    assert(tlx::is_power_of_two(comm.getSize()));

    tracker.parallel_shuffle_t.start(comm);

    // Vector is used to store about the same number of elements as v.
    std::vector<T> v1;
    v1.reserve(v.capacity());

    // Vector is used to store about half the number of elements as
    // v. This vector is used to receive data.
    std::vector<T> v_half;
    v_half.reserve(v.capacity() / 2);

    if (is_robust) {
        shuffle(async_gen, v, v_half, mpi_type, tag, comm);
    }

    tracker.parallel_shuffle_t.stop();

    tracker.local_sort_t.start(comm);
    sortLocally(v.begin(), v.end(), std::forward<Comp>(comp));
    tracker.local_sort_t.stop();

    RandomBitStore bit_store;
    _internal::sortRec(
        async_gen,
        bit_store,
        v,
        v1,
        v_half,
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

template <class Tracker, class T, class Comp>
void sort(
    Tracker&& tracker,
    MPI_Datatype mpi_type,
    std::vector<T>& v,
    int tag,
    std::mt19937_64& async_gen,
    MPI_Comm mpi_comm,
    Comp comp,
    bool is_robust
) {
    RBC::Comm comm;
    RBC::Create_Comm_from_MPI(mpi_comm, &comm);
    _internal::sort(
        async_gen,
        v,
        mpi_type,
        tag,
        std::forward<Tracker>(tracker),
        std::forward<Comp>(comp),
        is_robust,
        comm
    );
}

template <class T, class Comp>
void sort(
    MPI_Datatype mpi_type,
    std::vector<T>& v,
    int tag,
    std::mt19937_64& async_gen,
    MPI_Comm mpi_comm,
    Comp comp,
    bool is_robust
) {
    _internal::DummyTracker tracker;
    sort(tracker, mpi_type, v, tag, async_gen, mpi_comm, comp, is_robust);
}

template <class Tracker, class T, class Comp>
void sort(
    Tracker&& tracker,
    MPI_Datatype mpi_type,
    std::vector<T>& v,
    int tag,
    std::mt19937_64& async_gen,
    const RBC::Comm& comm,
    Comp comp,
    bool is_robust
) {
    _internal::sort(
        async_gen,
        v,
        mpi_type,
        tag,
        std::forward<Tracker>(tracker),
        std::forward<Comp>(comp),
        is_robust,
        comm
    );
}

template <class T, class Comp>
void sort(
    MPI_Datatype mpi_type,
    std::vector<T>& v,
    int tag,
    std::mt19937_64& async_gen,
    const RBC::Comm& comm,
    Comp comp,
    bool is_robust
) {
    _internal::DummyTracker tracker;
    sort(tracker, mpi_type, v, tag, async_gen, comm, comp, is_robust);
}
} // namespace RQuick