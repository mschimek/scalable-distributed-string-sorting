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
#include <random>
#include <utility>
#include <vector>

#include <RBC.hpp>
#include <tlx/math.hpp>

#include "./RandomBitStore.hpp"
#include "./Util.hpp"

namespace RQuick2 {
namespace BinTreeMedianSelection {
namespace _internal {

template <class StringPtr>
StringPtr selectMedians(
    StringPtr const& local_strptr,
    StringPtr const& recv_strptr,
    Container<StringPtr>& merge_strings,
    size_t const n,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen
) {
    assert(local_strptr.size() <= n);
    assert(recv_strptr.size() <= n);

    auto const& local_ss = local_strptr.active();
    auto const& recv_ss = recv_strptr.active();

    merge_strings.resize_strings(local_strptr.size() + recv_strptr.size());
    // todo consider using lcp merge here (it isn't really worth it though)
    std::merge(
        local_ss.begin(),
        local_ss.end(),
        recv_ss.begin(),
        recv_ss.end(),
        merge_strings.getStrings().begin(),
        Comparator<StringPtr>{}
    );

    auto const strptr = merge_strings.make_auto_ptr();
    if (merge_strings.size() <= n) {
        return strptr;
    } else {
        if ((merge_strings.size() - n) % 2 == 0) {
            auto const offset = (merge_strings.size() - n) / 2;

            assert(offset + n < strptr.size());
            return strptr.sub(offset, n);
        } else {
            auto const offset = (merge_strings.size() - n) / 2;
            auto const shift = bit_gen.getNextBit(async_gen);

            assert(offset + shift + n <= strptr.size());
            return strptr.sub(offset + shift, n);
        }
    }
}

template <class StringSet>
typename StringSet::String
selectMedian(StringSet const& ss, std::mt19937_64& async_gen, RandomBitStore& bit_gen) {
    if (!ss.empty()) {
        if (ss.size() % 2 == 0) {
            auto shift = bit_gen.getNextBit(async_gen);
            return ss[ss.begin() + ((ss.size() / 2) - shift)];
        } else {
            return ss[ss.begin() + (ss.size() / 2)];
        }
    } else {
        return StringSet::empty_string();
    }
}

} // namespace _internal

template <class StringPtr>
StringT<StringPtr> select(
    StringPtr strptr,
    TemporaryBuffers<StringPtr>& buffers,
    size_t const n,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    int const tag,
    const RBC::Comm& comm
) {
    auto const myrank = comm.getRank();

    assert(strptr.size() <= n);
    assert(strptr.active().check_order());

    int const tailing_zeros = tlx::ffs(comm.getRank()) - 1;
    int const log2_ceil = tlx::integer_log2_ceil(comm.getSize());
    int const iterations = comm.getRank() > 0 ? tailing_zeros : log2_ceil;

    for (int it = 0; it != iterations; ++it) {
        auto const source = myrank + (1 << it);

        buffers.recv_data.recv(source, tag, comm);
        buffers.recv_strings.resize_strings(buffers.recv_data.get_num_strings());
        buffers.recv_data.read_into(buffers.recv_strings.make_auto_ptr());

        auto const medians = _internal::selectMedians(
            strptr,
            buffers.recv_strings.make_auto_ptr(),
            buffers.merge_strings,
            n,
            async_gen,
            bit_gen
        );

        auto const& ss = medians.active();
        buffers.median_strings.resize_strings(ss.size());
        std::copy(ss.begin(), ss.end(), buffers.median_strings.getStrings().begin());
        // todo lcp values aren't really worth it here
        if constexpr (StringPtr::with_lcp) {
            std::copy_n(strptr.lcp(), strptr.size(), buffers.median_strings.lcp_array());
        }

        buffers.median_strings.make_contiguous(buffers.char_buffer);
        strptr = buffers.median_strings.make_auto_ptr();
    }

    if (myrank == 0) {
        auto median = _internal::selectMedian(strptr.active(), async_gen, bit_gen);

        if constexpr (StringPtr::with_lcp) {
            size_t lcp = 0;
            buffers.recv_data.write(StringPtr{{&median, &median + 1}, &lcp});
        } else {
            buffers.recv_data.write(StringPtr{{&median, &median + 1}});
        }
        buffers.recv_data.bcast_single(0, comm);
    } else {
        int const target = myrank - (1 << tailing_zeros);
        buffers.send_data.write(strptr);
        buffers.send_data.send(target, tag, comm);

        buffers.recv_data.bcast_single(0, comm);
    }

    buffers.median_strings.resize_strings(1);
    auto median_ptr = buffers.median_strings.make_auto_ptr();
    buffers.recv_data.read_into(median_ptr);
    std::swap(buffers.median_strings.raw_strings(), buffers.recv_data.raw_strs);

    return median_ptr.active()[median_ptr.active().begin()];
}

} // namespace BinTreeMedianSelection
} // namespace RQuick2
