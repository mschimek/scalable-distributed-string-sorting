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

#include "RandomBitStore.hpp"
#include "data.hpp"

namespace BinTreeMedianSelection {

// todo return type here is TBD
template <class StringSet, class Comp>
typename std::vector<unsigned char> select(
    StringSet const& ss,
    size_t n,
    Comp comp,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    int tag,
    RBC::Comm& comm
);

namespace _internal {

template <class StringSet, class Comp>
StringSet selectMedians(
    StringSet& local_ss,
    StringSet& recv_ss,
    std::vector<typename StringSet::String>& tmp_strs,
    size_t n,
    Comp comp,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen
) {
    assert(recv_v.size() <= n);
    assert(vs.size() <= n);

    tmp_strs.resize(recv_ss.size() + local_ss.size());
    // todo use string merge here
    std::merge(
        recv_ss.begin(),
        recv_ss.end(),
        local_ss.begin(),
        local_ss.end(),
        tmp_strs.begin(),
        std::forward<Comp>(comp)
    );

    if (tmp_strs.size() <= n) {
        return {tmp_strs.begin(), tmp_strs.end()};
    } else {
        if ((tmp_strs.size() - n) % 2 == 0) {
            auto const offset = (tmp_strs.size() - n) / 2;

            assert(offset + n < tmp_v.size());
            auto const begin = tmp_strs.begin() + offset;
            return {begin, begin + n};
        } else {
            auto const offset = (tmp_strs.size() - n) / 2;
            auto const shift = bit_gen.getNextBit(async_gen);

            assert(offset + padding_cnt + n <= tmp_v.size());
            auto const begin = tmp_strs.begin() + offset + shift;
            return {begin, begin + n};
        }
    }
}

// todo maybe return an iterator instead
template <class StringSet>
typename StringSet::String
selectMedian(StringSet const& ss, std::mt19937_64& async_gen, RandomBitStore& bit_gen) {
    if (!ss.empty()) {
        if (ss.size() % 2 == 0) {
            auto shift = bit_gen.getNextBit(async_gen);
            assert(shift <= 1);

            return ss[ss.begin() + ((ss.size() / 2) - shift)];
        } else {
            return ss[ss.begin() + (ss.size() / 2)];
        }
    } else {
        return StringSet::emptyString();
    }
}

} // namespace _internal

template <class StringSet, class Comp>
std::pair<RQuick::Data<StringSet>, typename StringSet::String> select(
    StringSet const& ss,
    size_t const n,
    Comp const comp,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    int const tag,
    const RBC::Comm& comm
) {
    auto const myrank = comm.getRank();

    assert(ss.size() <= n);
    assert(ss.check_order());

    // todo what about reserve here?
    RQuick::Data<StringSet> local_data;
    RQuick::Data<StringSet> recv_data;

    auto const make_string_set = [](auto& vec) -> StringSet {
        return {&*vec.begin(), &*vec.end()};
    };

    std::vector<typename StringSet::String> local_strs{ss.begin(), ss.end()};
    std::vector<typename StringSet::String> recv_strs;
    std::vector<typename StringSet::String> temp_strs;

    recv_strs.reserve(n);
    temp_strs.reserve(2 * n);

    int const tailing_zeros = tlx::ffs(comm.getRank()) - 1;
    int const iterations =
        comm.getRank() > 0 ? tailing_zeros : tlx::integer_log2_ceil(comm.getSize());

    for (int it = 0; it != iterations; ++it) {
        auto const source = myrank + (1 << it);

        recv_data.recv(source, tag, comm);
        recv_data.read_into(recv_strs);

        auto const medians = _internal::selectMedians(
            make_string_set(local_strs),
            make_string_set(recv_strs),
            temp_strs,
            n,
            std::forward<Comp>(comp),
            async_gen,
            bit_gen
        );
        local_data.write(medians);
        local_strs.resize(medians.size());
        std::copy(medians.begin(), medians.end(), local_strs.begin());
    }

    if (myrank == 0) {
        auto local_ss = make_string_set(local_strs);
        auto const median = _internal::selectMedian(local_ss, async_gen, bit_gen);
        recv_data.write({&median, &median + 1});
        auto str = recv_data.bcast_single(1, comm);
        return {std::move(recv_data), std::move(str)};
    } else {
        int const target = myrank - (1 << tailing_zeros);
        local_data.send(target, tag, comm);

        auto str = recv_data.bcast_single(0, comm);
        return {std::move(recv_data), std::move(str)};
    }
}

} // namespace BinTreeMedianSelection
