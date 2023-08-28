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
#include <cassert>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include <tlx/math.hpp>

#include "./RandomBitStore.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"

int globalIteration;
int globalRank;
namespace BinTreeMedianSelection {
template <class Data, class Comp, class Communicator>
Data select(
    Data&& data,
    size_t n,
    Comp&& comp,
    MPI_Datatype mpi_type,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    int tag,
    Communicator& comm
);

namespace _internal {
template <class Data, class Comp>
Data selectMedians(
    Data&& data,
    Data&& recvData,
    size_t n,
    Comp&& comp,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen
) {
    using StringContainer = typename Data::StringContainer;
    using StringSet = typename StringContainer::StringSet;
    using String = typename StringSet::String;
    // std::copy(recv_v.begin(), recv_v.end(), std::back_inserter(vs));
    // recv_v.clear();
    // return;
    // assert(recv_v.size() <= n);
    // assert(vs.size() <= n);

    StringContainer dataContainer = data.moveToContainer();
    StringContainer recvContainer = data.moveToContainer();
    // recv_v_container.make_string_set().print();
    // vs_container.make_string_set().print();
    std::vector<String> mergedStrings(dataContainer.size() + recvContainer.size());
    Data mergedData;

    std::merge(
        dataContainer.getStrings().begin(),
        dataContainer.getStrings().end(),
        recvContainer.getStrings().begin(),
        recvContainer.getStrings().end(),
        mergedStrings.begin(),
        std::forward<Comp>(comp)
    );

    {
        mergedData.rawStrings.resize(dataContainer.char_size() + recvContainer.char_size());
        uint64_t curPos = 0;
        for (auto const& str: mergedStrings) {
            std::copy(
                str.string,
                str.string + str.length + 1,
                mergedData.rawStrings.data() + curPos
            );
            curPos += str.length + 1;
            if constexpr (Data::isIndexed_) {
                mergedData.indices.push_back(str.index);
            }
        }
    }

    if (dataContainer.size() + recvContainer.size() <= n) {
        return mergedData;
    } else {
        StringContainer mergedContainer = mergedData.moveToContainer();
        Data returnData;
        auto ss = mergedContainer.make_string_set();
        if ((mergedContainer.size() - n) % 2 == 0) {
            auto const offset = (mergedContainer.size() - n) / 2;
            assert(offset + n < mergedContainer.size());

            for (size_t i = offset; i < offset + n; ++i) {
                auto const str = ss[ss.begin() + i];
                auto const length = ss.get_length(str) + 1;
                auto chars = ss.get_chars(str, 0);
                std::copy_n(chars, length, std::back_inserter(returnData.rawStrings));
                if constexpr (StringContainer::isIndexed) {
                    returnData.indices.push_back(str.index);
                }
            }
            return returnData;
        } else {
            // We cannot remove the same number of elements at
            // the right and left end.
            auto const offset = (mergedContainer.size() - n) / 2;
            auto const padding_cnt = bit_gen.getNextBit(async_gen);
            assert(padding_cnt <= 1);
            assert(offset + padding_cnt + n <= mergedContainer.size());

            for (size_t i = offset + padding_cnt; i < offset + padding_cnt + n; ++i) {
                auto const str = ss[ss.begin() + i];
                auto const length = ss.get_length(str) + 1;
                auto chars = ss.get_chars(str, 0);
                std::copy_n(chars, length, std::back_inserter(returnData.rawStrings));
                if constexpr (StringContainer::isIndexed) {
                    returnData.indices.push_back(str.index);
                }
            }
            return returnData;
        }
    }
}

int64_t selectMedian(const uint64_t size, std::mt19937_64& async_gen, RandomBitStore& bit_gen) {
    if (size == 0) {
        return -1;
    }

    assert(size > 0);
    if (size % 2 == 0) {
        if (bit_gen.getNextBit(async_gen)) {
            return size / 2;
        } else {
            return (size / 2) - 1;
        }
    } else {
        return size / 2;
    }
}
} // namespace _internal

template <class Data, class Comp, class Communicator>
Data select(
    Data&& data,
    size_t n,
    Comp&& comp,
    MPI_Datatype mpi_type,
    std::mt19937_64& async_gen,
    RandomBitStore& bit_gen,
    int tag,
    Communicator& comm
) {
    using StringContainer = typename Data::StringContainer;
    using namespace dss_schimek::measurement;
    auto& measuringTool = MeasuringTool::measuringTool();

    int32_t myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    // assert(static_cast<size_t>(end - begin) <= n);
    // assert(std::is_sorted(begin, end, std::forward<Comp>(comp)));

    auto const tailing_zeros = static_cast<unsigned>(tlx::ffs(myrank)) - 1;
    auto const logp = tlx::integer_log2_floor(nprocs);
    auto const iterations = std::min(tailing_zeros, logp);

    for (size_t it = 0; it != iterations; ++it) {
        auto const source = myrank + (1 << it);

        Data recvData = data.Recv(comm, source, tag);
        globalIteration = it;
        globalRank = myrank;
        data = _internal::selectMedians(
            std::move(data),
            std::move(recvData),
            n,
            std::forward<Comp>(comp),
            async_gen,
            bit_gen
        );
    }
    if (myrank == 0) {
        StringContainer container = data.moveToContainer();
        auto medianIndex = _internal::selectMedian(container.size(), async_gen, bit_gen);

        auto requestedRawString = container.getRawString(medianIndex);
        int32_t size = requestedRawString.size();
        measuringTool.addRawCommunication(sizeof(int), "");
        MPI_Bcast(&size, 1, MPI_INT, 0, comm);
        measuringTool.addRawCommunication(size, "");
        MPI_Bcast(requestedRawString.data(), size, MPI_BYTE, 0, comm);
        Data returnData{.rawStrings = requestedRawString, .indices = {}};

        if constexpr (StringContainer::isIndexed) {
            auto stringIndex = medianIndex < 0 ? 0 : container.strings()[medianIndex].index;

            measuringTool.addRawCommunication(sizeof(stringIndex), "");
            MPI_Bcast(&stringIndex, sizeof(stringIndex), MPI_BYTE, 0, comm);
            returnData.indices.push_back(stringIndex);
        }
        return returnData;
    } else {
        int target = myrank - (1 << tailing_zeros);
        // assert(v.size() <= n);
        data.Send(comm, target, tag);

        int32_t medianSize;
        measuringTool.addRawCommunication(sizeof(int), "");
        MPI_Bcast(&medianSize, 1, MPI_INT, 0, comm);
        Data returnData;
        returnData.rawStrings.resize(medianSize);
        measuringTool.addRawCommunication(medianSize, "");
        MPI_Bcast(returnData.rawStrings.data(), medianSize, MPI_BYTE, 0, comm);

        if constexpr (StringContainer::isIndexed) {
            returnData.indices.resize(1);
            measuringTool.addRawCommunication(sizeof(uint64_t), "");
            MPI_Bcast(returnData.indices.data(), sizeof(uint64_t), MPI_BYTE, 0, comm);
        }
        return returnData;
    }
}
} // namespace BinTreeMedianSelection
