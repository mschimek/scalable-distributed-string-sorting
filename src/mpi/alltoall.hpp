/*******************************************************************************
 * mpi/alltoall.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "mpi/allreduce.hpp"
#include "mpi/big_type.hpp"
#include "mpi/byte_encoder.hpp"
#include "mpi/environment.hpp"
#include "mpi/type_mapper.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringtools.hpp"
#include "util/measuringTool.hpp"
#include "util/structs.hpp"

namespace dss_schimek::mpi {

static constexpr bool debug_alltoall = false;

template <typename DataType>
inline std::vector<DataType>
alltoall(std::vector<DataType> const& send_data, environment env = environment()) {
    using dss_schimek::measurement::MeasuringTool;
    MeasuringTool& measuringTool = MeasuringTool::measuringTool();
    const size_t sentItems = send_data.size();
    measuringTool.addRawCommunication(sentItems * sizeof(DataType), "alltoall");

    std::vector<DataType> receive_data(send_data.size(), 0);
    data_type_mapper<DataType> dtm;
    MPI_Alltoall(
        send_data.data(),
        send_data.size() / env.size(),
        dtm.get_mpi_type(),
        receive_data.data(),
        send_data.size() / env.size(),
        dtm.get_mpi_type(),
        env.communicator()
    );
    return receive_data;
}

template <typename DataType>
inline std::vector<DataType> alltoallv_small(
    std::vector<DataType> const& send_data,
    std::vector<size_t> const& send_counts,
    environment env = environment()
) {
    using dss_schimek::measurement::MeasuringTool;
    MeasuringTool& measuringTool = MeasuringTool::measuringTool();

    std::vector<int32_t> real_send_counts(send_counts.size());
    for (size_t i = 0; i < send_counts.size(); ++i) {
        real_send_counts[i] = static_cast<int32_t>(send_counts[i]);
    }
    std::vector<int32_t> receive_counts = alltoall(real_send_counts, env);

    std::vector<int32_t> send_displacements(real_send_counts.size(), 0);
    std::vector<int32_t> receive_displacements(real_send_counts.size(), 0);
    for (size_t i = 1; i < real_send_counts.size(); ++i) {
        send_displacements[i] = send_displacements[i - 1] + real_send_counts[i - 1];
        receive_displacements[i] = receive_displacements[i - 1] + receive_counts[i - 1];
    }
    std::vector<DataType> receive_data(receive_counts.back() + receive_displacements.back());

    if constexpr (debug_alltoall) {
        for (int32_t i = 0; i < env.size(); ++i) {
            if (i == env.rank()) {
                std::cout << i << ": send_counts.size() " << send_counts.size() << std::endl;
                std::cout << i << ": send counts: ";
                for (auto const sc: real_send_counts) {
                    std::cout << sc << ", ";
                }
                std::cout << std::endl << "receive counts: ";

                for (auto const rc: receive_counts) {
                    std::cout << rc << ", ";
                }
                std::cout << std::endl;
            }
            env.barrier();
        }
    }
    const size_t sentItems =
        std::accumulate(real_send_counts.begin(), real_send_counts.end(), static_cast<size_t>(0u));
    measuringTool.addRawCommunication(sentItems * sizeof(DataType), "alltoallv_small");

    data_type_mapper<DataType> dtm;
    MPI_Alltoallv(
        send_data.data(),
        real_send_counts.data(),
        send_displacements.data(),
        dtm.get_mpi_type(),
        receive_data.data(),
        receive_counts.data(),
        receive_displacements.data(),
        dtm.get_mpi_type(),
        env.communicator()
    );
    return receive_data;
}

// uses builtin MPI_Alltoallv for message exchange
class AllToAllvSmall {
public:
    static std::string getName() { return "AlltoAllvSmall"; }
    template <typename DataType>
    static std::vector<DataType> alltoallv(
        DataType* const send_data,
        std::vector<size_t> const& send_counts,
        environment env = environment()
    ) {
        using dss_schimek::measurement::MeasuringTool;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        std::vector<int32_t> real_send_counts(send_counts.size());
        for (size_t i = 0; i < send_counts.size(); ++i) {
            real_send_counts[i] = static_cast<int32_t>(send_counts[i]);
        }
        std::vector<int32_t> receive_counts = alltoall(real_send_counts, env);

        std::vector<int32_t> send_displacements(real_send_counts.size(), 0);
        std::vector<int32_t> receive_displacements(real_send_counts.size(), 0);
        for (size_t i = 1; i < real_send_counts.size(); ++i) {
            send_displacements[i] = send_displacements[i - 1] + real_send_counts[i - 1];
            receive_displacements[i] = receive_displacements[i - 1] + receive_counts[i - 1];
        }
        std::vector<DataType> receive_data(receive_counts.back() + receive_displacements.back());

        if constexpr (debug_alltoall) {
            for (int32_t i = 0; i < env.size(); ++i) {
                if (i == env.rank()) {
                    std::cout << i << ": send_counts.size() " << send_counts.size() << std::endl;
                    std::cout << i << ": send counts: ";
                    for (auto const sc: real_send_counts) {
                        std::cout << sc << ", ";
                    }
                    std::cout << std::endl << "receive counts: ";

                    for (auto const rc: receive_counts) {
                        std::cout << rc << ", ";
                    }
                    std::cout << std::endl;
                }
                env.barrier();
            }
        }

        const size_t elemToSend =
            std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0u));
        measuringTool.addRawCommunication(elemToSend * sizeof(DataType), "alltoallv_small");

        data_type_mapper<DataType> dtm;
        MPI_Alltoallv(
            send_data,
            real_send_counts.data(),
            send_displacements.data(),
            dtm.get_mpi_type(),
            receive_data.data(),
            receive_counts.data(),
            receive_displacements.data(),
            dtm.get_mpi_type(),
            env.communicator()
        );
        return receive_data;
    }
};

// Uses direct P2P-messages for alltoallv-exchange
class AllToAllvDirectMessages {
public:
    static std::string getName() { return "AlltoAllvDirectMessages"; }
    template <typename DataType>
    static std::vector<DataType> alltoallv(
        DataType* const send_data,
        std::vector<size_t> const& send_counts,
        environment env = environment()
    ) {
        using dss_schimek::measurement::MeasuringTool;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        std::vector<size_t> receive_counts = alltoall(send_counts, env);
        std::vector<size_t> send_displacements(env.size(), 0);

        for (size_t i = 1; i < send_counts.size(); ++i) {
            send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
        }
        std::vector<size_t> receive_displacements(env.size(), 0);
        for (size_t i = 1; i < send_counts.size(); ++i) {
            receive_displacements[i] = receive_displacements[i - 1] + receive_counts[i - 1];
        }

        const size_t elemToSend =
            std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0u))
            - send_counts[env.rank()];
        measuringTool.addRawCommunication(
            elemToSend * sizeof(DataType),
            "alltoallv_directMessages"
        );

        std::vector<MPI_Request> mpi_request(2 * env.size());
        std::vector<DataType> receive_data(receive_displacements.back() + receive_counts.back());
        for (uint32_t i = 0; i < env.size(); ++i) {
            // start with self send/recv
            int32_t source = (env.rank() + (env.size() - i)) % env.size();
            auto receive_type = get_big_type<DataType>(receive_counts[source]);
            MPI_Irecv(
                receive_data.data() + receive_displacements[source],
                1,
                receive_type,
                source,
                44227,
                env.communicator(),
                &mpi_request[source]
            );
        }
        // dispatch sends
        for (uint32_t i = 0; i < env.size(); ++i) {
            int32_t target = (env.rank() + i) % env.size();
            auto send_type = get_big_type<DataType>(send_counts[target]);
            MPI_Isend(
                send_data + send_displacements[target],
                1,
                send_type,
                target,
                44227,
                env.communicator(),
                &mpi_request[env.size() + target]
            );
        }
        MPI_Waitall(2 * env.size(), mpi_request.data(), MPI_STATUSES_IGNORE);
        return receive_data;
    }
};

// Uses builtin MPI_Alltoallv- function for small messages and direct messages
// for larger messages
template <typename AllToAllvSmallPolicy>
class AllToAllvCombined {
public:
    static std::string getName() { return "AllToAllvCombined"; }
    template <typename DataType>
    static std::vector<DataType> alltoallv(
        DataType* const send_data,
        std::vector<size_t> const& send_counts,
        environment env = environment()
    ) {
        // std::cout << "MPI Routine : combined" << std::endl;
        using dss_schimek::measurement::MeasuringTool;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        size_t local_send_count =
            std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0u));

        std::vector<size_t> receive_counts = alltoall(send_counts, env);
        size_t local_receive_count =
            std::accumulate(receive_counts.begin(), receive_counts.end(), static_cast<size_t>(0u));

        size_t local_max = std::max(local_send_count, local_receive_count);
        size_t global_max = allreduce_max(local_max, env);

        if (global_max < env.mpi_max_int()) {
            return AllToAllvSmallPolicy::alltoallv(send_data, send_counts, env);
        } else {
            std::vector<size_t> send_displacements(env.size(), 0);
            for (size_t i = 1; i < send_counts.size(); ++i) {
                send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
            }
            std::vector<size_t> receive_displacements(env.size(), 0);
            for (size_t i = 1; i < send_counts.size(); ++i) {
                receive_displacements[i] = receive_displacements[i - 1] + receive_counts[i - 1];
            }

            const size_t elemToSend =
                std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0u));
            measuringTool.addRawCommunication(elemToSend * sizeof(DataType), "alltoallv_combined");

            std::vector<MPI_Request> mpi_request(2 * env.size());
            std::vector<DataType> receive_data(
                receive_displacements.back() + receive_counts.back()
            );
            for (uint32_t i = 0; i < env.size(); ++i) {
                // start with self send/recv
                int32_t source = (env.rank() + (env.size() - i)) % env.size();
                auto receive_type = get_big_type<DataType>(receive_counts[source]);
                MPI_Irecv(
                    receive_data.data() + receive_displacements[source],
                    1,
                    receive_type,
                    source,
                    44227,
                    env.communicator(),
                    &mpi_request[source]
                );
            }
            // dispatch sends
            for (uint32_t i = 0; i < env.size(); ++i) {
                int32_t target = (env.rank() + i) % env.size();
                auto send_type = get_big_type<DataType>(send_counts[target]);
                MPI_Isend(
                    send_data + send_displacements[target],
                    1,
                    send_type,
                    target,
                    44227,
                    env.communicator(),
                    &mpi_request[env.size() + target]
                );
            }
            MPI_Waitall(2 * env.size(), mpi_request.data(), MPI_STATUSES_IGNORE);
            return receive_data;
        }
    }
};

// template <typename DataType>
//  inline std::vector<DataType> alltoallv(std::vector<DataType>& send_data,
//      const std::vector<size_t>& send_counts, environment env = environment())
//      {

//    size_t local_send_count = std::accumulate(
//        send_counts.begin(), send_counts.end(), 0);

//    std::vector<size_t> receive_counts = alltoall(send_counts, env);
//    size_t local_receive_count = std::accumulate(
//        receive_counts.begin(), receive_counts.end(), 0);

//    size_t local_max = std::max(local_send_count, local_receive_count);
//    size_t global_max = allreduce_max(local_max, env);

//    if (global_max < env.mpi_max_int()) {
//      return alltoallv_small(send_data, send_counts, env);
//    } else {
//      std::vector<size_t> send_displacements(env.size(), 0);
//      for (size_t i = 1; i < send_counts.size(); ++i) {
//        send_displacements[i] = send_displacements[i - 1] + send_counts[i -
//        1];
//      }
//      std::vector<size_t> receive_displacements(env.size(), 0);
//      for (size_t i = 1; i < send_counts.size(); ++i) {
//        receive_displacements[i] =
//          receive_displacements[i - 1] + receive_counts[i - 1];
//      }

//      std::vector<MPI_Request> mpi_request(2 * env.size());
//      std::vector<DataType> receive_data(receive_displacements.back() +
//          receive_counts.back());
//      for (int32_t i = 0; i < env.size(); ++i) {
//        // start with self send/recv
//        auto source = (env.rank() + (env.size() - i)) % env.size();
//        auto receive_type = get_big_type<DataType>(receive_counts[source]);
//        MPI_Irecv(receive_data.data() + receive_displacements[source],
//            1,
//            receive_type,
//            source,
//            44227,
//            env.communicator(),
//            &mpi_request[source]);
//      }
//      // dispatch sends
//      for (int32_t i = 0; i < env.size(); ++i) {
//        auto target = (env.rank() + i) % env.size();
//        auto send_type = get_big_type<DataType>(send_counts[target]);
//        MPI_Isend(send_data.data() + send_displacements[target],
//            1,
//            send_type,
//            target,
//            44227,
//            env.communicator(),
//            &mpi_request[env.size() + target]);
//      }
//      MPI_Waitall(2 * env.size(), mpi_request.data(), MPI_STATUSES_IGNORE);
//      return receive_data;
//    }
//  }

template <typename StringSet, typename ByteEncoder>
static inline std::vector<size_t> computeSendCountsBytes(
    StringSet const& ss, std::vector<size_t> const& intervals, ByteEncoder const& byteEncoder
) {
    auto const begin = ss.begin();
    std::vector<size_t> sendCountsBytes(intervals.size(), 0);
    for (size_t interval = 0, offset = 0; interval < intervals.size(); ++interval) {
        size_t sendCountChars = 0;
        for (size_t j = offset; j < intervals[interval] + offset; ++j) {
            size_t stringLength = ss.get_length(ss[begin + j]);
            sendCountChars += stringLength + 1;
        }
        sendCountsBytes[interval] =
            byteEncoder.computeNumberOfSendBytes(sendCountChars, intervals[interval]);
        offset += intervals[interval];
    }
    return sendCountsBytes;
}

/*
 * Different functions for the alltoall exchange of strings.
 * AllToAllPolicy: Used MPIroutine, ie. AllToAllvSmall, AllToAllvCombined, etc.
 * ByteEncoderPolicy: Different policies for the Memory-Layout
 * (and the process of making strings contiguous (memcpy, std::copy)) for
 * strings and lcps values to send (see byte_encoder.hpp)
 *
 */

template <typename LcpIterator, typename IntervalIterator>
inline void setLcpAtStartOfInterval(
    LcpIterator lcpIt, const IntervalIterator begin, const IntervalIterator end
) {
    for (IntervalIterator it = begin; it != end; ++it) {
        if (*it == 0)
            continue;
        *lcpIt = 0;
        std::advance(lcpIt, *it);
    }
}

template <bool useCompression, typename AllToAllPolicy>
inline std::vector<uint64_t> sendLcps(
    std::vector<uint64_t>& lcps,
    std::vector<uint64_t> const& sendCounts,
    std::vector<uint64_t> const& recvCounts
) {
    using namespace dss_schimek;
    if constexpr (useCompression) {
        auto compressedData =
            IntegerCompression::writeRanges(sendCounts.begin(), sendCounts.end(), lcps.data());

        std::vector<uint8_t> recvCompressedLcps =
            AllToAllPolicy::alltoallv(compressedData.integers.data(), compressedData.counts);

        // decode Lcp Compression:
        return IntegerCompression::readRanges(
            recvCounts.begin(),
            recvCounts.end(),
            recvCompressedLcps.data()
        );
    } else {
        return AllToAllPolicy::alltoallv(lcps.data(), sendCounts);
    }
}

template <
    bool compressLcps,
    typename StringSet,
    typename AllToAllPolicy,
    typename ByteEncoderPolicy>
struct AllToAllStringImpl {
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = false;

    dss_schimek::StringLcpContainer<StringSet> alltoallv(
        dss_schimek::StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& sendCountsString,
        environment env = environment()
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("all_to_all_strings_intern_copy");

        const ByteEncoderPolicy byteEncoder;
        StringSet const& sendSet = container.make_string_set();

        std::vector<size_t> sendCountsTotal = computeSendCountsBytes<StringSet, ByteEncoderPolicy>(
            sendSet,
            sendCountsString,
            byteEncoder
        );

        size_t totalNumberSendBytes = std::accumulate(
            sendCountsTotal.begin(),
            sendCountsTotal.end(),
            static_cast<size_t>(0u)
        );

        auto stringLcpPtr = container.make_string_lcp_ptr();
        // for (size_t interval = 0, stringsWritten = 0;
        //     interval < sendCountsString.size(); ++interval) {
        //    *(stringLcpPtr.get_lcp() + stringsWritten) = 0;
        //    stringsWritten += sendCountsString[interval];
        //}
        setLcpAtStartOfInterval(
            stringLcpPtr.get_lcp(),
            sendCountsString.begin(),
            sendCountsString.end()
        );

        std::vector<unsigned char> buffer(totalNumberSendBytes);
        unsigned char* curPos = buffer.data();
        for (size_t interval = 0, stringsWritten = 0; interval < sendCountsString.size();
             ++interval) {
            auto begin = sendSet.begin() + stringsWritten;
            StringSet subSet = sendSet.sub(begin, begin + sendCountsString[interval]);
            curPos = byteEncoder.write(curPos, subSet, container.lcp_array() + stringsWritten);
            stringsWritten += sendCountsString[interval];
        }
        measuringTool.stop("all_to_all_strings_intern_copy");

        measuringTool.start("all_to_all_strings_mpi");
        std::vector<unsigned char> recv =
            AllToAllPolicy::alltoallv(buffer.data(), sendCountsTotal, env);
        measuringTool.stop("all_to_all_strings_mpi");
        measuringTool.add(totalNumberSendBytes, "bytes_sent");

        measuringTool.start("all_to_all_strings_read");
        auto [rawStrings, rawLcps] = byteEncoder.read(recv.data(), recv.size());
        measuringTool.stop("all_to_all_strings_read");
        return StringLcpContainer<StringSet>(std::move(rawStrings), std::move(rawLcps));
    }
};

template <bool compressLcps, typename StringSet, typename AllToAllPolicy>
struct AllToAllStringImpl<
    compressLcps,
    StringSet,
    AllToAllPolicy,
    dss_schimek::EmptyByteEncoderCopy> {
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = false;

    dss_schimek::StringLcpContainer<StringSet> alltoallv(
        dss_schimek::StringLcpContainer<StringSet>& send_data,
        std::vector<size_t> const& send_counts,
        environment env = environment()
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;

        using String = typename StringSet::String;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("all_to_all_strings_intern_copy");
        const StringSet ss = send_data.make_string_set();

        if (send_data.size() == 0)
            return dss_schimek::StringLcpContainer<StringSet>();

        std::vector<unsigned char> receive_buffer_char;
        std::vector<size_t> receive_buffer_lcp;
        std::vector<size_t> send_counts_lcp(send_counts);
        std::vector<size_t> send_counts_char(send_counts.size(), 0);

        auto stringLcpPtr = send_data.make_string_lcp_ptr();

        // for (size_t interval = 0, stringsWritten = 0;
        //     interval < send_counts.size(); ++interval) {
        //    *(stringLcpPtr.get_lcp() + stringsWritten) = 0;
        //    stringsWritten += send_counts[interval];
        //}
        setLcpAtStartOfInterval(stringLcpPtr.get_lcp(), send_counts.begin(), send_counts.end());

        std::vector<unsigned char> send_buffer(send_data.char_size());
        uint64_t curPos = 0;
        for (size_t interval = 0, offset = 0; interval < send_counts.size(); ++interval) {
            for (size_t j = offset; j < send_counts[interval] + offset; ++j) {
                String str = ss[ss.begin() + j];
                size_t string_length = ss.get_length(str) + 1;

                send_counts_char[interval] += string_length;
                std::copy(
                    ss.get_chars(str, 0),
                    ss.get_chars(str, 0) + string_length,
                    send_buffer.begin() + curPos
                );
                curPos += string_length;
            }
            offset += send_counts[interval];
        }

        measuringTool.stop("all_to_all_strings_intern_copy");
        measuringTool.start("all_to_all_strings_mpi");
        receive_buffer_char = AllToAllPolicy::alltoallv(send_buffer.data(), send_counts_char, env);

        auto recvNumberStrings = dss_schimek::mpi::alltoall(send_counts);

        auto recvLcpValues = sendLcps<compressLcps, AllToAllPolicy>(
            send_data.lcps(),
            send_counts,
            recvNumberStrings
        );

        measuringTool.stop("all_to_all_strings_mpi");
        measuringTool.add(send_data.char_size() + send_data.size(), "bytes_sent");

        // no bytes are read in this version only for evaluation layout
        measuringTool.start("all_to_all_strings_read");
        measuringTool.stop("all_to_all_strings_read");
        return dss_schimek::StringLcpContainer<StringSet>(
            std::move(receive_buffer_char),
            std::move(recvLcpValues)
        );
    }
};

template <bool compressLcps, typename StringSet, typename AllToAllPolicy>
struct AllToAllStringImpl<compressLcps, StringSet, AllToAllPolicy, dss_schimek::NoLcps> {
    static constexpr bool Lcps = false;
    static constexpr bool PrefixCompression = false;
    dss_schimek::StringLcpContainer<StringSet> alltoallv(
        dss_schimek::StringLcpContainer<StringSet>& send_data,
        std::vector<size_t> const& sendCountsString,
        environment env = environment()
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("all_to_all_strings_intern_copy");
        const EmptyByteEncoderMemCpy byteEncoder;
        const StringSet ss = send_data.make_string_set();

        if (send_data.size() == 0)
            return dss_schimek::StringLcpContainer<StringSet>();

        std::vector<unsigned char> receive_buffer_char;
        // std::vector<size_t> receive_buffer_lcp;
        // std::vector<size_t> send_counts_lcp(sendCountsString);
        std::vector<size_t> send_counts_char(sendCountsString.size());

        // auto stringLcpPtr = send_data.make_string_lcp_ptr();
        //  for (size_t interval = 0, stringsWritten = 0;
        //      interval < sendCountsString.size(); ++interval) {
        //     *(stringLcpPtr.get_lcp() + stringsWritten) = 0;
        //     stringsWritten += sendCountsString[interval];
        // }
        // setLcpAtStartOfInterval(stringLcpPtr.get_lcp(),
        //     sendCountsString.begin(), sendCountsString.end());

        std::vector<unsigned char> buffer(send_data.char_size());
        unsigned char* curPos = buffer.data();
        for (size_t interval = 0, stringsWritten = 0; interval < sendCountsString.size();
             ++interval) {
            auto begin = ss.begin() + stringsWritten;
            StringSet subSet = ss.sub(begin, begin + sendCountsString[interval]);
            size_t numWrittenChars = 0;
            std::tie(curPos, numWrittenChars) = byteEncoder.write(curPos, subSet);

            send_counts_char[interval] = numWrittenChars;
            stringsWritten += sendCountsString[interval];
        }

        send_data.deleteRawStrings();
        send_data.deleteStrings();

        measuringTool.stop("all_to_all_strings_intern_copy");
        measuringTool.start("all_to_all_strings_mpi");
        receive_buffer_char = AllToAllPolicy::alltoallv(buffer.data(), send_counts_char, env);
        send_data.deleteAll();

        measuringTool.stop("all_to_all_strings_mpi");
        measuringTool.add(send_data.char_size() + send_data.size(), "bytes_sent");

        // no bytes are read in this version only for evaluation layout
        measuringTool.start("all_to_all_strings_read");
        measuringTool.stop("all_to_all_strings_read");
        measuringTool.start("container_construction");
        dss_schimek::StringLcpContainer<StringSet> returnContainer(std::move(receive_buffer_char));
        measuringTool.stop("container_construction");
        return returnContainer;
    }
};
template <bool compressLcps, typename StringSet, typename AllToAllPolicy>
struct AllToAllStringImpl<
    compressLcps,
    StringSet,
    AllToAllPolicy,
    dss_schimek::EmptyByteEncoderMemCpy> {
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = false;
    dss_schimek::StringLcpContainer<StringSet> alltoallv(
        dss_schimek::StringLcpContainer<StringSet>& send_data,
        std::vector<size_t> const& sendCountsString,
        environment env = environment()
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("all_to_all_strings_intern_copy");
        const EmptyByteEncoderMemCpy byteEncoder;
        const StringSet ss = send_data.make_string_set();

        if (send_data.size() == 0)
            return dss_schimek::StringLcpContainer<StringSet>();

        std::vector<unsigned char> receive_buffer_char;
        std::vector<size_t> receive_buffer_lcp;
        std::vector<size_t> send_counts_lcp(sendCountsString);
        std::vector<size_t> send_counts_char(sendCountsString.size());

        auto stringLcpPtr = send_data.make_string_lcp_ptr();
        // for (size_t interval = 0, stringsWritten = 0;
        //     interval < sendCountsString.size(); ++interval) {
        //    *(stringLcpPtr.get_lcp() + stringsWritten) = 0;
        //    stringsWritten += sendCountsString[interval];
        //}
        setLcpAtStartOfInterval(
            stringLcpPtr.get_lcp(),
            sendCountsString.begin(),
            sendCountsString.end()
        );

        std::vector<unsigned char> buffer(send_data.char_size());
        unsigned char* curPos = buffer.data();
        for (size_t interval = 0, stringsWritten = 0; interval < sendCountsString.size();
             ++interval) {
            auto begin = ss.begin() + stringsWritten;
            StringSet subSet = ss.sub(begin, begin + sendCountsString[interval]);
            size_t numWrittenChars = 0;
            std::tie(curPos, numWrittenChars) = byteEncoder.write(curPos, subSet);

            send_counts_char[interval] = numWrittenChars;
            stringsWritten += sendCountsString[interval];
        }

        send_data.deleteRawStrings();
        send_data.deleteStrings();

        measuringTool.stop("all_to_all_strings_intern_copy");
        measuringTool.start("all_to_all_strings_mpi");
        receive_buffer_char = AllToAllPolicy::alltoallv(buffer.data(), send_counts_char, env);

        auto recvNumberStrings = dss_schimek::mpi::alltoall(sendCountsString);

        auto recvLcpValues = sendLcps<compressLcps, AllToAllPolicy>(
            send_data.lcps(),
            sendCountsString,
            recvNumberStrings
        );

        send_data.deleteAll();

        measuringTool.stop("all_to_all_strings_mpi");
        measuringTool.add(send_data.char_size() + send_data.size(), "bytes_sent");

        // no bytes are read in this version only for evaluation layout
        measuringTool.start("all_to_all_strings_read");
        measuringTool.stop("all_to_all_strings_read");
        measuringTool.start("container_construction");
        dss_schimek::StringLcpContainer<StringSet> returnContainer(
            std::move(receive_buffer_char),
            std::move(recvLcpValues)
        );
        measuringTool.stop("container_construction");
        return returnContainer;
    }
};

template <bool compressLcps, typename StringSet, typename AllToAllPolicy>
struct AllToAllStringImpl<
    compressLcps,
    StringSet,
    AllToAllPolicy,
    dss_schimek::EmptyLcpByteEncoderMemCpy> {
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = true;

    dss_schimek::StringLcpContainer<StringSet> alltoallv(
        dss_schimek::StringLcpContainer<StringSet>& send_data,
        std::vector<size_t> const& sendCountsString,
        environment env = environment()
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();
        measuringTool.start("all_to_all_strings_intern_copy");

        const EmptyLcpByteEncoderMemCpy byteEncoder;
        const StringSet ss = send_data.make_string_set();

        if (send_data.size() == 0)
            return dss_schimek::StringLcpContainer<StringSet>();

        std::vector<unsigned char> receive_buffer_char;
        std::vector<size_t> receive_buffer_lcp;
        std::vector<size_t> send_counts_lcp(sendCountsString);
        std::vector<size_t> send_counts_char(sendCountsString.size());

        std::vector<size_t>& lcps = send_data.lcps();
        setLcpAtStartOfInterval(lcps.begin(), sendCountsString.begin(), sendCountsString.end());
        const size_t L = std::accumulate(lcps.begin(), lcps.end(), static_cast<size_t>(0u));

        const size_t numCharsToSend = send_data.char_size() - L;
        // std::cout << " numCharsToSend: " << numCharsToSend << std::endl;
        std::vector<unsigned char> buffer(numCharsToSend);
        unsigned char* curPos = buffer.data();
        size_t totalNumWrittenChars = 0;

        for (size_t interval = 0, stringsWritten = 0; interval < sendCountsString.size();
             ++interval) {
            auto begin = ss.begin() + stringsWritten;
            StringSet subSet = ss.sub(begin, begin + sendCountsString[interval]);
            size_t numWrittenChars = 0;
            std::tie(curPos, numWrittenChars) =
                byteEncoder.write(curPos, subSet, lcps.data() + stringsWritten);

            totalNumWrittenChars += numWrittenChars;
            send_counts_char[interval] = numWrittenChars;
            stringsWritten += sendCountsString[interval];
        }
        send_data.deleteRawStrings();
        send_data.deleteStrings();

        measuringTool.stop("all_to_all_strings_intern_copy");
        measuringTool.start("all_to_all_strings_mpi");
        receive_buffer_char = AllToAllPolicy::alltoallv(buffer.data(), send_counts_char, env);

        auto recvNumberStrings = dss_schimek::mpi::alltoall(sendCountsString);

        auto recvLcpValues = sendLcps<compressLcps, AllToAllPolicy>(
            send_data.lcps(),
            sendCountsString,
            recvNumberStrings
        );
        send_data.deleteAll();
        measuringTool.stop("all_to_all_strings_mpi");
        measuringTool.add(numCharsToSend + send_data.lcps().size(), "bytes_sent");

        //// no bytes are read in this version only for evaluation layout
        measuringTool.start("all_to_all_strings_read");
        measuringTool.stop("all_to_all_strings_read");

        measuringTool.start("container_construction");
        dss_schimek::StringLcpContainer<StringSet> recvContainer(
            std::move(receive_buffer_char),
            std::move(recvLcpValues)
        );
        measuringTool.stop("container_construction");
        return recvContainer;
    }
};

template <typename StringSet>
struct RecvDataAllToAll {
    dss_schimek::StringLcpContainer<StringSet> container;
    std::vector<size_t> recvStringSizes;
    RecvDataAllToAll(
        dss_schimek::StringLcpContainer<StringSet>&& container,
        std::vector<size_t>&& recvStringSizes
    )
        : container(std::move(container)),
          recvStringSizes(std::move(recvStringSizes)) {}
};

// Function for All-To-All exchange of all strings in the
// distributed-merge-sort-algorithm, using the sequentialDelayedByteEncoder
template <bool compressLcps, typename StringSet, typename AllToAllPolicy>
struct AllToAllStringImpl<
    compressLcps,
    StringSet,
    AllToAllPolicy,
    dss_schimek::SequentialDelayedByteEncoder> {
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = false;

    dss_schimek::StringLcpContainer<StringSet> alltoallv(
        dss_schimek::StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& sendCountsString,
        environment env = environment()
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        const SequentialDelayedByteEncoder byteEncoder;

        measuringTool.start("all_to_all_strings_intern_copy");
        StringSet const& sendSet = container.make_string_set();
        std::vector<unsigned char> contiguousStrings =
            dss_schimek::getContiguousStrings(sendSet, container.char_size());

        std::vector<size_t> const& sendCountsLcp(sendCountsString);
        std::vector<size_t> sendCountsChar(sendCountsString.size(), 0);
        std::vector<size_t> sendCountsTotal(sendCountsString.size(), 0);

        auto stringLcpPtr = container.make_string_lcp_ptr();

        // for (size_t interval = 0, stringsWritten = 0;
        //     interval < sendCountsString.size(); ++interval) {
        //    *(stringLcpPtr.get_lcp() + stringsWritten) = 0;
        //    stringsWritten += sendCountsString[interval];
        //}
        setLcpAtStartOfInterval(
            stringLcpPtr.get_lcp(),
            sendCountsString.begin(),
            sendCountsString.end()
        );

        for (size_t interval = 0, offset = 0; interval < sendCountsString.size(); ++interval) {
            for (size_t j = offset; j < sendCountsString[interval] + offset; ++j) {
                size_t stringLength = sendSet.get_length(sendSet[sendSet.begin() + j]);
                sendCountsChar[interval] += stringLength + 1;
            }
            sendCountsTotal[interval] = byteEncoder.computeNumberOfSendBytes(
                sendCountsChar[interval],
                sendCountsLcp[interval]
            );
            offset += sendCountsString[interval];
        }

        size_t totalNumberSendBytes =
            byteEncoder.computeNumberOfSendBytes(sendCountsChar, sendCountsLcp);

        std::vector<unsigned char> buffer(totalNumberSendBytes);
        unsigned char* curPos = buffer.data();

        for (size_t interval = 0, offset = 0, stringsWritten = 0;
             interval < sendCountsString.size();
             ++interval) {
            curPos = byteEncoder.write(
                curPos,
                contiguousStrings.data() + offset,
                sendCountsChar[interval],
                container.lcp_array() + stringsWritten,
                sendCountsLcp[interval]
            );
            offset += sendCountsChar[interval];
            stringsWritten += sendCountsLcp[interval];
        }

        measuringTool.stop("all_to_all_strings_intern_copy");
        measuringTool.start("all_to_all_strings_mpi");
        std::vector<unsigned char> recv =
            AllToAllPolicy::alltoallv(buffer.data(), sendCountsTotal, env);
        measuringTool.stop("all_to_all_strings_mpi");
        measuringTool.add(totalNumberSendBytes, "bytes_sent");

        measuringTool.start("all_to_all_strings_read");
        auto [rawStrings, rawLcps] = byteEncoder.read(recv.data(), recv.size());
        measuringTool.stop("all_to_all_strings_read");
        return StringLcpContainer<StringSet>(std::move(rawStrings), std::move(rawLcps));
    }
};

// Method for the All-To-All exchange of the reduced strings, i.e. only
// distinguishing prefix - common prefix, in the prefix-doubling-algorithm
template <bool compressLcps, typename StringLcpPtr, typename AllToAllPolicy>
struct AllToAllStringImplPrefixDoubling {
    using StringSet = typename StringLcpPtr::StringSet;
    using ReturnStringSet = dss_schimek::UCharIndexPEIndexStringSet;
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = true;

    dss_schimek::StringLcpContainer<ReturnStringSet> alltoallv(
        dss_schimek::StringLcpContainer<StringSet>&& send_data,
        std::vector<size_t> const& sendCountsString,
        std::vector<size_t> const& distinguishingPrefixValues,
        environment env = environment()
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        measuringTool.start("all_to_all_strings_intern_copy");
        const EmptyPrefixDoublingLcpByteEncoderMemCpy byteEncoder;
        StringLcpPtr stringLcpPtr = send_data.make_string_lcp_ptr();
        const StringSet ss = stringLcpPtr.active();

        if (ss.size() == 0)
            return dss_schimek::StringLcpContainer<ReturnStringSet>();

        std::vector<unsigned char> receive_buffer_char;
        std::vector<size_t> receive_buffer_lcp;
        std::vector<size_t> send_counts_lcp(sendCountsString);
        std::vector<size_t> send_counts_char(sendCountsString.size());

        // for (size_t interval = 0, stringsWritten = 0;
        //     interval < sendCountsString.size(); ++interval) {
        //    *(stringLcpPtr.get_lcp() + stringsWritten) = 0;
        //    stringsWritten += sendCountsString[interval];
        //}
        setLcpAtStartOfInterval(
            stringLcpPtr.get_lcp(),
            sendCountsString.begin(),
            sendCountsString.end()
        );

        const size_t L = std::accumulate(
            stringLcpPtr.get_lcp(),
            stringLcpPtr.get_lcp() + stringLcpPtr.size(),
            static_cast<size_t>(0u)
        );
        const size_t D = std::accumulate(
            distinguishingPrefixValues.begin(),
            distinguishingPrefixValues.end(),
            static_cast<size_t>(0u)
        );
        measuringTool.add(L, "localL");
        measuringTool.add(D, "localD");

        const size_t numCharsToSend = stringLcpPtr.size() + D - L;

        // std::cout << "numCharsToSend : " << numCharsToSend << std::endl;
        // std::cout << "localL : " << L << std::endl;
        // std::cout << "localD : " << D << std::endl;
        std::vector<unsigned char> buffer(numCharsToSend);
        unsigned char* curPos = buffer.data();
        size_t totalNumWrittenChars = 0;
        for (size_t interval = 0, stringsWritten = 0; interval < sendCountsString.size();
             ++interval) {
            auto begin = ss.begin() + stringsWritten;
            StringSet subSet = ss.sub(begin, begin + sendCountsString[interval]);
            size_t numWrittenChars = 0;
            std::tie(curPos, numWrittenChars) = byteEncoder.write(
                curPos,
                subSet,
                stringLcpPtr.get_lcp() + stringsWritten,
                distinguishingPrefixValues.data() + stringsWritten
            );

            totalNumWrittenChars += numWrittenChars;
            send_counts_char[interval] = numWrittenChars;
            stringsWritten += sendCountsString[interval];
        }
        send_data.deleteRawStrings();
        send_data.deleteStrings();

        measuringTool.stop("all_to_all_strings_intern_copy");
        measuringTool.start("all_to_all_strings_mpi");
        receive_buffer_char = AllToAllPolicy::alltoallv(buffer.data(), send_counts_char, env);

        std::vector<size_t> recvNumberStrings = dss_schimek::mpi::alltoall(sendCountsString);

        auto recvLcpValues = sendLcps<compressLcps, AllToAllPolicy>(
            send_data.lcps(),
            sendCountsString,
            recvNumberStrings
        );
        send_data.deleteAll();

        // decode Lcp Compression:
        std::vector<size_t> offsets;
        offsets.reserve(env.size());
        offsets.push_back(0);
        std::partial_sum(
            sendCountsString.begin(),
            sendCountsString.end() - 1,
            std::back_inserter(offsets)
        );

        std::vector<size_t> recvOffsets = dss_schimek::mpi::alltoall(offsets);
        measuringTool.stop("all_to_all_strings_mpi");
        measuringTool.add(
            numCharsToSend + stringLcpPtr.size() * sizeof(size_t),
            "string_exchange_bytes_sent"
        );

        //// no bytes are read in this version only for evaluation layout
        measuringTool.start("all_to_all_strings_read");
        measuringTool.stop("all_to_all_strings_read");
        measuringTool.start("container_construction");
        dss_schimek::StringLcpContainer<ReturnStringSet> returnContainer(
            std::move(receive_buffer_char),
            std::move(recvLcpValues),
            recvNumberStrings,
            recvOffsets
        );
        measuringTool.stop("container_construction");
        return returnContainer;
    }
};

// Collect Strings specified by string index and PE index -> should be used for
// verification (probably not as efficient as it could be)
template <typename StringSet, typename Iterator>
dss_schimek::StringLcpContainer<StringSet> getStrings(
    Iterator requestBegin,
    Iterator requestEnd,
    StringSet localSS,
    dss_schimek::mpi::environment env = dss_schimek::mpi::environment()
) {
    using String = typename StringSet::String;
    using CharIt = typename StringSet::CharIterator;
    using MPIRoutine = dss_schimek::mpi::AllToAllvCombined<dss_schimek::mpi::AllToAllvSmall>;

    std::vector<size_t> requests(requestEnd - requestBegin);
    std::vector<size_t> requestSizes(env.size(), 0);

    // get size of each interval
    for (Iterator curIt = requestBegin; curIt != requestEnd; ++curIt) {
        const StringIndexPEIndex indices = *curIt;
        ++requestSizes[indices.PEIndex];
    }

    std::vector<size_t> offsets;
    offsets.reserve(env.size());
    offsets.push_back(0);
    std::partial_sum(requestSizes.begin(), requestSizes.end() - 1, std::back_inserter(offsets));

    for (Iterator curIt = requestBegin; curIt != requestEnd; ++curIt) {
        const StringIndexPEIndex indices = *curIt;
        requests[offsets[indices.PEIndex]++] = indices.stringIndex;
    };

    // exchange request
    std::vector<size_t> recvRequestSizes = dss_schimek::mpi::alltoall(requestSizes);
    std::vector<size_t> recvRequests = MPIRoutine::alltoallv(requests.data(), requestSizes);

    // collect strings to send back
    std::vector<std::vector<unsigned char>> rawStrings(env.size());
    std::vector<size_t> rawStringSizes(env.size());
    size_t offset = 0;

    for (size_t curRank = 0; curRank < env.size(); ++curRank) {
        for (size_t i = 0; i < recvRequestSizes[curRank]; ++i) {
            const size_t requestedIndex = recvRequests[offset + i];
            String str = localSS[localSS.begin() + requestedIndex];
            size_t length = localSS.get_length(str) + 1;
            CharIt charsStr = localSS.get_chars(str, 0);
            std::copy_n(charsStr, length, std::back_inserter(rawStrings[curRank]));
            rawStringSizes[curRank] += length;
        }
        offset += recvRequestSizes[curRank];
    }

    std::vector<unsigned char> rawStringsFlattened = flatten(rawStrings);

    std::vector<unsigned char> recvRequestedStrings =
        MPIRoutine::alltoallv(rawStringsFlattened.data(), rawStringSizes);

    dss_schimek::StringLcpContainer<StringSet> container(std::move(recvRequestedStrings));

    return container;
}
} // namespace dss_schimek::mpi
/******************************************************************************/
