// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <tlx/die.hpp>

#include "mpi/allreduce.hpp"
#include "mpi/big_type.hpp"
#include "mpi/byte_encoder.hpp"
#include "mpi/environment.hpp"
#include "mpi/type_mapper.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"
#include "util/structs.hpp"

namespace dss_schimek::mpi {

static constexpr bool debug_alltoall = false;

template <typename DataType>
inline std::vector<DataType> alltoall(std::vector<DataType> const& send_data, environment env) {
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
    std::vector<DataType> const& send_data, std::vector<size_t> const& send_counts, environment env
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
        std::accumulate(real_send_counts.begin(), real_send_counts.end(), size_t{0});
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
    static std::vector<DataType>
    alltoallv(DataType* const send_data, std::vector<size_t> const& send_counts, environment env) {
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
            std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
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
    static std::vector<DataType>
    alltoallv(DataType* const send_data, std::vector<size_t> const& send_counts, environment env) {
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

        const size_t elemToSend = std::accumulate(send_counts.begin(), send_counts.end(), size_t{0})
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
    static std::vector<DataType>
    alltoallv(DataType* const send_data, std::vector<size_t> const& send_counts, environment env) {
        // std::cout << "MPI Routine : combined" << std::endl;
        using dss_schimek::measurement::MeasuringTool;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        size_t local_send_count =
            std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});

        std::vector<size_t> recv_counts = alltoall(send_counts, env);
        size_t local_recv_count =
            std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{0});

        size_t local_max = std::max(local_send_count, local_recv_count);
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
                receive_displacements[i] = receive_displacements[i - 1] + recv_counts[i - 1];
            }

            const size_t elemToSend =
                std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
            measuringTool.addRawCommunication(elemToSend * sizeof(DataType), "alltoallv_combined");

            std::vector<MPI_Request> mpi_request(2 * env.size());
            std::vector<DataType> receive_data(receive_displacements.back() + recv_counts.back());
            for (uint32_t i = 0; i < env.size(); ++i) {
                // start with self send/recv
                int32_t source = (env.rank() + (env.size() - i)) % env.size();
                auto receive_type = get_big_type<DataType>(recv_counts[source]);
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


namespace _internal {

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

template <bool compress_prefixes>
std::pair<unsigned char*, size_t>
write_interval(unsigned char* buffer, auto const& ss, size_t const* lcps) {
    auto dest = buffer;
    for (auto const& str: ss) {
        size_t str_depth;
        if constexpr (compress_prefixes) {
            str_depth = *lcps++;
        } else {
            str_depth = 0;
        }

        auto str_len = ss.get_length(str) + 1;
        auto actual_len = str_len - str_depth;
        memcpy(dest, ss.get_chars(str, str_depth), actual_len);
        dest += actual_len;
    }
    return {buffer, static_cast<size_t>(dest - buffer)};
}

template <bool compress_prefixes>
std::pair<unsigned char*, size_t>
write_interval(unsigned char* buffer, auto const& ss, size_t const* lcps, size_t const* prefixes) {
    auto dest = buffer;
    for (auto const& str: ss) {
        size_t str_depth;
        if constexpr (compress_prefixes) {
            str_depth = *lcps++;
        } else {
            str_depth = 0;
        }

        auto str_len = *prefixes++;
        auto actual_len = str_len - str_depth;
        memcpy(dest, ss.get_chars(str, str_depth), actual_len);
        buffer += actual_len;
        *buffer++ = 0;
    }
    return {buffer, static_cast<size_t>(dest - buffer)};
}

template <bool compress_prefixes>
size_t get_num_send_chars(auto& container) {
    if constexpr (compress_prefixes) {
        auto lcps = container.lcps();
        auto L = std::accumulate(lcps.begin(), lcps.end(), size_t{0});
        return container.char_size() - L;
    } else {
        return container.char_size();
    }
}

template <bool compress_prefixes>
size_t get_num_send_chars(auto& container, std::vector<size_t> const& prefixes) {
    auto D = std::accumulate(prefixes.begin(), prefixes.end(), size_t{0});
    if constexpr (compress_prefixes) {
        auto lcps = container.lcps();
        auto L = std::accumulate(lcps.begin(), lcps.end(), size_t{0});
        return container.size() + D - L;
    } else {
        return container.size() + D;
    }
}

template <bool compress_prefixes>
std::pair<std::vector<unsigned char>, std::vector<size_t>>
write_send_buf(auto& container, std::vector<size_t> const& send_counts, auto const&... prefixes) {
    auto& lcps = container.lcps();
    setLcpAtStartOfInterval(lcps.begin(), send_counts.begin(), send_counts.end());

    auto num_send_chars = get_num_send_chars(container, prefixes...);
    std::vector<unsigned char> send_buf_char(num_send_chars);
    std::vector<size_t> send_counts_char(send_counts.size());

    auto ss = container.make_string_set();
    unsigned char* pos = send_buf_char.data();

    for (size_t interval = 0, strs_written = 0; interval < send_counts.size(); ++interval) {
        auto begin = ss.begin() + strs_written;
        auto end = begin + send_counts[interval];
        auto subset = ss.sub(begin, end);

        size_t chars_written = 0;
        std::tie(pos, chars_written) = write_interval<compress_prefixes>(
            pos,
            subset,
            lcps.data() + strs_written,
            (prefixes.data() + strs_written)...
        );

        send_counts_char[interval] = chars_written;
        strs_written += send_counts[interval];
    }
}


template <bool useCompression, typename AllToAllPolicy>
inline std::vector<uint64_t> send_u64s(
    std::vector<uint64_t>& values,
    std::vector<uint64_t> const& send_counts,
    std::vector<uint64_t> const& recv_counts,
    environment env
) {
    using namespace dss_schimek;
    if constexpr (useCompression) {
        auto compressedData =
            IntegerCompression::writeRanges(send_counts.begin(), send_counts.end(), values.data());

        std::vector<uint8_t> recvCompressedLcps =
            AllToAllPolicy::alltoallv(compressedData.integers.data(), compressedData.counts, env);

        // decode Lcp Compression:
        return IntegerCompression::readRanges(
            recv_counts.begin(),
            recv_counts.end(),
            recvCompressedLcps.data()
        );
    } else {
        // todo why isn't this using recv_counts?
        return AllToAllPolicy::alltoallv(values.data(), send_counts, env);
    }
}

template <bool compress_lcps, typename StringSet, typename AllToAllPolicy>
class StringSetSendImpl {
public:
    static StringLcpContainer<StringSet> alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<unsigned char>& send_buf_char,
        std::vector<size_t> const& send_counts_char,
        std::vector<size_t> const& send_counts,
        mpi::environment env
    ) {
        using dss_mehnert::measurement::MeasuringTool;
        auto& measuring_tool = MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_mpi");
        auto recv_buf_char = AllToAllPolicy::alltoallv(send_buf_char.data(), send_counts_char, env);

        auto recv_counts = mpi::alltoall(send_counts, env);
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, env);
        container.deleteAll();
        measuring_tool.stop("all_to_all_strings_mpi");

        measuring_tool.start("container_construction");
        StringLcpContainer<StringSet> recv_container{
            std::move(recv_buf_char),
            std::move(recv_buf_lcp)};
        measuring_tool.stop("container_construction");
        return recv_container;
    }
};

template <bool compress_lcps, typename AllToAllPolicy>
class StringSetSendImpl<compress_lcps, UCharLengthIndexPEIndexStringSet, AllToAllPolicy> {
    using StringSet = UCharLengthIndexPEIndexStringSet;

public:
    static StringLcpContainer<StringSet> alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<unsigned char>& send_buf_char,
        std::vector<size_t> const& send_counts_char,
        std::vector<size_t> const& send_counts,
        mpi::environment env
    ) {
        using dss_mehnert::measurement::MeasuringTool;
        auto& measuring_tool = MeasuringTool::measuringTool();

        auto ss = container.make_string_set();

        std::vector<size_t> send_buf_PE_idx(ss.size());
        auto get_PE_idx = [&ss](auto const& str) { return ss.getPEIndex(str); };
        std::transform(ss.begin(), ss.end(), send_buf_PE_idx.begin(), get_PE_idx);

        std::vector<size_t> send_buf_str_idx(ss.size());
        auto get_str_idx = [&ss](auto const& str) { return ss.getIndex(str); };
        std::transform(ss.begin(), ss.end(), send_buf_str_idx.begin(), get_str_idx);

        measuring_tool.start("all_to_all_strings_mpi");
        auto recv_buf_char = AllToAllPolicy::alltoallv(send_buf_char.data(), send_counts_char, env);

        auto recv_counts = mpi::alltoall(send_counts, env);
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, env);
        container.deleteAll();

        // todo does 7-bit compression make sense for PE and string indices
        auto recv_buf_PE_idx = send_lcps(send_buf_PE_idx, send_counts, recv_counts, env);
        auto recv_buf_str_idx = send_lcps(send_buf_str_idx, send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_mpi");

        measuring_tool.start("container_construction");
        StringLcpContainer<StringSet> recv_container{
            std::move(recv_buf_char),
            std::move(recv_buf_lcp),
            std::move(recv_buf_PE_idx),
            std::move(recv_buf_str_idx)};
        measuring_tool.stop("container_construction");
        return recv_container;
    }
};

} // namespace _internal


template <bool compress_lcps, bool compress_prefixes, typename StringSet, typename AllToAllPolicy>
class AllToAllStringImpl {
public:
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = compress_prefixes;

    StringLcpContainer<StringSet> alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        environment env
    ) {
        using dss_mehnert::measurement::MeasuringTool;
        auto& measuring_tool = MeasuringTool::measuringTool();

        if (container.empty()) {
            return {};
        }

        measuring_tool.start("all_to_all_strings_intern_copy");
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<compress_prefixes>(container, send_counts);
        container.deleteRawStrings();
        measuring_tool.stop("all_to_all_strings_intern_copy");

        using SendImpl = _internal::StringSetSendImpl<compress_lcps, StringSet, AllToAllPolicy>;
        return SendImpl::alltoallv(container, send_buf_char, send_counts_char, send_counts, env);
    }
};


// Method for the All-To-All exchange of the reduced strings, i.e. only
// distinguishing prefix - common prefix, in the prefix-doubling-algorithm
// todo implement optional prefix compression
template <bool compress_lcps, typename StringSet, typename AllToAllPolicy>
struct AllToAllStringImplPrefixDoubling
    : public AllToAllStringImpl<compress_lcps, true, StringSet, AllToAllPolicy> {
    using StringPEIndexSet = dss_schimek::UCharLengthIndexPEIndexStringSet;
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = true;

    StringLcpContainer<StringPEIndexSet> alltoallv(
        StringLcpContainer<StringSet>& send_data,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& dist_prefixes,
        environment env
    ) {
        using namespace dss_schimek;

        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_intern_copy");
        const EmptyPrefixDoublingLcpByteEncoderMemCpy byteEncoder;
        auto string_ptr = send_data.make_string_lcp_ptr();
        const StringSet ss = string_ptr.active();

        if (ss.size() == 0)
            return dss_schimek::StringLcpContainer<StringPEIndexSet>{};

        std::vector<unsigned char> recv_buf_char;
        std::vector<size_t> send_counts_char(send_counts.size());

        _internal::setLcpAtStartOfInterval(
            string_ptr.lcp(),
            send_counts.begin(),
            send_counts.end()
        );

        const size_t L =
            std::accumulate(string_ptr.lcp(), string_ptr.lcp() + string_ptr.size(), size_t{0});
        const size_t D = std::accumulate(dist_prefixes.begin(), dist_prefixes.end(), size_t{0});
        measuring_tool.add(L, "localL");
        measuring_tool.add(D, "localD");

        const size_t numCharsToSend = string_ptr.size() + D - L;
        std::vector<unsigned char> buffer(numCharsToSend);

        unsigned char* curPos = buffer.data();
        for (size_t interval = 0, strs_written = 0; interval < send_counts.size(); ++interval) {
            auto begin = ss.begin() + strs_written;

            size_t chars_written = 0;
            std::tie(curPos, chars_written) = byteEncoder.write(
                curPos,
                ss.sub(begin, begin + send_counts[interval]),
                string_ptr.lcp() + strs_written,
                dist_prefixes.data() + strs_written
            );

            send_counts_char[interval] = chars_written;
            strs_written += send_counts[interval];
        }
        send_data.deleteRawStrings();
        send_data.deleteStrings();

        measuring_tool.stop("all_to_all_strings_intern_copy");
        measuring_tool.start("all_to_all_strings_mpi");
        recv_buf_char = AllToAllPolicy::alltoallv(buffer.data(), send_counts_char, env);

        auto recv_counts = dss_schimek::mpi::alltoall(send_counts, env);
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(send_data.lcps(), send_counts, recv_counts, env);
        send_data.deleteAll();

        std::vector<size_t> offsets(send_counts.size());
        std::exclusive_scan(send_counts.begin(), send_counts.end(), offsets.begin(), 0);

        auto recv_offsets = dss_schimek::mpi::alltoall(offsets, env);
        measuring_tool.stop("all_to_all_strings_mpi");
        measuring_tool.add(
            numCharsToSend + string_ptr.size() * sizeof(size_t),
            "string_exchange_bytes_sent"
        );

        measuring_tool.start("all_to_all_strings_read");
        measuring_tool.stop("all_to_all_strings_read");

        measuring_tool.start("container_construction");
        dss_schimek::StringLcpContainer<StringPEIndexSet> recv_container{
            std::move(recv_buf_char),
            std::move(recv_buf_lcp),
            recv_counts,
            recv_offsets

        };
        measuring_tool.stop("container_construction");
        return recv_container;
    }

    StringLcpContainer<StringPEIndexSet> alltoallv(
        StringLcpContainer<StringPEIndexSet>& container,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& prefixes,
        environment env
    ) {
        using dss_mehnert::measurement::MeasuringTool;
        auto& measuring_tool = MeasuringTool::measuringTool();

        if (container.empty()) {
            return {};
        }

        measuring_tool.start("all_to_all_strings_intern_copy");
        // todo add compress_prefixes parameter
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<true>(container, send_counts, prefixes);
        container.deleteRawStrings();
        measuring_tool.stop("all_to_all_strings_intern_copy");

        using SendImpl = _internal::StringSetSendImpl<compress_lcps, StringSet, AllToAllPolicy>;
        return SendImpl::alltoallv(container, send_buf_char, send_counts_char, send_counts, env);
    }
};

// Collect Strings specified by string index and PE index -> should be used for
// verification (probably not as efficient as it could be)
template <typename StringSet, typename Iterator>
dss_schimek::StringLcpContainer<StringSet>
getStrings(Iterator requestBegin, Iterator requestEnd, StringSet localSS, environment env) {
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
    std::vector<size_t> recvRequestSizes = dss_schimek::mpi::alltoall(requestSizes, env);
    std::vector<size_t> recvRequests = MPIRoutine::alltoallv(requests.data(), requestSizes, env);

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
        MPIRoutine::alltoallv(rawStringsFlattened.data(), rawStringSizes, env);

    dss_schimek::StringLcpContainer<StringSet> container(std::move(recvRequestedStrings));

    return container;
}

} // namespace dss_schimek::mpi
