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
#include <span>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <tlx/die.hpp>

#include "mpi/allreduce.hpp"
#include "mpi/big_type.hpp"
#include "mpi/byte_encoder.hpp"
#include "mpi/environment.hpp"
#include "mpi/type_mapper.hpp"
#include "sorter/distributed/permutation.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"

namespace dss_schimek {
namespace mpi {

static constexpr bool debug_alltoall = false;

template <typename DataType>
inline std::vector<DataType> alltoall(std::vector<DataType> const& send_data, environment env) {
    using dss_schimek::measurement::MeasuringTool;
    MeasuringTool& measuringTool = MeasuringTool::measuringTool();
    size_t const sentItems = send_data.size();
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

// uses builtin MPI_Alltoallv for message exchange
class AllToAllvSmall {
public:
    static std::string getName() { return "AlltoAllvSmall"; }

    template <typename DataType>
    static std::vector<DataType>
    alltoallv(DataType* const send_data, std::vector<size_t> const& send_counts_, environment env) {
        using dss_schimek::measurement::MeasuringTool;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        std::vector<int32_t> send_counts{send_counts_.cbegin(), send_counts_.cend()};
        std::vector<int32_t> recv_counts = alltoall(send_counts, env);

        std::vector<int32_t> send_displs(send_counts.size());
        std::exclusive_scan(send_counts.cbegin(), send_counts.cend(), send_displs.begin(), 0);

        std::vector<int32_t> recv_displs(send_counts.size());
        std::exclusive_scan(recv_counts.cbegin(), recv_counts.cend(), recv_displs.begin(), 0);

        auto recv_total = recv_counts.back() + recv_displs.back();
        std::vector<DataType> receive_data(recv_total);

        auto send_total = send_counts.back() + send_displs.back();
        measuringTool.addRawCommunication(send_total * sizeof(DataType), "alltoallv_small");

        data_type_mapper<DataType> dtm;
        MPI_Alltoallv(
            send_data,
            send_counts.data(),
            send_displs.data(),
            dtm.get_mpi_type(),
            receive_data.data(),
            recv_counts.data(),
            recv_displs.data(),
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
        std::vector<size_t> recv_counts = alltoall(send_counts, env);
        return alltoallv(send_data, send_counts, recv_counts, env);
    }

    template <typename DataType>
    static std::vector<DataType> alltoallv(
        DataType* const send_data,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& recv_counts,
        environment env
    ) {
        using dss_schimek::measurement::MeasuringTool;
        MeasuringTool& measuringTool = MeasuringTool::measuringTool();

        std::vector<size_t> send_displs(env.size());
        std::exclusive_scan(
            send_counts.cbegin(),
            send_counts.cend(),
            send_displs.begin(),
            size_t{0}
        );

        std::vector<size_t> recv_displs(env.size());
        std::exclusive_scan(
            recv_counts.cbegin(),
            recv_counts.cend(),
            recv_displs.begin(),
            size_t{0}
        );

        auto send_total = send_displs.back() + send_counts.back() - send_counts[env.rank()];
        measuringTool.addRawCommunication(send_total * sizeof(DataType), "alltoallv_direct");

        auto recv_total = recv_displs.back() + recv_counts.back();
        std::vector<DataType> receive_data(recv_total);
        std::vector<MPI_Request> mpi_request(2 * env.size());

        for (uint32_t i = 0; i < env.size(); ++i) {
            int32_t source = (env.rank() + (env.size() - i)) % env.size();
            auto receive_type = get_big_type<DataType>(recv_counts[source]);
            MPI_Irecv(
                receive_data.data() + recv_displs[source],
                1,
                receive_type,
                source,
                44227,
                env.communicator(),
                &mpi_request[source]
            );
        }
        for (uint32_t i = 0; i < env.size(); ++i) {
            int32_t target = (env.rank() + i) % env.size();
            auto send_type = get_big_type<DataType>(send_counts[target]);
            MPI_Isend(
                send_data + send_displs[target],
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
            return AllToAllvDirectMessages::alltoallv(send_data, send_counts, recv_counts, env);
        }
    }
};

} // namespace mpi
} // namespace dss_schimek


namespace dss_mehnert {
namespace mpi {
namespace _internal {

template <typename LcpIt>
inline void set_interval_start_lcp(LcpIt lcp_it, std::span<size_t const> interval_sizes) {
    for (auto const& interval: interval_sizes) {
        if (interval != 0) {
            *lcp_it = 0;
            std::advance(lcp_it, interval);
        }
    }
}

template <bool compress_prefixes, typename StringSet>
std::pair<unsigned char*, size_t>
write_interval(unsigned char* buffer, StringSet const& ss, size_t const* lcps) {
    auto dest = buffer;
    for (auto const& str: ss) {
        size_t str_depth;
        if constexpr (compress_prefixes) {
            str_depth = *lcps++;
        } else {
            str_depth = 0;
        }

        auto const str_len = ss.get_length(str);
        auto const actual_len = str_len - str_depth;

        // todo this is using std::copy everywhere else
        memcpy(dest, ss.get_chars(str, str_depth), actual_len);
        dest += actual_len;
        *dest++ = 0;
    }
    return {dest, static_cast<size_t>(dest - buffer)};
}

template <bool compress_prefixes, typename StringSet>
std::pair<unsigned char*, size_t> write_interval(
    unsigned char* buffer, StringSet const& ss, size_t const* lcps, size_t const* prefixes
) {
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
        dest += actual_len;
        *dest++ = 0;
    }
    return {dest, static_cast<size_t>(dest - buffer)};
}

template <bool compress_prefixes, typename StringPtr>
size_t get_num_send_chars(StringPtr const& strptr) {
    // todo maybe add back optimization for uncompressed string set
    auto const N = strptr.active().get_sum_length();
    if constexpr (compress_prefixes) {
        auto const lcp_begin = strptr.lcp();
        auto const lcp_end = lcp_begin + strptr.size();
        auto const L = std::accumulate(lcp_begin, lcp_end, size_t{0});
        return strptr.size() + N - L;
    } else {
        return strptr.size() + N;
    }
}

template <bool compress_prefixes, typename StringPtr>
size_t get_num_send_chars(StringPtr const& strptr, std::span<size_t const> prefixes) {
    auto const D = std::accumulate(prefixes.begin(), prefixes.end(), size_t{0});
    if constexpr (compress_prefixes) {
        auto const lcp_begin = strptr.lcp();
        auto const lcp_end = lcp_begin + strptr.size();
        auto const L = std::accumulate(lcp_begin, lcp_end, size_t{0});
        return strptr.size() + D - L;
    } else {
        return strptr.size() + D;
    }
}

template <bool compress_prefixes, typename StringPtr, typename... Prefixes>
std::pair<std::vector<unsigned char>, std::vector<size_t>> write_send_buf(
    StringPtr const& strptr, std::vector<size_t> const& send_counts, Prefixes const&... prefixes
) {
    set_interval_start_lcp(strptr.lcp(), send_counts);

    auto const num_send_chars = get_num_send_chars<compress_prefixes>(strptr, prefixes...);

    // todo this should ideally avoid zero initialization
    std::vector<unsigned char> send_buf_char(num_send_chars);
    std::vector<size_t> send_counts_char(send_counts.size());

    unsigned char* pos = send_buf_char.data();
    for (size_t i = 0, strs_written = 0; i != send_counts.size(); ++i) {
        auto const interval = strptr.sub(strs_written, send_counts[i]);

        size_t chars_written = 0;
        std::tie(pos, chars_written) = write_interval<compress_prefixes>(
            pos,
            interval.active(),
            interval.lcp(),
            (prefixes.data() + strs_written)...
        );
        send_counts_char[i] = chars_written;
        strs_written += interval.size();
    }
    return {std::move(send_buf_char), std::move(send_counts_char)};
}


template <bool use_compression, typename AllToAllPolicy>
inline std::vector<uint64_t> send_u64s(
    std::vector<uint64_t>& values,
    std::vector<uint64_t> const& send_counts,
    std::vector<uint64_t> const& recv_counts,
    dss_schimek::mpi::environment env
) {
    using namespace dss_schimek;
    if constexpr (use_compression) {
        auto compressedData =
            IntegerCompression::writeRanges(send_counts.begin(), send_counts.end(), values.data());

        auto recv_lcps =
            AllToAllPolicy::alltoallv(compressedData.integers.data(), compressedData.counts, env);

        // decode Lcp Compression:
        return IntegerCompression::readRanges(
            recv_counts.begin(),
            recv_counts.end(),
            recv_lcps.data()
        );
    } else {
        return AllToAllPolicy::alltoallv(values.data(), send_counts, env);
    }
}

template <bool compress_lcps, typename StringSet, typename AllToAllPolicy>
class StringSetSendImpl {
public:
    static void alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<unsigned char>& send_buf_char,
        std::vector<size_t> const& send_counts_char,
        std::vector<size_t> const& send_counts,
        dss_schimek::mpi::environment env
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_mpi");
        measuring_tool.start("all_to_all_strings_send_chars");
        auto recv_buf_char = AllToAllPolicy::alltoallv(send_buf_char.data(), send_counts_char, env);
        send_buf_char.clear();
        send_buf_char.shrink_to_fit();

        auto recv_counts = dss_schimek::mpi::alltoall(send_counts, env);
        measuring_tool.stop("all_to_all_strings_send_chars");

        measuring_tool.start("all_to_all_strings_send_lcps");
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_send_lcps");

        measuring_tool.start("all_to_all_strings_send_idxs");
        measuring_tool.stop("all_to_all_strings_send_idxs");

        container.delete_all();
        measuring_tool.stop("all_to_all_strings_mpi");

        measuring_tool.start("all_to_all_strings_init_container");
        container.update(std::move(recv_buf_char), std::move(recv_buf_lcp));
        measuring_tool.stop("all_to_all_strings_init_container");
    }
};

template <bool compress_lcps, typename StringSet, typename AllToAllPolicy>
    requires(has_permutation_members<StringSet>)
class StringSetSendImpl<compress_lcps, StringSet, AllToAllPolicy> {
public:
    static void alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<unsigned char>& send_buf_char,
        std::vector<size_t> const& send_counts_char,
        std::vector<size_t> const& send_counts,
        dss_schimek::mpi::environment env
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        auto const ss = container.make_string_set();

        measuring_tool.start("all_to_all_strings_mpi");
        measuring_tool.start("all_to_all_strings_send_chars");
        auto recv_buf_char = AllToAllPolicy::alltoallv(send_buf_char.data(), send_counts_char, env);
        send_buf_char.clear();
        send_buf_char.shrink_to_fit();

        auto recv_counts = dss_schimek::mpi::alltoall(send_counts, env);
        measuring_tool.stop("all_to_all_strings_send_chars");


        // todo it would be possible to combine all three send_u64 calls ...
        measuring_tool.start("all_to_all_strings_send_lcps");
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_send_lcps");
        container.delete_lcps();

        // todo does 7-bit compression make sense for PE and string indices
        measuring_tool.start("all_to_all_strings_send_idxs");
        InputPermutation permutation{ss};
        container.delete_all();

        auto const send_idxs = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_rank = send_idxs(permutation.ranks(), send_counts, recv_counts, env);
        auto recv_buf_index = send_idxs(permutation.strings(), send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_send_idxs");
        measuring_tool.stop("all_to_all_strings_mpi");

        measuring_tool.start("all_to_all_strings_init_container");
        container.update(
            std::move(recv_buf_char),
            std::move(recv_buf_lcp),
            make_initializer<StringIndex>(std::move(recv_buf_index)),
            make_initializer<PEIndex>(std::move(recv_buf_rank))
        );
        measuring_tool.stop("all_to_all_strings_init_container");
    }
};

} // namespace _internal

template <bool compress_lcps, bool compress_prefixes, typename AllToAllPolicy>
class AllToAllStringImpl {
public:
    static constexpr bool PrefixCompression = compress_prefixes;

    template <typename StringSet>
    void alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_write_send_buf");
        auto const strptr = container.make_string_lcp_ptr();
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<compress_prefixes>(strptr, send_counts);
        container.delete_raw_strings();
        measuring_tool.stop("all_to_all_strings_write_send_buf");

        using SendImpl = _internal::StringSetSendImpl<compress_lcps, StringSet, AllToAllPolicy>;
        SendImpl::alltoallv(container, send_buf_char, send_counts_char, send_counts, comm);
    }

    template <typename StringSet>
    void alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        std::span<size_t const> prefixes,
        Communicator const& comm
    )
        requires(PermutationStringSet<StringSet>)
    {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_write_send_buf");
        auto const strptr = container.make_string_lcp_ptr();
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<compress_prefixes>(strptr, send_counts, prefixes);
        container.delete_raw_strings();
        measuring_tool.stop("all_to_all_strings_write_send_buf");

        using SendImpl = _internal::StringSetSendImpl<compress_lcps, StringSet, AllToAllPolicy>;
        SendImpl::alltoallv(container, send_buf_char, send_counts_char, send_counts, comm);
    }
};

} // namespace mpi
} // namespace dss_mehnert
