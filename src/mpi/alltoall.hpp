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

namespace dss_schimek {
namespace mpi {

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


namespace _internal {

template <typename LcpIter, typename IntervalIter>
inline void set_interval_start_lcp(LcpIter lcp_it, IntervalIter begin, IntervalIter end) {
    for (auto it = begin; it != end; ++it) {
        if (*it != 0) {
            *lcp_it = 0;
            std::advance(lcp_it, *it);
        }
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
    return {dest, static_cast<size_t>(dest - buffer)};
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
        dest += actual_len;
        *dest++ = 0;
    }
    return {dest, static_cast<size_t>(dest - buffer)};
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
    set_interval_start_lcp(lcps.begin(), send_counts.begin(), send_counts.end());

    auto num_send_chars = get_num_send_chars<compress_prefixes>(container, prefixes...);
    std::vector<unsigned char> send_buf_char(num_send_chars);
    std::vector<size_t> send_counts_char(send_counts.size());

    auto ss = container.make_string_set();
    unsigned char* pos = send_buf_char.data();

    for (size_t interval = 0, strs_written = 0; interval < send_counts.size(); ++interval) {
        auto begin = ss.begin() + strs_written;

        size_t chars_written = 0;
        std::tie(pos, chars_written) = write_interval<compress_prefixes>(
            pos,
            ss.sub(begin, begin + send_counts[interval]),
            lcps.data() + strs_written,
            (prefixes.data() + strs_written)...
        );

        send_counts_char[interval] = chars_written;
        strs_written += send_counts[interval];
    }

    return {std::move(send_buf_char), std::move(send_counts_char)};
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

        measuring_tool.start("all_to_all_strings_send_lcps");
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_send_lcps");

        measuring_tool.start("all_to_all_strings_send_idxs");
        measuring_tool.stop("all_to_all_strings_send_idxs");

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
        auto get_PE_idx = [&ss](auto const& str) { return str.getPEIndex(); };
        std::transform(ss.begin(), ss.end(), send_buf_PE_idx.begin(), get_PE_idx);

        std::vector<size_t> send_buf_str_idx(ss.size());
        auto get_str_idx = [&ss](auto const& str) { return str.getStringIndex(); };
        std::transform(ss.begin(), ss.end(), send_buf_str_idx.begin(), get_str_idx);

        measuring_tool.start("all_to_all_strings_mpi");
        auto recv_buf_char = AllToAllPolicy::alltoallv(send_buf_char.data(), send_counts_char, env);
        auto recv_counts = mpi::alltoall(send_counts, env);

        measuring_tool.start("all_to_all_strings_send_lcps");
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_send_lcps");
        container.deleteAll();

        // todo does 7-bit compression make sense for PE and string indices
        // todo this is pretty wasteful in terms of memory
        measuring_tool.start("all_to_all_strings_send_idxs");
        auto send_idxs = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_PE_idx = send_idxs(send_buf_PE_idx, send_counts, recv_counts, env);
        auto recv_buf_str_idx = send_idxs(send_buf_str_idx, send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_send_idxs");
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
template <bool compress_lcps, bool compress_prefixes, typename StringSet, typename AllToAllPolicy>
struct AllToAllStringImplPrefixDoubling : private AllToAllStringImpl<
                                              compress_lcps,
                                              compress_prefixes,
                                              UCharLengthIndexPEIndexStringSet,
                                              AllToAllPolicy> {
    using StringPEIndexSet = UCharLengthIndexPEIndexStringSet;
    static constexpr bool Lcps = true;
    static constexpr bool PrefixCompression = compress_prefixes;

    using AllToAllStringImpl<compress_lcps, compress_prefixes, StringPEIndexSet, AllToAllPolicy>::
        alltoallv;

    StringLcpContainer<StringPEIndexSet> alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& prefixes,
        environment env
    ) {
        using dss_mehnert::measurement::MeasuringTool;
        auto& measuring_tool = MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_intern_copy");
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<compress_prefixes>(container, send_counts, prefixes);
        container.deleteRawStrings();
        measuring_tool.stop("all_to_all_strings_intern_copy");


        measuring_tool.start("all_to_all_strings_mpi");
        auto recv_buf_char = AllToAllPolicy::alltoallv(send_buf_char.data(), send_counts_char, env);

        auto recv_counts = mpi::alltoall(send_counts, env);

        measuring_tool.start("all_to_all_strings_send_lcps");
        auto send_lcps = _internal::send_u64s<compress_lcps, AllToAllPolicy>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, env);
        measuring_tool.stop("all_to_all_strings_send_lcps");
        container.deleteAll();

        measuring_tool.start("all_to_all_strings_send_idxs");
        measuring_tool.stop("all_to_all_strings_send_idxs");

        std::vector<size_t> offsets(send_counts.size());
        std::exclusive_scan(send_counts.begin(), send_counts.end(), offsets.begin(), size_t{0});

        auto recv_offsets = mpi::alltoall(offsets, env);
        measuring_tool.stop("all_to_all_strings_mpi");

        measuring_tool.start("container_construction");
        StringLcpContainer<StringPEIndexSet> recv_container{
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

        measuring_tool.start("all_to_all_strings_intern_copy");
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<compress_prefixes>(container, send_counts, prefixes);
        container.deleteRawStrings();
        measuring_tool.stop("all_to_all_strings_intern_copy");

        using SendImpl =
            _internal::StringSetSendImpl<compress_lcps, StringPEIndexSet, AllToAllPolicy>;
        return SendImpl::alltoallv(container, send_buf_char, send_counts_char, send_counts, env);
    }
};

} // namespace mpi
} // namespace dss_schimek
