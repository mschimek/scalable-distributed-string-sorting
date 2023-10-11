// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/named_parameters.hpp>

#include "mpi/alltoall_combined.hpp"
#include "mpi/byte_encoder.hpp"
#include "mpi/plugin_helpers.hpp"
#include "sorter/distributed/permutation.hpp"
#include "strings/stringcontainer.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace mpi {

struct AlltoallStringsConfig {
    AlltoallvCombinedKind const alltoall_kind;
    bool const compress_lcps;
    bool const compress_prefixes;
};

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

        auto const str_len = *prefixes++;
        auto const actual_len = str_len - str_depth;
        memcpy(dest, ss.get_chars(str, str_depth), actual_len);
        dest += actual_len;
        *dest++ = 0;
    }
    return {dest, static_cast<size_t>(dest - buffer)};
}

template <bool compress_prefixes, typename StringPtr>
size_t get_num_send_chars(StringPtr const& strptr) {
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
    std::vector<typename StringPtr::StringSet::Char> send_buf_char(num_send_chars);
    std::vector<size_t> send_counts_char(send_counts.size());

    unsigned char* pos = send_buf_char.data();
    for (size_t i = 0, strs_written = 0; i != send_counts.size(); ++i) {
        auto const interval = strptr.sub(strs_written, send_counts[i]);

        std::tie(pos, send_counts_char[i]) = write_interval<compress_prefixes>(
            pos,
            interval.active(),
            interval.lcp(),
            (prefixes.data() + strs_written)...
        );
        strs_written += interval.size();
    }
    return {std::move(send_buf_char), std::move(send_counts_char)};
}


template <bool use_compression, typename Communicator>
inline std::vector<size_t> send_integers(
    std::vector<size_t>& values,
    std::vector<size_t> const& send_counts,
    std::vector<size_t> const& recv_counts,
    Communicator const& comm
) {
    constexpr auto alltoall_kind = AlltoallvCombinedKind::native;
    if constexpr (use_compression) {
        auto compressed_data = dss_schimek::IntegerCompression::writeRanges(
            send_counts.begin(),
            send_counts.end(),
            values.data()
        );
        auto recv_data = comm.template alltoallv_combined<alltoall_kind>(
            compressed_data.integers,
            compressed_data.counts
        );
        return dss_schimek::IntegerCompression::readRanges(
            recv_counts.begin(),
            recv_counts.end(),
            recv_data.begin()
        );
    } else {
        return comm.template alltoallv_combined<alltoall_kind>(values, send_counts, recv_counts);
    }
}

template <typename Permutation>
class PermutationSendImpl {};

template <>
class PermutationSendImpl<InputPermutation> {
public:
    template <AlltoallStringsConfig config, typename StringSet, typename Communicator>
    static void send(
        StringLcpContainer<StringSet>& container,
        std::vector<typename StringSet::Char>& recv_buf_char,
        std::vector<size_t>& recv_buf_lcp,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& recv_counts,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        // todo does 7-bit compression make sense for PE and string indices
        measuring_tool.start("all_to_all_strings_send_idxs");
        InputPermutation permutation{container.make_string_set()};
        container.delete_all();

        auto const send_idxs = _internal::send_integers<config.compress_lcps, Communicator>;
        auto recv_buf_rank = send_idxs(permutation.ranks(), send_counts, recv_counts, comm);
        auto recv_buf_index = send_idxs(permutation.strings(), send_counts, recv_counts, comm);
        measuring_tool.stop("all_to_all_strings_send_idxs");

        measuring_tool.start("all_to_all_strings_init_container");
        container.update(
            std::move(recv_buf_char),
            std::move(recv_buf_lcp),
            make_initializer<StringIndex>(recv_buf_index),
            make_initializer<PEIndex>(recv_buf_rank)
        );
        measuring_tool.stop("all_to_all_strings_init_container");
    }
};

template <>
class PermutationSendImpl<MultiLevelPermutation> {
public:
    template <AlltoallStringsConfig config, typename StringSet, typename Communicator>
    static void send(
        StringLcpContainer<StringSet>& container,
        std::vector<typename StringSet::Char>& recv_buf_char,
        std::vector<size_t>& recv_buf_lcp,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& recv_counts,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        // todo does 7-bit compression make sense for PE and string indices
        measuring_tool.start("all_to_all_strings_send_idxs");
        InputPermutation permutation{container.make_string_set()};
        container.delete_all();


        measuring_tool.stop("all_to_all_strings_send_idxs");

        measuring_tool.start("all_to_all_strings_init_container");
        container.update(std::move(recv_buf_char), std::move(recv_buf_lcp));

        // set PEIndex to indicate rank of origin PE
        auto str = container.get_strings().begin();
        for (size_t rank = 0; rank != recv_counts.size(); ++rank) {
            for (size_t i = 0; i != recv_counts[rank]; ++i) {
                (*str++).PEIndex = rank;
            }
        }
        measuring_tool.stop("all_to_all_strings_init_container");
    }
};

template <typename StringSet, typename Permutation>
class StringSetSendImpl {
    static_assert(std::is_same_v<Permutation, NoPermutation>);

public:
    template <AlltoallStringsConfig config, typename Communicator>
    static void alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<typename StringSet::Char>& send_buf_char,
        std::vector<size_t> const& send_counts_char,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& recv_counts,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_send_chars");
        auto recv_buf_char =
            comm.template alltoallv_combined<config.alltoall_kind>(send_buf_char, send_counts_char);
        send_buf_char.clear();
        send_buf_char.shrink_to_fit();
        measuring_tool.stop("all_to_all_strings_send_chars");

        measuring_tool.start("all_to_all_strings_send_lcps");
        auto send_lcps = _internal::send_integers<config.compress_lcps, Communicator>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, comm);
        measuring_tool.stop("all_to_all_strings_send_lcps");

        measuring_tool.start("all_to_all_strings_send_idxs");
        measuring_tool.stop("all_to_all_strings_send_idxs");

        measuring_tool.start("all_to_all_strings_init_container");
        container.update(std::move(recv_buf_char), std::move(recv_buf_lcp));
        measuring_tool.stop("all_to_all_strings_init_container");
    }
};

template <PermutationStringSet StringSet, typename Permutation>
class StringSetSendImpl<StringSet, Permutation> {
    static_assert(!std::is_same_v<Permutation, NoPermutation>);

public:
    template <AlltoallStringsConfig config, typename Communicator>
    static void alltoallv(
        StringLcpContainer<StringSet>& container,
        std::vector<typename StringSet::Char>& send_buf_char,
        std::vector<size_t> const& send_counts_char,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& recv_counts,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_send_chars");
        auto recv_buf_char =
            comm.template alltoallv_combined<config.alltoall_kind>(send_buf_char, send_counts_char);
        send_buf_char.clear();
        send_buf_char.shrink_to_fit();
        measuring_tool.stop("all_to_all_strings_send_chars");

        // todo it would be possible to combine all three send_u64 calls ...
        measuring_tool.start("all_to_all_strings_send_lcps");
        auto const send_lcps = _internal::send_integers<config.compress_lcps, Communicator>;
        auto recv_buf_lcp = send_lcps(container.lcps(), send_counts, recv_counts, comm);
        container.delete_lcps();
        measuring_tool.stop("all_to_all_strings_send_lcps");

        using SendImpl = PermutationSendImpl<Permutation>;
        constexpr auto send = SendImpl::template send<config, StringSet, Communicator>;
        send(container, recv_buf_char, recv_buf_lcp, send_counts, recv_counts, comm);
    }
};

} // namespace _internal

template <typename Comm>
class AlltoallStringsPlugin : public kamping::plugins::PluginBase<Comm, AlltoallStringsPlugin> {
public:
    template <AlltoallStringsConfig config, typename Permutation, typename StringSet>
    void alltoall_strings(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& recv_counts
    ) const {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_write_send_buf");
        auto const strptr = container.make_string_lcp_ptr();
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<config.compress_prefixes>(strptr, send_counts);
        container.delete_raw_strings();
        measuring_tool.stop("all_to_all_strings_write_send_buf");

        measuring_tool.start("all_to_all_strings_alltoallv");
        auto const& comm = this->to_communicator();
        using SendImpl = _internal::StringSetSendImpl<StringSet, Permutation>;
        constexpr auto alltoallv = SendImpl::template alltoallv<config, Comm>;
        alltoallv(container, send_buf_char, send_counts_char, send_counts, recv_counts, comm);
        measuring_tool.stop("all_to_all_strings_alltoallv");
    }

    template <AlltoallStringsConfig config, typename Permutation, typename StringSet>
    void alltoall_strings(
        StringLcpContainer<StringSet>& container,
        std::vector<size_t> const& send_counts,
        std::vector<size_t> const& recv_counts,
        std::span<size_t const> prefixes
    ) const {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("all_to_all_strings_write_send_buf");
        auto const strptr = container.make_string_lcp_ptr();
        auto [send_buf_char, send_counts_char] =
            _internal::write_send_buf<config.compress_prefixes>(strptr, send_counts, prefixes);
        container.delete_raw_strings();
        measuring_tool.stop("all_to_all_strings_write_send_buf");

        measuring_tool.start("all_to_all_strings_alltoallv");
        auto const& comm = this->to_communicator();
        using SendImpl = _internal::StringSetSendImpl<StringSet, Permutation>;
        constexpr auto alltoallv = SendImpl::template alltoallv<config, Comm>;
        alltoallv(container, send_buf_char, send_counts_char, send_counts, recv_counts, comm);
        measuring_tool.stop("all_to_all_strings_alltoallv");
    }
};

} // namespace mpi
} // namespace dss_mehnert
