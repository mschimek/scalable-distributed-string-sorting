// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "mpi/shift.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"

namespace dss_schimek {

template <typename StringSet>
std::vector<unsigned char> make_contiguous(StringSet const& ss) {
    std::vector<unsigned char> raw_strings;
    for (auto const& str: ss) {
        auto chars = ss.get_chars(str, 0);
        auto length = ss.get_length(str);
        std::copy_n(chars, length + 1, std::back_inserter(raw_strings));
    }
    return raw_strings;
}

template <typename StringPtr>
class CheckerWithCompleteExchange {
    using StringSet = typename StringPtr::StringSet;

public:
    void storeLocalInput(std::vector<unsigned char> const& localInputRawStrings) {
        local_input_chars_ = localInputRawStrings;
    }

    std::vector<unsigned char> local_input() { return local_input_chars_; }

    bool checkLcp(
        std::span<size_t const> input_lcps,
        std::span<size_t const> sorted_lcps,
        dss_mehnert::Communicator const& comm
    ) {
        size_t counter = 0u;
        for (size_t i = 0; i != input_lcps.size(); ++i) {
            if (input_lcps[i] == sorted_lcps[i]) {
                counter++;
            }
        }
        return sorted_lcps.size() == input_lcps.size()
               && counter + comm.size() >= sorted_lcps.size();
    }

    bool check(
        StringPtr const& sorted_strings, bool const check_lcp, dss_mehnert::Communicator const& comm
    ) {
        using namespace kamping;

        auto const& ss = sorted_strings.active();
        std::vector<unsigned char> global_input_chars, global_sorted_chars;
        comm.gatherv(send_buf(make_contiguous(ss)), recv_buf(global_sorted_chars));
        comm.gatherv(send_buf(local_input_chars_), recv_buf(global_input_chars));

        std::vector<size_t> sorted_lcps;
        if (check_lcp) {
            std::span const local_lcps{sorted_strings.lcp(), sorted_strings.size()};
            comm.gatherv(send_buf(local_lcps), recv_buf(sorted_lcps));
        }

        if (comm.is_root()) {
            StringLcpContainer<StringSet> container(std::move(global_input_chars));
            auto const strptr = container.make_string_lcp_ptr();
            tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
            container.make_contiguous();

            bool const lcps_correct = !check_lcp || checkLcp(container.lcps(), sorted_lcps, comm);
            bool const sorted_correctly = global_sorted_chars == container.raw_strings();
            bool const overallCorrect = lcps_correct && sorted_correctly;

            return comm.allreduce_single(send_buf({overallCorrect}), op(ops::logical_and<>{}));
        } else {
            return comm.allreduce_single(send_buf({true}), op(ops::logical_and<>{}));
        }
    }

private:
    std::vector<unsigned char> local_input_chars_;
};

template <typename StringSet>
bool is_sorted(StringSet const& ss, dss_mehnert::Communicator const& comm) {
    using namespace kamping;

    if (comm.allreduce_single(send_buf<size_t>(ss.empty()), op(std::plus<>{})) <= 1) {
        return ss.check_order();
    }

    auto const own_min_number = ss.empty() ? comm.size() : comm.rank();
    auto const own_max_number = ss.empty() ? -1 : comm.rank();
    auto const min_PE_with_data = comm.allreduce_single(send_buf(own_min_number), op(ops::min<>{}));
    auto const max_PE_with_data = comm.allreduce_single(send_buf(own_max_number), op(ops::max<>{}));

    auto const front = (ss.empty() ? ss.empty_string() : *ss.begin()).string;
    auto const back = (ss.empty() ? ss.empty_string() : *(ss.end() - 1)).string;
    auto const greater_string = dss_schimek::mpi::shift_string<true>(front, !ss.empty(), comm);
    auto const smaller_string = dss_schimek::mpi::shift_string<false>(back, !ss.empty(), comm);

    bool is_sorted = ss.check_order();
    if (!ss.empty()) {
        if (comm.rank() != min_PE_with_data) {
            is_sorted &= dss_schimek::scmp(smaller_string.data(), front) <= 0;
        }
        if (comm.rank() != max_PE_with_data) {
            is_sorted &= dss_schimek::scmp(back, greater_string.data()) <= 0;
        }
    }
    return comm.allreduce_single(send_buf({is_sorted}), op(ops::logical_and<>{}));
}

template <typename StringPtr>
bool is_complete_and_sorted(
    StringPtr const& strptr,
    size_t const initial_local_num_chars,
    size_t const current_local_num_chars,
    size_t const initial_local_num_strings,
    size_t const current_local_num_strings,
    dss_mehnert::Communicator const& comm
) {
    auto const allreduce_sum = [&comm](size_t const x) {
        return comm.allreduce_single(kamping::send_buf({x}), kamping::op(std::plus<>{}));
    };

    auto const initial_total_num_chars = allreduce_sum(initial_local_num_chars);
    auto const initial_total_num_strings = allreduce_sum(initial_local_num_strings);
    auto const current_total_num_chars = allreduce_sum(current_local_num_chars);
    auto const current_total_num_strings = allreduce_sum(current_local_num_strings);

    if (initial_total_num_chars != current_total_num_chars) {
        std::cout << "initial total num chars: " << initial_total_num_chars
                  << " current_total_num_chars: " << current_total_num_chars << std::endl;
        std::cout << "We've lost some chars" << std::endl;
        return false;
    }
    if (initial_total_num_strings != current_total_num_strings) {
        std::cout << "initial total num strings: " << initial_total_num_strings
                  << " current total num strings: " << current_total_num_strings << std::endl;
        std::cout << "We've lost some strings" << std::endl;
        return false;
    }
    return is_sorted(strptr.active(), comm);
}
} // namespace dss_schimek
