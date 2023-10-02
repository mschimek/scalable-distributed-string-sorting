// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/communicator.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/recv.hpp>
#include <tlx/math.hpp>
#include <tlx/math/div_ceil.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "mpi/shift.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"
#include "strings/stringtools.hpp"

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

    int const own_min_number = ss.empty() ? comm.size_signed() : comm.rank_signed();
    int const own_max_number = ss.empty() ? -1 : comm.rank_signed();
    int const min_PE_with_data = comm.allreduce_single(send_buf(own_min_number), op(ops::min<>{}));
    int const max_PE_with_data = comm.allreduce_single(send_buf(own_max_number), op(ops::max<>{}));

    auto const front = (ss.empty() ? ss.empty_string() : *ss.begin()).string;
    auto const back = (ss.empty() ? ss.empty_string() : *(ss.end() - 1)).string;
    auto const greater_string = dss_schimek::mpi::shift_string<true>(front, !ss.empty(), comm);
    auto const smaller_string = dss_schimek::mpi::shift_string<false>(back, !ss.empty(), comm);

    bool is_sorted = ss.check_order();
    if (!ss.empty()) {
        if (comm.rank_signed() != min_PE_with_data) {
            is_sorted &= dss_schimek::scmp(smaller_string.data(), front) <= 0;
        }
        if (comm.rank_signed() != max_PE_with_data) {
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

namespace dss_mehnert {

template <typename Char>
bool check_global_order(
    std::vector<Char> const& first_string,
    std::vector<Char> const& last_string,
    bool const is_empty,
    Communicator const& comm
) {
    using namespace kamping;

    assert_equal(first_string.back(), 0);
    assert_equal(last_string.back(), 0);

    auto const comm_reverse = comm.split(0, comm.size() - comm.rank() - 1);
    auto const predecessor = comm.exscan_single(
        send_buf(is_empty ? -1 : comm.rank_signed()),
        op(ops::max<>{}),
        values_on_rank_0(-1)
    );
    auto const successor = comm_reverse.exscan_single(
        send_buf(is_empty ? comm.size_signed() : comm.rank_signed()),
        op(ops::min<>{}),
        values_on_rank_0(comm.size_signed())
    );

    if (is_empty) {
        Request req_pred, req_succ;
        std::vector<Char> pred_string, succ_string;

        if (predecessor != -1) {
            comm.isend(send_buf(first_string), destination(predecessor), request(req_pred));
        }
        if (successor != comm.size_signed()) {
            comm.issend(send_buf(last_string), destination(successor), request(req_succ));
        }

        if (predecessor != -1) {
            comm.recv(recv_buf(pred_string), source(predecessor));
        }
        if (successor != comm.size_signed()) {
            comm.recv(recv_buf(succ_string), source(successor));
        }
        req_pred.wait();
        req_succ.wait();

        auto leq = [](auto const& lhs, auto const& rhs) {
            return dss_schimek::leq(lhs.data(), rhs.data());
        };
        return (predecessor == -1 || leq(pred_string, first_string))
               && (successor == comm.size_signed() || leq(last_string, succ_string));
    } else {
        return true;
    }
}

inline bool check_permutation_complete(
    InputPermutation const& permutation, size_t const local_size, Communicator const& comm
) {
    std::vector<int> counts(comm.size());
    for (auto const& rank: permutation.ranks()) {
        ++counts[rank];
    }

    std::vector<int> offsets(counts.size());
    std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0);

    std::vector<size_t> sorted_indices(permutation.size());
    for (size_t i = 0; i != permutation.size(); ++i) {
        sorted_indices[offsets[permutation.rank(i)]++] = permutation.string(i);
    }

    using namespace kamping;
    std::vector<size_t> global_indicies;
    comm.alltoallv(send_buf(sorted_indices), send_counts(counts), recv_buf(global_indicies));
    std::sort(global_indicies.begin(), global_indicies.end());

    bool is_complete = true;
    for (size_t i = 0; i != global_indicies.size(); ++i) {
        is_complete &= (i == global_indicies[i]);
    }
    return local_size == global_indicies.size() && is_complete;
}

template <typename StringSet>
class SpaceEfficientChecker {
public:
    using StringContainer = StringLcpContainer<StringSet>;
    using Char = StringContainer::Char;

    void store_container(StringContainer const& container) {
        assert(container.is_consistent());
        input_container_.set(std::vector{container.get_strings()});
        input_container_.set(std::vector{container.raw_strings()});
        input_container_.set(std::vector<size_t>(container.size()));

        auto const d_begin = input_container_.raw_strings().data();
        auto const begin = container.raw_strings().data();

        auto d_str = input_container_.strings();
        for (auto const& str: container.get_strings()) {
            (d_str++)->string = d_begin + (str.string - begin);
        }
        assert(input_container_.is_consistent());
    }

    bool is_sorted(InputPermutation const& permutation, Communicator const& comm) {
        using namespace kamping;

        auto const num_quantiles = compute_num_quantiles(comm);
        auto const quantile_size = tlx::div_ceil(permutation.size(), num_quantiles);
        assert(num_quantiles * quantile_size >= permutation.size());

        bool is_sorted = true;
        auto const &ranks = permutation.ranks(), &strings = permutation.strings();

        std::vector<Char> lower_bound{0};
        for (size_t i = 0; i != num_quantiles; ++i) {
            std::cout << "quantile" << i << std::endl;
            auto const begin = std::min(permutation.size(), i * quantile_size);
            auto const end = std::min(begin + quantile_size, permutation.size());

            InputPermutation const quantile_permutation{
                {ranks.begin() + begin, ranks.begin() + end},
                {strings.begin() + begin, strings.begin() + end}};

            auto quantile = dss_mehnert::sorter::apply_permutation(
                input_container_.make_string_set(),
                quantile_permutation,
                comm
            );

            is_sorted &= quantile.make_string_set().check_order();
            if (!quantile.empty()) {
                // note that strings are null-terminated at this point
                is_sorted &= dss_schimek::leq(lower_bound.data(), quantile.front().string);
                lower_bound = quantile.get_raw_string(quantile.size() - 1);
            }
        }


        std::vector<Char> first_string{0}, last_string{0};
        if (!permutation.empty()) {
            InputPermutation const sentinels{
                {ranks.front(), ranks.back()},
                {strings.front(), strings.back()}};

            auto sentinel_strings = dss_mehnert::sorter::apply_permutation(
                input_container_.make_string_set(),
                sentinels,
                comm
            );

            first_string = sentinel_strings.get_raw_string(0);
            last_string = sentinel_strings.get_raw_string(1);
        } else {
            InputPermutation const sentinels;
            dss_mehnert::sorter::apply_permutation(
                input_container_.make_string_set(),
                InputPermutation{},
                comm
            );
        }

        is_sorted &= check_global_order(first_string, last_string, permutation.empty(), comm);
        return comm.allreduce_single(send_buf({is_sorted}), op(ops::logical_and<>{}));
    }

    bool is_complete(InputPermutation const& permutation, Communicator const& comm) {
        using namespace kamping;
        auto is_complete = check_permutation_complete(permutation, input_container_.size(), comm);
        return comm.allreduce_single(send_buf({is_complete}), op(ops::logical_and<>{}));
    }

private:
    static constexpr size_t quantile_size = 500 * 1024 * 1024;
    StringContainer input_container_;

    size_t compute_num_quantiles(Communicator const& comm) {
        using namespace kamping;

        size_t const total_size = input_container_.make_string_set().get_sum_length();
        size_t const local_num_quantiles = tlx::div_ceil(total_size, quantile_size + 1);
        return comm.allreduce_single(send_buf(local_num_quantiles), op(ops::max<>{}));
    }
};

} // namespace dss_mehnert
