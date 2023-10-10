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
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringtools.hpp"

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

    if (!is_empty) {
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
        req_pred.wait(), req_succ.wait();

        auto leq = [](auto const& lhs, auto const& rhs) {
            return dss_schimek::leq(lhs.data(), rhs.data());
        };

        return (predecessor == -1 || leq(pred_string, first_string))
               && (successor == comm.size_signed() || leq(last_string, succ_string));
    } else {
        return true;
    }
}

template <typename StringSet>
bool check_permutation_global_order(
    StringSet const& ss, InputPermutation const& permutation, Communicator const& comm
) {
    std::vector<typename StringSet::Char> first_string{0}, last_string{0};
    if (!permutation.empty()) {
        InputPermutation const sentinels{
            {permutation.ranks().front(), permutation.ranks().back()},
            {permutation.strings().front(), permutation.strings().back()}};

        auto sentinel_strings = dss_mehnert::sorter::apply_permutation(ss, sentinels, comm);
        first_string = sentinel_strings.get_raw_string(0);
        last_string = sentinel_strings.get_raw_string(1);
    } else {
        dss_mehnert::sorter::apply_permutation(ss, InputPermutation{}, comm);
    }
    return check_global_order(first_string, last_string, permutation.empty(), comm);
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

// todo this should maybe be a function of StringContainer
template <typename StringSet>
void copy_container(StringLcpContainer<StringSet> const& src, StringLcpContainer<StringSet>& dst) {
    assert(src.is_consistent());
    dst.set(std::vector{src.get_strings()});
    dst.set(std::vector{src.raw_strings()});
    dst.set(std::vector<size_t>(src.size()));

    auto const d_begin = dst.raw_strings().data();
    auto const s_begin = src.raw_strings().data();

    auto d_str = dst.strings();
    for (auto const& s_str: src.get_strings()) {
        (d_str++)->string = d_begin + (s_str.string - s_begin);
    }
    assert(dst.is_consistent());
}

template <typename StringSet>
class MergeSortChecker {
public:
    using StringContainer = StringLcpContainer<StringSet>;
    using Char = StringContainer::Char;

    void store_container(StringContainer const& container) {
        copy_container(container, input_container_);
    }

    bool is_sorted(StringSet const& ss, Communicator const& comm) {
        using namespace kamping;

        bool is_sorted = true;
        if (!ss.check_order()) {
            std::cout << "strings are not sorted locally\n";
            is_sorted = false;
        }

        std::vector<Char> first_string{0}, last_string{0};
        if (!ss.empty()) {
            auto const &fst = *ss.begin(), lst = *(ss.end() - 1);
            auto const fst_length = ss.get_length(fst), lst_length = ss.get_length(lst);
            first_string.insert(first_string.end(), fst.string, fst.string + fst_length);
            last_string.insert(last_string.end(), lst.string, lst.string + lst_length);
            first_string.push_back(0), last_string.push_back(0);
        }
        if (!check_global_order(first_string, last_string, ss.empty(), comm)) {
            std::cout << "strings are not sorted globally\n";
            is_sorted = false;
        }
        return comm.allreduce_single(send_buf({is_sorted}), op(ops::logical_and<>{}));
    }

    bool is_complete(StringContainer const& sorted_container, Communicator const& comm) {
        auto const allreduce_sum = [&comm](size_t const x) {
            return comm.allreduce_single(kamping::send_buf({x}), kamping::op(std::plus<>{}));
        };

        auto const initial_num_chars = allreduce_sum(input_container_.char_size());
        auto const initial_num_strings = allreduce_sum(input_container_.size());
        auto const sorted_num_chars = allreduce_sum(sorted_container.char_size());
        auto const sorted_num_strings = allreduce_sum(sorted_container.size());
        return initial_num_chars == sorted_num_chars && initial_num_strings == sorted_num_strings;
    }

    bool check_lcps(
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

    bool
    check_exhaustive(StringContainer& sorted_container, dss_mehnert::Communicator const& comm) {
        using namespace kamping;

        sorted_container.make_contiguous();
        input_container_.make_contiguous();

        std::vector<unsigned char> global_input_chars, global_sorted_chars;
        comm.gatherv(send_buf(sorted_container.raw_strings()), recv_buf(global_sorted_chars));
        comm.gatherv(send_buf(input_container_.raw_strings()), recv_buf(global_input_chars));

        std::vector<size_t> sorted_lcps;
        comm.gatherv(send_buf(sorted_container.lcps()), recv_buf(sorted_lcps));

        bool overall_correct = true;
        if (comm.is_root()) {
            StringLcpContainer<StringSet> container(std::move(global_input_chars));
            auto const strptr = container.make_string_lcp_ptr();
            tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
            container.make_contiguous();

            bool const lcps_correct = check_lcps(container.lcps(), sorted_lcps, comm);
            bool const sorted_correctly = global_sorted_chars == container.raw_strings();
            overall_correct = lcps_correct && sorted_correctly;
        }
        return comm.allreduce_single(send_buf({overall_correct}), op(ops::logical_and<>{}));
    }

private:
    StringContainer input_container_;
};

template <typename StringSet>
class PrefixDoublingChecker {
public:
    using StringContainer = StringLcpContainer<StringSet>;
    using Char = StringContainer::Char;

    void store_container(StringContainer const& container) {
        copy_container(container, input_container_);
    }

    bool is_sorted(InputPermutation const& permutation, Communicator const& comm) {
        using namespace kamping;

        auto const ss = input_container_.make_string_set();

        bool is_sorted = true;
        auto sorted_strings = dss_mehnert::sorter::apply_permutation(ss, permutation, comm);
        if (!sorted_strings.make_string_set().check_order()) {
            sorted_strings.make_string_set().print();
            std::cout << "the permutation is not sorted locally\n";
            is_sorted = false;
        }

        if (!check_permutation_global_order(ss, permutation, comm)) {
            std::cout << "strings are not lexicographically increasing\n";
            is_sorted = false;
        }
        return comm.allreduce_single(send_buf({is_sorted}), op(ops::logical_and<>{}));
    }

    bool is_complete(InputPermutation const& permutation, Communicator const& comm) {
        using namespace kamping;
        auto is_complete = check_permutation_complete(permutation, input_container_.size(), comm);
        return comm.allreduce_single(send_buf({is_complete}), op(ops::logical_and<>{}));
    }

private:
    StringContainer input_container_;
};

template <typename StringSet>
class SpaceEfficientChecker {
public:
    using StringContainer = StringLcpContainer<StringSet>;
    using Char = StringContainer::Char;

    void store_container(StringContainer& container) {
        copy_container(container, input_container_);
    }

    bool is_sorted(std::vector<size_t> const& permutation_, Communicator const& comm) {
        using namespace kamping;

        auto const permutation = to_input_permutation(permutation_, comm);
        auto const num_quantiles = compute_num_quantiles(comm);
        auto const quantile_size = tlx::div_ceil(permutation.size(), num_quantiles);
        assert(num_quantiles * quantile_size >= permutation.size());

        bool is_sorted = true;
        auto const &ranks = permutation.ranks(), &strings = permutation.strings();

        std::vector<Char> lower_bound{0};
        for (size_t i = 0; i != num_quantiles; ++i) {
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

            if (!quantile.make_string_set().check_order()) {
                std::cout << "quantile is not sorted\n";
                is_sorted = false;
            }

            if (!quantile.empty()) {
                // note that strings are null-terminated at this point
                if (!dss_schimek::leq(lower_bound.data(), quantile.front().string)) {
                    std::cout << "quantiles are not lexicographically increasing\n";
                    is_sorted = false;
                };
                lower_bound = quantile.get_raw_string(quantile.size() - 1);
            }
        }

        auto const ss = input_container_.make_string_set();
        if (!check_permutation_global_order(ss, permutation, comm)) {
            std::cout << "strings are not distributed to the correct PEs\n";
            is_sorted = false;
        }
        return comm.allreduce_single(send_buf({is_sorted}), op(ops::logical_and<>{}));
    }

    bool is_complete(std::vector<size_t> const& global_permutation, Communicator const& comm) {
        using namespace kamping;
        auto permutation = to_input_permutation(global_permutation, comm);
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

    InputPermutation
    to_input_permutation(std::vector<size_t> const& global_permutation, Communicator const& comm) {
        using namespace kamping;
        auto const size = input_container_.size();
        auto const total_size = comm.allreduce_single(send_buf(size), op(std::plus<>{}));
        auto const chunk_size = tlx::div_ceil(total_size, comm.size());

        auto const dest_rank = [=](auto const i) { return i / chunk_size; };

        std::vector<std::array<size_t, 3>> triples(size), recv_triples;
        auto const get_global_index = [](auto const& triple) { return triple[0]; };
        auto const get_string = [](auto const& triple) { return triple[1]; };
        auto const get_rank = [](auto const& triple) { return triple[2]; };

        std::vector<int> counts(comm.size()), offsets(comm.size());
        for (auto const index: global_permutation) {
            ++counts[dest_rank(index)];
        }
        std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), size_t{0});

        for (size_t i = 0; auto const& index: global_permutation) {
            triples[offsets[dest_rank(index)]++] = {index, i++, comm.rank()};
        }

        comm.alltoallv(send_buf(triples), send_counts(counts), recv_buf(recv_triples));

        std::sort(recv_triples.begin(), recv_triples.end(), [&](auto const& lhs, auto const& rhs) {
            return get_global_index(lhs) < get_global_index(rhs);
        });

        std::vector<size_t> strings(recv_triples.size()), ranks(recv_triples.size());
        std::transform(recv_triples.begin(), recv_triples.end(), strings.begin(), get_string);
        std::transform(recv_triples.begin(), recv_triples.end(), ranks.begin(), get_rank);

        return InputPermutation{std::move(ranks), std::move(strings)};
    }
};

} // namespace dss_mehnert
