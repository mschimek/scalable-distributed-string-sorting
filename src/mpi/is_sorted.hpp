// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <limits>
#include <utility>

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/communicator.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/recv.hpp>
#include <tlx/die.hpp>
#include <tlx/math.hpp>
#include <tlx/math/div_ceil.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringtools.hpp"

namespace dss_mehnert {

inline std::pair<SimplePermutation, std::vector<size_t>> make_input_permutation(
    std::vector<size_t> const& global_permutation,
    Communicator const& comm,
    size_t const lower_bound = 0,
    size_t const upper_bound = std::numeric_limits<size_t>::max()
) {
    using namespace kamping;

    auto const size = global_permutation.size();
    auto const total_size = comm.allreduce_single(send_buf(size), op(std::plus<>{}));
    auto const chunk_size = tlx::div_ceil(total_size, comm.size());

    auto const dest_rank = [=](auto const i) { return i / chunk_size; };

    std::vector<std::array<size_t, 3>> triples(size), recv_triples;
    auto const get_global_index = [](auto const& triple) { return triple[0]; };
    auto const get_string = [](auto const& triple) { return triple[1]; };
    auto const get_rank = [](auto const& triple) { return triple[2]; };

    std::vector<int> counts(comm.size()), offsets(comm.size());
    for (auto const index: global_permutation) {
        if (lower_bound <= index && index < upper_bound) {
            ++counts[dest_rank(index)];
        }
    }
    std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), size_t{0});

    for (size_t i = 0; auto const& index: global_permutation) {
        if (lower_bound <= index && index < upper_bound) {
            triples[offsets[dest_rank(index)]++] = {index, i, comm.rank()};
        }
        ++i;
    }

    comm.alltoallv(send_buf(triples), send_counts(counts), recv_buf(recv_triples));

    std::sort(recv_triples.begin(), recv_triples.end(), [&](auto const& lhs, auto const& rhs) {
        return get_global_index(lhs) < get_global_index(rhs);
    });

    auto const recv_size = recv_triples.size();
    std::vector<size_t> strings(recv_size), ranks(recv_size), indices(recv_size);
    std::transform(recv_triples.begin(), recv_triples.end(), strings.begin(), get_string);
    std::transform(recv_triples.begin(), recv_triples.end(), ranks.begin(), get_rank);
    std::transform(recv_triples.begin(), recv_triples.end(), indices.begin(), get_global_index);

    return {SimplePermutation{std::move(ranks), std::move(strings)}, std::move(indices)};
}

inline auto reverse_comm(Communicator const& comm) {
    return comm.split(0, comm.size() - comm.rank());
}

template <typename T>
std::optional<T>
get_predecessor(T const& last_elem, bool const is_empty, Communicator const& comm) {
    using namespace kamping;

    auto const bottom = -1, top = comm.size_signed();
    auto const predecessor = comm.exscan_single(
        send_buf(is_empty ? bottom : comm.rank_signed()),
        op(ops::max<>{}),
        values_on_rank_0(bottom)
    );
    auto const successor = reverse_comm(comm).exscan_single(
        send_buf(is_empty ? top : comm.rank_signed()),
        op(ops::min<>{}),
        values_on_rank_0(top)
    );

    std::optional<T> result;
    if (!is_empty) {
        Request req_succ;
        T pred_elem;

        if (successor != top) {
            comm.issend(send_buf(last_elem), destination(successor), request(req_succ));
        }
        if (predecessor != bottom) {
            comm.recv(recv_buf(pred_elem), source(predecessor));
        }
        req_succ.wait();

        if (predecessor != bottom) {
            result.emplace(pred_elem);
        };
    }
    return result;
}

template <typename StringSet>
std::pair<std::vector<typename StringSet::Char>, std::vector<typename StringSet::Char>>
get_permutation_first_last_strings(
    StringSet const& ss, SimplePermutation const& permutation, Communicator const& comm
) {
    namespace pdms = dss_mehnert::sorter::prefix_doubling;

    std::vector<typename StringSet::Char> first_string{0}, last_string{0};
    if (permutation.empty()) {
        pdms::apply_permutation(ss, SimplePermutation{}, comm);
    } else {
        SimplePermutation const sentinels{
            {permutation.ranks().front(), permutation.ranks().back()},
            {permutation.strings().front(), permutation.strings().back()}};

        auto sentinel_strings = pdms::apply_permutation(ss, sentinels, comm);
        first_string = sentinel_strings.get_raw_string(0);
        last_string = sentinel_strings.get_raw_string(1);
    }
    return {std::move(first_string), std::move(last_string)};
}

template <typename Char>
bool check_global_order(
    std::vector<Char> const& first_string,
    std::vector<Char> const& last_string,
    bool const is_empty,
    Communicator const& comm
) {
    assert_equal(first_string.back(), 0);
    assert_equal(last_string.back(), 0);

    auto const pred_string = get_predecessor(last_string, is_empty, comm);
    if (pred_string) {
        return dss_schimek::leq(pred_string->data(), first_string.data());
    } else {
        return true;
    }
}

template <typename StringSet>
bool check_permutation_global_order(
    StringSet const& ss, SimplePermutation const& permutation, Communicator const& comm
) {
    auto const [first, last] = get_permutation_first_last_strings(ss, permutation, comm);
    return check_global_order(first, last, permutation.empty(), comm);
}

inline bool check_permutation_complete(
    SimplePermutation const& permutation, size_t const local_size, Communicator const& comm
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

    template <typename Subcommunicators>
    bool is_sorted(MultiLevelPermutation const& permutation, Subcommunicators const& comms) {
        return is_sorted(make_input_permutation(permutation, comms), comms);
    }

    template <typename Subcommunicators>
    bool is_sorted(SimplePermutation const& permutation, Subcommunicators const& comms) {
        using namespace kamping;
        namespace pdms = dss_mehnert::sorter::prefix_doubling;

        auto const& comm = comms.comm_root();
        auto const ss = input_container_.make_string_set();

        bool is_sorted = true;
        auto sorted_strings = pdms::apply_permutation(ss, permutation, comm);
        if (!sorted_strings.make_string_set().check_order()) {
            std::cout << "the permutation is not sorted locally\n";
            is_sorted = false;
        }

        if (!check_permutation_global_order(ss, permutation, comm)) {
            std::cout << "strings are not lexicographically increasing\n";
            is_sorted = false;
        }
        return comm.allreduce_single(send_buf({is_sorted}), op(ops::logical_and<>{}));
    }

    template <typename Subcommunicators>
    bool is_complete(MultiLevelPermutation const& permutation, Subcommunicators const& comms) {
        return is_complete(make_input_permutation(permutation, comms), comms);
    }

    template <typename Subcommunicators>
    bool is_complete(SimplePermutation const& permutation, Subcommunicators const& comms) {
        using namespace kamping;
        auto const& comm = comms.comm_root();
        auto is_complete = check_permutation_complete(permutation, input_container_.size(), comm);
        return comm.allreduce_single(send_buf({is_complete}), op(ops::logical_and<>{}));
    }

private:
    StringContainer input_container_;

    template <typename Subcommunicators>
    SimplePermutation make_input_permutation(
        MultiLevelPermutation const& permutation, Subcommunicators const& comms
    ) {
        std::vector<size_t> global_permutation(input_container_.size());
        permutation.apply(global_permutation, 0, comms);
        return dss_mehnert::make_input_permutation(global_permutation, comms.comm_root()).first;
    }
};

template <bool use_unique_permutation, typename StringSet>
class SpaceEfficientChecker {
public:
    using StringContainer = StringLcpContainer<StringSet>;
    using Char = StringContainer::Char;
    using String = StringContainer::String;

    void store_container(StringContainer& container) {
        copy_container(container, input_container_);
    }

    bool is_sorted(std::vector<size_t> const& global_ranks, Communicator const& comm) {
        using namespace kamping;
        namespace pdms = dss_mehnert::sorter::prefix_doubling;

        auto const num_quantiles = std::max<size_t>(1, compute_num_quantiles(comm));
        auto const max_elem = std::max_element(global_ranks.begin(), global_ranks.end());
        auto const max_rank = comm.allreduce_single(
            kamping::send_buf(global_ranks.empty() ? static_cast<size_t>(0) : *max_elem),
            kamping::op(kamping::ops::max<>{})
        );
        auto const quantile_size = tlx::div_ceil(max_rank + 1, num_quantiles);

        bool is_sorted = true;

        std::vector<Char> lower_bound{0};
        size_t rank_lower_bound = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i != num_quantiles; ++i) {
            auto const rank_lower = i * quantile_size, rank_upper = (i + 1) * quantile_size;

            auto const [quantile_permutation, quantile_ranks] =
                make_input_permutation(global_ranks, comm, rank_lower, rank_upper);
            auto quantile = pdms::apply_permutation(
                input_container_.make_string_set(),
                quantile_permutation,
                comm
            );

            auto const ss = quantile.make_string_set();
            {
                auto const last_string =
                    ss.empty() ? std::vector<Char>{0} : quantile.get_raw_string(ss.size() - 1);
                if (auto pred_string = get_predecessor(last_string, ss.empty(), comm)) {
                    lower_bound = std::move(*pred_string);
                }

                auto const last_rank = ss.empty() ? 0 : quantile_ranks.back();
                if (auto pred_rank = get_predecessor(last_rank, ss.empty(), comm)) {
                    rank_lower_bound = *pred_rank;
                }
            }

            if (!ss.empty()) {
                if (!check_quantile_order(ss, quantile_ranks, lower_bound, rank_lower_bound)) {
                    std::cout << "the strings are not sorted correctly\n";
                    is_sorted = false;
                }

                lower_bound = quantile.get_raw_string(quantile.size() - 1);
                rank_lower_bound = quantile_ranks.back();
            }


            int const last_non_empty = comm.allreduce_single(
                kamping::send_buf(ss.empty() ? -1 : comm.rank_signed()),
                kamping::op(kamping::ops::max<>{})
            );

            if (comm.is_valid_rank(last_non_empty)) {
                comm.bcast(send_recv_buf(lower_bound), root(last_non_empty));
                comm.bcast(send_recv_buf(rank_lower_bound), root(last_non_empty));
            }
        }
        return comm.allreduce_single(send_buf({is_sorted}), op(ops::logical_and<>{}));
    }

    bool is_complete(std::vector<size_t> const& global_ranks, Communicator const& comm) {
        using namespace kamping;

        auto recv_ranks = comm.gatherv(send_buf(global_ranks)).extract_recv_buffer();

        bool is_complete = true;
        if (comm.is_root()) {
            auto const begin = recv_ranks.begin(), end = recv_ranks.end();
            std::sort(begin, end);
            std::adjacent_difference(begin, end, begin);

            if (!recv_ranks.empty()) {
                is_complete &= recv_ranks.front() == 0;
                if (use_unique_permutation) {
                    is_complete &= std::all_of(begin + 1, end, [](auto const x) { return x == 1; });
                } else {
                    is_complete &= std::all_of(begin + 1, end, [](auto const x) { return x <= 1; });
                }
            }
        }
        return comm.allreduce_single(send_buf({is_complete}), op(ops::logical_and<>{}));
    }

private:
    static constexpr size_t quantile_size = 90 * 1024 * 1024;

    StringContainer input_container_;

    size_t compute_num_quantiles(Communicator const& comm) {
        using namespace kamping;

        size_t const total_size = input_container_.make_string_set().get_sum_length();
        size_t const local_num_quantiles = tlx::div_ceil(total_size, quantile_size + 1);
        return comm.allreduce_single(send_buf(local_num_quantiles), op(ops::max<>{}));
    }

    bool check_quantile_order(
        StringSet const& quantile_strings,
        std::span<size_t const> quantile_ranks,
        std::vector<Char> const& string_lower_bound,
        size_t const rank_lower_bound
    ) {
        assert_equal(quantile_strings.size(), quantile_ranks.size());

        auto pred_string = string_lower_bound.data();
        auto pred_rank = rank_lower_bound;
        for (size_t i = 0; i != quantile_strings.size(); ++i) {
            auto const& string = quantile_strings.at(i);
            auto const rank = quantile_ranks[i];

            auto const strings_ord = dss_schimek::scmp(pred_string, string.string);
            auto const ranks_eq = pred_rank == rank;

            auto const is_sorted = [=] {
                if constexpr (use_unique_permutation) {
                    return strings_ord <= 0;
                } else {
                    return ranks_eq ? strings_ord == 0 : strings_ord < 0;
                }
            }();

            if (!is_sorted) {
                return false;
            }
            pred_string = string.string, pred_rank = rank;
        }
        return true;
    }
};

} // namespace dss_mehnert
