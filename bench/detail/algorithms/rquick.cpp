// (c) 2023 Pascal Mehnert
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include "detail/algorithms/rquick.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <utility>

#include <kamping/collectives/barrier.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "dss/mpi/is_sorted.hpp"
#include "dss/mpi/print_strings.hpp"
#include "dss/sorter/RQuick2/RQuick.hpp"
#include "dss/sorter/RQuick2/Util.hpp"
#include "dss/strings/stringcontainer.hpp"
#include "dss/strings/stringset.hpp"
#include "dss/util/measuringTool.hpp"
#include "input/generation.hpp"

namespace dss_mehnert {
namespace bench {
namespace {

template <typename CharType, bool use_lcps>
class RQuickAlgorithm : public AlgorithmBase {
    using StringSet = dss_mehnert::StringSet<CharType, Length>;
    using StringPtr = std::conditional_t<
        use_lcps,
        tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>,
        tlx::sort_strings_detail::StringPtr<StringSet>>;
    using SortedContainer = RQuick2::Container<StringPtr>;

public:
    using AlgorithmBase::AlgorithmBase;

    void prepare() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.disableCommVolume();
        input_container_.delete_all();
        sorted_container_.reset();
        input_container_ = input::generate_strings<StringSet>(args_.input_config(comm_), comm_);

        if (args_.check_sorted || args_.check_complete) {
            checker_.store_container(input_container_);
        }
        measuring_tool.enableCommVolume();

        comm_.barrier();
    }

    void run() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        std::random_device rd;
        std::mt19937_64 gen{rd()};

        auto const tag = comm_.default_tag();
        auto const& mpi_comm = comm_.mpi_communicator();

        measuring_tool.start("none", "sorting_overall");
        RQuick2::Data<StringPtr> data{input_container_.release_raw_strings()};
        sorted_container_.emplace(
            RQuick2::sort(std::move(data), tag, gen, mpi_comm, args_.local_sorter)
        );
        measuring_tool.stop("none", "sorting_overall", comm_);
    }

    void verify() override {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.disable();
        measuring_tool.disableCommVolume();

        // is_sorted/is_complete only compare char/string counts, so they work regardless of
        // whether RQuick tracked LCP values
        if (args_.check_sorted) {
            auto const is_sorted = checker_.is_sorted(sorted_container_->make_string_set(), comm_);
            die_verbose_unless(is_sorted, "output is not sorted");
            auto const is_complete = checker_.is_complete(*sorted_container_, comm_);
            die_verbose_unless(is_complete, "output is missing chars or strings");
        }
        if (args_.check_complete) {
            // check_exhaustive cross-checks LCP values, which only exist with --rquick-lcp
            if constexpr (use_lcps) {
                auto const is_exact = checker_.check_exhaustive(*sorted_container_, comm_);
                die_verbose_unless(is_exact, "output is not a permutation of the input");
            } else if (comm_.is_root()) {
                std::cout << "--check-complete requires --rquick-lcp (no LCP array without it)\n";
            }
        }
        if (args_.print_sorted) {
            gather_and_print_strings(*sorted_container_, comm_);
        }
    }

private:
    StringLcpContainer<StringSet> input_container_;
    MergeSortChecker<StringSet> checker_;
    std::optional<SortedContainer> sorted_container_;
};

} // namespace

std::unique_ptr<AbstractAlgorithm> make_rquick(SorterArgs const& args, Communicator const& comm) {
    using CharType = unsigned char;

    if (args.rquick_lcp) {
        return std::make_unique<RQuickAlgorithm<CharType, true>>(args, comm);
    } else {
        return std::make_unique<RQuickAlgorithm<CharType, false>>(args, comm);
    }
}

} // namespace bench
} // namespace dss_mehnert
