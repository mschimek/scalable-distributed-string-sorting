// (c) 2019 Matthias Schimek
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include "detail/algorithms/shared_memory.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include <tlx/die.hpp>
#include <tlx/sort/strings/parallel_sample_sort.hpp>

#include "dss/strings/stringcontainer.hpp"
#include "dss/strings/stringset.hpp"
#include "input/input.hpp"

namespace dss_mehnert {
namespace bench {
namespace {

template <typename CharType>
class SharedMemoryAlgorithm : public AlgorithmBase {
    using String = SimpleString<CharType, CharType*>;
    using StringSet = GenericStringSet<String>;

public:
    SharedMemoryAlgorithm(SorterArgs const& args, Communicator const& comm)
        : AlgorithmBase{args, comm},
          input_container_{generate_strings<StringSet>(args, comm)},
          input_strings_{input_container_.get_strings()} {}

    // restore the original order of the input strings
    void prepare() override { input_container_.set(std::vector{input_strings_}); }

    void run() override {
        auto const before = std::chrono::high_resolution_clock::now();
        tlx::sort_strings_detail::parallel_sample_sort(input_container_.make_string_ptr(), 0, 0);
        auto const after = std::chrono::high_resolution_clock::now();
        auto const delta = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);
        size_t const elapsed = delta.count();

        std::cout << args_.get_prefix(comm_) << " key=sorting_overall max_time=" << elapsed
                  << std::endl;
    }

    void verify() override {
        if (args_.check_sorted) {
            auto const is_sorted = input_container_.make_string_set().check_order();
            die_verbose_unless(is_sorted, "output is not sorted");
        }
    }

    // the sort reports its own timing in run(), so there is nothing for the MeasuringTool to
    // write out here
    void report() override {}

private:
    StringLcpContainer<StringSet> input_container_;
    std::vector<String> input_strings_;
};

} // namespace

std::unique_ptr<AbstractAlgorithm>
make_shared_memory(SorterArgs const& args, Communicator const& comm) {
    tlx_die_unequal(comm.size_signed(), 1);

    return std::make_unique<SharedMemoryAlgorithm<unsigned char>>(args, comm);
}

} // namespace bench
} // namespace dss_mehnert
