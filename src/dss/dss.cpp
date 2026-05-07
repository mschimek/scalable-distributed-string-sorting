// (c) 2024 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include "dss/dss.hpp"

namespace dss {
template std::vector<unsigned char>
run_sorter<kDefaultAlltoallConfig, unsigned char, dss_mehnert::Communicator>(
    std::vector<unsigned char>&, dss_mehnert::Communicator const&,
    SamplerArgs const&, SplitterSorter);
}
