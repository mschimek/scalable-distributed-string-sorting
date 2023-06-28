// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <iostream>
#include <random>
#include <vector>

#include "mpi/alltoall.hpp"

namespace dss_schimek {
namespace mpi {

size_t randomDataAllToAllExchange(size_t sizeInBytesPerPE) {
    dss_schimek::mpi::environment env;
    std::random_device rd;
    std::mt19937 randGenerator(rd());
    std::uniform_int_distribution<unsigned char> dist(65, 90);
    std::vector<unsigned char> randDataToSend;
    std::vector<size_t> sendCounts(env.size(), sizeInBytesPerPE);
    randDataToSend.reserve(sizeInBytesPerPE * env.size());

    for (size_t i = 0; i < sizeInBytesPerPE * env.size(); ++i)
        randDataToSend.emplace_back(dist(randGenerator));

    std::vector<unsigned char> recvData =
        dss_schimek::mpi::alltoallv_small(randDataToSend, sendCounts);
    volatile size_t sum = std::accumulate(recvData.begin(), recvData.end(), 0);
    return sum;
}

} // namespace mpi
} // namespace dss_schimek
