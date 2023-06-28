/*******************************************************************************
 * dsss/mpi/environment.cpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <iostream>
#include "mpi/environment.hpp"
#include "util/macros.hpp"

namespace dss_schimek::mpi {

environment::environment() : environment(MPI_COMM_WORLD) { }

environment::environment(MPI_Comm communicator)
 : communicator_(communicator) {
  if (DSSS_UNLIKELY(!environment::initialized())) {
    MPI_Init(nullptr, nullptr);
  }
  int32_t world_rank_tmp;
  int32_t world_size_tmp;
  MPI_Comm_rank(communicator_, &world_rank_tmp);
  MPI_Comm_size(communicator_, &world_size_tmp);
  if (DSSS_UNLIKELY(world_rank_tmp < 0 || world_size_tmp < 0)) {
    std::cout << "rank: " << world_rank_tmp << " or "
              << "size: " << world_size_tmp << " is negative!" 
              << std::endl;
  }
  world_rank_ = static_cast<uint32_t>(world_rank_tmp);
  world_size_ = static_cast<uint32_t>(world_size_tmp);
}

environment::environment(const environment& other) = default;
environment::environment(environment&& other) = default;
environment::~environment() { }

environment& environment::operator = (environment&& other) {
  communicator_ = other.communicator_;
  world_rank_ = other.world_rank_;
  world_size_ = other.world_size_;
  return *this;
}

void environment::finalize() {
    MPI_Finalize();
  }

std::uint32_t environment::rank() const {
  return world_rank_;
}

std::uint32_t environment::size() const {
  return world_size_;
}

MPI_Comm environment::communicator() const {
  return communicator_;
}

void environment::barrier() const {
  MPI_Barrier(communicator_);
}

} // namespace dsss::mpi

/******************************************************************************/
