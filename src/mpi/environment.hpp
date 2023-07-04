/*******************************************************************************
 * dsss/mpi/environment.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <iostream>
#include <limits>

#include <kamping/communicator.hpp>
#include <mpi.h>

namespace dss_schimek {
namespace mpi {

/// \brief Provides an interface for MPI stats (e.g. communicator size/rank).
class environment {
public:
    environment();
    environment(MPI_Comm communicator);
    environment(environment const& other);
    environment(environment&& other);

    template <template <typename...> typename Container, template <typename> typename... Plugins>
    environment(kamping::Communicator<Container, Plugins...> comm)
        : environment(comm.mpi_communicator()) {}

    environment& operator=(environment&& other);

    ~environment();

    /// \brief Finalizes MPI.
    void finalize();

    /// \brief Checks if MPI has been initialized.
    /// \return \e true if MPI has been initialized \e false otherwise.
    static bool initialized() {
        std::int32_t initialized;
        MPI_Initialized(&initialized);
        return (initialized != 0);
    }

    /// \brief Checks if MPI has been finalized.
    /// \return \e true if MPI has been finalized \e false otherwise.
    static bool finalized() {
        std::int32_t finalized;
        MPI_Finalized(&finalized);
        return (finalized != 0);
    }

    void setCommunicator(MPI_Comm communicator) {
        communicator_ = communicator;
        int32_t world_rank_tmp;
        int32_t world_size_tmp;
        MPI_Comm_rank(communicator_, &world_rank_tmp);
        MPI_Comm_size(communicator_, &world_size_tmp);
        if (world_rank_tmp < 0 || world_size_tmp < 0) {
            std::cout << "rank: " << world_rank_tmp << " or "
                      << "size: " << world_size_tmp << " is negative!" << std::endl;
        }
        world_rank_ = static_cast<uint32_t>(world_rank_tmp);
        world_size_ = static_cast<uint32_t>(world_size_tmp);
    }

    /// \brief Gets rank of the processing element in the group (communicator).
    /// \return The rank of the processing element.
    std::uint32_t rank() const;

    /// \brief Gets size of the group (communicator).
    /// \return The size of the group.
    std::uint32_t size() const;

    /// \return The maximum integer value that MPI can handle as amount of
    /// that must be communicated
    constexpr std::size_t mpi_max_int() { return std::numeric_limits<std::int32_t>::max(); }

    /// \brief Gets the communiator the environment was creasted for.
    /// \return The communicator of the environment.
    MPI_Comm communicator() const;

    /// \brief Creates an MPI_Barrier for the communicator.
    void barrier() const;

private:
    MPI_Comm communicator_;
    std::uint32_t world_rank_;
    std::uint32_t world_size_;
};

} // namespace mpi
} // namespace dss_schimek

/******************************************************************************/
