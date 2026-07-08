// (c) 2024 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/sendrecv.hpp>
#include <kamping/parameter_objects.hpp>
#include <kamping/request.hpp>
#include <mpi.h>

#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace mpi {

// Selects how the 1-factor exchanges:
//   * windowed:     pipeline the exchanges over a fixed window of outstanding
//                   isend/irecv pairs.
//   * synchronized: perform the standard full 1-factor schedule as p (p-1 for even p)
//                   pairwise MPI_Sendrecv exchanges
enum class OneFactorMode { windowed, synchronized };

struct OneFactorParams {
    OneFactorMode mode = OneFactorMode::windowed;
    size_t num_slots = 16;
    bool use_issend = false;
};

// partner of `rank` in round k of the 1-factor schedule
//   * p odd:  for k in [0, p)      partner j := (k - i) mod p
//   * p even: for k in [0, p-1)    partner j := (k - i) mod (p-1),
//             remapping j == i to the center p-1; the center p-1 pairs
//             with j := (k * (p/2)) mod (p-1) (p/2 is the inverse of 2
//             modulo the odd p-1).
inline size_t onefactor_partner(size_t const p, size_t const rank, size_t const k) {
    if (p % 2 == 0) {
        size_t const q = p - 1;
        if (rank == q) {
            return (k * (p / 2)) % q; // center: solves 2*j == k (mod q)
        }
        size_t const j = (k + q - rank) % q;
        return j == rank ? q : j;
    } else {
        return (k + p - rank) % p;
    }
}

// number of rounds in the 1-factor schedule
inline size_t onefactor_num_rounds(size_t const p) { return (p % 2 == 0) ? p - 1 : p; }

// Sparse all-to-all using the 1-factor algorithm to schedule the pairwise
// exchanges, pipelined over a fixed window of `num_slots` outstanding
// isend/irecv pairs. The self-message is handled by a local copy instead of a
// round.
template <typename Communicator, typename SendBuf>
auto alltoallv_onefactor_windowed(
    Communicator const& comm,
    SendBuf&& send_buf,
    std::span<size_t const> send_counts,
    std::span<size_t const> recv_counts,
    OneFactorParams const& params = {}
) {
    using namespace kamping;
    using DataType = std::remove_reference_t<SendBuf>::value_type;

    auto const num_slots = params.num_slots;
    auto const use_issend = params.use_issend;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    auto const p = static_cast<size_t>(comm.size());
    auto const rank = static_cast<size_t>(comm.rank());

    std::vector<size_t> send_displs(p), recv_displs(p);
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), size_t{0});
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), size_t{0});

    auto const send_total = send_displs.back() + send_counts.back();
    measuring_tool.addRawCommunication(send_total * sizeof(DataType), "alltoallv_one_factor");

    auto const recv_total = recv_displs.back() + recv_counts.back();
    std::vector<DataType> receive_data(recv_total);

    // build schedule
    size_t const num_rounds = onefactor_num_rounds(p);
    std::vector<size_t> send_sched, recv_sched;
    send_sched.reserve(num_rounds);
    recv_sched.reserve(num_rounds);
    for (size_t k = 0; k < num_rounds; ++k) {
        size_t const j = onefactor_partner(p, rank, k);
        if (j == rank) {
            continue; // self-message handled by the local copy below
        }
        if (send_counts[j] > 0) {
            send_sched.push_back(j);
        }
        if (recv_counts[j] > 0) {
            recv_sched.push_back(j);
        }
    }

    // local copy of the self-message
    if (send_counts[rank] > 0) {
        std::copy_n(
            send_buf.data() + send_displs[rank],
            send_counts[rank],
            receive_data.data() + recv_displs[rank]
        );
    }

    static constexpr int msg_tag = 16228;
    size_t const slots = std::max<size_t>(1, num_slots);
    size_t const num_recv_slots = std::min(slots, recv_sched.size());
    size_t const num_send_slots = std::min(slots, send_sched.size());

    // requests: [0, num_recv_slots) are receive slots, the remainder sends
    std::vector<MPI_Request> requests(num_recv_slots + num_send_slots, MPI_REQUEST_NULL);
    size_t next_recv = 0, next_send = 0;

    auto post_recv = [&](size_t const slot) {
        if (next_recv < recv_sched.size()) {
            size_t const j = recv_sched[next_recv++];
            std::span<DataType> buf{receive_data.data() + recv_displs[j], recv_counts[j]};
            Request req;
            comm.irecv(
                kamping::recv_buf(buf),
                kamping::recv_count(static_cast<int>(recv_counts[j])),
                source(static_cast<int>(j)),
                tag(msg_tag),
                kamping::request(req)
            );
            requests[slot] = req.mpi_request();
        } else {
            requests[slot] = MPI_REQUEST_NULL;
        }
    };
    auto post_send = [&](size_t const slot) {
        if (next_send < send_sched.size()) {
            size_t const j = send_sched[next_send++];
            std::span<DataType const> buf{send_buf.data() + send_displs[j], send_counts[j]};
            Request req;
            if (use_issend) {
                comm.isend(
                    kamping::send_buf(buf),
                    destination(static_cast<int>(j)),
                    tag(msg_tag),
                    send_mode(send_modes::synchronous),
                    kamping::request(req)
                );
            } else {
                comm.isend(
                    kamping::send_buf(buf),
                    destination(static_cast<int>(j)),
                    tag(msg_tag),
                    kamping::request(req)
                );
            }
            requests[slot] = req.mpi_request();
        } else {
            requests[slot] = MPI_REQUEST_NULL;
        }
    };

    comm.barrier();
    kamping::measurements::timer().start("alltoallv_onefactor_windowed");

    // initially fill every slot
    for (size_t slot = 0; slot < num_recv_slots; ++slot) {
        post_recv(slot);
    }
    for (size_t slot = 0; slot < num_send_slots; ++slot) {
        post_send(num_recv_slots + slot);
    }

    // refill slots in schedule order
    for (int idx = MPI_UNDEFINED;;) {
        MPI_Waitany(static_cast<int>(requests.size()), requests.data(), &idx, MPI_STATUS_IGNORE);
        if (idx == MPI_UNDEFINED) {
            break;
        }
        if (static_cast<size_t>(idx) < num_recv_slots) {
            post_recv(static_cast<size_t>(idx));
        } else {
            post_send(static_cast<size_t>(idx));
        }
    }

    kamping::measurements::timer().stop_and_append();

    return receive_data;
}

// All-to-all using the 1-factor algorithm, executed as the full, schedule: p rounds (p-1 for even
// p), each performing a single synchronous MPI_Sendrecv with the round's partner. Unlike the
// windowed variant this issues an exchange every round (even empty ones), so there are exactly
// `onefactor_num_rounds(p)` pairwise exchanges. The self-message is handled by a
// local copy instead of a round.
template <typename Communicator, typename SendBuf>
auto alltoallv_onefactor_synchronized(
    Communicator const& comm,
    SendBuf&& send_buf,
    std::span<size_t const> send_counts,
    std::span<size_t const> recv_counts
) {
    using namespace kamping;
    using DataType = std::remove_reference_t<SendBuf>::value_type;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    auto const p = static_cast<size_t>(comm.size());
    auto const rank = static_cast<size_t>(comm.rank());

    std::vector<size_t> send_displs(p), recv_displs(p);
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), size_t{0});
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), size_t{0});

    auto const send_total = send_displs.back() + send_counts.back();
    measuring_tool.addRawCommunication(send_total * sizeof(DataType), "alltoallv_one_factor");

    auto const recv_total = recv_displs.back() + recv_counts.back();
    std::vector<DataType> receive_data(recv_total);

    // local copy of the self-message
    if (send_counts[rank] > 0) {
        std::copy_n(
            send_buf.data() + send_displs[rank],
            send_counts[rank],
            receive_data.data() + recv_displs[rank]
        );
    }

    static constexpr int msg_tag = 16228;
    size_t const num_rounds = onefactor_num_rounds(p);

    comm.barrier();
    kamping::measurements::timer().start("alltoallv_onefactor_synchronized");

    for (size_t k = 0; k < num_rounds; ++k) {
        size_t const j = onefactor_partner(p, rank, k);
        if (j == rank) {
            continue; // self-message handled by the local copy above
        }

        std::span<DataType const> sbuf{send_buf.data() + send_displs[j], send_counts[j]};
        std::span<DataType> rbuf{receive_data.data() + recv_displs[j], recv_counts[j]};
        comm.sendrecv(
            kamping::send_buf(sbuf),
            destination(static_cast<int>(j)),
            send_tag(msg_tag),
            kamping::recv_buf(rbuf),
            recv_count(static_cast<int>(recv_counts[j])),
            source(static_cast<int>(j)),
            recv_tag(msg_tag)
        );
    }

    kamping::measurements::timer().stop_and_append();

    return receive_data;
}

// Dispatches to the windowed or synchronized 1-factor implementation.
template <typename Communicator, typename SendBuf>
auto alltoallv_onefactor(
    Communicator const& comm,
    SendBuf&& send_buf,
    std::span<size_t const> send_counts,
    std::span<size_t const> recv_counts,
    OneFactorParams const& params = {}
) {
    if (params.mode == OneFactorMode::synchronized) {
        return alltoallv_onefactor_synchronized(
            comm,
            std::forward<SendBuf>(send_buf),
            send_counts,
            recv_counts
        );
    } else {
        return alltoallv_onefactor_windowed(
            comm,
            std::forward<SendBuf>(send_buf),
            send_counts,
            recv_counts,
            params
        );
    }
}

} // namespace mpi
} // namespace dss_mehnert
