// Single-rank MPI stub for HeCBench miniWeather-omp (record-replay builds without MPI).
// Supports exactly the calls miniWeather uses. With one rank every message is sent to
// itself: Isend copies into the posted Irecv with the same tag (or is held until it is
// posted), so the periodic halo exchange matches a real 1-rank MPI run.
#pragma once
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Request;
typedef struct { int dummy; } MPI_Status;
#define MPI_COMM_WORLD 0
#define MPI_DOUBLE 8
#define MPI_SUM 0
#define MPI_SUCCESS 0
#define MPI_STATUSES_IGNORE ((MPI_Status *)0)
#define MPI_STATUS_IGNORE ((MPI_Status *)0)

namespace mpi_stub {
struct Msg { void *buf; size_t bytes; int tag; bool done; };
inline std::vector<Msg> &recvs() { static std::vector<Msg> v; return v; }
inline std::vector<Msg> &sends() { static std::vector<Msg> v; return v; }
inline size_t bytes(int count, MPI_Datatype t) { return (size_t)count * (size_t)t; }
}

inline int MPI_Init(int *, char ***) { return MPI_SUCCESS; }
inline int MPI_Finalize() { return MPI_SUCCESS; }
inline int MPI_Comm_size(MPI_Comm, int *n) { *n = 1; return MPI_SUCCESS; }
inline int MPI_Comm_rank(MPI_Comm, int *r) { *r = 0; return MPI_SUCCESS; }
inline int MPI_Barrier(MPI_Comm) { return MPI_SUCCESS; }
inline int MPI_Allreduce(const void *in, void *out, int count, MPI_Datatype t, MPI_Op, MPI_Comm) {
  if (in != out) std::memcpy(out, in, mpi_stub::bytes(count, t));
  return MPI_SUCCESS;
}
inline int MPI_Irecv(void *buf, int count, MPI_Datatype t, int, int tag, MPI_Comm, MPI_Request *req) {
  using namespace mpi_stub;
  for (auto &s : sends())
    if (!s.done && s.tag == tag) {
      std::memcpy(buf, s.buf, bytes(count, t)); s.done = true; *req = -1; return MPI_SUCCESS;
    }
  recvs().push_back({buf, bytes(count, t), tag, false});
  *req = (int)recvs().size() - 1;
  return MPI_SUCCESS;
}
inline int MPI_Isend(const void *buf, int count, MPI_Datatype t, int, int tag, MPI_Comm, MPI_Request *req) {
  using namespace mpi_stub;
  for (auto &r : recvs())
    if (!r.done && r.tag == tag) {
      std::memcpy(r.buf, buf, bytes(count, t)); r.done = true; *req = -1; return MPI_SUCCESS;
    }
  sends().push_back({const_cast<void *>(buf), bytes(count, t), tag, false});
  *req = -1;
  return MPI_SUCCESS;
}
inline int MPI_Waitall(int, MPI_Request *, MPI_Status *) {
  using namespace mpi_stub;
  bool pending = false;
  for (auto &r : recvs()) pending |= !r.done;
  if (pending && sends().empty()) { std::fprintf(stderr, "mpi stub: unmatched receive\n"); std::abort(); }
  // All matched messages are complete; clear finished ones.
  auto clear = [](std::vector<Msg> &v) { std::vector<Msg> k; for (auto &m : v) if (!m.done) k.push_back(m); v.swap(k); };
  clear(recvs()); clear(sends());
  return MPI_SUCCESS;
}
