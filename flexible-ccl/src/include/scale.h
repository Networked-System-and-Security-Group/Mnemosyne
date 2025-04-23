#ifndef NCCL_SCALE_H_
#define NCCL_SCALE_H_

#include "core.h"
#include <cstddef>
#include "nccl.h"
#include "socket.h"
#include "bootstrap.h"

#define ADDR_LIST_LEN (512)
typedef ncclComm_t ncclCommIncomplete_t;

struct ncclNewRankInfoInternal {
  ncclCommIncomplete_t comm;
};

struct ncclCommInfoInternal {
  ncclCommIncomplete_t comm;
  ncclUniqueId *uniqueId;
};

// in `init.cc`
ncclResult_t ncclInit();
// in `init.cc`
ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config);
// in `init.cc`
int64_t ncclParamSetStackSize();
// in `init.cc`
ncclResult_t commAlloc(struct ncclComm *comm, struct ncclComm *parent, int ndev, int rank);
// in `init.cc`
uint64_t hashUniqueId(ncclUniqueId const &id);
// in `init.cc`
ncclResult_t initTransportsNewRank(struct ncclComm* comm, const struct ncclComm* peerComm);
// in `reinit.cc`
ncclResult_t devCommResetup(ncclComm_t comm);
// in `init.cc`
ncclResult_t devCommSetup(ncclComm_t comm);

#endif // NCCL_SCALE_H_
