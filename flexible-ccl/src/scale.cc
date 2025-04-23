#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "graph.h"
#include "nccl.h"
#include "group.h"
#include "enqueue.h"
#include "nccl_common.h"
#include "serialize.h"
#include <cassert>
#include <scale.h>
#include <cstdlib>
#include "tuner.h"
#include "transport.h"
#include "coll_net.h"

struct ncclCommAddRankAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
  // for ncclCommAddNewRank & ncclCommInitNewRank
  ncclNewRankInfo *newRankInfo;
  // for ncclCommInitNewRank
  struct ncclCommInfoInternal *peerInfo;
};

static ncclResult_t ncclCommInitNewRankFunc(struct ncclAsyncJob *job_) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job = (struct ncclCommAddRankAsyncJob *)job_;
  ncclComm_t comm = job->comm;
  int cudaDev = comm->cudaDev;
  ncclNewRankInfo *newRankInfo = job->newRankInfo;
  ncclComm_t peerCommInfo = job->peerInfo->comm;
  ncclUniqueId *commId = job->peerInfo->uniqueId;
  int nRanks = peerCommInfo->nRanks + 1;
  int myRank = nRanks - 1;
  size_t maxLocalSizeBytes = 0;
  int cudaArch;
  int archMajor, archMinor;
  unsigned long long commIdHash;
  struct ncclNewRankInfoInternal info;
  struct bootstrapState *state = (struct bootstrapState *)peerCommInfo->bootstrap;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), result, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), result, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), result, fail);
  cudaArch = 100 * archMajor + 10 * archMinor;

  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }

  NCCLCHECKGOTO(commAlloc(comm, NULL, nRanks, myRank), result, fail);
  // obtain a unique hash using the first commId
  comm->commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  commIdHash = hashUniqueId(*commId);
  INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", __func__,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  NCCLCHECKGOTO(bootstrapInitNew(comm, state), result, fail);
  comm->cudaArch = cudaArch;

  NCCLCHECKGOTO(initTransportsNewRank(comm, peerCommInfo), result, fail);
  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), result, fail);
  if (comm->tuner) {
    NCCLCHECK(comm->tuner->init(comm->nRanks, comm->nNodes, ncclDebugLog, &comm->tunerContext));
  }
  comm->initState = ncclSuccess;

  info.comm = comm;
  ncclInfoSerialize((char *)newRankInfo, &info);

exit:
  return result;
fail:
  comm->initState = result;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitNewRank, ncclComm_t* comm, ncclCommInfo* commInfo, ncclNewRankInfo* newRankInfo);
ncclResult_t ncclCommInitNewRank(ncclComm_t* newcomm, ncclCommInfo* commInfo, ncclNewRankInfo* newRankInfo) {
  ncclResult_t result = ncclSuccess;
  ncclCommInfoInternal *peerInfo = (ncclCommInfoInternal *)commInfo->internal;
  int cudaDev = -1;
  ncclComm_t comm = NULL;

  ncclInfoDeserialize(peerInfo);
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  CUDACHECKGOTO(cudaGetDevice(&cudaDev), result, fail);
  // first call ncclInit, this will setup the environment
  NCCLCHECKGOTO(ncclInit(), result, fail);

  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), result, fail);

  NCCLCHECKGOTO(ncclCalloc(&comm, 1), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), result, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), result, fail);
  comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.
  *comm->abortFlagRefCount = 1;
  comm->cudaDev = cudaDev;
  NCCLCHECKGOTO(parseCommConfig(comm, &config), result, fail);
  /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInternalError;
  *newcomm = comm;

  struct ncclCommAddRankAsyncJob *job;
  NCCLCHECKGOTO(ncclCalloc(&job, 1), result, fail);
  job->comm = comm;
  job->newRankInfo = newRankInfo;
  job->peerInfo = peerInfo;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitNewRankFunc, NULL, free, comm), result, fail);

exit:
  return ncclGroupErrCheck(result);
fail:
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

static ncclResult_t ncclCommAddNewRankFunc(struct ncclAsyncJob *job_) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job = (struct ncclCommAddRankAsyncJob *)job_;
  ncclComm_t comm = job->base.comm;
  int cudaDev = comm->cudaDev;
  struct ncclNewRankInfoInternal *newRankInfo = (ncclNewRankInfoInternal *)job->newRankInfo->internal;
  ncclComm_t newRankComm = newRankInfo->comm;
  struct bootstrapState *state = (struct bootstrapState *)comm->bootstrap;
  struct bootstrapState *newRankState = (struct bootstrapState *)newRankComm->bootstrap;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), result, fail);

  // update nRanks & bootstrap state
  int nRanks;
  nRanks = ++comm->nRanks;
  state->nranks = nRanks;
  NCCLCHECKGOTO(ncclRealloc(&state->peerP2pAddresses, nRanks - 1, nRanks), result, fail);
  NCCLCHECKGOTO(ncclRealloc(&state->peerProxyAddresses, nRanks - 1, nRanks), result, fail);
  NCCLCHECKGOTO(ncclRealloc(&state->peerProxyAddressesUDS, nRanks - 1, nRanks), result, fail);
  state->peerP2pAddresses[nRanks - 1] = newRankState->peerP2pAddresses[nRanks - 1];
  state->peerProxyAddresses[nRanks - 1] = newRankState->peerProxyAddresses[nRanks - 1];
  state->peerProxyAddressesUDS[nRanks - 1] = newRankState->peerProxyAddressesUDS[nRanks - 1];

  // update peerInfo
  NCCLCHECKGOTO(ncclRealloc(&comm->peerInfo, nRanks, nRanks + 1), result, fail);
  comm->peerInfo[nRanks] = comm->peerInfo[nRanks - 1];
  comm->peerInfo[nRanks - 1] = newRankComm->peerInfo[nRanks - 1];

  // update channels
  comm->connectSend[nRanks - 1] = comm->connectRecv[nRanks - 1] = 0;
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    int &prev = channel->ring.prev;
    int &next = channel->ring.next;
    if (prev == nRanks - 2) {
      prev = nRanks - 1;
      NCCLCHECKGOTO(ncclTransportP2pForcedConnect(comm, c, 1, &prev, 0, nullptr, 0), result, fail);
    } else if (next == 0) {
      next = nRanks - 1;
      NCCLCHECKGOTO(ncclTransportP2pForcedConnect(comm, c, 0, nullptr, 1, &next, 0), result, fail);
    }
  }

  // update graphs
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    comm->graphs[i] = newRankComm->graphs[i];
  }

exit:
  return result;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommAddNewRank, ncclComm_t comm, ncclNewRankInfo* newRankInfo);
ncclResult_t ncclCommAddNewRank(ncclComm_t comm, ncclNewRankInfo* newRankInfo) {
  ncclResult_t result = ncclSuccess;
  ncclNewRankInfoInternal *info = (ncclNewRankInfoInternal *)newRankInfo->internal;
  ncclInfoDeserialize(info);

  struct ncclCommAddRankAsyncJob *job;
  NCCLCHECKGOTO(ncclCalloc(&job, 1), result, fail);
  job->comm = comm;
  job->newRankInfo = newRankInfo;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommAddNewRankFunc, NULL, free, comm), result, fail);

ret:
  return result;
fail:
  goto ret;
}

NCCL_API(ncclResult_t, ncclCommExportInfo, ncclComm_t comm, ncclUniqueId* commId, ncclCommInfo* commInfo);
ncclResult_t ncclCommExportInfo(ncclComm_t comm, ncclUniqueId* commId, ncclCommInfo* commInfo) {
  struct ncclCommInfoInternal info{comm, commId};
  int offset = ncclInfoSerialize(commInfo->internal, &info);
  assert(offset <= sizeof(ncclCommInfo));
  return ncclSuccess;
}

static ncclResult_t ncclCommSetupNewRankFunc(struct ncclAsyncJob *job_) {
  ncclResult_t res = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job = (struct ncclCommAddRankAsyncJob *)job_;
  ncclComm_t comm = job->base.comm;
  int cudaDev = comm->cudaDev;
  int rank = comm->rank;
  int nRanks = comm->nRanks;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);

  if (rank == 0 || rank == nRanks - 2) {
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), res, fail);
    NCCLCHECKGOTO(devCommResetup(comm), res, fail);
  } else if (rank == nRanks - 1) {
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), res, fail);
    NCCLCHECKGOTO(devCommSetup(comm), res, fail);
  }

exit:
  return res;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommSetupNewRank, ncclComm_t comm);
ncclResult_t ncclCommSetupNewRank(ncclComm_t comm) {
  ncclResult_t res = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job;
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommSetupNewRankFunc, NULL, free, comm), res, fail);

exit:
  return res;
fail:
  goto exit;
}