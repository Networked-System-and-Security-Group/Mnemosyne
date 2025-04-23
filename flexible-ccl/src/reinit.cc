#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "debug.h"
#include "enqueue.h"
#include "gdrwrap.h"
#include "graph.h"
#include "group.h"
#include "nccl.h"
#include "transport.h"
#include <cstddef>

static ncclResult_t initDeviceFromComm(ncclComm_t comm) {
  // origin: ncclCommInitRankFunc
  // * init kernel attributes
  size_t maxLocalSizeBytes = 0;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  NCCLCHECK(ncclInitKernelsForDevice(comm->cudaArch, &maxLocalSizeBytes));
  if (maxLocalSizeBytes != 0) INFO(NCCL_REINIT, "maxLocalSizeBytes != 0");

  // origin: commAlloc
  // * create NVML handle
  // * create cuda streams (host & device)
  // * creaet cudaMemPoll
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  nvmlDevice_t nvmlDev;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  NCCLCHECK(int64ToBusId(comm->busId, busId));
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev));
  NCCLCHECK(ncclStrongStreamConstruct(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamConstruct(&comm->sharedRes->hostStream));
  do {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = comm->cudaDev;
    CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));
    uint64_t releaseThreshold = ~uint64_t(0);
    CUDACHECK(cudaMemPoolSetAttribute(comm->memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  // origin: initTransportsRank
  // - assume MNNVL, NVLS, CollNet is disabled
  // - assume RuntimeConnect is enabled
  // * init channel
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = &comm->channels[c];
    int nRanks = comm->nRanks;
    int nvlsRanks = comm->localRanks;
    int nPeers = nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */;
    NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream));
    NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + c, sharedRes->tpNRanks,
                                  sharedRes->deviceStream.cudaStream));
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, sharedRes->deviceStream.cudaStream));
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    for (int r = 0; r < nRanks; r++) {
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[c] + comm->topParentRanks[r]);
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1,
                                    sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
    /* guarantee addr has been copied into channel->devPeers */
    NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));
  }
  return ncclSuccess;
}

ncclResult_t devCommResetup(ncclComm_t comm) {
  // origin: devCommSetup
  // * devComm structure
  int nRanks = comm->nRanks;
  struct ncclDevCommAndChannels tmpCommAndChans;
  struct ncclDevCommAndChannels* devCommAndChans = NULL;
  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclCudaCallocAsync(&devCommAndChans, 1, comm->sharedRes->deviceStream.cudaStream));
  NCCLCHECK(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks,
                                comm->sharedRes->deviceStream.cudaStream));
  NCCLCHECK(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank,
                                comm->nRanks, comm->sharedRes->deviceStream.cudaStream));

  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.node = comm->node;
  tmpCommAndChans.comm.nNodes = comm->nNodes;
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;
  tmpCommAndChans.comm.isNvlink = ncclTopoPathAllNVLink(comm->topo);
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];
  if (ncclGdrCopy != NULL) {
    // The workFifoBuf lives in GDR mapped CUDA memory.
    NCCLCHECK(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes,
                                &comm->workFifoBufGdrHandle));
  } else {
    // The workFifoBuf lives in cudaHost memory.
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes));
    comm->workFifoBufDev = comm->workFifoBuf;
  }
  NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoConsumed, MAXCHANNELS));
  tmpCommAndChans.comm.workConsumed = comm->workFifoConsumed;
  for (int c = 0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECK(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks,
                                    nRanks, comm->sharedRes->deviceStream.cudaStream));
    }
  }
  NCCLCHECK(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, comm->sharedRes->deviceStream.cudaStream));
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream));
  return ncclSuccess;
}

typedef struct RemoveRankJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
  int rank;
} * RemoveRankJob_t;

static ncclResult_t ringRemoveRankAsync(struct ncclAsyncJob *job_) {
  ncclResult_t ret = ncclSuccess;
  RemoveRankJob_t job = (RemoveRankJob_t)job_;
  ncclComm_t comm = job->comm;
  struct ncclTopoGraph *graph = &comm->graphs[NCCL_ALGO_RING];
  int bootstrapTag = graph ? graph->id + 1 : 0;
  int targetRank = job->rank; // rank to be removed
  int myRank = comm->rank;    // current rank
  bool needSetup = false;

  // Single rank needs no removal
  if (comm->nRanks == 1)
    return ncclSuccess;

  // Switch to current rank's cuda device.
  CUDACHECKGOTO(cudaSetDevice(comm->localRank), ret, fail);

  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    int peerRecvNew = -1;                 // new recv peer if needed
    int peerSendNew = -1;                 // new recv peer if needed
    int peerRecvOld = channel->ring.prev; // recv peer before rank removal
    int peerSendOld = channel->ring.next; // send peer before rank removal
    bootstrapTag += 0x1000;

    if (targetRank == myRank) {
      // Case 1: Current rank needs to be removed.
      needSetup = true;
      NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, peerRecvOld, bootstrapTag, &peerSendOld, sizeof(int)), ret, fail);
      NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, peerSendOld, bootstrapTag, &peerRecvOld, sizeof(int)), ret, fail);
      channel->ring.prev = targetRank;
      channel->ring.next = targetRank;
      NCCLCHECKGOTO(ncclTransportP2pForcedConnect(comm, c, 1, &targetRank, 1, &targetRank, 0), ret, fail);
    } else if (targetRank == peerRecvOld) {
      // Case 2: Current rank needs connecting to `peerRecvNew`.
      needSetup = true;
      NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, targetRank, bootstrapTag, &peerRecvNew, sizeof(int)), ret, fail);
      channel->ring.prev = peerRecvNew;
      NCCLCHECKGOTO(ncclTransportP2pForcedConnect(comm, c, 1, &peerRecvNew, 0, nullptr, 0), ret, fail);
    } else if (targetRank == peerSendOld) {
      // Case 3: Current rank needs connecting to `peerSendNew`.
      needSetup = true;
      NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, targetRank, bootstrapTag, &peerSendNew, sizeof(int)), ret, fail);
      channel->ring.next = peerSendNew;
      NCCLCHECKGOTO(ncclTransportP2pForcedConnect(comm, c, 0, nullptr, 1, &peerSendNew, 0), ret, fail);
    }

    // Adjust current rank's index in the new ring.
    if (targetRank < myRank) {
      channel->ring.index--;
    }
  }

  // Setup new connections for affected ranks.
  if (needSetup) {
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, graph, 0), ret, fail);
  }

  // Adjust comm world size.
  if (targetRank == myRank) {
    comm->nRanks = 1;
  } else {
    comm->nRanks--;
  }

  // Resetup devComm.
  NCCLCHECKGOTO(devCommResetup(comm), ret, fail);
exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclRemoveRank, ncclComm_t comm, int rank);
ncclResult_t ncclRemoveRank(ncclComm_t comm, int rank) {
  ncclResult_t ret = ncclSuccess;
  RemoveRankJob_t job = nullptr;

  // Make sure that current `comm` is ready.
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  // Create & Init async job structure fro `ringRemoveRankAsync`.
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;
  job->rank = rank;

  // Launch async job for `ringRemoveRankAsync`.
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ringRemoveRankAsync, NULL, free, comm), ret, fail);
  job = nullptr;

exit:
  if (job) free(job);
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclRestoreRank, ncclComm_t comm, int rank);
ncclResult_t ncclRestoreRank(ncclComm_t comm, int rank) {
  INFO(NCCL_REINIT, "===============restoring rank %d on rank %d===============", rank, comm->rank);
  cudaSetDevice(comm->localRank);
  if (comm->rank == rank) {
    cudaDeviceReset();
    INFO(NCCL_REINIT, "reset GPU state on rank %d done", rank);
    // Stage 1: init device for new Device
    initDeviceFromComm(comm);
    INFO(NCCL_REINIT, "restoring GPU object on rank %d done", rank);
  }

  // Stage 2: connect to peers
  // TODO: free old connections gracefully
  bool resetup = false;
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    if (comm->rank == rank) {
      resetup = true;
      NCCLCHECK(ncclTransportP2pForcedConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0));
      INFO(NCCL_REINIT, "channel %d rank %d connect: %d -> %d -> %d", c, comm->rank, channel->ring.prev, comm->rank, channel->ring.next);
    } else {
      if (channel->ring.prev == rank) {
        resetup = true;
        NCCLCHECK(ncclTransportP2pForcedConnect(comm, c, 1, &channel->ring.prev, 0, NULL, 0));
        INFO(NCCL_REINIT, "channel %d rank %d connect: %d -> %d", c, comm->rank, channel->ring.prev, comm->rank);
      } else if (channel->ring.next == rank) {
        resetup = true;
        NCCLCHECK(ncclTransportP2pForcedConnect(comm, c, 0, NULL, 1, &channel->ring.next, 0));
        INFO(NCCL_REINIT, "channel %d rank %d connect: %d -> %d", c, comm->rank, comm->rank, channel->ring.next);
      }
    }
  }

  // Stage 3: devCommResetup
  if (resetup) {
    INFO(NCCL_REINIT, "resetup device comm on rank %d", comm->rank);
    ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0);
    INFO(NCCL_REINIT, "finished ncclTransportP2pSetup on rank %d", comm->rank);
    devCommResetup(comm);
    INFO(NCCL_REINIT, "finished devCommResetup on rank %d", comm->rank);
  }
  INFO(NCCL_REINIT, "=============restoring rank %d on rank %d done============", rank, comm->rank);
  return ncclSuccess;
}
