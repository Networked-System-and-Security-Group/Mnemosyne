#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define CUDACHECK(cmd)                                                    \
    do {                                                                  \
        cudaError_t err = cmd;                                            \
        if (err != cudaSuccess) {                                         \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NCCLCHECK(cmd)                                                    \
    do {                                                                  \
        ncclResult_t res = cmd;                                           \
        if (res != ncclSuccess) {                                         \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   ncclGetErrorString(res));                              \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

int main(int argc, char *argv[]) {
    ncclComm_t comms[10];
    int nFirst = 0;
    int nTotal = 0;
    int rank_new = 0;
    int size = 1024;
    float value = 0.0;
    struct timespec start, end;
    long seconds, nanoseconds;

    if (argc == 1) {
        nFirst = 2;
        nTotal = 3;
        rank_new = 2;
    } else {
        sscanf(argv[argc - 1], "%d", &nTotal);
        nFirst = nTotal - 1;
        rank_new = nFirst;
    }
    printf("Prepare to add 1 rank to %d ranks (%d in total).\n", nFirst, nTotal);

    // allocating and initializing device buffers
    float **sendbuff = (float **)malloc(nTotal * sizeof(float *));
    float **recvbuff = (float **)malloc(nTotal * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nTotal);
    for (int i = 0; i < nTotal; ++i) {
        value += 1.0;
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
        CUDACHECK(cudaMemcpy(&sendbuff[i][0], &value, sizeof(float), cudaMemcpyHostToDevice));
    }

    // initializing NCCL
    clock_gettime(CLOCK_MONOTONIC, &start);
    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(comms + i, nFirst, uniqueId, i));
    }
    NCCLCHECK(ncclGroupEnd());
    clock_gettime(CLOCK_MONOTONIC, &end);

    {
        seconds = end.tv_sec - start.tv_sec;
        nanoseconds = end.tv_nsec - start.tv_nsec;
        long elapsed = (seconds * 1000000) + (nanoseconds / 1000);
        printf("INIT TIME: %ld us\n", elapsed);
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; ++i)
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    // since all comms here are configured as blocking, this is not necessarily needed (I guess)
    for (int i = 0; i < nFirst; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // show all-reduce result
    for (int i = 0; i < nFirst; ++i) {
        CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
        printf("ncclAllReduce Success, sum result of rank %d: %f\n", i, value);
    }

    ncclCommInfo *exportedInfo = (ncclCommInfo *)malloc(sizeof(ncclCommInfo));
    ncclNewRankInfo *newRankInfo = (ncclNewRankInfo *)malloc(sizeof(ncclNewRankInfo));

    // step 1: export communicator info from any healthy rank
    // executed by any old rank
    // new API i: ncclCommExportInfo
    NCCLCHECK(ncclCommExportInfo(comms[0], &uniqueId, exportedInfo));
    clock_gettime(CLOCK_MONOTONIC, &start);
    // step 2: init new rank with exported communicator info
    // rely on step 1; executed by the new rank
    CUDACHECK(cudaSetDevice(rank_new));
    // new API ii: ncclCommInitNewRank
    NCCLCHECK(ncclCommInitNewRank(comms + rank_new, exportedInfo, newRankInfo));

    // step 3: updated metadata with new rank's info for previous ranks
    // rely on step 2; executed by all old ranks
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; i++) {
        // new API iii: ncclCommAddNewRank
        NCCLCHECK(ncclCommAddNewRank(comms[i], newRankInfo));
    }
    NCCLCHECK(ncclGroupEnd());

    // step 4: setup connections for each rank
    // rely on step 3; executed by all ranks (including old ones & the new one)
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nTotal; i++) {
        // new API iv: ncclCommSetupNewRank
        NCCLCHECK(ncclCommSetupNewRank(comms[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    clock_gettime(CLOCK_MONOTONIC, &end);

    {
        seconds = end.tv_sec - start.tv_sec;
        nanoseconds = end.tv_nsec - start.tv_nsec;
        long elapsed = (seconds * 1000000) + (nanoseconds / 1000);
        printf("ADD TIME: %ld us\n", elapsed);
    }

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nTotal; ++i)
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i],
                                sizeof(float), ncclFloat, ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    // show all-reduce result
    for (int i = 0; i < nTotal; ++i) {
        CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
        printf("ncclAllReduce Success, sum result of rank %d: %f\n", i, value);
    }

    // free device buffers & finalizing NCCL
    for (int i = 0; i < nTotal; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }
    // for (int i = 0; i < 3; ++i)
    //   ncclCommDestroy(comms[i]);
    free(exportedInfo);
    free(newRankInfo);

    printf("Success \n");
    return 0;
}
