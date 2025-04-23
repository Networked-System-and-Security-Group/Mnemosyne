#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <time.h>

#define CUDACHECK(cmd)                                           \
    do {                                                         \
        cudaError_t err = cmd;                                   \
        if (err != cudaSuccess) {                                \
            printf("Failed: Cuda error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

#define NCCLCHECK(cmd)                                           \
    do {                                                         \
        ncclResult_t res = cmd;                                  \
        if (res != ncclSuccess) {                                \
            printf("Failed, NCCL error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(int argc, char *argv[]) {
    ncclComm_t comms[3];
    int nDev = 3;
    int size = 1024;
    float value = 0.0;
    int rmDev1 = 1;
    struct timespec start, end;
    long seconds, nanoseconds;

    // allocating and initializing device buffers
    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
        value += 1.0;
        CUDACHECK(cudaMemcpy(&sendbuff[i][0], &value, sizeof(float), cudaMemcpyHostToDevice));
    }

    // initializing NCCL
    clock_gettime(CLOCK_MONOTONIC, &start);
    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(comms + i, nDev, uniqueId, i));
    }
    NCCLCHECK(ncclGroupEnd());
    clock_gettime(CLOCK_MONOTONIC, &end);

    {
        seconds = end.tv_sec - start.tv_sec;
        nanoseconds = end.tv_nsec - start.tv_nsec;
        long elapsed = (seconds * 1000000) + (nanoseconds / 1000);
        printf("INIT TIME: %ld us\n", elapsed);
    }

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    // since all comms here are configured as blocking, this is not necessarily needed (I guess)
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // show all-reduce result
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
        printf("ncclAllReduce Success, sum result of rank %d: %f\n", i, value);
    }

    // removing a rank
    clock_gettime(CLOCK_MONOTONIC, &start);
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclRemoveRank(comms[i], rmDev1));
    }
    NCCLCHECK(ncclGroupEnd());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("\nRank %d removed!\n", rmDev1);

    {
        seconds = end.tv_sec - start.tv_sec;
        nanoseconds = end.tv_nsec - start.tv_nsec;
        long elapsed = (seconds * 1000000) + (nanoseconds / 1000);
        printf("RM TIME: %ld us\n", elapsed);
    }

    // testing all-reduce after rank removal
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
      if (i == rmDev1) continue;
      NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < nDev; ++i) {
      if (i == rmDev1) continue;
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(s[i]));
    }
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
      printf("ncclAllReduce Success, sum result of rank %d: %f\n", i, value);
    }

    // free device buffers & finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }
    for (int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);

    printf("Success \n");
    return 0;
}
