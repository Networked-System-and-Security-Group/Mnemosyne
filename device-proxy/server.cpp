#include <iostream>
#include <cstring>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <cuda.h>
#include <string>
#include <pthread.h>
#include <cublas_v2.h>
#include "cublas_v2.h"
#include <mutex>
#include <unordered_map>
#include <cublasLt.h>
#include <nccl.h>
#include <atomic>
#define SHARED_MEM_SIZE (1024 * 1024 * 1024)

#define printf(...) while (0)

#define CHECK_CUBLAS(call) {                                  \
    cublasStatus_t err = call;                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                      \
        fprintf(stderr, "cuBLAS error %d at line %d\n", err, __LINE__); \
        exit(EXIT_FAILURE);                                   \
    }                                                        \
}
constexpr size_t SHM_SIZE = 1024 * 1024 * 1024;
constexpr size_t BUFFER_SIZE = SHM_SIZE - sizeof(std::atomic<size_t>) * 2;

static size_t get_nccl_type_size(ncclDataType_t datatype) {
    switch(datatype) {
        case ncclInt8:    return 1;
        case ncclUint8:   return 1;
        case ncclInt32:   return 4;
        case ncclUint32:  return 4;
        case ncclInt64:   return 8;
        case ncclUint64:  return 8;
        case ncclFloat16: return 2;
        case ncclFloat32: return 4;
        case ncclFloat64: return 8;
        case ncclBfloat16: return 2;
        default:
            fprintf(stderr, "Unsupported datatype: %d\n", datatype);
            exit(EXIT_FAILURE);
    }
}

enum CudaApiType : unsigned long long {

    cudaThreadExitType = 1,
    cudaThreadGetCacheConfigType = 2,
    cudaThreadGetLimitType = 3,
    cudaThreadSetCacheConfigType = 4,
    cudaThreadSetLimitType = 5,
    cudaThreadSynchronizeType = 6,

    cudaGetErrorStringType = 7,
    cudaGetLastErrorType = 8,
    cudaPeekAtLastErrorType = 9,
    cudaChooseDeviceType = 10,
    cudaGetDeviceType = 11,
    cudaGetDeviceCountType = 12,
    cudaGetDevicePropertiesType = 13,
    cudaSetDeviceType = 14,
    cudaSetDeviceFlagsType = 15,
    cudaSetValidDevicesType = 16,

    cudaStreamCreateType = 17,
    cudaStreamDestroyType = 18,
    cudaStreamQueryType = 19,
    cudaStreamSynchronizeType = 20,
    cudaStreamWaitEventType = 21,

    cudaEventCreateType = 22,
    cudaEventCreateWithFlagsType = 23,
    cudaEventDestroyType = 24,
    cudaEventElapsedTimeType = 25,
    cudaEventQueryType = 26,
    cudaEventRecordType = 27,
    cudaEventSynchronizeType = 28,

    cudaConfigureCallType = 29,
    cudaFuncGetAttributesType = 30,
    cudaFuncSetCacheConfigType = 31,
    cudaLaunchType = 32,
    cudaSetDoubleForDeviceType = 33,
    cudaSetDoubleForHostType = 34,
    cudaSetupArgumentType = 35,

    cudaFreeType = 36,
    cudaFreeArrayType = 37,
    cudaFreeHostType = 38,
    cudaGetSymbolAddressType = 39,
    cudaGetSymbolSizeType = 40,
    cudaHostAllocType = 41,
    cudaHostGetDevicePointerType = 42,
    cudaHostGetFlagsType = 43,
    cudaMallocType = 44,
    cudaMalloc3DType = 45,
    cudaMalloc3DArrayType = 46,
    cudaMallocArrayType = 47,
    cudaMallocHostType = 48,
    cudaMallocPitchType = 49,
    cudaMemcpyType = 50,
    cudaMemcpy2DType = 51,
    cudaMemcpy2DArrayToArrayType = 52,
    cudaMemcpy2DAsyncType = 53,
    cudaMemcpy2DFromArrayType = 54,
    cudaMemcpy2DFromArrayAsyncType = 55,
    cudaMemcpy2DToArrayType = 56,
    cudaMemcpy2DToArrayAsyncType = 57,
    cudaMemcpy3DType = 58,
    cudaMemcpy3DAsyncType = 59,
    cudaMemcpyArrayToArrayType = 60,
    cudaMemcpyAsyncType = 61,
    cudaMemcpyFromArrayType = 62,
    cudaMemcpyFromArrayAsyncType = 63,
    cudaMemcpyFromSymbolType = 64,
    cudaMemcpyFromSymbolAsyncType = 65,
    cudaMemcpyToArrayType = 66,
    cudaMemcpyToArrayAsyncType = 67,
    cudaMemcpyToSymbolType = 68,
    cudaMemcpyToSymbolAsyncType = 69,
    cudaMemGetInfoType = 70,
    cudaMemsetType = 71,
    cudaMemset2DType = 72,
    cudaMemset2DAsyncType = 73,
    cudaMemset3DType = 74,
    cudaMemset3DAsyncType = 75,
    cudaMemsetAsyncType = 76,
    make_cudaExtentType = 77,
    make_cudaPitchedPtrType = 78,
    make_cudaPosType = 79,

    cudaDeviceFlushGPUDirectRDMAWritesType = 80,
    cudaDeviceGetAttributeType = 81,
    cudaDeviceGetByPCIBusIdType = 82,
    cudaDeviceGetCacheConfigType = 83,
    cudaDeviceGetDefaultMemPoolType = 84,
    cudaDeviceGetLimitType = 85,
    cudaDeviceGetMemPoolType = 86,
    cudaDeviceGetNvSciSyncAttributesType = 87,
    cudaDeviceGetP2PAttributeType = 88,
    cudaDeviceGetPCIBusIdType = 89,
    cudaDeviceGetStreamPriorityRangeType = 90,
    cudaDeviceGetTexture1DLinearMaxWidthType = 91,
    cudaDeviceRegisterAsyncNotificationType = 92,
    cudaDeviceResetType = 93,
    cudaDeviceSetCacheConfigType = 94,
    cudaDeviceSetLimitType = 95,
    cudaDeviceSetMemPoolType = 96,
    cudaDeviceSynchronizeType = 97,
    cudaDeviceUnregisterAsyncNotificationType = 98,
    cudaInitDeviceType = 99,
    cudaIpcCloseMemHandleType = 100,
    cudaIpcGetEventHandleType = 101,
    cudaIpcGetMemHandleType = 102,
    cudaIpcOpenEventHandleType = 103,
    cudaIpcOpenMemHandleType = 104,
    cudaCtxResetPersistingL2CacheType = 105,
    cudaStreamAddCallbackType = 106,
    cudaStreamAttachMemAsyncType = 107,
    cudaStreamBeginCaptureType = 108,
    cudaStreamBeginCaptureToGraphType = 109,
    cudaStreamCopyAttributesType = 110,
    cudaStreamCreateWithFlagsType = 111,
    cudaStreamCreateWithPriorityType = 112,
    cudaStreamEndCaptureType = 113,
    cudaStreamGetAttributeType = 114,
    cudaStreamGetCaptureInfoType = 115,
    cudaStreamGetCaptureInfo_v3Type = 116,
    cudaStreamGetFlagsType = 117,
    cudaStreamGetIdType = 118,
    cudaStreamGetPriorityType = 119,
    cudaStreamIsCapturingType = 120,
    cudaStreamSetAttributeType = 121,
    cudaStreamUpdateCaptureDependenciesType = 122,
    cudaStreamUpdateCaptureDependencies_v2Type = 123,
    cudaThreadExchangeStreamCaptureModeType = 124,
    cudaFuncGetNameType = 125,
    cudaFuncGetParamInfoType = 126,
    cudaFuncSetAttributeType = 127,

    cudaLaunchCooperativeKernelType = 128,
    cudaLaunchCooperativeKernelMultiDeviceType = 129,

    cudaLaunchHostFuncType = 130,
    cudaLaunchKernelType = 131,
    cudaLaunchKernelExCType = 132,

    cudaDestroyExternalMemoryType = 133,
    cudaDestroyExternalSemaphoreType = 134,
    cudaExternalMemoryGetMappedBufferType = 135,
    cudaExternalMemoryGetMappedMipmappedArrayType = 136,
    cudaImportExternalMemoryType = 137,
    cudaImportExternalSemaphoreType = 138,
    cudaSignalExternalSemaphoresAsyncType = 139,
    cudaWaitExternalSemaphoresAsyncType = 140,
    cudaOccupancyAvailableDynamicSMemPerBlockType = 141,
    cudaOccupancyMaxActiveBlocksPerMultiprocessorType = 142,
    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsType = 143,
    cudaOccupancyMaxActiveClustersType = 144,
    cudaOccupancyMaxPotentialClusterSizeType = 145,

    cublasSgemmType = 146,
    cublasCreateType = 147,
    cublasSetStreamType = 148,
    cublasDestroyType = 149,
    cublasGetPropertyType = 150,
    cublasSetWorkspaceType = 151,
    cublasGetStreamType = 152,
    cublasGetPointerModeType = 153,
    cublasSetPointerModeType = 154,
    cublasSetVectorType = 155,
    cublasGetVectorType = 156,
    cublasSetMatrixType = 157,
    cublasGetMatrixType = 158,
    cublasSetVectorAsyncType = 159,
    cublasGetVectorAsyncType = 160,
    cublasSetMatrixAsyncType = 161,
    cublasGetMatrixAsyncType = 162,
    cublasSetAtomicsModeType = 163,
    cublasGetAtomicsModeType = 164,
    cublasSetMathModeType = 165,
    cublasGetMathModeType = 166,
    cublasSetSmCountTargetType = 167,
    cublasGetSmCountTargetType = 168,
    cublasLoggerConfigureType = 169,
    cublasGetLoggerCallbackType = 170,
    cublasSetLoggerCallbackType = 171,

    cublasLtCreateType = 172,
    cublasLtMatmulAlgoCheckType = 173,
    cublasLtDestroyType = 174,
    cublasLtGetPropertyType = 175,
    cublasLtGetStatusNameType = 176,
    cublasLtGetStatusStringType = 177,
    cublasLtHeuristicsCacheGetCapacityType = 178,
    cublasLtHeuristicsCacheSetCapacityType = 179,
    cublasLtGetVersionType = 180,
    cublasLtDisableCpuInstructionsSetMaskType = 181,
    cublasLtGetCudartVersionType = 182,
    cublasLtLoggerSetCallbackType = 183,
    cublasLtLoggerSetFileType = 184,
    cublasLtLoggerOpenFileType = 185,
    cublasLtLoggerSetLevelType = 186,
    cublasLtLoggerSetMaskType = 187,
    cublasLtLoggerForceDisableType = 188,
    cublasLtMatmulType = 189,
    cublasLtMatmulAlgoCapGetAttributeType = 190,
    cublasLtMatmulAlgoConfigGetAttributeType = 191,
    cublasLtMatmulAlgoConfigSetAttributeType = 192,
    cublasLtMatmulAlgoGetHeuristicType = 193,
    cublasLtMatmulAlgoGetIdsType = 194,
    cublasLtMatmulAlgoInitType = 195,
    cublasLtMatmulDescCreateType = 196,
    cublasLtMatmulDescInitType = 197,
    cublasLtMatmulDescDestroyType = 198,
    cublasLtMatmulDescGetAttributeType = 199,
    cublasLtMatmulDescSetAttributeType = 200,
    cublasLtMatmulPreferenceCreateType = 201,
    cublasLtMatmulPreferenceInitType = 202,
    cublasLtMatmulPreferenceDestroyType = 203,
    cublasLtMatmulPreferenceGetAttributeType = 204,
    cublasLtMatmulPreferenceSetAttributeType = 205,
    cublasLtMatrixLayoutCreateType = 206,
    cublasLtMatrixLayoutInitType = 207,
    cublasLtMatrixLayoutDestroyType = 208,
    cublasLtMatrixLayoutGetAttributeType = 209,
    cublasLtMatrixLayoutSetAttributeType = 210,
    cublasLtMatrixTransformType = 211,
    cublasLtMatrixTransformDescCreateType = 212,
    cublasLtMatrixTransformDescInitType = 213,
    cublasLtMatrixTransformDescDestroyType = 214,
    cublasLtMatrixTransformDescGetAttributeType = 215,
    cublasLtMatrixTransformDescSetAttributeType = 216,
    ncclGetLastErrorType = 217,
    ncclGetErrorStringType = 218,
    ncclGetVersionType = 219,
    ncclGetUniqueIdType = 220,
    ncclCommInitRankType = 221,
    ncclCommInitAllType = 222,
    ncclCommInitRankConfigType = 223,
    ncclCommInitRankScalableType = 224,
    ncclCommSplitType = 225,
    ncclCommFinalizeType = 226,
    ncclCommDestroyType = 227,
    ncclCommAbortType = 228,
    ncclCommGetAsyncErrorType = 229,
    ncclCommCountType = 230,
    ncclCommCuDeviceType = 231,
    ncclCommUserRankType = 232,
    ncclCommRegisterType = 233,
    ncclCommDeregisterType = 234,
    ncclMemAllocType = 235,
    ncclMemFreeType = 236,
    ncclAllReduceType = 237,
    ncclBroadcastType = 238,
    ncclBcastType = 239,
    ncclReduceType = 240,
    ncclAllGatherType = 241,
    ncclReduceScatterType = 242,
    ncclGroupStartType = 243,
    ncclGroupEndType = 244,
    ncclGroupSimulateEndType = 245,
    ncclSendType = 246,
    ncclRecvType = 247,
    ncclRedOpCreatePreMulSumType = 248,
    ncclRedOpDestroyType = 249,
    cublasSgemmStridedBatchedType = 250,
};

struct CudaRequestBase {
    size_t requestSize;
    CudaApiType type;
};
struct CudaResponseBase {
    size_t responseSize;
    cudaError_t result;
    cublasStatus_t status;
    ncclResult_t ncclResult;
};

struct cudaThreadExitRq : public CudaRequestBase {};
struct cudaThreadGetCacheConfigRq : public CudaRequestBase {
    enum cudaFuncCache *pCacheConfig;
};
struct cudaThreadGetCacheConfigRp : public CudaResponseBase {
    enum cudaFuncCache cacheConfig;
};
struct cudaThreadGetLimitRq : public CudaRequestBase {
    size_t Value;
    enum cudaLimit limit;
};
struct cudaThreadSetCacheConfigRq : public CudaRequestBase {
    enum cudaFuncCache cacheConfig;
};
struct cudaThreadSetLimitRq : public CudaRequestBase {
    enum cudaLimit limit;
    size_t value;
};
struct cudaThreadSynchronizeRq : public CudaRequestBase {};

struct cudaGetErrorStringRq : public CudaRequestBase {
    cudaError_t error;
};
struct cudaGetErrorStringRp : public CudaResponseBase {
    const char *result;
};
struct cudaGetLastErrorRq : public CudaRequestBase {};
struct cudaPeekAtLastErrorRq : public CudaRequestBase {};

struct cudaChooseDeviceRq : public CudaRequestBase {
    struct cudaDeviceProp prop;
};
struct cudaChooseDeviceRp : public CudaResponseBase {
    int device;
};
struct cudaGetDeviceRq : public CudaRequestBase {};
struct cudaGetDeviceRp : public CudaResponseBase {
    int device;
};
struct cudaGetDeviceCountRq : public CudaRequestBase {};
struct cudaGetDeviceCountRp : public CudaResponseBase {
    int count;
};
struct cudaGetDevicePropertiesRq : public CudaRequestBase {

    int device;
};
struct cudaGetDevicePropertiesRp : public CudaResponseBase {
    struct cudaDeviceProp prop;
};
struct cudaSetDeviceRq : public CudaRequestBase {
    int device;
};
struct cudaSetDeviceFlagsRq : public CudaRequestBase {
    unsigned int flags;
};
struct cudaSetValidDevicesRq : public CudaRequestBase {
    int *device_arr;
    int len;
};

struct cudaStreamCreateRq : public CudaRequestBase {};
struct cudaStreamCreateRp : public CudaResponseBase {
    cudaStream_t stream;
};
struct cudaStreamDestroyRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaStreamQueryRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaStreamSynchronizeRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaStreamWaitEventRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaEvent_t event;
    unsigned int flags;
};

struct cudaEventCreateRq : public CudaRequestBase {};
struct cudaEventCreateRp : public CudaResponseBase {
    cudaEvent_t event;
};
struct cudaEventQueryRq : public CudaRequestBase {
    cudaEvent_t event;
};
struct cudaEventCreateWithFlagsRq : public CudaRequestBase {
    unsigned int flags;
};
struct cudaEventCreateWithFlagsRp : public CudaResponseBase {
    cudaEvent_t event;
};
struct cudaEventDestroyRq : public CudaRequestBase {
    cudaEvent_t event;
};
struct cudaEventElapsedTimeRq : public CudaRequestBase {
    cudaEvent_t start;
    cudaEvent_t end;
};
struct cudaEventElapsedTimeRp : public CudaResponseBase {
    float ms;
};
struct cudaEventRecordRq : public CudaRequestBase {
    cudaEvent_t event;
    cudaStream_t stream;
};
struct cudaEventSynchronizeRq : public CudaRequestBase {
    cudaEvent_t event;
};

struct cudaFuncGetAttributesRq : public CudaRequestBase {
    const void *func;
};
struct cudaFuncGetAttributesRp : public CudaResponseBase {
    struct cudaFuncAttributes attr;
};
struct cudaFuncSetCacheConfigRq : public CudaRequestBase {
    const void *func;
    enum cudaFuncCache cacheConfig;
};
struct cudaSetDoubleForDeviceRq : public CudaRequestBase {};
struct cudaSetDoubleForDeviceRp : public CudaResponseBase {
    double d;
};
struct cudaSetDoubleForHostRq : public CudaRequestBase {};
struct cudaSetDoubleForHostRp : public CudaResponseBase {
    double d;
};

struct cudaDestroyExternalMemoryRq : public CudaRequestBase {
    cudaExternalMemory_t extMem;
};
struct cudaDestroyExternalMemoryRp : public CudaResponseBase {

};
struct cudaDestroyExternalSemaphoreRq : public CudaRequestBase {
    cudaExternalSemaphore_t extSem;
};
struct cudaDestroyExternalSemaphoreRp : public CudaResponseBase {

};
struct cudaExternalMemoryGetMappedBufferRq : public CudaRequestBase {
    cudaExternalMemory_t extMem;
    cudaExternalMemoryBufferDesc bufferDesc;
};
struct cudaExternalMemoryGetMappedBufferRp : public CudaResponseBase {

    void *devPtr;
};
struct cudaExternalMemoryGetMappedMipmappedArrayRq : public CudaRequestBase {
    cudaExternalMemory_t extMem;
    cudaExternalMemoryMipmappedArrayDesc mipmapDesc;
};
struct cudaExternalMemoryGetMappedMipmappedArrayRp : public CudaResponseBase {

    cudaMipmappedArray_t mipmap;
};
struct cudaImportExternalMemoryRq : public CudaRequestBase {
    cudaExternalMemory_t *extMem_out;
    cudaExternalMemoryHandleDesc memHandleDesc;

};
struct cudaImportExternalMemoryRp : public CudaResponseBase {

};
struct cudaImportExternalSemaphoreRq : public CudaRequestBase {
    cudaExternalSemaphore_t *extSem_out;
    cudaExternalSemaphoreHandleDesc semHandleDesc;
};
struct cudaImportExternalSemaphoreRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaSignalExternalSemaphoresAsyncRq : public CudaRequestBase {
    const cudaExternalSemaphore_t *extSemArray;
    const cudaExternalSemaphoreSignalParams *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
};
struct cudaSignalExternalSemaphoresAsyncRp : public CudaResponseBase {

};
struct cudaWaitExternalSemaphoresAsyncRq : public CudaRequestBase {
    const cudaExternalSemaphore_t *extSemArray;
    const cudaExternalSemaphoreWaitParams *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
};
struct cudaWaitExternalSemaphoresAsyncRp : public CudaResponseBase {

};

struct cudaCtxResetPersistingL2CacheRq : public CudaRequestBase {};
struct cudaStreamAddCallbackRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaStreamCallback_t callback;
    void *userData;
    unsigned int flags;
};
struct cudaStreamAttachMemAsyncRq : public CudaRequestBase {
    cudaStream_t stream;
    void *devPtr;
    size_t length = 0;
    unsigned int flags = cudaMemAttachSingle;
};
struct cudaStreamBeginCaptureRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaStreamCaptureMode mode;
};
struct cudaStreamBeginCaptureToGraphRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaGraph_t graph;
    const cudaGraphNode_t *dependencies;
    const cudaGraphEdgeData *dependencyData;
    size_t numDependencies;
    cudaStreamCaptureMode mode;
};
struct cudaStreamCopyAttributesRq : public CudaRequestBase {
    cudaStream_t dst;
    cudaStream_t src;
};
struct cudaStreamCreateWithFlagsRq : public CudaRequestBase {
    unsigned int flags;
};
struct cudaStreamCreateWithFlagsRp : public CudaResponseBase {

    cudaStream_t stream;
};
struct cudaStreamCreateWithPriorityRq : public CudaRequestBase {
    unsigned int flags;
    int priority;
};
struct cudaStreamCreateWithPriorityRp : public CudaResponseBase {

    cudaStream_t stream;
};
struct cudaStreamEndCaptureRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaStreamEndCaptureRp : public CudaResponseBase {

    cudaGraph_t graph;
};
struct cudaStreamGetAttributeRq : public CudaRequestBase {
    cudaStream_t hStream;
    cudaStreamAttrID attr;
};
struct cudaStreamGetAttributeRp : public CudaResponseBase {

    cudaStreamAttrValue value;
};
struct cudaStreamGetCaptureInfoRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaStreamGetCaptureInfoRp : public CudaResponseBase {

    cudaStreamCaptureStatus captureStatus;
    unsigned long long id;
    cudaGraph_t graph;
    const cudaGraphNode_t *dependencies;
    size_t numDependencies;
};
struct cudaStreamGetCaptureInfo_v3Rq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaStreamGetCaptureInfo_v3Rp : public CudaResponseBase {

    cudaStreamCaptureStatus captureStatus;
    unsigned long long id;
    cudaGraph_t graph;
    const cudaGraphNode_t *dependencies;
    const cudaGraphEdgeData *edgeData;
    size_t numDependencies;
};
struct cudaStreamGetFlagsRq : public CudaRequestBase {
    cudaStream_t hStream;
};
struct cudaStreamGetFlagsRp : public CudaResponseBase {

    unsigned int flags;
};
struct cudaStreamGetIdRq : public CudaRequestBase {
    cudaStream_t hStream;
};
struct cudaStreamGetIdRp : public CudaResponseBase {

    unsigned long long streamId;
};
struct cudaStreamGetPriorityRq : public CudaRequestBase {
    cudaStream_t hStream;
};
struct cudaStreamGetPriorityRp : public CudaResponseBase {

    int priority;
};
struct cudaStreamIsCapturingRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaStreamIsCapturingRp : public CudaResponseBase {

    cudaStreamCaptureStatus status;
};
struct cudaStreamSetAttributeRq : public CudaRequestBase {
    cudaStream_t hStream;
    cudaStreamAttrID attr;
    cudaStreamAttrValue value;
};
struct cudaStreamSetAttributeRp : public CudaResponseBase {

};
struct cudaStreamUpdateCaptureDependenciesRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaGraphNode_t *dependencies;
    size_t numDependencies;
    unsigned int flags;
};
struct cudaStreamUpdateCaptureDependenciesRp : public CudaResponseBase {

};
struct cudaStreamUpdateCaptureDependencies_v2Rq : public CudaRequestBase {
    cudaStream_t stream;
    cudaGraphNode_t *dependencies;
    const cudaGraphEdgeData *dependencyData;
    size_t numDependencies;
    unsigned int flags;
};
struct cudaStreamUpdateCaptureDependencies_v2Rp : public CudaResponseBase {

};
struct cudaThreadExchangeStreamCaptureModeRq : public CudaRequestBase {

};
struct cudaThreadExchangeStreamCaptureModeRp : public CudaResponseBase {

    cudaStreamCaptureMode mode;
};

struct cudaFuncGetNameRq : public CudaRequestBase {
    const void *func;
};
struct cudaFuncGetNameRp : public CudaResponseBase {

    const char **name;
};
struct cudaFuncGetParamInfoRq : public CudaRequestBase {
    const void *func;
    size_t paramIndex;
};
struct cudaFuncGetParamInfoRp : public CudaResponseBase {

    size_t paramOffset;
    size_t paramSize;
};
struct cudaFuncSetAttributeRq : public CudaRequestBase {
    const void *func;
    cudaFuncAttribute attr;
    int value;
};
struct cudaFuncSetAttributeRp : public CudaResponseBase {

};
struct cudaLaunchCooperativeKernelRq : public CudaRequestBase {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
};

struct cudaLaunchCooperativeKernelRp : public CudaResponseBase {

};
struct cudaLaunchCooperativeKernelMultiDeviceRq : public CudaRequestBase {
    cudaLaunchParams *launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
};
struct cudaLaunchCooperativeKernelMultiDeviceRp : public CudaResponseBase {

};
struct cudaLaunchHostFuncRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaHostFn_t fn;
    void *userData;
};
struct cudaLaunchHostFuncRp : public CudaResponseBase {

};
struct cudaLaunchKernelRq : public CudaRequestBase {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
    int argsSize;

    char *client_shared_mem;
    char* kernel_name_pointer;
    char* cubin_file_path_pointer;
};
struct cudaLaunchKernelRp : public CudaResponseBase {

};
struct cudaLaunchKernelExCRq : public CudaRequestBase {
    cudaLaunchConfig_t config;
    const void *func;
    void **args;
};
struct cudaLaunchKernelExCRp : public CudaResponseBase {

};

struct cudaDeviceFlushGPUDirectRDMAWritesRq : public CudaRequestBase {
    cudaFlushGPUDirectRDMAWritesTarget target;
    cudaFlushGPUDirectRDMAWritesScope scope;
};
struct cudaDeviceFlushGPUDirectRDMAWritesRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaDeviceGetAttributeRq : public CudaRequestBase {
    cudaDeviceAttr attr;
    int device;
};
struct cudaDeviceGetAttributeRp : public CudaResponseBase {
    cudaError_t result;
    int value;
};
struct cudaDeviceGetByPCIBusIdRq : public CudaRequestBase {
    char pciBusId[32];
};
struct cudaDeviceGetByPCIBusIdRp : public CudaResponseBase {
    cudaError_t result;
    int device;
};
struct cudaDeviceGetCacheConfigRq : public CudaRequestBase {

};
struct cudaDeviceGetCacheConfigRp : public CudaResponseBase {
    cudaError_t result;
    cudaFuncCache cacheConfig;
};
struct cudaDeviceGetDefaultMemPoolRq : public CudaRequestBase {
    int device;
};
struct cudaDeviceGetDefaultMemPoolRp : public CudaResponseBase {
    cudaError_t result;
    cudaMemPool_t memPool;
};
struct cudaDeviceGetLimitRq : public CudaRequestBase {
    cudaLimit limit;
};
struct cudaDeviceGetLimitRp : public CudaResponseBase {
    cudaError_t result;
    size_t value;
};
struct cudaDeviceGetMemPoolRq : public CudaRequestBase {
    int device;
};
struct cudaDeviceGetMemPoolRp : public CudaResponseBase {
    cudaError_t result;
    cudaMemPool_t memPool;
};
struct cudaDeviceGetNvSciSyncAttributesRq : public CudaRequestBase {
    int device;
    int flags;

};
struct cudaDeviceGetNvSciSyncAttributesRp : public CudaResponseBase {

    void *nvSciSyncAttrList;
};
struct cudaDeviceGetP2PAttributeRq : public CudaRequestBase {
    cudaDeviceP2PAttr attr;
    int srcDevice;
    int dstDevice;
};
struct cudaDeviceGetP2PAttributeRp : public CudaResponseBase {
    cudaError_t result;
    int value;
};
struct cudaDeviceGetPCIBusIdRq : public CudaRequestBase {
    int device;
    int len;
};
struct cudaDeviceGetPCIBusIdRp : public CudaResponseBase {
    cudaError_t result;
    char pciBusId[32];
};
struct cudaDeviceGetStreamPriorityRangeRq : public CudaRequestBase {

};
struct cudaDeviceGetStreamPriorityRangeRp : public CudaResponseBase {
    cudaError_t result;
    int leastPriority;
    int greatestPriority;
};
struct cudaDeviceGetTexture1DLinearMaxWidthRq : public CudaRequestBase {
    cudaChannelFormatDesc fmtDesc;
    int device;
};
struct cudaDeviceGetTexture1DLinearMaxWidthRp : public CudaResponseBase {
    cudaError_t result;
    size_t maxWidthInElements;
};
struct cudaDeviceRegisterAsyncNotificationRq : public CudaRequestBase {
    int device;
    cudaAsyncCallback callbackFunc;
    void *userData;
};
struct cudaDeviceRegisterAsyncNotificationRp : public CudaResponseBase {
    cudaError_t result;
    cudaAsyncCallbackHandle_t callback;
};
struct cudaDeviceResetRq : public CudaRequestBase {

};
struct cudaDeviceResetRp : public CudaResponseBase {};

struct cudaDeviceSetCacheConfigRq : public CudaRequestBase {
    cudaFuncCache cacheConfig;
};
struct cudaDeviceSetCacheConfigRp : public CudaResponseBase {};
struct cudaDeviceSetLimitRq : public CudaRequestBase {
    cudaLimit limit;
    size_t value;
};
struct cudaDeviceSetLimitRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaDeviceSetMemPoolRq : public CudaRequestBase {
    int device;
    cudaMemPool_t memPool;
};
struct cudaDeviceSetMemPoolRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaDeviceSynchronizeRq : public CudaRequestBase {

};
struct cudaDeviceSynchronizeRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaDeviceUnregisterAsyncNotificationRq : public CudaRequestBase {
    int device;
    cudaAsyncCallbackHandle_t callback;
};
struct cudaDeviceUnregisterAsyncNotificationRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaInitDeviceRq : public CudaRequestBase {
    int device;
    unsigned int deviceFlags;
    unsigned int flags;
};
struct cudaInitDeviceRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaIpcCloseMemHandleRq : public CudaRequestBase {
    void *devPtr;
};
struct cudaIpcCloseMemHandleRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaIpcGetEventHandleRq : public CudaRequestBase {
    cudaEvent_t event;
};
struct cudaIpcGetEventHandleRp : public CudaResponseBase {
    cudaError_t result;
    cudaIpcEventHandle_t handle;
};
struct cudaIpcGetMemHandleRq : public CudaRequestBase {
    void *devPtr;
};
struct cudaIpcGetMemHandleRp : public CudaResponseBase {
    cudaError_t result;
    cudaIpcMemHandle_t handle;
};
struct cudaIpcOpenEventHandleRq : public CudaRequestBase {
    cudaIpcEventHandle_t handle;
};
struct cudaIpcOpenEventHandleRp : public CudaResponseBase {
    cudaError_t result;
    cudaEvent_t event;
};
struct cudaIpcOpenMemHandleRq : public CudaRequestBase {
    cudaIpcMemHandle_t handle;
    unsigned int flags;
};
struct cudaIpcOpenMemHandleRp : public CudaResponseBase {
    cudaError_t result;
    void *devPtr;
};

struct cudaFreeRq : public CudaRequestBase {
    void *devPtr;
};
struct cudaFreeArrayRq : public CudaRequestBase {
    struct cudaArray *array;
};
struct cudaFreeHostRq : public CudaRequestBase {
    void *ptr;
};
struct cudaGetSymbolAddressRq : public CudaRequestBase {
    const void *symbol;
};
struct cudaGetSymbolAddressRp : public CudaResponseBase {
    void *devPtr;
};
struct cudaGetSymbolSizeRq : public CudaRequestBase {
    const void *symbol;
};
struct cudaGetSymbolSizeRp : public CudaResponseBase {
    size_t size;
};
struct cudaHostAllocRq : public CudaRequestBase {
    size_t size;
    unsigned int flags;
    size_t shmem_offset;
};
struct cudaHostAllocRp : public CudaResponseBase {
    void *pHost;
};
struct cudaHostGetDevicePointerRq : public CudaRequestBase {
    void *pHost;
    unsigned int flags;
};
struct cudaHostGetDevicePointerRp : public CudaResponseBase {
    void *pDevice;
};
struct cudaHostGetFlagsRq : public CudaRequestBase {

    void *pHost;
};
struct cudaHostGetFlagsRp : public CudaResponseBase {
    unsigned int pFlags;
};
struct cudaMallocRq : public CudaRequestBase {
    size_t size;
};
struct cudaMallocRp : public CudaResponseBase {
    void *devPtr;
};
struct cudaMalloc3DRq : public CudaRequestBase {
    struct cudaExtent extent;
};
struct cudaMalloc3DRp : public CudaResponseBase {
    struct cudaPitchedPtr *pitchedDevPtr;
};
struct cudaMalloc3DArrayRq : public CudaRequestBase {
    const struct cudaChannelFormatDesc *desc;
    struct cudaExtent extent;
    unsigned int flags;
};
struct cudaMalloc3DArrayRp : public CudaResponseBase {
    struct cudaArray *array;
};
struct cudaMallocArrayRq : public CudaRequestBase {
    const struct cudaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    unsigned int flags;
};
struct cudaMallocArrayRp : public CudaResponseBase {
    struct cudaArray *array;
};
struct cudaMallocHostRq : public CudaRequestBase {
    size_t size;
};
struct cudaMallocHostRp : public CudaResponseBase {
    void *ptr;
};
struct cudaMallocPitchRq : public CudaRequestBase {
    size_t width;
    size_t height;
};
struct cudaMallocPitchRp : public CudaResponseBase {
    size_t pitch;
    void *devPtr;
};
struct cudaMemcpyRq : public CudaRequestBase {
    void *dst;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    char *client_shared_mem;
    size_t shmem_offset;
};
struct cudaMemcpy2DRq : public CudaRequestBase {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DArrayToArrayRq : public CudaRequestBase {
    struct cudaArray *dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    const struct cudaArray *src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DAsyncRq : public CudaRequestBase {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpy2DFromArrayRq : public CudaRequestBase {
    void *dst;
    size_t dpitch;
    const struct cudaArray *src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DFromArrayAsyncRq : public CudaRequestBase {
    void *dst;
    size_t dpitch;
    const struct cudaArray *src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpy2DToArrayRq : public CudaRequestBase {
    struct cudaArray *dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DToArrayAsyncRq : public CudaRequestBase {
    struct cudaArray *dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpy3DRq : public CudaRequestBase {};
struct cudaMemcpy3DRp : public CudaResponseBase {
    const struct cudaMemcpy3DParms *p;
};
struct cudaMemcpy3DAsyncRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaMemcpy3DAsyncRp : public CudaResponseBase {
    const struct cudaMemcpy3DParms *p;
};
struct cudaMemcpyArrayToArrayRq : public CudaRequestBase {
    struct cudaArray *dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    const struct cudaArray *src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyAsyncRq : public CudaRequestBase {
    void *dst;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    char *client_shared_mem;
    size_t shmem_offset;
};
struct cudaMemcpyFromArrayRq : public CudaRequestBase {
    void *dst;
    const struct cudaArray *src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyFromArrayAsyncRq : public CudaRequestBase {
    void *dst;
    const struct cudaArray *src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpyFromSymbolRq : public CudaRequestBase {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyFromSymbolAsyncRq : public CudaRequestBase {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpyToArrayRq : public CudaRequestBase {
    struct cudaArray *dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyToArrayAsyncRq : public CudaRequestBase {
    struct cudaArray *dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpyToSymbolRq : public CudaRequestBase {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyToSymbolAsyncRq : public CudaRequestBase {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemGetInfoRq : public CudaRequestBase {};
struct cudaMemGetInfoRp : public CudaResponseBase {
    size_t total;
    size_t free;
};
struct cudaMemsetRq : public CudaRequestBase {
    void *devPtr;
    int value;
    size_t count;
};
struct cudaMemset2DRq : public CudaRequestBase {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
};
struct cudaMemset2DAsyncRq : public CudaRequestBase {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    cudaStream_t stream;
};
struct cudaMemset3DRq : public CudaRequestBase {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
};
struct cudaMemset3DAsyncRq : public CudaRequestBase {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
    cudaStream_t stream;
};
struct cudaMemsetAsyncRq : public CudaRequestBase {
    void *devPtr;
    int value;
    size_t count;
    cudaStream_t stream;
};

struct make_cudaExtentRq : public CudaRequestBase {
    size_t w;
    size_t h;
    size_t d;
};
struct make_cudaExtentRp {
    size_t responseSize;
    struct cudaExtent result;
};
struct make_cudaPitchedPtrRq : public CudaRequestBase {
    void *d;
    size_t p;
    size_t xsz;
    size_t ysz;
};
struct make_cudaPitchedPtrRp {
    size_t responseSize;
    struct cudaPitchedPtr result;
};
struct make_cudaPosRq : public CudaRequestBase {
    size_t x;
    size_t y;
    size_t z;
};
struct make_cudaPosRp {
    size_t responseSize;
    struct cudaPos result;
};

struct cudaOccupancyAvailableDynamicSMemPerBlockRq : public CudaRequestBase {
    const void *func;
    int numBlocks;
    int blockSize;
};
struct cudaOccupancyAvailableDynamicSMemPerBlockRp : public CudaResponseBase {
    size_t dynamicSmemSize;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorRq : public CudaRequestBase {
    const void *func;
    int blockSize;
    size_t dynamicSMemSize;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorRp : public CudaResponseBase {
    int numBlocks;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRq : public CudaRequestBase {
    const void *func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRp : public CudaResponseBase {
    int numBlocks;
};
struct cudaOccupancyMaxActiveClustersRq : public CudaRequestBase {
    const void *func;
    cudaLaunchConfig_t launchConfig;
};
struct cudaOccupancyMaxActiveClustersRp : public CudaResponseBase {
    int numClusters;
};
struct cudaOccupancyMaxPotentialClusterSizeRq : public CudaRequestBase {
    const void *func;
    cudaLaunchConfig_t launchConfig;
};
struct cudaOccupancyMaxPotentialClusterSizeRp : public CudaResponseBase {
    cudaError_t result;
    int clusterSize;
};
struct cublasSgemmRq : public CudaRequestBase {
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    float alpha;
    const float *A;
    int lda;
    const float *B;
    int ldb;
    float beta;
    float *C;
    int ldc;
    char *client_shared_mem;
};
struct cublasSgemmRp : public CudaResponseBase {};

struct cublasSgemmStridedBatchedRq : public CudaRequestBase {
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    float alpha;
    const float *A;
    int lda;
    long long int strideA;
    const float *B;
    int ldb;
    long long int strideB;
    float beta;
    float *C;
    int ldc;
    long long int strideC;
    int batchCount;
    char* client_shared_mem;
};

struct cublasSgemmStridedBatchedRp : public CudaResponseBase {
};

struct cublasCreateRq : public CudaRequestBase {};
struct cublasCreateRp : public CudaResponseBase {
    cublasHandle_t handle;
};
struct cublasSetStreamRq : public CudaRequestBase {
    cublasHandle_t handle;
    cudaStream_t streamId;
};

struct cublasSetStreamRp : public CudaResponseBase {
};
struct cublasDestroyRq : public CudaRequestBase {
    cublasHandle_t handle;
};

struct cublasDestroyRp : public CudaResponseBase {
};

struct cublasGetPropertyRq : public CudaRequestBase {
    libraryPropertyType type;
};

struct cublasGetPropertyRp : public CudaResponseBase {
    int value;
};

struct cublasSetWorkspaceRq : public CudaRequestBase {
    cublasHandle_t handle;
    void *workspace;
    size_t workspaceSizeInBytes;
};

struct cublasSetWorkspaceRp : public CudaResponseBase {
};

struct cublasGetStreamRq : public CudaRequestBase {
    cublasHandle_t handle;
};

struct cublasGetStreamRp : public CudaResponseBase {
    cudaStream_t streamId;
};

struct cublasGetPointerModeRq : public CudaRequestBase {
    cublasHandle_t handle;
};

struct cublasGetPointerModeRp : public CudaResponseBase {
    cublasPointerMode_t mode;
};

struct cublasSetPointerModeRq : public CudaRequestBase {
    cublasHandle_t handle;
    cublasPointerMode_t mode;
};

struct cublasSetPointerModeRp : public CudaResponseBase {
};

struct cublasSetVectorRq : public CudaRequestBase {
    int n;
    int elemSize;
    const void *x;
    int incx;
    void *y;
    int incy;
};

struct cublasSetVectorRp : public CudaResponseBase {
};

struct cublasGetVectorRq : public CudaRequestBase {
    int n;
    int elemSize;
    const void *x;
    int incx;
    void *y;
    int incy;
};

struct cublasGetVectorRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasSetMatrixRq : public CudaRequestBase {
    int rows;
    int cols;
    int elemSize;
    const void *A;
    int lda;
    void *B;
    int ldb;
};

struct cublasSetMatrixRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasGetMatrixRq : public CudaRequestBase {
    int rows;
    int cols;
    int elemSize;
    const void *A;
    int lda;
    void *B;
    int ldb;
};

struct cublasGetMatrixRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasSetVectorAsyncRq : public CudaRequestBase {
    int n;
    int elemSize;
    const void *hostPtr;
    int incx;
    void *devicePtr;
    int incy;
    cudaStream_t stream;
};

struct cublasSetVectorAsyncRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasGetVectorAsyncRq : public CudaRequestBase {
    int n;
    int elemSize;
    const void *devicePtr;
    int incx;
    void *hostPtr;
    int incy;
    cudaStream_t stream;
};

struct cublasGetVectorAsyncRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasSetMatrixAsyncRq : public CudaRequestBase {
    int rows;
    int cols;
    int elemSize;
    const void *A;
    int lda;
    void *B;
    int ldb;
    cudaStream_t stream;
};

struct cublasSetMatrixAsyncRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasGetMatrixAsyncRq : public CudaRequestBase {
    int rows;
    int cols;
    int elemSize;
    const void *A;
    int lda;
    void *B;
    int ldb;
    cudaStream_t stream;
};

struct cublasGetMatrixAsyncRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasSetAtomicsModeRq : public CudaRequestBase {
    cublasHandle_t handle;
    cublasAtomicsMode_t mode;
};

struct cublasSetAtomicsModeRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasGetAtomicsModeRq : public CudaRequestBase {
    cublasHandle_t handle;
};

struct cublasGetAtomicsModeRp : public CudaResponseBase {
    cublasStatus_t status;
    cublasAtomicsMode_t atomicsMode;
};

struct cublasSetMathModeRq : public CudaRequestBase {
    cublasHandle_t handle;
    cublasMath_t mode;
};

struct cublasSetMathModeRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasGetMathModeRq : public CudaRequestBase {
    cublasHandle_t handle;
};

struct cublasGetMathModeRp : public CudaResponseBase {
    cublasStatus_t status;
    cublasMath_t mathMode;
};

struct cublasSetSmCountTargetRq : public CudaRequestBase {
    cublasHandle_t handle;
    int smCountTarget;
};

struct cublasSetSmCountTargetRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasGetSmCountTargetRq : public CudaRequestBase {
    cublasHandle_t handle;
};

struct cublasGetSmCountTargetRp : public CudaResponseBase {
    cublasStatus_t status;
    int smCountTarget;
};

struct cublasLoggerConfigureRq : public CudaRequestBase {
    int logIsOn;
    int logToStdOut;
    int logToStdErr;
    const char* logFileName;
};

struct cublasLoggerConfigureRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasGetLoggerCallbackRq : public CudaRequestBase {

};

struct cublasGetLoggerCallbackRp : public CudaResponseBase {
    cublasStatus_t status;
    cublasLogCallback userCallback;
};

struct cublasSetLoggerCallbackRq : public CudaRequestBase {
    cublasLogCallback userCallback;
};

struct cublasSetLoggerCallbackRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasLtMatmulAlgoGetHeuristicRq : public CudaRequestBase {
    cublasLtHandle_t lightHandle;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Cdesc;
    cublasLtMatrixLayout_t Ddesc;
    cublasLtMatmulPreference_t preference;
    int requestedAlgoCount;
    size_t shared_mem_offset;
};

struct cublasLtMatmulAlgoGetHeuristicRp : public CudaResponseBase {

    int returnAlgoCount;

};
struct cublasLtCreateRq : public CudaRequestBase {

};

struct cublasLtCreateRp : public CudaResponseBase {
    cublasStatus_t status;
    cublasLtHandle_t handle;
};

struct cublasLtDestroyRq : public CudaRequestBase {
    cublasLtHandle_t lightHandle;
};

struct cublasLtDestroyRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasLtGetPropertyRq : public CudaRequestBase {
    libraryPropertyType type1;
};

struct cublasLtGetPropertyRp : public CudaResponseBase {
    cublasStatus_t status;
    int value;
};

struct cublasLtGetStatusNameRq : public CudaRequestBase {
    cublasStatus_t status;
};

struct cublasLtGetStatusNameRp : public CudaResponseBase {
    const char* statusName;
};

struct cublasLtGetStatusStringRq : public CudaRequestBase {
    cublasStatus_t status;
};

struct cublasLtGetStatusStringRp : public CudaResponseBase {
    const char* statusString;
};

struct cublasLtHeuristicsCacheGetCapacityRq : public CudaRequestBase {

};

struct cublasLtHeuristicsCacheGetCapacityRp : public CudaResponseBase {
    cublasStatus_t status;
    size_t capacity;
};

struct cublasLtHeuristicsCacheSetCapacityRq : public CudaRequestBase {
    size_t capacity;
};

struct cublasLtHeuristicsCacheSetCapacityRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtGetVersionRq : public CudaRequestBase {

};

struct cublasLtGetVersionRp : public CudaResponseBase {
    size_t version;
};

struct cublasLtDisableCpuInstructionsSetMaskRq : public CudaRequestBase {
    unsigned mask;
};

struct cublasLtDisableCpuInstructionsSetMaskRp : public CudaResponseBase {
    unsigned status;
};

struct cublasLtGetCudartVersionRq : public CudaRequestBase {

};

struct cublasLtGetCudartVersionRp : public CudaResponseBase {
    size_t version;
};

struct cublasLtLoggerSetCallbackRq : public CudaRequestBase {
    cublasLtLoggerCallback_t callback;
};

struct cublasLtLoggerSetCallbackRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasLtLoggerSetFileRq : public CudaRequestBase {
    FILE* file;
};

struct cublasLtLoggerSetFileRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasLtLoggerOpenFileRq : public CudaRequestBase {
    const char* logFile;
};

struct cublasLtLoggerOpenFileRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasLtLoggerSetLevelRq : public CudaRequestBase {
    int level;
};

struct cublasLtLoggerSetLevelRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtLoggerSetMaskRq : public CudaRequestBase {
    int mask;
};

struct cublasLtLoggerSetMaskRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtLoggerForceDisableRq : public CudaRequestBase {

};

struct cublasLtLoggerForceDisableRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatmulRq : public CudaRequestBase {
    cublasLtHandle_t lightHandle;
    cublasLtMatmulDesc_t computeDesc;
    float alpha;
    const void *A;
    cublasLtMatrixLayout_t Adesc;
    const void *B;
    cublasLtMatrixLayout_t Bdesc;
    float beta;
    const void *C;
    cublasLtMatrixLayout_t Cdesc;
    void *D;
    cublasLtMatrixLayout_t Ddesc;
    const cublasLtMatmulAlgo_t *algo;
    void *workspace;
    size_t workspaceSizeInBytes;
    cudaStream_t stream;
};

struct cublasLtMatmulRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatmulAlgoCapGetAttributeRq : public CudaRequestBase {
    const cublasLtMatmulAlgo_t *algo;
    cublasLtMatmulAlgoCapAttributes_t attr;
    void *buf;
    size_t sizeInBytes;
};

struct cublasLtMatmulAlgoCapGetAttributeRp : public CudaResponseBase {
    cublasStatus_t status;
    size_t sizeWritten;
};
struct cublasLtMatmulAlgoCheckRq : public CudaRequestBase {
    cublasLtHandle_t lightHandle;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Cdesc;
    cublasLtMatrixLayout_t Ddesc;
    const cublasLtMatmulAlgo_t *algo;
};

struct cublasLtMatmulAlgoCheckRp : public CudaResponseBase {
    cublasStatus_t status;
    cublasLtMatmulHeuristicResult_t result;
};
struct cublasLtMatmulAlgoConfigGetAttributeRq : public CudaRequestBase {
    const cublasLtMatmulAlgo_t *algo;
    cublasLtMatmulAlgoConfigAttributes_t attr;
    void *buf;
    size_t sizeInBytes;
};

struct cublasLtMatmulAlgoConfigGetAttributeRp : public CudaResponseBase {
    cublasStatus_t status;
    size_t sizeWritten;
};
struct cublasLtMatmulAlgoConfigSetAttributeRq : public CudaRequestBase {
    cublasLtMatmulAlgo_t *algo;
    cublasLtMatmulAlgoConfigAttributes_t attr;
    const void *buf;
    size_t sizeInBytes;
};

struct cublasLtMatmulAlgoConfigSetAttributeRp : public CudaResponseBase {
    cublasStatus_t status;
};

struct cublasLtMatmulAlgoGetIdsRq : public CudaRequestBase {
    cublasLtHandle_t lightHandle;
    cublasComputeType_t computeType;
    cudaDataType_t scaleType;
    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
    cudaDataType_t Dtype;
    int requestedAlgoCount;
};

struct cublasLtMatmulAlgoGetIdsRp : public CudaResponseBase {
    cublasStatus_t status;
    int returnAlgoCount;
    int algoIdsArray[32];
};
struct cublasLtMatmulAlgoInitRq : public CudaRequestBase {
    cublasLtHandle_t lightHandle;
    cublasComputeType_t computeType;
    cudaDataType_t scaleType;
    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
    cudaDataType_t Dtype;
    int algoId;
};

struct cublasLtMatmulAlgoInitRp : public CudaResponseBase {
    cublasStatus_t status;
    cublasLtMatmulAlgo_t algo;
};
struct cublasLtMatmulDescCreateRq : public CudaRequestBase {

    cublasComputeType_t computeType;
    cudaDataType_t scaleType;

};

struct cublasLtMatmulDescCreateRp : public CudaResponseBase {
    cublasLtMatmulDesc_t matmulDesc;
};
struct cublasLtMatmulDescInitRq : public CudaRequestBase {
    cublasLtMatmulDesc_t matmulDesc;
    cublasComputeType_t computeType;
    cudaDataType_t scaleType;
};

struct cublasLtMatmulDescInitRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatmulDescDestroyRq : public CudaRequestBase {
    cublasLtMatmulDesc_t matmulDesc;
};

struct cublasLtMatmulDescDestroyRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatmulDescGetAttributeRq : public CudaRequestBase {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescAttributes_t attr;
    size_t sizeInBytes;
};

struct cublasLtMatmulDescGetAttributeRp : public CudaResponseBase {
    cublasStatus_t status;
    size_t sizeWritten;
    char buf[32];
};
struct cublasLtMatmulDescSetAttributeRq : public CudaRequestBase {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescAttributes_t attr;
    size_t sizeInBytes;
    const void * buf;
    char * client_shared_mem;
};

struct cublasLtMatmulDescSetAttributeRp : public CudaResponseBase {

};
struct cublasLtMatmulPreferenceCreateRq : public CudaRequestBase {

};

struct cublasLtMatmulPreferenceCreateRp : public CudaResponseBase {

    cublasLtMatmulPreference_t pref;
};
struct cublasLtMatmulPreferenceInitRq : public CudaRequestBase {
    cublasLtMatmulPreference_t pref;
};

struct cublasLtMatmulPreferenceInitRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatmulPreferenceDestroyRq : public CudaRequestBase {
    cublasLtMatmulPreference_t pref;
};

struct cublasLtMatmulPreferenceDestroyRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatmulPreferenceGetAttributeRq : public CudaRequestBase {
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceAttributes_t attr;
    size_t sizeInBytes;
};

struct cublasLtMatmulPreferenceGetAttributeRp : public CudaResponseBase {
    cublasStatus_t status;
    size_t sizeWritten;
    char buf[32];
};
struct cublasLtMatmulPreferenceSetAttributeRq : public CudaRequestBase {
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceAttributes_t attr;
    size_t sizeInBytes;
    const void * buf;
    char * client_shared_mem;
};

struct cublasLtMatmulPreferenceSetAttributeRp : public CudaResponseBase {

};
struct cublasLtMatrixLayoutCreateRq : public CudaRequestBase {
    cudaDataType type1;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
};

struct cublasLtMatrixLayoutCreateRp : public CudaResponseBase {

    cublasLtMatrixLayout_t matLayout;
};
struct cublasLtMatrixLayoutInitRq : public CudaRequestBase {
    cublasLtMatrixLayout_t matLayout;
    cudaDataType type;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
};

struct cublasLtMatrixLayoutInitRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatrixLayoutDestroyRq : public CudaRequestBase {
    cublasLtMatrixLayout_t matLayout;
};

struct cublasLtMatrixLayoutDestroyRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatrixLayoutGetAttributeRq : public CudaRequestBase {
    cublasLtMatrixLayout_t matLayout;
    cublasLtMatrixLayoutAttribute_t attr;
    size_t sizeInBytes;
};

struct cublasLtMatrixLayoutGetAttributeRp : public CudaResponseBase {
    cublasStatus_t status;
    size_t sizeWritten;
    char buf[32];
};
struct cublasLtMatrixLayoutSetAttributeRq : public CudaRequestBase {
    cublasLtMatrixLayout_t matLayout;
    cublasLtMatrixLayoutAttribute_t attr;
    size_t sizeInBytes;
    char buf[32];
};

struct cublasLtMatrixLayoutSetAttributeRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatrixTransformRq : public CudaRequestBase {
    cublasLtHandle_t lightHandle;
    cublasLtMatrixTransformDesc_t transformDesc;
    char alpha[32];
    char beta[32];
    const void *A;
    const void *B;
    void *C;
    cudaStream_t stream;
};

struct cublasLtMatrixTransformRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatrixTransformDescCreateRq : public CudaRequestBase {
    cublasLtMatrixTransformDesc_t *transformDesc;
    cudaDataType scaleType;
};

struct cublasLtMatrixTransformDescCreateRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatrixTransformDescInitRq : public CudaRequestBase {
    cublasLtMatrixTransformDesc_t transformDesc;
    cudaDataType scaleType;
};

struct cublasLtMatrixTransformDescInitRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatrixTransformDescDestroyRq : public CudaRequestBase {
    cublasLtMatrixTransformDesc_t transformDesc;
};

struct cublasLtMatrixTransformDescDestroyRp : public CudaResponseBase {
    cublasStatus_t status;
};
struct cublasLtMatrixTransformDescGetAttributeRq : public CudaRequestBase {
    cublasLtMatrixTransformDesc_t transformDesc;
    cublasLtMatrixTransformDescAttributes_t attr;
    size_t sizeInBytes;
};

struct cublasLtMatrixTransformDescGetAttributeRp : public CudaResponseBase {
    size_t sizeWritten;
    char buf[32];
};
struct cublasLtMatrixTransformDescSetAttributeRq : public CudaRequestBase {
    cublasLtMatrixTransformDesc_t transformDesc;
    cublasLtMatrixTransformDescAttributes_t attr;
    size_t sizeInBytes;
    char buf[32];
};

struct cublasLtMatrixTransformDescSetAttributeRp : public CudaResponseBase {
};

struct ncclGetLastErrorRq : public CudaRequestBase {
    ncclComm_t comm;
    size_t shared_mem_offset;
};
struct ncclGetLastErrorRp : public CudaResponseBase {
    const char* errorMessage;
    size_t size;
};
struct ncclGetErrorStringRq : public CudaRequestBase {
    ncclResult_t result;
    size_t shared_mem_offset;
};
struct ncclGetErrorStringRp : public CudaResponseBase {
    const char* errorMessage;
    size_t size;
};
struct ncclGetVersionRq : public CudaRequestBase {

};
struct ncclGetVersionRp : public CudaResponseBase {
    ncclResult_t result;
    int version;
};
struct ncclGetUniqueIdRq : public CudaRequestBase {

    size_t shared_mem_offset;
};

struct ncclGetUniqueIdRp : public CudaResponseBase {

};
struct ncclCommInitRankRq : public CudaRequestBase {
    int nranks;
    ncclUniqueId commId;
    int rank;
};

struct ncclCommInitRankRp : public CudaResponseBase {
    ncclResult_t result;
    ncclComm_t comm;
};
struct ncclCommInitAllRq : public CudaRequestBase {
    int ndev;
    void* devlist;
    char* client_shared_mem;
};

struct ncclCommInitAllRp : public CudaResponseBase {
    ncclResult_t result;
    void* comms;
};
struct ncclCommInitRankConfigRq : public CudaRequestBase {
    int nranks;
    void * commId;
    int rank;
    void* config;
    char* client_shared_mem;
};
struct ncclCommInitRankConfigRp : public CudaResponseBase {
    ncclResult_t result;
    ncclComm_t comm;
};
struct ncclCommInitRankScalableRq : public CudaRequestBase {
    int nranks;
    int myrank;
    int nId;
    char* client_shared_mem;
    void* config;
    void* commIds;

};
struct ncclCommInitRankScalableRp : public CudaResponseBase {
    ncclResult_t result;
    ncclComm_t newcomm;
};
struct ncclCommSplitRq : public CudaRequestBase {
    ncclComm_t comm;
    int color;
    int key;
    void* config;
    char* client_shared_mem;
};

struct ncclCommSplitRp : public CudaResponseBase {
    ncclResult_t result;
    ncclComm_t newcomm;
};
struct ncclCommFinalizeRq : public CudaRequestBase {
    ncclComm_t comm;
};

struct ncclCommFinalizeRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclCommDestroyRq : public CudaRequestBase {
    ncclComm_t comm;
};

struct ncclCommDestroyRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclCommAbortRq : public CudaRequestBase {
    ncclComm_t comm;
};

struct ncclCommAbortRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclCommGetAsyncErrorRq : public CudaRequestBase {
    ncclComm_t comm;
};

struct ncclCommGetAsyncErrorRp : public CudaResponseBase {
    ncclResult_t result;
    ncclResult_t asyncError;
};
struct ncclCommCountRq : public CudaRequestBase {
    ncclComm_t comm;
};

struct ncclCommCountRp : public CudaResponseBase {
    ncclResult_t result;
    int count;
};
struct ncclCommCuDeviceRq : public CudaRequestBase {
    ncclComm_t comm;
};

struct ncclCommCuDeviceRp : public CudaResponseBase {
    ncclResult_t result;
    int device;
};
struct ncclCommUserRankRq : public CudaRequestBase {
    ncclComm_t comm;
};

struct ncclCommUserRankRp : public CudaResponseBase {
    ncclResult_t result;
    int rank;
};
struct ncclCommRegisterRq : public CudaRequestBase {
    ncclComm_t comm;
    void* buff;
    size_t size;
};

struct ncclCommRegisterRp : public CudaResponseBase {
    ncclResult_t result;
    void* handle;
};
struct ncclCommDeregisterRq : public CudaRequestBase {
    ncclComm_t comm;
    void* handle;
};

struct ncclCommDeregisterRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclMemAllocRq : public CudaRequestBase {
    size_t size;
};

struct ncclMemAllocRp : public CudaResponseBase {
    ncclResult_t result;
    void* ptr;
};
struct ncclMemFreeRq : public CudaRequestBase {
    void* ptr;
};

struct ncclMemFreeRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclAllReduceRq : public CudaRequestBase {
    const void* sendbuff;
    void* recvbuff;
    size_t count;
    ncclDataType_t datatype;
    ncclRedOp_t op;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclAllReduceRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclBroadcastRq : public CudaRequestBase {
    const void* sendbuff;
    void* recvbuff;
    size_t count;
    ncclDataType_t datatype;
    int root;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclBroadcastRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclBcastRq : public CudaRequestBase {
    void* buff;
    size_t count;
    ncclDataType_t datatype;
    int root;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclBcastRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclReduceRq : public CudaRequestBase {
    const void* sendbuff;
    void* recvbuff;
    size_t count;
    ncclDataType_t datatype;
    ncclRedOp_t op;
    int root;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclReduceRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclAllGatherRq : public CudaRequestBase {
    const void* sendbuff;
    void* recvbuff;
    size_t sendcount;
    ncclDataType_t datatype;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclAllGatherRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclReduceScatterRq : public CudaRequestBase {
    const void* sendbuff;
    void* recvbuff;
    size_t recvcount;
    ncclDataType_t datatype;
    ncclRedOp_t op;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclReduceScatterRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclGroupStartRq : public CudaRequestBase {

};

struct ncclGroupStartRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclGroupEndRq : public CudaRequestBase {

};

struct ncclGroupEndRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclGroupSimulateEndRq : public CudaRequestBase {
    ncclSimInfo_t* simInfo;
};

struct ncclGroupSimulateEndRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclSendRq : public CudaRequestBase {
    const void* sendbuff;
    size_t count;
    ncclDataType_t datatype;
    int peer;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclSendRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclRecvRq : public CudaRequestBase {
    void* recvbuff;
    size_t count;
    ncclDataType_t datatype;
    int peer;
    ncclComm_t comm;
    cudaStream_t stream;
};

struct ncclRecvRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclRedOpCreatePreMulSumRq : public CudaRequestBase {
    ncclRedOp_t* op;
    void* scalar;
    ncclDataType_t datatype;
    ncclScalarResidence_t residence;
    ncclComm_t comm;
};

struct ncclRedOpCreatePreMulSumRp : public CudaResponseBase {
    ncclResult_t result;
};
struct ncclRedOpDestroyRq : public CudaRequestBase {
    ncclRedOp_t op;
    ncclComm_t comm;
};

struct ncclRedOpDestroyRp : public CudaResponseBase {
    ncclResult_t result;
};

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        CUresult err = call;                                                                               \
        if (err != CUDA_SUCCESS) {                                                                         \
            const char *errorStr;                                                                          \
            cuGetErrorString(err, &errorStr);                                                              \
            std::cerr << "CUDA Error: " << errorStr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                            \
        }                                                                                                  \
    } while (0)
const char *SOCKET_PATH;

static std::mutex g_cache_mutex;
static std::unordered_map<std::string, CUmodule> g_module_cache;
static std::unordered_map<std::string, CUfunction> g_function_cache;

struct RingBuffer {
    std::atomic<size_t> write_ptr;
    std::atomic<size_t> read_ptr;
    char buffer[BUFFER_SIZE];

    RingBuffer() : write_ptr(0), read_ptr(0) {}

    size_t available_write_space() const {
        size_t w = write_ptr.load(std::memory_order_acquire);
        size_t r = read_ptr.load(std::memory_order_acquire);
        return (r + BUFFER_SIZE - w - sizeof(size_t)) % BUFFER_SIZE;
    }

    size_t available_read_space() const {
        size_t w = write_ptr.load(std::memory_order_acquire);
        size_t r = read_ptr.load(std::memory_order_acquire);
        return (w + BUFFER_SIZE - r) % BUFFER_SIZE;
    }
};
std::string requestMemoryPath;
std::string responseMemoryPath;
std::string sharedMemoryPath;
RingBuffer* get_request_shared_memory(bool create) {
    int shm_fd = shm_open(requestMemoryPath.c_str(), O_RDWR | O_CREAT, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return nullptr;
    }

    if (create) {

        if (ftruncate(shm_fd, SHM_SIZE) == -1) {
            perror("ftruncate");
            return nullptr;
        }
    }

    void *shm_ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap");
        return nullptr;
    }
    if (create) {
        memset(shm_ptr, 0, SHM_SIZE);
    }

    return static_cast<RingBuffer*>(shm_ptr);
}

RingBuffer* get_response_shared_memory(bool create) {
    int shm_fd = shm_open(responseMemoryPath.c_str(), O_RDWR | (create ? O_CREAT : 0), 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return nullptr;
    }

    if (create) {

        if (ftruncate(shm_fd, SHM_SIZE) == -1) {
            perror("ftruncate");
            return nullptr;
        }
    }

    void *shm_ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap");
        return nullptr;
    }
    if (create) {
        memset(shm_ptr, 0, SHM_SIZE);
    }

    return static_cast<RingBuffer*>(shm_ptr);
}

void* RequestReceive(RingBuffer* rb) {
    if (rb->available_read_space() < sizeof(size_t)) {
        return nullptr;
    }

    size_t r = rb->read_ptr.load(std::memory_order_relaxed);
    size_t* msg_size = (size_t*)(&rb->buffer[r]);

    if (rb->available_read_space() < *msg_size) {
        return nullptr;
    }

    void* res = (void*)&rb->buffer[r];
    r = (r + *msg_size) % BUFFER_SIZE;

    rb->read_ptr.store(r, std::memory_order_release);
    return res;
}

void* NewResponse(RingBuffer* rb) {
    return (void*)(&rb->buffer[rb->write_ptr.load(std::memory_order_relaxed)]);
}

bool ResponseSend(RingBuffer* rb, size_t size) {
    size_t total_size = size;

    if (rb->available_write_space() < total_size) {
        std::cerr << "buffer is full and unable to send the message\n";
        return false;
    }

    size_t w = rb->write_ptr.load(std::memory_order_relaxed);
    size_t next_w = (w + total_size) % BUFFER_SIZE;

    rb->write_ptr.store(next_w, std::memory_order_release);
    return true;
}

int main(int argc, char* argv[]) {
    CUdevice device;
    CUcontext context;
    CUDA_CHECK(cuInit(0));
    CUresult res;
    CUfunction function;

    requestMemoryPath = argv[1];
    responseMemoryPath = argv[2];
    sharedMemoryPath = argv[3];

    int shm_fd = shm_open(sharedMemoryPath.c_str(), O_RDWR | O_CREAT, 0666);
    if (shm_fd == -1) {
        perror("shm_open failed");
        exit(1);
    }

    static char *shared_mem = (char *)mmap(nullptr, 1024 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_mem == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }

    RingBuffer* request_rb = get_request_shared_memory(false);
    if (!request_rb) return 1;
    RingBuffer* response_rb = get_response_shared_memory(true);
    if (!response_rb) return 1;

    bool send_result;
    while (true) {
        CudaRequestBase* request_ptr = (CudaRequestBase*)(RequestReceive(request_rb));
        CudaRequestBase& request = *request_ptr;
        if (request_ptr) {
        switch (request.type) {
            case ncclGetLastErrorType: {
                ncclGetLastErrorRq& req = *reinterpret_cast<ncclGetLastErrorRq*>(request_ptr);

                ncclGetLastErrorRp& response = *((ncclGetLastErrorRp*)(NewResponse(response_rb)));

                const char* errorMessage = ncclGetLastError(req.comm);
                size_t error_message_size = strlen(errorMessage) + 1;
                if (req.shared_mem_offset + error_message_size > 1024 * 1024 * 1024) {
                    response.errorMessage = "Error: Not enough shared memory";
                    response.size = 0;
                } else {

                    strcpy(shared_mem + req.shared_mem_offset, errorMessage);
                    response.errorMessage = shared_mem + req.shared_mem_offset;
                    response.size = error_message_size;

                }

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclGetErrorStringType: {
                ncclGetErrorStringRq& req = *reinterpret_cast<ncclGetErrorStringRq*>(request_ptr);

                ncclGetErrorStringRp& response = *((ncclGetErrorStringRp*)(NewResponse(response_rb)));

                const char* errorMessage = ncclGetErrorString(req.result);
                size_t error_message_size = strlen(errorMessage) + 1;
                if (req.shared_mem_offset + error_message_size > 1024 * 1024 * 1024) {
                    response.errorMessage = "Error: Not enough shared memory";
                    response.size = 0;
                } else {

                    strcpy(shared_mem + req.shared_mem_offset, errorMessage);
                    response.errorMessage = shared_mem + req.shared_mem_offset;
                    response.size = error_message_size;

                }

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclGetVersionType: {
                ncclGetVersionRq& req = *reinterpret_cast<ncclGetVersionRq*>(request_ptr);

                ncclGetVersionRp& response = *((ncclGetVersionRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclGetVersion(&response.version);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclGetUniqueIdType: {
                ncclGetUniqueIdRq& req = *reinterpret_cast<ncclGetUniqueIdRq*>(request_ptr);

                ncclGetUniqueIdRp& response = *((ncclGetUniqueIdRp*)(NewResponse(response_rb)));
                ncclUniqueId* uniqueId = (ncclUniqueId*)(shared_mem + req.shared_mem_offset);

                response.ncclResult = ncclGetUniqueId(uniqueId);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommInitRankType: {
                ncclCommInitRankRq& req = *reinterpret_cast<ncclCommInitRankRq*>(request_ptr);

                ncclCommInitRankRp& response = *((ncclCommInitRankRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommInitRank(&response.comm, req.nranks, req.commId, req.rank);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommInitAllType: {
                ncclCommInitAllRq& req = *reinterpret_cast<ncclCommInitAllRq*>(request_ptr);

                const int* devlist = (const int *)(shared_mem + (size_t)((char*)req.devlist - req.client_shared_mem));
                ncclCommInitAllRp& response = *((ncclCommInitAllRp*)(NewResponse(response_rb)));

                ncclComm_t* comms = (ncclComm_t *)(devlist + req.ndev * sizeof(int));
                response.ncclResult = ncclCommInitAll(comms, req.ndev, devlist);
                int i;
                    for (i = 0;i < req.ndev;i++) {

                }

                for (i = 0;i < req.ndev;i++) {

                }

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommInitRankConfigType: {
                ncclCommInitRankConfigRq& req = *reinterpret_cast<ncclCommInitRankConfigRq*>(request_ptr);

                ncclConfig_t config = *(ncclConfig_t*)(shared_mem + (size_t)((char*)req.config - req.client_shared_mem));
                ncclUniqueId commId = *(ncclUniqueId *)(shared_mem + (size_t)((char*)req.commId - req.client_shared_mem));
                ncclCommInitRankConfigRp& response = *((ncclCommInitRankConfigRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommInitRankConfig(&response.comm, req.nranks, commId, req.rank, &config);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommInitRankScalableType: {
                ncclCommInitRankScalableRq& req = *reinterpret_cast<ncclCommInitRankScalableRq*>(request_ptr);

                ncclConfig_t config = *(ncclConfig_t *)(shared_mem + (size_t)((char*)req.config - req.client_shared_mem));

                ncclUniqueId* commIds = (ncclUniqueId *)(shared_mem + (size_t)((char*)req.commIds - req.client_shared_mem));

                ncclCommInitRankScalableRp& response = *((ncclCommInitRankScalableRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommInitRankScalable(&response.newcomm, req.nranks, req.myrank, req.nId, commIds, &config);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommSplitType: {
                ncclCommSplitRq& req = *reinterpret_cast<ncclCommSplitRq*>(request_ptr);

                ncclConfig_t config;
                if (req.config) {
                    config = *(ncclConfig_t *)(shared_mem + (size_t)((char*)req.config - req.client_shared_mem));
                } else {
                    config = {};
                }

                ncclCommSplitRp& response = *((ncclCommSplitRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommSplit(req.comm, req.color, req.key, &response.newcomm, &config);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommFinalizeType: {
                ncclCommFinalizeRq& req = *reinterpret_cast<ncclCommFinalizeRq*>(request_ptr);

                ncclCommFinalizeRp& response = *((ncclCommFinalizeRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommFinalize(req.comm);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommDestroyType: {
                ncclCommDestroyRq& req = *reinterpret_cast<ncclCommDestroyRq*>(request_ptr);

                ncclCommDestroyRp& response = *((ncclCommDestroyRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommDestroy(req.comm);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommAbortType: {
                ncclCommAbortRq& req = *reinterpret_cast<ncclCommAbortRq*>(request_ptr);

                ncclCommAbortRp& response = *((ncclCommAbortRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommAbort(req.comm);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommGetAsyncErrorType: {
                ncclCommGetAsyncErrorRq& req = *reinterpret_cast<ncclCommGetAsyncErrorRq*>(request_ptr);

                ncclCommGetAsyncErrorRp& response = *((ncclCommGetAsyncErrorRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommGetAsyncError(req.comm, &response.asyncError);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommCountType: {
                ncclCommCountRq& req = *reinterpret_cast<ncclCommCountRq*>(request_ptr);

                ncclCommCountRp& response = *((ncclCommCountRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommCount(req.comm, &response.count);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommCuDeviceType: {
                ncclCommCuDeviceRq& req = *reinterpret_cast<ncclCommCuDeviceRq*>(request_ptr);

                ncclCommCuDeviceRp& response = *((ncclCommCuDeviceRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommCuDevice(req.comm, &response.device);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommUserRankType: {
                ncclCommUserRankRq& req = *reinterpret_cast<ncclCommUserRankRq*>(request_ptr);

                ncclCommUserRankRp& response = *((ncclCommUserRankRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommUserRank(req.comm, &response.rank);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommRegisterType: {
                ncclCommRegisterRq& req = *reinterpret_cast<ncclCommRegisterRq*>(request_ptr);

                ncclCommRegisterRp& response = *((ncclCommRegisterRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommRegister(req.comm, req.buff, req.size, &response.handle);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclCommDeregisterType: {
                ncclCommDeregisterRq& req = *reinterpret_cast<ncclCommDeregisterRq*>(request_ptr);

                ncclCommDeregisterRp& response = *((ncclCommDeregisterRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclCommDeregister(req.comm, req.handle);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclMemAllocType: {
                ncclMemAllocRq& req = *reinterpret_cast<ncclMemAllocRq*>(request_ptr);

                ncclMemAllocRp& response = *((ncclMemAllocRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclMemAlloc(&response.ptr, req.size);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclMemFreeType: {
                ncclMemFreeRq& req = *reinterpret_cast<ncclMemFreeRq*>(request_ptr);

                ncclMemFreeRp& response = *((ncclMemFreeRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclMemFree(req.ptr);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclAllReduceType: {

                ncclAllReduceRq& req = *reinterpret_cast<ncclAllReduceRq*>(request_ptr);

                ncclAllReduceRp& response = *((ncclAllReduceRp*)(NewResponse(response_rb)));

                printf("[DEBUG] Received AllReduce request:\n");
                printf("  sendbuff=%p, recvbuff=%p\n", req.sendbuff, req.recvbuff);
                printf("  count=%ld, datatype=%d, op=%d, comm=%p, stream=%p\n",
                       req.count, req.datatype, req.op, req.comm, req.stream);

                void *host_send = malloc(req.count * get_nccl_type_size(req.datatype));
                void *host_recv = malloc(req.count * get_nccl_type_size(req.datatype));

                cudaMemcpy(host_send, req.sendbuff,
                          req.count * get_nccl_type_size(req.datatype), cudaMemcpyDeviceToHost);
                printf("Sendbuff first 5 elements: ");
                for(int i=0; i<5 && i<req.count; i++) {
                    printf("%f ", ((float*)host_send)[i]);
                }
                printf("\n");

                response.ncclResult = ncclAllReduce(req.sendbuff, req.recvbuff, req.count,
                                                    req.datatype, req.op, req.comm, req.stream);

                cudaError_t syncErr = cudaStreamSynchronize(req.stream);
                if(syncErr != cudaSuccess) {
                    printf("[ERROR] Stream sync failed: %s\n", cudaGetErrorString(syncErr));
                }

                cudaMemcpy(host_recv, req.recvbuff,
                          req.count * get_nccl_type_size(req.datatype), cudaMemcpyDeviceToHost);
                printf("Recvbuff first 5 elements: ");
                for(int i=0; i<5 && i<req.count; i++) {
                    printf("%f ", ((float*)host_recv)[i]);
                }
                printf("\n");

                free(host_send);
                free(host_recv);

                if(response.ncclResult != ncclSuccess) {
                    printf("[ERROR] ncclAllReduce failed: %s\n",
                           ncclGetErrorString(response.ncclResult));
                }

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclBroadcastType: {
                ncclBroadcastRq& req = *reinterpret_cast<ncclBroadcastRq*>(request_ptr);

                ncclBroadcastRp& response = *((ncclBroadcastRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclBroadcast(req.sendbuff, req.recvbuff, req.count,
                                                req.datatype, req.root, req.comm, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclBcastType: {
                ncclBcastRq& req = *reinterpret_cast<ncclBcastRq*>(request_ptr);

                ncclBcastRp& response = *((ncclBcastRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclBcast(req.buff, req.count, req.datatype,
                                            req.root, req.comm, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclReduceType: {
                ncclReduceRq& req = *reinterpret_cast<ncclReduceRq*>(request_ptr);

                ncclReduceRp& response = *((ncclReduceRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclReduce(req.sendbuff, req.recvbuff, req.count,
                                            req.datatype, req.op, req.root, req.comm, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclAllGatherType: {
                ncclAllGatherRq& req = *reinterpret_cast<ncclAllGatherRq*>(request_ptr);

                ncclAllGatherRp& response = *((ncclAllGatherRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclAllGather(req.sendbuff, req.recvbuff, req.sendcount,
                                                req.datatype, req.comm, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclReduceScatterType: {
                ncclReduceScatterRq& req = *reinterpret_cast<ncclReduceScatterRq*>(request_ptr);

                ncclReduceScatterRp& response = *((ncclReduceScatterRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclReduceScatter(req.sendbuff, req.recvbuff, req.recvcount,
                                                    req.datatype, req.op, req.comm, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclGroupStartType: {
                ncclGroupStartRq& req = *reinterpret_cast<ncclGroupStartRq*>(request_ptr);

                ncclGroupStartRp& response = *((ncclGroupStartRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclGroupStart();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclGroupEndType: {
                ncclGroupEndRq& req = *reinterpret_cast<ncclGroupEndRq*>(request_ptr);

                ncclGroupEndRp& response = *((ncclGroupEndRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclGroupEnd();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclGroupSimulateEndType: {
                ncclGroupSimulateEndRq& req = *reinterpret_cast<ncclGroupSimulateEndRq*>(request_ptr);

                ncclGroupSimulateEndRp& response = *((ncclGroupSimulateEndRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclGroupSimulateEnd(req.simInfo);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclSendType: {
                ncclSendRq& req = *reinterpret_cast<ncclSendRq*>(request_ptr);

                ncclSendRp& response = *((ncclSendRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclSend(req.sendbuff, req.count, req.datatype, req.peer, req.comm, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclRecvType: {
                ncclRecvRq& req = *reinterpret_cast<ncclRecvRq*>(request_ptr);

                ncclRecvRp& response = *((ncclRecvRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclRecv(req.recvbuff, req.count, req.datatype, req.peer, req.comm, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclRedOpCreatePreMulSumType: {
                ncclRedOpCreatePreMulSumRq& req = *reinterpret_cast<ncclRedOpCreatePreMulSumRq*>(request_ptr);

                ncclRedOpCreatePreMulSumRp& response = *((ncclRedOpCreatePreMulSumRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclRedOpCreatePreMulSum(req.op, req.scalar, req.datatype, req.residence, req.comm);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case ncclRedOpDestroyType: {
                ncclRedOpDestroyRq& req = *reinterpret_cast<ncclRedOpDestroyRq*>(request_ptr);

                ncclRedOpDestroyRp& response = *((ncclRedOpDestroyRp*)(NewResponse(response_rb)));

                response.ncclResult = ncclRedOpDestroy(req.op, req.comm);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaThreadExitType: {
                cudaThreadExitRq& req = *reinterpret_cast<cudaThreadExitRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaThreadExit();
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaThreadGetCacheConfigType: {
                cudaThreadGetCacheConfigRq& req = *reinterpret_cast<cudaThreadGetCacheConfigRq*>(request_ptr);

                cudaThreadGetCacheConfigRp& response = *((cudaThreadGetCacheConfigRp*)(NewResponse(response_rb)));
                response.result = cudaThreadGetCacheConfig(&response.cacheConfig);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaThreadGetLimitType: {
                cudaThreadGetLimitRq& req = *reinterpret_cast<cudaThreadGetLimitRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaThreadGetLimit(&req.Value, req.limit);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaThreadSetCacheConfigType: {
                cudaThreadSetCacheConfigRq& req = *reinterpret_cast<cudaThreadSetCacheConfigRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaThreadSetCacheConfig(req.cacheConfig);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaThreadSetLimitType: {
                cudaThreadSetLimitRq& req = *reinterpret_cast<cudaThreadSetLimitRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaThreadSetLimit(req.limit, req.value);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaThreadSynchronizeType: {
                cudaThreadSynchronizeRq& req = *reinterpret_cast<cudaThreadSynchronizeRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaThreadSynchronize();
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaGetErrorStringType: {
                cudaGetErrorStringRq& req = *reinterpret_cast<cudaGetErrorStringRq*>(request_ptr);

                cudaGetErrorStringRp& response = *((cudaGetErrorStringRp*)(NewResponse(response_rb)));
                response.result = cudaGetErrorString(req.error);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaGetLastErrorType: {
                cudaGetLastErrorRq& req = *reinterpret_cast<cudaGetLastErrorRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaGetLastError();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaPeekAtLastErrorType: {
                cudaPeekAtLastErrorRq& req = *reinterpret_cast<cudaPeekAtLastErrorRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaPeekAtLastError();
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaChooseDeviceType: {
                cudaChooseDeviceRq& req = *reinterpret_cast<cudaChooseDeviceRq*>(request_ptr);

                cudaChooseDeviceRp& response = *((cudaChooseDeviceRp*)(NewResponse(response_rb)));
                response.result = cudaChooseDevice(&response.device, &req.prop);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaGetDeviceType: {

                cudaGetDeviceRq& req = *reinterpret_cast<cudaGetDeviceRq*>(request_ptr);

                cudaGetDeviceRp& response = *((cudaGetDeviceRp*)(NewResponse(response_rb)));
                response.result = cudaGetDevice(&response.device);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));

                break;
            }
            case cudaGetDeviceCountType: {
                cudaGetDeviceCountRq& req = *reinterpret_cast<cudaGetDeviceCountRq*>(request_ptr);

                cudaGetDeviceCountRp& response = *((cudaGetDeviceCountRp*)(NewResponse(response_rb)));
                response.result = cudaGetDeviceCount(&response.count);
                const char *cudaVisibleDevices = std::getenv("CUDA_VISIBLE_DEVICES");
                if (cudaVisibleDevices != nullptr) {

                } else {

                }
                int device;
                cudaError_t err = cudaGetDevice(&device);
                printf("Current device ID: %d\n", device);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaGetDevicePropertiesType: {
                cudaGetDevicePropertiesRq& req = *reinterpret_cast<cudaGetDevicePropertiesRq*>(request_ptr);

                cudaGetDevicePropertiesRp& response = *((cudaGetDevicePropertiesRp*)(NewResponse(response_rb)));
                response.result = cudaGetDeviceProperties(&response.prop, req.device);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaSetDeviceType: {
                cudaSetDeviceRq& req = *reinterpret_cast<cudaSetDeviceRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaSetDevice(req.device);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaSetDeviceFlagsType: {
                cudaSetDeviceFlagsRq& req = *reinterpret_cast<cudaSetDeviceFlagsRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaSetDeviceFlags(req.flags);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaSetValidDevicesType: {
                cudaSetValidDevicesRq& req = *reinterpret_cast<cudaSetValidDevicesRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaSetValidDevices(req.device_arr, req.len);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaDeviceFlushGPUDirectRDMAWritesType: {
                cudaDeviceFlushGPUDirectRDMAWritesRq& req = *reinterpret_cast<cudaDeviceFlushGPUDirectRDMAWritesRq*>(request_ptr);

                cudaDeviceFlushGPUDirectRDMAWritesRp& response = *((cudaDeviceFlushGPUDirectRDMAWritesRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceFlushGPUDirectRDMAWrites(req.target, req.scope);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetAttributeType: {
                cudaDeviceGetAttributeRq& req = *reinterpret_cast<cudaDeviceGetAttributeRq*>(request_ptr);

                cudaDeviceGetAttributeRp& response = *((cudaDeviceGetAttributeRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetAttribute(&response.value, req.attr, req.device);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetByPCIBusIdType: {
                cudaDeviceGetByPCIBusIdRq& req = *reinterpret_cast<cudaDeviceGetByPCIBusIdRq*>(request_ptr);

                cudaDeviceGetByPCIBusIdRp& response = *((cudaDeviceGetByPCIBusIdRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetByPCIBusId(&response.device, req.pciBusId);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaDeviceGetCacheConfigType: {
                cudaDeviceGetCacheConfigRq& req = *reinterpret_cast<cudaDeviceGetCacheConfigRq*>(request_ptr);

                cudaDeviceGetCacheConfigRp& response = *((cudaDeviceGetCacheConfigRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetCacheConfig(&response.cacheConfig);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetDefaultMemPoolType: {
                cudaDeviceGetDefaultMemPoolRq& req = *reinterpret_cast<cudaDeviceGetDefaultMemPoolRq*>(request_ptr);

                cudaDeviceGetDefaultMemPoolRp& response = *((cudaDeviceGetDefaultMemPoolRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetDefaultMemPool(&response.memPool, req.device);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetLimitType: {
                cudaDeviceGetLimitRq& req = *reinterpret_cast<cudaDeviceGetLimitRq*>(request_ptr);

                cudaDeviceGetLimitRp& response = *((cudaDeviceGetLimitRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetLimit(&response.value, req.limit);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetMemPoolType: {
                cudaDeviceGetMemPoolRq& req = *reinterpret_cast<cudaDeviceGetMemPoolRq*>(request_ptr);

                cudaDeviceGetMemPoolRp& response = *((cudaDeviceGetMemPoolRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetMemPool(&response.memPool, req.device);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetNvSciSyncAttributesType: {
                cudaDeviceGetNvSciSyncAttributesRq& req = *reinterpret_cast<cudaDeviceGetNvSciSyncAttributesRq*>(request_ptr);

                cudaDeviceGetNvSciSyncAttributesRp& response = *((cudaDeviceGetNvSciSyncAttributesRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetNvSciSyncAttributes(response.nvSciSyncAttrList, req.device, req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetP2PAttributeType: {
                cudaDeviceGetP2PAttributeRq& req = *reinterpret_cast<cudaDeviceGetP2PAttributeRq*>(request_ptr);

                cudaDeviceGetP2PAttributeRp& response = *((cudaDeviceGetP2PAttributeRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetP2PAttribute(&response.value, req.attr, req.srcDevice, req.dstDevice);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceRegisterAsyncNotificationType: {
                cudaDeviceRegisterAsyncNotificationRq& req = *reinterpret_cast<cudaDeviceRegisterAsyncNotificationRq*>(request_ptr);

                cudaDeviceRegisterAsyncNotificationRp& response = *((cudaDeviceRegisterAsyncNotificationRp*)(NewResponse(response_rb)));
                response.result =
                    cudaDeviceRegisterAsyncNotification(req.device, req.callbackFunc, req.userData, &response.callback);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaDeviceGetPCIBusIdType: {
                cudaDeviceGetPCIBusIdRq& req = *reinterpret_cast<cudaDeviceGetPCIBusIdRq*>(request_ptr);

                cudaDeviceGetPCIBusIdRp& response = *((cudaDeviceGetPCIBusIdRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetPCIBusId(response.pciBusId, sizeof(response.pciBusId), req.device);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetStreamPriorityRangeType: {
                cudaDeviceGetStreamPriorityRangeRq& req = *reinterpret_cast<cudaDeviceGetStreamPriorityRangeRq*>(request_ptr);

                cudaDeviceGetStreamPriorityRangeRp& response = *((cudaDeviceGetStreamPriorityRangeRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceGetStreamPriorityRange(&response.leastPriority, &response.greatestPriority);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceGetTexture1DLinearMaxWidthType: {
                cudaDeviceGetTexture1DLinearMaxWidthRq& req = *reinterpret_cast<cudaDeviceGetTexture1DLinearMaxWidthRq*>(request_ptr);

                cudaDeviceGetTexture1DLinearMaxWidthRp& response = *((cudaDeviceGetTexture1DLinearMaxWidthRp*)(NewResponse(response_rb)));
                response.result =
                    cudaDeviceGetTexture1DLinearMaxWidth(&response.maxWidthInElements, &req.fmtDesc, req.device);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaDeviceResetType: {
                cudaDeviceResetRq& req = *reinterpret_cast<cudaDeviceResetRq*>(request_ptr);

                cudaDeviceResetRp& response = *((cudaDeviceResetRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceReset();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceSetCacheConfigType: {
                cudaDeviceSetCacheConfigRq& req = *reinterpret_cast<cudaDeviceSetCacheConfigRq*>(request_ptr);

                cudaDeviceSetCacheConfigRp& response = *((cudaDeviceSetCacheConfigRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceSetCacheConfig(req.cacheConfig);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceSetLimitType: {
                cudaDeviceSetLimitRq& req = *reinterpret_cast<cudaDeviceSetLimitRq*>(request_ptr);

                cudaDeviceSetLimitRp& response = *((cudaDeviceSetLimitRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceSetLimit(req.limit, req.value);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceSetMemPoolType: {
                cudaDeviceSetMemPoolRq& req = *reinterpret_cast<cudaDeviceSetMemPoolRq*>(request_ptr);

                cudaDeviceSetMemPoolRp& response = *((cudaDeviceSetMemPoolRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceSetMemPool(req.device, req.memPool);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceSynchronizeType: {
                cudaDeviceSynchronizeRq& req = *reinterpret_cast<cudaDeviceSynchronizeRq*>(request_ptr);

                cudaDeviceSynchronizeRp& response = *((cudaDeviceSynchronizeRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceSynchronize();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDeviceUnregisterAsyncNotificationType: {
                cudaDeviceUnregisterAsyncNotificationRq& req = *reinterpret_cast<cudaDeviceUnregisterAsyncNotificationRq*>(request_ptr);

                cudaDeviceUnregisterAsyncNotificationRp& response = *((cudaDeviceUnregisterAsyncNotificationRp*)(NewResponse(response_rb)));
                response.result = cudaDeviceUnregisterAsyncNotification(req.device, req.callback);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaInitDeviceType: {
                cudaInitDeviceRq& req = *reinterpret_cast<cudaInitDeviceRq*>(request_ptr);

                cudaInitDeviceRp& response = *((cudaInitDeviceRp*)(NewResponse(response_rb)));
                response.result = cudaInitDevice(req.device, req.deviceFlags, req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaIpcCloseMemHandleType: {
                cudaIpcCloseMemHandleRq& req = *reinterpret_cast<cudaIpcCloseMemHandleRq*>(request_ptr);

                cudaIpcCloseMemHandleRp& response = *((cudaIpcCloseMemHandleRp*)(NewResponse(response_rb)));
                response.result = cudaIpcCloseMemHandle(req.devPtr);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaIpcGetEventHandleType: {
                cudaIpcGetEventHandleRq& req = *reinterpret_cast<cudaIpcGetEventHandleRq*>(request_ptr);

                cudaIpcGetEventHandleRp& response = *((cudaIpcGetEventHandleRp*)(NewResponse(response_rb)));
                response.result = cudaIpcGetEventHandle(&response.handle, req.event);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaIpcGetMemHandleType: {
                cudaIpcGetMemHandleRq& req = *reinterpret_cast<cudaIpcGetMemHandleRq*>(request_ptr);

                cudaIpcGetMemHandleRp& response = *((cudaIpcGetMemHandleRp*)(NewResponse(response_rb)));
                response.result = cudaIpcGetMemHandle(&response.handle, req.devPtr);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaIpcOpenEventHandleType: {
                cudaIpcOpenEventHandleRq& req = *reinterpret_cast<cudaIpcOpenEventHandleRq*>(request_ptr);

                cudaIpcOpenEventHandleRp& response = *((cudaIpcOpenEventHandleRp*)(NewResponse(response_rb)));
                response.result = cudaIpcOpenEventHandle(&response.event, req.handle);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaIpcOpenMemHandleType: {
                cudaIpcOpenMemHandleRq& req = *reinterpret_cast<cudaIpcOpenMemHandleRq*>(request_ptr);

                cudaIpcOpenMemHandleRp& response = *((cudaIpcOpenMemHandleRp*)(NewResponse(response_rb)));
                response.result = cudaIpcOpenMemHandle(&response.devPtr, req.handle, req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaCtxResetPersistingL2CacheType: {
                cudaCtxResetPersistingL2CacheRq& req = *reinterpret_cast<cudaCtxResetPersistingL2CacheRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaCtxResetPersistingL2Cache();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamAddCallbackType: {
                cudaStreamAddCallbackRq& req = *reinterpret_cast<cudaStreamAddCallbackRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result =
                    cudaStreamAddCallback(req.stream, req.callback, req.userData, req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamAttachMemAsyncType: {
                cudaStreamAttachMemAsyncRq& req = *reinterpret_cast<cudaStreamAttachMemAsyncRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result =
                    cudaStreamAttachMemAsync(req.stream, req.devPtr, req.length, req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamBeginCaptureType: {
                cudaStreamBeginCaptureRq& req = *reinterpret_cast<cudaStreamBeginCaptureRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaStreamBeginCapture(req.stream, req.mode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamBeginCaptureToGraphType: {
                cudaStreamBeginCaptureToGraphRq& req = *reinterpret_cast<cudaStreamBeginCaptureToGraphRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaStreamBeginCaptureToGraph(
                    req.stream,
                    req.graph,
                    req.dependencies,
                    req.dependencyData,
                    req.numDependencies,
                    req.mode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamCopyAttributesType: {
                cudaStreamCopyAttributesRq& req = *reinterpret_cast<cudaStreamCopyAttributesRq*>(request_ptr);

                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaStreamCopyAttributes(req.dst, req.src);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamCreateWithFlagsType: {
                cudaStreamCreateWithFlagsRq& req = *reinterpret_cast<cudaStreamCreateWithFlagsRq*>(request_ptr);

                cudaStreamCreateWithFlagsRp& response = *((cudaStreamCreateWithFlagsRp*)(NewResponse(response_rb)));
                response.result = cudaStreamCreateWithFlags(&response.stream, req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamCreateWithPriorityType: {
                cudaStreamCreateWithPriorityRq& req = *reinterpret_cast<cudaStreamCreateWithPriorityRq*>(request_ptr);

                cudaStreamCreateWithPriorityRp& response = *((cudaStreamCreateWithPriorityRp*)(NewResponse(response_rb)));
                response.result =
                    cudaStreamCreateWithPriority(&response.stream, req.flags, req.priority);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamEndCaptureType: {
                cudaStreamEndCaptureRq& req = *reinterpret_cast<cudaStreamEndCaptureRq*>(request_ptr);

                cudaStreamEndCaptureRp& response = *((cudaStreamEndCaptureRp*)(NewResponse(response_rb)));
                response.result = cudaStreamEndCapture(req.stream, &response.graph);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamGetAttributeType: {
                cudaStreamGetAttributeRq& req = *reinterpret_cast<cudaStreamGetAttributeRq*>(request_ptr);

                cudaStreamGetAttributeRp& response = *((cudaStreamGetAttributeRp*)(NewResponse(response_rb)));
                response.result = cudaStreamGetAttribute(req.hStream, req.attr, &response.value);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamGetCaptureInfoType: {
                cudaStreamGetCaptureInfoRq& req = *reinterpret_cast<cudaStreamGetCaptureInfoRq*>(request_ptr);

                cudaStreamGetCaptureInfoRp& response = *((cudaStreamGetCaptureInfoRp*)(NewResponse(response_rb)));
                response.result = cudaStreamGetCaptureInfo(
                    req.stream,
                    &response.captureStatus,
                    &response.id,
                    &response.graph,
                    &response.dependencies,
                    &response.numDependencies);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamGetCaptureInfo_v3Type: {
                cudaStreamGetCaptureInfo_v3Rq& req = *reinterpret_cast<cudaStreamGetCaptureInfo_v3Rq*>(request_ptr);

                cudaStreamGetCaptureInfo_v3Rp& response = *((cudaStreamGetCaptureInfo_v3Rp*)(NewResponse(response_rb)));
                response.result = cudaStreamGetCaptureInfo_v3(
                    req.stream,
                    &response.captureStatus,
                    &response.id,
                    &response.graph,
                    &response.dependencies,
                    &response.edgeData,
                    &response.numDependencies);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamGetFlagsType: {
                cudaStreamGetFlagsRq& req = *reinterpret_cast<cudaStreamGetFlagsRq*>(request_ptr);

                cudaStreamGetFlagsRp& response = *((cudaStreamGetFlagsRp*)(NewResponse(response_rb)));
                response.result = cudaStreamGetFlags(req.hStream, &response.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamGetIdType: {
                cudaStreamGetIdRq& req = *reinterpret_cast<cudaStreamGetIdRq*>(request_ptr);

                cudaStreamGetIdRp& response = *((cudaStreamGetIdRp*)(NewResponse(response_rb)));
                response.result = cudaStreamGetId(req.hStream, &response.streamId);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamGetPriorityType: {
                cudaStreamGetPriorityRq& req = *reinterpret_cast<cudaStreamGetPriorityRq*>(request_ptr);

                cudaStreamGetPriorityRp& response = *((cudaStreamGetPriorityRp*)(NewResponse(response_rb)));
                response.result = cudaStreamGetPriority(req.hStream, &response.priority);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamIsCapturingType: {
                cudaStreamIsCapturingRq& req = *reinterpret_cast<cudaStreamIsCapturingRq*>(request_ptr);

                cudaStreamIsCapturingRp& response = *((cudaStreamIsCapturingRp*)(NewResponse(response_rb)));
                response.result = cudaStreamIsCapturing(req.stream, &response.status);
                if (response.result == cudaSuccess) {

                }

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));

                break;
            }
            case cudaStreamSetAttributeType: {
                cudaStreamSetAttributeRq& req = *reinterpret_cast<cudaStreamSetAttributeRq*>(request_ptr);

                cudaStreamSetAttributeRp& response = *((cudaStreamSetAttributeRp*)(NewResponse(response_rb)));
                response.result = cudaStreamSetAttribute(req.hStream, req.attr, &req.value);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamUpdateCaptureDependenciesType: {
                cudaStreamUpdateCaptureDependenciesRq& req = *reinterpret_cast<cudaStreamUpdateCaptureDependenciesRq*>(request_ptr);

                cudaStreamUpdateCaptureDependenciesRp& response = *((cudaStreamUpdateCaptureDependenciesRp*)(NewResponse(response_rb)));
                response.result = cudaStreamUpdateCaptureDependencies(
                    req.stream, req.dependencies, req.numDependencies, req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamUpdateCaptureDependencies_v2Type: {
                cudaStreamUpdateCaptureDependencies_v2Rq& req = *reinterpret_cast<cudaStreamUpdateCaptureDependencies_v2Rq*>(request_ptr);

                cudaStreamUpdateCaptureDependencies_v2Rp& response = *((cudaStreamUpdateCaptureDependencies_v2Rp*)(NewResponse(response_rb)));
                response.result = cudaStreamUpdateCaptureDependencies_v2(
                    req.stream,
                    req.dependencies,
                    req.dependencyData,
                    req.numDependencies,
                    req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaThreadExchangeStreamCaptureModeType: {
                cudaThreadExchangeStreamCaptureModeRq& req = *reinterpret_cast<cudaThreadExchangeStreamCaptureModeRq*>(request_ptr);

                cudaThreadExchangeStreamCaptureModeRp& response = *((cudaThreadExchangeStreamCaptureModeRp*)(NewResponse(response_rb)));
                response.result = cudaThreadExchangeStreamCaptureMode(&response.mode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaFuncGetNameType: {
                cudaFuncGetNameRq& req = *reinterpret_cast<cudaFuncGetNameRq*>(request_ptr);

                cudaFuncGetNameRp& response = *((cudaFuncGetNameRp*)(NewResponse(response_rb)));

                const char *name1 = "123";
                const char **name_use = &(name1);
                response.result = cudaFuncGetName(response.name, req.func);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaFuncGetParamInfoType: {
                cudaFuncGetParamInfoRq& req = *reinterpret_cast<cudaFuncGetParamInfoRq*>(request_ptr);

                cudaFuncGetParamInfoRp& response = *((cudaFuncGetParamInfoRp*)(NewResponse(response_rb)));
                response.result = cudaFuncGetParamInfo(
                    req.func, req.paramIndex, &response.paramOffset, &response.paramSize);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaFuncSetAttributeType: {
                cudaFuncSetAttributeRq& req = *reinterpret_cast<cudaFuncSetAttributeRq*>(request_ptr);

                cudaFuncSetAttributeRp& response = *((cudaFuncSetAttributeRp*)(NewResponse(response_rb)));
                response.result = cudaFuncSetAttribute(req.func, req.attr, req.value);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaLaunchCooperativeKernelType: {
                cudaLaunchCooperativeKernelRq& req = *reinterpret_cast<cudaLaunchCooperativeKernelRq*>(request_ptr);

                cudaLaunchCooperativeKernelRp& response = *((cudaLaunchCooperativeKernelRp*)(NewResponse(response_rb)));
                response.result = cudaLaunchCooperativeKernel(
                    req.func,
                    req.gridDim,
                    req.blockDim,
                    req.args,
                    req.sharedMem,
                    req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaLaunchCooperativeKernelMultiDeviceType: {
                cudaLaunchCooperativeKernelMultiDeviceRq& req = *reinterpret_cast<cudaLaunchCooperativeKernelMultiDeviceRq*>(request_ptr);

                cudaLaunchCooperativeKernelMultiDeviceRp& response = *((cudaLaunchCooperativeKernelMultiDeviceRp*)(NewResponse(response_rb)));
                response.result = cudaLaunchCooperativeKernelMultiDevice(
                    req.launchParamsList,
                    req.numDevices,
                    req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaLaunchHostFuncType: {
                cudaLaunchHostFuncRq& req = *reinterpret_cast<cudaLaunchHostFuncRq*>(request_ptr);

                cudaLaunchHostFuncRp& response = *((cudaLaunchHostFuncRp*)(NewResponse(response_rb)));
                response.result = cudaLaunchHostFunc(req.stream, req.fn, req.userData);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaLaunchKernelType: {
                cudaLaunchKernelRq& req = *reinterpret_cast<cudaLaunchKernelRq*>(request_ptr);

                CUmodule module;
                CUfunction function;

                char* kernel_name_pointer = shared_mem + (size_t)((char*)req.kernel_name_pointer - req.client_shared_mem);
                char* cubin_file_path_pointer = shared_mem + (size_t)((char*)req.cubin_file_path_pointer - req.client_shared_mem);
                std::string kernel_name(kernel_name_pointer);
                std::string cubin_file_path(cubin_file_path_pointer);

                std::lock_guard<std::mutex> lock(g_cache_mutex);

                auto mod_iter = g_module_cache.find(cubin_file_path);
                if (mod_iter == g_module_cache.end()) {

                    CUresult res = cuModuleLoad(&module, cubin_file_path.c_str());
                    if (res != CUDA_SUCCESS) {
                        std::cerr << "Failed to load module: " << cubin_file_path
                                  << " Error: " << res << std::endl;
                        cudaLaunchKernelRp& response = *((cudaLaunchKernelRp*)(NewResponse(response_rb)));
                        response.result = cudaErrorUnknown;
                        response.responseSize = sizeof(response);
                        ResponseSend(response_rb, sizeof(response));
                        break;
                    }
                    g_module_cache[cubin_file_path] = module;
                } else {
                    module = mod_iter->second;
                }

                std::string func_key = cubin_file_path + "||" + kernel_name;
                auto func_iter = g_function_cache.find(func_key);
                if (func_iter == g_function_cache.end()) {
                    CUresult res = cuModuleGetFunction(&function, module, kernel_name.c_str());
                    if (res != CUDA_SUCCESS) {
                        std::cerr << "Failed to get function: " << kernel_name
                                  << " Error: " << res << std::endl;
                        cudaLaunchKernelRp& response = *((cudaLaunchKernelRp*)(NewResponse(response_rb)));
                        response.result = cudaErrorUnknown;
                        response.responseSize = sizeof(response);
                        ResponseSend(response_rb, sizeof(response));
                        break;
                    }
                    g_function_cache[func_key] = function;
                } else {
                    function = func_iter->second;
                }

                const void *func = shared_mem + (size_t)((char *)req.func - req.client_shared_mem);
                void **args = (void **)(shared_mem + (size_t)((char *)req.args - req.client_shared_mem));
                for (int i = 0; i < req.argsSize; ++i) {
                    args[i] = (void *)(shared_mem + (size_t)((char *)args[i] - req.client_shared_mem));
                }

                CUresult res = cuLaunchKernel(
                    function,
                    req.gridDim.x, req.gridDim.y, req.gridDim.z,
                    req.blockDim.x, req.blockDim.y, req.blockDim.z,
                    req.sharedMem,
                    reinterpret_cast<CUstream>(req.stream),
                    args,
                    nullptr
                );

                cudaLaunchKernelRp& response = *((cudaLaunchKernelRp*)(NewResponse(response_rb)));
                if (res == CUDA_SUCCESS) {
                    response.result = cudaSuccess;
                } else {
                    std::cerr << "Launch failed: " << res << std::endl;
                    response.result = cudaErrorUnknown;
                }
                response.responseSize = sizeof(response);
                ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSgemmType: {
                cublasSgemmRq& req = *reinterpret_cast<cublasSgemmRq*>(request_ptr);

                cublasSgemmRp& response = *((cublasSgemmRp*)(NewResponse(response_rb)));
                const float *alpha = &req.alpha;
                const float *beta = &req.beta;
                response.status = cublasSgemm(
                    req.handle,
                    req.transa,
                    req.transb,
                    req.m,
                    req.n,
                    req.k,
                    alpha,
                    req.A,
                    req.lda,
                    req.B,
                    req.ldb,
                    beta,
                    req.C,
                    req.ldc);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSgemmStridedBatchedType: {
                cublasSgemmStridedBatchedRq& req = *reinterpret_cast<cublasSgemmStridedBatchedRq*>(request_ptr);

                cublasSgemmStridedBatchedRp& response = *((cublasSgemmStridedBatchedRp*)(NewResponse(response_rb)));
                const float *alpha = &req.alpha;
                const float *beta = &req.beta;

                response.status = cublasSgemmStridedBatched(
                    req.handle,
                    req.transa,
                    req.transb,
                    req.m,
                    req.n,
                    req.k,
                    alpha,
                    req.A,
                    req.lda,
                    req.strideA,
                    req.B,
                    req.ldb,
                    req.strideB,
                    beta,
                    req.C,
                    req.ldc,
                    req.strideC,
                    req.batchCount);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasCreateType: {
                cublasCreateRq& req = *reinterpret_cast<cublasCreateRq*>(request_ptr);

                cublasCreateRp& response = *((cublasCreateRp*)(NewResponse(response_rb)));
                cublasHandle_t handle = nullptr;
                response.status  = cublasCreate(&handle);
                response.handle = handle;

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetStreamType: {
                cublasSetStreamRq& req = *reinterpret_cast<cublasSetStreamRq*>(request_ptr);

                cublasSetStreamRp& response = *((cublasSetStreamRp*)(NewResponse(response_rb)));

                response.status = cublasSetStream(req.handle, req.streamId);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasDestroyType: {
                cublasDestroyRq& req = *reinterpret_cast<cublasDestroyRq*>(request_ptr);

                cublasDestroyRp& response = *((cublasDestroyRp*)(NewResponse(response_rb)));

                response.status = cublasDestroy(req.handle);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetPropertyType: {
                cublasGetPropertyRq& req = *reinterpret_cast<cublasGetPropertyRq*>(request_ptr);

                cublasGetPropertyRp& response = *((cublasGetPropertyRp*)(NewResponse(response_rb)));

                response.status = cublasGetProperty(req.type, &response.value);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cublasSetWorkspaceType: {
                cublasSetWorkspaceRq& req = *reinterpret_cast<cublasSetWorkspaceRq*>(request_ptr);

                cublasSetWorkspaceRp& response = *((cublasSetWorkspaceRp*)(NewResponse(response_rb)));

                response.status = cublasSetWorkspace(req.handle, req.workspace, req.workspaceSizeInBytes);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetStreamType: {
                cublasGetStreamRq& req = *reinterpret_cast<cublasGetStreamRq*>(request_ptr);

                cublasGetStreamRp& response = *((cublasGetStreamRp*)(NewResponse(response_rb)));

                response.status = cublasGetStream(req.handle, &response.streamId);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetPointerModeType: {
                cublasGetPointerModeRq& req = *reinterpret_cast<cublasGetPointerModeRq*>(request_ptr);

                cublasGetPointerModeRp& response = *((cublasGetPointerModeRp*)(NewResponse(response_rb)));

                response.status = cublasGetPointerMode(req.handle, &response.mode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetPointerModeType: {
                cublasSetPointerModeRq& req = *reinterpret_cast<cublasSetPointerModeRq*>(request_ptr);

                cublasSetPointerModeRp& response = *((cublasSetPointerModeRp*)(NewResponse(response_rb)));

                response.status = cublasSetPointerMode(req.handle, req.mode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetVectorType: {
                cublasSetVectorRq& req = *reinterpret_cast<cublasSetVectorRq*>(request_ptr);

                cublasSetVectorRp& response = *((cublasSetVectorRp*)(NewResponse(response_rb)));

                response.status = cublasSetVector(req.n, req.elemSize, req.x, req.incx, req.y, req.incy);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetVectorType: {
                cublasGetVectorRq& req = *reinterpret_cast<cublasGetVectorRq*>(request_ptr);

                cublasGetVectorRp& response = *((cublasGetVectorRp*)(NewResponse(response_rb)));

                response.status = cublasGetVector(req.n, req.elemSize, req.x, req.incx, req.y, req.incy);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetMatrixType: {
                cublasGetMatrixRq& req = *reinterpret_cast<cublasGetMatrixRq*>(request_ptr);

                cublasGetMatrixRp& response = *((cublasGetMatrixRp*)(NewResponse(response_rb)));

                response.status = cublasGetMatrix(req.rows, req.cols, req.elemSize, req.A, req.lda, req.B, req.ldb);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetMatrixType: {
                cublasSetMatrixRq& req = *reinterpret_cast<cublasSetMatrixRq*>(request_ptr);

                cublasSetMatrixRp& response = *((cublasSetMatrixRp*)(NewResponse(response_rb)));

                cudaMemcpy(req.B, req.A, req.rows * req.cols * req.elemSize, cudaMemcpyHostToDevice);

                response.status = CUBLAS_STATUS_SUCCESS;

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetVectorAsyncType: {
                cublasSetVectorAsyncRq& req = *reinterpret_cast<cublasSetVectorAsyncRq*>(request_ptr);

                cublasSetVectorAsyncRp& response = *((cublasSetVectorAsyncRp*)(NewResponse(response_rb)));

                response.status = cublasSetVectorAsync(req.n, req.elemSize, req.hostPtr, req.incx, req.devicePtr, req.incy, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetVectorAsyncType: {
                cublasGetVectorAsyncRq& req = *reinterpret_cast<cublasGetVectorAsyncRq*>(request_ptr);

                cublasGetVectorAsyncRp& response = *((cublasGetVectorAsyncRp*)(NewResponse(response_rb)));

                response.status = cublasGetVectorAsync(req.n, req.elemSize, req.devicePtr, req.incx, req.hostPtr, req.incy, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetMatrixAsyncType: {
                cublasSetMatrixAsyncRq& req = *reinterpret_cast<cublasSetMatrixAsyncRq*>(request_ptr);

                cublasSetMatrixAsyncRp& response = *((cublasSetMatrixAsyncRp*)(NewResponse(response_rb)));

                response.status = cublasSetMatrixAsync(req.rows, req.cols, req.elemSize, req.A, req.lda, req.B, req.ldb, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetMatrixAsyncType: {
                cublasGetMatrixAsyncRq& req = *reinterpret_cast<cublasGetMatrixAsyncRq*>(request_ptr);

                cublasGetMatrixAsyncRp& response = *((cublasGetMatrixAsyncRp*)(NewResponse(response_rb)));

                response.status = cublasGetMatrixAsync(req.rows, req.cols, req.elemSize, req.A, req.lda, req.B, req.ldb, req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetAtomicsModeType: {
                cublasSetAtomicsModeRq& req = *reinterpret_cast<cublasSetAtomicsModeRq*>(request_ptr);

                cublasSetAtomicsModeRp& response = *((cublasSetAtomicsModeRp*)(NewResponse(response_rb)));

                response.status = cublasSetAtomicsMode(req.handle, req.mode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetAtomicsModeType: {
                cublasGetAtomicsModeRq& req = *reinterpret_cast<cublasGetAtomicsModeRq*>(request_ptr);

                cublasGetAtomicsModeRp& response = *((cublasGetAtomicsModeRp*)(NewResponse(response_rb)));

                response.status = cublasGetAtomicsMode(req.handle, &response.atomicsMode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetMathModeType: {
                cublasSetMathModeRq& req = *reinterpret_cast<cublasSetMathModeRq*>(request_ptr);

                cublasSetMathModeRp& response = *((cublasSetMathModeRp*)(NewResponse(response_rb)));

                response.status = cublasSetMathMode(req.handle, req.mode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetMathModeType: {
                cublasGetMathModeRq& req = *reinterpret_cast<cublasGetMathModeRq*>(request_ptr);

                cublasGetMathModeRp& response = *((cublasGetMathModeRp*)(NewResponse(response_rb)));

                response.status = cublasGetMathMode(req.handle, &response.mathMode);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetSmCountTargetType: {
                cublasSetSmCountTargetRq& req = *reinterpret_cast<cublasSetSmCountTargetRq*>(request_ptr);

                cublasSetSmCountTargetRp& response = *((cublasSetSmCountTargetRp*)(NewResponse(response_rb)));

                response.status = cublasSetSmCountTarget(req.handle, req.smCountTarget);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetSmCountTargetType: {
                cublasGetSmCountTargetRq& req = *reinterpret_cast<cublasGetSmCountTargetRq*>(request_ptr);

                cublasGetSmCountTargetRp& response = *((cublasGetSmCountTargetRp*)(NewResponse(response_rb)));

                response.status = cublasGetSmCountTarget(req.handle, &response.smCountTarget);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLoggerConfigureType: {
                cublasLoggerConfigureRq& req = *reinterpret_cast<cublasLoggerConfigureRq*>(request_ptr);

                cublasLoggerConfigureRp& response = *((cublasLoggerConfigureRp*)(NewResponse(response_rb)));

                response.status = cublasLoggerConfigure(req.logIsOn, req.logToStdOut, req.logToStdErr, req.logFileName);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasGetLoggerCallbackType: {
                cublasGetLoggerCallbackRq& req = *reinterpret_cast<cublasGetLoggerCallbackRq*>(request_ptr);

                cublasGetLoggerCallbackRp& response = *((cublasGetLoggerCallbackRp*)(NewResponse(response_rb)));

                response.status = cublasGetLoggerCallback(&response.userCallback);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasSetLoggerCallbackType: {
                cublasSetLoggerCallbackRq& req = *reinterpret_cast<cublasSetLoggerCallbackRq*>(request_ptr);

                cublasSetLoggerCallbackRp& response = *((cublasSetLoggerCallbackRp*)(NewResponse(response_rb)));

                response.status = cublasSetLoggerCallback(req.userCallback);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulAlgoGetHeuristicType: {
                cublasLtMatmulAlgoGetHeuristicRq& req = *reinterpret_cast<cublasLtMatmulAlgoGetHeuristicRq*>(request_ptr);

                cublasLtMatmulAlgoGetHeuristicRp& response = *((cublasLtMatmulAlgoGetHeuristicRp*)(NewResponse(response_rb)));
                size_t size = req.requestedAlgoCount * sizeof(cublasLtMatmulHeuristicResult_t);
                if (req.shared_mem_offset + size > (long)1024 * 1024 * 1024) {
                    response.status = CUBLAS_STATUS_EXECUTION_FAILED;
                    response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                    break;
                }
                cublasLtMatmulHeuristicResult_t * heuristicResultsArray = (cublasLtMatmulHeuristicResult_t *)(shared_mem + req.shared_mem_offset);

                response.status = cublasLtMatmulAlgoGetHeuristic(
                    req.lightHandle,
                    req.operationDesc,
                    req.Adesc,
                    req.Bdesc,
                    req.Cdesc,
                    req.Ddesc,
                    req.preference,
                    req.requestedAlgoCount,
                    heuristicResultsArray,
                    &response.returnAlgoCount
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaLaunchKernelExCType: {
                cudaLaunchKernelExCRq& req = *reinterpret_cast<cudaLaunchKernelExCRq*>(request_ptr);

                cudaLaunchKernelExCRp& response = *((cudaLaunchKernelExCRp*)(NewResponse(response_rb)));
                response.result = cudaLaunchKernelExC(&req.config, req.func,
                                                      req.args);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDestroyExternalMemoryType: {
                cudaDestroyExternalMemoryRq& req = *reinterpret_cast<cudaDestroyExternalMemoryRq*>(request_ptr);

                cudaDestroyExternalMemoryRp& response = *((cudaDestroyExternalMemoryRp*)(NewResponse(response_rb)));
                response.result = cudaDestroyExternalMemory(req.extMem);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaDestroyExternalSemaphoreType: {
                cudaDestroyExternalSemaphoreRq& req = *reinterpret_cast<cudaDestroyExternalSemaphoreRq*>(request_ptr);

                cudaDestroyExternalSemaphoreRp& response = *((cudaDestroyExternalSemaphoreRp*)(NewResponse(response_rb)));
                response.result = cudaDestroyExternalSemaphore(req.extSem);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaExternalMemoryGetMappedBufferType: {
                cudaExternalMemoryGetMappedBufferRq& req = *reinterpret_cast<cudaExternalMemoryGetMappedBufferRq*>(request_ptr);

                cudaExternalMemoryGetMappedBufferRp& response = *((cudaExternalMemoryGetMappedBufferRp*)(NewResponse(response_rb)));
                response.result = cudaExternalMemoryGetMappedBuffer(
                    &response.devPtr,
                    req.extMem,
                    &req.bufferDesc);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaExternalMemoryGetMappedMipmappedArrayType: {
                cudaExternalMemoryGetMappedMipmappedArrayRq& req = *reinterpret_cast<cudaExternalMemoryGetMappedMipmappedArrayRq*>(request_ptr);

                cudaExternalMemoryGetMappedMipmappedArrayRp& response = *((cudaExternalMemoryGetMappedMipmappedArrayRp*)(NewResponse(response_rb)));
                response.result = cudaExternalMemoryGetMappedMipmappedArray(
                    &response.mipmap,
                    req.extMem,
                    &req.mipmapDesc);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaImportExternalMemoryType: {
                cudaImportExternalMemoryRq& req = *reinterpret_cast<cudaImportExternalMemoryRq*>(request_ptr);

                cudaImportExternalMemoryRp& response = *((cudaImportExternalMemoryRp*)(NewResponse(response_rb)));
                response.result = cudaImportExternalMemory(req.extMem_out,
                                                           &req.memHandleDesc);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaImportExternalSemaphoreType: {
                cudaImportExternalSemaphoreRq& req = *reinterpret_cast<cudaImportExternalSemaphoreRq*>(request_ptr);

                cudaImportExternalSemaphoreRp& response = *((cudaImportExternalSemaphoreRp*)(NewResponse(response_rb)));
                response.result = cudaImportExternalSemaphore(req.extSem_out,
                                                              &req.semHandleDesc);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaSignalExternalSemaphoresAsyncType: {
                cudaSignalExternalSemaphoresAsyncRq& req = *reinterpret_cast<cudaSignalExternalSemaphoresAsyncRq*>(request_ptr);

                cudaSignalExternalSemaphoresAsyncRp& response = *((cudaSignalExternalSemaphoresAsyncRp*)(NewResponse(response_rb)));
                response.result = cudaSignalExternalSemaphoresAsync(
                    req.extSemArray,
                    req.paramsArray,
                    req.numExtSems,
                    req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaWaitExternalSemaphoresAsyncType: {
                cudaWaitExternalSemaphoresAsyncRq& req = *reinterpret_cast<cudaWaitExternalSemaphoresAsyncRq*>(request_ptr);

                cudaWaitExternalSemaphoresAsyncRp& response = *((cudaWaitExternalSemaphoresAsyncRp*)(NewResponse(response_rb)));
                response.result = cudaWaitExternalSemaphoresAsync(
                    req.extSemArray,
                    req.paramsArray,
                    req.numExtSems,
                    req.stream);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaOccupancyAvailableDynamicSMemPerBlockType: {
                cudaOccupancyAvailableDynamicSMemPerBlockRq& req = *reinterpret_cast<cudaOccupancyAvailableDynamicSMemPerBlockRq*>(request_ptr);

                cudaOccupancyAvailableDynamicSMemPerBlockRp& response = *((cudaOccupancyAvailableDynamicSMemPerBlockRp*)(NewResponse(response_rb)));
                response.result = cudaOccupancyAvailableDynamicSMemPerBlock(
                    &response.dynamicSmemSize,
                    req.func,
                    req.numBlocks,
                    req.blockSize);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaOccupancyMaxActiveBlocksPerMultiprocessorType: {
                cudaOccupancyMaxActiveBlocksPerMultiprocessorRq& req = *reinterpret_cast<cudaOccupancyMaxActiveBlocksPerMultiprocessorRq*>(request_ptr);

                cudaOccupancyMaxActiveBlocksPerMultiprocessorRp& response = *((cudaOccupancyMaxActiveBlocksPerMultiprocessorRp*)(NewResponse(response_rb)));
                response.result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &response.numBlocks,
                    req.func,
                    req.blockSize,
                    req.dynamicSMemSize);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsType: {
                cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRq& req = *reinterpret_cast<cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRq*>(request_ptr);

                cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRp& response = *((cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRp*)(NewResponse(response_rb)));
                response.result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &response.numBlocks,
                    req.func,
                    req.blockSize,
                    req.dynamicSMemSize,
                    req.flags);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaOccupancyMaxActiveClustersType: {
                cudaOccupancyMaxActiveClustersRq& req = *reinterpret_cast<cudaOccupancyMaxActiveClustersRq*>(request_ptr);

                cudaOccupancyMaxActiveClustersRp& response = *((cudaOccupancyMaxActiveClustersRp*)(NewResponse(response_rb)));
                response.result = cudaOccupancyMaxActiveClusters(
                    &response.numClusters,
                    req.func,
                    &req.launchConfig);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaOccupancyMaxPotentialClusterSizeType: {
                cudaOccupancyMaxPotentialClusterSizeRq& req = *reinterpret_cast<cudaOccupancyMaxPotentialClusterSizeRq*>(request_ptr);

                cudaOccupancyMaxPotentialClusterSizeRp& response = *((cudaOccupancyMaxPotentialClusterSizeRp*)(NewResponse(response_rb)));
                response.result = cudaOccupancyMaxPotentialClusterSize(
                    &response.clusterSize,
                    req.func,
                    &req.launchConfig);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaStreamCreateType: {
                cudaStreamCreateRq& req = *reinterpret_cast<cudaStreamCreateRq*>(request_ptr);

                cudaStreamCreateRp& response = *((cudaStreamCreateRp*)(NewResponse(response_rb)));
                response.result = cudaStreamCreate(&response.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamDestroyType: {
                cudaStreamDestroyRq& req = *reinterpret_cast<cudaStreamDestroyRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaStreamDestroy(req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamQueryType: {
                cudaStreamQueryRq& req = *reinterpret_cast<cudaStreamQueryRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaStreamQuery(req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaStreamSynchronizeType: {
                cudaStreamSynchronizeRq& req = *reinterpret_cast<cudaStreamSynchronizeRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaStreamSynchronize(req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                printf("response.responseSize = sizeof(response); send_result: %ld\n", send_result);
                printf("response.result: %d\n", response.result);
                break;
            }
            case cudaStreamWaitEventType: {
                cudaStreamWaitEventRq& req = *reinterpret_cast<cudaStreamWaitEventRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaStreamWaitEvent(req.stream, req.event, req.flags);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaEventCreateType: {

                cudaEventCreateRq& req = *reinterpret_cast<cudaEventCreateRq*>(request_ptr);

                cudaEventCreateRp& response = *((cudaEventCreateRp*)(NewResponse(response_rb)));

                response.result = cudaEventCreate(&response.event);

                if (response.result == cudaSuccess) {

                }
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaEventQueryType: {
                cudaEventQueryRq& req = *reinterpret_cast<cudaEventQueryRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaEventQuery(req.event);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaEventCreateWithFlagsType: {
                cudaEventCreateWithFlagsRq& req = *reinterpret_cast<cudaEventCreateWithFlagsRq*>(request_ptr);

                cudaEventCreateRp& response = *((cudaEventCreateRp*)(NewResponse(response_rb)));
                response.result = cudaEventCreateWithFlags(&response.event, req.flags);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaEventDestroyType: {
                cudaEventDestroyRq& req = *reinterpret_cast<cudaEventDestroyRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaEventDestroy(req.event);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaEventElapsedTimeType: {
                cudaEventElapsedTimeRq& req = *reinterpret_cast<cudaEventElapsedTimeRq*>(request_ptr);

                cudaEventElapsedTimeRp& response = *((cudaEventElapsedTimeRp*)(NewResponse(response_rb)));
                response.result = cudaEventElapsedTime(&response.ms, req.start, req.end);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaEventRecordType: {
                cudaEventRecordRq& req = *reinterpret_cast<cudaEventRecordRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaEventRecord(req.event, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaEventSynchronizeType: {
                cudaEventSynchronizeRq& req = *reinterpret_cast<cudaEventSynchronizeRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaEventSynchronize(req.event);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaFuncGetAttributesType: {
                cudaFuncGetAttributesRq& req = *reinterpret_cast<cudaFuncGetAttributesRq*>(request_ptr);

                cudaFuncGetAttributesRp& response = *((cudaFuncGetAttributesRp*)(NewResponse(response_rb)));
                response.result = cudaFuncGetAttributes(&response.attr, req.func);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaFuncSetCacheConfigType: {
                cudaFuncSetCacheConfigRq& req = *reinterpret_cast<cudaFuncSetCacheConfigRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaFuncSetCacheConfig(req.func, req.cacheConfig);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaSetDoubleForDeviceType: {
                cudaSetDoubleForDeviceRq& req = *reinterpret_cast<cudaSetDoubleForDeviceRq*>(request_ptr);

                cudaSetDoubleForDeviceRp& response = *((cudaSetDoubleForDeviceRp*)(NewResponse(response_rb)));
                response.result = cudaSetDoubleForDevice(&response.d);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaSetDoubleForHostType: {
                cudaSetDoubleForHostRq& req = *reinterpret_cast<cudaSetDoubleForHostRq*>(request_ptr);

                cudaSetDoubleForHostRp& response = *((cudaSetDoubleForHostRp*)(NewResponse(response_rb)));
                response.result = cudaSetDoubleForHost(&response.d);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cudaFreeType: {
                cudaFreeRq& req = *reinterpret_cast<cudaFreeRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaFree(req.devPtr);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaFreeArrayType: {
                cudaFreeArrayRq& req = *reinterpret_cast<cudaFreeArrayRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaFreeArray(req.array);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaFreeHostType: {
                cudaFreeHostRq& req = *reinterpret_cast<cudaFreeHostRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaFreeHost(req.ptr);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaGetSymbolAddressType: {
                cudaGetSymbolAddressRq& req = *reinterpret_cast<cudaGetSymbolAddressRq*>(request_ptr);

                cudaGetSymbolAddressRp& response = *((cudaGetSymbolAddressRp*)(NewResponse(response_rb)));
                response.result = cudaGetSymbolAddress(&response.devPtr, req.symbol);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaGetSymbolSizeType: {
                cudaGetSymbolSizeRq& req = *reinterpret_cast<cudaGetSymbolSizeRq*>(request_ptr);

                cudaGetSymbolSizeRp& response = *((cudaGetSymbolSizeRp*)(NewResponse(response_rb)));
                response.result = cudaGetSymbolSize(&response.size, req.symbol);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaHostAllocType: {
                cudaHostAllocRq& req = *reinterpret_cast<cudaHostAllocRq*>(request_ptr);

                cudaHostAllocRp& response = *((cudaHostAllocRp*)(NewResponse(response_rb)));
                void *hostPtr = nullptr;
                void **pHost = &hostPtr;
                void *buf = (void*)(shared_mem + req.shmem_offset);
                printf("shmem pointer on server: %p\n", buf);
                printf("flags: %d", req.flags);

                response.result = cudaHostRegister(buf, req.size, cudaHostRegisterPortable);
                printf("register result: %d\n", response.result);
                response.pHost = NULL;
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaHostGetDevicePointerType: {
                cudaHostGetDevicePointerRq& req = *reinterpret_cast<cudaHostGetDevicePointerRq*>(request_ptr);

                cudaHostGetDevicePointerRp& response = *((cudaHostGetDevicePointerRp*)(NewResponse(response_rb)));
                response.result = cudaHostGetDevicePointer(&response.pDevice, req.pHost, req.flags);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaHostGetFlagsType: {
                cudaHostGetFlagsRq& req = *reinterpret_cast<cudaHostGetFlagsRq*>(request_ptr);

                cudaHostGetFlagsRp& response = *((cudaHostGetFlagsRp*)(NewResponse(response_rb)));
                response.result = cudaHostGetFlags(&response.pFlags, req.pHost);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMallocType: {
                cudaMallocRq& req = *reinterpret_cast<cudaMallocRq*>(request_ptr);
                cudaMallocRp& response = *((cudaMallocRp*)(NewResponse(response_rb)));

                printf("[Debug] before cudaMalloc, last error code is: %d\n", cudaGetLastError());
                printf("[Debug] req size %d\n", req.size);
                void* devPtr;
                cudaError_t result = cudaMalloc(&devPtr, req.size);
                printf("[Debug] Real cudaMalloc ptr %p with return code %d\n", devPtr, result);
                response.devPtr = devPtr;
                response.result = result;
                printf("[Debug] cudaMalloc ptr %p, size %ld\n", response.devPtr, req.size);
                printf("[Debug] after cudaMalloc, last error code is: %d\n", cudaGetLastError());

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMalloc3DType: {
                cudaMalloc3DRq& req = *reinterpret_cast<cudaMalloc3DRq*>(request_ptr);

                cudaMalloc3DRp& response = *((cudaMalloc3DRp*)(NewResponse(response_rb)));
                response.result = cudaMalloc3D(response.pitchedDevPtr, req.extent);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMalloc3DArrayType: {
                cudaMalloc3DArrayRq& req = *reinterpret_cast<cudaMalloc3DArrayRq*>(request_ptr);

                cudaMalloc3DArrayRp& response = *((cudaMalloc3DArrayRp*)(NewResponse(response_rb)));
                response.result = cudaMalloc3DArray(&response.array, req.desc, req.extent, req.flags);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMallocArrayType: {
                cudaMallocArrayRq& req = *reinterpret_cast<cudaMallocArrayRq*>(request_ptr);

                cudaMallocArrayRp& response = *((cudaMallocArrayRp*)(NewResponse(response_rb)));
                response.result = cudaMallocArray(&response.array, req.desc, req.width, req.height, req.flags);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMallocHostType: {
                cudaMallocHostRq& req = *reinterpret_cast<cudaMallocHostRq*>(request_ptr);

                cudaMallocHostRp& response = *((cudaMallocHostRp*)(NewResponse(response_rb)));
                response.result = cudaMallocHost(&response.ptr, req.size);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMallocPitchType: {
                cudaMallocPitchRq& req = *reinterpret_cast<cudaMallocPitchRq*>(request_ptr);

                cudaMallocPitchRp& response = *((cudaMallocPitchRp*)(NewResponse(response_rb)));
                response.result = cudaMallocPitch(&response.devPtr, &response.pitch, req.width, req.height);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyType: {
                cudaMemcpyRq& req = *reinterpret_cast<cudaMemcpyRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                cudaPointerAttributes dst_attributes;
                cudaPointerAttributes src_attributes;
                void *dst;
                const void *src;
                printf("server_shared_mem:%p\n", shared_mem);
                if (req.kind == cudaMemcpyHostToDevice) {
                    cudaError_t dst_status = cudaPointerGetAttributes(&dst_attributes, req.dst);
                    if (dst_status == cudaSuccess) {
                        if (dst_attributes.type == cudaMemoryTypeHost) {

                            dst = shared_mem + (size_t)((char *)req.dst - req.client_shared_mem);
                        } else if (dst_attributes.type == cudaMemoryTypeDevice) {
                            printf("dst pointer:%p", req.dst);

                            dst = req.dst;
                            src = shared_mem + (size_t)((char *)req.src - req.client_shared_mem);
                            printf("client_src:%p\n", req.src);
                            printf("server_src:%p\n", src);
                        } else if (dst_attributes.type == cudaMemoryTypeManaged) {

                        }
                    } else {

                        dst = shared_mem + (size_t)((char *)req.dst - req.client_shared_mem);

                    }
                }

                if (req.kind == cudaMemcpyDeviceToHost) {
                    cudaError_t src_status = cudaPointerGetAttributes(&src_attributes, req.src);
                    if (src_status == cudaSuccess) {
                        if (src_attributes.type == cudaMemoryTypeDevice) {

                            fflush(stdout);
                            src = req.src;

                            if (req.dst >= req.client_shared_mem && req.dst < req.client_shared_mem + 1024 * 1024 * 1024) {
                                dst = shared_mem + (size_t)((char *)req.dst - req.client_shared_mem);
                            }

                        }
                    } else {

                        src = shared_mem + (size_t)((char *)req.src - req.client_shared_mem);

                    }
                }

                if (req.kind == cudaMemcpyDeviceToDevice) {
                    src = req.src;
                    dst = req.dst;
                }

                printf("[Debug] dst address: %p\n", dst);
                printf("[Debug] src address: %p\n", src);

                cudaPointerAttributes dst_attr, src_attr;
                cudaPointerGetAttributes(&dst_attr, dst);
                cudaPointerGetAttributes(&src_attr, src);
                printf("[Debug] dst memory type: %s (device %d)\n",
                    (dst_attr.type == cudaMemoryTypeHost) ? "Host" : "Device",
                    dst_attr.device);
                printf("[Debug] src memory type: %s (device %d)\n",
                    (src_attr.type == cudaMemoryTypeHost) ? "Host" : "Device",
                    src_attr.device);

                const char* kind_str[] = {"HostToHost", "HostToDevice", "DeviceToHost", "DeviceToDevice"};
                printf("[Debug] cudaMemcpyKind: %s (%d)\n",
                    (req.kind <= 3) ? kind_str[req.kind] : "INVALID", req.kind);

                printf("[Debug] transfer count: %zu bytes\n", req.count);

                int current_device;
                cudaGetDevice(&current_device);
                printf("[Debug] current device: %d\n", current_device);

                response.result = cudaMemcpy(dst, src, req.count, req.kind);
                printf("[Debug] result: %s\n", cudaGetErrorString(response.result));

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy2DType: {
                cudaMemcpy2DRq& req = *reinterpret_cast<cudaMemcpy2DRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result =
                    cudaMemcpy2D(req.dst, req.dpitch, req.src, req.spitch, req.width, req.height, req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy2DArrayToArrayType: {
                cudaMemcpy2DArrayToArrayRq& req = *reinterpret_cast<cudaMemcpy2DArrayToArrayRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpy2DArrayToArray(
                    req.dst,
                    req.wOffsetDst,
                    req.hOffsetDst,
                    req.src,
                    req.wOffsetSrc,
                    req.hOffsetSrc,
                    req.width,
                    req.height,
                    req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy2DAsyncType: {
                cudaMemcpy2DAsyncRq& req = *reinterpret_cast<cudaMemcpy2DAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpy2DAsync(
                    req.dst, req.dpitch, req.src, req.spitch, req.width, req.height, req.kind, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy2DFromArrayType: {
                cudaMemcpy2DFromArrayRq& req = *reinterpret_cast<cudaMemcpy2DFromArrayRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpy2DFromArray(
                    req.dst, req.dpitch, req.src, req.wOffset, req.hOffset, req.width, req.height, req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy2DFromArrayAsyncType: {
                cudaMemcpy2DFromArrayAsyncRq& req = *reinterpret_cast<cudaMemcpy2DFromArrayAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpy2DFromArrayAsync(
                    req.dst,
                    req.dpitch,
                    req.src,
                    req.wOffset,
                    req.hOffset,
                    req.width,
                    req.height,
                    req.kind,
                    req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy2DToArrayType: {
                cudaMemcpy2DToArrayRq& req = *reinterpret_cast<cudaMemcpy2DToArrayRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpy2DToArray(
                    req.dst, req.wOffset, req.hOffset, req.src, req.spitch, req.width, req.height, req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy2DToArrayAsyncType: {
                cudaMemcpy2DToArrayAsyncRq& req = *reinterpret_cast<cudaMemcpy2DToArrayAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpy2DToArrayAsync(
                    req.dst,
                    req.wOffset,
                    req.hOffset,
                    req.src,
                    req.spitch,
                    req.width,
                    req.height,
                    req.kind,
                    req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy3DType: {
                cudaMemcpy3DRq& req = *reinterpret_cast<cudaMemcpy3DRq*>(request_ptr);

                cudaMemcpy3DRp& response = *((cudaMemcpy3DRp*)(NewResponse(response_rb)));
                response.result = cudaMemcpy3D(response.p);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpy3DAsyncType: {
                cudaMemcpy3DAsyncRq& req = *reinterpret_cast<cudaMemcpy3DAsyncRq*>(request_ptr);

                cudaMemcpy3DAsyncRp& response = *((cudaMemcpy3DAsyncRp*)(NewResponse(response_rb)));
                response.result = cudaMemcpy3DAsync(response.p, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyArrayToArrayType: {
                cudaMemcpyArrayToArrayRq& req = *reinterpret_cast<cudaMemcpyArrayToArrayRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpyArrayToArray(
                    req.dst,
                    req.wOffsetDst,
                    req.hOffsetDst,
                    req.src,
                    req.wOffsetSrc,
                    req.hOffsetSrc,
                    req.count,
                    req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyAsyncType: {
                cudaMemcpyAsyncRq& req = *reinterpret_cast<cudaMemcpyAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                cudaPointerAttributes dst_attributes;
                cudaPointerAttributes src_attributes;
                void *dst;
                const void *src;
                printf("server_shared_mem:%p\n", shared_mem);
                if (req.kind == cudaMemcpyHostToDevice) {
                    cudaError_t dst_status = cudaPointerGetAttributes(&dst_attributes, req.dst);
                    if (dst_status == cudaSuccess) {
                        if (dst_attributes.type == cudaMemoryTypeHost) {

                            dst = shared_mem + (size_t)((char *)req.dst - req.client_shared_mem);
                        } else if (dst_attributes.type == cudaMemoryTypeDevice) {
                            printf("dst pointer:%p", req.dst);

                            dst = req.dst;
                            src = shared_mem + (size_t)((char *)req.src - req.client_shared_mem);
                            printf("client_src:%p\n", req.src);
                            printf("server_src:%p\n", src);
                        } else if (dst_attributes.type == cudaMemoryTypeManaged) {

                        }
                    } else {

                        dst = shared_mem + (size_t)((char *)req.dst - req.client_shared_mem);

                    }
                }

                if (req.kind == cudaMemcpyDeviceToHost) {
                    cudaError_t src_status = cudaPointerGetAttributes(&src_attributes, req.src);
                    if (src_status == cudaSuccess) {
                        if (src_attributes.type == cudaMemoryTypeDevice) {

                            fflush(stdout);
                            src = req.src;

                            if (req.dst >= req.client_shared_mem && req.dst < req.client_shared_mem + 1024 * 1024 * 1024) {
                                dst = shared_mem + (size_t)((char *)req.dst - req.client_shared_mem);
                            }

                        }
                    } else {

                        src = shared_mem + (size_t)((char *)req.src - req.client_shared_mem);

                    }
                }

                if (req.kind == cudaMemcpyDeviceToDevice) {
                    src = req.src;
                    dst = req.dst;
                }

                printf("[Debug] dst address: %p\n", dst);
                printf("[Debug] src address: %p\n", src);

                cudaPointerAttributes dst_attr, src_attr;
                cudaPointerGetAttributes(&dst_attr, dst);
                cudaPointerGetAttributes(&src_attr, src);
                printf("[Debug] dst memory type: %s (device %d)\n",
                    (dst_attr.type == cudaMemoryTypeHost) ? "Host" : "Device",
                    dst_attr.device);
                printf("[Debug] src memory type: %s (device %d)\n",
                    (src_attr.type == cudaMemoryTypeHost) ? "Host" : "Device",
                    src_attr.device);

                const char* kind_str[] = {"HostToHost", "HostToDevice", "DeviceToHost", "DeviceToDevice"};
                printf("[Debug] cudaMemcpyKind: %s (%d)\n",
                    (req.kind <= 3) ? kind_str[req.kind] : "INVALID", req.kind);

                printf("[Debug] transfer count: %zu bytes\n", req.count);

                printf("[Debug] stream handle: %p\n", req.stream);
                if (req.stream) {
                    cudaError_t stream_state = cudaStreamQuery(req.stream);
                    printf("[Debug] stream state: %s\n",
                        (stream_state == cudaSuccess) ? "Valid" :
                        (stream_state == cudaErrorInvalidResourceHandle) ? "INVALID HANDLE" : "Busy");
                }

                int current_device;
                cudaGetDevice(&current_device);
                printf("[Debug] current device: %d\n", current_device);

                if (req.kind == cudaMemcpyHostToDevice) {
                    const float* host_src = static_cast<const float*>(src);
                    printf("[Debug] H2D - First 5 floats from host: ");
                    for (int i = 0; i < 5 && i < req.count/sizeof(float); ++i) {
                        printf("%f ", host_src[i]);
                    }
                    printf("\n");
                }

                response.result = cudaMemcpyAsync(dst, src, req.count, req.kind, req.stream);

                if (req.kind == cudaMemcpyDeviceToHost) {
                    const float* host_dst = static_cast<const float*>(dst);
                    printf("[Debug] D2H - First 5 floats to host: ");
                    for (int i = 0; i < 5 && i < req.count/sizeof(float); ++i) {
                        printf("%f ", host_dst[i]);
                    }
                    printf("\n");
                }

                printf("[Debug] result: %s\n", cudaGetErrorString(response.result));

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyFromArrayType: {
                cudaMemcpyFromArrayRq& req = *reinterpret_cast<cudaMemcpyFromArrayRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpyFromArray(req.dst, req.src, req.wOffset, req.hOffset, req.count, req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyFromArrayAsyncType: {
                cudaMemcpyFromArrayAsyncRq& req = *reinterpret_cast<cudaMemcpyFromArrayAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpyFromArrayAsync(
                    req.dst, req.src, req.wOffset, req.hOffset, req.count, req.kind, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyFromSymbolType: {
                cudaMemcpyFromSymbolRq& req = *reinterpret_cast<cudaMemcpyFromSymbolRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpyFromSymbol(req.dst, req.symbol, req.count, req.offset, req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyFromSymbolAsyncType: {
                cudaMemcpyFromSymbolAsyncRq& req = *reinterpret_cast<cudaMemcpyFromSymbolAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result =
                    cudaMemcpyFromSymbolAsync(req.dst, req.symbol, req.count, req.offset, req.kind, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyToArrayType: {
                cudaMemcpyToArrayRq& req = *reinterpret_cast<cudaMemcpyToArrayRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpyToArray(req.dst, req.wOffset, req.hOffset, req.src, req.count, req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyToArrayAsyncType: {
                cudaMemcpyToArrayAsyncRq& req = *reinterpret_cast<cudaMemcpyToArrayAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result =
                    cudaMemcpyToArrayAsync(req.dst, req.wOffset, req.hOffset, req.src, req.count, req.kind, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyToSymbolType: {
                cudaMemcpyToSymbolRq& req = *reinterpret_cast<cudaMemcpyToSymbolRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemcpyToSymbol(req.symbol, req.src, req.count, req.offset, req.kind);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemcpyToSymbolAsyncType: {
                cudaMemcpyToSymbolAsyncRq& req = *reinterpret_cast<cudaMemcpyToSymbolAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result =
                    cudaMemcpyToSymbolAsync(req.symbol, req.src, req.count, req.offset, req.kind, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemGetInfoType: {
                cudaMemGetInfoRq& req = *reinterpret_cast<cudaMemGetInfoRq*>(request_ptr);

                cudaMemGetInfoRp& response = *((cudaMemGetInfoRp*)(NewResponse(response_rb)));
                response.result = cudaMemGetInfo(&response.free, &response.total);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemsetType: {
                cudaMemsetRq& req = *reinterpret_cast<cudaMemsetRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemset(req.devPtr, req.value, req.count);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemset2DType: {
                cudaMemset2DRq& req = *reinterpret_cast<cudaMemset2DRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemset2D(req.devPtr, req.pitch, req.value, req.width, req.height);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemset2DAsyncType: {
                cudaMemset2DAsyncRq& req = *reinterpret_cast<cudaMemset2DAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result =
                    cudaMemset2DAsync(req.devPtr, req.pitch, req.value, req.width, req.height, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemset3DType: {
                cudaMemset3DRq& req = *reinterpret_cast<cudaMemset3DRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemset3D(req.pitchedDevPtr, req.value, req.extent);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemset3DAsyncType: {
                cudaMemset3DAsyncRq& req = *reinterpret_cast<cudaMemset3DAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemset3DAsync(req.pitchedDevPtr, req.value, req.extent, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cudaMemsetAsyncType: {
                cudaMemsetAsyncRq& req = *reinterpret_cast<cudaMemsetAsyncRq*>(request_ptr);
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));

                response.result = cudaMemsetAsync(req.devPtr, req.value, req.count, req.stream);
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            case cublasLtCreateType: {
                cublasLtCreateRq& req = *reinterpret_cast<cublasLtCreateRq*>(request_ptr);

                cublasLtCreateRp& response = *((cublasLtCreateRp*)(NewResponse(response_rb)));

                response.status = cublasLtCreate(&response.handle);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtDestroyType: {
                cublasLtDestroyRq& req = *reinterpret_cast<cublasLtDestroyRq*>(request_ptr);

                cublasLtDestroyRp& response = *((cublasLtDestroyRp*)(NewResponse(response_rb)));

                response.status = cublasLtDestroy(req.lightHandle);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtGetPropertyType: {
                cublasLtGetPropertyRq& req = *reinterpret_cast<cublasLtGetPropertyRq*>(request_ptr);

                cublasLtGetPropertyRp& response = *((cublasLtGetPropertyRp*)(NewResponse(response_rb)));

                response.status = cublasLtGetProperty(req.type1, &response.value);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtGetStatusNameType: {
                cublasLtGetStatusNameRq& req = *reinterpret_cast<cublasLtGetStatusNameRq*>(request_ptr);

                cublasLtGetStatusNameRp& response = *((cublasLtGetStatusNameRp*)(NewResponse(response_rb)));

                response.statusName = cublasLtGetStatusName(req.status);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtGetStatusStringType: {
                cublasLtGetStatusStringRq& req = *reinterpret_cast<cublasLtGetStatusStringRq*>(request_ptr);

                cublasLtGetStatusStringRp& response = *((cublasLtGetStatusStringRp*)(NewResponse(response_rb)));

                response.statusString = cublasLtGetStatusString(req.status);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtHeuristicsCacheGetCapacityType: {
                cublasLtHeuristicsCacheGetCapacityRq& req = *reinterpret_cast<cublasLtHeuristicsCacheGetCapacityRq*>(request_ptr);

                cublasLtHeuristicsCacheGetCapacityRp& response = *((cublasLtHeuristicsCacheGetCapacityRp*)(NewResponse(response_rb)));

                response.status = cublasLtHeuristicsCacheGetCapacity(&response.capacity);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtHeuristicsCacheSetCapacityType: {
                cublasLtHeuristicsCacheSetCapacityRq& req = *reinterpret_cast<cublasLtHeuristicsCacheSetCapacityRq*>(request_ptr);

                cublasLtHeuristicsCacheSetCapacityRp& response = *((cublasLtHeuristicsCacheSetCapacityRp*)(NewResponse(response_rb)));

                response.status = cublasLtHeuristicsCacheSetCapacity(req.capacity);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtGetVersionType: {
                cublasLtGetVersionRq& req = *reinterpret_cast<cublasLtGetVersionRq*>(request_ptr);

                cublasLtGetVersionRp& response = *((cublasLtGetVersionRp*)(NewResponse(response_rb)));

                response.version = cublasLtGetVersion();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            /* case cublasLtDisableCpuInstructionsSetMaskType: {
                cublasLtDisableCpuInstructionsSetMaskRq& req = *reinterpret_cast<cublasLtDisableCpuInstructionsSetMaskRq*>(request_ptr);
                req = *reinterpret_cast<Message_A*>(&request);
                cublasLtDisableCpuInstructionsSetMaskRp& response = *((cublasLtDisableCpuInstructionsSetMaskRp*)(NewResponse(response_rb)));

                response.status = cublasLtDisableCpuInstructionsSetMask(req.mask);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            } */
            case cublasLtGetCudartVersionType: {
                cublasLtGetCudartVersionRq& req = *reinterpret_cast<cublasLtGetCudartVersionRq*>(request_ptr);

                cublasLtGetCudartVersionRp& response = *((cublasLtGetCudartVersionRp*)(NewResponse(response_rb)));

                response.version = cublasLtGetCudartVersion();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtLoggerSetCallbackType: {
                cublasLtLoggerSetCallbackRq& req = *reinterpret_cast<cublasLtLoggerSetCallbackRq*>(request_ptr);

                cublasLtLoggerSetCallbackRp& response = *((cublasLtLoggerSetCallbackRp*)(NewResponse(response_rb)));

                response.status = cublasLtLoggerSetCallback(req.callback);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtLoggerSetFileType: {
                cublasLtLoggerSetFileRq& req = *reinterpret_cast<cublasLtLoggerSetFileRq*>(request_ptr);

                cublasLtLoggerSetFileRp& response = *((cublasLtLoggerSetFileRp*)(NewResponse(response_rb)));

                response.status = cublasLtLoggerSetFile(req.file);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtLoggerOpenFileType: {
                cublasLtLoggerOpenFileRq& req = *reinterpret_cast<cublasLtLoggerOpenFileRq*>(request_ptr);

                cublasLtLoggerOpenFileRp& response = *((cublasLtLoggerOpenFileRp*)(NewResponse(response_rb)));

                response.status = cublasLtLoggerOpenFile(req.logFile);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtLoggerSetLevelType: {
                cublasLtLoggerSetLevelRq& req = *reinterpret_cast<cublasLtLoggerSetLevelRq*>(request_ptr);

                cublasLtLoggerSetLevelRp& response = *((cublasLtLoggerSetLevelRp*)(NewResponse(response_rb)));

                response.status = cublasLtLoggerSetLevel(req.level);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtLoggerSetMaskType: {
                cublasLtLoggerSetMaskRq& req = *reinterpret_cast<cublasLtLoggerSetMaskRq*>(request_ptr);

                cublasLtLoggerSetMaskRp& response = *((cublasLtLoggerSetMaskRp*)(NewResponse(response_rb)));

                response.status = cublasLtLoggerSetMask(req.mask);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtLoggerForceDisableType: {
                cublasLtLoggerForceDisableRq& req = *reinterpret_cast<cublasLtLoggerForceDisableRq*>(request_ptr);

                cublasLtLoggerForceDisableRp& response = *((cublasLtLoggerForceDisableRp*)(NewResponse(response_rb)));

                response.status = cublasLtLoggerForceDisable();

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulType: {
                cublasLtMatmulRq& req = *reinterpret_cast<cublasLtMatmulRq*>(request_ptr);

                cublasLtMatmulRp& response = *((cublasLtMatmulRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmul(
                    req.lightHandle,
                    req.computeDesc,
                    &req.alpha,
                    req.A,
                    req.Adesc,
                    req.B,
                    req.Bdesc,
                    &req.beta,
                    req.C,
                    req.Cdesc,
                    req.D,
                    req.Ddesc,
                    NULL,
                    req.workspace,
                    req.workspaceSizeInBytes,
                    req.stream
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                printf("cublasLtMatmul end\n");
                break;
            }
            case cublasLtMatmulAlgoCapGetAttributeType: {
                cublasLtMatmulAlgoCapGetAttributeRq& req = *reinterpret_cast<cublasLtMatmulAlgoCapGetAttributeRq*>(request_ptr);

                cublasLtMatmulAlgoCapGetAttributeRp& response = *((cublasLtMatmulAlgoCapGetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulAlgoCapGetAttribute(
                    req.algo,
                    req.attr,
                    req.buf,
                    req.sizeInBytes,
                    &response.sizeWritten
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            /* case cublasLtMatmulAlgoCheckType: {
                cublasLtMatmulAlgoCheckRq& req = *reinterpret_cast<cublasLtMatmulAlgoCheckRq*>(request_ptr);
                req = *reinterpret_cast<Message_A*>(&request);
                cublasLtMatmulAlgoCheckRp& response = *((cublasLtMatmulAlgoCheckRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulAlgoCheck(
                    req.lightHandle,
                    req.operationDesc,
                    req.Adesc,
                    req.Bdesc,
                    req.Cdesc,
                    req.Ddesc,
                    req.algo,
                    &response.result
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }  */
            case cublasLtMatmulAlgoConfigGetAttributeType: {
                cublasLtMatmulAlgoConfigGetAttributeRq& req = *reinterpret_cast<cublasLtMatmulAlgoConfigGetAttributeRq*>(request_ptr);

                cublasLtMatmulAlgoConfigGetAttributeRp& response = *((cublasLtMatmulAlgoConfigGetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulAlgoConfigGetAttribute(
                    req.algo,
                    req.attr,
                    req.buf,
                    req.sizeInBytes,
                    &response.sizeWritten
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulAlgoConfigSetAttributeType: {
                cublasLtMatmulAlgoConfigSetAttributeRq& req = *reinterpret_cast<cublasLtMatmulAlgoConfigSetAttributeRq*>(request_ptr);

                cublasLtMatmulAlgoConfigSetAttributeRp& response = *((cublasLtMatmulAlgoConfigSetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulAlgoConfigSetAttribute(
                    req.algo,
                    req.attr,
                    req.buf,
                    req.sizeInBytes
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            /* case cublasLtMatmulAlgoGetHeuristicType: {
                cublasLtMatmulAlgoGetHeuristicRq& req = *reinterpret_cast<cublasLtMatmulAlgoGetHeuristicRq*>(request_ptr);
                req = *reinterpret_cast<Message_A*>(&request);
                cublasLtMatmulAlgoGetHeuristicRp& response = *((cublasLtMatmulAlgoGetHeuristicRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulAlgoGetHeuristic(
                    req.lightHandle,
                    req.operationDesc,
                    req.Adesc,
                    req.Bdesc,
                    req.Cdesc,
                    req.Ddesc,
                    req.preference,
                    req.requestedAlgoCount,
                    response.heuristicResultsArray,
                    &response.returnAlgoCount
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            } */
            case cublasLtMatmulAlgoGetIdsType: {
                cublasLtMatmulAlgoGetIdsRq& req = *reinterpret_cast<cublasLtMatmulAlgoGetIdsRq*>(request_ptr);

                cublasLtMatmulAlgoGetIdsRp& response = *((cublasLtMatmulAlgoGetIdsRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulAlgoGetIds(
                    req.lightHandle,
                    req.computeType,
                    req.scaleType,
                    req.Atype,
                    req.Btype,
                    req.Ctype,
                    req.Dtype,
                    req.requestedAlgoCount,
                    response.algoIdsArray,
                    &response.returnAlgoCount
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulAlgoInitType: {
                cublasLtMatmulAlgoInitRq& req = *reinterpret_cast<cublasLtMatmulAlgoInitRq*>(request_ptr);

                cublasLtMatmulAlgoInitRp& response = *((cublasLtMatmulAlgoInitRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulAlgoInit(
                    req.lightHandle,
                    req.computeType,
                    req.scaleType,
                    req.Atype,
                    req.Btype,
                    req.Ctype,
                    req.Dtype,
                    req.algoId,
                    &response.algo
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulDescCreateType: {

                cublasLtMatmulDescCreateRq& req = *reinterpret_cast<cublasLtMatmulDescCreateRq*>(request_ptr);

                cublasLtMatmulDescCreateRp& response = *((cublasLtMatmulDescCreateRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulDescCreate(
                    &response.matmulDesc,
                    req.computeType, req.scaleType
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulDescInitType: {
                cublasLtMatmulDescInitRq& req = *reinterpret_cast<cublasLtMatmulDescInitRq*>(request_ptr);

                cublasLtMatmulDescInitRp& response = *((cublasLtMatmulDescInitRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulDescInit(
                    req.matmulDesc,
                    req.computeType,
                    req.scaleType
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulDescDestroyType: {
                cublasLtMatmulDescDestroyRq& req = *reinterpret_cast<cublasLtMatmulDescDestroyRq*>(request_ptr);

                cublasLtMatmulDescDestroyRp& response = *((cublasLtMatmulDescDestroyRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulDescDestroy(req.matmulDesc);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulDescGetAttributeType: {
                cublasLtMatmulDescGetAttributeRq& req = *reinterpret_cast<cublasLtMatmulDescGetAttributeRq*>(request_ptr);

                cublasLtMatmulDescGetAttributeRp& response = *((cublasLtMatmulDescGetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulDescGetAttribute(
                    req.matmulDesc,
                    req.attr,
                    response.buf,
                    req.sizeInBytes,
                    &response.sizeWritten
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulDescSetAttributeType: {
                cublasLtMatmulDescSetAttributeRq& req = *reinterpret_cast<cublasLtMatmulDescSetAttributeRq*>(request_ptr);

                cublasLtMatmulDescSetAttributeRp& response = *((cublasLtMatmulDescSetAttributeRp*)(NewResponse(response_rb)));
                const void * buf = (void*)(shared_mem + (size_t)((char*)req.buf - req.client_shared_mem));
                printf("shmem pointer on server: %p", buf);

                response.status = cublasLtMatmulDescSetAttribute(
                    req.matmulDesc,
                    req.attr,
                    buf,
                    req.sizeInBytes
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulPreferenceCreateType: {
                cublasLtMatmulPreferenceCreateRq& req = *reinterpret_cast<cublasLtMatmulPreferenceCreateRq*>(request_ptr);

                cublasLtMatmulPreferenceCreateRp& response = *((cublasLtMatmulPreferenceCreateRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulPreferenceCreate(&response.pref);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulPreferenceInitType: {
                cublasLtMatmulPreferenceInitRq& req = *reinterpret_cast<cublasLtMatmulPreferenceInitRq*>(request_ptr);

                cublasLtMatmulPreferenceInitRp& response = *((cublasLtMatmulPreferenceInitRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulPreferenceInit(req.pref);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulPreferenceDestroyType: {
                cublasLtMatmulPreferenceDestroyRq& req = *reinterpret_cast<cublasLtMatmulPreferenceDestroyRq*>(request_ptr);

                cublasLtMatmulPreferenceDestroyRp& response = *((cublasLtMatmulPreferenceDestroyRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulPreferenceDestroy(req.pref);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulPreferenceGetAttributeType: {
                cublasLtMatmulPreferenceGetAttributeRq& req = *reinterpret_cast<cublasLtMatmulPreferenceGetAttributeRq*>(request_ptr);

                cublasLtMatmulPreferenceGetAttributeRp& response = *((cublasLtMatmulPreferenceGetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatmulPreferenceGetAttribute(
                    req.pref,
                    req.attr,
                    response.buf,
                    req.sizeInBytes,
                    &response.sizeWritten
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatmulPreferenceSetAttributeType: {
                cublasLtMatmulPreferenceSetAttributeRq& req = *reinterpret_cast<cublasLtMatmulPreferenceSetAttributeRq*>(request_ptr);

                cublasLtMatmulPreferenceSetAttributeRp& response = *((cublasLtMatmulPreferenceSetAttributeRp*)(NewResponse(response_rb)));
                const void * buf = (void*)(shared_mem + (size_t)((char*)req.buf - req.client_shared_mem));

                response.status = cublasLtMatmulPreferenceSetAttribute(
                    req.pref,
                    req.attr,
                    buf,
                    req.sizeInBytes
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixLayoutCreateType: {
                cublasLtMatrixLayoutCreateRq& req = *reinterpret_cast<cublasLtMatrixLayoutCreateRq*>(request_ptr);

                cublasLtMatrixLayoutCreateRp& response = *((cublasLtMatrixLayoutCreateRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixLayoutCreate(
                    &response.matLayout,
                    req.type1,
                    req.rows,
                    req.cols,
                    req.ld
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixLayoutInitType: {
                cublasLtMatrixLayoutInitRq& req = *reinterpret_cast<cublasLtMatrixLayoutInitRq*>(request_ptr);

                cublasLtMatrixLayoutInitRp& response = *((cublasLtMatrixLayoutInitRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixLayoutInit(
                    req.matLayout,
                    req.type,
                    req.rows,
                    req.cols,
                    req.ld
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixLayoutDestroyType: {
                cublasLtMatrixLayoutDestroyRq& req = *reinterpret_cast<cublasLtMatrixLayoutDestroyRq*>(request_ptr);

                cublasLtMatrixLayoutDestroyRp& response = *((cublasLtMatrixLayoutDestroyRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixLayoutDestroy(req.matLayout);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixLayoutGetAttributeType: {
                cublasLtMatrixLayoutGetAttributeRq& req = *reinterpret_cast<cublasLtMatrixLayoutGetAttributeRq*>(request_ptr);

                cublasLtMatrixLayoutGetAttributeRp& response = *((cublasLtMatrixLayoutGetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixLayoutGetAttribute(
                    req.matLayout,
                    req.attr,
                    response.buf,
                    req.sizeInBytes,
                    &response.sizeWritten
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixLayoutSetAttributeType: {
                cublasLtMatrixLayoutSetAttributeRq& req = *reinterpret_cast<cublasLtMatrixLayoutSetAttributeRq*>(request_ptr);

                cublasLtMatrixLayoutSetAttributeRp& response = *((cublasLtMatrixLayoutSetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixLayoutSetAttribute(
                    req.matLayout,
                    req.attr,
                    req.buf,
                    req.sizeInBytes
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            /* case cublasLtMatrixTransformType: {
                cublasLtMatrixTransformRq& req = *reinterpret_cast<cublasLtMatrixTransformRq*>(request_ptr);
                req = *reinterpret_cast<Message_A*>(&request);
                cublasLtMatrixTransformRp& response = *((cublasLtMatrixTransformRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixTransform(
                    req.lightHandle,
                    req.transformDesc,
                    req.alpha,
                    req.A,
                    req.Adesc,
                    req.beta,
                    req.B,
                    req.Bdesc,
                    req.C,
                    req.Cdesc,
                    req.stream
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            } */
            case cublasLtMatrixTransformDescCreateType: {
                cublasLtMatrixTransformDescCreateRq& req = *reinterpret_cast<cublasLtMatrixTransformDescCreateRq*>(request_ptr);

                cublasLtMatrixTransformDescCreateRp& response = *((cublasLtMatrixTransformDescCreateRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixTransformDescCreate(
                    req.transformDesc,
                    req.scaleType
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixTransformDescInitType: {
                cublasLtMatrixTransformDescInitRq& req = *reinterpret_cast<cublasLtMatrixTransformDescInitRq*>(request_ptr);

                cublasLtMatrixTransformDescInitRp& response = *((cublasLtMatrixTransformDescInitRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixTransformDescInit(
                    req.transformDesc,
                    req.scaleType
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixTransformDescDestroyType: {
                cublasLtMatrixTransformDescDestroyRq& req = *reinterpret_cast<cublasLtMatrixTransformDescDestroyRq*>(request_ptr);

                cublasLtMatrixTransformDescDestroyRp& response = *((cublasLtMatrixTransformDescDestroyRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixTransformDescDestroy(req.transformDesc);

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixTransformDescGetAttributeType: {
                cublasLtMatrixTransformDescGetAttributeRq& req = *reinterpret_cast<cublasLtMatrixTransformDescGetAttributeRq*>(request_ptr);

                cublasLtMatrixTransformDescGetAttributeRp& response = *((cublasLtMatrixTransformDescGetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixTransformDescGetAttribute(
                    req.transformDesc,
                    req.attr,
                    response.buf,
                    req.sizeInBytes,
                    &response.sizeWritten
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            case cublasLtMatrixTransformDescSetAttributeType: {
                cublasLtMatrixTransformDescSetAttributeRq& req = *reinterpret_cast<cublasLtMatrixTransformDescSetAttributeRq*>(request_ptr);

                cublasLtMatrixTransformDescSetAttributeRp& response = *((cublasLtMatrixTransformDescSetAttributeRp*)(NewResponse(response_rb)));

                response.status = cublasLtMatrixTransformDescSetAttribute(
                    req.transformDesc,
                    req.attr,
                    req.buf,
                    req.sizeInBytes
                );

                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }
            default:
                std::cerr << "Unknown request type" << std::endl;
                CudaResponseBase& response = *((CudaResponseBase*)(NewResponse(response_rb)));
                response.result = cudaErrorUnknown;
                response.responseSize = sizeof(response); send_result = ResponseSend(response_rb, sizeof(response));
                break;
            }

            if (send_result == false) {
                perror("send");

                break;
            }
        }
    }
    return 0;
}
