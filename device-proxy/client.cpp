#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <map>
#include <cstring>
#include <thread>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include <array>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstdlib>
#include <spawn.h>
#include <cstdint>
#include <type_traits>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fstream>
#include <sys/types.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nccl.h>
#include <atomic>

#define printf(...) while (0)

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
std::string sharedMemoryPath;
long SHARED_MEM_SIZE = 1024 * 1024 * 1024;

static int shm_fd = -1;
static int alloc = 0;
static int load = 0;

static char* initSharedMemory() {
    pid_t pid = getpid();
    sharedMemoryPath = "/cuda_shared_memory_"+ std::to_string(pid);
    shm_fd = shm_open(sharedMemoryPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open failed");
        exit(1);
    }

    if (ftruncate(shm_fd, (long)1024 * 1024 * 1024) == -1) {
        perror("ftruncate failed");
        exit(1);
    }

    static char* shared_mem =
        (char*)mmap(nullptr, (long)1024 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_mem == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }
    memset(shared_mem, 0, (long)1024 * 1024 * 1024);
    return shared_mem;
}
static size_t shared_mem_offset = 0;
static std::mutex alloc_mutex;
static char* global_shared_mem;

struct FunctionInfo {
    std::string cubin_file;
    int param_count;
    std::vector<int> param_lengths;
};

static std::unordered_map<std::string, FunctionInfo> function_map;

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
    enum cudaFuncCache* pCacheConfig;
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
    size_t shared_mem_offset;
};
struct cudaGetErrorStringRp : public CudaResponseBase {
    const char* error;
    size_t size;
};
struct cudaGetLastErrorRq : public CudaRequestBase {};

struct cudaPeekAtLastErrorRq : public CudaRequestBase {};

struct cudaChooseDeviceRq : public CudaRequestBase {
    const struct cudaDeviceProp* prop;
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
    int* device_arr;
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
    const void* func;
};
struct cudaFuncGetAttributesRp : public CudaResponseBase {
    struct cudaFuncAttributes attr;
};
struct cudaFuncSetCacheConfigRq : public CudaRequestBase {
    const void* func;
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

    void* devPtr;
};
struct cudaExternalMemoryGetMappedMipmappedArrayRq : public CudaRequestBase {
    cudaExternalMemory_t extMem;
    cudaExternalMemoryMipmappedArrayDesc mipmapDesc;
};
struct cudaExternalMemoryGetMappedMipmappedArrayRp : public CudaResponseBase {

    cudaMipmappedArray_t mipmap;
};
struct cudaImportExternalMemoryRq : public CudaRequestBase {
    cudaExternalMemory_t* extMem_out;
    cudaExternalMemoryHandleDesc memHandleDesc;

};
struct cudaImportExternalMemoryRp : public CudaResponseBase {

};
struct cudaImportExternalSemaphoreRq : public CudaRequestBase {
    cudaExternalSemaphore_t* extSem_out;
    cudaExternalSemaphoreHandleDesc semHandleDesc;
};
struct cudaImportExternalSemaphoreRp : public CudaResponseBase {
    cudaError_t result;
};
struct cudaSignalExternalSemaphoresAsyncRq : public CudaRequestBase {
    const cudaExternalSemaphore_t* extSemArray;
    const cudaExternalSemaphoreSignalParams* paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
};
struct cudaSignalExternalSemaphoresAsyncRp : public CudaResponseBase {

};
struct cudaWaitExternalSemaphoresAsyncRq : public CudaRequestBase {
    const cudaExternalSemaphore_t* extSemArray;
    const cudaExternalSemaphoreWaitParams* paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
};
struct cudaWaitExternalSemaphoresAsyncRp : public CudaResponseBase {

};

struct cudaCtxResetPersistingL2CacheRq : public CudaRequestBase {};
struct cudaStreamAddCallbackRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaStreamCallback_t callback;
    void* userData;
    unsigned int flags;
};
struct cudaStreamAttachMemAsyncRq : public CudaRequestBase {
    cudaStream_t stream;
    void* devPtr;
    size_t length = 0;
    unsigned int flags = cudaMemAttachSingle;
};
struct cudaStreamBeginCaptureRq : public CudaRequestBase {
    cudaStream_t stream;
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
    const cudaGraphNode_t* dependencies;
    size_t numDependencies;
};
struct cudaStreamGetCaptureInfo_v3Rq : public CudaRequestBase {
    cudaStream_t stream;
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
    cudaGraphNode_t* dependencies;
    size_t numDependencies;
    unsigned int flags;
};
struct cudaStreamUpdateCaptureDependenciesRp : public CudaResponseBase {

};

struct cudaStreamUpdateCaptureDependencies_v2Rp : public CudaResponseBase {

};
struct cudaThreadExchangeStreamCaptureModeRq : public CudaRequestBase {

};
struct cudaThreadExchangeStreamCaptureModeRp : public CudaResponseBase {

    cudaStreamCaptureMode mode;
};

struct cudaFuncGetNameRq : public CudaRequestBase {
    const void* func;
};
struct cudaFuncGetNameRp : public CudaResponseBase {

    const char** name;
};
struct cudaFuncGetParamInfoRq : public CudaRequestBase {
    const void* func;
    size_t paramIndex;
};
struct cudaFuncGetParamInfoRp : public CudaResponseBase {

    size_t paramOffset;
    size_t paramSize;
};
struct cudaFuncSetAttributeRq : public CudaRequestBase {
    const void* func;
    cudaFuncAttribute attr;
    int value;
};
struct cudaFuncSetAttributeRp : public CudaResponseBase {

};
struct cudaLaunchCooperativeKernelRq : public CudaRequestBase {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    void** args;
    size_t sharedMem;
    cudaStream_t stream;
};
struct cudaLaunchCooperativeKernelRp : public CudaResponseBase {

};
struct cudaLaunchCooperativeKernelMultiDeviceRq : public CudaRequestBase {
    cudaLaunchParams* launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
};
struct cudaLaunchCooperativeKernelMultiDeviceRp : public CudaResponseBase {

};
struct cudaLaunchHostFuncRq : public CudaRequestBase {
    cudaStream_t stream;
    cudaHostFn_t fn;
    void* userData;
};
struct cudaLaunchHostFuncRp : public CudaResponseBase {

};
struct cudaLaunchKernelRq : public CudaRequestBase {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    void** args;
    size_t sharedMem;
    cudaStream_t stream;
    int argsSize;

    char* client_shared_mem;
    char* kernel_name_pointer;
    char* cubin_file_path_pointer;

};
struct cudaLaunchKernelRp : public CudaResponseBase {

};
struct cudaLaunchKernelExCRq : public CudaRequestBase {
    cudaLaunchConfig_t config;
    const void* func;
    void** args;
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

    void* nvSciSyncAttrList;
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
    void* devPtr;
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
    void* devPtr;
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
    void* devPtr;
};

struct cudaFreeRq : public CudaRequestBase {
    void* devPtr;
};
struct cudaFreeArrayRq : public CudaRequestBase {
    struct cudaArray* array;
};
struct cudaFreeHostRq : public CudaRequestBase {
    void* ptr;
};
struct cudaGetSymbolAddressRq : public CudaRequestBase {
    const void* symbol;
};
struct cudaGetSymbolAddressRp : public CudaResponseBase {
    void* devPtr;
};
struct cudaGetSymbolSizeRq : public CudaRequestBase {
    const void* symbol;
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
    void* pHost;
};
struct cudaHostGetDevicePointerRq : public CudaRequestBase {
    void* pHost;
    unsigned int flags;
};
struct cudaHostGetDevicePointerRp : public CudaResponseBase {
    void* pDevice;
};
struct cudaHostGetFlagsRq : public CudaRequestBase {

    void* pHost;
};
struct cudaHostGetFlagsRp : public CudaResponseBase {
    unsigned int pFlags;
};
struct cudaMallocRq : public CudaRequestBase {
    size_t size;
};
struct cudaMallocRp : public CudaResponseBase {
    void* devPtr;
};
struct cudaMalloc3DRq : public CudaRequestBase {
    struct cudaExtent extent;
};
struct cudaMalloc3DRp : public CudaResponseBase {
    struct cudaPitchedPtr* pitchedDevPtr;
};
struct cudaMalloc3DArrayRq : public CudaRequestBase {
    const struct cudaChannelFormatDesc* desc;
    struct cudaExtent extent;
    unsigned int flags;
};
struct cudaMalloc3DArrayRp : public CudaResponseBase {
    struct cudaArray* array;
};
struct cudaMallocArrayRq : public CudaRequestBase {
    const struct cudaChannelFormatDesc* desc;
    size_t width;
    size_t height;
    unsigned int flags;
};
struct cudaMallocArrayRp : public CudaResponseBase {
    struct cudaArray* array;
};
struct cudaMallocHostRq : public CudaRequestBase {
    size_t size;
};
struct cudaMallocHostRp : public CudaResponseBase {
    void* ptr;
};
struct cudaMallocPitchRq : public CudaRequestBase {
    size_t width;
    size_t height;
};
struct cudaMallocPitchRp : public CudaResponseBase {
    size_t pitch;
    void* devPtr;
};
struct cudaMemcpyRq : public CudaRequestBase {
    void* dst;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DRq : public CudaRequestBase {
    void* dst;
    size_t dpitch;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DArrayToArrayRq : public CudaRequestBase {
    struct cudaArray* dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    const struct cudaArray* src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DAsyncRq : public CudaRequestBase {
    void* dst;
    size_t dpitch;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpy2DFromArrayRq : public CudaRequestBase {
    void* dst;
    size_t dpitch;
    const struct cudaArray* src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DFromArrayAsyncRq : public CudaRequestBase {
    void* dst;
    size_t dpitch;
    const struct cudaArray* src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpy2DToArrayRq : public CudaRequestBase {
    struct cudaArray* dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpy2DToArrayAsyncRq : public CudaRequestBase {
    struct cudaArray* dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpy3DRq : public CudaRequestBase {};
struct cudaMemcpy3DRp : public CudaResponseBase {
    const struct cudaMemcpy3DParms* p;
};
struct cudaMemcpy3DAsyncRq : public CudaRequestBase {
    cudaStream_t stream;
};
struct cudaMemcpy3DAsyncRp : public CudaResponseBase {
    const struct cudaMemcpy3DParms* p;
};
struct cudaMemcpyArrayToArrayRq : public CudaRequestBase {
    struct cudaArray* dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    const struct cudaArray* src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyAsyncRq : public CudaRequestBase {
    void* dst;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    char* client_shared_mem;
    size_t shmem_offset;
};
struct cudaMemcpyFromArrayRq : public CudaRequestBase {
    void* dst;
    const struct cudaArray* src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyFromArrayAsyncRq : public CudaRequestBase {
    void* dst;
    const struct cudaArray* src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpyFromSymbolRq : public CudaRequestBase {
    void* dst;
    const void* symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyFromSymbolAsyncRq : public CudaRequestBase {
    void* dst;
    const void* symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpyToArrayRq : public CudaRequestBase {
    struct cudaArray* dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyToArrayAsyncRq : public CudaRequestBase {
    struct cudaArray* dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
};
struct cudaMemcpyToSymbolRq : public CudaRequestBase {
    const void* symbol;
    const void* src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
};
struct cudaMemcpyToSymbolAsyncRq : public CudaRequestBase {
    const void* symbol;
    const void* src;
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
    void* devPtr;
    int value;
    size_t count;
};
struct cudaMemset2DRq : public CudaRequestBase {
    void* devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
};
struct cudaMemset2DAsyncRq : public CudaRequestBase {
    void* devPtr;
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
    void* devPtr;
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
    struct cudaExtent result;
};
struct make_cudaPitchedPtrRq : public CudaRequestBase {
    void* d;
    size_t p;
    size_t xsz;
    size_t ysz;
};
struct make_cudaPitchedPtrRp {
    struct cudaPitchedPtr result;
};
struct make_cudaPosRq : public CudaRequestBase {
    size_t x;
    size_t y;
    size_t z;
};
struct make_cudaPosRp {
    struct cudaPos result;
};

struct cudaOccupancyAvailableDynamicSMemPerBlockRq : public CudaRequestBase {
    const void* func;
    int numBlocks;
    int blockSize;
};
struct cudaOccupancyAvailableDynamicSMemPerBlockRp : public CudaResponseBase {
    size_t dynamicSmemSize;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorRq : public CudaRequestBase {
    const void* func;
    int blockSize;
    size_t dynamicSMemSize;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorRp : public CudaResponseBase {
    int numBlocks;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRq : public CudaRequestBase {
    const void* func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
};
struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRp : public CudaResponseBase {
    int numBlocks;
};
struct cudaOccupancyMaxActiveClustersRq : public CudaRequestBase {
    const void* func;
    cudaLaunchConfig_t launchConfig;
};
struct cudaOccupancyMaxActiveClustersRp : public CudaResponseBase {
    int numClusters;
};
struct cudaOccupancyMaxPotentialClusterSizeRq : public CudaRequestBase {
    const void* func;
    cudaLaunchConfig_t launchConfig;
};
struct cudaOccupancyMaxPotentialClusterSizeRp : public CudaResponseBase {
    cudaError_t result;
    int clusterSize;
};

struct cublasSgemmRq : public CudaRequestBase {
    cublasHandle_t handle; cublasOperation_t transa; cublasOperation_t transb; int m; int n; int k; float alpha;
        const float *A; int lda; const float *B; int ldb; float beta; float *C;
        int ldc;
        char* client_shared_mem;
};
struct cublasSgemmRp : public CudaResponseBase {
};

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

struct cublasCreateRq : public CudaRequestBase {
};
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

int client_fd = -1;

bool useServer = false;

constexpr size_t SHM_SIZE = 1024 * 1024 * 1024;
constexpr size_t BUFFER_SIZE = SHM_SIZE - sizeof(std::atomic<size_t>) * 2;
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
        size_t res = (w + BUFFER_SIZE - r) % BUFFER_SIZE;
        return res;
    }
};
std::string requestMemoryPath;
std::string responseMemoryPath;

RingBuffer* get_request_shared_memory(bool create) {
    int shm_fd = shm_open(requestMemoryPath.c_str(), O_RDWR | (create ? O_CREAT : 0), 0666);
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
bool waitForSharedMemory() {
    const int maxRetries = 10;
    const int sleepTime = 1;

    for (int i = 0; i < maxRetries; ++i) {
        int shm_fd = shm_open(responseMemoryPath.c_str(), O_RDWR, 0666);
        if (shm_fd != -1) {
            close(shm_fd);
            return true;
        }
        std::cerr << "Waiting for response shared memory to be created..." << std::endl;
        sleep(sleepTime);
    }
    return false;
}
RingBuffer * request_rb;
RingBuffer * response_rb;
bool connectToServer() {

    pid_t pid = getpid();
    requestMemoryPath = "/request_ring_buffer_" + std::to_string(pid);
    responseMemoryPath = "/response_ring_buffer_" + std::to_string(pid);
    request_rb = get_request_shared_memory(true);
    if (!request_rb) return false;

    pid_t server_pid = fork();

    sharedMemoryPath = "/cuda_shared_memory_"+ std::to_string(pid);
    if (server_pid == 0) {
        char* new_env[] = {NULL};
        char* args[] = {"./server_proxy", const_cast<char*>(requestMemoryPath.c_str()),const_cast<char*>(responseMemoryPath.c_str()),const_cast<char*>(sharedMemoryPath.c_str()),NULL};
        execve(args[0], args, new_env);
        perror("execve failed");
        exit(1);
    } else if (server_pid < 0) {
        perror("Fork failed");
        return false;
    }
    if (waitForSharedMemory()) {
        response_rb = get_response_shared_memory(false);

    } else {
        std::cerr << "Failed to find response shared memory after retries." << std::endl;
        return false;
    }
    return true;
}

void closeConnection() {
    if (client_fd >= 0) {
        close(client_fd);
        client_fd = -1;
    }
}

__attribute__((destructor)) void cleanupConnection() {
    if (useServer) {
        closeConnection();
    }
}

void* NewRequest(RingBuffer* rb) {
    return (void*)(&rb->buffer[rb->write_ptr.load(std::memory_order_relaxed)]);
}

bool RequestSend(RingBuffer* rb, size_t size) {
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

void* ResponseReceive(RingBuffer* rb) {
    while (true) {
        if (rb->available_read_space() >= sizeof(size_t)) {
            break;
        }
    }
    size_t r = rb->read_ptr.load(std::memory_order_relaxed);
    size_t *msg_size = (size_t*)(&rb->buffer[r]);

    if (rb->available_read_space() < *msg_size) {
        return NULL;
    }

    void* res = (void*)(&rb->buffer[r]);
    r = (r + *msg_size) % BUFFER_SIZE;

    rb->read_ptr.store(r, std::memory_order_release);
    return res;
}

const char* ncclGetLastError(ncclComm_t comm) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return nullptr;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return nullptr;
    }

    ncclGetLastErrorRq& request = *(ncclGetLastErrorRq*)NewRequest(request_rb);
    request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclGetLastErrorType;
    request.shared_mem_offset = shared_mem_offset;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return nullptr;
    }

    ncclGetLastErrorRp& response = *((ncclGetLastErrorRp*)(ResponseReceive(response_rb)));

    const char * errorMessage = (const char *)(global_shared_mem + shared_mem_offset);
    shared_mem_offset += response.size;

    return errorMessage;
}

ncclResult_t ncclGetVersion(int* version) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclGetVersionRq& request = *(ncclGetVersionRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = ncclGetVersionType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclGetVersionRp& response = *((ncclGetVersionRp*)(ResponseReceive(response_rb)));

    if (version) {
        *version = response.version;
    }

    return response.ncclResult;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    if (alloc == 0) {
        std::lock_guard<std::mutex> lock(alloc_mutex);
        global_shared_mem = initSharedMemory();
        alloc = 1;
    }

    ncclGetUniqueIdRq& request = *(ncclGetUniqueIdRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = ncclGetUniqueIdType;

    request.shared_mem_offset = shared_mem_offset;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclGetUniqueIdRp& response = *((ncclGetUniqueIdRp*)(ResponseReceive(response_rb)));

    ncclUniqueId* ptr_uniqueId = (ncclUniqueId*)(global_shared_mem + shared_mem_offset);
    memcpy(uniqueId, ptr_uniqueId, sizeof(ncclUniqueId));
    shared_mem_offset += sizeof(ncclUniqueId);

    return response.ncclResult;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommInitRankRq& request = *(ncclCommInitRankRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.nranks = nranks;
    request.commId = commId;
    request.rank = rank;
    request.type = ncclCommInitRankType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommInitRankRp& response = *((ncclCommInitRankRp*)(ResponseReceive(response_rb)));

    if (comm) {
        *comm = response.comm;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    int i;
    for (i = 0;i < ndev;i++) {

    }
    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommInitAllRq& request = *(ncclCommInitAllRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.ndev = ndev;
    request.type = ncclCommInitAllType;
    request.client_shared_mem = global_shared_mem;

    void* ptr_devlist;
    if (devlist != nullptr) {
        ptr_devlist = global_shared_mem + shared_mem_offset;
        memcpy(ptr_devlist, devlist, ndev * sizeof(int));

        shared_mem_offset += ndev * sizeof(int);
    }
    else {
        ptr_devlist = nullptr;
    }

    request.devlist = ptr_devlist;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommInitAllRp& response = *((ncclCommInitAllRp*)(ResponseReceive(response_rb)));

    ncclComm_t* ptr_comms = (ncclComm_t*)((const int *)ptr_devlist + ndev * sizeof(int));

    memcpy(comms, ptr_comms, ndev * sizeof(ncclComm_t));
    for (i = 0;i < ndev;i++) {

    }
    for (i = 0;i < ndev;i++) {

    }
    shared_mem_offset += ndev * sizeof(ncclComm_t);

    return response.ncclResult;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommInitRankConfigRq& request = *(ncclCommInitRankConfigRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.nranks = nranks;

    request.rank = rank;
    request.type = ncclCommInitRankConfigType;
    void * ptr_config = global_shared_mem + shared_mem_offset;
    memcpy(ptr_config, config, sizeof(ncclConfig_t));
    shared_mem_offset += sizeof(ncclConfig_t);
    void * ptr_commId = global_shared_mem + shared_mem_offset;
    memcpy(ptr_commId, &commId, sizeof(ncclUniqueId));
    shared_mem_offset += sizeof(ncclUniqueId);

    request.config = ptr_config;
    request.commId = ptr_commId;
    request.client_shared_mem = global_shared_mem;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommInitRankConfigRp& response = *((ncclCommInitRankConfigRp*)(ResponseReceive(response_rb)));

    if (comm) {
        *comm = response.comm;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommInitRankScalableRq& request = *(ncclCommInitRankScalableRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.nranks = nranks;
    request.myrank = myrank;
    request.nId = nId;
    request.type = ncclCommInitRankScalableType;

    size_t commIdsSize = nranks * sizeof(ncclUniqueId);
    void* ptr_commIds = global_shared_mem + shared_mem_offset;
    memcpy(ptr_commIds, commIds, commIdsSize);
    shared_mem_offset += commIdsSize;

    request.commIds = ptr_commIds;
    request.client_shared_mem = global_shared_mem;

    void* ptr_config = global_shared_mem + shared_mem_offset;
    memcpy(ptr_config, config, sizeof(ncclConfig_t));
    shared_mem_offset += sizeof(ncclConfig_t);
    request.config = ptr_config;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommInitRankScalableRp& response = *((ncclCommInitRankScalableRp*)(ResponseReceive(response_rb)));

    if (newcomm) {
        *newcomm = response.newcomm;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommSplitRq& request = *(ncclCommSplitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.color = color;
    request.key = key;
    request.client_shared_mem = global_shared_mem;

    if (config) {
        void* ptr_config = global_shared_mem + shared_mem_offset;
        memcpy(ptr_config, config, sizeof(ncclConfig_t));
        shared_mem_offset += sizeof(ncclConfig_t);
        request.config = ptr_config;
    } else {
        request.config = NULL;
    }

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommSplitRp& response = *((ncclCommSplitRp*)(ResponseReceive(response_rb)));

    if (newcomm) {
        *newcomm = response.newcomm;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommFinalizeRq& request = *(ncclCommFinalizeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclCommFinalizeType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommFinalizeRp& response = *((ncclCommFinalizeRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommDestroyRq& request = *(ncclCommDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclCommDestroyType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommDestroyRp& response = *((ncclCommDestroyRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommAbortRq& request = *(ncclCommAbortRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclCommAbortType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommAbortRp& response = *((ncclCommAbortRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommGetAsyncErrorRq& request = *(ncclCommGetAsyncErrorRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclCommGetAsyncErrorType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommGetAsyncErrorRp& response = *((ncclCommGetAsyncErrorRp*)(ResponseReceive(response_rb)));

    if (asyncError) {
        *asyncError = response.asyncError;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommCountRq& request = *(ncclCommCountRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclCommCountType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommCountRp& response = *((ncclCommCountRp*)(ResponseReceive(response_rb)));

    if (count) {
        *count = response.count;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommCuDeviceRq& request = *(ncclCommCuDeviceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclCommCuDeviceType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommCuDeviceRp& response = *((ncclCommCuDeviceRp*)(ResponseReceive(response_rb)));

    if (device) {
        *device = response.device;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommUserRankRq& request = *(ncclCommUserRankRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.type = ncclCommUserRankType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommUserRankRp& response = *((ncclCommUserRankRp*)(ResponseReceive(response_rb)));

    if (rank) {
        *rank = response.rank;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommRegisterRq& request = *(ncclCommRegisterRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.buff = buff;
    request.size = size;
    request.type = ncclCommRegisterType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommRegisterRp& response = *((ncclCommRegisterRp*)(ResponseReceive(response_rb)));

    if (handle) {
        *handle = response.handle;
    }

    return response.ncclResult;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclCommDeregisterRq& request = *(ncclCommDeregisterRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.comm = comm;
    request.handle = handle;
    request.type = ncclCommDeregisterType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclCommDeregisterRp& response = *((ncclCommDeregisterRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclMemAllocRq& request = *(ncclMemAllocRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.size = size;
    request.type = ncclMemAllocType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclMemAllocRp& response = *((ncclMemAllocRp*)(ResponseReceive(response_rb)));

    if (ptr) {
        *ptr = response.ptr;
    }

    return response.ncclResult;
}

ncclResult_t ncclMemFree(void* ptr) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclMemFreeRq& request = *(ncclMemFreeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.ptr = ptr;
    request.type = ncclMemFreeType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclMemFreeRp& response = *((ncclMemFreeRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclAllReduceRq& request = *(ncclAllReduceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.sendbuff = sendbuff;
    request.recvbuff = recvbuff;
    request.count = count;
    request.datatype = datatype;
    request.op = op;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclAllReduceType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclAllReduceRp& response = *((ncclAllReduceRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root,
                            ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclBroadcastRq& request = *(ncclBroadcastRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.sendbuff = sendbuff;
    request.recvbuff = recvbuff;
    request.count = count;
    request.datatype = datatype;
    request.root = root;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclBroadcastType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclBroadcastRp& response = *((ncclBroadcastRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                        int root, ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclBcastRq& request = *(ncclBcastRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.buff = buff;
    request.count = count;
    request.datatype = datatype;
    request.root = root;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclBcastType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclBcastRp& response = *((ncclBcastRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op,
                         int root, ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclReduceRq& request = *(ncclReduceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.sendbuff = sendbuff;
    request.recvbuff = recvbuff;
    request.count = count;
    request.datatype = datatype;
    request.op = op;
    request.root = root;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclReduceType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclReduceRp& response = *((ncclReduceRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                            ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclAllGatherRq& request = *(ncclAllGatherRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.sendbuff = sendbuff;
    request.recvbuff = recvbuff;
    request.sendcount = sendcount;
    request.datatype = datatype;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclAllGatherType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclAllGatherRp& response = *((ncclAllGatherRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                ncclDataType_t datatype, ncclRedOp_t op,
                                ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclReduceScatterRq& request = *(ncclReduceScatterRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.sendbuff = sendbuff;
    request.recvbuff = recvbuff;
    request.recvcount = recvcount;
    request.datatype = datatype;
    request.op = op;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclReduceScatterType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclReduceScatterRp& response = *((ncclReduceScatterRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclGroupStart() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclGroupStartRq& request = *(ncclGroupStartRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = ncclGroupStartType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclGroupStartRp& response = *((ncclGroupStartRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclGroupEnd() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclGroupEndRq& request = *(ncclGroupEndRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = ncclGroupEndType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclGroupEndRp& response = *((ncclGroupEndRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclGroupSimulateEndRq& request = *(ncclGroupSimulateEndRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.simInfo = simInfo;
    request.type = ncclGroupSimulateEndType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclGroupSimulateEndRp& response = *((ncclGroupSimulateEndRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                       int peer, ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclSendRq& request = *(ncclSendRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.sendbuff = sendbuff;
    request.count = count;
    request.datatype = datatype;
    request.peer = peer;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclSendType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclSendRp& response = *((ncclSendRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                       int peer, ncclComm_t comm, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclRecvRq& request = *(ncclRecvRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.recvbuff = recvbuff;
    request.count = count;
    request.datatype = datatype;
    request.peer = peer;
    request.comm = comm;
    request.stream = stream;
    request.type = ncclRecvType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclRecvRp& response = *((ncclRecvRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
                                        ncclScalarResidence_t residence, ncclComm_t comm) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclRedOpCreatePreMulSumRq& request = *(ncclRedOpCreatePreMulSumRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.op = op;
    request.scalar = scalar;
    request.datatype = datatype;
    request.residence = residence;
    request.comm = comm;
    request.type = ncclRedOpCreatePreMulSumType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclRedOpCreatePreMulSumRp& response = *((ncclRedOpCreatePreMulSumRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return ncclUnhandledCudaError;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return ncclSystemError;
    }

    ncclRedOpDestroyRq& request = *(ncclRedOpDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.op = op;
    request.comm = comm;
    request.type = ncclRedOpDestroyType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return ncclSystemError;
    }

    ncclRedOpDestroyRp& response = *((ncclRedOpDestroyRp*)(ResponseReceive(response_rb)));

    return response.ncclResult;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference,
    int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoGetHeuristicRq& request = *(cublasLtMatmulAlgoGetHeuristicRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulAlgoGetHeuristicType;
    request.lightHandle = lightHandle;
    request.operationDesc = operationDesc;
    request.Adesc = Adesc;
    request.Bdesc = Bdesc;
    request.Cdesc = Cdesc;
    request.Ddesc = Ddesc;
    request.preference = preference;
    request.requestedAlgoCount = requestedAlgoCount;
    request.shared_mem_offset = shared_mem_offset;
    void* ptr_heuristicResultsArray = global_shared_mem + shared_mem_offset;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    size_t size = requestedAlgoCount * sizeof(cublasLtMatmulHeuristicResult_t);

    cublasLtMatmulAlgoGetHeuristicRp& response = *((cublasLtMatmulAlgoGetHeuristicRp*)(ResponseReceive(response_rb)));

    memcpy(heuristicResultsArray, ptr_heuristicResultsArray, size);
    shared_mem_offset += size;
    *returnAlgoCount = response.returnAlgoCount;

    return response.status;
}
extern "C" cudaError_t cudaThreadExit() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaThreadExitRq& request = *(cudaThreadExitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaThreadExitType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache* pCacheConfig) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaThreadGetCacheConfigRq& request = *(cudaThreadGetCacheConfigRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaThreadGetCacheConfigType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaThreadGetCacheConfigRp& response = *((cudaThreadGetCacheConfigRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pCacheConfig = response.cacheConfig;
    }

    return response.result;
}

extern "C" cudaError_t cudaThreadGetLimit(size_t* pValue, enum cudaLimit limit) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaThreadGetLimitRq& request = *(cudaThreadGetLimitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaThreadGetLimitType;
    request.Value = *pValue;
    request.limit = limit;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaThreadSetCacheConfigRq& request = *(cudaThreadSetCacheConfigRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaThreadSetCacheConfigType;
    request.cacheConfig = cacheConfig;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaThreadSetLimitRq& request = *(cudaThreadSetLimitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaThreadSetLimitType;
    request.limit = limit;
    request.value = value;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaThreadSynchronize() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaThreadSynchronizeRq& request = *(cudaThreadSynchronizeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaThreadSynchronizeType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaGetLastError() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaGetLastErrorRq& request = *(cudaGetLastErrorRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaGetLastErrorType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaPeekAtLastError() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaPeekAtLastErrorRq& request = *(cudaPeekAtLastErrorRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaPeekAtLastErrorType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(
    cudaFlushGPUDirectRDMAWritesTarget target,
    cudaFlushGPUDirectRDMAWritesScope scope) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceFlushGPUDirectRDMAWritesRq& request = *(cudaDeviceFlushGPUDirectRDMAWritesRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceFlushGPUDirectRDMAWritesType;
    request.target = target;
    request.scope = scope;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceFlushGPUDirectRDMAWritesRp& response = *((cudaDeviceFlushGPUDirectRDMAWritesRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetAttributeRq& request = *(cudaDeviceGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetAttributeType;
    request.attr = attr;
    request.device = device;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetAttributeRp& response = *((cudaDeviceGetAttributeRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *value = response.value;
    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetByPCIBusIdRq& request = *(cudaDeviceGetByPCIBusIdRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetByPCIBusIdType;
    strncpy(request.pciBusId, pciBusId, sizeof(request.pciBusId));
    request.pciBusId[sizeof(request.pciBusId) - 1] = '\0';

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetByPCIBusIdRp& response = *((cudaDeviceGetByPCIBusIdRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *device = response.device;
    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetCacheConfigRq& request = *(cudaDeviceGetCacheConfigRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetCacheConfigType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetCacheConfigRp& response = *((cudaDeviceGetCacheConfigRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pCacheConfig = response.cacheConfig;
    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetDefaultMemPoolRq& request = *(cudaDeviceGetDefaultMemPoolRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetDefaultMemPoolType;
    request.device = device;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetDefaultMemPoolRp& response = *((cudaDeviceGetDefaultMemPoolRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *memPool = response.memPool;
    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetLimitRq& request = *(cudaDeviceGetLimitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetLimitType;
    request.limit = limit;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetLimitRp& response = *((cudaDeviceGetLimitRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pValue = response.value;
    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetMemPoolRq& request = *(cudaDeviceGetMemPoolRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetMemPoolType;
    request.device = device;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetMemPoolRp& response = *((cudaDeviceGetMemPoolRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *memPool = response.memPool;
    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetNvSciSyncAttributesRq& request = *(cudaDeviceGetNvSciSyncAttributesRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetNvSciSyncAttributesType;
    request.device = device;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetNvSciSyncAttributesRp& response = *((cudaDeviceGetNvSciSyncAttributesRp*)(ResponseReceive(response_rb)));
    response.nvSciSyncAttrList = nvSciSyncAttrList;

    if (response.result == cudaSuccess) {

    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetP2PAttributeRq& request = *(cudaDeviceGetP2PAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetP2PAttributeType;
    request.attr = attr;
    request.srcDevice = srcDevice;
    request.dstDevice = dstDevice;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetP2PAttributeRp& response = *((cudaDeviceGetP2PAttributeRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *value = response.value;
    }

    return response.result;
}
extern "C" cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetPCIBusIdRq& request = *(cudaDeviceGetPCIBusIdRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetPCIBusIdType;
    request.device = device;
    request.len = len;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetPCIBusIdRp& response = *((cudaDeviceGetPCIBusIdRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        strncpy(pciBusId, response.pciBusId, len);
    }

    return response.result;
}

extern "C" cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return cudaErrorUnknown;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetStreamPriorityRangeRq& request = *(cudaDeviceGetStreamPriorityRangeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetStreamPriorityRangeType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetStreamPriorityRangeRp& response = *((cudaDeviceGetStreamPriorityRangeRp*)(ResponseReceive(response_rb)));

    if (response.result != cudaSuccess) {

        std::cerr << "Error in response: " << response.result << std::endl;
        return response.result;
    }

    if (leastPriority) {
        *leastPriority = response.leastPriority;
    }
    if (greatestPriority) {
        *greatestPriority = response.greatestPriority;
    }

    return cudaSuccess;
}
extern "C" cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(
    size_t* maxWidthInElements,
    const cudaChannelFormatDesc* fmtDesc,
    int device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceGetTexture1DLinearMaxWidthRq& request = *(cudaDeviceGetTexture1DLinearMaxWidthRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceGetTexture1DLinearMaxWidthType;
    request.fmtDesc = *fmtDesc;
    request.device = device;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceGetTexture1DLinearMaxWidthRp& response = *((cudaDeviceGetTexture1DLinearMaxWidthRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *maxWidthInElements = response.maxWidthInElements;
    }

    return response.result;
}

extern "C" cudaError_t cudaDeviceReset() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceResetRq& request = *(cudaDeviceResetRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceResetType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceResetRp& response = *((cudaDeviceResetRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceSetCacheConfigRq& request = *(cudaDeviceSetCacheConfigRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceSetCacheConfigType;
    request.cacheConfig = cacheConfig;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceSetCacheConfigRp& response = *((cudaDeviceSetCacheConfigRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceSetLimitRq& request = *(cudaDeviceSetLimitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceSetLimitType;
    request.limit = limit;
    request.value = value;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceSetLimitRp& response = *((cudaDeviceSetLimitRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceSetMemPoolRq& request = *(cudaDeviceSetMemPoolRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceSetMemPoolType;
    request.device = device;
    request.memPool = memPool;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceSetMemPoolRp& response = *((cudaDeviceSetMemPoolRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaDeviceSynchronize() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDeviceSynchronizeRq& request = *(cudaDeviceSynchronizeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDeviceSynchronizeType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDeviceSynchronizeRp& response = *((cudaDeviceSynchronizeRp*)(ResponseReceive(response_rb)));

    return response.result;
}

extern "C" cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaInitDeviceRq& request = *(cudaInitDeviceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaInitDeviceType;
    request.device = device;
    request.deviceFlags = deviceFlags;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaInitDeviceRp& response = *((cudaInitDeviceRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaIpcCloseMemHandleRq& request = *(cudaIpcCloseMemHandleRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaIpcCloseMemHandleType;
    request.devPtr = devPtr;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaIpcCloseMemHandleRp& response = *((cudaIpcCloseMemHandleRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaIpcGetEventHandleRq& request = *(cudaIpcGetEventHandleRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaIpcGetEventHandleType;
    request.event = event;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaIpcGetEventHandleRp& response = *((cudaIpcGetEventHandleRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *handle = response.handle;
    }

    return response.result;
}
extern "C" cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaIpcGetMemHandleRq& request = *(cudaIpcGetMemHandleRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaIpcGetMemHandleType;
    request.devPtr = devPtr;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaIpcGetMemHandleRp& response = *((cudaIpcGetMemHandleRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *handle = response.handle;
    }

    return response.result;
}
extern "C" cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaIpcOpenEventHandleRq& request = *(cudaIpcOpenEventHandleRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaIpcOpenEventHandleType;
    request.handle = handle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaIpcOpenEventHandleRp& response = *((cudaIpcOpenEventHandleRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *event = response.event;
    }

    return response.result;
}
extern "C" cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaIpcOpenMemHandleRq& request = *(cudaIpcOpenMemHandleRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaIpcOpenMemHandleType;
    request.handle = handle;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaIpcOpenMemHandleRp& response = *((cudaIpcOpenMemHandleRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *devPtr = response.devPtr;
    }

    return response.result;
}
extern "C" cudaError_t cudaCtxResetPersistingL2Cache(void) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaCtxResetPersistingL2CacheRq& request = *(cudaCtxResetPersistingL2CacheRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaCtxResetPersistingL2CacheType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamAddCallback(
    cudaStream_t stream,
    cudaStreamCallback_t callback,
    void* userData,
    unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamAddCallbackRq& request = *(cudaStreamAddCallbackRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamAddCallbackType;
    request.stream = stream;
    request.callback = callback;
    request.userData = userData;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamAttachMemAsyncRq& request = *(cudaStreamAttachMemAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamAttachMemAsyncType;
    request.stream = stream;
    request.devPtr = devPtr;
    request.length = length;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamBeginCaptureRq& request = *(cudaStreamBeginCaptureRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamBeginCaptureType;
    request.stream = stream;
    request.mode = mode;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamCopyAttributesRq& request = *(cudaStreamCopyAttributesRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamCopyAttributesType;
    request.dst = dst;
    request.src = src;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamCreateWithFlagsRq& request = *(cudaStreamCreateWithFlagsRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamCreateWithFlagsType;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamCreateWithFlagsRp& response = *((cudaStreamCreateWithFlagsRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pStream = response.stream;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamCreateWithPriorityRq& request = *(cudaStreamCreateWithPriorityRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamCreateWithPriorityType;
    request.flags = flags;
    request.priority = priority;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamCreateWithPriorityRp& response = *((cudaStreamCreateWithPriorityRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pStream = response.stream;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamEndCaptureRq& request = *(cudaStreamEndCaptureRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamEndCaptureType;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamEndCaptureRp& response = *((cudaStreamEndCaptureRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pGraph = response.graph;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamGetAttribute(
    cudaStream_t hStream,
    cudaStreamAttrID attr,
    cudaStreamAttrValue* value_out) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamGetAttributeRq& request = *(cudaStreamGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamGetAttributeType;
    request.hStream = hStream;
    request.attr = attr;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamGetAttributeRp& response = *((cudaStreamGetAttributeRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *value_out = response.value;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamGetCaptureInfo(
    cudaStream_t stream,
    cudaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out,
    cudaGraph_t* graph_out,
    const cudaGraphNode_t** dependencies_out,
    size_t* numDependencies_out) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamGetCaptureInfoRq& request = *(cudaStreamGetCaptureInfoRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamGetCaptureInfoType;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamGetCaptureInfoRp& response = *((cudaStreamGetCaptureInfoRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *captureStatus_out = response.captureStatus;
        if (id_out)
            *id_out = response.id;
        if (graph_out)
            *graph_out = response.graph;
        if (dependencies_out)
            *dependencies_out = response.dependencies;
        if (numDependencies_out)
            *numDependencies_out = response.numDependencies;
    }

    return response.result;
}

extern "C" cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamGetFlagsRq& request = *(cudaStreamGetFlagsRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamGetFlagsType;
    request.hStream = hStream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamGetFlagsRp& response = *((cudaStreamGetFlagsRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *flags = response.flags;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamGetIdRq& request = *(cudaStreamGetIdRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamGetIdType;
    request.hStream = hStream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamGetIdRp& response = *((cudaStreamGetIdRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *streamId = response.streamId;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamGetPriorityRq& request = *(cudaStreamGetPriorityRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamGetPriorityType;
    request.hStream = hStream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamGetPriorityRp& response = *((cudaStreamGetPriorityRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *priority = response.priority;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamIsCapturingRq& request = *(cudaStreamIsCapturingRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamIsCapturingType;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {

        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamIsCapturingRp& response = *((cudaStreamIsCapturingRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {

        *pCaptureStatus = response.status;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamSetAttribute(
    cudaStream_t hStream,
    cudaStreamAttrID attr,
    const cudaStreamAttrValue* value) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamSetAttributeRq& request = *(cudaStreamSetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamSetAttributeType;
    request.hStream = hStream;
    request.attr = attr;

    if (value) {
        request.value = *value;
    }

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamSetAttributeRp& response = *((cudaStreamSetAttributeRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(
    cudaStream_t stream,
    cudaGraphNode_t* dependencies,
    size_t numDependencies,
    unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamUpdateCaptureDependenciesRq& request = *(cudaStreamUpdateCaptureDependenciesRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamUpdateCaptureDependenciesType;
    request.stream = stream;
    request.dependencies = dependencies;
    request.numDependencies = numDependencies;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamUpdateCaptureDependenciesRp& response = *((cudaStreamUpdateCaptureDependenciesRp*)(ResponseReceive(response_rb)));

    return response.result;
}

extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaThreadExchangeStreamCaptureModeRq& request = *(cudaThreadExchangeStreamCaptureModeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaThreadExchangeStreamCaptureModeType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaThreadExchangeStreamCaptureModeRp& response = *((cudaThreadExchangeStreamCaptureModeRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *mode = response.mode;
    }

    return response.result;
}

extern "C" cudaError_t cudaFuncGetParamInfo(
    const void* func,
    size_t paramIndex,
    size_t* paramOffset,
    size_t* paramSize) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaFuncGetParamInfoRq& request = *(cudaFuncGetParamInfoRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaFuncGetParamInfoType;
    request.func = func;
    request.paramIndex = paramIndex;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaFuncGetParamInfoRp& response = *((cudaFuncGetParamInfoRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        if (paramOffset)
            *paramOffset = response.paramOffset;
        if (paramSize)
            *paramSize = response.paramSize;
    }

    return response.result;
}
extern "C" cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaFuncSetAttributeRq& request = *(cudaFuncSetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaFuncSetAttributeType;
    request.func = func;
    request.attr = attr;
    request.value = value;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaFuncSetAttributeRp& response = *((cudaFuncSetAttributeRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" void* cudaGetParameterBuffer(size_t alignment, size_t size) {
    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return nullptr;
    }

    void* response;
    return response;
}
extern "C" void cudaGridDependencySynchronize() {

}

extern "C" cudaError_t cudaLaunchCooperativeKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaLaunchCooperativeKernelRq& request = *(cudaLaunchCooperativeKernelRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaLaunchCooperativeKernelType;
    request.func = func;
    request.gridDim = gridDim;
    request.blockDim = blockDim;
    request.args = args;
    request.sharedMem = sharedMem;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaLaunchCooperativeKernelRp& response = *((cudaLaunchCooperativeKernelRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaLaunchCooperativeKernelMultiDevice(
    cudaLaunchParams* launchParamsList,
    unsigned int numDevices,
    unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaLaunchCooperativeKernelMultiDeviceRq& request = *(cudaLaunchCooperativeKernelMultiDeviceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaLaunchCooperativeKernelMultiDeviceType;
    request.launchParamsList = launchParamsList;
    request.numDevices = numDevices;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaLaunchCooperativeKernelMultiDeviceRp& response = *((cudaLaunchCooperativeKernelMultiDeviceRp*)(ResponseReceive(response_rb)));

    return response.result;
}

extern "C" cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaLaunchHostFuncRq& request = *(cudaLaunchHostFuncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaLaunchHostFuncType;
    request.stream = stream;
    request.fn = fn;
    request.userData = userData;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaLaunchHostFuncRp& response = *((cudaLaunchHostFuncRp*)(ResponseReceive(response_rb)));

    return response.result;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, (last - first + 1));
}

void loadFunctionInfo(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open funcinfo.txt");
    }

    std::string line, current_function;
    FunctionInfo current_info;

    while (std::getline(file, line)) {

        line = trim(line);

        if (line.find("Function: ") == 0) {

            if (!current_function.empty()) {
                function_map[current_function] = current_info;
                current_info = FunctionInfo();
            }

            current_function = line.substr(10);
        } else if (line.find("Cubin File: ") == 0) {

            current_info.cubin_file = line.substr(12);
        } else if (line.find("Parameter Count: ") == 0) {

            current_info.param_count = std::stoi(line.substr(17));
        } else if (line.find("Parameter Lengths: ") == 0) {

            size_t start = line.find("[") + 1;
            size_t end = line.find("]");
            std::string lengths = line.substr(start, end - start);
            std::istringstream iss(lengths);
            std::string length;
            while (std::getline(iss, length, ',')) {
                current_info.param_lengths.push_back(std::stoi(trim(length)));
            }
        }
    }

    if (!current_function.empty()) {
        function_map[current_function] = current_info;
    }

    file.close();
}

const FunctionInfo& getFunctionInfo(const std::string& kernel_name) {
    auto it = function_map.find(kernel_name);
    if (it == function_map.end()) {
        throw std::runtime_error("Function information not found for kernel: " + kernel_name);
    }
    return it->second;
}

extern "C" cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    const char* name;
    cudaError_t error;
    static char* shared_mem;

    if (alloc == 0) {
        std::lock_guard<std::mutex> lock(alloc_mutex);
        shared_mem = initSharedMemory();
        global_shared_mem = shared_mem;
        alloc = 1;
    }

    if (load == 0) {
        try {
            loadFunctionInfo(FUNCINFO);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return cudaErrorUnknown;
        }
        load = 1;
    }
    std::lock_guard<std::mutex> lock(alloc_mutex);

    error = cudaFuncGetName(&name, func);

    std::string kernel_name(name);
    if (error == cudaSuccess) {
        printf("Kernel name: %s\n", kernel_name.c_str());
    } else {
        printf("Error retrieving kernel name: %s\n", cudaGetErrorString(error));
    }

    std::string cubin_file_path;
    std::vector<int> param_lengths;
    try {
        const FunctionInfo& info = getFunctionInfo(name);
        cubin_file_path = info.cubin_file;
        param_lengths = info.param_lengths;

        printf("Cubin File: %s\n", cubin_file_path.c_str());
        printf("Parameter Lengths: ");
        for (int len : param_lengths) {
            printf("%d ", len);
        }
        printf("\n");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return cudaErrorUnknown;
    }
    printf("param_lengths: %ld\n", param_lengths.size());
    size_t args_size = param_lengths.size() * sizeof(void*);
    if (shared_mem_offset + args_size > (size_t)(1024 * 1024 * 1024)) {
        std::cerr << "Shared memory out of space!" << std::endl;
        return cudaErrorUnknown;
    }

    void** shared_args = (void**)(global_shared_mem + shared_mem_offset);
    shared_mem_offset += args_size;

    std::vector<void*> ptr_args(param_lengths.size());
    for (size_t i = 0; i < param_lengths.size(); ++i) {
        size_t param_size = param_lengths[i];
        printf("param_size: %ld\n", param_size);

        if (shared_mem_offset + param_size > (size_t)(1024 * 1024 * 1024)) {
            std::cerr << "Shared memory out of space!" << std::endl;
            return cudaErrorUnknown;
        }

        void* ptr = global_shared_mem + shared_mem_offset;
        if (param_size < 8){
            param_size = 8;
        }
        memcpy(ptr, args[i], param_size);
        unsigned long long* address1 = reinterpret_cast<unsigned long long*>(args[i]);
        unsigned long long* address = reinterpret_cast<unsigned long long*>(ptr);
        printf("argsi address: %llx\n", *address1);
        printf("ptr address: %llx\n", *address);
        shared_args[i] = ptr;

        shared_mem_offset += param_size;
    }
    printf("Kernel arguments have been copied successfully.\n");
    size_t kernel_name_length = kernel_name.size();
    size_t cubin_file_path_length = cubin_file_path.size();

    if (shared_mem_offset + kernel_name_length > (size_t)(1024 * 1024 * 1024)) {
        std::cerr << "Shared memory out of space!" << std::endl;
        return cudaErrorUnknown;
    }

    char* kernel_name_pointer = global_shared_mem + shared_mem_offset;
    memcpy(kernel_name_pointer, kernel_name.c_str(), kernel_name_length);
    kernel_name_pointer[kernel_name_length] = '\0';
    shared_mem_offset += kernel_name_length + 1;

    if (shared_mem_offset + cubin_file_path_length > (size_t)(1024 * 1024 * 1024)) {
        std::cerr << "Shared memory out of space!" << std::endl;
        return cudaErrorUnknown;
    }

    char* cubin_file_path_pointer = global_shared_mem + shared_mem_offset;
    memcpy(cubin_file_path_pointer, cubin_file_path.c_str(), cubin_file_path_length);
    cubin_file_path_pointer[cubin_file_path_length] = '\0';
    shared_mem_offset += cubin_file_path_length + 1;

    cudaLaunchKernelRq& request = *(cudaLaunchKernelRq*)NewRequest(request_rb); request.requestSize = sizeof(request);

    request.type = cudaLaunchKernelType;
    request.gridDim = gridDim;
    request.blockDim = blockDim;
    request.sharedMem = sharedMem;
    request.stream = stream;
    request.argsSize = param_lengths.size();
    request.args = shared_args;
    request.func = func;
    request.client_shared_mem = global_shared_mem;
    request.kernel_name_pointer = kernel_name_pointer;
    request.cubin_file_path_pointer = cubin_file_path_pointer;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaLaunchKernelRp& response = *((cudaLaunchKernelRp*)(ResponseReceive(response_rb)));

    return response.result;
}
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSgemmRq& request = *(cublasSgemmRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSgemmType;
    request.handle = handle;
    request.transa = transa;
    request.transb = transb;
    request.m = m;
    request.n = n;
    request.k = k;
    request.alpha = *alpha;
    request.A = A;
    request.lda = lda;
    request.B = B;
    request.ldb = ldb;
    request.beta = *beta;
    request.C = C;
    request.ldc = ldc;
    request.client_shared_mem = global_shared_mem;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSgemmRp& response = *((cublasSgemmRp*)(ResponseReceive(response_rb)));

    return response.status;
}
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    long long int strideA,
    const float *B, int ldb,
    long long int strideB,
    const float *beta,
    float *C, int ldc,
    long long int strideC,
    int batchCount) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSgemmStridedBatchedRq& request = *(cublasSgemmStridedBatchedRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSgemmStridedBatchedType;
    request.handle = handle;
    request.transa = transa;
    request.transb = transb;
    request.m = m;
    request.n = n;
    request.k = k;
    request.alpha = *alpha;
    request.A = A;
    request.lda = lda;
    request.strideA = strideA;
    request.B = B;
    request.ldb = ldb;
    request.strideB = strideB;
    request.beta = *beta;
    request.C = C;
    request.ldc = ldc;
    request.strideC = strideC;
    request.batchCount = batchCount;
    request.client_shared_mem = global_shared_mem;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSgemmStridedBatchedRp& response = *((cublasSgemmStridedBatchedRp*)(ResponseReceive(response_rb)));

    return response.status;
}
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasCreateRq& request = *(cublasCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasCreateType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasCreateRp& response = *((cublasCreateRp*)(ResponseReceive(response_rb)));

    if (response.status == CUBLAS_STATUS_SUCCESS) {
        *handle = response.handle;
    }

    return response.status;
}
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetStreamRq& request = *(cublasSetStreamRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetStreamType;
    request.handle = handle;
    request.streamId = streamId;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetStreamRp& response = *((cublasSetStreamRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasDestroyRq& request = *(cublasDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasDestroyType;
    request.handle = handle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasDestroyRp& response = *((cublasDestroyRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetPropertyRq& request = *(cublasGetPropertyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = type;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetPropertyRp& response = *((cublasGetPropertyRp*)(ResponseReceive(response_rb)));

    *value = response.value;
    return response.status;
}

cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetWorkspaceRq& request = *(cublasSetWorkspaceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetWorkspaceType;
    request.handle = handle;
    request.workspace = workspace;
    request.workspaceSizeInBytes = workspaceSizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetWorkspaceRp& response = *((cublasSetWorkspaceRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetStreamRq& request = *(cublasGetStreamRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetStreamType;
    request.handle = handle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetStreamRp& response = *((cublasGetStreamRp*)(ResponseReceive(response_rb)));

    *streamId = response.streamId;
    return response.status;
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetPointerModeRq& request = *(cublasGetPointerModeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetPointerModeType;
    request.handle = handle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetPointerModeRp& response = *((cublasGetPointerModeRp*)(ResponseReceive(response_rb)));

    *mode = response.mode;
    return response.status;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetPointerModeRq& request = *(cublasSetPointerModeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetPointerModeType;
    request.handle = handle;
    request.mode = mode;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetPointerModeRp& response = *((cublasSetPointerModeRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetVectorRq& request = *(cublasSetVectorRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetVectorType;
    request.n = n;
    request.elemSize = elemSize;
    request.x = x;
    request.incx = incx;
    request.y = y;
    request.incy = incy;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetVectorRp& response = *((cublasSetVectorRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetVectorRq& request = *(cublasGetVectorRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetVectorType;
    request.n = n;
    request.elemSize = elemSize;
    request.x = x;
    request.incx = incx;
    request.y = y;
    request.incy = incy;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetVectorRp& response = *((cublasGetVectorRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize,
                                const void *A, int lda, void *B, int ldb) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetMatrixRq& request = *(cublasGetMatrixRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetMatrixType;
    request.rows = rows;
    request.cols = cols;
    request.elemSize = elemSize;
    request.A = A;
    request.lda = lda;
    request.B = B;
    request.ldb = ldb;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetMatrixRp& response = *((cublasGetMatrixRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                                const void *A, int lda, void *B, int ldb) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetMatrixRq& request = *(cublasSetMatrixRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetMatrixType;
    request.rows = rows;
    request.cols = cols;
    request.elemSize = elemSize;
    request.A = A;
    request.lda = lda;
    request.B = B;
    request.ldb = ldb;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetMatrixRp& response = *((cublasSetMatrixRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx,
                                     void *devicePtr, int incy, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetVectorAsyncRq& request = *(cublasSetVectorAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetVectorAsyncType;
    request.n = n;
    request.elemSize = elemSize;
    request.hostPtr = hostPtr;
    request.incx = incx;
    request.devicePtr = devicePtr;
    request.incy = incy;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetVectorAsyncRp& response = *((cublasSetVectorAsyncRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx,
                                     void *hostPtr, int incy, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetVectorAsyncRq& request = *(cublasGetVectorAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetVectorAsyncType;
    request.n = n;
    request.elemSize = elemSize;
    request.devicePtr = devicePtr;
    request.incx = incx;
    request.hostPtr = hostPtr;
    request.incy = incy;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetVectorAsyncRp& response = *((cublasGetVectorAsyncRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                                     int lda, void *B, int ldb, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetMatrixAsyncRq& request = *(cublasSetMatrixAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetMatrixAsyncType;
    request.rows = rows;
    request.cols = cols;
    request.elemSize = elemSize;
    request.A = A;
    request.lda = lda;
    request.B = B;
    request.ldb = ldb;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetMatrixAsyncRp& response = *((cublasSetMatrixAsyncRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                                     int lda, void *B, int ldb, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetMatrixAsyncRq& request = *(cublasGetMatrixAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetMatrixAsyncType;
    request.rows = rows;
    request.cols = cols;
    request.elemSize = elemSize;
    request.A = A;
    request.lda = lda;
    request.B = B;
    request.ldb = ldb;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetMatrixAsyncRp& response = *((cublasGetMatrixAsyncRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetAtomicsModeRq& request = *(cublasSetAtomicsModeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetAtomicsModeType;
    request.handle = handle;
    request.mode = mode;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetAtomicsModeRp& response = *((cublasSetAtomicsModeRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetAtomicsModeRq& request = *(cublasGetAtomicsModeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetAtomicsModeType;
    request.handle = handle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetAtomicsModeRp& response = *((cublasGetAtomicsModeRp*)(ResponseReceive(response_rb)));

    *mode = response.atomicsMode;
    return response.status;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetMathModeRq& request = *(cublasSetMathModeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetMathModeType;
    request.handle = handle;
    request.mode = mode;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetMathModeRp& response = *((cublasSetMathModeRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetMathModeRq& request = *(cublasGetMathModeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetMathModeType;
    request.handle = handle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetMathModeRp& response = *((cublasGetMathModeRp*)(ResponseReceive(response_rb)));

    *mode = response.mathMode;
    return response.status;
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetSmCountTargetRq& request = *(cublasSetSmCountTargetRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetSmCountTargetType;
    request.handle = handle;
    request.smCountTarget = smCountTarget;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetSmCountTargetRp& response = *((cublasSetSmCountTargetRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetSmCountTargetRq& request = *(cublasGetSmCountTargetRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetSmCountTargetType;
    request.handle = handle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetSmCountTargetRp& response = *((cublasGetSmCountTargetRp*)(ResponseReceive(response_rb)));

    *smCountTarget = response.smCountTarget;
    return response.status;
}

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLoggerConfigureRq& request = *(cublasLoggerConfigureRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLoggerConfigureType;
    request.logIsOn = logIsOn;
    request.logToStdOut = logToStdOut;
    request.logToStdErr = logToStdErr;
    request.logFileName = logFileName;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLoggerConfigureRp& response = *((cublasLoggerConfigureRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasGetLoggerCallback(cublasLogCallback* userCallback) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetLoggerCallbackRq& request = *(cublasGetLoggerCallbackRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasGetLoggerCallbackType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasGetLoggerCallbackRp& response = *((cublasGetLoggerCallbackRp*)(ResponseReceive(response_rb)));

    *userCallback = response.userCallback;
    return response.status;
}

cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetLoggerCallbackRq& request = *(cublasSetLoggerCallbackRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasSetLoggerCallbackType;
    request.userCallback = userCallback;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasSetLoggerCallbackRp& response = *((cublasSetLoggerCallbackRp*)(ResponseReceive(response_rb)));

    return response.status;
}

    extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaLaunchKernelExCRq& request = *(cudaLaunchKernelExCRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaLaunchKernelExCType;
    request.config = *config;
    request.func = func;
    request.args = args;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaLaunchKernelExCRp& response = *((cudaLaunchKernelExCRp*)(ResponseReceive(response_rb)));

    return response.result;
}

extern "C" cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDestroyExternalMemoryRq& request = *(cudaDestroyExternalMemoryRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDestroyExternalMemoryType;
    request.extMem = extMem;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDestroyExternalMemoryRp& response = *((cudaDestroyExternalMemoryRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaDestroyExternalSemaphoreRq& request = *(cudaDestroyExternalSemaphoreRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaDestroyExternalSemaphoreType;
    request.extSem = extSem;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaDestroyExternalSemaphoreRp& response = *((cudaDestroyExternalSemaphoreRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaExternalMemoryGetMappedBuffer(
    void** devPtr,
    cudaExternalMemory_t extMem,
    const cudaExternalMemoryBufferDesc* bufferDesc) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaExternalMemoryGetMappedBufferRq& request = *(cudaExternalMemoryGetMappedBufferRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaExternalMemoryGetMappedBufferType;
    request.extMem = extMem;
    request.bufferDesc = *bufferDesc;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaExternalMemoryGetMappedBufferRp& response = *((cudaExternalMemoryGetMappedBufferRp*)(ResponseReceive(response_rb)));

    *devPtr = response.devPtr;
    return response.result;
}
extern "C" cudaError_t cudaExternalMemoryGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap,
    cudaExternalMemory_t extMem,
    const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaExternalMemoryGetMappedMipmappedArrayRq& request = *(cudaExternalMemoryGetMappedMipmappedArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaExternalMemoryGetMappedMipmappedArrayType;
    request.extMem = extMem;
    request.mipmapDesc = *mipmapDesc;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaExternalMemoryGetMappedMipmappedArrayRp& response = *((cudaExternalMemoryGetMappedMipmappedArrayRp*)(ResponseReceive(response_rb)));

    *mipmap = response.mipmap;
    return response.result;
}
extern "C" cudaError_t cudaImportExternalMemory(
    cudaExternalMemory_t* extMem_out,
    const cudaExternalMemoryHandleDesc* memHandleDesc) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaImportExternalMemoryRq& request = *(cudaImportExternalMemoryRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaImportExternalMemoryType;
    request.extMem_out = extMem_out;
    request.memHandleDesc = *memHandleDesc;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaImportExternalMemoryRp& response = *((cudaImportExternalMemoryRp*)(ResponseReceive(response_rb)));

    return response.result;
}
extern "C" cudaError_t cudaImportExternalSemaphore(
    cudaExternalSemaphore_t* extSem_out,
    const cudaExternalSemaphoreHandleDesc* semHandleDesc) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaImportExternalSemaphoreRq& request = *(cudaImportExternalSemaphoreRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaImportExternalSemaphoreType;
    request.extSem_out = extSem_out;
    request.semHandleDesc = *semHandleDesc;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaImportExternalSemaphoreRp& response = *((cudaImportExternalSemaphoreRp*)(ResponseReceive(response_rb)));

    return response.result;
}

extern "C" cudaError_t cudaWaitExternalSemaphoresAsync(
    const cudaExternalSemaphore_t* extSemArray,
    const cudaExternalSemaphoreWaitParams* paramsArray,
    unsigned int numExtSems,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaWaitExternalSemaphoresAsyncRq& request = *(cudaWaitExternalSemaphoresAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaWaitExternalSemaphoresAsyncType;
    request.extSemArray = extSemArray;
    request.paramsArray = paramsArray;
    request.numExtSems = numExtSems;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaWaitExternalSemaphoresAsyncRp& response = *((cudaWaitExternalSemaphoresAsyncRp*)(ResponseReceive(response_rb)));

    return response.result;
}

extern "C" cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(
    size_t* dynamicSmemSize,
    const void* func,
    int numBlocks,
    int blockSize) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaOccupancyAvailableDynamicSMemPerBlockRq& request = *(cudaOccupancyAvailableDynamicSMemPerBlockRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaOccupancyAvailableDynamicSMemPerBlockType;
    request.func = func;
    request.numBlocks = numBlocks;
    request.blockSize = blockSize;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaOccupancyAvailableDynamicSMemPerBlockRp& response = *((cudaOccupancyAvailableDynamicSMemPerBlockRp*)(ResponseReceive(response_rb)));

    *dynamicSmemSize = response.dynamicSmemSize;
    return response.result;
}
extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks,
    const void* func,
    int blockSize,
    size_t dynamicSMemSize) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxActiveBlocksPerMultiprocessorRq& request = *(cudaOccupancyMaxActiveBlocksPerMultiprocessorRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaOccupancyMaxActiveBlocksPerMultiprocessorType;
    request.func = func;
    request.blockSize = blockSize;
    request.dynamicSMemSize = dynamicSMemSize;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxActiveBlocksPerMultiprocessorRp& response = *((cudaOccupancyMaxActiveBlocksPerMultiprocessorRp*)(ResponseReceive(response_rb)));

    *numBlocks = response.numBlocks;
    return response.result;
}
extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks,
    const void* func,
    int blockSize,
    size_t dynamicSMemSize,
    unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRq& request = *(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsType;
    request.func = func;
    request.blockSize = blockSize;
    request.dynamicSMemSize = dynamicSMemSize;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRp& response = *((cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsRp*)(ResponseReceive(response_rb)));

    *numBlocks = response.numBlocks;
    return response.result;
}
extern "C" cudaError_t cudaOccupancyMaxActiveClusters(
    int* numClusters,
    const void* func,
    const cudaLaunchConfig_t* launchConfig) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxActiveClustersRq& request = *(cudaOccupancyMaxActiveClustersRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaOccupancyMaxActiveClustersType;
    request.func = func;
    request.launchConfig = *launchConfig;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxActiveClustersRp& response = *((cudaOccupancyMaxActiveClustersRp*)(ResponseReceive(response_rb)));

    *numClusters = response.numClusters;
    return response.result;
}
extern "C" cudaError_t cudaOccupancyMaxPotentialClusterSize(
    int* clusterSize,
    const void* func,
    const cudaLaunchConfig_t* launchConfig) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxPotentialClusterSizeRq& request = *(cudaOccupancyMaxPotentialClusterSizeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaOccupancyMaxPotentialClusterSizeType;
    request.func = func;
    request.launchConfig = *launchConfig;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaOccupancyMaxPotentialClusterSizeRp& response = *((cudaOccupancyMaxPotentialClusterSizeRp*)(ResponseReceive(response_rb)));

    *clusterSize = response.clusterSize;
    return response.result;
}

extern "C" cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaChooseDeviceRq& request = *(cudaChooseDeviceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaChooseDeviceType;
    request.prop = prop;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaChooseDeviceRp& response = *((cudaChooseDeviceRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *device = response.device;
    }

    return response.result;
}

cudaError_t cudaGetDevice(int* device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaGetDeviceRq& request = *(cudaGetDeviceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaGetDeviceType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaGetDeviceRp& response = *((cudaGetDeviceRp*)(ResponseReceive(response_rb)));
    if (response.result == cudaSuccess) {
        *device = response.device;
    }
    return response.result;
}

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaGetDevicePropertiesRq& request = *(cudaGetDevicePropertiesRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.device = device;
    request.type = cudaGetDevicePropertiesType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaGetDevicePropertiesRp& response = *((cudaGetDevicePropertiesRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *prop = response.prop;
    }

    return response.result;
}

extern "C" cudaError_t cudaSetDevice(int device) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaSetDeviceRq& request = *(cudaSetDeviceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaSetDeviceType;
    request.device = device;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaSetDeviceFlagsRq& request = *(cudaSetDeviceFlagsRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaSetDeviceFlagsType;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaSetValidDevices(int* device_arr, int len) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaSetValidDevicesRq& request = *(cudaSetValidDevicesRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaSetValidDevicesType;
    request.device_arr = device_arr;
    request.len = len;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamCreateRq& request = *(cudaStreamCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamCreateType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaStreamCreateRp& response = *((cudaStreamCreateRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pStream = response.stream;
    }

    return response.result;
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamDestroyRq& request = *(cudaStreamDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamDestroyType;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamQueryRq& request = *(cudaStreamQueryRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamQueryType;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamSynchronizeRq& request = *(cudaStreamSynchronizeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamSynchronizeType;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaStreamWaitEventRq& request = *(cudaStreamWaitEventRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaStreamWaitEventType;
    request.stream = stream;
    request.event = event;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaEventCreate(cudaEvent_t* event) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaEventCreateRq& request = *(cudaEventCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaEventCreateType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaEventCreateRp& response = *((cudaEventCreateRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *event = response.event;
    }

    return response.result;
}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaEventCreateWithFlagsRq& request = *(cudaEventCreateWithFlagsRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaEventCreateWithFlagsType;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaEventCreateWithFlagsRp& response = *((cudaEventCreateWithFlagsRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *event = response.event;
    }
    return response.result;
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaEventDestroyRq& request = *(cudaEventDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaEventDestroyType;
    request.event = event;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaEventElapsedTimeRq& request = *(cudaEventElapsedTimeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaEventElapsedTimeType;
    request.start = start;
    request.end = end;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaEventElapsedTimeRp& response = *((cudaEventElapsedTimeRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *ms = response.ms;
    }

    return response.result;
}
extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaEventQueryRq& request = *(cudaEventQueryRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaEventQueryType;
    request.event = event;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaEventRecordRq& request = *(cudaEventRecordRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaEventRecordType;
    request.event = event;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaEventSynchronizeRq& request = *(cudaEventSynchronizeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaEventSynchronizeType;
    request.event = event;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaFuncGetAttributes(
    struct cudaFuncAttributes* attr,
    const void* func) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaFuncGetAttributesRq& request = *(cudaFuncGetAttributesRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaFuncGetAttributesType;
    request.func = func;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaFuncGetAttributesRp& response = *((cudaFuncGetAttributesRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *attr = response.attr;
    }

    return response.result;
}
extern "C" cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaFuncSetCacheConfigRq& request = *(cudaFuncSetCacheConfigRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaFuncSetCacheConfigType;
    request.func = func;
    request.cacheConfig = cacheConfig;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

extern "C" cudaError_t cudaSetDoubleForDevice(double* d) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaSetDoubleForDeviceRq& request = *(cudaSetDoubleForDeviceRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaSetDoubleForDeviceType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaSetDoubleForDeviceRp& response = *((cudaSetDoubleForDeviceRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *d = response.d;
    }

    return response.result;
}
extern "C" cudaError_t cudaSetDoubleForHost(double* d) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaSetDoubleForHostRq& request = *(cudaSetDoubleForHostRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaSetDoubleForHostType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaSetDoubleForHostRp& response = *((cudaSetDoubleForHostRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *d = response.d;
    }

    return response.result;
}

extern "C" cudaError_t cudaFree(void* devPtr) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaFreeRq& request = *(cudaFreeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaFreeType;
    request.devPtr = devPtr;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaFreeArray(struct cudaArray* array) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaFreeArrayRq& request = *(cudaFreeArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaFreeArrayType;
    request.array = array;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaFreeHost(void* ptr) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaFreeHostRq& request = *(cudaFreeHostRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaFreeHostType;
    request.ptr = ptr;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaGetSymbolAddressRq& request = *(cudaGetSymbolAddressRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaGetSymbolAddressType;
    request.symbol = symbol;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaGetSymbolAddressRp& response = *((cudaGetSymbolAddressRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *devPtr = response.devPtr;
    }

    return response.result;
}
extern "C" cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaGetSymbolSizeRq& request = *(cudaGetSymbolSizeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaGetSymbolSizeType;
    request.symbol = symbol;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaGetSymbolSizeRp& response = *((cudaGetSymbolSizeRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *size = response.size;
    }

    return response.result;
}
extern "C" cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    static char *shared_mem;
    if (alloc == 0) {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    shared_mem = initSharedMemory();
    global_shared_mem = shared_mem;
    alloc = 1;
    }
    if (shared_mem_offset + size > (long)1024*1024*1024) {
        return cudaErrorUnknown;
    }
    size_t current_offset = shared_mem_offset;
    void * ptr_buf = global_shared_mem + current_offset;
    shared_mem_offset += size;

    cudaHostAllocRq& request = *(cudaHostAllocRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaHostAllocType;
    request.size = size;
    request.flags = flags;
    request.shmem_offset = current_offset;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaHostAllocRp& response = *((cudaHostAllocRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pHost = ptr_buf;
    }

    return response.result;
}
extern "C" cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaHostGetDevicePointerRq& request = *(cudaHostGetDevicePointerRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaHostGetDevicePointerType;
    request.pHost = pHost;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaHostGetDevicePointerRp& response = *((cudaHostGetDevicePointerRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pDevice = response.pDevice;
    }

    return response.result;
}
extern "C" cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaHostGetFlagsRq& request = *(cudaHostGetFlagsRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaHostGetFlagsType;
    request.pHost = pHost;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaHostGetFlagsRp& response = *((cudaHostGetFlagsRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *pFlags = response.pFlags;
    }

    return response.result;
}
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMallocRq& request = *(cudaMallocRq*)NewRequest(request_rb); request.requestSize = sizeof(request);    request.type = cudaMallocType;
    request.size = size;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMallocRp& response = *((cudaMallocRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *devPtr = response.devPtr;
        printf("cudaMalloc memory: %p\n", response.devPtr);
        printf("cudaMalloc memory size: %ld\n", size);
    }

    return response.result;
}
extern "C" cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMalloc3DRq& request = *(cudaMalloc3DRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMalloc3DType;
    request.extent = extent;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMalloc3DRp& response = *((cudaMalloc3DRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        pitchedDevPtr = response.pitchedDevPtr;
    }

    return response.result;
}
extern "C" cudaError_t cudaMalloc3DArray(
    struct cudaArray** array,
    const struct cudaChannelFormatDesc* desc,
    struct cudaExtent extent,
    unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMalloc3DArrayRq& request = *(cudaMalloc3DArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMalloc3DArrayType;
    request.desc = desc;
    request.extent = extent;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMalloc3DArrayRp& response = *((cudaMalloc3DArrayRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *array = response.array;
    }

    return response.result;
}
extern "C" cudaError_t cudaMallocArray(
    struct cudaArray** array,
    const struct cudaChannelFormatDesc* desc,
    size_t width,
    size_t height,
    unsigned int flags) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMallocArrayRq& request = *(cudaMallocArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMallocArrayType;
    request.desc = desc;
    request.width = width;
    request.height = height;
    request.flags = flags;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMallocArrayRp& response = *((cudaMallocArrayRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *array = response.array;
    }

    return response.result;
}
extern "C" cudaError_t cudaMallocHost(void** ptr, size_t size) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMallocHostRq& request = *(cudaMallocHostRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMallocHostType;
    request.size = size;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMallocHostRp& response = *((cudaMallocHostRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *ptr = response.ptr;
    }

    return response.result;
}
extern "C" cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMallocPitchRq& request = *(cudaMallocPitchRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMallocPitchType;
    request.width = width;
    request.height = height;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMallocPitchRp& response = *((cudaMallocPitchRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *devPtr = response.devPtr;
        *pitch = response.pitch;
    }

    return response.result;
}
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    void* client_dst = dst;
    void* ptr_dst;
    void* dst1;
    if (kind == cudaMemcpyHostToDevice) {

        static char* shared_mem;
        if (alloc == 0) {
            std::lock_guard<std::mutex> lock(alloc_mutex);
            shared_mem = initSharedMemory();
            global_shared_mem = shared_mem;
            alloc = 1;
        }

        std::lock_guard<std::mutex> lock(alloc_mutex);

        if (shared_mem_offset + count > (long)1024 * 1024 * 1024) {

            return cudaErrorUnknown;
        }

        void* ptr_src = global_shared_mem + shared_mem_offset;
        shared_mem_offset += count;

        memcpy(ptr_src, src, count);
        printf("Data copied to shared memory: %p\n", ptr_src);
        src = ptr_src;

    }

    if (kind == cudaMemcpyDeviceToHost &&
        !(dst >= global_shared_mem && dst < global_shared_mem + 1024*1024*1024)) {
        static char* shared_mem;
        if (alloc == 0) {
            std::lock_guard<std::mutex> lock(alloc_mutex);
            shared_mem = initSharedMemory();
            global_shared_mem = shared_mem;
            alloc = 1;
        }

        std::lock_guard<std::mutex> lock(alloc_mutex);

        if (shared_mem_offset + count > (long)1024 * 1024 * 1024) {

            return cudaErrorUnknown;
        }

        ptr_dst = global_shared_mem + shared_mem_offset;
        shared_mem_offset += count;
        dst = ptr_dst;
    }

    cudaMemcpyAsyncRq& request = *(cudaMemcpyAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyAsyncType;
    request.dst = dst;
    request.src = src;
    request.kind = kind;
    request.client_shared_mem = global_shared_mem;
    request.count = count;
    printf("client_shared_mem:%p\n", global_shared_mem);
    printf("client_data_src:%p\n", src);
    printf("client_data_dst:%p\n", dst);
    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    if (kind == cudaMemcpyDeviceToHost &&
        !(dst >= global_shared_mem && dst < global_shared_mem + 1024*1024*1024)) {
        memcpy(client_dst, dst, count);
    }
    printf("copy win with result: %s\n", cudaGetErrorString(response.result));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy2D(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy2DRq& request = *(cudaMemcpy2DRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy2DType;
    request.dst = dst;
    request.dpitch = dpitch;
    request.src = src;
    request.spitch = spitch;
    request.width = width;
    request.height = height;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy2DArrayToArray(
    struct cudaArray* dst,
    size_t wOffsetDst,
    size_t hOffsetDst,
    const struct cudaArray* src,
    size_t wOffsetSrc,
    size_t hOffsetSrc,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy2DArrayToArrayRq& request = *(cudaMemcpy2DArrayToArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy2DArrayToArrayType;
    request.dst = dst;
    request.wOffsetDst = wOffsetDst;
    request.hOffsetDst = hOffsetDst;
    request.src = src;
    request.wOffsetSrc = wOffsetSrc;
    request.hOffsetSrc = hOffsetSrc;
    request.width = width;
    request.height = height;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy2DAsync(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy2DAsyncRq& request = *(cudaMemcpy2DAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy2DAsyncType;
    request.dst = dst;
    request.dpitch = dpitch;
    request.src = src;
    request.spitch = spitch;
    request.width = width;
    request.height = height;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy2DFromArray(
    void* dst,
    size_t dpitch,
    const struct cudaArray* src,
    size_t wOffset,
    size_t hOffset,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy2DFromArrayRq& request = *(cudaMemcpy2DFromArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy2DFromArrayType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.dpitch = dpitch;
    request.width = width;
    request.height = height;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(
    void* dst,
    size_t dpitch,
    const struct cudaArray* src,
    size_t wOffset,
    size_t hOffset,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy2DFromArrayAsyncRq& request = *(cudaMemcpy2DFromArrayAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy2DFromArrayAsyncType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.dpitch = dpitch;
    request.width = width;
    request.height = height;
    request.kind = kind;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy2DToArray(
    struct cudaArray* dst,
    size_t wOffset,
    size_t hOffset,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy2DToArrayRq& request = *(cudaMemcpy2DToArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy2DToArrayType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.spitch = spitch;
    request.width = width;
    request.height = height;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy2DToArrayAsync(
    struct cudaArray* dst,
    size_t wOffset,
    size_t hOffset,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy2DToArrayAsyncRq& request = *(cudaMemcpy2DToArrayAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy2DToArrayAsyncType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.spitch = spitch;
    request.width = width;
    request.height = height;
    request.kind = kind;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy3DRq& request = *(cudaMemcpy3DRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy3DType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMemcpy3DRp& response = *((cudaMemcpy3DRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        p = response.p;
    }

    return response.result;
}
extern "C" cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpy3DAsyncRq& request = *(cudaMemcpy3DAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpy3DAsyncType;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMemcpy3DAsyncRp& response = *((cudaMemcpy3DAsyncRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        p = response.p;
    }

    return response.result;
}
extern "C" cudaError_t cudaMemcpyArrayToArray(
    struct cudaArray* dst,
    size_t wOffsetDst,
    size_t hOffsetDst,
    const struct cudaArray* src,
    size_t wOffsetSrc,
    size_t hOffsetSrc,
    size_t count,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyArrayToArrayRq& request = *(cudaMemcpyArrayToArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyArrayToArrayType;
    request.dst = dst;
    request.wOffsetDst = wOffsetDst;
    request.hOffsetDst = hOffsetDst;
    request.src = src;
    request.wOffsetSrc = wOffsetSrc;
    request.hOffsetSrc = hOffsetSrc;
    request.count = count;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
            return cudaErrorUnknown;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }
    void* client_dst = dst;
    void* ptr_dst;
    void* dst1;
    if (kind == cudaMemcpyHostToDevice) {

        static char* shared_mem;
        if (alloc == 0) {
            std::lock_guard<std::mutex> lock(alloc_mutex);
            shared_mem = initSharedMemory();
            global_shared_mem = shared_mem;
            alloc = 1;
        }

        std::lock_guard<std::mutex> lock(alloc_mutex);

        if (shared_mem_offset + count > (long)1024 * 1024 * 1024) {

            return cudaErrorUnknown;
        }

        void* ptr_src = global_shared_mem + shared_mem_offset;
        shared_mem_offset += count;

        memcpy(ptr_src, src, count);
        printf("Data copied to shared memory: %p\n", ptr_src);
        src = ptr_src;

    }

    bool ooshmem = (kind == cudaMemcpyDeviceToHost &&
        !(dst >= global_shared_mem && dst < global_shared_mem + 1024*1024*1024));
    if (ooshmem) {
        printf("cudaMemcpyAsync: out of page-locked shared memory, so it need a shmem buffer\n");
        static char* shared_mem;
        if (alloc == 0) {
            std::lock_guard<std::mutex> lock(alloc_mutex);
            shared_mem = initSharedMemory();
            global_shared_mem = shared_mem;
            alloc = 1;
        }

        std::lock_guard<std::mutex> lock(alloc_mutex);

        if (shared_mem_offset + count > (long)1024 * 1024 * 1024) {

            return cudaErrorUnknown;
        }

        ptr_dst = global_shared_mem + shared_mem_offset;
        shared_mem_offset += count;
        dst = ptr_dst;
    }

    cudaMemcpyAsyncRq& request = *(cudaMemcpyAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyAsyncType;
    request.dst = dst;
    request.src = src;
    request.kind = kind;
    request.stream = stream;
    request.client_shared_mem = global_shared_mem;
    request.count = count;
    printf("client_shared_mem:%p\n", global_shared_mem);
    printf("client_data_src:%p\n", src);
    printf("client_data_dst:%p\n", dst);
    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    if (ooshmem) {
        const float* host_dst = static_cast<const float*>(dst);
        printf("[Client Debug] D2H - First 5 floats to host: ");
        printf("%f ", host_dst[0]);
        printf("\n");
        memcpy(client_dst, dst, count);
    }
    printf("copy win with result: %s\n", cudaGetErrorString(response.result));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyFromArray(
    void* dst,
    const struct cudaArray* src,
    size_t wOffset,
    size_t hOffset,
    size_t count,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyFromArrayRq& request = *(cudaMemcpyFromArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyFromArrayType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.count = count;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyFromArrayAsync(
    void* dst,
    const struct cudaArray* src,
    size_t wOffset,
    size_t hOffset,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyFromArrayAsyncRq& request = *(cudaMemcpyFromArrayAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyFromArrayAsyncType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.count = count;
    request.kind = kind;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyFromSymbol(
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyFromSymbolRq& request = *(cudaMemcpyFromSymbolRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyFromSymbolType;
    request.dst = dst;
    request.symbol = symbol;
    request.offset = offset;
    request.kind = kind;
    request.count = count;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyFromSymbolAsync(
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyFromSymbolAsyncRq& request = *(cudaMemcpyFromSymbolAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyFromSymbolAsyncType;
    request.dst = dst;
    request.symbol = symbol;
    request.offset = offset;
    request.kind = kind;
    request.count = count;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyToArray(
    struct cudaArray* dst,
    size_t wOffset,
    size_t hOffset,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyToArrayRq& request = *(cudaMemcpyToArrayRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyToArrayType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.count = count;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyToArrayAsync(
    struct cudaArray* dst,
    size_t wOffset,
    size_t hOffset,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyToArrayAsyncRq& request = *(cudaMemcpyToArrayAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyToArrayAsyncType;
    request.dst = dst;
    request.wOffset = wOffset;
    request.hOffset = hOffset;
    request.src = src;
    request.count = count;
    request.kind = kind;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyToSymbol(
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyToSymbolRq& request = *(cudaMemcpyToSymbolRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyToSymbolType;
    request.symbol = symbol;
    request.src = src;
    request.count = count;
    request.offset = offset;
    request.kind = kind;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemcpyToSymbolAsync(
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemcpyToSymbolAsyncRq& request = *(cudaMemcpyToSymbolAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemcpyToSymbolAsyncType;
    request.symbol = symbol;
    request.src = src;
    request.count = count;
    request.offset = offset;
    request.kind = kind;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemGetInfoRq& request = *(cudaMemGetInfoRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemGetInfoType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    cudaMemGetInfoRp& response = *((cudaMemGetInfoRp*)(ResponseReceive(response_rb)));

    if (response.result == cudaSuccess) {
        *free = response.free;
        *total = response.total;
    }

    return response.result;
}
extern "C" cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemsetRq& request = *(cudaMemsetRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemsetType;
    request.devPtr = devPtr;
    request.value = value;
    request.count = count;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemset2DRq& request = *(cudaMemset2DRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemset2DType;
    request.devPtr = devPtr;
    request.pitch = pitch;
    request.value = value;
    request.width = width;
    request.height = height;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemset2DAsync(
    void* devPtr,
    size_t pitch,
    int value,
    size_t width,
    size_t height,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemset2DAsyncRq& request = *(cudaMemset2DAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemset2DAsyncType;
    request.devPtr = devPtr;
    request.pitch = pitch;
    request.value = value;
    request.width = width;
    request.height = height;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemset3DRq& request = *(cudaMemset3DRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemset3DType;
    request.pitchedDevPtr = pitchedDevPtr;
    request.value = value;
    request.extent = extent;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemset3DAsync(
    struct cudaPitchedPtr pitchedDevPtr,
    int value,
    struct cudaExtent extent,
    cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemset3DAsyncRq& request = *(cudaMemset3DAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemset3DAsyncType;
    request.pitchedDevPtr = pitchedDevPtr;
    request.value = value;
    request.extent = extent;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}
extern "C" cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return cudaErrorUnknown;
    }

    cudaMemsetAsyncRq& request = *(cudaMemsetAsyncRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cudaMemsetAsyncType;
    request.devPtr = devPtr;
    request.value = value;
    request.count = count;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return cudaErrorUnknown;
    }

    CudaResponseBase& response = *((CudaResponseBase*)ResponseReceive(response_rb));

    return response.result;
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtCreateRq& request = *(cublasLtCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtCreateType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtCreateRp& response = *((cublasLtCreateRp*)(ResponseReceive(response_rb)));

    *lighthandle = response.handle;
    return response.status;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtDestroyRq& request = *(cublasLtDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtDestroyType;
    request.lightHandle = lightHandle;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtDestroyRp& response = *((cublasLtDestroyRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int *value) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtGetPropertyRq& request = *(cublasLtGetPropertyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtGetPropertyType;
    request.type1 = type;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtGetPropertyRp& response = *((cublasLtGetPropertyRp*)(ResponseReceive(response_rb)));

    *value = response.value;
    return response.status;
}

const char* cublasLtGetStatusName(cublasStatus_t status) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return nullptr;
    }

    cublasLtGetStatusNameRq& request = *(cublasLtGetStatusNameRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtGetStatusNameType;
    request.status = status;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return nullptr;
    }

    cublasLtGetStatusNameRp& response = *((cublasLtGetStatusNameRp*)(ResponseReceive(response_rb)));

    return response.statusName;
}

const char* cublasLtGetStatusString(cublasStatus_t status) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return nullptr;
    }

    cublasLtGetStatusStringRq& request = *(cublasLtGetStatusStringRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtGetStatusStringType;
    request.status = status;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return nullptr;
    }

    cublasLtGetStatusStringRp& response = *((cublasLtGetStatusStringRp*)(ResponseReceive(response_rb)));

    return response.statusString;
}

cublasStatus_t cublasLtHeuristicsCacheGetCapacity(size_t* capacity) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtHeuristicsCacheGetCapacityRq& request = *(cublasLtHeuristicsCacheGetCapacityRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtHeuristicsCacheGetCapacityType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtHeuristicsCacheGetCapacityRp& response = *((cublasLtHeuristicsCacheGetCapacityRp*)(ResponseReceive(response_rb)));

    *capacity = response.capacity;
    return response.status;
}

cublasStatus_t cublasLtHeuristicsCacheSetCapacity(size_t capacity) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtHeuristicsCacheSetCapacityRq& request = *(cublasLtHeuristicsCacheSetCapacityRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtHeuristicsCacheSetCapacityType;
    request.capacity = capacity;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtHeuristicsCacheSetCapacityRp& response = *((cublasLtHeuristicsCacheSetCapacityRp*)(ResponseReceive(response_rb)));

    return response.status;
}

size_t cublasLtGetVersion(void) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return 0;
    }

    cublasLtGetVersionRq& request = *(cublasLtGetVersionRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtGetVersionType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return 0;
    }

    cublasLtGetVersionRp& response = *((cublasLtGetVersionRp*)(ResponseReceive(response_rb)));

    return response.version;
}

unsigned cublasLtDisableCpuInstructionsSetMask(unsigned mask) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return 0;
    }

    cublasLtDisableCpuInstructionsSetMaskRq& request = *(cublasLtDisableCpuInstructionsSetMaskRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtDisableCpuInstructionsSetMaskType;
    request.mask = mask;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return 0;
    }

    cublasLtDisableCpuInstructionsSetMaskRp& response = *((cublasLtDisableCpuInstructionsSetMaskRp*)(ResponseReceive(response_rb)));

    return response.status;
}

size_t cublasLtGetCudartVersion(void) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return 0;
    }

    cublasLtGetCudartVersionRq& request = *(cublasLtGetCudartVersionRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtGetCudartVersionType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return 0;
    }

    cublasLtGetCudartVersionRp& response = *((cublasLtGetCudartVersionRp*)(ResponseReceive(response_rb)));

    return response.version;
}

cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetCallbackRq& request = *(cublasLtLoggerSetCallbackRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtLoggerSetCallbackType;
    request.callback = callback;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetCallbackRp& response = *((cublasLtLoggerSetCallbackRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtLoggerSetFile(FILE* file) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetFileRq& request = *(cublasLtLoggerSetFileRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtLoggerSetFileType;
    request.file = file;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetFileRp& response = *((cublasLtLoggerSetFileRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtLoggerOpenFile(const char* logFile) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerOpenFileRq& request = *(cublasLtLoggerOpenFileRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtLoggerOpenFileType;
    request.logFile = logFile;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerOpenFileRp& response = *((cublasLtLoggerOpenFileRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtLoggerSetLevel(int level) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetLevelRq& request = *(cublasLtLoggerSetLevelRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtLoggerSetLevelType;
    request.level = level;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetLevelRp& response = *((cublasLtLoggerSetLevelRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtLoggerSetMask(int mask) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetMaskRq& request = *(cublasLtLoggerSetMaskRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtLoggerSetMaskType;
    request.mask = mask;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerSetMaskRp& response = *((cublasLtLoggerSetMaskRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtLoggerForceDisable() {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerForceDisableRq& request = *(cublasLtLoggerForceDisableRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtLoggerForceDisableType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtLoggerForceDisableRp& response = *((cublasLtLoggerForceDisableRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void *alpha,
    const void *A,
    cublasLtMatrixLayout_t Adesc,
    const void *B,
    cublasLtMatrixLayout_t Bdesc,
    const void *beta,
    const void *C,
    cublasLtMatrixLayout_t Cdesc,
    void *D,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo,
    void *workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulRq& request = *(cublasLtMatmulRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulType;
    request.lightHandle = lightHandle;
    request.computeDesc = computeDesc;
    request.alpha = *(float*)alpha;
    request.A = A;
    request.Adesc = Adesc;
    request.B = B;
    request.Bdesc = Bdesc;
    request.beta = *(float*)beta;
    request.C = C;
    request.Cdesc = Cdesc;
    request.D = D;
    request.Ddesc = Ddesc;
    request.algo = algo;
    request.workspace = workspace;
    request.workspaceSizeInBytes = workspaceSizeInBytes;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulRp& response = *((cublasLtMatmulRp*)(ResponseReceive(response_rb)));

    printf("cublasLtMatmul end\n");
    return response.status;
}

cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(
    const cublasLtMatmulAlgo_t *algo,
    cublasLtMatmulAlgoCapAttributes_t attr,
    void *buf,
    size_t sizeInBytes,
    size_t *sizeWritten) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoCapGetAttributeRq& request = *(cublasLtMatmulAlgoCapGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulAlgoCapGetAttributeType;
    request.algo = algo;
    request.attr = attr;
    request.buf = buf;
    request.sizeInBytes = sizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoCapGetAttributeRp& response = *((cublasLtMatmulAlgoCapGetAttributeRp*)(ResponseReceive(response_rb)));

    *sizeWritten = response.sizeWritten;
    return response.status;
}

/* cublasStatus_t cublasLtMatmulAlgoCheck(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo,
    cublasLtMatmulHeuristicResult_t *result) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoCheckRq& request = *(cublasLtMatmulAlgoCheckRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.lightHandle = lightHandle;
    request.operationDesc = operationDesc;
    request.Adesc = Adesc;
    request.Bdesc = Bdesc;
    request.Cdesc = Cdesc;
    request.Ddesc = Ddesc;
    request.algo = algo;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoCheckRp& response = *((cublasLtMatmulAlgoCheckRp*)(ResponseReceive(response_rb)));

    *result = response.result;
    return response.status;
} */

cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(
    const cublasLtMatmulAlgo_t *algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    void *buf,
    size_t sizeInBytes,
    size_t *sizeWritten) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoConfigGetAttributeRq& request = *(cublasLtMatmulAlgoConfigGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulAlgoConfigGetAttributeType;
    request.algo = algo;
    request.attr = attr;
    request.buf = buf;
    request.sizeInBytes = sizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoConfigGetAttributeRp& response = *((cublasLtMatmulAlgoConfigGetAttributeRp*)(ResponseReceive(response_rb)));

    *sizeWritten = response.sizeWritten;
    return response.status;
}

cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(
    cublasLtMatmulAlgo_t *algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    const void *buf,
    size_t sizeInBytes) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoConfigSetAttributeRq& request = *(cublasLtMatmulAlgoConfigSetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulAlgoConfigSetAttributeType;
    request.algo = algo;
    request.attr = attr;
    request.buf = buf;
    request.sizeInBytes = sizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoConfigSetAttributeRp& response = *((cublasLtMatmulAlgoConfigSetAttributeRp*)(ResponseReceive(response_rb)));

    return response.status;
}

/* cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference,
    int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoGetHeuristicRq& request = *(cublasLtMatmulAlgoGetHeuristicRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.lightHandle = lightHandle;
    request.operationDesc = operationDesc;
    request.Adesc = Adesc;
    request.Bdesc = Bdesc;
    request.Cdesc = Cdesc;
    request.Ddesc = Ddesc;
    request.preference = preference;
    request.requestedAlgoCount = requestedAlgoCount;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoGetHeuristicRp& response = *((cublasLtMatmulAlgoGetHeuristicRp*)(ResponseReceive(response_rb)));

    *returnAlgoCount = response.returnAlgoCount;
    std::copy(std::begin(response.heuristicResultsArray), std::end(response.heuristicResultsArray), heuristicResultsArray);

    return response.status;
} */

cublasStatus_t cublasLtMatmulAlgoGetIds(
    cublasLtHandle_t lightHandle,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType,
    cudaDataType_t Atype,
    cudaDataType_t Btype,
    cudaDataType_t Ctype,
    cudaDataType_t Dtype,
    int requestedAlgoCount,
    int algoIdsArray[],
    int *returnAlgoCount) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoGetIdsRq& request = *(cublasLtMatmulAlgoGetIdsRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulAlgoGetIdsType;
    request.lightHandle = lightHandle;
    request.computeType = computeType;
    request.scaleType = scaleType;
    request.Atype = Atype;
    request.Btype = Btype;
    request.Ctype = Ctype;
    request.Dtype = Dtype;
    request.requestedAlgoCount = requestedAlgoCount;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoGetIdsRp& response = *((cublasLtMatmulAlgoGetIdsRp*)(ResponseReceive(response_rb)));

    *returnAlgoCount = response.returnAlgoCount;
    std::copy(std::begin(response.algoIdsArray), std::end(response.algoIdsArray), algoIdsArray);

    return response.status;
}

cublasStatus_t cublasLtMatmulAlgoInit(
    cublasLtHandle_t lightHandle,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType,
    cudaDataType_t Atype,
    cudaDataType_t Btype,
    cudaDataType_t Ctype,
    cudaDataType_t Dtype,
    int algoId,
    cublasLtMatmulAlgo_t *algo) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoInitRq& request = *(cublasLtMatmulAlgoInitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulAlgoInitType;
    request.lightHandle = lightHandle;
    request.computeType = computeType;
    request.scaleType = scaleType;
    request.Atype = Atype;
    request.Btype = Btype;
    request.Ctype = Ctype;
    request.Dtype = Dtype;
    request.algoId = algoId;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulAlgoInitRp& response = *((cublasLtMatmulAlgoInitRp*)(ResponseReceive(response_rb)));

    *algo = response.algo;
    return response.status;
}

cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t *matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescCreateRq& request = *(cublasLtMatmulDescCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulDescCreateType;
    request.computeType = computeType;
    request.scaleType = scaleType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescCreateRp& response = *((cublasLtMatmulDescCreateRp*)(ResponseReceive(response_rb)));

    *matmulDesc = response.matmulDesc;

    return response.status;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescDestroyRq& request = *(cublasLtMatmulDescDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulDescDestroyType;
    request.matmulDesc = matmulDesc;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescDestroyRp& response = *((cublasLtMatmulDescDestroyRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    void *buf,
    size_t sizeInBytes,
    size_t *sizeWritten) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescGetAttributeRq& request = *(cublasLtMatmulDescGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulDescGetAttributeType;
    request.matmulDesc = matmulDesc;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescGetAttributeRp& response = *((cublasLtMatmulDescGetAttributeRp*)(ResponseReceive(response_rb)));

    std::memcpy(buf, response.buf, response.sizeWritten);
    *sizeWritten = response.sizeWritten;
    return response.status;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void *buf,
    size_t sizeInBytes) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescSetAttributeRq& request = *(cublasLtMatmulDescSetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulDescSetAttributeType;
    request.matmulDesc = matmulDesc;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;

    static char *shared_mem ;

    if (alloc == 0) {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    shared_mem = initSharedMemory();
    global_shared_mem = shared_mem;
    alloc = 1;
    }
    if (shared_mem_offset + sizeInBytes > (long)1024*1024*1024) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    void* ptr_buf = global_shared_mem + shared_mem_offset;

    shared_mem_offset += sizeInBytes;
    memcpy(ptr_buf, buf, sizeInBytes);
    request.buf = ptr_buf;
    request.client_shared_mem = global_shared_mem;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulDescSetAttributeRp& response = *((cublasLtMatmulDescSetAttributeRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceCreateRq& request = *(cublasLtMatmulPreferenceCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulPreferenceCreateType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceCreateRp& response = *((cublasLtMatmulPreferenceCreateRp*)(ResponseReceive(response_rb)));

    *pref = response.pref;
    return response.status;
}

/* cublasStatus_t cublasLtMatmulPreferenceInit(cublasLtMatmulPreference_t pref) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceInitRq& request = *(cublasLtMatmulPreferenceInitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.pref = pref;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceInitRp& response = *((cublasLtMatmulPreferenceInitRp*)(ResponseReceive(response_rb)));

    return response.status;
} */

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceDestroyRq& request = *(cublasLtMatmulPreferenceDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulPreferenceDestroyType;
    request.pref = pref;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceDestroyRp& response = *((cublasLtMatmulPreferenceDestroyRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    void *buf,
    size_t sizeInBytes,
    size_t *sizeWritten) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceGetAttributeRq& request = *(cublasLtMatmulPreferenceGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulPreferenceGetAttributeType;
    request.pref = pref;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceGetAttributeRp& response = *((cublasLtMatmulPreferenceGetAttributeRp*)(ResponseReceive(response_rb)));

    std::memcpy(buf, response.buf, response.sizeWritten);
    *sizeWritten = response.sizeWritten;
    return response.status;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    const void *buf,
    size_t sizeInBytes) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceSetAttributeRq& request = *(cublasLtMatmulPreferenceSetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatmulPreferenceSetAttributeType;
    request.pref = pref;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;
    static char *shared_mem ;

    if (alloc == 0) {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    shared_mem = initSharedMemory();
    global_shared_mem = shared_mem;
    alloc = 1;
    }
    if (shared_mem_offset + sizeInBytes > (long)1024*1024*1024) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    void* ptr_buf = global_shared_mem + shared_mem_offset;

    shared_mem_offset += sizeInBytes;
    memcpy(ptr_buf, buf, sizeInBytes);
    request.buf = ptr_buf;
    request.client_shared_mem = global_shared_mem;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatmulPreferenceSetAttributeRp& response = *((cublasLtMatmulPreferenceSetAttributeRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t *matLayout,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutCreateRq& request = *(cublasLtMatrixLayoutCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixLayoutCreateType;
    request.type1 = type;
    request.rows = rows;
    request.cols = cols;
    request.ld = ld;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutCreateRp& response = *((cublasLtMatrixLayoutCreateRp*)(ResponseReceive(response_rb)));

    *matLayout = response.matLayout;
    return response.status;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutDestroyRq& request = *(cublasLtMatrixLayoutDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixLayoutDestroyType;
    request.matLayout = matLayout;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutDestroyRp& response = *((cublasLtMatrixLayoutDestroyRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    void *buf,
    size_t sizeInBytes,
    size_t *sizeWritten) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutGetAttributeRq& request = *(cublasLtMatrixLayoutGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixLayoutGetAttributeType;
    request.matLayout = matLayout;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutGetAttributeRp& response = *((cublasLtMatrixLayoutGetAttributeRp*)(ResponseReceive(response_rb)));

    std::memcpy(buf, response.buf, response.sizeWritten);
    *sizeWritten = response.sizeWritten;
    return response.status;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void *buf,
    size_t sizeInBytes) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutSetAttributeRq& request = *(cublasLtMatrixLayoutSetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixLayoutSetAttributeType;
    request.matLayout = matLayout;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;
    std::memcpy(request.buf, buf, sizeInBytes);

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixLayoutSetAttributeRp& response = *((cublasLtMatrixLayoutSetAttributeRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatrixTransform(
    cublasLtHandle_t lightHandle,
    cublasLtMatrixTransformDesc_t transformDesc,
    const void *alpha,
    const void *A,
    cublasLtMatrixLayout_t Adesc,
    const void *beta,
    const void *B,
    cublasLtMatrixLayout_t Bdesc,
    void *C,
    cublasLtMatrixLayout_t Cdesc,
    cudaStream_t stream) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformRq& request = *(cublasLtMatrixTransformRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixTransformType;
    request.lightHandle = lightHandle;
    request.transformDesc = transformDesc;
    std::memcpy(request.alpha, alpha, sizeof(request.alpha));
    std::memcpy(request.beta, beta, sizeof(request.beta));
    request.A = A;
    request.B = B;
    request.C = C;
    request.stream = stream;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformRp& response = *((cublasLtMatrixTransformRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatrixTransformDescCreate(
    cublasLtMatrixTransformDesc_t *transformDesc,
    cudaDataType scaleType) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescCreateRq& request = *(cublasLtMatrixTransformDescCreateRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixTransformDescCreateType;
    request.transformDesc = transformDesc;
    request.scaleType = scaleType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescCreateRp& response = *((cublasLtMatrixTransformDescCreateRp*)(ResponseReceive(response_rb)));

    return response.status;
}

/* cublasStatus_t cublasLtMatrixTransformDescInit(
    cublasLtMatrixTransformDesc_t transformDesc,
    cudaDataType scaleType) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescInitRq& request = *(cublasLtMatrixTransformDescInitRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.transformDesc = transformDesc;
    request.scaleType = scaleType;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescInitRp& response = *((cublasLtMatrixTransformDescInitRp*)(ResponseReceive(response_rb)));

    return response.status;
} */

cublasStatus_t cublasLtMatrixTransformDescDestroy(
    cublasLtMatrixTransformDesc_t transformDesc) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescDestroyRq& request = *(cublasLtMatrixTransformDescDestroyRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixTransformDescDestroyType;
    request.transformDesc = transformDesc;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescDestroyRp& response = *((cublasLtMatrixTransformDescDestroyRp*)(ResponseReceive(response_rb)));

    return response.status;
}

cublasStatus_t cublasLtMatrixTransformDescGetAttribute(
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr,
    void *buf,
    size_t sizeInBytes,
    size_t *sizeWritten) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescGetAttributeRq& request = *(cublasLtMatrixTransformDescGetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixTransformDescGetAttributeType;
    request.transformDesc = transformDesc;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescGetAttributeRp& response = *((cublasLtMatrixTransformDescGetAttributeRp*)(ResponseReceive(response_rb)));

    if (sizeWritten) {
        *sizeWritten = response.sizeWritten;
    }
    std::memcpy(buf, response.buf, response.sizeWritten);

    return response.status;
}

cublasStatus_t cublasLtMatrixTransformDescSetAttribute(
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr,
    const void *buf,
    size_t sizeInBytes) {

    if (!useServer) {
        if (!connectToServer()) {
            std::cerr << "Failed to connect to proxy server." << std::endl;
        }
        useServer = true;
    }

    if (request_rb == NULL) {
        std::cerr << "No active connection to server." << std::endl;
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescSetAttributeRq& request = *(cublasLtMatrixTransformDescSetAttributeRq*)NewRequest(request_rb); request.requestSize = sizeof(request);
    request.type = cublasLtMatrixTransformDescSetAttributeType;
    request.transformDesc = transformDesc;
    request.attr = attr;
    request.sizeInBytes = sizeInBytes;
    std::memcpy(request.buf, buf, sizeInBytes);

    if (!RequestSend(request_rb, sizeof(request))) {
        perror("send");
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cublasLtMatrixTransformDescSetAttributeRp& response = *((cublasLtMatrixTransformDescSetAttributeRp*)(ResponseReceive(response_rb)));

    return response.status;
}