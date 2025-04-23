# Server Proxy Demo

Our Server Proxy achieves the separation of PyTorch programs from the CUDA context on a single GPU and provides a demo program for training Transformer models.
This README will detail the deployment process of the Server Proxy and the execution method of the demo program.

## Prerequisites

Before starting the deployment and execution, please ensure the following conditions are met:

* Python is installed, and the PyTorch and NumPy libraries are configured.
* The nvidia-toolkit is installed, ensuring the `cuobjdump` tool is available.

## Server Proxy Deployment

### Dumping PyTorch CUDA Kernel

Since PyTorch CUDA Kernel needs to be executed in the server, it is necessary to dump all CUDA Kernels from the PyTorch dynamic link library `libtorch_cuda.so` for the server to read and use.

* For PyTorch installed directly via pip, the library file is usually located at `/path/to/python/site-packages/torch/lib/libtorch_cuda.so`
* For PyTorch installed via conda, the library file is usually located at `/path/to/conda/envs/<env_name>/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so`
* For PyTorch installed via venv, the library file is usually located at `/path/to/venv/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so`

Use the following command to enter the `cubin` directory and dump the CUDA binary file of the PyTorch dynamic link library:

```shell
cd cubin
cuobjdump -xelf all <PATH_TO_LIBTORCH_CUDA_SO>
```

### Parsing CUDA Kernel Metadata

Refer to the [NVIDIA official documentation](https://developer.nvidia.com/cuda-gpus) to obtain the GPU's compute capability.
Execute the following command in the `cubin` directory to parse the CUDA Kernel metadata for the server to read and use:

```shell
python cubinFunc.py --sm_version sm_XX > funcinfo.txt
```

### Compiling Server Proxy and Client Interceptor Library

Use the `make` command to compile, and you can specify the CUDA installation path via the `CUDA_HOME` variable:

```shell
make build CUDA_HOME=<CUDA_HOME>
```

## Demo Example Execution

By adding `LD_PRELOAD=libclient.so` before the Python execution command, we can intercept CUDA-related calls in PyTorch programs. To simplify the testing process, we provide pre-packaged make commands for functional testing, as follows:

```shell
make encoder # for encoder-only model
make decoder # for decoder-only model
make encdec # for encoder-decoder model
```