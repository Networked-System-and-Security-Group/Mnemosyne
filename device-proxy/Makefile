# Compiler and tools
CXX := g++
RM := rm -f

CURRENT_DIR := $(abspath .)
FUNCINFO := $(CURRENT_DIR)/cubin/funcinfo.txt

CUDA_HOME ?= /usr/local/cuda-12.4
CUDA_INC := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64

CXXFLAGS := -I$(CUDA_INC) -Wno-deprecated-declarations -fPIC -O3 -DFUNCINFO="\"$(FUNCINFO)\""
SERVER_LDFLAGS := -L$(CUDA_LIB) -lcudart -lcuda -lcublas -lcublasLt -lnccl -ldl
CLIENT_LDFLAGS := -L$(CUDA_LIB) -lcudart -lcublas -lnccl -ldl

TARGETS := server_proxy libclient.so

.PHONY: all build run single clean debug transformer

all: build run

build: $(TARGETS)

# Server target
server_proxy: server.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(SERVER_LDFLAGS)

# Client shared library
libclient.so: client.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $< $(CLIENT_LDFLAGS)

encoder: server_proxy libclient.so
	LD_PRELOAD=./libclient.so python3 encoder.py

decoder: server_proxy libclient.so
	LD_PRELOAD=./libclient.so python3 decoder.py

encdec: server_proxy libclient.so
	LD_PRELOAD=./libclient.so python3 encdec.py

# Cleanup
clean:
	$(RM) $(TARGETS)