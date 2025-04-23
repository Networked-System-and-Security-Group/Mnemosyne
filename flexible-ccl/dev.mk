# dev.mk - Development Makefile for Test cases intergrated with NCCL.
#
# Usage:
#   - To build the NCCL shared library:
#     run `make -f dev.mk -j` or `make -f dev.mk -j lib`.
#   - To build individual test cases:
#     run `make -f dev.mk -j build/test/<target>`;
#     the C++ source files should be placed in `./test/` folder.
#   - To clean test build files:
#     run `make -f dev.mk -j clean`.
#   - To clean all build artifacts:
#     run `make -f dev.mk -j clean-all`.
#
#   You can also add extra arguments at the end of each command,
#   e.g. `make -f dev.mk -j NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"`.
#
#

CXX = nvcc
CPPFLAGS = -Ibuild/include \
           -I/usr/lib/x86_64-linux-gnu/openmpi/include \
           -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi
CXXFLAGS = -std=c++17 -g -Xcompiler -pthread
LDFLAGS = -Lbuild/lib -lnccl \
          -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi
LIB_SRC = $(shell find src -type f)

.PHONY: lib clean clean-all

lib: build/lib/libnccl.so

build/lib/libnccl.so: $(LIB_SRC)
	$(MAKE) -C src lib DEBUG=1 $(MAKEFLAGS)

clean:
	rm -rf build/test/*

clean-all:
	rm -rf build

build/test/%: test/%.cc build/lib/libnccl.so
	mkdir -p build/test
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)
