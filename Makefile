

# Compiler
NVCC = nvcc
ABI ?= legacy16
ARCH ?= sm_86
BUILD_DIR ?= build/$(ABI)-$(ARCH)
# CXX = g++
# Compiler Flags
# -DNDEBUG to disable asserts
# -rdc=true
NVCCFLAGS = --maxrregcount=64 -arch=$(ARCH) -Iinclude -O3 -std=c++20 -lineinfo
ifeq ($(ABI),wide32)
override NVCCFLAGS += -DTOUCAN_WIDE_POOL
else ifneq ($(ABI),legacy16)
$(error ABI must be legacy16 or wide32)
endif
# CXXFLAGS = -Wall -Wextra -Iinclude -O3 -std=c++20

# Source and Include directories
SRC_DIR = src
INCLUDE_DIR = include

# Output
OUT_DIR = $(BUILD_DIR)
LIBRARY = $(OUT_DIR)/libToucanGPURT.a
TEST_RUN_BIN = $(OUT_DIR)/test_run

# Source and Object files
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(OUT_DIR)/%.o) $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(OUT_DIR)/%.o)
HEADERS = $(wildcard $(INCLUDE_DIR)/*)

# Default target
all: $(LIBRARY) $(TEST_RUN_BIN)

# Rule to make the shared library
$(LIBRARY): $(OBJECTS)
	ar rcs $@ $(OBJECTS)

$(TEST_RUN_BIN): $(LIBRARY) main.cpp
	$(NVCC) $(NVCCFLAGS) main.cpp $(LIBRARY) -o $(TEST_RUN_BIN)

# Rule to make object files
$(OUT_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(LIBRARY) $(TEST_RUN_BIN)

# Phony targets
.PHONY: all clean nanhu

nanhu:
	python3 build_nanhu.py --abi $(ABI) --arch $(ARCH) --output $(BUILD_DIR) --nvcc $(NVCC)
