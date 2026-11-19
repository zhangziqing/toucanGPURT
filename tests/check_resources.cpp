// Compile and exercise the patched setupGPU body using CUDA API test doubles.
// This verifies host resource decisions, not driver or GPU execution behavior.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>

using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaFuncAttributeMaxDynamicSharedMemorySize = 1;
struct cudaDeviceProp {
  const char *name = "resource test double";
  int major = 8, minor = 0, cooperativeLaunch = 1;
  int multiProcessorCount = 12, maxThreadsPerBlock = 1024;
  size_t sharedMemPerMultiprocessor = 102400, sharedMemPerBlock = 49152;
  size_t sharedMemPerBlockOptin = 98304, totalGlobalMem = 16ULL << 30;
};
struct cudaFuncAttributes { size_t sharedSizeBytes = 1024; int maxThreadsPerBlock = 1024; };
cudaDeviceProp configured;
cudaFuncAttributes attrs;
int occupancyCalls, attributeCalls, failKernel = -1;
bool useThreadLimit, rejectAttribute;
void evalSingleCycle() {}
void evalFreeRunningNCycles() {}
void evalFreeRunningNCycles_Large() {}
int kernelId(const void *kernel) {
  if (kernel == reinterpret_cast<const void *>(evalSingleCycle)) return 0;
  if (kernel == reinterpret_cast<const void *>(evalFreeRunningNCycles)) return 1;
  assert(kernel == reinterpret_cast<const void *>(evalFreeRunningNCycles_Large));
  return 2;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int) { *p = configured; return cudaSuccess; }
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *p, const void *) { *p = attrs; return cudaSuccess; }
cudaError_t cudaFuncSetAttribute(const void *, int, int value) {
  ++attributeCalls;
  assert(value >= 0);
  return rejectAttribute ? 1 : cudaSuccess;
}
const char *cudaGetErrorString(cudaError_t) { return "test failure"; }
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, const void *kernel,
                                                         int threads, size_t dynamic) {
  ++occupancyCalls;
  assert(dynamic != 0); // Occupancy must use the actual value-pool allocation.
  *blocks = std::min<int>(8, configured.sharedMemPerMultiprocessor / (dynamic + attrs.sharedSizeBytes));
  if ((useThreadLimit && threads > 256) || kernelId(kernel) == failKernel) *blocks = 0;
  return cudaSuccess;
}
class ToucanSimulator {
public:
  int maxBlocksPerSMForSingleCycleKernel{}, maxBlocksPerSMForMultiCycleKernel{};
  int maxThreadsPerBlock{}, numSMs{}, maxSharedMemoryPerSM{};
  size_t sharedMemPerBlock{40000};
  int setupGPU(int gpu);
};

#include "setup.inc"

int main() {
  ToucanSimulator sim;
  useThreadLimit = true;
  assert(sim.setupGPU(0) == 0);
  assert(occupancyCalls == 9 && attributeCalls == 3);
  assert(sim.maxThreadsPerBlock == 256);
  assert(sim.maxBlocksPerSMForSingleCycleKernel == 2);
  assert(sim.maxBlocksPerSMForMultiCycleKernel == 2);

  occupancyCalls = attributeCalls = 0;
  configured.sharedMemPerBlockOptin = 65536;
  sim.sharedMemPerBlock = 65534;
  assert(sim.setupGPU(0) != 0); // Static shared memory also consumes the block budget.
  assert(!occupancyCalls && !attributeCalls);

  configured.sharedMemPerBlockOptin = 98304;
  useThreadLimit = false;
#ifdef CHECK_WIDE_POOL
  sim.sharedMemPerBlock = 72560;
  assert(sim.setupGPU(0) == 0);
  assert(sim.maxBlocksPerSMForMultiCycleKernel == 1);
  configured.sharedMemPerBlockOptin = 65536;
  assert(sim.setupGPU(0) != 0);
  configured.sharedMemPerBlockOptin = 98304;
#endif
  sim.sharedMemPerBlock = 60000;
  assert(sim.setupGPU(0) == 0);
  assert(sim.maxBlocksPerSMForSingleCycleKernel == 1);
  failKernel = 2;
  assert(sim.setupGPU(0) != 0); // The dynamically scheduled kernel is checked too.
  failKernel = -1;
  rejectAttribute = true;
  assert(sim.setupGPU(0) != 0);
  rejectAttribute = false;
  configured.cooperativeLaunch = false;
  assert(sim.setupGPU(0) != 0);
  std::cout << "cooperative launch resource checks passed\n";
}
