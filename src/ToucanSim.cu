#include "ToucanGPUSim.h"
#include "SimEval.h"
#include "StageSchedule.h"
#include <cstdint>
#include <iostream>
#include <cassert>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace toucanGPUSim;


// #define MAX_THREADS_PER_BLOCK_OVERRIDE 512

int ToucanSimulator::setupGPU(int gpu_id) {
  // Initialize CUDA
  cudaError_t cudaStatus = cudaSetDevice(gpu_id);  // Selects the first CUDA device
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
    return 1;
  }

  // Query device properties
  cudaDeviceProp prop;
  cudaStatus = cudaGetDeviceProperties(&prop, gpu_id);  // Get properties of device 0
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed!" << std::endl;
    return 1;
  }

  printf("Current GPU: %s\n", prop.name);

  // cuda capability >= 7.0
  if (prop.major < 7) {
    std::cerr << "Require CUDA capability >= 7.0\n";
    return 1;
  }

  if (!prop.cooperativeLaunch) {
    std::cerr << "GPU does not support cooperative launch!" << std::endl;
    return 1;
  }

  numSMs = prop.multiProcessorCount;
  maxThreadsPerBlock = prop.maxThreadsPerBlock;
  maxBlocksPerSMForSingleCycleKernel = 0;
  maxSharedMemoryPerSM = prop.sharedMemPerMultiprocessor;

#ifdef MAX_THREADS_PER_BLOCK_OVERRIDE
  maxThreadsPerBlock = MAX_THREADS_PER_BLOCK_OVERRIDE;
#endif

  // A cooperative grid must fit simultaneously with its actual dynamic shared
  // memory, static shared storage, registers and thread count for every kernel.
  const void *kernels[] = {reinterpret_cast<const void *>(evalSingleCycle),
                          reinterpret_cast<const void *>(evalFreeRunningNCycles),
                          reinterpret_cast<const void *>(evalFreeRunningNCycles_Large)};
  for (const void *kernel : kernels) {
    cudaFuncAttributes attrs{};
    cudaStatus = cudaFuncGetAttributes(&attrs, kernel);
    if (cudaStatus != cudaSuccess) {
      std::cerr << "Cannot inspect simulation kernel: " << cudaGetErrorString(cudaStatus) << "\n";
      return 1;
    }
    const size_t blockLimit = std::max(prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);
    if (sharedMemPerBlock + attrs.sharedSizeBytes > blockLimit ||
        sharedMemPerBlock + attrs.sharedSizeBytes > prop.sharedMemPerMultiprocessor) {
      std::cerr << "Partition requires " << sharedMemPerBlock << " dynamic plus "
                << attrs.sharedSizeBytes << " static shared bytes; device block limit is "
                << blockLimit << "\n";
      return 1;
    }
    maxThreadsPerBlock = std::min(maxThreadsPerBlock, attrs.maxThreadsPerBlock);
    cudaStatus = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sharedMemPerBlock));
    if (cudaStatus != cudaSuccess) {
      std::cerr << "Cannot configure simulation shared memory: "
                << cudaGetErrorString(cudaStatus) << "\n";
      return 1;
    }
  }
  bool fits = false;
  for (; maxThreadsPerBlock >= 64; maxThreadsPerBlock >>= 1) {
    int blocks[3] = {0, 0, 0};
    fits = true;
    for (int i = 0; i < 3; ++i) {
      cudaStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks[i], kernels[i], maxThreadsPerBlock, sharedMemPerBlock);
      fits &= cudaStatus == cudaSuccess && blocks[i] > 0;
    }
    if (fits) {
      maxBlocksPerSMForSingleCycleKernel = blocks[0];
      maxBlocksPerSMForMultiCycleKernel = std::min(blocks[1], blocks[2]);
      break;
    }
  }
  if (!fits) {
    std::cerr << "No cooperative grid fits the partition shared-memory requirement\n";
    return 1;
  }

  // Print device information
  std::cout << "Device Name: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
  std::cout << "Multiprocessors: " << numSMs << std::endl;
  std::cout << "Max Threads Per Block: " << maxThreadsPerBlock << std::endl;
  std::cout << "SingleCycle: Max Blocks Per SM: " << maxBlocksPerSMForSingleCycleKernel << std::endl;
  std::cout << "SingleCycle: Max Cooperative Groups: " << maxBlocksPerSMForSingleCycleKernel * numSMs << std::endl;
  std::cout << "MultiCycle: Max Blocks Per SM: " << maxBlocksPerSMForMultiCycleKernel << std::endl;
  std::cout << "MultiCycle: Max Cooperative Groups: " << maxBlocksPerSMForMultiCycleKernel * numSMs << std::endl;

  return 0;
}

int ToucanSimulator::init(const int gpu_id, const std::string designBinFilename, const std::string symbolBinFilename, bool shouldRandomize, bool enablePrint) {
  dumpVCD = false;
  vcdOfs = nullptr;
  cycle_cnt = 0;

  // load design
  std::ifstream ifs_design(designBinFilename);
  if (!ifs_design.is_open()) {
    std::cerr << "Error opening file: " << designBinFilename << std::endl;
    return -1;
  }
  toucanGPUSim::deserializeSimDesignInfo(ifs_design, design);
  if (!ifs_design) {
    std::cerr << "Cannot load wide32-v1 GPU design\n";
    return -1;
  }

  std::cout << "Load design complete" << std::endl;;

  // load symbol
  std::ifstream ifs_symbol(symbolBinFilename);
  if (!ifs_symbol.is_open()) {
    std::cerr << "Error opening file: " << symbolBinFilename << std::endl;
    return -1;
  }
  toucanGPUSim::deserializeSimDebugInfo(ifs_symbol, symbols);

  std::cout << "Load symbols complete" << std::endl;;


  design.Init();
  std::cout << "Init done" << std::endl;

  if (shouldRandomize) {
    design.Randomize(0, symbols);
    std::cout << "Randomize done" << std::endl;
  }


  // Get thread block count
  maxNumPartsInEachRegion = 0;
  int regionId = 0;
  for (const auto &eachRegionParts: design.regionPartitionIds) {
    maxNumPartsInEachRegion = std::max(maxNumPartsInEachRegion, static_cast<int>(eachRegionParts.size()));
    std::cout << "Region " << regionId << " has " << eachRegionParts.size() << " partitions\n";
    regionId++;
  }

  std::cout << "Design has " << maxNumPartsInEachRegion << " max parts\n";

  // Get max value pool size
  maxValuePoolSize = 0;
  for (const auto &eachPart: design.parts) {
    maxValuePoolSize = std::max(maxValuePoolSize, static_cast<size_t>(eachPart.valuePoolSize));
  }
  try {
    (void)stagePartitionOffsets(design);
  } catch (const std::invalid_argument &error) {
    std::cerr << "Invalid simulation stages: " << error.what() << "\n";
    return -1;
  }
  sharedMemPerBlock = maxValuePoolSize;
  if (setupGPU(gpu_id) != 0)
    return -1;

  numBlocksForSingleCycleKernel = std::min(maxNumPartsInEachRegion, maxBlocksPerSMForSingleCycleKernel * numSMs);
  numBlocksForMultiCycleKernel = std::min(maxNumPartsInEachRegion, maxBlocksPerSMForMultiCycleKernel * numSMs);
  std::cout << "Single cycle kernel use " << numBlocksForSingleCycleKernel << " thread blocks.\n";
  std::cout << "Multi cycle kernel use " << numBlocksForMultiCycleKernel << " thread blocks." << std::endl;


  copy_netlist_to_gpu(design);

  setEnablePrint(enablePrint);

  return 0;
}

bool ToucanSimulator::eval() {
  if (dumpVCD) {
    std::cout << "VCD dump is not supported yet!\n";
    // dumpVcdWorker(cycle_cnt);
  }

  size_t numBlocks = numBlocksForSingleCycleKernel;
  size_t threadsPerBlock = maxThreadsPerBlock;

  void *kernelArgs[] = {nullptr};
  gpuErrchk(cudaLaunchCooperativeKernel(
    (void*)evalSingleCycle,
    numBlocks,
    threadsPerBlock,
    kernelArgs,
    sharedMemPerBlock
  ));
  
  gpuErrchk(cudaDeviceSynchronize());

  auto ret = get_eval_done();

  cycle_cnt++;
  return ret;
}

bool ToucanSimulator::eval_free_running(uint32_t max_cycles) {
  if (dumpVCD) {
    std::cout << "VCD dump is not supported yet!\n";
    // dumpVcdWorker(cycle_cnt);
  }

  size_t numBlocks = numBlocksForMultiCycleKernel;
  size_t threadsPerBlock = maxThreadsPerBlock;
  void *kernelArgs[] = {&max_cycles};

  if (maxNumPartsInEachRegion > numBlocks) {
    gpuErrchk(cudaLaunchCooperativeKernel(
      (void*)evalFreeRunningNCycles_Large,
      numBlocks,
      threadsPerBlock,
      kernelArgs,
      sharedMemPerBlock
    ));
  } else {
    gpuErrchk(cudaLaunchCooperativeKernel(
      (void*)evalFreeRunningNCycles,
      numBlocks,
      threadsPerBlock,
      kernelArgs,
      sharedMemPerBlock
    ));
  }


  gpuErrchk(cudaDeviceSynchronize());

  auto ret = get_eval_done();

  cycle_cnt += max_cycles;
  return ret;
}



void ToucanSimulator::setRegister(const std::string &signalName, uint64_t signalValue) {
  assert(symbols.regDebugInfo.contains(signalName));
  auto &signalLocs = symbols.regDebugInfo[signalName];
  assert(signalLocs.size() <= 16 && "Register wider than 64 bits are not supported by this function (value cannot fit into uint64_t)");


  write_reg_to_gpu(signalLocs, signalValue);
}

uint64_t ToucanSimulator::readRegister(const std::string &signalName) {
  assert(symbols.regDebugInfo.contains(signalName));
  auto &signalLocs = symbols.regDebugInfo[signalName];

  assert(signalLocs.size() <= 16 && "Register wider than 64 bits are not supported by this function (value cannot fit into uint64_t)");

  auto result = read_reg_from_gpu(signalLocs);

  return result;
}

uint64_t ToucanSimulator::readSignal(const std::string &signalName) {
  std::cerr << "Not implemented! This might not be necessary" << std::endl;
  return UINT64_MAX;
  // assert(symbols.signalDebugInfo.contains(signalName));
  // auto &signalLocs = symbols.signalDebugInfo[signalName];

  // assert(signalLocs.size() <= 16 && "Signal wider than 64 bits are not supported by this function (value cannot fit into uint64_t)");

  // return ToucanSimulator::getSmallSignalValue(signalLocs);
}

void ToucanSimulator::cleanup() {
  // TODO: release resources
}

#ifdef ENABLE_SIM_PROFILE
void ToucanSimulator::printProfile() {
  std::vector<int64_t> totalTicks, usefulTicks;

  int numTotalPartitions = design.parts.size();

  totalTicks.resize(numTotalPartitions);
  usefulTicks.resize(numTotalPartitions);

  copy_profile_data(usefulTicks.data(), totalTicks.data(), numTotalPartitions);

  for (int partId = 0; partId < numTotalPartitions; partId++) {
    auto time_on_work = usefulTicks[partId];
    auto time_per_cycle = totalTicks[partId];

    float useful_ratio = time_on_work * 100.0 / time_per_cycle;

    std::cout << "Part " << partId << ", useful " << useful_ratio << "%, ticks " << time_on_work << ", total ticks " << time_per_cycle << "\n";
  }
}
#endif
