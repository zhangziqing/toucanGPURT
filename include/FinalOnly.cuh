// Included after NanhuGpu.cuh; all callers synchronize before the next cycle.
#pragma once
#include "FinalOnly.h"

namespace nanhu_final {
__device__ Bindings bindings;
__device__ Summary progress;
__device__ uint64_t targetCycles;

__device__ void finishCycle(uint8_t *registers) {
  auto *state = nanhu_device::activeState;
  if (!state) return;
  auto &metadata = *reinterpret_cast<nanhu_device::Metadata *>(state->packet);
  if (metadata.status.code) return; // Keep the first device failure until download.
  const uint64_t cycle = progress.cycles;
  if (bindings.trap.read(registers) && progress.trapCycle == UINT64_MAX)
    progress.trapCycle = cycle;
  progress.errors |= bindings.exception.read(registers) | bindings.interrupt.read(registers);
  for (unsigned lane = 0; lane < 6; ++lane) if (bindings.enable[lane].read(registers)) {
    const uint64_t fused = bindings.fused[lane].read(registers);
    if (fused > 1) progress.errors |= uint64_t(1) << 63;
    if (!progress.records) progress.firstCommit = cycle;
    ++progress.records; progress.retired += 1 + fused;
  }
  for (const auto &p : state->bindings.ram) {
    progress.ramReads += bool(p.readEnable.read(registers));
    progress.ramWrites += bool(p.writeEnable.read(registers));
  }
  progress.flashReads += bool(state->bindings.flashEnable.read(registers));
  nanhu_device::finishCycle(registers, cycle + 1 == targetCycles);
  ++progress.cycles;
}
void configure(const Bindings &value, uint64_t cycles) {
  if (!cycles) throw std::runtime_error("Empty final-only execution");
  Summary initial;
  nanhu_device::checked(cudaMemcpyToSymbol(bindings, &value, sizeof(value)));
  nanhu_device::checked(cudaMemcpyToSymbol(progress, &initial, sizeof(initial)));
  nanhu_device::checked(cudaMemcpyToSymbol(targetCycles, &cycles, sizeof(cycles)));
}
void download(Summary &value) {
  nanhu_device::checked(cudaMemcpyFromSymbol(&value, progress, sizeof(value)));
}
} // namespace nanhu_final
