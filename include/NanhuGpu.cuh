// Included once in SimEval.cu; the existing final-stage barrier precedes this hook.
#pragma once
#include "NanhuGpu.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <stdexcept>

namespace nanhu_device {
struct State {
  Bindings bindings;
  uint64_t *memory = nullptr;
  uint32_t *observed = nullptr;
  uint8_t *packet = nullptr;
  uint64_t words = 0, cycle = 0;
  uint32_t count = 0;
};
static State hostState;
static State *stateDevice = nullptr;
__device__ State *activeState = nullptr;

__device__ void finishCycle(uint8_t *registers, bool observe = true) {
  State *state = activeState;
  if (!state) return;
  auto &metadata = *reinterpret_cast<Metadata *>(state->packet);
  metadata.cycle = state->cycle;
  // Observe the just-evaluated circuit before publishing next-cycle responses.
  auto *values = state->packet + sizeof(Metadata);
  if (observe) for (unsigned i = 0; i < state->count; ++i) values[i] = registers[state->observed[i]];
  const auto &b = state->bindings;
  Request request;
  for (unsigned i = 0; i < 4; ++i) {
    const auto &p = b.ram[i];
    request.ram[i] = {bool(p.readEnable.read(registers)), bool(p.writeEnable.read(registers)),
                     p.readIndex.read(registers), p.writeIndex.read(registers),
                     p.writeData.read(registers), p.writeMask.read(registers)};
  }
  request.flashEnable = b.flashEnable.read(registers);
  request.flashAddress = b.flashAddress.read(registers);
  request.sdRead = b.sdRead.read(registers); request.sdSet = b.sdSet.read(registers);
  metadata.status = step(state->memory, state->words, request, metadata.outputs);
  if (metadata.status.code) return;
  for (unsigned i = 0; i < 4; ++i) b.ram[i].readData.write(registers, metadata.outputs.ram[i]);
  b.flashData.write(registers, metadata.outputs.flash);
  ++state->cycle;
  b.reset.write(registers, state->cycle < 64);
}

static void checked(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void release() {
  State *none = nullptr;
  checked(cudaMemcpyToSymbol(activeState, &none, sizeof(none)));
  if (stateDevice) checked(cudaFree(stateDevice));
  if (hostState.packet) checked(cudaFree(hostState.packet));
  if (hostState.observed) checked(cudaFree(hostState.observed));
  if (hostState.memory) checked(cudaFree(hostState.memory));
  hostState = {}; stateDevice = nullptr;
}
void configure(const Bindings &bindings, const std::vector<uint8_t> &image,
               const std::vector<uint8_t> &registers, const std::vector<uint32_t> &observed,
               uint64_t words) {
  if (stateDevice) throw std::runtime_error("Nanhu devices already configured");
  if (!words || words > SIZE_MAX / 8 || (image.size() + 7) / 8 > words ||
      observed.empty() || observed.size() > UINT32_MAX)
    throw std::runtime_error("Invalid Nanhu device dimensions");
  for (auto slot : observed)
    if (slot >= registers.size()) throw std::runtime_error("Nanhu observation outside register pool");
  size_t available = 0, total = 0;
  checked(cudaMemGetInfo(&available, &total));
  if (words * 8 > available) throw std::runtime_error("Insufficient free GPU RAM for Nanhu devices");
  hostState.bindings = bindings; hostState.words = words; hostState.count = observed.size();
  try {
    checked(cudaMalloc(reinterpret_cast<void **>(&hostState.memory), words * 8));
    checked(cudaMemset(hostState.memory, 0, words * 8));
    checked(cudaMemcpy(hostState.memory, image.data(), image.size(), cudaMemcpyHostToDevice));
    checked(cudaMalloc(reinterpret_cast<void **>(&hostState.observed), observed.size() * sizeof(uint32_t)));
    checked(cudaMemcpy(hostState.observed, observed.data(), observed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    const size_t packetBytes = sizeof(Metadata) + observed.size();
    checked(cudaMalloc(reinterpret_cast<void **>(&hostState.packet), packetBytes));
    checked(cudaMemset(hostState.packet, 0, packetBytes));
    auto initial = registers;
    bindings.reset.write(initial.data(), 1); bindings.uartInput.write(initial.data(), 0);
    bindings.sdData.write(initial.data(), 0); bindings.flashData.write(initial.data(), 0);
    for (const auto &p : bindings.ram) p.readData.write(initial.data(), 0);
    checked(cudaMemcpy(regPool_device, initial.data(), initial.size(), cudaMemcpyHostToDevice));
    checked(cudaMalloc(reinterpret_cast<void **>(&stateDevice), sizeof(State)));
    checked(cudaMemcpy(stateDevice, &hostState, sizeof(State), cudaMemcpyHostToDevice));
    checked(cudaMemcpyToSymbol(activeState, &stateDevice, sizeof(stateDevice)));
  } catch (...) { release(); throw; }
}
void download(std::vector<uint8_t> &packet) {
  if (!stateDevice || packet.size() != sizeof(Metadata) + hostState.count)
    throw std::runtime_error("Invalid Nanhu observation packet");
  checked(cudaMemcpy(packet.data(), hostState.packet, packet.size(), cudaMemcpyDeviceToHost));
}
} // namespace nanhu_device
