// Pinned Nanhu helper semantics, shared by the CUDA hook and host checks.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define NANHU_HD __host__ __device__
#else
#define NANHU_HD
#endif

namespace nanhu_device {
struct Port {
  bool read = false, write = false;
  uint64_t readIndex = 0, writeIndex = 0, data = 0, mask = 0;
};
struct Request {
  Port ram[4]{};
  bool flashEnable = false, sdRead = false, sdSet = false;
  uint32_t flashAddress = 0;
};
struct Outputs {
  uint64_t ram[4]{}, flash = 0;
};
struct Status {
  uint32_t code = 0, port = 0;
  uint64_t address = 0;
};

// Validate before any update. All reads see the previous memory snapshot;
// disabled read/flash ports hold their output. Disjoint writes merge in order.
NANHU_HD inline Status step(uint64_t *memory, uint64_t words,
                            const Request &request, Outputs &outputs) {
  if (request.sdRead || request.sdSet) return {3, 0, 0};
  for (unsigned i = 0; i < 4; ++i) {
    const auto &p = request.ram[i];
    if (p.read && p.readIndex >= words) return {1, i, p.readIndex};
    if (p.write && p.writeIndex >= words) return {1, i, p.writeIndex};
    for (unsigned j = 0; j < i; ++j) {
      const auto &q = request.ram[j];
      if (p.write && q.write && p.writeIndex == q.writeIndex && (p.mask & q.mask))
        return {2, i, p.writeIndex};
    }
  }
  for (unsigned i = 0; i < 4; ++i)
    if (request.ram[i].read) outputs.ram[i] = memory[request.ram[i].readIndex];
  if (request.flashEnable) {
    const uint32_t index = request.flashAddress / 8;
    outputs.flash = index == 0 ? 0x01f292930010029bULL :
                    index == 1 ? 0x0000000000028067ULL : 0;
  }
  for (const auto &p : request.ram)
    if (p.write) memory[p.writeIndex] = (memory[p.writeIndex] & ~p.mask) | (p.data & p.mask);
  return {};
}

// Symbol fragments are stored most-significant nibble first by Toucan.
struct Signal {
  uint32_t count = 0, slots[16]{};
  uint8_t widths[16]{};
  NANHU_HD uint64_t read(const uint8_t *registers) const {
    uint64_t result = 0;
    for (unsigned i = 0; i < count; ++i) result = (result << 4) | registers[slots[i]];
    return result;
  }
  NANHU_HD void write(uint8_t *registers, uint64_t value) const {
    for (unsigned i = count; i > 0; --i) {
      registers[slots[i - 1]] = value & 15;
      value >>= 4;
    }
  }
};
struct RamBindings {
  Signal readEnable, writeEnable, readIndex, writeIndex, writeData, writeMask, readData;
};
struct Bindings {
  RamBindings ram[4];
  Signal reset, uartInput, sdData, flashEnable, flashAddress, flashData, sdRead, sdSet;
};
struct Metadata {
  uint64_t cycle = 0;
  Status status{};
  Outputs outputs{};
};
} // namespace nanhu_device
#undef NANHU_HD
