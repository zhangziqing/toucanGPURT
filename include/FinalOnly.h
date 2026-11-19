// GPU progress counters for fixed-length, final-observation-only execution.
#pragma once
#include "NanhuGpu.h"

namespace nanhu_final {
struct Bindings {
  nanhu_device::Signal trap, exception, interrupt, enable[6], fused[6];
};
struct Summary {
  uint64_t cycles = 0, records = 0, retired = 0;
  uint64_t firstCommit = UINT64_MAX, trapCycle = UINT64_MAX, errors = 0;
  uint64_t ramReads = 0, ramWrites = 0, flashReads = 0;
};
void configure(const Bindings &bindings, uint64_t cycles);
void download(Summary &summary);
} // namespace nanhu_final
