#include "WidePool.h"
#pragma once

#include "ToucanGPUGenDataTypes.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace toucanGPUSim {

// Stages are consecutive ranges in the existing serialized partition-ID table.
// All state effects run in the final stage, after combinational exchange writes.
inline std::vector<uint32_t> stagePartitionOffsets(const SimDesignInfo &design) {
  if (design.regionPartitionIds.empty() || design.parts.empty())
    throw std::invalid_argument("A simulation requires stages and partitions");
  if (design.parts.size() > std::numeric_limits<uint32_t>::max() ||
      design.regionPartitionIds.size() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Stage or partition count exceeds uint32_t");

  std::vector<uint32_t> offsets{0};
  size_t nextPart = 0;
  for (size_t stage = 0; stage < design.regionPartitionIds.size(); ++stage) {
    const bool finalStage = stage + 1 == design.regionPartitionIds.size();
    for (uint32_t partId : design.regionPartitionIds[stage]) {
      if (partId != nextPart || nextPart >= design.parts.size())
        throw std::invalid_argument("Stage partition IDs must be consecutive and unique");
      const auto &part = design.parts[nextPart++];
#ifdef TOUCAN_WIDE_POOL
      if (part.valuePoolSize > widePoolMaxBytes)
        throw std::invalid_argument("Partition exceeds wide pool maximum");
#else
      if (part.valuePoolSize >= std::numeric_limits<uint16_t>::max())
        throw std::invalid_argument("Partition value pool must be below 65535 bytes");
#endif
      const auto reg = part.op_last_regWrite;
      const auto exchange = part.op_last_exchangeWrite;
      if ((reg.count && uint64_t(reg.dat) + reg.count > part.valuePoolSize) ||
          (exchange.count && uint64_t(exchange.dat) + exchange.count > part.valuePoolSize))
        throw std::invalid_argument("Padded output copy exceeds partition value pool");
      if (uint64_t(reg.reg) + reg.count > design.regPoolSize ||
          uint64_t(exchange.exchange) + exchange.count > design.exchangePoolSize)
        throw std::invalid_argument("Padded output copy exceeds global pool");
      const bool effects = part.op_last_regWrite.count || !part.ops_last_memWrite.empty() ||
                           !part.ops_last_print.empty() || !part.ops_last_stop.empty();
      if (!finalStage && effects)
        throw std::invalid_argument("State effects are only allowed in the final stage");
      if (finalStage && design.regionPartitionIds.size() > 2 && !part.ops_l0_regRead.empty())
        throw std::invalid_argument("Staged designs must snapshot registers before the final stage");
      if (finalStage && part.op_last_exchangeWrite.count)
        throw std::invalid_argument("Final stage cannot write exchange outputs");
    }
    offsets.push_back(static_cast<uint32_t>(nextPart));
  }
  if (nextPart != design.parts.size())
    throw std::invalid_argument("Stage partition IDs do not cover all partitions");
  return offsets;
}

} // namespace toucanGPUSim
