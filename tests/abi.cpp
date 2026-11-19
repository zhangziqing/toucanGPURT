#include "StageSchedule.h"
#include <iostream>
#include <sstream>
#include <type_traits>

using namespace toucanGPUSim;

int main(int argc, char **argv) {
  static_assert(sizeof(CGMicroPartLUTMiddleLevelOp) == 4);
#ifdef TOUCAN_WIDE_POOL
  static_assert(std::is_same_v<PoolOffset, uint32_t>);
  constexpr unsigned limit = widePoolMaxBytes;
#else
  static_assert(std::is_same_v<PoolOffset, uint16_t>);
  constexpr unsigned limit = 65534;
#endif
  if (argc == 2) {
    std::ifstream in(argv[1], std::ios::binary);
    SimDesignInfo loaded{};
    deserializeSimDesignInfo(in, loaded);
    return in ? 1 : 0; // A file from the other ABI must fail before allocation.
  }
  SimDesignInfo design{};
  design.regPoolSize = design.exchangePoolSize = limit;
  design.parts.resize(2);
  design.regionPartitionIds = {{0}, {1}};
  auto &first = design.parts[0];
  first.valuePoolSize = limit;
  first.op_last_exchangeWrite = {0, 0, 16};
  first.ops_l0_regRead = {{1, PoolOffset(limit - 1)}};
  auto &last = design.parts[1];
  last.valuePoolSize = limit;
  last.ops_l0_exchangeRead = {{0, PoolOffset(limit - 1)}};
  last.op_last_regWrite = {0, 0, 16};
  std::ostringstream out;
  serializeSimDesignInfo(out, design);
  const auto payload = out.str();
#ifdef TOUCAN_WIDE_POOL
  assert(payload.substr(0, 8) == std::string(widePoolMagic, 8));
#else
  assert(payload.substr(0, 6) != "TCWIDE");
#endif
  std::cout.write(payload.data(), payload.size());
  std::istringstream in(payload);
  SimDesignInfo copy{};
  deserializeSimDesignInfo(in, copy);
  assert(in && copy.parts[0].ops_l0_regRead[0].result == limit - 1);
  std::ostringstream again;
  serializeSimDesignInfo(again, copy);
  assert(again.str() == payload);
  assert(stagePartitionOffsets(copy) == std::vector<uint32_t>({0, 1, 2}));
  auto reject = [&] {
    try { (void)stagePartitionOffsets(copy); } catch (const std::invalid_argument &) { return; }
    assert(false);
  };
  copy.parts[0].valuePoolSize = limit + 1;
  reject();
  copy.parts[0].valuePoolSize = limit;
  copy.parts[0].op_last_exchangeWrite = {0, PoolOffset(limit - 8), 16};
  reject();
  copy.parts[0].op_last_exchangeWrite = {0, 0, 16};
  copy.parts[0].op_last_regWrite = {0, 0, 16};
  reject(); // State effects cannot precede the final stage.
  for (const auto &bad : {std::string("TCWIDE2\0", 8), std::string("TC")}) {
    std::istringstream truncated(bad);
    SimDesignInfo unused{};
    deserializeSimDesignInfo(truncated, unused);
    assert(!truncated && unused.parts.empty());
  }
}
