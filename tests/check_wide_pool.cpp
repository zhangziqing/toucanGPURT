// Execute the runtime's packer and evaluator bodies with one active host lane.
#include "StageSchedule.h"
#include "WidePool.h"
#include <algorithm>
#include <array>
#include <sstream>

using namespace toucanGPUSim;

#define __host__
#define __device__
#define __forceinline__ inline
namespace cooperative_groups {
struct Group {
  unsigned thread_rank() const { return 0; }
  unsigned size() const { return 1; }
  void sync() const {}
};
Group this_thread_block() { return {}; }
template <unsigned N> Group tiled_partition(Group group) { return group; }
}
namespace cg = cooperative_groups;
uint8_t __shfl_sync(unsigned, uint8_t value, unsigned lane) {
  assert(lane % 32 == 0);
  return value;
}
unsigned atomicExch(unsigned *ptr, unsigned value) {
  auto before = *ptr;
  *ptr = value;
  return before;
}
uint8_t lutContent[4642];
uint8_t *regPool, *exchangePool, *memPool;
unsigned shouldStop;
bool enablePrint;
char **printMsgs;
#include "eval.inc"

int main() {
  static_assert(sizeof(CGMicroPartLUTMiddleLevelOp) == 4);
  constexpr unsigned edge = 65536, last = widePoolMaxBytes - 1;
  SimDesignInfo design{};
  design.regPoolSize = design.exchangePoolSize = widePoolMaxBytes;
  design.memPoolSize = 16;
  design.parts.resize(2);
  design.regionPartitionIds = {{0}, {1}};
  auto &first = design.parts[0];
  first.valuePoolSize = widePoolMaxBytes;
  first.ops_l0_regRead = {{1, edge - 2}, {2, edge - 1}, {3, edge}};
  first.op_last_exchangeWrite = {0, 16, edge + 16};
  CGMicroPartInfo lut{}, vec{}, add{}, mem{};
  lut.isLUTPart = true;
  lut.topLevel = {{0, 0, 0, edge}};
  lut.middleLevels = {{CGMicroPartLUTMiddleLevelOp(0, 0, 0, 32)}};
  lut.lastLevel = {{32, last}};
  vec.vecRead = {{edge - 2, 2, 0, false, 0, 0, 0, edge + 8, last, edge + 2}};
  add.vecArithAndLogic = {{edge - 2, edge + 2, 2, VEC_ARITH_ADD, 0, edge + 4}};
  mem.memRead = {{false, 0, edge + 8, edge + 16, edge + 6}};
  first.exec_mParts = {{lut}, {vec}, {add, mem}};
  auto &second = design.parts[1];
  second.valuePoolSize = widePoolMaxBytes;
  second.ops_l0_exchangeRead = {{edge, edge + 12}};
  second.op_last_regWrite = {0, 16, edge + 16};
  second.ops_last_memWrite = {{false, 0, edge + 16, edge + 12, edge + 8}};

  std::stringstream stream;
  serializeSimDesignInfo(stream, design);
  SimDesignInfo copy{};
  deserializeSimDesignInfo(stream, copy);
  assert(stream && stream.peek() == std::stringstream::traits_type::eof());
  std::stringstream roundtrip;
  serializeSimDesignInfo(roundtrip, copy);
  assert(roundtrip.str() == stream.str());
  assert(stagePartitionOffsets(copy) == std::vector<uint32_t>({0, 1, 2}));
  auto reject = [&] {
    try { (void)stagePartitionOffsets(copy); } catch (const std::invalid_argument &) { return; }
    assert(false);
  };
  ++copy.parts[0].valuePoolSize;
  reject();
  copy.parts[0].valuePoolSize = edge + 31;
  reject();
  copy.parts[0].valuePoolSize = widePoolMaxBytes;

  auto payload = stream.str();
  for (auto bad : {payload.substr(sizeof(widePoolMagic)), std::string("TCWIDE2\0", 8), std::string("TC")}) {
    std::istringstream input(bad);
    SimDesignInfo unused{};
    deserializeSimDesignInfo(input, unused);
    assert(!input && unused.parts.empty());
  }

  alignas(16) std::array<uint8_t, widePoolMaxBytes> pool{}, regs{}, exchange{};
  std::array<uint8_t, 16> memory{};
  regPool = regs.data(); exchangePool = exchange.data(); memPool = memory.data();
  regs[1] = 2; regs[2] = 3; regs[3] = 9;
  pool[edge + 8] = 1; memory[0] = 7;
  const auto &a = copy.parts[0];
  evalPartL0RegRead(pool.data(), a.ops_l0_regRead.data(), a.ops_l0_regRead.size());
  for (const auto &level : a.exec_mParts) for (const auto &op : level) {
    std::vector<char> netlist;
    copyMicroPartToNetlist(op, netlist);
    evalSingleMicroPart(pool.data(), nullptr, netlist.data());
  }
  assert(pool[last] == 9 && pool[edge + 2] == 3);
  assert(pool[edge + 4] == 5 && pool[edge + 5] == 3 && pool[edge + 6] == 7);
  assert(pool[0] == 0 && pool[4] == 0);
  pool[edge + 16] = 5;
  alignas(16) const auto exchangeWrite = a.op_last_exchangeWrite;
  evalLastLevelExgWrite(pool.data(), reinterpret_cast<const char *>(&exchangeWrite));
  assert(exchange[edge] == 5);
  const auto &b = copy.parts[1];
  evalPartL0ExgRead(pool.data(), b.ops_l0_exchangeRead.data(), b.ops_l0_exchangeRead.size());
  assert(pool[edge + 12] == 5);
  pool[edge + 16] = 0;
  struct alignas(16) Effects {
    CGRegWriteMetaInfo reg;
    alignas(16) CGMemWriteMetaInfo mem;
  } effects{b.op_last_regWrite, b.ops_last_memWrite[0]};
  evalLastLevel(pool.data(), reinterpret_cast<const char *>(&effects), 1, 1, 0, 0, 1);
  assert(regs[edge - 4] == 5 && memory[0] == 5);
}
