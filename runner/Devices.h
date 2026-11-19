// Host devices for the fixed single-clock model. All reads use the old memory
// snapshot; responses become visible in the next root-cycle evaluation.
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace kmhv2 {
struct RamPort {
  bool tick = true, read = false, write = false;
  uint64_t readIndex = 0, writeIndex = 0, data = 0, mask = 0;
};

class Devices {
  uint64_t wordCount;
  std::unordered_map<uint64_t, uint64_t> words;
  void check(uint64_t index) const {
    if (index >= wordCount) throw std::out_of_range("RAM helper word index");
  }
public:
  std::array<uint64_t, 4> ramOutput{};
  uint64_t flashOutput = 0;

  explicit Devices(uint64_t wordCount) : wordCount(wordCount) {
    if (!wordCount) throw std::invalid_argument("RAM must contain words");
  }
  void load(const std::vector<uint8_t> &bytes) {
    if ((bytes.size() + 7) / 8 > wordCount) throw std::out_of_range("RAM image size");
    for (size_t i = 0; i < bytes.size(); ++i) {
      unsigned shift = unsigned(i % 8) * 8;
      uint64_t mask = uint64_t(255) << shift;
      words[i / 8] = (peek(i / 8) & ~mask) | (uint64_t(bytes[i]) << shift);
    }
  }
  uint64_t peek(uint64_t index) const {
    check(index);
    auto found = words.find(index);
    return found == words.end() ? 0 : found->second;
  }
  static uint64_t flash(uint32_t address) {
    // Default CPU_XIANGSHAN_KMHV2 boot ROM from difftest/common/flash.cpp.
    constexpr std::array<uint64_t, 3> boot{
      0x0010029b74446073ULL, 0x3003307302a29313ULL, 0x0002806701f29293ULL};
    uint32_t index = (address & ~uint32_t(7)) / 8;
    return index < boot.size() ? boot[index] : 0;
  }
  void step(const std::array<RamPort, 4> &ports, bool flashTick,
            bool flashEnable, uint32_t flashAddress) {
    std::unordered_map<uint64_t, uint64_t> masks;
    for (const auto &p : ports) {
      if (!p.tick) continue;
      if (p.read) check(p.readIndex);
      if (!p.write) continue;
      check(p.writeIndex);
      if (masks[p.writeIndex] & p.mask)
        throw std::runtime_error("Overlapping RAM helper writes in one root cycle");
      masks[p.writeIndex] |= p.mask;
    }
    // Capture every enabled read before applying any writes, including writes
    // from another helper instance. Disabled read ports retain their data.
    for (size_t i = 0; i < ports.size(); ++i)
      if (ports[i].tick && ports[i].read) ramOutput[i] = peek(ports[i].readIndex);
    if (flashTick && flashEnable) flashOutput = flash(flashAddress);
    for (const auto &p : ports)
      if (p.tick && p.write)
        words[p.writeIndex] = (peek(p.writeIndex) & ~p.mask) | (p.data & p.mask);
  }
};
} // namespace kmhv2
