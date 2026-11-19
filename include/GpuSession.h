// Compact observation binding; device inputs are initialized once and updated on GPU.
#pragma once
#include "NanhuGpu.h"
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace nanhu_device {
class Session {
  bool started = false;
  std::set<uint32_t> selected;
  std::vector<uint32_t> slots;
  std::vector<uint8_t> packet;
public:
  Metadata metadata;
  ~Session() { if (started) { try { release(); } catch (...) {} } }
  bool active() const { return started; }
  void observe(const std::vector<std::tuple<uint32_t, uint32_t>> &fragments) {
    if (started) throw std::runtime_error("Cannot extend running observation set");
    for (auto [slot, width] : fragments) { (void)width; selected.insert(slot); }
  }
  template<class Simulator>
  void start(Simulator &sim, const std::vector<uint8_t> &image,
             const std::vector<uint8_t> &registers, const std::string &ram,
             const std::string &flash, const std::string &sd) {
    auto signal = [&](const std::string &name, unsigned width) {
      const auto &fragments = sim.symbols.regDebugInfo.at(name);
      if (fragments.empty() || fragments.size() > 16) throw std::runtime_error("Invalid device binding " + name);
      Signal result; unsigned total = 0;
      for (auto [slot, bits] : fragments) {
        if (slot >= registers.size() || !bits || bits > 4) throw std::runtime_error("Invalid device fragment " + name);
        result.slots[result.count] = slot; result.widths[result.count++] = bits; total += bits;
      }
      if (total != width) throw std::runtime_error("Wrong device width " + name);
      observe(fragments);
      return result;
    };
    Bindings bindings;
    bindings.reset = signal("SimTop.reset", 1);
    bindings.uartInput = signal("SimTop.difftest_uart_in_ch", 8);
    bindings.sdData = signal(sd + "data", 32);
    bindings.sdRead = signal(sd + "ren", 1); bindings.sdSet = signal(sd + "setAddr", 1);
    bindings.flashEnable = signal(flash + "ren", 1);
    bindings.flashAddress = signal(flash + "addr", 16); bindings.flashData = signal(flash + "data", 64);
    // The existing resolved-MLIR audit proves the four enable aliases and root clocks.
    const Signal readEnable = signal(ram + "0.r_0_enable", 1);
    const Signal writeEnable = signal(ram + "0.w_0_enable", 1);
    for (unsigned i = 0; i < 4; ++i) {
      auto &p = bindings.ram[i]; const auto prefix = ram + std::to_string(i) + ".";
      p.readEnable = readEnable; p.writeEnable = writeEnable;
      p.readIndex = signal(prefix + "r_0_index", 64); p.readData = signal(prefix + "r_0_data", 64);
      p.writeIndex = signal(prefix + "w_0_index", 64); p.writeData = signal(prefix + "w_0_data", 64);
      p.writeMask = signal(prefix + "w_0_mask", 64);
    }
    slots.assign(selected.begin(), selected.end()); packet.resize(sizeof(Metadata) + slots.size());
    configure(bindings, image, registers, slots, 17179869184ULL / 8);
    started = true;
  }
  void pull(std::vector<uint8_t> &registers) {
    download(packet);
    std::memcpy(&metadata, packet.data(), sizeof(metadata));
    if (metadata.status.code) throw std::runtime_error("GPU device error " +
      std::to_string(metadata.status.code) + " port " + std::to_string(metadata.status.port) +
      " address " + std::to_string(metadata.status.address));
    for (size_t i = 0; i < slots.size(); ++i) registers[slots[i]] = packet[sizeof(Metadata) + i];
  }
  size_t observationBytes() const { return packet.size(); }
};
} // namespace nanhu_device
