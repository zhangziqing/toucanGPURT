#include "NanhuDeviceModel.h"
#include "Devices.h"
#include <array>
#include <cassert>

int main() {
  using namespace nanhu_device;
  std::array<uint64_t, 4> memory{0x11223344, 0x55667788, 0, 0};
  Outputs outputs{};
  Request request{};
  request.ram[0] = {true, true, 0, 0, 0xaa, 0xff};
  request.ram[1] = {true, true, 0, 0, 0xbb00, 0xff00};
  kmhv2::Devices reference(memory.size());
  std::vector<uint8_t> image(memory.size() * 8);
  for (size_t i = 0; i < memory.size(); ++i)
    for (unsigned b = 0; b < 8; ++b) image[i * 8 + b] = memory[i] >> (b * 8);
  reference.load(image);
  std::array<kmhv2::RamPort, 4> ports{};
  for (unsigned i = 0; i < 4; ++i) {
    const auto &p = request.ram[i];
    ports[i] = {true, p.read, p.write, p.readIndex, p.writeIndex, p.data, p.mask};
  }
  reference.step(ports, false, false, 0);
  assert(step(memory.data(), memory.size(), request, outputs).code == 0);
  assert(outputs.ram[0] == 0x11223344 && outputs.ram[1] == outputs.ram[0]);
  for (unsigned i = 0; i < 4; ++i) {
    assert(outputs.ram[i] == reference.ramOutput[i]);
    assert(memory[i] == reference.peek(i));
  }
  assert(memory[0] == 0x1122bbaa);
  const auto before = memory;
  request.ram[1].mask = 0xff;
  assert(step(memory.data(), memory.size(), request, outputs).code == 2);
  assert(memory == before);
  request = {};
  request.ram[0].read = true; request.ram[0].readIndex = memory.size();
  assert(step(memory.data(), memory.size(), request, outputs).code == 1);
  assert(memory == before);
  request = {}; request.sdRead = true;
  assert(step(memory.data(), memory.size(), request, outputs).code == 3);
  request = {};
  const auto held = outputs;
  assert(step(memory.data(), memory.size(), request, outputs).code == 0);
  for (unsigned i = 0; i < 4; ++i) assert(outputs.ram[i] == held.ram[i]);
  request.flashEnable = true;
  assert(step(memory.data(), memory.size(), request, outputs).code == 0);
  assert(outputs.flash == 0x01f292930010029bULL);
  request.flashAddress = 8;
  assert(step(memory.data(), memory.size(), request, outputs).code == 0);
  assert(outputs.flash == 0x0000000000028067ULL);
  Signal signal{};
  signal.count = 3; signal.slots[0] = 2; signal.slots[1] = 0; signal.slots[2] = 1;
  std::array<uint8_t, 3> fragments{};
  signal.write(fragments.data(), 0xabc);
  assert(fragments[0] == 0xb && fragments[1] == 0xc && fragments[2] == 0xa);
  assert(signal.read(fragments.data()) == 0xabc);
}
