// Execute the pinned basic-difftest Nanhu with synchronous host devices.
#include "ToucanGPUSim.h"
#include "Devices.h"
#include "rv64_loop_oracle.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#ifdef NANHU_GPU_DEVICES
#include "GpuSession.h"
#endif

// Host pointer exported by the pinned SimEval.cu. Whole-pool transfers avoid
// thousands of one-byte CUDA copies per cycle. eval() has synchronized first.
extern uint8_t *regPool_device;

class Pool {
  toucanGPUSim::ToucanSimulator &sim;
  std::vector<uint8_t> bytes;
#ifdef NANHU_GPU_DEVICES
  mutable nanhu_device::Session session;
#endif
  void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  }
public:
  explicit Pool(toucanGPUSim::ToucanSimulator &sim) : sim(sim), bytes(sim.design.regPoolSize) { pull(); }
  void require(const std::string &name, unsigned width) const {
    auto found = sim.symbols.regDebugInfo.find(name);
    if (found == sim.symbols.regDebugInfo.end()) throw std::runtime_error("Missing symbol " + name);
    unsigned total = 0;
    for (auto [id, bits] : found->second) {
      if (id >= bytes.size() || !bits || bits > 4) throw std::runtime_error("Invalid symbol " + name);
      total += bits;
    }
    if (total != width || found->second.size() > 16) throw std::runtime_error("Wrong width " + name);
#ifdef NANHU_GPU_DEVICES
    session.observe(found->second);
#endif
  }
  bool contains(const std::string &name) const { return sim.symbols.regDebugInfo.contains(name); }
  uint64_t get(const std::string &name) const {
    uint64_t value = 0;
    for (auto [id, width] : sim.symbols.regDebugInfo.at(name)) {
      auto part = bytes.at(id);
      if (part >= (1u << width)) throw std::runtime_error("Invalid register fragment " + name);
      value = (value << 4) | part;
    }
    return value;
  }
  void set(const std::string &name, uint64_t value) {
    const auto &locations = sim.symbols.regDebugInfo.at(name);
    for (auto it = locations.rbegin(); it != locations.rend(); ++it) {
      auto [id, width] = *it;
      if ((value & 15) >= (1u << width)) throw std::runtime_error("Input width overflow " + name);
      bytes.at(id) = value & 15;
      value >>= 4;
    }
    if (value) throw std::runtime_error("Input overflow " + name);
  }
  void pull() {
#ifdef NANHU_GPU_DEVICES
    if (session.active()) { session.pull(bytes); return; }
#endif
    check(cudaMemcpy(bytes.data(), regPool_device, bytes.size(), cudaMemcpyDeviceToHost));
  }
  void push() {
#ifdef NANHU_GPU_DEVICES
    if (session.active()) return;
#endif
    check(cudaMemcpy(regPool_device, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
  }
#ifdef NANHU_GPU_DEVICES
  void start(const std::vector<uint8_t> &image, const std::string &ram,
             const std::string &flash, const std::string &sd) { session.start(sim, image, bytes, ram, flash, sd); }
  const nanhu_device::Metadata &deviceMetadata() const { return session.metadata; }
  size_t observationBytes() const { return session.observationBytes(); }
#endif
};

int main(int argc, char **argv) try {
  if (argc != 6) throw std::runtime_error("usage: nanhu DESIGN IMAGE OUTPUT MAX_CYCLES LOOP_COUNT");
  const std::string directory = argv[1], output = argv[3];
  const auto limit = std::stoull(argv[4]), loops = std::stoull(argv[5]);
  if (!limit || limit > 10000000 || !loops) throw std::runtime_error("Invalid run limits");
  std::ifstream imageFile(argv[2], std::ios::binary);
  if (!imageFile) throw std::runtime_error("Missing workload");
  std::vector<uint8_t> image((std::istreambuf_iterator<char>(imageFile)), {});
  if (image.empty()) throw std::runtime_error("Empty workload");
  LoopOracle oracle(image);
  std::filesystem::create_directories(output);
  std::ofstream trace(output + "/commits.tsv"), uart(output + "/uart.txt"), bus(output + "/devices.tsv");
  if (!trace || !uart || !bus) throw std::runtime_error("Cannot open output files");
  toucanGPUSim::ToucanSimulator sim{};
  if (sim.init(0, directory + "/GPUSimDesign.bin", directory + "/GPUSimIOSymbols.bin", false, false))
    throw std::runtime_error("GPU initialization failed");
  Pool pool(sim);
  const std::string core = "SimTop.cpu.l_soc.core_with_l2.core.";
  const std::string arch = core + "exuBlocks_scheduler_difftest_module.dpic.io_value_";
  const std::string trap = core + "ctrlBlock_rob_difftest_module_6.dpic.";
  const std::string ram = "SimTop.cpu.l_simAXIMem.ram.rdata_mem.helper_";
  const std::string flash = "SimTop.cpu.l_simMMIO.flash.flash.";
  const std::string sd = "SimTop.cpu.l_simMMIO.sd.sdHelper.";
  const std::string event = core + "exuBlocks_fuBlock.exeUnits_6.csr.difftest_module.dpic.";
  auto commit = [&](unsigned lane) {
    return core + "ctrlBlock_rob_difftest_module" + (lane ? "_" + std::to_string(lane) : "") + ".dpic.";
  };
  pool.require("SimTop.reset", 1);
  pool.require("SimTop.difftest_uart_in_ch", 8);
  pool.require("SimTop.difftest_uart_out_valid", 1);
  pool.require("SimTop.difftest_uart_out_ch", 8);
  pool.require(flash + "ren", 1); pool.require(flash + "addr", 16); pool.require(flash + "data", 64);
  pool.require(sd + "ren", 1); pool.require(sd + "setAddr", 1); pool.require(sd + "data", 32);
  pool.require(trap + "io_hasTrap", 1);
  pool.require(trap + "io_instrCnt", 64);
  pool.require(trap + "io_cycleCnt", 64);
  pool.require(event + "io_exception", 32); pool.require(event + "io_interrupt", 32);
  for (unsigned i = 0; i < 32; ++i) pool.require(arch + std::to_string(i), 64);
  for (unsigned lane = 0; lane < 6; ++lane) {
    const auto prefix = commit(lane);
    for (const char *field : {"enable", "io_skip", "io_isRVC", "io_rfwen", "io_fpwen"})
      pool.require(prefix + field, 1);
    for (const char *field : {"io_index", "io_special", "io_wdest"}) pool.require(prefix + field, 8);
  }
  pool.require(ram + "0.r_0_enable", 1); pool.require(ram + "0.w_0_enable", 1);
  for (unsigned i = 0; i < 4; ++i) {
    const auto prefix = ram + std::to_string(i) + ".";
    for (const char *field : {"r_0_data", "r_0_index", "w_0_data", "w_0_index", "w_0_mask"})
      pool.require(prefix + field, 64);
    if (pool.contains(prefix + "r_0_async")) pool.set(prefix + "r_0_async", 0);
  }
  std::vector<std::string> watchNames;
  std::ifstream watchInput(std::filesystem::path(argv[2]).parent_path() / "nanhu-watch.tsv");
  std::string watchName;
  unsigned watchCount, watchWidth;
  while (watchInput >> watchName >> watchWidth >> watchCount) {
    if (!watchCount || watchCount > 16) throw std::runtime_error("Invalid watch fragments");
    std::vector<std::tuple<uint32_t, uint32_t>> fragments(watchCount);
    for (auto &[id, width] : fragments)
      if (!(watchInput >> id >> width)) throw std::runtime_error("Incomplete watch metadata");
    auto found = sim.symbols.regDebugInfo.find(watchName);
    if (found != sim.symbols.regDebugInfo.end() && found->second != fragments)
      throw std::runtime_error("Conflicting watch metadata");
    sim.symbols.regDebugInfo[watchName] = fragments;
    pool.require(watchName, watchWidth);
    watchNames.push_back(watchName);
  }
  std::ofstream watch(output + "/watch.bin", std::ios::binary);
  if (!watch) throw std::runtime_error("Cannot open state watch output");
  // Nanhu's four helpers share read/write enables and the implicit root clock;
  // the delivery's source audit proves these aliases for the pinned design.
#ifdef NANHU_GPU_DEVICES
  pool.start(image, ram, flash, sd);
  struct { std::array<uint64_t, 4> ramOutput; } devices;
  std::cout << "GPU resident RAM/flash; observation bytes/cycle=" << pool.observationBytes() << '\n';
#else
  kmhv2::Devices devices(17179869184ULL / 8);
  devices.load(image);
#endif
  pool.set("SimTop.difftest_uart_in_ch", 0); pool.set(sd + "data", 0);
  uint64_t retired = 0, records = 0, firstCommit = 0, trapCycle = 0;
  uint64_t ramReads = 0, ramWrites = 0, flashReads = 0, comparisons = 0;
  bool sawTrap = false;
  std::set<unsigned> defined{0};
  double evaluationSeconds = 0;
  const auto start = std::chrono::steady_clock::now();
  for (uint64_t cycle = 0; cycle < limit; ++cycle) {
    pool.set("SimTop.reset", cycle < 64);
    pool.push();
    const auto evalStart = std::chrono::steady_clock::now();
    if (sim.eval()) throw std::runtime_error("RTL stop before workload completion");
    evaluationSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - evalStart).count();
    pool.pull();
    if (!watchNames.empty() && (cycle == 63 || cycle == 735 || (cycle >= 8190 && cycle <= 8356))) {
      watch.write(reinterpret_cast<const char *>(&cycle), sizeof(cycle));
      for (const auto &name : watchNames) {
        const uint64_t value = pool.get(name);
        watch.write(reinterpret_cast<const char *>(&value), sizeof(value));
      }
    }
    if (pool.get(sd + "ren") || pool.get(sd + "setAddr")) throw std::runtime_error("Unexpected SD access");
    std::array<kmhv2::RamPort, 4> ports;
    for (unsigned i = 0; i < ports.size(); ++i) {
      const auto prefix = ram + std::to_string(i) + ".";
      ports[i] = {true, bool(pool.get(ram + "0.r_0_enable")), bool(pool.get(ram + "0.w_0_enable")),
                  pool.get(prefix + "r_0_index"), pool.get(prefix + "w_0_index"),
                  pool.get(prefix + "w_0_data"), pool.get(prefix + "w_0_mask")};
      ramReads += ports[i].read; ramWrites += ports[i].write;
    }
#ifdef NANHU_GPU_DEVICES
    if (pool.deviceMetadata().cycle != cycle) throw std::runtime_error("GPU device cycle mismatch");
    std::copy_n(pool.deviceMetadata().outputs.ram, 4, devices.ramOutput.begin());
#else
    devices.step(ports, false, false, 0);
#endif
    for (unsigned i = 0; i < ports.size(); ++i) {
      const auto &p = ports[i];
      if (p.read || p.write) bus << cycle << '\t' << i << '\t' << p.read << '\t' << p.write
        << '\t' << std::hex << p.readIndex << '\t' << p.writeIndex << '\t' << p.data << '\t' << p.mask
        << '\t' << pool.get(ram + std::to_string(i) + ".r_0_data") << '\t' << devices.ramOutput[i]
        << std::dec << '\n';
#ifndef NANHU_GPU_DEVICES
      pool.set(ram + std::to_string(i) + ".r_0_data", devices.ramOutput[i]);
#endif
    }
    if (pool.get(flash + "ren")) {
      const uint64_t boot[] = {0x01f292930010029bULL, 0x0000000000028067ULL};
      const auto index = pool.get(flash + "addr") / 8;
#ifdef NANHU_GPU_DEVICES
      if (pool.deviceMetadata().outputs.flash != (index < 2 ? boot[index] : 0))
        throw std::runtime_error("GPU flash output mismatch");
#else
      pool.set(flash + "data", index < 2 ? boot[index] : 0);
#endif
      bus << cycle << "\tflash\t" << std::hex << pool.get(flash + "addr") << '\t'
          << (index < 2 ? boot[index] : 0) << std::dec << '\n';
      ++flashReads;
    }
    if (cycle < 64) continue;
    if (pool.get(event + "io_exception") || pool.get(event + "io_interrupt"))
      throw std::runtime_error("Unexpected architectural exception/interrupt");
    if (pool.get("SimTop.difftest_uart_out_valid")) uart.put(char(pool.get("SimTop.difftest_uart_out_ch")));
    if (pool.get(trap + "io_hasTrap") && !sawTrap) { sawTrap = true; trapCycle = cycle; }
    bool active = false;
    for (unsigned lane = 0; lane < 6; ++lane) {
      const auto prefix = commit(lane);
      if (!pool.get(prefix + "enable")) continue;
      if (oracle.trapped) break;
      active = true;
      if (!retired) firstCommit = cycle;
      const auto fused = pool.get(prefix + "io_special");
      if (fused > 1 || pool.get(prefix + "io_skip") || pool.get(prefix + "io_isRVC") ||
          pool.get(prefix + "io_fpwen") || pool.get(prefix + "io_index") != lane)
        throw std::runtime_error("Unexpected commit metadata at cycle " + std::to_string(cycle));
      auto insn = oracle.instruction();
      auto expectedPc = oracle.pc;
      // Basic difftest omits observed PC/instruction. These trace fields are
      // explicitly oracle values; write destinations and GPR values are DUT.
      for (unsigned step = 0; step <= fused; ++step) {
        insn = oracle.instruction();
        const auto op = insn & 127, rd = (insn >> 7) & 31;
        if (op != 0x23 && op != 0x63 && op != 0x0f && op != 0x6b) defined.insert(rd);
        oracle.step(); ++retired;
      }
      const auto op = insn & 127, rd = (insn >> 7) & 31;
      const bool writes = rd && op != 0x23 && op != 0x63 && op != 0x0f && op != 0x6b;
      if (pool.get(prefix + "io_rfwen") != writes || (writes && pool.get(prefix + "io_wdest") != rd))
        throw std::runtime_error("Commit destination mismatch cycle " + std::to_string(cycle));
      trace << cycle << '\t' << lane << '\t' << std::hex << expectedPc << '\t' << insn
            << '\t' << pool.get(prefix + "io_wdest") << '\t' << fused << std::dec << '\n';
      ++records;
    }
    if (active) for (auto reg : defined) {
      const auto actual = pool.get(arch + std::to_string(reg));
      ++comparisons;
      if (actual != oracle.reg[reg]) {
        std::cerr << "cycle=" << cycle << " retired=" << retired << " x" << reg << " actual=0x"
                  << std::hex << actual << " expected=0x" << oracle.reg[reg] << std::dec << '\n';
        throw std::runtime_error("Architectural register mismatch");
      }
    }
    if (sawTrap && oracle.trapped) {
      if (oracle.trap_code || oracle.reg[7] != loops || !ramReads || !flashReads ||
          pool.get(arch + "10") || pool.get(arch + "7") != loops)
        throw std::runtime_error("Workload completion mismatch");
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::ofstream result(output + "/result.json");
      result << "{\"status\":\"passed\",\"total_cycles\":" << cycle + 1 << ",\"reset_cycles\":64"
             << ",\"first_commit_cycle\":" << firstCommit << ",\"trap_cycle\":" << trapCycle
             << ",\"commit_records\":" << records << ",\"retired_with_fusion\":" << retired
             << ",\"loop_count\":" << loops << ",\"register_comparisons\":" << comparisons
             << ",\"ram_reads\":" << ramReads << ",\"ram_writes\":" << ramWrites
             << ",\"flash_reads\":" << flashReads << ",\"evaluation_seconds\":" << evaluationSeconds
             << ",\"wall_seconds\":" << wall << ",\"cycles_per_second\":" << (cycle + 1) / wall
             << ",\"observed_commit_pc_instruction\":false,\"registers\":[";
      for (unsigned i = 0; i < 32; ++i) result << (i ? "," : "") << pool.get(arch + std::to_string(i));
      result << "]";
#ifdef NANHU_GPU_DEVICES
      result << ",\"device_model\":\"gpu_resident\",\"observation_bytes_per_cycle\":" << pool.observationBytes()
             << ",\"per_cycle_device_input_h2d_bytes\":0,\"ram_bytes\":17179869184";
#endif
      result << "}\n";
      std::cout << "PASS Nanhu: " << retired << " retired instructions, " << loops << " checked RAM-loop iterations\n";
      return 0;
    }
    if (sawTrap && cycle - trapCycle > 8) throw std::runtime_error("Trap without matching delayed commits");
    if (cycle % 1000 == 0) {
      trace.flush();
      std::cerr << "root_cycle=" << cycle << " retired=" << retired << " ram_reads=" << ramReads << '\n';
    }
  }
  throw std::runtime_error("Cycle limit before workload completion");
} catch (const std::exception &error) { std::cerr << "FAIL Nanhu: " << error.what() << '\n'; return 1; }
