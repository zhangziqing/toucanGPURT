// Reuse the accepted runner's binding and device initialization; skip cycle checks.
#pragma once
#include "FinalOnly.h"
extern uint8_t *memPool_device;

template<class PoolType, class CommitName>
int runFinal(toucanGPUSim::ToucanSimulator &sim, PoolType &pool, LoopOracle &oracle,
             const std::string &output, uint64_t cycles, uint64_t loops, const std::string &mode,
             const std::string &arch, const std::string &trap, const std::string &event,
             CommitName commit) {
  if ((loops != 1 && loops != 1000) || cycles != (loops == 1 ? 8511u : 10645u) ||
      (mode != "single-final" && mode != "batch-final"))
    throw std::runtime_error("Expected checked loop1/loop1000 fixed endpoint and final-only mode");
  while (!oracle.trapped && oracle.instructions < 100000) oracle.step();
  if (!oracle.trapped || oracle.trap_code || oracle.reg[7] != loops)
    throw std::runtime_error("Independent final oracle failed");
  auto signal = [&](const std::string &name) {
    nanhu_device::Signal result;
    for (auto [slot, width] : sim.symbols.regDebugInfo.at(name)) {
      if (result.count >= 16) throw std::runtime_error("Invalid progress binding");
      result.slots[result.count] = slot; result.widths[result.count++] = width;
    }
    return result;
  };
  nanhu_final::Bindings bindings;
  bindings.trap = signal(trap + "io_hasTrap");
  bindings.exception = signal(event + "io_exception");
  bindings.interrupt = signal(event + "io_interrupt");
  for (unsigned i = 0; i < 6; ++i) {
    bindings.enable[i] = signal(commit(i) + "enable");
    bindings.fused[i] = signal(commit(i) + "io_special");
  }
  nanhu_final::configure(bindings, cycles);
  const auto start = std::chrono::steady_clock::now();
  if (mode == "batch-final") {
    if (sim.eval_free_running(cycles)) throw std::runtime_error("RTL stop during fixed run");
  } else {
    for (uint64_t i = 0; i < cycles; ++i)
      if (sim.eval()) throw std::runtime_error("RTL stop during fixed run");
  }
  const auto evaluated = std::chrono::steady_clock::now();
  pool.pull();
  nanhu_final::Summary progress;
  nanhu_final::download(progress);
  const auto downloaded = std::chrono::steady_clock::now();
  const double evaluation = std::chrono::duration<double>(evaluated - start).count();
  const double wall = std::chrono::duration<double>(downloaded - start).count();
  if (progress.cycles != cycles || pool.deviceMetadata().cycle + 1 != cycles || progress.errors ||
      progress.retired != oracle.instructions || progress.records != oracle.instructions ||
      progress.firstCommit != 631 || progress.trapCycle != (loops == 1 ? 8507u : 10641u) ||
      progress.ramReads != (loops == 1 ? 120u : 108u) || progress.ramWrites || progress.flashReads != 3)
    throw std::runtime_error("Final GPU progress differs from accepted v5 endpoint");
  for (unsigned i = 0; i < 32; ++i)
    if (pool.get(arch + std::to_string(i)) != oracle.reg[i])
      throw std::runtime_error("Final GPR differs from independent oracle: " + std::to_string(i));
  // Validation-only final state copies occur after performance timing, never mid-run.
  auto dump = [&](const char *name, uint8_t *device, size_t size) {
    std::vector<uint8_t> bytes(size);
    if (size && cudaMemcpy(bytes.data(), device, size, cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("Cannot download final validation state");
    std::ofstream file(output + "/" + name, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    if (!file) throw std::runtime_error("Cannot save final validation state");
  };
  dump("regpool-end.bin", regPool_device, sim.design.regPoolSize);
  dump("mempool-end.bin", memPool_device, sim.design.memPoolSize);
  std::ofstream result(output + "/result.json");
  result << std::setprecision(12) << "{\"status\":\"passed\",\"mode\":\"" << mode
         << "\",\"total_cycles\":" << cycles << ",\"loop_count\":" << loops
         << ",\"evaluation_seconds\":" << evaluation << ",\"wall_seconds\":" << wall
         << ",\"cycles_per_second\":" << cycles / wall << ",\"retired\":" << progress.retired
         << ",\"records\":" << progress.records << ",\"trap_cycle\":" << progress.trapCycle
         << ",\"first_commit_cycle\":" << progress.firstCommit << ",\"ram_reads\":" << progress.ramReads
         << ",\"ram_writes\":" << progress.ramWrites << ",\"flash_reads\":" << progress.flashReads
         << ",\"observation_downloads\":1,\"cpu_cycle_checks\":0,\"observation_bytes\":" << pool.observationBytes()
         << ",\"summary_bytes\":" << sizeof(progress) << ",\"launches\":" << (mode == "batch-final" ? 1 : cycles)
         << ",\"registers\":[";
  for (unsigned i = 0; i < 32; ++i) result << (i ? "," : "") << pool.get(arch + std::to_string(i));
  result << "]}\n";
  if (!result) throw std::runtime_error("Cannot save final-only result");
  std::cout << "PASS final-only " << mode << ": " << cycles << " cycles, " << progress.retired
            << " retired, final 32 GPRs/trap/devices match; " << cycles / wall << " cycles/s\n";
  return 0;
}
