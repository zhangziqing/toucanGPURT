#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Small independent RV64I interpreter for checked-loop.S and the default flash.
// Unsupported instructions fail; it is never used for arbitrary benchmarks.
struct LoopOracle {
    std::array<uint64_t, 32> reg{};
    std::vector<uint8_t> ram;
    uint64_t pc = 0x10000000;
    uint64_t instructions = 0;
    bool trapped = false;
    uint64_t trap_code = 0;
    explicit LoopOracle(const std::vector<uint8_t>& program) : ram(4096) {
        if (program.size() > ram.size()) throw std::runtime_error("Oracle program too large");
        std::copy(program.begin(), program.end(), ram.begin());
    }
    static int64_t sext(uint64_t value, unsigned bits) {
        return int64_t(value << (64 - bits)) >> (64 - bits);
    }
    uint64_t load(uint64_t addr, unsigned size) const {
        static const uint64_t flash[] = {0x01f292930010029bULL, 0x0000000000028067ULL};
        uint64_t value = 0;
        if (addr >= 0x10000000 && addr + size <= 0x10000010)
            std::memcpy(&value, reinterpret_cast<const uint8_t*>(flash) + addr - 0x10000000, size);
        else if (addr >= 0x80000000 && addr + size <= 0x80000000 + ram.size())
            std::memcpy(&value, ram.data() + addr - 0x80000000, size);
        else throw std::runtime_error("Oracle read outside checked program memory");
        return value;
    }
    uint32_t instruction() const { return load(pc, 4); }
    void step() {
        if (trapped) throw std::runtime_error("Oracle stepped beyond trap");
        const uint32_t insn = instruction();
        const unsigned op = insn & 127, rd = (insn >> 7) & 31, f = (insn >> 12) & 7;
        const unsigned rs1 = (insn >> 15) & 31, rs2 = (insn >> 20) & 31;
        const uint64_t a = reg[rs1], b = reg[rs2];
        const int64_t imm = sext(insn >> 20, 12);
        uint64_t next = pc + 4;
        switch (op) {
            case 0x13:
                if (f == 0) reg[rd] = a + imm;
                else if (f == 1) reg[rd] = a << ((insn >> 20) & 63);
                else if (f == 4) reg[rd] = a ^ uint64_t(imm);
                else throw std::runtime_error("Oracle unsupported OP-IMM");
                break;
            case 0x1b:
                if (f != 0) throw std::runtime_error("Oracle unsupported OP-IMM-32");
                reg[rd] = int64_t(int32_t(a + imm));
                break;
            case 0x33:
                if (f != 4 || (insn >> 25)) throw std::runtime_error("Oracle unsupported OP");
                reg[rd] = a ^ b;
                break;
            case 0x37: reg[rd] = int64_t(int32_t(insn & 0xfffff000)); break;
            case 0x17: reg[rd] = pc + int64_t(int32_t(insn & 0xfffff000)); break;
            case 0x67: next = (a + imm) & ~uint64_t(1); reg[rd] = pc + 4; break;
            case 0x03:
                if (f != 3) throw std::runtime_error("Oracle supports only LD");
                reg[rd] = load(a + imm, 8);
                break;
            case 0x23: {
                if (f != 3) throw std::runtime_error("Oracle supports only SD");
                auto off = sext(((insn >> 25) << 5) | ((insn >> 7) & 31), 12);
                const auto addr = a + off;
                if (addr < 0x80000000 || addr + 8 > 0x80000000 + ram.size())
                    throw std::runtime_error("Oracle store outside checked program memory");
                std::memcpy(ram.data() + addr - 0x80000000, &b, 8);
                break;
            }
            case 0x63: {
                if (f != 1) throw std::runtime_error("Oracle supports only BNE");
                const auto off = sext(((insn >> 31) << 12) | (((insn >> 7) & 1) << 11) |
                    (((insn >> 25) & 63) << 5) | (((insn >> 8) & 15) << 1), 13);
                if (a != b) next = pc + off;
                break;
            }
            case 0x0f: break; // FENCE
            case 0x6b: trapped = true; trap_code = a; break;
            default: throw std::runtime_error("Oracle unsupported instruction opcode");
        }
        reg[0] = 0;
        pc = next;
        ++instructions;
    }
};
