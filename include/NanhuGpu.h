// Optional host interface to the pinned Nanhu CUDA device hook.
#pragma once
#include "NanhuDeviceModel.h"
#include <vector>

namespace nanhu_device {
void configure(const Bindings &bindings, const std::vector<uint8_t> &image,
               const std::vector<uint8_t> &registers, const std::vector<uint32_t> &observed,
               uint64_t words);
void download(std::vector<uint8_t> &packet);
void release();
} // namespace nanhu_device
