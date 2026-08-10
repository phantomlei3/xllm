/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <tuple>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

TEST(ScaledQuantizeMluTest, MapsGeluPytorchTanhToGelu) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::Tensor x =
      torch::arange(-128, 128, fp32_options).reshape({2, 128}).to(bf16_options);
  torch::Tensor smooth = torch::ones({128}, fp32_options);

  auto [alias_output, alias_scale] =
      kernel::mlu::scaled_quantize(x,
                                   smooth,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   "gelu_pytorch_tanh");
  auto [gelu_output, gelu_scale] = kernel::mlu::scaled_quantize(x,
                                                                smooth,
                                                                std::nullopt,
                                                                std::nullopt,
                                                                std::nullopt,
                                                                std::nullopt,
                                                                std::nullopt,
                                                                std::nullopt,
                                                                "gelu");

  EXPECT_TRUE(torch::equal(alias_output.cpu(), gelu_output.cpu()));
  EXPECT_TRUE(torch::equal(alias_scale.cpu(), gelu_scale.cpu()));
}

}  // namespace
}  // namespace xllm
