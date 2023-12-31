/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "cutlass_gemm.h"

#include <cstdint>
#include <optional>
#include <string>
#include <dlfcn.h>

namespace gemm_universal {

using BlockDimFn = void (*)(int32_t m, int32_t n, int32_t k, uint32_t* x,
                            uint32_t* y, uint32_t* z);
using ThreadDimFn = void (*)(uint32_t* x, uint32_t* y, uint32_t* z);
using SharedMemoryBytesFn = int32_t (*)();
using CanImplementFn = bool (*)(int32_t m, int32_t n, int32_t k);
using InitializeFn = void (*)(void* params, int32_t m, int32_t n, int32_t k,
                              void* a, void* b, void* c, int32_t device_sms,
                              int32_t sm_occupancy);
using KernelSymboFn = void* (*)();

static constexpr const char* kBlockDimFn = "__xla_block_dim";
static constexpr const char* kThreadDimFn = "__xla_thread_dim";
static constexpr const char* kSharedMemoryBytes = "__xla_shared_memory_bytes";
static constexpr const char* kCanImplement = "__xla_can_implement";
static constexpr const char* kInitialize = "__xla_initialize";
static constexpr const char* kKernelSymbol = "__xla_kernel_symbol";

static void* Dlopen(const char* path) {
#if defined(PLATFORM_WINDOWS)
  return nullptr;
#else
  return dlopen(path, RTLD_LAZY);
#endif  // defined(PLATFORM_WINDOWS)
}

static void* Dlsym(void* handle, const char* name) {
#if defined(PLATFORM_WINDOWS)
  return nullptr;
#else
  return dlsym(handle, name);
#endif  // defined(PLATFORM_WINDOWS)
}

//===----------------------------------------------------------------------===//
// CUTLASS Host Side Adaptor
//===----------------------------------------------------------------------===//

std::optional<Adaptor<DlOpenedKernel>> Adaptor<DlOpenedKernel>::Load(
    const std::string& path) {
  VLOG(3) << "Load CUTLASS adaptor from a shared library: " << path;

  void* library = Dlopen(path.c_str());
  if (library == nullptr) return std::nullopt;

  auto resolve = [&](const char* name) -> void* {
    void* sym = Dlsym(library, name);
    if (sym == nullptr) {
      LOG(ERROR) << "Failed to resolve CUTLASS adaptor function: " << name
                 << " in library: " << path;
    }
    return sym;
  };

  void* block_dim_fn = resolve(kBlockDimFn);
  if (block_dim_fn == nullptr) return std::nullopt;

  void* thread_dim_fn = resolve(kThreadDimFn);
  if (thread_dim_fn == nullptr) return std::nullopt;

  void* shared_memory_bytes_fn = resolve(kSharedMemoryBytes);
  if (shared_memory_bytes_fn == nullptr) return std::nullopt;

  void* can_implement_fn = resolve(kCanImplement);
  if (shared_memory_bytes_fn == nullptr) return std::nullopt;

  void* initialize_fn = resolve(kInitialize);
  if (shared_memory_bytes_fn == nullptr) return std::nullopt;

  return Adaptor(library, block_dim_fn, thread_dim_fn, shared_memory_bytes_fn,
                 can_implement_fn, initialize_fn);
}

std::optional<Dim3> Adaptor<DlOpenedKernel>::ClusterDim() const {
  return std::nullopt;
}

Dim3 Adaptor<DlOpenedKernel>::BlockDim(int32_t m, int32_t n, int32_t k) const {
  Dim3 dim;
  reinterpret_cast<BlockDimFn>(block_dim_fn_)(m, n, k, &dim.x, &dim.y, &dim.z);
  return dim;
}

Dim3 Adaptor<DlOpenedKernel>::ThreadDim() const {
  Dim3 dim;
  reinterpret_cast<ThreadDimFn>(thread_dim_fn_)(&dim.x, &dim.y, &dim.z);
  return dim;
}

int32_t Adaptor<DlOpenedKernel>::SharedMemoryBytes() const {
  return reinterpret_cast<SharedMemoryBytesFn>(shared_memory_bytes_fn_)();
}

bool Adaptor<DlOpenedKernel>::CanImplement(const Arguments& args) const {
  return reinterpret_cast<CanImplementFn>(can_implement_fn_)(args.m, args.n,
                                                             args.k);
}

void Adaptor<DlOpenedKernel>::Initialize(void* params, const Arguments& args,
                                         int32_t device_sms,
                                         int32_t sm_occupancy) const {
  reinterpret_cast<InitializeFn>(initialize_fn_)(params, args.m, args.n, args.k,
                                                 args.a, args.b, args.c,
                                                 device_sms, sm_occupancy);
}

Adaptor<DlOpenedKernel>::Adaptor(void* handle, void* block_dim_fn,
                                 void* thread_dim_fn,
                                 void* shared_memory_bytes_fn,
                                 void* can_implement_fn, void* initialize_fn)
    : handle_(handle),
      block_dim_fn_(block_dim_fn),
      thread_dim_fn_(thread_dim_fn),
      shared_memory_bytes_fn_(shared_memory_bytes_fn),
      can_implement_fn_(can_implement_fn),
      initialize_fn_(initialize_fn) {}


}  // namespace xla::gpu::kernel::gemm_universal

