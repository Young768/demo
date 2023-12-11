
#include <cstdint>

#include "cutlass_gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

namespace gemm_universal {

using GemmOperation =
    cutlass::gemm::device::GemmUniversal<float, cutlass::layout::RowMajor,
                                         float, cutlass::layout::RowMajor,
                                         float, cutlass::layout::RowMajor>;


#define XLA_GPU_DEFINE_CUTLASS_GEMM_TRAITS(TAG, OPERATION) \
  template <>                                              \
  struct Traits<TAG> {                                     \
    using Operation = OPERATION;                           \
    using Arguments = typename Operation::Arguments;       \
    using Kernel = typename Operation::GemmKernel;         \
    using Params = typename Kernel::Params;                \
  }


XLA_GPU_DEFINE_CUTLASS_GEMM_TRAITS(F32xF32ToF32<Arch::kDefault>,
                                   GemmOperation);

using CutlassGemm = F32xF32ToF32<Arch::kDefault>;
extern template struct Adaptor<CutlassGemm>;
extern template struct DeviceKernel<CutlassGemm>;

template struct Adaptor<F32xF32ToF32<Arch::kDefault>>;

extern "C" void __xla_block_dim(int32_t m, int32_t n, int32_t k, uint32_t* x,
                                uint32_t* y, uint32_t* z) {
  Adaptor<CutlassGemm> adaptor;
  auto dim = adaptor.BlockDim(m, n, k);
  *x = dim.x;
  *y = dim.y;
  *z = dim.z;
}

extern "C" void __xla_thread_dim(uint32_t* x, uint32_t* y, uint32_t* z) {
  Adaptor<CutlassGemm> adaptor;
  auto dim = adaptor.ThreadDim();
  *x = dim.x;
  *y = dim.y;
  *z = dim.z;
}

extern "C" int32_t __xla_shared_memory_bytes() {
  Adaptor<CutlassGemm> adaptor;
  return adaptor.SharedMemoryBytes();
}

extern "C" bool __xla_can_implement(int32_t m, int32_t n, int32_t k) {
  Adaptor<CutlassGemm> adaptor;
  Arguments arguments = {m, n, k};
  return adaptor.CanImplement(arguments);
}

extern "C" void __xla_initialize(void* params, int32_t m, int32_t n, int32_t k,
                                 void* a, void* b, void* c, int32_t device_sms,
                                 int32_t sm_occupancy) {
  Adaptor<CutlassGemm> adaptor;
  Arguments arguments = {m, n, k, a, b, c};
  adaptor.Initialize(params, arguments, device_sms, sm_occupancy);
}

extern "C" void* __xla_kernel_symbol() {
  DeviceKernel<CutlassGemm> kernel;
  return kernel.symbol();
}

}  // namespace xla::gpu::kernel::gemm_universal
