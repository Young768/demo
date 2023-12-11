

#include <cstdint>
#include <optional>
#include <string>

namespace gemm_universal {


enum class Arch { kDefault, kSm80 };

template <Arch arch>
struct Bf16xBf16ToBf16 {};

template <Arch arch>
struct F32xF32ToF32 {};


//===----------------------------------------------------------------------===//
// CUTLASS gemm arguments
//===----------------------------------------------------------------------===//

struct Arguments {
  int32_t m;
  int32_t n;
  int32_t k;

  void* a;
  void* b;
  void* c;
};



//===----------------------------------------------------------------------===//
// CUTLASS Host Side Adaptor
//===----------------------------------------------------------------------===//

template <typename Tag>
struct Traits;

struct Dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// This is a type-erased adaptor that has all details required for launching
// CUTLASS kernel on a device. At run time device kernel parameters is really
// just a bag of bytes that driver sends to a kernel, so we rely on it to hide
// CUTLASS templates inside individual build targets and don't leak them into
// XLA, as they contain device code and can't be parsed by regular clang.
template <typename Tag>
class Adaptor {
 public:
  std::optional<Dim3> ClusterDim() const;
  Dim3 BlockDim(int32_t m, int32_t n, int32_t k) const;
  Dim3 ThreadDim() const;

  int32_t SharedMemoryBytes() const;

  bool CanImplement(const Arguments& args) const;
  void Initialize(void* params, const Arguments& args, int32_t device_sms,
                  int32_t sm_occupancy) const;
};


//===----------------------------------------------------------------------===//
// CUTLASS Device Side Adaptor
//===----------------------------------------------------------------------===//

// We keep device side adaptor separate from host side adaptor so that we could
// easily split host and device code compilation if needed.

template <typename Tag>
class DeviceKernel {
 public:
  void* symbol() const;
};


}  



