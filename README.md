# build CMD

/usr/local/cuda/bin/nvcc \
-O3 \
-DNDEBUG \
-std=c++17 \
-arch compute_90a \
-code sm_90a \
-Xcompiler -Wall,-Wextra \
-Xcompiler -Wno-unused-parameter \
-Xcompiler=-Wconversion \
-Xcompiler=-fno-strict-aliasing \
--expt-relaxed-constexpr \
-I ../cutlass/include \
-I ../cutlass/tools/util/include \
-Xcompiler -shared,-fPIC \
-o ./gemm_f32.so \
./gemm_f32.cu \
-lcuda
