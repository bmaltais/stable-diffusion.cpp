// ggml's GROUP_NORM with its affine and an optional SiLU in one kernel (Fugue #382), shared by
// group_norm.cu and conv2d_f16.cu (which hands its F16 output straight to a convolution).
#pragma once
#include "runtime.h"

namespace SD_CUDA_NAMESPACE {
    namespace sd_group_norm {
        // ggml's block_reduce<SUM>: a warp's butterfly, then every warp sums the warps' totals.
        template <int Block>
        static __device__ __forceinline__ float block_sum(float value, float* shared) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                value += cuda_shuffle_xor(value, offset, 32);
            if constexpr (Block > 32) {
                const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
                if (lane == 0)
                    shared[warp] = value;
                __syncthreads();
                value = lane < Block / 32 ? shared[lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += cuda_shuffle_xor(value, offset, 32);
            }
            return value;
        }

        template <int Block, bool Silu, typename Out = float>
        static __global__ void group_norm_affine(const float* x, Out* dst, const float* gamma, const float* beta, const int group_size,
                                                 const int ne_elements, const float eps, const int plane, const int channels) {
            const int start = blockIdx.x * group_size + threadIdx.x;
            const int end   = min(blockIdx.x * group_size + group_size, ne_elements);
            __shared__ float sums[64];

            float tmp = 0.0f;
            for (int j = start; j < end; j += Block)
                tmp += x[j];
            tmp              = block_sum<Block>(tmp, sums);
            const float mean = tmp / group_size;

            tmp = 0.0f;
            for (int j = start; j < end; j += Block) {
                const float xi = x[j] - mean;
                tmp += xi * xi;
            }
            tmp                  = block_sum<Block>(tmp, sums + 32);
            const float variance = tmp / group_size;
            const float scale    = rsqrtf(variance + eps);

            for (int j = start; j < end; j += Block) {
                const int c = (j / plane) % channels;
                float value = __fadd_rn(__fmul_rn(__fmul_rn(x[j] - mean, scale), gamma[c]), beta[c]);
                if constexpr (Silu)
                    value = value / (1.0f + expf(-value));
                dst[j] = Out(value);
            }
        }
    }
}
