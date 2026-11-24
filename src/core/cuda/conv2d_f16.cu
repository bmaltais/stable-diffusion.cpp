// sd.conv2d_f16 on CUDA: a 2D convolution and its bias through a CUTLASS implicit-GEMM kernel
// (Fugue #382). ggml's tensors are channels-first (W fastest), CUTLASS's are channels-last, so the
// input is transposed to NHWC F16 (the same F16 rounding ggml's direct convolution applies), the
// kernel accumulates in F32 and stores an NHWC F16 scratch (67 dB from ggml's F32 output on Klein's
// pictures, and half the bytes), and one more pass writes the NCHW F32 output with
// the bias added and, when the graph adds a residual right after, the residual too. A nearest 2x
// upsample before the convolution (a VAE decoder's) is read in place rather than written first. FLUX.2's VAE
// convolutions on an RTX 3090 run at 120-145 TFLOPS this way against about 27 for ggml's kernel.
#include <climits>
#include <cstdlib>
#include "core/ggml_extensions.h"
#include "runtime.h"

#ifdef SD_CUDA_CONV_CUTLASS
#include <cuda_fp16.h>
#include <mutex>
#include <unordered_map>
// CUTLASS needs its kernel header before the device one.
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "group_norm.cuh"

namespace SD_CUDA_NAMESPACE {
    namespace sd_conv2d {
        // One tile for every VAE shape: within 10% of the best of the tiles tried on each shape.
        using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
            cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC, float,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>, cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
        using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

        static bool fusion_disabled() {
            static const bool value = [] {
                const char* fusion = std::getenv("GGML_CUDA_DISABLE_FUSION");
                return fusion && std::atoi(fusion);
            }();
            return value;
        }

        static int compute_capability(int device) {
            static std::mutex mutex;
            static std::unordered_map<int, int> cache;
            std::lock_guard<std::mutex> lock(mutex);
            auto found = cache.find(device);
            if (found != cache.end())
                return found->second;
            cudaDeviceProp props = {};
            const int cc         = cudaGetDeviceProperties(&props, device) == cudaSuccess ? props.major * 10 + props.minor : 0;
            cache.emplace(device, cc);
            return cc;
        }

        struct Shape {
            int n, h, w, c, k, r, s, p, q;
            SDConv2dParams params;
        };
        static Shape shape(const ggml_tensor* conv, const SDConv2dParams& params) {
            const auto *x = conv->src[0], *kernel = conv->src[1];
            const int u   = params.upsample;
            return {int(x->ne[3]), int(x->ne[1]) * u, int(x->ne[0]) * u, int(x->ne[2]), int(kernel->ne[3]), int(kernel->ne[1]), int(kernel->ne[0]),
                    int(conv->ne[1]), int(conv->ne[0]), params};
        }
        static cutlass::conv::Conv2dProblemSize problem(const Shape& s) {
            return cutlass::conv::Conv2dProblemSize(s.n, s.h, s.w, s.c, s.k, s.r, s.s, s.p, s.q, s.params.p1, s.params.p0, s.params.s1,
                                                    s.params.s0, s.params.d1, s.params.d0, cutlass::conv::Mode::kCrossCorrelation, 1);
        }
        static size_t aligned(size_t bytes) {
            return (bytes + 255) / 256 * 256;
        }
        struct Layout {
            size_t normalized, input, kernel, output, total;
        };
        // `normalized`: a group norm's F16 output that this step makes itself (match_norm).
        static Layout layout(const Shape& s, bool normalized = false) {
            const size_t input = aligned(size_t(s.n) * s.h * s.w * s.c * sizeof(half));
            const size_t output = aligned(size_t(s.n) * s.p * s.q * s.k * sizeof(half));
            Layout l;
            l.normalized = 0;
            l.input      = normalized ? input : 0;
            l.kernel     = l.input + input;
            l.total      = l.kernel + aligned(size_t(s.k) * s.r * s.s * s.c * sizeof(half));
            // The group norm's F16 output is read only by the transpose to NHWC, which comes before the
            // convolution writes: the output takes its place when it fits (a VAE decode's compute
            // buffer 3.3 -> 2.8 GB at 1024x1024, Fugue #382).
            if (normalized && output <= input) {
                l.output = l.normalized;
            } else {
                l.output = l.total;
                l.total += output;
            }
            return l;
        }

        static bool supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
            SDConv2dParams params;
            if (!sd_ggml_conv2d_describe(node, &params))
                return false;
            const auto info = cuda_device_info(device);
            if (info.device < 0 || compute_capability(info.device) < 80)
                return false;
            const auto* x = node->src[0];
            // 128-bit accesses: channels in eights; extents within CUTLASS's int arithmetic.
            if (x->ne[2] % 8 || node->ne[2] % 8 || ggml_nelements(x) * params.upsample * params.upsample > INT_MAX || ggml_nelements(node) > INT_MAX ||
                ggml_nelements(node->src[1]) > INT_MAX)
                return false;
            Conv::Arguments args(problem(shape(node, params)), {nullptr, {}}, {nullptr, {}}, {nullptr, {}}, {nullptr, {}}, {1.f, 0.f});
            return Conv::can_implement(args) == cutlass::Status::kSuccess && Conv::get_workspace_size(args) == 0;
        }

        // The convolution, then an ADD of a residual of its shape.
        static const ggml_tensor* residual_of(const ggml_tensor* conv, const ggml_tensor* add) {
            if (add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || !ggml_are_same_shape(add, conv))
                return nullptr;
            const auto* other = add->src[0] == conv ? add->src[1] : add->src[1] == conv ? add->src[0]
                                                                                        : nullptr;
            if (!other || other == conv || other->type != GGML_TYPE_F32 || !ggml_is_contiguous(other) || !ggml_are_same_shape(other, conv))
                return nullptr;
            return other;
        }
        // A group norm op, its SiLU, then this convolution of it (and maybe a residual ADD): the norm writes
        // F16 for the convolution (half the bytes, the same values), and its F32 output is never made.
        static int match_norm(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
            auto* g = const_cast<ggml_cgraph*>(graph);
            if (fusion_disabled() || first + 2 >= ggml_graph_n_nodes(g))
                return 0;
            const auto* norm = ggml_graph_node(g, first);
            const auto* silu = ggml_graph_node(g, first + 1);
            const auto* conv = ggml_graph_node(g, first + 2);
            SDGroupNormParams np;
            SDConv2dParams cp;
            if (!sd_ggml_group_norm_describe(norm, &np) || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
                silu->src[0] != norm || silu->type != GGML_TYPE_F32 || !ggml_are_same_shape(silu, norm) || conv->src[0] != silu ||
                !supports(device, conv, nullptr) || !sd_ggml_conv2d_describe(conv, &cp) || cp.upsample != 1)
                return 0;
            // and the residual ADD after it, as match takes it
            return first + 3 < ggml_graph_n_nodes(g) && residual_of(conv, ggml_graph_node(g, first + 3)) ? 4 : 3;
        }
        static int match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
            auto* g = const_cast<ggml_cgraph*>(graph);
            if (fusion_disabled() || first + 1 >= ggml_graph_n_nodes(g))
                return 0;
            const auto* conv = ggml_graph_node(g, first);
            const auto* add  = ggml_graph_node(g, first + 1);
            return supports(device, conv, nullptr) && residual_of(conv, add) ? 2 : 0;
        }

        static uintptr_t address(const ggml_tensor* tensor) {
            size_t offset = 0;
            while (tensor && !tensor->data && tensor->view_src) {
                offset += tensor->view_offs;
                tensor = tensor->view_src;
            }
            return tensor && tensor->data ? reinterpret_cast<uintptr_t>(tensor->data) + offset : 0;
        }
        static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
            const auto x = address(a), y = address(b);
            if (!x || !y)
                return false;
            return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
        }

        // The step's convolution: its first node, or its third after a group norm and its SiLU.
        static int conv_index(ggml_tensor* const* nodes, int count) {
            return count >= 3 && ggml_ext_op_get(nodes[0]) == sd_ggml_group_norm_op() ? 2 : 0;
        }
        static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* workspace, void*) {
            SDConv2dParams params;
            const int at = conv_index(nodes, count);
            if (!sd_ggml_conv2d_describe(nodes[at], &params))
                return GGML_STATUS_FAILED;
            *workspace = layout(shape(nodes[at], params), at == 2).total;
            return GGML_STATUS_SUCCESS;
        }
        static void destroy(void*, void*) {}

        // NCHW F32 -> NHWC F16, through 32x32 tiles in shared memory, scaling up `factor` times (nearest,
        // as ggml_upscale) on the way: `plane` and `width` are the output's, `source_plane` the input's.
        template <typename In>
        static __global__ void to_nhwc(const In* src, half* dst, int channels, int plane, int width, int source_plane, int factor) {
            __shared__ float tile[32][33];
            const int n = blockIdx.z;
            const int c0 = blockIdx.y * 32, i0 = blockIdx.x * 32;
            src += size_t(n) * channels * source_plane;
            dst += size_t(n) * channels * plane;
            const int source_width = width / factor;
            for (int y = threadIdx.y; y < 32; y += blockDim.y) {
                const int c = c0 + y, i = i0 + threadIdx.x;
                const int at = factor == 1 ? i : (i / width / factor) * source_width + (i % width) / factor;
                tile[y][threadIdx.x] = c < channels && i < plane ? float(src[size_t(c) * source_plane + at]) : 0.f;
            }
            __syncthreads();
            for (int y = threadIdx.y; y < 32; y += blockDim.y) {
                const int i = i0 + y, c = c0 + threadIdx.x;
                if (c < channels && i < plane)
                    dst[size_t(i) * channels + c] = __float2half(tile[threadIdx.x][y]);
            }
        }
        // ggml's kernel [S][R][C][K] (S fastest) -> CUTLASS's KRSC.
        static __global__ void kernel_to_krsc(const half* src, half* dst, int k, int c, int r, int s) {
            const int64_t i     = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            const int64_t total = int64_t(k) * c * r * s;
            if (i >= total)
                return;
            const int sx = int(i % s), ry = int(i / s % r), ch = int(i / (int64_t(s) * r) % c), ko = int(i / (int64_t(s) * r * c));
            dst[((int64_t(ko) * r + ry) * s + sx) * c + ch] = src[i];
        }
        // NHWC F32 -> NCHW F32 with the bias, and the residual when there is one, as ggml's ADDs round them.
        static __global__ void to_nchw(const half* src, float* dst, const float* bias, const float* residual, int channels, int plane) {
            __shared__ float tile[32][33];
            const int n = blockIdx.z;
            const int c0 = blockIdx.y * 32, i0 = blockIdx.x * 32;
            src += size_t(n) * channels * plane;
            dst += size_t(n) * channels * plane;
            if (residual)
                residual += size_t(n) * channels * plane;
            for (int y = threadIdx.y; y < 32; y += blockDim.y) {
                const int i = i0 + y, c = c0 + threadIdx.x;
                tile[y][threadIdx.x] = c < channels && i < plane ? __half2float(src[size_t(i) * channels + c]) : 0.f;
            }
            __syncthreads();
            for (int y = threadIdx.y; y < 32; y += blockDim.y) {
                const int c = c0 + y, i = i0 + threadIdx.x;
                if (c < channels && i < plane) {
                    const size_t at = size_t(c) * plane + i;
                    float value     = __fadd_rn(tile[threadIdx.x][y], bias[c]);
                    if (residual)
                        value = __fadd_rn(residual[at], value);
                    dst[at] = value;
                }
            }
        }

        static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void*) {
            ggml_ext_gpu_context context;
            SDConv2dParams params;
            const int at     = conv_index(nodes, count);
            const auto* conv = nodes[at];
            auto* dst        = nodes[count - 1];
            if (!cuda_context(execution->backend, &context) || !sd_ggml_conv2d_describe(conv, &params) || !execution->workspace)
                return GGML_STATUS_FAILED;
            const auto* residual = dst != conv ? residual_of(conv, dst) : nullptr;
            if (dst != conv && !residual)
                return GGML_STATUS_FAILED;
            const Shape s    = shape(conv, params);
            const Layout l   = layout(s, at == 2);
            auto* scratch    = static_cast<char*>(ggml_backend_buffer_get_base(execution->workspace));
            auto* input      = reinterpret_cast<half*>(scratch + l.input);
            auto* kernel     = reinterpret_cast<half*>(scratch + l.kernel);
            auto* output     = reinterpret_cast<cutlass::half_t*>(scratch + l.output);
            const auto stream = static_cast<cudaStream_t>(context.stream);
            // The final pass reads the residual and writes dst element by element at the same index, so
            // dst may be the residual's own storage, but no other overlap.
            if ((overlaps(dst, residual) && address(dst) != address(residual)) || overlaps(dst, conv->src[0]) || overlaps(dst, conv->src[1]) ||
                (at == 2 && (overlaps(dst, nodes[0]->src[0]) || overlaps(dst, nodes[0]->src[1]) || overlaps(dst, nodes[0]->src[2]))) ||
                overlaps(dst, conv->src[2]))
                return GGML_STATUS_FAILED;

            const int plane_in = s.h * s.w, plane_out = s.p * s.q, u = params.upsample;
            const dim3 grid_in((plane_in + 31) / 32, (s.c + 31) / 32, s.n);
            if (at == 2) {
                // The group norm and its SiLU write F16 here, the value the transpose would round to.
                SDGroupNormParams norm;
                const auto* x = nodes[0]->src[0];
                if (!sd_ggml_group_norm_describe(nodes[0], &norm))
                    return GGML_STATUS_FAILED;
                auto* normalized     = reinterpret_cast<half*>(scratch + l.normalized);
                const int channels   = int(x->ne[2]);
                const int plane      = int(x->ne[0] * x->ne[1]);
                const int group_size = plane * (channels / norm.groups);
                const unsigned grid  = unsigned(norm.groups * x->ne[3]);
                const auto* gamma    = static_cast<const float*>(nodes[0]->src[1]->data);
                const auto* beta     = static_cast<const float*>(nodes[0]->src[2]->data);
                if (group_size < 1024)
                    sd_group_norm::group_norm_affine<32, true, half><<<grid, 32, 0, stream>>>(static_cast<const float*>(x->data), normalized, gamma, beta,
                                                                                              group_size, int(ggml_nelements(x)), norm.eps, plane, channels);
                else
                    sd_group_norm::group_norm_affine<1024, true, half><<<grid, 1024, 0, stream>>>(static_cast<const float*>(x->data), normalized, gamma,
                                                                                                  beta, group_size, int(ggml_nelements(x)), norm.eps, plane, channels);
                to_nhwc<<<grid_in, dim3(32, 8), 0, stream>>>(normalized, input, s.c, plane_in, s.w, plane_in, 1);
            } else {
                to_nhwc<<<grid_in, dim3(32, 8), 0, stream>>>(static_cast<const float*>(conv->src[0]->data), input, s.c, plane_in, s.w, plane_in / (u * u), u);
            }
            const int64_t weights = int64_t(s.k) * s.c * s.r * s.s;
            kernel_to_krsc<<<unsigned((weights + 255) / 256), 256, 0, stream>>>(static_cast<const half*>(conv->src[1]->data), kernel, s.k, s.c, s.r, s.s);
            if (cudaGetLastError() != cudaSuccess)
                return GGML_STATUS_FAILED;

            Conv::Arguments args(problem(s), {reinterpret_cast<cutlass::half_t*>(input), cutlass::layout::TensorNHWC::packed({s.n, s.h, s.w, s.c})},
                                 {reinterpret_cast<cutlass::half_t*>(kernel), cutlass::layout::TensorNHWC::packed({s.k, s.r, s.s, s.c})},
                                 {output, cutlass::layout::TensorNHWC::packed({s.n, s.p, s.q, s.k})},
                                 {output, cutlass::layout::TensorNHWC::packed({s.n, s.p, s.q, s.k})}, {1.f, 0.f});
            Conv op;
            if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess || op(stream) != cutlass::Status::kSuccess)
                return GGML_STATUS_FAILED;

            to_nchw<<<dim3((plane_out + 31) / 32, (s.k + 31) / 32, s.n), dim3(32, 8), 0, stream>>>(
                reinterpret_cast<const half*>(output), static_cast<float*>(dst->data), static_cast<const float*>(conv->src[2]->data),
                residual ? static_cast<const float*>(residual->data) : nullptr, s.k, plane_out);
            return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    void register_conv2d_f16(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.conv2d_f16.cutlass." SD_CUDA_NAME ".v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_conv2d_op();
        desc.priority             = 100;
        desc.supports             = sd_conv2d::supports;
        desc.prepare              = sd_conv2d::prepare;
        desc.destroy              = sd_conv2d::destroy;
        desc.execute              = sd_conv2d::execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
        desc.name     = "sd.conv2d_f16_residual.cutlass." SD_CUDA_NAME ".v1";
        desc.priority = 240;
        desc.supports = nullptr;
        desc.match    = sd_conv2d::match;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
        // Above the norm's own SiLU fusion (240), which takes the same first two nodes.
        desc.name      = "sd.group_norm_conv2d_f16.cutlass." SD_CUDA_NAME ".v1";
        desc.custom_op = sd_ggml_group_norm_op();
        desc.priority  = 260;
        desc.match     = sd_conv2d::match_norm;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
#else
namespace SD_CUDA_NAMESPACE {
    void register_conv2d_f16(ggml_backend_dev_t) {}
}
#endif
