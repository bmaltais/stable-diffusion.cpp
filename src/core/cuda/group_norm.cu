// Group norm with its affine, and the SiLU after it when there is one, in one kernel (Fugue #382).
//
// ggml's GROUP_NORM, then MUL by gamma and ADD of beta, then (in a resnet block) SILU each read and
// write the whole activation. ggml_ext_group_norm makes one op of the first three where a kernel
// runs it (sd_ggml_group_norm_op); this file is that kernel, and it takes a following SILU too. It
// computes the group statistics exactly as ggml's group_norm_f32 does (the same sums, the same
// reduction tree) and rounds each step as the separate kernels do, so the values are the same.
#include <climits>
#include <cstdlib>
#include "core/ggml_extensions.h"
#include "runtime.h"
#include "group_norm.cuh"

namespace SD_CUDA_NAMESPACE {
    namespace sd_group_norm {
        static bool fusion_disabled() {
            static const bool value = [] {
                const char* fusion = std::getenv("GGML_CUDA_DISABLE_FUSION");
                return fusion && std::atoi(fusion);
            }();
            return value;
        }

        static bool supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
            SDGroupNormParams params;
            return sd_ggml_group_norm_describe(node, &params) && cuda_device_info(device).device >= 0;
        }

        // The op, then a SILU of it.
        static int match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
            auto* g = const_cast<ggml_cgraph*>(graph);
            if (fusion_disabled() || first + 1 >= ggml_graph_n_nodes(g))
                return 0;
            const auto* norm = ggml_graph_node(g, first);
            const auto* silu = ggml_graph_node(g, first + 1);
            if (!supports(device, norm, nullptr) || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
                silu->src[0] != norm || silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(silu) || !ggml_are_same_shape(silu, norm))
                return 0;
            return 2;
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
        // Writing exactly over the input is safe: a group's block reads its values before it writes them.
        static bool needs_scratch(const ggml_tensor* dst, const ggml_tensor* norm) {
            return (overlaps(dst, norm->src[0]) && address(dst) != address(norm->src[0])) || overlaps(dst, norm->src[1]) ||
                   overlaps(dst, norm->src[2]);
        }

        static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* workspace, void*) {
            const auto* dst  = nodes[count - 1];
            const bool fresh = !dst->data && !dst->buffer && !dst->view_src;
            *workspace       = fresh || (dst->data && !needs_scratch(dst, nodes[0])) ? 0 : ggml_nbytes(dst);
            return GGML_STATUS_SUCCESS;
        }
        static void destroy(void*, void*) {}

        static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void*) {
            ggml_ext_gpu_context context;
            SDGroupNormParams params;
            const auto* norm = nodes[0];
            auto* dst        = nodes[count - 1];
            if (!cuda_context(execution->backend, &context) || !sd_ggml_group_norm_describe(norm, &params))
                return GGML_STATUS_FAILED;
            const bool alias = needs_scratch(dst, norm);
            if (alias && !execution->workspace)
                return GGML_STATUS_FAILED;
            auto* output         = static_cast<float*>(alias ? ggml_backend_buffer_get_base(execution->workspace) : dst->data);
            const auto stream    = static_cast<CUDAStream>(context.stream);
            const auto* x        = norm->src[0];
            const int channels   = int(x->ne[2]);
            const int plane      = int(x->ne[0] * x->ne[1]);
            const int group_size = plane * (channels / params.groups);
            const int elements   = int(ggml_nelements(x));
            const unsigned grid  = unsigned(params.groups * x->ne[3]);
            const auto* in       = static_cast<const float*>(x->data);
            const auto* gamma    = static_cast<const float*>(norm->src[1]->data);
            const auto* beta     = static_cast<const float*>(norm->src[2]->data);
            const bool silu      = count == 2;
            // ggml's launch: a warp for a group under 1024 values, else 1024 threads.
            if (group_size < 1024) {
                if (silu)
                    group_norm_affine<32, true><<<grid, 32, 0, stream>>>(in, output, gamma, beta, group_size, elements, params.eps, plane, channels);
                else
                    group_norm_affine<32, false><<<grid, 32, 0, stream>>>(in, output, gamma, beta, group_size, elements, params.eps, plane, channels);
            } else if (silu) {
                group_norm_affine<1024, true><<<grid, 1024, 0, stream>>>(in, output, gamma, beta, group_size, elements, params.eps, plane, channels);
            } else {
                group_norm_affine<1024, false><<<grid, 1024, 0, stream>>>(in, output, gamma, beta, group_size, elements, params.eps, plane, channels);
            }
            CUDAError error = cuda_last_error();
            if (error == cuda_success && alias)
                error = cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream);
            return error == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    void register_group_norm(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.group_norm_affine." SD_CUDA_NAME ".v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_group_norm_op();
        desc.priority             = 100;
        desc.supports             = sd_group_norm::supports;
        desc.prepare              = sd_group_norm::prepare;
        desc.destroy              = sd_group_norm::destroy;
        desc.execute              = sd_group_norm::execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
        desc.name     = "sd.group_norm_affine_silu." SD_CUDA_NAME ".v1";
        desc.priority = 240;
        desc.supports = nullptr;
        desc.match    = sd_group_norm::match;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
