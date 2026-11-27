#ifndef __SD_DGL_MODELS_DIFFUSION_PREDICTION_H__
#define __SD_DGL_MODELS_DIFFUSION_PREDICTION_H__

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>

#include "core/tensor.hpp"
#include "model/common/rope_circular.hpp"

struct AttentionProcessor;
struct FluxImageBlockProcessor;
struct UNetBlockProcessor;
struct SDCondition;

namespace sd_sample {
    struct SampleStepCacheDispatcher;
}

namespace sd::dgl::models {

    struct ControlBinding;
    struct UNetControls;

    struct DiffusionPrediction {
        sd::Tensor<float> output;
        float skip_scale   = 0.f;
        float output_scale = 0.f;
    };

    struct PredictionParameters {
        float sigma, timestep, input_scale, skip_scale, output_scale;
    };

    struct DiffusionPredictionOptions {
        sd::Tensor<float> vace_context, audio_embedding, video_positions;
        std::vector<int> skip_layers;
        float guidance                      = 3.5f;
        float vace_strength                 = 1.f;
        float fps                           = 24.f;
        float video_sigma_shift             = 12.f;
        float audio_sigma_shift             = 3.f;
        int audio_length                    = 0;
        Rope::RefIndexMode ref_index_mode   = Rope::RefIndexMode::FIXED;
        Rope::RefRoPEAnchor ref_rope_anchor = Rope::RefRoPEAnchor::MODEL_DEFAULT;
        bool force_ref_timestep_zero        = false;
        // Krea 2 Edit layout (Krea2DiffusionExtra): one boost and one position (0 aligned, 1 beside, 2 bag) per reference.
        bool krea2_edit = false;
        std::vector<float> ref_boosts;
        std::vector<int> ref_positions;
        std::function<PredictionParameters(PredictionParameters)> transform;
    };

    struct DiffusionPredictionStep {
        const sd::Tensor<float>& base;
        const sd::Tensor<float>& mask;
        int index            = 1;
        size_t schedule_size = 0;
        // Native caches use the signed solver marker to identify intermediate evaluations.
        int solver_index                            = 1;
        bool completes_step                         = true;
        float target_sigma                          = std::numeric_limits<float>::quiet_NaN();
        float lookahead_sigma                       = std::numeric_limits<float>::quiet_NaN();
        float semantic_sigma                        = std::numeric_limits<float>::quiet_NaN();
        bool mask_input                             = false;
        bool soft_mask                              = false;
        sd_sample::SampleStepCacheDispatcher* cache = nullptr;
        std::vector<const ControlBinding*> controls;
        std::function<UNetControls(const Tensor<float>&, const Tensor<float>&, const SDCondition&)> compute_controls;
        const AttentionProcessor* attn2_processor           = nullptr;
        const FluxImageBlockProcessor* flux_image_processor = nullptr;
        const UNetBlockProcessor* unet_block_processor      = nullptr;
        uint64_t qwen_prefix_id                             = 0;

        void blend_mask(Tensor<float>& latent) const {
            if (mask.empty())
                return;
            if (!soft_mask || schedule_size < 2) {
                latent = latent * mask + base * (1.f - mask);
                return;
            }
            // fugue.soft_mask, differential diffusion (arXiv 2306.00950, Algorithm 1): the mask is a
            // change map, and a cell with value m is held until the step where the fraction of
            // steps left, t / k, falls below m. The paper's map is 1 for no change and its test is
            // map < (k - t) / k, which is m > t / k for m = 1 - map. Step i is t = k - 1 - i.
            const int steps       = static_cast<int>(schedule_size) - 1;
            const float threshold = static_cast<float>(steps - 1 - std::min(index, steps - 1)) / steps;
            sd::Tensor<float> released = mask;
            for (float& value : released.values())
                value = value > threshold ? 1.f : 0.f;
            latent = latent * released + base * (1.f - released);
        }
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_DIFFUSION_PREDICTION_H__
