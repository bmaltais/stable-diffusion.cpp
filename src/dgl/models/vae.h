#ifndef __SD_DGL_MODELS_VAE_H__
#define __SD_DGL_MODELS_VAE_H__

#include <cstdint>
#include <list>
#include <mutex>
#include <string>

#include "config.h"
#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

class RNG;

namespace sd::dgl::models {
    struct VAELoadParams {
        ModelLoadParams model;
        sd_vae_format_t format = SD_VAE_FORMAT_AUTO;
    };

    class VAE final : public Model {
        ModelRuntime runtime_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        // A LoRA makes a model variant of its own, so only changed files (refresh_weights) or a rebuild
        // change this model's weights.
        void invalidate_cached_state() override { clear_encode_cache(); }

        // Encodes of the same pixels are kept while the model lives (Fugue #382: an edit's reference
        // pictures are encoded again for every seed). Only an encode that does not depend on the RNG
        // is kept; the key is the input's content (two hashes), shape, posterior and tiling.
        struct EncodeCacheEntry {
            std::string key;
            Tensor<float> latents;
        };
        std::list<EncodeCacheEntry> encode_cache_;  // most recently used first
        size_t encode_cache_bytes_ = 0;
        std::mutex encode_cache_mutex_;
        bool encode_is_deterministic() const;
        std::string encode_cache_key(const Tensor<float>& input, bool posterior_mode) const;
        void clear_encode_cache();
        bool build_variant(const Model& source, ModelLoader loader) override;

        bool build_runners();
        bool uses_tae() const;
        bool tae_               = false;
        bool decode_only_       = false;
        sd_vae_format_t format_ = SD_VAE_FORMAT_AUTO;

    public:
        SDVersion version() const { return runtime_.version; }
        int get_latent_channel() const;
        bool circular_x = false;
        bool circular_y = false;
        void apply_circular_axes(bool x, bool y);
        std::shared_ptr<::VAE> first_stage_model;
        sd::TilingConfig vae_tiling_params = {false, false, 0, 0, 0.5f, 0, 0, nullptr};

        VAE(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        bool init(VAELoadParams params);
        bool init_tae(ModelLoadParams params, bool decode_only);
        int get_image_channels() const;
        int get_vae_scale_factor();
        std::vector<int64_t> latent_shape(int64_t width, int64_t height, int frames = 0, int batch = 1);
        int video_frames_to_latent_frames(int frames);
        int align_video_frames(int frames);
        void validate_vae_latent(const Tensor<float>& latent) const;
        Tensor<float> encode_vae_latents(const Tensor<float>& input, bool posterior_mode, const std::shared_ptr<RNG>& rng);
        Tensor<float> encode_vae_latents_uncached(const Tensor<float>& input, bool posterior_mode, const std::shared_ptr<RNG>& rng);
        Tensor<float> vae_to_dit_latents(const Tensor<float>& latent);
        Tensor<float> dit_to_vae_latents(const Tensor<float>& latent);
        Tensor<float> decode_vae_latents(const Tensor<float>& latent, bool decode_video);
    };
}

#endif  // __SD_DGL_MODELS_VAE_H__
