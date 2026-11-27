// sd_fugue_inspect_model: what a model file is, from its header alone (Fugue, bmaltais/fugue#703).
//
// ModelLoader::init_from_file reads only headers: the safetensors JSON header, the GGUF metadata and
// tensor table, the zip listing or pickle of a checkpoint. No tensor data is read, no backend or
// context is made. The detection is the one a generation does (dgl::models::resolve_family): the
// file's own names, then again with the "model.diffusion_model." prefix.

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include <json.hpp>

#include "model.h"
#include "model_definition.h"
#include "model_loader.h"
#include "stable-diffusion.h"

namespace {

    const char* vae_kind_name(ModelVAEKind kind) {
        switch (kind) {
            case ModelVAEKind::KL: return "kl";
            case ModelVAEKind::FLUX: return "flux";
            case ModelVAEKind::FLUX2: return "flux2";
            case ModelVAEKind::WAN: return "wan";
            case ModelVAEKind::HUNYUAN: return "hunyuan";
            case ModelVAEKind::LTXAV: return "ltxav";
            case ModelVAEKind::MINIMAX_H3: return "minimax_h3";
            case ModelVAEKind::MAGE: return "mage";
        }
        return "kl";
    }

    nlohmann::json inspect(const char* path) {
        if (!path || !*path)
            return {{"error", "no path"}};
        ModelLoader loader;
        ModelLoader::FileId file;
        if (!loader.add_file(path, "", &file))
            return {{"error", std::string("cannot read the header of '") + path + "'"}};
        SDVersion version = loader.get_sd_version();
        if (version == VERSION_COUNT) {
            loader.set_file_prefix(file, "model.diffusion_model.");
            version = loader.get_sd_version();
        }
        nlohmann::json out = {{"recognised", false}};
        const auto* definition = version == VERSION_COUNT ? nullptr : find_model_definition(version);
        if (!definition)
            return out;
        out["recognised"]      = true;
        out["version"]         = sd_version_name(version);
        out["id"]              = definition->id ? definition->id : "";
        out["is_dit"]          = definition->traits.is_dit;
        out["video"]           = definition->traits.supports_video_generation;
        out["inpaint"]         = definition->traits.is_inpaint;
        out["latent_channels"] = definition->geometry.latent_channels;
        out["vae_kind"]        = vae_kind_name(definition->traits.vae_kind);
        return out;
    }

}  // namespace

char* sd_fugue_inspect_model(const char* path) {
    nlohmann::json out;
    try {
        out = inspect(path);
    } catch (const std::exception& error) {
        out = {{"error", error.what()}};
    } catch (...) {
        out = {{"error", "unknown failure"}};
    }
    const std::string text = out.dump();
    char* copy             = static_cast<char*>(std::malloc(text.size() + 1));
    if (copy)
        std::memcpy(copy, text.c_str(), text.size() + 1);
    return copy;
}

void sd_fugue_string_free(char* text) {
    std::free(text);
}
