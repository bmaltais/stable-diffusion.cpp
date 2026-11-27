#include "api/generation_program.h"

#include <algorithm>
#include <cmath>

#include "core/file_access.h"
#include "core/image_geometry.h"
#include "core/util.h"
#include "model_loader.h"

namespace sd::api {
    static bool supports_reference_image_guidance(SDVersion version) {
        return version == VERSION_FLUX || version == VERSION_CHROMA || sd_version_is_flux2(version) ||
               (sd_version_is_qwen_image(version) && version != VERSION_QWEN_IMAGE_2_1) ||
               sd_version_is_mage_flow(version) || sd_version_is_longcat(version) ||
               sd_version_is_z_image(version) || sd_version_is_boogu_image(version);
    }

    // ref_image_args: only "resize_before_vae=true|false" (the name upstream gave it before the DGL
    // engine, which sizes references in generation_image.cpp); false keeps each reference as it came.
    static bool keep_reference_size(const char* args) {
        const std::string value = args ? args : "";
        if (value.empty() || value == "resize_before_vae=true")
            return false;
        if (value == "resize_before_vae=false")
            return true;
        throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "ref_image_args supports only resize_before_vae=true or resize_before_vae=false");
    }

    // moss_edit_format: "" or "ai-toolkit" keep the library's layout; "moss-image-edit" is ComfyUI's
    // Moss-Image-Edit (text_start anchor, references only through the VAE).
    static bool moss_image_edit_layout(const char* format) {
        const std::string value = format ? format : "";
        if (value.empty() || value == "ai-toolkit")
            return false;
        if (value == "moss-image-edit")
            return true;
        throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "moss_edit_format supports only ai-toolkit or moss-image-edit");
    }

    // krea2_edit_format: "" or "ostris" keep the library's Krea 2 edit layout; "krea2edit" is the layout of the Krea 2
    // Identity Edit and General Edit LoRAs, ported from ComfyUI-Krea2Edit (Apache-2.0, github.com/lbouaraba/comfyui-krea2edit,
    // v2.0.0): see krea2_fit, image_references and Krea2DiffusionExtra.
    static bool krea2_edit_layout(SDVersion version, const char* format) {
        const std::string value = format ? format : "";
        if (value.empty() || value == "ostris")
            return false;
        if (value != "krea2edit")
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "krea2_edit_format supports only ostris or krea2edit");
        return sd_version_is_krea2(version);
    }

    static std::string krea2_position(const char* value, const char* option) {
        const std::string position = value && *value ? value : "aligned";
        if (position != "aligned" && position != "beside" && position != "bag")
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, std::string(option) + " supports only aligned, beside or bag");
        return position;
    }

    // How a Krea 2 Edit reference fits the output (ComfyUI-Krea2Edit's _fit_encode_image, fit_mode "fit"): a picture
    // within 8% of the output's aspect fills it exactly after a centre crop; any other is fitted inside it, each side
    // floored to 16 and the picture cut from the middle so that the fitted side lands on that grid at the same scale
    // (no squash). The diffusion model centres a smaller reference on the output's grid.
    struct Krea2Fit {
        int crop_x, crop_y, crop_width, crop_height, width, height;
    };

    static Krea2Fit krea2_fit(int iw, int ih, int pw, int ph) {
        const auto round_half_even = [](double value) { return int(std::nearbyint(value)); };
        const double sc     = std::min(double(ph) / ih, double(pw) / iw);
        constexpr double kCropTolerance = 0.08;
        Krea2Fit fit{};
        if (ih * sc >= ph * (1 - kCropTolerance) && iw * sc >= pw * (1 - kCropTolerance)) {
            const double s  = std::max(double(ph) / ih, double(pw) / iw);
            fit.crop_height = std::min(ih, round_half_even(ph / s));
            fit.crop_width  = std::min(iw, round_half_even(pw / s));
            fit.width       = pw;
            fit.height      = ph;
        } else {
            fit.height      = std::min(std::max(16, int(ih * sc) / 16 * 16), std::max(16, ph / 16 * 16));
            fit.width       = std::min(std::max(16, int(iw * sc) / 16 * 16), std::max(16, pw / 16 * 16));
            fit.crop_height = std::min(ih, std::max(1, round_half_even(fit.height / sc)));
            fit.crop_width  = std::min(iw, std::max(1, round_half_even(fit.width / sc)));
        }
        fit.crop_x = (iw - fit.crop_width) / 2;
        fit.crop_y = (ih - fit.crop_height) / 2;
        return fit;
    }

    static std::string reference_pixels(GenerationTemplate& request, const sd_tensor_t* tensor, size_t index, bool encode) {
        const auto version = request.definition.version;
        const auto name    = "reference_" + std::to_string(index);
        bind_tensor(request.program, name, tensor);
        int width = int(sd_tensor_dim(tensor, 1)), height = int(sd_tensor_dim(tensor, 2));
        const int target_width = request.values.at("width"), target_height = request.values.at("height");
        const int factor = sd_version_is_qwen_image(version) ? 32 : 16;
        bool crop_only   = false;
        if (krea2_edit_layout(version, request.image.fugue.krea2_edit_format)) {
            const auto fit    = krea2_fit(width, height, target_width, target_height);
            const auto pixels = name + "_pixels";
            LOG_INFO("reference %zu: %dx%d -> %dx%d before the VAE (Krea 2 Edit fit, crop %dx%d at %d,%d)", index, width, height,
                     fit.width, fit.height, fit.crop_width, fit.crop_height, fit.crop_x, fit.crop_y);
            request.fields["references"] += request.render("image_reference", {{"name", name}, {"channels", request.definition.geometry.image_channels}, {"width", fit.width}, {"height", fit.height}, {"fit_crop", true}, {"center_crop", false}, {"crop_only", false}, {"resize_width", fit.width}, {"resize_height", fit.height}, {"crop_x", fit.crop_x}, {"crop_y", fit.crop_y}, {"crop_width", fit.crop_width}, {"crop_height", fit.crop_height}, {"mode", literal("bicubic")}, {"encode", encode}, {"augment", false}, {"latent", "reference_latent_" + std::to_string(index)}});
            return pixels;
        }
        if (version == VERSION_P_H3_IMAGE || version == VERSION_MINIMAX_H3_IMAGE) {
            const double scale = version == VERSION_P_H3_IMAGE ? std::min(1.0, std::sqrt(double(target_width) * target_height / (double(width) * height))) : double(std::clamp(std::min(width, height), 1024, 2048)) / std::min(width, height);
            width              = std::max(32, int(std::nearbyint(width * scale / 32)) * 32);
            height             = std::max(32, int(std::nearbyint(height * scale / 32)) * 32);
        } else if (version == VERSION_MOSS_IMAGE && !keep_reference_size(request.image.fugue.ref_image_args)) {
            const double scale = std::min(1.0, 1024.0 / std::max(width, height));
            width              = std::max(factor, int(std::round(width * scale / factor)) * factor);
            height             = std::max(factor, int(std::round(height * scale / factor)) * factor);
        } else if (version != VERSION_ANIMA && !sd_version_is_pid(version) && keep_reference_size(request.image.fugue.ref_image_args)) {
            // As each came, cut from the middle down to the VAE's multiple (the VAE would rescale it
            // otherwise); a side under one multiple stays.
            crop_only = true;
            width     = width >= factor ? width / factor * factor : width;
            height    = height >= factor ? height / factor * factor : height;
        } else if (version != VERSION_ANIMA && !sd_version_is_pid(version)) {
            if (sd_version_is_llada_image(version) || sd_version_is_mage_flow(version) || (version == VERSION_FIBO2 && request.image.ref_images_count == 1)) {
                width  = target_width;
                height = target_height;
            } else {
                const int pixels = version == VERSION_QWEN_IMAGE_2_1 ? target_width * target_height : 1024 * 1024;
                if (version == VERSION_FIBO2) {
                    std::tie(width, height) = sd::select_image_size_bucket(width, height, pixels, factor, target_width, target_height);
                } else {
                    const double w = std::sqrt(double(std::min(pixels, target_width * target_height)) * width / height);
                    height         = std::max(factor, int(std::round(w * height / width / factor)) * factor);
                    width          = std::max(factor, int(std::round(w / factor)) * factor);
                }
            }
        }
        LOG_INFO("reference %zu: %dx%d -> %dx%d before the VAE", index, int(sd_tensor_dim(tensor, 1)), int(sd_tensor_dim(tensor, 2)), width, height);
        const auto pixels        = name + "_pixels";
        const auto scale         = crop_only ? 1.0 : std::max(double(width) / sd_tensor_dim(tensor, 1), double(height) / sd_tensor_dim(tensor, 2));
        const auto resize_width  = int(std::ceil(sd_tensor_dim(tensor, 1) * scale));
        const auto resize_height = int(std::ceil(sd_tensor_dim(tensor, 2) * scale));
        request.fields["references"] += request.render("image_reference", {{"name", name}, {"channels", request.definition.geometry.image_channels}, {"width", width}, {"height", height}, {"fit_crop", false}, {"center_crop", version == VERSION_FIBO2 || crop_only}, {"crop_only", crop_only}, {"resize_width", resize_width}, {"resize_height", resize_height}, {"crop_x", (resize_width - width) / 2}, {"crop_y", (resize_height - height) / 2}, {"crop_width", width}, {"crop_height", height}, {"mode", literal(sd_version_is_minimax_h3_image(version) ? "lanczos" : "bilinear")}, {"encode", encode}, {"augment", version == VERSION_MINIMAX_H3_IMAGE}, {"latent", "reference_latent_" + std::to_string(index)}});
        return pixels;
    }

    static bool prompt_has_vision(const GenerationTemplate& request) {
        const auto* vision_path = request.context.model.config.context.llm_vision_path;
        if (vision_path && *vision_path)
            return true;
        const auto path = request.values.at("text_path").get<std::string>();
        require_file_access(path, SD_DGL_FILE_READ_MODEL);
        ModelLoader metadata;
        if (!metadata.init_from_file(path))
            throw dgl::Error(SD_STATUS_LOAD_FAILED, "cannot inspect text encoder metadata: " + path);
        const auto names = metadata.get_tensor_names();
        return std::any_of(names.begin(), names.end(), [](const auto& name) {
            return name.find("visual.") != std::string::npos;
        });
    }

    static void image_references(GenerationTemplate& request) {
        const auto version = request.definition.version;
        const auto refs    = bound_tensors(request.options, "ref_images");
        if (refs.empty() && !sd_version_is_unet_edit(version))
            return;
        const bool moss_edit  = version == VERSION_MOSS_IMAGE;
        const bool qwen_edit  = sd_version_is_qwen_image(version) && version != VERSION_QWEN_IMAGE_LAYERED;
        const bool moss_layout = moss_edit && moss_image_edit_layout(request.image.fugue.moss_edit_format);
        const bool has_vision  = (moss_edit || qwen_edit) && !moss_layout && prompt_has_vision(request);
        const bool krea2_edit  = krea2_edit_layout(version, request.image.fugue.krea2_edit_format);
        if (moss_layout)
            LOG_INFO("MOSS-Image edit layout moss-image-edit: %zu reference(s) through the VAE only, text_start anchor, no vision", refs.size());
        else if (moss_edit && !has_vision)
            LOG_INFO("MOSS-Image HQ edit is using text-only prompt conditioning; set llm_vision_path (--llm_vision) to add reference images to Qwen3.5");
        std::string pixels, latents;
        const bool encode = version != VERSION_HIDREAM_O1;
        for (size_t i = 0; i < refs.size(); ++i) {
            const auto image  = reference_pixels(request, refs[i], i, encode);
            const auto latent = "reference_latent_" + std::to_string(i);
            if (encode)
                latents += (latents.empty() ? "" : ", ") + latent;
            auto prompt_image = version == VERSION_QWEN_IMAGE_2_1 || version == VERSION_FIBO2 || sd_version_is_llada_image(version) ? image : "reference_" + std::to_string(i) + "_original";
            if (krea2_edit) {
                // The vision encoder sees each reference at most krea2_grounding_px on the long side (ComfyUI-Krea2Edit's
                // grounding_px, an area downscale there, never up), whatever its size before the VAE.
                const int limit = request.image.fugue.krea2_grounding_px;
                const int w = int(sd_tensor_dim(refs[i], 1)), h = int(sd_tensor_dim(refs[i], 2));
                if (limit > 0 && std::max(w, h) > limit) {
                    const double s     = double(limit) / std::max(w, h);
                    const int gw       = std::max(1, int(std::nearbyint(w * s))), gh = std::max(1, int(std::nearbyint(h * s)));
                    const auto grounded = "reference_" + std::to_string(i) + "_grounded";
                    request.fields["references"] += "    " + grounded + " = tensor::interpolate(" + prompt_image + ", size = [" +
                                                    std::to_string(gw) + ", " + std::to_string(gh) + "], mode = \"bilinear\", antialias = true)\n";
                    prompt_image = grounded;
                }
            }
            pixels += (pixels.empty() ? "" : ", ") + prompt_image;
        }
        if (sd_version_is_minimax_h3_image(version)) {
            auto blocks = nlohmann::json::array();
            for (size_t i = 0; i < refs.size(); ++i)
                blocks.push_back({{"kind", "image"}, {"video", i}});
            request.fields["predictor_options"] += ", reference_blocks = " + literal(blocks);
            request.fields["negative_predictor_options"] += ", reference_blocks = " + literal(blocks);
            request.fields["reference_predictor_options"] += ", reference_blocks = " + literal(blocks);
        }
        if (has_vision || sd_version_is_minimax_h3_image(version) ||
            version == VERSION_FIBO2 || sd_version_is_llada_image(version) || sd_version_is_krea2(version) ||
            sd_version_is_mage_flow(version) || sd_version_is_longcat(version) || sd_version_is_boogu_image(version)) {
            request.fields["prompt_options"] += ", media = [" + pixels + "]";
            if (!sd_version_is_llada_image(version) && version != VERSION_FIBO2)
                request.fields["negative_prompt_options"] += ", media = [" + pixels + "]";
            if (krea2_edit) {
                request.fields["prompt_options"] += ", image_labels = false";
                request.fields["negative_prompt_options"] += ", image_labels = false";
            }
        }
        if (encode && !sd_version_is_unet_edit(version)) {
            const bool increase = sd_version_is_flux2(version) || sd_version_is_flux3(version) ||
                                  sd_version_is_qwen_image(version) || sd_version_is_minimax_h3_image(version) ||
                                  version == VERSION_FIBO2 || sd_version_is_krea2(version) ||
                                  version == VERSION_ANIMA || sd_version_is_mage_flow(version) || moss_edit;
            std::string options = ", ref_latents = [" + latents + "], ref_index_mode = " +
                                  literal(version == VERSION_QWEN_IMAGE_LAYERED ? "decrease" : increase ? "increase"
                                                                                                        : "fixed") +
                                  ", force_ref_timestep_zero = " + literal(sd_version_is_llada_image(version) || (sd_version_is_krea2(version) && !krea2_edit) || moss_edit);
            if (krea2_edit) {
                // The last reference is the person (or the only one), an earlier one the scene: ComfyUI-Krea2Edit's
                // ref_boost/ref_position and ref_boost_a/ref_position_a.
                const auto& fugue = request.image.fugue;
                std::vector<float> boosts;
                std::vector<std::string> positions;
                for (size_t i = 0; i < refs.size(); ++i) {
                    const bool last = i + 1 == refs.size();
                    boosts.push_back(last ? fugue.krea2_ref_boost : fugue.krea2_ref_boost_a);
                    positions.push_back(last ? krea2_position(fugue.krea2_ref_position, "krea2_ref_position")
                                             : krea2_position(fugue.krea2_ref_position_a, "krea2_ref_position_a"));
                }
                LOG_INFO("Krea 2 Edit layout: %zu reference(s) at the current timestep, boosts %s, positions %s, vision at most %d px",
                         refs.size(), nlohmann::json(boosts).dump().c_str(), nlohmann::json(positions).dump().c_str(), fugue.krea2_grounding_px);
                options += ", krea2_edit = true, ref_boosts = " + literal(nlohmann::json(boosts)) + ", ref_positions = " + literal(nlohmann::json(positions));
            }
            if (moss_edit)
                options += moss_layout ? ", ref_rope_anchor = \"text_start\"" : ", ref_rope_anchor = \"image_start\"";
            request.fields["predictor_options"] += options;
            request.fields["negative_predictor_options"] += options;
            request.fields["reference_predictor_options"] += options;
        }
        if (sd_version_is_unet_edit(version)) {
            request.fields["predictor_options"] += ", concat = edit_latent";
            request.fields["negative_predictor_options"] += ", concat = edit_latent";
        }
    }

    static void image_inpaint(GenerationTemplate& request) {
        const bool inpaint = sd_version_is_inpaint(request.definition.version);
        const auto* mask   = bound_tensor(request.options, "mask_image");
        if (!mask && !inpaint)
            return;
        if (mask) {
            if (!request.image.init_image.data)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "mask_image requires init_image");
            bind_tensor(request.program, "input_mask", mask);
        }
        if (inpaint) {
            request.fields["predictor_options"] += ", concat = inpaint_condition";
            request.fields["negative_predictor_options"] += ", concat = inpaint_condition";
        }
        request.fields["sample_options"] += inpaint ? ", base = base, denoise_mask = denoise_mask" : ", base = base, denoise_mask = mask_latent";
    }

    static void image_control(GenerationTemplate& request) {
        const auto version = request.definition.version;
        if (!request.image.control_image.data) {
            if (version == VERSION_FLUX_CONTROLS) {
                request.fields["predictor_options"] += ", concat = tensor::zeros_like(base)";
                request.fields["negative_predictor_options"] += ", concat = tensor::zeros_like(base)";
            }
            return;
        }
        bind_tensor(request.program, "input_control", bound_tensor(request.options, "control_image"));
        if (version == VERSION_FLUX_CONTROLS || version == VERSION_FLEX_2) {
            if (version == VERSION_FLUX_CONTROLS) {
                request.fields["predictor_options"] += ", concat = control_latent";
                request.fields["negative_predictor_options"] += ", concat = control_latent";
            }
            return;
        }
        const auto& settings = request.context.model.config.context;
        if (!settings.control_net_path || !*settings.control_net_path)
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "control_image requires control_net_path");
        request.values["control_path"]       = settings.control_net_path;
        request.values["control_strength"]   = request.image.control_strength;
        request.fields["control_options"]    = arguments(loader_options(settings, SDBackendModule::CONTROL_NET, request.definition));
        request.fields["control_hint"]       = sd_version_is_z_image(version) ? "control_hint" : "control_pixels";
        request.fields["control_type"]       = literal(sd_control_type_name(request.image.control_type));
        request.fields["positive_condition"] = "controlled_positive";
        request.fields["negative_condition"] = "controlled_negative";
    }

    void prepare_image_program(GenerationTemplate& request) {
        const auto version            = request.definition.version;
        const auto& settings          = request.context.model.config.context;
        const auto prediction         = settings.prediction == PREDICTION_COUNT ? request.context.model.default_prediction : settings.prediction;
        const bool inpaint            = sd_version_is_inpaint(version);
        const bool mask               = bound_tensor(request.options, "mask_image");
        const bool refs               = request.image.ref_images_count > 0;
        const bool control            = request.image.control_image.data;
        const bool native_control     = version == VERSION_FLUX_CONTROLS || version == VERSION_FLEX_2;
        const auto image_scale        = request.image.sample_params.guidance.img_cfg;
        const bool reference_guidance = refs && supports_reference_image_guidance(version);
        if (std::isfinite(image_scale) && image_scale != 1.f &&
            !sd_version_is_inpaint_or_unet_edit(version) && !reference_guidance)
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "sample.guidance.img_cfg is not supported by this model with these inputs");
        if (version != VERSION_QWEN_IMAGE_LAYERED && request.options && scalar_values(request.options).contains("qwen_image_layers"))
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "qwen_image_layers requires Qwen Image Layered");
        const bool image_guidance = (inpaint || sd_version_is_control(version) || (sd_version_is_unet_edit(version) && refs) || reference_guidance) && std::isfinite(image_scale) && image_scale != 1.f;
        request.flags             = {
            {"has_init", bool(request.image.init_image.data)}, {"has_mask", mask}, {"mask_stage", mask || inpaint}, {"references", refs}, {"unet_edit", sd_version_is_unet_edit(version)}, {"inpaint_model", inpaint}, {"flux_fill", version == VERSION_FLUX_FILL}, {"flex", version == VERSION_FLEX_2}, {"control_image", control}, {"native_control", native_control}, {"controlnet", control && !native_control}, {"z_control", control && sd_version_is_z_image(version)}, {"image_guidance", image_guidance}, {"minit2i", sd_version_is_minit2i(version)}, {"ming", version == VERSION_MING_IMAGE}, {"ideogram", sd_version_is_ideogram4(version)}, {"separate_negative_model", false}, {"sdxl_zero", sd_version_is_sdxl(version) && text(request.image.negative_prompt).empty() && prediction != EDM_V_PRED}, {"layered", version == VERSION_QWEN_IMAGE_LAYERED}, {"ph3", version == VERSION_P_H3_IMAGE}, {"h3_reference_noise", version == VERSION_MINIMAX_H3_IMAGE && refs}, {"custom_sigmas", request.image.sample_params.custom_sigmas_count > 0}};
        request.fields["predictor_core"]          = request.fields["predictor_options"];
        request.flags["raw_reference"]            = version == VERSION_SD1_PIX2PIX;
        request.flags["reference_image_guidance"] = reference_guidance && image_guidance;
        request.flags["hidream_o1"]               = version == VERSION_HIDREAM_O1;
        request.flags["control_image_guidance"]   = image_guidance && version == VERSION_FLUX_CONTROLS;
        if (version == VERSION_MOSS_IMAGE)
            request.fields["negative_predictor_options"] += ", skip_layers = [22]";
        if (sd_version_is_sensenova_u1(version)) {
            request.fields["negative_prompt_input"] = "negative_prompt_raw";
            request.fields["negative_prompt_options"] += ", raw = true";
        }
        request.fields["entry"] = mask || inpaint ? "image_inpaint" : refs || sd_version_is_unet_edit(version) ? "image_edit"
                                                                                                               : "image";
        if (version == VERSION_QWEN_IMAGE_LAYERED) {
            if (request.image.init_image.data)
                throw dgl::Error(SD_STATUS_UNSUPPORTED, "Qwen layered uses ref_images rather than init_image");
            request.fields["entry"]  = "image_layered";
            request.values["layers"] = request.image.qwen_image_layers + 1;
        }
        if (sd_version_is_minimax_h3_image(version)) {
            request.fields["entry"] = "image_h3";
            const auto options      = version == VERSION_P_H3_IMAGE ? ", video_sigma_shift = 1, audio_length = 2, audio_sigma_shift = 1" : ", video_sigma_shift = 1";
            request.fields["predictor_options"] += options;
            request.fields["negative_predictor_options"] += options;
        }
        if (sd_version_is_ideogram4(version) && !text(settings.uncond_diffusion_model_path).empty()) {
            request.flags["separate_negative_model"] = true;
            request.values["negative_model_path"]    = settings.uncond_diffusion_model_path;
            request.fields["negative_model"]         = "negative_model";
            request.fields["schedule_predictor"]     = "pos";
        }
        if (sd_version_is_pid(version) || sd_version_is_minit2i(version) || version == VERSION_IRIS) {
            request.fields["entry"] = "image_pixel";
            if (sd_version_is_pid(version) && !refs)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "PiD requires a reference image");
        }
        request.fields["unconditional_prompt_options"] = request.fields["negative_prompt_options"];
        image_references(request);
        image_control(request);
        image_inpaint(request);
        if (image_guidance) {
            request.values["image_cfg_scale"] = image_scale;
            request.fields["guidance_options"] += ", image_only = image_only, image_scale = image_cfg_scale";
            request.fields["image_only_condition"]       = request.fields["negative_condition"];
            request.fields["image_only_options"]         = request.fields["negative_predictor_options"];
            request.fields["negative_predictor_options"] = request.fields["predictor_core"];
            if (reference_guidance)
                request.fields["negative_condition"] = request.flags.at("controlnet").get<bool>() ? "controlled_unconditional" : "unconditional_text";
            else
                request.fields["negative_predictor_options"] += ", concat = image_unconditional_concat";
        }
        if (sd_version_is_sefi_image(version)) {
            request.fields["entry"] = "image_sefi";
            auto options            = nlohmann::json::object();
            for (const auto& [key, value] : parse_key_value_args(request.image.sample_params.extra_sample_args, "SeFi scheduler arg")) {
                if (key != "sefi_alpha" && key != "sefi_delta_t")
                    continue;
                float number;
                if (parse_strict_float(value, number))
                    options[key == "sefi_alpha" ? "alpha" : "delta"] = number;
                else
                    LOG_WARN("ignoring invalid SeFi scheduler arg '%s=%s'", key.c_str(), value.c_str());
            }
            request.fields["sefi_schedule_options"] = arguments(options);
        }
    }
}
