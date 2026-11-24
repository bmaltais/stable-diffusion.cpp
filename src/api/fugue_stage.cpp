#include "api/fugue_stage.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <sstream>

#include "api/fugue_stages.h"
#include "api/generation_program.h"
#include "api/result.h"
#include "api/tensor.h"
#include "core/util.h"
#include "dgl/bindings/arguments.h"
#include "dgl/bindings/condition.h"
#include "dgl/models/denoising_parameterization.h"
#include "dgl/ops/predictor.h"
#include "dgl/syntax/syntax.h"
#include "model/vae/geometry.hpp"
#include "runtime/denoiser.hpp"

namespace sd::api::fugue_stage {

    using Replacements = std::map<std::string, std::vector<std::string>>;

    [[noreturn]] static void refuse(const std::string& why) {
        throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "fugue: " + why);
    }

    static std::string quoted(const std::string& value) {
        return literal(nlohmann::json(value));
    }

    static void collect_references(const dgl::Expr& expr, std::set<std::string>& names) {
        if (expr.form == dgl::Expr::Form::Reference)
            names.insert(expr.name.substr(0, expr.name.find('.')));
        for (const auto& [name, argument] : expr.arguments)
            if (argument)
                collect_references(*argument, names);
        for (const auto& statement : expr.body)
            if (statement.value)
                collect_references(*statement.value, names);
    }

    static std::vector<std::string> split_lines(const std::string& source) {
        std::vector<std::string> lines;
        std::istringstream in(source);
        for (std::string line; std::getline(in, line);)
            lines.push_back(line);
        return lines;
    }

    static size_t main_line(const std::vector<std::string>& lines) {
        for (size_t i = 0; i < lines.size(); ++i)
            if (lines[i].rfind("fn main(", 0) == 0)
                return i;
        refuse("the rendered program has no main");
    }

    // A rendered program: one statement per line in main (the templates are written so).
    class Rendered {
    public:
        explicit Rendered(const std::string& source)
            : lines_(split_lines(source)), main_(main_line(lines_)), syntax_(dgl::parse_syntax(source, true)) {
            for (const auto& statement : syntax_.entry.body) {
                if (!statement.value || static_cast<size_t>(statement.value->line - 1) <= main_)
                    continue;
                if (statement.form == dgl::Statement::Form::Return)
                    returned_ = &statement;
                else if (statement.form == dgl::Statement::Form::Binding)
                    bound_.insert(statement.name);
            }
            if (!returned_)
                refuse("the rendered program returns nothing");
        }

        bool has(const std::string& name) const { return bound_.count(name) != 0; }

        const char* either(const char* first, const char* second) const {
            if (has(first))
                return first;
            if (has(second))
                return second;
            refuse(std::string("the program has neither '") + first + "' nor '" + second + "'");
        }

        // The right-hand side of a binding.
        std::string value_of(const std::string& name) const {
            for (const auto& statement : syntax_.entry.body)
                if (statement.name == name && statement.value && static_cast<size_t>(statement.value->line - 1) > main_) {
                    const auto& line = lines_[statement.value->line - 1];
                    const auto equal = line.find(" = ");
                    if (equal == std::string::npos)
                        break;
                    return line.substr(equal + 3);
                }
            refuse("the program has no value '" + name + "'");
        }

        // The name the program returns as its latent.
        std::string result_latent() const {
            for (const auto& [name, value] : returned_->value->arguments)
                if (name == "latent" && value && value->form == dgl::Expr::Form::Reference)
                    return value->name;
            refuse("the program returns no latent");
        }

        // Replaces statements, returns `result` (empty: what the program returns), and keeps only
        // the statements that result depends on.
        void cut(GenerationProgram& program, const Replacements& replace, const std::string& result) const {
            for (const auto& [name, lines] : replace)
                if (!has(name))
                    refuse("the program has no value '" + name + "'");
            std::vector<std::string> body;
            for (const auto& statement : syntax_.entry.body) {
                if (!statement.value || static_cast<size_t>(statement.value->line - 1) <= main_ ||
                    statement.form == dgl::Statement::Form::Return)
                    continue;
                const auto replaced = replace.find(statement.name);
                if (replaced != replace.end())
                    body.insert(body.end(), replaced->second.begin(), replaced->second.end());
                else
                    body.push_back(lines_[statement.value->line - 1]);
            }
            const size_t return_at   = static_cast<size_t>(returned_->value->line - 1);
            const std::string ending = result.empty() ? lines_[return_at] : "    return " + result;
            std::string head, tail;
            for (size_t i = 0; i < main_; ++i)
                head += lines_[i] + "\n";
            for (size_t i = return_at + 2; i < lines_.size(); ++i)
                tail += lines_[i] + "\n";
            const auto main_with = [&](const std::vector<std::string>& parameters, const std::vector<std::string>& statements) {
                std::string text = head + "fn main(";
                for (size_t i = 0; i < parameters.size(); ++i)
                    text += (i ? ", " : "") + parameters[i];
                text += ") {\n";
                for (const auto& line : statements)
                    text += line + "\n";
                return text + ending + "\n}\n" + tail;
            };

            const auto candidate       = main_with(syntax_.entry.parameters, body);
            const auto candidate_lines = split_lines(candidate);
            const size_t candidate_main = main_line(candidate_lines);
            const auto parsed          = dgl::parse_syntax(candidate, true);
            std::set<std::string> needed;
            std::vector<size_t> kept;
            for (auto statement = parsed.entry.body.rbegin(); statement != parsed.entry.body.rend(); ++statement) {
                if (!statement->value || static_cast<size_t>(statement->value->line - 1) <= candidate_main)
                    continue;
                if (statement->form == dgl::Statement::Form::Return) {
                    collect_references(*statement->value, needed);
                } else if (statement->form == dgl::Statement::Form::Binding && needed.count(statement->name)) {
                    needed.erase(statement->name);
                    collect_references(*statement->value, needed);
                    kept.push_back(static_cast<size_t>(statement->value->line - 1));
                }
            }
            std::reverse(kept.begin(), kept.end());
            std::vector<std::string> statements;
            for (const size_t line : kept)
                statements.push_back(candidate_lines[line]);
            std::vector<std::string> parameters;
            for (const auto& parameter : syntax_.entry.parameters)
                if (needed.count(parameter))
                    parameters.push_back(parameter);
            for (auto input = program.inputs.begin(); input != program.inputs.end();)
                input = needed.count(input->first) ? std::next(input) : program.inputs.erase(input);
            program.source = main_with(parameters, statements);
        }

    private:
        std::vector<std::string> lines_;
        size_t main_;
        dgl::Syntax syntax_;
        std::set<std::string> bound_;
        const dgl::Statement* returned_ = nullptr;
    };

    static fugue_stages::File read_stage(const std::string& path, const char* kind, const std::string& version) {
        require_file_access(path, SD_DGL_FILE_READ);
        fugue_stages::File file;
        if (!fugue_stages::read_file(path, file))
            refuse("cannot read '" + path + "': " + file.error);
        if (file.meta["kind"] != kind)
            refuse("'" + path + "' is not a " + kind + " file");
        if (file.meta["version"] != version)
            refuse("'" + path + "' was made for " + file.meta["version"] + ", not " + version);
        return file;
    }

    static std::vector<sd::Tensor<float>> read_latents(const std::string& path, const std::string& version) {
        auto file = read_stage(path, "fugue/latent", version);
        std::vector<sd::Tensor<float>> latents;
        if (!fugue_stages::get_list(file, "latent", latents))
            refuse("cannot read latent '" + path + "': " + file.error);
        if (latents.empty())
            refuse("no latent in '" + path + "'");
        for (const auto& latent : latents)
            if (latent.empty())
                refuse("an empty latent in '" + path + "'");
        return latents;
    }

    struct LatentSize {
        int width  = 0;
        int height = 0;
        int frames = 0;
        int fps    = 0;
    };

    static void save_latents(const std::string& path, const std::vector<sd::Tensor<float>>& latents, const std::string& version, const LatentSize& size) {
        require_file_access(path, SD_DGL_FILE_WRITE);
        fugue_stages::File file;
        file.meta["kind"]    = "fugue/latent";
        file.meta["version"] = version;
        // The image size, for Fugue's latent descriptor (spec 2.6); decoding reads the tensor's shape.
        file.meta["width"]  = std::to_string(size.width);
        file.meta["height"] = std::to_string(size.height);
        if (size.frames > 0) {
            file.meta["frames"] = std::to_string(size.frames);
            file.meta["fps"]    = std::to_string(size.fps);
        }
        fugue_stages::put_list(file, "latent", latents);
        if (!fugue_stages::write_file(path, file))
            refuse("cannot write latent '" + path + "'");
    }

    static void save_conditions(const std::string& path, const dgl::Value& value, const std::string& version) {
        require_file_access(path, SD_DGL_FILE_WRITE);
        fugue_stages::File file;
        file.meta["kind"]    = "fugue/conditioning";
        file.meta["version"] = version;
        fugue_stages::put_cond(file, "cond", dgl::read_condition(value.fields.at("fugue_positive").fields));
        fugue_stages::put_cond(file, "uncond", dgl::read_condition(value.fields.at("fugue_negative").fields));
        const auto unconditional = value.fields.find("fugue_unconditional");
        if (unconditional != value.fields.end())
            fugue_stages::put_cond(file, "img_uncond", dgl::read_condition(unconditional->second.fields));
        if (!fugue_stages::write_file(path, file))
            refuse("cannot write conditioning '" + path + "'");
    }

    static int field_int(const dgl::Value& value, const char* name) {
        return static_cast<int>(value.fields.at(name).scalar.get<int64_t>());
    }

    static LatentSize result_size(const dgl::Value& value) {
        LatentSize size{field_int(value, "fugue_width"), field_int(value, "fugue_height")};
        if (value.fields.count("fugue_frames")) {
            size.frames = field_int(value, "fugue_frames");
            size.fps    = field_int(value, "fugue_fps");
        }
        return size;
    }

    static const sd::Tensor<float>& result_latent(const dgl::Value& value) {
        return dgl::read_tensor<float>(value.fields.at("fugue_latent"));
    }

    static void check_stop(const std::string& stop) {
        if (!stop.empty() && stop != "init" && stop != "cond" && stop != "sample")
            refuse("unknown stage '" + stop + "' to stop after; use init, cond or sample");
    }

    // The statements that read the conditioning file in place of the encode.
    static void read_conditions(Replacements& replace, const Rendered& rendered, const std::string& path, const std::string& version) {
        const auto load = [&](const char* set) {
            return "fugue::load_condition(" + quoted(path) + ", " + quoted(set) + ", " + quoted(version) + ")";
        };
        const std::string positive = rendered.either("positive", "positive_text");
        const std::string negative = rendered.either("negative", "negative_text");
        replace[positive]          = {"    " + positive + " = " + load("cond")};
        replace[negative]          = {"    " + negative + " = " + load("uncond")};
        if (rendered.has("unconditional_text"))
            replace["unconditional_text"] = {"    unconditional_text = " + load("img_uncond")};
    }

    static std::string conditions_result(const Rendered& rendered) {
        std::string result = std::string("{fugue_positive = ") + rendered.either("positive", "positive_text") +
                             ", fugue_negative = " + rendered.either("negative", "negative_text");
        if (rendered.has("unconditional_text"))
            result += ", fugue_unconditional = unconditional_text";
        return result + "}";
    }

    struct NoiseLevel {
        bool enabled     = false;
        bool force_first = false;
    };

    static NoiseLevel noise_level(const char* args) {
        NoiseLevel reading;
        for (const auto& [key, value] : parse_key_value_args(args, "img2img arg")) {
            bool flag = false;
            if (key != "strength_as_noise_level" && key != "force_first_sigma")
                continue;
            if (!parse_strict_bool(value, flag)) {
                LOG_WARN("ignoring invalid img2img sample arg '%s=%s'", key.c_str(), value.c_str());
                continue;
            }
            (key == "strength_as_noise_level" ? reading.enabled : reading.force_first) = flag;
        }
        return reading;
    }

    static bool any_option(const FugueStageConfig& fugue) {
        return fugue.init_latent_in || fugue.init_latent_out || fugue.cond_in || fugue.cond_out || fugue.latent_in ||
               fugue.latent_out || fugue.stop_after || fugue.init_video || fugue.soft_mask;
    }

    bool wanted(const ImageConfig& image) {
        return any_option(image.fugue) || (image.init_image.data && image.strength < 1.f && noise_level(image.sample_params.extra_sample_args).enabled);
    }

    bool wanted(const VideoConfig& video) {
        return any_option(video.fugue);
    }

    // The sampling options of the patch: the noise-level reading of strength and the soft mask.
    static void sampling_edits(Replacements& replace, const Rendered& rendered, const ImageConfig& image, bool inpaint) {
        if (image.strength < 1.f && rendered.has("sigmas") && rendered.value_of("sigmas").rfind("slice_sigmas_by_strength(", 0) == 0) {
            const auto reading = noise_level(image.sample_params.extra_sample_args);
            if (reading.enabled)
                replace["sigmas"] = {std::string("    sigmas = fugue::noise_level_sigmas(pos, full_sigmas, strength, force_first = ") +
                                     (reading.force_first ? "true" : "false") + ")"};
        }
        // An inpaint model reads the mask as a channel it was trained to see as 0 or 1.
        if (!image.fugue.soft_mask || !image.mask_image.data || inpaint)
            return;
        const auto mask             = rendered.value_of("mask_pixels");
        const std::string threshold = "tensor::threshold(";
        const std::string closing   = ", threshold = 0.5)";
        if (mask.rfind(threshold, 0) != 0 || mask.size() <= threshold.size() + closing.size() ||
            mask.compare(mask.size() - closing.size(), closing.size(), closing) != 0)
            refuse("fugue.soft_mask: this program does not round its mask where expected");
        replace["mask_pixels"] = {"    mask_pixels = " + mask.substr(threshold.size(), mask.size() - threshold.size() - closing.size())};
        for (const char* name : {"latent", "hires_latent"}) {
            if (!rendered.has(name))
                continue;
            const auto call = rendered.value_of(name);
            if (call.find("sample(") != std::string::npos && call.back() == ')')
                replace[name] = {"    " + std::string(name) + " = " + call.substr(0, call.size() - 1) + ", soft_mask = true)"};
        }
    }

    static std::string size_fields(bool video) {
        return video ? ", fugue_width = width, fugue_height = height, fugue_frames = frames, fugue_fps = fps"
                     : ", fugue_width = width, fugue_height = height";
    }

    sd_status_t run_image(GenerationContext& context, std::shared_ptr<sd_opt_t> options, OptConfig config, int64_t seed, sd_result_t** out) {
        const std::string stop = text(config.image.fugue.stop_after);
        check_stop(stop);
        const auto& definition = context.model.definition(context);
        const std::string version = sd_version_name(definition.version);
        const bool inpaint        = sd_version_is_inpaint(definition.version);
        if (config.image.fugue.init_latent_in && inpaint)
            refuse(version + " cannot start from a latent alone; give it the image");
        // A start latent without a picture still takes the image-to-image flow; the blank picture
        // is cut away with the encode it would go through.
        api::TensorRef blank;
        if (config.image.fugue.init_latent_in && !config.image.init_image.data) {
            blank = allocate_tensor(SD_TENSOR_U8, {3, config.image.width, config.image.height});
            const auto set = sd_opt_set_tensor(options.get(), "init_image", blank.get());
            if (set != SD_STATUS_OK)
                return set;
            const auto status = resolve(options.get(), Image, config);
            if (status != SD_STATUS_OK)
                return status;
        }
        const auto& fugue = config.image.fugue;
        std::vector<sd::Tensor<float>> sampled_in;
        if (fugue.latent_in)
            sampled_in = read_latents(fugue.latent_in, version);
        const bool single = stop == "init" || stop == "cond";
        const int runs    = fugue.latent_in ? static_cast<int>(sampled_in.size()) : single ? 1
                                                                                           : config.image.batch_count;
        auto result = std::make_unique<sd_result_t>();
        std::vector<sd::Tensor<float>> sampled;
        LatentSize size;
        for (int i = 0; i < runs; ++i) {
            if (context.cancellation.load() == SD_CANCEL_NEW_LATENTS)
                break;
            auto program = image_program(context, config.image, options.get(), int64_t((uint64_t(seed) + uint64_t(i)) & INT64_MAX));
            const Rendered rendered(program.source);
            Replacements replace;
            std::string returned;
            if (stop == "init") {
                returned = std::string("{fugue_latent = ") + rendered.either("image_base", "base") + size_fields(false) + "}";
            } else if (stop == "cond") {
                returned = conditions_result(rendered);
            } else if (fugue.latent_in) {
                const auto latent = rendered.result_latent();
                replace[latent]   = {"    " + latent + " = fugue::load_latent(" + quoted(fugue.latent_in) + ", " + quoted(version) + ", index = " + std::to_string(i) + ")"};
            } else {
                sampling_edits(replace, rendered, config.image, inpaint);
                if (fugue.cond_in)
                    read_conditions(replace, rendered, fugue.cond_in, version);
                if (fugue.init_latent_in) {
                    const std::string start = rendered.either("image_base", "base");
                    replace[start]          = {"    " + start + " = fugue::load_latent(" + quoted(fugue.init_latent_in) + ", " + quoted(version) + ", width = width, height = height)"};
                }
                if (stop == "sample")
                    returned = "{fugue_latent = " + rendered.result_latent() + size_fields(false) + "}";
            }
            rendered.cut(program, replace, returned);
            auto value = execute_program(context, program);
            if (stop == "init") {
                if (!fugue.init_latent_out)
                    refuse("stopping after init writes init_latent_out");
                save_latents(fugue.init_latent_out, {result_latent(value)}, version, result_size(value));
            } else if (stop == "cond") {
                if (!fugue.cond_out)
                    refuse("stopping after cond writes cond_out");
                save_conditions(fugue.cond_out, value, version);
            } else if (stop == "sample") {
                sampled.push_back(result_latent(value));
                size = result_size(value);
            } else {
                append_program_image(*result, value, program.source);
            }
        }
        if (stop == "sample") {
            if (!fugue.latent_out)
                refuse("stopping after sample writes latent_out");
            if (sampled.empty())
                throw dgl::Error(SD_STATUS_CANCELLED, "generation cancelled before the next image");
            save_latents(fugue.latent_out, sampled, version, size);
        } else if (stop.empty() && result->images.empty()) {
            throw dgl::Error(SD_STATUS_CANCELLED, "generation cancelled before the next image");
        }
        context.check_cancelled();
        *out = result.release();
        return SD_STATUS_OK;
    }

    sd_status_t run_video(GenerationContext& context, std::shared_ptr<sd_opt_t> options, const OptConfig& config, int64_t seed, sd_result_t** out) {
        const auto& video = config.video;
        const auto& fugue = video.fugue;
        const std::string stop = text(fugue.stop_after);
        check_stop(stop);
        const auto& definition    = context.model.definition(context);
        const std::string version = sd_version_name(definition.version);
        if (fugue.init_latent_out && !fugue.init_video)
            refuse("a video's init latent is made from init_video");
        if (fugue.init_latent_in && fugue.init_video)
            refuse("init_video and init_latent_in both start the clip");
        if (video.hires.enabled || sd_version_is_ltxav(definition.version) || sd_version_is_minimax_h3(definition.version) ||
            definition.version == VERSION_WAN2_2_S2V)
            refuse(version + " cannot run in stages (latent upscale or audio)");

        auto program = video_program(context, video, options.get(), seed);
        const Rendered rendered(program.source);
        Replacements replace;
        std::string returned;
        const std::string latent = rendered.result_latent();
        if (fugue.latent_in) {
            if (read_latents(fugue.latent_in, version).size() != 1)
                refuse(std::string("the latent '") + fugue.latent_in + "' does not hold one latent");
            replace[latent] = {"    " + latent + " = fugue::load_latent(" + quoted(fugue.latent_in) + ", " + quoted(version) + ")"};
        } else {
            std::vector<std::string> start;
            if (fugue.init_video) {
                start = {"    fugue_clip = fugue::load_clip(" + quoted(fugue.init_video) + ", width, height, frames)",
                         "    fugue_clip_rng = make_rng(rng_type, seed = seed)",
                         "    fugue_start = vae_to_dit_latents(vae, vae_encode(vae, fugue_clip, rng = fugue_clip_rng" + program.vae_options + "))"};
            } else if (fugue.init_latent_in) {
                start = {"    fugue_start = fugue::load_latent(" + quoted(fugue.init_latent_in) + ", " + quoted(version) + ", width = width, height = height, frames = frames)"};
            }
            if (!start.empty()) {
                start.push_back("    base = fugue_start");
                replace["base"] = start;
            }
            if (stop == "init") {
                if (!fugue.init_video)
                    refuse("a video's init stage encodes init_video");
                returned = "{fugue_latent = fugue_start" + size_fields(true) + "}";
            } else if (stop == "cond") {
                returned = conditions_result(rendered);
            } else {
                if (fugue.cond_in)
                    read_conditions(replace, rendered, fugue.cond_in, version);
                // Redraw by `strength` as image to image does: sampling starts part way down the
                // schedule, so the first sigmas (the high-noise pass first) are dropped.
                if (!start.empty() && video.strength < 1.f) {
                    replace["sigmas"] = {"    fugue_schedule = " + rendered.value_of("sigmas"),
                                         "    fugue_steps = tensor::numel(fugue_schedule) - 1",
                                         "    sigmas = fugue_schedule[fugue_steps - math::min(math::floor(fugue_steps * " + literal(double(video.strength)) + "), fugue_steps - 1):]"};
                    if (rendered.has("split") && rendered.value_of("split").find("high_noise_steps") != std::string::npos)
                        replace["split"] = {"    split = math::min(0 - math::min(0 - (high_noise_steps - (fugue_steps + 1 - tensor::numel(sigmas))), 0), tensor::numel(sigmas) - 1)"};
                }
                if (stop == "sample")
                    returned = "{fugue_latent = " + latent + size_fields(true) + "}";
            }
        }
        rendered.cut(program, replace, returned);
        auto value  = execute_program(context, program);
        auto result = std::make_unique<sd_result_t>();
        if (stop == "init") {
            if (!fugue.init_latent_out)
                refuse("stopping after init writes init_latent_out");
            save_latents(fugue.init_latent_out, {result_latent(value)}, version, result_size(value));
        } else if (stop == "cond") {
            if (!fugue.cond_out)
                refuse("stopping after cond writes cond_out");
            save_conditions(fugue.cond_out, value, version);
        } else if (stop == "sample") {
            if (!fugue.latent_out)
                refuse("stopping after sample writes latent_out");
            save_latents(fugue.latent_out, {result_latent(value)}, version, result_size(value));
        } else {
            set_program_video(*result, value, program.source);
        }
        context.check_cancelled();
        *out = result.release();
        return SD_STATUS_OK;
    }

    // A clip to redraw (video-to-video): a YUV4MPEG2 file (4:2:0, the layout the helper writes a
    // decoded video in) as the VAE's input, [W, H, frames, 3, 1] in 0..1 with the full-range BT.601
    // matrix the helper writes it with. Its size and length must be the request's.
    static sd::Tensor<float> read_y4m(const std::string& path, int width, int height, int frames) {
        std::unique_ptr<FILE, int (*)(FILE*)> in(std::fopen(path.c_str(), "rb"), std::fclose);
        if (!in)
            refuse("cannot read video '" + path + "'");
        char line[512];
        if (!std::fgets(line, sizeof line, in.get()) || std::string(line).rfind("YUV4MPEG2", 0) != 0)
            refuse("'" + path + "' is not a YUV4MPEG2 file");
        int w = 0, h = 0;
        bool is420 = true;
        for (const char* p = line; *p != '\0'; ++p) {
            if (*p != ' ')
                continue;
            if (p[1] == 'W')
                w = std::atoi(p + 2);
            else if (p[1] == 'H')
                h = std::atoi(p + 2);
            else if (p[1] == 'C')
                is420 = std::string(p + 2).rfind("420", 0) == 0;
        }
        if (!is420 || w != width || h != height)
            refuse("the video '" + path + "' must be 4:2:0 and " + std::to_string(width) + "x" + std::to_string(height));
        const int64_t cw = (w + 1) / 2, ch = (h + 1) / 2;
        std::vector<unsigned char> y(static_cast<size_t>(w) * h), u(static_cast<size_t>(cw * ch)), v(static_cast<size_t>(cw * ch));
        sd::Tensor<float> video({width, height, frames, 3, 1});
        float* out          = video.data();
        const int64_t plane = static_cast<int64_t>(width) * height;
        for (int f = 0; f < frames; ++f) {
            if (!std::fgets(line, sizeof line, in.get()) || std::string(line).rfind("FRAME", 0) != 0 ||
                std::fread(y.data(), 1, y.size(), in.get()) != y.size() || std::fread(u.data(), 1, u.size(), in.get()) != u.size() ||
                std::fread(v.data(), 1, v.size(), in.get()) != v.size())
                refuse("the video '" + path + "' has fewer than the " + std::to_string(frames) + " frames the request says");
            for (int j = 0; j < height; ++j)
                for (int i = 0; i < width; ++i) {
                    const float Y = y[static_cast<size_t>(j) * w + i];
                    const float U = u[static_cast<size_t>(j / 2) * cw + i / 2] - 128.f;
                    const float V = v[static_cast<size_t>(j / 2) * cw + i / 2] - 128.f;
                    const float r = Y + 1.402f * V, g = Y - 0.344136f * U - 0.714136f * V, b = Y + 1.772f * U;
                    const int64_t at                   = static_cast<int64_t>(j) * width + i;
                    out[(0 * frames + f) * plane + at] = std::clamp(r, 0.f, 255.f) / 255.f;
                    out[(1 * frames + f) * plane + at] = std::clamp(g, 0.f, 255.f) / 255.f;
                    out[(2 * frames + f) * plane + at] = std::clamp(b, 0.f, 255.f) / 255.f;
                }
        }
        // More frames than the request's would be cut silently: say so.
        if (std::fgets(line, sizeof line, in.get()) && std::string(line).rfind("FRAME", 0) == 0)
            refuse("the video '" + path + "' has more than the " + std::to_string(frames) + " frames the request says");
        return video;
    }

    static dgl::Value load_condition(dgl::Values& args, dgl::RunState&) {
        const auto path = dgl::get_string(args, "path");
        const auto set  = dgl::get_string(args, "set");
        auto file       = read_stage(path, "fugue/conditioning", dgl::get_string(args, "version"));
        // An older encode could leave adapter state here, which this engine's generation API cannot
        // apply: refuse it rather than sample without the adapter.
        for (const auto& [name, blob] : file.blobs)
            for (const char* adapter : {"ip_adapter.", "pulid.", "photomaker."})
                if (name.rfind(adapter, 0) == 0)
                    refuse("the conditioning '" + path + "' uses " + adapter + "*, which this engine's generation API no longer takes");
        SDCondition condition;
        if (!fugue_stages::get_cond(file, set, condition))
            refuse("cannot read conditioning '" + path + "': " + file.error);
        // With a guidance of 1 an older encode left the negative empty: it is the positive.
        if (condition.empty() && set == "uncond" && !fugue_stages::get_cond(file, "cond", condition))
            refuse("cannot read conditioning '" + path + "': " + file.error);
        if (condition.empty())
            refuse("the conditioning '" + path + "' has no '" + set + "' set");
        return dgl::condition_value(std::move(condition));
    }

    static dgl::Value load_latent(dgl::Values& args, dgl::RunState&) {
        const auto path = dgl::get_string(args, "path");
        auto file       = read_stage(path, "fugue/latent", dgl::get_string(args, "version"));
        const auto number = [&](const char* key) {
            const auto found = file.meta.find(key);
            return found == file.meta.end() ? 0 : std::atoi(found->second.c_str());
        };
        const auto fits = [&](const char* key, const char* meta) {
            const auto wanted = dgl::get_int(args, key);
            return wanted == 0 || number(meta) == 0 || number(meta) == wanted;
        };
        if (!fits("width", "width") || !fits("height", "height") || !fits("frames", "frames"))
            refuse("the latent '" + path + "' does not fit this request's size");
        std::vector<sd::Tensor<float>> latents;
        if (!fugue_stages::get_list(file, "latent", latents))
            refuse("cannot read latent '" + path + "': " + file.error);
        const auto index = dgl::get_int(args, "index");
        if (index < 0 || index >= static_cast<int64_t>(latents.size()) || latents[static_cast<size_t>(index)].empty())
            refuse("the latent '" + path + "' has no item " + std::to_string(index));
        return dgl::tensor_value(std::move(latents[static_cast<size_t>(index)]));
    }

    static dgl::Value load_clip(dgl::Values& args, dgl::RunState&) {
        const auto path = dgl::get_string(args, "path");
        require_file_access(path, SD_DGL_FILE_READ);
        return dgl::tensor_value(read_y4m(path, int(dgl::get_int(args, "width")), int(dgl::get_int(args, "height")), int(dgl::get_int(args, "frames"))));
    }

    // Strength as a noise level (the option upstream's pipeline had before DGL): the schedule from
    // the first sigma at or below the level, with the level itself as the first one when asked.
    static dgl::Value noise_level_sigmas(dgl::Values& args, dgl::RunState&) {
        const auto predict      = dgl::object<dgl::ops::Predictor>(args, "predict");
        const auto* denoiser    = dgl::ops::schedule_predictor(*predict).resolved_parameterization().native();
        const auto& full        = dgl::sigma_tensor(args, "sigmas").values();
        const float target      = const_cast<Denoiser*>(denoiser)->noise_level_to_sigma(float(dgl::get_number(args, "strength")));
        size_t start            = 0;
        for (size_t i = 0; i < full.size(); ++i)
            if (full[i] <= target) {
                start = i;
                break;
            }
        start = std::min(start, full.size() - 2);  // Leave at least one step
        std::vector<float> sigmas(full.begin() + static_cast<std::ptrdiff_t>(start), full.end());
        if (target > 0 && dgl::get_bool(args, "force_first") && target > sigmas[1])
            sigmas[0] = target;
        return dgl::tensor_value(sd::Tensor<float>::from_vector(std::move(sigmas)));
    }

    void register_operations(dgl::Registry& registry) {
        using dgl::Kind;
        registry["fugue::load_condition"] = dgl::make_operation(
            {dgl::required("path", Kind::String), dgl::required("set", Kind::String), dgl::required("version", Kind::String)},
            dgl::condition_signature(), load_condition);
        registry["fugue::load_latent"] = dgl::make_operation(
            {dgl::required("path", Kind::String), dgl::required("version", Kind::String), dgl::integer("index", 0, INT32_MAX, 0),
             dgl::integer("width", 0, INT32_MAX, 0), dgl::integer("height", 0, INT32_MAX, 0), dgl::integer("frames", 0, INT32_MAX, 0)},
            {Kind::Tensor}, load_latent);
        registry["fugue::load_clip"] = dgl::make_operation(
            {dgl::required("path", Kind::String), dgl::integer("width", 1, INT32_MAX), dgl::integer("height", 1, INT32_MAX), dgl::integer("frames", 1, INT32_MAX)},
            {Kind::Tensor}, load_clip);
        registry["fugue::noise_level_sigmas"] = dgl::make_operation(
            {dgl::required("predict", Kind::Predictor), dgl::required("sigmas", Kind::Tensor), dgl::number("strength", 0, 1),
             dgl::optional("force_first", Kind::Boolean, false)},
            {Kind::Tensor}, noise_level_sigmas);
    }

}  // namespace sd::api::fugue_stage
