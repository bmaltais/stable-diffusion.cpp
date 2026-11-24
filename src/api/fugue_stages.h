// Fugue's stage split (github.com/bmaltais/fugue, helper/sdcpp.patch): the files that carry
// conditioning and latents between the stages of a generation, so encode, sample and decode
// can run as separate calls. The `fugue.*` image options (src/api/options.cpp) choose what is
// read, written and where a run stops (api/fugue_stage.cpp).
//
// A file is in the safetensors layout: an 8-byte header length, a JSON header, raw tensors.
// Every list field records its length in the metadata, so an empty tensor inside a list keeps
// its place (it is stored with shape [0]); a tensor is read back only as the type it was stored
// as.
#ifndef __SD_API_FUGUE_STAGES_H__
#define __SD_API_FUGUE_STAGES_H__

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "conditioning/conditioner.hpp"
#include "core/tensor.hpp"
#include "json.hpp"

namespace fugue_stages {

    struct Blob {
        std::string dtype;  // "F32" or "I32"
        std::vector<int64_t> shape;
        std::vector<uint8_t> bytes;
    };

    // The tensors of a file by name, and its string metadata.
    struct File {
        std::map<std::string, Blob> blobs;
        std::map<std::string, std::string> meta;
        std::string error;  // why reading failed, else empty

        bool fail(const std::string& why) {
            if (error.empty()) {
                error = why;
            }
            return false;
        }
    };

    template <typename T>
    inline const char* dtype_of();
    template <>
    inline const char* dtype_of<float>() { return "F32"; }
    template <>
    inline const char* dtype_of<int32_t>() { return "I32"; }

    // An empty tensor is stored only when `keep_empty` (a list item), with shape [0].
    template <typename T>
    inline void put(File& f, const std::string& name, const sd::Tensor<T>& t, bool keep_empty = false) {
        if (t.empty() && !keep_empty) {
            return;
        }
        Blob b;
        b.dtype = dtype_of<T>();
        b.shape = t.empty() ? std::vector<int64_t>{0} : t.shape();
        b.bytes.resize(t.empty() ? 0 : static_cast<size_t>(t.numel()) * sizeof(T));
        if (!b.bytes.empty()) {
            std::memcpy(b.bytes.data(), t.data(), b.bytes.size());
        }
        f.blobs[name] = std::move(b);
    }

    // Reads `name` into `out`; absent leaves `out` empty. Fails on a type or size mismatch.
    template <typename T>
    inline bool get(File& f, const std::string& name, sd::Tensor<T>& out, bool required = false) {
        out      = {};
        auto it  = f.blobs.find(name);
        if (it == f.blobs.end()) {
            return required ? f.fail("missing tensor " + name) : true;
        }
        const Blob& b = it->second;
        if (b.dtype != dtype_of<T>()) {
            return f.fail("tensor " + name + " is " + b.dtype + ", expected " + dtype_of<T>());
        }
        uint64_t numel = 1;
        for (int64_t d : b.shape) {
            if (d < 0 || (d > 0 && numel > (UINT64_MAX / sizeof(T)) / static_cast<uint64_t>(d))) {
                return f.fail("tensor " + name + " has a bad shape");
            }
            numel *= static_cast<uint64_t>(d);
        }
        if (numel * sizeof(T) != b.bytes.size()) {
            return f.fail("tensor " + name + " has the wrong size for its shape");
        }
        if (numel == 0) {
            return true;
        }
        std::vector<T> data(static_cast<size_t>(numel));
        std::memcpy(data.data(), b.bytes.data(), b.bytes.size());
        out = sd::Tensor<T>(b.shape, std::move(data));
        return true;
    }

    inline void put_count(File& f, const std::string& name, size_t n) { f.meta[name + ".count"] = std::to_string(n); }

    inline bool get_count(File& f, const std::string& name, size_t& n) {
        n       = 0;
        auto it = f.meta.find(name + ".count");
        if (it == f.meta.end()) {
            return true;
        }
        try {
            n = static_cast<size_t>(std::stoull(it->second));
        } catch (...) {
            return f.fail("bad count for " + name);
        }
        return true;
    }

    inline void put_list(File& f, const std::string& name, const std::vector<sd::Tensor<float>>& list) {
        put_count(f, name, list.size());
        for (size_t i = 0; i < list.size(); i++) {
            put(f, name + "." + std::to_string(i), list[i], true);
        }
    }

    inline bool get_list(File& f, const std::string& name, std::vector<sd::Tensor<float>>& list) {
        list.clear();
        size_t n = 0;
        if (!get_count(f, name, n)) {
            return false;
        }
        list.resize(n);
        for (size_t i = 0; i < n; i++) {
            if (!get(f, name + "." + std::to_string(i), list[i], true)) {
                return false;
            }
        }
        return true;
    }

    inline bool write_file(const std::string& path, const File& f) {
        nlohmann::json header = nlohmann::json::object();
        nlohmann::json md     = nlohmann::json::object();
        for (const auto& [k, v] : f.meta) {
            md[k] = v;
        }
        header["__metadata__"] = md;
        uint64_t offset        = 0;
        for (const auto& [name, b] : f.blobs) {
            header[name] = {{"dtype", b.dtype}, {"shape", b.shape}, {"data_offsets", {offset, offset + b.bytes.size()}}};
            offset += b.bytes.size();
        }
        std::string head = header.dump();
        while (head.size() % 8 != 0) {
            head.push_back(' ');
        }
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        uint64_t n = head.size();
        out.write(reinterpret_cast<const char*>(&n), 8);
        out.write(head.data(), static_cast<std::streamsize>(head.size()));
        for (const auto& [name, b] : f.blobs) {
            out.write(reinterpret_cast<const char*>(b.bytes.data()), static_cast<std::streamsize>(b.bytes.size()));
        }
        return static_cast<bool>(out);
    }

    inline bool read_file(const std::string& path, File& f) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return f.fail("cannot open " + path);
        }
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), 8);
        if (!in || n == 0 || n > (64u << 20)) {
            return f.fail(path + " is not a tensor file");
        }
        std::string head(n, '\0');
        in.read(head.data(), static_cast<std::streamsize>(n));
        nlohmann::json header = nlohmann::json::parse(head, nullptr, false);
        if (!in || header.is_discarded() || !header.is_object()) {
            return f.fail(path + " has a bad header");
        }
        const std::streamoff base = 8 + static_cast<std::streamoff>(n);
        try {
            for (auto it = header.begin(); it != header.end(); ++it) {
                if (it.key() == "__metadata__") {
                    for (auto m = it.value().begin(); m != it.value().end(); ++m) {
                        if (m.value().is_string()) {
                            f.meta[m.key()] = m.value().get<std::string>();
                        }
                    }
                    continue;
                }
                Blob b;
                b.dtype            = it.value().at("dtype").get<std::string>();
                b.shape            = it.value().at("shape").get<std::vector<int64_t>>();
                const uint64_t beg = it.value().at("data_offsets").at(0).get<uint64_t>();
                const uint64_t end = it.value().at("data_offsets").at(1).get<uint64_t>();
                if (end < beg) {
                    return f.fail(path + ": bad offsets for " + it.key());
                }
                b.bytes.resize(end - beg);
                in.seekg(base + static_cast<std::streamoff>(beg));
                in.read(reinterpret_cast<char*>(b.bytes.data()), static_cast<std::streamsize>(b.bytes.size()));
                if (!in) {
                    return f.fail(path + " is cut short");
                }
                f.blobs[it.key()] = std::move(b);
            }
        } catch (const std::exception& e) {
            return f.fail(path + " has a bad header: " + e.what());
        }
        return true;
    }

    // Every field of SDCondition, under `p`. A field added upstream must be added to put_cond and
    // get_cond: this fails to compile when SDCondition changes size (as at a move of the pin).
    static_assert(sizeof(SDCondition) == 552, "SDCondition changed: carry its new fields");
    inline void put_cond(File& f, const std::string& p, const SDCondition& c) {
        put(f, p + ".c_crossattn", c.c_crossattn);
        put(f, p + ".c_vector", c.c_vector);
        put(f, p + ".c_concat", c.c_concat);
        put(f, p + ".c_t5_ids", c.c_t5_ids);
        put(f, p + ".c_t5_weights", c.c_t5_weights);
        put(f, p + ".c_input_ids", c.c_input_ids);
        put(f, p + ".c_position_ids", c.c_position_ids);
        put(f, p + ".c_token_types", c.c_token_types);
        put(f, p + ".c_vinput_mask", c.c_vinput_mask);
        put_count(f, p + ".c_image_embeds", c.c_image_embeds.size());
        for (size_t i = 0; i < c.c_image_embeds.size(); i++) {
            const std::string name = p + ".c_image_embeds." + std::to_string(i);
            f.meta[name + ".index"] = std::to_string(c.c_image_embeds[i].first);
            put(f, name, c.c_image_embeds[i].second, true);
        }
        put_list(f, p + ".c_ref_images", c.c_ref_images);
        put_list(f, p + ".c_ref_audios", c.c_ref_audios);
        put_list(f, p + ".extra_c_crossattns", c.extra_c_crossattns);
        if (!c.c_reference_blocks.empty()) {
            std::vector<int32_t> rows;
            for (const auto& block : c.c_reference_blocks) {
                rows.push_back(static_cast<int32_t>(block.kind));
                rows.push_back(block.video_index);
                rows.push_back(block.audio_index);
            }
            const int64_t n = static_cast<int64_t>(c.c_reference_blocks.size());
            put(f, p + ".c_reference_blocks", sd::Tensor<int32_t>({3, n}, std::move(rows)));
        }
    }

    inline bool get_cond(File& f, const std::string& p, SDCondition& c) {
        c = {};
        bool ok = get(f, p + ".c_crossattn", c.c_crossattn) && get(f, p + ".c_vector", c.c_vector) &&
                  get(f, p + ".c_concat", c.c_concat) && get(f, p + ".c_t5_ids", c.c_t5_ids) &&
                  get(f, p + ".c_t5_weights", c.c_t5_weights) && get(f, p + ".c_input_ids", c.c_input_ids) &&
                  get(f, p + ".c_position_ids", c.c_position_ids) && get(f, p + ".c_token_types", c.c_token_types) &&
                  get(f, p + ".c_vinput_mask", c.c_vinput_mask);
        size_t n = 0;
        if (!ok || !get_count(f, p + ".c_image_embeds", n)) {
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            const std::string name = p + ".c_image_embeds." + std::to_string(i);
            sd::Tensor<float> t;
            int index = 0;
            try {
                index = std::stoi(f.meta.at(name + ".index"));
            } catch (...) {
                return f.fail("bad index for " + name);
            }
            if (!get(f, name, t, true)) {
                return false;
            }
            c.c_image_embeds.emplace_back(index, std::move(t));
        }
        if (!get_list(f, p + ".c_ref_images", c.c_ref_images) || !get_list(f, p + ".c_ref_audios", c.c_ref_audios) ||
            !get_list(f, p + ".extra_c_crossattns", c.extra_c_crossattns)) {
            return false;
        }
        sd::Tensor<int32_t> blocks;
        if (!get(f, p + ".c_reference_blocks", blocks)) {
            return false;
        }
        if (!blocks.empty()) {
            if (blocks.shape().size() != 2 || blocks.shape()[0] != 3) {
                return f.fail("bad shape for " + p + ".c_reference_blocks");
            }
            const int32_t* rows = blocks.data();
            for (int64_t i = 0; i < blocks.shape()[1]; i++) {
                MiniMaxH3ReferenceBlock block;
                block.kind        = static_cast<MiniMaxH3ReferenceKind>(rows[3 * i]);
                block.video_index = rows[3 * i + 1];
                block.audio_index = rows[3 * i + 2];
                c.c_reference_blocks.push_back(block);
            }
        }
        return true;
    }

    inline void put_float(File& f, const std::string& name, float v) {
        put(f, name, sd::Tensor<float>({1}, std::vector<float>{v}));
    }

    inline bool get_float(File& f, const std::string& name, float& v) {
        sd::Tensor<float> t;
        if (!get(f, name, t)) {
            return false;
        }
        if (!t.empty()) {
            v = t.data()[0];
        }
        return true;
    }

}  // namespace fugue_stages

#endif  // __SD_API_FUGUE_STAGES_H__
