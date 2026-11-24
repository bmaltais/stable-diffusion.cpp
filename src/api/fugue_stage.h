// Fugue's stage split on the DGL engine (github.com/bmaltais/fugue, helper/sdcpp.patch): encode,
// sample and decode run as separate calls. The library renders the whole program of a request as
// always; this module cuts it at named values, replacing what a stage reads from a file and keeping
// only the statements the stage's result needs. A request without a `fugue.*` option, or the
// noise-level reading of strength, never comes here.
#ifndef __SD_API_FUGUE_STAGE_H__
#define __SD_API_FUGUE_STAGE_H__

#include <memory>

#include "api/generation.h"
#include "dgl/registry.h"

namespace sd::api::fugue_stage {

    bool wanted(const ImageConfig& image);
    bool wanted(const VideoConfig& video);

    sd_status_t run_image(GenerationContext& context, std::shared_ptr<sd_opt_t> options, OptConfig config, int64_t seed, sd_result_t** out);
    sd_status_t run_video(GenerationContext& context, std::shared_ptr<sd_opt_t> options, const OptConfig& config, int64_t seed, sd_result_t** out);

    // The fugue:: built-ins the cut programs call.
    void register_operations(dgl::Registry& registry);

}  // namespace sd::api::fugue_stage

#endif  // __SD_API_FUGUE_STAGE_H__
