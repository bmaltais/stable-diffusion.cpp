#ifndef __SD_DGL_DIAGNOSTIC_H__
#define __SD_DGL_DIAGNOSTIC_H__

#include <stdexcept>
#include <string>

#include "stable-diffusion.h"

namespace sd::dgl {
    struct Error : std::runtime_error {
        sd_status_t status;

        Error(sd_status_t status, const std::string& message)
            : std::runtime_error(message), status(status) {}
    };

    // The error for a model whose compute returned nothing: `what` and the reason the last
    // failed computation on this thread recorded (sd::ComputeFailure), out of memory when it was.
    [[noreturn]] void throw_compute_failure(const std::string& what);
}  // namespace sd::dgl

#endif  // __SD_DGL_DIAGNOSTIC_H__
