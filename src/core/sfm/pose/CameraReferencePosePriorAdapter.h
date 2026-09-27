#pragma once

#include "placamera/reference/CameraReferencePosePrior.h"

#include <plabundle/constraints.h>
#include <placamera/types.h>

#include <optional>
#include <string>
#include <vector>

namespace xjw
{

    struct CameraReferenceTarget
    {
        placamera::ImageId imageId;
        placamera::FrameId worldFrame;
    };

    struct CameraReferencePosePriorAdapterResult
    {
        bool valid = false;
        std::vector<std::optional<plabundle::CameraPosePrior>> priors;
        std::size_t matchedReferenceCount = 0;
        std::size_t ignoredReferenceCount = 0;
        std::vector<placamera::ImageId> ignoredReferenceImages;
        std::string commonTransformProvenanceHash;
        std::string error;

        bool ok() const noexcept
        {
            return valid;
        }

        bool hasEnabledPriors() const noexcept
        {
            return matchedReferenceCount > 0;
        }
    };

    /**
     * Align image-keyed external pose priors with an ordered camera identity set.
     *
     * References outside the current solve (for example, a local BA window) are
     * ignored. A matched reference must use the same world frame as its target
     * camera; no transform is inferred at this boundary. The output is always
     * indexed like `targets` and leaves unmatched entries disabled.
     */
    class CameraReferencePosePriorAdapter final
    {
    public:
        static CameraReferencePosePriorAdapterResult
        toBundleAdjustPriors(const std::vector<CameraReferenceTarget>& targets,
                             const std::vector<placamera::reference::ResolvedCameraPosePrior>& references,
                             double defaultPositionSigmaMeters = 1.0,
                             double defaultRotationSigmaDegrees = 2.0);
    };

} // namespace xjw
