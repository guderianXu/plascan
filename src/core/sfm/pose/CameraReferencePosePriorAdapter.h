#pragma once

#include "BundleAdjustProblem.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/reference/resolve/CameraReferencePosePrior.h"

#include <string>
#include <vector>

namespace xjw
{

    struct CameraReferencePosePriorAdapterResult
    {
        bool valid = false;
        std::vector<BACameraPosePrior> priors;
        std::size_t matchedReferenceCount = 0;
        std::size_t ignoredReferenceCount = 0;
        std::vector<camera_core::ImageId> ignoredReferenceImages;
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
     * Align image-keyed external pose priors with the ordered numeric camera set.
     *
     * References outside the current solve (for example, a local BA window) are
     * ignored.  A matched reference must use the same world frame as its numeric
     * camera; no transform is inferred at this boundary.  The output is always
     * indexed like `cameras` and leaves unmatched entries disabled.
     */
    class CameraReferencePosePriorAdapter final
    {
    public:
        static CameraReferencePosePriorAdapterResult
        toBundleAdjustPriors(const std::vector<camera_models::frame_pinhole::FramePinholeNumericState>& cameras,
                             const std::vector<camera_reference::ResolvedCameraPosePrior>& references,
                             double defaultPositionSigmaMeters = 1.0,
                             double defaultRotationSigmaDegrees = 2.0);
    };

} // namespace xjw
