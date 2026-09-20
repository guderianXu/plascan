#pragma once

#include "CameraCapabilities.h"
#include "../model/CameraInstanceSet.h"

#include <optional>
#include <string>
#include <vector>

namespace xjw::camera_core
{

    /**
     * @brief Numeric operations that consume a canonical camera-instance set.
     *
     * The operation is deliberately independent from a concrete camera model.  A
     * model is eligible when it exposes the capabilities required by the
     * operation and its instances satisfy the operation's frame contract.
     */
    enum class CameraOperation
    {
        StaticSfM,
        BundleAdjustment,
        DenseMvs,
        OrthoProjection,
        RpcAerialTriangulation,
        PushbroomAerialTriangulation,
        ExternalPoseReference,
    };

    struct CameraOperationRequirements
    {
        CapabilitySet capabilities;
        bool requireCommonWorldFrame = true;
    };

    struct CameraOperationPlan
    {
        struct Failure
        {
            ImageId image;
            std::string reason;
        };

        CameraOperation operation = CameraOperation::StaticSfM;
        CameraOperationRequirements requirements;
        std::optional<xjw::coordinate_system::CoordinateFrameId> commonWorldFrame;
        std::string inputError;
        std::vector<Failure> capabilityFailures;
        std::vector<Failure> frameFailures;
        std::vector<Failure> failures;

        bool ok() const noexcept
        {
            return inputError.empty() && failures.empty();
        }

        std::string failureMessage() const;
    };

    const CameraOperationRequirements& requirementsFor(CameraOperation operation) noexcept;

    const char* cameraOperationName(CameraOperation operation) noexcept;

    CameraOperationPlan planCameraOperation(const CameraInstanceSet& instances, CameraOperation operation);

} // namespace xjw::camera_core
