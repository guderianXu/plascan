#pragma once

#include "camera/reference/model/CameraReferenceObservation.h"
#include "camera/core/model/CameraInstance.h"

#include <optional>
#include <string>

namespace xjw::camera_reference
{

    enum class CameraReferenceComparisonStatus
    {
        Compared,
        UnresolvedReference,
        RegionModelWithoutStaticPose,
        FrameMismatch,
    };

    struct CameraReferenceComparison
    {
        CameraReferenceComparisonStatus status = CameraReferenceComparisonStatus::UnresolvedReference;
        std::optional<double> positionError;
        std::optional<double> rotationErrorRadians;
        std::string reason;

        bool compared() const noexcept
        {
            return status == CameraReferenceComparisonStatus::Compared;
        }
    };

    class CameraReferenceComparator
    {
    public:
        static CameraReferenceComparison compare(const camera_core::CameraInstance& instance,
                                                 const ResolvedCameraReference& reference);
    };

} // namespace xjw::camera_reference
