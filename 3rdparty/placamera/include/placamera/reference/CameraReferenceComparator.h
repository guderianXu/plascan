#pragma once

#include "placamera/reference/CameraReferenceObservation.h"

#include <placamera/model.h>

#include <optional>
#include <string>

namespace placamera::reference
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
        static CameraReferenceComparison compare(const placamera::RasterModel& instance,
                                                 const ResolvedCameraReference& reference);
    };

} // namespace placamera::reference
