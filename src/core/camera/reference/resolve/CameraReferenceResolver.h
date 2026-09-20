#pragma once

#include "camera/reference/model/CameraReferenceObservation.h"
#include "coordinate_system/transform/CoordinateTransformService.h"

#include <string>

namespace xjw::camera_reference
{

    struct CameraReferenceResolveOptions
    {
        std::string orientationConvention;
    };

    class CameraReferenceResolver
    {
    public:
        explicit CameraReferenceResolver(const xjw::coordinate_system::CoordinateTransformService* transforms)
            : _transforms(transforms)
        {
        }

        ResolvedCameraReference resolve(const CameraReferenceObservation& observation,
                                        const xjw::coordinate_system::CoordinateFrameId& targetFrame,
                                        const CameraReferenceResolveOptions& options) const;

    private:
        const xjw::coordinate_system::CoordinateTransformService* _transforms = nullptr;
    };

} // namespace xjw::camera_reference
