#pragma once

#include "placamera/reference/CameraReferenceObservation.h"
#include <placoordinate/transform/CoordinateTransformService.h>

#include <string>

namespace placamera::reference
{

    struct CameraReferenceResolveOptions
    {
        std::string orientationConvention;
    };

    class CameraReferenceResolver
    {
    public:
        explicit CameraReferenceResolver(const placoordinate::CoordinateTransformService* transforms)
            : _transforms(transforms)
        {
        }

        ResolvedCameraReference resolve(const CameraReferenceObservation& observation,
                                        const placoordinate::CoordinateFrameId& targetFrame,
                                        const CameraReferenceResolveOptions& options) const;

    private:
        const placoordinate::CoordinateTransformService* _transforms = nullptr;
    };

} // namespace placamera::reference
