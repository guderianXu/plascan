#pragma once

#include "LineScanDefinition.h"

namespace xjw::camera_models::linescan
{

    struct FocalPlaneCoordinate
    {
        double xMillimeters = 0.0;
        double yMillimeters = 0.0;
    };

    /** Bidirectional detector/focal-plane mapping for a line-scan definition. */
    class LineScanOpticsTransform
    {
    public:
        static bool pixelToUndistortedFocal(const LineScanDefinition& definition,
                                            double sample,
                                            FocalPlaneCoordinate* focal);

        static bool undistortedFocalToPixel(const LineScanDefinition& definition,
                                            const FocalPlaneCoordinate& focal,
                                            double* sample,
                                            double* detectorLineResidualPixels);
    };

} // namespace xjw::camera_models::linescan
