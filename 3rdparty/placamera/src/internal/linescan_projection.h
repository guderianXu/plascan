#pragma once

#include "placamera/linescan_camera.h"

namespace placamera::internal
{

    EvaluationResult<FocalPlaneCoordinate> lineScanPixelToUndistortedFocal(const LineScanOptics& optics,
                                                                           LineScanPixelConvention convention,
                                                                           double sample,
                                                                           const EvaluationOptions& options);

    EvaluationResult<LineScanDetectorProjection> lineScanUndistortedFocalToPixel(const LineScanOptics& optics,
                                                                                 LineScanPixelConvention convention,
                                                                                 const FocalPlaneCoordinate& focal,
                                                                                 const EvaluationOptions& options);

    EvaluationResult<LineScanProjectionDetails> projectLineScanAtLine(const LineScanModel& model,
                                                                      const GroundCoordinate& ground,
                                                                      double line,
                                                                      const EvaluationOptions& options);
    EvaluationResult<LineScanProjectionDetails> projectLineScanAtLine(const LineScanModel& model,
                                                                      const GroundCoordinate& ground,
                                                                      double line,
                                                                      const LineScanTrajectoryBias& bias,
                                                                      const EvaluationOptions& options);

    EvaluationResult<LineScanProjectionDetails>
    projectLineScan(const LineScanModel& model, const GroundCoordinate& ground, const EvaluationOptions& options);

    EvaluationResult<ImagingLocus>
    lineScanImagingLocus(const LineScanModel& model, const ImageCoordinate& image, const EvaluationOptions& options);

} // namespace placamera::internal
