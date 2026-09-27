#pragma once

#include "placamera/rpc_camera.h"

#include <cstddef>
#include <vector>

namespace placamera
{

    enum class RpcImageCorrectionModel
    {
        Translation,
        Affine,
    };

    struct RpcControlPointObservation
    {
        GeodeticCoordinate ground;
        ImageCoordinate observedImage;
        double weight = 1.0;
    };

    struct RpcBiasAdjustmentOptions
    {
        RpcImageCorrectionModel model = RpcImageCorrectionModel::Affine;
        bool robust = true;
        double huberThresholdPixels = 2.0;
        int maximumIterations = 10;
    };

    struct RpcBiasAdjustmentResult
    {
        RpcImageCorrection correction;
        double rmsBeforePixels = 0.0;
        double rmsAfterPixels = 0.0;
        double maximumResidualPixels = 0.0;
        std::size_t observationCount = 0;
        int iterations = 0;
    };

    EvaluationResult<RpcBiasAdjustmentResult>
    estimateRpcImageCorrection(const RpcModel& camera,
                               const std::vector<RpcControlPointObservation>& observations,
                               const RpcBiasAdjustmentOptions& options = {});

    struct RpcIntersectionOptions
    {
        double pixelTolerance = 1.0e-5;
        double positionToleranceMeters = 1.0e-3;
        double derivativeStepMeters = 0.5;
        int maximumIterations = 30;
    };

    struct RpcIntersectionResult
    {
        GroundCoordinate cartesian;
        GeodeticCoordinate geodetic;
        double reprojectionRmsPixels = 0.0;
        int iterations = 0;
    };

    EvaluationResult<RpcIntersectionResult> intersectRpc(const RpcModel& first,
                                                         const ImageCoordinate& firstImage,
                                                         const RpcModel& second,
                                                         const ImageCoordinate& secondImage,
                                                         const RpcIntersectionOptions& options = {});

} // namespace placamera
