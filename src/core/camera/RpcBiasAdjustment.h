#pragma once

#include "RpcCameraModel.h"

#include <cstddef>
#include <string>
#include <vector>

namespace xjw
{

    enum class RpcImageCorrectionModel
    {
        Translation,
        Affine
    };

    struct RpcControlPointObservation
    {
        RpcCameraModel::GeodeticCoordinate ground;
        CameraImageCoordinate observedImage;
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
        RpcCameraModel::ImageCorrection correction;
        double rmsBeforePixels = 0.0;
        double rmsAfterPixels = 0.0;
        double maximumResidualPixels = 0.0;
        std::size_t observationCount = 0;
        int iterations = 0;
    };

    /**
     * @brief Estimate an image-space RPC correction from geodetic control points.
     *
     * The input camera's existing correction is ignored while forming raw RPC
     * residuals. The returned correction can be installed with
     * RpcCameraModel::setImageCorrection().
     */
    bool estimateRpcImageCorrection(const RpcCameraModel& camera,
                                    const std::vector<RpcControlPointObservation>& observations,
                                    RpcBiasAdjustmentResult* result,
                                    std::string* errorMessage = nullptr);

    bool estimateRpcImageCorrection(const RpcCameraModel& camera,
                                    const std::vector<RpcControlPointObservation>& observations,
                                    RpcBiasAdjustmentResult* result,
                                    const RpcBiasAdjustmentOptions& options,
                                    std::string* errorMessage = nullptr);

} // namespace xjw
