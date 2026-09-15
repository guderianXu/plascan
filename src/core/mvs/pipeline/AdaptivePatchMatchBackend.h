#pragma once
#include "MvsPipelineInternals.h"
namespace xjw::mvs::pipeline_detail
{

    class AdaptivePatchMatchBackend final : public IPatchMatchBackend
    {
    public:
        explicit AdaptivePatchMatchBackend(int ref_idx) : _refIdx(ref_idx)
        {
        }

        bool
        estimate(const PatchMatchBackendRequest& request, DepthLevelResult& result, std::string* error_message) override
        {
            const std::string stage_label = "Level " + std::to_string(request.levelConfig.level) + " PatchMatch";
            cv::Mat confidence;
            const cv::Mat* hint = request.prior && !request.prior->center.empty() ? &request.prior->center : nullptr;
            const cv::Mat* radius = request.prior && !request.prior->radius.empty() ? &request.prior->radius : nullptr;
            LOG_DEBUG("[MVS][帧 %d][PatchMatch] %s ds=%d iterations=%d patch=%d",
                      _refIdx,
                      stage_label.c_str(),
                      request.levelConfig.patchMatch.downsampleFactor,
                      request.levelConfig.patchMatch.numIterations,
                      request.levelConfig.patchMatch.patchHalf * 2 + 1);

            PatchMatchAuxiliaryInput auxiliary_input;
            auxiliary_input.sourceDepthMaps = request.sourceDepthMaps.empty() ? nullptr : &request.sourceDepthMaps;
            PatchMatchAuxiliaryOutput auxiliary_output;
            auxiliary_output.photometricSourceMask = &result.photometricSourceMask;

            if (!estimatePatchMatchWithAdaptiveCuda(
                    stage_label.c_str(),
                    _refIdx,
                    request.referenceImage,
                    request.sourceImages,
                    request.referenceCamera,
                    request.sourceCameras,
                    request.zNear,
                    request.zFar,
                    request.levelConfig.patchMatch,
                    result.depth,
                    &confidence,
                    error_message,
                    hint,
                    radius,
                    request.referenceValidMask.empty() ? nullptr : &request.referenceValidMask,
                    request.sourceValidMasks.empty() ? nullptr : &request.sourceValidMasks,
                    &auxiliary_input,
                    &auxiliary_output))
            {
                return false;
            }

            result.level = request.levelConfig.level;
            result.downsampleFactor = request.levelConfig.patchMatch.downsampleFactor;
            result.confidence = confidence;
            result.validMask = result.depth > 0.0f;
            result.supportCount = cv::Mat(result.depth.size(), CV_16U, cv::Scalar(0));
            for (int row = 0; row < result.depth.rows; ++row)
            {
                const std::int32_t* mask_row = result.photometricSourceMask.empty()
                                                   ? nullptr
                                                   : result.photometricSourceMask.ptr<std::int32_t>(row);
                std::uint16_t* support_row = result.supportCount.ptr<std::uint16_t>(row);
                const std::uint8_t* valid_row = result.validMask.ptr<std::uint8_t>(row);
                for (int column = 0; column < result.depth.cols; ++column)
                {
                    if (valid_row[column] == 0)
                    {
                        continue;
                    }
                    support_row[column] = mask_row ? static_cast<std::uint16_t>(selectedSourceCount(
                                                         static_cast<std::uint32_t>(mask_row[column])))
                                                   : static_cast<std::uint16_t>(request.sourceImages.size());
                }
            }
            result.uncertainty = cv::Mat(result.depth.size(), CV_32F, cv::Scalar(0.0f));
            for (int row = 0; row < result.depth.rows; ++row)
            {
                for (int column = 0; column < result.depth.cols; ++column)
                {
                    const float depth = result.depth.at<float>(row, column);
                    if (depth <= 0.0f || !std::isfinite(depth))
                    {
                        continue;
                    }
                    const float confidence_value =
                        confidence.empty() ? 0.5f : std::clamp(confidence.at<float>(row, column), 0.0f, 1.0f);
                    result.uncertainty.at<float>(row, column) =
                        std::max(0.001f, depth * 0.05f * (1.0f - confidence_value));
                }
            }
            return true;
        }

    private:
        int _refIdx = -1;
    };
} // namespace xjw::mvs::pipeline_detail
