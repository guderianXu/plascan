#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;
    FusionFrameInput MvsPipelineService::buildFusionFrame(const DepthFrameResult& res)
    {
        FusionFrameInput frame;
        if (!res.eligibleForFusion())
        {
            return frame;
        }
        frame.cameraModel =
            res.cameraModel.isValid() ? res.cameraModel : mvsPinholeCamera(_views[res.refViewIdx].camera);
        const CameraView& view = _views[res.refViewIdx];
        frame.sourceCamera = view.camera;
        frame.viewIndex = res.refViewIdx;
        frame.sourceImageIndices = res.sourceViewIndices;
        frame.imgW = res.depthMap ? res.depthMap->cols : 0;
        frame.imgH = res.depthMap ? res.depthMap->rows : 0;
        frame.imagePath = mvsRasterPath(view);
        {
            std::lock_guard<std::mutex> lock(_preparedRasterArtifactsMutex);
            if (res.refViewIdx >= 0 && res.refViewIdx < static_cast<int>(_preparedRasterArtifacts.size()))
            {
                const MvsPreparedRasterArtifact& prepared =
                    _preparedRasterArtifacts[static_cast<std::size_t>(res.refViewIdx)];
                if (!prepared.imagePath.empty() && prepared.camera.isValid())
                {
                    frame.imagePath = prepared.imagePath;
                    frame.sourceCamera = prepared.camera;
                }
            }
        }

        if (!res.depthMap || res.depthMap->empty())
            return frame;

        const cv::Mat& rawDepth = *res.depthMap;
        cv::Mat filteredDepth = rawDepth.clone();
        cv::Mat filteredConfidence = res.confidence ? res.confidence->clone() : cv::Mat();

        if (res.depthPostprocessApplied)
        {
            frame.depthPostprocess = res.depthPostprocess;
        }
        else
        {
            FusionConfig fusion_config = _config.fusion;
            const DepthFilterSettings filter_settings =
                depthFilterSettings(_effectiveDepthFilterMode, static_cast<int>(res.sourceViewIndices.size()));
            fusion_config.localDepthOutlierRelThresh = filter_settings.localDepthOutlierRelThreshold;
            fusion_config.minSpeckleComponentArea = filter_settings.minComponentArea;
            fusion_config.minConsistentViews = filter_settings.minConsistentViews;
            fusion_config.confidenceThresh = depthConfidenceThresholds(_effectiveSceneProfile,
                                                                       _effectiveDepthFilterMode,
                                                                       static_cast<int>(res.sourceViewIndices.size()),
                                                                       _config.patchMatch.confidenceThresh,
                                                                       fusion_config.confidenceThresh)
                                                 .fusion;
            const DepthPostProcessEvidence postprocess_evidence = depthPostProcessEvidence(res);
            frame.depthPostprocess = DepthPostprocessor::postprocessFusionDepthMap(filteredDepth,
                                                               filteredConfidence,
                                                               fusion_config,
                                                               res.refViewIdx,
                                                               static_cast<int>(_views.size()),
                                                               nullptr,
                                                               &postprocess_evidence,
                                                               res.preparedRasterSize);
        }

        frame.depthMap = filteredDepth;
        frame.confidence = filteredConfidence;
        frame.normalMap = res.normalMap ? res.normalMap->clone() : cv::Mat();
        frame.geometrySupportCount = res.geometrySupportCount ? res.geometrySupportCount->clone() : cv::Mat();
        frame.validMask = res.validMask ? res.validMask->clone() : (filteredDepth > 0.0f);
        if (!frame.validMask.empty())
        {
            frame.validMask.setTo(cv::Scalar(0), filteredDepth <= 0.0f);
        }
        if (!frame.geometrySupportCount.empty())
        {
            frame.geometrySupportCount.setTo(cv::Scalar(0), filteredDepth <= 0.0f);
        }
        if (!frame.normalMap.empty())
        {
            frame.normalMap.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), filteredDepth <= 0.0f);
        }
        frame.confidence.release();
        return frame;
    }
} // namespace xjw::mvs
