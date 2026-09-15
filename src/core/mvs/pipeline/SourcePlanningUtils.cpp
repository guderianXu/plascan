#include "MvsPipelineInternals.h"

namespace xjw::mvs::pipeline_detail
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    FramePinholeCamera mvsPinholeCamera(const FramePinholeCamera& camera)
    {
        FramePinholeCamera result = camera.normalizedForPositiveDepth();
        result.setDistortion(FramePinholeCamera::Distortion{});
        return result;
    }

    cv::Mat restoreNativePyramidArtifact(const cv::Mat& artifact, const cv::Size& working_size)
    {
        if (artifact.empty() || artifact.size() == working_size)
        {
            return artifact;
        }

        cv::Mat restored;
        cv::resize(artifact, restored, working_size, 0.0, 0.0, cv::INTER_NEAREST);
        return restored;
    }

    QString normalizedMvsPathKey(const std::string& path)
    {
        const QString rawPath = QString::fromStdString(path).trimmed();
        if (rawPath.isEmpty())
        {
            return QString();
        }

        const QFileInfo info(rawPath);
        QString absolutePath = info.exists() ? info.canonicalFilePath() : info.absoluteFilePath();
        if (absolutePath.isEmpty())
        {
            absolutePath = rawPath;
        }

        return QDir::cleanPath(absolutePath).replace(QLatin1Char('\\'), QLatin1Char('/')).toCaseFolded();
    }

    const std::string& mvsRasterPath(const CameraView& view)
    {
        return view.preparedImagePath.empty() ? view.imagePath : view.preparedImagePath;
    }

    std::string mvsSourcePairKey(const QString& keyA, const QString& keyB)
    {
        if (keyA.isEmpty() || keyB.isEmpty() || keyA == keyB)
        {
            return std::string();
        }

        const QString pairKey = keyA < keyB ? keyA + QLatin1Char('\n') + keyB : keyB + QLatin1Char('\n') + keyA;
        return pairKey.toStdString();
    }

    std::string mvsSourcePairKey(const std::string& imageA, const std::string& imageB)
    {
        return mvsSourcePairKey(normalizedMvsPathKey(imageA), normalizedMvsPathKey(imageB));
    }

    MvsSourcePairQualityLookup buildMvsSourcePairQualityLookup(const std::vector<MvsSourcePairQuality>& qualities)
    {
        MvsSourcePairQualityLookup lookup;
        lookup.qualitiesByPairKey.reserve(qualities.size());
        for (const MvsSourcePairQuality& quality : qualities)
        {
            const std::string key = mvsSourcePairKey(quality.imageA, quality.imageB);
            if (key.empty())
            {
                continue;
            }

            auto it = lookup.qualitiesByPairKey.find(key);
            if (it == lookup.qualitiesByPairKey.end() || quality.geometricInliers > it->second.geometricInliers)
            {
                lookup.qualitiesByPairKey[key] = quality;
            }
        }
        return lookup;
    }

    DepthPostProcessEvidence depthPostProcessEvidence(const DepthFrameResult& frame)
    {
        DepthPostProcessEvidence evidence;
        if (frame.photometricConfidence)
        {
            evidence.photometricConfidence = *frame.photometricConfidence;
        }
        if (frame.geometricConfidence)
        {
            evidence.geometricConfidence = *frame.geometricConfidence;
        }
        if (frame.geometrySupportCount)
        {
            evidence.geometrySupportCount = *frame.geometrySupportCount;
        }
        if (frame.inverseDepthRelativeSpread)
        {
            evidence.inverseDepthRelativeSpread = *frame.inverseDepthRelativeSpread;
        }
        if (frame.adaptiveGeometrySupportWeight)
        {
            evidence.adaptiveSupportWeight = *frame.adaptiveGeometrySupportWeight;
        }
        if (frame.adaptiveGeometryEffectiveViewCount)
        {
            evidence.adaptiveEffectiveViewCount = *frame.adaptiveGeometryEffectiveViewCount;
        }
        if (frame.adaptiveGeometryConflictRatio)
        {
            evidence.adaptiveConflictRatio = *frame.adaptiveGeometryConflictRatio;
        }
        return evidence;
    }

    bool usesAdaptiveGeometryEvidence(const DepthGenConfig& config, MvsSceneProfile sceneProfile)
    {
        return config.enableAdaptiveGeometryEvidence && sceneProfile == MvsSceneProfile::OrbitalObject;
    }

    void releaseStoredDepthFramePixelStorage(std::vector<DepthFrameResult>& frames)
    {
        for (DepthFrameResult& frame : frames)
        {
            frame.releasePixelStorage();
        }
    }

    void releaseStoredDepthFrameStreamingPixelStorage(std::vector<DepthFrameResult>& frames)
    {
        for (DepthFrameResult& frame : frames)
        {
            frame.releaseStreamingPixelStorage();
        }
    }

    cv::Size patchMatchWorkSize(const cv::Mat& image, const PatchMatchConfig& config)
    {
        const int ds = std::max(1, config.downsampleFactor);
        return cv::Size(std::max(1, image.cols / ds), std::max(1, image.rows / ds));
    }

    std::vector<int>
    consistencySourceIndicesForFrame(const std::vector<DepthFrameResult>& frames, int refIdx, int viewCount)
    {
        std::vector<int> sources;
        if (refIdx >= 0 && refIdx < static_cast<int>(frames.size()))
        {
            for (int sourceIdx : frames[refIdx].sourceViewIndices)
            {
                if (sourceIdx < 0 || sourceIdx >= viewCount || sourceIdx == refIdx)
                {
                    continue;
                }
                if (std::find(sources.begin(), sources.end(), sourceIdx) == sources.end())
                {
                    sources.push_back(sourceIdx);
                }
            }
        }

        if (!sources.empty())
        {
            return sources;
        }

        sources.reserve(static_cast<size_t>(std::max(0, viewCount - 1)));
        for (int idx = 0; idx < viewCount; ++idx)
        {
            if (idx != refIdx)
            {
                sources.push_back(idx);
            }
        }
        return sources;
    }

    PatchMatchConfig patchMatchConfigForRecordedWorker(PatchMatchConfig config, std::string_view workerId)
    {
        std::optional<DepthComputeWorker> worker = depthComputeWorkerFromId(workerId);
        if (!worker && asciiLowerCopy(workerId) == "gpu")
        {
            worker = DepthComputeWorker{DepthComputeBackend::Cuda, -1};
        }
        if (!worker)
        {
            return config;
        }

        config.backend = worker->backend == DepthComputeBackend::Cuda     ? PatchMatchBackend::Cuda
                         : worker->backend == DepthComputeBackend::OpenCl ? PatchMatchBackend::OpenCl
                                                                          : PatchMatchBackend::Cpu;
        config.cudaDeviceIndex = worker->backend == DepthComputeBackend::Cuda ? worker->deviceIndex : -1;
        config.openClDeviceIndex = worker->backend == DepthComputeBackend::OpenCl ? worker->deviceIndex : -1;
        config.cudaFallbackToCpu = false;
        config.openClFallbackToCpu = false;
        return config;
    }

    void updateDepthCompletenessAfterPostprocess(DepthFrameResult& result,
                                                 const cv::Mat& depth,
                                                 const DepthPostProcessStats& stats)
    {
        result.depthCompleteness.preFusionPostprocessValidCount = stats.validBeforePostprocess;
        result.depthCompleteness.postConfidenceFilterValidCount = stats.validAfterConfidenceFilter;
        result.depthCompleteness.postFusionPostprocessValidCount = stats.validAfterPostprocess;
        if (result.depthCompleteness.fusionPostprocessRetentionRatio < 0.0f && stats.validBeforePostprocess > 0)
        {
            result.depthCompleteness.fusionPostprocessRetentionRatio =
                static_cast<float>(stats.validAfterPostprocess) / static_cast<float>(stats.validBeforePostprocess);
        }
        if (!result.supportRegionMask || result.supportRegionMask->empty())
        {
            return;
        }
        cv::Mat effective_mask = *result.supportRegionMask;
        if (effective_mask.size() != depth.size())
        {
            cv::resize(effective_mask, effective_mask, depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
        }
        result.depthCompleteness.finalMetrics = analyzeDepthCompleteness(
            depth, effective_mask, kSmallHoleAreaFraction, effectiveMinimumSmallHoleArea(result, depth.size()));
    }
} // namespace xjw::mvs::pipeline_detail
