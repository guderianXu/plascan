#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "DepthPostprocessor.h"
#include "Logger.h"
#include "DepthMissingReason.h"
#include "DepthPyramidPolicy.h"
#include "MvsQualityReport.h"

namespace xjw::mvs
{

    DepthPostProcessStats DepthPostprocessor::postprocessFusionDepthMap(cv::Mat& depthMap,
                                                                        cv::Mat& confidenceMap,
                                                                        const FusionConfig& config,
                                                                        int refIdx,
                                                                        int viewCount,
                                                                        cv::Mat* missingReasonMap,
                                                                        const DepthPostProcessEvidence* evidence,
                                                                        const cv::Size& rasterPixelDomainSize)
    {
        DepthPostProcessStats stats;
        if (depthMap.empty() || depthMap.type() != CV_32F)
        {
            return stats;
        }

        stats.validBeforePostprocess = cv::countNonZero(depthMap > 0.0f);
        stats.validAfterConfidenceFilter = stats.validBeforePostprocess;
        stats.validAfterPostprocess = stats.validBeforePostprocess;
        if (stats.validBeforePostprocess <= 0)
        {
            return stats;
        }

        const cv::Size raster_size = rasterPixelDomainSize.width > 0 && rasterPixelDomainSize.height > 0
                                         ? rasterPixelDomainSize
                                         : depthMap.size();
        const DepthPixelDomainScale pixel_scale = depthPixelDomainScale(raster_size, depthMap.size());
        const int quantized_boundary_edge_radius = scaleDepthPixelRadius(1, pixel_scale);
        const int boundary_edge_radius = config.enableBoundaryAwareRetention && pixel_scale.usesReducedGrid()
                                             ? std::max(1, quantized_boundary_edge_radius)
                                             : quantized_boundary_edge_radius;
        const int boundary_protection_radius =
            pixel_scale.usesReducedGrid()
                ? std::max(config.enableBoundaryAwareRetention && config.boundaryProtectionRadiusPixels > 0 ? 1 : 0,
                           static_cast<int>(std::floor(static_cast<float>(config.boundaryProtectionRadiusPixels) *
                                                           pixel_scale.linearScale +
                                                       1.0e-6f)))
                : scaleDepthPixelRadius(config.boundaryProtectionRadiusPixels, pixel_scale);
        const int local_outlier_kernel = scaleDepthLocalOutlierKernel(config.localDepthOutlierKernelSize, pixel_scale);
        const int same_layer_radius = scaleDepthPixelRadius(1, pixel_scale);
        const int minimum_speckle_area = scaleDepthPixelArea(config.minSpeckleComponentArea, pixel_scale);
        if (pixel_scale.usesReducedGrid())
        {
            LOG_DEBUG("[MVS][帧 %d][像素域] raster=%dx%d grid=%dx%d scale=%.6f "
                      "boundary_edge=1->%d boundary_protection=%d->%d "
                      "local_kernel=%d->%d same_layer_radius=1->%d speckle_area=%d->%d",
                      refIdx,
                      raster_size.width,
                      raster_size.height,
                      depthMap.cols,
                      depthMap.rows,
                      pixel_scale.linearScale,
                      boundary_edge_radius,
                      config.boundaryProtectionRadiusPixels,
                      boundary_protection_radius,
                      config.localDepthOutlierKernelSize,
                      local_outlier_kernel,
                      same_layer_radius,
                      config.minSpeckleComponentArea,
                      minimum_speckle_area);
        }

        float confThresh = config.confidenceThresh;
        if (viewCount <= 2)
        {
            confThresh = 0.0f;
        }

        const bool hasConfidence =
            !confidenceMap.empty() && confidenceMap.size() == depthMap.size() && confidenceMap.type() == CV_32F;
        if (hasConfidence)
        {
            double cMin = 0.0;
            double cMax = 0.0;
            cv::minMaxLoc(confidenceMap, &cMin, &cMax);
            const cv::Mat validMask = depthMap > 0.0f;
            const cv::Scalar cMean = cv::mean(confidenceMap, validMask);
            LOG_DEBUG("[MVS][帧 %d][后处理] confidence min=%.4f max=%.4f mean=%.4f threshold=%.4f",
                      refIdx,
                      cMin,
                      cMax,
                      cMean[0],
                      confThresh);

            if (viewCount > 2 && config.enableAdaptiveConfidenceFilter)
            {
                const DepthMapQualityMetrics quality = analyzeDepthMapQuality(depthMap, confidenceMap, viewCount);
                const bool suspiciousFullCoverage = quality.validCoverage >= config.adaptiveFullCoverageThreshold &&
                                                    quality.meanConfidence > 0.0f &&
                                                    quality.meanConfidence < config.adaptiveLowMeanConfidenceThreshold;
                if (suspiciousFullCoverage)
                {
                    const float strictThreshold =
                        std::max(config.adaptiveStrictConfidenceThreshold, quality.recommendedFusionConfidence);
                    if (strictThreshold > confThresh)
                    {
                        LOG_DEBUG("[MVS][帧 %d][后处理] 低置信满幅深度: coverage=%.3f mean_confidence=%.3f "
                                  "threshold=%.3f->%.3f",
                                  refIdx,
                                  quality.validCoverage,
                                  quality.meanConfidence,
                                  confThresh,
                                  strictThreshold);
                        confThresh = strictThreshold;
                    }
                }
            }

            if (confThresh > 0.0f)
            {
                cv::Mat beforeConfidence = depthMap.clone();
                cv::Mat protectedBoundary;
                if (config.enableBoundaryAwareRetention && boundary_edge_radius > 0)
                {
                    const cv::Mat valid_before = beforeConfidence > 0.0f;
                    cv::Mat eroded;
                    cv::erode(
                        valid_before,
                        eroded,
                        cv::getStructuringElement(
                            cv::MORPH_ELLIPSE, cv::Size(2 * boundary_edge_radius + 1, 2 * boundary_edge_radius + 1)));
                    cv::subtract(valid_before, eroded, protectedBoundary);
                    const int radius = std::clamp(boundary_protection_radius, 0, 8);
                    if (radius > 0)
                    {
                        cv::dilate(
                            protectedBoundary,
                            protectedBoundary,
                            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1)));
                        cv::bitwise_and(protectedBoundary, valid_before, protectedBoundary);
                    }
                }
                const bool hasGeometryEvidence = evidence && evidence->geometrySupportCount.type() == CV_16UC1 &&
                                                 evidence->geometrySupportCount.size() == depthMap.size() &&
                                                 evidence->inverseDepthRelativeSpread.type() == CV_32FC1 &&
                                                 evidence->inverseDepthRelativeSpread.size() == depthMap.size();
                const bool hasIndependentGeometryConfidence = evidence &&
                                                              evidence->geometricConfidence.type() == CV_32FC1 &&
                                                              evidence->geometricConfidence.size() == depthMap.size();
                const bool hasPhotometricConfidence = evidence && evidence->photometricConfidence.type() == CV_32FC1 &&
                                                      evidence->photometricConfidence.size() == depthMap.size();
                const bool hasAdaptiveGeometryEvidence =
                    hasGeometryEvidence && evidence->adaptiveSupportWeight.type() == CV_32FC1 &&
                    evidence->adaptiveSupportWeight.size() == depthMap.size() &&
                    evidence->adaptiveEffectiveViewCount.type() == CV_32FC1 &&
                    evidence->adaptiveEffectiveViewCount.size() == depthMap.size() &&
                    evidence->adaptiveConflictRatio.type() == CV_32FC1 &&
                    evidence->adaptiveConflictRatio.size() == depthMap.size();
                int lowConfidenceCandidateCount = 0;
                int geometrySupportedRetainedCount = 0;
                int independentGeometryConfidenceRetainedCount = 0;
                int boundaryGeometryRetainedCount = 0;
#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : lowConfidenceCandidateCount,                                   \
                                                        geometrySupportedRetainedCount,                                \
                                                        independentGeometryConfidenceRetainedCount,                    \
                                                        boundaryGeometryRetainedCount)
#endif
                for (int v = 0; v < depthMap.rows; ++v)
                {
                    float* depthRow = depthMap.ptr<float>(v);
                    const float* confRow = confidenceMap.ptr<float>(v);
                    const float* photometricRow =
                        hasPhotometricConfidence ? evidence->photometricConfidence.ptr<float>(v) : confRow;
                    const float* geometricConfidenceRow =
                        hasIndependentGeometryConfidence ? evidence->geometricConfidence.ptr<float>(v) : nullptr;
                    const std::uint16_t* geometrySupportRow =
                        hasGeometryEvidence ? evidence->geometrySupportCount.ptr<std::uint16_t>(v) : nullptr;
                    const float* inverseDepthSpreadRow =
                        hasGeometryEvidence ? evidence->inverseDepthRelativeSpread.ptr<float>(v) : nullptr;
                    const float* adaptiveSupportRow =
                        hasAdaptiveGeometryEvidence ? evidence->adaptiveSupportWeight.ptr<float>(v) : nullptr;
                    const float* adaptiveEffectiveViewRow =
                        hasAdaptiveGeometryEvidence ? evidence->adaptiveEffectiveViewCount.ptr<float>(v) : nullptr;
                    const float* adaptiveConflictRow =
                        hasAdaptiveGeometryEvidence ? evidence->adaptiveConflictRatio.ptr<float>(v) : nullptr;
                    const std::uint8_t* boundaryRow =
                        !protectedBoundary.empty() ? protectedBoundary.ptr<std::uint8_t>(v) : nullptr;
                    for (int u = 0; u < depthMap.cols; ++u)
                    {
                        if (depthRow[u] > 0.0f && confRow[u] < confThresh)
                        {
                            ++lowConfidenceCandidateCount;
                            bool retainForGeometry =
                                config.enableGeometrySupportedLowConfidenceRetention &&
                                confRow[u] >= config.geometrySupportedMinimumConfidence && hasGeometryEvidence &&
                                geometrySupportRow[u] >= config.geometrySupportedMinimumObservationCount &&
                                std::isfinite(inverseDepthSpreadRow[u]) && inverseDepthSpreadRow[u] >= 0.0f &&
                                inverseDepthSpreadRow[u] <= config.geometrySupportedMaximumInverseDepthSpread;
                            if (retainForGeometry && hasAdaptiveGeometryEvidence)
                            {
                                retainForGeometry =
                                    std::isfinite(adaptiveSupportRow[u]) &&
                                    adaptiveSupportRow[u] >= config.geometrySupportedMinimumAdaptiveSupportWeight &&
                                    std::isfinite(adaptiveEffectiveViewRow[u]) &&
                                    adaptiveEffectiveViewRow[u] >=
                                        config.geometrySupportedMinimumAdaptiveEffectiveViews &&
                                    std::isfinite(adaptiveConflictRow[u]) &&
                                    adaptiveConflictRow[u] <= config.geometrySupportedMaximumAdaptiveConflictRatio;
                            }
                            const bool retainForIndependentGeometry =
                                !retainForGeometry && hasIndependentGeometryConfidence &&
                                geometricConfidenceRow[u] >= 0.55f && photometricRow[u] >= 0.10f &&
                                hasGeometryEvidence &&
                                geometrySupportRow[u] >= config.geometrySupportedMinimumObservationCount &&
                                std::isfinite(inverseDepthSpreadRow[u]) && inverseDepthSpreadRow[u] >= 0.0f &&
                                inverseDepthSpreadRow[u] <= config.geometrySupportedMaximumInverseDepthSpread;
                            const bool retainForBoundary =
                                !retainForGeometry && boundaryRow && boundaryRow[u] != 0 &&
                                config.enableBoundaryAwareRetention && confRow[u] >= config.boundaryMinimumConfidence &&
                                hasGeometryEvidence &&
                                geometrySupportRow[u] >= config.boundaryMinimumObservationCount &&
                                std::isfinite(inverseDepthSpreadRow[u]) && inverseDepthSpreadRow[u] >= 0.0f &&
                                inverseDepthSpreadRow[u] <= config.boundaryMaximumInverseDepthSpread &&
                                (!hasAdaptiveGeometryEvidence ||
                                 (std::isfinite(adaptiveConflictRow[u]) && adaptiveConflictRow[u] <= 0.60f));
                            if (retainForGeometry)
                            {
                                ++geometrySupportedRetainedCount;
                            }
                            else if (retainForIndependentGeometry)
                            {
                                ++independentGeometryConfidenceRetainedCount;
                            }
                            else if (retainForBoundary)
                            {
                                ++boundaryGeometryRetainedCount;
                            }
                            else
                            {
                                depthRow[u] = 0.0f;
                            }
                        }
                    }
                }

                stats.lowConfidenceCandidateCount = lowConfidenceCandidateCount;
                stats.geometrySupportedLowConfidenceRetained = geometrySupportedRetainedCount;
                stats.independentGeometryConfidenceRetained = independentGeometryConfidenceRetainedCount;
                stats.boundaryGeometryRetained = boundaryGeometryRetainedCount;

                int validAfterConfidence = cv::countNonZero(depthMap > 0.0f);
                LOG_DEBUG("[MVS][帧 %d][后处理] 置信度过滤 %d->%d threshold=%.4f "
                          "candidates=%d geometry_retained=%d independent_geometry_retained=%d "
                          "boundary_retained=%d",
                          refIdx,
                          stats.validBeforePostprocess,
                          validAfterConfidence,
                          confThresh,
                          lowConfidenceCandidateCount,
                          geometrySupportedRetainedCount,
                          independentGeometryConfidenceRetainedCount,
                          boundaryGeometryRetainedCount);

                if (validAfterConfidence < stats.validBeforePostprocess / 20)
                {
                    LOG_WARN("[MVS][帧 %d][后处理] 置信度过滤后可信像素过少 %d->%d，"
                             "保留严格结果并交由帧质量门判定",
                             refIdx,
                             stats.validBeforePostprocess,
                             validAfterConfidence);
                }

                stats.validAfterConfidenceFilter = validAfterConfidence;
                stats.confidenceRemoved = std::max(0, stats.validBeforePostprocess - validAfterConfidence);
                if (missingReasonMap)
                {
                    markDepthLossReason(
                        *missingReasonMap, beforeConfidence, depthMap, DepthMissingReason::LowConfidence);
                }
            }
        }
        stats.effectiveConfidenceThreshold = confThresh;

        if (config.enableLocalDepthOutlierFilter)
        {
            const cv::Mat before_local_filter = depthMap.clone();
            stats.localDepthOutlierRemoved = removeLocalDepthOutliers(depthMap,
                                                                      confidenceMap,
                                                                      local_outlier_kernel,
                                                                      config.localDepthOutlierRelThresh,
                                                                      config.maxLocalDepthOutlierRemovalRatio,
                                                                      refIdx,
                                                                      same_layer_radius);
            if (missingReasonMap)
            {
                markDepthLossReason(
                    *missingReasonMap, before_local_filter, depthMap, DepthMissingReason::LocalDepthOutlier);
            }
        }

        if (config.enableSpeckleFilter)
        {
            const cv::Mat before_speckle_filter = depthMap.clone();
            stats.smallComponentRemoved = removeSmallDepthComponents(
                depthMap, confidenceMap, minimum_speckle_area, config.maxSpeckleRemovalRatio, refIdx);
            stats.speckleRemoved = stats.smallComponentRemoved;
            if (missingReasonMap)
            {
                markDepthLossReason(
                    *missingReasonMap, before_speckle_filter, depthMap, DepthMissingReason::SmallComponent);
            }
        }

        stats.validAfterPostprocess = cv::countNonZero(depthMap > 0.0f);
        if (stats.confidenceRemoved > 0 || stats.localDepthOutlierRemoved > 0 || stats.smallComponentRemoved > 0)
        {
            LOG_DEBUG("[MVS][帧 %d][后处理] before=%d after_confidence=%d confidence_removed=%d "
                      "geometry_retained=%d local_removed=%d speckle_removed=%d after=%d",
                      refIdx,
                      stats.validBeforePostprocess,
                      stats.validAfterConfidenceFilter,
                      stats.confidenceRemoved,
                      stats.geometrySupportedLowConfidenceRetained,
                      stats.localDepthOutlierRemoved,
                      stats.speckleRemoved,
                      stats.validAfterPostprocess);
        }
        return stats;
    }
} // namespace xjw::mvs
