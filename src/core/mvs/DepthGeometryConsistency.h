#pragma once

#include "AdaptiveGeometryEvidencePolicy.h"
#include "DepthFrameQualityGate.h"
#include "MvsTypes.h"

#include <QJsonObject>

#include <opencv2/core.hpp>

namespace xjw
{
namespace mvs
{

struct ProjectedDepthConsistencyResult
{
    DepthConsistencyEvidence evidence = DepthConsistencyEvidence::Unverifiable;
    cv::Point sourcePixel{-1, -1};
    float relativeDepthError = 0.0f;
    float roundTripErrorPixels = 0.0f;
    float consistentReferenceDepth = 0.0f;
    float worldSurfaceResidual = 0.0f;
    float jointWorldPixelFootprint = 0.0f; ///< Reference lateral + target epipolar 1 px uncertainty.
    bool continuousGeometryValid = false;
};

struct GeometryEvidenceMaps
{
    cv::Mat supportCount;              ///< reference + consistent sources (CV_16UC1)
    cv::Mat sourceMask;                ///< bit N = source-plan ordinal N (CV_16UC1)
    cv::Mat inverseDepthMean;           ///< reference/source inverse-depth mean (CV_32FC1)
    cv::Mat inverseDepthRelativeSpread; ///< standard deviation / mean (CV_32FC1)
};

struct AdaptiveGeometryEvidenceAccumulatorMaps
{
    cv::Mat positiveSupport;        ///< Sum of continuous positive source support (CV_32FC1).
    cv::Mat squaredPositiveSupport; ///< Sum of squared positive support (CV_32FC1).
    cv::Mat conflict;               ///< Sum of visible conflict evidence (CV_32FC1).
    cv::Mat observable;             ///< Sum of observable source reliability (CV_32FC1).
};

struct AdaptiveGeometryEvidenceMaps
{
    cv::Mat supportWeight;      ///< Continuous fusion weight in [0, 1] (CV_32FC1).
    cv::Mat effectiveViewCount; ///< Kish effective view count including the reference (CV_32FC1).
    cv::Mat conflictRatio;      ///< Visible conflict / observable evidence in [0, 1] (CV_32FC1).
};

ProjectedDepthConsistencyResult evaluateProjectedDepthConsistency(
    const Camera &referenceCamera,
    const cv::Point2f &referencePixel,
    float referenceDepth,
    const Camera &sourceCamera,
    const cv::Mat &sourceDepth,
    float relativeThreshold,
    int searchRadius = 1,
    float maximumRoundTripErrorPixels = 3.0f,
    bool computeContinuousMetrics = true);

AdaptiveGeometryEvidenceClass adaptiveGeometryEvidenceClass(
    const ProjectedDepthConsistencyResult &result);

bool shouldRetainDepthFromConsistencyVotes(int sourceViewCount,
                                           int consistentVotes,
                                           int occludedVotes,
                                           int contradictedVotes,
                                           int minimumSourceConfirmations = 1);

cv::Mat makeGeometrySupportCount(const cv::Mat &retainedDepth,
                                 const cv::Mat &consistentVotes);

GeometryEvidenceMaps makeGeometryEvidenceMaps(
    const cv::Mat &retainedDepth,
    const cv::Mat &consistentVotes,
    const cv::Mat &sourceMask,
    const cv::Mat &sourceInverseDepthSum,
    const cv::Mat &sourceInverseDepthSquaredSum);

AdaptiveGeometryEvidenceAccumulatorMaps makeAdaptiveGeometryEvidenceAccumulatorMaps(
    cv::Size size);

AdaptiveGeometryEvidenceMaps makeAdaptiveGeometryEvidenceMaps(
    const cv::Mat &retainedDepth,
    const AdaptiveGeometryEvidenceAccumulatorMaps &accumulatorMaps,
    const AdaptiveGeometryEvidenceOptions &options = {});

QJsonObject geometryEvidenceDiagnosticsToJson(
    const cv::Mat &depth,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthRelativeSpread,
    const cv::Mat &crossViewRepairedMask,
    const cv::Mat &supportRegionMask);

} // namespace mvs
} // namespace xjw
