#pragma once

#include "AdaptiveGeometryEvidencePolicy.h"
#include "DepthFrameQualityGate.h"
#include "MvsTypes.h"

#include <QJsonObject>
#include <QString>

#include <opencv2/core.hpp>

#include <array>
#include <vector>

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

struct AdaptiveGeometryEvidenceSummary
{
    bool validInputs = false;
    int observablePixelCount = 0;
    float effectiveViewCountMean = -1.0f;
    float conflictRatioMean = -1.0f;
};

struct DiscreteGeometryCoreSummary
{
    bool validInputs = false;
    int validPixelCount = 0;
    int corePixelCount = 0;
    float coreRatio = -1.0f;
};

struct GeometrySourceOrdinalContract
{
    bool valid = false;
    bool persistMask = false;
    QString errorMessage;
};

/// Validate the exact bit-to-view mapping used by geometry source masks.
/// A zero mask with no ordinal table is a valid explicit "no source evidence"
/// state and is normalized by writers by omitting both fields.
GeometrySourceOrdinalContract validateGeometrySourceOrdinalContract(
    const cv::Mat &sourceMask,
    const std::vector<int> &sourceViewIndices,
    int referenceViewIndex,
    int viewCount,
    cv::Size expectedSize = {});

ProjectedDepthConsistencyResult evaluateProjectedDepthConsistency(
    const FramePinholeCamera &referenceCamera,
    const cv::Point2f &referencePixel,
    float referenceDepth,
    const FramePinholeCamera &sourceCamera,
    const cv::Mat &sourceDepth,
    float relativeThreshold,
    int searchRadius = 1,
    float maximumRoundTripErrorPixels = 3.0f,
    bool computeContinuousMetrics = true,
    bool evaluateSubpixelFootprint = false);

/**
 * @brief Evaluate one source view using an already unprojected reference point.
 *
 * The reference world point is independent of the source view.  Callers that
 * compare one reference pixel against several sources can therefore unproject
 * it once and reuse the exact double-precision result without changing source
 * order or vote accumulation semantics.
 */
ProjectedDepthConsistencyResult evaluateProjectedDepthConsistencyFromReferenceWorld(
    const FramePinholeCamera &referenceCamera,
    const cv::Point2f &referencePixel,
    float referenceDepth,
    const std::array<double, 3> &referenceWorld,
    const FramePinholeCamera &sourceCamera,
    const cv::Mat &sourceDepth,
    float relativeThreshold,
    int searchRadius = 1,
    float maximumRoundTripErrorPixels = 3.0f,
    bool computeContinuousMetrics = true,
    bool evaluateSubpixelFootprint = false);

AdaptiveGeometryEvidenceClass adaptiveGeometryEvidenceClass(
    const ProjectedDepthConsistencyResult &result);

/// Summarizes pixels for which at least one source produced observable
/// evidence. Reference-only pixels have effectiveViewCount == 1 and zero
/// conflict, and must not dilute either frame-level mean.
AdaptiveGeometryEvidenceSummary summarizeAdaptiveGeometryEvidence(
    const AdaptiveGeometryEvidenceMaps &maps);

/// Summarize the final retained pixels that jointly have broad discrete
/// multi-view support and a tightly clustered inverse-depth estimate.  This
/// independent hard-evidence core is the guarded fallback when the continuous
/// adaptive residual is miscalibrated for very narrow-FOV orbital imagery.
DiscreteGeometryCoreSummary summarizeDiscreteGeometryCore(
    const cv::Mat &retainedDepth,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthRelativeSpread,
    const cv::Mat &supportRegionMask = cv::Mat(),
    int minimumObservationCount =
        kDiscreteGeometryCoreMinimumObservationCount,
    float maximumInverseDepthSpread =
        kDiscreteGeometryCoreMaximumInverseDepthSpread);

/// Build manifest diagnostics while keeping the pre-consistency candidate
/// domain separate from the source-observable evidence domain used by the
/// adaptive quality means.
QJsonObject adaptiveGeometryEvidenceDiagnosticsToJson(
    const cv::Mat &retainedDepth,
    const AdaptiveGeometryEvidenceMaps &maps);

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
