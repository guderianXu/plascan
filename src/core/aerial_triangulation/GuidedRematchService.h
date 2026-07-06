#pragma once

#include "common/SfmTypes.h"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace xjw
{
namespace gui
{

enum class GuidedRematchSource
{
    GuidedRematch
};

struct GuidedRematchOptions
{
    double minOverlapScore = 0.2;
    int targetInlierCount = 80;
    double epipolarBandPx = 2.0;
    int maxMatches = 0;
};

struct GuidedRematchPair
{
    bool hasRegisteredCameraA = false;
    bool hasRegisteredCameraB = false;
    double overlapScore = 0.0;
    int geometricInlierCount = 0;
    bool permanentlyRejected = false;
};

struct GuidedRematchMatch
{
    int queryIndex = -1;
    int trainIndex = -1;
    float score = 0.0f;
    float descriptorDistance = 0.0f;
    double epipolarDistancePx = 0.0;
    GuidedRematchSource source = GuidedRematchSource::GuidedRematch;
    bool replacesExistingMatch = false;
};

struct GuidedRematchInput
{
    GuidedRematchPair pair;
    GuidedRematchOptions options;
    cv::Mat fundamentalMatrix;
    std::vector<cv::Point2f> keypointsA;
    std::vector<cv::Point2f> keypointsB;
    cv::Mat descriptorsA;
    cv::Mat descriptorsB;
    std::vector<std::pair<int, int>> existingMatches;
};

struct GuidedRematchResult
{
    bool executed = false;
    std::string rejectReason;
    int consideredCandidates = 0;
    int skippedExistingQueries = 0;
    int skippedExistingTrains = 0;
    std::vector<GuidedRematchMatch> matches;
};

struct GuidedRematchMergeResult
{
    std::vector<xjw::FeatureMatch> matches;
    int addedMatchCount = 0;
    int skippedExistingMatchCount = 0;
    int skippedInvalidMatchCount = 0;
};

bool isEligibleForGuidedRematch(const GuidedRematchPair &pair,
                                const GuidedRematchOptions &options);

GuidedRematchResult generateGuidedRematchCandidates(const GuidedRematchInput &input);

GuidedRematchMergeResult mergeGuidedRematchMatches(const std::vector<xjw::FeatureMatch> &existingMatches,
                                                   const GuidedRematchResult &guidedResult);

} // namespace gui
} // namespace xjw
