#include "GuidedRematchService.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace
{

bool isUsableFundamentalMatrix(const cv::Mat &matrix)
{
    return matrix.rows == 3 && matrix.cols == 3 &&
           (matrix.type() == CV_32F || matrix.type() == CV_64F);
}

double matrixValue(const cv::Mat &matrix, int row, int col)
{
    if (matrix.type() == CV_32F)
    {
        return static_cast<double>(matrix.at<float>(row, col));
    }
    return matrix.at<double>(row, col);
}

cv::Mat descriptorsAsFloat(const cv::Mat &descriptors)
{
    if (descriptors.empty())
    {
        return cv::Mat();
    }

    if (descriptors.type() == CV_32F)
    {
        return descriptors;
    }

    cv::Mat converted;
    descriptors.convertTo(converted, CV_32F);
    return converted;
}

float descriptorDistance(const cv::Mat &descriptors_a,
                         int index_a,
                         const cv::Mat &descriptors_b,
                         int index_b)
{
    const float *row_a = descriptors_a.ptr<float>(index_a);
    const float *row_b = descriptors_b.ptr<float>(index_b);
    double sum = 0.0;
    for (int col = 0; col < descriptors_a.cols; ++col)
    {
        const double delta = static_cast<double>(row_a[col] - row_b[col]);
        sum += delta * delta;
    }
    return static_cast<float>(std::sqrt(sum));
}

double epipolarDistance(const cv::Mat &fundamental_matrix,
                        const cv::Point2f &point_a,
                        const cv::Point2f &point_b)
{
    const double x = static_cast<double>(point_a.x);
    const double y = static_cast<double>(point_a.y);
    const double a = matrixValue(fundamental_matrix, 0, 0) * x +
                     matrixValue(fundamental_matrix, 0, 1) * y +
                     matrixValue(fundamental_matrix, 0, 2);
    const double b = matrixValue(fundamental_matrix, 1, 0) * x +
                     matrixValue(fundamental_matrix, 1, 1) * y +
                     matrixValue(fundamental_matrix, 1, 2);
    const double c = matrixValue(fundamental_matrix, 2, 0) * x +
                     matrixValue(fundamental_matrix, 2, 1) * y +
                     matrixValue(fundamental_matrix, 2, 2);

    const double denominator = std::sqrt(a * a + b * b);
    if (denominator <= std::numeric_limits<double>::epsilon())
    {
        return std::numeric_limits<double>::infinity();
    }

    return std::abs(a * static_cast<double>(point_b.x) +
                    b * static_cast<double>(point_b.y) + c) / denominator;
}

} // namespace

namespace xjw
{
namespace gui
{

bool isEligibleForGuidedRematch(const GuidedRematchPair &pair,
                                const GuidedRematchOptions &options)
{
    if (!pair.hasRegisteredCameraA || !pair.hasRegisteredCameraB)
    {
        return false;
    }
    if (pair.permanentlyRejected)
    {
        return false;
    }
    if (pair.overlapScore <= options.minOverlapScore)
    {
        return false;
    }

    const int target_inlier_count = std::max(1, options.targetInlierCount);
    return pair.geometricInlierCount < target_inlier_count;
}

GuidedRematchResult generateGuidedRematchCandidates(const GuidedRematchInput &input)
{
    GuidedRematchResult result;
    if (!isEligibleForGuidedRematch(input.pair, input.options))
    {
        result.rejectReason = "pair_not_eligible";
        return result;
    }
    if (!isUsableFundamentalMatrix(input.fundamentalMatrix))
    {
        result.rejectReason = "invalid_fundamental_matrix";
        return result;
    }

    const cv::Mat descriptors_a = descriptorsAsFloat(input.descriptorsA);
    const cv::Mat descriptors_b = descriptorsAsFloat(input.descriptorsB);
    if (descriptors_a.empty() || descriptors_b.empty() ||
        descriptors_a.rows != static_cast<int>(input.keypointsA.size()) ||
        descriptors_b.rows != static_cast<int>(input.keypointsB.size()) ||
        descriptors_a.cols != descriptors_b.cols)
    {
        result.rejectReason = "invalid_descriptors";
        return result;
    }

    result.executed = true;

    std::set<int> used_queries;
    std::set<int> used_trains;
    for (const auto &match : input.existingMatches)
    {
        if (match.first >= 0)
        {
            used_queries.insert(match.first);
        }
        if (match.second >= 0)
        {
            used_trains.insert(match.second);
        }
    }

    const double epipolar_band_px = std::max(0.0, input.options.epipolarBandPx);
    const int max_matches = std::max(0, input.options.maxMatches);

    for (int query_index = 0; query_index < static_cast<int>(input.keypointsA.size()); ++query_index)
    {
        if (used_queries.count(query_index) > 0)
        {
            ++result.skippedExistingQueries;
            continue;
        }

        int best_train_index = -1;
        float best_descriptor_distance = std::numeric_limits<float>::infinity();
        double best_epipolar_distance = std::numeric_limits<double>::infinity();

        for (int train_index = 0; train_index < static_cast<int>(input.keypointsB.size()); ++train_index)
        {
            if (used_trains.count(train_index) > 0)
            {
                ++result.skippedExistingTrains;
                continue;
            }

            const double line_distance = epipolarDistance(input.fundamentalMatrix,
                                                         input.keypointsA[static_cast<std::size_t>(query_index)],
                                                         input.keypointsB[static_cast<std::size_t>(train_index)]);
            if (line_distance > epipolar_band_px)
            {
                continue;
            }

            ++result.consideredCandidates;
            const float distance = descriptorDistance(descriptors_a, query_index, descriptors_b, train_index);
            if (distance < best_descriptor_distance)
            {
                best_descriptor_distance = distance;
                best_epipolar_distance = line_distance;
                best_train_index = train_index;
            }
        }

        if (best_train_index < 0)
        {
            continue;
        }

        GuidedRematchMatch match;
        match.queryIndex = query_index;
        match.trainIndex = best_train_index;
        match.descriptorDistance = best_descriptor_distance;
        match.epipolarDistancePx = best_epipolar_distance;
        match.score = 1.0f / (1.0f + std::max(0.0f, best_descriptor_distance));
        match.source = GuidedRematchSource::GuidedRematch;
        match.replacesExistingMatch = false;

        result.matches.push_back(match);
        used_queries.insert(query_index);
        used_trains.insert(best_train_index);

        if (max_matches > 0 && static_cast<int>(result.matches.size()) >= max_matches)
        {
            break;
        }
    }

    return result;
}

GuidedRematchMergeResult mergeGuidedRematchMatches(const std::vector<xjw::FeatureMatch> &existingMatches,
                                                   const GuidedRematchResult &guidedResult)
{
    GuidedRematchMergeResult result;
    result.matches = existingMatches;
    if (!guidedResult.executed)
    {
        return result;
    }

    std::set<int> used_queries;
    std::set<int> used_trains;
    for (const xjw::FeatureMatch &match : existingMatches)
    {
        if (match.idx1 != xjw::kInvalidFeatureIdx)
        {
            used_queries.insert(static_cast<int>(match.idx1));
        }
        if (match.idx2 != xjw::kInvalidFeatureIdx)
        {
            used_trains.insert(static_cast<int>(match.idx2));
        }
    }

    for (const GuidedRematchMatch &match : guidedResult.matches)
    {
        if (match.queryIndex < 0 || match.trainIndex < 0)
        {
            ++result.skippedInvalidMatchCount;
            continue;
        }
        if (used_queries.count(match.queryIndex) > 0 || used_trains.count(match.trainIndex) > 0)
        {
            ++result.skippedExistingMatchCount;
            continue;
        }

        xjw::FeatureMatch sfmMatch;
        sfmMatch.idx1 = static_cast<xjw::FeatureIdx>(match.queryIndex);
        sfmMatch.idx2 = static_cast<xjw::FeatureIdx>(match.trainIndex);
        sfmMatch.score = match.score;
        result.matches.push_back(sfmMatch);
        used_queries.insert(match.queryIndex);
        used_trains.insert(match.trainIndex);
        ++result.addedMatchCount;
    }

    return result;
}

} // namespace gui
} // namespace xjw
