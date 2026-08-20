#include "SiftGuidedMatcher.h"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace xjw::image_matching
{
    namespace
    {

        double
        sampsonDistance(const std::array<double, 9>& fundamental, const cv::Point2f& point0, const cv::Point2f& point1)
        {
            const double fx0x = fundamental[0] * point0.x + fundamental[1] * point0.y + fundamental[2];
            const double fx0y = fundamental[3] * point0.x + fundamental[4] * point0.y + fundamental[5];
            const double fx0z = fundamental[6] * point0.x + fundamental[7] * point0.y + fundamental[8];
            const double ftx1x = fundamental[0] * point1.x + fundamental[3] * point1.y + fundamental[6];
            const double ftx1y = fundamental[1] * point1.x + fundamental[4] * point1.y + fundamental[7];
            const double numerator = point1.x * fx0x + point1.y * fx0y + fx0z;
            const double denominator = fx0x * fx0x + fx0y * fx0y + ftx1x * ftx1x + ftx1y * ftx1y;
            if (!std::isfinite(denominator) || denominator <= 1e-15)
            {
                return std::numeric_limits<double>::infinity();
            }
            return std::sqrt((numerator * numerator) / denominator);
        }

        std::vector<int> unmatchedIndices(int count, const std::vector<int>& used)
        {
            std::unordered_set<int> usedSet;
            usedSet.reserve(used.size());
            for (const int index : used)
            {
                if (index >= 0 && index < count)
                {
                    usedSet.insert(index);
                }
            }
            std::vector<int> result;
            result.reserve(static_cast<std::size_t>(std::max(0, count - static_cast<int>(usedSet.size()))));
            for (int index = 0; index < count; ++index)
            {
                if (!usedSet.contains(index))
                {
                    result.push_back(index);
                }
            }
            return result;
        }

        cv::Mat descriptorSubset(const cv::Mat& descriptors, const std::vector<int>& indices)
        {
            cv::Mat result(static_cast<int>(indices.size()), descriptors.cols, CV_32F);
            for (int row = 0; row < static_cast<int>(indices.size()); ++row)
            {
                descriptors.row(indices[static_cast<std::size_t>(row)]).copyTo(result.row(row));
            }
            return result;
        }

        struct GuidedCandidate
        {
            int peerIndex = -1;
            float distance = std::numeric_limits<float>::infinity();
            float ratio = 1.0f;
        };

        std::vector<GuidedCandidate> guidedCandidates(const FeatureSet& queryFeatures,
                                                      const FeatureSet& trainFeatures,
                                                      const std::vector<int>& queryIndices,
                                                      const std::vector<int>& trainIndices,
                                                      const std::array<double, 9>& fundamental,
                                                      bool reverse,
                                                      const SiftGuidedMatchOptions& options)
        {
            const cv::Mat query = descriptorSubset(queryFeatures.descriptors, queryIndices);
            const cv::Mat train = descriptorSubset(trainFeatures.descriptors, trainIndices);
            const int neighborCount = std::min({options.descriptorNeighbors, train.rows, 16});
            std::vector<GuidedCandidate> result(queryIndices.size());
            if (query.empty() || train.empty() || neighborCount <= 0)
            {
                return result;
            }

            std::vector<std::vector<cv::DMatch>> neighbors;
            cv::BFMatcher(cv::NORM_L2, false).knnMatch(query, train, neighbors, neighborCount);
            for (int queryRow = 0; queryRow < static_cast<int>(neighbors.size()); ++queryRow)
            {
                const int queryIndex = queryIndices[static_cast<std::size_t>(queryRow)];
                float secondDistance = std::numeric_limits<float>::infinity();
                GuidedCandidate best;
                for (const cv::DMatch& match : neighbors[static_cast<std::size_t>(queryRow)])
                {
                    if (match.trainIdx < 0 || match.trainIdx >= static_cast<int>(trainIndices.size()))
                    {
                        continue;
                    }
                    const int trainIndex = trainIndices[static_cast<std::size_t>(match.trainIdx)];
                    const cv::Point2f point0 = reverse
                                                   ? trainFeatures.keypoints[static_cast<std::size_t>(trainIndex)].pt
                                                   : queryFeatures.keypoints[static_cast<std::size_t>(queryIndex)].pt;
                    const cv::Point2f point1 = reverse
                                                   ? queryFeatures.keypoints[static_cast<std::size_t>(queryIndex)].pt
                                                   : trainFeatures.keypoints[static_cast<std::size_t>(trainIndex)].pt;
                    if (sampsonDistance(fundamental, point0, point1) > options.epipolarThresholdPixels)
                    {
                        continue;
                    }
                    if (best.peerIndex < 0)
                    {
                        best.peerIndex = trainIndex;
                        best.distance = match.distance;
                    }
                    else
                    {
                        secondDistance = match.distance;
                        break;
                    }
                }
                if (best.peerIndex >= 0)
                {
                    best.ratio =
                        std::isfinite(secondDistance) && secondDistance > 1e-6f ? best.distance / secondDistance : 0.0f;
                    result[static_cast<std::size_t>(queryRow)] = best;
                }
            }
            return result;
        }

    } // namespace

    std::vector<SiftGuidedMatch> findGuidedSiftMatches(const FeatureSet& features0,
                                                       const FeatureSet& features1,
                                                       const std::array<double, 9>& fundamental,
                                                       const std::vector<int>& existingFeatureIds0,
                                                       const std::vector<int>& existingFeatureIds1,
                                                       const SiftGuidedMatchOptions& options)
    {
        if (!features0.isConsistent() || !features1.isConsistent() || features0.descriptors.type() != CV_32F ||
            features1.descriptors.type() != CV_32F)
        {
            return {};
        }
        const std::vector<int> unmatched0 = unmatchedIndices(features0.size(), existingFeatureIds0);
        const std::vector<int> unmatched1 = unmatchedIndices(features1.size(), existingFeatureIds1);
        if (unmatched0.empty() || unmatched1.empty())
        {
            return {};
        }

        const std::vector<GuidedCandidate> forward =
            guidedCandidates(features0, features1, unmatched0, unmatched1, fundamental, false, options);
        const std::vector<GuidedCandidate> reverse =
            guidedCandidates(features1, features0, unmatched1, unmatched0, fundamental, true, options);
        std::vector<int> reverseRowByFeature(static_cast<std::size_t>(features1.size()), -1);
        for (int row = 0; row < static_cast<int>(unmatched1.size()); ++row)
        {
            reverseRowByFeature[static_cast<std::size_t>(unmatched1[static_cast<std::size_t>(row)])] = row;
        }

        std::vector<SiftGuidedMatch> result;
        for (int row = 0; row < static_cast<int>(forward.size()); ++row)
        {
            const GuidedCandidate& candidate = forward[static_cast<std::size_t>(row)];
            if (candidate.peerIndex < 0 || candidate.ratio > options.maximumDescriptorRatio)
            {
                continue;
            }
            const int reverseRow = reverseRowByFeature[static_cast<std::size_t>(candidate.peerIndex)];
            if (reverseRow < 0 || reverseRow >= static_cast<int>(reverse.size()))
            {
                continue;
            }
            const GuidedCandidate& reverseCandidate = reverse[static_cast<std::size_t>(reverseRow)];
            const int index0 = unmatched0[static_cast<std::size_t>(row)];
            if (reverseCandidate.peerIndex != index0 || reverseCandidate.ratio > options.maximumDescriptorRatio)
            {
                continue;
            }
            const float distance = std::max(candidate.distance, reverseCandidate.distance);
            const double residual = sampsonDistance(
                fundamental,
                features0.keypoints[static_cast<std::size_t>(index0)].pt,
                features1.keypoints[static_cast<std::size_t>(candidate.peerIndex)].pt);
            if (!std::isfinite(residual) || residual > options.epipolarThresholdPixels)
            {
                continue;
            }
            result.push_back({index0,
                              candidate.peerIndex,
                              std::clamp(1.0f - 0.5f * distance * distance, 0.0f, 1.0f),
                              candidate.ratio,
                              reverseCandidate.ratio,
                              static_cast<float>(residual)});
        }
        std::stable_sort(result.begin(),
                         result.end(),
                         [](const SiftGuidedMatch& left, const SiftGuidedMatch& right)
                         { return left.confidence > right.confidence; });
        if (options.maximumAdditionalMatches > 0 && static_cast<int>(result.size()) > options.maximumAdditionalMatches)
        {
            result.resize(static_cast<std::size_t>(options.maximumAdditionalMatches));
        }
        return result;
    }

} // namespace xjw::image_matching
