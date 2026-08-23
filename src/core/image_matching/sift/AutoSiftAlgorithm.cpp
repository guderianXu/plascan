#include "AutoSiftAlgorithm.h"

#include "SiftFeatureExtractor.h"
#include "SiftComputeBackend.h"
#include "SiftMatchFilter.h"
#include "../ImageMatchingRegistry.h"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xjw::image_matching
{
    namespace
    {

        void validateSiftFeatures(const FeatureSet& features)
        {
            if (!features.isConsistent() || features.sourceAlgorithm != "sift" ||
                features.descriptors.type() != CV_32F || features.descriptors.cols != 128)
            {
                throw std::invalid_argument("Auto SIFT matcher requires consistent CV_32F SIFT descriptors");
            }
        }

        cv::Mat normalizedSiftDescriptors(const cv::Mat& descriptors)
        {
            cv::Mat normalized = descriptors.clone();
            for (int rowIndex = 0; rowIndex < normalized.rows; ++rowIndex)
            {
                cv::Mat row = normalized.row(rowIndex);
                const double norm = cv::norm(row, cv::NORM_L2);
                if (norm > 1e-12)
                {
                    row /= norm;
                }
            }
            return normalized;
        }

        std::vector<SiftNearestMatch> cpuNearestMatches(const cv::Mat& queryDescriptors,
                                                        const cv::Mat& trainDescriptors)
        {
            std::vector<std::vector<cv::DMatch>> neighbors;
            cv::BFMatcher matcher(cv::NORM_L2, false);
            matcher.knnMatch(queryDescriptors, trainDescriptors, neighbors, std::min(2, trainDescriptors.rows));

            std::vector<SiftNearestMatch> matches(static_cast<std::size_t>(queryDescriptors.rows));
            for (int index = 0; index < static_cast<int>(neighbors.size()); ++index)
            {
                const auto& candidates = neighbors[static_cast<std::size_t>(index)];
                if (candidates.empty())
                {
                    continue;
                }
                const float distance = candidates.front().distance;
                const float similarity = std::clamp(1.0f - 0.5f * distance * distance, 0.0f, 1.0f);
                float ambiguity = 1.0f;
                if (candidates.size() > 1 && candidates[1].distance > 1e-6f)
                {
                    ambiguity = std::clamp(distance / candidates[1].distance, 0.0f, 1.0f);
                }
                matches[static_cast<std::size_t>(index)] = {candidates.front().trainIdx, similarity, ambiguity};
            }
            return matches;
        }

        std::vector<int> channelIndices(const FeatureSet& features, bool recovery)
        {
            std::vector<int> indices;
            indices.reserve(features.keypoints.size());
            for (int index = 0; index < features.size(); ++index)
            {
                const bool isRecovery =
                    features.keypoints[static_cast<std::size_t>(index)].class_id == 1;
                if (isRecovery == recovery)
                {
                    indices.push_back(index);
                }
            }
            return indices;
        }

        cv::Mat descriptorSubset(const cv::Mat& descriptors, const std::vector<int>& indices)
        {
            cv::Mat subset(static_cast<int>(indices.size()), descriptors.cols, CV_32F);
            for (int row = 0; row < static_cast<int>(indices.size()); ++row)
            {
                descriptors.row(indices[static_cast<std::size_t>(row)]).copyTo(subset.row(row));
            }
            return subset;
        }

        MatchResult matchDescriptorChannel(const cv::Mat& descriptors0,
                                           const cv::Mat& descriptors1,
                                           SiftComputeBackend backend,
                                           int cudaDevice,
                                           const SiftMatchFilterOptions& filterOptions)
        {
            if (descriptors0.empty() || descriptors1.empty())
            {
                return {};
            }
            if (backend == SiftComputeBackend::Cpu)
            {
                return filterSiftMutualMatches(cpuNearestMatches(descriptors0, descriptors1),
                                               cpuNearestMatches(descriptors1, descriptors0),
                                               filterOptions);
            }
            return filterSiftMutualMatches(
                matchSiftOnGpu(backend, descriptors0, descriptors1, cudaDevice),
                matchSiftOnGpu(backend, descriptors1, descriptors0, cudaDevice),
                filterOptions);
        }

        void appendChannelMatches(MatchResult* destination,
                                  const MatchResult& source,
                                  const std::vector<int>& indices0,
                                  const std::vector<int>& indices1)
        {
            if (!destination)
            {
                return;
            }
            for (const cv::DMatch& match : source.cvMatches)
            {
                if (match.queryIdx < 0 || match.trainIdx < 0 ||
                    match.queryIdx >= static_cast<int>(indices0.size()) ||
                    match.trainIdx >= static_cast<int>(indices1.size()))
                {
                    continue;
                }
                const int index0 = indices0[static_cast<std::size_t>(match.queryIdx)];
                const int index1 = indices1[static_cast<std::size_t>(match.trainIdx)];
                destination->matches0[static_cast<std::size_t>(index0)] = index1;
                destination->matches1[static_cast<std::size_t>(index1)] = index0;
                destination->matchingScores0[static_cast<std::size_t>(index0)] =
                    source.matchingScores0[static_cast<std::size_t>(match.queryIdx)];
                destination->matchingScores1[static_cast<std::size_t>(index1)] =
                    source.matchingScores1[static_cast<std::size_t>(match.trainIdx)];
            }
        }

    } // namespace

    AutoSiftAlgorithm::AutoSiftAlgorithm(ImageMatchingRuntimeConfig config) : _config(std::move(config))
    {
    }

    ImageMatchingAlgorithmDescriptor AutoSiftAlgorithm::descriptor() const
    {
        ImageMatchingAlgorithmDescriptor value;
        value.id = QString::fromLatin1(kAutoSiftAlgorithmId);
        value.displayName = QStringLiteral("Auto SIFT（CUDA / Metal / OpenCL / CPU）");
        value.version = kAutoSiftAlgorithmVersion;
        value.inputModel = AlgorithmInputModel::ReusableFeatures;
        value.requiresCuda = false;
        value.suppliesStableFeatureIds = true;
        return value;
    }

    FeatureSet AutoSiftAlgorithm::extract(const ImageFeatureInput& input) const
    {
        return SiftFeatureExtractor::extract(input, _config);
    }

    MatchResult AutoSiftAlgorithm::matchFeatures(const FeatureSet& features0, const FeatureSet& features1)
    {
        validateSiftFeatures(features0);
        validateSiftFeatures(features1);

        const cv::Mat descriptors0 = normalizedSiftDescriptors(features0.descriptors);
        const cv::Mat descriptors1 = normalizedSiftDescriptors(features1.descriptors);
        SiftMatchFilterOptions filterOptions;
        filterOptions.confidenceThreshold = _config.matchThreshold;
        filterOptions.maximumRatio = _config.siftMaximumRatio;
        filterOptions.minimumAdaptiveRatio = _config.siftMinimumAdaptiveRatio;
        filterOptions.adaptiveRatio = _config.adaptiveSiftRatio;
        const SiftComputeBackend backend = resolveSiftBackend(_config.siftBackend, _config.cudaDevice);
        const std::vector<int> recovery0 = channelIndices(features0, true);
        const std::vector<int> recovery1 = channelIndices(features1, true);
        if (recovery0.empty() && recovery1.empty())
        {
            return matchDescriptorChannel(
                descriptors0, descriptors1, backend, _config.cudaDevice, filterOptions);
        }

        const std::vector<int> base0 = channelIndices(features0, false);
        const std::vector<int> base1 = channelIndices(features1, false);
        MatchResult result;
        result.sourceAlgorithm = kAutoSiftAlgorithmId;
        result.matches0.assign(features0.keypoints.size(), -1);
        result.matches1.assign(features1.keypoints.size(), -1);
        result.matchingScores0.assign(features0.keypoints.size(), 0.0f);
        result.matchingScores1.assign(features1.keypoints.size(), 0.0f);
        appendChannelMatches(&result,
                             matchDescriptorChannel(descriptorSubset(descriptors0, base0),
                                                    descriptorSubset(descriptors1, base1),
                                                    backend,
                                                    _config.cudaDevice,
                                                    filterOptions),
                             base0,
                             base1);
        appendChannelMatches(&result,
                             matchDescriptorChannel(descriptorSubset(descriptors0, recovery0),
                                                    descriptorSubset(descriptors1, recovery1),
                                                    backend,
                                                    _config.cudaDevice,
                                                    filterOptions),
                             recovery0,
                             recovery1);
        result.buildCvMatchesFromIndices();
        return result;
    }

    void registerAutoSiftAlgorithm()
    {
        const ImageMatchingAlgorithmDescriptor descriptor =
            AutoSiftAlgorithm(ImageMatchingRuntimeConfig{}).descriptor();
        QString ignoredError;
        ImageMatchingRegistry::registerAlgorithm(
            descriptor,
            [](const ImageMatchingRuntimeConfig& config) { return std::make_unique<AutoSiftAlgorithm>(config); },
            &ignoredError);
    }

} // namespace xjw::image_matching
