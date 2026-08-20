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
        if (backend == SiftComputeBackend::Cpu)
        {
            const std::vector<SiftNearestMatch> forward = cpuNearestMatches(descriptors0, descriptors1);
            const std::vector<SiftNearestMatch> reverse = cpuNearestMatches(descriptors1, descriptors0);
            return filterSiftMutualMatches(forward, reverse, filterOptions);
        }
        const std::vector<SiftNearestMatch> forward =
            matchSiftOnGpu(backend, descriptors0, descriptors1, _config.cudaDevice);
        const std::vector<SiftNearestMatch> reverse =
            matchSiftOnGpu(backend, descriptors1, descriptors0, _config.cudaDevice);
        return filterSiftMutualMatches(forward, reverse, filterOptions);
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
