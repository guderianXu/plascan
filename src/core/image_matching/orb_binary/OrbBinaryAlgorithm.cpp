#include "OrbBinaryAlgorithm.h"

#include "../ImageMatchingRegistry.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xjw::image_matching
{
namespace
{

cv::Mat normalizedGrayImage(const ImageFeatureInput &input)
{
    cv::Mat gray;
    if (!input.grayImage.empty())
    {
        gray = input.grayImage;
    }
    else if (!input.colorImage.empty())
    {
        const int conversion = input.colorImage.channels() == 4
            ? cv::COLOR_BGRA2GRAY
            : cv::COLOR_BGR2GRAY;
        cv::cvtColor(input.colorImage, gray, conversion);
    }

    if (gray.empty())
    {
        throw std::invalid_argument("ORB binary matching requires a non-empty image");
    }
    if (gray.type() == CV_8UC1)
    {
        return gray;
    }

    cv::Mat converted;
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(gray, &minimum, &maximum);
    const double scale = maximum <= 1.0 ? 255.0 : 1.0;
    gray.convertTo(converted, CV_8U, scale);
    return converted;
}

std::vector<int> strongestKeypointIndices(const std::vector<cv::KeyPoint> &keypoints,
                                          int maximumCount)
{
    std::vector<int> indices(keypoints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
                     [&keypoints](int lhs, int rhs)
                     {
                         const cv::KeyPoint &left = keypoints[static_cast<std::size_t>(lhs)];
                         const cv::KeyPoint &right = keypoints[static_cast<std::size_t>(rhs)];
                         if (left.response != right.response)
                         {
                             return left.response > right.response;
                         }
                         if (left.pt.y != right.pt.y)
                         {
                             return left.pt.y < right.pt.y;
                         }
                         return left.pt.x < right.pt.x;
                     });
    if (maximumCount > 0 && static_cast<int>(indices.size()) > maximumCount)
    {
        indices.resize(static_cast<std::size_t>(maximumCount));
    }
    return indices;
}

FeatureSet selectFeatures(const std::vector<cv::KeyPoint> &keypoints,
                          const cv::Mat &descriptors,
                          int maximumCount,
                          const ImageFeatureInput &input,
                          int imageWidth,
                          int imageHeight)
{
    FeatureSet result;
    if (keypoints.empty() || descriptors.empty())
    {
        return result;
    }

    const std::vector<int> indices = strongestKeypointIndices(keypoints, maximumCount);
    result.keypoints.reserve(indices.size());
    result.scores.reserve(indices.size());
    result.descriptors.create(static_cast<int>(indices.size()), descriptors.cols, CV_8U);
    const double coordinateScale = input.coordinateScale > 0.0 ? input.coordinateScale : 1.0;
    for (int outputIndex = 0; outputIndex < static_cast<int>(indices.size()); ++outputIndex)
    {
        const int sourceIndex = indices[static_cast<std::size_t>(outputIndex)];
        cv::KeyPoint keypoint = keypoints[static_cast<std::size_t>(sourceIndex)];
        keypoint.pt.x = static_cast<float>(keypoint.pt.x * coordinateScale);
        keypoint.pt.y = static_cast<float>(keypoint.pt.y * coordinateScale);
        keypoint.size = static_cast<float>(keypoint.size * coordinateScale);
        result.keypoints.push_back(keypoint);
        result.scores.push_back(keypoint.response);
        descriptors.row(sourceIndex).copyTo(result.descriptors.row(outputIndex));
    }

    result.descriptorsL2Normalized = false;
    result.sourceAlgorithm = kOrbBinaryAlgorithmId;
    result.computeBackend = "opencv_cpu";
    result.imageWidth = input.originalWidth > 0 ? input.originalWidth : imageWidth;
    result.imageHeight = input.originalHeight > 0 ? input.originalHeight : imageHeight;
    return result;
}

struct BinaryNeighbor
{
    int index = -1;
    float distance = 0.0f;
    float ratio = 1.0f;
    bool accepted = false;
};

std::vector<BinaryNeighbor> binaryNeighbors(const cv::Mat &query,
                                            const cv::Mat &train,
                                            float maximumRatio)
{
    std::vector<std::vector<cv::DMatch>> candidates;
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    matcher.knnMatch(query, train, candidates, std::min(2, train.rows));

    std::vector<BinaryNeighbor> result(static_cast<std::size_t>(query.rows));
    for (int queryIndex = 0; queryIndex < static_cast<int>(candidates.size()); ++queryIndex)
    {
        const auto &neighbors = candidates[static_cast<std::size_t>(queryIndex)];
        if (neighbors.empty())
        {
            continue;
        }

        BinaryNeighbor neighbor;
        neighbor.index = neighbors.front().trainIdx;
        neighbor.distance = neighbors.front().distance;
        if (neighbors.size() < 2 || neighbors[1].distance <= 0.0f)
        {
            neighbor.ratio = 0.0f;
            neighbor.accepted = true;
        }
        else
        {
            neighbor.ratio = neighbors.front().distance / neighbors[1].distance;
            neighbor.accepted = neighbor.ratio <= maximumRatio;
        }
        result[static_cast<std::size_t>(queryIndex)] = neighbor;
    }
    return result;
}

void validateBinaryFeatures(const FeatureSet &features)
{
    if (!features.isConsistent() ||
        features.sourceAlgorithm != kOrbBinaryAlgorithmId ||
        features.descriptors.type() != CV_8U)
    {
        throw std::invalid_argument(
            "ORB binary matcher requires consistent CV_8U binary descriptors");
    }
}

} // namespace

OrbBinaryAlgorithm::OrbBinaryAlgorithm(ImageMatchingRuntimeConfig config)
    : _config(std::move(config))
{
}

ImageMatchingAlgorithmDescriptor OrbBinaryAlgorithm::descriptor() const
{
    ImageMatchingAlgorithmDescriptor value;
    value.id = QString::fromLatin1(kOrbBinaryAlgorithmId);
    value.displayName = QStringLiteral("ORB（二进制兼容基线）");
    value.version = kOrbBinaryAlgorithmVersion;
    value.inputModel = AlgorithmInputModel::ReusableFeatures;
    value.requiresCuda = false;
    value.suppliesStableFeatureIds = true;
    return value;
}

FeatureSet OrbBinaryAlgorithm::extract(const ImageFeatureInput &input) const
{
    const cv::Mat gray = normalizedGrayImage(input);
    const int maximumKeypoints = std::max(1, _config.maxKeypoints);
    const int fastThreshold = std::clamp(
        static_cast<int>(std::lround(_config.siftDetectionThreshold * 2000.0f)),
        5,
        50);
    const cv::Ptr<cv::ORB> detector = cv::ORB::create(
        maximumKeypoints,
        1.2f,
        8,
        31,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        31,
        fastThreshold);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(gray, input.validMask, keypoints, descriptors, false);
    return selectFeatures(
        keypoints, descriptors, maximumKeypoints, input, gray.cols, gray.rows);
}

MatchResult OrbBinaryAlgorithm::matchFeatures(const FeatureSet &features0,
                                              const FeatureSet &features1)
{
    validateBinaryFeatures(features0);
    validateBinaryFeatures(features1);
    if (features0.descriptors.cols != features1.descriptors.cols)
    {
        throw std::invalid_argument("ORB binary descriptor sizes do not match");
    }

    const float maximumRatio = std::clamp(_config.siftMaximumRatio, 0.0f, 1.0f);
    const std::vector<BinaryNeighbor> forward = binaryNeighbors(
        features0.descriptors, features1.descriptors, maximumRatio);
    const std::vector<BinaryNeighbor> reverse = binaryNeighbors(
        features1.descriptors, features0.descriptors, maximumRatio);

    MatchResult result;
    result.sourceAlgorithm = kOrbBinaryAlgorithmId;
    result.matches0.assign(features0.keypoints.size(), -1);
    result.matches1.assign(features1.keypoints.size(), -1);
    result.matchingScores0.assign(features0.keypoints.size(), 0.0f);
    result.matchingScores1.assign(features1.keypoints.size(), 0.0f);
    const float descriptorBits = static_cast<float>(std::max(1, features0.descriptors.cols * 8));
    for (int index0 = 0; index0 < static_cast<int>(forward.size()); ++index0)
    {
        const BinaryNeighbor &candidate = forward[static_cast<std::size_t>(index0)];
        if (!candidate.accepted || candidate.index < 0 ||
            candidate.index >= static_cast<int>(reverse.size()))
        {
            continue;
        }
        const BinaryNeighbor &backward = reverse[static_cast<std::size_t>(candidate.index)];
        if (!backward.accepted || backward.index != index0)
        {
            continue;
        }

        const float confidence = std::clamp(
            1.0f - candidate.distance / descriptorBits, 0.0f, 1.0f);
        result.matches0[static_cast<std::size_t>(index0)] = candidate.index;
        result.matches1[static_cast<std::size_t>(candidate.index)] = index0;
        result.matchingScores0[static_cast<std::size_t>(index0)] = confidence;
        result.matchingScores1[static_cast<std::size_t>(candidate.index)] = confidence;
    }
    result.buildCvMatchesFromIndices();
    return result;
}

void registerOrbBinaryAlgorithm()
{
    const ImageMatchingAlgorithmDescriptor descriptor =
        OrbBinaryAlgorithm(ImageMatchingRuntimeConfig{}).descriptor();
    QString ignoredError;
    ImageMatchingRegistry::registerAlgorithm(
        descriptor,
        [](const ImageMatchingRuntimeConfig &config)
        {
            return std::make_unique<OrbBinaryAlgorithm>(config);
        },
        &ignoredError);
}

} // namespace xjw::image_matching
