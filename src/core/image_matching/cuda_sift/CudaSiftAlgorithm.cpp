#include "CudaSiftAlgorithm.h"

#include "CudaSiftMatchFilter.h"
#include "../ImageMatchingRegistry.h"
#include "../sift/SiftFeatureExtractor.h"

#include <opencv2/features2d.hpp>

#if defined(PLASCAN_HAS_CUDA_SIFT)
#include "cudaSift.h"

#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xjw::image_matching
{
namespace
{

#if defined(PLASCAN_HAS_CUDA_SIFT)
void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CUDA SIFT ") + operation + ": " + cudaGetErrorString(status));
    }
}

class SiftDataBuffer
{
public:
    explicit SiftDataBuffer(const FeatureSet &features)
    {
        const int count = features.size();
        InitSiftData(_data, std::max(1, count), true, true);
        _data.numPts = count;

#ifdef MANAGEDMEM
        SiftPoint *points = _data.m_data;
#else
        SiftPoint *points = _data.h_data;
#endif
        std::memset(points, 0, sizeof(SiftPoint) * static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            const cv::KeyPoint &keypoint =
                features.keypoints[static_cast<std::size_t>(index)];
            SiftPoint &point = points[index];
            point.xpos = keypoint.pt.x;
            point.ypos = keypoint.pt.y;
            point.scale = keypoint.size;
            point.orientation = keypoint.angle;
            std::copy(features.descriptors.ptr<float>(index),
                      features.descriptors.ptr<float>(index) + 128,
                      point.data);
        }

#ifndef MANAGEDMEM
        checkCuda(cudaMemcpy(_data.d_data,
                             _data.h_data,
                             sizeof(SiftPoint) * static_cast<std::size_t>(count),
                             cudaMemcpyHostToDevice),
                  "descriptor upload failed");
#endif
    }

    ~SiftDataBuffer() noexcept
    {
        FreeSiftData(_data);
    }

    SiftDataBuffer(const SiftDataBuffer &) = delete;
    SiftDataBuffer &operator=(const SiftDataBuffer &) = delete;

    SiftData &data()
    {
        return _data;
    }

    std::vector<CudaSiftNearestMatch> nearestMatches() const
    {
#ifdef MANAGEDMEM
        const SiftPoint *points = _data.m_data;
#else
        const SiftPoint *points = _data.h_data;
#endif
        std::vector<CudaSiftNearestMatch> matches(
            static_cast<std::size_t>(_data.numPts));
        for (int index = 0; index < _data.numPts; ++index)
        {
            matches[static_cast<std::size_t>(index)] = {
                points[index].match,
                points[index].score,
                points[index].ambiguity};
        }
        return matches;
    }

private:
    SiftData _data{};
};
#endif

void validateSiftFeatures(const FeatureSet &features)
{
    if (!features.isConsistent() || features.sourceAlgorithm != "sift" ||
        features.descriptors.type() != CV_32F || features.descriptors.cols != 128)
    {
        throw std::invalid_argument(
            "CUDA SIFT matcher requires consistent CV_32F SIFT descriptors");
    }
}

cv::Mat normalizedSiftDescriptors(const cv::Mat &descriptors)
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

std::vector<CudaSiftNearestMatch> cpuNearestMatches(
    const cv::Mat &queryDescriptors,
    const cv::Mat &trainDescriptors)
{
    std::vector<std::vector<cv::DMatch>> neighbors;
    cv::BFMatcher matcher(cv::NORM_L2, false);
    matcher.knnMatch(queryDescriptors,
                     trainDescriptors,
                     neighbors,
                     std::min(2, trainDescriptors.rows));

    std::vector<CudaSiftNearestMatch> matches(
        static_cast<std::size_t>(queryDescriptors.rows));
    for (int index = 0; index < static_cast<int>(neighbors.size()); ++index)
    {
        const auto &candidates = neighbors[static_cast<std::size_t>(index)];
        if (candidates.empty())
        {
            continue;
        }
        const float distance = candidates.front().distance;
        const float similarity = std::clamp(
            1.0f - 0.5f * distance * distance, 0.0f, 1.0f);
        float ambiguity = 1.0f;
        if (candidates.size() > 1 && candidates[1].distance > 1e-6f)
        {
            ambiguity = std::clamp(
                distance / candidates[1].distance, 0.0f, 1.0f);
        }
        matches[static_cast<std::size_t>(index)] = {
            candidates.front().trainIdx,
            similarity,
            ambiguity};
    }
    return matches;
}

MatchResult matchSiftOnCpu(const FeatureSet &features0,
                           const FeatureSet &features1,
                           float threshold)
{
    const cv::Mat descriptors0 = normalizedSiftDescriptors(features0.descriptors);
    const cv::Mat descriptors1 = normalizedSiftDescriptors(features1.descriptors);
    const std::vector<CudaSiftNearestMatch> forward =
        cpuNearestMatches(descriptors0, descriptors1);
    const std::vector<CudaSiftNearestMatch> reverse =
        cpuNearestMatches(descriptors1, descriptors0);
    return filterCudaSiftMutualMatches(forward, reverse, threshold);
}

} // namespace

CudaSiftAlgorithm::CudaSiftAlgorithm(ImageMatchingRuntimeConfig config)
    : _config(std::move(config))
{
}

ImageMatchingAlgorithmDescriptor CudaSiftAlgorithm::descriptor() const
{
    ImageMatchingAlgorithmDescriptor value;
    value.id = QString::fromLatin1(kCudaSiftAlgorithmId);
    value.displayName = QStringLiteral("CUDA SIFT 匹配（可回退 CPU）");
    value.version = kCudaSiftAlgorithmVersion;
    value.inputModel = AlgorithmInputModel::ReusableFeatures;
    value.requiresCuda = false;
    value.suppliesStableFeatureIds = true;
    return value;
}

FeatureSet CudaSiftAlgorithm::extract(const ImageFeatureInput &input) const
{
    return SiftFeatureExtractor::extract(input, _config);
}

MatchResult CudaSiftAlgorithm::matchFeatures(const FeatureSet &features0,
                                             const FeatureSet &features1)
{
    validateSiftFeatures(features0);
    validateSiftFeatures(features1);

    if (_config.forceCpuSift)
    {
        return matchSiftOnCpu(features0, features1, _config.matchThreshold);
    }

#if defined(PLASCAN_HAS_CUDA_SIFT)
    try
    {
        checkCuda(cudaSetDevice(std::max(0, _config.cudaDevice)),
                  "device selection failed");

        SiftDataBuffer buffer0(features0);
        SiftDataBuffer buffer1(features1);
        MatchSiftData(buffer0.data(), buffer1.data());
        const std::vector<CudaSiftNearestMatch> forward = buffer0.nearestMatches();
        MatchSiftData(buffer1.data(), buffer0.data());
        const std::vector<CudaSiftNearestMatch> reverse = buffer1.nearestMatches();
        return filterCudaSiftMutualMatches(
            forward, reverse, _config.matchThreshold);
    }
    catch (...)
    {
        if (!_config.allowCpuSiftFallback)
        {
            throw;
        }
    }
    return matchSiftOnCpu(features0, features1, _config.matchThreshold);
#else
    if (_config.allowCpuSiftFallback)
    {
        return matchSiftOnCpu(features0, features1, _config.matchThreshold);
    }
    throw std::runtime_error(
        "CUDA SIFT matcher is unavailable and CPU fallback is disabled");
#endif
}

void registerCudaSiftAlgorithm()
{
    ImageMatchingAlgorithmDescriptor descriptor;
    descriptor.id = QString::fromLatin1(kCudaSiftAlgorithmId);
    descriptor.displayName = QStringLiteral("CUDA SIFT 匹配（可回退 CPU）");
    descriptor.version = kCudaSiftAlgorithmVersion;
    descriptor.inputModel = AlgorithmInputModel::ReusableFeatures;
    descriptor.requiresCuda = false;
    descriptor.suppliesStableFeatureIds = true;

    QString ignoredError;
    ImageMatchingRegistry::registerAlgorithm(
        descriptor,
        [](const ImageMatchingRuntimeConfig &config)
        {
            return std::make_unique<CudaSiftAlgorithm>(config);
        },
        &ignoredError);
}

} // namespace xjw::image_matching
