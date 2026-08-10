#include "CudaSiftAlgorithm.h"

#include "CudaSiftMatchFilter.h"
#include "../ImageMatchingRegistry.h"
#include "../sift/SiftFeatureExtractor.h"

#include "cudaSift.h"

#include <cuda_runtime_api.h>

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

    ~SiftDataBuffer()
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

void validateSiftFeatures(const FeatureSet &features)
{
    if (!features.isConsistent() || features.sourceAlgorithm != "sift" ||
        features.descriptors.type() != CV_32F || features.descriptors.cols != 128)
    {
        throw std::invalid_argument(
            "CUDA SIFT matcher requires consistent CV_32F SIFT descriptors");
    }
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
    value.displayName = QStringLiteral("CUDA SIFT 匹配");
    value.version = kCudaSiftAlgorithmVersion;
    value.inputModel = AlgorithmInputModel::ReusableFeatures;
    value.requiresCuda = true;
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
    checkCuda(cudaSetDevice(std::max(0, _config.cudaDevice)),
              "device selection failed");

    SiftDataBuffer buffer0(features0);
    SiftDataBuffer buffer1(features1);
    MatchSiftData(buffer0.data(), buffer1.data());
    const std::vector<CudaSiftNearestMatch> forward = buffer0.nearestMatches();
    MatchSiftData(buffer1.data(), buffer0.data());
    const std::vector<CudaSiftNearestMatch> reverse = buffer1.nearestMatches();
    return filterCudaSiftMutualMatches(forward, reverse, _config.matchThreshold);
}

void registerCudaSiftAlgorithm()
{
    ImageMatchingAlgorithmDescriptor descriptor;
    descriptor.id = QString::fromLatin1(kCudaSiftAlgorithmId);
    descriptor.displayName = QStringLiteral("CUDA SIFT 匹配");
    descriptor.version = kCudaSiftAlgorithmVersion;
    descriptor.inputModel = AlgorithmInputModel::ReusableFeatures;
    descriptor.requiresCuda = true;
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
