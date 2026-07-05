#include "CudaSiftFeatureExtractor.h"

#include "FeatureData.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(PLASCAN_HAS_CUDA_SIFT)
#include <cuda_runtime.h>

#include "cudaImage.h"
#include "cudaSift.h"
#endif

namespace
{

bool inImageRange(const cv::Mat &image, int x, int y)
{
    return x >= 0 && y >= 0 && x < image.cols && y < image.rows;
}

float normalizedGrayValue(const cv::Mat &image, int x, int y)
{
    const unsigned char value = image.at<unsigned char>(y, x);
    return static_cast<float>(value) / 255.0f;
}

bool passesCommonFilters(const cv::Mat &image,
                         const cv::KeyPoint &keypoint,
                         const SuperPointConfig &config)
{
    const int x = static_cast<int>(std::round(keypoint.pt.x));
    const int y = static_cast<int>(std::round(keypoint.pt.y));
    if (!inImageRange(image, x, y))
    {
        return false;
    }

    const int border = std::max(0, config.remove_borders);
    if (x < border || y < border || x >= image.cols - border || y >= image.rows - border)
    {
        return false;
    }

    const float gray = normalizedGrayValue(image, x, y);
    return gray >= config.grayscale_min && gray <= config.grayscale_max;
}

float finiteOrDefault(float value, float fallback)
{
    return std::isfinite(value) ? value : fallback;
}

float cudaSiftThreshold(const SuperPointConfig &config)
{
    if (config.detection_threshold > 1.0f)
    {
        return config.detection_threshold;
    }
    return std::clamp(config.detection_threshold * 1000.0f, 0.5f, 20.0f);
}

#if defined(PLASCAN_HAS_CUDA_SIFT)
struct SiftDataGuard
{
    explicit SiftDataGuard(int maxPoints)
    {
        InitSiftData(data, maxPoints, true, true);
        initialized = true;
    }

    ~SiftDataGuard()
    {
        if (initialized)
        {
            FreeSiftData(data);
        }
    }

    SiftDataGuard(const SiftDataGuard &) = delete;
    SiftDataGuard &operator=(const SiftDataGuard &) = delete;

    SiftData data{};
    bool initialized = false;
};

struct TempMemoryGuard
{
    TempMemoryGuard(int width, int height, int octaves)
    {
        memory = AllocSiftTempMemory(width, height, octaves, false);
    }

    ~TempMemoryGuard()
    {
        FreeSiftTempMemory(memory);
    }

    TempMemoryGuard(const TempMemoryGuard &) = delete;
    TempMemoryGuard &operator=(const TempMemoryGuard &) = delete;

    float *memory = nullptr;
};
#endif

} // namespace

namespace xjw::feature_extractors
{

bool isCudaSiftAvailable()
{
#if defined(PLASCAN_HAS_CUDA_SIFT)
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0;
#else
    return false;
#endif
}

FeatureOutput detectCudaSift(const cv::Mat &grayImage,
                             const SuperPointConfig &config,
                             int cudaDevice)
{
#if !defined(PLASCAN_HAS_CUDA_SIFT)
    (void)grayImage;
    (void)config;
    (void)cudaDevice;
    throw std::runtime_error("CUDA SIFT backend is not built");
#else
    if (grayImage.empty())
    {
        throw std::runtime_error("input image is empty");
    }
    if (grayImage.type() != CV_8U)
    {
        throw std::runtime_error("CUDA SIFT expects CV_8U grayscale image");
    }

    InitCuda(std::max(0, cudaDevice));

    cv::Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F);
    if (!floatImage.isContinuous())
    {
        floatImage = floatImage.clone();
    }

    CudaImage cudaImage;
    cudaImage.Allocate(grayImage.cols,
                       grayImage.rows,
                       iAlignUp(grayImage.cols, 128),
                       false,
                       nullptr,
                       reinterpret_cast<float *>(floatImage.data));
    cudaImage.Download();

    const int maxPoints = std::max(config.max_num_keypoints > 0 ? config.max_num_keypoints * 2 : 20000,
                                   1024);
    constexpr int numOctaves = 5;
    SiftDataGuard siftData(maxPoints);
    TempMemoryGuard tempMemory(grayImage.cols, grayImage.rows, numOctaves);

    ExtractSift(siftData.data,
                cudaImage,
                numOctaves,
                1.0,
                cudaSiftThreshold(config),
                0.0f,
                false,
                tempMemory.memory);

#ifdef MANAGEDMEM
    SiftPoint *points = siftData.data.m_data;
#else
    SiftPoint *points = siftData.data.h_data;
#endif
    const int pointCount = std::max(0, std::min(siftData.data.numPts, siftData.data.maxPts));

    std::vector<int> keepIndices;
    keepIndices.reserve(static_cast<std::size_t>(pointCount));
    std::vector<cv::KeyPoint> allKeypoints(static_cast<std::size_t>(pointCount));
    for (int index = 0; index < pointCount; ++index)
    {
        const SiftPoint &point = points[index];
        cv::KeyPoint keypoint;
        keypoint.pt.x = finiteOrDefault(point.xpos, 0.0f);
        keypoint.pt.y = finiteOrDefault(point.ypos, 0.0f);
        keypoint.size = std::max(1.0f, finiteOrDefault(point.scale, 1.0f));
        keypoint.angle = finiteOrDefault(point.orientation, -1.0f);
        keypoint.response = finiteOrDefault(std::abs(point.sharpness), 0.0f);
        allKeypoints[static_cast<std::size_t>(index)] = keypoint;

        if (passesCommonFilters(grayImage, keypoint, config))
        {
            keepIndices.push_back(index);
        }
    }

    std::sort(keepIndices.begin(), keepIndices.end(), [&allKeypoints](int lhs, int rhs)
    {
        return allKeypoints[static_cast<std::size_t>(lhs)].response >
               allKeypoints[static_cast<std::size_t>(rhs)].response;
    });

    if (config.max_num_keypoints > 0 && static_cast<int>(keepIndices.size()) > config.max_num_keypoints)
    {
        keepIndices.resize(static_cast<std::size_t>(config.max_num_keypoints));
    }

    cv::Mat descriptors(static_cast<int>(keepIndices.size()), 128, CV_32F);
    FeatureOutput output;
    output.imageWidth = grayImage.cols;
    output.imageHeight = grayImage.rows;
    output.keypoints.reserve(keepIndices.size());
    output.scores.reserve(keepIndices.size());

    for (std::size_t outputIndex = 0; outputIndex < keepIndices.size(); ++outputIndex)
    {
        const int sourceIndex = keepIndices[outputIndex];
        const SiftPoint &point = points[sourceIndex];
        const cv::KeyPoint &keypoint = allKeypoints[static_cast<std::size_t>(sourceIndex)];

        output.keypoints.push_back(keypoint);
        output.scores.push_back(keypoint.response);
        float *dst = descriptors.ptr<float>(static_cast<int>(outputIndex));
        std::copy(point.data, point.data + 128, dst);
    }

    output.descriptors = FeatureData::cvDescriptorsToTensor(descriptors, 128, "sift");
    return output;
#endif
}

} // namespace xjw::feature_extractors
