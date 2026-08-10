#include "SiftFeatureExtractor.h"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(PLASCAN_HAS_CUDA_SIFT)
#include <cuda_runtime.h>

#include "cudaImage.h"
#include "cudaSift.h"
#endif

namespace xjw::image_matching
{
namespace
{

bool pointIsAllowed(const ImageFeatureInput &input,
                    const cv::KeyPoint &keypoint,
                    const ImageMatchingRuntimeConfig &runtime)
{
    const int x = static_cast<int>(std::lround(keypoint.pt.x));
    const int y = static_cast<int>(std::lround(keypoint.pt.y));
    if (x < 0 || y < 0 || x >= input.grayImage.cols || y >= input.grayImage.rows)
    {
        return false;
    }
    const int border = std::max(0, runtime.removeBorders);
    if (x < border || y < border || x >= input.grayImage.cols - border ||
        y >= input.grayImage.rows - border)
    {
        return false;
    }

    const float gray = static_cast<float>(input.grayImage.at<unsigned char>(y, x)) / 255.0f;
    if (gray < runtime.grayscaleMin || gray > runtime.grayscaleMax)
    {
        return false;
    }
    return input.validMask.empty() || input.validMask.at<unsigned char>(y, x) != 0;
}

void restoreOriginalCoordinates(FeatureSet *features, double coordinateScale)
{
    if (!features || coordinateScale <= 0.0 || coordinateScale == 1.0)
    {
        return;
    }
    const float scale = static_cast<float>(coordinateScale);
    for (cv::KeyPoint &keypoint : features->keypoints)
    {
        keypoint.pt.x *= scale;
        keypoint.pt.y *= scale;
        keypoint.size *= scale;
    }
}

FeatureSet selectAndConvert(const ImageFeatureInput &input,
                            const ImageMatchingRuntimeConfig &runtime,
                            const std::vector<cv::KeyPoint> &keypoints,
                            const cv::Mat &descriptors)
{
    std::vector<int> selected;
    selected.reserve(keypoints.size());
    for (int index = 0; index < static_cast<int>(keypoints.size()); ++index)
    {
        if (pointIsAllowed(input, keypoints[static_cast<std::size_t>(index)], runtime))
        {
            selected.push_back(index);
        }
    }
    std::stable_sort(selected.begin(), selected.end(),
                     [&](int left, int right)
                     {
                         const float leftResponse = keypoints[static_cast<std::size_t>(left)].response;
                         const float rightResponse = keypoints[static_cast<std::size_t>(right)].response;
                         return leftResponse == rightResponse ? left < right
                                                              : leftResponse > rightResponse;
                     });
    if (runtime.maxKeypoints > 0 &&
        static_cast<int>(selected.size()) > runtime.maxKeypoints)
    {
        selected.resize(static_cast<std::size_t>(runtime.maxKeypoints));
    }

    FeatureSet features;
    features.imageWidth = input.originalWidth > 0 ? input.originalWidth : input.grayImage.cols;
    features.imageHeight = input.originalHeight > 0 ? input.originalHeight : input.grayImage.rows;
    features.keypoints.reserve(selected.size());
    features.scores.reserve(selected.size());
    if (!descriptors.empty())
    {
        features.descriptors.create(static_cast<int>(selected.size()), descriptors.cols, CV_32F);
    }
    for (int outputIndex = 0; outputIndex < static_cast<int>(selected.size()); ++outputIndex)
    {
        const int sourceIndex = selected[static_cast<std::size_t>(outputIndex)];
        features.keypoints.push_back(keypoints[static_cast<std::size_t>(sourceIndex)]);
        features.scores.push_back(keypoints[static_cast<std::size_t>(sourceIndex)].response);
        if (!features.descriptors.empty())
        {
            cv::Mat sourceRow;
            if (descriptors.type() == CV_32F)
            {
                sourceRow = descriptors.row(sourceIndex);
            }
            else
            {
                descriptors.row(sourceIndex).convertTo(sourceRow, CV_32F);
            }
            sourceRow.copyTo(features.descriptors.row(outputIndex));
        }
    }
    restoreOriginalCoordinates(&features, input.coordinateScale);
    return features;
}

FeatureSet extractCpu(const ImageFeatureInput &input,
                      const ImageMatchingRuntimeConfig &runtime)
{
    const int detectorLimit = runtime.maxKeypoints > 0 ? runtime.maxKeypoints * 2 : 0;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(detectorLimit);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(input.grayImage, cv::noArray(), keypoints, descriptors, false);
    return selectAndConvert(input, runtime, keypoints, descriptors);
}

#if defined(PLASCAN_HAS_CUDA_SIFT)
struct SiftDataGuard
{
    explicit SiftDataGuard(int maxPoints)
    {
        InitSiftData(data, maxPoints, true, true);
    }

    ~SiftDataGuard()
    {
        FreeSiftData(data);
    }

    SiftDataGuard(const SiftDataGuard &) = delete;
    SiftDataGuard &operator=(const SiftDataGuard &) = delete;
    SiftData data{};
};

struct TempMemoryGuard
{
    TempMemoryGuard(int width, int height, int octaves)
        : data(AllocSiftTempMemory(width, height, octaves, false))
    {
    }

    ~TempMemoryGuard()
    {
        FreeSiftTempMemory(data);
    }

    TempMemoryGuard(const TempMemoryGuard &) = delete;
    TempMemoryGuard &operator=(const TempMemoryGuard &) = delete;
    float *data = nullptr;
};

FeatureSet extractCuda(const ImageFeatureInput &input,
                       const ImageMatchingRuntimeConfig &runtime)
{
    InitCuda(std::max(0, runtime.cudaDevice));

    cv::Mat floatImage;
    input.grayImage.convertTo(floatImage, CV_32F);
    if (!floatImage.isContinuous())
    {
        floatImage = floatImage.clone();
    }

    CudaImage cudaImage;
    cudaImage.Allocate(input.grayImage.cols,
                       input.grayImage.rows,
                       iAlignUp(input.grayImage.cols, 128),
                       false,
                       nullptr,
                       reinterpret_cast<float *>(floatImage.data));
    cudaImage.Download();

    const int maximumPoints = std::max(runtime.maxKeypoints > 0
                                           ? runtime.maxKeypoints * 2
                                           : 40000,
                                       1024);
    constexpr int octaveCount = 5;
    SiftDataGuard siftData(maximumPoints);
    TempMemoryGuard temporary(input.grayImage.cols, input.grayImage.rows, octaveCount);
    const float threshold = runtime.siftDetectionThreshold > 1.0f
        ? runtime.siftDetectionThreshold
        : std::clamp(runtime.siftDetectionThreshold * 1000.0f, 0.1f, 20.0f);
    ExtractSift(siftData.data,
                cudaImage,
                octaveCount,
                1.0,
                threshold,
                0.0f,
                false,
                temporary.data);

#ifdef MANAGEDMEM
    const SiftPoint *points = siftData.data.m_data;
#else
    const SiftPoint *points = siftData.data.h_data;
#endif
    const int pointCount = std::clamp(siftData.data.numPts, 0, siftData.data.maxPts);
    std::vector<cv::KeyPoint> keypoints(static_cast<std::size_t>(pointCount));
    cv::Mat descriptors(pointCount, 128, CV_32F);
    for (int index = 0; index < pointCount; ++index)
    {
        const SiftPoint &point = points[index];
        cv::KeyPoint &keypoint = keypoints[static_cast<std::size_t>(index)];
        keypoint.pt.x = std::isfinite(point.xpos) ? point.xpos : 0.0f;
        keypoint.pt.y = std::isfinite(point.ypos) ? point.ypos : 0.0f;
        keypoint.size = std::max(1.0f, std::isfinite(point.scale) ? point.scale : 1.0f);
        keypoint.angle = std::isfinite(point.orientation) ? point.orientation : -1.0f;
        keypoint.response = std::isfinite(point.sharpness) ? std::abs(point.sharpness) : 0.0f;
        std::copy(point.data, point.data + 128, descriptors.ptr<float>(index));
    }
    return selectAndConvert(input, runtime, keypoints, descriptors);
}
#endif

} // namespace

bool SiftFeatureExtractor::isCudaAvailable(int deviceIndex)
{
#if defined(PLASCAN_HAS_CUDA_SIFT)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess ||
        deviceIndex < 0 || deviceIndex >= count)
    {
        return false;
    }

    int previousDevice = 0;
    const bool restoreDevice = cudaGetDevice(&previousDevice) == cudaSuccess;
    const cudaError_t selectStatus = cudaSetDevice(deviceIndex);
    const cudaError_t initializeStatus = selectStatus == cudaSuccess
        ? cudaFree(nullptr)
        : selectStatus;
    if (restoreDevice && previousDevice != deviceIndex)
    {
        cudaSetDevice(previousDevice);
    }
    return initializeStatus == cudaSuccess;
#else
    (void)deviceIndex;
    return false;
#endif
}

FeatureSet SiftFeatureExtractor::extract(const ImageFeatureInput &input,
                                         const ImageMatchingRuntimeConfig &runtime,
                                         bool *usedCuda)
{
    if (usedCuda)
    {
        *usedCuda = false;
    }
    if (input.grayImage.empty() || input.grayImage.type() != CV_8U)
    {
        throw std::invalid_argument("SIFT requires a non-empty CV_8U grayscale image");
    }
    if (!input.validMask.empty() &&
        (input.validMask.type() != CV_8U || input.validMask.size() != input.grayImage.size()))
    {
        throw std::invalid_argument("SIFT valid mask must be CV_8U and match the input image size");
    }

#if defined(PLASCAN_HAS_CUDA_SIFT)
    if (!runtime.forceCpuSift && isCudaAvailable(runtime.cudaDevice))
    {
        try
        {
            FeatureSet features = extractCuda(input, runtime);
            if (usedCuda)
            {
                *usedCuda = true;
            }
            return features;
        }
        catch (...)
        {
            if (!runtime.allowCpuSiftFallback)
            {
                throw;
            }
        }
    }
#endif
    if (!runtime.allowCpuSiftFallback)
    {
        throw std::runtime_error("CUDA SIFT is unavailable and CPU fallback is disabled");
    }
    return extractCpu(input, runtime);
}

} // namespace xjw::image_matching
