#include "TraditionalFeatureExtractor.h"

#include "CudaSiftFeatureExtractor.h"
#include "FeatureData.h"

#include "OpenCvCompat.h"
#include "string_utils/StringTransform.h"

#include <torch/torch.h>

#if __has_include(<opencv2/xfeatures2d.hpp>)
#include <opencv2/xfeatures2d.hpp>
#define HAS_XFEATURES2D 1
#endif
#include <algorithm>
#include <cmath>
#include <stdexcept>

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

bool hasDarkPixelInNeighborhood(const cv::Mat &image,
                                const cv::KeyPoint &keypoint,
                                int radius,
                                float threshold)
{
    if (radius <= 0)
    {
        return false;
    }

    const int cx = static_cast<int>(std::round(keypoint.pt.x));
    const int cy = static_cast<int>(std::round(keypoint.pt.y));
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = cx + dx;
            const int y = cy + dy;
            if (!inImageRange(image, x, y))
            {
                return true;
            }
            if (normalizedGrayValue(image, x, y) < threshold)
            {
                return true;
            }
        }
    }
    return false;
}

} // namespace

namespace xjw::feature_extractors
{

TraditionalFeatureConfig traditionalFeatureConfigFromSuperPoint(const SuperPointConfig &config)
{
    TraditionalFeatureConfig traditionalConfig;
    traditionalConfig.maxKeypoints = config.max_num_keypoints;
    traditionalConfig.maxImageSize = config.max_image_size;
    traditionalConfig.removeBorders = config.remove_borders;
    traditionalConfig.descriptorDim = config.descriptor_dim;
    traditionalConfig.grayscaleMin = config.grayscale_min;
    traditionalConfig.grayscaleMax = config.grayscale_max;
    traditionalConfig.allowDeviceFallback = config.allow_device_fallback;
    traditionalConfig.detectionThreshold = config.detection_threshold;
    return traditionalConfig;
}

std::string TraditionalFeatureExtractor::normalizeAlgorithmName(const std::string &algorithmName)
{
    const std::string normalized = xjw::common::string_utils::asciiLowerCopy(algorithmName);

    if (normalized == "orb" || normalized == "sift" || normalized == "superpoint"
        || normalized == "disk" || normalized == "aliked"
        || normalized == "surf" || normalized == "akaze")
    {
        return normalized;
    }
    return "superpoint";
}

bool TraditionalFeatureExtractor::isTraditionalAlgorithm(const std::string &normalizedName)
{
    return normalizedName == "orb" || normalizedName == "sift"
        || normalizedName == "surf" || normalizedName == "akaze";
}

FeatureOutput TraditionalFeatureExtractor::detect(const cv::Mat &grayImage,
                                                  const TraditionalFeatureConfig &config,
                                                  const std::string &normalizedName)
{
    return detect(grayImage, config, normalizedName, false, 0);
}

FeatureOutput TraditionalFeatureExtractor::detect(const cv::Mat &grayImage,
                                                  const TraditionalFeatureConfig &config,
                                                  const std::string &normalizedName,
                                                  bool useCuda,
                                                  int cudaDevice)
{
    if (grayImage.empty())
    {
        throw std::runtime_error("input image is empty");
    }
    if (grayImage.type() != CV_8U)
    {
        throw std::runtime_error("traditional feature extractor expects CV_8U grayscale image");
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    const int maxKpForDetector = config.maxKeypoints > 0 ? config.maxKeypoints : 20000;
    if (normalizedName == "orb")
    {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(maxKpForDetector);
        orb->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors, false);
    }
    else if (normalizedName == "sift")
    {
        if (useCuda)
        {
            if (!isCudaSiftAvailable())
            {
                if (!config.allowDeviceFallback)
                {
                    throw std::runtime_error("CUDA SIFT requested but no CUDA SIFT device is available");
                }
            }
            else
            {
                try
                {
                    SuperPointConfig cudaConfig;
                    cudaConfig.max_num_keypoints = config.maxKeypoints;
                    cudaConfig.max_image_size = config.maxImageSize;
                    cudaConfig.remove_borders = config.removeBorders;
                    cudaConfig.descriptor_dim = config.descriptorDim;
                    cudaConfig.grayscale_min = config.grayscaleMin;
                    cudaConfig.grayscale_max = config.grayscaleMax;
                    cudaConfig.allow_device_fallback = config.allowDeviceFallback;
                    cudaConfig.detection_threshold = config.detectionThreshold;
                    return detectCudaSift(grayImage, cudaConfig, cudaDevice);
                }
                catch (...)
                {
                    if (!config.allowDeviceFallback)
                    {
                        throw;
                    }
                }
            }
        }

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(maxKpForDetector > 0 ? maxKpForDetector : 0);
        sift->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors, false);
    }
    else if (normalizedName == "akaze")
    {
        cv::Ptr<xjw::opencv_compat::AkazeFeature> akaze = xjw::opencv_compat::AkazeFeature::create();
        akaze->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors, false);
    }
    else if (normalizedName == "surf")
    {
        // SURF (GPU via CUDA if available, else CPU)
        // hessianThreshold: 越大关键点越少 (默认 100, 卫星影像推荐 400-800)
#ifdef HAS_XFEATURES2D
        auto surf = cv::xfeatures2d::SURF::create(400.0, 4, 3, false, false);
        surf->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors, false);
#else
        throw std::runtime_error("SURF requires opencv_contrib (xfeatures2d not available)");
#endif
    }
    else
    {
        throw std::runtime_error("unsupported traditional feature algorithm");
    }

    std::vector<int> keepIndices;
    keepIndices.reserve(keypoints.size());
    const int border = std::max(0, config.removeBorders);

    for (size_t index = 0; index < keypoints.size(); ++index)
    {
        const cv::KeyPoint &keypoint = keypoints[index];
        const int x = static_cast<int>(std::round(keypoint.pt.x));
        const int y = static_cast<int>(std::round(keypoint.pt.y));
        if (!inImageRange(grayImage, x, y))
        {
            continue;
        }
        if (x < border || y < border || x >= grayImage.cols - border || y >= grayImage.rows - border)
        {
            continue;
        }

        const float gray = normalizedGrayValue(grayImage, x, y);
        if (gray < config.grayscaleMin || gray > config.grayscaleMax)
        {
            continue;
        }

        keepIndices.push_back(static_cast<int>(index));
    }

    std::sort(keepIndices.begin(), keepIndices.end(), [&keypoints](int lhs, int rhs)
    {
        return keypoints[lhs].response > keypoints[rhs].response;
    });

    if (config.maxKeypoints > 0 && static_cast<int>(keepIndices.size()) > config.maxKeypoints)
    {
        keepIndices.resize(config.maxKeypoints);
    }

    FeatureOutput output;
    output.imageWidth = grayImage.cols;
    output.imageHeight = grayImage.rows;
    output.keypoints.reserve(keepIndices.size());
    output.scores.reserve(keepIndices.size());

    cv::Mat selectedDescriptors;
    if (!descriptors.empty())
    {
        selectedDescriptors = cv::Mat(static_cast<int>(keepIndices.size()), descriptors.cols, descriptors.type());
    }

    for (size_t outputIndex = 0; outputIndex < keepIndices.size(); ++outputIndex)
    {
        const int keypointIndex = keepIndices[outputIndex];
        output.keypoints.push_back(keypoints[keypointIndex]);
        output.scores.push_back(keypoints[keypointIndex].response);
        if (!selectedDescriptors.empty())
        {
            descriptors.row(keypointIndex).copyTo(selectedDescriptors.row(static_cast<int>(outputIndex)));
        }
    }

    int descriptorDim = config.descriptorDim;
    if (normalizedName == "sift" && !selectedDescriptors.empty())
    {
        descriptorDim = selectedDescriptors.cols;
    }

    output.descriptors = FeatureData::cvDescriptorsToTensor(selectedDescriptors,
                                                            descriptorDim,
                                                            normalizedName);
    return output;
}

FeatureOutput TraditionalFeatureExtractor::detect(const cv::Mat &grayImage,
                                                  const SuperPointConfig &config,
                                                  const std::string &normalizedName)
{
    return detect(grayImage, traditionalFeatureConfigFromSuperPoint(config), normalizedName, false, 0);
}

FeatureOutput TraditionalFeatureExtractor::detect(const cv::Mat &grayImage,
                                                  const SuperPointConfig &config,
                                                  const std::string &normalizedName,
                                                  bool useCuda,
                                                  int cudaDevice)
{
    return detect(grayImage,
                  traditionalFeatureConfigFromSuperPoint(config),
                  normalizedName,
                  useCuda,
                  cudaDevice);
}

} // namespace xjw::feature_extractors
