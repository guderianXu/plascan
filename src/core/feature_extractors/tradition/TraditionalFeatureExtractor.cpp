#include "TraditionalFeatureExtractor.h"

#include "FeatureData.h"

#include <torch/torch.h>

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

std::string TraditionalFeatureExtractor::normalizeAlgorithmName(const std::string &algorithmName)
{
    std::string normalized = algorithmName;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c)
                   {
                       return static_cast<char>(std::tolower(c));
                   });

    if (normalized == "orb" || normalized == "sift" || normalized == "superpoint"
        || normalized == "disk" || normalized == "aliked")
    {
        return normalized;
    }
    return "superpoint";
}

bool TraditionalFeatureExtractor::isTraditionalAlgorithm(const std::string &normalizedName)
{
    return normalizedName == "orb" || normalizedName == "sift";
}

SuperPointOutput TraditionalFeatureExtractor::detect(const cv::Mat &grayImage,
                                                     const SuperPointConfig &config,
                                                     const std::string &normalizedName)
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

    const int maxKpForDetector = config.max_num_keypoints > 0 ? config.max_num_keypoints : 20000;
    if (normalizedName == "orb")
    {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(maxKpForDetector);
        orb->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors, false);
    }
    else if (normalizedName == "sift")
    {
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(maxKpForDetector > 0 ? maxKpForDetector : 0);
        sift->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors, false);
    }
    else
    {
        throw std::runtime_error("unsupported traditional feature algorithm");
    }

    std::vector<int> keepIndices;
    keepIndices.reserve(keypoints.size());
    const int border = std::max(0, config.remove_borders);

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
        if (gray < config.grayscale_min || gray > config.grayscale_max)
        {
            continue;
        }

        keepIndices.push_back(static_cast<int>(index));
    }

    std::sort(keepIndices.begin(), keepIndices.end(), [&keypoints](int lhs, int rhs)
    {
        return keypoints[lhs].response > keypoints[rhs].response;
    });

    if (config.max_num_keypoints > 0 && static_cast<int>(keepIndices.size()) > config.max_num_keypoints)
    {
        keepIndices.resize(config.max_num_keypoints);
    }

    SuperPointOutput output;
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

    output.descriptors = FeatureData::cvDescriptorsToTensor(selectedDescriptors,
                                                            config.descriptor_dim,
                                                            normalizedName);
    return output;
}

} // namespace xjw::feature_extractors
