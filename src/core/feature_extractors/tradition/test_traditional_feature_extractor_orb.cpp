#include "OpenCvCompat.h"
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "TraditionalFeatureExtractor.h"

namespace
{

struct OrbFilterStats
{
    int rawCount = 0;
    int afterBorder = 0;
    int afterGrayRange = 0;
};

bool inImageRange(const cv::Mat &image, int x, int y)
{
    return x >= 0 && y >= 0 && x < image.cols && y < image.rows;
}

float normalizedGrayValue(const cv::Mat &image, int x, int y)
{
    return static_cast<float>(image.at<unsigned char>(y, x)) / 255.0f;
}

OrbFilterStats collectOrbStats(const cv::Mat &grayImage, const SuperPointConfig &config)
{
    OrbFilterStats stats;

    const int maxFeatures = config.max_num_keypoints > 0 ? config.max_num_keypoints : 20000;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(maxFeatures);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors, false);
    stats.rawCount = static_cast<int>(keypoints.size());

    const int border = std::max(0, config.remove_borders);

    for (const cv::KeyPoint &keypoint : keypoints)
    {
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
        ++stats.afterBorder;

        const float gray = normalizedGrayValue(grayImage, x, y);
        if (gray < config.grayscale_min || gray > config.grayscale_max)
        {
            continue;
        }
        ++stats.afterGrayRange;
    }

    return stats;
}

std::string buildDiagnosticMessage(const OrbFilterStats &stats)
{
    std::ostringstream oss;
    oss << "ORB diagnostic counts => raw=" << stats.rawCount
        << ", afterBorder=" << stats.afterBorder
        << ", afterGrayRange=" << stats.afterGrayRange;
    return oss.str();
}

} // namespace

int main()
{
    try
    {
        const std::filesystem::path imagePath = std::filesystem::path(TEST_DATA_DIR) / "img" / "1.png";
        if (!std::filesystem::exists(imagePath))
        {
            std::cerr << "[ERROR] Test image not found: " << imagePath.string() << std::endl;
            return 2;
        }

        cv::Mat grayImage = cv::imread(imagePath.string(), cv::IMREAD_GRAYSCALE);
        if (grayImage.empty())
        {
            std::cerr << "[ERROR] Failed to read image: " << imagePath.string() << std::endl;
            return 3;
        }

        SuperPointConfig config;
        const OrbFilterStats stats = collectOrbStats(grayImage, config);
        std::cout << buildDiagnosticMessage(stats) << std::endl;

        if (stats.rawCount <= 0)
        {
            std::cerr << "[ERROR] Raw ORB produced no keypoints." << std::endl;
            return 4;
        }

        const auto output = xjw::feature_extractors::TraditionalFeatureExtractor::detect(grayImage, config, "orb");
        std::cout << "TraditionalFeatureExtractor output keypoints=" << output.keypoints.size() << std::endl;

        if (static_cast<int>(output.keypoints.size()) != stats.afterGrayRange)
        {
            std::cerr << "[ERROR] output count mismatch, output=" << output.keypoints.size()
                      << ", expected=" << stats.afterGrayRange << std::endl;
            return 5;
        }

        if (output.keypoints.empty())
        {
            std::cerr << "[ERROR] TraditionalFeatureExtractor filtered out all ORB keypoints. "
                      << buildDiagnosticMessage(stats) << std::endl;
            return 6;
        }

        std::cout << "[OK] ORB diagnostic passed." << std::endl;
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[EXCEPTION] " << ex.what() << std::endl;
        return 7;
    }
}
