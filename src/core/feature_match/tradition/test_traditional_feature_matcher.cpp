#include "OpenCvCompat.h"
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "TraditionalFeatureMatcher.h"

namespace
{

cv::Mat loadGrayImage(const std::string &path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        throw std::runtime_error("failed to load image: " + path);
    }
    return image;
}

void detectOrb(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
    orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors, false);
}

void detectSift(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(4000);
    sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors, false);
}

} // namespace

int main()
{
    try
    {
        const std::string image0Path = std::string(TEST_DATA_DIR) + "/img/1.png";
        const std::string image1Path = std::string(TEST_DATA_DIR) + "/img/2.png";

        const cv::Mat image0 = loadGrayImage(image0Path);
        const cv::Mat image1 = loadGrayImage(image1Path);

        std::vector<cv::KeyPoint> orbKp0;
        std::vector<cv::KeyPoint> orbKp1;
        cv::Mat orbDesc0;
        cv::Mat orbDesc1;
        detectOrb(image0, orbKp0, orbDesc0);
        detectOrb(image1, orbKp1, orbDesc1);

        xjw::feature_match::tradition::TraditionalMatchConfig orbCfg;
        orbCfg.algorithmName = "orb_bf_hamming";
        const xjw::feature_match::MatchResult orbMatches =
            xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
                orbDesc0,
                orbDesc1,
                static_cast<int>(orbKp0.size()),
                static_cast<int>(orbKp1.size()),
                orbCfg);

        std::vector<cv::KeyPoint> siftKp0;
        std::vector<cv::KeyPoint> siftKp1;
        cv::Mat siftDesc0;
        cv::Mat siftDesc1;
        detectSift(image0, siftKp0, siftDesc0);
        detectSift(image1, siftKp1, siftDesc1);

        xjw::feature_match::tradition::TraditionalMatchConfig siftBfCfg;
        siftBfCfg.algorithmName = "sift_bf_l2";
        const xjw::feature_match::MatchResult siftBfMatches =
            xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
                siftDesc0,
                siftDesc1,
                static_cast<int>(siftKp0.size()),
                static_cast<int>(siftKp1.size()),
                siftBfCfg);

        xjw::feature_match::tradition::TraditionalMatchConfig siftFlannCfg;
        siftFlannCfg.algorithmName = "sift_flann";
        const xjw::feature_match::MatchResult siftFlannMatches =
            xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
                siftDesc0,
                siftDesc1,
                static_cast<int>(siftKp0.size()),
                static_cast<int>(siftKp1.size()),
                siftFlannCfg);

        std::cout << "ORB keypoints: " << orbKp0.size() << " / " << orbKp1.size() << std::endl;
        std::cout << "ORB matches: " << orbMatches.numMatches << std::endl;
        std::cout << "SIFT keypoints: " << siftKp0.size() << " / " << siftKp1.size() << std::endl;
        std::cout << "SIFT BF matches: " << siftBfMatches.numMatches << std::endl;
        std::cout << "SIFT FLANN matches: " << siftFlannMatches.numMatches << std::endl;

        if (orbMatches.empty() || siftBfMatches.empty() || siftFlannMatches.empty())
        {
            std::cerr << "[ERROR] Traditional matcher diagnostic failed: one or more algorithms produced zero matches."
                      << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "[OK] Traditional matcher diagnostic passed." << std::endl;
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
