#pragma once

#include "../match.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace xjw::feature_match::tradition
{

struct TraditionalMatchConfig
{
    std::string algorithmName;
    float ratioTestThreshold = 0.75f;
    bool requireMutualConsistency = true;
};

class TraditionalFeatureMatcher
{
public:
    static std::string normalizeAlgorithmName(const std::string &algorithmName);

    static xjw::feature_match::MatchResult match(const cv::Mat &descriptors0,
                                                 const cv::Mat &descriptors1,
                                                 int numKeypoints0,
                                                 int numKeypoints1,
                                                 const TraditionalMatchConfig &config);

    static std::vector<cv::DMatch> matchDescriptors(const cv::Mat &descriptors0,
                                                    const cv::Mat &descriptors1,
                                                    const TraditionalMatchConfig &config);
};

} // namespace xjw::feature_match::tradition
