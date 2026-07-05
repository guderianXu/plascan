#pragma once

#include "match.h"

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::feature_match
{

enum class OutlierMethod
{
    None,
    FundamentalRansac,
    FundamentalUsacMagsac,
    HomographyRansac,
    AffineRansac
};

struct OutlierFilterConfig
{
    OutlierMethod method = OutlierMethod::FundamentalUsacMagsac;
    double reprojThreshold = 1.5;
    double confidence = 0.9999;
    int maxIters = 10000;
    int minInliers = 20;
};

class MatchGeometryFilter
{
public:
    static MatchResult filter(const MatchResult &input,
                              const std::vector<cv::KeyPoint> &keypoints0,
                              const std::vector<cv::KeyPoint> &keypoints1,
                              const OutlierFilterConfig &config,
                              int *inlierCount = nullptr);
};

} // namespace xjw::feature_match
