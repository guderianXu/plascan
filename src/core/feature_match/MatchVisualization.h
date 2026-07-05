#pragma once

#include "match.h"

#include <QString>

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::feature_match
{

bool saveMatchVisualization(const cv::Mat &image0,
                            const cv::Mat &image1,
                            const std::vector<cv::KeyPoint> &keypoints0,
                            const std::vector<cv::KeyPoint> &keypoints1,
                            const MatchResult &result,
                            const QString &outputPath);

} // namespace xjw::feature_match
