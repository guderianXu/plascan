#pragma once

#include <opencv2/core/types.hpp>

#include <QString>

#include <vector>

namespace xjw::gui::views
{

std::vector<cv::KeyPoint> loadFeatureKeypointsFromFile(const QString &featurePath);

std::vector<cv::KeyPoint> loadFeatureKeypointsForImage(const QString &plascanPath,
                                                       const QString &imagePath);

} // namespace xjw::gui::views
