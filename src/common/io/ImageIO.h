#pragma once

#include <opencv2/core/mat.hpp>

#include <QString>

#include <filesystem>
#include <string>
#include <vector>

namespace xjw::common::io
{

cv::Mat readImage(const QString &path, int flags);
cv::Mat readImage(const QString &path, int flags, QString *errorMessage);
cv::Mat readImage(const std::string &path, int flags);
cv::Mat readImage(const std::filesystem::path &path, int flags);

bool writeImage(const QString &path,
                const cv::Mat &image,
                const std::vector<int> &params = {});
bool writeImage(const std::string &path,
                const cv::Mat &image,
                const std::vector<int> &params = {});
bool writeImage(const std::filesystem::path &path,
                const cv::Mat &image,
                const std::vector<int> &params = {});

} // namespace xjw::common::io
