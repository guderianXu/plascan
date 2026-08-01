#pragma once

#include <opencv2/core/types.hpp>

#include <QString>

#include <vector>

namespace xjw::gui::views
{

/**
 * @brief 从单影像匹配分片中读取参与过匹配的关键点观测。
 *
 * `.pimatch` 只保存真正被像对匹配引用的观测，不包含描述子。GUI 因而展示的
 * 点集与连接点、匹配统计使用同一份持久化结果，不再依赖独立特征文件。
 */
std::vector<cv::KeyPoint> loadMatchedKeypointsFromFile(const QString &matchFilePath);

std::vector<cv::KeyPoint> loadMatchedKeypointsForImage(const QString &plascanPath,
                                                       const QString &imagePath);

} // namespace xjw::gui::views
