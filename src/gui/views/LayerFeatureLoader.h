#pragma once

#include <opencv2/core/types.hpp>

#include <QString>

#include <vector>

namespace xjw::gui::views
{

enum class FeaturePointSource
{
    ExtractedFeatures,
    RawMatches,
    ValidTiePoints
};

struct FeaturePointLoadResult
{
    std::vector<cv::KeyPoint> keypoints;
    QString sourcePath;
    QString message;
    bool available = false;
};

QString featurePointSourceToken(FeaturePointSource source);
FeaturePointSource featurePointSourceFromToken(const QString &token);
QString featurePointSourceDisplayName(FeaturePointSource source);

/**
 * @brief 从单影像匹配分片中读取参与过匹配的关键点观测。
 *
 * `.pimatch` 只保存真正被像对匹配引用的观测，不包含描述子。GUI 因而展示的
 * 点集与连接点、匹配统计使用同一份持久化结果，不再依赖独立特征文件。
 */
std::vector<cv::KeyPoint> loadMatchedKeypointsFromFile(const QString &matchFilePath);

std::vector<cv::KeyPoint> loadMatchedKeypointsForImage(const QString &plascanPath,
                                                       const QString &imagePath);

FeaturePointLoadResult loadFeaturePointsForImage(const QString &plascanPath,
                                                 const QString &imagePath,
                                                 FeaturePointSource source);

} // namespace xjw::gui::views
