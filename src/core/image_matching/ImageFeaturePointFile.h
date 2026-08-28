#pragma once

/**
 * @file ImageFeaturePointFile.h
 * @brief 逐影像持久化特征点几何，供 GUI 复查提取结果。
 *
 * 描述子仍然只存在于匹配任务内存。该文件仅保存显示和审计需要的坐标、尺度、
 * 方向与响应值，避免为了查看特征点重新运行模型或把大描述子写入工程。
 */

#include "ImageMatchTypes.h"

#include <QString>

#include <cstdint>
#include <vector>

namespace xjw::image_matching
{

inline constexpr std::uint32_t kImageFeaturePointFormatVersion = 1;
inline constexpr const char *kImageFeaturePointFileSuffix = ".pifeature";

struct ImageFeaturePointCatalog
{
    ImageIdentity owner;
    QString algorithmId;
    std::uint32_t algorithmVersion = 0;
    std::uint32_t featureSchemaVersion = 0;
    std::vector<KeypointObservation> observations;
};

class ImageFeaturePointFile
{
public:
    static QString filePathForImage(const QString &directory,
                                    const QString &imagePath);

    static bool write(const QString &filePath,
                      const ImageFeaturePointCatalog &catalog,
                      QString *errorMessage = nullptr);

    static bool read(const QString &filePath,
                     ImageFeaturePointCatalog *catalog,
                     QString *errorMessage = nullptr);
};

} // namespace xjw::image_matching
