#pragma once

#include <QString>

#include <optional>

namespace xjw::common::io
{

/// 空三初始化所需的最小 EXIF 子集；数值保持物理单位，不混入像素换算。
struct ImageExifMetadata
{
    QString make;
    QString model;
    std::optional<double> focalLengthMm;
    std::optional<double> focalLength35Mm;
};

/**
 * @brief 从 JPEG APP1/TIFF EXIF 读取相机型号和焦距。
 * @details 不依赖 GUI 图像插件；支持 Intel/Motorola 字节序、ASCII、SHORT、LONG 和 RATIONAL。
 */
std::optional<ImageExifMetadata> readImageExifMetadata(const QString &path,
                                                       QString *errorMessage = nullptr);

} // namespace xjw::common::io
