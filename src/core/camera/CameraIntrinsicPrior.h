#pragma once

#include "io/ImageExifMetadata.h"

#include <QString>

#include <optional>

/**
 * @brief 从影像元数据得到的针孔相机焦距先验。
 *
 * focalScale 与空三模块约定一致，等于“像素焦距 / 影像最长边”。
 * strong 表示来源足以跳过无标定焦距粗搜索，但仍不等同于实验室标定文件。
 */
struct CameraIntrinsicPrior
{
    double focalPixels = 0.0;
    double focalScale = 0.0;
    QString source;
    QString make;
    QString model;
    bool strong = false;
};

/**
 * @brief 根据已解析 EXIF 和影像尺寸估计焦距先验。
 *
 * 优先使用 35 mm 等效焦距；其次使用物理焦距与已知传感器宽度；最后仅对
 * 固定镜头、型号唯一的相机使用内置目录。目录是可扩展数据表，不依赖文件名猜测。
 */
std::optional<CameraIntrinsicPrior> estimateCameraIntrinsicPrior(
    const xjw::common::io::ImageExifMetadata &metadata,
    int imageWidth,
    int imageHeight);
