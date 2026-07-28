#pragma once

#include <QMap>
#include <QJsonObject>
#include <QStringList>

namespace xjw::aerial_triangulation
{

// 仅显式导入、人工设置或来自有效 EXIF 的内参可作为下一次空三的先验。
// SfM 自标定结果和旧工程中缺少来源信息的相机都必须重新进入焦距搜索。
bool isTrustedProjectCameraIntrinsic(const QJsonObject &cameraObject);

struct CameraIntrinsicPriorSanitizationResult
{
    int inspectedCameraCount = 0;
    int dominantGroupCount = 0;
    int normalizedCameraCount = 0;
    double dominantMedianFocalPixels = 0.0;
    QStringList normalizedImagePaths;
};

// 无外部相机文件时，工程中可能保留上次 SfM 的错误焦距。
// 对同一组影像的显著焦距离群值做鲁棒回归，避免错误自标定污染新的空三初始化。
CameraIntrinsicPriorSanitizationResult sanitizeProjectCameraIntrinsicPriors(
    const QStringList &imagePaths,
    QMap<QString, QJsonObject> *cameraByPath);

} // namespace xjw::aerial_triangulation
