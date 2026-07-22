#pragma once

#include <QMap>
#include <QJsonObject>
#include <QStringList>

namespace xjw::aerial_triangulation
{

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
