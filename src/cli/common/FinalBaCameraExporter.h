#pragma once

/**
 * @file FinalBaCameraExporter.h
 * @brief 最终 BA 相机集的事务式 Tsai 导出接口。
 */

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

namespace xjw::cli
{

/// imageCameraList 可直接作为后续重建 CLI 的输入清单。
struct FinalBaCameraExportResult
{
    QString outputDir;
    QString imageCameraList;
    QStringList cameraPaths;
};

/**
 * @brief 导出与 images 一一对应的最终 BA 相机及影像配对清单。
 *
 * finalCameraMetadata 必须覆盖每幅影像；函数不接受部分相机集，也不会覆盖已存在的
 * outputDir。调用方必须只在正式重建成功后传入胜出模型的相机元数据。
 */
bool exportFinalBaCameras(const QStringList &images,
                          const QMap<QString, QJsonObject> &finalCameraMetadata,
                          const QString &outputDir,
                          FinalBaCameraExportResult *result,
                          QString *errorMessage = nullptr);

} // namespace xjw::cli
