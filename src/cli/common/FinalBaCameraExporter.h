#pragma once

/**
 * @file FinalBaCameraExporter.h
 * @brief 最终 BA 相机集的事务式 Tsai 导出接口。
 */

#include <placamera/instance_set.h>
#include <QString>
#include <QStringList>

#include <vector>

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
     * imageIds 与 images 一一对应，finalCameras 必须覆盖每幅影像；函数不接受部分相机集，
     * 也不会覆盖已存在的 outputDir。
     */
    bool exportFinalBaCameras(const QStringList& images,
                              const std::vector<placamera::ImageId>& imageIds,
                              const placamera::CameraInstanceSet& finalCameras,
                              const QString& outputDir,
                              FinalBaCameraExportResult* result,
                              QString* errorMessage = nullptr);

} // namespace xjw::cli
