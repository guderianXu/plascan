#pragma once

#include "model/ModelFileResolver.h"

#include <QList>
#include <QString>

namespace xjw::common::model
{

/**
 * @brief GitHub Release 中的单个不可变模型资产。
 *
 * 文件大小和 SHA-256 固定在客户端目录中。下载器必须同时验证两者，避免网络
 * 中断、Release 资产误替换或代理缓存异常产生一个可打开但不可反序列化的 engine。
 */
struct ModelAssetFile
{
    QString fileName;
    QString downloadUrl;
    QString sha256;
    qint64 bytes = 0;
};

/**
 * @brief 可直接由 PlaScan 运行时加载的一组模型文件。
 *
 * packageDirectory 是相对于统一模型根目录的目录名；entryPointFile 是设置对话框
 * 下载成功后写回运行配置的模型、engine 或 manifest。compatibilitySummary 必须在
 * 下载界面展示模型格式、运行后端和平台兼容性约束。
 */
struct ModelAssetPackage
{
    QString id;
    QString displayName;
    QString packageDirectory;
    QString entryPointFile;
    QString releaseTag;
    QString compatibilitySummary;
    QList<ModelAssetFile> files;

    bool isValid() const;
    qint64 totalBytes() const;
};

/// 返回 RTX 50/SM 12.0 上生成的 SIFT + LightGlue K4096 运行包。
ModelAssetPackage lightGlueTensorRtPackage();

/// 返回跨平台 OpenCV DNN 可加载的 U2Net v1 ONNX 蒙版模型。
ModelAssetPackage u2NetOnnxPackage();

/// 返回指定 LoMa-R 静态 K 桶；非 1024/2048/3840 的值保守归一化为 1024。
ModelAssetPackage loMaRTensorRtPackage(int keypointBudget);

/// 计算包的可写安装目录。源码运行和安装版的根目录由 resolver 统一判定。
QString modelPackageInstallDirectory(const ModelAssetPackage &package,
                                     const ModelFileResolver &resolver = ModelFileResolver());

/// 返回下载完成后应写入算法设置的 engine/manifest 绝对路径。
QString modelPackageEntryPoint(const ModelAssetPackage &package,
                               const ModelFileResolver &resolver = ModelFileResolver());

} // namespace xjw::common::model
