#pragma once

#include "MvsTypes.h"

#include <QString>

#include <vector>

namespace xjw::core::project
{

/**
 * @brief 点云创建工作流的稀疏点云准备结果。
 *
 * 预处理放在 GUI 线程之外执行。失败原因必须保留下来，避免把“空点云”
 * 继续传给 PatchMatch 后才得到难以定位的深度范围错误。
 */
struct PointCloudInputPreparationResult
{
    bool ok = false;
    xjw::mvs::SparseCloud cloud;
    QString errorMessage;
};

/**
 * @brief 加载正式 SfM 稀疏点和逐点观测 track，供 recovered 深度场景使用。
 *
 * 必须提供正式点观测 sidecar，以保持点与 track 身份；只有 PLY 时明确失败。
 */
PointCloudInputPreparationResult preparePointCloudInput(
    const QString &sparseCloudPath,
    const std::vector<xjw::mvs::CameraView> &views,
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto,
    const QString &sparsePointSidecarPath = {});

} // namespace xjw::core::project
