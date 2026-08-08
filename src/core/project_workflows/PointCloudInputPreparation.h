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
 * @brief 加载并过滤正式 SfM 稀疏点云，为 MVS 深度范围估计提供输入。
 *
 * 设备选择与后续点云阶段保持一致；Auto 按 CUDA、OpenCL、CPU 逐级选择。
 */
PointCloudInputPreparationResult preparePointCloudInput(
    const QString &sparseCloudPath,
    const std::vector<xjw::mvs::CameraView> &views,
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto);

} // namespace xjw::core::project
