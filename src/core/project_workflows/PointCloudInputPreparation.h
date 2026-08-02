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
 * 这里固定使用 CPU 后端，避免预处理与随后启动的 CUDA PatchMatch 同时抢占显存。
 */
PointCloudInputPreparationResult preparePointCloudInput(
    const QString &sparseCloudPath,
    const std::vector<xjw::mvs::CameraView> &views);

} // namespace xjw::core::project
