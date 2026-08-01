#pragma once

/**
 * @file AerialTriangulationResultWriter.h
 * @brief 将已选中的内存 SfM 模型事务式转换为正式空三资产。
 *
 * 写出器负责稀疏 PLY、相机更新、质量报告和工程结果扩展数据；不参与候选排序，
 * 也不在失败候选上产生“正式”结果。
 */

#include "model/AerialTriangulationOptions.h"
#include "reconstruction/SfmAttemptRunner.h"

#include <QString>

namespace xjw::aerial_triangulation
{

/// 正式结果提交器。
class AerialTriangulationResultWriter
{
public:
    /**
     * @brief 写出稀疏云并填充 execution.result 的回写字段。
     *
     * execution/reconstruction 必须来自同一 SfmAttemptRunner。成功前所有相机更新
     * 保存在 pendingCamUpdates，由上层项目服务统一提交，避免部分写回。
     */
    bool write(const PreparedAerialTriangulationInput &input,
               SfmAttemptExecutionResult *execution,
               QString *errorMessage = nullptr) const;
};

} // namespace xjw::aerial_triangulation
