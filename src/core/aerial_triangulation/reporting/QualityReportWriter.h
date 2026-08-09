#pragma once

/**
 * @file QualityReportWriter.h
 * @brief 从最终 SfmReconstruction 生成逐点、逐相机和全局质量报告。
 *
 * 该类只构建 JSON，不写文件。所有统计必须基于最终 BA/重三角化后的状态，
 * 禁止复用候选试算中的过期误差。
 */

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"

#include <QJsonArray>
#include <QJsonObject>

namespace xjw
{
class SfmReconstruction;
}

namespace xjw::aerial_triangulation
{

/// 一次正式稀疏结果的完整报告片段。
struct SparseQualityReport
{
    QJsonArray points; ///< 每点坐标、轨迹长度、误差和交会角。
    QJsonArray perCameraResiduals; ///< 每相机观测数与残差分布。
    QJsonObject qualityMetadata; ///< 工程/MVS 消费的稳定质量字段。
    QJsonObject diagnostics; ///< 更详细的开发和候选分析信息。
};

/// 无状态质量报告构建器。
class QualityReportWriter
{
public:
    /**
     * @brief 仅计算焦距候选排序需要的网络质量摘要。
     *
     * 该路径不构造逐点、逐观测或逐相机 QJsonArray，适合并行候选结束后的快速排序。
     */
    static QJsonObject buildSparseQualitySummary(
        const PreparedAerialTriangulationInput &input,
        const xjw::SfmReconstruction &reconstruction,
        const AerialTriangulationReconstructionResult &result);

    /// 根据最终重建和工作流上下文生成报告，不修改 reconstruction/result。
    static SparseQualityReport build(
        const PreparedAerialTriangulationInput &input,
        const xjw::SfmReconstruction &reconstruction,
        const AerialTriangulationReconstructionResult &result);
};

} // namespace xjw::aerial_triangulation
