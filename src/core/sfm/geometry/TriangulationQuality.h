#pragma once

/**
 * @file TriangulationQuality.h
 * @brief SfM 清理连接点时的投影几何质量计算。
 *
 * 只保留在质量报告中实际使用的逐点指标；交会和有符号投影由 PlaCamera 负责。
 */

#include <placamera/frame_camera.h>

#include <array>
#include <limits>
#include <vector>

namespace xjw
{

    struct TiePointQualityObservation
    {
        const placamera::FramePinholeModel* camera = nullptr;
        double measurementScale = 1.0;
        std::array<double, 2> imagePoint{
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}};
    };

    /**
     * @brief 与 Metashape 2.3.2 Clean Tie Points 对齐的逐点质量指标。
     *
     * `reprojectionError` 是按投影尺度归一化后的最大残差；
     * `reconstructionUncertainty` 是固定相机、无尺度加权点法矩阵的条件数平方根；
     * `imageCount` 和 `projectionAccuracy` 分别是有效观测数与原始投影尺度均值。
     */
    struct CleanTiePointQuality
    {
        double reprojectionError = 0.0;
        double reconstructionUncertainty = 0.0;
        std::size_t imageCount = 0;
        double projectionAccuracy = 0.0;
        bool hasProjectionGeometry = false;
    };

    CleanTiePointQuality evaluateCleanTiePointQuality(const std::vector<TiePointQualityObservation>& observations,
                                                      const std::array<double, 3>& worldPoint);

    /**
     * @brief 计算固定相机条件下的三维点重建不确定度。
     *
     * 返回点协方差椭球最大/最小半轴之比，即点法矩阵条件数的平方根。
     * 相机内外参不确定度不传播到该指标；退化几何返回有限上限值，输入不足返回 NaN。
     */
    double reconstructionUncertainty(const std::vector<TiePointQualityObservation>& observations,
                                     const std::array<double, 3>& worldPoint);

    /// 返回全部观测特征尺度的算术平均；任一尺度缺失或非法时返回 NaN。
    double projectionAccuracy(const std::vector<TiePointQualityObservation>& observations);

} // namespace xjw
