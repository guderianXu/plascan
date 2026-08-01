#pragma once

/**
 * @file ReferenceTerrainPrior.h
 * @brief 将参考 DEM/地形栅格转换为 BA 高程平面软约束。
 *
 * 栅格采用仿射原点加像元尺寸的二维索引，当前仅约束世界 Z 与插值高程的距离；
 * 不负责坐标参考系转换。调用方必须保证稀疏点和 DEM 已位于同一平面/高程坐标系。
 */

#include "BundleAdjust.h"

#include <vector>

namespace xjw
{

struct ReferenceTerrainGrid
{
    int width = 0; ///< 栅格列数。
    int height = 0; ///< 栅格行数。
    double originX = 0.0; ///< 左上/首像元参考 X，语义由输入仿射约定决定。
    double originY = 0.0; ///< 左上/首像元参考 Y。
    double pixelSizeX = 1.0; ///< 每列 X 增量，可为负。
    double pixelSizeY = 1.0; ///< 每行 Y 增量，可为负。
    double nodata = -9999.0; ///< 无效高程哨兵。
    std::vector<double> heights; ///< 行优先，数量必须等于 width*height。
};

struct ReferenceTerrainPriorOptions
{
    bool enabled = true; ///< false 时不附加任何约束。
    double sigmaMeters = 1.0; ///< 高程先验标准差，用于残差归一化。
    double maxAssociationDistanceMeters = 2.0; ///< 初始点与 DEM 高程的最大关联距离。
    double huberDeltaMeters = 0.5; ///< BA 中该约束的 Huber 阈值。
};

struct ReferenceTerrainPriorStats
{
    int inputTrackCount = 0; ///< 检查的 BA track 数。
    int associatedTrackCount = 0; ///< 成功附加高程约束数。
    int rejectedNoHeightCount = 0; ///< 落在栅格外或邻域含 nodata 数。
    int rejectedByDistanceCount = 0; ///< 初值与 DEM 相差过大数。
    double rmsBeforeMeters = 0.0; ///< 关联点优化前高程残差 RMS。
    double medianAbsBeforeMeters = 0.0; ///< 关联点优化前绝对残差中位数。
};

/// 参考地形先验的采样、关联和 BA 配置助手。
class ReferenceTerrainPrior
{
public:
    /// 对 (x,y) 双线性采样；越界、退化像元或 nodata 邻域返回 nodata 并置 ok=false。
    static double sampleHeight(const ReferenceTerrainGrid &grid,
                               double x,
                               double y,
                               bool *ok = nullptr);

    /// 对满足距离门控的 track 追加水平高程平面约束，并返回关联统计。
    static ReferenceTerrainPriorStats attachHeightPlaneConstraints(const ReferenceTerrainGrid &grid,
                                                                   std::vector<BATrack> *tracks,
                                                                   const ReferenceTerrainPriorOptions &options);

    /// 生成启用激光/点到面约束所需的 BAOptions 片段。
    static BAOptions makeBundleAdjustOptions(const ReferenceTerrainPriorOptions &options);
};

} // namespace xjw
