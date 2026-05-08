#pragma once

#include "DemDomTypes.h"

#include <plapoint/core/point_cloud.h>
#include "Camera.h"

#include <QString>
#include <opencv2/core.hpp>
#include <vector>

namespace xjw
{

/**
 * @brief DEM 核心生成器。
 *
 * 负责将输入点云栅格化为 DEM，并按需要导出栅格补密后的点云表示。
 */
class DemGenerator
{
public:
    /**
     * @brief 从点云生成 DEM 栅格。
     */
    static bool generateFromPointCloud(const PlaPointCloud &pointCloud,
                                       const DemGenerationOptions &options,
                                       DemGridData *demGrid,
                                       PlaPointCloud *denseCloud,
                                       QString *errorMsg = nullptr);

    /**
     * @brief 从深度图直接生成 DEM（类似 ASP 的方法）。
     *
     * 这个方法直接从深度图生成 DEM，避免通过稀疏点云重采样导致的覆盖率损失。
     * 每个深度图像素都会被反投影到 3D 空间，然后直接栅格化到 DEM 网格。
     *
     * @param depthMaps 深度图列表（每个深度图对应一个视图）
     * @param cameras 相机参数列表（与深度图一一对应）
     * @param options DEM 生成参数
     * @param demGrid 输出的 DEM 栅格数据
     * @param errorMsg 错误信息（可选）
     * @return 成功返回 true，失败返回 false
     */
    static bool generateFromDepthMaps(const std::vector<cv::Mat> &depthMaps,
                                      const std::vector<Camera> &cameras,
                                      const DemGenerationOptions &options,
                                      DemGridData *demGrid,
                                      QString *errorMsg = nullptr);

    /**
     * @brief 根据点云与参数估算 DEM 栅格宽度。
     */
    static int estimateGridResolution(const PlaPointCloud &pointCloud,
                                      const DemGenerationOptions &options);
};

} // namespace xjw