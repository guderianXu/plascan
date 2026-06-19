#pragma once

#include "DemDomTypes.h"

#include <plapoint/core/point_cloud.h>

#include <QString>

namespace xjw
{

/**
 * @brief DEM/DOM 文件读写辅助类。
 */
class DemDomIO
{
public:
    /** @brief 写出 DEM 预览 PNG。 */
    static bool writeDemPreviewPng(const DemGridData &demGrid,
                                   const QString &outputPath,
                                   QString *errorMsg = nullptr);

    /** @brief 写出 DEM 栅格。 */
    static bool writeDemRaster(const DemGridData &demGrid,
                               const QString &outputPath,
                               DemRasterFormat format,
                               QString *errorMsg = nullptr);

    /** @brief 读取 DEM 栅格。 */
    static bool readDemRaster(const QString &inputPath,
                              DemGridData *demGrid,
                              QString *errorMsg = nullptr);

    /** @brief 写出 DEM 的误差、点数、置信度和覆盖率质量栅格。 */
    static bool writeDemQualityRasters(const DemGridData &demGrid,
                                       const QString &outputDir,
                                       DemQualityArtifacts *artifacts,
                                       QString *errorMsg = nullptr);

    /** @brief 写出栅格补密后的 XYZ 点云。 */
    static bool writeDenseCloudXyz(const PlaPointCloud &denseCloud,
                                   const QString &outputPath,
                                   QString *errorMsg = nullptr);

    /** @brief 将 DEM 栅格写出为 PLY 三角网。 */
    static bool writeMeshPlyFromDemGrid(const DemGridData &demGrid,
                                        const QString &outputPath,
                                        int *vertexCount,
                                        int *faceCount,
                                        QString *errorMsg = nullptr);

    /** @brief 将 DEM 栅格写出为 PLY 密集点云。 */
    static bool writeDenseCloudPly(const DemGridData &demGrid,
                                   const QString &outputPath,
                                   int *pointCount = nullptr,
                                   QString *errorMsg = nullptr);

    /** @brief 写出 DOM 图像。 */
    static bool writeDomImage(const cv::Mat &domImage,
                              const QString &outputPath,
                              DomImageFormat format,
                              QString *errorMsg = nullptr);

    /**
     * @brief 写出带地理坐标参考的 DOM GeoTIFF。
     *
     * 与 writeDemRaster() 使用相同的地变换参数和投影 WKT，
     * 适用于小行星投影等需要 DOM 与 DEM 严格配准的场景。
     *
     * @param domImage   BGR 彩色 DOM 图像（CV_8UC3）
     * @param demGrid    提供范围、步长与投影 WKT 的 DEM 栅格元数据
     * @param outputPath 输出文件路径（.tif）
     */
    static bool writeDomGeoTiff(const cv::Mat &domImage,
                                const DemGridData &demGrid,
                                const QString &outputPath,
                                QString *errorMsg = nullptr);
};

} // namespace xjw
