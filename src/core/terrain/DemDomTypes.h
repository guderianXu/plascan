#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>

#include <QString>

#include <opencv2/core.hpp>

namespace xjw
{

// Convenience alias for the CPU point cloud type
using PlaPointCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

/**
 * @brief DEM 栅格输出格式。
 */
enum class DemRasterFormat
{
    Float32Tiff,
    UInt16Tiff
};

/**
 * @brief DOM 图像输出格式。
 */
enum class DomImageFormat
{
    Png,
    Tiff
};

/**
 * @brief 地形产品投影参数。
 *
 * 当前用于保存坐标系描述与原点信息，便于未来扩展到 GeoTIFF/ENVI 元数据写出。
 */
struct DemProjectionParameters
{
    QString coordinateSystem;
    QString projectionWkt;
    double originX = 0.0;
    double originY = 0.0;
};

/**
 * @brief DEM 生成参数。
 */
struct DemGenerationOptions
{
    enum class ElevationAggregation
    {
        Mean,
        Min,
        Max,
        WeightedAverage,
        Median,
        StdDev,
        Count,
        Nmad,
        Percentile80
    };

    double gridResolution = 0.0;
    int minGridSize = 2;
    int maxGridSize = 4096;
    int holeFillIterations = 20;        // 增加到20次迭代，填充更多空洞
    int holeFillMinNeighbors = 3;       // 降低到3个邻居即可填充
    int holeFillSearchRadius = 5;       // 增加搜索半径到5像素
    bool useSubPixelBilinearSplat = true;
    ElevationAggregation elevationAggregation = ElevationAggregation::Mean;
    bool generateDenseCloud = false;
    bool generateMesh = true;
    DemRasterFormat rasterFormat = DemRasterFormat::Float32Tiff;
    DemProjectionParameters projection;
};

/**
 * @brief DOM 生成参数。
 */
struct DomGenerationOptions
{
    double outputResolution = 0.0;
    DomImageFormat imageFormat = DomImageFormat::Png;
    bool resizeToDem = true;
    bool enableSharpnessWeighting = true;
    bool enableExposureCompensation = true;
    double minBlendWeight = 0.05;
};

/**
 * @brief DEM 栅格数据。
 *
 * elevation 存储 Z 坐标（或高程）。当 worldX / worldY 非空时，
 * 表示完整的三维点云栅格（类似 ASP run-PC.tif 的 3-band XYZ 格式）。
 */
struct DemGridData
{
    int width = 0;
    int height = 0;
    double minX = 0.0;
    double minY = 0.0;
    double stepX = 1.0;
    double stepY = 1.0;
    cv::Mat elevation;   ///< Z 坐标 (CV_32FC1)
    cv::Mat worldX;      ///< X 坐标 (CV_32FC1)，可选
    cv::Mat worldY;      ///< Y 坐标 (CV_32FC1)，可选
    cv::Mat color;       ///< RGB 顶点颜色 (CV_8UC3)，可选
    cv::Mat validMask;
    cv::Mat triangulationError; ///< 三角化误差 (CV_32FC1)，可选
    cv::Mat pointCount;  ///< 每个 DEM 单元的输入点数量 (CV_32SC1)，可选
    cv::Mat confidence;  ///< 每个 DEM 单元的平均置信度 (CV_32FC1)，可选
    cv::Mat coverageMask; ///< 与 validMask 同尺寸的覆盖率掩码 (CV_8UC1)，可选
    DemProjectionParameters projection;

    bool hasWorldXY() const { return !worldX.empty() && !worldY.empty(); }
    bool hasColor() const { return !color.empty() && color.rows == height && color.cols == width && color.type() == CV_8UC3; }
    bool hasTriangulationError() const { return !triangulationError.empty(); }
    bool hasPointCount() const { return !pointCount.empty() && pointCount.rows == height && pointCount.cols == width; }
    bool hasConfidence() const { return !confidence.empty() && confidence.rows == height && confidence.cols == width; }
    bool hasCoverageMask() const { return !coverageMask.empty() && coverageMask.rows == height && coverageMask.cols == width; }

    bool isValid() const
    {
        return width > 0 && height > 0 &&
               !elevation.empty() && !validMask.empty() &&
               elevation.rows == height && elevation.cols == width &&
               validMask.rows == height && validMask.cols == width;
    }
};

/**
 * @brief DEM 产物摘要。
 */
struct DemArtifacts
{
    QString demPath;
    QString previewPath;
    QString denseCloudPath;
    QString meshPath;
    int densePointCount = 0;
    int meshVertexCount = 0;
    int meshFaceCount = 0;
};

/**
 * @brief DEM 质量栅格产物。
 */
struct DemQualityArtifacts
{
    QString errorPath;
    QString countPath;
    QString confidencePath;
    QString coveragePath;
};

/**
 * @brief DOM 产物摘要。
 */
struct DomArtifacts
{
    QString domPath;
    int sourceImageCount = 0;
    int width = 0;
    int height = 0;
};

/**
 * @brief 带纹理的网格输入（plapoint 原生类型）。
 *
 * 由 plapoint::io::readObj + cv::imread 填充，
 * 供 DemGenerator 与 DomGenerator 直接消费。
 * texture 为空时表示无纹理（只能生成 DEM，不能生成 DOM）。
 */
struct TerrainMeshInput
{
    PlaPointCloud mesh;
    cv::Mat texture;
};

} // namespace xjw
