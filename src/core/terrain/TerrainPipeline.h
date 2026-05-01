#pragma once

#include <QString>
#include <QJsonObject>
#include <QStringList>

#include <vector>

namespace cv { class Mat; }

namespace xjw
{

class Camera;

/**
 * @brief 地形产品生成流水线外观类。
 *
 * 该类对上层维持稳定入口，内部将 DEM 与 DOM 能力分别委托给
 * DemGenerator、DomGenerator 与 DemDomIO 模块实现。
 */
class TerrainPipeline
{
public:
    /**
     * @brief 从点云文件生成 DEM 相关产品。
     *
     * 输入点云支持 OBJ、PLY 和 XYZ。输出包括：
     * 1. DEM 栅格文件 dem.tif
     * 2. DEM 预览图 depth_map.png
     * 3. 可选的栅格补密点云 dense_cloud.xyz
     * 4. 可选的三角网模型 model_from_dense.ply
     */
    static bool generateDemProducts(const QString &pointCloudPath,
                                    const QString &outputDir,
                                    double demResolution,
                                    const QString &demType,
                                    bool generateDenseCloud,
                                    QJsonObject *result,
                                    QString *errorMsg = nullptr);

    /**
     * @brief 从深度图 + 相机直接生成 DEM（图像空间栅格化，类似 ASP）。
     *
     * 直接在参考图像空间生成 DEM，避免点云中间步骤的覆盖率损失。
     * DEM 尺寸与参考深度图一致，覆盖率接近 100%。
     */
    static bool generateDemFromDepthMaps(const std::vector<cv::Mat> &depthMaps,
                                         const std::vector<Camera> &cameras,
                                         const QString &outputDir,
                                         QJsonObject *result,
                                         QString *errorMsg = nullptr);

    /**
     * @brief 结合 DEM 与输入影像生成 DOM。
     *
     * 当前实现采用多张影像统一缩放后平均叠加，并使用 DEM 有效区裁剪输出。
     */
    static bool generateOrthoProduct(const QStringList &images,
                                     const QString &demPath,
                                     const QString &outputPath,
                                     double resolution,
                                     const QJsonObject &projectMeta,
                                     QJsonObject *result,
                                     QString *errorMsg = nullptr);

    static bool generateOrthoProduct(const QStringList &images,
                                     const QString &demPath,
                                     const QString &outputPath,
                                     double resolution,
                                     QJsonObject *result,
                                     QString *errorMsg = nullptr);

    /**
     * @brief 从单个 OBJ+MTL+纹理文件同时生成 DEM 与 DOM。
     *
     * DEM：光栅化 OBJ 顶点 Z 坐标到规则 XY 网格。
     * DOM：三角光栅化 + 重心坐标 UV 插值，采样 MTL 引用的 PNG 纹理。
     *
     * 输出 result 键：
     *   dem_tif, depth_png, dom_png (无纹理时为空), has_texture,
     *   grid_width, grid_height, source_obj, created_at
     */
    static bool generateFromObjMtl(const QString &objPath,
                                   const QString &outputDir,
                                   double demResolution,
                                   QJsonObject *result,
                                   QString *errorMsg = nullptr);

    /**
     * @brief 从目录中所有 OBJ 瓦片拼接生成 DEM 与 DOM。
     *
     * 第一遍收集全部顶点位置计算整体边界并生成 DEM；
     * 第二遍对每块瓦片做带纹理光栅化并合并 DOM。
     * 适用于摄影测量分块重建的多瓦片结果。
     *
     * 输出 result 键：
     *   dem_tif, depth_png, dom_png, tile_count,
     *   grid_width, grid_height, source_dir, created_at
     */
    static bool generateFromObjMtlDir(const QString &dirPath,
                                      const QString &outputDir,
                                      double demResolution,
                                      QJsonObject *result,
                                      QString *errorMsg = nullptr);

    /**
     * @brief 从带纹理 OBJ 同时生成三种小天体投影的 DEM+DOM GeoTIFF 产品。
     *
     * 使用 AsteroidProjection 模块自动计算质心、参考半径与三轴椭球参数，
     * 为以下三种投影分别生成带 WKT 地理参考信息的 GeoTIFF：
     *   - 极射赤平投影  (stere)
     *   - 方位等距投影  (aeqd)
     *   - 三轴椭球投影  (triaxial)
     *
     * 输出文件（写入 outputDir/products/）：
     *   dem_stere.tif, dom_stere.tif
     *   dem_aeqd.tif,  dom_aeqd.tif
     *   dem_triaxial.tif, dom_triaxial.tif
     *   depth_map.png（原正射 DEM 预览，作为质量检查参考）
     *
     * 输出 result 键见实现注释。
     */
    static bool generateFromObjMtlWithAsteroidProjections(
        const QString &objPath,
        const QString &outputDir,
        double demResolution,
        QJsonObject *result,
        QString *errorMsg = nullptr);
};

} // namespace xjw
