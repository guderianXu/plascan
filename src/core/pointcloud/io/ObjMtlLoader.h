#pragma once

// =============================================================================
// 文件: ObjMtlLoader.h
// 功能: 加载 OBJ 网格及其 MTL 中引用的纹理，填充 TexturedMesh。
// =============================================================================

#include "data/PointCloud.h"

#include <QString>

#include <opencv2/core.hpp>

namespace xjw::pointcloud
{

/**
 * @brief 带纹理的 OBJ 网格。
 *
 * 由 ObjMtlLoader::load() 填充，供 DemGenerator 与 DomGenerator 直接消费。
 * texture 为空时表示无纹理（只能生成 DEM，不能生成 DOM）。
 */
struct TexturedMesh
{
    PointCloud mesh;   ///< 顶点 + 逐顶点 UV + 三角面
    cv::Mat texture;   ///< BGR 纹理图像，empty() 时表示无纹理
};

/**
 * @brief OBJ+MTL 带纹理网格加载器。
 *
 * 调用 PointCloudIO::readObjPointCloud 读取几何数据（同时解析 MTL 中的 map_Kd）,
 * 再用 OpenCV 按 textureImageFile() 加载纹理图像。
 *
 * 若 MTL 文件缺失、不含 map_Kd 或图像加载失败，则 out->texture 为空，
 * 不视为错误——调用方可检查 out->texture.empty() 决定是否生成 DOM。
 */
class ObjMtlLoader
{
public:
    /**
     * @brief 加载单个 OBJ 文件及其引用的第一张纹理。
     *
     * @param objPath   OBJ 文件路径。
     * @param out       输出 TexturedMesh，不得为空。
     * @param errorMsg  失败时写入错误说明，可为 nullptr。
     * @return 成功（顶点数据有效）时返回 true；OBJ 无法读取时返回 false。
     */
    static bool load(const QString &objPath,
                     TexturedMesh *out,
                     QString *errorMsg = nullptr);
};

} // namespace xjw::pointcloud
