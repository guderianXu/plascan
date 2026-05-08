#pragma once

// =============================================================================
// 文件: ObjMtlLoader.h (terrain)
// 功能: 加载 OBJ 网格及其 MTL 中引用的纹理，填充 TerrainMeshInput。
// =============================================================================

#include "DemDomTypes.h"

#include <QString>

namespace xjw
{

/**
 * @brief OBJ+MTL 带纹理网格加载器。
 *
 * 调用 plapoint::io::readObj 读取几何数据，自行解析 MTL 中的 map_Kd,
 * 再用 OpenCV 加载纹理图像。
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
     * @param out       输出 TerrainMeshInput，不得为空。
     * @param errorMsg  失败时写入错误说明，可为 nullptr。
     * @return 成功（顶点数据有效）时返回 true；OBJ 无法读取时返回 false。
     */
    static bool load(const QString &objPath,
                     TerrainMeshInput *out,
                     QString *errorMsg = nullptr);
};

} // namespace xjw
