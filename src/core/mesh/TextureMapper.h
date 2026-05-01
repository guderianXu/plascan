#pragma once

#include <functional>
#include <string>

namespace xjw::mesh
{

/**
 * @brief 纹理映射配置。
 */
struct TextureMappingConfig
{
    int textureSize = 4096;
    int padding = 4;
    bool keepUnmapped = true;
    std::string blendMethod = "加权平均";
    std::string uvMethod = "自动 (基于面法线)";
    std::function<void(const std::string &, int)> progressFn;
};

/**
 * @brief 纹理映射输出结果。
 */
struct TextureMappingResult
{
    std::string modelObjPath;
    std::string modelMtlPath;
    std::string texturePngPath;
    int textureSize = 0;
    std::string textureAlgorithm;
    std::string uvMethod;
    std::string blendMethod;
};

/**
 * @brief 负责为网格生成 UV、纹理图集以及 OBJ/MTL 输出。
 */
class TextureMapper
{
public:
    /**
     * @brief 从网格文件生成带纹理模型。
     */
    static bool generateTexturedModelFromMeshFile(const std::string &meshPath,
                                                  const std::string &productsDir,
                                                  const TextureMappingConfig &config,
                                                  TextureMappingResult *result,
                                                  std::string *errorMsg = nullptr);
};

} // namespace xjw::mesh