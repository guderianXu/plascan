#include "CameraTextureMapperV4.h"

#include "TextureMappingV4Internal.h"

#include <exception>

namespace xjw::mesh
{

bool generateCameraTexturedModelV4(const std::string &meshPath,
                                   const std::string &productsDir,
                                   const TextureMappingConfig &config,
                                   const QVector<MeshColorView> &views,
                                   TextureMappingResult *result,
                                   std::string *errorMsg)
{
    if (result)
    {
        *result = TextureMappingResult();
    }

    texture_v4::PipelineData data;
    try
    {
        if (!texture_v4::prepareInputs(meshPath, views, config, &data, result, errorMsg) ||
            !texture_v4::selectTextureViews(config, &data, result, errorMsg) ||
            !texture_v4::buildAndPackCharts(config, &data, result, errorMsg) ||
            !texture_v4::prepareTextureSourcePyramids(config, &data, result, errorMsg) ||
            !texture_v4::bakeAndExport(productsDir, config, &data, result, errorMsg))
        {
            return false;
        }
    }
    catch (const std::exception &exception)
    {
        if (errorMsg)
        {
            *errorMsg = "Natural 纹理生成失败（网格 " + meshPath + "）: " + exception.what();
        }
        return false;
    }

    result->textureAlgorithm = "recovered_natural_texture_v1";
    result->uvMethod = "natural_mapping_camera_charts";
    switch (config.blendMode)
    {
    case TextureBlendMode::BestView:
        result->blendMethod = "best_view";
        break;
    case TextureBlendMode::WeightedAverage:
        result->blendMethod = "weighted_average";
        break;
    case TextureBlendMode::Natural:
        result->blendMethod = "natural_multiband";
        break;
    }
    return true;
}

} // namespace xjw::mesh
