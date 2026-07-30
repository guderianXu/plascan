#include "CameraTextureMapperV4.h"

#include "TextureMappingV4Internal.h"

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
    if (!texture_v4::prepareInputs(meshPath, views, config, &data, result, errorMsg) ||
        !texture_v4::selectTextureViews(config, &data, result, errorMsg) ||
        !texture_v4::buildAndPackCharts(config, &data, result, errorMsg) ||
        !texture_v4::bakeAndExport(productsDir, config, &data, result, errorMsg))
    {
        return false;
    }

    result->textureAlgorithm = "camera_projected_atlas_v4";
    result->uvMethod = "connected_projective_charts";
    switch (config.blendMode)
    {
    case TextureBlendMode::BestView:
        result->blendMethod = "best_view";
        break;
    case TextureBlendMode::WeightedAverage:
        result->blendMethod = "weighted_average";
        break;
    case TextureBlendMode::Natural:
        result->blendMethod = "natural_robust";
        break;
    }
    return true;
}

} // namespace xjw::mesh
