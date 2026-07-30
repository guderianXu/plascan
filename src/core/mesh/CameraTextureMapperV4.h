#pragma once

#include "TextureMapper.h"

#include <QVector>

namespace xjw::mesh
{

struct MeshColorView;

bool generateCameraTexturedModelV4(const std::string &meshPath,
                                   const std::string &productsDir,
                                   const TextureMappingConfig &config,
                                   const QVector<MeshColorView> &views,
                                   TextureMappingResult *result,
                                   std::string *errorMsg);

} // namespace xjw::mesh
