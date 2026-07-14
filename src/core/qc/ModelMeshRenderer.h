#pragma once

#include "ModelImageQualityTypes.h"

#include "MeshTypes.h"

namespace xjw::qc
{

class ModelMeshRenderer
{
public:
    ModelRenderResult render(const xjw::mesh::TriMesh &mesh,
                             const xjw::Camera &camera,
                             const cv::Size &imageSize) const;
};

} // namespace xjw::qc
