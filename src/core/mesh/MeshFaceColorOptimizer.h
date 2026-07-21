#pragma once

#include "MeshColorizer.h"

namespace xjw::mesh
{

struct FaceColorCoherenceStatistics
{
    int assignedFaceCount = 0;
    int recoloredVertexCount = 0;
};

FaceColorCoherenceStatistics applyFaceCoherentPrimaryViews(
    TriMesh *mesh,
    const QVector<MeshColorView> &views,
    const MeshColorOptions &options);

} // namespace xjw::mesh
