#pragma once

#include "DepthTsdfSurfaceBuilder.h"
#include "MeshBoundaryAttribution.h"

#include <QJsonObject>
#include <QStringList>

#include <vector>

namespace xjw::mesh
{

QJsonObject buildMeshAcquisitionGapReport(
    const std::vector<MeshBoundaryEdgeAttribution> &edges,
    const DepthTsdfLayout &layout,
    const QStringList &sourceLabels,
    const std::vector<int> &sourceIndices,
    int inputFrameCount,
    bool sourceMappingComplete);

} // namespace xjw::mesh
