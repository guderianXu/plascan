#pragma once

#include "MeshTypes.h"

#include <QJsonObject>
#include <QString>

#include <functional>

namespace xjw::mesh
{

struct RpcPlaneSweepModelResult
{
    TriMesh mesh;
    QJsonObject diagnostics;
};

// Builds the standard-RPC production mesh directly from exactly two decoded
// RPC00B rasters.  Failure is terminal; it never delegates to a pinhole,
// recovered signed-depth/OOC, TSDF, Poisson, or sparse-DEM path.
RpcPlaneSweepModelResult buildRpcPlaneSweepModel(const QJsonObject& settings,
                                                 int targetFaces,
                                                 const std::function<bool()>& isCancelled,
                                                 const std::function<void(const QString&, int)>& progress);

} // namespace xjw::mesh
