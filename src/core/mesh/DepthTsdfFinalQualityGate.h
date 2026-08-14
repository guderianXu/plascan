#pragma once

#include "MeshTopologyQuality.h"

#include <QJsonObject>
#include <QString>

namespace xjw::mesh
{

enum class DepthTsdfFinalQualityPolicy
{
    Strict,
    ObservationOnly
};

struct DepthTsdfFinalQualityGateResult
{
    bool passed = false;
    QString reason;
    QJsonObject diagnostics;
};

/**
 * Evaluates the mesh that is about to be persisted by the depth-TSDF workflow.
 *
 * Observation-only output is allowed to have arbitrary open boundaries. Every
 * other strict quality constraint remains active, including manifoldness,
 * component dominance, triangle quality and topological complexity.
 */
DepthTsdfFinalQualityGateResult evaluateDepthTsdfFinalQualityGate(
    const MeshTopologyQualityStatistics &statistics,
    DepthTsdfFinalQualityPolicy policy,
    const MeshTopologyQualityThresholds &strictThresholds = {});

DepthTsdfFinalQualityGateResult evaluateDepthTsdfFinalQualityGate(
    const TriMesh &mesh,
    DepthTsdfFinalQualityPolicy policy,
    const MeshTopologyQualityThresholds &strictThresholds = {});

} // namespace xjw::mesh
