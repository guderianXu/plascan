#pragma once

#include "MeshTopologyQuality.h"

#include <QJsonObject>
#include <QString>

#include <cstdint>
#include <vector>

namespace xjw::mesh
{

enum DepthTsdfRecoveryTransactionRejectionFlag : std::uint32_t
{
    DepthTsdfRecoveryTransactionAccepted = 0,
    DepthTsdfRecoveryTransactionInvalidBaseline = 1U << 0U,
    DepthTsdfRecoveryTransactionInvalidCandidate = 1U << 1U,
    DepthTsdfRecoveryTransactionBoundaryNotReduced = 1U << 2U,
    DepthTsdfRecoveryTransactionComponentGrowth = 1U << 3U,
    DepthTsdfRecoveryTransactionNonManifoldEdgeGrowth = 1U << 4U,
    DepthTsdfRecoveryTransactionNonManifoldVertexGrowth = 1U << 5U,
    DepthTsdfRecoveryTransactionTopologicalComplexityGrowth = 1U << 6U,
    DepthTsdfRecoveryTransactionLargestComponentRatioRegression = 1U << 7U,
    DepthTsdfRecoveryTransactionExtremeAspectRatioRegression = 1U << 8U,
    DepthTsdfRecoveryTransactionSupportSizeMismatch = 1U << 9U
};

struct DepthTsdfRecoveryTransactionOptions
{
    double ratioTolerance = 1.0e-12;
};

struct DepthTsdfRecoveryTransactionEvaluation
{
    bool accepted = false;
    std::uint32_t rejectionFlags = DepthTsdfRecoveryTransactionAccepted;
    QString reason;
    QJsonObject diagnostics;
};

/**
 * Compares a recovery candidate against an extraction made from the same TSDF
 * field and the support mask captured before recovery.
 *
 * Recovery is fail-closed: boundary edges must decrease strictly. All other
 * protected topology and triangle-quality metrics must be no worse than the
 * baseline.
 */
DepthTsdfRecoveryTransactionEvaluation evaluateDepthTsdfRecoveryTransaction(
    const MeshTopologyQualityStatistics &baseline,
    const MeshTopologyQualityStatistics &candidate,
    const DepthTsdfRecoveryTransactionOptions &options = {});

DepthTsdfRecoveryTransactionEvaluation evaluateDepthTsdfRecoveryTransaction(
    const TriMesh &baseline,
    const TriMesh &candidate,
    const DepthTsdfRecoveryTransactionOptions &options = {});

/**
 * Leaves candidateSupport untouched when the evaluation is accepted. On
 * rejection (or a support-size mismatch), restores the exact baseline mask.
 */
bool commitDepthTsdfRecoveryTransaction(
    const DepthTsdfRecoveryTransactionEvaluation &evaluation,
    const std::vector<std::uint8_t> &baselineSupport,
    std::vector<std::uint8_t> *candidateSupport);

} // namespace xjw::mesh
