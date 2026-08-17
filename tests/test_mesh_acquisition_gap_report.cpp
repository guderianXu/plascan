#include "MeshAcquisitionGapReport.h"

#include <gtest/gtest.h>

#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>

#include <array>
#include <cstdint>
#include <vector>

namespace
{

xjw::mesh::DepthTsdfLayout makeLayout()
{
    xjw::mesh::DepthTsdfLayout layout;
    layout.ok = true;
    layout.boundsMin = {-1.0f, -1.0f, -1.0f};
    layout.boundsMax = {1.0f, 1.0f, 1.0f};
    layout.cells = {2, 2, 2};
    layout.voxelSize = {1.0f, 1.0f, 1.0f};
    layout.sampleCount = 27;
    return layout;
}

xjw::mesh::MeshBoundaryEdgeAttribution edge(
    std::array<float, 3> midpoint,
    xjw::mesh::MeshBoundaryAttributionReason primary,
    xjw::mesh::MeshBoundaryAttributionReason cause,
    xjw::mesh::DepthGeometrySourceMask source_mask)
{
    xjw::mesh::MeshBoundaryEdgeAttribution result;
    result.midpoint = midpoint;
    result.firstPoint = midpoint;
    result.secondPoint = midpoint;
    result.normal = midpoint;
    result.reason = primary;
    result.evidenceReason = cause;
    result.sourceMask = source_mask;
    result.length = 0.5f;
    return result;
}

} // namespace

TEST(MeshAcquisitionGapReportTest, AccountsForEveryEdgeAndSourceWithoutDuplication)
{
    const std::vector<xjw::mesh::MeshBoundaryEdgeAttribution> edges{
        edge({1.0f, 0.0f, 0.0f},
             xjw::mesh::MeshBoundaryAttributionReason::NoObservation,
             xjw::mesh::MeshBoundaryAttributionReason::NoObservation,
             0),
        edge({-1.0f, 0.0f, 0.8f},
             xjw::mesh::MeshBoundaryAttributionReason::SupportGateRejected,
             xjw::mesh::MeshBoundaryAttributionReason::InsufficientSource,
             0x1),
        edge({0.0f, 1.0f, -0.8f},
             xjw::mesh::MeshBoundaryAttributionReason::DepthSpreadRejected,
             xjw::mesh::MeshBoundaryAttributionReason::DepthSpreadRejected,
             0x3),
        edge({0.0f, 0.0f, 0.0f},
             xjw::mesh::MeshBoundaryAttributionReason::ExtractionOrPostprocess,
             xjw::mesh::MeshBoundaryAttributionReason::ExtractionOrPostprocess,
             0)};
    QStringList labels;
    std::vector<int> source_indices;
    for (int index = 0; index < 16; ++index)
    {
        labels.append(QStringLiteral("frame_%1").arg(index));
        source_indices.push_back(100 + index);
    }

    const QJsonObject report = xjw::mesh::buildMeshAcquisitionGapReport(
        edges, makeLayout(), labels, source_indices, 16, true);

    EXPECT_EQ(report.value(QStringLiteral("boundary_edge_count")).toInt(), 4);
    EXPECT_TRUE(report.value(QStringLiteral("records_complete")).toBool());
    EXPECT_EQ(report.value(QStringLiteral("orientation_bin_count")).toInt(), 36);
    EXPECT_EQ(
        report.value(QStringLiteral("unknown_orientation_edge_count")).toInt(),
        1);
    const QJsonObject causes =
        report.value(QStringLiteral("root_cause_counts")).toObject();
    EXPECT_EQ(causes.value(QStringLiteral("no_observation")).toInt(), 1);
    EXPECT_EQ(causes.value(QStringLiteral("insufficient_source")).toInt(), 1);
    EXPECT_EQ(causes.value(QStringLiteral("depth_spread")).toInt(), 1);
    EXPECT_EQ(
        causes.value(QStringLiteral("extraction_or_postprocess")).toInt(), 1);

    const QJsonArray bins =
        report.value(QStringLiteral("orientation_bins")).toArray();
    ASSERT_EQ(bins.size(), 36);
    int binned_edge_count = 0;
    for (const QJsonValue &value : bins)
    {
        binned_edge_count +=
            value.toObject().value(QStringLiteral("edge_count")).toInt();
    }
    EXPECT_EQ(binned_edge_count, 3);

    const QJsonArray sources = report.value(QStringLiteral("sources")).toArray();
    ASSERT_EQ(sources.size(), 16);
    EXPECT_EQ(sources[0].toObject()
                  .value(QStringLiteral("participating_edge_count"))
                  .toInt(),
              2);
    EXPECT_EQ(sources[1].toObject()
                  .value(QStringLiteral("participating_edge_count"))
                  .toInt(),
              1);
    EXPECT_EQ(sources[0].toObject()
                  .value(QStringLiteral("source_slot"))
                  .toInt(),
              0);
    EXPECT_EQ(sources[0].toObject()
                  .value(QStringLiteral("source_index"))
                  .toInt(),
              100);
    EXPECT_TRUE(report.value(
        QStringLiteral("source_mask_source_mapping_complete")).toBool());
    EXPECT_EQ(report.value(QStringLiteral("highest_risk_sectors"))
                  .toArray()
                  .size(),
              3);
    EXPECT_EQ(report.value(QStringLiteral("rematch_target_count")).toInt(), 1);
    const QJsonArray targets =
        report.value(QStringLiteral("rematch_targets")).toArray();
    ASSERT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].toObject()
                  .value(QStringLiteral("root_cause"))
                  .toString(),
              QStringLiteral("depth_spread"));
    const QString encoded_source_mask = targets[0].toObject()
        .value(QStringLiteral("source_mask"))
        .toString();
    EXPECT_EQ(encoded_source_mask.size(), 66);
    EXPECT_EQ(encoded_source_mask,
              QStringLiteral("0x") + QString(63, QLatin1Char('0')) +
                  QStringLiteral("3"));
}

TEST(MeshAcquisitionGapReportTest, DeclaresIncompleteSourceMappingExplicitly)
{
    QStringList labels;
    std::vector<int> source_indices;
    for (int index = 0; index < 16; ++index)
    {
        labels.append(QStringLiteral("frame_%1").arg(index));
        source_indices.push_back(index);
    }
    const QJsonObject report = xjw::mesh::buildMeshAcquisitionGapReport(
        {}, makeLayout(), labels, source_indices, 20, false);

    EXPECT_FALSE(report.value(
        QStringLiteral("source_mask_source_mapping_complete")).toBool());
    EXPECT_EQ(report.value(QStringLiteral("input_frame_count")).toInt(), 20);
    EXPECT_EQ(report.value(QStringLiteral("sources")).toArray().size(), 16);
}
