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
    std::uint16_t source_mask)
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
    for (int index = 0; index < 16; ++index)
    {
        labels.append(QStringLiteral("frame_%1").arg(index));
    }

    const QJsonObject report = xjw::mesh::buildMeshAcquisitionGapReport(
        edges, makeLayout(), labels, 16);

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

    const QJsonArray frames = report.value(QStringLiteral("frames")).toArray();
    ASSERT_EQ(frames.size(), 16);
    EXPECT_EQ(frames[0].toObject()
                  .value(QStringLiteral("participating_edge_count"))
                  .toInt(),
              2);
    EXPECT_EQ(frames[1].toObject()
                  .value(QStringLiteral("participating_edge_count"))
                  .toInt(),
              1);
    EXPECT_TRUE(report.value(
        QStringLiteral("source_mask_frame_mapping_complete")).toBool());
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
}

TEST(MeshAcquisitionGapReportTest, DeclaresSourceMaskLimitBeyondSixteenFrames)
{
    QStringList labels;
    for (int index = 0; index < 16; ++index)
    {
        labels.append(QStringLiteral("frame_%1").arg(index));
    }
    const QJsonObject report = xjw::mesh::buildMeshAcquisitionGapReport(
        {}, makeLayout(), labels, 20);

    EXPECT_FALSE(report.value(
        QStringLiteral("source_mask_frame_mapping_complete")).toBool());
    EXPECT_EQ(report.value(QStringLiteral("frames")).toArray().size(), 16);
}
