#include "commands/MarkerChangeSet.h"

#include <gtest/gtest.h>

using xjw::control_points::MarkerChangeSet;
using xjw::control_points::MarkerProjection;
using xjw::control_points::MarkerRole;
using xjw::control_points::MarkerSet;
using xjw::control_points::ProjectionState;

TEST(MarkerChangeSetTest, RevertsProjectionReplacementExactly)
{
    MarkerSet set;
    const auto marker_id = set.addMarker(QStringLiteral("point 1"), MarkerRole::TieMarker);
    MarkerProjection original;
    original.imageId = QStringLiteral("image-uuid-1");
    original.xy = QPointF(10.0, 20.0);
    original.state = ProjectionState::ManualPinned;
    set.upsertProjection(marker_id, original);

    MarkerProjection replacement = original;
    replacement.xy = QPointF(30.0, 40.0);
    const MarkerChangeSet change = MarkerChangeSet::replaceProjection(
        set, marker_id, original.imageId, replacement);

    change.apply(&set);
    EXPECT_EQ(set.marker(marker_id).projection(original.imageId).xy, QPointF(30.0, 40.0));
    change.revert(&set);
    EXPECT_EQ(set.marker(marker_id).projection(original.imageId).xy, QPointF(10.0, 20.0));
}

TEST(MarkerChangeSetTest, ReportsAffectedMarkerAndDescription)
{
    MarkerSet set;
    const auto marker_id = set.addMarker(QStringLiteral("point 1"), MarkerRole::TieMarker);
    MarkerProjection projection;
    projection.imageId = QStringLiteral("image-uuid-1");
    projection.xy = QPointF(10.0, 20.0);

    const MarkerChangeSet change = MarkerChangeSet::replaceProjection(
        set, marker_id, projection.imageId, projection);

    EXPECT_EQ(change.affectedMarkerIds(), QVector<QString>{marker_id});
    EXPECT_FALSE(change.description().isEmpty());
}

TEST(MarkerChangeSetTest, CreatesMarkerAndProjectionAsOneReversibleEdit)
{
    MarkerSet set;
    MarkerProjection projection;
    projection.imageId = QStringLiteral("image-uuid-1");
    projection.xy = QPointF(12.0, 34.0);
    projection.state = ProjectionState::ManualPinned;

    const MarkerChangeSet change = MarkerChangeSet::createMarkerWithProjection(
        set, QStringLiteral("point 1"), MarkerRole::TieMarker, projection);
    ASSERT_EQ(change.affectedMarkerIds().size(), 1);
    const QString marker_id = change.affectedMarkerIds().front();

    change.apply(&set);
    EXPECT_EQ(set.marker(marker_id).projection(projection.imageId).xy, projection.xy);
    change.revert(&set);
    EXPECT_TRUE(set.markers().isEmpty());
}

TEST(MarkerChangeSetTest, RemovesAndBlocksProjectionReversibly)
{
    MarkerSet set;
    const QString marker_id = set.addMarker(QStringLiteral("point 1"), MarkerRole::TieMarker);
    MarkerProjection projection;
    projection.imageId = QStringLiteral("image-uuid-1");
    projection.xy = QPointF(12.0, 34.0);
    projection.state = ProjectionState::ManualPinned;
    projection.source = QStringLiteral("manual");
    set.upsertProjection(marker_id, projection);

    const MarkerChangeSet block = MarkerChangeSet::setProjectionState(
        set, marker_id, projection.imageId, ProjectionState::Blocked, QStringLiteral("屏蔽标记投影"));
    block.apply(&set);
    EXPECT_EQ(set.marker(marker_id).projection(projection.imageId).state, ProjectionState::Blocked);
    block.revert(&set);
    EXPECT_EQ(set.marker(marker_id).projection(projection.imageId).state, ProjectionState::ManualPinned);

    const MarkerChangeSet remove = MarkerChangeSet::removeProjection(set, marker_id, projection.imageId);
    remove.apply(&set);
    EXPECT_TRUE(set.marker(marker_id).projections.isEmpty());
    remove.revert(&set);
    EXPECT_EQ(set.marker(marker_id).projection(projection.imageId), projection);
}

TEST(MarkerChangeSetTest, UpdatesMarkerPropertiesAndReferenceAsOneEdit)
{
    MarkerSet set;
    const QString marker_id = set.addMarker(QStringLiteral("point 1"), MarkerRole::TieMarker);
    xjw::control_points::ReferenceCoordinate reference;
    reference.x = 1.0;
    reference.y = 2.0;
    reference.z = 3.0;
    reference.sigmaX = 0.02;
    reference.sigmaY = 0.02;
    reference.sigmaZ = 0.05;
    reference.sourceCrs = QStringLiteral("EPSG:4978");

    const MarkerChangeSet change = MarkerChangeSet::updateMarker(
        set,
        marker_id,
        QStringLiteral("GCP 1"),
        MarkerRole::ControlPoint,
        true,
        reference);
    change.apply(&set);
    EXPECT_EQ(set.marker(marker_id).label, QStringLiteral("GCP 1"));
    EXPECT_EQ(set.marker(marker_id).role, MarkerRole::ControlPoint);
    ASSERT_TRUE(set.marker(marker_id).referenceCoordinate.has_value());
    EXPECT_EQ(set.marker(marker_id).referenceCoordinate.value(), reference);

    change.revert(&set);
    EXPECT_EQ(set.marker(marker_id).label, QStringLiteral("point 1"));
    EXPECT_EQ(set.marker(marker_id).role, MarkerRole::TieMarker);
    EXPECT_FALSE(set.marker(marker_id).referenceCoordinate.has_value());
}
