#include "model/MarkerSet.h"
#include "model/MarkerSetValidator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

using xjw::control_points::MarkerId;
using xjw::control_points::MarkerModelError;
using xjw::control_points::MarkerProjection;
using xjw::control_points::MarkerRole;
using xjw::control_points::MarkerSet;
using xjw::control_points::MarkerSetValidator;
using xjw::control_points::ProjectionState;
using xjw::control_points::markerRoleUsesReferenceConstraint;

TEST(MarkerSetTest, RejectsDuplicateLabelsAndProjectionPerImage)
{
    MarkerSet set;
    const MarkerId first = set.addMarker(QStringLiteral("GCP001"), MarkerRole::ControlPoint);
    EXPECT_THROW(set.addMarker(QStringLiteral("GCP001"), MarkerRole::TieMarker), MarkerModelError);

    MarkerProjection projection;
    projection.imageId = QStringLiteral("image-uuid-1");
    projection.xy = QPointF(120.25, 330.75);
    projection.state = ProjectionState::ManualPinned;
    set.upsertProjection(first, projection);
    set.upsertProjection(first, projection);

    ASSERT_EQ(set.marker(first).projections.size(), 1u);
}

TEST(MarkerSetTest, KeepsControlCheckAndTieRolesDistinct)
{
    EXPECT_TRUE(markerRoleUsesReferenceConstraint(MarkerRole::ControlPoint));
    EXPECT_FALSE(markerRoleUsesReferenceConstraint(MarkerRole::CheckPoint));
    EXPECT_FALSE(markerRoleUsesReferenceConstraint(MarkerRole::TieMarker));
}

TEST(MarkerSetTest, RejectsInvalidProjectionCoordinates)
{
    MarkerSet set;
    const MarkerId marker_id = set.addMarker(QStringLiteral("point 1"), MarkerRole::TieMarker);

    MarkerProjection projection;
    projection.imageId = QStringLiteral("image-uuid-1");
    projection.xy = QPointF(std::numeric_limits<double>::quiet_NaN(), 10.0);

    EXPECT_THROW(set.upsertProjection(marker_id, projection), MarkerModelError);
}

TEST(MarkerSetTest, ValidatorAcceptsAConsistentSet)
{
    MarkerSet set;
    const MarkerId marker_id = set.addMarker(QStringLiteral("CP001"), MarkerRole::ControlPoint);

    MarkerProjection projection;
    projection.imageId = QStringLiteral("image-uuid-1");
    projection.xy = QPointF(12.5, 42.0);
    projection.state = ProjectionState::ManualPinned;
    set.upsertProjection(marker_id, projection);

    const auto issues = MarkerSetValidator::validate(set);
    EXPECT_TRUE(issues.empty());
}

TEST(MarkerSetTest, ValidatorRejectsDuplicateCodedTargetIdentity)
{
    MarkerSet set;
    const MarkerId first = set.addMarker(QStringLiteral("target 7"), MarkerRole::TieMarker);
    const MarkerId second = set.addMarker(QStringLiteral("target 7 duplicate"), MarkerRole::TieMarker);
    const xjw::control_points::TargetIdentity identity{
        QStringLiteral("tag36h11"), 7, 0.0, QStringLiteral("apriltag")};
    set.setTargetIdentity(first, identity);
    set.setTargetIdentity(second, identity);

    const auto issues = MarkerSetValidator::validate(set);
    EXPECT_TRUE(std::any_of(issues.cbegin(), issues.cend(), [](const auto &issue)
    {
        return issue.code == QLatin1String("duplicate_target_identity");
    }));
}
