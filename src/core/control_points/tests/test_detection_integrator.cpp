#include "detection/DetectionIntegrator.h"

#include <gtest/gtest.h>

#include <QLineF>

#include <algorithm>

namespace xjw::control_points
{
namespace
{

MarkerDetectionObservation observation(const QString &imageId,
                                       const QPointF &center,
                                       int targetId = 7)
{
    MarkerDetectionObservation value;
    value.imageId = imageId;
    value.imagePathSnapshot = QStringLiteral("E:/images/%1.png").arg(imageId);
    value.imageContentSignature = QStringLiteral("signature-%1").arg(imageId);
    value.detection.family = MarkerTargetFamily::AprilTag36h11;
    value.detection.targetId = targetId;
    value.detection.center = center;
    value.detection.confidence = 0.95;
    value.detection.centerSigmaPx = 0.2;
    value.detection.source = QStringLiteral("apriltag:tag36h11");
    return value;
}

TEST(DetectionIntegratorTest, MergesCodedIdentityAcrossImagesWithoutOverwritingManualProjection)
{
    MarkerSet set;
    const MarkerId marker_id = set.addMarker(QStringLiteral("target 7"), MarkerRole::TieMarker);
    set.setTargetIdentity(marker_id, {QStringLiteral("tag36h11"), 7, 0.0, QStringLiteral("apriltag")});

    MarkerProjection manual;
    manual.imageId = QStringLiteral("image-a");
    manual.xy = QPointF(10.0, 10.0);
    manual.state = ProjectionState::ManualPinned;
    manual.source = QStringLiteral("manual");
    set.upsertProjection(marker_id, manual);

    const QVector<MarkerDetectionObservation> detections{
        observation(QStringLiteral("image-a"), QPointF(30.0, 30.0)),
        observation(QStringLiteral("image-b"), QPointF(40.25, 50.75)),
    };
    const QHash<QString, QString> signatures{
        {QStringLiteral("image-a"), QStringLiteral("signature-image-a")},
        {QStringLiteral("image-b"), QStringLiteral("signature-image-b")},
    };

    const DetectionIntegrationResult result = DetectionIntegrator::integrate(set, detections, signatures);
    ASSERT_EQ(result.markerSet.markers().size(), 1);
    const Marker &marker = result.markerSet.marker(marker_id);
    EXPECT_EQ(marker.projection(QStringLiteral("image-a")).state, ProjectionState::ManualPinned);
    EXPECT_EQ(marker.projection(QStringLiteral("image-a")).xy, QPointF(10.0, 10.0));
    EXPECT_EQ(marker.projection(QStringLiteral("image-b")).state, ProjectionState::AutoDetected);
    EXPECT_EQ(marker.projection(QStringLiteral("image-b")).xy, QPointF(40.25, 50.75));
    ASSERT_EQ(result.conflicts.size(), 1);
    EXPECT_EQ(result.conflicts.front().reason, QStringLiteral("manual_projection_mismatch"));
}

TEST(DetectionIntegratorTest, RoutesDuplicateStaleAndNonCodedCandidatesToReview)
{
    MarkerSet set;
    auto first = observation(QStringLiteral("image-a"), QPointF(20.0, 20.0));
    auto duplicate = observation(QStringLiteral("image-a"), QPointF(80.0, 80.0));
    duplicate.detection.confidence = 0.6;
    auto stale = observation(QStringLiteral("image-b"), QPointF(30.0, 30.0), 8);
    stale.imageContentSignature = QStringLiteral("old-signature");
    auto non_coded = observation(QStringLiteral("image-c"), QPointF(50.0, 50.0), -1);
    non_coded.detection.family = MarkerTargetFamily::NonCodedCircle;

    const DetectionIntegrationResult result = DetectionIntegrator::integrate(
        set,
        {first, duplicate, stale, non_coded},
        {{QStringLiteral("image-a"), QStringLiteral("signature-image-a")},
         {QStringLiteral("image-b"), QStringLiteral("signature-image-b")},
         {QStringLiteral("image-c"), QStringLiteral("signature-image-c")}});

    ASSERT_EQ(result.markerSet.markers().size(), 1);
    EXPECT_EQ(result.pendingReview.size(), 3);
    EXPECT_TRUE(std::any_of(result.conflicts.cbegin(), result.conflicts.cend(), [](const DetectionConflict &item)
    {
        return item.reason == QLatin1String("duplicate_coded_id");
    }));
    EXPECT_TRUE(std::any_of(result.conflicts.cbegin(), result.conflicts.cend(), [](const DetectionConflict &item)
    {
        return item.reason == QLatin1String("stale_image");
    }));
}

} // namespace
} // namespace xjw::control_points
