#include "io/MarkerSetStore.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QTemporaryDir>

using xjw::control_points::MarkerProjection;
using xjw::control_points::MarkerRole;
using xjw::control_points::MarkerSet;
using xjw::control_points::MarkerSetStore;
using xjw::control_points::ProjectionState;
using xjw::control_points::TargetIdentity;

namespace
{

MarkerSet makeMarkerSetWithEveryProjectionState()
{
    MarkerSet set;
    const auto marker_id = set.addMarker(QStringLiteral("point 1"), MarkerRole::TieMarker);
    TargetIdentity identity;
    identity.family = QStringLiteral("tag36h11");
    identity.encodedId = 7;
    identity.rotationDegrees = 90.0;
    identity.generationSource = QStringLiteral("apriltag");
    set.setTargetIdentity(marker_id, identity);
    const QVector<ProjectionState> states{
        ProjectionState::ManualPinned,
        ProjectionState::AutoDetected,
        ProjectionState::Predicted,
        ProjectionState::Blocked,
        ProjectionState::Disabled
    };

    for (int index = 0; index < states.size(); ++index)
    {
        MarkerProjection projection;
        projection.imageId = QStringLiteral("image-%1").arg(index);
        projection.imagePathSnapshot = QStringLiteral("E:/images/%1.png").arg(index);
        projection.xy = QPointF(100.25 + index, 200.75 + index);
        projection.state = states[index];
        projection.sigmaPx = 0.5 + index;
        projection.confidence = 0.9 - 0.1 * index;
        projection.residualPx = 0.2 * index;
        projection.source = QStringLiteral("test");
        projection.imageContentSignature = QStringLiteral("signature-%1").arg(index);
        set.upsertProjection(marker_id, projection);
    }
    return set;
}

} // namespace

TEST(MarkerSetStoreTest, SavesAtomicallyAndRoundTripsAllProjectionStates)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const MarkerSet expected = makeMarkerSetWithEveryProjectionState();
    MarkerSetStore store(dir.filePath(QStringLiteral("marker_set.json")));
    const auto saved = store.save(expected);
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    const auto loaded = store.load();
    ASSERT_TRUE(loaded.ok) << qPrintable(loaded.error);
    EXPECT_EQ(loaded.markerSet, expected);
}

TEST(MarkerSetStoreTest, RefusesCorruptJsonWithoutOverwritingTheSource)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString path = dir.filePath(QStringLiteral("marker_set.json"));

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    const QByteArray corrupt = QByteArrayLiteral("{not-json");
    ASSERT_EQ(file.write(corrupt), corrupt.size());
    file.close();

    MarkerSetStore store(path);
    const auto loaded = store.load();
    EXPECT_FALSE(loaded.ok);
    EXPECT_FALSE(loaded.error.isEmpty());

    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(file.readAll(), corrupt);
}
