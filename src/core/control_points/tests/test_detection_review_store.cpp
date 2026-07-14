#include "detection/DetectionReviewStore.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QTemporaryDir>

namespace xjw::control_points
{
namespace
{

DetectionReviewEntry reviewEntry()
{
    DetectionReviewEntry entry;
    entry.id = QStringLiteral("review-1");
    entry.reason = QStringLiteral("unassociated_non_coded");
    entry.message = QStringLiteral("非编码标靶需要人工确认身份");
    entry.observation.imageId = QStringLiteral("image-a");
    entry.observation.imagePathSnapshot = QStringLiteral("E:/images/a.png");
    entry.observation.imageContentSignature = QStringLiteral("sha256:a");
    entry.observation.detection.family = MarkerTargetFamily::NonCodedCircle;
    entry.observation.detection.targetId = -1;
    entry.observation.detection.center = QPointF(123.25, 456.75);
    entry.observation.detection.corners = QPolygonF{
        QPointF(120.0, 450.0),
        QPointF(130.0, 450.0),
        QPointF(130.0, 460.0),
        QPointF(120.0, 460.0),
    };
    entry.observation.detection.confidence = 0.87;
    entry.observation.detection.centerSigmaPx = 0.18;
    entry.observation.detection.decisionMargin = 31.5;
    entry.observation.detection.hamming = 1;
    entry.observation.detection.sizePx = 24.0;
    entry.observation.detection.rotationDegrees = 90.0;
    entry.observation.detection.source = QStringLiteral("noncoded:circle");
    return entry;
}

TEST(DetectionReviewStoreTest, RoundTripsPendingObservationAndConflictMetadata)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("detection_review.json"));

    DetectionReviewQueue queue;
    queue.sourceRevision = 17;
    queue.entries.push_back(reviewEntry());

    const DetectionReviewIoResult saved = DetectionReviewStore(path).save(queue);
    ASSERT_TRUE(saved.ok) << saved.error.toStdString();

    const DetectionReviewIoResult loaded = DetectionReviewStore(path).load();
    ASSERT_TRUE(loaded.ok) << loaded.error.toStdString();
    EXPECT_EQ(loaded.queue.schemaVersion, 1);
    EXPECT_EQ(loaded.queue.sourceRevision, 17u);
    ASSERT_EQ(loaded.queue.entries.size(), 1);
    const DetectionReviewEntry &entry = loaded.queue.entries.front();
    EXPECT_EQ(entry.id, QStringLiteral("review-1"));
    EXPECT_EQ(entry.reason, QStringLiteral("unassociated_non_coded"));
    EXPECT_EQ(entry.observation.imageId, QStringLiteral("image-a"));
    EXPECT_EQ(entry.observation.detection.family, MarkerTargetFamily::NonCodedCircle);
    EXPECT_EQ(entry.observation.detection.center, QPointF(123.25, 456.75));
    ASSERT_EQ(entry.observation.detection.corners.size(), 4);
    EXPECT_DOUBLE_EQ(entry.observation.detection.confidence, 0.87);
    EXPECT_EQ(entry.observation.detection.source, QStringLiteral("noncoded:circle"));
}

TEST(DetectionReviewStoreTest, RejectsDamagedJsonWithoutReplacingIt)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("detection_review.json"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_GT(file.write("{ damaged"), 0);
    file.close();

    const DetectionReviewIoResult loaded = DetectionReviewStore(path).load();
    EXPECT_FALSE(loaded.ok);
    EXPECT_TRUE(loaded.error.contains(QStringLiteral("JSON")));

    QFile unchanged(path);
    ASSERT_TRUE(unchanged.open(QIODevice::ReadOnly));
    EXPECT_EQ(unchanged.readAll(), QByteArray("{ damaged"));
}

} // namespace
} // namespace xjw::control_points
