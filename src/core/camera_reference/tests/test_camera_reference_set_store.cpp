#include "io/CameraReferenceSetJson.h"
#include "io/CameraReferenceSetStore.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>

using xjw::camera_reference::CameraReferenceLeverArm;
using xjw::camera_reference::CameraReferenceRecord;
using xjw::camera_reference::CameraReferenceSet;
using xjw::camera_reference::CameraReferenceSetJson;
using xjw::camera_reference::CameraReferenceSetStore;
using xjw::camera_reference::CameraReferenceSolverFrame;
using xjw::camera_reference::CameraReferenceSource;
using xjw::camera_reference::UnmatchedCameraReferenceRecord;

namespace
{

CameraReferenceSet makeReferenceSet()
{
    CameraReferenceSet referenceSet;
    referenceSet.setImageSetFingerprint(QStringLiteral("images-sha256"));

    CameraReferenceSource source;
    source.kind = QStringLiteral("metashape_camera_reference");
    source.displayName = QStringLiteral("Cameras_WGS84.txt");
    source.contentSha256 = QString(64, QLatin1Char('a'));
    source.sourceCrs = QStringLiteral("EPSG:4979");
    source.axisOrder = QStringLiteral("longitude_latitude");
    source.verticalDatum = QStringLiteral("ellipsoidal");
    source.verticalUnit = QStringLiteral("m");
    source.orientationConvention = QStringLiteral("metashape_yaw_pitch_roll");
    source.angleUnit = QStringLiteral("deg");
    referenceSet.replaceSource(source);

    CameraReferenceSolverFrame frame;
    frame.frameId = QStringLiteral("local-enu-1");
    frame.kind = QStringLiteral("local_enu");
    frame.unit = QStringLiteral("m");
    frame.originEcefMeters = {{1000.0, 2000.0, 3000.0}};
    frame.targetCrs = QStringLiteral("EPSG:4978");
    frame.targetCrsWkt = QStringLiteral("GEODCRS[\"WGS 84\"]");
    frame.normalizationHash = QStringLiteral("normalization-sha256");
    referenceSet.replaceSolverFrame(frame);

    CameraReferenceLeverArm leverArm;
    leverArm.id = QStringLiteral("sensor:0");
    leverArm.vectorMeters = {{0.3682, -0.1815, -0.032}};
    leverArm.vectorFrame = QStringLiteral("sensor_body");
    leverArm.vectorDirection = QStringLiteral("antenna_to_camera");
    leverArm.source = QStringLiteral("metashape_sensor_antenna");
    referenceSet.addLeverArm(leverArm);

    CameraReferenceRecord record;
    record.imageUuid = QStringLiteral("image-uuid-1");
    record.imagePathSnapshot = QStringLiteral("E:/images/IMG_0001.JPG");
    record.sourceLabel = QStringLiteral("IMG_0001.JPG");
    record.sensorKey = QStringLiteral("0");
    record.leverArmId = leverArm.id;
    record.raw.position = {{36.0, 38.0, 225.0}};
    record.raw.positionSigma = {{0.02, 0.02, 0.05}};
    record.raw.positionSigmaFrame = QStringLiteral("local_enu");
    record.raw.positionSigmaUnit = QStringLiteral("m");
    record.raw.horizontalSigmaMeters = 0.0283;
    record.raw.orientationYprDegrees = {{90.0, 1.0, -2.0}};
    record.raw.orientationSigmaDegrees = {{5.0, 5.0, 5.0}};
    record.raw.timestamp = QStringLiteral("2024-01-02T03:04:05Z");
    record.resolved.cameraCenterMeters = {{1.0, 2.0, 3.0}};
    record.resolved.rotationCameraToWorld = {{1.0, 0.0, 0.0,
                                               0.0, 1.0, 0.0,
                                               0.0, 0.0, 1.0}};
    record.resolved.positionSigmaMeters = {{0.02, 0.02, 0.05}};
    record.resolved.rotationSigmaDegrees = {{5.0, 5.0, 5.0}};
    record.resolved.positionUsable = true;
    record.resolved.orientationUsable = true;
    record.resolved.leverArmApplied = true;
    record.resolved.frameId = frame.frameId;
    record.resolved.normalizationHash = frame.normalizationHash;
    referenceSet.addRecord(record);

    UnmatchedCameraReferenceRecord unmatched;
    unmatched.sourceLabel = QStringLiteral("IMG_0475.JPG");
    unmatched.leverArmId = leverArm.id;
    unmatched.raw = record.raw;
    unmatched.raw.position = {{36.1, 38.1, 226.0}};
    unmatched.reason = QStringLiteral("project_image_not_found");
    referenceSet.addUnmatchedRecord(unmatched);
    return referenceSet;
}

bool writeBytes(const QString &path, const QByteArray &bytes)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly)
        && file.write(bytes) == bytes.size();
}

} // namespace

TEST(CameraReferenceSetStoreTest, MissingFileLoadsAsEmptySet)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    const auto result = CameraReferenceSetStore(
        directory.filePath(QStringLiteral("missing.json"))).load();
    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    EXPECT_TRUE(result.referenceSet.records().isEmpty());
    EXPECT_TRUE(result.referenceSet.leverArms().isEmpty());
    EXPECT_TRUE(result.referenceSet.unmatchedRecords().isEmpty());
}

TEST(CameraReferenceSetStoreTest, SavesAtomicallyAndRoundTripsAllFields)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    const CameraReferenceSet expected = makeReferenceSet();
    CameraReferenceSetStore store(
        directory.filePath(QStringLiteral("camera_reference_set.json")));
    const auto saved = store.save(expected);
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    const auto loaded = store.load();
    ASSERT_TRUE(loaded.ok) << qPrintable(loaded.error);
    EXPECT_EQ(loaded.referenceSet, expected);
    EXPECT_EQ(loaded.referenceSet.source().sourceCrs, QStringLiteral("EPSG:4979"));
    EXPECT_EQ(loaded.referenceSet.solverFrame().frameId, QStringLiteral("local-enu-1"));
    ASSERT_TRUE(loaded.referenceSet.records().front().raw.horizontalSigmaMeters);
    EXPECT_DOUBLE_EQ(*loaded.referenceSet.records().front().raw.horizontalSigmaMeters, 0.0283);
    EXPECT_EQ(loaded.referenceSet.records().front().raw.positionSigmaFrame,
              QStringLiteral("local_enu"));
    EXPECT_EQ(loaded.referenceSet.records().front().raw.positionSigmaUnit,
              QStringLiteral("m"));
    ASSERT_EQ(loaded.referenceSet.unmatchedRecords().size(), 1);
    ASSERT_TRUE(loaded.referenceSet.unmatchedRecords().front().raw.position);
    EXPECT_DOUBLE_EQ((*loaded.referenceSet.unmatchedRecords().front().raw.position)[2], 226.0);
    EXPECT_EQ(loaded.referenceSet.unmatchedRecords().front().leverArmId,
              QStringLiteral("sensor:0"));
}

TEST(CameraReferenceSetStoreTest, RejectsHigherSchemaVersion)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("camera_reference_set.json"));

    QJsonObject object = CameraReferenceSetJson::encode(makeReferenceSet());
    object[QStringLiteral("schema_version")] = CameraReferenceSet::CurrentSchemaVersion + 1;
    ASSERT_TRUE(writeBytes(path, QJsonDocument(object).toJson(QJsonDocument::Compact)));

    const auto loaded = CameraReferenceSetStore(path).load();
    EXPECT_FALSE(loaded.ok);
    EXPECT_TRUE(loaded.error.contains(QStringLiteral("schema_version")));
}

TEST(CameraReferenceSetStoreTest, RejectsSchemaVersionOutsideIntRange)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("camera_reference_set.json"));

    QJsonObject object = CameraReferenceSetJson::encode(makeReferenceSet());
    object[QStringLiteral("schema_version")] = 1.0e100;
    ASSERT_TRUE(writeBytes(path, QJsonDocument(object).toJson(QJsonDocument::Compact)));

    const auto loaded = CameraReferenceSetStore(path).load();
    EXPECT_FALSE(loaded.ok);
    EXPECT_TRUE(loaded.error.contains(QStringLiteral("整数范围")));
}

TEST(CameraReferenceSetStoreTest, RejectsInvalidRotationsAndWhitespaceIds)
{
    CameraReferenceSet referenceSet = makeReferenceSet();
    CameraReferenceSolverFrame invalidFrame = referenceSet.solverFrame();
    invalidFrame.rotationSolverToEcef = {{-1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0}};
    EXPECT_THROW(referenceSet.replaceSolverFrame(invalidFrame),
                 xjw::camera_reference::CameraReferenceModelError);

    CameraReferenceLeverArm invalidLeverArm = referenceSet.leverArms().front();
    invalidLeverArm.id = QStringLiteral(" sensor:1 ");
    EXPECT_THROW(referenceSet.addLeverArm(invalidLeverArm),
                 xjw::camera_reference::CameraReferenceModelError);
}

TEST(CameraReferenceSetStoreTest, CorruptJsonRemainsUntouched)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("camera_reference_set.json"));
    const QByteArray corrupt = QByteArrayLiteral("{not-json");
    ASSERT_TRUE(writeBytes(path, corrupt));

    const auto loaded = CameraReferenceSetStore(path).load();
    EXPECT_FALSE(loaded.ok);
    EXPECT_FALSE(loaded.error.isEmpty());

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(file.readAll(), corrupt);
}
