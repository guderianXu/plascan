#include "reference/CameraReferenceCsvExporter.h"
#include "reference/CameraReferenceProjectIdentity.h"
#include "reference/CameraReferenceTreeModel.h"
#include "reference/MetashapeCameraReferenceSetBuilder.h"
#include "reference/ProjectCameraReferenceRepository.h"
#include "project/services/MetashapeCameraReferenceImporter.h"
#include "widgets/ReferenceMarkerModels.h"

#include "ProjectData.h"
#include "io/CameraReferenceSetStore.h"
#include "io/MarkerSetJson.h"
#include "project/ProjectIO.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QModelIndex>
#include <QTemporaryDir>

#include <cmath>

namespace camera_reference = xjw::camera_reference;
namespace control_points = xjw::control_points;
namespace reference = xjw::gui::reference;

namespace
{

camera_reference::CameraReferenceSet makeCameraReferenceSet()
{
    camera_reference::CameraReferenceSet reference_set;
    camera_reference::CameraReferenceSource source;
    source.kind = QStringLiteral("agisoft_camera_reference");
    source.displayName = QStringLiteral("Cameras_WGS84.txt");
    source.sourceCrs = QStringLiteral("EPSG:4979");
    source.axisOrder = QStringLiteral("longitude_latitude");
    source.verticalDatum = QStringLiteral("ellipsoidal");
    source.verticalUnit = QStringLiteral("m");
    source.orientationConvention = QStringLiteral("metashape_ypr_unresolved");
    source.angleUnit = QStringLiteral("deg");
    reference_set.replaceSource(source);

    camera_reference::CameraReferenceRecord matched;
    matched.imageUuid = QStringLiteral("image-uuid-1");
    matched.imagePathSnapshot = QStringLiteral("source/IMG_0001.JPG");
    matched.sourceLabel = QStringLiteral("IMG_0001.JPG");
    matched.raw.position = camera_reference::Vector3d{{31.4656473, 59.84424741, 54.454}};
    matched.raw.positionSigma = camera_reference::Vector3d{{0.2, 0.3, 0.4}};
    matched.raw.positionSigmaFrame = QStringLiteral("local_enu");
    matched.raw.positionSigmaUnit = QStringLiteral("m");
    matched.resolved.error = QStringLiteral("solver frame 尚未建立");
    reference_set.addRecord(matched);

    camera_reference::UnmatchedCameraReferenceRecord unmatched;
    unmatched.sourceLabel = QStringLiteral("IMG_MISSING.JPG");
    unmatched.raw.position = camera_reference::Vector3d{{31.5, 59.9, 55.0}};
    unmatched.reason = QStringLiteral("项目中未找到同名影像");
    reference_set.addUnmatchedRecord(unmatched);
    return reference_set;
}

QJsonObject makeProjectMetadata()
{
    const QJsonObject camera{
        {QStringLiteral("C"), QJsonArray{101.0, 202.0, 303.0}},
        {QStringLiteral("R"), QJsonArray{1.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0,
                                         0.0, 0.0, 1.0}}
    };
    const QJsonObject image{
        {QStringLiteral("image_uuid"), QStringLiteral("image-uuid-1")},
        {QStringLiteral("path"), QStringLiteral("project/images/IMG_0001.JPG")},
        {QStringLiteral("camera"), camera}
    };
    return QJsonObject{
        {QStringLiteral("project_files"),
         QJsonObject{{QStringLiteral("images"), QJsonArray{image}}}}
    };
}

void addProjection(control_points::MarkerSet *marker_set,
                   const control_points::MarkerId &marker_id,
                   const QString &image_id,
                   double residual)
{
    control_points::MarkerProjection projection;
    projection.imageId = image_id;
    projection.imagePathSnapshot = image_id + QStringLiteral(".JPG");
    projection.xy = QPointF(100.0, 200.0);
    projection.state = control_points::ProjectionState::ManualPinned;
    projection.residualPx = residual;
    marker_set->upsertProjection(marker_id, projection);
}

void setReferenceCoordinate(control_points::MarkerSet *marker_set,
                            const control_points::MarkerId &marker_id,
                            double x,
                            double y,
                            double z)
{
    control_points::ReferenceCoordinate coordinate;
    coordinate.x = x;
    coordinate.y = y;
    coordinate.z = z;
    coordinate.sigmaX = 0.1;
    coordinate.sigmaY = 0.2;
    coordinate.sigmaZ = 0.3;
    coordinate.sourceCrs = QStringLiteral("EPSG:4978");
    coordinate.axisOrder = QStringLiteral("traditional_gis");
    coordinate.verticalDatum = QStringLiteral("ellipsoidal");
    coordinate.verticalUnit = QStringLiteral("m");
    marker_set->setReferenceCoordinate(marker_id, coordinate);
}

struct MarkerFixture
{
    control_points::MarkerSet markerSet;
    control_points::MarkerId controlId;
    control_points::MarkerId checkId;
    control_points::MarkerId tieId;
    control_points::ScaleBarId controlScaleId;
    control_points::ScaleBarId checkScaleId;
};

MarkerFixture makeMarkerFixture()
{
    MarkerFixture fixture;
    fixture.controlId = fixture.markerSet.addMarker(
        QStringLiteral("GCP-1"), control_points::MarkerRole::ControlPoint);
    fixture.checkId = fixture.markerSet.addMarker(
        QStringLiteral("CHK-1"), control_points::MarkerRole::CheckPoint);
    fixture.tieId = fixture.markerSet.addMarker(
        QStringLiteral("TIE-1"), control_points::MarkerRole::TieMarker);

    setReferenceCoordinate(&fixture.markerSet, fixture.controlId, 10.0, 20.0, 30.0);
    setReferenceCoordinate(&fixture.markerSet, fixture.checkId, 40.0, 50.0, 60.0);
    addProjection(&fixture.markerSet, fixture.controlId, QStringLiteral("control-image"), 3.0);
    addProjection(&fixture.markerSet, fixture.checkId, QStringLiteral("check-image"), 4.0);
    addProjection(&fixture.markerSet, fixture.tieId, QStringLiteral("tie-image"), 100.0);

    fixture.controlScaleId = fixture.markerSet.addScaleBar(
        QStringLiteral("CONTROL-SCALE"),
        fixture.controlId,
        fixture.checkId,
        10.0,
        0.1,
        control_points::ScaleBarRole::Control);
    fixture.checkScaleId = fixture.markerSet.addScaleBar(
        QStringLiteral("CHECK-SCALE"),
        fixture.checkId,
        fixture.tieId,
        20.0,
        0.2,
        control_points::ScaleBarRole::Check);
    return fixture;
}

} // namespace

TEST(CameraReferenceTreeModelTest, ShowsSourceEstimatedErrorAndUnmatchedRoles)
{
    using Model = reference::CameraReferenceTreeModel;
    const camera_reference::CameraReferenceSet reference_set = makeCameraReferenceSet();
    const QJsonObject metadata = makeProjectMetadata();
    Model model;

    model.setReferenceData(reference_set, metadata, reference::ReferenceDisplayMode::Source);

    ASSERT_EQ(model.rowCount(), 3);
    EXPECT_EQ(model.headerData(Model::XColumn, Qt::Horizontal).toString(), QStringLiteral("经度 (°)"));
    EXPECT_EQ(model.headerData(Model::YColumn, Qt::Horizontal).toString(), QStringLiteral("纬度 (°)"));
    const QModelIndex source_camera = model.index(1, Model::LabelColumn);
    ASSERT_TRUE(source_camera.isValid());
    EXPECT_EQ(source_camera.data(Model::NodeTypeRole).toInt(),
              static_cast<int>(Model::NodeType::Camera));
    EXPECT_EQ(source_camera.data(Model::ImageUuidRole).toString(), QStringLiteral("image-uuid-1"));
    EXPECT_EQ(model.index(1, Model::XColumn).data().toDouble(), 31.4656473);
    EXPECT_EQ(model.index(1, Model::YColumn).data().toDouble(), 59.84424741);
    EXPECT_EQ(model.index(1, Model::ZColumn).data().toDouble(), 54.454);

    const QModelIndex unmatched_group = model.index(2, Model::LabelColumn);
    ASSERT_TRUE(unmatched_group.isValid());
    EXPECT_EQ(unmatched_group.data(Model::NodeTypeRole).toInt(),
              static_cast<int>(Model::NodeType::UnmatchedGroup));
    ASSERT_EQ(model.rowCount(unmatched_group), 1);
    const QModelIndex unmatched = model.index(0, Model::LabelColumn, unmatched_group);
    EXPECT_EQ(unmatched.data().toString(), QStringLiteral("IMG_MISSING.JPG"));
    EXPECT_EQ(unmatched.data(Model::NodeTypeRole).toInt(),
              static_cast<int>(Model::NodeType::UnmatchedRecord));
    EXPECT_EQ(model.index(0, Model::StatusColumn, unmatched_group).data().toString(),
              QStringLiteral("项目中未找到同名影像"));
    EXPECT_DOUBLE_EQ(model.index(0, Model::XColumn, unmatched_group).data().toDouble(), 31.5);
    EXPECT_DOUBLE_EQ(model.index(0, Model::YColumn, unmatched_group).data().toDouble(), 59.9);
    EXPECT_DOUBLE_EQ(model.index(0, Model::ZColumn, unmatched_group).data().toDouble(), 55.0);

    model.setReferenceData(reference_set, metadata, reference::ReferenceDisplayMode::Estimated);
    EXPECT_EQ(model.index(1, Model::XColumn).data().toDouble(), 101.0);
    EXPECT_EQ(model.index(1, Model::YColumn).data().toDouble(), 202.0);
    EXPECT_EQ(model.index(1, Model::ZColumn).data().toDouble(), 303.0);
    EXPECT_EQ(model.index(1, Model::StatusColumn).data().toString(), QStringLiteral("已解算"));

    model.setReferenceData(reference_set, metadata, reference::ReferenceDisplayMode::Error);
    EXPECT_EQ(model.index(1, Model::StatusColumn).data().toString(),
              QStringLiteral("待坐标/姿态转换"));
    EXPECT_FALSE(model.index(1, Model::XColumn).data().isValid());
}

TEST(CameraReferenceTreeModelTest, DisabledCameraDoesNotAffectTotalError)
{
    using Model = reference::CameraReferenceTreeModel;
    camera_reference::CameraReferenceSet reference_set;
    camera_reference::CameraReferenceSource source;
    source.kind = QStringLiteral("test");
    source.sourceCrs = QStringLiteral("EPSG:4978");
    source.axisOrder = QStringLiteral("traditional_gis");
    source.verticalUnit = QStringLiteral("m");
    reference_set.replaceSource(source);

    camera_reference::CameraReferenceSolverFrame frame;
    frame.frameId = QStringLiteral("solver-frame");
    frame.kind = QStringLiteral("local");
    frame.normalizationHash = QStringLiteral("normalization");
    reference_set.replaceSolverFrame(frame);

    camera_reference::CameraReferenceRecord record;
    record.imageUuid = QStringLiteral("image-uuid-1");
    record.imagePathSnapshot = QStringLiteral("IMG_0001.JPG");
    record.sourceLabel = QStringLiteral("IMG_0001.JPG");
    record.enabled = false;
    record.resolved.cameraCenterMeters = camera_reference::Vector3d{{1.0, 2.0, 3.0}};
    record.resolved.positionSigmaMeters = camera_reference::Vector3d{{0.1, 0.1, 0.1}};
    record.resolved.positionUsable = true;
    record.resolved.frameId = frame.frameId;
    record.resolved.normalizationHash = frame.normalizationHash;
    reference_set.addRecord(record);

    Model model;
    model.setReferenceData(reference_set,
                           makeProjectMetadata(),
                           reference::ReferenceDisplayMode::Error);

    EXPECT_DOUBLE_EQ(model.index(1, Model::XColumn).data().toDouble(), 100.0);
    EXPECT_FALSE(model.index(0, Model::XColumn).data().isValid());
}

TEST(CameraReferenceBuilderTest, PreservesMatchedAndUnmatchedRawObservations)
{
    xjw::gui::reference_import::MetashapeCameraReferenceImportResult imported;
    xjw::gui::reference_import::RawCameraReferenceRecord matched;
    matched.fileName = QStringLiteral("folder\\IMG_0001.JPG");
    matched.wgs84LatitudeDegrees = 59.0;
    matched.wgs84LongitudeDegrees = 31.0;
    matched.wgs84EllipsoidalHeightMeters = 54.0;
    matched.rollDegrees = 1.0;
    matched.pitchDegrees = 2.0;
    matched.yawDegrees = 3.0;
    matched.timeText = QStringLiteral("2026-01-01T00:00:00Z");
    matched.stdDevNorthMeters = 0.2;
    matched.stdDevEastMeters = 0.3;
    matched.stdDevUpMeters = 0.4;
    matched.stdDevHorizontalMeters = 0.5;
    auto unmatched = matched;
    unmatched.fileName = QStringLiteral("folder\\IMG_MISSING.JPG");
    unmatched.wgs84LongitudeDegrees = 32.0;
    imported.records = {matched, unmatched};
    imported.leverArm = xjw::gui::reference_import::LeverArm{0.1, -0.2, 0.3};

    const camera_reference::CameraReferenceSet reference_set =
        reference::buildMetashapeCameraReferenceSet(
            imported,
            makeProjectMetadata(),
            QStringLiteral("C:\\Metadata\\Cameras_WGS84.txt"),
            QStringLiteral("C:\\Metadata\\GNSS_offset.txt"),
            QString(64, QLatin1Char('a')));

    ASSERT_EQ(reference_set.records().size(), 1);
    ASSERT_EQ(reference_set.unmatchedRecords().size(), 1);
    EXPECT_EQ(reference_set.source().displayName, QStringLiteral("Cameras_WGS84.txt"));
    const auto &raw = reference_set.records().front().raw;
    ASSERT_TRUE(raw.position);
    EXPECT_EQ(*raw.position, (camera_reference::Vector3d{{31.0, 59.0, 54.0}}));
    ASSERT_TRUE(raw.positionSigma);
    EXPECT_EQ(*raw.positionSigma, (camera_reference::Vector3d{{0.3, 0.2, 0.4}}));
    EXPECT_EQ(raw.positionSigmaFrame, QStringLiteral("local_enu"));
    EXPECT_EQ(raw.positionSigmaUnit, QStringLiteral("m"));
    ASSERT_TRUE(raw.orientationYprDegrees);
    EXPECT_EQ(*raw.orientationYprDegrees,
              (camera_reference::Vector3d{{3.0, 2.0, 1.0}}));
    const auto &unmatched_raw = reference_set.unmatchedRecords().front().raw;
    ASSERT_TRUE(unmatched_raw.position);
    EXPECT_DOUBLE_EQ((*unmatched_raw.position)[0], 32.0);
    EXPECT_EQ(reference_set.unmatchedRecords().front().leverArmId,
              QStringLiteral("default_gnss_offset"));
}

TEST(CameraReferenceCsvExporterTest, WritesMatchedAndUnmatchedRowsBeforeCommit)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("camera_references.csv"));
    QString error;
    ASSERT_TRUE(reference::exportCameraReferenceCsv(makeCameraReferenceSet(), path, &error))
        << qPrintable(error);

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text));
    const QByteArray csv = file.readAll();
    EXPECT_TRUE(csv.contains("match_status"));
    EXPECT_TRUE(csv.contains("IMG_0001.JPG"));
    EXPECT_TRUE(csv.contains("IMG_MISSING.JPG"));
    EXPECT_TRUE(csv.contains("unmatched"));
    EXPECT_EQ(csv.count('\n'), 3);
}

TEST(ProjectCameraReferenceRepositoryTest, ClearsPublishedStateWhenReloadFails)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString projectPath = directory.filePath(QStringLiteral("references.plascan"));
    ProjectData projectData;
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("references")));

    const QString sidecarPath = xjw::common::project::ProjectIO::cameraReferenceSetPath(
        projectPath);
    const auto saved = camera_reference::CameraReferenceSetStore(sidecarPath).save(
        makeCameraReferenceSet());
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    reference::ProjectCameraReferenceRepository repository(&projectData);
    QString error;
    ASSERT_TRUE(repository.open(&error)) << qPrintable(error);
    ASSERT_FALSE(repository.referenceSet().records().isEmpty());

    QFile corrupt(sidecarPath);
    ASSERT_TRUE(corrupt.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_GT(corrupt.write("{broken"), 0);
    corrupt.close();

    EXPECT_FALSE(repository.open(&error));
    EXPECT_TRUE(repository.referenceSet().records().isEmpty());
    EXPECT_TRUE(repository.referenceSet().unmatchedRecords().isEmpty());
    EXPECT_TRUE(error.contains(QStringLiteral("JSON")));
}

TEST(ProjectCameraReferenceRepositoryTest, RejectsStaleImageSetFingerprint)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString projectPath = directory.filePath(QStringLiteral("stale.plascan"));
    ProjectData projectData;
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("stale")));

    camera_reference::CameraReferenceSet reference_set = makeCameraReferenceSet();
    reference_set.setImageSetFingerprint(QStringLiteral("stale-fingerprint"));
    const QString sidecarPath = xjw::common::project::ProjectIO::cameraReferenceSetPath(
        projectPath);
    const auto saved = camera_reference::CameraReferenceSetStore(sidecarPath).save(reference_set);
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    reference::ProjectCameraReferenceRepository repository(&projectData);
    QString error;
    EXPECT_FALSE(repository.open(&error));
    EXPECT_TRUE(repository.referenceSet().records().isEmpty());
    EXPECT_TRUE(error.contains(QStringLiteral("影像集合已变化")));
}

TEST(CameraReferenceProjectIdentityTest, UsesStableUuidsAcrossPathRelocationAndOrder)
{
    QJsonObject metadata = makeProjectMetadata();
    QJsonObject files = metadata.value(QStringLiteral("project_files")).toObject();
    QJsonArray images = files.value(QStringLiteral("images")).toArray();
    QJsonObject second{
        {QStringLiteral("image_uuid"), QStringLiteral("image-uuid-2")},
        {QStringLiteral("path"), QStringLiteral("project/images/IMG_0002.JPG")}
    };
    images.append(second);
    files[QStringLiteral("images")] = images;
    metadata[QStringLiteral("project_files")] = files;

    QJsonObject reordered = metadata;
    QJsonObject reordered_files = files;
    QJsonObject first = images.at(0).toObject();
    first[QStringLiteral("path")] = QStringLiteral("D:/moved/IMG_0001.JPG");
    reordered_files[QStringLiteral("images")] = QJsonArray{second, first};
    reordered[QStringLiteral("project_files")] = reordered_files;

    EXPECT_EQ(reference::cameraReferenceImageSetFingerprint(metadata),
              reference::cameraReferenceImageSetFingerprint(reordered));

    first[QStringLiteral("image_uuid")] = QStringLiteral("image-uuid-replaced");
    reordered_files[QStringLiteral("images")] = QJsonArray{second, first};
    reordered[QStringLiteral("project_files")] = reordered_files;
    EXPECT_NE(reference::cameraReferenceImageSetFingerprint(metadata),
              reference::cameraReferenceImageSetFingerprint(reordered));
}

TEST(MarkerReferenceTreeModelTest, GroupsReferenceMarkersAndExcludesTieMarkers)
{
    using Model = reference::MarkerReferenceTreeModel;
    const MarkerFixture fixture = makeMarkerFixture();
    Model model(fixture.markerSet);

    ASSERT_EQ(model.rowCount(), 1);
    const QModelIndex total = model.index(0, Model::LabelColumn);
    ASSERT_EQ(model.rowCount(total), 2);
    EXPECT_EQ(total.data(Model::NodeTypeRole).toInt(),
              static_cast<int>(Model::NodeType::TotalError));

    const QModelIndex control_group = model.index(0, Model::LabelColumn, total);
    const QModelIndex check_group = model.index(1, Model::LabelColumn, total);
    EXPECT_EQ(control_group.data().toString(), QStringLiteral("控制点"));
    EXPECT_EQ(check_group.data().toString(), QStringLiteral("检查点"));
    EXPECT_EQ(control_group.data(Model::NodeTypeRole).toInt(),
              static_cast<int>(Model::NodeType::Group));
    ASSERT_EQ(model.rowCount(control_group), 1);
    ASSERT_EQ(model.rowCount(check_group), 1);

    const QModelIndex control = model.index(0, Model::LabelColumn, control_group);
    const QModelIndex check = model.index(0, Model::LabelColumn, check_group);
    EXPECT_EQ(control.data().toString(), QStringLiteral("GCP-1"));
    EXPECT_EQ(check.data().toString(), QStringLiteral("CHK-1"));
    EXPECT_NE(control.data().toString(), QStringLiteral("TIE-1"));
    EXPECT_NE(check.data().toString(), QStringLiteral("TIE-1"));
    EXPECT_EQ(control.data(Model::NodeTypeRole).toInt(),
              static_cast<int>(Model::NodeType::Marker));
    EXPECT_EQ(control.data(Model::MarkerIdRole).toString(), fixture.controlId);
    EXPECT_EQ(control.data(Model::MarkerRoleRole).toInt(),
              static_cast<int>(control_points::MarkerRole::ControlPoint));
    EXPECT_EQ(check.data(Model::MarkerIdRole).toString(), fixture.checkId);
    EXPECT_EQ(check.data(Model::MarkerRoleRole).toInt(),
              static_cast<int>(control_points::MarkerRole::CheckPoint));

    EXPECT_DOUBLE_EQ(model.index(0, Model::SourceXColumn, control_group).data().toDouble(), 10.0);
    EXPECT_DOUBLE_EQ(model.index(0, Model::SourceYColumn, control_group).data().toDouble(), 20.0);
    EXPECT_DOUBLE_EQ(model.index(0, Model::SourceZColumn, control_group).data().toDouble(), 30.0);
    EXPECT_DOUBLE_EQ(model.index(0, Model::ResidualColumn, control_group).data().toDouble(), 3.0);
    EXPECT_DOUBLE_EQ(model.index(0, Model::ResidualColumn, check_group).data().toDouble(), 4.0);
    EXPECT_NEAR(model.index(0, Model::ResidualColumn).data().toDouble(),
                std::sqrt(12.5),
                1.0e-12);
}

TEST(ScaleBarReferenceTreeModelTest, GroupsScaleBarsAndExposesValuesResidualsAndRoles)
{
    using Model = reference::ScaleBarReferenceTreeModel;
    const MarkerFixture fixture = makeMarkerFixture();
    QJsonObject encoded = control_points::MarkerSetJson::encode(fixture.markerSet);
    QJsonArray scale_bars = encoded.value(QStringLiteral("scale_bars")).toArray();
    ASSERT_EQ(scale_bars.size(), 2);
    for (int index = 0; index < scale_bars.size(); ++index)
    {
        QJsonObject scale_bar = scale_bars.at(index).toObject();
        const bool is_control = scale_bar.value(QStringLiteral("id")).toString()
            == fixture.controlScaleId;
        scale_bar.insert(QStringLiteral("estimated_distance"), is_control ? 10.4 : 19.5);
        scale_bar.insert(QStringLiteral("residual"), is_control ? 0.4 : -0.5);
        scale_bars[index] = scale_bar;
    }
    encoded.insert(QStringLiteral("scale_bars"), scale_bars);
    control_points::MarkerSet marker_set;
    QString error;
    ASSERT_TRUE(control_points::MarkerSetJson::decode(encoded, &marker_set, &error))
        << qPrintable(error);

    Model model(marker_set);

    ASSERT_EQ(model.rowCount(), 1);
    const QModelIndex total = model.index(0, Model::LabelColumn);
    ASSERT_EQ(model.rowCount(total), 2);
    const QModelIndex control_group = model.index(0, Model::LabelColumn, total);
    const QModelIndex check_group = model.index(1, Model::LabelColumn, total);
    EXPECT_EQ(control_group.data().toString(), QStringLiteral("控制标尺"));
    EXPECT_EQ(check_group.data().toString(), QStringLiteral("检查标尺"));
    ASSERT_EQ(model.rowCount(control_group), 1);
    ASSERT_EQ(model.rowCount(check_group), 1);

    const QModelIndex control = model.index(0, Model::LabelColumn, control_group);
    const QModelIndex check = model.index(0, Model::LabelColumn, check_group);
    EXPECT_EQ(control.data(Model::NodeTypeRole).toInt(),
              static_cast<int>(Model::NodeType::ScaleBar));
    EXPECT_EQ(control.data(Model::ScaleBarIdRole).toString(), fixture.controlScaleId);
    EXPECT_EQ(control.data(Model::ScaleBarRoleRole).toInt(),
              static_cast<int>(control_points::ScaleBarRole::Control));
    EXPECT_EQ(control.data(Model::FirstMarkerIdRole).toString(), fixture.controlId);
    EXPECT_EQ(control.data(Model::SecondMarkerIdRole).toString(), fixture.checkId);
    EXPECT_EQ(check.data(Model::ScaleBarIdRole).toString(), fixture.checkScaleId);
    EXPECT_EQ(check.data(Model::ScaleBarRoleRole).toInt(),
              static_cast<int>(control_points::ScaleBarRole::Check));

    EXPECT_DOUBLE_EQ(model.index(0, Model::SourceValueColumn, control_group).data().toDouble(),
                     10.0);
    EXPECT_DOUBLE_EQ(model.index(0, Model::EstimatedValueColumn, control_group).data().toDouble(),
                     10.4);
    EXPECT_DOUBLE_EQ(model.index(0, Model::ResidualColumn, control_group).data().toDouble(), 0.4);
    EXPECT_DOUBLE_EQ(model.index(0, Model::SourceValueColumn, check_group).data().toDouble(), 20.0);
    EXPECT_DOUBLE_EQ(model.index(0, Model::EstimatedValueColumn, check_group).data().toDouble(),
                     19.5);
    EXPECT_DOUBLE_EQ(model.index(0, Model::ResidualColumn, check_group).data().toDouble(), -0.5);
    EXPECT_NEAR(model.index(0, Model::ResidualColumn).data().toDouble(),
                std::sqrt(0.205),
                1.0e-12);
}

TEST(ReferenceMarkerModelsTest, DisabledItemsDoNotAffectTotalResiduals)
{
    MarkerFixture fixture = makeMarkerFixture();
    fixture.markerSet.setMarkerEnabled(fixture.controlId, false);
    reference::MarkerReferenceTreeModel marker_model(fixture.markerSet);
    EXPECT_DOUBLE_EQ(marker_model.index(
        0, reference::MarkerReferenceTreeModel::ResidualColumn).data().toDouble(), 4.0);

    QJsonObject encoded = control_points::MarkerSetJson::encode(fixture.markerSet);
    QJsonArray scale_bars = encoded.value(QStringLiteral("scale_bars")).toArray();
    for (int index = 0; index < scale_bars.size(); ++index)
    {
        QJsonObject scale_bar = scale_bars.at(index).toObject();
        const bool is_control = scale_bar.value(QStringLiteral("id")).toString()
            == fixture.controlScaleId;
        scale_bar[QStringLiteral("enabled")] = !is_control;
        scale_bar[QStringLiteral("estimated_distance")] = is_control ? 10.4 : 19.5;
        scale_bar[QStringLiteral("residual")] = is_control ? 0.4 : -0.5;
        scale_bars[index] = scale_bar;
    }
    encoded[QStringLiteral("scale_bars")] = scale_bars;
    control_points::MarkerSet marker_set;
    QString error;
    ASSERT_TRUE(control_points::MarkerSetJson::decode(encoded, &marker_set, &error))
        << qPrintable(error);

    reference::ScaleBarReferenceTreeModel scale_model(marker_set);
    EXPECT_DOUBLE_EQ(scale_model.index(
        0, reference::ScaleBarReferenceTreeModel::ResidualColumn).data().toDouble(), 0.5);
}
