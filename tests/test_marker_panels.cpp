#include "CanvasWidget.h"
#include "DetectMarkersDialog.h"
#include "detection/DetectionReviewStore.h"
#include "MarkerFocusMeasurementDialog.h"
#include "MarkerDetectionReviewDialog.h"
#include "MarkerReferencePanel.h"
#include "MarkerWorkspaceController.h"
#include "PrintMarkersDialog.h"
#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"

#include <gtest/gtest.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QImage>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTest>

#include <algorithm>
#include <memory>

namespace
{

QString writePanelImage(QTemporaryDir *directory, const QString &name)
{
    const QString path = directory->filePath(name);
    QImage image(320, 240, QImage::Format_RGB32);
    image.fill(name.contains(QStringLiteral("right")) ? Qt::black : Qt::white);
    return image.save(path) ? path : QString();
}

struct PanelFixture
{
    QTemporaryDir directory;
    QString firstImage;
    QString secondImage;
    ProjectData project;
    CanvasWidget canvas;
    std::unique_ptr<xjw::gui::markers::MarkerWorkspaceController> controller;

    bool initialize()
    {
        firstImage = writePanelImage(&directory, QStringLiteral("left.png"));
        secondImage = writePanelImage(&directory, QStringLiteral("right.png"));
        if (firstImage.isEmpty() || secondImage.isEmpty()) return false;
        if (!project.createProject(directory.filePath(QStringLiteral("panel.plascan")),
                                   QStringLiteral("panel"))) return false;
        if (!project.addImages({firstImage, secondImage})) return false;
        const QStringList projectImages = project.getAllImages();
        if (projectImages.size() != 2) return false;
        firstImage = projectImages[0];
        secondImage = projectImages[1];
        controller = std::make_unique<xjw::gui::markers::MarkerWorkspaceController>(
            &canvas, &project);
        QString error;
        if (!controller->openProject(&error)) return false;
        return controller->executePhotoCommand(
            xjw::gui::markers::MarkerPhotoCommand::AddNewMarker,
            firstImage,
            QPointF(50.0, 60.0),
            {},
            &error);
    }
};

} // namespace

TEST(MarkerPanelTest, AppliesControlPointRoleAndSeparateReferenceAccuracy)
{
    PanelFixture fixture;
    ASSERT_TRUE(fixture.initialize());
    const QString marker_id = fixture.controller->markerSet().markers().front().id;

    xjw::gui::markers::MarkerReferencePanel panel;
    panel.setController(fixture.controller.get());
    panel.selectMarker(marker_id);
    panel.show();

    auto *role = panel.findChild<QComboBox *>(QStringLiteral("markerRoleCombo"));
    auto *has_reference = panel.findChild<QCheckBox *>(QStringLiteral("markerHasReferenceCheck"));
    auto *x = panel.findChild<QDoubleSpinBox *>(QStringLiteral("markerReferenceX"));
    auto *sigma_xy = panel.findChild<QDoubleSpinBox *>(QStringLiteral("markerReferenceSigmaXY"));
    auto *sigma_z = panel.findChild<QDoubleSpinBox *>(QStringLiteral("markerReferenceSigmaZ"));
    auto *apply = panel.findChild<QPushButton *>(QStringLiteral("applyMarkerPropertiesButton"));
    ASSERT_NE(role, nullptr);
    ASSERT_NE(has_reference, nullptr);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(sigma_xy, nullptr);
    ASSERT_NE(sigma_z, nullptr);
    ASSERT_NE(apply, nullptr);

    role->setCurrentIndex(role->findData(
        static_cast<int>(xjw::control_points::MarkerRole::ControlPoint)));
    has_reference->setChecked(true);
    x->setValue(123.5);
    sigma_xy->setValue(0.02);
    sigma_z->setValue(0.05);
    apply->click();

    const auto &marker = fixture.controller->markerSet().marker(marker_id);
    EXPECT_EQ(marker.role, xjw::control_points::MarkerRole::ControlPoint);
    ASSERT_TRUE(marker.referenceCoordinate.has_value());
    EXPECT_DOUBLE_EQ(marker.referenceCoordinate->x, 123.5);
    EXPECT_DOUBLE_EQ(marker.referenceCoordinate->sigmaX, 0.02);
    EXPECT_DOUBLE_EQ(marker.referenceCoordinate->sigmaY, 0.02);
    EXPECT_DOUBLE_EQ(marker.referenceCoordinate->sigmaZ, 0.05);
}

TEST(MarkerFocusMeasurementDialogTest, ConfirmingCandidateCreatesPinnedProjection)
{
    PanelFixture fixture;
    ASSERT_TRUE(fixture.initialize());
    const QString marker_id = fixture.controller->markerSet().markers().front().id;

    xjw::gui::markers::MarkerFocusMeasurementDialog dialog;
    ASSERT_TRUE(dialog.setContext(fixture.controller.get(),
                                  &fixture.project,
                                  marker_id,
                                  fixture.firstImage,
                                  fixture.secondImage));
    dialog.show();
    dialog.setCandidatePixel(QPointF(90.0, 110.0));
    auto *confirm = dialog.findChild<QPushButton *>(QStringLiteral("confirmMarkerProjectionButton"));
    ASSERT_NE(confirm, nullptr);
    confirm->click();

    const auto &marker = fixture.controller->markerSet().marker(marker_id);
    ASSERT_EQ(marker.projections.size(), 2);
    const auto candidate = std::find_if(marker.projections.cbegin(), marker.projections.cend(),
                                        [&fixture](const auto &projection)
    {
        return QDir::cleanPath(projection.imagePathSnapshot)
            == QDir::cleanPath(fixture.secondImage);
    });
    ASSERT_NE(candidate, marker.projections.cend());
    EXPECT_EQ(candidate->xy, QPointF(90.0, 110.0));
    EXPECT_EQ(candidate->state, xjw::control_points::ProjectionState::ManualPinned);
    QTest::qWait(50);
}

TEST(DetectMarkersDialogTest, ExposesSupportedFamiliesAndUsesWholeProjectProgress)
{
    PanelFixture fixture;
    ASSERT_TRUE(fixture.initialize());

    xjw::gui::markers::DetectMarkersDialog dialog;
    ASSERT_TRUE(dialog.setContext(fixture.controller.get(), &fixture.project));

    auto *family = dialog.findChild<QComboBox *>(QStringLiteral("markerTargetFamilyCombo"));
    auto *progress = dialog.findChild<QProgressBar *>(QStringLiteral("markerDetectionProgress"));
    auto *start = dialog.findChild<QPushButton *>(QStringLiteral("startMarkerDetectionButton"));
    ASSERT_NE(family, nullptr);
    ASSERT_NE(progress, nullptr);
    ASSERT_NE(start, nullptr);
    EXPECT_EQ(family->count(), 13);
    EXPECT_EQ(progress->minimum(), 0);
    EXPECT_EQ(progress->maximum(), 2);
    EXPECT_TRUE(start->isEnabled());

    const int circular_index = family->findData(
        static_cast<int>(xjw::control_points::MarkerTargetFamily::Circular12Bit));
    ASSERT_GE(circular_index, 0);
    auto *model = qobject_cast<QStandardItemModel *>(family->model());
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(model->item(circular_index)->isEnabled());
}

TEST(PrintMarkersDialogTest, ExposesPhysicalLayoutControlsAndDisablesUnavailableCircularCodes)
{
    xjw::gui::markers::PrintMarkersDialog dialog;
    auto *family = dialog.findChild<QComboBox *>(QStringLiteral("printMarkerFamilyCombo"));
    auto *diameter = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("printMarkerDiameterMm"));
    auto *generate = dialog.findChild<QPushButton *>(QStringLiteral("generateMarkerPdfButton"));
    ASSERT_NE(family, nullptr);
    ASSERT_NE(diameter, nullptr);
    ASSERT_NE(generate, nullptr);
    EXPECT_EQ(family->count(), 13);
    EXPECT_DOUBLE_EQ(diameter->value(), 30.0);

    const int circular_index = family->findData(
        static_cast<int>(xjw::control_points::MarkerTargetFamily::Circular12Bit));
    auto *model = qobject_cast<QStandardItemModel *>(family->model());
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(model->item(circular_index)->isEnabled());
}

TEST(MarkerDetectionReviewTest, PersistsNonCodedCandidateUntilUserAcceptsIt)
{
    PanelFixture fixture;
    ASSERT_TRUE(fixture.initialize());
    const QJsonArray images = fixture.project.coreFilesMeta()
                                  .value(QStringLiteral("images"))
                                  .toArray();
    ASSERT_FALSE(images.isEmpty());
    const QString image_id = images.first().toObject()
                                 .value(QStringLiteral("image_uuid"))
                                 .toString();
    ASSERT_FALSE(image_id.isEmpty());

    xjw::gui::markers::MarkerDetectionTaskResult task;
    task.baseRevision = fixture.controller->markerRevision();
    xjw::control_points::MarkerDetectionObservation observation;
    observation.imageId = image_id;
    observation.imagePathSnapshot = fixture.firstImage;
    observation.detection.family =
        xjw::control_points::MarkerTargetFamily::NonCodedCircle;
    observation.detection.center = QPointF(120.25, 90.75);
    observation.detection.confidence = 0.93;
    observation.detection.centerSigmaPx = 0.2;
    observation.detection.source = QStringLiteral("noncoded:circle");
    task.observations.push_back(observation);

    xjw::control_points::DetectionIntegrationResult integration;
    QString error;
    ASSERT_TRUE(fixture.controller->applyDetectionTaskResult(task, &integration, &error))
        << error.toStdString();
    ASSERT_EQ(fixture.controller->detectionReviewQueue().entries.size(), 1);
    const auto entry = fixture.controller->detectionReviewQueue().entries.front();
    EXPECT_EQ(entry.reason, QStringLiteral("unassociated_non_coded"));

    const QString review_path = xjw::common::project::ProjectIO::markerDetectionReviewPath(
        fixture.project.currentProjectPath());
    const auto persisted = xjw::control_points::DetectionReviewStore(review_path).load();
    ASSERT_TRUE(persisted.ok) << persisted.error.toStdString();
    ASSERT_EQ(persisted.queue.entries.size(), 1);

    const int marker_count_before = fixture.controller->markerSet().markers().size();
    ASSERT_TRUE(fixture.controller->acceptDetectionReview(entry.id, {}, &error))
        << error.toStdString();
    EXPECT_TRUE(fixture.controller->detectionReviewQueue().entries.isEmpty());
    ASSERT_EQ(fixture.controller->markerSet().markers().size(), marker_count_before + 1);
    const auto &projection = fixture.controller->markerSet().markers().back().projections.front();
    EXPECT_EQ(projection.imageId, image_id);
    EXPECT_EQ(projection.xy, QPointF(120.25, 90.75));
    EXPECT_EQ(projection.state, xjw::control_points::ProjectionState::AutoDetected);

    const auto after_accept = xjw::control_points::DetectionReviewStore(review_path).load();
    ASSERT_TRUE(after_accept.ok) << after_accept.error.toStdString();
    EXPECT_TRUE(after_accept.queue.entries.isEmpty());
}

TEST(MarkerDetectionReviewDialogTest, AcceptsSelectedCandidateIntoNewMarker)
{
    PanelFixture fixture;
    ASSERT_TRUE(fixture.initialize());
    const QJsonArray images = fixture.project.coreFilesMeta()
                                  .value(QStringLiteral("images"))
                                  .toArray();
    const QString image_id = images.first().toObject()
                                 .value(QStringLiteral("image_uuid"))
                                 .toString();

    xjw::gui::markers::MarkerDetectionTaskResult task;
    task.baseRevision = fixture.controller->markerRevision();
    xjw::control_points::MarkerDetectionObservation observation;
    observation.imageId = image_id;
    observation.imagePathSnapshot = fixture.firstImage;
    observation.detection.family =
        xjw::control_points::MarkerTargetFamily::NonCodedFourQuadrant;
    observation.detection.center = QPointF(80.0, 70.0);
    observation.detection.confidence = 0.88;
    observation.detection.source = QStringLiteral("noncoded:four-quadrant");
    task.observations.push_back(observation);
    xjw::control_points::DetectionIntegrationResult integration;
    QString error;
    ASSERT_TRUE(fixture.controller->applyDetectionTaskResult(task, &integration, &error));

    xjw::gui::markers::MarkerDetectionReviewDialog dialog;
    ASSERT_TRUE(dialog.setController(fixture.controller.get()));
    auto *table = dialog.findChild<QTableWidget *>(QStringLiteral("markerDetectionReviewTable"));
    auto *assignment = dialog.findChild<QComboBox *>(QStringLiteral("markerDetectionAssignmentCombo"));
    auto *accept = dialog.findChild<QPushButton *>(QStringLiteral("acceptMarkerDetectionButton"));
    auto *discard = dialog.findChild<QPushButton *>(QStringLiteral("discardMarkerDetectionButton"));
    ASSERT_NE(table, nullptr);
    ASSERT_NE(assignment, nullptr);
    ASSERT_NE(accept, nullptr);
    ASSERT_NE(discard, nullptr);
    ASSERT_EQ(table->rowCount(), 1);
    EXPECT_EQ(assignment->count(), fixture.controller->markerSet().markers().size() + 1);
    EXPECT_TRUE(accept->isEnabled());
    EXPECT_TRUE(discard->isEnabled());

    const int marker_count_before = fixture.controller->markerSet().markers().size();
    accept->click();
    EXPECT_EQ(table->rowCount(), 0);
    EXPECT_EQ(fixture.controller->markerSet().markers().size(), marker_count_before + 1);
    EXPECT_TRUE(fixture.controller->detectionReviewQueue().entries.isEmpty());
}

TEST(MarkerDetectionReviewTest, CodedCandidateReusesExistingTargetIdentity)
{
    PanelFixture fixture;
    ASSERT_TRUE(fixture.initialize());
    const QString image_id = fixture.project.coreFilesMeta()
                                 .value(QStringLiteral("images"))
                                 .toArray()
                                 .first()
                                 .toObject()
                                 .value(QStringLiteral("image_uuid"))
                                 .toString();

    xjw::control_points::MarkerDetectionObservation best;
    best.imageId = image_id;
    best.imagePathSnapshot = fixture.firstImage;
    best.detection.family = xjw::control_points::MarkerTargetFamily::AprilTag36h11;
    best.detection.targetId = 5;
    best.detection.center = QPointF(90.0, 80.0);
    best.detection.confidence = 0.95;
    best.detection.source = QStringLiteral("apriltag:tag36h11");
    auto duplicate = best;
    duplicate.detection.center = QPointF(150.0, 120.0);
    duplicate.detection.confidence = 0.65;

    xjw::gui::markers::MarkerDetectionTaskResult task;
    task.baseRevision = fixture.controller->markerRevision();
    task.observations = {best, duplicate};
    xjw::control_points::DetectionIntegrationResult integration;
    QString error;
    ASSERT_TRUE(fixture.controller->applyDetectionTaskResult(task, &integration, &error));
    ASSERT_EQ(fixture.controller->detectionReviewQueue().entries.size(), 1);
    const int marker_count_before = fixture.controller->markerSet().markers().size();
    const QString entry_id = fixture.controller->detectionReviewQueue().entries.front().id;

    ASSERT_TRUE(fixture.controller->acceptDetectionReview(entry_id, {}, &error))
        << error.toStdString();
    EXPECT_EQ(fixture.controller->markerSet().markers().size(), marker_count_before);
    int target_identity_count = 0;
    for (const auto &marker : fixture.controller->markerSet().markers())
    {
        if (marker.targetIdentity.has_value()
            && marker.targetIdentity->family == QLatin1String("tag36h11")
            && marker.targetIdentity->encodedId == 5)
        {
            ++target_identity_count;
        }
    }
    EXPECT_EQ(target_identity_count, 1);
}
