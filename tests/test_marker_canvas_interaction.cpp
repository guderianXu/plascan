#include "CanvasWidget.h"
#include "MarkerWorkspaceController.h"
#include "ProjectData.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QContextMenuEvent>
#include <QDir>
#include <QImage>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

namespace
{

QString createTestImage(QTemporaryDir *directory)
{
    const QString path = directory->filePath(QStringLiteral("marker_canvas.png"));
    QImage image(640, 480, QImage::Format_RGB32);
    image.fill(Qt::white);
    return image.save(path) ? path : QString();
}

} // namespace

TEST(MarkerCanvasInteractionTest, RightClickUsesOriginalPixelAfterViewRotation)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = createTestImage(&directory);
    ASSERT_FALSE(image_path.isEmpty());

    CanvasWidget canvas;
    canvas.resize(900, 600);
    canvas.show();

    QSignalSpy ready_spy(&canvas, &CanvasWidget::displayImageReadyChanged);
    canvas.showImage(image_path);
    ASSERT_TRUE(ready_spy.wait(5000));
    ASSERT_TRUE(canvas.hasDisplayImage());

    canvas.setViewRotationDegrees(90);
    QSignalSpy context_spy(&canvas, &CanvasWidget::imageContextRequested);

    const QPointF expected_pixel(123.0, 210.0);
    const QPoint viewport_position = canvas.mapFromScene(expected_pixel);
    ASSERT_TRUE(canvas.viewport()->rect().contains(viewport_position));
    QContextMenuEvent context_event(QContextMenuEvent::Mouse,
                                    viewport_position,
                                    canvas.viewport()->mapToGlobal(viewport_position));
    QApplication::sendEvent(canvas.viewport(), &context_event);

    ASSERT_EQ(context_spy.count(), 1);
    const QList<QVariant> arguments = context_spy.takeFirst();
    EXPECT_EQ(QDir::cleanPath(arguments.at(0).toString()), QDir::cleanPath(image_path));
    const QPointF actual_pixel = arguments.at(1).toPointF();
    EXPECT_NEAR(actual_pixel.x(), expected_pixel.x(), 1.0);
    EXPECT_NEAR(actual_pixel.y(), expected_pixel.y(), 1.0);
}

TEST(MarkerCanvasInteractionTest, RightClickOutsideImageDoesNotEmitContext)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = createTestImage(&directory);
    ASSERT_FALSE(image_path.isEmpty());

    CanvasWidget canvas;
    canvas.resize(1000, 600);
    canvas.show();

    QSignalSpy ready_spy(&canvas, &CanvasWidget::displayImageReadyChanged);
    canvas.showImage(image_path);
    ASSERT_TRUE(ready_spy.wait(5000));

    QSignalSpy context_spy(&canvas, &CanvasWidget::imageContextRequested);
    const QPoint outside_position(5, 300);
    QContextMenuEvent context_event(QContextMenuEvent::Mouse,
                                    outside_position,
                                    canvas.viewport()->mapToGlobal(outside_position));
    QApplication::sendEvent(canvas.viewport(), &context_event);
    EXPECT_EQ(context_spy.count(), 0);
}

TEST(MarkerCanvasInteractionTest, SwitchingImageKeepsCurrentTransformUntilReplacementIsReady)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image_path = createTestImage(&directory);
    ASSERT_FALSE(first_image_path.isEmpty());

    const QString second_image_path = directory.filePath(QStringLiteral("marker_canvas_second.png"));
    QImage second_image(1200, 320, QImage::Format_RGB32);
    second_image.fill(Qt::black);
    ASSERT_TRUE(second_image.save(second_image_path));

    CanvasWidget canvas;
    canvas.resize(900, 600);
    canvas.show();

    QSignalSpy ready_spy(&canvas, &CanvasWidget::displayImageReadyChanged);
    canvas.showImage(first_image_path);
    ASSERT_TRUE(ready_spy.wait(5000));

    canvas.zoomIn();
    const QTransform transform_before_switch = canvas.transform();
    canvas.showImage(second_image_path);

    EXPECT_EQ(canvas.transform(), transform_before_switch);
    ASSERT_TRUE(ready_spy.wait(5000));
    EXPECT_TRUE(canvas.hasDisplayImage());
    EXPECT_EQ(canvas.viewRotationDegrees(), 0);
}

TEST(MarkerCanvasInteractionTest, PhotoCommandsCreateMoveBlockAndRemoveOneProjection)
{
    using xjw::control_points::ProjectionState;
    using xjw::gui::markers::MarkerPhotoCommand;
    using xjw::gui::markers::MarkerWorkspaceController;

    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = createTestImage(&directory);
    ASSERT_FALSE(image_path.isEmpty());

    ProjectData project;
    const QString project_path = directory.filePath(QStringLiteral("marker_test.plascan"));
    ASSERT_TRUE(project.createProject(project_path, QStringLiteral("marker_test")));
    ASSERT_TRUE(project.addImages({image_path}));
    const QString project_image_path = project.getAllImages().constFirst();
    ASSERT_FALSE(project_image_path.isEmpty());

    CanvasWidget canvas;
    MarkerWorkspaceController controller(&canvas, &project);
    QString error;
    ASSERT_TRUE(controller.openProject(&error)) << error.toStdString();

    ASSERT_TRUE(controller.executePhotoCommand(
        MarkerPhotoCommand::AddNewMarker,
        project_image_path,
        QPointF(10.0, 20.0),
        {},
        &error))
        << error.toStdString();
    ASSERT_EQ(controller.markerSet().markers().size(), 1);
    const QString marker_id = controller.markerSet().markers().front().id;

    ASSERT_TRUE(controller.executePhotoCommand(
        MarkerPhotoCommand::PlaceExistingMarker,
        project_image_path,
        QPointF(30.0, 40.0),
        marker_id,
        &error));
    const auto &moved_marker = controller.markerSet().marker(marker_id);
    ASSERT_EQ(moved_marker.projections.size(), 1);
    EXPECT_EQ(moved_marker.projections.front().xy, QPointF(30.0, 40.0));

    ASSERT_TRUE(controller.executePhotoCommand(
        MarkerPhotoCommand::BlockProjection, project_image_path, {}, marker_id, &error));
    EXPECT_EQ(controller.markerSet().marker(marker_id).projections.front().state,
              ProjectionState::Blocked);

    ASSERT_TRUE(controller.executePhotoCommand(
        MarkerPhotoCommand::UnblockProjection, project_image_path, {}, marker_id, &error));
    EXPECT_EQ(controller.markerSet().marker(marker_id).projections.front().state,
              ProjectionState::ManualPinned);

    ASSERT_TRUE(controller.executePhotoCommand(
        MarkerPhotoCommand::RemoveProjection, project_image_path, {}, marker_id, &error));
    EXPECT_TRUE(controller.markerSet().marker(marker_id).projections.isEmpty());
}
