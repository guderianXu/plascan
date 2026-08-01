// ============================================================
// test_gui_project_utils.cpp — GUI project 层公共工具与三角化服务测试
// ============================================================

#include <gtest/gtest.h>

#include "project/ProjectIO.h"
#include "ProjectCameraImportService.h"
#include "ProjectResourceCleanupService.h"
#include "ProjectTiePointResultService.h"
#include "ProjectData.h"
#include "DepthFrameUtils.h"
#include "ProjectFilesManager.h"
#include "ProjectDashboardSummary.h"
#include "ProjectReferenceDatasets.h"
#include "ProjectReferenceTerrainBa.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "ProjectSurveyControl.h"
#include "io/MarkerSetStore.h"
#include "TriangulationService.h"
#include "ProjectWorkflowReports.h"
#include "ProjectWorkflowUtils.h"
#include "ProjectResultRecords.h"
#include "ProjectMetadataOperations.h"
#include "ProjectDenseWorkflowConfig.h"
#include "ProjectModelWorkflowPolicy.h"
#include "runtime/PythonRuntimeLocator.h"
#include "json/JsonObjectMerge.h"
#include "io/JsonObjectFile.h"
#include "ImageViewRotationSettings.h"
#include "DialogSettingStore.h"
#include "project/SparseResultQuality.h"
#include "tie_points/CleanTiePointsDialog.h"
#include "tie_points/CreateTiePointsDialog.h"
#include "tie_points/ThinTiePointsDialog.h"
#include "reconstruction/AerialTriangulationDialog.h"
#include "application/AboutDialog.h"
#include "reconstruction/MapProjectDialog.h"
#include "camera/SurveyControlDialog.h"
#include "image/GenerateMaskDialog.h"
#include "reconstruction/GenerateModelDialog.h"
#include "ModelDropSupport.h"
#include "DataTreeWidget.h"
#include "CanvasWidget.h"
#include "DualImageViewer.h"
#include "ImageViewWidget.h"
#include "MatchLineOverlay.h"
#include "DepthOverlayData.h"
#include "DepthOverlayController.h"
#include "FeatureResidualLoader.h"
#include "PhotoStripWidget.h"
#include "ProjectDashboardWidget.h"
#include "MatchValidityAnalyzer.h"
#include "MainMenu.h"
#include "ProjectUiHydrator.h"
#include "GuiTaskRunner.h"
#include "WorkspacePanelController.h"
#include "HenuBrandWidget.h"
#include "TaskStatusWidget.h"
#include "ObjRenderPreparation.h"

#include "Camera.h"
#include "DemDomIO.h"
#include "io/PathIO.h"

#include <plapoint/io/obj_io.h>
#include <plapoint/io/ply_io.h>

#include <QApplication>
#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDialogButtonBox>
#include <QDockWidget>
#include <QDir>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QGroupBox>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QItemSelectionModel>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QKeySequence>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMetaObject>
#include <QFileInfo>
#include <QPushButton>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStyle>
#include <QTemporaryDir>
#include <QTextStream>
#include <QToolBar>
#include <QToolButton>
#include <QTreeView>
#include <QHeaderView>
#include <QUrl>
#include <QWidget>
#include <QWidgetAction>
#include <QStandardItemModel>
#include <QSignalSpy>
#include <QTableWidget>
#include <QThread>
#include <QtTest/QTest>
#include <QtEndian>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

TEST(DepthOverlayDataTest, ResolvesExactReferenceAndRequestedLevels)
{
    const QJsonObject level_one{
        {QStringLiteral("level"), 1},
        {QStringLiteral("raw_depth_path"), QStringLiteral("a_l1.bin")},
        {QStringLiteral("valid_mask_path"), QStringLiteral("a_l1_mask.png")},
        {QStringLiteral("preview_path"), QStringLiteral("a_l1.png")}};
    const QJsonObject level_two{
        {QStringLiteral("level"), 2},
        {QStringLiteral("raw_depth_path"), QStringLiteral("a_l2.bin")},
        {QStringLiteral("valid_mask_path"), QStringLiteral("a_l2_mask.png")}};
    const QJsonObject level_three{
        {QStringLiteral("level"), 3},
        {QStringLiteral("raw_depth_path"), QStringLiteral("a_l3.bin")},
        {QStringLiteral("valid_mask_path"), QStringLiteral("a_l3_mask.png")}};
    const QJsonObject record_a{
        {QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
        {QStringLiteral("raw_depth_path"), QStringLiteral("a_final.bin")},
        {QStringLiteral("valid_mask_path"), QStringLiteral("a_final_mask.png")},
        {QStringLiteral("depth_png"), QStringLiteral("a_final.png")},
        {QStringLiteral("pyramid_levels"), QJsonArray{level_three, level_one, level_two}}};
    const QJsonObject record_b{
        {QStringLiteral("ref_image"), QStringLiteral("E:/images/b.png")},
        {QStringLiteral("raw_depth_path"), QStringLiteral("b_final.bin")},
        {QStringLiteral("valid_mask_path"), QStringLiteral("b_final_mask.png")}};
    const QJsonObject metadata{
        {QStringLiteral("depth_map_results"), QJsonArray{record_b, record_a}}};

    const auto final_artifact = xjw::gui::views::resolveDepthOverlayArtifact(
        metadata,
        QStringLiteral("e:\\images\\a.png"),
        xjw::gui::views::DepthOverlayLevel::Final);
    ASSERT_TRUE(final_artifact.has_value());
    EXPECT_EQ(final_artifact->referenceImage, QStringLiteral("E:/images/a.png"));
    EXPECT_EQ(final_artifact->rawDepthPath, QStringLiteral("a_final.bin"));
    EXPECT_EQ(final_artifact->validMaskPath, QStringLiteral("a_final_mask.png"));
    EXPECT_EQ(final_artifact->previewPath, QStringLiteral("a_final.png"));
    EXPECT_EQ(final_artifact->level, 0);

    const std::array<std::pair<xjw::gui::views::DepthOverlayLevel, QString>, 3> expected_levels{{
        {xjw::gui::views::DepthOverlayLevel::Level1, QStringLiteral("a_l1.bin")},
        {xjw::gui::views::DepthOverlayLevel::Level2, QStringLiteral("a_l2.bin")},
        {xjw::gui::views::DepthOverlayLevel::Level3, QStringLiteral("a_l3.bin")}}};
    for (const auto &[level, expected_path] : expected_levels)
    {
        const auto artifact = xjw::gui::views::resolveDepthOverlayArtifact(
            metadata, QStringLiteral("E:/images/a.png"), level);
        ASSERT_TRUE(artifact.has_value());
        EXPECT_EQ(artifact->rawDepthPath, expected_path);
    }
}

TEST(DepthOverlayDataTest, DoesNotFallbackAcrossImagesOrLevels)
{
    const QJsonObject record_a{
        {QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
        {QStringLiteral("pyramid_levels"),
         QJsonArray{QJsonObject{{QStringLiteral("level"), 2},
                                {QStringLiteral("raw_depth_path"), QStringLiteral("a_l2.bin")},
                                {QStringLiteral("valid_mask_path"), QStringLiteral("a_l2_mask.png")}}}}};
    const QJsonObject record_b{
        {QStringLiteral("ref_image"), QStringLiteral("E:/images/b.png")},
        {QStringLiteral("raw_depth_path"), QStringLiteral("b_final.bin")},
        {QStringLiteral("valid_mask_path"), QStringLiteral("b_final_mask.png")}};
    const QJsonObject metadata{
        {QStringLiteral("depth_map_results"), QJsonArray{record_a, record_b}}};

    EXPECT_FALSE(xjw::gui::views::resolveDepthOverlayArtifact(
        metadata, QStringLiteral("E:/images/b.png"), xjw::gui::views::DepthOverlayLevel::Level2));
    EXPECT_FALSE(xjw::gui::views::resolveDepthOverlayArtifact(
        metadata, QStringLiteral("E:/images/missing.png"), xjw::gui::views::DepthOverlayLevel::Final));
}

TEST(DepthOverlayControllerTest, AnyArtifactRemainsAvailableWhenSelectedLevelIsMissing)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString depth_path = directory.filePath(QStringLiteral("a_final.bin"));
    const QString mask_path = directory.filePath(QStringLiteral("a_final_mask.png"));
    QFile depth_file(depth_path);
    ASSERT_TRUE(depth_file.open(QIODevice::WriteOnly));
    depth_file.write("depth");
    depth_file.close();
    QImage mask(1, 1, QImage::Format_Grayscale8);
    mask.fill(255);
    ASSERT_TRUE(mask.save(mask_path));

    const QJsonObject record{
        {QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
        {QStringLiteral("raw_depth_path"), depth_path},
        {QStringLiteral("valid_mask_path"), mask_path},
        {QStringLiteral("pyramid_levels"),
         QJsonArray{QJsonObject{{QStringLiteral("level"), 2},
                                {QStringLiteral("valid_coverage"), 0.75}}}}};
    xjw::gui::widgets::DepthOverlayController controller;
    controller.setProjectMetadata(
        QJsonObject{{QStringLiteral("depth_map_results"), QJsonArray{record}}});

    EXPECT_TRUE(controller.anyArtifactAvailable(QStringLiteral("E:/images/a.png")));
    EXPECT_TRUE(controller.artifactAvailable(
        QStringLiteral("E:/images/a.png"), xjw::gui::views::DepthOverlayLevel::Final));
    EXPECT_FALSE(controller.artifactAvailable(
        QStringLiteral("E:/images/a.png"), xjw::gui::views::DepthOverlayLevel::Level2));
}

TEST(DepthOverlayDataTest, RejectsArtifactWithoutRequiredValidMask)
{
    const QJsonObject metadata{
        {QStringLiteral("depth_map_results"),
         QJsonArray{QJsonObject{{QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
                                {QStringLiteral("raw_depth_path"), QStringLiteral("a.bin")}}}}};

    EXPECT_FALSE(xjw::gui::views::resolveDepthOverlayArtifact(
        metadata, QStringLiteral("E:/images/a.png"), xjw::gui::views::DepthOverlayLevel::Final));
}

TEST(DepthOverlayDataTest, ResolvesNewestMatchingDepthRecord)
{
    const QJsonObject old_record{{QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
                                 {QStringLiteral("raw_depth_path"), QStringLiteral("old.bin")},
                                 {QStringLiteral("valid_mask_path"), QStringLiteral("old_mask.png")}};
    const QJsonObject new_record{{QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
                                 {QStringLiteral("raw_depth_path"), QStringLiteral("new.bin")},
                                 {QStringLiteral("valid_mask_path"), QStringLiteral("new_mask.png")}};
    const QJsonObject metadata{
        {QStringLiteral("depth_map_results"), QJsonArray{old_record, new_record}}};

    const auto artifact = xjw::gui::views::resolveDepthOverlayArtifact(
        metadata, QStringLiteral("E:/images/a.png"), xjw::gui::views::DepthOverlayLevel::Final);

    ASSERT_TRUE(artifact);
    EXPECT_EQ(artifact->rawDepthPath, QStringLiteral("new.bin"));
}

TEST(DepthOverlayDataTest, ColorizationMakesInvalidPixelsTransparent)
{
    const cv::Mat depth = (cv::Mat_<float>(2, 3) << 1.0f, 2.0f, 0.0f,
                                                    3.0f, std::nanf(""), 1000.0f);
    const cv::Mat valid = (cv::Mat_<uchar>(2, 3) << 255, 255, 255,
                                                    255, 255, 0);

    const QImage overlay = xjw::gui::views::colorizeDepthOverlay(depth, valid, 150);

    ASSERT_EQ(overlay.size(), QSize(3, 2));
    EXPECT_EQ(qAlpha(overlay.pixel(2, 0)), 0);
    EXPECT_EQ(qAlpha(overlay.pixel(1, 1)), 0);
    EXPECT_EQ(qAlpha(overlay.pixel(2, 1)), 0);
    EXPECT_EQ(qAlpha(overlay.pixel(0, 0)), 150);
}

TEST(DepthOverlayDataTest, ColorizationAcceptsOnlyFullyValidMaskPixels)
{
    const cv::Mat depth = (cv::Mat_<float>(1, 3) << 1.0f, 2.0f, 3.0f);
    const cv::Mat valid = (cv::Mat_<uchar>(1, 3) << 255, 128, 1);

    const QImage overlay = xjw::gui::views::colorizeDepthOverlay(depth, valid, 190);

    ASSERT_EQ(overlay.size(), QSize(3, 1));
    EXPECT_EQ(qAlpha(overlay.pixel(0, 0)), 190);
    EXPECT_EQ(qAlpha(overlay.pixel(1, 0)), 0);
    EXPECT_EQ(qAlpha(overlay.pixel(2, 0)), 0);
}

TEST(DepthOverlayDataTest, ColorizationUsesRobustP2P98Range)
{
    cv::Mat depth(1, 101, CV_32FC1);
    for (int column = 0; column < 100; ++column)
    {
        depth.at<float>(0, column) = static_cast<float>(column + 1);
    }
    depth.at<float>(0, 100) = 10000.0f;
    const cv::Mat valid(1, depth.cols, CV_8UC1, cv::Scalar(255));

    const QImage overlay = xjw::gui::views::colorizeDepthOverlay(depth, valid, 255);

    ASSERT_FALSE(overlay.isNull());
    EXPECT_EQ(overlay.pixel(0, 0), overlay.pixel(2, 0));
    EXPECT_EQ(overlay.pixel(98, 0), overlay.pixel(100, 0));
    EXPECT_NE(overlay.pixel(50, 0), overlay.pixel(2, 0));
    EXPECT_NE(overlay.pixel(50, 0), overlay.pixel(98, 0));
}

TEST(DepthOverlayDataTest, ColorizationUsesRedNearBlueFarAndWritesRgbaChannelsInQtOrder)
{
    cv::Mat depth(1, 100, CV_32FC1);
    for (int column = 0; column < depth.cols; ++column)
    {
        depth.at<float>(0, column) = static_cast<float>(column + 1);
    }
    const cv::Mat valid(depth.size(), CV_8UC1, cv::Scalar(255));

    const QImage overlay = xjw::gui::views::colorizeDepthOverlay(depth, valid, 123);

    ASSERT_FALSE(overlay.isNull());
    const QColor near_color = overlay.pixelColor(0, 0);
    EXPECT_EQ(near_color.red(), 122);
    EXPECT_EQ(near_color.green(), 4);
    EXPECT_EQ(near_color.blue(), 3);
    EXPECT_EQ(near_color.alpha(), 123);
    EXPECT_GT(near_color.red(), near_color.blue());

    const QColor far_color = overlay.pixelColor(depth.cols - 1, 0);
    EXPECT_EQ(far_color.red(), 48);
    EXPECT_EQ(far_color.green(), 18);
    EXPECT_EQ(far_color.blue(), 59);
    EXPECT_EQ(far_color.alpha(), 123);
    EXPECT_GT(far_color.blue(), far_color.red());
}

TEST(DepthOverlayDataTest, LargePeriodicDepthDoesNotBiasPercentileSampling)
{
    cv::Mat depth(1024, 2048, CV_32FC1);
    for (int row = 0; row < depth.rows; ++row)
    {
        float *depth_row = depth.ptr<float>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            depth_row[column] = column % 2 == 0 ? 1.0f : 100.0f;
        }
    }
    const cv::Mat valid(depth.size(), CV_8UC1, cv::Scalar(255));

    const QImage overlay = xjw::gui::views::colorizeDepthOverlay(depth, valid, 255);

    ASSERT_FALSE(overlay.isNull());
    EXPECT_NE(overlay.pixel(0, 0), overlay.pixel(1, 0));
}

TEST(DepthOverlayDataTest, RenderReportsAllInvalidDepth)
{
    const cv::Mat depth = (cv::Mat_<float>(1, 3) << 0.0f, -1.0f, std::nanf(""));
    const cv::Mat valid(1, 3, CV_8UC1, cv::Scalar(255));
    const xjw::gui::views::DepthOverlayRenderOptions options;

    const auto result = xjw::gui::views::renderDepthOverlay(depth, valid, options);

    EXPECT_TRUE(result.overlay.isNull());
    EXPECT_TRUE(result.intensityBase.isNull());
    EXPECT_FALSE(result.errorMessage.isEmpty());
}

TEST(DepthOverlayDataTest, IntensityUsesSourceImageAndMvsValidMask)
{
    const cv::Mat depth = (cv::Mat_<float>(1, 2) << 1.0f, 2.0f);
    const cv::Mat valid = (cv::Mat_<uchar>(1, 2) << 255, 0);
    QImage source(2, 1, QImage::Format_RGB888);
    source.setPixelColor(0, 0, QColor(255, 0, 0));
    source.setPixelColor(1, 0, QColor(0, 255, 0));
    xjw::gui::views::DepthOverlayRenderOptions options;
    options.opacity = 200;
    options.showIntensity = true;

    const auto result = xjw::gui::views::renderDepthOverlay(depth, valid, options, source);

    ASSERT_TRUE(result.errorMessage.isEmpty()) << qPrintable(result.errorMessage);
    ASSERT_EQ(result.overlay.size(), QSize(2, 1));
    ASSERT_EQ(result.intensityBase.size(), QSize(2, 1));
    EXPECT_EQ(qAlpha(result.overlay.pixel(0, 0)), 200);
    EXPECT_EQ(qAlpha(result.overlay.pixel(1, 0)), 0);
    EXPECT_GT(qGray(result.intensityBase.pixel(0, 0)), 0);
    EXPECT_EQ(qGray(result.intensityBase.pixel(1, 0)), 0);
}

TEST(DepthOverlayDataTest, LoadsRawDepthStorageAndValidMask)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString depth_path = directory.filePath(QStringLiteral("depth.bin"));
    const QString mask_path = directory.filePath(QStringLiteral("valid.png"));
    const cv::Mat depth = (cv::Mat_<float>(1, 2) << 4.0f, 8.0f);
    const cv::Mat valid = (cv::Mat_<uchar>(1, 2) << 255, 0);
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(depth_path, depth).ok);
    ASSERT_TRUE(cv::imwrite(mask_path.toStdString(), valid));

    xjw::gui::views::DepthOverlayArtifact artifact;
    artifact.rawDepthPath = depth_path;
    artifact.validMaskPath = mask_path;
    const xjw::gui::views::DepthOverlayRenderOptions options;
    const auto result = xjw::gui::views::loadDepthOverlay(artifact, options);

    ASSERT_TRUE(result.errorMessage.isEmpty()) << qPrintable(result.errorMessage);
    ASSERT_EQ(result.overlay.size(), QSize(2, 1));
    EXPECT_EQ(qAlpha(result.overlay.pixel(0, 0)), options.opacity);
    EXPECT_EQ(qAlpha(result.overlay.pixel(1, 0)), 0);
}

TEST(DepthOverlayDataTest, LoadsDepthArtifactsFromUnicodePaths)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString unicode_directory = directory.filePath(QStringLiteral("深度图"));
    ASSERT_TRUE(QDir().mkpath(unicode_directory));
    const QString depth_path = QDir(unicode_directory).filePath(QStringLiteral("深度.bin"));
    const QString mask_path = QDir(unicode_directory).filePath(QStringLiteral("有效蒙版.png"));
    const cv::Mat depth = (cv::Mat_<float>(1, 2) << 4.0f, 8.0f);
    const cv::Mat valid = (cv::Mat_<uchar>(1, 2) << 255, 0);
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(depth_path, depth).ok);
    ASSERT_TRUE(xjw::common::io::writeImage(mask_path, valid));

    xjw::gui::views::DepthOverlayArtifact artifact;
    artifact.rawDepthPath = depth_path;
    artifact.validMaskPath = mask_path;
    const auto result = xjw::gui::views::loadDepthOverlay(
        artifact, xjw::gui::views::DepthOverlayRenderOptions{});

    ASSERT_TRUE(result.errorMessage.isEmpty()) << qPrintable(result.errorMessage);
    ASSERT_EQ(result.overlay.size(), QSize(2, 1));
    EXPECT_EQ(qAlpha(result.overlay.pixel(0, 0)), 150);
    EXPECT_EQ(qAlpha(result.overlay.pixel(1, 0)), 0);
}

TEST(DepthOverlayDataTest, ExplainsResolutionLimitedLevelThree)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = directory.filePath(QStringLiteral("image.png"));
    const QString depth_path = directory.filePath(QStringLiteral("depth.bin"));
    const QString mask_path = directory.filePath(QStringLiteral("mask.png"));
    ASSERT_TRUE(QImage(8, 6, QImage::Format_RGB32).save(image_path));
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
        depth_path, cv::Mat(6, 8, CV_32FC1, cv::Scalar(2.0f))).ok);
    ASSERT_TRUE(cv::imwrite(mask_path.toStdString(),
                           cv::Mat(6, 8, CV_8UC1, cv::Scalar(255))));

    const QJsonObject level_1{
        {QStringLiteral("level"), 1},
        {QStringLiteral("raw_depth_path"), depth_path},
        {QStringLiteral("valid_mask_path"), mask_path}
    };
    const QJsonObject level_2{
        {QStringLiteral("level"), 2},
        {QStringLiteral("raw_depth_path"), depth_path},
        {QStringLiteral("valid_mask_path"), mask_path}
    };
    const QJsonObject record{
        {QStringLiteral("ref_image"), image_path},
        {QStringLiteral("raw_depth_path"), depth_path},
        {QStringLiteral("valid_mask_path"), mask_path},
        {QStringLiteral("grid_width"), 640},
        {QStringLiteral("grid_height"), 480},
        {QStringLiteral("pyramid_active_level_count"), 2},
        {QStringLiteral("pyramid_minimum_short_side"), 160},
        {QStringLiteral("pyramid_levels"), QJsonArray{level_1, level_2}}
    };
    const QJsonObject metadata{
        {QStringLiteral("depth_map_results"), QJsonArray{record}}
    };

    const auto level_2_status = xjw::gui::views::resolveDepthOverlayAvailability(
        metadata, image_path, xjw::gui::views::DepthOverlayLevel::Level2);
    const auto level_3_status = xjw::gui::views::resolveDepthOverlayAvailability(
        metadata, image_path, xjw::gui::views::DepthOverlayLevel::Level3);

    EXPECT_TRUE(level_2_status.available);
    EXPECT_FALSE(level_3_status.available);
    EXPECT_EQ(level_3_status.code,
              xjw::gui::views::DepthOverlayAvailabilityCode::NotComputedForResolution);
    EXPECT_TRUE(level_3_status.reason.contains(QStringLiteral("640×480")));
    EXPECT_TRUE(level_3_status.reason.contains(QStringLiteral("只生成 2 层")));
    EXPECT_TRUE(level_3_status.reason.contains(QStringLiteral("160")));
}

TEST(DepthOverlayDataTest, DistinguishesMissingArtifactFromUncomputedLevel)
{
    const QString image_path = QStringLiteral("E:/images/image.png");
    const QJsonObject level_2{
        {QStringLiteral("level"), 2},
        {QStringLiteral("raw_depth_path"), QStringLiteral("E:/missing/depth.bin")},
        {QStringLiteral("valid_mask_path"), QStringLiteral("E:/missing/mask.png")}
    };
    const QJsonObject record{
        {QStringLiteral("ref_image"), image_path},
        {QStringLiteral("pyramid_active_level_count"), 2},
        {QStringLiteral("pyramid_levels"), QJsonArray{level_2}}
    };

    const auto status = xjw::gui::views::resolveDepthOverlayAvailability(
        QJsonObject{{QStringLiteral("depth_map_results"), QJsonArray{record}}},
        image_path,
        xjw::gui::views::DepthOverlayLevel::Level2);

    EXPECT_FALSE(status.available);
    EXPECT_EQ(status.code,
              xjw::gui::views::DepthOverlayAvailabilityCode::ArtifactMissing);
    EXPECT_TRUE(status.reason.contains(QStringLiteral("不存在")));
}

TEST(DepthOverlayControllerTest, NewRequestInvalidatesEarlierGeneration)
{
    const QJsonObject record{{QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
                             {QStringLiteral("raw_depth_path"), QStringLiteral("final.bin")},
                             {QStringLiteral("valid_mask_path"), QStringLiteral("final_mask.png")}};
    xjw::gui::widgets::DepthOverlayController controller;
    controller.setProjectMetadata(
        QJsonObject{{QStringLiteral("depth_map_results"), QJsonArray{record}}});

    controller.request(QStringLiteral("E:/images/a.png"),
                       xjw::gui::views::DepthOverlayLevel::Final,
                       xjw::gui::views::DepthOverlayRenderOptions{});
    const quint64 first_generation = controller.requestGeneration();
    controller.request(QStringLiteral("E:/images/a.png"),
                       xjw::gui::views::DepthOverlayLevel::Level1,
                       xjw::gui::views::DepthOverlayRenderOptions{});
    const quint64 second_generation = controller.requestGeneration();

    EXPECT_GT(second_generation, first_generation);
    EXPECT_FALSE(controller.isCurrentGeneration(first_generation));
    EXPECT_TRUE(controller.isCurrentGeneration(second_generation));
}

TEST(DepthOverlayControllerTest, CacheKeyIncludesLevelOpacityAndIntensity)
{
    xjw::gui::views::DepthOverlayArtifact artifact;
    artifact.rawDepthPath = QStringLiteral("E:/depth/final.bin");
    artifact.validMaskPath = QStringLiteral("E:/depth/final_mask.png");
    artifact.level = 0;
    xjw::gui::views::DepthOverlayRenderOptions base_options;
    const QString base_key = xjw::gui::widgets::DepthOverlayController::cacheKeyForArtifact(
        artifact, xjw::gui::views::DepthOverlayLevel::Final, base_options,
        QStringLiteral("E:/images/a.png"));

    auto opacity_options = base_options;
    opacity_options.opacity = 220;
    auto intensity_options = base_options;
    intensity_options.showIntensity = true;

    EXPECT_NE(base_key, xjw::gui::widgets::DepthOverlayController::cacheKeyForArtifact(
                            artifact, xjw::gui::views::DepthOverlayLevel::Level1, base_options,
                            QStringLiteral("E:/images/a.png")));
    EXPECT_NE(base_key, xjw::gui::widgets::DepthOverlayController::cacheKeyForArtifact(
                            artifact, xjw::gui::views::DepthOverlayLevel::Final, opacity_options,
                            QStringLiteral("E:/images/a.png")));
    EXPECT_NE(base_key, xjw::gui::widgets::DepthOverlayController::cacheKeyForArtifact(
                            artifact, xjw::gui::views::DepthOverlayLevel::Final, intensity_options,
                            QStringLiteral("E:/images/a.png")));
}

TEST(LayerRendererDepthOverlayTest, KeepsDepthSeparateFromBaseImageAndClearsItSafely)
{
    QGraphicsScene scene;
    LayerRenderer renderer(&scene);
    const QImage base(8, 6, QImage::Format_RGB32);
    const QImage overlay(4, 3, QImage::Format_RGBA8888);

    ASSERT_TRUE(renderer.addImageLayer(base, 0));
    ASSERT_TRUE(renderer.setDepthOverlay(overlay, {}, 10));
    EXPECT_TRUE(renderer.hasDepthOverlay());

    bool found_base = false;
    bool found_depth = false;
    for (QGraphicsItem *item : scene.items())
    {
        found_base = found_base || qFuzzyCompare(item->zValue(), 0.0);
        found_depth = found_depth || qFuzzyCompare(item->zValue(), 10.0);
    }
    EXPECT_TRUE(found_base);
    EXPECT_TRUE(found_depth);

    renderer.clearDepthOverlay();
    EXPECT_FALSE(renderer.hasDepthOverlay());
    EXPECT_EQ(scene.items().size(), 1);
}

TEST(CanvasDepthOverlayTest, KeepsSelectedModeWhenTheDisplayedImageChanges)
{
    CanvasWidget canvas;
    canvas.setDepthOverlayEnabled(true);
    canvas.setDepthOverlayLevel(xjw::gui::views::DepthOverlayLevel::Level2);
    canvas.setDepthIntensityVisible(true);

    canvas.showImage(QString());

    EXPECT_TRUE(canvas.depthOverlayEnabled());
    EXPECT_EQ(canvas.depthOverlayLevel(), xjw::gui::views::DepthOverlayLevel::Level2);
    EXPECT_TRUE(canvas.depthIntensityVisible());
}

TEST(CanvasDepthOverlayTest, SuppressesDiagnosticsWithoutChangingUserPreferences)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    const QString image_path = directory.filePath(QStringLiteral("image.png"));
    const QString depth_path = directory.filePath(QStringLiteral("depth_0.bin"));
    const QString mask_path = directory.filePath(QStringLiteral("depth_0_mask.png"));

    QImage image(16, 12, QImage::Format_RGB32);
    image.fill(Qt::darkGray);
    ASSERT_TRUE(image.save(image_path));

    const cv::Mat depth(12, 16, CV_32FC1, cv::Scalar(2.0f));
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(depth_path, depth).ok);
    ASSERT_TRUE(cv::imwrite(mask_path.toStdString(),
                           cv::Mat(12, 16, CV_8UC1, cv::Scalar(255))));

    const QJsonObject record{
        {QStringLiteral("ref_image"), image_path},
        {QStringLiteral("raw_depth_path"), depth_path},
        {QStringLiteral("valid_mask_path"), mask_path},
        {QStringLiteral("grid_width"), 16},
        {QStringLiteral("grid_height"), 12}
    };

    CanvasWidget canvas;
    canvas.setProperty("currentProjectPath", directory.filePath(QStringLiteral("test.plascan")));
    canvas.setProjectMetadata(QJsonObject{
        {QStringLiteral("depth_map_results"), QJsonArray{record}}
    });

    LayerRenderer::FeatureDisplayOptions options;
    options.showPoints = true;
    options.showResiduals = true;
    canvas.applyFeatureDisplayOptions(options);
    canvas.showImage(image_path);
    QTRY_VERIFY_WITH_TIMEOUT(canvas.hasDisplayImage(), 5000);

    canvas.setDepthOverlayEnabled(true);
    QTRY_VERIFY_WITH_TIMEOUT(canvas.depthOverlayVisible(), 5000);
    EXPECT_TRUE(canvas.showsInterestPoints());
    EXPECT_TRUE(canvas.showsFeatureResiduals());
    EXPECT_TRUE(canvas.featureDiagnosticsSuppressed());

    canvas.setDepthOverlayEnabled(false);
    EXPECT_FALSE(canvas.featureDiagnosticsSuppressed());
    EXPECT_TRUE(canvas.showsInterestPoints());
    EXPECT_TRUE(canvas.showsFeatureResiduals());
}

namespace {

xjw::Camera makeCamera(double cx, double cy, double cz,
                       double fu = 1200.0, double fv = 1200.0)
{
    xjw::Camera cam;
    cam.setIntrinsics(fu, fv, 512.0, 384.0);
    const std::array<double, 9> rotation = {1.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0,
                                            0.0, 0.0, 1.0};
    const std::array<double, 3> center = {cx, cy, cz};
    cam.setPose(rotation, center);
    return cam;
}

bool projectPoint(const xjw::Camera &camera,
                  const std::array<double, 3> &xyz,
                  double *u,
                  double *v)
{
    if (!u || !v)
    {
        return false;
    }

    const double worldPoint[3] = {xyz[0], xyz[1], xyz[2]};
    double uv[2] = {0.0, 0.0};
    if (!camera.projectWorldPoint(worldPoint, uv))
    {
        return false;
    }

    *u = uv[0];
    *v = uv[1];
    return true;
}

QMenu *findTopLevelMenuByTitle(QMenuBar *menuBar, const QString &title)
{
    if (!menuBar)
    {
        return nullptr;
    }

    for (QAction *action : menuBar->actions())
    {
        if (action && action->menu() && action->menu()->title() == title)
        {
            return action->menu();
        }
    }

    return nullptr;
}

QMenu *findSubMenuByTitle(QMenu *menu, const QString &title)
{
    if (!menu)
    {
        return nullptr;
    }

    for (QAction *action : menu->actions())
    {
        if (action && action->menu() && action->menu()->title() == title)
        {
            return action->menu();
        }
    }

    return nullptr;
}

QStringList directActionTexts(QMenu *menu)
{
    QStringList texts;
    if (!menu)
    {
        return texts;
    }

    for (QAction *action : menu->actions())
    {
        if (action && !action->isSeparator() && !action->menu())
        {
            texts.push_back(action->text());
        }
    }

    return texts;
}

QString runtimePythonRelativePath()
{
#ifdef Q_OS_WIN
    return QStringLiteral(".venv/Scripts/python.exe");
#else
    return QStringLiteral(".venv/bin/python");
#endif
}

QString createFakeRuntimePython(const QString &sourceRoot)
{
    const QString pythonPath = QDir(sourceRoot).filePath(runtimePythonRelativePath());
    QDir().mkpath(QFileInfo(pythonPath).absolutePath());
    QFile file(pythonPath);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write("python");
    file.close();
    return pythonPath;
}

class ScopedEnvVar
{
public:
    explicit ScopedEnvVar(const char *name)
        : _name(name)
        , _hadPrevious(qEnvironmentVariableIsSet(name))
        , _previous(qgetenv(name))
    {
        qunsetenv(_name);
    }

    ScopedEnvVar(const char *name, const QString &value)
        : _name(name)
        , _hadPrevious(qEnvironmentVariableIsSet(name))
        , _previous(qgetenv(name))
    {
        qputenv(_name, value.toUtf8());
    }

    ~ScopedEnvVar()
    {
        if (_hadPrevious)
        {
            qputenv(_name, _previous);
        }
        else
        {
            qunsetenv(_name);
        }
    }

private:
    const char *_name;
    bool _hadPrevious = false;
    QByteArray _previous;
};

void writeMinimalTsai(const QString &path, double fu, double cx)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out(&file);
    out << "VERSION_3\n";
    out << "PINHOLE\n";
    out << "TSAI\n";
    out << "fu = " << QString::number(fu, 'f', 8) << "\n";
    out << "fv = " << QString::number(fu, 'f', 8) << "\n";
    out << "cu = " << QString::number(cx, 'f', 8) << "\n";
    out << "cv = 500\n";
    out << "u_direction = 1 0 0\n";
    out << "v_direction = 0 1 0\n";
    out << "w_direction = 0 0 1\n";
    out << "pitch = 1\n";
    out << "k1 = -0.01\n";
    out << "k2 = 0.02\n";
    out << "k3 = -0.03\n";
    out << "p1 = 0.0001\n";
    out << "p2 = -0.0002\n";
    out << "C = 1 2 3\n";
    out << "R = 1 0 0 0 1 0 0 0 1\n";
}

void writeMinimalPointCloudPly(const QString &path,
                               const std::vector<std::array<double, 3>> &points)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out(&file);
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "end_header\n";
    for (const auto &point : points)
    {
        out << QString::number(point[0], 'f', 6) << ' '
            << QString::number(point[1], 'f', 6) << ' '
            << QString::number(point[2], 'f', 6) << '\n';
    }
}

void putU16Le(QByteArray *bytes, qsizetype offset, quint16 value)
{
    ASSERT_TRUE(bytes);
    ASSERT_LE(offset + static_cast<qsizetype>(sizeof(value)), bytes->size());
    qToLittleEndian<quint16>(value, reinterpret_cast<uchar *>(bytes->data() + offset));
}

void putU32Le(QByteArray *bytes, qsizetype offset, quint32 value)
{
    ASSERT_TRUE(bytes);
    ASSERT_LE(offset + static_cast<qsizetype>(sizeof(value)), bytes->size());
    qToLittleEndian<quint32>(value, reinterpret_cast<uchar *>(bytes->data() + offset));
}

void putI32Le(QByteArray *bytes, qsizetype offset, qint32 value)
{
    ASSERT_TRUE(bytes);
    ASSERT_LE(offset + static_cast<qsizetype>(sizeof(value)), bytes->size());
    qToLittleEndian<qint32>(value, reinterpret_cast<uchar *>(bytes->data() + offset));
}

void putF64Le(QByteArray *bytes, qsizetype offset, double value)
{
    ASSERT_TRUE(bytes);
    ASSERT_LE(offset + static_cast<qsizetype>(sizeof(value)), bytes->size());
    quint64 raw = 0;
    std::memcpy(&raw, &value, sizeof(value));
    qToLittleEndian<quint64>(raw, reinterpret_cast<uchar *>(bytes->data() + offset));
}

void writeMinimalPointCloudLas(const QString &path,
                               const std::vector<std::array<double, 3>> &points)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));

    constexpr qsizetype headerSize = 227;
    constexpr qsizetype pointRecordLength = 20;
    constexpr double scale = 0.001;

    QByteArray bytes(headerSize, '\0');
    bytes[0] = 'L';
    bytes[1] = 'A';
    bytes[2] = 'S';
    bytes[3] = 'F';
    bytes[24] = 1;
    bytes[25] = 2;
    putU16Le(&bytes, 94, static_cast<quint16>(headerSize));
    putU32Le(&bytes, 96, static_cast<quint32>(headerSize));
    putU32Le(&bytes, 100, 0);
    bytes[104] = 0;
    putU16Le(&bytes, 105, static_cast<quint16>(pointRecordLength));
    putU32Le(&bytes, 107, static_cast<quint32>(points.size()));
    putF64Le(&bytes, 131, scale);
    putF64Le(&bytes, 139, scale);
    putF64Le(&bytes, 147, scale);
    putF64Le(&bytes, 155, 0.0);
    putF64Le(&bytes, 163, 0.0);
    putF64Le(&bytes, 171, 0.0);

    for (const auto &point : points)
    {
        QByteArray record(pointRecordLength, '\0');
        putI32Le(&record, 0, static_cast<qint32>(std::llround(point[0] / scale)));
        putI32Le(&record, 4, static_cast<qint32>(std::llround(point[1] / scale)));
        putI32Le(&record, 8, static_cast<qint32>(std::llround(point[2] / scale)));
        bytes.append(record);
    }

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_EQ(file.write(bytes), bytes.size());
}

void writeMinimalColoredPointCloudPly(const QString &path,
                                      const std::vector<std::array<double, 3>> &points,
                                      const std::vector<std::array<int, 3>> &colors)
{
    ASSERT_EQ(points.size(), colors.size());
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out(&file);
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto &point = points[i];
        const auto &color = colors[i];
        out << QString::number(point[0], 'f', 6) << ' '
            << QString::number(point[1], 'f', 6) << ' '
            << QString::number(point[2], 'f', 6) << ' '
            << color[0] << ' '
            << color[1] << ' '
            << color[2] << '\n';
    }
}

void writeMinimalSparseSidecar(const QString &path,
                               const std::vector<int> &trackLens)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));

    QJsonArray points;
    for (int i = 0; i < static_cast<int>(trackLens.size()); ++i)
    {
        QJsonArray xyz;
        xyz.append(static_cast<double>(i));
        xyz.append(0.0);
        xyz.append(0.0);

        QJsonObject point;
        point[QStringLiteral("point_xyz")] = xyz;
        point[QStringLiteral("rms_reproj_px")] = 0.5;
        point[QStringLiteral("min_tri_angle_deg")] = 3.0;
        point[QStringLiteral("track_len")] = trackLens[static_cast<std::size_t>(i)];
        points.append(point);
    }

    QJsonObject root;
    root[QStringLiteral("quality_metrics_available")] = true;
    root[QStringLiteral("point_count")] = static_cast<int>(trackLens.size());
    root[QStringLiteral("points")] = points;

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(QJsonDocument(root).toJson());
    file.close();
}

TEST(ProjectCameraImportServiceTest, BatchImportRecordsActualTsaiSourceFileInCameraMeta)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("IMG_001.JPG"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("jpg");
    imageFile.close();

    const QString tsaiPath = QDir(tempDir.path()).filePath(QStringLiteral("cameras/IMG_001.tsai"));
    writeMinimalTsai(tsaiPath, 1234.0, 640.0);

    xjw::gui::project::BatchCameraImportResult result;
    const auto status = xjw::gui::project::buildBatchCameraImport(
        QFileInfo(tsaiPath).absolutePath(),
        QStringList{imagePath},
        &result);

    ASSERT_EQ(status, xjw::gui::project::BatchCameraImportStatus::Ok);
    ASSERT_EQ(result.cameraMetaByImage.size(), 1);
    const QJsonObject camera = result.cameraMetaByImage.constBegin().value();
    EXPECT_EQ(QDir::cleanPath(camera.value(QStringLiteral("source_file")).toString()),
              QDir::cleanPath(QFileInfo(tsaiPath).absoluteFilePath()));
    EXPECT_DOUBLE_EQ(camera.value(QStringLiteral("fu")).toDouble(), 1234.0);
    EXPECT_NEAR(camera.value(QStringLiteral("k1")).toDouble(), -0.01, 1e-12);
}

TEST(ProjectDataCameraMetadataTest, SetImageCamerasClearsLegacyTopLevelCameraFile)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("camera_meta.plascan"));
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("IMG_002.JPG"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("jpg");
    imageFile.close();

    ProjectData data;
    ASSERT_TRUE(data.createProject(projectPath, QStringLiteral("camera_meta")));
    ASSERT_TRUE(data.addImages(QStringList{imagePath}));

    QJsonObject core = data.coreFilesMeta();
    QJsonArray images = core.value(QStringLiteral("images")).toArray();
    ASSERT_EQ(images.size(), 1);
    QJsonObject imageObject = images.at(0).toObject();
    imageObject[QStringLiteral("camera_file")] = QStringLiteral("stale/old.tsai");
    images[0] = imageObject;
    core[QStringLiteral("images")] = images;
    data.updateMetadata(core, false);

    QJsonObject camera;
    camera[QStringLiteral("model")] = QStringLiteral("tsai");
    camera[QStringLiteral("source_file")] = QStringLiteral("fresh/new.tsai");
    camera[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    camera[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    camera[QStringLiteral("pitch")] = 1.0;
    camera[QStringLiteral("fu")] = 1111.0;
    camera[QStringLiteral("fv")] = 1111.0;
    camera[QStringLiteral("cu")] = 500.0;
    camera[QStringLiteral("cv")] = 400.0;
    camera[QStringLiteral("C")] = QJsonArray{0.0, 0.0, 0.0};
    camera[QStringLiteral("R")] = QJsonArray{1.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0,
                                             0.0, 0.0, 1.0};

    int updated = 0;
    QString error;
    ASSERT_TRUE(data.setImageCameras(QMap<QString, QJsonObject>{{imagePath, camera}}, &updated, &error))
        << error.toStdString();
    EXPECT_EQ(updated, 1);

    const QJsonObject updatedImage =
        data.coreFilesMeta().value(QStringLiteral("images")).toArray().at(0).toObject();
    EXPECT_FALSE(updatedImage.contains(QStringLiteral("camera_file")));
    EXPECT_EQ(updatedImage.value(QStringLiteral("camera")).toObject()
                  .value(QStringLiteral("source_file")).toString(),
              QStringLiteral("fresh/new.tsai"));
}

TEST(ProjectDataAsyncOpenTest, OpensProjectFromSnapshotAndAppliesResultsLater)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("async_open.plascan"));
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("IMG_003.JPG"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("jpg");
    imageFile.close();

    ProjectData source;
    ASSERT_TRUE(source.createProject(projectPath, QStringLiteral("async_open")));
    ASSERT_TRUE(source.addImages(QStringList{imagePath}));
    ASSERT_TRUE(source.appendResultRecord(
        QStringLiteral("ipmatch_results"),
        QJsonObject{{QStringLiteral("image0"), imagePath},
                    {QStringLiteral("image1"), imagePath},
                    {QStringLiteral("output"), QStringLiteral("matches/IMG_003.match")}},
        true));
    QString saveError;
    ASSERT_TRUE(source.saveProject(&saveError)) << saveError.toStdString();

    const ProjectOpenSnapshot openSnapshot = ProjectData::loadProjectOpenSnapshot(projectPath);
    ASSERT_TRUE(openSnapshot.success) << openSnapshot.errorMessage.toStdString();
    EXPECT_EQ(openSnapshot.projectPath, projectPath);
    EXPECT_EQ(openSnapshot.filesMeta.value(QStringLiteral("images")).toArray().size(), 1);

    ProjectData reopened;
    QString error;
    ASSERT_TRUE(reopened.openProjectFromSnapshot(openSnapshot, &error)) << error.toStdString();
    EXPECT_EQ(reopened.currentProjectPath(), projectPath);
    EXPECT_EQ(reopened.coreFilesMeta().value(QStringLiteral("images")).toArray().size(), 1);

    const ProjectResultsSnapshot resultsSnapshot = ProjectData::loadProjectResultsSnapshot(projectPath);
    ASSERT_TRUE(resultsSnapshot.success) << resultsSnapshot.errorMessage.toStdString();
    ASSERT_TRUE(reopened.applyResultsSnapshot(resultsSnapshot, &error)) << error.toStdString();
    EXPECT_EQ(reopened.metadata().value(QStringLiteral("ipmatch_results")).toArray().size(), 1);
}

QJsonObject buildImageEntry(const QString &path, const xjw::Camera &camera)
{
    QJsonObject imageObject;
    imageObject[QStringLiteral("path")] = path;
    imageObject[QStringLiteral("camera")] = xjw::common::project::cameraToJson(camera);
    return imageObject;
}

QLineEdit *findLineEditByPlaceholder(QWidget *root, const QString &text)
{
    const QList<QLineEdit *> edits = root->findChildren<QLineEdit *>();
    for (QLineEdit *edit : edits)
    {
        if (edit->placeholderText().contains(text))
        {
            return edit;
        }
    }
    return nullptr;
}

QLineEdit *findModelPathEdit(QWidget *root)
{
    return findLineEditByPlaceholder(root, QStringLiteral("模型路径"));
}

QToolButton *findToolButton(QWidget *root, const QString &text)
{
    const QList<QToolButton *> buttons = root->findChildren<QToolButton *>();
    for (QToolButton *button : buttons)
    {
        if (button->text() == text)
        {
            return button;
        }
    }
    return nullptr;
}

QLabel *findLabelContaining(QWidget *root, const QString &text)
{
    const QList<QLabel *> labels = root->findChildren<QLabel *>();
    for (QLabel *label : labels)
    {
        if (label->text().contains(text))
        {
            return label;
        }
    }
    return nullptr;
}

QString readProjectSourceFile(const QString &relativePath)
{
    const QString projectRoot = QFileInfo(QStringLiteral(TEST_DATA_DIR)).absoluteDir().absolutePath();
    QFile file(QDir(projectRoot).filePath(relativePath));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

QString readIncrementalSfmProjectImplementation()
{
    return readProjectSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfm.cpp")) +
           readProjectSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfmDetail.cpp")) +
           readProjectSourceFile(QStringLiteral("src/core/sfm/pipeline/InitialPairInitializer.cpp")) +
           readProjectSourceFile(QStringLiteral("src/core/sfm/pipeline/ImageRegistrationEngine.cpp")) +
           readProjectSourceFile(QStringLiteral("src/core/sfm/pipeline/KnownPoseReconstructor.cpp")) +
           readProjectSourceFile(QStringLiteral("src/core/sfm/pipeline/SfmBundleAdjustCoordinator.cpp"));
}

QJsonArray productionSparsePoints()
{
    return QJsonArray{
        QJsonObject{{QStringLiteral("track_len"), 2},
                    {QStringLiteral("rms_reproj_px"), 0.8}},
        QJsonObject{{QStringLiteral("track_len"), 3},
                    {QStringLiteral("rms_reproj_px"), 0.6}},
        QJsonObject{{QStringLiteral("track_len"), 4},
                    {QStringLiteral("rms_reproj_px"), 0.7}}
    };
}

QJsonObject sparseResultRecord(int index,
                               const QString &displayName,
                               const QString &operation,
                               const QString &operationDisplayName,
                               int sparsePointCount,
                               const QJsonObject &quality)
{
    QJsonObject record{
        {QStringLiteral("index"), index},
        {QStringLiteral("display_name"), displayName},
        {QStringLiteral("operation"), operation},
        {QStringLiteral("operation_display_name"), operationDisplayName},
        {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("E:/tmp/%1.xyz").arg(displayName)},
        {QStringLiteral("sparse_point_count"), sparsePointCount}
    };
    return xjw::gui::project::mergeSparseQualityIntoRecord(record, quality);
}

} // namespace

TEST(ProjectSupportUtilsTest, CollectMatchedPairsUsesFilenameWithSuffix)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString matchesDir = xjw::common::project::ProjectIO::ipmatchOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(matchesDir));

    QFile matchFile(QDir(matchesDir).filePath(QStringLiteral("1__2.match")));
    ASSERT_TRUE(matchFile.open(QIODevice::WriteOnly));
    matchFile.write("dummy");
    matchFile.close();

    QFile sidecarFile(QDir(matchesDir).filePath(QStringLiteral("1__2.match.json")));
    ASSERT_TRUE(sidecarFile.open(QIODevice::WriteOnly));
    sidecarFile.write(QJsonDocument(QJsonObject{{QStringLiteral("image0_name"), QStringLiteral("1")},
                                                {QStringLiteral("image1_name"), QStringLiteral("2")}}).toJson());
    sidecarFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/1.jpg")}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/2.png")}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/3.tif")}});

    QJsonArray ipmatchResults;
    ipmatchResults.append(QJsonObject{{QStringLiteral("image0"), QStringLiteral("/tmp/1.jpg")},
                                      {QStringLiteral("image1"), QStringLiteral("/tmp/3.tif")}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    meta[QStringLiteral("ipmatch_results")] = ipmatchResults;

    const QVector<QPair<QString, QString>> pairs =
        xjw::common::project::collectMatchedImageNamePairs(projectPath, meta);

    EXPECT_TRUE(pairs.contains(qMakePair(QStringLiteral("1.jpg"), QStringLiteral("2.png"))));
    EXPECT_TRUE(pairs.contains(qMakePair(QStringLiteral("1.jpg"), QStringLiteral("3.tif"))));
}

TEST(ProjectSupportUtilsTest, CollectSettledNoMatchPairsUsesFilenameWithSuffix)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString matchesDir = xjw::common::project::ProjectIO::ipmatchOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(matchesDir));

    QJsonArray noMatchPairs;
    noMatchPairs.append(QJsonObject{{QStringLiteral("image0"), QStringLiteral("/tmp/1.jpg")},
                                    {QStringLiteral("image1"), QStringLiteral("/tmp/2.png")},
                                    {QStringLiteral("feature_algorithm"), QStringLiteral("disk")},
                                    {QStringLiteral("match_algorithm"), QStringLiteral("lightglue")}});
    noMatchPairs.append(QJsonObject{{QStringLiteral("image0"), QStringLiteral("3")},
                                    {QStringLiteral("image1"), QStringLiteral("4")},
                                    {QStringLiteral("feature_algorithm"), QStringLiteral("disk")},
                                    {QStringLiteral("match_algorithm"), QStringLiteral("lightglue")}});

    QFile noMatchFile(QDir(matchesDir).filePath(QStringLiteral("no_match_pairs.json")));
    ASSERT_TRUE(noMatchFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    noMatchFile.write(QJsonDocument(noMatchPairs).toJson(QJsonDocument::Compact));
    noMatchFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/1.jpg")}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/2.png")}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/3.tif")}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/4.tif")}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;

    const QVector<QPair<QString, QString>> pairs =
        xjw::common::project::collectSettledNoMatchImageNamePairs(projectPath, meta);

    EXPECT_TRUE(pairs.contains(qMakePair(QStringLiteral("1.jpg"), QStringLiteral("2.png"))));
    EXPECT_TRUE(pairs.contains(qMakePair(QStringLiteral("3.tif"), QStringLiteral("4.tif"))));
}

TEST(ProjectDashboardSummaryTest, EmptyMetadataShowsMissingReadOnlyWorkflow)
{
    const QJsonObject meta;
    const auto summary = xjw::gui::project::buildProjectDashboardSummary(meta);

    EXPECT_EQ(summary.imageCount, 0);
    EXPECT_EQ(summary.cameraCount, 0);
    EXPECT_EQ(summary.reportResultCount, 0);
    EXPECT_EQ(summary.referenceDatasetCount, 0);
    EXPECT_GE(summary.workflowSteps.size(), 8);

    xjw::gui::project::ProjectDashboardStep step;
    ASSERT_TRUE(xjw::gui::project::projectDashboardStepById(summary, QStringLiteral("images"), &step));
    EXPECT_EQ(step.state, xjw::gui::project::ProjectDashboardStepState::Missing);
    EXPECT_TRUE(step.detail.contains(QStringLiteral("导入")));

    ASSERT_TRUE(xjw::gui::project::projectDashboardStepById(summary, QStringLiteral("reference_lidar"), &step));
    EXPECT_EQ(step.state, xjw::gui::project::ProjectDashboardStepState::Missing);
}

TEST(ProjectDashboardSummaryTest, SummarizesWorkflowReportsAndReferenceDatasets)
{
    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/img_001.tif")},
                              {QStringLiteral("camera"), QJsonObject{{QStringLiteral("fu"), 1000.0}}}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/img_002.tif")},
                              {QStringLiteral("camera"), QJsonObject{{QStringLiteral("fu"), 1000.0}}}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/img_003.tif")}});

    QJsonObject meta;
    meta[QStringLiteral("project_files")] = QJsonObject{{QStringLiteral("images"), images}};
    meta[QStringLiteral("ipfind_results")] = QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("ip.json")}}};
    meta[QStringLiteral("ipmatch_results")] = QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("matches.json")}}};
    meta[QStringLiteral("aerial_triangulation_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("sparse_point_count"), 1200}}};
    meta[QStringLiteral("bundle_adjust_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("mean_rms_after"), 0.8}}};
    meta[QStringLiteral("depth_map_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("result_type"), QStringLiteral("mvs_depth")}},
                   QJsonObject{{QStringLiteral("result_type"), QStringLiteral("legacy_preview")}}};
    meta[QStringLiteral("dense_cloud_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("point_count"), 54000}}};
    meta[QStringLiteral("model_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("model_ply"), QStringLiteral("mesh.ply")}}};
    meta[QStringLiteral("dem_results")] = QJsonArray{QJsonObject{{QStringLiteral("dem_tif"), QStringLiteral("dem.tif")}}};
    meta[QStringLiteral("ortho_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("output_path"), QStringLiteral("dom.tif")}}};
    meta[QStringLiteral("reference_datasets")] =
        QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("scan.las")},
                               {QStringLiteral("type"), QStringLiteral("lidar")},
                               {QStringLiteral("role"), QStringLiteral("ba_prior")}},
                   QJsonObject{{QStringLiteral("path"), QStringLiteral("check.xyz")},
                               {QStringLiteral("type"), QStringLiteral("point_cloud")},
                               {QStringLiteral("role"), QStringLiteral("validation")}},
                   QJsonObject{{QStringLiteral("path"), QStringLiteral("ref_dem.tif")},
                               {QStringLiteral("type"), QStringLiteral("dem")},
                               {QStringLiteral("role"), QStringLiteral("validation")}}};
    meta[QStringLiteral("report_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("reconstruction_quality")},
                               {QStringLiteral("path"), QStringLiteral("quality.json")}},
                   QJsonObject{{QStringLiteral("type"), QStringLiteral("reference_quality")},
                               {QStringLiteral("path"), QStringLiteral("reference_quality.json")}},
                   QJsonObject{{QStringLiteral("type"), QStringLiteral("other")},
                               {QStringLiteral("path"), QStringLiteral("other.json")}}};

    const auto summary = xjw::gui::project::buildProjectDashboardSummary(meta);

    EXPECT_EQ(summary.imageCount, 3);
    EXPECT_EQ(summary.cameraCount, 2);
    EXPECT_EQ(summary.featureResultCount, 1);
    EXPECT_EQ(summary.matchResultCount, 1);
    EXPECT_EQ(summary.sparseResultCount, 1);
    EXPECT_EQ(summary.bundleAdjustResultCount, 1);
    EXPECT_EQ(summary.depthMapResultCount, 1);
    EXPECT_EQ(summary.denseCloudResultCount, 1);
    EXPECT_EQ(summary.modelResultCount, 1);
    EXPECT_EQ(summary.demResultCount, 1);
    EXPECT_EQ(summary.orthoResultCount, 1);
    EXPECT_EQ(summary.referenceDatasetCount, 3);
    EXPECT_EQ(summary.lidarReferenceCount, 1);
    EXPECT_EQ(summary.pointCloudReferenceCount, 1);
    EXPECT_EQ(summary.baPriorReferenceCount, 1);
    EXPECT_EQ(summary.reportResultCount, 3);
    EXPECT_EQ(summary.qualityReportCount, 2);
    EXPECT_EQ(summary.qualityReports.size(), 2);

    xjw::gui::project::ProjectDashboardStep step;
    ASSERT_TRUE(xjw::gui::project::projectDashboardStepById(summary, QStringLiteral("sparse_ba"), &step));
    EXPECT_EQ(step.state, xjw::gui::project::ProjectDashboardStepState::Complete);
    EXPECT_TRUE(step.detail.contains(QStringLiteral("BA")));

    ASSERT_TRUE(xjw::gui::project::projectDashboardStepById(summary, QStringLiteral("reference_lidar"), &step));
    EXPECT_EQ(step.state, xjw::gui::project::ProjectDashboardStepState::Complete);
    EXPECT_TRUE(step.detail.contains(QStringLiteral("BA约束")));
}

TEST(ProjectDashboardSummaryTest, DoesNotMutateInputMetadata)
{
    QJsonObject meta;
    meta[QStringLiteral("images")] =
        QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/img_001.tif")}}};
    meta[QStringLiteral("reference_datasets")] =
        QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("scan.las")},
                               {QStringLiteral("type"), QStringLiteral("lidar")}}};
    const QByteArray before = QJsonDocument(meta).toJson(QJsonDocument::Compact);

    const auto summary = xjw::gui::project::buildProjectDashboardSummary(meta);

    EXPECT_EQ(summary.imageCount, 1);
    EXPECT_EQ(QJsonDocument(meta).toJson(QJsonDocument::Compact), before);
}

TEST(ProjectDashboardSummaryTest, IgnoresPointOnlyModelRecords)
{
    QJsonObject pointOnlyModelRecord;
    pointOnlyModelRecord[QStringLiteral("kind")] = QStringLiteral("mesh");
    pointOnlyModelRecord[QStringLiteral("model_ply")] = QStringLiteral("/tmp/products/model_from_mesh.ply");
    pointOnlyModelRecord[QStringLiteral("vertex_count")] = 1058511291;
    pointOnlyModelRecord[QStringLiteral("face_count")] = 0;

    QJsonObject validMeshRecord;
    validMeshRecord[QStringLiteral("kind")] = QStringLiteral("mesh");
    validMeshRecord[QStringLiteral("model_ply")] = QStringLiteral("/tmp/products/terrain_mesh.ply");
    validMeshRecord[QStringLiteral("vertex_count")] = 3286949;
    validMeshRecord[QStringLiteral("face_count")] = 6502504;

    QJsonObject meta;
    meta[QStringLiteral("model_results")] = QJsonArray{pointOnlyModelRecord, validMeshRecord};

    const auto summary = xjw::gui::project::buildProjectDashboardSummary(meta);

    EXPECT_EQ(summary.modelResultCount, 1);
}

TEST(ProjectResultRecordsTest, DenseCloudAndMeshRecordsKeepDistinctProductKinds)
{
    const QJsonObject dense = xjw::gui::project::makeDenseResultRecord(
        QStringLiteral("2026-06-28T00:00:00Z"),
        QStringLiteral("E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply"),
        1058511291,
        QStringLiteral("E:/code/test/agisoft_aerial_gcps/sparse_cloud.ply"));

    EXPECT_EQ(dense.value(QStringLiteral("kind")).toString(), QStringLiteral("dense_cloud"));
    EXPECT_EQ(dense.value(QStringLiteral("result_type")).toString(), QStringLiteral("dense_cloud"));
    EXPECT_EQ(dense.value(QStringLiteral("dense_cloud_xyz")).toString(),
              QStringLiteral("E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply"));
    EXPECT_EQ(dense.value(QStringLiteral("point_count")).toInt(), 1058511291);
    EXPECT_EQ(dense.value(QStringLiteral("face_count")).toInt(-1), 0);
    EXPECT_FALSE(dense.contains(QStringLiteral("model_ply")));

    const QJsonObject mesh = xjw::gui::project::makeModelResultRecord(
        QStringLiteral("2026-06-28T00:00:00Z"),
        QStringLiteral("mvs_dense_cloud_mesh"),
        QStringLiteral("E:/code/test/agisoft_aerial_gcps/mvs_output/products/model_from_mesh.ply"),
        3286949,
        6502504,
        QString(),
        QStringLiteral("E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply"));

    EXPECT_EQ(mesh.value(QStringLiteral("kind")).toString(), QStringLiteral("mesh"));
    EXPECT_EQ(mesh.value(QStringLiteral("result_type")).toString(), QStringLiteral("mesh"));
    EXPECT_EQ(mesh.value(QStringLiteral("model_ply")).toString(),
              QStringLiteral("E:/code/test/agisoft_aerial_gcps/mvs_output/products/model_from_mesh.ply"));
    EXPECT_EQ(mesh.value(QStringLiteral("vertex_count")).toInt(), 3286949);
    EXPECT_EQ(mesh.value(QStringLiteral("face_count")).toInt(), 6502504);
    EXPECT_EQ(mesh.value(QStringLiteral("source_dense_cloud")).toString(),
              QStringLiteral("E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply"));
    EXPECT_NE(mesh.value(QStringLiteral("model_ply")).toString(),
              mesh.value(QStringLiteral("source_dense_cloud")).toString());
}

TEST(ProjectSupportUtilsTest, CameraJsonRoundTripPreservesUnitsAndDepthDirection)
{
    xjw::Camera sourceCamera = makeCamera(1.25, -2.5, 3.75, 1200.0, 1195.0);
    sourceCamera.setAxisDirections(-1, 1);
    sourceCamera.setDepthAxisFlipped(true);
    sourceCamera.setDistortion(0.01, -0.001, 0.0001, 0.0002, -0.0003);

    const QJsonObject cameraJson = xjw::common::project::cameraToJson(sourceCamera);

    xjw::Camera restoredCamera;
    ASSERT_TRUE(xjw::common::project::cameraFromJson(cameraJson, &restoredCamera));

    EXPECT_DOUBLE_EQ(restoredCamera.focalX(), sourceCamera.focalX());
    EXPECT_DOUBLE_EQ(restoredCamera.focalY(), sourceCamera.focalY());
    EXPECT_DOUBLE_EQ(restoredCamera.principalX(), sourceCamera.principalX());
    EXPECT_DOUBLE_EQ(restoredCamera.principalY(), sourceCamera.principalY());
    EXPECT_EQ(restoredCamera.uAxisSign(), sourceCamera.uAxisSign());
    EXPECT_EQ(restoredCamera.vAxisSign(), sourceCamera.vAxisSign());
    EXPECT_EQ(restoredCamera.depthAxisFlipped(), sourceCamera.depthAxisFlipped());

    const auto sourceCenter = sourceCamera.cameraCenter();
    const auto restoredCenter = restoredCamera.cameraCenter();
    EXPECT_DOUBLE_EQ(restoredCenter[0], sourceCenter[0]);
    EXPECT_DOUBLE_EQ(restoredCenter[1], sourceCenter[1]);
    EXPECT_DOUBLE_EQ(restoredCenter[2], sourceCenter[2]);
}

TEST(ProjectSupportUtilsTest, ResolveFeaturePathFromTokenSupportsSuffix)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = xjw::common::project::ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("1.jpg"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    const QString spPath = QDir(ipDir).filePath(QStringLiteral("1.sp"));
    QFile spFile(spPath);
    ASSERT_TRUE(spFile.open(QIODevice::WriteOnly));
    spFile.write("sp");
    spFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), imagePath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;

    EXPECT_EQ(xjw::common::project::resolveProjectImagePathFromToken(QStringLiteral("1.jpg"), meta), imagePath);
    EXPECT_EQ(xjw::common::project::resolveProjectFeaturePathFromToken(projectPath, meta, QStringLiteral("1.jpg")), spPath);
    EXPECT_EQ(xjw::common::project::resolveProjectFeaturePathFromToken(projectPath, meta, imagePath), spPath);
}

TEST(ProjectSupportUtilsTest, InfersDiskFeatureSuffixWhenProjectHasOnlyDskOutputs)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = xjw::common::project::ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("66.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    const QString dskPath = QDir(ipDir).filePath(QStringLiteral("66.dsk"));
    QFile dskFile(dskPath);
    ASSERT_TRUE(dskFile.open(QIODevice::WriteOnly));
    dskFile.write("dsk");
    dskFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), imagePath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;

    EXPECT_EQ(xjw::common::project::inferPreferredFeatureSuffix(projectPath, meta),
              QStringLiteral(".dsk"));
}

TEST(ProjectSupportUtilsTest, InfersSiftFeatureSuffixBeforeLegacyDskOutputs)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = xjw::common::project::ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("66.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    for (const QString &suffix : {QStringLiteral(".dsk"), QStringLiteral(".sift")})
    {
        QFile featureFile(QDir(ipDir).filePath(QStringLiteral("66") + suffix));
        ASSERT_TRUE(featureFile.open(QIODevice::WriteOnly));
        featureFile.write("feature");
        featureFile.close();
    }

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), imagePath}}
    };

    EXPECT_EQ(xjw::common::project::inferPreferredFeatureSuffix(projectPath, meta),
              QStringLiteral(".sift"));

    const QStringList suffixes = xjw::common::project::projectFeatureSuffixes(projectPath, meta);
    ASSERT_FALSE(suffixes.isEmpty());
    EXPECT_EQ(suffixes.first(), QStringLiteral(".sift"));
}

TEST(ProjectSupportUtilsTest, ResolvesUnavailableLegacyDskSuffixToAvailableSiftFeature)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = xjw::common::project::ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("66.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    QFile siftFile(QDir(ipDir).filePath(QStringLiteral("66.sift")));
    ASSERT_TRUE(siftFile.open(QIODevice::WriteOnly));
    siftFile.write("sift");
    siftFile.close();

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), imagePath}}
    };

    EXPECT_EQ(xjw::common::project::resolvePreferredFeatureSuffix(projectPath,
                                                               meta,
                                                               QStringLiteral(".dsk")),
              QStringLiteral(".sift"));
}

TEST(ProjectSupportUtilsTest, DetectsWhetherProjectHasRequestedFeatureSuffix)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = xjw::common::project::ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("66.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    QFile dskFile(QDir(ipDir).filePath(QStringLiteral("66.dsk")));
    ASSERT_TRUE(dskFile.open(QIODevice::WriteOnly));
    dskFile.write("dsk");
    dskFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), imagePath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;

    EXPECT_TRUE(xjw::common::project::projectHasFeatureSuffix(projectPath, meta, QStringLiteral(".dsk")));
    EXPECT_FALSE(xjw::common::project::projectHasFeatureSuffix(projectPath, meta, QStringLiteral(".sp")));
}

TEST(ProjectSupportUtilsTest, ListsOnlyFeatureSuffixesPresentInProject)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = xjw::common::project::ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("75.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    QFile dskFile(QDir(ipDir).filePath(QStringLiteral("75.dsk")));
    ASSERT_TRUE(dskFile.open(QIODevice::WriteOnly));
    dskFile.write("dsk");
    dskFile.close();

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), imagePath}}
    };

    const QStringList suffixes = xjw::common::project::projectFeatureSuffixes(projectPath, meta);
    EXPECT_EQ(suffixes, QStringList{QStringLiteral(".dsk")});
}

TEST(DepthFrameUtilsTest, ExistingFrameArtifactsRequirePreviewAndRawDepth)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString pngPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_0.png"));
    EXPECT_FALSE(xjw::core::project::depthFrameArtifactsExist(pngPath));

    QFile pngFile(pngPath);
    ASSERT_TRUE(pngFile.open(QIODevice::WriteOnly));
    pngFile.write("png");
    pngFile.close();

    EXPECT_FALSE(xjw::core::project::depthFrameArtifactsExist(pngPath));

    QFile rawFile(xjw::core::project::rawDepthStoragePath(pngPath));
    ASSERT_TRUE(rawFile.open(QIODevice::WriteOnly));
    rawFile.write("raw");
    rawFile.close();

    EXPECT_TRUE(xjw::core::project::depthFrameArtifactsExist(pngPath));
    EXPECT_FALSE(xjw::core::project::depthFrameArtifactsExist(pngPath, true));

    QFile confidenceFile(xjw::core::project::rawConfidenceStoragePath(pngPath));
    ASSERT_TRUE(confidenceFile.open(QIODevice::WriteOnly));
    confidenceFile.write("confidence");
    confidenceFile.close();

    EXPECT_TRUE(xjw::core::project::depthFrameArtifactsExist(pngPath, true));
}

TEST(DepthFrameUtilsTest, FastBinaryDepthArtifactsIgnoreLegacyYamlFiles)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString pngPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_3.png"));
    EXPECT_TRUE(xjw::core::project::rawDepthStoragePath(pngPath).endsWith(QStringLiteral(".bin")));
    EXPECT_TRUE(xjw::core::project::rawConfidenceStoragePath(pngPath).endsWith(QStringLiteral("_conf.bin")));

    QFile pngFile(pngPath);
    ASSERT_TRUE(pngFile.open(QIODevice::WriteOnly));
    pngFile.write("png");
    pngFile.close();

    QFile legacyRawFile(QDir(tempDir.path()).filePath(QStringLiteral("depth_3.yml.gz")));
    ASSERT_TRUE(legacyRawFile.open(QIODevice::WriteOnly));
    legacyRawFile.write("legacy-depth");
    legacyRawFile.close();

    EXPECT_FALSE(xjw::core::project::depthFrameArtifactsExist(pngPath));
    EXPECT_FALSE(xjw::core::project::depthFrameArtifactsExist(pngPath, true));

    QFile legacyConfidenceFile(QDir(tempDir.path()).filePath(QStringLiteral("depth_3_conf.yml.gz")));
    ASSERT_TRUE(legacyConfidenceFile.open(QIODevice::WriteOnly));
    legacyConfidenceFile.write("legacy-confidence");
    legacyConfidenceFile.close();

    EXPECT_FALSE(xjw::core::project::depthFrameArtifactsExist(pngPath, true));

    QFile rawFile(xjw::core::project::rawDepthStoragePath(pngPath));
    ASSERT_TRUE(rawFile.open(QIODevice::WriteOnly));
    rawFile.write("raw");
    rawFile.close();

    EXPECT_TRUE(xjw::core::project::depthFrameArtifactsExist(pngPath));
    EXPECT_FALSE(xjw::core::project::depthFrameArtifactsExist(pngPath, true));

    QFile confidenceFile(xjw::core::project::rawConfidenceStoragePath(pngPath));
    ASSERT_TRUE(confidenceFile.open(QIODevice::WriteOnly));
    confidenceFile.write("confidence");
    confidenceFile.close();

    EXPECT_TRUE(xjw::core::project::depthFrameArtifactsExist(pngPath, true));
}

TEST(DepthFrameUtilsTest, StoredDepthCollectionRequiresCurrentBinaryDepth)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString pngPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_5.png"));
    QFile pngFile(pngPath);
    ASSERT_TRUE(pngFile.open(QIODevice::WriteOnly));
    pngFile.write("png");
    pngFile.close();

    const QString legacyRawPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_5.yml.gz"));
    QFile legacyRawFile(legacyRawPath);
    ASSERT_TRUE(legacyRawFile.open(QIODevice::WriteOnly));
    legacyRawFile.write("legacy-depth");
    legacyRawFile.close();

    QJsonObject record;
    record[QStringLiteral("ref_image")] = QStringLiteral("image_5.jpg");
    record[QStringLiteral("source_images")] =
        QJsonArray{QStringLiteral("image_4.jpg"), QStringLiteral("image_6.jpg")};
    record[QStringLiteral("depth_png")] = pngPath;
    record[QStringLiteral("raw_depth_path")] = xjw::core::project::rawDepthStoragePath(pngPath);

    QJsonObject meta;
    meta[QStringLiteral("depth_map_results")] = QJsonArray{record};

    const auto result = xjw::core::project::collectLatestStoredDepthFrames(meta);
    EXPECT_FALSE(result.status.ok);
    EXPECT_TRUE(result.frames.empty());

    QFile rawFile(xjw::core::project::rawDepthStoragePath(pngPath));
    ASSERT_TRUE(rawFile.open(QIODevice::WriteOnly));
    rawFile.write("raw");
    rawFile.close();

    const auto refreshedResult = xjw::core::project::collectLatestStoredDepthFrames(meta);
    ASSERT_TRUE(refreshedResult.status.ok) << refreshedResult.status.errorMessage.toStdString();
    ASSERT_EQ(refreshedResult.frames.size(), 1u);
    EXPECT_EQ(refreshedResult.frames.front().rawDepthPath, xjw::core::project::rawDepthStoragePath(pngPath));
    EXPECT_EQ(refreshedResult.frames.front().sourceImages,
              QStringList({QStringLiteral("image_4.jpg"), QStringLiteral("image_6.jpg")}));
}

TEST(DepthFrameUtilsTest, ResolvesPlannedFusionSourcesWithinStoredBatch)
{
    std::vector<xjw::core::project::StoredDepthFrameRecord> frames(3);
    frames[0].refImage = QStringLiteral("E:/images/ref.jpg");
    frames[0].sourceImages = {QStringLiteral("E:/images/right.jpg"),
                              QStringLiteral("E:/images/missing.jpg")};
    frames[1].refImage = QStringLiteral("E:/images/left.jpg");
    frames[2].refImage = QStringLiteral("E:/images/right.jpg");

    EXPECT_EQ(xjw::core::project::storedFusionSourceIndices(frames, 0),
              std::vector<int>({2}));
}

TEST(DepthFrameUtilsTest, StoredDepthCollectionCanSelectRequestedBatchDirectory)
{
    QTemporaryDir first_batch;
    QTemporaryDir second_batch;
    ASSERT_TRUE(first_batch.isValid());
    ASSERT_TRUE(second_batch.isValid());

    QJsonArray records;
    for (const auto &batch : {first_batch.path(), second_batch.path()})
    {
        for (int index = 0; index < 2; ++index)
        {
            const QString png_path = QDir(batch).filePath(QStringLiteral("depth_%1.png").arg(index));
            const QString raw_path = QDir(batch).filePath(QStringLiteral("depth_%1.bin").arg(index));
            for (const QString &path : {png_path, raw_path})
            {
                QFile file(path);
                ASSERT_TRUE(file.open(QIODevice::WriteOnly));
                file.write("x");
            }

            QJsonObject record;
            record[QStringLiteral("ref_image")] =
                QStringLiteral("%1_image_%2.jpg").arg(QFileInfo(batch).fileName()).arg(index);
            record[QStringLiteral("depth_png")] = png_path;
            record[QStringLiteral("raw_depth_path")] = raw_path;
            records.append(record);
        }
    }

    QJsonObject metadata;
    metadata[QStringLiteral("depth_map_results")] = records;

    const auto result = xjw::core::project::collectStoredDepthFramesForDirectory(
        metadata,
        first_batch.path());

    ASSERT_TRUE(result.status.ok) << result.status.errorMessage.toStdString();
    ASSERT_EQ(result.frames.size(), 2u);
    EXPECT_EQ(QDir::cleanPath(result.batchDir), QDir::cleanPath(first_batch.path()));
    for (const auto &frame : result.frames)
    {
        EXPECT_EQ(QFileInfo(frame.rawDepthPath).absolutePath(), first_batch.path());
    }
}

TEST(ModelWorkflowPolicyTest, PointCloudSourceRunsMeshDirectly)
{
    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("point_cloud");
    settings[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/dense_cloud.ply");

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, QJsonObject());

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("source_path")).toString(),
              QStringLiteral("E:/tmp/dense_cloud.ply"));
    EXPECT_GE(decision.modelSettings.value(QStringLiteral("threads")).toInt(), 1);
    EXPECT_LE(decision.modelSettings.value(QStringLiteral("threads")).toInt(), 8);
}

TEST(ModelWorkflowPolicyTest, InteractiveModelWorkersLeaveGuiHeadroom)
{
    EXPECT_EQ(xjw::gui::project::recommendedInteractiveModelWorkerCount(32), 8);
    EXPECT_EQ(xjw::gui::project::recommendedInteractiveModelWorkerCount(8), 6);
    EXPECT_EQ(xjw::gui::project::recommendedInteractiveModelWorkerCount(4), 2);
    EXPECT_EQ(xjw::gui::project::recommendedInteractiveModelWorkerCount(2), 1);
    EXPECT_EQ(xjw::gui::project::recommendedInteractiveModelWorkerCount(-1), 2);
}

TEST(ModelWorkflowPolicyTest, ExplicitModelWorkerCountIsPreserved)
{
    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("point_cloud");
    settings[QStringLiteral("threads")] = 12;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(
        settings,
        QJsonObject());

    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("threads")).toInt(), 12);
}

TEST(ModelWorkflowPolicyTest, ProjectDepthInputSignatureTracksImagesCamerasAndAerialTriangulation)
{
    QJsonObject metadata;
    metadata[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/tmp/image_0.jpg")},
                    {QStringLiteral("camera"),
                     QJsonObject{{QStringLiteral("fu"), 1000.0},
                                 {QStringLiteral("center"), QJsonArray{0.0, 0.0, 10.0}}}}}
    };
    metadata[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("operation"), QStringLiteral("sfm")},
                    {QStringLiteral("run_id"), QStringLiteral("run-a")}}
    };

    const QString initial_signature =
        xjw::gui::project::projectDepthInputSignature(metadata);
    ASSERT_FALSE(initial_signature.isEmpty());

    QJsonObject camera_changed = metadata;
    QJsonArray changed_images = camera_changed.value(QStringLiteral("images")).toArray();
    QJsonObject changed_image = changed_images.at(0).toObject();
    QJsonObject changed_camera = changed_image.value(QStringLiteral("camera")).toObject();
    changed_camera[QStringLiteral("fu")] = 1001.0;
    changed_image[QStringLiteral("camera")] = changed_camera;
    changed_images[0] = changed_image;
    camera_changed[QStringLiteral("images")] = changed_images;
    EXPECT_NE(xjw::gui::project::projectDepthInputSignature(camera_changed),
              initial_signature);

    QJsonObject at_changed = metadata;
    at_changed[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("operation"), QStringLiteral("sfm")},
                    {QStringLiteral("run_id"), QStringLiteral("run-b")}}
    };
    EXPECT_NE(xjw::gui::project::projectDepthInputSignature(at_changed),
              initial_signature);

    QJsonObject multiple_results = metadata;
    multiple_results[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("run_id"), QStringLiteral("run-a")}},
        QJsonObject{{QStringLiteral("run_id"), QStringLiteral("run-b")}}
    };
    EXPECT_NE(xjw::gui::project::projectDepthInputSignature(multiple_results, 0),
              xjw::gui::project::projectDepthInputSignature(multiple_results, 1));
}

TEST(ModelWorkflowPolicyTest, CompleteDepthBatchRunsTsdfDirectlyWithoutDenseCloud)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    QJsonArray depth_records;
    for (int index = 0; index < 2; ++index)
    {
        const QString depth_png = QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString raw_depth = QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.bin").arg(index));
        for (const QString &path : {depth_png, raw_depth})
        {
            QFile artifact(path);
            ASSERT_TRUE(artifact.open(QIODevice::WriteOnly));
            artifact.write("x");
        }
        QJsonObject depth_record;
        depth_record[QStringLiteral("ref_image")] = QStringLiteral("image_%1.jpg").arg(index);
        depth_record[QStringLiteral("depth_png")] = depth_png;
        depth_record[QStringLiteral("raw_depth_path")] = raw_depth;
        depth_record[QStringLiteral("config_hash")] = QStringLiteral("config-a");
        depth_record[QStringLiteral("algorithm_revision")] =
            xjw::mvs::kMvsDepthAlgorithmRevision;
        depth_records.append(depth_record);
    }

    const QString dense_cloud = QDir(temp_dir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    QFile file(dense_cloud);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write("ply\nformat ascii 1.0\nelement vertex 1\n"
               "property float x\nproperty float y\nproperty float z\n"
               "end_header\n0 0 0\n");
    file.close();

    QJsonObject dense_record;
    dense_record[QStringLiteral("dense_cloud_xyz")] = dense_cloud;
    dense_record[QStringLiteral("source_depth_map_dir")] = temp_dir.path();
    dense_record[QStringLiteral("source_depth_map_count")] = 2;
    dense_record[QStringLiteral("source_depth_config_hash")] = QStringLiteral("config-a");
    dense_record[QStringLiteral("fusion_pipeline_version")] =
        xjw::gui::project::kDenseFusionPipelineVersion;
    dense_record[QStringLiteral("point_count")] = 1;
    QJsonObject metadata;
    metadata[QStringLiteral("depth_map_results")] = depth_records;
    metadata[QStringLiteral("dense_cloud_results")] = QJsonArray{dense_record};

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("source_path")] = temp_dir.path();
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("source_data")).toString(),
              QStringLiteral("depth_maps"));
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("reconstruction_mode")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_FALSE(decision.modelSettings.contains(QStringLiteral("source_point_cloud_path")));

    dense_record.remove(QStringLiteral("fusion_pipeline_version"));
    metadata[QStringLiteral("dense_cloud_results")] = QJsonArray{dense_record};
    const auto legacy_decision =
        xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);
    EXPECT_EQ(legacy_decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(legacy_decision.modelSettings.value(QStringLiteral("reconstruction_mode")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_FALSE(legacy_decision.modelSettings.contains(QStringLiteral("source_point_cloud_path")));

    QJsonArray stale_depth_records = depth_records;
    for (qsizetype index = 0; index < stale_depth_records.size(); ++index)
    {
        QJsonObject stale_record = stale_depth_records.at(index).toObject();
        stale_record.remove(QStringLiteral("algorithm_revision"));
        stale_depth_records[index] = stale_record;
    }
    metadata[QStringLiteral("depth_map_results")] = stale_depth_records;
    const auto stale_depth_decision =
        xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);
    EXPECT_EQ(stale_depth_decision.action,
              xjw::gui::project::ModelWorkflowAction::GenerateDepthMapsThenMesh);
    EXPECT_TRUE(stale_depth_decision.reason.contains(QStringLiteral("旧版算法")));
}

TEST(ModelWorkflowPolicyTest, CompleteDepthBatchDoesNotRequireValidDenseCloud)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    QJsonArray depth_records;
    for (int index = 0; index < 2; ++index)
    {
        const QString depth_png =
            QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString raw_depth =
            QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.bin").arg(index));
        for (const QString &path : {depth_png, raw_depth})
        {
            QFile artifact(path);
            ASSERT_TRUE(artifact.open(QIODevice::WriteOnly));
            artifact.write("x");
        }
        depth_records.append(QJsonObject{
            {QStringLiteral("ref_image"), QStringLiteral("image_%1.jpg").arg(index)},
            {QStringLiteral("depth_png"), depth_png},
            {QStringLiteral("raw_depth_path"), raw_depth},
            {QStringLiteral("config_hash"), QStringLiteral("config-a")},
            {QStringLiteral("algorithm_revision"),
             xjw::mvs::kMvsDepthAlgorithmRevision}
        });
    }

    const QString dense_cloud =
        QDir(temp_dir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    QFile file(dense_cloud);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n");
    file.close();

    QJsonObject dense_record;
    dense_record[QStringLiteral("dense_cloud_xyz")] = dense_cloud;
    dense_record[QStringLiteral("source_depth_map_dir")] = temp_dir.path();
    dense_record[QStringLiteral("source_depth_map_count")] = 2;
    dense_record[QStringLiteral("source_depth_config_hash")] = QStringLiteral("config-a");
    dense_record[QStringLiteral("point_count")] = 0;

    QJsonObject metadata;
    metadata[QStringLiteral("depth_map_results")] = depth_records;
    metadata[QStringLiteral("dense_cloud_results")] = QJsonArray{dense_record};

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision =
        xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);
    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("reconstruction_mode")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_FALSE(decision.modelSettings.contains(QStringLiteral("source_point_cloud_path")));
}

TEST(ModelWorkflowPolicyTest, CompleteDepthBatchIgnoresTruncatedDenseCloud)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    QJsonArray depth_records;
    for (int index = 0; index < 2; ++index)
    {
        const QString depth_png =
            QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString raw_depth =
            QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.bin").arg(index));
        for (const QString &path : {depth_png, raw_depth})
        {
            QFile artifact(path);
            ASSERT_TRUE(artifact.open(QIODevice::WriteOnly));
            artifact.write("x");
        }
        depth_records.append(QJsonObject{
            {QStringLiteral("ref_image"), QStringLiteral("image_%1.jpg").arg(index)},
            {QStringLiteral("depth_png"), depth_png},
            {QStringLiteral("raw_depth_path"), raw_depth},
            {QStringLiteral("config_hash"), QStringLiteral("config-a")},
            {QStringLiteral("algorithm_revision"),
             xjw::mvs::kMvsDepthAlgorithmRevision}
        });
    }

    const QString dense_cloud =
        QDir(temp_dir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    QFile file(dense_cloud);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write("ply\nformat ascii 1.0\nelement vertex 1000000\n"
               "property float x\nproperty float y\nproperty float z\n"
               "end_header\n0 0 0\n");
    file.close();

    QJsonObject dense_record;
    dense_record[QStringLiteral("dense_cloud_xyz")] = dense_cloud;
    dense_record[QStringLiteral("source_depth_map_dir")] = temp_dir.path();
    dense_record[QStringLiteral("source_depth_map_count")] = 2;
    dense_record[QStringLiteral("source_depth_config_hash")] = QStringLiteral("config-a");
    dense_record[QStringLiteral("point_count")] = 1000000;

    QJsonObject metadata;
    metadata[QStringLiteral("depth_map_results")] = depth_records;
    metadata[QStringLiteral("dense_cloud_results")] = QJsonArray{dense_record};

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision =
        xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);
    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("reconstruction_mode")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_FALSE(decision.modelSettings.contains(QStringLiteral("source_point_cloud_path")));
}

TEST(ModelWorkflowPolicyTest, DepthFramesFromPreviousProjectInputAreRegenerated)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    QJsonObject metadata;
    metadata[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/tmp/current.jpg")},
                    {QStringLiteral("camera"),
                     QJsonObject{{QStringLiteral("fu"), 1200.0}}}}
    };
    metadata[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("run_id"), QStringLiteral("current-at")}}
    };

    QJsonArray depth_records;
    for (int index = 0; index < 2; ++index)
    {
        const QString depth_png =
            QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString raw_depth =
            QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.bin").arg(index));
        for (const QString &path : {depth_png, raw_depth})
        {
            QFile artifact(path);
            ASSERT_TRUE(artifact.open(QIODevice::WriteOnly));
            artifact.write("x");
        }
        depth_records.append(QJsonObject{
            {QStringLiteral("ref_image"), QStringLiteral("image_%1.jpg").arg(index)},
            {QStringLiteral("depth_png"), depth_png},
            {QStringLiteral("raw_depth_path"), raw_depth},
            {QStringLiteral("config_hash"), QStringLiteral("config-a")},
            {QStringLiteral("project_input_signature"), QStringLiteral("stale-input")}
        });
    }
    metadata[QStringLiteral("depth_map_results")] = depth_records;

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision =
        xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);
    EXPECT_EQ(decision.action,
              xjw::gui::project::ModelWorkflowAction::GenerateDepthMapsThenMesh);
    EXPECT_EQ(decision.depthSettings.value(QStringLiteral("workflow_action")).toString(),
              QStringLiteral("generate_depth_maps"));
}

TEST(ModelWorkflowPolicyTest, DisabledDepthReuseRegeneratesEvenWhenDenseCloudExists)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    QFile dense_cloud(QDir(temp_dir.path()).filePath(QStringLiteral("dense_cloud.ply")));
    ASSERT_TRUE(dense_cloud.open(QIODevice::WriteOnly | QIODevice::Text));
    dense_cloud.write("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n");
    dense_cloud.close();

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = false;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, QJsonObject());

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::GenerateDepthMapsThenMesh);
    EXPECT_TRUE(decision.depthSettings.value(QStringLiteral("force_depth_recompute")).toBool());
}

TEST(ModelWorkflowPolicyTest, StoredFramesFromAnotherDirectoryAreNotReused)
{
    QTemporaryDir selected_batch;
    QTemporaryDir other_batch;
    ASSERT_TRUE(selected_batch.isValid());
    ASSERT_TRUE(other_batch.isValid());

    QJsonArray records;
    for (int index = 0; index < 2; ++index)
    {
        const QString depth_png = QDir(other_batch.path()).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString raw_depth = QDir(other_batch.path()).filePath(QStringLiteral("depth_%1.bin").arg(index));
        for (const QString &path : {depth_png, raw_depth})
        {
            QFile file(path);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            file.write("x");
        }
        QJsonObject record;
        record[QStringLiteral("ref_image")] = QStringLiteral("image_%1.jpg").arg(index);
        record[QStringLiteral("depth_png")] = depth_png;
        record[QStringLiteral("raw_depth_path")] = raw_depth;
        record[QStringLiteral("config_hash")] = QStringLiteral("config-a");
        records.append(record);
    }

    QJsonObject metadata;
    metadata[QStringLiteral("depth_map_results")] = records;
    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("depthMapSourcePath")] = selected_batch.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::GenerateDepthMapsThenMesh);
}

TEST(ModelWorkflowPolicyTest, PartialDepthBatchIsCompletedBeforeTsdf)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    QJsonArray records;
    for (int index = 0; index < 2; ++index)
    {
        const QString depth_png = QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString raw_depth = QDir(temp_dir.path()).filePath(QStringLiteral("depth_%1.bin").arg(index));
        for (const QString &path : {depth_png, raw_depth})
        {
            QFile file(path);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            file.write("x");
        }
        QJsonObject record;
        record[QStringLiteral("ref_image")] = QStringLiteral("image_%1.jpg").arg(index);
        record[QStringLiteral("depth_png")] = depth_png;
        record[QStringLiteral("raw_depth_path")] = raw_depth;
        record[QStringLiteral("batch_frame_count")] = 9;
        record[QStringLiteral("config_hash")] = QStringLiteral("config-a");
        records.append(record);
    }

    QJsonObject metadata;
    metadata[QStringLiteral("depth_map_results")] = records;
    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::GenerateDepthMapsThenMesh);
    EXPECT_EQ(decision.depthSettings.value(QStringLiteral("workflow_action")).toString(),
              QStringLiteral("generate_depth_maps"));
}

TEST(ModelWorkflowPolicyTest, DepthFileSourceIsNormalizedToItsDirectory)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString depth_file = QDir(temp_dir.path()).filePath(QStringLiteral("depth_0.bin"));
    QFile file(depth_file);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    file.write("x");
    file.close();

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("depthMapSourcePath")] = depth_file;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, QJsonObject());

    EXPECT_EQ(QDir::cleanPath(decision.depthMapSourcePath), QDir::cleanPath(temp_dir.path()));
    EXPECT_EQ(QDir::cleanPath(decision.depthSettings.value(QStringLiteral("output_dir")).toString()),
              QDir::cleanPath(temp_dir.path()));
}

TEST(ModelWorkflowPolicyTest, DenseSettingsPreserveSelectedAerialTriangulationIndex)
{
    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("at_index")] = 3;

    const QJsonObject dense_settings =
        xjw::gui::project::denseSettingsFromModelSettings(settings, QStringLiteral("E:/tmp/mvs"));

    EXPECT_EQ(dense_settings.value(QStringLiteral("at_index")).toInt(-1), 3);
    EXPECT_FALSE(dense_settings.contains(QStringLiteral("atIndex")));
}

TEST(ModelWorkflowPolicyTest, DepthMapsWithStoredFramesRunTsdfDirectly)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString depth_0 = QDir(temp_dir.path()).filePath(QStringLiteral("depth_0.png"));
    const QString raw_0 = QDir(temp_dir.path()).filePath(QStringLiteral("depth_0.bin"));
    const QString confidence_0 = QDir(temp_dir.path()).filePath(QStringLiteral("depth_0_conf.bin"));
    const QString depth_1 = QDir(temp_dir.path()).filePath(QStringLiteral("depth_1.png"));
    const QString raw_1 = QDir(temp_dir.path()).filePath(QStringLiteral("depth_1.bin"));
    const QString confidence_1 = QDir(temp_dir.path()).filePath(QStringLiteral("depth_1_conf.bin"));
    for (const QString &path : {depth_0, raw_0, confidence_0, depth_1, raw_1, confidence_1})
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        file.write("x");
    }

    QJsonObject frame_0;
    frame_0[QStringLiteral("status")] = QStringLiteral("completed");
    frame_0[QStringLiteral("ref_image")] = QStringLiteral("image_000.jpg");
    frame_0[QStringLiteral("depth_png")] = depth_0;
    frame_0[QStringLiteral("raw_depth_path")] = raw_0;
    frame_0[QStringLiteral("raw_confidence_path")] = confidence_0;
    frame_0[QStringLiteral("config_hash")] = QStringLiteral("config-a");
    frame_0[QStringLiteral("algorithm_revision")] =
        xjw::mvs::kMvsDepthAlgorithmRevision;
    frame_0[QStringLiteral("grid_width")] = 1;
    frame_0[QStringLiteral("grid_height")] = 1;

    QJsonObject frame_1 = frame_0;
    frame_1[QStringLiteral("ref_image")] = QStringLiteral("image_001.jpg");
    frame_1[QStringLiteral("depth_png")] = depth_1;
    frame_1[QStringLiteral("raw_depth_path")] = raw_1;
    frame_1[QStringLiteral("raw_confidence_path")] = confidence_1;

    QJsonObject metadata;
    metadata[QStringLiteral("depth_map_results")] = QJsonArray{frame_0, frame_1};

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("source_path")] = temp_dir.path();
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, metadata);

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("reconstruction_mode")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_FALSE(decision.modelSettings.contains(QStringLiteral("source_point_cloud_path")));
}

TEST(ModelWorkflowPolicyTest, DepthMapsMissingFramesEstimateThenRunTsdf)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("source_path")] = temp_dir.path();
    settings[QStringLiteral("depthMapSourcePath")] = temp_dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;
    settings[QStringLiteral("quality")] = QStringLiteral("high");

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, QJsonObject());

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::GenerateDepthMapsThenMesh);
    EXPECT_TRUE(decision.depthSettings.value(QStringLiteral("pipeline_mode")).toBool());
    EXPECT_EQ(decision.depthSettings.value(QStringLiteral("workflow_action")).toString(),
              QStringLiteral("generate_depth_maps"));
    EXPECT_EQ(decision.depthSettings.value(QStringLiteral("output_dir")).toString(), temp_dir.path());
    EXPECT_EQ(decision.depthSettings.value(QStringLiteral("qualityProfile")).toString(),
              QStringLiteral("high_quality"));
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("source_data")).toString(),
              QStringLiteral("depth_maps"));
}

TEST(GenerateModelDialogTest, OffersAutomaticDepthMapsWithoutExistingDepthArtifacts)
{
    QJsonObject tie_points;
    tie_points[QStringLiteral("source_data")] = QStringLiteral("tie_points");
    tie_points[QStringLiteral("source_label")] = QStringLiteral("连接点");
    tie_points[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/sparse.ply");
    tie_points[QStringLiteral("display")] = QStringLiteral("连接点");
    tie_points[QStringLiteral("supported")] = true;

    QJsonObject model;
    model[QStringLiteral("source_data")] = QStringLiteral("model");
    model[QStringLiteral("source_label")] = QStringLiteral("模型");
    model[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/model.ply");
    model[QStringLiteral("display")] = QStringLiteral("模型");
    model[QStringLiteral("supported")] = true;

    QJsonObject legacy_settings;
    legacy_settings[QStringLiteral("source_data")] = QStringLiteral("tie_points");
    legacy_settings[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/sparse.ply");

    GenerateModelDialog dialog;
    dialog.applySettings(legacy_settings);
    dialog.setSourceCandidates(QJsonArray{tie_points, model});

    QComboBox *source_combo = nullptr;
    for (QComboBox *combo : dialog.findChildren<QComboBox *>())
    {
        if (combo->findData(QStringLiteral("tie_points")) >= 0 &&
            combo->findData(QStringLiteral("model")) >= 0)
        {
            source_combo = combo;
            break;
        }
    }

    ASSERT_NE(source_combo, nullptr);
    EXPECT_GE(source_combo->findData(QStringLiteral("depth_maps")), 0);
    EXPECT_EQ(source_combo->currentData().toString(), QStringLiteral("depth_maps"));
}

TEST(GenerateModelDialogTest, ReusesDepthMapsByDefaultForLegacySettings)
{
    QJsonObject depth_maps;
    depth_maps[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    depth_maps[QStringLiteral("source_label")] = QStringLiteral("深度图");
    depth_maps[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/mvs_output");
    depth_maps[QStringLiteral("display")] = QStringLiteral("深度图 - mvs_output");
    depth_maps[QStringLiteral("supported")] = true;

    GenerateModelDialog dialog;
    dialog.applySettings(QJsonObject());
    dialog.setSourceCandidates(QJsonArray{depth_maps});

    QCheckBox *reuse_check = nullptr;
    for (QCheckBox *check : dialog.findChildren<QCheckBox *>())
    {
        if (check->text() == QStringLiteral("重用深度图"))
        {
            reuse_check = check;
            break;
        }
    }

    ASSERT_NE(reuse_check, nullptr);
    EXPECT_TRUE(reuse_check->isChecked());
}

TEST(GenerateModelDialogTest, DisablesDepthMapReuseWhenProjectHasNoDepthMaps)
{
    QJsonObject tie_points;
    tie_points[QStringLiteral("source_data")] = QStringLiteral("tie_points");
    tie_points[QStringLiteral("source_label")] = QStringLiteral("连接点");
    tie_points[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/sparse.ply");
    tie_points[QStringLiteral("display")] = QStringLiteral("连接点");
    tie_points[QStringLiteral("supported")] = true;

    GenerateModelDialog dialog;
    dialog.applySettings(QJsonObject{{QStringLiteral("reuseDepthMaps"), true}});
    dialog.setSourceCandidates(QJsonArray{tie_points});

    auto *reuse_check = dialog.findChild<QCheckBox *>(QStringLiteral("reuseDepthMapsCheck"));
    ASSERT_NE(reuse_check, nullptr);
    EXPECT_FALSE(reuse_check->isEnabled());
    EXPECT_FALSE(reuse_check->isChecked());
    EXPECT_TRUE(reuse_check->toolTip().contains(QStringLiteral("没有可复用的深度图")));
}

TEST(GenerateModelDialogTest, EnablesDepthMapReuseWhenProjectHasDepthMaps)
{
    QJsonObject depth_maps;
    depth_maps[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    depth_maps[QStringLiteral("source_label")] = QStringLiteral("深度图");
    depth_maps[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/mvs_output");
    depth_maps[QStringLiteral("display")] = QStringLiteral("深度图 - mvs_output");
    depth_maps[QStringLiteral("supported")] = true;

    GenerateModelDialog dialog;
    dialog.applySettings(QJsonObject{{QStringLiteral("reuseDepthMaps"), true}});
    dialog.setSourceCandidates(QJsonArray{depth_maps});

    auto *reuse_check = dialog.findChild<QCheckBox *>(QStringLiteral("reuseDepthMapsCheck"));
    ASSERT_NE(reuse_check, nullptr);
    EXPECT_TRUE(reuse_check->isEnabled());
    EXPECT_TRUE(reuse_check->isChecked());
}

TEST(GenerateModelDialogTest, AcceptsBeforeDispatchingModelWorkflow)
{
    QJsonObject depth_maps;
    depth_maps[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    depth_maps[QStringLiteral("source_label")] = QStringLiteral("深度图");
    depth_maps[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/mvs_output");
    depth_maps[QStringLiteral("display")] = QStringLiteral("深度图 - mvs_output");
    depth_maps[QStringLiteral("supported")] = true;

    GenerateModelDialog dialog;
    dialog.setSourceCandidates(QJsonArray{depth_maps});
    bool accepted_when_dispatched = false;
    QJsonObject dispatched_settings;
    QObject::connect(&dialog,
                     &GenerateModelDialog::runRequested,
                     &dialog,
                     [&dialog, &accepted_when_dispatched, &dispatched_settings](const QJsonObject &settings)
                     {
                         accepted_when_dispatched = dialog.result() == QDialog::Accepted;
                         dispatched_settings = settings;
                     });

    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "onRun", Qt::DirectConnection));
    EXPECT_TRUE(accepted_when_dispatched);
    EXPECT_EQ(dialog.result(), QDialog::Accepted);
    EXPECT_EQ(dispatched_settings.value(QStringLiteral("export_format")).toString(),
              QStringLiteral("OBJ"));
}

TEST(ModelWorkflowContractTest, GenerateDenseCloudEmitsReadyAfterWritingPointCloud)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("emit self->denseCloudResultReady(")))
        << "Successful dense-cloud generation must provide the current output path to model workflows.";
    EXPECT_TRUE(block.contains(QStringLiteral("pipelineMode")))
        << "Pipeline mode must suppress intermediate modal success dialogs.";
}

TEST(DenseDepthReuseTest, ExistingDepthDetectionRequiresRawDepthArtifact)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int collectStart = source.indexOf(QStringLiteral("QSet<int> collectExistingDepthFrameIndices"));
    ASSERT_GE(collectStart, 0);
    const int collectEnd = source.indexOf(QStringLiteral("ExistingDepthAction askExistingDepthAction"), collectStart);
    ASSERT_GT(collectEnd, collectStart);
    const QString collectBlock = source.mid(collectStart, collectEnd - collectStart);
    EXPECT_TRUE(collectBlock.contains(QStringLiteral("depthFrameArtifactsExist(pngPath)")))
        << collectBlock.toStdString();

}

TEST(DenseDepthCameraLookupTest, MvsCameraLookupUsesNormalizedImageKeys)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("cameraForImagePath(camMap, imgPath")))
        << "Dense estimation should query cameras with the same normalized key used by ProjectManager.";
    EXPECT_TRUE(source.contains(QStringLiteral("cameraForImagePath(_cameraMap, _records[index].refImage")))
        << "Stored-depth fusion cache should normalize ref_image before camera lookup.";
    EXPECT_FALSE(source.contains(QStringLiteral("camMap.value(imgPath)")));
    EXPECT_FALSE(source.contains(QStringLiteral("camMap.value(stored.refImage)")));
}

TEST(DepthMapMetadataTest, DemPreviewsStayOutOfMvsDepthResults)
{
    const QString terrainSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    const QString modelSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString treeSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    ASSERT_FALSE(terrainSource.isEmpty());
    ASSERT_FALSE(modelSource.isEmpty());
    ASSERT_FALSE(treeSource.isEmpty());

    EXPECT_FALSE(terrainSource.contains(
        QStringLiteral("upsertMetaArrayRecordByPath(&metaUpdated, QStringLiteral(\"depth_map_results\")")))
        << "DEM product previews should be stored on dem_results.depth_preview_png, not as MVS depth maps.";
    EXPECT_TRUE(terrainSource.contains(QStringLiteral("collectLatestStoredDepthFrames")))
        << "DEM generation may read MVS depth_map_results as input, but must not write DEM previews there.";
    EXPECT_FALSE(modelSource.contains(QStringLiteral("depth_map_results")))
        << "Model/terrain previews should not be inserted into the MVS depth map result list.";
    EXPECT_FALSE(treeSource.contains(QStringLiteral("createSection(QStringLiteral(\"深度图\")")))
        << "MVS depth artifacts are photo overlays and must not become standalone workspace resources.";
    EXPECT_TRUE(treeSource.contains(QStringLiteral("depth_preview_png")))
        << "DEM previews should be available from the DEM section instead.";
}

TEST(TerrainPipelineAsyncTest, AutoDemConsumesOnlyCurrentDenseCloudSignal)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::denseCloudResultReady")))
        << "The automatic DEM pipeline should consume the exact dense cloud produced by the current MVS run.";
    EXPECT_TRUE(source.contains(QStringLiteral("const QString plyPath = denseCloudPath.trimmed();")))
        << "DEM generation should use the PLY path carried by the dense-cloud-ready signal.";
    EXPECT_FALSE(source.contains(QStringLiteral("pendingDenseResultCount")))
        << "Counting metadata records is still vulnerable to unrelated metadata updates.";
    EXPECT_FALSE(source.contains(QStringLiteral("denseArr.size() <= pendingDenseResultCount")))
        << "Unrelated metadata changes or pre-existing dense clouds must not drive DEM generation.";
    EXPECT_FALSE(source.contains(QStringLiteral("denseIndex = pendingDenseResultCount")))
        << "DEM generation should not scan dense metadata records to infer this run's output.";
    EXPECT_FALSE(source.contains(QStringLiteral("denseArr.at(pendingDenseResultCount)")))
        << "The first new dense record can be incomplete; scan new records for the first existing output instead.";
    EXPECT_FALSE(source.contains(QStringLiteral("const QJsonObject lastRecord = denseArr.last().toObject()")))
        << "Using last() can pick an old or unrelated dense cloud record.";
}

TEST(TerrainPipelineAsyncTest, AutoDemPipelineConnectionsUseSharedCleanupState)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("DemPipelineConnectionState")))
        << "Metadata and MVS-finished signal handles should share a cleanup state.";
    EXPECT_TRUE(source.contains(QStringLiteral("std::make_shared<DemPipelineConnectionState>")))
        << "The connection state should be owned by the queued callbacks, not raw new/delete.";
    EXPECT_TRUE(source.contains(QStringLiteral("disconnectDemPipelineConnections")))
        << "Both signal handles should be disconnected together on success or failure.";
    EXPECT_FALSE(source.contains(QStringLiteral("new QMetaObject::Connection")))
        << "Raw connection handles leak when the owner is destroyed or when the success path returns early.";
    EXPECT_FALSE(source.contains(QStringLiteral("delete connMeta")));
    EXPECT_FALSE(source.contains(QStringLiteral("delete connMvsFail")));
}

TEST(TerrainPipelineAsyncTest, AutoDemGenerationRunsOffGuiThreadAfterMvs)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void ProjectTerrainProductsManager::runFullDemPipelineInBackground"));
    ASSERT_GE(start, 0);
    const QString block = source.mid(start);

    const int denseReadyStart = block.indexOf(QStringLiteral("denseCloudResultReady"));
    ASSERT_GE(denseReadyStart, 0);
    const int mvsFailureStart = block.indexOf(QStringLiteral("// 同时监听 MVS 失败"), denseReadyStart);
    ASSERT_GT(mvsFailureStart, denseReadyStart);
    const QString denseReadyBlock = block.mid(denseReadyStart, mvsFailureStart - denseReadyStart);

    EXPECT_TRUE(source.contains(QStringLiteral("runAutomaticDemGenerationTask")))
        << "The expensive DEM generation work should be isolated in a worker helper.";
    EXPECT_TRUE(denseReadyBlock.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "MVS success should start DEM generation through the guarded GUI task runner.";
    EXPECT_FALSE(block.contains(QStringLiteral("ProjectManager::projectMetadataChanged")))
        << "The full DEM pipeline should consume the exact dense cloud result signal, not infer completion "
           "from unrelated metadata updates.";
    EXPECT_FALSE(denseReadyBlock.contains(QStringLiteral("xjw::TerrainPipeline::generateDemFromDepthMaps(")))
        << "Depth-map DEM rasterization must not run inside the GUI dense-ready callback.";
    EXPECT_FALSE(denseReadyBlock.contains(QStringLiteral("runDemProducts(plyPath")))
        << "Dense-cloud fallback DEM rasterization must not run inside the GUI dense-ready callback.";
}

TEST(TerrainPipelineAsyncTest, AutoDemPipelineScopesDenseCloudSignalToCurrentMvsRun)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void ProjectTerrainProductsManager::runFullDemPipelineInBackground"));
    ASSERT_GE(start, 0);
    const QString block = source.mid(start);

    const int denseReadyStart = block.indexOf(QStringLiteral("denseCloudResultReady"));
    ASSERT_GE(denseReadyStart, 0);
    const int mvsFailureStart = block.indexOf(QStringLiteral("// 同时监听 MVS 失败"), denseReadyStart);
    ASSERT_GT(mvsFailureStart, denseReadyStart);
    const QString denseReadyBlock = block.mid(denseReadyStart, mvsFailureStart - denseReadyStart);

    EXPECT_TRUE(block.contains(QStringLiteral("const QString expectedMvsOutputDir")))
        << "The full DEM pipeline should reserve a dedicated MVS output directory for this run.";
    EXPECT_TRUE(block.contains(QStringLiteral("mvsSettings[QStringLiteral(\"output_dir\")] = expectedMvsOutputDir")))
        << "The MVS request should write into the expected directory instead of the shared default mvs_output.";
    EXPECT_TRUE(denseReadyBlock.contains(QStringLiteral("pathIsInsideDirectory(plyPath, expectedMvsOutputDir)")))
        << "The dense-cloud-ready callback must ignore outputs from unrelated dense-cloud jobs.";
    EXPECT_FALSE(denseReadyBlock.contains(QStringLiteral("return; // 成功时由 projectMetadataChanged 处理")))
        << "The success path is now driven by denseCloudResultReady, not projectMetadataChanged.";
}

TEST(TerrainPipelineAsyncTest, FullDemPipelineUsesBoundedFeaturePairPlanning)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void ProjectTerrainProductsManager::runFullDemPipelineInBackground"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("// 步骤 3-5:"), start);
    ASSERT_GT(end, start);
    const QString matchingBlock = source.mid(start, end - start);

    EXPECT_TRUE(source.contains(QStringLiteral("#include \"MatchPhotosTask.h\"")));
    EXPECT_TRUE(matchingBlock.contains(QStringLiteral("options.pairPolicy.exhaustiveMaxImages = 80")))
        << "The full DEM pipeline should delegate bounded pair planning to MatchPhotosTask.";
    EXPECT_TRUE(matchingBlock.contains(QStringLiteral("options.pairPolicy.sequenceWindow = 4")))
        << "Large projects need a bounded sequence fallback.";
    EXPECT_FALSE(matchingBlock.contains(QStringLiteral("for (int j = i + 1; j < ctx.images.size(); ++j)")))
        << "Large DEM runs must not plan all image pairs by default.";
}

TEST(TerrainPipelineAsyncTest, FullDemPipelineFeedsReferenceCamerasToMatchPhotosTask)
{
    const QString header = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.h"));
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void ProjectTerrainProductsManager::startFullDemPipelineAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(
        QStringLiteral("void ProjectTerrainProductsManager::startDemFromDenseCloudAsync"),
        start);
    ASSERT_GT(end, start);
    const QString startBlock = source.mid(start, end - start);

    const int matchStart = source.indexOf(
        QStringLiteral("void ProjectTerrainProductsManager::runFullDemPipelineInBackground"));
    ASSERT_GE(matchStart, 0);
    const int matchEnd = source.indexOf(QStringLiteral("// 步骤 3-5:"), matchStart);
    ASSERT_GT(matchEnd, matchStart);
    const QString matchingBlock = source.mid(matchStart, matchEnd - matchStart);

    EXPECT_TRUE(header.contains(QStringLiteral("referenceCameras")))
        << "The DEM pipeline context should carry imported reference cameras into the background worker.";
    EXPECT_TRUE(startBlock.contains(QStringLiteral("getCamerasForImages(images")))
        << "The GUI-thread setup should read the project's imported camera centers before launching the worker.";
    EXPECT_TRUE(startBlock.contains(QStringLiteral("ctx.referenceCameras")))
        << "The start function should store reference cameras on the DEM pipeline context.";
    EXPECT_TRUE(matchingBlock.contains(QStringLiteral("context.referenceCameras = ctx.referenceCameras")))
        << "The background matching step should feed reference cameras into MatchPhotosTask.";
    EXPECT_TRUE(matchingBlock.contains(QStringLiteral("options.useReferencePreselection")))
        << "Reference preselection should be enabled when imported cameras are available.";
}

TEST(TerrainPipelineAsyncTest, DenseCloudResultSignalCarriesOutputPath)
{
    const QString denseHeader = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.h"));
    const QString reconstructionHeader = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.h"));
    const QString managerHeader = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString denseSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString reconstructionSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp"));
    const QString managerSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(denseHeader.isEmpty());
    ASSERT_FALSE(reconstructionHeader.isEmpty());
    ASSERT_FALSE(managerHeader.isEmpty());
    ASSERT_FALSE(denseSource.isEmpty());
    ASSERT_FALSE(reconstructionSource.isEmpty());
    ASSERT_FALSE(managerSource.isEmpty());

    const QString signalDecl =
        QStringLiteral("void denseCloudResultReady(const QString &denseCloudPath, int pointCount);");
    EXPECT_TRUE(denseHeader.contains(signalDecl));
    EXPECT_TRUE(reconstructionHeader.contains(signalDecl));
    EXPECT_TRUE(managerHeader.contains(signalDecl));
    EXPECT_TRUE(denseSource.contains(QStringLiteral("emit self->denseCloudResultReady(outputPly, pointCount);")))
        << "MVS completion should publish the exact PLY path produced by the current run.";
    EXPECT_TRUE(reconstructionSource.contains(
        QStringLiteral("&ProjectDenseReconstructionManager::denseCloudResultReady")));
    EXPECT_TRUE(reconstructionSource.contains(
        QStringLiteral("&ProjectReconstructionManager::denseCloudResultReady")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("&ProjectReconstructionManager::denseCloudResultReady")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("&ProjectManager::denseCloudResultReady")));
}

TEST(TerrainPipelineAsyncTest, DenseCloudDemRunsOffGuiThread)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::startDemFromDenseCloudAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::startMapProjectAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "The Async entry point should run DEM/DOM IO and rasterization through the guarded task runner.";
    EXPECT_TRUE(block.contains(QStringLiteral("runDemProducts(resolvedDenseCloud")))
        << "The worker should call the non-UI terrain function and report errors on the GUI thread.";
    EXPECT_FALSE(block.contains(QStringLiteral("(void)QtConcurrent::run([self,")))
        << "Open-coded QtConcurrent can still start terrain work after the manager owner is destroyed.";
    EXPECT_FALSE(block.contains(QStringLiteral("runDemProductsOrWarn")))
        << "runDemProductsOrWarn shows QMessageBox and must stay on the GUI thread.";
}

TEST(TerrainPipelineAsyncTest, StereoPoint2DemRunsOffGuiThread)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::startStereoAndPoint2DemAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::startFullDemPipelineAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "The legacy stereo-to-DEM Async entry point should also keep DEM/DOM IO off the GUI thread.";
    EXPECT_TRUE(block.contains(QStringLiteral("runDemProducts(resolvedDenseCloud")))
        << "The worker should call the non-UI terrain function and report errors after returning to the GUI thread.";
    EXPECT_FALSE(block.contains(QStringLiteral("runDemProductsOrWarn")))
        << "The Async entry point must not call the QMessageBox wrapper directly.";
}

TEST(TerrainPipelineAsyncTest, MapProjectRunsOffGuiThread)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::startMapProjectAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::runFullDemPipelineInBackground"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "The DOM/map projection Async entry point should keep ortho IO and rasterization off the GUI thread.";
    EXPECT_TRUE(block.contains(QStringLiteral("runOrthoProduct(sourceImages")))
        << "The worker should call the non-UI ortho function and report errors after returning to the GUI thread.";
    EXPECT_FALSE(block.contains(QStringLiteral("runOrthoProductOrWarn")))
        << "The Async entry point must not call the QMessageBox wrapper directly.";
}

TEST(TerrainPipelineAsyncTest, TerrainProductsManagerDropsBlockingUiWrappers)
{
    const QString header = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.h"));
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_FALSE(header.contains(QStringLiteral("runDemProductsOrWarn")));
    EXPECT_FALSE(header.contains(QStringLiteral("runOrthoProductOrWarn")));
    EXPECT_FALSE(source.contains(QStringLiteral("ProjectTerrainProductsManager::runDemProductsOrWarn")));
    EXPECT_FALSE(source.contains(QStringLiteral("ProjectTerrainProductsManager::runOrthoProductOrWarn")));
}

TEST(GuiAsyncLifetimeTest, TerrainAndDenseBackgroundCallbacksUseQPointerGuards)
{
    const QString terrainSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    const QString denseSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString managerSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(terrainSource.isEmpty());
    ASSERT_FALSE(denseSource.isEmpty());
    ASSERT_FALSE(managerSource.isEmpty());

    EXPECT_TRUE(terrainSource.contains(QStringLiteral("#include <QPointer>")));
    EXPECT_TRUE(terrainSource.contains(QStringLiteral("QPointer<ProjectTerrainProductsManager> self(this)")));
    EXPECT_TRUE(denseSource.contains(QStringLiteral("#include <QPointer>")));
    EXPECT_TRUE(denseSource.contains(QStringLiteral("QPointer<ProjectDenseReconstructionManager> self(this)")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("#include <QPointer>")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("QPointer<ProjectManager> self(this)")));
}

TEST(GuiTaskRunnerTest, DeliversBackgroundExceptionToGuiCallback)
{
    QObject owner;
    bool callbackInvoked = false;
    QString errorMessage;
    QThread *callbackThread = nullptr;

    QFuture<void> future = xjw::gui::tasks::runGuardedWithOutcome(
        &owner,
        []() -> int
        {
            throw std::runtime_error("intentional test failure");
        },
        [&callbackInvoked, &errorMessage, &callbackThread](QObject *,
                                                            xjw::gui::tasks::TaskOutcome<int> outcome)
        {
            callbackInvoked = true;
            errorMessage = outcome.errorMessage;
            callbackThread = QThread::currentThread();
        });

    QElapsedTimer timer;
    timer.start();
    while (!callbackInvoked && timer.elapsed() < 3000)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 25);
        QThread::msleep(5);
    }
    future.waitForFinished();
    QCoreApplication::processEvents(QEventLoop::AllEvents, 25);

    EXPECT_TRUE(callbackInvoked);
    EXPECT_EQ(errorMessage, QStringLiteral("intentional test failure"));
    EXPECT_EQ(callbackThread, owner.thread());
}

TEST(GuiAsyncLifetimeTest, FullDemPipelineUsesGuardedTaskRunner)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::startFullDemPipelineAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void ProjectTerrainProductsManager::startDemFromDenseCloudAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(source.contains(QStringLiteral("#include \"GuiTaskRunner.h\"")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "The full DEM pipeline worker should use the shared guarded runner.";
    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<ProjectTerrainProductsManager> self(this)")))
        << "The worker should not capture the DEM manager as a raw this pointer.";
    EXPECT_TRUE(block.contains(QStringLiteral("[self, ctx]()")))
        << "The worker should capture the guarded pointer and context only.";
    EXPECT_TRUE(block.contains(QStringLiteral("runFullDemPipelineInBackground(ctx)")))
        << "The existing background implementation should remain off the GUI thread.";
    EXPECT_FALSE(block.contains(QStringLiteral("QtConcurrent::run([self, ctx")))
        << "Do not open-code QtConcurrent for GUI-owned long-running workflows.";
    EXPECT_FALSE(block.contains(QStringLiteral("[this, ctx]()")))
        << "The guarded runner cannot protect a work closure that captures raw this.";
}

TEST(GuiAsyncLifetimeTest, BundleAdjustProgressCallbackUsesQPointerGuard)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int callbackStart = source.indexOf(QStringLiteral("opts.baOpt.progressCallback ="));
    ASSERT_GE(callbackStart, 0);
    const int callbackEnd = source.indexOf(
        QStringLiteral("emit atProgressChanged(QStringLiteral(\"光束法平差准备中...\"), 1);"),
        callbackStart);
    ASSERT_GT(callbackEnd, callbackStart);
    const QString callbackBlock = source.mid(callbackStart, callbackEnd - callbackStart);

    EXPECT_TRUE(source.contains(QStringLiteral("QPointer<ProjectManager> baProgressSelf")))
        << "BA progress callbacks can outlive the GUI owner and must use QPointer.";
    EXPECT_TRUE(callbackBlock.contains(QStringLiteral("[baProgressSelf, cancelFlag]")))
        << "The callback should capture only the guarded ProjectManager pointer and cancellation flag.";
    EXPECT_FALSE(callbackBlock.contains(QStringLiteral("[self = this, cancelFlag]")))
        << "Capturing raw this in the BA progress callback can enqueue events after ProjectManager is destroyed.";
    EXPECT_FALSE(callbackBlock.contains(QStringLiteral("QMetaObject::invokeMethod(\n                self,")))
        << "Queued BA progress updates must target baProgressSelf.data(), not a raw this alias.";
}

TEST(GuiAsyncLifetimeTest, BundleAdjustUsesGuardedTaskRunner)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectManager::startBundleAdjustAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void ProjectManager::startGenerateModelAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(source.contains(QStringLiteral("#include \"GuiTaskRunner.h\"")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "Bundle-adjust background execution should use the shared guarded task runner.";
    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<ProjectManager> self(this)")));
    EXPECT_FALSE(block.contains(QStringLiteral("(void)QtConcurrent::run(")))
        << "Open-coded QtConcurrent lacks the shared owner check before work starts.";
}

TEST(GuiAsyncLifetimeTest, EstimateDepthMapCallbacksUseQPointerGuard)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("bool ProjectDenseReconstructionManager::startEstimateDepthMapsAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startFuseDepthMapsAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<ProjectDenseReconstructionManager> self(this)")))
        << "Depth-map estimation signal callbacks must share a guarded manager pointer.";
    EXPECT_TRUE(block.contains(QStringLiteral("[self, projectPath](const QString &stage, float ratio)")))
        << "Progress callbacks should not capture raw this.";
    EXPECT_TRUE(block.contains(QStringLiteral("batch_frame_count = selectedImages.size()")))
        << "Artifact callbacks should not capture raw this.";
    EXPECT_TRUE(block.contains(QStringLiteral("selectedImageCount = selectedImages.size()")))
        << "Finished callbacks should not capture raw this.";
    EXPECT_TRUE(block.contains(QStringLiteral("emit self->depthMapBatchReady(mvsOutDir, selectedImageCount)")));
    EXPECT_FALSE(block.contains(QStringLiteral("[this](const QString &stage, float ratio)")));
    EXPECT_FALSE(block.contains(QStringLiteral("[this, sparseXyz, mvsOutDir]")));
    EXPECT_FALSE(block.contains(QStringLiteral("[this](bool success)")));
}

TEST(GuiAsyncLifetimeTest, GenerateDenseCloudCallbacksUseQPointerGuard)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<ProjectDenseReconstructionManager> self(this)")))
        << "Dense-cloud generation callbacks must share a guarded manager pointer.";
    EXPECT_TRUE(block.contains(QStringLiteral("[self, projectPath](const QString &stage, float ratio)")))
        << "Progress callbacks should not capture raw this.";
    EXPECT_TRUE(block.contains(QStringLiteral("batch_frame_count = selectedImages.size()")))
        << "Artifact callbacks should not capture raw this.";
    EXPECT_TRUE(block.contains(QStringLiteral("source_depth_map_count = static_cast<int>(selectedImages.size())")))
        << "Point cloud callbacks should not capture raw this.";
    EXPECT_TRUE(block.contains(QStringLiteral("[self, settings, continueMissingMode, projectPath](bool success)")))
        << "Finished callbacks should reuse the guarded pointer.";
    EXPECT_FALSE(block.contains(QStringLiteral("[this](const QString &stage, float ratio)")));
    EXPECT_FALSE(block.contains(QStringLiteral("[this, sparseXyz, mvsOutDir]")));
    EXPECT_FALSE(block.contains(QStringLiteral("[this, mvsOutDir]")));
    EXPECT_FALSE(block.contains(QStringLiteral("[self = QPointer<ProjectDenseReconstructionManager>(this)")));
}

TEST(GuiAsyncLifetimeTest, DenseCloudFusionUsesGuardedTaskRunner)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startFuseDepthMapsAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(source.contains(QStringLiteral("#include \"GuiTaskRunner.h\"")));
    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<ProjectDenseReconstructionManager> self(this)")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "Dense-cloud fusion should use the shared guarded runner before starting long-running work.";
    EXPECT_FALSE(block.contains(QStringLiteral("(void)QtConcurrent::run([self,")))
        << "Open-coded QtConcurrent can still start fusion after the manager owner has been destroyed.";
}

TEST(GuiAsyncLifetimeTest, DenseCloudRefineUsesGuardedTaskRunner)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::cancelMvs"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(source.contains(QStringLiteral("#include \"GuiTaskRunner.h\"")));
    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<ProjectDenseReconstructionManager> self(this)")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "Dense-cloud post-processing should use the shared guarded runner before loading large clouds.";
    EXPECT_FALSE(block.contains(QStringLiteral("(void)QtConcurrent::run([self,")))
        << "Open-coded QtConcurrent can still start dense refine work after the manager owner is destroyed.";
}

TEST(GuiAsyncLifetimeTest, DepthMapSparsePreloadWorkersGuardGeneratorLifetime)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const auto blockBetween = [&source](const QString &begin, const QString &finish) {
        const int start = source.indexOf(begin);
        EXPECT_GE(start, 0);
        const int end = source.indexOf(finish, start);
        EXPECT_GT(end, start);
        return source.mid(start, end - start);
    };

    const QString estimateBlock = blockBetween(
        QStringLiteral("bool ProjectDenseReconstructionManager::startEstimateDepthMapsAsync"),
        QStringLiteral("ProjectDenseReconstructionManager::startFuseDepthMapsAsync"));
    const QString denseBlock = blockBetween(
        QStringLiteral("ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"),
        QStringLiteral("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync"));

    for (const QString &block : {estimateBlock, denseBlock})
    {
        EXPECT_TRUE(block.contains(QStringLiteral("QPointer<DepthMapGenerator> genSelf(gen)")))
            << "Sparse preload workers should guard the generator before leaving the GUI thread.";
        EXPECT_TRUE(block.contains(QStringLiteral("const QString projectPath = _owner ? _owner->currentProjectPath() : QString()")))
            << "Sparse preload workers must bind their result to the project that launched them.";
        EXPECT_TRUE(block.contains(QStringLiteral("QtConcurrent::run([self, genSelf, sparseXyz, views, request, projectPath]()")))
            << "The worker should capture the guarded generator pointer, not raw gen.";
        EXPECT_TRUE(block.contains(QStringLiteral("QMetaObject::invokeMethod(genSelf.data(), [self, genSelf, sparseCloud, projectPath]()")))
            << "Sparse cloud handoff should be posted back through the guarded generator.";
        EXPECT_TRUE(block.contains(QStringLiteral("self->_owner->currentProjectPath() != projectPath")))
            << "Sparse preload completion must not start depth estimation after the user switches projects.";
        EXPECT_FALSE(block.contains(QStringLiteral("QtConcurrent::run([gen, sparseXyz, views, request]()")));
        EXPECT_FALSE(block.contains(QStringLiteral("gen->setSparseCloud(sparse)")));
        EXPECT_FALSE(block.contains(QStringLiteral("QMetaObject::invokeMethod(gen, \"start\"")));
    }
}

TEST(GuiAsyncLifetimeTest, DepthMapGeneratorOwnsAndJoinsBackgroundFuture)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const int destructorStart = source.indexOf(QStringLiteral("DepthMapGenerator::~DepthMapGenerator()"));
    ASSERT_GE(destructorStart, 0);
    const int setViewsStart = source.indexOf(QStringLiteral("void DepthMapGenerator::setViews"), destructorStart);
    ASSERT_GT(setViewsStart, destructorStart);
    const QString destructorBlock = source.mid(destructorStart, setViewsStart - destructorStart);

    const int startStart = source.indexOf(QStringLiteral("void DepthMapGenerator::start()"));
    ASSERT_GE(startStart, 0);
    const int rangeStart = source.indexOf(QStringLiteral("bool DepthMapGenerator::estimateDepthRange"), startStart);
    ASSERT_GT(rangeStart, startStart);
    const QString startBlock = source.mid(startStart, rangeStart - startStart);

    EXPECT_TRUE(header.contains(QStringLiteral("#include <QFuture>")));
    EXPECT_TRUE(header.contains(QStringLiteral("QFuture<void> _backgroundFuture")));
    EXPECT_TRUE(destructorBlock.contains(QStringLiteral("requestCancel();")));
    EXPECT_TRUE(destructorBlock.contains(QStringLiteral("_backgroundFuture.waitForFinished();")))
        << "DepthMapGenerator must not be destroyed while its background MVS worker still uses members.";
    EXPECT_TRUE(startBlock.contains(QStringLiteral("_backgroundFuture = QtConcurrent::run([this]()")))
        << "start() should retain the background future so destruction can join it safely.";
    EXPECT_TRUE(startBlock.contains(QStringLiteral("if (_backgroundFuture.isRunning())")))
        << "start() should avoid launching overlapping background workers on the same generator.";
}

TEST(GuiAsyncLifetimeTest, ProjectModelTasksUseQPointerGuards)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int meshStart = source.indexOf(
        QStringLiteral("ProjectModelManager::startMeshReconstructionAsync"));
    ASSERT_GE(meshStart, 0);
    const int textureStart = source.indexOf(
        QStringLiteral("void ProjectModelManager::startTextureMappingAsync"), meshStart);
    ASSERT_GT(textureStart, meshStart);
    const int finalizerStart = source.indexOf(
        QStringLiteral("void ProjectModelManager::finalizeModelGenerationSuccess"), textureStart);
    ASSERT_GT(finalizerStart, textureStart);
    const QString meshBlock = source.mid(meshStart, textureStart - meshStart);
    const QString textureBlock = source.mid(textureStart, finalizerStart - textureStart);

    EXPECT_TRUE(source.contains(QStringLiteral("#include <QPointer>")));
    EXPECT_TRUE(source.contains(QStringLiteral("makeProgressReporter(QPointer<ProjectModelManager> manager,")))
        << "Background mesh workflow progress must post through a guarded manager pointer.";
    EXPECT_TRUE(source.contains(QStringLiteral("QPointer<ProjectManager> owner,")))
        << "Progress callbacks must also guard the owning ProjectManager.";
    EXPECT_TRUE(source.contains(QStringLiteral("owner->currentProjectPath() != projectPath")))
        << "Model progress callbacks must not update UI after the user switches projects.";
    EXPECT_FALSE(source.contains(QStringLiteral("makeProgressReporter(this)")));
    EXPECT_TRUE(source.contains(QStringLiteral("QObject::connect(watcher, &QFutureWatcher<ModelTaskResult>::finished,\n"
                                             "                     watcher,")))
        << "Model task finished callbacks should be tied to the watcher lifetime.";
    EXPECT_FALSE(source.contains(QStringLiteral("QObject::connect(watcher, &QFutureWatcher<ModelTaskResult>::finished,\n"
                                              "                     owner,")));

    EXPECT_TRUE(meshBlock.contains(QStringLiteral("QPointer<ProjectModelManager> self(this)")));
    EXPECT_TRUE(meshBlock.contains(QStringLiteral("const QString projectPath = _owner ? _owner->currentProjectPath() : QString()")))
        << "Mesh tasks must bind results to the project that launched them.";
    EXPECT_TRUE(meshBlock.contains(QStringLiteral("QPointer<ProjectManager> ownerGuard(_owner)")));
    EXPECT_TRUE(meshBlock.contains(QStringLiteral(
        "[self,\n"
        "         ownerGuard,\n"
        "         resolvedSource,\n"
        "         effectiveSettings,\n"
        "         projectPath,\n"
        "         cancel_flag]() -> ModelTaskResult")));
    EXPECT_TRUE(meshBlock.contains(QStringLiteral("makeProgressReporter(self, ownerGuard, projectPath)")));
    EXPECT_TRUE(meshBlock.contains(QStringLiteral("[self, ownerGuard, resolvedSource, effectiveSettings, projectPath, dialogTitle](const ModelTaskResult &task)")));
    EXPECT_TRUE(meshBlock.contains(QStringLiteral("ownerGuard->currentProjectPath() != projectPath")))
        << "Mesh task completion must not write results after the user switches projects.";
    EXPECT_TRUE(meshBlock.contains(QStringLiteral("[self, resolvedSource, effectiveSettings, dialogTitle](const QJsonObject &taskResult)")));
    EXPECT_FALSE(meshBlock.contains(QStringLiteral("[this, denseCloudPath, outputRoot, settings]")));
    EXPECT_FALSE(meshBlock.contains(QStringLiteral("[this, denseCloudPath, settings]")));

    EXPECT_TRUE(textureBlock.contains(QStringLiteral("QPointer<ProjectModelManager> self(this)")));
    EXPECT_TRUE(textureBlock.contains(QStringLiteral("const QString projectPath = _owner ? _owner->currentProjectPath() : QString()")))
        << "Texture tasks must bind results to the project that launched them.";
    EXPECT_TRUE(textureBlock.contains(QStringLiteral("QPointer<ProjectManager> ownerGuard(_owner)")));
    EXPECT_TRUE(textureBlock.contains(QStringLiteral(
        "[self,\n"
        "         ownerGuard,\n"
        "         meshPath,\n"
        "         productsDir,\n"
        "         depthMapSourcePath,\n"
        "         settings,\n"
        "         projectPath,\n"
        "         cancel_flag,\n"
        "         allow_vertex_color_fallback]() -> ModelTaskResult")));
    EXPECT_TRUE(textureBlock.contains(QStringLiteral("makeProgressReporter(self, ownerGuard, projectPath)")));
    EXPECT_TRUE(textureBlock.contains(QStringLiteral(
        "[self,\n"
        "         ownerGuard,\n"
        "         meshPath,\n"
        "         baseRecord,\n"
        "         projectPath,\n"
        "         cancel_flag](const ModelTaskResult &task)")));
    EXPECT_TRUE(textureBlock.contains(QStringLiteral("request.isCancelled = [cancel_flag]()")));
    EXPECT_TRUE(textureBlock.contains(QStringLiteral("ownerGuard->currentProjectPath() != projectPath")))
        << "Texture task completion must not write results after the user switches projects.";
    EXPECT_TRUE(textureBlock.contains(QStringLiteral("[self, meshPath, baseRecord](const QJsonObject &taskResult)")));
    EXPECT_FALSE(textureBlock.contains(QStringLiteral("[this, meshPath, productsDir, settings]")));
    EXPECT_FALSE(textureBlock.contains(QStringLiteral("[this, meshPath, baseRecord]")));
}

TEST(GuiAsyncLifetimeTest, CameraSceneAsyncLoadCallbacksUseQPointerGuards)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const auto blockBetween = [&source](const QString &begin, const QString &finish) {
        const int start = source.indexOf(begin);
        EXPECT_GE(start, 0);
        const int end = source.indexOf(finish, start);
        EXPECT_GT(end, start);
        return source.mid(start, end - start);
    };

    const QString xyzBlock = blockBetween(
        QStringLiteral("void CameraSceneWidget::loadPointCloudFromXyz"),
        QStringLiteral("void CameraSceneWidget::loadModelFromPly"));
    const QString plyBlock = blockBetween(
        QStringLiteral("void CameraSceneWidget::loadModelFromPly"),
        QStringLiteral("void CameraSceneWidget::loadModelFromObj"));
    const QString objBlock = blockBetween(
        QStringLiteral("void CameraSceneWidget::loadModelFromObj"),
        QStringLiteral("QVector3D CameraSceneWidget::sceneCenter"));

    for (const QString &block : {xyzBlock, plyBlock})
    {
        EXPECT_TRUE(block.contains(QStringLiteral("QPointer<CameraSceneWidget> self(this)")));
        EXPECT_TRUE(block.contains(
            QStringLiteral("connect(watcher, &QFutureWatcher<std::shared_ptr<RenderCloud>>::finished,\n"
                           "            watcher,")))
            << "Async 3D load finished callbacks should be tied to the watcher lifetime.";
        EXPECT_TRUE(block.contains(QStringLiteral("[self, watcher, gen]()")))
            << "Finished callbacks from async 3D loading must not capture raw this.";
        EXPECT_TRUE(block.contains(QStringLiteral("if (!self)")));
        EXPECT_FALSE(block.contains(
            QStringLiteral("connect(watcher, &QFutureWatcher<std::shared_ptr<RenderCloud>>::finished,\n"
                           "            this,")));
        EXPECT_FALSE(block.contains(QStringLiteral("[this, watcher, gen]()")));
    }
    EXPECT_TRUE(objBlock.contains(QStringLiteral("QPointer<CameraSceneWidget> self(this)")));
    EXPECT_TRUE(objBlock.contains(
        QStringLiteral("connect(watcher, &QFutureWatcher<ObjLoadResult>::finished,\n"
                       "            watcher,")))
        << "OBJ geometry and texture completion must remain tied to the watcher lifetime.";
    EXPECT_TRUE(objBlock.contains(QStringLiteral("[self, watcher, gen]()")));
    EXPECT_TRUE(objBlock.contains(QStringLiteral("if (!self)")));
    EXPECT_FALSE(objBlock.contains(
        QStringLiteral("connect(watcher, &QFutureWatcher<ObjLoadResult>::finished,\n"
                       "            this,")));
    EXPECT_FALSE(objBlock.contains(QStringLiteral("[this, watcher, gen]()")));
    EXPECT_TRUE(plyBlock.contains(QStringLiteral("QMetaObject::invokeMethod(self.data(), [self, gen, percent, statusText]()")))
        << "PLY load progress emitted from a worker thread must be queued back through the guarded widget.";
    EXPECT_TRUE(objBlock.contains(QStringLiteral("QMetaObject::invokeMethod(self.data(), [self, gen, percent, statusText]()")))
        << "OBJ and MTL loading must keep parsing and image IO off the GUI thread.";
    EXPECT_FALSE(plyBlock.contains(QStringLiteral("if (self)\n            {\n                emit self->plyLoadProgressChanged(gen, percent, statusText);")))
        << "The PLY worker must not directly emit signals through a GUI object from the worker thread.";
}

TEST(GuiAsyncLifetimeTest, ImageViewAsyncLoadCallbackUsesQPointerGuard)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/ImageViewWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("bool ImageViewWidget::loadImage"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void ImageViewWidget::setMatchPoints"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<ImageViewWidget> self(this)")));
    EXPECT_TRUE(block.contains(QStringLiteral("connect(watcher, &QFutureWatcher<QImage>::finished,\n"
                                             "            watcher,")))
        << "Image view decode finished callbacks should be tied to the watcher lifetime.";
    EXPECT_TRUE(block.contains(QStringLiteral("[self, watcher, imagePath]()")));
    EXPECT_TRUE(block.contains(QStringLiteral("if (!self)")));
    EXPECT_FALSE(block.contains(QStringLiteral("connect(watcher, &QFutureWatcher<QImage>::finished,\n"
                                              "            this,")));
    EXPECT_FALSE(block.contains(QStringLiteral("[this, watcher, imagePath]()")))
        << "Async image decode callbacks must not capture the view widget through raw this.";
}

TEST(GuiAsyncLifetimeTest, RestoredActiveImageSingleShotChecksProjectBeforeLoading)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void MainWindow::applyUiSettings"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void MainWindow::closeEvent"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(
        QStringLiteral("const QString projectPath = _projectManager ? _projectManager->currentProjectPath() : QString();")))
        << "Delayed active-image restore must remember the project it belongs to.";
    EXPECT_TRUE(block.contains(QStringLiteral("[this, imagePath, projectPath]()")))
        << "Delayed active-image restore should compare against the captured project path.";
    EXPECT_TRUE(block.contains(QStringLiteral("_projectManager->currentProjectPath() != projectPath")))
        << "Do not let stale UI settings from a previous project switch the central image view later.";
}

TEST(GuiAsyncLifetimeTest, ProjectOpenDefersHeavyWidgetHydrationUntilAfterOpenSignalReturns)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString hydratorSource = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/ProjectUiHydrator.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(hydratorSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ProjectUiHydrator *_projectUiHydrator{};")))
        << "MainWindow should delegate project UI hydration to one generation-safe helper.";

    const int openedStart = source.indexOf(QStringLiteral("void MainWindow::onProjectOpened"));
    ASSERT_GE(openedStart, 0);
    const int openedEnd = source.indexOf(QStringLiteral("void MainWindow::scheduleProjectMetadataRefresh"),
                                         openedStart);
    ASSERT_GT(openedEnd, openedStart);
    const QString openedBlock = source.mid(openedStart, openedEnd - openedStart);

    EXPECT_TRUE(openedBlock.contains(
        QStringLiteral("scheduleProjectMetadataRefresh(_projectManager->coreProjectMeta())")))
        << "Project-open progress must be able to finish before heavy widgets are hydrated.";
    EXPECT_FALSE(openedBlock.contains(QStringLiteral("_dataTree->loadFromJson(coreMeta)")));
    EXPECT_FALSE(openedBlock.contains(QStringLiteral("_workspaceCenter->setProjectMeta(coreMeta)")));
    EXPECT_FALSE(openedBlock.contains(QStringLiteral("_photoStrip->loadFromJson(coreMeta)")));

    EXPECT_TRUE(hydratorSource.contains(QStringLiteral("QPointer<ProjectUiHydrator> self(this)")))
        << "Delayed widget hydration must not keep a raw helper pointer.";
    EXPECT_TRUE(hydratorSource.contains(QStringLiteral("QTimer::singleShot(0, this")))
        << "Widget hydration should be posted back to the event loop instead of running inside projectOpened.";
    EXPECT_TRUE(hydratorSource.contains(QStringLiteral("generation != self->_generation")))
        << "Stale delayed hydration from a previous project must not update the current window.";
}

TEST(GuiAsyncLifetimeTest, CameraSetupUsesGuiTaskRunnerForBackgroundSfm)
{
    const QString runnerSource = readProjectSourceFile(QStringLiteral("src/gui/tasks/GuiTaskRunner.h"));
    const QString cameraSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectCameraSetupManager.cpp"));
    ASSERT_FALSE(runnerSource.isEmpty());
    ASSERT_FALSE(cameraSource.isEmpty());

    EXPECT_TRUE(runnerSource.contains(QStringLiteral("runGuarded")));
    EXPECT_TRUE(runnerSource.contains(QStringLiteral("postGuarded")));
    EXPECT_TRUE(runnerSource.contains(QStringLiteral("QPointer<Owner>")));

    EXPECT_TRUE(cameraSource.contains(QStringLiteral("#include \"GuiTaskRunner.h\"")));
    EXPECT_TRUE(cameraSource.contains(QStringLiteral("xjw::gui::tasks::runGuarded")));
    EXPECT_TRUE(cameraSource.contains(QStringLiteral("xjw::gui::tasks::postGuarded")));
    EXPECT_TRUE(cameraSource.contains(QStringLiteral("QPointer<ProjectManager> ownerGuard(_owner)")))
        << "Camera SFM initialization must guard callbacks against ProjectManager lifetime.";
    EXPECT_TRUE(cameraSource.contains(
        QStringLiteral("const QString projectPath = _owner ? _owner->currentProjectPath() : QString();")))
        << "Camera SFM initialization must bind all callbacks to the project active at launch.";
    EXPECT_TRUE(cameraSource.contains(QStringLiteral("ownerGuard->currentProjectPath() != projectPath")))
        << "Camera SFM initialization must not write results after switching projects.";
    EXPECT_FALSE(cameraSource.contains(QStringLiteral("QtConcurrent::run([self, opts")))
        << "Camera SFM initialization should use the shared guarded runner instead of open-coded QtConcurrent.";
}

TEST(GuiAsyncLifetimeTest, CanvasFeatureLoadCallbacksUseRequestGeneration)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void CanvasWidget::startSpLoadForImage"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void CanvasWidget::reloadInterestPoints"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(header.contains(QStringLiteral("int _featureLoadGeneration{0}")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include <QPointer>")));
    EXPECT_TRUE(block.contains(QStringLiteral("const int generation = ++_featureLoadGeneration")));
    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<CanvasWidget> self(this)")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "Canvas feature-load results should not be pulled through QFutureWatcher::result().";
    EXPECT_TRUE(block.contains(
        QStringLiteral("[imagePathCopy, activeSuffix, projectPath, shouldEstimateOrientation]()")));
    EXPECT_TRUE(block.contains(QStringLiteral("std::vector<cv::KeyPoint> kps")))
        << "Finished callback should receive keypoints as a guarded task result.";
    EXPECT_TRUE(block.contains(QStringLiteral("generation != self->_featureLoadGeneration")))
        << "Late feature-load completions from an older image/suffix must not update the current canvas.";
    EXPECT_TRUE(block.contains(QStringLiteral("const QString cacheKey = featureCacheKey(imagePathCopy, activeSuffix)")));
    EXPECT_TRUE(block.contains(QStringLiteral("const QString key = self->featureCacheKey(imagePathCopy, activeSuffix)")));
    EXPECT_FALSE(header.contains(QStringLiteral("QFutureWatcher<std::vector<cv::KeyPoint>> *_spWatcher")));
    EXPECT_FALSE(block.contains(QStringLiteral("QFutureWatcher<std::vector<cv::KeyPoint>")));
    EXPECT_FALSE(block.contains(QStringLiteral("watcher->result()")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_lastRequestedSpPath")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_lastRequestedSpSuffix")));
    EXPECT_FALSE(source.contains(QStringLiteral("connect(m_spWatcher, &QFutureWatcher<std::vector<cv::KeyPoint>>::finished")));
    EXPECT_FALSE(block.contains(
        QStringLiteral("connect(watcher, &QFutureWatcher<std::vector<cv::KeyPoint>>::finished,\n"
                       "            this,")));
}

TEST(GuiAsyncLifetimeTest, CanvasImageLoadCallbacksUseQPointerGuard)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void CanvasWidget::showImage"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("void CanvasWidget::showMatchedPair"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("QPointer<CanvasWidget> self(this)")));
    EXPECT_TRUE(block.contains(QStringLiteral("connect(watcher, &QFutureWatcher<QImage>::finished, watcher,")))
        << "Canvas image decode finished callbacks should be tied to the watcher lifetime.";
    EXPECT_TRUE(block.contains(QStringLiteral("[self, watcher, loadedPath = pathCopy]()")));
    EXPECT_TRUE(block.contains(QStringLiteral("if (!self)")));
    EXPECT_FALSE(block.contains(QStringLiteral("connect(watcher, &QFutureWatcher<QImage>::finished, this,")));
    EXPECT_FALSE(block.contains(QStringLiteral("[this, watcher, loadedPath = pathCopy]()")))
        << "Canvas image decode callbacks must not capture the view through raw this.";
}

TEST(FeatureNamingCleanupTest, CanvasWidgetDoesNotIncludeTorchExtractorHeaders)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_FALSE(source.contains(QStringLiteral("#include \"SuperPoint.h\"")))
        << "CanvasWidget only reads serialized features and should not pull LibTorch extractor headers into the view.";
    EXPECT_FALSE(source.contains(QStringLiteral("QtTorchMacroGuard")));
    EXPECT_FALSE(source.contains(QStringLiteral("NEED_RESTORE_SLOTS")));
    EXPECT_FALSE(source.contains(QStringLiteral("#include \"FeatureFileIO.h\"")));
    EXPECT_FALSE(source.contains(QStringLiteral("#include \"FeatureOutput.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"LayerFeatureLoader.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("loadFeatureKeypointsFromFile")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include <opencv2/imgcodecs.hpp>")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include <opencv2/imgproc.hpp>")));
}

TEST(CodeStyleTest, GuiSupportFilesUseSpacesInsteadOfTabs)
{
    const QStringList files = {
        QStringLiteral("src/gui/config/settings/GuiSettingsStore.h"),
        QStringLiteral("src/gui/config/settings/ProjectDialogJsonSettingBase.cpp"),
        QStringLiteral("src/gui/config/settings/ProjectDialogJsonSettingBase.h"),
    };

    for (const QString &path : files)
    {
        const QString source = readProjectSourceFile(path);
        ASSERT_FALSE(source.isEmpty()) << qPrintable(path);
        EXPECT_FALSE(source.contains(QChar('\t'))) << qPrintable(path) << " should use spaces instead of tabs.";
    }
}

TEST(CodeStyleTest, SettingsFilesUseLowerCamelPrivateMemberNames)
{
    const QString projectDialogHeader =
        readProjectSourceFile(QStringLiteral("src/gui/config/settings/ProjectDialogJsonSettingBase.h"));
    const QString dialogStoreHeader =
        readProjectSourceFile(QStringLiteral("src/gui/config/settings/DialogSettingStore.h"));
    const QString projectDialogSource =
        readProjectSourceFile(QStringLiteral("src/gui/config/settings/ProjectDialogJsonSettingBase.cpp"));
    const QString dialogStoreSource =
        readProjectSourceFile(QStringLiteral("src/gui/config/settings/DialogSettingStore.cpp"));
    ASSERT_FALSE(projectDialogHeader.isEmpty());
    ASSERT_FALSE(dialogStoreHeader.isEmpty());
    ASSERT_FALSE(projectDialogSource.isEmpty());
    ASSERT_FALSE(dialogStoreSource.isEmpty());

    EXPECT_TRUE(projectDialogHeader.contains(QStringLiteral("QString _plascanPath;")));
    EXPECT_FALSE(projectDialogHeader.contains(QStringLiteral("m_plascanPath")));
    EXPECT_FALSE(projectDialogSource.contains(QStringLiteral("m_plascanPath")));

    EXPECT_TRUE(dialogStoreHeader.contains(QStringLiteral("QString _dialogKey;")));
    EXPECT_TRUE(dialogStoreHeader.contains(QStringLiteral("return _dialogKey;")));
    EXPECT_TRUE(dialogStoreSource.contains(QStringLiteral(", _dialogKey(dialogKey.trimmed())")));
    EXPECT_FALSE(dialogStoreHeader.contains(QStringLiteral("m_dialogKey")));
    EXPECT_FALSE(dialogStoreSource.contains(QStringLiteral("m_dialogKey")));
}

TEST(CodeStyleTest, ProjectConfigManagersUseLowerCamelPrivateMemberNames)
{
    const QString configHeader = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectConfigManager.h"));
    const QString configSource = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectConfigManager.cpp"));
    const QString uiHeader = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectUiConfigManager.h"));
    const QString uiSource = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectUiConfigManager.cpp"));
    const QString workflowHeader =
        readProjectSourceFile(QStringLiteral("src/gui/config/ProjectWorkflowConfigManager.h"));
    const QString workflowSource =
        readProjectSourceFile(QStringLiteral("src/gui/config/ProjectWorkflowConfigManager.cpp"));
    ASSERT_FALSE(configHeader.isEmpty());
    ASSERT_FALSE(configSource.isEmpty());
    ASSERT_FALSE(uiHeader.isEmpty());
    ASSERT_FALSE(uiSource.isEmpty());
    ASSERT_FALSE(workflowHeader.isEmpty());
    ASSERT_FALSE(workflowSource.isEmpty());

    EXPECT_TRUE(configHeader.contains(QStringLiteral("QJsonObject _config;")));
    EXPECT_TRUE(configHeader.contains(QStringLiteral("return _config;")));
    EXPECT_TRUE(configHeader.contains(QStringLiteral("_config = data;")));
    EXPECT_FALSE(configHeader.contains(QStringLiteral("m_config")));
    EXPECT_FALSE(configSource.contains(QStringLiteral("m_config")));

    EXPECT_TRUE(uiHeader.contains(QStringLiteral("QJsonObject _ui;")));
    EXPECT_TRUE(uiHeader.contains(QStringLiteral("_ui = data;")));
    EXPECT_TRUE(uiHeader.contains(QStringLiteral("return _ui;")));
    EXPECT_FALSE(uiHeader.contains(QStringLiteral("m_ui")));
    EXPECT_FALSE(uiSource.contains(QStringLiteral("m_ui")));

    EXPECT_TRUE(workflowHeader.contains(QStringLiteral("QJsonObject _workflow;")));
    EXPECT_TRUE(workflowHeader.contains(QStringLiteral("_workflow = data;")));
    EXPECT_TRUE(workflowHeader.contains(QStringLiteral("return _workflow;")));
    EXPECT_FALSE(workflowHeader.contains(QStringLiteral("m_workflow")));
    EXPECT_FALSE(workflowSource.contains(QStringLiteral("m_workflow")));
}

TEST(CodeStyleTest, AppConfigManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/config/AppConfigManager.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/config/AppConfigManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("WindowStateManager _windowState;")));
    EXPECT_TRUE(header.contains(QStringLiteral("RecentProjectsManager _recentProjects;")));
    EXPECT_TRUE(header.contains(QStringLiteral("FileDialogStateManager _fileDialogs;")));
    EXPECT_TRUE(header.contains(QStringLiteral("return &_windowState;")));
    EXPECT_TRUE(header.contains(QStringLiteral("return &_recentProjects;")));
    EXPECT_TRUE(header.contains(QStringLiteral("return &_fileDialogs;")));
    EXPECT_TRUE(source.contains(QStringLiteral(", _windowState(this)")));
    EXPECT_TRUE(source.contains(QStringLiteral(", _recentProjects(this)")));
    EXPECT_TRUE(source.contains(QStringLiteral(", _fileDialogs(this)")));

    const QStringList oldNames = {
        QStringLiteral("m_windowState"),
        QStringLiteral("m_recentProjects"),
        QStringLiteral("m_fileDialogs"),
    };
    for (const QString &oldName : oldNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, PlascanArchiveUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/common/project/PlascanArchive.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/common/project/PlascanArchive.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QString _path;"),
        QStringLiteral("bool _valid{false};"),
        QStringLiteral("void *_impl{};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_path"),
        QStringLiteral("m_valid"),
        QStringLiteral("m_impl"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(")"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(";"))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectResourceCleanupServiceSourceKeepsLinesWithinStyleLimit)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/services/ProjectResourceCleanupService.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "ProjectResourceCleanupService.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, ProjectFilesManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/project/data/ProjectFilesManager.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/data/ProjectFilesManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QJsonObject _coreFiles;"),
        QStringLiteral("QJsonObject _resultFiles;"),
        QStringLiteral("bool _resultsDirty = false;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_coreFiles"),
        QStringLiteral("m_resultFiles"),
        QStringLiteral("m_resultsDirty"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectDataUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/project/data/ProjectData.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/data/ProjectData.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QString _projectPath;"),
        QStringLiteral("mutable ProjectFilesManager _filesManager;"),
        QStringLiteral("ProjectConfigManager _configManager;"),
        QStringLiteral("bool _isDirty = false;"),
        QStringLiteral("mutable bool _resultsLoaded = false;"),
        QStringLiteral("QTimer *_archiveSyncTimer{};"),
        QStringLiteral("bool _resultsDirtyForArchive{false};"),
        QStringLiteral("bool _coreFileDirtyForArchive{false};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_projectPath"),
        QStringLiteral("m_filesManager"),
        QStringLiteral("m_configManager"),
        QStringLiteral("m_isDirty"),
        QStringLiteral("m_resultsLoaded"),
        QStringLiteral("m_archiveSyncTimer"),
        QStringLiteral("m_resultsDirtyForArchive"),
        QStringLiteral("m_coreFileDirtyForArchive"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectTerrainProductsManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.h"));
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("ProjectManager *_owner = nullptr;"),
        QStringLiteral("ProjectData *_projectData = nullptr;"),
        QStringLiteral("QWidget *_parentWidget = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_owner"),
        QStringLiteral("m_projectData"),
        QStringLiteral("m_parentWidget"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, WorkspacePanelControllerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/WorkspacePanelController.h"));
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/WorkspacePanelController.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QVector<Entry> _entries;")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _applying{};")));
    EXPECT_FALSE(header.contains(QStringLiteral("m_entries")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_entries")));
}

TEST(CodeStyleTest, ReferencePanelWidgetUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/ReferencePanelWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/ReferencePanelWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QTableWidget *_table{};"),
        QStringLiteral("QPushButton *_exactImportBtn{};"),
        QStringLiteral("QPushButton *_batchImportBtn{};"),
        QStringLiteral("QPushButton *_clearCameraBtn{};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_table"),
        QStringLiteral("m_exactImportBtn"),
        QStringLiteral("m_batchImportBtn"),
        QStringLiteral("m_clearCameraBtn"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, TaskStatusWidgetUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/TaskStatusWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/TaskStatusWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QLabel *_statusLabel = nullptr;"),
        QStringLiteral("QProgressBar *_progressBar = nullptr;"),
        QStringLiteral("QToolButton *_cancelButton = nullptr;"),
        QStringLiteral("QString _cancelText;"),
        QStringLiteral("QString _cancellingText;"),
        QStringLiteral("bool _active = false;"),
        QStringLiteral("bool _cancelling = false;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_statusLabel"),
        QStringLiteral("m_progressBar"),
        QStringLiteral("m_cancelButton"),
        QStringLiteral("m_cancelText"),
        QStringLiteral("m_cancellingText"),
        QStringLiteral("m_active"),
        QStringLiteral("m_cancelling"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, DisparityHeatmapOverlayUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("cv::Mat _disparity;"),
        QStringLiteral("QImage  _heatmapImage;"),
        QStringLiteral("QPixmap _heatmap;"),
        QStringLiteral("float   _opacity     = 0.6f;"),
        QStringLiteral("float   _dispMin     = 0.0f;"),
        QStringLiteral("float   _dispMax     = 256.0f;"),
        QStringLiteral("bool    _autoRange   = true;"),
        QStringLiteral("int     _colormap    = 2;"),
        QStringLiteral("bool    _showInvalid = false;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_disparity"),
        QStringLiteral("m_heatmapImage"),
        QStringLiteral("m_heatmap"),
        QStringLiteral("m_opacity"),
        QStringLiteral("m_dispMin"),
        QStringLiteral("m_dispMax"),
        QStringLiteral("m_autoRange"),
        QStringLiteral("m_colormap"),
        QStringLiteral("m_showInvalid"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, MatchLineOverlayUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/MatchLineOverlay.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/MatchLineOverlay.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("ImageViewWidget *_leftView;"),
        QStringLiteral("ImageViewWidget *_rightView;"),
        QStringLiteral("QVector<QPointF> _ptsA;"),
        QStringLiteral("QVector<QPointF> _ptsB;"),
        QStringLiteral("QVector<bool> _inlierMask;"),
        QStringLiteral("QColor _lineColor;"),
        QStringLiteral("qreal _lineWidth;"),
        QStringLiteral("qreal _opacity;"),
        QStringLiteral("int _maxDisplayCount;"),
        QStringLiteral("bool _showOnlyInliers;"),
        QStringLiteral("bool _showEndPoints;"),
        QStringLiteral("bool _rainbowMode;"),
        QStringLiteral("bool _showOnlyHighlighted;"),
        QStringLiteral("QVector<int> _highlightIndices;"),
        QStringLiteral("mutable QVector<int> _cachedVisibleMatches;"),
        QStringLiteral("mutable bool _cacheValid;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_leftView"),
        QStringLiteral("m_rightView"),
        QStringLiteral("m_ptsA"),
        QStringLiteral("m_ptsB"),
        QStringLiteral("m_inlierMask"),
        QStringLiteral("m_lineColor"),
        QStringLiteral("m_lineWidth"),
        QStringLiteral("m_opacity"),
        QStringLiteral("m_maxDisplayCount"),
        QStringLiteral("m_showOnlyInliers"),
        QStringLiteral("m_showEndPoints"),
        QStringLiteral("m_rainbowMode"),
        QStringLiteral("m_showOnlyHighlighted"),
        QStringLiteral("m_highlightIndices"),
        QStringLiteral("m_cachedVisibleMatches"),
        QStringLiteral("m_cacheValid"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ImageViewWidgetUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/ImageViewWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/ImageViewWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QGraphicsView *_view;"),
        QStringLiteral("QGraphicsScene *_scene;"),
        QStringLiteral("QGraphicsPixmapItem *_imageItem;"),
        QStringLiteral("QVector<QPointF> _matchPoints;"),
        QStringLiteral("QVector<QGraphicsEllipseItem*> _pointItems;"),
        QStringLiteral("QString _imagePath;"),
        QStringLiteral("int _highlightedIndex;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_view"),
        QStringLiteral("m_scene"),
        QStringLiteral("m_imageItem"),
        QStringLiteral("m_matchPoints"),
        QStringLiteral("m_pointItems"),
        QStringLiteral("m_imagePath"),
        QStringLiteral("m_highlightedIndex"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }

    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_view"))) << "Qt Designer object name must stay stable";
}

TEST(CodeStyleTest, DualImageViewerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QPointer<ImageViewWidget> _leftView;"),
        QStringLiteral("QPointer<ImageViewWidget> _rightView;"),
        QStringLiteral("QPointer<MatchLineOverlay> _overlay;"),
        QStringLiteral("QPointer<DisparityHeatmapOverlay> _disparityOverlay;"),
        QStringLiteral("int _overlayMode = 0;"),
        QStringLiteral("bool _syncEnabled;"),
        QStringLiteral("bool _syncing;"),
        QStringLiteral("QVector<QPointF> _matchPtsA;"),
        QStringLiteral("QVector<QPointF> _matchPtsB;"),
        QStringLiteral("QTimer *_overlayUpdateTimer;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_leftView"),
        QStringLiteral("m_rightView"),
        QStringLiteral("m_overlay"),
        QStringLiteral("m_disparityOverlay"),
        QStringLiteral("m_overlayMode"),
        QStringLiteral("m_syncEnabled"),
        QStringLiteral("m_syncing"),
        QStringLiteral("m_matchPtsA"),
        QStringLiteral("m_matchPtsB"),
        QStringLiteral("m_overlayUpdateTimer"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }

    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_leftView"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_rightView"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_splitter"))) << "Qt Designer object name must stay stable";
}

TEST(CodeStyleTest, CanvasWidgetUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("LayerRenderer *_layerRenderer{};"),
        QStringLiteral("bool _showInterestPoints{true};"),
        QStringLiteral("QString _activeFeatureSuffix{QStringLiteral(\".sp\")};"),
        QStringLiteral("LayerRenderer::FeatureDisplayOptions _currentFeatureOpts;"),
        QStringLiteral("QString _currentImagePath;"),
        QStringLiteral("QFutureWatcher<QImage> *_imageWatcher{nullptr};"),
        QStringLiteral("std::map<QString, std::pair<QDateTime, std::vector<cv::KeyPoint>>> _spCache;"),
        QStringLiteral("int _featureLoadGeneration{0};"),
        QStringLiteral("double _zoomFactor{1.0};"),
        QStringLiteral("const double _zoomStep{1.15};"),
        QStringLiteral("const double _zoomMin{0.05};"),
        QStringLiteral("const double _zoomMax{50.0};"),
        QStringLiteral("bool _isPanning{false};"),
        QStringLiteral("QPoint _lastPanPoint{};"),
        QStringLiteral("const int _panThreshold{4};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    EXPECT_TRUE(header.contains(QStringLiteral("return _activeFeatureSuffix;")))
        << "CanvasWidget accessor should use the renamed member";

    const QStringList oldMemberNames = {
        QStringLiteral("m_layerRenderer"),
        QStringLiteral("m_showInterestPoints"),
        QStringLiteral("m_activeFeatureSuffix"),
        QStringLiteral("m_currentFeatureOpts"),
        QStringLiteral("m_currentImagePath"),
        QStringLiteral("m_spWatcher"),
        QStringLiteral("m_imageWatcher"),
        QStringLiteral("m_spCache"),
        QStringLiteral("m_zoomFactor"),
        QStringLiteral("m_zoomStep"),
        QStringLiteral("m_zoomMin"),
        QStringLiteral("m_zoomMax"),
        QStringLiteral("m_isPanning"),
        QStringLiteral("m_lastPanPoint"),
        QStringLiteral("m_panThreshold"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(")"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(";"))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, CanvasWidgetSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "CanvasWidget.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, ObservationNetworkViewUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/ObservationNetworkView.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/ObservationNetworkView.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QGraphicsScene *_scene = nullptr;"),
        QStringLiteral("xjw::ObservationNetwork _net;"),
        QStringLiteral("QVector<QPointF> _pos;"),
        QStringLiteral("QVector<double> _nodeRadii;"),
        QStringLiteral("QVector<int> _visibleEdgeIndices;"),
        QStringLiteral("QVector<int> _visibleLabelIndices;"),
        QStringLiteral("QVector<QVector<int>> _nodeEdgeAdjacency;"),
        QStringLiteral("QTimer *_forceTimer = nullptr;"),
        QStringLiteral("int _forceIter = 0;"),
        QStringLiteral("double _temp = 0.0;"),
        QStringLiteral("bool _autoFitPending = false;"),
        QStringLiteral("int _selectedNodeIndex = -1;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_scene"),
        QStringLiteral("m_net"),
        QStringLiteral("m_pos"),
        QStringLiteral("m_nodeRadii"),
        QStringLiteral("m_visibleEdgeIndices"),
        QStringLiteral("m_visibleLabelIndices"),
        QStringLiteral("m_nodeEdgeAdjacency"),
        QStringLiteral("m_forceTimer"),
        QStringLiteral("m_forceIter"),
        QStringLiteral("m_temp"),
        QStringLiteral("m_autoFitPending"),
        QStringLiteral("m_selectedNodeIndex"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(")"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(";"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("["))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, WorkspaceCenterWidgetUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("Ui::WorkspaceCenterWidget *_ui = nullptr;"),
        QStringLiteral("QPushButton *_modelBtn = nullptr;"),
        QStringLiteral("QPushButton *_imageBtn = nullptr;"),
        QStringLiteral("QPushButton *_compareBtn = nullptr;"),
        QStringLiteral("QPushButton *_obsNetBtn = nullptr;"),
        QStringLiteral("QStackedWidget *_stack = nullptr;"),
        QStringLiteral("CameraSceneWidget *_modelView = nullptr;"),
        QStringLiteral("CanvasWidget *_canvas = nullptr;"),
        QStringLiteral("DualImageViewer *_dualImageViewer = nullptr;"),
        QStringLiteral("ObservationNetworkView *_obsNetView = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_ui"),
        QStringLiteral("m_modelBtn"),
        QStringLiteral("m_imageBtn"),
        QStringLiteral("m_compareBtn"),
        QStringLiteral("m_obsNetBtn"),
        QStringLiteral("m_stack"),
        QStringLiteral("m_modelView"),
        QStringLiteral("m_canvas"),
        QStringLiteral("m_dualImageViewer"),
        QStringLiteral("m_obsNetView"),
        QStringLiteral("m_henuBrandBadge"),
        QStringLiteral("m_henuBrandSeal"),
        QStringLiteral("m_henuBrandName"),
        QStringLiteral("m_henuBrandSubTitle"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }

    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_modelBtn"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_imageBtn"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_compareBtn"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_obsNetBtn"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_stack"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_modelView"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_canvas"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_dualImageViewer"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("_ui->m_obsNetView"))) << "Qt Designer object name must stay stable";
}

TEST(CodeStyleTest, LogPanelUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/panels/LogPanel.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/panels/LogPanel.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QTextEdit *_text{nullptr};"),
        QStringLiteral("QComboBox *_levelCombo{nullptr};"),
        QStringLiteral("QPushButton *_clearBtn{nullptr};"),
        QStringLiteral("QPushButton *_saveBtn{nullptr};"),
        QStringLiteral("Logger::Level _displayLevel{Logger::Debug};"),
        QStringLiteral("int _sinkId{0};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_text"),
        QStringLiteral("m_levelCombo"),
        QStringLiteral("m_clearBtn"),
        QStringLiteral("m_saveBtn"),
        QStringLiteral("m_displayLevel"),
        QStringLiteral("m_sinkId"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }

    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_levelCombo"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_clearBtn"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_saveBtn"))) << "Qt Designer object name must stay stable";
    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_text"))) << "Qt Designer object name must stay stable";
}

TEST(CodeStyleTest, DataTreeWidgetUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QTreeView *_view{};"),
        QStringLiteral("QStandardItemModel *_model{};"),
        QStringLiteral("QString _currentPlascanPath{};"),
        QStringLiteral("QJsonObject _lastMeta{};"),
        QStringLiteral("QStringList _transientModels{};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_view"),
        QStringLiteral("m_model"),
        QStringLiteral("m_currentPlascanPath"),
        QStringLiteral("m_lastMeta"),
        QStringLiteral("m_transientModels"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }

    EXPECT_TRUE(source.contains(QStringLiteral("ui.m_view"))) << "Qt Designer object name must stay stable";
}

TEST(CodeStyleTest, LayerRendererUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QGraphicsScene *_scene{};"),
        QStringLiteral("QList<QGraphicsPixmapItem *> _layers{};"),
        QStringLiteral("QList<QGraphicsItem *> _featureItems{};"),
        QStringLiteral("QList<QGraphicsItem *> _matchItems{};"),
        QStringLiteral("QRectF _imageBounds{};"),
        QStringLiteral("QString _currentProjectPath;"),
        QStringLiteral("FeatureDisplayOptions _featureOpts;"),
        QStringLiteral("MatchDisplayOptions _matchOpts;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_scene"),
        QStringLiteral("m_layers"),
        QStringLiteral("m_featureItems"),
        QStringLiteral("m_matchItems"),
        QStringLiteral("m_imageBounds"),
        QStringLiteral("m_currentProjectPath"),
        QStringLiteral("m_featureOpts"),
        QStringLiteral("m_matchOpts"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, CameraModel3DDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QRegularExpression legacyMemberPattern(QStringLiteral("\\bm_[A-Za-z0-9_]+"));
    const QString sourceWithoutGeneratedUiObjects = QString(source)
        .replace(QStringLiteral("form.m_scene"), QStringLiteral("form.scene"))
        .replace(QStringLiteral("form.m_summaryLabel"), QStringLiteral("form.summaryLabel"));
    EXPECT_FALSE(header.contains(legacyMemberPattern))
        << "CameraModel3DDialog private members should use _lowerCamelCase.";
    EXPECT_FALSE(sourceWithoutGeneratedUiObjects.contains(legacyMemberPattern))
        << "CameraModel3DDialog source should not reference m_ private members.";
    EXPECT_TRUE(header.contains(QStringLiteral("ProjectManager *_projectManager = nullptr;")));
    EXPECT_TRUE(source.contains(QStringLiteral("form.m_scene")));
    EXPECT_TRUE(source.contains(QStringLiteral("form.m_summaryLabel")));
}

TEST(CodeStyleTest, LayerRendererHeaderKeepsLinesWithinStyleLimit)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    ASSERT_FALSE(header.isEmpty());

    const QStringList lines = header.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "LayerRenderer.h:" << (i + 1) << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, LayerRendererSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "LayerRenderer.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, LayerOverlayItemsUsesLowerCamelPrivateMemberNames)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/views/LayerOverlayItems.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("std::vector<cv::KeyPoint> _keypoints;"),
        QStringLiteral("LayerRenderer::FeatureDisplayOptions _options;"),
        QStringLiteral("QRectF _bounds;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(source.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_keypoints"),
        QStringLiteral("m_options"),
        QStringLiteral("m_bounds"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(")"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(";"))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, LoggerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/common/log/Logger.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/common/log/Logger.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("mutable std::mutex _mutex;"),
        QStringLiteral("std::string _logDir;"),
        QStringLiteral("std::string _logFilePath;"),
        QStringLiteral("std::ofstream _file;"),
        QStringLiteral("std::uintmax_t _maxSize{5u * 1024u * 1024u};"),
        QStringLiteral("int _maxFiles{3};"),
        QStringLiteral("int _nextSinkId{1};"),
        QStringLiteral("std::unordered_map<int, SinkCallback> _sinks;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_mutex"),
        QStringLiteral("m_logDir"),
        QStringLiteral("m_logFilePath"),
        QStringLiteral("m_file"),
        QStringLiteral("m_maxSize"),
        QStringLiteral("m_maxFiles"),
        QStringLiteral("m_nextSinkId"),
        QStringLiteral("m_sinks"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectDashboardWidgetUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/ProjectDashboardWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/ProjectDashboardWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QLabel *_summaryLabel = nullptr;"),
        QStringLiteral("QLabel *_referenceLabel = nullptr;"),
        QStringLiteral("QLabel *_taskLabel = nullptr;"),
        QStringLiteral("QTableWidget *_taskTable = nullptr;"),
        QStringLiteral("QTableWidget *_referenceTable = nullptr;"),
        QStringLiteral("QTableWidget *_workflowTable = nullptr;"),
        QStringLiteral("QTableWidget *_qualityTable = nullptr;"),
        QStringLiteral("QTableWidget *_qualityAlertTable = nullptr;"),
        QStringLiteral("QTableWidget *_reportTable = nullptr;"),
        QStringLiteral("QJsonArray _taskSnapshots;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_summaryLabel"),
        QStringLiteral("m_referenceLabel"),
        QStringLiteral("m_taskLabel"),
        QStringLiteral("m_taskTable"),
        QStringLiteral("m_referenceTable"),
        QStringLiteral("m_workflowTable"),
        QStringLiteral("m_qualityTable"),
        QStringLiteral("m_qualityAlertTable"),
        QStringLiteral("m_reportTable"),
        QStringLiteral("m_taskSnapshots"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, CameraStoresGroupedStateAsSingleSourceOfTruth)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/camera/Camera.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/camera/Camera.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("Intrinsics _intrinsics;")));
    EXPECT_TRUE(header.contains(QStringLiteral("Distortion _distortion;")));
    EXPECT_TRUE(header.contains(QStringLiteral("Pose _pose;")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _isLoaded = false;")));

    const QStringList oldMemberNames = {
        QStringLiteral("_fu"),
        QStringLiteral("_fv"),
        QStringLiteral("_cu"),
        QStringLiteral("_cv"),
        QStringLiteral("_C"),
        QStringLiteral("_R"),
        QStringLiteral("_k1"),
        QStringLiteral("_k2"),
        QStringLiteral("_k3"),
        QStringLiteral("_p1"),
        QStringLiteral("_p2"),
        QStringLiteral("_pitch"),
        QStringLiteral("_u_dir"),
        QStringLiteral("_v_dir"),
        QStringLiteral("_depth_flipped_z"),
        QStringLiteral("_loaded"),
    };
    auto containsIdentifier = [](const QString &text, const QString &identifier)
    {
        const QString pattern = QStringLiteral("(?<![A-Za-z0-9_])%1(?![A-Za-z0-9_])")
                                    .arg(QRegularExpression::escape(identifier));
        return QRegularExpression(pattern).match(text).hasMatch();
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(containsIdentifier(header, oldName)) << qPrintable(oldName);
        EXPECT_FALSE(containsIdentifier(source, oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, CameraIsTheOnlyPublicCameraModel)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/camera/Camera.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/camera/Camera.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("Camera normalizedForPositiveDepth() const;")));
    EXPECT_TRUE(header.contains(QStringLiteral("Camera scaledIntrinsics(double scaleX, double scaleY) const;")));
    EXPECT_TRUE(source.contains(QStringLiteral("Camera Camera::normalizedForPositiveDepth() const")));
    EXPECT_TRUE(source.contains(QStringLiteral("Camera Camera::scaledIntrinsics")));
}

TEST(CodeStyleTest, MultiViewTrackBuilderUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/sfm/tracks/MultiViewTrackBuilder.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/sfm/tracks/MultiViewTrackBuilder.cpp"));
    const QString disjointSetHeader = readProjectSourceFile(QStringLiteral("src/core/sfm/common/DisjointSet.h"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(disjointSetHeader.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("std::vector<Edge> _edges;")));
    EXPECT_TRUE(header.contains(QStringLiteral("std::map<ImageId, std::vector<FeatureKeypoint>> _keypointsByImage;")));
    EXPECT_TRUE(source.contains(QStringLiteral("detail::DisjointSet disjointSet;")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::map<ObservationKey, int> indexByKey;")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::vector<ObservationKey> keys;")));
    EXPECT_TRUE(disjointSetHeader.contains(QStringLiteral("std::vector<int> _parent;")));
    EXPECT_TRUE(disjointSetHeader.contains(QStringLiteral("std::vector<int> _rank;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_edges"),
        QStringLiteral("m_indexByKey"),
        QStringLiteral("m_keys"),
        QStringLiteral("m_parent"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(disjointSetHeader.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, DenseMatchCoreUsesLowerCamelPrivateMemberNames)
{
    const QHash<QString, QStringList> expectedByHeader = {
        {QStringLiteral("src/core/dense_match/BlockMatcher.h"),
         {QStringLiteral("DenseMatchConfig _config;")}},
        {QStringLiteral("src/core/dense_match/DenseMatchService.h"),
         {QStringLiteral("DenseMatchConfig _config;"),
          QStringLiteral("cv::Mat _left, _right;")}},
        {QStringLiteral("src/core/dense_match/DisparityValidator.h"),
         {QStringLiteral("DenseMatchConfig _config;")}},
        {QStringLiteral("src/core/dense_match/SgmMatcher.h"),
         {QStringLiteral("DenseMatchConfig _config;")}},
        {QStringLiteral("src/core/dense_match/SubpixelRefiner.h"),
         {QStringLiteral("DenseMatchConfig _config;")}},
        {QStringLiteral("src/core/dense_match/opencv/OpenCVSgbmWrapper.h"),
         {QStringLiteral("DenseMatchConfig _config;")}},
    };

    for (auto it = expectedByHeader.cbegin(); it != expectedByHeader.cend(); ++it)
    {
        const QString header = readProjectSourceFile(it.key());
        ASSERT_FALSE(header.isEmpty()) << qPrintable(it.key());
        for (const QString &expectedMember : it.value())
        {
            EXPECT_TRUE(header.contains(expectedMember))
                << qPrintable(it.key() + QStringLiteral(": ") + expectedMember);
        }
    }

    const QStringList files = {
        QStringLiteral("src/core/dense_match/BlockMatcher.h"),
        QStringLiteral("src/core/dense_match/BlockMatcher.cpp"),
        QStringLiteral("src/core/dense_match/DenseMatchService.h"),
        QStringLiteral("src/core/dense_match/DenseMatchService.cpp"),
        QStringLiteral("src/core/dense_match/DisparityValidator.h"),
        QStringLiteral("src/core/dense_match/DisparityValidator.cpp"),
        QStringLiteral("src/core/dense_match/SgmMatcher.h"),
        QStringLiteral("src/core/dense_match/SgmMatcher.cpp"),
        QStringLiteral("src/core/dense_match/SubpixelRefiner.h"),
        QStringLiteral("src/core/dense_match/SubpixelRefiner.cpp"),
        QStringLiteral("src/core/dense_match/opencv/OpenCVSgbmWrapper.h"),
        QStringLiteral("src/core/dense_match/opencv/OpenCVSgbmWrapper.cpp"),
    };
    const QStringList oldMemberNames = {
        QStringLiteral("m_cfg"),
        QStringLiteral("m_left"),
        QStringLiteral("m_right"),
    };
    for (const QString &path : files)
    {
        const QString source = readProjectSourceFile(path);
        ASSERT_FALSE(source.isEmpty()) << qPrintable(path);
        for (const QString &oldName : oldMemberNames)
        {
            EXPECT_FALSE(source.contains(oldName)) << qPrintable(path + QStringLiteral(": ") + oldName);
        }
    }
}

TEST(CodeStyleTest, TorchFeatureWrappersUseLowerCamelPrivateMemberNames)
{
    const QHash<QString, QStringList> expectedByHeader = {
        {QStringLiteral("src/core/feature_extractors/disk/DiskExtractor.h"),
         {QStringLiteral("DiskConfig _config;"),
          QStringLiteral("torch::jit::script::Module _model;"),
          QStringLiteral("torch::Device _device{torch::kCPU};")}},
        {QStringLiteral("src/core/feature_extractors/aliked/AlikedExtractor.h"),
         {QStringLiteral("AlikedConfig _config;"),
          QStringLiteral("torch::jit::script::Module _model;"),
          QStringLiteral("torch::Device _device{torch::kCPU};")}},
        {QStringLiteral("src/core/feature_match/loftr/LoFTRMatcher.h"),
         {QStringLiteral("LoFTRConfig _config;"),
          QStringLiteral("torch::jit::script::Module _model;"),
          QStringLiteral("torch::Device _device{torch::kCPU};")}},
    };
    for (auto it = expectedByHeader.cbegin(); it != expectedByHeader.cend(); ++it)
    {
        const QString header = readProjectSourceFile(it.key());
        ASSERT_FALSE(header.isEmpty()) << qPrintable(it.key());
        for (const QString &expectedMember : it.value())
        {
            EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(it.key() + QStringLiteral(": ") + expectedMember);
        }
    }

    const QStringList files = {
        QStringLiteral("src/core/feature_extractors/disk/DiskExtractor.h"),
        QStringLiteral("src/core/feature_extractors/disk/DiskExtractor.cpp"),
        QStringLiteral("src/core/feature_extractors/aliked/AlikedExtractor.h"),
        QStringLiteral("src/core/feature_extractors/aliked/AlikedExtractor.cpp"),
        QStringLiteral("src/core/feature_match/loftr/LoFTRMatcher.h"),
        QStringLiteral("src/core/feature_match/loftr/LoFTRMatcher.cpp"),
    };
    const QStringList oldMemberNames = {
        QStringLiteral("m_cfg"),
        QStringLiteral("m_model"),
        QStringLiteral("m_device"),
    };
    for (const QString &path : files)
    {
        const QString source = readProjectSourceFile(path);
        ASSERT_FALSE(source.isEmpty()) << qPrintable(path);
        for (const QString &oldName : oldMemberNames)
        {
            EXPECT_FALSE(source.contains(oldName)) << qPrintable(path + QStringLiteral(": ") + oldName);
        }
    }
}

TEST(CodeStyleTest, ExtractorFactoryUsesLowerCamelPrivateMemberNames)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/feature_extractors/ExtractorFactory.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("std::string _algorithm;")));
    EXPECT_TRUE(source.contains(QStringLiteral("SuperPointConfig _config;")));

    EXPECT_FALSE(source.contains(QStringLiteral("m_algo")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_cfg")));
}

TEST(CodeStyleTest, MatcherFactoryUsesLowerCamelPrivateMemberNames)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/feature_match/MatcherFactory.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("std::string _algorithm;")));
    EXPECT_TRUE(source.contains(QStringLiteral("MatcherConfig _config;")));

    EXPECT_FALSE(source.contains(QStringLiteral("m_algo")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_cfg")));
}

TEST(CodeStyleTest, LightGlueMatcherHeaderKeepsLinesWithinStyleLimit)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/feature_match/lightglue/LightGlueMatcher.h"));
    ASSERT_FALSE(header.isEmpty());

    const QStringList lines = header.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "LightGlueMatcher.h:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, PointCloudPreprocessHeaderKeepsLinesWithinStyleLimit)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/mesh/PointCloudPreprocess.h"));
    ASSERT_FALSE(header.isEmpty());

    const QStringList lines = header.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "PointCloudPreprocess.h:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, IntersectionSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/intersection/Intersection.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "Intersection.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, EpipolarRectifierSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/EpipolarRectifier.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "EpipolarRectifier.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, OverlapAnalyzerSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/overlap/OverlapAnalyzer.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "OverlapAnalyzer.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, VocabularyOverlapRetrieverSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/overlap/VocabularyOverlapRetriever.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "VocabularyOverlapRetriever.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, BaInputBuilderSourceKeepsLinesWithinStyleLimit)
{
    const QStringList sources{
        QStringLiteral("src/core/sfm/project/BaInputBuilder.cpp"),
        QStringLiteral("src/core/sfm/project/ProjectMatchInputReader.cpp"),
        QStringLiteral("src/core/sfm/project/BaTrackBuilder.cpp"),
        QStringLiteral("src/core/sfm/project/SurveyControlBaAdapter.cpp"),
        QStringLiteral("src/core/sfm/project/MarkerBaAdapter.cpp"),
    };
    for (const QString &path : sources)
    {
        const QString source = readProjectSourceFile(path);
        ASSERT_FALSE(source.isEmpty()) << qPrintable(path);
        const QStringList lines = source.split(QLatin1Char('\n'));
        for (int i = 0; i < lines.size(); ++i)
        {
            EXPECT_LE(lines.at(i).size(), 120)
                << qPrintable(path) << ':' << (i + 1)
                << " has " << lines.at(i).size() << " characters";
        }
    }
}

TEST(CodeStyleTest, SuperPointBatchSourceKeepsLinesWithinStyleLimit)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/feature_extractors/superpoint/SuperPointBatch.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "SuperPointBatch.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, SuperPointSourceKeepsLinesWithinStyleLimit)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/feature_extractors/superpoint/SuperPoint.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "SuperPoint.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, ProjectModelManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ProjectManager *_owner = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("ProjectData *_projectData = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QWidget *_parentWidget = nullptr;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_owner"),
        QStringLiteral("m_projectData"),
        QStringLiteral("m_parentWidget"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectModelManagerSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "ProjectModelManager.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, ProjectReconstructionManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ProjectSparseReconstructionManager *_sparseManager = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("ProjectDenseReconstructionManager *_denseManager = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("ProjectModelManager *_modelManager = nullptr;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_sparseManager"),
        QStringLiteral("m_denseManager"),
        QStringLiteral("m_modelManager"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QWidget *_parent = nullptr;"),
        QStringLiteral("ProjectData *_projectData = nullptr;"),
        QStringLiteral("FileDialogStateManager *_fileDialogState = nullptr;"),
        QStringLiteral("ProjectReconstructionManager *_reconstructionManager = nullptr;"),
        QStringLiteral("ProjectTerrainProductsManager *_terrainProductsManager = nullptr;"),
        QStringLiteral("ProjectCameraSetupManager *_cameraSetupManager = nullptr;"),
        QStringLiteral("ProjectTaskDispatcher *_taskDispatcher = nullptr;"),
        QStringLiteral("ProjectUiCommands *_uiCommands = nullptr;"),
        QStringLiteral("std::shared_ptr<std::atomic<bool>> _atCancelFlag;"),
        QStringLiteral("QMap<QString, QJsonObject> _pendingBaCameraMeta;"),
        QStringLiteral("QMap<QString, QJsonObject> _pendingBaBeforeCameraMeta;"),
        QStringLiteral("QJsonObject _pendingBaResult;"),
        QStringLiteral("bool _hasPendingBaPreview = false;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_parent"),
        QStringLiteral("m_projectData"),
        QStringLiteral("m_fileDialogState"),
        QStringLiteral("m_reconstructionManager"),
        QStringLiteral("m_terrainProductsManager"),
        QStringLiteral("m_cameraSetupManager"),
        QStringLiteral("m_taskDispatcher"),
        QStringLiteral("m_uiCommands"),
        QStringLiteral("m_atCancelFlag"),
        QStringLiteral("m_pendingBaCameraMeta"),
        QStringLiteral("m_pendingBaBeforeCameraMeta"),
        QStringLiteral("m_pendingBaResult"),
        QStringLiteral("m_hasPendingBaPreview"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectSparseReconstructionManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectSparseReconstructionManager.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectSparseReconstructionManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ProjectManager *_owner = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("ProjectData *_projectData = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QWidget *_parentWidget = nullptr;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_owner"),
        QStringLiteral("m_projectData"),
        QStringLiteral("m_parentWidget"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ProjectDenseReconstructionManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.h"));
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedHeaderMembers = {
        QStringLiteral("ProjectManager *_owner = nullptr;"),
        QStringLiteral("ProjectData *_projectData = nullptr;"),
        QStringLiteral("QPointer<QObject> _activeMvsGenerator;"),
        QStringLiteral("std::shared_ptr<std::atomic_bool> _activeMvsCancelFlag;"),
    };
    for (const QString &expectedMember : expectedHeaderMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList expectedSourceMembers = {
        QStringLiteral("const std::vector<StoredDepthFrameRecord> &_records;"),
        QStringLiteral("const QMap<QString, xjw::Camera> &_cameraMap;"),
        QStringLiteral("xjw::mvs::FusionConfig _fusionConfig;"),
        QStringLiteral("int _viewCount = 0;"),
        QStringLiteral("int _fusionMaxImageDim = 0;"),
        QStringLiteral("int _capacity = 1;"),
        QStringLiteral("std::list<int> _lru;"),
        QStringLiteral("std::unordered_map<int, CacheEntry> _cache;"),
    };
    for (const QString &expectedMember : expectedSourceMembers)
    {
        EXPECT_TRUE(source.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_owner"),
        QStringLiteral("m_projectData"),
        QStringLiteral("m_parentWidget"),
        QStringLiteral("m_activeMvsGenerator"),
        QStringLiteral("m_activeMvsCancelFlag"),
        QStringLiteral("m_records"),
        QStringLiteral("m_camMap"),
        QStringLiteral("m_confidenceThreshold"),
        QStringLiteral("m_viewCount"),
        QStringLiteral("m_fusionMaxImageDim"),
        QStringLiteral("m_capacity"),
        QStringLiteral("m_lru"),
        QStringLiteral("m_cache"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, GroundBackProjectorUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/overlap/GroundBackProjector.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/overlap/GroundBackProjector.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("std::vector<std::array<double, 3>> _points;"),
        QStringLiteral("std::vector<DemKdTree2D::Point> _xyPoints;"),
        QStringLiteral("DemKdTree2D _index;"),
        QStringLiteral("double _meanHeight = 0.0;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_points"),
        QStringLiteral("m_xyPoints"),
        QStringLiteral("m_index"),
        QStringLiteral("m_meanHeight"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, SparseCloudValidatorUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/mvs/SparseCloudValidator.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/SparseCloudValidator.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("SparseCloudValidatorOptions _options;")));
    EXPECT_TRUE(header.contains(QStringLiteral(": _options(opts)")));
    EXPECT_FALSE(header.contains(QStringLiteral("m_opts")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_opts")));
}

TEST(CodeStyleTest, DepthMapFusionUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapFusion.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapFusion.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("StereoFusionConfig _config;")));
    EXPECT_TRUE(header.contains(QStringLiteral("std::vector<cv::Mat> _filteredDepths;")));
    EXPECT_FALSE(header.contains(QStringLiteral("m_config")));
    EXPECT_FALSE(header.contains(QStringLiteral("m_filteredDepths")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_config")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_filteredDepths")));
}

TEST(CodeStyleTest, AspPointCloudMetricsSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/AspPointCloudMetrics.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "AspPointCloudMetrics.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, DepthMapGeneratorUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("std::vector<CameraView> _views;"),
        QStringLiteral("SparseCloud _sparse;"),
        QStringLiteral("DepthGenConfig _config;"),
        QStringLiteral("std::atomic<bool> _cancelled{false};"),
        QStringLiteral("std::string _outputDir;"),
        QStringLiteral("std::vector<DepthFrameResult> _depthFrames;"),
        QStringLiteral("std::vector<uint8_t> _skipFrameMask;"),
        QStringLiteral("std::vector<cv::Mat> _grayCache;"),
        QStringLiteral("std::vector<cv::Mat> _validRegionMasks;"),
        QStringLiteral("std::vector<FrameMvsCache> _frameCaches;"),
        QStringLiteral("std::vector<uint64_t> _visibilityBits;"),
        QStringLiteral("std::vector<int> _pairCommonCounts;"),
        QStringLiteral("size_t _visibilityWordCount = 0;"),
        QStringLiteral("bool _frameCachesReady = false;"),
        QStringLiteral("QString _workspaceManifestPath;"),
        QStringLiteral("QString _depthConfigHash;"),
        QStringLiteral("MvsWorkspaceManifest _workspaceManifest;"),
        QStringLiteral("std::mutex _workspaceManifestMutex;"),
        QStringLiteral("std::vector<cv::Mat> _filteredDepths;"),
        QStringLiteral("mutable std::mutex   _filteredDepthsMutex;"),
    };
    for (const QString &member : expectedMembers)
    {
        EXPECT_TRUE(header.contains(member)) << qPrintable(member);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_views"),
        QStringLiteral("m_sparse"),
        QStringLiteral("m_config"),
        QStringLiteral("m_cancelled"),
        QStringLiteral("m_outputDir"),
        QStringLiteral("m_depthFrames"),
        QStringLiteral("m_skipFrameMask"),
        QStringLiteral("m_grayCache"),
        QStringLiteral("m_contentMasks"),
        QStringLiteral("m_frameCaches"),
        QStringLiteral("m_visibilityBits"),
        QStringLiteral("m_pairCommonCounts"),
        QStringLiteral("m_visibilityWordCount"),
        QStringLiteral("m_frameCachesReady"),
        QStringLiteral("m_workspaceManifestPath"),
        QStringLiteral("m_depthConfigHash"),
        QStringLiteral("m_workspaceManifest"),
        QStringLiteral("m_workspaceManifestMutex"),
        QStringLiteral("m_filteredDepths"),
        QStringLiteral("m_filteredDepthsMutex"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, LaserConstraintMapUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/lidar/LaserConstraintMap.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/lidar/LaserConstraintMap.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("std::vector<LaserPlaneSample> _samples;")));
    EXPECT_TRUE(header.contains(QStringLiteral("plapoint::search::SpatialKdTree<3, double> _index;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_samples"),
        QStringLiteral("m_index"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, StereoDenseCloudPipelineUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/mvs/StereoDenseCloudPipeline.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/StereoDenseCloudPipeline.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("StereoPipelineConfig _config;")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _cancelled = false;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_config"),
        QStringLiteral("m_cancelled"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, MvsWorkspaceManifestUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/mvs/MvsWorkspaceManifest.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/MvsWorkspaceManifest.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QString _configHash;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QVector<MvsDepthFrameRecord> _frames;")));
    EXPECT_FALSE(header.contains(QStringLiteral("m_configHash")));
    EXPECT_FALSE(header.contains(QStringLiteral("m_frames")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_configHash")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_frames")));
}

TEST(CodeStyleTest, TerrainProductManifestUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/terrain/TerrainProductManifest.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/core/terrain/TerrainProductManifest.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QVector<TerrainProductRecord> _records;")));
    EXPECT_FALSE(header.contains(QStringLiteral("m_records")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_records")));
}

TEST(CodeStyleTest, TerrainProductManifestDocumentsGuiAliasesAsStableCompatibilityFields)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/terrain/TerrainProductManifest.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("GUI compatibility aliases")));
    EXPECT_FALSE(source.contains(QStringLiteral("legacy GUI field names during the transition")))
        << "DEM/DOM alias fields are still consumed by GUI/project metadata; document them as compatibility aliases.";
}

TEST(CodeStyleTest, DomGeneratorSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/terrain/DomGenerator.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "DomGenerator.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, DemDomIOSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/terrain/DemDomIO.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "DemDomIO.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, TextureMapperSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mesh/TextureMapper.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "TextureMapper.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, ForwardIntersectionCheckDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/ForwardIntersectionCheckDialog.h"));
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/ForwardIntersectionCheckDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("ProjectManager *_projectManager{};"),
        QStringLiteral("QComboBox *_image1Combo{};"),
        QStringLiteral("QComboBox *_image2Combo{};"),
        QStringLiteral("QComboBox *_pickModeCombo{};"),
        QStringLiteral("QLabel *_hintLabel{};"),
        QStringLiteral("QPushButton *_deleteSelectedBtn{};"),
        QStringLiteral("QPushButton *_clearManualBtn{};"),
        QStringLiteral("QPushButton *_runBtn{};"),
        QStringLiteral("QTableWidget *_pairTable{};"),
        QStringLiteral("QTableWidget *_resultTable{};"),
        QStringLiteral("QTabWidget *_tabWidget{};"),
        QStringLiteral("DualImageViewer *_viewer{};"),
        QStringLiteral("QVector<QPointF> _manualPts1;"),
        QStringLiteral("QVector<QPointF> _manualPts2;"),
        QStringLiteral("QVector<QPointF> _currentPts1;"),
        QStringLiteral("QVector<QPointF> _currentPts2;"),
        QStringLiteral("QVector<xjw::Intersection::Result> _currentResults;"),
        QStringLiteral("bool _currentPairsEditable{false};"),
        QStringLiteral("int _pendingFirstSide{-1};"),
        QStringLiteral("QPointF _pendingFirstPoint{};"),
        QStringLiteral("int _currentHighlighted{-1};"),
        QStringLiteral("int _resultSortCol{-1};"),
        QStringLiteral("Qt::SortOrder _resultSortOrder{Qt::DescendingOrder};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_projectManager"),
        QStringLiteral("m_image1Combo"),
        QStringLiteral("m_image2Combo"),
        QStringLiteral("m_pickModeCombo"),
        QStringLiteral("m_hintLabel"),
        QStringLiteral("m_deleteSelectedBtn"),
        QStringLiteral("m_clearManualBtn"),
        QStringLiteral("m_runBtn"),
        QStringLiteral("m_pairTable"),
        QStringLiteral("m_resultTable"),
        QStringLiteral("m_tabWidget"),
        QStringLiteral("m_viewer"),
        QStringLiteral("m_manualPts1"),
        QStringLiteral("m_manualPts2"),
        QStringLiteral("m_currentPts1"),
        QStringLiteral("m_currentPts2"),
        QStringLiteral("m_currentResults"),
        QStringLiteral("m_currentPairsEditable"),
        QStringLiteral("m_pendingFirstSide"),
        QStringLiteral("m_pendingFirstPoint"),
        QStringLiteral("m_currentHighlighted"),
        QStringLiteral("m_resultSortCol"),
        QStringLiteral("m_resultSortOrder"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ForwardIntersectionCheckDialogSourceKeepsLinesWithinStyleLimit)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/ForwardIntersectionCheckDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "ForwardIntersectionCheckDialog.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, MatchPairSelectorDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("ProjectManager *_projectManager;"),
        QStringLiteral("QComboBox *_imageComboBox;"),
        QStringLiteral("QTableWidget *_matchTable;"),
        QStringLiteral("QPushButton *_viewDetailBtn;"),
        QStringLiteral("QPushButton *_refreshBtn;"),
        QStringLiteral("QLabel *_statusLabel;"),
        QStringLiteral("QStringList _allImages;"),
        QStringLiteral("QString _currentImage;"),
        QStringLiteral("QList<MatchInfo> _currentMatches;"),
        QStringLiteral("int _selectedMatchIndex;"),
        QStringLiteral("QString _matchDir;"),
        QStringLiteral("QTimer *_refreshTimer = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_projectManager"),
        QStringLiteral("m_imageComboBox"),
        QStringLiteral("m_matchTable"),
        QStringLiteral("m_viewDetailBtn"),
        QStringLiteral("m_refreshBtn"),
        QStringLiteral("m_statusLabel"),
        QStringLiteral("m_allImages"),
        QStringLiteral("m_currentImage"),
        QStringLiteral("m_currentMatches"),
        QStringLiteral("m_selectedMatchIndex"),
        QStringLiteral("m_matchDir"),
        QStringLiteral("m_refreshTimer"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ForwardIntersectionResultsDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/ForwardIntersectionResultsDialog.h"));
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/ForwardIntersectionResultsDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ProjectManager *_projectManager{};")));
    EXPECT_TRUE(header.contains(QStringLiteral("QComboBox *_pairCombo{};")));
    EXPECT_TRUE(header.contains(QStringLiteral("QJsonArray _allResults;")));

    const QStringList oldMemberUses = {
        QStringLiteral("m_projectManager"),
        QStringLiteral("m_pairCombo"),
        QStringLiteral("m_table"),
        QStringLiteral("m_detailTable"),
        QStringLiteral("m_allResults"),
    };
    for (const QString &oldName : oldMemberUses)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
    }

    EXPECT_FALSE(source.contains(QStringLiteral("m_projectManager")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_pairCombo->")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_table->")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_detailTable->")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_allResults")));
}

TEST(CodeStyleTest, ForwardIntersectionResultsDialogSourceKeepsLinesWithinStyleLimit)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/ForwardIntersectionResultsDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "ForwardIntersectionResultsDialog.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, CameraConvertDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraConvertDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraConvertDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QComboBox *_formatCombo = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLineEdit *_inputEdit = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLineEdit *_outputEdit = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QCheckBox *_overwriteCheck = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLabel *_statusLabel = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QTextEdit *_resultEdit = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPushButton *_runButton = nullptr;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_formatCombo"),
        QStringLiteral("m_inputEdit"),
        QStringLiteral("m_outputEdit"),
        QStringLiteral("m_overwriteCheck"),
        QStringLiteral("m_statusLabel"),
        QStringLiteral("m_resultEdit"),
        QStringLiteral("m_runButton"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("connect(") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, TextureMappingDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/TextureMappingDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/TextureMappingDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QComboBox *_blendCombo = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QComboBox *_texSizeCombo = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QComboBox *_uvMethodCombo = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QCheckBox *_colorCorrCheck = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QCheckBox *_ghostFilterCheck = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QDoubleSpinBox *_seamsMarginSpin = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QSpinBox *_paddingSpin = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QCheckBox *_keepUnmappedCheck = nullptr;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_blendCombo"),
        QStringLiteral("m_texSizeCombo"),
        QStringLiteral("m_uvMethodCombo"),
        QStringLiteral("m_colorCorrCheck"),
        QStringLiteral("m_ghostFilterCheck"),
        QStringLiteral("m_seamsMarginSpin"),
        QStringLiteral("m_paddingSpin"),
        QStringLiteral("m_keepUnmappedCheck"),
        QStringLiteral("m_threadsSpin"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("connect(") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, MapProjectDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/MapProjectDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/MapProjectDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QListWidget *_imageList = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLineEdit *_demEdit = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLineEdit *_outputEdit = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QDoubleSpinBox *_resolutionSpin = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QString _projectRoot;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_imageList"),
        QStringLiteral("m_demEdit"),
        QStringLiteral("m_outputEdit"),
        QStringLiteral("m_resolutionSpin"),
        QStringLiteral("m_projectRoot"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("connect(") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, SurveyControlDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/SurveyControlDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/SurveyControlDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QLabel *_summaryLabel = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLabel *_sourceLabel = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLabel *_statusLabel = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPushButton *_importCsvButton = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QTableWidget *_controlPointTable = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QTableWidget *_checkPointTable = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QTableWidget *_scaleBarTable = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject _metadata;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_summaryLabel"),
        QStringLiteral("m_sourceLabel"),
        QStringLiteral("m_statusLabel"),
        QStringLiteral("m_importCsvButton"),
        QStringLiteral("m_controlPointTable"),
        QStringLiteral("m_checkPointTable"),
        QStringLiteral("m_scaleBarTable"),
        QStringLiteral("m_metadata"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("connect(") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, CreateDemDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/CreateDemDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/CreateDemDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ProjectManager *_projectManager = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QStringList _availableImages;")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _running = false;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPushButton *_autoModeBtn = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPushButton *_manualModeBtn = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QStackedWidget *_modeStack = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QListWidget *_imageList = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLabel *_camStatusLabel = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("class QLineEdit *_denseEdit = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QProgressBar *_progressBar = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QLabel *_stageLabel = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPushButton *_runBtn = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPushButton *_closeBtn = nullptr;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_projectManager"),
        QStringLiteral("m_availableImages"),
        QStringLiteral("m_running"),
        QStringLiteral("m_autoModeBtn"),
        QStringLiteral("m_manualModeBtn"),
        QStringLiteral("m_modeStack"),
        QStringLiteral("m_imageList"),
        QStringLiteral("m_camStatusLabel"),
        QStringLiteral("m_denseEdit"),
        QStringLiteral("m_progressBar"),
        QStringLiteral("m_stageLabel"),
        QStringLiteral("m_runBtn"),
        QStringLiteral("m_closeBtn"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("connect(") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, WorkflowReportDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/application/WorkflowReportDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/application/WorkflowReportDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ChartType _type;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QStringList _labels;")));
    EXPECT_TRUE(header.contains(QStringLiteral("std::vector<double> _before;")));
    EXPECT_TRUE(header.contains(QStringLiteral("std::vector<double> _after;")));
    EXPECT_TRUE(header.contains(QStringLiteral("std::vector<double> _values;")));
    EXPECT_TRUE(header.contains(QStringLiteral("double _arcValue = 0.0;")));
    EXPECT_TRUE(header.contains(QStringLiteral("double _arcTotal = 1.0;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QString _arcLabel;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QString _unit;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QString _assetsDir;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QTabWidget *_tabs = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPushButton *_refreshBtn = nullptr;")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_type"),
        QStringLiteral("m_labels"),
        QStringLiteral("m_before"),
        QStringLiteral("m_after"),
        QStringLiteral("m_values"),
        QStringLiteral("m_arcValue"),
        QStringLiteral("m_arcTotal"),
        QStringLiteral("m_arcLabel"),
        QStringLiteral("m_unit"),
        QStringLiteral("m_assetsDir"),
        QStringLiteral("m_tabs"),
        QStringLiteral("m_refreshBtn"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, FeaturePointVisualizationDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QComboBox *_suffixCombo{nullptr};"),
        QStringLiteral("QCheckBox *_showPointsChk{nullptr};"),
        QStringLiteral("QCheckBox *_showScaleChk{nullptr};"),
        QStringLiteral("QCheckBox *_showOrientationChk{nullptr};"),
        QStringLiteral("QCheckBox *_useFillChk{nullptr};"),
        QStringLiteral("QSpinBox *_pointSizeSpin{nullptr};"),
        QStringLiteral("QDoubleSpinBox *_scaleMultiplierSpin{nullptr};"),
        QStringLiteral("QSlider *_opacitySlider{nullptr};"),
        QStringLiteral("QLabel *_opacityLabel{nullptr};"),
        QStringLiteral("QPushButton *_pointColorBtn{nullptr};"),
        QStringLiteral("QPushButton *_scaleColorBtn{nullptr};"),
        QStringLiteral("QPushButton *_orientColorBtn{nullptr};"),
        QStringLiteral("QComboBox *_markerShapeCombo{nullptr};"),
        QStringLiteral("QSpinBox *_maxDisplaySpin{nullptr};"),
        QStringLiteral("QCheckBox *_showTopScoresChk{nullptr};"),
        QStringLiteral("QLabel *_previewLabel{nullptr};"),
        QStringLiteral("QPushButton *_applyBtn{nullptr};"),
        QStringLiteral("QPushButton *_resetBtn{nullptr};"),
        QStringLiteral("QPushButton *_closeBtn{nullptr};"),
        QStringLiteral("QColor _pointColor{0, 120, 255};"),
        QStringLiteral("QColor _scaleColor{255, 255, 0};"),
        QStringLiteral("QColor _orientColor{255, 0, 0};"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_suffixCombo"),
        QStringLiteral("m_showPointsChk"),
        QStringLiteral("m_showScaleChk"),
        QStringLiteral("m_showOrientationChk"),
        QStringLiteral("m_useFillChk"),
        QStringLiteral("m_pointSizeSpin"),
        QStringLiteral("m_scaleMultiplierSpin"),
        QStringLiteral("m_opacitySlider"),
        QStringLiteral("m_opacityLabel"),
        QStringLiteral("m_pointColorBtn"),
        QStringLiteral("m_scaleColorBtn"),
        QStringLiteral("m_orientColorBtn"),
        QStringLiteral("m_markerShapeCombo"),
        QStringLiteral("m_maxDisplaySpin"),
        QStringLiteral("m_showTopScoresChk"),
        QStringLiteral("m_previewLabel"),
        QStringLiteral("m_applyBtn"),
        QStringLiteral("m_resetBtn"),
        QStringLiteral("m_closeBtn"),
        QStringLiteral("m_pointColor"),
        QStringLiteral("m_scaleColor"),
        QStringLiteral("m_orientColor"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, OverlapAnalysisDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/OverlapAnalysisDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/OverlapAnalysisDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("ProjectManager *_projectManager = nullptr;"),
        QStringLiteral("QListWidget *_imageList = nullptr;"),
        QStringLiteral("QLineEdit *_demPathEdit = nullptr;"),
        QStringLiteral("QCheckBox *_useFixedZCheck = nullptr;"),
        QStringLiteral("QDoubleSpinBox *_fixedZSpin = nullptr;"),
        QStringLiteral("QDoubleSpinBox *_neighborSpin = nullptr;"),
        QStringLiteral("QLabel *_summaryLabel = nullptr;"),
        QStringLiteral("QTableWidget *_resultTable = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_projectManager"),
        QStringLiteral("m_imageList"),
        QStringLiteral("m_demPathEdit"),
        QStringLiteral("m_useFixedZCheck"),
        QStringLiteral("m_fixedZSpin"),
        QStringLiteral("m_neighborSpin"),
        QStringLiteral("m_summaryLabel"),
        QStringLiteral("m_resultTable"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, OverlapAnalysisDialogSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/OverlapAnalysisDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "OverlapAnalysisDialog.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}

TEST(CodeStyleTest, ProjectCameraSetupManagerUsesLowerCamelPrivateMemberNames)
{
    const QString header =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectCameraSetupManager.h"));
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectCameraSetupManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("ProjectManager *_owner = nullptr;"),
        QStringLiteral("ProjectData *_projectData = nullptr;"),
        QStringLiteral("QWidget *_parentWidget = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_owner"),
        QStringLiteral("m_projectData"),
        QStringLiteral("m_parentWidget"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(DepthMapPersistenceTest, SavesFrameArtifactsBeforeFinalConsistencyPass)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int workerSave = source.indexOf(
        QStringLiteral("saveQueue.enqueue(i, res, QStringLiteral(\"初始\"))"));
    const int waitBeforeConsistency = source.indexOf(QStringLiteral("saveQueue.waitUntilIdle()"));
    const int consistencyPass = source.indexOf(QStringLiteral("crossCheckDepthConsistency();"));
    const int finalSave = source.indexOf(
        QStringLiteral("saveQueue.enqueue(i, res, QStringLiteral(\"过滤后\"))"));

    ASSERT_GE(workerSave, 0);
    ASSERT_GE(waitBeforeConsistency, 0);
    ASSERT_GE(consistencyPass, 0);
    ASSERT_GE(finalSave, 0);
    EXPECT_LT(workerSave, consistencyPass)
        << "Each completed depth frame should be persisted before the all-frame consistency pass.";
    EXPECT_LT(waitBeforeConsistency, consistencyPass)
        << "Initial async saves must drain before the consistency pass mutates depth maps.";
    EXPECT_LT(consistencyPass, finalSave)
        << "The final consistency-filtered depth maps should still overwrite the provisional artifacts.";
}

TEST(DisparityHeatmapOverlayTest, InvalidPixelsUseAlphaMask)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QImage::Format_RGBA8888")))
        << "Heatmap storage should carry alpha so invalid disparity can be transparent.";
    EXPECT_TRUE(source.contains(QStringLiteral("alphaRow[col] = validRow[col] ? 255 : 0")))
        << "Invalid disparity pixels should be masked out, not colorized as low disparity.";
    EXPECT_TRUE(source.contains(QStringLiteral("if (_showInvalid)")))
        << "setShowInvalid() must affect the rendered invalid-pixel state.";
    EXPECT_TRUE(header.contains(QStringLiteral("QImage heatmapImage() const")))
        << "Tests and callers need a mask-aware image accessor.";
}

TEST(SparseResultQualityTest, BuildsHistogramAndClassifiesPairwisePreview)
{
    QJsonArray points;
    points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                              {QStringLiteral("rms_reproj_px"), 1.0},
                              {QStringLiteral("min_tri_angle_deg"), 5.0}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                              {QStringLiteral("rms_reproj_px"), 2.0},
                              {QStringLiteral("min_tri_angle_deg"), 7.0}});

    const QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        2,
        false,
        xjw::gui::project::kSparseResultKindPairwisePreview);

    EXPECT_EQ(quality.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindPairwisePreview);
    EXPECT_EQ(quality.value(QStringLiteral("camera_count")).toInt(), 2);
    EXPECT_EQ(quality.value(QStringLiteral("point_count")).toInt(), 2);
    EXPECT_DOUBLE_EQ(quality.value(QStringLiteral("two_view_ratio")).toDouble(), 1.0);
    EXPECT_FALSE(quality.value(QStringLiteral("ba_applied")).toBool());

    const QJsonObject histogram = quality.value(QStringLiteral("track_len_histogram")).toObject();
    EXPECT_EQ(histogram.value(QStringLiteral("2")).toInt(), 2);
    EXPECT_TRUE(xjw::gui::project::isPairwisePreviewSparseResult(quality));
    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(quality));
}

TEST(SparseResultQualityTest, AcceptsFormalSfmWithMultiViewSupport)
{
    QJsonArray points;
    points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                              {QStringLiteral("rms_reproj_px"), 0.8}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 3},
                              {QStringLiteral("rms_reproj_px"), 0.6}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 4},
                              {QStringLiteral("rms_reproj_px"), 0.7}});

    const QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        3,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);

    EXPECT_TRUE(xjw::gui::project::isProductionSparseResult(quality));
    EXPECT_FALSE(xjw::gui::project::sparseResultBlockingReason(quality).contains(QStringLiteral("两视")));
    EXPECT_DOUBLE_EQ(quality.value(QStringLiteral("two_view_ratio")).toDouble(), 1.0 / 3.0);
    EXPECT_EQ(quality.value(QStringLiteral("median_track_len")).toInt(), 3);
}

TEST(SparseResultQualityTest, RejectsFormalSfmWhenAlmostAllTracksAreTwoView)
{
    QJsonArray points;
    for (int i = 0; i < 99; ++i)
    {
        points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                                  {QStringLiteral("rms_reproj_px"), 0.8}});
    }
    points.append(QJsonObject{{QStringLiteral("track_len"), 3},
                              {QStringLiteral("rms_reproj_px"), 0.9}});

    const QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        60,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);

    EXPECT_GT(quality.value(QStringLiteral("two_view_ratio")).toDouble(), 0.95);
    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(quality));
    EXPECT_TRUE(xjw::gui::project::sparseResultBlockingReason(quality).contains(QStringLiteral("两视")));
}

TEST(SparseResultQualityTest, RejectsFormalSfmWhenTooFewSelectedImagesRegister)
{
    QJsonArray points;
    points.append(QJsonObject{{QStringLiteral("track_len"), 3},
                              {QStringLiteral("rms_reproj_px"), 0.6}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 4},
                              {QStringLiteral("rms_reproj_px"), 0.7}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 5},
                              {QStringLiteral("rms_reproj_px"), 0.8}});

    QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        35,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);
    quality[QStringLiteral("input_image_count")] = 444;
    quality[QStringLiteral("registered_image_count")] = 35;

    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(quality));
    EXPECT_TRUE(xjw::gui::project::sparseResultBlockingReason(quality).contains(QStringLiteral("注册影像")));
}

TEST(SparseResultQualityTest, RejectsFormalSfmWhenQualityGateBlocksMvs)
{
    const QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        productionSparsePoints(),
        60,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction,
        QString(),
        QString(),
        80);

    QJsonObject sparseQuality;
    sparseQuality[QStringLiteral("quality_gate")] = QJsonObject{
        {QStringLiteral("acceptable_for_mvs"), false},
        {QStringLiteral("status"), QStringLiteral("warn")},
        {QStringLiteral("warnings"), QJsonArray{
             QStringLiteral("high_reprojection_error"),
             QStringLiteral("weak_triangulation_angle"),
             QStringLiteral("poor_observation_spatial_coverage")}}
    };

    QJsonObject record = xjw::gui::project::mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")}},
        quality);
    record[QStringLiteral("sfm_diagnostics")] = QJsonObject{
        {QStringLiteral("sparse_quality"), sparseQuality}
    };

    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(record));
    const QString reason = xjw::gui::project::sparseResultBlockingReason(record);
    EXPECT_TRUE(reason.contains(QStringLiteral("质量门控")));
    EXPECT_TRUE(reason.contains(QStringLiteral("重投影")));
    EXPECT_TRUE(reason.contains(QStringLiteral("三角角")));
    EXPECT_TRUE(reason.contains(QStringLiteral("空间覆盖")));
}

TEST(SparseResultQualityTest, LegacyTriangulationRecordsAreShownAsPairwisePreview)
{
    const QJsonObject legacyRecord{
        {QStringLiteral("operation"), QStringLiteral("triangulation")},
        {QStringLiteral("source"), QStringLiteral("triangulation")}
    };

    EXPECT_TRUE(xjw::gui::project::isPairwisePreviewSparseResult(legacyRecord));
    EXPECT_EQ(xjw::gui::project::sparseOperationDisplayName(QStringLiteral("triangulation")),
              QStringLiteral("两视预览云"));
}

TEST(SparsePointWorkflowUtilsTest, LocalOptimAcceptsExternalPlyWithoutSidecar)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString plyPath = QDir(tempDir.path()).filePath(QStringLiteral("external_sparse.ply"));
    writeMinimalPointCloudPly(plyPath, {
        {0.00, 0.0, 0.0},
        {0.02, 0.0, 0.0},
        {0.04, 0.0, 0.0},
        {5.00, 5.0, 5.0}
    });

    xjw::gui::project::SparsePointContext context;
    context.sparseCloudPath = plyPath;

    QJsonObject settings;
    settings[QStringLiteral("sourceKind")] = QStringLiteral("external_ply");
    settings[QStringLiteral("externalSparseCloudPath")] = plyPath;
    settings[QStringLiteral("voxelSize")] = 0.2;
    settings[QStringLiteral("minVoxelPoints")] = 2;
    settings[QStringLiteral("localReprojFilter")] = true;

    xjw::gui::project::SparsePointOperationResult result;
    QString errorMessage;
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    EXPECT_TRUE(xjw::gui::project::runSparsePointLocalOptim(context,
                                                           settings,
                                                           outputDir,
                                                           &result,
                                                           &errorMessage))
        << errorMessage.toStdString();

    EXPECT_EQ(result.inputCount, 4);
    EXPECT_EQ(result.outputCount, 3);
    EXPECT_TRUE(QFileInfo::exists(result.sparseCloudPath));
    EXPECT_TRUE(QFileInfo::exists(result.sidecarPath));

    QFile sidecarFile(result.sidecarPath);
    ASSERT_TRUE(sidecarFile.open(QIODevice::ReadOnly));
    const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();
    EXPECT_EQ(sidecar.value(QStringLiteral("point_count")).toInt(), 3);
    EXPECT_EQ(QDir::cleanPath(sidecar.value(QStringLiteral("source_ply")).toString()),
              QDir::cleanPath(plyPath));
    EXPECT_FALSE(sidecar.value(QStringLiteral("quality_metrics_available")).toBool(true));
}

TEST(SparsePointWorkflowUtilsTest, ProjectResultModeUsesSidecarWhenExternalPathSettingIsEmpty)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString sourcePlyPath = QDir(tempDir.path()).filePath(QStringLiteral("source_sparse.ply"));
    const QString sidecarPath = QDir(tempDir.path()).filePath(QStringLiteral("sparse_cloud_points.json"));
    writeMinimalPointCloudPly(sourcePlyPath, {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0}
    });
    writeMinimalSparseSidecar(sidecarPath, {1, 3, 4});

    xjw::gui::project::SparsePointContext context;
    context.sparseCloudPath = sourcePlyPath;
    context.sidecarPath = sidecarPath;

    QJsonObject settings;
    settings[QStringLiteral("sourceKind")] = QStringLiteral("project_result");
    settings[QStringLiteral("externalSparseCloudPath")] = QString();
    settings[QStringLiteral("filterByReprojError")] = false;
    settings[QStringLiteral("filterByTrackLen")] = true;
    settings[QStringLiteral("minTrackLen")] = 3;
    settings[QStringLiteral("filterByTriAngle")] = false;
    settings[QStringLiteral("filterByStatistical")] = false;
    settings[QStringLiteral("filterByDensity")] = false;

    xjw::gui::project::SparsePointOperationResult result;
    QString errorMessage;
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out_project"));
    EXPECT_TRUE(xjw::gui::project::runSparsePointOutlierRemoval(context,
                                                               settings,
                                                               outputDir,
                                                               &result,
                                                               &errorMessage))
        << errorMessage.toStdString();

    EXPECT_EQ(result.inputCount, 3);
    EXPECT_EQ(result.outputCount, 2);

    QFile sidecarFile(result.sidecarPath);
    ASSERT_TRUE(sidecarFile.open(QIODevice::ReadOnly));
    const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();
    EXPECT_EQ(sidecar.value(QStringLiteral("point_count")).toInt(), 2);
    EXPECT_TRUE(sidecar.value(QStringLiteral("quality_metrics_available")).toBool(false));
    EXPECT_FALSE(sidecar.contains(QStringLiteral("source_ply")));
}

TEST(SparsePointWorkflowUtilsTest, OutlierRemovalPreservesSourcePlyRgbWhenSidecarHasNoColors)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString sourcePlyPath = QDir(tempDir.path()).filePath(QStringLiteral("colored_sparse.ply"));
    const QString sidecarPath = QDir(tempDir.path()).filePath(QStringLiteral("sparse_cloud_points.json"));
    writeMinimalColoredPointCloudPly(sourcePlyPath,
                                     {
                                         {0.0, 0.0, 0.0},
                                         {1.0, 0.0, 0.0},
                                         {2.0, 0.0, 0.0}
                                     },
                                     {
                                         {10, 20, 30},
                                         {40, 50, 60},
                                         {70, 80, 90}
                                     });
    writeMinimalSparseSidecar(sidecarPath, {1, 3, 4});

    xjw::gui::project::SparsePointContext context;
    context.sparseCloudPath = sourcePlyPath;
    context.sidecarPath = sidecarPath;

    QJsonObject settings;
    settings[QStringLiteral("sourceKind")] = QStringLiteral("project_result");
    settings[QStringLiteral("externalSparseCloudPath")] = QString();
    settings[QStringLiteral("filterByReprojError")] = false;
    settings[QStringLiteral("filterByTrackLen")] = true;
    settings[QStringLiteral("minTrackLen")] = 3;
    settings[QStringLiteral("filterByTriAngle")] = false;
    settings[QStringLiteral("filterByStatistical")] = false;
    settings[QStringLiteral("filterByDensity")] = false;

    xjw::gui::project::SparsePointOperationResult result;
    QString errorMessage;
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out_colored"));
    ASSERT_TRUE(xjw::gui::project::runSparsePointOutlierRemoval(context,
                                                               settings,
                                                               outputDir,
                                                               &result,
                                                               &errorMessage))
        << errorMessage.toStdString();

    EXPECT_EQ(result.inputCount, 3);
    EXPECT_EQ(result.outputCount, 2);
    auto outputCloud = plapoint::io::readPly<float>(result.sparseCloudPath.toStdString());
    ASSERT_TRUE(outputCloud != nullptr);
    ASSERT_EQ(outputCloud->size(), 2u);
    ASSERT_TRUE(outputCloud->hasColors());
    EXPECT_EQ(outputCloud->colors()->getValue(0, 0), 40);
    EXPECT_EQ(outputCloud->colors()->getValue(0, 1), 50);
    EXPECT_EQ(outputCloud->colors()->getValue(0, 2), 60);
    EXPECT_EQ(outputCloud->colors()->getValue(1, 0), 70);
    EXPECT_EQ(outputCloud->colors()->getValue(1, 1), 80);
    EXPECT_EQ(outputCloud->colors()->getValue(1, 2), 90);
}

TEST(MainMenuTest, ToolsMenuExposesCameraConversionAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.cameraConvertAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("相机格式转换")));

    bool foundInToolsMenu = false;
    const QList<QMenu *> menus = window.menuBar()->findChildren<QMenu *>();
    for (QMenu *candidate : menus)
    {
        if (candidate && candidate->title() == QStringLiteral("工具"))
        {
            foundInToolsMenu = candidate->actions().contains(action);
            break;
        }
    }
    EXPECT_TRUE(foundInToolsMenu);
}

TEST(MainMenuTest, ToolsMenuExposesGenerateMaskAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.generateMaskAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("生成蒙版")));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionGenerateMask"));

    QMenu *toolsMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    ASSERT_NE(toolsMenu, nullptr);
    EXPECT_TRUE(toolsMenu->actions().contains(action));
}

TEST(MainMenuTest, ToolsMenuExposesReferenceDatasetImportAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.importReferenceDatasetAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("导入参考 DEM/LiDAR")));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionImportReferenceDataset"));

    bool foundInToolsMenu = false;
    const QList<QMenu *> menus = window.menuBar()->findChildren<QMenu *>();
    for (QMenu *candidate : menus)
    {
        if (candidate && candidate->title() == QStringLiteral("工具"))
        {
            foundInToolsMenu = candidate->actions().contains(action);
            break;
        }
    }
    EXPECT_TRUE(foundInToolsMenu);
}

TEST(MainMenuTest, ToolsMenuExposesReferenceQualityCheckAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.referenceQualityCheckAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("点云/DEM 精度检查")));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionReferenceQualityCheck"));

    QMenu *toolsMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    ASSERT_NE(toolsMenu, nullptr);
    EXPECT_TRUE(toolsMenu->actions().contains(action));
}

TEST(MainMenuTest, ToolsMenuExposesReferenceTerrainBundleAdjustAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.referenceTerrainBundleAdjustAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("参考地形约束重新平差")));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionReferenceTerrainBundleAdjust"));

    QMenu *toolsMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    ASSERT_NE(toolsMenu, nullptr);
    EXPECT_TRUE(toolsMenu->actions().contains(action));
}

TEST(MainMenuTest, ToolsMenuExposesSurveyControlAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.surveyControlAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("测绘控制")));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionSurveyControl"));

    QMenu *toolsMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    ASSERT_NE(toolsMenu, nullptr);
    EXPECT_TRUE(toolsMenu->actions().contains(action));
}

TEST(MainMenuTest, ToolsMenuExposesDetectMarkersInMarkerSubmenu)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.detectMarkersAction();
    ASSERT_NE(action, nullptr);
    EXPECT_EQ(action->text(), QStringLiteral("检测标靶..."));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionDetectMarkers"));

    QMenu *tools_menu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    ASSERT_NE(tools_menu, nullptr);
    QMenu *markers_menu = findSubMenuByTitle(tools_menu, QStringLiteral("标记"));
    ASSERT_NE(markers_menu, nullptr);
    EXPECT_TRUE(markers_menu->actions().contains(action));
}

TEST(MainMenuTest, ToolsMenuExposesPrintMarkersInMarkerSubmenu)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.printMarkersAction();
    ASSERT_NE(action, nullptr);
    EXPECT_EQ(action->text(), QStringLiteral("打印标靶..."));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionPrintMarkers"));

    QMenu *tools_menu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    QMenu *markers_menu = findSubMenuByTitle(tools_menu, QStringLiteral("标记"));
    ASSERT_NE(markers_menu, nullptr);
    EXPECT_TRUE(markers_menu->actions().contains(action));
}

TEST(MainMenuTest, ToolsMenuExposesMarkerDetectionReviewInMarkerSubmenu)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.reviewMarkerDetectionsAction();
    ASSERT_NE(action, nullptr);
    EXPECT_EQ(action->text(), QStringLiteral("复核检测候选..."));
    EXPECT_EQ(action->objectName(), QStringLiteral("actionReviewMarkerDetections"));

    QMenu *tools_menu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    QMenu *markers_menu = findSubMenuByTitle(tools_menu, QStringLiteral("标记"));
    ASSERT_NE(markers_menu, nullptr);
    EXPECT_TRUE(markers_menu->actions().contains(action));
}

TEST(MainMenuTest, ToolsMenuExposesTiePointsSubmenu)
{
    QMainWindow window;
    MainMenu menu(&window);

    ASSERT_NE(menu.createTiePointsAction(), nullptr);
    ASSERT_NE(menu.thinTiePointsAction(), nullptr);
    ASSERT_NE(menu.cleanTiePointsAction(), nullptr);
    ASSERT_NE(menu.viewTiePointMatchesAction(), nullptr);

    EXPECT_EQ(menu.createTiePointsAction()->text(), QStringLiteral("创建连接点..."));
    EXPECT_EQ(menu.thinTiePointsAction()->text(), QStringLiteral("稀释连接点..."));
    EXPECT_EQ(menu.cleanTiePointsAction()->text(), QStringLiteral("Clean Tie Points..."));
    EXPECT_EQ(menu.viewTiePointMatchesAction()->text(), QStringLiteral("查看匹配..."));

    QMenu *toolsMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    ASSERT_NE(toolsMenu, nullptr);
    QMenu *tiePointsMenu = findSubMenuByTitle(toolsMenu, QStringLiteral("连接点"));
    ASSERT_NE(tiePointsMenu, nullptr);

    const QStringList tiePointActions = {
        QStringLiteral("创建连接点..."),
        QStringLiteral("稀释连接点..."),
        QStringLiteral("Clean Tie Points..."),
        QStringLiteral("查看匹配...")
    };
    EXPECT_EQ(directActionTexts(tiePointsMenu).join(QStringLiteral("|")),
              tiePointActions.join(QStringLiteral("|")));
    EXPECT_FALSE(directActionTexts(toolsMenu).contains(QStringLiteral("查看匹配...")));
}

TEST(MainMenuImageRotationTest, ExposesViewActionsAndStableToolbarButtons)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *rotateLeft = menu.rotateImageLeftAction();
    QAction *rotateRight = menu.rotateImageRightAction();
    ASSERT_NE(rotateLeft, nullptr);
    ASSERT_NE(rotateRight, nullptr);
    EXPECT_EQ(rotateLeft->toolTip(), QStringLiteral("向左旋转"));
    EXPECT_EQ(rotateRight->toolTip(), QStringLiteral("向右旋转"));
    EXPECT_FALSE(rotateLeft->icon().isNull());
    EXPECT_FALSE(rotateRight->icon().isNull());
    EXPECT_FALSE(rotateLeft->isEnabled());
    EXPECT_FALSE(rotateRight->isEnabled());

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    QMenu *imageMenu = findSubMenuByTitle(viewMenu, QStringLiteral("影像显示"));
    ASSERT_NE(imageMenu, nullptr);
    EXPECT_TRUE(imageMenu->actions().contains(rotateLeft));
    EXPECT_TRUE(imageMenu->actions().contains(rotateRight));
    EXPECT_LT(imageMenu->actions().indexOf(rotateLeft),
              imageMenu->actions().indexOf(menu.showFeaturePointsAction()));
    EXPECT_LT(imageMenu->actions().indexOf(rotateRight),
              imageMenu->actions().indexOf(menu.showFeaturePointsAction()));

    QToolBar *toolBar = menu.toolBar();
    ASSERT_NE(toolBar, nullptr);
    menu.setContextualToolbarVisibility(false, true);
    auto *leftButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageLeft"));
    auto *rightButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageRight"));
    ASSERT_NE(leftButton, nullptr);
    ASSERT_NE(rightButton, nullptr);
    EXPECT_EQ(leftButton->defaultAction(), rotateLeft);
    EXPECT_EQ(rightButton->defaultAction(), rotateRight);
    EXPECT_EQ(leftButton->width(), 36);
    EXPECT_EQ(leftButton->height(), 36);
    EXPECT_EQ(rightButton->width(), 36);
    EXPECT_EQ(rightButton->height(), 36);
    auto opaqueBounds = [](const QIcon &icon)
    {
        const QImage image = icon.pixmap(QSize(56, 56)).toImage().convertToFormat(QImage::Format_ARGB32);
        QRect bounds;
        bool hasOpaquePixel = false;
        for (int y = 0; y < image.height(); ++y)
        {
            for (int x = 0; x < image.width(); ++x)
            {
                if (qAlpha(image.pixel(x, y)) == 0)
                {
                    continue;
                }
                const QRect pixelRect(x, y, 1, 1);
                bounds = hasOpaquePixel ? bounds.united(pixelRect) : pixelRect;
                hasOpaquePixel = true;
            }
        }
        return bounds;
    };

    EXPECT_EQ(leftButton->iconSize(), QSize(26, 26));
    EXPECT_EQ(rightButton->iconSize(), QSize(26, 26));
    EXPECT_GE(opaqueBounds(rotateLeft->icon()).width(), 50);
    EXPECT_GE(opaqueBounds(rotateLeft->icon()).height(), 50);
    EXPECT_GE(opaqueBounds(rotateRight->icon()).width(), 50);
    EXPECT_GE(opaqueBounds(rotateRight->icon()).height(), 50);

    window.show();
    QCoreApplication::processEvents();
    menu.setContextualToolbarVisibility(false, true);

    EXPECT_TRUE(leftButton->isVisible());
    EXPECT_TRUE(rightButton->isVisible());
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraVisibility")),
              nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraImageVisibility")),
              nullptr);

    menu.setContextualToolbarVisibility(true, false);
    auto *cameraButton =
        toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraVisibility"));
    auto *cameraImageButton =
        toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraImageVisibility"));
    ASSERT_NE(cameraButton, nullptr);
    ASSERT_NE(cameraImageButton, nullptr);
    EXPECT_TRUE(cameraButton->isVisible());
    EXPECT_TRUE(cameraImageButton->isVisible());
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageLeft")), nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageRight")), nullptr);

    menu.setContextualToolbarVisibility(false, true);
    leftButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageLeft"));
    rightButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageRight"));
    ASSERT_NE(leftButton, nullptr);
    ASSERT_NE(rightButton, nullptr);
    EXPECT_TRUE(leftButton->isVisible());
    EXPECT_TRUE(rightButton->isVisible());
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraVisibility")),
              nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraImageVisibility")),
              nullptr);

    menu.setContextualToolbarVisibility(false, false);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraVisibility")),
              nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraImageVisibility")),
              nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageLeft")), nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonRotateImageRight")), nullptr);
}

TEST(MainMenuImageRotationTest, RotationButtonsUseDedicatedFullAreaPainter)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/ToolbarButton.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("ToolbarButton::paintEvent")));
    EXPECT_TRUE(source.contains(QStringLiteral("paintButtonIcon(painter, this, rect());")));
    EXPECT_TRUE(source.contains(QStringLiteral("painter.drawPixmap(iconTopLeft, pixmap);")));
}

TEST(MainMenuZoomTest, ExposesLargeToolbarButtonsAndStandardShortcuts)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *zoomIn = menu.zoomInAction();
    QAction *zoomOut = menu.zoomOutAction();
    ASSERT_NE(zoomIn, nullptr);
    ASSERT_NE(zoomOut, nullptr);
    EXPECT_TRUE(zoomIn->shortcuts().contains(QKeySequence::ZoomIn));
    EXPECT_TRUE(zoomOut->shortcuts().contains(QKeySequence::ZoomOut));
    EXPECT_EQ(zoomIn->toolTip(), QStringLiteral("放大"));
    EXPECT_EQ(zoomOut->toolTip(), QStringLiteral("缩小"));
    EXPECT_FALSE(zoomIn->icon().isNull());
    EXPECT_FALSE(zoomOut->icon().isNull());

    QToolBar *toolBar = menu.toolBar();
    ASSERT_NE(toolBar, nullptr);
    auto *zoomInButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonZoomIn"));
    auto *zoomOutButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonZoomOut"));
    ASSERT_NE(zoomInButton, nullptr);
    ASSERT_NE(zoomOutButton, nullptr);
    EXPECT_EQ(zoomInButton->defaultAction(), zoomIn);
    EXPECT_EQ(zoomOutButton->defaultAction(), zoomOut);
    EXPECT_EQ(zoomInButton->size(), QSize(36, 36));
    EXPECT_EQ(zoomOutButton->size(), QSize(36, 36));
    EXPECT_EQ(zoomInButton->iconSize(), QSize(26, 26));
    EXPECT_EQ(zoomOutButton->iconSize(), QSize(26, 26));
}

TEST(MainMenuToolbarTemplateTest, UsesOneCompactTemplateForEveryToolbarCommand)
{
    QMainWindow window;
    MainMenu menu(&window);
    QToolBar *toolBar = menu.toolBar();
    ASSERT_NE(toolBar, nullptr);
    menu.setContextualToolbarVisibility(true, false);

    const QStringList compactButtonNames = {
        QStringLiteral("toolButtonSaveProject"),
        QStringLiteral("toolButtonZoomIn"),
        QStringLiteral("toolButtonZoomOut"),
        QStringLiteral("toolButtonManualPointCloudPrune")
    };
    for (const QString &name : compactButtonNames)
    {
        auto *button = toolBar->findChild<QToolButton *>(name);
        ASSERT_NE(button, nullptr) << name.toStdString();
        EXPECT_EQ(button->size(), QSize(36, 36));
        EXPECT_EQ(button->iconSize(), QSize(26, 26));
        EXPECT_EQ(button->toolButtonStyle(), Qt::ToolButtonIconOnly);
    }

    const QStringList splitButtonNames = {
        QStringLiteral("toolButtonModelCameraVisibility"),
        QStringLiteral("toolButtonModelCameraImageVisibility")
    };
    for (const QString &name : splitButtonNames)
    {
        auto *button = toolBar->findChild<QToolButton *>(name);
        ASSERT_NE(button, nullptr) << name.toStdString();
        EXPECT_EQ(button->size(), QSize(50, 36));
        EXPECT_EQ(button->iconSize(), QSize(26, 26));
        EXPECT_EQ(button->toolButtonStyle(), Qt::ToolButtonIconOnly);
    }

    EXPECT_FALSE(toolBar->actions().contains(menu.saveAction()));
    EXPECT_FALSE(toolBar->actions().contains(menu.manualPointCloudPruneAction()));
}

TEST(MainMenuImageOverlayToolbarTest, ExposesImageOnlyPointMaskAndResetCommands)
{
    QMainWindow window;
    MainMenu menu(&window);
    QToolBar *toolBar = menu.toolBar();
    ASSERT_NE(toolBar, nullptr);

    QAction *showPoints = window.findChild<QAction *>(QStringLiteral("actionShowFeaturePoints"));
    QAction *showResiduals = window.findChild<QAction *>(QStringLiteral("actionShowFeatureResiduals"));
    QAction *showMask = window.findChild<QAction *>(QStringLiteral("actionShowMaskOverlay"));
    ASSERT_NE(showPoints, nullptr);
    ASSERT_NE(showResiduals, nullptr);
    ASSERT_NE(showMask, nullptr);
    EXPECT_TRUE(showPoints->isCheckable());
    EXPECT_TRUE(showResiduals->isCheckable());
    EXPECT_TRUE(showMask->isCheckable());

    menu.setContextualToolbarVisibility(false, true);
    auto *pointsButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonShowFeaturePoints"));
    auto *maskButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonShowMaskOverlay"));
    auto *resetButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonResetImageView"));
    ASSERT_NE(pointsButton, nullptr);
    ASSERT_NE(maskButton, nullptr);
    ASSERT_NE(resetButton, nullptr);
    EXPECT_EQ(pointsButton->size(), QSize(50, 36));
    EXPECT_EQ(maskButton->size(), QSize(36, 36));
    EXPECT_EQ(resetButton->size(), QSize(36, 36));
    ASSERT_NE(pointsButton->menu(), nullptr);
    EXPECT_TRUE(pointsButton->menu()->actions().contains(showResiduals));
    EXPECT_TRUE(pointsButton->menu()->actions().contains(menu.featureVisualizationAction()));

    menu.setContextualToolbarVisibility(true, false);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonShowFeaturePoints")), nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonShowMaskOverlay")), nullptr);
    EXPECT_EQ(toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonResetImageView")), nullptr);
}

TEST(MainMenuDepthOverlayToolbarTest, ExposesMetashapeStyleDepthLevelsAndIntensity)
{
    QMainWindow window;
    MainMenu menu(&window);
    QToolBar *tool_bar = menu.toolBar();
    ASSERT_NE(tool_bar, nullptr);

    QAction *show_depth = menu.showDepthOverlayAction();
    QAction *all_levels = menu.depthOverlayAllLevelsAction();
    QAction *level_1 = menu.depthOverlayLevel1Action();
    QAction *level_2 = menu.depthOverlayLevel2Action();
    QAction *level_3 = menu.depthOverlayLevel3Action();
    QAction *show_intensity = menu.showDepthIntensityAction();
    ASSERT_NE(show_depth, nullptr);
    ASSERT_NE(all_levels, nullptr);
    ASSERT_NE(level_1, nullptr);
    ASSERT_NE(level_2, nullptr);
    ASSERT_NE(level_3, nullptr);
    ASSERT_NE(show_intensity, nullptr);
    EXPECT_TRUE(show_depth->isCheckable());
    EXPECT_FALSE(show_depth->isChecked());
    EXPECT_TRUE(all_levels->isChecked());
    EXPECT_FALSE(level_1->isChecked());
    EXPECT_FALSE(level_2->isChecked());
    EXPECT_FALSE(level_3->isChecked());
    EXPECT_FALSE(show_intensity->isChecked());

    menu.setContextualToolbarVisibility(false, true);
    auto *depth_button = tool_bar->findChild<QToolButton *>(
        QStringLiteral("toolButtonShowDepthOverlay"));
    ASSERT_NE(depth_button, nullptr);
    EXPECT_EQ(depth_button->size(), QSize(50, 36));
    ASSERT_NE(depth_button->menu(), nullptr);
    const QList<QAction *> actions = depth_button->menu()->actions();
    ASSERT_EQ(actions.size(), 6);
    EXPECT_EQ(actions[0], all_levels);
    EXPECT_EQ(actions[1], level_1);
    EXPECT_EQ(actions[2], level_2);
    EXPECT_EQ(actions[3], level_3);
    EXPECT_TRUE(actions[4]->isSeparator());
    EXPECT_EQ(actions[5], show_intensity);

    level_2->setChecked(true);
    EXPECT_FALSE(all_levels->isChecked());
    EXPECT_TRUE(level_2->isChecked());

    menu.setContextualToolbarVisibility(true, false);
    EXPECT_EQ(tool_bar->findChild<QToolButton *>(QStringLiteral("toolButtonShowDepthOverlay")),
              nullptr);
}

TEST(MainMenuDepthOverlayToolbarTest, MissingLevelDoesNotDisableDepthOverlayButton)
{
    QMainWindow window;
    MainMenu menu(&window);
    menu.setContextualToolbarVisibility(false, true);
    menu.setImageDisplayReady(true);
    menu.setDepthOverlayAvailable(true);
    menu.setDepthOverlayLevelsAvailable(
        true,
        true,
        false,
        false,
        {},
        {},
        QStringLiteral("Level 2 测试原因"),
        QStringLiteral("640×480 只生成 2 层"));

    EXPECT_TRUE(menu.showDepthOverlayAction()->isEnabled());
    EXPECT_TRUE(menu.depthOverlayAllLevelsAction()->isEnabled());
    EXPECT_TRUE(menu.depthOverlayLevel1Action()->isEnabled());
    EXPECT_FALSE(menu.depthOverlayLevel2Action()->isEnabled());
    EXPECT_FALSE(menu.depthOverlayLevel3Action()->isEnabled());
    EXPECT_EQ(menu.depthOverlayLevel2Action()->toolTip(),
              QStringLiteral("Level 2 测试原因"));
    EXPECT_EQ(menu.depthOverlayLevel3Action()->statusTip(),
              QStringLiteral("640×480 只生成 2 层"));
    EXPECT_TRUE(menu.showDepthIntensityAction()->isEnabled());
}

TEST(FeatureResidualVisualizationTest, ExportsAndLoadsTrueReprojectionVectorsAsynchronously)
{
    const QString aerialSource =
        readProjectSourceFile(QStringLiteral("src/core/aerial_triangulation/reporting/QualityReportWriter.cpp"));
    const QString loaderHeader =
        readProjectSourceFile(QStringLiteral("src/gui/views/FeatureResidualLoader.h"));
    const QString loaderSource =
        readProjectSourceFile(QStringLiteral("src/gui/views/FeatureResidualLoader.cpp"));
    const QString canvasSource =
        readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));

    EXPECT_TRUE(aerialSource.contains(QStringLiteral("projected_xy")));
    EXPECT_TRUE(aerialSource.contains(QStringLiteral("residual_xy")));
    EXPECT_TRUE(aerialSource.contains(QStringLiteral("projectWorldPoint")));
    EXPECT_TRUE(loaderHeader.contains(QStringLiteral("FeatureResidualVector")));
    EXPECT_TRUE(loaderHeader.contains(QStringLiteral("loadFeatureResidualsForImage")));
    EXPECT_TRUE(loaderSource.contains(QStringLiteral("sparse_cloud_points_json")));
    EXPECT_TRUE(loaderSource.contains(QStringLiteral("projected_xy")));
    EXPECT_TRUE(canvasSource.contains(QStringLiteral("QtConcurrent::run")));
    EXPECT_TRUE(canvasSource.contains(QStringLiteral("startResidualLoadForImage")));
    EXPECT_TRUE(canvasSource.contains(QStringLiteral("_residualLoadGeneration")));
}

TEST(FeatureResidualVisualizationTest, DialogControlsResidualExtentAndRendererDrawsVectors)
{
    const QString rendererHeader =
        readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString overlaySource =
        readProjectSourceFile(QStringLiteral("src/gui/views/LayerOverlayItems.cpp"));
    const QString dialogHeader =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.h"));
    const QString dialogSource =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.cpp"));

    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("showResiduals")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("residualScale")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("minimumResidualPx")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("maximumResidualLengthPx")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("residualColor")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("createFeatureResidualOverlayItem")));
    EXPECT_TRUE(dialogHeader.contains(QStringLiteral("_showResidualsChk")));
    EXPECT_TRUE(dialogHeader.contains(QStringLiteral("_residualScaleSpin")));
    EXPECT_TRUE(dialogHeader.contains(QStringLiteral("_minimumResidualSpin")));
    EXPECT_TRUE(dialogHeader.contains(QStringLiteral("_maximumResidualLengthSpin")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("opts.showResiduals")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("opts.residualScale")));
}

TEST(FeatureResidualLoaderTest, SelectsOnlyTheCurrentImagesTrueResidualVectors)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("中文项目.plascan"));
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("影像甲.tif"));
    const QString otherImagePath = QDir(tempDir.path()).filePath(QStringLiteral("影像乙.tif"));
    const QString sidecarPath = QDir(tempDir.path()).filePath(QStringLiteral("sfm_sparse_points.json"));
    ProjectData projectData;
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("residuals")));

    const QJsonObject observation{
        {QStringLiteral("image_path"), imagePath},
        {QStringLiteral("xy"), QJsonArray{10.0, 20.0}},
        {QStringLiteral("projected_xy"), QJsonArray{13.0, 24.0}},
        {QStringLiteral("residual_xy"), QJsonArray{3.0, 4.0}}
    };
    const QJsonObject otherObservation{
        {QStringLiteral("image_path"), otherImagePath},
        {QStringLiteral("xy"), QJsonArray{1.0, 2.0}},
        {QStringLiteral("projected_xy"), QJsonArray{8.0, 9.0}}
    };
    QFile sidecar(sidecarPath);
    ASSERT_TRUE(sidecar.open(QIODevice::WriteOnly));
    sidecar.write(QJsonDocument(QJsonObject{
        {QStringLiteral("points"), QJsonArray{
            QJsonObject{{QStringLiteral("observations"), QJsonArray{observation, otherObservation}}}
        }}
    }).toJson());
    sidecar.close();

    ASSERT_TRUE(QDir().mkpath(xjw::common::project::ProjectIO::tmpDir(projectPath)));
    QFile results(xjw::common::project::ProjectIO::tempResultsPath(projectPath));
    ASSERT_TRUE(results.open(QIODevice::WriteOnly));
    results.write(QJsonDocument(QJsonObject{
        {QStringLiteral("aerial_triangulation_results"), QJsonArray{
            QJsonObject{{QStringLiteral("files"), QJsonObject{
                {QStringLiteral("sparse_cloud_points_json"), sidecarPath}
            }}}
        }}
    }).toJson());
    results.close();

    const auto residuals = xjw::gui::views::loadFeatureResidualsForImage(projectPath, imagePath);
    ASSERT_EQ(residuals.size(), 1);
    EXPECT_EQ(residuals.first().observed, QPointF(10.0, 20.0));
    EXPECT_EQ(residuals.first().projected, QPointF(13.0, 24.0));
    EXPECT_DOUBLE_EQ(residuals.first().magnitudePx, 5.0);
}

TEST(MainMenuToolbarTemplateTest, ExtractsReusableToolbarComponentsFromMainMenu)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/menu/ToolbarButton.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/ToolbarButton.cpp"));
    const QString mainMenuSource = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(mainMenuSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("struct ToolbarMetrics")));
    EXPECT_TRUE(header.contains(QStringLiteral("ButtonExtent = 36")));
    EXPECT_TRUE(header.contains(QStringLiteral("IconExtent = 26")));
    EXPECT_TRUE(header.contains(QStringLiteral("SplitButtonWidth = 50")));
    EXPECT_TRUE(header.contains(QStringLiteral("createToolbarButton")));
    EXPECT_TRUE(header.contains(QStringLiteral("createToolbarSplitButton")));
    EXPECT_TRUE(source.contains(QStringLiteral("backgroundColor")));
    EXPECT_TRUE(source.contains(QStringLiteral("Qt::NoPen")));
    EXPECT_FALSE(mainMenuSource.contains(QStringLiteral("class ToolbarIconButton")));
    EXPECT_FALSE(mainMenuSource.contains(QStringLiteral("class ToolbarSplitButton")));
}

TEST(MainMenuToolbarTemplateTest, BrandWidgetFitsCompactToolbarHeight)
{
    HenuBrandWidget brand;
    EXPECT_LE(brand.minimumSizeHint().height(), 40);
    EXPECT_LE(brand.sizeHint().height(), 40);
    EXPECT_LE(brand.minimumSizeHint().width(), 180);
    EXPECT_EQ(brand.sizePolicy().horizontalPolicy(), QSizePolicy::Preferred);
}

TEST(MainMenuZoomTest, WorkflowCommandsRemainInMenusButAreRemovedFromToolbar)
{
    QMainWindow window;
    MainMenu menu(&window);
    QToolBar *toolBar = menu.toolBar();
    ASSERT_NE(toolBar, nullptr);

    const QList<QAction *> removedToolbarActions = {
        menu.addPhotoAction(),
        menu.addFolderAction(),
        menu.workflowAerialTriangulationAction(),
        menu.createDEMAction(),
        menu.generateOrthoAction()
    };
    for (QAction *action : removedToolbarActions)
    {
        ASSERT_NE(action, nullptr);
        EXPECT_FALSE(toolBar->actions().contains(action));
    }

    QMenu *workflowMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工作流程"));
    ASSERT_NE(workflowMenu, nullptr);
    for (QAction *action : removedToolbarActions)
    {
        EXPECT_TRUE(workflowMenu->actions().contains(action));
    }
}

TEST(MainMenuWorkflowTest, WorkflowCommandsDoNotShowLeadingIcons)
{
    QMainWindow window;

    auto *projectMenu = window.menuBar()->addMenu(QStringLiteral("项目"));
    projectMenu->setObjectName(QStringLiteral("menuProject"));
    auto *newProject = new QAction(QStringLiteral("新建项目"), &window);
    newProject->setObjectName(QStringLiteral("actionNewProject"));
    projectMenu->addAction(newProject);

    auto *workflowMenu = window.menuBar()->addMenu(QStringLiteral("工作流程"));
    workflowMenu->setObjectName(QStringLiteral("menuWorkflow"));

    auto makeWorkflowAction = [&window, workflowMenu](const char *objectName, const QString &text)
    {
        auto *action = new QAction(
            window.style()->standardIcon(QStyle::SP_MessageBoxInformation), text, &window);
        action->setObjectName(QString::fromLatin1(objectName));
        workflowMenu->addAction(action);
        return action;
    };

    const QList<QAction *> workflowActions{
        makeWorkflowAction("actionWorkflowAerialTriangulation", QStringLiteral("空中三角测量...")),
        makeWorkflowAction("actionGenerateModel", QStringLiteral("生成模型...")),
        makeWorkflowAction("actionCreateDEM", QStringLiteral("创建 DEM")),
        makeWorkflowAction("actionGenerateOrtho", QStringLiteral("生成正射影像"))
    };

    MainMenu menu(&window);

    for (QAction *action : workflowActions)
    {
        ASSERT_NE(action, nullptr);
        EXPECT_TRUE(action->icon().isNull()) << qPrintable(action->text());
        EXPECT_TRUE(workflowMenu->actions().contains(action));
    }
}

TEST(MainWindowZoomTest, DispatchesZoomToActiveImageOrModelView)
{
    const QString sceneHeader =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString sceneSource =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString mainWindowSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(sceneHeader.isEmpty());
    ASSERT_FALSE(sceneSource.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("void zoomIn();")));
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("void zoomOut();")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("void CameraSceneWidget::zoomIn()")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("void CameraSceneWidget::zoomOut()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("ViewMode::Image")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("_canvas->zoomIn();")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("_canvas->zoomOut();")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("_workspaceCenter->modelView()->zoomIn();")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("_workspaceCenter->modelView()->zoomOut();")));
}

TEST(MainWindowImageRotationTest, ConnectsActionsAndPersistsPerImageRotation)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject _imageViewRotations;")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("&QAction::triggered, _canvas, &CanvasWidget::rotateLeft")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("&QAction::triggered, _canvas, &CanvasWidget::rotateRight")));
    EXPECT_TRUE(source.contains(QStringLiteral("&CanvasWidget::displayImageReadyChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("&CanvasWidget::viewRotationChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("\"image_view_rotations\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("projectImageStateKey(path)")));
    EXPECT_TRUE(source.contains(QStringLiteral("\"active_image_id\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("withImageViewRotation(")));
    EXPECT_TRUE(source.contains(QStringLiteral("_imageViewRotations = QJsonObject{};")));
}

TEST(ContextualToolbarTest, WorkspaceModeDrivesModelAndImageToolbarGroups)
{
    const QString workspaceHeader =
        readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.h"));
    const QString workspaceSource =
        readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    const QString mainWindowSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(workspaceHeader.isEmpty());
    ASSERT_FALSE(workspaceSource.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_TRUE(workspaceHeader.contains(QStringLiteral("enum class ViewMode")));
    EXPECT_TRUE(workspaceHeader.contains(QStringLiteral("ViewMode currentViewMode() const;")));
    EXPECT_TRUE(workspaceHeader.contains(QStringLiteral("void viewModeChanged(ViewMode mode);")));
    EXPECT_TRUE(workspaceSource.contains(QStringLiteral("&QStackedWidget::currentChanged")));
    EXPECT_TRUE(workspaceSource.contains(QStringLiteral("emit viewModeChanged(currentViewMode());")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("&WorkspaceCenterWidget::viewModeChanged")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("setContextualToolbarVisibility(")));
}

TEST(ImageViewRotationSettingsTest, NormalizesQuarterTurnsAndRejectsArbitraryAngles)
{
    using xjw::gui::config::normalizeImageViewRotationDegrees;

    EXPECT_EQ(normalizeImageViewRotationDegrees(-90), 270);
    EXPECT_EQ(normalizeImageViewRotationDegrees(450), 90);
    EXPECT_EQ(normalizeImageViewRotationDegrees(720), 0);
    EXPECT_EQ(normalizeImageViewRotationDegrees(45), 0);
}

TEST(ImageViewRotationSettingsTest, PersistsChinesePathAndRemovesZeroRotation)
{
    using xjw::gui::config::imageViewRotationForPath;
    using xjw::gui::config::withImageViewRotation;

    const QString storedPath = QStringLiteral("G:/姿态训练/影像/IMG_001.TIF");
    const QString lookupPath = QStringLiteral("g:/姿态训练/影像/img_001.tif");

    QJsonObject rotations;
    rotations = withImageViewRotation(rotations, storedPath, 90);
#ifdef Q_OS_WIN
    EXPECT_EQ(imageViewRotationForPath(rotations, lookupPath), 90);
#else
    EXPECT_EQ(imageViewRotationForPath(rotations, storedPath), 90);
#endif

    rotations = withImageViewRotation(rotations, storedPath, 0);
    EXPECT_TRUE(rotations.isEmpty());
}

TEST(ImageViewRotationSettingsTest, IgnoresInvalidStoredRotation)
{
    using xjw::gui::config::imageViewRotationForPath;
    using xjw::gui::config::imageViewRotationPathKey;

    const QString path = QStringLiteral("C:/project/image.tif");
    const QJsonObject rotations{
        {imageViewRotationPathKey(path), 45}
    };

    EXPECT_EQ(imageViewRotationForPath(rotations, path), 0);
    EXPECT_EQ(imageViewRotationForPath(QJsonObject{}, path), 0);
    EXPECT_EQ(imageViewRotationForPath(rotations, QString()), 0);
}

TEST(CanvasImageRotationTest, RotatesLoadedImageByQuarterTurns)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("旋转测试.png"));
    QImage image(40, 20, QImage::Format_RGB32);
    image.fill(QColor(20, 80, 140));
    ASSERT_TRUE(image.save(imagePath));

    CanvasWidget canvas;
    canvas.resize(320, 240);
    canvas.show();
    canvas.showImage(imagePath);
    QTRY_VERIFY_WITH_TIMEOUT(canvas.hasDisplayImage(), 5000);

    QSignalSpy rotationSpy(&canvas, &CanvasWidget::viewRotationChanged);
    canvas.rotateRight();
    EXPECT_EQ(canvas.viewRotationDegrees(), 90);
    ASSERT_EQ(rotationSpy.count(), 1);
    EXPECT_EQ(rotationSpy.first().at(0).toString(), imagePath);
    EXPECT_EQ(rotationSpy.first().at(1).toInt(), 90);

    canvas.rotateRight();
    canvas.rotateRight();
    canvas.rotateRight();
    EXPECT_EQ(canvas.viewRotationDegrees(), 0);

    canvas.rotateLeft();
    EXPECT_EQ(canvas.viewRotationDegrees(), 270);
}

TEST(CanvasImageRotationTest, ResetViewPreservesRotationAndEmptyCanvasIgnoresUserRotation)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("reset.png"));
    QImage image(30, 50, QImage::Format_RGB32);
    image.fill(Qt::white);
    ASSERT_TRUE(image.save(imagePath));

    CanvasWidget canvas;
    canvas.resize(320, 240);
    canvas.show();
    canvas.showImage(imagePath);
    QTRY_VERIFY_WITH_TIMEOUT(canvas.hasDisplayImage(), 5000);

    canvas.setViewRotationDegrees(90);
    canvas.zoomIn();
    canvas.resetView();
    EXPECT_EQ(canvas.viewRotationDegrees(), 90);

    canvas.showImage(QString());
    EXPECT_FALSE(canvas.hasDisplayImage());
    EXPECT_EQ(canvas.viewRotationDegrees(), 0);
    canvas.rotateRight();
    EXPECT_EQ(canvas.viewRotationDegrees(), 0);
}

TEST(CanvasImageRotationTest, SwitchingImagesClearsPreviousViewRotation)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));
    ASSERT_TRUE(QImage(96, 64, QImage::Format_RGB32).save(firstPath));
    ASSERT_TRUE(QImage(64, 96, QImage::Format_RGB32).save(secondPath));

    CanvasWidget canvas;
    canvas.resize(420, 320);
    canvas.show();
    canvas.showImage(firstPath);
    QTRY_VERIFY_WITH_TIMEOUT(canvas.hasDisplayImage(), 5000);
    canvas.rotateRight();
    ASSERT_EQ(canvas.viewRotationDegrees(), 90);
    ASSERT_GT(std::abs(canvas.transform().m12()), 0.01);

    canvas.showImage(secondPath);
    QTRY_VERIFY_WITH_TIMEOUT(canvas.hasDisplayImage(), 5000);
    EXPECT_EQ(canvas.viewRotationDegrees(), 0);
    EXPECT_LT(std::abs(canvas.transform().m12()), 0.01);
}

TEST(CanvasDepthMapDisplayTest, DoesNotTreatDepthValidityMaskAsPhotoMaskOverlay)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString depthPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_0.png"));
    const QString validityMaskPath =
        QDir(tempDir.path()).filePath(QStringLiteral("depth_0_mask.png"));

    QImage depthPreview(120, 80, QImage::Format_RGB32);
    depthPreview.fill(QColor(40, 90, 160));
    ASSERT_TRUE(depthPreview.save(depthPath));
    QImage validityMask(480, 320, QImage::Format_Grayscale8);
    validityMask.fill(0);
    for (int y = 20; y < 300; ++y)
    {
        uchar *row = validityMask.scanLine(y);
        std::fill(row + 20, row + 460, static_cast<uchar>(255));
    }
    ASSERT_TRUE(validityMask.save(validityMaskPath));

    CanvasWidget canvas;
    canvas.resize(640, 480);
    canvas.show();
    canvas.showImage(depthPath);
    QTRY_VERIFY_WITH_TIMEOUT(canvas.hasDisplayImage(), 5000);

    EXPECT_LE(canvas.scene()->sceneRect().width(), 121.0);
    EXPECT_LE(canvas.scene()->sceneRect().height(), 81.0);
}

TEST(TiePointResultServiceTest, SelectsLatestExistingSparseCloudFromLegacyHistory)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString oldPath = QDir(tempDir.path()).filePath(QStringLiteral("old/sparse.ply"));
    const QString missingPath = QDir(tempDir.path()).filePath(QStringLiteral("missing/sparse.ply"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(oldPath).absolutePath()));

    QFile oldFile(oldPath);
    ASSERT_TRUE(oldFile.open(QIODevice::WriteOnly));
    ASSERT_EQ(oldFile.write("ply"), 3);
    oldFile.close();

    const QJsonObject oldRecord{
        {QStringLiteral("sparse_point_count"), 2314},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), oldPath}}}
    };
    const QJsonObject missingRecord{
        {QStringLiteral("sparse_point_count"), 9999},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), missingPath}}}
    };
    const QJsonObject meta{
        {QStringLiteral("aerial_triangulation_results"), QJsonArray{oldRecord, missingRecord}}
    };

    const auto selection = xjw::gui::project::ProjectTiePointResultService::selectCurrent(
        meta,
        QDir(tempDir.path()).filePath(QStringLiteral("legacy.plascan")));

    ASSERT_TRUE(selection.isValid());
    EXPECT_EQ(selection.sourceIndex, 0);
    EXPECT_EQ(selection.pointCount, 2314);
    EXPECT_EQ(QDir::cleanPath(selection.sparseCloudPath), QDir::cleanPath(oldPath));
}

TEST(TiePointResultServiceTest, ReturnsInvalidSelectionWhenNoSparseCloudExists)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QJsonObject meta{
        {QStringLiteral("aerial_triangulation_results"),
         QJsonArray{QJsonObject{
             {QStringLiteral("sparse_point_count"), 12},
             {QStringLiteral("files"),
              QJsonObject{{QStringLiteral("sparse_cloud_xyz"),
                           QStringLiteral("missing/sparse.ply")}}}
         }}}
    };

    const auto selection = xjw::gui::project::ProjectTiePointResultService::selectCurrent(
        meta,
        QDir(tempDir.path()).filePath(QStringLiteral("missing.plascan")));

    EXPECT_FALSE(selection.isValid());
    EXPECT_EQ(selection.sourceIndex, -1);
    EXPECT_EQ(selection.pointCount, -1);
    EXPECT_TRUE(selection.sparseCloudPath.isEmpty());
}

TEST(TiePointResultServiceTest, ReplaceKeepsOnlyNewRecordAndProtectsSharedOutputDirectory)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("replace.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("replace")));

    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("tie_points"));
    ASSERT_TRUE(QDir().mkpath(outputDir));
    const QString oldSparsePath = QDir(outputDir).filePath(QStringLiteral("old_sparse.ply"));
    const QString oldReportPath = QDir(outputDir).filePath(QStringLiteral("old_report.json"));
    const QString newSparsePath = QDir(outputDir).filePath(QStringLiteral("current_sparse.ply"));
    const QString newReportPath = QDir(outputDir).filePath(QStringLiteral("current_report.json"));
    for (const QString &path : {oldSparsePath, oldReportPath, newSparsePath, newReportPath})
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write("data"), 0);
    }

    const QJsonObject oldRecord{
        {QStringLiteral("output_dir"), outputDir},
        {QStringLiteral("sparse_point_count"), 100},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), oldSparsePath},
                     {QStringLiteral("quality_report"), oldReportPath}}}
    };
    ASSERT_TRUE(projectData.appendResultRecord(QStringLiteral("aerial_triangulation_results"), oldRecord));

    const QJsonObject newRecord{
        {QStringLiteral("output_dir"), outputDir},
        {QStringLiteral("sparse_point_count"), 200},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), newSparsePath},
                     {QStringLiteral("quality_report"), newReportPath}}}
    };

    const auto result = xjw::gui::project::ProjectTiePointResultService::replaceCurrent(
        &projectData,
        newRecord);

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.removedRecordCount, 1);
    EXPECT_TRUE(result.cleanupWarnings.isEmpty());
    const QJsonArray records = projectData.metadata()
                                   .value(QStringLiteral("aerial_triangulation_results"))
                                   .toArray();
    ASSERT_EQ(records.size(), 1);
    const QJsonObject storedRecord = records.first().toObject();
    EXPECT_EQ(storedRecord.value(QStringLiteral("files")), newRecord.value(QStringLiteral("files")));
    EXPECT_FALSE(storedRecord.value(QStringLiteral("reconstruction_generation_id"))
                     .toString()
                     .isEmpty());
    EXPECT_FALSE(QFileInfo::exists(oldSparsePath));
    EXPECT_FALSE(QFileInfo::exists(oldReportPath));
    EXPECT_TRUE(QFileInfo::exists(newSparsePath));
    EXPECT_TRUE(QFileInfo::exists(newReportPath));
    EXPECT_TRUE(QFileInfo(outputDir).isDir());
}

TEST(TiePointResultServiceTest, ReplaceInvalidatesAllDerivedReconstructionResults)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("invalidate.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("invalidate")));

    const QString sparsePath = QDir(tempDir.path()).filePath(QStringLiteral("current_sparse.ply"));
    QFile sparseFile(sparsePath);
    ASSERT_TRUE(sparseFile.open(QIODevice::WriteOnly));
    ASSERT_GT(sparseFile.write("ply"), 0);
    sparseFile.close();

    const QString oldDepthPath = QDir(tempDir.path()).filePath(QStringLiteral("old_depth.bin"));
    const QString oldPreviewPath = QDir(tempDir.path()).filePath(QStringLiteral("old_depth.png"));
    const QString sourceImagePath = QDir(tempDir.path()).filePath(QStringLiteral("source_image.png"));
    for (const QString &path : {oldDepthPath, oldPreviewPath, sourceImagePath})
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write("artifact"), 0);
    }

    QJsonObject metadata = projectData.metadata();
    metadata[QStringLiteral("depth_map_results")] = QJsonArray{QJsonObject{
        {QStringLiteral("depth_png"), oldPreviewPath},
        {QStringLiteral("raw_depth_path"), oldDepthPath},
        {QStringLiteral("ref_image"), sourceImagePath}}};
    metadata[QStringLiteral("dense_cloud_results")] = QJsonArray{QJsonObject{{QStringLiteral("path"),
                                                                               QStringLiteral("old_dense.ply")}}};
    metadata[QStringLiteral("model_results")] = QJsonArray{QJsonObject{{QStringLiteral("path"),
                                                                         QStringLiteral("old_model.ply")}}};
    metadata[QStringLiteral("dem_results")] = QJsonArray{QJsonObject{{QStringLiteral("dem_tif"),
                                                                       QStringLiteral("old_dem.tif")}}};
    metadata[QStringLiteral("ortho_results")] = QJsonArray{QJsonObject{{QStringLiteral("ortho_tif"),
                                                                         QStringLiteral("old_ortho.tif")}}};
    projectData.updateMetadata(metadata, false);

    const QJsonObject newRecord{
        {QStringLiteral("sparse_point_count"), 200},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), sparsePath}}}
    };
    const auto result = xjw::gui::project::ProjectTiePointResultService::replaceCurrent(
        &projectData,
        newRecord);

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    const QJsonObject updated = projectData.metadata();
    const QJsonArray atResults = updated.value(QStringLiteral("aerial_triangulation_results")).toArray();
    ASSERT_EQ(atResults.size(), 1);
    EXPECT_FALSE(atResults.first().toObject()
                     .value(QStringLiteral("reconstruction_generation_id"))
                     .toString()
                     .isEmpty());
    EXPECT_TRUE(updated.value(QStringLiteral("depth_map_results")).toArray().isEmpty());
    EXPECT_TRUE(updated.value(QStringLiteral("dense_cloud_results")).toArray().isEmpty());
    EXPECT_TRUE(updated.value(QStringLiteral("model_results")).toArray().isEmpty());
    EXPECT_TRUE(updated.value(QStringLiteral("dem_results")).toArray().isEmpty());
    EXPECT_TRUE(updated.value(QStringLiteral("ortho_results")).toArray().isEmpty());
    EXPECT_FALSE(QFileInfo::exists(oldDepthPath));
    EXPECT_FALSE(QFileInfo::exists(oldPreviewPath));
    EXPECT_TRUE(QFileInfo::exists(sourceImagePath));
}

TEST(TiePointResultServiceTest, ReplaceRejectsMissingNewSparseCloudWithoutChangingOldResult)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("invalid_replace.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("invalid_replace")));

    const QString oldSparsePath = QDir(tempDir.path()).filePath(QStringLiteral("old_sparse.ply"));
    QFile oldFile(oldSparsePath);
    ASSERT_TRUE(oldFile.open(QIODevice::WriteOnly));
    ASSERT_GT(oldFile.write("ply"), 0);
    oldFile.close();

    const QJsonObject oldRecord{
        {QStringLiteral("sparse_point_count"), 100},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), oldSparsePath}}}
    };
    ASSERT_TRUE(projectData.appendResultRecord(QStringLiteral("aerial_triangulation_results"), oldRecord));

    const QJsonObject missingRecord{
        {QStringLiteral("sparse_point_count"), 200},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"),
                      QDir(tempDir.path()).filePath(QStringLiteral("missing.ply"))}}}
    };
    const auto result = xjw::gui::project::ProjectTiePointResultService::replaceCurrent(
        &projectData,
        missingRecord);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(oldSparsePath));
    const QJsonArray records = projectData.metadata()
                                   .value(QStringLiteral("aerial_triangulation_results"))
                                   .toArray();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records.first().toObject(), oldRecord);
}

TEST(TiePointResultServiceTest, ReplaceRejectsMissingListedSidecar)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("missing_sidecar.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("missing_sidecar")));
    const QString sparsePath = QDir(tempDir.path()).filePath(QStringLiteral("sparse.ply"));
    QFile sparseFile(sparsePath);
    ASSERT_TRUE(sparseFile.open(QIODevice::WriteOnly));
    ASSERT_GT(sparseFile.write("ply"), 0);
    sparseFile.close();

    const QJsonObject record{
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), sparsePath},
                     {QStringLiteral("quality_report"),
                      QDir(tempDir.path()).filePath(QStringLiteral("missing.json"))}}}
    };
    const auto result = xjw::gui::project::ProjectTiePointResultService::replaceCurrent(
        &projectData,
        record);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.isEmpty());
    EXPECT_TRUE(projectData.metadata()
                    .value(QStringLiteral("aerial_triangulation_results"))
                    .toArray()
                    .isEmpty());
    EXPECT_TRUE(QFileInfo::exists(sparsePath));
}

TEST(TiePointResultServiceTest, DeleteAllRemovesArtifactsBeforeClearingMetadata)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("delete.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("delete")));

    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("tie_points_delete"));
    ASSERT_TRUE(QDir().mkpath(outputDir));
    const QString sparsePath = QDir(outputDir).filePath(QStringLiteral("sparse.ply"));
    const QString reportPath = QDir(outputDir).filePath(QStringLiteral("report.json"));
    for (const QString &path : {sparsePath, reportPath})
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write("data"), 0);
    }

    const QJsonObject record{
        {QStringLiteral("output_dir"), outputDir},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), sparsePath},
                     {QStringLiteral("quality_report"), reportPath}}}
    };
    ASSERT_TRUE(projectData.appendResultRecord(QStringLiteral("aerial_triangulation_results"), record));

    const auto result = xjw::gui::project::ProjectTiePointResultService::deleteAll(&projectData);

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.removedRecordCount, 1);
    EXPECT_FALSE(QFileInfo::exists(sparsePath));
    EXPECT_FALSE(QFileInfo::exists(reportPath));
    EXPECT_FALSE(QFileInfo::exists(outputDir));
    EXPECT_TRUE(projectData.metadata()
                    .value(QStringLiteral("aerial_triangulation_results"))
                    .toArray()
                    .isEmpty());
}

TEST(TiePointResultServiceTest, DeleteAllKeepsMetadataWhenSparseCloudPathIsDirectory)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("unsafe_delete.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("unsafe_delete")));
    const QString invalidSparsePath = QDir(tempDir.path()).filePath(QStringLiteral("not_a_file"));
    ASSERT_TRUE(QDir().mkpath(invalidSparsePath));

    const QJsonObject record{
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), invalidSparsePath}}}
    };
    ASSERT_TRUE(projectData.appendResultRecord(QStringLiteral("aerial_triangulation_results"), record));

    const auto result = xjw::gui::project::ProjectTiePointResultService::deleteAll(&projectData);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(QFileInfo(invalidSparsePath).isDir());
    const QJsonArray records = projectData.metadata()
                                   .value(QStringLiteral("aerial_triangulation_results"))
                                   .toArray();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records.first().toObject(), record);
}

TEST(TiePointResultIntegrationTest, ReplacingTwiceKeepsOnlyLatestTiePointRecord)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("metadata_replace.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("metadata_replace")));
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.ply"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.ply"));
    for (const QString &path : {firstPath, secondPath})
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write("ply"), 0);
    }

    const auto firstResult = xjw::gui::project::replaceTiePointResult(
        &projectData,
        firstPath,
        100,
        QStringList{QStringLiteral("image_1.tif")},
        tempDir.path());
    ASSERT_TRUE(firstResult.success) << firstResult.errorMessage.toStdString();
    const QString firstGeneration = firstResult.reconstructionGenerationId;
    ASSERT_FALSE(firstGeneration.isEmpty());

    const auto secondResult = xjw::gui::project::replaceTiePointResult(
        &projectData,
        secondPath,
        200,
        QStringList{QStringLiteral("image_1.tif"), QStringLiteral("image_2.tif")},
        tempDir.path());
    ASSERT_TRUE(secondResult.success) << secondResult.errorMessage.toStdString();
    EXPECT_NE(secondResult.reconstructionGenerationId, firstGeneration);

    const QJsonArray records = projectData.metadata()
                                   .value(QStringLiteral("aerial_triangulation_results"))
                                   .toArray();
    ASSERT_EQ(records.size(), 1);
    const QJsonObject current = records.first().toObject();
    EXPECT_EQ(current.value(QStringLiteral("sparse_point_count")).toInt(), 200);
    EXPECT_EQ(current.value(QStringLiteral("files"))
                  .toObject()
                  .value(QStringLiteral("sparse_cloud_xyz"))
                  .toString(),
              secondPath);
    EXPECT_FALSE(QFileInfo::exists(firstPath));
    EXPECT_TRUE(QFileInfo::exists(secondPath));
}

TEST(AboutDialogTest, ResolvesPythonEnvironmentWithExplicitExecutableFirst)
{
    QProcessEnvironment env;
    env.insert(QStringLiteral("PLASCAN_PYTHON"), QStringLiteral("E:/fallback/python.exe"));

    EXPECT_EQ(AboutDialog::pythonEnvironmentPath(env), QStringLiteral("E:/fallback/python.exe"));

    env.insert(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE"), QStringLiteral("E:/code/plascan/.venv/Scripts/python.exe"));
    EXPECT_EQ(AboutDialog::pythonEnvironmentPath(env), QStringLiteral("E:/code/plascan/.venv/Scripts/python.exe"));
}

TEST(AboutDialogTest, ShowsSelectablePythonEnvironmentPath)
{
    QProcessEnvironment env;
    env.insert(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE"), QStringLiteral("E:/code/plascan/.venv/Scripts/python.exe"));

    AboutDialog dialog(env);
    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("关于 PlaScan"));

    auto *pythonLabel = dialog.findChild<QLabel *>(QStringLiteral("pythonEnvironmentValueLabel"));
    ASSERT_NE(pythonLabel, nullptr);
    EXPECT_EQ(pythonLabel->text(), QStringLiteral("E:/code/plascan/.venv/Scripts/python.exe"));
    EXPECT_TRUE(pythonLabel->textInteractionFlags().testFlag(Qt::TextSelectableByMouse));
}

TEST(AboutDialogTest, ShowsUnconfiguredMessageWhenPythonEnvIsMissing)
{
    const QProcessEnvironment env;

    AboutDialog dialog(env);
    auto *pythonLabel = dialog.findChild<QLabel *>(QStringLiteral("pythonEnvironmentValueLabel"));
    ASSERT_NE(pythonLabel, nullptr);
    EXPECT_TRUE(pythonLabel->text().contains(QStringLiteral("未配置")));
    EXPECT_TRUE(pythonLabel->text().contains(QStringLiteral("setup_python_runtime.py")));
}

TEST(PythonRuntimeLocatorTest, ResolvesRepositoryLocalVenvWhenEnvironmentIsMissing)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    const QString pythonPath = createFakeRuntimePython(sourceRoot);

    const QString resolved = xjw::common::runtime::resolvePythonExecutable(QProcessEnvironment(), sourceRoot);
    EXPECT_EQ(QFileInfo(resolved).absoluteFilePath(), QFileInfo(pythonPath).absoluteFilePath());
}

TEST(PythonRuntimeLocatorTest, ReadsGeneratedEnvironmentFileWhenVenvIsMissing)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    const QString pythonPath = QDir(sourceRoot).filePath(QStringLiteral("custom/python.exe"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(pythonPath).absolutePath()));
    QFile pythonFile(pythonPath);
    ASSERT_TRUE(pythonFile.open(QIODevice::WriteOnly | QIODevice::Text));
    pythonFile.write("python");
    pythonFile.close();

    const QString envPath = QDir(sourceRoot).filePath(QStringLiteral("build/env/plascan-env.json"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(envPath).absolutePath()));
    QFile envFile(envPath);
    ASSERT_TRUE(envFile.open(QIODevice::WriteOnly | QIODevice::Text));
    envFile.write(QJsonDocument(QJsonObject{
        {QStringLiteral("PLASCAN_PYTHON_EXECUTABLE"), pythonPath}
    }).toJson());
    envFile.close();

    const QString resolved = xjw::common::runtime::resolvePythonExecutable(QProcessEnvironment(), sourceRoot);
    EXPECT_EQ(QFileInfo(resolved).absoluteFilePath(), QFileInfo(pythonPath).absoluteFilePath());
}

TEST(PythonRuntimeLocatorTest, UsesExplicitEnvironmentValueBeforeRepositoryVenv)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    createFakeRuntimePython(sourceRoot);
    const QString configuredPath = QDir(sourceRoot).filePath(QStringLiteral("configured/python.exe"));

    QProcessEnvironment environment;
    environment.insert(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE"), configuredPath);
    EXPECT_EQ(xjw::common::runtime::resolvePythonExecutable(environment, sourceRoot), configuredPath);
}

TEST(PythonRuntimeLocatorTest, GuiStartupBindsPythonRuntimeBeforeMainWindowIsShown)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int appStart = source.indexOf(QStringLiteral("SafeApplication app(argc, argv);"));
    ASSERT_GE(appStart, 0);
    const int bindStart = source.indexOf(QStringLiteral("bindPythonRuntime();"), appStart);
    ASSERT_GT(bindStart, appStart);
    const int windowStart = source.indexOf(QStringLiteral("MainWindow mainWindow;"), appStart);
    ASSERT_GT(windowStart, bindStart);
}

TEST(JsonObjectUtilityTest, DeepMergeRecursivelyPreservesUntouchedObjectMembers)
{
    const QJsonObject base{
        {QStringLiteral("render"), QJsonObject{{QStringLiteral("background"), QStringLiteral("black")},
                                                {QStringLiteral("opacity"), 0.5}}},
        {QStringLiteral("items"), QJsonArray{1, 2}},
    };
    const QJsonObject patch{
        {QStringLiteral("render"), QJsonObject{{QStringLiteral("opacity"), 0.8}}},
        {QStringLiteral("items"), QJsonArray{3}},
    };

    const QJsonObject merged = xjw::common::json::deepMergeObjects(base, patch);
    const QJsonObject render = merged.value(QStringLiteral("render")).toObject();
    EXPECT_EQ(render.value(QStringLiteral("background")).toString(), QStringLiteral("black"));
    EXPECT_DOUBLE_EQ(render.value(QStringLiteral("opacity")).toDouble(), 0.8);
    EXPECT_EQ(merged.value(QStringLiteral("items")).toArray(), QJsonArray({3}));
}

TEST(DialogSettingStoreTest, DoesNotOverwriteCorruptProjectDialogJson)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString plascanPath = QDir(tempDir.path()).filePath(QStringLiteral("sample.plascan"));
    const QString dialogPath = QDir(tempDir.path()).filePath(QStringLiteral("project_dialog.json"));
    const QByteArray invalidJson("{ invalid json");
    QFile dialogFile(dialogPath);
    ASSERT_TRUE(dialogFile.open(QIODevice::WriteOnly));
    ASSERT_EQ(dialogFile.write(invalidJson), invalidJson.size());
    dialogFile.close();

    DialogSettingStore store(QStringLiteral("model"));
    store.setProjectPath(plascanPath);
    QString errorMessage;
    EXPECT_FALSE(store.merge(QJsonObject{{QStringLiteral("quality"), QStringLiteral("high")}}, &errorMessage));
    EXPECT_TRUE(errorMessage.contains(QStringLiteral("无法解析 JSON 文件")));

    QString readError;
    EXPECT_EQ(xjw::common::io::readFileBytes(dialogPath, &readError), invalidJson);
    EXPECT_TRUE(readError.isEmpty());
}

TEST(DialogSettingStoreTest, CreatesAndMergesProjectDialogJson)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString plascanPath = QDir(tempDir.path()).filePath(QStringLiteral("sample.plascan"));
    DialogSettingStore store(QStringLiteral("model"));
    store.setProjectPath(plascanPath);
    ASSERT_TRUE(store.merge(QJsonObject{{QStringLiteral("quality"), QStringLiteral("high")}}));
    ASSERT_TRUE(store.merge(QJsonObject{{QStringLiteral("region"), QJsonObject{{QStringLiteral("size"), 250}}}}));

    const auto result = xjw::common::io::readJsonObjectFile(
        QDir(tempDir.path()).filePath(QStringLiteral("project_dialog.json")));
    ASSERT_TRUE(result.success);
    const QJsonObject settings = result.object.value(QStringLiteral("model")).toObject();
    EXPECT_EQ(settings.value(QStringLiteral("quality")).toString(), QStringLiteral("high"));
    EXPECT_EQ(settings.value(QStringLiteral("region")).toObject().value(QStringLiteral("size")).toInt(), 250);
}

TEST(MainMenuTest, HelpAboutActionOpensAboutDialog)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("#include \"application/AboutDialog.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("AboutDialog dialog(mw);")));
    EXPECT_TRUE(source.contains(QStringLiteral("dialog.exec()")));
    EXPECT_FALSE(source.contains(QStringLiteral("PlaScan: 行星表面摄影测量处理系统\"), 3000)")))
        << "The About action should open a dialog instead of only writing a transient status bar message.";
}

TEST(GenerateMaskDialogTest, DefaultsToBlackBackgroundReplacementForSelectedImages)
{
    GenerateMaskDialog dialog(QStringList{QStringLiteral("a.png"), QStringLiteral("b.png")});
    const QJsonObject settings = dialog.collectSettings();

    EXPECT_EQ(settings.value(QStringLiteral("method")).toString(), QStringLiteral("black_background"));
    EXPECT_EQ(settings.value(QStringLiteral("operation")).toString(), QStringLiteral("replace"));
    EXPECT_EQ(settings.value(QStringLiteral("scope")).toString(), QStringLiteral("selected_images"));
    EXPECT_TRUE(settings.value(QStringLiteral("auto_threshold")).toBool());
    EXPECT_GE(settings.value(QStringLiteral("min_component_area")).toInt(), 1);
    EXPECT_GE(settings.value(QStringLiteral("morphology_radius")).toInt(), 0);
}

TEST(GenerateMaskDialogTest, ShowsOnlyParametersForSelectedMethod)
{
    GenerateMaskDialog dialog(QStringList{QStringLiteral("a.png")});
    auto *methodCombo = dialog.findChild<QComboBox *>(QStringLiteral("methodCombo"));
    auto *thresholdPanel = dialog.findChild<QWidget *>(QStringLiteral("thresholdParameterPanel"));
    auto *sam21Panel = dialog.findChild<QWidget *>(QStringLiteral("sam21ParameterPanel"));
    auto *u2netPanel = dialog.findChild<QWidget *>(QStringLiteral("u2netParameterPanel"));

    ASSERT_NE(methodCombo, nullptr);
    ASSERT_NE(thresholdPanel, nullptr);
    ASSERT_NE(sam21Panel, nullptr);
    ASSERT_NE(u2netPanel, nullptr);

    const auto selectMethod = [methodCombo](const QString &token)
    {
        const int index = methodCombo->findData(token);
        ASSERT_GE(index, 0);
        methodCombo->setCurrentIndex(index);
    };

    selectMethod(QStringLiteral("black_background"));
    EXPECT_FALSE(thresholdPanel->isHidden());
    EXPECT_TRUE(sam21Panel->isHidden());
    EXPECT_TRUE(u2netPanel->isHidden());

    selectMethod(QStringLiteral("u2net"));
    EXPECT_TRUE(thresholdPanel->isHidden());
    EXPECT_TRUE(sam21Panel->isHidden());
    EXPECT_FALSE(u2netPanel->isHidden());

    selectMethod(QStringLiteral("sam21"));
    EXPECT_TRUE(thresholdPanel->isHidden());
    EXPECT_FALSE(sam21Panel->isHidden());
    EXPECT_TRUE(u2netPanel->isHidden());
}

TEST(GenerateMaskDialogTest, ExposesSam21TorchScriptCpuAndCudaSettings)
{
    GenerateMaskDialog dialog(QStringList{QStringLiteral("a.png")});
    auto *methodCombo = dialog.findChild<QComboBox *>(QStringLiteral("methodCombo"));
    auto *deviceCombo = dialog.findChild<QComboBox *>(QStringLiteral("sam21DeviceCombo"));
    auto *variantCombo = dialog.findChild<QComboBox *>(QStringLiteral("sam21VariantCombo"));
    auto *fallbackCheck = dialog.findChild<QCheckBox *>(QStringLiteral("sam21AllowFallbackCheck"));

    ASSERT_NE(methodCombo, nullptr);
    ASSERT_NE(deviceCombo, nullptr);
    ASSERT_NE(variantCombo, nullptr);
    ASSERT_NE(fallbackCheck, nullptr);

    const int samIndex = methodCombo->findData(QStringLiteral("sam21"));
    ASSERT_GE(samIndex, 0);
    methodCombo->setCurrentIndex(samIndex);

    EXPECT_GE(deviceCombo->findData(QStringLiteral("cuda")), 0);
    EXPECT_GE(deviceCombo->findData(QStringLiteral("cpu")), 0);
    EXPECT_EQ(variantCombo->currentData().toString(), QStringLiteral("tiny"));
    EXPECT_TRUE(fallbackCheck->isChecked());

    const QJsonObject settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("method")).toString(), QStringLiteral("sam21"));
    EXPECT_EQ(settings.value(QStringLiteral("sam21_variant")).toString(), QStringLiteral("tiny"));
    EXPECT_EQ(settings.value(QStringLiteral("sam21_device")).toString(), QStringLiteral("cuda"));
    EXPECT_TRUE(settings.value(QStringLiteral("sam21_allow_fallback")).toBool());
    EXPECT_EQ(settings.value(QStringLiteral("sam21_prompt_mode")).toString(), QStringLiteral("full_image_box"));
}

TEST(GenerateMaskDialogTest, ExposesU2NetOnnxCpuAndCudaSettings)
{
    GenerateMaskDialog dialog(QStringList{QStringLiteral("a.png")});
    auto *methodCombo = dialog.findChild<QComboBox *>(QStringLiteral("methodCombo"));
    auto *deviceCombo = dialog.findChild<QComboBox *>(QStringLiteral("u2netDeviceCombo"));
    auto *fallbackCheck = dialog.findChild<QCheckBox *>(QStringLiteral("u2netAllowFallbackCheck"));
    auto *inputSizeSpin = dialog.findChild<QSpinBox *>(QStringLiteral("u2netInputSizeSpin"));
    auto *thresholdSpin = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("u2netMaskThresholdSpin"));
    auto *statusLabel = dialog.findChild<QLabel *>(QStringLiteral("u2netModelStatusLabel"));

    ASSERT_NE(methodCombo, nullptr);
    ASSERT_NE(deviceCombo, nullptr);
    ASSERT_NE(fallbackCheck, nullptr);
    ASSERT_NE(inputSizeSpin, nullptr);
    ASSERT_NE(thresholdSpin, nullptr);
    ASSERT_NE(statusLabel, nullptr);

    const int u2netIndex = methodCombo->findData(QStringLiteral("u2net"));
    ASSERT_GE(u2netIndex, 0);
    methodCombo->setCurrentIndex(u2netIndex);

    EXPECT_GE(deviceCombo->findData(QStringLiteral("cuda")), 0);
    EXPECT_GE(deviceCombo->findData(QStringLiteral("cpu")), 0);
    EXPECT_TRUE(fallbackCheck->isChecked());
    EXPECT_EQ(inputSizeSpin->value(), 320);
    EXPECT_DOUBLE_EQ(thresholdSpin->value(), 0.5);
    EXPECT_TRUE(statusLabel->isEnabled());

    const QJsonObject settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("method")).toString(), QStringLiteral("u2net"));
    EXPECT_EQ(settings.value(QStringLiteral("u2net_device")).toString(), QStringLiteral("cuda"));
    EXPECT_TRUE(settings.value(QStringLiteral("u2net_allow_fallback")).toBool());
    EXPECT_EQ(settings.value(QStringLiteral("u2net_input_size")).toInt(), 320);
    EXPECT_DOUBLE_EQ(settings.value(QStringLiteral("u2net_mask_threshold")).toDouble(), 0.5);
}

TEST(GenerateMaskWorkflowTest, ProjectManagerUsesCommonIoForTiffMaskGeneration)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectManager::openGenerateMaskDialog"));
    const int end = source.indexOf(QStringLiteral("void ProjectManager::runReferenceQualityCheck"), start);
    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);

    const QString block = source.mid(start, end - start);
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"io/PathIO.h\"")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::common::io::readImage(imagePath, cv::IMREAD_UNCHANGED)")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::common::io::readImage(maskPath, cv::IMREAD_GRAYSCALE)")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::common::io::writeImage(maskPath, generated)")));
    EXPECT_FALSE(block.contains(QStringLiteral("QImage sourceImage(imagePath)")))
        << "Mask generation must not use QImage for source TIFF reading; use common/io PathIO instead.";
}

TEST(GenerateMaskWorkflowTest, Sam21MaskGenerationUsesTorchScriptAndAutoPrompt)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectManager::openGenerateMaskDialog"));
    const int end = source.indexOf(QStringLiteral("void ProjectManager::runReferenceQualityCheck"), start);
    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);

    const QString block = source.mid(start, end - start);
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"Sam21MaskGenerator.h\"")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::mask::Sam21MaskGenerator")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::mask::Sam21Prompt::autoBox")));
    EXPECT_TRUE(block.contains(QStringLiteral("sam21TorchScriptModelNames")));
    EXPECT_FALSE(block.contains(QStringLiteral("methodToken == QStringLiteral(\"sam21\") && options.method")))
        << "SAM2.1 must not be implemented as a black-background variant.";
}

TEST(GenerateMaskWorkflowTest, U2NetMaskGenerationUsesBundledOnnxAndOpenCvDnn)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString cmake = readProjectSourceFile(QStringLiteral("src/core/mask/CMakeLists.txt"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(cmake.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectManager::openGenerateMaskDialog"));
    const int end = source.indexOf(QStringLiteral("void ProjectManager::runReferenceQualityCheck"), start);
    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);

    const QString block = source.mid(start, end - start);
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"u2net/U2NetMaskGenerator.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"model/U2NetModelCatalog.h\"")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("u2net/U2NetMaskGenerator.cpp")))
        << "U2Net should live in src/core/mask/u2net as its own mask submodule.";
    EXPECT_TRUE(readProjectSourceFile(QStringLiteral("src/core/mask/u2net/U2NetMaskGenerator.h"))
                    .contains(QStringLiteral("class U2NetMaskGenerator")));
    EXPECT_TRUE(readProjectSourceFile(QStringLiteral("src/core/mask/u2net/U2NetMaskGenerator.cpp"))
                    .contains(QStringLiteral("#include \"U2NetMaskGenerator.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("u2netMaskConfigFromSettings")));
    EXPECT_TRUE(block.contains(QStringLiteral("methodToken == QLatin1String(\"u2net\")")));
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::mask::U2NetMaskGenerator")));
    EXPECT_TRUE(block.contains(QStringLiteral("u2netGenerator->generate(source)")));
    EXPECT_TRUE(block.contains(QStringLiteral("U2Net_v1.onnx")));
    EXPECT_TRUE(block.contains(QStringLiteral("deviceLabel()")));
}

TEST(GenerateMaskWorkflowTest, RunsMaskGenerationOffGuiThreadWithTaskStatusProgress)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("void ProjectManager::openGenerateMaskDialog"));
    const int end = source.indexOf(QStringLiteral("void ProjectManager::runReferenceQualityCheck"), start);
    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);

    const QString block = source.mid(start, end - start);
    EXPECT_TRUE(block.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "SAM2.1 mask generation must not run on the GUI thread.";
    EXPECT_TRUE(block.contains(QStringLiteral("std::atomic<bool>")))
        << "Mask generation should keep a worker-visible cancel flag.";
    EXPECT_TRUE(block.contains(QStringLiteral("if (_maskGenerationCancelFlag)")))
        << "Without a modal progress dialog, duplicate mask generation runs must be rejected.";
    EXPECT_TRUE(block.contains(QStringLiteral("reportProgress(++completed)")))
        << "The worker should report per-image progress while generating masks.";
    EXPECT_TRUE(block.contains(QStringLiteral("emit self->maskGenerationProgressChanged")))
        << "The GUI should show long SAM2.1 inference progress in the main task status area.";
    EXPECT_FALSE(block.contains(QStringLiteral("new QProgressDialog")))
        << "Mask generation progress belongs in the main-window task status area.";

    const int acceptedStart = block.indexOf(QStringLiteral("dialog.collectSettings()"));
    ASSERT_GE(acceptedStart, 0);
    const int runnerStart = block.indexOf(QStringLiteral("xjw::gui::tasks::runGuarded"), acceptedStart);
    ASSERT_GT(runnerStart, acceptedStart);
    const QString guiThreadBlock = block.mid(acceptedStart, runnerStart - acceptedStart);
    EXPECT_FALSE(guiThreadBlock.contains(QStringLiteral("std::make_unique<xjw::mask::Sam21MaskGenerator>")))
        << "Loading SAM2.1 TorchScript models must happen in the worker thread.";
    EXPECT_FALSE(guiThreadBlock.contains(QStringLiteral("sam21Generator->generate")))
        << "SAM2.1 inference must happen in the worker thread.";
}

TEST(GenerateMaskWorkflowTest, UsesMainWindowTaskStatusInsteadOfModalProgressDialog)
{
    const QString managerHeader = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString managerSource = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString mainHeader = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString mainSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(managerHeader.isEmpty());
    ASSERT_FALSE(managerSource.isEmpty());
    ASSERT_FALSE(mainHeader.isEmpty());
    ASSERT_FALSE(mainSource.isEmpty());

    const int start = managerSource.indexOf(QStringLiteral("void ProjectManager::openGenerateMaskDialog"));
    const int end = managerSource.indexOf(QStringLiteral("void ProjectManager::runReferenceQualityCheck"), start);
    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);
    const QString block = managerSource.mid(start, end - start);

    EXPECT_TRUE(managerHeader.contains(QStringLiteral("void maskGenerationProgressChanged(const QString &stage, int done, int total);")));
    EXPECT_TRUE(managerHeader.contains(QStringLiteral("void maskGenerationFinished(bool success);")));
    EXPECT_TRUE(managerHeader.contains(QStringLiteral("void cancelMaskGeneration();")));
    EXPECT_TRUE(block.contains(QStringLiteral("emit self->maskGenerationProgressChanged")));
    EXPECT_TRUE(block.contains(QStringLiteral("emit self->maskGenerationFinished")));
    EXPECT_FALSE(block.contains(QStringLiteral("new QProgressDialog")))
        << "Mask generation progress belongs in the main-window task status area, not a modal dialog.";

    EXPECT_TRUE(mainHeader.contains(QStringLiteral("TaskStatusWidget* _maskTaskStatus{};")));
    EXPECT_TRUE(mainHeader.contains(QStringLiteral("void onMaskGenerationProgress(const QString &stage, int done, int total);")));
    EXPECT_TRUE(mainHeader.contains(QStringLiteral("void onMaskGenerationFinished(bool success);")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("&ProjectManager::maskGenerationProgressChanged")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("&ProjectManager::maskGenerationFinished")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("_maskTaskStatus = createTaskStatus")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("生成蒙版 %1/%2")));
}

TEST(GenerateMaskWorkflowTest, DialogShowsSam21InstallStateAndButton)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/image/GenerateMaskDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/image/GenerateMaskDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("sam21InstallRequested")))
        << "The mask dialog should expose a model-install request signal.";
    EXPECT_TRUE(header.contains(QStringLiteral("refreshSam21ModelStatus")))
        << "The dialog must refresh installed/missing model labels after installation.";
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"model/Sam21ModelCatalog.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("xjw::common::model::sam21ModelStatus")));
    EXPECT_TRUE(source.contains(QStringLiteral("安装模型")));
    EXPECT_TRUE(source.contains(QStringLiteral("未安装")));
}

TEST(GenerateMaskWorkflowTest, ProjectManagerInstallsSam21ModelsWithBundledPythonProcess)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("void installSam21Model(const QString &variantToken")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include <QProcess>")));
    EXPECT_TRUE(source.contains(QStringLiteral("install_sam21_model.py")));
    EXPECT_TRUE(source.contains(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE")));
    EXPECT_TRUE(source.contains(QStringLiteral("PLASCAN_PYTHON")));
    EXPECT_TRUE(source.contains(QStringLiteral("new QProgressDialog")));
    EXPECT_TRUE(source.contains(QStringLiteral("process->start(pythonExecutable, arguments)")))
        << "Model installation should be launched as an asynchronous process, not by blocking the GUI thread.";
}

TEST(CanvasWidgetTest, ExposesMaskContourOverlayApi)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    const QString rendererHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(rendererHeader.isEmpty());
    ASSERT_FALSE(rendererSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("void reloadMaskOverlay()")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setShowMaskOverlay(bool show)")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("addMaskContourLayer")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("clearMaskLayers")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("extractMaskContours")));
    EXPECT_TRUE(source.contains(QStringLiteral("xjw::common::project::ProjectIO::findMaskForImage")));
}

TEST(TiePointsDialogTest, MetashapeStyleDefaultsAreExposed)
{
    CreateTiePointsDialog createDialog;
    EXPECT_EQ(createDialog.accuracy(), QStringLiteral("highest"));
    EXPECT_EQ(createDialog.keypointLimit(), 40000);
    EXPECT_EQ(createDialog.keypointLimitPerMegapixel(), 1000);
    EXPECT_EQ(createDialog.tiePointLimit(), 4000);
    EXPECT_TRUE(createDialog.useGenericPreselection());
    EXPECT_FALSE(createDialog.useReferencePreselection());
    EXPECT_FALSE(createDialog.useGuidedMatching());
    EXPECT_TRUE(createDialog.excludePinnedTiePoints());
    EXPECT_EQ(createDialog.maskApplyMode(), QStringLiteral("none"));

    auto *maskModeCombo = createDialog.findChild<QComboBox *>(QStringLiteral("m_maskModeCombo"));
    ASSERT_NE(maskModeCombo, nullptr);
    EXPECT_TRUE(maskModeCombo->isEnabled());
    EXPECT_GE(maskModeCombo->findData(QStringLiteral("none")), 0);
    EXPECT_GE(maskModeCombo->findData(QStringLiteral("keypoints")), 0);
    EXPECT_GE(maskModeCombo->findData(QStringLiteral("tiepoints")), 0);

    const int tiePointMaskIndex = maskModeCombo->findData(QStringLiteral("tiepoints"));
    ASSERT_GE(tiePointMaskIndex, 0);
    maskModeCombo->setCurrentIndex(tiePointMaskIndex);
    EXPECT_EQ(createDialog.maskApplyMode(), QStringLiteral("tiepoints"));

    ThinTiePointsDialog thinDialog;
    EXPECT_EQ(thinDialog.tiePointLimit(), 500);

    CleanTiePointsDialog cleanDialog;
    EXPECT_EQ(cleanDialog.criterion(), CleanTiePointsDialog::Criterion::None);
    EXPECT_FALSE(cleanDialog.deleteRequested());
}

TEST(TiePointsDialogTest, GuidedMatchingSwitchesKeypointLimitToPerMegapixelDisplay)
{
    CreateTiePointsDialog dialog;
    auto *guidedCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_guidedMatchingCheck"));
    auto *keypointLabel = dialog.findChild<QLabel *>(QStringLiteral("m_keypointLimitLabel"));

    ASSERT_NE(guidedCheck, nullptr);
    ASSERT_NE(keypointLabel, nullptr);

    EXPECT_EQ(keypointLabel->text(), QStringLiteral("关键点限制:"));
    guidedCheck->setChecked(true);

    EXPECT_TRUE(dialog.useGuidedMatching());
    EXPECT_EQ(dialog.keypointLimitPerMegapixel(), 1000);
    EXPECT_EQ(keypointLabel->text(), QStringLiteral("每百万像素的关键点限制:"));
}

TEST(TiePointsDialogTest, AdvancedSectionIsCollapsible)
{
    CreateTiePointsDialog dialog;
    auto *advancedToggle = dialog.findChild<QToolButton *>(QStringLiteral("m_advancedToggle"));
    auto *advancedContent = dialog.findChild<QWidget *>(QStringLiteral("m_advancedContent"));
    auto *generalGroup = dialog.findChild<QGroupBox *>(QStringLiteral("m_generalGroup"));
    auto *advancedGroup = dialog.findChild<QGroupBox *>(QStringLiteral("m_advancedGroup"));
    auto *accuracyCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_accuracyCombo"));

    ASSERT_NE(advancedToggle, nullptr);
    ASSERT_NE(advancedContent, nullptr);
    ASSERT_NE(generalGroup, nullptr);
    ASSERT_NE(advancedGroup, nullptr);
    ASSERT_NE(accuracyCombo, nullptr);

    EXPECT_EQ(advancedToggle->text(), QStringLiteral("高级"));
    EXPECT_TRUE(advancedToggle->isCheckable());
    EXPECT_TRUE(advancedToggle->isChecked());
    EXPECT_EQ(advancedToggle->arrowType(), Qt::DownArrow);
    EXPECT_FALSE(advancedContent->isHidden());
    EXPECT_TRUE(advancedGroup->isAncestorOf(advancedContent));
    EXPECT_EQ(accuracyCombo->currentData().toString(), QStringLiteral("highest"));

    dialog.show();
    QApplication::processEvents();
    const int expandedHeight = dialog.height();

    advancedToggle->setChecked(false);
    QApplication::processEvents();
    EXPECT_TRUE(advancedContent->isHidden());
    EXPECT_TRUE(advancedGroup->isHidden());
    EXPECT_EQ(advancedToggle->arrowType(), Qt::RightArrow);
    EXPECT_LT(dialog.height(), expandedHeight - 80);
    EXPECT_EQ(dialog.keypointLimit(), 40000);
    EXPECT_EQ(dialog.tiePointLimit(), 4000);

    advancedToggle->setChecked(true);
    QApplication::processEvents();
    EXPECT_FALSE(advancedContent->isHidden());
    EXPECT_EQ(advancedToggle->arrowType(), Qt::DownArrow);
}

TEST(TiePointsDialogTest, MainWindowPassesCreateTiePointsAdvancedOptionsToTask)
{
    const QString mainWindowSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/TiePointWorkflowController.cpp"));
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("options.useExplicitKeypointLimit = true")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("options.keypointLimitPerMegapixel")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("options.maxTiePointsPerImage = dlg.tiePointLimit()")));
    EXPECT_TRUE(mainWindowSource.contains(
        QStringLiteral("options.excludeStationaryTiePoints = dlg.excludePinnedTiePoints()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("options.maskApplyMode = dlg.maskApplyMode()")));
    EXPECT_TRUE(controllerSource.contains(
        QStringLiteral("context.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages(projectPath, images)")))
        << "创建连接点任务必须把项目中的蒙版文件传入核心匹配上下文。";
}

TEST(MainMenuTest, ModelMenuOwnsCheckedCameraVisibilityAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.toggleCamerasAction();
    ASSERT_NE(action, nullptr);
    EXPECT_EQ(action->text(), QStringLiteral("显示相机"));
    EXPECT_TRUE(action->isCheckable());
    EXPECT_TRUE(action->isChecked());

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    EXPECT_FALSE(viewMenu->actions().contains(action));

    QMenu *modelMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("模型"));
    ASSERT_NE(modelMenu, nullptr);
    QMenu *displayMenu = findSubMenuByTitle(modelMenu, QStringLiteral("显示/隐藏项目"));
    ASSERT_NE(displayMenu, nullptr);
    EXPECT_TRUE(displayMenu->actions().contains(action));
}

TEST(MainMenuTest, ModelMenuExposesMetashapeStyleDisplayHideActions)
{
    QMainWindow window;
    MainMenu menu(&window);

    QMenu *modelMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("模型"));
    ASSERT_NE(modelMenu, nullptr);
    QMenu *displayMenu = findSubMenuByTitle(modelMenu, QStringLiteral("显示/隐藏项目"));
    ASSERT_NE(displayMenu, nullptr);

    ASSERT_NE(menu.toggleGizmoAction(), nullptr);
    ASSERT_NE(menu.toggleCamerasAction(), nullptr);
    ASSERT_NE(menu.toggleDependentCamerasAction(), nullptr);
    ASSERT_NE(menu.toggleCameraThumbnailsAction(), nullptr);
    ASSERT_NE(menu.toggleCameraImagesAction(), nullptr);
    ASSERT_NE(menu.showCameraImagesInForegroundAction(), nullptr);
    ASSERT_NE(menu.showCameraImagesInBackgroundAction(), nullptr);
    ASSERT_NE(menu.lockCameraImageAction(), nullptr);

    EXPECT_TRUE(displayMenu->actions().contains(menu.toggleGizmoAction()));
    EXPECT_TRUE(displayMenu->actions().contains(menu.toggleCamerasAction()));
    EXPECT_TRUE(displayMenu->actions().contains(menu.toggleDependentCamerasAction()));
    EXPECT_TRUE(displayMenu->actions().contains(menu.toggleCameraThumbnailsAction()));
    QMenu *imageMenu = findSubMenuByTitle(displayMenu, QStringLiteral("显示图像"));
    ASSERT_NE(imageMenu, nullptr);
    EXPECT_TRUE(imageMenu->actions().contains(menu.toggleCameraImagesAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.showCameraImagesInForegroundAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.showCameraImagesInBackgroundAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.lockCameraImageAction()));

    EXPECT_EQ(menu.toggleGizmoAction()->text(), QStringLiteral("显示轨迹球"));
    EXPECT_TRUE(menu.toggleGizmoAction()->isCheckable());
    EXPECT_TRUE(menu.toggleGizmoAction()->isChecked());

    EXPECT_EQ(menu.toggleDependentCamerasAction()->text(), QStringLiteral("显示从属相机"));
    EXPECT_FALSE(menu.toggleDependentCamerasAction()->isEnabled());
    EXPECT_TRUE(menu.toggleDependentCamerasAction()->toolTip().contains(QStringLiteral("暂未建立模型与从属相机关系")));

    EXPECT_EQ(menu.toggleCameraThumbnailsAction()->text(), QStringLiteral("显示缩略图"));
    EXPECT_TRUE(menu.toggleCameraThumbnailsAction()->isCheckable());
    EXPECT_TRUE(menu.toggleCameraThumbnailsAction()->isChecked());
    EXPECT_EQ(menu.toggleCameraThumbnailsAction()->actionGroup(), nullptr);
    menu.toggleCameraThumbnailsAction()->setChecked(false);
    EXPECT_FALSE(menu.toggleCameraThumbnailsAction()->isChecked());

    EXPECT_EQ(menu.toggleCameraImagesAction()->text(), QStringLiteral("显示图像"));
    EXPECT_TRUE(menu.toggleCameraImagesAction()->isCheckable());
    EXPECT_FALSE(menu.toggleCameraImagesAction()->isChecked());
    EXPECT_EQ(menu.showCameraImagesInForegroundAction()->text(), QStringLiteral("在前景中显示"));
    EXPECT_TRUE(menu.showCameraImagesInForegroundAction()->isCheckable());
    EXPECT_TRUE(menu.showCameraImagesInForegroundAction()->isChecked());
    EXPECT_EQ(menu.showCameraImagesInBackgroundAction()->text(), QStringLiteral("在后景中显示"));
    EXPECT_TRUE(menu.showCameraImagesInBackgroundAction()->isCheckable());
    EXPECT_FALSE(menu.showCameraImagesInBackgroundAction()->isChecked());
    EXPECT_EQ(menu.lockCameraImageAction()->text(), QStringLiteral("锁定图像"));
    EXPECT_TRUE(menu.lockCameraImageAction()->isCheckable());
    EXPECT_FALSE(menu.lockCameraImageAction()->isChecked());
}

TEST(MainMenuTest, WindowMenuUsesCompactNativeActions)
{
    QMainWindow window;
    MainMenu menu(&window);

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    QMenu *windowMenu = findSubMenuByTitle(viewMenu, QStringLiteral("窗口"));
    ASSERT_NE(windowMenu, nullptr);

    QStringList actionTexts;
    for (QAction *action : windowMenu->actions())
    {
        ASSERT_NE(action, nullptr);
        EXPECT_EQ(qobject_cast<QWidgetAction *>(action), nullptr)
            << "窗口子菜单应使用原生 QAction，不能嵌入大按钮面板。";
        if (!action->isSeparator())
        {
            actionTexts.push_back(action->text());
            if (action->text() == QStringLiteral("恢复默认窗口布局"))
            {
                EXPECT_FALSE(action->isCheckable());
            }
            else
            {
                EXPECT_TRUE(action->isCheckable());
            }
        }
    }

    EXPECT_EQ(actionTexts,
              QStringList({QStringLiteral("工作区"),
                           QStringLiteral("属性"),
                           QStringLiteral("照片"),
                           QStringLiteral("日志"),
                           QStringLiteral("主工具栏"),
                           QStringLiteral("河南大学校徽"),
                           QStringLiteral("恢复默认窗口布局")}));

    QAction *toolbarAction = window.findChild<QAction *>(QStringLiteral("actionToggleMainToolbar"));
    ASSERT_NE(toolbarAction, nullptr);
    EXPECT_TRUE(toolbarAction->isChecked());
    QAction *restoreAction = menu.restoreDefaultWindowLayoutAction();
    ASSERT_NE(restoreAction, nullptr);
    EXPECT_FALSE(restoreAction->isCheckable());
}

TEST(MainMenuTest, ViewMenuSeparatesImageAndModelDisplayCommands)
{
    QMainWindow window;
    MainMenu menu(&window);

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    QMenu *imageMenu = findSubMenuByTitle(viewMenu, QStringLiteral("影像显示"));
    ASSERT_NE(imageMenu, nullptr);

    EXPECT_FALSE(viewMenu->actions().contains(menu.toggleGizmoAction()));
    EXPECT_FALSE(viewMenu->actions().contains(menu.toggleCamerasAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.rotateImageLeftAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.rotateImageRightAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.showFeaturePointsAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.showFeatureResidualsAction()));
    EXPECT_TRUE(imageMenu->actions().contains(menu.showMaskOverlayAction()));

    QAction *fullScreenAction = window.findChild<QAction *>(QStringLiteral("actionToggleFullScreen"));
    ASSERT_NE(fullScreenAction, nullptr);
    EXPECT_EQ(fullScreenAction->shortcut(), QKeySequence(Qt::Key_F11));
}

TEST(MainMenuTest, ModelContextDisablesImageOnlyCommands)
{
    QMainWindow window;
    MainMenu menu(&window);

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    QMenu *imageMenu = findSubMenuByTitle(viewMenu, QStringLiteral("影像显示"));
    ASSERT_NE(imageMenu, nullptr);

    menu.setContextualToolbarVisibility(true, false);

    EXPECT_FALSE(imageMenu->isEnabled());
    EXPECT_FALSE(menu.rotateImageLeftAction()->isEnabled());
    EXPECT_FALSE(menu.rotateImageRightAction()->isEnabled());
    EXPECT_FALSE(menu.showFeaturePointsAction()->isEnabled());
    EXPECT_FALSE(menu.showFeatureResidualsAction()->isEnabled());
    EXPECT_FALSE(menu.showMaskOverlayAction()->isEnabled());
    EXPECT_FALSE(menu.showDepthOverlayAction()->isEnabled());

    menu.setContextualToolbarVisibility(false, true);
    EXPECT_TRUE(imageMenu->isEnabled());
    EXPECT_FALSE(menu.rotateImageLeftAction()->isEnabled());
    EXPECT_FALSE(menu.showFeaturePointsAction()->isEnabled());

    menu.setImageDisplayReady(true);
    EXPECT_TRUE(menu.rotateImageLeftAction()->isEnabled());
    EXPECT_TRUE(menu.showFeaturePointsAction()->isEnabled());
    EXPECT_TRUE(menu.showMaskOverlayAction()->isEnabled());
    EXPECT_FALSE(menu.showDepthOverlayAction()->isEnabled());

    menu.setDepthOverlayAvailable(true);
    EXPECT_TRUE(menu.showDepthOverlayAction()->isEnabled());

    menu.setContextualToolbarVisibility(true, false);
    EXPECT_FALSE(imageMenu->isEnabled());
    EXPECT_FALSE(menu.showDepthOverlayAction()->isEnabled());
}

TEST(WorkspacePanelControllerTest, ProvidesOneRegistryForVisibilityAndPersistence)
{
    const QString header = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/WorkspacePanelController.h"));
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/WorkspacePanelController.cpp"));

    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(header.contains(QStringLiteral("bool registerDock")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool registerToolBar")));
    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject visibilitySnapshot() const")));
    EXPECT_TRUE(header.contains(QStringLiteral("QList<QAction *> actions")));
    EXPECT_TRUE(header.contains(QStringLiteral("void applyVisibility")));
    EXPECT_TRUE(header.contains(QStringLiteral("void ensureRequiredProjectPanelsVisible")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setPanelVisible")));
    EXPECT_TRUE(source.contains(QStringLiteral("QDockWidget::visibilityChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("QToolBar::visibilityChanged")));
}

TEST(WorkspacePanelControllerTest, StableDescriptorsDefinePersistenceAndRequiredPanels)
{
    const auto workspace =
        workspacePanelDescriptor(WorkspacePanelId::Workspace);
    const auto properties =
        workspacePanelDescriptor(WorkspacePanelId::Properties);
    const auto photos =
        workspacePanelDescriptor(WorkspacePanelId::Photos);
    const auto log =
        workspacePanelDescriptor(WorkspacePanelId::Log);
    const auto toolbar =
        workspacePanelDescriptor(WorkspacePanelId::MainToolbar);

    EXPECT_EQ(workspace.settingKey, QStringLiteral("workspace_visible"));
    EXPECT_EQ(properties.settingKey, QStringLiteral("properties_visible"));
    EXPECT_EQ(photos.settingKey, QStringLiteral("photos_visible"));
    EXPECT_EQ(log.settingKey, QStringLiteral("log_visible"));
    EXPECT_EQ(toolbar.settingKey, QStringLiteral("main_toolbar_visible"));
    EXPECT_TRUE(workspace.requiredForProject);
    EXPECT_TRUE(properties.requiredForProject);
    EXPECT_TRUE(photos.requiredForProject);
    EXPECT_FALSE(log.requiredForProject);
    EXPECT_FALSE(toolbar.requiredForProject);
    EXPECT_TRUE(workspace.defaultVisible);
    EXPECT_FALSE(log.defaultVisible);
    EXPECT_EQ(workspace.kind, WorkspacePanelKind::Dock);
    EXPECT_EQ(log.kind, WorkspacePanelKind::Dock);
    EXPECT_EQ(toolbar.kind, WorkspacePanelKind::ToolBar);
}

TEST(WorkspacePanelControllerTest, KeepsActionsWidgetsAndSnapshotInSync)
{
    QMainWindow window;
    QDockWidget dock(QStringLiteral("工作区"), &window);
    window.addDockWidget(Qt::LeftDockWidgetArea, &dock);
    QAction action(QStringLiteral("工作区"), &window);
    action.setCheckable(true);
    WorkspacePanelController controller;
    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Workspace, &action, &dock));

    window.show();
    QApplication::processEvents();
    EXPECT_TRUE(action.isChecked());

    action.setChecked(false);
    QApplication::processEvents();
    EXPECT_TRUE(dock.isHidden());
    EXPECT_FALSE(controller.visibilitySnapshot().value(
        QStringLiteral("workspace_visible")).toBool(true));

    dock.setVisible(true);
    QApplication::processEvents();
    EXPECT_TRUE(action.isChecked());
}

TEST(WorkspacePanelControllerTest, AppliesSavedVisibilityBeforeDefaults)
{
    QMainWindow window;
    QDockWidget dock(QStringLiteral("工作区"), &window);
    QToolBar toolBar(QStringLiteral("主工具栏"), &window);
    window.addDockWidget(Qt::LeftDockWidgetArea, &dock);
    window.addToolBar(&toolBar);
    QAction dockAction(QStringLiteral("工作区"), &window);
    QAction toolBarAction(QStringLiteral("主工具栏"), &window);
    WorkspacePanelController controller;
    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Workspace, &dockAction, &dock));
    ASSERT_TRUE(controller.registerToolBar(
        WorkspacePanelId::MainToolbar, &toolBarAction, &toolBar));

    controller.applyVisibility(QJsonObject{
        {QStringLiteral("workspace_visible"), false},
        {QStringLiteral("main_toolbar_visible"), false}
    });
    EXPECT_TRUE(dock.isHidden());
    EXPECT_TRUE(toolBar.isHidden());
    EXPECT_FALSE(dockAction.isChecked());
    EXPECT_FALSE(toolBarAction.isChecked());

    controller.applyVisibility(QJsonObject{});
    EXPECT_FALSE(dock.isHidden());
    EXPECT_FALSE(toolBar.isHidden());
    EXPECT_TRUE(dockAction.isChecked());
    EXPECT_TRUE(toolBarAction.isChecked());
}

TEST(WorkspacePanelControllerTest, TabSelectionDoesNotPersistInactiveDockAsHidden)
{
    QMainWindow window;
    QDockWidget workspaceDock(QStringLiteral("工作区"), &window);
    QDockWidget propertiesDock(QStringLiteral("属性"), &window);
    window.addDockWidget(Qt::LeftDockWidgetArea, &workspaceDock);
    window.addDockWidget(Qt::LeftDockWidgetArea, &propertiesDock);
    window.tabifyDockWidget(&workspaceDock, &propertiesDock);

    QAction workspaceAction(QStringLiteral("工作区"), &window);
    QAction propertiesAction(QStringLiteral("属性"), &window);
    WorkspacePanelController controller;
    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Workspace,
        &workspaceAction,
        &workspaceDock));
    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Properties,
        &propertiesAction,
        &propertiesDock));
    QSignalSpy visibilitySpy(&controller,
                             &WorkspacePanelController::visibilitySettingChanged);

    window.show();
    workspaceDock.raise();
    QApplication::processEvents();
    visibilitySpy.clear();

    propertiesDock.raise();
    QApplication::processEvents();

    EXPECT_EQ(visibilitySpy.count(), 0)
        << "切换 Dock 标签只改变遮挡状态，不应持久化为显式隐藏。";
    EXPECT_TRUE(workspaceAction.isChecked());
    EXPECT_TRUE(propertiesAction.isChecked());
    EXPECT_TRUE(controller.visibilitySnapshot()
                    .value(QStringLiteral("workspace_visible"))
                    .toBool(false));
    EXPECT_TRUE(controller.visibilitySnapshot()
                    .value(QStringLiteral("properties_visible"))
                    .toBool(false));
}

TEST(WorkspacePanelControllerTest, RejectsDuplicatePanelRegistration)
{
    QMainWindow window;
    QDockWidget firstDock(QStringLiteral("工作区"), &window);
    QDockWidget secondDock(QStringLiteral("另一个工作区"), &window);
    QAction firstAction(QStringLiteral("工作区"), &window);
    QAction secondAction(QStringLiteral("另一个工作区"), &window);
    WorkspacePanelController controller;

    EXPECT_TRUE(controller.registerDock(
        WorkspacePanelId::Workspace, &firstAction, &firstDock));
    EXPECT_FALSE(controller.registerDock(
        WorkspacePanelId::Workspace, &secondAction, &secondDock));

    controller.setPanelVisible(WorkspacePanelId::Workspace, false);
    EXPECT_TRUE(firstDock.isHidden());
    EXPECT_FALSE(firstAction.isChecked());
    EXPECT_FALSE(secondDock.isHidden());
}

TEST(WorkspacePanelControllerTest, RestoresRequiredPanelsWithoutOpeningOptionalPanels)
{
    QMainWindow window;
    QDockWidget workspaceDock(QStringLiteral("工作区"), &window);
    QDockWidget logDock(QStringLiteral("日志"), &window);
    window.addDockWidget(Qt::LeftDockWidgetArea, &workspaceDock);
    window.addDockWidget(Qt::BottomDockWidgetArea, &logDock);
    QAction workspaceAction(QStringLiteral("工作区"), &window);
    QAction logAction(QStringLiteral("日志"), &window);
    WorkspacePanelController controller;

    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Workspace, &workspaceAction, &workspaceDock));
    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Log, &logAction, &logDock));
    controller.setPanelVisible(WorkspacePanelId::Workspace, false);
    controller.setPanelVisible(WorkspacePanelId::Log, false);

    controller.ensureRequiredProjectPanelsVisible();

    EXPECT_FALSE(workspaceDock.isHidden());
    EXPECT_TRUE(workspaceAction.isChecked());
    EXPECT_TRUE(logDock.isHidden());
    EXPECT_FALSE(logAction.isChecked());
}

TEST(WorkspacePanelControllerTest, ExposesRegisteredActionsByWindowItemKind)
{
    QMainWindow window;
    QDockWidget workspaceDock(QStringLiteral("工作区"), &window);
    QDockWidget logDock(QStringLiteral("日志"), &window);
    QToolBar toolBar(QStringLiteral("主工具栏"), &window);
    QAction workspaceAction(QStringLiteral("工作区"), &window);
    QAction logAction(QStringLiteral("日志"), &window);
    QAction toolBarAction(QStringLiteral("主工具栏"), &window);
    WorkspacePanelController controller;

    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Workspace, &workspaceAction, &workspaceDock));
    ASSERT_TRUE(controller.registerDock(
        WorkspacePanelId::Log, &logAction, &logDock));
    ASSERT_TRUE(controller.registerToolBar(
        WorkspacePanelId::MainToolbar, &toolBarAction, &toolBar));

    EXPECT_EQ(controller.actions(WorkspacePanelKind::Dock),
              QList<QAction *>({&workspaceAction, &logAction}));
    EXPECT_EQ(controller.actions(WorkspacePanelKind::ToolBar),
              QList<QAction *>({&toolBarAction}));
}

TEST(ProjectUiHydratorTest, NewRequestCancelsRemainingStagesFromOlderMetadata)
{
    ProjectUiHydrator hydrator;
    QStringList appliedStages;
    const QJsonObject newerMetadata{{QStringLiteral("revision"), 2}};

    hydrator.setStages({
        [&hydrator, &appliedStages, newerMetadata](const QJsonObject &metadata)
        {
            const int revision = metadata.value(QStringLiteral("revision")).toInt();
            appliedStages.append(QStringLiteral("%1:0").arg(revision));
            if (revision == 1)
            {
                hydrator.schedule(newerMetadata);
            }
        },
        [&appliedStages](const QJsonObject &metadata)
        {
            appliedStages.append(QStringLiteral("%1:1")
                                     .arg(metadata.value(QStringLiteral("revision")).toInt()));
        }
    });

    hydrator.schedule(QJsonObject{{QStringLiteral("revision"), 1}});
    QElapsedTimer timer;
    timer.start();
    while (appliedStages.size() < 3 && timer.elapsed() < 1000)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        QThread::msleep(5);
    }

    EXPECT_EQ(appliedStages,
              QStringList({QStringLiteral("1:0"),
                           QStringLiteral("2:0"),
                           QStringLiteral("2:1")}));
}

TEST(ProjectUiHydratorTest, CoalescesRapidMetadataRefreshesToLatestRevision)
{
    ProjectUiHydrator hydrator;
    QVector<int> appliedRevisions;
    hydrator.setStages({
        [&appliedRevisions](const QJsonObject &metadata)
        {
            appliedRevisions.append(metadata.value(QStringLiteral("revision")).toInt());
        }
    });

    hydrator.schedule(QJsonObject{{QStringLiteral("revision"), 1}});
    hydrator.schedule(QJsonObject{{QStringLiteral("revision"), 2}});
    hydrator.schedule(QJsonObject{{QStringLiteral("revision"), 3}});

    QElapsedTimer timer;
    timer.start();
    while (appliedRevisions.isEmpty() && timer.elapsed() < 1000)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        QThread::msleep(5);
    }

    EXPECT_EQ(appliedRevisions, QVector<int>({3}));
}

TEST(MainMenuTest, ToolbarExposesMetashapeStyleCameraVisibilityButton)
{
    QMainWindow window;
    MainMenu menu(&window);

    QToolBar *toolBar = menu.toolBar();
    ASSERT_NE(toolBar, nullptr);
    auto *cameraButton = toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraVisibility"));
    ASSERT_NE(cameraButton, nullptr);

    EXPECT_EQ(cameraButton->defaultAction(), menu.toggleCamerasAction());
    EXPECT_EQ(cameraButton->popupMode(), QToolButton::MenuButtonPopup);
    EXPECT_EQ(cameraButton->toolTip(), QStringLiteral("显示相机"));
    EXPECT_EQ(cameraButton->iconSize(), QSize(26, 26));
    EXPECT_EQ(cameraButton->size(), QSize(50, 36));
    EXPECT_TRUE(cameraButton->styleSheet().isEmpty());
    ASSERT_NE(cameraButton->menu(), nullptr);
    EXPECT_FALSE(cameraButton->icon().isNull());

    ASSERT_NE(menu.toggleCameraThumbnailsAction(), nullptr);
    ASSERT_NE(menu.toggleDependentCamerasAction(), nullptr);
    ASSERT_NE(menu.toggleLocalAxesAction(), nullptr);
    EXPECT_TRUE(cameraButton->menu()->actions().contains(menu.toggleCameraThumbnailsAction()));
    EXPECT_TRUE(cameraButton->menu()->actions().contains(menu.toggleDependentCamerasAction()));
    EXPECT_TRUE(cameraButton->menu()->actions().contains(menu.toggleLocalAxesAction()));

    EXPECT_EQ(menu.toggleLocalAxesAction()->text(), QStringLiteral("显示本地轴"));
    EXPECT_TRUE(menu.toggleLocalAxesAction()->isCheckable());
    EXPECT_FALSE(menu.toggleLocalAxesAction()->isChecked());
}

TEST(MainMenuTest, ToolbarExposesMetashapeStyleImageVisibilityButton)
{
    QMainWindow window;
    MainMenu menu(&window);

    QToolBar *toolBar = menu.toolBar();
    ASSERT_NE(toolBar, nullptr);
    auto *imageButton =
        toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraImageVisibility"));
    ASSERT_NE(imageButton, nullptr);

    EXPECT_EQ(imageButton->defaultAction(), menu.toggleCameraImagesAction());
    EXPECT_EQ(imageButton->popupMode(), QToolButton::MenuButtonPopup);
    EXPECT_EQ(imageButton->toolTip(), QStringLiteral("显示图像"));
    EXPECT_EQ(imageButton->iconSize(), QSize(26, 26));
    EXPECT_EQ(imageButton->size(), QSize(50, 36));
    EXPECT_TRUE(imageButton->styleSheet().isEmpty());
    ASSERT_NE(imageButton->menu(), nullptr);
    EXPECT_FALSE(imageButton->icon().isNull());

    ASSERT_NE(menu.toggleCameraImagesAction(), nullptr);
    ASSERT_NE(menu.showCameraImagesInForegroundAction(), nullptr);
    ASSERT_NE(menu.showCameraImagesInBackgroundAction(), nullptr);
    ASSERT_NE(menu.lockCameraImageAction(), nullptr);
    EXPECT_FALSE(imageButton->menu()->actions().contains(menu.toggleCameraImagesAction()));
    EXPECT_TRUE(imageButton->menu()->actions().contains(menu.showCameraImagesInForegroundAction()));
    EXPECT_TRUE(imageButton->menu()->actions().contains(menu.showCameraImagesInBackgroundAction()));
    EXPECT_TRUE(imageButton->menu()->actions().contains(menu.lockCameraImageAction()));
}

TEST(MainMenuTest, ToolbarCameraButtonUsesStillCameraIcon)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int iconStart = source.indexOf(QStringLiteral("QIcon makeCameraToolbarIcon()"));
    ASSERT_GE(iconStart, 0);
    const int iconEnd = source.indexOf(QStringLiteral("QIcon makeCameraImageToolbarIcon()"), iconStart);
    ASSERT_GT(iconEnd, iconStart);
    const QString iconBody = source.mid(iconStart, iconEnd - iconStart);

    EXPECT_TRUE(iconBody.contains(QStringLiteral("cameraBodyRect")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("cameraTopRect")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("cameraLensRect")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("QPixmap pixmap(56, 56)")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("QRectF cameraBodyRect(2.0, 17.0, 52.0, 35.0)")));
    EXPECT_FALSE(iconBody.contains(QStringLiteral("QPolygonF lens")));
    EXPECT_FALSE(iconBody.contains(QStringLiteral("drawLine(QPointF")));
}

TEST(MainMenuTest, ToolbarImageButtonUsesPictureIcon)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int iconStart = source.indexOf(QStringLiteral("QIcon makeCameraImageToolbarIcon()"));
    ASSERT_GE(iconStart, 0);
    const int iconEnd = source.indexOf(QStringLiteral("MainMenu::MainMenu"), iconStart);
    ASSERT_GT(iconEnd, iconStart);
    const QString iconBody = source.mid(iconStart, iconEnd - iconStart);

    EXPECT_TRUE(iconBody.contains(QStringLiteral("imageFrameRect")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("imageMountain")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("imageSunRect")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("QPixmap pixmap(56, 56)")));
    EXPECT_TRUE(iconBody.contains(QStringLiteral("QRectF imageFrameRect(3.0, 3.0, 50.0, 50.0)")));
}

TEST(MainMenuTest, ToolbarSplitButtonsPaintIconsAcrossButtonArea)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/ToolbarButton.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("ToolbarSplitButton::paintEvent")))
        << "相机/图像快捷按钮必须自绘图标区域，不能继续依赖 Qt 默认小图标绘制。";
    EXPECT_TRUE(source.contains(QStringLiteral("drawToolbarSplitButtonArrow")));
    EXPECT_TRUE(source.contains(QStringLiteral("painter.drawPixmap(iconTopLeft, pixmap)")));
    EXPECT_TRUE(source.contains(QStringLiteral("new ToolbarSplitButton(toolBar)")));
}

TEST(MainWindowTest, CameraToolbarLocalAxesActionConnectsToModelView)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("toggleLocalAxesAction()")));
    EXPECT_TRUE(source.contains(QStringLiteral("actionToggleLocalAxes")));
    EXPECT_TRUE(source.contains(QStringLiteral("toolButtonModelCameraVisibility")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("toggleLocalAxesAction()")));
    EXPECT_TRUE(mainWindowSource.contains(
        QStringLiteral("&CameraSceneWidget::setShowCameraLocalAxes")));
}

TEST(MainMenuTest, WindowMenuExposesCheckedHenanUniversityBrandAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.toggleHenanUniversityBrandAction();
    ASSERT_NE(action, nullptr);
    EXPECT_EQ(action->text(), QStringLiteral("河南大学校徽"));
    EXPECT_TRUE(action->isCheckable());
    EXPECT_TRUE(action->isChecked());

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    QMenu *windowMenu = findSubMenuByTitle(viewMenu, QStringLiteral("窗口"));
    ASSERT_NE(windowMenu, nullptr);
    EXPECT_TRUE(windowMenu->actions().contains(action));
}

TEST(HenuBrandWidgetTest, BrandIsToolbarMastheadWithEmblem)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/HenuBrandWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/HenuBrandWidget.cpp"));
    const QString guiSources = readProjectSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    const QString qrc = readProjectSourceFile(QStringLiteral("resources/resources.qrc"));
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString workspaceHeader = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.h"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(guiSources.isEmpty());
    ASSERT_FALSE(qrc.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(workspaceHeader.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("class HenuBrandWidget")));
    EXPECT_TRUE(header.contains(QStringLiteral("QPixmap _emblemPixmap")));
    EXPECT_TRUE(header.contains(QStringLiteral("void drawHenuEmblem")));
    EXPECT_TRUE(source.contains(QStringLiteral("_emblemPixmap(QStringLiteral(\":/icons/henu_logo.png\"))")));
    EXPECT_TRUE(source.contains(QStringLiteral("painter.drawPixmap")));
    EXPECT_TRUE(source.contains(QStringLiteral("河大")));
    EXPECT_TRUE(source.contains(QStringLiteral("1912")));
    EXPECT_TRUE(source.contains(QStringLiteral("HENU · PlaScan 三维重建")));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("widgets/HenuBrandWidget.cpp")));
    EXPECT_TRUE(qrc.contains(QStringLiteral("icons/henu_logo.png")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("new QWidgetAction(toolBar)")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("toolBar->insertAction(firstAction, _henuBrandAction)")));
    EXPECT_FALSE(workspaceHeader.contains(QStringLiteral("henuBrandBadge")));
}

TEST(CameraSceneWidgetTest, CameraVisibilityToggleIsExposedAndGuardsCameraOverlayOnly)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("void setShowCameras(bool show)")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool areCamerasVisible() const")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _showCameras = true")));
    EXPECT_TRUE(source.contains(QStringLiteral("void CameraSceneWidget::setShowCameras(bool show)")));
    EXPECT_TRUE(source.contains(QStringLiteral("if (_showCameras)")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("toggleCamerasAction()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("&CameraSceneWidget::setShowCameras")));
}

TEST(CameraSceneWidgetTest, UsesQrhiWidgetWithVulkanBackend)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("#include <QRhiWidget>")));
    EXPECT_TRUE(header.contains(QStringLiteral("class CameraSceneWidget : public QRhiWidget")));
    EXPECT_TRUE(source.contains(QStringLiteral("setApi(QRhiWidget::Api::Vulkan)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void initialize(QRhiCommandBuffer *cb) override;")));
    EXPECT_TRUE(header.contains(QStringLiteral("void render(QRhiCommandBuffer *cb) override;")));
    EXPECT_TRUE(header.contains(QStringLiteral("void releaseResources() override;")));
    EXPECT_TRUE(header.contains(QStringLiteral("RhiPipelineSet _modelPointPipeline;")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawRhiBuffer(cb, &_modelPointBuffer, &_modelPointPipeline, uniforms)")));
    EXPECT_TRUE(source.contains(QStringLiteral("rhi()->clipSpaceCorrMatrix()")));
}

TEST(CameraSceneWidgetTest, QrhiWidgetDoesNotPaintDirectlyWithQPainter)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    auto cameraSceneFunctionBody = [](const QString &text, const QString &signature)
    {
        const int start = text.indexOf(signature);
        if (start < 0)
        {
            return QString();
        }
        int end = text.indexOf(QStringLiteral("\nvoid CameraSceneWidget::"), start + signature.size());
        if (end < 0)
        {
            end = text.size();
        }
        return text.mid(start, end - start);
    };

    const QString renderBody = cameraSceneFunctionBody(
        source,
        QStringLiteral("void CameraSceneWidget::render(QRhiCommandBuffer *cb)"));
    const QString requestOverlayBody = cameraSceneFunctionBody(
        source,
        QStringLiteral("void CameraSceneWidget::requestOverlayUpdate()"));
    const QString paintOverlayBody = cameraSceneFunctionBody(
        source,
        QStringLiteral("void CameraSceneWidget::paintOverlay(QPainter &painter)"));

    ASSERT_FALSE(renderBody.isEmpty());
    ASSERT_FALSE(requestOverlayBody.isEmpty());
    ASSERT_FALSE(paintOverlayBody.isEmpty());
    EXPECT_FALSE(renderBody.contains(QStringLiteral("QPainter painter(this)")));
    EXPECT_TRUE(renderBody.contains(QStringLiteral("requestOverlayUpdate()")));
    EXPECT_FALSE(paintOverlayBody.contains(QStringLiteral("QPainter painter(this)")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraSceneOverlayWidget")));
}

TEST(CameraSceneWidgetTest, RemovesLegacyRenderingDependencies)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString guiCmake = readProjectSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));
    const QString packages = readProjectSourceFile(QStringLiteral("cmake/PlascanPackages.cmake"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(guiCmake.isEmpty());
    ASSERT_FALSE(packages.isEmpty());

    const QString legacyApiName = QStringLiteral("Open") + QStringLiteral("GL");
    const QStringList forbidden = {
        QStringLiteral("Q") + legacyApiName + QStringLiteral("Widget"),
        QStringLiteral("Q") + legacyApiName + QStringLiteral("Functions_4_3_Core"),
        QStringLiteral("Q") + legacyApiName + QStringLiteral("Buffer"),
        QStringLiteral("Q") + legacyApiName + QStringLiteral("ShaderProgram"),
        QStringLiteral("Q") + legacyApiName + QStringLiteral("VertexArrayObject"),
        QStringLiteral("initialize") + QStringLiteral("GL"),
        QStringLiteral("resize") + QStringLiteral("GL"),
        QStringLiteral("paint") + QStringLiteral("GL"),
    };
    for (const QString &token : forbidden)
    {
        EXPECT_FALSE(header.contains(token)) << qPrintable(token);
        EXPECT_FALSE(source.contains(token)) << qPrintable(token);
    }

    EXPECT_FALSE(guiCmake.contains(QStringLiteral("Qt6::") + legacyApiName));
    EXPECT_FALSE(guiCmake.contains(QStringLiteral("Qt6::") + legacyApiName + QStringLiteral("Widgets")));
    EXPECT_FALSE(packages.contains(legacyApiName + QStringLiteral(" ") + legacyApiName + QStringLiteral("Widgets")));
}

TEST(CameraSceneWidgetTest, RegistersQrhiShaderResources)
{
    const QString guiCmake = readProjectSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));
    ASSERT_FALSE(guiCmake.isEmpty());

    EXPECT_TRUE(guiCmake.contains(QStringLiteral("qt_add_shaders(plascan_gui")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("${CMAKE_CURRENT_SOURCE_DIR}/shaders")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_color.vert")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_color.frag")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_mesh.vert")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_mesh.frag")));
}

TEST(CameraSceneWidgetTest, ModelViewDoesNotDrawInvalidWorldOriginLabel)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_FALSE(source.contains(QStringLiteral("XYZ(0,0,0)")));
    EXPECT_FALSE(header.contains(QStringLiteral("setShowWorldOrigin")));
    EXPECT_FALSE(source.contains(QStringLiteral("setShowWorldOrigin")));
    EXPECT_FALSE(mainWindowSource.contains(QStringLiteral("world_origin_visible")));
}

TEST(CameraSceneWidgetTest, CameraOverlayUsesMetashapeStyleImagePlanes)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("float cameraImagePlaneHalfExtent(const CameraPose &pose")));
    EXPECT_TRUE(header.contains(QStringLiteral("void drawFloorPivotCross(QPainter &painter)")));
    EXPECT_TRUE(source.contains(QStringLiteral("cameraImagePlaneHalfExtent(pose, matrices.modelView)")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawCameraDirectionArrow(painter, pose")));
    EXPECT_TRUE(source.contains(QStringLiteral("cameraImagePlaneCorners")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawCameraThumbnails(cb")));
    EXPECT_TRUE(source.contains(QStringLiteral("_thumbnailPipeline.pipeline->setDepthTest(true)")));
    EXPECT_FALSE(source.contains(QStringLiteral("painter.drawPolygon(imagePlane)")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawFloorPivotCross(painter)")));
    EXPECT_TRUE(source.contains(QStringLiteral("floorPivot")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawCameraLabel = highlighted")));
}

TEST(CameraSceneWidgetTest, CameraImagePlanesSupportAsyncThumbnailsAndImageMode)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("enum class CameraImagePlaneMode")));
    EXPECT_TRUE(header.contains(QStringLiteral("enum class CameraImageDisplayLayer")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setShowCameraThumbnails(bool show)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setShowCameraImage(bool show)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setCameraImagePlaneMode(CameraImagePlaneMode mode)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setCameraImageDisplayLayer(CameraImageDisplayLayer layer)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setCameraImageLocked(bool locked)")));
    EXPECT_TRUE(header.contains(QStringLiteral("CameraImagePlaneMode cameraImagePlaneMode() const")));
    EXPECT_TRUE(header.contains(QStringLiteral("void updateCameraOverlay()")));
    EXPECT_TRUE(header.contains(QStringLiteral("QHash<QString, QImage> _cameraImageCache")));
    EXPECT_TRUE(header.contains(QStringLiteral("QSet<QString> _cameraImageLoadsInFlight")));
    EXPECT_TRUE(header.contains(QStringLiteral("static CameraPlaneImageResult loadCameraPlaneImage")));

    EXPECT_TRUE(source.contains(QStringLiteral("QtConcurrent::run(&CameraSceneWidget::loadCameraPlaneImage")));
    EXPECT_TRUE(source.contains(QStringLiteral("xjw::gui::views::loadImageForDisplay")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraImagePlaneMode::Thumbnail")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraImagePlaneMode::Image")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraImageDisplayLayer::Foreground")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraImageDisplayLayer::Background")));
    EXPECT_TRUE(header.contains(QStringLiteral("struct RhiImagePipelineSet")));
    EXPECT_TRUE(source.contains(QStringLiteral("bool CameraSceneWidget::ensureImagePipeline")));
    EXPECT_TRUE(source.contains(QStringLiteral("void CameraSceneWidget::drawActiveCameraImage")));
    EXPECT_TRUE(source.contains(QStringLiteral("cameraImagePlaneCorners")));
    EXPECT_TRUE(source.contains(QStringLiteral("void CameraSceneWidget::updateCameraOverlay()")));
    EXPECT_TRUE(source.contains(QStringLiteral("updateCameraOverlay();")));

    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("toggleCameraThumbnailsAction()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("toggleCameraImagesAction()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("showCameraImagesInForegroundAction()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("showCameraImagesInBackgroundAction()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("lockCameraImageAction()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("&CameraSceneWidget::setShowCameraThumbnails")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("&CameraSceneWidget::setShowCameraImage")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("CameraSceneWidget::CameraImageDisplayLayer::Foreground")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("CameraSceneWidget::CameraImageDisplayLayer::Background")));
}

TEST(MainWindowTest, ReferenceDatasetActionsConnectToProjectManager)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString mainWindow = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(mainWindow.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("bindActions(MainMenu")));
    EXPECT_TRUE(mainWindow.contains(QStringLiteral("_menuWorkflowController->bindActions(_mainMenu)")));

    EXPECT_TRUE(source.contains(QStringLiteral("importReferenceDatasetAction()")));
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::importReferenceDataset")));
    EXPECT_TRUE(source.contains(QStringLiteral("referenceQualityCheckAction()")));
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::runReferenceQualityCheck")));
    EXPECT_TRUE(source.contains(QStringLiteral("referenceTerrainBundleAdjustAction()")));
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::prepareReferenceTerrainBundleAdjust")));
    EXPECT_TRUE(source.contains(QStringLiteral("surveyControlAction()")));
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::openSurveyControlDialog")));

    EXPECT_FALSE(mainWindow.contains(QStringLiteral("&ProjectManager::importReferenceDataset")));
    EXPECT_FALSE(mainWindow.contains(QStringLiteral("&ProjectManager::runReferenceQualityCheck")));
    EXPECT_FALSE(mainWindow.contains(QStringLiteral("&ProjectManager::prepareReferenceTerrainBundleAdjust")));
    EXPECT_FALSE(mainWindow.contains(QStringLiteral("&ProjectManager::openSurveyControlDialog")));
}

TEST(MainWindowTest, PhotoStripClickSelectsCameraWithoutOpeningImageView)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int selectedConnection = source.indexOf(QStringLiteral("&PhotoStripWidget::photoSelected"));
    const int activatedConnection = source.indexOf(QStringLiteral("&PhotoStripWidget::photoActivated"));
    ASSERT_GE(selectedConnection, 0);
    ASSERT_GT(activatedConnection, selectedConnection);

    const QString selectedBlock = source.mid(selectedConnection, activatedConnection - selectedConnection);
    EXPECT_TRUE(selectedBlock.contains(QStringLiteral("selectPhoto(path, false)")));

    const int canvasConnection = source.indexOf(QStringLiteral("if (_canvas)"), activatedConnection);
    ASSERT_GT(canvasConnection, activatedConnection);
    const QString activatedBlock = source.mid(activatedConnection, canvasConnection - activatedConnection);
    EXPECT_TRUE(activatedBlock.contains(QStringLiteral("selectPhoto(path, true)")));

    const int selectPhotoStart = source.indexOf(QStringLiteral("void MainWindow::selectPhoto"));
    const int selectResourceStart = source.indexOf(QStringLiteral("void MainWindow::selectResource"), selectPhotoStart);
    ASSERT_GE(selectPhotoStart, 0);
    ASSERT_GT(selectResourceStart, selectPhotoStart);
    const QString selectPhotoBody = source.mid(selectPhotoStart, selectResourceStart - selectPhotoStart);
    EXPECT_TRUE(selectPhotoBody.contains(QStringLiteral("highlightCameraForImage(imagePath)")));
    EXPECT_TRUE(selectPhotoBody.contains(QStringLiteral("if (openImage)")));
    EXPECT_TRUE(selectPhotoBody.contains(QStringLiteral("showImageView(imagePath)")));
}

TEST(MainWindowTest, PhotoStripMaskRequestUsesSelectedImages)
{
    const QString mainSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString managerHeader = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString managerSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(mainSource.isEmpty());
    ASSERT_FALSE(managerHeader.isEmpty());
    ASSERT_FALSE(managerSource.isEmpty());

    EXPECT_TRUE(mainSource.contains(QStringLiteral("&PhotoStripWidget::generateMaskRequested")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("openGenerateMaskDialogForImages(imagePaths)")));
    EXPECT_TRUE(managerHeader.contains(
        QStringLiteral("void openGenerateMaskDialogForImages(const QStringList &selectedImages);")));
    EXPECT_TRUE(managerSource.contains(
        QStringLiteral("GenerateMaskDialog dialog(selectedImages, currentImage, _parent)")));
    EXPECT_TRUE(managerSource.contains(
        QStringLiteral("xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, imagePath)")));
    EXPECT_TRUE(managerSource.contains(
        QStringLiteral("maskTargetsFromSettings(settings, resolvedAllImages)")));
}

TEST(MainWindowTest, SelectionAndPhotoPanelsUseMovableDockWidgets)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.ui"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(ui.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QDockWidget *_workspaceDock{}")));
    EXPECT_TRUE(header.contains(QStringLiteral("QDockWidget *_propertiesDock{}")));
    EXPECT_TRUE(header.contains(QStringLiteral("QDockWidget *_photosDock{}")));
    EXPECT_TRUE(header.contains(QStringLiteral("QDockWidget*      _logDock{}")));

    EXPECT_TRUE(source.contains(QStringLiteral("void configureMovableDock(QDockWidget *dock)")));
    EXPECT_TRUE(source.contains(QStringLiteral("dock->setAllowedAreas(Qt::AllDockWidgetAreas)")));
    EXPECT_TRUE(source.contains(QStringLiteral("QDockWidget::DockWidgetMovable")));
    EXPECT_TRUE(source.contains(QStringLiteral("QDockWidget::DockWidgetFloatable")));
    EXPECT_TRUE(source.contains(QStringLiteral("dock->setMinimumSize(DockMinWidth, DockMinHeight)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_workspaceCenter->setMinimumSize(240, 160)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_workspaceDock = new QDockWidget(tr(\"工作区\"), this)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_propertiesDock = new QDockWidget(tr(\"资源属性\"), this)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_photosDock = new QDockWidget(tr(\"照片\"), this)")));
    EXPECT_TRUE(source.contains(QStringLiteral("configureMovableDock(_logDock)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_logDock->setTitleBarWidget(nullptr)")));
    EXPECT_TRUE(source.contains(QStringLiteral("splitDockWidget(_workspaceDock, _propertiesDock, Qt::Vertical)")));
    EXPECT_TRUE(source.contains(QStringLiteral("addDockWidget(Qt::BottomDockWidgetArea, _photosDock)")));

    EXPECT_TRUE(source.contains(
        QStringLiteral("_workspacePanels->registerDock(WorkspacePanelId::Workspace")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("_workspacePanels->registerDock(WorkspacePanelId::Properties")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("_workspacePanels->registerDock(WorkspacePanelId::Photos")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("_workspacePanels->registerDock(WorkspacePanelId::Log")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("_workspacePanels->registerToolBar(WorkspacePanelId::MainToolbar")));
    EXPECT_FALSE(source.contains(QStringLiteral("&MainWindow::onToggleLogAction")));
    EXPECT_FALSE(header.contains(QStringLiteral("featureInfoAction()")));
    EXPECT_FALSE(header.contains(QStringLiteral("_featureInfoAct")));
    EXPECT_FALSE(source.contains(QStringLiteral("actionFeatureInfo")));
    EXPECT_FALSE(source.contains(QStringLiteral("_featureInfoAct")));
    EXPECT_FALSE(ui.contains(QStringLiteral("actionFeatureInfo")));
    EXPECT_FALSE(ui.contains(QStringLiteral("兴趣点信息")));
    EXPECT_FALSE(source.contains(QStringLiteral("photosFrame->setMaximumHeight(320)")));
    EXPECT_FALSE(source.contains(QStringLiteral("_rightPanelSplitter")));
    EXPECT_FALSE(source.contains(QStringLiteral("_logDock->setTitleBarWidget(titleBar)")));
    EXPECT_TRUE(ui.contains(QStringLiteral("<set>Qt::AllDockWidgetAreas</set>")));
    EXPECT_FALSE(ui.contains(QStringLiteral("<set>Qt::BottomDockWidgetArea</set>")));
}

TEST(DialogSettingStoreTest, SuccessfulSaveNotifiesProjectWorkspace)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    DialogSettingStore store(QStringLiteral("generate_model"));
    store.setProjectPath(
        QDir(tempDir.path()).filePath(QStringLiteral("callback.plascan")));
    int notificationCount = 0;
    store.setChangeCallback([&notificationCount]()
    {
        ++notificationCount;
    });

    QString error;
    ASSERT_TRUE(store.save(
        QJsonObject{{QStringLiteral("quality"), QStringLiteral("high")}},
        &error)) << qPrintable(error);
    EXPECT_EQ(notificationCount, 1);
}

TEST(MainWindowTest, ProjectOpenRestoresAndPersistsDockPanelState)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject currentUiSettingsSnapshot() const")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool restoreProjectDockState(const QJsonObject &settings)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void restoreDefaultProjectDockLayout();")));
    EXPECT_TRUE(header.contains(QStringLiteral("void persistCurrentUiSettings()")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _applyingUiSettings{}")));
    EXPECT_TRUE(source.contains(QStringLiteral("_mainMenu->setManagedWindowActions(")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("_workspacePanels->actions(WorkspacePanelKind::Dock)")));
    EXPECT_FALSE(source.contains(
        QStringLiteral("connect(_mainMenu->openAction(), &QAction::triggered, _projectManager, &ProjectManager::openProject)")));
    EXPECT_TRUE(source.contains(QStringLiteral("persistCurrentUiSettings();\n                _projectManager->openProject();")));
    EXPECT_TRUE(source.contains(QStringLiteral("openProjectFromPath(p);")));
    EXPECT_FALSE(source.contains(
        QStringLiteral("connect(_projectManager, &ProjectManager::projectCreated, this, &MainWindow::onProjectOpened)")));
    EXPECT_TRUE(source.contains(
        QStringLiteral("connect(_projectManager, &ProjectManager::projectClosed, this, &MainWindow::onProjectClosed)")));

    const int projectOpenStart = source.indexOf(QStringLiteral("void MainWindow::onProjectOpened"));
    const int applyStart = source.indexOf(QStringLiteral("void MainWindow::applyUiSettings"), projectOpenStart);
    ASSERT_GE(projectOpenStart, 0);
    ASSERT_GT(applyStart, projectOpenStart);
    const QString projectOpenBlock = source.mid(projectOpenStart, applyStart - projectOpenStart);
    const int firstPersist = projectOpenBlock.indexOf(QStringLiteral("persistCurrentUiSettings();"));
    ASSERT_GE(firstPersist, 0);
    EXPECT_TRUE(projectOpenBlock.contains(
        QStringLiteral("_projectManager->loadUiSettings()")));
    EXPECT_FALSE(projectOpenBlock.contains(
        QStringLiteral("_uiSetting->setProjectPath")));
    EXPECT_TRUE(projectOpenBlock.contains(QStringLiteral("persistCurrentUiSettings();\n}")));

    const int closeStart = source.indexOf(QStringLiteral("void MainWindow::closeEvent"), applyStart);
    ASSERT_GT(closeStart, applyStart);
    const QString applyBlock = source.mid(applyStart, closeStart - applyStart);
    EXPECT_TRUE(applyBlock.contains(QStringLiteral("const QJsonObject settings = ui;")));
    EXPECT_TRUE(applyBlock.contains(
        QStringLiteral("const bool restoredDockState = restoreProjectDockState(settings);")));
    EXPECT_TRUE(applyBlock.contains(QStringLiteral("_workspacePanels->syncActions();")));
    EXPECT_TRUE(applyBlock.contains(QStringLiteral("_workspacePanels->applyVisibility(settings);")));
    EXPECT_TRUE(applyBlock.contains(QStringLiteral("QScopedValueRollback<bool> applyingRollback")));
    EXPECT_FALSE(applyBlock.contains(QStringLiteral("if (ui.isEmpty())\n    {\n        return;\n    }")));
    EXPECT_FALSE(applyBlock.contains(QStringLiteral("ensurePanelVisibilityDefaults")));

    const int snapshotStart = source.indexOf(QStringLiteral("QJsonObject MainWindow::currentUiSettingsSnapshot"));
    const int saveStart = source.indexOf(QStringLiteral("void MainWindow::saveUiSetting"), snapshotStart);
    ASSERT_GE(snapshotStart, 0);
    ASSERT_GT(saveStart, snapshotStart);
    const QString snapshotBlock = source.mid(snapshotStart, saveStart - snapshotStart);
    EXPECT_TRUE(snapshotBlock.contains(QStringLiteral("_workspacePanels->visibilitySnapshot()")));
    EXPECT_TRUE(snapshotBlock.contains(QStringLiteral("QStringLiteral(\"bottom_panel\")")));
    EXPECT_TRUE(snapshotBlock.contains(QStringLiteral("QStringLiteral(\"dock_layout_version\")")));
    EXPECT_TRUE(snapshotBlock.contains(QStringLiteral("QStringLiteral(\"dock_state\")")));
    EXPECT_TRUE(snapshotBlock.contains(QStringLiteral("saveState().toBase64()")));

    const QString closeBlock = source.mid(closeStart);
    EXPECT_TRUE(closeBlock.contains(QStringLiteral("persistCurrentUiSettings();")));
    EXPECT_TRUE(closeBlock.contains(QStringLiteral("const bool hasProject")));
    EXPECT_TRUE(closeBlock.contains(QStringLiteral("if (hasProject)")));
    EXPECT_TRUE(closeBlock.contains(QStringLiteral("if (hasProject && _projectManager->isDirty())")));
    const int cancelIndex = closeBlock.indexOf(QStringLiteral("QMessageBox::Cancel"));
    const int persistIndex = closeBlock.indexOf(QStringLiteral("persistCurrentUiSettings();"));
    ASSERT_GE(cancelIndex, 0);
    ASSERT_LT(persistIndex, cancelIndex);
}

TEST(MainWindowTest, DefaultDockLayoutKeepsPropertiesAndPhotosSideBySide)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int layoutStart = source.indexOf(QStringLiteral("void MainWindow::restoreDefaultProjectDockLayout"));
    ASSERT_GE(layoutStart, 0);
    const int restoreStart = source.indexOf(QStringLiteral("bool MainWindow::restoreProjectDockState"), layoutStart);
    ASSERT_GT(restoreStart, layoutStart);
    const QString layoutBlock = source.mid(layoutStart, restoreStart - layoutStart);

    const int leftWorkspaceIndex = layoutBlock.indexOf(
        QStringLiteral("addDockWidget(Qt::LeftDockWidgetArea, _workspaceDock)"));
    const int splitPropertiesIndex = layoutBlock.indexOf(
        QStringLiteral("splitDockWidget(_workspaceDock, _propertiesDock, Qt::Vertical)"));
    const int bottomPhotosIndex = layoutBlock.indexOf(
        QStringLiteral("addDockWidget(Qt::BottomDockWidgetArea, _photosDock)"));
    ASSERT_GE(leftWorkspaceIndex, 0);
    ASSERT_GE(splitPropertiesIndex, 0);
    ASSERT_GE(bottomPhotosIndex, 0);
    EXPECT_LT(leftWorkspaceIndex, splitPropertiesIndex)
        << "The properties dock should be split under the workspace before creating the bottom photo dock.";
    EXPECT_LT(splitPropertiesIndex, bottomPhotosIndex)
        << "Adding photos after the left dock stack keeps photos beside properties instead of underneath them.";

    EXPECT_TRUE(layoutBlock.contains(QStringLiteral("resizeDocks({_workspaceDock}, {320}, Qt::Horizontal)")));
    EXPECT_TRUE(layoutBlock.contains(
        QStringLiteral("_workspacePanels->ensureRequiredProjectPanelsVisible();")));
    EXPECT_TRUE(layoutBlock.contains(
        QStringLiteral("resizeDocks({_workspaceDock, _propertiesDock}, {560, 190}, Qt::Vertical)")));
    EXPECT_TRUE(layoutBlock.contains(QStringLiteral("resizeDocks({_photosDock}, {120}, Qt::Vertical)")));
}

TEST(MainWindowTest, UsesOneGenerationSafeProjectUiHydrator)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString hydratorHeader = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/ProjectUiHydrator.h"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_FALSE(hydratorHeader.isEmpty());
    EXPECT_TRUE(header.contains(QStringLiteral("class ProjectUiHydrator;")));
    EXPECT_TRUE(header.contains(QStringLiteral("ProjectUiHydrator *_projectUiHydrator{};")));
    EXPECT_FALSE(header.contains(QStringLiteral("scheduleProjectUiHydration")));
    EXPECT_FALSE(source.contains(QStringLiteral("void MainWindow::scheduleProjectUiHydration")));
    EXPECT_FALSE(source.contains(QStringLiteral("_metadataRefreshQueued")));
    EXPECT_FALSE(source.contains(QStringLiteral("_pendingMetadataRefresh")));
    EXPECT_TRUE(source.contains(QStringLiteral("_projectUiHydrator->schedule(meta)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_projectUiHydrator->cancel()")));
}

TEST(MainWindowTest, KeepsOwnedWidgetsAndManagersPrivate)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    ASSERT_FALSE(header.isEmpty());

    const int canvasGetter = header.indexOf(QStringLiteral("CanvasWidget* canvas() const"));
    const int brandMember = header.indexOf(QStringLiteral("HenuBrandWidget*  _henuBrandWidget"));
    const int privateAfterGetter = header.indexOf(QStringLiteral("private:"), canvasGetter);
    ASSERT_GE(canvasGetter, 0);
    ASSERT_GT(brandMember, canvasGetter);
    ASSERT_GT(privateAfterGetter, canvasGetter);
    EXPECT_LT(privateAfterGetter, brandMember)
        << "MainWindow 只应公开稳定访问器，不应公开可变 Widget/管理器指针。";
}

TEST(MainWindowTest, DesignerViewMenuDoesNotDuplicateRuntimeOrdering)
{
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.ui"));
    ASSERT_FALSE(ui.isEmpty());

    const int viewStart = ui.indexOf(QStringLiteral("<widget class=\"QMenu\" name=\"menuView\">"));
    const int workflowStart = ui.indexOf(
        QStringLiteral("<widget class=\"QMenu\" name=\"menuWorkflow\">"), viewStart);
    ASSERT_GE(viewStart, 0);
    ASSERT_GT(workflowStart, viewStart);
    const QString viewSection = ui.mid(viewStart, workflowStart - viewStart);

    EXPECT_FALSE(viewSection.contains(QStringLiteral("<addaction name=\"actionZoomIn\"/>")));
    EXPECT_FALSE(viewSection.contains(QStringLiteral("<addaction name=\"actionToggleGizmo\"/>")));
    EXPECT_FALSE(viewSection.contains(QStringLiteral("<addaction name=\"actionFeatureVisualization\"/>")));
    EXPECT_FALSE(viewSection.contains(QStringLiteral("<addaction name=\"menuWindow\"/>")));
}

TEST(MainWindowTest, OldDockLayoutStateMigratesToDefaultSideBySideLayout)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("constexpr int ProjectDockLayoutVersion = 2")));

    const int restoreStart = source.indexOf(QStringLiteral("bool MainWindow::restoreProjectDockState"));
    const int ensureStart = source.indexOf(QStringLiteral("void MainWindow::persistCurrentUiSettings"), restoreStart);
    ASSERT_GE(restoreStart, 0);
    ASSERT_GT(ensureStart, restoreStart);
    const QString restoreBlock = source.mid(restoreStart, ensureStart - restoreStart);

    EXPECT_TRUE(restoreBlock.contains(QStringLiteral("ProjectDockLayoutVersion")))
        << "Saved dock states need an explicit layout version so old project states can be migrated.";
    EXPECT_TRUE(restoreBlock.contains(QStringLiteral("restoreDefaultProjectDockLayout();")))
        << "Projects without the current layout version should open with the default side-by-side dock layout.";
    EXPECT_TRUE(restoreBlock.contains(QStringLiteral("!restoreState(state)")))
        << "Corrupt or incompatible dock state must fall back to the default side-by-side layout.";

    const int versionCheck = restoreBlock.indexOf(QStringLiteral("ProjectDockLayoutVersion"));
    const int restoreState = restoreBlock.indexOf(QStringLiteral("restoreState(state)"));
    ASSERT_GE(versionCheck, 0);
    ASSERT_GE(restoreState, 0);
    EXPECT_LT(versionCheck, restoreState)
        << "Old dock states should be rejected before QMainWindow restores their geometry.";
}

TEST(MainWindowTest, ProjectOpenUsesDefaultsOnlyWhenVisibilityWasNotSaved)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_FALSE(source.contains(QStringLiteral("void enforceRequiredPanelVisibility(QJsonObject &settings)")));
    EXPECT_FALSE(header.contains(QStringLiteral("void ensureRequiredProjectDocksVisible();")));
    EXPECT_FALSE(source.contains(QStringLiteral("ensureRequiredProjectDocksVisible();")));
    EXPECT_TRUE(header.contains(QStringLiteral("WorkspacePanelController *_workspacePanels{};")));

    const int applyStart = source.indexOf(QStringLiteral("void MainWindow::applyUiSettings"));
    const int closeStart = source.indexOf(QStringLiteral("void MainWindow::closeEvent"), applyStart);
    ASSERT_GE(applyStart, 0);
    ASSERT_GT(closeStart, applyStart);
    const QString applyBlock = source.mid(applyStart, closeStart - applyStart);
    const int restoreIndex = applyBlock.indexOf(
        QStringLiteral("const bool restoredDockState = restoreProjectDockState(settings);"));
    const int syncActionsIndex = applyBlock.indexOf(QStringLiteral("_workspacePanels->syncActions();"));
    const int applyVisibilityIndex = applyBlock.indexOf(
        QStringLiteral("_workspacePanels->applyVisibility(settings);"));
    ASSERT_GE(restoreIndex, 0);
    ASSERT_GE(syncActionsIndex, 0);
    ASSERT_GE(applyVisibilityIndex, 0);
    EXPECT_LT(restoreIndex, syncActionsIndex);
    EXPECT_LT(syncActionsIndex, applyVisibilityIndex);
    EXPECT_TRUE(applyBlock.contains(QStringLiteral("if (restoredDockState)")));
    EXPECT_TRUE(applyBlock.contains(QStringLiteral("else")));
}

TEST(ProjectOpenResponsivenessTest, ProjectManagerLoadsProjectSnapshotOffGuiThread)
{
    const QString managerHeader = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString managerSource = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString commandsHeader = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectUiCommands.h"));
    const QString commandsSource = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectUiCommands.cpp"));
    ASSERT_FALSE(managerHeader.isEmpty());
    ASSERT_FALSE(managerSource.isEmpty());
    ASSERT_FALSE(commandsHeader.isEmpty());
    ASSERT_FALSE(commandsSource.isEmpty());

    EXPECT_TRUE(managerHeader.contains(QStringLiteral("void projectOpenStarted(const QString &plascanPath);")));
    EXPECT_TRUE(managerHeader.contains(QStringLiteral("void projectOpenProgressChanged(const QString &message, int percent);")));
    EXPECT_TRUE(managerHeader.contains(QStringLiteral("void projectOpenFinished(bool success, const QString &message);")));
    EXPECT_TRUE(managerHeader.contains(QStringLiteral("void loadProjectResultsAsync(const QString &plascanPath);")));
    EXPECT_TRUE(managerHeader.contains(QStringLiteral("bool _projectOpenInProgress")));
    EXPECT_TRUE(commandsHeader.contains(QStringLiteral("bool selectProjectByDialog(QString *selectedPath) const;")));

    const int openStart = managerSource.indexOf(QStringLiteral("void ProjectManager::openProjectFromPath"));
    const int saveStart = managerSource.indexOf(QStringLiteral("void ProjectManager::saveProject"), openStart);
    ASSERT_GE(openStart, 0);
    ASSERT_GT(saveStart, openStart);
    const QString openBlock = managerSource.mid(openStart, saveStart - openStart);

    EXPECT_TRUE(openBlock.contains(QStringLiteral("emit projectOpenStarted(projectPath);")));
    EXPECT_TRUE(openBlock.contains(QStringLiteral("xjw::gui::tasks::runGuarded(")));
    EXPECT_TRUE(openBlock.contains(QStringLiteral("ProjectData::loadProjectOpenSnapshot(projectPath)")))
        << "Archive IO and JSON parsing should happen in the worker, not inside ProjectData on the GUI thread.";
    EXPECT_TRUE(openBlock.contains(QStringLiteral("openProjectFromSnapshot(snapshot")))
        << "The GUI thread should only apply the already-loaded snapshot.";
    EXPECT_TRUE(openBlock.contains(QStringLiteral("loadProjectResultsAsync(projectPath);")))
        << "Heavy result metadata should be loaded after the core project opens.";
    EXPECT_TRUE(openBlock.contains(QStringLiteral("if (!snapshot.resultsLoaded)")))
        << "Core metadata must not be cleared by a later empty async result load.";
    EXPECT_FALSE(openBlock.contains(QStringLiteral("_uiCommands->openProjectFromPath(plascanPath)")))
        << "The old synchronous UI command path blocks the GUI while the archive is read.";

    EXPECT_TRUE(commandsSource.contains(QStringLiteral("bool ProjectUiCommands::selectProjectByDialog")));
    EXPECT_TRUE(commandsSource.contains(QStringLiteral("selectProjectByDialog(&plascanPath)")));
}

TEST(ProjectOpenResponsivenessTest, ProjectManagerScansImageFoldersOffGuiThread)
{
    const QString managerSource = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString commandsHeader = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectUiCommands.h"));
    const QString commandsSource = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectUiCommands.cpp"));
    ASSERT_FALSE(managerSource.isEmpty());
    ASSERT_FALSE(commandsHeader.isEmpty());
    ASSERT_FALSE(commandsSource.isEmpty());

    EXPECT_TRUE(commandsHeader.contains(QStringLiteral("bool selectImageFolder(QString *selectedFolder) const;")))
        << "Folder selection should be separated from the potentially slow folder scan.";
    EXPECT_TRUE(commandsSource.contains(QStringLiteral("bool ProjectUiCommands::selectImageFolder")))
        << "The UI command layer should keep the directory dialog logic reusable.";

    const int addStart = managerSource.indexOf(QStringLiteral("void ProjectManager::addFolder"));
    const int nextStart = managerSource.indexOf(QStringLiteral("bool ProjectManager::importCameraForImage"), addStart);
    ASSERT_GE(addStart, 0);
    ASSERT_GT(nextStart, addStart);
    const QString addBlock = managerSource.mid(addStart, nextStart - addStart);

    EXPECT_TRUE(addBlock.contains(QStringLiteral("selectImageFolder(&folder)")));
    EXPECT_TRUE(addBlock.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "Directory scanning can touch slow disks or large folders and must not run on the GUI thread.";
    EXPECT_TRUE(addBlock.contains(QStringLiteral("scanImageFolder(folder)")));
    EXPECT_TRUE(addBlock.contains(QStringLiteral("addImages(scan.imagePaths")))
        << "Only the final ProjectData mutation should run back on the GUI thread.";
    EXPECT_FALSE(addBlock.contains(QStringLiteral("_uiCommands->addFolder()")))
        << "The old synchronous path scans and updates metadata inside the action handler.";
}

TEST(ProjectOpenResponsivenessTest, MainWindowShowsProgressAndAvoidsFullMetaDuringOpen)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QProgressDialog*  _openProgressDialog{}")));
    EXPECT_TRUE(header.contains(QStringLiteral("void onProjectOpenStarted(const QString &plascanPath);")));
    EXPECT_TRUE(header.contains(QStringLiteral("void onProjectOpenProgressChanged(const QString &message, int percent);")));
    EXPECT_TRUE(header.contains(QStringLiteral("void onProjectOpenFinished(bool success, const QString &message);")));

    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::projectOpenStarted")));
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::projectOpenProgressChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("&ProjectManager::projectOpenFinished")));
    EXPECT_TRUE(source.contains(QStringLiteral("new QProgressDialog(tr(\"正在打开项目...\"), QString(), 0, 100, this)")));

    const int openSlotStart = source.indexOf(QStringLiteral("void MainWindow::onProjectOpened"));
    const int closedSlotStart = source.indexOf(QStringLiteral("void MainWindow::onProjectClosed"), openSlotStart);
    ASSERT_GE(openSlotStart, 0);
    ASSERT_GT(closedSlotStart, openSlotStart);
    const QString openSlot = source.mid(openSlotStart, closedSlotStart - openSlotStart);
    EXPECT_TRUE(openSlot.contains(
        QStringLiteral("scheduleProjectMetadataRefresh(_projectManager->coreProjectMeta())")));
    EXPECT_TRUE(openSlot.contains(QStringLiteral("coreProjectMeta()")));
    EXPECT_FALSE(openSlot.contains(QStringLiteral("_projectManager->currentMeta()")))
        << "Opening the first viewport must not synchronously trigger project_results.json loading.";

    const int projectOpenedLambda = source.indexOf(
        QStringLiteral("connect(_projectManager, &ProjectManager::projectOpened, this, [this]"));
    EXPECT_LT(projectOpenedLambda, 0)
        << "The extra projectOpened lambda duplicated refresh work and called currentMeta() during open.";
}

TEST(ProjectOpenResponsivenessTest, MainWindowDefersMetadataWidgetRefresh)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("scheduleProjectMetadataRefresh")))
        << "Metadata refresh should be a named queued helper, not direct widget slots.";

    const int setupStart = source.indexOf(QStringLiteral("void MainWindow::setupProjectManager"));
    const int setupEnd = source.indexOf(QStringLiteral("void MainWindow::refreshDashboardTaskSnapshots"), setupStart);
    ASSERT_GE(setupStart, 0);
    ASSERT_GT(setupEnd, setupStart);
    const QString setupBlock = source.mid(setupStart, setupEnd - setupStart);

    EXPECT_TRUE(setupBlock.contains(QStringLiteral("scheduleProjectMetadataRefresh(meta)")));
    EXPECT_FALSE(setupBlock.contains(
        QStringLiteral("connect(_projectManager, &ProjectManager::projectMetadataChanged, _dashboard")));
    EXPECT_FALSE(setupBlock.contains(
        QStringLiteral("connect(_projectManager, &ProjectManager::projectMetadataChanged, _dataTree")));
    EXPECT_TRUE(setupBlock.contains(QStringLiteral("_projectUiHydrator->setStages")));
    EXPECT_TRUE(setupBlock.contains(QStringLiteral("_workspaceCenter->setProjectMeta(meta)")))
        << "Model view refresh should be one of the hydrator stages.";
    EXPECT_TRUE(setupBlock.contains(QStringLiteral("_photoStrip->loadFromJson(meta)")));

    const int refreshStart = source.indexOf(QStringLiteral("void MainWindow::scheduleProjectMetadataRefresh"));
    ASSERT_GE(refreshStart, 0);
    const int refreshEnd = source.indexOf(QStringLiteral("void MainWindow::onProjectClosed"), refreshStart);
    ASSERT_GT(refreshEnd, refreshStart);
    const QString refreshBlock = source.mid(refreshStart, refreshEnd - refreshStart);

    EXPECT_TRUE(refreshBlock.contains(QStringLiteral("_projectUiHydrator->schedule(meta)")));
    EXPECT_FALSE(refreshBlock.contains(QStringLiteral("QTimer::singleShot")));
    EXPECT_FALSE(refreshBlock.contains(QStringLiteral("_dashboard->loadFromJson(meta)")));
    EXPECT_FALSE(refreshBlock.contains(QStringLiteral("_dataTree->loadFromJson(meta)")));
}

TEST(MainWindowMenuWiringTest, CameraConversionActionIsConnectedToWorkflowController)
{
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString controllerHeader =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(controllerHeader.isEmpty());
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("_menuWorkflowController->bindActions(_mainMenu)")));
    EXPECT_TRUE(controllerHeader.contains(QStringLiteral("bindActions(MainMenu")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("cameraConvertAction")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("openCameraConvertDialog")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("CameraConvertDialog")));
}

TEST(CodeStyleTest, MainWindowUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("Ui::MainWindow*  _ui{};"),
        QStringLiteral("QSplitter*        _mainSplitter{};"),
        QStringLiteral("QTabWidget*       _leftTabs{};"),
        QStringLiteral("ProjectDashboardWidget* _dashboard{};"),
        QStringLiteral("DataTreeWidget*   _dataTree{};"),
        QStringLiteral("ReferencePanelWidget* _referencePanel{};"),
        QStringLiteral("WorkspaceCenterWidget* _workspaceCenter{};"),
        QStringLiteral("CanvasWidget*     _canvas{};"),
        QStringLiteral("HenuBrandWidget*  _henuBrandWidget{};"),
        QStringLiteral("QWidgetAction*    _henuBrandAction{};"),
        QStringLiteral("LogPanel*         _log{};"),
        QStringLiteral("MainMenu*         _mainMenu{};"),
        QStringLiteral("AppConfigManager* _config{};"),
        QStringLiteral("ProjectData*      _projectData{};"),
        QStringLiteral("MenuWorkflowController* _menuWorkflowController{};"),
        QStringLiteral("ReconstructionWorkflowController* _reconController{};"),
        QStringLiteral("ProjectManager*   _projectManager{};"),
        QStringLiteral("QProgressDialog*  _saveProgressDialog{};"),
        QStringLiteral("TaskStatusWidget* _mvsTaskStatus{};"),
        QStringLiteral("TaskStatusWidget* _meshTaskStatus{};"),
        QStringLiteral("TaskStatusWidget* _atTaskStatus{};"),
        QStringLiteral("TaskStatusWidget* _sgTaskStatus{};"),
        QStringLiteral("TaskStatusWidget* _maskTaskStatus{};"),
        QStringLiteral("QDockWidget*      _logDock{};"),
        QStringLiteral("QString           _lastSelectedImage;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_ui"),
        QStringLiteral("m_mainSplitter"),
        QStringLiteral("m_leftTabs"),
        QStringLiteral("m_dashboard"),
        QStringLiteral("m_dataTree"),
        QStringLiteral("m_referencePanel"),
        QStringLiteral("m_workspaceCenter"),
        QStringLiteral("m_canvas"),
        QStringLiteral("m_henuBrandWidget"),
        QStringLiteral("m_henuBrandAction"),
        QStringLiteral("m_log"),
        QStringLiteral("m_mainMenu"),
        QStringLiteral("m_config"),
        QStringLiteral("m_projectData"),
        QStringLiteral("m_menuWorkflowController"),
        QStringLiteral("m_reconController"),
        QStringLiteral("m_projectManager"),
        QStringLiteral("m_saveProgressDialog"),
        QStringLiteral("m_mvsTaskStatus"),
        QStringLiteral("m_meshTaskStatus"),
        QStringLiteral("m_atTaskStatus"),
        QStringLiteral("m_sgTaskStatus"),
        QStringLiteral("m_spTaskStatus"),
        QStringLiteral("m_dmTaskStatus"),
        QStringLiteral("m_overlapTaskStatus"),
        QStringLiteral("m_obsNetTaskStatus"),
        QStringLiteral("m_maskTaskStatus"),
        QStringLiteral("m_logDock"),
        QStringLiteral("m_logBtn"),
        QStringLiteral("m_featureMatchingSetting"),
        QStringLiteral("m_uiSetting"),
        QStringLiteral("m_lastSelectedImage"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName)) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, MenuWorkflowControllerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("DialogSettingStore *_featurePointVisualizationSetting = nullptr;"),
        QStringLiteral("DialogSettingStore *_baSetting = nullptr;"),
        QStringLiteral("DialogSettingStore *_mapSetting = nullptr;"),
        QStringLiteral("DialogSettingStore *_dcSetting = nullptr;"),
        QStringLiteral("DialogSettingStore *_aerialTriangulationSetting = nullptr;"),
        QStringLiteral("QPointer<QMainWindow> _mainWindow;"),
        QStringLiteral("ProjectManager *_projectManager = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const int publicIndex = header.indexOf(QStringLiteral("public:"));
    const int ctorIndex = header.indexOf(QStringLiteral("explicit MenuWorkflowController"), publicIndex);
    ASSERT_GE(publicIndex, 0);
    ASSERT_GT(ctorIndex, publicIndex);
    const QString publicDataBlock = header.mid(publicIndex, ctorIndex - publicIndex);
    EXPECT_FALSE(publicDataBlock.contains(QStringLiteral("DialogSettingStore *")));

    const QStringList oldMemberNames = {
        QStringLiteral("m_featureExtractionSetting"),
        QStringLiteral("m_vocabOverlapSetting"),
        QStringLiteral("m_featurePointVisualizationSetting"),
        QStringLiteral("m_baSetting"),
        QStringLiteral("m_mapSetting"),
        QStringLiteral("m_dcSetting"),
        QStringLiteral("m_threeDSetting"),
        QStringLiteral("m_aerialTriangulationSetting"),
        QStringLiteral("m_mainWindow"),
        QStringLiteral("m_projectManager"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
    }
}

TEST(CodeStyleTest, ReconstructionWorkflowControllerUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("QPointer<QMainWindow> _mainWindow;"),
        QStringLiteral("ProjectManager       *_projectManager = nullptr;"),
        QStringLiteral("DialogSettingStore *_generateModelStore = nullptr;"),
        QStringLiteral("DialogSettingStore *_texStore          = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_mainWindow"),
        QStringLiteral("m_projectManager"),
        QStringLiteral("m_obsNetStore"),
        QStringLiteral("m_initPoseStore"),
        QStringLiteral("m_triStore"),
        QStringLiteral("m_reconBaStore"),
        QStringLiteral("m_sparsePostStore"),
        QStringLiteral("m_denseMatchStore"),
        QStringLiteral("m_depthEstStore"),
        QStringLiteral("m_depthFuseStore"),
        QStringLiteral("m_denseRefStore"),
        QStringLiteral("m_meshStore"),
        QStringLiteral("m_texStore"),
        QStringLiteral("m_exportStore"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(","))) << qPrintable(oldName);
    }
}

TEST(MainWindowChromeTest, KeepsNativeMaximizeControlAvailable)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("Qt::WindowMaximizeButtonHint")));
}

TEST(WindowStateManagerTest, PersistsAndRestoresMaximizedState)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/config/settings/WindowStateManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("MainWindow/isMaximized")));
    EXPECT_TRUE(source.contains(QStringLiteral("mainWindow->isMaximized()")));
    EXPECT_TRUE(source.contains(QStringLiteral("wasMaximized")));
    EXPECT_TRUE(source.contains(QStringLiteral("Qt::WindowMaximized")));
    EXPECT_FALSE(source.contains(QStringLiteral("MainWindow/state")));
    EXPECT_FALSE(source.contains(QStringLiteral("saveState()")));
    EXPECT_FALSE(source.contains(QStringLiteral("restoreState(")));
}

TEST(AerialTriangulationDialogTest, UsesMetashapeStyleDefaultsAndCollectsSettings)
{
    AerialTriangulationDialog dialog;
    dialog.setImageCount(9);
    dialog.setReferencePreselectionAvailable(true, 9, 9);

    auto *qualityCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_qualityCombo"));
    auto *genericPreselectionCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_genericPreselectionCheck"));
    auto *referencePreselectionCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_referencePreselectionCheck"));
    auto *referenceSourceCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_referenceSourceCombo"));
    auto *resetAlignmentCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_resetAlignmentCheck"));
    auto *saveAfterEachStepCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_saveAfterEachStepCheck"));
    auto *keypointLimitSpin = dialog.findChild<QSpinBox *>(QStringLiteral("m_keypointLimitSpin"));
    auto *tiepointLimitSpin = dialog.findChild<QSpinBox *>(QStringLiteral("m_tiepointLimitSpin"));
    auto *maskApplyCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_maskApplyCombo"));
    auto *excludeFixedTiePointsCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_excludeFixedTiePointsCheck"));
    auto *guidedImageMatchingCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_guidedImageMatchingCheck"));
    auto *adaptiveCameraModelCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_adaptiveCameraModelCheck"));
    auto *reuseExistingMatchesCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_reuseExistingMatchesCheck"));
    auto *lockInputCameraPosesCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_lockInputCameraPosesCheck"));
    auto *statusLabel = dialog.findChild<QLabel *>(QStringLiteral("m_statusLabel"));
    auto *advancedToggle = dialog.findChild<QToolButton *>(QStringLiteral("m_advancedToggle"));
    auto *advancedContent = dialog.findChild<QWidget *>(QStringLiteral("m_advancedContent"));

    ASSERT_NE(qualityCombo, nullptr);
    ASSERT_NE(genericPreselectionCheck, nullptr);
    ASSERT_NE(referencePreselectionCheck, nullptr);
    ASSERT_NE(referenceSourceCombo, nullptr);
    ASSERT_NE(resetAlignmentCheck, nullptr);
    ASSERT_NE(saveAfterEachStepCheck, nullptr);
    ASSERT_NE(keypointLimitSpin, nullptr);
    ASSERT_NE(tiepointLimitSpin, nullptr);
    ASSERT_NE(maskApplyCombo, nullptr);
    ASSERT_NE(excludeFixedTiePointsCheck, nullptr);
    ASSERT_NE(guidedImageMatchingCheck, nullptr);
    ASSERT_NE(adaptiveCameraModelCheck, nullptr);
    ASSERT_NE(reuseExistingMatchesCheck, nullptr);
    ASSERT_NE(lockInputCameraPosesCheck, nullptr);
    ASSERT_NE(statusLabel, nullptr);
    ASSERT_NE(advancedToggle, nullptr);
    ASSERT_NE(advancedContent, nullptr);

    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("空中三角测量"));
    EXPECT_TRUE(statusLabel->isHidden());
    ASSERT_EQ(qualityCombo->count(), 5);
    EXPECT_EQ(qualityCombo->itemText(0), QStringLiteral("最高"));
    EXPECT_EQ(qualityCombo->itemData(0).toString(), QStringLiteral("highest"));
    EXPECT_EQ(qualityCombo->itemText(1), QStringLiteral("高"));
    EXPECT_EQ(qualityCombo->itemData(1).toString(), QStringLiteral("high"));
    EXPECT_EQ(qualityCombo->itemText(2), QStringLiteral("中"));
    EXPECT_EQ(qualityCombo->itemData(2).toString(), QStringLiteral("medium"));
    EXPECT_EQ(qualityCombo->itemText(3), QStringLiteral("低"));
    EXPECT_EQ(qualityCombo->itemData(3).toString(), QStringLiteral("low"));
    EXPECT_EQ(qualityCombo->itemText(4), QStringLiteral("最低"));
    EXPECT_EQ(qualityCombo->itemData(4).toString(), QStringLiteral("lowest"));
    EXPECT_EQ(qualityCombo->currentData().toString(), QStringLiteral("high"));

    ASSERT_EQ(referenceSourceCombo->count(), 3);
    EXPECT_EQ(referenceSourceCombo->itemText(0), QStringLiteral("导入参考"));
    EXPECT_EQ(referenceSourceCombo->itemData(0).toString(), QStringLiteral("source_code"));
    EXPECT_EQ(referenceSourceCombo->itemText(1), QStringLiteral("已估位姿"));
    EXPECT_EQ(referenceSourceCombo->itemData(1).toString(), QStringLiteral("estimated"));
    EXPECT_EQ(referenceSourceCombo->itemText(2), QStringLiteral("照片序列"));
    EXPECT_EQ(referenceSourceCombo->itemData(2).toString(), QStringLiteral("sequence"));
    EXPECT_TRUE(referenceSourceCombo->itemData(0, Qt::ToolTipRole).toString().contains(QStringLiteral("相机文件")));
    EXPECT_TRUE(referenceSourceCombo->itemData(1, Qt::ToolTipRole).toString().contains(QStringLiteral("已有对齐")));
    EXPECT_TRUE(referenceSourceCombo->itemData(2, Qt::ToolTipRole).toString().contains(QStringLiteral("影像顺序")));

    EXPECT_FALSE(advancedToggle->isChecked());
    EXPECT_TRUE(advancedContent->isHidden());
    advancedToggle->setChecked(true);
    EXPECT_FALSE(advancedContent->isHidden());
    EXPECT_EQ(advancedToggle->arrowType(), Qt::DownArrow);
    EXPECT_GE(dialog.minimumHeight(), 560);
    advancedToggle->setChecked(false);
    EXPECT_TRUE(advancedContent->isHidden());
    EXPECT_EQ(advancedToggle->arrowType(), Qt::RightArrow);

    EXPECT_TRUE(genericPreselectionCheck->isChecked());
    EXPECT_FALSE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(referenceSourceCombo->isEnabled());
    EXPECT_TRUE(resetAlignmentCheck->isChecked());
    EXPECT_FALSE(saveAfterEachStepCheck->isChecked());
    EXPECT_EQ(keypointLimitSpin->value(), 40000);
    EXPECT_EQ(tiepointLimitSpin->value(), 4000);
    EXPECT_EQ(maskApplyCombo->currentData().toString(), QStringLiteral("keypoints"));
    EXPECT_TRUE(excludeFixedTiePointsCheck->isChecked());
    EXPECT_FALSE(guidedImageMatchingCheck->isChecked());
    EXPECT_TRUE(adaptiveCameraModelCheck->isChecked());
    EXPECT_TRUE(adaptiveCameraModelCheck->toolTip().contains(QStringLiteral("焦距初始化")));
    EXPECT_TRUE(adaptiveCameraModelCheck->toolTip().contains(QStringLiteral("BA")));
    EXPECT_TRUE(reuseExistingMatchesCheck->isChecked());
    EXPECT_TRUE(reuseExistingMatchesCheck->toolTip().contains(QStringLiteral("SfM/BA")));
    EXPECT_FALSE(lockInputCameraPosesCheck->isChecked());
    EXPECT_TRUE(lockInputCameraPosesCheck->toolTip().contains(
        QStringLiteral("Middlebury")));

    QJsonObject settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("workflow_kind")).toString(),
              QStringLiteral("aerial_triangulation_dialog_only"));
    EXPECT_EQ(settings.value(QStringLiteral("quality")).toString(), QStringLiteral("high"));
    EXPECT_TRUE(settings.value(QStringLiteral("generic_preselection")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("reference_preselection")).toBool());
    EXPECT_EQ(settings.value(QStringLiteral("keypoint_limit")).toInt(), 40000);
    EXPECT_EQ(settings.value(QStringLiteral("tiepoint_limit")).toInt(), 4000);
    EXPECT_EQ(settings.value(QStringLiteral("mask_apply_mode")).toString(), QStringLiteral("keypoints"));
    EXPECT_TRUE(settings.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("reuse_existing_matches")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("lock_input_camera_poses")).toBool());

    QJsonObject appliedSettings;
    appliedSettings[QStringLiteral("quality")] = QStringLiteral("highest");
    appliedSettings[QStringLiteral("generic_preselection")] = false;
    appliedSettings[QStringLiteral("reference_preselection")] = true;
    appliedSettings[QStringLiteral("reference_preselection_source")] = QStringLiteral("sequence");
    appliedSettings[QStringLiteral("reset_current_alignment")] = false;
    appliedSettings[QStringLiteral("save_project_after_each_step")] = true;
    appliedSettings[QStringLiteral("keypoint_limit")] = 12000;
    appliedSettings[QStringLiteral("tiepoint_limit")] = 1800;
    appliedSettings[QStringLiteral("mask_apply_mode")] = QStringLiteral("tiepoints");
    appliedSettings[QStringLiteral("exclude_fixed_tie_points")] = false;
    appliedSettings[QStringLiteral("guided_image_matching")] = true;
    appliedSettings[QStringLiteral("adaptive_camera_model_fitting")] = true;
    appliedSettings[QStringLiteral("reuse_existing_matches")] = false;
    appliedSettings[QStringLiteral("lock_input_camera_poses")] = true;
    dialog.applySettings(appliedSettings);

    settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("quality")).toString(), QStringLiteral("highest"));
    EXPECT_FALSE(settings.value(QStringLiteral("generic_preselection")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("reference_preselection")).toBool());
    EXPECT_EQ(settings.value(QStringLiteral("reference_preselection_source")).toString(),
              QStringLiteral("sequence"));
    EXPECT_FALSE(settings.value(QStringLiteral("reset_current_alignment")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("save_project_after_each_step")).toBool());
    EXPECT_EQ(settings.value(QStringLiteral("keypoint_limit")).toInt(), 12000);
    EXPECT_EQ(settings.value(QStringLiteral("tiepoint_limit")).toInt(), 1800);
    EXPECT_EQ(settings.value(QStringLiteral("mask_apply_mode")).toString(), QStringLiteral("tiepoints"));
    EXPECT_FALSE(settings.value(QStringLiteral("exclude_fixed_tie_points")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("guided_image_matching")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("reuse_existing_matches")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("lock_input_camera_poses")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("reset_current_alignment")).toBool());

    resetAlignmentCheck->setChecked(true);
    settings = dialog.collectSettings();
    EXPECT_FALSE(settings.value(QStringLiteral("lock_input_camera_poses")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("reset_current_alignment")).toBool());
}

TEST(AerialTriangulationDialogTest, ReferencePreselectionStaysClickableWhenCamerasAreIncomplete)
{
    AerialTriangulationDialog dialog;
    dialog.setImageCount(3);

    auto *referencePreselectionCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_referencePreselectionCheck"));
    auto *referenceSourceCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_referenceSourceCombo"));
    ASSERT_NE(referencePreselectionCheck, nullptr);
    ASSERT_NE(referenceSourceCombo, nullptr);

    QJsonObject requested;
    requested[QStringLiteral("reference_preselection")] = true;
    requested[QStringLiteral("reference_preselection_source")] = QStringLiteral("sequence");

    dialog.applySettings(requested);
    EXPECT_TRUE(referencePreselectionCheck->isEnabled())
        << "参考预选入口应保持可点击，不能让用户误以为 checkbox 坏掉。";
    EXPECT_TRUE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(referenceSourceCombo->isEnabled());
    EXPECT_TRUE(dialog.collectSettings().value(QStringLiteral("reference_preselection")).toBool());

    dialog.setReferencePreselectionAvailable(true, 3, 3);
    dialog.applySettings(requested);
    EXPECT_TRUE(referencePreselectionCheck->isEnabled());
    EXPECT_TRUE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(referenceSourceCombo->isEnabled());
    EXPECT_TRUE(dialog.collectSettings().value(QStringLiteral("reference_preselection")).toBool());

    dialog.setReferencePreselectionAvailable(false, 1, 3);
    EXPECT_TRUE(referencePreselectionCheck->isEnabled());
    EXPECT_TRUE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(referenceSourceCombo->isEnabled());
    EXPECT_TRUE(dialog.collectSettings().value(QStringLiteral("reference_preselection")).toBool());
    EXPECT_TRUE(referencePreselectionCheck->toolTip().contains(QStringLiteral("相机")));

    referencePreselectionCheck->setChecked(false);
    EXPECT_FALSE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(referenceSourceCombo->isEnabled());
    EXPECT_FALSE(dialog.collectSettings().value(QStringLiteral("reference_preselection")).toBool());
}

TEST(AerialTriangulationDialogTest, ReferencePreselectionTogglesFromVisibleCheckboxClick)
{
    AerialTriangulationDialog dialog;
    dialog.setReferencePreselectionAvailable(true, 3, 3);
    dialog.show();
    QApplication::processEvents();

    auto *referencePreselectionCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_referencePreselectionCheck"));
    auto *referenceSourceCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_referenceSourceCombo"));
    ASSERT_NE(referencePreselectionCheck, nullptr);
    ASSERT_NE(referenceSourceCombo, nullptr);
    ASSERT_TRUE(referencePreselectionCheck->isVisibleTo(&dialog));
    ASSERT_TRUE(referencePreselectionCheck->isEnabled());
    ASSERT_FALSE(referencePreselectionCheck->isChecked());

    const QPoint clickPoint(referencePreselectionCheck->rect().left() + 56,
                            referencePreselectionCheck->rect().center().y());
    QTest::mouseClick(referencePreselectionCheck, Qt::LeftButton, Qt::NoModifier, clickPoint);
    QApplication::processEvents();

    EXPECT_TRUE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(referenceSourceCombo->isEnabled());
    EXPECT_TRUE(dialog.collectSettings().value(QStringLiteral("reference_preselection")).toBool());

    QTest::mouseClick(referencePreselectionCheck, Qt::LeftButton, Qt::NoModifier, clickPoint);
    QApplication::processEvents();

    EXPECT_FALSE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(referenceSourceCombo->isEnabled());
    EXPECT_FALSE(dialog.collectSettings().value(QStringLiteral("reference_preselection")).toBool());
}

TEST(AerialTriangulationDialogTest, SelectingReferenceSourceEnablesReferencePreselection)
{
    AerialTriangulationDialog dialog;
    dialog.setReferencePreselectionAvailable(true, 3, 3);

    auto *referencePreselectionCheck =
        dialog.findChild<QCheckBox *>(QStringLiteral("m_referencePreselectionCheck"));
    auto *referenceSourceCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_referenceSourceCombo"));
    ASSERT_NE(referencePreselectionCheck, nullptr);
    ASSERT_NE(referenceSourceCombo, nullptr);
    ASSERT_FALSE(referencePreselectionCheck->isChecked());
    ASSERT_TRUE(referenceSourceCombo->isEnabled())
        << "参考来源下拉框必须可点，否则用户会以为“导入参考”这一行坏了。";

    const int sequenceIndex = referenceSourceCombo->findData(QStringLiteral("sequence"));
    ASSERT_GE(sequenceIndex, 0);
    referenceSourceCombo->setCurrentIndex(sequenceIndex);

    EXPECT_TRUE(referencePreselectionCheck->isChecked());
    EXPECT_TRUE(dialog.collectSettings().value(QStringLiteral("reference_preselection")).toBool());
    EXPECT_EQ(dialog.collectSettings().value(QStringLiteral("reference_preselection_source")).toString(),
              QStringLiteral("sequence"));
}

TEST(AerialTriangulationDialogTest, InputControlsHaveReadableHeights)
{
    AerialTriangulationDialog dialog;
    auto *advancedToggle = dialog.findChild<QToolButton *>(QStringLiteral("m_advancedToggle"));
    ASSERT_NE(advancedToggle, nullptr);
    advancedToggle->setChecked(true);

    const QList<QWidget *> inputControls = {
        dialog.findChild<QComboBox *>(QStringLiteral("m_qualityCombo")),
        dialog.findChild<QComboBox *>(QStringLiteral("m_referenceSourceCombo")),
        dialog.findChild<QComboBox *>(QStringLiteral("m_maskApplyCombo")),
        dialog.findChild<QSpinBox *>(QStringLiteral("m_keypointLimitSpin")),
        dialog.findChild<QSpinBox *>(QStringLiteral("m_tiepointLimitSpin")),
    };
    for (QWidget *control : inputControls)
    {
        ASSERT_NE(control, nullptr);
        EXPECT_GE(control->minimumHeight(), 28) << control->objectName().toStdString();
    }

    const QList<QWidget *> checkBoxes = {
        dialog.findChild<QCheckBox *>(QStringLiteral("m_genericPreselectionCheck")),
        dialog.findChild<QCheckBox *>(QStringLiteral("m_referencePreselectionCheck")),
        dialog.findChild<QCheckBox *>(QStringLiteral("m_resetAlignmentCheck")),
        dialog.findChild<QCheckBox *>(QStringLiteral("m_saveAfterEachStepCheck")),
        dialog.findChild<QCheckBox *>(QStringLiteral("m_excludeFixedTiePointsCheck")),
        dialog.findChild<QCheckBox *>(QStringLiteral("m_guidedImageMatchingCheck")),
        dialog.findChild<QCheckBox *>(QStringLiteral("m_adaptiveCameraModelCheck")),
    };
    for (QWidget *checkBox : checkBoxes)
    {
        ASSERT_NE(checkBox, nullptr);
        EXPECT_GE(checkBox->minimumHeight(), 24) << checkBox->objectName().toStdString();
    }
}

TEST(AerialTriangulationDialogTest, CheckedBoxesUseCheckmarkIcon)
{
    const QString qss = readProjectSourceFile(QStringLiteral("resources/styles/app.qss"));
    const QString qrc = readProjectSourceFile(QStringLiteral("resources/resources.qrc"));
    ASSERT_FALSE(qss.isEmpty());
    ASSERT_FALSE(qrc.isEmpty());

    const int checkedStart = qss.indexOf(QStringLiteral("QCheckBox::indicator:checked"));
    ASSERT_GE(checkedStart, 0);
    const int checkedEnd = qss.indexOf(QStringLiteral("}"), checkedStart);
    ASSERT_GT(checkedEnd, checkedStart);
    const QString checkedBlock = qss.mid(checkedStart, checkedEnd - checkedStart);

    EXPECT_TRUE(checkedBlock.contains(QStringLiteral("image: url(:/icons/checkmark_white.xpm)")))
        << "Checked boxes should render a check mark instead of a solid blue square.";
    EXPECT_TRUE(qrc.contains(QStringLiteral("icons/checkmark_white.xpm")));
}

TEST(DepthQualityProfileTest, MapsStableIdsToFinalDownsample)
{
    using xjw::gui::project::DepthQualityProfile;
    using xjw::gui::project::depthQualityDownsample;
    using xjw::gui::project::depthQualityProfileFromId;
    using xjw::gui::project::depthQualityProfileId;

    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Highest), 1);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::High), 2);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Medium), 4);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Low), 8);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Lowest), 16);

    EXPECT_EQ(depthQualityProfileId(DepthQualityProfile::Highest), QStringLiteral("highest"));
    EXPECT_EQ(depthQualityProfileId(DepthQualityProfile::Medium), QStringLiteral("medium"));
    EXPECT_EQ(depthQualityProfileFromId(QStringLiteral("high")), DepthQualityProfile::High);
    EXPECT_EQ(depthQualityProfileFromId(QStringLiteral("unknown")), DepthQualityProfile::Medium);
}

TEST(DepthQualityProfileTest, ExplicitSettingsAreNotRaisedByDefaultProfile)
{
    QJsonObject json;
    json[QStringLiteral("qualityProfile")] = QStringLiteral("medium");
    json[QStringLiteral("minViews")] = 3;
    json[QStringLiteral("minConsistentViews")] = 2;
    json[QStringLiteral("confidence")] = 0.25;
    json[QStringLiteral("minConfidence")] = 0.30;

    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(json);
    EXPECT_EQ(settings.qualityProfile, QStringLiteral("medium"));
    EXPECT_EQ(settings.minViews, 3);
    EXPECT_EQ(settings.minConsistentViews, 2);
    EXPECT_FLOAT_EQ(settings.patchMatchConfidence, 0.25f);
    EXPECT_FLOAT_EQ(settings.fusionMinConfidence, 0.30f);
}

TEST(DenseWorkflowConfigTest, MediumProfileFillsMissingProductionDefaults)
{
    QJsonObject legacySettings;
    legacySettings[QStringLiteral("qualityProfile")] = QStringLiteral("medium");

    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(legacySettings);
    EXPECT_EQ(settings.qualityProfile, QStringLiteral("medium"));
    EXPECT_EQ(settings.minViews, 6);
    EXPECT_EQ(settings.minConsistentViews, 3);
    EXPECT_FLOAT_EQ(settings.patchMatchConfidence, 0.60f);
    EXPECT_FLOAT_EQ(settings.fusionMinConfidence, 0.65f);

    const auto config = xjw::gui::project::buildDepthGenConfig(settings, 444);
    EXPECT_GE(config.numSourceViews, 6);
    EXPECT_GE(config.patchMatch.numSourceViews, 6);
    EXPECT_EQ(config.fusion.minConsistentViews, 3);
}

TEST(DenseWorkflowConfigTest, LowProfileKeepsThreeSourceViews)
{
    QJsonObject previewSettings;
    previewSettings[QStringLiteral("qualityProfile")] = QStringLiteral("low");

    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(previewSettings);
    EXPECT_EQ(settings.qualityProfile, QStringLiteral("low"));
    EXPECT_EQ(settings.minViews, 3);

    const auto config = xjw::gui::project::buildDepthGenConfig(settings, 444);
    EXPECT_EQ(config.numSourceViews, 3);
}

TEST(DenseWorkflowConfigTest, MapsAdaptiveDepthPyramidSettings)
{
    QJsonObject json;
    json[QStringLiteral("sceneProfile")] = QStringLiteral("aerial_terrain");
    json[QStringLiteral("depthFilterMode")] = QStringLiteral("aggressive");
    json[QStringLiteral("saveIntermediatePyramidLevels")] = true;

    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(json);
    const auto config = xjw::gui::project::buildDepthGenConfig(settings, 9);

    EXPECT_EQ(config.sceneProfile, xjw::mvs::MvsSceneProfile::AerialTerrain);
    EXPECT_EQ(config.depthFilterMode, xjw::mvs::DepthFilterMode::Aggressive);
    EXPECT_FALSE(config.adaptiveDepthFilterMode);
    EXPECT_TRUE(config.saveIntermediatePyramidLevels);
}

TEST(DenseWorkflowConfigTest, AutoDepthFilterKeepsAdaptiveMode)
{
    QJsonObject json;
    json[QStringLiteral("sceneProfile")] = QStringLiteral("orbital_object");
    json[QStringLiteral("depthFilterMode")] = QStringLiteral("auto");

    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(json);
    const auto config = xjw::gui::project::buildDepthGenConfig(settings, 16);

    EXPECT_EQ(config.sceneProfile, xjw::mvs::MvsSceneProfile::OrbitalObject);
    EXPECT_EQ(config.depthFilterMode, xjw::mvs::DepthFilterMode::Moderate);
    EXPECT_TRUE(config.adaptiveDepthFilterMode);
    EXPECT_TRUE(config.saveIntermediatePyramidLevels);
}

TEST(DenseWorkflowConfigTest, DefaultCudaSchedulingKeepsGuiResponsive)
{
    QJsonObject json;
    json[QStringLiteral("threads")] = 8;
    json[QStringLiteral("cuda")] = true;

    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(json);
    const auto config = xjw::gui::project::buildDepthGenConfig(settings, 16);

    EXPECT_EQ(config.gpuFrameWorkerCount, 1);
    EXPECT_EQ(config.cpuFrameWorkerCount, 0);
    EXPECT_EQ(config.cpuWorkerCount, 7);
}

TEST(DenseWorkflowConfigTest, CpuThreadBudgetIsSharedAcrossFrameWorkers)
{
    QJsonObject json;
    json[QStringLiteral("threads")] = 8;
    json[QStringLiteral("cuda")] = true;
    json[QStringLiteral("gpu_frame_workers")] = 2;
    json[QStringLiteral("cpu_frame_workers")] = 2;

    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(json);
    const auto config = xjw::gui::project::buildDepthGenConfig(settings, 16);

    EXPECT_EQ(config.gpuFrameWorkerCount, 2);
    EXPECT_EQ(config.cpuFrameWorkerCount, 2);
    EXPECT_EQ(config.cpuWorkerCount, 1);
}

TEST(DepthPyramidDiagnosticsTest, IntermediateLevelPreviewsFlowToPhotoOverlayOnly)
{
    const QString generator = readProjectSourceFile(
        QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString overlay = readProjectSourceFile(QStringLiteral("src/gui/views/DepthOverlayData.cpp"));
    const QString tree = readProjectSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    ASSERT_FALSE(generator.isEmpty());
    ASSERT_FALSE(overlay.isEmpty());
    ASSERT_FALSE(tree.isEmpty());

    EXPECT_TRUE(generator.contains(QStringLiteral("preview_path")));
    EXPECT_TRUE(generator.contains(QStringLiteral("confidence_preview_path")));
    EXPECT_TRUE(overlay.contains(QStringLiteral("pyramid_levels")));
    EXPECT_TRUE(overlay.contains(QStringLiteral("preview_path")));
    EXPECT_FALSE(tree.contains(QStringLiteral("pyramid_levels")));
    EXPECT_FALSE(tree.contains(QStringLiteral("appendItemRow(depth_frame_item")));
}

TEST(DenseCloudPostProcessMetadataTest, TerrainSpikeFilterPublishesProductionTerrainStage)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int refineStart = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync"));
    ASSERT_GE(refineStart, 0);
    const int recordStart = source.indexOf(QStringLiteral("QJsonObject record = makeDenseResultRecord"), refineStart);
    ASSERT_GE(recordStart, 0);
    const int upsertStart = source.indexOf(QStringLiteral("upsertProjectRecordByPath"), recordStart);
    ASSERT_GT(upsertStart, recordStart);
    const QString recordBlock = source.mid(recordStart, upsertStart - recordStart);

    EXPECT_TRUE(recordBlock.contains(QStringLiteral("QStringLiteral(\"production\")")));
    EXPECT_TRUE(recordBlock.contains(QStringLiteral("QStringLiteral(\"terrain\")")));
    EXPECT_TRUE(recordBlock.contains(QStringLiteral("QStringLiteral(\"dense_cloud_surface_cleanup\")")));
    EXPECT_TRUE(recordBlock.contains(QStringLiteral("QStringLiteral(\"streaming_cli\")")));
    EXPECT_TRUE(recordBlock.contains(QStringLiteral("request.terrainFilterPasses")));
    EXPECT_TRUE(recordBlock.contains(QStringLiteral("dense_refine_report")));

    const int fallbackRecordStart = source.indexOf(QStringLiteral("const bool terrainProductionCloud"), upsertStart);
    ASSERT_GT(fallbackRecordStart, upsertStart);
    const int fallbackUpsertStart = source.indexOf(QStringLiteral("upsertProjectRecordByPath"), fallbackRecordStart);
    ASSERT_GT(fallbackUpsertStart, fallbackRecordStart);
    const QString fallbackRecordBlock = source.mid(fallbackRecordStart, fallbackUpsertStart - fallbackRecordStart);

    EXPECT_TRUE(fallbackRecordBlock.contains(QStringLiteral("request.terrainSpikeFilterEnabled")));
    EXPECT_TRUE(fallbackRecordBlock.contains(QStringLiteral("QStringLiteral(\"production\")")));
    EXPECT_TRUE(fallbackRecordBlock.contains(QStringLiteral("QStringLiteral(\"terrain\")")));
    EXPECT_TRUE(fallbackRecordBlock.contains(QStringLiteral("QStringLiteral(\"dense_cloud_surface_cleanup\")")));
    EXPECT_TRUE(fallbackRecordBlock.contains(QStringLiteral("QStringLiteral(\"refined\")")));
}

TEST(TaskStatusWidgetTest, ShowsProgressAndPreservesCancellingState)
{
    TaskStatusWidget widget;
    widget.setCancellable(true);
    widget.setCancellingText(QStringLiteral("正在取消特征匹配..."));

    bool cancelEmitted = false;
    QObject::connect(&widget, &TaskStatusWidget::cancelRequested, &widget,
                     [&cancelEmitted]()
                     {
                         cancelEmitted = true;
                     });

    widget.begin(QStringLiteral("特征匹配 0/5"), 0, 5);
    EXPECT_TRUE(widget.isActive());
    EXPECT_EQ(widget.statusText(), QStringLiteral("特征匹配 0/5"));
    EXPECT_EQ(widget.progressValue(), 0);
    EXPECT_EQ(widget.progressMaximum(), 5);

    widget.updateProgress(QStringLiteral("特征匹配 2/5"), 2);
    EXPECT_EQ(widget.statusText(), QStringLiteral("特征匹配 2/5"));
    EXPECT_EQ(widget.progressValue(), 2);

    QToolButton *cancelButton = widget.findChild<QToolButton *>(QStringLiteral("cancelButton"));
    ASSERT_NE(cancelButton, nullptr);
    ASSERT_TRUE(cancelButton->isEnabled());
    cancelButton->click();

    EXPECT_TRUE(cancelEmitted);
    EXPECT_TRUE(widget.isCancelling());
    EXPECT_FALSE(cancelButton->isEnabled());
    EXPECT_EQ(cancelButton->text(), QStringLiteral("正在取消"));
    EXPECT_EQ(widget.statusText(), QStringLiteral("正在取消特征匹配..."));

    widget.updateProgress(QStringLiteral("特征匹配 3/5"), 3);
    EXPECT_EQ(widget.progressValue(), 3);
    EXPECT_EQ(widget.statusText(), QStringLiteral("正在取消特征匹配..."));

    widget.finish();
    EXPECT_FALSE(widget.isActive());
    EXPECT_FALSE(widget.isCancelling());
    EXPECT_EQ(cancelButton->text(), QStringLiteral("取消"));
}

TEST(ProjectDashboardWidgetTest, LoadsMetadataIntoReadOnlyWorkflowAndReferenceSummary)
{
    ProjectDashboardWidget widget;

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/img_001.tif")},
                              {QStringLiteral("camera"), QJsonObject{{QStringLiteral("fu"), 1000.0}}}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/img_002.tif")}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    meta[QStringLiteral("ipfind_results")] = QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("ip.json")}}};
    meta[QStringLiteral("reference_datasets")] =
        QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("scan.las")},
                               {QStringLiteral("type"), QStringLiteral("lidar")},
                               {QStringLiteral("role"), QStringLiteral("ba_prior")}}};
    meta[QStringLiteral("report_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("reference_quality")},
                               {QStringLiteral("path"), QStringLiteral("reference_quality.json")}}};

    widget.loadFromJson(meta);

    auto *summaryLabel = widget.findChild<QLabel *>(QStringLiteral("dashboardSummaryLabel"));
    ASSERT_NE(summaryLabel, nullptr);
    EXPECT_TRUE(summaryLabel->text().contains(QStringLiteral("影像 2")));
    EXPECT_TRUE(summaryLabel->text().contains(QStringLiteral("相机 1/2")));

    auto *referenceLabel = widget.findChild<QLabel *>(QStringLiteral("dashboardReferenceLabel"));
    ASSERT_NE(referenceLabel, nullptr);
    EXPECT_TRUE(referenceLabel->text().contains(QStringLiteral("LiDAR 1")));
    EXPECT_TRUE(referenceLabel->text().contains(QStringLiteral("BA约束 1")));

    auto *referenceTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardReferenceTable"));
    ASSERT_NE(referenceTable, nullptr);
    EXPECT_EQ(referenceTable->rowCount(), 1);
    EXPECT_EQ(referenceTable->editTriggers(), QAbstractItemView::NoEditTriggers);
    EXPECT_TRUE(referenceTable->item(0, 0)->text().contains(QStringLiteral("LiDAR")));
    EXPECT_TRUE(referenceTable->item(0, 1)->text().contains(QStringLiteral("BA约束")));

    auto *workflowTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardWorkflowTable"));
    ASSERT_NE(workflowTable, nullptr);
    EXPECT_GE(workflowTable->rowCount(), 8);
    EXPECT_EQ(workflowTable->editTriggers(), QAbstractItemView::NoEditTriggers);

    bool foundReferenceStep = false;
    for (int row = 0; row < workflowTable->rowCount(); ++row)
    {
        const QTableWidgetItem *stageItem = workflowTable->item(row, 1);
        const QTableWidgetItem *detailItem = workflowTable->item(row, 2);
        if (stageItem && stageItem->text().contains(QStringLiteral("LiDAR")))
        {
            foundReferenceStep = true;
            ASSERT_NE(detailItem, nullptr);
            EXPECT_TRUE(detailItem->text().contains(QStringLiteral("BA约束")));
        }
    }
    EXPECT_TRUE(foundReferenceStep);

    auto *reportTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardReportTable"));
    ASSERT_NE(reportTable, nullptr);
    EXPECT_EQ(reportTable->rowCount(), 1);
    EXPECT_EQ(reportTable->editTriggers(), QAbstractItemView::NoEditTriggers);
}

TEST(ProjectDashboardWidgetTest, HidesEmptyDetailTablesAndShowsPopulatedOnes)
{
    ProjectDashboardWidget widget;
    widget.clear();

    auto *taskTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardTaskTable"));
    auto *referenceTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardReferenceTable"));
    auto *qualityTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardQualityTable"));
    auto *reportTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardReportTable"));
    auto *workflowTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardWorkflowTable"));
    ASSERT_NE(taskTable, nullptr);
    ASSERT_NE(referenceTable, nullptr);
    ASSERT_NE(qualityTable, nullptr);
    ASSERT_NE(reportTable, nullptr);
    ASSERT_NE(workflowTable, nullptr);

    EXPECT_TRUE(taskTable->isHidden());
    EXPECT_TRUE(referenceTable->isHidden());
    EXPECT_TRUE(qualityTable->isHidden());
    EXPECT_TRUE(reportTable->isHidden());
    EXPECT_FALSE(workflowTable->isHidden());

    widget.setTaskSnapshots(QJsonArray{
        QJsonObject{{QStringLiteral("name"), QStringLiteral("MVS")},
                    {QStringLiteral("status_text"), QStringLiteral("运行中")},
                    {QStringLiteral("active"), true}}
    });
    EXPECT_FALSE(taskTable->isHidden());

    widget.loadFromJson(QJsonObject{
        {QStringLiteral("reference_datasets"),
         QJsonArray{QJsonObject{{QStringLiteral("path"), QStringLiteral("scan.las")},
                                {QStringLiteral("type"), QStringLiteral("lidar")}}}},
        {QStringLiteral("report_results"),
         QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("reconstruction_quality")},
                                {QStringLiteral("path"), QStringLiteral("quality.json")},
                                {QStringLiteral("sparse_point_count"), 100}}}}
    });
    EXPECT_FALSE(referenceTable->isHidden());
    EXPECT_FALSE(qualityTable->isHidden());
    EXPECT_FALSE(reportTable->isHidden());
}

TEST(ProjectDashboardWidgetTest, MainWindowUiExposesOverviewTab)
{
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.ui"));
    ASSERT_FALSE(ui.isEmpty());

    EXPECT_TRUE(ui.contains(QStringLiteral("ProjectDashboardWidget")));
    EXPECT_TRUE(ui.contains(QStringLiteral("<string>概览</string>")));
}

TEST(ProjectDashboardWidgetTest, MainWindowRefreshesDashboardFromProjectMetadata)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("ProjectDashboardWidget")));
    EXPECT_TRUE(source.contains(QStringLiteral("_dashboard")));
    EXPECT_TRUE(source.contains(QStringLiteral("scheduleProjectMetadataRefresh")));
    EXPECT_TRUE(source.contains(QStringLiteral("_dashboard->loadFromJson(meta)")));
    EXPECT_FALSE(source.contains(QStringLiteral("projectMetadataChanged, _dashboard")));
}

TEST(ProjectDashboardWidgetTest, ShowsQualityMetricsFromRegisteredReports)
{
    ProjectDashboardWidget widget;

    QJsonObject meta;
    meta[QStringLiteral("report_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("reconstruction_quality")},
                               {QStringLiteral("path"), QStringLiteral("quality.json")},
                               {QStringLiteral("total_image_count"), 12},
                               {QStringLiteral("registered_image_count"), 10},
                               {QStringLiteral("sparse_point_count"), 4200},
                               {QStringLiteral("dense_point_count"), 90000},
                               {QStringLiteral("mvs_valid_coverage"), 0.82},
                               {QStringLiteral("dem_coverage"), 0.76}}};

    widget.loadFromJson(meta);

    auto *qualityTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardQualityTable"));
    ASSERT_NE(qualityTable, nullptr);
    EXPECT_EQ(qualityTable->editTriggers(), QAbstractItemView::NoEditTriggers);
    EXPECT_GE(qualityTable->rowCount(), 4);

    bool foundMvsCoverage = false;
    bool foundSparsePoints = false;
    for (int row = 0; row < qualityTable->rowCount(); ++row)
    {
        const QTableWidgetItem *metricItem = qualityTable->item(row, 0);
        const QTableWidgetItem *valueItem = qualityTable->item(row, 1);
        ASSERT_NE(metricItem, nullptr);
        ASSERT_NE(valueItem, nullptr);
        if (metricItem->text().contains(QStringLiteral("MVS覆盖")))
        {
            foundMvsCoverage = true;
            EXPECT_TRUE(valueItem->text().contains(QStringLiteral("82")));
        }
        if (metricItem->text().contains(QStringLiteral("稀疏点")))
        {
            foundSparsePoints = true;
            EXPECT_TRUE(valueItem->text().contains(QStringLiteral("4200")));
        }
    }
    EXPECT_TRUE(foundMvsCoverage);
    EXPECT_TRUE(foundSparsePoints);
}

TEST(ProjectDashboardWidgetTest, ShowsUnavailableMvsCoverageWithoutWarning)
{
    ProjectDashboardWidget widget;
    widget.loadFromJson(QJsonObject{
        {QStringLiteral("report_results"),
         QJsonArray{QJsonObject{
             {QStringLiteral("type"), QStringLiteral("reconstruction_quality")},
             {QStringLiteral("path"), QStringLiteral("quality.json")},
             {QStringLiteral("total_image_count"), 16},
             {QStringLiteral("registered_image_count"), 16}}}}});

    auto *qualityTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardQualityTable"));
    ASSERT_NE(qualityTable, nullptr);
    bool foundUnavailableCoverage = false;
    for (int row = 0; row < qualityTable->rowCount(); ++row)
    {
        if (qualityTable->item(row, 0)->text().contains(QStringLiteral("MVS覆盖")))
        {
            foundUnavailableCoverage = true;
            EXPECT_EQ(qualityTable->item(row, 1)->text(), QStringLiteral("—"));
        }
    }
    EXPECT_TRUE(foundUnavailableCoverage);

    auto *alertTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardQualityAlertTable"));
    ASSERT_NE(alertTable, nullptr);
    for (int row = 0; row < alertTable->rowCount(); ++row)
    {
        EXPECT_FALSE(alertTable->item(row, 2)->text().contains(QStringLiteral("MVS覆盖")));
    }
}

TEST(ProjectDashboardWidgetTest, ShowsReferenceQualityAlertsAndErrorMetricsReadOnly)
{
    ProjectDashboardWidget widget;

    QJsonObject meta;
    meta[QStringLiteral("report_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("reference_quality")},
                               {QStringLiteral("path"), QStringLiteral("reference_quality.json")},
                               {QStringLiteral("status"), QStringLiteral("missing_project_products")},
                               {QStringLiteral("comparison_available"), false},
                               {QStringLiteral("rmse_m"), 0.184},
                               {QStringLiteral("p95_distance_m"), 0.420}},
                   QJsonObject{{QStringLiteral("type"), QStringLiteral("reference_terrain_prior_preflight")},
                               {QStringLiteral("path"), QStringLiteral("reference_prior.json")},
                               {QStringLiteral("ready"), false},
                               {QStringLiteral("status"), QStringLiteral("missing_aerial_triangulation")},
                               {QStringLiteral("ba_prior_reference_count"), 1}},
                   QJsonObject{{QStringLiteral("type"), QStringLiteral("reconstruction_quality")},
                               {QStringLiteral("path"), QStringLiteral("quality.json")},
                               {QStringLiteral("total_image_count"), 10},
                               {QStringLiteral("registered_image_count"), 7},
                               {QStringLiteral("mvs_valid_coverage"), 0.55},
                               {QStringLiteral("dem_coverage"), 0.41}}};

    widget.loadFromJson(meta);

    auto *alertTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardQualityAlertTable"));
    ASSERT_NE(alertTable, nullptr);
    EXPECT_EQ(alertTable->editTriggers(), QAbstractItemView::NoEditTriggers);
    EXPECT_GE(alertTable->rowCount(), 4);

    bool foundReferenceBlocker = false;
    bool foundTerrainPriorBlocker = false;
    bool foundRmseMetric = false;
    bool foundCoverageWarning = false;
    for (int row = 0; row < alertTable->rowCount(); ++row)
    {
        const QTableWidgetItem *levelItem = alertTable->item(row, 0);
        const QTableWidgetItem *sourceItem = alertTable->item(row, 1);
        const QTableWidgetItem *detailItem = alertTable->item(row, 2);
        ASSERT_NE(levelItem, nullptr);
        ASSERT_NE(sourceItem, nullptr);
        ASSERT_NE(detailItem, nullptr);

        if (sourceItem->text().contains(QStringLiteral("参考数据质检"))
            && detailItem->text().contains(QStringLiteral("missing_project_products")))
        {
            foundReferenceBlocker = true;
            EXPECT_TRUE(levelItem->text().contains(QStringLiteral("阻塞")));
        }
        if (sourceItem->text().contains(QStringLiteral("参考地形平差"))
            && detailItem->text().contains(QStringLiteral("missing_aerial_triangulation")))
        {
            foundTerrainPriorBlocker = true;
            EXPECT_TRUE(levelItem->text().contains(QStringLiteral("阻塞")));
        }
        if (detailItem->text().contains(QStringLiteral("RMSE"))
            && detailItem->text().contains(QStringLiteral("0.184")))
        {
            foundRmseMetric = true;
        }
        if (sourceItem->text().contains(QStringLiteral("重建质量"))
            && detailItem->text().contains(QStringLiteral("MVS覆盖")))
        {
            foundCoverageWarning = true;
            EXPECT_TRUE(levelItem->text().contains(QStringLiteral("注意")));
        }
    }

    EXPECT_TRUE(foundReferenceBlocker);
    EXPECT_TRUE(foundTerrainPriorBlocker);
    EXPECT_TRUE(foundRmseMetric);
    EXPECT_TRUE(foundCoverageWarning);
}

TEST(ProjectDashboardWidgetTest, ShowsReadOnlyRunningTaskSnapshots)
{
    ProjectDashboardWidget widget;

    QJsonArray tasks;
    tasks.append(QJsonObject{{QStringLiteral("name"), QStringLiteral("MVS/稠密重建")},
                             {QStringLiteral("status_text"), QStringLiteral("正在估计深度图")},
                             {QStringLiteral("active"), true},
                             {QStringLiteral("cancelling"), false},
                             {QStringLiteral("progress_value"), 32},
                             {QStringLiteral("progress_maximum"), 100}});
    tasks.append(QJsonObject{{QStringLiteral("name"), QStringLiteral("空三/BA")},
                             {QStringLiteral("status_text"), QStringLiteral("等待")},
                             {QStringLiteral("active"), false},
                             {QStringLiteral("progress_value"), 0},
                             {QStringLiteral("progress_maximum"), 100}});

    widget.setTaskSnapshots(tasks);

    auto *taskLabel = widget.findChild<QLabel *>(QStringLiteral("dashboardTaskLabel"));
    ASSERT_NE(taskLabel, nullptr);
    EXPECT_TRUE(taskLabel->text().contains(QStringLiteral("当前运行任务 1")));
    EXPECT_TRUE(taskLabel->text().contains(QStringLiteral("只读")));

    auto *taskTable = widget.findChild<QTableWidget *>(QStringLiteral("dashboardTaskTable"));
    ASSERT_NE(taskTable, nullptr);
    EXPECT_EQ(taskTable->editTriggers(), QAbstractItemView::NoEditTriggers);
    ASSERT_EQ(taskTable->rowCount(), 1);
    EXPECT_TRUE(taskTable->item(0, 0)->text().contains(QStringLiteral("MVS")));
    EXPECT_TRUE(taskTable->item(0, 1)->text().contains(QStringLiteral("正在估计深度图")));
    EXPECT_TRUE(taskTable->item(0, 2)->text().contains(QStringLiteral("32/100")));
}

TEST(ProjectDashboardWidgetTest, MainWindowMirrorsTaskStatusSnapshotsReadOnly)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("refreshDashboardTaskSnapshots")));
    EXPECT_TRUE(source.contains(QStringLiteral("setTaskSnapshots")));
    EXPECT_TRUE(source.contains(QStringLiteral("TaskStatusWidget *widget")));
    EXPECT_TRUE(source.contains(QStringLiteral("cancelRequested")));
}

TEST(AerialTriangulationModuleLayoutTest, AlignPhotosCodeLivesInDedicatedCoreModuleWithoutPipelineCompat)
{
    const QString workflowHeader = readProjectSourceFile(
        QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.h"));
    const QString moduleCmake = readProjectSourceFile(
        QStringLiteral("src/core/aerial_triangulation/CMakeLists.txt"));
    const QString guiSources = readProjectSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    const QString cliCmake = readProjectSourceFile(
        QStringLiteral("src/cli/workflows/CMakeLists.txt"));

    EXPECT_TRUE(workflowHeader.contains(QStringLiteral("class AerialTriangulationWorkflow")));
    EXPECT_TRUE(moduleCmake.contains(QStringLiteral("add_library(aerial_triangulation STATIC")));
    EXPECT_TRUE(moduleCmake.contains(QStringLiteral("workflow/AerialTriangulationWorkflow.cpp")));

    EXPECT_FALSE(QFileInfo(QStringLiteral("src/core/pipeline/SFMService.h")).exists());
    EXPECT_FALSE(QFileInfo(QStringLiteral("src/core/pipeline/SFMService.cpp")).exists());
    EXPECT_FALSE(QFileInfo(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationWorkflow.h")).exists());
    EXPECT_FALSE(QFileInfo(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationWorkflow.cpp")).exists());

    EXPECT_FALSE(guiSources.contains(QStringLiteral("aerial_triangulation/AerialTriangulation")));
    EXPECT_TRUE(cliCmake.contains(QStringLiteral("aerial_triangulation")));
    EXPECT_FALSE(guiSources.contains(QStringLiteral("../core/pipeline/SFMService.cpp")));
    EXPECT_FALSE(cliCmake.contains(QStringLiteral("src/core/pipeline/SFMService.cpp")));
}

TEST(AerialTriangulationWorkflowTest, SparseOnlyWorkflowStopsBeforeDenseStages)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_FALSE(header.contains(QStringLiteral("openAerialTriangulationDialog")));
    EXPECT_TRUE(header.contains(QStringLiteral("openWorkflowAerialTriangulationDialog")));
    EXPECT_TRUE(header.contains(QStringLiteral("startAerialTriangulationWorkflow")));
    EXPECT_TRUE(source.contains(QStringLiteral("DialogSettingKeys::AerialTriangulation")));
    const int workflowDialogStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openWorkflowAerialTriangulationDialog"));
    ASSERT_GE(workflowDialogStart, 0);
    const int workflowDialogEnd = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"),
        workflowDialogStart);
    ASSERT_GT(workflowDialogEnd, workflowDialogStart);
    const QString workflowDialogBody = source.mid(workflowDialogStart, workflowDialogEnd - workflowDialogStart);
    EXPECT_TRUE(workflowDialogBody.contains(QStringLiteral("AerialTriangulationDialog")));
    EXPECT_TRUE(workflowDialogBody.contains(QStringLiteral("dlg.exec()")));
    EXPECT_TRUE(workflowDialogBody.contains(QStringLiteral("dlg.collectSettings()")));
    EXPECT_TRUE(workflowDialogBody.contains(QStringLiteral("startAerialTriangulationWorkflow(settings)")));

    EXPECT_TRUE(source.contains(QStringLiteral("source\"] = QStringLiteral(\"aerial_triangulation\")"))
                || source.contains(QStringLiteral("source\", QStringLiteral(\"aerial_triangulation\")"))
                || source.contains(QStringLiteral("resultRecordExtra[QStringLiteral(\"source\")] = QStringLiteral(\"aerial_triangulation\")")));

    const int sparseStart = source.indexOf(QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(sparseStart, 0);
    const int nextFunction = source.indexOf(QStringLiteral("void MenuWorkflowController::openOverlapAnalysisDialog"),
                                            sparseStart);
    ASSERT_GT(nextFunction, sparseStart);
    const QString sparseBlock = source.mid(sparseStart, nextFunction - sparseStart);
    EXPECT_TRUE(sparseBlock.contains(QStringLiteral("AerialTriangulationWorkflow::run")));
    EXPECT_TRUE(sparseBlock.contains(QStringLiteral("replaceTiePointResult")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startGenerateDenseCloudAsync")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startDenseCloudRefineAsync")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startMeshReconstructionAsync")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startDenseMatch")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startMVS")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startMesh")));
}

TEST(AerialTriangulationWorkflowTest, ReferencePreselectionRequiresCompleteCameraReferences)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    const int dialogStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openWorkflowAerialTriangulationDialog"));
    ASSERT_GE(dialogStart, 0);
    const int workflowStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"),
        dialogStart);
    ASSERT_GT(workflowStart, dialogStart);
    const QString dialogBody = source.mid(dialogStart, workflowStart - dialogStart);
    EXPECT_TRUE(dialogBody.contains(QStringLiteral("getCamerasForImages(images, &hasAllReferenceCameras)")));
    EXPECT_TRUE(dialogBody.contains(QStringLiteral("dlg.setReferencePreselectionAvailable")))
        << "空三参数对话框要在无相机文件时禁用参考预选。";

    EXPECT_TRUE(source.contains(QStringLiteral("sanitizeAerialTriangulationReferencePreselection")));
    const int helperStart = source.indexOf(
        QStringLiteral("QJsonObject MenuWorkflowController::sanitizeAerialTriangulationReferencePreselection"));
    ASSERT_GE(helperStart, 0);
    const int helperEnd = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"),
        helperStart);
    ASSERT_GT(helperEnd, helperStart);
    const QString helperBody = source.mid(helperStart, helperEnd - helperStart);
    EXPECT_TRUE(helperBody.contains(QStringLiteral("settings[QStringLiteral(\"reference_preselection\")] = false")));
    EXPECT_TRUE(helperBody.contains(QStringLiteral("getCamerasForImages(images, &hasAllReferenceCameras)")))
        << "后端启动前也要重新检查相机文件，不能只依赖 UI。";
    EXPECT_TRUE(helperBody.contains(QStringLiteral("参考预选已关闭")));

    const int startEnd = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"),
        workflowStart);
    ASSERT_GT(startEnd, workflowStart);
    const QString startBody = source.mid(workflowStart, startEnd - workflowStart);
    EXPECT_TRUE(startBody.contains(
        QStringLiteral("sanitizeAerialTriangulationReferencePreselection(runSettings, images)")));
}

TEST(AerialTriangulationWorkflowTest, TiePointPreparationUsesUnifiedDeviceMapping)
{
    const QString guiSource = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString workflow = readProjectSourceFile(
        QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));
    ASSERT_FALSE(guiSource.isEmpty());
    ASSERT_FALSE(workflow.isEmpty());

    const int helperStart = workflow.indexOf(
        QStringLiteral("matchphotos::ComputeDevice computeDevice"));
    ASSERT_GE(helperStart, 0);
    const int helperEnd = workflow.indexOf(
        QStringLiteral("QString normalizeMatcher"),
        helperStart);
    ASSERT_GT(helperEnd, helperStart);
    const QString helperBody = workflow.mid(helperStart, helperEnd - helperStart);

    EXPECT_TRUE(helperBody.contains(QStringLiteral("return matchphotos::ComputeDevice::Cuda")));
    EXPECT_TRUE(helperBody.contains(QStringLiteral("return matchphotos::ComputeDevice::Cpu")))
        << "用户显式写入 cpu 时仍应保留 CPU 调试路径。";
    EXPECT_TRUE(helperBody.contains(QStringLiteral("return matchphotos::ComputeDevice::Auto")));

    const int optionsStart = workflow.indexOf(
        QStringLiteral("AerialTriangulationResolvedConfig AerialTriangulationWorkflow::resolveConfig"));
    ASSERT_GE(optionsStart, 0);
    const int optionsEnd = workflow.indexOf(
        QStringLiteral("AerialTriangulationResult AerialTriangulationWorkflow::run"),
        optionsStart);
    ASSERT_GT(optionsEnd, optionsStart);
    const QString optionsBody = workflow.mid(optionsStart, optionsEnd - optionsStart);

    EXPECT_TRUE(optionsBody.contains(QStringLiteral("tieOptions.device = computeDevice(options.device)")));
    EXPECT_TRUE(guiSource.contains(
        QStringLiteral("settings.value(QStringLiteral(\"device\")).toString(QStringLiteral(\"auto\"))")));
}

TEST(AerialTriangulationWorkflowTest, TiePointPreparationPassesMaskOptionsToMatchPhotosTask)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString workflow = readProjectSourceFile(
        QStringLiteral("src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(workflow.isEmpty());

    const int optionsStart = workflow.indexOf(
        QStringLiteral("AerialTriangulationResolvedConfig AerialTriangulationWorkflow::resolveConfig"));
    ASSERT_GE(optionsStart, 0);
    const int optionsEnd = workflow.indexOf(
        QStringLiteral("AerialTriangulationResult AerialTriangulationWorkflow::run"),
        optionsStart);
    ASSERT_GT(optionsEnd, optionsStart);
    const QString optionsBody = workflow.mid(optionsStart, optionsEnd - optionsStart);

    EXPECT_TRUE(optionsBody.contains(QStringLiteral("tieOptions.maskApplyMode")))
        << "空三对话框的掩膜参数必须传给创建连接点任务。";
    EXPECT_TRUE(optionsBody.contains(QStringLiteral("options.maskApplyMode")));
    EXPECT_TRUE(optionsBody.contains(QStringLiteral("tieOptions.reuseExistingFeatures")))
        << "将掩膜应用于关键点时，旧特征文件可能没有经过掩膜过滤，必须强制重提。";

    const int unifiedStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"));
    ASSERT_GE(unifiedStart, 0);
    const int unifiedEnd = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openOverlapAnalysisDialog"),
        unifiedStart);
    ASSERT_GT(unifiedEnd, unifiedStart);
    const QString unifiedBody = source.mid(unifiedStart, unifiedEnd - unifiedStart);

    EXPECT_TRUE(unifiedBody.contains(
        QStringLiteral("workflowOptions.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages(projectPath, images)")))
        << "空三自动创建连接点时也必须按项目影像装载蒙版路径。";
    EXPECT_TRUE(optionsBody.contains(QStringLiteral("tieContext.progressCallback")))
        << "空三自动创建连接点不能只显示粗略文字，必须透传 MatchPhotosTask 的阶段进度。";
    EXPECT_TRUE(optionsBody.contains(QStringLiteral("tiePointProgress")))
        << "连接点准备阶段的特征提取、配对、匹配进度要映射到空三总进度条。";
}

TEST(AerialTriangulationWorkflowTest, DefaultsToSiftLightGlueWhenDialogOmitsMatchPipeline)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int startBegin = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(startBegin, 0);
    const int launchBegin = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"),
        startBegin);
    ASSERT_GT(launchBegin, startBegin);
    const QString startBody = source.mid(startBegin, launchBegin - startBegin);

    EXPECT_TRUE(startBody.contains(
        QStringLiteral("runSettings.value(QStringLiteral(\"feature_algorithm\")).toString(QStringLiteral(\"sift\"))")))
        << "空三参数对话框没有算法字段时，前置检查必须按创建连接点的 SIFT + LightGlue 默认链路检查缓存。";
    EXPECT_FALSE(startBody.contains(
        QStringLiteral("runSettings.value(QStringLiteral(\"feature_algorithm\")).toString(QStringLiteral(\"disk\"))")));

    const int launchEnd = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openOverlapAnalysisDialog"),
        launchBegin);
    ASSERT_GT(launchEnd, launchBegin);
    const QString launchBody = source.mid(launchBegin, launchEnd - launchBegin);

    EXPECT_TRUE(launchBody.contains(QStringLiteral("settings.value(QStringLiteral(\"feature_algorithm\"))")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral(".toString(QStringLiteral(\"sift\"))")))
        << "正式 SfM 检查匹配缓存时也必须沿用 SIFT + LightGlue，避免把刚生成的 SIFT 匹配误判为 DISK 不兼容。";
    EXPECT_FALSE(launchBody.contains(QStringLiteral(".toString(QStringLiteral(\"disk\"))")));
}

TEST(AerialTriangulationWorkflowTest, CompletedButUnusableMatchingOnlyBlocksWhenReusingExistingMatches)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("blockOnMatchQuality")));
    EXPECT_TRUE(source.contains(QStringLiteral("summary.blockOnMatchQuality = matchingProducedNoUsableEdges")));
    EXPECT_TRUE(source.contains(QStringLiteral("匹配阶段已完成，但没有可用于空三的连接边")));
    EXPECT_TRUE(source.contains(QStringLiteral("不会自动重新跑完整匹配")));

    const int callbackStart = source.indexOf(
        QStringLiteral("bool autoFillMissing = false;"));
    ASSERT_GE(callbackStart, 0);
    const int launchStart = source.indexOf(
        QStringLiteral("controller->runUnifiedAerialTriangulation"),
        callbackStart);
    ASSERT_GT(launchStart, callbackStart);
    const QString callbackBody = source.mid(callbackStart, launchStart - callbackStart);

    EXPECT_TRUE(callbackBody.contains(QStringLiteral("reuseExistingMatches")));
    EXPECT_TRUE(callbackBody.contains(QStringLiteral("if (prereq.blockOnMatchQuality && reuseExistingMatches)")));
    EXPECT_TRUE(callbackBody.contains(QStringLiteral("atProgressFinished(false)")));
    EXPECT_TRUE(callbackBody.contains(QStringLiteral("return;")));
    EXPECT_LT(callbackBody.indexOf(QStringLiteral("if (prereq.blockOnMatchQuality && reuseExistingMatches)")),
              callbackBody.indexOf(QStringLiteral("if (!prereq.missingMessages.isEmpty())")));
}

TEST(AerialTriangulationWorkflowTest, PreflightEmitsPrerequisiteReportAndRecommendation)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("ReconstructionPrerequisiteReport")));
    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject prerequisiteReport")));
    EXPECT_TRUE(source.contains(QStringLiteral("prereq.prerequisiteReport")));
    EXPECT_TRUE(source.contains(QStringLiteral("空三上游数据就绪：复用已有匹配")));
    EXPECT_TRUE(source.contains(QStringLiteral("空三缺少部分匹配：只补齐缺失 pair")));
    EXPECT_TRUE(source.contains(QStringLiteral("创建连接点流程将自动提取特征并匹配")));
}

TEST(AerialTriangulationWorkflowTest, DoesNotAutoRematchWhenPrerequisitesArePresent)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int sparseStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(sparseStart, 0);
    const int launchStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"),
        sparseStart);
    ASSERT_GT(launchStart, sparseStart);
    const QString startBody = source.mid(sparseStart, launchStart - sparseStart);

    EXPECT_TRUE(startBody.contains(QStringLiteral("bool autoFillMissing = false")))
        << "AT should not prepare tie points when prerequisites are already present and reset alignment is disabled.";
    EXPECT_TRUE(startBody.contains(QStringLiteral("if (!prereq.missingMessages.isEmpty())")));
    EXPECT_TRUE(startBody.contains(QStringLiteral("autoFillMissing = true;")))
        << "Missing features or matches should be handled by the Create Tie Points pipeline without a prompt.";
    EXPECT_FALSE(startBody.contains(QStringLiteral("confirmAutoFillMissingSparseInputs")))
        << "A successful preflight must not be interpreted as a request to regenerate failed/skipped matches.";
}

TEST(AerialTriangulationWorkflowTest, WorkflowDialogStartsAerialTriangulationWorkflow)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int dialogStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openWorkflowAerialTriangulationDialog"));
    ASSERT_GE(dialogStart, 0);
    const int nextFunction = source.indexOf(QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"),
                                            dialogStart);
    ASSERT_GT(nextFunction, dialogStart);
    const QString dialogBody = source.mid(dialogStart, nextFunction - dialogStart);

    EXPECT_TRUE(dialogBody.contains(QStringLiteral("AerialTriangulationDialog")));
    EXPECT_TRUE(dialogBody.contains(QStringLiteral("dlg.exec()")));
    EXPECT_TRUE(dialogBody.contains(QStringLiteral("dlg.collectSettings()")));
    EXPECT_TRUE(dialogBody.contains(QStringLiteral("startAerialTriangulationWorkflow(settings)")));
}

TEST(AerialTriangulationWorkflowTest, StartDoesPrerequisiteAndSfmWorkOffGuiThread)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("runUnifiedAerialTriangulation")));
    EXPECT_TRUE(header.contains(QStringLiteral("static SparsePrerequisiteSummary summarizeSparsePrerequisites")));

    const int sparseStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(sparseStart, 0);
    const int launchStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"),
        sparseStart);
    ASSERT_GT(launchStart, sparseStart);
    const QString startBody = source.mid(sparseStart, launchStart - sparseStart);

    EXPECT_TRUE(startBody.contains(QStringLiteral("xjw::gui::tasks::runGuarded")));
    EXPECT_TRUE(startBody.contains(QStringLiteral("summarizeSparsePrerequisites(images")));
    EXPECT_FALSE(startBody.contains(QStringLiteral("QFutureWatcher<SparsePrerequisiteSummary>")));
    const int preflightLaunch = startBody.indexOf(QStringLiteral("xjw::gui::tasks::runGuarded"));
    ASSERT_GE(preflightLaunch, 0);
    EXPECT_FALSE(startBody.left(preflightLaunch).contains(QStringLiteral("summarizeSparsePrerequisites(")));

    const int workflowEnd = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openOverlapAnalysisDialog"),
        launchStart);
    ASSERT_GT(workflowEnd, launchStart);
    const QString launchBody = source.mid(launchStart, workflowEnd - launchStart);

    EXPECT_TRUE(launchBody.contains(QStringLiteral("xjw::gui::tasks::runGuarded")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("xjw::gui::tasks::postGuarded")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("AerialTriangulationWorkflow::run")));
    EXPECT_FALSE(launchBody.contains(QStringLiteral("QFutureWatcher<xjw::aerial_triangulation::AerialTriangulationResult>")));
    const int sfmLaunch = launchBody.indexOf(QStringLiteral("xjw::gui::tasks::runGuarded"));
    ASSERT_GE(sfmLaunch, 0);
    EXPECT_FALSE(launchBody.left(sfmLaunch).contains(QStringLiteral("AerialTriangulationWorkflow::run")));
}

TEST(AerialTriangulationWorkflowTest, SfmLaunchReusesGeneratedPairConstraints)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int launchStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"));
    ASSERT_GE(launchStart, 0);
    const int nextFunction = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openOverlapAnalysisDialog"),
        launchStart);
    ASSERT_GT(nextFunction, launchStart);
    const QString launchBody = source.mid(launchStart, nextFunction - launchStart);

    EXPECT_TRUE(source.contains(QStringLiteral("loadGeneratedPairConstraints")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("shouldUseStoredGeneratedPairConstraints(settings)")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("? loadGeneratedPairConstraints")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral(": QStringList()")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("照片序列预选已启用，SfM 跳过历史候选配对约束")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("workflowOptions.restrictPairs = true")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("workflowOptions.allowedPairs = allowedPairs")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("storedPairsStale")));
}

TEST(AerialTriangulationWorkflowTest, SfmLaunchUsesSelectedMatchPipeline)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int launchStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"));
    ASSERT_GE(launchStart, 0);
    const int nextFunction = source.indexOf(
        QStringLiteral("void MenuWorkflowController::openOverlapAnalysisDialog"),
        launchStart);
    ASSERT_GT(nextFunction, launchStart);
    const QString launchBody = source.mid(launchStart, nextFunction - launchStart);

    EXPECT_TRUE(launchBody.contains(QStringLiteral("workflowOptions.featureAlgorithm")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("workflowOptions.matchAlgorithm")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("settings.value(QStringLiteral(\"feature_algorithm\")")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("settings.value(QStringLiteral(\"match_algorithm\")")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("AerialTriangulationWorkflow::resolveConfig")));
}

TEST(MainWindowProgressTest, FeatureMatchProgressUsesTaskEstimateAndClampsDisplay)
{
    const QString mainWindowSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/TiePointWorkflowController.cpp"));
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(controllerSource.contains(QStringLiteral("estimatedPairCount")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("imageCount + estimatedPairCount")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("std::clamp(done")));
}

TEST(MainWindowCancelTest, FeatureMatchCancelGivesImmediateFeedbackAndEmitsSignal)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("sgCancelRequested")));
    EXPECT_TRUE(source.contains(QStringLiteral("正在取消特征匹配")));
    EXPECT_TRUE(source.contains(QStringLiteral("_sgTaskStatus")));
    EXPECT_TRUE(source.contains(QStringLiteral("TaskStatusWidget::cancelRequested")));
    EXPECT_TRUE(source.contains(QStringLiteral("emit sgCancelRequested()")));
}

TEST(BundleAdjustStatusBarTest, UsesAtProgressWidgetWithCancelableCoreOptimization)
{
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString projectManagerSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString bundleAdjustHeader = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.h"));
    const QString bundleAdjustSource = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.cpp"));
    const QString serviceSource = readProjectSourceFile(QStringLiteral("src/gui/project/services/BundleAdjustService.cpp"));
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(projectManagerSource.isEmpty());
    ASSERT_FALSE(bundleAdjustHeader.isEmpty());
    ASSERT_FALSE(bundleAdjustSource.isEmpty());
    ASSERT_FALSE(serviceSource.isEmpty());

    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("正在取消空三/光束法平差")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("std::make_shared<std::atomic<bool>>(false)")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("opts.baOpt.cancelFlag = cancelFlag")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("opts.baOpt.progressCallback")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("光束法平差优化中")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("emit self->atProgressFinished")));
    EXPECT_TRUE(bundleAdjustHeader.contains(QStringLiteral("std::shared_ptr<std::atomic<bool>> cancelFlag")));
    EXPECT_TRUE(bundleAdjustHeader.contains(QStringLiteral("progressCallback")));
    EXPECT_TRUE(bundleAdjustSource.contains(QStringLiteral("isCancelled(options)")));
    EXPECT_TRUE(serviceSource.contains(QStringLiteral("用户取消了光束法平差")));
}

TEST(AerialTriangulationCancelTest, CancelledWorkflowSkipsGuiThreadMetadataWriteback)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"));
    ASSERT_GE(start, 0);
    const int finish = source.indexOf(QStringLiteral("void MenuWorkflowController::openOverlapAnalysisDialog"), start);
    ASSERT_GT(finish, start);
    const QString block = source.mid(start, finish - start);

    const int callbackStart = block.indexOf(
        QStringLiteral("AerialTriangulationResult workflowResult) mutable"));
    ASSERT_GE(callbackStart, 0);
    const QString callback = block.mid(callbackStart);
    const int cancelCheck = callback.indexOf(QStringLiteral("if (wasCanceled)"));
    const int appendFeature = callback.indexOf(QStringLiteral("appendIpfindResult"));
    const int appendMatch = callback.indexOf(QStringLiteral("appendIpmatchResult"));
    const int setCameras = callback.indexOf(QStringLiteral("replaceImageCameras"));
    ASSERT_GE(cancelCheck, 0);
    ASSERT_GE(appendFeature, 0);
    ASSERT_GE(appendMatch, 0);
    ASSERT_GE(setCameras, 0);
    EXPECT_LT(cancelCheck, appendFeature)
        << "Cancellation must return before writing many feature records on the GUI thread.";
    EXPECT_LT(cancelCheck, appendMatch)
        << "Cancellation must return before writing many match records on the GUI thread.";
    EXPECT_LT(cancelCheck, setCameras)
        << "Cancellation must return before camera writeback and result dialogs.";
    EXPECT_TRUE(callback.mid(cancelCheck, appendFeature - cancelCheck)
                    .contains(QStringLiteral("emit pmGuard->atProgressFinished(false);")));
}

TEST(ForwardIntersectionCheckDialogTest, AutoModeAcceptsPythonLightGlueSidecarTokens)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/ForwardIntersectionCheckDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("feature0_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("feature1_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("sp0_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("sp1_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("image0_name")));
    EXPECT_TRUE(source.contains(QStringLiteral("image1_name")));
}

TEST(FeatureVisualizationSettingsTest, PersistsAndRestoresActiveFeatureSuffix)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("feature_suffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("setActiveFeatureSuffix")));
}

TEST(FeatureVisualizationSettingsTest, DefaultsToOnePixelCrossMarker)
{
    const QString rendererHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString overlaySource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerOverlayItems.cpp"));
    const QString uiDefaults = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectUiConfigManager.cpp"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.cpp"));
    const QString dialogUi = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.ui"));
    ASSERT_FALSE(rendererHeader.isEmpty());
    ASSERT_FALSE(overlaySource.isEmpty());
    ASSERT_FALSE(uiDefaults.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());
    ASSERT_FALSE(dialogUi.isEmpty());

    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("int pointSize = 1")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("markerShape = QStringLiteral(\"cross\")")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("const double crossRadius")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("crossPen.setWidthF(1.0)")));

    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("featureDisplay[\"pointSize\"]         = 1")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("featureDisplay[\"markerShape\"]       = QStringLiteral(\"cross\")")));

    EXPECT_TRUE(dialogSource.contains(QStringLiteral("_markerShapeCombo->setCurrentIndex(2)")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("_pointSizeSpin->setValue(1)")));
    EXPECT_TRUE(dialogUi.contains(QStringLiteral("<number>1</number>")));
}

TEST(FeatureVisualizationSettingsTest, DefaultsPointColorToBlue)
{
    const QString rendererHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString uiDefaults = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectUiConfigManager.cpp"));
    const QString dialogHeader = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.h"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/FeaturePointVisualizationDialog.cpp"));
    ASSERT_FALSE(rendererHeader.isEmpty());
    ASSERT_FALSE(uiDefaults.isEmpty());
    ASSERT_FALSE(dialogHeader.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());

    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("QColor pointColor = QColor(0, 120, 255)")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor[\"r\"] = 0")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor[\"g\"] = 120")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor[\"b\"] = 255")));
    EXPECT_TRUE(dialogHeader.contains(QStringLiteral("QColor _pointColor{0, 120, 255}")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("_pointColor = QColor(0, 120, 255)")));
    EXPECT_FALSE(dialogSource.contains(QStringLiteral("点颜色黄")));
}

TEST(FeatureVisualizationSettingsTest, ProjectOpenRestoresFeatureSuffixEvenWhenUiSettingsAreEmpty)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int applyStart = source.indexOf(QStringLiteral("void MainWindow::applyUiSettings"));
    ASSERT_GE(applyStart, 0);
    const int applyIndex = source.indexOf(QStringLiteral("applySavedFeatureDisplayOptions(ui)"), applyStart);
    const int normalizeIndex = source.indexOf(QStringLiteral("QJsonObject settings = ui;"), applyStart);
    ASSERT_GE(applyIndex, 0);
    ASSERT_GE(normalizeIndex, 0);
    EXPECT_LT(applyIndex, normalizeIndex);
}

TEST(FeatureVisualizationSettingsTest, VisualizationRestoreInfersSuffixFromExistingFeatureFiles)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("inferPreferredFeatureSuffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("inferredSuffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("setActiveFeatureSuffix(inferredSuffix)")));
}

TEST(FeatureVisualizationSettingsTest, SavedSuffixIsUsedOnlyWhenProjectContainsThatSuffix)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("projectHasFeatureSuffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("savedSuffixUsable")));
}

TEST(MapProjectDialogTest, DefaultsToOneClickDomEngineeringSettings)
{
    MapProjectDialog dialog;
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectRoot = tempDir.path();
    const QString demPath = QDir(projectRoot).filePath(QStringLiteral("assets/dem/relative_dem/dem.tif"));
    const QString expectedOutput = QDir(projectRoot).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(demPath).absolutePath()));
    QFile demFile(demPath);
    ASSERT_TRUE(demFile.open(QIODevice::WriteOnly));
    demFile.write("dem");
    demFile.close();

    const QStringList images{
        QDir(projectRoot).filePath(QStringLiteral("assets/img/1.png")),
        QDir(projectRoot).filePath(QStringLiteral("assets/img/2.png"))
    };

    dialog.setAvailableImages(images);
    dialog.setProjectRoot(projectRoot);
    dialog.setDefaultDemPath(demPath);

    auto *imageList = dialog.findChild<QListWidget *>(QStringLiteral("m_imageList"));
    auto *demEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_demEdit"));
    auto *outputEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_outputEdit"));
    auto *resolutionSpin = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("m_resolutionSpin"));
    auto *runButton = dialog.findChild<QPushButton *>(QStringLiteral("runBtn"));

    ASSERT_NE(imageList, nullptr);
    ASSERT_NE(demEdit, nullptr);
    ASSERT_NE(outputEdit, nullptr);
    ASSERT_NE(resolutionSpin, nullptr);
    ASSERT_NE(runButton, nullptr);

    ASSERT_EQ(imageList->count(), images.size());
    for (int i = 0; i < imageList->count(); ++i)
    {
        EXPECT_EQ(imageList->item(i)->checkState(), Qt::Checked);
    }
    EXPECT_EQ(demEdit->text(), demPath);
    EXPECT_EQ(outputEdit->text(), expectedOutput);
    EXPECT_DOUBLE_EQ(resolutionSpin->value(), 0.0);
    EXPECT_TRUE(resolutionSpin->specialValueText().contains(QStringLiteral("自动")));
    EXPECT_TRUE(runButton->text().contains(QStringLiteral("一键")));
    EXPECT_TRUE(runButton->text().contains(QStringLiteral("DOM")));

    bool emitted = false;
    QStringList emittedImages;
    QString emittedDem;
    QString emittedOutput;
    double emittedResolution = -1.0;
    QObject::connect(&dialog, &MapProjectDialog::requestRunMapProject, &dialog,
                     [&](const QStringList &runImages,
                         const QString &runDem,
                         const QString &runOutput,
                         double runResolution)
                     {
                         emitted = true;
                         emittedImages = runImages;
                         emittedDem = runDem;
                         emittedOutput = runOutput;
                         emittedResolution = runResolution;
                     });
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "onRun", Qt::DirectConnection));

    EXPECT_TRUE(emitted);
    EXPECT_EQ(emittedImages, images);
    EXPECT_EQ(emittedDem, demPath);
    EXPECT_EQ(emittedOutput, expectedOutput);
    EXPECT_DOUBLE_EQ(emittedResolution, 0.0);
}

TEST(MapProjectDialogTest, KeepsLatestDemDefaultWhenSavedDemIsMissing)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectRoot = tempDir.path();
    const QString latestDemPath = QDir(projectRoot).filePath(QStringLiteral("assets/dem/latest/dem.tif"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(latestDemPath).absolutePath()));
    QFile latestDemFile(latestDemPath);
    ASSERT_TRUE(latestDemFile.open(QIODevice::WriteOnly));
    latestDemFile.write("dem");
    latestDemFile.close();

    MapProjectDialog dialog;
    dialog.setProjectRoot(projectRoot);
    dialog.setDefaultDemPath(latestDemPath);

    auto *demEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_demEdit"));
    ASSERT_NE(demEdit, nullptr);
    ASSERT_EQ(demEdit->text(), latestDemPath);

    QJsonObject savedSettings;
    savedSettings[QStringLiteral("dem_path")] =
        QDir(projectRoot).filePath(QStringLiteral("assets/dem/old/missing_dem.tif"));
    savedSettings[QStringLiteral("output_path")] =
        QDir(projectRoot).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
    dialog.applySettings(savedSettings);

    EXPECT_EQ(demEdit->text(), latestDemPath);
}

TEST(MapProjectDialogTest, ValidatesDemFileExistsBeforeRunning)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/MapProjectDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QFileInfo demInfo")));
    EXPECT_TRUE(source.contains(QStringLiteral("demInfo.exists()")));
    EXPECT_TRUE(source.contains(QStringLiteral("不是有效的 DEM 文件")));
}

TEST(CreateDemDialogTest, UiAdvertisesOneClickDemWorkflow)
{
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/dialogs/reconstruction/CreateDemDialog.ui"));
    ASSERT_FALSE(ui.isEmpty());

    EXPECT_TRUE(ui.contains(QStringLiteral("自动模式")));
    EXPECT_TRUE(ui.contains(QStringLiteral("手动模式")));
    EXPECT_TRUE(ui.contains(QStringLiteral("选择 2 张立体影像")));
    EXPECT_TRUE(ui.contains(QStringLiteral("已有密集点云")));
    EXPECT_TRUE(ui.contains(QStringLiteral("m_stageLabel")));
    EXPECT_TRUE(ui.contains(QStringLiteral("m_progressBar")));
    EXPECT_TRUE(ui.contains(QStringLiteral("m_runBtn")));
    EXPECT_TRUE(ui.contains(QStringLiteral("一键生成 DEM")));
}

TEST(CreateDemDialogTest, OpenCreateDemDialogRequiresProjectManagerBeforeShowingManualRunDialog)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int openIndex = source.indexOf(QStringLiteral("void MenuWorkflowController::openCreateDemDialog()"));
    const int newDialogIndex = source.indexOf(QStringLiteral("new CreateDemDialog"), openIndex);
    const int warningIndex = source.indexOf(QStringLiteral("请先打开项目"), openIndex);
    ASSERT_GE(openIndex, 0);
    ASSERT_GE(newDialogIndex, 0);
    ASSERT_GE(warningIndex, 0);
    EXPECT_LT(warningIndex, newDialogIndex);
}

TEST(GuiMainTest, WindowsConsoleUsesUtf8)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("SetConsoleOutputCP(CP_UTF8)")));
    EXPECT_TRUE(source.contains(QStringLiteral("SetConsoleCP(CP_UTF8)")));
    EXPECT_TRUE(source.contains(QStringLiteral("configureConsoleEncoding()")));
}

TEST(FeatureNamingCleanupTest, LightGlueDocumentationDoesNotAdvertiseSuperPointAsLegacy)
{
    const QString header =
        readProjectSourceFile(QStringLiteral("src/core/feature_match/lightglue/LightGlueMatcher.h"));
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("SuperPoint")))
        << "The algorithm remains supported, but it should be described as an explicit compatibility path.";
    EXPECT_FALSE(header.contains(QStringLiteral("SuperPoint legacy")))
        << "Do not present the supported SuperPoint fallback as a legacy GUI/interface path.";
}

TEST(FeatureNamingCleanupTest, FeatureExtractorReadmeUsesFeatureDataForSharedOutput)
{
    const QString readme =
        readProjectSourceFile(QStringLiteral("src/core/feature_extractors/README.md"));
    ASSERT_FALSE(readme.isEmpty());

    EXPECT_TRUE(readme.contains(QStringLiteral("FeatureData")))
        << "Feature extractor docs should describe the current shared feature container.";
    EXPECT_FALSE(readme.contains(QStringLiteral("SuperPointOutput")))
        << "Do not describe the shared output abstraction with the old SuperPoint-specific type name.";
}

TEST(FeatureNamingCleanupTest, ProjectManagerDoesNotIncludeLegacyTorchAlgorithmHeaders)
{
    const QString managerSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(managerSource.isEmpty());

    EXPECT_FALSE(managerSource.contains(QStringLiteral("#include \"compat/QtTorchMacroGuard.h\"")))
        << "ProjectManager should not include Torch headers directly; keep LibTorch warning guards in narrow workers.";
    EXPECT_FALSE(managerSource.contains(QStringLiteral("#include \"SuperPoint.h\"")));
    EXPECT_FALSE(managerSource.contains(QStringLiteral("#include \"SuperGlueMatcher.h\"")));
    EXPECT_FALSE(managerSource.contains(QStringLiteral("#include \"SuperGlueMatchIO.h\"")));
    EXPECT_FALSE(managerSource.contains(QStringLiteral("#include \"FeatureFileIO.h\"")));
}

TEST(FeatureNamingCleanupTest, TorchFeatureWrappersSuppressLibTorchC4267Warnings)
{
    const QStringList files = {
        QStringLiteral("src/core/feature_extractors/ExtractorFactory.cpp"),
        QStringLiteral("src/core/feature_extractors/disk/DiskExtractor.cpp"),
        QStringLiteral("src/core/feature_extractors/aliked/AlikedExtractor.cpp"),
        QStringLiteral("src/core/feature_match/loftr/LoFTRMatcher.cpp"),
    };
    for (const QString &path : files)
    {
        const QString source = readProjectSourceFile(path);
        ASSERT_FALSE(source.isEmpty()) << qPrintable(path);
        EXPECT_TRUE(source.contains(QStringLiteral("#pragma warning(push)"))) << qPrintable(path);
        EXPECT_TRUE(source.contains(QStringLiteral("#pragma warning(disable: 4267)")))
            << qPrintable(path + QStringLiteral(": LibTorch emits MSVC C4267 from external templates."));
        EXPECT_TRUE(source.contains(QStringLiteral("#pragma warning(pop)"))) << qPrintable(path);
    }
}

TEST(FeatureNamingCleanupTest, MatchPhotosTaskGuardsTorchIncludesAfterQtHeaders)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/task/MatchPhotosTask.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int qtHeaderIndex = source.indexOf(QStringLiteral("#include \"MatchPhotosTask.h\""));
    const int torchHeaderIndex = source.indexOf(QStringLiteral("#include \"FeatureFileIO.h\""));
    ASSERT_GE(qtHeaderIndex, 0);
    ASSERT_GT(torchHeaderIndex, qtHeaderIndex);

    const QString preTorchBlock = source.mid(qtHeaderIndex, torchHeaderIndex - qtHeaderIndex);
    EXPECT_TRUE(preTorchBlock.contains(QStringLiteral("#undef slots")));
    EXPECT_TRUE(preTorchBlock.contains(QStringLiteral("#undef signals")));
    EXPECT_TRUE(preTorchBlock.contains(QStringLiteral("#undef emit")));
    EXPECT_TRUE(preTorchBlock.contains(QStringLiteral("PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS")));

    const int restoreIndex = source.indexOf(QStringLiteral("#define slots Q_SLOTS"), torchHeaderIndex);
    const int nextQtIncludeIndex = source.indexOf(QStringLiteral("#include <QFileInfo>"), torchHeaderIndex);
    ASSERT_GE(restoreIndex, 0);
    ASSERT_GT(nextQtIncludeIndex, torchHeaderIndex);
    EXPECT_LT(restoreIndex, nextQtIncludeIndex);
}

TEST(FeatureNamingCleanupTest, MatcherFactorySuppressesLibTorchC4267Warnings)
{
    const QString path = QStringLiteral("src/core/feature_match/MatcherFactory.cpp");
    const QString source = readProjectSourceFile(path);
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(source.contains(QStringLiteral("#pragma warning(push)"))) << qPrintable(path);
    EXPECT_TRUE(source.contains(QStringLiteral("#pragma warning(disable: 4267)")))
        << qPrintable(path + QStringLiteral(": LibTorch emits MSVC C4267 from external templates."));
    EXPECT_TRUE(source.contains(QStringLiteral("#pragma warning(pop)"))) << qPrintable(path);
}

TEST(FeatureNamingCleanupTest, GuiTestsDoNotCompileObsoleteCompatibilityTranslationUnits)
{
    const QString testsCMake = readProjectSourceFile(QStringLiteral("tests/CMakeLists.txt"));
    const QString projectBaInputBuilderCompat =
        readProjectSourceFile(QStringLiteral("src/gui/project/services/ProjectBaInputBuilder.cpp"));
    const QString projectTriangulationCompat =
        readProjectSourceFile(QStringLiteral("src/gui/project/services/ProjectTriangulationService.cpp"));
    ASSERT_FALSE(testsCMake.isEmpty());

    EXPECT_FALSE(testsCMake.contains(QStringLiteral("ProjectBaInputBuilder.cpp")))
        << "The GUI BA input compatibility wrapper is header-only; tests should link the core implementation directly.";
    EXPECT_FALSE(testsCMake.contains(QStringLiteral("ProjectTriangulationService.cpp")))
        << "The GUI triangulation compatibility wrapper is header-only; tests should link the core implementation directly.";
    EXPECT_TRUE(projectBaInputBuilderCompat.isEmpty())
        << "Remove empty compatibility translation units once they are not part of any target.";
    EXPECT_TRUE(projectTriangulationCompat.isEmpty())
        << "Remove empty compatibility translation units once they are not part of any target.";
}

TEST(FeatureNamingCleanupTest, GuiSfmCallersUseCoreServicesDirectly)
{
    const QString testsSource = readProjectSourceFile(QStringLiteral("tests/test_gui_project_utils.cpp"));
    const QString guiSources = readProjectSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    const QString bundleAdjustHeader =
        readProjectSourceFile(QStringLiteral("src/gui/project/support/ProjectBundleAdjustExecution.h"));
    const QString projectManagerSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString sparseManagerHeader =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectSparseReconstructionManager.h"));
    const QString sparseManagerSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectSparseReconstructionManager.cpp"));
    ASSERT_FALSE(testsSource.isEmpty());
    ASSERT_FALSE(guiSources.isEmpty());
    ASSERT_FALSE(bundleAdjustHeader.isEmpty());
    ASSERT_FALSE(projectManagerSource.isEmpty());
    ASSERT_FALSE(sparseManagerHeader.isEmpty());
    ASSERT_FALSE(sparseManagerSource.isEmpty());

    const int includeBlockEnd = testsSource.indexOf(QStringLiteral("namespace"));
    const QString testsIncludeBlock = includeBlockEnd > 0 ? testsSource.left(includeBlockEnd) : testsSource;

    for (const QString &source : {testsIncludeBlock,
                                  guiSources,
                                  bundleAdjustHeader,
                                  projectManagerSource,
                                  sparseManagerHeader,
                                  sparseManagerSource})
    {
        EXPECT_FALSE(source.contains(QStringLiteral("ProjectBaInputBuilder")))
            << "GUI code should use core::project BaInputBuilder directly instead of a GUI compatibility wrapper.";
        EXPECT_FALSE(source.contains(QStringLiteral("ProjectTriangulationService")))
            << "GUI code should use core::project TriangulationService directly instead of a GUI compatibility wrapper.";
    }

    EXPECT_TRUE(bundleAdjustHeader.contains(QStringLiteral("#include \"project/BaInputBuilder.h\"")));
    EXPECT_TRUE(sparseManagerHeader.contains(QStringLiteral("#include \"TriangulationService.h\"")));
    EXPECT_TRUE(sparseManagerSource.contains(QStringLiteral("xjw::core::project::TriangulationService::run")));
}

TEST(MainWindowFeatureRefreshTest, BatchFeatureAppendDoesNotSynchronouslyReloadNonCurrentImages)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int connectIndex = source.indexOf(QStringLiteral("ipfindResultAppended"));
    ASSERT_GE(connectIndex, 0);
    const int blockEnd = source.indexOf(QStringLiteral("if (_config)"), connectIndex);
    ASSERT_GE(blockEnd, connectIndex);
    const QString block = source.mid(connectIndex, blockEnd - connectIndex);

    EXPECT_TRUE(block.contains(QStringLiteral("currentImagePath()")));
    EXPECT_TRUE(block.contains(QStringLiteral("isCurrentImage")));
    EXPECT_TRUE(block.contains(QStringLiteral("reloadInterestPoints(imagePath)")));
    EXPECT_FALSE(block.contains(QStringLiteral("immediateReloadInterestPoints(imagePath)")));
}

TEST(CanvasWidgetResponsivenessTest, ImageSwitchUsesBackgroundLoadAndIgnoresStaleResults)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    const QString rendererHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(rendererHeader.isEmpty());
    ASSERT_FALSE(rendererSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QFutureWatcher<QImage> *_imageWatcher")));
    EXPECT_TRUE(source.contains(QStringLiteral("QtConcurrent::run([pathCopy, projectPath]")));
    EXPECT_TRUE(source.contains(QStringLiteral("LayerRenderer::loadImageForDisplay(pathCopy, projectPath)")));
    EXPECT_TRUE(source.contains(QStringLiteral("QDir::cleanPath(loadedPath) != QDir::cleanPath(self->_currentImagePath)")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("static QImage loadImageForDisplay")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("bool addImageLayer(const QImage &image, int z = 0)")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("QPixmap::fromImage(image)")));
}

TEST(CanvasWidgetResponsivenessTest, LayerRendererDelegatesImageLoadingToDedicatedLoader)
{
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    const QString loaderHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerImageLoader.h"));
    const QString loaderSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerImageLoader.cpp"));
    ASSERT_FALSE(rendererSource.isEmpty());
    ASSERT_FALSE(loaderHeader.isEmpty());
    ASSERT_FALSE(loaderSource.isEmpty());

    EXPECT_TRUE(rendererSource.contains(QStringLiteral("#include \"LayerImageLoader.h\"")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("return xjw::gui::views::loadImageForDisplay(path, plascanPath)")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("convertTo8BitGeoTiff_GDAL")))
        << "Image conversion/cache logic should stay out of the scene renderer.";
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("needsConvertTo8Bit_GDAL")))
        << "Image conversion/cache logic should stay out of the scene renderer.";
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("loadImageWithOpenCvByteDecode")))
        << "Image decoding fallback should stay in the image loader.";

    EXPECT_TRUE(loaderHeader.contains(QStringLiteral("QImage loadImageForDisplay")));
    EXPECT_TRUE(loaderSource.contains(QStringLiteral("convertTo8BitGeoTiff_GDAL")));
    EXPECT_TRUE(loaderSource.contains(QStringLiteral("loadImageWithOpenCvByteDecode")));
}

TEST(CanvasWidgetResponsivenessTest, LayerRendererDelegatesOverlayDrawingToDedicatedItems)
{
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    const QString overlayHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerOverlayItems.h"));
    const QString overlaySource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerOverlayItems.cpp"));
    ASSERT_FALSE(rendererSource.isEmpty());
    ASSERT_FALSE(overlayHeader.isEmpty());
    ASSERT_FALSE(overlaySource.isEmpty());

    EXPECT_TRUE(rendererSource.contains(QStringLiteral("#include \"LayerOverlayItems.h\"")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("createFeatureOverlayItem(keypoints, _featureOpts, _imageBounds)")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("createMatchOverlayItems(ptsA, ptsB, _matchOpts, bOffsetX)")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("class BatchedFeatureOverlayItem")))
        << "Feature overlay item implementation should stay out of LayerRenderer.";
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("void drawKeypoint(")))
        << "Keypoint painting details should stay out of LayerRenderer.";
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("addEllipse(a.x()-3")))
        << "Match endpoint item construction should stay out of LayerRenderer.";
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("addLine(a.x(), a.y()")))
        << "Match line item construction should stay out of LayerRenderer.";

    EXPECT_TRUE(overlayHeader.contains(QStringLiteral("createFeatureOverlayItem")));
    EXPECT_TRUE(overlayHeader.contains(QStringLiteral("createMatchOverlayItems")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("class BatchedFeatureOverlayItem")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("new QGraphicsEllipseItem(a.x() - 3")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("new QGraphicsLineItem(a.x(), a.y()")));
}

TEST(CanvasWidgetResponsivenessTest, LayerRendererDelegatesFeatureFileLoadingToDedicatedLoader)
{
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    const QString featureLoaderHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerFeatureLoader.h"));
    const QString featureLoaderSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerFeatureLoader.cpp"));
    ASSERT_FALSE(rendererSource.isEmpty());
    ASSERT_FALSE(featureLoaderHeader.isEmpty());
    ASSERT_FALSE(featureLoaderSource.isEmpty());

    EXPECT_TRUE(rendererSource.contains(QStringLiteral("#include \"LayerFeatureLoader.h\"")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("loadFeatureKeypointsForImage(_currentProjectPath, imagePath)")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("#include \"FeatureOutput.h\"")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("#include \"FeatureFileIO.h\"")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("xjw::common::project::ProjectIO::findFeatureForImage")))
        << "Feature sidecar lookup should stay out of the scene renderer.";
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("FeatureFileIO::read")))
        << "Feature file decoding should stay out of the scene renderer.";

    EXPECT_TRUE(featureLoaderHeader.contains(QStringLiteral("loadFeatureKeypointsFromFile")));
    EXPECT_TRUE(featureLoaderHeader.contains(QStringLiteral("loadFeatureKeypointsForImage")));
    EXPECT_TRUE(featureLoaderSource.contains(QStringLiteral("xjw::common::project::ProjectIO::findFeatureForImage")));
    EXPECT_TRUE(featureLoaderSource.contains(QStringLiteral("loadFeatureKeypointsFromFile(featurePath)")));
    EXPECT_TRUE(featureLoaderSource.contains(QStringLiteral("FeatureFileIO::read")));
    EXPECT_TRUE(featureLoaderSource.contains(QStringLiteral("output.keypoints[i].response = output.scores[i]")));
    EXPECT_TRUE(featureLoaderSource.contains(QStringLiteral("#pragma warning(disable: 4267)")))
        << "LibTorch emits MSVC C4267 from external templates; keep it local to the feature loader.";
}

TEST(CanvasWidgetResponsivenessTest, LayerRendererDelegatesStitchedPairDebugOutput)
{
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    const QString debugHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerStitchedDebug.h"));
    const QString debugSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerStitchedDebug.cpp"));
    ASSERT_FALSE(rendererSource.isEmpty());
    ASSERT_FALSE(debugHeader.isEmpty());
    ASSERT_FALSE(debugSource.isEmpty());

    EXPECT_TRUE(rendererSource.contains(QStringLiteral("#include \"LayerStitchedDebug.h\"")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("recordStitchedImagePairDebug(")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("QCryptographicHash")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("hexSha1")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("wrote debug stitched image")));
    EXPECT_FALSE(rendererSource.contains(QStringLiteral("qgraphicsitem_cast")));

    EXPECT_TRUE(debugHeader.contains(QStringLiteral("recordStitchedImagePairDebug")));
    EXPECT_TRUE(debugSource.contains(QStringLiteral("QCryptographicHash::hash")));
    EXPECT_TRUE(debugSource.contains(QStringLiteral("wrote debug stitched image")));
    EXPECT_TRUE(debugSource.contains(QStringLiteral("qgraphicsitem_cast<QGraphicsPixmapItem")));
}

TEST(CanvasWidgetResponsivenessTest, StaleFeatureLoadsDoNotPaintOverCurrentImage)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int finishedIndex =
        source.indexOf(QStringLiteral("xjw::gui::tasks::runGuarded"), source.indexOf(
                           QStringLiteral("void CanvasWidget::startSpLoadForImage")));
    ASSERT_GE(finishedIndex, 0);
    const QString finishedBlock = source.mid(finishedIndex, 5200);

    EXPECT_TRUE(finishedBlock.contains(QStringLiteral("generation != self->_featureLoadGeneration")))
        << "Late completions must be dropped after another image/suffix request starts.";
    EXPECT_TRUE(finishedBlock.contains(
        QStringLiteral("QDir::cleanPath(imagePathCopy) == QDir::cleanPath(self->_currentImagePath)")));
    EXPECT_TRUE(finishedBlock.contains(QStringLiteral("if (isCurrentImage && self->_layerRenderer)")));
}

TEST(CanvasWidgetResponsivenessTest, FeatureLoadEstimatesOrientationOnlyWhenDisplayed)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int startIndex = source.indexOf(QStringLiteral("void CanvasWidget::startSpLoadForImage"));
    ASSERT_GE(startIndex, 0);
    const int readImageIndex = source.indexOf(
        QStringLiteral("xjw::common::io::readImage(imagePathCopy"), startIndex);
    ASSERT_GE(readImageIndex, startIndex);
    const QString loadBlock = source.mid(startIndex, readImageIndex - startIndex + 600);

    EXPECT_TRUE(loadBlock.contains(QStringLiteral("const QString projectPath = property(\"currentProjectPath\").toString()")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("const bool shouldEstimateOrientation = _currentFeatureOpts.showOrientation")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("imagePathCopy")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("activeSuffix")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("projectPath")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("shouldEstimateOrientation")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("if (shouldEstimateOrientation)")));
}

TEST(TriangulationServiceTest, ExportsInitialSparseCloud)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString matchesDir = xjw::common::project::ProjectIO::ipmatchOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(matchesDir));

    const QString image0Path = QDir(tempDir.path()).filePath(QStringLiteral("1.jpg"));
    const QString image1Path = QDir(tempDir.path()).filePath(QStringLiteral("2.jpg"));

    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);

    const std::vector<std::array<double, 3>> points = {
        {3.0, -1.0, 35.0},
        {4.0,  0.5, 40.0},
        {5.0,  1.2, 45.0}
    };

    QJsonArray matchedPoints0;
    QJsonArray matchedPoints1;
    for (const std::array<double, 3> &point : points)
    {
        double u0 = 0.0;
        double v0 = 0.0;
        double u1 = 0.0;
        double v1 = 0.0;
        ASSERT_TRUE(projectPoint(camera0, point, &u0, &v0));
        ASSERT_TRUE(projectPoint(camera1, point, &u1, &v1));

        matchedPoints0.append(QJsonArray{u0, v0});
        matchedPoints1.append(QJsonArray{u1, v1});
    }

    const QString matchPath = QDir(matchesDir).filePath(QStringLiteral("1__2.match"));
    QFile matchFile(matchPath);
    ASSERT_TRUE(matchFile.open(QIODevice::WriteOnly));
    matchFile.write("dummy");
    matchFile.close();

    QFile sidecarFile(matchPath + QStringLiteral(".json"));
    ASSERT_TRUE(sidecarFile.open(QIODevice::WriteOnly));
    const QJsonObject sidecarObject{
        {QStringLiteral("image0_path"), image0Path},
        {QStringLiteral("image1_path"), image1Path},
        {QStringLiteral("matched_points0"), matchedPoints0},
        {QStringLiteral("matched_points1"), matchedPoints1}
    };
    sidecarFile.write(QJsonDocument(sidecarObject).toJson());
    sidecarFile.close();

    QJsonArray images;
    images.append(buildImageEntry(image0Path, camera0));
    images.append(buildImageEntry(image1Path, camera1));

    QJsonArray ipmatchResults;
    ipmatchResults.append(QJsonObject{{QStringLiteral("image0"), image0Path},
                                      {QStringLiteral("image1"), image1Path},
                                      {QStringLiteral("output"), matchPath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    meta[QStringLiteral("ipmatch_results")] = ipmatchResults;

    xjw::core::project::TriangulationServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;
    options.minObservations = 2;
    options.ignoreTwoViewTracks = false;
    options.minTrackLength = 2;

    const auto result = xjw::core::project::TriangulationService::run(
        meta, QStringList{image0Path, image1Path}, options);

    EXPECT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_GT(result.exportedPointCount, 0);
    EXPECT_TRUE(QFileInfo::exists(result.sparseCloudPath));

    const QJsonObject quality = result.resultJson.value(QStringLiteral("quality")).toObject();
    EXPECT_EQ(result.resultJson.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindPairwisePreview);
    EXPECT_EQ(quality.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindPairwisePreview);
    EXPECT_EQ(quality.value(QStringLiteral("point_count")).toInt(), result.exportedPointCount);
    EXPECT_TRUE(xjw::gui::project::isPairwisePreviewSparseResult(result.resultJson));
    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(result.resultJson));
}

TEST(TriangulationServiceTest, UsesSidecarV2IndicesForMultiViewTracks)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString matchesDir = xjw::common::project::ProjectIO::ipmatchOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(matchesDir));

    const QString image0Path = QDir(tempDir.path()).filePath(QStringLiteral("1.jpg"));
    const QString image1Path = QDir(tempDir.path()).filePath(QStringLiteral("2.jpg"));
    const QString image2Path = QDir(tempDir.path()).filePath(QStringLiteral("3.jpg"));

    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::Camera camera2 = makeCamera(16.0, 0.0, 0.0);
    const std::array<double, 3> point = {6.0, 0.5, 36.0};

    double u0 = 0.0;
    double v0 = 0.0;
    double u1 = 0.0;
    double v1 = 0.0;
    double u2 = 0.0;
    double v2 = 0.0;
    ASSERT_TRUE(projectPoint(camera0, point, &u0, &v0));
    ASSERT_TRUE(projectPoint(camera1, point, &u1, &v1));
    ASSERT_TRUE(projectPoint(camera2, point, &u2, &v2));

    auto writeSidecar = [](const QString &path,
                           const QString &imageA,
                           const QString &imageB,
                           const QPointF &uvA,
                           const QPointF &uvB,
                           int featureA,
                           int featureB) {
        QFile matchFile(path);
        ASSERT_TRUE(matchFile.open(QIODevice::WriteOnly));
        matchFile.write("dummy");
        matchFile.close();

        QFile sidecarFile(path + QStringLiteral(".json"));
        ASSERT_TRUE(sidecarFile.open(QIODevice::WriteOnly));
        QJsonArray matchedPoints0;
        QJsonArray matchedPoints1;
        QJsonArray matchedIndices0;
        QJsonArray matchedIndices1;
        matchedPoints0.append(QJsonArray{uvA.x(), uvA.y()});
        matchedPoints1.append(QJsonArray{uvB.x(), uvB.y()});
        matchedIndices0.append(featureA);
        matchedIndices1.append(featureB);
        const QJsonObject sidecarObject{
            {QStringLiteral("feature_format_version"), 2},
            {QStringLiteral("image0_path"), imageA},
            {QStringLiteral("image1_path"), imageB},
            {QStringLiteral("matched_points0"), matchedPoints0},
            {QStringLiteral("matched_points1"), matchedPoints1},
            {QStringLiteral("matched_indices0"), matchedIndices0},
            {QStringLiteral("matched_indices1"), matchedIndices1}
        };
        sidecarFile.write(QJsonDocument(sidecarObject).toJson());
        sidecarFile.close();
    };

    const QString match01Path = QDir(matchesDir).filePath(QStringLiteral("1__2.match"));
    const QString match02Path = QDir(matchesDir).filePath(QStringLiteral("1__3.match"));
    writeSidecar(match01Path, image0Path, image1Path, QPointF(u0, v0), QPointF(u1, v1), 7, 13);
    writeSidecar(match02Path, image0Path, image2Path, QPointF(u0, v0), QPointF(u2, v2), 7, 29);

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        buildImageEntry(image0Path, camera0),
        buildImageEntry(image1Path, camera1),
        buildImageEntry(image2Path, camera2)
    };
    meta[QStringLiteral("ipmatch_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("image0"), image0Path},
                    {QStringLiteral("image1"), image1Path},
                    {QStringLiteral("output"), match01Path}},
        QJsonObject{{QStringLiteral("image0"), image0Path},
                    {QStringLiteral("image1"), image2Path},
                    {QStringLiteral("output"), match02Path}}
    };

    xjw::core::project::TriangulationServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;
    options.minObservations = 3;
    options.ignoreTwoViewTracks = true;
    options.minTrackLength = 3;

    const auto result = xjw::core::project::TriangulationService::run(
        meta, QStringList{image0Path, image1Path, image2Path}, options);

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    ASSERT_EQ(result.exportedPointCount, 1);

    const QJsonArray points = result.resultJson.value(QStringLiteral("points")).toArray();
    ASSERT_EQ(points.size(), 1);
    EXPECT_EQ(points.at(0).toObject().value(QStringLiteral("track_len")).toInt(), 3);
}

TEST(ProjectTriangulationUiTest, FinalizeTriangulationStoresPreviewQualityMetadata)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectSparseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("mergeSparseQualityIntoRecord")));
    EXPECT_TRUE(source.contains(QStringLiteral("result.resultJson.value(QStringLiteral(\"quality\"))")));
    EXPECT_TRUE(source.contains(QStringLiteral("两视预览云")));
}

TEST(ProjectTriangulationUiTest, SparseManagerLongTasksUseGuardedRunner)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectSparseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int triangulationStart = source.indexOf(
        QStringLiteral("void ProjectSparseReconstructionManager::startTriangulationAsync"));
    ASSERT_GE(triangulationStart, 0);
    const int workflowStart = source.indexOf(
        QStringLiteral("void ProjectSparseReconstructionManager::startSparsePointWorkflow"),
        triangulationStart);
    ASSERT_GT(workflowStart, triangulationStart);
    const QString triangulationBody = source.mid(triangulationStart, workflowStart - triangulationStart);

    EXPECT_TRUE(triangulationBody.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "Two-view preview triangulation should not launch open-coded background work.";
    EXPECT_TRUE(triangulationBody.contains(QStringLiteral("xjw::core::project::TriangulationService::run")))
        << "The guarded worker should still run the triangulation service off the GUI thread.";
    EXPECT_TRUE(triangulationBody.contains(QStringLiteral("QPointer<ProjectManager> ownerGuard(_owner)")))
        << "Triangulation completion must guard the owning ProjectManager.";
    EXPECT_TRUE(triangulationBody.contains(
        QStringLiteral("const QString projectPath = _owner ? _owner->currentProjectPath() : QString();")))
        << "Triangulation output must be tied to the project active at launch.";
    EXPECT_TRUE(triangulationBody.contains(QStringLiteral("ownerGuard->currentProjectPath() != projectPath")))
        << "Triangulation must not append AT results after switching projects.";
    EXPECT_FALSE(triangulationBody.contains(QStringLiteral("(void)QtConcurrent::run([self,")))
        << "Open-coded QtConcurrent can race with manager destruction.";

    const int workflowEnd = source.indexOf(
        QStringLiteral("} // namespace"),
        workflowStart);
    const QString workflowBody = source.mid(workflowStart,
                                            workflowEnd > workflowStart ? workflowEnd - workflowStart : -1);
    EXPECT_TRUE(workflowBody.contains(QStringLiteral("xjw::gui::tasks::runGuarded")))
        << "Sparse post-processing workflows should share the guarded GUI task runner.";
    EXPECT_TRUE(workflowBody.contains(QStringLiteral("runSparsePointWorkflowResult")))
        << "The guarded worker should still run the sparse point workflow off the GUI thread.";
    EXPECT_TRUE(workflowBody.contains(QStringLiteral("QPointer<ProjectManager> ownerGuard(_owner)")))
        << "Sparse post-processing completion must guard the owning ProjectManager.";
    EXPECT_TRUE(workflowBody.contains(
        QStringLiteral("const QString projectPath = _owner ? _owner->currentProjectPath() : QString();")))
        << "Sparse post-processing output must be tied to the project active at launch.";
    EXPECT_TRUE(workflowBody.contains(QStringLiteral("ownerGuard->currentProjectPath() != projectPath")))
        << "Sparse post-processing must not append AT results after switching projects.";
    EXPECT_FALSE(workflowBody.contains(QStringLiteral("(void)QtConcurrent::run([self,")))
        << "Open-coded QtConcurrent can race with manager destruction.";
}

TEST(SfmSparseResultMetadataTest, ScaleAwareBaConsumesTrackConfidenceWeights)
{
    const QString baHeader = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.h"));
    const QString baSource = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.cpp"));
    const QString baInputBuilder =
        readProjectSourceFile(QStringLiteral("src/core/sfm/project/BaTrackBuilder.cpp"));
    const QString incrementalSfm = readIncrementalSfmProjectImplementation();
    ASSERT_FALSE(baHeader.isEmpty());
    ASSERT_FALSE(baSource.isEmpty());
    ASSERT_FALSE(baInputBuilder.isEmpty());
    ASSERT_FALSE(incrementalSfm.isEmpty());

    EXPECT_TRUE(baHeader.contains(QStringLiteral("double weight")));
    EXPECT_TRUE(baSource.contains(QStringLiteral("observationWeight")));
    EXPECT_TRUE(baSource.contains(QStringLiteral("observationWeight(obs)")));
    EXPECT_TRUE(baInputBuilder.contains(QStringLiteral("track.confidence")));
    EXPECT_TRUE(baInputBuilder.contains(QStringLiteral("baObservation.weight")));
    EXPECT_TRUE(incrementalSfm.contains(QStringLiteral("pt.track.confidence")));
    EXPECT_TRUE(incrementalSfm.contains(QStringLiteral("obs.weight")));
}

TEST(SfmSparseResultMetadataTest, BundleAdjustAutoEnablesSurveyControlConstraints)
{
    const QString execution = readProjectSourceFile(
        QStringLiteral("src/gui/project/support/ProjectBundleAdjustExecution.cpp"));
    const QString baHeader = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.h"));
    const QString baService = readProjectSourceFile(
        QStringLiteral("src/gui/project/services/BundleAdjustService.cpp"));
    ASSERT_FALSE(execution.isEmpty());
    ASSERT_FALSE(baHeader.isEmpty());
    ASSERT_FALSE(baService.isEmpty());

    EXPECT_TRUE(execution.contains(QStringLiteral("baInput.surveyControlTrackCount > 0")));
    EXPECT_TRUE(execution.contains(QStringLiteral("options.baOpt.enableControlPointConstraints = true")));
    EXPECT_TRUE(execution.contains(QStringLiteral("baInput.scaleBarConstraints")));
    EXPECT_TRUE(execution.contains(QStringLiteral("options.baOpt.enableScaleBarConstraints = true")));
    EXPECT_TRUE(execution.contains(QStringLiteral("options.baOpt.scaleBarConstraints = baInput.scaleBarConstraints")));
    EXPECT_TRUE(baHeader.contains(QStringLiteral("BAControlPointConstraint")));
    EXPECT_TRUE(baHeader.contains(QStringLiteral("BAScaleBarConstraint")));
    EXPECT_TRUE(baService.contains(QStringLiteral("control_point_constraints_summary")));
    EXPECT_TRUE(baService.contains(QStringLiteral("scale_bar_constraints_summary")));
}

TEST(SfmSparseResultMetadataTest, OneClickWorkflowPreservesProductionQualityRecord)
{
    const QString controller = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(controller.isEmpty());

    EXPECT_TRUE(controller.contains(QStringLiteral("result.resultRecordExtra")));
    EXPECT_TRUE(controller.contains(QStringLiteral("result.sfmDiagnostics")));
    EXPECT_TRUE(controller.contains(QStringLiteral("at_report.json")));
}

TEST(BundleAdjustSparseResultMetadataTest, ExportedSparseCloudCarriesFormalQualityMetadata)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    QJsonArray points;
    points.append(QJsonObject{
        {QStringLiteral("valid"), true},
        {QStringLiteral("converged"), true},
        {QStringLiteral("rms_after"), 0.5},
        {QStringLiteral("track_len"), 3},
        {QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}}
    });
    points.append(QJsonObject{
        {QStringLiteral("valid"), true},
        {QStringLiteral("converged"), true},
        {QStringLiteral("rms_after"), 0.7},
        {QStringLiteral("track_len"), 2},
        {QStringLiteral("point_xyz"), QJsonArray{2.0, 3.0, 4.0}}
    });

    const QJsonObject baResult{
        {QStringLiteral("camera_count"), 3},
        {QStringLiteral("mean_rms_after"), 0.6},
        {QStringLiteral("points"), points}
    };

    const auto exportResult = xjw::gui::project::exportBundleAdjustSparseCloud(
        baResult,
        QStringList{QStringLiteral("1.jpg"), QStringLiteral("2.jpg"), QStringLiteral("3.jpg")},
        tempDir.path(),
        true);

    ASSERT_TRUE(exportResult.exported) << exportResult.errorMessage.toStdString();
    EXPECT_EQ(exportResult.extraRecord.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindSparsePostprocess);
    EXPECT_EQ(exportResult.extraRecord.value(QStringLiteral("source_result_kind")).toString(),
              xjw::gui::project::kSparseResultKindSfmSparseReconstruction);
    EXPECT_TRUE(xjw::gui::project::isProductionSparseResult(exportResult.extraRecord));

    const QString sidecarPath = exportResult.extraRecord.value(QStringLiteral("files")).toObject()
                                    .value(QStringLiteral("sparse_cloud_points_json")).toString();
    QFile sidecarFile(sidecarPath);
    ASSERT_TRUE(sidecarFile.open(QIODevice::ReadOnly));
    const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();
    EXPECT_EQ(sidecar.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindSparsePostprocess);
    EXPECT_TRUE(xjw::gui::project::isProductionSparseResult(sidecar));
}

TEST(DownstreamSparseInputGateTest, ResolveSparseContextSkipsPreviewAndRequiresProduction)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString previewSidecar = QDir(tempDir.path()).filePath(QStringLiteral("preview_points.json"));
    const QString formalSidecar = QDir(tempDir.path()).filePath(QStringLiteral("formal_points.json"));
    QFile previewFile(previewSidecar);
    ASSERT_TRUE(previewFile.open(QIODevice::WriteOnly));
    previewFile.write("{}");
    previewFile.close();
    QFile formalFile(formalSidecar);
    ASSERT_TRUE(formalFile.open(QIODevice::WriteOnly));
    formalFile.write("{}");
    formalFile.close();

    const QJsonObject previewQuality = xjw::gui::project::buildSparseQualityMetadata(
        QJsonArray{QJsonObject{{QStringLiteral("track_len"), 2},
                               {QStringLiteral("rms_reproj_px"), 1.0}}},
        2,
        false,
        xjw::gui::project::kSparseResultKindPairwisePreview);
    const QJsonObject formalQuality = xjw::gui::project::buildSparseQualityMetadata(
        QJsonArray{QJsonObject{{QStringLiteral("track_len"), 2},
                               {QStringLiteral("rms_reproj_px"), 0.8}},
                   QJsonObject{{QStringLiteral("track_len"), 3},
                               {QStringLiteral("rms_reproj_px"), 0.6}}},
        3,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);

    QJsonObject formalRecord = xjw::gui::project::mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("output_dir"), tempDir.path()},
                    {QStringLiteral("files"), QJsonObject{
                         {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("formal.ply")},
                         {QStringLiteral("sparse_cloud_points_json"), formalSidecar}}},
                    {QStringLiteral("selected_images"), QJsonArray{
                         QStringLiteral("1.jpg"), QStringLiteral("2.jpg"), QStringLiteral("3.jpg")}}},
        formalQuality);
    QJsonObject previewRecord = xjw::gui::project::mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("output_dir"), tempDir.path()},
                    {QStringLiteral("files"), QJsonObject{
                         {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("preview.ply")},
                         {QStringLiteral("sparse_cloud_points_json"), previewSidecar}}},
                    {QStringLiteral("selected_images"), QJsonArray{
                         QStringLiteral("1.jpg"), QStringLiteral("2.jpg")}}},
        previewQuality);

    QJsonObject meta;
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{formalRecord, previewRecord};

    xjw::gui::project::SparsePointContext context;
    QString errorMessage;
    EXPECT_TRUE(xjw::gui::project::resolveSparsePointContext(meta, -1, &context, &errorMessage));
    EXPECT_EQ(context.sourceResultIndex, 0);

    errorMessage.clear();
    EXPECT_FALSE(xjw::gui::project::resolveSparsePointContext(meta, 1, &context, &errorMessage));
    EXPECT_TRUE(errorMessage.contains(QStringLiteral("两视")));

    errorMessage.clear();
    EXPECT_TRUE(xjw::gui::project::resolveSparsePointContext(meta, 0, &context, &errorMessage));
    EXPECT_EQ(context.sourceResultIndex, 0);
}

TEST(DownstreamSparseInputGateTest, DenseAndDemUseProductionSparseInputs)
{
    const QString denseSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString terrainSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(denseSource.isEmpty());
    ASSERT_FALSE(terrainSource.isEmpty());

    EXPECT_TRUE(denseSource.contains(QStringLiteral("findLatestProductionAtResultIndex")));
    EXPECT_TRUE(denseSource.contains(QStringLiteral("sparseResultBlockingReason")));
    EXPECT_TRUE(terrainSource.contains(QStringLiteral("findLatestProductionAtResultIndex")));
    EXPECT_FALSE(terrainSource.contains(QStringLiteral("startTriangulationAsync(triangulationSettings)")));
}

TEST(DownstreamSparseInputGateTest, OneClickDenseStageUsesCurrentSfmResultIndex)
{
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(controllerSource.contains(QStringLiteral("sfm_at_index")));
    EXPECT_TRUE(controllerSource.contains(
        QStringLiteral("settings.value(QStringLiteral(\"sfm_at_index\")).toInt(-1)")));
}

TEST(DownstreamSparseInputGateTest, OneClickWorkflowStopsDenseWhenCurrentSfmQualityIsBlocked)
{
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(controllerSource.contains(QStringLiteral("isProductionSparseResult(resultRecordExtra)")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("sparseResultBlockingReason(resultRecordExtra)")));
}

TEST(FeatureMatcherScriptLookupTest, MatcherFactoryPythonAdapterUsesSourceAwareScriptLookup)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/feature_match/MatcherFactory.cpp"));
    const QString cmake = readProjectSourceFile(QStringLiteral("src/core/feature_match/CMakeLists.txt"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(cmake.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QString findScriptFile")));
    EXPECT_TRUE(source.contains(QStringLiteral("PLASCAN_SCRIPT_DIR")));
    EXPECT_TRUE(source.contains(QStringLiteral("PLASCAN_SOURCE_DIR")));
    EXPECT_TRUE(source.contains(QStringLiteral("findScriptFile(scriptName)")));
    EXPECT_TRUE(source.contains(QStringLiteral("match_roma.py")));
    EXPECT_TRUE(source.contains(QStringLiteral("run_dedode.py")));
    EXPECT_FALSE(source.contains(QStringLiteral("run_disk_aliked.py")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("PLASCAN_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")));
}

TEST(AerialTriangulationWorkflowTest, ConfirmsExistingDepthMapsWillBeInvalidatedBeforeStarting)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(start, 0);
    const int finish = source.indexOf(
        QStringLiteral("void MenuWorkflowController::runUnifiedAerialTriangulation"), start);
    ASSERT_GT(finish, start);
    const QString block = source.mid(start, finish - start);

    const int depthCheck = block.indexOf(QStringLiteral("depth_map_results"));
    const int confirmation = block.indexOf(QStringLiteral("深度图将在空中三角测量成功后失效并从项目中移除"));
    const int prerequisiteWork = block.indexOf(QStringLiteral("runGuarded"));
    ASSERT_GE(depthCheck, 0);
    ASSERT_GE(confirmation, 0);
    ASSERT_GE(prerequisiteWork, 0);
    EXPECT_LT(depthCheck, prerequisiteWork);
    EXPECT_LT(confirmation, prerequisiteWork);
    EXPECT_TRUE(block.contains(QStringLiteral("QMessageBox::Yes")));
    EXPECT_TRUE(block.contains(QStringLiteral("QMessageBox::No")));
}

TEST(ModelGenerationWorkflowTest, SmallDepthMapBatchesFallbackOnlyAfterStrictFusionCollapses)
{
    const QString denseSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(denseSource.isEmpty());

    const int start = denseSource.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startFuseDepthMapsAsync"));
    ASSERT_GE(start, 0);
    const int finish = denseSource.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"), start);
    ASSERT_GT(finish, start);
    const QString block = denseSource.mid(start, finish - start);

    EXPECT_TRUE(block.contains(QStringLiteral(
        "fusionCfg.minNumPixels = std::max(1, request.minConsistentViews)")));
    EXPECT_TRUE(block.contains(QStringLiteral(
        "fusionCfg.enableLowYieldFallback = totalFrames <= 32")));
    EXPECT_FALSE(block.contains(QStringLiteral("std::min(fusionCfg.minNumPixels, 2)")))
        << "Small projects must try the configured production consensus before an observed low-yield fallback.";
}

TEST(MatchViewerSidecarOrderTest, ReordersCachedPointsWhenDisplayOrderIsReversed)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("parseMatchFile(matchFile, imgA, imgB")));
    EXPECT_TRUE(source.contains(QStringLiteral("displayOrderForMatchFile")));
    EXPECT_TRUE(source.contains(QStringLiteral("appendSidecarMatchedPoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("reversed ? p1 : p0")));
}

TEST(MatchViewerValidityTest, UsesTiePointTrackValidityBeforeSparsePointFallback)
{
    const QString analyzerHeader =
        readProjectSourceFile(QStringLiteral("src/gui/widgets/MatchValidityAnalyzer.h"));
    const QString analyzerSource =
        readProjectSourceFile(QStringLiteral("src/gui/widgets/MatchValidityAnalyzer.cpp"));
    const QString dualSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.cpp"));
    const QString overlaySource = readProjectSourceFile(QStringLiteral("src/gui/widgets/MatchLineOverlay.cpp"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchViewerDialog.cpp"));
    const QString guiSources = readProjectSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    ASSERT_FALSE(analyzerHeader.isEmpty());
    ASSERT_FALSE(analyzerSource.isEmpty());
    ASSERT_FALSE(dualSource.isEmpty());
    ASSERT_FALSE(overlaySource.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());
    ASSERT_FALSE(guiSources.isEmpty());

    EXPECT_TRUE(analyzerHeader.contains(QStringLiteral("MatchValidityResult")));
    EXPECT_TRUE(analyzerSource.contains(QStringLiteral("matched_indices0")));
    EXPECT_TRUE(analyzerSource.contains(QStringLiteral("latest_tie_points.json")));
    EXPECT_TRUE(analyzerSource.contains(QStringLiteral("tiePointSidecarPaths")));
    EXPECT_TRUE(analyzerSource.contains(QStringLiteral("sfm_sparse_points.json")));
    EXPECT_TRUE(analyzerSource.contains(QStringLiteral("QStringLiteral(\"tracks\")")));
    EXPECT_TRUE(analyzerSource.contains(QStringLiteral("observations")));
    EXPECT_TRUE(dualSource.contains(QStringLiteral("analyzeMatchTrackValidity(matchFile, imgA, imgB)")));
    EXPECT_TRUE(dualSource.contains(QStringLiteral("emit matchValidityLoaded")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("const QColor inlierColor(0, 80, 255);")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("const QColor outlierColor(255, 0, 0);")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("有效连接点：%1 | 无效匹配：%2")));
    EXPECT_TRUE(guiSources.contains(QStringLiteral("widgets/MatchValidityAnalyzer.cpp")));
}

TEST(MatchViewerValidityTest, IgnoresSparsePointsWithDuplicateImageObservations)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("imageA.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("imageB.png"));
    const QString matchFile = QDir(tempDir.path()).filePath(QStringLiteral("imageA__imageB_lightglue.match"));
    const QString sparseSidecar = QDir(tempDir.path()).filePath(QStringLiteral("sfm_sparse_points.json"));

    QJsonObject matchSidecar;
    matchSidecar.insert(QStringLiteral("feature_format_version"), 2);
    matchSidecar.insert(QStringLiteral("image0_path"), imageA);
    matchSidecar.insert(QStringLiteral("image0_name"), QStringLiteral("imageA"));
    matchSidecar.insert(QStringLiteral("image1_path"), imageB);
    matchSidecar.insert(QStringLiteral("image1_name"), QStringLiteral("imageB"));
    matchSidecar.insert(QStringLiteral("matched_indices0"), QJsonArray{10, 11});
    matchSidecar.insert(QStringLiteral("matched_indices1"), QJsonArray{20, 20});

    QFile matchSidecarFile(matchFile + QStringLiteral(".json"));
    ASSERT_TRUE(matchSidecarFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    matchSidecarFile.write(QJsonDocument(matchSidecar).toJson(QJsonDocument::Compact));
    matchSidecarFile.close();

    QJsonArray duplicateObservations;
    duplicateObservations.append(QJsonObject{
        {QStringLiteral("image_path"), imageA},
        {QStringLiteral("image_name"), QStringLiteral("imageA")},
        {QStringLiteral("feature_idx"), 10}
    });
    duplicateObservations.append(QJsonObject{
        {QStringLiteral("image_path"), imageA},
        {QStringLiteral("image_name"), QStringLiteral("imageA")},
        {QStringLiteral("feature_idx"), 11}
    });
    duplicateObservations.append(QJsonObject{
        {QStringLiteral("image_path"), imageB},
        {QStringLiteral("image_name"), QStringLiteral("imageB")},
        {QStringLiteral("feature_idx"), 20}
    });
    QJsonObject sparseRoot;
    sparseRoot.insert(QStringLiteral("points"), QJsonArray{
        QJsonObject{{QStringLiteral("observations"), duplicateObservations}}
    });

    QFile sparseFile(sparseSidecar);
    ASSERT_TRUE(sparseFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sparseFile.write(QJsonDocument(sparseRoot).toJson(QJsonDocument::Compact));
    sparseFile.close();

    MatchValidityContext context;
    context.sparseSidecarPaths = QStringList{sparseSidecar};
    const MatchValidityResult result = analyzeMatchTrackValidity(matchFile, imageA, imageB, context);

    ASSERT_TRUE(result.hasTrackValidity);
    ASSERT_EQ(result.inlierMask.size(), 2);
    EXPECT_FALSE(result.inlierMask.at(0));
    EXPECT_FALSE(result.inlierMask.at(1));
    EXPECT_EQ(result.validCount, 0);
    EXPECT_EQ(result.invalidCount, 2);
}

TEST(MatchViewerValidityTest, PrefersTiePointTracksWhoseFeatureIndicesMatchRawPairs)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("imageA.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("imageB.png"));
    const QString matchFile = QDir(tempDir.path()).filePath(QStringLiteral("imageA__imageB_lightglue.match"));
    const QString tiePointSidecar = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));
    const QString sparseSidecar = QDir(tempDir.path()).filePath(QStringLiteral("sfm_sparse_points.json"));

    QJsonObject matchSidecar;
    matchSidecar.insert(QStringLiteral("image0_path"), imageA);
    matchSidecar.insert(QStringLiteral("image0_name"), QStringLiteral("imageA"));
    matchSidecar.insert(QStringLiteral("image1_path"), imageB);
    matchSidecar.insert(QStringLiteral("image1_name"), QStringLiteral("imageB"));
    matchSidecar.insert(QStringLiteral("matched_indices0"), QJsonArray{10, 11});
    matchSidecar.insert(QStringLiteral("matched_indices1"), QJsonArray{20, 21});
    QFile matchSidecarFile(matchFile + QStringLiteral(".json"));
    ASSERT_TRUE(matchSidecarFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    matchSidecarFile.write(QJsonDocument(matchSidecar).toJson(QJsonDocument::Compact));
    matchSidecarFile.close();

    const QJsonArray validObservations{
        QJsonObject{{QStringLiteral("image_path"), imageA}, {QStringLiteral("feature_idx"), 10}},
        QJsonObject{{QStringLiteral("image_path"), imageB}, {QStringLiteral("feature_idx"), 20}}
    };
    QFile tiePointFile(tiePointSidecar);
    ASSERT_TRUE(tiePointFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    tiePointFile.write(QJsonDocument(QJsonObject{{QStringLiteral("tracks"), QJsonArray{
        QJsonObject{{QStringLiteral("observations"), validObservations}}
    }}}).toJson(QJsonDocument::Compact));
    tiePointFile.close();

    QFile sparseFile(sparseSidecar);
    ASSERT_TRUE(sparseFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sparseFile.write(QJsonDocument(QJsonObject{{QStringLiteral("points"), QJsonArray{
        QJsonObject{{QStringLiteral("observations"), QJsonArray{
            QJsonObject{{QStringLiteral("image_path"), imageA}, {QStringLiteral("feature_idx"), 99}},
            QJsonObject{{QStringLiteral("image_path"), imageB}, {QStringLiteral("feature_idx"), 98}}
        }}}
    }}}).toJson(QJsonDocument::Compact));
    sparseFile.close();

    MatchValidityContext context;
    context.tiePointSidecarPaths = QStringList{tiePointSidecar};
    context.sparseSidecarPaths = QStringList{sparseSidecar};
    const MatchValidityResult result = analyzeMatchTrackValidity(matchFile, imageA, imageB, context);

    ASSERT_TRUE(result.hasTrackValidity);
    EXPECT_EQ(result.sparseSidecarPath, tiePointSidecar);
    EXPECT_EQ(result.validCount, 1);
    EXPECT_EQ(result.invalidCount, 1);
    ASSERT_EQ(result.inlierMask.size(), 2);
    EXPECT_TRUE(result.inlierMask.at(0));
    EXPECT_FALSE(result.inlierMask.at(1));
}

TEST(MatchPairSelectorOverlapCandidatesTest, ListsOverlapPairsEvenWhenNoMatchFileExists)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("overlapCandidate")));
    EXPECT_TRUE(header.contains(QStringLiteral("loadOverlapCandidatesForImage")));
    EXPECT_TRUE(source.contains(QStringLiteral("vocabulary_overlap_pairs.json")));
    EXPECT_TRUE(source.contains(QStringLiteral("vocabulary_overlap_pairs.lis")));
    EXPECT_TRUE(source.contains(QStringLiteral("重叠候选")));
    EXPECT_TRUE(source.contains(QStringLiteral("matchFilePath.isEmpty()")));
}

TEST(MatchPairSelectorTrackValidityTest, ShowsMetashapeStyleCounts)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("bool hasTrackValidity = false;")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"MatchValidityAnalyzer.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("headers << tr(\"图像\") << tr(\"原始匹配\")")));
    EXPECT_TRUE(source.contains(QStringLiteral("<< tr(\"有效连接点\") << tr(\"无效匹配\")")));
    EXPECT_TRUE(source.contains(QStringLiteral("<< tr(\"最佳算法\") << tr(\"状态\")")));
    EXPECT_TRUE(source.contains(QStringLiteral("const MatchValidityContext validityContext")))
        << "Sparse-track sidecar discovery must be done once per selector scan, not once per table row.";
    EXPECT_TRUE(source.contains(QStringLiteral("analyzeMatchTrackValidity")));
    EXPECT_TRUE(source.contains(QStringLiteral("info.hasTrackValidity = true;")));
    EXPECT_TRUE(source.contains(QStringLiteral("原始 %3，有效连接点 %4，无效匹配 %5")));
    EXPECT_TRUE(source.contains(QStringLiteral("已对齐")));
    EXPECT_TRUE(source.contains(QStringLiteral("const bool hasValidityStats = info.hasTrackValidity;")))
        << "Metashape 风格的有效点必须来自空三后保留的 tie point，不能回退到两视图几何内点。";
    EXPECT_FALSE(source.contains(QStringLiteral("info.hasTrackValidity || info.hasInlierStats")))
        << "几何验证内点不是 Metashape View Matches 里的有效连接点。";
}

TEST(MatchPairSelectorResponsivenessTest, DefersHeavyMatchScanToBackgroundWorker)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const int ctorStart = source.indexOf(QStringLiteral("MatchPairSelectorDialog::MatchPairSelectorDialog"));
    ASSERT_GE(ctorStart, 0);
    const int dtorStart = source.indexOf(QStringLiteral("MatchPairSelectorDialog::~MatchPairSelectorDialog"), ctorStart);
    ASSERT_GT(dtorStart, ctorStart);
    const QString ctorBody = source.mid(ctorStart, dtorStart - ctorStart);

    const int loadStart = source.indexOf(QStringLiteral("void MatchPairSelectorDialog::loadMatchPairsForImage"));
    ASSERT_GE(loadStart, 0);
    const int parseStart = source.indexOf(QStringLiteral("QList<MatchPairSelectorDialog::MatchInfo> MatchPairSelectorDialog::parseMatchDataForImage"),
                                          loadStart);
    ASSERT_GT(parseStart, loadStart);
    const QString loadBody = source.mid(loadStart, parseStart - loadStart);

    EXPECT_TRUE(header.contains(QStringLiteral("QFutureWatcher<MatchInfoList> *_matchLoadWatcher = nullptr;")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include <QtConcurrent/QtConcurrent>")));
    EXPECT_TRUE(source.contains(QStringLiteral("startAsyncMatchPairLoad(imagePath)")));
    EXPECT_TRUE(source.contains(QStringLiteral("onMatchPairsLoaded")));
    EXPECT_TRUE(source.contains(QStringLiteral("QtConcurrent::run([snapshot, imagePath]()")));
    EXPECT_TRUE(source.contains(QStringLiteral("QTimer::singleShot(0, this, &MatchPairSelectorDialog::onRefresh);")));
    EXPECT_FALSE(ctorBody.contains(QStringLiteral("loadProjectImages();")))
        << "The selector must show first and load the initial match list after the event loop starts.";
    EXPECT_FALSE(loadBody.contains(QStringLiteral("_currentMatches = parseMatchDataForImage(imagePath);")))
        << "Scanning match files and sparse validity sidecars on the GUI thread freezes large projects.";
}

TEST(MatchPairSelectorResponsivenessTest, PrioritizesCurrentImageBeforeFullCatalogScan)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("_priorityMatchLoadWatcher")))
        << "The selector should keep a separate watcher for fast current-image results.";
    EXPECT_TRUE(source.contains(QStringLiteral("parsePriorityMatchDataForImageFromSnapshot")))
        << "The first worker should scan only match files that name the current image.";
    EXPECT_TRUE(source.contains(QStringLiteral("candidateMatchFilesForImage")))
        << "Fast loading must avoid a full MatchResultCatalog scan before showing rows.";
    EXPECT_TRUE(header.contains(QStringLiteral("startFullMatchPairLoad")))
        << "The full catalog scan should have its own delayed launcher.";
    EXPECT_TRUE(source.contains(QStringLiteral("startFullMatchPairLoad(makeSnapshot(), imagePath, generation)")))
        << "The full catalog scan should start only after the priority result returns to the UI thread.";
    EXPECT_TRUE(source.contains(QStringLiteral("parseMatchDataForImageFromSnapshot(snapshot, imagePath, progressCallback)")))
        << "The full catalog scan should still run in the background to fill validity and variants.";
    EXPECT_TRUE(source.contains(QStringLiteral("priorityLoad")))
        << "Loaded results must be distinguished so quick rows do not mark the full scan complete.";
    EXPECT_TRUE(source.contains(QStringLiteral("正在后台访问全部匹配数据")))
        << "After quick rows are visible the status should tell the user the full directory scan is still loading.";

    const int priorityStart = source.indexOf(QStringLiteral("void MatchPairSelectorDialog::startAsyncMatchPairLoad"));
    ASSERT_GE(priorityStart, 0);
    const int fullStart = source.indexOf(QStringLiteral("void MatchPairSelectorDialog::startFullMatchPairLoad"),
                                         priorityStart);
    ASSERT_GT(fullStart, priorityStart);
    const QString priorityStartBody = source.mid(priorityStart, fullStart - priorityStart);
    EXPECT_FALSE(priorityStartBody.contains(QStringLiteral("parseMatchDataForImageFromSnapshot(snapshot, imagePath)")))
        << "Starting the full scan together with the priority scan still lets the full directory scan compete for IO.";
}

TEST(MatchPairSelectorResponsivenessTest, ShowsPercentageProgressDuringFullMatchScan)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.cpp"));
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.ui"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(ui.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QProgressBar")))
        << "The selector should keep a progress bar pointer instead of relying on status text only.";
    EXPECT_TRUE(header.contains(QStringLiteral("_scanProgressBar")))
        << "The scan progress bar needs a stable member so async callbacks can update it.";
    EXPECT_TRUE(ui.contains(QStringLiteral("QProgressBar")))
        << "The progress indicator must be part of the dialog layout.";
    EXPECT_TRUE(ui.contains(QStringLiteral("m_scanProgressBar")))
        << "The progress bar object name should be stable for UI tests and future styling.";
    EXPECT_TRUE(source.contains(QStringLiteral("_scanProgressBar = ui.m_scanProgressBar")));
    EXPECT_TRUE(source.contains(QStringLiteral("_scanProgressBar->setRange(0, 100)")))
        << "The full match scan should show a real percentage instead of an indeterminate busy bar.";
    EXPECT_TRUE(source.contains(QStringLiteral("_scanProgressBar->setValue(0)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_scanProgressBar->setValue(percent)")))
        << "Catalog progress callbacks should drive the progress bar value.";
    EXPECT_FALSE(source.contains(QStringLiteral("_scanProgressBar->setRange(0, 0)")))
        << "The selector now has catalog file counts, so it must not keep the bar indeterminate.";
    EXPECT_TRUE(source.contains(QStringLiteral("_scanProgressBar->setVisible(true)")))
        << "Starting current-image scan should make progress visible immediately.";
    EXPECT_TRUE(source.contains(QStringLiteral("_scanProgressBar->setVisible(false)")))
        << "Finishing the full background statistics pass should hide the progress bar.";
    EXPECT_TRUE(source.contains(QStringLiteral("config.progressCallback")))
        << "The full catalog scan must receive file-level progress from MatchResultCatalog.";
    EXPECT_TRUE(source.contains(QStringLiteral("QMetaObject::invokeMethod")))
        << "Worker-thread catalog progress must be marshalled back to the GUI thread.";
    EXPECT_TRUE(source.contains(QStringLiteral("正在优先加载当前影像匹配")))
        << "The first stage should tell users why the table may still be empty.";
    EXPECT_TRUE(source.contains(QStringLiteral("正在后台访问全部匹配数据：%1%")))
        << "The progress bar should report the whole match directory access progress, not current-image progress.";

    const int progressStart = source.indexOf(QStringLiteral("void MatchPairSelectorDialog::setFullScanProgress"));
    ASSERT_GE(progressStart, 0);
    const int loadedStart = source.indexOf(QStringLiteral("void MatchPairSelectorDialog::onMatchPairsLoaded"),
                                           progressStart);
    ASSERT_GT(loadedStart, progressStart);
    const QString progressBody = source.mid(progressStart, loadedStart - progressStart);
    EXPECT_TRUE(progressBody.contains(QStringLiteral("if (!_matchLoadWatcher)")))
        << "Late queued progress events must not show the bar again after the full scan has finished.";
}

TEST(MatchPairSelectorCatalogTest, UsesCatalogGroupsAndPassesVariantsToViewer)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.cpp"));
    const QString aerialSources = readProjectSourceFile(
        QStringLiteral("src/core/aerial_triangulation/CMakeLists.txt"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(aerialSources.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QVector<xjw::aerial_triangulation::MatchVariant> variants;")));
    EXPECT_TRUE(source.contains(QStringLiteral("#include \"preparation/MatchResultCatalog.h\"")));
    EXPECT_TRUE(source.contains(QStringLiteral("MatchResultCatalogConfig")));

    const int parseStart = source.indexOf(QStringLiteral("MatchPairSelectorDialog::parseMatchDataForImageFromSnapshot"));
    ASSERT_GE(parseStart, 0);
    const int overlapStart = source.indexOf(QStringLiteral("MatchPairSelectorDialog::loadOverlapCandidatesForImage"),
                                            parseStart);
    ASSERT_GT(overlapStart, parseStart);
    const QString parseBody = source.mid(parseStart, overlapStart - parseStart);
    EXPECT_TRUE(parseBody.contains(QStringLiteral("config.targetImagePath = imagePath")));
    EXPECT_TRUE(parseBody.contains(QStringLiteral("config.targetImagePaths = snapshot.allImages")));
    EXPECT_TRUE(source.contains(QStringLiteral(".scan()")));
    EXPECT_TRUE(source.contains(QStringLiteral("bestVariantIndex")));
    EXPECT_TRUE(source.contains(QStringLiteral("const QString base = imageBaseToken(imgPath);")));
    EXPECT_TRUE(source.contains(QStringLiteral("baseToPath.insert(base, imgPath)")));
    EXPECT_TRUE(source.contains(QStringLiteral("setMatchVariants(info.variants, info.matchFilePath)")));
    EXPECT_TRUE(source.contains(QStringLiteral("const QStringList current_project_images")));
    EXPECT_TRUE(source.contains(QStringLiteral("resolveProjectImagePathFromToken(\n"
                                               "                                      match.imagePath")));
    EXPECT_TRUE(aerialSources.contains(QStringLiteral("preparation/MatchResultCatalog.cpp")));
}

TEST(MatchViewerEmptyMatchTest, CanOpenImagePairWithoutSparseMatchFile)
{
    const QString viewerSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.cpp"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchViewerDialog.cpp"));
    ASSERT_FALSE(viewerSource.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());

    EXPECT_TRUE(viewerSource.contains(QStringLiteral("matchFile.trimmed().isEmpty()")));
    EXPECT_TRUE(viewerSource.contains(QStringLiteral("QVector<QPointF>{}, QVector<QPointF>{}")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("尚未生成匹配")));
}

TEST(MatchViewerVariantSwitchTest, ExposesCompactVariantComboAndReloadsSparseMatch)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchViewerDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchViewerDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("setMatchVariants")));
    EXPECT_TRUE(header.contains(QStringLiteral("QComboBox *_variantCombo = nullptr;")));
    EXPECT_TRUE(header.contains(QStringLiteral("QVector<xjw::aerial_triangulation::MatchVariant> _matchVariants;")));
    EXPECT_TRUE(source.contains(QStringLiteral("onVariantChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("_variantCombo->setVisible(_variantCombo->count() > 1)")));
    EXPECT_TRUE(source.contains(QStringLiteral("loadMatchPair(_imageA, _imageB, _matchFile)")));
    EXPECT_TRUE(source.contains(QStringLiteral("几何内点 %1 / 原始 %2")));
}

TEST(MatchViewerResponsivenessTest, LimitsDefaultSparseRenderingWork)
{
    const QString imageViewSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/ImageViewWidget.cpp"));
    const QString overlaySource = readProjectSourceFile(QStringLiteral("src/gui/widgets/MatchLineOverlay.cpp"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchViewerDialog.cpp"));
    ASSERT_FALSE(imageViewSource.isEmpty());
    ASSERT_FALSE(overlaySource.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());

    EXPECT_TRUE(imageViewSource.contains(QStringLiteral("constexpr int maxInitialPointItems = 20000")))
        << "Large match files must not create one QGraphicsEllipseItem per match on the GUI thread.";
    EXPECT_TRUE(imageViewSource.contains(QStringLiteral("const int pointCount = static_cast<int>(points.size())")));
    EXPECT_TRUE(imageViewSource.contains(QStringLiteral("const int pointItemCount = std::min(pointCount, maxInitialPointItems)")));
    EXPECT_TRUE(overlaySource.contains(QStringLiteral("_maxDisplayCount(5000)")))
        << "Sparse match lines should have a finite default draw budget.";
    EXPECT_FALSE(dialogSource.contains(QStringLiteral("_maxCountSpin->setValue(0);")))
        << "Opening a large match file must not auto-switch the viewer back to unlimited rendering.";
}

TEST(MatchViewerEndpointVisibilityTest, HidesAllEndpointsWhenNoMatchLineIsVisible)
{
    DualImageViewer viewer;
    viewer.resize(800, 600);
    viewer.show();

    const QVector<QPointF> leftPoints{
        QPointF(20.0, 20.0),
        QPointF(40.0, 40.0),
        QPointF(60.0, 60.0)
    };
    const QVector<QPointF> rightPoints{
        QPointF(25.0, 25.0),
        QPointF(45.0, 45.0),
        QPointF(65.0, 65.0)
    };

    viewer.leftView()->setMatchPoints(leftPoints);
    viewer.rightView()->setMatchPoints(rightPoints);
    viewer.overlay()->setMatches(leftPoints, rightPoints);
    viewer.overlay()->setInlierMask(QVector<bool>{false, false, false});
    viewer.overlay()->setShowOnlyInliers(true);
    QTest::qWait(30);

    auto visiblePointCount = [](ImageViewWidget *imageView)
    {
        int count = 0;
        const QList<QGraphicsItem *> items = imageView->view()->scene()->items();
        for (QGraphicsItem *item : items)
        {
            if (qgraphicsitem_cast<QGraphicsEllipseItem *>(item) && item->isVisible())
            {
                ++count;
            }
        }
        return count;
    };

    EXPECT_EQ(visiblePointCount(viewer.leftView()), 0);
    EXPECT_EQ(visiblePointCount(viewer.rightView()), 0);
}

TEST(CodeStyleTest, MatchViewerDialogUsesLowerCamelPrivateMemberNames)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchViewerDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/tie_points/MatchViewerDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const QStringList expectedMembers = {
        QStringLiteral("DualImageViewer *_viewer = nullptr;"),
        QStringLiteral("QTabWidget *_tabWidget = nullptr;"),
        QStringLiteral("QWidget *_sparseTab = nullptr;"),
        QStringLiteral("QWidget *_denseTab = nullptr;"),
        QStringLiteral("int _initialTab = 0;"),
        QStringLiteral("QString _disparityFile;"),
        QStringLiteral("bool _sparseMatchFileMissing = false;"),
        QStringLiteral("QCheckBox *_syncModeChk = nullptr;"),
        QStringLiteral("QPushButton *_fitBtn = nullptr;"),
        QStringLiteral("QPushButton *_resetBtn = nullptr;"),
        QStringLiteral("QPushButton *_zoomInBtn = nullptr;"),
        QStringLiteral("QPushButton *_zoomOutBtn = nullptr;"),
        QStringLiteral("QComboBox *_variantCombo = nullptr;"),
        QStringLiteral("QPushButton *_lineColorBtn = nullptr;"),
        QStringLiteral("QDoubleSpinBox *_lineWidthSpin = nullptr;"),
        QStringLiteral("QSlider *_opacitySlider = nullptr;"),
        QStringLiteral("QSpinBox *_maxCountSpin = nullptr;"),
        QStringLiteral("QCheckBox *_showEndPointsChk = nullptr;"),
        QStringLiteral("QCheckBox *_showOnlyInliersChk = nullptr;"),
        QStringLiteral("QCheckBox *_rainbowChk = nullptr;"),
        QStringLiteral("QSlider *_denseOpacitySlider = nullptr;"),
        QStringLiteral("QComboBox *_denseColormapCombo = nullptr;"),
        QStringLiteral("QCheckBox *_denseAutoRangeChk = nullptr;"),
        QStringLiteral("QDoubleSpinBox *_denseMinSpin = nullptr;"),
        QStringLiteral("QDoubleSpinBox *_denseMaxSpin = nullptr;"),
        QStringLiteral("QGroupBox *_denseDisplayGroup = nullptr;"),
        QStringLiteral("QLabel *_statusLabel = nullptr;"),
        QStringLiteral("QString _matchFile;"),
        QStringLiteral("int _totalMatches = 0;"),
        QStringLiteral("int _validMatches = -1;"),
        QStringLiteral("int _invalidMatches = -1;"),
        QStringLiteral("QVector<xjw::aerial_triangulation::MatchVariant> _matchVariants;"),
        QStringLiteral("DialogSettingStore *_setting = nullptr;"),
    };
    for (const QString &expectedMember : expectedMembers)
    {
        EXPECT_TRUE(header.contains(expectedMember)) << qPrintable(expectedMember);
    }

    const QStringList oldMemberNames = {
        QStringLiteral("m_viewer"),
        QStringLiteral("m_tabWidget"),
        QStringLiteral("m_sparseTab"),
        QStringLiteral("m_denseTab"),
        QStringLiteral("m_initialTab"),
        QStringLiteral("m_disparityFile"),
        QStringLiteral("m_sparseMatchFileMissing"),
        QStringLiteral("m_syncModeChk"),
        QStringLiteral("m_fitBtn"),
        QStringLiteral("m_resetBtn"),
        QStringLiteral("m_zoomInBtn"),
        QStringLiteral("m_zoomOutBtn"),
        QStringLiteral("m_lineColorBtn"),
        QStringLiteral("m_lineWidthSpin"),
        QStringLiteral("m_opacitySlider"),
        QStringLiteral("m_maxCountSpin"),
        QStringLiteral("m_showEndPointsChk"),
        QStringLiteral("m_showOnlyInliersChk"),
        QStringLiteral("m_rainbowChk"),
        QStringLiteral("m_denseOpacitySlider"),
        QStringLiteral("m_denseColormapCombo"),
        QStringLiteral("m_denseAutoRangeChk"),
        QStringLiteral("m_denseMinSpin"),
        QStringLiteral("m_denseMaxSpin"),
        QStringLiteral("m_denseDisplayGroup"),
        QStringLiteral("m_statusLabel"),
        QStringLiteral("m_matchFile"),
        QStringLiteral("m_totalMatches"),
        QStringLiteral("m_setting"),
    };
    for (const QString &oldName : oldMemberNames)
    {
        EXPECT_FALSE(header.contains(oldName)) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("->"))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral(" ="))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(oldName + QStringLiteral("."))) << qPrintable(oldName);
        EXPECT_FALSE(source.contains(QStringLiteral("&") + oldName)) << qPrintable(oldName);
    }
    EXPECT_FALSE(source.contains(QStringLiteral("m_matchFile(")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_totalMatches(")));
}

TEST(MatchViewerVisualizationTest, ImageDisplayKeepsRawPixelOrientationForMatchCoordinates)
{
    const QString imageViewSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/ImageViewWidget.cpp"));
    const QString loaderSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerImageLoader.cpp"));
    ASSERT_FALSE(imageViewSource.isEmpty());
    ASSERT_FALSE(loaderSource.isEmpty());

    EXPECT_TRUE(imageViewSource.contains(QStringLiteral("LayerRenderer::loadImageForDisplay(imagePath, QString())")));
    EXPECT_TRUE(loaderSource.contains(QStringLiteral("setAutoTransform(false)")));
    EXPECT_FALSE(loaderSource.contains(QStringLiteral("setAutoTransform(true)")));
}

TEST(MatchViewerVisualizationTest, UsesSharedDisplayLoaderAndReportsAsyncImageFailures)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/ImageViewWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/ImageViewWidget.cpp"));
    const QString dualSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(dualSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("imageLoadFailed")));
    EXPECT_TRUE(source.contains(QStringLiteral("LayerRenderer::loadImageForDisplay(imagePath, QString())")));
    EXPECT_TRUE(source.contains(QStringLiteral("emit self->imageLoadFailed(imagePath")));
    EXPECT_TRUE(dualSource.contains(QStringLiteral("&ImageViewWidget::imageLoadFailed")));
    EXPECT_TRUE(dualSource.contains(QStringLiteral("emit loadFailed(message)")));
}

TEST(ImageDisplayDecodeTest, FallsBackToOpenCvByteDecodeWhenQtImagePluginCannotRead)
{
    const QString loader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerImageLoader.cpp"));
    const QString pathIo = readProjectSourceFile(QStringLiteral("src/common/io/PathIO.cpp"));
    ASSERT_FALSE(loader.isEmpty());
    ASSERT_FALSE(pathIo.isEmpty());

    EXPECT_TRUE(loader.contains(QStringLiteral("xjw::common::io::readImage(path, cv::IMREAD_UNCHANGED)")));
    EXPECT_TRUE(pathIo.contains(QStringLiteral("readFileBytes(path)")));
    EXPECT_TRUE(pathIo.contains(QStringLiteral("cv::imdecode")));
}

TEST(WindowsBuildScriptTest, SyncsQtImageFormatPluginsForDirectBinRuns)
{
    const QString source = readProjectSourceFile(QStringLiteral("scripts/build_win/build_windows_cuda.ps1"));
    ASSERT_FALSE(source.isEmpty());

    const int syncIndex = source.indexOf(QStringLiteral("function Sync-QtRuntime"));
    const int nextIndex = source.indexOf(QStringLiteral("function Sync-TorchRuntime"), syncIndex);
    ASSERT_GE(syncIndex, 0);
    ASSERT_GT(nextIndex, syncIndex);
    const QString syncBlock = source.mid(syncIndex, nextIndex - syncIndex);

    EXPECT_TRUE(syncBlock.contains(QStringLiteral("imageformats")));
    EXPECT_TRUE(syncBlock.contains(QStringLiteral("qjpeg")));
}

TEST(WindowsBuildScriptTest, GuiTargetOutputsPlascanExeOnWindows)
{
    const QString cmake = readProjectSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));
    const QString readme = readProjectSourceFile(QStringLiteral("scripts/build_win/README.md"));
    ASSERT_FALSE(cmake.isEmpty());
    ASSERT_FALSE(readme.isEmpty());

    EXPECT_TRUE(cmake.contains(QStringLiteral("if(WIN32)")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("set(PLASCAN_GUI_OUTPUT_NAME \"plascan\")")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("set(PLASCAN_GUI_OUTPUT_NAME \"plascan_gui.bin\")")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("OUTPUT_NAME \"${PLASCAN_GUI_OUTPUT_NAME}\"")));
    EXPECT_TRUE(readme.contains(QStringLiteral("bin\\plascan.exe")));
    EXPECT_FALSE(readme.contains(QStringLiteral("bin\\plascan_gui.bin.exe")));
}

TEST(CMakeTestRuntimeTest, WindowsTorchBackedTestsUseWholeExecutableCtestRegistration)
{
    const QString source = readProjectSourceFile(QStringLiteral("cmake/PlascanTestRuntime.cmake"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin")))
        << "Windows test discovery must find vcpkg DLLs even when VCPKG_INSTALLED_DIR is not set in the shell.";
    EXPECT_TRUE(source.contains(QStringLiteral("add_test(NAME ${target_name} COMMAND ${target_name})")))
        << "Torch-backed tests should not be executed just to enumerate test cases on Windows.";
    EXPECT_TRUE(source.contains(QStringLiteral("set_tests_properties(${target_name} PROPERTIES")))
        << "The whole-executable ctest entry still needs the runtime PATH.";
}

TEST(ModelDropSupportTest, AcceptsStandaloneModelAndPointCloudFiles)
{
    const QUrl projectUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/example.plascan"));
    const QUrl plyUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/model.ply"));
    const QUrl xyzUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/cloud.xyz"));
    const QUrl imageUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/image.png"));

    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/model.ply")));
    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/model.obj")));
    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/cloud.xyz")));
    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/cloud.txt")));
    EXPECT_FALSE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/project.plascan")));
    EXPECT_FALSE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/image.png")));

    EXPECT_EQ(xjw::gui::main_window::firstStandaloneModelFile({projectUrl, imageUrl, plyUrl, xyzUrl}),
              QStringLiteral("/tmp/model.ply"));
    EXPECT_TRUE(xjw::gui::main_window::firstStandaloneModelFile({imageUrl}).isEmpty());
}

TEST(ModelDropSupportTest, WorkspaceRoutesObjModelsToObjLoader)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int showModelStart = source.indexOf(QStringLiteral("void WorkspaceCenterWidget::showModelFile"));
    ASSERT_GE(showModelStart, 0);
    const int showPointCloudStart = source.indexOf(QStringLiteral("void WorkspaceCenterWidget::showPointCloudFile"),
                                                   showModelStart);
    ASSERT_GT(showPointCloudStart, showModelStart);
    const QString showModelBlock = source.mid(showModelStart, showPointCloudStart - showModelStart);

    EXPECT_TRUE(showModelBlock.contains(QStringLiteral("ext == QLatin1String(\"obj\")")));
    EXPECT_TRUE(showModelBlock.contains(QStringLiteral("_modelView->loadModelFromObj(modelPath)")));
    EXPECT_TRUE(showModelBlock.contains(QStringLiteral("showModelView()")));
}

TEST(DataTreeWidgetTest, ShowsTemporaryDroppedModelUntilCleared)
{
    DataTreeWidget tree;
    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray();
    tree.loadFromJson(meta);

    const QString modelPath = QStringLiteral("/tmp/temporary_model.ply");
    tree.addTransientModel(modelPath);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *modelSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("3D模型 (1)")))
        {
            modelSection = item;
            break;
        }
    }
    ASSERT_NE(modelSection, nullptr);
    ASSERT_EQ(modelSection->rowCount(), 1);
    EXPECT_EQ(modelSection->child(0, 0)->text(), QStringLiteral("temporary_model.ply  [临时]"));
    EXPECT_EQ(modelSection->child(0, 1)->text(), modelPath);

    tree.clearTransientResources();

    bool foundClearedModelSection = false;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("3D模型")))
        {
            foundClearedModelSection = true;
            break;
        }
    }
    EXPECT_FALSE(foundClearedModelSection);
}

TEST(DataTreeWidgetTest, ClearProjectRemovesMetadataAndTransientResources)
{
    DataTreeWidget tree;
    tree.setProjectPath(QStringLiteral("E:/projects/previous.plascan"));
    tree.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/projects/photo.png")}}}}});
    tree.addTransientModel(QStringLiteral("E:/projects/temporary.ply"));

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);
    ASSERT_GT(model->rowCount(), 0);

    tree.clearProject();

    EXPECT_EQ(model->rowCount(), 0);
    tree.addTransientModel(QStringLiteral("E:/projects/next_temporary.ply"));
    ASSERT_EQ(model->rowCount(), 1);
    EXPECT_TRUE(model->item(0, 0)->text().startsWith(QStringLiteral("3D模型")));
}

TEST(DataTreeWidgetTest, GroupsActiveResourcesUnderChunkRoots)
{
    DataTreeWidget tree;
    const QJsonArray chunks{
        QJsonObject{
            {QStringLiteral("id"), QStringLiteral("chunk-1")},
            {QStringLiteral("name"), QStringLiteral("区块 1")},
            {QStringLiteral("directory"), QStringLiteral("1")},
            {QStringLiteral("image_count"), 4},
            {QStringLiteral("tie_point_count"), 320}},
        QJsonObject{
            {QStringLiteral("id"), QStringLiteral("chunk-3")},
            {QStringLiteral("name"), QStringLiteral("区块 3")},
            {QStringLiteral("directory"), QStringLiteral("3")}}
    };
    tree.setChunkContext(chunks, QStringLiteral("chunk-3"));
    tree.loadFromJson(QJsonObject{
        {QStringLiteral("images"),
         QJsonArray{QJsonObject{
             {QStringLiteral("path"),
              QStringLiteral("/tmp/images/current.jpg")}}}},
        {QStringLiteral("aerial_triangulation_results"),
         QJsonArray{QJsonObject{
             {QStringLiteral("sparse_point_count"), 1238},
             {QStringLiteral("selected_images"),
              QJsonArray{QStringLiteral(
                  "/tmp/images/current.jpg")}},
             {QStringLiteral("files"),
              QJsonObject{
                  {QStringLiteral("sparse_cloud_xyz"),
                   QStringLiteral("/tmp/sparse/current.ply")}}}}}}
    });

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model =
        qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(view->header()->isHidden());
    ASSERT_EQ(model->rowCount(), 1);

    QStandardItem *workspace = model->item(0, 0);
    ASSERT_NE(workspace, nullptr);
    EXPECT_EQ(
        workspace->text(),
        QStringLiteral("工作区 (2个块, 5个图像)"));
    EXPECT_FALSE(workspace->icon().isNull());
    ASSERT_EQ(workspace->rowCount(), 2);

    QStandardItem *inactiveChunk = workspace->child(0, 0);
    ASSERT_NE(inactiveChunk, nullptr);
    EXPECT_EQ(
        inactiveChunk->text(),
        QStringLiteral("区块 1 (4个图像, 320个连接点)"));
    EXPECT_EQ(inactiveChunk->rowCount(), 0);
    EXPECT_FALSE(inactiveChunk->font().bold());

    QStandardItem *activeChunk = workspace->child(1, 0);
    ASSERT_NE(activeChunk, nullptr);
    EXPECT_EQ(
        activeChunk->text(),
        QStringLiteral("区块 3 (1个图像, 1,238个连接点)"));
    EXPECT_TRUE(activeChunk->font().bold());
    EXPECT_FALSE(activeChunk->icon().isNull());
    ASSERT_EQ(activeChunk->rowCount(), 2);
    QStandardItem *photos = activeChunk->child(0, 0);
    ASSERT_NE(photos, nullptr);
    EXPECT_TRUE(
        photos->text().startsWith(QStringLiteral("图像 (1/1 对齐)")));
    ASSERT_EQ(photos->rowCount(), 1);
    EXPECT_EQ(
        photos->child(0, 0)->text(), QStringLiteral("current.jpg"));
    EXPECT_FALSE(photos->child(0, 0)->icon().isNull());
}

TEST(DataTreeWidgetTest, DoesNotShowPointOnlyModelRecordAsThreeDModel)
{
    DataTreeWidget tree;

    QJsonObject denseRecord;
    denseRecord[QStringLiteral("kind")] = QStringLiteral("dense_cloud");
    denseRecord[QStringLiteral("dense_cloud_xyz")] = QStringLiteral("/tmp/mvs_output/dense_cloud.ply");
    denseRecord[QStringLiteral("point_count")] = 1058511291;
    denseRecord[QStringLiteral("face_count")] = 0;

    QJsonObject pointOnlyModelRecord;
    pointOnlyModelRecord[QStringLiteral("kind")] = QStringLiteral("mesh");
    pointOnlyModelRecord[QStringLiteral("model_ply")] = QStringLiteral("/tmp/mvs_output/products/model_from_mesh.ply");
    pointOnlyModelRecord[QStringLiteral("vertex_count")] = 1058511291;
    pointOnlyModelRecord[QStringLiteral("face_count")] = 0;

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray();
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{denseRecord};
    meta[QStringLiteral("model_results")] = QJsonArray{pointOnlyModelRecord};
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    auto findSection = [model](const QString &prefix) -> QStandardItem *
    {
        for (int row = 0; row < model->rowCount(); ++row)
        {
            QStandardItem *item = model->item(row, 0);
            if (item && item->text().startsWith(prefix))
            {
                return item;
            }
        }
        return nullptr;
    };

    QStandardItem *denseSection = findSection(QStringLiteral("稠密点云 (1)"));
    ASSERT_NE(denseSection, nullptr);
    ASSERT_EQ(denseSection->rowCount(), 1);
    EXPECT_EQ(denseSection->child(0, 1)->text(), QStringLiteral("/tmp/mvs_output/dense_cloud.ply"));

    QStandardItem *modelSection = findSection(QStringLiteral("3D模型"));
    EXPECT_EQ(modelSection, nullptr);
}

TEST(DataTreeWidgetTest, ShowsAlignedPhotoRatioAndHidesEmptySections)
{
    DataTreeWidget tree;

    QJsonObject alignedCamera;
    alignedCamera[QStringLiteral("C")] = QJsonArray{0.0, 0.0, 0.0};
    alignedCamera[QStringLiteral("R")] = QJsonArray{1.0, 0.0, 0.0,
                                                    0.0, 1.0, 0.0,
                                                    0.0, 0.0, 1.0};

    QJsonObject image0;
    image0[QStringLiteral("path")] = QStringLiteral("/tmp/images/image_001.jpg");
    image0[QStringLiteral("camera")] = alignedCamera;

    QJsonObject image1;
    image1[QStringLiteral("path")] = QStringLiteral("/tmp/images/image_002.jpg");

    QJsonObject image2;
    image2[QStringLiteral("path")] = QStringLiteral("/tmp/images/image_003.jpg");

    QJsonObject atRecord;
    atRecord[QStringLiteral("sparse_point_count")] = 1873;
    atRecord[QStringLiteral("selected_images")] = QJsonArray{
        QStringLiteral("/tmp/images/image_002.jpg")
    };
    atRecord[QStringLiteral("files")] = QJsonObject{
        {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("/tmp/sfm/sparse.xyz")}
    };

    QJsonObject reportRecord;
    reportRecord[QStringLiteral("path")] = QStringLiteral("/tmp/reports/quality.json");

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{image0, image1, image2};
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{atRecord};
    meta[QStringLiteral("depth_map_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("result_type"), QStringLiteral("mvs_depth")}}
    };
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{QJsonObject{}};
    meta[QStringLiteral("model_results")] = QJsonArray{QJsonObject{}};
    meta[QStringLiteral("dem_results")] = QJsonArray{QJsonObject{}};
    meta[QStringLiteral("ortho_results")] = QJsonArray{QJsonObject{}};
    meta[QStringLiteral("reference_datasets")] = QJsonArray{QJsonObject{}};
    meta[QStringLiteral("report_results")] = QJsonArray{reportRecord};
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStringList sections;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item)
        {
            sections.append(item->text());
        }
    }

    EXPECT_TRUE(sections.contains(QStringLiteral("图像 (2/3 对齐)")));
    EXPECT_TRUE(sections.contains(QStringLiteral("连接点 (1,873个点)")));
    EXPECT_TRUE(sections.contains(QStringLiteral("报告 (1)")));
    EXPECT_FALSE(sections.join(QLatin1Char('|')).contains(QStringLiteral("观测网络")));
    EXPECT_FALSE(sections.join(QLatin1Char('|')).contains(QStringLiteral("深度图")));
    EXPECT_FALSE(sections.join(QLatin1Char('|')).contains(QStringLiteral("稠密点云")));
    EXPECT_FALSE(sections.join(QLatin1Char('|')).contains(QStringLiteral("3D模型")));
    EXPECT_FALSE(sections.join(QLatin1Char('|')).contains(QStringLiteral("DEM")));
    EXPECT_FALSE(sections.join(QLatin1Char('|')).contains(QStringLiteral("正射影像")));
    EXPECT_FALSE(sections.join(QLatin1Char('|')).contains(QStringLiteral("参考数据")));
}

TEST(DataTreeWidgetTest, ShowsCurrentMaskedPhotoCountAndHidesEmptyMaskSection)
{
    DataTreeWidget tree;

    QJsonObject firstImage{
        {QStringLiteral("path"), QStringLiteral("/tmp/images/image_001.jpg")},
        {QStringLiteral("mask_path"), QStringLiteral("/tmp/masks/image_001_mask.png")}
    };
    QJsonObject secondImage{
        {QStringLiteral("path"), QStringLiteral("/tmp/images/image_002.jpg")}
    };
    const QJsonObject thirdImage{
        {QStringLiteral("path"), QStringLiteral("/tmp/images/image_003.jpg")},
        {QStringLiteral("mask_path"), QStringLiteral("   ")}
    };

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    auto findMaskSection = [model]() -> QStandardItem *
    {
        for (int row = 0; row < model->rowCount(); ++row)
        {
            QStandardItem *item = model->item(row, 0);
            if (item && item->text().startsWith(QStringLiteral("掩膜")))
            {
                return item;
            }
        }
        return nullptr;
    };

    tree.loadFromJson(QJsonObject{
        {QStringLiteral("images"), QJsonArray{firstImage, secondImage, thirdImage}}
    });
    QStandardItem *maskSection = findMaskSection();
    ASSERT_NE(maskSection, nullptr);
    EXPECT_EQ(maskSection->text(), QStringLiteral("掩膜 (1)"));
    EXPECT_EQ(maskSection->rowCount(), 0);
    EXPECT_FALSE(model->hasChildren(maskSection->index()));
    EXPECT_FALSE(maskSection->icon().isNull());

    secondImage[QStringLiteral("mask_path")] = QStringLiteral("/tmp/masks/image_002_mask.png");
    tree.loadFromJson(QJsonObject{
        {QStringLiteral("images"), QJsonArray{firstImage, secondImage, thirdImage}}
    });
    maskSection = findMaskSection();
    ASSERT_NE(maskSection, nullptr);
    EXPECT_EQ(maskSection->text(), QStringLiteral("掩膜 (2)"));

    firstImage.remove(QStringLiteral("mask_path"));
    secondImage.remove(QStringLiteral("mask_path"));
    tree.loadFromJson(QJsonObject{
        {QStringLiteral("images"), QJsonArray{firstImage, secondImage, thirdImage}}
    });
    EXPECT_EQ(findMaskSection(), nullptr);
}

TEST(DataTreeWidgetTest, ShowsOnlyLatestTiePointAsTopLevelLeaf)
{
    DataTreeWidget tree;
    const QString oldPath = QStringLiteral("C:/project/tie_points/old.ply");
    const QString currentPath = QStringLiteral("C:/project/tie_points/current.ply");
    const QJsonObject oldRecord{
        {QStringLiteral("sparse_point_count"), 100},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), oldPath}}}
    };
    const QJsonObject currentRecord{
        {QStringLiteral("sparse_point_count"), 2314},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), currentPath}}}
    };
    tree.loadFromJson(QJsonObject{
        {QStringLiteral("aerial_triangulation_results"), QJsonArray{oldRecord, currentRecord}}
    });

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *tiePoints = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("连接点")))
        {
            ASSERT_EQ(tiePoints, nullptr);
            tiePoints = item;
        }
    }

    ASSERT_NE(tiePoints, nullptr);
    EXPECT_EQ(tiePoints->text(), QStringLiteral("连接点 (2,314个点)"));
    EXPECT_EQ(tiePoints->rowCount(), 0);
    EXPECT_FALSE(model->hasChildren(tiePoints->index()));
    EXPECT_FALSE(tiePoints->icon().isNull());
    EXPECT_EQ(model->item(tiePoints->row(), 1)->text(), currentPath);
    EXPECT_EQ(model->item(tiePoints->row(), 2)->text(), QStringLiteral("generated"));
}

TEST(DataTreeWidgetTest, ActivatesTopLevelTiePointLeaf)
{
    DataTreeWidget tree;
    const QString currentPath = QStringLiteral("C:/project/tie_points/current.ply");
    const QJsonObject currentRecord{
        {QStringLiteral("sparse_point_count"), 2314},
        {QStringLiteral("files"),
         QJsonObject{{QStringLiteral("sparse_cloud_xyz"), currentPath}}}
    };
    tree.loadFromJson(QJsonObject{
        {QStringLiteral("aerial_triangulation_results"), QJsonArray{currentRecord}}
    });

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->rowCount(), 1);
    const QModelIndex tiePointIndex = model->index(0, 0);

    QSignalSpy resourceSpy(&tree, &DataTreeWidget::resourceActivated);
    ASSERT_TRUE(QMetaObject::invokeMethod(view,
                                          "activated",
                                          Qt::DirectConnection,
                                          Q_ARG(QModelIndex, tiePointIndex)));

    ASSERT_EQ(resourceSpy.count(), 1);
    const QList<QVariant> args = resourceSpy.takeFirst();
    ASSERT_EQ(args.size(), 2);
    EXPECT_EQ(args.at(0).toString(), QStringLiteral("连接点"));
    EXPECT_EQ(args.at(1).toString(), currentPath);
}

TEST(ProjectResourceCleanupServiceTest, DeletesAllDepthLevelsWithoutDeletingSourcePhotos)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = tempDir.filePath(QStringLiteral("depth_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("depth_cleanup")));

    const QString sourcePhoto = tempDir.filePath(QStringLiteral("source.png"));
    const QString finalDepth = tempDir.filePath(QStringLiteral("depth_0.bin"));
    const QString finalMask = tempDir.filePath(QStringLiteral("depth_0_mask.png"));
    const QString level2Depth = tempDir.filePath(QStringLiteral("depth_0_level_2.bin"));
    const QString level2Support = tempDir.filePath(QStringLiteral("depth_0_level_2_support.bin"));
    const QString level2Uncertainty = tempDir.filePath(QStringLiteral("depth_0_level_2_uncertainty.bin"));
    const QString level2Mask = tempDir.filePath(QStringLiteral("depth_0_level_2_mask.png"));
    const QString level2Preview = tempDir.filePath(QStringLiteral("depth_0_level_2.png"));
    for (const QString &path : {sourcePhoto,
                                finalDepth,
                                finalMask,
                                level2Depth,
                                level2Support,
                                level2Uncertainty,
                                level2Mask,
                                level2Preview})
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write("artifact"), 0);
    }

    const QJsonObject record{
        {QStringLiteral("ref_image"), sourcePhoto},
        {QStringLiteral("raw_depth_path"), finalDepth},
        {QStringLiteral("valid_mask_path"), finalMask},
        {QStringLiteral("pyramid_levels"),
         QJsonArray{QJsonObject{{QStringLiteral("level"), 2},
                                {QStringLiteral("raw_depth_path"), level2Depth},
                                {QStringLiteral("raw_support_count_path"), level2Support},
                                {QStringLiteral("raw_uncertainty_path"), level2Uncertainty},
                                {QStringLiteral("valid_mask_path"), level2Mask},
                                {QStringLiteral("preview_path"), level2Preview}}}}};
    QJsonObject metadata = projectData.metadata();
    metadata[QStringLiteral("depth_map_results")] = QJsonArray{record};
    projectData.updateMetadata(metadata, false);

    const auto result = xjw::gui::project::ProjectResourceCleanupService::cleanupGeneratedData(
        &projectData,
        QStringLiteral("深度图"),
        QStringList{finalDepth});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.removedCount, 1);
    EXPECT_TRUE(projectData.metadata().value(QStringLiteral("depth_map_results")).toArray().isEmpty());
    EXPECT_TRUE(QFileInfo::exists(sourcePhoto));
    EXPECT_FALSE(QFileInfo::exists(finalDepth));
    EXPECT_FALSE(QFileInfo::exists(finalMask));
    EXPECT_FALSE(QFileInfo::exists(level2Depth));
    EXPECT_FALSE(QFileInfo::exists(level2Support));
    EXPECT_FALSE(QFileInfo::exists(level2Uncertainty));
    EXPECT_FALSE(QFileInfo::exists(level2Mask));
    EXPECT_FALSE(QFileInfo::exists(level2Preview));
}

TEST(TiePointResultIntegrationTest, ProjectManagerRoutesTiePointDeletionToDedicatedService)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int methodStart = source.indexOf(QStringLiteral("void ProjectManager::deleteGeneratedData"));
    const int methodEnd = source.indexOf(QStringLiteral("void ProjectManager::packResource"), methodStart);
    ASSERT_GE(methodStart, 0);
    ASSERT_GT(methodEnd, methodStart);
    const QString method = source.mid(methodStart, methodEnd - methodStart);
    EXPECT_TRUE(method.contains(QStringLiteral("ProjectTiePointResultService::deleteAll")));
}

TEST(ProjectMetadataOperationsTest, ResolveLatestDenseCloudPrefersCleanedProductionCloudForMeshing)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_select.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("dense_select")));

    const QString cleanedPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud_refined.ply"));
    const QString rawPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    QFile cleanedFile(cleanedPath);
    ASSERT_TRUE(cleanedFile.open(QIODevice::WriteOnly));
    cleanedFile.write("cleaned");
    cleanedFile.close();
    QFile rawFile(rawPath);
    ASSERT_TRUE(rawFile.open(QIODevice::WriteOnly));
    rawFile.write("raw");
    rawFile.close();

    QJsonObject cleanedRecord;
    cleanedRecord[QStringLiteral("dense_cloud_xyz")] = cleanedPath;
    cleanedRecord[QStringLiteral("point_count")] = 900;
    cleanedRecord[QStringLiteral("quality_stage")] = QStringLiteral("cleaned");
    cleanedRecord[QStringLiteral("operation")] = QStringLiteral("dense_refine");

    QJsonObject rawRecord;
    rawRecord[QStringLiteral("dense_cloud_xyz")] = rawPath;
    rawRecord[QStringLiteral("point_count")] = 1200;
    rawRecord[QStringLiteral("quality_stage")] = QStringLiteral("raw");
    rawRecord[QStringLiteral("operation")] = QStringLiteral("mvs_fusion");

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{cleanedRecord, rawRecord};
    projectData.updateMetadata(meta, true);

    QString selectedPath;
    QString error;
    ASSERT_TRUE(xjw::gui::project::resolveLatestDenseCloudPath(&projectData,
                                                               &selectedPath,
                                                               &error)) << error.toStdString();
    EXPECT_EQ(QDir::cleanPath(selectedPath), QDir::cleanPath(cleanedPath));
}

TEST(ProjectMetadataOperationsTest, ResolveLatestDenseCloudUsesStageWhenSelectingRefinedCloudForMeshing)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_stage_select.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("dense_stage_select")));

    const QString refinedPath = QDir(tempDir.path()).filePath(QStringLiteral("candidate_a.ply"));
    const QString laterRawPath = QDir(tempDir.path()).filePath(QStringLiteral("candidate_b.ply"));
    QFile refinedFile(refinedPath);
    ASSERT_TRUE(refinedFile.open(QIODevice::WriteOnly));
    refinedFile.write("refined");
    refinedFile.close();
    QFile rawFile(laterRawPath);
    ASSERT_TRUE(rawFile.open(QIODevice::WriteOnly));
    rawFile.write("raw");
    rawFile.close();

    QJsonObject refinedRecord;
    refinedRecord[QStringLiteral("dense_cloud_xyz")] = refinedPath;
    refinedRecord[QStringLiteral("stage")] = QStringLiteral("refined");
    refinedRecord[QStringLiteral("operation")] = QStringLiteral("mvs_output");

    QJsonObject laterRawRecord;
    laterRawRecord[QStringLiteral("dense_cloud_xyz")] = laterRawPath;
    laterRawRecord[QStringLiteral("stage")] = QStringLiteral("raw");
    laterRawRecord[QStringLiteral("operation")] = QStringLiteral("mvs_output");

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{refinedRecord, laterRawRecord};
    projectData.updateMetadata(meta, true);

    QString selectedPath;
    QString error;
    ASSERT_TRUE(xjw::gui::project::resolveLatestDenseCloudPath(&projectData,
                                                               &selectedPath,
                                                               &error)) << error.toStdString();
    EXPECT_EQ(QDir::cleanPath(selectedPath), QDir::cleanPath(refinedPath));
}

TEST(ProjectMetadataOperationsTest, ResolveLatestDenseCloudPrefersProductionTerrainCloudOverLaterDebugRefinedCloud)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_production_select.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("dense_production_select")));

    const QString productionPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud_production.ply"));
    const QString laterDebugRefinedPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud_refined_debug.ply"));
    QFile productionFile(productionPath);
    ASSERT_TRUE(productionFile.open(QIODevice::WriteOnly));
    productionFile.write("production");
    productionFile.close();
    QFile debugFile(laterDebugRefinedPath);
    ASSERT_TRUE(debugFile.open(QIODevice::WriteOnly));
    debugFile.write("debug refined");
    debugFile.close();

    QJsonObject productionRecord;
    productionRecord[QStringLiteral("dense_cloud_xyz")] = productionPath;
    productionRecord[QStringLiteral("stage")] = QStringLiteral("production");
    productionRecord[QStringLiteral("quality_stage")] = QStringLiteral("terrain");
    productionRecord[QStringLiteral("operation")] = QStringLiteral("dense_cloud_surface_cleanup");
    productionRecord[QStringLiteral("point_count")] = 300000000;

    QJsonObject laterDebugRefinedRecord;
    laterDebugRefinedRecord[QStringLiteral("dense_cloud_xyz")] = laterDebugRefinedPath;
    laterDebugRefinedRecord[QStringLiteral("stage")] = QStringLiteral("refined");
    laterDebugRefinedRecord[QStringLiteral("quality_stage")] = QStringLiteral("debug");
    laterDebugRefinedRecord[QStringLiteral("operation")] = QStringLiteral("dense_refine");
    laterDebugRefinedRecord[QStringLiteral("point_count")] = 310000000;

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{productionRecord, laterDebugRefinedRecord};
    projectData.updateMetadata(meta, true);

    QString selectedPath;
    QString error;
    ASSERT_TRUE(xjw::gui::project::resolveLatestDenseCloudPath(&projectData,
                                                               &selectedPath,
                                                               &error)) << error.toStdString();
    EXPECT_EQ(QDir::cleanPath(selectedPath), QDir::cleanPath(productionPath));
}

TEST(DataTreeWidgetTest, ResultOnlyMetadataMaterializesOneDepthSummary)
{
    DataTreeWidget tree;

    QJsonObject image;
    image[QStringLiteral("path")] = QStringLiteral("/tmp/ref_image_001.jpg");
    image[QStringLiteral("storage")] = QStringLiteral("reference");

    QJsonArray images;
    images.append(image);

    QJsonObject initialMeta;
    initialMeta[QStringLiteral("images")] = images;
    tree.loadFromJson(initialMeta);

    QJsonObject depthRecord;
    depthRecord[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
    depthRecord[QStringLiteral("depth_png")] = QStringLiteral("/tmp/mvs_output/depth_0.png");
    depthRecord[QStringLiteral("raw_depth_path")] = QStringLiteral("/tmp/mvs_output/depth_0.bin");
    depthRecord[QStringLiteral("ref_image")] = QStringLiteral("/tmp/ref_image_001.jpg");
    depthRecord[QStringLiteral("grid_width")] = 6000;
    depthRecord[QStringLiteral("grid_height")] = 4000;

    QJsonObject oldDepthRecord = depthRecord;
    oldDepthRecord[QStringLiteral("depth_png")] = QStringLiteral("/tmp/mvs_output/depth_0_old.png");
    oldDepthRecord[QStringLiteral("raw_depth_path")] = QStringLiteral("/tmp/mvs_output/depth_0_old.bin");

    QJsonArray depthResults;
    depthResults.append(oldDepthRecord);
    depthResults.append(depthRecord);

    QJsonObject resultOnlyMeta;
    resultOnlyMeta[QStringLiteral("depth_map_results")] = depthResults;
    tree.loadFromJson(resultOnlyMeta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *depthSummary = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("深度图")))
        {
            ASSERT_EQ(depthSummary, nullptr);
            depthSummary = item;
        }
    }
    ASSERT_NE(depthSummary, nullptr);
    EXPECT_TRUE(depthSummary->text().startsWith(QStringLiteral("深度图（1")));
    EXPECT_EQ(depthSummary->rowCount(), 0);
    EXPECT_FALSE(model->hasChildren(depthSummary->index()));
    EXPECT_EQ(depthSummary->data(Qt::UserRole + 3).toStringList().size(), 2);
}

TEST(DataTreeWidgetTest, DepthDiagnosticsCreateOnlyOneAggregateWorkspaceResource)
{
    DataTreeWidget tree;

    QJsonObject depthRecord;
    depthRecord[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
    depthRecord[QStringLiteral("depth_png")] = QStringLiteral("/tmp/mvs_output/depth_7.png");
    depthRecord[QStringLiteral("grid_width")] = 1600;
    depthRecord[QStringLiteral("grid_height")] = 1200;
    depthRecord[QStringLiteral("device")] = QStringLiteral("GPU");
    depthRecord[QStringLiteral("scene_profile")] = QStringLiteral("orbital_object");
    depthRecord[QStringLiteral("filter_mode")] = QStringLiteral("mild");
    depthRecord[QStringLiteral("quality_profile")] = QStringLiteral("highest");
    depthRecord[QStringLiteral("acceptance")] = QStringLiteral("accepted");
    depthRecord[QStringLiteral("pyramid_levels")] = QJsonArray{
        QJsonObject{{QStringLiteral("level"), 3}},
        QJsonObject{{QStringLiteral("level"), 2}},
        QJsonObject{{QStringLiteral("level"), 1}}
    };

    QJsonObject meta;
    meta[QStringLiteral("depth_map_results")] = QJsonArray{depthRecord};
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    ASSERT_EQ(model->rowCount(), 1);
    QStandardItem *depthSummary = model->item(0, 0);
    ASSERT_NE(depthSummary, nullptr);
    EXPECT_TRUE(depthSummary->text().startsWith(QStringLiteral("深度图（1")));
    EXPECT_TRUE(depthSummary->text().contains(QStringLiteral("超高质量")));
    EXPECT_TRUE(depthSummary->text().contains(QStringLiteral("轻度过滤")));
    EXPECT_EQ(depthSummary->rowCount(), 0);

    QSignalSpy imageSpy(&tree, &DataTreeWidget::imageActivated);
    QSignalSpy resourceSpy(&tree, &DataTreeWidget::resourceActivated);
    ASSERT_TRUE(QMetaObject::invokeMethod(view,
                                          "activated",
                                          Qt::DirectConnection,
                                          Q_ARG(QModelIndex, depthSummary->index())));
    EXPECT_EQ(imageSpy.count(), 0);
    EXPECT_EQ(resourceSpy.count(), 0);
}

TEST(DataTreeWidgetTest, ResourceRowsAreSortedByFileNameAscending)
{
    DataTreeWidget tree;

    QJsonArray images;
    for (const QString &path : {
             QStringLiteral("/tmp/images/image_010.jpg"),
             QStringLiteral("/tmp/images/image_002.jpg"),
             QStringLiteral("/tmp/images/image_001.jpg")})
    {
        QJsonObject image;
        image[QStringLiteral("path")] = path;
        image[QStringLiteral("storage")] = QStringLiteral("reference");
        images.append(image);
    }

    QJsonArray depthResults;
    for (const QString &path : {
             QStringLiteral("/tmp/mvs_output/depth_10.png"),
             QStringLiteral("/tmp/mvs_output/depth_2.png")})
    {
        QJsonObject depthRecord;
        depthRecord[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
        depthRecord[QStringLiteral("depth_png")] = path;
        depthRecord[QStringLiteral("raw_depth_path")] = xjw::core::project::rawDepthStoragePath(path);
        depthResults.append(depthRecord);
    }

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    meta[QStringLiteral("depth_map_results")] = depthResults;
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    auto findSection = [model](const QString &prefix) -> QStandardItem *
    {
        for (int row = 0; row < model->rowCount(); ++row)
        {
            QStandardItem *item = model->item(row, 0);
            if (item && item->text().startsWith(prefix))
            {
                return item;
            }
        }
        return nullptr;
    };

    QStandardItem *photoSection = findSection(QStringLiteral("图像 (0/3 对齐)"));
    ASSERT_NE(photoSection, nullptr);
    ASSERT_EQ(photoSection->rowCount(), 3);
    EXPECT_EQ(photoSection->child(0, 0)->text(), QStringLiteral("image_001.jpg"));
    EXPECT_EQ(photoSection->child(1, 0)->text(), QStringLiteral("image_002.jpg"));
    EXPECT_EQ(photoSection->child(2, 0)->text(), QStringLiteral("image_010.jpg"));

    QStandardItem *depthSummary = findSection(QStringLiteral("深度图"));
    ASSERT_NE(depthSummary, nullptr);
    EXPECT_EQ(depthSummary->rowCount(), 0);
}

TEST(DataTreeWidgetTest, DemSectionShowsQualityRasterProducts)
{
    DataTreeWidget tree;

    QJsonObject demRecord;
    demRecord[QStringLiteral("dem_tif")] = QStringLiteral("/tmp/terrain/products/dem.tif");
    demRecord[QStringLiteral("depth_preview_png")] = QStringLiteral("/tmp/terrain/products/depth_map.png");
    demRecord[QStringLiteral("error_path")] = QStringLiteral("/tmp/terrain/products/dem_error.tif");
    demRecord[QStringLiteral("count_path")] = QStringLiteral("/tmp/terrain/products/dem_count.tif");
    demRecord[QStringLiteral("confidence_path")] = QStringLiteral("/tmp/terrain/products/dem_confidence.tif");
    demRecord[QStringLiteral("coverage_path")] = QStringLiteral("/tmp/terrain/products/dem_coverage.tif");

    QJsonArray demResults;
    demResults.append(demRecord);

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray();
    meta[QStringLiteral("dem_results")] = demResults;
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *demSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("DEM (1)")))
        {
            demSection = item;
            break;
        }
    }

    ASSERT_NE(demSection, nullptr);
    ASSERT_EQ(demSection->rowCount(), 6);
    QSet<QString> paths;
    for (int row = 0; row < demSection->rowCount(); ++row)
    {
        paths.insert(demSection->child(row, 1)->text());
    }
    EXPECT_TRUE(paths.contains(QStringLiteral("/tmp/terrain/products/dem.tif")));
    EXPECT_TRUE(paths.contains(QStringLiteral("/tmp/terrain/products/depth_map.png")));
    EXPECT_TRUE(paths.contains(QStringLiteral("/tmp/terrain/products/dem_count.tif")));
    EXPECT_TRUE(paths.contains(QStringLiteral("/tmp/terrain/products/dem_confidence.tif")));
    EXPECT_TRUE(paths.contains(QStringLiteral("/tmp/terrain/products/dem_coverage.tif")));
    EXPECT_TRUE(paths.contains(QStringLiteral("/tmp/terrain/products/dem_error.tif")));
}

TEST(DataTreeWidgetTest, ReportResultsAppearAndSortByFileName)
{
    DataTreeWidget tree;

    QJsonArray reports;
    for (const QString &path : {
             QStringLiteral("/tmp/reports/quality_10.json"),
             QStringLiteral("/tmp/reports/quality_2.json")})
    {
        QJsonObject report;
        report[QStringLiteral("type")] = QStringLiteral("reconstruction_quality");
        report[QStringLiteral("path")] = path;
        reports.append(report);
    }

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray();
    meta[QStringLiteral("report_results")] = reports;
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *reportSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("报告 (2)")))
        {
            reportSection = item;
            break;
        }
    }

    ASSERT_NE(reportSection, nullptr);
    ASSERT_EQ(reportSection->rowCount(), 2);
    EXPECT_EQ(reportSection->child(0, 0)->text(), QStringLiteral("quality_2.json  [reconstruction_quality]"));
    EXPECT_EQ(reportSection->child(1, 0)->text(), QStringLiteral("quality_10.json  [reconstruction_quality]"));
}

TEST(DataTreeWidgetTest, ReferenceDatasetsAppearAndSortByFileName)
{
    DataTreeWidget tree;

    QJsonArray references;
    {
        QJsonObject reference;
        reference[QStringLiteral("type")] = QStringLiteral("lidar");
        reference[QStringLiteral("role")] = QStringLiteral("validation");
        reference[QStringLiteral("path")] = QStringLiteral("/tmp/reference/lidar_10.laz");
        references.append(reference);
    }
    {
        QJsonObject reference;
        reference[QStringLiteral("type")] = QStringLiteral("dem");
        reference[QStringLiteral("role")] = QStringLiteral("ba_prior");
        reference[QStringLiteral("path")] = QStringLiteral("/tmp/reference/dem_2.tif");
        references.append(reference);
    }

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray();
    meta[QStringLiteral("reference_datasets")] = references;
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *referenceSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("参考数据 (2)")))
        {
            referenceSection = item;
            break;
        }
    }

    ASSERT_NE(referenceSection, nullptr);
    ASSERT_EQ(referenceSection->rowCount(), 2);
    EXPECT_EQ(referenceSection->child(0, 0)->text(), QStringLiteral("dem_2.tif  [DEM] [BA约束]"));
    EXPECT_EQ(referenceSection->child(0, 1)->text(), QStringLiteral("/tmp/reference/dem_2.tif"));
    EXPECT_EQ(referenceSection->child(1, 0)->text(), QStringLiteral("lidar_10.laz  [LiDAR] [精度检查]"));
    EXPECT_EQ(referenceSection->child(1, 1)->text(), QStringLiteral("/tmp/reference/lidar_10.laz"));
}

TEST(ProjectFilesManagerTest, ReportResultsAreStoredAsProjectResults)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/data/ProjectFilesManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("report_results")));
}

TEST(ProjectFilesManagerTest, ReferenceDatasetsAreStoredAsProjectResults)
{
    ProjectFilesManager files;

    QJsonObject image;
    image[QStringLiteral("path")] = QStringLiteral("/tmp/images/img_001.jpg");

    QJsonObject reference;
    reference[QStringLiteral("type")] = QStringLiteral("dem");
    reference[QStringLiteral("role")] = QStringLiteral("validation");
    reference[QStringLiteral("path")] = QStringLiteral("/tmp/reference/dem_001.tif");

    QJsonObject legacyMeta;
    legacyMeta[QStringLiteral("images")] = QJsonArray{image};
    legacyMeta[QStringLiteral("reference_datasets")] = QJsonArray{reference};

    EXPECT_TRUE(ProjectFilesManager::isResultKey(QStringLiteral("reference_datasets")));

    files.setData(legacyMeta);

    EXPECT_FALSE(files.coreData().contains(QStringLiteral("reference_datasets")));
    ASSERT_TRUE(files.resultsData().contains(QStringLiteral("reference_datasets")));
    EXPECT_EQ(files.resultsData().value(QStringLiteral("reference_datasets")).toArray().size(), 1);
    EXPECT_EQ(files.data().value(QStringLiteral("reference_datasets")).toArray().at(0).toObject()
                  .value(QStringLiteral("path")).toString(),
              QStringLiteral("/tmp/reference/dem_001.tif"));
}

TEST(ProjectReferenceDatasetsTest, RegisterReferenceDatasetUpsertsByPath)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_project.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("reference_project")));

    const QString referencePath = QDir(tempDir.path()).filePath(QStringLiteral("dem_001.tif"));
    QFile referenceFile(referencePath);
    ASSERT_TRUE(referenceFile.open(QIODevice::WriteOnly));
    referenceFile.write("dem");
    referenceFile.close();

    QString error;
    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referencePath,
                                                            QStringLiteral("dem"),
                                                            QStringLiteral("validation"),
                                                            &error)) << error.toStdString();

    QJsonArray references = projectData.metadata().value(QStringLiteral("reference_datasets")).toArray();
    ASSERT_EQ(references.size(), 1);
    QJsonObject record = references.at(0).toObject();
    EXPECT_EQ(record.value(QStringLiteral("path")).toString(), QFileInfo(referencePath).absoluteFilePath());
    EXPECT_EQ(record.value(QStringLiteral("type")).toString(), QStringLiteral("dem"));
    EXPECT_EQ(record.value(QStringLiteral("role")).toString(), QStringLiteral("validation"));
    EXPECT_EQ(record.value(QStringLiteral("storage")).toString(), QStringLiteral("reference"));
    EXPECT_TRUE(record.contains(QStringLiteral("created_at")));

    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referencePath,
                                                            QStringLiteral("dem"),
                                                            QStringLiteral("ba_prior"),
                                                            &error)) << error.toStdString();

    references = projectData.metadata().value(QStringLiteral("reference_datasets")).toArray();
    ASSERT_EQ(references.size(), 1);
    record = references.at(0).toObject();
    EXPECT_EQ(record.value(QStringLiteral("role")).toString(), QStringLiteral("ba_prior"));
}

TEST(ProjectReferenceDatasetsTest, QualityReportRegistersReferenceReadiness)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_quality.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("reference_quality")));

    const QString referenceDemPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_dem.tif"));
    QFile referenceDem(referenceDemPath);
    ASSERT_TRUE(referenceDem.open(QIODevice::WriteOnly));
    referenceDem.write("dem");
    referenceDem.close();

    QString error;
    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referenceDemPath,
                                                            QStringLiteral("dem"),
                                                            QStringLiteral("validation"),
                                                            &error)) << error.toStdString();

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dem_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QDir(tempDir.path()).filePath(QStringLiteral("candidate_dem.tif"))},
                    {QStringLiteral("coverage_ratio"), 0.72}}
    };
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud.ply"))},
                    {QStringLiteral("point_count"), 1200}}
    };
    projectData.updateMetadata(meta, true);

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(
        &projectData,
        QStringLiteral("reference_quality_test"));

    ASSERT_TRUE(result.saved) << result.errorMessage.toStdString();
    EXPECT_TRUE(QFile::exists(result.jsonPath));
    EXPECT_TRUE(QFile::exists(result.csvPath));
    EXPECT_EQ(result.record.value(QStringLiteral("type")).toString(), QStringLiteral("reference_quality"));
    EXPECT_EQ(result.record.value(QStringLiteral("reference_count")).toInt(), 1);
    EXPECT_TRUE(result.record.value(QStringLiteral("comparison_available")).toBool());

    const QJsonArray reports = projectData.metadata().value(QStringLiteral("report_results")).toArray();
    ASSERT_EQ(reports.size(), 1);
    const QJsonObject reportRecord = reports.at(0).toObject();
    EXPECT_EQ(reportRecord.value(QStringLiteral("path")).toString(), result.jsonPath);
    EXPECT_EQ(reportRecord.value(QStringLiteral("csv_path")).toString(), result.csvPath);
    EXPECT_EQ(reportRecord.value(QStringLiteral("type")).toString(), QStringLiteral("reference_quality"));
}

TEST(ProjectSurveyControlTest, ImportsCsvIntoProjectMetadata)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("survey_control_project.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("survey_control_project")));

    const QString csvPath = QDir(tempDir.path()).filePath(QStringLiteral("survey_control.csv"));
    QFile csvFile(csvPath);
    ASSERT_TRUE(csvFile.open(QIODevice::WriteOnly | QIODevice::Text));
    csvFile.write("role,id,x,y,z,sigma_m,from_id,to_id,measured_m\n"
                  "control,GCP001,1,2,3,0.02,,,\n"
                  "check,CHK001,4,5,6,0.05,,,\n"
                  "scale_bar,SB001,,,,0.01,GCP001,CHK001,7.5\n");
    csvFile.close();

    const auto result = xjw::gui::project::importSurveyControlCsv(
        &projectData,
        csvPath,
        QStringLiteral(""));

    ASSERT_TRUE(result.imported) << result.errorMessage.toStdString();
    EXPECT_EQ(result.controlPointCount, 1);
    EXPECT_EQ(result.checkPointCount, 1);
    EXPECT_EQ(result.scaleBarCount, 1);

    const QJsonObject metadata = projectData.coreFilesMeta();
    EXPECT_FALSE(metadata.contains(QStringLiteral("survey_control")));
    const QJsonObject markerMetadata = metadata.value(QStringLiteral("marker_set")).toObject();
    EXPECT_EQ(markerMetadata.value(QStringLiteral("path")).toString(),
              QStringLiteral("assets/control_points/marker_set.json"));
    EXPECT_EQ(markerMetadata.value(QStringLiteral("marker_count")).toInt(), 2);
    EXPECT_EQ(markerMetadata.value(QStringLiteral("scale_bar_count")).toInt(), 1);

    const auto loaded = xjw::control_points::MarkerSetStore(xjw::common::project::ProjectIO::markerSetPath(projectPath)).load();
    ASSERT_TRUE(loaded.ok) << qPrintable(loaded.error);
    EXPECT_EQ(loaded.markerSet.markers().size(), 2u);
    EXPECT_EQ(loaded.markerSet.scaleBars().size(), 1u);

    const auto reportResult = xjw::gui::project::writeReconstructionQualityProjectReport(
        &projectData,
        QStringLiteral("survey_control_quality"));
    ASSERT_TRUE(reportResult.saved) << reportResult.errorMessage.toStdString();

    const QJsonObject reportRecord = reportResult.record;
    EXPECT_EQ(reportRecord.value(QStringLiteral("control_point_count")).toInt(), 1);
    EXPECT_EQ(reportRecord.value(QStringLiteral("check_point_count")).toInt(), 1);
    EXPECT_EQ(reportRecord.value(QStringLiteral("scale_bar_count")).toInt(), 1);
}

TEST(SurveyControlDialogTest, PopulatesTablesFromProjectMetadata)
{
    SurveyControlDialog dialog;
    dialog.setSurveyControlMetadata(QJsonObject{
        {QStringLiteral("source_path"), QStringLiteral("E:/code/test/control.csv")},
        {QStringLiteral("control_points"), QJsonArray{
            QJsonObject{{QStringLiteral("id"), QStringLiteral("GCP001")},
                        {QStringLiteral("x"), 1.0},
                        {QStringLiteral("y"), 2.0},
                        {QStringLiteral("z"), 3.0},
                        {QStringLiteral("sigma_m"), 0.02},
                        {QStringLiteral("enabled"), true}}
        }},
        {QStringLiteral("check_points"), QJsonArray{
            QJsonObject{{QStringLiteral("id"), QStringLiteral("CHK001")},
                        {QStringLiteral("x"), 4.0},
                        {QStringLiteral("y"), 5.0},
                        {QStringLiteral("z"), 6.0},
                        {QStringLiteral("residual"), QJsonObject{{QStringLiteral("total_m"), 0.08}}},
                        {QStringLiteral("enabled"), true}}
        }},
        {QStringLiteral("scale_bars"), QJsonArray{
            QJsonObject{{QStringLiteral("id"), QStringLiteral("SB001")},
                        {QStringLiteral("from_id"), QStringLiteral("GCP001")},
                        {QStringLiteral("to_id"), QStringLiteral("CHK001")},
                        {QStringLiteral("measured_m"), 7.5},
                        {QStringLiteral("sigma_m"), 0.01},
                        {QStringLiteral("enabled"), true}}
        }}
    });

    auto *summary = dialog.findChild<QLabel *>(QStringLiteral("surveyControlSummaryLabel"));
    ASSERT_NE(summary, nullptr);
    EXPECT_TRUE(summary->text().contains(QStringLiteral("控制点 1")));
    EXPECT_TRUE(summary->text().contains(QStringLiteral("检查点 1")));
    EXPECT_TRUE(summary->text().contains(QStringLiteral("比例尺 1")));

    auto *controlTable = dialog.findChild<QTableWidget *>(QStringLiteral("surveyControlPointTable"));
    auto *checkTable = dialog.findChild<QTableWidget *>(QStringLiteral("surveyCheckPointTable"));
    auto *scaleTable = dialog.findChild<QTableWidget *>(QStringLiteral("surveyScaleBarTable"));
    ASSERT_NE(controlTable, nullptr);
    ASSERT_NE(checkTable, nullptr);
    ASSERT_NE(scaleTable, nullptr);
    EXPECT_EQ(controlTable->rowCount(), 1);
    EXPECT_EQ(checkTable->rowCount(), 1);
    EXPECT_EQ(scaleTable->rowCount(), 1);
    EXPECT_EQ(controlTable->item(0, 0)->text(), QStringLiteral("GCP001"));
    EXPECT_EQ(checkTable->item(0, 0)->text(), QStringLiteral("CHK001"));
    EXPECT_EQ(scaleTable->item(0, 1)->text(), QStringLiteral("GCP001"));

    auto *importButton = dialog.findChild<QPushButton *>(QStringLiteral("surveyControlImportCsvButton"));
    ASSERT_NE(importButton, nullptr);
    EXPECT_TRUE(importButton->text().contains(QStringLiteral("导入 CSV")));
}

TEST(ProjectReferenceDatasetsTest, QualityReportComputesSameGridDemDifferenceMetrics)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_quality_dem_diff.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("reference_quality_dem_diff")));

    xjw::DemGridData referenceDem;
    referenceDem.width = 2;
    referenceDem.height = 2;
    referenceDem.minX = 0.0;
    referenceDem.minY = 0.0;
    referenceDem.stepX = 1.0;
    referenceDem.stepY = 1.0;
    referenceDem.elevation = cv::Mat(referenceDem.height, referenceDem.width, CV_32FC1);
    referenceDem.validMask = cv::Mat(referenceDem.height, referenceDem.width, CV_8UC1, cv::Scalar(255));
    referenceDem.elevation.at<float>(0, 0) = 10.0f;
    referenceDem.elevation.at<float>(0, 1) = 11.0f;
    referenceDem.elevation.at<float>(1, 0) = 12.0f;
    referenceDem.elevation.at<float>(1, 1) = 13.0f;

    xjw::DemGridData candidateDem = referenceDem;
    candidateDem.elevation = cv::Mat(referenceDem.height, referenceDem.width, CV_32FC1);
    candidateDem.elevation.at<float>(0, 0) = 11.0f;
    candidateDem.elevation.at<float>(0, 1) = 10.0f;
    candidateDem.elevation.at<float>(1, 0) = 15.0f;
    candidateDem.elevation.at<float>(1, 1) = 13.0f;
    candidateDem.validMask = cv::Mat(candidateDem.height, candidateDem.width, CV_8UC1, cv::Scalar(255));

    const QString referenceDemPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_dem.tif"));
    const QString candidateDemPath = QDir(tempDir.path()).filePath(QStringLiteral("candidate_dem.tif"));
    QString ioError;
    ASSERT_TRUE(xjw::DemDomIO::writeDemRaster(referenceDem,
                                              referenceDemPath,
                                              xjw::DemRasterFormat::Float32Tiff,
                                              &ioError)) << ioError.toStdString();
    ASSERT_TRUE(xjw::DemDomIO::writeDemRaster(candidateDem,
                                              candidateDemPath,
                                              xjw::DemRasterFormat::Float32Tiff,
                                              &ioError)) << ioError.toStdString();

    QString error;
    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referenceDemPath,
                                                            QStringLiteral("dem"),
                                                            QStringLiteral("validation"),
                                                            &error)) << error.toStdString();

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dem_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), candidateDemPath},
                    {QStringLiteral("coverage_ratio"), 1.0}}
    };
    projectData.updateMetadata(meta, true);

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(
        &projectData,
        QStringLiteral("reference_quality_dem_diff_test"));

    ASSERT_TRUE(result.saved) << result.errorMessage.toStdString();
    EXPECT_TRUE(result.record.value(QStringLiteral("comparison_available")).toBool());
    EXPECT_EQ(result.record.value(QStringLiteral("dem_valid_count")).toInt(), 4);
    EXPECT_NEAR(result.record.value(QStringLiteral("dem_mean_error_m")).toDouble(), 0.75, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("dem_rmse_m")).toDouble(), std::sqrt(11.0 / 4.0), 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("dem_p95_m")).toDouble(), 3.0, 1e-9);

    const QJsonArray reports = projectData.metadata().value(QStringLiteral("report_results")).toArray();
    ASSERT_EQ(reports.size(), 1);
    const QJsonObject reportRecord = reports.at(0).toObject();
    EXPECT_NEAR(reportRecord.value(QStringLiteral("dem_rmse_m")).toDouble(), std::sqrt(11.0 / 4.0), 1e-9);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("rmse_m")).toDouble(), std::sqrt(11.0 / 4.0), 1e-9);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("p95_distance_m")).toDouble(), 3.0, 1e-9);

    const QString differencePath = reportRecord.value(QStringLiteral("dem_difference_path")).toString();
    const QString absDifferencePath = reportRecord.value(QStringLiteral("dem_abs_difference_path")).toString();
    EXPECT_TRUE(QFileInfo::exists(differencePath)) << differencePath.toStdString();
    EXPECT_TRUE(QFileInfo::exists(absDifferencePath)) << absDifferencePath.toStdString();

    xjw::DemGridData diffGrid;
    ASSERT_TRUE(xjw::DemDomIO::readDemRaster(differencePath, &diffGrid, &ioError)) << ioError.toStdString();
    ASSERT_EQ(diffGrid.width, 2);
    ASSERT_EQ(diffGrid.height, 2);
    EXPECT_NEAR(diffGrid.elevation.at<float>(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(diffGrid.elevation.at<float>(0, 1), -1.0f, 1e-6f);
    EXPECT_NEAR(diffGrid.elevation.at<float>(1, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(diffGrid.elevation.at<float>(1, 1), 0.0f, 1e-6f);
}

TEST(ProjectReferenceDatasetsTest, QualityReportComputesPairedPointCloudAlignmentMetrics)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_quality_cloud_diff.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("reference_quality_cloud_diff")));

    const QString candidateCloudPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    const QString referenceCloudPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_cloud.ply"));
    writeMinimalPointCloudPly(candidateCloudPath, {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    });
    writeMinimalPointCloudPly(referenceCloudPath, {
        {10.0, -4.0, 1.5},
        {12.0, -4.0, 1.5},
        {10.0, 0.0, 1.5},
        {10.0, -4.0, 7.5}
    });

    QString error;
    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referenceCloudPath,
                                                            QStringLiteral("point_cloud"),
                                                            QStringLiteral("validation"),
                                                            &error)) << error.toStdString();

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), candidateCloudPath},
                    {QStringLiteral("point_count"), 4}}
    };
    projectData.updateMetadata(meta, true);

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(
        &projectData,
        QStringLiteral("reference_quality_cloud_diff_test"));

    ASSERT_TRUE(result.saved) << result.errorMessage.toStdString();
    EXPECT_TRUE(result.record.value(QStringLiteral("comparison_available")).toBool());
    EXPECT_EQ(result.record.value(QStringLiteral("cloud_pair_count")).toInt(), 4);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_alignment_scale")).toDouble(), 2.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_rmse_before_m")).toDouble(), std::sqrt(125.0), 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_rmse_m")).toDouble(), 0.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_p95_m")).toDouble(), 0.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("rmse_m")).toDouble(), 0.0, 1e-9);

    const QJsonArray reports = projectData.metadata().value(QStringLiteral("report_results")).toArray();
    ASSERT_EQ(reports.size(), 1);
    const QJsonObject reportRecord = reports.at(0).toObject();
    EXPECT_EQ(reportRecord.value(QStringLiteral("cloud_pair_count")).toInt(), 4);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("cloud_rmse_before_m")).toDouble(), std::sqrt(125.0), 1e-9);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("cloud_rmse_m")).toDouble(), 0.0, 1e-9);

    const QString begErrorsPath = reportRecord.value(QStringLiteral("cloud_beg_errors_csv_path")).toString();
    const QString endErrorsPath = reportRecord.value(QStringLiteral("cloud_end_errors_csv_path")).toString();
    const QString transformPath = reportRecord.value(QStringLiteral("cloud_transform_json_path")).toString();
    EXPECT_TRUE(QFileInfo::exists(begErrorsPath)) << begErrorsPath.toStdString();
    EXPECT_TRUE(QFileInfo::exists(endErrorsPath)) << endErrorsPath.toStdString();
    EXPECT_TRUE(QFileInfo::exists(transformPath)) << transformPath.toStdString();

    QFile transformFile(transformPath);
    ASSERT_TRUE(transformFile.open(QIODevice::ReadOnly));
    const QJsonObject transformJson = QJsonDocument::fromJson(transformFile.readAll()).object();
    EXPECT_NEAR(transformJson.value(QStringLiteral("scale")).toDouble(), 2.0, 1e-9);
    EXPECT_NEAR(transformJson.value(QStringLiteral("translation")).toObject().value(QStringLiteral("x")).toDouble(), 10.0, 1e-9);
}

TEST(ProjectReferenceDatasetsTest, QualityReportReadsUncompressedLasReferenceCloud)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_quality_las_diff.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("reference_quality_las_diff")));

    const QString candidateCloudPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    const QString referenceCloudPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_cloud.las"));
    writeMinimalPointCloudPly(candidateCloudPath, {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    });
    writeMinimalPointCloudLas(referenceCloudPath, {
        {10.0, -4.0, 1.5},
        {12.0, -4.0, 1.5},
        {10.0, 0.0, 1.5},
        {10.0, -4.0, 7.5}
    });

    QString error;
    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referenceCloudPath,
                                                            QStringLiteral("lidar"),
                                                            QStringLiteral("validation"),
                                                            &error)) << error.toStdString();

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), candidateCloudPath},
                    {QStringLiteral("point_count"), 4}}
    };
    projectData.updateMetadata(meta, true);

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(
        &projectData,
        QStringLiteral("reference_quality_las_diff_test"));

    ASSERT_TRUE(result.saved) << result.errorMessage.toStdString();
    EXPECT_TRUE(result.record.value(QStringLiteral("comparison_available")).toBool());
    EXPECT_EQ(result.record.value(QStringLiteral("cloud_pair_count")).toInt(), 4);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_alignment_scale")).toDouble(), 2.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_rmse_before_m")).toDouble(), std::sqrt(125.0), 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_rmse_m")).toDouble(), 0.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_p95_m")).toDouble(), 0.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("rmse_m")).toDouble(), 0.0, 1e-9);

    const QJsonArray reports = projectData.metadata().value(QStringLiteral("report_results")).toArray();
    ASSERT_EQ(reports.size(), 1);
    const QJsonObject reportRecord = reports.at(0).toObject();
    EXPECT_TRUE(reportRecord.value(QStringLiteral("cloud_difference_available")).toBool());
    EXPECT_EQ(reportRecord.value(QStringLiteral("cloud_pair_count")).toInt(), 4);
}

TEST(ProjectReferenceDatasetsTest, QualityReportAlignsUnpairedReferenceCloudByNearestNeighbor)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_quality_unpaired_cloud.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("reference_quality_unpaired_cloud")));

    const QString candidateCloudPath = QDir(tempDir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    const QString referenceCloudPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_cloud.ply"));
    writeMinimalPointCloudPly(candidateCloudPath, {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    });
    writeMinimalPointCloudPly(referenceCloudPath, {
        {10.0, -4.0, 1.5},
        {11.0, -4.0, 1.5},
        {10.0, -2.0, 1.5},
        {10.0, -4.0, 4.5},
        {30.0, 20.0, 10.0},
        {-15.0, 8.0, 2.0}
    });

    QString error;
    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referenceCloudPath,
                                                            QStringLiteral("point_cloud"),
                                                            QStringLiteral("validation"),
                                                            &error)) << error.toStdString();

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), candidateCloudPath},
                    {QStringLiteral("point_count"), 4}}
    };
    projectData.updateMetadata(meta, true);

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(
        &projectData,
        QStringLiteral("reference_quality_unpaired_cloud_test"));

    ASSERT_TRUE(result.saved) << result.errorMessage.toStdString();
    EXPECT_TRUE(result.record.value(QStringLiteral("cloud_difference_available")).toBool());
    EXPECT_EQ(result.record.value(QStringLiteral("cloud_alignment_method")).toString(),
              QStringLiteral("nearest_neighbor_icp"));
    EXPECT_EQ(result.record.value(QStringLiteral("cloud_pair_count")).toInt(), 4);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_alignment_scale")).toDouble(), 1.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_alignment_tx_m")).toDouble(), 10.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_alignment_ty_m")).toDouble(), -4.0, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_alignment_tz_m")).toDouble(), 1.5, 1e-9);
    EXPECT_NEAR(result.record.value(QStringLiteral("cloud_rmse_m")).toDouble(), 0.0, 1e-9);

    EXPECT_TRUE(QFileInfo::exists(result.record.value(QStringLiteral("cloud_beg_errors_csv_path")).toString()));
    EXPECT_TRUE(QFileInfo::exists(result.record.value(QStringLiteral("cloud_end_errors_csv_path")).toString()));
    EXPECT_TRUE(QFileInfo::exists(result.record.value(QStringLiteral("cloud_transform_json_path")).toString()));
}

TEST(ProjectReferenceDatasetsTest, TerrainPriorPreflightReportsBundleAdjustReadiness)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_prior.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("reference_prior")));

    const QString referenceDemPath = QDir(tempDir.path()).filePath(QStringLiteral("reference_dem.tif"));
    QFile referenceDem(referenceDemPath);
    ASSERT_TRUE(referenceDem.open(QIODevice::WriteOnly));
    referenceDem.write("dem");
    referenceDem.close();

    QString error;
    ASSERT_TRUE(xjw::gui::project::registerReferenceDataset(&projectData,
                                                            referenceDemPath,
                                                            QStringLiteral("dem"),
                                                            QStringLiteral("ba_prior"),
                                                            &error)) << error.toStdString();

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QDir(tempDir.path()).filePath(QStringLiteral("sparse_cloud.ply"))},
                    {QStringLiteral("sparse_point_count"), 42}}
    };
    projectData.updateMetadata(meta, true);

    const auto result = xjw::gui::project::writeReferenceTerrainPriorPreflightReport(
        &projectData,
        QStringLiteral("reference_prior_preflight_test"));

    ASSERT_TRUE(result.saved) << result.errorMessage.toStdString();
    EXPECT_TRUE(QFile::exists(result.jsonPath));
    EXPECT_TRUE(QFile::exists(result.csvPath));
    EXPECT_EQ(result.record.value(QStringLiteral("type")).toString(),
              QStringLiteral("reference_terrain_prior_preflight"));
    EXPECT_TRUE(result.record.value(QStringLiteral("ready")).toBool());
    EXPECT_EQ(result.record.value(QStringLiteral("ba_prior_reference_count")).toInt(), 1);

    const QJsonArray reports = projectData.metadata().value(QStringLiteral("report_results")).toArray();
    ASSERT_EQ(reports.size(), 1);
    EXPECT_EQ(reports.at(0).toObject().value(QStringLiteral("path")).toString(), result.jsonPath);
}

TEST(ProjectReferenceTerrainBaTest, AppliesReferenceDemAsBundleAdjustSoftPrior)
{
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());

    xjw::DemGridData dem;
    dem.width = 3;
    dem.height = 3;
    dem.minX = 0.0;
    dem.minY = 0.0;
    dem.stepX = 1.0;
    dem.stepY = 1.0;
    dem.elevation = cv::Mat(dem.height, dem.width, CV_32FC1, cv::Scalar(5.0f));
    dem.validMask = cv::Mat(dem.height, dem.width, CV_8UC1, cv::Scalar(255));

    const QString demPath = tmp.filePath(QStringLiteral("reference_dem.tif"));
    QString ioError;
    ASSERT_TRUE(xjw::DemDomIO::writeDemRaster(dem, demPath, xjw::DemRasterFormat::Float32Tiff, &ioError))
        << ioError.toStdString();

    std::vector<xjw::BATrack> tracks(2);
    tracks[0].initialPoint = {{1.0, 1.0, 5.1}};
    tracks[1].initialPoint = {{1.0, 1.0, 8.0}};

    xjw::gui::BaServiceOptions options;
    options.enableReferenceTerrainPrior = true;
    options.referenceTerrainDemPath = demPath;
    options.referenceTerrainSigmaMeters = 0.25;
    options.referenceTerrainMaxAssociationDistanceMeters = 0.5;
    options.referenceTerrainHuberDeltaMeters = 0.3;

    const auto result = xjw::gui::project::applyReferenceTerrainPriorToBundleAdjust(&tracks, &options);

    EXPECT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_TRUE(result.summary.value(QStringLiteral("enabled")).toBool());
    EXPECT_EQ(result.summary.value(QStringLiteral("source_type")).toString(), QStringLiteral("DEM"));
    EXPECT_EQ(result.summary.value(QStringLiteral("input_tracks")).toInt(), 2);
    EXPECT_EQ(result.summary.value(QStringLiteral("associated_tracks")).toInt(), 1);
    EXPECT_EQ(result.summary.value(QStringLiteral("rejected_by_distance")).toInt(), 1);
    EXPECT_TRUE(options.baOpt.enableLaserPlaneConstraints);
    EXPECT_NEAR(options.baOpt.laserPlaneWeight, 4.0, 1e-9);
    ASSERT_EQ(tracks[0].laserPlaneConstraints.size(), 1);
    EXPECT_TRUE(tracks[1].laserPlaneConstraints.empty());
}

TEST(ProjectReferenceTerrainBaTest, MenuWorkflowStartsBundleAdjustWithReferenceTerrainSettings)
{
    const QString managerSource = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(managerSource.isEmpty());

    EXPECT_TRUE(managerSource.contains(QStringLiteral("void ProjectManager::prepareReferenceTerrainBundleAdjust()")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("firstReferenceDemPriorPath")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("firstReferenceLaserPriorPath")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("enable_reference_terrain_prior")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("reference_terrain_dem_path")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("enable_laser_constraints")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("laser_constraint_cloud_path")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("laser_missing_normals_as_height_planes")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("startBundleAdjustAsync(images, outputDir")));
}

TEST(ProjectWorkflowReportsTest, ReconstructionQualityReportIsRegisteredInProjectMetadata)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("quality_project.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("quality_project")));

    QJsonObject sparseQuality;
    sparseQuality[QStringLiteral("registered_image_count")] = 1;
    sparseQuality[QStringLiteral("total_image_count")] = 2;
    sparseQuality[QStringLiteral("point_count")] = 42;

    QJsonObject sfmDiagnostics;
    sfmDiagnostics[QStringLiteral("sparse_quality")] = sparseQuality;

    QJsonObject atRecord;
    atRecord[QStringLiteral("path")] = QStringLiteral("/tmp/sparse.ply");
    atRecord[QStringLiteral("sfm_diagnostics")] = sfmDiagnostics;

    QJsonObject meta = projectData.metadata();
    meta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/img_001.jpg")},
                    {QStringLiteral("registered"), true}},
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/img_002.jpg")},
                    {QStringLiteral("registered"), false}}
    };
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{atRecord};
    meta[QStringLiteral("depth_map_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("status"), QStringLiteral("completed")},
                    {QStringLiteral("valid_ratio"), 0.5}}
    };
    meta[QStringLiteral("dem_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("coverage_ratio"), 0.75}}
    };

    QJsonObject gcp;
    gcp[QStringLiteral("id")] = QStringLiteral("GCP001");
    gcp[QStringLiteral("residual")] = QJsonObject{{QStringLiteral("total_m"), 0.02}};

    QJsonObject checkpoint;
    checkpoint[QStringLiteral("id")] = QStringLiteral("CHK001");
    checkpoint[QStringLiteral("residual")] = QJsonObject{{QStringLiteral("total_m"), 0.08}};

    QJsonObject scaleBar;
    scaleBar[QStringLiteral("id")] = QStringLiteral("SB001");
    scaleBar[QStringLiteral("measured_m")] = 10.0;
    scaleBar[QStringLiteral("estimated_m")] = 10.03;

    QJsonObject surveyControl;
    surveyControl[QStringLiteral("control_points")] = QJsonArray{gcp};
    surveyControl[QStringLiteral("check_points")] = QJsonArray{checkpoint};
    surveyControl[QStringLiteral("scale_bars")] = QJsonArray{scaleBar};
    meta[QStringLiteral("survey_control")] = surveyControl;
    projectData.updateMetadata(meta, true);

    const auto result = xjw::gui::project::writeReconstructionQualityProjectReport(
        &projectData,
        QStringLiteral("quality_stage4"));

    ASSERT_TRUE(result.saved) << result.errorMessage.toStdString();
    EXPECT_TRUE(QFile::exists(result.jsonPath));
    EXPECT_TRUE(QFile::exists(result.csvPath));

    const QJsonArray reports = projectData.metadata().value(QStringLiteral("report_results")).toArray();
    ASSERT_EQ(reports.size(), 1);
    const QJsonObject reportRecord = reports.at(0).toObject();
    EXPECT_EQ(reportRecord.value(QStringLiteral("path")).toString(), result.jsonPath);
    EXPECT_EQ(reportRecord.value(QStringLiteral("csv_path")).toString(), result.csvPath);
    EXPECT_EQ(reportRecord.value(QStringLiteral("type")).toString(), QStringLiteral("reconstruction_quality"));
    EXPECT_EQ(reportRecord.value(QStringLiteral("registered_image_count")).toInt(), 1);
    EXPECT_EQ(reportRecord.value(QStringLiteral("control_point_count")).toInt(), 1);
    EXPECT_EQ(reportRecord.value(QStringLiteral("check_point_count")).toInt(), 1);
    EXPECT_EQ(reportRecord.value(QStringLiteral("scale_bar_count")).toInt(), 1);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("control_point_rmse_m")).toDouble(), 0.02, 1e-9);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("check_point_rmse_m")).toDouble(), 0.08, 1e-9);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("scale_bar_rmse_m")).toDouble(), 0.03, 1e-9);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("mvs_valid_coverage")).toDouble(), 0.5, 1e-9);
    EXPECT_NEAR(reportRecord.value(QStringLiteral("dem_coverage")).toDouble(), 0.75, 1e-9);
}

TEST(ProjectManagerQualityReportTest, PipelineStageBoundariesRefreshReconstructionQualityReport)
{
    const QString managerHeader = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString managerSource = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString denseSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString terrainSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(managerHeader.isEmpty());
    ASSERT_FALSE(managerSource.isEmpty());
    ASSERT_FALSE(denseSource.isEmpty());
    ASSERT_FALSE(terrainSource.isEmpty());

    EXPECT_TRUE(managerHeader.contains(QStringLiteral("refreshReconstructionQualityReport")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("writeReconstructionQualityProjectReport")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("ProjectManager::refreshReconstructionQualityReport")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("refreshReconstructionQualityReport();")));
    EXPECT_TRUE(denseSource.contains(QStringLiteral("_owner->refreshReconstructionQualityReport()")));
    EXPECT_TRUE(terrainSource.contains(QStringLiteral("_owner->refreshReconstructionQualityReport()")));
}

TEST(ProjectDenseReconstructionManagerContractTest, RefreshesQualityBeforeDepthToFusionTransition)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int functionStart = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"));
    const int functionEnd = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync"), functionStart);
    ASSERT_GE(functionStart, 0);
    ASSERT_GT(functionEnd, functionStart);
    const QString block = source.mid(functionStart, functionEnd - functionStart);
    const int finishedConnection = block.indexOf(QStringLiteral("&DepthMapGenerator::finished"));
    const int refresh = block.indexOf(QStringLiteral("refreshReconstructionQualityReport()"),
                                      finishedConnection);
    const int transition = block.indexOf(QStringLiteral("startFuseDepthMapsAsync(settings)"),
                                         finishedConnection);
    ASSERT_GE(finishedConnection, 0);
    ASSERT_GE(refresh, 0);
    ASSERT_GE(transition, 0);
    EXPECT_LT(refresh, transition);
}

TEST(ProjectDenseReconstructionManagerContractTest, RefreshesQualityAfterFusionRegistration)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int functionStart = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startFuseDepthMapsAsync"));
    const int functionEnd = source.indexOf(
        QStringLiteral("ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"), functionStart);
    ASSERT_GE(functionStart, 0);
    ASSERT_GT(functionEnd, functionStart);
    const QString block = source.mid(functionStart, functionEnd - functionStart);
    const int registration = block.indexOf(QStringLiteral("QStringLiteral(\"dense_cloud_results\")"));
    const int refresh = block.indexOf(QStringLiteral("refreshReconstructionQualityReport()"), registration);
    const int ready = block.indexOf(QStringLiteral("emit self->denseCloudResultReady(outputPly, pointCount)"),
                                    registration);
    ASSERT_GE(registration, 0);
    ASSERT_GE(refresh, 0);
    ASSERT_GE(ready, 0);
    EXPECT_LT(refresh, ready);
}

TEST(DataTreeWidgetTest, SelectionClickDoesNotActivateImageUntilItemActivation)
{
    DataTreeWidget tree;
    const QString imagePath = QStringLiteral("/tmp/aerial_image_001.jpg");

    QJsonObject image;
    image[QStringLiteral("path")] = imagePath;
    image[QStringLiteral("storage")] = QStringLiteral("reference");

    QJsonArray images;
    images.append(image);

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *photoSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("图像 (0/1 对齐)")))
        {
            photoSection = item;
            break;
        }
    }
    ASSERT_NE(photoSection, nullptr);
    ASSERT_EQ(photoSection->rowCount(), 1);

    const QModelIndex imageIndex = photoSection->child(0, 0)->index();
    ASSERT_TRUE(imageIndex.isValid());

    QSignalSpy imageSpy(&tree, &DataTreeWidget::imageActivated);
    QSignalSpy resourceSpy(&tree, &DataTreeWidget::resourceActivated);

    ASSERT_TRUE(QMetaObject::invokeMethod(view,
                                          "clicked",
                                          Qt::DirectConnection,
                                          Q_ARG(QModelIndex, imageIndex)));
    EXPECT_EQ(imageSpy.count(), 0);
    EXPECT_EQ(resourceSpy.count(), 0);

    ASSERT_TRUE(QMetaObject::invokeMethod(view,
                                          "activated",
                                          Qt::DirectConnection,
                                          Q_ARG(QModelIndex, imageIndex)));
    ASSERT_EQ(imageSpy.count(), 1);
    ASSERT_EQ(resourceSpy.count(), 1);
    EXPECT_EQ(imageSpy.takeFirst().at(0).toString(), imagePath);
    const QList<QVariant> resourceArgs = resourceSpy.takeFirst();
    ASSERT_EQ(resourceArgs.size(), 2);
    EXPECT_EQ(resourceArgs.at(0).toString(), QStringLiteral("照片"));
    EXPECT_EQ(resourceArgs.at(1).toString(), imagePath);
}

TEST(DataTreeWidgetTest, ContextMenuUsesSameResourcePathResolutionAsActivation)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QString DataTreeWidget::resolveResourcePath")));
    EXPECT_TRUE(source.contains(QStringLiteral("path = resolveResourcePath(path)")));
    EXPECT_TRUE(source.contains(QStringLiteral("resourceFromIndex(i, &rowSection, &rowPath)")));
    EXPECT_TRUE(source.contains(QStringLiteral("paths << rowPath")));
}

TEST(DataTreeWidgetTest, PhotoContextMenuRoutesSelectedImageToMatchViewer)
{
    const QString treeHeader = readProjectSourceFile(
        QStringLiteral("src/gui/widgets/DataTreeWidget.h"));
    const QString treeSource = readProjectSourceFile(
        QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    const QString selectorHeader = readProjectSourceFile(
        QStringLiteral("src/gui/dialogs/tie_points/MatchPairSelectorDialog.h"));
    const QString mainWindowSource = readProjectSourceFile(
        QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(treeHeader.isEmpty());
    ASSERT_FALSE(treeSource.isEmpty());
    ASSERT_FALSE(selectorHeader.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_TRUE(treeHeader.contains(QStringLiteral("void viewMatchesRequested(const QString &imagePath);")));
    EXPECT_TRUE(treeSource.contains(QStringLiteral("viewMatchesAct = menu.addAction(tr(\"查看匹配...\"))")));
    EXPECT_TRUE(treeSource.contains(QStringLiteral("sectionName == QStringLiteral(\"照片\")")));
    EXPECT_TRUE(treeSource.contains(QStringLiteral("emit viewMatchesRequested(paths.first())")));
    EXPECT_TRUE(selectorHeader.contains(QStringLiteral("void setInitialImagePath(const QString &imagePath);")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("&DataTreeWidget::viewMatchesRequested")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("dialog->setInitialImagePath(initialImagePath)")));
}

TEST(ProjectSurveyControlTest, KeepsMetadataUnchangedWhenSidecarSaveFails)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("blocked.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("blocked")));

    const QString csvPath = QDir(tempDir.path()).filePath(QStringLiteral("control.csv"));
    QFile csvFile(csvPath);
    ASSERT_TRUE(csvFile.open(QIODevice::WriteOnly | QIODevice::Text));
    csvFile.write("role,id,x,y,z\ncontrol,GCP001,1,2,3\n");
    csvFile.close();

    ASSERT_TRUE(QDir().mkpath(xjw::common::project::ProjectIO::projectAssetsDir(projectPath)));
    QFile blocker(xjw::common::project::ProjectIO::projectControlPointsDir(projectPath));
    ASSERT_TRUE(blocker.open(QIODevice::WriteOnly));
    blocker.write("not-a-directory");
    blocker.close();

    const QJsonObject before = projectData.coreFilesMeta();
    const auto result = xjw::gui::project::importSurveyControlCsv(&projectData, csvPath, {});

    EXPECT_FALSE(result.imported);
    EXPECT_FALSE(result.errorMessage.isEmpty());
    EXPECT_EQ(projectData.coreFilesMeta(), before);
    EXPECT_FALSE(projectData.coreFilesMeta().contains(QStringLiteral("survey_control")));
}

TEST(ProjectSurveyControlTest, MigratesLegacyMetadataOnceAndRemovesOldKey)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    ProjectData projectData;
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("legacy.plascan"));
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("image.png"));
    ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("legacy")));
    ASSERT_TRUE(projectData.addImages({imagePath}));

    QJsonObject metadata = projectData.coreFilesMeta();
    metadata[QStringLiteral("survey_control")] = QJsonObject{
        {QStringLiteral("control_points"), QJsonArray{
            QJsonObject{
                {QStringLiteral("id"), QStringLiteral("GCP001")},
                {QStringLiteral("x"), 1.0},
                {QStringLiteral("y"), 2.0},
                {QStringLiteral("z"), 3.0},
                {QStringLiteral("observations"), QJsonArray{
                    QJsonObject{
                        {QStringLiteral("image_path"), imagePath},
                        {QStringLiteral("u"), 10.0},
                        {QStringLiteral("v"), 20.0}
                    }
                }}
            }
        }}
    };
    projectData.updateMetadata(metadata, true);

    const auto migration = xjw::gui::project::migrateLegacySurveyControl(&projectData);

    ASSERT_TRUE(migration.imported) << qPrintable(migration.errorMessage);
    EXPECT_FALSE(projectData.coreFilesMeta().contains(QStringLiteral("survey_control")));
    EXPECT_TRUE(projectData.coreFilesMeta().contains(QStringLiteral("marker_set")));
    const auto loaded = xjw::control_points::MarkerSetStore(xjw::common::project::ProjectIO::markerSetPath(projectPath)).load();
    ASSERT_TRUE(loaded.ok) << qPrintable(loaded.error);
    ASSERT_EQ(loaded.markerSet.markers().size(), 1u);
    ASSERT_EQ(loaded.markerSet.markers()[0].projections.size(), 1u);
    EXPECT_EQ(loaded.markerSet.markers()[0].projections[0].state,
              xjw::control_points::ProjectionState::ManualPinned);
}

TEST(PhotoStripWidgetTest, ClickSelectsPhotoAndActivationOpensPhoto)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("image_001.png"));
    QImage testImage(16, 12, QImage::Format_RGB32);
    testImage.fill(QColor(40, 90, 160));
    ASSERT_TRUE(testImage.save(imagePath));

    PhotoStripWidget strip;
    QJsonArray images;
    QJsonObject image;
    image[QStringLiteral("path")] = imagePath;
    image[QStringLiteral("name")] = QFileInfo(imagePath).fileName();
    images.append(image);
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), images}});

    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    ASSERT_EQ(list->count(), 1);
    EXPECT_TRUE(list->isWrapping());
    EXPECT_EQ(list->flow(), QListView::LeftToRight);
    EXPECT_EQ(list->horizontalScrollBarPolicy(), Qt::ScrollBarAlwaysOff);
    EXPECT_NE(list->verticalScrollBarPolicy(), Qt::ScrollBarAlwaysOff);

    QSignalSpy selectedSpy(&strip, &PhotoStripWidget::photoSelected);
    QSignalSpy activatedSpy(&strip, &PhotoStripWidget::photoActivated);

    QListWidgetItem *item = list->item(0);
    ASSERT_NE(item, nullptr);
    const qint64 placeholderIconKey = item->icon().cacheKey();
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < 5000 && item->icon().cacheKey() == placeholderIconKey)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
        QThread::msleep(10);
    }
    EXPECT_NE(item->icon().cacheKey(), placeholderIconKey);

    const QPoint itemPosition = list->visualItemRect(item).center();
    QTest::mouseClick(list->viewport(), Qt::LeftButton, Qt::NoModifier, itemPosition);
    ASSERT_EQ(selectedSpy.count(), 1);
    const QString selectedPath = selectedSpy.takeFirst().at(0).toString();
    EXPECT_EQ(selectedPath, QDir::cleanPath(imagePath));
    EXPECT_EQ(activatedSpy.count(), 0);

    emit list->itemActivated(item);
    ASSERT_EQ(activatedSpy.count(), 1);
    const QString activatedPath = activatedSpy.takeFirst().at(0).toString();
    EXPECT_EQ(activatedPath, selectedPath);
}

TEST(PhotoStripWidgetTest, ProjectSwitchDiscardsOldThumbnailRequests)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));
    QImage firstImage(32, 24, QImage::Format_RGB32);
    firstImage.fill(QColor(210, 40, 40));
    QImage secondImage(32, 24, QImage::Format_RGB32);
    secondImage.fill(QColor(40, 120, 210));
    ASSERT_TRUE(firstImage.save(firstPath));
    ASSERT_TRUE(secondImage.save(secondPath));

    PhotoStripWidget strip;
    strip.setProjectPath(QDir(tempDir.path()).filePath(QStringLiteral("first.plascan")));
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), firstPath}}}}});
    strip.setProjectPath(QDir(tempDir.path()).filePath(QStringLiteral("second.plascan")));
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), secondPath}}}}});

    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    ASSERT_EQ(list->count(), 1);
    QListWidgetItem *item = list->item(0);
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(item->data(Qt::UserRole + 1).toString(), QDir::cleanPath(secondPath));

    const qint64 placeholderIconKey = item->icon().cacheKey();
    QElapsedTimer timer;
    timer.start();
    while (item->icon().cacheKey() == placeholderIconKey && timer.elapsed() < 3000)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 25);
        QThread::msleep(5);
    }
    EXPECT_NE(item->icon().cacheKey(), placeholderIconKey);
    EXPECT_EQ(list->count(), 1);
    EXPECT_EQ(list->item(0)->data(Qt::UserRole + 1).toString(), QDir::cleanPath(secondPath));
}

TEST(PhotoStripWidgetTest, ExtendedSelectionSurvivesCurrentPhotoSynchronization)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));
    QImage testImage(16, 12, QImage::Format_RGB32);
    testImage.fill(QColor(40, 90, 160));
    ASSERT_TRUE(testImage.save(firstPath));
    ASSERT_TRUE(testImage.save(secondPath));

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), firstPath}});
    images.append(QJsonObject{{QStringLiteral("path"), secondPath}});

    PhotoStripWidget strip;
    strip.resize(600, 240);
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), images}});
    strip.show();
    QCoreApplication::processEvents();
    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    ASSERT_EQ(list->count(), 2);
    EXPECT_EQ(list->selectionMode(), QAbstractItemView::ExtendedSelection);

    list->item(0)->setSelected(true);
    list->item(1)->setSelected(true);
    list->setCurrentItem(list->item(1), QItemSelectionModel::NoUpdate);
    strip.setCurrentPhoto(firstPath);

    EXPECT_TRUE(list->item(0)->isSelected());
    EXPECT_TRUE(list->item(1)->isSelected());
    EXPECT_EQ(list->currentItem(), list->item(0));
    QTest::qWait(100);
}

TEST(PhotoStripWidgetTest, CtrlAndShiftSelectionSurvivesPhotoSynchronization)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    QJsonArray images;
    QStringList imagePaths;
    for (int index = 0; index < 4; ++index)
    {
        const QString imagePath = QDir(tempDir.path()).filePath(
            QStringLiteral("image_%1.png").arg(index));
        QImage testImage(16, 12, QImage::Format_RGB32);
        testImage.fill(QColor(40 + index, 90, 160));
        ASSERT_TRUE(testImage.save(imagePath));
        imagePaths.push_back(imagePath);
        images.append(QJsonObject{{QStringLiteral("path"), imagePath}});
    }

    PhotoStripWidget strip;
    strip.resize(1000, 220);
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), images}});
    strip.show();
    QCoreApplication::processEvents();

    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    ASSERT_EQ(list->count(), 4);
    QObject::connect(&strip,
                     &PhotoStripWidget::photoSelected,
                     &strip,
                     &PhotoStripWidget::setCurrentPhoto);

    const auto clickItem = [list](int row, Qt::KeyboardModifiers modifiers = Qt::NoModifier)
    {
        const QPoint position = list->visualItemRect(list->item(row)).center();
        QTest::mouseClick(list->viewport(), Qt::LeftButton, modifiers, position);
        QCoreApplication::processEvents();
    };

    clickItem(0);
    clickItem(1, Qt::ControlModifier);
    EXPECT_TRUE(list->item(0)->isSelected());
    EXPECT_TRUE(list->item(1)->isSelected());

    clickItem(0, Qt::ControlModifier);
    EXPECT_FALSE(list->item(0)->isSelected());
    EXPECT_TRUE(list->item(1)->isSelected());

    clickItem(1);
    clickItem(3, Qt::ShiftModifier);
    EXPECT_FALSE(list->item(0)->isSelected());
    EXPECT_TRUE(list->item(1)->isSelected());
    EXPECT_TRUE(list->item(2)->isSelected());
    EXPECT_TRUE(list->item(3)->isSelected());
    QTest::qWait(100);
}

TEST(PhotoStripWidgetTest, ContextMenuRequestsMasksForSelectedPhotos)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/PhotoStripWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(source.contains(QStringLiteral("&QListWidget::customContextMenuRequested")));
    EXPECT_TRUE(source.contains(QStringLiteral("&PhotoStripWidget::showPhotoContextMenu")));

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));
    QImage testImage(16, 12, QImage::Format_RGB32);
    testImage.fill(QColor(40, 90, 160));
    ASSERT_TRUE(testImage.save(firstPath));
    ASSERT_TRUE(testImage.save(secondPath));

    PhotoStripWidget strip;
    strip.resize(600, 240);
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), firstPath}},
        QJsonObject{{QStringLiteral("path"), secondPath}},
        QJsonObject{{QStringLiteral("path"), firstPath}}
    }}});
    strip.show();
    QCoreApplication::processEvents();

    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    list->item(0)->setSelected(true);
    list->item(1)->setSelected(true);
    list->item(2)->setSelected(true);
    QSignalSpy requestSpy(&strip, &PhotoStripWidget::generateMaskRequested);

    const QPoint itemPosition = list->visualItemRect(list->item(0)).center();
    QTest::mouseClick(list->viewport(), Qt::RightButton, Qt::NoModifier, itemPosition);
    ASSERT_TRUE(QMetaObject::invokeMethod(&strip,
                                          "showPhotoContextMenu",
                                          Qt::DirectConnection,
                                          Q_ARG(QPoint, itemPosition)));
    QCoreApplication::processEvents();
    auto *menu = qobject_cast<QMenu *>(QApplication::activePopupWidget());
    ASSERT_NE(menu, nullptr);
    ASSERT_EQ(menu->actions().size(), 1);
    EXPECT_EQ(menu->actions().first()->text(), QStringLiteral("生成蒙版..."));
    menu->actions().first()->trigger();

    ASSERT_EQ(requestSpy.count(), 1);
    EXPECT_EQ(requestSpy.takeFirst().at(0).toStringList(), QStringList({firstPath, secondPath}));
    QTest::qWait(100);
}

TEST(PhotoStripWidgetTest, ContextMenuSelectsAnUnselectedClickedPhoto)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));
    QImage testImage(16, 12, QImage::Format_RGB32);
    testImage.fill(QColor(40, 90, 160));
    ASSERT_TRUE(testImage.save(firstPath));
    ASSERT_TRUE(testImage.save(secondPath));

    PhotoStripWidget strip;
    strip.resize(600, 240);
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), firstPath}},
        QJsonObject{{QStringLiteral("path"), secondPath}}
    }}});
    strip.show();
    QCoreApplication::processEvents();

    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    list->item(0)->setSelected(true);
    QSignalSpy requestSpy(&strip, &PhotoStripWidget::generateMaskRequested);

    const QPoint itemPosition = list->visualItemRect(list->item(1)).center();
    QTest::mouseClick(list->viewport(), Qt::RightButton, Qt::NoModifier, itemPosition);
    ASSERT_TRUE(QMetaObject::invokeMethod(&strip,
                                          "showPhotoContextMenu",
                                          Qt::DirectConnection,
                                          Q_ARG(QPoint, itemPosition)));
    QCoreApplication::processEvents();
    auto *menu = qobject_cast<QMenu *>(QApplication::activePopupWidget());
    ASSERT_NE(menu, nullptr);
    ASSERT_EQ(menu->actions().size(), 1);
    menu->actions().first()->trigger();

    EXPECT_FALSE(list->item(0)->isSelected());
    EXPECT_TRUE(list->item(1)->isSelected());
    ASSERT_EQ(requestSpy.count(), 1);
    EXPECT_EQ(requestSpy.takeFirst().at(0).toStringList(), QStringList({secondPath}));
    QTest::qWait(100);
}

TEST(PhotoStripWidgetTest, ThumbnailLoadingUsesSharedDisplayImageLoader)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/PhotoStripWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/PhotoStripWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("LayerImageLoader.h")));
    EXPECT_TRUE(source.contains(QStringLiteral("xjw::gui::views::loadImageForDisplay(imagePath, projectPath)")));
    EXPECT_FALSE(source.contains(QStringLiteral("QFileIconProvider")));
}

TEST(ProjectIOTest, ResolvesProjectRelativeAndAbsoluteResourcePaths)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("sample.plascan"));
    const QString relativePath = QStringLiteral("assets/images/photo.png");
    const QString absolutePath = QDir(tempDir.path()).filePath(QStringLiteral("external/photo.png"));

    EXPECT_EQ(xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, relativePath),
              QDir::cleanPath(QDir(tempDir.path()).filePath(relativePath)));
    EXPECT_EQ(xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, absolutePath),
              QDir::cleanPath(absolutePath));
}

TEST(CameraModel3DDialogTest, PlyFloatIntensityIsScaledToByteRange)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString plyPath = QDir(tempDir.path()).filePath(QStringLiteral("float_intensity.ply"));
    QFile file(plyPath);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    QTextStream stream(&file);
    stream << "ply\n"
           << "format ascii 1.0\n"
           << "element vertex 1\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property float intensity\n"
           << "end_header\n"
           << "1 2 3 0.5\n";
    stream.flush();
    file.close();

    auto cloud = plapoint::io::readPly<float>(plyPath.toStdString());
    ASSERT_TRUE(cloud != nullptr);
    ASSERT_EQ(cloud->size(), 1u);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->getValue(0, 0), 128);
    EXPECT_EQ(cloud->colors()->getValue(0, 1), 128);
    EXPECT_EQ(cloud->colors()->getValue(0, 2), 128);
}

TEST(CameraModel3DDialogTest, ObjReaderAcceptsWhitespacePrefixedTriangularMesh)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString objPath = QDir(tempDir.path()).filePath(QStringLiteral("bennu_style.obj"));
    QFile file(objPath);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream stream(&file);
    stream << " v -0.11839  0.14872  0.12645\n"
           << " v -0.11736  0.14919  0.12680\n"
           << " v -0.11584  0.14973  0.12736\n"
           << " v -0.11411  0.15040  0.12814\n"
           << " f  1 2 3\n"
           << " f  1 3 4\n";
    stream.flush();
    file.close();

    auto cloud = plapoint::io::readObj<float>(objPath.toStdString());
    ASSERT_TRUE(cloud != nullptr);
    EXPECT_EQ(cloud->size(), 4u);
    ASSERT_TRUE(cloud->hasFaces());
    ASSERT_EQ(cloud->faces()->rows(), 2);
    EXPECT_EQ(cloud->faces()->getValue(0, 0), 0);
    EXPECT_EQ(cloud->faces()->getValue(0, 2), 2);
    EXPECT_EQ(cloud->faces()->getValue(1, 2), 3);
}

TEST(CameraModel3DDialogTest, ObjReaderPreservesPerFaceTextureSeams)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString objPath = QDir(tempDir.path()).filePath(QStringLiteral("seamed.obj"));
    QFile file(objPath);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream stream(&file);
    stream << "v 0 0 0\n"
           << "v 1 0 0\n"
           << "v 1 1 0\n"
           << "v 0 1 0\n"
           << "vt 0 0\n"
           << "vt 1 0\n"
           << "vt 1 1\n"
           << "vt 0.25 0.25\n"
           << "vt 0.75 0.75\n"
           << "vt 0 1\n"
           << "f 1/1 2/2 3/3\n"
           << "f 1/4 3/5 4/6\n";
    stream.flush();
    file.close();

    auto cloud = plapoint::io::readObj<float>(objPath.toStdString());
    ASSERT_TRUE(cloud != nullptr);
    ASSERT_TRUE(cloud->hasTextureCoords());
    ASSERT_TRUE(cloud->hasFaceTextureIndices());
    ASSERT_EQ(cloud->faceTextureIndices()->rows(), 2);
    EXPECT_EQ(cloud->faceTextureIndices()->getValue(0, 0), 0);
    EXPECT_EQ(cloud->faceTextureIndices()->getValue(1, 0), 3);
    EXPECT_NE(cloud->faceTextureIndices()->getValue(0, 0),
              cloud->faceTextureIndices()->getValue(1, 0));

    const ObjRenderPreparation prepared = prepareObjRenderData(*cloud, true);
    ASSERT_TRUE(prepared.isValid());
    EXPECT_TRUE(prepared.hasTexture);
    EXPECT_EQ(prepared.vertexCount, 6);
    EXPECT_EQ(prepared.strideBytes, 11 * static_cast<int>(sizeof(float)));
    const float *renderVertices = reinterpret_cast<const float *>(prepared.vertexData.constData());
    EXPECT_FLOAT_EQ(renderVertices[6], -1.0f);
    EXPECT_FLOAT_EQ(renderVertices[7], -1.0f);
    EXPECT_FLOAT_EQ(renderVertices[8], -1.0f);
    EXPECT_FLOAT_EQ(renderVertices[9], 0.0f);
    EXPECT_FLOAT_EQ(renderVertices[10], 0.0f);
    EXPECT_FLOAT_EQ(renderVertices[3 * 11 + 9], 0.25f);
    EXPECT_FLOAT_EQ(renderVertices[3 * 11 + 10], 0.25f);
}

TEST(CameraModel3DDialogTest, ObjReaderBenchmarkUsesConfiguredModel)
{
    const QString objPath = qEnvironmentVariable("PLASCAN_OBJ_BENCHMARK_PATH");
    if (objPath.isEmpty())
    {
        GTEST_SKIP() << "PLASCAN_OBJ_BENCHMARK_PATH is not set";
    }
    ASSERT_TRUE(QFileInfo::exists(objPath));

    QElapsedTimer timer;
    timer.start();
    auto cloud = plapoint::io::readObj<float>(objPath.toStdString());
    const qint64 parseElapsedMs = timer.elapsed();

    ASSERT_TRUE(cloud != nullptr);
    ASSERT_GT(cloud->size(), 0u);
    ASSERT_TRUE(cloud->hasFaces());
    timer.restart();
    const ObjRenderPreparation prepared = prepareObjRenderData(*cloud, true);
    const qint64 prepareElapsedMs = timer.elapsed();
    ASSERT_TRUE(prepared.isValid());
    RecordProperty("obj_parse_elapsed_ms", parseElapsedMs);
    RecordProperty("obj_prepare_elapsed_ms", prepareElapsedMs);
    RecordProperty("obj_vertex_count", static_cast<qint64>(cloud->size()));
    RecordProperty("obj_face_count", static_cast<qint64>(cloud->faces()->rows()));
    std::cout << "OBJ benchmark: parse " << parseElapsedMs
              << " ms, prepare " << prepareElapsedMs << " ms, "
              << cloud->size() << " vertices, " << cloud->faces()->rows()
              << " faces\n";
}

TEST(CameraModel3DDialogTest, ObjLoadingShowsProgressOverlay)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int objStart = source.indexOf(QStringLiteral("void CameraSceneWidget::loadModelFromObj"));
    ASSERT_GE(objStart, 0);
    const int nextFunction = source.indexOf(QStringLiteral("QVector3D CameraSceneWidget::sceneCenter"), objStart);
    ASSERT_GT(nextFunction, objStart);
    const QString objBlock = source.mid(objStart, nextFunction - objStart);

    EXPECT_TRUE(objBlock.contains(QStringLiteral("正在加载 OBJ 模型")));
    EXPECT_TRUE(objBlock.contains(QStringLiteral("正在解析 OBJ 模型")));
    EXPECT_TRUE(objBlock.contains(QStringLiteral("loadObjWithMaterialTexture")));
    EXPECT_TRUE(objBlock.contains(QStringLiteral("正在上传 OBJ 模型")));
    EXPECT_TRUE(objBlock.contains(QStringLiteral("emit plyLoadProgressChanged(gen, 0")));
    const QString queuedProgress =
        QStringLiteral("QMetaObject::invokeMethod(self.data(), [self, gen, percent, statusText]()");
    EXPECT_TRUE(objBlock.contains(queuedProgress));
    EXPECT_TRUE(objBlock.contains(QStringLiteral("OBJ 模型加载失败或为空")));
}

TEST(CameraModel3DDialogTest, ObjMaterialTextureUsesFaceUvRhiPipeline)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString vertexShader = readProjectSourceFile(
        QStringLiteral("src/gui/shaders/camera_scene_textured_mesh.vert"));
    const QString fragmentShader = readProjectSourceFile(
        QStringLiteral("src/gui/shaders/camera_scene_textured_mesh.frag"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(vertexShader.isEmpty());
    ASSERT_FALSE(fragmentShader.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("hasFaceTextureIndices")));
    EXPECT_TRUE(source.contains(QStringLiteral("faceTextureIndices()->getValue")));
    EXPECT_TRUE(source.contains(QStringLiteral("ensureTexturedMeshPipeline")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawTexturedMesh")));
    EXPECT_TRUE(header.contains(QStringLiteral("RhiTexturedMeshPipelineSet")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("1.0 - aTexCoord.y")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("aColor")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("sampler2D modelTexture")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("texture(modelTexture, vTexCoord)")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("hasVertexColor")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("textureWeight")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("srgbToLinear")));
    EXPECT_FALSE(fragmentShader.contains(QStringLiteral("0.55 + 0.75 * diff")));
}

TEST(CameraModel3DDialogTest, DenseCameraScenesThrottleLabelsAndCardSize)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("maxVisibleCameraLabels")));
    EXPECT_TRUE(source.contains(QStringLiteral("_poses.size() <= maxVisibleCameraLabels")));
    EXPECT_TRUE(source.contains(QStringLiteral("cameraCardBase()")));
    EXPECT_FALSE(source.contains(QStringLiteral("const float base = qMax(0.1f, r * 0.06f);")));
}

TEST(CameraModel3DDialogTest, CompactModelsUseScaleRelativeSceneFraming)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("qMax(1.0e-4f, dists[p95] * 1.15f)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cachedRadius * 0.002f")));
    EXPECT_FALSE(source.contains(QStringLiteral("_cachedRadius = qMax(1.0f, dists[p95] * 1.15f)")));
}

TEST(CameraModel3DDialogTest, ModelViewMinimumSizeDoesNotLimitDockResizing)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("setMinimumSize(240, 160)")));
    EXPECT_TRUE(source.contains(QStringLiteral("setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding)")));
    EXPECT_FALSE(source.contains(QStringLiteral("setMinimumSize(760, 520)")));
}

TEST(CameraModel3DDialogTest, UsesCameraToWorldRotationWithoutTransposeForCards)
{
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString workspaceSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    ASSERT_FALSE(dialogSource.isEmpty());
    ASSERT_FALSE(workspaceSource.isEmpty());

    const QString combined = dialogSource + workspaceSource;
    EXPECT_FALSE(combined.contains(QStringLiteral("pose.rotation = rot.transposed();")));
    EXPECT_TRUE(combined.contains(QStringLiteral("pose.rotation = rot;")));
}

TEST(CameraModel3DDialogTest, CameraPhotoPlanesUseDepthTestedQrhiGeometry)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("ensureCameraThumbnailPipeline")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawCameraThumbnails")));
    EXPECT_TRUE(source.contains(QStringLiteral("_thumbnailPipeline.pipeline->setDepthTest(true)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_thumbnailPipeline.pipeline->setDepthWrite(true)")));
    EXPECT_FALSE(source.contains(QStringLiteral("drawImageOnCameraPlane")));
}

TEST(CameraModel3DDialogTest, LargeBinaryPlyLoadsAsBoundedStreamingPreview)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("kDefaultPreviewPlyVertices = 3'000'000")));
    EXPECT_TRUE(source.contains(QStringLiteral("kMaxPreviewPlyVertices = 5'000'000")));
    EXPECT_TRUE(source.contains(QStringLiteral("availableSystemMemoryBytes")));
    EXPECT_TRUE(source.contains(QStringLiteral("choosePreviewPlyVertexLimit")));
    EXPECT_TRUE(source.contains(QStringLiteral("kMaxPreviewPlyVertices")));
    EXPECT_TRUE(source.contains(QStringLiteral("parsePlyPreviewHeader")));
    EXPECT_TRUE(source.contains(QStringLiteral("readBinaryPlyPreview")));
    EXPECT_TRUE(source.contains(QStringLiteral("faceCount")));
    EXPECT_TRUE(source.contains(QStringLiteral("PlyPreviewProgressCallback")));
    EXPECT_TRUE(source.contains(QStringLiteral("emit plyLoadProgressChanged")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawPlyLoadProgressOverlay")));
    EXPECT_TRUE(source.contains(QStringLiteral("_plyLoadProgressPercent")));
    EXPECT_TRUE(source.contains(QStringLiteral("preview.header.vertexCount > kMaxDirectPlyVertices")));
    EXPECT_TRUE(source.contains(QStringLiteral("preview.header.faceCount == 0")));
    EXPECT_TRUE(source.contains(QStringLiteral("file.seek(recordOffset + static_cast<qint64>(i) * preview.header.vertexStride)")));
    EXPECT_TRUE(source.contains(QStringLiteral("[3D] PLY 过大，使用预览抽样")));
    EXPECT_TRUE(header.contains(QStringLiteral("plyLoadProgressChanged")));
}

TEST(CameraModel3DDialogTest, PlyLoadProgressDoesNotRegressAfterFinished)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("if (!_loading && percent < 100)")))
        << "Late queued PLY progress from the worker must not re-enable the loading overlay after the model loaded.";
    EXPECT_TRUE(source.contains(QStringLiteral("正在完整加载 PLY 点云")))
        << "Direct PLY loading should advance the overlay beyond the header parsing stage.";
}

TEST(DenseCloudRefineTest, ReportsPlaPointProcessingDeviceForGui)
{
    const QString guiSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(guiSource.isEmpty());

    EXPECT_TRUE(guiSource.contains(QStringLiteral("plapoint::ProcessingReport")));
    EXPECT_TRUE(guiSource.contains(QStringLiteral("processingDeviceLabel")));
    EXPECT_TRUE(guiSource.contains(QStringLiteral("usedDevice")));
}

TEST(DenseCloudRefineTest, NormalEstimationUsesRequestedPlaPointDevice)
{
    const QString guiSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(guiSource.isEmpty());

    EXPECT_TRUE(guiSource.contains(QStringLiteral("estimateNormals(cloud, request.normalK, request.processingDevice")));
    EXPECT_FALSE(guiSource.contains(QStringLiteral("NormalEstimation<float, plamatrix::Device::CPU> ne")));
}

TEST(SurfaceReconstructorTest, HeightGridFallbackUsesPlaPointGpuCapablePath)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mesh/SurfaceReconstructorHeightGrid.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("#include <plapoint/gpu/height_grid.h>")));
    EXPECT_TRUE(source.contains(QStringLiteral("buildHeightGridWithRequestedDevice")));
    EXPECT_TRUE(source.contains(QStringLiteral("config.preprocessingDevice")));
    EXPECT_TRUE(source.contains(QStringLiteral("plapoint::gpu::buildHeightGrid")));
}

TEST(MainMenuTest, WorkflowMenuExposesCurrentPhotogrammetryStages)
{
    QMainWindow window;
    MainMenu menu(&window);

    QMenu *workflowMenu = nullptr;
    for (QAction *action : window.menuBar()->actions())
    {
        if (action && action->text() == QStringLiteral("工作流程"))
        {
            workflowMenu = action->menu();
            break;
        }
    }
    ASSERT_NE(workflowMenu, nullptr);

    QStringList visibleActions;
    for (QAction *action : workflowMenu->actions())
    {
        if (action && !action->isSeparator())
        {
            visibleActions.append(action->text());
        }
    }

    const int aerialIndex = visibleActions.indexOf(QStringLiteral("空中三角测量..."));
    ASSERT_GE(aerialIndex, 0);
    EXPECT_FALSE(visibleActions.contains(QStringLiteral("三维重建")));
    EXPECT_FALSE(visibleActions.contains(QStringLiteral("创建密集点云")));
    EXPECT_FALSE(visibleActions.contains(QStringLiteral("生成模型")));
    EXPECT_EQ(visibleActions.count(QStringLiteral("生成模型...")), 1);

    QMenu *reconstructionMenu = nullptr;
    for (QAction *action : window.menuBar()->actions())
    {
        if (action && action->text() == QStringLiteral("重建"))
        {
            reconstructionMenu = action->menu();
            break;
        }
    }
    EXPECT_EQ(reconstructionMenu, nullptr);
    ASSERT_NE(menu.workflowAerialTriangulationAction(), nullptr);
    EXPECT_EQ(menu.workflowAerialTriangulationAction()->text(), QStringLiteral("空中三角测量..."));
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
