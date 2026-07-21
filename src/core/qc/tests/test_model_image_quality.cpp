#include "MeshTypes.h"
#include "ModelMeshRenderer.h"
#include "ModelImageMetrics.h"
#include "ModelGeometryComparator.h"
#include "ModelImageQualityEvaluator.h"

#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QTemporaryDir>

namespace
{

TEST(ModelMeshIoTest, RoundTripsVerticesFacesNormalsAndColors)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    xjw::mesh::TriMesh source;
    source.vertices = {
        {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 255, 0, 0},
        {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 0, 255, 0},
        {0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f, 0, 0, 255},
    };
    source.faces = {{{0, 1, 2}}};

    const QString path = QDir(directory.path()).filePath(QStringLiteral("triangle.ply"));
    std::string error;
    ASSERT_TRUE(source.savePLY(path.toStdString(), &error)) << error;

    xjw::mesh::TriMesh loaded;
    ASSERT_TRUE(xjw::mesh::TriMesh::loadPLY(path.toStdString(), &loaded, &error)) << error;
    ASSERT_EQ(loaded.vertices.size(), 3U);
    ASSERT_EQ(loaded.faces.size(), 1U);
    EXPECT_FLOAT_EQ(loaded.vertices[1].x, 1.0f);
    EXPECT_FLOAT_EQ(loaded.vertices[2].ny, 0.0f);
    EXPECT_EQ(loaded.vertices[1].g, 255);
    EXPECT_EQ(loaded.faces[0].v[2], 2);
}

TEST(ModelMeshRendererTest, KeepsNearestTriangleInZBuffer)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {-0.5f, -0.5f, 3.0f, 0.0f, 0.0f, -1.0f, 0, 0, 255},
        {0.5f, -0.5f, 3.0f, 0.0f, 0.0f, -1.0f, 0, 0, 255},
        {0.0f, 0.5f, 3.0f, 0.0f, 0.0f, -1.0f, 0, 0, 255},
        {-0.5f, -0.5f, 2.0f, 0.0f, 0.0f, -1.0f, 255, 0, 0},
        {0.5f, -0.5f, 2.0f, 0.0f, 0.0f, -1.0f, 255, 0, 0},
        {0.0f, 0.5f, 2.0f, 0.0f, 0.0f, -1.0f, 255, 0, 0},
    };
    mesh.faces = {{{0, 1, 2}}, {{3, 4, 5}}};

    xjw::Camera camera;
    camera.setIntrinsics(100.0, 100.0, 64.0, 64.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});

    xjw::qc::ModelMeshRenderer renderer;
    const xjw::qc::ModelRenderResult render =
        renderer.render(mesh, camera, cv::Size(128, 128));

    ASSERT_TRUE(render.ok) << render.error.toStdString();
    EXPECT_EQ(render.validMask.at<std::uint8_t>(64, 64), 255);
    const cv::Vec3b center = render.color.at<cv::Vec3b>(64, 64);
    EXPECT_LT(center[0], 20);
    EXPECT_LT(center[1], 20);
    EXPECT_GT(center[2], 235);
    EXPECT_NEAR(render.depth.at<float>(64, 64), 2.0f, 1.0e-3f);
}

TEST(ModelImageMetricsTest, MeasuresMaskOverlapAndSymmetricEdgeDistance)
{
    cv::Mat reference = cv::Mat::zeros(80, 80, CV_8UC1);
    cv::Mat rendered = cv::Mat::zeros(80, 80, CV_8UC1);
    cv::rectangle(reference, cv::Rect(10, 20, 30, 30), cv::Scalar(255), cv::FILLED);
    cv::rectangle(rendered, cv::Rect(20, 20, 30, 30), cv::Scalar(255), cv::FILLED);

    const xjw::qc::ModelViewQuality quality =
        xjw::qc::evaluateModelViewMasks(reference, rendered);

    EXPECT_NEAR(quality.silhouetteIou, 0.5, 1.0e-6);
    EXPECT_NEAR(quality.referenceCoverage, 2.0 / 3.0, 1.0e-6);
    EXPECT_NEAR(quality.floatingPixelRate, 1.0 / 3.0, 1.0e-6);
    EXPECT_GE(quality.edgeP90Pixels, 9.0);
    EXPECT_LE(quality.edgeP90Pixels, 10.5);
}

TEST(ModelImageMetricsTest, IdenticalForegroundHasPerfectAppearanceScore)
{
    cv::Mat source(64, 64, CV_8UC3);
    for (int y = 0; y < source.rows; ++y)
    {
        for (int x = 0; x < source.cols; ++x)
        {
            source.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<std::uint8_t>(x * 3),
                static_cast<std::uint8_t>(y * 3),
                static_cast<std::uint8_t>((x + y) * 2));
        }
    }
    const cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(255));

    xjw::qc::ModelViewQuality quality = xjw::qc::evaluateModelViewMasks(mask, mask);
    xjw::qc::evaluateModelViewAppearance(source, source, mask, &quality);

    ASSERT_TRUE(quality.appearanceAvailable);
    EXPECT_GT(quality.foregroundSsim, 0.999);
    EXPECT_GT(quality.foregroundPsnr, 90.0);
}

TEST(ModelImageMetricsTest, DinoMaskKeepsLargestBrightForeground)
{
    cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::rectangle(image, cv::Rect(30, 25, 40, 50), cv::Scalar(150, 160, 170), cv::FILLED);
    cv::rectangle(image, cv::Rect(3, 3, 4, 4), cv::Scalar(255, 255, 255), cv::FILLED);

    const cv::Mat mask = xjw::qc::buildDinoForegroundMask(image);

    ASSERT_FALSE(mask.empty());
    EXPECT_EQ(mask.at<std::uint8_t>(50, 50), 255);
    EXPECT_EQ(mask.at<std::uint8_t>(4, 4), 0);
    EXPECT_GT(cv::countNonZero(mask), 1800);
}

TEST(ModelImageMetricsTest, DinoMaskPreservesLargeArchitecturalOpenings)
{
    cv::Mat image = cv::Mat::zeros(160, 200, CV_8UC3);
    cv::rectangle(image, cv::Rect(30, 20, 140, 120),
                  cv::Scalar(170, 180, 190), cv::FILLED);
    cv::rectangle(image, cv::Rect(65, 50, 70, 55), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(image, cv::Rect(45, 35, 2, 2), cv::Scalar(0, 0, 0), cv::FILLED);

    const cv::Mat mask = xjw::qc::buildDinoForegroundMask(image);

    ASSERT_FALSE(mask.empty());
    EXPECT_EQ(mask.at<std::uint8_t>(75, 100), 0)
        << "A large opening must remain background for silhouette quality.";
    EXPECT_EQ(mask.at<std::uint8_t>(36, 46), 255)
        << "Small dark texture holes should still be suppressed.";
    EXPECT_EQ(mask.at<std::uint8_t>(30, 40), 255);
}

TEST(ModelImageMetricsTest, MeasuresAerialStructureEdgeShift)
{
    cv::Mat source = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat rendered = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::line(source, cv::Point(30, 10), cv::Point(30, 90),
             cv::Scalar(255, 255, 255), 3);
    cv::line(rendered, cv::Point(33, 10), cv::Point(33, 90),
             cv::Scalar(255, 255, 255), 3);
    const cv::Mat mask(100, 100, CV_8UC1, cv::Scalar(255));
    xjw::qc::ModelViewQuality quality;

    xjw::qc::evaluateModelViewStructure(source, rendered, mask, &quality);

    EXPECT_GE(quality.edgeP90Pixels, 2.5);
    EXPECT_LE(quality.edgeP90Pixels, 3.5);
}

TEST(ModelGeometryComparatorTest, ReportsLargestConnectedFaceComponent)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(7);
    mesh.vertices[0].x = 0.0f;
    mesh.vertices[1].x = 1.0f;
    mesh.vertices[2].y = 1.0f;
    mesh.vertices[3].z = 1.0f;
    mesh.vertices[4].x = 10.0f;
    mesh.vertices[5].x = 11.0f;
    mesh.vertices[6].x = 10.0f;
    mesh.vertices[6].y = 1.0f;
    mesh.faces = {{{0, 1, 2}}, {{0, 3, 1}}, {{0, 2, 3}}, {{1, 3, 2}},
                  {{4, 5, 6}}};

    const xjw::qc::ModelGeometryQuality quality =
        xjw::qc::ModelGeometryComparator::analyzeMesh(mesh);

    EXPECT_EQ(quality.componentCount, 2);
    EXPECT_NEAR(quality.largestComponentFaceRatio, 0.8, 1.0e-6);
    EXPECT_GT(quality.largestFloatingDiagonalRatio, 0.05);
}

TEST(ModelGeometryComparatorTest, ComputesSymmetricNearestNeighborDistances)
{
    const std::vector<xjw::qc::Point3D> source = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
    };
    const std::vector<xjw::qc::Point3D> reference = {
        {0.1, 0.0, 0.0},
        {1.1, 0.0, 0.0},
        {2.1, 0.0, 0.0},
    };

    const xjw::qc::ReferenceGeometryQuality quality =
        xjw::qc::ModelGeometryComparator::comparePointSets(source, reference);

    ASSERT_TRUE(quality.available);
    EXPECT_NEAR(quality.p50, 0.1, 1.0e-6);
    EXPECT_NEAR(quality.p95, 0.1, 1.0e-6);
    EXPECT_NEAR(quality.rmse, 0.1, 1.0e-6);
}

TEST(ModelGeometryComparatorTest, AppliesSimilarityAndCropsReferenceToSourceRegion)
{
    const std::vector<xjw::qc::Point3D> source = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 1.0, 0.0},
    };
    const std::vector<xjw::qc::Point3D> reference = {
        {10.0, 20.0, 5.0},
        {12.0, 20.0, 5.0},
        {10.0, 22.0, 5.0},
        {12.0, 22.0, 5.0},
        {1000.0, 1000.0, 1000.0},
    };
    xjw::qc::SimilarityTransform transform;
    transform.scale = 2.0;
    transform.translation = {10.0, 20.0, 5.0};

    const xjw::qc::ReferenceGeometryQuality quality =
        xjw::qc::ModelGeometryComparator::compareAlignedPointSets(
            source, reference, &transform, true);

    ASSERT_TRUE(quality.available) << quality.error.toStdString();
    EXPECT_EQ(quality.referencePointCount, 4U);
    EXPECT_NEAR(quality.rmse, 0.0, 1.0e-9);
    EXPECT_NEAR(quality.sourceCoverage, 1.0, 1.0e-9);
    EXPECT_NEAR(quality.referenceCoverage, 1.0, 1.0e-9);
}

TEST(ModelImageQualityEvaluatorTest, AppliesDinoHardQualityGates)
{
    xjw::qc::ModelViewQuality view;
    view.renderOk = true;
    view.referenceCoverage = 0.95;
    view.silhouetteIou = 0.92;
    view.edgeP90Pixels = 2.0;
    view.appearanceAvailable = true;
    view.foregroundSsim = 0.80;
    xjw::qc::ModelGeometryQuality geometry;
    geometry.componentCount = 1;
    geometry.largestComponentFaceRatio = 0.90;

    EXPECT_TRUE(xjw::qc::ModelImageQualityEvaluator::qualityFailures(
        xjw::qc::ModelSceneType::Dino, {view}, geometry).isEmpty());

    geometry.largestComponentFaceRatio = 0.80;
    const QStringList failures = xjw::qc::ModelImageQualityEvaluator::qualityFailures(
        xjw::qc::ModelSceneType::Dino, {view}, geometry);
    EXPECT_FALSE(failures.isEmpty());
    EXPECT_TRUE(failures.join(QLatin1Char('|')).contains(QStringLiteral("主连通分量")));
}

TEST(ModelImageQualityEvaluatorTest, WritesSyntheticViewDiagnostics)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString mesh_path = QDir(directory.path()).filePath(QStringLiteral("plane.ply"));
    const QString image_path = QDir(directory.path()).filePath(QStringLiteral("source.png"));
    const QString output_path = QDir(directory.path()).filePath(QStringLiteral("quality"));

    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {-0.5f, -0.5f, 2.0f, 0.0f, 0.0f, -1.0f, 255, 255, 255},
        {0.5f, -0.5f, 2.0f, 0.0f, 0.0f, -1.0f, 255, 255, 255},
        {0.5f, 0.5f, 2.0f, 0.0f, 0.0f, -1.0f, 255, 255, 255},
        {-0.5f, 0.5f, 2.0f, 0.0f, 0.0f, -1.0f, 255, 255, 255},
    };
    mesh.faces = {{{0, 1, 2}}, {{0, 2, 3}}};
    std::string error;
    ASSERT_TRUE(mesh.savePLY(mesh_path.toStdString(), &error)) << error;

    cv::Mat source = cv::Mat::zeros(128, 128, CV_8UC3);
    cv::rectangle(source, cv::Rect(39, 39, 51, 51), cv::Scalar(255, 255, 255), cv::FILLED);
    ASSERT_TRUE(xjw::common::io::writeImage(image_path, source));

    xjw::qc::ModelValidationView validation;
    validation.id = QStringLiteral("synthetic");
    validation.imagePath = image_path;
    validation.camera.setIntrinsics(100.0, 100.0, 64.0, 64.0);
    validation.camera.setPose({1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0},
                              {0.0, 0.0, 0.0});

    xjw::qc::ModelImageQualityOptions options;
    options.meshPath = mesh_path;
    options.outputDirectory = output_path;
    options.maximumRenderDimension = 128;
    options.sceneType = xjw::qc::ModelSceneType::Dino;
    options.validationViews = {validation};

    const xjw::qc::ModelImageQualityResult result =
        xjw::qc::ModelImageQualityEvaluator().evaluate(options);

    ASSERT_EQ(result.views.size(), 1);
    EXPECT_TRUE(result.views[0].renderOk) << result.views[0].error.toStdString();
    EXPECT_TRUE(QFileInfo::exists(QDir(output_path).filePath(
        QStringLiteral("model_quality_report.json"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(output_path).filePath(
        QStringLiteral("contact_sheet.png"))));
    const QDir comparison(QDir(output_path).filePath(QStringLiteral("comparisons/synthetic")));
    EXPECT_TRUE(QFileInfo::exists(comparison.filePath(QStringLiteral("reference_edge.png"))));
    EXPECT_TRUE(QFileInfo::exists(comparison.filePath(QStringLiteral("rendered_edge.png"))));
    EXPECT_TRUE(QFileInfo::exists(
        comparison.filePath(QStringLiteral("edge_distance_bidirectional.png"))));
    EXPECT_TRUE(QFileInfo::exists(
        comparison.filePath(QStringLiteral("edge_p90_tail_mask.png"))));
    EXPECT_TRUE(result.views[0].edgeTail.available);
}

TEST(ModelImageQualityEvaluatorTest, LoadsValidationViewsFromMvsManifest)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = QDir(directory.path()).filePath(QStringLiteral("source.png"));
    const cv::Mat source(96, 128, CV_8UC3, cv::Scalar(20, 30, 40));
    ASSERT_TRUE(xjw::common::io::writeImage(image_path, source));

    QJsonObject camera;
    camera[QStringLiteral("fx")] = 400.0;
    camera[QStringLiteral("fy")] = 410.0;
    camera[QStringLiteral("cx")] = 64.0;
    camera[QStringLiteral("cy")] = 48.0;
    camera[QStringLiteral("rotation_world_to_camera")] =
        QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    camera[QStringLiteral("translation_world_to_camera")] = QJsonArray{0.0, 0.0, 0.0};
    camera[QStringLiteral("camera_center")] = QJsonArray{0.0, 0.0, 0.0};

    QJsonObject frame;
    frame[QStringLiteral("status")] = QStringLiteral("completed");
    frame[QStringLiteral("ref_index")] = 7;
    frame[QStringLiteral("ref_image")] = image_path;
    frame[QStringLiteral("raw_depth_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7.bin"));
    frame[QStringLiteral("raw_geometry_support_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7_geometry_support.bin"));
    frame[QStringLiteral("raw_geometry_source_mask_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7_geometry_source_mask.bin"));
    frame[QStringLiteral("raw_inverse_depth_mean_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7_inverse_depth_mean.bin"));
    frame[QStringLiteral("raw_inverse_depth_spread_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7_inverse_depth_spread.bin"));
    frame[QStringLiteral("cross_view_repaired_mask_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7_cross_view_repaired_mask.png"));
    frame[QStringLiteral("valid_mask_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7_mask.png"));
    frame[QStringLiteral("support_mask_path")] = QDir(directory.path()).filePath(
        QStringLiteral("depth_7_support.png"));
    frame[QStringLiteral("acceptance")] = QStringLiteral("accepted");
    frame[QStringLiteral("fusion_eligible")] = true;
    frame[QStringLiteral("grid_width")] = 128;
    frame[QStringLiteral("grid_height")] = 96;
    frame[QStringLiteral("camera_model")] = camera;
    QFile depth(frame.value(QStringLiteral("raw_depth_path")).toString());
    ASSERT_TRUE(depth.open(QIODevice::WriteOnly));
    depth.write("depth");
    depth.close();

    const QJsonObject manifest{{QStringLiteral("frames"), QJsonArray{frame}}};
    QSaveFile manifest_file(QDir(directory.path()).filePath(QStringLiteral("mvs_manifest.json")));
    ASSERT_TRUE(manifest_file.open(QIODevice::WriteOnly));
    manifest_file.write(QJsonDocument(manifest).toJson());
    ASSERT_TRUE(manifest_file.commit());

    QString error;
    const QVector<xjw::qc::ModelValidationView> views =
        xjw::qc::ModelImageQualityEvaluator::validationViewsFromMvsWorkspace(
            directory.path(), &error);

    ASSERT_TRUE(error.isEmpty()) << error.toStdString();
    ASSERT_EQ(views.size(), 1);
    EXPECT_EQ(views[0].id, QStringLiteral("source"));
    EXPECT_EQ(views[0].imagePath, image_path);
    EXPECT_EQ(views[0].cameraWidth, 128);
    EXPECT_EQ(views[0].cameraHeight, 96);
    EXPECT_EQ(views[0].depthPath, frame.value(QStringLiteral("raw_depth_path")).toString());
    EXPECT_EQ(views[0].geometrySupportPath,
              frame.value(QStringLiteral("raw_geometry_support_path")).toString());
    EXPECT_EQ(views[0].geometrySourceMaskPath,
              frame.value(QStringLiteral("raw_geometry_source_mask_path")).toString());
    EXPECT_EQ(views[0].inverseDepthMeanPath,
              frame.value(QStringLiteral("raw_inverse_depth_mean_path")).toString());
    EXPECT_EQ(views[0].inverseDepthSpreadPath,
              frame.value(QStringLiteral("raw_inverse_depth_spread_path")).toString());
    EXPECT_EQ(views[0].crossViewRepairedMaskPath,
              frame.value(QStringLiteral("cross_view_repaired_mask_path")).toString());
    EXPECT_EQ(views[0].frameAcceptance, QStringLiteral("accepted"));
    EXPECT_TRUE(views[0].fusionEligible);
    EXPECT_DOUBLE_EQ(views[0].camera.focalX(), 400.0);
}

} // namespace
