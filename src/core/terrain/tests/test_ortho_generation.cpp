#include "DemDomIO.h"
#include "OrthoGenerationOptions.h"
#include "OrthoProjector.h"
#include "PointCloudDomGenerator.h"
#include "ProjectCameraIO.h"
#include "TerrainPipeline.h"
#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QTemporaryDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <gdal_priv.h>

#include <array>
#include <atomic>

namespace
{

xjw::DemGridData makeDemGrid()
{
    xjw::DemGridData grid;
    grid.width = 8;
    grid.height = 4;
    grid.minX = 10.0;
    grid.minY = 20.0;
    grid.stepX = 2.0;
    grid.stepY = 3.0;
    grid.elevation = cv::Mat(grid.height, grid.width, CV_32FC1, cv::Scalar(100.0f));
    grid.validMask = cv::Mat(grid.height, grid.width, CV_8UC1, cv::Scalar(255));
    return grid;
}

xjw::DemGridData makeProjectionDemGrid()
{
    xjw::DemGridData grid;
    grid.width = 5;
    grid.height = 5;
    grid.minX = 0.0;
    grid.minY = 0.0;
    grid.stepX = 1.0;
    grid.stepY = 1.0;
    grid.elevation = cv::Mat(grid.height, grid.width, CV_32FC1, cv::Scalar(0.0f));
    grid.validMask = cv::Mat(grid.height, grid.width, CV_8UC1, cv::Scalar(255));
    return grid;
}

xjw::FramePinholeCamera makeProjectionCamera(double centerZ = -10.0)
{
    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(10.0, 10.0, 32.0, 32.0);
    camera.setPose(
        std::array<double, 9>{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0},
        std::array<double, 3>{2.0, 2.0, centerZ});
    return camera;
}

xjw::OrthoImageInput writeProjectionInput(QTemporaryDir *directory,
                                          const QString &fileName,
                                          const cv::Scalar &bgr,
                                          const cv::Mat &mask = {})
{
    xjw::OrthoImageInput input;
    input.imagePath = QDir(directory->path()).filePath(fileName);
    input.camera = makeProjectionCamera();
    EXPECT_TRUE(xjw::common::io::writeImage(
        input.imagePath,
        cv::Mat(64, 64, CV_8UC3, bgr)));
    if (!mask.empty())
    {
        input.exclusionMaskPath =
            QDir(directory->path()).filePath(fileName + QStringLiteral("_mask.png"));
        EXPECT_TRUE(xjw::common::io::writeImage(input.exclusionMaskPath, mask));
    }
    return input;
}

TEST(OrthoGenerationOptionsTest, ParsesStableDefaultsAndLegacyResolution)
{
    xjw::OrthoGenerationOptions options;
    QString error;
    ASSERT_TRUE(xjw::OrthoGenerationOptions::fromJson(
        QJsonObject{{QStringLiteral("resolution"), 0.25}},
        &options,
        &error))
        << error.toStdString();

    EXPECT_EQ(options.projectionType, xjw::OrthoProjectionType::DemGrid);
    EXPECT_EQ(options.surfaceType, xjw::OrthoSurfaceType::Dem);
    EXPECT_EQ(options.colorSource, xjw::OrthoColorSource::Images);
    EXPECT_EQ(options.blendMode, xjw::OrthoBlendMode::Mosaic);
    EXPECT_EQ(options.sizingMode, xjw::OrthoSizingMode::PixelSize);
    EXPECT_DOUBLE_EQ(options.pixelSizeX, 0.25);
    EXPECT_DOUBLE_EQ(options.pixelSizeY, 0.25);
    EXPECT_FALSE(options.useProjectMasks);
}

TEST(OrthoGenerationOptionsTest, AcceptsLegacyResolutionModeToken)
{
    xjw::OrthoGenerationOptions options;
    QString error;
    ASSERT_TRUE(xjw::OrthoGenerationOptions::fromJson(
        QJsonObject{
            {QStringLiteral("resolution_mode"),
             QStringLiteral("maximum_dimension")},
            {QStringLiteral("maximum_dimension"), 2048}},
        &options,
        &error))
        << error.toStdString();
    EXPECT_EQ(options.sizingMode, xjw::OrthoSizingMode::MaximumDimension);
    EXPECT_EQ(options.maximumDimension, 2048);
}

TEST(OrthoGenerationOptionsTest, RoundTripsUserFacingSettings)
{
    const QJsonObject input{
        {QStringLiteral("projection_type"), QStringLiteral("dem_grid")},
        {QStringLiteral("surface_type"), QStringLiteral("dem")},
        {QStringLiteral("color_source"), QStringLiteral("images")},
        {QStringLiteral("blend_mode"), QStringLiteral("weighted_average")},
        {QStringLiteral("sizing_mode"), QStringLiteral("maximum_dimension")},
        {QStringLiteral("pixel_size_x"), 0.12},
        {QStringLiteral("pixel_size_y"), 0.18},
        {QStringLiteral("maximum_dimension"), 8192},
        {QStringLiteral("bounds_enabled"), true},
        {QStringLiteral("min_x"), 10.0},
        {QStringLiteral("min_y"), 20.0},
        {QStringLiteral("max_x"), 30.0},
        {QStringLiteral("max_y"), 50.0},
        {QStringLiteral("color_correction"), false},
        {QStringLiteral("sharpness_weighting"), true},
        {QStringLiteral("ghost_filter"), true},
        {QStringLiteral("fill_holes"), true},
        {QStringLiteral("hole_fill_max_area"), 128},
        {QStringLiteral("hole_fill_radius"), 2.0},
        {QStringLiteral("use_project_masks"), false},
        {QStringLiteral("maximum_pixel_count"), 25000000}
    };

    xjw::OrthoGenerationOptions options;
    QString error;
    ASSERT_TRUE(xjw::OrthoGenerationOptions::fromJson(input, &options, &error))
        << error.toStdString();

    const QJsonObject resolved = options.toResolvedJson();
    EXPECT_EQ(resolved.value(QStringLiteral("blend_mode")).toString(),
              QStringLiteral("weighted_average"));
    EXPECT_EQ(resolved.value(QStringLiteral("sizing_mode")).toString(),
              QStringLiteral("maximum_dimension"));
    EXPECT_DOUBLE_EQ(resolved.value(QStringLiteral("pixel_size_x")).toDouble(), 0.12);
    EXPECT_DOUBLE_EQ(resolved.value(QStringLiteral("pixel_size_y")).toDouble(), 0.18);
    EXPECT_TRUE(resolved.value(QStringLiteral("bounds_enabled")).toBool());
    EXPECT_DOUBLE_EQ(resolved.value(QStringLiteral("min_x")).toDouble(), 10.0);
    EXPECT_DOUBLE_EQ(resolved.value(QStringLiteral("max_y")).toDouble(), 50.0);
    EXPECT_FALSE(resolved.value(QStringLiteral("color_correction")).toBool());
    EXPECT_TRUE(resolved.value(QStringLiteral("ghost_filter")).toBool());
    EXPECT_TRUE(resolved.value(QStringLiteral("fill_holes")).toBool());
}

TEST(OrthoGenerationOptionsTest, RejectsInvalidCombinationAndInvalidBounds)
{
    xjw::OrthoGenerationOptions options;
    QString error;
    EXPECT_FALSE(xjw::OrthoGenerationOptions::fromJson(
        QJsonObject{{QStringLiteral("projection_type"), QStringLiteral("cylindrical")}},
        &options,
        &error));
    EXPECT_TRUE(error.contains(QStringLiteral("组合")));

    error.clear();
    EXPECT_FALSE(xjw::OrthoGenerationOptions::fromJson(
        QJsonObject{
            {QStringLiteral("bounds_enabled"), true},
            {QStringLiteral("min_x"), 4.0},
            {QStringLiteral("max_x"), 4.0},
            {QStringLiteral("min_y"), 1.0},
            {QStringLiteral("max_y"), 2.0}},
        &options,
        &error));
    EXPECT_TRUE(error.contains(QStringLiteral("边界")));
}

TEST(OrthoGenerationOptionsTest, AcceptsPointCloudPlanarAndGlobalModes)
{
    for (const QString &projection : {
             QStringLiteral("planar"), QStringLiteral("cylindrical")})
    {
        xjw::OrthoGenerationOptions options;
        QString error;
        ASSERT_TRUE(xjw::OrthoGenerationOptions::fromJson(
            QJsonObject{
                {QStringLiteral("projection_type"), projection},
                {QStringLiteral("surface_type"), QStringLiteral("point_cloud")},
                {QStringLiteral("color_source"), QStringLiteral("point_colors")},
                {QStringLiteral("sizing_mode"), QStringLiteral("maximum_dimension")},
                {QStringLiteral("maximum_dimension"), 512}},
            &options,
            &error)) << error.toStdString();
        EXPECT_EQ(options.surfaceType, xjw::OrthoSurfaceType::PointCloud);
        EXPECT_EQ(options.colorSource, xjw::OrthoColorSource::PointColors);
    }
}

TEST(PointCloudDomGeneratorTest, PlanarModeKeepsHighestPointColor)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(3, 3);
    points(0, 0) = 0.25f; points(0, 1) = 0.25f; points(0, 2) = 1.0f;
    points(1, 0) = 0.25f; points(1, 1) = 0.25f; points(1, 2) = 2.0f;
    points(2, 0) = 1.25f; points(2, 1) = 1.25f; points(2, 2) = 1.0f;
    xjw::PlaPointCloud cloud(std::move(points));
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(3, 3);
    colors(0, 0) = 255; colors(0, 1) = 0; colors(0, 2) = 0;
    colors(1, 0) = 0; colors(1, 1) = 255; colors(1, 2) = 0;
    colors(2, 0) = 0; colors(2, 1) = 0; colors(2, 2) = 255;
    cloud.setColors(std::move(colors));

    xjw::OrthoGenerationOptions options;
    options.projectionType = xjw::OrthoProjectionType::Planar;
    options.surfaceType = xjw::OrthoSurfaceType::PointCloud;
    options.colorSource = xjw::OrthoColorSource::PointColors;
    options.pixelSizeX = 1.0;
    options.pixelSizeY = 1.0;
    xjw::PointCloudDomResult result;
    QString error;
    ASSERT_TRUE(xjw::PointCloudDomGenerator::generate(
        cloud, options, &result, &error)) << error.toStdString();
    ASSERT_EQ(result.imageBgr.type(), CV_8UC3);
    EXPECT_EQ(result.imageBgr.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 255, 0));
    EXPECT_TRUE(result.reference.projection.projectionWkt.contains(QStringLiteral("LOCAL_CS")));
}

TEST(PointCloudDomGeneratorTest, RejectsCloudWithoutRgbColors)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(2, 3);
    points(0, 0) = 0.0f; points(0, 1) = 0.0f; points(0, 2) = 0.0f;
    points(1, 0) = 1.0f; points(1, 1) = 1.0f; points(1, 2) = 1.0f;
    xjw::PlaPointCloud cloud(std::move(points));
    xjw::OrthoGenerationOptions options;
    options.projectionType = xjw::OrthoProjectionType::Planar;
    options.surfaceType = xjw::OrthoSurfaceType::PointCloud;
    options.colorSource = xjw::OrthoColorSource::PointColors;
    QJsonObject estimate;
    QString error;
    EXPECT_FALSE(xjw::PointCloudDomGenerator::estimate(
        cloud, options, &estimate, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("RGB")));
}

TEST(PointCloudDomGeneratorTest, GlobalModeProducesFullBodyProjectedGrid)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points(0, 0) = 10.0f; points(0, 1) = 0.0f; points(0, 2) = 0.0f;
    points(1, 0) = 0.0f; points(1, 1) = 10.0f; points(1, 2) = 0.0f;
    points(2, 0) = -10.0f; points(2, 1) = 0.0f; points(2, 2) = 0.0f;
    points(3, 0) = 0.0f; points(3, 1) = -10.0f; points(3, 2) = 0.0f;
    xjw::PlaPointCloud cloud(std::move(points));
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(4, 3);
    colors.fill(128);
    cloud.setColors(std::move(colors));

    xjw::OrthoGenerationOptions options;
    options.projectionType = xjw::OrthoProjectionType::SimpleCylindrical;
    options.surfaceType = xjw::OrthoSurfaceType::PointCloud;
    options.colorSource = xjw::OrthoColorSource::PointColors;
    options.sizingMode = xjw::OrthoSizingMode::MaximumDimension;
    options.maximumDimension = 360;
    xjw::PointCloudDomResult result;
    QString error;
    ASSERT_TRUE(xjw::PointCloudDomGenerator::generate(
        cloud, options, &result, &error)) << error.toStdString();
    EXPECT_EQ(result.reference.width, 360);
    EXPECT_EQ(result.reference.height, 180);
    EXPECT_NEAR(result.resolvedOptions.referenceRadius, 10.0, 1e-6);
    EXPECT_TRUE(result.reference.projection.projectionWkt.contains(
        QStringLiteral("Equirectangular")));
    EXPECT_EQ(result.projectedPointCount, 4);

    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString outputPath =
        QDir(directory.path()).filePath(QStringLiteral("asteroid_global_dom.tif"));
    ASSERT_TRUE(xjw::DemDomIO::writeDomGeoTiff(
        result.imageBgr, result.validMask, result.reference, outputPath, &error))
        << error.toStdString();
    GDALDataset *dataset = static_cast<GDALDataset *>(
        GDALOpen(outputPath.toUtf8().constData(), GA_ReadOnly));
    ASSERT_NE(dataset, nullptr);
    EXPECT_EQ(dataset->GetRasterCount(), 4);
    EXPECT_FALSE(QString::fromUtf8(dataset->GetProjectionRef()).isEmpty());
    double geoTransform[6]{};
    EXPECT_EQ(dataset->GetGeoTransform(geoTransform), CE_None);
    EXPECT_GT(geoTransform[1], 0.0);
    EXPECT_LT(geoTransform[5], 0.0);
    GDALClose(dataset);
}

TEST(PointCloudDomPipelineTest, RoutesColoredPointCloudWithoutImagesOrDem)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString pointCloudPath =
        QDir(directory.path()).filePath(QStringLiteral("colored_cloud.ply"));
    QFile pointCloudFile(pointCloudPath);
    ASSERT_TRUE(pointCloudFile.open(QIODevice::WriteOnly | QIODevice::Text));
    pointCloudFile.write(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "0 0 1 255 0 0\n1 0 2 0 255 0\n"
        "0 1 3 0 0 255\n1 1 4 255 255 0\n");
    pointCloudFile.close();

    const QString outputPath =
        QDir(directory.path()).filePath(QStringLiteral("planar_dom.tif"));
    const QJsonObject settings{
        {QStringLiteral("projection_type"), QStringLiteral("planar")},
        {QStringLiteral("surface_type"), QStringLiteral("point_cloud")},
        {QStringLiteral("color_source"), QStringLiteral("point_colors")},
        {QStringLiteral("sizing_mode"), QStringLiteral("maximum_dimension")},
        {QStringLiteral("maximum_dimension"), 32}};
    QJsonObject result;
    QString error;
    ASSERT_TRUE(xjw::TerrainPipeline::generateOrthoProduct(
        {}, pointCloudPath, outputPath, settings, {}, &result, &error))
        << error.toStdString();
    EXPECT_TRUE(QFileInfo::exists(outputPath));
    EXPECT_EQ(result.value(QStringLiteral("algorithm_version")).toString(),
              QStringLiteral("point_cloud_dom_v1"));
    EXPECT_EQ(result.value(QStringLiteral("point_cloud_path")).toString(), pointCloudPath);
    EXPECT_TRUE(result.value(QStringLiteral("projection_wkt_present")).toBool());
}

TEST(OrthoGridPlannerTest, SupportsIndependentPixelSizeAndCustomBounds)
{
    xjw::OrthoGenerationOptions options;
    options.pixelSizeX = 2.0;
    options.pixelSizeY = 3.0;
    options.bounds.enabled = true;
    options.bounds.minX = 12.0;
    options.bounds.maxX = 18.0;
    options.bounds.minY = 21.0;
    options.bounds.maxY = 27.0;

    xjw::OrthoOutputGrid output;
    QString error;
    ASSERT_TRUE(xjw::OrthoProjector::planOutputGrid(
        makeDemGrid(), options, &output, &error))
        << error.toStdString();

    EXPECT_EQ(output.reference.width, 3);
    EXPECT_EQ(output.reference.height, 2);
    EXPECT_DOUBLE_EQ(output.reference.stepX, 2.0);
    EXPECT_DOUBLE_EQ(output.reference.stepY, 3.0);
    EXPECT_DOUBLE_EQ(output.minEdgeX, 12.0);
    EXPECT_DOUBLE_EQ(output.maxEdgeX, 18.0);
    EXPECT_DOUBLE_EQ(output.minEdgeY, 21.0);
    EXPECT_DOUBLE_EQ(output.maxEdgeY, 27.0);
}

TEST(OrthoGridPlannerTest, MaximumDimensionPreservesDemPixelAspect)
{
    xjw::OrthoGenerationOptions options;
    options.sizingMode = xjw::OrthoSizingMode::MaximumDimension;
    options.maximumDimension = 4;

    xjw::OrthoOutputGrid output;
    QString error;
    ASSERT_TRUE(xjw::OrthoProjector::planOutputGrid(
        makeDemGrid(), options, &output, &error))
        << error.toStdString();

    EXPECT_EQ(output.reference.width, 4);
    EXPECT_EQ(output.reference.height, 2);
    EXPECT_DOUBLE_EQ(output.reference.stepX, 4.0);
    EXPECT_DOUBLE_EQ(output.reference.stepY, 6.0);
}

TEST(OrthoGridPlannerTest, RejectsOutputAbovePixelBudget)
{
    xjw::OrthoGenerationOptions options;
    options.pixelSizeX = 0.01;
    options.pixelSizeY = 0.01;
    options.maximumPixelCount = 1000;

    xjw::OrthoOutputGrid output;
    QString error;
    EXPECT_FALSE(xjw::OrthoProjector::planOutputGrid(
        makeDemGrid(), options, &output, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("像素数")));
}

TEST(OrthoImageInputTest, PrefersExactPathAndRejectsAmbiguousFileName)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString firstDirectory =
        QDir(directory.path()).filePath(QStringLiteral("first"));
    const QString secondDirectory =
        QDir(directory.path()).filePath(QStringLiteral("second"));
    ASSERT_TRUE(QDir().mkpath(firstDirectory));
    ASSERT_TRUE(QDir().mkpath(secondDirectory));
    const QString firstPath =
        QDir(firstDirectory).filePath(QStringLiteral("same.png"));
    const QString secondPath =
        QDir(secondDirectory).filePath(QStringLiteral("same.png"));

    const QJsonObject projectMeta{
        {QStringLiteral("images"),
         QJsonArray{
             QJsonObject{
                 {QStringLiteral("path"), firstPath},
                 {QStringLiteral("camera"),
                  xjw::common::project::cameraToJson(
                      makeProjectionCamera(-10.0))}},
             QJsonObject{
                 {QStringLiteral("path"), secondPath},
                 {QStringLiteral("camera"),
                  xjw::common::project::cameraToJson(
                      makeProjectionCamera(-20.0))}}}}};

    std::vector<xjw::OrthoImageInput> inputs;
    QString error;
    ASSERT_TRUE(xjw::OrthoProjector::buildImageInputs(
        {secondPath}, projectMeta, &inputs, &error))
        << error.toStdString();
    ASSERT_EQ(inputs.size(), 1U);
    EXPECT_DOUBLE_EQ(inputs.front().camera.cameraCenter()[2], -20.0);

    const QString ambiguousPath =
        QDir(directory.path()).filePath(QStringLiteral("other/same.png"));
    EXPECT_FALSE(xjw::OrthoProjector::buildImageInputs(
        {ambiguousPath}, projectMeta, &inputs, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("唯一匹配")));

    const QJsonObject differentExtensionMeta{
        {QStringLiteral("images"),
         QJsonArray{
             QJsonObject{
                 {QStringLiteral("path"),
                  QDir(firstDirectory).filePath(QStringLiteral("photo.jpg"))},
                 {QStringLiteral("camera"),
                  xjw::common::project::cameraToJson(
                      makeProjectionCamera(-10.0))}}}}};
    EXPECT_FALSE(xjw::OrthoProjector::buildImageInputs(
        {QDir(secondDirectory).filePath(QStringLiteral("photo.png"))},
        differentExtensionMeta,
        &inputs,
        &error));

    const QJsonObject duplicateExactMeta{
        {QStringLiteral("images"),
         QJsonArray{
             projectMeta.value(QStringLiteral("images")).toArray().at(0),
             projectMeta.value(QStringLiteral("images")).toArray().at(0)}}};
    EXPECT_FALSE(xjw::OrthoProjector::buildImageInputs(
        {firstPath}, duplicateExactMeta, &inputs, &error));
}

TEST(OrthoPipelineEstimateTest, SeparatesDemMetadataFromPlannedOutput)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const xjw::DemGridData dem = makeDemGrid();
    const QString demPath =
        QDir(directory.path()).filePath(QStringLiteral("estimate_dem.tif"));
    QString error;
    ASSERT_TRUE(xjw::DemDomIO::writeDemRaster(
        dem, demPath, xjw::DemRasterFormat::Float32Tiff, &error))
        << error.toStdString();

    QJsonObject estimate;
    ASSERT_TRUE(xjw::TerrainPipeline::estimateOrthoProduct(
        demPath,
        QJsonObject{
            {QStringLiteral("pixel_size_x"), 1.0},
            {QStringLiteral("pixel_size_y"), 1.5}},
        &estimate,
        &error))
        << error.toStdString();

    EXPECT_DOUBLE_EQ(
        estimate.value(QStringLiteral("dem_pixel_size_x")).toDouble(), 2.0);
    EXPECT_DOUBLE_EQ(
        estimate.value(QStringLiteral("dem_pixel_size_y")).toDouble(), 3.0);
    EXPECT_DOUBLE_EQ(
        estimate.value(QStringLiteral("pixel_size_x")).toDouble(), 1.0);
    EXPECT_DOUBLE_EQ(
        estimate.value(QStringLiteral("pixel_size_y")).toDouble(), 1.5);
    EXPECT_DOUBLE_EQ(
        estimate.value(QStringLiteral("dem_min_x")).toDouble(), 9.0);
    EXPECT_DOUBLE_EQ(
        estimate.value(QStringLiteral("dem_max_x")).toDouble(), 25.0);
    EXPECT_EQ(estimate.value(QStringLiteral("width")).toInt(), 16);
    EXPECT_EQ(estimate.value(QStringLiteral("height")).toInt(), 8);
}

TEST(OrthoPipelineEstimateTest, GenerationAlsoRejectsRotatedDem)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString demPath =
        QDir(directory.path()).filePath(QStringLiteral("rotated_dem.tif"));
    QString error;
    ASSERT_TRUE(xjw::DemDomIO::writeDemRaster(
        makeDemGrid(),
        demPath,
        xjw::DemRasterFormat::Float32Tiff,
        &error))
        << error.toStdString();

    GDALAllRegister();
    const QByteArray encodedPath = QFile::encodeName(demPath);
    GDALDataset *dataset = static_cast<GDALDataset *>(
        GDALOpen(encodedPath.constData(), GA_Update));
    ASSERT_NE(dataset, nullptr);
    double rotatedTransform[6]{9.0, 2.0, 0.1, 32.0, 0.0, -3.0};
    ASSERT_EQ(dataset->SetGeoTransform(rotatedTransform), CE_None);
    GDALClose(dataset);

    QJsonObject estimate;
    EXPECT_FALSE(xjw::TerrainPipeline::estimateOrthoProduct(
        demPath, QJsonObject(), &estimate, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("北向上")));

    QJsonObject result;
    EXPECT_FALSE(xjw::TerrainPipeline::generateOrthoProduct(
        {QStringLiteral("unused.png")},
        demPath,
        QDir(directory.path()).filePath(QStringLiteral("unused.tif")),
        QJsonObject(),
        QJsonObject(),
        &result,
        &error));
    EXPECT_TRUE(error.contains(QStringLiteral("北向上")));
}

TEST(OrthoGeoTiffTest, WritesRgbBandsAndPreservesOutputGrid)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    xjw::DemGridData reference;
    reference.width = 2;
    reference.height = 2;
    reference.minX = 10.0;
    reference.minY = 20.0;
    reference.stepX = 2.0;
    reference.stepY = 3.0;

    cv::Mat image(2, 2, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(11, 22, 33);
    image.at<cv::Vec3b>(0, 1) = cv::Vec3b(44, 55, 66);
    image.at<cv::Vec3b>(1, 0) = cv::Vec3b(77, 88, 99);
    image.at<cv::Vec3b>(1, 1) = cv::Vec3b(111, 122, 133);

    const QString outputPath =
        QDir(directory.path()).filePath(QStringLiteral("ortho_rgb.tif"));
    QString error;
    ASSERT_TRUE(xjw::DemDomIO::writeDomGeoTiff(
        image, reference, outputPath, &error))
        << error.toStdString();

    const cv::Mat loaded =
        xjw::common::io::readImage(outputPath, cv::IMREAD_COLOR);
    ASSERT_EQ(loaded.type(), CV_8UC3);
    ASSERT_EQ(loaded.size(), image.size());
    const cv::Vec3b northWest = loaded.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(northWest[0], 77);
    EXPECT_EQ(northWest[1], 88);
    EXPECT_EQ(northWest[2], 99);

    xjw::DemGridData metadata;
    ASSERT_TRUE(xjw::DemDomIO::readDemMetadata(
        outputPath, &metadata, &error))
        << error.toStdString();
    EXPECT_EQ(metadata.width, 2);
    EXPECT_EQ(metadata.height, 2);
    EXPECT_DOUBLE_EQ(metadata.minX, 10.0);
    EXPECT_DOUBLE_EQ(metadata.minY, 20.0);
    EXPECT_DOUBLE_EQ(metadata.stepX, 2.0);
    EXPECT_DOUBLE_EQ(metadata.stepY, 3.0);
}

TEST(OrthoGeoTiffTest, WritesCoverageAlphaForInvalidPixels)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    xjw::DemGridData reference;
    reference.width = 2;
    reference.height = 2;
    reference.minX = 10.0;
    reference.minY = 20.0;
    reference.stepX = 2.0;
    reference.stepY = 3.0;
    const cv::Mat image(2, 2, CV_8UC3, cv::Scalar(11, 22, 33));
    cv::Mat validMask(2, 2, CV_8UC1, cv::Scalar(255));
    validMask.at<uchar>(1, 0) = 0;

    const QString outputPath =
        QDir(directory.path()).filePath(QStringLiteral("ortho_rgba.tif"));
    QString error;
    ASSERT_TRUE(xjw::DemDomIO::writeDomGeoTiff(
        image, validMask, reference, outputPath, &error))
        << error.toStdString();

    const cv::Mat loaded =
        xjw::common::io::readImage(outputPath, cv::IMREAD_UNCHANGED);
    ASSERT_EQ(loaded.type(), CV_8UC4);
    ASSERT_EQ(loaded.size(), image.size());
    EXPECT_EQ(loaded.at<cv::Vec4b>(0, 0)[3], 0);
    EXPECT_EQ(loaded.at<cv::Vec4b>(1, 0)[3], 255);
}

TEST(OrthoProjectorTest, WeightedAverageCombinesValidCameraColors)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const std::vector<xjw::OrthoImageInput> inputs{
        writeProjectionInput(
            &directory, QStringLiteral("red.png"), cv::Scalar(0, 0, 200)),
        writeProjectionInput(
            &directory, QStringLiteral("blue.png"), cv::Scalar(200, 0, 0))
    };

    xjw::OrthoGenerationOptions options;
    options.blendMode = xjw::OrthoBlendMode::WeightedAverage;
    options.colorCorrection = false;
    options.useProjectMasks = false;

    xjw::OrthoProjectionResult result;
    QString error;
    ASSERT_TRUE(xjw::OrthoProjector::project(
        makeProjectionDemGrid(), inputs, options, 0.0, &result, &error))
        << error.toStdString();

    EXPECT_EQ(result.filledPixelCount, 25);
    EXPECT_EQ(result.contributingCameraCount, 2);
    const cv::Vec3b center = result.imageBgr.at<cv::Vec3b>(2, 2);
    EXPECT_NEAR(center[0], 100, 1);
    EXPECT_EQ(center[1], 0);
    EXPECT_NEAR(center[2], 100, 1);
}

TEST(OrthoProjectorTest, ProjectMaskAndHoleFillHandleSmallInteriorGap)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    cv::Mat exclusionMask(64, 64, CV_8UC1, cv::Scalar(0));
    exclusionMask.at<uchar>(32, 32) = 255;
    const std::vector<xjw::OrthoImageInput> inputs{
        writeProjectionInput(
            &directory,
            QStringLiteral("masked.png"),
            cv::Scalar(30, 80, 160),
            exclusionMask)
    };

    xjw::OrthoGenerationOptions options;
    options.colorCorrection = false;
    options.useProjectMasks = true;
    options.fillHoles = true;
    options.holeFillMaxArea = 4;
    options.holeFillRadius = 1.0;

    xjw::OrthoProjectionResult result;
    QString error;
    ASSERT_TRUE(xjw::OrthoProjector::project(
        makeProjectionDemGrid(), inputs, options, 0.0, &result, &error))
        << error.toStdString();

    EXPECT_EQ(result.filledPixelCount, 24);
    EXPECT_EQ(result.holeFilledPixelCount, 1);
    EXPECT_NE(result.holeFilledMask.at<uchar>(2, 2), 0);
    EXPECT_EQ(result.imageBgr.at<cv::Vec3b>(2, 2), cv::Vec3b(30, 80, 160));
}

TEST(OrthoProjectorTest, HoleFillDoesNotCrossInternalDemNoDataBoundary)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    cv::Mat exclusionMask(64, 64, CV_8UC1, cv::Scalar(0));
    exclusionMask.at<uchar>(32, 32) = 255;
    const std::vector<xjw::OrthoImageInput> inputs{
        writeProjectionInput(
            &directory,
            QStringLiteral("masked_nodata.png"),
            cv::Scalar(30, 80, 160),
            exclusionMask)
    };

    xjw::DemGridData dem = makeProjectionDemGrid();
    dem.validMask.at<uchar>(2, 3) = 0;
    xjw::OrthoGenerationOptions options;
    options.colorCorrection = false;
    options.useProjectMasks = true;
    options.fillHoles = true;
    options.holeFillMaxArea = 4;
    options.holeFillRadius = 1.0;

    xjw::OrthoProjectionResult result;
    QString error;
    ASSERT_TRUE(xjw::OrthoProjector::project(
        dem, inputs, options, 0.0, &result, &error))
        << error.toStdString();
    EXPECT_EQ(result.holeFilledPixelCount, 0);
    EXPECT_EQ(result.holeFilledMask.at<uchar>(2, 2), 0);
}

TEST(OrthoProjectorTest, RejectsZeroCoverageAndHonorsCancellation)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    xjw::OrthoImageInput input = writeProjectionInput(
        &directory, QStringLiteral("behind.png"), cv::Scalar(10, 20, 30));
    input.camera = makeProjectionCamera(10.0);

    xjw::OrthoProjectionResult result;
    QString error;
    EXPECT_FALSE(xjw::OrthoProjector::project(
        makeProjectionDemGrid(),
        std::vector<xjw::OrthoImageInput>{input},
        xjw::OrthoGenerationOptions{},
        0.0,
        &result,
        &error));
    EXPECT_TRUE(error.contains(QStringLiteral("有效像素")));

    std::atomic_bool cancelled{true};
    error.clear();
    EXPECT_FALSE(xjw::OrthoProjector::project(
        makeProjectionDemGrid(),
        std::vector<xjw::OrthoImageInput>{input},
        xjw::OrthoGenerationOptions{},
        0.0,
        &result,
        &error,
        &cancelled));
    EXPECT_TRUE(error.contains(QStringLiteral("取消")));
}

TEST(OrthoProjectorTest, ReportsSelectedImageWithoutCameraInsteadOfSkippingIt)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    xjw::OrthoImageInput invalid = writeProjectionInput(
        &directory, QStringLiteral("missing_camera.png"), cv::Scalar(1, 2, 3));
    invalid.camera = xjw::FramePinholeCamera();
    const xjw::OrthoImageInput valid = writeProjectionInput(
        &directory, QStringLiteral("valid_camera.png"), cv::Scalar(4, 5, 6));

    xjw::OrthoProjectionResult result;
    QString error;
    EXPECT_FALSE(xjw::OrthoProjector::project(
        makeProjectionDemGrid(),
        std::vector<xjw::OrthoImageInput>{invalid, valid},
        xjw::OrthoGenerationOptions{},
        0.0,
        &result,
        &error));
    EXPECT_TRUE(error.contains(QStringLiteral("missing_camera.png")));
    EXPECT_TRUE(error.contains(QStringLiteral("相机参数")));
}

TEST(OrthoProjectorTest, RejectsLoadedCameraWithInvalidIntrinsics)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    xjw::OrthoImageInput input = writeProjectionInput(
        &directory,
        QStringLiteral("invalid_intrinsics.png"),
        cv::Scalar(5, 6, 7));
    input.camera.setIntrinsics(0.0, 0.0, 32.0, 32.0);

    xjw::OrthoProjectionResult result;
    QString error;
    EXPECT_FALSE(xjw::OrthoProjector::project(
        makeProjectionDemGrid(),
        std::vector<xjw::OrthoImageInput>{input},
        xjw::OrthoGenerationOptions{},
        0.0,
        &result,
        &error));
    EXPECT_TRUE(error.contains(QStringLiteral("内参")));
}

} // namespace
