#include "DemDomIO.h"
#include "SmallBodyGlobalProductGenerator.h"
#include "SmallBodyMeshRaycaster.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QTemporaryDir>

#include <atomic>
#include <cmath>
#include <limits>

namespace
{

xjw::TerrainMeshInput makeOctahedron(float radius = 10.0f, bool withColors = true)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(6, 3);
    const float coordinates[6][3] = {
        {radius, 0.0f, 0.0f}, {-radius, 0.0f, 0.0f},
        {0.0f, radius, 0.0f}, {0.0f, -radius, 0.0f},
        {0.0f, 0.0f, radius}, {0.0f, 0.0f, -radius}};
    for (int row = 0; row < 6; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            points(row, col) = coordinates[row][col];
        }
    }

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(8, 3);
    const int indices[8][3] = {
        {4, 0, 2}, {4, 2, 1}, {4, 1, 3}, {4, 3, 0},
        {5, 2, 0}, {5, 1, 2}, {5, 3, 1}, {5, 0, 3}};
    for (int row = 0; row < 8; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            faces(row, col) = indices[row][col];
        }
    }

    xjw::TerrainMeshInput input;
    input.mesh = xjw::PlaPointCloud(std::move(points));
    input.mesh.setFaces(std::move(faces));
    if (withColors)
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(6, 3);
        const std::uint8_t rgb[6][3] = {
            {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
            {255, 255, 0}, {255, 255, 255}, {0, 0, 0}};
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                colors(row, col) = rgb[row][col];
            }
        }
        input.mesh.setColors(std::move(colors));
    }
    return input;
}

} // namespace

TEST(SmallBodyGlobalOptionsTest, RejectsInertialFrame)
{
    xjw::SmallBodyGlobalOptions options;
    options.bodyFixedFrame = QStringLiteral("J2000");
    QString error;
    EXPECT_FALSE(options.validate(&error));
    EXPECT_TRUE(error.contains(QStringLiteral("体固连")));
}

TEST(SmallBodyGlobalOptionsTest, RejectsInvalidIdentityAndCentralMeridian)
{
    xjw::SmallBodyGlobalOptions options;
    QString error;
    options.targetName.clear();
    EXPECT_FALSE(options.validate(&error));
    EXPECT_TRUE(error.contains(QStringLiteral("目标天体")));

    options.targetName = QStringLiteral("SyntheticBody");
    options.centralMeridianDeg = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(options.validate(&error));
    EXPECT_TRUE(error.contains(QStringLiteral("中央经线")));

    options.centralMeridianDeg = 0.0;
    options.angularResolutionDeg = 0.01;
    options.maximumPixelCount = 1000000000;
    EXPECT_FALSE(options.validate(&error));
    EXPECT_TRUE(error.contains(QStringLiteral("安全像素")));
}

TEST(SmallBodyMeshRaycasterTest, ReturnsNearestRadialSurfaceAndVertexColor)
{
    const xjw::TerrainMeshInput input = makeOctahedron();
    xjw::SmallBodyMeshRaycaster raycaster;
    QString error;
    ASSERT_TRUE(raycaster.initialize(input, cv::Vec3d(0.0, 0.0, 0.0), &error)) << qPrintable(error);

    xjw::SmallBodyMeshRaycaster::Hit hit;
    ASSERT_TRUE(raycaster.intersect(cv::Vec3d(1.0, 0.0, 0.0), &hit));
    EXPECT_NEAR(hit.radius, 10.0, 1e-6);
    EXPECT_EQ(hit.colorBgr, cv::Vec3b(0, 0, 255));
    EXPECT_NEAR(hit.reliability, 1.0 / std::sqrt(3.0), 1e-6);
    EXPECT_FALSE(hit.ambiguous);
}

TEST(SmallBodyMeshRaycasterTest, RejectsUnsupportedRepeatingTextureCoordinates)
{
    xjw::TerrainMeshInput input = makeOctahedron();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coordinates(6, 2);
    for (int row = 0; row < 6; ++row)
    {
        texture_coordinates(row, 0) = 0.5f;
        texture_coordinates(row, 1) = 0.5f;
    }
    texture_coordinates(0, 0) = 1.25f;
    input.mesh.setTextureCoords(std::move(texture_coordinates));
    input.texture = cv::Mat(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));

    xjw::SmallBodyMeshRaycaster raycaster;
    QString error;
    EXPECT_FALSE(raycaster.initialize(
        input, cv::Vec3d(0.0, 0.0, 0.0), &error));
    EXPECT_TRUE(error.contains(QStringLiteral("[0,1]")));
}

TEST(SmallBodyGlobalProductGeneratorTest, CreatesNativeRegisteredProducts)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    xjw::SmallBodyGlobalOptions options;
    options.targetName = QStringLiteral("SyntheticBody");
    options.bodyFixedFrame = QStringLiteral("SYNTHETIC_FIXED");
    options.automaticCenter = false;
    options.bodyCenter = cv::Vec3d(0.0, 0.0, 0.0);
    options.referenceRadiusM = 10.0;
    options.angularResolutionDeg = 60.0;
    options.centralMeridianDeg = -30.0;
    options.maximumPixelCount = 1000;
    options.writeReportPreview = true;

    xjw::SmallBodyGlobalProducts products;
    QString error;
    ASSERT_TRUE(xjw::SmallBodyGlobalProductGenerator::generateFromMesh(
        makeOctahedron(), QStringLiteral("synthetic"), temporary_directory.path(),
        options, &products, &error)) << qPrintable(error);

    ASSERT_EQ(products.radialDem.width, 6);
    ASSERT_EQ(products.radialDem.height, 3);
    EXPECT_EQ(cv::countNonZero(products.validMask), 18);
    EXPECT_NEAR(products.coverageRatio, 1.0, 1e-12);
    EXPECT_NEAR(products.solidAngleWeightedCoverageRatio, 1.0, 1e-12);
    EXPECT_NEAR(products.radialDem.elevation.at<float>(1, 0), 10.0f, 1e-5f);
    EXPECT_NEAR(products.elevationDem.elevation.at<float>(1, 0), 0.0f, 1e-5f);
    EXPECT_EQ(products.domBgr.at<cv::Vec3b>(1, 0), cv::Vec3b(0, 0, 255));

    EXPECT_TRUE(QFileInfo::exists(products.radialDemPath));
    EXPECT_TRUE(QFileInfo::exists(products.elevationDemPath));
    EXPECT_TRUE(QFileInfo::exists(products.domPath));
    EXPECT_TRUE(QFileInfo::exists(products.reliabilityPath));
    EXPECT_TRUE(QFileInfo::exists(products.ambiguityPath));
    EXPECT_TRUE(QFileInfo::exists(products.reportPath));
    EXPECT_TRUE(QFileInfo::exists(products.previewPath));
    const QImage preview(products.previewPath);
    EXPECT_EQ(preview.size(), QSize(2000, 1180));
    EXPECT_FALSE(products.report.value(QStringLiteral("difference_available")).toBool(true));
    EXPECT_EQ(products.report.value(QStringLiteral("dom_color_source")).toString(),
              QStringLiteral("vertex_color"));

    xjw::DemGridData metadata;
    ASSERT_TRUE(xjw::DemDomIO::readDemMetadata(products.radialDemPath, &metadata, &error))
        << qPrintable(error);
    EXPECT_DOUBLE_EQ(metadata.minX - metadata.stepX * 0.5, 0.0);
    EXPECT_DOUBLE_EQ(metadata.minY - metadata.stepY * 0.5, -90.0);
    EXPECT_EQ(metadata.projection.metadata.value(QStringLiteral("TARGET_NAME")),
              QStringLiteral("SyntheticBody"));
    EXPECT_EQ(metadata.projection.metadata.value(QStringLiteral("BODY_FIXED_FRAME")),
              QStringLiteral("SYNTHETIC_FIXED"));
    EXPECT_EQ(metadata.projection.metadata.value(QStringLiteral("LONGITUDE_DIRECTION")),
              QStringLiteral("positive_east"));
    EXPECT_EQ(metadata.projection.metadata.value(QStringLiteral("PRODUCT_TYPE")),
              QStringLiteral("small_body_global_radial_dem"));
    EXPECT_EQ(metadata.projection.metadata.value(QStringLiteral("BAND_UNIT")),
              QStringLiteral("m"));
    EXPECT_EQ(metadata.projection.metadata.value(QStringLiteral("INTERSECTION_SELECTION")),
              QStringLiteral("nearest_positive"));

    xjw::DemGridData dom_metadata;
    ASSERT_TRUE(xjw::DemDomIO::readDemMetadata(products.domPath, &dom_metadata, &error))
        << qPrintable(error);
    EXPECT_EQ(dom_metadata.projection.metadata.value(QStringLiteral("VERTICAL_REFERENCE")),
              QStringLiteral("not_applicable"));
    EXPECT_EQ(dom_metadata.projection.metadata.value(QStringLiteral("PRODUCT_TYPE")),
              QStringLiteral("surface_colour_dom"));

    xjw::DemGridData ambiguity;
    ASSERT_TRUE(xjw::DemDomIO::readDemRaster(products.ambiguityPath, &ambiguity, &error))
        << qPrintable(error);
    EXPECT_EQ(cv::countNonZero(ambiguity.elevation), 0);
    EXPECT_EQ(ambiguity.projection.metadata.value(QStringLiteral("PRODUCT_TYPE")),
              QStringLiteral("radial_multi_surface_ambiguity_mask"));
    EXPECT_EQ(ambiguity.projection.metadata.value(QStringLiteral("BAND_UNIT")),
              QStringLiteral("1"));

    xjw::SmallBodyGlobalProducts duplicate_products;
    EXPECT_FALSE(xjw::SmallBodyGlobalProductGenerator::generateFromMesh(
        makeOctahedron(), QStringLiteral("synthetic"), temporary_directory.path(),
        options, &duplicate_products, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("已存在")));
    EXPECT_TRUE(QFileInfo::exists(products.reportPath));
}

TEST(SmallBodyGlobalProductGeneratorTest, RejectsFakeDomForUncoloredMesh)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    xjw::SmallBodyGlobalOptions options;
    options.angularResolutionDeg = 60.0;
    options.maximumPixelCount = 1000;
    xjw::SmallBodyGlobalProducts products;
    QString error;
    EXPECT_FALSE(xjw::SmallBodyGlobalProductGenerator::generateFromMesh(
        makeOctahedron(10.0f, false), QStringLiteral("synthetic"), temporary_directory.path(),
        options, &products, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("不会用固定灰色")));
}

TEST(SmallBodyGlobalProductGeneratorTest, ConvertsKilometerMeshCoordinatesToMeters)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    xjw::SmallBodyGlobalOptions options;
    options.targetName = QStringLiteral("KilometerBody");
    options.surfaceCoordinateUnit = QStringLiteral("km");
    options.angularResolutionDeg = 60.0;
    options.maximumPixelCount = 1000;
    options.writeReportPreview = false;

    xjw::SmallBodyGlobalProducts products;
    QString error;
    ASSERT_TRUE(xjw::SmallBodyGlobalProductGenerator::generateFromMesh(
        makeOctahedron(0.01f), QStringLiteral("synthetic_km"), temporary_directory.path(),
        options, &products, &error)) << qPrintable(error);
    EXPECT_NEAR(products.referenceRadiusM, 10.0, 1.0e-5);
    EXPECT_EQ(products.report.value(QStringLiteral("frame")).toObject()
                  .value(QStringLiteral("source_surface_unit")).toString(),
              QStringLiteral("km"));
    EXPECT_EQ(products.radialDem.projection.metadata.value(
                  QStringLiteral("SOURCE_SURFACE_UNIT")),
              QStringLiteral("km"));
}

TEST(SmallBodyGlobalProductGeneratorTest, PreservesValidMinus9999Elevation)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    xjw::SmallBodyGlobalOptions options;
    options.automaticCenter = false;
    options.bodyCenter = cv::Vec3d(0.0, 0.0, 0.0);
    options.referenceRadiusM = 10000.0;
    options.angularResolutionDeg = 60.0;
    options.centralMeridianDeg = -30.0;
    options.maximumPixelCount = 1000;
    options.writeReportPreview = false;

    xjw::SmallBodyGlobalProducts products;
    QString error;
    ASSERT_TRUE(xjw::SmallBodyGlobalProductGenerator::generateFromMesh(
        makeOctahedron(1.0f), QStringLiteral("synthetic_minus_9999"),
        temporary_directory.path(), options, &products, &error)) << qPrintable(error);

    xjw::DemGridData round_trip;
    ASSERT_TRUE(xjw::DemDomIO::readDemRaster(
        products.elevationDemPath, &round_trip, &error)) << qPrintable(error);
    EXPECT_EQ(cv::countNonZero(round_trip.validMask), 18);
    EXPECT_FLOAT_EQ(round_trip.elevation.at<float>(1, 0), -9999.0f);
}

TEST(SmallBodyGlobalProductGeneratorTest, HonorsPreCancelledRequest)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    xjw::SmallBodyGlobalOptions options;
    options.angularResolutionDeg = 60.0;
    options.maximumPixelCount = 1000;

    std::atomic_bool cancel_requested = true;
    xjw::SmallBodyGlobalProducts products;
    QString error;
    EXPECT_FALSE(xjw::SmallBodyGlobalProductGenerator::generateFromMesh(
        makeOctahedron(), QStringLiteral("synthetic"), temporary_directory.path(),
        options, &products, &error, &cancel_requested));
    EXPECT_TRUE(error.contains(QStringLiteral("取消")));
    EXPECT_TRUE(QDir(temporary_directory.path()).entryList(
        QDir::Files | QDir::NoDotAndDotDot).isEmpty());
}
