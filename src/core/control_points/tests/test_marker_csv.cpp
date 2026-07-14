#include "io/MarkerCsv.h"

#include <gtest/gtest.h>

namespace cp = xjw::control_points;

TEST(MarkerCsvTest, PreservesSeparateAccuracyAndUsableThreeDimensionalCrs)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,sigma_x_m,sigma_y_m,sigma_z_m,source_crs\n"
        "control,GCP001,3657660.1,255768.2,5201382.3,0.01,0.02,0.05,EPSG:4978\n");

    const cp::MarkerCsvImportResult result = cp::parseMarkerCsv(csv);

    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    const cp::ReferenceCoordinate &reference =
        result.markerSet.markers().front().referenceCoordinate.value();
    EXPECT_DOUBLE_EQ(reference.sigmaX, 0.01);
    EXPECT_DOUBLE_EQ(reference.sigmaY, 0.02);
    EXPECT_DOUBLE_EQ(reference.sigmaZ, 0.05);
    EXPECT_TRUE(reference.referenceUsable) << qPrintable(reference.referenceError);
}

TEST(MarkerCsvTest, PreservesButBlocksUnknownVerticalDatum)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,source_crs,axis_order,vertical_unit\n"
        "control,GCP001,116.391,39.907,50,EPSG:4326,longitude_latitude,m\n");

    const cp::MarkerCsvImportResult result = cp::parseMarkerCsv(csv);

    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    const cp::ReferenceCoordinate &reference =
        result.markerSet.markers().front().referenceCoordinate.value();
    EXPECT_FALSE(reference.referenceUsable);
    EXPECT_TRUE(reference.referenceError.contains(QStringLiteral("垂直基准")));
    EXPECT_DOUBLE_EQ(reference.z, 50.0);
}

TEST(MarkerCsvTest, PreservesButBlocksInvalidCrs)
{
    cp::MarkerCsvImportOptions options;
    options.sourceCrs = QStringLiteral("EPSG:invalid");
    options.verticalDatum = QStringLiteral("ellipsoidal");
    options.verticalUnit = QStringLiteral("m");

    const cp::MarkerCsvImportResult result = cp::parseMarkerCsv(
        QStringLiteral("role,id,x,y,z\ncontrol,GCP001,1,2,3\n"), options);

    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    const cp::ReferenceCoordinate &reference =
        result.markerSet.markers().front().referenceCoordinate.value();
    EXPECT_FALSE(reference.referenceUsable);
    EXPECT_TRUE(reference.referenceError.contains(QStringLiteral("CRS")));
}

TEST(MarkerCsvTest, RoundTripPreservesReferenceMetadata)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,sigma_x_m,sigma_y_m,sigma_z_m,source_crs,axis_order,"
        "vertical_datum,vertical_unit\n"
        "check,CHK001,116.391,39.907,50,0.01,0.02,0.05,EPSG:4326,"
        "longitude_latitude,ellipsoidal,m\n");
    const cp::MarkerCsvImportResult first = cp::parseMarkerCsv(csv);
    ASSERT_TRUE(first.ok) << qPrintable(first.error);

    const cp::MarkerCsvImportResult second = cp::parseMarkerCsv(cp::markerSetToCsv(first.markerSet));

    ASSERT_TRUE(second.ok) << qPrintable(second.error);
    const cp::ReferenceCoordinate &reference =
        second.markerSet.markers().front().referenceCoordinate.value();
    EXPECT_EQ(reference.sourceCrs, QStringLiteral("EPSG:4326"));
    EXPECT_EQ(reference.axisOrder, QStringLiteral("longitude_latitude"));
    EXPECT_EQ(reference.verticalDatum, QStringLiteral("ellipsoidal"));
    EXPECT_EQ(reference.verticalUnit, QStringLiteral("m"));
    EXPECT_TRUE(reference.referenceUsable) << qPrintable(reference.referenceError);
    EXPECT_DOUBLE_EQ(reference.sigmaX, 0.01);
    EXPECT_DOUBLE_EQ(reference.sigmaY, 0.02);
    EXPECT_DOUBLE_EQ(reference.sigmaZ, 0.05);
}

