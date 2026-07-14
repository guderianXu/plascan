#include "io/MarkerCsv.h"

#include <gtest/gtest.h>

using xjw::control_points::MarkerCsvImportOptions;
using xjw::control_points::MarkerRole;
using xjw::control_points::ProjectionState;
using xjw::control_points::parseMarkerCsv;
using xjw::control_points::markerSetToCsv;

TEST(SurveyControlImport, ParsesMixedControlCheckpointAndScaleBarCsv)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,sigma_m,enabled,from_id,to_id,measured_m\n"
        "control,GCP001,1.0,2.0,3.0,0.02,true,,,\n"
        "check,CHK001,4.0,5.0,6.0,0.05,false,,,\n"
        "scale_bar,SB001,,,,0.01,true,GCP001,CHK001,7.5\n");

    const auto result = parseMarkerCsv(csv);

    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    EXPECT_EQ(result.controlPointCount, 1);
    EXPECT_EQ(result.checkPointCount, 1);
    EXPECT_EQ(result.scaleBarCount, 1);
    ASSERT_EQ(result.markerSet.markers().size(), 2u);
    EXPECT_EQ(result.markerSet.markers()[0].label, QStringLiteral("GCP001"));
    EXPECT_EQ(result.markerSet.markers()[0].role, MarkerRole::ControlPoint);
    ASSERT_TRUE(result.markerSet.markers()[0].referenceCoordinate.has_value());
    EXPECT_NEAR(result.markerSet.markers()[0].referenceCoordinate->x, 1.0, 1e-12);
    EXPECT_NEAR(result.markerSet.markers()[0].referenceCoordinate->sigmaX, 0.02, 1e-12);
    EXPECT_FALSE(result.markerSet.markers()[1].enabled);
    ASSERT_EQ(result.markerSet.scaleBars().size(), 1u);
    EXPECT_EQ(result.markerSet.scaleBars()[0].label, QStringLiteral("SB001"));
    EXPECT_NEAR(result.markerSet.scaleBars()[0].measuredDistance, 7.5, 1e-12);
}

TEST(SurveyControlImport, UsesDefaultRoleWhenRoleColumnIsMissing)
{
    const QString csv = QStringLiteral(
        "id,x,y,z\n"
        "GCP001,1.0,2.0,3.0\n"
        "GCP002,4.0,5.0,6.0\n");

    MarkerCsvImportOptions options;
    options.defaultRole = QStringLiteral("control");
    const auto result = parseMarkerCsv(csv, options);

    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    EXPECT_EQ(result.controlPointCount, 2);
    EXPECT_EQ(result.checkPointCount, 0);
}

TEST(SurveyControlImport, AcceptsSemicolonSeparatedControlCsv)
{
    const QString csv = QStringLiteral(
        "role;id;x;y;z;sigma_m\n"
        "control;GCP001;1.0;2.0;3.0;0.02\n"
        "check;CHK001;4.0;5.0;6.0;0.05\n");

    const auto result = parseMarkerCsv(csv);
    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    EXPECT_EQ(result.controlPointCount, 1);
    EXPECT_EQ(result.checkPointCount, 1);
}

TEST(SurveyControlImport, AggregatesImageObservationsByControlPointId)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,sigma_m,image_path,pixel_x,pixel_y\n"
        "control,GCP001,1.0,2.0,3.0,0.02,E:/images/img_001.jpg,123.5,456.5\n"
        "control,GCP001,1.0,2.0,3.0,0.02,E:/images/img_002.jpg,223.5,556.5\n");
    MarkerCsvImportOptions options;
    options.imageIdentityByPath = {
        {QStringLiteral("E:/images/img_001.jpg"), QStringLiteral("image-uuid-1")},
        {QStringLiteral("E:/images/img_002.jpg"), QStringLiteral("image-uuid-2")}
    };

    const auto result = parseMarkerCsv(csv, options);

    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    ASSERT_EQ(result.markerSet.markers().size(), 1u);
    const auto &projections = result.markerSet.markers()[0].projections;
    ASSERT_EQ(projections.size(), 2u);
    EXPECT_EQ(projections[0].imageId, QStringLiteral("image-uuid-1"));
    EXPECT_EQ(projections[0].state, ProjectionState::ManualPinned);
    EXPECT_EQ(projections[0].xy, QPointF(123.5, 456.5));
    EXPECT_EQ(projections[1].imageId, QStringLiteral("image-uuid-2"));
}

TEST(SurveyControlImport, ReportsMissingRequiredColumns)
{
    const QString csv = QStringLiteral(
        "role,x,y,z\n"
        "control,1.0,2.0,3.0\n");

    const auto result = parseMarkerCsv(csv);
    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.error.contains(QStringLiteral("id")));
}

TEST(SurveyControlImport, ExportsAParseableMarkerSet)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,sigma_m,from_id,to_id,measured_m\n"
        "control,GCP001,1.0,2.0,3.0,0.02,,,\n"
        "check,CHK001,4.0,5.0,6.0,0.05,,,\n"
        "scale_bar,SB001,,,,0.01,GCP001,CHK001,7.5\n");
    const auto first = parseMarkerCsv(csv);
    ASSERT_TRUE(first.ok) << qPrintable(first.error);

    const auto second = parseMarkerCsv(markerSetToCsv(first.markerSet));

    ASSERT_TRUE(second.ok) << qPrintable(second.error);
    EXPECT_EQ(second.controlPointCount, 1);
    EXPECT_EQ(second.checkPointCount, 1);
    EXPECT_EQ(second.scaleBarCount, 1);
    EXPECT_EQ(second.markerSet.scaleBars()[0].label, QStringLiteral("SB001"));
}
