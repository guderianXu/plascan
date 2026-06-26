#include "SurveyControlImport.h"

#include <QJsonArray>
#include <QJsonObject>

#include <gtest/gtest.h>

using xjw::qc::SurveyControlImportOptions;
using xjw::qc::parseSurveyControlCsv;

TEST(SurveyControlImport, ParsesMixedControlCheckpointAndScaleBarCsv)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,sigma_m,enabled,from_id,to_id,measured_m\n"
        "control,GCP001,1.0,2.0,3.0,0.02,true,,,\n"
        "check,CHK001,4.0,5.0,6.0,0.05,false,,,\n"
        "scale_bar,SB001,,,,0.01,true,GCP001,CHK001,7.5\n");

    const auto result = parseSurveyControlCsv(csv);

    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_EQ(result.controlPointCount, 1);
    EXPECT_EQ(result.checkPointCount, 1);
    EXPECT_EQ(result.scaleBarCount, 1);

    const QJsonArray controls = result.surveyControl.value(QStringLiteral("control_points")).toArray();
    ASSERT_EQ(controls.size(), 1);
    const QJsonObject gcp = controls.at(0).toObject();
    EXPECT_EQ(gcp.value(QStringLiteral("id")).toString(), QStringLiteral("GCP001"));
    EXPECT_NEAR(gcp.value(QStringLiteral("x")).toDouble(), 1.0, 1e-12);
    EXPECT_NEAR(gcp.value(QStringLiteral("sigma_m")).toDouble(), 0.02, 1e-12);
    EXPECT_TRUE(gcp.value(QStringLiteral("enabled")).toBool(false));

    const QJsonArray checks = result.surveyControl.value(QStringLiteral("check_points")).toArray();
    ASSERT_EQ(checks.size(), 1);
    const QJsonObject checkpoint = checks.at(0).toObject();
    EXPECT_EQ(checkpoint.value(QStringLiteral("id")).toString(), QStringLiteral("CHK001"));
    EXPECT_FALSE(checkpoint.value(QStringLiteral("enabled")).toBool(true));

    const QJsonArray scaleBars = result.surveyControl.value(QStringLiteral("scale_bars")).toArray();
    ASSERT_EQ(scaleBars.size(), 1);
    const QJsonObject scaleBar = scaleBars.at(0).toObject();
    EXPECT_EQ(scaleBar.value(QStringLiteral("id")).toString(), QStringLiteral("SB001"));
    EXPECT_EQ(scaleBar.value(QStringLiteral("from_id")).toString(), QStringLiteral("GCP001"));
    EXPECT_EQ(scaleBar.value(QStringLiteral("to_id")).toString(), QStringLiteral("CHK001"));
    EXPECT_NEAR(scaleBar.value(QStringLiteral("measured_m")).toDouble(), 7.5, 1e-12);
}

TEST(SurveyControlImport, UsesDefaultRoleWhenRoleColumnIsMissing)
{
    const QString csv = QStringLiteral(
        "id,x,y,z\n"
        "GCP001,1.0,2.0,3.0\n"
        "GCP002,4.0,5.0,6.0\n");

    SurveyControlImportOptions options;
    options.defaultRole = QStringLiteral("control");

    const auto result = parseSurveyControlCsv(csv, options);

    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_EQ(result.controlPointCount, 2);
    EXPECT_EQ(result.checkPointCount, 0);
    EXPECT_EQ(result.scaleBarCount, 0);
}

TEST(SurveyControlImport, AcceptsSemicolonSeparatedControlCsv)
{
    const QString csv = QStringLiteral(
        "role;id;x;y;z;sigma_m\n"
        "control;GCP001;1.0;2.0;3.0;0.02\n"
        "check;CHK001;4.0;5.0;6.0;0.05\n");

    const auto result = parseSurveyControlCsv(csv);

    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_EQ(result.controlPointCount, 1);
    EXPECT_EQ(result.checkPointCount, 1);

    const QJsonArray controls = result.surveyControl.value(QStringLiteral("control_points")).toArray();
    ASSERT_EQ(controls.size(), 1);
    EXPECT_EQ(controls.at(0).toObject().value(QStringLiteral("id")).toString(), QStringLiteral("GCP001"));
}

TEST(SurveyControlImport, AggregatesImageObservationsByControlPointId)
{
    const QString csv = QStringLiteral(
        "role,id,x,y,z,sigma_m,image_path,pixel_x,pixel_y\n"
        "control,GCP001,1.0,2.0,3.0,0.02,E:/images/img_001.jpg,123.5,456.5\n"
        "control,GCP001,1.0,2.0,3.0,0.02,E:/images/img_002.jpg,223.5,556.5\n");

    const auto result = parseSurveyControlCsv(csv);

    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_EQ(result.controlPointCount, 1);

    const QJsonArray controls = result.surveyControl.value(QStringLiteral("control_points")).toArray();
    ASSERT_EQ(controls.size(), 1);
    const QJsonObject gcp = controls.at(0).toObject();
    EXPECT_EQ(gcp.value(QStringLiteral("id")).toString(), QStringLiteral("GCP001"));

    const QJsonArray observations = gcp.value(QStringLiteral("observations")).toArray();
    ASSERT_EQ(observations.size(), 2);
    EXPECT_EQ(observations.at(0).toObject().value(QStringLiteral("image_path")).toString(),
              QStringLiteral("E:/images/img_001.jpg"));
    EXPECT_NEAR(observations.at(0).toObject().value(QStringLiteral("u")).toDouble(), 123.5, 1e-12);
    EXPECT_NEAR(observations.at(0).toObject().value(QStringLiteral("v")).toDouble(), 456.5, 1e-12);
    EXPECT_EQ(observations.at(1).toObject().value(QStringLiteral("image_path")).toString(),
              QStringLiteral("E:/images/img_002.jpg"));
}

TEST(SurveyControlImport, ReportsMissingRequiredColumns)
{
    const QString csv = QStringLiteral(
        "role,x,y,z\n"
        "control,1.0,2.0,3.0\n");

    const auto result = parseSurveyControlCsv(csv);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.error.contains(QStringLiteral("id")));
}
