#include "io/SurveyControlMigration.h"

#include <gtest/gtest.h>

#include <QHash>
#include <QJsonArray>
#include <QJsonObject>

using xjw::control_points::ProjectionState;
using xjw::control_points::SurveyControlMigrationResult;
using xjw::control_points::migrateSurveyControl;

namespace
{

QJsonObject makeLegacySurveyControlWithControlCheckAndScaleBar()
{
    const QJsonObject control{
        {QStringLiteral("id"), QStringLiteral("GCP001")},
        {QStringLiteral("enabled"), true},
        {QStringLiteral("x"), 1.0},
        {QStringLiteral("y"), 2.0},
        {QStringLiteral("z"), 3.0},
        {QStringLiteral("sigma_m"), 0.02},
        {QStringLiteral("observations"), QJsonArray{
            QJsonObject{
                {QStringLiteral("image_path"), QStringLiteral("E:/images/a.png")},
                {QStringLiteral("u"), 120.25},
                {QStringLiteral("v"), 330.75}
            }
        }}
    };
    const QJsonObject check{
        {QStringLiteral("id"), QStringLiteral("CHK001")},
        {QStringLiteral("enabled"), true},
        {QStringLiteral("x"), 4.0},
        {QStringLiteral("y"), 5.0},
        {QStringLiteral("z"), 6.0},
        {QStringLiteral("sigma_m"), 0.05}
    };
    const QJsonObject scaleBar{
        {QStringLiteral("id"), QStringLiteral("SB001")},
        {QStringLiteral("from_id"), QStringLiteral("GCP001")},
        {QStringLiteral("to_id"), QStringLiteral("CHK001")},
        {QStringLiteral("measured_m"), 7.5},
        {QStringLiteral("sigma_m"), 0.01},
        {QStringLiteral("enabled"), true}
    };
    return QJsonObject{
        {QStringLiteral("control_points"), QJsonArray{control}},
        {QStringLiteral("check_points"), QJsonArray{check}},
        {QStringLiteral("scale_bars"), QJsonArray{scaleBar}}
    };
}

} // namespace

TEST(SurveyControlMigrationTest, ConvertsPinnedObservationsAndScaleBarIds)
{
    const QHash<QString, QString> identities{
        {QStringLiteral("E:/images/a.png"), QStringLiteral("image-uuid-a")}
    };
    const SurveyControlMigrationResult result =
        migrateSurveyControl(makeLegacySurveyControlWithControlCheckAndScaleBar(), identities);

    ASSERT_TRUE(result.ok) << qPrintable(result.error);
    EXPECT_EQ(result.markerSet.markers().size(), 2u);
    ASSERT_EQ(result.markerSet.scaleBars().size(), 1u);
    ASSERT_EQ(result.markerSet.markers()[0].projections.size(), 1u);
    EXPECT_EQ(result.markerSet.markers()[0].projections[0].state, ProjectionState::ManualPinned);
    EXPECT_EQ(result.markerSet.markers()[0].projections[0].imageId, QStringLiteral("image-uuid-a"));
    EXPECT_EQ(result.markerSet.scaleBars()[0].firstMarkerId, result.markerSet.markers()[0].id);
    EXPECT_EQ(result.markerSet.scaleBars()[0].secondMarkerId, result.markerSet.markers()[1].id);
    EXPECT_EQ(result.migratedMarkers, 2);
    EXPECT_EQ(result.migratedProjections, 1);
    EXPECT_EQ(result.migratedScaleBars, 1);
}

TEST(SurveyControlMigrationTest, RejectsScaleBarsWithUnknownEndpoints)
{
    QJsonObject legacy = makeLegacySurveyControlWithControlCheckAndScaleBar();
    QJsonArray scaleBars = legacy.value(QStringLiteral("scale_bars")).toArray();
    QJsonObject scaleBar = scaleBars[0].toObject();
    scaleBar[QStringLiteral("to_id")] = QStringLiteral("missing");
    scaleBars[0] = scaleBar;
    legacy[QStringLiteral("scale_bars")] = scaleBars;

    const auto result = migrateSurveyControl(
        legacy,
        {{QStringLiteral("E:/images/a.png"), QStringLiteral("image-uuid-a")}});
    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.error.contains(QStringLiteral("missing")));
}
