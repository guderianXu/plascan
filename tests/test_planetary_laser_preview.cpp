#include <gtest/gtest.h>

#include "ProjectBundleAdjustWorkflow.h"

#include <QJsonObject>
#include <QString>

TEST(PlanetaryLaserPreviewTest, ShowsShotCountRangeRmsAndInputPath)
{
    QJsonObject planetarySummary;
    planetarySummary[QStringLiteral("enabled")] = true;
    planetarySummary[QStringLiteral("range_constraint_count")] = 3;
    planetarySummary[QStringLiteral("total_shots")] = 4;
    planetarySummary[QStringLiteral("range_rms_before_m")] = 2.0;
    planetarySummary[QStringLiteral("range_rms_after_m")] = 0.25;
    planetarySummary[QStringLiteral("target")] = QStringLiteral("MOON");
    planetarySummary[QStringLiteral("body_fixed_frame")] = QStringLiteral("IAU_MOON");
    planetarySummary[QStringLiteral("data_path")] = QStringLiteral("E:/data/lola.json");

    QJsonObject result;
    result[QStringLiteral("track_count")] = 10;
    result[QStringLiteral("optimized_count")] = 9;
    result[QStringLiteral("mean_rms_before")] = 1.0;
    result[QStringLiteral("mean_rms_after")] = 0.5;
    result[QStringLiteral("planetary_laser_range_summary")] = planetarySummary;

    const auto presentation =
        xjw::gui::project::buildBundleAdjustPreviewPresentation(result, 2);

    EXPECT_TRUE(presentation.detailedText.contains(
        QStringLiteral("行星激光测距 shot: 3 / 4")));
    EXPECT_TRUE(presentation.detailedText.contains(
        QStringLiteral("2.0000 m → 0.2500 m")));
    EXPECT_TRUE(presentation.detailedText.contains(QStringLiteral("MOON / IAU_MOON")));
    EXPECT_TRUE(presentation.detailedText.contains(QStringLiteral("E:/data/lola.json")));
    EXPECT_FALSE(presentation.qualityWarning);
}

TEST(PlanetaryLaserPreviewTest, WarnsWhenRangeRmsGrows)
{
    QJsonObject planetarySummary;
    planetarySummary[QStringLiteral("enabled")] = true;
    planetarySummary[QStringLiteral("range_constraint_count")] = 1;
    planetarySummary[QStringLiteral("total_shots")] = 1;
    planetarySummary[QStringLiteral("range_rms_before_m")] = 1.0;
    planetarySummary[QStringLiteral("range_rms_after_m")] = 1.5;

    QJsonObject result;
    result[QStringLiteral("planetary_laser_range_summary")] = planetarySummary;

    const auto presentation =
        xjw::gui::project::buildBundleAdjustPreviewPresentation(result, 1);
    EXPECT_TRUE(presentation.qualityWarning);
}

TEST(PlanetaryLaserPreviewTest, WarnsWhenZeroRangeResidualBecomesPositive)
{
    QJsonObject planetarySummary;
    planetarySummary[QStringLiteral("enabled")] = true;
    planetarySummary[QStringLiteral("range_constraint_count")] = 1;
    planetarySummary[QStringLiteral("total_shots")] = 1;
    planetarySummary[QStringLiteral("range_rms_before_m")] = 0.0;
    planetarySummary[QStringLiteral("range_rms_after_m")] = 0.01;

    QJsonObject result;
    result[QStringLiteral("planetary_laser_range_summary")] = planetarySummary;

    const auto presentation =
        xjw::gui::project::buildBundleAdjustPreviewPresentation(result, 1);
    EXPECT_TRUE(presentation.qualityWarning);
}
