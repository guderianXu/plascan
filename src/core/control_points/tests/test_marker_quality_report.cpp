#include <gtest/gtest.h>

#include "quality/MarkerQualityReport.h"

#include <cmath>

TEST(MarkerQualityReportTest, SeparatesControlAndCheckPointStatistics)
{
    xjw::control_points::ControlNetworkResult network;
    network.ok = true;
    network.controlResiduals = {
        {"C1", xjw::control_points::MarkerRole::ControlPoint,
         {{0.1, 0.0, 0.0}}, 0.1, 1.0, true},
        {"C2", xjw::control_points::MarkerRole::ControlPoint,
         {{0.0, 0.2, 0.0}}, 0.2, 2.0, true},
        {"C3", xjw::control_points::MarkerRole::ControlPoint,
         {{1.0, 0.0, 0.0}}, 1.0, 10.0, false},
    };
    network.checkPointResiduals = {
        {"K1", xjw::control_points::MarkerRole::CheckPoint,
         {{0.0, 0.0, 0.3}}, 0.3, 3.0, false},
    };

    const auto report = xjw::control_points::buildMarkerQualityReport(network, {});

    EXPECT_TRUE(report.valid);
    EXPECT_EQ(report.controls.totalCount, 3);
    EXPECT_EQ(report.controls.inlierCount, 2);
    EXPECT_NEAR(report.controls.rms, std::sqrt((0.01 + 0.04 + 1.0) / 3.0), 1.0e-12);
    EXPECT_EQ(report.checkPoints.totalCount, 1);
    EXPECT_NEAR(report.checkPoints.rms, 0.3, 1.0e-12);
}

TEST(MarkerQualityReportTest, SeparatesControlAndCheckScaleBars)
{
    xjw::control_points::ControlNetworkResult network;
    network.ok = true;
    QVector<xjw::control_points::ScaleBarResidual> residuals{
        {QStringLiteral("S1"), xjw::control_points::ScaleBarRole::Control, 2.0, 2.02, 0.02},
        {QStringLiteral("S2"), xjw::control_points::ScaleBarRole::Check, 3.0, 2.90, -0.10},
    };

    const auto report = xjw::control_points::buildMarkerQualityReport(network, residuals);

    EXPECT_EQ(report.controlScaleBars.totalCount, 1);
    EXPECT_NEAR(report.controlScaleBars.rms, 0.02, 1.0e-12);
    EXPECT_EQ(report.checkScaleBars.totalCount, 1);
    EXPECT_NEAR(report.checkScaleBars.rms, 0.10, 1.0e-12);
}
