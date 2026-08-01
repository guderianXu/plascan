#include "application/WorkflowSettingsDialog.h"

#include <QApplication>
#include <QJsonObject>

#include <gtest/gtest.h>

TEST(WorkflowSettingsDialogTest, DefaultsDescribeOnlyRegisteredProductionAlgorithm)
{
    const QJsonObject settings = WorkflowSettingsDialog::defaultSettings();
    EXPECT_EQ(settings.value(QStringLiteral("workflow_settings_version")).toInt(), 2);
    EXPECT_EQ(settings.value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("sift_lightglue"));
    EXPECT_EQ(settings.value(QStringLiteral("device")).toString(),
              QStringLiteral("cuda"));
    EXPECT_TRUE(settings.value(QStringLiteral("lightglue_tensorrt_engine")).toString().isEmpty());
    EXPECT_GT(settings.value(QStringLiteral("threads")).toInt(), 0);
    EXPECT_GT(settings.value(QStringLiteral("geometry_max_iterations")).toInt(), 0);
}

TEST(WorkflowSettingsDialogTest, AppliesAndCollectsAerialTriangulationDetails)
{
    WorkflowSettingsDialog dialog;
    QJsonObject requested;
    requested[QStringLiteral("threads")] = 12;
    requested[QStringLiteral("cuda_device")] = 2;
    requested[QStringLiteral("lightglue_tensorrt_engine")] =
        QStringLiteral("D:/models/lightglue.engine");
    requested[QStringLiteral("cuda_parallel_pairs")] = 3;
    requested[QStringLiteral("feature_prefetch_depth")] = 4;
    requested[QStringLiteral("feature_max_image_dim")] = 4096;
    requested[QStringLiteral("match_threshold")] = 0.275;
    requested[QStringLiteral("geometry_reprojection_threshold_px")] = 2.25;
    requested[QStringLiteral("geometry_min_inliers")] = 48;
    requested[QStringLiteral("geometry_max_iterations")] = 25000;
    requested[QStringLiteral("tie_point_grid_columns")] = 10;
    requested[QStringLiteral("tie_point_grid_rows")] = 6;
    requested[QStringLiteral("tie_point_grid_cell_limit")] = 75;
    requested[QStringLiteral("stationary_tie_point_max_pixel_motion")] = 1.75;

    dialog.applySettings(requested);
    const QJsonObject collected = dialog.collectSettings();
    EXPECT_EQ(collected.value(QStringLiteral("threads")).toInt(), 12);
    EXPECT_EQ(collected.value(QStringLiteral("cuda_device")).toInt(), 2);
    EXPECT_EQ(collected.value(QStringLiteral("lightglue_tensorrt_engine")).toString(),
              QStringLiteral("D:/models/lightglue.engine"));
    EXPECT_EQ(collected.value(QStringLiteral("cuda_parallel_pairs")).toInt(), 3);
    EXPECT_EQ(collected.value(QStringLiteral("feature_prefetch_depth")).toInt(), 4);
    EXPECT_EQ(collected.value(QStringLiteral("feature_max_image_dim")).toInt(), 4096);
    EXPECT_DOUBLE_EQ(collected.value(QStringLiteral("match_threshold")).toDouble(), 0.275);
    EXPECT_DOUBLE_EQ(
        collected.value(QStringLiteral("geometry_reprojection_threshold_px")).toDouble(),
        2.25);
    EXPECT_EQ(collected.value(QStringLiteral("geometry_min_inliers")).toInt(), 48);
    EXPECT_EQ(collected.value(QStringLiteral("geometry_max_iterations")).toInt(), 25000);
    EXPECT_EQ(collected.value(QStringLiteral("tie_point_grid_columns")).toInt(), 10);
    EXPECT_EQ(collected.value(QStringLiteral("tie_point_grid_rows")).toInt(), 6);
    EXPECT_EQ(collected.value(QStringLiteral("tie_point_grid_cell_limit")).toInt(), 75);
    EXPECT_DOUBLE_EQ(
        collected.value(QStringLiteral("stationary_tie_point_max_pixel_motion")).toDouble(),
        1.75);
}

TEST(WorkflowSettingsDialogTest, FillsMissingFieldsFromStableDefaults)
{
    WorkflowSettingsDialog dialog;
    dialog.applySettings(QJsonObject{{QStringLiteral("threads"), 3}});
    const QJsonObject collected = dialog.collectSettings();
    const QJsonObject defaults = WorkflowSettingsDialog::defaultSettings();

    EXPECT_EQ(collected.value(QStringLiteral("threads")).toInt(), 3);
    EXPECT_EQ(collected.value(QStringLiteral("geometry_min_inliers")),
              defaults.value(QStringLiteral("geometry_min_inliers")));
    EXPECT_EQ(collected.value(QStringLiteral("tie_point_grid_columns")),
              defaults.value(QStringLiteral("tie_point_grid_columns")));
}

int main(int argc, char **argv)
{
    // 对话框测试不需要桌面会话；固定 offscreen 避免 CI 和远程开发机依赖显示器。
    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    QApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
