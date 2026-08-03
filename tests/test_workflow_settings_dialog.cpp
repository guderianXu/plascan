#include "application/WorkflowSettingsDialog.h"

#include <QApplication>
#include <QComboBox>
#include <QJsonObject>
#include <QLineEdit>
#include <QPushButton>
#include <QStackedWidget>

#include <gtest/gtest.h>

namespace
{

QJsonObject aerialSettings(const QJsonObject &settings)
{
    return settings.value(QStringLiteral("workflows"))
        .toObject()
        .value(QStringLiteral("aerial_triangulation"))
        .toObject();
}

} // namespace

TEST(WorkflowSettingsDialogTest, DefaultsUseWorkflowScopedSchema)
{
    const QJsonObject settings = WorkflowSettingsDialog::defaultSettings();
    EXPECT_EQ(settings.value(QStringLiteral("workflow_settings_version")).toInt(), 5);
    EXPECT_EQ(settings.value(QStringLiteral("selected_workflow")).toString(),
              QStringLiteral("aerial_triangulation"));

    const QJsonObject workflows = settings.value(QStringLiteral("workflows")).toObject();
    EXPECT_TRUE(workflows.contains(QStringLiteral("aerial_triangulation")));
    EXPECT_TRUE(workflows.contains(QStringLiteral("reconstruction")));
    EXPECT_TRUE(workflows.contains(QStringLiteral("dem")));
    EXPECT_TRUE(workflows.contains(QStringLiteral("orthomosaic")));

    const QJsonObject aerial = aerialSettings(settings);
    EXPECT_EQ(aerial.value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("sift_lightglue"));
    EXPECT_TRUE(aerial.value(QStringLiteral("lightglue_tensorrt_engine")).toString().isEmpty());
    EXPECT_TRUE(aerial.value(QStringLiteral("loma_r_tensorrt_package")).toString().isEmpty());
    EXPECT_EQ(aerial.value(QStringLiteral("loma_r_keypoint_budget")).toInt(), 0);
    EXPECT_FALSE(settings.contains(QStringLiteral("threads")));
    EXPECT_FALSE(settings.contains(QStringLiteral("geometry_max_iterations")));
}

TEST(WorkflowSettingsDialogTest, ExposesFourWorkflowPagesAndOnlyAerialIsEditable)
{
    WorkflowSettingsDialog dialog;
    auto *workflowSelector = dialog.findChild<QComboBox *>(QStringLiteral("workflowSelector"));
    auto *workflowPages = dialog.findChild<QStackedWidget *>(QStringLiteral("workflowPages"));
    auto *algorithmSelector = dialog.findChild<QComboBox *>(
        QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto *downloadButton = dialog.findChild<QPushButton *>(
        QStringLiteral("aerialDownloadModelButton"));

    ASSERT_NE(workflowSelector, nullptr);
    ASSERT_NE(workflowPages, nullptr);
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(downloadButton, nullptr);
    EXPECT_EQ(workflowSelector->count(), 4);
    EXPECT_EQ(workflowSelector->currentData().toString(),
              QStringLiteral("aerial_triangulation"));
    EXPECT_TRUE(workflowPages->currentWidget()->isEnabled());
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("sift_lightglue")), 0);
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("loma_r")), 0);

    workflowSelector->setCurrentIndex(
        workflowSelector->findData(QStringLiteral("reconstruction")));
    EXPECT_FALSE(workflowPages->currentWidget()->isEnabled());
}

TEST(WorkflowSettingsDialogTest, SwitchesAndPersistsAlgorithmSpecificTensorRtResources)
{
    WorkflowSettingsDialog dialog;
    auto *algorithmSelector = dialog.findChild<QComboBox *>(
        QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto *resourceEdit = dialog.findChild<QLineEdit *>(
        QStringLiteral("aerialMatchingResourceEdit"));
    auto *lomaBudget = dialog.findChild<QComboBox *>(
        QStringLiteral("aerialLoMaRKeypointBudgetCombo"));
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(resourceEdit, nullptr);
    ASSERT_NE(lomaBudget, nullptr);

    algorithmSelector->setCurrentIndex(
        algorithmSelector->findData(QStringLiteral("sift_lightglue")));
    resourceEdit->setText(QStringLiteral("D:/models/lightglue.engine"));
    algorithmSelector->setCurrentIndex(
        algorithmSelector->findData(QStringLiteral("loma_r")));
    resourceEdit->setText(QStringLiteral("D:/models/loma_r_tensorrt.json"));
    lomaBudget->setCurrentIndex(lomaBudget->findData(3840));

    const QJsonObject aerial = aerialSettings(dialog.collectSettings());
    EXPECT_EQ(aerial.value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("loma_r"));
    EXPECT_EQ(aerial.value(QStringLiteral("lightglue_tensorrt_engine")).toString(),
              QStringLiteral("D:/models/lightglue.engine"));
    EXPECT_EQ(aerial.value(QStringLiteral("loma_r_tensorrt_package")).toString(),
              QStringLiteral("D:/models/loma_r_tensorrt.json"));
    EXPECT_EQ(aerial.value(QStringLiteral("loma_r_keypoint_budget")).toInt(), 3840);
}

TEST(WorkflowSettingsDialogTest, AppliesAndCollectsWorkflowScopedAerialSettings)
{
    WorkflowSettingsDialog dialog;
    QJsonObject requested = WorkflowSettingsDialog::defaultSettings();
    requested[QStringLiteral("selected_workflow")] = QStringLiteral("dem");
    QJsonObject workflows = requested.value(QStringLiteral("workflows")).toObject();
    QJsonObject aerial = workflows.value(QStringLiteral("aerial_triangulation")).toObject();
    aerial[QStringLiteral("algorithm_id")] = QStringLiteral("sift_lightglue");
    aerial[QStringLiteral("lightglue_tensorrt_engine")] =
        QStringLiteral("D:/models/lightglue.engine");
    workflows[QStringLiteral("aerial_triangulation")] = aerial;
    requested[QStringLiteral("workflows")] = workflows;

    dialog.applySettings(requested);
    const QJsonObject collected = dialog.collectSettings();
    EXPECT_EQ(collected.value(QStringLiteral("selected_workflow")).toString(),
              QStringLiteral("dem"));
    EXPECT_EQ(aerialSettings(collected).value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("sift_lightglue"));
    EXPECT_EQ(
        aerialSettings(collected).value(QStringLiteral("lightglue_tensorrt_engine")).toString(),
        QStringLiteral("D:/models/lightglue.engine"));
}

TEST(WorkflowSettingsDialogTest, MigratesLegacyFlatSettingsWithoutKeepingTuningFields)
{
    WorkflowSettingsDialog dialog;
    QJsonObject legacy;
    legacy[QStringLiteral("workflow_settings_version")] = 2;
    legacy[QStringLiteral("algorithm_id")] = QStringLiteral("sift_lightglue");
    legacy[QStringLiteral("lightglue_tensorrt_engine")] =
        QStringLiteral("D:/legacy/lightglue.engine");
    legacy[QStringLiteral("threads")] = 24;
    legacy[QStringLiteral("geometry_max_iterations")] = 25000;

    dialog.applySettings(legacy);
    const QJsonObject collected = dialog.collectSettings();
    EXPECT_EQ(collected.value(QStringLiteral("workflow_settings_version")).toInt(), 5);
    EXPECT_EQ(
        aerialSettings(collected).value(QStringLiteral("lightglue_tensorrt_engine")).toString(),
        QStringLiteral("D:/legacy/lightglue.engine"));
    EXPECT_FALSE(collected.contains(QStringLiteral("threads")));
    EXPECT_FALSE(collected.contains(QStringLiteral("geometry_max_iterations")));
}

int main(int argc, char **argv)
{
    // 对话框测试不需要桌面会话；固定 offscreen 避免 CI 和远程开发机依赖显示器。
    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    QApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
