#include "application/WorkflowSettingsDialog.h"

#include <QApplication>
#include <QComboBox>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStackedWidget>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

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

TEST(WorkflowSettingsDialogTest, ExposesFourWorkflowPagesAndLabelsUnavailableSettings)
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
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("cuda_sift")), 0);
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("loma_r")), 0);

    workflowSelector->setCurrentIndex(
        workflowSelector->findData(QStringLiteral("reconstruction")));
    EXPECT_TRUE(workflowPages->currentWidget()->isEnabled());
    auto *unavailableMessage = workflowPages->currentWidget()->findChild<QLabel *>(
        QStringLiteral("workflowUnavailableMessage_reconstruction"));
    ASSERT_NE(unavailableMessage, nullptr);
    EXPECT_TRUE(unavailableMessage->text().contains(QStringLiteral("尚未开放")));
    EXPECT_TRUE(workflowPages->currentWidget()->findChildren<QComboBox *>().isEmpty());
}

TEST(WorkflowSettingsDialogTest, CudaSiftRequiresNoExternalModel)
{
    WorkflowSettingsDialog dialog;
    auto *algorithmSelector = dialog.findChild<QComboBox *>(
        QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto *resourceEdit = dialog.findChild<QLineEdit *>(
        QStringLiteral("aerialMatchingResourceEdit"));
    auto *resourceStatus = dialog.findChild<QLabel *>(
        QStringLiteral("aerialMatchingResourceStatusLabel"));
    auto *downloadButton = dialog.findChild<QPushButton *>(
        QStringLiteral("aerialDownloadModelButton"));
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(resourceEdit, nullptr);
    ASSERT_NE(downloadButton, nullptr);

    const int cudaSiftIndex = algorithmSelector->findData(QStringLiteral("cuda_sift"));
    ASSERT_GE(cudaSiftIndex, 0);
    algorithmSelector->setCurrentIndex(cudaSiftIndex);

    EXPECT_FALSE(resourceEdit->isEnabled());
    EXPECT_TRUE(downloadButton->isHidden());
    EXPECT_EQ(aerialSettings(dialog.collectSettings())
                  .value(QStringLiteral("algorithm_id"))
                  .toString(),
              QStringLiteral("cuda_sift"));
    ASSERT_NE(resourceStatus, nullptr);
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("无需下载模型")));
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("回退")));
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
    const bool lists_tests = std::any_of(argv, argv + argc, [](const char *argument)
    {
        return argument && std::strcmp(argument, "--gtest_list_tests") == 0;
    });
    ::testing::InitGoogleTest(&argc, argv);
    if (lists_tests)
    {
        return RUN_ALL_TESTS();
    }

    // 对话框测试不需要桌面会话；固定 offscreen 避免 CI 和远程开发机依赖显示器。
    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    QApplication application(argc, argv);
    return RUN_ALL_TESTS();
}
