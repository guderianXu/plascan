#include "application/WorkflowSettingsDialog.h"
#include "PatchMatchCUDA.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStackedWidget>
#include <QToolButton>
#include <QWidget>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

namespace
{

    QJsonObject aerialSettings(const QJsonObject& settings)
    {
        return settings.value(QStringLiteral("workflows"))
            .toObject()
            .value(QStringLiteral("aerial_triangulation"))
            .toObject();
    }

    QJsonObject modelSettings(const QJsonObject& settings)
    {
        return settings.value(QStringLiteral("workflows"))
            .toObject()
            .value(QStringLiteral("generate_model"))
            .toObject();
    }

} // namespace

TEST(WorkflowSettingsDialogTest, DefaultsUseWorkflowScopedSchema)
{
    const QJsonObject settings = WorkflowSettingsDialog::defaultSettings();
    EXPECT_EQ(settings.value(QStringLiteral("workflow_settings_version")).toInt(), 9);
    EXPECT_EQ(settings.value(QStringLiteral("selected_workflow")).toString(), QStringLiteral("aerial_triangulation"));

    const QJsonObject workflows = settings.value(QStringLiteral("workflows")).toObject();
    EXPECT_TRUE(workflows.contains(QStringLiteral("aerial_triangulation")));
    EXPECT_FALSE(workflows.contains(QStringLiteral("reconstruction")));
    EXPECT_TRUE(workflows.contains(QStringLiteral("generate_model")));
    EXPECT_TRUE(workflows.contains(QStringLiteral("dem")));
    EXPECT_TRUE(workflows.contains(QStringLiteral("orthomosaic")));

    const QJsonObject aerial = aerialSettings(settings);
    EXPECT_EQ(aerial.value(QStringLiteral("algorithm_id")).toString(), QStringLiteral("plamatch_hct"));
    EXPECT_TRUE(aerial.value(QStringLiteral("lightglue_tensorrt_engine")).toString().isEmpty());
    EXPECT_TRUE(aerial.value(QStringLiteral("loma_r_tensorrt_package")).toString().isEmpty());
    EXPECT_EQ(aerial.value(QStringLiteral("loma_r_keypoint_budget")).toInt(), 0);
    EXPECT_DOUBLE_EQ(aerial.value(QStringLiteral("sift_maximum_ratio")).toDouble(), 0.98);
    EXPECT_DOUBLE_EQ(aerial.value(QStringLiteral("sift_minimum_adaptive_ratio")).toDouble(), 0.78);
    EXPECT_TRUE(aerial.value(QStringLiteral("adaptive_sift_ratio")).toBool());
    EXPECT_EQ(modelSettings(settings).value(QStringLiteral("compute_mode")).toString(), QStringLiteral("hybrid"));
    EXPECT_FALSE(settings.contains(QStringLiteral("threads")));
    EXPECT_FALSE(settings.contains(QStringLiteral("geometry_max_iterations")));
}

TEST(WorkflowSettingsDialogTest, ExposesModelGenerationComputeModesAndDeviceDetection)
{
    WorkflowSettingsDialog dialog;
    auto* workflowSelector = dialog.findChild<QComboBox*>(QStringLiteral("workflowSelector"));
    auto* workflowPages = dialog.findChild<QStackedWidget*>(QStringLiteral("workflowPages"));
    auto* algorithmSelector = dialog.findChild<QComboBox*>(QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto* downloadButton = dialog.findChild<QPushButton*>(QStringLiteral("aerialDownloadModelButton"));

    ASSERT_NE(workflowSelector, nullptr);
    ASSERT_NE(workflowPages, nullptr);
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(downloadButton, nullptr);
    EXPECT_EQ(workflowSelector->count(), 4);
    EXPECT_EQ(workflowSelector->currentData().toString(), QStringLiteral("aerial_triangulation"));
    EXPECT_TRUE(workflowPages->currentWidget()->isEnabled());
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("sift_lightglue")), 0);
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("auto_sift")), 0);
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("plamatch_hct")), 0);
    EXPECT_EQ(algorithmSelector->currentData().toString(), QStringLiteral("plamatch_hct"));
    EXPECT_LT(algorithmSelector->findData(QStringLiteral("cuda_sift")), 0);
    EXPECT_LT(algorithmSelector->findData(QStringLiteral("orb_binary")), 0);
    EXPECT_GE(algorithmSelector->findData(QStringLiteral("loma_r")), 0);

    workflowSelector->setCurrentIndex(workflowSelector->findData(QStringLiteral("generate_model")));
    auto* computeMode = workflowPages->currentWidget()->findChild<QComboBox*>(QStringLiteral("modelComputeModeCombo"));
    auto* cudaStatus = workflowPages->currentWidget()->findChild<QLabel*>(QStringLiteral("modelCudaDeviceStatusLabel"));
    auto* openClStatus =
        workflowPages->currentWidget()->findChild<QLabel*>(QStringLiteral("modelOpenClDeviceStatusLabel"));
    auto* policy = workflowPages->currentWidget()->findChild<QLabel*>(QStringLiteral("modelComputePolicyLabel"));
    auto* detectButton =
        workflowPages->currentWidget()->findChild<QPushButton*>(QStringLiteral("modelDetectComputeDevicesButton"));
    ASSERT_NE(computeMode, nullptr);
    ASSERT_NE(cudaStatus, nullptr);
    ASSERT_NE(openClStatus, nullptr);
    ASSERT_NE(policy, nullptr);
    ASSERT_NE(detectButton, nullptr);
    EXPECT_EQ(computeMode->count(), 3);
    EXPECT_GE(computeMode->findData(QStringLiteral("cuda")), 0);
    EXPECT_GE(computeMode->findData(QStringLiteral("opencl")), 0);
    EXPECT_GE(computeMode->findData(QStringLiteral("hybrid")), 0);
    EXPECT_FALSE(cudaStatus->text().isEmpty());
    EXPECT_FALSE(openClStatus->text().isEmpty());
    EXPECT_FALSE(policy->text().isEmpty());

    const int cuda_device_count = xjw::mvs::PatchMatchDepthEstimator::cudaDeviceCount();
    for (int device_index = 0; device_index < cuda_device_count; ++device_index)
    {
        const QString device_name =
            QString::fromStdString(xjw::mvs::PatchMatchDepthEstimator::cudaDeviceName(device_index));
        if (!device_name.isEmpty())
        {
            EXPECT_TRUE(cudaStatus->text().contains(device_name));
        }
    }
    const auto opencl_devices = xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    for (const xjw::mvs::OpenClDeviceInfo& device : opencl_devices)
    {
        EXPECT_TRUE(openClStatus->text().contains(QString::fromStdString(device.name)));
        EXPECT_TRUE(openClStatus->text().contains(QString::fromStdString(device.vendor)));
    }
}

TEST(WorkflowSettingsDialogTest, RemovedOrbSettingFallsBackToPlaMatchHct)
{
    WorkflowSettingsDialog dialog;
    QJsonObject requested = WorkflowSettingsDialog::defaultSettings();
    QJsonObject workflows = requested.value(QStringLiteral("workflows")).toObject();
    QJsonObject aerial = workflows.value(QStringLiteral("aerial_triangulation")).toObject();
    aerial[QStringLiteral("algorithm_id")] = QStringLiteral("orb_binary");
    workflows[QStringLiteral("aerial_triangulation")] = aerial;
    requested[QStringLiteral("workflows")] = workflows;

    dialog.applySettings(requested);

    auto* algorithmSelector = dialog.findChild<QComboBox*>(QStringLiteral("aerialMatchingAlgorithmCombo"));
    ASSERT_NE(algorithmSelector, nullptr);
    EXPECT_EQ(algorithmSelector->currentData().toString(), QStringLiteral("plamatch_hct"));
    EXPECT_EQ(aerialSettings(dialog.collectSettings()).value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("plamatch_hct"));
}

TEST(WorkflowSettingsDialogTest, PlaMatchHctRequiresNoExternalModel)
{
    WorkflowSettingsDialog dialog;
    auto* algorithmSelector = dialog.findChild<QComboBox*>(QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto* resourceEdit = dialog.findChild<QLineEdit*>(QStringLiteral("aerialMatchingResourceEdit"));
    auto* resourceStatus = dialog.findChild<QLabel*>(QStringLiteral("aerialMatchingResourceStatusLabel"));
    auto* downloadButton = dialog.findChild<QPushButton*>(QStringLiteral("aerialDownloadModelButton"));
    auto* resourceRow = dialog.findChild<QWidget*>(QStringLiteral("aerialMatchingResourceRow"));
    auto* lomaBudget = dialog.findChild<QComboBox*>(QStringLiteral("aerialLoMaRKeypointBudgetCombo"));
    auto* siftRatio = dialog.findChild<QDoubleSpinBox*>(QStringLiteral("aerialSiftMaximumRatioSpin"));
    auto* adaptiveSiftRatio = dialog.findChild<QCheckBox*>(QStringLiteral("aerialAdaptiveSiftRatioCheck"));
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(resourceEdit, nullptr);
    ASSERT_NE(resourceStatus, nullptr);
    ASSERT_NE(downloadButton, nullptr);
    ASSERT_NE(resourceRow, nullptr);
    ASSERT_NE(lomaBudget, nullptr);
    ASSERT_NE(siftRatio, nullptr);
    ASSERT_NE(adaptiveSiftRatio, nullptr);

    const int algorithmIndex = algorithmSelector->findData(QStringLiteral("plamatch_hct"));
    ASSERT_GE(algorithmIndex, 0);
    algorithmSelector->setCurrentIndex(algorithmIndex);

    EXPECT_FALSE(resourceEdit->isEnabled());
    EXPECT_TRUE(resourceRow->isHidden());
    EXPECT_TRUE(lomaBudget->isHidden());
    EXPECT_TRUE(siftRatio->isHidden());
    EXPECT_TRUE(adaptiveSiftRatio->isHidden());
    EXPECT_TRUE(downloadButton->isHidden());
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("无需下载模型")));
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("CPU")));
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("CUDA")));
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("OpenCL")));
    EXPECT_EQ(aerialSettings(dialog.collectSettings()).value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("plamatch_hct"));
}

TEST(WorkflowSettingsDialogTest, NormalizesModelGenerationComputeMode)
{
    QJsonObject settings = WorkflowSettingsDialog::defaultSettings();
    QJsonObject workflows = settings.value(QStringLiteral("workflows")).toObject();
    workflows[QStringLiteral("generate_model")] =
        QJsonObject{{QStringLiteral("compute_mode"), QStringLiteral("OPENCL")}};
    settings[QStringLiteral("workflows")] = workflows;

    EXPECT_EQ(
        WorkflowSettingsDialog::modelGenerationSettings(settings).value(QStringLiteral("compute_mode")).toString(),
        QStringLiteral("opencl"));

    workflows[QStringLiteral("generate_model")] =
        QJsonObject{{QStringLiteral("compute_mode"), QStringLiteral("invalid")}};
    settings[QStringLiteral("workflows")] = workflows;
    EXPECT_EQ(
        WorkflowSettingsDialog::modelGenerationSettings(settings).value(QStringLiteral("compute_mode")).toString(),
        QStringLiteral("hybrid"));
}

TEST(WorkflowSettingsDialogTest, AutoSiftRequiresNoExternalModel)
{
    WorkflowSettingsDialog dialog;
    auto* algorithmSelector = dialog.findChild<QComboBox*>(QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto* resourceEdit = dialog.findChild<QLineEdit*>(QStringLiteral("aerialMatchingResourceEdit"));
    auto* resourceStatus = dialog.findChild<QLabel*>(QStringLiteral("aerialMatchingResourceStatusLabel"));
    auto* downloadButton = dialog.findChild<QPushButton*>(QStringLiteral("aerialDownloadModelButton"));
    auto* resourceRow = dialog.findChild<QWidget*>(QStringLiteral("aerialMatchingResourceRow"));
    auto* lomaBudget = dialog.findChild<QComboBox*>(QStringLiteral("aerialLoMaRKeypointBudgetCombo"));
    auto* siftRatio = dialog.findChild<QDoubleSpinBox*>(QStringLiteral("aerialSiftMaximumRatioSpin"));
    auto* adaptiveSiftRatio = dialog.findChild<QCheckBox*>(QStringLiteral("aerialAdaptiveSiftRatioCheck"));
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(resourceEdit, nullptr);
    ASSERT_NE(downloadButton, nullptr);
    ASSERT_NE(resourceRow, nullptr);
    ASSERT_NE(lomaBudget, nullptr);
    ASSERT_NE(siftRatio, nullptr);
    ASSERT_NE(adaptiveSiftRatio, nullptr);

    const int autoSiftIndex = algorithmSelector->findData(QStringLiteral("auto_sift"));
    ASSERT_GE(autoSiftIndex, 0);
    algorithmSelector->setCurrentIndex(autoSiftIndex);

    EXPECT_FALSE(resourceEdit->isEnabled());
    EXPECT_TRUE(resourceRow->isHidden());
    EXPECT_TRUE(lomaBudget->isHidden());
    EXPECT_FALSE(siftRatio->isHidden());
    EXPECT_FALSE(adaptiveSiftRatio->isHidden());
    EXPECT_TRUE(downloadButton->isHidden());
    EXPECT_EQ(aerialSettings(dialog.collectSettings()).value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("auto_sift"));
    ASSERT_NE(resourceStatus, nullptr);
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("无需下载模型")));
    EXPECT_TRUE(resourceStatus->text().contains(QStringLiteral("回退")));

    siftRatio->setValue(0.91);
    adaptiveSiftRatio->setChecked(false);
    const QJsonObject aerial = aerialSettings(dialog.collectSettings());
    EXPECT_DOUBLE_EQ(aerial.value(QStringLiteral("sift_maximum_ratio")).toDouble(), 0.91);
    EXPECT_FALSE(aerial.value(QStringLiteral("adaptive_sift_ratio")).toBool());
}

TEST(WorkflowSettingsDialogTest, ShowsOnlyControlsUsedBySelectedMatchingAlgorithm)
{
    WorkflowSettingsDialog dialog;
    dialog.show();
    QApplication::processEvents();

    auto* algorithmSelector = dialog.findChild<QComboBox*>(QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto* resourceRow = dialog.findChild<QWidget*>(QStringLiteral("aerialMatchingResourceRow"));
    auto* resourceBrowse = dialog.findChild<QToolButton*>(QStringLiteral("aerialMatchingResourceBrowseButton"));
    auto* lomaBudget = dialog.findChild<QComboBox*>(QStringLiteral("aerialLoMaRKeypointBudgetCombo"));
    auto* siftRatio = dialog.findChild<QDoubleSpinBox*>(QStringLiteral("aerialSiftMaximumRatioSpin"));
    auto* adaptiveSiftRatio = dialog.findChild<QCheckBox*>(QStringLiteral("aerialAdaptiveSiftRatioCheck"));
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(resourceRow, nullptr);
    ASSERT_NE(resourceBrowse, nullptr);
    ASSERT_NE(lomaBudget, nullptr);
    ASSERT_NE(siftRatio, nullptr);
    ASSERT_NE(adaptiveSiftRatio, nullptr);

    algorithmSelector->setCurrentIndex(algorithmSelector->findData(QStringLiteral("sift_lightglue")));
    EXPECT_TRUE(resourceRow->isVisibleTo(&dialog));
    EXPECT_TRUE(resourceBrowse->isVisibleTo(&dialog));
    EXPECT_FALSE(lomaBudget->isVisibleTo(&dialog));
    EXPECT_FALSE(siftRatio->isVisibleTo(&dialog));
    EXPECT_FALSE(adaptiveSiftRatio->isVisibleTo(&dialog));

    algorithmSelector->setCurrentIndex(algorithmSelector->findData(QStringLiteral("loma_r")));
    EXPECT_TRUE(resourceRow->isVisibleTo(&dialog));
    EXPECT_TRUE(resourceBrowse->isVisibleTo(&dialog));
    EXPECT_TRUE(lomaBudget->isVisibleTo(&dialog));
    EXPECT_FALSE(siftRatio->isVisibleTo(&dialog));
    EXPECT_FALSE(adaptiveSiftRatio->isVisibleTo(&dialog));

    algorithmSelector->setCurrentIndex(algorithmSelector->findData(QStringLiteral("auto_sift")));
    EXPECT_FALSE(resourceRow->isVisibleTo(&dialog));
    EXPECT_FALSE(resourceBrowse->isVisibleTo(&dialog));
    EXPECT_FALSE(lomaBudget->isVisibleTo(&dialog));
    EXPECT_TRUE(siftRatio->isVisibleTo(&dialog));
    EXPECT_TRUE(adaptiveSiftRatio->isVisibleTo(&dialog));
}

TEST(WorkflowSettingsDialogTest, SwitchesAndPersistsAlgorithmSpecificTensorRtResources)
{
    WorkflowSettingsDialog dialog;
    auto* algorithmSelector = dialog.findChild<QComboBox*>(QStringLiteral("aerialMatchingAlgorithmCombo"));
    auto* resourceEdit = dialog.findChild<QLineEdit*>(QStringLiteral("aerialMatchingResourceEdit"));
    auto* lomaBudget = dialog.findChild<QComboBox*>(QStringLiteral("aerialLoMaRKeypointBudgetCombo"));
    ASSERT_NE(algorithmSelector, nullptr);
    ASSERT_NE(resourceEdit, nullptr);
    ASSERT_NE(lomaBudget, nullptr);

    algorithmSelector->setCurrentIndex(algorithmSelector->findData(QStringLiteral("sift_lightglue")));
    resourceEdit->setText(QStringLiteral("D:/models/lightglue.engine"));
    algorithmSelector->setCurrentIndex(algorithmSelector->findData(QStringLiteral("loma_r")));
    resourceEdit->setText(QStringLiteral("D:/models/loma_r_tensorrt.json"));
    lomaBudget->setCurrentIndex(lomaBudget->findData(3840));

    const QJsonObject aerial = aerialSettings(dialog.collectSettings());
    EXPECT_EQ(aerial.value(QStringLiteral("algorithm_id")).toString(), QStringLiteral("loma_r"));
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
    aerial[QStringLiteral("lightglue_tensorrt_engine")] = QStringLiteral("D:/models/lightglue.engine");
    workflows[QStringLiteral("aerial_triangulation")] = aerial;
    requested[QStringLiteral("workflows")] = workflows;

    dialog.applySettings(requested);
    const QJsonObject collected = dialog.collectSettings();
    EXPECT_EQ(collected.value(QStringLiteral("selected_workflow")).toString(), QStringLiteral("dem"));
    EXPECT_EQ(aerialSettings(collected).value(QStringLiteral("algorithm_id")).toString(),
              QStringLiteral("sift_lightglue"));
    EXPECT_EQ(aerialSettings(collected).value(QStringLiteral("lightglue_tensorrt_engine")).toString(),
              QStringLiteral("D:/models/lightglue.engine"));
}

TEST(WorkflowSettingsDialogTest, MigratesLegacyFlatSettingsWithoutKeepingTuningFields)
{
    WorkflowSettingsDialog dialog;
    QJsonObject legacy;
    legacy[QStringLiteral("workflow_settings_version")] = 2;
    legacy[QStringLiteral("algorithm_id")] = QStringLiteral("sift_lightglue");
    legacy[QStringLiteral("lightglue_tensorrt_engine")] = QStringLiteral("D:/legacy/lightglue.engine");
    legacy[QStringLiteral("sift_maximum_ratio")] = 0.9;
    legacy[QStringLiteral("sift_minimum_adaptive_ratio")] = 0.7;
    legacy[QStringLiteral("adaptive_sift_ratio")] = false;
    legacy[QStringLiteral("threads")] = 24;
    legacy[QStringLiteral("geometry_max_iterations")] = 25000;
    legacy[QStringLiteral("selected_workflow")] = QStringLiteral("reconstruction");
    legacy[QStringLiteral("workflows")] = QJsonObject{
        {QStringLiteral("reconstruction"), QJsonObject{{QStringLiteral("quality"), QStringLiteral("legacy")}}},
        {QStringLiteral("future_workflow"), QJsonObject{{QStringLiteral("enabled"), true}}}};

    dialog.applySettings(legacy);
    const QJsonObject collected = dialog.collectSettings();
    EXPECT_EQ(collected.value(QStringLiteral("workflow_settings_version")).toInt(), 9);
    EXPECT_EQ(aerialSettings(collected).value(QStringLiteral("lightglue_tensorrt_engine")).toString(),
              QStringLiteral("D:/legacy/lightglue.engine"));
    EXPECT_DOUBLE_EQ(aerialSettings(collected).value(QStringLiteral("sift_maximum_ratio")).toDouble(), 0.9);
    EXPECT_DOUBLE_EQ(aerialSettings(collected).value(QStringLiteral("sift_minimum_adaptive_ratio")).toDouble(), 0.7);
    EXPECT_FALSE(aerialSettings(collected).value(QStringLiteral("adaptive_sift_ratio")).toBool());
    EXPECT_FALSE(collected.contains(QStringLiteral("threads")));
    EXPECT_FALSE(collected.contains(QStringLiteral("geometry_max_iterations")));
    EXPECT_EQ(collected.value(QStringLiteral("selected_workflow")).toString(), QStringLiteral("aerial_triangulation"));
    const QJsonObject collected_workflows = collected.value(QStringLiteral("workflows")).toObject();
    EXPECT_FALSE(collected_workflows.contains(QStringLiteral("reconstruction")));
    EXPECT_TRUE(collected_workflows.contains(QStringLiteral("future_workflow")));
}

int main(int argc, char** argv)
{
    const bool lists_tests =
        std::any_of(argv,
                    argv + argc,
                    [](const char* argument) { return argument && std::strcmp(argument, "--gtest_list_tests") == 0; });
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
