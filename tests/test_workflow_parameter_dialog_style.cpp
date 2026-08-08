#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFormLayout>
#include <QGroupBox>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSignalSpy>
#include <QSlider>
#include <QToolButton>

#include "reconstruction/AerialTriangulationDialog.h"
#include "reconstruction/CreatePointCloudDialog.h"
#include "reconstruction/GenerateModelDialog.h"
#include "reconstruction/TextureMappingDialog.h"
#include "tie_points/CleanTiePointsDialog.h"
#include "tie_points/CreateTiePointsDialog.h"
#include "tie_points/ThinTiePointsDialog.h"

namespace
{

QJsonObject pointCloudCandidate()
{
    return QJsonObject{
        {QStringLiteral("source_data"), QStringLiteral("point_cloud")},
        {QStringLiteral("source_label"), QStringLiteral("点云")},
        {QStringLiteral("source_path"), QStringLiteral("E:/tmp/cloud.ply")},
        {QStringLiteral("display"), QStringLiteral("cloud.ply")},
        {QStringLiteral("supported"), true}
    };
}

} // namespace

TEST(WorkflowParameterDialogStyleTest, GenerateModelUsesCompactScopedLayout)
{
    GenerateModelDialog dialog;
    EXPECT_TRUE(dialog.property("workflowParameterDialog").toBool());
    EXPECT_EQ(dialog.minimumWidth(), 520);
    ASSERT_NE(dialog.layout(), nullptr);
    EXPECT_EQ(dialog.layout()->sizeConstraint(), QLayout::SetMinimumSize);

    auto *generalGroup = dialog.findChild<QGroupBox *>(QStringLiteral("workflowGeneralGroup"));
    auto *regionGroup = dialog.findChild<QGroupBox *>(QStringLiteral("workflowRegionGroup"));
    auto *advancedGroup = dialog.findChild<QGroupBox *>(QStringLiteral("workflowAdvancedGroup"));
    ASSERT_NE(generalGroup, nullptr);
    ASSERT_NE(regionGroup, nullptr);
    ASSERT_NE(advancedGroup, nullptr);

    auto *generalForm = qobject_cast<QFormLayout *>(generalGroup->layout());
    ASSERT_NE(generalForm, nullptr);
    EXPECT_EQ(generalForm->fieldGrowthPolicy(), QFormLayout::AllNonFixedFieldsGrow);
    EXPECT_EQ(generalForm->horizontalSpacing(), 12);
    EXPECT_EQ(generalForm->verticalSpacing(), 6);

    auto *sourceItems = dialog.findChild<QComboBox *>(QStringLiteral("modelSourceItemCombo"));
    ASSERT_NE(sourceItems, nullptr);
    EXPECT_EQ(sourceItems->sizeAdjustPolicy(), QComboBox::AdjustToMinimumContentsLengthWithIcon);
    EXPECT_EQ(sourceItems->minimumContentsLength(), 24);
    EXPECT_TRUE(sourceItems->isHidden());
    EXPECT_EQ(generalForm->labelForField(sourceItems), nullptr);
    EXPECT_EQ(dialog.findChild<QLabel *>(QStringLiteral("workflowStatusLabel")), nullptr);

    auto *buttonBox = dialog.findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox"));
    auto *scrollArea = dialog.findChild<QScrollArea *>(QStringLiteral("workflowParameterScrollArea"));
    ASSERT_NE(buttonBox, nullptr);
    ASSERT_NE(scrollArea, nullptr);
    EXPECT_TRUE(scrollArea->widgetResizable());
    EXPECT_GT(scrollArea->minimumHeight(), 0);
    EXPECT_EQ(scrollArea->minimumHeight(), scrollArea->maximumHeight());
    EXPECT_EQ(scrollArea->horizontalScrollBarPolicy(), Qt::ScrollBarAlwaysOff);
    EXPECT_EQ(scrollArea->findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox")), nullptr);
    EXPECT_NE(scrollArea->findChild<QGroupBox *>(QStringLiteral("workflowAdvancedGroup")), nullptr);
    EXPECT_TRUE(buttonBox->centerButtons());
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Ok)->text(), QStringLiteral("确定"));
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Cancel)->text(), QStringLiteral("取消"));
}

TEST(WorkflowParameterDialogStyleTest, CreatePointCloudMatchesMetashapeParameterLayout)
{
    CreatePointCloudDialog dialog;
    dialog.setProjectState(true, true, true);

    EXPECT_TRUE(dialog.property("workflowParameterDialog").toBool());
    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("创建点云"));
    ASSERT_NE(dialog.findChild<QGroupBox *>(QStringLiteral("pointCloudGeneralGroup")), nullptr);
    ASSERT_NE(dialog.findChild<QGroupBox *>(QStringLiteral("pointCloudAdvancedGroup")), nullptr);

    auto *source = dialog.findChild<QLabel *>(QStringLiteral("pointCloudSourceValueLabel"));
    auto *quality = dialog.findChild<QComboBox *>(QStringLiteral("pointCloudQualityCombo"));
    auto *mvs_backend = dialog.findChild<QComboBox *>(
        QStringLiteral("pointCloudMvsBackendCombo"));
    auto *point_backend = dialog.findChild<QComboBox *>(
        QStringLiteral("pointCloudProcessingBackendCombo"));
    auto *filter = dialog.findChild<QComboBox *>(QStringLiteral("pointCloudDepthFilterCombo"));
    auto *reuse = dialog.findChild<QCheckBox *>(QStringLiteral("reuseDepthMapsCheck"));
    auto *colors = dialog.findChild<QCheckBox *>(QStringLiteral("calculatePointColorsCheck"));
    auto *confidence_note =
        dialog.findChild<QLabel *>(QStringLiteral("pointCloudConfidenceNote"));
    auto *replace =
        dialog.findChild<QCheckBox *>(QStringLiteral("replaceDefaultPointCloudCheck"));
    ASSERT_NE(source, nullptr);
    ASSERT_NE(quality, nullptr);
    ASSERT_NE(mvs_backend, nullptr);
    ASSERT_NE(point_backend, nullptr);
    ASSERT_NE(filter, nullptr);
    ASSERT_NE(reuse, nullptr);
    ASSERT_NE(colors, nullptr);
    ASSERT_NE(confidence_note, nullptr);
    ASSERT_NE(replace, nullptr);

    EXPECT_TRUE(source->text().contains(QStringLiteral("深度图")));
    EXPECT_EQ(quality->currentData().toString(), QStringLiteral("highest"));
    EXPECT_EQ(mvs_backend->currentData().toString(), QStringLiteral("auto"));
    EXPECT_EQ(point_backend->currentData().toString(), QStringLiteral("auto"));
    EXPECT_TRUE(mvs_backend->currentText().contains(QStringLiteral("CUDA")));
    EXPECT_TRUE(mvs_backend->currentText().contains(QStringLiteral("OpenCL")));
    EXPECT_EQ(filter->currentData().toString(), QStringLiteral("mild"));
    EXPECT_TRUE(reuse->isChecked());
    EXPECT_TRUE(colors->isChecked());
    EXPECT_TRUE(confidence_note->text().contains(QStringLiteral("暂不保存")));
    EXPECT_EQ(dialog.findChild<QCheckBox *>(
                  QStringLiteral("calculatePointConfidenceCheck")), nullptr);
    EXPECT_TRUE(replace->isEnabled());
}

TEST(WorkflowParameterDialogStyleTest, CreatePointCloudRestoresAndSubmitsEffectiveSettings)
{
    CreatePointCloudDialog dialog;
    dialog.applySettings(QJsonObject{
        {QStringLiteral("qualityProfile"), QStringLiteral("high")},
        {QStringLiteral("patchMatchBackend"), QStringLiteral("opencl")},
        {QStringLiteral("processingDevice"), QStringLiteral("cuda")},
        {QStringLiteral("depthFilterMode"), QStringLiteral("aggressive")},
        {QStringLiteral("reuseDepthMaps"), false},
        {QStringLiteral("saveAfterEachStep"), true},
        {QStringLiteral("calculatePointColors"), false},
        {QStringLiteral("calculatePointConfidence"), true},
        {QStringLiteral("replaceDefaultPointCloud"), true}
    });
    dialog.setProjectState(true, true, true);

    QSignalSpy run_spy(&dialog, &CreatePointCloudDialog::runRequested);
    auto *button_box = dialog.findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(button_box, nullptr);
    button_box->button(QDialogButtonBox::Ok)->click();
    ASSERT_EQ(run_spy.count(), 1);

    const QJsonObject settings = run_spy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("source_data")).toString(),
              QStringLiteral("depth_maps"));
    EXPECT_EQ(settings.value(QStringLiteral("qualityProfile")).toString(),
              QStringLiteral("high"));
    EXPECT_EQ(settings.value(QStringLiteral("patchMatchBackend")).toString(),
              QStringLiteral("opencl"));
    EXPECT_EQ(settings.value(QStringLiteral("processingDevice")).toString(),
              QStringLiteral("cuda"));
    EXPECT_EQ(settings.value(QStringLiteral("depthFilterMode")).toString(),
              QStringLiteral("aggressive"));
    EXPECT_FALSE(settings.value(QStringLiteral("reuseDepthMaps")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("force_depth_recompute")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("saveAfterEachStep")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("keepColor")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("calculatePointConfidence")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("replaceDefaultPointCloud")).toBool());
}

TEST(WorkflowParameterDialogStyleTest, CreatePointCloudBlocksWithoutProductionSparseResult)
{
    CreatePointCloudDialog dialog;
    dialog.setProjectState(false, false, false, QStringLiteral("缺少正式空三结果"));

    auto *button_box = dialog.findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox"));
    auto *status = dialog.findChild<QLabel *>(QStringLiteral("workflowStatusLabel"));
    auto *reuse = dialog.findChild<QCheckBox *>(QStringLiteral("reuseDepthMapsCheck"));
    auto *replace =
        dialog.findChild<QCheckBox *>(QStringLiteral("replaceDefaultPointCloudCheck"));
    ASSERT_NE(button_box, nullptr);
    ASSERT_NE(status, nullptr);
    ASSERT_NE(reuse, nullptr);
    ASSERT_NE(replace, nullptr);
    EXPECT_FALSE(button_box->button(QDialogButtonBox::Ok)->isEnabled());
    EXPECT_EQ(status->text(), QStringLiteral("缺少正式空三结果"));
    EXPECT_FALSE(reuse->isEnabled());
    EXPECT_FALSE(replace->isEnabled());
}

TEST(WorkflowParameterDialogStyleTest, AdvancedSectionAndRegionControlsKeepBehavior)
{
    GenerateModelDialog dialog;
    dialog.setSourceCandidates(QJsonArray{pointCloudCandidate()});

    auto *toggle = dialog.findChild<QToolButton *>(QStringLiteral("workflowAdvancedToggle"));
    auto *advanced = dialog.findChild<QGroupBox *>(QStringLiteral("workflowAdvancedGroup"));
    auto *split = dialog.findChild<QCheckBox *>(QStringLiteral("splitRegionCheck"));
    auto *blockSize = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("blockSizeSpin"));
    auto *coordinate = dialog.findChild<QLabel *>(QStringLiteral("coordinateSystemLabel"));
    auto *origin = dialog.findChild<QLabel *>(QStringLiteral("gridOriginLabel"));
    auto *scrollArea = dialog.findChild<QScrollArea *>(QStringLiteral("workflowParameterScrollArea"));
    auto *buttonBox = dialog.findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(toggle, nullptr);
    ASSERT_NE(advanced, nullptr);
    ASSERT_NE(split, nullptr);
    ASSERT_NE(blockSize, nullptr);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_NE(origin, nullptr);
    ASSERT_NE(scrollArea, nullptr);
    ASSERT_NE(buttonBox, nullptr);

    EXPECT_FALSE(toggle->isChecked());
    EXPECT_TRUE(advanced->isHidden());
    EXPECT_FALSE(blockSize->isEnabled());
    EXPECT_FALSE(coordinate->isEnabled());
    EXPECT_FALSE(origin->isEnabled());

    split->setChecked(true);
    EXPECT_TRUE(blockSize->isEnabled());
    EXPECT_TRUE(coordinate->isEnabled());
    EXPECT_TRUE(origin->isEnabled());

    dialog.show();
    QApplication::processEvents();
    const qreal scaleFactor = qEnvironmentVariable("QT_SCALE_FACTOR", QStringLiteral("1.0")).toDouble();
    if (scaleFactor >= 2.0)
    {
        EXPECT_TRUE(scrollArea->verticalScrollBar()->isVisible());
        EXPECT_TRUE(buttonBox->isVisible());
    }
    else
    {
        EXPECT_FALSE(scrollArea->verticalScrollBar()->isVisible());
    }
    dialog.resize(dialog.width() + 160, dialog.height());
    QApplication::processEvents();
    const int resizedWidth = dialog.width();
    toggle->setChecked(true);
    QApplication::processEvents();
    EXPECT_FALSE(advanced->isHidden());
    EXPECT_EQ(dialog.width(), resizedWidth);
    toggle->setChecked(false);
    QApplication::processEvents();
    EXPECT_TRUE(advanced->isHidden());
    EXPECT_EQ(dialog.width(), resizedWidth);
}

TEST(WorkflowParameterDialogStyleTest, LayoutMigrationPreservesModelSettings)
{
    GenerateModelDialog dialog;
    dialog.applySettings(QJsonObject{
        {QStringLiteral("source_data"), QStringLiteral("point_cloud")},
        {QStringLiteral("source_path"), QStringLiteral("E:/tmp/cloud.ply")},
        {QStringLiteral("quality"), QStringLiteral("low")},
        {QStringLiteral("targetFaces"), 60000}
    });
    dialog.setSourceCandidates(QJsonArray{pointCloudCandidate()});

    QSignalSpy runSpy(&dialog, &GenerateModelDialog::runRequested);
    auto *buttonBox = dialog.findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(buttonBox, nullptr);
    buttonBox->button(QDialogButtonBox::Ok)->click();
    ASSERT_EQ(runSpy.count(), 1);

    const QJsonObject settings = runSpy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("source_data")).toString(), QStringLiteral("point_cloud"));
    EXPECT_EQ(settings.value(QStringLiteral("source_path")).toString(), QStringLiteral("E:/tmp/cloud.ply"));
    EXPECT_EQ(settings.value(QStringLiteral("quality")).toString(), QStringLiteral("low"));
    EXPECT_FALSE(settings.contains(QStringLiteral("threads")));
    EXPECT_EQ(settings.value(QStringLiteral("depthQualityProfile")).toString(),
              QStringLiteral("low"));
    EXPECT_EQ(settings.value(QStringLiteral("targetFaces")).toInt(), 60000);

    auto *depthQualityLabel =
        dialog.findChild<QLabel *>(QStringLiteral("effectiveDepthQualityLabel"));
    ASSERT_NE(depthQualityLabel, nullptr);
    EXPECT_TRUE(depthQualityLabel->text().contains(QStringLiteral("比例 0.125")));
    EXPECT_TRUE(depthQualityLabel->text().contains(QStringLiteral("4 轮")));
    EXPECT_TRUE(depthQualityLabel->text().contains(QStringLiteral("9×9 邻域")));
    EXPECT_TRUE(depthQualityLabel->text().contains(QStringLiteral("源视角 3")));
}

TEST(WorkflowParameterDialogStyleTest, TextureMappingExplainsFixedOptionsAndKeepsStableSchema)
{
    TextureMappingDialog dialog;
    EXPECT_EQ(dialog.findChild<QComboBox *>(QStringLiteral("m_textureTypeCombo")), nullptr);
    EXPECT_EQ(dialog.findChild<QComboBox *>(QStringLiteral("m_sourceCombo")), nullptr);
    EXPECT_EQ(dialog.findChild<QComboBox *>(QStringLiteral("m_uvMethodCombo")), nullptr);
    EXPECT_EQ(dialog.findChild<QCheckBox *>(QStringLiteral("m_saveEachStepCheck")), nullptr);
    EXPECT_EQ(dialog.findChild<QCheckBox *>(QStringLiteral("m_useAssignedImagesCheck")), nullptr);
    EXPECT_EQ(dialog.findChild<QCheckBox *>(QStringLiteral("m_transferTextureCheck")), nullptr);

    auto *fixed_note = dialog.findChild<QLabel *>(QStringLiteral("fixedWorkflowLabel"));
    auto *unsupported_note = dialog.findChild<QLabel *>(QStringLiteral("unsupportedOptionsNote"));
    ASSERT_NE(fixed_note, nullptr);
    ASSERT_NE(unsupported_note, nullptr);
    EXPECT_TRUE(fixed_note->text().contains(QStringLiteral("全部已定向影像")));
    EXPECT_TRUE(unsupported_note->text().contains(QStringLiteral("不支持")));

    QSignalSpy settings_spy(&dialog, &TextureMappingDialog::settingsChanged);
    dialog.applySettings(QJsonObject{
        {QStringLiteral("saveEachStep"), true},
        {QStringLiteral("useAssignedImages"), true},
        {QStringLiteral("transferTexture"), true},
        {QStringLiteral("textureSize"), 4096}
    });
    EXPECT_EQ(settings_spy.count(), 0);

    QSignalSpy run_spy(&dialog, &TextureMappingDialog::runRequested);
    auto *button_box = dialog.findChild<QDialogButtonBox *>(QStringLiteral("m_buttonBox"));
    ASSERT_NE(button_box, nullptr);
    auto *run_button = button_box->button(QDialogButtonBox::Ok);
    ASSERT_NE(run_button, nullptr);
    run_button->click();
    ASSERT_EQ(run_spy.count(), 1);
    const QJsonObject settings = run_spy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("textureType")).toString(),
              QStringLiteral("texture_mapping"));
    EXPECT_EQ(settings.value(QStringLiteral("sourceData")).toString(),
              QStringLiteral("images"));
    EXPECT_EQ(settings.value(QStringLiteral("mappingMode")).toString(),
              QStringLiteral("auto_projective"));
    EXPECT_FALSE(settings.value(QStringLiteral("saveEachStep")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("useAssignedImages")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("transferTexture")).toBool());
}

TEST(WorkflowParameterDialogStyleTest, TiePointAndAerialDialogsReuseStableControlsAndButtons)
{
    CreateTiePointsDialog create_dialog;
    ThinTiePointsDialog thin_dialog;
    CleanTiePointsDialog clean_dialog;
    AerialTriangulationDialog aerial_dialog;

    for (QDialog *dialog : {
             static_cast<QDialog *>(&create_dialog),
             static_cast<QDialog *>(&thin_dialog),
             static_cast<QDialog *>(&clean_dialog),
             static_cast<QDialog *>(&aerial_dialog)})
    {
        auto *button_box = dialog->findChild<QDialogButtonBox *>();
        ASSERT_NE(button_box, nullptr);
        ASSERT_NE(button_box->button(QDialogButtonBox::Ok), nullptr);
        ASSERT_NE(button_box->button(QDialogButtonBox::Cancel), nullptr);
        EXPECT_EQ(button_box->button(QDialogButtonBox::Ok)->text(), QStringLiteral("确定"));
        EXPECT_EQ(button_box->button(QDialogButtonBox::Cancel)->text(), QStringLiteral("取消"));
    }

    auto *accuracy_combo =
        create_dialog.findChild<QComboBox *>(QStringLiteral("m_accuracyCombo"));
    auto *generic_check =
        create_dialog.findChild<QCheckBox *>(QStringLiteral("m_genericPreselectionCheck"));
    ASSERT_NE(accuracy_combo, nullptr);
    ASSERT_NE(generic_check, nullptr);
    EXPECT_GE(accuracy_combo->minimumHeight(), 28);
    EXPECT_GE(accuracy_combo->minimumWidth(), 240);
    EXPECT_GE(generic_check->minimumHeight(), 24);
    EXPECT_EQ(create_dialog.findChild<QLabel *>(
                  QStringLiteral("m_preselectionStatusLabel")), nullptr);

    EXPECT_EQ(clean_dialog.windowTitle(), QStringLiteral("清理连接点"));
    EXPECT_EQ(clean_dialog.findChild<QSlider *>(), nullptr);

    aerial_dialog.setImageCount(6);
    auto *status = aerial_dialog.findChild<QLabel *>(QStringLiteral("m_statusLabel"));
    ASSERT_NE(status, nullptr);
    EXPECT_FALSE(status->isHidden());
    EXPECT_TRUE(status->text().contains(QStringLiteral("6")));
}

TEST(WorkflowParameterDialogStyleTest, StylesheetRulesAreScopedToWorkflowDialogs)
{
    QFile styleFile(QStringLiteral(PLASCAN_SOURCE_DIR "/resources/styles/app.qss"));
    ASSERT_TRUE(styleFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString style = QString::fromUtf8(styleFile.readAll());

    EXPECT_TRUE(style.contains(QStringLiteral("QDialog[workflowParameterDialog=\"true\"] QLabel")));
    EXPECT_TRUE(style.contains(QStringLiteral(
        "QDialog[workflowParameterDialog=\"true\"] QScrollArea#workflowParameterScrollArea")));
    EXPECT_TRUE(style.contains(QStringLiteral(
        "QDialog[workflowParameterDialog=\"true\"] QToolButton#workflowAdvancedToggle")));
    EXPECT_TRUE(style.contains(QStringLiteral(
        "QDialog[workflowParameterDialog=\"true\"] QLabel#workflowStatusLabel")));
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
