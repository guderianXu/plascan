#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFile>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSignalSpy>
#include <QSlider>
#include <QStandardItemModel>
#include <QToolButton>
#include <QTemporaryDir>

#include "reconstruction/AerialTriangulationDialog.h"
#include "reconstruction/CreatePointCloudDialog.h"
#include "reconstruction/CreateDemDialog.h"
#include "reconstruction/GenerateModelDialog.h"
#include "reconstruction/TextureMappingDialog.h"
#include "tie_points/CleanTiePointsDialog.h"
#include "tie_points/CreateTiePointsDialog.h"
#include "tie_points/ThinTiePointsDialog.h"

#include <algorithm>
#include <cstring>

namespace
{

    QJsonObject depthMapsCandidate()
    {
        return QJsonObject{{QStringLiteral("source_data"), QStringLiteral("depth_maps")},
                           {QStringLiteral("source_label"), QStringLiteral("深度图")},
                           {QStringLiteral("source_path"), QStringLiteral("E:/tmp/depth")},
                           {QStringLiteral("display"), QStringLiteral("depth")},
                           {QStringLiteral("supported"), true}};
    }

} // namespace

TEST(WorkflowParameterDialogStyleTest, CreateDemHasOnlyOnePrimaryActionRow)
{
    CreateDemDialog dialog;
    dialog.setAvailableImages({QStringLiteral("E:/images/left.tif"),
                               QStringLiteral("E:/images/right.tif")});

    auto *mode = dialog.findChild<QComboBox *>(QStringLiteral("m_modeCombo"));
    ASSERT_NE(mode, nullptr);
    mode->setCurrentIndex(1);

    const QList<QPushButton *> run_buttons =
        dialog.findChildren<QPushButton *>(QStringLiteral("m_runBtn"));
    const QList<QPushButton *> close_buttons =
        dialog.findChildren<QPushButton *>(QStringLiteral("m_closeBtn"));
    ASSERT_EQ(run_buttons.size(), 1);
    ASSERT_EQ(close_buttons.size(), 1);
    EXPECT_EQ(run_buttons.constFirst()->text(), QStringLiteral("生成影像立体 DEM"));
    EXPECT_EQ(close_buttons.constFirst()->text(), QStringLiteral("关闭"));
}

TEST(WorkflowParameterDialogStyleTest, GenerateModelMatchesReferenceDialogLayout)
{
    GenerateModelDialog dialog;
    EXPECT_EQ(dialog.objectName(), QStringLiteral("BuildModelDialog"));
    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("生成网格"));
    EXPECT_TRUE(dialog.windowFlags().testFlag(Qt::FramelessWindowHint));
    EXPECT_TRUE(dialog.isModal());
    EXPECT_EQ(dialog.minimumWidth(), 404);
    EXPECT_EQ(dialog.maximumWidth(), 404);
    ASSERT_NE(dialog.layout(), nullptr);
    EXPECT_EQ(dialog.layout()->contentsMargins(), QMargins(12, 10, 12, 10));
    EXPECT_EQ(dialog.layout()->spacing(), 4);

    auto* title = dialog.findChild<QLabel*>(QStringLiteral("DialogTitle"));
    auto* close = dialog.findChild<QPushButton*>(QStringLiteral("DialogClose"));
    auto* generalGroup = dialog.findChild<QFrame*>(QStringLiteral("workflowGeneralGroup"));
    auto* blocksGroup = dialog.findChild<QFrame*>(QStringLiteral("groupBlocks"));
    auto* advancedGroup = dialog.findChild<QFrame*>(QStringLiteral("workflowAdvancedGroup"));
    ASSERT_NE(title, nullptr);
    ASSERT_NE(close, nullptr);
    ASSERT_NE(generalGroup, nullptr);
    ASSERT_NE(blocksGroup, nullptr);
    ASSERT_NE(advancedGroup, nullptr);
    EXPECT_EQ(title->text(), QStringLiteral("生成网格"));
    EXPECT_EQ(close->text(), QStringLiteral("×"));

    auto* generalBody = generalGroup->findChild<QFrame*>(QStringLiteral("SectionBody"));
    ASSERT_NE(generalBody, nullptr);
    auto* generalForm = qobject_cast<QFormLayout*>(generalBody->layout());
    ASSERT_NE(generalForm, nullptr);
    EXPECT_EQ(generalForm->fieldGrowthPolicy(), QFormLayout::AllNonFixedFieldsGrow);
    EXPECT_EQ(generalForm->horizontalSpacing(), 12);
    EXPECT_EQ(generalForm->verticalSpacing(), 6);

    EXPECT_NE(dialog.findChild<QComboBox*>(QStringLiteral("modelSurfaceTypeCombo")), nullptr);
    auto* quality = dialog.findChild<QComboBox*>(QStringLiteral("modelQualityCombo"));
    auto* interpolation = dialog.findChild<QComboBox*>(QStringLiteral("modelInterpolationCombo"));
    auto* filtering = dialog.findChild<QComboBox*>(QStringLiteral("modelDepthFilteringCombo"));
    ASSERT_NE(quality, nullptr);
    ASSERT_NE(interpolation, nullptr);
    ASSERT_NE(filtering, nullptr);
    ASSERT_EQ(quality->count(), 5);
    EXPECT_EQ(quality->itemData(0).toString(), QStringLiteral("highest"));
    EXPECT_EQ(quality->itemData(1).toString(), QStringLiteral("high"));
    EXPECT_EQ(quality->itemData(2).toString(), QStringLiteral("medium"));
    EXPECT_EQ(quality->itemData(3).toString(), QStringLiteral("low"));
    EXPECT_EQ(quality->itemData(4).toString(), QStringLiteral("lowest"));
    ASSERT_EQ(interpolation->count(), 3);
    EXPECT_EQ(interpolation->itemData(0).toString(), QStringLiteral("disabled"));
    EXPECT_EQ(interpolation->itemData(1).toString(), QStringLiteral("enabled"));
    EXPECT_EQ(interpolation->itemData(2).toString(), QStringLiteral("extrapolated"));
    EXPECT_EQ(filtering->currentData().toString(), QStringLiteral("mild"));
    EXPECT_NE(dialog.findChild<QCheckBox*>(QStringLiteral("checkSplitInBlocks")), nullptr);
    EXPECT_NE(dialog.findChild<QDoubleSpinBox*>(QStringLiteral("editBlocksSize")), nullptr);
    auto* saveAfterStep = dialog.findChild<QCheckBox*>(QStringLiteral("checkSaveProject"));
    auto* strictMasks = dialog.findChild<QCheckBox*>(QStringLiteral("checkStrictVolumetricMasks"));
    ASSERT_NE(saveAfterStep, nullptr);
    ASSERT_NE(strictMasks, nullptr);
    EXPECT_FALSE(saveAfterStep->isEnabled());
    EXPECT_FALSE(strictMasks->isEnabled());
    EXPECT_NE(dialog.findChild<QCheckBox*>(QStringLiteral("checkVertexColors")), nullptr);
    EXPECT_NE(dialog.findChild<QPushButton*>(QStringLiteral("buttonBlocksPreview")), nullptr);

    auto* sourceItems = dialog.findChild<QComboBox*>(QStringLiteral("modelSourceItemCombo"));
    ASSERT_NE(sourceItems, nullptr);
    EXPECT_EQ(sourceItems->sizeAdjustPolicy(), QComboBox::AdjustToMinimumContentsLengthWithIcon);
    EXPECT_EQ(sourceItems->minimumContentsLength(), 10);
    EXPECT_TRUE(sourceItems->isHidden());
    EXPECT_EQ(generalForm->labelForField(sourceItems), nullptr);
    EXPECT_EQ(dialog.findChild<QLabel*>(QStringLiteral("workflowStatusLabel")), nullptr);

    auto* faceCountMode = dialog.findChild<QComboBox*>(QStringLiteral("modelFaceCountModeCombo"));
    auto* customFaceCount = dialog.findChild<QSpinBox*>(QStringLiteral("modelCustomFaceCountSpin"));
    ASSERT_NE(faceCountMode, nullptr);
    ASSERT_NE(customFaceCount, nullptr);
    ASSERT_EQ(faceCountMode->count(), 4);
    EXPECT_EQ(faceCountMode->itemData(0).toString(), QStringLiteral("high"));
    EXPECT_EQ(faceCountMode->itemData(1).toString(), QStringLiteral("medium"));
    EXPECT_EQ(faceCountMode->itemData(2).toString(), QStringLiteral("low"));
    EXPECT_EQ(faceCountMode->itemData(3).toString(), QStringLiteral("custom"));
    EXPECT_EQ(faceCountMode->itemText(0), QStringLiteral("高"));
    EXPECT_EQ(faceCountMode->itemText(1), QStringLiteral("中"));
    EXPECT_EQ(faceCountMode->itemText(2), QStringLiteral("低"));
    EXPECT_EQ(faceCountMode->itemText(3), QStringLiteral("自定义"));
    EXPECT_EQ(customFaceCount->minimum(), 1);
    EXPECT_EQ(customFaceCount->maximum(), 2000000);

    auto* buttonBox = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(buttonBox, nullptr);
    EXPECT_EQ(dialog.findChild<QScrollArea*>(QStringLiteral("workflowParameterScrollArea")), nullptr);
    EXPECT_TRUE(buttonBox->centerButtons());
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Ok)->text(), QStringLiteral("OK"));
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Cancel)->text(), QStringLiteral("Cancel"));
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Ok)->width(), 86);
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Cancel)->width(), 86);
}

TEST(WorkflowParameterDialogStyleTest, CreatePointCloudMatchesMetashapeParameterLayout)
{
    CreatePointCloudDialog dialog;
    dialog.setProjectState(true, true, true);

    EXPECT_TRUE(dialog.property("workflowParameterDialog").toBool());
    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("创建点云"));
    ASSERT_NE(dialog.findChild<QGroupBox*>(QStringLiteral("pointCloudGeneralGroup")), nullptr);
    ASSERT_NE(dialog.findChild<QGroupBox*>(QStringLiteral("pointCloudAdvancedGroup")), nullptr);

    auto* source = dialog.findChild<QLabel*>(QStringLiteral("pointCloudSourceValueLabel"));
    auto* quality = dialog.findChild<QComboBox*>(QStringLiteral("pointCloudQualityCombo"));
    auto* mvs_backend = dialog.findChild<QComboBox*>(QStringLiteral("pointCloudMvsBackendCombo"));
    auto* point_backend = dialog.findChild<QComboBox*>(QStringLiteral("pointCloudProcessingBackendCombo"));
    auto* filter = dialog.findChild<QComboBox*>(QStringLiteral("pointCloudDepthFilterCombo"));
    auto* reuse = dialog.findChild<QCheckBox*>(QStringLiteral("reuseDepthMapsCheck"));
    auto* colors = dialog.findChild<QCheckBox*>(QStringLiteral("calculatePointColorsCheck"));
    auto* confidence_note = dialog.findChild<QLabel*>(QStringLiteral("pointCloudConfidenceNote"));
    auto* replace = dialog.findChild<QCheckBox*>(QStringLiteral("replaceDefaultPointCloudCheck"));
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
    EXPECT_EQ(dialog.findChild<QCheckBox*>(QStringLiteral("calculatePointConfidenceCheck")), nullptr);
    EXPECT_TRUE(replace->isEnabled());
}

TEST(WorkflowParameterDialogStyleTest, CreatePointCloudRestoresAndSubmitsEffectiveSettings)
{
    CreatePointCloudDialog dialog;
    dialog.applySettings(QJsonObject{{QStringLiteral("qualityProfile"), QStringLiteral("high")},
                                     {QStringLiteral("patchMatchBackend"), QStringLiteral("opencl")},
                                     {QStringLiteral("processingDevice"), QStringLiteral("cuda")},
                                     {QStringLiteral("depthFilterMode"), QStringLiteral("aggressive")},
                                     {QStringLiteral("reuseDepthMaps"), false},
                                     {QStringLiteral("saveAfterEachStep"), true},
                                     {QStringLiteral("calculatePointColors"), false},
                                     {QStringLiteral("calculatePointConfidence"), true},
                                     {QStringLiteral("replaceDefaultPointCloud"), true}});
    dialog.setProjectState(true, true, true);

    QSignalSpy run_spy(&dialog, &CreatePointCloudDialog::runRequested);
    auto* button_box = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(button_box, nullptr);
    button_box->button(QDialogButtonBox::Ok)->click();
    ASSERT_EQ(run_spy.count(), 1);

    const QJsonObject settings = run_spy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("source_data")).toString(), QStringLiteral("depth_maps"));
    EXPECT_EQ(settings.value(QStringLiteral("qualityProfile")).toString(), QStringLiteral("high"));
    EXPECT_EQ(settings.value(QStringLiteral("patchMatchBackend")).toString(), QStringLiteral("opencl"));
    EXPECT_EQ(settings.value(QStringLiteral("processingDevice")).toString(), QStringLiteral("cuda"));
    EXPECT_EQ(settings.value(QStringLiteral("depthFilterMode")).toString(), QStringLiteral("aggressive"));
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

    auto* button_box = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    auto* status = dialog.findChild<QLabel*>(QStringLiteral("workflowStatusLabel"));
    auto* reuse = dialog.findChild<QCheckBox*>(QStringLiteral("reuseDepthMapsCheck"));
    auto* replace = dialog.findChild<QCheckBox*>(QStringLiteral("replaceDefaultPointCloudCheck"));
    ASSERT_NE(button_box, nullptr);
    ASSERT_NE(status, nullptr);
    ASSERT_NE(reuse, nullptr);
    ASSERT_NE(replace, nullptr);
    EXPECT_FALSE(button_box->button(QDialogButtonBox::Ok)->isEnabled());
    EXPECT_EQ(status->text(), QStringLiteral("缺少正式空三结果"));
    EXPECT_FALSE(reuse->isEnabled());
    EXPECT_FALSE(replace->isEnabled());
}

TEST(WorkflowParameterDialogStyleTest, AdvancedSectionUsesReferenceCollapsibleControls)
{
    GenerateModelDialog dialog;
    dialog.setSourceCandidates(QJsonArray{depthMapsCandidate()});

    QToolButton* toggle = nullptr;
    for (QToolButton* candidate : dialog.findChildren<QToolButton*>(QStringLiteral("SectionHeader")))
    {
        if (candidate->text() == QStringLiteral("高级"))
        {
            toggle = candidate;
            break;
        }
    }
    auto* advancedSection = dialog.findChild<QFrame*>(QStringLiteral("workflowAdvancedGroup"));
    auto* advanced = advancedSection ? advancedSection->findChild<QFrame*>(QStringLiteral("SectionBody")) : nullptr;
    auto* buttonBox = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(toggle, nullptr);
    ASSERT_NE(advanced, nullptr);
    ASSERT_NE(buttonBox, nullptr);

    EXPECT_FALSE(toggle->isChecked());
    EXPECT_TRUE(advanced->isHidden());

    dialog.show();
    QApplication::processEvents();
    const int resizedWidth = dialog.width();
    const int collapsedHeight = dialog.height();
    toggle->setChecked(true);
    QApplication::processEvents();
    EXPECT_FALSE(advanced->isHidden());
    EXPECT_EQ(dialog.width(), resizedWidth);
    EXPECT_GT(dialog.height(), collapsedHeight);
    toggle->setChecked(false);
    QApplication::processEvents();
    EXPECT_TRUE(advanced->isHidden());
    EXPECT_EQ(dialog.width(), resizedWidth);
    EXPECT_TRUE(buttonBox->isVisible());
}

TEST(WorkflowParameterDialogStyleTest, ModelBlockControlsExposeReferenceStateWithoutSilentFallback)
{
    GenerateModelDialog dialog;
    dialog.setSourceCandidates(QJsonArray{depthMapsCandidate()});

    auto* split = dialog.findChild<QCheckBox*>(QStringLiteral("checkSplitInBlocks"));
    auto* blockSize = dialog.findChild<QDoubleSpinBox*>(QStringLiteral("editBlocksSize"));
    auto* preview = dialog.findChild<QPushButton*>(QStringLiteral("buttonBlocksPreview"));
    auto* buttonBox = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(split, nullptr);
    ASSERT_NE(blockSize, nullptr);
    ASSERT_NE(preview, nullptr);
    ASSERT_NE(buttonBox, nullptr);
    ASSERT_NE(buttonBox->button(QDialogButtonBox::Ok), nullptr);

    EXPECT_FALSE(split->isChecked());
    EXPECT_FALSE(blockSize->isEnabled());
    EXPECT_FALSE(preview->isEnabled());
    EXPECT_TRUE(buttonBox->button(QDialogButtonBox::Ok)->isEnabled());

    split->setChecked(true);
    EXPECT_TRUE(blockSize->isEnabled());
    EXPECT_TRUE(preview->isEnabled());
    EXPECT_FALSE(buttonBox->button(QDialogButtonBox::Ok)->isEnabled());
    EXPECT_TRUE(buttonBox->button(QDialogButtonBox::Ok)->toolTip().contains(QStringLiteral("分块模型调度")));

    split->setChecked(false);
    EXPECT_FALSE(blockSize->isEnabled());
    EXPECT_FALSE(preview->isEnabled());
    EXPECT_TRUE(buttonBox->button(QDialogButtonBox::Ok)->isEnabled());
}

TEST(WorkflowParameterDialogStyleTest, LayoutMigrationCanonicalizesModelSettings)
{
    GenerateModelDialog dialog;
    dialog.applySettings(QJsonObject{{QStringLiteral("source_data"), QStringLiteral("depth_maps")},
                                     {QStringLiteral("source_path"), QStringLiteral("E:/tmp/depth")},
                                     {QStringLiteral("quality"), QStringLiteral("low")},
                                     {QStringLiteral("qualityProfile"), QStringLiteral("lite")},
                                     {QStringLiteral("modelQualityProfile"), QStringLiteral("detail")},
                                     {QStringLiteral("targetFaces"), 60000},
                                     {QStringLiteral("splitIntoBlocks"), true},
                                     {QStringLiteral("blockSizeMeters"), 256.0},
                                     {QStringLiteral("skipBoundaryBlocks"), true},
                                     {QStringLiteral("saveAfterEachStep"), true},
                                     {QStringLiteral("strictVolumetricMasks"), true}});
    dialog.setSourceCandidates(QJsonArray{depthMapsCandidate()});

    QSignalSpy runSpy(&dialog, &GenerateModelDialog::runRequested);
    auto* buttonBox = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(buttonBox, nullptr);
    buttonBox->button(QDialogButtonBox::Ok)->click();
    ASSERT_EQ(runSpy.count(), 1);

    const QJsonObject settings = runSpy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("modelGenerationContractRevision")).toInt(), 2);
    EXPECT_EQ(settings.value(QStringLiteral("source_data")).toString(), QStringLiteral("depth_maps"));
    EXPECT_EQ(settings.value(QStringLiteral("source_path")).toString(), QStringLiteral("E:/tmp/depth"));
    EXPECT_FALSE(settings.contains(QStringLiteral("threads")));
    EXPECT_EQ(settings.value(QStringLiteral("depthQualityProfile")).toString(), QStringLiteral("medium"));
    EXPECT_EQ(settings.value(QStringLiteral("surfaceQualityProfile")).toString(), QStringLiteral("recovered_ooc"));
    EXPECT_EQ(settings.value(QStringLiteral("reconstruction_mode")).toString(), QStringLiteral("recovered_ooc"));
    EXPECT_EQ(settings.value(QStringLiteral("faceCountMode")).toString(), QStringLiteral("high"));
    EXPECT_EQ(settings.value(QStringLiteral("faceCountCustom")).toInt(), 60000);
    EXPECT_EQ(settings.value(QStringLiteral("simplifyTargetFaces")).toInt(), 0);
    EXPECT_EQ(settings.value(QStringLiteral("requestedTargetFaces")).toInt(), 0);
    for (const QString& retired : {QStringLiteral("quality"),
                                   QStringLiteral("qualityProfile"),
                                   QStringLiteral("modelQualityProfile"),
                                   QStringLiteral("targetFaces"),
                                   QStringLiteral("splitIntoBlocks"),
                                   QStringLiteral("blockSizeMeters"),
                                   QStringLiteral("skipBoundaryBlocks"),
                                   QStringLiteral("saveAfterEachStep"),
                                   QStringLiteral("strictVolumetricMasks")})
    {
        EXPECT_FALSE(settings.contains(retired)) << qPrintable(retired);
    }

    auto* quality = dialog.findChild<QComboBox*>(QStringLiteral("modelQualityCombo"));
    ASSERT_NE(quality, nullptr);
    EXPECT_EQ(quality->currentData().toString(), QStringLiteral("medium"));
}

TEST(WorkflowParameterDialogStyleTest, GenerateModelFixesRecoveredPipelineParameters)
{
    GenerateModelDialog dialog;
    dialog.applySettings(QJsonObject{{QStringLiteral("source_data"), QStringLiteral("depth_maps")},
                                     {QStringLiteral("source_path"), QStringLiteral("E:/tmp/depth")},
                                     {QStringLiteral("interpolation"), QStringLiteral("disabled")},
                                     {QStringLiteral("depthFiltering"), QStringLiteral("aggressive")}});
    dialog.setSourceCandidates(QJsonArray{depthMapsCandidate()});

    QSignalSpy run_spy(&dialog, &GenerateModelDialog::runRequested);
    auto* button_box = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(button_box, nullptr);
    button_box->button(QDialogButtonBox::Ok)->click();
    ASSERT_EQ(run_spy.count(), 1);

    const QJsonObject settings = run_spy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("interpolation")).toString(), QStringLiteral("disabled"));
    EXPECT_FALSE(settings.contains(QStringLiteral("calculateVertexColors")));
    EXPECT_FALSE(settings.contains(QStringLiteral("surface_type")));
    EXPECT_EQ(settings.value(QStringLiteral("depthFiltering")).toString(), QStringLiteral("aggressive"));
}

TEST(WorkflowParameterDialogStyleTest, RecoveredModelShowsQualityAndInterpolationControls)
{
    GenerateModelDialog dialog;
    dialog.setSourceCandidates(QJsonArray{depthMapsCandidate()});
    auto* quality = dialog.findChild<QComboBox*>(QStringLiteral("modelQualityCombo"));
    auto* interpolation = dialog.findChild<QComboBox*>(QStringLiteral("modelInterpolationCombo"));
    ASSERT_NE(quality, nullptr);
    ASSERT_NE(interpolation, nullptr);
    EXPECT_EQ(quality->currentData().toString(), QStringLiteral("medium"));
    EXPECT_EQ(interpolation->currentData().toString(), QStringLiteral("enabled"));
}

TEST(WorkflowParameterDialogStyleTest, TextureMappingExplainsFixedOptionsAndKeepsStableSchema)
{
    TextureMappingDialog dialog;
    EXPECT_EQ(dialog.findChild<QComboBox*>(QStringLiteral("m_textureTypeCombo")), nullptr);
    EXPECT_EQ(dialog.findChild<QComboBox*>(QStringLiteral("m_sourceCombo")), nullptr);
    EXPECT_EQ(dialog.findChild<QComboBox*>(QStringLiteral("m_uvMethodCombo")), nullptr);
    EXPECT_EQ(dialog.findChild<QCheckBox*>(QStringLiteral("m_saveEachStepCheck")), nullptr);
    EXPECT_EQ(dialog.findChild<QCheckBox*>(QStringLiteral("m_useAssignedImagesCheck")), nullptr);
    EXPECT_EQ(dialog.findChild<QCheckBox*>(QStringLiteral("m_transferTextureCheck")), nullptr);

    auto* fixed_note = dialog.findChild<QLabel*>(QStringLiteral("fixedWorkflowLabel"));
    auto* unsupported_note = dialog.findChild<QLabel*>(QStringLiteral("unsupportedOptionsNote"));
    ASSERT_NE(fixed_note, nullptr);
    ASSERT_NE(unsupported_note, nullptr);
    EXPECT_TRUE(fixed_note->text().contains(QStringLiteral("有效相机与深度数据")));
    EXPECT_TRUE(unsupported_note->text().contains(QStringLiteral("不支持")));

    QSignalSpy settings_spy(&dialog, &TextureMappingDialog::settingsChanged);
    dialog.applySettings(QJsonObject{{QStringLiteral("saveEachStep"), true},
                                     {QStringLiteral("useAssignedImages"), true},
                                     {QStringLiteral("transferTexture"), true},
                                     {QStringLiteral("textureSize"), 4096}});
    EXPECT_EQ(settings_spy.count(), 0);

    QSignalSpy run_spy(&dialog, &TextureMappingDialog::runRequested);
    auto* button_box = dialog.findChild<QDialogButtonBox*>(QStringLiteral("m_buttonBox"));
    ASSERT_NE(button_box, nullptr);
    auto* run_button = button_box->button(QDialogButtonBox::Ok);
    ASSERT_NE(run_button, nullptr);
    run_button->click();
    ASSERT_EQ(run_spy.count(), 1);
    const QJsonObject settings = run_spy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("textureType")).toString(), QStringLiteral("texture_mapping"));
    EXPECT_EQ(settings.value(QStringLiteral("textureMappingSettingsRevision")).toInt(), 3);
    EXPECT_EQ(settings.value(QStringLiteral("sourceData")).toString(), QStringLiteral("images"));
    EXPECT_EQ(settings.value(QStringLiteral("mappingMode")).toString(), QStringLiteral("natural_mapping"));
    EXPECT_EQ(settings.value(QStringLiteral("holeFillMode")).toString(), QStringLiteral("neighbor_view_recovery"));
    EXPECT_EQ(settings.value(QStringLiteral("imageDownscale")).toInt(), 2);
    EXPECT_EQ(settings.value(QStringLiteral("antiAliasing")).toInt(), 1);
    EXPECT_FALSE(settings.value(QStringLiteral("colorCorrection")).toBool());
    EXPECT_DOUBLE_EQ(settings.value(QStringLiteral("sharpeningStrength")).toDouble(), 1.0);
    EXPECT_FALSE(settings.value(QStringLiteral("saveEachStep")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("useAssignedImages")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("transferTexture")).toBool());

    TextureMappingDialog legacy_dialog;
    auto* legacy_sharpening = legacy_dialog.findChild<QDoubleSpinBox*>(QStringLiteral("m_seamsMarginSpin"));
    ASSERT_NE(legacy_sharpening, nullptr);
    legacy_dialog.applySettings(QJsonObject{{QStringLiteral("sharpeningStrength"), 0.35}});
    EXPECT_DOUBLE_EQ(legacy_sharpening->value(), 1.0);
    legacy_dialog.applySettings(QJsonObject{{QStringLiteral("textureMappingSettingsRevision"), 2},
                                            {QStringLiteral("sharpeningStrength"), 0.35}});
    EXPECT_DOUBLE_EQ(legacy_sharpening->value(), 0.35);
}

TEST(WorkflowParameterDialogStyleTest, TextureMappingMigratesLegacyAlgorithmAndRoundTripsQuality)
{
    TextureMappingDialog dialog;
    EXPECT_EQ(dialog.findChild<QComboBox*>(QStringLiteral("m_blendCombo")), nullptr);
    auto* anti_aliasing = dialog.findChild<QComboBox*>(QStringLiteral("m_antiAliasingCombo"));
    ASSERT_NE(anti_aliasing, nullptr);
    auto* texture_size = dialog.findChild<QComboBox*>(QStringLiteral("m_texSizeCombo"));
    ASSERT_NE(texture_size, nullptr);
    EXPECT_EQ(texture_size->findText(QStringLiteral("16384")), -1);
    dialog.applySettings(QJsonObject{{QStringLiteral("blendMode"), QStringLiteral("weighted_average")},
                                     {QStringLiteral("padding"), 64},
                                     {QStringLiteral("keepUnmapped"), false},
                                     {QStringLiteral("antiAliasing"), 4},
                                     {QStringLiteral("imageDownscale"), 4},
                                     {QStringLiteral("holeFillMode"), QStringLiteral("disabled")},
                                     {QStringLiteral("outOfFocusFilter"), true}});
    QSignalSpy run_spy(&dialog, &TextureMappingDialog::runRequested);
    auto* buttons = dialog.findChild<QDialogButtonBox*>(QStringLiteral("m_buttonBox"));
    ASSERT_NE(buttons, nullptr);
    buttons->button(QDialogButtonBox::Ok)->click();
    ASSERT_EQ(run_spy.count(), 1);
    const QJsonObject settings = run_spy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("blendMode")).toString(), QStringLiteral("natural"));
    EXPECT_EQ(settings.value(QStringLiteral("pipeline")).toString(), QStringLiteral("recovered_natural_v1"));
    EXPECT_EQ(settings.value(QStringLiteral("padding")).toInt(), 2);
    EXPECT_EQ(settings.value(QStringLiteral("antiAliasing")).toInt(), 4);
    EXPECT_EQ(settings.value(QStringLiteral("imageDownscale")).toInt(), 4);
    EXPECT_TRUE(settings.value(QStringLiteral("keepUnmapped")).toBool());
    EXPECT_TRUE(settings.value(QStringLiteral("outOfFocusFilter")).toBool());
    EXPECT_FALSE(settings.value(QStringLiteral("holeFill")).toBool());
}

TEST(WorkflowParameterDialogStyleTest, TiePointAndAerialDialogsReuseStableControlsAndButtons)
{
    CreateTiePointsDialog create_dialog;
    ThinTiePointsDialog thin_dialog;
    CleanTiePointsDialog clean_dialog;
    AerialTriangulationDialog aerial_dialog;

    const QList<QPair<QDialog*, QString>> dialog_actions{{&create_dialog, QStringLiteral("创建")},
                                                         {&thin_dialog, QStringLiteral("稀释")},
                                                         {&clean_dialog, QStringLiteral("确定")},
                                                         {&aerial_dialog, QStringLiteral("开始")}};
    for (const auto& [dialog, expected_action] : dialog_actions)
    {
        auto* button_box = dialog->findChild<QDialogButtonBox*>();
        ASSERT_NE(button_box, nullptr);
        ASSERT_NE(button_box->button(QDialogButtonBox::Ok), nullptr);
        ASSERT_NE(button_box->button(QDialogButtonBox::Cancel), nullptr);
        EXPECT_EQ(button_box->button(QDialogButtonBox::Ok)->text(), expected_action);
        EXPECT_EQ(button_box->button(QDialogButtonBox::Cancel)->text(), QStringLiteral("取消"));
    }

    auto* accuracy_combo = create_dialog.findChild<QComboBox*>(QStringLiteral("m_accuracyCombo"));
    auto* generic_check = create_dialog.findChild<QCheckBox*>(QStringLiteral("m_genericPreselectionCheck"));
    ASSERT_NE(accuracy_combo, nullptr);
    ASSERT_NE(generic_check, nullptr);
    EXPECT_GE(accuracy_combo->minimumHeight(), 28);
    EXPECT_GE(accuracy_combo->minimumWidth(), 240);
    EXPECT_GE(generic_check->minimumHeight(), 24);
    EXPECT_EQ(create_dialog.findChild<QLabel*>(QStringLiteral("m_preselectionStatusLabel")), nullptr);

    EXPECT_EQ(clean_dialog.windowTitle(), QStringLiteral("清理连接点"));
    EXPECT_EQ(clean_dialog.windowModality(), Qt::NonModal);
    EXPECT_EQ(clean_dialog.windowFlags() & Qt::WindowType_Mask, Qt::Tool);
    EXPECT_NE(clean_dialog.findChild<QSlider*>(QStringLiteral("cleanTiePointsLevelSlider")), nullptr);
    EXPECT_NE(clean_dialog.findChild<QDoubleSpinBox*>(QStringLiteral("cleanTiePointsLevelSpin")), nullptr);
    EXPECT_NE(clean_dialog.findChild<QLabel*>(QStringLiteral("cleanTiePointsPreviewHintLabel")), nullptr);

    aerial_dialog.setImageCount(6);
    auto* status = aerial_dialog.findChild<QLabel*>(QStringLiteral("m_statusLabel"));
    ASSERT_NE(status, nullptr);
    EXPECT_FALSE(status->isHidden());
    EXPECT_TRUE(status->text().contains(QStringLiteral("6")));
}

TEST(CleanTiePointsDialogTest, SliderSpinBoxAndCandidateCountDriveLivePreview)
{
    CleanTiePointsDialog dialog;
    auto* criterion_combo = dialog.findChild<QComboBox*>(QStringLiteral("cleanTiePointsCriterionCombo"));
    auto* level_spin = dialog.findChild<QDoubleSpinBox*>(QStringLiteral("cleanTiePointsLevelSpin"));
    auto* level_slider = dialog.findChild<QSlider*>(QStringLiteral("cleanTiePointsLevelSlider"));
    auto* candidate_label = dialog.findChild<QLabel*>(QStringLiteral("cleanTiePointsCandidateCountLabel"));
    auto* button_box = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    auto* delete_button = dialog.findChild<QPushButton*>(QStringLiteral("cleanTiePointsDeleteButton"));
    ASSERT_NE(criterion_combo, nullptr);
    ASSERT_NE(level_spin, nullptr);
    ASSERT_NE(level_slider, nullptr);
    ASSERT_NE(candidate_label, nullptr);
    ASSERT_NE(button_box, nullptr);
    ASSERT_NE(delete_button, nullptr);

    CleanTiePointsDialog::CriterionConfiguration configuration;
    configuration.minimum = 0.0;
    configuration.maximum = 4.0;
    configuration.defaultLevel = 1.2;
    configuration.singleStep = 0.1;
    configuration.decimals = 2;
    dialog.setCriterionConfiguration(CleanTiePointsDialog::Criterion::ReprojectionError, configuration);

    QSignalSpy preview_spy(&dialog, &CleanTiePointsDialog::previewRequested);
    QSignalSpy cleared_spy(&dialog, &CleanTiePointsDialog::previewCleared);
    QSignalSpy stage_delete_spy(&dialog, &CleanTiePointsDialog::stageDeleteRequested);
    dialog.setCriterion(CleanTiePointsDialog::Criterion::ReprojectionError);
    EXPECT_EQ(dialog.criterion(), CleanTiePointsDialog::Criterion::ReprojectionError);
    EXPECT_DOUBLE_EQ(dialog.level(), 1.2);
    EXPECT_EQ(level_spin->decimals(), 2);
    EXPECT_DOUBLE_EQ(level_spin->minimum(), 0.0);
    EXPECT_DOUBLE_EQ(level_spin->maximum(), 4.0);
    EXPECT_EQ(level_slider->value(), 700);
    EXPECT_EQ(preview_spy.count(), 1);
    EXPECT_FALSE(delete_button->isEnabled());
    EXPECT_FALSE(button_box->button(QDialogButtonBox::Ok)->isEnabled());
    EXPECT_TRUE(candidate_label->text().contains(QStringLiteral("等待预览")));

    dialog.setCandidateCount(12, 100);
    EXPECT_EQ(dialog.candidateCount(), 12);
    EXPECT_EQ(dialog.totalPointCount(), 100);
    EXPECT_TRUE(delete_button->isEnabled());
    EXPECT_FALSE(button_box->button(QDialogButtonBox::Ok)->isEnabled());
    EXPECT_TRUE(candidate_label->text().contains(QStringLiteral("12 / 100")));

    delete_button->click();
    EXPECT_EQ(stage_delete_spy.count(), 1);
    EXPECT_FALSE(dialog.deleteRequested());
    dialog.confirmStagedDeletion(CleanTiePointsDialog::Criterion::ReprojectionError, 1.2, 12, 88);
    EXPECT_TRUE(dialog.deleteRequested());
    EXPECT_TRUE(dialog.hasStagedDeletion(CleanTiePointsDialog::Criterion::ReprojectionError));
    EXPECT_DOUBLE_EQ(dialog.stagedLevel(CleanTiePointsDialog::Criterion::ReprojectionError), 1.2);
    EXPECT_EQ(dialog.stagedDeletionCount(), 12);
    EXPECT_EQ(dialog.remainingPointCount(), 88);
    EXPECT_TRUE(button_box->button(QDialogButtonBox::Ok)->isEnabled());
    EXPECT_TRUE(candidate_label->text().contains(QStringLiteral("已暂删：12")));

    level_slider->setValue(750);
    EXPECT_DOUBLE_EQ(level_spin->value(), 1.0);
    EXPECT_DOUBLE_EQ(dialog.level(), 1.0);
    EXPECT_EQ(preview_spy.count(), 2);
    EXPECT_EQ(dialog.candidateCount(), -1);
    EXPECT_FALSE(delete_button->isEnabled());

    level_spin->setValue(2.0);
    EXPECT_EQ(level_slider->value(), 500);
    EXPECT_EQ(preview_spy.count(), 3);

    dialog.setCandidateCount(0, 100);
    EXPECT_FALSE(delete_button->isEnabled());
    dialog.setCandidateCount(100, 100);
    EXPECT_FALSE(delete_button->isEnabled());
    dialog.setCandidateCount(25, 100);
    EXPECT_TRUE(delete_button->isEnabled());

    dialog.reject();
    EXPECT_EQ(cleared_spy.count(), 1);
}

TEST(CleanTiePointsDialogTest, SliderRightAlwaysUsesMoreAggressiveThreshold)
{
    CleanTiePointsDialog dialog;
    auto *level_slider = dialog.findChild<QSlider*>(
        QStringLiteral("cleanTiePointsLevelSlider"));
    auto *level_spin = dialog.findChild<QDoubleSpinBox*>(
        QStringLiteral("cleanTiePointsLevelSpin"));
    auto *left_label = dialog.findChild<QLabel*>(
        QStringLiteral("cleanTiePointsMinimumLabel"));
    auto *right_label = dialog.findChild<QLabel*>(
        QStringLiteral("cleanTiePointsMaximumLabel"));
    ASSERT_NE(level_slider, nullptr);
    ASSERT_NE(level_spin, nullptr);
    ASSERT_NE(left_label, nullptr);
    ASSERT_NE(right_label, nullptr);

    CleanTiePointsDialog::CriterionConfiguration configuration;
    configuration.minimum = 0.0;
    configuration.maximum = 10.0;
    configuration.defaultLevel = 5.0;
    configuration.singleStep = 0.1;
    configuration.decimals = 1;

    for (const CleanTiePointsDialog::Criterion criterion : {
             CleanTiePointsDialog::Criterion::ReprojectionError,
             CleanTiePointsDialog::Criterion::ReconstructionUncertainty,
             CleanTiePointsDialog::Criterion::ImageCount,
             CleanTiePointsDialog::Criterion::ProjectionAccuracy,
             CleanTiePointsDialog::Criterion::MinimumTriangulationAngle})
    {
        dialog.setCriterionConfiguration(criterion, configuration);
        dialog.setCriterion(criterion);
        level_slider->setValue(100);
        const double left_level = level_spin->value();
        level_slider->setValue(900);
        const double right_level = level_spin->value();

        const bool rejects_above =
            criterion == CleanTiePointsDialog::Criterion::ReprojectionError
            || criterion == CleanTiePointsDialog::Criterion::ReconstructionUncertainty
            || criterion == CleanTiePointsDialog::Criterion::ProjectionAccuracy;
        if (rejects_above)
        {
            EXPECT_GT(left_level, right_level);
            EXPECT_EQ(left_label->text(), QStringLiteral("10.0"));
            EXPECT_EQ(right_label->text(), QStringLiteral("0.0"));
        }
        else
        {
            EXPECT_LT(left_level, right_level);
            EXPECT_EQ(left_label->text(), QStringLiteral("0.0"));
            EXPECT_EQ(right_label->text(), QStringLiteral("10.0"));
        }
    }
}

TEST(CleanTiePointsDialogTest, ExposesAllCriteriaAndRejectsMissingMetadata)
{
    CleanTiePointsDialog dialog;
    auto* criterion_combo = dialog.findChild<QComboBox*>(QStringLiteral("cleanTiePointsCriterionCombo"));
    auto* availability_label = dialog.findChild<QLabel*>(QStringLiteral("cleanTiePointsAvailabilityLabel"));
    auto* button_box = dialog.findChild<QDialogButtonBox*>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(criterion_combo, nullptr);
    ASSERT_NE(availability_label, nullptr);
    ASSERT_NE(button_box, nullptr);
    auto* model = qobject_cast<QStandardItemModel*>(criterion_combo->model());
    ASSERT_NE(model, nullptr);

    for (const CleanTiePointsDialog::Criterion criterion : {
             CleanTiePointsDialog::Criterion::ReprojectionError,
             CleanTiePointsDialog::Criterion::ReconstructionUncertainty,
             CleanTiePointsDialog::Criterion::ImageCount,
             CleanTiePointsDialog::Criterion::ProjectionAccuracy,
             CleanTiePointsDialog::Criterion::MinimumTriangulationAngle})
    {
        EXPECT_TRUE(dialog.criterionConfiguration(criterion).available);
        const int index = criterion_combo->findData(static_cast<int>(criterion));
        ASSERT_GE(index, 0);
        EXPECT_TRUE(model->item(index)->isEnabled());
    }

    CleanTiePointsDialog::CriterionConfiguration unavailable =
        dialog.criterionConfiguration(CleanTiePointsDialog::Criterion::ProjectionAccuracy);
    unavailable.available = false;
    unavailable.unavailableReason = QStringLiteral("当前成果缺少 projection_accuracy 字段");
    dialog.setCriterionConfiguration(
        CleanTiePointsDialog::Criterion::ProjectionAccuracy, unavailable);
    const int unavailable_index = criterion_combo->findData(
        static_cast<int>(CleanTiePointsDialog::Criterion::ProjectionAccuracy));
    ASSERT_GE(unavailable_index, 0);
    EXPECT_FALSE(model->item(unavailable_index)->isEnabled());
    EXPECT_TRUE(availability_label->text().contains(QStringLiteral("投影精度")));

    dialog.setCriterion(CleanTiePointsDialog::Criterion::ReprojectionError);
    dialog.setCriterion(CleanTiePointsDialog::Criterion::ProjectionAccuracy);
    EXPECT_EQ(dialog.criterion(), CleanTiePointsDialog::Criterion::ReprojectionError);

    criterion_combo->setCurrentIndex(unavailable_index);
    EXPECT_EQ(dialog.criterion(), CleanTiePointsDialog::Criterion::None);
    EXPECT_FALSE(button_box->button(QDialogButtonBox::Ok)->isEnabled());
}

TEST(CleanTiePointsDialogTest, MergesRepeatedThresholdsByRejectionDirection)
{
    CleanTiePointsDialog dialog;
    dialog.confirmStagedDeletion(
        CleanTiePointsDialog::Criterion::ReconstructionUncertainty, 10.0, 2, 98);
    dialog.confirmStagedDeletion(
        CleanTiePointsDialog::Criterion::ReconstructionUncertainty, 15.0, 3, 97);
    dialog.confirmStagedDeletion(
        CleanTiePointsDialog::Criterion::ProjectionAccuracy, 2.0, 4, 96);
    dialog.confirmStagedDeletion(
        CleanTiePointsDialog::Criterion::ProjectionAccuracy, 3.0, 5, 95);
    dialog.confirmStagedDeletion(
        CleanTiePointsDialog::Criterion::ImageCount, 3.0, 6, 94);
    dialog.confirmStagedDeletion(
        CleanTiePointsDialog::Criterion::ImageCount, 2.0, 7, 93);

    EXPECT_DOUBLE_EQ(dialog.stagedLevel(
                         CleanTiePointsDialog::Criterion::ReconstructionUncertainty),
                     10.0);
    EXPECT_DOUBLE_EQ(dialog.stagedLevel(
                         CleanTiePointsDialog::Criterion::ProjectionAccuracy),
                     2.0);
    EXPECT_DOUBLE_EQ(dialog.stagedLevel(CleanTiePointsDialog::Criterion::ImageCount),
                     3.0);
}

TEST(WorkflowParameterDialogStyleTest, StylesheetRulesAreScopedToWorkflowDialogs)
{
    QFile styleFile(QStringLiteral(PLASCAN_SOURCE_DIR "/resources/styles/app.qss"));
    ASSERT_TRUE(styleFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString style = QString::fromUtf8(styleFile.readAll());

    EXPECT_TRUE(style.contains(QStringLiteral("QDialog[workflowParameterDialog=\"true\"] QLabel")));
    EXPECT_TRUE(style.contains(
        QStringLiteral("QDialog[workflowParameterDialog=\"true\"] QScrollArea#workflowParameterScrollArea")));
    EXPECT_TRUE(
        style.contains(QStringLiteral("QDialog[workflowParameterDialog=\"true\"] QToolButton#workflowAdvancedToggle")));
    EXPECT_TRUE(style.contains(QStringLiteral("QDialog[workflowParameterDialog=\"true\"] QLabel#workflowStatusLabel")));
    EXPECT_TRUE(style.contains(QStringLiteral("QDialog#BuildModelDialog")));
    EXPECT_TRUE(style.contains(QStringLiteral("QToolButton[modelSectionHeader=\"true\"]")));
    EXPECT_TRUE(style.contains(QStringLiteral("QFrame[modelSectionBody=\"true\"]")));
    EXPECT_TRUE(style.contains(QStringLiteral("QDialog#BuildModelPreviewDialog")));
}

int main(int argc, char** argv)
{
    const bool lists_tests =
        std::any_of(argv,
                    argv + argc,
                    [](const char* argument) { return argument && std::strcmp(argument, "--gtest_list_tests") == 0; });
    testing::InitGoogleTest(&argc, argv);
    if (lists_tests)
    {
        return RUN_ALL_TESTS();
    }

    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);
    return RUN_ALL_TESTS();
}
