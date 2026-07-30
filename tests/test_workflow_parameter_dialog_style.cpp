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
#include <QToolButton>

#include "reconstruction/GenerateModelDialog.h"

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
    EXPECT_EQ(settings.value(QStringLiteral("targetFaces")).toInt(), 60000);
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
