#include "reconstruction/MapProjectDialog.h"

#include "DemDomIO.h"
#include "DemDomTypes.h"
#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileInfo>
#include <QGroupBox>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTemporaryDir>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstring>

namespace
{

template<typename Widget>
Widget *requiredWidget(MapProjectDialog *dialog, const char *objectName)
{
    Widget *widget = dialog->findChild<Widget *>(QString::fromLatin1(objectName));
    EXPECT_NE(widget, nullptr) << objectName;
    return widget;
}

struct DialogFixture
{
    QTemporaryDir directory;
    QString demPath;
    QString firstImage;
    QString secondImage;
    QString pointCloudPath;

    bool createFiles()
    {
        if (!directory.isValid())
        {
            return false;
        }

        xjw::DemGridData dem;
        dem.width = 8;
        dem.height = 4;
        dem.minX = 100.0;
        dem.minY = 200.0;
        dem.stepX = 2.0;
        dem.stepY = 3.0;
        dem.elevation =
            cv::Mat(dem.height, dem.width, CV_32FC1, cv::Scalar(42.0f));
        dem.validMask =
            cv::Mat(dem.height, dem.width, CV_8UC1, cv::Scalar(255));

        demPath = QDir(directory.path()).filePath(QStringLiteral("input_dem.tif"));
        QString error;
        if (!xjw::DemDomIO::writeDemRaster(
                dem, demPath, xjw::DemRasterFormat::Float32Tiff, &error))
        {
            ADD_FAILURE() << error.toStdString();
            return false;
        }

        firstImage = QDir(directory.path()).filePath(QStringLiteral("ready.png"));
        secondImage =
            QDir(directory.path()).filePath(QStringLiteral("not_ready.png"));
        pointCloudPath =
            QDir(directory.path()).filePath(QStringLiteral("colored_cloud.ply"));
        QFile pointCloudFile(pointCloudPath);
        if (!pointCloudFile.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            return false;
        }
        pointCloudFile.write(
            "ply\nformat ascii 1.0\nelement vertex 4\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
            "10 0 0 255 0 0\n0 10 0 0 255 0\n"
            "-10 0 0 0 0 255\n0 -10 0 255 255 0\n");
        pointCloudFile.close();
        return xjw::common::io::writeImage(
                   firstImage,
                   cv::Mat(16, 16, CV_8UC3, cv::Scalar(10, 20, 30)))
            && xjw::common::io::writeImage(
                   secondImage,
                   cv::Mat(16, 16, CV_8UC3, cv::Scalar(30, 20, 10)));
    }

    void configure(MapProjectDialog *dialog,
                   const QStringList &readyImages) const
    {
        dialog->setAvailableImages({firstImage, secondImage});
        dialog->setProjectRoot(directory.path());
        dialog->setDefaultDemPath(demPath);
        dialog->setDefaultPointCloudPath(pointCloudPath);
        dialog->setImageReadiness(readyImages, 1);
        ASSERT_TRUE(QMetaObject::invokeMethod(
            dialog, "estimateNow", Qt::DirectConnection));
    }
};

TEST(MapProjectDialogTest, ExposesSupportedWorkflowAndActualDemDefaults)
{
    DialogFixture fixture;
    ASSERT_TRUE(fixture.createFiles());

    MapProjectDialog dialog;
    fixture.configure(&dialog, {fixture.firstImage});

    auto *demProjection = requiredWidget<QRadioButton>(
        &dialog, "orthoProjectionDemGridRadio");
    auto *planarProjection = requiredWidget<QRadioButton>(
        &dialog, "orthoProjectionPlanarRadio");
    auto *cylindricalProjection = requiredWidget<QRadioButton>(
        &dialog, "orthoProjectionCylindricalRadio");
    auto *colorSource =
        requiredWidget<QLabel>(&dialog, "orthoColorSourceLabel");
    auto *fillHoles =
        requiredWidget<QCheckBox>(&dialog, "orthoFillHolesCheck");
    auto *useMasks =
        requiredWidget<QCheckBox>(&dialog, "orthoUseProjectMasksCheck");
    auto *imageList =
        requiredWidget<QListWidget>(&dialog, "orthoImageList");
    auto *pixelSizeX =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoPixelSizeXSpin");
    auto *pixelSizeY =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoPixelSizeYSpin");
    auto *output =
        requiredWidget<QLineEdit>(&dialog, "orthoOutputPathEdit");
    auto *totalSize =
        requiredWidget<QLabel>(&dialog, "orthoTotalSizeLabel");
    auto *runButton =
        requiredWidget<QPushButton>(&dialog, "orthoRunButton");

    ASSERT_NE(demProjection, nullptr);
    ASSERT_NE(planarProjection, nullptr);
    ASSERT_NE(cylindricalProjection, nullptr);
    ASSERT_NE(colorSource, nullptr);
    ASSERT_NE(fillHoles, nullptr);
    ASSERT_NE(useMasks, nullptr);
    ASSERT_NE(imageList, nullptr);
    ASSERT_NE(pixelSizeX, nullptr);
    ASSERT_NE(pixelSizeY, nullptr);
    ASSERT_NE(output, nullptr);
    ASSERT_NE(totalSize, nullptr);
    ASSERT_NE(runButton, nullptr);

    EXPECT_TRUE(demProjection->isChecked());
    EXPECT_FALSE(planarProjection->isEnabled());
    EXPECT_FALSE(cylindricalProjection->isEnabled());
    EXPECT_TRUE(colorSource->text().contains(QStringLiteral("影像")));
    EXPECT_EQ(dialog.findChild<QCheckBox *>(
                  QStringLiteral("orthoRefineSeamsCheck")), nullptr);
    EXPECT_TRUE(fillHoles->isChecked());
    EXPECT_TRUE(useMasks->isEnabled());
    EXPECT_FALSE(useMasks->isChecked());

    ASSERT_EQ(imageList->count(), 2);
    EXPECT_EQ(imageList->item(0)->checkState(), Qt::Checked);
    EXPECT_TRUE(imageList->item(0)->flags().testFlag(Qt::ItemIsEnabled));
    EXPECT_EQ(imageList->item(1)->checkState(), Qt::Unchecked);
    EXPECT_FALSE(imageList->item(1)->flags().testFlag(Qt::ItemIsEnabled));

    EXPECT_DOUBLE_EQ(pixelSizeX->value(), 2.0);
    EXPECT_DOUBLE_EQ(pixelSizeY->value(), 3.0);
    EXPECT_TRUE(totalSize->text().contains(QStringLiteral("8 × 4")));
    EXPECT_EQ(
        QDir::cleanPath(output->text()),
        QDir(fixture.directory.path())
            .filePath(QStringLiteral("assets/ortho/relative_dom.tif")));
    EXPECT_EQ(runButton->text(), QStringLiteral("生成"));
}

TEST(MapProjectDialogTest, SwitchesToPointCloudPlanarAndGlobalProjection)
{
    DialogFixture fixture;
    ASSERT_TRUE(fixture.createFiles());

    MapProjectDialog dialog;
    fixture.configure(&dialog, {fixture.firstImage});
    auto *surface = requiredWidget<QComboBox>(&dialog, "orthoSurfaceCombo");
    auto *demProjection = requiredWidget<QRadioButton>(
        &dialog, "orthoProjectionDemGridRadio");
    auto *planarProjection = requiredWidget<QRadioButton>(
        &dialog, "orthoProjectionPlanarRadio");
    auto *cylindricalProjection = requiredWidget<QRadioButton>(
        &dialog, "orthoProjectionCylindricalRadio");
    auto *surfacePath = requiredWidget<QLineEdit>(&dialog, "orthoDemPathEdit");
    auto *colorSource = requiredWidget<QLabel>(&dialog, "orthoColorSourceLabel");
    auto *bodyReference = requiredWidget<QWidget>(&dialog, "orthoBodyReferenceAutoCheck");
    auto *maximumMode = requiredWidget<QRadioButton>(
        &dialog, "orthoMaximumDimensionModeRadio");
    ASSERT_NE(surface, nullptr);
    ASSERT_NE(colorSource, nullptr);
    surface->setCurrentIndex(surface->findData(QStringLiteral("point_cloud")));
    EXPECT_FALSE(demProjection->isEnabled());
    EXPECT_TRUE(planarProjection->isEnabled());
    EXPECT_TRUE(planarProjection->isChecked());
    EXPECT_TRUE(cylindricalProjection->isEnabled());
    EXPECT_EQ(surfacePath->text(), fixture.pointCloudPath);
    EXPECT_TRUE(colorSource->text().contains(QStringLiteral("点颜色 RGB")));

    cylindricalProjection->setChecked(true);
    maximumMode->setChecked(true);
    EXPECT_TRUE(bodyReference->isVisibleTo(&dialog));

    QJsonObject emittedSettings;
    QSignalSpy runSpy(&dialog, &MapProjectDialog::requestRunMapProject);
    QObject::connect(
        &dialog,
        &MapProjectDialog::requestRunMapProject,
        &dialog,
        [&emittedSettings](const QJsonObject &settings)
        {
            emittedSettings = settings;
        });
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "onRun", Qt::DirectConnection));
    if (emittedSettings.isEmpty())
    {
        ASSERT_TRUE(runSpy.wait(5000));
    }
    ASSERT_FALSE(emittedSettings.isEmpty());
    EXPECT_EQ(emittedSettings.value(QStringLiteral("projection_type")).toString(),
              QStringLiteral("cylindrical"));
    EXPECT_EQ(emittedSettings.value(QStringLiteral("surface_type")).toString(),
              QStringLiteral("point_cloud"));
    EXPECT_EQ(emittedSettings.value(QStringLiteral("color_source")).toString(),
              QStringLiteral("point_colors"));
    EXPECT_EQ(emittedSettings.value(QStringLiteral("point_cloud_path")).toString(),
              fixture.pointCloudPath);
}

TEST(MapProjectDialogTest, EmitsCompleteSettingsForCustomRegion)
{
    DialogFixture fixture;
    ASSERT_TRUE(fixture.createFiles());

    MapProjectDialog dialog;
    fixture.configure(&dialog, {fixture.firstImage, fixture.secondImage});

    auto *blend =
        requiredWidget<QComboBox>(&dialog, "orthoBlendModeCombo");
    auto *maximumMode = requiredWidget<QRadioButton>(
        &dialog, "orthoMaximumDimensionModeRadio");
    auto *maximumDimension =
        requiredWidget<QSpinBox>(&dialog, "orthoMaximumDimensionSpin");
    auto *boundsEnabled =
        requiredWidget<QCheckBox>(&dialog, "orthoBoundsEnabledCheck");
    auto *minX =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoMinXSpin");
    auto *maxX =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoMaxXSpin");
    auto *minY =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoMinYSpin");
    auto *maxY =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoMaxYSpin");
    auto *ghostFilter =
        requiredWidget<QCheckBox>(&dialog, "orthoGhostFilterCheck");
    auto *useMasks =
        requiredWidget<QCheckBox>(&dialog, "orthoUseProjectMasksCheck");
    auto *output =
        requiredWidget<QLineEdit>(&dialog, "orthoOutputPathEdit");

    ASSERT_NE(blend, nullptr);
    ASSERT_NE(maximumMode, nullptr);
    ASSERT_NE(maximumDimension, nullptr);
    ASSERT_NE(boundsEnabled, nullptr);
    ASSERT_NE(minX, nullptr);
    ASSERT_NE(maxX, nullptr);
    ASSERT_NE(minY, nullptr);
    ASSERT_NE(maxY, nullptr);
    ASSERT_NE(ghostFilter, nullptr);
    ASSERT_NE(useMasks, nullptr);
    ASSERT_NE(output, nullptr);

    blend->setCurrentIndex(
        blend->findData(QStringLiteral("weighted_average")));
    maximumMode->setChecked(true);
    maximumDimension->setValue(3);
    boundsEnabled->setChecked(true);
    minX->setValue(99.0);
    maxX->setValue(107.0);
    minY->setValue(198.5);
    maxY->setValue(204.5);
    ghostFilter->setChecked(true);
    useMasks->setChecked(true);
    const QString outputPath =
        QDir(fixture.directory.path()).filePath(QStringLiteral("custom.tif"));
    output->setText(outputPath);

    QJsonObject emittedSettings;
    QObject::connect(
        &dialog,
        &MapProjectDialog::requestRunMapProject,
        &dialog,
        [&emittedSettings](const QJsonObject &settings)
        {
            emittedSettings = settings;
        });

    ASSERT_TRUE(QMetaObject::invokeMethod(
        &dialog, "onRun", Qt::DirectConnection));
    ASSERT_FALSE(emittedSettings.isEmpty());
    EXPECT_EQ(
        emittedSettings.value(QStringLiteral("projection_type")).toString(),
        QStringLiteral("dem_grid"));
    EXPECT_EQ(
        emittedSettings.value(QStringLiteral("blend_mode")).toString(),
        QStringLiteral("weighted_average"));
    EXPECT_EQ(
        emittedSettings.value(QStringLiteral("sizing_mode")).toString(),
        QStringLiteral("maximum_dimension"));
    EXPECT_EQ(
        emittedSettings.value(QStringLiteral("resolution_mode")).toString(),
        QStringLiteral("maximum_dimension"));
    EXPECT_EQ(
        emittedSettings.value(QStringLiteral("maximum_dimension")).toInt(),
        3);
    EXPECT_TRUE(
        emittedSettings.value(QStringLiteral("bounds_enabled")).toBool());
    EXPECT_DOUBLE_EQ(
        emittedSettings.value(QStringLiteral("min_x")).toDouble(),
        99.0);
    EXPECT_DOUBLE_EQ(
        emittedSettings.value(QStringLiteral("max_y")).toDouble(),
        204.5);
    EXPECT_TRUE(
        emittedSettings.value(QStringLiteral("ghost_filter")).toBool());
    EXPECT_TRUE(
        emittedSettings.value(QStringLiteral("use_project_masks")).toBool());
    EXPECT_EQ(
        emittedSettings.value(QStringLiteral("images")).toArray().size(),
        2);
    EXPECT_EQ(
        emittedSettings.value(QStringLiteral("output_path")).toString(),
        outputPath);
    EXPECT_TRUE(
        emittedSettings.value(QStringLiteral("pixel_size_auto")).toBool(false));
    EXPECT_FALSE(
        emittedSettings.value(QStringLiteral("bounds_auto")).toBool(true));
}

TEST(MapProjectDialogTest, ReportsProgressAndRequestsSafeCancellation)
{
    DialogFixture fixture;
    ASSERT_TRUE(fixture.createFiles());

    MapProjectDialog dialog;
    fixture.configure(&dialog, {fixture.firstImage});
    QSignalSpy cancelSpy(&dialog, &MapProjectDialog::requestCancelMapProject);

    dialog.onPipelineStarted();
    dialog.onPipelineProgress(QStringLiteral("正射投影"), 47);

    auto *parameterGroup =
        requiredWidget<QGroupBox>(&dialog, "orthoParametersGroup");
    auto *progress =
        requiredWidget<QProgressBar>(&dialog, "orthoProgressBar");
    auto *cancelButton =
        requiredWidget<QPushButton>(&dialog, "orthoCancelButton");
    auto *runButton =
        requiredWidget<QPushButton>(&dialog, "orthoRunButton");
    ASSERT_NE(parameterGroup, nullptr);
    ASSERT_NE(progress, nullptr);
    ASSERT_NE(cancelButton, nullptr);
    ASSERT_NE(runButton, nullptr);

    EXPECT_FALSE(parameterGroup->isEnabled());
    EXPECT_EQ(progress->value(), 47);
    EXPECT_FALSE(runButton->isEnabled());
    cancelButton->click();
    EXPECT_EQ(cancelSpy.count(), 1);
    EXPECT_FALSE(cancelButton->isEnabled());
    EXPECT_TRUE(cancelButton->text().contains(QStringLiteral("取消")));

    dialog.onPipelineFinished(false, QStringLiteral("正射影像生成已取消"));
    EXPECT_TRUE(parameterGroup->isEnabled());
    EXPECT_TRUE(runButton->isEnabled());
    EXPECT_EQ(runButton->text(), QStringLiteral("重试"));
}

TEST(MapProjectDialogTest, KeepsCurrentDemWhenSavedPathNoLongerExists)
{
    DialogFixture fixture;
    ASSERT_TRUE(fixture.createFiles());

    MapProjectDialog dialog;
    fixture.configure(&dialog, {fixture.firstImage});
    QJsonObject settings;
    settings[QStringLiteral("dem_path")] =
        QDir(fixture.directory.path()).filePath(QStringLiteral("missing.tif"));
    settings[QStringLiteral("output_path")] =
        QDir(fixture.directory.path()).filePath(QStringLiteral("saved.tif"));
    settings[QStringLiteral("pixel_size_x")] = 77.0;
    settings[QStringLiteral("pixel_size_y")] = 88.0;
    settings[QStringLiteral("pixel_size_auto")] = true;
    settings[QStringLiteral("bounds_enabled")] = false;
    settings[QStringLiteral("bounds_auto")] = true;
    settings[QStringLiteral("min_x")] = -1000.0;
    settings[QStringLiteral("max_x")] = 1000.0;
    settings[QStringLiteral("min_y")] = -2000.0;
    settings[QStringLiteral("max_y")] = 2000.0;
    dialog.applySettings(settings);
    ASSERT_TRUE(QMetaObject::invokeMethod(
        &dialog, "estimateNow", Qt::DirectConnection));

    auto *demPath =
        requiredWidget<QLineEdit>(&dialog, "orthoDemPathEdit");
    auto *pixelSizeX =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoPixelSizeXSpin");
    auto *pixelSizeY =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoPixelSizeYSpin");
    auto *minX =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoMinXSpin");
    auto *maxY =
        requiredWidget<QDoubleSpinBox>(&dialog, "orthoMaxYSpin");
    ASSERT_NE(demPath, nullptr);
    ASSERT_NE(pixelSizeX, nullptr);
    ASSERT_NE(pixelSizeY, nullptr);
    ASSERT_NE(minX, nullptr);
    ASSERT_NE(maxY, nullptr);
    EXPECT_EQ(demPath->text(), fixture.demPath);
    EXPECT_DOUBLE_EQ(pixelSizeX->value(), 2.0);
    EXPECT_DOUBLE_EQ(pixelSizeY->value(), 3.0);
    EXPECT_DOUBLE_EQ(minX->value(), 99.0);
    EXPECT_DOUBLE_EQ(maxY->value(), 210.5);
}

} // namespace

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

    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    QApplication application(argc, argv);
    return RUN_ALL_TESTS();
}
