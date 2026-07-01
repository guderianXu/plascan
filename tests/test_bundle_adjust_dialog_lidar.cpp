#include "BundleAdjustDialog.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QJsonObject>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>

namespace
{

template <typename WidgetT>
WidgetT *findRequiredChild(BundleAdjustDialog &dialog, const QString &objectName)
{
    auto *widget = dialog.findChild<WidgetT *>(objectName);
    EXPECT_NE(widget, nullptr) << objectName.toStdString();
    return widget;
}

QString readProjectSourceFile(const QString &relativePath)
{
    const QString absolutePath = QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(relativePath);
    QFile file(absolutePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        ADD_FAILURE() << "Failed to open " << absolutePath.toStdString();
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

QJsonObject lastSettings(const QSignalSpy &spy)
{
    EXPECT_GT(spy.count(), 0);
    if (spy.isEmpty())
    {
        return {};
    }
    return spy.last().at(0).toJsonObject();
}

} // namespace

TEST(BundleAdjustDialogLidarTest, EmitsLaserConstraintOptionsWhenRun)
{
    BundleAdjustDialog dialog;
    dialog.setAvailableImages({QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")});
    dialog.setDefaultOutputDir(QStringLiteral("E:/code/plascan/testData/ba_lidar_smoke"));

    auto *enableCheck = findRequiredChild<QCheckBox>(dialog, QStringLiteral("m_enableLaserConstraintsCheck"));
    auto *cloudEdit = findRequiredChild<QLineEdit>(dialog, QStringLiteral("m_laserConstraintCloudEdit"));
    auto *maxDistanceSpin = findRequiredChild<QDoubleSpinBox>(
        dialog, QStringLiteral("m_laserAssociationMaxDistanceSpin"));
    auto *voxelSizeSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserVoxelSizeSpin"));
    auto *maxCurvatureSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserMaxCurvatureSpin"));
    auto *maxSamplesSpin = findRequiredChild<QSpinBox>(dialog, QStringLiteral("m_laserMaxSamplesSpin"));
    auto *missingNormalsCheck = findRequiredChild<QCheckBox>(
        dialog, QStringLiteral("m_laserMissingNormalsAsHeightPlanesCheck"));
    auto *weightSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserWeightSpin"));
    auto *laserHuberSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserHuberDeltaSpin"));
    auto *runButton = dialog.findChild<QPushButton *>(QStringLiteral("runBtn"));
    ASSERT_NE(runButton, nullptr);

    ASSERT_NE(enableCheck, nullptr);
    ASSERT_NE(cloudEdit, nullptr);
    ASSERT_NE(maxDistanceSpin, nullptr);
    ASSERT_NE(voxelSizeSpin, nullptr);
    ASSERT_NE(maxCurvatureSpin, nullptr);
    ASSERT_NE(maxSamplesSpin, nullptr);
    ASSERT_NE(missingNormalsCheck, nullptr);
    ASSERT_NE(weightSpin, nullptr);
    ASSERT_NE(laserHuberSpin, nullptr);

    enableCheck->setChecked(true);
    cloudEdit->setText(QStringLiteral("E:/code/plascan/testData/MUN-FRL/cloud_registered.ply"));
    maxDistanceSpin->setValue(1.25);
    voxelSizeSpin->setValue(0.08);
    maxCurvatureSpin->setValue(0.12);
    maxSamplesSpin->setValue(12345);
    missingNormalsCheck->setChecked(true);
    weightSpin->setValue(4.5);
    laserHuberSpin->setValue(0.35);

    QSignalSpy runSpy(&dialog, &BundleAdjustDialog::requestRunBundleAdjust);
    ASSERT_TRUE(runSpy.isValid());

    runButton->click();

    ASSERT_EQ(runSpy.count(), 1);
    const QList<QVariant> args = runSpy.takeFirst();
    ASSERT_GE(args.size(), 5);
    const QJsonObject options = args.at(4).toJsonObject();

    EXPECT_TRUE(options.value(QStringLiteral("enable_laser_constraints")).toBool());
    EXPECT_EQ(options.value(QStringLiteral("laser_constraint_cloud_path")).toString(),
              QStringLiteral("E:/code/plascan/testData/MUN-FRL/cloud_registered.ply"));
    EXPECT_DOUBLE_EQ(options.value(QStringLiteral("laser_association_max_distance_m")).toDouble(), 1.25);
    EXPECT_DOUBLE_EQ(options.value(QStringLiteral("laser_voxel_size_m")).toDouble(), 0.08);
    EXPECT_DOUBLE_EQ(options.value(QStringLiteral("laser_max_curvature")).toDouble(), 0.12);
    EXPECT_EQ(options.value(QStringLiteral("laser_max_samples")).toInt(), 12345);
    EXPECT_TRUE(options.value(QStringLiteral("laser_missing_normals_as_height_planes")).toBool());
    EXPECT_DOUBLE_EQ(options.value(QStringLiteral("laser_weight")).toDouble(), 4.5);
    EXPECT_DOUBLE_EQ(options.value(QStringLiteral("laser_huber_delta_m")).toDouble(), 0.35);
}

TEST(BundleAdjustDialogLidarTest, PersistsLaserConstraintSettings)
{
    BundleAdjustDialog dialog;

    auto *enableCheck = findRequiredChild<QCheckBox>(dialog, QStringLiteral("m_enableLaserConstraintsCheck"));
    auto *cloudEdit = findRequiredChild<QLineEdit>(dialog, QStringLiteral("m_laserConstraintCloudEdit"));
    auto *maxDistanceSpin = findRequiredChild<QDoubleSpinBox>(
        dialog, QStringLiteral("m_laserAssociationMaxDistanceSpin"));
    auto *voxelSizeSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserVoxelSizeSpin"));
    auto *maxCurvatureSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserMaxCurvatureSpin"));
    auto *maxSamplesSpin = findRequiredChild<QSpinBox>(dialog, QStringLiteral("m_laserMaxSamplesSpin"));
    auto *missingNormalsCheck = findRequiredChild<QCheckBox>(
        dialog, QStringLiteral("m_laserMissingNormalsAsHeightPlanesCheck"));
    auto *weightSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserWeightSpin"));
    auto *laserHuberSpin = findRequiredChild<QDoubleSpinBox>(dialog, QStringLiteral("m_laserHuberDeltaSpin"));

    ASSERT_NE(enableCheck, nullptr);
    ASSERT_NE(cloudEdit, nullptr);
    ASSERT_NE(maxDistanceSpin, nullptr);
    ASSERT_NE(voxelSizeSpin, nullptr);
    ASSERT_NE(maxCurvatureSpin, nullptr);
    ASSERT_NE(maxSamplesSpin, nullptr);
    ASSERT_NE(missingNormalsCheck, nullptr);
    ASSERT_NE(weightSpin, nullptr);
    ASSERT_NE(laserHuberSpin, nullptr);

    QJsonObject settings;
    settings[QStringLiteral("enable_laser_constraints")] = true;
    settings[QStringLiteral("laser_constraint_cloud_path")] = QStringLiteral("E:/data/cloud_registered.ply");
    settings[QStringLiteral("laser_association_max_distance_m")] = 2.0;
    settings[QStringLiteral("laser_voxel_size_m")] = 0.15;
    settings[QStringLiteral("laser_max_curvature")] = 0.18;
    settings[QStringLiteral("laser_max_samples")] = 25000;
    settings[QStringLiteral("laser_missing_normals_as_height_planes")] = true;
    settings[QStringLiteral("laser_weight")] = 3.25;
    settings[QStringLiteral("laser_huber_delta_m")] = 0.45;
    dialog.applySettings(settings);

    EXPECT_TRUE(enableCheck->isChecked());
    EXPECT_EQ(cloudEdit->text(), QStringLiteral("E:/data/cloud_registered.ply"));
    EXPECT_DOUBLE_EQ(maxDistanceSpin->value(), 2.0);
    EXPECT_DOUBLE_EQ(voxelSizeSpin->value(), 0.15);
    EXPECT_DOUBLE_EQ(maxCurvatureSpin->value(), 0.18);
    EXPECT_EQ(maxSamplesSpin->value(), 25000);
    EXPECT_TRUE(missingNormalsCheck->isChecked());
    EXPECT_DOUBLE_EQ(weightSpin->value(), 3.25);
    EXPECT_DOUBLE_EQ(laserHuberSpin->value(), 0.45);

    QSignalSpy settingsSpy(&dialog, &BundleAdjustDialog::settingsChanged);
    ASSERT_TRUE(settingsSpy.isValid());
    weightSpin->setValue(6.75);

    const QJsonObject emitted = lastSettings(settingsSpy);
    EXPECT_TRUE(emitted.value(QStringLiteral("enable_laser_constraints")).toBool());
    EXPECT_EQ(emitted.value(QStringLiteral("laser_constraint_cloud_path")).toString(),
              QStringLiteral("E:/data/cloud_registered.ply"));
    EXPECT_DOUBLE_EQ(emitted.value(QStringLiteral("laser_association_max_distance_m")).toDouble(), 2.0);
    EXPECT_DOUBLE_EQ(emitted.value(QStringLiteral("laser_voxel_size_m")).toDouble(), 0.15);
    EXPECT_DOUBLE_EQ(emitted.value(QStringLiteral("laser_max_curvature")).toDouble(), 0.18);
    EXPECT_EQ(emitted.value(QStringLiteral("laser_max_samples")).toInt(), 25000);
    EXPECT_TRUE(emitted.value(QStringLiteral("laser_missing_normals_as_height_planes")).toBool());
    EXPECT_DOUBLE_EQ(emitted.value(QStringLiteral("laser_weight")).toDouble(), 6.75);
    EXPECT_DOUBLE_EQ(emitted.value(QStringLiteral("laser_huber_delta_m")).toDouble(), 0.45);
}

TEST(BundleAdjustDialogPerformanceTest, SetAvailableImagesDoesNotPersistDuringInitialization)
{
    BundleAdjustDialog dialog;
    QSignalSpy settingsSpy(&dialog, &BundleAdjustDialog::settingsChanged);
    ASSERT_TRUE(settingsSpy.isValid());

    QStringList images;
    for (int i = 0; i < 500; ++i)
    {
        images.append(QStringLiteral("image_%1.jpg").arg(i, 4, 10, QLatin1Char('0')));
    }

    dialog.setAvailableImages(images);

    EXPECT_EQ(settingsSpy.count(), 0)
        << "Bulk image list initialization must not synchronously persist settings for every item.";
}

TEST(ProjectManagerBundleAdjustLidarTest, MapsLaserSettingsToBundleAdjustServiceOptions)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));

    EXPECT_TRUE(source.contains(QStringLiteral("opts.enableLaserConstraints")));
    EXPECT_TRUE(source.contains(QStringLiteral("enable_laser_constraints")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserConstraintCloudPath")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_constraint_cloud_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserAssociationMaxDistanceMeters")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_association_max_distance_m")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserVoxelSizeMeters")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_voxel_size_m")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserMaxCurvature")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_max_curvature")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserMaxSamples")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_max_samples")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserUseMissingNormalsAsHeightPlanes")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_missing_normals_as_height_planes")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserWeight")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_weight")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.laserHuberDeltaMeters")));
    EXPECT_TRUE(source.contains(QStringLiteral("laser_huber_delta_m")));
}

int main(int argc, char **argv)
{
    qRegisterMetaType<QStringList>("QStringList");
    qRegisterMetaType<QJsonObject>("QJsonObject");
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
