// ============================================================
// test_gui_project_utils.cpp — GUI project 层公共工具与三角化服务测试
// ============================================================

#include <gtest/gtest.h>

#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "ProjectTriangulationService.h"
#include "FeatureExtractionDialog.h"
#include "FeatureMatchingDialog.h"
#include "InitCameraPoseDialog.h"
#include "ThreeDReconstructionDialog.h"
#include "MapProjectDialog.h"
#include "BundleAdjustDialog.h"
#include "ModelDropSupport.h"
#include "DataTreeWidget.h"
#include "MainMenu.h"
#include "TaskStatusWidget.h"

#include "Camera.h"

#include <plapoint/io/ply_io.h>

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QGroupBox>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMetaObject>
#include <QFileInfo>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTemporaryDir>
#include <QTextStream>
#include <QToolButton>
#include <QTreeView>
#include <QUrl>
#include <QStandardItemModel>

#include <array>
#include <vector>

namespace {

xjw::Camera makeCamera(double cx, double cy, double cz,
                       double fu = 1200.0, double fv = 1200.0)
{
    xjw::Camera cam;
    cam.setIntrinsics(fu, fv, 512.0, 384.0);
    const std::array<double, 9> rotation = {1.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0,
                                            0.0, 0.0, 1.0};
    const std::array<double, 3> center = {cx, cy, cz};
    cam.setPose(rotation, center);
    return cam;
}

bool projectPoint(const xjw::Camera &camera,
                  const std::array<double, 3> &xyz,
                  double *u,
                  double *v)
{
    if (!u || !v)
    {
        return false;
    }

    const double worldPoint[3] = {xyz[0], xyz[1], xyz[2]};
    double uv[2] = {0.0, 0.0};
    if (!camera.projectWorldPoint(worldPoint, uv))
    {
        return false;
    }

    *u = uv[0];
    *v = uv[1];
    return true;
}

QJsonObject buildImageEntry(const QString &path, const xjw::Camera &camera)
{
    QJsonObject imageObject;
    imageObject[QStringLiteral("path")] = path;
    imageObject[QStringLiteral("camera")] = xjw::gui::project::cameraToJson(camera);
    return imageObject;
}

QLineEdit *findLineEditByPlaceholder(QWidget *root, const QString &text)
{
    const QList<QLineEdit *> edits = root->findChildren<QLineEdit *>();
    for (QLineEdit *edit : edits)
    {
        if (edit->placeholderText().contains(text))
        {
            return edit;
        }
    }
    return nullptr;
}

QLineEdit *findModelPathEdit(QWidget *root)
{
    return findLineEditByPlaceholder(root, QStringLiteral("模型路径"));
}

QToolButton *findToolButton(QWidget *root, const QString &text)
{
    const QList<QToolButton *> buttons = root->findChildren<QToolButton *>();
    for (QToolButton *button : buttons)
    {
        if (button->text() == text)
        {
            return button;
        }
    }
    return nullptr;
}

QLabel *findLabelContaining(QWidget *root, const QString &text)
{
    const QList<QLabel *> labels = root->findChildren<QLabel *>();
    for (QLabel *label : labels)
    {
        if (label->text().contains(text))
        {
            return label;
        }
    }
    return nullptr;
}

QString readProjectSourceFile(const QString &relativePath)
{
    const QString projectRoot = QFileInfo(QStringLiteral(TEST_DATA_DIR)).absoluteDir().absolutePath();
    QFile file(QDir(projectRoot).filePath(relativePath));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

} // namespace

TEST(ProjectSupportUtilsTest, CollectMatchedPairsUsesFilenameWithSuffix)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString matchesDir = ProjectIO::ipmatchOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(matchesDir));

    QFile matchFile(QDir(matchesDir).filePath(QStringLiteral("1__2.match")));
    ASSERT_TRUE(matchFile.open(QIODevice::WriteOnly));
    matchFile.write("dummy");
    matchFile.close();

    QFile sidecarFile(QDir(matchesDir).filePath(QStringLiteral("1__2.match.json")));
    ASSERT_TRUE(sidecarFile.open(QIODevice::WriteOnly));
    sidecarFile.write(QJsonDocument(QJsonObject{{QStringLiteral("image0_name"), QStringLiteral("1")},
                                                {QStringLiteral("image1_name"), QStringLiteral("2")}}).toJson());
    sidecarFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/1.jpg")}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/2.png")}});
    images.append(QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/3.tif")}});

    QJsonArray ipmatchResults;
    ipmatchResults.append(QJsonObject{{QStringLiteral("image0"), QStringLiteral("/tmp/1.jpg")},
                                      {QStringLiteral("image1"), QStringLiteral("/tmp/3.tif")}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    meta[QStringLiteral("ipmatch_results")] = ipmatchResults;

    const QVector<QPair<QString, QString>> pairs =
        xjw::gui::project::collectMatchedImageNamePairs(projectPath, meta);

    EXPECT_TRUE(pairs.contains(qMakePair(QStringLiteral("1.jpg"), QStringLiteral("2.png"))));
    EXPECT_TRUE(pairs.contains(qMakePair(QStringLiteral("1.jpg"), QStringLiteral("3.tif"))));
}

TEST(ProjectSupportUtilsTest, CameraJsonRoundTripPreservesUnitsAndDepthDirection)
{
    xjw::Camera sourceCamera = makeCamera(1.25, -2.5, 3.75, 1200.0, 1195.0);
    sourceCamera.setAxisDirections(-1, 1);
    sourceCamera.setDepthAxisFlipped(true);
    sourceCamera.setDistortion(0.01, -0.001, 0.0001, 0.0002, -0.0003);

    const QJsonObject cameraJson = xjw::gui::project::cameraToJson(sourceCamera);

    xjw::Camera restoredCamera;
    ASSERT_TRUE(xjw::gui::project::cameraFromJson(cameraJson, &restoredCamera));

    EXPECT_DOUBLE_EQ(restoredCamera.focalX(), sourceCamera.focalX());
    EXPECT_DOUBLE_EQ(restoredCamera.focalY(), sourceCamera.focalY());
    EXPECT_DOUBLE_EQ(restoredCamera.principalX(), sourceCamera.principalX());
    EXPECT_DOUBLE_EQ(restoredCamera.principalY(), sourceCamera.principalY());
    EXPECT_EQ(restoredCamera.uAxisSign(), sourceCamera.uAxisSign());
    EXPECT_EQ(restoredCamera.vAxisSign(), sourceCamera.vAxisSign());
    EXPECT_EQ(restoredCamera.depthAxisFlipped(), sourceCamera.depthAxisFlipped());

    const auto sourceCenter = sourceCamera.cameraCenter();
    const auto restoredCenter = restoredCamera.cameraCenter();
    EXPECT_DOUBLE_EQ(restoredCenter[0], sourceCenter[0]);
    EXPECT_DOUBLE_EQ(restoredCenter[1], sourceCenter[1]);
    EXPECT_DOUBLE_EQ(restoredCenter[2], sourceCenter[2]);
}

TEST(ProjectSupportUtilsTest, ResolveFeaturePathFromTokenSupportsSuffix)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("1.jpg"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    const QString spPath = QDir(ipDir).filePath(QStringLiteral("1.sp"));
    QFile spFile(spPath);
    ASSERT_TRUE(spFile.open(QIODevice::WriteOnly));
    spFile.write("sp");
    spFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), imagePath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;

    EXPECT_EQ(xjw::gui::project::resolveProjectImagePathFromToken(QStringLiteral("1.jpg"), meta), imagePath);
    EXPECT_EQ(xjw::gui::project::resolveProjectFeaturePathFromToken(projectPath, meta, QStringLiteral("1.jpg")), spPath);
    EXPECT_EQ(xjw::gui::project::resolveProjectFeaturePathFromToken(projectPath, meta, imagePath), spPath);
}

TEST(ProjectSupportUtilsTest, InfersDiskFeatureSuffixWhenProjectHasOnlyDskOutputs)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("66.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    const QString dskPath = QDir(ipDir).filePath(QStringLiteral("66.dsk"));
    QFile dskFile(dskPath);
    ASSERT_TRUE(dskFile.open(QIODevice::WriteOnly));
    dskFile.write("dsk");
    dskFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), imagePath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;

    EXPECT_EQ(xjw::gui::project::inferPreferredFeatureSuffix(projectPath, meta),
              QStringLiteral(".dsk"));
}

TEST(ProjectSupportUtilsTest, DetectsWhetherProjectHasRequestedFeatureSuffix)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("66.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    QFile dskFile(QDir(ipDir).filePath(QStringLiteral("66.dsk")));
    ASSERT_TRUE(dskFile.open(QIODevice::WriteOnly));
    dskFile.write("dsk");
    dskFile.close();

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), imagePath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;

    EXPECT_TRUE(xjw::gui::project::projectHasFeatureSuffix(projectPath, meta, QStringLiteral(".dsk")));
    EXPECT_FALSE(xjw::gui::project::projectHasFeatureSuffix(projectPath, meta, QStringLiteral(".sp")));
}

TEST(ProjectSupportUtilsTest, ListsOnlyFeatureSuffixesPresentInProject)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString ipDir = ProjectIO::ipfindOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(ipDir));

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("75.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("img");
    imageFile.close();

    QFile dskFile(QDir(ipDir).filePath(QStringLiteral("75.dsk")));
    ASSERT_TRUE(dskFile.open(QIODevice::WriteOnly));
    dskFile.write("dsk");
    dskFile.close();

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), imagePath}}
    };

    const QStringList suffixes = xjw::gui::project::projectFeatureSuffixes(projectPath, meta);
    EXPECT_EQ(suffixes, QStringList{QStringLiteral(".dsk")});
}

TEST(MainMenuTest, ToolsMenuExposesCameraConversionAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.cameraConvertAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("相机格式转换")));

    bool foundInToolsMenu = false;
    const QList<QMenu *> menus = window.menuBar()->findChildren<QMenu *>();
    for (QMenu *candidate : menus)
    {
        if (candidate && candidate->title() == QStringLiteral("工具"))
        {
            foundInToolsMenu = candidate->actions().contains(action);
            break;
        }
    }
    EXPECT_TRUE(foundInToolsMenu);
}

TEST(MainWindowMenuWiringTest, CameraConversionActionIsConnectedToWorkflowController)
{
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("cameraConvertAction")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("openCameraConvertDialog")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("CameraConvertDialog")));
}

TEST(MainWindowChromeTest, KeepsNativeMaximizeControlAvailable)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("Qt::WindowMaximizeButtonHint")));
}

TEST(WindowStateManagerTest, PersistsAndRestoresMaximizedState)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/config/settings/WindowStateManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("MainWindow/isMaximized")));
    EXPECT_TRUE(source.contains(QStringLiteral("mainWindow->isMaximized()")));
    EXPECT_TRUE(source.contains(QStringLiteral("wasMaximized")));
    EXPECT_TRUE(source.contains(QStringLiteral("Qt::WindowMaximized")));
}

TEST(FeatureExtractionDialogTest, DiskSelectionShowsResolvedModelPath)
{
    FeatureExtractionDialog dialog;

    QJsonObject settings;
    settings[QStringLiteral("feature_algorithm")] = QStringLiteral("disk");
    settings[QStringLiteral("use_cuda")] = true;
    dialog.applySettings(settings);

    QLineEdit *modelPathEdit = findModelPathEdit(&dialog);
    ASSERT_NE(modelPathEdit, nullptr);
    EXPECT_FALSE(modelPathEdit->text().trimmed().isEmpty());
    EXPECT_TRUE(modelPathEdit->text().contains(QStringLiteral("disk_extractor")));
    EXPECT_TRUE(modelPathEdit->text().endsWith(QStringLiteral(".torchscript")));
}

TEST(FeatureExtractionDialogTest, DeviceSelectionControlsCudaModelAndConfig)
{
    FeatureExtractionDialog dialog;

    QJsonObject settings;
    settings[QStringLiteral("feature_algorithm")] = QStringLiteral("disk");
    settings[QStringLiteral("device")] = QStringLiteral("CUDA");
    settings[QStringLiteral("use_cuda")] = false;
    dialog.applySettings(settings);

    QLineEdit *modelPathEdit = findModelPathEdit(&dialog);
    ASSERT_NE(modelPathEdit, nullptr);
    EXPECT_TRUE(modelPathEdit->text().contains(QStringLiteral("disk_extractor_cuda")))
        << modelPathEdit->text().toStdString();

    QJsonObject emittedConfig;
    QObject::connect(&dialog, &FeatureExtractionDialog::runRequested, &dialog,
                     [&emittedConfig](const QJsonObject &config, const QStringList &)
                     {
                         emittedConfig = config;
                     });
    dialog.setProjectImages(QStringList{QStringLiteral("/tmp/1.png")});
    QPushButton *runButton = dialog.findChild<QPushButton *>(QStringLiteral("m_runBtn"));
    ASSERT_NE(runButton, nullptr);
    runButton->click();

    EXPECT_TRUE(emittedConfig.value(QStringLiteral("use_cuda")).toBool());
    EXPECT_EQ(emittedConfig.value(QStringLiteral("device")).toString(), QStringLiteral("CUDA"));
}

TEST(FeatureExtractionDialogTest, CudaCheckboxControlsRuntimeDevice)
{
    FeatureExtractionDialog dialog;

    QJsonObject settings;
    settings[QStringLiteral("feature_algorithm")] = QStringLiteral("disk");
    settings[QStringLiteral("device")] = QStringLiteral("CPU");
    dialog.applySettings(settings);

    QCheckBox *cudaCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_useCudaChk"));
    ASSERT_NE(cudaCheck, nullptr);
    cudaCheck->setChecked(true);

    QJsonObject emittedConfig;
    QObject::connect(&dialog, &FeatureExtractionDialog::runRequested, &dialog,
                     [&emittedConfig](const QJsonObject &config, const QStringList &)
                     {
                         emittedConfig = config;
                     });
    dialog.setProjectImages(QStringList{QStringLiteral("/tmp/1.png")});
    QPushButton *runButton = dialog.findChild<QPushButton *>(QStringLiteral("m_runBtn"));
    ASSERT_NE(runButton, nullptr);
    runButton->click();

    EXPECT_TRUE(emittedConfig.value(QStringLiteral("use_cuda")).toBool());
    EXPECT_EQ(emittedConfig.value(QStringLiteral("device")).toString(), QStringLiteral("CUDA"));
}

TEST(FeatureExtractionDialogTest, DefaultsToDiskAlgorithm)
{
    FeatureExtractionDialog dialog;

    QComboBox *algorithmCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_algorithmCombo"));
    ASSERT_NE(algorithmCombo, nullptr);
    EXPECT_EQ(algorithmCombo->currentData().toString(), QStringLiteral("disk"));

    QPushButton *resetButton = dialog.findChild<QPushButton *>(QStringLiteral("m_resetBtn"));
    ASSERT_NE(resetButton, nullptr);
    resetButton->click();

    EXPECT_EQ(algorithmCombo->currentData().toString(), QStringLiteral("disk"));
}

TEST(FeatureExtractionDialogTest, GrayscaleThresholdLivesInAdvancedParameters)
{
    for (const QString &algorithm : {
             QStringLiteral("superpoint"),
             QStringLiteral("disk"),
             QStringLiteral("aliked"),
             QStringLiteral("orb"),
             QStringLiteral("sift")
         })
    {
        FeatureExtractionDialog dialog;

        QJsonObject settings;
        settings[QStringLiteral("feature_algorithm")] = algorithm;
        dialog.applySettings(settings);

        QToolButton *advancedButton = findToolButton(&dialog, QStringLiteral("高级参数"));
        ASSERT_NE(advancedButton, nullptr);
        advancedButton->setChecked(true);
        dialog.show();
        QApplication::processEvents();

        QGroupBox *advancedGroup = dialog.findChild<QGroupBox *>(QStringLiteral("m_advancedGroup"));
        QWidget *grayRangeWidget = dialog.findChild<QWidget *>(QStringLiteral("m_grayRangeWidget"));
        ASSERT_NE(advancedGroup, nullptr);
        ASSERT_NE(grayRangeWidget, nullptr);

        EXPECT_TRUE(advancedGroup->isAncestorOf(grayRangeWidget)) << algorithm.toStdString();
        EXPECT_TRUE(grayRangeWidget->isVisibleTo(&dialog)) << algorithm.toStdString();
    }
}

TEST(FeatureExtractionDialogTest, GrayscaleThresholdUsesPixelValuesAndEmitsNormalizedRange)
{
    FeatureExtractionDialog dialog;

    QSpinBox *minSpin = dialog.findChild<QSpinBox *>(QStringLiteral("m_grayscaleMinSpin"));
    QSpinBox *maxSpin = dialog.findChild<QSpinBox *>(QStringLiteral("m_grayscaleMaxSpin"));
    ASSERT_NE(minSpin, nullptr);
    ASSERT_NE(maxSpin, nullptr);

    EXPECT_EQ(minSpin->minimum(), 0);
    EXPECT_EQ(minSpin->maximum(), 255);
    EXPECT_EQ(minSpin->value(), 5);
    EXPECT_EQ(maxSpin->minimum(), 0);
    EXPECT_EQ(maxSpin->maximum(), 255);
    EXPECT_EQ(maxSpin->value(), 255);
    EXPECT_TRUE(minSpin->toolTip().contains(QStringLiteral("0-255")));

    QJsonObject emittedConfig;
    QObject::connect(&dialog, &FeatureExtractionDialog::runRequested, &dialog,
                     [&emittedConfig](const QJsonObject &config, const QStringList &)
                     {
                         emittedConfig = config;
                     });

    dialog.setProjectImages(QStringList{QStringLiteral("/tmp/1.png")});
    QPushButton *runButton = dialog.findChild<QPushButton *>(QStringLiteral("m_runBtn"));
    ASSERT_NE(runButton, nullptr);
    runButton->click();

    EXPECT_NEAR(emittedConfig.value(QStringLiteral("grayscale_min")).toDouble(), 5.0 / 255.0, 1e-6);
    EXPECT_DOUBLE_EQ(emittedConfig.value(QStringLiteral("grayscale_max")).toDouble(), 1.0);
    EXPECT_EQ(emittedConfig.value(QStringLiteral("grayscale_min_px")).toInt(), 5);
    EXPECT_EQ(emittedConfig.value(QStringLiteral("grayscale_max_px")).toInt(), 255);
}

TEST(FeatureExtractionDialogTest, NativeFeatureRunnerReceivesGrayscaleRange)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/tasks/SuperPointRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("extractorCfg.grayscaleMin = spConfig.grayscale_min")));
    EXPECT_TRUE(source.contains(QStringLiteral("extractorCfg.grayscaleMax = spConfig.grayscale_max")));
    EXPECT_TRUE(source.contains(QStringLiteral("config[\"grayscale_min\"].toDouble(5.0 / 255.0)")));
}

TEST(FeatureExtractionDialogTest, DiskSelectionHidesSuperPointOnlyAdvancedRows)
{
    FeatureExtractionDialog dialog;

    QJsonObject settings;
    settings[QStringLiteral("feature_algorithm")] = QStringLiteral("disk");
    dialog.applySettings(settings);

    QToolButton *advancedButton = findToolButton(&dialog, QStringLiteral("高级参数"));
    ASSERT_NE(advancedButton, nullptr);
    advancedButton->setChecked(true);
    dialog.show();
    QApplication::processEvents();

    QLabel *descriptorLabel = findLabelContaining(&dialog, QStringLiteral("描述子维度"));
    QLabel *gridLabel = findLabelContaining(&dialog, QStringLiteral("网格大小"));
    ASSERT_NE(descriptorLabel, nullptr);
    ASSERT_NE(gridLabel, nullptr);

    EXPECT_FALSE(descriptorLabel->isVisibleTo(&dialog));
    EXPECT_FALSE(gridLabel->isVisibleTo(&dialog));
}

TEST(FeatureExtractionDialogTest, DiskSelectionShowsAdvancedApplicabilityHint)
{
    FeatureExtractionDialog dialog;

    QJsonObject settings;
    settings[QStringLiteral("feature_algorithm")] = QStringLiteral("disk");
    dialog.applySettings(settings);

    QToolButton *advancedButton = findToolButton(&dialog, QStringLiteral("高级参数"));
    ASSERT_NE(advancedButton, nullptr);
    advancedButton->setChecked(true);
    dialog.show();
    QApplication::processEvents();

    QLabel *advancedHintLabel = findLabelContaining(&dialog, QStringLiteral("DISK/ALIKED"));
    ASSERT_NE(advancedHintLabel, nullptr);
    EXPECT_TRUE(advancedHintLabel->isVisibleTo(&dialog));
}

TEST(FeatureExtractionDialogTest, PreservesConfiguredPythonExecutable)
{
    FeatureExtractionDialog dialog;

    const QString pythonPath = QStringLiteral("/tmp/plascan-lightglue-env/bin/python");
    QJsonObject settings;
    settings[QStringLiteral("python_executable")] = pythonPath;
    dialog.applySettings(settings);

    QLineEdit *pythonPathEdit = findLineEditByPlaceholder(&dialog, QStringLiteral("Python"));
    ASSERT_NE(pythonPathEdit, nullptr);
    EXPECT_EQ(pythonPathEdit->text(), pythonPath);

    QJsonObject emittedSettings;
    QObject::connect(&dialog, &FeatureExtractionDialog::settingsChanged, &dialog,
                     [&emittedSettings](const QJsonObject &settings)
                     {
                         emittedSettings = settings;
                     });

    const QString changedPythonPath = QStringLiteral("/tmp/another-lightglue-env/bin/python");
    pythonPathEdit->setText(changedPythonPath);
    QApplication::processEvents();

    EXPECT_EQ(emittedSettings.value(QStringLiteral("python_executable")).toString(), changedPythonPath);
}

TEST(FeatureMatchingDialogTest, DefaultsToLightGlueAlgorithm)
{
    FeatureMatchingDialog dialog;

    QComboBox *algorithmCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_matchAlgorithmCombo"));
    ASSERT_NE(algorithmCombo, nullptr);
    EXPECT_EQ(algorithmCombo->currentData().toString(), QStringLiteral("lightglue"));

    QStackedWidget *paramStack = dialog.findChild<QStackedWidget *>(QStringLiteral("m_paramStack"));
    ASSERT_NE(paramStack, nullptr);
    EXPECT_EQ(paramStack->currentIndex(), 1);

    QPushButton *resetButton = dialog.findChild<QPushButton *>(QStringLiteral("m_resetBtn"));
    ASSERT_NE(resetButton, nullptr);
    resetButton->click();

    EXPECT_EQ(algorithmCombo->currentData().toString(), QStringLiteral("lightglue"));
    EXPECT_EQ(paramStack->currentIndex(), 1);
}

TEST(FeatureMatchingDialogTest, ProjectAvailableSuffixesConstrainLightGlueChoicesAfterApplyingSettings)
{
    FeatureMatchingDialog dialog;
    dialog.setAvailableFeatureSuffixes(QStringList{QStringLiteral(".dsk")});

    QJsonObject settings;
    settings[QStringLiteral("match_algorithm")] = QStringLiteral("lightglue");
    settings[QStringLiteral("feature_suffix")] = QStringLiteral("__all__");
    dialog.applySettings(settings);

    QComboBox *suffixCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_featureSuffixCombo"));
    ASSERT_NE(suffixCombo, nullptr);
    ASSERT_EQ(suffixCombo->count(), 1);
    EXPECT_EQ(suffixCombo->itemData(0).toString(), QStringLiteral(".dsk"));
    EXPECT_EQ(dialog.selectedFeatureSuffix(), QStringLiteral(".dsk"));
}

TEST(InitCameraPoseDialogTest, ExposesSelectedMatchPipelineInSettings)
{
    InitCameraPoseDialog dialog;

    QComboBox *matchAlgorithmCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_matchAlgorithmCombo"));
    ASSERT_NE(matchAlgorithmCombo, nullptr);
    EXPECT_EQ(matchAlgorithmCombo->currentData().toString(), QStringLiteral("lightglue"));

    QComboBox *featureSuffixCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_featureSuffixCombo"));
    ASSERT_NE(featureSuffixCombo, nullptr);
    EXPECT_EQ(featureSuffixCombo->currentData().toString(), QStringLiteral(".dsk"));

    QJsonObject emittedSettings;
    QObject::connect(&dialog, &InitCameraPoseDialog::settingsChanged, &dialog,
                     [&emittedSettings](const QJsonObject &settings)
                     {
                         emittedSettings = settings;
                     });

    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitSettingsNow", Qt::DirectConnection));
    EXPECT_EQ(emittedSettings.value(QStringLiteral("match_algorithm")).toString(), QStringLiteral("lightglue"));
    EXPECT_EQ(emittedSettings.value(QStringLiteral("feature_algorithm")).toString(), QStringLiteral("disk"));
    EXPECT_EQ(emittedSettings.value(QStringLiteral("feature_suffix")).toString(), QStringLiteral(".dsk"));
}

TEST(InitCameraPoseDialogTest, SfmInitializationUsesSelectedMatchPipeline)
{
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    const QString managerSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectCameraSetupManager.cpp"));
    const QString sfmServiceSource =
        readProjectSourceFile(QStringLiteral("src/core/pipeline/SFMService.cpp"));
    ASSERT_FALSE(controllerSource.isEmpty());
    ASSERT_FALSE(managerSource.isEmpty());
    ASSERT_FALSE(sfmServiceSource.isEmpty());

    EXPECT_TRUE(controllerSource.contains(QStringLiteral("projectFeatureSuffixes")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("setAvailableFeatureSuffixes")));

    EXPECT_TRUE(managerSource.contains(QStringLiteral("settings.value(QStringLiteral(\"feature_algorithm\")")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("settings.value(QStringLiteral(\"match_algorithm\")")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("opts.featureAlgorithm")));
    EXPECT_TRUE(managerSource.contains(QStringLiteral("opts.matchAlgorithm")));

    EXPECT_TRUE(sfmServiceSource.contains(QStringLiteral("isExistingMatchOnlyMode")));
    EXPECT_TRUE(sfmServiceSource.contains(QStringLiteral("compatibleFeatureSuffixes(matchAlgorithm)")));
}

TEST(ThreeDReconstructionDialogTest, UsesUiDefaultsAndImageCountGate)
{
    ThreeDReconstructionDialog dialog;

    auto *outputEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_outputDirEdit"));
    auto *startButton = dialog.findChild<QPushButton *>(QStringLiteral("m_startBtn"));
    auto *featureGrayMinSpin = dialog.findChild<QSpinBox *>(QStringLiteral("m_featureGrayMinSpin"));
    auto *generateDemCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_generateDemCheck"));
    auto *generateDomCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_generateDomCheck"));
    auto *demResolutionSpin = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("m_demResolutionSpin"));
    ASSERT_NE(outputEdit, nullptr);
    ASSERT_NE(startButton, nullptr);
    ASSERT_NE(featureGrayMinSpin, nullptr);
    EXPECT_EQ(generateDemCheck, nullptr);
    EXPECT_EQ(generateDomCheck, nullptr);
    EXPECT_EQ(demResolutionSpin, nullptr);
    EXPECT_EQ(featureGrayMinSpin->minimum(), 0);
    EXPECT_EQ(featureGrayMinSpin->maximum(), 255);
    EXPECT_EQ(featureGrayMinSpin->value(), 5);

    dialog.setImageCount(1);
    EXPECT_FALSE(startButton->isEnabled());
    dialog.setImageCount(2);
    EXPECT_TRUE(startButton->isEnabled());

    dialog.setDefaultOutputDir(QStringLiteral("/tmp/plascan-model"));
    QJsonObject settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("quality")).toString(), QStringLiteral("standard"));
    EXPECT_EQ(settings.value(QStringLiteral("device")).toString(), QStringLiteral("auto"));
    EXPECT_GE(settings.value(QStringLiteral("threads")).toInt(), 1);
    EXPECT_EQ(settings.value(QStringLiteral("output_dir")).toString(),
              QDir::cleanPath(QStringLiteral("/tmp/plascan-model")));
    EXPECT_TRUE(settings.value(QStringLiteral("export_obj")).toBool());
    EXPECT_EQ(settings.value(QStringLiteral("feature_grayscale_min_px")).toInt(), 5);
    EXPECT_NEAR(settings.value(QStringLiteral("feature_grayscale_min")).toDouble(), 5.0 / 255.0, 1e-6);
    EXPECT_FALSE(settings.contains(QStringLiteral("generate_dem")));
    EXPECT_FALSE(settings.contains(QStringLiteral("generate_dom")));
    EXPECT_FALSE(settings.contains(QStringLiteral("dem_resolution")));
    EXPECT_FALSE(settings.contains(QStringLiteral("dom_resolution")));

    QJsonObject appliedSettings;
    appliedSettings[QStringLiteral("quality")] = QStringLiteral("fast");
    appliedSettings[QStringLiteral("device")] = QStringLiteral("cpu");
    appliedSettings[QStringLiteral("threads")] = 3;
    appliedSettings[QStringLiteral("output_dir")] = QStringLiteral("/tmp/another-model");
    appliedSettings[QStringLiteral("export_obj")] = true;
    dialog.applySettings(appliedSettings);

    settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("quality")).toString(), QStringLiteral("fast"));
    EXPECT_EQ(settings.value(QStringLiteral("device")).toString(), QStringLiteral("cpu"));
    EXPECT_EQ(settings.value(QStringLiteral("threads")).toInt(), 3);
    EXPECT_EQ(settings.value(QStringLiteral("output_dir")).toString(),
              QDir::cleanPath(QStringLiteral("/tmp/another-model")));
    EXPECT_TRUE(settings.value(QStringLiteral("export_obj")).toBool());
    EXPECT_FALSE(settings.contains(QStringLiteral("generate_dem")));
    EXPECT_FALSE(settings.contains(QStringLiteral("generate_dom")));
    EXPECT_FALSE(settings.contains(QStringLiteral("dem_resolution")));
    EXPECT_FALSE(settings.contains(QStringLiteral("dom_resolution")));
}

TEST(BundleAdjustDialogTest, KeepsActionButtonsOutsideScrollableParameterArea)
{
    BundleAdjustDialog dialog;

    QScrollArea *parameterScrollArea = dialog.findChild<QScrollArea *>(QStringLiteral("parameterScrollArea"));
    ASSERT_NE(parameterScrollArea, nullptr);
    EXPECT_TRUE(parameterScrollArea->widgetResizable());

    QWidget *parameterScrollWidget = dialog.findChild<QWidget *>(QStringLiteral("parameterScrollWidget"));
    ASSERT_NE(parameterScrollWidget, nullptr);

    QWidget *debugGroup = dialog.findChild<QWidget *>(QStringLiteral("debugGroup"));
    ASSERT_NE(debugGroup, nullptr);
    EXPECT_TRUE(parameterScrollWidget->isAncestorOf(debugGroup));

    QPushButton *runButton = dialog.findChild<QPushButton *>(QStringLiteral("runBtn"));
    QPushButton *closeButton = dialog.findChild<QPushButton *>(QStringLiteral("closeBtn"));
    ASSERT_NE(runButton, nullptr);
    ASSERT_NE(closeButton, nullptr);
    EXPECT_FALSE(parameterScrollWidget->isAncestorOf(runButton));
    EXPECT_FALSE(parameterScrollWidget->isAncestorOf(closeButton));
}

TEST(ParameterDialogLayoutTest, UiFilesKeepLongDialogFootersOutsideScrollableParameterArea)
{
    const QString meshUi = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MeshReconstructionDialog.ui"));
    ASSERT_FALSE(meshUi.isEmpty());
    EXPECT_TRUE(meshUi.contains(QStringLiteral("QScrollArea\" name=\"parameterScrollArea")));
    EXPECT_TRUE(meshUi.contains(QStringLiteral("<bool>true</bool>")));
    EXPECT_LT(meshUi.indexOf(QStringLiteral("parameterScrollWidget")),
              meshUi.indexOf(QStringLiteral("systemGroup")));
    EXPECT_LT(meshUi.indexOf(QStringLiteral("systemGroup")),
              meshUi.indexOf(QStringLiteral("buttonLayout")));

    const QString triangulationUi =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/TriangulationDialog.ui"));
    ASSERT_FALSE(triangulationUi.isEmpty());
    EXPECT_TRUE(triangulationUi.contains(QStringLiteral("QScrollArea\" name=\"parameterScrollArea")));
    EXPECT_LT(triangulationUi.indexOf(QStringLiteral("parameterScrollWidget")),
              triangulationUi.indexOf(QStringLiteral("suggestGroup")));
    EXPECT_LT(triangulationUi.indexOf(QStringLiteral("suggestGroup")),
              triangulationUi.indexOf(QStringLiteral("buttonLayout")));

    const QString sparseUi =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/SparseCloudPostProcessDialog.ui"));
    ASSERT_FALSE(sparseUi.isEmpty());
    EXPECT_TRUE(sparseUi.contains(QStringLiteral("QScrollArea\" name=\"parameterScrollArea")));
    EXPECT_LT(sparseUi.indexOf(QStringLiteral("parameterScrollWidget")),
              sparseUi.indexOf(QStringLiteral("m_spatialGroup")));
    EXPECT_LT(sparseUi.indexOf(QStringLiteral("m_spatialGroup")),
              sparseUi.indexOf(QStringLiteral("buttonBox")));
}

TEST(DenseMatchDialogLayoutTest, UiFileKeepsRightParametersScrollableAndFooterFixed)
{
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/dialogs/DenseMatchDialog.ui"));
    ASSERT_FALSE(ui.isEmpty());
    EXPECT_TRUE(ui.contains(QStringLiteral("QScrollArea\" name=\"rightParameterScrollArea")));
    EXPECT_LT(ui.indexOf(QStringLiteral("rightParameterScrollWidget")),
              ui.indexOf(QStringLiteral("postGroup")));
    EXPECT_LT(ui.indexOf(QStringLiteral("postGroup")),
              ui.indexOf(QStringLiteral("buttonLayout")));

    const QString featureExtractionUi =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/FeatureExtractionDialog.ui"));
    ASSERT_FALSE(featureExtractionUi.isEmpty());
    EXPECT_TRUE(featureExtractionUi.contains(QStringLiteral("QScrollArea\" name=\"rightParameterScrollArea")));
    EXPECT_LT(featureExtractionUi.indexOf(QStringLiteral("rightParameterScrollWidget")),
              featureExtractionUi.indexOf(QStringLiteral("m_debugGroup")));
    EXPECT_LT(featureExtractionUi.indexOf(QStringLiteral("m_debugGroup")),
              featureExtractionUi.indexOf(QStringLiteral("bottomLayout")));

    const QString featureMatchingUi =
        readProjectSourceFile(QStringLiteral("src/gui/dialogs/FeatureMatchingDialog.ui"));
    ASSERT_FALSE(featureMatchingUi.isEmpty());
    EXPECT_TRUE(featureMatchingUi.contains(QStringLiteral("QScrollArea\" name=\"rightParameterScrollArea")));
    EXPECT_LT(featureMatchingUi.indexOf(QStringLiteral("rightParameterScrollWidget")),
              featureMatchingUi.indexOf(QStringLiteral("m_debugGroup")));
    EXPECT_LT(featureMatchingUi.indexOf(QStringLiteral("m_debugGroup")),
              featureMatchingUi.indexOf(QStringLiteral("buttonLayout")));
}

TEST(DepthMapEstimateDialogTooltipTest, UiExplainsCostFunctionAndParameters)
{
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/dialogs/DepthMapEstimateDialog.ui"));
    ASSERT_FALSE(ui.isEmpty());

    const QStringList expectedPhrases = {
        QStringLiteral("AD"),
        QStringLiteral("灰度绝对差"),
        QStringLiteral("SD"),
        QStringLiteral("平方差"),
        QStringLiteral("NCC"),
        QStringLiteral("归一化互相关"),
        QStringLiteral("Census"),
        QStringLiteral("局部灰度排序"),
        QStringLiteral("Ternary Census"),
        QStringLiteral("三值"),
        QStringLiteral("分辨率缩放"),
        QStringLiteral("迭代次数"),
        QStringLiteral("窗口大小"),
        QStringLiteral("最少视图"),
        QStringLiteral("深度搜索范围"),
        QStringLiteral("置信度阈值"),
        QStringLiteral("Tile"),
        QStringLiteral("CPU 线程数")
    };

    for (const QString &phrase : expectedPhrases)
    {
        EXPECT_TRUE(ui.contains(phrase)) << phrase.toStdString();
    }
}

TEST(DepthMapEstimateDialogTooltipTest, CostFunctionComboItemsHaveTooltips)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/DepthMapEstimateDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("Qt::ToolTipRole")));
    EXPECT_TRUE(source.contains(QStringLiteral("costFunctionToolTip")));
    EXPECT_TRUE(source.contains(QStringLiteral("updateCostFunctionToolTip")));
}

TEST(TaskStatusWidgetTest, ShowsProgressAndPreservesCancellingState)
{
    TaskStatusWidget widget;
    widget.setCancellable(true);
    widget.setCancellingText(QStringLiteral("正在取消特征匹配..."));

    bool cancelEmitted = false;
    QObject::connect(&widget, &TaskStatusWidget::cancelRequested, &widget,
                     [&cancelEmitted]()
                     {
                         cancelEmitted = true;
                     });

    widget.begin(QStringLiteral("特征匹配 0/5"), 0, 5);
    EXPECT_TRUE(widget.isActive());
    EXPECT_EQ(widget.statusText(), QStringLiteral("特征匹配 0/5"));
    EXPECT_EQ(widget.progressValue(), 0);
    EXPECT_EQ(widget.progressMaximum(), 5);

    widget.updateProgress(QStringLiteral("特征匹配 2/5"), 2);
    EXPECT_EQ(widget.statusText(), QStringLiteral("特征匹配 2/5"));
    EXPECT_EQ(widget.progressValue(), 2);

    QToolButton *cancelButton = widget.findChild<QToolButton *>(QStringLiteral("cancelButton"));
    ASSERT_NE(cancelButton, nullptr);
    ASSERT_TRUE(cancelButton->isEnabled());
    cancelButton->click();

    EXPECT_TRUE(cancelEmitted);
    EXPECT_TRUE(widget.isCancelling());
    EXPECT_FALSE(cancelButton->isEnabled());
    EXPECT_EQ(cancelButton->text(), QStringLiteral("正在取消"));
    EXPECT_EQ(widget.statusText(), QStringLiteral("正在取消特征匹配..."));

    widget.updateProgress(QStringLiteral("特征匹配 3/5"), 3);
    EXPECT_EQ(widget.progressValue(), 3);
    EXPECT_EQ(widget.statusText(), QStringLiteral("正在取消特征匹配..."));

    widget.finish();
    EXPECT_FALSE(widget.isActive());
    EXPECT_FALSE(widget.isCancelling());
    EXPECT_EQ(cancelButton->text(), QStringLiteral("取消"));
}

TEST(MenuWorkflowControllerTest, DenseStageAdvancesOnMvsSuccessWithoutRequiringChangedOutputPath)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_FALSE(source.contains(QStringLiteral("densePath == beforePath")));
    EXPECT_TRUE(source.contains(QStringLiteral("mvsProgressFinished")));
    EXPECT_TRUE(source.contains(QStringLiteral("startThreeDReconstructionDenseRefineStage(settings)")));
    EXPECT_TRUE(source.contains(QStringLiteral("startDenseCloudRefineAsync(refineSettings)")));
    EXPECT_TRUE(source.contains(QStringLiteral("startThreeDReconstructionMeshStage(settings)")));
}

TEST(MainWindowProgressTest, FeatureMatchProgressExpandsAllFeatureModeAndClampsDisplay)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("projectFeatureSuffixes")));
    EXPECT_TRUE(source.contains(QStringLiteral("feature_suffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::clamp(done")));
}

TEST(MainWindowFeatureMatchingTest, DialogUsesProjectFeatureSuffixesInsteadOfCurrentCanvasOnly)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("projectFeatureSuffixes")));
    EXPECT_FALSE(source.contains(QStringLiteral("m_canvas->availableFeatureSuffixes()")));
}

TEST(MainWindowCancelTest, StatusBarCancelButtonsGiveImmediateFeedbackAndEmitSignals)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("sgCancelRequested")));
    EXPECT_TRUE(header.contains(QStringLiteral("spCancelRequested")));
    EXPECT_TRUE(source.contains(QStringLiteral("正在取消特征匹配")));
    EXPECT_TRUE(source.contains(QStringLiteral("正在取消特征提取")));
    EXPECT_TRUE(source.contains(QStringLiteral("m_sgTaskStatus")));
    EXPECT_TRUE(source.contains(QStringLiteral("m_spTaskStatus")));
    EXPECT_TRUE(source.contains(QStringLiteral("TaskStatusWidget::cancelRequested")));
    EXPECT_TRUE(source.contains(QStringLiteral("emit sgCancelRequested()")));
    EXPECT_TRUE(source.contains(QStringLiteral("emit spCancelRequested()")));
}

TEST(BundleAdjustStatusBarTest, UsesAtProgressWidgetWithCancelableCoreOptimization)
{
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString projectManagerSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString bundleAdjustHeader = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.h"));
    const QString bundleAdjustSource = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.cpp"));
    const QString serviceSource = readProjectSourceFile(QStringLiteral("src/gui/project/services/BundleAdjustService.cpp"));
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(projectManagerSource.isEmpty());
    ASSERT_FALSE(bundleAdjustHeader.isEmpty());
    ASSERT_FALSE(bundleAdjustSource.isEmpty());
    ASSERT_FALSE(serviceSource.isEmpty());

    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("正在取消空三/光束法平差")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("std::make_shared<std::atomic<bool>>(false)")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("opts.baOpt.cancelFlag = cancelFlag")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("opts.baOpt.progressCallback")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("光束法平差优化中")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("emit self->atProgressFinished")));
    EXPECT_TRUE(bundleAdjustHeader.contains(QStringLiteral("std::shared_ptr<std::atomic<bool>> cancelFlag")));
    EXPECT_TRUE(bundleAdjustHeader.contains(QStringLiteral("progressCallback")));
    EXPECT_TRUE(bundleAdjustSource.contains(QStringLiteral("isCancelled(options)")));
    EXPECT_TRUE(serviceSource.contains(QStringLiteral("用户取消了光束法平差")));
}

TEST(FeatureMatchRunnerCancelTest, PythonProcessesArePolledForCancellation)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/pipeline/FeatureMatchRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("waitForProcessOrCancel")));
    EXPECT_TRUE(source.contains(QStringLiteral("waitForFinished(100)")));
    EXPECT_TRUE(source.contains(QStringLiteral("cancelFlag.load()")));
    EXPECT_TRUE(source.contains(QStringLiteral("process.kill()")));
}

TEST(FeatureMatchRunnerTest, AllFeatureModeFiltersUnavailableSuffixesAndSummarizesMissingFeatures)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/pipeline/FeatureMatchRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("projectFeatureSuffixes")));
    EXPECT_TRUE(source.contains(QStringLiteral("availableSuffixes")));
    EXPECT_TRUE(source.contains(QStringLiteral("缺失特征文件")));
    EXPECT_FALSE(source.contains(QStringLiteral("LOG_ERROR(\"%s\", qUtf8Printable(QString(\"特征文件不存在: %1 或 %2\")")));
}

TEST(DenseMatchCancelTest, StatusBarCancelSignalStopsRemainingPairs)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/ReconstructionWorkflowController.cpp"));
    const QString projectManagerHeader =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString projectManagerSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());
    ASSERT_FALSE(controllerSource.isEmpty());
    ASSERT_FALSE(projectManagerHeader.isEmpty());
    ASSERT_FALSE(projectManagerSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("dmCancelRequested")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("m_dmTaskStatus")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("emit dmCancelRequested()")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("MainWindow::dmCancelRequested")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("dmCancelFlag")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("hideDmProgress(!cancelled)")));
    EXPECT_TRUE(projectManagerHeader.contains(QStringLiteral("std::shared_ptr<std::atomic<bool>> cancelFlag")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("cancelFlag && cancelFlag->load()")));
    EXPECT_TRUE(projectManagerSource.contains(QStringLiteral("密集匹配已请求取消")));
}

TEST(ForwardIntersectionCheckDialogTest, AutoModeAcceptsPythonLightGlueSidecarTokens)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/ForwardIntersectionCheckDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("feature0_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("feature1_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("sp0_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("sp1_path")));
    EXPECT_TRUE(source.contains(QStringLiteral("image0_name")));
    EXPECT_TRUE(source.contains(QStringLiteral("image1_name")));
}

TEST(FeatureVisualizationSettingsTest, PersistsAndRestoresActiveFeatureSuffix)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("feature_suffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("setActiveFeatureSuffix")));
}

TEST(FeatureVisualizationSettingsTest, DefaultsToOnePixelCrossMarker)
{
    const QString rendererHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    const QString uiDefaults = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectUiConfigManager.cpp"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/SuperPointVisualizationDialog.cpp"));
    const QString dialogUi = readProjectSourceFile(QStringLiteral("src/gui/dialogs/SuperPointVisualizationDialog.ui"));
    ASSERT_FALSE(rendererHeader.isEmpty());
    ASSERT_FALSE(rendererSource.isEmpty());
    ASSERT_FALSE(uiDefaults.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());
    ASSERT_FALSE(dialogUi.isEmpty());

    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("int pointSize = 1")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("markerShape = QStringLiteral(\"cross\")")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("const double crossRadius")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("crossPen.setWidthF(1.0)")));

    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("featureDisplay[\"pointSize\"]         = 1")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("featureDisplay[\"markerShape\"]       = QStringLiteral(\"cross\")")));

    EXPECT_TRUE(dialogSource.contains(QStringLiteral("m_markerShapeCombo->setCurrentIndex(2)")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("m_pointSizeSpin->setValue(1)")));
    EXPECT_TRUE(dialogUi.contains(QStringLiteral("<number>1</number>")));
}

TEST(FeatureVisualizationSettingsTest, ProjectOpenRestoresFeatureSuffixEvenWhenUiSettingsAreEmpty)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int applyIndex = source.indexOf(QStringLiteral("applySavedFeatureDisplayOptions(ui)"));
    const int emptyReturnIndex = source.indexOf(QStringLiteral("if (ui.isEmpty())"), source.indexOf(QStringLiteral("void MainWindow::applyUiSettings")));
    ASSERT_GE(applyIndex, 0);
    ASSERT_GE(emptyReturnIndex, 0);
    EXPECT_LT(applyIndex, emptyReturnIndex);
}

TEST(FeatureVisualizationSettingsTest, VisualizationRestoreInfersSuffixFromExistingFeatureFiles)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("inferPreferredFeatureSuffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("inferredSuffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("setActiveFeatureSuffix(inferredSuffix)")));
}

TEST(FeatureVisualizationSettingsTest, SavedSuffixIsUsedOnlyWhenProjectContainsThatSuffix)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("projectHasFeatureSuffix")));
    EXPECT_TRUE(source.contains(QStringLiteral("savedSuffixUsable")));
}

TEST(MapProjectDialogTest, DefaultsToOneClickDomEngineeringSettings)
{
    MapProjectDialog dialog;
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectRoot = tempDir.path();
    const QString demPath = QDir(projectRoot).filePath(QStringLiteral("assets/dem/relative_dem/dem.tif"));
    const QString expectedOutput = QDir(projectRoot).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(demPath).absolutePath()));
    QFile demFile(demPath);
    ASSERT_TRUE(demFile.open(QIODevice::WriteOnly));
    demFile.write("dem");
    demFile.close();

    const QStringList images{
        QDir(projectRoot).filePath(QStringLiteral("assets/img/1.png")),
        QDir(projectRoot).filePath(QStringLiteral("assets/img/2.png"))
    };

    dialog.setAvailableImages(images);
    dialog.setProjectRoot(projectRoot);
    dialog.setDefaultDemPath(demPath);

    auto *imageList = dialog.findChild<QListWidget *>(QStringLiteral("m_imageList"));
    auto *demEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_demEdit"));
    auto *outputEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_outputEdit"));
    auto *resolutionSpin = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("m_resolutionSpin"));
    auto *runButton = dialog.findChild<QPushButton *>(QStringLiteral("runBtn"));

    ASSERT_NE(imageList, nullptr);
    ASSERT_NE(demEdit, nullptr);
    ASSERT_NE(outputEdit, nullptr);
    ASSERT_NE(resolutionSpin, nullptr);
    ASSERT_NE(runButton, nullptr);

    ASSERT_EQ(imageList->count(), images.size());
    for (int i = 0; i < imageList->count(); ++i)
    {
        EXPECT_EQ(imageList->item(i)->checkState(), Qt::Checked);
    }
    EXPECT_EQ(demEdit->text(), demPath);
    EXPECT_EQ(outputEdit->text(), expectedOutput);
    EXPECT_DOUBLE_EQ(resolutionSpin->value(), 0.0);
    EXPECT_TRUE(resolutionSpin->specialValueText().contains(QStringLiteral("自动")));
    EXPECT_TRUE(runButton->text().contains(QStringLiteral("一键")));
    EXPECT_TRUE(runButton->text().contains(QStringLiteral("DOM")));

    bool emitted = false;
    QStringList emittedImages;
    QString emittedDem;
    QString emittedOutput;
    double emittedResolution = -1.0;
    QObject::connect(&dialog, &MapProjectDialog::requestRunMapProject, &dialog,
                     [&](const QStringList &runImages,
                         const QString &runDem,
                         const QString &runOutput,
                         double runResolution)
                     {
                         emitted = true;
                         emittedImages = runImages;
                         emittedDem = runDem;
                         emittedOutput = runOutput;
                         emittedResolution = runResolution;
                     });
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "onRun", Qt::DirectConnection));

    EXPECT_TRUE(emitted);
    EXPECT_EQ(emittedImages, images);
    EXPECT_EQ(emittedDem, demPath);
    EXPECT_EQ(emittedOutput, expectedOutput);
    EXPECT_DOUBLE_EQ(emittedResolution, 0.0);
}

TEST(MapProjectDialogTest, KeepsLatestDemDefaultWhenSavedDemIsMissing)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectRoot = tempDir.path();
    const QString latestDemPath = QDir(projectRoot).filePath(QStringLiteral("assets/dem/latest/dem.tif"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(latestDemPath).absolutePath()));
    QFile latestDemFile(latestDemPath);
    ASSERT_TRUE(latestDemFile.open(QIODevice::WriteOnly));
    latestDemFile.write("dem");
    latestDemFile.close();

    MapProjectDialog dialog;
    dialog.setProjectRoot(projectRoot);
    dialog.setDefaultDemPath(latestDemPath);

    auto *demEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_demEdit"));
    ASSERT_NE(demEdit, nullptr);
    ASSERT_EQ(demEdit->text(), latestDemPath);

    QJsonObject savedSettings;
    savedSettings[QStringLiteral("dem_path")] =
        QDir(projectRoot).filePath(QStringLiteral("assets/dem/old/missing_dem.tif"));
    savedSettings[QStringLiteral("output_path")] =
        QDir(projectRoot).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
    dialog.applySettings(savedSettings);

    EXPECT_EQ(demEdit->text(), latestDemPath);
}

TEST(MapProjectDialogTest, ValidatesDemFileExistsBeforeRunning)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MapProjectDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QFileInfo demInfo")));
    EXPECT_TRUE(source.contains(QStringLiteral("demInfo.exists()")));
    EXPECT_TRUE(source.contains(QStringLiteral("不是有效的 DEM 文件")));
}

TEST(CreateDemDialogTest, UiAdvertisesOneClickDemWorkflow)
{
    const QString ui = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CreateDemDialog.ui"));
    ASSERT_FALSE(ui.isEmpty());

    EXPECT_TRUE(ui.contains(QStringLiteral("自动模式")));
    EXPECT_TRUE(ui.contains(QStringLiteral("手动模式")));
    EXPECT_TRUE(ui.contains(QStringLiteral("选择 2 张立体影像")));
    EXPECT_TRUE(ui.contains(QStringLiteral("已有密集点云")));
    EXPECT_TRUE(ui.contains(QStringLiteral("m_stageLabel")));
    EXPECT_TRUE(ui.contains(QStringLiteral("m_progressBar")));
    EXPECT_TRUE(ui.contains(QStringLiteral("m_runBtn")));
    EXPECT_TRUE(ui.contains(QStringLiteral("一键生成 DEM")));
}

TEST(CreateDemDialogTest, OpenCreateDemDialogRequiresProjectManagerBeforeShowingManualRunDialog)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int openIndex = source.indexOf(QStringLiteral("void MenuWorkflowController::openCreateDemDialog()"));
    const int newDialogIndex = source.indexOf(QStringLiteral("new CreateDemDialog"), openIndex);
    const int warningIndex = source.indexOf(QStringLiteral("请先打开项目"), openIndex);
    ASSERT_GE(openIndex, 0);
    ASSERT_GE(newDialogIndex, 0);
    ASSERT_GE(warningIndex, 0);
    EXPECT_LT(warningIndex, newDialogIndex);
}

TEST(SuperPointRunnerTest, DiskAndAlikedUseNativeTorchscriptExtractor)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/tasks/SuperPointRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("ExtractorFactory.h")));
    EXPECT_TRUE(source.contains(QStringLiteral("createExtractor(")));
    EXPECT_TRUE(source.contains(QStringLiteral("featureAlgorithm.toStdString()")));
    EXPECT_TRUE(source.contains(QStringLiteral("C++ TorchScript")));
    EXPECT_FALSE(source.contains(QStringLiteral("runPythonExtractor")));
}

TEST(SuperPointRunnerTest, FeatureExtractionLogUsesSelectedAlgorithmName)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_FALSE(source.contains(QStringLiteral("开始在后台线程执行 SuperPoint...")));
    EXPECT_TRUE(source.contains(QStringLiteral("开始在后台线程执行 %1 特征提取")));
}

TEST(MainWindowFeatureRefreshTest, BatchFeatureAppendDoesNotSynchronouslyReloadNonCurrentImages)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int connectIndex = source.indexOf(QStringLiteral("ipfindResultAppended"));
    ASSERT_GE(connectIndex, 0);
    const int blockEnd = source.indexOf(QStringLiteral("if (m_config)"), connectIndex);
    ASSERT_GE(blockEnd, connectIndex);
    const QString block = source.mid(connectIndex, blockEnd - connectIndex);

    EXPECT_TRUE(block.contains(QStringLiteral("currentImagePath()")));
    EXPECT_TRUE(block.contains(QStringLiteral("isCurrentImage")));
    EXPECT_TRUE(block.contains(QStringLiteral("reloadInterestPoints(imagePath)")));
    EXPECT_FALSE(block.contains(QStringLiteral("immediateReloadInterestPoints(imagePath)")));
}

TEST(ProjectTriangulationServiceTest, ExportsInitialSparseCloud)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString matchesDir = ProjectIO::ipmatchOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(matchesDir));

    const QString image0Path = QDir(tempDir.path()).filePath(QStringLiteral("1.jpg"));
    const QString image1Path = QDir(tempDir.path()).filePath(QStringLiteral("2.jpg"));

    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);

    const std::vector<std::array<double, 3>> points = {
        {3.0, -1.0, 35.0},
        {4.0,  0.5, 40.0},
        {5.0,  1.2, 45.0}
    };

    QJsonArray matchedPoints0;
    QJsonArray matchedPoints1;
    for (const std::array<double, 3> &point : points)
    {
        double u0 = 0.0;
        double v0 = 0.0;
        double u1 = 0.0;
        double v1 = 0.0;
        ASSERT_TRUE(projectPoint(camera0, point, &u0, &v0));
        ASSERT_TRUE(projectPoint(camera1, point, &u1, &v1));

        matchedPoints0.append(QJsonArray{u0, v0});
        matchedPoints1.append(QJsonArray{u1, v1});
    }

    const QString matchPath = QDir(matchesDir).filePath(QStringLiteral("1__2.match"));
    QFile matchFile(matchPath);
    ASSERT_TRUE(matchFile.open(QIODevice::WriteOnly));
    matchFile.write("dummy");
    matchFile.close();

    QFile sidecarFile(matchPath + QStringLiteral(".json"));
    ASSERT_TRUE(sidecarFile.open(QIODevice::WriteOnly));
    const QJsonObject sidecarObject{
        {QStringLiteral("image0_path"), image0Path},
        {QStringLiteral("image1_path"), image1Path},
        {QStringLiteral("matched_points0"), matchedPoints0},
        {QStringLiteral("matched_points1"), matchedPoints1}
    };
    sidecarFile.write(QJsonDocument(sidecarObject).toJson());
    sidecarFile.close();

    QJsonArray images;
    images.append(buildImageEntry(image0Path, camera0));
    images.append(buildImageEntry(image1Path, camera1));

    QJsonArray ipmatchResults;
    ipmatchResults.append(QJsonObject{{QStringLiteral("image0"), image0Path},
                                      {QStringLiteral("image1"), image1Path},
                                      {QStringLiteral("output"), matchPath}});

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    meta[QStringLiteral("ipmatch_results")] = ipmatchResults;

    xjw::gui::project::TriangulationServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;
    options.minObservations = 2;
    options.ignoreTwoViewTracks = false;
    options.minTrackLength = 2;

    const auto result = xjw::gui::project::ProjectTriangulationService::run(
        meta, QStringList{image0Path, image1Path}, options);

    EXPECT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_GT(result.exportedPointCount, 0);
    EXPECT_TRUE(QFileInfo::exists(result.sparseCloudPath));
}

TEST(ModelDropSupportTest, AcceptsStandaloneModelAndPointCloudFiles)
{
    const QUrl projectUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/example.plascan"));
    const QUrl plyUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/model.ply"));
    const QUrl xyzUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/cloud.xyz"));
    const QUrl imageUrl = QUrl::fromLocalFile(QStringLiteral("/tmp/image.png"));

    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/model.ply")));
    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/model.obj")));
    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/cloud.xyz")));
    EXPECT_TRUE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/cloud.txt")));
    EXPECT_FALSE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/project.plascan")));
    EXPECT_FALSE(xjw::gui::main_window::isStandaloneModelFile(QStringLiteral("/tmp/image.png")));

    EXPECT_EQ(xjw::gui::main_window::firstStandaloneModelFile({projectUrl, imageUrl, plyUrl, xyzUrl}),
              QStringLiteral("/tmp/model.ply"));
    EXPECT_TRUE(xjw::gui::main_window::firstStandaloneModelFile({imageUrl}).isEmpty());
}

TEST(DataTreeWidgetTest, ShowsTemporaryDroppedModelUntilCleared)
{
    DataTreeWidget tree;
    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray();
    tree.loadFromJson(meta);

    const QString modelPath = QStringLiteral("/tmp/temporary_model.ply");
    tree.addTransientModel(modelPath);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *modelSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("3D模型 (1)")))
        {
            modelSection = item;
            break;
        }
    }
    ASSERT_NE(modelSection, nullptr);
    ASSERT_EQ(modelSection->rowCount(), 1);
    EXPECT_EQ(modelSection->child(0, 0)->text(), QStringLiteral("temporary_model.ply  [临时]"));
    EXPECT_EQ(modelSection->child(0, 1)->text(), modelPath);

    tree.clearTransientResources();

    bool foundClearedModelSection = false;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("3D模型 (0)")))
        {
            foundClearedModelSection = true;
            EXPECT_EQ(item->rowCount(), 0);
            break;
        }
    }
    EXPECT_TRUE(foundClearedModelSection);
}

TEST(DataTreeWidgetTest, ContextMenuUsesSameResourcePathResolutionAsActivation)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QString DataTreeWidget::resolveResourcePath")));
    EXPECT_TRUE(source.contains(QStringLiteral("path = resolveResourcePath(path)")));
    EXPECT_TRUE(source.contains(QStringLiteral("paths << resolveResourcePath")));
}

TEST(CameraModel3DDialogTest, PlyFloatIntensityIsScaledToByteRange)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString plyPath = QDir(tempDir.path()).filePath(QStringLiteral("float_intensity.ply"));
    QFile file(plyPath);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
    QTextStream stream(&file);
    stream << "ply\n"
           << "format ascii 1.0\n"
           << "element vertex 1\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property float intensity\n"
           << "end_header\n"
           << "1 2 3 0.5\n";
    file.close();

    auto cloud = plapoint::io::readPly<float>(plyPath.toStdString());
    ASSERT_TRUE(cloud != nullptr);
    ASSERT_EQ(cloud->size(), 1u);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->getValue(0, 0), 128);
    EXPECT_EQ(cloud->colors()->getValue(0, 1), 128);
    EXPECT_EQ(cloud->colors()->getValue(0, 2), 128);
}

TEST(CameraModel3DDialogTest, DenseCameraScenesThrottleLabelsAndFrustumSize)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("maxVisibleCameraLabels")));
    EXPECT_TRUE(source.contains(QStringLiteral("m_poses.size() <= maxVisibleCameraLabels")));
    EXPECT_TRUE(source.contains(QStringLiteral("cameraFrustumBase()")));
    EXPECT_FALSE(source.contains(QStringLiteral("const float base = qMax(0.1f, r * 0.06f);")));
}

TEST(CameraModel3DDialogTest, UsesCameraToWorldRotationWithoutTransposeForFrustums)
{
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const QString workspaceSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    ASSERT_FALSE(dialogSource.isEmpty());
    ASSERT_FALSE(workspaceSource.isEmpty());

    const QString combined = dialogSource + workspaceSource;
    EXPECT_FALSE(combined.contains(QStringLiteral("pose.rotation = rot.transposed();")));
    EXPECT_TRUE(combined.contains(QStringLiteral("pose.rotation = rot;")));
}

TEST(DenseCloudRefineTest, ReportsPlaPointProcessingDeviceForGui)
{
    const QString guiSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(guiSource.isEmpty());

    EXPECT_TRUE(guiSource.contains(QStringLiteral("plapoint::ProcessingReport")));
    EXPECT_TRUE(guiSource.contains(QStringLiteral("processingDeviceLabel")));
    EXPECT_TRUE(guiSource.contains(QStringLiteral("usedDevice")));
}

TEST(DenseCloudRefineTest, NormalEstimationUsesRequestedPlaPointDevice)
{
    const QString guiSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(guiSource.isEmpty());

    EXPECT_TRUE(guiSource.contains(QStringLiteral("estimateNormals(cloud, request.normalK, request.processingDevice")));
    EXPECT_FALSE(guiSource.contains(QStringLiteral("NormalEstimation<float, plamatrix::Device::CPU> ne")));
}

TEST(SurfaceReconstructorTest, HeightGridFallbackUsesPlaPointGpuCapablePath)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mesh/SurfaceReconstructorHeightGrid.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("#include <plapoint/gpu/height_grid.h>")));
    EXPECT_TRUE(source.contains(QStringLiteral("buildHeightGridWithRequestedDevice")));
    EXPECT_TRUE(source.contains(QStringLiteral("config.preprocessingDevice")));
    EXPECT_TRUE(source.contains(QStringLiteral("plapoint::gpu::buildHeightGrid")));
}

TEST(ReconstructPipelineCliTest, LargeDenseCloudIsVoxelThinnedBeforeSor)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int preVoxelIndex = source.indexOf(QStringLiteral("大点云预降采样"));
    const int refineIndex = source.indexOf(QStringLiteral("refineDenseCloud(std::move(refineInput)"));
    ASSERT_GE(preVoxelIndex, 0);
    ASSERT_GE(refineIndex, 0);
    EXPECT_LT(preVoxelIndex, refineIndex);
}

TEST(ReconstructPipelineCliTest, LargeDenseCloudIsPreAggregatedBeforePlaPointRefinement)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("voxelDownsampleFusedPoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("开始大点云预降采样")));
    EXPECT_TRUE(source.contains(QStringLiteral("完成大点云预降采样")));

    const int preAggregateIndex = source.indexOf(QStringLiteral("voxelDownsampleFusedPointsToTarget"));
    const int preAggregateArgIndex = source.indexOf(QStringLiteral("fusedCloud"), preAggregateIndex);
    const int plaCloudIndex = source.indexOf(QStringLiteral("fusedPointsToPointCloud(fusedCloud"));
    ASSERT_GE(preAggregateIndex, 0);
    ASSERT_GE(preAggregateArgIndex, 0);
    ASSERT_GE(plaCloudIndex, 0);
    EXPECT_LT(preAggregateIndex, plaCloudIndex);
    EXPECT_LT(preAggregateArgIndex, plaCloudIndex);
}

TEST(ReconstructPipelineCliTest, LargeDenseCloudPreAggregationCapsPlaPointRefineInput)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("kMaxRefineInputPoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("voxelDownsampleFusedPointsToTarget")));
    EXPECT_TRUE(source.contains(QStringLiteral("targetPoints=%zu")));
}

TEST(ReconstructPipelineCliTest, LongRunningCliProgressIsFlushedAndThrottled)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("lastMeshProgressPercent")));
    EXPECT_TRUE(source.contains(QStringLiteral("lastMeshProgressStage")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::fflush(stdout);")));
}

TEST(ReconstructPipelineCliTest, FinalSummaryReportsElapsedTimings)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("timings")));
    EXPECT_TRUE(source.contains(QStringLiteral("total_elapsed_ms")));
    EXPECT_TRUE(source.contains(QStringLiteral("elapsed_total=%.3fs")));
    EXPECT_TRUE(source.contains(QStringLiteral("elapsed_sfm=%.3fs")));
    EXPECT_TRUE(source.contains(QStringLiteral("elapsed_mvs=%.3fs")));
}

TEST(MainMenuTest, SparseReconstructionMenuPlacesVocabularyOverlapBetweenFeatureSteps)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int detectIndex = source.indexOf(QStringLiteral("m_detectFeaturesAct"));
    const int overlapIndex = source.indexOf(QStringLiteral("m_vocabularyOverlapAct"));
    const int matchIndex = source.indexOf(QStringLiteral("m_matchFeaturesAct"));

    ASSERT_GE(detectIndex, 0);
    ASSERT_GE(overlapIndex, 0);
    ASSERT_GE(matchIndex, 0);
    EXPECT_LT(detectIndex, overlapIndex);
    EXPECT_LT(overlapIndex, matchIndex);
}

TEST(VocabularyOverlapDialogTest, UiDefinesRequiredControls)
{
    const QString uiSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/VocabularyOverlapDialog.ui"));
    ASSERT_FALSE(uiSource.isEmpty());

    const QStringList requiredControls = {
        QStringLiteral("m_imageList"),
        QStringLiteral("m_overlapMethodCombo"),
        QStringLiteral("m_referenceBodyCombo"),
        QStringLiteral("m_autoReferenceElevationCheck"),
        QStringLiteral("m_referenceElevationSpin"),
        QStringLiteral("m_featureAlgorithmCombo"),
        QStringLiteral("m_branchFactorSpin"),
        QStringLiteral("m_treeDepthSpin"),
        QStringLiteral("m_topKSpin"),
        QStringLiteral("m_minSimilaritySpin"),
        QStringLiteral("m_overlapThreadsSpin"),
        QStringLiteral("m_useFlannAssignmentCheck"),
        QStringLiteral("m_useInvertedIndexCheck"),
        QStringLiteral("m_useCudaOverlapCheck"),
        QStringLiteral("m_enableGeometryCheck"),
        QStringLiteral("m_pairTable"),
        QStringLiteral("m_applyToMatchingCheck"),
        QStringLiteral("m_runBtn")
    };

    for (const QString &controlName : requiredControls)
    {
        EXPECT_TRUE(uiSource.contains(controlName)) << controlName.toStdString();
    }
}

TEST(VocabularyOverlapDialogTest, DialogKeyIsAvailableForPersistence)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/config/settings/DialogSettingKeys.h"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("VocabularyOverlap")));
    EXPECT_TRUE(source.contains(QStringLiteral("vocabulary_overlap")));
}

TEST(VocabularyOverlapDialogTest, RetrievalRunsAsynchronously)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/VocabularyOverlapDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/VocabularyOverlapDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QFutureWatcher")));
    EXPECT_TRUE(header.contains(QStringLiteral("overlapProgressChanged")));
    EXPECT_TRUE(header.contains(QStringLiteral("cancelRun")));
    EXPECT_TRUE(source.contains(QStringLiteral("QtConcurrent::run")));
    EXPECT_TRUE(source.contains(QStringLiteral("setUiBusy")));

    const int onRunIndex = source.indexOf(QStringLiteral("void VocabularyOverlapDialog::onRun()"));
    const int onExportIndex = source.indexOf(QStringLiteral("void VocabularyOverlapDialog::onExportLis()"));
    ASSERT_GE(onRunIndex, 0);
    ASSERT_GT(onExportIndex, onRunIndex);
    const QString onRunBody = source.mid(onRunIndex, onExportIndex - onRunIndex);
    EXPECT_FALSE(onRunBody.contains(QStringLiteral("VocabularyOverlapRetriever::retrieve")));
}

TEST(VocabularyOverlapDialogTest, SupportsCameraModelModeAndStatusProgress)
{
    const QString dialogHeader = readProjectSourceFile(QStringLiteral("src/gui/dialogs/VocabularyOverlapDialog.h"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/VocabularyOverlapDialog.cpp"));
    const QString mainHeader = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString mainSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString menuSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(dialogHeader.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());
    ASSERT_FALSE(mainHeader.isEmpty());
    ASSERT_FALSE(mainSource.isEmpty());
    ASSERT_FALSE(menuSource.isEmpty());

    EXPECT_TRUE(dialogSource.contains(QStringLiteral("OverlapAnalyzer::analyze")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("imageCameraFromEntry")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("m_overlapMethodCombo")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("ReferenceBody::Earth")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("ReferenceBody::Moon")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("ReferenceBody::Mars")));
    EXPECT_TRUE(mainHeader.contains(QStringLiteral("m_overlapTaskStatus")));
    EXPECT_TRUE(mainHeader.contains(QStringLiteral("overlapCancelRequested")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("onOverlapProgress")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("onOverlapFinished")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("overlapProgressChanged")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("overlapCancelRequested")));
}

TEST(VocabularyOverlapDialogTest, DisplaysResultsWithoutOverwritingSummary)
{
    const QString dialogHeader = readProjectSourceFile(QStringLiteral("src/gui/dialogs/VocabularyOverlapDialog.h"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/VocabularyOverlapDialog.cpp"));
    ASSERT_FALSE(dialogHeader.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());

    EXPECT_TRUE(dialogHeader.contains(QStringLiteral("updateMethodUi(bool refreshSummary = true)")));

    const int setUiBusyIndex = dialogSource.indexOf(QStringLiteral("void VocabularyOverlapDialog::setUiBusy"));
    const int handleProgressIndex = dialogSource.indexOf(QStringLiteral("void VocabularyOverlapDialog::handleProgress"));
    ASSERT_GE(setUiBusyIndex, 0);
    ASSERT_GT(handleProgressIndex, setUiBusyIndex);
    const QString setUiBusyBody = dialogSource.mid(setUiBusyIndex, handleProgressIndex - setUiBusyIndex);
    EXPECT_TRUE(setUiBusyBody.contains(QStringLiteral("updateMethodUi(false)")));

    const int finishIndex = dialogSource.indexOf(QStringLiteral("void VocabularyOverlapDialog::handleRunFinished"));
    const int writeOutputsIndex = dialogSource.indexOf(QStringLiteral("bool VocabularyOverlapDialog::writeOutputs"));
    ASSERT_GE(finishIndex, 0);
    ASSERT_GT(writeOutputsIndex, finishIndex);
    const QString finishBody = dialogSource.mid(finishIndex, writeOutputsIndex - finishIndex);
    EXPECT_TRUE(finishBody.contains(QStringLiteral("populatePairTable();")));
    EXPECT_TRUE(finishBody.contains(QStringLiteral("m_summaryLabel->setText(QStringLiteral(\"候选 %1，对外输出 %2，词汇数 %3\")")));
    EXPECT_TRUE(finishBody.contains(QStringLiteral("setUiBusy(false);")));

    const int tableIndex = dialogSource.indexOf(QStringLiteral("void VocabularyOverlapDialog::populatePairTable"));
    const int methodUiIndex = dialogSource.indexOf(QStringLiteral("void VocabularyOverlapDialog::updateMethodUi"));
    ASSERT_GE(tableIndex, 0);
    ASSERT_GT(methodUiIndex, tableIndex);
    const QString tableBody = dialogSource.mid(tableIndex, methodUiIndex - tableIndex);
    EXPECT_TRUE(tableBody.contains(QStringLiteral("m_pairTable->setRowCount")));
    EXPECT_TRUE(tableBody.contains(QStringLiteral("m_pairTable->setVisible(true)")));
    EXPECT_TRUE(tableBody.contains(QStringLiteral("m_pairTable->viewport()->update()")));
}

TEST(MenuWorkflowControllerTest, VocabularyOverlapAppliesGeneratedPairsToFeatureMatchingSettings)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("VocabularyOverlapDialog")));
    EXPECT_TRUE(source.contains(QStringLiteral("DialogSettingKeys::VocabularyOverlap")));
    EXPECT_TRUE(source.contains(QStringLiteral("DialogSettingKeys::SuperGlue")));
    EXPECT_TRUE(source.contains(QStringLiteral("generated_pairs")));
}

TEST(MainMenuTest, WorkflowMenuExposesOnlyOneClickThreeDReconstruction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QMenu *workflowMenu = nullptr;
    for (QAction *action : window.menuBar()->actions())
    {
        if (action && action->text() == QStringLiteral("工作流程"))
        {
            workflowMenu = action->menu();
            break;
        }
    }
    ASSERT_NE(workflowMenu, nullptr);

    QStringList visibleActions;
    for (QAction *action : workflowMenu->actions())
    {
        if (action && !action->isSeparator())
        {
            visibleActions.append(action->text());
        }
    }

    EXPECT_TRUE(visibleActions.contains(QStringLiteral("三维重建")));
    EXPECT_FALSE(visibleActions.contains(QStringLiteral("空中三角测量")));
    EXPECT_FALSE(visibleActions.contains(QStringLiteral("创建密集点云")));
    EXPECT_FALSE(visibleActions.contains(QStringLiteral("生成模型")));
    ASSERT_NE(menu.threeDReconstructionAction(), nullptr);
    EXPECT_EQ(menu.threeDReconstructionAction()->text(), QStringLiteral("三维重建"));
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
