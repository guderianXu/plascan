// ============================================================
// test_gui_project_utils.cpp — GUI project 层公共工具与三角化服务测试
// ============================================================

#include <gtest/gtest.h>

#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "ProjectTriangulationService.h"
#include "FeatureExtractionDialog.h"
#include "ThreeDReconstructionDialog.h"
#include "MapProjectDialog.h"
#include "ModelDropSupport.h"
#include "DataTreeWidget.h"
#include "MainMenu.h"

#include "Camera.h"

#include <plapoint/io/ply_io.h>

#include <QApplication>
#include <QDir>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFile>
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
    EXPECT_TRUE(modelPathEdit->text().endsWith(QStringLiteral(".pt")));
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

TEST(ThreeDReconstructionDialogTest, UsesUiDefaultsAndImageCountGate)
{
    ThreeDReconstructionDialog dialog;

    auto *outputEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_outputDirEdit"));
    auto *startButton = dialog.findChild<QPushButton *>(QStringLiteral("m_startBtn"));
    auto *generateDemCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_generateDemCheck"));
    auto *generateDomCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_generateDomCheck"));
    auto *demResolutionSpin = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("m_demResolutionSpin"));
    ASSERT_NE(outputEdit, nullptr);
    ASSERT_NE(startButton, nullptr);
    EXPECT_EQ(generateDemCheck, nullptr);
    EXPECT_EQ(generateDomCheck, nullptr);
    EXPECT_EQ(demResolutionSpin, nullptr);

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
    EXPECT_FALSE(settings.value(QStringLiteral("export_obj")).toBool());
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

TEST(MenuWorkflowControllerTest, DenseStageAdvancesOnMvsSuccessWithoutRequiringChangedOutputPath)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_FALSE(source.contains(QStringLiteral("densePath == beforePath")));
    EXPECT_TRUE(source.contains(QStringLiteral("mvsProgressFinished")));
    EXPECT_TRUE(source.contains(QStringLiteral("startThreeDReconstructionMeshStage(settings)")));
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

TEST(SuperPointRunnerTest, PythonExtractorSupportsConfigEnvVenvAndDiagnosticLogging)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/tasks/SuperPointRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("python_executable")));
    EXPECT_TRUE(source.contains(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE")));
    EXPECT_TRUE(source.contains(QStringLiteral("VIRTUAL_ENV")));
    EXPECT_TRUE(source.contains(QStringLiteral("默认 plascan Python 环境")));
    EXPECT_TRUE(source.contains(QStringLiteral(".local/share/mamba/envs/plascan")));
    EXPECT_TRUE(source.contains(QStringLiteral("Python 可执行文件")));
    EXPECT_TRUE(source.contains(QStringLiteral("脚本路径")));
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
