// ============================================================
// test_gui_project_utils.cpp — GUI project 层公共工具与三角化服务测试
// ============================================================

#include <gtest/gtest.h>

#include "ProjectIO.h"
#include "ProjectCameraImportService.h"
#include "ProjectData.h"
#include "ProjectDepthFrameUtils.h"
#include "ProjectSupportUtils.h"
#include "ProjectTriangulationService.h"
#include "ProjectWorkflowReports.h"
#include "ProjectWorkflowUtils.h"
#include "project/SparseResultQuality.h"
#include "FeatureExtractionDialog.h"
#include "FeatureMatchingDialog.h"
#include "InitCameraPoseDialog.h"
#include "ThreeDReconstructionDialog.h"
#include "SparseCloudPostProcessDialog.h"
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
#include <QDialogButtonBox>
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
#include <QSignalSpy>

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

QMenu *findTopLevelMenuByTitle(QMenuBar *menuBar, const QString &title)
{
    if (!menuBar)
    {
        return nullptr;
    }

    for (QAction *action : menuBar->actions())
    {
        if (action && action->menu() && action->menu()->title() == title)
        {
            return action->menu();
        }
    }

    return nullptr;
}

QMenu *findSubMenuByTitle(QMenu *menu, const QString &title)
{
    if (!menu)
    {
        return nullptr;
    }

    for (QAction *action : menu->actions())
    {
        if (action && action->menu() && action->menu()->title() == title)
        {
            return action->menu();
        }
    }

    return nullptr;
}

QStringList directActionTexts(QMenu *menu)
{
    QStringList texts;
    if (!menu)
    {
        return texts;
    }

    for (QAction *action : menu->actions())
    {
        if (action && !action->isSeparator() && !action->menu())
        {
            texts.push_back(action->text());
        }
    }

    return texts;
}

void writeMinimalTsai(const QString &path, double fu, double cx)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out(&file);
    out << "VERSION_3\n";
    out << "PINHOLE\n";
    out << "TSAI\n";
    out << "fu = " << QString::number(fu, 'f', 8) << "\n";
    out << "fv = " << QString::number(fu, 'f', 8) << "\n";
    out << "cu = " << QString::number(cx, 'f', 8) << "\n";
    out << "cv = 500\n";
    out << "u_direction = 1 0 0\n";
    out << "v_direction = 0 1 0\n";
    out << "w_direction = 0 0 1\n";
    out << "pitch = 1\n";
    out << "k1 = -0.01\n";
    out << "k2 = 0.02\n";
    out << "k3 = -0.03\n";
    out << "p1 = 0.0001\n";
    out << "p2 = -0.0002\n";
    out << "C = 1 2 3\n";
    out << "R = 1 0 0 0 1 0 0 0 1\n";
}

void writeMinimalPointCloudPly(const QString &path,
                               const std::vector<std::array<double, 3>> &points)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out(&file);
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "end_header\n";
    for (const auto &point : points)
    {
        out << QString::number(point[0], 'f', 6) << ' '
            << QString::number(point[1], 'f', 6) << ' '
            << QString::number(point[2], 'f', 6) << '\n';
    }
}

void writeMinimalColoredPointCloudPly(const QString &path,
                                      const std::vector<std::array<double, 3>> &points,
                                      const std::vector<std::array<int, 3>> &colors)
{
    ASSERT_EQ(points.size(), colors.size());
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out(&file);
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto &point = points[i];
        const auto &color = colors[i];
        out << QString::number(point[0], 'f', 6) << ' '
            << QString::number(point[1], 'f', 6) << ' '
            << QString::number(point[2], 'f', 6) << ' '
            << color[0] << ' '
            << color[1] << ' '
            << color[2] << '\n';
    }
}

void writeMinimalSparseSidecar(const QString &path,
                               const std::vector<int> &trackLens)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));

    QJsonArray points;
    for (int i = 0; i < static_cast<int>(trackLens.size()); ++i)
    {
        QJsonArray xyz;
        xyz.append(static_cast<double>(i));
        xyz.append(0.0);
        xyz.append(0.0);

        QJsonObject point;
        point[QStringLiteral("point_xyz")] = xyz;
        point[QStringLiteral("rms_reproj_px")] = 0.5;
        point[QStringLiteral("min_tri_angle_deg")] = 3.0;
        point[QStringLiteral("track_len")] = trackLens[static_cast<std::size_t>(i)];
        points.append(point);
    }

    QJsonObject root;
    root[QStringLiteral("quality_metrics_available")] = true;
    root[QStringLiteral("point_count")] = static_cast<int>(trackLens.size());
    root[QStringLiteral("points")] = points;

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(QJsonDocument(root).toJson());
    file.close();
}

TEST(ProjectCameraImportServiceTest, BatchImportRecordsActualTsaiSourceFileInCameraMeta)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("IMG_001.JPG"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("jpg");
    imageFile.close();

    const QString tsaiPath = QDir(tempDir.path()).filePath(QStringLiteral("cameras/IMG_001.tsai"));
    writeMinimalTsai(tsaiPath, 1234.0, 640.0);

    xjw::gui::project::BatchCameraImportResult result;
    const auto status = xjw::gui::project::buildBatchCameraImport(
        QFileInfo(tsaiPath).absolutePath(),
        QStringList{imagePath},
        &result);

    ASSERT_EQ(status, xjw::gui::project::BatchCameraImportStatus::Ok);
    ASSERT_EQ(result.cameraMetaByImage.size(), 1);
    const QJsonObject camera = result.cameraMetaByImage.constBegin().value();
    EXPECT_EQ(QDir::cleanPath(camera.value(QStringLiteral("source_file")).toString()),
              QDir::cleanPath(QFileInfo(tsaiPath).absoluteFilePath()));
    EXPECT_DOUBLE_EQ(camera.value(QStringLiteral("fu")).toDouble(), 1234.0);
    EXPECT_NEAR(camera.value(QStringLiteral("k1")).toDouble(), -0.01, 1e-12);
}

TEST(ProjectDataCameraMetadataTest, SetImageCamerasClearsLegacyTopLevelCameraFile)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("camera_meta.plascan"));
    const QString imagePath = QDir(tempDir.path()).filePath(QStringLiteral("IMG_002.JPG"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    imageFile.write("jpg");
    imageFile.close();

    ProjectData data;
    ASSERT_TRUE(data.createProject(projectPath, QStringLiteral("camera_meta")));
    ASSERT_TRUE(data.addImages(QStringList{imagePath}));

    QJsonObject core = data.coreFilesMeta();
    QJsonArray images = core.value(QStringLiteral("images")).toArray();
    ASSERT_EQ(images.size(), 1);
    QJsonObject imageObject = images.at(0).toObject();
    imageObject[QStringLiteral("camera_file")] = QStringLiteral("stale/old.tsai");
    images[0] = imageObject;
    core[QStringLiteral("images")] = images;
    data.updateMetadata(core, false);

    QJsonObject camera;
    camera[QStringLiteral("model")] = QStringLiteral("tsai");
    camera[QStringLiteral("source_file")] = QStringLiteral("fresh/new.tsai");
    camera[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    camera[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    camera[QStringLiteral("pitch")] = 1.0;
    camera[QStringLiteral("fu")] = 1111.0;
    camera[QStringLiteral("fv")] = 1111.0;
    camera[QStringLiteral("cu")] = 500.0;
    camera[QStringLiteral("cv")] = 400.0;
    camera[QStringLiteral("C")] = QJsonArray{0.0, 0.0, 0.0};
    camera[QStringLiteral("R")] = QJsonArray{1.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0,
                                             0.0, 0.0, 1.0};

    int updated = 0;
    QString error;
    ASSERT_TRUE(data.setImageCameras(QMap<QString, QJsonObject>{{imagePath, camera}}, &updated, &error))
        << error.toStdString();
    EXPECT_EQ(updated, 1);

    const QJsonObject updatedImage =
        data.coreFilesMeta().value(QStringLiteral("images")).toArray().at(0).toObject();
    EXPECT_FALSE(updatedImage.contains(QStringLiteral("camera_file")));
    EXPECT_EQ(updatedImage.value(QStringLiteral("camera")).toObject()
                  .value(QStringLiteral("source_file")).toString(),
              QStringLiteral("fresh/new.tsai"));
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

QJsonArray productionSparsePoints()
{
    return QJsonArray{
        QJsonObject{{QStringLiteral("track_len"), 2},
                    {QStringLiteral("rms_reproj_px"), 0.8}},
        QJsonObject{{QStringLiteral("track_len"), 3},
                    {QStringLiteral("rms_reproj_px"), 0.6}},
        QJsonObject{{QStringLiteral("track_len"), 4},
                    {QStringLiteral("rms_reproj_px"), 0.7}}
    };
}

QJsonObject sparseResultRecord(int index,
                               const QString &displayName,
                               const QString &operation,
                               const QString &operationDisplayName,
                               int sparsePointCount,
                               const QJsonObject &quality)
{
    QJsonObject record{
        {QStringLiteral("index"), index},
        {QStringLiteral("display_name"), displayName},
        {QStringLiteral("operation"), operation},
        {QStringLiteral("operation_display_name"), operationDisplayName},
        {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("E:/tmp/%1.xyz").arg(displayName)},
        {QStringLiteral("sparse_point_count"), sparsePointCount}
    };
    return xjw::gui::project::mergeSparseQualityIntoRecord(record, quality);
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

TEST(ProjectDepthFrameUtilsTest, ExistingFrameArtifactsRequirePreviewAndRawDepth)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString pngPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_0.png"));
    EXPECT_FALSE(xjw::gui::project::depthFrameArtifactsExist(pngPath));

    QFile pngFile(pngPath);
    ASSERT_TRUE(pngFile.open(QIODevice::WriteOnly));
    pngFile.write("png");
    pngFile.close();

    EXPECT_FALSE(xjw::gui::project::depthFrameArtifactsExist(pngPath));

    QFile rawFile(xjw::gui::project::rawDepthStoragePath(pngPath));
    ASSERT_TRUE(rawFile.open(QIODevice::WriteOnly));
    rawFile.write("raw");
    rawFile.close();

    EXPECT_TRUE(xjw::gui::project::depthFrameArtifactsExist(pngPath));
    EXPECT_FALSE(xjw::gui::project::depthFrameArtifactsExist(pngPath, true));

    QFile confidenceFile(xjw::gui::project::rawConfidenceStoragePath(pngPath));
    ASSERT_TRUE(confidenceFile.open(QIODevice::WriteOnly));
    confidenceFile.write("confidence");
    confidenceFile.close();

    EXPECT_TRUE(xjw::gui::project::depthFrameArtifactsExist(pngPath, true));
}

TEST(ProjectDepthFrameUtilsTest, FastBinaryDepthArtifactsKeepLegacyYamlCompatibility)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString pngPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_3.png"));
    EXPECT_TRUE(xjw::gui::project::rawDepthStoragePath(pngPath).endsWith(QStringLiteral(".bin")));
    EXPECT_TRUE(xjw::gui::project::rawConfidenceStoragePath(pngPath).endsWith(QStringLiteral("_conf.bin")));

    QFile pngFile(pngPath);
    ASSERT_TRUE(pngFile.open(QIODevice::WriteOnly));
    pngFile.write("png");
    pngFile.close();

    QFile legacyRawFile(QDir(tempDir.path()).filePath(QStringLiteral("depth_3.yml.gz")));
    ASSERT_TRUE(legacyRawFile.open(QIODevice::WriteOnly));
    legacyRawFile.write("legacy-depth");
    legacyRawFile.close();

    EXPECT_TRUE(xjw::gui::project::depthFrameArtifactsExist(pngPath));
    EXPECT_FALSE(xjw::gui::project::depthFrameArtifactsExist(pngPath, true));

    QFile legacyConfidenceFile(QDir(tempDir.path()).filePath(QStringLiteral("depth_3_conf.yml.gz")));
    ASSERT_TRUE(legacyConfidenceFile.open(QIODevice::WriteOnly));
    legacyConfidenceFile.write("legacy-confidence");
    legacyConfidenceFile.close();

    EXPECT_TRUE(xjw::gui::project::depthFrameArtifactsExist(pngPath, true));
}

TEST(ProjectDepthFrameUtilsTest, StoredDepthCollectionResolvesLegacyYamlFallback)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString pngPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_5.png"));
    QFile pngFile(pngPath);
    ASSERT_TRUE(pngFile.open(QIODevice::WriteOnly));
    pngFile.write("png");
    pngFile.close();

    const QString legacyRawPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_5.yml.gz"));
    QFile legacyRawFile(legacyRawPath);
    ASSERT_TRUE(legacyRawFile.open(QIODevice::WriteOnly));
    legacyRawFile.write("legacy-depth");
    legacyRawFile.close();

    QJsonObject record;
    record[QStringLiteral("ref_image")] = QStringLiteral("image_5.jpg");
    record[QStringLiteral("depth_png")] = pngPath;
    record[QStringLiteral("raw_depth_path")] = xjw::gui::project::rawDepthStoragePath(pngPath);

    QJsonObject meta;
    meta[QStringLiteral("depth_map_results")] = QJsonArray{record};

    const auto result = xjw::gui::project::collectLatestStoredDepthFrames(meta);
    ASSERT_TRUE(result.status.ok) << result.status.errorMessage.toStdString();
    ASSERT_EQ(result.frames.size(), 1u);
    EXPECT_EQ(result.frames.front().rawDepthPath, legacyRawPath);
}

TEST(DenseDepthReuseTest, ExistingDepthReuseRequiresRawDepthArtifact)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int collectStart = source.indexOf(QStringLiteral("QSet<int> collectExistingDepthFrameIndices"));
    ASSERT_GE(collectStart, 0);
    const int collectEnd = source.indexOf(QStringLiteral("ExistingDepthAction askExistingDepthAction"), collectStart);
    ASSERT_GT(collectEnd, collectStart);
    const QString collectBlock = source.mid(collectStart, collectEnd - collectStart);
    EXPECT_TRUE(collectBlock.contains(QStringLiteral("depthFrameArtifactsExist(pngPath)")))
        << collectBlock.toStdString();

    const int upsertStart = source.indexOf(QStringLiteral("void upsertExistingDepthRecords"));
    ASSERT_GE(upsertStart, 0);
    const int upsertEnd = source.indexOf(QStringLiteral("} // namespace"), upsertStart);
    ASSERT_GT(upsertEnd, upsertStart);
    const QString upsertBlock = source.mid(upsertStart, upsertEnd - upsertStart);
    EXPECT_TRUE(upsertBlock.contains(QStringLiteral("depthFrameArtifactsExist(pngPath)")))
        << upsertBlock.toStdString();
}

TEST(DenseDepthCameraLookupTest, MvsCameraLookupUsesNormalizedImageKeys)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("cameraForImagePath(camMap, imgPath")))
        << "Dense estimation should query cameras with the same normalized key used by ProjectManager.";
    EXPECT_TRUE(source.contains(QStringLiteral("cameraForImagePath(m_camMap, m_records[index].refImage")))
        << "Stored-depth fusion cache should normalize ref_image before camera lookup.";
    EXPECT_FALSE(source.contains(QStringLiteral("camMap.value(imgPath)")));
    EXPECT_FALSE(source.contains(QStringLiteral("camMap.value(stored.refImage)")));
}

TEST(DepthMapMetadataTest, DemPreviewsStayOutOfMvsDepthResults)
{
    const QString terrainSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    const QString modelSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString treeSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    ASSERT_FALSE(terrainSource.isEmpty());
    ASSERT_FALSE(modelSource.isEmpty());
    ASSERT_FALSE(treeSource.isEmpty());

    EXPECT_FALSE(terrainSource.contains(QStringLiteral("depth_map_results")))
        << "DEM product previews should be stored on dem_results.depth_preview_png, not as MVS depth maps.";
    EXPECT_FALSE(modelSource.contains(QStringLiteral("depth_map_results")))
        << "Model/terrain previews should not be inserted into the MVS depth map result list.";
    EXPECT_TRUE(treeSource.contains(QStringLiteral("depthResultKind(obj) == QStringLiteral(\"mvs_depth\")")))
        << "The depth-map tree should filter out non-MVS preview records.";
    EXPECT_TRUE(treeSource.contains(QStringLiteral("depth_preview_png")))
        << "DEM previews should be available from the DEM section instead.";
}

TEST(DepthMapPersistenceTest, SavesFrameArtifactsBeforeFinalConsistencyPass)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int workerSave = source.indexOf(
        QStringLiteral("saveQueue.enqueue(i, res, QStringLiteral(\"初始\"))"));
    const int waitBeforeConsistency = source.indexOf(QStringLiteral("saveQueue.waitUntilIdle()"));
    const int consistencyPass = source.indexOf(QStringLiteral("crossCheckDepthConsistency();"));
    const int finalSave = source.indexOf(
        QStringLiteral("saveQueue.enqueue(i, res, QStringLiteral(\"过滤后\"))"));

    ASSERT_GE(workerSave, 0);
    ASSERT_GE(waitBeforeConsistency, 0);
    ASSERT_GE(consistencyPass, 0);
    ASSERT_GE(finalSave, 0);
    EXPECT_LT(workerSave, consistencyPass)
        << "Each completed depth frame should be persisted before the all-frame consistency pass.";
    EXPECT_LT(waitBeforeConsistency, consistencyPass)
        << "Initial async saves must drain before the consistency pass mutates depth maps.";
    EXPECT_LT(consistencyPass, finalSave)
        << "The final consistency-filtered depth maps should still overwrite the provisional artifacts.";
}

TEST(DisparityHeatmapOverlayTest, InvalidPixelsUseAlphaMask)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QImage::Format_RGBA8888")))
        << "Heatmap storage should carry alpha so invalid disparity can be transparent.";
    EXPECT_TRUE(source.contains(QStringLiteral("alphaRow[col] = validRow[col] ? 255 : 0")))
        << "Invalid disparity pixels should be masked out, not colorized as low disparity.";
    EXPECT_TRUE(source.contains(QStringLiteral("if (m_showInvalid)")))
        << "setShowInvalid() must affect the rendered invalid-pixel state.";
    EXPECT_TRUE(header.contains(QStringLiteral("QImage heatmapImage() const")))
        << "Tests and callers need a mask-aware image accessor.";
}

TEST(SparseResultQualityTest, BuildsHistogramAndClassifiesPairwisePreview)
{
    QJsonArray points;
    points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                              {QStringLiteral("rms_reproj_px"), 1.0},
                              {QStringLiteral("min_tri_angle_deg"), 5.0}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                              {QStringLiteral("rms_reproj_px"), 2.0},
                              {QStringLiteral("min_tri_angle_deg"), 7.0}});

    const QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        2,
        false,
        xjw::gui::project::kSparseResultKindPairwisePreview);

    EXPECT_EQ(quality.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindPairwisePreview);
    EXPECT_EQ(quality.value(QStringLiteral("camera_count")).toInt(), 2);
    EXPECT_EQ(quality.value(QStringLiteral("point_count")).toInt(), 2);
    EXPECT_DOUBLE_EQ(quality.value(QStringLiteral("two_view_ratio")).toDouble(), 1.0);
    EXPECT_FALSE(quality.value(QStringLiteral("ba_applied")).toBool());

    const QJsonObject histogram = quality.value(QStringLiteral("track_len_histogram")).toObject();
    EXPECT_EQ(histogram.value(QStringLiteral("2")).toInt(), 2);
    EXPECT_TRUE(xjw::gui::project::isPairwisePreviewSparseResult(quality));
    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(quality));
}

TEST(SparseResultQualityTest, AcceptsFormalSfmWithMultiViewSupport)
{
    QJsonArray points;
    points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                              {QStringLiteral("rms_reproj_px"), 0.8}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 3},
                              {QStringLiteral("rms_reproj_px"), 0.6}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 4},
                              {QStringLiteral("rms_reproj_px"), 0.7}});

    const QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        3,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);

    EXPECT_TRUE(xjw::gui::project::isProductionSparseResult(quality));
    EXPECT_FALSE(xjw::gui::project::sparseResultBlockingReason(quality).contains(QStringLiteral("两视")));
    EXPECT_DOUBLE_EQ(quality.value(QStringLiteral("two_view_ratio")).toDouble(), 1.0 / 3.0);
    EXPECT_EQ(quality.value(QStringLiteral("median_track_len")).toInt(), 3);
}

TEST(SparseResultQualityTest, RejectsFormalSfmWhenAlmostAllTracksAreTwoView)
{
    QJsonArray points;
    for (int i = 0; i < 99; ++i)
    {
        points.append(QJsonObject{{QStringLiteral("track_len"), 2},
                                  {QStringLiteral("rms_reproj_px"), 0.8}});
    }
    points.append(QJsonObject{{QStringLiteral("track_len"), 3},
                              {QStringLiteral("rms_reproj_px"), 0.9}});

    const QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        60,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);

    EXPECT_GT(quality.value(QStringLiteral("two_view_ratio")).toDouble(), 0.95);
    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(quality));
    EXPECT_TRUE(xjw::gui::project::sparseResultBlockingReason(quality).contains(QStringLiteral("两视")));
}

TEST(SparseResultQualityTest, RejectsFormalSfmWhenTooFewSelectedImagesRegister)
{
    QJsonArray points;
    points.append(QJsonObject{{QStringLiteral("track_len"), 3},
                              {QStringLiteral("rms_reproj_px"), 0.6}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 4},
                              {QStringLiteral("rms_reproj_px"), 0.7}});
    points.append(QJsonObject{{QStringLiteral("track_len"), 5},
                              {QStringLiteral("rms_reproj_px"), 0.8}});

    QJsonObject quality = xjw::gui::project::buildSparseQualityMetadata(
        points,
        35,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);
    quality[QStringLiteral("input_image_count")] = 444;
    quality[QStringLiteral("registered_image_count")] = 35;

    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(quality));
    EXPECT_TRUE(xjw::gui::project::sparseResultBlockingReason(quality).contains(QStringLiteral("注册影像")));
}

TEST(SparseResultQualityTest, LegacyTriangulationRecordsAreShownAsPairwisePreview)
{
    const QJsonObject legacyRecord{
        {QStringLiteral("operation"), QStringLiteral("triangulation")},
        {QStringLiteral("source"), QStringLiteral("triangulation")}
    };

    EXPECT_TRUE(xjw::gui::project::isPairwisePreviewSparseResult(legacyRecord));
    EXPECT_EQ(xjw::gui::project::sparseOperationDisplayName(QStringLiteral("triangulation")),
              QStringLiteral("两视预览云"));
}

TEST(SparseCloudPostProcessDialogTest, ListsOnlyProductionSparseInputs)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/SparseCloudPostProcessDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(source.contains(QStringLiteral("isProductionSparseResult(record)")));
    EXPECT_TRUE(source.contains(QStringLiteral("continue;")));
    EXPECT_FALSE(source.contains(QStringLiteral("isPairwisePreviewSparseResult(record) &&")));
}

TEST(SparseCloudPostProcessDialogTest, FiltersPreviewRecordsAndKeepsSettingsAligned)
{
    const QJsonObject previewQuality = xjw::gui::project::buildSparseQualityMetadata(
        QJsonArray{
            QJsonObject{{QStringLiteral("track_len"), 2},
                        {QStringLiteral("rms_reproj_px"), 1.0}},
            QJsonObject{{QStringLiteral("track_len"), 2},
                        {QStringLiteral("rms_reproj_px"), 1.2}}
        },
        2,
        false,
        xjw::gui::project::kSparseResultKindPairwisePreview);
    const QJsonObject sfmQuality = xjw::gui::project::buildSparseQualityMetadata(
        productionSparsePoints(),
        3,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);
    const QJsonObject postprocessQuality = xjw::gui::project::buildSparseQualityMetadata(
        productionSparsePoints(),
        3,
        true,
        xjw::gui::project::kSparseResultKindSparsePostprocess,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction,
        QStringLiteral("sfm-11"));

    const QJsonObject previewRecord = sparseResultRecord(7,
                                                         QStringLiteral("preview"),
                                                         QStringLiteral("triangulation"),
                                                         QStringLiteral("两视预览云"),
                                                         120,
                                                         previewQuality);
    const QJsonObject sfmRecord = sparseResultRecord(11,
                                                     QStringLiteral("sfm"),
                                                     QStringLiteral("workflow_aerial_triangulation"),
                                                     QStringLiteral("空中三角测量"),
                                                     420,
                                                     sfmQuality);
    const QJsonObject postprocessRecord = sparseResultRecord(13,
                                                             QStringLiteral("postprocess"),
                                                             QStringLiteral("sparse_postprocess"),
                                                             QStringLiteral("稀疏云后处理"),
                                                             260,
                                                             postprocessQuality);

    ASSERT_FALSE(xjw::gui::project::isProductionSparseResult(previewRecord));
    ASSERT_TRUE(xjw::gui::project::isProductionSparseResult(sfmRecord));
    ASSERT_TRUE(xjw::gui::project::isProductionSparseResult(postprocessRecord));

    SparseCloudPostProcessDialog dialog;
    auto *sourceCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_sourceCombo"));
    auto *statsLabel = dialog.findChild<QLabel *>(QStringLiteral("m_statsLabel"));
    auto *buttonBox = dialog.findChild<QDialogButtonBox *>();
    ASSERT_NE(sourceCombo, nullptr);
    ASSERT_NE(statsLabel, nullptr);
    ASSERT_NE(buttonBox, nullptr);
    QPushButton *okButton = buttonBox->button(QDialogButtonBox::Ok);
    ASSERT_NE(okButton, nullptr);

    dialog.setAvailableSparseClouds(QJsonArray{previewRecord, sfmRecord, postprocessRecord});

    ASSERT_EQ(sourceCombo->count(), 2);
    EXPECT_EQ(sourceCombo->itemData(0).toInt(), 11);
    EXPECT_EQ(sourceCombo->itemData(1).toInt(), 13);
    for (int i = 0; i < sourceCombo->count(); ++i)
    {
        EXPECT_NE(sourceCombo->itemData(i).toInt(), 7);
    }
    EXPECT_TRUE(okButton->isEnabled());
    EXPECT_EQ(sourceCombo->currentIndex(), 0);
    EXPECT_TRUE(statsLabel->text().contains(QStringLiteral("420 个三维点")));
    EXPECT_TRUE(statsLabel->text().contains(QStringLiteral("空中三角测量")));

    QJsonObject runSettings;
    QObject::connect(&dialog,
                     &SparseCloudPostProcessDialog::runRequested,
                     [&runSettings](const QJsonObject &settings)
                     {
                         runSettings = settings;
                     });
    okButton->click();
    EXPECT_EQ(runSettings.value(QStringLiteral("sourceAtIndex")).toInt(-1), 11);

    sourceCombo->setCurrentIndex(1);
    EXPECT_TRUE(statsLabel->text().contains(QStringLiteral("260 个三维点")));
    EXPECT_TRUE(statsLabel->text().contains(QStringLiteral("稀疏云后处理")));
}

TEST(SparseCloudPostProcessDialogTest, AcceptsSummarizedFormalSfmRecordsUsingPointCount)
{
    const QJsonObject quality{
        {QStringLiteral("result_kind"), xjw::gui::project::kSparseResultKindSfmSparseReconstruction},
        {QStringLiteral("ba_applied"), true},
        {QStringLiteral("camera_count"), 444},
        {QStringLiteral("registered_image_count"), 444},
        {QStringLiteral("input_image_count"), 444},
        {QStringLiteral("point_count"), 588257},
        {QStringLiteral("two_view_ratio"), 0.6851767169791435},
        {QStringLiteral("median_track_len"), 3},
        {QStringLiteral("mean_reproj_px"), 0.8},
        {QStringLiteral("median_reproj_px"), 0.7}
    };
    const QJsonObject formalRecord{
        {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")},
        {QStringLiteral("operation_display_name"), QStringLiteral("稀疏点云")},
        {QStringLiteral("point_count"), 588257},
        {QStringLiteral("camera_count"), 444},
        {QStringLiteral("ba_applied"), true},
        {QStringLiteral("quality"), quality},
        {QStringLiteral("files"), QJsonObject{
            {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("E:/tmp/sfm_sparse.ply")},
            {QStringLiteral("sparse_cloud_points_json"), QStringLiteral("E:/tmp/sfm_sparse_points.json")}
        }},
        {QStringLiteral("selected_images"), QJsonArray{
            QStringLiteral("image_0002.JPG"),
            QStringLiteral("image_0445.JPG")
        }}
    };
    const QJsonArray summary = xjw::gui::project::summarizeAtResults(
        QJsonObject{{QStringLiteral("aerial_triangulation_results"), QJsonArray{formalRecord}}});
    ASSERT_EQ(summary.size(), 1);
    EXPECT_EQ(summary.at(0).toObject().value(QStringLiteral("sparse_point_count")).toInt(), 588257);
    EXPECT_TRUE(xjw::gui::project::isProductionSparseResult(summary.at(0).toObject()));

    SparseCloudPostProcessDialog dialog;
    auto *sourceCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_sourceCombo"));
    auto *statsLabel = dialog.findChild<QLabel *>(QStringLiteral("m_statsLabel"));
    auto *buttonBox = dialog.findChild<QDialogButtonBox *>();
    ASSERT_NE(sourceCombo, nullptr);
    ASSERT_NE(statsLabel, nullptr);
    ASSERT_NE(buttonBox, nullptr);
    QPushButton *okButton = buttonBox->button(QDialogButtonBox::Ok);
    ASSERT_NE(okButton, nullptr);

    dialog.setAvailableSparseClouds(summary);

    ASSERT_EQ(sourceCombo->count(), 1);
    EXPECT_TRUE(okButton->isEnabled());
    EXPECT_TRUE(statsLabel->text().contains(QStringLiteral("588257 个三维点")));
    EXPECT_TRUE(statsLabel->text().contains(QStringLiteral("稀疏点云")));
}

TEST(SparseCloudPostProcessDialogTest, PopulatingSourcesDoesNotEmitSettingsChanged)
{
    const QJsonObject sfmQuality = xjw::gui::project::buildSparseQualityMetadata(
        productionSparsePoints(),
        3,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);
    const QJsonObject sfmRecord = sparseResultRecord(21,
                                                     QStringLiteral("sfm-init"),
                                                     QStringLiteral("workflow_aerial_triangulation"),
                                                     QStringLiteral("空中三角测量"),
                                                     420,
                                                     sfmQuality);

    SparseCloudPostProcessDialog dialog;
    QSignalSpy settingsSpy(&dialog, &SparseCloudPostProcessDialog::settingsChanged);
    dialog.setAvailableSparseClouds(QJsonArray{sfmRecord});

    EXPECT_EQ(settingsSpy.count(), 0);
}

TEST(SparseCloudPostProcessDialogTest, DisablesRunWhenOnlyPreviewSparseInputsExist)
{
    const QJsonObject previewQuality = xjw::gui::project::buildSparseQualityMetadata(
        QJsonArray{
            QJsonObject{{QStringLiteral("track_len"), 2},
                        {QStringLiteral("rms_reproj_px"), 0.9}},
            QJsonObject{{QStringLiteral("track_len"), 2},
                        {QStringLiteral("rms_reproj_px"), 1.1}}
        },
        2,
        false,
        xjw::gui::project::kSparseResultKindPairwisePreview);
    const QJsonObject previewRecord = sparseResultRecord(7,
                                                         QStringLiteral("preview-only"),
                                                         QStringLiteral("triangulation"),
                                                         QStringLiteral("两视预览云"),
                                                         120,
                                                         previewQuality);
    ASSERT_FALSE(xjw::gui::project::isProductionSparseResult(previewRecord));

    SparseCloudPostProcessDialog dialog;
    auto *sourceCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_sourceCombo"));
    auto *statsLabel = dialog.findChild<QLabel *>(QStringLiteral("m_statsLabel"));
    auto *buttonBox = dialog.findChild<QDialogButtonBox *>();
    auto *reprojCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_reprojCheck"));
    ASSERT_NE(sourceCombo, nullptr);
    ASSERT_NE(statsLabel, nullptr);
    ASSERT_NE(buttonBox, nullptr);
    ASSERT_NE(reprojCheck, nullptr);
    QPushButton *okButton = buttonBox->button(QDialogButtonBox::Ok);
    ASSERT_NE(okButton, nullptr);

    QJsonObject changedSettings;
    QObject::connect(&dialog,
                     &SparseCloudPostProcessDialog::settingsChanged,
                     [&changedSettings](const QJsonObject &settings)
                     {
                         changedSettings = settings;
                     });

    dialog.setAvailableSparseClouds(QJsonArray{previewRecord});

    EXPECT_EQ(sourceCombo->count(), 0);
    EXPECT_FALSE(okButton->isEnabled());
    EXPECT_TRUE(statsLabel->text().isEmpty());

    reprojCheck->setChecked(!reprojCheck->isChecked());
    EXPECT_EQ(changedSettings.value(QStringLiteral("sourceAtIndex")).toInt(0), -1);
}

TEST(SparseCloudPostProcessDialogTest, EmitsExternalPlyInputSettingsAndDisablesQualityFilters)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString plyPath = QDir(tempDir.path()).filePath(QStringLiteral("external_sparse.ply"));
    writeMinimalPointCloudPly(plyPath, {
        {0.0, 0.0, 0.0},
        {0.1, 0.0, 0.0},
        {0.2, 0.0, 0.0}
    });

    SparseCloudPostProcessDialog dialog;
    auto *sourceModeCombo = dialog.findChild<QComboBox *>(QStringLiteral("m_sourceModeCombo"));
    auto *externalPathEdit = dialog.findChild<QLineEdit *>(QStringLiteral("m_externalPathEdit"));
    auto *buttonBox = dialog.findChild<QDialogButtonBox *>();
    ASSERT_NE(sourceModeCombo, nullptr);
    ASSERT_NE(externalPathEdit, nullptr);
    ASSERT_NE(buttonBox, nullptr);
    QPushButton *okButton = buttonBox->button(QDialogButtonBox::Ok);
    ASSERT_NE(okButton, nullptr);

    const int externalIndex = sourceModeCombo->findData(QStringLiteral("external_ply"));
    ASSERT_GE(externalIndex, 0);
    sourceModeCombo->setCurrentIndex(externalIndex);
    externalPathEdit->setText(plyPath);

    QJsonObject runSettings;
    QObject::connect(&dialog,
                     &SparseCloudPostProcessDialog::runRequested,
                     [&runSettings](const QJsonObject &settings)
                     {
                         runSettings = settings;
                     });
    okButton->click();

    EXPECT_EQ(runSettings.value(QStringLiteral("sourceKind")).toString(), QStringLiteral("external_ply"));
    EXPECT_EQ(QDir::cleanPath(runSettings.value(QStringLiteral("externalSparseCloudPath")).toString()),
              QDir::cleanPath(plyPath));
    EXPECT_EQ(runSettings.value(QStringLiteral("sourceAtIndex")).toInt(0), -1);
    EXPECT_FALSE(runSettings.value(QStringLiteral("filterByReprojError")).toBool(true));
    EXPECT_FALSE(runSettings.value(QStringLiteral("filterByTrackLen")).toBool(true));
    EXPECT_FALSE(runSettings.value(QStringLiteral("filterByTriAngle")).toBool(true));
    EXPECT_FALSE(runSettings.value(QStringLiteral("localReprojFilter")).toBool(true));
}

TEST(SparsePointWorkflowUtilsTest, LocalOptimAcceptsExternalPlyWithoutSidecar)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString plyPath = QDir(tempDir.path()).filePath(QStringLiteral("external_sparse.ply"));
    writeMinimalPointCloudPly(plyPath, {
        {0.00, 0.0, 0.0},
        {0.02, 0.0, 0.0},
        {0.04, 0.0, 0.0},
        {5.00, 5.0, 5.0}
    });

    xjw::gui::project::SparsePointContext context;
    context.sparseCloudPath = plyPath;

    QJsonObject settings;
    settings[QStringLiteral("sourceKind")] = QStringLiteral("external_ply");
    settings[QStringLiteral("externalSparseCloudPath")] = plyPath;
    settings[QStringLiteral("voxelSize")] = 0.2;
    settings[QStringLiteral("minVoxelPoints")] = 2;
    settings[QStringLiteral("localReprojFilter")] = true;

    xjw::gui::project::SparsePointOperationResult result;
    QString errorMessage;
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    EXPECT_TRUE(xjw::gui::project::runSparsePointLocalOptim(context,
                                                           settings,
                                                           outputDir,
                                                           &result,
                                                           &errorMessage))
        << errorMessage.toStdString();

    EXPECT_EQ(result.inputCount, 4);
    EXPECT_EQ(result.outputCount, 3);
    EXPECT_TRUE(QFileInfo::exists(result.sparseCloudPath));
    EXPECT_TRUE(QFileInfo::exists(result.sidecarPath));

    QFile sidecarFile(result.sidecarPath);
    ASSERT_TRUE(sidecarFile.open(QIODevice::ReadOnly));
    const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();
    EXPECT_EQ(sidecar.value(QStringLiteral("point_count")).toInt(), 3);
    EXPECT_EQ(QDir::cleanPath(sidecar.value(QStringLiteral("source_ply")).toString()),
              QDir::cleanPath(plyPath));
    EXPECT_FALSE(sidecar.value(QStringLiteral("quality_metrics_available")).toBool(true));
}

TEST(SparsePointWorkflowUtilsTest, ProjectResultModeUsesSidecarWhenExternalPathSettingIsEmpty)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString sourcePlyPath = QDir(tempDir.path()).filePath(QStringLiteral("source_sparse.ply"));
    const QString sidecarPath = QDir(tempDir.path()).filePath(QStringLiteral("sparse_cloud_points.json"));
    writeMinimalPointCloudPly(sourcePlyPath, {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0}
    });
    writeMinimalSparseSidecar(sidecarPath, {1, 3, 4});

    xjw::gui::project::SparsePointContext context;
    context.sparseCloudPath = sourcePlyPath;
    context.sidecarPath = sidecarPath;

    QJsonObject settings;
    settings[QStringLiteral("sourceKind")] = QStringLiteral("project_result");
    settings[QStringLiteral("externalSparseCloudPath")] = QString();
    settings[QStringLiteral("filterByReprojError")] = false;
    settings[QStringLiteral("filterByTrackLen")] = true;
    settings[QStringLiteral("minTrackLen")] = 3;
    settings[QStringLiteral("filterByTriAngle")] = false;
    settings[QStringLiteral("filterByStatistical")] = false;
    settings[QStringLiteral("filterByDensity")] = false;

    xjw::gui::project::SparsePointOperationResult result;
    QString errorMessage;
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out_project"));
    EXPECT_TRUE(xjw::gui::project::runSparsePointOutlierRemoval(context,
                                                               settings,
                                                               outputDir,
                                                               &result,
                                                               &errorMessage))
        << errorMessage.toStdString();

    EXPECT_EQ(result.inputCount, 3);
    EXPECT_EQ(result.outputCount, 2);

    QFile sidecarFile(result.sidecarPath);
    ASSERT_TRUE(sidecarFile.open(QIODevice::ReadOnly));
    const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();
    EXPECT_EQ(sidecar.value(QStringLiteral("point_count")).toInt(), 2);
    EXPECT_TRUE(sidecar.value(QStringLiteral("quality_metrics_available")).toBool(false));
    EXPECT_FALSE(sidecar.contains(QStringLiteral("source_ply")));
}

TEST(SparsePointWorkflowUtilsTest, OutlierRemovalPreservesSourcePlyRgbWhenSidecarHasNoColors)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString sourcePlyPath = QDir(tempDir.path()).filePath(QStringLiteral("colored_sparse.ply"));
    const QString sidecarPath = QDir(tempDir.path()).filePath(QStringLiteral("sparse_cloud_points.json"));
    writeMinimalColoredPointCloudPly(sourcePlyPath,
                                     {
                                         {0.0, 0.0, 0.0},
                                         {1.0, 0.0, 0.0},
                                         {2.0, 0.0, 0.0}
                                     },
                                     {
                                         {10, 20, 30},
                                         {40, 50, 60},
                                         {70, 80, 90}
                                     });
    writeMinimalSparseSidecar(sidecarPath, {1, 3, 4});

    xjw::gui::project::SparsePointContext context;
    context.sparseCloudPath = sourcePlyPath;
    context.sidecarPath = sidecarPath;

    QJsonObject settings;
    settings[QStringLiteral("sourceKind")] = QStringLiteral("project_result");
    settings[QStringLiteral("externalSparseCloudPath")] = QString();
    settings[QStringLiteral("filterByReprojError")] = false;
    settings[QStringLiteral("filterByTrackLen")] = true;
    settings[QStringLiteral("minTrackLen")] = 3;
    settings[QStringLiteral("filterByTriAngle")] = false;
    settings[QStringLiteral("filterByStatistical")] = false;
    settings[QStringLiteral("filterByDensity")] = false;

    xjw::gui::project::SparsePointOperationResult result;
    QString errorMessage;
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out_colored"));
    ASSERT_TRUE(xjw::gui::project::runSparsePointOutlierRemoval(context,
                                                               settings,
                                                               outputDir,
                                                               &result,
                                                               &errorMessage))
        << errorMessage.toStdString();

    EXPECT_EQ(result.inputCount, 3);
    EXPECT_EQ(result.outputCount, 2);
    auto outputCloud = plapoint::io::readPly<float>(result.sparseCloudPath.toStdString());
    ASSERT_TRUE(outputCloud != nullptr);
    ASSERT_EQ(outputCloud->size(), 2u);
    ASSERT_TRUE(outputCloud->hasColors());
    EXPECT_EQ(outputCloud->colors()->getValue(0, 0), 40);
    EXPECT_EQ(outputCloud->colors()->getValue(0, 1), 50);
    EXPECT_EQ(outputCloud->colors()->getValue(0, 2), 60);
    EXPECT_EQ(outputCloud->colors()->getValue(1, 0), 70);
    EXPECT_EQ(outputCloud->colors()->getValue(1, 1), 80);
    EXPECT_EQ(outputCloud->colors()->getValue(1, 2), 90);
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

TEST(MainMenuTest, ViewMenuExposesCheckedCameraVisibilityAction)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.toggleCamerasAction();
    ASSERT_NE(action, nullptr);
    EXPECT_EQ(action->text(), QStringLiteral("显示相机"));
    EXPECT_TRUE(action->isCheckable());
    EXPECT_TRUE(action->isChecked());

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    EXPECT_TRUE(viewMenu->actions().contains(action));
}

TEST(CameraSceneWidgetTest, CameraVisibilityToggleIsExposedAndGuardsCameraOverlayOnly)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const QString mainWindowSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(mainWindowSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("void setShowCameras(bool show)")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool areCamerasVisible() const")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool m_showCameras = true")));
    EXPECT_TRUE(source.contains(QStringLiteral("void CameraSceneWidget::setShowCameras(bool show)")));
    EXPECT_TRUE(source.contains(QStringLiteral("if (m_showCameras)")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("toggleCamerasAction()")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("&CameraSceneWidget::setShowCameras")));
}

TEST(MainMenuTest, TriangulationActionNamesPairwisePreviewCloud)
{
    QMainWindow window;
    MainMenu menu(&window);

    QAction *action = menu.triangulateAction();
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(action->text().contains(QStringLiteral("两视预览云")));
    EXPECT_FALSE(action->text().contains(QStringLiteral("初始稀疏点云")));
}

TEST(MainMenuTest, SparseReconstructionSeparatesMainFlowAndAdvancedTools)
{
    QMainWindow window;
    MainMenu menu(&window);

    ASSERT_NE(menu.detectFeaturesAction(), nullptr);
    ASSERT_NE(menu.vocabularyOverlapAction(), nullptr);
    ASSERT_NE(menu.matchFeaturesAction(), nullptr);
    ASSERT_NE(menu.aerialTriangulationAction(), nullptr);
    ASSERT_NE(menu.sparseCloudPostProcessAction(), nullptr);
    ASSERT_NE(menu.viewMatchesAction(), nullptr);
    ASSERT_NE(menu.buildObsNetworkAction(), nullptr);
    ASSERT_NE(menu.initCameraPoseAction(), nullptr);
    ASSERT_NE(menu.triangulateAction(), nullptr);
    ASSERT_NE(menu.reconBundleAdjustAction(), nullptr);

    EXPECT_EQ(menu.aerialTriangulationAction()->text(), QStringLiteral("空中三角测量..."));
    EXPECT_EQ(menu.triangulateAction()->text(), QStringLiteral("生成两视预览云..."));
    EXPECT_FALSE(menu.triangulateAction()->text().contains(QStringLiteral("空中三角")));

    QMenu *reconstructionMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("重建"));
    ASSERT_NE(reconstructionMenu, nullptr);
    QMenu *sparseMenu = findSubMenuByTitle(reconstructionMenu, QStringLiteral("稀疏重建"));
    ASSERT_NE(sparseMenu, nullptr);
    QMenu *advancedMenu = findSubMenuByTitle(sparseMenu, QStringLiteral("高级工具"));
    ASSERT_NE(advancedMenu, nullptr);

    const QStringList mainFlowActions = {
        QStringLiteral("特征点提取"),
        QStringLiteral("重叠对规划..."),
        QStringLiteral("连接点匹配"),
        QStringLiteral("空中三角测量..."),
        QStringLiteral("稀疏点云后处理...")
    };
    const QStringList advancedActions = {
        QStringLiteral("查看匹配"),
        QStringLiteral("构建观测网络..."),
        QStringLiteral("初始化相机位姿..."),
        QStringLiteral("生成两视预览云..."),
        QStringLiteral("单独光束法平差...")
    };
    EXPECT_EQ(directActionTexts(sparseMenu).join(QStringLiteral("|")),
              mainFlowActions.join(QStringLiteral("|")));
    EXPECT_EQ(directActionTexts(advancedMenu).join(QStringLiteral("|")),
              advancedActions.join(QStringLiteral("|")));

    const QList<QAction *> sparseActions = sparseMenu->actions();
    ASSERT_GE(sparseActions.size(), 7);
    EXPECT_TRUE(sparseActions.at(5)->isSeparator());
    ASSERT_NE(sparseActions.at(6)->menu(), nullptr);
    EXPECT_EQ(sparseActions.at(6)->menu(), advancedMenu);

    QMenu *toolsMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("工具"));
    ASSERT_NE(toolsMenu, nullptr);
    EXPECT_FALSE(directActionTexts(toolsMenu).contains(QStringLiteral("查看匹配")));
}

TEST(MainMenuTest, UiBackedSparseReconstructionBindsExistingActionsAndKeepsMenuTree)
{
    QMainWindow window;

    auto *projectMenu = window.menuBar()->addMenu(QStringLiteral("项目"));
    projectMenu->setObjectName(QStringLiteral("menuProject"));
    auto *newProject = new QAction(QStringLiteral("新建项目"), &window);
    newProject->setObjectName(QStringLiteral("actionNewProject"));
    projectMenu->addAction(newProject);

    auto makeAction = [&window](const char *objectName, const QString &text) {
        auto *action = new QAction(text, &window);
        action->setObjectName(QString::fromLatin1(objectName));
        return action;
    };

    QAction *detectFeatures = makeAction("actionDetectFeatures", QStringLiteral("特征点提取"));
    QAction *vocabularyOverlap = makeAction("actionVocabularyOverlap", QStringLiteral("重叠对规划..."));
    QAction *matchFeatures = makeAction("actionMatchFeatures", QStringLiteral("连接点匹配"));
    QAction *aerialTriangulation = makeAction("actionAerialTriangulation", QStringLiteral("空中三角测量..."));
    QAction *sparseCloudPostProcess =
        makeAction("actionSparseCloudPostProcess", QStringLiteral("稀疏点云后处理..."));
    QAction *viewMatches = makeAction("actionViewMatches", QStringLiteral("查看匹配"));
    QAction *buildObsNetwork = makeAction("actionBuildObsNetwork", QStringLiteral("构建观测网络..."));
    QAction *initCameraPose = makeAction("actionInitCameraPose", QStringLiteral("初始化相机位姿..."));
    QAction *triangulate = makeAction("actionTriangulate", QStringLiteral("生成两视预览云..."));
    QAction *reconBundleAdjust = makeAction("actionReconBundleAdjust", QStringLiteral("单独光束法平差..."));
    QAction *manualPointCloudPrune = makeAction("actionManualPointCloudPrune", QStringLiteral("手动点云剔除"));
    QAction *viewWorkflowReport = makeAction("actionViewWorkflowReport", QStringLiteral("查看工作流程报告..."));

    auto *reconstructionMenu = window.menuBar()->addMenu(QStringLiteral("重建"));
    reconstructionMenu->setObjectName(QStringLiteral("menuReconstruction"));
    auto *sparseMenu = reconstructionMenu->addMenu(QStringLiteral("稀疏重建"));
    sparseMenu->setObjectName(QStringLiteral("menuSparseReconstruction"));
    auto *advancedMenu = new QMenu(QStringLiteral("高级工具"), sparseMenu);
    advancedMenu->setObjectName(QStringLiteral("menuSparseAdvancedTools"));

    advancedMenu->addAction(viewMatches);
    advancedMenu->addAction(buildObsNetwork);
    advancedMenu->addAction(initCameraPose);
    advancedMenu->addAction(triangulate);
    advancedMenu->addAction(reconBundleAdjust);
    sparseMenu->addAction(detectFeatures);
    sparseMenu->addAction(vocabularyOverlap);
    sparseMenu->addAction(matchFeatures);
    sparseMenu->addAction(aerialTriangulation);
    sparseMenu->addAction(sparseCloudPostProcess);
    sparseMenu->addSeparator();
    sparseMenu->addMenu(advancedMenu);

    auto *toolsMenu = window.menuBar()->addMenu(QStringLiteral("工具"));
    toolsMenu->setObjectName(QStringLiteral("menuTools"));
    toolsMenu->addAction(manualPointCloudPrune);
    toolsMenu->addSeparator();
    toolsMenu->addAction(viewWorkflowReport);

    MainMenu menu(&window);

    EXPECT_EQ(menu.detectFeaturesAction(), detectFeatures);
    EXPECT_EQ(menu.vocabularyOverlapAction(), vocabularyOverlap);
    EXPECT_EQ(menu.matchFeaturesAction(), matchFeatures);
    EXPECT_EQ(menu.aerialTriangulationAction(), aerialTriangulation);
    EXPECT_EQ(menu.sparseCloudPostProcessAction(), sparseCloudPostProcess);
    EXPECT_EQ(menu.viewMatchesAction(), viewMatches);
    EXPECT_EQ(menu.buildObsNetworkAction(), buildObsNetwork);
    EXPECT_EQ(menu.initCameraPoseAction(), initCameraPose);
    EXPECT_EQ(menu.triangulateAction(), triangulate);
    EXPECT_EQ(menu.reconBundleAdjustAction(), reconBundleAdjust);
    EXPECT_EQ(menu.aerialTriangulationAction()->text(), QStringLiteral("空中三角测量..."));

    const QStringList mainFlowActions = {
        QStringLiteral("特征点提取"),
        QStringLiteral("重叠对规划..."),
        QStringLiteral("连接点匹配"),
        QStringLiteral("空中三角测量..."),
        QStringLiteral("稀疏点云后处理...")
    };
    const QStringList advancedActions = {
        QStringLiteral("查看匹配"),
        QStringLiteral("构建观测网络..."),
        QStringLiteral("初始化相机位姿..."),
        QStringLiteral("生成两视预览云..."),
        QStringLiteral("单独光束法平差...")
    };
    EXPECT_EQ(directActionTexts(sparseMenu).join(QStringLiteral("|")),
              mainFlowActions.join(QStringLiteral("|")));
    EXPECT_EQ(directActionTexts(advancedMenu).join(QStringLiteral("|")),
              advancedActions.join(QStringLiteral("|")));
    EXPECT_FALSE(directActionTexts(toolsMenu).contains(QStringLiteral("查看匹配")));
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

TEST(ThreeDReconstructionDialogTest, AerialTriangulationModeUsesSparseOnlyLabels)
{
    ThreeDReconstructionDialog dialog;
    auto *titleLabel = dialog.findChild<QLabel *>(QStringLiteral("m_titleLabel"));
    auto *exportObjCheck = dialog.findChild<QCheckBox *>(QStringLiteral("m_exportObjCheck"));
    auto *startButton = dialog.findChild<QPushButton *>(QStringLiteral("m_startBtn"));
    ASSERT_NE(titleLabel, nullptr);
    ASSERT_NE(exportObjCheck, nullptr);
    ASSERT_NE(startButton, nullptr);

    dialog.setMode(ThreeDReconstructionDialog::Mode::AerialTriangulation);
    dialog.setImageCount(12);
    dialog.setDefaultOutputDir(QStringLiteral("E:/tmp/at"));

    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("空中三角测量"));
    EXPECT_EQ(titleLabel->text(), QStringLiteral("<b>一键生成正式 SfM/BA 稀疏云</b>"));
    EXPECT_FALSE(exportObjCheck->isChecked());
    EXPECT_TRUE(exportObjCheck->isHidden());
    EXPECT_EQ(startButton->text(), QStringLiteral("开始空三"));

    QJsonObject settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("workflow_kind")).toString(),
              QStringLiteral("aerial_triangulation"));
    EXPECT_FALSE(settings.value(QStringLiteral("export_obj")).toBool(true));

    QJsonObject appliedSettings;
    appliedSettings[QStringLiteral("export_obj")] = true;
    dialog.applySettings(appliedSettings);
    settings = dialog.collectSettings();
    EXPECT_FALSE(settings.value(QStringLiteral("export_obj")).toBool(true));

    dialog.setMode(ThreeDReconstructionDialog::Mode::ThreeDReconstruction);
    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("三维重建"));
    EXPECT_EQ(titleLabel->text(), QStringLiteral("<b>一键生成三维模型</b>"));
    EXPECT_FALSE(exportObjCheck->isHidden());
    EXPECT_TRUE(exportObjCheck->isChecked());
    EXPECT_EQ(startButton->text(), QStringLiteral("开始重建"));

    settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("workflow_kind")).toString(),
              QStringLiteral("three_d_reconstruction"));
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

TEST(AerialTriangulationWorkflowTest, MainWindowWiresSparseOnlyAerialTriangulationAction)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int actionIndex = source.indexOf(QStringLiteral("aerialTriangulationAction()"));
    ASSERT_GE(actionIndex, 0);
    const int nextMenuAction = source.indexOf(QStringLiteral("if (m_mainMenu->"), actionIndex + 1);
    ASSERT_GT(nextMenuAction, actionIndex);
    const QString connectBlock = source.mid(actionIndex, nextMenuAction - actionIndex);

    EXPECT_TRUE(connectBlock.contains(QStringLiteral("connect(m_mainMenu->aerialTriangulationAction()"))
                || connectBlock.contains(QStringLiteral("aerialTriangulationAction()")));
    EXPECT_TRUE(connectBlock.contains(QStringLiteral("&QAction::triggered")));
    EXPECT_TRUE(connectBlock.contains(QStringLiteral("m_menuWorkflowController")));
    EXPECT_TRUE(connectBlock.contains(QStringLiteral("&MenuWorkflowController::openAerialTriangulationDialog")));
    EXPECT_FALSE(connectBlock.contains(QStringLiteral("openThreeDReconstructionDialog")));
}

TEST(AerialTriangulationWorkflowTest, SparseOnlyWorkflowStopsBeforeDenseStages)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("openAerialTriangulationDialog")));
    EXPECT_TRUE(header.contains(QStringLiteral("startAerialTriangulationWorkflow")));
    EXPECT_TRUE(source.contains(QStringLiteral("DialogSettingKeys::AerialTriangulation")));
    EXPECT_TRUE(source.contains(QStringLiteral("setMode(ThreeDReconstructionDialog::Mode::AerialTriangulation)")));
    EXPECT_TRUE(source.contains(QStringLiteral("source\"] = QStringLiteral(\"aerial_triangulation\")"))
                || source.contains(QStringLiteral("source\", QStringLiteral(\"aerial_triangulation\")"))
                || source.contains(QStringLiteral("resultRecordExtra[QStringLiteral(\"source\")] = QStringLiteral(\"aerial_triangulation\")")));

    const int sparseStart = source.indexOf(QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(sparseStart, 0);
    const int nextFunction = source.indexOf(QStringLiteral("void MenuWorkflowController::startThreeDReconstructionWorkflow"),
                                            sparseStart);
    ASSERT_GT(nextFunction, sparseStart);
    const QString sparseBlock = source.mid(sparseStart, nextFunction - sparseStart);
    EXPECT_TRUE(sparseBlock.contains(QStringLiteral("SFMService::run(runOpts)")));
    EXPECT_TRUE(sparseBlock.contains(QStringLiteral("appendAtResult")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startThreeDReconstructionWorkflow")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startThreeDReconstructionDenseStage")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startThreeDReconstructionDenseRefineStage")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startThreeDReconstructionMeshStage")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startGenerateDenseCloudAsync")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startDenseCloudRefineAsync")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startMeshReconstructionAsync")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startDenseMatch")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startMVS")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startMesh")));
}

TEST(AerialTriangulationWorkflowTest, MissingUpstreamDataOffersAutoFillOrManualReturn)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("SparsePrerequisiteSummary")));
    EXPECT_TRUE(source.contains(QStringLiteral("自动补齐缺失步骤")));
    EXPECT_TRUE(source.contains(QStringLiteral("返回手动处理")));
    EXPECT_TRUE(source.contains(QStringLiteral("autoGenerateMissingMatches = autoFillMissing")));
    EXPECT_TRUE(source.contains(QStringLiteral("缺少连接点")));

    const int summaryStart = source.indexOf(
        QStringLiteral("MenuWorkflowController::summarizeSparsePrerequisites"));
    ASSERT_GE(summaryStart, 0);
    const int promptStart = source.indexOf(
        QStringLiteral("bool MenuWorkflowController::confirmAutoFillMissingSparseInputs"),
        summaryStart);
    ASSERT_GT(promptStart, summaryStart);
    const QString summaryBody = source.mid(summaryStart, promptStart - summaryStart);
    EXPECT_TRUE(summaryBody.contains(QStringLiteral("ProjectIO::findFeatureForImage")));
    EXPECT_TRUE(summaryBody.contains(QStringLiteral("collectMatchedImageNamePairs")));
    EXPECT_FALSE(summaryBody.contains(QStringLiteral("summary.hasFeatures = !suffixes.isEmpty()")));
    EXPECT_FALSE(summaryBody.contains(QStringLiteral("summary.hasMatches = !matches.isEmpty()")));
    EXPECT_FALSE(summaryBody.contains(QStringLiteral("meta.value(QStringLiteral(\"ipmatch_results\"))")));

    const int sparseStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(sparseStart, 0);
    const int threeDStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startThreeDReconstructionWorkflow"),
        sparseStart);
    ASSERT_GT(threeDStart, sparseStart);
    const QString sparseBody = source.mid(sparseStart, threeDStart - sparseStart);
    EXPECT_TRUE(sparseBody.contains(QStringLiteral("const QJsonObject projectMeta = pm->currentMeta()")));
    EXPECT_TRUE(sparseBody.contains(QStringLiteral("opts.projectMeta = projectMeta")));
    EXPECT_FALSE(sparseBody.contains(QStringLiteral("opts.projectMeta = pm->coreProjectMeta()")));
}

TEST(AerialTriangulationWorkflowTest, PreflightReusesGeneratedPairPlanBeforeReportingMissingMatches)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int summaryStart = source.indexOf(
        QStringLiteral("MenuWorkflowController::summarizeSparsePrerequisites"));
    ASSERT_GE(summaryStart, 0);
    const int promptStart = source.indexOf(
        QStringLiteral("bool MenuWorkflowController::confirmAutoFillMissingSparseInputs"),
        summaryStart);
    ASSERT_GT(promptStart, summaryStart);
    const QString summaryBody = source.mid(summaryStart, promptStart - summaryStart);

    EXPECT_TRUE(summaryBody.contains(QStringLiteral("loadGeneratedPairConstraints")));
    EXPECT_TRUE(summaryBody.contains(QStringLiteral("usedStoredPairs")));
    EXPECT_TRUE(summaryBody.contains(QStringLiteral("storedPairsStale")));
    EXPECT_TRUE(summaryBody.contains(QStringLiteral("generatedPairCoveredCount")));
    EXPECT_TRUE(summaryBody.contains(QStringLiteral("generatedPairRequiredCount")));
    EXPECT_FALSE(summaryBody.contains(QStringLiteral("coveredPairCount == requiredPairCount")));
}

TEST(AerialTriangulationWorkflowTest, DialogStartsWorkflowWithQueuedConnection)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int dialogStart = source.indexOf(QStringLiteral("void MenuWorkflowController::openAerialTriangulationDialog"));
    ASSERT_GE(dialogStart, 0);
    const int nextFunction = source.indexOf(QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"),
                                            dialogStart);
    ASSERT_GT(nextFunction, dialogStart);
    const QString dialogBody = source.mid(dialogStart, nextFunction - dialogStart);

    const int runConnect = dialogBody.indexOf(QStringLiteral("&ThreeDReconstructionDialog::runRequested"));
    ASSERT_GE(runConnect, 0);
    const int dialogExec = dialogBody.indexOf(QStringLiteral("dlg->exec()"), runConnect);
    ASSERT_GT(dialogExec, runConnect);
    const QString runConnectBlock = dialogBody.mid(runConnect, dialogExec - runConnect);
    EXPECT_TRUE(runConnectBlock.contains(QStringLiteral("Qt::QueuedConnection")));
}

TEST(AerialTriangulationWorkflowTest, StartDoesPrerequisiteAndSfmWorkOffGuiThread)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("launchAerialTriangulationSfm")));
    EXPECT_TRUE(header.contains(QStringLiteral("static SparsePrerequisiteSummary summarizeSparsePrerequisites")));

    const int sparseStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(sparseStart, 0);
    const int launchStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::launchAerialTriangulationSfm"),
        sparseStart);
    ASSERT_GT(launchStart, sparseStart);
    const QString startBody = source.mid(sparseStart, launchStart - sparseStart);

    EXPECT_TRUE(startBody.contains(QStringLiteral("QFutureWatcher<SparsePrerequisiteSummary>")));
    EXPECT_TRUE(startBody.contains(QStringLiteral("watcher->setFuture(QtConcurrent::run")));
    EXPECT_TRUE(startBody.contains(QStringLiteral("summarizeSparsePrerequisites(images, projectMeta, projectPath)")));
    const int preflightLaunch = startBody.indexOf(QStringLiteral("watcher->setFuture(QtConcurrent::run"));
    ASSERT_GE(preflightLaunch, 0);
    EXPECT_FALSE(startBody.left(preflightLaunch).contains(QStringLiteral("summarizeSparsePrerequisites(")));

    const int threeDStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startThreeDReconstructionWorkflow"),
        launchStart);
    ASSERT_GT(threeDStart, launchStart);
    const QString launchBody = source.mid(launchStart, threeDStart - launchStart);

    EXPECT_TRUE(launchBody.contains(QStringLiteral("QFutureWatcher<xjw::gui::SFMServiceResult>")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("watcher->setFuture(QtConcurrent::run")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("xjw::gui::SFMService::run(runOpts)")));
    const int sfmLaunch = launchBody.indexOf(QStringLiteral("watcher->setFuture(QtConcurrent::run"));
    ASSERT_GE(sfmLaunch, 0);
    EXPECT_FALSE(launchBody.left(sfmLaunch).contains(QStringLiteral("SFMService::run")));
}

TEST(AerialTriangulationWorkflowTest, SfmLaunchReusesGeneratedPairConstraints)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int launchStart = source.indexOf(
        QStringLiteral("void MenuWorkflowController::launchAerialTriangulationSfm"));
    ASSERT_GE(launchStart, 0);
    const int nextFunction = source.indexOf(
        QStringLiteral("void MenuWorkflowController::startThreeDReconstructionWorkflow"),
        launchStart);
    ASSERT_GT(nextFunction, launchStart);
    const QString launchBody = source.mid(launchStart, nextFunction - launchStart);

    EXPECT_TRUE(source.contains(QStringLiteral("loadGeneratedPairConstraints")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("const QStringList allowedPairs = loadGeneratedPairConstraints")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("opts.restrictPairs = true")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("opts.allowedPairs = allowedPairs")));
    EXPECT_TRUE(launchBody.contains(QStringLiteral("storedPairsStale")));
}

TEST(SfmServicePairPlanningTest, ProjectMetaCamerasEnableBoundedPairPlanning)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/pipeline/SFMService.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("loadKnownCameraCentersFromProjectMeta")));
    EXPECT_TRUE(source.contains(QStringLiteral("hasProjectMetaCameraCenters")));
    EXPECT_TRUE(source.contains(QStringLiteral("pairPlanOptions.knownCameraCenters = projectMetaCameraCenters")));
    EXPECT_TRUE(source.contains(QStringLiteral("匹配候选对")));
    EXPECT_TRUE(source.contains(QStringLiteral("validIdByPath")));
    EXPECT_TRUE(source.contains(QStringLiteral("pairKey.split(QStringLiteral(\"\\n\"))")));
}

TEST(SfmServiceKnownPoseModeTest, CompleteProjectMetaCamerasEnableKnownPoseMode)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/pipeline/SFMService.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("hasCompleteProjectMetaCameras")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "sfmOpts.useKnownCameraPoses = hasCompleteCameraFiles || hasCompleteProjectMetaCameras")));
    EXPECT_TRUE(source.contains(QStringLiteral("projectImageMetaByPath(opts.projectMeta, true)")));
    EXPECT_TRUE(source.contains(QStringLiteral("使用项目元数据已知外参初值模式")));
    EXPECT_TRUE(source.contains(QStringLiteral("相机位姿参与全局 BA 微调")));
    EXPECT_FALSE(source.contains(QStringLiteral("固定相机位姿并直接三角化")));
    EXPECT_TRUE(source.contains(QStringLiteral("const bool baApplied = sfmResult.baTracksTotal > 0")));
    EXPECT_TRUE(source.contains(QStringLiteral("sfmOpts.baOptions.cancelFlag = opts.cancelFlag")));
    EXPECT_TRUE(source.contains(QStringLiteral("sfmOpts.baOptions.progressCallback")));
    EXPECT_TRUE(source.contains(QStringLiteral("正在进行光束法平差")));

    const QString incremental = readProjectSourceFile(QStringLiteral("src/core/sfm/pipeline/IncrementalSfm.cpp"));
    const QString baHeader = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.h"));
    const QString baCore = readProjectSourceFile(QStringLiteral("src/core/bundle_adjust/BundleAdjust.cpp"));
    ASSERT_FALSE(incremental.isEmpty());
    ASSERT_FALSE(baHeader.isEmpty());
    ASSERT_FALSE(baCore.isEmpty());
    EXPECT_TRUE(incremental.contains(QStringLiteral("baOpt.cameraPosePriors")));
    EXPECT_TRUE(incremental.contains(QStringLiteral("buildCameraPosePriorsFromInputCameras")));
    EXPECT_TRUE(incremental.contains(QStringLiteral("alignReconstructionToKnownPosePriors")));
    EXPECT_TRUE(incremental.contains(QStringLiteral("Known-pose Sim3/RANSAC alignment before BA")));
    EXPECT_TRUE(incremental.contains(QStringLiteral("prior.cameraToWorldRotation = inputCamera.cameraToWorldRotation()")));
    EXPECT_TRUE(baHeader.contains(QStringLiteral("struct BACameraPosePrior")));
    EXPECT_TRUE(baHeader.contains(QStringLiteral("cameraPosePriorHuberDelta")));
    EXPECT_TRUE(baCore.contains(QStringLiteral("cameraPosePriorResidual")));
    EXPECT_TRUE(baCore.contains(QStringLiteral("huberWeight(residualNorm, opt.cameraPosePriorHuberDelta)")));
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

TEST(FeatureVisualizationSettingsTest, DefaultsPointColorToBlue)
{
    const QString rendererHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString uiDefaults = readProjectSourceFile(QStringLiteral("src/gui/config/ProjectUiConfigManager.cpp"));
    const QString dialogHeader = readProjectSourceFile(QStringLiteral("src/gui/dialogs/SuperPointVisualizationDialog.h"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/SuperPointVisualizationDialog.cpp"));
    ASSERT_FALSE(rendererHeader.isEmpty());
    ASSERT_FALSE(uiDefaults.isEmpty());
    ASSERT_FALSE(dialogHeader.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());

    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("QColor pointColor = QColor(0, 120, 255)")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor[\"r\"] = 0")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor[\"g\"] = 120")));
    EXPECT_TRUE(uiDefaults.contains(QStringLiteral("pointColor[\"b\"] = 255")));
    EXPECT_TRUE(dialogHeader.contains(QStringLiteral("QColor m_pointColor{0, 120, 255}")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("m_pointColor = QColor(0, 120, 255)")));
    EXPECT_FALSE(dialogSource.contains(QStringLiteral("点颜色黄")));
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

TEST(CanvasWidgetResponsivenessTest, ImageSwitchUsesBackgroundLoadAndIgnoresStaleResults)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    const QString rendererHeader = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.h"));
    const QString rendererSource = readProjectSourceFile(QStringLiteral("src/gui/views/LayerRenderer.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(rendererHeader.isEmpty());
    ASSERT_FALSE(rendererSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QFutureWatcher<QImage> *m_imageWatcher")));
    EXPECT_TRUE(source.contains(QStringLiteral("QtConcurrent::run([pathCopy, projectPath]")));
    EXPECT_TRUE(source.contains(QStringLiteral("LayerRenderer::loadImageForDisplay(pathCopy, projectPath)")));
    EXPECT_TRUE(source.contains(QStringLiteral("QDir::cleanPath(loadedPath) != QDir::cleanPath(m_currentImagePath)")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("static QImage loadImageForDisplay")));
    EXPECT_TRUE(rendererHeader.contains(QStringLiteral("bool addImageLayer(const QImage &image, int z = 0)")));
    EXPECT_TRUE(rendererSource.contains(QStringLiteral("QPixmap::fromImage(image)")));
}

TEST(CanvasWidgetResponsivenessTest, StaleFeatureLoadsDoNotPaintOverCurrentImage)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int finishedIndex = source.indexOf(QStringLiteral("m_spWatcher, &QFutureWatcher<std::vector<cv::KeyPoint>>::finished"));
    ASSERT_GE(finishedIndex, 0);
    const QString finishedBlock = source.mid(finishedIndex, 1800);

    EXPECT_TRUE(finishedBlock.contains(QStringLiteral("QDir::cleanPath(imagePath) == QDir::cleanPath(m_currentImagePath)")));
    EXPECT_TRUE(finishedBlock.contains(QStringLiteral("if (isCurrentImage && m_layerRenderer)")));
}

TEST(CanvasWidgetResponsivenessTest, FeatureLoadEstimatesOrientationOnlyWhenDisplayed)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/CanvasWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int startIndex = source.indexOf(QStringLiteral("void CanvasWidget::startSpLoadForImage"));
    ASSERT_GE(startIndex, 0);
    const int imreadIndex = source.indexOf(QStringLiteral("cv::imread(imagePathCopy.toStdString()"), startIndex);
    ASSERT_GE(imreadIndex, startIndex);
    const QString loadBlock = source.mid(startIndex, imreadIndex - startIndex + 600);

    EXPECT_TRUE(loadBlock.contains(QStringLiteral("const QString projectPath = property(\"currentProjectPath\").toString()")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("const bool shouldEstimateOrientation = m_currentFeatureOpts.showOrientation")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("[imagePathCopy, activeSuffix, projectPath, shouldEstimateOrientation]()")));
    EXPECT_TRUE(loadBlock.contains(QStringLiteral("if (shouldEstimateOrientation)")));
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

    const QJsonObject quality = result.resultJson.value(QStringLiteral("quality")).toObject();
    EXPECT_EQ(result.resultJson.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindPairwisePreview);
    EXPECT_EQ(quality.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindPairwisePreview);
    EXPECT_EQ(quality.value(QStringLiteral("point_count")).toInt(), result.exportedPointCount);
    EXPECT_TRUE(xjw::gui::project::isPairwisePreviewSparseResult(result.resultJson));
    EXPECT_FALSE(xjw::gui::project::isProductionSparseResult(result.resultJson));
}

TEST(ProjectTriangulationServiceTest, UsesSidecarV2IndicesForMultiViewTracks)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("demo.plascan"));
    const QString matchesDir = ProjectIO::ipmatchOutputDir(projectPath);
    ASSERT_TRUE(QDir().mkpath(matchesDir));

    const QString image0Path = QDir(tempDir.path()).filePath(QStringLiteral("1.jpg"));
    const QString image1Path = QDir(tempDir.path()).filePath(QStringLiteral("2.jpg"));
    const QString image2Path = QDir(tempDir.path()).filePath(QStringLiteral("3.jpg"));

    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::Camera camera2 = makeCamera(16.0, 0.0, 0.0);
    const std::array<double, 3> point = {6.0, 0.5, 36.0};

    double u0 = 0.0;
    double v0 = 0.0;
    double u1 = 0.0;
    double v1 = 0.0;
    double u2 = 0.0;
    double v2 = 0.0;
    ASSERT_TRUE(projectPoint(camera0, point, &u0, &v0));
    ASSERT_TRUE(projectPoint(camera1, point, &u1, &v1));
    ASSERT_TRUE(projectPoint(camera2, point, &u2, &v2));

    auto writeSidecar = [](const QString &path,
                           const QString &imageA,
                           const QString &imageB,
                           const QPointF &uvA,
                           const QPointF &uvB,
                           int featureA,
                           int featureB) {
        QFile matchFile(path);
        ASSERT_TRUE(matchFile.open(QIODevice::WriteOnly));
        matchFile.write("dummy");
        matchFile.close();

        QFile sidecarFile(path + QStringLiteral(".json"));
        ASSERT_TRUE(sidecarFile.open(QIODevice::WriteOnly));
        QJsonArray matchedPoints0;
        QJsonArray matchedPoints1;
        QJsonArray matchedIndices0;
        QJsonArray matchedIndices1;
        matchedPoints0.append(QJsonArray{uvA.x(), uvA.y()});
        matchedPoints1.append(QJsonArray{uvB.x(), uvB.y()});
        matchedIndices0.append(featureA);
        matchedIndices1.append(featureB);
        const QJsonObject sidecarObject{
            {QStringLiteral("feature_format_version"), 2},
            {QStringLiteral("image0_path"), imageA},
            {QStringLiteral("image1_path"), imageB},
            {QStringLiteral("matched_points0"), matchedPoints0},
            {QStringLiteral("matched_points1"), matchedPoints1},
            {QStringLiteral("matched_indices0"), matchedIndices0},
            {QStringLiteral("matched_indices1"), matchedIndices1}
        };
        sidecarFile.write(QJsonDocument(sidecarObject).toJson());
        sidecarFile.close();
    };

    const QString match01Path = QDir(matchesDir).filePath(QStringLiteral("1__2.match"));
    const QString match02Path = QDir(matchesDir).filePath(QStringLiteral("1__3.match"));
    writeSidecar(match01Path, image0Path, image1Path, QPointF(u0, v0), QPointF(u1, v1), 7, 13);
    writeSidecar(match02Path, image0Path, image2Path, QPointF(u0, v0), QPointF(u2, v2), 7, 29);

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        buildImageEntry(image0Path, camera0),
        buildImageEntry(image1Path, camera1),
        buildImageEntry(image2Path, camera2)
    };
    meta[QStringLiteral("ipmatch_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("image0"), image0Path},
                    {QStringLiteral("image1"), image1Path},
                    {QStringLiteral("output"), match01Path}},
        QJsonObject{{QStringLiteral("image0"), image0Path},
                    {QStringLiteral("image1"), image2Path},
                    {QStringLiteral("output"), match02Path}}
    };

    xjw::gui::project::TriangulationServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;
    options.minObservations = 3;
    options.ignoreTwoViewTracks = true;
    options.minTrackLength = 3;

    const auto result = xjw::gui::project::ProjectTriangulationService::run(
        meta, QStringList{image0Path, image1Path, image2Path}, options);

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    ASSERT_EQ(result.exportedPointCount, 1);

    const QJsonArray points = result.resultJson.value(QStringLiteral("points")).toArray();
    ASSERT_EQ(points.size(), 1);
    EXPECT_EQ(points.at(0).toObject().value(QStringLiteral("track_len")).toInt(), 3);
}

TEST(ProjectTriangulationUiTest, FinalizeTriangulationStoresPreviewQualityMetadata)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectSparseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("mergeSparseQualityIntoRecord")));
    EXPECT_TRUE(source.contains(QStringLiteral("result.resultJson.value(QStringLiteral(\"quality\"))")));
    EXPECT_TRUE(source.contains(QStringLiteral("两视预览云")));
}

TEST(SfmSparseResultMetadataTest, SfmServicePublishesProductionQualityRecord)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/core/pipeline/SFMService.h"));
    const QString service = readProjectSourceFile(QStringLiteral("src/core/pipeline/SFMService.cpp"));
    const QString workflow = readProjectSourceFile(QStringLiteral("src/gui/project/support/ProjectSfmWorkflow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(service.isEmpty());
    ASSERT_FALSE(workflow.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject qualityMetadata")));
    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject resultRecordExtra")));
    EXPECT_TRUE(header.contains(QStringLiteral("QJsonObject sfmDiagnostics")));
    EXPECT_TRUE(service.contains(QStringLiteral("kSparseResultKindSfmSparseReconstruction")));
    EXPECT_TRUE(service.contains(QStringLiteral("result.qualityMetadata")));
    EXPECT_TRUE(service.contains(QStringLiteral("result.resultRecordExtra")));
    EXPECT_TRUE(service.contains(QStringLiteral("sfm_diagnostics")));
    EXPECT_TRUE(service.contains(QStringLiteral("triangulation_angle_deg")));
    EXPECT_TRUE(workflow.contains(QStringLiteral("result.resultRecordExtra")));
}

TEST(SfmSparseResultMetadataTest, OneClickWorkflowPreservesProductionQualityRecord)
{
    const QString controller = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(controller.isEmpty());

    EXPECT_TRUE(controller.contains(QStringLiteral("result.resultRecordExtra")));
    EXPECT_TRUE(controller.contains(QStringLiteral("result.sfmDiagnostics")));
    EXPECT_TRUE(controller.contains(QStringLiteral("at_report.json")));
}

TEST(BundleAdjustSparseResultMetadataTest, ExportedSparseCloudCarriesFormalQualityMetadata)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    QJsonArray points;
    points.append(QJsonObject{
        {QStringLiteral("valid"), true},
        {QStringLiteral("converged"), true},
        {QStringLiteral("rms_after"), 0.5},
        {QStringLiteral("track_len"), 3},
        {QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}}
    });
    points.append(QJsonObject{
        {QStringLiteral("valid"), true},
        {QStringLiteral("converged"), true},
        {QStringLiteral("rms_after"), 0.7},
        {QStringLiteral("track_len"), 2},
        {QStringLiteral("point_xyz"), QJsonArray{2.0, 3.0, 4.0}}
    });

    const QJsonObject baResult{
        {QStringLiteral("camera_count"), 3},
        {QStringLiteral("mean_rms_after"), 0.6},
        {QStringLiteral("points"), points}
    };

    const auto exportResult = xjw::gui::project::exportBundleAdjustSparseCloud(
        baResult,
        QStringList{QStringLiteral("1.jpg"), QStringLiteral("2.jpg"), QStringLiteral("3.jpg")},
        tempDir.path(),
        true);

    ASSERT_TRUE(exportResult.exported) << exportResult.errorMessage.toStdString();
    EXPECT_EQ(exportResult.extraRecord.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindSparsePostprocess);
    EXPECT_EQ(exportResult.extraRecord.value(QStringLiteral("source_result_kind")).toString(),
              xjw::gui::project::kSparseResultKindSfmSparseReconstruction);
    EXPECT_TRUE(xjw::gui::project::isProductionSparseResult(exportResult.extraRecord));

    const QString sidecarPath = exportResult.extraRecord.value(QStringLiteral("files")).toObject()
                                    .value(QStringLiteral("sparse_cloud_points_json")).toString();
    QFile sidecarFile(sidecarPath);
    ASSERT_TRUE(sidecarFile.open(QIODevice::ReadOnly));
    const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();
    EXPECT_EQ(sidecar.value(QStringLiteral("result_kind")).toString(),
              xjw::gui::project::kSparseResultKindSparsePostprocess);
    EXPECT_TRUE(xjw::gui::project::isProductionSparseResult(sidecar));
}

TEST(DownstreamSparseInputGateTest, ResolveSparseContextSkipsPreviewAndRequiresProduction)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString previewSidecar = QDir(tempDir.path()).filePath(QStringLiteral("preview_points.json"));
    const QString formalSidecar = QDir(tempDir.path()).filePath(QStringLiteral("formal_points.json"));
    QFile previewFile(previewSidecar);
    ASSERT_TRUE(previewFile.open(QIODevice::WriteOnly));
    previewFile.write("{}");
    previewFile.close();
    QFile formalFile(formalSidecar);
    ASSERT_TRUE(formalFile.open(QIODevice::WriteOnly));
    formalFile.write("{}");
    formalFile.close();

    const QJsonObject previewQuality = xjw::gui::project::buildSparseQualityMetadata(
        QJsonArray{QJsonObject{{QStringLiteral("track_len"), 2},
                               {QStringLiteral("rms_reproj_px"), 1.0}}},
        2,
        false,
        xjw::gui::project::kSparseResultKindPairwisePreview);
    const QJsonObject formalQuality = xjw::gui::project::buildSparseQualityMetadata(
        QJsonArray{QJsonObject{{QStringLiteral("track_len"), 2},
                               {QStringLiteral("rms_reproj_px"), 0.8}},
                   QJsonObject{{QStringLiteral("track_len"), 3},
                               {QStringLiteral("rms_reproj_px"), 0.6}}},
        3,
        true,
        xjw::gui::project::kSparseResultKindSfmSparseReconstruction);

    QJsonObject formalRecord = xjw::gui::project::mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("output_dir"), tempDir.path()},
                    {QStringLiteral("files"), QJsonObject{
                         {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("formal.ply")},
                         {QStringLiteral("sparse_cloud_points_json"), formalSidecar}}},
                    {QStringLiteral("selected_images"), QJsonArray{
                         QStringLiteral("1.jpg"), QStringLiteral("2.jpg"), QStringLiteral("3.jpg")}}},
        formalQuality);
    QJsonObject previewRecord = xjw::gui::project::mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("output_dir"), tempDir.path()},
                    {QStringLiteral("files"), QJsonObject{
                         {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("preview.ply")},
                         {QStringLiteral("sparse_cloud_points_json"), previewSidecar}}},
                    {QStringLiteral("selected_images"), QJsonArray{
                         QStringLiteral("1.jpg"), QStringLiteral("2.jpg")}}},
        previewQuality);

    QJsonObject meta;
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{formalRecord, previewRecord};

    xjw::gui::project::SparsePointContext context;
    QString errorMessage;
    EXPECT_TRUE(xjw::gui::project::resolveSparsePointContext(meta, -1, &context, &errorMessage));
    EXPECT_EQ(context.sourceResultIndex, 0);

    errorMessage.clear();
    EXPECT_FALSE(xjw::gui::project::resolveSparsePointContext(meta, 1, &context, &errorMessage));
    EXPECT_TRUE(errorMessage.contains(QStringLiteral("两视")));

    errorMessage.clear();
    EXPECT_TRUE(xjw::gui::project::resolveSparsePointContext(meta, 0, &context, &errorMessage));
    EXPECT_EQ(context.sourceResultIndex, 0);
}

TEST(DownstreamSparseInputGateTest, DenseAndDemUseProductionSparseInputs)
{
    const QString denseSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString terrainSource =
        readProjectSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    ASSERT_FALSE(denseSource.isEmpty());
    ASSERT_FALSE(terrainSource.isEmpty());

    EXPECT_TRUE(denseSource.contains(QStringLiteral("findLatestProductionAtResultIndex")));
    EXPECT_TRUE(denseSource.contains(QStringLiteral("sparseResultBlockingReason")));
    EXPECT_TRUE(terrainSource.contains(QStringLiteral("findLatestProductionAtResultIndex")));
    EXPECT_FALSE(terrainSource.contains(QStringLiteral("startTriangulationAsync(triangulationSettings)")));
}

TEST(DownstreamSparseInputGateTest, OneClickDenseStageUsesCurrentSfmResultIndex)
{
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(controllerSource.contains(QStringLiteral("sfm_at_index")));
    EXPECT_TRUE(controllerSource.contains(
        QStringLiteral("settings.value(QStringLiteral(\"sfm_at_index\")).toInt(-1)")));
}

TEST(DownstreamSparseInputGateTest, OneClickWorkflowStopsDenseWhenCurrentSfmQualityIsBlocked)
{
    const QString controllerSource =
        readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(controllerSource.isEmpty());

    EXPECT_TRUE(controllerSource.contains(QStringLiteral("isProductionSparseResult(resultRecordExtra)")));
    EXPECT_TRUE(controllerSource.contains(QStringLiteral("sparseResultBlockingReason(resultRecordExtra)")));
}

TEST(FeatureMatchSidecarTest, NativeRunnerWritesSidecarV2Indices)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/pipeline/FeatureMatchRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("feature_format_version")));
    EXPECT_TRUE(source.contains(QStringLiteral("matched_indices0")));
    EXPECT_TRUE(source.contains(QStringLiteral("matched_indices1")));
    EXPECT_TRUE(source.contains(QStringLiteral("matched_scores")));
}

TEST(FeatureMatchSidecarTest, FormalSfmRejectsLegacyCoordinateOnlyMatchCaches)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/core/pipeline/SFMService.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int compatibleStart = source.indexOf(QStringLiteral("existingMatchCompatible"));
    ASSERT_GE(compatibleStart, 0);
    const int candidateStart = source.indexOf(QStringLiteral("appendCandidatePair"), compatibleStart);
    ASSERT_GT(candidateStart, compatibleStart);
    const QString compatibilityBlock = source.mid(compatibleStart, candidateStart - compatibleStart);

    EXPECT_TRUE(compatibilityBlock.contains(QStringLiteral("feature_format_version")));
    EXPECT_TRUE(compatibilityBlock.contains(QStringLiteral("matched_indices0")));
    EXPECT_TRUE(compatibilityBlock.contains(QStringLiteral("matched_indices1")));
    EXPECT_TRUE(compatibilityBlock.contains(QStringLiteral("缺少 V2 特征索引")));
}

TEST(MatchViewerSidecarOrderTest, ReordersCachedPointsWhenDisplayOrderIsReversed)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("parseMatchFile(matchFile, imgA, imgB")));
    EXPECT_TRUE(source.contains(QStringLiteral("displayOrderForMatchFile")));
    EXPECT_TRUE(source.contains(QStringLiteral("appendSidecarMatchedPoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("reversed ? p1 : p0")));
}

TEST(MatchPairSelectorOverlapCandidatesTest, ListsOverlapPairsEvenWhenNoMatchFileExists)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchPairSelectorDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchPairSelectorDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("overlapCandidate")));
    EXPECT_TRUE(header.contains(QStringLiteral("loadOverlapCandidatesForImage")));
    EXPECT_TRUE(source.contains(QStringLiteral("vocabulary_overlap_pairs.json")));
    EXPECT_TRUE(source.contains(QStringLiteral("vocabulary_overlap_pairs.lis")));
    EXPECT_TRUE(source.contains(QStringLiteral("重叠候选")));
    EXPECT_TRUE(source.contains(QStringLiteral("matchFilePath.isEmpty()")));
}

TEST(MatchViewerEmptyMatchTest, CanOpenImagePairWithoutSparseMatchFile)
{
    const QString viewerSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/DualImageViewer.cpp"));
    const QString dialogSource = readProjectSourceFile(QStringLiteral("src/gui/dialogs/MatchViewerDialog.cpp"));
    ASSERT_FALSE(viewerSource.isEmpty());
    ASSERT_FALSE(dialogSource.isEmpty());

    EXPECT_TRUE(viewerSource.contains(QStringLiteral("matchFile.trimmed().isEmpty()")));
    EXPECT_TRUE(viewerSource.contains(QStringLiteral("QVector<QPointF>{}, QVector<QPointF>{}")));
    EXPECT_TRUE(dialogSource.contains(QStringLiteral("尚未生成匹配")));
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

TEST(DataTreeWidgetTest, ResultOnlyMetadataUpdateRefreshesDepthMapSection)
{
    DataTreeWidget tree;

    QJsonObject image;
    image[QStringLiteral("path")] = QStringLiteral("/tmp/ref_image_001.jpg");
    image[QStringLiteral("storage")] = QStringLiteral("reference");

    QJsonArray images;
    images.append(image);

    QJsonObject initialMeta;
    initialMeta[QStringLiteral("images")] = images;
    tree.loadFromJson(initialMeta);

    QJsonObject depthRecord;
    depthRecord[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
    depthRecord[QStringLiteral("depth_png")] = QStringLiteral("/tmp/mvs_output/depth_0.png");
    depthRecord[QStringLiteral("raw_depth_path")] = QStringLiteral("/tmp/mvs_output/depth_0.yml.gz");
    depthRecord[QStringLiteral("ref_image")] = QStringLiteral("/tmp/ref_image_001.jpg");
    depthRecord[QStringLiteral("grid_width")] = 6000;
    depthRecord[QStringLiteral("grid_height")] = 4000;

    QJsonArray depthResults;
    depthResults.append(depthRecord);

    QJsonObject resultOnlyMeta;
    resultOnlyMeta[QStringLiteral("depth_map_results")] = depthResults;
    tree.loadFromJson(resultOnlyMeta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *depthSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("深度图 (1)")))
        {
            depthSection = item;
            break;
        }
    }

    ASSERT_NE(depthSection, nullptr);
    ASSERT_EQ(depthSection->rowCount(), 1);
    EXPECT_EQ(depthSection->child(0, 0)->text(), QStringLiteral("depth_0.png  [6000x4000]"));
    EXPECT_EQ(depthSection->child(0, 1)->text(), QStringLiteral("/tmp/mvs_output/depth_0.png"));
}

TEST(DataTreeWidgetTest, ResourceRowsAreSortedByFileNameAscending)
{
    DataTreeWidget tree;

    QJsonArray images;
    for (const QString &path : {
             QStringLiteral("/tmp/images/image_010.jpg"),
             QStringLiteral("/tmp/images/image_002.jpg"),
             QStringLiteral("/tmp/images/image_001.jpg")})
    {
        QJsonObject image;
        image[QStringLiteral("path")] = path;
        image[QStringLiteral("storage")] = QStringLiteral("reference");
        images.append(image);
    }

    QJsonArray depthResults;
    for (const QString &path : {
             QStringLiteral("/tmp/mvs_output/depth_10.png"),
             QStringLiteral("/tmp/mvs_output/depth_2.png")})
    {
        QJsonObject depthRecord;
        depthRecord[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
        depthRecord[QStringLiteral("depth_png")] = path;
        depthRecord[QStringLiteral("raw_depth_path")] = path + QStringLiteral(".yml.gz");
        depthResults.append(depthRecord);
    }

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    meta[QStringLiteral("depth_map_results")] = depthResults;
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    auto findSection = [model](const QString &prefix) -> QStandardItem *
    {
        for (int row = 0; row < model->rowCount(); ++row)
        {
            QStandardItem *item = model->item(row, 0);
            if (item && item->text().startsWith(prefix))
            {
                return item;
            }
        }
        return nullptr;
    };

    QStandardItem *photoSection = findSection(QStringLiteral("照片 (3)"));
    ASSERT_NE(photoSection, nullptr);
    ASSERT_EQ(photoSection->rowCount(), 3);
    EXPECT_EQ(photoSection->child(0, 0)->text(), QStringLiteral("image_001.jpg"));
    EXPECT_EQ(photoSection->child(1, 0)->text(), QStringLiteral("image_002.jpg"));
    EXPECT_EQ(photoSection->child(2, 0)->text(), QStringLiteral("image_010.jpg"));

    QStandardItem *depthSection = findSection(QStringLiteral("深度图 (2)"));
    ASSERT_NE(depthSection, nullptr);
    ASSERT_EQ(depthSection->rowCount(), 2);
    EXPECT_EQ(depthSection->child(0, 0)->text(), QStringLiteral("depth_2.png"));
    EXPECT_EQ(depthSection->child(1, 0)->text(), QStringLiteral("depth_10.png"));
}

TEST(DataTreeWidgetTest, SelectionClickDoesNotActivateImageUntilItemActivation)
{
    DataTreeWidget tree;
    const QString imagePath = QStringLiteral("/tmp/aerial_image_001.jpg");

    QJsonObject image;
    image[QStringLiteral("path")] = imagePath;
    image[QStringLiteral("storage")] = QStringLiteral("reference");

    QJsonArray images;
    images.append(image);

    QJsonObject meta;
    meta[QStringLiteral("images")] = images;
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);

    QStandardItem *photoSection = nullptr;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *item = model->item(row, 0);
        if (item && item->text().startsWith(QStringLiteral("照片 (1)")))
        {
            photoSection = item;
            break;
        }
    }
    ASSERT_NE(photoSection, nullptr);
    ASSERT_EQ(photoSection->rowCount(), 1);

    const QModelIndex imageIndex = photoSection->child(0, 0)->index();
    ASSERT_TRUE(imageIndex.isValid());

    QSignalSpy imageSpy(&tree, &DataTreeWidget::imageActivated);
    QSignalSpy resourceSpy(&tree, &DataTreeWidget::resourceActivated);

    ASSERT_TRUE(QMetaObject::invokeMethod(view,
                                          "clicked",
                                          Qt::DirectConnection,
                                          Q_ARG(QModelIndex, imageIndex)));
    EXPECT_EQ(imageSpy.count(), 0);
    EXPECT_EQ(resourceSpy.count(), 0);

    ASSERT_TRUE(QMetaObject::invokeMethod(view,
                                          "activated",
                                          Qt::DirectConnection,
                                          Q_ARG(QModelIndex, imageIndex)));
    ASSERT_EQ(imageSpy.count(), 1);
    ASSERT_EQ(resourceSpy.count(), 1);
    EXPECT_EQ(imageSpy.takeFirst().at(0).toString(), imagePath);
    const QList<QVariant> resourceArgs = resourceSpy.takeFirst();
    ASSERT_EQ(resourceArgs.size(), 2);
    EXPECT_EQ(resourceArgs.at(0).toString(), QStringLiteral("照片"));
    EXPECT_EQ(resourceArgs.at(1).toString(), imagePath);
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
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
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
    stream.flush();
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
