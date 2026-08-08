#include "ProjectCameraSetupManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"
#include "ProjectCameraImportService.h"
#include "ProjectCameraInitialization.h"
#include "ProjectResultRecords.h"
#include "ProjectOpenGuard.h"
#include "ProjectSfmWorkflow.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "workflow/AerialTriangulationWorkflow.h"

#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QImageReader>
#include <QMessageBox>
#include <QPointer>
#include <QSet>
#include <QSize>
#include <QDateTime>

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>

using xjw::gui::project::existingCameraImages;
using xjw::gui::project::focalPixelsFromExif;
using xjw::gui::project::finalizeInitializedCameraPoses;
using xjw::gui::project::InitPoseFinalizeResult;
using xjw::gui::project::makeInitializedCameraMeta;
using xjw::gui::project::resolveInitTargets;
using xjw::gui::project::withPreparedCameras;

ProjectCameraSetupManager::ProjectCameraSetupManager(ProjectManager *owner,
                                                     ProjectData *projectData,
                                                     QWidget *parentWidget,
                                                     QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(projectData)
    , _parentWidget(parentWidget)
{
    connect(this, &ProjectCameraSetupManager::atProgressChanged,
            _owner, &ProjectManager::atProgressChanged);
    connect(this, &ProjectCameraSetupManager::atProgressFinished,
            _owner, &ProjectManager::atProgressFinished);
    connect(this, &ProjectCameraSetupManager::matchPairReady,
            _owner, &ProjectManager::matchPairReady);
}

// ── 单张相机导入 ──────────────────────────────────────────────────────────────
bool ProjectCameraSetupManager::importCameraForImage(const QString &imagePath)
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开或创建项目"))) return false;

    const QString dir = _owner->getLastUsedDir(QStringLiteral("camera_tsai"));
    const QString tsaiPath = QFileDialog::getOpenFileName(
        _parentWidget,
        QStringLiteral("选择相机文件 (.tsai)"),
        dir,
        QStringLiteral("Tsai相机文件 (*.tsai *.TSAI)")
    );
    if (tsaiPath.isEmpty()) return false;

    _owner->saveLastUsedDir(QStringLiteral("camera_tsai"), QFileInfo(tsaiPath).absolutePath());

    xjw::gui::project::SingleCameraImportResult importResult;
    const xjw::gui::project::SingleCameraImportStatus importStatus =
        xjw::gui::project::buildSingleCameraImport(imagePath, tsaiPath, &importResult);
    if (importStatus != xjw::gui::project::SingleCameraImportStatus::Ok) {
        QMessageBox::critical(_parentWidget, QStringLiteral("错误"), importResult.error);
        return false;
    }

    QString err;
    if (!_projectData->setImageCamera(importResult.imageAbsPath, importResult.cameraMeta, &err)) {
        QMessageBox::critical(_parentWidget, QStringLiteral("错误"), QStringLiteral("导入相机失败: %1").arg(err));
        return false;
    }

    QMessageBox::information(_parentWidget,
                             QStringLiteral("导入成功"),
                             QStringLiteral("已为影像 %1 导入相机文件。")
                                 .arg(QFileInfo(importResult.imageAbsPath).fileName()));
    return true;
}

// ── 批量相机导入 ──────────────────────────────────────────────────────────────
bool ProjectCameraSetupManager::importCamerasByFilenameBatch()
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开或创建项目"))) return false;

    const QString dir = _owner->getLastUsedDir(QStringLiteral("camera_tsai"));
    const QString folder = QFileDialog::getExistingDirectory(
        _parentWidget,
        QStringLiteral("选择包含 .tsai 的文件夹"),
        dir
    );
    if (folder.isEmpty()) return false;

    _owner->saveLastUsedDir(QStringLiteral("camera_tsai"), folder);

    const QStringList images = _projectData->getAllImages();
    xjw::gui::project::BatchCameraImportResult importResult;
    const xjw::gui::project::BatchCameraImportStatus importStatus =
        xjw::gui::project::buildBatchCameraImport(folder, images, &importResult);

    if (importStatus == xjw::gui::project::BatchCameraImportStatus::NoTsaiFiles)
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("提示"), QStringLiteral("所选文件夹中没有 .tsai 文件"));
        return false;
    }
    if (importStatus == xjw::gui::project::BatchCameraImportStatus::NoProjectImages)
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("提示"), QStringLiteral("项目中没有可匹配的影像"));
        return false;
    }

    for (const QString &errMsg : importResult.parseErrors)
    {
        LOG_WARN(errMsg);
    }

    if (importStatus == xjw::gui::project::BatchCameraImportStatus::NoImportable)
    {
        QMessageBox::warning(
            _parentWidget,
            QStringLiteral("提示"),
            QStringLiteral("没有可导入的相机文件。未匹配: %1，重名冲突: %2，解析失败: %3")
                .arg(importResult.unmatchedCount)
                .arg(importResult.ambiguousCount)
                .arg(importResult.parseFailedCount)
        );
        return false;
    }

    int updatedCount = 0;
    QString err;
    if (!_projectData->setImageCameras(importResult.cameraMetaByImage, &updatedCount, &err))
    {
        QMessageBox::critical(_parentWidget, QStringLiteral("错误"), QStringLiteral("批量导入失败: %1").arg(err));
        return false;
    }

    QMessageBox::information(
        _parentWidget,
        QStringLiteral("批量导入完成"),
        QStringLiteral("已写入 %1 条相机记录（未匹配: %2，重名冲突: %3，解析失败: %4）。")
            .arg(updatedCount)
            .arg(importResult.unmatchedCount)
            .arg(importResult.ambiguousCount)
            .arg(importResult.parseFailedCount)
    );
    return true;
}

// ── 相机内参初始化（EXIF / 默认焦距）────────────────────────────────────────
bool ProjectCameraSetupManager::initializeCamerasFromExifOrDefault(const QJsonObject &settings)
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开或创建项目"))) return false;

    QString targetErr;
    const QStringList targetImages = resolveInitTargets(_projectData, settings, &targetErr);
    if (targetImages.isEmpty())
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("初始化相机位姿"), targetErr);
        return false;
    }

    const bool overwriteExisting = settings.value(QStringLiteral("overwriteExisting")).toBool(false);
    const bool exifAuto = settings.value(QStringLiteral("exifAuto")).toBool(true);
    const double defaultFocalMm = settings.value(QStringLiteral("defaultFocal")).toDouble(50.0);
    const double sensorWidthMm = settings.value(QStringLiteral("sensorWidth")).toDouble(23.5);

    const QSet<QString> existing = existingCameraImages(_projectData->coreFilesMeta());

    QMap<QString, QJsonObject> cameraMetaByImage;
    int skippedExisting = 0;
    int exifCount = 0;
    int fallbackCount = 0;
    int invalidSizeCount = 0;

    for (const QString &imagePathRaw : targetImages)
    {
        const QString imagePath = QDir::cleanPath(QFileInfo(imagePathRaw).absoluteFilePath());
        if (!overwriteExisting && existing.contains(imagePath))
        {
            ++skippedExisting;
            continue;
        }

        QImageReader reader(imagePath);
        const QSize size = reader.size();
        if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
        {
            ++invalidSizeCount;
            continue;
        }

        QString focalSource = QStringLiteral("default_mm");
        double focalPx = defaultFocalMm / std::max(1e-9, sensorWidthMm) * size.width();
        if (exifAuto)
        {
            if (const auto exifPx = focalPixelsFromExif(imagePath, size, sensorWidthMm, &focalSource); exifPx.has_value())
            {
                focalPx = *exifPx;
                ++exifCount;
            }
            else
            {
                ++fallbackCount;
            }
        }
        else
        {
            ++fallbackCount;
        }

        QJsonObject camObj = makeInitializedCameraMeta(
            focalPx, focalPx,
            size.width() * 0.5, size.height() * 0.5,
            0.0, 0.0, 0.0, 0.0,
            QStringLiteral("init_from_exif_or_default"),
            QStringLiteral("none"),
            size);
        camObj[QStringLiteral("focal_source")] = focalSource;
        camObj[QStringLiteral("focal_px")] = focalPx;
        camObj[QStringLiteral("default_focal_mm")] = defaultFocalMm;
        camObj[QStringLiteral("sensor_width_mm")] = sensorWidthMm;
        cameraMetaByImage.insert(imagePath, camObj);
    }

    if (cameraMetaByImage.isEmpty())
    {
        QMessageBox::warning(
            _parentWidget,
            QStringLiteral("初始化相机位姿"),
            QStringLiteral("没有可写入的影像。已跳过已有相机: %1，尺寸无法读取: %2。")
                .arg(skippedExisting)
                .arg(invalidSizeCount));
        return false;
    }

    int updatedCount = 0;
    QString err;
    if (!_projectData->setImageCameras(cameraMetaByImage, &updatedCount, &err))
    {
        QMessageBox::critical(_parentWidget, QStringLiteral("错误"), QStringLiteral("写入相机初值失败: %1").arg(err));
        return false;
    }

    QMessageBox::information(
        _parentWidget,
        QStringLiteral("初始化完成"),
        QStringLiteral("已写入 %1 张影像的相机初值。EXIF 成功: %2，默认焦距回退: %3，跳过已有相机: %4，尺寸失败: %5。")
            .arg(updatedCount)
            .arg(exifCount)
            .arg(fallbackCount)
            .arg(skippedExisting)
            .arg(invalidSizeCount));
    return true;
}

// ── 相机内参初始化（手工内参）────────────────────────────────────────────────
bool ProjectCameraSetupManager::initializeCamerasFromIntrinsics(const QJsonObject &settings)
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开或创建项目"))) return false;

    QString targetErr;
    const QStringList targetImages = resolveInitTargets(_projectData, settings, &targetErr);
    if (targetImages.isEmpty())
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("初始化相机位姿"), targetErr);
        return false;
    }

    const bool overwriteExisting = settings.value(QStringLiteral("overwriteExisting")).toBool(false);
    const double fx = settings.value(QStringLiteral("fx")).toDouble(0.0);
    const double fy = settings.value(QStringLiteral("fy")).toDouble(0.0);
    const double cxInput = settings.value(QStringLiteral("cx")).toDouble(-1.0);
    const double cyInput = settings.value(QStringLiteral("cy")).toDouble(-1.0);
    const QString distortionModel = settings.value(QStringLiteral("distortionModel")).toString(QStringLiteral("Brown (k1, k2, p1, p2)"));

    if (fx <= 0.0 || fy <= 0.0)
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("初始化相机位姿"), QStringLiteral("fx/fy 必须大于 0。"));
        return false;
    }

    const QSet<QString> existing = existingCameraImages(_projectData->coreFilesMeta());

    QMap<QString, QJsonObject> cameraMetaByImage;
    int skippedExisting = 0;
    int autoPrincipalPointCount = 0;
    int invalidSizeCount = 0;

    double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;
    if (distortionModel.contains(QStringLiteral("径向")))
    {
        k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
        k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
    }
    else if (distortionModel.contains(QStringLiteral("Brown")))
    {
        k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
        k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
        p1 = settings.value(QStringLiteral("p1")).toDouble(0.0);
        p2 = settings.value(QStringLiteral("p2")).toDouble(0.0);
    }

    for (const QString &imagePathRaw : targetImages)
    {
        const QString imagePath = QDir::cleanPath(QFileInfo(imagePathRaw).absoluteFilePath());
        if (!overwriteExisting && existing.contains(imagePath))
        {
            ++skippedExisting;
            continue;
        }

        QImageReader reader(imagePath);
        const QSize size = reader.size();
        if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
        {
            ++invalidSizeCount;
            continue;
        }

        const double cx = (cxInput <= 0.0) ? (size.width() * 0.5) : cxInput;
        const double cy = (cyInput <= 0.0) ? (size.height() * 0.5) : cyInput;
        if (cxInput <= 0.0 || cyInput <= 0.0)
        {
            ++autoPrincipalPointCount;
        }

        QJsonObject camObj = makeInitializedCameraMeta(
            fx, fy, cx, cy,
            k1, k2, p1, p2,
            QStringLiteral("init_from_intrinsics"),
            distortionModel,
            size);
        camObj[QStringLiteral("focal_px")] = fx;
        camObj[QStringLiteral("focal_px_y")] = fy;
        cameraMetaByImage.insert(imagePath, camObj);
    }

    if (cameraMetaByImage.isEmpty())
    {
        QMessageBox::warning(
            _parentWidget,
            QStringLiteral("初始化相机位姿"),
            QStringLiteral("没有可写入的影像。已跳过已有相机: %1，尺寸无法读取: %2。")
                .arg(skippedExisting)
                .arg(invalidSizeCount));
        return false;
    }

    int updatedCount = 0;
    QString err;
    if (!_projectData->setImageCameras(cameraMetaByImage, &updatedCount, &err))
    {
        QMessageBox::critical(_parentWidget, QStringLiteral("错误"), QStringLiteral("写入相机初值失败: %1").arg(err));
        return false;
    }

    QMessageBox::information(
        _parentWidget,
        QStringLiteral("初始化完成"),
        QStringLiteral("已写入 %1 张影像的相机初值。跳过已有相机: %2，自动主点: %3，尺寸失败: %4。")
            .arg(updatedCount)
            .arg(skippedExisting)
            .arg(autoPrincipalPointCount)
            .arg(invalidSizeCount));
    return true;
}

// ── 使用 SFM 初始化相机位姿 ─────────────────────────────────────────────────
bool ProjectCameraSetupManager::initializeCameraPosesWithSFM(const QJsonObject &settings)
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开或创建项目"))) return false;

    const int mode = settings.value(QStringLiteral("mode")).toInt();
    if (mode != 0 && mode != 1)
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("初始化相机位姿"), QStringLiteral("当前模式不适用相对定向初始化。"));
        return false;
    }

    const QStringList allImages = _projectData ? _projectData->getAllImages() : QStringList{};
    if (allImages.size() < 2)
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("初始化相机位姿"), QStringLiteral("至少需要 2 张影像才能进行相对定向初始化。"));
        return false;
    }

    QString targetErr;
    const QStringList targetImages = resolveInitTargets(_projectData, settings, &targetErr);
    if (targetImages.isEmpty())
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("初始化相机位姿"), targetErr);
        return false;
    }

    const bool overwriteExisting = settings.value(QStringLiteral("overwriteExisting")).toBool(false);
    const QJsonObject baseMeta = _owner->coreProjectMeta();
    const QJsonObject fullMeta = _owner->currentMeta();
    const QSet<QString> existing = existingCameraImages(baseMeta);

    QMap<QString, QJsonObject> preparedCameras;
    int preparedCount = 0;
    int keptExistingCount = 0;
    int invalidSizeCount = 0;
    int exifCount = 0;
    int fallbackCount = 0;

    const double defaultFocalMm = settings.value(QStringLiteral("defaultFocal")).toDouble(50.0);
    const double sensorWidthMm = settings.value(QStringLiteral("sensorWidth")).toDouble(23.5);
    const bool exifAuto = settings.value(QStringLiteral("exifAuto")).toBool(true);
    const double fxInput = settings.value(QStringLiteral("fx")).toDouble(0.0);
    const double fyInput = settings.value(QStringLiteral("fy")).toDouble(0.0);
    const double cxInput = settings.value(QStringLiteral("cx")).toDouble(-1.0);
    const double cyInput = settings.value(QStringLiteral("cy")).toDouble(-1.0);
    const QString distortionModel = settings.value(QStringLiteral("distortionModel")).toString(QStringLiteral("Brown (k1, k2, p1, p2)"));

    double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;
    if (mode == 1)
    {
        if (fxInput <= 0.0 || fyInput <= 0.0)
        {
            QMessageBox::warning(_parentWidget, QStringLiteral("初始化相机位姿"), QStringLiteral("仅有内参模式下，fx/fy 必须大于 0。"));
            return false;
        }
        if (distortionModel.contains(QStringLiteral("径向")))
        {
            k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
            k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
        }
        else if (distortionModel.contains(QStringLiteral("Brown")))
        {
            k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
            k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
            p1 = settings.value(QStringLiteral("p1")).toDouble(0.0);
            p2 = settings.value(QStringLiteral("p2")).toDouble(0.0);
        }
    }

    for (const QString &imagePathRaw : allImages)
    {
        const QString imagePath = QDir::cleanPath(QFileInfo(imagePathRaw).absoluteFilePath());
        if (!overwriteExisting && existing.contains(imagePath))
        {
            ++keptExistingCount;
            continue;
        }

        QImageReader reader(imagePath);
        const QSize size = reader.size();
        if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
        {
            ++invalidSizeCount;
            continue;
        }

        QJsonObject camObj;
        if (mode == 0)
        {
            QString focalSource = QStringLiteral("default_mm");
            double focalPx = defaultFocalMm / std::max(1e-9, sensorWidthMm) * size.width();
            if (exifAuto)
            {
                if (const auto exifPx = focalPixelsFromExif(imagePath, size, sensorWidthMm, &focalSource); exifPx.has_value())
                {
                    focalPx = *exifPx;
                    ++exifCount;
                }
                else
                {
                    ++fallbackCount;
                }
            }
            else
            {
                ++fallbackCount;
            }

            camObj = makeInitializedCameraMeta(
                focalPx, focalPx,
                size.width() * 0.5, size.height() * 0.5,
                0.0, 0.0, 0.0, 0.0,
                QStringLiteral("init_pose_intrinsics_from_exif_or_default"),
                QStringLiteral("none"),
                size);
            camObj[QStringLiteral("focal_source")] = focalSource;
            camObj[QStringLiteral("default_focal_mm")] = defaultFocalMm;
            camObj[QStringLiteral("sensor_width_mm")] = sensorWidthMm;
        }
        else
        {
            const double cx = (cxInput <= 0.0) ? (size.width() * 0.5) : cxInput;
            const double cy = (cyInput <= 0.0) ? (size.height() * 0.5) : cyInput;
            camObj = makeInitializedCameraMeta(
                fxInput, fyInput,
                cx, cy,
                k1, k2, p1, p2,
                QStringLiteral("init_pose_intrinsics_manual"),
                distortionModel,
                size);
        }

        preparedCameras.insert(imagePath, camObj);
        ++preparedCount;
    }

    if (preparedCameras.isEmpty() && existing.isEmpty())
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("初始化相机位姿"),
                             QStringLiteral("没有可用于求解的内参初值。尺寸失败: %1。")
                                .arg(invalidSizeCount));
        return false;
    }

    const auto session = _owner->currentSessionContext();
    const QString projectPath = session.projectPath;
    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const QString outputDir = QDir(assetsDir).filePath(QStringLiteral("aerial_triangulation/init_pose_%1").arg(timestamp));
    QDir().mkpath(outputDir);

    xjw::aerial_triangulation::AerialTriangulationOptions workflowOptions;
    workflowOptions.images = allImages;
    workflowOptions.projectPath = projectPath;
    workflowOptions.projectMeta = withPreparedCameras(fullMeta, preparedCameras, overwriteExisting);
    workflowOptions.outputDir = outputDir;
    const int qualityLevel = settings.value(QStringLiteral("quality")).toInt(1);
    workflowOptions.quality = qualityLevel <= 0
        ? QStringLiteral("low")
        : (qualityLevel == 1 ? QStringLiteral("medium")
                             : (qualityLevel == 2 ? QStringLiteral("high")
                                                  : QStringLiteral("highest")));
    // 工程可能从其它机器复制而来；线程预算必须在本机启动时自动解析。
    workflowOptions.threads = 0;
    workflowOptions.matchingAlgorithmId =
        settings.value(QStringLiteral("algorithm_id"))
            .toString(QStringLiteral("sift_lightglue"))
            .trimmed()
            .toLower();
    workflowOptions.resetAlignment = false;
    workflowOptions.autoGenerateMissingMatches = false;

    LOG_INFO(QStringLiteral("初始化相机位姿: 使用影像匹配算法 %1")
        .arg(workflowOptions.matchingAlgorithmId));

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    _owner->setAtCancelFlag(cancelFlag);
    workflowOptions.cancelFlag = cancelFlag;

    QPointer<ProjectCameraSetupManager> self(this);
    QPointer<ProjectManager> ownerGuard(_owner);
    workflowOptions.progressFn = [self, ownerGuard, session](const QString &stage, int pct)
    {
        if (!self || !ownerGuard || !ownerGuard->isCurrentSession(session))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(self, [ownerGuard, session, stage, pct](ProjectCameraSetupManager *manager)
        {
            if (!ownerGuard || !ownerGuard->isCurrentSession(session))
            {
                return;
            }
            emit manager->atProgressChanged(stage, pct);
        });
    };

    workflowOptions.pairMatchedFn = [self, ownerGuard, session](const QString &img0, const QString &img1,
                                                         const QString &matchPath, int numMatches)
    {
        if (!self || !ownerGuard || !ownerGuard->isCurrentSession(session))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(self,
                                     [ownerGuard, session, img0, img1, matchPath, numMatches](
                                         ProjectCameraSetupManager *manager)
        {
            if (!ownerGuard || !ownerGuard->isCurrentSession(session))
            {
                return;
            }
            emit manager->matchPairReady(img0, img1, matchPath, numMatches);
        });
    };

    const QSet<QString> targetSet = [&]()
    {
        QSet<QString> out;
        for (const QString &p : targetImages)
        {
            out.insert(QDir::cleanPath(QFileInfo(p).absoluteFilePath()));
        }
        return out;
    }();

    emit atProgressChanged(QStringLiteral("启动初始化相机位姿..."), 0);

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [workflowOptions]() mutable
        {
            return xjw::aerial_triangulation::AerialTriangulationWorkflow::run(workflowOptions);
        },
        [outputDir,
         allImages,
         targetSet,
         existing,
         overwriteExisting,
         preparedCount,
         keptExistingCount,
         invalidSizeCount,
         exifCount,
         fallbackCount,
         ownerGuard,
         session](ProjectCameraSetupManager *manager,
                      xjw::gui::tasks::TaskOutcome<
                          xjw::aerial_triangulation::AerialTriangulationResult> outcome) mutable
        {
            if (!ownerGuard || !ownerGuard->isCurrentSession(session))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                emit manager->atProgressFinished(false);
                QMessageBox::warning(manager->_parentWidget,
                                     QStringLiteral("初始化相机位姿"),
                                     outcome.errorMessage);
                return;
            }

            auto workflowResult = std::move(*outcome.value);

            const xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result =
                workflowResult.reconstructionResult;
            if (!result.success)
            {
                emit manager->atProgressFinished(false);
                QMessageBox::warning(
                    manager->_parentWidget,
                    QStringLiteral("初始化相机位姿"),
                    result.errorMessage.isEmpty() ? QStringLiteral("相对定向 / SFM 初始化失败") : result.errorMessage);
                return;
            }

            manager->_owner->appendImageMatchResults(
                xjw::gui::project::makeImageMatchResultRecords(workflowResult.tiePointResult));

            const InitPoseFinalizeResult finalizeResult = finalizeInitializedCameraPoses(manager->_projectData,
                                                                                        result,
                                                                                        targetSet,
                                                                                        existing,
                                                                                        overwriteExisting,
                                                                                        allImages,
                                                                                        outputDir);
            if (!finalizeResult.success)
            {
                emit manager->atProgressFinished(false);
                QMessageBox::critical(manager->_parentWidget,
                                      QStringLiteral("初始化相机位姿"),
                                      finalizeResult.errorMessage);
                return;
            }

            LOG_INFO(QStringLiteral("初始化相机位姿完成: 注册=%1 点数=%2 回写=%3")
                .arg(result.numRegisteredImages)
                .arg(result.numPoints3D)
                .arg(finalizeResult.updatedCameraCount));

            emit manager->atProgressFinished(true);
            QMessageBox::information(
                manager->_parentWidget,
                QStringLiteral("初始化相机位姿"),
                QStringLiteral("初始化完成。注册影像: %1，三维点: %2，回写相机: %3。\n"
                               "内参初值准备: %4，保留已有相机: %5，尺寸失败: %6，EXIF 成功: %7，默认焦距回退: %8。")
                    .arg(result.numRegisteredImages)
                    .arg(result.numPoints3D)
                    .arg(finalizeResult.updatedCameraCount)
                    .arg(preparedCount)
                    .arg(keptExistingCount)
                    .arg(invalidSizeCount)
                    .arg(exifCount)
                    .arg(fallbackCount));
        });

    return true;
}
