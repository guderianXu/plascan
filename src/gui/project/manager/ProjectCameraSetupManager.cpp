#include "ProjectCameraSetupManager.h"

#include "GuiTaskRunner.h"
#include "Logger.h"
#include "ProjectCameraImportService.h"
#include "ProjectCameraInitialization.h"
#include "ProjectResultRecords.h"
#include "ProjectSfmWorkflow.h"
#include "project/ProjectIO.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "workflow/AerialTriangulationWorkflow.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QImageReader>
#include <QPointer>
#include <QSet>
#include <QSize>

#include <algorithm>
#include <atomic>
#include <utility>

using xjw::gui::project::existingCameraImages;
using xjw::gui::project::finalizeInitializedCameraPoses;
using xjw::gui::project::focalPixelsFromExif;
using xjw::gui::project::InitPoseFinalizeResult;
using xjw::gui::project::makeInitializedCameraMeta;
using xjw::gui::project::resolveInitTargets;
using xjw::gui::project::withPreparedCameras;

namespace
{

    constexpr QDir::Filters kDialogFilters = QDir::AllEntries | QDir::Hidden | QDir::AllDirs | QDir::NoDotAndDotDot;

    QSet<QString> normalizedPathSet(const QStringList& paths)
    {
        QSet<QString> result;
        for (const QString& path : paths)
        {
            result.insert(QDir::cleanPath(QFileInfo(path).absoluteFilePath()));
        }
        return result;
    }

} // namespace

ProjectCameraSetupManager::ProjectCameraSetupManager(xjw::gui::project::ProjectSession* session,
                                                     ProjectUiMessageAdapter* messages,
                                                     QObject* parent)
    : QObject(parent), _session(session), _messages(messages),
      _sfmRunner([](xjw::aerial_triangulation::AerialTriangulationOptions options)
                 { return xjw::aerial_triangulation::AerialTriangulationWorkflow::run(options); })
{
}

ProjectCameraSetupManager::~ProjectCameraSetupManager()
{
    if (_sfmContext.cancelFlag)
    {
        _sfmContext.cancelFlag->store(true, std::memory_order_relaxed);
    }
    waitForActiveTask();
}

bool ProjectCameraSetupManager::requireProject(const QString& message) const
{
    if (_session && _session->hasProject())
    {
        return true;
    }
    if (_messages)
    {
        _messages->warning(nullptr, QStringLiteral("提示"), message);
    }
    return false;
}

bool ProjectCameraSetupManager::contextMatches(const xjw::gui::project::ProjectTaskContext& taskContext,
                                               bool requireCurrent,
                                               bool allowCancelled) const
{
    if (!taskContext.cancelFlag || _sfmContext.taskId != taskContext.taskId ||
        _sfmContext.cancelFlag != taskContext.cancelFlag)
    {
        return false;
    }
    if (!allowCancelled && taskContext.cancelFlag->load(std::memory_order_relaxed))
    {
        return false;
    }
    return !requireCurrent || (_session && _session->isCurrent(taskContext.session));
}

void ProjectCameraSetupManager::completeTask(const xjw::gui::project::ProjectTaskContext& taskContext, bool success)
{
    if (!contextMatches(taskContext, false, true))
    {
        return;
    }
    const bool terminal_success = success && _session && _session->isCurrent(taskContext.session) &&
                                  !taskContext.cancelFlag->load(std::memory_order_relaxed);
    _sfmContext = {};
    emit atProgressFinished(terminal_success);
}

void ProjectCameraSetupManager::setDirectoryAccessors(
    std::function<QString(const QString& key)> getLastDir,
    std::function<void(const QString& key, const QString& dir)> saveLastDir)
{
    _getLastDir = std::move(getLastDir);
    _saveLastDir = std::move(saveLastDir);
}

QString ProjectCameraSetupManager::readLastDir(const QString& key) const
{
    if (_getLastDir)
    {
        const QString directory = _getLastDir(key);
        if (!directory.isEmpty())
        {
            return directory;
        }
    }
    return QDir::homePath();
}

void ProjectCameraSetupManager::writeLastDir(const QString& key, const QString& dir) const
{
    if (_saveLastDir)
    {
        _saveLastDir(key, dir);
    }
}

void ProjectCameraSetupManager::setSfmRunnerForTesting(SfmRunner runner)
{
    _sfmRunner = std::move(runner);
}

void ProjectCameraSetupManager::pruneFinishedFutures()
{
    for (auto it = _futures.begin(); it != _futures.end();)
    {
        if (it->isFinished())
        {
            it = _futures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void ProjectCameraSetupManager::trackFuture(QFuture<void> future)
{
    pruneFinishedFutures();
    _futures.push_back(std::move(future));
}

void ProjectCameraSetupManager::trackFutureForTesting(QFuture<void> future)
{
    trackFuture(std::move(future));
}

void ProjectCameraSetupManager::waitForActiveTask()
{
    for (QFuture<void>& future : _futures)
    {
        future.waitForFinished();
    }
    _futures.clear();
}

bool ProjectCameraSetupManager::isRunning() const noexcept
{
    return _sfmContext.cancelFlag != nullptr;
}

bool ProjectCameraSetupManager::hasPendingWork() const noexcept
{
    return std::any_of(_futures.cbegin(),
                       _futures.cend(),
                       [](const QFuture<void>& future) { return future.isValid() && !future.isFinished(); });
}

void ProjectCameraSetupManager::cancelActiveTask(const xjw::gui::project::ProjectTaskContext& taskContext)
{
    if (contextMatches(taskContext, false, true) && taskContext.cancelFlag)
    {
        taskContext.cancelFlag->store(true, std::memory_order_relaxed);
    }
}

bool ProjectCameraSetupManager::importCameraForImage(const QString& imagePath)
{
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages)
    {
        return false;
    }

    const xjw::gui::project::ProjectSessionContext session_context = _session->context();
    const UiDialogResult selected = _messages->selectOpenFile(nullptr,
                                                              QStringLiteral("选择相机文件 (.tsai)"),
                                                              readLastDir(QStringLiteral("camera_tsai")),
                                                              QStringLiteral("Tsai相机文件 (*.tsai *.TSAI)"),
                                                              kDialogFilters);
    if (!selected.accepted || selected.text.isEmpty() || !_session->isCurrent(session_context))
    {
        return false;
    }

    writeLastDir(QStringLiteral("camera_tsai"), QFileInfo(selected.text).absolutePath());

    xjw::gui::project::SingleCameraImportResult import_result;
    const xjw::gui::project::SingleCameraImportStatus import_status =
        xjw::gui::project::buildSingleCameraImport(imagePath, selected.text, &import_result);
    if (import_status != xjw::gui::project::SingleCameraImportStatus::Ok)
    {
        _messages->critical(nullptr, QStringLiteral("错误"), import_result.error);
        return false;
    }

    int updated_count = 0;
    QString error;
    if (!_session->setCameraInstances(
            session_context, {{import_result.imageAbsPath, import_result.cameraMeta}}, &updated_count, &error))
    {
        if (_session->isCurrent(session_context))
        {
            _messages->critical(nullptr, QStringLiteral("错误"), QStringLiteral("导入相机失败: %1").arg(error));
        }
        return false;
    }

    if (!_session->isCurrent(session_context))
    {
        return false;
    }
    _messages->information(
        nullptr,
        QStringLiteral("导入成功"),
        QStringLiteral("已为影像 %1 导入相机文件。").arg(QFileInfo(import_result.imageAbsPath).fileName()));
    return true;
}

bool ProjectCameraSetupManager::importCamerasByFilenameBatch()
{
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages)
    {
        return false;
    }

    const xjw::gui::project::ProjectSessionContext session_context = _session->context();
    const UiDialogResult selected = _messages->selectDirectory(
        nullptr, QStringLiteral("选择包含 .tsai 的文件夹"), readLastDir(QStringLiteral("camera_tsai")), kDialogFilters);
    if (!selected.accepted || selected.text.isEmpty() || !_session->isCurrent(session_context))
    {
        return false;
    }

    writeLastDir(QStringLiteral("camera_tsai"), selected.text);
    const QStringList images = _session->allImages();
    xjw::gui::project::BatchCameraImportResult import_result;
    const xjw::gui::project::BatchCameraImportStatus import_status =
        xjw::gui::project::buildBatchCameraImport(selected.text, images, &import_result);

    if (import_status == xjw::gui::project::BatchCameraImportStatus::NoTsaiFiles)
    {
        _messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("所选文件夹中没有 .tsai 文件"));
        return false;
    }
    if (import_status == xjw::gui::project::BatchCameraImportStatus::NoProjectImages)
    {
        _messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("项目中没有可匹配的影像"));
        return false;
    }
    for (const QString& parse_error : import_result.parseErrors)
    {
        LOG_WARN(parse_error);
    }
    if (import_status == xjw::gui::project::BatchCameraImportStatus::NoImportable)
    {
        _messages->warning(nullptr,
                           QStringLiteral("提示"),
                           QStringLiteral("没有可导入的相机文件。未匹配: %1，重名冲突: %2，解析失败: %3")
                               .arg(import_result.unmatchedCount)
                               .arg(import_result.ambiguousCount)
                               .arg(import_result.parseFailedCount));
        return false;
    }

    int updated_count = 0;
    QString error;
    if (!_session->setCameraInstances(session_context, import_result.cameraMetaByImage, &updated_count, &error))
    {
        if (_session->isCurrent(session_context))
        {
            _messages->critical(nullptr, QStringLiteral("错误"), QStringLiteral("批量导入失败: %1").arg(error));
        }
        return false;
    }
    if (!_session->isCurrent(session_context))
    {
        return false;
    }

    _messages->information(nullptr,
                           QStringLiteral("批量导入完成"),
                           QStringLiteral("已写入 %1 条相机记录（未匹配: %2，重名冲突: %3，解析失败: %4）。")
                               .arg(updated_count)
                               .arg(import_result.unmatchedCount)
                               .arg(import_result.ambiguousCount)
                               .arg(import_result.parseFailedCount));
    return true;
}

bool ProjectCameraSetupManager::initializeCamerasFromExifOrDefault(const QJsonObject& settings)
{
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages)
    {
        return false;
    }

    const xjw::gui::project::ProjectSessionContext session_context = _session->context();
    const QStringList all_images = _session->allImages();
    QString target_error;
    const QStringList target_images = resolveInitTargets(all_images, settings, &target_error);
    if (target_images.isEmpty())
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), target_error);
        return false;
    }

    const bool overwrite_existing = settings.value(QStringLiteral("overwriteExisting")).toBool(false);
    const bool exif_auto = settings.value(QStringLiteral("exifAuto")).toBool(true);
    const double default_focal_mm = settings.value(QStringLiteral("defaultFocal")).toDouble(50.0);
    const double sensor_width_mm = settings.value(QStringLiteral("sensorWidth")).toDouble(23.5);
    const QSet<QString> existing = existingCameraImages(_session->coreMetadata());

    QMap<QString, QJsonObject> cameras;
    int skipped_existing = 0;
    int exif_count = 0;
    int fallback_count = 0;
    int invalid_size_count = 0;
    for (const QString& raw_path : target_images)
    {
        const QString image_path = QDir::cleanPath(QFileInfo(raw_path).absoluteFilePath());
        if (!overwrite_existing && existing.contains(image_path))
        {
            ++skipped_existing;
            continue;
        }

        QImageReader reader(image_path);
        const QSize size = reader.size();
        if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
        {
            ++invalid_size_count;
            continue;
        }

        QString focal_source = QStringLiteral("default_mm");
        double focal_px = default_focal_mm / std::max(1e-9, sensor_width_mm) * size.width();
        if (exif_auto)
        {
            if (const auto exif_px = focalPixelsFromExif(image_path, size, sensor_width_mm, &focal_source);
                exif_px.has_value())
            {
                focal_px = *exif_px;
                ++exif_count;
            }
            else
            {
                ++fallback_count;
            }
        }
        else
        {
            ++fallback_count;
        }

        QJsonObject camera = makeInitializedCameraMeta(focal_px,
                                                       focal_px,
                                                       size.width() * 0.5,
                                                       size.height() * 0.5,
                                                       0.0,
                                                       0.0,
                                                       0.0,
                                                       0.0,
                                                       QStringLiteral("init_from_exif_or_default"),
                                                       QStringLiteral("none"),
                                                       size);
        camera[QStringLiteral("focal_source")] = focal_source;
        camera[QStringLiteral("focal_px")] = focal_px;
        camera[QStringLiteral("default_focal_mm")] = default_focal_mm;
        camera[QStringLiteral("sensor_width_mm")] = sensor_width_mm;
        cameras.insert(image_path, camera);
    }

    if (cameras.isEmpty())
    {
        _messages->warning(nullptr,
                           QStringLiteral("初始化相机位姿"),
                           QStringLiteral("没有可写入的影像。已跳过已有相机: %1，尺寸无法读取: %2。")
                               .arg(skipped_existing)
                               .arg(invalid_size_count));
        return false;
    }

    int updated_count = 0;
    QString error;
    if (!_session->setCameraInstances(session_context, cameras, &updated_count, &error))
    {
        if (_session->isCurrent(session_context))
        {
            _messages->critical(nullptr, QStringLiteral("错误"), QStringLiteral("写入相机初值失败: %1").arg(error));
        }
        return false;
    }
    if (!_session->isCurrent(session_context))
    {
        return false;
    }

    _messages->information(
        nullptr,
        QStringLiteral("初始化完成"),
        QStringLiteral("已写入 %1 张影像的相机初值。EXIF 成功: %2，默认焦距回退: %3，跳过已有相机: %4，尺寸失败: %5。")
            .arg(updated_count)
            .arg(exif_count)
            .arg(fallback_count)
            .arg(skipped_existing)
            .arg(invalid_size_count));
    return true;
}

bool ProjectCameraSetupManager::initializeCamerasFromIntrinsics(const QJsonObject& settings)
{
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages)
    {
        return false;
    }

    const xjw::gui::project::ProjectSessionContext session_context = _session->context();
    QString target_error;
    const QStringList target_images = resolveInitTargets(_session->allImages(), settings, &target_error);
    if (target_images.isEmpty())
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), target_error);
        return false;
    }

    const bool overwrite_existing = settings.value(QStringLiteral("overwriteExisting")).toBool(false);
    const double fx = settings.value(QStringLiteral("fx")).toDouble(0.0);
    const double fy = settings.value(QStringLiteral("fy")).toDouble(0.0);
    const double cx_input = settings.value(QStringLiteral("cx")).toDouble(-1.0);
    const double cy_input = settings.value(QStringLiteral("cy")).toDouble(-1.0);
    const QString distortion_model =
        settings.value(QStringLiteral("distortionModel")).toString(QStringLiteral("Brown (k1, k2, p1, p2)"));
    if (fx <= 0.0 || fy <= 0.0)
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("fx/fy 必须大于 0。"));
        return false;
    }

    const QSet<QString> existing = existingCameraImages(_session->coreMetadata());
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    if (distortion_model.contains(QStringLiteral("径向")))
    {
        k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
        k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
    }
    else if (distortion_model.contains(QStringLiteral("Brown")))
    {
        k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
        k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
        p1 = settings.value(QStringLiteral("p1")).toDouble(0.0);
        p2 = settings.value(QStringLiteral("p2")).toDouble(0.0);
    }

    QMap<QString, QJsonObject> cameras;
    int skipped_existing = 0;
    int auto_principal_point_count = 0;
    int invalid_size_count = 0;
    for (const QString& raw_path : target_images)
    {
        const QString image_path = QDir::cleanPath(QFileInfo(raw_path).absoluteFilePath());
        if (!overwrite_existing && existing.contains(image_path))
        {
            ++skipped_existing;
            continue;
        }

        QImageReader reader(image_path);
        const QSize size = reader.size();
        if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
        {
            ++invalid_size_count;
            continue;
        }

        const double cx = cx_input <= 0.0 ? size.width() * 0.5 : cx_input;
        const double cy = cy_input <= 0.0 ? size.height() * 0.5 : cy_input;
        if (cx_input <= 0.0 || cy_input <= 0.0)
        {
            ++auto_principal_point_count;
        }
        QJsonObject camera = makeInitializedCameraMeta(
            fx, fy, cx, cy, k1, k2, p1, p2, QStringLiteral("init_from_intrinsics"), distortion_model, size);
        camera[QStringLiteral("focal_px")] = fx;
        camera[QStringLiteral("focal_px_y")] = fy;
        cameras.insert(image_path, camera);
    }

    if (cameras.isEmpty())
    {
        _messages->warning(nullptr,
                           QStringLiteral("初始化相机位姿"),
                           QStringLiteral("没有可写入的影像。已跳过已有相机: %1，尺寸无法读取: %2。")
                               .arg(skipped_existing)
                               .arg(invalid_size_count));
        return false;
    }

    int updated_count = 0;
    QString error;
    if (!_session->setCameraInstances(session_context, cameras, &updated_count, &error))
    {
        if (_session->isCurrent(session_context))
        {
            _messages->critical(nullptr, QStringLiteral("错误"), QStringLiteral("写入相机初值失败: %1").arg(error));
        }
        return false;
    }
    if (!_session->isCurrent(session_context))
    {
        return false;
    }

    _messages->information(nullptr,
                           QStringLiteral("初始化完成"),
                           QStringLiteral("已写入 %1 张影像的相机初值。跳过已有相机: %2，自动主点: %3，尺寸失败: %4。")
                               .arg(updated_count)
                               .arg(skipped_existing)
                               .arg(auto_principal_point_count)
                               .arg(invalid_size_count));
    return true;
}

bool ProjectCameraSetupManager::initializeCameraPosesWithSFM(const QJsonObject& settings,
                                                             const xjw::gui::project::ProjectTaskContext& taskContext)
{
    _sfmContext = taskContext;
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages || !contextMatches(taskContext))
    {
        _sfmContext = {};
        return false;
    }

    const int mode = settings.value(QStringLiteral("mode")).toInt();
    if (mode != 0 && mode != 1)
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("当前模式不适用相对定向初始化。"));
        _sfmContext = {};
        return false;
    }

    const QStringList all_images = _session->allImages();
    if (all_images.size() < 2)
    {
        _messages->warning(
            nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("至少需要 2 张影像才能进行相对定向初始化。"));
        _sfmContext = {};
        return false;
    }

    QString target_error;
    const QStringList target_images = resolveInitTargets(all_images, settings, &target_error);
    if (target_images.isEmpty())
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), target_error);
        _sfmContext = {};
        return false;
    }

    const bool overwrite_existing = settings.value(QStringLiteral("overwriteExisting")).toBool(false);
    const QJsonObject base_meta = _session->coreMetadata();
    const QJsonObject full_meta = _session->metadata();
    const QSet<QString> existing = existingCameraImages(base_meta);

    QMap<QString, QJsonObject> prepared_cameras;
    int prepared_count = 0;
    int kept_existing_count = 0;
    int invalid_size_count = 0;
    int exif_count = 0;
    int fallback_count = 0;

    const double default_focal_mm = settings.value(QStringLiteral("defaultFocal")).toDouble(50.0);
    const double sensor_width_mm = settings.value(QStringLiteral("sensorWidth")).toDouble(23.5);
    const bool exif_auto = settings.value(QStringLiteral("exifAuto")).toBool(true);
    const double fx_input = settings.value(QStringLiteral("fx")).toDouble(0.0);
    const double fy_input = settings.value(QStringLiteral("fy")).toDouble(0.0);
    const double cx_input = settings.value(QStringLiteral("cx")).toDouble(-1.0);
    const double cy_input = settings.value(QStringLiteral("cy")).toDouble(-1.0);
    const QString distortion_model =
        settings.value(QStringLiteral("distortionModel")).toString(QStringLiteral("Brown (k1, k2, p1, p2)"));

    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    if (mode == 1)
    {
        if (fx_input <= 0.0 || fy_input <= 0.0)
        {
            _messages->warning(
                nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("仅有内参模式下，fx/fy 必须大于 0。"));
            _sfmContext = {};
            return false;
        }
        if (distortion_model.contains(QStringLiteral("径向")))
        {
            k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
            k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
        }
        else if (distortion_model.contains(QStringLiteral("Brown")))
        {
            k1 = settings.value(QStringLiteral("k1")).toDouble(0.0);
            k2 = settings.value(QStringLiteral("k2")).toDouble(0.0);
            p1 = settings.value(QStringLiteral("p1")).toDouble(0.0);
            p2 = settings.value(QStringLiteral("p2")).toDouble(0.0);
        }
    }

    for (const QString& raw_path : all_images)
    {
        const QString image_path = QDir::cleanPath(QFileInfo(raw_path).absoluteFilePath());
        if (!overwrite_existing && existing.contains(image_path))
        {
            ++kept_existing_count;
            continue;
        }

        QImageReader reader(image_path);
        const QSize size = reader.size();
        if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
        {
            ++invalid_size_count;
            continue;
        }

        QJsonObject camera;
        if (mode == 0)
        {
            QString focal_source = QStringLiteral("default_mm");
            double focal_px = default_focal_mm / std::max(1e-9, sensor_width_mm) * size.width();
            if (exif_auto)
            {
                if (const auto exif_px = focalPixelsFromExif(image_path, size, sensor_width_mm, &focal_source);
                    exif_px.has_value())
                {
                    focal_px = *exif_px;
                    ++exif_count;
                }
                else
                {
                    ++fallback_count;
                }
            }
            else
            {
                ++fallback_count;
            }
            camera = makeInitializedCameraMeta(focal_px,
                                               focal_px,
                                               size.width() * 0.5,
                                               size.height() * 0.5,
                                               0.0,
                                               0.0,
                                               0.0,
                                               0.0,
                                               QStringLiteral("init_pose_intrinsics_from_exif_or_default"),
                                               QStringLiteral("none"),
                                               size);
            camera[QStringLiteral("focal_source")] = focal_source;
            camera[QStringLiteral("default_focal_mm")] = default_focal_mm;
            camera[QStringLiteral("sensor_width_mm")] = sensor_width_mm;
        }
        else
        {
            const double cx = cx_input <= 0.0 ? size.width() * 0.5 : cx_input;
            const double cy = cy_input <= 0.0 ? size.height() * 0.5 : cy_input;
            camera = makeInitializedCameraMeta(fx_input,
                                               fy_input,
                                               cx,
                                               cy,
                                               k1,
                                               k2,
                                               p1,
                                               p2,
                                               QStringLiteral("init_pose_intrinsics_manual"),
                                               distortion_model,
                                               size);
        }
        prepared_cameras.insert(image_path, camera);
        ++prepared_count;
    }

    if (prepared_cameras.isEmpty() && existing.isEmpty())
    {
        _messages->warning(nullptr,
                           QStringLiteral("初始化相机位姿"),
                           QStringLiteral("没有可用于求解的内参初值。尺寸失败: %1。").arg(invalid_size_count));
        _sfmContext = {};
        return false;
    }

    const QString project_path = taskContext.session.projectPath;
    const QString assets_dir = xjw::common::project::ProjectIO::projectAssetsDir(project_path);
    const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const QString output_dir =
        QDir(assets_dir).filePath(QStringLiteral("aerial_triangulation/init_pose_%1").arg(timestamp));
    QDir().mkpath(output_dir);

    xjw::aerial_triangulation::AerialTriangulationOptions workflow_options;
    workflow_options.images = all_images;
    workflow_options.projectPath = project_path;
    workflow_options.projectMeta = withPreparedCameras(full_meta, prepared_cameras, overwrite_existing);
    workflow_options.outputDir = output_dir;
    const int quality_level = settings.value(QStringLiteral("quality")).toInt(1);
    if (quality_level <= 0)
    {
        workflow_options.quality = QStringLiteral("low");
    }
    else if (quality_level == 1)
    {
        workflow_options.quality = QStringLiteral("medium");
    }
    else if (quality_level == 2)
    {
        workflow_options.quality = QStringLiteral("high");
    }
    else
    {
        workflow_options.quality = QStringLiteral("highest");
    }
    workflow_options.threads = 0;
    workflow_options.matchingAlgorithmId =
        settings.value(QStringLiteral("algorithm_id")).toString(QStringLiteral("plamatch_hct")).trimmed().toLower();
    workflow_options.resetAlignment = false;
    workflow_options.autoGenerateMissingMatches = false;
    workflow_options.cancelFlag = taskContext.cancelFlag;

    LOG_INFO(QStringLiteral("初始化相机位姿: 使用影像匹配算法 %1").arg(workflow_options.matchingAlgorithmId));

    QPointer<ProjectCameraSetupManager> self(this);
    workflow_options.progressFn = [self, taskContext](const QString& stage, int percent)
    {
        if (!self || !self->contextMatches(taskContext))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(self,
                                     [taskContext, stage, percent](ProjectCameraSetupManager* manager)
                                     {
                                         if (manager->contextMatches(taskContext))
                                         {
                                             emit manager->atProgressChanged(stage, percent);
                                         }
                                     });
    };
    workflow_options.pairMatchedFn =
        [self, taskContext](const QString& img0, const QString& img1, const QString& match_path, int num_matches)
    {
        if (!self || !self->contextMatches(taskContext))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(
            self,
            [taskContext, img0, img1, match_path, num_matches](ProjectCameraSetupManager* manager)
            {
                if (manager->contextMatches(taskContext))
                {
                    emit manager->matchPairReady(img0, img1, match_path, num_matches);
                }
            });
    };

    const QSet<QString> target_set = normalizedPathSet(target_images);
    const SfmRunner runner = _sfmRunner;
    emit atProgressChanged(QStringLiteral("启动初始化相机位姿..."), 0);
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [runner, workflow_options = std::move(workflow_options)]() mutable
        { return runner(std::move(workflow_options)); },
        [taskContext,
         output_dir,
         all_images,
         target_set,
         existing,
         overwrite_existing,
         full_meta,
         prepared_count,
         kept_existing_count,
         invalid_size_count,
         exif_count,
         fallback_count](
            ProjectCameraSetupManager* manager,
            xjw::gui::tasks::TaskOutcome<xjw::aerial_triangulation::AerialTriangulationResult> outcome) mutable
        {
            if (!manager->contextMatches(taskContext, false, true))
            {
                return;
            }
            if (!manager->_session || !manager->_session->isCurrent(taskContext.session) ||
                taskContext.cancelFlag->load(std::memory_order_relaxed))
            {
                manager->completeTask(taskContext, false);
                return;
            }
            if (!outcome.succeeded())
            {
                manager->completeTask(taskContext, false);
                manager->_messages->warning(nullptr, QStringLiteral("初始化相机位姿"), outcome.errorMessage);
                return;
            }

            xjw::aerial_triangulation::AerialTriangulationResult workflow_result = std::move(*outcome.value);
            const xjw::aerial_triangulation::AerialTriangulationReconstructionResult& result =
                workflow_result.reconstructionResult;
            if (!result.success)
            {
                manager->completeTask(taskContext, false);
                manager->_messages->warning(nullptr,
                                            QStringLiteral("初始化相机位姿"),
                                            result.errorMessage.isEmpty() ? QStringLiteral("相对定向 / SFM 初始化失败")
                                                                          : result.errorMessage);
                return;
            }

            const QVector<ProjectImageMatchResultRecord> match_records =
                xjw::gui::project::makeImageMatchResultRecords(workflow_result.tiePointResult);
            QString error;
            if (!match_records.isEmpty() &&
                !manager->_session->appendImageMatchResults(taskContext.session, match_records, &error))
            {
                manager->completeTask(taskContext, false);
                if (manager->_session && manager->_session->isCurrent(taskContext.session))
                {
                    manager->_messages->critical(nullptr,
                                                 QStringLiteral("初始化相机位姿"),
                                                 QStringLiteral("SFM 已完成，但写入匹配结果失败: %1").arg(error));
                }
                return;
            }
            for (const ProjectImageMatchResultRecord& record : match_records)
            {
                if (!manager->contextMatches(taskContext))
                {
                    manager->completeTask(taskContext, false);
                    return;
                }
                emit manager->imageMatchResultAppended(record.image);
            }

            const InitPoseFinalizeResult finalize_result = finalizeInitializedCameraPoses(
                result, target_set, existing, overwrite_existing, full_meta, all_images, output_dir);
            int updated_count = 0;
            if (!finalize_result.cameraUpdates.empty() &&
                !manager->_session->setCameraInstancesById(
                    taskContext.session, finalize_result.cameraUpdates, &updated_count, &error))
            {
                manager->completeTask(taskContext, false);
                if (manager->_session && manager->_session->isCurrent(taskContext.session))
                {
                    manager->_messages->critical(nullptr,
                                                 QStringLiteral("初始化相机位姿"),
                                                 QStringLiteral("SFM 已完成，但回写相机结果失败: %1").arg(error));
                }
                return;
            }
            if (!manager->contextMatches(taskContext))
            {
                manager->completeTask(taskContext, false);
                return;
            }

            if (!finalize_result.sparseCloudPath.isEmpty())
            {
                const xjw::gui::project::TiePointMutationResult tie_point_result =
                    manager->_session->replaceTiePointResult(taskContext.session,
                                                             finalize_result.sparseCloudPath,
                                                             finalize_result.sparsePointCount,
                                                             finalize_result.selectedImages,
                                                             finalize_result.outputDir,
                                                             finalize_result.resultRecordExtra);
                if (!tie_point_result.success)
                {
                    manager->completeTask(taskContext, false);
                    if (manager->_session && manager->_session->isCurrent(taskContext.session))
                    {
                        manager->_messages->critical(
                            nullptr,
                            QStringLiteral("初始化相机位姿"),
                            QStringLiteral("SFM 已完成，但写入连接点结果失败: %1").arg(tie_point_result.errorMessage));
                    }
                    return;
                }
            }
            if (!manager->contextMatches(taskContext))
            {
                manager->completeTask(taskContext, false);
                return;
            }

            LOG_INFO(QStringLiteral("初始化相机位姿完成: 注册=%1 点数=%2 回写=%3")
                         .arg(result.numRegisteredImages)
                         .arg(result.numPoints3D)
                         .arg(updated_count));
            manager->completeTask(taskContext, true);
            manager->_messages->information(
                nullptr,
                QStringLiteral("初始化相机位姿"),
                QStringLiteral("初始化完成。注册影像: %1，三维点: %2，回写相机: %3。\n"
                               "内参初值准备: %4，保留已有相机: %5，尺寸失败: %6，EXIF 成功: %7，默认焦距回退: %8。")
                    .arg(result.numRegisteredImages)
                    .arg(result.numPoints3D)
                    .arg(updated_count)
                    .arg(prepared_count)
                    .arg(kept_existing_count)
                    .arg(invalid_size_count)
                    .arg(exif_count)
                    .arg(fallback_count));
        }));
    return true;
}
