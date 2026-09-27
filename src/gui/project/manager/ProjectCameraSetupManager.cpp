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
#include <QPointer>
#include <QSet>

#include <algorithm>
#include <atomic>
#include <utility>

using xjw::gui::project::CameraInitializationMode;
using xjw::gui::project::CameraInitializationRequest;
using xjw::gui::project::CameraInitializationResult;
using xjw::gui::project::finalizeInitializedCameraPoses;
using xjw::gui::project::InitPoseFinalizeResult;
using xjw::gui::project::prepareCameraInitializations;
using xjw::gui::project::resolveInitTargets;
using xjw::gui::project::withPreparedCameras;

namespace
{

    constexpr QDir::Filters kDialogFilters = QDir::AllEntries | QDir::Hidden | QDir::AllDirs | QDir::NoDotAndDotDot;

    struct SfmSetupRunResult
    {
        CameraInitializationResult initialization;
        xjw::aerial_triangulation::AerialTriangulationResult workflow;
        QString outputDir;
        QString error;
        bool cancelled = false;
    };

    struct SingleCameraImportRunResult
    {
        xjw::gui::project::PreparedFrameCameraImport import;
        xjw::gui::project::SingleCameraImportStatus status = xjw::gui::project::SingleCameraImportStatus::Cancelled;
    };

    struct CameraProjectImportRunResult
    {
        xjw::gui::project::CameraProjectImportResult import;
        xjw::gui::project::CameraProjectImportStatus status =
            xjw::gui::project::CameraProjectImportStatus::Cancelled;
    };

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
    if (_taskContext.cancelFlag)
    {
        _taskContext.cancelFlag->store(true, std::memory_order_relaxed);
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
    if (!taskContext.cancelFlag || _taskContext.taskId != taskContext.taskId ||
        _taskContext.cancelFlag != taskContext.cancelFlag)
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
    _taskContext = {};
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

void ProjectCameraSetupManager::setHeavyWorkEnteredObserverForTesting(HeavyWorkEnteredObserver observer)
{
    _heavyWorkEnteredObserver = std::move(observer);
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
    return _taskContext.cancelFlag != nullptr;
}

bool ProjectCameraSetupManager::hasPendingWork() const noexcept
{
    return _taskContext.cancelFlag != nullptr ||
           std::any_of(_futures.cbegin(),
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

std::function<void(int, int)>
ProjectCameraSetupManager::makeProgressReporter(const xjw::gui::project::ProjectTaskContext& taskContext,
                                                const QString& stage,
                                                int firstPercent,
                                                int lastPercent) const
{
    const QPointer<ProjectCameraSetupManager> self(const_cast<ProjectCameraSetupManager*>(this));
    const std::shared_ptr<std::atomic<bool>> cancel_flag = taskContext.cancelFlag;
    const auto last_reported = std::make_shared<std::atomic<int>>(firstPercent - 1);
    return [self, cancel_flag, last_reported, taskContext, stage, firstPercent, lastPercent](int completed, int total)
    {
        if (!cancel_flag || cancel_flag->load(std::memory_order_relaxed))
        {
            return;
        }
        const int bounded_total = std::max(1, total);
        const int bounded_completed = std::clamp(completed, 0, bounded_total);
        const int percent = firstPercent + (lastPercent - firstPercent) * bounded_completed / bounded_total;
        int previous = last_reported->load(std::memory_order_relaxed);
        while (percent > previous &&
               !last_reported->compare_exchange_weak(previous, percent, std::memory_order_relaxed))
        {
        }
        if (percent <= previous)
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
}

bool ProjectCameraSetupManager::importCameraForImage(const QString& imagePath,
                                                     const xjw::gui::project::ProjectTaskContext& taskContext)
{
    _taskContext = taskContext;
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages || !contextMatches(taskContext))
    {
        _taskContext = {};
        return false;
    }

    const UiDialogResult selected = _messages->selectOpenFile(nullptr,
                                                              QStringLiteral("选择相机文件 (.tsai)"),
                                                              readLastDir(QStringLiteral("camera_tsai")),
                                                              QStringLiteral("Tsai相机文件 (*.tsai *.TSAI)"),
                                                              kDialogFilters);
    if (!selected.accepted || selected.text.isEmpty() || !contextMatches(taskContext))
    {
        _taskContext = {};
        return false;
    }
    writeLastDir(QStringLiteral("camera_tsai"), QFileInfo(selected.text).absolutePath());

    const std::shared_ptr<std::atomic<bool>> cancel_flag = taskContext.cancelFlag;
    const HeavyWorkEnteredObserver observer = _heavyWorkEnteredObserver;
    const auto progress = makeProgressReporter(taskContext, QStringLiteral("正在解析相机文件..."), 5, 90);
    emit atProgressChanged(QStringLiteral("正在准备导入相机文件..."), 0);
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [imagePath, tsai_path = selected.text, cancel_flag, observer, progress]()
        {
            SingleCameraImportRunResult run;
            if (cancel_flag->load(std::memory_order_relaxed))
            {
                return run;
            }
            if (observer)
            {
                observer(QStringLiteral("camera-import-single"));
            }
            run.status = xjw::gui::project::buildSingleCameraImport(
                imagePath, tsai_path, &run.import, cancel_flag.get(), progress);
            return run;
        },
        [taskContext](ProjectCameraSetupManager* manager,
                      xjw::gui::tasks::TaskOutcome<SingleCameraImportRunResult> outcome)
        {
            if (!manager->contextMatches(taskContext, false, true))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                const bool report_error = manager->contextMatches(taskContext);
                manager->completeTask(taskContext, false);
                if (report_error)
                {
                    manager->_messages->critical(nullptr, QStringLiteral("错误"), outcome.errorMessage);
                }
                return;
            }
            SingleCameraImportRunResult run = std::move(*outcome.value);
            if (run.status == xjw::gui::project::SingleCameraImportStatus::Cancelled ||
                !manager->contextMatches(taskContext))
            {
                manager->completeTask(taskContext, false);
                return;
            }
            if (run.status != xjw::gui::project::SingleCameraImportStatus::Ok)
            {
                manager->completeTask(taskContext, false);
                manager->_messages->critical(nullptr, QStringLiteral("错误"), run.import.error);
                return;
            }

            placamera::CameraInstanceSet cameras;
            QMap<QString, QJsonObject> annotations;
            QString error;
            const bool bound = xjw::gui::project::bindImportedFrameCameras(manager->_session->coreMetadata(),
                                                                           manager->_session->projectPath(),
                                                                           {run.import},
                                                                           &cameras,
                                                                           &annotations,
                                                                           &error);
            int updated_count = 0;
            if (!bound || !manager->_session->upsertNativeCameraInstances(
                              taskContext.session, cameras, annotations, &updated_count, &error))
            {
                const bool report_error = manager->_session->isCurrent(taskContext.session);
                manager->completeTask(taskContext, false);
                if (report_error)
                {
                    manager->_messages->critical(
                        nullptr, QStringLiteral("错误"), QStringLiteral("导入相机失败: %1").arg(error));
                }
                return;
            }
            emit manager->atProgressChanged(QStringLiteral("相机文件导入完成"), 100);
            manager->completeTask(taskContext, true);
            manager->_messages->information(
                nullptr,
                QStringLiteral("导入成功"),
                QStringLiteral("已为影像 %1 导入相机文件。").arg(QFileInfo(run.import.imageAbsPath).fileName()));
        }));
    return true;
}

bool ProjectCameraSetupManager::importCameraProject(const xjw::gui::project::ProjectTaskContext& taskContext)
{
    _taskContext = taskContext;
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages || !contextMatches(taskContext))
    {
        _taskContext = {};
        return false;
    }

    const QString file_input = QStringLiteral("工程文件");
    const QString directory_input = QStringLiteral("工程目录");
    const UiDialogResult input_type = _messages->getItem(
        nullptr, QStringLiteral("导入相机工程"), QStringLiteral("输入类型"), {file_input, directory_input}, 0);
    if (!input_type.accepted || !contextMatches(taskContext))
    {
        _taskContext = {};
        return false;
    }

    const QString last_dir = readLastDir(QStringLiteral("camera_project"));
    const UiDialogResult selected =
        input_type.text == directory_input
            ? _messages->selectDirectory(nullptr, QStringLiteral("选择相机工程目录"), last_dir, kDialogFilters)
            : _messages->selectOpenFile(nullptr,
                                        QStringLiteral("选择相机工程文件"),
                                        last_dir,
                                        QStringLiteral("相机工程文件 (*.txt *.camera *.xml *.psx);;所有文件 (*)"),
                                        kDialogFilters);
    if (!selected.accepted || selected.text.isEmpty() || !contextMatches(taskContext))
    {
        _taskContext = {};
        return false;
    }
    writeLastDir(QStringLiteral("camera_project"),
                 QFileInfo(selected.text).isDir() ? selected.text : QFileInfo(selected.text).absolutePath());

    const QStringList images = _session->allImages();
    const std::shared_ptr<std::atomic<bool>> cancel_flag = taskContext.cancelFlag;
    const HeavyWorkEnteredObserver observer = _heavyWorkEnteredObserver;
    const auto progress = makeProgressReporter(taskContext, QStringLiteral("正在解析并匹配相机工程..."), 5, 90);
    emit atProgressChanged(QStringLiteral("正在准备导入相机工程..."), 0);
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [input_path = selected.text, images, cancel_flag, observer, progress]()
        {
            CameraProjectImportRunResult run;
            if (cancel_flag->load(std::memory_order_relaxed))
            {
                return run;
            }
            if (observer)
            {
                observer(QStringLiteral("camera-import-project"));
            }
            run.status = xjw::gui::project::buildCameraProjectImport(
                input_path, images, &run.import, cancel_flag.get(), progress);
            return run;
        },
        [taskContext](ProjectCameraSetupManager* manager,
                      xjw::gui::tasks::TaskOutcome<CameraProjectImportRunResult> outcome)
        {
            if (!manager->contextMatches(taskContext, false, true))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                const bool report_error = manager->contextMatches(taskContext);
                manager->completeTask(taskContext, false);
                if (report_error)
                {
                    manager->_messages->critical(nullptr, QStringLiteral("错误"), outcome.errorMessage);
                }
                return;
            }
            CameraProjectImportRunResult run = std::move(*outcome.value);
            if (run.status == xjw::gui::project::CameraProjectImportStatus::Cancelled ||
                !manager->contextMatches(taskContext))
            {
                manager->completeTask(taskContext, false);
                return;
            }
            for (const QString& warning : run.import.warnings)
            {
                LOG_WARN(warning);
            }
            for (const QString& import_error : run.import.importErrors)
            {
                LOG_WARN(import_error);
            }
            if (run.status == xjw::gui::project::CameraProjectImportStatus::ParseFailed)
            {
                manager->completeTask(taskContext, false);
                manager->_messages->critical(
                    nullptr, QStringLiteral("错误"), QStringLiteral("相机工程解析失败: %1").arg(run.import.error));
                return;
            }
            if (run.status == xjw::gui::project::CameraProjectImportStatus::NoProjectImages)
            {
                manager->completeTask(taskContext, false);
                manager->_messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("项目中没有可匹配的影像"));
                return;
            }
            if (run.status == xjw::gui::project::CameraProjectImportStatus::NoImportable)
            {
                manager->completeTask(taskContext, false);
                manager->_messages->warning(
                    nullptr,
                    QStringLiteral("提示"),
                    QStringLiteral("没有可导入的相机记录。未匹配: %1，重名冲突: %2，不支持: %3%4")
                        .arg(run.import.unmatchedCount)
                        .arg(run.import.ambiguousCount)
                        .arg(run.import.unsupportedCount)
                        .arg(run.import.error.isEmpty() ? QString() : QStringLiteral("，%1").arg(run.import.error)));
                return;
            }

            placamera::CameraInstanceSet cameras;
            QMap<QString, QJsonObject> annotations;
            QString error;
            const bool bound = xjw::gui::project::bindImportedFrameCameras(manager->_session->coreMetadata(),
                                                                           manager->_session->projectPath(),
                                                                           run.import.cameras,
                                                                           &cameras,
                                                                           &annotations,
                                                                           &error);
            int updated_count = 0;
            if (!bound || !manager->_session->upsertNativeCameraInstances(
                              taskContext.session, cameras, annotations, &updated_count, &error))
            {
                const bool report_error = manager->_session->isCurrent(taskContext.session);
                manager->completeTask(taskContext, false);
                if (report_error)
                {
                    manager->_messages->critical(
                        nullptr, QStringLiteral("错误"), QStringLiteral("相机工程导入失败: %1").arg(error));
                }
                return;
            }
            emit manager->atProgressChanged(QStringLiteral("相机工程导入完成"), 100);
            manager->completeTask(taskContext, true);
            manager->_messages->information(
                nullptr,
                QStringLiteral("导入完成"),
                QStringLiteral("已从 %1 写入 %2 条相机记录（未匹配: %3，重名冲突: %4，不支持: %5）。")
                    .arg(run.import.sourceFormat)
                    .arg(updated_count)
                    .arg(run.import.unmatchedCount)
                    .arg(run.import.ambiguousCount)
                    .arg(run.import.unsupportedCount));
        }));
    return true;
}

bool ProjectCameraSetupManager::initializeCamerasFromExifOrDefault(
    const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& taskContext)
{
    return startCameraInitialization(settings, taskContext, true);
}

bool ProjectCameraSetupManager::initializeCamerasFromIntrinsics(
    const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& taskContext)
{
    return startCameraInitialization(settings, taskContext, false);
}

bool ProjectCameraSetupManager::startCameraInitialization(const QJsonObject& settings,
                                                          const xjw::gui::project::ProjectTaskContext& taskContext,
                                                          bool useExif)
{
    _taskContext = taskContext;
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages || !contextMatches(taskContext))
    {
        _taskContext = {};
        return false;
    }

    const QStringList all_images = _session->allImages();
    QString target_error;
    const QStringList target_images = resolveInitTargets(all_images, settings, &target_error);
    if (target_images.isEmpty())
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), target_error);
        _taskContext = {};
        return false;
    }
    if (!useExif && (settings.value(QStringLiteral("fx")).toDouble(0.0) <= 0.0 ||
                     settings.value(QStringLiteral("fy")).toDouble(0.0) <= 0.0))
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("fx/fy 必须大于 0。"));
        _taskContext = {};
        return false;
    }

    CameraInitializationRequest request;
    request.images = target_images;
    request.projectMetadata = _session->coreMetadata();
    request.settings = settings;
    request.mode = useExif ? CameraInitializationMode::ExifOrDefault : CameraInitializationMode::Intrinsics;
    request.source = useExif ? QStringLiteral("init_from_exif_or_default") : QStringLiteral("init_from_intrinsics");

    const std::shared_ptr<std::atomic<bool>> cancel_flag = taskContext.cancelFlag;
    const HeavyWorkEnteredObserver observer = _heavyWorkEnteredObserver;
    const auto progress = makeProgressReporter(taskContext,
                                               useExif ? QStringLiteral("正在读取影像尺寸与 EXIF...")
                                                       : QStringLiteral("正在读取影像尺寸..."),
                                               5,
                                               90);
    emit atProgressChanged(QStringLiteral("正在准备相机初值..."), 0);
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [request = std::move(request), cancel_flag, observer, progress, useExif]()
        {
            if (cancel_flag->load(std::memory_order_relaxed))
            {
                CameraInitializationResult cancelled;
                cancelled.cancelled = true;
                return cancelled;
            }
            if (observer)
            {
                observer(useExif ? QStringLiteral("camera-init-exif") : QStringLiteral("camera-init-intrinsics"));
            }
            return prepareCameraInitializations(request, cancel_flag.get(), progress);
        },
        [taskContext, useExif](ProjectCameraSetupManager* manager,
                               xjw::gui::tasks::TaskOutcome<CameraInitializationResult> outcome)
        {
            if (!manager->contextMatches(taskContext, false, true))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                const bool report_error = manager->contextMatches(taskContext);
                manager->completeTask(taskContext, false);
                if (report_error)
                {
                    manager->_messages->critical(nullptr, QStringLiteral("错误"), outcome.errorMessage);
                }
                return;
            }
            CameraInitializationResult result = std::move(*outcome.value);
            if (result.cancelled || !manager->contextMatches(taskContext))
            {
                manager->completeTask(taskContext, false);
                return;
            }
            if (!result.error.isEmpty())
            {
                manager->completeTask(taskContext, false);
                manager->_messages->warning(nullptr, QStringLiteral("初始化相机位姿"), result.error);
                return;
            }
            if (result.cameras.empty())
            {
                manager->completeTask(taskContext, false);
                manager->_messages->warning(nullptr,
                                            QStringLiteral("初始化相机位姿"),
                                            QStringLiteral("没有可写入的影像。已跳过已有相机: %1，尺寸无法读取: %2。")
                                                .arg(result.skippedExisting)
                                                .arg(result.invalidSizeCount));
                return;
            }

            int updated_count = 0;
            QString error;
            if (!manager->_session->upsertNativeCameraInstances(
                    taskContext.session, result.cameras, result.annotationsByImageId, &updated_count, &error))
            {
                const bool report_error = manager->_session->isCurrent(taskContext.session);
                manager->completeTask(taskContext, false);
                if (report_error)
                {
                    manager->_messages->critical(
                        nullptr, QStringLiteral("错误"), QStringLiteral("写入相机初值失败: %1").arg(error));
                }
                return;
            }
            emit manager->atProgressChanged(QStringLiteral("相机初值初始化完成"), 100);
            manager->completeTask(taskContext, true);
            if (useExif)
            {
                manager->_messages->information(
                    nullptr,
                    QStringLiteral("初始化完成"),
                    QStringLiteral("已写入 %1 张影像的相机初值。EXIF 成功: %2，默认焦距回退: %3，"
                                   "跳过已有相机: %4，尺寸失败: %5。")
                        .arg(updated_count)
                        .arg(result.exifCount)
                        .arg(result.fallbackCount)
                        .arg(result.skippedExisting)
                        .arg(result.invalidSizeCount));
            }
            else
            {
                manager->_messages->information(
                    nullptr,
                    QStringLiteral("初始化完成"),
                    QStringLiteral("已写入 %1 张影像的相机初值。跳过已有相机: %2，自动主点: %3，"
                                   "尺寸失败: %4。")
                        .arg(updated_count)
                        .arg(result.skippedExisting)
                        .arg(result.autoPrincipalPointCount)
                        .arg(result.invalidSizeCount));
            }
        }));
    return true;
}

bool ProjectCameraSetupManager::initializeCameraPosesWithSFM(const QJsonObject& settings,
                                                             const xjw::gui::project::ProjectTaskContext& taskContext)
{
    _taskContext = taskContext;
    if (!requireProject(QStringLiteral("请先打开或创建项目")) || !_messages || !contextMatches(taskContext))
    {
        _taskContext = {};
        return false;
    }

    const int mode = settings.value(QStringLiteral("mode")).toInt();
    if (mode != 0 && mode != 1)
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("当前模式不适用相对定向初始化。"));
        _taskContext = {};
        return false;
    }

    const QStringList all_images = _session->allImages();
    if (all_images.size() < 2)
    {
        _messages->warning(
            nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("至少需要 2 张影像才能进行相对定向初始化。"));
        _taskContext = {};
        return false;
    }

    QString target_error;
    const QStringList target_images = resolveInitTargets(all_images, settings, &target_error);
    if (target_images.isEmpty())
    {
        _messages->warning(nullptr, QStringLiteral("初始化相机位姿"), target_error);
        _taskContext = {};
        return false;
    }

    const bool overwrite_existing = settings.value(QStringLiteral("overwriteExisting")).toBool(false);
    const QJsonObject base_meta = _session->coreMetadata();
    const QJsonObject full_meta = _session->metadata();
    if (mode == 1 && (settings.value(QStringLiteral("fx")).toDouble(0.0) <= 0.0 ||
                      settings.value(QStringLiteral("fy")).toDouble(0.0) <= 0.0))
    {
        _messages->warning(
            nullptr, QStringLiteral("初始化相机位姿"), QStringLiteral("仅有内参模式下，fx/fy 必须大于 0。"));
        _taskContext = {};
        return false;
    }

    const QString project_path = taskContext.session.projectPath;
    const QString assets_dir = xjw::common::project::ProjectIO::projectAssetsDir(project_path);
    const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const QString output_dir =
        QDir(assets_dir).filePath(QStringLiteral("aerial_triangulation/init_pose_%1").arg(timestamp));

    CameraInitializationRequest initialization_request;
    initialization_request.images = all_images;
    initialization_request.projectMetadata = base_meta;
    initialization_request.settings = settings;
    initialization_request.mode =
        mode == 0 ? CameraInitializationMode::ExifOrDefault : CameraInitializationMode::Intrinsics;
    initialization_request.source = mode == 0 ? QStringLiteral("init_pose_intrinsics_from_exif_or_default")
                                              : QStringLiteral("init_pose_intrinsics_manual");

    const std::shared_ptr<std::atomic<bool>> cancel_flag = taskContext.cancelFlag;
    const HeavyWorkEnteredObserver observer = _heavyWorkEnteredObserver;
    const auto preparation_progress =
        makeProgressReporter(taskContext, QStringLiteral("正在读取影像尺寸与 EXIF..."), 1, 15);
    const QPointer<ProjectCameraSetupManager> self(this);
    const auto workflow_last_reported = std::make_shared<std::atomic<int>>(14);
    const auto workflow_progress =
        [self, cancel_flag, workflow_last_reported, taskContext](const QString& stage, int percent)
    {
        if (cancel_flag->load(std::memory_order_relaxed))
        {
            return;
        }
        const int mapped_percent = 15 + std::clamp(percent, 0, 100) * 80 / 100;
        int previous = workflow_last_reported->load(std::memory_order_relaxed);
        while (mapped_percent > previous &&
               !workflow_last_reported->compare_exchange_weak(previous, mapped_percent, std::memory_order_relaxed))
        {
        }
        if (mapped_percent <= previous)
        {
            return;
        }
        xjw::gui::tasks::postGuarded(self,
                                     [taskContext, stage, mapped_percent](ProjectCameraSetupManager* manager)
                                     {
                                         if (manager->contextMatches(taskContext))
                                         {
                                             emit manager->atProgressChanged(stage, mapped_percent);
                                         }
                                     });
    };
    const auto pair_matched = [self, cancel_flag, taskContext](
                                  const QString& img0, const QString& img1, const QString& match_path, int num_matches)
    {
        if (cancel_flag->load(std::memory_order_relaxed))
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
        [runner,
         initialization_request = std::move(initialization_request),
         settings,
         all_images,
         project_path,
         full_meta,
         output_dir,
         overwrite_existing,
         cancel_flag,
         observer,
         preparation_progress,
         workflow_progress,
         pair_matched]() mutable
        {
            SfmSetupRunResult run;
            run.outputDir = output_dir;
            if (cancel_flag->load(std::memory_order_relaxed))
            {
                run.cancelled = true;
                return run;
            }
            if (observer)
            {
                observer(QStringLiteral("camera-sfm"));
            }
            run.initialization =
                prepareCameraInitializations(initialization_request, cancel_flag.get(), preparation_progress);
            if (run.initialization.cancelled || cancel_flag->load(std::memory_order_relaxed))
            {
                run.cancelled = true;
                return run;
            }
            if (!run.initialization.error.isEmpty())
            {
                run.error = run.initialization.error;
                return run;
            }
            if (run.initialization.cameras.empty() && run.initialization.existingImages.isEmpty())
            {
                run.error =
                    QStringLiteral("没有可用于求解的内参初值。尺寸失败: %1。").arg(run.initialization.invalidSizeCount);
                return run;
            }
            if (!QDir().mkpath(output_dir) && !QDir(output_dir).exists())
            {
                run.error = QStringLiteral("无法创建相机初始化输出目录: %1").arg(output_dir);
                return run;
            }

            xjw::aerial_triangulation::AerialTriangulationOptions workflow_options;
            workflow_options.images = all_images;
            workflow_options.projectPath = project_path;
            if (!withPreparedCameras(
                    full_meta, run.initialization, overwrite_existing, &workflow_options.projectMeta, &run.error))
            {
                return run;
            }
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
            workflow_options.matchingAlgorithmId = settings.value(QStringLiteral("algorithm_id"))
                                                       .toString(QStringLiteral("plamatch_hct"))
                                                       .trimmed()
                                                       .toLower();
            workflow_options.resetAlignment = false;
            workflow_options.autoGenerateMissingMatches = false;
            workflow_options.cancelFlag = cancel_flag;
            workflow_options.progressFn = workflow_progress;
            workflow_options.pairMatchedFn = pair_matched;
            LOG_INFO(QStringLiteral("初始化相机位姿: 使用影像匹配算法 %1").arg(workflow_options.matchingAlgorithmId));
            run.workflow = runner(std::move(workflow_options));
            run.cancelled = cancel_flag->load(std::memory_order_relaxed);
            return run;
        },
        [taskContext, all_images, target_set, overwrite_existing, full_meta](
            ProjectCameraSetupManager* manager, xjw::gui::tasks::TaskOutcome<SfmSetupRunResult> outcome) mutable
        {
            if (!manager->contextMatches(taskContext, false, true))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                const bool report_error = manager->contextMatches(taskContext);
                manager->completeTask(taskContext, false);
                if (report_error)
                {
                    manager->_messages->warning(nullptr, QStringLiteral("初始化相机位姿"), outcome.errorMessage);
                }
                return;
            }
            SfmSetupRunResult run = std::move(*outcome.value);
            if (run.cancelled || !manager->contextMatches(taskContext))
            {
                manager->completeTask(taskContext, false);
                return;
            }
            if (!run.error.isEmpty())
            {
                manager->completeTask(taskContext, false);
                manager->_messages->warning(nullptr, QStringLiteral("初始化相机位姿"), run.error);
                return;
            }

            xjw::aerial_triangulation::AerialTriangulationResult workflow_result = std::move(run.workflow);
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

            const InitPoseFinalizeResult finalize_result =
                finalizeInitializedCameraPoses(result,
                                               target_set,
                                               run.initialization.existingImages,
                                               overwrite_existing,
                                               full_meta,
                                               all_images,
                                               run.outputDir);
            if (!finalize_result.errorMessage.isEmpty())
            {
                manager->completeTask(taskContext, false);
                manager->_messages->critical(nullptr,
                                             QStringLiteral("初始化相机位姿"),
                                             finalize_result.errorMessage);
                return;
            }
            int updated_count = 0;
            if (!finalize_result.cameraInstances.empty() &&
                !manager->_session->upsertNativeCameraInstances(taskContext.session,
                                                                finalize_result.cameraInstances,
                                                                finalize_result.cameraAnnotationsByImageId,
                                                                &updated_count,
                                                                &error))
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
            emit manager->atProgressChanged(QStringLiteral("相机位姿初始化完成"), 100);
            manager->completeTask(taskContext, true);
            manager->_messages->information(
                nullptr,
                QStringLiteral("初始化相机位姿"),
                QStringLiteral("初始化完成。注册影像: %1，三维点: %2，回写相机: %3。\n"
                               "内参初值准备: %4，保留已有相机: %5，尺寸失败: %6，EXIF 成功: %7，默认焦距回退: %8。")
                    .arg(result.numRegisteredImages)
                    .arg(result.numPoints3D)
                    .arg(updated_count)
                    .arg(run.initialization.cameras.size())
                    .arg(run.initialization.skippedExisting)
                    .arg(run.initialization.invalidSizeCount)
                    .arg(run.initialization.exifCount)
                    .arg(run.initialization.fallbackCount));
        }));
    return true;
}
