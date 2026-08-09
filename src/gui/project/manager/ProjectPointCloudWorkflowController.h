#pragma once

#include <QObject>
#include <QJsonObject>
#include <QPointer>
#include <QStringList>

#include <atomic>
#include <memory>

class ProjectData;
class ProjectManager;
class QWidget;
struct PointCloudWorkflowContext;

namespace xjw::gui::project
{

inline QString classifyStoredMvsBackendDevices(const QStringList &devices)
{
    bool has_cuda = false;
    bool has_opencl = false;
    bool has_cpu = false;
    for (const QString &value : devices)
    {
        const QString device = value.trimmed().toLower();
        has_cuda = has_cuda || device.startsWith(QStringLiteral("cuda")) ||
            device.startsWith(QStringLiteral("gpu"));
        has_opencl = has_opencl || device.startsWith(QStringLiteral("opencl"));
        has_cpu = has_cpu || device.startsWith(QStringLiteral("cpu"));
    }

    if (has_cuda && has_opencl && !has_cpu)
    {
        return QStringLiteral("hybrid");
    }

    const int backend_count = static_cast<int>(has_cuda) +
        static_cast<int>(has_opencl) + static_cast<int>(has_cpu);
    if (backend_count > 1)
    {
        return QStringLiteral("mixed");
    }
    if (has_cuda)
    {
        return QStringLiteral("cuda");
    }
    if (has_opencl)
    {
        return QStringLiteral("opencl");
    }
    if (has_cpu)
    {
        return QStringLiteral("cpu");
    }
    return QStringLiteral("unknown");
}

inline bool canReuseStoredMvsBackend(const QString &requestedBackend,
                                     const QString &storedBackend)
{
    const QString requested_backend = requestedBackend.trimmed().toLower();
    const QString stored_backend = storedBackend.trimmed().toLower();
    const bool stored_backend_is_uniform = stored_backend == QStringLiteral("cuda") ||
        stored_backend == QStringLiteral("opencl") ||
        stored_backend == QStringLiteral("cpu");
    if (requested_backend == QStringLiteral("auto"))
    {
        return stored_backend_is_uniform || stored_backend == QStringLiteral("hybrid");
    }
    return stored_backend_is_uniform && requested_backend == stored_backend;
}

} // namespace xjw::gui::project

/**
 * @brief “创建点云”工作流的 GUI 协调器。
 *
 * 该类只负责把项目内正式空三结果接到 MVS：可选复用深度图，否则先估计
 * 深度图，随后从磁盘流式融合并把点云记录写回项目。算法实现仍位于 core/mvs，
 * 因而这里不包含 PatchMatch 或融合数学逻辑。
 */
class ProjectPointCloudWorkflowController : public QObject
{
    Q_OBJECT

public:
    explicit ProjectPointCloudWorkflowController(ProjectManager *owner,
                                                 ProjectData *projectData,
                                                 QWidget *parentWidget,
                                                 QObject *parent = nullptr);
    ~ProjectPointCloudWorkflowController() override;

    bool startCreatePointCloudAsync(const QJsonObject &settings);
    bool startDepthMapsOnlyAsync(const QJsonObject &settings);
    void cancelActiveTask();
    bool isRunning() const;

signals:
    void pointCloudProgressChanged(const QString &stage, int percent);
    void pointCloudProgressFinished(bool success);
    void pointCloudResultReady(const QString &path, int pointCount);
    void depthMapBatchReady(const QString &outputDirectory, int frameCount);

private:
    bool startWorkflow(const QJsonObject &settings, bool depthMapsOnly);
    void startDepthEstimation(
        const std::shared_ptr<PointCloudWorkflowContext> &context);
    void startFusion(
        const std::shared_ptr<PointCloudWorkflowContext> &context);
    void finishTask(bool success);
    void failTask(const QString &message,
                  const QString &title = QStringLiteral("创建点云"));

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    QPointer<QObject> _activeGenerator;
    std::shared_ptr<std::atomic_bool> _cancelFlag;
    bool _isRunning = false;
};
