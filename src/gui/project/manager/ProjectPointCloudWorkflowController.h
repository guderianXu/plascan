#pragma once

#include <QObject>
#include <QJsonObject>
#include <QPointer>

#include <atomic>
#include <memory>

class ProjectData;
class ProjectManager;
class QWidget;
struct PointCloudWorkflowContext;

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
    void cancelActiveTask();
    bool isRunning() const;

signals:
    void pointCloudProgressChanged(const QString &stage, int percent);
    void pointCloudProgressFinished(bool success);
    void pointCloudResultReady(const QString &path, int pointCount);

private:
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
