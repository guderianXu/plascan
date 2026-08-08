#pragma once

#include <QString>

#include <atomic>
#include <memory>

#include <plapoint/core/point_cloud.h>

namespace xjw::gui::point_cloud
{

using SnapshotCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

struct PointCloudSnapshotStageResult;

class PointCloudSnapshotGuard final
{
public:
    ~PointCloudSnapshotGuard();

    PointCloudSnapshotGuard(const PointCloudSnapshotGuard &) = delete;
    PointCloudSnapshotGuard &operator=(const PointCloudSnapshotGuard &) = delete;

    QString finalPath() const;
    QString temporaryPath() const;
    bool isPending() const;

    bool commit(QString *errorMessage = nullptr);
    bool discard(QString *errorMessage = nullptr);

private:
    friend struct PointCloudSnapshotStageResult;
    friend PointCloudSnapshotStageResult stagePointCloudSnapshot(
        const QString &finalPath,
        const SnapshotCloud &cloud,
        const std::atomic_bool *cancellationFlag);

    PointCloudSnapshotGuard(QString finalPath, QString temporaryPath);

    struct State;
    std::unique_ptr<State> _state;
};

struct PointCloudSnapshotStageResult
{
    bool success = false;
    QString path;
    QString errorMessage;
    int pointCount = 0;
    std::shared_ptr<PointCloudSnapshotGuard> guard;

    QString temporaryPath() const;
    bool commit(QString *errorMessage = nullptr) const;
    bool discard(QString *errorMessage = nullptr) const;
};

PointCloudSnapshotStageResult stagePointCloudSnapshot(
    const QString &finalPath,
    const SnapshotCloud &cloud,
    const std::atomic_bool *cancellationFlag = nullptr);

} // namespace xjw::gui::point_cloud
