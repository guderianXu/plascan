#include "ProjectUiHydrator.h"

#include <QPointer>
#include <QTimer>

#include <utility>

ProjectUiHydrator::ProjectUiHydrator(QObject *parent)
    : QObject(parent)
{
    _coalesceTimer.setSingleShot(true);
    _coalesceTimer.setInterval(40);
    connect(&_coalesceTimer, &QTimer::timeout, this, &ProjectUiHydrator::startPendingRefresh);
}

void ProjectUiHydrator::setStages(QVector<Stage> stages)
{
    _stages = std::move(stages);
}

void ProjectUiHydrator::schedule(const QJsonObject &metadata)
{
    _pendingMetadata = QSharedPointer<const QJsonObject>::create(metadata);
    ++_generation;
    _coalesceTimer.start();
}

void ProjectUiHydrator::cancel()
{
    ++_generation;
    _pendingMetadata.reset();
    _coalesceTimer.stop();
}

void ProjectUiHydrator::startPendingRefresh()
{
    const QSharedPointer<const QJsonObject> metadata = _pendingMetadata;
    _pendingMetadata.reset();
    if (!metadata)
    {
        return;
    }

    scheduleStage(_generation, 0, metadata);
}

void ProjectUiHydrator::scheduleStage(quint64 generation,
                                      int stageIndex,
                                      const QSharedPointer<const QJsonObject> &metadata)
{
    if (generation != _generation || !metadata || stageIndex < 0 || stageIndex >= _stages.size())
    {
        return;
    }

    QPointer<ProjectUiHydrator> self(this);
    QTimer::singleShot(0, this, [self, generation, stageIndex, metadata]()
    {
        if (!self || generation != self->_generation || stageIndex >= self->_stages.size())
        {
            return;
        }

        const Stage stage = self->_stages.at(stageIndex);
        if (stage)
        {
            stage(*metadata);
        }

        self->scheduleStage(generation, stageIndex + 1, metadata);
    });
}
