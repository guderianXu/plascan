#include "ProjectUiHydrator.h"

#include <QPointer>
#include <QTimer>

#include <utility>

ProjectUiHydrator::ProjectUiHydrator(QObject *parent)
    : QObject(parent)
{
}

void ProjectUiHydrator::setStages(QVector<Stage> stages)
{
    _stages = std::move(stages);
}

void ProjectUiHydrator::schedule(const QJsonObject &metadata)
{
    const quint64 generation = ++_generation;
    scheduleStage(generation, 0, metadata);
}

void ProjectUiHydrator::cancel()
{
    ++_generation;
}

void ProjectUiHydrator::scheduleStage(quint64 generation,
                                      int stageIndex,
                                      const QJsonObject &metadata)
{
    if (generation != _generation || stageIndex < 0 || stageIndex >= _stages.size())
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
            stage(metadata);
        }

        self->scheduleStage(generation, stageIndex + 1, metadata);
    });
}
