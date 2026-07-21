#pragma once

#include <QJsonObject>
#include <QObject>
#include <QVector>

#include <functional>

class ProjectUiHydrator : public QObject
{
    Q_OBJECT
public:
    using Stage = std::function<void(const QJsonObject &)>;

    explicit ProjectUiHydrator(QObject *parent = nullptr);

    void setStages(QVector<Stage> stages);
    void schedule(const QJsonObject &metadata);
    void cancel();

private:
    void scheduleStage(quint64 generation, int stageIndex, const QJsonObject &metadata);

    QVector<Stage> _stages;
    quint64 _generation{};
};
