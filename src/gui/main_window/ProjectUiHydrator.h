#pragma once

#include <QJsonObject>
#include <QObject>
#include <QSharedPointer>
#include <QTimer>
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
    void startPendingRefresh();
    void scheduleStage(quint64 generation,
                       int stageIndex,
                       const QSharedPointer<const QJsonObject> &metadata);

    QVector<Stage> _stages;
    quint64 _generation{};
    QTimer _coalesceTimer;
    QSharedPointer<const QJsonObject> _pendingMetadata;
};
