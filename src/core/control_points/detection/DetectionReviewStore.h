#pragma once

#include "DetectionIntegrator.h"

#include <QDateTime>
#include <QString>
#include <QVector>

namespace xjw::control_points
{

struct DetectionReviewEntry
{
    QString id;
    QString reason;
    QString message;
    MarkerDetectionObservation observation;
};

struct DetectionReviewQueue
{
    int schemaVersion = 1;
    quint64 sourceRevision = 0;
    QDateTime createdAt = QDateTime::currentDateTimeUtc();
    QDateTime updatedAt = createdAt;
    QVector<DetectionReviewEntry> entries;
};

struct DetectionReviewIoResult
{
    bool ok = false;
    DetectionReviewQueue queue;
    QString error;
};

QString detectionReviewEntryId(const MarkerDetectionObservation &observation,
                               const QString &reason);

class DetectionReviewStore final
{
public:
    explicit DetectionReviewStore(QString path);

    DetectionReviewIoResult load() const;
    DetectionReviewIoResult save(const DetectionReviewQueue &queue) const;
    QString path() const;

private:
    QString _path;
};

} // namespace xjw::control_points
