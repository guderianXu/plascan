#pragma once

#include "detection/DetectionIntegrator.h"

#include <QFutureWatcher>
#include <QObject>
#include <QStringList>

#include <atomic>
#include <memory>

namespace xjw::gui::markers
{

struct MarkerDetectionImage
{
    QString imageId;
    QString imagePath;
    QString maskPath;
    QString imageContentSignature;
};

struct MarkerDetectionJob
{
    quint64 baseRevision = 0;
    QVector<MarkerDetectionImage> images;
    QVector<control_points::MarkerTargetFamily> targetFamilies;
    control_points::MarkerDetectionOptions detectorOptions;
    int maxConcurrentImages = 0;
};

struct MarkerDetectionProgress
{
    int imagesCompleted = 0;
    int imageCount = 0;
    int candidatesDetected = 0;
    int markersMerged = 0;
    QString currentImage;
};

struct MarkerDetectionTaskResult
{
    quint64 baseRevision = 0;
    QVector<control_points::MarkerDetectionObservation> observations;
    QStringList errors;
    bool cancelled = false;
};

struct MarkerTaskSharedState;

class MarkerTaskRunner final : public QObject
{
    Q_OBJECT

public:
    explicit MarkerTaskRunner(QObject *parent = nullptr);
    ~MarkerTaskRunner() override;

    bool start(const MarkerDetectionJob &job);
    void cancel();
    bool isRunning() const noexcept;

signals:
    void progressChanged(const MarkerDetectionProgress &progress);
    void finished(const MarkerDetectionTaskResult &result);

private:
    QFutureWatcher<MarkerDetectionTaskResult> _watcher;
    std::shared_ptr<MarkerTaskSharedState> _state;
    bool _running = false;
};

} // namespace xjw::gui::markers

Q_DECLARE_METATYPE(xjw::gui::markers::MarkerDetectionProgress)
Q_DECLARE_METATYPE(xjw::gui::markers::MarkerDetectionTaskResult)
