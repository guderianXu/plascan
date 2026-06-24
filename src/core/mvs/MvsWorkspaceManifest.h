#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace xjw::mvs
{

struct DepthGenConfig;

struct MvsDepthFrameRecord
{
    int refIndex = -1;
    QString refImage;
    QStringList sourceImages;
    QJsonArray sourcePlan;
    int sourceViewCount = 0;
    double meanSourceQualityScore = 0.0;
    double minSourceQualityScore = 0.0;
    double meanDepthConfidence = 0.0;
    int validPixelCount = 0;
    QString status;
    QString device;
    QString depthPng;
    QString rawDepthPath;
    QString rawConfidencePath;
    QString validMaskPath;
    int gridWidth = 0;
    int gridHeight = 0;
    qint64 elapsedMs = 0;
    QString error;
    QString configHash;

    QJsonObject toJson() const;
    static MvsDepthFrameRecord fromJson(const QJsonObject &object);
};

class MvsWorkspaceManifest
{
public:
    bool load(const QString &path, QString *errorMsg = nullptr);
    bool saveAtomic(const QString &path, QString *errorMsg = nullptr) const;

    void clear();

    QString configHash() const;
    void setConfigHash(const QString &hash);

    const QVector<MvsDepthFrameRecord> &frames() const;
    QVector<MvsDepthFrameRecord> completedFramesSortedByName() const;

    void upsertFrame(const MvsDepthFrameRecord &record);
    void markRunning(int refIndex, const QString &refImage, const QString &configHash);
    void markCompleted(const MvsDepthFrameRecord &record);
    void markFailed(int refIndex, const QString &error);

    bool hasReusableCompletedFrame(int refIndex, const QString &configHash) const;
    QJsonObject toJson() const;

private:
    int findFrameIndex(int refIndex) const;

    QString _configHash;
    QVector<MvsDepthFrameRecord> _frames;
};

QString makeMvsDepthConfigHash(const DepthGenConfig &config, int viewCount);

} // namespace xjw::mvs
