#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace xjw::mvs
{

struct DepthGenConfig;

// Increment whenever a production depth algorithm change makes persisted
// depth maps unsuitable for transparent reuse by a newer build.
inline constexpr int kMvsDepthAlgorithmRevision = 12;

struct MvsDepthFrameRecord
{
    int refIndex = -1;
    QString refImage;
    QStringList sourceImages;
    QVector<int> sourceIndices;
    QJsonArray sourcePlan;
    int sourceViewCount = 0;
    int requestedSourceViewCount = 0;
    int sourceViewShortfall = 0;
    QString sourceViewShortfallReason;
    double meanSourceQualityScore = 0.0;
    double minSourceQualityScore = 0.0;
    double meanDepthConfidence = 0.0;
    double effectivePatchMatchConfidenceThreshold = 0.0;
    int validPixelCount = 0;
    double validCoverage = -1.0;
    QJsonObject depthQuality;
    QJsonObject depthCompleteness;
    QJsonObject crossViewRepairDiagnostics;
    QJsonObject geometryEvidenceDiagnostics;
    QJsonObject qualityDecision;
    QJsonArray pyramidLevels;
    QString maskSource;
    double maskCoverage = -1.0;
    int selectedLevel = 0;
    QString fallbackReason;
    int pyramidRequestedLevelCount = 3;
    int pyramidActiveLevelCount = 0;
    int pyramidMinimumShortSide = 0;
    QString pyramidDegradedReason;
    QString sceneProfile;
    QString filterMode;
    QString acceptance;
    QJsonObject depthPostprocess;
    QJsonObject cameraModel;
    QString status;
    QString device;
    QString depthPng;
    QString rawDepthPath;
    QString rawConfidencePath;
    QString rawGeometrySupportPath;
    QString rawAdaptiveGeometrySupportWeightPath;
    QString rawAdaptiveGeometryEffectiveViewCountPath;
    QString rawAdaptiveGeometryConflictWeightPath;
    QString rawGeometrySourceMaskPath;
    QString rawInverseDepthMeanPath;
    QString rawInverseDepthSpreadPath;
    QString crossViewRepairedMaskPath;
    QString validMaskPath;
    QString supportMaskPath;
    int gridWidth = 0;
    int gridHeight = 0;
    qint64 elapsedMs = 0;
    QString error;
    QString configHash;
    int algorithmRevision = 0;

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
