#pragma once

#include "MvsTypes.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>

#include <cstdint>
#include <mutex>
#include <set>
#include <vector>

namespace cv
{
class Mat;
}

namespace xjw::mvs
{

struct DepthFrameResult;

enum class MvsStageSnapshotStage : int
{
    PatchMatchOutput = 0,
    CrossViewConsistency = 1,
    ConfidencePostprocess = 2,
    FinalAdmission = 3
};

QString mvsStageSnapshotStageId(MvsStageSnapshotStage stage);

/// Bounded, non-authoritative replay diagnostics. Snapshot failures never
/// affect depth estimation, acceptance, cache reuse, or model publication.
class MvsStageSnapshotRecorder final
{
public:
    MvsStageSnapshotRecorder(const DepthGenConfig &config, int viewCount);
    ~MvsStageSnapshotRecorder();

    MvsStageSnapshotRecorder(const MvsStageSnapshotRecorder &) = delete;
    MvsStageSnapshotRecorder &operator=(const MvsStageSnapshotRecorder &) = delete;

    bool enabled() const;
    bool selected(int referenceIndex) const;
    QString manifestPath() const;
    QString initializationError() const;

    void capture(int referenceIndex,
                 MvsStageSnapshotStage stage,
                 const QString &boundary,
                 const DepthFrameResult &result,
                 const cv::Mat &depth,
                 const cv::Mat &confidence,
                 const cv::Mat &validMask = cv::Mat()) noexcept;
    void finalize() noexcept;

private:
    void appendRecordLocked(const QJsonObject &record);
    bool writeManifestLocked(QString *errorMessage);

    mutable std::mutex _mutex;
    QString _directory;
    QString _manifestPath;
    QString _initializationError;
    std::set<int> _selectedReferences;
    std::set<QString> _recordedStageKeys;
    std::vector<QJsonObject> _records;
    QJsonArray _errors;
    int _maximumLongEdge = 1024;
    std::uint64_t _budgetBytes = 0;
    std::uint64_t _usedBytes = 0;
    bool _enabled = false;
    bool _finalized = false;
};

} // namespace xjw::mvs
