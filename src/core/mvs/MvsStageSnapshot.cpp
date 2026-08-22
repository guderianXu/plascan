#include "MvsStageSnapshot.h"

#include "DepthCompletenessMetrics.h"
#include "DepthMapGenerator.h"
#include "MvsQualityReport.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSaveFile>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace xjw::mvs
{

namespace
{

constexpr std::array<char, 16> kFastDepthMatMagic{
    'P', 'L', 'A', 'S', 'D', 'E', 'P', 'T',
    'H', 'M', 'A', 'T', '0', '1', '\0', '\0'};

struct StageDepthMatHeader
{
    char magic[16] = {};
    qint32 rows = 0;
    qint32 cols = 0;
    qint32 type = 0;
    quint32 reserved = 0;
    quint64 dataBytes = 0;
};

static_assert(sizeof(StageDepthMatHeader) == 40,
              "Stage snapshots must use the stable fast-depth layout");

QString stageKey(int reference_index, MvsStageSnapshotStage stage)
{
    return QStringLiteral("%1:%2")
        .arg(reference_index)
        .arg(static_cast<int>(stage));
}

int stageOrder(const QJsonObject &record)
{
    const QString stage = record.value(QStringLiteral("stage")).toString();
    if (stage == QStringLiteral("patchmatch_output")) return 0;
    if (stage == QStringLiteral("cross_view_consistency")) return 1;
    if (stage == QStringLiteral("confidence_postprocess")) return 2;
    if (stage == QStringLiteral("final_admission")) return 3;
    return 4;
}

QString fileSha256(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return QString();
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        hash.addData(file.read(1024 * 1024));
    }
    return QString::fromLatin1(hash.result().toHex());
}

bool writeMatrixAtomically(const QString &path,
                           const cv::Mat &matrix,
                           QString *error_message)
{
    if (matrix.empty())
    {
        if (error_message) *error_message = QStringLiteral("矩阵为空");
        return false;
    }
    if (QFileInfo::exists(path))
    {
        if (error_message)
        {
            *error_message = QStringLiteral("阶段快照已存在，拒绝覆盖：%1").arg(path);
        }
        return false;
    }

    const cv::Mat contiguous = matrix.isContinuous() ? matrix : matrix.clone();
    StageDepthMatHeader header;
    std::memcpy(header.magic, kFastDepthMatMagic.data(), kFastDepthMatMagic.size());
    header.rows = contiguous.rows;
    header.cols = contiguous.cols;
    header.type = contiguous.type();
    header.dataBytes = static_cast<quint64>(
        contiguous.total() * contiguous.elemSize());

    QSaveFile file(path);
    file.setDirectWriteFallback(false);
    if (!file.open(QIODevice::WriteOnly) ||
        file.write(reinterpret_cast<const char *>(&header), sizeof(header)) !=
            static_cast<qint64>(sizeof(header)) ||
        file.write(reinterpret_cast<const char *>(contiguous.data),
                   static_cast<qint64>(header.dataBytes)) !=
            static_cast<qint64>(header.dataBytes) ||
        !file.commit())
    {
        if (error_message)
        {
            *error_message = QStringLiteral("无法原子写入阶段快照：%1").arg(path);
        }
        return false;
    }
    return true;
}

cv::Size boundedSize(const cv::Size &input, int maximum_long_edge)
{
    const int long_edge = std::max(input.width, input.height);
    if (long_edge <= maximum_long_edge)
    {
        return input;
    }
    const double scale = static_cast<double>(maximum_long_edge) /
                         static_cast<double>(long_edge);
    return cv::Size(std::max(1, static_cast<int>(std::lround(input.width * scale))),
                    std::max(1, static_cast<int>(std::lround(input.height * scale))));
}

cv::Mat resizedNearest(const cv::Mat &matrix, const cv::Size &size)
{
    if (matrix.size() == size)
    {
        return matrix.clone();
    }
    cv::Mat output;
    cv::resize(matrix, output, size, 0.0, 0.0, cv::INTER_NEAREST);
    return output;
}

QJsonArray doublesToJson(const double *values, int count)
{
    QJsonArray array;
    for (int index = 0; index < count; ++index) array.append(values[index]);
    return array;
}

QJsonObject cameraToJson(const FramePinholeCamera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto rotation = camera.worldToCameraRotation();
    const auto translation = camera.worldToCameraTranslation();
    const auto center = camera.cameraCenter();
    return QJsonObject{
        {QStringLiteral("fx"), intrinsics.focalX},
        {QStringLiteral("fy"), intrinsics.focalY},
        {QStringLiteral("cx"), intrinsics.principalX},
        {QStringLiteral("cy"), intrinsics.principalY},
        {QStringLiteral("rotation_world_to_camera"),
         doublesToJson(rotation.data(), 9)},
        {QStringLiteral("translation_world_to_camera"),
         doublesToJson(translation.data(), 3)},
        {QStringLiteral("camera_center"), doublesToJson(center.data(), 3)}};
}

QJsonObject artifactToJson(const QString &path, const cv::Mat &matrix)
{
    const QFileInfo info(path);
    return QJsonObject{
        {QStringLiteral("path"), info.absoluteFilePath()},
        {QStringLiteral("sha256"), fileSha256(path)},
        {QStringLiteral("size_bytes"), static_cast<double>(info.size())},
        {QStringLiteral("rows"), matrix.rows},
        {QStringLiteral("cols"), matrix.cols},
        {QStringLiteral("opencv_type"), matrix.type()}};
}

QString boundaryForStage(MvsStageSnapshotStage stage)
{
    switch (stage)
    {
    case MvsStageSnapshotStage::PatchMatchOutput:
        return QStringLiteral("single_frame_output_after_sparse_prior_and_local_filter");
    case MvsStageSnapshotStage::CrossViewConsistency:
        return QStringLiteral("after_cross_view_filter_and_repair_before_confidence_postprocess");
    case MvsStageSnapshotStage::ConfidencePostprocess:
        return QStringLiteral("after_confidence_postprocess_and_anchored_repair");
    case MvsStageSnapshotStage::FinalAdmission:
        return QStringLiteral("after_final_quality_evaluation_before_artifact_publication");
    }
    return QStringLiteral("unknown");
}

} // namespace

QString mvsStageSnapshotStageId(MvsStageSnapshotStage stage)
{
    switch (stage)
    {
    case MvsStageSnapshotStage::PatchMatchOutput:
        return QStringLiteral("patchmatch_output");
    case MvsStageSnapshotStage::CrossViewConsistency:
        return QStringLiteral("cross_view_consistency");
    case MvsStageSnapshotStage::ConfidencePostprocess:
        return QStringLiteral("confidence_postprocess");
    case MvsStageSnapshotStage::FinalAdmission:
        return QStringLiteral("final_admission");
    }
    return QStringLiteral("unknown");
}

MvsStageSnapshotRecorder::MvsStageSnapshotRecorder(
    const DepthGenConfig &config,
    int view_count)
    : _directory(QString::fromStdString(config.stageSnapshotDirectory)),
      _maximumLongEdge(config.stageSnapshotMaximumLongEdge),
      _budgetBytes(config.stageSnapshotBudgetBytes)
{
    for (int reference_index : config.stageSnapshotReferenceIndices)
    {
        if (reference_index >= 0 && reference_index < view_count)
        {
            _selectedReferences.insert(reference_index);
        }
        else
        {
            _errors.append(QStringLiteral("reference_index_out_of_range:%1")
                               .arg(reference_index));
        }
    }
    if (_selectedReferences.empty()) return;
    if (_directory.trimmed().isEmpty() || _maximumLongEdge <= 0 || _budgetBytes == 0)
    {
        _initializationError = QStringLiteral("阶段快照目录、尺寸上限或预算无效");
        return;
    }
    _directory = QFileInfo(_directory).absoluteFilePath();
    _manifestPath = QDir(_directory).filePath(QStringLiteral("manifest.json"));
    if (QFileInfo::exists(_manifestPath) || !QDir().mkpath(_directory))
    {
        _initializationError = QStringLiteral("阶段快照目录不可用或 manifest 已存在：%1")
                                   .arg(_directory);
        return;
    }
    _enabled = true;
    QString error;
    if (!writeManifestLocked(&error))
    {
        _enabled = false;
        _initializationError = error;
    }
}

MvsStageSnapshotRecorder::~MvsStageSnapshotRecorder()
{
    finalize();
}

bool MvsStageSnapshotRecorder::enabled() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _enabled;
}

bool MvsStageSnapshotRecorder::selected(int reference_index) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _enabled && _selectedReferences.contains(reference_index);
}

QString MvsStageSnapshotRecorder::manifestPath() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _manifestPath;
}

QString MvsStageSnapshotRecorder::initializationError() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _initializationError;
}

void MvsStageSnapshotRecorder::capture(
    int reference_index,
    MvsStageSnapshotStage stage,
    const QString &boundary,
    const DepthFrameResult &result,
    const cv::Mat &depth,
    const cv::Mat &confidence,
    const cv::Mat &valid_mask) noexcept
{
    try
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const QString key = stageKey(reference_index, stage);
        if (!_enabled || !_selectedReferences.contains(reference_index) ||
            _recordedStageKeys.contains(key))
        {
            return;
        }
        _recordedStageKeys.insert(key);

        QJsonObject record{
            {QStringLiteral("ref_index"), reference_index},
            {QStringLiteral("stage"), mvsStageSnapshotStageId(stage)},
            {QStringLiteral("boundary"), boundary.isEmpty()
                 ? boundaryForStage(stage) : boundary}};
        if (depth.empty() || confidence.empty() || depth.type() != CV_32FC1 ||
            confidence.type() != CV_32FC1 || confidence.size() != depth.size())
        {
            record.insert(QStringLiteral("status"), QStringLiteral("unavailable"));
            record.insert(QStringLiteral("reason"),
                          QStringLiteral("missing_or_incompatible_depth_confidence"));
            appendRecordLocked(record);
            return;
        }

        const cv::Size snapshot_size = boundedSize(depth.size(), _maximumLongEdge);
        const cv::Mat snapshot_depth = resizedNearest(depth, snapshot_size);
        const cv::Mat snapshot_confidence = resizedNearest(confidence, snapshot_size);
        cv::Mat authoritative_mask = valid_mask;
        if (authoritative_mask.empty() || authoritative_mask.size() != depth.size())
        {
            authoritative_mask = depth > 0.0f;
        }
        else if (authoritative_mask.type() != CV_8UC1)
        {
            authoritative_mask.convertTo(authoritative_mask, CV_8UC1);
        }
        cv::Mat snapshot_mask = resizedNearest(authoritative_mask, snapshot_size);
        snapshot_mask.setTo(cv::Scalar(0), snapshot_depth <= 0.0f);
        cv::Mat snapshot_reliability;
        if (result.depthLayerReliabilityClass &&
            result.depthLayerReliabilityClass->type() == CV_8UC1 &&
            result.depthLayerReliabilityClass->size() == depth.size())
        {
            snapshot_reliability = resizedNearest(
                *result.depthLayerReliabilityClass, snapshot_size);
        }
        cv::Mat snapshot_photometric_confidence;
        if (result.photometricConfidence &&
            result.photometricConfidence->type() == CV_32FC1 &&
            result.photometricConfidence->size() == depth.size())
        {
            snapshot_photometric_confidence = resizedNearest(
                *result.photometricConfidence, snapshot_size);
        }
        cv::Mat snapshot_geometric_confidence;
        if (result.geometricConfidence &&
            result.geometricConfidence->type() == CV_32FC1 &&
            result.geometricConfidence->size() == depth.size())
        {
            snapshot_geometric_confidence = resizedNearest(
                *result.geometricConfidence, snapshot_size);
        }
        cv::Mat snapshot_geometry_rerank;
        if (stage == MvsStageSnapshotStage::CrossViewConsistency &&
            result.geometryRerankMaps &&
            result.geometryRerankMaps->compatible(depth.size()))
        {
            cv::Mat source_count;
            cv::Mat baseline_count;
            cv::Mat decision_action;
            result.geometryRerankMaps->supportingSourceCount.convertTo(
                source_count, CV_32FC1);
            result.geometryRerankMaps->baselineSectorCount.convertTo(
                baseline_count, CV_32FC1);
            result.geometryRerankMaps->decisionAction.convertTo(
                decision_action, CV_32FC1);
            cv::Mat geometry_rerank;
            cv::merge(
                std::vector<cv::Mat>{
                    result.geometryRerankMaps->nativeCost,
                    result.geometryRerankMaps->candidateCost,
                    result.geometryRerankMaps->costAdvantage,
                    result.geometryRerankMaps->effectiveSourceWeight,
                    result.geometryRerankMaps->relativeCorrection,
                    result.geometryRerankMaps->weakestSourceConfidence,
                    source_count,
                    baseline_count,
                    decision_action},
                geometry_rerank);
            snapshot_geometry_rerank = resizedNearest(
                geometry_rerank, snapshot_size);
        }

        const std::uint64_t required_bytes =
            3ull * sizeof(StageDepthMatHeader) +
            static_cast<std::uint64_t>(snapshot_depth.total() * snapshot_depth.elemSize()) +
            static_cast<std::uint64_t>(snapshot_confidence.total() * snapshot_confidence.elemSize()) +
            static_cast<std::uint64_t>(snapshot_mask.total() * snapshot_mask.elemSize()) +
            (snapshot_photometric_confidence.empty()
                 ? 0ull
                 : sizeof(StageDepthMatHeader) +
                       static_cast<std::uint64_t>(
                           snapshot_photometric_confidence.total() *
                           snapshot_photometric_confidence.elemSize())) +
            (snapshot_geometric_confidence.empty()
                 ? 0ull
                 : sizeof(StageDepthMatHeader) +
                       static_cast<std::uint64_t>(
                           snapshot_geometric_confidence.total() *
                           snapshot_geometric_confidence.elemSize())) +
            (snapshot_reliability.empty()
                 ? 0ull
                 : sizeof(StageDepthMatHeader) +
                       static_cast<std::uint64_t>(
                           snapshot_reliability.total() *
                           snapshot_reliability.elemSize())) +
            (snapshot_geometry_rerank.empty()
                 ? 0ull
                 : sizeof(StageDepthMatHeader) +
                       static_cast<std::uint64_t>(
                           snapshot_geometry_rerank.total() *
                           snapshot_geometry_rerank.elemSize()));
        if (required_bytes > _budgetBytes - std::min(_usedBytes, _budgetBytes))
        {
            record.insert(QStringLiteral("status"), QStringLiteral("skipped"));
            record.insert(QStringLiteral("reason"), QStringLiteral("snapshot_budget_exhausted"));
            record.insert(QStringLiteral("required_bytes"), static_cast<double>(required_bytes));
            appendRecordLocked(record);
            return;
        }

        const QString stage_id = mvsStageSnapshotStageId(stage);
        const QString frame_directory = QDir(_directory).filePath(
            QStringLiteral("ref_%1").arg(reference_index, 4, 10, QLatin1Char('0')));
        if (!QDir().mkpath(frame_directory))
        {
            record.insert(QStringLiteral("status"), QStringLiteral("failed"));
            record.insert(QStringLiteral("reason"), QStringLiteral("cannot_create_frame_directory"));
            appendRecordLocked(record);
            return;
        }
        const QString prefix = QDir(frame_directory).filePath(stage_id);
        const QString depth_path = prefix + QStringLiteral("_depth.bin");
        const QString confidence_path = prefix + QStringLiteral("_confidence.bin");
        const QString photometric_confidence_path =
            prefix + QStringLiteral("_photometric_confidence.bin");
        const QString geometric_confidence_path =
            prefix + QStringLiteral("_geometric_confidence.bin");
        const QString mask_path = prefix + QStringLiteral("_valid_mask.bin");
        const QString reliability_path =
            prefix + QStringLiteral("_depth_layer_reliability.bin");
        const QString geometry_rerank_path =
            prefix + QStringLiteral("_geometry_rerank.bin");
        QString write_error;
        const bool depth_ok = writeMatrixAtomically(depth_path, snapshot_depth, &write_error);
        const bool confidence_ok = depth_ok && writeMatrixAtomically(
            confidence_path, snapshot_confidence, &write_error);
        const bool mask_ok = confidence_ok && writeMatrixAtomically(
            mask_path, snapshot_mask, &write_error);
        const bool photometric_confidence_ok = mask_ok &&
            (snapshot_photometric_confidence.empty() || writeMatrixAtomically(
                 photometric_confidence_path,
                 snapshot_photometric_confidence,
                 &write_error));
        const bool geometric_confidence_ok = photometric_confidence_ok &&
            (snapshot_geometric_confidence.empty() || writeMatrixAtomically(
                 geometric_confidence_path,
                 snapshot_geometric_confidence,
                 &write_error));
        const bool reliability_ok = geometric_confidence_ok &&
            (snapshot_reliability.empty() || writeMatrixAtomically(
                 reliability_path, snapshot_reliability, &write_error));
        const bool geometry_rerank_ok = reliability_ok &&
            (snapshot_geometry_rerank.empty() || writeMatrixAtomically(
                 geometry_rerank_path, snapshot_geometry_rerank, &write_error));
        if (!depth_ok || !confidence_ok || !mask_ok ||
            !photometric_confidence_ok || !geometric_confidence_ok ||
            !reliability_ok || !geometry_rerank_ok)
        {
            QFile::remove(depth_path);
            QFile::remove(confidence_path);
            QFile::remove(photometric_confidence_path);
            QFile::remove(geometric_confidence_path);
            QFile::remove(mask_path);
            QFile::remove(reliability_path);
            QFile::remove(geometry_rerank_path);
            record.insert(QStringLiteral("status"), QStringLiteral("failed"));
            record.insert(QStringLiteral("reason"), write_error);
            appendRecordLocked(record);
            return;
        }

        _usedBytes += required_bytes;
        FramePinholeCamera camera = result.cameraModel;
        if (camera.isValid() && snapshot_size != depth.size())
        {
            camera = camera.scaledIntrinsics(
                static_cast<double>(snapshot_size.width) / depth.cols,
                static_cast<double>(snapshot_size.height) / depth.rows);
        }
        record.insert(QStringLiteral("status"), QStringLiteral("captured"));
        record.insert(QStringLiteral("original_width"), depth.cols);
        record.insert(QStringLiteral("original_height"), depth.rows);
        record.insert(QStringLiteral("snapshot_width"), snapshot_size.width);
        record.insert(QStringLiteral("snapshot_height"), snapshot_size.height);
        record.insert(QStringLiteral("valid_pixel_count"), cv::countNonZero(snapshot_mask));
        record.insert(QStringLiteral("effective_native_final_depth_grid"),
                      result.effectiveNativeFinalDepthGrid);
        record.insert(QStringLiteral("camera_model"),
                      camera.isValid() ? cameraToJson(camera) : QJsonObject{});
        record.insert(QStringLiteral("pixel_domain_diagnostics"),
                      result.pixelDomainDiagnostics);
        record.insert(QStringLiteral("quality_metrics"),
                      depthMapQualityMetricsToJson(result.qualityMetrics));
        record.insert(QStringLiteral("quality_decision"),
                      depthFrameQualityDecisionToJson(result.qualityDecision));
        record.insert(QStringLiteral("depth_completeness"),
                      depthCompletenessDiagnosticsToJson(result.depthCompleteness));
        record.insert(QStringLiteral("depth"), artifactToJson(depth_path, snapshot_depth));
        record.insert(QStringLiteral("confidence"),
                      artifactToJson(confidence_path, snapshot_confidence));
        if (!snapshot_photometric_confidence.empty())
        {
            record.insert(
                QStringLiteral("photometric_confidence"),
                artifactToJson(
                    photometric_confidence_path,
                    snapshot_photometric_confidence));
        }
        if (!snapshot_geometric_confidence.empty())
        {
            record.insert(
                QStringLiteral("geometric_confidence"),
                artifactToJson(
                    geometric_confidence_path,
                    snapshot_geometric_confidence));
        }
        record.insert(QStringLiteral("valid_mask"), artifactToJson(mask_path, snapshot_mask));
        if (!snapshot_reliability.empty())
        {
            record.insert(
                QStringLiteral("depth_layer_reliability"),
                artifactToJson(reliability_path, snapshot_reliability));
        }
        if (!snapshot_geometry_rerank.empty())
        {
            QJsonObject artifact = artifactToJson(
                geometry_rerank_path, snapshot_geometry_rerank);
            artifact.insert(
                QStringLiteral("channel_order"),
                QJsonArray{
                    QStringLiteral("native_cost"),
                    QStringLiteral("candidate_cost"),
                    QStringLiteral("cost_advantage"),
                    QStringLiteral("effective_source_weight"),
                    QStringLiteral("relative_correction"),
                    QStringLiteral("weakest_source_confidence"),
                    QStringLiteral("supporting_source_count"),
                    QStringLiteral("baseline_sector_count"),
                    QStringLiteral("decision_action")});
            record.insert(QStringLiteral("geometry_rerank"), artifact);
        }
        appendRecordLocked(record);
    }
    catch (const std::exception &error)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _errors.append(QStringLiteral("capture_exception:%1")
                           .arg(QString::fromUtf8(error.what())));
        QString ignored;
        writeManifestLocked(&ignored);
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _errors.append(QStringLiteral("capture_exception:unknown"));
        QString ignored;
        writeManifestLocked(&ignored);
    }
}

void MvsStageSnapshotRecorder::appendRecordLocked(const QJsonObject &record)
{
    _records.push_back(record);
    QString error;
    if (!writeManifestLocked(&error)) _errors.append(error);
}

bool MvsStageSnapshotRecorder::writeManifestLocked(QString *error_message)
{
    std::vector<QJsonObject> records = _records;
    std::sort(records.begin(), records.end(), [](const QJsonObject &left,
                                                 const QJsonObject &right)
    {
        const int left_ref = left.value(QStringLiteral("ref_index")).toInt(-1);
        const int right_ref = right.value(QStringLiteral("ref_index")).toInt(-1);
        return left_ref != right_ref ? left_ref < right_ref
                                     : stageOrder(left) < stageOrder(right);
    });
    QJsonArray record_array;
    for (const QJsonObject &record : records) record_array.append(record);
    QJsonArray selected_array;
    for (int reference_index : _selectedReferences) selected_array.append(reference_index);
    QJsonArray stage_array;
    for (int stage = 0; stage < 4; ++stage)
    {
        stage_array.append(mvsStageSnapshotStageId(
            static_cast<MvsStageSnapshotStage>(stage)));
    }
    const QJsonObject manifest{
        {QStringLiteral("schema"), QStringLiteral("plascan_mvs_stage_snapshots_v1")},
        {QStringLiteral("status"), !_errors.isEmpty()
             ? QStringLiteral("diagnostic_error")
             : (_finalized ? QStringLiteral("complete") : QStringLiteral("active"))},
        {QStringLiteral("authoritative"), false},
        {QStringLiteral("selected_ref_indices"), selected_array},
        {QStringLiteral("expected_stages"), stage_array},
        {QStringLiteral("maximum_long_edge"), _maximumLongEdge},
        {QStringLiteral("budget_bytes"), static_cast<double>(_budgetBytes)},
        {QStringLiteral("used_bytes"), static_cast<double>(_usedBytes)},
        {QStringLiteral("records"), record_array},
        {QStringLiteral("finalized"), _finalized},
        {QStringLiteral("errors"), _errors}};
    QSaveFile file(_manifestPath);
    file.setDirectWriteFallback(false);
    if (!file.open(QIODevice::WriteOnly) ||
        file.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented)) < 0 ||
        !file.commit())
    {
        if (error_message)
        {
            *error_message = QStringLiteral("无法原子写入阶段快照 manifest：%1")
                                 .arg(_manifestPath);
        }
        return false;
    }
    return true;
}

void MvsStageSnapshotRecorder::finalize() noexcept
{
    try
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_enabled || _finalized) return;
        for (int reference_index : _selectedReferences)
        {
            for (int stage_index = 0; stage_index < 4; ++stage_index)
            {
                const auto stage = static_cast<MvsStageSnapshotStage>(stage_index);
                const QString key = stageKey(reference_index, stage);
                if (_recordedStageKeys.contains(key)) continue;
                _recordedStageKeys.insert(key);
                _records.push_back(QJsonObject{
                    {QStringLiteral("ref_index"), reference_index},
                    {QStringLiteral("stage"), mvsStageSnapshotStageId(stage)},
                    {QStringLiteral("boundary"), boundaryForStage(stage)},
                    {QStringLiteral("status"), QStringLiteral("missing")},
                    {QStringLiteral("reason"), QStringLiteral("stage_not_reached")}});
            }
        }
        _finalized = true;
        QString error;
        if (!writeManifestLocked(&error)) _errors.append(error);
    }
    catch (...)
    {
    }
}

} // namespace xjw::mvs
