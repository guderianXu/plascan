#include "MvsWorkspaceManifest.h"

#include "DepthFrameQualityGate.h"
#include "MvsTypes.h"
#include "io/PathIO.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QThread>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace xjw::mvs
{

namespace
{
constexpr std::array<char, 16> kFastDepthMatMagic{
    'P', 'L', 'A', 'S', 'D', 'E', 'P', 'T', 'H', 'M', 'A', 'T', '0', '1', '\0', '\0'};

struct FastDepthMatHeader
{
    char magic[16] = {};
    qint32 rows = 0;
    qint32 cols = 0;
    qint32 type = 0;
    quint32 reserved = 0;
    quint64 dataBytes = 0;
};

static_assert(sizeof(FastDepthMatHeader) == 40, "Fast depth matrix header layout must remain backward compatible");

struct FastDepthMatShape
{
    int rows = 0;
    int cols = 0;
    int type = 0;
};

struct PngShape
{
    int rows = 0;
    int cols = 0;
};

enum class ConsistencyPublicationState
{
    Unavailable,
    Completed,
    Invalid
};

ConsistencyPublicationState consistencyPublicationState(const QJsonObject& diagnostics)
{
    constexpr std::array<const char*, 3> kRequiredCountKeys{
        "pre_consistency_valid_count", "post_consistency_valid_count", "published_post_consistency_valid_count"};
    int present_count = 0;
    for (const char* key : kRequiredCountKeys)
    {
        const QString json_key = QString::fromLatin1(key);
        if (!diagnostics.contains(json_key))
        {
            continue;
        }
        ++present_count;
        const QJsonValue value = diagnostics.value(json_key);
        const double numeric_value = value.toDouble(-1.0);
        if (!value.isDouble() || !std::isfinite(numeric_value) || numeric_value < 0.0 ||
            std::floor(numeric_value) != numeric_value ||
            numeric_value > static_cast<double>(std::numeric_limits<int>::max()))
        {
            return ConsistencyPublicationState::Invalid;
        }
    }
    if (present_count == 0)
    {
        return diagnostics.value(QStringLiteral("consistency_publication_fallback_applied")).toBool(false)
                   ? ConsistencyPublicationState::Invalid
                   : ConsistencyPublicationState::Unavailable;
    }
    return present_count == static_cast<int>(kRequiredCountKeys.size()) ? ConsistencyPublicationState::Completed
                                                                        : ConsistencyPublicationState::Invalid;
}

template <typename T> void addHashValue(QCryptographicHash* hash, const T& value)
{
    const QByteArrayView bytes(reinterpret_cast<const char*>(&value), static_cast<qsizetype>(sizeof(value)));
    hash->addData(bytes);
}

void addFramedHashData(QCryptographicHash* hash, const QByteArray& data)
{
    const qint64 size = data.size();
    addHashValue(hash, size);
    hash->addData(data);
}

QString normalizedInputPath(const std::string& path)
{
    const QString raw_path = xjw::common::io::fromUtf8Path(path).trimmed();
    if (raw_path.isEmpty())
    {
        return QString();
    }

    const QFileInfo file_info(raw_path);
    QString normalized_path = file_info.exists() ? file_info.canonicalFilePath() : file_info.absoluteFilePath();
    if (normalized_path.isEmpty())
    {
        normalized_path = raw_path;
    }
    normalized_path = QDir::cleanPath(normalized_path).replace(QLatin1Char('\\'), QLatin1Char('/'));
#ifdef Q_OS_WIN
    normalized_path = normalized_path.toCaseFolded();
#endif
    return normalized_path;
}

QByteArray fileContentFingerprint(QFile* file, qint64 file_size, bool* readable)
{
    constexpr qint64 kFullHashMaximumBytes = 8LL * 1024LL * 1024LL;
    constexpr qint64 kSampleBytes = 64LL * 1024LL;

    QCryptographicHash content_hash(QCryptographicHash::Sha256);
    *readable = false;
    if (!file || !file->isOpen() || file_size < 0)
    {
        return QByteArray();
    }

    if (file_size <= kFullHashMaximumBytes)
    {
        content_hash.addData(QByteArrayLiteral("full\0"));
        const QByteArray contents = file->readAll();
        if (contents.size() != file_size)
        {
            return QByteArray();
        }
        content_hash.addData(contents);
        *readable = true;
        return content_hash.result();
    }

    content_hash.addData(QByteArrayLiteral("sampled\0"));
    const std::array<qint64, 3> offsets{
        0, std::max<qint64>(0, file_size / 2 - kSampleBytes / 2), std::max<qint64>(0, file_size - kSampleBytes)};
    for (const qint64 offset : offsets)
    {
        if (!file->seek(offset))
        {
            return QByteArray();
        }
        const qint64 bytes_to_read = std::min(kSampleBytes, file_size - offset);
        const QByteArray sample = file->read(bytes_to_read);
        if (sample.size() != bytes_to_read)
        {
            return QByteArray();
        }
        addHashValue(&content_hash, offset);
        addFramedHashData(&content_hash, sample);
    }
    *readable = true;
    return content_hash.result();
}

void addFileFingerprint(QCryptographicHash* hash, const char* role, const std::string& path)
{
    addFramedHashData(hash, QByteArray(role));
    const QString normalized_path = normalizedInputPath(path);
    addFramedHashData(hash, normalized_path.toUtf8());

    const QString io_path = xjw::common::io::fromUtf8Path(path).trimmed();
    const QFileInfo file_info(io_path);
    const bool exists = !io_path.isEmpty() && file_info.exists() && file_info.isFile();
    addHashValue(hash, exists);
    const qint64 file_size = exists ? file_info.size() : -1;
    const qint64 modified_msecs = exists ? file_info.lastModified().toMSecsSinceEpoch() : -1;
    addHashValue(hash, file_size);
    addHashValue(hash, modified_msecs);

    bool readable = false;
    QByteArray content_fingerprint;
    if (exists)
    {
        QFile file(io_path);
        if (file.open(QIODevice::ReadOnly))
        {
            content_fingerprint = fileContentFingerprint(&file, file_size, &readable);
        }
    }
    addHashValue(hash, readable);
    addFramedHashData(hash, content_fingerprint);
}

bool isNonEmptyFile(const QString& path)
{
    const QFileInfo file_info(path);
    return !path.trimmed().isEmpty() && file_info.exists() && file_info.isFile() && file_info.size() > 0;
}

quint32 readPngUint32(const char* bytes)
{
    return (static_cast<quint32>(static_cast<unsigned char>(bytes[0])) << 24) |
           (static_cast<quint32>(static_cast<unsigned char>(bytes[1])) << 16) |
           (static_cast<quint32>(static_cast<unsigned char>(bytes[2])) << 8) |
           static_cast<quint32>(static_cast<unsigned char>(bytes[3]));
}

bool readPngShape(const QString& path, PngShape* shape)
{
    constexpr std::array<unsigned char, 8> kPngSignature{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    if (!shape || !isNonEmptyFile(path))
    {
        return false;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }

    const QByteArray header = file.read(33);
    if (header.size() != 33 || std::memcmp(header.constData(), kPngSignature.data(), kPngSignature.size()) != 0 ||
        readPngUint32(header.constData() + 8) != 13 || std::memcmp(header.constData() + 12, "IHDR", 4) != 0)
    {
        return false;
    }

    const quint32 width = readPngUint32(header.constData() + 16);
    const quint32 height = readPngUint32(header.constData() + 20);
    const int bit_depth = static_cast<unsigned char>(header[24]);
    const int color_type = static_cast<unsigned char>(header[25]);
    const bool valid_bit_depth =
        (color_type == 0 &&
         (bit_depth == 1 || bit_depth == 2 || bit_depth == 4 || bit_depth == 8 || bit_depth == 16)) ||
        (color_type == 2 && (bit_depth == 8 || bit_depth == 16)) ||
        (color_type == 3 && (bit_depth == 1 || bit_depth == 2 || bit_depth == 4 || bit_depth == 8)) ||
        (color_type == 4 && (bit_depth == 8 || bit_depth == 16)) ||
        (color_type == 6 && (bit_depth == 8 || bit_depth == 16));
    if (width == 0 || height == 0 || width > static_cast<quint32>(std::numeric_limits<int>::max()) ||
        height > static_cast<quint32>(std::numeric_limits<int>::max()) || !valid_bit_depth || header[26] != 0 ||
        header[27] != 0 || (header[28] != 0 && header[28] != 1))
    {
        return false;
    }

    shape->rows = static_cast<int>(height);
    shape->cols = static_cast<int>(width);
    bool saw_image_data = false;
    while (!file.atEnd())
    {
        const QByteArray chunk_header = file.read(8);
        if (chunk_header.size() != 8)
        {
            return false;
        }
        const quint32 chunk_size = readPngUint32(chunk_header.constData());
        const QByteArray chunk_type = chunk_header.mid(4, 4);
        const qint64 remaining = file.size() - file.pos();
        if (remaining < static_cast<qint64>(chunk_size) + 4 ||
            !file.seek(file.pos() + static_cast<qint64>(chunk_size) + 4))
        {
            return false;
        }
        if (chunk_type == QByteArrayLiteral("IDAT"))
        {
            saw_image_data = true;
        }
        if (chunk_type == QByteArrayLiteral("IEND"))
        {
            return chunk_size == 0 && saw_image_data && file.atEnd();
        }
    }
    return false;
}

bool readFastDepthMatShape(const QString& path, FastDepthMatShape* shape)
{
    if (!shape || !isNonEmptyFile(path))
    {
        return false;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }

    FastDepthMatHeader header;
    if (file.read(reinterpret_cast<char*>(&header), sizeof(header)) != static_cast<qint64>(sizeof(header)) ||
        std::memcmp(header.magic, kFastDepthMatMagic.data(), kFastDepthMatMagic.size()) != 0 || header.rows <= 0 ||
        header.cols <= 0 || header.type < 0 || header.dataBytes == 0)
    {
        return false;
    }

    const int depth = CV_MAT_DEPTH(header.type);
    const int channels = CV_MAT_CN(header.type);
    if (depth < CV_8U || depth > CV_64F || channels <= 0 || channels > CV_CN_MAX ||
        header.type != CV_MAKETYPE(depth, channels))
    {
        return false;
    }

    const quint64 element_size = static_cast<quint64>(CV_ELEM_SIZE(header.type));
    const quint64 rows = static_cast<quint64>(header.rows);
    const quint64 cols = static_cast<quint64>(header.cols);
    if (element_size == 0 || rows > std::numeric_limits<quint64>::max() / cols ||
        rows * cols > std::numeric_limits<quint64>::max() / element_size)
    {
        return false;
    }

    const quint64 expected_data_bytes = rows * cols * element_size;
    if (expected_data_bytes > std::numeric_limits<quint64>::max() - static_cast<quint64>(sizeof(header)))
    {
        return false;
    }
    const quint64 expected_file_bytes = static_cast<quint64>(sizeof(header)) + expected_data_bytes;
    if (header.dataBytes != expected_data_bytes ||
        expected_file_bytes > static_cast<quint64>(std::numeric_limits<qint64>::max()) ||
        QFileInfo(path).size() != static_cast<qint64>(expected_file_bytes))
    {
        return false;
    }

    shape->rows = header.rows;
    shape->cols = header.cols;
    shape->type = header.type;
    return true;
}

    QString frameSortName(const MvsDepthFrameRecord &record)
{
    if (!record.refImage.isEmpty())
    {
        return QFileInfo(record.refImage).fileName();
    }
    if (!record.depthPng.isEmpty())
    {
        return QFileInfo(record.depthPng).fileName();
    }
    return QString::number(record.refIndex);
}

QStringList jsonArrayToStringList(const QJsonArray &array)
{
    QStringList result;
    result.reserve(array.size());
    for (const QJsonValue &value : array)
    {
        result.push_back(value.toString());
    }
    return result;
}

QJsonArray stringListToJsonArray(const QStringList &strings)
{
    QJsonArray array;
    for (const QString &string : strings)
    {
        array.push_back(string);
    }
    return array;
}

bool containsReason(const QJsonArray &reasons, const QString &expected)
{
    return std::any_of(
        reasons.cbegin(), reasons.cend(), [&expected](const QJsonValue &value)
        {
            return value.toString() == expected;
        });
}

bool hasConsistentQualityDecisionAcceptance(
    const QString &acceptance,
    const QJsonObject &quality_decision)
{
    if (!quality_decision.contains(QStringLiteral("acceptance")))
    {
        return true;
    }
    const QJsonValue nested_acceptance = quality_decision.value(
        QStringLiteral("acceptance"));
    return nested_acceptance.isString() &&
        nested_acceptance.toString().trimmed().compare(
            acceptance.trimmed(), Qt::CaseInsensitive) == 0;
}

}

MvsDepthFrameQualification qualifyMvsDepthFrameArtifact(
    const QJsonObject &artifact)
{
    const QJsonObject quality_decision = artifact.value(
        QStringLiteral("quality_decision")).toObject();
    MvsDepthFrameQualification qualification;
    qualification.acceptance = artifact.value(
        QStringLiteral("acceptance")).toString();
    const QJsonValue fusion_eligible = artifact.value(
        QStringLiteral("fusion_eligible"));
    qualification.fusionEligibilityKnown = fusion_eligible.isBool();
    qualification.fusionEligible =
        qualification.fusionEligibilityKnown && fusion_eligible.toBool();
    qualification.role = hasConsistentQualityDecisionAcceptance(
                             qualification.acceptance,
                             quality_decision)
        ? qualifyDepthFrameRole(
              qualification.acceptance,
              qualification.fusionEligibilityKnown,
              qualification.fusionEligible,
              artifact.value(QStringLiteral("status")).toString())
        : DepthFrameRole::Excluded;

    const QJsonArray quality_reasons = quality_decision.value(
        QStringLiteral("reasons")).toArray();
    qualification.useDiscreteGeometryFallback = containsReason(
        quality_reasons,
        QStringLiteral("adaptive_geometry_fallback_to_discrete_core"));

    return qualification;
}

QJsonObject MvsDepthFrameRecord::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("ref_index"), refIndex);
    object.insert(QStringLiteral("ref_image"), refImage);
    if (!preparedImage.isEmpty())
    {
        object.insert(QStringLiteral("prepared_image"), preparedImage);
    }
    if (!preparedValidMaskPath.isEmpty())
    {
        object.insert(
            QStringLiteral("prepared_valid_mask_path"),
            preparedValidMaskPath);
    }
    if (!preparedCameraModel.isEmpty())
    {
        object.insert(
            QStringLiteral("prepared_camera_model"),
            preparedCameraModel);
    }
    object.insert(QStringLiteral("source_images"), stringListToJsonArray(sourceImages));
    QJsonArray source_indices;
    for (const int source_index : sourceIndices)
    {
        source_indices.append(source_index);
    }
    object.insert(QStringLiteral("source_indices"), source_indices);
    QJsonArray geometry_source_indices;
    for (const int source_index : geometrySourceIndices)
    {
        geometry_source_indices.append(source_index);
    }
    object.insert(QStringLiteral("geometry_source_indices"), geometry_source_indices);
    object.insert(QStringLiteral("source_plan"), sourcePlan);
    object.insert(QStringLiteral("quality_profile"), qualityProfile);
    object.insert(QStringLiteral("configured_source_view_count"), configuredSourceViewCount);
    object.insert(QStringLiteral("source_view_count"), sourceViewCount);
    object.insert(QStringLiteral("requested_source_view_count"), requestedSourceViewCount);
    object.insert(QStringLiteral("source_view_shortfall"), sourceViewShortfall);
    object.insert(QStringLiteral("source_view_shortfall_reason"), sourceViewShortfallReason);
    object.insert(QStringLiteral("consistency_publication_expected"), consistencyPublicationExpected);
    object.insert(QStringLiteral("geometric_guidance_pass_expected"), geometricGuidancePassExpected);
    object.insert(QStringLiteral("geometric_guidance_pass_applied"), geometricGuidancePassApplied);
    object.insert(QStringLiteral("source_quality_mean"), meanSourceQualityScore);
    object.insert(QStringLiteral("source_quality_min"), minSourceQualityScore);
    object.insert(QStringLiteral("depth_confidence_mean"), meanDepthConfidence);
    object.insert(QStringLiteral("effective_patch_match_confidence_threshold"), effectivePatchMatchConfidenceThreshold);
    object.insert(QStringLiteral("valid_pixel_count"), validPixelCount);
    if (validCoverage >= 0.0 && validCoverage <= 1.0 && std::isfinite(validCoverage))
    {
        object.insert(QStringLiteral("valid_coverage"), validCoverage);
    }
    object.insert(QStringLiteral("depth_quality"), depthQuality);
    object.insert(QStringLiteral("depth_completeness"), depthCompleteness);
    object.insert(QStringLiteral("missing_reason_summary"),
                  missingReasonSummary);
    object.insert(
        QStringLiteral("cross_view_repair_diagnostics"),
        crossViewRepairDiagnostics);
    object.insert(
        QStringLiteral("targeted_gap_recovery_diagnostics"),
        targetedGapRecoveryDiagnostics);
    object.insert(
        QStringLiteral("residual_reestimation_diagnostics"),
        residualReestimationDiagnostics);
    object.insert(
        QStringLiteral("learned_candidate_diagnostics"),
        learnedCandidateDiagnostics);
    object.insert(QStringLiteral("depth_provenance_summary"),
                  depthProvenanceSummary);
    object.insert(
        QStringLiteral("geometry_evidence_diagnostics"),
        geometryEvidenceDiagnostics);
    object.insert(QStringLiteral("pose_refinement_diagnostics"),
                  poseRefinementDiagnostics);
    if (!derivedCameraModel.isEmpty())
    {
        object.insert(QStringLiteral("derived_camera_model"), derivedCameraModel);
    }
    object.insert(QStringLiteral("quality_decision"), qualityDecision);
    object.insert(QStringLiteral("pyramid_levels"), pyramidLevels);
    object.insert(QStringLiteral("mask_source"), maskSource);
    object.insert(QStringLiteral("mask_coverage"), maskCoverage);
    object.insert(QStringLiteral("selected_level"), selectedLevel);
    object.insert(QStringLiteral("fallback_reason"), fallbackReason);
    object.insert(QStringLiteral("pyramid_requested_level_count"),
                  pyramidRequestedLevelCount);
    object.insert(QStringLiteral("pyramid_active_level_count"),
                  pyramidActiveLevelCount);
    object.insert(QStringLiteral("pyramid_minimum_short_side"),
                  pyramidMinimumShortSide);
    object.insert(QStringLiteral("pyramid_degraded_reason"),
                  pyramidDegradedReason);
    object.insert(QStringLiteral("scene_profile"), sceneProfile);
    object.insert(QStringLiteral("filter_mode"), filterMode);
    object.insert(QStringLiteral("acceptance"), acceptance);
    if (fusionEligibilityKnown)
    {
        object.insert(QStringLiteral("fusion_eligible"), fusionEligible);
    }
    object.insert(QStringLiteral("depth_postprocess"), depthPostprocess);
    object.insert(QStringLiteral("camera_model"), cameraModel);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("device"), device);
    object.insert(QStringLiteral("depth_png"), depthPng);
    object.insert(QStringLiteral("raw_depth_path"), rawDepthPath);
    object.insert(QStringLiteral("raw_confidence_path"), rawConfidencePath);
    object.insert(QStringLiteral("raw_photometric_source_mask_path"),
                  rawPhotometricSourceMaskPath);
    object.insert(QStringLiteral("raw_geometry_support_path"), rawGeometrySupportPath);
    object.insert(QStringLiteral("raw_adaptive_geometry_support_weight_path"),
                  rawAdaptiveGeometrySupportWeightPath);
    object.insert(QStringLiteral("raw_adaptive_geometry_effective_view_count_path"),
                  rawAdaptiveGeometryEffectiveViewCountPath);
    object.insert(QStringLiteral("raw_adaptive_geometry_conflict_ratio_path"),
                  rawAdaptiveGeometryConflictRatioPath);
    object.insert(QStringLiteral("raw_geometry_source_mask_path"), rawGeometrySourceMaskPath);
    object.insert(QStringLiteral("raw_inverse_depth_mean_path"), rawInverseDepthMeanPath);
    object.insert(QStringLiteral("raw_inverse_depth_spread_path"), rawInverseDepthSpreadPath);
    object.insert(QStringLiteral("cross_view_repaired_mask_path"), crossViewRepairedMaskPath);
    object.insert(QStringLiteral("targeted_gap_recovered_mask_path"),
                  targetedGapRecoveredMaskPath);
    object.insert(QStringLiteral("residual_reestimated_mask_path"),
                  residualReestimatedMaskPath);
    object.insert(QStringLiteral("depth_provenance_path"),
                  depthProvenancePath);
    object.insert(QStringLiteral("valid_mask_path"), validMaskPath);
    object.insert(QStringLiteral("support_mask_path"), supportMaskPath);
    object.insert(QStringLiteral("missing_reason_path"), missingReasonPath);
    object.insert(QStringLiteral("missing_reason_preview_path"),
                  missingReasonPreviewPath);
    object.insert(QStringLiteral("effective_native_final_depth_grid"),
                  effectiveNativeFinalDepthGrid);
    object.insert(QStringLiteral("pixel_domain_diagnostics"),
                  pixelDomainDiagnostics);
    object.insert(QStringLiteral("grid_width"), gridWidth);
    object.insert(QStringLiteral("grid_height"), gridHeight);
    object.insert(QStringLiteral("elapsed_ms"), QString::number(elapsedMs));
    object.insert(QStringLiteral("error"), error);
    object.insert(QStringLiteral("config_hash"), configHash);
    object.insert(QStringLiteral("algorithm_revision"), algorithmRevision);
    return object;
}

MvsDepthFrameRecord MvsDepthFrameRecord::fromJson(const QJsonObject &object)
{
    MvsDepthFrameRecord record;
    record.refIndex = object.value(QStringLiteral("ref_index")).toInt(-1);
    record.refImage = object.value(QStringLiteral("ref_image")).toString();
    record.preparedImage = object.value(
        QStringLiteral("prepared_image")).toString();
    record.preparedValidMaskPath = object.value(
        QStringLiteral("prepared_valid_mask_path")).toString();
    record.preparedCameraModel = object.value(
        QStringLiteral("prepared_camera_model")).toObject();
    record.sourceImages = jsonArrayToStringList(object.value(QStringLiteral("source_images")).toArray());
    for (const QJsonValue &value : object.value(QStringLiteral("source_indices")).toArray())
    {
        record.sourceIndices.push_back(value.toInt(-1));
    }
    for (const QJsonValue &value :
         object.value(QStringLiteral("geometry_source_indices")).toArray())
    {
        record.geometrySourceIndices.push_back(value.toInt(-1));
    }
    record.sourcePlan = object.value(QStringLiteral("source_plan")).toArray();
    record.qualityProfile = object.value(QStringLiteral("quality_profile")).toString();
    record.configuredSourceViewCount = object.value(QStringLiteral("configured_source_view_count"))
                                           .toInt(object.value(QStringLiteral("requested_source_view_count")).toInt(0));
    record.sourceViewCount = object.value(QStringLiteral("source_view_count")).toInt(0);
    record.requestedSourceViewCount =
        object.value(QStringLiteral("requested_source_view_count")).toInt(record.sourceViewCount);
    record.sourceViewShortfall = object.value(QStringLiteral("source_view_shortfall"))
                                     .toInt(std::max(0, record.requestedSourceViewCount - record.sourceViewCount));
    record.sourceViewShortfallReason = object.value(QStringLiteral("source_view_shortfall_reason")).toString();
    record.consistencyPublicationExpected =
        object.value(QStringLiteral("consistency_publication_expected")).toBool(false);
    record.geometricGuidancePassExpected =
        object.value(QStringLiteral("geometric_guidance_pass_expected")).toBool(false);
    record.geometricGuidancePassApplied = object.value(QStringLiteral("geometric_guidance_pass_applied")).toBool(false);
    record.meanSourceQualityScore = object.value(QStringLiteral("source_quality_mean")).toDouble(0.0);
    record.minSourceQualityScore = object.value(QStringLiteral("source_quality_min")).toDouble(0.0);
    record.meanDepthConfidence = object.value(QStringLiteral("depth_confidence_mean")).toDouble(0.0);
    record.effectivePatchMatchConfidenceThreshold =
        object.value(QStringLiteral("effective_patch_match_confidence_threshold")).toDouble(0.0);
    record.validPixelCount = object.value(QStringLiteral("valid_pixel_count")).toInt(0);
    record.validCoverage = object.value(QStringLiteral("valid_coverage")).toDouble(-1.0);
    record.depthQuality = object.value(QStringLiteral("depth_quality")).toObject();
    record.depthCompleteness = object.value(
        QStringLiteral("depth_completeness")).toObject();
    record.missingReasonSummary = object.value(
        QStringLiteral("missing_reason_summary")).toObject();
    record.crossViewRepairDiagnostics = object.value(
        QStringLiteral("cross_view_repair_diagnostics")).toObject();
    record.targetedGapRecoveryDiagnostics = object.value(
        QStringLiteral("targeted_gap_recovery_diagnostics")).toObject();
    record.residualReestimationDiagnostics = object.value(
        QStringLiteral("residual_reestimation_diagnostics")).toObject();
    record.learnedCandidateDiagnostics = object.value(
        QStringLiteral("learned_candidate_diagnostics")).toObject();
    record.depthProvenanceSummary = object.value(
        QStringLiteral("depth_provenance_summary")).toObject();
    record.geometryEvidenceDiagnostics = object.value(
        QStringLiteral("geometry_evidence_diagnostics")).toObject();
    record.poseRefinementDiagnostics = object.value(
        QStringLiteral("pose_refinement_diagnostics")).toObject();
    record.derivedCameraModel = object.value(
        QStringLiteral("derived_camera_model")).toObject();
    record.qualityDecision = object.value(QStringLiteral("quality_decision")).toObject();
    record.pyramidLevels = object.value(QStringLiteral("pyramid_levels")).toArray();
    record.maskSource = object.value(QStringLiteral("mask_source")).toString();
    record.maskCoverage = object.value(QStringLiteral("mask_coverage")).toDouble(-1.0);
    record.selectedLevel = object.value(QStringLiteral("selected_level")).toInt(0);
    record.fallbackReason = object.value(QStringLiteral("fallback_reason")).toString();
    record.pyramidRequestedLevelCount = object.value(
        QStringLiteral("pyramid_requested_level_count")).toInt(3);
    record.pyramidActiveLevelCount = object.value(
        QStringLiteral("pyramid_active_level_count")).toInt(
            record.pyramidLevels.size());
    record.pyramidMinimumShortSide = object.value(
        QStringLiteral("pyramid_minimum_short_side")).toInt(0);
    record.pyramidDegradedReason = object.value(
        QStringLiteral("pyramid_degraded_reason")).toString();
    record.sceneProfile = canonicalDepthSceneProfile(
        object.value(QStringLiteral("scene_profile")).toString());
    record.filterMode = object.value(QStringLiteral("filter_mode")).toString();
    record.depthPostprocess = object.value(QStringLiteral("depth_postprocess")).toObject();
    record.cameraModel = object.value(QStringLiteral("camera_model")).toObject();
    record.status = object.value(QStringLiteral("status")).toString();
    record.device = object.value(QStringLiteral("device")).toString();
    record.depthPng = object.value(QStringLiteral("depth_png")).toString();
    record.rawDepthPath = object.value(QStringLiteral("raw_depth_path")).toString();
    record.rawConfidencePath = object.value(QStringLiteral("raw_confidence_path")).toString();
    record.rawPhotometricSourceMaskPath = object.value(
        QStringLiteral("raw_photometric_source_mask_path")).toString();
    record.rawGeometrySupportPath = object.value(
        QStringLiteral("raw_geometry_support_path")).toString();
    record.rawAdaptiveGeometrySupportWeightPath = object.value(
        QStringLiteral("raw_adaptive_geometry_support_weight_path")).toString();
    record.rawAdaptiveGeometryEffectiveViewCountPath = object.value(
        QStringLiteral("raw_adaptive_geometry_effective_view_count_path")).toString();
    record.rawAdaptiveGeometryConflictRatioPath = object.value(
        QStringLiteral("raw_adaptive_geometry_conflict_ratio_path")).toString();
    record.rawGeometrySourceMaskPath = object.value(
        QStringLiteral("raw_geometry_source_mask_path")).toString();
    record.rawInverseDepthMeanPath = object.value(
        QStringLiteral("raw_inverse_depth_mean_path")).toString();
    record.rawInverseDepthSpreadPath = object.value(
        QStringLiteral("raw_inverse_depth_spread_path")).toString();
    record.crossViewRepairedMaskPath = object.value(
        QStringLiteral("cross_view_repaired_mask_path")).toString();
    record.targetedGapRecoveredMaskPath = object.value(
        QStringLiteral("targeted_gap_recovered_mask_path")).toString();
    record.residualReestimatedMaskPath = object.value(
        QStringLiteral("residual_reestimated_mask_path")).toString();
    record.depthProvenancePath = object.value(
        QStringLiteral("depth_provenance_path")).toString();
    record.validMaskPath = object.value(QStringLiteral("valid_mask_path")).toString();
    record.supportMaskPath = object.value(QStringLiteral("support_mask_path")).toString();
    record.missingReasonPath = object.value(
        QStringLiteral("missing_reason_path")).toString();
    record.missingReasonPreviewPath = object.value(
        QStringLiteral("missing_reason_preview_path")).toString();
    record.effectiveNativeFinalDepthGrid = object.value(
        QStringLiteral("effective_native_final_depth_grid")).toBool(false);
    record.pixelDomainDiagnostics = object.value(
        QStringLiteral("pixel_domain_diagnostics")).toObject();
    record.gridWidth = object.value(QStringLiteral("grid_width")).toInt(0);
    record.gridHeight = object.value(QStringLiteral("grid_height")).toInt(0);
    const QJsonValue elapsed = object.value(QStringLiteral("elapsed_ms"));
    record.elapsedMs = elapsed.isString() ? elapsed.toString().toLongLong()
                                          : static_cast<qint64>(elapsed.toDouble(0.0));
    record.error = object.value(QStringLiteral("error")).toString();
    record.configHash = object.value(QStringLiteral("config_hash")).toString();
    record.algorithmRevision = object.value(
        QStringLiteral("algorithm_revision")).toInt(0);
    const MvsDepthFrameQualification qualification =
        qualifyMvsDepthFrameArtifact(object);
    record.acceptance = qualification.acceptance;
    record.fusionEligibilityKnown = qualification.fusionEligibilityKnown;
    record.fusionEligible = qualification.fusionEligible;
    record.role = qualification.role;
    return record;
}

bool MvsWorkspaceManifest::load(const QString &path, QString *errorMsg)
{
    clear();

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMsg)
        {
            *errorMsg = file.errorString();
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    {
        if (errorMsg)
        {
            *errorMsg = parseError.errorString();
        }
        return false;
    }

    const QJsonObject root = doc.object();
    _configHash = root.value(QStringLiteral("config_hash")).toString();
    const QJsonArray frames = root.value(QStringLiteral("frames")).toArray();
    _frames.reserve(frames.size());
    for (const QJsonValue &value : frames)
    {
        if (value.isObject())
        {
            _frames.push_back(MvsDepthFrameRecord::fromJson(value.toObject()));
        }
    }
    return true;
}

bool MvsWorkspaceManifest::saveAtomic(const QString &path, QString *errorMsg) const
{
    const QFileInfo info(path);
    if (!info.absoluteDir().exists() && !QDir().mkpath(info.absolutePath()))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法创建目录: %1").arg(info.absolutePath());
        }
        return false;
    }

    const QByteArray payload = QJsonDocument(toJson()).toJson(QJsonDocument::Indented);
    QString last_error;
    constexpr int maximum_attempts = 5;
    for (int attempt = 0; attempt < maximum_attempts; ++attempt)
    {
        QSaveFile file(path);
        if (file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        {
            const qint64 written = file.write(payload);
            if (written == payload.size() && file.commit())
            {
                return true;
            }
            last_error = written == payload.size()
                ? file.errorString()
                : QStringLiteral("写入 manifest 数据不完整");
            if (file.isOpen())
            {
                file.cancelWriting();
            }
        }
        else
        {
            last_error = file.errorString();
        }

        if (attempt + 1 < maximum_attempts)
        {
            QThread::msleep(static_cast<unsigned long>(20 * (attempt + 1)));
        }
    }

    if (errorMsg)
    {
        *errorMsg = last_error;
    }
    return false;
}

void MvsWorkspaceManifest::clear()
{
    _configHash.clear();
    _frames.clear();
}

QString MvsWorkspaceManifest::configHash() const
{
    return _configHash;
}

void MvsWorkspaceManifest::setConfigHash(const QString &hash)
{
    _configHash = hash;
}

const QVector<MvsDepthFrameRecord> &MvsWorkspaceManifest::frames() const
{
    return _frames;
}

QVector<MvsDepthFrameRecord> MvsWorkspaceManifest::completedFramesSortedByName() const
{
    QVector<MvsDepthFrameRecord> result;
    for (const MvsDepthFrameRecord &record : _frames)
    {
        if (record.status == QStringLiteral("completed"))
        {
            result.push_back(record);
        }
    }

    std::stable_sort(result.begin(), result.end(),
                     [](const MvsDepthFrameRecord &lhs, const MvsDepthFrameRecord &rhs)
                     {
                         return xjw::common::io::naturalFileNameLessThan(
                             frameSortName(lhs), frameSortName(rhs));
                     });
    return result;
}

void MvsWorkspaceManifest::upsertFrame(const MvsDepthFrameRecord &record)
{
    MvsDepthFrameRecord normalized = record;
    normalized.role = qualifyMvsDepthFrameArtifact(
        normalized.toJson()).role;
    const int index = findFrameIndex(normalized.refIndex);
    if (index >= 0)
    {
        _frames[index] = normalized;
    }
    else
    {
        _frames.push_back(normalized);
    }
}

void MvsWorkspaceManifest::markRunning(int refIndex, const QString &refImage, const QString &configHash)
{
    MvsDepthFrameRecord record;
    const int index = findFrameIndex(refIndex);
    if (index >= 0)
    {
        record = _frames[index];
    }
    record.refIndex = refIndex;
    record.refImage = refImage;
    record.status = QStringLiteral("running");
    record.error.clear();
    record.configHash = configHash;
    record.algorithmRevision = kMvsDepthAlgorithmRevision;
    upsertFrame(record);
}

void MvsWorkspaceManifest::markCompleted(const MvsDepthFrameRecord& record)
{
    MvsDepthFrameRecord completed = record;
    const bool has_explicit_depth_completeness = !record.depthCompleteness.isEmpty();
    const bool has_explicit_acceptance = !record.acceptance.trimmed().isEmpty();
    const bool has_explicit_qualification_update = has_explicit_acceptance || record.fusionEligibilityKnown;
    // A completed artifact belongs to the running implementation even when a
    // replay caller seeded the record from an older manifest. Keeping the
    // caller's stale value makes the root and per-frame revisions disagree.
    completed.algorithmRevision = kMvsDepthAlgorithmRevision;
    const int index = findFrameIndex(completed.refIndex);
    if (completed.sourcePlan.isEmpty() && index >= 0)
    {
        completed.sourcePlan = _frames[index].sourcePlan;
    }
    if (completed.sourceViewCount <= 0 && index >= 0)
    {
        completed.sourceViewCount = _frames[index].sourceViewCount;
        completed.requestedSourceViewCount = _frames[index].requestedSourceViewCount;
        completed.sourceViewShortfall = _frames[index].sourceViewShortfall;
        completed.sourceViewShortfallReason = _frames[index].sourceViewShortfallReason;
        completed.meanSourceQualityScore = _frames[index].meanSourceQualityScore;
        completed.minSourceQualityScore = _frames[index].minSourceQualityScore;
    }
    if (!completed.geometricGuidancePassExpected && !completed.geometricGuidancePassApplied && index >= 0)
    {
        completed.geometricGuidancePassExpected = _frames[index].geometricGuidancePassExpected;
        completed.geometricGuidancePassApplied = _frames[index].geometricGuidancePassApplied;
    }
    if (!completed.consistencyPublicationExpected && !has_explicit_depth_completeness && index >= 0)
    {
        completed.consistencyPublicationExpected = _frames[index].consistencyPublicationExpected;
    }
    if (completed.validPixelCount <= 0 && index >= 0)
    {
        completed.validPixelCount = _frames[index].validPixelCount;
        completed.meanDepthConfidence = _frames[index].meanDepthConfidence;
    }
    if (completed.depthQuality.isEmpty() && index >= 0)
    {
        completed.depthQuality = _frames[index].depthQuality;
    }
    if (!has_explicit_depth_completeness && index >= 0)
    {
        completed.depthCompleteness = _frames[index].depthCompleteness;
    }
    if (completed.geometryEvidenceDiagnostics.isEmpty() && index >= 0)
    {
        completed.geometryEvidenceDiagnostics = _frames[index].geometryEvidenceDiagnostics;
    }
    if (completed.crossViewRepairDiagnostics.isEmpty() && index >= 0)
    {
        completed.crossViewRepairDiagnostics = _frames[index].crossViewRepairDiagnostics;
    }
    if (completed.depthProvenanceSummary.isEmpty() && index >= 0)
    {
        completed.depthProvenanceSummary =
            _frames[index].depthProvenanceSummary;
    }
    if (completed.qualityDecision.isEmpty() &&
        !has_explicit_qualification_update && index >= 0)
    {
        completed.qualityDecision = _frames[index].qualityDecision;
    }
    if (completed.pyramidLevels.isEmpty() && index >= 0)
    {
        completed.pyramidLevels = _frames[index].pyramidLevels;
    }
    if (completed.maskSource.isEmpty() && index >= 0)
    {
        completed.maskSource = _frames[index].maskSource;
    }
    if (completed.maskCoverage < 0.0 && index >= 0)
    {
        completed.maskCoverage = _frames[index].maskCoverage;
    }
    if (completed.selectedLevel <= 0 && index >= 0)
    {
        completed.selectedLevel = _frames[index].selectedLevel;
    }
    if (completed.fallbackReason.isEmpty() && index >= 0)
    {
        completed.fallbackReason = _frames[index].fallbackReason;
    }
    if (completed.sceneProfile.isEmpty() && index >= 0)
    {
        completed.sceneProfile = _frames[index].sceneProfile;
    }
    if (completed.filterMode.isEmpty() && index >= 0)
    {
        completed.filterMode = _frames[index].filterMode;
    }
    // acceptance and fusion_eligible are one qualification decision. Inherit
    // them only when the update supplies neither half; mixing a new half with
    // a stale half can silently recreate a Primary role.
    if (!has_explicit_acceptance && !completed.fusionEligibilityKnown &&
        index >= 0)
    {
        completed.acceptance = _frames[index].acceptance;
        completed.fusionEligibilityKnown =
            _frames[index].fusionEligibilityKnown;
        completed.fusionEligible = _frames[index].fusionEligible;
    }
    if (completed.depthPostprocess.isEmpty() && index >= 0)
    {
        completed.depthPostprocess = _frames[index].depthPostprocess;
    }
    if (completed.cameraModel.isEmpty() && index >= 0)
    {
        completed.cameraModel = _frames[index].cameraModel;
    }
    if (completed.preparedImage.isEmpty() && index >= 0)
    {
        completed.preparedImage = _frames[index].preparedImage;
    }
    if (completed.preparedValidMaskPath.isEmpty() && index >= 0)
    {
        completed.preparedValidMaskPath =
            _frames[index].preparedValidMaskPath;
    }
    if (completed.preparedCameraModel.isEmpty() && index >= 0)
    {
        completed.preparedCameraModel =
            _frames[index].preparedCameraModel;
    }
    if (completed.poseRefinementDiagnostics.isEmpty() && index >= 0)
    {
        completed.poseRefinementDiagnostics =
            _frames[index].poseRefinementDiagnostics;
    }
    if (completed.derivedCameraModel.isEmpty() && index >= 0)
    {
        completed.derivedCameraModel = _frames[index].derivedCameraModel;
    }
    completed.status = QStringLiteral("completed");
    completed.error.clear();
    upsertFrame(completed);
}

void MvsWorkspaceManifest::markFailed(int refIndex, const QString &error)
{
    MvsDepthFrameRecord record;
    const int index = findFrameIndex(refIndex);
    if (index >= 0)
    {
        record = _frames[index];
    }
    record.refIndex = refIndex;
    record.status = QStringLiteral("failed");
    record.error = error;
    upsertFrame(record);
}

void MvsWorkspaceManifest::updatePoseRefinement(
    int refIndex,
    const QJsonObject &diagnostics,
    const QJsonObject &derivedCameraModel)
{
    const int index = findFrameIndex(refIndex);
    if (index < 0)
    {
        return;
    }
    _frames[index].poseRefinementDiagnostics = diagnostics;
    _frames[index].derivedCameraModel = derivedCameraModel;
}

bool MvsWorkspaceManifest::hasReusableCompletedFrame(int refIndex, const QString& configHash) const
{
    const int index = findFrameIndex(refIndex);
    if (index < 0)
    {
        return false;
    }
    const MvsDepthFrameRecord& record = _frames[index];
    if (record.status != QStringLiteral("completed") || record.configHash != configHash ||
        record.algorithmRevision != kMvsDepthAlgorithmRevision)
    {
        return false;
    }
    if (record.geometricGuidancePassExpected != record.geometricGuidancePassApplied)
    {
        return false;
    }
    FastDepthMatShape raw_depth_shape;
    FastDepthMatShape raw_confidence_shape;
    PngShape depth_preview_shape;
    PngShape valid_mask_shape;
    PngShape support_mask_shape;
    if (!readFastDepthMatShape(record.rawDepthPath, &raw_depth_shape) ||
        !readFastDepthMatShape(record.rawConfidencePath, &raw_confidence_shape) ||
        !readPngShape(record.depthPng, &depth_preview_shape) ||
        !readPngShape(record.validMaskPath, &valid_mask_shape) ||
        !readPngShape(record.supportMaskPath, &support_mask_shape) || raw_depth_shape.type != CV_32FC1 ||
        raw_confidence_shape.type != CV_32FC1 || record.gridWidth <= 0 || record.gridHeight <= 0 ||
        raw_depth_shape.cols != record.gridWidth || raw_depth_shape.rows != record.gridHeight ||
        raw_depth_shape.rows != raw_confidence_shape.rows || raw_depth_shape.cols != raw_confidence_shape.cols ||
        raw_depth_shape.rows != valid_mask_shape.rows || raw_depth_shape.cols != valid_mask_shape.cols ||
        raw_depth_shape.rows != support_mask_shape.rows || raw_depth_shape.cols != support_mask_shape.cols)
    {
        return false;
    }

    const auto matches_fast_mat = [&raw_depth_shape](const QString& path, int expected_type)
    {
        FastDepthMatShape shape;
        return readFastDepthMatShape(path, &shape) && shape.type == expected_type &&
               shape.rows == raw_depth_shape.rows && shape.cols == raw_depth_shape.cols;
    };
    const auto matches_png = [&raw_depth_shape](const QString& path)
    {
        PngShape shape;
        return readPngShape(path, &shape) && shape.rows == raw_depth_shape.rows && shape.cols == raw_depth_shape.cols;
    };

    if (record.algorithmRevision >= kMvsPreparedRasterProvenanceRevision)
    {
        PngShape prepared_image_shape;
        PngShape prepared_valid_mask_shape;
        if (!readPngShape(record.preparedImage, &prepared_image_shape) ||
            !readPngShape(record.preparedValidMaskPath, &prepared_valid_mask_shape) ||
            prepared_image_shape.rows != prepared_valid_mask_shape.rows ||
            prepared_image_shape.cols != prepared_valid_mask_shape.cols || record.preparedCameraModel.isEmpty())
        {
            return false;
        }
    }

    if (record.algorithmRevision >= kMvsDepthProvenanceRevision && !matches_png(record.depthProvenancePath))
    {
        return false;
    }

    if (record.algorithmRevision >= kMvsJointViewAndGeometricGuidanceRevision &&
        !matches_fast_mat(record.rawPhotometricSourceMaskPath, CV_32SC1))
    {
        return false;
    }

    const ConsistencyPublicationState consistency_state = consistencyPublicationState(record.depthCompleteness);
    if (consistency_state == ConsistencyPublicationState::Invalid)
    {
        return false;
    }
    const bool has_completed_consistency = consistency_state == ConsistencyPublicationState::Completed;
    if (record.consistencyPublicationExpected != has_completed_consistency)
    {
        return false;
    }

    if (record.algorithmRevision >= kMvsGeometryFusionSupportRevision && has_completed_consistency &&
        (!matches_fast_mat(record.rawGeometrySupportPath, CV_16UC1) ||
         !matches_fast_mat(record.rawInverseDepthSpreadPath, CV_32FC1)))
    {
        return false;
    }

    if (record.algorithmRevision >= kMvsGeometrySourceOrdinalRevision)
    {
        const bool has_source_mask_path = !record.rawGeometrySourceMaskPath.trimmed().isEmpty();
        const bool has_source_ordinals = !record.geometrySourceIndices.isEmpty();
        if (has_source_mask_path != has_source_ordinals ||
            (has_source_mask_path && !matches_fast_mat(record.rawGeometrySourceMaskPath, CV_16UC1)) ||
            record.geometrySourceIndices.size() > 16)
        {
            return false;
        }
        for (qsizetype ordinal = 0; ordinal < record.geometrySourceIndices.size(); ++ordinal)
        {
            const int source_index = record.geometrySourceIndices[ordinal];
            if (source_index < 0 || source_index == record.refIndex ||
                std::find(record.geometrySourceIndices.cbegin(),
                          record.geometrySourceIndices.cbegin() + ordinal,
                          source_index) != record.geometrySourceIndices.cbegin() + ordinal)
            {
                return false;
            }
        }
    }

    if (!isKnownDepthSceneProfile(record.sceneProfile))
    {
        return false;
    }
    const bool requires_adaptive_geometry_evidence =
        record.algorithmRevision >= kMvsAdaptiveGeometryEvidenceRevision &&
        has_completed_consistency &&
        isOrbitalDepthSceneProfile(record.sceneProfile);
    if (!requires_adaptive_geometry_evidence)
    {
        return true;
    }

    return matches_fast_mat(record.rawInverseDepthMeanPath, CV_32FC1) &&
           matches_png(record.crossViewRepairedMaskPath) &&
           matches_fast_mat(record.rawAdaptiveGeometrySupportWeightPath, CV_32FC1) &&
           matches_fast_mat(record.rawAdaptiveGeometryEffectiveViewCountPath, CV_32FC1) &&
           matches_fast_mat(record.rawAdaptiveGeometryConflictRatioPath, CV_32FC1);
}

QJsonObject MvsWorkspaceManifest::toJson() const
{
    QJsonObject root;
    root.insert(QStringLiteral("schema"), QStringLiteral("plascan.mvs.workspace.v2"));
    root.insert(QStringLiteral("algorithm_revision"), kMvsDepthAlgorithmRevision);
    root.insert(QStringLiteral("config_hash"), _configHash);

    QJsonArray frames;
    for (const MvsDepthFrameRecord& record : _frames)
    {
        frames.push_back(record.toJson());
    }
    root.insert(QStringLiteral("frames"), frames);
    return root;
}

int MvsWorkspaceManifest::findFrameIndex(int refIndex) const
{
    for (int i = 0; i < _frames.size(); ++i)
    {
        if (_frames[i].refIndex == refIndex)
        {
            return i;
        }
    }
    return -1;
}

QString makeMvsDepthConfigHash(const DepthGenConfig &config, int viewCount)
{
    QJsonObject patch;
    patch.insert(QStringLiteral("num_iterations"), config.patchMatch.numIterations);
    patch.insert(QStringLiteral("patch_half"), config.patchMatch.patchHalf);
    patch.insert(QStringLiteral("num_source_views"), config.patchMatch.numSourceViews);
    patch.insert(QStringLiteral("confidence_thresh"), config.patchMatch.confidenceThresh);
    patch.insert(QStringLiteral("minimum_masked_patch_support_ratio"),
                 config.patchMatch.minimumMaskedPatchSupportRatio);
    patch.insert(QStringLiteral("photometric_uniqueness"),
                 config.patchMatch.enablePhotometricUniqueness);
    patch.insert(QStringLiteral("photometric_uniqueness_relative_depth_step"),
                 config.patchMatch.photometricUniquenessRelativeDepthStep);
    patch.insert(QStringLiteral("photometric_uniqueness_minimum_margin"),
                 config.patchMatch.photometricUniquenessMinimumMargin);
    patch.insert(QStringLiteral("photometric_uniqueness_minimum_confidence_scale"),
                 config.patchMatch.photometricUniquenessMinimumConfidenceScale);
    patch.insert(QStringLiteral("per_pixel_source_selection"),
                 config.patchMatch.enablePerPixelSourceSelection);
    patch.insert(QStringLiteral("source_selection_neighbor_bonus"),
                 config.patchMatch.sourceSelectionNeighborBonus);
    patch.insert(QStringLiteral("asymmetric_propagation"),
                 config.patchMatch.enableAsymmetricPropagation);
    patch.insert(QStringLiteral("geometric_guidance_pass"),
                 config.patchMatch.enableGeometricGuidancePass);
    patch.insert(QStringLiteral("geometric_guidance_iterations"),
                 config.patchMatch.geometricGuidanceIterations);
    patch.insert(QStringLiteral("geometric_guidance_weight"),
                 config.patchMatch.geometricGuidanceWeight);
    patch.insert(QStringLiteral("geometric_guidance_max_error_pixels"),
                 config.patchMatch.geometricGuidanceMaxErrorPixels);
    patch.insert(QStringLiteral("geometric_guidance_relative_depth_radius"),
                 config.patchMatch.geometricGuidanceRelativeDepthRadius);
    patch.insert(QStringLiteral("backend"), static_cast<int>(config.patchMatch.backend));
    patch.insert(QStringLiteral("downsample_factor"), config.patchMatch.downsampleFactor);
    patch.insert(QStringLiteral("median_blur"), config.patchMatch.doMedianBlur);
    patch.insert(QStringLiteral("median_kernel"), config.patchMatch.medianKernelSize);
    patch.insert(QStringLiteral("bilateral"), config.patchMatch.doBilateralFilter);
    patch.insert(QStringLiteral("bilateral_d"), config.patchMatch.bilateralD);
    patch.insert(QStringLiteral("bilateral_sigma_color"), config.patchMatch.bilateralSigmaColor);
    patch.insert(QStringLiteral("bilateral_sigma_space"), config.patchMatch.bilateralSigmaSpace);
    patch.insert(QStringLiteral("geom_consistency"), config.patchMatch.geomConsistency);
    patch.insert(QStringLiteral("geom_consistency_max_err"), config.patchMatch.geomConsistencyMaxErr);
    patch.insert(QStringLiteral("epipolar_rectified"), config.patchMatch.epipolarRectified);
    patch.insert(QStringLiteral("cuda_parallel_sweep"), config.patchMatch.cudaUseParallelSweep);
    patch.insert(QStringLiteral("cuda_fallback_to_cpu"), config.patchMatch.cudaFallbackToCpu);
    patch.insert(QStringLiteral("cuda_device_index"), config.patchMatch.cudaDeviceIndex);
    patch.insert(QStringLiteral("opencl_fallback_to_cpu"), config.patchMatch.openClFallbackToCpu);
    patch.insert(QStringLiteral("opencl_device_index"), config.patchMatch.openClDeviceIndex);
    // cpuThreadCount and the CUDA launch dimensions only control scheduling.
    // cancelFlag is transient process state. None belongs to the persisted
    // result contract, so changing them must not invalidate a valid cache.

    QJsonObject fusion;
    fusion.insert(QStringLiteral("min_consistent_views"), config.fusion.minConsistentViews);
    fusion.insert(QStringLiteral("rel_depth_thresh"), config.fusion.relDepthThresh);
    fusion.insert(QStringLiteral("pixel_thresh"), config.fusion.pixelThresh);
    fusion.insert(QStringLiteral("confidence_thresh"), config.fusion.confidenceThresh);
    fusion.insert(QStringLiteral("adaptive_confidence"), config.fusion.enableAdaptiveConfidenceFilter);
    fusion.insert(QStringLiteral("adaptive_full_coverage"), config.fusion.adaptiveFullCoverageThreshold);
    fusion.insert(QStringLiteral("adaptive_low_mean_confidence"),
                  config.fusion.adaptiveLowMeanConfidenceThreshold);
    fusion.insert(QStringLiteral("adaptive_strict_confidence"),
                  config.fusion.adaptiveStrictConfidenceThreshold);
    fusion.insert(QStringLiteral("geometry_supported_low_confidence_retention"),
                  config.fusion.enableGeometrySupportedLowConfidenceRetention);
    fusion.insert(QStringLiteral("geometry_supported_minimum_confidence"),
                  config.fusion.geometrySupportedMinimumConfidence);
    fusion.insert(QStringLiteral("geometry_supported_minimum_observation_count"),
                  config.fusion.geometrySupportedMinimumObservationCount);
    fusion.insert(QStringLiteral("geometry_supported_maximum_inverse_depth_spread"),
                  config.fusion.geometrySupportedMaximumInverseDepthSpread);
    fusion.insert(QStringLiteral("geometry_supported_minimum_adaptive_support_weight"),
                  config.fusion.geometrySupportedMinimumAdaptiveSupportWeight);
    fusion.insert(QStringLiteral("geometry_supported_minimum_adaptive_effective_views"),
                  config.fusion.geometrySupportedMinimumAdaptiveEffectiveViews);
    fusion.insert(QStringLiteral("geometry_supported_maximum_adaptive_conflict_ratio"),
                  config.fusion.geometrySupportedMaximumAdaptiveConflictRatio);
    fusion.insert(QStringLiteral("boundary_aware_retention"),
                  config.fusion.enableBoundaryAwareRetention);
    fusion.insert(QStringLiteral("boundary_protection_radius_pixels"),
                  config.fusion.boundaryProtectionRadiusPixels);
    fusion.insert(QStringLiteral("boundary_minimum_confidence"),
                  config.fusion.boundaryMinimumConfidence);
    fusion.insert(QStringLiteral("boundary_minimum_observation_count"),
                  config.fusion.boundaryMinimumObservationCount);
    fusion.insert(QStringLiteral("boundary_maximum_inverse_depth_spread"),
                  config.fusion.boundaryMaximumInverseDepthSpread);
    fusion.insert(QStringLiteral("sigma_fusion"), config.fusion.doSigmaFusion);
    fusion.insert(QStringLiteral("sigma_multiplier"), config.fusion.sigmaMultiplier);
    fusion.insert(QStringLiteral("inpaint"), config.fusion.doInpaint);
    fusion.insert(QStringLiteral("inpaint_radius_factor"), config.fusion.inpaintRadiusFactor);
    fusion.insert(QStringLiteral("inpaint_radius"), config.fusion.inpaintRadius);
    fusion.insert(QStringLiteral("local_outlier"), config.fusion.enableLocalDepthOutlierFilter);
    fusion.insert(QStringLiteral("local_outlier_kernel"), config.fusion.localDepthOutlierKernelSize);
    fusion.insert(QStringLiteral("local_outlier_rel_thresh"), config.fusion.localDepthOutlierRelThresh);
    fusion.insert(QStringLiteral("local_outlier_max_removal_ratio"),
                  config.fusion.maxLocalDepthOutlierRemovalRatio);
    fusion.insert(QStringLiteral("speckle_filter"), config.fusion.enableSpeckleFilter);
    fusion.insert(QStringLiteral("speckle_min_area"), config.fusion.minSpeckleComponentArea);
    fusion.insert(QStringLiteral("speckle_max_removal_ratio"), config.fusion.maxSpeckleRemovalRatio);

    QJsonObject root;
    root.insert(QStringLiteral("schema"), QStringLiteral("plascan.mvs.depth.config.v5"));
    root.insert(QStringLiteral("algorithm_revision"), kMvsDepthAlgorithmRevision);
    root.insert(QStringLiteral("view_count"), viewCount);
    root.insert(QStringLiteral("input_signature"),
                QString::fromStdString(config.inputSignature));
    root.insert(QStringLiteral("quality_profile"),
                QString::fromStdString(config.qualityProfile));
    root.insert(QStringLiteral("configured_source_view_count"),
                config.configuredSourceViewCount > 0
                    ? config.configuredSourceViewCount
                    : config.numSourceViews);
    root.insert(QStringLiteral("num_source_views"), config.numSourceViews);
    if (std::isfinite(config.sourceMaximumAngleDegCap) &&
        config.sourceMaximumAngleDegCap > 0.0f)
    {
        root.insert(QStringLiteral("source_maximum_angle_deg_cap"),
                    config.sourceMaximumAngleDegCap);
    }
    if (config.evaluateCompleteVisibilityCandidatePool)
    {
        root.insert(
            QStringLiteral("evaluate_complete_visibility_candidate_pool"),
            true);
    }
    if (std::isfinite(config.sourceAngleSoftRankingStrength) &&
        config.sourceAngleSoftRankingStrength > 0.0f)
    {
        root.insert(QStringLiteral("source_angle_soft_ranking_strength"),
                    config.sourceAngleSoftRankingStrength);
    }
    root.insert(QStringLiteral("z_near_scale"), config.zNearScale);
    root.insert(QStringLiteral("z_far_scale"), config.zFarScale);
    root.insert(QStringLiteral("scene_profile"), static_cast<int>(config.sceneProfile));
    root.insert(QStringLiteral("depth_filter_mode"), static_cast<int>(config.depthFilterMode));
    root.insert(QStringLiteral("adaptive_depth_filter_mode"), config.adaptiveDepthFilterMode);
    root.insert(QStringLiteral("preserve_native_final_depth_grid"),
                config.preserveNativeFinalDepthGrid);
    root.insert(QStringLiteral("resolved_image_cache_strategy"),
                QString::fromStdString(config.resolvedImageCacheStrategy));
    root.insert(QStringLiteral("resolved_image_cache_capacity"),
                config.resolvedImageCacheCapacity);
    root.insert(QStringLiteral("enable_adaptive_geometry_evidence"),
                config.enableAdaptiveGeometryEvidence);
    const DepthPoseRefinementOptions &pose_refinement =
        config.depthPoseRefinement;
    QJsonObject pose_refinement_json;
    pose_refinement_json.insert(QStringLiteral("enabled"),
                                pose_refinement.enabled);
    pose_refinement_json.insert(
        QStringLiteral("emit_derived_camera_candidates"),
        pose_refinement.emitDerivedCameraCandidates);
    pose_refinement_json.insert(QStringLiteral("sampling_stride_pixels"),
                                pose_refinement.samplingStridePixels);
    pose_refinement_json.insert(QStringLiteral("maximum_samples_per_camera"),
                                pose_refinement.maximumSamplesPerCamera);
    pose_refinement_json.insert(
        QStringLiteral("maximum_source_frames_per_camera"),
        pose_refinement.maximumSourceFramesPerCamera);
    pose_refinement_json.insert(
        QStringLiteral("minimum_adaptive_support_weight"),
        pose_refinement.minimumAdaptiveSupportWeight);
    pose_refinement_json.insert(
        QStringLiteral("minimum_adaptive_effective_view_count"),
        pose_refinement.minimumAdaptiveEffectiveViewCount);
    pose_refinement_json.insert(
        QStringLiteral("maximum_adaptive_conflict_ratio"),
        pose_refinement.maximumAdaptiveConflictRatio);
    pose_refinement_json.insert(
        QStringLiteral("maximum_correspondence_relative_depth_error"),
        pose_refinement.maximumCorrespondenceRelativeDepthError);
    pose_refinement_json.insert(
        QStringLiteral("occlusion_relative_depth_tolerance"),
        pose_refinement.occlusionRelativeDepthTolerance);
    pose_refinement_json.insert(
        QStringLiteral("minimum_evidence_sample_coverage"),
        pose_refinement.minimumEvidenceSampleCoverage);
    pose_refinement_json.insert(
        QStringLiteral("minimum_projection_retention_ratio"),
        pose_refinement.minimumProjectionRetentionRatio);
    pose_refinement_json.insert(
        QStringLiteral("anchor_camera_index"),
        pose_refinement.optimizer.anchorCameraIndex);
    pose_refinement_json.insert(
        QStringLiteral("maximum_iterations"),
        pose_refinement.optimizer.maximumIterations);
    pose_refinement_json.insert(
        QStringLiteral("minimum_correspondences"),
        pose_refinement.optimizer.minimumCorrespondences);
    pose_refinement_json.insert(QStringLiteral("huber_delta"),
                                pose_refinement.optimizer.huberDelta);
    pose_refinement_json.insert(QStringLiteral("damping"), pose_refinement.optimizer.damping);
    pose_refinement_json.insert(QStringLiteral("convergence_translation"),
                                pose_refinement.optimizer.convergenceTranslation);
    pose_refinement_json.insert(QStringLiteral("convergence_rotation_radians"),
                                pose_refinement.optimizer.convergenceRotationRadians);
    pose_refinement_json.insert(QStringLiteral("maximum_translation"),
                                pose_refinement.optimizer.maximumTranslation);
    pose_refinement_json.insert(
        QStringLiteral("maximum_rotation_degrees"),
        pose_refinement.optimizer.maximumRotationDegrees);
    pose_refinement_json.insert(
        QStringLiteral("required_p90_improvement_ratio"),
        pose_refinement.optimizer.requiredP90ImprovementRatio);
    if (pose_refinement.enabled)
    {
        root.insert(QStringLiteral("depth_pose_refinement"),
                    pose_refinement_json);
    }
    root.insert(QStringLiteral("cross_view_hole_repair_source_count"),
                config.crossViewHoleRepairSourceCount);
    root.insert(QStringLiteral("targeted_gap_recovery"),
                config.enableTargetedGapRecovery);
    root.insert(QStringLiteral("targeted_gap_recovery_source_count"),
                config.targetedGapRecoverySourceCount);
    root.insert(QStringLiteral("targeted_gap_recovery_hypothesis_count"),
                config.targetedGapRecoveryHypothesisCount);
    root.insert(QStringLiteral("targeted_gap_recovery_confidence"),
                config.targetedGapRecoveryConfidence);
    root.insert(
        QStringLiteral("targeted_gap_recovery_prior_relative_difference"),
        config.targetedGapRecoveryPriorRelativeDifference);
    root.insert(
        QStringLiteral("targeted_gap_recovery_consensus_inverse_depth_spread"),
        config.targetedGapRecoveryConsensusInverseDepthSpread);
    root.insert(
        QStringLiteral("targeted_gap_recovery_consensus_prior_relative_difference"),
        config.targetedGapRecoveryConsensusPriorRelativeDifference);
    root.insert(QStringLiteral("targeted_gap_surface_prior"),
                config.enableTargetedGapSurfacePrior);
    root.insert(QStringLiteral("targeted_gap_surface_prior_maximum_anchor_spread"),
                config.targetedGapSurfacePriorMaximumAnchorSpread);
    root.insert(QStringLiteral("targeted_gap_surface_prior_maximum_fit_residual"),
                config.targetedGapSurfacePriorMaximumFitResidual);
    root.insert(
        QStringLiteral("targeted_gap_recovery_maximum_prior_distance_pixels"),
        config.targetedGapRecoveryMaximumPriorDistancePixels);
    root.insert(QStringLiteral("post_consistency_residual_reestimation"),
                config.enablePostConsistencyResidualReestimation);
    root.insert(QStringLiteral("post_consistency_residual_source_count"),
                config.postConsistencyResidualSourceCount);
    root.insert(QStringLiteral("post_consistency_residual_confidence"),
                config.postConsistencyResidualConfidence);
    root.insert(QStringLiteral(
                    "post_consistency_residual_maximum_layer_spread"),
                config.postConsistencyResidualMaximumLayerSpread);
    root.insert(QStringLiteral(
                    "post_consistency_residual_maximum_prior_radius"),
                config.postConsistencyResidualMaximumPriorRadius);
    root.insert(QStringLiteral("two_source_cross_view_growth"),
                config.enableTwoSourceCrossViewGrowth);
    if (config.enableDepthLayerReliabilityAnchorGate)
    {
        root.insert(QStringLiteral("depth_layer_reliability_anchor_gate"),
                    true);
    }
    if (config.enableDepthLayerReliabilityGuidedCorrection)
    {
        root.insert(
            QStringLiteral("depth_layer_reliability_guided_correction"),
            true);
    }
    root.insert(QStringLiteral("two_source_growth_distance_pixels"),
                config.twoSourceGrowthDistancePixels);
    root.insert(QStringLiteral("two_source_growth_inverse_depth_spread"),
                config.twoSourceGrowthInverseDepthSpread);
    root.insert(QStringLiteral("two_source_growth_normal_angle_degrees"),
                config.twoSourceGrowthNormalAngleDegrees);
    root.insert(QStringLiteral("two_source_growth_maximum_component_area"),
                config.twoSourceGrowthMaximumComponentArea);
    root.insert(QStringLiteral("learned_mvs_candidates"),
                config.enableLearnedMvsCandidates);
    root.insert(QStringLiteral("learned_mvs_candidate_directory"),
                QString::fromStdString(config.learnedMvsCandidateDirectory));
    root.insert(QStringLiteral("learned_mvs_minimum_confidence"),
                config.learnedMvsMinimumConfidence);
    root.insert(QStringLiteral("learned_mvs_minimum_geometry_observations"),
                config.learnedMvsMinimumGeometryObservations);
    root.insert(QStringLiteral("learned_mvs_maximum_inverse_depth_spread"),
                config.learnedMvsMaximumInverseDepthSpread);
    root.insert(QStringLiteral("learned_mvs_maximum_relative_depth_difference"),
                config.learnedMvsMaximumRelativeDepthDifference);
    root.insert(QStringLiteral("learned_mvs_replacement_confidence_margin"),
                config.learnedMvsReplacementConfidenceMargin);
    root.insert(QStringLiteral("save_intermediate_pyramid_levels"),
                config.saveIntermediatePyramidLevels);
    root.insert(QStringLiteral("patch_match"), patch);
    root.insert(QStringLiteral("fusion"), fusion);

    const QByteArray canonical = QJsonDocument(root).toJson(QJsonDocument::Compact);
    return QString::fromLatin1(QCryptographicHash::hash(canonical, QCryptographicHash::Sha256).toHex());
}

QString
makeMvsDepthInputHash(const DepthGenConfig& config, const std::vector<CameraView>& views, const SparseCloud& sparse)
{
    QCryptographicHash hash(QCryptographicHash::Sha256);
    addFramedHashData(&hash, QByteArrayLiteral("plascan.mvs.depth.input.v2"));
    addFramedHashData(&hash, makeMvsDepthConfigHash(config, static_cast<int>(views.size())).toUtf8());

    for (const CameraView& view : views)
    {
        addFileFingerprint(&hash, "original_image", view.imagePath);
        addFileFingerprint(&hash, "prepared_image", view.preparedImagePath);
        addFileFingerprint(&hash, "prepared_valid_mask", view.preparedValidMaskPath);
        addFileFingerprint(&hash, "project_valid_region_mask", view.validRegionMaskPath);
        addFramedHashData(&hash, QByteArray::fromStdString(view.preparedValidMaskSource));
        addHashValue(&hash, view.imageWidth);
        addHashValue(&hash, view.imageHeight);

        const xjw::FramePinholeCamera::Intrinsics intrinsics = view.camera.intrinsics();
        addHashValue(&hash, intrinsics.focalX);
        addHashValue(&hash, intrinsics.focalY);
        addHashValue(&hash, intrinsics.principalX);
        addHashValue(&hash, intrinsics.principalY);
        addHashValue(&hash, intrinsics.pixelPitch);
        addHashValue(&hash, intrinsics.uAxisSign);
        addHashValue(&hash, intrinsics.vAxisSign);

        const xjw::FramePinholeCamera::Distortion distortion = view.camera.distortion();
        addHashValue(&hash, distortion.radialK1);
        addHashValue(&hash, distortion.radialK2);
        addHashValue(&hash, distortion.radialK3);
        addHashValue(&hash, distortion.tangentialP1);
        addHashValue(&hash, distortion.tangentialP2);

        const auto rotation = view.camera.cameraToWorldRotation();
        const auto center = view.camera.cameraCenter();
        for (const double value : rotation)
        {
            addHashValue(&hash, value);
        }
        for (const double value : center)
        {
            addHashValue(&hash, value);
        }
        const bool depth_axis_flipped = view.camera.depthAxisFlipped();
        addHashValue(&hash, depth_axis_flipped);
    }

    if (config.enableLearnedMvsCandidates)
    {
        addFramedHashData(&hash, QByteArrayLiteral("learned_mvs_candidate_inputs"));
        const bool has_candidate_directory = !config.learnedMvsCandidateDirectory.empty();
        const QDir candidate_directory(xjw::common::io::fromUtf8Path(config.learnedMvsCandidateDirectory));
        for (std::size_t frame_index = 0; frame_index < views.size(); ++frame_index)
        {
            addHashValue(&hash, frame_index);
            const QString depth_path =
                has_candidate_directory
                    ? candidate_directory.filePath(
                          QStringLiteral("learned_depth_%1.bin").arg(static_cast<qulonglong>(frame_index)))
                    : QString();
            const QString confidence_path =
                has_candidate_directory
                    ? candidate_directory.filePath(
                          QStringLiteral("learned_depth_%1_conf.bin").arg(static_cast<qulonglong>(frame_index)))
                    : QString();
            addFileFingerprint(&hash, "learned_candidate_depth", xjw::common::io::toUtf8Path(depth_path));
            addFileFingerprint(&hash, "learned_candidate_confidence", xjw::common::io::toUtf8Path(confidence_path));
        }
    }

    addHashValue(&hash, config.requireVerifiedSourcePairs);
    addHashValue(&hash, config.minSourcePairGeometricInliers);
    const std::size_t pair_quality_count = config.sourcePairQualities.size();
    addHashValue(&hash, pair_quality_count);
    for (const MvsSourcePairQuality& quality : config.sourcePairQualities)
    {
        addFramedHashData(&hash, QByteArray::fromStdString(quality.imageA));
        addFramedHashData(&hash, QByteArray::fromStdString(quality.imageB));
        addHashValue(&hash, quality.totalMatches);
        addHashValue(&hash, quality.geometricInliers);
        addHashValue(&hash, quality.verified);
        addHashValue(&hash, quality.hasVerificationStatistics);
        addHashValue(&hash, quality.geometricCoverage);
        addFramedHashData(&hash, QByteArray::fromStdString(quality.verificationReason));
    }

    const std::size_t sparse_point_count = sparse.points.size();
    addHashValue(&hash, sparse_point_count);
    for (const float value : sparse.minPt)
    {
        addHashValue(&hash, value);
    }
    for (const float value : sparse.maxPt)
    {
        addHashValue(&hash, value);
    }
    for (const auto& point : sparse.points)
    {
        for (const float coordinate : point)
        {
            addHashValue(&hash, coordinate);
        }
    }
    return QString::fromLatin1(hash.result().toHex());
}

} // namespace xjw::mvs
