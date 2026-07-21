#include "ProjectModelWorkflowPolicy.h"

#include "DepthFrameUtils.h"
#include "ProjectWorkflowUtils.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QThread>
#include <QtGlobal>

namespace xjw::gui::project
{

int recommendedInteractiveModelWorkerCount(int ideal_thread_count)
{
    const int available_threads = ideal_thread_count > 0 ? ideal_thread_count : 4;
    return qBound(1, available_threads - 2, 8);
}

namespace
{

QString modelQualityToDenseProfile(const QString &quality)
{
    const QString normalized = quality.trimmed().toLower();
    if (normalized == QStringLiteral("ultra") || normalized == QStringLiteral("high"))
    {
        return QStringLiteral("high_quality");
    }
    if (normalized == QStringLiteral("low"))
    {
        return QStringLiteral("fast_preview");
    }
    return QStringLiteral("standard");
}

double modelQualityToDenseResolutionScale(const QString &quality)
{
    const QString normalized = quality.trimmed().toLower();
    if (normalized == QStringLiteral("ultra"))
    {
        return 1.0;
    }
    if (normalized == QStringLiteral("low"))
    {
        return 0.25;
    }
    return 0.5;
}

int modelQualityToDenseIterations(const QString &quality)
{
    const QString normalized = quality.trimmed().toLower();
    if (normalized == QStringLiteral("ultra"))
    {
        return 10;
    }
    if (normalized == QStringLiteral("high"))
    {
        return 8;
    }
    if (normalized == QStringLiteral("low"))
    {
        return 4;
    }
    return 6;
}

QString depthSourcePathFromSettings(const QJsonObject &settings)
{
    QString source_path =
        settings.value(QStringLiteral("depthMapSourcePath")).toString().trimmed();
    if (source_path.isEmpty())
    {
        source_path = settings.value(QStringLiteral("source_path")).toString().trimmed();
    }
    if (source_path.isEmpty())
    {
        return QString();
    }

    const QFileInfo source_info(source_path);
    return QDir::cleanPath(source_info.isDir()
                               ? source_info.absoluteFilePath()
                               : source_info.absolutePath());
}

QString comparablePath(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

int plyScalarByteSize(const QString &type)
{
    const QString normalized = type.trimmed().toLower();
    if (normalized == QStringLiteral("char") ||
        normalized == QStringLiteral("uchar") ||
        normalized == QStringLiteral("int8") ||
        normalized == QStringLiteral("uint8"))
    {
        return 1;
    }
    if (normalized == QStringLiteral("short") ||
        normalized == QStringLiteral("ushort") ||
        normalized == QStringLiteral("int16") ||
        normalized == QStringLiteral("uint16"))
    {
        return 2;
    }
    if (normalized == QStringLiteral("int") ||
        normalized == QStringLiteral("uint") ||
        normalized == QStringLiteral("int32") ||
        normalized == QStringLiteral("uint32") ||
        normalized == QStringLiteral("float") ||
        normalized == QStringLiteral("float32"))
    {
        return 4;
    }
    if (normalized == QStringLiteral("double") ||
        normalized == QStringLiteral("float64") ||
        normalized == QStringLiteral("int64") ||
        normalized == QStringLiteral("uint64"))
    {
        return 8;
    }
    return 0;
}

bool hasValidPlyVertices(const QString &path,
                         const QJsonObject &record)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }

    constexpr qint64 kMaxPlyHeaderBytes = 64 * 1024;
    const QByteArray header = file.read(kMaxPlyHeaderBytes);
    if (!header.startsWith("ply\n") && !header.startsWith("ply\r\n"))
    {
        return false;
    }
    const int header_end = header.indexOf("end_header");
    if (header_end < 0)
    {
        return false;
    }

    const QString header_text = QString::fromLatin1(header.left(header_end));
    const QRegularExpression vertex_expression(
        QStringLiteral("(?:^|[\\r\\n])element\\s+vertex\\s+(\\d+)(?:[\\r\\n]|$)"));
    const QRegularExpressionMatch match = vertex_expression.match(header_text);
    if (!match.hasMatch())
    {
        return false;
    }

    bool count_ok = false;
    const qint64 vertex_count = match.captured(1).toLongLong(&count_ok);
    if (!count_ok || vertex_count <= 0)
    {
        return false;
    }
    if (record.contains(QStringLiteral("point_count")) &&
        record.value(QStringLiteral("point_count")).toVariant().toLongLong() != vertex_count)
    {
        return false;
    }
    const int payload_offset = header.indexOf('\n', header_end);
    if (payload_offset < 0 || file.size() <= payload_offset + 1)
    {
        return false;
    }

    bool ascii_format = false;
    bool binary_format = false;
    bool vertex_element = false;
    int vertex_property_count = 0;
    qint64 vertex_stride = 0;
    const QStringList header_lines = header_text.split(QRegularExpression(QStringLiteral("[\\r\\n]+")),
                                                       Qt::SkipEmptyParts);
    for (const QString &header_line : header_lines)
    {
        const QString line = header_line.trimmed();
        if (line.startsWith(QStringLiteral("format ")))
        {
            ascii_format = line.startsWith(QStringLiteral("format ascii "));
            binary_format = line.startsWith(QStringLiteral("format binary_little_endian ")) ||
                line.startsWith(QStringLiteral("format binary_big_endian "));
        }
        else if (line.startsWith(QStringLiteral("element ")))
        {
            vertex_element = line.startsWith(QStringLiteral("element vertex "));
        }
        else if (vertex_element && line.startsWith(QStringLiteral("property ")))
        {
            const QStringList tokens = line.split(QLatin1Char(' '), Qt::SkipEmptyParts);
            if (tokens.size() < 3 || tokens.at(1) == QStringLiteral("list"))
            {
                return false;
            }
            const int property_size = plyScalarByteSize(tokens.at(1));
            if (property_size <= 0)
            {
                return false;
            }
            ++vertex_property_count;
            vertex_stride += property_size;
        }
    }
    if ((!ascii_format && !binary_format) || vertex_property_count <= 0)
    {
        return false;
    }

    const qint64 payload_bytes = file.size() - payload_offset - 1;
    const qint64 minimum_vertex_bytes = binary_format
        ? vertex_stride
        : static_cast<qint64>(vertex_property_count) * 2;
    if (minimum_vertex_bytes <= 0 ||
        vertex_count > payload_bytes / minimum_vertex_bytes)
    {
        return false;
    }
    return true;
}

QString reusableDenseCloudForDepthSource(
    const QString &depth_map_source_path,
    const QJsonObject &project_metadata,
    const xjw::core::project::StoredDepthFramesResult &stored_frames,
    const QString &depth_config_hash,
    const QString &project_input_signature)
{
    if (depth_map_source_path.trimmed().isEmpty() || !stored_frames.status.ok ||
        stored_frames.frames.size() < 2 || depth_config_hash.isEmpty())
    {
        return QString();
    }

    const QString directory_path = comparablePath(depth_map_source_path);
    const QString dense_cloud_path = QDir(directory_path).filePath(QStringLiteral("dense_cloud.ply"));
    const QFileInfo dense_info(dense_cloud_path);
    if (!dense_info.isFile())
    {
        return QString();
    }

    bool metadata_matches = false;
    const QJsonArray dense_results =
        project_metadata.value(QStringLiteral("dense_cloud_results")).toArray();
    for (const QJsonValue &value : dense_results)
    {
        const QJsonObject record = value.toObject();
        if (comparablePath(record.value(QStringLiteral("dense_cloud_xyz")).toString()) !=
                comparablePath(dense_cloud_path) ||
            comparablePath(record.value(QStringLiteral("source_depth_map_dir")).toString()) !=
                directory_path ||
            record.value(QStringLiteral("source_depth_map_count")).toInt() !=
                static_cast<int>(stored_frames.frames.size()) ||
            record.value(QStringLiteral("source_depth_config_hash")).toString() !=
                depth_config_hash ||
            record.value(QStringLiteral("fusion_pipeline_version")).toInt() !=
                kDenseFusionPipelineVersion ||
            (!project_input_signature.isEmpty() &&
             record.value(QStringLiteral("source_project_input_signature")).toString() !=
                 project_input_signature) ||
            !hasValidPlyVertices(dense_cloud_path, record))
        {
            continue;
        }
        metadata_matches = true;
        break;
    }
    if (!metadata_matches)
    {
        return QString();
    }

    for (const auto &frame : stored_frames.frames)
    {
        if (QFileInfo(frame.rawDepthPath).lastModified() > dense_info.lastModified())
        {
            return QString();
        }
    }
    return dense_cloud_path;
}

QString consistentDepthConfigHash(
    const xjw::core::project::StoredDepthFramesResult &stored_frames)
{
    QString config_hash;
    for (const auto &frame : stored_frames.frames)
    {
        if (frame.configHash.isEmpty())
        {
            return QString();
        }
        if (config_hash.isEmpty())
        {
            config_hash = frame.configHash;
        }
        else if (config_hash != frame.configHash)
        {
            return QString();
        }
    }
    return config_hash;
}

QString consistentProjectInputSignature(
    const xjw::core::project::StoredDepthFramesResult &stored_frames)
{
    QString input_signature;
    for (const auto &frame : stored_frames.frames)
    {
        if (frame.projectInputSignature.isEmpty())
        {
            return QString();
        }
        if (input_signature.isEmpty())
        {
            input_signature = frame.projectInputSignature;
        }
        else if (input_signature != frame.projectInputSignature)
        {
            return QString();
        }
    }
    return input_signature;
}

int expectedDepthFrameCount(const QJsonObject &project_metadata,
                            const QString &depth_map_source_path,
                            int available_frame_count)
{
    const QString target_directory = comparablePath(depth_map_source_path);
    int expected_count = 0;
    const QJsonArray records =
        project_metadata.value(QStringLiteral("depth_map_results")).toArray();
    for (const QJsonValue &value : records)
    {
        const QJsonObject record = value.toObject();
        const QString raw_depth_path =
            record.value(QStringLiteral("raw_depth_path")).toString();
        if (raw_depth_path.trimmed().isEmpty())
        {
            continue;
        }
        if (comparablePath(QFileInfo(raw_depth_path).absolutePath()).compare(
                target_directory,
                Qt::CaseInsensitive) != 0)
        {
            continue;
        }
        expected_count = qMax(
            expected_count,
            record.value(QStringLiteral("batch_frame_count")).toInt(0));
    }
    return qMax(expected_count, available_frame_count);
}

} // namespace

QString projectDepthInputSignature(const QJsonObject &project_metadata,
                                   int aerial_triangulation_result_index)
{
    const QJsonArray images = project_metadata.value(QStringLiteral("images")).toArray();
    const QJsonArray at_results =
        project_metadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (images.isEmpty() && at_results.isEmpty())
    {
        return QString();
    }

    QJsonObject signature_input;
    signature_input[QStringLiteral("images")] = images;
    int at_index = aerial_triangulation_result_index;
    if (at_index < 0 || at_index >= at_results.size())
    {
        at_index = findLatestProductionAtResultIndex(project_metadata);
    }
    if (at_index < 0 && !at_results.isEmpty())
    {
        at_index = at_results.size() - 1;
    }
    if (at_index >= 0)
    {
        signature_input[QStringLiteral("aerial_triangulation_result")] =
            at_results.at(at_index).toObject();
    }

    const QByteArray payload =
        QJsonDocument(signature_input).toJson(QJsonDocument::Compact);
    return QString::fromLatin1(
        QCryptographicHash::hash(payload, QCryptographicHash::Sha256).toHex());
}

QJsonObject denseSettingsFromModelSettings(const QJsonObject &settings,
                                            const QString &depth_map_source_path)
{
    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("high"));

    QJsonObject dense_settings;
    dense_settings[QStringLiteral("pipeline_mode")] = true;
    dense_settings[QStringLiteral("output_dir")] = depth_map_source_path;
    dense_settings[QStringLiteral("qualityProfile")] = modelQualityToDenseProfile(quality);
    dense_settings[QStringLiteral("resScale")] = modelQualityToDenseResolutionScale(quality);
    dense_settings[QStringLiteral("iterations")] = modelQualityToDenseIterations(quality);
    dense_settings[QStringLiteral("threads")] = qMax(1, settings.value(QStringLiteral("threads")).toInt(8));
    dense_settings[QStringLiteral("cuda")] = settings.value(QStringLiteral("cuda")).toBool(true);
    dense_settings[QStringLiteral("keepColor")] =
        settings.value(QStringLiteral("calculateVertexColors")).toBool(true);
    dense_settings[QStringLiteral("keepNormals")] = true;
    dense_settings[QStringLiteral("minViews")] = settings.value(QStringLiteral("minViews")).toInt(6);
    dense_settings[QStringLiteral("minConsistentViews")] =
        settings.value(QStringLiteral("minConsistentViews")).toInt(3);
    dense_settings[QStringLiteral("confidence")] =
        settings.value(QStringLiteral("depthPatchConfidence")).toDouble(0.60);
    dense_settings[QStringLiteral("minConfidence")] =
        settings.value(QStringLiteral("depthFusionConfidence")).toDouble(0.65);
    dense_settings[QStringLiteral("fusionMaxImageDim")] =
        settings.value(QStringLiteral("fusionMaxImageDim")).toInt(2048);
    dense_settings[QStringLiteral("geomConsistency")] =
        settings.value(QStringLiteral("geomConsistency")).toBool(true);
    dense_settings[QStringLiteral("maxReprojError")] =
        settings.value(QStringLiteral("maxReprojError")).toDouble(1.5);
    dense_settings[QStringLiteral("depthConsistency")] =
        settings.value(QStringLiteral("depthConsistency")).toDouble(1.5);
    dense_settings[QStringLiteral("speckleMinArea")] =
        settings.value(QStringLiteral("speckleMinArea")).toInt(16);
    if (settings.contains(QStringLiteral("at_index")))
    {
        dense_settings[QStringLiteral("at_index")] =
            settings.value(QStringLiteral("at_index"));
    }
    return dense_settings;
}

ModelWorkflowDecision decideModelGenerationWorkflow(const QJsonObject &settings,
                                                     const QJsonObject &project_metadata)
{
    ModelWorkflowDecision decision;
    decision.modelSettings = settings;
    if (!decision.modelSettings.contains(QStringLiteral("threads")))
    {
        decision.modelSettings[QStringLiteral("threads")] =
            recommendedInteractiveModelWorkerCount(QThread::idealThreadCount());
    }

    const QString source_data =
        settings.value(QStringLiteral("source_data")).toString(QStringLiteral("point_cloud"));
    if (source_data != QStringLiteral("depth_maps"))
    {
        decision.reason = QStringLiteral("源数据不是深度图，直接执行模型生成。");
        return decision;
    }

    decision.depthMapSourcePath = depthSourcePathFromSettings(settings);
    decision.modelSettings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    decision.modelSettings[QStringLiteral("depthMapSourcePath")] = decision.depthMapSourcePath;
    decision.modelSettings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
    decision.modelSettings.remove(QStringLiteral("source_point_cloud_path"));
    const bool reuse_depth_maps = settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    const auto stored_frames = decision.depthMapSourcePath.isEmpty()
        ? xjw::core::project::collectLatestStoredDepthFrames(project_metadata)
        : xjw::core::project::collectStoredDepthFramesForDirectory(
              project_metadata,
              decision.depthMapSourcePath);
    const int expected_frame_count = expectedDepthFrameCount(
        project_metadata,
        decision.depthMapSourcePath,
        static_cast<int>(stored_frames.frames.size()));
    const QString depth_config_hash = consistentDepthConfigHash(stored_frames);
    const int requested_at_index = settings.value(QStringLiteral("at_index")).toInt(-1);
    const QString current_input_signature =
        projectDepthInputSignature(project_metadata, requested_at_index);
    const QString stored_input_signature = consistentProjectInputSignature(stored_frames);
    const bool input_signature_matches = current_input_signature.isEmpty() ||
        stored_input_signature == current_input_signature;
    const bool stored_batch_complete = stored_frames.status.ok &&
        stored_frames.frames.size() >= 2 &&
        static_cast<int>(stored_frames.frames.size()) >= expected_frame_count &&
        !depth_config_hash.isEmpty() && input_signature_matches;

    if (reuse_depth_maps && stored_batch_complete)
    {
        decision.reason = QStringLiteral("已有完整深度图批次，直接进行 TSDF 表面重建。");
        return decision;
    }

    decision.action = ModelWorkflowAction::GenerateDepthMapsThenMesh;
    decision.depthSettings = denseSettingsFromModelSettings(settings, decision.depthMapSourcePath);
    decision.depthSettings[QStringLiteral("workflow_action")] =
        QStringLiteral("generate_depth_maps");
    decision.depthSettings[QStringLiteral("force_depth_recompute")] = !reuse_depth_maps;
    decision.depthSettings[QStringLiteral("expected_depth_frame_count")] = expected_frame_count;
    decision.reason = reuse_depth_maps
        ? QStringLiteral("缺少完整可复用深度图批次，先自动估计深度图。")
        : QStringLiteral("未启用深度图复用，重新估计深度图。");
    return decision;
}

} // namespace xjw::gui::project
