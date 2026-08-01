#include "ProjectModelWorkflowPolicy.h"

#include "DepthFrameUtils.h"
#include "ProjectWorkflowUtils.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QtGlobal>

#include <algorithm>

namespace xjw::gui::project
{

int recommendedInteractiveModelWorkerCount(int ideal_thread_count)
{
    const int available_threads = ideal_thread_count > 0 ? ideal_thread_count : 4;
    return qBound(1, available_threads - 2, 8);
}

namespace
{


QString comparablePath(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
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

QString consistentReconstructionGenerationId(
    const xjw::core::project::StoredDepthFramesResult &stored_frames)
{
    QString generation_id;
    for (const auto &frame : stored_frames.frames)
    {
        if (frame.reconstructionGenerationId.isEmpty())
        {
            return QString();
        }
        if (generation_id.isEmpty())
        {
            generation_id = frame.reconstructionGenerationId;
        }
        else if (generation_id != frame.reconstructionGenerationId)
        {
            return QString();
        }
    }
    return generation_id;
}

QJsonObject selectedAerialTriangulationResult(const QJsonObject &project_metadata,
                                               int requested_index)
{
    const QJsonArray at_results =
        project_metadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
    int at_index = requested_index;
    if (at_index < 0 || at_index >= at_results.size())
    {
        at_index = findLatestProductionAtResultIndex(project_metadata);
    }
    if (at_index < 0 && !at_results.isEmpty())
    {
        at_index = at_results.size() - 1;
    }
    return at_index >= 0 ? at_results.at(at_index).toObject() : QJsonObject();
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

StoredDepthBatchCompatibility assessStoredDepthBatchCompatibility(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path,
    int aerial_triangulation_result_index)
{
    StoredDepthBatchCompatibility result;
    const auto stored_frames = depth_map_source_path.trimmed().isEmpty()
        ? xjw::core::project::collectLatestStoredDepthFrames(project_metadata)
        : xjw::core::project::collectStoredDepthFramesForDirectory(
              project_metadata,
              depth_map_source_path);
    result.frameCount = static_cast<int>(stored_frames.frames.size());
    if (!stored_frames.status.ok)
    {
        result.reason = QStringLiteral("无法使用所选深度图批次：%1")
                            .arg(stored_frames.status.errorMessage);
        return result;
    }
    if (stored_frames.frames.size() < 2)
    {
        result.reason = QStringLiteral("可用深度图数量不足（至少需要 2 张）。");
        return result;
    }

    const QJsonObject selected_at_result = selectedAerialTriangulationResult(
        project_metadata,
        aerial_triangulation_result_index);
    const int expected_frame_count = qMax(
        expectedDepthFrameCount(project_metadata,
                                stored_frames.batchDir,
                                result.frameCount),
        selected_at_result.value(QStringLiteral("selected_images"))
            .toArray()
            .size());
    if (result.frameCount < expected_frame_count)
    {
        result.reason = QStringLiteral(
            "深度图批次不完整（已有 %1/%2 帧），不能直接作为可复用批次。"
            "请继续估计缺失深度图或重新计算。")
                            .arg(result.frameCount)
                            .arg(expected_frame_count);
        return result;
    }
    const bool algorithm_revision_matches = std::all_of(
        stored_frames.frames.begin(),
        stored_frames.frames.end(),
        [](const xjw::core::project::StoredDepthFrameRecord &frame)
        {
            return frame.algorithmRevision == xjw::mvs::kMvsDepthAlgorithmRevision;
        });
    if (!algorithm_revision_matches)
    {
        result.reason = QStringLiteral(
            "所选深度图批次由旧版算法生成，不能作为当前模型的几何输入。"
            "请重新估计深度图后再生成模型。");
        return result;
    }

    const QString current_input_signature =
        projectDepthInputSignature(project_metadata, aerial_triangulation_result_index);
    const QString stored_input_signature = consistentProjectInputSignature(stored_frames);
    if (!current_input_signature.isEmpty() && stored_input_signature.isEmpty())
    {
        result.reason = QStringLiteral(
            "所选深度图批次缺少一致的工程输入签名，无法确认其相机参数与当前空三结果一致。"
            "请从“工作流程 → 生成模型”重新估计深度图。");
        return result;
    }
    if (!current_input_signature.isEmpty() &&
        stored_input_signature != current_input_signature)
    {
        result.reason = QStringLiteral(
            "所选深度图批次已过期：当前工程的影像、相机参数或空三结果已发生变化。"
            "为避免融合错误位姿下的深度，请重新估计深度图后再生成模型。");
        return result;
    }

    const QString current_generation_id =
        selected_at_result.value(QStringLiteral("reconstruction_generation_id"))
            .toString();
    const QString stored_generation_id =
        consistentReconstructionGenerationId(stored_frames);
    if (!current_generation_id.isEmpty() && stored_generation_id.isEmpty())
    {
        result.reason = QStringLiteral(
            "所选深度图批次缺少一致的重建代次，无法确认其属于当前空三结果。"
            "请重新估计深度图后再生成模型。");
        return result;
    }
    if (!current_generation_id.isEmpty() &&
        stored_generation_id != current_generation_id)
    {
        result.reason = QStringLiteral(
            "所选深度图批次属于旧的重建代次，不能与当前相机解进行融合。"
            "请重新估计深度图后再生成模型。");
        return result;
    }

    result.compatible = true;
    return result;
}


} // namespace xjw::gui::project
