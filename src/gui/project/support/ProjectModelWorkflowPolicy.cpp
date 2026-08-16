#include "ProjectModelWorkflowPolicy.h"

#include "ProjectDepthBatchLineage.h"
#include "DepthFrameQualificationPolicy.h"
#include "DepthFrameUtils.h"
#include "ProjectWorkflowUtils.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QStringList>
#include <QtGlobal>

#include <algorithm>
#include <set>

namespace xjw::gui::project
{

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

bool samePath(const QString &lhs, const QString &rhs)
{
    const QString comparable_lhs = comparablePath(lhs);
    const QString comparable_rhs = comparablePath(rhs);
    return !comparable_lhs.isEmpty() &&
        comparable_lhs.compare(comparable_rhs, Qt::CaseInsensitive) == 0;
}

bool pathIsInsideDirectory(const QString &path, const QString &directory)
{
    const QString comparable_path = comparablePath(path);
    QString comparable_directory = comparablePath(directory);
    if (comparable_path.isEmpty() || comparable_directory.isEmpty())
    {
        return false;
    }
    if (!comparable_directory.endsWith(QDir::separator()))
    {
        comparable_directory += QDir::separator();
    }
    return comparable_path.startsWith(comparable_directory,
                                      Qt::CaseInsensitive);
}

QString atRecordPath(const QJsonObject &record, const QString &key)
{
    const QJsonObject files = record.value(QStringLiteral("files")).toObject();
    const QString nested_path = files.value(key).toString().trimmed();
    return nested_path.isEmpty()
        ? record.value(key).toString().trimmed()
        : nested_path;
}

QString resolveExistingRecordFile(const QJsonObject &record,
                                  const QString &raw_path)
{
    const QString trimmed_path = raw_path.trimmed();
    if (trimmed_path.isEmpty())
    {
        return QString();
    }

    QStringList candidates;
    if (QDir::isRelativePath(trimmed_path))
    {
        const QString output_dir =
            record.value(QStringLiteral("output_dir")).toString().trimmed();
        if (!output_dir.isEmpty())
        {
            candidates.append(QDir(output_dir).filePath(trimmed_path));
        }
    }
    candidates.append(trimmed_path);

    for (const QString &candidate : candidates)
    {
        const QFileInfo info(candidate);
        if (info.exists() && info.isFile())
        {
            return QDir::cleanPath(info.absoluteFilePath());
        }
    }
    return QString();
}

bool depthRecordMatchesSource(const QJsonObject &record,
                              const QString &depth_map_source_path)
{
    const QFileInfo source_info(depth_map_source_path);
    const bool source_is_directory = source_info.isDir();
    const QString source_path =
        QDir::cleanPath(source_info.absoluteFilePath());
    if (source_path.isEmpty())
    {
        return false;
    }

    const QString output_dir =
        record.value(QStringLiteral("mvs_output_dir")).toString().trimmed();
    if (source_is_directory && samePath(output_dir, source_path))
    {
        return true;
    }

    const QStringList artifact_keys = {
        QStringLiteral("depth_png"),
        QStringLiteral("depth_path"),
        QStringLiteral("raw_depth_path"),
        QStringLiteral("confidence_png"),
        QStringLiteral("raw_confidence_path"),
        QStringLiteral("valid_mask_path"),
        QStringLiteral("provenance_path"),
        QStringLiteral("normal_path")
    };
    for (const QString &key : artifact_keys)
    {
        const QString artifact_path = record.value(key).toString().trimmed();
        if (artifact_path.isEmpty())
        {
            continue;
        }
        if ((source_is_directory &&
             pathIsInsideDirectory(artifact_path, source_path)) ||
            (!source_is_directory && samePath(artifact_path, source_path)))
        {
            return true;
        }
    }

    return !source_is_directory &&
        samePath(output_dir, source_info.absolutePath());
}

QStringList matchingDepthSparseClouds(const QJsonObject &project_metadata,
                                      const QString &depth_map_source_path)
{
    QStringList paths;
    const QJsonArray records =
        project_metadata.value(QStringLiteral("depth_map_results")).toArray();
    for (int index = records.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = records.at(index).toObject();
        if (!depthRecordMatchesSource(record, depth_map_source_path))
        {
            continue;
        }
        const QString sparse_path =
            record.value(QStringLiteral("source_sparse_cloud"))
                .toString()
                .trimmed();
        if (sparse_path.isEmpty())
        {
            continue;
        }
        const bool already_present = std::any_of(
            paths.cbegin(),
            paths.cend(),
            [&sparse_path](const QString &candidate)
            {
                return samePath(candidate, sparse_path);
            });
        if (!already_present)
        {
            paths.append(sparse_path);
        }
    }
    return paths;
}

SparseScaffoldSource scaffoldFromAerialTriangulationResults(
    const QJsonObject &project_metadata,
    const QStringList &source_sparse_clouds)
{
    const QJsonArray records = project_metadata
        .value(QStringLiteral("aerial_triangulation_results"))
        .toArray();
    for (const QString &source_sparse_cloud : source_sparse_clouds)
    {
        for (int index = records.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = records.at(index).toObject();
            const QString raw_point_cloud =
                atRecordPath(record, QStringLiteral("sparse_cloud_xyz"));
            const QString point_cloud =
                resolveExistingRecordFile(record, raw_point_cloud);
            if (point_cloud.isEmpty() ||
                (!samePath(source_sparse_cloud, raw_point_cloud) &&
                 !samePath(source_sparse_cloud, point_cloud)))
            {
                continue;
            }

            const QString points_json = resolveExistingRecordFile(
                record,
                atRecordPath(record,
                             QStringLiteral("sparse_cloud_points_json")));
            if (!points_json.isEmpty())
            {
                return {point_cloud, points_json};
            }
        }
    }
    return {};
}

QString chunkRootForDepthSource(const QString &depth_map_source_path)
{
    const QFileInfo source_info(depth_map_source_path);
    QDir current(source_info.isDir()
                     ? source_info.absoluteFilePath()
                     : source_info.absolutePath());
    while (true)
    {
        if (current.dirName().compare(QStringLiteral("mvs_output"),
                                      Qt::CaseInsensitive) == 0)
        {
            if (current.cdUp())
            {
                return QDir::cleanPath(current.absolutePath());
            }
            return QString();
        }

        const QString before = current.absolutePath();
        if (!current.cdUp() ||
            current.absolutePath().compare(before, Qt::CaseInsensitive) == 0)
        {
            return QString();
        }
    }
}

SparseScaffoldSource canonicalSparseScaffold(
    const QString &depth_map_source_path)
{
    const QString chunk_root = chunkRootForDepthSource(depth_map_source_path);
    if (chunk_root.isEmpty())
    {
        return {};
    }

    const QDir sparse_directory(QDir(chunk_root).filePath(
        QStringLiteral("assets/aerial_triangulation/sfm_sparse")));
    const QString point_cloud = sparse_directory.filePath(
        QStringLiteral("sfm_sparse.ply"));
    const QString points_json = sparse_directory.filePath(
        QStringLiteral("sfm_sparse_points.json"));
    const QFileInfo point_cloud_info(point_cloud);
    const QFileInfo points_json_info(points_json);
    if (!point_cloud_info.exists() || !point_cloud_info.isFile() ||
        !points_json_info.exists() || !points_json_info.isFile())
    {
        return {};
    }
    return {QDir::cleanPath(point_cloud_info.absoluteFilePath()),
            QDir::cleanPath(points_json_info.absoluteFilePath())};
}

int consistentProjectInputSignatureVersion(
    const xjw::core::project::StoredDepthFramesResult &stored_frames)
{
    int version = -1;
    for (const auto &frame : stored_frames.frames)
    {
        if (version < 0)
        {
            version = frame.projectInputSignatureVersion;
        }
        else if (version != frame.projectInputSignatureVersion)
        {
            return 0;
        }
    }
    return version;
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

QString normalizedSceneProfile(const QString &profile)
{
    return profile.trimmed().toLower();
}

bool isExplicitSceneProfile(const QString &profile)
{
    const QString normalized = normalizedSceneProfile(profile);
    return normalized == QStringLiteral("aerial_terrain") ||
        normalized == QStringLiteral("orbital_object");
}

QString sceneProfileDisplayName(const QString &profile)
{
    const QString normalized = normalizedSceneProfile(profile);
    if (normalized == QStringLiteral("aerial_terrain"))
    {
        return QStringLiteral("航拍地形");
    }
    if (normalized == QStringLiteral("orbital_object"))
    {
        return QStringLiteral("环拍目标");
    }
    return normalized.isEmpty() ? QStringLiteral("未记录") : normalized;
}

QString depthBatchSceneCompatibilityReason(
    const xjw::core::project::StoredDepthFramesResult &stored_frames,
    const QString &expected_scene_profile)
{
    std::set<QString> stored_profiles;
    int missing_profile_count = 0;
    for (const auto &frame : stored_frames.frames)
    {
        const QString profile = normalizedSceneProfile(frame.sceneProfile);
        if (profile.isEmpty())
        {
            ++missing_profile_count;
        }
        else
        {
            stored_profiles.insert(profile);
        }
    }

    if (stored_profiles.size() > 1 ||
        (!stored_profiles.empty() && missing_profile_count > 0))
    {
        return QStringLiteral(
            "所选深度图批次的场景类型记录不一致，不能安全复用。"
            "请按当前场景重新估计完整深度图批次。");
    }
    if (!stored_profiles.empty() &&
        !isExplicitSceneProfile(*stored_profiles.begin()))
    {
        return QStringLiteral(
            "所选深度图批次记录了无法识别的场景类型“%1”，不能安全复用。"
            "请重新估计深度图后再生成模型。")
            .arg(*stored_profiles.begin());
    }

    const QString expected_profile = normalizedSceneProfile(expected_scene_profile);
    if (stored_profiles.empty())
    {
        if (isExplicitSceneProfile(expected_profile))
        {
            return QStringLiteral(
                "所选深度图批次缺少场景类型记录，无法确认它与当前%1策略兼容。"
                "请重新估计深度图后再生成模型。")
                .arg(sceneProfileDisplayName(expected_profile));
        }
        return QStringLiteral(
            "所选深度图批次缺少自动分类得到的场景类型记录，不能安全复用。"
            "请重新估计深度图后再生成模型。");
    }
    if (!isExplicitSceneProfile(expected_profile))
    {
        return QString();
    }

    const QString stored_profile = *stored_profiles.begin();
    if (stored_profile != expected_profile)
    {
        return QStringLiteral(
            "所选深度图批次按%1策略生成，但当前工程需要%2策略，不能安全复用。"
            "请按当前场景重新估计深度图后再生成模型。")
            .arg(sceneProfileDisplayName(stored_profile),
                 sceneProfileDisplayName(expected_profile));
    }
    return QString();
}

QString depthBatchFusionEligibilityReason(
    const xjw::core::project::StoredDepthFramesResult &stored_frames,
    bool allow_orbital_sparse_scaffold_fallback)
{
    const bool orbital_batch = std::all_of(
        stored_frames.frames.cbegin(),
        stored_frames.frames.cend(),
        [](const xjw::core::project::StoredDepthFrameRecord &frame)
        {
            return normalizedSceneProfile(frame.sceneProfile) ==
                QStringLiteral("orbital_object");
        });
    const int minimum_primary_frame_count = orbital_batch
        ? xjw::mvs::minimumOrbitalPrimaryDepthFrameCount(
              static_cast<int>(stored_frames.frames.size()))
        : 1;

    int accepted_count = 0;
    int validation_only_count = 0;
    int fusion_eligible_count = 0;
    int primary_frame_count = 0;
    bool has_complete_quality_metadata = true;
    for (const auto &frame : stored_frames.frames)
    {
        const QString acceptance = frame.acceptance.trimmed().toLower();
        has_complete_quality_metadata = has_complete_quality_metadata &&
            !acceptance.isEmpty() && frame.fusionEligibilityKnown;
        if (acceptance == QStringLiteral("accepted"))
        {
            ++accepted_count;
        }
        else if (acceptance == QStringLiteral("validation_only"))
        {
            ++validation_only_count;
        }
        if (frame.fusionEligible)
        {
            ++fusion_eligible_count;
        }
        if (xjw::mvs::isPrimaryFusionFrame(
                frame.acceptance, frame.fusionEligible))
        {
            ++primary_frame_count;
        }
    }

    if (!has_complete_quality_metadata)
    {
        return QStringLiteral(
            "所选深度图批次缺少完整的帧质量资格记录（acceptance/fusion_eligible），"
            "无法确认它能安全进入多视融合。请重新估计深度图后再生成模型。");
    }
    const bool sparse_scaffold_can_carry_global_shape =
        orbital_batch &&
        allow_orbital_sparse_scaffold_fallback &&
        primary_frame_count >= 2;
    if (primary_frame_count < minimum_primary_frame_count &&
        !sparse_scaffold_can_carry_global_shape)
    {
        return QStringLiteral(
            "所选深度图批次没有足够的融合主帧：accepted=%1，fusion_eligible=%2，"
            "validation_only=%3；至少需要 %4 帧同时满足 accepted 和 fusion_eligible。"
            "辅助验证帧不能替代多相机 TSDF 主输入，请重新估计深度图。")
            .arg(accepted_count)
            .arg(fusion_eligible_count)
            .arg(validation_only_count)
            .arg(minimum_primary_frame_count);
    }
    return QString();
}

} // namespace

QString projectDepthInputSignature(const QJsonObject &project_metadata,
                                   int aerial_triangulation_result_index)
{
    return canonicalProjectDepthInputSignature(
        project_metadata,
        aerial_triangulation_result_index,
        kProjectDepthInputSignatureVersion);
}

SparseScaffoldSource resolveSparseScaffoldSource(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path)
{
    if (depth_map_source_path.trimmed().isEmpty())
    {
        return {};
    }

    const SparseScaffoldSource matched_source =
        scaffoldFromAerialTriangulationResults(
            project_metadata,
            matchingDepthSparseClouds(project_metadata,
                                      depth_map_source_path));
    if (!matched_source.pointCloudPath.isEmpty() &&
        !matched_source.pointsJsonPath.isEmpty())
    {
        return matched_source;
    }
    return canonicalSparseScaffold(depth_map_source_path);
}

StoredDepthBatchCompatibility assessStoredDepthBatchCompatibility(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path,
    int aerial_triangulation_result_index,
    const QString &expected_scene_profile,
    bool allow_orbital_sparse_scaffold_fallback)
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
    const QString effective_expected_scene_profile = normalizedSceneProfile(
        expected_scene_profile);
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
            return frame.algorithmRevision >=
                    xjw::mvs::kMvsMinimumModelCompatibleRevision &&
                frame.algorithmRevision <= xjw::mvs::kMvsDepthAlgorithmRevision;
        });
    if (!algorithm_revision_matches)
    {
        result.reason = QStringLiteral(
            "所选深度图批次的算法版本不在当前模型兼容范围内，"
            "不能作为当前模型的几何输入。"
            "请重新估计深度图后再生成模型。");
        return result;
    }

    const QString scene_compatibility_reason = depthBatchSceneCompatibilityReason(
        stored_frames,
        effective_expected_scene_profile);
    if (!scene_compatibility_reason.isEmpty())
    {
        result.reason = scene_compatibility_reason;
        return result;
    }

    const QString fusion_eligibility_reason = depthBatchFusionEligibilityReason(
        stored_frames,
        allow_orbital_sparse_scaffold_fallback);
    if (!fusion_eligibility_reason.isEmpty())
    {
        result.reason = fusion_eligibility_reason;
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
        const int signature_version =
            consistentProjectInputSignatureVersion(stored_frames);
        const bool verified_legacy_batch =
            signature_version == 1 &&
            !current_generation_id.isEmpty() &&
            current_generation_id == stored_generation_id &&
            legacyDepthCamerasMatchCurrentProject(stored_frames, project_metadata);
        if (!verified_legacy_batch)
        {
            result.reason = QStringLiteral(
                "所选深度图批次已过期：当前工程的影像、相机参数或空三结果已发生变化。"
                "为避免融合错误位姿下的深度，请重新估计深度图后再生成模型。");
            return result;
        }
    }

    result.compatible = true;
    return result;
}


} // namespace xjw::gui::project
