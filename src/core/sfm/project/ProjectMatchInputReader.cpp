#include "ProjectMatchInputReader.h"

#include "ImageMatchFile.h"
#include "ProjectCameraIO.h"
#include "camera/core/capabilities/CameraOperationPlan.h"
#include "camera/core/model/CameraInstanceSet.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/project/CameraProjectRuntime.h"
#include "camera/project/CameraProjectRecords.h"
#include "project/ProjectCommonUtils.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QMap>
#include <QSet>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace xjw::core::project
{
    namespace
    {

        using xjw::image_matching::ImageMatchFile;
        using xjw::image_matching::ImageMatchShard;
        using xjw::image_matching::MatchRecordFlag;
        using xjw::image_matching::NeighborMatchBlock;

        /**
         * @brief 一个无向像对当前最优的持久化算法变体。
         *
         * 对称分片会各保存一份有向邻接块，同一像对也允许多个配置指纹并存。读取层先
         * 把每个候选转换为 SfM 的统一方向，再按几何内点数、连接点数和原始匹配数排序，
         * 最终只向轨迹构建器提交一个变体，避免重复观测改变 BA 权重。
         */
        struct PairCandidate
        {
            ProjectMatchPair pair;
            int geometricInlierCount = 0;
            int tiePointMatchCount = 0;
            int rawMatchCount = 0;
            std::int64_t createdTimeMs = 0;
        };

        QString normalizedPath(const QString& path)
        {
            const QString trimmed = path.trimmed();
            return trimmed.isEmpty() ? QString() : xjw::common::project::normalizePath(trimmed);
        }

        struct SelectedRuntimeInstance
        {
            QString normalizedPath;
            std::shared_ptr<const xjw::camera_core::CameraInstance> instance;
        };

        QString canonicalPairKey(int leftCameraIndex, int rightCameraIndex)
        {
            if (leftCameraIndex < 0 || rightCameraIndex < 0 || leftCameraIndex == rightCameraIndex)
            {
                return QString();
            }
            const int first = std::min(leftCameraIndex, rightCameraIndex);
            const int second = std::max(leftCameraIndex, rightCameraIndex);
            return QStringLiteral("%1\n%2").arg(first).arg(second);
        }

        QString imageFileNameKey(const QString& path)
        {
            return QFileInfo(QDir::fromNativeSeparators(path.trimmed())).fileName().toCaseFolded();
        }

        QJsonObject numericStateSnapshot(
            const xjw::camera_models::frame_pinhole::FramePinholeNumericState& state)
        {
            return xjw::common::project::serializeFramePinholeNumericState(state);
        }

        bool betterCandidate(const PairCandidate& left, const PairCandidate& right)
        {
            if (left.geometricInlierCount != right.geometricInlierCount)
            {
                return left.geometricInlierCount > right.geometricInlierCount;
            }
            if (left.tiePointMatchCount != right.tiePointMatchCount)
            {
                return left.tiePointMatchCount > right.tiePointMatchCount;
            }
            if (left.rawMatchCount != right.rawMatchCount)
            {
                return left.rawMatchCount > right.rawMatchCount;
            }
            return left.createdTimeMs > right.createdTimeMs;
        }

        /**
         * @brief 把 owner->peer 邻接块转换为 SfM 像对。
         *
         * 只保留通过几何模型验证的记录。owner 坐标由分片 observations 按稳定 featureId
         * 查找，peer 坐标直接来自邻接记录；两端 featureId 可跨多个像对合并成多视轨迹。
         */
        PairCandidate makePairCandidate(const ImageMatchShard& shard,
                                        const NeighborMatchBlock& block,
                                        int owner_camera_index,
                                        int peer_camera_index)
        {
            PairCandidate candidate;
            candidate.pair.cameraIndexA = owner_camera_index;
            candidate.pair.cameraIndexB = peer_camera_index;
            candidate.pair.indexed = true;
            candidate.geometricInlierCount = static_cast<int>(block.geometryInlierCount);
            candidate.tiePointMatchCount = static_cast<int>(block.tiePointMatchCount);
            candidate.rawMatchCount = static_cast<int>(block.rawMatchCount);
            candidate.createdTimeMs = block.createdTimeMs;
            candidate.pair.observations.reserve(block.geometryInlierCount);

            for (const xjw::image_matching::MatchRecord& match : block.matches)
            {
                if (!xjw::image_matching::hasFlag(match.flags, MatchRecordFlag::GeometryInlier))
                {
                    continue;
                }
                const xjw::image_matching::KeypointObservation* owner_observation =
                    block.findOwnerObservation(match.ownerFeatureId);
                if (!owner_observation)
                {
                    continue;
                }

                ProjectMatchObservationPair observation;
                observation.pixelA = {static_cast<double>(owner_observation->x),
                                      static_cast<double>(owner_observation->y)};
                observation.pixelB = {static_cast<double>(match.peerX), static_cast<double>(match.peerY)};
                observation.featureA = static_cast<xjw::FeatureIdx>(match.ownerFeatureId);
                observation.featureB = static_cast<xjw::FeatureIdx>(match.peerFeatureId);
                observation.score = std::clamp(static_cast<double>(match.confidence), 0.0, 1.0);
                candidate.pair.observations.push_back(observation);
            }
            return candidate;
        }

    } // namespace

    int cameraIndexForImageToken(const QString& imageToken, const QMap<QString, int>& cameraIndexByPath)
    {
        if (imageToken.trimmed().isEmpty())
        {
            return -1;
        }

        const QString normalized_token = normalizedPath(imageToken);
        const auto direct = cameraIndexByPath.constFind(normalized_token);
        // 该公共入口供控制点、标记等输入共同使用，只执行严格规范路径解析。
        // 即使存在 exact key，也要继续扫描等价的未规范化别名，避免 QMap 的
        // 迭代顺序或 key 形态掩盖一对多身份。
        if (direct != cameraIndexByPath.constEnd() && direct.value() < 0)
        {
            return -1;
        }
        int resolved_index = direct == cameraIndexByPath.constEnd() ? -1 : direct.value();
        for (auto it = cameraIndexByPath.constBegin(); it != cameraIndexByPath.constEnd(); ++it)
        {
            if (xjw::common::project::pathTokenMatchesImage(imageToken, it.key()))
            {
                if (it.value() < 0)
                {
                    return -1;
                }
                if (resolved_index >= 0 && resolved_index != it.value())
                {
                    // Multiple aliases matching the same token must never select
                    // the first QMap entry by iteration order.
                    return -1;
                }
                resolved_index = it.value();
            }
        }
        return resolved_index;
    }

    int cameraIndexForRelocatedMatchToken(const QString& imageToken,
                                          const QMap<QString, int>& cameraIndexByPath,
                                          const QStringList& allChunkImagePaths)
    {
        const int strictIndex = cameraIndexForImageToken(imageToken, cameraIndexByPath);
        if (strictIndex >= 0)
        {
            return strictIndex;
        }

        const QString fileName = imageFileNameKey(imageToken);
        if (fileName.isEmpty())
        {
            return -1;
        }

        int fullChunkMatches = 0;
        for (const QString& path : allChunkImagePaths)
        {
            if (imageFileNameKey(path) == fileName)
            {
                ++fullChunkMatches;
            }
        }
        if (fullChunkMatches != 1)
        {
            return -1;
        }

        int resolvedIndex = -1;
        for (auto it = cameraIndexByPath.constBegin(); it != cameraIndexByPath.constEnd(); ++it)
        {
            if (imageFileNameKey(it.key()) != fileName)
            {
                continue;
            }
            if (resolvedIndex >= 0 && resolvedIndex != it.value())
            {
                return -1;
            }
            resolvedIndex = it.value();
        }
        return resolvedIndex;
    }

    bool readProjectMatchInput(const QJsonObject& meta,
                               const QStringList& selectedImages,
                               int minMatches,
                               ProjectMatchInput* input)
    {
        if (!input)
        {
            return false;
        }
        *input = {};

        // Decode the canonical definition/instance graph before constructing the
        // numerical pinhole view required by the current SfM kernels.  This
        // keeps the capability boundary at the project entry point: RPC and
        // pushbroom instances are reported as unsupported instead of being
        // silently interpreted as static pinhole cameras.
        const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
            meta, xjw::camera_models::makeBuiltinCameraModelRegistry());
        if (!runtime.ok())
        {
            input->diagnostics.firstCameraError = runtime.errors.join(QStringLiteral("; "));
            return false;
        }

        // 第一阶段：按 selectedImages 的集合约束建立当前相机索引。数组实际顺序沿用
        // 工程 images，以保证相机 JSON、影像路径和后续事务式回写始终一一对应。
        QSet<QString> selected_normalized;
        QStringList selected_normalized_order;
        selected_normalized_order.reserve(selectedImages.size());
        for (const QString& path : selectedImages)
        {
            const QString normalized_path = normalizedPath(path);
            if (normalized_path.isEmpty())
            {
                input->diagnostics.firstInputError = QStringLiteral("selected image path is empty");
                return false;
            }
            if (selected_normalized.contains(normalized_path))
            {
                input->diagnostics.firstInputError =
                    QStringLiteral("selected images contain the same normalized path more than once: %1")
                        .arg(normalized_path);
                return false;
            }
            selected_normalized.insert(normalized_path);
            selected_normalized_order.append(normalized_path);
        }

        const QJsonArray image_array = meta.value(QStringLiteral("images")).toArray();
        QStringList allChunkImagePaths;
        allChunkImagePaths.reserve(image_array.size());
        QSet<QString> selected_found_paths;
        QSet<QString> image_ids_in_document;
        QMap<QString, int> image_row_by_path;
        std::vector<SelectedRuntimeInstance> selectedRuntimeInstances;
        xjw::camera_core::CameraInstanceSet selectedInstances;
        auto rejectSelectedCamera = [input](const QString& normalized_path,
                                             const QString& image_id,
                                             const QString& reason)
        {
            ++input->diagnostics.unsupportedCameraCount;
            const QString identity = image_id.isEmpty() ? QStringLiteral("<missing image_uuid>") : image_id;
            input->diagnostics.firstCameraError =
                QStringLiteral("selected image %1 (%2) cannot enter the static frame-pinhole solver: %3")
                    .arg(identity, normalized_path, reason);
            return false;
        };
        for (int image_row = 0; image_row < image_array.size(); ++image_row)
        {
            const QJsonValue& value = image_array.at(image_row);
            const QJsonObject object = value.toObject();
            const QString normalized_path = normalizedPath(object.value(QStringLiteral("path")).toString());
            if (!normalized_path.isEmpty())
            {
                if (image_row_by_path.contains(normalized_path))
                {
                    input->diagnostics.firstInputError =
                        QStringLiteral("project images contain duplicate normalized paths: %1")
                            .arg(normalized_path);
                    return false;
                }
                image_row_by_path.insert(normalized_path, image_row);
                allChunkImagePaths.append(normalized_path);
            }
            const QString imageId = object.value(QStringLiteral("image_uuid")).toString().trimmed();
            if (!imageId.isEmpty())
            {
                if (image_ids_in_document.contains(imageId))
                {
                    input->diagnostics.firstInputError =
                        QStringLiteral("project images contain duplicate image_uuid: %1").arg(imageId);
                    return false;
                }
                image_ids_in_document.insert(imageId);
            }
            if (!selected_normalized.contains(normalized_path))
            {
                continue;
            }
            selected_found_paths.insert(normalized_path);

            if (imageId.isEmpty())
            {
                input->diagnostics.firstInputError =
                    QStringLiteral("selected image %1 has no canonical image_uuid").arg(normalized_path);
                return false;
            }
            const auto runtimeInstance = runtime.instances.forImage(
                xjw::camera_core::ImageId(imageId.toStdString()));
            if (!runtimeInstance.ok())
            {
                return rejectSelectedCamera(normalized_path,
                                            imageId,
                                            QString::fromStdString(runtimeInstance.error));
            }

            std::string frameError;
            if (!selectedInstances.add(runtimeInstance.instance, &frameError))
            {
                return rejectSelectedCamera(normalized_path,
                                            imageId,
                                            QString::fromStdString(frameError));
            }
            selectedRuntimeInstances.push_back({normalized_path, runtimeInstance.instance});
        }

        for (const QString& selected_path : selected_normalized_order)
        {
            if (!selected_found_paths.contains(selected_path))
            {
                input->diagnostics.firstInputError =
                    QStringLiteral("selected image is not present in project images: %1").arg(selected_path);
                return false;
            }
        }

        // The operation planner is the single capability/frame boundary for the static
        // SfM/BA input.  RPC, pushbroom and future camera models are rejected as a
        // complete selection when they do not expose the static operation contract;
        // no pinhole subset is silently constructed.
        const xjw::camera_core::CameraOperationPlan operationPlan =
            xjw::camera_core::planCameraOperation(
                selectedInstances, xjw::camera_core::CameraOperation::StaticSfM);
        if (!operationPlan.ok())
        {
            input->diagnostics.unsupportedCameraCount +=
                static_cast<int>(operationPlan.capabilityFailures.size());
            input->diagnostics.firstCameraError =
                QString::fromStdString(operationPlan.failureMessage());
            return false;
        }

        // Convert typed instances only after the selected set has passed the frame
        // contract.  Keep the conversion in temporary arrays so a later identity
        // or numeric-state failure cannot leave a partial solver input in `input`.
        std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> resolvedCameras;
        std::vector<std::shared_ptr<const xjw::camera_models::frame_pinhole::FramePinholeInstance>>
            resolvedCameraInstances;
        std::vector<xjw::camera_core::ImageId> resolvedImageIds;
        QStringList resolvedImagePaths;
        QMap<QString, int> resolvedCameraIndexByImageId;
        QMap<QString, int> resolvedCameraIndexByPath;
        QMap<QString, QJsonObject> resolvedBeforeCamMeta;
        resolvedCameras.reserve(selectedRuntimeInstances.size());
        resolvedCameraInstances.reserve(selectedRuntimeInstances.size());
        resolvedImageIds.reserve(selectedRuntimeInstances.size());
        resolvedImagePaths.reserve(static_cast<qsizetype>(selectedRuntimeInstances.size()));
        for (const SelectedRuntimeInstance& selected : selectedRuntimeInstances)
        {
            const QString& normalized_path = selected.normalizedPath;
            const std::shared_ptr<const xjw::camera_core::CameraInstance>& instance = selected.instance;
            xjw::camera_models::frame_pinhole::FramePinholeNumericState numericState;
            std::string numericStateError;
            const auto pinhole =
                std::dynamic_pointer_cast<const xjw::camera_models::frame_pinhole::FramePinholeInstance>(instance);
            if (!pinhole)
            {
                return rejectSelectedCamera(
                    normalized_path,
                    QString::fromStdString(instance->imageId().value()),
                    QStringLiteral("camera instance is not a frame-pinhole instance"));
            }
            const bool converted = xjw::camera_models::frame_pinhole::makeFramePinholeNumericState(
                pinhole, &numericState, &numericStateError);
            if (!converted)
            {
                return rejectSelectedCamera(normalized_path,
                                            QString::fromStdString(instance->imageId().value()),
                                            QString::fromStdString(numericStateError));
            }

            const int camera_index = static_cast<int>(resolvedCameras.size());
            const QString image_id = QString::fromStdString(instance->imageId().value());
            if (resolvedCameraIndexByPath.contains(normalized_path)
                || resolvedCameraIndexByImageId.contains(image_id))
            {
                input->diagnostics.firstInputError =
                    QStringLiteral("selected camera identity or path is duplicated: image_uuid=%1, path=%2")
                        .arg(image_id, normalized_path);
                return false;
            }
            if (numericState.imageId() != instance->imageId())
            {
                input->diagnostics.firstInputError =
                    QStringLiteral("numeric camera identity does not match the selected camera instance: %1")
                        .arg(image_id);
                return false;
            }
            resolvedCameraIndexByPath.insert(normalized_path, camera_index);
            resolvedCameraIndexByImageId.insert(image_id, camera_index);
            resolvedCameraInstances.push_back(pinhole);
            resolvedCameras.push_back(std::move(numericState));
            resolvedImageIds.push_back(instance->imageId());
            resolvedImagePaths.append(normalized_path);
            resolvedBeforeCamMeta.insert(normalized_path, numericStateSnapshot(resolvedCameras.back()));
        }

        input->cameras = std::move(resolvedCameras);
        input->cameraInstances = std::move(resolvedCameraInstances);
        input->imageIdByIndex = std::move(resolvedImageIds);
        input->imagePathByIndex = std::move(resolvedImagePaths);
        input->cameraIndexByImageId = std::move(resolvedCameraIndexByImageId);
        input->cameraIndexByPath = std::move(resolvedCameraIndexByPath);
        input->beforeCamMeta = std::move(resolvedBeforeCamMeta);

        // 第二阶段：每个 image_match_results 记录只指向一幅影像的唯一分片。使用
        // output 路径去重后读取，不扫描目录，也不依赖文件名编码影像对。
        QSet<QString> visited_files;
        QMap<QString, PairCandidate> best_pairs;
        const QJsonArray match_results = meta.value(QStringLiteral("image_match_results")).toArray();
        input->diagnostics.matchResultRecordCount = match_results.size();
        for (const QJsonValue& value : match_results)
        {
            const QString output_path =
                QFileInfo(value.toObject().value(QStringLiteral("output")).toString()).absoluteFilePath();
            const QString normalized_output = normalizedPath(output_path);
            if (normalized_output.isEmpty() || visited_files.contains(normalized_output))
            {
                continue;
            }
            visited_files.insert(normalized_output);
            if (!QFileInfo::exists(output_path))
            {
                if (input->diagnostics.firstShardReadError.isEmpty())
                {
                    input->diagnostics.firstShardReadError =
                        QStringLiteral("match shard does not exist: %1").arg(output_path);
                }
                continue;
            }
            ++input->diagnostics.existingShardCount;

            ImageMatchShard shard;
            QString read_error;
            if (!ImageMatchFile::read(output_path, &shard, &read_error))
            {
                if (input->diagnostics.firstShardReadError.isEmpty())
                {
                    input->diagnostics.firstShardReadError = read_error;
                }
                continue;
            }
            ++input->diagnostics.readableShardCount;
            const int owner_index =
                cameraIndexForRelocatedMatchToken(shard.owner.path, input->cameraIndexByPath, allChunkImagePaths);
            if (owner_index < 0)
            {
                continue;
            }
            ++input->diagnostics.resolvedOwnerShardCount;

            for (const NeighborMatchBlock& block : shard.neighbors)
            {
                if (!block.geometryPassed)
                {
                    continue;
                }
                ++input->diagnostics.geometryPassedBlockCount;
                const int peer_index =
                    cameraIndexForRelocatedMatchToken(block.peer.path, input->cameraIndexByPath, allChunkImagePaths);
                const QString pair_key = canonicalPairKey(owner_index, peer_index);
                if (peer_index < 0 || peer_index == owner_index || pair_key.isEmpty())
                {
                    continue;
                }
                ++input->diagnostics.resolvedPeerBlockCount;

                PairCandidate candidate = makePairCandidate(shard, block, owner_index, peer_index);
                if (minMatches > 0 && static_cast<int>(candidate.pair.observations.size()) < minMatches)
                {
                    ++input->diagnostics.rejectedByMinMatchesCount;
                    continue;
                }

                const auto existing = best_pairs.constFind(pair_key);
                if (existing == best_pairs.constEnd() || betterCandidate(candidate, existing.value()))
                {
                    best_pairs[pair_key] = std::move(candidate);
                }
            }
        }

        // 第三阶段：QMap 的规范 pair key 保证输出顺序稳定，便于复现实验和单元测试。
        input->pairs.reserve(static_cast<std::size_t>(best_pairs.size()));
        for (auto it = best_pairs.begin(); it != best_pairs.end(); ++it)
        {
            input->indexedObservationCount += static_cast<int>(it.value().pair.observations.size());
            input->pairs.push_back(std::move(it.value().pair));
        }
        input->diagnostics.acceptedPairCount = static_cast<int>(input->pairs.size());
        return true;
    }

} // namespace xjw::core::project
