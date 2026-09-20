#include "reconstruction/RpcAerialTriangulationRunner.h"
#include "engine/RpcEngine.h"

#include "reconstruction/SfmAttemptRunner.h"

#include "ProjectCameraIO.h"
#include "RpcRasterIO.h"
#include "io/ImageIO.h"
#include "io/PathIO.h"
#include "log/Logger.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/models/rpc/RpcInstance.h"
#include "camera/project/CameraProjectRecords.h"
#include "camera/project/CameraProjectRuntime.h"

#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"
#include "project/SparseResultQuality.h"

#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include "reporting/SparsePlyWriter.h"
#include "file/FileIO.h"
#include <QSet>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace xjw::aerial_triangulation
{
    namespace
    {

        using Observation = engine::RpcObservation;
        using RpcPoint = engine::RpcPoint;
        using CameraResidualAccumulator = engine::CameraResidualAccumulator;

        bool resolveRpcCameraBindings(const PreparedAerialTriangulationInput& input,
                                      const std::map<ImageId, std::shared_ptr<const camera_models::rpc::RpcInstance>>&
                                          cameras,
                                      std::vector<SolverCameraBinding>* bindings,
                                      QString* errorMessage)
        {
            if (!bindings)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("RPC 相机绑定输出为空");
                }
                return false;
            }
            bindings->clear();

            if (input.imageIds.size() != static_cast<std::size_t>(input.images.size()))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("RPC 空三要求 imageIds 与 images 一一对应");
                }
                return false;
            }

            std::vector<SolverCameraBinding> canonicalBindings;
            bool hasCanonicalBindings = false;
            const QJsonObject projectFiles = xjw::common::project::projectFilesRootObject(input.projectMeta);
            QMap<QString, QJsonObject> imageMetadata;
            for (const QJsonValue& value : projectFiles.value(QStringLiteral("images")).toArray())
            {
                const QJsonObject image = value.toObject();
                const QString path = xjw::common::project::normalizePath(
                    image.value(QStringLiteral("path")).toString());
                if (path.isEmpty())
                {
                    continue;
                }
                if (imageMetadata.contains(path))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 空三的 canonical image path 不唯一: %1").arg(path);
                    }
                    return false;
                }
                imageMetadata.insert(path, image);
            }
            const bool hasCameraInstances = projectFiles.value(QStringLiteral("camera_instances")).isArray();
            const bool hasCameraDefinitions = projectFiles.value(QStringLiteral("camera_definitions")).isArray();
            if (hasCameraInstances != hasCameraDefinitions)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("RPC 空三的 canonical camera_definitions/camera_instances 不完整");
                }
                return false;
            }
            if (hasCameraInstances && hasCameraDefinitions &&
                (!projectFiles.value(QStringLiteral("camera_instances")).toArray().isEmpty() ||
                 !projectFiles.value(QStringLiteral("camera_definitions")).toArray().isEmpty()))
            {
                const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
                    projectFiles, xjw::camera_models::makeBuiltinCameraModelRegistry());
                if (!runtime.ok())
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 空三 canonical 相机集合无效: %1")
                                             .arg(runtime.errors.join(QStringLiteral("; ")));
                    }
                    return false;
                }
                canonicalBindings.reserve(static_cast<std::size_t>(input.images.size()));
                for (std::size_t index = 0; index < static_cast<std::size_t>(input.images.size()); ++index)
                {
                    const auto imageIt = imageMetadata.constFind(
                        xjw::common::project::normalizePath(input.images.at(static_cast<int>(index))));
                    if (imageIt == imageMetadata.constEnd())
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral("RPC 空三的 canonical image entry 缺失: %1")
                                                 .arg(input.images.at(static_cast<int>(index)));
                        }
                        return false;
                    }
                    const QString declaredImageId = imageIt.value().value(QStringLiteral("image_uuid")).toString().trimmed();
                    if (declaredImageId != QString::fromStdString(input.imageIds.at(index).value()))
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral("RPC 空三影像路径与 ImageId 不一致: %1")
                                                 .arg(input.images.at(static_cast<int>(index)));
                        }
                        return false;
                    }
                    const auto lookup = runtime.instances.forImage(input.imageIds.at(index));
                    if (!lookup.ok())
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral("RPC 空三无法解析影像 %1 的 canonical 相机: %2")
                                                 .arg(input.images.at(static_cast<int>(index)),
                                                      QString::fromStdString(lookup.error));
                        }
                        return false;
                    }
                    canonicalBindings.push_back({lookup.instance->instanceId(),
                                                 lookup.instance->imageId(),
                                                 lookup.instance->definition().worldFrame()});
                }
                hasCanonicalBindings = canonicalBindings.size() == static_cast<std::size_t>(input.images.size());
            }

            if (input.cameraBindings.empty())
            {
                if (!hasCanonicalBindings)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral(
                            "RPC 空三缺少 canonical cameraBindings；请传入工程相机快照或显式绑定");
                    }
                    return false;
                }
                *bindings = canonicalBindings;
            }
            else
            {
                if (input.cameraBindings.size() != static_cast<std::size_t>(input.images.size()))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 空三要求 cameraBindings 与 images 一一对应");
                    }
                    return false;
                }
                *bindings = input.cameraBindings;
                if (hasCanonicalBindings)
                {
                    for (std::size_t index = 0; index < bindings->size(); ++index)
                    {
                        const SolverCameraBinding& actual = bindings->at(index);
                        const SolverCameraBinding& expected = canonicalBindings.at(index);
                        if (actual.instanceId != expected.instanceId || actual.imageId != expected.imageId ||
                            actual.worldFrame != expected.worldFrame)
                        {
                            if (errorMessage)
                            {
                                *errorMessage = QStringLiteral(
                                    "RPC 空三显式 cameraBindings 与 canonical 工程相机快照不一致（影像 %1）")
                                                     .arg(input.images.at(static_cast<int>(index)));
                            }
                            return false;
                        }
                    }
                }
            }

            std::unordered_set<std::string> instanceIds;
            std::unordered_set<std::string> imageIds;
            instanceIds.reserve(bindings->size());
            imageIds.reserve(bindings->size());
            for (std::size_t index = 0; index < bindings->size(); ++index)
            {
                const SolverCameraBinding& binding = bindings->at(index);
                if (binding.instanceId.value().empty() || binding.imageId.value().empty() ||
                    binding.worldFrame.value().empty())
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 空三 cameraBindings 不能包含空的 instance/image/frame identity");
                    }
                    return false;
                }
                if (binding.imageId != input.imageIds.at(index))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 空三相机绑定与 ImageId 顺序不一致（影像 %1）")
                                             .arg(input.images.at(static_cast<int>(index)));
                    }
                    return false;
                }
                if (!instanceIds.insert(binding.instanceId.value()).second ||
                    !imageIds.insert(binding.imageId.value()).second)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 空三包含重复的 camera instance/image identity");
                    }
                    return false;
                }
                if (binding.worldFrame.value() != "EPSG:4978")
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 空三要求 WGS84 ECEF world frame EPSG:4978，影像 %1 使用 %2")
                                             .arg(input.images.at(static_cast<int>(index)),
                                                  QString::fromStdString(binding.worldFrame.value()));
                    }
                    return false;
                }
                const auto camera = cameras.find(static_cast<ImageId>(index));
                if (camera == cameras.cend() || !camera->second)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 相机集合缺少数值索引 %1").arg(index);
                    }
                    return false;
                }
                const QString cameraFrame = QString::fromStdString(camera->second->rpcDefinition().worldFrame().value());
                if (cameraFrame != QString::fromStdString(binding.worldFrame.value()))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 数值相机 frame 与 canonical binding 不一致（影像 %1）")
                                             .arg(input.images.at(static_cast<int>(index)));
                    }
                    return false;
                }
            }
            return true;
        }

        bool canceled(const PreparedAerialTriangulationInput& input)
        {
            return input.cancelFlag && input.cancelFlag->load(std::memory_order_relaxed);
        }

        void reportProgress(const PreparedAerialTriangulationInput& input, const QString& stage, int percent)
        {
            if (input.progressFn)
            {
                input.progressFn(stage, std::clamp(percent, 0, 100));
            }
        }

        void sampleColors(const PreparedAerialTriangulationInput& input,
                          const PreparedTiePointGraph& graph,
                          std::vector<RpcPoint>* points)
        {
            QMap<ImageId, cv::Mat> imageCache;
            for (RpcPoint& point : *points)
            {
                if (point.observations.empty())
                {
                    continue;
                }
                const Observation& observation = point.observations.front();
                if (observation.imageId >= static_cast<ImageId>(input.images.size()))
                {
                    LOG_WARN(QStringLiteral("RPC 空三点云包含超出输入范围的观测索引 %1")
                                 .arg(static_cast<qulonglong>(observation.imageId)));
                    continue;
                }
                if (!imageCache.contains(observation.imageId))
                {
                    const QString path = input.images.value(static_cast<int>(observation.imageId));
                    cv::Mat decoded;
                    QString read_error;
                    try
                    {
                        decoded = xjw::common::io::readImage(
                            path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION, &read_error);
                    }
                    catch (const cv::Exception& exception)
                    {
                        read_error = QString::fromUtf8(exception.what());
                    }
                    if (decoded.empty() || decoded.type() != CV_8UC3)
                    {
                        if (read_error.isEmpty())
                        {
                            read_error = QStringLiteral("预期非空 8 位三通道影像");
                        }
                        LOG_WARN(QStringLiteral("RPC 空三点云颜色读取失败 %1: %2").arg(path, read_error));
                        decoded.release();
                    }
                    imageCache.insert(observation.imageId, decoded);
                }
                const cv::Mat& image = imageCache[observation.imageId];
                if (image.empty())
                {
                    continue;
                }
                const camera_models::rpc::ImagePoint coordinate = engine::rpcImageCoordinate(graph, observation);
                const int x = std::clamp(qRound(coordinate.sample), 0, image.cols - 1);
                const int y = std::clamp(qRound(coordinate.line), 0, image.rows - 1);
                const cv::Vec3b color = image.at<cv::Vec3b>(y, x);
                point.color = {color[2], color[1], color[0]};
            }
        }

        bool writePly(const QString& path, const std::vector<RpcPoint>& points, QString* errorMessage)
        {
            std::string error;
            const bool ok = writeSparsePly(
                xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(path)),
                points.size(),
                [&points](std::size_t index) { return SparsePlyVertex{points[index].localEnu, points[index].color}; },
                "PlaScan RPC aerial triangulation local ENU",
                &error);
            if (errorMessage)
            {
                *errorMessage = QString::fromStdString(error);
            }
            return ok;
        }

        bool pointJson(const RpcPoint& point,
                       const std::vector<camera_core::ImageId>& imageIds,
                       QJsonObject* output,
                       QString* errorMessage)
        {
            if (!output)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("RPC 点 sidecar 输出对象为空");
                }
                return false;
            }
            QJsonArray observations;
            for (const Observation& observation : point.observations)
            {
                if (observation.imageId >= static_cast<ImageId>(imageIds.size()))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("RPC 点 sidecar 观测超出 canonical ImageId 范围");
                    }
                    return false;
                }
                QJsonObject observationObject{
                    {QStringLiteral("camera_index"), static_cast<qint64>(observation.imageId)},
                    {QStringLiteral("image_id"),
                     QString::fromStdString(imageIds.at(static_cast<std::size_t>(observation.imageId)).value())},
                    {QStringLiteral("feature_idx"), static_cast<qint64>(observation.featureIdx)}};
                observations.append(observationObject);
            }
            *output = QJsonObject{
                {QStringLiteral("xyz"), QJsonArray{point.localEnu[0], point.localEnu[1], point.localEnu[2]}},
                {QStringLiteral("geodetic_wgs84"), QJsonArray{point.geodetic[0], point.geodetic[1], point.geodetic[2]}},
                {QStringLiteral("rms_reproj_px"), point.rmsPixels},
                {QStringLiteral("maximum_reproj_px"), point.maximumResidualPixels},
                {QStringLiteral("track_len"), static_cast<int>(point.observations.size())},
                {QStringLiteral("observations"), observations}};
            return true;
        }

    } // namespace

    RpcCameraInput RpcAerialTriangulationRunner::inspectInput(const PreparedAerialTriangulationInput& input)
    {
        RpcCameraInput result;
        const QJsonObject projectFiles = xjw::common::project::projectFilesRootObject(input.projectMeta);
        const bool hasCanonicalCameraKeys = projectFiles.contains(QStringLiteral("camera_instances")) ||
                                             projectFiles.contains(QStringLiteral("camera_definitions"));
        for (const QJsonValue& value : projectFiles.value(QStringLiteral("images")).toArray())
        {
            const QJsonObject image = value.toObject();
            if (image.contains(QStringLiteral("camera")) || image.contains(QStringLiteral("camera_file")))
            {
                result.errorMessage = QStringLiteral(
                    "工程影像包含已废弃的嵌入式相机字段；请先写入 canonical camera_instances。");
                return result;
            }
        }
        if (hasCanonicalCameraKeys)
        {
            const bool hasInstances = projectFiles.value(QStringLiteral("camera_instances")).isArray();
            const bool hasDefinitions = projectFiles.value(QStringLiteral("camera_definitions")).isArray();
            if (!hasInstances || !hasDefinitions)
            {
                result.errorMessage = QStringLiteral(
                    "RPC 空三要求完整的 canonical camera_definitions/camera_instances；不能从影像文件回退读取相机。");
                return result;
            }

            if (input.imageIds.size() != input.images.size())
            {
                result.errorMessage = QStringLiteral("RPC 空三要求每幅影像都有 canonical ImageId");
                return result;
            }

            const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
                projectFiles, xjw::camera_models::makeBuiltinCameraModelRegistry());
            if (!runtime.ok())
            {
                result.errorMessage = QStringLiteral("RPC 空三 canonical 相机集合无效: %1")
                                           .arg(runtime.errors.join(QStringLiteral("; ")));
                return result;
            }
            std::vector<xjw::camera_core::ImageId> imageIds(input.imageIds.begin(), input.imageIds.end());
            bool hasRpcInstance = false;
            bool hasNonRpcInstance = false;
            for (const xjw::camera_core::ImageId& imageId : imageIds)
            {
                const auto lookup = runtime.instances.forImage(imageId);
                if (!lookup.ok())
                {
                    result.errorMessage = QStringLiteral("RPC 空三无法解析影像 %1 的 canonical 相机: %2")
                                               .arg(QString::fromStdString(imageId.value()),
                                                    QString::fromStdString(lookup.error));
                    result.status = RpcCameraInputStatus::Mixed;
                    return result;
                }
                if (std::dynamic_pointer_cast<const xjw::camera_models::rpc::RpcInstance>(lookup.instance))
                {
                    hasRpcInstance = true;
                }
                else
                {
                    hasNonRpcInstance = true;
                }
            }
            if (!hasRpcInstance)
            {
                // The RPC detector is only a workflow selector.  A canonical
                // pinhole or pushbroom collection must continue to its own
                // planner instead of being reported as a mixed RPC failure.
                result.status = RpcCameraInputStatus::None;
                return result;
            }
            if (hasNonRpcInstance)
            {
                result.status = RpcCameraInputStatus::Mixed;
                result.errorMessage = QStringLiteral("RPC 空三不能混用 RPC 与非 RPC canonical 相机");
                return result;
            }
            const auto plan = runtime.planOperationForImages(
                imageIds, xjw::camera_core::CameraOperation::RpcAerialTriangulation);
            if (!plan.ok())
            {
                result.errorMessage = QStringLiteral("RPC 空三相机能力校验失败：%1")
                                           .arg(QString::fromStdString(plan.failureMessage()));
                result.status = RpcCameraInputStatus::Mixed;
                return result;
            }

            result.cameras.clear();
            for (std::size_t index = 0; index < imageIds.size(); ++index)
            {
                const auto lookup = runtime.instances.forImage(imageIds.at(index));
                const auto rpcInstance = lookup.ok()
                                              ? std::dynamic_pointer_cast<const xjw::camera_models::rpc::RpcInstance>(
                                                    lookup.instance)
                                              : nullptr;
                if (!rpcInstance)
                {
                    result.cameras.clear();
                    result.errorMessage = QStringLiteral("RPC 空三无法解析影像 %1 的 canonical RPC 相机")
                                               .arg(input.images.at(static_cast<int>(index)));
                    result.status = RpcCameraInputStatus::Mixed;
                    return result;
                }
                result.cameras.emplace(static_cast<ImageId>(index), std::move(rpcInstance));
            }
            result.status = RpcCameraInputStatus::Complete;
            return result;
        }

        // A standalone RPC input has no project camera graph yet.  Reading
        // vendor raster metadata is an explicit import boundary; it is not a
        // fallback once a canonical project graph exists.
        QStringList nonRpcImages;

        for (int index = 0; index < input.images.size(); ++index)
        {
            const QString& imagePath = input.images.at(index);
            const std::string suffix = std::to_string(index);
            const camera_core::ImageId imageId = input.imageIds.size() == static_cast<std::size_t>(input.images.size())
                                                     ? input.imageIds.at(static_cast<std::size_t>(index))
                                                     : camera_core::ImageId("rpc-import-image-" + suffix);
            std::string importError;
            auto camera = camera_models::rpc::importRpcRasterInstance(
                common::io::toUtf8Path(imagePath),
                camera_core::CameraDefinitionId("rpc-import-definition-" + suffix),
                camera_core::CameraInstanceId("rpc-import-instance-" + suffix),
                imageId,
                xjw::coordinate_system::CoordinateFrameId("EPSG:4978"),
                &importError);

            if (camera)
            {
                result.cameras.emplace(static_cast<ImageId>(index), std::move(camera));
            }
            else
            {
                nonRpcImages.append(QFileInfo(imagePath).fileName());
            }
        }

        if (result.cameras.empty())
        {
            result.status = RpcCameraInputStatus::None;
            return result;
        }
        if (result.cameras.size() == input.images.size())
        {
            result.status = RpcCameraInputStatus::Complete;
            return result;
        }

        result.status = RpcCameraInputStatus::Mixed;
        result.errorMessage = QStringLiteral("RPC 空三要求本次选择的全部影像都具有有效 RPC00B；缺失或非 RPC 影像: %1")
                                  .arg(nonRpcImages.join(QStringLiteral("、")));
        return result;
    }

    AerialTriangulationReconstructionResult
    RpcAerialTriangulationRunner::run(const PreparedAerialTriangulationInput& input,
                                      const std::map<
                                          ImageId,
                                          std::shared_ptr<const camera_models::rpc::RpcInstance>>& cameras) const
    {
        AerialTriangulationReconstructionResult result;
        if (input.images.size() < 2 || cameras.size() != input.images.size())
        {
            result.errorMessage = QStringLiteral("RPC 空三至少需要两张且每张都具有有效 RPC00B 的影像");
            result.summary = result.errorMessage;
            return result;
        }
        if (input.imageIds.size() != input.images.size())
        {
            result.errorMessage = QStringLiteral("RPC 空三写回要求每幅影像都有 canonical ImageId");
            result.summary = result.errorMessage;
            return result;
        }
        std::vector<SolverCameraBinding> updateBindings;
        if (!resolveRpcCameraBindings(input, cameras, &updateBindings, &result.errorMessage))
        {
            result.summary = result.errorMessage;
            return result;
        }
        std::string directory_error;
        if (input.outputDir.trimmed().isEmpty() ||
            !xjw::common::file::ensureDirectory(
                xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(input.outputDir)), &directory_error))
        {
            result.errorMessage = directory_error.empty()
                                      ? QStringLiteral("无法创建 RPC 空三输出目录: %1").arg(input.outputDir)
                                      : QString::fromStdString(directory_error);
            result.summary = result.errorMessage;
            return result;
        }

        reportProgress(input, QStringLiteral("读取 RPC 连接点图"), 0);
        std::shared_ptr<const PreparedTiePointGraph> graph = input.preparedTiePointGraph;
        if (!graph)
        {
            auto loadedGraph = std::make_shared<PreparedTiePointGraph>();
            if (!SfmAttemptRunner::readTiePointGraph(
                    input.tiePointPath, input.images, loadedGraph.get(), &result.errorMessage))
            {
                result.summary = result.errorMessage;
                return result;
            }
            graph = std::move(loadedGraph);
        }
        if (canceled(input))
        {
            result.errorMessage = QStringLiteral("用户取消");
            result.summary = result.errorMessage;
            return result;
        }

        const double maximumRmsPixels = input.quality >= 3 ? 1.5 : (input.quality >= 2 ? 2.0 : 3.0);
        engine::RpcInput numericalInput;
        numericalInput.graph = graph;
        numericalInput.maximumRmsPixels = maximumRmsPixels;
        numericalInput.cancelFlag = input.cancelFlag;
        for (auto camera = cameras.cbegin(); camera != cameras.cend(); ++camera)
        {
            numericalInput.cameras.emplace(camera->first, camera->second);
        }
        numericalInput.progressFn = [&input](const std::string& stage, int percent)
        { reportProgress(input, QString::fromStdString(stage), percent); };
        engine::RpcResult numericalResult = engine::runRpc(numericalInput);
        if (!numericalResult.success)
        {
            result.errorMessage = QString::fromStdString(numericalResult.error);
            result.summary = result.errorMessage;
            return result;
        }
        auto& points = numericalResult.points;
        const auto& cameraResiduals = numericalResult.cameraResiduals;
        const auto& origin = numericalResult.origin;
        if (points.empty())
        {
            result.errorMessage = QStringLiteral("RPC 空三没有生成可发布的稀疏点");
            result.summary = result.errorMessage;
            return result;
        }
        reportProgress(input, QStringLiteral("建立局部 ENU 坐标并写出稀疏云"), 85);
        sampleColors(input, *graph, &points);
        const QString plyPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse.ply"));
        if (!writePly(plyPath, points, &result.errorMessage))
        {
            result.summary = result.errorMessage;
            return result;
        }

        QJsonArray pointArray;
        double squaredErrorSum = 0.0;
        QSet<ImageId> registeredImages;
        for (const RpcPoint& point : points)
        {
            QJsonObject serializedPoint;
            if (!pointJson(point, input.imageIds, &serializedPoint, &result.errorMessage))
            {
                result.summary = result.errorMessage;
                return result;
            }
            pointArray.append(serializedPoint);
            squaredErrorSum += point.rmsPixels * point.rmsPixels;
            for (const Observation& observation : point.observations)
            {
                registeredImages.insert(observation.imageId);
            }
        }
        const double meanRms = std::sqrt(squaredErrorSum / points.size());
        QJsonObject quality = xjw::common::project::buildSparseQualityMetadata(
            pointArray,
            registeredImages.size(),
            true,
            xjw::common::project::kSparseResultKindSfmSparseReconstruction,
            QString(),
            QString(),
            input.images.size());
        quality.insert(QStringLiteral("camera_model"), QStringLiteral("rpc00b"));
        quality.insert(QStringLiteral("absolute_sensor_model"), true);
        quality.insert(QStringLiteral("coordinate_frame"), QStringLiteral("local_enu_wgs84"));
        quality.insert(QStringLiteral("rpc_point_adjustment"), true);
        quality.insert(QStringLiteral("rpc_bias_adjustment"), false);
        quality.insert(
            QStringLiteral("quality_gate"),
            QJsonObject{{QStringLiteral("acceptable_for_mvs"), false},
                        {QStringLiteral("warnings"), QJsonArray{QStringLiteral("rpc_requires_rpc_dense_workflow")}},
                        {QStringLiteral("maximum_rpc_rms_px"), maximumRmsPixels}});

        QJsonArray perCamera;
        for (ImageId imageId : registeredImages)
        {
            if (imageId >= static_cast<ImageId>(input.images.size()) ||
                imageId >= static_cast<ImageId>(input.imageIds.size()))
            {
                result.errorMessage = QStringLiteral("RPC 空三注册相机超出 canonical ImageId 范围");
                result.summary = result.errorMessage;
                return result;
            }
            const auto residualIt = cameraResiduals.find(imageId);
            if (residualIt == cameraResiduals.end())
            {
                result.errorMessage = QStringLiteral("RPC 空三缺少注册相机的残差统计");
                result.summary = result.errorMessage;
                return result;
            }
            const CameraResidualAccumulator accumulator = residualIt->second;
            const double rms = accumulator.observationCount > 0
                                   ? std::sqrt(accumulator.squaredErrorSum / accumulator.observationCount)
                                   : 0.0;
            perCamera.append(QJsonObject{{QStringLiteral("camera_index"), static_cast<qint64>(imageId)},
                                         {QStringLiteral("image_id"),
                                          QString::fromStdString(input.imageIds.at(static_cast<std::size_t>(imageId)).value())},
                                         {QStringLiteral("image_path"), input.images.value(static_cast<int>(imageId))},
                                         {QStringLiteral("camera_model"), QStringLiteral("rpc00b")},
                                         {QStringLiteral("observation_count"), accumulator.observationCount},
                                         {QStringLiteral("rms_reproj_px"), rms},
                                         {QStringLiteral("maximum_reproj_px"), accumulator.maximumResidual}});

            const auto camera = cameras.find(imageId);
            if (camera == cameras.cend() || !camera->second)
            {
                result.errorMessage = QStringLiteral("RPC 空三缺少注册相机实例");
                result.summary = result.errorMessage;
                return result;
            }
            QJsonObject cameraObject = xjw::common::project::serializeRpcInstance(*camera->second);
            cameraObject.insert(QStringLiteral("intrinsic_source"), QStringLiteral("embedded_rpc00b"));
            cameraObject.insert(QStringLiteral("pose_source"), QStringLiteral("rpc00b"));
            cameraObject.insert(QStringLiteral("adjustment_status"), QStringLiteral("rpc_fixed_model"));
            cameraObject.insert(QStringLiteral("rpc_adjustment_mode"), QStringLiteral("fixed_sensor_point_only"));
            result.cameraInstanceUpdates.push_back(
                {input.imageIds.at(static_cast<std::size_t>(imageId)),
                 updateBindings.at(static_cast<std::size_t>(imageId)).instanceId,
                 updateBindings.at(static_cast<std::size_t>(imageId)).worldFrame,
                 cameraObject});
        }

        const QJsonObject originJson{{QStringLiteral("longitude_deg"), origin[0]},
                                     {QStringLiteral("latitude_deg"), origin[1]},
                                     {QStringLiteral("ellipsoidal_height_m"), origin[2]}};
        const QString sidecarPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse_points.json"));
        QJsonObject diagnostics{
            {QStringLiteral("camera_model"), QStringLiteral("rpc00b")},
            {QStringLiteral("rpc_model_fixed"), true},
            {QStringLiteral("rpc_bias_adjustment_applied"), false},
            {QStringLiteral("rpc_point_adjustment_applied"), true},
            {QStringLiteral("input_track_count"), static_cast<qint64>(numericalResult.inputTrackCount)},
            {QStringLiteral("accepted_track_count"), static_cast<qint64>(points.size())},
            {QStringLiteral("rejected_track_count"),
             static_cast<qint64>(numericalResult.inputTrackCount - points.size())},
            {QStringLiteral("maximum_reprojection_error_px"), maximumRmsPixels},
            {QStringLiteral("local_enu_origin_wgs84"), originJson}};
        QJsonObject sidecar = xjw::common::project::mergeSparseQualityIntoRecord(
            QJsonObject{{QStringLiteral("schema"), QStringLiteral("plascan.rpc_aerial_triangulation.v1")},
                        {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")},
                        {QStringLiteral("camera_model"), QStringLiteral("rpc00b")},
                        {QStringLiteral("coordinate_frame"), QStringLiteral("local_enu_wgs84")},
                        {QStringLiteral("local_enu_origin_wgs84"), originJson},
                        {QStringLiteral("points"), pointArray},
                        {QStringLiteral("per_camera"), perCamera},
                        {QStringLiteral("sfm_diagnostics"), diagnostics}},
            quality);
        const QByteArray sidecar_bytes = QJsonDocument(sidecar).toJson(QJsonDocument::Compact);
        std::string file_error;
        const bool sidecar_written = xjw::common::file::writeFileAtomic(
            xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(sidecarPath)),
            std::string_view(sidecar_bytes.constData(), static_cast<std::size_t>(sidecar_bytes.size())),
            &file_error);
        const QString writeError = QString::fromStdString(file_error);
        if (!sidecar_written)
        {
            result.errorMessage =
                writeError.isEmpty() ? QStringLiteral("无法写入 RPC 空三质量文件: %1").arg(sidecarPath) : writeError;
            result.summary = result.errorMessage;
            return result;
        }

        result.success = true;
        result.numRegisteredImages = registeredImages.size();
        result.numPoints3D = static_cast<int>(points.size());
        result.meanReprojError = meanRms;
        result.baRmsBefore = meanRms;
        result.baRmsAfter = meanRms;
        result.baTracksTotal = static_cast<int>(numericalResult.inputTrackCount);
        result.baTracksOptimized = static_cast<int>(points.size());
        result.baTracksFiltered = static_cast<int>(numericalResult.inputTrackCount - points.size());
        result.sparseCloudPath = plyPath;
        result.qualityMetadata = quality;
        result.sfmDiagnostics = diagnostics;
        result.perCameraResiduals = perCamera;
        result.resultRecordExtra = xjw::common::project::mergeSparseQualityIntoRecord(
            QJsonObject{
                {QStringLiteral("source"), QStringLiteral("aerial_triangulation")},
                {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")},
                {QStringLiteral("camera_model"), QStringLiteral("rpc00b")},
                {QStringLiteral("absolute_sensor_model"), true},
                {QStringLiteral("coordinate_frame"), QStringLiteral("local_enu_wgs84")},
                {QStringLiteral("local_enu_origin_wgs84"), originJson},
                {QStringLiteral("files"), QJsonObject{{QStringLiteral("sparse_cloud_points_json"), sidecarPath}}},
                {QStringLiteral("sfm_diagnostics"), diagnostics}},
            quality);
        result.summary = QStringLiteral("RPC 空三成功：注册 %1/%2 张影像，%3 个地面点，RMS %4 px")
                             .arg(result.numRegisteredImages)
                             .arg(input.images.size())
                             .arg(result.numPoints3D)
                             .arg(result.meanReprojError, 0, 'f', 3);
        reportProgress(input, QStringLiteral("RPC 空三完成"), 100);
        return result;
    }

} // namespace xjw::aerial_triangulation
