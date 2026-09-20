/**
 * @file SfmAttemptRunner.cpp
 * @brief 将连接点 sidecar 和工程先验装配为一次隔离的 IncrementalSfm 试算。
 *
 * 一次 attempt 的职责包括：
 * 1. 严格读取当前影像集合对应的多视连接点图；
 * 2. 选择相机文件、可信工程内参或影像尺寸估计内参；
 * 3. 装载人工标记、控制点和比例尺先验；
 * 4. 配置增量 SfM/BA 并运行；
 * 5. 返回内存重建、诊断和待提交相机更新。
 *
 * 本类不写稀疏点云、不直接修改工程。焦距搜索因此可以并发运行多个互不污染的
 * attempt，并由上层只提交胜出结果。
 */

#include "reconstruction/SfmAttemptRunner.h"
#include "engine/PinholeEngine.h"
#include "engine/TiePointGraphReader.h"
#include "file/FileIO.h"
#include "reconstruction/CameraIntrinsicPriorSanitizer.h"
#include "reconstruction/MarkerPriorLoader.h"
#include "search/SfmSearchPolicy.h"

#include "io/PathIO.h"
#include "io/ImageExifMetadata.h"
#include "log/Logger.h"
#include "ProjectCameraIO.h"
#include "BundleAdjustAdaptiveCameraModel.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/project/CameraProjectRuntime.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"
#include "pipeline/IncrementalSfm.h"

#include <QMap>
#include <QSet>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSize>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>

namespace xjw::aerial_triangulation
{
    namespace
    {

        bool fail(const QString& message, QString* errorMessage)
        {
            if (errorMessage)
            {
                *errorMessage = message;
            }
            return false;
        }

        std::string sensorKeyForImage(const QString& imagePath)
        {
            const auto metadata = xjw::common::io::readImageExifMetadata(imagePath);
            if (!metadata.has_value())
            {
                return {};
            }
            const QString make = metadata->make.trimmed().toLower();
            const QString model = metadata->model.trimmed().toLower();
            if (make.isEmpty() && model.isEmpty())
            {
                return {};
            }
            QString key = make + QLatin1Char('/') + model;
            if (metadata->focalLengthMm.has_value() && std::isfinite(*metadata->focalLengthMm) &&
                *metadata->focalLengthMm > 0.0)
            {
                key += QStringLiteral("/f=%1mm").arg(*metadata->focalLengthMm, 0, 'f', 3);
            }
            return key.toUtf8().toStdString();
        }

        /// 用项目的路径 token 规则将持久化影像路径映射到本次输入索引。
        int selectedImageIndex(const QString& path, const QStringList& selectedImages)
        {
            for (int index = 0; index < selectedImages.size(); ++index)
            {
                if (xjw::common::project::pathTokenMatchesImage(path, selectedImages.at(index)))
                {
                    return index;
                }
            }
            return -1;
        }

        /// 质量等级对初始化强度和 BA 调度频率的映射。
        struct SfmQualityPreset
        {
            int initMinMatches = 25;
            int initMinInliers = 5;
            int localBaInterval = 3;
            int globalBaInterval = 10;
        };

        SfmQualityPreset presetForQuality(int quality)
        {
            switch (std::clamp(quality, 0, 3))
            {
            case 0:
                return {15, 5, 5, 15};
            case 1:
                return {20, 5, 4, 12};
            case 2:
                return {25, 5, 3, 10};
            case 3:
            default:
                return {30, 5, 3, 10};
            }
        }

        bool cameraMetadataHasUsablePose(const QJsonObject& cameraObject)
        {
            return !cameraObject.isEmpty() &&
                   !cameraObject.value(QStringLiteral("pose_initialized_as_identity")).toBool(false);
        }

        bool bindInputCameraIdentity(const PreparedAerialTriangulationInput& input,
                                     int imageIndex,
                                     const QString& imagePath,
                                     xjw::camera_models::frame_pinhole::FramePinholeNumericState* camera,
                                     QString* errorMessage)
        {
            if (!camera)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("数值相机输出为空");
                }
                return false;
            }

            if (input.cameraBindings.empty())
            {
                if (!input.cameraReferencePosePriors.empty())
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("外部相机姿态参考要求调用方提供与 images 对齐的显式 cameraBindings");
                    }
                    return false;
                }
                return true;
            }
            if (input.cameraBindings.size() != static_cast<std::size_t>(input.images.size()))
            {
                if (errorMessage)
                {
                    *errorMessage =
                        QStringLiteral("cameraBindings 必须与 images 一一对应；影像 %1 无法安全绑定").arg(imagePath);
                }
                return false;
            }

            const SolverCameraBinding& binding = input.cameraBindings.at(static_cast<std::size_t>(imageIndex));

            std::string bindError;
            if (!camera->hasBoundIdentity() &&
                !camera->bindIdentity(binding.instanceId, binding.imageId, binding.worldFrame, &bindError))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法绑定影像 %1 的数值相机身份/frame: %2")
                                        .arg(imagePath, QString::fromStdString(bindError));
                }
                return false;
            }
            if (camera->hasBoundIdentity() &&
                (camera->instanceId() != binding.instanceId || camera->imageId() != binding.imageId ||
                 camera->worldFrame() != binding.worldFrame))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("影像 %1 的数值相机身份/frame 与显式输入不一致").arg(imagePath);
                }
                return false;
            }
            return true;
        }

        bool validateInputCameraBindings(const PreparedAerialTriangulationInput& input, QString* errorMessage)
        {
            if (input.cameraBindings.empty())
            {
                if (!input.cameraReferencePosePriors.empty())
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("外部相机姿态参考要求调用方提供与 images 对齐的显式 cameraBindings");
                    }
                    return false;
                }
                return true;
            }
            if (input.cameraBindings.size() != static_cast<std::size_t>(input.images.size()))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("cameraBindings 必须与 images 一一对应");
                }
                return false;
            }
            if (!input.imageIds.empty() && input.imageIds.size() != static_cast<std::size_t>(input.images.size()))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("imageIds 必须与 images 一一对应");
                }
                return false;
            }

            std::unordered_set<std::string> instanceIds;
            std::unordered_set<std::string> imageIds;
            instanceIds.reserve(input.cameraBindings.size());
            imageIds.reserve(input.cameraBindings.size());
            const xjw::coordinate_system::CoordinateFrameId& commonFrame = input.cameraBindings.front().worldFrame;
            for (std::size_t index = 0; index < input.cameraBindings.size(); ++index)
            {
                const SolverCameraBinding& binding = input.cameraBindings.at(index);
                if (binding.instanceId.value().empty() || binding.imageId.value().empty() ||
                    binding.worldFrame.value().empty())
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("cameraBindings 不能包含空的 instance/image/frame identity");
                    }
                    return false;
                }
                if (!instanceIds.insert(binding.instanceId.value()).second ||
                    !imageIds.insert(binding.imageId.value()).second)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("cameraBindings 包含重复的 camera instance/image identity");
                    }
                    return false;
                }
                if (!input.imageIds.empty() && binding.imageId != input.imageIds.at(index))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("cameraBindings[%1] 与 imageIds 顺序不一致")
                                            .arg(static_cast<qulonglong>(index));
                    }
                    return false;
                }
                if (binding.worldFrame != commonFrame)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("cameraBindings 混用 world frame；必须先显式归一化");
                    }
                    return false;
                }
            }
            return true;
        }

        struct CanonicalBindingResolution
        {
            bool collectionPresent = false;
            bool complete = false;
            std::vector<SolverCameraBinding> bindings;
            std::shared_ptr<const xjw::camera_project::CameraProjectRuntimeResult> runtime;
            QString error;
        };

        /**
         * Resolve identities from the canonical project camera collection.
         *
         * The image path is used only to select the already persisted image
         * entry.  The returned identity and frame always come from the
         * decoded camera instance; no identifier is derived from a path.
         */
        CanonicalBindingResolution resolveCanonicalBindings(const PreparedAerialTriangulationInput& input)
        {
            CanonicalBindingResolution result;
            if (input.projectMeta.isEmpty())
            {
                return result;
            }

            const QJsonObject projectFiles = xjw::common::project::projectFilesRootObject(input.projectMeta);
            if (!projectFiles.value(QStringLiteral("camera_instances")).isArray() ||
                !projectFiles.value(QStringLiteral("camera_definitions")).isArray())
            {
                return result;
            }
            result.collectionPresent = true;
            if (!input.imageIds.empty() && input.imageIds.size() != static_cast<std::size_t>(input.images.size()))
            {
                result.error = QStringLiteral("imageIds 必须与 images 一一对应");
                return result;
            }

            auto runtime = std::make_shared<xjw::camera_project::CameraProjectRuntimeResult>(
                xjw::camera_project::CameraProjectRuntime::load(projectFiles,
                                                                xjw::camera_models::makeBuiltinCameraModelRegistry()));
            if (!runtime->ok())
            {
                result.error = QStringLiteral("canonical camera collection is invalid: %1")
                                   .arg(runtime->errors.join(QStringLiteral("; ")));
                return result;
            }
            result.runtime = runtime;

            QMap<QString, QJsonObject> imageMetadata;
            for (const QJsonValue& value : xjw::common::project::projectImageEntries(input.projectMeta))
            {
                const QJsonObject image = value.toObject();
                const QString path =
                    xjw::common::project::normalizePath(image.value(QStringLiteral("path")).toString());
                if (path.isEmpty())
                {
                    continue;
                }
                if (imageMetadata.contains(path))
                {
                    result.error = QStringLiteral("canonical image path is ambiguous: %1").arg(path);
                    return result;
                }
                imageMetadata.insert(path, image);
            }
            result.bindings.reserve(static_cast<std::size_t>(input.images.size()));
            for (const QString& imagePath : input.images)
            {
                const auto imageIt = imageMetadata.constFind(xjw::common::project::normalizePath(imagePath));
                if (imageIt == imageMetadata.constEnd())
                {
                    result.error = QStringLiteral("canonical image entry is missing for %1").arg(imagePath);
                    result.bindings.clear();
                    return result;
                }
                const QString imageUuid = imageIt.value().value(QStringLiteral("image_uuid")).toString().trimmed();
                if (imageUuid.isEmpty())
                {
                    result.error = QStringLiteral("canonical image entry has no image_uuid for %1").arg(imagePath);
                    result.bindings.clear();
                    return result;
                }
                if (!input.imageIds.empty() &&
                    imageUuid != QString::fromStdString(input.imageIds.at(result.bindings.size()).value()))
                {
                    result.error = QStringLiteral("canonical image_uuid 与输入 ImageId 不一致 for %1").arg(imagePath);
                    result.bindings.clear();
                    return result;
                }

                const auto lookup = runtime->instances.forImage(camera_core::ImageId(imageUuid.toStdString()));
                if (!lookup.ok())
                {
                    result.error = QStringLiteral("canonical camera instance is missing for %1: %2")
                                       .arg(imagePath, QString::fromStdString(lookup.error));
                    result.bindings.clear();
                    return result;
                }
                result.bindings.push_back({lookup.instance->instanceId(),
                                           lookup.instance->imageId(),
                                           lookup.instance->definition().worldFrame()});
            }
            result.complete = result.bindings.size() == static_cast<std::size_t>(input.images.size());
            if (!result.complete)
            {
                result.bindings.clear();
                result.error = QStringLiteral("canonical camera bindings do not cover every selected image");
            }
            return result;
        }

        bool bindingsMatchCanonical(const std::vector<SolverCameraBinding>& actual,
                                    const std::vector<SolverCameraBinding>& expected,
                                    QString* errorMessage)
        {
            if (actual.size() != expected.size())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("显式 cameraBindings 与 canonical camera_instances 数量不一致");
                }
                return false;
            }
            for (std::size_t index = 0; index < actual.size(); ++index)
            {
                if (actual[index].instanceId != expected[index].instanceId ||
                    actual[index].imageId != expected[index].imageId ||
                    actual[index].worldFrame != expected[index].worldFrame)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("显式 cameraBindings[%1] 与 canonical camera_instances "
                                                       "不一致；禁止按路径或序号替换身份")
                                            .arg(static_cast<qulonglong>(index));
                    }
                    return false;
                }
            }
            return true;
        }

        xjw::camera_models::frame_pinhole::FramePinholeNumericState
        cameraWithIdentityPose(xjw::camera_models::frame_pinhole::FramePinholeNumericState camera)
        {
            camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0});
            return camera;
        }

        /**
         * @brief 根据工作流输入统一配置 IncrementalSfm 和 BA。
         *
         * GPU/CPU 仅影响 BA 后端选择；特征与匹配已经在上游完成。自适应相机模型拟合
         * 会声明共享 Brown 内参上限，并在最终全局 BA 前逐项筛选；完整已知位姿保持输入内参稳定。
         */
        void configureSfmOptions(const PreparedAerialTriangulationInput& input,
                                 const std::shared_ptr<std::atomic<int>>& registeredProgress,
                                 IncrementalSfmOptions* options)
        {
            const SfmQualityPreset preset = presetForQuality(input.quality);
            options->initMinNumMatches = preset.initMinMatches;
            options->initMinNumInliers = preset.initMinInliers;
            options->localBAInterval = preset.localBaInterval;
            options->globalBAInterval = preset.globalBaInterval;
            options->executionProfile = input.coarseFocalEvaluation ? SfmExecutionProfile::CoarseEvaluation
                                                                    : SfmExecutionProfile::FullRefinement;
            options->maxRegisteredImages = input.maxRegisteredImages;
            if (input.maxTracksPerImage >= 0)
            {
                options->maxTracksPerImage = input.maxTracksPerImage;
            }
            if (input.maxTracksPerGridCell >= 0)
            {
                options->maxTracksPerGridCell = input.maxTracksPerGridCell;
            }
            if (input.trackThinningGridColumns > 0)
            {
                options->trackThinningGridColumns = input.trackThinningGridColumns;
            }
            if (input.trackThinningGridRows > 0)
            {
                options->trackThinningGridRows = input.trackThinningGridRows;
            }

            if (options->executionProfile == SfmExecutionProfile::CoarseEvaluation)
            {
                // 焦距候选粗筛不执行正式的逐批完整块 BA；数百图候选继续放宽局部/全局间隔，
                // 胜出焦距正式重放时再进入参考 20/10 次、五轮结构刷新调度。
                const SfmBaSchedule baSchedule = resolveSfmBaSchedule(input.images.size(),
                                                                      options->localBAInterval,
                                                                      options->localBANumImages,
                                                                      options->globalBAInterval);
                options->localBAInterval = baSchedule.localInterval;
                options->localBANumImages = baSchedule.localWindowImages;
                options->globalBAInterval = baSchedule.globalInterval;
            }
            options->baOptions.cancelFlag = input.cancelFlag;
            options->baOptions.numThreads = resolveSfmThreadBudget(input.threads);
            options->baOptions.progressCallback =
                [progress = input.progressFn, cancelFlag = input.cancelFlag, registeredProgress](
                    int currentIteration, int maxIterations, double avgRms, int validPoints)
            {
                if (cancelFlag && cancelFlag->load())
                {
                    return false;
                }
                if (progress)
                {
                    QString stage = QStringLiteral("光束法平差：迭代 %1/%2，RMS %3")
                                        .arg(currentIteration)
                                        .arg(maxIterations)
                                        .arg(avgRms, 0, 'f', 4);
                    if (validPoints > 0)
                    {
                        stage += QStringLiteral("，有效点 %1").arg(validPoints);
                    }
                    // BA 会在多次局部/全局阶段重复进入，百分比沿用当前已注册影像进度，
                    // 只更新真实迭代信息，避免整体工作流进度条来回跳动。
                    progress(stage, registeredProgress ? registeredProgress->load() : 0);
                }
                return true;
            };

            if (input.useInitialPairHint)
            {
                options->autoSelectInitPair = false;
                options->initImageId1 = input.initialImageId1;
                options->initImageId2 = input.initialImageId2;
            }

            const bool cpuOnly = input.device.trimmed().compare(QStringLiteral("cpu"), Qt::CaseInsensitive) == 0;
            if (cpuOnly)
            {
                options->baOptions.backend = BABackend::PlaMatrixCpu;
            }
            else
            {
                options->baOptions.backend = BABackend::Auto;
                options->baOptions.enableBackendQualityGate = true;
                options->baOptions.maxAcceptedRmsGrowth = 1.25;
                options->baOptions.minAcceptedValidTrackRatio = 0.60;
                options->baOptions.allowBackendFallback = true;
            }

            if (input.quality >= 2)
            {
                options->filterMaxReprojError = 1.5;
                options->filterMinTriAngle = 2.0;
                options->iterativeBARounds = 4;
            }
            options->baOptions.filterMaxReprojError = options->filterMaxReprojError;

            // 普通“对齐照片”不向相机中心附加共面/航带形状先验。穹顶误差由标定组、
            // 分阶段内参释放、参考 gauge 和异常点过滤共同处理；显式专业流程仍可直接
            // 使用 BA 层的 cameraPlaneConstraint。
            options->preserveCameraLayerDuringSelfCalibration = false;
            options->baOptions.hasTrustedSharedFocalPrior = input.hasTrustedFocalPrior;

            options->useSequencePoseRecovery = input.useSequencePoseRecovery;
            options->enforceSequencePoseConsistency = input.enforceSequencePoseConsistency;
            options->sequenceLoopClosure = input.sequenceLoopClosure;
            // 最终五轮方差精化后不自动启动另一套摘除/重注册/最终 BA；该能力保留为
            // IncrementalSfm 的显式非参考扩展，但正式“对齐照片”路径保持一次最终周期。
            options->repairParallelAerialPoseOutliers = false;
            if (input.adaptiveCameraModelFitting)
            {
                options->adaptiveCameraModelFitting = true;
                // 参考 MetaShape 用户手册公开的“声明最大模型、按数据证据自适应选择参数”概念。
                // 这里只声明 PlaScan 允许的最大 Brown 模型，实际自由度由独立评分选择。
                options->baOptions.refineSharedFocalLength = true;
                options->baOptions.refineSharedFocalAspectRatio = true;
                options->baOptions.refineSharedPrincipalPoint = true;
                options->baOptions.refineSharedRadialDistortion = true;
                options->baOptions.refineSharedHighOrderDistortion = true;
                if (input.hasTrustedFocalPrior)
                {
                    // 35mm 等效焦距存在取整、裁切和对焦误差，只作为弱先验。边界覆盖常见
                    // EXIF 量化误差，同时显著窄于完全无标定搜索。
                    options->baOptions.minSharedFocalScale = 0.94;
                    options->baOptions.maxSharedFocalScale = 1.06;
                }
                else
                {
                    options->baOptions.minSharedFocalScale = 0.90;
                    options->baOptions.maxSharedFocalScale = 1.10;
                }
                if (cpuOnly)
                {
                    options->baOptions.backend = BABackend::PlaMatrixCpu;
                }
            }
        }

    } // namespace

    QSize SfmAttemptRunner::resolveInputImageSize(const QString& imagePath)
    {
        return xjw::common::io::readImageSize(imagePath);
    }

    SfmAttemptExecutionResult SfmAttemptRunner::run(const PreparedAerialTriangulationInput& input) const
    {
        SfmAttemptExecutionResult execution;
        PreparedAerialTriangulationInput bindingInput = input;
        const CanonicalBindingResolution canonicalBindings = resolveCanonicalBindings(input);
        const bool autoBoundFromCanonical = input.cameraBindings.empty() && canonicalBindings.complete;
        if (canonicalBindings.complete)
        {
            if (bindingInput.cameraBindings.empty())
            {
                bindingInput.cameraBindings = canonicalBindings.bindings;
            }
            else if (!bindingsMatchCanonical(
                         bindingInput.cameraBindings, canonicalBindings.bindings, &execution.result.errorMessage))
            {
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
        }
        else if (canonicalBindings.collectionPresent && !canonicalBindings.error.isEmpty())
        {
            execution.result.errorMessage =
                QStringLiteral("无法安全解析 canonical 相机绑定：%1").arg(canonicalBindings.error);
            execution.result.summary = execution.result.errorMessage;
            return execution;
        }

        if (!validateInputCameraBindings(bindingInput, &execution.result.errorMessage))
        {
            execution.result.summary = execution.result.errorMessage;
            return execution;
        }

        std::vector<camera_core::ImageId> selectedImageIds;
        if (!bindingInput.imageIds.empty())
        {
            if (bindingInput.imageIds.size() != static_cast<std::size_t>(bindingInput.images.size()))
            {
                execution.result.errorMessage = QStringLiteral("imageIds 必须与 images 一一对应");
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
            selectedImageIds = bindingInput.imageIds;
        }
        else if (bindingInput.cameraBindings.size() == static_cast<std::size_t>(bindingInput.images.size()))
        {
            selectedImageIds.reserve(bindingInput.cameraBindings.size());
            for (const SolverCameraBinding& binding : bindingInput.cameraBindings)
            {
                selectedImageIds.push_back(binding.imageId);
            }
        }

        // canonical camera instances are the only source of model identity.  Once the
        // collection covers this selection, static SfM must pass the model-agnostic
        // capability plan before any legacy pinhole numeric state can be constructed.
        // RPC/line-scan inputs therefore fail here instead of being silently replaced
        // by an estimated focal-length pinhole camera below.
        if (canonicalBindings.complete)
        {
            if (!canonicalBindings.runtime)
            {
                execution.result.errorMessage = QStringLiteral("canonical camera runtime is unavailable");
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
            const camera_core::CameraOperationPlan plan = canonicalBindings.runtime->planOperationForImages(
                selectedImageIds, camera_core::CameraOperation::StaticSfM);
            if (!plan.ok())
            {
                execution.result.errorMessage = QString::fromStdString(plan.failureMessage());
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
        }
        std::optional<Logger::ScopedThreadMinimumLevel> coarseLogFilter;
        if (input.coarseFocalEvaluation)
        {
            // 焦距候选会并行产生大量逐相机 INFO/WARN。PnP 失败在候选阶段只是评分
            // 证据，不是生产故障；统一由 Pipeline 汇总，正式重建仍输出完整 WARN。
            coarseLogFilter.emplace(Logger::Error);
        }

        // 阶段 1：把持久化多视轨迹转换为 SfM 需要的每影像关键点和 pairwise matches。
        if (input.preparedTiePointGraph)
        {
            execution.graph = input.preparedTiePointGraph;
        }
        else
        {
            auto graph = std::make_shared<PreparedTiePointGraph>();
            if (!readTiePointGraph(input.tiePointPath, input.images, graph.get(), &execution.result.errorMessage))
            {
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
            execution.graph = std::move(graph);
        }
        if (!execution.graph)
        {
            execution.result.errorMessage = QStringLiteral("连接点图未准备");
            execution.result.summary = execution.result.errorMessage;
            return execution;
        }
        const PreparedTiePointGraph& graph = *execution.graph;

        if (input.cancelFlag && input.cancelFlag->load())
        {
            execution.result.errorMessage = QStringLiteral("用户取消");
            execution.result.summary = execution.result.errorMessage;
            return execution;
        }

        IncrementalSfmOptions sfmOptions;
        const auto registeredProgress = std::make_shared<std::atomic<int>>(0);
        configureSfmOptions(input, registeredProgress, &sfmOptions);
        sfmOptions.cameraReferencePosePriors = input.cameraReferencePosePriors;

        // 阶段 2：相机来源优先级为完整相机文件、可信工程相机、影像尺寸估算。
        const bool hasCompleteCameraFiles =
            input.cameraPaths.size() == input.images.size() && !input.images.isEmpty() &&
            std::all_of(input.cameraPaths.cbegin(),
                        input.cameraPaths.cend(),
                        [](const QString& path)
                        {
                            if (path.trimmed().isEmpty() || !QFileInfo::exists(path))
                            {
                                return false;
                            }
                            xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
                            return xjw::common::project::loadFramePinholeNumericStateFromFile(path, &camera);
                        });

        CameraIntrinsicsByImageId projectCameraByImageId;
        FramePinholeStatesByImageId canonicalPinholeByImageId;
        QSet<QString> projectPoseImageIds;
        std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> canonicalPinholeStates;
        if (canonicalBindings.complete)
        {
            std::string conversionError;
            if (!canonicalBindings.runtime->framePinholeStatesForImages(
                    selectedImageIds, &canonicalPinholeStates, &conversionError))
            {
                execution.result.errorMessage = QStringLiteral("canonical static SfM camera conversion failed: %1")
                                                    .arg(QString::fromStdString(conversionError));
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
            canonicalPinholeByImageId.reserve(canonicalPinholeStates.size());
            for (std::size_t index = 0; index < canonicalPinholeStates.size(); ++index)
            {
                canonicalPinholeByImageId.emplace(selectedImageIds.at(index), canonicalPinholeStates.at(index));
            }
        }
        const bool projectCameraIdentityAvailable =
            selectedImageIds.size() == static_cast<std::size_t>(bindingInput.images.size());
        int rejectedProjectIntrinsicCount = 0;
        if ((input.useProjectCameraIntrinsics || input.useProjectCameraPoses) && !input.projectMeta.isEmpty())
        {
            if (!projectCameraIdentityAvailable)
            {
                execution.result.errorMessage =
                    QStringLiteral("使用工程相机先验时必须提供与影像对齐的 canonical ImageId 或 cameraBindings");
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
            for (const QJsonValue& imageValue : xjw::common::project::projectImageEntries(input.projectMeta))
            {
                const QJsonObject image = imageValue.toObject();
                const QString imageIdText = image.value(QStringLiteral("image_uuid")).toString().trimmed();
                if (imageIdText.isEmpty())
                {
                    continue;
                }
                try
                {
                    const camera_core::ImageId imageId = camera_core::ImageId(imageIdText.toStdString());
                    const QJsonObject cameraObject =
                        xjw::common::project::projectCameraModelParameters(input.projectMeta, image);
                    xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
                    bool validCamera = false;
                    if (canonicalBindings.complete)
                    {
                        validCamera = std::find(selectedImageIds.cbegin(), selectedImageIds.cend(), imageId) !=
                                      selectedImageIds.cend();
                    }
                    else
                    {
                        validCamera = !cameraObject.isEmpty() &&
                                      xjw::common::project::decodeFramePinholeNumericState(cameraObject, &camera) &&
                                      camera.isValid();
                    }
                    if (validCamera)
                    {
                        const bool trustedIntrinsic = isTrustedProjectCameraIntrinsic(cameraObject);
                        if (input.useProjectCameraPoses || trustedIntrinsic)
                        {
                            projectCameraByImageId.emplace(imageId, cameraObject);
                        }
                        else
                        {
                            ++rejectedProjectIntrinsicCount;
                        }
                        if (input.useProjectCameraPoses && cameraMetadataHasUsablePose(cameraObject))
                        {
                            projectPoseImageIds.insert(imageIdText);
                        }
                    }
                }
                catch (const std::exception& exception)
                {
                    Q_UNUSED(exception);
                    continue;
                }
            }
        }

        // 只有所有影像都具备真实外参时才进入 known-pose 路径，禁止混合已知/未知位姿。
        bool hasCompleteProjectPoseCameras = !input.images.isEmpty();
        if (!projectCameraIdentityAvailable)
        {
            hasCompleteProjectPoseCameras = false;
        }
        for (const camera_core::ImageId& imageId : selectedImageIds)
        {
            if (!projectPoseImageIds.contains(QString::fromStdString(imageId.value())))
            {
                hasCompleteProjectPoseCameras = false;
                break;
            }
        }

        // 重置当前对齐时只应复用可信内参。工程文件可能保存过上次失败 SfM 的自标定结果，
        // 少数严重错误焦距会把单张相机中心吸附到模型附近，且不会明显拉高全局 RMS。
        CameraIntrinsicPriorSanitizationResult intrinsicSanitization;
        if (!hasCompleteCameraFiles && input.useProjectCameraIntrinsics && !input.useProjectCameraPoses)
        {
            if (canonicalBindings.complete)
            {
                intrinsicSanitization =
                    sanitizeProjectCameraIntrinsicPriors(selectedImageIds, &canonicalPinholeByImageId);
            }
            else
            {
                intrinsicSanitization = sanitizeProjectCameraIntrinsicPriors(selectedImageIds, &projectCameraByImageId);
            }
        }
        sfmOptions.useKnownCameraPoses = hasCompleteCameraFiles || hasCompleteProjectPoseCameras;
        if (sfmOptions.useKnownCameraPoses)
        {
            // 完整已知位姿路径复用输入相机标定，不再运行自由网络自标定。否则 adaptive
            // 已打开的径向/高阶参数会与这里冻结的焦距形成不合法且不可观的混合模型。
            sfmOptions.adaptiveCameraModelFitting = false;
            sfmOptions.baOptions.refineSharedFocalLength = false;
            sfmOptions.baOptions.refineSharedFocalAspectRatio = false;
            sfmOptions.baOptions.refineSharedPrincipalPoint = false;
            sfmOptions.baOptions.refineSharedRadialDistortion = false;
            sfmOptions.baOptions.refineSharedHighOrderDistortion = false;
            sfmOptions.baOptions.useSharedIntrinsicParameterMask = false;
            sfmOptions.refineKnownCameraPoseWithSoftPrior = !input.lockInputCameraPoses;
        }
        else
        {
            sfmOptions.pnpOptions.allowRelaxedInlierRatio = true;
            sfmOptions.pnpOptions.minInlierRatio = std::max(sfmOptions.pnpOptions.minInlierRatio, 0.25);
            sfmOptions.pnpOptions.minNumInliers = std::max(sfmOptions.pnpOptions.minNumInliers, 12);
            sfmOptions.pnpOptions.relaxedMinInlierRatio = std::min(sfmOptions.pnpOptions.relaxedMinInlierRatio, 0.05);
            sfmOptions.pnpOptions.relaxedMinNumInliers = std::max(sfmOptions.pnpOptions.relaxedMinNumInliers, 24);
            sfmOptions.pnpOptions.allowStrictSmallSupportRecovery = true;
            sfmOptions.pnpOptions.strictSmallSupportMinInliers = 8;
            sfmOptions.pnpOptions.strictSmallSupportMinInlierRatio = 0.80;
            sfmOptions.pnpOptions.strictSmallSupportMinGridCells = 3;
        }

        // 阶段 3：创建影像节点。重置对齐时工程相机只保留内参并将外参置为单位位姿。
        engine::PinholeInput numericalInput;
        numericalInput.options = sfmOptions;
        numericalInput.graph = execution.graph;
        numericalInput.cancelFlag = input.cancelFlag;
        QMap<QString, ImageId> imageIdByCanonicalId;
        for (int index = 0; index < input.images.size(); ++index)
        {
            const ImageId imageId = static_cast<ImageId>(index);
            const QString& imagePath = input.images.at(index);
            const std::string sensorKey = sensorKeyForImage(imagePath);
            if (index < static_cast<int>(selectedImageIds.size()))
            {
                imageIdByCanonicalId.insert(
                    QString::fromStdString(selectedImageIds.at(static_cast<std::size_t>(index)).value()), imageId);
            }
            const auto nativePath = xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(imagePath));

            if (hasCompleteCameraFiles)
            {
                xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
                if (!xjw::common::project::loadFramePinholeNumericStateFromFile(input.cameraPaths.at(index), &camera) ||
                    !camera.isValid())
                {
                    execution.result.errorMessage =
                        QStringLiteral("无法读取有效外部相机: %1").arg(input.cameraPaths.at(index));
                    execution.result.summary = execution.result.errorMessage;
                    return execution;
                }
                const QSize imageSize = resolveInputImageSize(imagePath);
                if (imageSize.isValid())
                {
                    camera.setImageSize({imageSize.width(), imageSize.height()});
                }
                if (!bindInputCameraIdentity(bindingInput, index, imagePath, &camera, &execution.result.errorMessage))
                {
                    execution.result.summary = execution.result.errorMessage;
                    return execution;
                }
                numericalInput.images.push_back({imageId, nativePath, std::move(camera), sensorKey});
                continue;
            }

            const auto projectCamera =
                projectCameraIdentityAvailable
                    ? projectCameraByImageId.find(selectedImageIds.at(static_cast<std::size_t>(index)))
                    : projectCameraByImageId.end();
            if ((input.useProjectCameraIntrinsics || input.useProjectCameraPoses) &&
                projectCamera != projectCameraByImageId.end())
            {
                xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
                bool resolved = false;
                if (canonicalBindings.complete)
                {
                    if (static_cast<std::size_t>(index) >= selectedImageIds.size())
                    {
                        execution.result.errorMessage = QStringLiteral(
                            "canonical static SfM camera selection is not aligned with the input image order");
                        execution.result.summary = execution.result.errorMessage;
                        return execution;
                    }
                    const auto canonicalState =
                        canonicalPinholeByImageId.find(selectedImageIds.at(static_cast<std::size_t>(index)));
                    if (canonicalState != canonicalPinholeByImageId.end())
                    {
                        camera = canonicalState->second;
                        resolved = true;
                    }
                }
                else
                {
                    resolved = xjw::common::project::decodeFramePinholeNumericState(projectCamera->second, &camera);
                }
                if (resolved && camera.isValid())
                {
                    if (!input.useProjectCameraPoses)
                    {
                        // 重置对齐时只复用内参，外参重新由相对定向/增量注册估计。
                        camera = cameraWithIdentityPose(camera);
                    }
                    const QSize imageSize = resolveInputImageSize(imagePath);
                    if (imageSize.isValid())
                    {
                        camera.setImageSize({imageSize.width(), imageSize.height()});
                    }
                    if (!bindInputCameraIdentity(
                            bindingInput, index, imagePath, &camera, &execution.result.errorMessage))
                    {
                        execution.result.summary = execution.result.errorMessage;
                        return execution;
                    }
                    numericalInput.images.push_back({imageId, nativePath, std::move(camera), sensorKey});
                    continue;
                }
            }

            const QSize imageSize = resolveInputImageSize(imagePath);
            if (!imageSize.isValid())
            {
                execution.result.errorMessage =
                    QStringLiteral("无法读取影像尺寸，不能初始化相机内参: %1").arg(imagePath);
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
            const double focal =
                std::max(imageSize.width(), imageSize.height()) * std::max(0.1, input.estimatedFocalScale);
            xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
            camera.setIntrinsics(focal, focal, imageSize.width() * 0.5, imageSize.height() * 0.5);
            camera.setImageSize({imageSize.width(), imageSize.height()});
            if (!bindInputCameraIdentity(bindingInput, index, imagePath, &camera, &execution.result.errorMessage))
            {
                execution.result.summary = execution.result.errorMessage;
                return execution;
            }
            numericalInput.images.push_back({imageId, nativePath, std::move(camera), sensorKey});
        }

        // 阶段 4：人工标记和比例尺作为 prior track/control constraint 注入，
        // 不伪装成普通自动连接点。
        const MarkerPriorLoadResult markerPriors = MarkerPriorLoader::load(
            input.markerSetPath, input.projectMeta, imageIdByCanonicalId, input.coordinateContext.get());
        if (!markerPriors.ok)
        {
            execution.result.errorMessage = markerPriors.errorMessage;
            execution.result.summary = markerPriors.errorMessage;
            return execution;
        }
        numericalInput.priorTracks = markerPriors.tracks;
        numericalInput.scaleBars = markerPriors.scaleBars;

        // 工程数据适配到此结束；数值引擎只消费标准类型的相机、观测和回调。
        numericalInput.progressFn = [&input, registeredProgress](int registered, int total, const std::string& message)
        {
            if (input.cancelFlag && input.cancelFlag->load())
            {
                return false;
            }
            const int percent = total > 0 ? std::clamp(static_cast<int>(100.0 * registered / total), 0, 100) : 0;
            registeredProgress->store(percent);
            if (input.progressFn)
            {
                input.progressFn(QString::fromStdString(message), percent);
            }
            return true;
        };
        const IncrementalSfmResult sfmResult = engine::runPinhole(numericalInput);

        execution.reconstruction = sfmResult.reconstruction;
        if (hasCompleteCameraFiles && execution.reconstruction)
        {
            for (int index = 0; index < input.cameraPaths.size(); ++index)
            {
                execution.reconstruction->image(static_cast<ImageId>(index)).cameraPath =
                    xjw::common::io::toUtf8Path(input.cameraPaths.at(index));
            }
        }
        execution.result.success = sfmResult.success;
        execution.result.numRegisteredImages = sfmResult.numRegisteredImages;
        execution.result.numPoints3D = sfmResult.numPoints3D;
        execution.result.meanReprojError = sfmResult.meanReprojError;
        execution.result.baRmsBefore = sfmResult.baRmsBefore;
        execution.result.baRmsAfter = sfmResult.baRmsAfter;
        execution.result.baTracksTotal = sfmResult.baTracksTotal;
        execution.result.baTracksOptimized = sfmResult.baTracksOptimized;
        execution.result.baTracksFiltered = sfmResult.baTracksFiltered;
        execution.result.summary = QString::fromStdString(sfmResult.summary);

        // 数值后端、先验接纳和初始化选择全部进入稳定诊断字段，便于 GUI/CLI 对比。
        QJsonObject diagnostics;
        diagnostics.insert(QStringLiteral("selected_initial_pair"),
                           QJsonArray{static_cast<int>(sfmResult.selectedInitialImageId1),
                                      static_cast<int>(sfmResult.selectedInitialImageId2)});
        diagnostics.insert(QStringLiteral("prior_tracks_accepted"), sfmResult.priorTracksAccepted);
        diagnostics.insert(QStringLiteral("prior_tracks_rejected"), sfmResult.priorTracksRejected);
        diagnostics.insert(QStringLiteral("marker_prior_tracks_loaded"), static_cast<int>(markerPriors.tracks.size()));
        diagnostics.insert(QStringLiteral("marker_prior_scale_bars_loaded"),
                           static_cast<int>(markerPriors.scaleBars.size()));
        diagnostics.insert(QStringLiteral("control_network_applied"), sfmResult.controlNetworkApplied);
        diagnostics.insert(QStringLiteral("control_point_constraints"), sfmResult.controlPointConstraintCount);
        diagnostics.insert(QStringLiteral("ba_requested_backend"),
                           QString::fromLatin1(BundleAdjust::backendName(sfmResult.baRequestedBackend)));
        diagnostics.insert(QStringLiteral("ba_used_backend"),
                           QString::fromLatin1(BundleAdjust::backendName(sfmResult.baUsedBackend)));
        diagnostics.insert(QStringLiteral("ba_solve_status"),
                           QString::fromLatin1(BundleAdjust::solveStatusName(sfmResult.baSolveStatus)));
        diagnostics.insert(QStringLiteral("ba_solution_usable"), sfmResult.baSolutionUsable);
        diagnostics.insert(QStringLiteral("ba_result_applied"), sfmResult.baResultApplied);
        diagnostics.insert(QStringLiteral("ba_backend_fallback"), sfmResult.baBackendFallback);
        diagnostics.insert(QStringLiteral("ba_observations"), sfmResult.baObservationCount);
        diagnostics.insert(QStringLiteral("ba_total_seconds"), sfmResult.baTotalSeconds);
        diagnostics.insert(QStringLiteral("hierarchical_ba_planned_blocks"), sfmResult.hierarchicalBAPlannedBlocks);
        diagnostics.insert(QStringLiteral("hierarchical_ba_applied_blocks"), sfmResult.hierarchicalBAAppliedBlocks);
        diagnostics.insert(QStringLiteral("hierarchical_ba_updated_cameras"), sfmResult.hierarchicalBAUpdatedCameras);
        diagnostics.insert(QStringLiteral("hierarchical_ba_total_seconds"), sfmResult.hierarchicalBATotalSeconds);
        diagnostics.insert(QStringLiteral("independent_camera_blocks"), sfmResult.independentCameraBlocks);
        diagnostics.insert(QStringLiteral("independent_block_merge_inliers"), sfmResult.independentBlockMergeInliers);
        diagnostics.insert(QStringLiteral("aerial_pose_outliers_detected"), sfmResult.aerialPoseOutliersDetected);
        diagnostics.insert(QStringLiteral("aerial_pose_outliers_repaired"), sfmResult.aerialPoseOutliersRepaired);
        diagnostics.insert(QStringLiteral("aerial_pose_outliers_rejected"),
                           sfmResult.aerialPoseOutliersDetected - sfmResult.aerialPoseOutliersRepaired);
        diagnostics.insert(QStringLiteral("ba_refined_intrinsic_count"), sfmResult.baRefinedIntrinsicCount);
        diagnostics.insert(QStringLiteral("ba_refined_calibration_group_count"),
                           sfmResult.baRefinedCalibrationGroupCount);
        diagnostics.insert(QStringLiteral("ba_self_calibration_stages_run"), sfmResult.baSelfCalibrationStagesRun);
        diagnostics.insert(QStringLiteral("ba_adaptive_camera_model_fitting_evaluated"),
                           sfmResult.baAdaptiveCameraModelFittingEvaluated);
        diagnostics.insert(QStringLiteral("ba_adaptive_camera_model_fitting_applied"),
                           sfmResult.baAdaptiveCameraModelFittingApplied);
        diagnostics.insert(QStringLiteral("ba_adaptive_camera_model"),
                           QString::fromStdString(sfmResult.baAdaptiveCameraModel));
        diagnostics.insert(QStringLiteral("ba_adaptive_camera_model_reason"),
                           QString::fromStdString(sfmResult.baAdaptiveCameraModelReason));
        QJsonObject intrinsicParameterEnabled;
        QJsonObject intrinsicParameterReliability;
        QJsonObject intrinsicParameterIncrementalInformationScore;
        QJsonObject intrinsicParameterSensitivity;
        for (std::size_t index = 0; index < kBAIntrinsicParameterCount; ++index)
        {
            const auto parameter = static_cast<BAIntrinsicParameter>(index);
            const QString parameterName = QString::fromLatin1(baIntrinsicParameterName(parameter));
            intrinsicParameterEnabled.insert(parameterName, sfmResult.baIntrinsicParameterMask[index]);
            intrinsicParameterReliability.insert(parameterName, sfmResult.baIntrinsicParameterReliability[index]);
            intrinsicParameterIncrementalInformationScore.insert(
                parameterName, sfmResult.baIntrinsicParameterIncrementalInformationScore[index]);
            intrinsicParameterSensitivity.insert(parameterName, sfmResult.baIntrinsicParameterSensitivity[index]);
        }
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_enabled"), intrinsicParameterEnabled);
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_reliability"), intrinsicParameterReliability);
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_incremental_information_score"),
                           intrinsicParameterIncrementalInformationScore);
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_sensitivity"), intrinsicParameterSensitivity);
        diagnostics.insert(QStringLiteral("ba_intrinsic_observability_evaluated"),
                           sfmResult.baAdaptiveCameraModelFittingEvaluated);
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_mask_source"),
                           sfmResult.baAdaptiveCameraModelFittingEvaluated
                               ? QStringLiteral("adaptive_assessment")
                               : (sfmResult.baRefinedIntrinsicCount > 0 ? QStringLiteral("reference_ba_stage")
                                                                        : QStringLiteral("fixed")));
        QJsonObject cameraModelObservability;
        cameraModelObservability.insert(QStringLiteral("geometry_strength"), sfmResult.baCameraModelGeometryStrength);
        cameraModelObservability.insert(QStringLiteral("optical_axis_concentration"),
                                        sfmResult.baCameraModelOpticalAxisConcentration);
        cameraModelObservability.insert(QStringLiteral("median_triangulation_angle_degrees"),
                                        sfmResult.baCameraModelMedianTriangulationAngle);
        cameraModelObservability.insert(QStringLiteral("normalized_radius_p90"),
                                        sfmResult.baCameraModelNormalizedRadiusP90);
        cameraModelObservability.insert(QStringLiteral("occupied_peripheral_sectors"),
                                        sfmResult.baCameraModelOccupiedPeripheralSectors);
        cameraModelObservability.insert(QStringLiteral("observation_count"), sfmResult.baCameraModelObservationCount);
        cameraModelObservability.insert(QStringLiteral("multi_view_track_ratio"),
                                        sfmResult.baCameraModelMultiViewTrackRatio);
        cameraModelObservability.insert(QStringLiteral("observation_support"),
                                        sfmResult.baCameraModelObservationSupport);
        cameraModelObservability.insert(QStringLiteral("peripheral_coverage"),
                                        sfmResult.baCameraModelPeripheralCoverage);
        cameraModelObservability.insert(QStringLiteral("sector_coverage"), sfmResult.baCameraModelSectorCoverage);
        cameraModelObservability.insert(QStringLiteral("image_axis_balance"), sfmResult.baCameraModelImageAxisBalance);
        diagnostics.insert(QStringLiteral("ba_camera_model_observability"), cameraModelObservability);
        diagnostics.insert(QStringLiteral("ba_shared_focal_scale"), sfmResult.baSharedFocalScale);
        diagnostics.insert(QStringLiteral("ba_shared_focal_aspect_scale"), sfmResult.baSharedFocalAspectScale);
        diagnostics.insert(QStringLiteral("ba_shared_principal_offset_x_px"), sfmResult.baSharedPrincipalOffsetX);
        diagnostics.insert(QStringLiteral("ba_shared_principal_offset_y_px"), sfmResult.baSharedPrincipalOffsetY);
        diagnostics.insert(QStringLiteral("ba_shared_radial_k1"), sfmResult.baSharedRadialK1);
        diagnostics.insert(QStringLiteral("ba_shared_radial_k2"), sfmResult.baSharedRadialK2);
        diagnostics.insert(QStringLiteral("ba_shared_radial_k3"), sfmResult.baSharedRadialK3);
        diagnostics.insert(QStringLiteral("ba_shared_tangential_p1"), sfmResult.baSharedTangentialP1);
        diagnostics.insert(QStringLiteral("ba_shared_tangential_p2"), sfmResult.baSharedTangentialP2);
        const QJsonObject intrinsicReference{{QStringLiteral("focal_scale"), 1.0},
                                             {QStringLiteral("focal_aspect_scale"), 1.0},
                                             {QStringLiteral("principal_offset_x_px"), 0.0},
                                             {QStringLiteral("principal_offset_y_px"), 0.0},
                                             {QStringLiteral("radial_k1"), 0.0},
                                             {QStringLiteral("radial_k2"), 0.0},
                                             {QStringLiteral("radial_k3"), 0.0},
                                             {QStringLiteral("tangential_p1"), 0.0},
                                             {QStringLiteral("tangential_p2"), 0.0}};
        const QJsonObject intrinsicFinal{{QStringLiteral("focal_scale"), sfmResult.baSharedFocalScale},
                                         {QStringLiteral("focal_aspect_scale"), sfmResult.baSharedFocalAspectScale},
                                         {QStringLiteral("principal_offset_x_px"), sfmResult.baSharedPrincipalOffsetX},
                                         {QStringLiteral("principal_offset_y_px"), sfmResult.baSharedPrincipalOffsetY},
                                         {QStringLiteral("radial_k1"), sfmResult.baSharedRadialK1},
                                         {QStringLiteral("radial_k2"), sfmResult.baSharedRadialK2},
                                         {QStringLiteral("radial_k3"), sfmResult.baSharedRadialK3},
                                         {QStringLiteral("tangential_p1"), sfmResult.baSharedTangentialP1},
                                         {QStringLiteral("tangential_p2"), sfmResult.baSharedTangentialP2}};
        QJsonObject intrinsicDelta;
        for (auto it = intrinsicFinal.constBegin(); it != intrinsicFinal.constEnd(); ++it)
        {
            intrinsicDelta.insert(it.key(), it.value().toDouble() - intrinsicReference.value(it.key()).toDouble());
        }
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_reference_definition"),
                           QStringLiteral("normalized_stable_calibration_group_reference"));
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_reference"), intrinsicReference);
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_final"), intrinsicFinal);
        diagnostics.insert(QStringLiteral("ba_intrinsic_parameter_delta"), intrinsicDelta);
        diagnostics.insert(QStringLiteral("ba_backend_message"), QString::fromStdString(sfmResult.baBackendMessage));
        diagnostics.insert(QStringLiteral("project_intrinsic_prior_inspected"),
                           intrinsicSanitization.inspectedCameraCount);
        diagnostics.insert(QStringLiteral("project_intrinsic_prior_dominant_group"),
                           intrinsicSanitization.dominantGroupCount);
        diagnostics.insert(QStringLiteral("project_intrinsic_prior_median_focal_px"),
                           intrinsicSanitization.dominantMedianFocalPixels);
        diagnostics.insert(QStringLiteral("project_intrinsic_prior_normalized"),
                           intrinsicSanitization.normalizedCameraCount);
        QJsonArray normalizedIntrinsicImageIds;
        for (const camera_core::ImageId& imageId : intrinsicSanitization.normalizedImageIds)
        {
            normalizedIntrinsicImageIds.append(QString::fromStdString(imageId.value()));
        }
        diagnostics.insert(QStringLiteral("project_intrinsic_prior_normalized_image_ids"), normalizedIntrinsicImageIds);
        diagnostics.insert(QStringLiteral("project_intrinsic_prior_rejected"), rejectedProjectIntrinsicCount);
        diagnostics.insert(QStringLiteral("input_max_tracks_per_image"), sfmOptions.maxTracksPerImage);
        diagnostics.insert(QStringLiteral("input_max_tracks_per_grid_cell"), sfmOptions.maxTracksPerGridCell);
        diagnostics.insert(QStringLiteral("input_track_thinning_grid_columns"), sfmOptions.trackThinningGridColumns);
        diagnostics.insert(QStringLiteral("input_track_thinning_grid_rows"), sfmOptions.trackThinningGridRows);
        diagnostics.insert(QStringLiteral("camera_binding_source"),
                           autoBoundFromCanonical ? QStringLiteral("canonical_project_instances")
                                                  : (input.cameraBindings.empty() ? QStringLiteral("caller_or_unbound")
                                                                                  : QStringLiteral("explicit_input")));
        diagnostics.insert(QStringLiteral("camera_binding_count"),
                           static_cast<int>(bindingInput.cameraBindings.size()));

        std::vector<double> final_camera_focals;
        if (execution.reconstruction)
        {
            for (const ImageId image_id : execution.reconstruction->registeredImageIds())
            {
                if (!execution.reconstruction->hasCamera(image_id))
                {
                    continue;
                }
                const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera =
                    execution.reconstruction->camera(image_id);
                const double focal_x = camera.focalX();
                const double focal_y = camera.focalY();
                if (std::isfinite(focal_x) && focal_x > 0.0 && std::isfinite(focal_y) && focal_y > 0.0)
                {
                    final_camera_focals.push_back(std::sqrt(focal_x * focal_y));
                }
            }
        }
        std::sort(final_camera_focals.begin(), final_camera_focals.end());
        diagnostics.insert(QStringLiteral("final_camera_focal_count"), static_cast<int>(final_camera_focals.size()));
        if (!final_camera_focals.empty())
        {
            const std::size_t middle = final_camera_focals.size() / 2;
            const double median_focal = final_camera_focals.size() % 2 == 0
                                            ? 0.5 * (final_camera_focals[middle - 1] + final_camera_focals[middle])
                                            : final_camera_focals[middle];
            diagnostics.insert(QStringLiteral("final_camera_focal_median_px"), median_focal);
            diagnostics.insert(QStringLiteral("final_camera_focal_min_px"), final_camera_focals.front());
            diagnostics.insert(QStringLiteral("final_camera_focal_max_px"), final_camera_focals.back());
        }
        execution.result.sfmDiagnostics = diagnostics;

        if (!sfmResult.success)
        {
            execution.result.errorMessage =
                execution.result.summary.isEmpty() ? QStringLiteral("SfM 重建失败") : execution.result.summary;
            return execution;
        }

        // 仅生成待回写对象；工程服务必须在上层正式写出成功后统一提交。
        if (execution.reconstruction)
        {
            for (int index = 0; index < input.images.size(); ++index)
            {
                const ImageId imageId = static_cast<ImageId>(index);
                if (execution.reconstruction->isRegistered(imageId))
                {
                    QJsonObject cameraObject = xjw::common::project::serializeFramePinholeNumericState(
                        execution.reconstruction->camera(imageId));
                    cameraObject.insert(QStringLiteral("intrinsic_source"), QStringLiteral("sfm_estimated"));
                    cameraObject.insert(QStringLiteral("pose_source"), QStringLiteral("sfm_estimated"));
                    if (!execution.reconstruction->camera(imageId).hasBoundIdentity())
                    {
                        execution.result.errorMessage =
                            QStringLiteral("SfM 相机写回缺少 canonical ImageId，拒绝按路径生成身份: %1")
                                .arg(input.images.at(index));
                        execution.result.summary = execution.result.errorMessage;
                        return execution;
                    }
                    execution.result.cameraInstanceUpdates.push_back(
                        {execution.reconstruction->camera(imageId).imageId(),
                         execution.reconstruction->camera(imageId).instanceId(),
                         execution.reconstruction->camera(imageId).worldFrame(),
                         cameraObject});
                }
            }
        }
        return execution;
    }

    bool SfmAttemptRunner::readTiePointGraph(const QString& tiePointPath,
                                             const QStringList& selectedImages,
                                             PreparedTiePointGraph* graph,
                                             QString* errorMessage)
    {
        std::vector<std::filesystem::path> paths;
        paths.reserve(selectedImages.size());
        for (const QString& path : selectedImages)
        {
            paths.push_back(xjw::common::file::pathFromUtf8(
                xjw::common::io::toUtf8Path(xjw::common::project::normalizePath(path))));
        }
        std::string error;
        const bool ok = engine::readTiePointGraph(
            xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(tiePointPath)),
            paths,
            graph,
            &error,
            [&selectedImages](std::string_view token) {
                return selectedImageIndex(QString::fromUtf8(token.data(), static_cast<qsizetype>(token.size())),
                                          selectedImages);
            });
        if (errorMessage)
        {
            *errorMessage = QString::fromStdString(error);
        }
        return ok;
    }

} // namespace xjw::aerial_triangulation
