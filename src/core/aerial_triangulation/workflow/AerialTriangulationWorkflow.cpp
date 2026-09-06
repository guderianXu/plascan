/**
 * @file AerialTriangulationWorkflow.cpp
 * @brief “对齐照片”外部参数到连接点任务和 SfM 管线的统一编排实现。
 *
 * 本文件只处理工作流语义：质量预设、路径、蒙版、预选、缓存复用和进度映射。
 * 特征/匹配算法由 MatchPhotosTask 实现，位姿恢复和 BA 由 AerialTriangulationPipeline
 * 实现。保持这条边界可以保证 GUI 与 CLI 使用同一套实际参数。
 */

#include "workflow/AerialTriangulationWorkflow.h"

#include "ImageMatchRepository.h"
#include "log/Logger.h"
#include "preparation/TiePointPreparation.h"
#include "project/ProjectIO.h"
#include "search/SfmSearchPolicy.h"
#include "sift/SiftComputeBackend.h"
#include "workflow/AerialTriangulationPipeline.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSet>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace xjw::aerial_triangulation
{
    namespace
    {

        /// 将自由文本参数收敛为稳定小写 token。
        QString normalizedToken(QString value, const QString& fallback = QString())
        {
            value = value.trimmed().toLower();
            return value.isEmpty() ? fallback : value;
        }

        /// 将 GUI/CLI 的设备文本映射为 MatchPhotosTask 设备枚举。
        matchphotos::ComputeDevice computeDevice(const QString& device)
        {
            const QString token = normalizedToken(device, QStringLiteral("auto"));
            if (token == QStringLiteral("cpu"))
            {
                return matchphotos::ComputeDevice::Cpu;
            }
            if (token == QStringLiteral("cuda") || token == QStringLiteral("gpu"))
            {
                return matchphotos::ComputeDevice::Cuda;
            }
            if (token == QStringLiteral("opencl"))
            {
                return matchphotos::ComputeDevice::OpenCl;
            }
            if (token == QStringLiteral("metal"))
            {
                return matchphotos::ComputeDevice::Metal;
            }
            return matchphotos::ComputeDevice::Auto;
        }

        struct ClosedSequenceEvidence
        {
            bool detected = false;
            double opticalAxisConcentration = 1.0;
            double inwardAxisMedian = -1.0;
            double adjacentDistanceMadRatio = 1.0;
            double adjacentDistanceMaximumRatio = std::numeric_limits<double>::infinity();
        };

        double median(std::vector<double> values)
        {
            if (values.empty())
            {
                return 0.0;
            }
            std::sort(values.begin(), values.end());
            return 0.5 * (values[(values.size() - 1) / 2] + values[values.size() / 2]);
        }

        double vectorNorm(const std::array<double, 3>& value)
        {
            return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        }

        const FramePinholeCamera* referenceCameraForImage(const QMap<QString, FramePinholeCamera>& cameras,
                                                          const QString& image)
        {
            const auto direct = cameras.constFind(image);
            if (direct != cameras.constEnd())
            {
                return &direct.value();
            }
            const QString normalized = QDir::fromNativeSeparators(QDir::cleanPath(image));
            for (auto it = cameras.constBegin(); it != cameras.constEnd(); ++it)
            {
                if (QDir::fromNativeSeparators(QDir::cleanPath(it.key())).compare(normalized, Qt::CaseInsensitive) == 0)
                {
                    return &it.value();
                }
            }
            return nullptr;
        }

        ClosedSequenceEvidence detectEstimatedClosedSequence(const QStringList& images,
                                                             const QMap<QString, FramePinholeCamera>& referenceCameras)
        {
            ClosedSequenceEvidence evidence;
            if (images.size() < 6 || referenceCameras.size() < images.size())
            {
                return evidence;
            }

            std::vector<std::array<double, 3>> centers;
            std::vector<std::array<double, 3>> axes;
            centers.reserve(static_cast<std::size_t>(images.size()));
            axes.reserve(static_cast<std::size_t>(images.size()));
            std::array<double, 3> meanCenter{};
            std::array<double, 3> meanAxis{};
            for (const QString& image : images)
            {
                const FramePinholeCamera* referenceCamera = referenceCameraForImage(referenceCameras, image);
                if (!referenceCamera || !referenceCamera->isValid())
                {
                    return evidence;
                }
                const FramePinholeCamera camera = referenceCamera->normalizedForPositiveDepth();
                const auto center = camera.cameraCenter();
                const auto rotation = camera.cameraToWorldRotation();
                const std::array<double, 3> axis{{rotation[2], rotation[5], rotation[8]}};
                centers.push_back(center);
                axes.push_back(axis);
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    meanCenter[dimension] += center[dimension];
                    meanAxis[dimension] += axis[dimension];
                }
            }
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                meanCenter[dimension] /= images.size();
                meanAxis[dimension] /= images.size();
            }
            evidence.opticalAxisConcentration = vectorNorm(meanAxis);

            std::vector<double> inwardDots;
            std::vector<double> adjacentDistances;
            inwardDots.reserve(centers.size());
            adjacentDistances.reserve(centers.size());
            for (std::size_t index = 0; index < centers.size(); ++index)
            {
                std::array<double, 3> towardMean{{meanCenter[0] - centers[index][0],
                                                  meanCenter[1] - centers[index][1],
                                                  meanCenter[2] - centers[index][2]}};
                const double towardNorm = vectorNorm(towardMean);
                const double axisNorm = vectorNorm(axes[index]);
                if (towardNorm <= 1.0e-12 || axisNorm <= 1.0e-12)
                {
                    return evidence;
                }
                inwardDots.push_back(
                    (towardMean[0] * axes[index][0] + towardMean[1] * axes[index][1] + towardMean[2] * axes[index][2]) /
                    (towardNorm * axisNorm));

                const auto& next = centers[(index + 1) % centers.size()];
                const std::array<double, 3> delta{
                    {centers[index][0] - next[0], centers[index][1] - next[1], centers[index][2] - next[2]}};
                const double distance = vectorNorm(delta);
                if (!std::isfinite(distance) || distance <= 1.0e-12)
                {
                    return evidence;
                }
                adjacentDistances.push_back(distance);
            }

            evidence.inwardAxisMedian = median(std::move(inwardDots));
            const double adjacentMedian = median(adjacentDistances);
            std::vector<double> deviations;
            deviations.reserve(adjacentDistances.size());
            double maximumDistance = 0.0;
            for (double distance : adjacentDistances)
            {
                deviations.push_back(std::abs(distance - adjacentMedian));
                maximumDistance = std::max(maximumDistance, distance);
            }
            if (adjacentMedian <= 1.0e-12)
            {
                return evidence;
            }
            evidence.adjacentDistanceMadRatio = median(std::move(deviations)) / adjacentMedian;
            evidence.adjacentDistanceMaximumRatio = maximumDistance / adjacentMedian;
            evidence.detected = evidence.opticalAxisConcentration < 0.85 && evidence.inwardAxisMedian >= 0.5 &&
                                evidence.adjacentDistanceMadRatio <= 0.35 &&
                                evidence.adjacentDistanceMaximumRatio <= 2.5;
            return evidence;
        }

        /**
         * @brief 把 MatchPhotosTask 的分阶段进度映射到空三总进度的前 35%。
         *
         * 每个阶段使用固定区间，避免影像数或 pair 数变化造成总体进度倒退。
         */
        int tiePointProgress(const QString& stageId, int current, int maximum)
        {
            struct Range
            {
                const char* id;
                int first;
                int last;
            };
            static constexpr Range ranges[] = {{"algorithm_selection", 1, 3},
                                               {"model_prepare", 3, 8},
                                               {"feature", 8, 18},
                                               {"generic_preselection", 18, 22},
                                               {"reference_preselection", 22, 25},
                                               {"pair_selection", 25, 28},
                                               {"matching", 28, 33},
                                               {"geometry", 33, 34},
                                               {"guided_match", 34, 35},
                                               {"track_build", 35, 35}};
            Range selected{"unknown", 1, 35};
            const QString token = normalizedToken(stageId, QStringLiteral("unknown"));
            if (token == QStringLiteral("model_prepare") && maximum <= 0)
            {
                return -1;
            }
            for (const Range& range : ranges)
            {
                if (token == QLatin1String(range.id))
                {
                    selected = range;
                    break;
                }
            }
            const double fraction =
                std::clamp(static_cast<double>(std::max(0, current)) / std::max(1, maximum), 0.0, 1.0);
            return selected.first + static_cast<int>(std::round((selected.last - selected.first) * fraction));
        }

        /**
         * @brief 清理当前影像集合对应的匹配变体和最终连接点文件。
         *
         * 匹配目录属于当前工程块，一个 `.pimatch` 文件对应一幅影像。重建连接点时
         * 通过唯一仓库入口清理这些分片；不存在独立特征文件或 JSON sidecar。
         */
        bool clearTiePointCache(const AerialTriangulationResolvedConfig& config, QString* errorMessage)
        {
            QStringList failed;
            image_matching::ImageMatchRepository repository(config.tiePointContext.matchDirectory);
            QString matchClearError;
            if (!repository.clear(&matchClearError))
            {
                failed.append(matchClearError);
            }
            const QString tiePointPath = config.pipelineInput.tiePointPath;
            if (QFileInfo::exists(tiePointPath) && !QFile::remove(tiePointPath))
            {
                failed.append(tiePointPath);
            }
            if (failed.isEmpty())
            {
                return true;
            }
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法清理旧连接点缓存：\n%1").arg(failed.join(QLatin1Char('\n')));
            }
            return false;
        }

    } // namespace

    AerialTriangulationResolvedConfig
    AerialTriangulationWorkflow::resolveConfig(const AerialTriangulationOptions& options)
    {
        AerialTriangulationResolvedConfig resolved;

        // 第一阶段：收敛统一算法标识和质量预设。算法是否存在、版本和设备要求
        // 由 MatchPhotosAlgorithmSelector 通过注册表验证，空三层不拆解算法名称。
        const matchphotos::AlignmentAccuracy accuracy = matchphotos::alignmentAccuracyFromName(options.quality);
        const QString algorithmId = normalizedToken(options.matchingAlgorithmId, QStringLiteral("plamatch_hct"));

        // 第二阶段：所有缓存和正式结果都从工程根目录推导，避免 GUI/CLI 路径分叉。
        const QString projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(options.projectPath);
        const QString assetsDirectory = options.assetsDir.isEmpty()
                                            ? QDir(projectRoot).filePath(QStringLiteral("assets"))
                                            : QDir::cleanPath(options.assetsDir);
        const QString canonicalTiePointPath =
            QDir(assetsDirectory).filePath(QStringLiteral("tie_points/latest_tie_points.json"));

        // 第三阶段：装配纯 SfM/BA 输入。resetAlignment 只影响外参复用，
        // 不隐式改变连接点缓存策略。
        PreparedAerialTriangulationInput& pipeline = resolved.pipelineInput;
        pipeline.images = options.images;
        pipeline.cameraPaths = options.cameraPaths;
        pipeline.projectPath = options.projectPath;
        pipeline.markerSetPath = options.projectPath.isEmpty()
                                     ? QString()
                                     : xjw::common::project::ProjectIO::markerSetPath(options.projectPath);
        pipeline.tiePointPath = canonicalTiePointPath;
        pipeline.outputDir = QDir(QDir::cleanPath(options.outputDir)).filePath(QStringLiteral("sfm_sparse"));
        pipeline.projectMeta = options.projectMeta;
        // Align Photos Accuracy 只改变特征输入尺度；SfM/BA 保持正式 High 基线。
        pipeline.quality = 2;
        pipeline.threads = resolveSfmThreadBudget(options.threads);
        pipeline.device = normalizedToken(options.device, QStringLiteral("auto"));
        pipeline.useProjectCameraIntrinsics = true;
        pipeline.useProjectCameraPoses = !options.resetAlignment;
        pipeline.adaptiveCameraModelFitting = options.adaptiveCameraModelFitting;
        pipeline.lockInputCameraPoses = options.lockInputCameraPoses;
        const matchphotos::ReferencePreselectionMode referencePreselectionMode =
            matchphotos::referencePreselectionModeFromName(options.referenceMode);
        const QString normalizedReferenceMode = matchphotos::referencePreselectionModeName(referencePreselectionMode);
        const bool usesPhotoSequence = options.referencePreselection &&
                                       referencePreselectionMode == matchphotos::ReferencePreselectionMode::Sequential;
        const ClosedSequenceEvidence estimatedSequence =
            options.referencePreselection &&
                    referencePreselectionMode == matchphotos::ReferencePreselectionMode::Estimated
                ? detectEstimatedClosedSequence(options.images, options.referenceCameras)
                : ClosedSequenceEvidence{};
        const bool usesSequenceGeometry = usesPhotoSequence || estimatedSequence.detected;
        const bool usesClosedSequenceGeometry = estimatedSequence.detected;
        // “照片序列”提供稳定的相邻位姿插值/外推初值，但不代表相机中心必须满足等距轨迹。
        // 位姿恢复与硬距离门控必须解耦：前者帮助连续航带跨过弱纹理帧，后者在环拍、变焦
        // 或物体旋转序列上反而可能误拒绝正确 PnP，因此保持默认关闭。
        pipeline.useSequencePoseRecovery = usesSequenceGeometry;
        pipeline.enforceSequencePoseConsistency = false;
        // “照片序列”只说明输入顺序可靠，不说明最后一张与第一张相邻。把普通航带
        // 强制闭环会在序列两端制造高优先级伪边，使首尾几张照片形成低残差但错误的
        // 刚性分支。只有已有位姿明确检测到环拍时才启用首尾闭环。
        pipeline.sequenceLoopClosure = usesClosedSequenceGeometry;
        pipeline.useInitialPairHint = options.useInitialPairHint;
        pipeline.initialImageId1 = options.initialImageId1;
        pipeline.initialImageId2 = options.initialImageId2;
        pipeline.cancelFlag = options.cancelFlag;
        if (options.progressFn)
        {
            pipeline.progressFn = [progress = options.progressFn](const QString& stage, int percent)
            { progress(stage, 35 + std::clamp(percent, 0, 100) * 59 / 100); };
        }

        // 第四阶段：装配 MatchPhotosTask 参数。连接点上限在多视轨迹形成后按影像和
        // 网格分配；关键点上限则约束每张影像的检测结果。
        matchphotos::MatchPhotosOptions& tieOptions = resolved.tiePointOptions;
        tieOptions.planOnly = false;
        tieOptions.profile = matchphotos::MatchPhotosProfile::Auto;
        tieOptions.accuracy = accuracy;
        tieOptions.device = computeDevice(options.device);
        tieOptions.pairPolicy = matchphotos::makePairSelectionPolicy(matchphotos::PairSelectionPreset::Auto);
        tieOptions.algorithmId = algorithmId;
        tieOptions.lightGlueTensorRtEnginePath = QDir::cleanPath(options.lightGlueTensorRtEnginePath.trimmed());
        if (options.lightGlueTensorRtEnginePath.trimmed().isEmpty())
        {
            tieOptions.lightGlueTensorRtEnginePath.clear();
        }
        tieOptions.lomaRTensorRtPackagePath = QDir::cleanPath(options.lomaRTensorRtPackagePath.trimmed());
        if (options.lomaRTensorRtPackagePath.trimmed().isEmpty())
        {
            tieOptions.lomaRTensorRtPackagePath.clear();
        }
        tieOptions.lomaRKeypointBudget = options.lomaRKeypointBudget;
        tieOptions.maskApplyMode = normalizedToken(options.maskApplyMode, QStringLiteral("none"));
        tieOptions.cudaDevice = std::max(0, options.cudaDevice);
        tieOptions.cudaParallelPairs = std::max(0, options.cudaParallelPairs);
        tieOptions.featurePrefetchDepth = std::clamp(options.featurePrefetchDepth, 1, 4);
        tieOptions.maxImageDim = std::max(0, options.featureMaxImageDim);
        tieOptions.matchThreshold = std::clamp(options.matchThreshold, 0.0f, 1.0f);
        tieOptions.siftMaximumRatio = std::clamp(options.siftMaximumRatio, 0.0f, 1.0f);
        tieOptions.siftMinimumAdaptiveRatio =
            std::clamp(options.siftMinimumAdaptiveRatio, 0.0f, tieOptions.siftMaximumRatio);
        tieOptions.adaptiveSiftRatio = options.adaptiveSiftRatio;
        tieOptions.geometryReprojThreshold = std::max(0.1, options.geometryReprojThreshold);
        tieOptions.geometryMinInliers = std::max(8, options.geometryMinInliers);
        tieOptions.geometryMinInlierRatio = std::clamp(options.geometryMinInlierRatio, 0.01, 0.95);
        tieOptions.geometryMinGridCoverage = std::clamp(options.geometryMinGridCoverage, 0.01, 1.0);
        tieOptions.geometryMaxIterations = std::max(100, options.geometryMaxIterations);
        tieOptions.guidedMatchingMode = options.guidedImageMatching ? matchphotos::GuidedMatchingMode::Automatic
                                                                    : matchphotos::GuidedMatchingMode::Disabled;
        tieOptions.useExplicitKeypointLimit = true;
        tieOptions.maxKeypoints = std::max(0, options.keypointLimit);
        tieOptions.keypointLimitPerMegapixel = 0;
        tieOptions.maxTiePointsPerImage = std::max(0, options.tiepointLimit);
        tieOptions.tiePointGridColumns = std::clamp(options.tiePointGridColumns, 1, 64);
        tieOptions.tiePointGridRows = std::clamp(options.tiePointGridRows, 1, 64);
        const int tiePointGridCellCount = tieOptions.tiePointGridColumns * tieOptions.tiePointGridRows;
        tieOptions.maxTiePointsPerGridCell =
            options.maxTiePointsPerGridCell > 0
                ? options.maxTiePointsPerGridCell
                : (options.tiepointLimit > 0
                       ? std::max(1,
                                  (options.tiepointLimit + std::max(1, tiePointGridCellCount) - 1) /
                                      std::max(1, tiePointGridCellCount))
                       : 0);
        // MatchPhotosTask 已按“对齐照片”的逐影像水位策略选择完整多视轨迹。
        // 这些字段只作为直接 SfM/已知位姿兼容参数保留；标准自由网会复用持久化完整轨迹，
        // 不再执行第二次 PlaScan 网格硬配额抽稀。
        pipeline.maxTracksPerImage = tieOptions.maxTiePointsPerImage;
        pipeline.maxTracksPerGridCell = tieOptions.maxTiePointsPerGridCell;
        pipeline.trackThinningGridColumns = tieOptions.tiePointGridColumns;
        pipeline.trackThinningGridRows = tieOptions.tiePointGridRows;
        tieOptions.useGenericPreselection = options.genericPreselection;
        tieOptions.useReferencePreselection = options.referencePreselection;
        tieOptions.excludeStationaryTiePoints = options.excludeFixedTiePoints;
        tieOptions.stationaryTiePointMaxPixelMotion = std::max(0.0f, options.stationaryTiePointMaxPixelMotion);
        // 原始匹配缓存键包含蒙版应用阶段及两侧蒙版文件指纹。keypoints 模式下
        // 只有完全相同的蒙版输入才能命中，因此无需再无条件禁用安全缓存复用。
        tieOptions.reuseExistingMatches = options.reuseExistingMatches;

        // 第五阶段：所有正式匹配算法共用“对齐照片”兼容的 PlaMatch-HCT
        // generic → reference union 预筛选；算法选择只影响后续正式特征与匹配。
        const QString referenceMode = normalizedReferenceMode;
        tieOptions.referencePreselectionMode = referencePreselectionMode;
        const bool hasReferencePosition =
            !options.referencePositions.isEmpty() || !options.referenceCameras.isEmpty() ||
            (!options.cameraPaths.isEmpty() && options.cameraPaths.size() == options.images.size());
        const bool hasReferenceCameraPose =
            !options.referenceCameras.isEmpty() ||
            (!options.cameraPaths.isEmpty() && options.cameraPaths.size() == options.images.size());
        tieOptions.guidedUseReferenceCameraPoses = options.guidedImageMatching && options.referencePreselection &&
                                                   hasReferenceCameraPose &&
                                                   referenceMode != QStringLiteral("estimated");
        QString pairPlanningMode =
            options.genericPreselection ? QStringLiteral("generic") : QStringLiteral("all_pairs");
        if (options.referencePreselection)
        {
            // 不把 sequential 改写为 PairSelector 线性窗口，也不因参考坐标缺失
            // 关闭 reference；Source/Estimated 的索引邻域回退由公共预选器执行。
            pairPlanningMode = matchphotos::referencePreselectionModeName(tieOptions.referencePreselectionMode);
        }
        if (!options.allowedPairs.isEmpty())
        {
            tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::ManualOnly;
            pairPlanningMode = QStringLiteral("manual");
        }

        // 第六阶段：提供项目路径、缓存目录、蒙版及进度回调等运行时上下文。
        matchphotos::MatchPhotosContext& tieContext = resolved.tiePointContext;
        tieContext.projectPath = options.projectPath;
        tieContext.workingDirectory = assetsDirectory;
        tieContext.matchDirectory = options.matchDir.isEmpty()
                                        ? QDir(assetsDirectory).filePath(QStringLiteral("image_matches"))
                                        : QDir::cleanPath(options.matchDir);
        tieContext.pairInput.images = options.images;
        tieContext.pairInput.manualPairKeys = options.allowedPairs;
        tieContext.referenceCameras = options.referenceCameras;
        tieContext.referencePositions = options.referencePositions;
        tieContext.maskPaths = options.maskPaths;
        tieContext.cancelFlag = options.cancelFlag.get();
        tieContext.computeDeviceCallback = options.computeDeviceFn;
        if (options.progressFn)
        {
            tieContext.progressCallback = [progress = options.progressFn](
                                              const QString& stageId, const QString& message, int current, int maximum)
            { progress(message.isEmpty() ? stageId : message, tiePointProgress(stageId, current, maximum)); };
        }

        // 最后独立决定“是否执行连接点任务”和“是否先清缓存”。重置对齐并不等于
        // 重匹配；用户可以重置相机后继续使用同一套匹配/连接点观测。
        // autoGenerateMissingMatches 只决定缓存缺失时是否允许补算，不应让已经存在的
        // 完整 selected-track 点网每次都重跑特征、匹配和 TrackBuild。强制重建由
        // reuseExistingMatches=false 单独表达，引导匹配则仍显式刷新旧点网。
        const bool tiePointCacheExists = QFileInfo::exists(canonicalTiePointPath);
        const std::optional<int> cachedTiePointLimit = storedTiePointLimit(canonicalTiePointPath);
        const bool tiePointLimitChanged =
            cachedTiePointLimit.has_value() && *cachedTiePointLimit != tieOptions.maxTiePointsPerImage;
        resolved.cachedTiePointLimit = cachedTiePointLimit.value_or(-1);
        resolved.prepareTiePoints = options.guidedImageMatching || !options.reuseExistingMatches ||
                                    !tiePointCacheExists || tiePointLimitChanged;
        resolved.forceRebuildTiePoints = !options.reuseExistingMatches;
        if (resolved.forceRebuildTiePoints)
        {
            resolved.tiePointLimitAction = QStringLiteral("full_rebuild");
        }
        else if (!tiePointCacheExists)
        {
            resolved.tiePointLimitAction = QStringLiteral("generate");
        }
        else if (tiePointLimitChanged)
        {
            // `.pimatch` 和特征缓存仍按各自指纹复用，只重新执行多视轨迹选择并覆盖
            // latest_tie_points.json，避免把旧 8000 配额点网继续交给 4000 配额的 SfM。
            resolved.tiePointLimitAction = options.guidedImageMatching
                                               ? QStringLiteral("guided_refresh_and_rebuild_track_selection")
                                               : QStringLiteral("rebuild_track_selection");
        }
        else if (options.guidedImageMatching)
        {
            resolved.tiePointLimitAction = QStringLiteral("guided_refresh");
        }
        else
        {
            resolved.tiePointLimitAction = cachedTiePointLimit.has_value()
                                               ? QStringLiteral("reuse_cache")
                                               : QStringLiteral("reuse_cache_unknown_limit");
        }
        QJsonObject& settings = resolved.resolvedSettings;
        settings.insert(QStringLiteral("quality"), matchphotos::alignmentAccuracyName(accuracy));
        settings.insert(QStringLiteral("alignment_downscale"), matchphotos::alignmentAccuracyDownscale(accuracy));
        // 特征提取和匹配现在属于同一个可版本化算法实现，配置中只记录稳定算法 ID。
        settings.insert(QStringLiteral("matching_algorithm_id"), algorithmId);
        settings.insert(QStringLiteral("lightglue_tensorrt_engine"), tieOptions.lightGlueTensorRtEnginePath);
        settings.insert(QStringLiteral("loma_r_tensorrt_package"), tieOptions.lomaRTensorRtPackagePath);
        settings.insert(QStringLiteral("loma_r_keypoint_budget"), tieOptions.lomaRKeypointBudget);
        settings.insert(QStringLiteral("pair_planning_mode"), pairPlanningMode);
        settings.insert(QStringLiteral("sequence_pair_window"), tieOptions.pairPolicy.sequenceWindow);
        settings.insert(QStringLiteral("sequence_loop_closure"), pipeline.sequenceLoopClosure);
        settings.insert(QStringLiteral("sequence_geometry_source"),
                        usesPhotoSequence ? QStringLiteral("explicit_sequence")
                                          : (estimatedSequence.detected ? QStringLiteral("estimated_pose_detected")
                                                                        : QStringLiteral("none")));
        settings.insert(QStringLiteral("estimated_sequence_optical_axis_concentration"),
                        estimatedSequence.opticalAxisConcentration);
        settings.insert(QStringLiteral("estimated_sequence_inward_axis_median"), estimatedSequence.inwardAxisMedian);
        settings.insert(QStringLiteral("estimated_sequence_adjacent_mad_ratio"),
                        estimatedSequence.adjacentDistanceMadRatio);
        settings.insert(QStringLiteral("estimated_sequence_adjacent_maximum_ratio"),
                        std::isfinite(estimatedSequence.adjacentDistanceMaximumRatio)
                            ? estimatedSequence.adjacentDistanceMaximumRatio
                            : 0.0);
        settings.insert(QStringLiteral("reference_preselection_mode"),
                        matchphotos::referencePreselectionModeName(tieOptions.referencePreselectionMode));
        settings.insert(QStringLiteral("reference_preselection_neighbors"), tieOptions.referencePreselectionNeighbors);
        settings.insert(QStringLiteral("reference_overlap_geometry"), QStringLiteral("plamatch_reference_compatible"));
        settings.insert(QStringLiteral("keypoint_limit"), options.keypointLimit);
        settings.insert(QStringLiteral("tiepoint_limit"), options.tiepointLimit);
        settings.insert(QStringLiteral("tiepoint_limit_requested"), tieOptions.maxTiePointsPerImage);
        settings.insert(QStringLiteral("tiepoint_limit_cached"), resolved.cachedTiePointLimit);
        settings.insert(QStringLiteral("tiepoint_limit_action"), resolved.tiePointLimitAction);
        settings.insert(QStringLiteral("mask_apply_mode"), tieOptions.maskApplyMode);
        settings.insert(QStringLiteral("reset_current_alignment"), options.resetAlignment);
        settings.insert(QStringLiteral("reuse_existing_matches"), options.reuseExistingMatches);
        settings.insert(QStringLiteral("use_project_camera_intrinsics"), pipeline.useProjectCameraIntrinsics);
        settings.insert(QStringLiteral("use_project_camera_poses"), pipeline.useProjectCameraPoses);
        settings.insert(QStringLiteral("adaptive_camera_model_fitting"), options.adaptiveCameraModelFitting);
        settings.insert(QStringLiteral("lock_input_camera_poses"), pipeline.lockInputCameraPoses);
        settings.insert(QStringLiteral("cuda_parallel_pairs_requested"), options.cudaParallelPairs);
        settings.insert(QStringLiteral("cuda_parallel_pairs_effective"), 0);
        settings.insert(QStringLiteral("cuda_device"), tieOptions.cudaDevice);
        settings.insert(QStringLiteral("threads"), pipeline.threads);
        settings.insert(QStringLiteral("feature_prefetch_depth"), tieOptions.featurePrefetchDepth);
        settings.insert(QStringLiteral("feature_max_image_dim"), tieOptions.maxImageDim);
        settings.insert(QStringLiteral("match_threshold"), tieOptions.matchThreshold);
        settings.insert(QStringLiteral("sift_maximum_ratio"), tieOptions.siftMaximumRatio);
        settings.insert(QStringLiteral("sift_minimum_adaptive_ratio"), tieOptions.siftMinimumAdaptiveRatio);
        settings.insert(QStringLiteral("adaptive_sift_ratio"), tieOptions.adaptiveSiftRatio);
        settings.insert(QStringLiteral("geometry_reprojection_threshold_px"), tieOptions.geometryReprojThreshold);
        settings.insert(QStringLiteral("geometry_min_inliers"), tieOptions.geometryMinInliers);
        settings.insert(QStringLiteral("geometry_min_inlier_ratio"), tieOptions.geometryMinInlierRatio);
        settings.insert(QStringLiteral("geometry_min_grid_coverage"), tieOptions.geometryMinGridCoverage);
        settings.insert(QStringLiteral("geometry_max_iterations"), tieOptions.geometryMaxIterations);
        settings.insert(QStringLiteral("tie_point_track_builder"), QStringLiteral("align_photos_reference"));
        settings.insert(QStringLiteral("tie_point_spatial_selection"),
                        QStringLiteral("per_image_water_fill_up_to_16x16"));
        settings.insert(QStringLiteral("stationary_tie_point_rule"), QStringLiteral("four_times_mean_feature_scale"));
        settings.insert(QStringLiteral("reference_preselection_available"),
                        referenceMode == QStringLiteral("sequence") || hasReferencePosition);
        settings.insert(QStringLiteral("tie_point_preparation"),
                        resolved.forceRebuildTiePoints
                            ? QStringLiteral("force_rebuild")
                            : (options.guidedImageMatching ? QStringLiteral("guided_refresh")
                                                           : (resolved.prepareTiePoints ? QStringLiteral("fill_missing")
                                                                                        : QStringLiteral("reuse"))));
        return resolved;
    }

    std::optional<int> AerialTriangulationWorkflow::storedTiePointLimit(const QString& tiePointPath)
    {
        QFile file(tiePointPath);
        if (!file.open(QIODevice::ReadOnly))
        {
            return std::nullopt;
        }

        // Writer 把 summary/settings 放在 tracks 之前。仅读取文件头可避免为了显示一个
        // 配额值而把数百 MiB 至数 GiB 的轨迹 JSON 展开进内存。
        const QString header = QString::fromUtf8(file.read(256 * 1024));
        static const QRegularExpression settingsExpression(QStringLiteral("\\\"settings\\\"\\s*:\\s*\\{([^{}]*)\\}"),
                                                           QRegularExpression::DotMatchesEverythingOption);
        static const QRegularExpression limitExpression(QStringLiteral("\\\"tiepoint_limit\\\"\\s*:\\s*(-?\\d+)"));
        const QRegularExpressionMatch settingsMatch = settingsExpression.match(header);
        if (!settingsMatch.hasMatch())
        {
            return std::nullopt;
        }
        const QRegularExpressionMatch limitMatch = limitExpression.match(settingsMatch.captured(1));
        if (!limitMatch.hasMatch())
        {
            return std::nullopt;
        }
        bool ok = false;
        const int limit = limitMatch.captured(1).toInt(&ok);
        return ok && limit >= 0 ? std::optional<int>(limit) : std::nullopt;
    }

    AerialTriangulationResult AerialTriangulationWorkflow::run(const AerialTriangulationOptions& options)
    {
        return run(options,
                   [](const PreparedAerialTriangulationInput& input)
                   { return AerialTriangulationPipeline().run(input); });
    }

    AerialTriangulationResult AerialTriangulationWorkflow::run(const AerialTriangulationOptions& options,
                                                               const PipelineRunner& pipelineRunner,
                                                               const TiePointRunner& tiePointRunner)
    {
        AerialTriangulationResult result;
        result.config = resolveConfig(options);
        Logger::instance()->infof("[AERIAL] tiepoint_limit requested=%d cached=%d action=%s",
                                  result.config.tiePointOptions.maxTiePointsPerImage,
                                  result.config.cachedTiePointLimit,
                                  result.config.tiePointLimitAction.toUtf8().constData());
        if (!pipelineRunner)
        {
            result.reconstructionResult.errorMessage = QStringLiteral("空中三角测量 workflow 缺少 SfM 管线执行器");
            result.reconstructionResult.summary = result.reconstructionResult.errorMessage;
            return result;
        }

        // 连接点准备是可选前置阶段：复用完整缓存时直接进入 SfM。
        if (result.config.prepareTiePoints)
        {
            result.tiePointPreparationExecuted = true;
            if (result.config.forceRebuildTiePoints)
            {
                QString cleanupError;
                if (!clearTiePointCache(result.config, &cleanupError))
                {
                    result.reconstructionResult.errorMessage = cleanupError;
                    result.reconstructionResult.summary = cleanupError;
                    return result;
                }
            }
            // runner 注入只改变执行实现，不改变 resolved options，供单元测试验证边界。
            result.tiePointResult =
                TiePointPreparation::run(result.config.tiePointOptions, result.config.tiePointContext, tiePointRunner);
            if (!result.tiePointResult.success)
            {
                result.reconstructionResult.errorMessage =
                    QStringLiteral("连接点准备失败: %1").arg(result.tiePointResult.errorMessage);
                result.reconstructionResult.summary = result.reconstructionResult.errorMessage;
                return result;
            }
            if (!result.tiePointResult.tiePointPath.trimmed().isEmpty())
            {
                result.config.pipelineInput.tiePointPath = result.tiePointResult.tiePointPath;
            }
            const std::optional<int> actualTiePointLimit =
                storedTiePointLimit(result.config.pipelineInput.tiePointPath);
            result.config.resolvedSettings.insert(
                QStringLiteral("tiepoint_limit_actual"),
                actualTiePointLimit.value_or(result.config.tiePointOptions.maxTiePointsPerImage));
            const matchphotos::MatchPhotosAlgorithmPlan& algorithmPlan = result.tiePointResult.algorithmPlan;
            if (!algorithmPlan.computeDeviceDisplayName.trimmed().isEmpty())
            {
                result.config.resolvedSettings.insert(
                    QStringLiteral("matching_compute_backend"),
                    QString::fromLatin1(image_matching::siftBackendName(algorithmPlan.executionBackend)));
                result.config.resolvedSettings.insert(QStringLiteral("matching_compute_device_name"),
                                                      algorithmPlan.computeDeviceName);
                result.config.resolvedSettings.insert(QStringLiteral("matching_compute_device_display"),
                                                      algorithmPlan.computeDeviceDisplayName);
            }
            for (const matchphotos::MatchPhotosMatchRecord& match : result.tiePointResult.matches)
            {
                const int effectiveWorkers =
                    match.settings.value(QStringLiteral("cuda_parallel_pairs_effective")).toInt();
                if (effectiveWorkers > 0)
                {
                    result.config.resolvedSettings.insert(QStringLiteral("cuda_parallel_pairs_effective"),
                                                          effectiveWorkers);
                    break;
                }
            }
            if (options.pairMatchedFn)
            {
                for (const matchphotos::MatchPhotosMatchRecord& match : result.tiePointResult.matches)
                {
                    options.pairMatchedFn(
                        match.image0Path, match.image1Path, match.image0MatchFilePath, match.matchCount);
                }
            }
        }
        else
        {
            result.config.resolvedSettings.insert(QStringLiteral("tiepoint_limit_actual"),
                                                  result.config.cachedTiePointLimit);
        }

        // 连接点成功或已确认可复用后，才允许创建正式 SfM 候选。
        result.reconstructionResult = pipelineRunner(result.config.pipelineInput);
        // 将解析后的真实设置与输入连接点路径固化到结果记录，保证工程重开后可追溯。
        QJsonObject extra = result.reconstructionResult.resultRecordExtra;
        extra.insert(QStringLiteral("workflow_kind"), QStringLiteral("aerial_triangulation_align_photos"));
        extra.insert(QStringLiteral("resolved_settings"), result.config.resolvedSettings);
        extra.insert(QStringLiteral("tie_point_path"), result.config.pipelineInput.tiePointPath);
        if (result.tiePointPreparationExecuted)
        {
            extra.insert(QStringLiteral("tie_point_track_count"), result.tiePointResult.trackCount);
            extra.insert(QStringLiteral("tie_point_match_file_count"),
                         static_cast<int>(result.tiePointResult.matches.size()));
        }
        result.reconstructionResult.resultRecordExtra = extra;
        return result;
    }

} // namespace xjw::aerial_triangulation
