#include "MvsPipelineInternals.h"
#include "AdaptivePatchMatchBackend.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    // =============================================================================
    DepthFrameResult MvsPipelineService::computeDepthForView(
        int refIdx,
        const DepthGenConfig* configOverride,
        const std::function<bool(const DepthLevelSummary&, std::string*)>& firstLevelCompletionGate,
        const std::vector<cv::Mat>* frozenDepthMaps)
    {
        DepthFrameResult result;
        result.refViewIdx = refIdx;
        result.success = false;

        const DepthGenConfig& config = configOverride ? *configOverride : _config;
        result.sourceAngleCapEnabled =
            std::isfinite(config.sourceMaximumAngleDegCap) && config.sourceMaximumAngleDegCap > 0.0f;
        FrameTiming timing;
        const auto totalStart = Clock::now();
        auto stageStart = totalStart;
        const auto cancelled = [this, &result](const char* stage) -> bool
        {
            if (!_cancelled.load())
            {
                return false;
            }
            result.errorMsg = std::string("深度估计已取消: ") + stage;
            return true;
        };

        const CameraView& refView = _views[refIdx];
        if (cancelled("读取参考影像前"))
        {
            return result;
        }

        std::string referenceLoadError;
        MvsImageCache::ImageLease referenceLease = acquireImageFrame(refIdx, &referenceLoadError);
        if (!referenceLease)
        {
            result.errorMsg = "无法获取参考帧图像: " + refView.imagePath + " (" + referenceLoadError + ")";
            return result;
        }
        if (cancelled("读取参考影像后"))
        {
            return result;
        }
        cv::Mat refImg = referenceLease->preparedGray;
        const FramePinholeCamera refCam = referenceLease->preparedCamera;
        const int W = refImg.cols;
        const int H = refImg.rows;
        result.preparedRasterSize = cv::Size(W, H);

        // 选择源帧（从缓存中取，省去重复加载）
        const int NV = static_cast<int>(_views.size());
        int numSrc = std::min(config.numSourceViews, NV - 1);
        std::vector<cv::Mat> srcGrays;
        std::vector<FramePinholeCamera> srcCams;
        std::vector<int> sourceIndices;
        std::vector<MvsImageCache::ImageLease> sourceLeases;
        sourceLeases.reserve(static_cast<std::size_t>(std::max(0, numSrc)));

        const std::vector<int> selectedSources = sourceViewIndicesForFrame(refIdx, numSrc);
        for (int si : selectedSources)
        {
            if (si < 0 || si >= NV || si == refIdx)
            {
                continue;
            }
            std::string sourceLoadError;
            MvsImageCache::ImageLease sourceLease = acquireImageFrame(si, &sourceLoadError);
            if (!sourceLease)
            {
                LOG_WARN(QStringLiteral("[MVS] 跳过源帧 %1：图像 provider 获取失败：%2")
                             .arg(si)
                             .arg(QString::fromStdString(sourceLoadError)));
                continue;
            }
            cv::Mat srcImg = sourceLease->preparedGray;
            FramePinholeCamera sourceCamera = sourceLease->preparedCamera;
            if (srcImg.cols != W || srcImg.rows != H)
            {
                const double scaleX = static_cast<double>(W) / std::max(1, srcImg.cols);
                const double scaleY = static_cast<double>(H) / std::max(1, srcImg.rows);
                cv::resize(srcImg, srcImg, cv::Size(W, H));
                sourceCamera = sourceCamera.scaledIntrinsics(scaleX, scaleY);
            }
            srcGrays.push_back(srcImg);
            srcCams.push_back(sourceCamera);
            sourceIndices.push_back(si);
            sourceLeases.push_back(std::move(sourceLease));
            if (static_cast<int>(srcGrays.size()) >= numSrc)
                break;
        }

        if (srcGrays.empty())
        {
            timing.sourceMs = elapsedMs(stageStart, Clock::now());
            result.errorMsg = "没有可用的源帧";
            return result;
        }
        if (cancelled("选择源视图后"))
        {
            return result;
        }
        result.sourceViewIndices = sourceIndices;
        if (_frameCachesReady && refIdx >= 0 && refIdx < static_cast<int>(_frameCaches.size()))
        {
            const auto& cache = _frameCaches[static_cast<size_t>(refIdx)];
            result.sourceAngleDiagnostics = cache.sourceAngleDiagnostics;
            result.completeVisibilityCandidatePoolEnabled = cache.completeVisibilityCandidatePoolEnabled;
            result.completePoolChangedLegacyPlan = cache.completePoolChangedLegacyPlan;
            result.requestedSourceViewCount = cache.requestedSourceViewCount;
            result.sourceViewShortfall =
                std::max(0, result.requestedSourceViewCount - static_cast<int>(result.sourceViewIndices.size()));
            result.sourceViewShortfallReason = cache.sourceViewShortfallReason;
            result.sourceViewPlan.reserve(result.sourceViewIndices.size());
            for (const int sourceIndex : result.sourceViewIndices)
            {
                const auto score = std::find_if(cache.sourceViewScores.cbegin(),
                                                cache.sourceViewScores.cend(),
                                                [sourceIndex](const MvsSourcePlanEntry& entry)
                                                { return entry.viewIndex == sourceIndex; });
                if (score != cache.sourceViewScores.cend())
                {
                    result.sourceViewPlan.push_back(*score);
                }
            }
        }
        if (result.requestedSourceViewCount <= 0)
        {
            result.requestedSourceViewCount = static_cast<int>(result.sourceViewIndices.size());
            result.sourceViewShortfall = 0;
        }

        {
            std::ostringstream oss;
            for (size_t k = 0; k < sourceIndices.size(); ++k)
            {
                if (k > 0)
                {
                    oss << ",";
                }
                oss << sourceIndices[k];
            }
            LOG_DEBUG("[MVS][帧 %d][源视图] 共视评分选择 [%s]", refIdx, oss.str().c_str());

            if (_frameCachesReady && refIdx >= 0 && refIdx < static_cast<int>(_frameCaches.size()) &&
                !_frameCaches[static_cast<size_t>(refIdx)].sourceViewScores.empty())
            {
                std::ostringstream details;
                int detailCount = 0;
                const auto& scores = _frameCaches[static_cast<size_t>(refIdx)].sourceViewScores;
                for (int sourceIndex : sourceIndices)
                {
                    const auto it = std::find_if(scores.begin(),
                                                 scores.end(),
                                                 [sourceIndex](const MvsSourcePlanEntry& score)
                                                 { return score.viewIndex == sourceIndex; });
                    if (it == scores.end())
                    {
                        continue;
                    }
                    if (detailCount > 0)
                    {
                        details << "; ";
                    }
                    details << it->viewIndex << "(tracks=" << it->sharedTracks << ",inliers=" << it->geometricInliers
                            << ",verified_pair=" << (it->verifiedPairGeometry ? 1 : 0)
                            << ",angle=" << it->medianTriangulationAngleDeg << ",coverage=" << it->coverageScore
                            << ",score=" << it->score << ")";
                    ++detailCount;
                }
                if (detailCount > 0)
                {
                    LOG_DEBUG(QStringLiteral("[MVS][帧 %1][源视图] 诊断: %2")
                                  .arg(refIdx)
                                  .arg(QString::fromStdString(details.str())));
                }
            }
        }
        timing.sourceMs = elapsedMs(stageStart, Clock::now());

        const int minSourceViews = sourceIndices.empty() ? 0 : 1;
        const std::vector<size_t> visibleSparsePointIndices = [this, refIdx, &sourceIndices, minSourceViews]()
        {
            std::vector<size_t> points = visibleSparsePointIndicesForFrame(refIdx, sourceIndices, minSourceViews);
            if (points.empty() && minSourceViews > 0)
            {
                points = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
                LOG_DEBUG("[MVS][帧 %d][稀疏引导] 共视点为空，回退参考帧可见点=%zu", refIdx, points.size());
            }
            return points;
        }();

        // 自适应置信度阈值：源视图越少，NCC 方差越大，需适当降低阈值
        PatchMatchConfig pmCfg = config.patchMatch;
        pmCfg.cpuThreadCount = std::max(1, config.cpuWorkerCount);
        if ((int)srcGrays.size() == 1)
        {
            // 单源视图：使用较低但非零的置信度阈值。
            // 完全归零会保留所有随机初始化深度（90% 以上暗像素 NCC=0 → conf=0），
            // 导致可视化和 crossCheck 被海量噪声淹没。
            // 阈值 0.10 ≈ NCC > 0.1，可过滤纯随机噪声但保留弱匹配。
            pmCfg.confidenceThresh = 0.10f;
            LOG_DEBUG("[MVS][帧 %d][配置] 单源视图模式，GPU confidence_threshold=0.10", refIdx);
        }
        else if ((int)srcGrays.size() <= 2)
        {
            // 2 源视图：适当降低阈值但不可过低，0.10 会保留大量低质量匹配→噪声
            pmCfg.confidenceThresh = std::min(pmCfg.confidenceThresh, 0.20f);
        }
        const DepthConfidenceThresholds confidence_thresholds =
            depthConfidenceThresholds(_effectiveSceneProfile,
                                      _effectiveDepthFilterMode,
                                      static_cast<int>(srcGrays.size()),
                                      pmCfg.confidenceThresh,
                                      config.fusion.confidenceThresh);
        pmCfg.confidenceThresh = confidence_thresholds.patchMatch;
        result.effectivePatchMatchConfidenceThreshold = pmCfg.confidenceThresh;

        // 详细相机诊断仅写入 Debug 日志，避免默认终端被逐帧矩阵信息淹没。
        const std::array<double, 9> refRotation = refCam.worldToCameraRotation();
        const std::array<double, 3> refCenter = refCam.cameraCenter();
        LOG_DEBUG("[MVS][帧 %d][配置] image=%dx%d sources=%d det_rotation=%.4f center=[%.3f,%.3f,%.3f]",
                  refIdx,
                  W,
                  H,
                  static_cast<int>(srcCams.size()),
                  det3(refRotation.data()),
                  refCenter[0],
                  refCenter[1],
                  refCenter[2]);

        // 深度范围
        stageStart = Clock::now();
        float zNear, zFar;
        std::vector<size_t> depthRangeVisiblePoints = visibleSparsePointIndices;
        if (depthRangeVisiblePoints.size() < 5 && minSourceViews > 0)
        {
            depthRangeVisiblePoints = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
            LOG_DEBUG(
                "[MVS][帧 %d][深度范围] 共视点不足，回退参考帧可见点=%zu", refIdx, depthRangeVisiblePoints.size());
        }
        estimateDepthRangeFromVisiblePoints(refIdx, depthRangeVisiblePoints, zNear, zFar);
        const float originalZNear = zNear;
        const float originalZFar = zFar;
        LOG_DEBUG("[MVS][帧 %d][深度范围] near=%.4f far=%.4f", refIdx, zNear, zFar);
        timing.rangeMs = elapsedMs(stageStart, Clock::now());
        if (cancelled("深度范围估计后"))
        {
            return result;
        }

        // =========================================================================
        // ★ 极线校正（仅双目立体对时启用）
        //   将两张图像校正到极线对齐状态，使 PatchMatch 的搜索从 2D 降为近似 1D，
        //   显著降低匹配噪声。
        //   始终以较小索引为 left 进行校正，避免不同帧顺序产生不同校正几何。
        // =========================================================================
        mvs::EpipolarRectifier::RectifiedPair rectPair;
        bool useRectified = false;
        cv::Mat workRefImg = refImg;
        std::vector<cv::Mat> workSrcGrays = srcGrays;
        FramePinholeCamera workRefCam = refCam;
        std::vector<FramePinholeCamera> workSrcCams = srcCams;
        cv::Mat referenceValidMask = referenceLease->validMask;
        std::vector<cv::Mat> source_valid_masks;
        source_valid_masks.reserve(sourceIndices.size());
        bool has_source_valid_mask = false;
        for (std::size_t sourceOrdinal = 0; sourceOrdinal < sourceIndices.size(); ++sourceOrdinal)
        {
            cv::Mat source_mask = sourceLeases[sourceOrdinal]->validMask;
            if (!source_mask.empty() && source_mask.size() != cv::Size(W, H))
            {
                cv::resize(source_mask, source_mask, cv::Size(W, H), 0.0, 0.0, cv::INTER_NEAREST);
            }
            has_source_valid_mask = has_source_valid_mask || !source_mask.empty();
            source_valid_masks.push_back(std::move(source_mask));
        }
        if (!has_source_valid_mask)
        {
            source_valid_masks.clear();
        }
        if (!referenceValidMask.empty())
        {
            result.maskSource = referenceLease->validMaskSource.empty() ? "technical" : referenceLease->validMaskSource;
        }
        else
        {
            result.maskSource = "full_image";
        }
        result.maskCoverage = referenceValidMask.empty()
                                  ? 1.0f
                                  : static_cast<float>(cv::countNonZero(referenceValidMask)) /
                                        static_cast<float>(std::max<std::size_t>(1, referenceValidMask.total()));
        cv::Mat workReferenceValidMask = referenceValidMask;
        std::vector<cv::Mat> work_source_valid_masks = source_valid_masks;

        stageStart = Clock::now();
        if (srcGrays.size() == 1)
        {
            int srcIdx = sourceIndices.empty() ? -1 : sourceIndices.front();

            bool refIsCanonicalLeft = (srcIdx < 0 || refIdx < srcIdx);

            cv::Mat canonLeft = refIsCanonicalLeft ? refImg : srcGrays[0];
            cv::Mat canonRight = refIsCanonicalLeft ? srcGrays[0] : refImg;
            auto camL = refIsCanonicalLeft ? refCam : srcCams[0];
            auto camR = refIsCanonicalLeft ? srcCams[0] : refCam;

            std::string rectErr;
            if (mvs::EpipolarRectifier::rectify(canonLeft, canonRight, camL, camR, rectPair, &rectErr))
            {
                if (refIsCanonicalLeft)
                {
                    workRefImg = rectPair.rectLeft;
                    workSrcGrays = {rectPair.rectRight};
                    workRefCam = rectPair.rectCamLeft;
                    workSrcCams = {rectPair.rectCamRight};
                    rectPair.refIsRight = false;
                }
                else
                {
                    workRefImg = rectPair.rectRight;
                    workSrcGrays = {rectPair.rectLeft};
                    workRefCam = rectPair.rectCamRight;
                    workSrcCams = {rectPair.rectCamLeft};
                    rectPair.refIsRight = true;
                }
                float rectified_z_near = 0.0f;
                float rectified_z_far = 0.0f;
                if (!mvs::EpipolarRectifier::rectifiedDepthRange(
                        refCam, workRefCam, W, H, zNear, zFar, rectified_z_near, rectified_z_far))
                {
                    workRefImg = refImg;
                    workSrcGrays = srcGrays;
                    workRefCam = refCam;
                    workSrcCams = srcCams;
                    LOG_WARN("[MVS][帧 %d][极线校正] 深度范围无法转换，使用原始图像", refIdx);
                }
                else
                {
                    zNear = rectified_z_near;
                    zFar = rectified_z_far;
                    useRectified = true;
                    if (!referenceValidMask.empty())
                    {
                        const cv::Mat& reference_homography = refIsCanonicalLeft ? rectPair.H1 : rectPair.H2;
                        cv::warpPerspective(referenceValidMask,
                                            workReferenceValidMask,
                                            reference_homography,
                                            workRefImg.size(),
                                            cv::INTER_NEAREST,
                                            cv::BORDER_CONSTANT,
                                            cv::Scalar(0));
                    }
                    if (!source_valid_masks.empty() && !source_valid_masks.front().empty())
                    {
                        const cv::Mat& source_homography = refIsCanonicalLeft ? rectPair.H2 : rectPair.H1;
                        cv::warpPerspective(source_valid_masks.front(),
                                            work_source_valid_masks.front(),
                                            source_homography,
                                            workSrcGrays.front().size(),
                                            cv::INTER_NEAREST,
                                            cv::BORDER_CONSTANT,
                                            cv::Scalar(0));
                    }
                    LOG_DEBUG("[MVS][帧 %d][极线校正] 成功 reference=%s near=%.4f far=%.4f",
                              refIdx,
                              refIsCanonicalLeft ? "left" : "right",
                              zNear,
                              zFar);
                }
            }
            else
            {
                LOG_WARN("[MVS][帧 %d][极线校正] 失败，使用原始图像: %s", refIdx, rectErr.c_str());
            }
        }
        timing.rectifyMs = elapsedMs(stageStart, Clock::now());
        if (cancelled("极线校正后"))
        {
            return result;
        }

        // 三级深度金字塔：Level 3 全局结构、Level 2 几何稳定、Level 1 细节恢复。
        cv::Mat depthMap;
        cv::Mat confMap;
        cv::Mat normalMap;
        cv::Mat supportCount;
        cv::Mat photometricSourceMask;
        DepthPyramidConfig pyramid_config = makeDepthPyramidConfig(pmCfg, workRefImg.cols, workRefImg.rows);
        result.pyramidRequestedLevelCount = 3;
        result.pyramidActiveLevelCount = pyramid_config.activeLevelCount;
        result.pyramidMinimumShortSide = depthPyramidMinimumLevelShortSide();
        result.pyramidDegradedReason = pyramid_config.degradedReason;
        pyramid_config.sceneProfile = _effectiveSceneProfile;
        pyramid_config.filterMode = _effectiveDepthFilterMode;
        pyramid_config.returnNativeFinalResolution = shouldPreserveNativeFinalDepthGrid(
            config.preserveNativeFinalDepthGrid, _effectiveSceneProfile, useRectified);
        pyramid_config.saveIntermediateLevels = _config.saveIntermediatePyramidLevels;
        if (config.preserveNativeFinalDepthGrid)
        {
            if (pyramid_config.returnNativeFinalResolution)
            {
                LOG_INFO(QStringLiteral("[MVS] 帧 %1 启用实验原生最终深度网格").arg(refIdx));
            }
            else
            {
                const QString ignored_reason =
                    useRectified ? QStringLiteral("rectified_frame") : QStringLiteral("non_custom_scene");
                LOG_WARN(QStringLiteral("[MVS] 帧 %1 忽略原生最终深度网格请求：%2；保持全尺寸契约")
                             .arg(refIdx)
                             .arg(ignored_reason));
            }
        }
        for (int level_index = 0; level_index < pyramid_config.activeLevelCount; ++level_index)
        {
            PatchMatchConfig& level_config = pyramid_config.levels[level_index].patchMatch;
            level_config.epipolarRectified = useRectified;
            if (pyramid_config.levels[level_index].level == 3)
            {
                level_config.confidenceThresh = std::min(level_config.confidenceThresh, 0.08f);
            }
            else if (pyramid_config.levels[level_index].level == 2)
            {
                level_config.confidenceThresh = std::min(level_config.confidenceThresh, 0.25f);
            }
        }

        stageStart = Clock::now();
        const std::vector<ProjectedSparseDepthSample> workRefSparseSamples = collectProjectedSparseDepthSamples(
            _sparse, workRefCam, workRefImg.cols, workRefImg.rows, visibleSparsePointIndices);
        std::array<cv::Mat, 3> pyramid_sparse_hints;
        for (int level_index = 0; level_index < pyramid_config.activeLevelCount; ++level_index)
        {
            const cv::Size hint_size = patchMatchWorkSize(workRefImg, pyramid_config.levels[level_index].patchMatch);
            pyramid_sparse_hints[level_index] =
                level_index == 0 ? buildHintDepthFromProjectedSamples(
                                       refIdx, hint_size.width, hint_size.height, workRefSparseSamples)
                                 : buildSparseSeedDepthFromProjectedSamples(
                                       refIdx, hint_size.width, hint_size.height, workRefSparseSamples);
        }

        const PatchMatchConfig& support_mask_config =
            pyramid_config.levels[pyramid_config.activeLevelCount - 1].patchMatch;
        const cv::Size supportMaskSize = patchMatchWorkSize(refImg, support_mask_config);
        std::vector<ProjectedSparseDepthSample> rectifiedSupportSamples;
        const std::vector<ProjectedSparseDepthSample>* supportSamples = &workRefSparseSamples;
        if (useRectified)
        {
            rectifiedSupportSamples =
                collectProjectedSparseDepthSamples(_sparse, refCam, W, H, visibleSparsePointIndices);
            supportSamples = &rectifiedSupportSamples;
        }
        cv::Mat sparseSupportMask = buildSparseSupportMaskFromProjectedSamples(
            refIdx, supportMaskSize.width, supportMaskSize.height, *supportSamples);
        result.projectedSparseDepthSamples = *supportSamples;
        timing.hintMs = elapsedMs(stageStart, Clock::now());
        if (cancelled("构建三级深度先验后"))
        {
            return result;
        }

        stageStart = Clock::now();
        AdaptivePatchMatchBackend pyramid_backend(refIdx);
        DepthPyramidEstimator pyramid_estimator(&pyramid_backend);
        DepthPyramidRequest pyramid_request;
        pyramid_request.referenceImage = workRefImg;
        pyramid_request.referenceValidMask = workReferenceValidMask;
        pyramid_request.sourceImages = workSrcGrays;
        pyramid_request.sourceValidMasks = work_source_valid_masks;
        pyramid_request.guideImage = workRefImg;
        pyramid_request.referenceCamera = workRefCam;
        pyramid_request.sourceCameras = workSrcCams;
        pyramid_request.sparseDepthHints = pyramid_sparse_hints;
        if (frozenDepthMaps && !useRectified)
        {
            pyramid_request.sparseDepthHintRelativeRadius = config.patchMatch.geometricGuidanceRelativeDepthRadius;
            pyramid_request.sourceDepthMaps.reserve(sourceIndices.size());
            for (const int source_index : sourceIndices)
            {
                pyramid_request.sourceDepthMaps.push_back(
                    source_index >= 0 && source_index < static_cast<int>(frozenDepthMaps->size())
                        ? (*frozenDepthMaps)[static_cast<std::size_t>(source_index)]
                        : cv::Mat());
            }
            if (refIdx >= 0 && refIdx < static_cast<int>(frozenDepthMaps->size()) &&
                !(*frozenDepthMaps)[static_cast<std::size_t>(refIdx)].empty())
            {
                for (int level_index = 0; level_index < pyramid_config.activeLevelCount; ++level_index)
                {
                    const cv::Size hint_size =
                        patchMatchWorkSize(workRefImg, pyramid_config.levels[level_index].patchMatch);
                    cv::resize((*frozenDepthMaps)[static_cast<std::size_t>(refIdx)],
                               pyramid_request.sparseDepthHints[level_index],
                               hint_size,
                               0.0,
                               0.0,
                               cv::INTER_NEAREST);
                }
            }
        }
        pyramid_request.zNear = zNear;
        pyramid_request.zFar = zFar;
        pyramid_request.pyramidConfig = pyramid_config;
        pyramid_request.cancelFlag = pmCfg.cancelFlag;
        if (pyramid_config.activeLevelCount > 1)
        {
            pyramid_request.firstLevelCompletionGate = firstLevelCompletionGate;
        }

        DepthPyramidResult pyramid_result = pyramid_estimator.estimate(pyramid_request);
        if (cancelled("三级 PatchMatch 后"))
        {
            return result;
        }
        if (!pyramid_result.success)
        {
            result.errorMsg = pyramid_result.errorMessage;
            return result;
        }

        depthMap = std::move(pyramid_result.finalLevel.depth);
        confMap = std::move(pyramid_result.finalLevel.confidence);
        normalMap = std::move(pyramid_result.finalLevel.normalMap);
        supportCount = std::move(pyramid_result.finalLevel.supportCount);
        photometricSourceMask = std::move(pyramid_result.finalLevel.photometricSourceMask);
        result.selectedLevel = pyramid_result.finalLevel.level;
        result.effectiveNativeFinalDepthGrid = pyramid_config.returnNativeFinalResolution;
        {
            std::vector<std::string> fallback_reasons;
            if (!pyramid_config.degradedReason.empty())
            {
                fallback_reasons.push_back(pyramid_config.degradedReason);
            }
            if (!pyramid_result.errorMessage.empty())
            {
                fallback_reasons.push_back(pyramid_result.errorMessage);
            }
            std::ostringstream reason_stream;
            for (std::size_t index = 0; index < fallback_reasons.size(); ++index)
            {
                if (index > 0)
                {
                    reason_stream << "; ";
                }
                reason_stream << fallback_reasons[index];
            }
            result.fallbackReason = reason_stream.str();
        }
        result.pyramidLevels = std::move(pyramid_result.levelSummaries);
        result.intermediatePyramidLevels = std::move(pyramid_result.intermediateLevels);
        for (const DepthLevelSummary& summary : result.pyramidLevels)
        {
            LOG_DEBUG("[MVS][帧 %d][金字塔] level=%d ds=%d valid=%d coverage=%.1f%% "
                      "confidence=%.3f elapsed=%.1f ms status=%s",
                      refIdx,
                      summary.level,
                      summary.downsampleFactor,
                      summary.validPixelCount,
                      summary.validCoverage * 100.0f,
                      summary.meanConfidence,
                      summary.elapsedMs,
                      summary.success ? "ok" : "failed");
        }
        if (!pyramid_result.errorMessage.empty())
        {
            LOG_WARN("[MVS][帧 %d][金字塔] 降级使用 level=%d: %s",
                     refIdx,
                     pyramid_result.finalLevel.level,
                     pyramid_result.errorMessage.c_str());
        }

        // ── 极线校正反变换：将校正空间的深度图映射回原始图像空间 ──────────────
        if (useRectified && !depthMap.empty())
        {
            cv::Mat unrectified_depth = mvs::EpipolarRectifier::unrectifyDepth(depthMap, rectPair, refCam, W, H);
            if (unrectified_depth.empty())
            {
                result.errorMsg = "rectified depth could not be transformed to the reference camera";
                LOG_ERROR("[MVS][帧 %d][极线校正] 深度坐标反变换失败", refIdx);
                return result;
            }
            depthMap = std::move(unrectified_depth);
            if (!confMap.empty())
            {
                confMap = mvs::EpipolarRectifier::unrectifyNearest(confMap, rectPair, W, H);
                const cv::Mat positive_confidence = confMap > 0.0f;
                confMap.setTo(0.0f, ~positive_confidence);
                confMap.setTo(0.0f, depthMap <= 0.0f);
            }
            if (!supportCount.empty())
            {
                supportCount = mvs::EpipolarRectifier::unrectifyNearest(supportCount, rectPair, W, H);
            }
            if (!photometricSourceMask.empty())
            {
                photometricSourceMask = mvs::EpipolarRectifier::unrectifyNearest(photometricSourceMask, rectPair, W, H);
            }
            // Rectified normals are expressed in the rectified camera frame. Re-estimate them from
            // fused depth later instead of attaching vectors in the wrong coordinate frame.
            normalMap.release();
            LOG_DEBUG("[MVS][帧 %d][极线校正] 深度图已映射回原始空间", refIdx);
        }
        result.depthCompleteness.pyramidValidCount = cv::countNonZero(depthMap > 0.0f);
        timing.patchmatchMs = elapsedMs(stageStart, Clock::now());
        if (cancelled("深度图反变换后"))
        {
            return result;
        }

        // 在去畸变参考影像空间再次应用有效区域，消除极线反变换带来的边界泄漏。
        // 项目蒙版优先；未提供项目蒙版时使用 CLAHE 前计算的内容掩码。
        stageStart = Clock::now();
        cv::Mat effectiveReferenceMask;
        {
            if (!referenceLease->validMask.empty())
            {
                effectiveReferenceMask = referenceLease->validMask;
            }
            else
            {
                LOG_DEBUG("[MVS][帧 %d][掩码] 内容掩码已自动跳过", refIdx);
            }

            if (!effectiveReferenceMask.empty())
            {
                // 适配深度图尺寸（PatchMatch 可能有上采样）
                if (effectiveReferenceMask.size() != depthMap.size())
                {
                    cv::resize(
                        effectiveReferenceMask, effectiveReferenceMask, depthMap.size(), 0, 0, cv::INTER_NEAREST);
                }
                cv::compare(effectiveReferenceMask, 0, effectiveReferenceMask, cv::CMP_GT);

                int beforeMask = cv::countNonZero(depthMap > 0);
                depthMap.setTo(0, ~effectiveReferenceMask);
                if (!confMap.empty())
                {
                    confMap.setTo(0, ~effectiveReferenceMask);
                }
                int afterMask = cv::countNonZero(depthMap > 0);

                if (afterMask < beforeMask)
                {
                    LOG_DEBUG("[MVS][帧 %d][掩码] 有效区域过滤 %d->%d", refIdx, beforeMask, afterMask);
                }
            }
        }

        if (effectiveReferenceMask.empty())
        {
            effectiveReferenceMask = cv::Mat(depthMap.size(), CV_8UC1, cv::Scalar(255));
        }
        result.depthCompleteness.afterMaskValidCount = cv::countNonZero(depthMap > 0.0f);

        const SparseDepthResidualSummary pre_recovery_sparse_residual = summarizeSparseDepthResidual(
            depthMap, result.projectedSparseDepthSamples, effectiveSparseResidualRadius(result, depthMap.size()));
        const bool sparse_geometry_allows_targeted_recovery =
            permitsTargetedGapRecovery(_effectiveSceneProfile, _effectiveDepthFilterMode, pre_recovery_sparse_residual);
        cv::Mat targeted_gap_recovered_mask;
        DepthGapTargetedRecoveryStats targeted_gap_stats;
        if (config.enableTargetedGapRecovery && _effectiveSceneProfile == MvsSceneProfile::OrbitalObject &&
            !useRectified && srcGrays.size() >= 2 && sparse_geometry_allows_targeted_recovery)
        {
            DepthGapTargetedRecoveryOptions recovery_options;
            recovery_options.minimumCandidateConfidence = std::clamp(config.targetedGapRecoveryConfidence, 0.0f, 1.0f);
            recovery_options.maximumCandidatePriorRelativeDifference =
                std::max(0.0f, config.targetedGapRecoveryPriorRelativeDifference);
            recovery_options.maximumConsensusInverseDepthRelativeSpread =
                std::max(0.0f, config.targetedGapRecoveryConsensusInverseDepthSpread);
            recovery_options.maximumConsensusPriorRelativeDifference =
                std::max(recovery_options.maximumCandidatePriorRelativeDifference,
                         config.targetedGapRecoveryConsensusPriorRelativeDifference);
            if (srcGrays.size() >= 4 && config.targetedGapRecoveryHypothesisCount >= 2)
            {
                recovery_options.enableSurfaceAwarePrior = config.enableTargetedGapSurfacePrior;
                recovery_options.maximumSurfaceAnchorInverseDepthRelativeSpread =
                    std::max(0.0f, config.targetedGapSurfacePriorMaximumAnchorSpread);
                recovery_options.maximumSurfacePriorFitRelativeResidual =
                    std::max(0.0f, config.targetedGapSurfacePriorMaximumFitResidual);
                recovery_options.missingPriorRadiusRatio = std::max(
                    recovery_options.missingPriorRadiusRatio, recovery_options.maximumConsensusPriorRelativeDifference);
            }
            recovery_options.maximumPriorDistancePixels =
                std::max(1, config.targetedGapRecoveryMaximumPriorDistancePixels);
            const DepthGapTarget recovery_target =
                buildDepthGapTarget(depthMap, effectiveReferenceMask, recovery_options);
            targeted_gap_stats.supportPixelCount = recovery_target.supportPixelCount;
            targeted_gap_stats.requestedGapPixelCount = recovery_target.requestedGapPixelCount;
            targeted_gap_stats.priorCoveredGapPixelCount = recovery_target.priorCoveredGapPixelCount;
            targeted_gap_stats.skippedReason = recovery_target.skippedReason;
            if (recovery_target.valid)
            {
                const int recovery_source_count =
                    std::clamp(config.targetedGapRecoverySourceCount, 1, static_cast<int>(srcGrays.size()));
                const int requested_hypothesis_count =
                    std::clamp(config.targetedGapRecoveryHypothesisCount, 1, recovery_source_count);
                const int hypothesis_count =
                    recovery_source_count >= 4 ? std::min(requested_hypothesis_count, recovery_source_count / 2) : 1;
                std::vector<std::vector<int>> source_groups(static_cast<std::size_t>(hypothesis_count));
                for (int source_ordinal = 0; source_ordinal < recovery_source_count; ++source_ordinal)
                {
                    source_groups[static_cast<std::size_t>(source_ordinal % hypothesis_count)].push_back(
                        source_ordinal);
                }

                std::vector<cv::Mat> candidate_depths;
                std::vector<cv::Mat> candidate_confidences;
                int successful_source_count = 0;
                QStringList hypothesis_errors;
                for (int hypothesis_index = 0; hypothesis_index < hypothesis_count; ++hypothesis_index)
                {
                    const std::vector<int>& source_group = source_groups[static_cast<std::size_t>(hypothesis_index)];
                    std::vector<cv::Mat> recovery_images;
                    std::vector<FramePinholeCamera> recovery_cameras;
                    std::vector<cv::Mat> recovery_source_masks;
                    recovery_images.reserve(source_group.size());
                    recovery_cameras.reserve(source_group.size());
                    recovery_source_masks.reserve(source_group.size());
                    for (const int source_ordinal : source_group)
                    {
                        recovery_images.push_back(srcGrays[static_cast<std::size_t>(source_ordinal)]);
                        recovery_cameras.push_back(srcCams[static_cast<std::size_t>(source_ordinal)]);
                        if (!source_valid_masks.empty())
                        {
                            recovery_source_masks.push_back(
                                source_valid_masks[static_cast<std::size_t>(source_ordinal)]);
                        }
                    }

                    PatchMatchConfig recovery_config = pmCfg;
                    recovery_config.downsampleFactor = std::max(1, pyramid_result.finalLevel.downsampleFactor);
                    recovery_config.numIterations = std::clamp(std::max(6, pmCfg.numIterations / 2), 6, 10);
                    recovery_config.patchHalf = std::max(3, pmCfg.patchHalf - 1);
                    recovery_config.confidenceThresh = std::min(pmCfg.confidenceThresh, 0.18f);
                    recovery_config.minimumMaskedPatchSupportRatio =
                        std::min(recovery_config.minimumMaskedPatchSupportRatio, 0.25f);
                    recovery_config.geomConsistency = false;

                    cv::Mat candidate_depth;
                    cv::Mat candidate_confidence;
                    std::string recovery_error;
                    const bool recovery_ok = estimatePatchMatchWithAdaptiveCuda(
                        "targeted gap PatchMatch",
                        refIdx,
                        refImg,
                        recovery_images,
                        refCam,
                        recovery_cameras,
                        zNear,
                        zFar,
                        recovery_config,
                        candidate_depth,
                        &candidate_confidence,
                        &recovery_error,
                        &recovery_target.hintDepth,
                        &recovery_target.hintRadius,
                        &recovery_target.estimationMask,
                        recovery_source_masks.empty() ? nullptr : &recovery_source_masks);
                    if (recovery_ok)
                    {
                        candidate_depths.push_back(std::move(candidate_depth));
                        candidate_confidences.push_back(std::move(candidate_confidence));
                        successful_source_count += static_cast<int>(source_group.size());
                    }
                    else
                    {
                        hypothesis_errors.push_back(QStringLiteral("group_%1:%2")
                                                        .arg(hypothesis_index)
                                                        .arg(QString::fromStdString(recovery_error)));
                    }
                }
                if (!candidate_depths.empty())
                {
                    targeted_gap_stats = mergeMultiHypothesisTargetedDepthGapCandidates(depthMap,
                                                                                        confMap,
                                                                                        candidate_depths,
                                                                                        candidate_confidences,
                                                                                        recovery_target,
                                                                                        &targeted_gap_recovered_mask,
                                                                                        recovery_options);
                    targeted_gap_stats.sourceCount = successful_source_count;
                    targeted_gap_stats.attemptedHypothesisCount = hypothesis_count;
                    targeted_gap_stats.failedHypothesisCount =
                        hypothesis_count - static_cast<int>(candidate_depths.size());
                    if (!supportCount.empty() && supportCount.size() == targeted_gap_recovered_mask.size())
                    {
                        supportCount.setTo(cv::Scalar(successful_source_count), targeted_gap_recovered_mask);
                    }
                    LOG_INFO(QStringLiteral("[MVS] 帧 %1 缺口定向 PatchMatch: target=%2 "
                                            "candidate=%3 consensus=%4 recovered=%5 (%6%) "
                                            "sources=%7 hypotheses=%8/%9")
                                 .arg(refIdx)
                                 .arg(targeted_gap_stats.priorCoveredGapPixelCount)
                                 .arg(targeted_gap_stats.candidatePixelCount)
                                 .arg(targeted_gap_stats.consensusCandidatePixelCount)
                                 .arg(targeted_gap_stats.recoveredPixelCount)
                                 .arg(targeted_gap_stats.recoveryRatio * 100.0f, 0, 'f', 1)
                                 .arg(successful_source_count)
                                 .arg(static_cast<int>(candidate_depths.size()))
                                 .arg(hypothesis_count));
                    if (!hypothesis_errors.isEmpty())
                    {
                        targeted_gap_stats.skippedReason = QStringLiteral("partial_hypothesis_failure:%1")
                                                               .arg(hypothesis_errors.join(QStringLiteral(";")));
                    }
                }
                else
                {
                    targeted_gap_stats.attempted = true;
                    targeted_gap_stats.attemptedHypothesisCount = hypothesis_count;
                    targeted_gap_stats.failedHypothesisCount = hypothesis_count;
                    targeted_gap_stats.skippedReason =
                        QStringLiteral("all_hypotheses_failed:%1").arg(hypothesis_errors.join(QStringLiteral(";")));
                    LOG_WARN(QStringLiteral("[MVS] 帧 %1 缺口定向 PatchMatch 失败: %2")
                                 .arg(refIdx)
                                 .arg(hypothesis_errors.join(QStringLiteral(";"))));
                }
            }
        }
        else
        {
            targeted_gap_stats.skippedReason =
                !config.enableTargetedGapRecovery
                    ? QStringLiteral("disabled")
                    : (_effectiveSceneProfile != MvsSceneProfile::OrbitalObject
                           ? QStringLiteral("non_orbital_scene")
                           : (!sparse_geometry_allows_targeted_recovery
                                  ? QStringLiteral("sparse_absolute_depth_residual_not_primary")
                                  : QStringLiteral("insufficient_sources_or_rectified_pair")));
        }
        result.targetedGapRecoveryDiagnostics = depthGapTargetedRecoveryStatsToJson(targeted_gap_stats);
        result.targetedGapRecoveryDiagnostics.insert(QStringLiteral("pre_recovery_sparse_residual_available"),
                                                     pre_recovery_sparse_residual.available);
        result.targetedGapRecoveryDiagnostics.insert(QStringLiteral("pre_recovery_sparse_valid_sample_count"),
                                                     pre_recovery_sparse_residual.validSampleCount);
        result.targetedGapRecoveryDiagnostics.insert(QStringLiteral("pre_recovery_sparse_median_absolute_log_error"),
                                                     pre_recovery_sparse_residual.medianAbsoluteLogError);
        result.targetedGapRecoveryDiagnostics.insert(
            QStringLiteral("sparse_residual_primary_threshold"),
            sparseDepthResidualValidationThreshold(_effectiveSceneProfile, _effectiveDepthFilterMode));
        result.targetedGapRecoveryDiagnostics.insert(QStringLiteral("sparse_geometry_admitted"),
                                                     sparse_geometry_allows_targeted_recovery);

        if (!sparseSupportMask.empty())
        {
            cv::Mat supportMask = sparseSupportMask;
            if (supportMask.size() != depthMap.size())
            {
                cv::resize(supportMask, supportMask, depthMap.size(), 0, 0, cv::INTER_NEAREST);
            }

            DepthPostprocessor::applySparseSupportPrior(depthMap, confMap, supportMask, refIdx);
        }
        result.depthCompleteness.afterSparseSupportValidCount = cv::countNonZero(depthMap > 0.0f);

        const DepthFilterSettings quality_filter_settings =
            depthFilterSettings(_effectiveDepthFilterMode, static_cast<int>(sourceIndices.size()));
        FusionConfig pixel_domain_fusion_config = config.fusion;
        pixel_domain_fusion_config.minSpeckleComponentArea = quality_filter_settings.minComponentArea;
        const DepthPixelDomainScale output_pixel_scale =
            depthPixelDomainScale(result.preparedRasterSize, depthMap.size());
        const int output_local_kernel =
            scaleDepthLocalOutlierKernel(config.fusion.localDepthOutlierKernelSize, output_pixel_scale);
        const int output_same_layer_radius =
            scaleDepthPixelRadius(kFullRasterLocalSameLayerRadiusPixels, output_pixel_scale);
        result.pixelDomainDiagnostics = makePixelDomainDiagnostics(result.preparedRasterSize,
                                                                   depthMap.size(),
                                                                   pixel_domain_fusion_config,
                                                                   config.preserveNativeFinalDepthGrid,
                                                                   result.effectiveNativeFinalDepthGrid);
        cv::Mat missing_reason_map = initializeDepthMissingReasonMap(depthMap, effectiveReferenceMask);
        const cv::Mat before_output_filter = depthMap.clone();
        result.depthCompleteness.preOutputFilterValidCount = cv::countNonZero(depthMap > 0.0f);
        if (config.fusion.enableLocalDepthOutlierFilter)
        {
            const int previewOutliers = DepthPostprocessor::removeLocalDepthOutliers(depthMap,
                                                                 confMap,
                                                                 output_local_kernel,
                                                                 quality_filter_settings.localDepthOutlierRelThreshold,
                                                                 config.fusion.maxLocalDepthOutlierRemovalRatio,
                                                                 refIdx,
                                                                 output_same_layer_radius);
            if (previewOutliers > 0)
            {
                LOG_DEBUG("[MVS][帧 %d][后处理] 输出前局部突刺移除=%d", refIdx, previewOutliers);
            }
            result.depthCompleteness.outputFilterRemovedCount = previewOutliers;
        }
        markDepthLossReason(missing_reason_map, before_output_filter, depthMap, DepthMissingReason::LocalDepthOutlier);
        result.depthCompleteness.postOutputFilterValidCount = cv::countNonZero(depthMap > 0.0f);
        if (result.depthCompleteness.preOutputFilterValidCount > 0)
        {
            result.depthCompleteness.outputFilterRetentionRatio =
                static_cast<float>(result.depthCompleteness.postOutputFilterValidCount) /
                static_cast<float>(result.depthCompleteness.preOutputFilterValidCount);
        }
        if (cancelled("深度后处理后"))
        {
            return result;
        }

        cv::Mat finalValidMask = depthMap > 0.0f;
        if (!supportCount.empty())
        {
            if (supportCount.size() != depthMap.size())
            {
                cv::resize(supportCount, supportCount, depthMap.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            supportCount.setTo(cv::Scalar(0), ~finalValidMask);
        }
        if (!photometricSourceMask.empty())
        {
            if (photometricSourceMask.size() != depthMap.size())
            {
                cv::resize(photometricSourceMask, photometricSourceMask, depthMap.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            photometricSourceMask.setTo(cv::Scalar(0), ~finalValidMask);
        }
        if (!normalMap.empty())
        {
            if (normalMap.size() != depthMap.size())
            {
                cv::resize(normalMap, normalMap, depthMap.size(), 0.0, 0.0, cv::INTER_LINEAR);
            }
            normalMap.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), ~finalValidMask);
        }

        result.qualityMetrics = analyzeDepthMapQuality(
            depthMap, confMap, static_cast<int>(sourceIndices.size()), originalZNear, originalZFar);
        result.depthCompleteness.finalMetrics =
            analyzeDepthCompleteness(depthMap,
                                     effectiveReferenceMask,
                                     kSmallHoleAreaFraction,
                                     effectiveMinimumSmallHoleArea(result, depthMap.size()));
        DepthFrameQualityInput quality_input;
        quality_input.sceneProfile = _effectiveSceneProfile;
        quality_input.filterMode = _effectiveDepthFilterMode;
        quality_input.sourceViewCount = static_cast<int>(sourceIndices.size());
        quality_input.validCoverage = result.qualityMetrics.validCoverage;
        quality_input.largestComponentRatio = result.qualityMetrics.largestComponentRatio;
        quality_input.meanConfidence = result.qualityMetrics.meanConfidence;
        quality_input.multiViewConsistencyAvailable = false;
        quality_input.depthAtSearchBoundaryRatio = result.qualityMetrics.depthAtSearchBoundaryRatio;
        quality_input.hasProjectSupportMask = result.maskSource == "project" && result.maskCoverage < 0.999f;
        quality_input.validWithinMaskRatio = result.depthCompleteness.finalMetrics.validInputs
                                                 ? result.depthCompleteness.finalMetrics.validWithinMaskRatio
                                                 : -1.0f;
        quality_input.outputFilterRetentionRatio = result.depthCompleteness.outputFilterRetentionRatio;
        result.sparseDepthResidual = summarizeSparseDepthResidual(
            depthMap, result.projectedSparseDepthSamples, effectiveSparseResidualRadius(result, depthMap.size()));
        quality_input.sparseDepthResidual = result.sparseDepthResidual;
        result.qualityDecision = evaluateDepthFrame(quality_input);
        detail::applySourceAngleCapShortfallSafety(result);
        result.initialQualityAcceptanceAvailable = true;
        result.initialQualityAcceptance = result.qualityDecision.acceptance;

        const QString quality_reasons = [&result]()
        {
            QStringList values;
            for (const std::string& reason : result.qualityDecision.reasons)
            {
                values.append(QString::fromStdString(reason));
            }
            return values.join(QStringLiteral(","));
        }();
        LOG_DEBUG(QStringLiteral("[MVS] 帧 %1 质量门: acceptance=%2 coverage=%3 largest_component=%4 "
                                 "boundary=%5 confidence=%6 reasons=%7")
                      .arg(refIdx)
                      .arg(QString::fromLatin1(depthFrameAcceptanceId(result.qualityDecision.acceptance)))
                      .arg(result.qualityMetrics.validCoverage, 0, 'f', 3)
                      .arg(result.qualityMetrics.largestComponentRatio, 0, 'f', 3)
                      .arg(result.qualityMetrics.depthAtSearchBoundaryRatio, 0, 'f', 3)
                      .arg(result.qualityMetrics.meanConfidence, 0, 'f', 3)
                      .arg(quality_reasons));

        timing.filterMs = elapsedMs(stageStart, Clock::now());
        timing.totalMs = elapsedMs(totalStart, Clock::now());
        LOG_INFO(QStringLiteral("[MVS] 帧 %1 耗时统计: source=%2 ms range=%3 ms hint=%4 ms rectify=%5 ms "
                                "patchmatch=%6 ms filter=%7 ms total=%8 ms")
                     .arg(refIdx)
                     .arg(timing.sourceMs, 0, 'f', 1)
                     .arg(timing.rangeMs, 0, 'f', 1)
                     .arg(timing.hintMs, 0, 'f', 1)
                     .arg(timing.rectifyMs, 0, 'f', 1)
                     .arg(timing.patchmatchMs, 0, 'f', 1)
                     .arg(timing.filterMs, 0, 'f', 1)
                     .arg(timing.totalMs, 0, 'f', 1));

        result.depthMap = QSharedPointer<cv::Mat>::create(depthMap);
        result.confidence = QSharedPointer<cv::Mat>::create(confMap);
        result.normalMap = normalMap.empty() ? QSharedPointer<cv::Mat>() : QSharedPointer<cv::Mat>::create(normalMap);
        result.supportCount =
            supportCount.empty() ? QSharedPointer<cv::Mat>() : QSharedPointer<cv::Mat>::create(supportCount);
        result.photometricSourceMask = photometricSourceMask.empty()
                                           ? QSharedPointer<cv::Mat>()
                                           : QSharedPointer<cv::Mat>::create(photometricSourceMask);
        result.validMask = QSharedPointer<cv::Mat>::create(finalValidMask);
        result.supportRegionMask = QSharedPointer<cv::Mat>::create(effectiveReferenceMask);
        result.targetedGapRecoveredMask = targeted_gap_recovered_mask.empty()
                                              ? QSharedPointer<cv::Mat>()
                                              : QSharedPointer<cv::Mat>::create(targeted_gap_recovered_mask);
        result.targetedGapRecoveredMaskExpected = !targeted_gap_recovered_mask.empty();
        result.depthProvenance =
            QSharedPointer<cv::Mat>::create(initializeDepthProvenance(depthMap, targeted_gap_recovered_mask));
        result.missingReasonMap = QSharedPointer<cv::Mat>::create(missing_reason_map);
        result.cameraModel = cameraForDepthGrid(refCam, cv::Size(W, H), depthMap.size());
        result.success = true;
        return result;
    }
} // namespace xjw::mvs
