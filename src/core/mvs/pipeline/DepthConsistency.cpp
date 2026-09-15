#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    // =============================================================================
    // 双视图深度图左右一致性检查
    // 对每个深度像素：反投影到 3D → 投影到另一视图 → 比较另一视图的深度值
    //
    // 策略:
    //   多视图 (≥3): "需要确认" — 像素必须得到至少一个其他视图的深度一致性确认
    //   单源视图 (1): "仅移除矛盾" — 仅移除被其他视图明确否定的像素；
    //                 对方深度为 0 或超出投影范围时，保留原像素（疑罪从无）
    //   双源视图 (2): "需要确认" — 至少由一个相邻源视图在 10% 相对误差内确认；
    //                 避免环拍稀疏环每帧只有两个相邻来源时保留互相冲突的完整深度。
    // =============================================================================
    void MvsPipelineService::crossCheckDepthConsistency()
    {
        const int NV = static_cast<int>(_views.size());
        if (NV < 2)
            return;
        if (_cancelled.load())
        {
            return;
        }

        const int cross_view_source_count =
            recommendedMvsCrossViewSourceCount(_effectiveSceneProfile, _config.crossViewHoleRepairSourceCount, NV);
        const std::vector<DepthConsistencyFrameSourcePlan> source_plans =
            freezeDepthConsistencySourcePlans(_depthFrames, NV, _effectiveSceneProfile, cross_view_source_count);

        // ── 先保存所有帧的原始深度图拷贝，避免顺序处理的级联清除问题 ──────
        const bool capture_adaptive_reliability =
            _config.enableAdaptiveGeometryEvidence && _effectiveSceneProfile == MvsSceneProfile::OrbitalObject;
        const bool capture_projected_confidence =
            capture_adaptive_reliability || _config.enableDepthLayerReliabilityGuidedCorrection;
        std::vector<cv::Mat> origDepths(NV);
        std::vector<cv::Mat> origConfidences(capture_projected_confidence ? static_cast<std::size_t>(NV) : 0U);
        for (int i = 0; i < NV; ++i)
        {
            if (_cancelled.load())
            {
                return;
            }
            if (_depthFrames[i].eligibleAsConsistencySource() && _depthFrames[i].depthMap)
            {
                origDepths[i] = _depthFrames[i].depthMap->clone();
                if (capture_projected_confidence && _depthFrames[i].confidence && !_depthFrames[i].confidence->empty())
                {
                    origConfidences[static_cast<std::size_t>(i)] = _depthFrames[i].confidence->clone();
                }
            }
        }

        for (int i = 0; i < NV; ++i)
        {
            if (_cancelled.load())
            {
                return;
            }
            if (!_depthFrames[i].eligibleForConsistencyCheck() || !_depthFrames[i].depthMap)
            {
                continue;
            }
            cv::Mat& depthI = *_depthFrames[i].depthMap;
            const FramePinholeCamera camI = _depthFrames[i].cameraModel.isValid() ? _depthFrames[i].cameraModel
                                                                                  : mvsPinholeCamera(_views[i].camera);

            // origDepths 本就是为防止顺序处理级联修改而保留的快照，
            // 同时作为本帧回退源，避免再复制一张全分辨率深度图。
            const cv::Mat& depthBackup = origDepths[static_cast<std::size_t>(i)];

            const auto frame_start = Clock::now();
            const int rowWorkers = resolvedTotalCpuThreadBudget(_config);
            const DepthConsistencyFrameSourcePlan& source_plan = source_plans[static_cast<std::size_t>(i)];
            const std::vector<int>& consistencySources = source_plan.consistencySourceIndices;
            const std::vector<int>& repairSources = source_plan.geometrySourceViewIndices;
            _depthFrames[i].geometrySourceViewIndices = repairSources;
            const bool fewViews = useContradictionOnlyDepthConsistency(static_cast<int>(consistencySources.size()));
            const float relThresh = depthConsistencyRelativeThreshold(
                _effectiveSceneProfile, static_cast<int>(consistencySources.size()), _effectiveDepthFilterMode);
            const int minimum_source_confirmations = minimumDepthConsistencySourceConfirmations(
                _effectiveSceneProfile, _effectiveDepthFilterMode, static_cast<int>(consistencySources.size()));
            const int beforeValid = cv::countNonZero(depthI > 0.0f);
            const CrossViewHoleRepairOptions repair_options = orbitalCrossViewHoleRepairOptions(_config);

            cv::Mat consistent_votes(depthI.size(), CV_16U, cv::Scalar(0));
            cv::Mat occluded_votes(depthI.size(), CV_16U, cv::Scalar(0));
            cv::Mat contradicted_votes(depthI.size(), CV_16U, cv::Scalar(0));
            cv::Mat unverifiable_votes(depthI.size(), CV_16U, cv::Scalar(0));
            cv::Mat geometry_source_mask(depthI.size(), CV_16U, cv::Scalar(0));
            cv::Mat source_inverse_depth_sum(depthI.size(), CV_32F, cv::Scalar(0.0f));
            cv::Mat source_inverse_depth_squared_sum(depthI.size(), CV_32F, cv::Scalar(0.0f));
            const bool generate_adaptive_evidence =
                _config.enableAdaptiveGeometryEvidence && _effectiveSceneProfile == MvsSceneProfile::OrbitalObject;
            const bool generate_projected_source_layers = _effectiveSceneProfile == MvsSceneProfile::OrbitalObject ||
                                                          _config.enableDepthLayerReliabilityGuidedCorrection;
            AdaptiveGeometryEvidenceAccumulatorMaps adaptive_evidence_accumulator =
                generate_adaptive_evidence ? makeAdaptiveGeometryEvidenceAccumulatorMaps(depthI.size())
                                           : AdaptiveGeometryEvidenceAccumulatorMaps{};
            std::vector<cv::Mat> projected_sources;
            std::vector<ProjectedDepthEvidence> projected_source_evidence;
            if (generate_projected_source_layers)
            {
                projected_sources.resize(repairSources.size());
                if (_config.enableDepthLayerReliabilityGuidedCorrection)
                {
                    projected_source_evidence.resize(repairSources.size());
                }
            }
            std::vector<DepthConsistencySourceInput> consistency_inputs;
            consistency_inputs.reserve(consistencySources.size());

            for (int source_ordinal = 0; source_ordinal < static_cast<int>(repairSources.size()); ++source_ordinal)
            {
                const float source_progress =
                    (static_cast<float>(i) + static_cast<float>(source_ordinal) /
                                                 static_cast<float>(std::max<std::size_t>(1, repairSources.size()))) /
                    static_cast<float>(NV);
                progressChanged(QStringLiteral("多视一致性：帧 %1/%2，源视图 %3/%4")
                                    .arg(i + 1)
                                    .arg(NV)
                                    .arg(source_ordinal + 1)
                                    .arg(repairSources.size()),
                                source_progress);
                const int j = repairSources[static_cast<std::size_t>(source_ordinal)];
                if (_cancelled.load())
                {
                    return;
                }
                if (j == i)
                {
                    continue;
                }
                if (origDepths[j].empty())
                {
                    continue;
                }
                const cv::Mat& depthJ = origDepths[j];
                const FramePinholeCamera camJ = _depthFrames[j].cameraModel.isValid()
                                                    ? _depthFrames[j].cameraModel
                                                    : mvsPinholeCamera(_views[j].camera);

                if (std::find(consistencySources.begin(), consistencySources.end(), j) != consistencySources.end())
                {
                    DepthConsistencySourceInput input;
                    input.depth = depthJ;
                    input.camera = camJ;
                    if (generate_adaptive_evidence && !origConfidences[static_cast<std::size_t>(j)].empty())
                    {
                        input.confidence = origConfidences[static_cast<std::size_t>(j)];
                    }
                    input.reliabilityWeight = sourceGeometryReliabilityWeight(_depthFrames[i], j);
                    input.sourceOrdinal = source_ordinal;
                    const DepthPixelDomainScale source_pixel_scale =
                        pixelDomainScaleForResult(_depthFrames[j], depthJ.size());
                    input.searchRadiusPixels =
                        scaleDepthPixelRadius(kFullRasterConsistencySearchRadiusPixels, source_pixel_scale);
                    input.evaluateSubpixelFootprint =
                        source_pixel_scale.usesReducedGrid() && input.searchRadiusPixels == 0;
                    consistency_inputs.push_back(std::move(input));
                }
                if (generate_projected_source_layers)
                {
                    if (_config.enableDepthLayerReliabilityGuidedCorrection &&
                        !origConfidences[static_cast<std::size_t>(j)].empty())
                    {
                        ProjectedDepthEvidence evidence =
                            projectSourceDepthEvidenceToReference(depthJ,
                                                                  origConfidences[static_cast<std::size_t>(j)],
                                                                  camJ,
                                                                  camI,
                                                                  depthI.size(),
                                                                  repair_options.maximumProjectionDistancePixels,
                                                                  cameraBaselineSector(camI, camJ),
                                                                  nullptr,
                                                                  rowWorkers,
                                                                  &_cancelled);
                        projected_sources[static_cast<std::size_t>(source_ordinal)] = evidence.depth;
                        projected_source_evidence[static_cast<std::size_t>(source_ordinal)] = std::move(evidence);
                    }
                    else
                    {
                        projected_sources[static_cast<std::size_t>(source_ordinal)] =
                            projectSourceDepthToReference(depthJ,
                                                          camJ,
                                                          camI,
                                                          depthI.size(),
                                                          repair_options.maximumProjectionDistancePixels,
                                                          nullptr,
                                                          rowWorkers,
                                                          &_cancelled);
                    }
                }
            }

            accumulateDepthConsistency(
                depthI,
                camI,
                consistency_inputs,
                relThresh,
                scaleDepthPixelDistance(kFullRasterConsistencyRoundTripPixels,
                                        pixelDomainScaleForResult(_depthFrames[i], depthI.size())),
                rowWorkers,
                _cancelled,
                consistent_votes,
                occluded_votes,
                contradicted_votes,
                unverifiable_votes,
                geometry_source_mask,
                source_inverse_depth_sum,
                source_inverse_depth_squared_sum,
                generate_adaptive_evidence ? &adaptive_evidence_accumulator : nullptr);

            std::string referenceImageError;
            MvsImageCache::ImageLease referenceImageLease = acquireImageFrame(i, &referenceImageError);
            const cv::Mat* referenceGray = referenceImageLease ? &referenceImageLease->preparedGray : nullptr;
            if (!referenceImageLease)
            {
                LOG_WARN(QStringLiteral("[MVS][帧 %1][一致性] 参考影像不可用，"
                                        "跳过影像引导修复与深度层可靠性诊断：%2")
                             .arg(i)
                             .arg(QString::fromStdString(referenceImageError)));
            }
            const PreRepairDepthLayerReliability depth_layer_reliability = analyzePreRepairDepthLayerReliability(
                _depthFrames[i],
                depthBackup,
                referenceGray,
                consistent_votes,
                occluded_votes,
                contradicted_votes,
                geometry_source_mask,
                source_inverse_depth_sum,
                source_inverse_depth_squared_sum,
                generate_adaptive_evidence ? &adaptive_evidence_accumulator : nullptr);
            if (depth_layer_reliability.result.validInputs)
            {
                _depthFrames[i].depthLayerReliabilityClass =
                    QSharedPointer<cv::Mat>::create(depth_layer_reliability.result.classMap);
            }

            cv::Mat repair_mask = _depthFrames[i].supportRegionMask && !_depthFrames[i].supportRegionMask->empty()
                                      ? *_depthFrames[i].supportRegionMask
                                      : cv::Mat(depthI.size(), CV_8UC1, cv::Scalar(255));
            if (repair_mask.size() != depthI.size())
            {
                cv::resize(repair_mask, repair_mask, depthI.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            const cv::Mat consistentMask = makeDepthConsistencyMask(depthI,
                                                                    static_cast<int>(consistencySources.size()),
                                                                    minimum_source_confirmations,
                                                                    consistent_votes,
                                                                    occluded_votes,
                                                                    contradicted_votes,
                                                                    rowWorkers,
                                                                    &_cancelled);

            if (_cancelled.load())
            {
                return;
            }
            progressChanged(QStringLiteral("多视一致性：帧 %1/%2，选择主深度层").arg(i + 1).arg(NV),
                            static_cast<float>(i) / static_cast<float>(NV));

            _depthFrames[i].crossViewRepairedMask =
                QSharedPointer<cv::Mat>::create(depthI.size(), CV_8UC1, cv::Scalar(0));
            DominantDepthLayerSelectionStats layer_selection_stats;
            DepthGeometryHypothesisRerankMaps geometry_rerank_maps;
            if (generate_adaptive_evidence)
            {
                // Revision 15 chooses one observable depth layer before TSDF.
                // A stable projected-source cluster may refine/switch the native
                // hypothesis or transfer measured depth into a missing pixel.
                // Ambiguous native samples remain available at reduced confidence
                // so consistency filtering cannot create an entire missing sector.
                depthBackup.copyTo(depthI);
                depthI.setTo(0.0f, repair_mask == 0);
                DominantDepthLayerSelectionOptions layer_selection_options;
                layer_selection_options.enableReliabilityGuidedCorrection =
                    _config.enableDepthLayerReliabilityGuidedCorrection;
                layer_selection_options.restrictToReliabilityGuidedCandidates =
                    !generate_adaptive_evidence && _config.enableDepthLayerReliabilityGuidedCorrection;
                const cv::Mat* native_reliability_classes = _config.enableDepthLayerReliabilityGuidedCorrection
                                                                ? &depth_layer_reliability.result.classMap
                                                                : nullptr;
                layer_selection_stats = selectDominantProjectedDepthLayer(
                    depthI,
                    repair_mask,
                    projected_sources,
                    consistent_votes,
                    contradicted_votes,
                    layer_selection_options,
                    _depthFrames[i].confidence ? _depthFrames[i].confidence.data() : nullptr,
                    _depthFrames[i].crossViewRepairedMask.data(),
                    &geometry_source_mask,
                    &source_inverse_depth_sum,
                    &source_inverse_depth_squared_sum,
                    &consistent_votes,
                    rowWorkers,
                    &_cancelled,
                    native_reliability_classes,
                    nullptr,
                    _config.enableDepthLayerReliabilityGuidedCorrection ? &projected_source_evidence : nullptr,
                    _config.enableDepthLayerReliabilityGuidedCorrection ? &geometry_rerank_maps : nullptr);
            }
            else if (_config.enableDepthLayerReliabilityGuidedCorrection)
            {
                cv::Mat guided_depth = depthBackup.clone();
                guided_depth.setTo(0.0f, repair_mask == 0);
                cv::Mat guided_confidence =
                    _depthFrames[i].confidence ? _depthFrames[i].confidence->clone() : cv::Mat();
                cv::Mat guided_geometry_source_mask = geometry_source_mask.clone();
                cv::Mat guided_inverse_sum = source_inverse_depth_sum.clone();
                cv::Mat guided_inverse_squared_sum = source_inverse_depth_squared_sum.clone();
                cv::Mat guided_votes = consistent_votes.clone();
                cv::Mat guided_changed_mask;
                DominantDepthLayerSelectionOptions layer_selection_options;
                layer_selection_options.enableReliabilityGuidedCorrection = true;
                layer_selection_options.restrictToReliabilityGuidedCandidates = true;
                layer_selection_stats =
                    selectDominantProjectedDepthLayer(guided_depth,
                                                      repair_mask,
                                                      projected_sources,
                                                      consistent_votes,
                                                      contradicted_votes,
                                                      layer_selection_options,
                                                      guided_confidence.empty() ? nullptr : &guided_confidence,
                                                      _depthFrames[i].crossViewRepairedMask.data(),
                                                      &guided_geometry_source_mask,
                                                      &guided_inverse_sum,
                                                      &guided_inverse_squared_sum,
                                                      &guided_votes,
                                                      rowWorkers,
                                                      &_cancelled,
                                                      &depth_layer_reliability.result.classMap,
                                                      &guided_changed_mask,
                                                      &projected_source_evidence,
                                                      &geometry_rerank_maps);
                depthBackup.copyTo(depthI);
                depthI.setTo(0.0f, repair_mask == 0);
                depthI.setTo(0.0f, consistentMask == 0);
                if (!guided_changed_mask.empty())
                {
                    guided_depth.copyTo(depthI, guided_changed_mask);
                    if (_depthFrames[i].confidence && !guided_confidence.empty())
                    {
                        guided_confidence.copyTo(*_depthFrames[i].confidence, guided_changed_mask);
                    }
                    guided_geometry_source_mask.copyTo(geometry_source_mask, guided_changed_mask);
                    guided_inverse_sum.copyTo(source_inverse_depth_sum, guided_changed_mask);
                    guided_inverse_squared_sum.copyTo(source_inverse_depth_squared_sum, guided_changed_mask);
                    guided_votes.copyTo(consistent_votes, guided_changed_mask);
                }
            }
            else
            {
                depthI.setTo(0, consistentMask == 0);
            }
            if (_config.enableDepthLayerReliabilityGuidedCorrection && geometry_rerank_maps.compatible(depthI.size()))
            {
                _depthFrames[i].geometryRerankMaps =
                    QSharedPointer<DepthGeometryHypothesisRerankMaps>::create(std::move(geometry_rerank_maps));
            }
            WeakNativeDepthRetentionStats weak_native_retention;
            if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                const cv::Mat empty_confidence;
                const cv::Mat& original_confidence =
                    _depthFrames[i].confidence ? *_depthFrames[i].confidence : empty_confidence;
                weak_native_retention = retainWeaklyVerifiedNativeDepth(
                    depthBackup,
                    original_confidence,
                    repair_mask,
                    consistent_votes,
                    contradicted_votes,
                    {},
                    &depthI,
                    _depthFrames[i].confidence ? _depthFrames[i].confidence.data() : nullptr);
            }
            cv::Mat anchored_interpolation_mask;
            cv::Mat native_interpolation_anchor_eligibility_mask;
            const cv::Mat* native_interpolation_anchor_eligibility = nullptr;
            if (_config.enableDepthLayerReliabilityAnchorGate)
            {
                native_interpolation_anchor_eligibility_mask = cv::Mat(depthI.size(), CV_8UC1, cv::Scalar(0));
                if (depth_layer_reliability.result.validInputs &&
                    depth_layer_reliability.result.classMap.type() == CV_8UC1 &&
                    depth_layer_reliability.result.classMap.size() == depthI.size())
                {
                    cv::compare(depth_layer_reliability.result.classMap,
                                static_cast<std::uint8_t>(DepthLayerReliabilityClass::Reliable),
                                native_interpolation_anchor_eligibility_mask,
                                cv::CMP_EQ);
                }
                native_interpolation_anchor_eligibility = &native_interpolation_anchor_eligibility_mask;
            }
            progressChanged(QStringLiteral("多视一致性：帧 %1/%2，修复内部缺口").arg(i + 1).arg(NV),
                            static_cast<float>(i) / static_cast<float>(NV));
            const CrossViewHoleRepairStats repair_stats = repairDepthHolesFromProjectedSources(
                depthI,
                repair_mask,
                projected_sources,
                repair_options,
                _depthFrames[i].confidence ? _depthFrames[i].confidence.data() : nullptr,
                &consistent_votes,
                _depthFrames[i].crossViewRepairedMask.data(),
                &geometry_source_mask,
                &source_inverse_depth_sum,
                &source_inverse_depth_squared_sum,
                &camI,
                referenceGray,
                &anchored_interpolation_mask,
                rowWorkers,
                &_cancelled,
                native_interpolation_anchor_eligibility);
            if (_cancelled.load())
            {
                return;
            }
            WeakNativeDepthRetentionOptions unconfirmed_backfill_options;
            unconfirmed_backfill_options.minimumConfirmationCount = std::numeric_limits<int>::max();
            unconfirmed_backfill_options.retainUnconfirmedWithoutContradiction = true;
            const cv::Mat empty_backfill_confidence;
            const cv::Mat& original_backfill_confidence =
                _depthFrames[i].confidence ? *_depthFrames[i].confidence : empty_backfill_confidence;
            const WeakNativeDepthRetentionStats unconfirmed_native_backfill = retainWeaklyVerifiedNativeDepth(
                depthBackup,
                original_backfill_confidence,
                repair_mask,
                consistent_votes,
                contradicted_votes,
                unconfirmed_backfill_options,
                &depthI,
                _depthFrames[i].confidence ? _depthFrames[i].confidence.data() : nullptr);
            _depthFrames[i].crossViewRepairDiagnostics = crossViewHoleRepairStatsToJson(repair_stats);
            _depthFrames[i].crossViewRepairDiagnostics.insert(QStringLiteral("depth_layer_reliability"),
                                                              depth_layer_reliability.diagnostics);
            _depthFrames[i].crossViewRepairDiagnostics.insert(
                QStringLiteral("dominant_depth_layer_selection"),
                dominantDepthLayerSelectionStatsToJson(layer_selection_stats));
            _depthFrames[i].crossViewRepairDiagnostics.insert(
                QStringLiteral("weak_native_retention"),
                QJsonObject{{QStringLiteral("considered_pixel_count"),
                             static_cast<double>(weak_native_retention.consideredPixelCount)},
                            {QStringLiteral("retained_pixel_count"),
                             static_cast<double>(weak_native_retention.retainedPixelCount)},
                            {QStringLiteral("retained_unconfirmed_pixel_count"),
                             static_cast<double>(weak_native_retention.retainedUnconfirmedPixelCount)},
                            {QStringLiteral("rejected_contradiction_pixel_count"),
                             static_cast<double>(weak_native_retention.rejectedContradictionPixelCount)},
                            {QStringLiteral("rejected_no_confirmation_pixel_count"),
                             static_cast<double>(weak_native_retention.rejectedNoConfirmationPixelCount)},
                            {QStringLiteral("confidence_multiplier"), 0.55},
                            {QStringLiteral("minimum_retained_confidence"), 0.80}});
            _depthFrames[i].crossViewRepairDiagnostics.insert(
                QStringLiteral("unconfirmed_native_backfill"),
                QJsonObject{{QStringLiteral("considered_pixel_count"),
                             static_cast<double>(unconfirmed_native_backfill.consideredPixelCount)},
                            {QStringLiteral("retained_pixel_count"),
                             static_cast<double>(unconfirmed_native_backfill.retainedPixelCount)},
                            {QStringLiteral("rejected_contradiction_pixel_count"),
                             static_cast<double>(unconfirmed_native_backfill.rejectedContradictionPixelCount)}});
            _depthFrames[i].depthCompleteness.crossViewRepairedCount +=
                static_cast<int>(repair_stats.repairedPixelCount + layer_selection_stats.switchedNativePixelCount +
                                 layer_selection_stats.transferredMissingPixelCount);
            cv::Mat restoration_mask = _depthFrames[i].supportRegionMask && !_depthFrames[i].supportRegionMask->empty()
                                           ? *_depthFrames[i].supportRegionMask
                                           : cv::Mat(depthI.size(), CV_8UC1, cv::Scalar(255));
            if (restoration_mask.size() != depthI.size())
            {
                cv::resize(restoration_mask, restoration_mask, depthI.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            const cv::Mat empty_confidence;
            const cv::Mat& restoration_confidence =
                _depthFrames[i].confidence ? *_depthFrames[i].confidence : empty_confidence;
            const int restored_count =
                restoreSmallInteriorDepthHoles(depthI,
                                               depthBackup,
                                               restoration_confidence,
                                               restoration_mask,
                                               0.75f,
                                               0.12f,
                                               kSmallHoleAreaFraction,
                                               effectiveMinimumSmallHoleArea(_depthFrames[i], depthI.size()));
            _depthFrames[i].depthCompleteness.restoredFromPrefilterCount += restored_count;
            int afterValid = cv::countNonZero(depthI > 0);
            float keepRate = beforeValid > 0 ? 100.f * afterValid / beforeValid : 0.f;
            const double frame_elapsed_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - frame_start).count();
            LOG_INFO("[MVS][帧 %d][一致性] mode=%s workers=%d sources=%zu/%zu confirmations=%d "
                     "cross_view=%llu anchored=%llu two_source=%llu/%llu valid=%d->%d "
                     "retention=%.1f%% elapsed=%.1f ms",
                     i,
                     fewViews ? "conflict_only" : "confirmed",
                     rowWorkers,
                     consistencySources.size(),
                     repairSources.size(),
                     minimum_source_confirmations,
                     static_cast<unsigned long long>(repair_stats.repairedPixelCount),
                     static_cast<unsigned long long>(repair_stats.anchoredInterpolation.interpolatedPixelCount),
                     static_cast<unsigned long long>(repair_stats.twoSourceGrownPixelCount),
                     static_cast<unsigned long long>(repair_stats.twoSourceCandidatePixelCount),
                     beforeValid,
                     afterValid,
                     keepRate,
                     frame_elapsed_ms);

            const int observed_after_valid = afterValid;
            bool original_depth_fallback_applied = false;
            // 只恢复诊断载体，不把恢复后的覆盖率当成一致性成功证据。
            if (afterValid < beforeValid / 10 && beforeValid > 100)
            {
                LOG_WARN("[MVS][帧 %d][一致性] 保留率过低 %.1f%%，"
                         "仅恢复原始深度作为诊断载体",
                         i,
                         keepRate);
                depthBackup.copyTo(depthI);
                _depthFrames[i].crossViewRepairedMask->setTo(cv::Scalar(0));
                anchored_interpolation_mask.setTo(cv::Scalar(0));
                afterValid = beforeValid;
                original_depth_fallback_applied = true;
            }
            if (!_depthFrames[i].depthProvenance || _depthFrames[i].depthProvenance->empty())
            {
                _depthFrames[i].depthProvenance = QSharedPointer<cv::Mat>::create(initializeDepthProvenance(
                    depthBackup,
                    _depthFrames[i].targetedGapRecoveredMask ? *_depthFrames[i].targetedGapRecoveredMask : cv::Mat()));
            }
            updateDepthProvenance(*_depthFrames[i].depthProvenance,
                                  depthI,
                                  _depthFrames[i].targetedGapRecoveredMask ? *_depthFrames[i].targetedGapRecoveredMask
                                                                           : cv::Mat(),
                                  *_depthFrames[i].crossViewRepairedMask,
                                  anchored_interpolation_mask);
            const GeometryEvidenceMaps geometry_evidence = makeGeometryEvidenceMaps(depthI,
                                                                                    consistent_votes,
                                                                                    geometry_source_mask,
                                                                                    source_inverse_depth_sum,
                                                                                    source_inverse_depth_squared_sum);
            _depthFrames[i].geometrySupportCount = QSharedPointer<cv::Mat>::create(geometry_evidence.supportCount);
            _depthFrames[i].geometrySourceMask = QSharedPointer<cv::Mat>::create(geometry_evidence.sourceMask);
            _depthFrames[i].inverseDepthMean = QSharedPointer<cv::Mat>::create(geometry_evidence.inverseDepthMean);
            _depthFrames[i].inverseDepthRelativeSpread =
                QSharedPointer<cv::Mat>::create(geometry_evidence.inverseDepthRelativeSpread);
            if (!_depthFrames[i].missingReasonMap || _depthFrames[i].missingReasonMap->empty())
            {
                _depthFrames[i].missingReasonMap =
                    QSharedPointer<cv::Mat>::create(initializeDepthMissingReasonMap(depthBackup, repair_mask));
            }
            markDepthLossReason(*_depthFrames[i].missingReasonMap,
                                depthBackup,
                                depthI,
                                DepthMissingReason::InsufficientGeometrySupport);
            finalizeDepthMissingReasonMap(*_depthFrames[i].missingReasonMap,
                                          depthI,
                                          repair_mask,
                                          geometry_evidence.supportCount,
                                          contradicted_votes);
            if (generate_adaptive_evidence)
            {
                const AdaptiveGeometryEvidenceMaps adaptive_evidence =
                    makeAdaptiveGeometryEvidenceMaps(depthBackup, adaptive_evidence_accumulator);
                _depthFrames[i].adaptiveGeometrySupportWeight =
                    QSharedPointer<cv::Mat>::create(adaptive_evidence.supportWeight);
                _depthFrames[i].adaptiveGeometryEffectiveViewCount =
                    QSharedPointer<cv::Mat>::create(adaptive_evidence.effectiveViewCount);
                _depthFrames[i].adaptiveGeometryConflictRatio =
                    QSharedPointer<cv::Mat>::create(adaptive_evidence.conflictRatio);
            }
            const DepthConsistencyPublicationSummary consistency_publication = summarizeDepthConsistencyPublication(
                beforeValid, observed_after_valid, afterValid, original_depth_fallback_applied, true);
            const float consistency_keep_rate = consistency_publication.observedRetentionRatio;
            _depthFrames[i].depthCompleteness.preConsistencyValidCount = beforeValid;
            _depthFrames[i].depthCompleteness.postConsistencyValidCount = observed_after_valid;
            _depthFrames[i].depthCompleteness.consistencyRetentionRatio = consistency_keep_rate;
            _depthFrames[i].depthCompleteness.publishedPostConsistencyValidCount = afterValid;
            _depthFrames[i].depthCompleteness.publishedConsistencyRetentionRatio =
                consistency_publication.publishedRetentionRatio;
            _depthFrames[i].depthCompleteness.consistencyPublicationFallbackApplied = original_depth_fallback_applied;
            const DepthConsistencyVoteTotals vote_totals = summarizeDepthConsistencyVotes(
                consistent_votes, occluded_votes, contradicted_votes, unverifiable_votes, rowWorkers);
            _depthFrames[i].depthCompleteness.consistencyConfirmedObservationCount =
                static_cast<int>(vote_totals.consistent);
            _depthFrames[i].depthCompleteness.consistencyOccludedObservationCount =
                static_cast<int>(vote_totals.occluded);
            _depthFrames[i].depthCompleteness.consistencyContradictedObservationCount =
                static_cast<int>(vote_totals.contradicted);
            _depthFrames[i].depthCompleteness.consistencyUnverifiableObservationCount =
                static_cast<int>(vote_totals.unverifiable);
            _depthFrames[i].depthCompleteness.consistencyRejectedPixelCount =
                std::max(0, beforeValid - observed_after_valid);
            AdaptiveGeometryEvidenceMaps adaptive_geometry_maps;
            adaptive_geometry_maps.effectiveViewCount = _depthFrames[i].adaptiveGeometryEffectiveViewCount
                                                            ? *_depthFrames[i].adaptiveGeometryEffectiveViewCount
                                                            : cv::Mat();
            adaptive_geometry_maps.conflictRatio = _depthFrames[i].adaptiveGeometryConflictRatio
                                                       ? *_depthFrames[i].adaptiveGeometryConflictRatio
                                                       : cv::Mat();
            const AdaptiveGeometryEvidenceSummary adaptive_summary =
                summarizeAdaptiveGeometryEvidence(adaptive_geometry_maps);
            const cv::Mat discrete_support_region =
                _depthFrames[i].supportRegionMask ? *_depthFrames[i].supportRegionMask : cv::Mat();
            const DiscreteGeometryCoreSummary discrete_summary =
                summarizeDiscreteGeometryCore(depthI,
                                              geometry_evidence.supportCount,
                                              geometry_evidence.inverseDepthRelativeSpread,
                                              discrete_support_region);
            updateDepthFrameQualityAfterConsistency(_depthFrames[i],
                                                    depthI,
                                                    restoration_confidence,
                                                    _effectiveSceneProfile,
                                                    _effectiveDepthFilterMode,
                                                    true,
                                                    adaptive_summary,
                                                    discrete_summary);
            captureStageSnapshot(i,
                                 MvsStageSnapshotStage::CrossViewConsistency,
                                 QStringLiteral("after_cross_view_filter_and_repair_before_confidence_postprocess"),
                                 _depthFrames[i],
                                 depthI,
                                 restoration_confidence,
                                 depthI > 0.0f);
            progressChanged(QStringLiteral("多视一致性：已处理 %1/%2，单帧耗时 %3 秒")
                                .arg(i + 1)
                                .arg(NV)
                                .arg(frame_elapsed_ms / 1000.0, 0, 'f', 1),
                            static_cast<float>(i + 1) / static_cast<float>(NV));
        }
    }
} // namespace xjw::mvs
