#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    void MvsPipelineService::prepareFrameCaches()
    {
        if (_cancelled.load(std::memory_order_relaxed))
        {
            clearFrameCaches();
            return;
        }
        if (_frameCachesReady)
        {
            return;
        }

        const int NV = static_cast<int>(_views.size());
        const size_t pointCount = _sparse.points.size();
        _frameCaches.assign(static_cast<size_t>(std::max(0, NV)), FrameMvsCache{});
        _visibilityWordCount = 0;
        _visibilityBits.clear();
        _visibilityAdjacency.assign(static_cast<size_t>(std::max(0, NV)), {});

        if (NV <= 0 || pointCount == 0)
        {
            _frameCachesReady = true;
            return;
        }

        const auto start = Clock::now();
        constexpr size_t kParallelVisibilityPointThreshold = 20000;
        std::vector<std::string> activeImagePaths;
        std::vector<QString> activeImagePathKeys;
        activeImagePaths.reserve(_views.size());
        activeImagePathKeys.reserve(_views.size());
        for (const CameraView& view : _views)
        {
            activeImagePaths.push_back(view.imagePath);
            activeImagePathKeys.push_back(normalizedMvsPathKey(view.imagePath));
        }
        const std::vector<MvsSourcePairQuality> activePairQualities =
            filterMvsSourcePairQualitiesForImages(_config.sourcePairQualities, activeImagePaths);
        const MvsSourcePairQualityLookup pairQualityLookup = buildMvsSourcePairQualityLookup(activePairQualities);

        MvsVisibilityGraphBuildOptions visibilityOptions;
        const int hardwareThreadCount = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
        const int requestedThreadCount = resolvedTotalCpuThreadBudget(_config);
        visibilityOptions.workerCount = pointCount >= kParallelVisibilityPointThreshold && NV > 1
                                            ? std::clamp(std::min(requestedThreadCount, hardwareThreadCount), 1, 16)
                                            : 1;
        visibilityOptions.cancelFlag = &_cancelled;

        std::unordered_map<std::string, int> viewIndexByPathKey;
        viewIndexByPathKey.reserve(activeImagePathKeys.size());
        for (int viewIndex = 0; viewIndex < NV; ++viewIndex)
        {
            viewIndexByPathKey.emplace(activeImagePathKeys[static_cast<std::size_t>(viewIndex)].toStdString(),
                                       viewIndex);
        }
        for (const MvsSourcePairQuality& quality : activePairQualities)
        {
            if (!quality.verified)
            {
                continue;
            }
            const auto first = viewIndexByPathKey.find(normalizedMvsPathKey(quality.imageA).toStdString());
            const auto second = viewIndexByPathKey.find(normalizedMvsPathKey(quality.imageB).toStdString());
            if (first != viewIndexByPathKey.end() && second != viewIndexByPathKey.end())
            {
                visibilityOptions.requiredPairs.push_back({first->second, second->second});
            }
        }

        MvsVisibilityGraph visibilityGraph = MvsVisibilityGraphBuilder::build(_views, _sparse, visibilityOptions);
        if (visibilityGraph.cancelled || _cancelled.load(std::memory_order_relaxed))
        {
            clearFrameCaches();
            return;
        }
        _visibilityWordCount = visibilityGraph.statistics.visibilityWordCount;
        _visibilityBits = std::move(visibilityGraph.visibilityBits);
        _visibilityAdjacency = std::move(visibilityGraph.neighborsByView);
        for (int viewIndex = 0; viewIndex < NV; ++viewIndex)
        {
            _frameCaches[static_cast<std::size_t>(viewIndex)].visiblePointIndices =
                std::move(visibilityGraph.visiblePointIndicesByView[static_cast<std::size_t>(viewIndex)]);
        }
        _frameCachesReady = true;

        const bool hasSourcePairQuality = !pairQualityLookup.qualitiesByPairKey.empty();
        const bool requireVerifiedSourcePairs = _config.requireVerifiedSourcePairs && hasSourcePairQuality;
        const int minVerifiedPairInliers = std::max(1, _config.minSourcePairGeometricInliers);
        int referenceTrackFallbackCount = 0;
        if (_config.requireVerifiedSourcePairs && !_config.sourcePairQualities.empty() && !hasSourcePairQuality)
        {
            LOG_WARN(QStringLiteral("[MVS] 配置中的 %1 个几何验证对均不属于当前 %2 张影像，"
                                    "已忽略旧引用并回退到当前空三稀疏轨迹")
                         .arg(_config.sourcePairQualities.size())
                         .arg(NV));
        }

        auto sampledMedianAngle = [this, NV](int refIdx, int sourceIdx) -> float
        {
            if (refIdx < 0 || refIdx >= NV || sourceIdx < 0 || sourceIdx >= NV)
            {
                return 0.f;
            }

            constexpr size_t kMaxAngleSamples = 2048;
            std::vector<float> angles;
            angles.reserve(kMaxAngleSamples);
            for (size_t pointIndex : _frameCaches[static_cast<size_t>(refIdx)].visiblePointIndices)
            {
                if (!isSparsePointVisibleInFrame(sourceIdx, pointIndex))
                {
                    continue;
                }
                const float angle =
                    mvsTriangulationAngleDeg(_views[refIdx], _views[sourceIdx], _sparse.points[pointIndex]);
                if (angle <= 0.0f)
                {
                    continue;
                }
                angles.push_back(angle);
                if (angles.size() >= kMaxAngleSamples)
                {
                    break;
                }
            }

            if (angles.empty())
            {
                return 0.f;
            }

            const auto mid = angles.begin() + static_cast<long>(angles.size() / 2);
            std::nth_element(angles.begin(), mid, angles.end());
            return *mid;
        };

        for (int refIdx = 0; refIdx < NV; ++refIdx)
        {
            const int desiredSourceCount = std::max(1, _config.numSourceViews);
            const size_t refVisibleCount = _frameCaches[static_cast<size_t>(refIdx)].visiblePointIndices.size();

            MvsSourcePlannerOptions plannerOptions;
            plannerOptions.refIndex = refIdx;
            plannerOptions.viewCount = NV;
            plannerOptions.maxSources = desiredSourceCount;
            plannerOptions.sourceAngleSoftRankingStrength = _config.sourceAngleSoftRankingStrength;
            plannerOptions.auditSourceRanking = _config.evaluateCompleteVisibilityCandidatePool;
            plannerOptions.rejectAngleOutliers = true;
            plannerOptions.minTriangulationAngleDeg = 0.2f;
            const float recommended_maximum_angle =
                recommendedMvsSourceMaximumAngleDeg(_effectiveSceneProfile, desiredSourceCount);
            plannerOptions.maxTriangulationAngleDeg =
                constrainMvsSourceMaximumAngleDeg(recommended_maximum_angle, _config.sourceMaximumAngleDegCap);
            struct RankedSourceCandidate
            {
                int viewIndex = -1;
                int commonVisiblePoints = 0;
                int sequenceDistance = 0;
                bool verifiedPair = false;
            };

            std::vector<RankedSourceCandidate> rankedSourceCandidates;
            const auto& visibilityNeighbors = _visibilityAdjacency[static_cast<std::size_t>(refIdx)];
            rankedSourceCandidates.reserve(visibilityNeighbors.size());
            for (const MvsVisibilityNeighbor& neighbor : visibilityNeighbors)
            {
                const MvsSourcePairQuality* pairQuality =
                    pairQualityLookup.findNormalized(activeImagePathKeys[static_cast<std::size_t>(refIdx)],
                                                     activeImagePathKeys[static_cast<std::size_t>(neighbor.viewIndex)]);
                const bool verifiedPair = pairQuality && pairQuality->verified;
                if (neighbor.sharedTrackCount <= 0 && !verifiedPair)
                {
                    continue;
                }

                rankedSourceCandidates.push_back({neighbor.viewIndex,
                                                  neighbor.sharedTrackCount,
                                                  std::abs(neighbor.viewIndex - refIdx),
                                                  verifiedPair});
            }

            std::sort(rankedSourceCandidates.begin(),
                      rankedSourceCandidates.end(),
                      [](const RankedSourceCandidate& a, const RankedSourceCandidate& b)
                      {
                          if (a.verifiedPair != b.verifiedPair)
                          {
                              return a.verifiedPair;
                          }
                          if (a.commonVisiblePoints != b.commonVisiblePoints)
                          {
                              return a.commonVisiblePoints > b.commonVisiblePoints;
                          }
                          if (a.sequenceDistance != b.sequenceDistance)
                          {
                              return a.sequenceDistance < b.sequenceDistance;
                          }
                          return a.viewIndex < b.viewIndex;
                      });

            bool referenceHasPairQuality = false;
            for (const RankedSourceCandidate& candidate : rankedSourceCandidates)
            {
                if (pairQualityLookup.findNormalized(
                        activeImagePathKeys[static_cast<std::size_t>(refIdx)],
                        activeImagePathKeys[static_cast<std::size_t>(candidate.viewIndex)]))
                {
                    referenceHasPairQuality = true;
                    break;
                }
            }
            const bool requireVerifiedSourcePairsForReference = requireVerifiedSourcePairs && referenceHasPairQuality;
            if (requireVerifiedSourcePairsForReference)
            {
                plannerOptions.minGeometricInliers = minVerifiedPairInliers;
                plannerOptions.allowWeakKnownOverlap = false;
                plannerOptions.requireVerifiedPairGeometry = true;
                plannerOptions.allowSequenceFallback = false;
            }
            else if (_config.numSourceViews >= 5 && _config.fusion.minConsistentViews >= 3)
            {
                plannerOptions.minSharedTracks = 20;
                plannerOptions.minGeometricInliers = 20;
                plannerOptions.minSourceQualityScore = 0.35f;
                plannerOptions.allowWeakKnownOverlap = false;
            }
            if (requireVerifiedSourcePairs && !referenceHasPairQuality)
            {
                ++referenceTrackFallbackCount;
            }

            std::vector<MvsSourceCandidate> candidates;
            candidates.reserve(rankedSourceCandidates.size());
            int provenSourceCount = 0;
            int currentSourceScoreCutoff = -1;
            int legacyEvaluatedCandidateCount = 0;
            int completeEvaluatedCandidateCount = 0;
            bool legacyCandidatePoolClosed = false;
            std::vector<float> legacyCandidateAngles;
            legacyCandidateAngles.reserve(rankedSourceCandidates.size());
            for (const RankedSourceCandidate& candidate : rankedSourceCandidates)
            {
                // remaining candidates are sorted by common count; once enough stronger sources are proven,
                // avoid spending more time sampling triangulation angles for weaker candidates.
                if (!legacyCandidatePoolClosed && provenSourceCount >= desiredSourceCount &&
                    candidate.commonVisiblePoints <= currentSourceScoreCutoff)
                {
                    legacyCandidatePoolClosed = true;
                    if (!_config.evaluateCompleteVisibilityCandidatePool)
                    {
                        break;
                    }
                }
                const bool legacy_evaluated_candidate = !legacyCandidatePoolClosed;

                const float medianAngle = sampledMedianAngle(refIdx, candidate.viewIndex);
                MvsSourceCandidate sourceCandidate;
                sourceCandidate.viewIndex = candidate.viewIndex;
                sourceCandidate.sharedTracks = candidate.commonVisiblePoints;
                const MvsSourcePairQuality* pairQuality = pairQualityLookup.findNormalized(
                    activeImagePathKeys[static_cast<std::size_t>(refIdx)],
                    activeImagePathKeys[static_cast<std::size_t>(candidate.viewIndex)]);
                const bool pairVerificationFailed =
                    pairQuality && pairQuality->hasVerificationStatistics && !pairQuality->verified;
                const bool pairVerificationMissing = pairQuality && !pairQuality->hasVerificationStatistics;
                sourceCandidate.geometricInliers = pairQuality && pairQuality->hasVerificationStatistics
                                                       ? std::max(0, pairQuality->geometricInliers)
                                                       : candidate.commonVisiblePoints;
                sourceCandidate.verifiedPairGeometry =
                    pairQuality && pairQuality->verified && pairQuality->geometricInliers > 0;
                sourceCandidate.verificationStatus =
                    sourceCandidate.verifiedPairGeometry
                        ? MvsSourceVerificationStatus::Verified
                        : (pairVerificationFailed ? MvsSourceVerificationStatus::Failed
                                                  : (pairVerificationMissing || referenceHasPairQuality
                                                         ? MvsSourceVerificationStatus::MissingStatistics
                                                         : MvsSourceVerificationStatus::NotRequested));
                sourceCandidate.pairTotalMatches = pairQuality ? std::max(0, pairQuality->totalMatches) : 0;
                sourceCandidate.pairCoverageScore = pairQuality ? pairQuality->geometricCoverage : 0.0f;
                sourceCandidate.verificationReason =
                    pairQuality ? pairQuality->verificationReason
                                : (referenceHasPairQuality ? "pair_not_present_in_current_catalog"
                                                           : "pair_verification_not_requested");
                sourceCandidate.medianTriangulationAngleDeg = medianAngle;
                sourceCandidate.coverageScore = refVisibleCount > 0
                                                    ? std::clamp(static_cast<float>(candidate.commonVisiblePoints) /
                                                                     static_cast<float>(refVisibleCount),
                                                                 0.0f,
                                                                 1.0f)
                                                    : 0.0f;
                sourceCandidate.baselineScore = std::clamp(medianAngle / 20.0f, 0.0f, 1.0f);
                sourceCandidate.sequenceDistance = candidate.sequenceDistance;
                sourceCandidate.knownOverlap = !pairVerificationFailed && (candidate.commonVisiblePoints > 0 ||
                                                                           sourceCandidate.verifiedPairGeometry);
                candidates.push_back(sourceCandidate);
                ++completeEvaluatedCandidateCount;
                if (legacy_evaluated_candidate)
                {
                    ++legacyEvaluatedCandidateCount;
                    legacyCandidateAngles.push_back(medianAngle);
                }
                const bool hasRequiredPairQuality =
                    !plannerOptions.requireVerifiedPairGeometry ||
                    (sourceCandidate.verifiedPairGeometry &&
                     sourceCandidate.geometricInliers >= plannerOptions.minGeometricInliers);
                if (legacy_evaluated_candidate && hasRequiredPairQuality &&
                    medianAngle >= plannerOptions.minTriangulationAngleDeg &&
                    medianAngle <= plannerOptions.maxTriangulationAngleDeg)
                {
                    ++provenSourceCount;
                    if (provenSourceCount >= desiredSourceCount)
                    {
                        currentSourceScoreCutoff = candidate.commonVisiblePoints;
                    }
                }
            }
            float scene_maximum_angle = recommended_maximum_angle;
            if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                scene_maximum_angle =
                    adaptiveMvsSourceMaximumAngleDeg(_effectiveSceneProfile, desiredSourceCount, legacyCandidateAngles);
                // Failed SfM pairs are never promoted back into SfM. For a short
                // orbital sequence, a narrowly qualified failed pair may still be
                // useful as PatchMatch source-only evidence. PatchMatch now uses
                // majority support (3 of 4 sources), so a fourth independently
                // qualified direction can reject a mutually consistent wrong
                // layer without letting that single weaker view dominate NCC.
                plannerOptions.allowFailedPairBackfill = true;
                plannerOptions.failedPairBackfillMaximumTotalSources = 4;
                plannerOptions.failedPairBackfillMinimumInliers = 12;
                plannerOptions.failedPairBackfillMinimumMatches = 14;
                plannerOptions.failedPairBackfillMinimumSharedTracks = 20;
                plannerOptions.failedPairBackfillMinimumCoverage = 0.1875f;
                plannerOptions.failedPairBackfillMinimumWilsonLowerBound = 0.50f;
                plannerOptions.failedPairBackfillMaximumAngleDeg = 65.0f;
                plannerOptions.allowStrictFailedPairBackfill = true;
                plannerOptions.strictFailedPairBackfillMinimumInliers = 24;
                plannerOptions.strictFailedPairBackfillMinimumMatches = 32;
                plannerOptions.strictFailedPairBackfillMinimumSharedTracks = 40;
                plannerOptions.strictFailedPairBackfillMinimumCoverage = 0.30f;
                plannerOptions.strictFailedPairBackfillMinimumWilsonLowerBound = 0.65f;
                plannerOptions.strictFailedPairBackfillMaximumAngleDeg = 55.0f;
            }

            plannerOptions.maxTriangulationAngleDeg =
                constrainMvsSourceMaximumAngleDeg(scene_maximum_angle, _config.sourceMaximumAngleDegCap);
            const bool angle_cap_enabled =
                std::isfinite(_config.sourceMaximumAngleDegCap) && _config.sourceMaximumAngleDegCap > 0.0f;
            const bool angle_cap_applied =
                angle_cap_enabled && plannerOptions.maxTriangulationAngleDeg < scene_maximum_angle;
            if (angle_cap_enabled)
            {
                // Sequence fallback has no measured triangulation angle. Letting it
                // refill a cap-induced shortfall would silently bypass the explicit
                // experiment boundary with an unaudited, potentially wider view.
                plannerOptions.allowSequenceFallback = false;
            }
            LOG_DEBUG("[MVS][帧 %d][源视角角度] scene_max=%.3f cap=%.3f "
                      "effective_max=%.3f applied=%s",
                      refIdx,
                      scene_maximum_angle,
                      _config.sourceMaximumAngleDegCap,
                      plannerOptions.maxTriangulationAngleDeg,
                      angle_cap_applied ? "true" : "false");

            const MvsSourcePlan sourcePlan = requireVerifiedSourcePairsForReference
                                                 ? planMvsSourceViewsVerifiedFirst(candidates, plannerOptions)
                                                 : planMvsSourceViews(candidates, plannerOptions);
            MvsSourcePlan legacy_pool_plan;
            bool complete_pool_changed_legacy_plan = false;
            if (_config.evaluateCompleteVisibilityCandidatePool)
            {
                const std::size_t legacy_candidate_count =
                    std::min(candidates.size(), static_cast<std::size_t>(std::max(0, legacyEvaluatedCandidateCount)));
                const std::vector<MvsSourceCandidate> legacy_candidates(candidates.cbegin(),
                                                                        candidates.cbegin() + legacy_candidate_count);
                MvsSourcePlannerOptions legacy_options = plannerOptions;
                legacy_options.sourceAngleSoftRankingStrength = 0.0f;
                legacy_options.auditSourceRanking = false;
                legacy_pool_plan = requireVerifiedSourcePairsForReference
                                       ? planMvsSourceViewsVerifiedFirst(legacy_candidates, legacy_options)
                                       : planMvsSourceViews(legacy_candidates, legacy_options);
                complete_pool_changed_legacy_plan = legacy_pool_plan.selected.size() != sourcePlan.selected.size();
                if (!complete_pool_changed_legacy_plan)
                {
                    for (std::size_t index = 0; index < sourcePlan.selected.size(); ++index)
                    {
                        const MvsSourcePlanEntry& legacy_entry = legacy_pool_plan.selected[index];
                        const MvsSourcePlanEntry& complete_entry = sourcePlan.selected[index];
                        if (legacy_entry.viewIndex != complete_entry.viewIndex ||
                            legacy_entry.tier != complete_entry.tier)
                        {
                            complete_pool_changed_legacy_plan = true;
                            break;
                        }
                    }
                }
            }
            QJsonObject source_angle_diagnostics =
                mvsSourceAngleDiagnosticsToJson(MvsSourceAnglePolicy{_config.sourceMaximumAngleDegCap,
                                                                     scene_maximum_angle,
                                                                     plannerOptions.maxTriangulationAngleDeg,
                                                                     angle_cap_enabled,
                                                                     angle_cap_applied,
                                                                     plannerOptions.allowSequenceFallback},
                                                sourcePlan);
            if (_config.evaluateCompleteVisibilityCandidatePool)
            {
                QJsonArray legacy_selected;
                for (const MvsSourcePlanEntry& entry : legacy_pool_plan.selected)
                {
                    legacy_selected.append(mvsSourcePlanEntryToJson(entry));
                }
                source_angle_diagnostics.insert(QStringLiteral("legacy_pool_selected"), legacy_selected);
                source_angle_diagnostics.insert(QStringLiteral("complete_pool_changed_legacy_plan"),
                                                complete_pool_changed_legacy_plan);
                source_angle_diagnostics.insert(
                    QStringLiteral("soft_ranking"),
                    mvsSourceRankingDiagnosticsToJson(
                        MvsSourceRankingPolicy{true,
                                               NV <= kMvsVisibilityFullPairViewLimit,
                                               NV,
                                               static_cast<int>(rankedSourceCandidates.size()),
                                               legacyEvaluatedCandidateCount,
                                               completeEvaluatedCandidateCount,
                                               _config.sourceAngleSoftRankingStrength,
                                               plannerOptions.softMaxTriangulationAngleDeg,
                                               plannerOptions.maxTriangulationAngleDeg},
                        sourcePlan));
            }
            _frameCaches[static_cast<size_t>(refIdx)].sourceAngleDiagnostics = std::move(source_angle_diagnostics);
            _frameCaches[static_cast<size_t>(refIdx)].completeVisibilityCandidatePoolEnabled =
                _config.evaluateCompleteVisibilityCandidatePool;
            _frameCaches[static_cast<size_t>(refIdx)].completePoolChangedLegacyPlan = complete_pool_changed_legacy_plan;

            auto& sources = _frameCaches[static_cast<size_t>(refIdx)].sourceViewIndices;
            _frameCaches[static_cast<size_t>(refIdx)].sourceViewScores = sourcePlan.selected;
            _frameCaches[static_cast<size_t>(refIdx)].requestedSourceViewCount = desiredSourceCount;
            _frameCaches[static_cast<size_t>(refIdx)].sourceViewShortfall = sourcePlan.sourceViewShortfall;
            if (sourcePlan.sourceViewShortfall > 0)
            {
                bool verification_failed = false;
                bool verification_missing = false;
                bool angle_rejected = false;
                bool evidence_missing = false;
                for (const MvsSourceRejectedCandidate& rejected : sourcePlan.rejected)
                {
                    verification_failed = verification_failed ||
                                          rejected.candidate.verificationStatus == MvsSourceVerificationStatus::Failed;
                    verification_missing = verification_missing || rejected.candidate.verificationStatus ==
                                                                       MvsSourceVerificationStatus::MissingStatistics;
                    angle_rejected = angle_rejected || rejected.reason == MvsSourceRejectReason::TriangulationAngle;
                    evidence_missing = evidence_missing || rejected.reason == MvsSourceRejectReason::NoEvidence;
                }
                std::string reason = "insufficient_qualified_sources";
                if (verification_failed)
                {
                    reason = "pair_geometry_verification_failed";
                }
                else if (verification_missing)
                {
                    reason = "missing_pair_verification_statistics";
                }
                else if (angle_rejected)
                {
                    reason = "safe_baseline_source_shortfall";
                }
                else if (evidence_missing)
                {
                    reason = "insufficient_overlap_evidence";
                }
                _frameCaches[static_cast<size_t>(refIdx)].sourceViewShortfallReason = std::move(reason);
            }
            sources.reserve(static_cast<size_t>(std::min(NV - 1, desiredSourceCount)));
            for (const auto& score : sourcePlan.selected)
            {
                if (score.score <= 0.f)
                {
                    continue;
                }
                sources.push_back(score.viewIndex);
                if (static_cast<int>(sources.size()) >= desiredSourceCount)
                {
                    break;
                }
            }

            if (sources.empty())
            {
                if (plannerOptions.allowSequenceFallback)
                {
                    const MvsSourcePlan fallbackPlan = planMvsSourceViews({}, plannerOptions);
                    _frameCaches[static_cast<size_t>(refIdx)].sourceViewScores = fallbackPlan.selected;
                    for (const auto& entry : fallbackPlan.selected)
                    {
                        sources.push_back(entry.viewIndex);
                    }
                }
            }

            auto& cache = _frameCaches[static_cast<size_t>(refIdx)];
            cache.sourceSharedPointIndices.reserve(cache.visiblePointIndices.size());
            for (size_t pointIndex : cache.visiblePointIndices)
            {
                for (int sourceIdx : cache.sourceViewIndices)
                {
                    if (sourceIdx < 0 || sourceIdx >= NV || sourceIdx == refIdx)
                    {
                        continue;
                    }
                    if (isSparsePointVisibleInFrame(sourceIdx, pointIndex))
                    {
                        cache.sourceSharedPointIndices.push_back(pointIndex);
                        break;
                    }
                }
            }
        }

        if (referenceTrackFallbackCount > 0)
        {
            LOG_WARN(QStringLiteral("[MVS] %1/%2 个参考帧在当前影像集合中没有几何验证对，"
                                    "已逐帧回退到当前空三稀疏轨迹，避免源视图被旧引用清空")
                         .arg(referenceTrackFallbackCount)
                         .arg(NV));
        }
        LOG_INFO(QStringLiteral("[MVS] MVS 可见性缓存完成: views=%1 points=%2 elapsed=%3 ms")
                     .arg(NV)
                     .arg(static_cast<qulonglong>(pointCount))
                     .arg(elapsedMs(start, Clock::now()), 0, 'f', 1));
    }
} // namespace xjw::mvs
