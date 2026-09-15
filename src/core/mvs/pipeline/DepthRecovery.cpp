#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    void MvsPipelineService::recoverResidualDepthAfterConsistency()
    {
        const int view_count = static_cast<int>(_views.size());
        const bool reliability_guided_local_pass =
            _config.enableDepthLayerReliabilityGuidedCorrection && _effectiveSceneProfile == MvsSceneProfile::Custom;
        if (!_config.enablePostConsistencyResidualReestimation ||
            (_effectiveSceneProfile != MvsSceneProfile::OrbitalObject && !reliability_guided_local_pass) ||
            view_count < 4 || static_cast<int>(_depthFrames.size()) != view_count)
        {
            return;
        }
        if (!_imageCache)
        {
            LOG_WARN("[MVS][残余重估] 图像缓存不可用，跳过一致性后局部重估");
            return;
        }

        std::vector<cv::Mat> frozen_depths(static_cast<std::size_t>(view_count));
        std::vector<cv::Mat> frozen_confidences(static_cast<std::size_t>(view_count));
        for (int frame_index = 0; frame_index < view_count; ++frame_index)
        {
            if (_depthFrames[frame_index].eligibleAsConsistencySource() && _depthFrames[frame_index].depthMap &&
                !_depthFrames[frame_index].depthMap->empty())
            {
                // Recovery writes only to pending results until all frame workers
                // have joined, so cv::Mat's shared immutable view is a sufficient
                // snapshot here; cloning every full-resolution frame is redundant.
                frozen_depths[static_cast<std::size_t>(frame_index)] = *_depthFrames[frame_index].depthMap;
                if (_depthFrames[frame_index].confidence && !_depthFrames[frame_index].confidence->empty())
                {
                    frozen_confidences[static_cast<std::size_t>(frame_index)] = *_depthFrames[frame_index].confidence;
                }
            }
        }

        struct PendingResidualRecovery
        {
            cv::Mat depth;
            cv::Mat confidence;
            cv::Mat recoveredMask;
            cv::Mat geometrySourceMask;
            cv::Mat sourceInverseDepthSum;
            cv::Mat sourceInverseDepthSquaredSum;
            cv::Mat confirmedSourceCount;
            DepthResidualReestimationStats stats;
        };
        std::vector<PendingResidualRecovery> pending(static_cast<std::size_t>(view_count));
        DepthResidualReestimationOptions base_options{
            64,
            0.001f,
            4,
            2,
            2,
            std::max(0.0f, _config.postConsistencyResidualMaximumLayerSpread),
            std::clamp(_config.postConsistencyResidualMaximumPriorRadius, 0.005f, 0.25f),
            std::clamp(_config.postConsistencyResidualConfidence, 0.0f, 1.0f)};
        if (reliability_guided_local_pass)
        {
            base_options.minimumLayerSourceCount = 3;
            base_options.minimumLayerSectorCount = 2;
            base_options.minimumGeometryConfirmationCount = 3;
            base_options.minimumGeometrySectorCount = 2;
            base_options.allowValidDepthReplacement = true;
            base_options.minimumReplacementCostAdvantage = 0.08f;
            base_options.ambiguousMaximumRelativeCorrection = 0.01f;
            base_options.minimumCandidateConfidence = std::min(base_options.minimumCandidateConfidence, 0.18f);
        }
        const int residual_frame_workers = std::clamp(_config.gpuFrameWorkerCount, 1, std::min(2, view_count));
        const int row_workers = std::max(1, resolvedTotalCpuThreadBudget(_config) / residual_frame_workers);
        parallelForRows(
            view_count,
            residual_frame_workers,
            [&](int frame_index)
            {
                if (_cancelled.load())
                {
                    return;
                }
                DepthFrameResult& frame = _depthFrames[frame_index];
                PendingResidualRecovery& recovery = pending[static_cast<std::size_t>(frame_index)];
                if (!frame.eligibleForConsistencyCheck() || !frame.depthMap || frame.depthMap->empty())
                {
                    recovery.stats.skippedReason = QStringLiteral("frame_unavailable");
                    return;
                }
                std::string referenceImageError;
                MvsImageCache::ImageLease referenceImageLease = acquireImageFrame(frame_index, &referenceImageError);
                if (!referenceImageLease)
                {
                    recovery.stats.skippedReason = QStringLiteral("reference_image_unavailable");
                    LOG_WARN(QStringLiteral("[MVS][帧 %1][残余重估] 参考影像不可用：%2")
                                 .arg(frame_index)
                                 .arg(QString::fromStdString(referenceImageError)));
                    return;
                }
                const FramePinholeCamera reference_camera =
                    frame.cameraModel.isValid() ? frame.cameraModel : mvsPinholeCamera(_views[frame_index].camera);
                const std::vector<int> source_indices =
                    reliability_guided_local_pass
                        ? frame.geometrySourceViewIndices
                        : orbitalHoleRepairSourceIndices(
                              _depthFrames,
                              consistencySourceIndicesForFrame(_depthFrames, frame_index, view_count),
                              frame_index,
                              view_count,
                              _config.postConsistencyResidualSourceCount);
                cv::Mat support_mask = frame.supportRegionMask && !frame.supportRegionMask->empty()
                                           ? *frame.supportRegionMask
                                           : cv::Mat(frame.depthMap->size(), CV_8UC1, cv::Scalar(255));
                DepthResidualReestimationPreflight preflight;
                if (reliability_guided_local_pass && frame.depthLayerReliabilityClass && frame.geometryRerankMaps &&
                    frame.geometryRerankMaps->compatible(frame.depthMap->size()))
                {
                    cv::Mat ambiguous;
                    cv::Mat rejected;
                    cv::compare(*frame.depthLayerReliabilityClass,
                                static_cast<std::uint8_t>(DepthLayerReliabilityClass::AmbiguousLowTexture),
                                ambiguous,
                                cv::CMP_EQ);
                    cv::compare(*frame.depthLayerReliabilityClass,
                                static_cast<std::uint8_t>(DepthLayerReliabilityClass::RejectedLayer),
                                rejected,
                                cv::CMP_EQ);
                    cv::Mat weak_mask;
                    cv::bitwise_or(ambiguous, rejected, weak_mask);
                    cv::bitwise_and(weak_mask,
                                    frame.geometryRerankMaps->decisionAction ==
                                        static_cast<std::uint8_t>(DepthGeometryHypothesisAction::None),
                                    weak_mask);
                    preflight = inspectDepthReestimationMask(support_mask, weak_mask, base_options);
                }
                else
                {
                    preflight = inspectDepthResidualReestimationNeed(*frame.depthMap, support_mask, base_options);
                }
                recovery.stats.supportPixelCount = preflight.supportPixelCount;
                recovery.stats.requestedResidualPixelCount = preflight.requestedResidualPixelCount;
                recovery.stats.skippedReason = preflight.skippedReason;
                if (!preflight.shouldProjectSources)
                {
                    return;
                }
                std::vector<cv::Mat> projected_sources;
                std::vector<ProjectedDepthEvidence> projected_source_evidence;
                std::vector<int> source_sector_ids;
                std::vector<cv::Mat> source_images;
                std::vector<FramePinholeCamera> source_cameras;
                std::vector<cv::Mat> source_masks;
                std::vector<MvsImageCache::ImageLease> sourceImageLeases;
                sourceImageLeases.reserve(source_indices.size());
                for (const int source_index : source_indices)
                {
                    if (source_index < 0 || source_index >= view_count ||
                        frozen_depths[static_cast<std::size_t>(source_index)].empty())
                    {
                        continue;
                    }
                    std::string sourceImageError;
                    MvsImageCache::ImageLease sourceImageLease = acquireImageFrame(source_index, &sourceImageError);
                    if (!sourceImageLease)
                    {
                        LOG_WARN(QStringLiteral("[MVS][帧 %1][残余重估] 源影像 %2 不可用：%3")
                                     .arg(frame_index)
                                     .arg(source_index)
                                     .arg(QString::fromStdString(sourceImageError)));
                        continue;
                    }
                    const FramePinholeCamera source_camera = _depthFrames[source_index].cameraModel.isValid()
                                                                 ? _depthFrames[source_index].cameraModel
                                                                 : mvsPinholeCamera(_views[source_index].camera);
                    const int sector = cameraBaselineSector(reference_camera, source_camera);
                    if (reliability_guided_local_pass &&
                        !frozen_confidences[static_cast<std::size_t>(source_index)].empty())
                    {
                        ProjectedDepthEvidence evidence = projectSourceDepthEvidenceToReference(
                            frozen_depths[static_cast<std::size_t>(source_index)],
                            frozen_confidences[static_cast<std::size_t>(source_index)],
                            source_camera,
                            reference_camera,
                            frame.depthMap->size(),
                            0.8f,
                            sector,
                            nullptr,
                            row_workers,
                            &_cancelled);
                        projected_sources.push_back(evidence.depth);
                        projected_source_evidence.push_back(std::move(evidence));
                    }
                    else
                    {
                        projected_sources.push_back(
                            projectSourceDepthToReference(frozen_depths[static_cast<std::size_t>(source_index)],
                                                          source_camera,
                                                          reference_camera,
                                                          frame.depthMap->size(),
                                                          0.8f,
                                                          nullptr,
                                                          row_workers,
                                                          &_cancelled));
                        if (reliability_guided_local_pass)
                        {
                            ProjectedDepthEvidence unavailable_evidence;
                            unavailable_evidence.baselineSector = sector;
                            projected_source_evidence.push_back(std::move(unavailable_evidence));
                        }
                    }
                    source_sector_ids.push_back(sector);
                    source_images.push_back(sourceImageLease->preparedGray);
                    source_cameras.push_back(source_camera);
                    source_masks.push_back(_depthFrames[source_index].supportRegionMask &&
                                                   !_depthFrames[source_index].supportRegionMask->empty()
                                               ? *_depthFrames[source_index].supportRegionMask
                                               : cv::Mat(source_images.back().size(), CV_8UC1, cv::Scalar(255)));
                    sourceImageLeases.push_back(std::move(sourceImageLease));
                }
                const DepthResidualReestimationTarget target =
                    buildDepthResidualReestimationTarget(*frame.depthMap,
                                                         support_mask,
                                                         projected_sources,
                                                         source_sector_ids,
                                                         base_options,
                                                         std::move(preflight));
                recovery.stats.supportPixelCount = target.supportPixelCount;
                recovery.stats.requestedResidualPixelCount = target.requestedResidualPixelCount;
                recovery.stats.layerCoveredPixelCount = target.layerCoveredPixelCount;
                recovery.stats.insufficientSourcePixelCount = target.insufficientSourcePixelCount;
                recovery.stats.insufficientSectorPixelCount = target.insufficientSectorPixelCount;
                recovery.stats.layerSpreadRejectedPixelCount = target.layerSpreadRejectedPixelCount;
                recovery.stats.sourceCount = static_cast<int>(source_images.size());
                recovery.stats.skippedReason = target.skippedReason;
                if (!target.valid || source_images.size() < 4)
                {
                    if (recovery.stats.skippedReason.isEmpty())
                    {
                        recovery.stats.skippedReason = QStringLiteral("insufficient_patchmatch_sources");
                    }
                    return;
                }

                const std::vector<std::vector<int>> source_groups =
                    buildDepthResidualPatchMatchSourceGroups(source_sector_ids);
                std::vector<cv::Mat> candidate_depths;
                std::vector<cv::Mat> candidate_confidences;
                recovery.stats.attemptedHypothesisCount = 2;
                double minimum_hint = 0.0;
                double maximum_hint = 0.0;
                cv::minMaxLoc(target.hintDepth, &minimum_hint, &maximum_hint, nullptr, nullptr, target.residualMask);
                const float z_near = std::max(1.0e-4f, static_cast<float>(minimum_hint * 0.90));
                const float z_far = std::max(z_near * 1.01f, static_cast<float>(maximum_hint * 1.10));
                for (const std::vector<int>& source_group : source_groups)
                {
                    std::vector<cv::Mat> group_images;
                    std::vector<FramePinholeCamera> group_cameras;
                    std::vector<cv::Mat> group_masks;
                    for (const int source_ordinal : source_group)
                    {
                        group_images.push_back(source_images[static_cast<std::size_t>(source_ordinal)]);
                        group_cameras.push_back(source_cameras[static_cast<std::size_t>(source_ordinal)]);
                        group_masks.push_back(source_masks[static_cast<std::size_t>(source_ordinal)]);
                    }
                    PatchMatchConfig patch_match = patchMatchConfigForRecordedWorker(_config.patchMatch, frame.device);
                    patch_match.numIterations = std::clamp(std::max(6, patch_match.numIterations / 2), 6, 10);
                    patch_match.patchHalf = std::max(3, patch_match.patchHalf - 1);
                    patch_match.confidenceThresh =
                        std::min(patch_match.confidenceThresh, reliability_guided_local_pass ? 0.10f : 0.18f);
                    patch_match.minimumMaskedPatchSupportRatio =
                        std::min(patch_match.minimumMaskedPatchSupportRatio, 0.25f);
                    patch_match.geomConsistency = false;
                    patch_match.cancelFlag = &_cancelled;
                    cv::Mat candidate_depth;
                    cv::Mat candidate_confidence;
                    std::string error_message;
                    if (estimatePatchMatchWithAdaptiveCuda("post-consistency residual PatchMatch",
                                                           frame_index,
                                                           referenceImageLease->preparedGray,
                                                           group_images,
                                                           reference_camera,
                                                           group_cameras,
                                                           z_near,
                                                           z_far,
                                                           patch_match,
                                                           candidate_depth,
                                                           &candidate_confidence,
                                                           &error_message,
                                                           &target.hintDepth,
                                                           &target.hintRadius,
                                                           &target.estimationMask,
                                                           &group_masks))
                    {
                        candidate_depths.push_back(std::move(candidate_depth));
                        candidate_confidences.push_back(std::move(candidate_confidence));
                    }
                    else
                    {
                        LOG_WARN("[MVS][帧 %d][残余重估] PatchMatch 组失败: %s", frame_index, error_message.c_str());
                    }
                }
                recovery.stats.successfulHypothesisCount = static_cast<int>(candidate_depths.size());
                recovery.stats.failedHypothesisCount = 2 - recovery.stats.successfulHypothesisCount;
                if (candidate_depths.size() != 2)
                {
                    recovery.stats.attempted = true;
                    recovery.stats.skippedReason = QStringLiteral("incomplete_hypothesis_pair");
                    return;
                }
                recovery.depth = frame.depthMap->clone();
                recovery.confidence = frame.confidence && !frame.confidence->empty()
                                          ? frame.confidence->clone()
                                          : cv::Mat(frame.depthMap->size(), CV_32FC1, cv::Scalar(0.0f));
                recovery.geometrySourceMask = cv::Mat(frame.depthMap->size(), CV_16UC1, cv::Scalar(0));
                recovery.sourceInverseDepthSum = cv::Mat(frame.depthMap->size(), CV_32FC1, cv::Scalar(0.0f));
                recovery.sourceInverseDepthSquaredSum = cv::Mat(frame.depthMap->size(), CV_32FC1, cv::Scalar(0.0f));
                recovery.confirmedSourceCount = cv::Mat(frame.depthMap->size(), CV_16UC1, cv::Scalar(0));
                DepthResidualReestimationEvidenceOutputs evidence_outputs;
                evidence_outputs.geometrySourceMask = &recovery.geometrySourceMask;
                evidence_outputs.sourceInverseDepthSum = &recovery.sourceInverseDepthSum;
                evidence_outputs.sourceInverseDepthSquaredSum = &recovery.sourceInverseDepthSquaredSum;
                evidence_outputs.confirmedSourceCount = &recovery.confirmedSourceCount;
                evidence_outputs.rerankMaps = reliability_guided_local_pass ? frame.geometryRerankMaps.data() : nullptr;
                recovery.stats = mergeDepthResidualReestimationCandidates(
                    recovery.depth,
                    recovery.confidence,
                    candidate_depths,
                    candidate_confidences,
                    target,
                    projected_sources,
                    source_sector_ids,
                    &recovery.recoveredMask,
                    base_options,
                    reliability_guided_local_pass ? frame.depthLayerReliabilityClass.data() : nullptr,
                    reliability_guided_local_pass ? &projected_source_evidence : nullptr,
                    reliability_guided_local_pass ? &evidence_outputs : nullptr);
                recovery.stats.attemptedHypothesisCount = 2;
                recovery.stats.successfulHypothesisCount = 2;
                recovery.stats.sourceCount = static_cast<int>(source_images.size());
                LOG_INFO("[MVS][帧 %d][残余重估] target=%d layer=%d candidate=%d "
                         "consensus=%d recovered=%d",
                         frame_index,
                         recovery.stats.requestedResidualPixelCount,
                         recovery.stats.layerCoveredPixelCount,
                         recovery.stats.candidatePixelCount,
                         recovery.stats.consensusCandidatePixelCount,
                         recovery.stats.recoveredPixelCount);
            });

        if (_cancelled.load())
        {
            return;
        }
        for (int frame_index = 0; frame_index < view_count; ++frame_index)
        {
            DepthFrameResult& frame = _depthFrames[frame_index];
            PendingResidualRecovery& recovery = pending[static_cast<std::size_t>(frame_index)];
            frame.residualReestimationDiagnostics = depthResidualReestimationStatsToJson(recovery.stats);
            if (recovery.stats.recoveredPixelCount <= 0 || recovery.depth.empty())
            {
                continue;
            }
            *frame.depthMap = std::move(recovery.depth);
            if (!frame.confidence)
            {
                frame.confidence = QSharedPointer<cv::Mat>::create();
            }
            *frame.confidence = std::move(recovery.confidence);
            frame.residualReestimatedMask = QSharedPointer<cv::Mat>::create(std::move(recovery.recoveredMask));
            if (!frame.depthProvenance || frame.depthProvenance->empty())
            {
                frame.depthProvenance = QSharedPointer<cv::Mat>::create(initializeDepthProvenance(
                    *frame.depthMap, frame.targetedGapRecoveredMask ? *frame.targetedGapRecoveredMask : cv::Mat()));
            }
            updateDepthProvenance(*frame.depthProvenance,
                                  *frame.depthMap,
                                  frame.targetedGapRecoveredMask ? *frame.targetedGapRecoveredMask : cv::Mat(),
                                  frame.crossViewRepairedMask ? *frame.crossViewRepairedMask : cv::Mat(),
                                  cv::Mat(),
                                  *frame.residualReestimatedMask);
            if (reliability_guided_local_pass && frame.geometrySourceMask &&
                frame.geometrySourceMask->size() == frame.residualReestimatedMask->size())
            {
                recovery.geometrySourceMask.copyTo(*frame.geometrySourceMask, *frame.residualReestimatedMask);
            }
            if (frame.geometrySupportCount &&
                frame.geometrySupportCount->size() == frame.residualReestimatedMask->size())
            {
                cv::Mat retained_support;
                recovery.confirmedSourceCount.convertTo(retained_support, CV_16UC1, 1.0, 1.0);
                retained_support.copyTo(*frame.geometrySupportCount, *frame.residualReestimatedMask);
            }
            if (frame.inverseDepthMean && frame.inverseDepthMean->size() == frame.depthMap->size())
            {
                cv::Mat source_count_float;
                recovery.confirmedSourceCount.convertTo(source_count_float, CV_32FC1);
                cv::Mat recovered_inverse_depth_mean;
                cv::divide(
                    recovery.sourceInverseDepthSum, source_count_float, recovered_inverse_depth_mean, 1.0, CV_32FC1);
                recovered_inverse_depth_mean.copyTo(*frame.inverseDepthMean, *frame.residualReestimatedMask);
            }
            if (frame.inverseDepthRelativeSpread && frame.inverseDepthRelativeSpread->size() == frame.depthMap->size())
            {
                cv::Mat source_count_float;
                recovery.confirmedSourceCount.convertTo(source_count_float, CV_32FC1);
                cv::Mat inverse_mean;
                cv::Mat inverse_squared_mean;
                cv::divide(recovery.sourceInverseDepthSum, source_count_float, inverse_mean, 1.0, CV_32FC1);
                cv::divide(
                    recovery.sourceInverseDepthSquaredSum, source_count_float, inverse_squared_mean, 1.0, CV_32FC1);
                cv::Mat variance = inverse_squared_mean - inverse_mean.mul(inverse_mean);
                cv::max(variance, 0.0f, variance);
                cv::sqrt(variance, variance);
                cv::divide(variance, inverse_mean, variance);
                variance.copyTo(*frame.inverseDepthRelativeSpread, *frame.residualReestimatedMask);
            }
        }
    }
} // namespace xjw::mvs
