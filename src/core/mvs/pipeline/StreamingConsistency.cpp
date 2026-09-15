#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    bool MvsPipelineService::crossCheckDepthConsistencyStreaming()
    {
        const int view_count = static_cast<int>(_views.size());
        if (view_count < 2)
        {
            return true;
        }
        if (_consistencyDepthDirectory.empty())
        {
            errorOccurred(QStringLiteral("流式深度一致性检查缺少缓存目录"));
            return false;
        }

        const uint64_t largest_frame_bytes = std::max<uint64_t>(1, largestDepthFrameBytes(_views) / 2ull);
        const SystemMemorySnapshot memory = querySystemMemorySnapshot();
        const uint64_t available_budget =
            memory.valid ? memory.availablePhysicalBytes / 4ull : largest_frame_bytes * 3ull;
        const uint64_t cache_budget =
            std::max<uint64_t>(largest_frame_bytes * 2ull, std::min<uint64_t>(available_budget, 2ull * kBytesPerGiB));
        const QDir storage_dir(QString::fromStdString(_consistencyDepthDirectory));

        DepthConsistencyCache cache(
            [storage_dir](int frame_index, DepthConsistencyFrame& frame, std::string* error_message)
            {
                const QString depth_path = storage_dir.filePath(QStringLiteral("depth_%1.bin").arg(frame_index));
                const xjw::common::OperationResult load_result =
                    xjw::core::project::loadDepthMatStorage(depth_path, &frame.depth);
                if (!load_result.ok)
                {
                    if (error_message)
                    {
                        *error_message = load_result.errorMessage.toStdString();
                    }
                    return false;
                }
                const QString confidence_path =
                    storage_dir.filePath(QStringLiteral("depth_%1_conf.bin").arg(frame_index));
                if (QFileInfo::exists(confidence_path))
                {
                    const xjw::common::OperationResult confidence_result =
                        xjw::core::project::loadDepthMatStorage(confidence_path, &frame.confidence);
                    if (!confidence_result.ok)
                    {
                        frame.confidence.release();
                    }
                }
                frame.frameIndex = frame_index;
                return true;
            },
            static_cast<std::size_t>(cache_budget));

        struct PendingDepthReplacement
        {
            QString originalPath;
            QString filteredPath;
            QString originalConfidencePath;
            QString filteredConfidencePath;
            QString filteredGeometrySupportPath;
            QString filteredGeometrySourceMaskPath;
            QString filteredInverseDepthMeanPath;
            QString filteredInverseDepthSpreadPath;
            QString filteredAdaptiveSupportWeightPath;
            QString filteredAdaptiveEffectiveViewCountPath;
            QString filteredAdaptiveConflictRatioPath;
            QString filteredCrossViewRepairedMaskPath;
            QString filteredResidualReestimatedMaskPath;
            QString filteredDepthProvenancePath;
            QString filteredMissingReasonPath;
            cv::Size expectedDepthSize;
            bool targetedGapRecoveredMaskExpected = false;
            int frameIndex = -1;
        };
        std::vector<PendingDepthReplacement> pending_replacements;
        pending_replacements.reserve(static_cast<std::size_t>(view_count));

        auto remove_pending_files = [&pending_replacements]()
        {
            for (const PendingDepthReplacement& replacement : pending_replacements)
            {
                QFile::remove(replacement.filteredPath);
                QFile::remove(replacement.filteredConfidencePath);
                QFile::remove(replacement.filteredGeometrySupportPath);
                QFile::remove(replacement.filteredGeometrySourceMaskPath);
                QFile::remove(replacement.filteredInverseDepthMeanPath);
                QFile::remove(replacement.filteredInverseDepthSpreadPath);
                if (!replacement.filteredAdaptiveSupportWeightPath.isEmpty())
                {
                    QFile::remove(replacement.filteredAdaptiveSupportWeightPath);
                    QFile::remove(replacement.filteredAdaptiveEffectiveViewCountPath);
                    QFile::remove(replacement.filteredAdaptiveConflictRatioPath);
                }
                QFile::remove(replacement.filteredCrossViewRepairedMaskPath);
                QFile::remove(replacement.filteredResidualReestimatedMaskPath);
                QFile::remove(replacement.filteredDepthProvenancePath);
                QFile::remove(replacement.filteredMissingReasonPath);
            }
        };

        const int row_workers = resolvedTotalCpuThreadBudget(_config);
        const int cross_view_source_count = recommendedMvsCrossViewSourceCount(
            _effectiveSceneProfile, _config.crossViewHoleRepairSourceCount, view_count);
        const std::vector<DepthConsistencyFrameSourcePlan> source_plans = freezeDepthConsistencySourcePlans(
            _depthFrames, view_count, _effectiveSceneProfile, cross_view_source_count);
        int completed_frames = 0;

        for (int frame_index = 0; frame_index < view_count; ++frame_index)
        {
            const auto frame_start = Clock::now();
            if (_cancelled.load())
            {
                remove_pending_files();
                return false;
            }
            if (!_depthFrames[frame_index].eligibleForConsistencyCheck())
            {
                continue;
            }

            std::string load_error;
            const DepthConsistencyCache::FrameHandle reference = cache.acquire(frame_index, &load_error);
            if (!reference || reference->depth.empty())
            {
                remove_pending_files();
                const QString message = QStringLiteral("流式一致性检查读取参考帧 %1 失败：%2")
                                            .arg(frame_index)
                                            .arg(QString::fromStdString(load_error));
                LOG_WARN(QStringLiteral("[MVS] %1").arg(message));
                errorOccurred(message);
                return false;
            }

            std::string referenceImageError;
            MvsImageCache::ImageLease referenceImageLease = acquireImageFrame(frame_index, &referenceImageError);
            const cv::Mat* referenceGray = referenceImageLease ? &referenceImageLease->preparedGray : nullptr;
            if (!referenceImageLease)
            {
                LOG_WARN(QStringLiteral("[MVS][帧 %1][流式一致性] 参考影像不可用，"
                                        "跳过影像引导修复：%2")
                             .arg(frame_index)
                             .arg(QString::fromStdString(referenceImageError)));
            }

            cv::Mat filtered_depth = reference->depth.clone();
            cv::Mat filtered_confidence = reference->confidence.empty() ? cv::Mat() : reference->confidence.clone();
            const QString targeted_recovered_path =
                storage_dir.filePath(QStringLiteral("depth_%1_targeted_gap_recovered_mask.png").arg(frame_index));
            cv::Mat targeted_gap_recovered_mask;
            const bool targeted_gap_recovered_mask_expected =
                _depthFrames[frame_index].targetedGapRecoveredMaskExpected;
            if (targeted_gap_recovered_mask_expected)
            {
                if (!QFileInfo::exists(targeted_recovered_path))
                {
                    remove_pending_files();
                    const QString message = QStringLiteral("流式一致性读取帧 %1 定向恢复掩码失败："
                                                           "权威 checkpoint 工件已丢失 %2")
                                                .arg(frame_index)
                                                .arg(targeted_recovered_path);
                    LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                    errorOccurred(message);
                    return false;
                }
                targeted_gap_recovered_mask = xjw::common::io::readImage(
                    xjw::common::io::toUtf8Path(targeted_recovered_path), cv::IMREAD_GRAYSCALE);
                if (targeted_gap_recovered_mask.empty() || targeted_gap_recovered_mask.type() != CV_8UC1 ||
                    targeted_gap_recovered_mask.size() != filtered_depth.size())
                {
                    remove_pending_files();
                    const QString message = QStringLiteral("流式一致性读取帧 %1 定向恢复掩码失败："
                                                           "%2 的类型或尺寸无效")
                                                .arg(frame_index)
                                                .arg(targeted_recovered_path);
                    LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                    errorOccurred(message);
                    return false;
                }
            }
            else if (QFileInfo::exists(targeted_recovered_path))
            {
                remove_pending_files();
                const QString message = QStringLiteral("流式一致性读取帧 %1 定向恢复掩码失败："
                                                       "checkpoint 声明无该证据，但发现遗留工件 %2")
                                            .arg(frame_index)
                                            .arg(targeted_recovered_path);
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                errorOccurred(message);
                return false;
            }
            const FramePinholeCamera reference_camera = _depthFrames[frame_index].cameraModel.isValid()
                                                            ? _depthFrames[frame_index].cameraModel
                                                            : mvsPinholeCamera(_views[frame_index].camera);
            const DepthConsistencyFrameSourcePlan& source_plan = source_plans[static_cast<std::size_t>(frame_index)];
            const std::vector<int>& source_indices = source_plan.consistencySourceIndices;
            const std::vector<int>& repair_source_indices = source_plan.geometrySourceViewIndices;
            _depthFrames[frame_index].geometrySourceViewIndices = repair_source_indices;
            const float relative_threshold = depthConsistencyRelativeThreshold(
                _effectiveSceneProfile, static_cast<int>(source_indices.size()), _effectiveDepthFilterMode);
            const int minimum_source_confirmations = minimumDepthConsistencySourceConfirmations(
                _effectiveSceneProfile, _effectiveDepthFilterMode, static_cast<int>(source_indices.size()));
            const CrossViewHoleRepairOptions repair_options = orbitalCrossViewHoleRepairOptions(_config);
            cv::Mat consistent_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
            cv::Mat occluded_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
            cv::Mat contradicted_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
            cv::Mat unverifiable_votes(reference->depth.size(), CV_16U, cv::Scalar(0));
            cv::Mat geometry_source_mask(reference->depth.size(), CV_16U, cv::Scalar(0));
            cv::Mat source_inverse_depth_sum(reference->depth.size(), CV_32F, cv::Scalar(0.0f));
            cv::Mat source_inverse_depth_squared_sum(reference->depth.size(), CV_32F, cv::Scalar(0.0f));
            const bool generate_adaptive_evidence =
                _config.enableAdaptiveGeometryEvidence && _effectiveSceneProfile == MvsSceneProfile::OrbitalObject;
            const bool generate_projected_source_layers = _effectiveSceneProfile == MvsSceneProfile::OrbitalObject ||
                                                          _config.enableDepthLayerReliabilityGuidedCorrection;
            AdaptiveGeometryEvidenceAccumulatorMaps adaptive_evidence_accumulator =
                generate_adaptive_evidence ? makeAdaptiveGeometryEvidenceAccumulatorMaps(reference->depth.size())
                                           : AdaptiveGeometryEvidenceAccumulatorMaps{};
            std::vector<cv::Mat> projected_sources;
            std::vector<ProjectedDepthEvidence> projected_source_evidence;
            if (generate_projected_source_layers)
            {
                projected_sources.resize(repair_source_indices.size());
                if (_config.enableDepthLayerReliabilityGuidedCorrection)
                {
                    projected_source_evidence.resize(repair_source_indices.size());
                }
            }
            std::vector<DepthConsistencyCache::FrameHandle> consistency_source_handles;
            std::vector<DepthConsistencySourceInput> consistency_inputs;
            consistency_source_handles.reserve(source_indices.size());
            consistency_inputs.reserve(source_indices.size());
            uint64_t consistency_batch_bytes = 0;
            const uint64_t estimated_source_bytes =
                std::max<uint64_t>(largest_frame_bytes, static_cast<uint64_t>(reference->byteSize()));
            const uint64_t consistency_batch_budget = std::max<uint64_t>(
                estimated_source_bytes,
                cache_budget > reference->byteSize() ? cache_budget - static_cast<uint64_t>(reference->byteSize())
                                                     : estimated_source_bytes);
            const float maximum_round_trip_error_pixels =
                scaleDepthPixelDistance(kFullRasterConsistencyRoundTripPixels,
                                        pixelDomainScaleForResult(_depthFrames[frame_index], reference->depth.size()));
            auto flush_consistency_batch = [&]()
            {
                accumulateDepthConsistency(reference->depth,
                                           reference_camera,
                                           consistency_inputs,
                                           relative_threshold,
                                           maximum_round_trip_error_pixels,
                                           row_workers,
                                           _cancelled,
                                           consistent_votes,
                                           occluded_votes,
                                           contradicted_votes,
                                           unverifiable_votes,
                                           geometry_source_mask,
                                           source_inverse_depth_sum,
                                           source_inverse_depth_squared_sum,
                                           generate_adaptive_evidence ? &adaptive_evidence_accumulator : nullptr);
                consistency_inputs.clear();
                consistency_source_handles.clear();
                consistency_batch_bytes = 0;
            };

            for (int source_ordinal = 0; source_ordinal < static_cast<int>(repair_source_indices.size());
                 ++source_ordinal)
            {
                const float source_progress =
                    (static_cast<float>(completed_frames) +
                     static_cast<float>(source_ordinal) /
                         static_cast<float>(std::max<std::size_t>(1, repair_source_indices.size()))) /
                    static_cast<float>(std::max(1, view_count));
                progressChanged(QStringLiteral("流式多视一致性：帧 %1/%2，源视图 %3/%4")
                                    .arg(frame_index + 1)
                                    .arg(view_count)
                                    .arg(source_ordinal + 1)
                                    .arg(repair_source_indices.size()),
                                source_progress);
                const int source_index = repair_source_indices[static_cast<std::size_t>(source_ordinal)];
                if (_cancelled.load())
                {
                    remove_pending_files();
                    return false;
                }
                if (source_index == frame_index)
                {
                    continue;
                }

                const bool is_consistency_source =
                    std::find(source_indices.begin(), source_indices.end(), source_index) != source_indices.end();
                if (!consistency_inputs.empty() &&
                    (!is_consistency_source ||
                     consistency_batch_bytes + estimated_source_bytes > consistency_batch_budget))
                {
                    // Release pinned consistency handles before acquiring a
                    // repair-only source, leaving one transient-frame slot in the
                    // bounded cache instead of exceeding the budget by one frame.
                    flush_consistency_batch();
                }

                const DepthConsistencyCache::FrameHandle source = cache.acquire(source_index, &load_error);
                if (!source || source->depth.empty())
                {
                    LOG_WARN(QStringLiteral("[MVS] 流式一致性检查跳过无法读取的源帧 %1：%2")
                                 .arg(source_index)
                                 .arg(QString::fromStdString(load_error)));
                    continue;
                }
                if (is_consistency_source)
                {
                    const uint64_t source_bytes = static_cast<uint64_t>(source->byteSize());
                    if (!consistency_inputs.empty() &&
                        consistency_batch_bytes + source_bytes > consistency_batch_budget)
                    {
                        flush_consistency_batch();
                    }
                    DepthConsistencySourceInput input;
                    input.depth = source->depth;
                    input.camera = _depthFrames[source_index].cameraModel.isValid()
                                       ? _depthFrames[source_index].cameraModel
                                       : mvsPinholeCamera(_views[source_index].camera);
                    input.confidence = source->confidence;
                    input.reliabilityWeight = sourceGeometryReliabilityWeight(_depthFrames[frame_index], source_index);
                    input.sourceOrdinal = source_ordinal;
                    const DepthPixelDomainScale source_pixel_scale =
                        pixelDomainScaleForResult(_depthFrames[source_index], source->depth.size());
                    input.searchRadiusPixels =
                        scaleDepthPixelRadius(kFullRasterConsistencySearchRadiusPixels, source_pixel_scale);
                    input.evaluateSubpixelFootprint =
                        source_pixel_scale.usesReducedGrid() && input.searchRadiusPixels == 0;
                    consistency_inputs.push_back(std::move(input));
                    consistency_source_handles.push_back(source);
                    consistency_batch_bytes += source_bytes;
                }
                if (generate_projected_source_layers)
                {
                    const FramePinholeCamera source_camera = _depthFrames[source_index].cameraModel.isValid()
                                                                 ? _depthFrames[source_index].cameraModel
                                                                 : mvsPinholeCamera(_views[source_index].camera);
                    if (_config.enableDepthLayerReliabilityGuidedCorrection && !source->confidence.empty())
                    {
                        ProjectedDepthEvidence evidence =
                            projectSourceDepthEvidenceToReference(source->depth,
                                                                  source->confidence,
                                                                  source_camera,
                                                                  reference_camera,
                                                                  filtered_depth.size(),
                                                                  repair_options.maximumProjectionDistancePixels,
                                                                  cameraBaselineSector(reference_camera, source_camera),
                                                                  nullptr,
                                                                  row_workers,
                                                                  &_cancelled);
                        projected_sources[static_cast<std::size_t>(source_ordinal)] = evidence.depth;
                        projected_source_evidence[static_cast<std::size_t>(source_ordinal)] = std::move(evidence);
                    }
                    else
                    {
                        projected_sources[static_cast<std::size_t>(source_ordinal)] =
                            projectSourceDepthToReference(source->depth,
                                                          source_camera,
                                                          reference_camera,
                                                          filtered_depth.size(),
                                                          repair_options.maximumProjectionDistancePixels,
                                                          nullptr,
                                                          row_workers,
                                                          &_cancelled);
                    }
                }
            }
            flush_consistency_batch();
            const auto source_stage_end = Clock::now();

            const PreRepairDepthLayerReliability depth_layer_reliability = analyzePreRepairDepthLayerReliability(
                _depthFrames[frame_index],
                reference->depth,
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
                _depthFrames[frame_index].depthLayerReliabilityClass =
                    QSharedPointer<cv::Mat>::create(depth_layer_reliability.result.classMap);
            }

            cv::Mat repair_mask =
                _depthFrames[frame_index].supportRegionMask && !_depthFrames[frame_index].supportRegionMask->empty()
                    ? *_depthFrames[frame_index].supportRegionMask
                    : cv::Mat(filtered_depth.size(), CV_8UC1, cv::Scalar(255));
            if (repair_mask.size() != filtered_depth.size())
            {
                cv::resize(repair_mask, repair_mask, filtered_depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            if (!_depthFrames[frame_index].missingReasonMap || _depthFrames[frame_index].missingReasonMap->empty())
            {
                const QString reason_path =
                    storage_dir.filePath(QStringLiteral("depth_%1_missing_reason.png").arg(frame_index));
                cv::Mat reason_map =
                    xjw::common::io::readImage(xjw::common::io::toUtf8Path(reason_path), cv::IMREAD_GRAYSCALE);
                if (reason_map.empty() || reason_map.type() != CV_8UC1 || reason_map.size() != filtered_depth.size())
                {
                    remove_pending_files();
                    const QString message = QStringLiteral("流式一致性读取帧 %1 缺失原因图失败："
                                                           "%2 的类型或尺寸无效")
                                                .arg(frame_index)
                                                .arg(reason_path);
                    LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                    errorOccurred(message);
                    return false;
                }
                _depthFrames[frame_index].missingReasonMap = QSharedPointer<cv::Mat>::create(reason_map);
            }
            const cv::Mat consistent_mask = makeDepthConsistencyMask(filtered_depth,
                                                                     static_cast<int>(source_indices.size()),
                                                                     minimum_source_confirmations,
                                                                     consistent_votes,
                                                                     occluded_votes,
                                                                     contradicted_votes,
                                                                     row_workers,
                                                                     &_cancelled);

            if (_cancelled.load())
            {
                remove_pending_files();
                return false;
            }
            progressChanged(
                QStringLiteral("流式多视一致性：帧 %1/%2，选择主深度层").arg(frame_index + 1).arg(view_count),
                static_cast<float>(completed_frames) / static_cast<float>(std::max(1, view_count)));

            const int valid_before = cv::countNonZero(reference->depth > 0.0f);
            _depthFrames[frame_index].crossViewRepairedMask =
                QSharedPointer<cv::Mat>::create(filtered_depth.size(), CV_8UC1, cv::Scalar(0));
            DominantDepthLayerSelectionStats layer_selection_stats;
            DepthGeometryHypothesisRerankMaps geometry_rerank_maps;
            if (generate_adaptive_evidence)
            {
                reference->depth.copyTo(filtered_depth);
                filtered_depth.setTo(0.0f, repair_mask == 0);
                DominantDepthLayerSelectionOptions layer_selection_options;
                layer_selection_options.enableReliabilityGuidedCorrection =
                    _config.enableDepthLayerReliabilityGuidedCorrection;
                layer_selection_options.restrictToReliabilityGuidedCandidates =
                    !generate_adaptive_evidence && _config.enableDepthLayerReliabilityGuidedCorrection;
                const cv::Mat* native_reliability_classes = _config.enableDepthLayerReliabilityGuidedCorrection
                                                                ? &depth_layer_reliability.result.classMap
                                                                : nullptr;
                layer_selection_stats = selectDominantProjectedDepthLayer(
                    filtered_depth,
                    repair_mask,
                    projected_sources,
                    consistent_votes,
                    contradicted_votes,
                    layer_selection_options,
                    filtered_confidence.empty() ? nullptr : &filtered_confidence,
                    _depthFrames[frame_index].crossViewRepairedMask.data(),
                    &geometry_source_mask,
                    &source_inverse_depth_sum,
                    &source_inverse_depth_squared_sum,
                    &consistent_votes,
                    row_workers,
                    &_cancelled,
                    native_reliability_classes,
                    nullptr,
                    _config.enableDepthLayerReliabilityGuidedCorrection ? &projected_source_evidence : nullptr,
                    _config.enableDepthLayerReliabilityGuidedCorrection ? &geometry_rerank_maps : nullptr);
            }
            else if (_config.enableDepthLayerReliabilityGuidedCorrection)
            {
                cv::Mat guided_depth = reference->depth.clone();
                guided_depth.setTo(0.0f, repair_mask == 0);
                cv::Mat guided_confidence = reference->confidence.empty() ? cv::Mat() : reference->confidence.clone();
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
                                                      _depthFrames[frame_index].crossViewRepairedMask.data(),
                                                      &guided_geometry_source_mask,
                                                      &guided_inverse_sum,
                                                      &guided_inverse_squared_sum,
                                                      &guided_votes,
                                                      row_workers,
                                                      &_cancelled,
                                                      &depth_layer_reliability.result.classMap,
                                                      &guided_changed_mask,
                                                      &projected_source_evidence,
                                                      &geometry_rerank_maps);
                reference->depth.copyTo(filtered_depth);
                filtered_depth.setTo(0.0f, repair_mask == 0);
                filtered_depth.setTo(0.0f, consistent_mask == 0);
                if (!guided_changed_mask.empty())
                {
                    guided_depth.copyTo(filtered_depth, guided_changed_mask);
                    if (!filtered_confidence.empty() && !guided_confidence.empty())
                    {
                        guided_confidence.copyTo(filtered_confidence, guided_changed_mask);
                    }
                    guided_geometry_source_mask.copyTo(geometry_source_mask, guided_changed_mask);
                    guided_inverse_sum.copyTo(source_inverse_depth_sum, guided_changed_mask);
                    guided_inverse_squared_sum.copyTo(source_inverse_depth_squared_sum, guided_changed_mask);
                    guided_votes.copyTo(consistent_votes, guided_changed_mask);
                }
            }
            else
            {
                filtered_depth.setTo(0.0f, consistent_mask == 0);
            }
            if (_config.enableDepthLayerReliabilityGuidedCorrection &&
                geometry_rerank_maps.compatible(filtered_depth.size()))
            {
                _depthFrames[frame_index].geometryRerankMaps =
                    QSharedPointer<DepthGeometryHypothesisRerankMaps>::create(std::move(geometry_rerank_maps));
            }
            WeakNativeDepthRetentionStats weak_native_retention;
            if (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                weak_native_retention =
                    retainWeaklyVerifiedNativeDepth(reference->depth,
                                                    reference->confidence,
                                                    repair_mask,
                                                    consistent_votes,
                                                    contradicted_votes,
                                                    {},
                                                    &filtered_depth,
                                                    filtered_confidence.empty() ? nullptr : &filtered_confidence);
            }
            const auto layer_stage_end = Clock::now();
            cv::Mat anchored_interpolation_mask;
            cv::Mat native_interpolation_anchor_eligibility_mask;
            const cv::Mat* native_interpolation_anchor_eligibility = nullptr;
            if (_config.enableDepthLayerReliabilityAnchorGate)
            {
                native_interpolation_anchor_eligibility_mask = cv::Mat(filtered_depth.size(), CV_8UC1, cv::Scalar(0));
                if (depth_layer_reliability.result.validInputs &&
                    depth_layer_reliability.result.classMap.type() == CV_8UC1 &&
                    depth_layer_reliability.result.classMap.size() == filtered_depth.size())
                {
                    cv::compare(depth_layer_reliability.result.classMap,
                                static_cast<std::uint8_t>(DepthLayerReliabilityClass::Reliable),
                                native_interpolation_anchor_eligibility_mask,
                                cv::CMP_EQ);
                }
                native_interpolation_anchor_eligibility = &native_interpolation_anchor_eligibility_mask;
            }
            progressChanged(
                QStringLiteral("流式多视一致性：帧 %1/%2，修复内部缺口").arg(frame_index + 1).arg(view_count),
                static_cast<float>(completed_frames) / static_cast<float>(std::max(1, view_count)));
            const CrossViewHoleRepairStats repair_stats =
                repairDepthHolesFromProjectedSources(filtered_depth,
                                                     repair_mask,
                                                     projected_sources,
                                                     repair_options,
                                                     filtered_confidence.empty() ? nullptr : &filtered_confidence,
                                                     &consistent_votes,
                                                     _depthFrames[frame_index].crossViewRepairedMask.data(),
                                                     &geometry_source_mask,
                                                     &source_inverse_depth_sum,
                                                     &source_inverse_depth_squared_sum,
                                                     &reference_camera,
                                                     referenceGray,
                                                     &anchored_interpolation_mask,
                                                     row_workers,
                                                     &_cancelled,
                                                     native_interpolation_anchor_eligibility);
            if (_cancelled.load())
            {
                remove_pending_files();
                return false;
            }
            WeakNativeDepthRetentionOptions unconfirmed_backfill_options;
            unconfirmed_backfill_options.minimumConfirmationCount = std::numeric_limits<int>::max();
            unconfirmed_backfill_options.retainUnconfirmedWithoutContradiction = true;
            const WeakNativeDepthRetentionStats unconfirmed_native_backfill =
                retainWeaklyVerifiedNativeDepth(reference->depth,
                                                reference->confidence,
                                                repair_mask,
                                                consistent_votes,
                                                contradicted_votes,
                                                unconfirmed_backfill_options,
                                                &filtered_depth,
                                                filtered_confidence.empty() ? nullptr : &filtered_confidence);
            _depthFrames[frame_index].crossViewRepairDiagnostics = crossViewHoleRepairStatsToJson(repair_stats);
            _depthFrames[frame_index].crossViewRepairDiagnostics.insert(QStringLiteral("depth_layer_reliability"),
                                                                        depth_layer_reliability.diagnostics);
            _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
                QStringLiteral("dominant_depth_layer_selection"),
                dominantDepthLayerSelectionStatsToJson(layer_selection_stats));
            _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
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
            _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
                QStringLiteral("unconfirmed_native_backfill"),
                QJsonObject{{QStringLiteral("considered_pixel_count"),
                             static_cast<double>(unconfirmed_native_backfill.consideredPixelCount)},
                            {QStringLiteral("retained_pixel_count"),
                             static_cast<double>(unconfirmed_native_backfill.retainedPixelCount)},
                            {QStringLiteral("rejected_contradiction_pixel_count"),
                             static_cast<double>(unconfirmed_native_backfill.rejectedContradictionPixelCount)}});
            _depthFrames[frame_index].depthCompleteness.crossViewRepairedCount +=
                static_cast<int>(repair_stats.repairedPixelCount + layer_selection_stats.switchedNativePixelCount +
                                 layer_selection_stats.transferredMissingPixelCount);
            cv::Mat restoration_mask =
                _depthFrames[frame_index].supportRegionMask && !_depthFrames[frame_index].supportRegionMask->empty()
                    ? *_depthFrames[frame_index].supportRegionMask
                    : cv::Mat(filtered_depth.size(), CV_8UC1, cv::Scalar(255));
            if (restoration_mask.size() != filtered_depth.size())
            {
                cv::resize(restoration_mask, restoration_mask, filtered_depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            const int restored_count = restoreSmallInteriorDepthHoles(
                filtered_depth,
                reference->depth,
                reference->confidence,
                restoration_mask,
                0.75f,
                0.12f,
                kSmallHoleAreaFraction,
                effectiveMinimumSmallHoleArea(_depthFrames[frame_index], filtered_depth.size()));
            _depthFrames[frame_index].depthCompleteness.restoredFromPrefilterCount += restored_count;
            const auto repair_stage_end = Clock::now();
            DepthResidualReestimationStats residual_stats;
            cv::Mat residual_reestimated_mask;
            const bool reliability_guided_local_pass = _config.enableDepthLayerReliabilityGuidedCorrection &&
                                                       _effectiveSceneProfile == MvsSceneProfile::Custom;
            if (_config.enablePostConsistencyResidualReestimation &&
                (_effectiveSceneProfile == MvsSceneProfile::OrbitalObject || reliability_guided_local_pass) &&
                referenceImageLease)
            {
                progressChanged(
                    QStringLiteral("流式多视一致性：帧 %1/%2，残余缺口局部重估").arg(frame_index + 1).arg(view_count),
                    static_cast<float>(completed_frames) / static_cast<float>(std::max(1, view_count)));
                std::vector<cv::Mat> residual_projected_sources;
                std::vector<ProjectedDepthEvidence> residual_projected_source_evidence;
                std::vector<int> residual_sector_ids;
                std::vector<cv::Mat> residual_source_images;
                std::vector<FramePinholeCamera> residual_source_cameras;
                std::vector<cv::Mat> residual_source_masks;
                std::vector<MvsImageCache::ImageLease> residualSourceImageLeases;
                residualSourceImageLeases.reserve(repair_source_indices.size());
                for (int source_ordinal = 0; source_ordinal < static_cast<int>(projected_sources.size());
                     ++source_ordinal)
                {
                    const int source_index = repair_source_indices[static_cast<std::size_t>(source_ordinal)];
                    const cv::Mat& projected = projected_sources[static_cast<std::size_t>(source_ordinal)];
                    if (projected.empty() || source_index < 0 || source_index >= view_count)
                    {
                        continue;
                    }
                    std::string sourceImageError;
                    MvsImageCache::ImageLease sourceImageLease = acquireImageFrame(source_index, &sourceImageError);
                    if (!sourceImageLease)
                    {
                        LOG_WARN(QStringLiteral("[MVS][帧 %1][流式残余重估] 源影像 %2 不可用：%3")
                                     .arg(frame_index)
                                     .arg(source_index)
                                     .arg(QString::fromStdString(sourceImageError)));
                        continue;
                    }
                    const FramePinholeCamera source_camera = _depthFrames[source_index].cameraModel.isValid()
                                                                 ? _depthFrames[source_index].cameraModel
                                                                 : mvsPinholeCamera(_views[source_index].camera);
                    residual_projected_sources.push_back(projected);
                    if (reliability_guided_local_pass)
                    {
                        residual_projected_source_evidence.push_back(
                            projected_source_evidence[static_cast<std::size_t>(source_ordinal)]);
                    }
                    residual_sector_ids.push_back(cameraBaselineSector(reference_camera, source_camera));
                    residual_source_images.push_back(sourceImageLease->preparedGray);
                    residual_source_cameras.push_back(source_camera);
                    residual_source_masks.push_back(
                        _depthFrames[source_index].supportRegionMask &&
                                !_depthFrames[source_index].supportRegionMask->empty()
                            ? *_depthFrames[source_index].supportRegionMask
                            : cv::Mat(residual_source_images.back().size(), CV_8UC1, cv::Scalar(255)));
                    residualSourceImageLeases.push_back(std::move(sourceImageLease));
                }
                DepthResidualReestimationOptions residual_options;
                residual_options.maximumLayerInverseDepthRelativeSpread =
                    std::max(0.0f, _config.postConsistencyResidualMaximumLayerSpread);
                residual_options.maximumPriorRadiusRatio =
                    std::clamp(_config.postConsistencyResidualMaximumPriorRadius, 0.005f, 0.25f);
                residual_options.minimumCandidateConfidence =
                    std::clamp(_config.postConsistencyResidualConfidence, 0.0f, 1.0f);
                if (reliability_guided_local_pass)
                {
                    residual_options.minimumLayerSourceCount = 3;
                    residual_options.minimumLayerSectorCount = 2;
                    residual_options.minimumGeometryConfirmationCount = 3;
                    residual_options.minimumGeometrySectorCount = 2;
                    residual_options.allowValidDepthReplacement = true;
                    residual_options.minimumReplacementCostAdvantage = 0.08f;
                    residual_options.ambiguousMaximumRelativeCorrection = 0.01f;
                    residual_options.minimumCandidateConfidence =
                        std::min(residual_options.minimumCandidateConfidence, 0.18f);
                }
                DepthResidualReestimationPreflight residual_preflight;
                if (reliability_guided_local_pass && depth_layer_reliability.result.validInputs &&
                    geometry_rerank_maps.compatible(filtered_depth.size()))
                {
                    cv::Mat ambiguous;
                    cv::Mat rejected;
                    cv::compare(depth_layer_reliability.result.classMap,
                                static_cast<std::uint8_t>(DepthLayerReliabilityClass::AmbiguousLowTexture),
                                ambiguous,
                                cv::CMP_EQ);
                    cv::compare(depth_layer_reliability.result.classMap,
                                static_cast<std::uint8_t>(DepthLayerReliabilityClass::RejectedLayer),
                                rejected,
                                cv::CMP_EQ);
                    cv::Mat weak_mask;
                    cv::bitwise_or(ambiguous, rejected, weak_mask);
                    cv::bitwise_and(weak_mask,
                                    geometry_rerank_maps.decisionAction ==
                                        static_cast<std::uint8_t>(DepthGeometryHypothesisAction::None),
                                    weak_mask);
                    residual_preflight = inspectDepthReestimationMask(repair_mask, weak_mask, residual_options);
                }
                else
                {
                    residual_preflight =
                        inspectDepthResidualReestimationNeed(filtered_depth, repair_mask, residual_options);
                }
                const DepthResidualReestimationTarget residual_target =
                    buildDepthResidualReestimationTarget(filtered_depth,
                                                         repair_mask,
                                                         residual_projected_sources,
                                                         residual_sector_ids,
                                                         residual_options,
                                                         std::move(residual_preflight));
                residual_stats.supportPixelCount = residual_target.supportPixelCount;
                residual_stats.requestedResidualPixelCount = residual_target.requestedResidualPixelCount;
                residual_stats.layerCoveredPixelCount = residual_target.layerCoveredPixelCount;
                residual_stats.insufficientSourcePixelCount = residual_target.insufficientSourcePixelCount;
                residual_stats.insufficientSectorPixelCount = residual_target.insufficientSectorPixelCount;
                residual_stats.layerSpreadRejectedPixelCount = residual_target.layerSpreadRejectedPixelCount;
                residual_stats.sourceCount = static_cast<int>(residual_source_images.size());
                residual_stats.skippedReason = residual_target.skippedReason;
                if (residual_target.valid && residual_source_images.size() >= 4)
                {
                    const std::vector<std::vector<int>> source_groups =
                        buildDepthResidualPatchMatchSourceGroups(residual_sector_ids);
                    double minimum_hint = 0.0;
                    double maximum_hint = 0.0;
                    cv::minMaxLoc(residual_target.hintDepth,
                                  &minimum_hint,
                                  &maximum_hint,
                                  nullptr,
                                  nullptr,
                                  residual_target.residualMask);
                    const float z_near = std::max(1.0e-4f, static_cast<float>(minimum_hint * 0.90));
                    const float z_far = std::max(z_near * 1.01f, static_cast<float>(maximum_hint * 1.10));
                    std::vector<cv::Mat> candidate_depths;
                    std::vector<cv::Mat> candidate_confidences;
                    residual_stats.attemptedHypothesisCount = 2;
                    for (const std::vector<int>& source_group : source_groups)
                    {
                        std::vector<cv::Mat> group_images;
                        std::vector<FramePinholeCamera> group_cameras;
                        std::vector<cv::Mat> group_masks;
                        for (const int source_ordinal : source_group)
                        {
                            group_images.push_back(residual_source_images[static_cast<std::size_t>(source_ordinal)]);
                            group_cameras.push_back(residual_source_cameras[static_cast<std::size_t>(source_ordinal)]);
                            group_masks.push_back(residual_source_masks[static_cast<std::size_t>(source_ordinal)]);
                        }
                        PatchMatchConfig patch_match = patchMatchConfigForRecordedWorker(
                            _config.patchMatch, _depthFrames[static_cast<std::size_t>(frame_index)].device);
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
                        std::string residual_error;
                        if (estimatePatchMatchWithAdaptiveCuda("streaming post-consistency residual PatchMatch",
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
                                                               &residual_error,
                                                               &residual_target.hintDepth,
                                                               &residual_target.hintRadius,
                                                               &residual_target.estimationMask,
                                                               &group_masks))
                        {
                            candidate_depths.push_back(std::move(candidate_depth));
                            candidate_confidences.push_back(std::move(candidate_confidence));
                        }
                        else
                        {
                            LOG_WARN("[MVS][帧 %d][流式残余重估] PatchMatch 组失败: %s",
                                     frame_index,
                                     residual_error.c_str());
                        }
                    }
                    residual_stats.successfulHypothesisCount = static_cast<int>(candidate_depths.size());
                    residual_stats.failedHypothesisCount = 2 - residual_stats.successfulHypothesisCount;
                    if (candidate_depths.size() == 2)
                    {
                        DepthResidualReestimationEvidenceOutputs evidence_outputs;
                        evidence_outputs.geometrySourceMask = &geometry_source_mask;
                        evidence_outputs.sourceInverseDepthSum = &source_inverse_depth_sum;
                        evidence_outputs.sourceInverseDepthSquaredSum = &source_inverse_depth_squared_sum;
                        evidence_outputs.confirmedSourceCount = &consistent_votes;
                        evidence_outputs.rerankMaps = reliability_guided_local_pass ? &geometry_rerank_maps : nullptr;
                        residual_stats = mergeDepthResidualReestimationCandidates(
                            filtered_depth,
                            filtered_confidence,
                            candidate_depths,
                            candidate_confidences,
                            residual_target,
                            residual_projected_sources,
                            residual_sector_ids,
                            &residual_reestimated_mask,
                            residual_options,
                            reliability_guided_local_pass ? &depth_layer_reliability.result.classMap : nullptr,
                            reliability_guided_local_pass ? &residual_projected_source_evidence : nullptr,
                            reliability_guided_local_pass ? &evidence_outputs : nullptr);
                        residual_stats.attemptedHypothesisCount = 2;
                        residual_stats.successfulHypothesisCount = 2;
                        residual_stats.sourceCount = static_cast<int>(residual_source_images.size());
                        if (residual_stats.recoveredPixelCount > 0)
                        {
                            if (!reliability_guided_local_pass)
                            {
                                consistent_votes.setTo(cv::Scalar(2), residual_reestimated_mask);
                                // Legacy Orbital recovery proves two grouped
                                // hypotheses but cannot identify exact source bits.
                                geometry_source_mask.setTo(cv::Scalar(0), residual_reestimated_mask);
                                cv::Mat recovered_inverse_depth;
                                cv::divide(1.0f, filtered_depth, recovered_inverse_depth);
                                cv::Mat recovered_inverse_depth_sum = recovered_inverse_depth * 2.0f;
                                recovered_inverse_depth_sum.copyTo(source_inverse_depth_sum, residual_reestimated_mask);
                                cv::Mat recovered_inverse_depth_squared_sum =
                                    recovered_inverse_depth.mul(recovered_inverse_depth) * 2.0f;
                                recovered_inverse_depth_squared_sum.copyTo(source_inverse_depth_squared_sum,
                                                                           residual_reestimated_mask);
                            }
                        }
                    }
                    else
                    {
                        residual_stats.attempted = true;
                        residual_stats.skippedReason = QStringLiteral("incomplete_hypothesis_pair");
                    }
                }
                _depthFrames[frame_index].residualReestimationDiagnostics =
                    depthResidualReestimationStatsToJson(residual_stats);
                if (!residual_reestimated_mask.empty())
                {
                    _depthFrames[frame_index].residualReestimatedMask =
                        QSharedPointer<cv::Mat>::create(residual_reestimated_mask);
                }
            }
            const auto residual_stage_end = Clock::now();
            progressChanged(QStringLiteral("流式多视一致性：帧 %1/%2，深度后处理").arg(frame_index + 1).arg(view_count),
                            static_cast<float>(completed_frames) / static_cast<float>(std::max(1, view_count)));
            if (!filtered_confidence.empty())
            {
                filtered_confidence.setTo(0.0f, filtered_depth <= 0.0f);
            }
            int valid_after = cv::countNonZero(filtered_depth > 0.0f);
            const int observed_valid_after = valid_after;
            bool original_depth_fallback_applied = false;
            if (valid_before > 100 && valid_after < valid_before / 10)
            {
                const float observed_retention =
                    100.0f * static_cast<float>(valid_after) / static_cast<float>(valid_before);
                LOG_WARN("[MVS][帧 %d][流式一致性] 保留率过低 %.1f%%，"
                         "仅恢复原始深度作为诊断载体",
                         frame_index,
                         observed_retention);
                reference->depth.copyTo(filtered_depth);
                if (!filtered_confidence.empty())
                {
                    reference->confidence.copyTo(filtered_confidence);
                }
                _depthFrames[frame_index].crossViewRepairedMask->setTo(cv::Scalar(0));
                anchored_interpolation_mask.setTo(cv::Scalar(0));
                valid_after = valid_before;
                original_depth_fallback_applied = true;
            }
            if (!_depthFrames[frame_index].depthProvenance || _depthFrames[frame_index].depthProvenance->empty())
            {
                _depthFrames[frame_index].depthProvenance = QSharedPointer<cv::Mat>::create(
                    initializeDepthProvenance(reference->depth, targeted_gap_recovered_mask));
            }
            updateDepthProvenance(*_depthFrames[frame_index].depthProvenance,
                                  filtered_depth,
                                  targeted_gap_recovered_mask,
                                  *_depthFrames[frame_index].crossViewRepairedMask,
                                  anchored_interpolation_mask,
                                  residual_reestimated_mask);
            const DepthConsistencyPublicationSummary consistency_publication = summarizeDepthConsistencyPublication(
                valid_before, observed_valid_after, valid_after, original_depth_fallback_applied, true);
            const float consistency_keep_rate = consistency_publication.observedRetentionRatio;
            _depthFrames[frame_index].depthCompleteness.preConsistencyValidCount = valid_before;
            _depthFrames[frame_index].depthCompleteness.postConsistencyValidCount = observed_valid_after;
            _depthFrames[frame_index].depthCompleteness.consistencyRetentionRatio = consistency_keep_rate;
            _depthFrames[frame_index].depthCompleteness.publishedPostConsistencyValidCount = valid_after;
            _depthFrames[frame_index].depthCompleteness.publishedConsistencyRetentionRatio =
                consistency_publication.publishedRetentionRatio;
            _depthFrames[frame_index].depthCompleteness.consistencyPublicationFallbackApplied =
                original_depth_fallback_applied;
            const DepthConsistencyVoteTotals vote_totals = summarizeDepthConsistencyVotes(
                consistent_votes, occluded_votes, contradicted_votes, unverifiable_votes, row_workers);
            _depthFrames[frame_index].depthCompleteness.consistencyConfirmedObservationCount =
                static_cast<int>(vote_totals.consistent);
            _depthFrames[frame_index].depthCompleteness.consistencyOccludedObservationCount =
                static_cast<int>(vote_totals.occluded);
            _depthFrames[frame_index].depthCompleteness.consistencyContradictedObservationCount =
                static_cast<int>(vote_totals.contradicted);
            _depthFrames[frame_index].depthCompleteness.consistencyUnverifiableObservationCount =
                static_cast<int>(vote_totals.unverifiable);
            _depthFrames[frame_index].depthCompleteness.consistencyRejectedPixelCount =
                std::max(0, valid_before - observed_valid_after);

            FusionConfig fusion_config = _config.fusion;
            const DepthFilterSettings filter_settings =
                depthFilterSettings(_effectiveDepthFilterMode, static_cast<int>(source_indices.size()));
            fusion_config.localDepthOutlierRelThresh = filter_settings.localDepthOutlierRelThreshold;
            fusion_config.minSpeckleComponentArea = filter_settings.minComponentArea;
            fusion_config.minConsistentViews = filter_settings.minConsistentViews;
            fusion_config.confidenceThresh = depthConfidenceThresholds(_effectiveSceneProfile,
                                                                       _effectiveDepthFilterMode,
                                                                       static_cast<int>(source_indices.size()),
                                                                       _config.patchMatch.confidenceThresh,
                                                                       fusion_config.confidenceThresh)
                                                 .fusion;
            const GeometryEvidenceMaps postprocess_input_geometry_evidence =
                makeGeometryEvidenceMaps(filtered_depth,
                                         consistent_votes,
                                         geometry_source_mask,
                                         source_inverse_depth_sum,
                                         source_inverse_depth_squared_sum);
            const AdaptiveGeometryEvidenceMaps adaptive_evidence =
                generate_adaptive_evidence
                    ? makeAdaptiveGeometryEvidenceMaps(reference->depth, adaptive_evidence_accumulator)
                    : AdaptiveGeometryEvidenceMaps{};
            const AdaptiveGeometryEvidenceSummary adaptive_summary =
                summarizeAdaptiveGeometryEvidence(adaptive_evidence);
            DepthPostProcessEvidence postprocess_evidence;
            postprocess_evidence.geometrySupportCount = postprocess_input_geometry_evidence.supportCount;
            postprocess_evidence.inverseDepthRelativeSpread =
                postprocess_input_geometry_evidence.inverseDepthRelativeSpread;
            postprocess_evidence.adaptiveSupportWeight = adaptive_evidence.supportWeight;
            postprocess_evidence.adaptiveEffectiveViewCount = adaptive_evidence.effectiveViewCount;
            postprocess_evidence.adaptiveConflictRatio = adaptive_evidence.conflictRatio;
            postprocess_evidence = updateDepthEvidenceConfidence(_depthFrames[frame_index],
                                                                 filtered_depth,
                                                                 filtered_confidence,
                                                                 std::move(postprocess_evidence),
                                                                 _config.enableDepthLayerReliabilityGuidedCorrection);
            markDepthLossReason(*_depthFrames[frame_index].missingReasonMap,
                                reference->depth,
                                filtered_depth,
                                DepthMissingReason::InsufficientGeometrySupport);
            captureStageSnapshot(frame_index,
                                 MvsStageSnapshotStage::CrossViewConsistency,
                                 QStringLiteral("after_cross_view_filter_and_repair_before_confidence_postprocess"),
                                 _depthFrames[frame_index],
                                 filtered_depth,
                                 filtered_confidence,
                                 filtered_depth > 0.0f);
            _depthFrames[frame_index].depthPostprocess =
                DepthPostprocessor::postprocessFusionDepthMap(filtered_depth,
                                          filtered_confidence,
                                          fusion_config,
                                          frame_index,
                                          view_count,
                                          _depthFrames[frame_index].missingReasonMap.data(),
                                          &postprocess_evidence,
                                          _depthFrames[frame_index].preparedRasterSize);
            _depthFrames[frame_index].depthPostprocessApplied = true;
            cv::Mat final_interpolation_mask;
            const DepthAnchoredHoleInterpolationStats final_repair =
                repairPostprocessedInternalDepthHoles(_depthFrames[frame_index],
                                                      filtered_depth,
                                                      filtered_confidence,
                                                      _effectiveSceneProfile,
                                                      &final_interpolation_mask);
            updateDepthProvenance(*_depthFrames[frame_index].depthProvenance,
                                  filtered_depth,
                                  targeted_gap_recovered_mask,
                                  *_depthFrames[frame_index].crossViewRepairedMask,
                                  final_interpolation_mask,
                                  residual_reestimated_mask);
            _depthFrames[frame_index].crossViewRepairDiagnostics.insert(
                QStringLiteral("postprocess_anchored_interpolation"),
                depthAnchoredHoleInterpolationStatsToJson(final_repair));
            _depthFrames[frame_index].depthCompleteness.crossViewRepairedCount +=
                static_cast<int>(final_repair.interpolatedPixelCount);
            if (final_repair.interpolatedPixelCount > 0)
            {
                LOG_DEBUG("[MVS][帧 %d][后处理] 锚定修复 pixels=%llu components=%llu",
                          frame_index,
                          static_cast<unsigned long long>(final_repair.interpolatedPixelCount),
                          static_cast<unsigned long long>(final_repair.acceptedComponentCount));
            }
            updateDepthCompletenessAfterPostprocess(
                _depthFrames[frame_index], filtered_depth, _depthFrames[frame_index].depthPostprocess);
            valid_after = cv::countNonZero(filtered_depth > 0.0f);
            const GeometryEvidenceMaps geometry_evidence = makeGeometryEvidenceMaps(filtered_depth,
                                                                                    consistent_votes,
                                                                                    geometry_source_mask,
                                                                                    source_inverse_depth_sum,
                                                                                    source_inverse_depth_squared_sum);
            const cv::Mat discrete_support_region =
                _depthFrames[frame_index].supportRegionMask ? *_depthFrames[frame_index].supportRegionMask : cv::Mat();
            const DiscreteGeometryCoreSummary discrete_summary =
                summarizeDiscreteGeometryCore(filtered_depth,
                                              geometry_evidence.supportCount,
                                              geometry_evidence.inverseDepthRelativeSpread,
                                              discrete_support_region);
            updateDepthFrameQualityAfterConsistency(_depthFrames[frame_index],
                                                    filtered_depth,
                                                    filtered_confidence,
                                                    _effectiveSceneProfile,
                                                    _effectiveDepthFilterMode,
                                                    true,
                                                    adaptive_summary,
                                                    discrete_summary);
            calibrateFinalDepthConfidenceMap(filtered_confidence, _depthFrames[frame_index].depthProvenance.data());
            finalizeDepthMissingReasonMap(*_depthFrames[frame_index].missingReasonMap,
                                          filtered_depth,
                                          repair_mask,
                                          geometry_evidence.supportCount,
                                          contradicted_votes);
            captureStageSnapshot(frame_index,
                                 MvsStageSnapshotStage::ConfidencePostprocess,
                                 QStringLiteral("after_confidence_postprocess_and_anchored_repair"),
                                 _depthFrames[frame_index],
                                 filtered_depth,
                                 filtered_confidence,
                                 filtered_depth > 0.0f);
            const cv::Mat& geometry_support = geometry_evidence.supportCount;
            const auto postprocess_stage_end = Clock::now();
            progressChanged(QStringLiteral("流式多视一致性：帧 %1/%2，写入结果").arg(frame_index + 1).arg(view_count),
                            static_cast<float>(completed_frames) / static_cast<float>(std::max(1, view_count)));

            const QString original_path = storage_dir.filePath(QStringLiteral("depth_%1.bin").arg(frame_index));
            const QString filtered_path =
                storage_dir.filePath(QStringLiteral("depth_%1_consistency.bin").arg(frame_index));
            const QString original_confidence_path =
                storage_dir.filePath(QStringLiteral("depth_%1_conf.bin").arg(frame_index));
            const QString filtered_confidence_path =
                storage_dir.filePath(QStringLiteral("depth_%1_conf_consistency.bin").arg(frame_index));
            const QString filtered_geometry_support_path =
                storage_dir.filePath(QStringLiteral("depth_%1_geometry_support_consistency.bin").arg(frame_index));
            const QString filtered_geometry_source_mask_path =
                storage_dir.filePath(QStringLiteral("depth_%1_geometry_source_mask_consistency.bin").arg(frame_index));
            const QString filtered_inverse_depth_mean_path =
                storage_dir.filePath(QStringLiteral("depth_%1_inverse_depth_mean_consistency.bin").arg(frame_index));
            const QString filtered_inverse_depth_spread_path =
                storage_dir.filePath(QStringLiteral("depth_%1_inverse_depth_spread_consistency.bin").arg(frame_index));
            const QString filtered_adaptive_support_weight_path =
                generate_adaptive_evidence
                    ? storage_dir.filePath(
                          QStringLiteral("depth_%1_adaptive_geometry_support_weight_consistency.bin").arg(frame_index))
                    : QString();
            const QString filtered_adaptive_effective_view_count_path =
                generate_adaptive_evidence
                    ? storage_dir.filePath(
                          QStringLiteral("depth_%1_adaptive_geometry_effective_view_count_consistency.bin")
                              .arg(frame_index))
                    : QString();
            const QString filtered_adaptive_conflict_ratio_path =
                generate_adaptive_evidence
                    ? storage_dir.filePath(
                          QStringLiteral("depth_%1_adaptive_geometry_conflict_ratio_consistency.bin").arg(frame_index))
                    : QString();
            const QString filtered_cross_view_repaired_mask_path =
                storage_dir.filePath(QStringLiteral("depth_%1_cross_view_repaired_consistency.bin").arg(frame_index));
            const QString filtered_residual_reestimated_mask_path =
                storage_dir.filePath(QStringLiteral("depth_%1_residual_reestimated_consistency.bin").arg(frame_index));
            const QString filtered_depth_provenance_path =
                storage_dir.filePath(QStringLiteral("depth_%1_provenance_consistency.bin").arg(frame_index));
            const QString filtered_missing_reason_path =
                storage_dir.filePath(QStringLiteral("depth_%1_missing_reason_consistency.bin").arg(frame_index));

            PendingDepthReplacement replacement;
            replacement.originalPath = original_path;
            replacement.filteredPath = filtered_path;
            replacement.originalConfidencePath = original_confidence_path;
            replacement.filteredConfidencePath = filtered_confidence.empty() ? QString() : filtered_confidence_path;
            replacement.filteredGeometrySupportPath = filtered_geometry_support_path;
            replacement.filteredGeometrySourceMaskPath = filtered_geometry_source_mask_path;
            replacement.filteredInverseDepthMeanPath = filtered_inverse_depth_mean_path;
            replacement.filteredInverseDepthSpreadPath = filtered_inverse_depth_spread_path;
            replacement.filteredAdaptiveSupportWeightPath = filtered_adaptive_support_weight_path;
            replacement.filteredAdaptiveEffectiveViewCountPath = filtered_adaptive_effective_view_count_path;
            replacement.filteredAdaptiveConflictRatioPath = filtered_adaptive_conflict_ratio_path;
            replacement.filteredCrossViewRepairedMaskPath = filtered_cross_view_repaired_mask_path;
            replacement.filteredResidualReestimatedMaskPath = filtered_residual_reestimated_mask_path;
            replacement.filteredDepthProvenancePath = filtered_depth_provenance_path;
            replacement.filteredMissingReasonPath = filtered_missing_reason_path;
            replacement.expectedDepthSize = filtered_depth.size();
            replacement.targetedGapRecoveredMaskExpected = targeted_gap_recovered_mask_expected;
            replacement.frameIndex = frame_index;
            pending_replacements.push_back(replacement);
            const xjw::common::OperationResult write_result =
                xjw::core::project::writeDepthMatStorage(filtered_path, filtered_depth);
            if (!write_result.ok)
            {
                remove_pending_files();
                LOG_WARN(QStringLiteral("[MVS] %1").arg(write_result.errorMessage));
                errorOccurred(write_result.errorMessage);
                return false;
            }
            if (!filtered_confidence.empty())
            {
                const xjw::common::OperationResult confidence_write_result =
                    xjw::core::project::writeDepthMatStorage(filtered_confidence_path, filtered_confidence);
                if (!confidence_write_result.ok)
                {
                    QFile::remove(filtered_path);
                    remove_pending_files();
                    errorOccurred(confidence_write_result.errorMessage);
                    return false;
                }
            }
            const xjw::common::OperationResult support_write_result =
                xjw::core::project::writeDepthMatStorage(filtered_geometry_support_path, geometry_support);
            if (!support_write_result.ok)
            {
                QFile::remove(filtered_path);
                QFile::remove(filtered_confidence_path);
                remove_pending_files();
                errorOccurred(support_write_result.errorMessage);
                return false;
            }
            const xjw::common::OperationResult source_mask_write_result = xjw::core::project::writeDepthMatStorage(
                filtered_geometry_source_mask_path, geometry_evidence.sourceMask);
            const xjw::common::OperationResult inverse_mean_write_result = xjw::core::project::writeDepthMatStorage(
                filtered_inverse_depth_mean_path, geometry_evidence.inverseDepthMean);
            const xjw::common::OperationResult inverse_spread_write_result = xjw::core::project::writeDepthMatStorage(
                filtered_inverse_depth_spread_path, geometry_evidence.inverseDepthRelativeSpread);
            if (!source_mask_write_result.ok || !inverse_mean_write_result.ok || !inverse_spread_write_result.ok)
            {
                QFile::remove(filtered_path);
                QFile::remove(filtered_confidence_path);
                QFile::remove(filtered_geometry_support_path);
                QFile::remove(filtered_geometry_source_mask_path);
                QFile::remove(filtered_inverse_depth_mean_path);
                QFile::remove(filtered_inverse_depth_spread_path);
                remove_pending_files();
                const QString message = !source_mask_write_result.ok ? source_mask_write_result.errorMessage
                                                                     : (!inverse_mean_write_result.ok
                                                                            ? inverse_mean_write_result.errorMessage
                                                                            : inverse_spread_write_result.errorMessage);
                errorOccurred(message);
                return false;
            }
            if (generate_adaptive_evidence)
            {
                const xjw::common::OperationResult support_weight_result = xjw::core::project::writeDepthMatStorage(
                    filtered_adaptive_support_weight_path, adaptive_evidence.supportWeight);
                const xjw::common::OperationResult effective_view_result = xjw::core::project::writeDepthMatStorage(
                    filtered_adaptive_effective_view_count_path, adaptive_evidence.effectiveViewCount);
                const xjw::common::OperationResult conflict_ratio_result = xjw::core::project::writeDepthMatStorage(
                    filtered_adaptive_conflict_ratio_path, adaptive_evidence.conflictRatio);
                if (!support_weight_result.ok || !effective_view_result.ok || !conflict_ratio_result.ok)
                {
                    QFile::remove(filtered_path);
                    QFile::remove(filtered_confidence_path);
                    QFile::remove(filtered_geometry_support_path);
                    QFile::remove(filtered_geometry_source_mask_path);
                    QFile::remove(filtered_inverse_depth_mean_path);
                    QFile::remove(filtered_inverse_depth_spread_path);
                    QFile::remove(filtered_adaptive_support_weight_path);
                    QFile::remove(filtered_adaptive_effective_view_count_path);
                    QFile::remove(filtered_adaptive_conflict_ratio_path);
                    remove_pending_files();
                    const QString message = !support_weight_result.ok
                                                ? support_weight_result.errorMessage
                                                : (!effective_view_result.ok ? effective_view_result.errorMessage
                                                                             : conflict_ratio_result.errorMessage);
                    errorOccurred(message);
                    return false;
                }
            }
            auto write_optional_pixel_map = [](const QString& path,
                                               const QSharedPointer<cv::Mat>& matrix) -> xjw::common::OperationResult
            {
                if (!matrix || matrix->empty())
                {
                    QFile::remove(path);
                    return {true, QString()};
                }
                return xjw::core::project::writeDepthMatStorage(path, *matrix);
            };
            const xjw::common::OperationResult repaired_mask_result = write_optional_pixel_map(
                filtered_cross_view_repaired_mask_path, _depthFrames[frame_index].crossViewRepairedMask);
            const xjw::common::OperationResult residual_mask_result = write_optional_pixel_map(
                filtered_residual_reestimated_mask_path, _depthFrames[frame_index].residualReestimatedMask);
            const xjw::common::OperationResult provenance_result =
                write_optional_pixel_map(filtered_depth_provenance_path, _depthFrames[frame_index].depthProvenance);
            const xjw::common::OperationResult missing_reason_result =
                write_optional_pixel_map(filtered_missing_reason_path, _depthFrames[frame_index].missingReasonMap);
            if (!repaired_mask_result.ok || !residual_mask_result.ok || !provenance_result.ok ||
                !missing_reason_result.ok)
            {
                const QString message =
                    !repaired_mask_result.ok
                        ? repaired_mask_result.errorMessage
                        : (!residual_mask_result.ok ? residual_mask_result.errorMessage
                                                    : (!provenance_result.ok ? provenance_result.errorMessage
                                                                             : missing_reason_result.errorMessage));
                remove_pending_files();
                errorOccurred(message);
                return false;
            }

            // The transactional temporary files now own these per-frame pixel
            // products. Keeping all masks resident until every frame completed was
            // responsible for the linear, stair-step memory growth in streaming
            // consistency. Metadata and diagnostics remain in DepthFrameResult.
            _depthFrames[frame_index].crossViewRepairedMask.clear();
            _depthFrames[frame_index].residualReestimatedMask.clear();
            _depthFrames[frame_index].depthProvenance.clear();
            _depthFrames[frame_index].missingReasonMap.clear();
            const auto write_stage_end = Clock::now();

            ++completed_frames;
            const float ratio = static_cast<float>(completed_frames) / static_cast<float>(std::max(1, view_count));
            progressChanged(QStringLiteral("多视一致性：已处理 %1/%2").arg(completed_frames).arg(view_count), ratio);
            LOG_INFO(QStringLiteral("[MVS] 流式一致性 frame=%1 workers=%2 valid=%3->%4 "
                                    "sources_preferred=%5 sources_frozen=%6 "
                                    "twoSource=%7/%8 cache=%9/%10 MiB "
                                    "timing_ms(source=%11 layer=%12 repair=%13 residual=%14 "
                                    "post=%15 write=%16 total=%17)")
                         .arg(frame_index)
                         .arg(row_workers)
                         .arg(valid_before)
                         .arg(valid_after)
                         .arg(source_indices.size())
                         .arg(repair_source_indices.size())
                         .arg(repair_stats.twoSourceGrownPixelCount)
                         .arg(repair_stats.twoSourceCandidatePixelCount)
                         .arg(cache.currentBytes() / (1024 * 1024))
                         .arg(cache.memoryBudgetBytes() / (1024 * 1024))
                         .arg(elapsedMs(frame_start, source_stage_end), 0, 'f', 1)
                         .arg(elapsedMs(source_stage_end, layer_stage_end), 0, 'f', 1)
                         .arg(elapsedMs(layer_stage_end, repair_stage_end), 0, 'f', 1)
                         .arg(elapsedMs(repair_stage_end, residual_stage_end), 0, 'f', 1)
                         .arg(elapsedMs(residual_stage_end, postprocess_stage_end), 0, 'f', 1)
                         .arg(elapsedMs(postprocess_stage_end, write_stage_end), 0, 'f', 1)
                         .arg(elapsedMs(frame_start, write_stage_end), 0, 'f', 1));
        }

        for (const PendingDepthReplacement& replacement : pending_replacements)
        {
            const QString backup_path = replacement.originalPath + QStringLiteral(".original.bin");
            QFile::remove(backup_path);

            if (!replacement.filteredConfidencePath.isEmpty())
            {
                const QString confidence_backup = replacement.originalConfidencePath + QStringLiteral(".original.bin");
                QFile::remove(confidence_backup);
                if (QFileInfo::exists(replacement.originalConfidencePath) &&
                    !QFile::rename(replacement.originalConfidencePath, confidence_backup))
                {
                    remove_pending_files();
                    errorOccurred(QStringLiteral("无法备份原始置信度缓存：%1").arg(replacement.originalConfidencePath));
                    return false;
                }
                if (!QFile::rename(replacement.filteredConfidencePath, replacement.originalConfidencePath))
                {
                    if (QFileInfo::exists(confidence_backup))
                    {
                        QFile::rename(confidence_backup, replacement.originalConfidencePath);
                    }
                    remove_pending_files();
                    errorOccurred(
                        QStringLiteral("无法提交一致性过滤置信度缓存：%1").arg(replacement.originalConfidencePath));
                    return false;
                }
                QFile::remove(confidence_backup);
            }
            if (!QFile::rename(replacement.originalPath, backup_path))
            {
                remove_pending_files();
                errorOccurred(QStringLiteral("无法备份原始深度缓存：%1").arg(replacement.originalPath));
                return false;
            }
            if (!QFile::rename(replacement.filteredPath, replacement.originalPath))
            {
                QFile::rename(backup_path, replacement.originalPath);
                remove_pending_files();
                errorOccurred(QStringLiteral("无法提交一致性过滤深度缓存：%1").arg(replacement.originalPath));
                return false;
            }
            QFile::remove(backup_path);

            cv::Mat filtered_depth;
            const xjw::common::OperationResult load_result =
                xjw::core::project::loadDepthMatStorage(replacement.originalPath, &filtered_depth);
            if (!load_result.ok || filtered_depth.empty() || filtered_depth.type() != CV_32FC1 ||
                filtered_depth.size() != replacement.expectedDepthSize)
            {
                remove_pending_files();
                const QString detail = !load_result.ok ? load_result.errorMessage : QStringLiteral("类型或尺寸无效");
                const QString message = QStringLiteral("流式一致性提交帧 %1 后无法重读权威深度：%2（%3）")
                                            .arg(replacement.frameIndex)
                                            .arg(replacement.originalPath)
                                            .arg(detail);
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                markManifestFrameFailed(replacement.frameIndex, message);
                errorOccurred(message);
                return false;
            }
            else
            {
                cv::Mat filtered_confidence;
                if (QFileInfo::exists(replacement.originalConfidencePath))
                {
                    (void)xjw::core::project::loadDepthMatStorage(replacement.originalConfidencePath,
                                                                  &filtered_confidence);
                }
                DepthFrameResult artifact_result = _depthFrames[replacement.frameIndex];
                artifact_result.depthMap = QSharedPointer<cv::Mat>::create(filtered_depth);
                artifact_result.confidence = QSharedPointer<cv::Mat>::create(filtered_confidence);
                const QString photometric_source_mask_path = storage_dir.filePath(
                    QStringLiteral("depth_%1_photometric_source_mask.bin").arg(replacement.frameIndex));
                cv::Mat photometric_source_mask;
                const xjw::common::OperationResult photometric_source_mask_result =
                    xjw::core::project::loadDepthMatStorage(photometric_source_mask_path, &photometric_source_mask);
                if (!photometric_source_mask_result.ok || photometric_source_mask.empty() ||
                    photometric_source_mask.type() != CV_32SC1 ||
                    photometric_source_mask.size() != filtered_depth.size())
                {
                    remove_pending_files();
                    const QString detail = photometric_source_mask_result.ok
                                               ? QStringLiteral("类型或尺寸无效")
                                               : photometric_source_mask_result.errorMessage;
                    errorOccurred(QStringLiteral("流式一致性无法恢复帧 %1 的 PatchMatch 光度来源掩码：%2（%3）")
                                      .arg(replacement.frameIndex)
                                      .arg(photometric_source_mask_path)
                                      .arg(detail));
                    return false;
                }
                artifact_result.photometricSourceMask =
                    QSharedPointer<cv::Mat>::create(std::move(photometric_source_mask));
                const QString targeted_recovered_path = storage_dir.filePath(
                    QStringLiteral("depth_%1_targeted_gap_recovered_mask.png").arg(replacement.frameIndex));
                cv::Mat targeted_recovered;
                if (replacement.targetedGapRecoveredMaskExpected)
                {
                    targeted_recovered = xjw::common::io::readImage(
                        xjw::common::io::toUtf8Path(targeted_recovered_path), cv::IMREAD_GRAYSCALE);
                    if (targeted_recovered.empty() || targeted_recovered.type() != CV_8UC1 ||
                        targeted_recovered.size() != filtered_depth.size())
                    {
                        remove_pending_files();
                        errorOccurred(QStringLiteral("流式一致性提交帧 %1 时无法恢复权威定向恢复掩码：%2")
                                          .arg(replacement.frameIndex)
                                          .arg(targeted_recovered_path));
                        return false;
                    }
                    artifact_result.targetedGapRecoveredMask = QSharedPointer<cv::Mat>::create(targeted_recovered);
                }
                const QString provenance_path =
                    storage_dir.filePath(QStringLiteral("depth_%1_provenance.png").arg(replacement.frameIndex));
                cv::Mat provenance =
                    xjw::common::io::readImage(xjw::common::io::toUtf8Path(provenance_path), cv::IMREAD_GRAYSCALE);
                if (!provenance.empty() && provenance.type() == CV_8UC1 && provenance.size() == filtered_depth.size())
                {
                    artifact_result.depthProvenance = QSharedPointer<cv::Mat>::create(provenance);
                }
                auto load_optional_pixel_map =
                    [filtered_depth](const QString& path,
                                     QSharedPointer<cv::Mat>& target) -> xjw::common::OperationResult
                {
                    if (path.isEmpty() || !QFileInfo::exists(path))
                    {
                        target.clear();
                        return {true, QString()};
                    }
                    cv::Mat matrix;
                    const xjw::common::OperationResult result = xjw::core::project::loadDepthMatStorage(path, &matrix);
                    if (!result.ok)
                    {
                        return result;
                    }
                    if (matrix.type() != CV_8UC1 || matrix.size() != filtered_depth.size())
                    {
                        return {false, QStringLiteral("流式一致性像素图格式无效：%1").arg(path)};
                    }
                    target = QSharedPointer<cv::Mat>::create(std::move(matrix));
                    return {true, QString()};
                };
                const xjw::common::OperationResult repaired_mask_result = load_optional_pixel_map(
                    replacement.filteredCrossViewRepairedMaskPath, artifact_result.crossViewRepairedMask);
                const xjw::common::OperationResult residual_mask_result = load_optional_pixel_map(
                    replacement.filteredResidualReestimatedMaskPath, artifact_result.residualReestimatedMask);
                const xjw::common::OperationResult provenance_result =
                    load_optional_pixel_map(replacement.filteredDepthProvenancePath, artifact_result.depthProvenance);
                const xjw::common::OperationResult missing_reason_result =
                    load_optional_pixel_map(replacement.filteredMissingReasonPath, artifact_result.missingReasonMap);
                if (!repaired_mask_result.ok || !residual_mask_result.ok || !provenance_result.ok ||
                    !missing_reason_result.ok)
                {
                    const QString message =
                        !repaired_mask_result.ok
                            ? repaired_mask_result.errorMessage
                            : (!residual_mask_result.ok ? residual_mask_result.errorMessage
                                                        : (!provenance_result.ok ? provenance_result.errorMessage
                                                                                 : missing_reason_result.errorMessage));
                    remove_pending_files();
                    errorOccurred(message);
                    return false;
                }
                cv::Mat geometry_support;
                const xjw::common::OperationResult geometry_support_result =
                    xjw::core::project::loadDepthMatStorage(replacement.filteredGeometrySupportPath, &geometry_support);
                if (!geometry_support_result.ok || geometry_support.empty())
                {
                    remove_pending_files();
                    errorOccurred(geometry_support_result.errorMessage);
                    return false;
                }
                artifact_result.geometrySupportCount = QSharedPointer<cv::Mat>::create(geometry_support);
                cv::Mat geometry_source_mask;
                cv::Mat inverse_depth_mean;
                cv::Mat inverse_depth_spread;
                const xjw::common::OperationResult source_mask_result = xjw::core::project::loadDepthMatStorage(
                    replacement.filteredGeometrySourceMaskPath, &geometry_source_mask);
                const xjw::common::OperationResult inverse_mean_result = xjw::core::project::loadDepthMatStorage(
                    replacement.filteredInverseDepthMeanPath, &inverse_depth_mean);
                const xjw::common::OperationResult inverse_spread_result = xjw::core::project::loadDepthMatStorage(
                    replacement.filteredInverseDepthSpreadPath, &inverse_depth_spread);
                if (!source_mask_result.ok || !inverse_mean_result.ok || !inverse_spread_result.ok)
                {
                    remove_pending_files();
                    const QString message = !source_mask_result.ok
                                                ? source_mask_result.errorMessage
                                                : (!inverse_mean_result.ok ? inverse_mean_result.errorMessage
                                                                           : inverse_spread_result.errorMessage);
                    errorOccurred(message);
                    return false;
                }
                artifact_result.geometrySourceMask = QSharedPointer<cv::Mat>::create(geometry_source_mask);
                artifact_result.inverseDepthMean = QSharedPointer<cv::Mat>::create(inverse_depth_mean);
                artifact_result.inverseDepthRelativeSpread = QSharedPointer<cv::Mat>::create(inverse_depth_spread);
                if (!replacement.filteredAdaptiveSupportWeightPath.isEmpty())
                {
                    cv::Mat adaptive_support_weight;
                    cv::Mat adaptive_effective_view_count;
                    cv::Mat adaptive_conflict_ratio;
                    const xjw::common::OperationResult adaptive_support_result =
                        xjw::core::project::loadDepthMatStorage(replacement.filteredAdaptiveSupportWeightPath,
                                                                &adaptive_support_weight);
                    const xjw::common::OperationResult adaptive_view_result = xjw::core::project::loadDepthMatStorage(
                        replacement.filteredAdaptiveEffectiveViewCountPath, &adaptive_effective_view_count);
                    const xjw::common::OperationResult adaptive_conflict_result =
                        xjw::core::project::loadDepthMatStorage(replacement.filteredAdaptiveConflictRatioPath,
                                                                &adaptive_conflict_ratio);
                    if (!adaptive_support_result.ok || !adaptive_view_result.ok || !adaptive_conflict_result.ok)
                    {
                        remove_pending_files();
                        const QString message =
                            !adaptive_support_result.ok
                                ? adaptive_support_result.errorMessage
                                : (!adaptive_view_result.ok ? adaptive_view_result.errorMessage
                                                            : adaptive_conflict_result.errorMessage);
                        errorOccurred(message);
                        return false;
                    }
                    artifact_result.adaptiveGeometrySupportWeight =
                        QSharedPointer<cv::Mat>::create(adaptive_support_weight);
                    artifact_result.adaptiveGeometryEffectiveViewCount =
                        QSharedPointer<cv::Mat>::create(adaptive_effective_view_count);
                    artifact_result.adaptiveGeometryConflictRatio =
                        QSharedPointer<cv::Mat>::create(adaptive_conflict_ratio);
                }
                captureStageSnapshot(replacement.frameIndex,
                                     MvsStageSnapshotStage::FinalAdmission,
                                     QStringLiteral("after_final_quality_evaluation_before_artifact_publication"),
                                     artifact_result,
                                     artifact_result.depthMap ? *artifact_result.depthMap : cv::Mat(),
                                     artifact_result.confidence ? *artifact_result.confidence : cv::Mat(),
                                     artifact_result.validMask ? *artifact_result.validMask : cv::Mat());
                if (!saveDepthFrameArtifacts(replacement.frameIndex, artifact_result, QStringLiteral("一致性过滤后")))
                {
                    remove_pending_files();
                    return false;
                }
                const auto remove_if_present = [](const QString& path)
                {
                    if (!path.isEmpty())
                    {
                        QFile::remove(path);
                    }
                };
                remove_if_present(replacement.filteredGeometrySupportPath);
                remove_if_present(replacement.filteredGeometrySourceMaskPath);
                remove_if_present(replacement.filteredInverseDepthMeanPath);
                remove_if_present(replacement.filteredInverseDepthSpreadPath);
                remove_if_present(replacement.filteredAdaptiveSupportWeightPath);
                remove_if_present(replacement.filteredAdaptiveEffectiveViewCountPath);
                remove_if_present(replacement.filteredAdaptiveConflictRatioPath);
                remove_if_present(replacement.filteredCrossViewRepairedMaskPath);
                remove_if_present(replacement.filteredResidualReestimatedMaskPath);
                remove_if_present(replacement.filteredDepthProvenancePath);
                remove_if_present(replacement.filteredMissingReasonPath);
            }
        }
        LOG_INFO(QStringLiteral("[MVS] 流式一致性检查完成：frames=%1 cachePeak=%2 MiB budget=%3 MiB")
                     .arg(pending_replacements.size())
                     .arg(cache.peakBytes() / (1024 * 1024))
                     .arg(cache.memoryBudgetBytes() / (1024 * 1024)));
        return true;
    }
} // namespace xjw::mvs
