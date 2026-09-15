#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    bool MvsPipelineService::saveDepthFrameArtifacts(int frameIndex,
                                                     const DepthFrameResult& result,
                                                     const QString& stageLabel)
    {
        if (frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()) || !result.success || !result.depthMap ||
            result.depthMap->empty())
        {
            return true;
        }

        const bool final_artifacts = stageLabel != QStringLiteral("初始");
        const bool consistency_publication_expected =
            detail::expectsConsistencyPublication(result, static_cast<int>(_views.size()));
        const bool consistency_publication_completed = detail::hasCompletedConsistencyPublication(result);
        if (final_artifacts && consistency_publication_expected != consistency_publication_completed)
        {
            const QString state_error = consistency_publication_expected
                                            ? QStringLiteral("期望多视一致性证据，但三项发布计数未完整记录")
                                            : QStringLiteral("未进入多视一致性阶段，却携带了已完成的发布计数");
            const QString message = QStringLiteral("帧 %1 的%2 MVS 一致性发布状态无效：%3；"
                                                   "已在写盘和公共发布前失败关闭")
                                        .arg(frameIndex)
                                        .arg(stageLabel)
                                        .arg(state_error);
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            markManifestFrameFailed(frameIndex, message);
            errorOccurred(message);
            return false;
        }

        const bool durable_publication = !_workspaceManifestPath.isEmpty();
        const std::string preview_directory =
            !_outputDir.empty() ? _outputDir : (durable_publication ? _config.intermediateDir : std::string());
        const bool savePreviewPng = !preview_directory.empty();
        const bool has_raw_directory =
            !_config.intermediateDir.empty() || !_outputDir.empty() || _streamConsistencyStorageEnabled;
        const bool saveRawDepth =
            has_raw_directory && ((_config.saveIntermediateDepthMaps || _config.saveIntermediatePyramidLevels) ||
                                  _streamConsistencyStorageEnabled || durable_publication);
        if (!savePreviewPng && !saveRawDepth)
        {
            return true;
        }

        std::string saveErr;
        const auto saveStart = Clock::now();
        const std::string pngPath = preview_directory + "/depth_" + std::to_string(frameIndex) + ".png";
        const std::string raw_directory =
            _streamConsistencyStorageEnabled
                ? _consistencyDepthDirectory
                : (!_config.intermediateDir.empty() ? _config.intermediateDir : _outputDir);
        const std::string rawDepthPath = raw_directory + "/depth_" + std::to_string(frameIndex) + ".bin";
        const std::string rawConfidencePath = raw_directory + "/depth_" + std::to_string(frameIndex) + "_conf.bin";
        const std::string rawPhotometricSourceMaskPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_photometric_source_mask.bin";
        const std::string rawGeometrySupportPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_geometry_support.bin";
        const std::string rawGeometrySourceMaskPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_geometry_source_mask.bin";
        const std::string rawInverseDepthMeanPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_inverse_depth_mean.bin";
        const std::string rawInverseDepthSpreadPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_inverse_depth_spread.bin";
        const std::string rawAdaptiveGeometrySupportWeightPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_adaptive_geometry_support_weight.bin";
        const std::string rawAdaptiveGeometryEffectiveViewCountPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_adaptive_geometry_effective_view_count.bin";
        const std::string rawAdaptiveGeometryConflictRatioPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_adaptive_geometry_conflict_ratio.bin";
        const std::string crossViewRepairedMaskPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_cross_view_repaired_mask.png";
        const std::string targetedGapRecoveredMaskPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_targeted_gap_recovered_mask.png";
        const std::string residualReestimatedMaskPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_residual_reestimated_mask.png";
        const std::string depthProvenancePath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_provenance.png";
        const std::string validMaskPath = raw_directory + "/depth_" + std::to_string(frameIndex) + "_mask.png";
        const std::string supportMaskPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_support_mask.png";
        const std::string missingReasonPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_missing_reason.png";
        const std::string missingReasonPreviewPath =
            raw_directory + "/depth_" + std::to_string(frameIndex) + "_missing_reason_preview.png";
        const GeometrySourceOrdinalContract geometry_source_contract = validateGeometrySourceOrdinalContract(
            result.geometrySourceMask && !result.geometrySourceMask->empty() ? *result.geometrySourceMask : cv::Mat(),
            result.geometrySourceViewIndices,
            frameIndex,
            static_cast<int>(_views.size()),
            result.depthMap->size());
        if (!geometry_source_contract.valid)
        {
            const QString message = QStringLiteral("帧 %1 的几何来源位序契约无效：%2（来源数=%3）；拒绝写盘以避免 "
                                                   "geometry_source_mask 被错解")
                                        .arg(frameIndex)
                                        .arg(geometry_source_contract.errorMessage)
                                        .arg(result.geometrySourceViewIndices.size());
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            markManifestFrameFailed(frameIndex, message);
            errorOccurred(message);
            return false;
        }
        bool previewSaved = !savePreviewPng;
        double previewMs = 0.0;
        if (savePreviewPng)
        {
            const auto previewStart = Clock::now();
            if (!saveDepthPreviewPng(pngPath, *result.depthMap, &saveErr))
            {
                LOG_WARN(
                    QStringLiteral("[MVS] 保存%1深度预览失败: %2").arg(stageLabel, QString::fromStdString(saveErr)));
                errorOccurred(QString::fromStdString(saveErr));
                markManifestFrameFailed(frameIndex, QString::fromStdString(saveErr));
                return false;
            }

            previewMs = elapsedMs(previewStart, Clock::now());
            previewSaved = true;
            LOG_DEBUG(QStringLiteral("[MVS] 帧 %1 %2深度预览已保存: %3 (%4x%5)")
                          .arg(frameIndex)
                          .arg(stageLabel)
                          .arg(QString::fromStdString(pngPath))
                          .arg(result.depthMap->cols)
                          .arg(result.depthMap->rows));
        }

        bool rawSaved = true;
        double rawMs = 0.0;
        if (saveRawDepth)
        {
            const auto rawStart = Clock::now();
            if (!writeFastDepthMatStorage(rawDepthPath, *result.depthMap, &saveErr))
            {
                rawSaved = false;
                LOG_WARN(
                    QStringLiteral("[MVS] 保存%1原始深度失败: %2").arg(stageLabel, QString::fromStdString(saveErr)));
                errorOccurred(QString::fromStdString(saveErr));
                markManifestFrameFailed(frameIndex, QString::fromStdString(saveErr));
            }
            rawMs = elapsedMs(rawStart, Clock::now());
        }

        double confidenceMs = 0.0;
        bool confidenceSaved = false;
        if (saveRawDepth && result.confidence && !result.confidence->empty())
        {
            const auto confidenceStart = Clock::now();
            if (!writeFastDepthMatStorage(rawConfidencePath, *result.confidence, &saveErr))
            {
                LOG_WARN(QStringLiteral("[MVS] 保存%1置信图失败: %2").arg(stageLabel, QString::fromStdString(saveErr)));
            }
            else
            {
                confidenceSaved = true;
            }
            confidenceMs = elapsedMs(confidenceStart, Clock::now());
        }

        bool maskSaved = false;
        bool supportMaskSaved = false;
        cv::Mat supportMask;
        double maskMs = 0.0;
        if (saveRawDepth)
        {
            const auto maskStart = Clock::now();
            cv::Mat validMask = (*result.depthMap > 0.0f);
            if (!validMask.empty())
            {
                maskSaved = xjw::common::io::writeImage(validMaskPath, validMask);
                if (!maskSaved)
                {
                    LOG_WARN(QStringLiteral("[MVS] 保存%1有效掩码失败: %2")
                                 .arg(stageLabel, QString::fromStdString(validMaskPath)));
                }
            }
            if (!result.supportRegionMask || result.supportRegionMask->empty())
            {
                supportMask = cv::Mat(result.depthMap->size(), CV_8UC1, cv::Scalar(255));
            }
            else
            {
                if (result.supportRegionMask->size() == result.depthMap->size())
                {
                    supportMask = result.supportRegionMask->clone();
                }
                else
                {
                    cv::resize(
                        *result.supportRegionMask, supportMask, result.depthMap->size(), 0.0, 0.0, cv::INTER_NEAREST);
                }
                if (supportMask.type() != CV_8UC1)
                {
                    supportMask.convertTo(supportMask, CV_8UC1);
                }
                cv::threshold(supportMask, supportMask, 0.0, 255.0, cv::THRESH_BINARY);
            }
            supportMaskSaved = xjw::common::io::writeImage(supportMaskPath, supportMask);
            if (!supportMaskSaved)
            {
                LOG_WARN(QStringLiteral("[MVS] 保存%1支持掩码失败: %2")
                             .arg(stageLabel, QString::fromStdString(supportMaskPath)));
            }
            maskMs = elapsedMs(maskStart, Clock::now());
        }

        bool missingReasonSaved = false;
        bool missingReasonPreviewSaved = false;
        QJsonObject missingReasonSummaryJson;
        if (saveRawDepth && !supportMask.empty())
        {
            cv::Mat missing_reason = result.missingReasonMap && !result.missingReasonMap->empty()
                                         ? result.missingReasonMap->clone()
                                         : initializeDepthMissingReasonMap(*result.depthMap, supportMask);
            finalizeDepthMissingReasonMap(missing_reason,
                                          *result.depthMap,
                                          supportMask,
                                          result.geometrySupportCount && !result.geometrySupportCount->empty()
                                              ? *result.geometrySupportCount
                                              : cv::Mat());
            const DepthMissingReasonSummary missing_summary = summarizeDepthMissingReasons(missing_reason, supportMask);
            missingReasonSummaryJson = depthMissingReasonSummaryToJson(missing_summary);
            missingReasonSaved = xjw::common::io::writeImage(missingReasonPath, missing_reason);
            const cv::Mat missing_preview = makeDepthMissingReasonPreview(missing_reason);
            missingReasonPreviewSaved =
                !missing_preview.empty() && xjw::common::io::writeImage(missingReasonPreviewPath, missing_preview);
            if (!missingReasonSaved || !missingReasonPreviewSaved)
            {
                LOG_WARN(QStringLiteral("[MVS] 保存%1深度缺失原因诊断失败: frame=%2").arg(stageLabel).arg(frameIndex));
            }
        }

        bool geometrySupportSaved = false;
        if (saveRawDepth && result.geometrySupportCount && !result.geometrySupportCount->empty())
        {
            cv::Mat geometry_support = result.geometrySupportCount->clone();
            if (geometry_support.type() == CV_16UC1 && geometry_support.size() == result.depthMap->size())
            {
                geometry_support.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
                geometrySupportSaved = writeFastDepthMatStorage(rawGeometrySupportPath, geometry_support, &saveErr);
            }
            if (!geometrySupportSaved)
            {
                LOG_WARN(QStringLiteral("[MVS] 保存%1跨视几何支持图失败: %2")
                             .arg(stageLabel, QString::fromStdString(saveErr)));
            }
        }

        auto save_geometry_evidence = [&](const QSharedPointer<cv::Mat>& evidence,
                                          int expected_type,
                                          const std::string& path,
                                          const QString& label,
                                          bool mask_invalid_depth)
        {
            if (!saveRawDepth || !evidence || evidence->empty())
            {
                return false;
            }
            cv::Mat stored = evidence->clone();
            if (stored.type() != expected_type || stored.size() != result.depthMap->size())
            {
                LOG_WARN(QStringLiteral("[MVS] 跳过帧 %1 %2：类型或尺寸不匹配").arg(frameIndex).arg(label));
                return false;
            }
            if (mask_invalid_depth)
            {
                stored.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
            }
            if (!writeFastDepthMatStorage(path, stored, &saveErr))
            {
                LOG_WARN(QStringLiteral("[MVS] 保存帧 %1 %2失败: %3")
                             .arg(frameIndex)
                             .arg(label, QString::fromStdString(saveErr)));
                return false;
            }
            return true;
        };
        const bool photometricSourceMaskSaved = save_geometry_evidence(result.photometricSourceMask,
                                                                       CV_32SC1,
                                                                       rawPhotometricSourceMaskPath,
                                                                       QStringLiteral("PatchMatch 光度来源掩码"),
                                                                       true);
        if (saveRawDepth && !photometricSourceMaskSaved)
        {
            const QString message =
                QStringLiteral("帧 %1 的 PatchMatch 光度来源掩码写盘失败；拒绝发布不完整的 revision-%2 工件")
                    .arg(frameIndex)
                    .arg(kMvsDepthAlgorithmRevision);
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            markManifestFrameFailed(frameIndex, message);
            errorOccurred(message);
            return false;
        }
        bool geometrySourceMaskSaved = false;
        if (geometry_source_contract.persistMask)
        {
            geometrySourceMaskSaved = save_geometry_evidence(
                result.geometrySourceMask, CV_16UC1, rawGeometrySourceMaskPath, QStringLiteral("跨视来源掩码"), true);
            if (saveRawDepth && !geometrySourceMaskSaved)
            {
                const QString message =
                    QStringLiteral("帧 %1 的几何来源掩码写盘失败；拒绝保存不完整的位序契约").arg(frameIndex);
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                markManifestFrameFailed(frameIndex, message);
                errorOccurred(message);
                return false;
            }
        }
        else if (saveRawDepth)
        {
            const QString normalized_mask_path = QString::fromStdString(rawGeometrySourceMaskPath);
            if (QFileInfo::exists(normalized_mask_path) && !QFile::remove(normalized_mask_path))
            {
                const QString message = QStringLiteral("帧 %1 无跨视来源证据，但无法移除旧来源掩码：%2")
                                            .arg(frameIndex)
                                            .arg(normalized_mask_path);
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                markManifestFrameFailed(frameIndex, message);
                errorOccurred(message);
                return false;
            }
        }
        const bool inverseDepthMeanSaved = save_geometry_evidence(
            result.inverseDepthMean, CV_32FC1, rawInverseDepthMeanPath, QStringLiteral("逆深度均值图"), true);
        const bool inverseDepthSpreadSaved = save_geometry_evidence(result.inverseDepthRelativeSpread,
                                                                    CV_32FC1,
                                                                    rawInverseDepthSpreadPath,
                                                                    QStringLiteral("逆深度离散度图"),
                                                                    true);
        const bool adaptiveGeometrySupportWeightSaved = save_geometry_evidence(result.adaptiveGeometrySupportWeight,
                                                                               CV_32FC1,
                                                                               rawAdaptiveGeometrySupportWeightPath,
                                                                               QStringLiteral("连续几何支持权重"),
                                                                               false);
        const bool adaptiveGeometryEffectiveViewCountSaved =
            save_geometry_evidence(result.adaptiveGeometryEffectiveViewCount,
                                   CV_32FC1,
                                   rawAdaptiveGeometryEffectiveViewCountPath,
                                   QStringLiteral("连续几何有效视图数"),
                                   false);
        const bool adaptiveGeometryConflictRatioSaved = save_geometry_evidence(result.adaptiveGeometryConflictRatio,
                                                                               CV_32FC1,
                                                                               rawAdaptiveGeometryConflictRatioPath,
                                                                               QStringLiteral("连续几何冲突比例"),
                                                                               false);
        // 连续几何证据由所有原始深度完成后的跨视一致性阶段生成。这里保存的是
        // 单帧初始产物，此时证据图尚不存在，不能据此把有效深度帧标记为失败。
        // 一致性阶段会原子写入三张 revision 14 证据图，并在任一写入失败时终止。
        bool crossViewRepairedMaskSaved = false;
        if (saveRawDepth && result.crossViewRepairedMask && !result.crossViewRepairedMask->empty())
        {
            cv::Mat repaired_mask = result.crossViewRepairedMask->clone();
            if (repaired_mask.size() == result.depthMap->size())
            {
                if (repaired_mask.type() != CV_8UC1)
                {
                    repaired_mask.convertTo(repaired_mask, CV_8UC1);
                }
                cv::threshold(repaired_mask, repaired_mask, 0.0, 255.0, cv::THRESH_BINARY);
                repaired_mask.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
                crossViewRepairedMaskSaved = xjw::common::io::writeImage(crossViewRepairedMaskPath, repaired_mask);
            }
        }
        bool targetedGapRecoveredMaskSaved = false;
        const bool targeted_gap_recovered_mask_expected = saveRawDepth && result.targetedGapRecoveredMaskExpected;
        if (targeted_gap_recovered_mask_expected)
        {
            if (!result.targetedGapRecoveredMask || result.targetedGapRecoveredMask->empty())
            {
                const QString message = QStringLiteral("帧 %1 声明需要定向恢复掩码，"
                                                       "但内存工件为空；拒绝发布")
                                            .arg(frameIndex);
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                markManifestFrameFailed(frameIndex, message);
                errorOccurred(message);
                return false;
            }
            cv::Mat recovered_mask = result.targetedGapRecoveredMask->clone();
            if (recovered_mask.size() == result.depthMap->size())
            {
                if (recovered_mask.type() != CV_8UC1)
                {
                    recovered_mask.convertTo(recovered_mask, CV_8UC1);
                }
                cv::threshold(recovered_mask, recovered_mask, 0.0, 255.0, cv::THRESH_BINARY);
                recovered_mask.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
                targetedGapRecoveredMaskSaved =
                    xjw::common::io::writeImage(targetedGapRecoveredMaskPath, recovered_mask);
            }
        }
        else if (saveRawDepth && result.targetedGapRecoveredMask && !result.targetedGapRecoveredMask->empty())
        {
            const QString message = QStringLiteral("帧 %1 包含定向恢复掩码，"
                                                   "但权威存在性标志为 false；拒绝发布")
                                        .arg(frameIndex);
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            markManifestFrameFailed(frameIndex, message);
            errorOccurred(message);
            return false;
        }
        if (targeted_gap_recovered_mask_expected && !targetedGapRecoveredMaskSaved)
        {
            const QString message = QStringLiteral("帧 %1 的定向恢复掩码写盘失败；"
                                                   "拒绝发布可能改变流式置信度语义的工件")
                                        .arg(frameIndex);
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            markManifestFrameFailed(frameIndex, message);
            errorOccurred(message);
            return false;
        }
        if (saveRawDepth && !targeted_gap_recovered_mask_expected)
        {
            const QString stale_path = QString::fromStdString(targetedGapRecoveredMaskPath);
            if (QFileInfo::exists(stale_path) && !QFile::remove(stale_path))
            {
                const QString message =
                    QStringLiteral("帧 %1 当前无定向恢复证据，但无法移除旧掩码：%2").arg(frameIndex).arg(stale_path);
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                markManifestFrameFailed(frameIndex, message);
                errorOccurred(message);
                return false;
            }
        }
        bool residualReestimatedMaskSaved = false;
        if (saveRawDepth && result.residualReestimatedMask && !result.residualReestimatedMask->empty())
        {
            cv::Mat recovered_mask = result.residualReestimatedMask->clone();
            if (recovered_mask.size() == result.depthMap->size())
            {
                if (recovered_mask.type() != CV_8UC1)
                {
                    recovered_mask.convertTo(recovered_mask, CV_8UC1);
                }
                cv::threshold(recovered_mask, recovered_mask, 0.0, 255.0, cv::THRESH_BINARY);
                recovered_mask.setTo(cv::Scalar(0), *result.depthMap <= 0.0f);
                residualReestimatedMaskSaved = xjw::common::io::writeImage(residualReestimatedMaskPath, recovered_mask);
            }
        }
        bool depthProvenanceSaved = false;
        QJsonObject depthProvenanceSummaryJson;
        if (saveRawDepth)
        {
            cv::Mat provenance =
                result.depthProvenance && !result.depthProvenance->empty()
                    ? result.depthProvenance->clone()
                    : initializeDepthProvenance(*result.depthMap,
                                                result.targetedGapRecoveredMask ? *result.targetedGapRecoveredMask
                                                                                : cv::Mat());
            updateDepthProvenance(provenance,
                                  *result.depthMap,
                                  result.targetedGapRecoveredMask ? *result.targetedGapRecoveredMask : cv::Mat(),
                                  result.crossViewRepairedMask ? *result.crossViewRepairedMask : cv::Mat(),
                                  cv::Mat(),
                                  result.residualReestimatedMask ? *result.residualReestimatedMask : cv::Mat());
            depthProvenanceSummaryJson =
                depthProvenanceSummaryToJson(summarizeDepthProvenance(provenance, *result.depthMap));
            depthProvenanceSaved = xjw::common::io::writeImage(depthProvenancePath, provenance);
            if (!depthProvenanceSaved)
            {
                LOG_WARN(QStringLiteral("[MVS] 保存%1深度来源图失败: frame=%2").arg(stageLabel).arg(frameIndex));
            }
        }

        std::unordered_map<int, QJsonObject> pyramid_level_paths;
        if (_config.saveIntermediatePyramidLevels && saveRawDepth)
        {
            for (const DepthLevelResult& level : result.intermediatePyramidLevels)
            {
                if (level.depth.empty())
                {
                    continue;
                }

                // Estimator-owned intermediate levels are already normalized to their native
                // PatchMatch grid.  Do not derive this size from the final result: the experimental
                // native-final mode intentionally makes that result smaller than the prepared raster.
                const cv::Size working_size = level.depth.size();
                const cv::Mat native_depth = restoreNativePyramidArtifact(level.depth, working_size);
                const cv::Mat native_confidence = restoreNativePyramidArtifact(level.confidence, working_size);
                const cv::Mat native_support = restoreNativePyramidArtifact(level.supportCount, working_size);
                const cv::Mat native_uncertainty = restoreNativePyramidArtifact(level.uncertainty, working_size);
                const cv::Mat native_mask = restoreNativePyramidArtifact(level.validMask, working_size);

                const std::string level_prefix =
                    raw_directory + "/depth_" + std::to_string(frameIndex) + "_level_" + std::to_string(level.level);
                const std::string level_depth_path = level_prefix + ".bin";
                const std::string level_confidence_path = level_prefix + "_conf.bin";
                const std::string level_support_path = level_prefix + "_support.bin";
                const std::string level_uncertainty_path = level_prefix + "_uncertainty.bin";
                const std::string level_mask_path = level_prefix + "_mask.png";
                const std::string level_preview_path = preview_directory + "/depth_" + std::to_string(frameIndex) +
                                                       "_level_" + std::to_string(level.level) + ".png";
                const std::string level_confidence_preview_path = preview_directory + "/depth_" +
                                                                  std::to_string(frameIndex) + "_level_" +
                                                                  std::to_string(level.level) + "_conf.png";
                if (!writeFastDepthMatStorage(level_depth_path, native_depth, &saveErr))
                {
                    LOG_WARN(QStringLiteral("[MVS] 保存帧 %1 Level %2 深度失败: %3")
                                 .arg(frameIndex)
                                 .arg(level.level)
                                 .arg(QString::fromStdString(saveErr)));
                    continue;
                }

                QJsonObject paths;
                paths.insert(QStringLiteral("raw_depth_path"), QString::fromStdString(level_depth_path));
                paths.insert(QStringLiteral("artifact_width"), working_size.width);
                paths.insert(QStringLiteral("artifact_height"), working_size.height);
                if (!native_confidence.empty())
                {
                    if (writeFastDepthMatStorage(level_confidence_path, native_confidence, &saveErr))
                    {
                        paths.insert(QStringLiteral("raw_confidence_path"),
                                     QString::fromStdString(level_confidence_path));
                    }
                    else
                    {
                        LOG_WARN(QStringLiteral("[MVS] 保存帧 %1 Level %2 置信图失败: %3")
                                     .arg(frameIndex)
                                     .arg(level.level)
                                     .arg(QString::fromStdString(saveErr)));
                    }
                }

                if (!native_support.empty() && writeFastDepthMatStorage(level_support_path, native_support, &saveErr))
                {
                    paths.insert(QStringLiteral("raw_support_count_path"), QString::fromStdString(level_support_path));
                }
                if (!native_uncertainty.empty() &&
                    writeFastDepthMatStorage(level_uncertainty_path, native_uncertainty, &saveErr))
                {
                    paths.insert(QStringLiteral("raw_uncertainty_path"),
                                 QString::fromStdString(level_uncertainty_path));
                }
                if (!native_mask.empty() && xjw::common::io::writeImage(level_mask_path, native_mask))
                {
                    paths.insert(QStringLiteral("valid_mask_path"), QString::fromStdString(level_mask_path));
                }

                if (savePreviewPng)
                {
                    if (saveDepthPreviewPng(level_preview_path, native_depth, &saveErr))
                    {
                        paths.insert(QStringLiteral("preview_path"), QString::fromStdString(level_preview_path));
                    }
                    if (!native_confidence.empty())
                    {
                        cv::Mat confidence_preview;
                        native_confidence.convertTo(confidence_preview, CV_8U, 255.0);
                        if (xjw::common::io::writeImage(level_confidence_preview_path, confidence_preview))
                        {
                            paths.insert(QStringLiteral("confidence_preview_path"),
                                         QString::fromStdString(level_confidence_preview_path));
                        }
                    }
                }
                pyramid_level_paths.emplace(level.level, std::move(paths));
            }
        }

        LOG_DEBUG(QStringLiteral("[MVS] 保存%1深度产物耗时: frame=%2 preview=%3 ms raw=%4 ms confidence=%5 ms "
                                 "mask=%6 ms total=%7 ms")
                      .arg(stageLabel)
                      .arg(frameIndex)
                      .arg(previewMs, 0, 'f', 1)
                      .arg(rawMs, 0, 'f', 1)
                      .arg(confidenceMs, 0, 'f', 1)
                      .arg(maskMs, 0, 'f', 1)
                      .arg(elapsedMs(saveStart, Clock::now()), 0, 'f', 1));

        QStringList missing_required_artifacts;
        const auto require_artifact = [&missing_required_artifacts](bool saved, const QString& label)
        {
            if (!saved)
            {
                missing_required_artifacts.push_back(label);
            }
        };
        if (saveRawDepth)
        {
            require_artifact(rawSaved, QStringLiteral("原始深度"));
            require_artifact(confidenceSaved, QStringLiteral("原始置信度"));
            require_artifact(maskSaved, QStringLiteral("有效掩码"));
            require_artifact(supportMaskSaved, QStringLiteral("支持区域掩码"));
            require_artifact(depthProvenanceSaved, QStringLiteral("深度来源图"));
            require_artifact(missingReasonSaved, QStringLiteral("深度缺失原因图"));

            const bool final_consistency_artifacts = final_artifacts && consistency_publication_completed;
            if (final_consistency_artifacts)
            {
                require_artifact(geometrySupportSaved, QStringLiteral("跨视图几何支持图"));
                require_artifact(inverseDepthSpreadSaved, QStringLiteral("逆深度离散度图"));
            }
            if (final_consistency_artifacts && _effectiveSceneProfile == MvsSceneProfile::OrbitalObject)
            {
                require_artifact(inverseDepthMeanSaved, QStringLiteral("逆深度均值图"));
                require_artifact(crossViewRepairedMaskSaved, QStringLiteral("跨视图修复掩码"));
                require_artifact(adaptiveGeometrySupportWeightSaved, QStringLiteral("连续几何支持权重图"));
                require_artifact(adaptiveGeometryEffectiveViewCountSaved, QStringLiteral("连续几何有效视图数图"));
                require_artifact(adaptiveGeometryConflictRatioSaved, QStringLiteral("连续几何冲突比例图"));
            }
        }
        if (!missing_required_artifacts.empty())
        {
            const QString message =
                QStringLiteral("帧 %1 的%2 MVS 工件写盘不完整，缺少或写入失败：%3；已拒绝发布并标记该帧失败")
                    .arg(frameIndex)
                    .arg(stageLabel)
                    .arg(missing_required_artifacts.join(QStringLiteral("、")));
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            markManifestFrameFailed(frameIndex, message);
            errorOccurred(message);
            return false;
        }

        if (previewSaved && rawSaved && savePreviewPng)
        {
            MvsPreparedRasterArtifact prepared_raster;
            QString prepared_raster_error;
            if (!ensurePreparedRasterArtifact(frameIndex, &prepared_raster, &prepared_raster_error))
            {
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(prepared_raster_error));
                errorOccurred(prepared_raster_error);
                markManifestFrameFailed(frameIndex, prepared_raster_error);
                return false;
            }
            QJsonArray sourceImages;
            QJsonArray sourceIndices;
            QJsonArray geometrySourceIndices;
            QJsonArray sourcePlan;
            QStringList sourceImageList;
            const bool hasResultSourcePlan = !result.sourceViewPlan.empty();
            const bool hasSourceScoreCache = frameIndex >= 0 && frameIndex < static_cast<int>(_frameCaches.size()) &&
                                             !_frameCaches[static_cast<size_t>(frameIndex)].sourceViewScores.empty();
            for (const int sourceIndex : result.sourceViewIndices)
            {
                if (sourceIndex < 0 || sourceIndex >= static_cast<int>(_views.size()))
                {
                    continue;
                }
                sourceIndices.append(sourceIndex);
                const QString sourceImage = QString::fromStdString(_views[sourceIndex].imagePath);
                sourceImages.append(sourceImage);
                sourceImageList.append(sourceImage);
                const std::vector<MvsSourcePlanEntry>* scores = nullptr;
                if (hasResultSourcePlan)
                {
                    scores = &result.sourceViewPlan;
                }
                else if (hasSourceScoreCache)
                {
                    scores = &_frameCaches[static_cast<size_t>(frameIndex)].sourceViewScores;
                }
                if (scores)
                {
                    const auto it = std::find_if(scores->cbegin(),
                                                 scores->cend(),
                                                 [sourceIndex](const MvsSourcePlanEntry& entry)
                                                 { return entry.viewIndex == sourceIndex; });
                    if (it != scores->cend())
                    {
                        QJsonObject sourcePlanEntry = mvsSourcePlanEntryToJson(*it);
                        sourcePlanEntry.insert(QStringLiteral("source_image"), sourceImage);
                        sourcePlan.append(sourcePlanEntry);
                    }
                }
            }

            const SourceQualitySummary sourceQualitySummary =
                summarizeSourceQuality(sourcePlan, sourceImageList.size());
            const cv::Mat* confidenceMap =
                (result.confidence && !result.confidence->empty()) ? result.confidence.data() : nullptr;
            const DepthConfidenceSummary depthConfidenceSummary =
                summarizeDepthConfidence(*result.depthMap, confidenceMap);
            DepthCompletenessDiagnostics depthCompleteness = result.depthCompleteness;
            cv::Mat completenessMask;
            if (result.supportRegionMask && !result.supportRegionMask->empty())
            {
                completenessMask = *result.supportRegionMask;
                if (completenessMask.size() != result.depthMap->size())
                {
                    cv::resize(
                        completenessMask, completenessMask, result.depthMap->size(), 0.0, 0.0, cv::INTER_NEAREST);
                }
            }
            else
            {
                completenessMask = cv::Mat(result.depthMap->size(), CV_8UC1, cv::Scalar(255));
            }
            depthCompleteness.finalMetrics =
                analyzeDepthCompleteness(*result.depthMap,
                                         completenessMask,
                                         kSmallHoleAreaFraction,
                                         effectiveMinimumSmallHoleArea(result, result.depthMap->size()));
            if (result.depthPostprocessApplied)
            {
                depthCompleteness.preFusionPostprocessValidCount = result.depthPostprocess.validBeforePostprocess;
                depthCompleteness.postConfidenceFilterValidCount = result.depthPostprocess.validAfterConfidenceFilter;
                depthCompleteness.postFusionPostprocessValidCount = result.depthPostprocess.validAfterPostprocess;
            }
            const QJsonObject depthCompletenessJson = depthCompletenessDiagnosticsToJson(depthCompleteness);
            const cv::Mat empty_geometry_evidence;
            QJsonObject geometryEvidenceDiagnostics = geometryEvidenceDiagnosticsToJson(
                *result.depthMap,
                result.geometrySupportCount && !result.geometrySupportCount->empty() ? *result.geometrySupportCount
                                                                                     : empty_geometry_evidence,
                result.inverseDepthRelativeSpread && !result.inverseDepthRelativeSpread->empty()
                    ? *result.inverseDepthRelativeSpread
                    : empty_geometry_evidence,
                result.crossViewRepairedMask && !result.crossViewRepairedMask->empty() ? *result.crossViewRepairedMask
                                                                                       : empty_geometry_evidence,
                result.supportRegionMask && !result.supportRegionMask->empty() ? *result.supportRegionMask
                                                                               : empty_geometry_evidence);
            const DiscreteGeometryCoreSummary discrete_geometry_core = summarizeDiscreteGeometryCore(
                *result.depthMap,
                result.geometrySupportCount && !result.geometrySupportCount->empty() ? *result.geometrySupportCount
                                                                                     : empty_geometry_evidence,
                result.inverseDepthRelativeSpread && !result.inverseDepthRelativeSpread->empty()
                    ? *result.inverseDepthRelativeSpread
                    : empty_geometry_evidence,
                result.supportRegionMask && !result.supportRegionMask->empty() ? *result.supportRegionMask
                                                                               : empty_geometry_evidence);
            geometryEvidenceDiagnostics.insert(QStringLiteral("discrete_geometry_core_available"),
                                               discrete_geometry_core.validInputs);
            if (discrete_geometry_core.validInputs)
            {
                geometryEvidenceDiagnostics.insert(QStringLiteral("discrete_geometry_core_valid_pixel_count"),
                                                   discrete_geometry_core.validPixelCount);
                geometryEvidenceDiagnostics.insert(QStringLiteral("discrete_geometry_core_pixel_count"),
                                                   discrete_geometry_core.corePixelCount);
                geometryEvidenceDiagnostics.insert(QStringLiteral("discrete_geometry_core_ratio"),
                                                   discrete_geometry_core.coreRatio);
            }
            for (const int sourceIndex : result.geometrySourceViewIndices)
            {
                geometrySourceIndices.append(sourceIndex);
            }
            AdaptiveGeometryEvidenceMaps adaptive_evidence_maps;
            adaptive_evidence_maps.supportWeight =
                result.adaptiveGeometrySupportWeight ? *result.adaptiveGeometrySupportWeight : cv::Mat();
            adaptive_evidence_maps.effectiveViewCount =
                result.adaptiveGeometryEffectiveViewCount ? *result.adaptiveGeometryEffectiveViewCount : cv::Mat();
            adaptive_evidence_maps.conflictRatio =
                result.adaptiveGeometryConflictRatio ? *result.adaptiveGeometryConflictRatio : cv::Mat();
            const QJsonObject adaptiveEvidenceDiagnostics =
                adaptiveGeometryEvidenceDiagnosticsToJson(*result.depthMap, adaptive_evidence_maps);
            for (auto it = adaptiveEvidenceDiagnostics.constBegin(); it != adaptiveEvidenceDiagnostics.constEnd(); ++it)
            {
                geometryEvidenceDiagnostics.insert(it.key(), it.value());
            }
            geometryEvidenceDiagnostics.insert(QStringLiteral("adaptive_reliability_model"),
                                               QStringLiteral("source_confidence_x_source_quality"));
            const cv::Mat emptyConfidence;
            const DepthMapQualityMetrics depthQualityMetrics =
                result.qualityMetrics.width > 0
                    ? result.qualityMetrics
                    : analyzeDepthMapQuality(*result.depthMap,
                                             confidenceMap ? *confidenceMap : emptyConfidence,
                                             sourceQualitySummary.sourceViewCount);
            QJsonObject depthQualityJson = depthMapQualityMetricsToJson(depthQualityMetrics);
            if (result.initialQualityAcceptanceAvailable)
            {
                depthQualityJson.insert(QStringLiteral("initial_acceptance"),
                                        QString::fromLatin1(depthFrameAcceptanceId(result.initialQualityAcceptance)));
            }
            for (auto it = depthCompletenessJson.constBegin(); it != depthCompletenessJson.constEnd(); ++it)
            {
                depthQualityJson.insert(it.key(), it.value());
            }
            const int requestedSourceViewCount =
                result.requestedSourceViewCount > 0
                    ? result.requestedSourceViewCount
                    : (hasSourceScoreCache ? _frameCaches[static_cast<size_t>(frameIndex)].requestedSourceViewCount
                                           : sourceQualitySummary.sourceViewCount);
            const int sourceViewShortfall =
                std::max(0, requestedSourceViewCount - sourceQualitySummary.sourceViewCount);
            const QString sourceViewShortfallReason =
                !result.sourceViewShortfallReason.empty()
                    ? QString::fromStdString(result.sourceViewShortfallReason)
                    : (hasSourceScoreCache
                           ? QString::fromStdString(
                                 _frameCaches[static_cast<size_t>(frameIndex)].sourceViewShortfallReason)
                           : QString());
            const QJsonObject sourceAngleDiagnostics =
                !result.sourceAngleDiagnostics.isEmpty()
                    ? result.sourceAngleDiagnostics
                    : (frameIndex >= 0 && frameIndex < static_cast<int>(_frameCaches.size())
                           ? _frameCaches[static_cast<size_t>(frameIndex)].sourceAngleDiagnostics
                           : QJsonObject{});
            depthQualityJson[QStringLiteral("requested_source_view_count")] = requestedSourceViewCount;
            depthQualityJson[QStringLiteral("source_view_shortfall")] = sourceViewShortfall;
            depthQualityJson[QStringLiteral("source_view_shortfall_reason")] = sourceViewShortfallReason;
            depthQualityJson[QStringLiteral("verified_source_view_count")] =
                sourceQualitySummary.verifiedSourceViewCount;
            depthQualityJson[QStringLiteral("backfill_source_view_count")] =
                sourceQualitySummary.backfillSourceViewCount;
            depthQualityJson[QStringLiteral("sequence_fallback_source_view_count")] =
                sourceQualitySummary.sequenceFallbackSourceViewCount;
            depthQualityJson[QStringLiteral("source_angle_diagnostics")] = sourceAngleDiagnostics;
            const QJsonObject qualityDecisionJson = depthFrameQualityDecisionToJson(result.qualityDecision);
            QJsonArray pyramidLevelsJson = depthPyramidLevelsToJson(result.pyramidLevels);
            for (qsizetype index = 0; index < pyramidLevelsJson.size(); ++index)
            {
                QJsonObject level_object = pyramidLevelsJson.at(index).toObject();
                const int level = level_object.value(QStringLiteral("level")).toInt();
                if (level == 1)
                {
                    level_object.insert(QStringLiteral("preview_path"), QString::fromStdString(pngPath));
                    level_object.insert(QStringLiteral("raw_depth_path"),
                                        saveRawDepth ? QString::fromStdString(rawDepthPath) : QString());
                    level_object.insert(QStringLiteral("raw_confidence_path"),
                                        confidenceSaved ? QString::fromStdString(rawConfidencePath) : QString());
                    level_object.insert(QStringLiteral("raw_geometry_support_path"),
                                        geometrySupportSaved ? QString::fromStdString(rawGeometrySupportPath)
                                                             : QString());
                    level_object.insert(QStringLiteral("raw_geometry_source_mask_path"),
                                        geometrySourceMaskSaved ? QString::fromStdString(rawGeometrySourceMaskPath)
                                                                : QString());
                    level_object.insert(QStringLiteral("raw_inverse_depth_mean_path"),
                                        inverseDepthMeanSaved ? QString::fromStdString(rawInverseDepthMeanPath)
                                                              : QString());
                    level_object.insert(QStringLiteral("raw_inverse_depth_spread_path"),
                                        inverseDepthSpreadSaved ? QString::fromStdString(rawInverseDepthSpreadPath)
                                                                : QString());
                    level_object.insert(QStringLiteral("raw_adaptive_geometry_support_weight_path"),
                                        adaptiveGeometrySupportWeightSaved
                                            ? QString::fromStdString(rawAdaptiveGeometrySupportWeightPath)
                                            : QString());
                    level_object.insert(QStringLiteral("raw_adaptive_geometry_effective_view_count_path"),
                                        adaptiveGeometryEffectiveViewCountSaved
                                            ? QString::fromStdString(rawAdaptiveGeometryEffectiveViewCountPath)
                                            : QString());
                    level_object.insert(QStringLiteral("raw_adaptive_geometry_conflict_ratio_path"),
                                        adaptiveGeometryConflictRatioSaved
                                            ? QString::fromStdString(rawAdaptiveGeometryConflictRatioPath)
                                            : QString());
                    level_object.insert(QStringLiteral("cross_view_repaired_mask_path"),
                                        crossViewRepairedMaskSaved ? QString::fromStdString(crossViewRepairedMaskPath)
                                                                   : QString());
                    level_object.insert(QStringLiteral("valid_mask_path"),
                                        maskSaved ? QString::fromStdString(validMaskPath) : QString());
                    level_object.insert(QStringLiteral("missing_reason_path"),
                                        missingReasonSaved ? QString::fromStdString(missingReasonPath) : QString());
                    level_object.insert(QStringLiteral("targeted_gap_recovered_mask_path"),
                                        targetedGapRecoveredMaskSaved
                                            ? QString::fromStdString(targetedGapRecoveredMaskPath)
                                            : QString());
                    level_object.insert(
                        QStringLiteral("residual_reestimated_mask_path"),
                        residualReestimatedMaskSaved ? QString::fromStdString(residualReestimatedMaskPath) : QString());
                    level_object.insert(QStringLiteral("depth_provenance_path"),
                                        depthProvenanceSaved ? QString::fromStdString(depthProvenancePath) : QString());
                    level_object.insert(QStringLiteral("missing_reason_preview_path"),
                                        missingReasonPreviewSaved ? QString::fromStdString(missingReasonPreviewPath)
                                                                  : QString());
                }
                const auto path_it = pyramid_level_paths.find(level);
                if (path_it != pyramid_level_paths.end())
                {
                    for (auto value_it = path_it->second.constBegin(); value_it != path_it->second.constEnd();
                         ++value_it)
                    {
                        level_object.insert(value_it.key(), value_it.value());
                    }
                }
                pyramidLevelsJson.replace(index, level_object);
            }
            const QString sceneProfile = sceneProfileId(_effectiveSceneProfile);
            const QString filterMode = depthFilterModeId(_effectiveDepthFilterMode);
            const QString acceptance = QString::fromLatin1(depthFrameAcceptanceId(result.qualityDecision.acceptance));
            const QJsonObject depthPostprocessJson = depthPostProcessStatsToJson(result.depthPostprocess);

            QJsonObject artifact;
            artifact[QStringLiteral("ref_index")] = frameIndex;
            artifact[QStringLiteral("depth_png")] = QString::fromStdString(pngPath);
            artifact[QStringLiteral("raw_depth_path")] =
                saveRawDepth ? QString::fromStdString(rawDepthPath) : QString();
            artifact[QStringLiteral("raw_confidence_path")] =
                confidenceSaved ? QString::fromStdString(rawConfidencePath) : QString();
            artifact[QStringLiteral("raw_photometric_source_mask_path")] =
                photometricSourceMaskSaved ? QString::fromStdString(rawPhotometricSourceMaskPath) : QString();
            artifact[QStringLiteral("raw_geometry_support_path")] =
                geometrySupportSaved ? QString::fromStdString(rawGeometrySupportPath) : QString();
            artifact[QStringLiteral("raw_geometry_source_mask_path")] =
                geometrySourceMaskSaved ? QString::fromStdString(rawGeometrySourceMaskPath) : QString();
            artifact[QStringLiteral("raw_inverse_depth_mean_path")] =
                inverseDepthMeanSaved ? QString::fromStdString(rawInverseDepthMeanPath) : QString();
            artifact[QStringLiteral("raw_inverse_depth_spread_path")] =
                inverseDepthSpreadSaved ? QString::fromStdString(rawInverseDepthSpreadPath) : QString();
            artifact[QStringLiteral("raw_adaptive_geometry_support_weight_path")] =
                adaptiveGeometrySupportWeightSaved ? QString::fromStdString(rawAdaptiveGeometrySupportWeightPath)
                                                   : QString();
            artifact[QStringLiteral("raw_adaptive_geometry_effective_view_count_path")] =
                adaptiveGeometryEffectiveViewCountSaved
                    ? QString::fromStdString(rawAdaptiveGeometryEffectiveViewCountPath)
                    : QString();
            artifact[QStringLiteral("raw_adaptive_geometry_conflict_ratio_path")] =
                adaptiveGeometryConflictRatioSaved ? QString::fromStdString(rawAdaptiveGeometryConflictRatioPath)
                                                   : QString();
            artifact[QStringLiteral("cross_view_repaired_mask_path")] =
                crossViewRepairedMaskSaved ? QString::fromStdString(crossViewRepairedMaskPath) : QString();
            artifact[QStringLiteral("targeted_gap_recovered_mask_path")] =
                targetedGapRecoveredMaskSaved ? QString::fromStdString(targetedGapRecoveredMaskPath) : QString();
            artifact[QStringLiteral("residual_reestimated_mask_path")] =
                residualReestimatedMaskSaved ? QString::fromStdString(residualReestimatedMaskPath) : QString();
            artifact[QStringLiteral("depth_provenance_path")] =
                depthProvenanceSaved ? QString::fromStdString(depthProvenancePath) : QString();
            artifact[QStringLiteral("valid_mask_path")] = maskSaved ? QString::fromStdString(validMaskPath) : QString();
            artifact[QStringLiteral("support_mask_path")] =
                supportMaskSaved ? QString::fromStdString(supportMaskPath) : QString();
            artifact[QStringLiteral("missing_reason_path")] =
                missingReasonSaved ? QString::fromStdString(missingReasonPath) : QString();
            artifact[QStringLiteral("missing_reason_preview_path")] =
                missingReasonPreviewSaved ? QString::fromStdString(missingReasonPreviewPath) : QString();
            artifact[QStringLiteral("ref_image")] = QString::fromStdString(_views[frameIndex].imagePath);
            artifact[QStringLiteral("prepared_image")] = QString::fromStdString(prepared_raster.imagePath);
            artifact[QStringLiteral("prepared_valid_mask_path")] =
                QString::fromStdString(prepared_raster.validMaskPath);
            artifact[QStringLiteral("prepared_camera_model")] = cameraModelToJson(prepared_raster.camera);
            artifact[QStringLiteral("source_images")] = sourceImages;
            artifact[QStringLiteral("source_indices")] = sourceIndices;
            artifact[QStringLiteral("geometry_source_indices")] = geometrySourceIndices;
            artifact[QStringLiteral("source_plan")] = sourcePlan;
            artifact[QStringLiteral("source_angle_diagnostics")] = sourceAngleDiagnostics;
            artifact[QStringLiteral("quality_profile")] = QString::fromStdString(_config.qualityProfile);
            artifact[QStringLiteral("configured_source_view_count")] = _configuredSourceViewCount;
            artifact[QStringLiteral("source_view_count")] = sourceQualitySummary.sourceViewCount;
            artifact[QStringLiteral("requested_source_view_count")] = requestedSourceViewCount;
            artifact[QStringLiteral("source_view_shortfall")] = sourceViewShortfall;
            artifact[QStringLiteral("source_view_shortfall_reason")] = sourceViewShortfallReason;
            artifact[QStringLiteral("consistency_publication_expected")] =
                detail::expectsConsistencyPublication(result, static_cast<int>(_views.size()));
            artifact[QStringLiteral("geometric_guidance_pass_expected")] = result.geometricGuidancePassExpected;
            artifact[QStringLiteral("geometric_guidance_pass_applied")] = result.geometricGuidancePassApplied;
            artifact[QStringLiteral("verified_source_view_count")] = sourceQualitySummary.verifiedSourceViewCount;
            artifact[QStringLiteral("backfill_source_view_count")] = sourceQualitySummary.backfillSourceViewCount;
            artifact[QStringLiteral("source_quality_mean")] = sourceQualitySummary.meanQuality;
            artifact[QStringLiteral("source_quality_min")] = sourceQualitySummary.minQuality;
            artifact[QStringLiteral("depth_confidence_mean")] = depthConfidenceSummary.meanConfidence;
            artifact[QStringLiteral("effective_patch_match_confidence_threshold")] =
                result.effectivePatchMatchConfidenceThreshold;
            artifact[QStringLiteral("valid_pixel_count")] = depthConfidenceSummary.validPixelCount;
            artifact[QStringLiteral("valid_coverage")] = static_cast<double>(result.qualityMetrics.validCoverage);
            artifact[QStringLiteral("depth_quality")] = depthQualityJson;
            artifact[QStringLiteral("depth_completeness")] = depthCompletenessJson;
            artifact[QStringLiteral("missing_reason_summary")] = missingReasonSummaryJson;
            artifact[QStringLiteral("cross_view_repair_diagnostics")] = result.crossViewRepairDiagnostics;
            artifact[QStringLiteral("targeted_gap_recovery_diagnostics")] = result.targetedGapRecoveryDiagnostics;
            artifact[QStringLiteral("residual_reestimation_diagnostics")] = result.residualReestimationDiagnostics;
            artifact[QStringLiteral("evidence_confidence_diagnostics")] = result.evidenceConfidenceDiagnostics;
            artifact[QStringLiteral("learned_candidate_diagnostics")] = result.learnedCandidateDiagnostics;
            artifact[QStringLiteral("depth_provenance_summary")] = depthProvenanceSummaryJson;
            artifact[QStringLiteral("geometry_evidence_diagnostics")] = geometryEvidenceDiagnostics;
            if (!result.poseRefinementDiagnostics.isEmpty())
            {
                artifact[QStringLiteral("pose_refinement_diagnostics")] = result.poseRefinementDiagnostics;
            }
            if (result.derivedCameraModel.isValid())
            {
                artifact[QStringLiteral("derived_camera_model")] = cameraModelToJson(result.derivedCameraModel);
            }
            if (depthCompleteness.finalMetrics.validInputs)
            {
                artifact[QStringLiteral("mask_pixel_count")] = depthCompleteness.finalMetrics.maskPixelCount;
                artifact[QStringLiteral("valid_within_mask_count")] =
                    depthCompleteness.finalMetrics.validWithinMaskCount;
                artifact[QStringLiteral("valid_within_mask_ratio")] =
                    depthCompleteness.finalMetrics.validWithinMaskRatio;
            }
            artifact[QStringLiteral("quality_decision")] = qualityDecisionJson;
            artifact[QStringLiteral("pyramid_levels")] = pyramidLevelsJson;
            artifact[QStringLiteral("mask_source")] = QString::fromStdString(result.maskSource);
            artifact[QStringLiteral("mask_coverage")] = result.maskCoverage;
            artifact[QStringLiteral("selected_level")] = result.selectedLevel;
            artifact[QStringLiteral("fallback_reason")] = QString::fromStdString(result.fallbackReason);
            artifact[QStringLiteral("pyramid_requested_level_count")] = result.pyramidRequestedLevelCount;
            artifact[QStringLiteral("pyramid_active_level_count")] = result.pyramidActiveLevelCount;
            artifact[QStringLiteral("pyramid_minimum_short_side")] = result.pyramidMinimumShortSide;
            artifact[QStringLiteral("pyramid_degraded_reason")] = QString::fromStdString(result.pyramidDegradedReason);
            artifact[QStringLiteral("scene_profile")] = sceneProfile;
            artifact[QStringLiteral("filter_mode")] = filterMode;
            artifact[QStringLiteral("acceptance")] = acceptance;
            artifact[QStringLiteral("fusion_eligible")] = result.eligibleForFusion();
            artifact[QStringLiteral("depth_postprocess")] = depthPostprocessJson;
            artifact[QStringLiteral("camera_model")] = cameraModelToJson(
                result.cameraModel.isValid() ? result.cameraModel : mvsPinholeCamera(_views[frameIndex].camera));
            artifact[QStringLiteral("status")] =
                final_artifacts ? QStringLiteral("completed") : QStringLiteral("running");
            artifact[QStringLiteral("stage")] = stageLabel;
            artifact[QStringLiteral("device")] =
                QString::fromStdString(result.device.empty() ? "unknown" : result.device);
            artifact[QStringLiteral("elapsed_ms")] = result.elapsedMs;
            artifact[QStringLiteral("effective_native_final_depth_grid")] = result.effectiveNativeFinalDepthGrid;
            artifact[QStringLiteral("pixel_domain_diagnostics")] = result.pixelDomainDiagnostics;
            artifact[QStringLiteral("grid_width")] = result.depthMap->cols;
            artifact[QStringLiteral("grid_height")] = result.depthMap->rows;
            artifact[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
            artifact[QStringLiteral("config_hash")] = _depthConfigHash;
            artifact[QStringLiteral("algorithm_revision")] = kMvsDepthAlgorithmRevision;
            artifact[QStringLiteral("manifest_path")] = _workspaceManifestPath;

            MvsDepthFrameRecord record;
            record.refIndex = frameIndex;
            record.refImage = QString::fromStdString(_views[frameIndex].imagePath);
            record.preparedImage = QString::fromStdString(prepared_raster.imagePath);
            record.preparedValidMaskPath = QString::fromStdString(prepared_raster.validMaskPath);
            record.preparedCameraModel = cameraModelToJson(prepared_raster.camera);
            record.sourceImages = sourceImageList;
            for (const QJsonValue& source_value : sourceIndices)
            {
                record.sourceIndices.push_back(source_value.toInt(-1));
            }
            for (const QJsonValue& source_value : geometrySourceIndices)
            {
                record.geometrySourceIndices.push_back(source_value.toInt(-1));
            }
            record.sourcePlan = sourcePlan;
            record.qualityProfile = QString::fromStdString(_config.qualityProfile);
            record.configuredSourceViewCount = _configuredSourceViewCount;
            record.sourceViewCount = sourceQualitySummary.sourceViewCount;
            record.requestedSourceViewCount = requestedSourceViewCount;
            record.sourceViewShortfall = sourceViewShortfall;
            record.sourceViewShortfallReason = sourceViewShortfallReason;
            record.consistencyPublicationExpected =
                detail::expectsConsistencyPublication(result, static_cast<int>(_views.size()));
            record.geometricGuidancePassExpected = result.geometricGuidancePassExpected;
            record.geometricGuidancePassApplied = result.geometricGuidancePassApplied;
            record.meanSourceQualityScore = sourceQualitySummary.meanQuality;
            record.minSourceQualityScore = sourceQualitySummary.minQuality;
            record.meanDepthConfidence = depthConfidenceSummary.meanConfidence;
            record.effectivePatchMatchConfidenceThreshold = result.effectivePatchMatchConfidenceThreshold;
            record.validPixelCount = depthConfidenceSummary.validPixelCount;
            record.validCoverage = static_cast<double>(result.qualityMetrics.validCoverage);
            record.depthQuality = depthQualityJson;
            record.depthCompleteness = depthCompletenessJson;
            record.missingReasonSummary = missingReasonSummaryJson;
            record.crossViewRepairDiagnostics = result.crossViewRepairDiagnostics;
            record.targetedGapRecoveryDiagnostics = result.targetedGapRecoveryDiagnostics;
            record.residualReestimationDiagnostics = result.residualReestimationDiagnostics;
            record.learnedCandidateDiagnostics = result.learnedCandidateDiagnostics;
            record.depthProvenanceSummary = depthProvenanceSummaryJson;
            record.geometryEvidenceDiagnostics = geometryEvidenceDiagnostics;
            record.poseRefinementDiagnostics = result.poseRefinementDiagnostics;
            if (result.derivedCameraModel.isValid())
            {
                record.derivedCameraModel = cameraModelToJson(result.derivedCameraModel);
            }
            record.qualityDecision = qualityDecisionJson;
            record.pyramidLevels = pyramidLevelsJson;
            record.maskSource = QString::fromStdString(result.maskSource);
            record.maskCoverage = result.maskCoverage;
            record.selectedLevel = result.selectedLevel;
            record.fallbackReason = QString::fromStdString(result.fallbackReason);
            record.pyramidRequestedLevelCount = result.pyramidRequestedLevelCount;
            record.pyramidActiveLevelCount = result.pyramidActiveLevelCount;
            record.pyramidMinimumShortSide = result.pyramidMinimumShortSide;
            record.pyramidDegradedReason = QString::fromStdString(result.pyramidDegradedReason);
            record.sceneProfile = sceneProfile;
            record.filterMode = filterMode;
            record.acceptance = acceptance;
            record.fusionEligible = result.eligibleForFusion();
            record.fusionEligibilityKnown = true;
            record.depthPostprocess = depthPostprocessJson;
            record.cameraModel = cameraModelToJson(
                result.cameraModel.isValid() ? result.cameraModel : mvsPinholeCamera(_views[frameIndex].camera));
            record.status = final_artifacts ? QStringLiteral("completed") : QStringLiteral("running");
            record.device = QString::fromStdString(result.device.empty() ? "unknown" : result.device);
            record.depthPng = QString::fromStdString(pngPath);
            record.rawDepthPath = saveRawDepth ? QString::fromStdString(rawDepthPath) : QString();
            record.rawConfidencePath = confidenceSaved ? QString::fromStdString(rawConfidencePath) : QString();
            record.rawPhotometricSourceMaskPath =
                photometricSourceMaskSaved ? QString::fromStdString(rawPhotometricSourceMaskPath) : QString();
            record.rawGeometrySupportPath =
                geometrySupportSaved ? QString::fromStdString(rawGeometrySupportPath) : QString();
            record.rawGeometrySourceMaskPath =
                geometrySourceMaskSaved ? QString::fromStdString(rawGeometrySourceMaskPath) : QString();
            record.rawInverseDepthMeanPath =
                inverseDepthMeanSaved ? QString::fromStdString(rawInverseDepthMeanPath) : QString();
            record.rawInverseDepthSpreadPath =
                inverseDepthSpreadSaved ? QString::fromStdString(rawInverseDepthSpreadPath) : QString();
            record.rawAdaptiveGeometrySupportWeightPath =
                adaptiveGeometrySupportWeightSaved ? QString::fromStdString(rawAdaptiveGeometrySupportWeightPath)
                                                   : QString();
            record.rawAdaptiveGeometryEffectiveViewCountPath =
                adaptiveGeometryEffectiveViewCountSaved
                    ? QString::fromStdString(rawAdaptiveGeometryEffectiveViewCountPath)
                    : QString();
            record.rawAdaptiveGeometryConflictRatioPath =
                adaptiveGeometryConflictRatioSaved ? QString::fromStdString(rawAdaptiveGeometryConflictRatioPath)
                                                   : QString();
            record.crossViewRepairedMaskPath =
                crossViewRepairedMaskSaved ? QString::fromStdString(crossViewRepairedMaskPath) : QString();
            record.targetedGapRecoveredMaskPath =
                targetedGapRecoveredMaskSaved ? QString::fromStdString(targetedGapRecoveredMaskPath) : QString();
            record.residualReestimatedMaskPath =
                residualReestimatedMaskSaved ? QString::fromStdString(residualReestimatedMaskPath) : QString();
            record.depthProvenancePath = depthProvenanceSaved ? QString::fromStdString(depthProvenancePath) : QString();
            record.validMaskPath = maskSaved ? QString::fromStdString(validMaskPath) : QString();
            record.supportMaskPath = supportMaskSaved ? QString::fromStdString(supportMaskPath) : QString();
            record.missingReasonPath = missingReasonSaved ? QString::fromStdString(missingReasonPath) : QString();
            record.missingReasonPreviewPath =
                missingReasonPreviewSaved ? QString::fromStdString(missingReasonPreviewPath) : QString();
            record.effectiveNativeFinalDepthGrid = result.effectiveNativeFinalDepthGrid;
            record.pixelDomainDiagnostics = result.pixelDomainDiagnostics;
            record.gridWidth = result.depthMap->cols;
            record.gridHeight = result.depthMap->rows;
            record.elapsedMs = static_cast<qint64>(std::llround(result.elapsedMs));
            record.configHash = _depthConfigHash;
            record.algorithmRevision = kMvsDepthAlgorithmRevision;

            {
                std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
                if (final_artifacts)
                {
                    _workspaceManifest.markCompleted(record);
                }
                else
                {
                    _workspaceManifest.upsertFrame(record);
                }
                QString manifestError;
                if (!persistWorkspaceManifest(&manifestError))
                {
                    LOG_WARN(QStringLiteral("[MVS] 写入%1 manifest 失败: %2")
                                 .arg(final_artifacts ? QStringLiteral("完成") : QStringLiteral("checkpoint"),
                                      manifestError));
                    errorOccurred(manifestError);
                    return false;
                }
            }
            if (final_artifacts)
            {
                depthMapSaved(QString::fromStdString(pngPath),
                              result.depthMap->cols,
                              result.depthMap->rows,
                              QString::fromStdString(_views[frameIndex].imagePath));
                depthMapArtifactSaved(artifact);
            }
        }

        return previewSaved && rawSaved;
    }

    void MvsPipelineService::captureStageSnapshot(int frameIndex,
                                                  MvsStageSnapshotStage stage,
                                                  const QString& boundary,
                                                  const DepthFrameResult& result,
                                                  const cv::Mat& depth,
                                                  const cv::Mat& confidence,
                                                  const cv::Mat& validMask)
    {
        if (!_stageSnapshotRecorder || !_stageSnapshotRecorder->selected(frameIndex))
        {
            return;
        }
        _stageSnapshotRecorder->capture(frameIndex, stage, boundary, result, depth, confidence, validMask);
    }
} // namespace xjw::mvs
