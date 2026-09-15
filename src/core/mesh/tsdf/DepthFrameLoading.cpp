#include "DepthTsdfInternals.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;

    DepthTsdfFrameLoadResult DepthTsdfSurfaceBuilder::loadFrames(const QVector<DepthFrameArtifact>& artifacts,
                                                                 int requestedWorkerCount)
    {
        const auto started_at = std::chrono::steady_clock::now();
        DepthTsdfFrameLoadResult result;
        if (artifacts.size() > std::numeric_limits<int>::max())
        {
            result.errorMessage = QStringLiteral("too many TSDF frame artifacts to load");
            result.elapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at)
                    .count();
            return result;
        }
        result.discoveredArtifactCount = static_cast<int>(artifacts.size());
        QString canonical_scene_profile;
        for (const DepthFrameArtifact& artifact : artifacts)
        {
            if (artifact.role == xjw::mvs::DepthFrameRole::Excluded)
            {
                continue;
            }
            if (!xjw::mvs::extendCanonicalDepthSceneProfileBatch(artifact.sceneProfile, &canonical_scene_profile))
            {
                result.errorMessage =
                    frameArtifactError(artifact,
                                       QStringLiteral("usable depth-frame scene_profile is unknown or "
                                                      "inconsistent with the batch: %1")
                                           .arg(artifact.sceneProfile));
                result.elapsedMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at)
                        .count();
                return result;
            }
        }
        const QVector<DepthFrameArtifact> selected_artifacts = selectMemoryBoundedDepthFrameArtifacts(artifacts);
        const int artifact_count = static_cast<int>(selected_artifacts.size());

        int maximum_workers = 1;
#ifdef MESHING_OPENMP
        maximum_workers = std::max(1, omp_get_max_threads());
#endif
        const int desired_workers = requestedWorkerCount > 0 ? requestedWorkerCount : std::max(1, maximum_workers - 2);
        // Loading one frame temporarily owns several full-resolution evidence
        // matrices. A bounded queue keeps NVMe throughput high without multiplying
        // the peak memory footprint by every logical CPU thread.
        result.effectiveWorkerCount =
            std::clamp(desired_workers, 1, std::max(1, std::min({8, maximum_workers, artifact_count})));

        std::vector<DepthTsdfFrameLoadResult> partial_results(static_cast<std::size_t>(artifact_count));
        std::vector<std::exception_ptr> loading_errors(static_cast<std::size_t>(artifact_count));
        std::atomic<int> first_failed_index{artifact_count};
        const auto record_failure = [&first_failed_index](int artifact_index) noexcept
        {
            int previous = first_failed_index.load(std::memory_order_relaxed);
            while (artifact_index < previous &&
                   !first_failed_index.compare_exchange_weak(
                       previous, artifact_index, std::memory_order_relaxed, std::memory_order_relaxed))
            {
            }
        };
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(dynamic, 1)                                                                          \
    num_threads(result.effectiveWorkerCount) if (result.effectiveWorkerCount > 1)
#endif
        for (int artifact_index = 0; artifact_index < artifact_count; ++artifact_index)
        {
            if (artifact_index > first_failed_index.load(std::memory_order_relaxed))
            {
                continue;
            }
            const std::size_t result_index = static_cast<std::size_t>(artifact_index);
            try
            {
                partial_results[result_index] =
                    loadFramesSequential(QVector<DepthFrameArtifact>{selected_artifacts[artifact_index]}, 0);
                if (!partial_results[result_index].ok)
                {
                    record_failure(artifact_index);
                }
            }
            catch (...)
            {
                loading_errors[result_index] = std::current_exception();
                record_failure(artifact_index);
            }
        }

        for (int artifact_index = 0; artifact_index < artifact_count; ++artifact_index)
        {
            const std::size_t result_index = static_cast<std::size_t>(artifact_index);
            DepthTsdfFrameLoadResult& partial = partial_results[result_index];
            if (loading_errors[result_index])
            {
                try
                {
                    std::rethrow_exception(loading_errors[result_index]);
                }
                catch (const std::exception& error)
                {
                    partial.errorMessage = frameArtifactError(
                        selected_artifacts[artifact_index],
                        QStringLiteral("unexpected loading exception: %1").arg(QString::fromUtf8(error.what())));
                }
                catch (...)
                {
                    partial.errorMessage =
                        frameArtifactError(selected_artifacts[artifact_index],
                                           QStringLiteral("unexpected non-standard loading exception"));
                }
            }
            if (!partial.ok)
            {
                result.errorMessage = partial.errorMessage;
                result.elapsedMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at)
                        .count();
                return result;
            }
            for (DepthTsdfFrame& frame : partial.frames)
            {
                if (frame.auxiliarySurfaceOnly)
                {
                    ++result.auxiliaryFrameCount;
                }
                else
                {
                    ++result.primaryFrameCount;
                }
                result.frames.push_back(std::move(frame));
            }
        }
        if (result.frames.size() < 3)
        {
            result.errorMessage =
                QStringLiteral("TSDF requires at least 3 usable primary or auxiliary depth frames; loaded=%1")
                    .arg(result.frames.size());
        }
        else
        {
            result.ok = true;
        }
        result.elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at)
                .count();
        return result;
    }

    DepthTsdfFrameLoadResult DepthTsdfSurfaceBuilder::loadFramesSequential(const QVector<DepthFrameArtifact>& artifacts,
                                                                           int minimumFrameCount)
    {
        DepthTsdfFrameLoadResult result;
        for (const DepthFrameArtifact& artifact : artifacts)
        {
            const bool validation_only = loadsAsAuxiliarySurfaceOnly(artifact);
            if (!canLoadDepthFrameArtifact(artifact))
            {
                continue;
            }
            if (!xjw::mvs::isKnownDepthSceneProfile(artifact.sceneProfile))
            {
                result.errorMessage = frameArtifactError(
                    artifact, QStringLiteral("scene_profile is missing or unknown: %1").arg(artifact.sceneProfile));
                return result;
            }
            if (!artifact.hasCameraModel || !artifact.cameraModel.isValid())
            {
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("camera is invalid"));
                return result;
            }
            if (artifact.depthPath.isEmpty())
            {
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("raw depth path is empty"));
                return result;
            }
            const bool uses_recovered_reference_filter =
                artifact.depthProducer.compare(QStringLiteral("recovered_scene_d4"), Qt::CaseInsensitive) == 0;
            // recovered_scene_d4 has already completed the reference model's
            // PatchMatch filter and three-level voting chain.  Its contract does
            // not contain PlaScan's post-estimation geometry/support products, so
            // requiring those files would incorrectly reintroduce the removed
            // PlaScan quality gate at consumption time.
            const bool requires_current_orbital_evidence = !uses_recovered_reference_filter &&
                                                           artifact.algorithmRevision >= 11 &&
                                                           xjw::mvs::isOrbitalDepthSceneProfile(artifact.sceneProfile);
            const bool requires_adaptive_orbital_evidence =
                requires_current_orbital_evidence && artifact.algorithmRevision >= 13;
            const bool requires_conflict_ratio = requires_adaptive_orbital_evidence && artifact.algorithmRevision >= 14;
            const bool requires_depth_provenance =
                requires_current_orbital_evidence && artifact.algorithmRevision >= 17;
            const bool uses_exact_geometry_source_ordinals =
                artifact.algorithmRevision >= xjw::mvs::kMvsGeometrySourceOrdinalRevision;
            const bool requires_legacy_geometry_source_mask =
                requires_current_orbital_evidence && !uses_exact_geometry_source_ordinals;
            if (requires_current_orbital_evidence &&
                (artifact.geometrySupportPath.isEmpty() ||
                 (requires_legacy_geometry_source_mask && artifact.geometrySourceMaskPath.isEmpty()) ||
                 artifact.inverseDepthMeanPath.isEmpty() || artifact.inverseDepthSpreadPath.isEmpty() ||
                 artifact.crossViewRepairedMaskPath.isEmpty() ||
                 (requires_depth_provenance && artifact.depthProvenancePath.isEmpty())))
            {
                result.errorMessage =
                    frameArtifactError(artifact,
                                       QStringLiteral("current orbital depth evidence is incomplete; "
                                                      "regenerate depth maps so geometry support, inverse-depth "
                                                      "statistics, repair provenance, and depth provenance are "
                                                      "present; an exact source mask and ordinal table may only "
                                                      "be omitted together when no source evidence exists"));
                return result;
            }
            if (requires_adaptive_orbital_evidence &&
                (artifact.adaptiveGeometrySupportWeightPath.isEmpty() ||
                 artifact.adaptiveGeometryEffectiveViewCountPath.isEmpty() ||
                 (requires_conflict_ratio && artifact.adaptiveGeometryConflictRatioPath.isEmpty())))
            {
                result.errorMessage = frameArtifactError(
                    artifact,
                    requires_conflict_ratio
                        ? QStringLiteral("current orbital adaptive geometry evidence is incomplete; "
                                         "regenerate depth maps so support weight, effective view count, "
                                         "and conflict ratio are all present")
                        : QStringLiteral("current orbital adaptive geometry evidence is incomplete; "
                                         "regenerate depth maps so support weight and effective view count "
                                         "are both present"));
                return result;
            }

            DepthTsdfFrame frame;
            frame.refIndex = artifact.refIndex;
            frame.refImage = artifact.refImage;
            frame.sceneProfile = artifact.sceneProfile;
            frame.algorithmRevision = artifact.algorithmRevision;
            frame.camera = artifact.cameraModel;
            const bool has_geometry_source_mask = !artifact.geometrySourceMaskPath.isEmpty();
            const bool has_geometry_source_ordinals = !artifact.geometrySourceIndices.isEmpty();
            if (uses_exact_geometry_source_ordinals &&
                (has_geometry_source_mask != has_geometry_source_ordinals ||
                 artifact.geometrySourceIndices.size() > static_cast<qsizetype>(kDepthGeometryLocalSourceSlotCount)))
            {
                result.errorMessage =
                    frameArtifactError(artifact,
                                       QStringLiteral("geometry source mask and ordinal table must either both "
                                                      "be present or both be omitted, and the table cannot "
                                                      "exceed 16 entries; regenerate revision %1 depth maps")
                                           .arg(xjw::mvs::kMvsGeometrySourceOrdinalRevision));
                return result;
            }
            if (uses_exact_geometry_source_ordinals)
            {
                QSet<int> unique_source_indices;
                for (const int source_index : artifact.geometrySourceIndices)
                {
                    if (source_index < 0 || source_index == artifact.refIndex ||
                        unique_source_indices.contains(source_index))
                    {
                        result.errorMessage =
                            frameArtifactError(artifact,
                                               QStringLiteral("geometry source ordinal table contains a "
                                                              "negative, reference, or duplicate source index"));
                        return result;
                    }
                    unique_source_indices.insert(source_index);
                }
            }
            frame.geometrySourceIndices =
                uses_exact_geometry_source_ordinals
                    ? artifact.geometrySourceIndices
                    : (artifact.geometrySourceIndices.isEmpty() ? artifact.sourceIndices
                                                                : artifact.geometrySourceIndices);
            QString reason;
            if (!loadFloatMatrix(artifact.depthPath, &frame.depth, &reason))
            {
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("depth: %1").arg(reason));
                return result;
            }

            if (artifact.confidencePath.isEmpty())
            {
                frame.confidence = cv::Mat(frame.depth.size(), CV_32FC1, cv::Scalar(1.0f));
            }
            else if (!loadFloatMatrix(artifact.confidencePath, &frame.confidence, &reason) ||
                     frame.confidence.size() != frame.depth.size())
            {
                if (reason.isEmpty())
                {
                    reason = QStringLiteral("confidence dimensions do not match depth");
                }
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("confidence: %1").arg(reason));
                return result;
            }

            if (artifact.geometrySupportPath.isEmpty())
            {
                frame.geometrySupportCount = cv::Mat(frame.depth.size(), CV_16UC1, cv::Scalar(0));
            }
            else if (!loadUnsignedShortMatrix(artifact.geometrySupportPath, &frame.geometrySupportCount, &reason))
            {
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("geometry support: %1").arg(reason));
                return result;
            }
            else if (frame.geometrySupportCount.size() != frame.depth.size())
            {
                if (!artifact.pyramidFallback)
                {
                    result.errorMessage =
                        frameArtifactError(artifact, QStringLiteral("geometry support dimensions do not match depth"));
                    return result;
                }
                cv::resize(frame.geometrySupportCount,
                           frame.geometrySupportCount,
                           frame.depth.size(),
                           0.0,
                           0.0,
                           cv::INTER_NEAREST);
            }

            if (artifact.geometrySourceMaskPath.isEmpty())
            {
                frame.geometrySourceMask = cv::Mat(frame.depth.size(), CV_16UC1, cv::Scalar(0));
            }
            else if (!loadUnsignedShortMatrix(artifact.geometrySourceMaskPath, &frame.geometrySourceMask, &reason))
            {
                result.errorMessage =
                    frameArtifactError(artifact, QStringLiteral("geometry source mask: %1").arg(reason));
                return result;
            }
            else if (frame.geometrySourceMask.size() != frame.depth.size())
            {
                if (!artifact.pyramidFallback)
                {
                    result.errorMessage = frameArtifactError(
                        artifact, QStringLiteral("geometry source mask dimensions do not match depth"));
                    return result;
                }
                cv::resize(frame.geometrySourceMask,
                           frame.geometrySourceMask,
                           frame.depth.size(),
                           0.0,
                           0.0,
                           cv::INTER_NEAREST);
            }
            if (!artifact.geometrySourceMaskPath.isEmpty() &&
                artifact.algorithmRevision >= xjw::mvs::kMvsGeometrySourceOrdinalRevision &&
                frame.geometrySourceIndices.size() < static_cast<qsizetype>(kDepthGeometryLocalSourceSlotCount))
            {
                const int source_count = frame.geometrySourceIndices.size();
                const std::uint16_t allowed_bits =
                    source_count == 0 ? 0 : static_cast<std::uint16_t>((std::uint32_t{1} << source_count) - 1U);
                double maximum_mask_value = 0.0;
                cv::minMaxLoc(frame.geometrySourceMask, nullptr, &maximum_mask_value);
                if (maximum_mask_value > static_cast<double>(allowed_bits))
                {
                    result.errorMessage =
                        frameArtifactError(artifact,
                                           QStringLiteral("geometry source mask uses a bit outside its %1-entry "
                                                          "ordinal table; regenerate the depth map")
                                               .arg(source_count));
                    return result;
                }
            }

            auto load_optional_float_evidence = [&](const QString& path, cv::Mat* destination, const QString& label)
            {
                if (path.isEmpty())
                {
                    *destination = cv::Mat(frame.depth.size(), CV_32FC1, cv::Scalar(0.0f));
                    return true;
                }
                if (!loadFloatMatrix(path, destination, &reason))
                {
                    result.errorMessage = frameArtifactError(artifact, QStringLiteral("%1: %2").arg(label, reason));
                    return false;
                }
                if (destination->size() != frame.depth.size())
                {
                    if (!artifact.pyramidFallback)
                    {
                        result.errorMessage = frameArtifactError(
                            artifact, QStringLiteral("%1: dimensions do not match depth").arg(label));
                        return false;
                    }
                    cv::resize(*destination, *destination, frame.depth.size(), 0.0, 0.0, cv::INTER_AREA);
                }
                return true;
            };
            if (!load_optional_float_evidence(
                    artifact.inverseDepthMeanPath, &frame.inverseDepthMean, QStringLiteral("inverse depth mean")) ||
                !load_optional_float_evidence(artifact.inverseDepthSpreadPath,
                                              &frame.inverseDepthRelativeSpread,
                                              QStringLiteral("inverse depth spread")))
            {
                return result;
            }

            auto load_optional_adaptive_evidence = [&](const QString& path, cv::Mat* destination, const QString& label)
            {
                destination->release();
                if (path.isEmpty())
                {
                    return true;
                }
                reason.clear();
                if (!loadFloatMatrix(path, destination, &reason))
                {
                    result.errorMessage = frameArtifactError(artifact, QStringLiteral("%1: %2").arg(label, reason));
                    return false;
                }
                if (destination->size() != frame.depth.size())
                {
                    if (!artifact.pyramidFallback)
                    {
                        result.errorMessage = frameArtifactError(
                            artifact, QStringLiteral("%1: dimensions do not match depth").arg(label));
                        return false;
                    }
                    cv::resize(*destination, *destination, frame.depth.size(), 0.0, 0.0, cv::INTER_AREA);
                }
                return true;
            };
            if (!load_optional_adaptive_evidence(artifact.adaptiveGeometrySupportWeightPath,
                                                 &frame.adaptiveGeometrySupportWeight,
                                                 QStringLiteral("adaptive geometry support weight")) ||
                !load_optional_adaptive_evidence(artifact.adaptiveGeometryEffectiveViewCountPath,
                                                 &frame.adaptiveGeometryEffectiveViewCount,
                                                 QStringLiteral("adaptive geometry effective view count")) ||
                !load_optional_adaptive_evidence(artifact.adaptiveGeometryConflictRatioPath,
                                                 &frame.adaptiveGeometryConflictRatio,
                                                 QStringLiteral("adaptive geometry conflict ratio")))
            {
                return result;
            }

            auto validate_adaptive_evidence =
                [&](const cv::Mat& matrix, const QString& label, bool bounded_probability, bool effective_view_count)
            {
                if (matrix.empty())
                {
                    return true;
                }
                for (int row = 0; row < matrix.rows; ++row)
                {
                    const float* values = matrix.ptr<float>(row);
                    for (int column = 0; column < matrix.cols; ++column)
                    {
                        const float value = values[column];
                        if (!std::isfinite(value) || value < 0.0f || (bounded_probability && value > 1.0f))
                        {
                            result.errorMessage = frameArtifactError(
                                artifact,
                                QStringLiteral("%1 contains an invalid value at row=%2 column=%3: value=%4")
                                    .arg(label)
                                    .arg(row)
                                    .arg(column)
                                    .arg(value));
                            return false;
                        }
                        if (effective_view_count && value > 0.0f && value < 1.0f)
                        {
                            result.errorMessage = frameArtifactError(
                                artifact,
                                QStringLiteral("adaptive geometry effective view count must be zero or "
                                               "at least one at row=%1 column=%2; value=%3")
                                    .arg(row)
                                    .arg(column)
                                    .arg(value));
                            return false;
                        }
                    }
                }
                return true;
            };
            if (!validate_adaptive_evidence(frame.adaptiveGeometrySupportWeight,
                                            QStringLiteral("adaptive geometry support weight"),
                                            true,
                                            false) ||
                !validate_adaptive_evidence(frame.adaptiveGeometryEffectiveViewCount,
                                            QStringLiteral("adaptive geometry effective view count"),
                                            false,
                                            true) ||
                !validate_adaptive_evidence(frame.adaptiveGeometryConflictRatio,
                                            QStringLiteral("adaptive geometry conflict ratio"),
                                            true,
                                            false))
            {
                return result;
            }

            if (artifact.crossViewRepairedMaskPath.isEmpty())
            {
                frame.crossViewRepairedMask = cv::Mat(frame.depth.size(), CV_8UC1, cv::Scalar(0));
            }
            else if (!loadMask(artifact.crossViewRepairedMaskPath,
                               frame.depth.size(),
                               &frame.crossViewRepairedMask,
                               &reason,
                               artifact.pyramidFallback))
            {
                result.errorMessage =
                    frameArtifactError(artifact, QStringLiteral("cross-view repaired mask: %1").arg(reason));
                return result;
            }

            if (artifact.depthProvenancePath.isEmpty())
            {
                frame.depthProvenance =
                    cv::Mat(frame.depth.size(),
                            CV_8UC1,
                            cv::Scalar(static_cast<std::uint8_t>(xjw::mvs::DepthProvenance::NativePatchMatch)));
                frame.depthProvenance.setTo(cv::Scalar(0), frame.depth <= 0.0f);
            }
            else if (!loadByteMap(artifact.depthProvenancePath,
                                  frame.depth.size(),
                                  &frame.depthProvenance,
                                  &reason,
                                  artifact.pyramidFallback))
            {
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("depth provenance: %1").arg(reason));
                return result;
            }

            if (artifact.validMaskPath.isEmpty())
            {
                frame.depthValidMask = frame.depth > 0.0f;
            }
            else if (!loadMask(artifact.validMaskPath,
                               frame.depth.size(),
                               &frame.depthValidMask,
                               &reason,
                               artifact.pyramidFallback))
            {
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("depth-valid mask: %1").arg(reason));
                return result;
            }

            if (artifact.supportMaskPath.isEmpty())
            {
                frame.supportMask = cv::Mat(frame.depth.size(), CV_8UC1, cv::Scalar(255));
            }
            else if (!loadMask(artifact.supportMaskPath,
                               frame.depth.size(),
                               &frame.supportMask,
                               &reason,
                               artifact.pyramidFallback))
            {
                result.errorMessage = frameArtifactError(artifact, QStringLiteral("support mask: %1").arg(reason));
                return result;
            }

            if (!artifact.refImage.isEmpty() && QFileInfo::exists(artifact.refImage))
            {
                frame.colorBgr =
                    xjw::common::io::readImage(xjw::common::io::toUtf8Path(artifact.refImage), cv::IMREAD_COLOR);
                if (!frame.colorBgr.empty() && frame.colorBgr.size() != frame.depth.size())
                {
                    cv::resize(frame.colorBgr, frame.colorBgr, frame.depth.size(), 0.0, 0.0, cv::INTER_AREA);
                }
            }
            frame.frameQualityWeight = artifact.meanConfidence >= 0.0
                                           ? static_cast<float>(std::clamp(artifact.meanConfidence, 0.05, 1.0))
                                           : 1.0f;
            frame.auxiliarySurfaceOnly = validation_only;
            frame.validWithinMaskRatio = artifact.validWithinMaskRatio;
            frame.consistencyRetentionRatio = artifact.consistencyRetentionRatio;
            frame.largestComponentRatio = artifact.largestComponentRatio;
            frame.meanConfidence = artifact.meanConfidence;
            frame.sparseAbsoluteDepthMedianLogError = artifact.sparseAbsoluteDepthMedianLogError;
            frame.sourceViewCount = artifact.sourceViewCount;
            frame.qualityReasonCount = artifact.qualityReasons.size();
            frame.trustedGeometryCorePixelCount = artifact.trustedGeometryCorePixelCount;
            frame.auxiliaryBridgeEligible = artifact.role == xjw::mvs::DepthFrameRole::CoverageAuxiliary &&
                                            artifact.algorithmRevision >= xjw::mvs::kMvsGeometrySourceOrdinalRevision;
            frame.auxiliaryBridgeSelected = artifact.auxiliaryBridgeSelected;
            frame.useAdaptiveGeometryEvidence = !artifact.useDiscreteGeometryFallback && requires_conflict_ratio &&
                                                !frame.adaptiveGeometrySupportWeight.empty() &&
                                                !frame.adaptiveGeometryEffectiveViewCount.empty() &&
                                                !frame.adaptiveGeometryConflictRatio.empty();
            result.frames.push_back(std::move(frame));
        }

        if (result.frames.size() < minimumFrameCount)
        {
            result.errorMessage =
                QStringLiteral("TSDF requires at least %1 usable primary or auxiliary depth frames; loaded=%2")
                    .arg(minimumFrameCount)
                    .arg(result.frames.size());
            return result;
        }
        result.ok = true;
        return result;
    }
} // namespace xjw::mesh
