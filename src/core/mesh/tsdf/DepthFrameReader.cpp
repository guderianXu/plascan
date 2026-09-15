#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    QString frameArtifactError(const DepthFrameArtifact& artifact, const QString& reason)
    {
        return QStringLiteral("Invalid TSDF frame ref_index=%1 depth=%2 confidence=%3 geometry_support=%4 "
                              "depth_valid_mask=%5 support_mask=%6: %7")
            .arg(artifact.refIndex)
            .arg(artifact.depthPath,
                 artifact.confidencePath,
                 artifact.geometrySupportPath,
                 artifact.validMaskPath,
                 artifact.supportMaskPath,
                 reason);
    }

    bool loadsAsAuxiliarySurfaceOnly(const DepthFrameArtifact& artifact)
    {
        return artifact.role == xjw::mvs::DepthFrameRole::CoverageAuxiliary;
    }

    bool canLoadDepthFrameArtifact(const DepthFrameArtifact& artifact)
    {
        return artifact.role != xjw::mvs::DepthFrameRole::Excluded;
    }

    DepthAuxiliaryBridgeNode bridgeNode(const DepthFrameArtifact& artifact, int frame_index)
    {
        DepthAuxiliaryBridgeNode node;
        node.frameIndex = frame_index;
        node.refIndex = artifact.refIndex;
        node.primary = xjw::mvs::isPrimaryFusionFrame(artifact.role);
        node.geometrySourceIndices.assign(artifact.geometrySourceIndices.cbegin(),
                                          artifact.geometrySourceIndices.cend());
        node.sparseAbsoluteDepthMedianLogError = artifact.sparseAbsoluteDepthMedianLogError;
        node.validWithinMaskRatio = artifact.validWithinMaskRatio;
        node.consistencyRetentionRatio = artifact.consistencyRetentionRatio;
        node.largestComponentRatio = artifact.largestComponentRatio;
        node.meanConfidence = artifact.meanConfidence;
        node.sourceViewCount = artifact.sourceViewCount;
        node.qualityReasonCount = artifact.qualityReasons.size();
        node.trustedPixelCount = artifact.role == xjw::mvs::DepthFrameRole::CoverageAuxiliary &&
                                         artifact.algorithmRevision >= xjw::mvs::kMvsGeometrySourceOrdinalRevision
                                     ? artifact.trustedGeometryCorePixelCount
                                     : 0;
        return node;
    }

    DepthAuxiliaryBridgeNode bridgeNode(const DepthTsdfFrame& frame, int frame_index)
    {
        DepthAuxiliaryBridgeNode node;
        node.frameIndex = frame_index;
        node.refIndex = frame.refIndex;
        node.primary = !frame.auxiliarySurfaceOnly;
        node.geometrySourceIndices.assign(frame.geometrySourceIndices.cbegin(), frame.geometrySourceIndices.cend());
        node.sparseAbsoluteDepthMedianLogError = frame.sparseAbsoluteDepthMedianLogError;
        node.validWithinMaskRatio = frame.validWithinMaskRatio;
        node.consistencyRetentionRatio = frame.consistencyRetentionRatio;
        node.largestComponentRatio = frame.largestComponentRatio;
        node.meanConfidence = frame.meanConfidence;
        node.sourceViewCount = frame.sourceViewCount;
        node.qualityReasonCount = frame.qualityReasonCount;
        node.trustedPixelCount = frame.auxiliaryBridgeEligible ? frame.trustedGeometryCorePixelCount : 0;
        return node;
    }

    std::uint64_t estimatedResidentDepthFrameBytes(const DepthFrameArtifact& artifact)
    {
        // Full orbital evidence keeps seven float fields, two uint16 fields,
        // four byte masks, and a BGR source image resident for each frame.
        constexpr std::uint64_t kResidentBytesPerPixel = 39u;
        std::uint64_t pixel_count = 0;
        std::uint64_t estimated_bytes = 0;
        if (artifact.gridWidth > 0 && artifact.gridHeight > 0 &&
            checkedMultiply(static_cast<std::uint64_t>(artifact.gridWidth),
                            static_cast<std::uint64_t>(artifact.gridHeight),
                            &pixel_count) &&
            checkedMultiply(pixel_count, kResidentBytesPerPixel, &estimated_bytes))
        {
            return estimated_bytes;
        }

        const qint64 depth_file_bytes = QFileInfo(artifact.depthPath).size();
        if (depth_file_bytes <= 0 ||
            static_cast<std::uint64_t>(depth_file_bytes) > std::numeric_limits<std::uint64_t>::max() / 10u)
        {
            return 0u;
        }
        return static_cast<std::uint64_t>(depth_file_bytes) * 10u;
    }

    QVector<DepthFrameArtifact> selectMemoryBoundedDepthFrameArtifacts(const QVector<DepthFrameArtifact>& artifacts)
    {
        QVector<DepthFrameArtifact> usable;
        usable.reserve(artifacts.size());
        for (const DepthFrameArtifact& artifact : artifacts)
        {
            if (canLoadDepthFrameArtifact(artifact))
            {
                usable.push_back(artifact);
            }
        }
        if (usable.size() <= 3)
        {
            return usable;
        }

        const std::uint64_t available_bytes = availablePhysicalMemoryBytes();
        std::uint64_t maximum_frame_bytes = 0;
        for (const DepthFrameArtifact& artifact : usable)
        {
            maximum_frame_bytes = std::max(maximum_frame_bytes, estimatedResidentDepthFrameBytes(artifact));
        }
        if (available_bytes == 0 || maximum_frame_bytes == 0)
        {
            return usable;
        }

        // Leave memory for the TSDF volume, extraction buffers, Qt, and temporary
        // matrix decoders. Primary fusion frames carry the geometry that passed the
        // quality gate, so retain them before optional auxiliary surface-only
        // frames. When auxiliary frames must be sampled, use calibrated camera
        // positions below instead of assuming import identifiers encode angle.
        const std::uint64_t frame_budget_bytes = available_bytes * 55u / 100u;
        const int usable_count = static_cast<int>(usable.size());
        const int maximum_frame_count =
            std::clamp(static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(usable_count),
                                                                frame_budget_bytes / maximum_frame_bytes)),
                       3,
                       usable_count);
        if (maximum_frame_count >= usable_count)
        {
            return usable;
        }

        QVector<DepthFrameArtifact> primary;
        QVector<DepthFrameArtifact> auxiliary;
        primary.reserve(usable.size());
        auxiliary.reserve(usable.size());
        for (const DepthFrameArtifact& artifact : usable)
        {
            (loadsAsAuxiliarySurfaceOnly(artifact) ? auxiliary : primary).push_back(artifact);
        }
        const auto sort_by_ref_index = [](QVector<DepthFrameArtifact>* frames)
        {
            std::sort(frames->begin(),
                      frames->end(),
                      [](const DepthFrameArtifact& lhs, const DepthFrameArtifact& rhs)
                      { return lhs.refIndex < rhs.refIndex; });
        };
        sort_by_ref_index(&primary);
        sort_by_ref_index(&auxiliary);

        QVector<DepthFrameArtifact> selected;
        selected.reserve(maximum_frame_count);
        const auto append_uniform_samples =
            [&selected](const QVector<DepthFrameArtifact>& candidates, int requested_count)
        {
            const int sample_count = std::clamp(requested_count, 0, static_cast<int>(candidates.size()));
            for (int slot = 0; slot < sample_count; ++slot)
            {
                const qint64 numerator = static_cast<qint64>(2 * slot + 1) * candidates.size();
                const int index = static_cast<int>(numerator / (2 * sample_count));
                selected.push_back(candidates[index]);
            }
        };
        const int primary_count = std::min(maximum_frame_count, static_cast<int>(primary.size()));
        append_uniform_samples(primary, primary_count);
        const int auxiliary_count = maximum_frame_count - primary_count;

        // Reserve the exact validation frames needed to connect the primary
        // geometry-source graph before camera-coverage sampling spends the
        // auxiliary memory budget. This is deliberately graph based: ref_index is
        // an import identifier and cannot be used as an angular surrogate.
        std::vector<DepthAuxiliaryBridgeNode> bridge_nodes;
        bridge_nodes.reserve(static_cast<std::size_t>(usable.size()));
        for (int index = 0; index < usable.size(); ++index)
        {
            bridge_nodes.push_back(bridgeNode(usable[index], index));
        }
        const DepthAuxiliaryBridgeSelectionResult bridge_selection = DepthAuxiliaryBridgeSelector::select(bridge_nodes);
        QSet<int> reserved_bridge_refs;
        if (bridge_selection.connected &&
            static_cast<int>(bridge_selection.selectedAuxiliaryRefIndices.size()) <= auxiliary_count)
        {
            for (const int ref_index : bridge_selection.selectedAuxiliaryRefIndices)
            {
                reserved_bridge_refs.insert(ref_index);
            }
        }
        QVector<DepthFrameArtifact> remaining_auxiliary;
        remaining_auxiliary.reserve(auxiliary.size());
        for (DepthFrameArtifact artifact : auxiliary)
        {
            if (reserved_bridge_refs.contains(artifact.refIndex))
            {
                artifact.auxiliaryBridgeSelected = true;
                selected.push_back(std::move(artifact));
            }
            else
            {
                remaining_auxiliary.push_back(std::move(artifact));
            }
        }
        auxiliary = std::move(remaining_auxiliary);
        const int remaining_auxiliary_count = auxiliary_count - reserved_bridge_refs.size();
        bool selected_auxiliary_by_coverage = false;
        const bool orbital_selection =
            remaining_auxiliary_count > 0 && primary_count >= 3 &&
            std::all_of(selected.cbegin(),
                        selected.cend(),
                        [](const DepthFrameArtifact& artifact)
                        {
                            return xjw::mvs::isOrbitalDepthSceneProfile(artifact.sceneProfile) &&
                                   artifact.hasCameraModel && artifact.cameraModel.isValid();
                        }) &&
            std::all_of(auxiliary.cbegin(),
                        auxiliary.cend(),
                        [](const DepthFrameArtifact& artifact)
                        {
                            return xjw::mvs::isOrbitalDepthSceneProfile(artifact.sceneProfile) &&
                                   artifact.hasCameraModel && artifact.cameraModel.isValid();
                        });
        if (orbital_selection)
        {
            // ref_index is an import identifier, not a camera angle.  Rank equal-
            // coverage alternatives by absolute-depth agreement, then let the
            // orbital policy greedily fill the largest actual camera-space gaps.
            std::sort(auxiliary.begin(),
                      auxiliary.end(),
                      [](const DepthFrameArtifact& lhs, const DepthFrameArtifact& rhs)
                      {
                          const double lhs_residual = lhs.sparseAbsoluteDepthMedianLogError >= 0.0
                                                          ? lhs.sparseAbsoluteDepthMedianLogError
                                                          : std::numeric_limits<double>::infinity();
                          const double rhs_residual = rhs.sparseAbsoluteDepthMedianLogError >= 0.0
                                                          ? rhs.sparseAbsoluteDepthMedianLogError
                                                          : std::numeric_limits<double>::infinity();
                          if (lhs_residual != rhs_residual)
                          {
                              return lhs_residual < rhs_residual;
                          }
                          if (lhs.qualityReasons.size() != rhs.qualityReasons.size())
                          {
                              return lhs.qualityReasons.size() < rhs.qualityReasons.size();
                          }
                          if (lhs.meanConfidence != rhs.meanConfidence)
                          {
                              return lhs.meanConfidence > rhs.meanConfidence;
                          }
                          if (lhs.validWithinMaskRatio != rhs.validWithinMaskRatio)
                          {
                              return lhs.validWithinMaskRatio > rhs.validWithinMaskRatio;
                          }
                          return lhs.refIndex < rhs.refIndex;
                      });
            std::vector<DepthFusionView> fixed_views;
            fixed_views.reserve(static_cast<std::size_t>(selected.size()));
            for (int index = 0; index < static_cast<int>(selected.size()); ++index)
            {
                fixed_views.push_back({index, selected[index].refIndex, selected[index].cameraModel.cameraCenter()});
            }
            std::vector<DepthFusionView> candidate_views;
            candidate_views.reserve(static_cast<std::size_t>(auxiliary.size()));
            for (int index = 0; index < static_cast<int>(auxiliary.size()); ++index)
            {
                candidate_views.push_back(
                    {index, auxiliary[index].refIndex, auxiliary[index].cameraModel.cameraCenter()});
            }
            const std::vector<int> selected_indices = DepthFusionFramePolicy::selectCoverageComplementaryCandidates(
                fixed_views, candidate_views, remaining_auxiliary_count);
            if (static_cast<int>(selected_indices.size()) == remaining_auxiliary_count)
            {
                for (const int index : selected_indices)
                {
                    selected.push_back(auxiliary[index]);
                }
                selected_auxiliary_by_coverage = true;
            }
        }
        if (!selected_auxiliary_by_coverage)
        {
            sort_by_ref_index(&auxiliary);
            append_uniform_samples(auxiliary, remaining_auxiliary_count);
        }
        std::sort(selected.begin(),
                  selected.end(),
                  [](const DepthFrameArtifact& lhs, const DepthFrameArtifact& rhs)
                  { return lhs.refIndex < rhs.refIndex; });
        return selected;
    }

    bool loadFloatMatrix(const QString& path, cv::Mat* matrix, QString* reason)
    {
        if (!matrix)
        {
            return false;
        }
        const xjw::common::OperationResult status = xjw::core::project::loadDepthMatStorage(path, matrix);
        if (!status.ok || matrix->empty())
        {
            if (reason)
            {
                *reason = status.errorMessage.isEmpty() ? QStringLiteral("matrix is empty") : status.errorMessage;
            }
            return false;
        }
        if (matrix->type() != CV_32FC1)
        {
            if (reason)
            {
                *reason = QStringLiteral("expected CV_32FC1, got type=%1").arg(matrix->type());
            }
            return false;
        }
        return true;
    }

    bool loadUnsignedShortMatrix(const QString& path, cv::Mat* matrix, QString* reason)
    {
        if (!matrix)
        {
            return false;
        }
        const xjw::common::OperationResult status = xjw::core::project::loadDepthMatStorage(path, matrix);
        if (!status.ok || matrix->empty())
        {
            if (reason)
            {
                *reason = status.errorMessage.isEmpty() ? QStringLiteral("matrix is empty") : status.errorMessage;
            }
            return false;
        }
        if (matrix->type() != CV_16UC1)
        {
            if (reason)
            {
                *reason = QStringLiteral("expected CV_16UC1, got type=%1").arg(matrix->type());
            }
            return false;
        }
        return true;
    }

    bool loadMask(const QString& path, const cv::Size& size, cv::Mat* mask, QString* reason, bool allow_resize)
    {
        if (!mask)
        {
            return false;
        }
        *mask = xjw::common::io::readImage(xjw::common::io::toUtf8Path(path), cv::IMREAD_GRAYSCALE);
        if (mask->empty())
        {
            if (reason)
            {
                *reason = QStringLiteral("mask cannot be read");
            }
            return false;
        }
        if (mask->type() == CV_8UC1 && mask->size() != size && allow_resize)
        {
            cv::resize(*mask, *mask, size, 0.0, 0.0, cv::INTER_NEAREST);
        }
        if (mask->type() != CV_8UC1 || mask->size() != size)
        {
            if (reason)
            {
                *reason = QStringLiteral("expected CV_8UC1 %1x%2, got type=%3 %4x%5")
                              .arg(size.width)
                              .arg(size.height)
                              .arg(mask->type())
                              .arg(mask->cols)
                              .arg(mask->rows);
            }
            return false;
        }
        cv::threshold(*mask, *mask, 0.0, 255.0, cv::THRESH_BINARY);
        return true;
    }
} // namespace xjw::mesh::tsdf_detail
