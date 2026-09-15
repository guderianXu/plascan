#include "DepthTsdfInternals.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;

    DepthTsdfBoundsResult DepthTsdfSurfaceBuilder::estimateBounds(const QVector<DepthTsdfFrame>& frames,
                                                                  bool use_evidence_aware_bounds,
                                                                  bool preserve_per_frame_coverage_bounds)
    {
        constexpr std::uint64_t minimum_trusted_sample_count = 500;
        constexpr double minimum_trusted_sample_ratio = 0.10;

        DepthTsdfBoundsResult result;
        if (frames.size() < 3)
        {
            result.errorMessage = QStringLiteral("TSDF bounds require at least 3 usable depth frames");
            return result;
        }
        if (std::none_of(frames.cbegin(),
                         frames.cend(),
                         [](const DepthTsdfFrame& frame) { return !frame.auxiliarySurfaceOnly; }))
        {
            result.errorMessage = QStringLiteral("TSDF bounds require at least 1 primary depth frame");
            return result;
        }

        std::array<std::vector<float>, 3> candidate_coordinates;
        std::array<std::vector<float>, 3> trusted_coordinates;
        std::array<std::vector<float>, 3> candidate_frame_lows;
        std::array<std::vector<float>, 3> candidate_frame_highs;
        std::array<std::vector<float>, 3> trusted_frame_lows;
        std::array<std::vector<float>, 3> trusted_frame_highs;
        for (const DepthTsdfFrame& frame : frames)
        {
            if (frame.auxiliarySurfaceOnly || !frame.camera.isValid() || frame.depth.type() != CV_32FC1 ||
                frame.depthValidMask.type() != CV_8UC1 || frame.depthValidMask.size() != frame.depth.size() ||
                frame.supportMask.type() != CV_8UC1 || frame.supportMask.size() != frame.depth.size())
            {
                continue;
            }
            const bool has_geometry_evidence = frame.geometrySupportCount.type() == CV_16UC1 &&
                                               frame.geometrySupportCount.size() == frame.depth.size();
            const bool has_adaptive_evidence = frame.useAdaptiveGeometryEvidence &&
                                               frame.adaptiveGeometryEffectiveViewCount.type() == CV_32FC1 &&
                                               frame.adaptiveGeometryEffectiveViewCount.size() == frame.depth.size();
            std::array<std::vector<float>, 3> frame_candidate_coordinates;
            std::array<std::vector<float>, 3> frame_trusted_coordinates;
            const int stride =
                std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(frame.depth.total()) / 6000.0))));
            for (int row = 0; row < frame.depth.rows; row += stride)
            {
                for (int column = 0; column < frame.depth.cols; column += stride)
                {
                    if (frame.depthValidMask.at<std::uint8_t>(row, column) == 0 ||
                        frame.supportMask.at<std::uint8_t>(row, column) == 0)
                    {
                        continue;
                    }
                    const float depth = frame.depth.at<float>(row, column);
                    if (!std::isfinite(depth) || depth <= 0.0f)
                    {
                        continue;
                    }
                    const double pixel[2] = {static_cast<double>(column), static_cast<double>(row)};
                    double world[3] = {};
                    if (!frame.camera.unprojectPixel(pixel, depth, world) || !std::isfinite(world[0]) ||
                        !std::isfinite(world[1]) || !std::isfinite(world[2]))
                    {
                        continue;
                    }
                    const bool trusted = has_geometry_evidence &&
                                         frame.geometrySupportCount.at<std::uint16_t>(row, column) >= 2 &&
                                         (!has_adaptive_evidence ||
                                          frame.adaptiveGeometryEffectiveViewCount.at<float>(row, column) >= 2.0f);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const float coordinate = static_cast<float>(world[axis]);
                        candidate_coordinates[axis].push_back(coordinate);
                        frame_candidate_coordinates[axis].push_back(coordinate);
                        if (trusted)
                        {
                            trusted_coordinates[axis].push_back(coordinate);
                            frame_trusted_coordinates[axis].push_back(coordinate);
                        }
                    }
                }
            }
            const auto append_frame_extent = [](std::array<std::vector<float>, 3>* frame_coordinates,
                                                std::array<std::vector<float>, 3>* frame_lows,
                                                std::array<std::vector<float>, 3>* frame_highs)
            {
                if (frame_coordinates == nullptr || frame_lows == nullptr || frame_highs == nullptr ||
                    (*frame_coordinates)[0].size() < 100)
                {
                    return;
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    std::vector<float>& coordinates = (*frame_coordinates)[axis];
                    std::sort(coordinates.begin(), coordinates.end());
                    const std::size_t last = coordinates.size() - 1;
                    (*frame_lows)[axis].push_back(coordinates[static_cast<std::size_t>(last * 0.01)]);
                    (*frame_highs)[axis].push_back(coordinates[static_cast<std::size_t>(last * 0.99)]);
                }
            };
            append_frame_extent(&frame_candidate_coordinates, &candidate_frame_lows, &candidate_frame_highs);
            append_frame_extent(&frame_trusted_coordinates, &trusted_frame_lows, &trusted_frame_highs);
        }

        result.candidateSampleCount = candidate_coordinates[0].size();
        result.trustedSampleCount = trusted_coordinates[0].size();
        if (result.candidateSampleCount < 500)
        {
            result.selectionReason = QStringLiteral("insufficient_candidate_samples");
            result.errorMessage =
                QStringLiteral("Insufficient finite TSDF bound samples: %1").arg(result.candidateSampleCount);
            return result;
        }
        const bool sufficient_trusted_samples =
            use_evidence_aware_bounds && result.trustedSampleCount >= minimum_trusted_sample_count &&
            static_cast<double>(result.trustedSampleCount) >=
                static_cast<double>(result.candidateSampleCount) * minimum_trusted_sample_ratio;
        result.usedEvidenceAwareSamples = sufficient_trusted_samples;
        result.fellBackToCandidateSamples = use_evidence_aware_bounds && !sufficient_trusted_samples;
        result.selectionReason = !use_evidence_aware_bounds ? QStringLiteral("candidate_samples_requested")
                                 : sufficient_trusted_samples
                                     ? QStringLiteral("trusted_multiview_evidence")
                                     : (result.trustedSampleCount < minimum_trusted_sample_count
                                            ? QStringLiteral("insufficient_trusted_sample_count")
                                            : QStringLiteral("insufficient_trusted_sample_ratio"));
        std::array<std::vector<float>, 3>& coordinates =
            sufficient_trusted_samples ? trusted_coordinates : candidate_coordinates;
        std::array<std::vector<float>, 3>& frame_lows =
            sufficient_trusted_samples ? trusted_frame_lows : candidate_frame_lows;
        std::array<std::vector<float>, 3>& frame_highs =
            sufficient_trusted_samples ? trusted_frame_highs : candidate_frame_highs;
        const bool used_per_frame_coverage_bounds =
            preserve_per_frame_coverage_bounds && frame_lows[0].size() >= 3 && frame_highs[0].size() >= 3;
        if (used_per_frame_coverage_bounds)
        {
            result.selectionReason += QStringLiteral("_per_frame_coverage");
        }
        result.sampleCount = coordinates[0].size();
        for (int axis = 0; axis < 3; ++axis)
        {
            std::sort(coordinates[axis].begin(), coordinates[axis].end());
            const std::size_t last = coordinates[axis].size() - 1;
            float low = coordinates[axis][static_cast<std::size_t>(last * 0.01)];
            float high = coordinates[axis][static_cast<std::size_t>(last * 0.99)];
            if (used_per_frame_coverage_bounds)
            {
                std::sort(frame_lows[axis].begin(), frame_lows[axis].end());
                std::sort(frame_highs[axis].begin(), frame_highs[axis].end());
                const std::size_t frame_last = frame_lows[axis].size() - 1;
                low = frame_lows[axis][static_cast<std::size_t>(frame_last * 0.02)];
                high = frame_highs[axis][static_cast<std::size_t>(frame_last * 0.98)];
            }
            const float padding = std::max((high - low) * 0.08f, 1.0e-5f);
            result.minimum[axis] = low - padding;
            result.maximum[axis] = high + padding;
        }
        result.ok = true;
        return result;
    }

    QVector<float> DepthTsdfSurfaceBuilder::robustFrameQualityWeights(const QVector<float>& rawWeights,
                                                                      float minimumMultiplier,
                                                                      float madFloor,
                                                                      float penaltyOnset,
                                                                      float penaltyStrength,
                                                                      float* median,
                                                                      float* scale)
    {
        QVector<float> result;
        result.reserve(rawWeights.size());
        if (rawWeights.isEmpty())
        {
            if (median)
            {
                *median = 0.0f;
            }
            if (scale)
            {
                *scale = 0.0f;
            }
            return result;
        }

        QVector<float> sorted_weights;
        sorted_weights.reserve(rawWeights.size());
        for (const float weight : rawWeights)
        {
            sorted_weights.push_back(std::clamp(std::isfinite(weight) ? weight : 0.0f, 0.0f, 1.0f));
        }
        std::sort(sorted_weights.begin(), sorted_weights.end());
        const auto medianOfSorted = [](const QVector<float>& values)
        {
            const int middle = values.size() / 2;
            return values.size() % 2 == 0 ? 0.5f * (values[middle - 1] + values[middle]) : values[middle];
        };
        const float robust_median = medianOfSorted(sorted_weights);
        QVector<float> deviations;
        deviations.reserve(sorted_weights.size());
        for (const float weight : sorted_weights)
        {
            deviations.push_back(std::fabs(weight - robust_median));
        }
        std::sort(deviations.begin(), deviations.end());
        const float robust_scale = std::max(std::max(0.0f, madFloor), 1.4826f * medianOfSorted(deviations));
        const float bounded_minimum_multiplier = std::clamp(minimumMultiplier, 0.0f, 1.0f);
        const float bounded_onset = std::max(0.0f, penaltyOnset);
        const float bounded_strength = std::max(0.0f, penaltyStrength);
        for (const float raw_weight : rawWeights)
        {
            const float weight = std::clamp(std::isfinite(raw_weight) ? raw_weight : 0.0f, 0.0f, 1.0f);
            const float low_tail_distance =
                robust_scale > 0.0f ? std::max(0.0f, (robust_median - weight) / robust_scale - bounded_onset) : 0.0f;
            const float multiplier = std::max(bounded_minimum_multiplier,
                                              std::exp(-bounded_strength * low_tail_distance * low_tail_distance));
            result.push_back(weight * multiplier);
        }
        if (median)
        {
            *median = robust_median;
        }
        if (scale)
        {
            *scale = robust_scale;
        }
        return result;
    }
} // namespace xjw::mesh
