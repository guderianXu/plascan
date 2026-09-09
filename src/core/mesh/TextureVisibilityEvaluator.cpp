#include "TextureMappingV4Internal.h"

#include "TextureCandidateCost.h"
#include "TextureLabelOptimizer.h"
#include "TextureOverlapExposure.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>

namespace xjw::mesh::texture_v4
{

namespace
{

constexpr std::array<std::array<double, 3>, 7> kSampleWeights{{
    {{1.0, 0.0, 0.0}},
    {{0.0, 1.0, 0.0}},
    {{0.0, 0.0, 1.0}},
    {{0.5, 0.5, 0.0}},
    {{0.0, 0.5, 0.5}},
    {{0.5, 0.0, 0.5}},
    {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}}
}};

bool cancelled(const TextureMappingConfig &config)
{
    return config.isCancelled && config.isCancelled();
}

bool projectColorTriangle(const PreparedView &view,
                          const FaceGeometry &face,
                          std::array<QPointF, 3> *pixels)
{
    for (int corner = 0; corner < 3; ++corner)
    {
        double pixel[2]{};
        double depth = 0.0;
        if (!view.colorCamera.projectWorldPointWithDepth(
                face.vertices[corner].data(), pixel, depth) ||
            !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
            !std::isfinite(depth) || depth <= 1.0e-8 ||
            pixel[0] < 0.0 || pixel[1] < 0.0 ||
            pixel[0] > view.colorBgr.cols - 1.0 ||
            pixel[1] > view.colorBgr.rows - 1.0)
        {
            return false;
        }
        (*pixels)[corner] = QPointF(pixel[0], pixel[1]);
    }
    return true;
}

double projectedArea(const std::array<QPointF, 3> &pixels)
{
    return std::fabs(
        (pixels[1].x() - pixels[0].x()) * (pixels[2].y() - pixels[0].y()) -
        (pixels[1].y() - pixels[0].y()) * (pixels[2].x() - pixels[0].x())) * 0.5;
}

float viewAngleScore(const PreparedView &view, const FaceGeometry &face)
{
    const std::array<double, 3> center = view.colorCamera.cameraCenter();
    float direction[3]{
        static_cast<float>(center[0] - face.centroid[0]),
        static_cast<float>(center[1] - face.centroid[1]),
        static_cast<float>(center[2] - face.centroid[2])};
    const float length = std::sqrt(
        direction[0] * direction[0] +
        direction[1] * direction[1] +
        direction[2] * direction[2]);
    if (length <= 1.0e-10f)
    {
        return 0.0f;
    }
    for (float &value : direction)
    {
        value /= length;
    }
    return std::fabs(
        face.normal[0] * direction[0] +
        face.normal[1] * direction[1] +
        face.normal[2] * direction[2]);
}

bool evaluateEvidence(const PreparedView &view,
                      const FaceGeometry &face,
                      const TextureMappingConfig &config,
                      double median_edge_length,
                      bool strict,
                      float *depth_score,
                      TextureMappingResult *result)
{
    int valid_depth_samples = 0;
    float accumulated_score = 0.0f;
    for (int sample_index = 0;
         sample_index < static_cast<int>(kSampleWeights.size());
         ++sample_index)
    {
        const auto &weights = kSampleWeights[static_cast<std::size_t>(sample_index)];
        double world[3]{};
        for (int axis = 0; axis < 3; ++axis)
        {
            world[axis] =
                weights[0] * face.vertices[0][axis] +
                weights[1] * face.vertices[1][axis] +
                weights[2] * face.vertices[2][axis];
        }
        double pixel[2]{};
        double camera_depth = 0.0;
        if (!view.evidenceCamera.projectWorldPointWithDepth(
                world, pixel, camera_depth) ||
            !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
            !std::isfinite(camera_depth) || camera_depth <= 0.0 ||
            pixel[0] < 0.0 || pixel[1] < 0.0 ||
            pixel[0] > view.supportMask->cols - 1.0 ||
            pixel[1] > view.supportMask->rows - 1.0)
        {
            ++result->rejectedProjectionCount;
            return false;
        }
        const int column = static_cast<int>(std::lround(pixel[0]));
        const int row = static_cast<int>(std::lround(pixel[1]));
        if (row < 0 || column < 0 ||
            row >= view.supportMask->rows || column >= view.supportMask->cols ||
            view.supportMask->at<std::uint8_t>(row, column) == 0)
        {
            ++result->rejectedMaskCount;
            return false;
        }
        if (view.depthValidMask->at<std::uint8_t>(row, column) == 0)
        {
            if (strict)
            {
                ++result->rejectedDepthCount;
                return false;
            }
            continue;
        }

        const float observed_depth = view.depth->at<float>(row, column);
        const float confidence = view.confidence->at<float>(row, column);
        if (!std::isfinite(observed_depth) || observed_depth <= 0.0f ||
            !std::isfinite(confidence) || confidence < config.minimumConfidence)
        {
            if (strict)
            {
                ++result->rejectedDepthCount;
                return false;
            }
            continue;
        }
        const float tolerance = std::max(
            static_cast<float>(config.edgeLengthDepthTolerance * median_edge_length),
            config.relativeDepthTolerance *
                std::fabs(static_cast<float>(camera_depth)));
        const float residual =
            std::fabs(observed_depth - static_cast<float>(camera_depth));
        const float allowed = strict ? tolerance : tolerance * 2.0f;
        if (residual > allowed)
        {
            ++result->rejectedDepthCount;
            return false;
        }
        const float ratio = residual / std::max(allowed, 1.0e-8f);
        accumulated_score +=
            confidence * std::exp(-0.5f * ratio * ratio);
        ++valid_depth_samples;
    }

    const int required_samples = strict ? static_cast<int>(kSampleWeights.size()) : 5;
    if (valid_depth_samples < required_samples)
    {
        ++result->rejectedDepthCount;
        return false;
    }
    *depth_score = accumulated_score /
        static_cast<float>(std::max(valid_depth_samples, 1));
    return true;
}

FaceCandidate evaluateCandidate(const PreparedView &view,
                                int view_index,
                                int face_index,
                                const FaceGeometry &face,
                                const TextureMappingConfig &config,
                                double median_edge_length,
                                bool strict,
                                TextureMappingResult *result)
{
    ++result->candidateEvaluationCount;
    std::array<QPointF, 3> pixels{};
    if (!projectColorTriangle(view, face, &pixels))
    {
        ++result->rejectedProjectionCount;
        return {};
    }
    const double area = projectedArea(pixels);
    // Dense reconstruction commonly creates triangles smaller than one source
    // pixel.  Rejecting them individually tears otherwise valid connected
    // charts into fallback-colored facets.  Keep only a numerical degeneracy
    // guard here; resolutionScore still makes larger projections win.
    if (!std::isfinite(area) || area <= 1.0e-8)
    {
        ++result->rejectedResolutionCount;
        return {};
    }
    const float angle_score = viewAngleScore(view, face);
    if (angle_score < config.minimumViewCosine)
    {
        ++result->rejectedAngleCount;
        return {};
    }
    const bool require_final_mesh_visibility = area >= 1.0;
    if (require_final_mesh_visibility &&
        !isFinalMeshFaceVisibleSomewhere(view, face_index))
    {
        ++result->rejectedVisibilityCount;
        return {};
    }

    float depth_score = 0.0f;
    if (!evaluateEvidence(view,
                          face,
                          config,
                          median_edge_length,
                          strict,
                          &depth_score,
                          result))
    {
        return {};
    }

    const QPointF center =
        (pixels[0] + pixels[1] + pixels[2]) / 3.0;
    const double normalized_x =
        (center.x() - view.colorBgr.cols * 0.5) /
        std::max(1.0, view.colorBgr.cols * 0.5);
    const double normalized_y =
        (center.y() - view.colorBgr.rows * 0.5) /
        std::max(1.0, view.colorBgr.rows * 0.5);
    const float center_score = std::clamp(
        static_cast<float>(1.0 - 0.35 * std::sqrt(
            normalized_x * normalized_x + normalized_y * normalized_y)),
        0.35f,
        1.0f);
    const float projected_resolution = std::clamp(
        static_cast<float>(std::sqrt(area)), 0.10f, 64.0f);
    const int focus_column = std::clamp(
        static_cast<int>(std::lround(center.x())),
        0,
        view.focusQuality.cols - 1);
    const int focus_row = std::clamp(
        static_cast<int>(std::lround(center.y())),
        0,
        view.focusQuality.rows - 1);
    const float local_sharpness = std::max(
        0.0f,
        view.focusQuality.at<float>(focus_row, focus_column));
    const float score =
        view.qualityWeight *
        depth_score *
        angle_score *
        projected_resolution *
        std::sqrt(center_score);
    return {view_index,
            score,
            angle_score,
            std::clamp(projected_resolution / 8.0f, 0.10f, 1.0f),
            local_sharpness,
            depth_score,
            projected_resolution * std::sqrt(view.qualityWeight),
            0,
            strict,
            require_final_mesh_visibility};
}

const FaceCandidate *candidateForView(const FaceAssignment &assignment, int view_index)
{
    for (const FaceCandidate &candidate : assignment.candidates)
    {
        if (candidate.viewIndex == view_index)
        {
            return &candidate;
        }
    }
    return nullptr;
}

bool sampleProjectedColor(const PreparedView &view,
                          const std::array<double, 3> &world,
                          cv::Vec3f *color)
{
    double pixel[2]{};
    double depth = 0.0;
    if (!view.colorCamera.projectWorldPointWithDepth(
            world.data(), pixel, depth) ||
        !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
        !std::isfinite(depth) || depth <= 0.0 ||
        pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > view.colorBgr.cols - 1.0 ||
        pixel[1] > view.colorBgr.rows - 1.0)
    {
        return false;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    if (row < 0 || column < 0 ||
        row >= view.colorBgr.rows || column >= view.colorBgr.cols)
    {
        return false;
    }
    *color = applyLinearSrgbExposureGain(
        cv::Vec3f(view.colorBgr.at<cv::Vec3b>(row, column)),
        view.exposureGain);
    return true;
}

void buildRecoveredUnaryCosts(const PipelineData &data,
                              int face_index,
                              int maximum_candidates,
                              const TextureMappingConfig &config,
                              FaceAssignment *assignment)
{
    std::sort(assignment->candidates.begin(),
              assignment->candidates.end(),
              [](const auto &left, const auto &right)
    {
        return left.score > right.score ||
            (left.score == right.score && left.viewIndex < right.viewIndex);
    });
    if (assignment->candidates.size() > maximum_candidates)
    {
        assignment->candidates.resize(maximum_candidates);
    }
    struct LumaSample
    {
        int candidateIndex = -1;
        float value = 0.0f;
    };
    QVector<LumaSample> samples;
    samples.reserve(assignment->candidates.size());
    for (int candidate_index = 0;
         candidate_index < assignment->candidates.size();
         ++candidate_index)
    {
        const FaceCandidate &candidate =
            assignment->candidates[candidate_index];
        cv::Vec3f color;
        if (!sampleProjectedColor(data.views[candidate.viewIndex],
                                  data.geometry[face_index].centroid,
                                  &color))
        {
            continue;
        }
        samples.push_back({candidate_index,
                           0.0722f * color[0] +
                               0.7152f * color[1] +
                               0.2126f * color[2]});
    }
    if (config.enableGhostFilter && samples.size() >= 3)
    {
        QVector<float> values;
        values.reserve(samples.size());
        for (const LumaSample &sample : samples)
        {
            values.push_back(sample.value);
        }
        std::sort(values.begin(), values.end());
        const float median = values[values.size() / 2];
        for (float &value : values)
        {
            value = std::fabs(value - median);
        }
        std::sort(values.begin(), values.end());
        const float median_absolute_deviation = values[values.size() / 2];
        const float scale = std::max(
            10.0f, 2.5f * 1.4826f * median_absolute_deviation);
        for (const LumaSample &sample : samples)
        {
            const float normalized =
                (sample.value - median) / scale;
            const float consistency =
                std::exp(-0.5f * normalized * normalized);
            FaceCandidate &candidate =
                assignment->candidates[sample.candidateIndex];
            candidate.photometricConsistency *= consistency;
        }
    }

    std::vector<TextureCandidateQuality> qualities;
    qualities.reserve(static_cast<std::size_t>(assignment->candidates.size()));
    std::vector<float> resolutions;
    resolutions.reserve(static_cast<std::size_t>(assignment->candidates.size()));
    for (const FaceCandidate &candidate : assignment->candidates)
    {
        qualities.push_back({candidate.sharpness,
                             candidate.photometricConsistency,
                             candidate.projectedResolution,
                             candidate.angleScore});
        resolutions.push_back(candidate.projectedResolution);
    }
    std::sort(resolutions.begin(), resolutions.end(), std::greater<float>());
    const float face_weight = resolutions.size() >= 2
        ? resolutions[1]
        : resolutions.front();
    const std::vector<std::int32_t> unary_costs =
        buildTextureCandidateUnaryCosts(
            qualities,
            face_weight,
            TextureCandidateCostFlags{
                config.enableOutOfFocusFilter,
                config.enableGhostFilter,
                true,
                true});
    for (int index = 0; index < assignment->candidates.size(); ++index)
    {
        FaceCandidate &candidate = assignment->candidates[index];
        candidate.unaryCost = unary_costs[static_cast<std::size_t>(index)];
        candidate.score = 1000.0f /
            (1000.0f + static_cast<float>(candidate.unaryCost));
    }
    std::sort(assignment->candidates.begin(),
              assignment->candidates.end(),
              [](const FaceCandidate &left, const FaceCandidate &right)
    {
        return left.unaryCost < right.unaryCost ||
            (left.unaryCost == right.unaryCost &&
             left.viewIndex < right.viewIndex);
    });
}

bool optimizeCameraLabels(const TextureMappingConfig &config,
                          PipelineData *data,
                          TextureMappingResult *result,
                          std::string *error_msg)
{
    constexpr TextureLabelCost kNoCameraUnary = 10'000'000;
    const int no_camera_label = data->views.size();
    TextureLabelOptimizationProblem problem;
    problem.nodeCount = data->assignments.size();
    problem.labelCount = no_camera_label + 1;
    problem.maximumPasses = std::clamp(config.labelOptimizationPasses, 1, 4);
    problem.initialLabels.reserve(static_cast<std::size_t>(problem.nodeCount));
    for (const FaceAssignment &assignment : data->assignments)
    {
        problem.initialLabels.push_back(
            assignment.candidates.isEmpty()
            ? no_camera_label
            : assignment.candidates.front().viewIndex);
    }
    for (int face_index = 0; face_index < data->geometry.size(); ++face_index)
    {
        for (const int neighbor : data->geometry[face_index].neighbors)
        {
            if (neighbor <= face_index)
            {
                continue;
            }
            const double relative_edge_length = std::clamp(
                std::min(data->geometry[face_index].meanEdgeLength,
                         data->geometry[neighbor].meanEdgeLength) /
                    std::max(data->medianEdgeLength, 1.0e-12),
                0.25,
                4.0);
            const double pairwise_scale =
                config.labelSmoothness *
                (1000.0 + 250.0 * config.labelColorPenalty);
            problem.edges.push_back({
                face_index,
                neighbor,
                static_cast<TextureLabelCost>(std::lround(
                    pairwise_scale * relative_edge_length))});
        }
    }
    problem.unaryCost = [data, no_camera_label](int face_index, int label)
    {
        if (label == no_camera_label)
        {
            return kNoCameraUnary;
        }
        const FaceCandidate *candidate = candidateForView(
            data->assignments[face_index], label);
        return candidate
            ? static_cast<TextureLabelCost>(candidate->unaryCost)
            : kForbiddenTextureLabelCost;
    };
    problem.isCancelled = config.isCancelled;
    problem.progressFn = [&config, &problem](int pass, int label)
    {
        if (config.progressFn)
        {
            config.progressFn("正在执行相机标签图割，第 " + std::to_string(pass + 1) +
                                  " 轮，标签 " + std::to_string(label + 1) + "/" +
                                  std::to_string(problem.labelCount) + "...",
                              46 + static_cast<int>((static_cast<std::int64_t>(pass) * problem.labelCount + label) *
                                  9 / (static_cast<std::int64_t>(problem.maximumPasses) * problem.labelCount)));
        }
    };

    const TextureLabelOptimizationResult optimized =
        optimizeTextureLabels(problem);
    if (!optimized.solved)
    {
        if (optimized.cancelled)
        {
            result->cancelled = true;
        }
        if (error_msg)
        {
            *error_msg = optimized.cancelled
                ? "纹理映射已取消"
                : "Natural 纹理相机图割优化失败: " + optimized.error;
        }
        return false;
    }
    for (int face_index = 0; face_index < data->assignments.size(); ++face_index)
    {
        FaceAssignment &assignment = data->assignments[face_index];
        const int old_label =
            problem.initialLabels[static_cast<std::size_t>(face_index)];
        const int label =
            optimized.labels[static_cast<std::size_t>(face_index)];
        assignment.optimized = label != old_label;
        if (label == no_camera_label)
        {
            assignment.primaryView = -1;
            assignment.primaryScore = -1.0f;
            continue;
        }
        const FaceCandidate *candidate = candidateForView(assignment, label);
        if (!candidate)
        {
            if (error_msg)
            {
                *error_msg = "Natural 纹理图割选择了不可用的相机标签";
            }
            return false;
        }
        assignment.primaryView = label;
        assignment.primaryScore = candidate->score;
    }
    return true;
}

} // namespace

bool selectTextureViews(const TextureMappingConfig &config,
                        PipelineData *data,
                        TextureMappingResult *result,
                        std::string *errorMsg)
{
    if (!data || !result)
    {
        return false;
    }
    if (!buildFinalMeshVisibility(config, data, result, errorMsg))
    {
        return false;
    }
    if (!estimateOverlapExposureGains(config, data, result, errorMsg))
    {
        return false;
    }
    if (config.progressFn)
    {
        config.progressFn("正在评估三角面纹理候选...", 20);
    }
    const int maximum_candidates =
        std::clamp(config.maximumCandidateViews, 1, 16);
    for (int face_index = 0; face_index < data->geometry.size(); ++face_index)
    {
        if ((face_index % 2048 == 0) && cancelled(config))
        {
            result->cancelled = true;
            if (errorMsg)
            {
                *errorMsg = "纹理映射已取消";
            }
            return false;
        }
        FaceAssignment &assignment = data->assignments[face_index];
        for (int view_index = 0; view_index < data->views.size(); ++view_index)
        {
            FaceCandidate candidate = evaluateCandidate(
                data->views[view_index],
                view_index,
                face_index,
                data->geometry[face_index],
                config,
                data->medianEdgeLength,
                true,
                result);
            if (candidate.viewIndex >= 0 && candidate.score > 0.0f)
            {
                assignment.candidates.push_back(candidate);
            }
        }
        if (!assignment.candidates.isEmpty())
        {
            buildRecoveredUnaryCosts(
                *data,
                face_index,
                maximum_candidates,
                config,
                &assignment);
        }

        if (assignment.candidates.isEmpty() &&
            config.holeFillMode == TextureHoleFillMode::NeighborViewRecovery)
        {
            for (int view_index = 0; view_index < data->views.size(); ++view_index)
            {
                FaceCandidate candidate = evaluateCandidate(
                    data->views[view_index],
                    view_index,
                    face_index,
                    data->geometry[face_index],
                    config,
                    data->medianEdgeLength,
                    false,
                    result);
                if (candidate.viewIndex >= 0 && candidate.score > 0.0f)
                {
                    assignment.candidates.push_back(candidate);
                }
            }
            if (!assignment.candidates.isEmpty())
            {
                buildRecoveredUnaryCosts(
                    *data,
                    face_index,
                    maximum_candidates,
                    config,
                    &assignment);
            }
            assignment.relaxed = !assignment.candidates.isEmpty();
        }
        if (!assignment.candidates.isEmpty())
        {
            assignment.primaryView = assignment.candidates.front().viewIndex;
            assignment.primaryScore = assignment.candidates.front().score;
        }
    }

    if (config.progressFn)
    {
        config.progressFn("正在执行 Natural 相机标签图割...", 46);
    }
    if (!optimizeCameraLabels(config, data, result, errorMsg))
    {
        return false;
    }

    std::set<int> used_views;
    for (const FaceAssignment &assignment : data->assignments)
    {
        if (assignment.primaryView < 0)
        {
            ++result->unmappedFaceCount;
            continue;
        }
        used_views.insert(assignment.primaryView);
        ++result->mappedFaceCount;
        if (assignment.relaxed)
        {
            ++result->fallbackMappedFaceCount;
        }
        else
        {
            ++result->strictMappedFaceCount;
        }
        if (assignment.optimized)
        {
            ++result->coherenceAdjustedFaceCount;
        }
    }
    result->usedViewCount = static_cast<int>(used_views.size());
    if (result->mappedFaceCount == 0 && !config.keepUnmapped)
    {
        if (errorMsg)
        {
            *errorMsg = "纹理 v4 没有任何三角面通过相机可见性检查";
        }
        return false;
    }
    return true;
}

} // namespace xjw::mesh::texture_v4
