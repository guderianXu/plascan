#include "TextureMappingV4Internal.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>

namespace xjw::mesh::texture_v4
{

bool passesUnaryQualityFloor(const FaceAssignment &assignment,
                             const FaceCandidate &candidate,
                             float replacement_ratio)
{
    if (assignment.candidates.isEmpty())
    {
        return false;
    }
    return candidate.score >=
        assignment.candidates.front().score * replacement_ratio;
}

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
    for (const auto &weights : kSampleWeights)
    {
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
                world, pixel, camera_depth))
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
    if (area < 0.50)
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
    if (config.enableOutOfFocusFilter && view.sharpnessWeight < 0.35f)
    {
        ++result->rejectedResolutionCount;
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
    const float resolution_score = std::clamp(
        static_cast<float>(std::sqrt(area) / 8.0), 0.10f, 1.0f);
    const float score =
        view.qualityWeight *
        std::pow(depth_score, 2.0f) *
        std::pow(angle_score, 4.0f) *
        resolution_score *
        center_score *
        std::clamp(view.sharpnessWeight, 0.20f, 1.5f);
    return {view_index,
            score,
            angle_score,
            resolution_score,
            strict};
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
            world.data(), pixel, depth))
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
    *color = view.colorBgr.at<cv::Vec3b>(row, column);
    return true;
}

double seamColorPenalty(const PipelineData &data,
                        int face_index,
                        int view_index,
                        int neighbor,
                        int neighbor_view)
{
    std::array<double, 3> midpoint{};
    for (int axis = 0; axis < 3; ++axis)
    {
        midpoint[axis] =
            (data.geometry[face_index].centroid[axis] +
             data.geometry[neighbor].centroid[axis]) * 0.5;
    }
    cv::Vec3f first;
    cv::Vec3f second;
    if (!sampleProjectedColor(data.views[view_index], midpoint, &first) ||
        !sampleProjectedColor(data.views[neighbor_view], midpoint, &second))
    {
        return 1.0;
    }
    const cv::Vec3f difference = first - second;
    return std::clamp(
        static_cast<double>(std::sqrt(difference.dot(difference))) /
            (std::sqrt(3.0) * 255.0),
        0.0,
        1.0);
}

double assignmentEnergy(int face_index,
                        int view_index,
                        const PipelineData &data,
                        const TextureMappingConfig &config)
{
    const FaceCandidate *candidate =
        candidateForView(data.assignments[face_index], view_index);
    if (!candidate || candidate->score <= 0.0f)
    {
        return std::numeric_limits<double>::infinity();
    }
    double energy = -std::log(candidate->score + 1.0e-12f);
    const FaceGeometry &face = data.geometry[face_index];
    for (const int neighbor : face.neighbors)
    {
        if (neighbor < 0 || neighbor >= data.assignments.size())
        {
            continue;
        }
        const int neighbor_view = data.assignments[neighbor].primaryView;
        if (neighbor_view >= 0 && neighbor_view != view_index)
        {
            const double edge_weight = std::clamp(
                face.meanEdgeLength /
                    std::max(data.geometry[neighbor].meanEdgeLength, 1.0e-12),
                0.25,
                4.0);
            energy += config.labelSmoothness * edge_weight;
            energy += config.labelColorPenalty *
                seamColorPenalty(
                    data, face_index, view_index, neighbor, neighbor_view);
        }
    }
    return energy;
}

void mergeSmallLabelIslands(const TextureMappingConfig &config,
                            PipelineData *data)
{
    if (config.minimumChartFaces <= 1)
    {
        return;
    }
    QVector<bool> visited(data->assignments.size(), false);
    for (int seed = 0; seed < data->assignments.size(); ++seed)
    {
        if (visited[seed] || data->assignments[seed].primaryView < 0)
        {
            continue;
        }
        const int source_label = data->assignments[seed].primaryView;
        QVector<int> component;
        std::queue<int> pending;
        pending.push(seed);
        visited[seed] = true;
        while (!pending.empty())
        {
            const int face_index = pending.front();
            pending.pop();
            component.push_back(face_index);
            for (const int neighbor : data->geometry[face_index].neighbors)
            {
                if (neighbor >= 0 && neighbor < visited.size() &&
                    !visited[neighbor] &&
                    data->assignments[neighbor].primaryView == source_label)
                {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        if (component.size() >= config.minimumChartFaces)
        {
            continue;
        }

        std::map<int, int> boundary_votes;
        for (const int face_index : component)
        {
            for (const int neighbor : data->geometry[face_index].neighbors)
            {
                const int label = data->assignments[neighbor].primaryView;
                if (label >= 0 && label != source_label)
                {
                    ++boundary_votes[label];
                }
            }
        }
        if (boundary_votes.empty())
        {
            continue;
        }
        const auto target = std::max_element(
            boundary_votes.begin(),
            boundary_votes.end(),
            [](const auto &left, const auto &right)
            {
                return left.second < right.second ||
                    (left.second == right.second && left.first > right.first);
            });
        const int target_label = target->first;
        bool can_merge = true;
        for (const int face_index : component)
        {
            const FaceAssignment &assignment = data->assignments[face_index];
            const FaceCandidate *candidate =
                candidateForView(assignment, target_label);
            if (!candidate || !passesUnaryQualityFloor(
                    assignment,
                    *candidate,
                    config.coherentReplacementRatio))
            {
                can_merge = false;
                break;
            }
        }
        if (!can_merge)
        {
            continue;
        }
        for (const int face_index : component)
        {
            FaceAssignment &assignment = data->assignments[face_index];
            assignment.primaryView = target_label;
            assignment.primaryScore =
                candidateForView(assignment, target_label)->score;
            assignment.optimized = true;
        }
    }
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
        std::sort(assignment.candidates.begin(),
                  assignment.candidates.end(),
                  [](const auto &left, const auto &right)
        {
            return left.score > right.score ||
                (left.score == right.score && left.viewIndex < right.viewIndex);
        });
        if (assignment.candidates.size() > maximum_candidates)
        {
            assignment.candidates.resize(maximum_candidates);
        }

        if (assignment.candidates.isEmpty() &&
            config.holeFillMode == TextureHoleFillMode::NeighborViewRecovery)
        {
            for (int view_index = 0; view_index < data->views.size(); ++view_index)
            {
                FaceCandidate candidate = evaluateCandidate(
                    data->views[view_index],
                    view_index,
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
            std::sort(assignment.candidates.begin(),
                      assignment.candidates.end(),
                      [](const auto &left, const auto &right)
            {
                return left.score > right.score ||
                    (left.score == right.score && left.viewIndex < right.viewIndex);
            });
            if (assignment.candidates.size() > maximum_candidates)
            {
                assignment.candidates.resize(maximum_candidates);
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
        config.progressFn("正在优化纹理相机连续性...", 46);
    }
    for (int pass = 0; pass < config.labelOptimizationPasses; ++pass)
    {
        int changed = 0;
        for (int face_index = 0; face_index < data->assignments.size(); ++face_index)
        {
            FaceAssignment &assignment = data->assignments[face_index];
            if (assignment.primaryView < 0)
            {
                continue;
            }
            std::set<int> labels;
            for (const FaceCandidate &candidate : assignment.candidates)
            {
                labels.insert(candidate.viewIndex);
            }
            for (const int neighbor : data->geometry[face_index].neighbors)
            {
                const int label = data->assignments[neighbor].primaryView;
                if (candidateForView(assignment, label))
                {
                    labels.insert(label);
                }
            }

            const int old_label = assignment.primaryView;
            const double old_energy = assignmentEnergy(
                face_index,
                old_label,
                *data,
                config);
            double best_energy = old_energy;
            int best_label = old_label;
            for (const int label : labels)
            {
                const FaceCandidate *candidate = candidateForView(assignment, label);
                if (!candidate || !passesUnaryQualityFloor(
                        assignment,
                        *candidate,
                        config.coherentReplacementRatio))
                {
                    continue;
                }
                const double energy = assignmentEnergy(
                    face_index,
                    label,
                    *data,
                    config);
                if (energy + 1.0e-9 < best_energy)
                {
                    best_energy = energy;
                    best_label = label;
                }
            }
            if (best_label != old_label)
            {
                assignment.primaryView = best_label;
                assignment.primaryScore =
                    candidateForView(assignment, best_label)->score;
                assignment.optimized = true;
                ++changed;
            }
        }
        if (changed == 0 ||
            changed < std::max(
                1, static_cast<int>(data->assignments.size() / 1000)))
        {
            break;
        }
    }
    mergeSmallLabelIslands(config, data);

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
