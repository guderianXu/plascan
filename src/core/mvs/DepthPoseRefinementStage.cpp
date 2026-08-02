#include "DepthPoseRefinementStage.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>

namespace xjw::mvs
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

bool hasCompleteEvidence(const DepthPoseRefinementFrame &frame)
{
    const cv::Size size = frame.depthMap.size();
    const bool normal_source_valid = frame.normalMap.empty() ||
        (frame.normalMap.type() == CV_32FC3 &&
         frame.normalMap.size() == size);
    return frame.camera.isValid() &&
        !frame.depthMap.empty() && frame.depthMap.type() == CV_32FC1 &&
        normal_source_valid &&
        !frame.adaptiveSupportWeight.empty() &&
        frame.adaptiveSupportWeight.type() == CV_32FC1 &&
        frame.adaptiveSupportWeight.size() == size &&
        !frame.adaptiveEffectiveViewCount.empty() &&
        frame.adaptiveEffectiveViewCount.type() == CV_32FC1 &&
        frame.adaptiveEffectiveViewCount.size() == size &&
        !frame.adaptiveConflictRatio.empty() &&
        frame.adaptiveConflictRatio.type() == CV_32FC1 &&
        frame.adaptiveConflictRatio.size() == size;
}

bool unproject(const Camera &camera,
               int column,
               int row,
               float depth,
               cv::Vec3d *world)
{
    if (!world || !std::isfinite(depth) || depth <= 0.0f)
    {
        return false;
    }
    const double pixel[2] = {
        static_cast<double>(column),
        static_cast<double>(row)};
    double point[3] = {0.0, 0.0, 0.0};
    if (!camera.unprojectPixel(pixel, static_cast<double>(depth), point))
    {
        return false;
    }
    *world = cv::Vec3d(point[0], point[1], point[2]);
    return cv::checkRange(*world);
}

cv::Matx33d cameraToWorldRotation(const Camera &camera)
{
    const std::array<double, 9> values = camera.cameraToWorldRotation();
    return cv::Matx33d(
        values[0], values[1], values[2],
        values[3], values[4], values[5],
        values[6], values[7], values[8]);
}

bool surfaceNormalWorldAt(const DepthPoseRefinementFrame &frame,
                          int row,
                          int column,
                          cv::Vec3d *normal_world)
{
    if (!normal_world)
    {
        return false;
    }

    if (!frame.normalMap.empty())
    {
        const cv::Vec3f normal_camera =
            frame.normalMap.at<cv::Vec3f>(row, column);
        cv::Vec3d candidate = cameraToWorldRotation(frame.camera) *
            cv::Vec3d(normal_camera[0], normal_camera[1], normal_camera[2]);
        const double length = cv::norm(candidate);
        if (std::isfinite(length) && length > 1.0e-8)
        {
            *normal_world = candidate * (1.0 / length);
            return true;
        }
    }

    // The current adaptive PatchMatch wrapper does not expose its internal
    // plane normals. Reconstruct the local tangent plane deterministically
    // from the completed depth instead of declaring otherwise complete
    // cross-view evidence unusable.
    if (row <= 0 || row + 1 >= frame.depthMap.rows ||
        column <= 0 || column + 1 >= frame.depthMap.cols)
    {
        return false;
    }
    cv::Vec3d left;
    cv::Vec3d right;
    cv::Vec3d upper;
    cv::Vec3d lower;
    if (!unproject(frame.camera,
                   column - 1,
                   row,
                   frame.depthMap.at<float>(row, column - 1),
                   &left) ||
        !unproject(frame.camera,
                   column + 1,
                   row,
                   frame.depthMap.at<float>(row, column + 1),
                   &right) ||
        !unproject(frame.camera,
                   column,
                   row - 1,
                   frame.depthMap.at<float>(row - 1, column),
                   &upper) ||
        !unproject(frame.camera,
                   column,
                   row + 1,
                   frame.depthMap.at<float>(row + 1, column),
                   &lower))
    {
        return false;
    }
    cv::Vec3d candidate = (right - left).cross(lower - upper);
    const double length = cv::norm(candidate);
    if (!std::isfinite(length) || length <= 1.0e-8)
    {
        return false;
    }
    *normal_world = candidate * (1.0 / length);
    return true;
}

bool projectToFrame(const cv::Vec3d &world,
                    const DepthPoseRefinementFrame &frame,
                    int *column,
                    int *row,
                    double *projectedDepth)
{
    const double point[3] = {world[0], world[1], world[2]};
    double pixel[2] = {0.0, 0.0};
    double depth = 0.0;
    if (!frame.camera.projectWorldPointWithDepth(point, pixel, depth) ||
        !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
        !std::isfinite(depth) || depth <= 0.0)
    {
        return false;
    }
    const int projected_column = static_cast<int>(std::llround(pixel[0]));
    const int projected_row = static_cast<int>(std::llround(pixel[1]));
    if (projected_column < 0 || projected_row < 0 ||
        projected_column >= frame.depthMap.cols ||
        projected_row >= frame.depthMap.rows)
    {
        return false;
    }
    if (column)
    {
        *column = projected_column;
    }
    if (row)
    {
        *row = projected_row;
    }
    if (projectedDepth)
    {
        *projectedDepth = depth;
    }
    return true;
}

bool evidencePasses(const DepthPoseRefinementFrame &frame,
                    int row,
                    int column,
                    const DepthPoseRefinementOptions &options)
{
    const float support = frame.adaptiveSupportWeight.at<float>(row, column);
    const float effective_views =
        frame.adaptiveEffectiveViewCount.at<float>(row, column);
    const float conflict = frame.adaptiveConflictRatio.at<float>(row, column);
    return std::isfinite(support) && std::isfinite(effective_views) &&
        std::isfinite(conflict) &&
        support >= options.minimumAdaptiveSupportWeight &&
        effective_views >= options.minimumAdaptiveEffectiveViewCount &&
        conflict <= options.maximumAdaptiveConflictRatio;
}

double confidenceAt(const DepthPoseRefinementFrame &frame,
                    int row,
                    int column)
{
    if (frame.confidence.empty() || frame.confidence.type() != CV_32FC1 ||
        frame.confidence.size() != frame.depthMap.size())
    {
        return 1.0;
    }
    const float value = frame.confidence.at<float>(row, column);
    return std::isfinite(value)
        ? std::clamp(static_cast<double>(value), 0.0, 1.0)
        : 0.0;
}

double rotationDegrees(const cv::Matx33d &rotation)
{
    const double cosine = std::clamp(
        (cv::trace(rotation) - 1.0) * 0.5,
        -1.0,
        1.0);
    return std::acos(cosine) * 180.0 / kPi;
}

std::vector<int> deterministicSources(
    const DepthPoseRefinementFrame &frame,
    int maximum_source_count)
{
    std::vector<int> sources;
    sources.reserve(frame.sourceCameraIndices.size());
    for (const int source : frame.sourceCameraIndices)
    {
        if (source == frame.cameraIndex ||
            std::find(sources.begin(), sources.end(), source) != sources.end())
        {
            continue;
        }
        sources.push_back(source);
        if (static_cast<int>(sources.size()) >= maximum_source_count)
        {
            break;
        }
    }
    return sources;
}

double correctedProjectionRetention(
    const DepthPoseAlignmentCorrection &correction,
    const std::vector<DepthPoseAlignmentSample> &samples,
    const std::unordered_map<int, const DepthPoseRefinementFrame *>
        &frame_by_index,
    double maximum_relative_depth_error)
{
    int sample_count = 0;
    int retained_count = 0;
    for (const DepthPoseAlignmentSample &sample : samples)
    {
        if (sample.cameraIndex != correction.cameraIndex)
        {
            continue;
        }
        ++sample_count;
        const auto target = frame_by_index.find(sample.targetCameraIndex);
        if (target == frame_by_index.end())
        {
            continue;
        }
        const cv::Vec3d corrected =
            DepthPoseAlignmentRefiner::applyCorrection(correction,
                                                        sample.sourcePointWorld);
        int column = -1;
        int row = -1;
        double projected_depth = 0.0;
        if (!projectToFrame(corrected,
                            *target->second,
                            &column,
                            &row,
                            &projected_depth))
        {
            continue;
        }
        const float target_depth = target->second->depthMap.at<float>(row, column);
        if (!std::isfinite(target_depth) || target_depth <= 0.0f)
        {
            continue;
        }
        const double relative_error = std::abs(
            projected_depth - static_cast<double>(target_depth)) /
            std::max(projected_depth, static_cast<double>(target_depth));
        if (relative_error <= maximum_relative_depth_error)
        {
            ++retained_count;
        }
    }
    return sample_count > 0
        ? static_cast<double>(retained_count) /
            static_cast<double>(sample_count)
        : 0.0;
}

} // namespace

DepthPoseRefinementStageResult DepthPoseRefinementStage::buildCandidates(
    const std::vector<DepthPoseRefinementFrame> &frames,
    const DepthPoseRefinementOptions &options)
{
    DepthPoseRefinementStageResult result;
    result.enabled = options.enabled;
    result.anchorCameraIndex = options.optimizer.anchorCameraIndex;
    if (!options.enabled)
    {
        return result;
    }

    std::unordered_map<int, const DepthPoseRefinementFrame *> frame_by_index;
    for (const DepthPoseRefinementFrame &frame : frames)
    {
        frame_by_index.emplace(frame.cameraIndex, &frame);
    }

    std::vector<DepthPoseAlignmentSample> samples;
    std::map<int, int> evidence_pixel_counts;
    std::map<int, int> occluded_counts;
    std::map<int, int> depth_conflict_counts;
    const int stride = std::max(1, options.samplingStridePixels);
    const int maximum_samples = std::max(1, options.maximumSamplesPerCamera);
    for (const DepthPoseRefinementFrame &frame : frames)
    {
        if (!hasCompleteEvidence(frame))
        {
            continue;
        }
        const std::vector<int> sources = deterministicSources(
            frame,
            std::max(1, options.maximumSourceFramesPerCamera));
        int generated_for_camera = 0;
        for (int row = stride / 2;
             row < frame.depthMap.rows && generated_for_camera < maximum_samples;
             row += stride)
        {
            for (int column = stride / 2;
                 column < frame.depthMap.cols && generated_for_camera < maximum_samples;
                 column += stride)
            {
                const float depth = frame.depthMap.at<float>(row, column);
                if (!std::isfinite(depth) || depth <= 0.0f ||
                    !evidencePasses(frame, row, column, options))
                {
                    continue;
                }
                ++evidence_pixel_counts[frame.cameraIndex];
                cv::Vec3d source_world;
                if (!unproject(frame.camera, column, row, depth, &source_world))
                {
                    continue;
                }
                for (const int source_index : sources)
                {
                    if (generated_for_camera >= maximum_samples)
                    {
                        break;
                    }
                    const auto target_iterator = frame_by_index.find(source_index);
                    if (target_iterator == frame_by_index.end() ||
                        !hasCompleteEvidence(*target_iterator->second))
                    {
                        continue;
                    }
                    const DepthPoseRefinementFrame &target =
                        *target_iterator->second;
                    int target_column = -1;
                    int target_row = -1;
                    double projected_depth = 0.0;
                    if (!projectToFrame(source_world,
                                        target,
                                        &target_column,
                                        &target_row,
                                        &projected_depth) ||
                        !evidencePasses(
                            target, target_row, target_column, options))
                    {
                        continue;
                    }
                    const float target_depth =
                        target.depthMap.at<float>(target_row, target_column);
                    if (!std::isfinite(target_depth) || target_depth <= 0.0f)
                    {
                        continue;
                    }
                    const double occlusion_tolerance =
                        options.occlusionRelativeDepthTolerance *
                        std::max(projected_depth,
                                 static_cast<double>(target_depth));
                    if (static_cast<double>(target_depth) + occlusion_tolerance <
                        projected_depth)
                    {
                        ++occluded_counts[frame.cameraIndex];
                        continue;
                    }
                    const double relative_depth_error = std::abs(
                        projected_depth - static_cast<double>(target_depth)) /
                        std::max(projected_depth,
                                 static_cast<double>(target_depth));
                    if (relative_depth_error >
                        options.maximumCorrespondenceRelativeDepthError)
                    {
                        ++depth_conflict_counts[frame.cameraIndex];
                        continue;
                    }
                    cv::Vec3d target_world;
                    if (!unproject(target.camera,
                                   target_column,
                                   target_row,
                                   target_depth,
                                   &target_world))
                    {
                        continue;
                    }
                    cv::Vec3d target_normal_world;
                    if (!surfaceNormalWorldAt(target,
                                              target_row,
                                              target_column,
                                              &target_normal_world))
                    {
                        continue;
                    }
                    const double support = std::min(
                        static_cast<double>(
                            frame.adaptiveSupportWeight.at<float>(row, column)),
                        static_cast<double>(target.adaptiveSupportWeight.at<float>(
                            target_row, target_column)));
                    const double conflict = std::max(
                        static_cast<double>(
                            frame.adaptiveConflictRatio.at<float>(row, column)),
                        static_cast<double>(target.adaptiveConflictRatio.at<float>(
                            target_row, target_column)));
                    DepthPoseAlignmentSample sample;
                    sample.cameraIndex = frame.cameraIndex;
                    sample.targetCameraIndex = target.cameraIndex;
                    sample.sourcePointWorld = source_world;
                    sample.targetPointWorld = target_world;
                    sample.targetNormalWorld = target_normal_world;
                    sample.confidence = std::min(
                        confidenceAt(frame, row, column),
                        confidenceAt(target, target_row, target_column)) *
                        std::max(0.0, support) *
                        std::clamp(1.0 - conflict, 0.0, 1.0);
                    samples.push_back(sample);
                    ++generated_for_camera;
                }
            }
        }
    }

    DepthPoseAlignmentOptions optimizer = options.optimizer;
    optimizer.enabled = true;
    const DepthPoseAlignmentResult refined =
        DepthPoseAlignmentRefiner::refine(samples, optimizer);
    std::map<int, DepthPoseAlignmentCorrection> corrections;
    for (const DepthPoseAlignmentCorrection &correction : refined.corrections)
    {
        corrections.emplace(correction.cameraIndex, correction);
    }

    result.candidates.reserve(frames.size());
    for (const DepthPoseRefinementFrame &frame : frames)
    {
        DepthPoseRefinementCandidate candidate;
        candidate.cameraIndex = frame.cameraIndex;
        candidate.evidenceComplete = hasCompleteEvidence(frame);
        candidate.evidencePixelCount = evidence_pixel_counts[frame.cameraIndex];
        candidate.occludedCandidateCount = occluded_counts[frame.cameraIndex];
        candidate.depthConflictCandidateCount =
            depth_conflict_counts[frame.cameraIndex];
        const auto correction_iterator = corrections.find(frame.cameraIndex);
        if (!candidate.evidenceComplete)
        {
            candidate.reason = "incomplete_geometry_evidence";
        }
        else if (correction_iterator == corrections.end())
        {
            candidate.reason = "no_usable_correspondences";
        }
        else
        {
            candidate.correction = correction_iterator->second;
            candidate.generatedCorrespondenceCount =
                candidate.correction.correspondenceCount;
            candidate.evidenceSampleCoverage = candidate.evidencePixelCount > 0
                ? std::min(
                    1.0,
                    static_cast<double>(candidate.generatedCorrespondenceCount) /
                        static_cast<double>(candidate.evidencePixelCount))
                : 0.0;
            candidate.correctionTranslation = cv::norm(
                candidate.correction.translation);
            candidate.correctionRotationDegrees = rotationDegrees(
                candidate.correction.rotation);
            if (!candidate.correction.accepted)
            {
                candidate.reason = candidate.correction.reason;
            }
            else
            {
                candidate.projectionRetentionRatio =
                    correctedProjectionRetention(
                        candidate.correction,
                        samples,
                        frame_by_index,
                        options.maximumCorrespondenceRelativeDepthError);
                if (candidate.evidenceSampleCoverage <
                    options.minimumEvidenceSampleCoverage)
                {
                    candidate.reason = "insufficient_evidence_sample_coverage";
                }
                else if (candidate.projectionRetentionRatio <
                    options.minimumProjectionRetentionRatio)
                {
                    candidate.reason = "projection_coverage_regressed";
                }
                else
                {
                    candidate.accepted = true;
                    candidate.reason = "accepted_candidate";
                    if (options.emitDerivedCameraCandidates)
                    {
                        candidate.derivedCamera = deriveCameraCandidate(
                            frame.camera,
                            candidate.correction);
                    }
                }
            }
        }
        candidate.correction.accepted = candidate.accepted;
        candidate.correction.reason = candidate.reason;
        result.acceptedAny = result.acceptedAny || candidate.accepted;
        result.candidates.push_back(std::move(candidate));
    }
    return result;
}

Camera DepthPoseRefinementStage::deriveCameraCandidate(
    const Camera &camera,
    const DepthPoseAlignmentCorrection &correction)
{
    if (!camera.isValid() || !correction.accepted)
    {
        return camera;
    }
    const cv::Matx33d old_camera_to_world = cameraToWorldRotation(camera);
    // The requested world-to-camera update is R_wc' = R_wc * Q^T.
    // Camera stores R_cw, so the equivalent update is R_cw' = Q * R_cw.
    const cv::Matx33d new_camera_to_world =
        correction.rotation * old_camera_to_world;
    const std::array<double, 3> old_center_array = camera.cameraCenter();
    const cv::Vec3d old_center(
        old_center_array[0], old_center_array[1], old_center_array[2]);
    const cv::Vec3d new_center =
        correction.rotation * (old_center - correction.pivotWorld) +
        correction.pivotWorld + correction.translation;
    Camera derived = camera;
    derived.setPose(
        std::array<double, 9>{
            new_camera_to_world(0, 0), new_camera_to_world(0, 1),
            new_camera_to_world(0, 2), new_camera_to_world(1, 0),
            new_camera_to_world(1, 1), new_camera_to_world(1, 2),
            new_camera_to_world(2, 0), new_camera_to_world(2, 1),
            new_camera_to_world(2, 2)},
        std::array<double, 3>{
            new_center[0], new_center[1], new_center[2]});
    return derived;
}

} // namespace xjw::mvs
