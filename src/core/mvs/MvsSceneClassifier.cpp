#include "MvsSceneClassifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

namespace xjw
{
namespace mvs
{
namespace
{

using Vec3 = cv::Vec3d;

Vec3 normalize(const Vec3 &value)
{
    const double length = cv::norm(value);
    return length > 1e-12 ? value / length : Vec3(0.0, 0.0, 0.0);
}

bool covarianceEigen(const std::vector<Vec3> &values,
                     cv::Vec3d &eigenvalues,
                     cv::Matx33d &eigenvectors)
{
    if (values.size() < 3)
    {
        return false;
    }

    Vec3 mean(0.0, 0.0, 0.0);
    for (const Vec3 &value : values)
    {
        mean += value;
    }
    mean *= 1.0 / static_cast<double>(values.size());

    cv::Matx33d covariance = cv::Matx33d::zeros();
    for (const Vec3 &value : values)
    {
        const Vec3 delta = value - mean;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                covariance(row, column) += delta[row] * delta[column];
            }
        }
    }
    covariance *= 1.0 / static_cast<double>(values.size() - 1);

    cv::Mat eigenvalue_matrix;
    cv::Mat eigenvector_matrix;
    if (!cv::eigen(cv::Mat(covariance), eigenvalue_matrix, eigenvector_matrix))
    {
        return false;
    }
    for (int index = 0; index < 3; ++index)
    {
        eigenvalues[index] = eigenvalue_matrix.at<double>(index, 0);
        for (int column = 0; column < 3; ++column)
        {
            eigenvectors(index, column) = eigenvector_matrix.at<double>(index, column);
        }
    }
    return true;
}

Vec3 meanPoint(const std::vector<Vec3> &points)
{
    Vec3 result(0.0, 0.0, 0.0);
    for (const Vec3 &point : points)
    {
        result += point;
    }
    if (!points.empty())
    {
        result *= 1.0 / static_cast<double>(points.size());
    }
    return result;
}

} // namespace

MvsSceneClassification classifyMvsScene(const std::vector<CameraView> &views,
                                        const SparseCloud &sparse_cloud)
{
    MvsSceneClassification result;
    if (views.size() < 3 || sparse_cloud.points.size() < 6)
    {
        result.reason = "insufficient cameras or sparse points; using orbital profile";
        return result;
    }

    std::vector<Vec3> cloud_points;
    cloud_points.reserve(sparse_cloud.points.size());
    for (const std::array<float, 3> &point : sparse_cloud.points)
    {
        if (std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]))
        {
            cloud_points.emplace_back(point[0], point[1], point[2]);
        }
    }
    if (cloud_points.size() < 6)
    {
        result.reason = "insufficient finite sparse points; using orbital profile";
        return result;
    }

    cv::Vec3d cloud_eigenvalues;
    cv::Matx33d cloud_eigenvectors;
    if (!covarianceEigen(cloud_points, cloud_eigenvalues, cloud_eigenvectors))
    {
        result.reason = "sparse cloud PCA failed; using orbital profile";
        return result;
    }
    const double largest_cloud_scale = std::sqrt(std::max(0.0, cloud_eigenvalues[0]));
    const double thickness = std::sqrt(std::max(0.0, cloud_eigenvalues[2]));
    result.planeThicknessRatio = largest_cloud_scale > 1e-9
        ? static_cast<float>(thickness / largest_cloud_scale)
        : 1.0f;
    const Vec3 plane_normal = normalize(Vec3(cloud_eigenvectors(2, 0),
                                             cloud_eigenvectors(2, 1),
                                             cloud_eigenvectors(2, 2)));
    const Vec3 cloud_center = meanPoint(cloud_points);

    std::vector<Vec3> camera_centers;
    camera_centers.reserve(views.size());
    double convergence_sum = 0.0;
    int convergent_count = 0;
    int down_looking_count = 0;
    int positive_plane_side_count = 0;
    int negative_plane_side_count = 0;
    int valid_camera_count = 0;
    constexpr double kMinimumAlignment = 0.8191520443;
    const double plane_side_epsilon = std::max(1e-9, largest_cloud_scale * 1e-6);

    for (const CameraView &view : views)
    {
        const FramePinholeCamera camera = view.camera.normalizedForPositiveDepth();
        if (!camera.isValid())
        {
            continue;
        }
        const std::array<double, 3> camera_center = camera.cameraCenter();
        const std::array<double, 9> rotation = camera.worldToCameraRotation();
        const Vec3 center(camera_center[0], camera_center[1], camera_center[2]);
        const Vec3 optical_axis = normalize(Vec3(rotation[6], rotation[7], rotation[8]));
        const Vec3 direction_to_cloud = normalize(cloud_center - center);
        const double convergence = optical_axis.dot(direction_to_cloud);
        const double signed_plane_distance = plane_normal.dot(center - cloud_center);
        const double plane_axis_alignment = optical_axis.dot(plane_normal);
        const double forward_plane_distance = std::abs(plane_axis_alignment) > 1e-12
            ? -signed_plane_distance / plane_axis_alignment
            : -1.0;

        camera_centers.push_back(center);
        convergence_sum += std::max(-1.0, std::min(1.0, convergence));
        ++valid_camera_count;
        if (signed_plane_distance > plane_side_epsilon)
        {
            ++positive_plane_side_count;
        }
        else if (signed_plane_distance < -plane_side_epsilon)
        {
            ++negative_plane_side_count;
        }
        if (convergence >= kMinimumAlignment)
        {
            ++convergent_count;
        }
        // Aerial cameras need to look approximately along the terrain normal
        // and meet the fitted terrain plane in front of the camera. Do not
        // require the optical axis to point at the global sparse-cloud centre:
        // that assumption rejects valid cameras near the ends of long strips.
        if (std::abs(plane_axis_alignment) >= kMinimumAlignment &&
            forward_plane_distance > plane_side_epsilon)
        {
            ++down_looking_count;
        }
    }

    if (valid_camera_count == 0)
    {
        result.reason = "no valid camera poses; using orbital profile";
        return result;
    }
    result.opticalAxisConvergence = static_cast<float>(convergence_sum / valid_camera_count);
    result.downLookingConsistency = static_cast<float>(down_looking_count) / valid_camera_count;
    const int off_plane_camera_count = positive_plane_side_count + negative_plane_side_count;
    const float camera_side_consistency = off_plane_camera_count > 0
        ? static_cast<float>(std::max(positive_plane_side_count, negative_plane_side_count)) /
            off_plane_camera_count
        : 0.0f;

    cv::Vec3d camera_eigenvalues;
    cv::Matx33d camera_eigenvectors;
    if (covarianceEigen(camera_centers, camera_eigenvalues, camera_eigenvectors) &&
        camera_eigenvalues[0] > 1e-12)
    {
        result.cameraCenterLinearity = static_cast<float>(
            1.0 - std::max(0.0, camera_eigenvalues[1]) / camera_eigenvalues[0]);
    }

    const bool aerial = result.downLookingConsistency >= 0.75f &&
                        camera_side_consistency >= 0.75f &&
                        result.planeThicknessRatio <= 0.20f;
    result.profile = aerial ? MvsSceneProfile::AerialTerrain : MvsSceneProfile::OrbitalObject;

    std::ostringstream message;
    message << (aerial ? "aerial terrain" : "orbital object")
            << ": down-looking=" << result.downLookingConsistency
            << ", convergence=" << result.opticalAxisConvergence
            << ", plane-thickness=" << result.planeThicknessRatio
            << ", camera-side=" << camera_side_consistency
            << ", convergent-cameras=" << convergent_count << '/' << valid_camera_count;
    result.reason = message.str();
    return result;
}

int recommendedMvsSourceViewCount(MvsSceneProfile scene_profile,
                                  int downsample_factor,
                                  int configured_count,
                                  int view_count)
{
    if (view_count <= 1)
    {
        return 0;
    }

    const bool high_quality = std::max(1, downsample_factor) <= 2;
    int scene_target = 4;
    if (scene_profile == MvsSceneProfile::AerialTerrain)
    {
        scene_target = high_quality ? 8 : 6;
    }
    else if (scene_profile == MvsSceneProfile::OrbitalObject)
    {
        // A six-source ring is useful for dense 16-view object captures, but it
        // is destructive for sparser rings: on a 12-view sequence the +/-3
        // cameras are already close to a 90-degree baseline. Requiring those
        // views to confirm every high-resolution pixel removes valid grazing
        // surfaces around the largest angular gap. Keep the four-view plan
        // unless the ring has enough cameras to supply six nearby views.
        scene_target = high_quality && view_count >= 16 ? 6 : 4;
        return std::min(scene_target, view_count - 1);
    }

    const int requested_count = std::max(std::max(1, configured_count), scene_target);
    return std::min(requested_count, view_count - 1);
}

float recommendedMvsSourceMaximumAngleDeg(MvsSceneProfile sceneProfile,
                                          int requested_source_count)
{
    if (sceneProfile != MvsSceneProfile::OrbitalObject)
    {
        return 35.0f;
    }

    // Dense object rings such as Middlebury Dino have about 22.5 degrees
    // between adjacent views. A 47-degree gate can never satisfy a six-view
    // high-quality request because it only admits the +/-1 and +/-2 cameras.
    // Keep the conservative gate for four-view jobs, and admit the +/-3 pair
    // for high-quality six-source jobs.
    return requested_source_count >= 6 ? 70.0f : 47.0f;
}

float adaptiveMvsSourceMaximumAngleDeg(
    MvsSceneProfile sceneProfile,
    int requested_source_count,
    const std::vector<float> &candidate_angles_degrees)
{
    const float configured_maximum =
        recommendedMvsSourceMaximumAngleDeg(
            sceneProfile, requested_source_count);
    if (sceneProfile != MvsSceneProfile::OrbitalObject ||
        requested_source_count <= 0)
    {
        return configured_maximum;
    }

    std::vector<float> valid_angles;
    valid_angles.reserve(candidate_angles_degrees.size());
    for (const float angle : candidate_angles_degrees)
    {
        if (std::isfinite(angle) && angle > 0.0f)
        {
            valid_angles.push_back(angle);
        }
    }
    if (valid_angles.size() <
        static_cast<std::size_t>(requested_source_count))
    {
        return configured_maximum;
    }

    std::sort(valid_angles.begin(), valid_angles.end());
    const float requested_angle =
        valid_angles[static_cast<std::size_t>(requested_source_count - 1)];
    const float sampling_margin_angle = requested_angle * 1.05f;
    const std::size_t dense_candidate_threshold =
        static_cast<std::size_t>(requested_source_count) * 4;
    if (valid_angles.size() >= dense_candidate_threshold &&
        requested_angle < configured_maximum * 0.60f)
    {
        // Hundreds of closely sampled orbital frames are not equivalent to a
        // sparse Dino-style ring. Keeping the sparse-ring 70--90 degree gate
        // lets high-track but appearance-incompatible views outrank the local
        // angular neighbours and makes PatchMatch hypotheses mutually
        // contradictory. Use the local sampling density when the candidate
        // pool can comfortably supply the full source set nearby.
        return std::clamp(
            std::max(requested_angle * 1.25f, requested_angle + 2.0f),
            12.0f,
            configured_maximum);
    }
    const float safety_cap =
        requested_source_count >= 6 ? 90.0f : 70.0f;
    return std::clamp(
        std::max(configured_maximum, sampling_margin_angle),
        configured_maximum,
        safety_cap);
}

} // namespace mvs
} // namespace xjw
