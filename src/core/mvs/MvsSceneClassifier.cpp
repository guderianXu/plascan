#include "MvsSceneClassifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace xjw
{
namespace mvs
{
namespace
{

using Vec3 = cv::Vec3d;

constexpr double kRadiansToDegrees = 57.295779513082320876;
constexpr double kMinimumAlignment = 0.8191520443;
// Orbital-only recovery assumes a near-planar, well-covered camera ring whose
// optical axes converge on the reconstructed object. These gates deliberately
// classify ambiguous free-form captures as Custom instead of guessing Orbital.
constexpr double kMinimumRingInPlaneBalance = 0.25;
constexpr double kMaximumRingNonPlanarity = 0.20;
constexpr double kMaximumRingRadiusMadRatio = 0.30;
constexpr double kMaximumRingCenterOffsetRatio = 0.50;
constexpr double kMaximumRingAngularGapDegrees = 120.001;
constexpr double kMaximumRingMedianAxisErrorDegrees = 30.0;
constexpr double kMaximumRingP90AxisErrorDegrees = 45.0;

bool finiteVector(const Vec3 &value)
{
    return std::isfinite(value[0]) &&
           std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

bool vectorLexicographicLess(const Vec3 &left, const Vec3 &right)
{
    if (left[0] != right[0])
    {
        return left[0] < right[0];
    }
    if (left[1] != right[1])
    {
        return left[1] < right[1];
    }
    return left[2] < right[2];
}

void sortVectors(std::vector<Vec3> &values)
{
    std::sort(values.begin(), values.end(), vectorLexicographicLess);
}

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

    std::vector<Vec3> ordered_values = values;
    sortVectors(ordered_values);

    Vec3 mean(0.0, 0.0, 0.0);
    for (const Vec3 &value : ordered_values)
    {
        mean += value;
    }
    mean *= 1.0 / static_cast<double>(values.size());

    cv::Matx33d covariance = cv::Matx33d::zeros();
    for (const Vec3 &value : ordered_values)
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
    std::vector<Vec3> ordered_points = points;
    sortVectors(ordered_points);

    Vec3 result(0.0, 0.0, 0.0);
    for (const Vec3 &point : ordered_points)
    {
        result += point;
    }
    if (!points.empty())
    {
        result *= 1.0 / static_cast<double>(points.size());
    }
    return result;
}

double quantile(std::vector<double> values, double probability)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double position = std::clamp(probability, 0.0, 1.0) *
                            static_cast<double>(values.size() - 1);
    const std::size_t lower_index = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper_index = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower_index);
    return values[lower_index] * (1.0 - fraction) +
           values[upper_index] * fraction;
}

int distinctPointCount(const std::vector<Vec3> &points, double tolerance)
{
    std::vector<int> parents(points.size());
    std::iota(parents.begin(), parents.end(), 0);
    auto root = [&parents](int index)
    {
        while (parents[static_cast<std::size_t>(index)] != index)
        {
            index = parents[static_cast<std::size_t>(index)];
        }
        return index;
    };
    for (std::size_t left = 0; left < points.size(); ++left)
    {
        for (std::size_t right = left + 1; right < points.size(); ++right)
        {
            if (cv::norm(points[left] - points[right]) > tolerance)
            {
                continue;
            }
            const int left_root = root(static_cast<int>(left));
            const int right_root = root(static_cast<int>(right));
            if (left_root != right_root)
            {
                parents[static_cast<std::size_t>(right_root)] = left_root;
            }
        }
    }

    int count = 0;
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        if (root(static_cast<int>(index)) == static_cast<int>(index))
        {
            ++count;
        }
    }
    return count;
}

std::string joinFailures(const std::vector<std::string> &failures)
{
    std::ostringstream message;
    for (std::size_t index = 0; index < failures.size(); ++index)
    {
        if (index > 0)
        {
            message << ',';
        }
        message << failures[index];
    }
    return message.str();
}

} // namespace

MvsSceneClassification classifyMvsScene(const std::vector<CameraView> &views,
                                        const SparseCloud &sparse_cloud)
{
    MvsSceneClassification result;
    if (views.size() < 3 || sparse_cloud.points.size() < 6)
    {
        result.reason = "insufficient cameras or sparse points; using general profile";
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
        result.reason = "insufficient finite sparse points; using general profile";
        return result;
    }

    cv::Vec3d cloud_eigenvalues;
    cv::Matx33d cloud_eigenvectors;
    if (!covarianceEigen(cloud_points, cloud_eigenvalues, cloud_eigenvectors))
    {
        result.reason = "sparse cloud PCA failed; using general profile";
        return result;
    }
    const double largest_cloud_scale = std::sqrt(std::max(0.0, cloud_eigenvalues[0]));
    if (!std::isfinite(largest_cloud_scale) || largest_cloud_scale <= 1e-9 ||
        !std::isfinite(cloud_eigenvalues[1]) ||
        cloud_eigenvalues[1] <= cloud_eigenvalues[0] * 1e-12)
    {
        result.reason = "sparse cloud is degenerate; using general profile";
        return result;
    }
    const double thickness = std::sqrt(std::max(0.0, cloud_eigenvalues[2]));
    result.planeThicknessRatio = static_cast<float>(thickness / largest_cloud_scale);
    const Vec3 plane_normal = normalize(Vec3(cloud_eigenvectors(2, 0),
                                             cloud_eigenvectors(2, 1),
                                             cloud_eigenvectors(2, 2)));
    const Vec3 cloud_center = meanPoint(cloud_points);
    if (!finiteVector(cloud_center) || cv::norm(plane_normal) <= 1e-12)
    {
        result.reason = "sparse cloud centre or plane is invalid; using general profile";
        return result;
    }

    std::vector<Vec3> camera_centers;
    camera_centers.reserve(views.size());
    std::vector<double> convergence_values;
    convergence_values.reserve(views.size());
    std::vector<double> convergence_error_degrees;
    convergence_error_degrees.reserve(views.size());
    double convergence_sum = 0.0;
    int convergent_count = 0;
    int down_looking_count = 0;
    int positive_plane_side_count = 0;
    int negative_plane_side_count = 0;
    int valid_camera_count = 0;
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
        const Vec3 raw_optical_axis(rotation[6], rotation[7], rotation[8]);
        if (!finiteVector(center) || !finiteVector(raw_optical_axis) ||
            cv::norm(raw_optical_axis) <= 1e-12)
        {
            continue;
        }
        const Vec3 optical_axis = normalize(raw_optical_axis);
        const Vec3 cloud_offset = cloud_center - center;
        const double cloud_distance = cv::norm(cloud_offset);
        const Vec3 direction_to_cloud = cloud_distance > plane_side_epsilon
            ? cloud_offset / cloud_distance
            : Vec3(0.0, 0.0, 0.0);
        const double convergence = optical_axis.dot(direction_to_cloud);
        const double signed_plane_distance = plane_normal.dot(center - cloud_center);
        const double plane_axis_alignment = optical_axis.dot(plane_normal);
        const double forward_plane_distance = std::abs(plane_axis_alignment) > 1e-12
            ? -signed_plane_distance / plane_axis_alignment
            : -1.0;

        camera_centers.push_back(center);
        ++valid_camera_count;
        if (cloud_distance > plane_side_epsilon && std::isfinite(convergence))
        {
            const double clamped_convergence = std::clamp(convergence, -1.0, 1.0);
            convergence_values.push_back(clamped_convergence);
            convergence_error_degrees.push_back(
                std::acos(clamped_convergence) * kRadiansToDegrees);
            if (clamped_convergence >= kMinimumAlignment)
            {
                ++convergent_count;
            }
        }
        if (signed_plane_distance > plane_side_epsilon)
        {
            ++positive_plane_side_count;
        }
        else if (signed_plane_distance < -plane_side_epsilon)
        {
            ++negative_plane_side_count;
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
        result.reason = "no valid camera poses; using general profile";
        return result;
    }
    std::sort(convergence_values.begin(), convergence_values.end());
    for (const double convergence : convergence_values)
    {
        convergence_sum += convergence;
    }
    result.validCameraCount = valid_camera_count;
    result.convergentCameraCount = convergent_count;
    result.opticalAxisConvergence = convergence_values.empty()
        ? 0.0f
        : static_cast<float>(convergence_sum / convergence_values.size());
    if (!convergence_error_degrees.empty())
    {
        result.orbitalOpticalAxisMedianErrorDegrees = static_cast<float>(
            quantile(convergence_error_degrees, 0.50));
        result.orbitalOpticalAxisP90ErrorDegrees = static_cast<float>(
            quantile(convergence_error_degrees, 0.90));
    }
    result.downLookingConsistency = static_cast<float>(down_looking_count) / valid_camera_count;
    const int off_plane_camera_count = positive_plane_side_count + negative_plane_side_count;
    const float camera_side_consistency = off_plane_camera_count > 0
        ? static_cast<float>(std::max(positive_plane_side_count, negative_plane_side_count)) /
            off_plane_camera_count
        : 0.0f;

    cv::Vec3d camera_eigenvalues;
    cv::Matx33d camera_eigenvectors;
    const bool has_camera_pca =
        covarianceEigen(camera_centers, camera_eigenvalues, camera_eigenvectors) &&
        camera_eigenvalues[0] > 1e-12;
    if (has_camera_pca)
    {
        const double second_camera_scale = std::max(0.0, camera_eigenvalues[1]);
        const double third_camera_scale = std::max(0.0, camera_eigenvalues[2]);
        result.cameraCenterInPlaneBalance = static_cast<float>(
            second_camera_scale / camera_eigenvalues[0]);
        result.cameraCenterLinearity = static_cast<float>(
            1.0 - result.cameraCenterInPlaneBalance);
        result.cameraCenterNonPlanarity = second_camera_scale > 1e-12
            ? static_cast<float>(third_camera_scale / second_camera_scale)
            : 1.0f;
    }

    double camera_diameter = 0.0;
    for (std::size_t left = 0; left < camera_centers.size(); ++left)
    {
        for (std::size_t right = left + 1; right < camera_centers.size(); ++right)
        {
            camera_diameter = std::max(
                camera_diameter, cv::norm(camera_centers[left] - camera_centers[right]));
        }
    }
    const double distinct_tolerance = std::max(1e-9, camera_diameter * 1e-6);
    result.distinctCameraCenterCount = distinctPointCount(camera_centers, distinct_tolerance);

    bool has_projected_ring = false;
    if (has_camera_pca && result.distinctCameraCenterCount >= 3)
    {
        const Vec3 first_plane_axis(camera_eigenvectors(0, 0),
                                    camera_eigenvectors(0, 1),
                                    camera_eigenvectors(0, 2));
        const Vec3 second_plane_axis(camera_eigenvectors(1, 0),
                                     camera_eigenvectors(1, 1),
                                     camera_eigenvectors(1, 2));
        const Vec3 camera_center_mean = meanPoint(camera_centers);
        std::vector<double> projected_radii;
        std::vector<double> projected_angles;
        projected_radii.reserve(camera_centers.size());
        projected_angles.reserve(camera_centers.size());
        for (const Vec3 &camera_center : camera_centers)
        {
            const Vec3 cloud_relative_center = camera_center - cloud_center;
            const double projected_x = cloud_relative_center.dot(first_plane_axis);
            const double projected_y = cloud_relative_center.dot(second_plane_axis);
            projected_radii.push_back(std::hypot(projected_x, projected_y));
            projected_angles.push_back(std::atan2(projected_y, projected_x));
        }

        const double median_radius = quantile(projected_radii, 0.50);
        if (std::isfinite(median_radius) && median_radius > distinct_tolerance)
        {
            std::vector<double> radius_deviations;
            radius_deviations.reserve(projected_radii.size());
            for (const double radius : projected_radii)
            {
                radius_deviations.push_back(std::abs(radius - median_radius));
            }
            result.orbitalProjectedRadiusMadRatio = static_cast<float>(
                quantile(radius_deviations, 0.50) / median_radius);

            const Vec3 cloud_to_camera_mean = camera_center_mean - cloud_center;
            result.orbitalProjectedCenterOffsetRatio = static_cast<float>(
                std::hypot(cloud_to_camera_mean.dot(first_plane_axis),
                           cloud_to_camera_mean.dot(second_plane_axis)) /
                median_radius);

            const double minimum_projected_radius = *std::min_element(
                projected_radii.begin(), projected_radii.end());
            if (minimum_projected_radius > distinct_tolerance)
            {
                std::sort(projected_angles.begin(), projected_angles.end());
                double maximum_gap = 0.0;
                for (std::size_t index = 1; index < projected_angles.size(); ++index)
                {
                    maximum_gap = std::max(
                        maximum_gap, projected_angles[index] - projected_angles[index - 1]);
                }
                maximum_gap = std::max(
                    maximum_gap,
                    projected_angles.front() + 2.0 * CV_PI - projected_angles.back());
                result.orbitalMaximumAngularGapDegrees = static_cast<float>(
                    maximum_gap * kRadiansToDegrees);
                has_projected_ring = true;
            }
        }
    }

    const bool aerial = result.distinctCameraCenterCount >= 3 &&
                        result.downLookingConsistency >= 0.75f &&
                        camera_side_consistency >= 0.75f &&
                        result.planeThicknessRatio <= 0.20f;
    std::vector<std::string> orbital_failures;
    if (result.distinctCameraCenterCount < 3)
    {
        orbital_failures.push_back("fewer-than-3-distinct-cameras");
    }
    if (!has_camera_pca)
    {
        orbital_failures.push_back("camera-pca-degenerate");
    }
    else
    {
        if (result.cameraCenterInPlaneBalance < kMinimumRingInPlaneBalance)
        {
            orbital_failures.push_back("camera-centers-collinear");
        }
        if (result.cameraCenterNonPlanarity > kMaximumRingNonPlanarity)
        {
            orbital_failures.push_back("camera-centers-nonplanar");
        }
    }
    if (!has_projected_ring)
    {
        orbital_failures.push_back("projected-ring-degenerate");
    }
    else
    {
        if (result.orbitalProjectedRadiusMadRatio > kMaximumRingRadiusMadRatio)
        {
            orbital_failures.push_back("projected-radius-irregular");
        }
        if (result.orbitalProjectedCenterOffsetRatio > kMaximumRingCenterOffsetRatio)
        {
            orbital_failures.push_back("ring-not-centered-on-cloud");
        }
        if (result.orbitalMaximumAngularGapDegrees > kMaximumRingAngularGapDegrees)
        {
            orbital_failures.push_back("incomplete-angular-ring");
        }
    }
    if (convergence_error_degrees.size() != camera_centers.size())
    {
        orbital_failures.push_back("undefined-axis-convergence");
    }
    else
    {
        if (result.orbitalOpticalAxisMedianErrorDegrees >
            kMaximumRingMedianAxisErrorDegrees)
        {
            orbital_failures.push_back("median-axis-not-convergent");
        }
        if (result.orbitalOpticalAxisP90ErrorDegrees >
            kMaximumRingP90AxisErrorDegrees)
        {
            orbital_failures.push_back("axis-tail-not-convergent");
        }
    }
    result.orbitalGatePassed = orbital_failures.empty();
    result.profile = aerial
        ? MvsSceneProfile::AerialTerrain
        : (result.orbitalGatePassed
               ? MvsSceneProfile::OrbitalObject
               : MvsSceneProfile::Custom);

    std::ostringstream message;
    message << (aerial
                    ? "aerial terrain"
                    : (result.orbitalGatePassed
                           ? "orbital object"
                           : "general capture"))
            << ": down-looking=" << result.downLookingConsistency
            << ", convergence=" << result.opticalAxisConvergence
            << ", plane-thickness=" << result.planeThicknessRatio
            << ", camera-side=" << camera_side_consistency
            << ", convergent-cameras=" << convergent_count << '/' << valid_camera_count
            << ", distinct-cameras=" << result.distinctCameraCenterCount
            << ", in-plane-balance=" << result.cameraCenterInPlaneBalance
            << ", camera-nonplanarity=" << result.cameraCenterNonPlanarity
            << ", radius-mad-ratio=" << result.orbitalProjectedRadiusMadRatio
            << ", center-offset-ratio=" << result.orbitalProjectedCenterOffsetRatio
            << ", max-angular-gap-deg=" << result.orbitalMaximumAngularGapDegrees
            << ", axis-error-median-deg="
            << result.orbitalOpticalAxisMedianErrorDegrees
            << ", axis-error-p90-deg=" << result.orbitalOpticalAxisP90ErrorDegrees;
    if (!aerial && !result.orbitalGatePassed)
    {
        message << ", orbital-gate-failures=" << joinFailures(orbital_failures);
    }
    result.reason = message.str();
    return result;
}

DepthFilterMode recommendedMvsDepthFilterMode(MvsSceneProfile scene_profile)
{
    return scene_profile == MvsSceneProfile::OrbitalObject
        ? DepthFilterMode::Mild
        : DepthFilterMode::Moderate;
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

float constrainMvsSourceMaximumAngleDeg(float scene_maximum_degrees,
                                        float configured_cap_degrees)
{
    if (!std::isfinite(configured_cap_degrees) ||
        configured_cap_degrees <= 0.0f)
    {
        return scene_maximum_degrees;
    }
    return std::min(scene_maximum_degrees, configured_cap_degrees);
}

} // namespace mvs
} // namespace xjw
