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
    int valid_camera_count = 0;
    constexpr double kMinimumAlignment = 0.8191520443;

    for (const CameraView &view : views)
    {
        const Camera camera = view.camera.normalizedForPositiveDepth();
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

        camera_centers.push_back(center);
        convergence_sum += std::max(-1.0, std::min(1.0, convergence));
        ++valid_camera_count;
        if (convergence >= kMinimumAlignment)
        {
            ++convergent_count;
        }
        if (convergence >= kMinimumAlignment &&
            std::abs(optical_axis.dot(plane_normal)) >= kMinimumAlignment)
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

    cv::Vec3d camera_eigenvalues;
    cv::Matx33d camera_eigenvectors;
    if (covarianceEigen(camera_centers, camera_eigenvalues, camera_eigenvectors) &&
        camera_eigenvalues[0] > 1e-12)
    {
        result.cameraCenterLinearity = static_cast<float>(
            1.0 - std::max(0.0, camera_eigenvalues[1]) / camera_eigenvalues[0]);
    }

    const bool aerial = result.downLookingConsistency >= 0.75f &&
                        result.planeThicknessRatio <= 0.20f;
    result.profile = aerial ? MvsSceneProfile::AerialTerrain : MvsSceneProfile::OrbitalObject;

    std::ostringstream message;
    message << (aerial ? "aerial terrain" : "orbital object")
            << ": down-looking=" << result.downLookingConsistency
            << ", convergence=" << result.opticalAxisConvergence
            << ", plane-thickness=" << result.planeThicknessRatio
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
        scene_target = high_quality ? 6 : 4;
    }

    const int requested_count = std::max(std::max(1, configured_count), scene_target);
    return std::min(requested_count, view_count - 1);
}

} // namespace mvs
} // namespace xjw
