#include "ColmapImageUndistorter.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace xjw::camera
{
namespace
{

int expectedParameterCount(const std::string &model)
{
    static const std::unordered_map<std::string, int> counts{
        {"SIMPLE_PINHOLE", 3},
        {"PINHOLE", 4},
        {"SIMPLE_RADIAL", 4},
        {"RADIAL", 5},
        {"OPENCV", 8},
        {"FULL_OPENCV", 12},
        {"SIMPLE_RADIAL_FISHEYE", 4},
        {"RADIAL_FISHEYE", 5},
        {"OPENCV_FISHEYE", 8},
        {"THIN_PRISM_FISHEYE", 12},
    };
    const auto found = counts.find(model);
    return found == counts.end() ? -1 : found->second;
}

std::array<double, 4> pinholeParameters(const ColmapRasterModel &camera)
{
    if (camera.model == "SIMPLE_PINHOLE" ||
        camera.model == "SIMPLE_RADIAL" ||
        camera.model == "RADIAL" ||
        camera.model == "SIMPLE_RADIAL_FISHEYE" ||
        camera.model == "RADIAL_FISHEYE")
    {
        return {camera.params[0], camera.params[0],
                camera.params[1] - 0.5, camera.params[2] - 0.5};
    }
    return {camera.params[0], camera.params[1],
            camera.params[2] - 0.5, camera.params[3] - 0.5};
}

cv::Point2d projectSourceRaster(
    const ColmapRasterModel &camera,
    double x,
    double y)
{
    double u = x;
    double v = y;
    const double r2_pinhole = x * x + y * y;
    if (camera.model == "SIMPLE_RADIAL")
    {
        const double radial = 1.0 + camera.params[3] * r2_pinhole;
        u = radial * x;
        v = radial * y;
    }
    else if (camera.model == "RADIAL")
    {
        const double r4 = r2_pinhole * r2_pinhole;
        const double radial = 1.0 + camera.params[3] * r2_pinhole +
            camera.params[4] * r4;
        u = radial * x;
        v = radial * y;
    }
    else if (camera.model == "OPENCV" || camera.model == "FULL_OPENCV")
    {
        const double r4 = r2_pinhole * r2_pinhole;
        const double r6 = r4 * r2_pinhole;
        const double k1 = camera.params[4];
        const double k2 = camera.params[5];
        const double p1 = camera.params[6];
        const double p2 = camera.params[7];
        double radial = 1.0 + k1 * r2_pinhole + k2 * r4;
        if (camera.model == "FULL_OPENCV")
        {
            radial = (radial + camera.params[8] * r6) /
                (1.0 + camera.params[9] * r2_pinhole +
                 camera.params[10] * r4 + camera.params[11] * r6);
        }
        u = radial * x + 2.0 * p1 * x * y +
            p2 * (r2_pinhole + 2.0 * x * x);
        v = radial * y + p1 * (r2_pinhole + 2.0 * y * y) +
            2.0 * p2 * x * y;
    }
    else if (camera.model == "SIMPLE_RADIAL_FISHEYE" ||
             camera.model == "RADIAL_FISHEYE" ||
             camera.model == "OPENCV_FISHEYE" ||
             camera.model == "THIN_PRISM_FISHEYE")
    {
        const double radius = std::sqrt(r2_pinhole);
        const double theta = std::atan(radius);
        const double scale = radius > 1.0e-12 ? theta / radius : 1.0;
        const double theta_x = scale * x;
        const double theta_y = scale * y;
        const double r2 = theta_x * theta_x + theta_y * theta_y;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        if (camera.model == "SIMPLE_RADIAL_FISHEYE" ||
            camera.model == "RADIAL_FISHEYE")
        {
            const double k1 = camera.params[3];
            const double k2 = camera.model == "RADIAL_FISHEYE"
                ? camera.params[4] : 0.0;
            const double radial = 1.0 + k1 * r2 + k2 * r4;
            u = radial * theta_x;
            v = radial * theta_y;
        }
        else if (camera.model == "OPENCV_FISHEYE")
        {
            const double radial = 1.0 + camera.params[4] * r2 +
                camera.params[5] * r4 + camera.params[6] * r6 +
                camera.params[7] * r8;
            u = radial * theta_x;
            v = radial * theta_y;
        }
        else
        {
            const double radial = 1.0 + camera.params[4] * r2 +
                camera.params[5] * r4 + camera.params[8] * r6 +
                camera.params[9] * r8;
            const double xy = theta_x * theta_y;
            u = radial * theta_x + 2.0 * camera.params[6] * xy +
                camera.params[7] * (r2 + 2.0 * theta_x * theta_x) +
                camera.params[10] * r2;
            v = radial * theta_y + camera.params[6] *
                (r2 + 2.0 * theta_y * theta_y) +
                2.0 * camera.params[7] * xy + camera.params[11] * r2;
        }
    }

    const auto pinhole = pinholeParameters(camera);
    return cv::Point2d(
        pinhole[0] * u + pinhole[2],
        pinhole[1] * v + pinhole[3]);
}

bool sourceCoordinateIsValid(
    const cv::Point2d &point,
    const ColmapRasterModel &camera)
{
    return std::isfinite(point.x) && std::isfinite(point.y) &&
        point.x >= 0.0 && point.y >= 0.0 &&
        point.x <= camera.width - 1.0 &&
        point.y <= camera.height - 1.0;
}

bool sampledTargetIsValid(
    const ColmapRasterModel &camera,
    const std::array<double, 4> &pinhole,
    double focalScale)
{
    constexpr int samples = 128;
    for (int row_sample = 0; row_sample < samples; ++row_sample)
    {
        const double row = (camera.height - 1.0) * row_sample / (samples - 1.0);
        for (int column_sample = 0; column_sample < samples; ++column_sample)
        {
            if (row_sample != 0 && row_sample != samples - 1 &&
                column_sample != 0 && column_sample != samples - 1)
            {
                continue;
            }
            const double column = (camera.width - 1.0) *
                column_sample / (samples - 1.0);
            const double x = (column - pinhole[2]) / (pinhole[0] * focalScale);
            const double y = (row - pinhole[3]) / (pinhole[1] * focalScale);
            if (!sourceCoordinateIsValid(projectSourceRaster(camera, x, y), camera))
            {
                return false;
            }
        }
    }
    return true;
}

double conservativeFocalScale(
    const ColmapRasterModel &camera,
    const std::array<double, 4> &pinhole)
{
    if (sampledTargetIsValid(camera, pinhole, 1.0))
    {
        return 1.0;
    }
    double invalid_scale = 1.0;
    double valid_scale = 1.05;
    while (valid_scale <= 4.0 &&
           !sampledTargetIsValid(camera, pinhole, valid_scale))
    {
        invalid_scale = valid_scale;
        valid_scale *= 1.05;
    }
    if (valid_scale > 4.0)
    {
        throw std::runtime_error(
            "COLMAP distortion requires more than 4x focal crop");
    }
    for (int iteration = 0; iteration < 24; ++iteration)
    {
        const double middle = 0.5 * (invalid_scale + valid_scale);
        if (sampledTargetIsValid(camera, pinhole, middle))
        {
            valid_scale = middle;
        }
        else
        {
            invalid_scale = middle;
        }
    }
    return valid_scale * 1.002;
}

} // namespace

bool isSupportedColmapPreUndistortModel(const ColmapRasterModel &camera)
{
    return camera.width > 0 && camera.height > 0 &&
        expectedParameterCount(camera.model) ==
            static_cast<int>(camera.params.size()) &&
        std::all_of(camera.params.cbegin(), camera.params.cend(),
                    [](double value) { return std::isfinite(value); });
}

ColmapUndistortedRaster undistortColmapRaster(
    const cv::Mat &source,
    const ColmapRasterModel &camera)
{
    if (!isSupportedColmapPreUndistortModel(camera))
    {
        throw std::invalid_argument(
            "unsupported or malformed COLMAP camera model: " + camera.model);
    }
    if (source.empty() || source.cols != camera.width ||
        source.rows != camera.height)
    {
        throw std::invalid_argument(
            "source raster dimensions do not match COLMAP camera");
    }
    const auto pinhole = pinholeParameters(camera);
    const double focal_scale = conservativeFocalScale(camera, pinhole);
    cv::Mat map_x(camera.height, camera.width, CV_32FC1);
    cv::Mat map_y(camera.height, camera.width, CV_32FC1);
    cv::Mat valid(camera.height, camera.width, CV_8UC1, cv::Scalar(255));
    cv::parallel_for_(cv::Range(0, camera.height), [&](const cv::Range &range) {
        for (int row = range.start; row < range.end; ++row)
        {
            float *x_row = map_x.ptr<float>(row);
            float *y_row = map_y.ptr<float>(row);
            std::uint8_t *valid_row = valid.ptr<std::uint8_t>(row);
            for (int column = 0; column < camera.width; ++column)
            {
                const double x = (column - pinhole[2]) /
                    (pinhole[0] * focal_scale);
                const double y = (row - pinhole[3]) /
                    (pinhole[1] * focal_scale);
                const cv::Point2d projected = projectSourceRaster(camera, x, y);
                x_row[column] = static_cast<float>(projected.x);
                y_row[column] = static_cast<float>(projected.y);
                if (!sourceCoordinateIsValid(projected, camera))
                {
                    valid_row[column] = 0;
                }
            }
        }
    });
    if (cv::countNonZero(valid) != camera.width * camera.height)
    {
        throw std::runtime_error(
            "conservative COLMAP pre-undistortion left invalid output pixels");
    }
    ColmapUndistortedRaster result;
    cv::remap(
        source, result.image, map_x, map_y, cv::INTER_LINEAR,
        cv::BORDER_CONSTANT, cv::Scalar());
    result.validMask = std::move(valid);
    result.rasterIndexIntrinsics = {
        pinhole[0] * focal_scale, 0.0, pinhole[2],
        0.0, pinhole[1] * focal_scale, pinhole[3],
        0.0, 0.0, 1.0};
    result.focalScale = focal_scale;
    return result;
}

} // namespace xjw::camera
