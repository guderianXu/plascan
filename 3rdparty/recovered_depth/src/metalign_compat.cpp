#include "metalign/geometry.hpp"
#include "metalign/image.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metalign
{
Mat3 Mat3::identity()
{
    Mat3 result;
    result(0, 0) = 1.0;
    result(1, 1) = 1.0;
    result(2, 2) = 1.0;
    return result;
}

Vec2 operator+(Vec2 lhs, Vec2 rhs) { return {lhs.x + rhs.x, lhs.y + rhs.y}; }
Vec2 operator-(Vec2 lhs, Vec2 rhs) { return {lhs.x - rhs.x, lhs.y - rhs.y}; }
Vec2 operator*(Vec2 value, double scale) { return {value.x * scale, value.y * scale}; }
Vec3 operator+(Vec3 lhs, Vec3 rhs) { return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z}; }
Vec3 operator-(Vec3 lhs, Vec3 rhs) { return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z}; }
Vec3 operator*(Vec3 value, double scale) { return {value.x * scale, value.y * scale, value.z * scale}; }
Vec3 operator/(Vec3 value, double scale) { return {value.x / scale, value.y / scale, value.z / scale}; }
double dot(Vec2 lhs, Vec2 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
double dot(Vec3 lhs, Vec3 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
Vec3 cross(Vec3 lhs, Vec3 rhs)
{
    return {lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x};
}
double norm(Vec2 value) { return std::sqrt(dot(value, value)); }
double norm(Vec3 value) { return std::sqrt(dot(value, value)); }
Vec3 normalized(Vec3 value)
{
    const double length = norm(value);
    return length > 0.0 ? value / length : Vec3{};
}
Mat3 transpose(const Mat3 &matrix)
{
    Mat3 result;
    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            result(row, column) = matrix(column, row);
        }
    }
    return result;
}
Mat3 operator*(const Mat3 &lhs, const Mat3 &rhs)
{
    Mat3 result;
    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            for (std::size_t inner = 0; inner < 3; ++inner)
            {
                result(row, column) += lhs(row, inner) * rhs(inner, column);
            }
        }
    }
    return result;
}
Vec3 operator*(const Mat3 &matrix, Vec3 vector)
{
    return {matrix(0, 0) * vector.x + matrix(0, 1) * vector.y + matrix(0, 2) * vector.z,
            matrix(1, 0) * vector.x + matrix(1, 1) * vector.y + matrix(1, 2) * vector.z,
            matrix(2, 0) * vector.x + matrix(2, 1) * vector.y + matrix(2, 2) * vector.z};
}
Mat3 operator*(const Mat3 &matrix, double scale)
{
    Mat3 result = matrix;
    for (double &value : result.v)
    {
        value *= scale;
    }
    return result;
}
Mat3 operator+(const Mat3 &lhs, const Mat3 &rhs)
{
    Mat3 result;
    for (std::size_t index = 0; index < result.v.size(); ++index)
    {
        result.v[index] = lhs.v[index] + rhs.v[index];
    }
    return result;
}
double determinant(const Mat3 &matrix)
{
    return matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1))
        - matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0))
        + matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
}
Mat3 inverse(const Mat3 &matrix)
{
    const double value = determinant(matrix);
    if (std::abs(value) < 1.0e-18)
    {
        throw std::runtime_error("cannot invert singular 3x3 matrix");
    }
    Mat3 result;
    result(0, 0) = (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) / value;
    result(0, 1) = (matrix(0, 2) * matrix(2, 1) - matrix(0, 1) * matrix(2, 2)) / value;
    result(0, 2) = (matrix(0, 1) * matrix(1, 2) - matrix(0, 2) * matrix(1, 1)) / value;
    result(1, 0) = (matrix(1, 2) * matrix(2, 0) - matrix(1, 0) * matrix(2, 2)) / value;
    result(1, 1) = (matrix(0, 0) * matrix(2, 2) - matrix(0, 2) * matrix(2, 0)) / value;
    result(1, 2) = (matrix(0, 2) * matrix(1, 0) - matrix(0, 0) * matrix(1, 2)) / value;
    result(2, 0) = (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0)) / value;
    result(2, 1) = (matrix(0, 1) * matrix(2, 0) - matrix(0, 0) * matrix(2, 1)) / value;
    result(2, 2) = (matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0)) / value;
    return result;
}
Vec3 camera_center(const Pose &pose)
{
    if (pose.center)
    {
        return *pose.center;
    }
    return transpose(pose.rotation) * (pose.translation * -1.0);
}

Vec2 project_local(const CameraModel &camera, Vec3 local)
{
    if (std::abs(local.z) < 1.0e-15)
    {
        const double infinity = std::numeric_limits<double>::infinity();
        return {infinity, infinity};
    }
    const double inverse_z = 1.0 / local.z;
    const double x = local.x * inverse_z;
    const double y = local.y * inverse_z;
    const double x_squared = x * x;
    const double y_squared = y * y;
    const double r2 = x_squared + y_squared;
    const double r4 = r2 * r2;
    const double radial = 1.0 + camera.k1 * r2 + camera.k2 * r4
        + camera.k3 * r4 * r2 + camera.k4 * r4 * r4;
    const double radial_delta = radial - 1.0;
    double diagonal_x = 3.0 * x;
    diagonal_x *= x;
    diagonal_x += y_squared;
    double diagonal_y = 3.0 * y;
    diagonal_y *= y;
    diagonal_y += x_squared;
    double cross_x = camera.p2 + camera.p2;
    cross_x *= x;
    cross_x *= y;
    double cross_y = camera.p1 + camera.p1;
    cross_y *= x;
    cross_y *= y;
    const double tangential_x = camera.p1 * diagonal_x + cross_x + camera.p3 * r2 + camera.p4 * r4;
    const double tangential_y = cross_y + camera.p2 * diagonal_y;
    const double distorted_x = x + (x * radial_delta + tangential_x);
    const double distorted_y = y + (y * radial_delta + tangential_y);
    return {(camera.f + camera.b1) * distorted_x + camera.b2 * distorted_y + camera.cx,
            camera.f * distorted_y + camera.cy};
}

Vec2 project(const CameraModel &camera, const Pose &pose, Vec3 point)
{
    const Vec3 local = pose.center
        ? pose.rotation * (point - *pose.center)
        : pose.rotation * point + pose.translation;
    return project_local(camera, local);
}

Vec3 bearing(const CameraModel &camera, Vec2 pixel)
{
    const double centre_x = camera.cx - camera.cx_offset;
    const double centre_y = camera.cy - camera.cy_offset;
    const double distorted_y = ((pixel.y - centre_y) - camera.cy_offset) / camera.f;
    const double corrected_x = (pixel.x - centre_x) - camera.cx_offset;
    const double distorted_x = (corrected_x - camera.b2 * distorted_y) / (camera.f + camera.b1);
    double x = distorted_x;
    double y = distorted_y;
    for (std::size_t iteration = 0; iteration < 10; ++iteration)
    {
        const double x_squared = x * x;
        const double y_squared = y * y;
        const double r2 = x_squared + y_squared;
        const double radial = 1.0 + r2 * (camera.k1 + r2 * (camera.k2 + r2 * (camera.k3 + r2 * camera.k4)));
        if (std::abs(radial) < 1.0e-12)
        {
            break;
        }
        double diagonal_x = 3.0 * x;
        diagonal_x *= x;
        diagonal_x += y_squared;
        double diagonal_y = 3.0 * y;
        diagonal_y *= y;
        diagonal_y += x_squared;
        double cross_x = camera.p2 + camera.p2;
        cross_x *= x;
        cross_x *= y;
        double cross_y = camera.p1 + camera.p1;
        cross_y *= x;
        cross_y *= y;
        const double tangential_x = camera.p1 * diagonal_x + cross_x + camera.p3 * r2 + camera.p4 * r2 * r2;
        const double tangential_y = cross_y + camera.p2 * diagonal_y;
        const double inverse_radial = 1.0 / radial;
        const double next_x = (distorted_x - tangential_x) * inverse_radial;
        const double next_y = (distorted_y - tangential_y) * inverse_radial;
        const double change_x = next_x - x;
        const double change_y = next_y - y;
        x = next_x;
        y = next_y;
        if (iteration >= 4 && change_x * change_x + change_y * change_y < 1.0e-10)
        {
            break;
        }
    }
    return {x, y, 1.0};
}

Image load_gray_image(const std::filesystem::path &path, bool)
{
    const cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty() || bgr.type() != CV_8UC3)
    {
        throw std::runtime_error("cannot decode recovered PatchMatch image: " + path.string());
    }
    Image result;
    result.width = static_cast<std::size_t>(bgr.cols);
    result.height = static_cast<std::size_t>(bgr.rows);
    result.gray.resize(result.width * result.height);
    for (int row = 0; row < bgr.rows; ++row)
    {
        const cv::Vec3b *source = bgr.ptr<cv::Vec3b>(row);
        for (int column = 0; column < bgr.cols; ++column)
        {
            const double luminance = static_cast<double>(source[column][2]) * 0.299
                + static_cast<double>(source[column][1]) * 0.587
                + static_cast<double>(source[column][0]) * 0.114;
            const auto code = static_cast<std::uint8_t>(std::clamp(static_cast<int>(luminance), 0, 255));
            result.gray[static_cast<std::size_t>(row) * result.width + static_cast<std::size_t>(column)]
                = static_cast<float>(static_cast<double>(code) / 255.0);
        }
    }
    return result;
}

float sample_bilinear(const Image &image, double x, double y)
{
    if (image.empty())
    {
        return 0.0F;
    }
    x = std::clamp(x, 0.0, static_cast<double>(image.width - 1));
    y = std::clamp(y, 0.0, static_cast<double>(image.height - 1));
    const std::size_t x0 = static_cast<std::size_t>(std::floor(x));
    const std::size_t y0 = static_cast<std::size_t>(std::floor(y));
    const std::size_t x1 = std::min(x0 + 1, image.width - 1);
    const std::size_t y1 = std::min(y0 + 1, image.height - 1);
    const double dx = x - static_cast<double>(x0);
    const double dy = y - static_cast<double>(y0);
    const double top = image.at(x0, y0) * (1.0 - dx) + image.at(x1, y0) * dx;
    const double bottom = image.at(x0, y1) * (1.0 - dx) + image.at(x1, y1) * dx;
    return static_cast<float>(top * (1.0 - dy) + bottom * dy);
}
} // namespace metalign
