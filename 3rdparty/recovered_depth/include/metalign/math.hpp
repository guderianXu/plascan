#pragma once

#include <array>
#include <cstddef>
#include <optional>

namespace metalign
{
struct Vec2
{
    double x = 0.0;
    double y = 0.0;
};

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Mat3
{
    std::array<double, 9> v{};

    double &operator()(std::size_t row, std::size_t column)
    {
        return v[row * 3 + column];
    }

    double operator()(std::size_t row, std::size_t column) const
    {
        return v[row * 3 + column];
    }

    static Mat3 identity();
};

struct Pose
{
    Mat3 rotation = Mat3::identity();
    Vec3 translation{};
    std::optional<Vec3> center;
};

Vec2 operator+(Vec2 lhs, Vec2 rhs);
Vec2 operator-(Vec2 lhs, Vec2 rhs);
Vec2 operator*(Vec2 value, double scale);
Vec3 operator+(Vec3 lhs, Vec3 rhs);
Vec3 operator-(Vec3 lhs, Vec3 rhs);
Vec3 operator*(Vec3 value, double scale);
Vec3 operator/(Vec3 value, double scale);
double dot(Vec2 lhs, Vec2 rhs);
double dot(Vec3 lhs, Vec3 rhs);
Vec3 cross(Vec3 lhs, Vec3 rhs);
double norm(Vec2 value);
double norm(Vec3 value);
Vec3 normalized(Vec3 value);
Mat3 transpose(const Mat3 &matrix);
Mat3 operator*(const Mat3 &lhs, const Mat3 &rhs);
Vec3 operator*(const Mat3 &matrix, Vec3 vector);
Mat3 operator*(const Mat3 &matrix, double scale);
Mat3 operator+(const Mat3 &lhs, const Mat3 &rhs);
double determinant(const Mat3 &matrix);
Mat3 inverse(const Mat3 &matrix);
Vec3 camera_center(const Pose &pose);
} // namespace metalign
