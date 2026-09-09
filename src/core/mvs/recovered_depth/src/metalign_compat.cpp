#include "metalign/geometry.hpp"
#include "metalign/image.hpp"

#include <algorithm>
#include <cctype>
#include <csetjmp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <jpeglib.h>
#include <opencv2/imgcodecs.hpp>

namespace metalign
{
    namespace
    {
        struct JpegError
        {
            jpeg_error_mgr base{};
            jmp_buf jump{};
            char message[JMSG_LENGTH_MAX]{};
        };

        std::vector<std::uint8_t> readImageBytes(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
                throw std::runtime_error("cannot open recovered image: " + path.string());
            const auto size = file.tellg();
            if (size <= 0 || static_cast<std::uintmax_t>(size) > std::numeric_limits<unsigned long>::max())
                throw std::runtime_error("invalid recovered image size: " + path.string());
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
                throw std::runtime_error("cannot read recovered image: " + path.string());
            return bytes;
        }

        struct JpegDecodeState
        {
            jpeg_decompress_struct info{};
            JpegError error{};
            std::vector<std::uint8_t> rgb;

            ~JpegDecodeState()
            {
                if (info.mem)
                    jpeg_destroy_decompress(&info);
            }
        };

        void jpegErrorExit(j_common_ptr info)
        {
            auto* error = reinterpret_cast<JpegError*>(info->err);
            (*info->err->format_message)(info, error->message);
            longjmp(error->jump, 1);
        }

        Image fromRgb(std::size_t width, std::size_t height, const std::vector<std::uint8_t>& rgb)
        {
            Image result;
            result.width = width;
            result.height = height;
            result.gray.resize(width * height);
            for (std::size_t pixel = 0; pixel < width * height; ++pixel)
            {
                const double red = static_cast<double>(rgb[pixel * 3]);
                const double green = static_cast<double>(rgb[pixel * 3 + 1]);
                const double blue = static_cast<double>(rgb[pixel * 3 + 2]);
                const double luminance = red * 0.299 + green * 0.587 + blue * 0.114;
                const auto code = static_cast<std::uint8_t>(std::clamp(static_cast<int>(luminance), 0, 255));
                result.gray[pixel] = static_cast<float>(static_cast<double>(code) / 255.0);
            }
            return result;
        }

        Image loadJpegGray(const std::filesystem::path& path, bool rgb_only = false)
        {
            const auto encoded = readImageBytes(path);
            // Mutable decoder state lives on the heap. libjpeg longjmp must
            // neither bypass vector construction nor invalidate local state.
            const auto state = std::make_unique<JpegDecodeState>();
            auto& info = state->info;
            auto& error = state->error;
            info.err = jpeg_std_error(&error.base);
            error.base.error_exit = jpegErrorExit;
            if (setjmp(error.jump))
            {
                throw std::runtime_error("recovered JPEG decode failed: " + path.string() + ": " + error.message);
            }
            jpeg_create_decompress(&info);
            jpeg_mem_src(&info, encoded.data(), static_cast<unsigned long>(encoded.size()));
            jpeg_read_header(&info, TRUE);
            info.out_color_space = JCS_RGB;
            info.dct_method = JDCT_ISLOW;
            info.do_fancy_upsampling = TRUE;
            info.do_block_smoothing = TRUE;
            jpeg_start_decompress(&info);
            const std::size_t width = info.output_width;
            const std::size_t height = info.output_height;
            auto& rgb = state->rgb;
            rgb.resize(width * height * 3);
            while (info.output_scanline < info.output_height)
            {
                JSAMPROW row = rgb.data() + static_cast<std::size_t>(info.output_scanline) * width * 3;
                jpeg_read_scanlines(&info, &row, 1);
            }
            jpeg_finish_decompress(&info);
            jpeg_destroy_decompress(&info);
            if (rgb_only)
            {
                Image result;
                result.width = width;
                result.height = height;
                result.rgb = std::move(rgb);
                return result;
            }
            return fromRgb(width, height, rgb);
        }

        std::string lowercaseExtension(const std::filesystem::path& path)
        {
            std::string extension = path.extension().string();
            std::transform(extension.begin(),
                           extension.end(),
                           extension.begin(),
                           [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
            return extension;
        }
    } // namespace

    Mat3 Mat3::identity()
    {
        Mat3 result;
        result(0, 0) = 1.0;
        result(1, 1) = 1.0;
        result(2, 2) = 1.0;
        return result;
    }

    Vec2 operator+(Vec2 lhs, Vec2 rhs)
    {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }
    Vec2 operator-(Vec2 lhs, Vec2 rhs)
    {
        return {lhs.x - rhs.x, lhs.y - rhs.y};
    }
    Vec2 operator*(Vec2 value, double scale)
    {
        return {value.x * scale, value.y * scale};
    }
    Vec3 operator+(Vec3 lhs, Vec3 rhs)
    {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    Vec3 operator-(Vec3 lhs, Vec3 rhs)
    {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    Vec3 operator*(Vec3 value, double scale)
    {
        return {value.x * scale, value.y * scale, value.z * scale};
    }
    Vec3 operator/(Vec3 value, double scale)
    {
        return {value.x / scale, value.y / scale, value.z / scale};
    }
    double dot(Vec2 lhs, Vec2 rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y;
    }
    double dot(Vec3 lhs, Vec3 rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }
    Vec3 cross(Vec3 lhs, Vec3 rhs)
    {
        return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x};
    }
    double norm(Vec2 value)
    {
        return std::sqrt(dot(value, value));
    }
    double norm(Vec3 value)
    {
        return std::sqrt(dot(value, value));
    }
    Vec3 normalized(Vec3 value)
    {
        const double length = norm(value);
        return length > 0.0 ? value / length : Vec3{};
    }
    Mat3 transpose(const Mat3& matrix)
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
    Mat3 operator*(const Mat3& lhs, const Mat3& rhs)
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
    Vec3 operator*(const Mat3& matrix, Vec3 vector)
    {
        return {matrix(0, 0) * vector.x + matrix(0, 1) * vector.y + matrix(0, 2) * vector.z,
                matrix(1, 0) * vector.x + matrix(1, 1) * vector.y + matrix(1, 2) * vector.z,
                matrix(2, 0) * vector.x + matrix(2, 1) * vector.y + matrix(2, 2) * vector.z};
    }
    Mat3 operator*(const Mat3& matrix, double scale)
    {
        Mat3 result = matrix;
        for (double& value : result.v)
        {
            value *= scale;
        }
        return result;
    }
    Mat3 operator+(const Mat3& lhs, const Mat3& rhs)
    {
        Mat3 result;
        for (std::size_t index = 0; index < result.v.size(); ++index)
        {
            result.v[index] = lhs.v[index] + rhs.v[index];
        }
        return result;
    }
    double determinant(const Mat3& matrix)
    {
        return matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
               matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
               matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
    }
    Mat3 inverse(const Mat3& matrix)
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
    Vec3 camera_center(const Pose& pose)
    {
        if (pose.center)
        {
            return *pose.center;
        }
        return transpose(pose.rotation) * (pose.translation * -1.0);
    }

    Vec2 project_local(const CameraModel& camera, Vec3 local)
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
        const double radial = 1.0 + camera.k1 * r2 + camera.k2 * r4 + camera.k3 * r4 * r2 + camera.k4 * r4 * r4;
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

    Vec2 project(const CameraModel& camera, const Pose& pose, Vec3 point)
    {
        const Vec3 local =
            pose.center ? pose.rotation * (point - *pose.center) : pose.rotation * point + pose.translation;
        return project_local(camera, local);
    }

    Vec3 bearing(const CameraModel& camera, Vec2 pixel)
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

    Image load_gray_image(const std::filesystem::path& path, bool)
    {
        const std::string extension = lowercaseExtension(path);
        if (extension == ".jpg" || extension == ".jpeg")
        {
            return loadJpegGray(path);
        }
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
            const cv::Vec3b* source = bgr.ptr<cv::Vec3b>(row);
            for (int column = 0; column < bgr.cols; ++column)
            {
                const double luminance = static_cast<double>(source[column][2]) * 0.299 +
                                         static_cast<double>(source[column][1]) * 0.587 +
                                         static_cast<double>(source[column][0]) * 0.114;
                const auto code = static_cast<std::uint8_t>(std::clamp(static_cast<int>(luminance), 0, 255));
                result.gray[static_cast<std::size_t>(row) * result.width + static_cast<std::size_t>(column)] =
                    static_cast<float>(static_cast<double>(code) / 255.0);
            }
        }
        return result;
    }

    Image load_rgb_image(const std::filesystem::path& path)
    {
        const auto extension = lowercaseExtension(path);
        if (extension == ".jpg" || extension == ".jpeg")
        {
            return loadJpegGray(path, true);
        }
        const auto bgr = cv::imdecode(readImageBytes(path), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (bgr.empty() || bgr.type() != CV_8UC3)
        {
            throw std::runtime_error("cannot decode recovered RGB image: " + path.string());
        }
        Image result;
        result.width = static_cast<std::size_t>(bgr.cols);
        result.height = static_cast<std::size_t>(bgr.rows);
        result.rgb.resize(result.width * result.height * 3);
        for (int row = 0; row < bgr.rows; ++row)
        {
            const auto* source = bgr.ptr<cv::Vec3b>(row);
            for (int col = 0; col < bgr.cols; ++col)
            {
                const auto offset = (static_cast<std::size_t>(row) * result.width + static_cast<std::size_t>(col)) * 3;
                result.rgb[offset] = source[col][2];
                result.rgb[offset + 1] = source[col][1];
                result.rgb[offset + 2] = source[col][0];
            }
        }
        return result;
    }

} // namespace metalign
