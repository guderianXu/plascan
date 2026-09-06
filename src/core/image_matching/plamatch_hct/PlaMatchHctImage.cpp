#include "PlaMatchHctImage.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace
{

    cv::Mat normalizedGray(const cv::Mat& source)
    {
        if (source.empty())
        {
            return {};
        }

        cv::Mat gray;
        if (source.channels() == 1)
        {
            gray = source;
        }
        else
        {
            cv::cvtColor(source, gray, source.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
        }

        cv::Mat normalized;
        if (gray.depth() == CV_8U)
        {
            gray.convertTo(normalized, CV_32F, 1.0 / 255.0);
        }
        else
        {
            double minimum = 0.0;
            double maximum = 0.0;
            cv::minMaxLoc(gray, &minimum, &maximum);
            gray.convertTo(normalized, CV_32F, maximum > 1.0 ? 1.0 / 255.0 : 1.0);
        }
        return normalized;
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

namespace xjw::image_matching
{

    metalign::Image makePlaMatchHctImage(const cv::Mat& grayImage, const cv::Mat& colorImage)
    {
        if (grayImage.empty() && colorImage.empty())
        {
            throw std::invalid_argument("PlaMatch-HCT requires a non-empty image");
        }

        metalign::Image result;
        if (!colorImage.empty())
        {
            cv::Mat bgr;
            if (colorImage.depth() != CV_8U)
            {
                double minimum = 0.0;
                double maximum = 0.0;
                cv::minMaxLoc(colorImage.reshape(1), &minimum, &maximum);
                colorImage.convertTo(bgr, CV_8U, maximum <= 1.0 ? 255.0 : 1.0);
            }
            else
            {
                bgr = colorImage;
            }
            if (bgr.channels() == 4)
            {
                cv::cvtColor(bgr, bgr, cv::COLOR_BGRA2BGR);
            }
            if (bgr.channels() != 3)
            {
                throw std::invalid_argument("PlaMatch-HCT color input must have 3 or 4 channels");
            }

            result.width = static_cast<std::size_t>(bgr.cols);
            result.height = static_cast<std::size_t>(bgr.rows);
            result.rgb.resize(result.width * result.height * 3U);
            result.gray.resize(result.width * result.height);
            for (int y = 0; y < bgr.rows; ++y)
            {
                const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
                for (int x = 0; x < bgr.cols; ++x)
                {
                    const cv::Vec3b pixel = row[x];
                    const std::size_t index = static_cast<std::size_t>(y) * result.width + static_cast<std::size_t>(x);
                    const std::uint8_t red = pixel[2];
                    const std::uint8_t green = pixel[1];
                    const std::uint8_t blue = pixel[0];
                    result.rgb[index * 3U] = red;
                    result.rgb[index * 3U + 1U] = green;
                    result.rgb[index * 3U + 2U] = blue;
                    const double luminance = static_cast<double>(red) * 0.299 + static_cast<double>(green) * 0.587 +
                                             static_cast<double>(blue) * 0.114;
                    const auto code = static_cast<std::uint8_t>(std::clamp(static_cast<int>(luminance), 0, 255));
                    result.gray[index] = static_cast<float>(code) / 255.0F;
                }
            }
            return result;
        }

        const cv::Mat gray = normalizedGray(grayImage);
        result.width = static_cast<std::size_t>(gray.cols);
        result.height = static_cast<std::size_t>(gray.rows);
        result.gray.resize(result.width * result.height);
        for (int row = 0; row < gray.rows; ++row)
        {
            std::memcpy(result.gray.data() + static_cast<std::size_t>(row) * result.width,
                        gray.ptr<float>(row),
                        result.width * sizeof(float));
        }
        return result;
    }

    metalign::Image makePlaMatchHctMask(const cv::Mat& validMask)
    {
        metalign::Image result;
        if (validMask.empty())
        {
            return result;
        }
        cv::Mat mask;
        if (validMask.channels() == 1)
        {
            mask = validMask;
        }
        else
        {
            cv::cvtColor(validMask, mask, validMask.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
        }
        cv::compare(mask, 0, mask, cv::CMP_GT);
        mask.convertTo(mask, CV_32F, 1.0 / 255.0);
        result.width = static_cast<std::size_t>(mask.cols);
        result.height = static_cast<std::size_t>(mask.rows);
        result.gray.resize(result.width * result.height);
        for (int row = 0; row < mask.rows; ++row)
        {
            std::memcpy(result.gray.data() + static_cast<std::size_t>(row) * result.width,
                        mask.ptr<float>(row),
                        result.width * sizeof(float));
        }
        return result;
    }

} // namespace xjw::image_matching

namespace metalign
{

    std::uint32_t jpeg_decoder_version_number()
    {
        return 0U;
    }

    Image load_image(const std::filesystem::path& path, bool)
    {
        const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("cannot read image: " + path.string());
        }
        return xjw::image_matching::makePlaMatchHctImage({}, image);
    }

    float sample_bilinear(const Image& image, double x, double y)
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

    Image upsample_highest(const Image& source, bool)
    {
        if (source.empty())
        {
            throw std::invalid_argument("cannot upsample an empty image");
        }
        Image result;
        result.width = source.width * 2U - 1U;
        result.height = source.height * 2U - 1U;
        result.gray.resize(result.width * result.height);
        result.focal_length_35mm = source.focal_length_35mm;
        for (std::size_t source_y = 0; source_y < source.height; ++source_y)
        {
            const std::size_t output_y = source_y * 2U;
            for (std::size_t x = 0; x + 1U < source.width; ++x)
            {
                const float left = source.at(x, source_y);
                result.at(x * 2U, output_y) = left;
                result.at(x * 2U + 1U, output_y) = (left + source.at(x + 1U, source_y)) * 0.5F;
            }
            result.at(result.width - 1U, output_y) = source.at(source.width - 1U, source_y);
        }
        for (std::size_t source_y = 0; source_y + 1U < source.height; ++source_y)
        {
            const std::size_t output_y = source_y * 2U;
            for (std::size_t x = 0; x < result.width; ++x)
            {
                result.at(x, output_y + 1U) = (result.at(x, output_y) + result.at(x, output_y + 2U)) * 0.5F;
            }
        }
        return result;
    }

    Image gaussian_blur(const Image& source, double sigma, bool)
    {
        if (source.empty() || sigma <= 0.0)
        {
            return source;
        }
        const int radius = std::max(1, static_cast<int>(4.0 * sigma + 1.0));
        std::vector<float> kernel(static_cast<std::size_t>(radius + 1));
        double sum = -1.0;
        const float sigmaSquared = static_cast<float>(sigma * sigma);
        const double exponent = -0.5 / static_cast<double>(sigmaSquared);
        for (int index = 0; index <= radius; ++index)
        {
            const float value = static_cast<float>(std::exp(exponent * index * index));
            kernel[static_cast<std::size_t>(index)] = value;
            sum += static_cast<double>(value + value);
        }
        for (float& value : kernel)
        {
            value = static_cast<float>(static_cast<double>(value) / sum);
        }

        Image result = source;
        result.gray.assign(source.width * source.height, 0.0F);
        std::vector<float> temporary(source.width * source.height, 0.0F);
        for (std::size_t y = 0; y < source.height; ++y)
        {
            for (std::size_t x = 0; x < source.width; ++x)
            {
                float value = source.at(x, y) * kernel[0];
                for (int offset = 1; offset <= radius; ++offset)
                {
                    const std::size_t left =
                        x < static_cast<std::size_t>(offset) ? 0 : x - static_cast<std::size_t>(offset);
                    const std::size_t right = std::min(source.width - 1, x + static_cast<std::size_t>(offset));
                    value += (source.at(left, y) + source.at(right, y)) * kernel[static_cast<std::size_t>(offset)];
                }
                temporary[y * source.width + x] = value;
            }
        }
        for (std::size_t y = 0; y < source.height; ++y)
        {
            for (std::size_t x = 0; x < source.width; ++x)
            {
                float value = temporary[y * source.width + x] * kernel[0];
                for (int offset = 1; offset <= radius; ++offset)
                {
                    const std::size_t top =
                        y < static_cast<std::size_t>(offset) ? 0 : y - static_cast<std::size_t>(offset);
                    const std::size_t bottom = std::min(source.height - 1, y + static_cast<std::size_t>(offset));
                    value += (temporary[top * source.width + x] + temporary[bottom * source.width + x]) *
                             kernel[static_cast<std::size_t>(offset)];
                }
                result.at(x, y) = value;
            }
        }
        return result;
    }

    Image resize_bilinear(const Image& source, std::size_t width, std::size_t height)
    {
        if (source.empty() || width == 0 || height == 0)
        {
            throw std::invalid_argument("invalid resize dimensions");
        }
        Image result;
        result.width = width;
        result.height = height;
        result.gray.resize(width * height);
        const double scaleX = static_cast<double>(source.width) / static_cast<double>(width);
        const double scaleY = static_cast<double>(source.height) / static_cast<double>(height);
        for (std::size_t y = 0; y < height; ++y)
        {
            for (std::size_t x = 0; x < width; ++x)
            {
                result.at(x, y) = sample_bilinear(source,
                                                  (static_cast<double>(x) + 0.5) * scaleX - 0.5,
                                                  (static_cast<double>(y) + 0.5) * scaleY - 0.5);
            }
        }
        return result;
    }

    Image downsample_half(const Image& source)
    {
        const Image blurred = gaussian_blur(source, 1.0);
        return resize_bilinear(
            blurred, std::max<std::size_t>(1, source.width / 2), std::max<std::size_t>(1, source.height / 2));
    }

    std::vector<std::filesystem::path> collect_images(const std::filesystem::path& directory)
    {
        if (!std::filesystem::is_directory(directory))
        {
            throw std::runtime_error("image directory does not exist: " + directory.string());
        }
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const std::string extension = lowercaseExtension(entry.path());
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".tif" ||
                extension == ".tiff")
            {
                paths.push_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
        return paths;
    }

} // namespace metalign
