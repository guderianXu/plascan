#include "DepthOverlayData.h"

#include "DepthFrameUtils.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xjw::gui::views
{
namespace
{

QString normalizedPath(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return {};
    }

    QString portable_path = path.trimmed();
    // Project metadata can contain paths written on another OS. Qt only
    // treats backslashes as native separators when the current host is Windows.
    portable_path.replace(QLatin1Char('\\'), QLatin1Char('/'));
    portable_path = QDir::cleanPath(portable_path);
    const bool is_windows_drive_path = portable_path.size() >= 3
        && portable_path.at(0).isLetter()
        && portable_path.at(1) == QLatin1Char(':')
        && portable_path.at(2) == QLatin1Char('/');
    if (is_windows_drive_path)
    {
        return portable_path;
    }

    const QFileInfo file_info(portable_path);
    return QDir::cleanPath(file_info.absoluteFilePath());
}

bool pathsMatch(const QString &left, const QString &right)
{
    const QString normalized_left = normalizedPath(left);
    const QString normalized_right = normalizedPath(right);
    if (normalized_left.isEmpty() || normalized_right.isEmpty())
    {
        return false;
    }

    const bool left_is_windows_drive = normalized_left.size() >= 3
        && normalized_left.at(0).isLetter()
        && normalized_left.at(1) == QLatin1Char(':')
        && normalized_left.at(2) == QLatin1Char('/');
    const bool right_is_windows_drive = normalized_right.size() >= 3
        && normalized_right.at(0).isLetter()
        && normalized_right.at(1) == QLatin1Char(':')
        && normalized_right.at(2) == QLatin1Char('/');
    if (left_is_windows_drive || right_is_windows_drive)
    {
        return left_is_windows_drive && right_is_windows_drive
            && normalized_left.compare(normalized_right, Qt::CaseInsensitive) == 0;
    }

#ifdef Q_OS_WIN
    return normalized_left.compare(normalized_right, Qt::CaseInsensitive) == 0;
#else
    return normalized_left == normalized_right;
#endif
}

int levelNumber(DepthOverlayLevel level)
{
    switch (level)
    {
    case DepthOverlayLevel::Level1:
        return 1;
    case DepthOverlayLevel::Level2:
        return 2;
    case DepthOverlayLevel::Level3:
        return 3;
    case DepthOverlayLevel::Final:
        return 0;
    }
    return 0;
}

DepthOverlayArtifact artifactFromObject(const QJsonObject &object,
                                        const QString &reference_image,
                                        int level,
                                        const QString &preview_key)
{
    DepthOverlayArtifact artifact;
    artifact.referenceImage = reference_image;
    artifact.rawDepthPath = object.value(QStringLiteral("raw_depth_path")).toString();
    artifact.validMaskPath = object.value(QStringLiteral("valid_mask_path")).toString();
    artifact.previewPath = object.value(preview_key).toString();
    artifact.level = level;
    return artifact;
}

bool isValidDepthPixel(const cv::Mat &depth,
                       const cv::Mat &valid_mask,
                       int row,
                       int column)
{
    const float value = depth.at<float>(row, column);
    return valid_mask.at<uchar>(row, column) == 255
        && std::isfinite(value)
        && value > 0.0f;
}

std::vector<float> sampleValidDepths(const cv::Mat &depth,
                                     const cv::Mat &valid_mask)
{
    std::size_t valid_count = 0;
    for (int row = 0; row < depth.rows; ++row)
    {
        for (int column = 0; column < depth.cols; ++column)
        {
            if (isValidDepthPixel(depth, valid_mask, row, column))
            {
                ++valid_count;
            }
        }
    }

    constexpr std::size_t maximum_samples = 1'000'000;
    const std::size_t sample_count = std::min(valid_count, maximum_samples);
    std::vector<float> samples;
    samples.reserve(sample_count);

    auto splitmix64 = [](std::uint64_t value)
    {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    };
    auto sampleIndex = [&](std::size_t sample_number)
    {
        const std::size_t stratum_begin = sample_number * valid_count / sample_count;
        const std::size_t stratum_end = (sample_number + 1) * valid_count / sample_count;
        const std::size_t stratum_size = std::max<std::size_t>(1, stratum_end - stratum_begin);
        return stratum_begin + static_cast<std::size_t>(splitmix64(sample_number) % stratum_size);
    };

    std::size_t sample_number = 0;
    std::size_t next_sample_index = sample_count == 0
        ? valid_count
        : sampleIndex(sample_number);
    std::size_t valid_index = 0;
    for (int row = 0; row < depth.rows; ++row)
    {
        for (int column = 0; column < depth.cols; ++column)
        {
            if (!isValidDepthPixel(depth, valid_mask, row, column))
            {
                continue;
            }
            if (valid_index == next_sample_index)
            {
                samples.push_back(depth.at<float>(row, column));
                ++sample_number;
                next_sample_index = sample_number < sample_count
                    ? sampleIndex(sample_number)
                    : valid_count;
            }
            ++valid_index;
        }
    }
    return samples;
}

float percentile(std::vector<float> *values, double fraction)
{
    if (!values || values->empty())
    {
        return 0.0f;
    }
    const std::size_t index = static_cast<std::size_t>(
        std::floor(fraction * static_cast<double>(values->size() - 1)));
    std::nth_element(values->begin(), values->begin() + index, values->end());
    return values->at(index);
}

bool hasCompatibleDepthInputs(const cv::Mat &depth, const cv::Mat &valid_mask)
{
    return !depth.empty()
        && depth.type() == CV_32FC1
        && !valid_mask.empty()
        && valid_mask.type() == CV_8UC1
        && depth.size() == valid_mask.size();
}

bool hasRequiredArtifactPaths(const DepthOverlayArtifact &artifact)
{
    return !artifact.rawDepthPath.trimmed().isEmpty()
        && !artifact.validMaskPath.trimmed().isEmpty();
}

QString resolvedArtifactPath(const QString &path, const QString &project_path)
{
    if (path.trimmed().isEmpty())
    {
        return {};
    }
    QString portable_path = path.trimmed();
    portable_path.replace(QLatin1Char('\\'), QLatin1Char('/'));
    portable_path = QDir::cleanPath(portable_path);
    const QFileInfo artifact_info(portable_path);
    if (artifact_info.isAbsolute() || project_path.trimmed().isEmpty())
    {
        return artifact_info.absoluteFilePath();
    }
    return QFileInfo(QFileInfo(project_path).absoluteDir(), portable_path).absoluteFilePath();
}

QImage buildIntensityBase(const QImage &source_image,
                          const cv::Mat &valid_mask,
                          const cv::Mat &depth)
{
    if (source_image.isNull())
    {
        return {};
    }

    QImage intensity = source_image.convertToFormat(QImage::Format_Grayscale8);
    if (intensity.size() != QSize(depth.cols, depth.rows))
    {
        intensity = intensity.scaled(
            depth.cols, depth.rows, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }
    for (int row = 0; row < depth.rows; ++row)
    {
        uchar *scan_line = intensity.scanLine(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (!isValidDepthPixel(depth, valid_mask, row, column))
            {
                scan_line[column] = 0;
            }
        }
    }
    return intensity;
}

} // namespace

std::optional<QJsonObject> resolveDepthOverlayRecord(
    const QJsonObject &project_metadata,
    const QString &image_path)
{
    const QJsonArray records = project_metadata.value(QStringLiteral("depth_map_results")).toArray();
    for (qsizetype record_index = records.size() - 1; record_index >= 0; --record_index)
    {
        const QJsonObject record = records.at(record_index).toObject();
        if (pathsMatch(record.value(QStringLiteral("ref_image")).toString(), image_path))
        {
            return record;
        }
    }
    return std::nullopt;
}

std::optional<DepthOverlayArtifact> resolveDepthOverlayArtifact(
    const QJsonObject &project_metadata,
    const QString &image_path,
    DepthOverlayLevel level)
{
    const QJsonArray records = project_metadata.value(QStringLiteral("depth_map_results")).toArray();
    for (qsizetype record_index = records.size() - 1; record_index >= 0; --record_index)
    {
        const QJsonObject record = records.at(record_index).toObject();
        const QString reference_image = record.value(QStringLiteral("ref_image")).toString();
        if (!pathsMatch(reference_image, image_path))
        {
            continue;
        }

        if (level == DepthOverlayLevel::Final)
        {
            const DepthOverlayArtifact artifact = artifactFromObject(
                record, reference_image, 0, QStringLiteral("depth_png"));
            return hasRequiredArtifactPaths(artifact)
                ? std::optional<DepthOverlayArtifact>(artifact)
                : std::nullopt;
        }

        const int requested_level = levelNumber(level);
        const QJsonArray pyramid_levels = record.value(QStringLiteral("pyramid_levels")).toArray();
        for (const QJsonValue &level_value : pyramid_levels)
        {
            const QJsonObject level_object = level_value.toObject();
            if (level_object.value(QStringLiteral("level")).toInt() == requested_level)
            {
                const DepthOverlayArtifact artifact = artifactFromObject(
                    level_object, reference_image, requested_level, QStringLiteral("preview_path"));
                return hasRequiredArtifactPaths(artifact)
                    ? std::optional<DepthOverlayArtifact>(artifact)
                    : std::nullopt;
            }
        }
        return std::nullopt;
    }

    return std::nullopt;
}

DepthOverlayArtifact resolveDepthOverlayArtifactPaths(
    const DepthOverlayArtifact &artifact,
    const QString &project_path)
{
    DepthOverlayArtifact resolved = artifact;
    resolved.rawDepthPath = resolvedArtifactPath(artifact.rawDepthPath, project_path);
    resolved.validMaskPath = resolvedArtifactPath(artifact.validMaskPath, project_path);
    resolved.previewPath = resolvedArtifactPath(artifact.previewPath, project_path);
    return resolved;
}

DepthOverlayAvailability resolveDepthOverlayAvailability(
    const QJsonObject &project_metadata,
    const QString &image_path,
    DepthOverlayLevel level,
    const QString &project_path)
{
    DepthOverlayAvailability result;
    const auto record = resolveDepthOverlayRecord(project_metadata, image_path);
    if (!record)
    {
        result.reason = QStringLiteral("当前照片没有深度图记录。");
        return result;
    }

    const auto artifact = resolveDepthOverlayArtifact(project_metadata, image_path, level);
    if (artifact)
    {
        const DepthOverlayArtifact resolved_artifact =
            resolveDepthOverlayArtifactPaths(*artifact, project_path);
        if (!QFileInfo::exists(resolved_artifact.rawDepthPath) ||
            !QFileInfo::exists(resolved_artifact.validMaskPath))
        {
            result.code = DepthOverlayAvailabilityCode::ArtifactMissing;
            result.reason = QStringLiteral("该级别的深度文件或有效掩码不存在，请重新生成深度图。");
            return result;
        }
        result.available = true;
        result.code = DepthOverlayAvailabilityCode::Available;
        return result;
    }

    const int requested_level = levelNumber(level);
    const QJsonArray levels = record->value(QStringLiteral("pyramid_levels")).toArray();
    const int active_level_count = record->value(
        QStringLiteral("pyramid_active_level_count")).toInt(levels.size());
    if (requested_level > 0 && requested_level > active_level_count)
    {
        result.code = DepthOverlayAvailabilityCode::NotComputedForResolution;
        const int width = record->value(QStringLiteral("grid_width")).toInt(0);
        const int height = record->value(QStringLiteral("grid_height")).toInt(0);
        const int minimum_short_side = record->value(
            QStringLiteral("pyramid_minimum_short_side")).toInt(160);
        result.reason = width > 0 && height > 0
            ? QStringLiteral("当前深度分辨率 %1×%2 只生成 %3 层；Level %4 未计算（每层短边至少 %5 像素）。")
                  .arg(width)
                  .arg(height)
                  .arg(active_level_count)
                  .arg(requested_level)
                  .arg(minimum_short_side)
            : QStringLiteral("当前深度只生成 %1 层；Level %2 未计算。")
                  .arg(active_level_count)
                  .arg(requested_level);
        return result;
    }

    result.code = DepthOverlayAvailabilityCode::NotPersisted;
    result.reason = requested_level > 0
        ? QStringLiteral("Level %1 已计算但没有保存可视化栅格，请重新生成并保存该级别。")
              .arg(requested_level)
        : QStringLiteral("最终深度栅格没有保存，请重新生成深度图。");
    return result;
}

QImage colorizeDepthOverlay(const cv::Mat &depth,
                            const cv::Mat &valid_mask,
                            int opacity)
{
    if (!hasCompatibleDepthInputs(depth, valid_mask))
    {
        return {};
    }

    std::vector<float> lower_samples = sampleValidDepths(depth, valid_mask);
    if (lower_samples.empty())
    {
        return {};
    }
    const float lower = percentile(&lower_samples, 0.02);
    const float upper = percentile(&lower_samples, 0.98);
    const float range = std::max(upper - lower, 1.0e-6f);

    cv::Mat grayscale_lut(1, 256, CV_8UC1);
    for (int value = 0; value < grayscale_lut.cols; ++value)
    {
        grayscale_lut.at<uchar>(0, value) = static_cast<uchar>(value);
    }
    cv::Mat color_lut;
    cv::applyColorMap(grayscale_lut, color_lut, cv::COLORMAP_TURBO);

    QImage overlay(depth.cols, depth.rows, QImage::Format_RGBA8888);
    if (overlay.isNull())
    {
        return {};
    }
    const uchar alpha = static_cast<uchar>(std::clamp(opacity, 0, 255));
    for (int row = 0; row < depth.rows; ++row)
    {
        uchar *rgba_row = overlay.scanLine(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (!isValidDepthPixel(depth, valid_mask, row, column))
            {
                const int offset = column * 4;
                rgba_row[offset] = 0;
                rgba_row[offset + 1] = 0;
                rgba_row[offset + 2] = 0;
                rgba_row[offset + 3] = 0;
                continue;
            }
            const float value = std::clamp(depth.at<float>(row, column), lower, upper);
            const int normalized_index = std::clamp(
                static_cast<int>(std::lround(255.0f * (value - lower) / range)), 0, 255);
            // Depth grows away from the camera, while the overlay convention is
            // red for near surfaces and blue for far surfaces.
            const int lut_index = 255 - normalized_index;
            const cv::Vec3b color = color_lut.at<cv::Vec3b>(0, lut_index);
            const int offset = column * 4;
            rgba_row[offset] = color[2];
            rgba_row[offset + 1] = color[1];
            rgba_row[offset + 2] = color[0];
            rgba_row[offset + 3] = alpha;
        }
    }

    return overlay;
}

DepthOverlayRenderResult renderDepthOverlay(
    const cv::Mat &depth,
    const cv::Mat &valid_mask,
    const DepthOverlayRenderOptions &options,
    const QImage &source_image)
{
    DepthOverlayRenderResult result;
    if (!hasCompatibleDepthInputs(depth, valid_mask))
    {
        result.errorMessage = QStringLiteral("深度矩阵或有效蒙版的类型、尺寸不匹配");
        return result;
    }

    result.overlay = colorizeDepthOverlay(depth, valid_mask, options.opacity);
    if (result.overlay.isNull())
    {
        result.errorMessage = QStringLiteral("深度图没有可显示的有限正深度像素");
        return result;
    }
    if (options.showIntensity)
    {
        result.intensityBase = buildIntensityBase(source_image, valid_mask, depth);
        if (result.intensityBase.isNull())
        {
            result.errorMessage = QStringLiteral("显示强度需要可读取的原始影像");
        }
    }
    return result;
}

DepthOverlayRenderResult loadDepthOverlay(
    const DepthOverlayArtifact &artifact,
    const DepthOverlayRenderOptions &options,
    const QImage &source_image)
{
    cv::Mat depth;
    const xjw::common::OperationResult load_status =
        xjw::core::project::loadDepthMatStorage(artifact.rawDepthPath, &depth);
    if (!load_status.ok)
    {
        return {{}, {}, load_status.errorMessage};
    }

    const cv::Mat valid_mask = xjw::common::io::readImage(
        artifact.validMaskPath, cv::IMREAD_GRAYSCALE);
    if (valid_mask.empty())
    {
        return {{}, {}, QStringLiteral("无法读取深度有效蒙版：%1").arg(artifact.validMaskPath)};
    }
    return renderDepthOverlay(depth, valid_mask, options, source_image);
}

} // namespace xjw::gui::views
