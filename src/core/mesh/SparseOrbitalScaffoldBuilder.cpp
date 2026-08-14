#include "SparseOrbitalScaffoldBuilder.h"

#include "SparseSfmPointQualityReader.h"

#include "io/PathIO.h"

#include <QJsonDocument>
#include <QJsonObject>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/io/ply_io.h>

#include <algorithm>
#include <cmath>
#include <cwctype>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>

namespace xjw::mesh
{
namespace
{

using PlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
constexpr float kLengthEpsilon = 1.0e-12f;

void setError(std::string *error, const std::string &message)
{
    if (error)
    {
        *error = message;
    }
}

std::filesystem::path normalizedAbsolutePath(const std::filesystem::path &path)
{
    std::error_code error;
    const auto absolute_path = std::filesystem::absolute(path, error);
    if (error)
    {
        return {};
    }
    const auto normalized_path = std::filesystem::weakly_canonical(absolute_path, error);
    return error ? std::filesystem::path{} : normalized_path.lexically_normal();
}

bool hasPlyExtension(const std::filesystem::path &path)
{
    std::wstring extension = path.extension().wstring();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](wchar_t value)
                   {
                       return static_cast<wchar_t>(std::towlower(value));
                   });
    return extension == L".ply";
}

bool hasJsonExtension(const std::filesystem::path &path)
{
    std::wstring extension = path.extension().wstring();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](wchar_t value)
                   {
                       return static_cast<wchar_t>(std::towlower(value));
                   });
    return extension == L".json";
}

std::optional<std::filesystem::path> existingPly(
    const std::filesystem::path &path)
{
    if (path.empty() || !hasPlyExtension(path))
    {
        return std::nullopt;
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        return std::nullopt;
    }
    const auto normalized_path = normalizedAbsolutePath(path);
    return normalized_path.empty()
        ? std::nullopt
        : std::optional<std::filesystem::path>(normalized_path);
}

std::optional<std::filesystem::path> existingJson(
    const std::filesystem::path &path)
{
    if (path.empty() || !hasJsonExtension(path))
    {
        return std::nullopt;
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        return std::nullopt;
    }
    const auto normalized_path = normalizedAbsolutePath(path);
    return normalized_path.empty()
        ? std::nullopt
        : std::optional<std::filesystem::path>(normalized_path);
}

bool isInsideProjectRoot(const std::filesystem::path &root,
                         const std::filesystem::path &candidate)
{
    const auto comparison = xjw::common::io::comparePathsSafely(root, candidate);
    return comparison.valid
        && (comparison.equivalent || comparison.firstIsAncestorOfSecond);
}

std::optional<std::filesystem::path> sparsePathFromReport(
    const std::filesystem::path &report_path,
    const std::filesystem::path &project_root)
{
    QString read_error;
    const QByteArray bytes = xjw::common::io::readFileBytes(report_path, &read_error);
    if (!read_error.isEmpty() || bytes.isEmpty())
    {
        return std::nullopt;
    }

    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject())
    {
        return std::nullopt;
    }

    const QJsonObject object = document.object();
    QString path_text = object.value(QStringLiteral("sparse_cloud_path")).toString();
    if (path_text.isEmpty())
    {
        path_text = object.value(QStringLiteral("sparse_cloud_xyz")).toString();
    }
    if (path_text.isEmpty())
    {
        return std::nullopt;
    }

    std::filesystem::path candidate = xjw::common::io::toFilesystemPath(path_text);
    if (candidate.is_relative())
    {
        candidate = report_path.parent_path() / candidate;
    }
    const auto resolved = existingPly(candidate);
    if (!resolved || !isInsideProjectRoot(project_root, *resolved))
    {
        return std::nullopt;
    }
    return resolved;
}

float median(std::vector<float> values)
{
    if (values.empty())
    {
        return 0.0f;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const float upper = values[middle];
    if ((values.size() & 1U) != 0U)
    {
        return upper;
    }
    const float lower = *std::max_element(values.begin(), values.begin() + middle);
    return 0.5f * (lower + upper);
}

std::array<float, 3> robustCenter(
    const std::vector<std::array<float, 3>> &points)
{
    std::array<std::vector<float>, 3> coordinates;
    for (auto &coordinate : coordinates)
    {
        coordinate.reserve(points.size());
    }
    for (const auto &point : points)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            coordinates[static_cast<std::size_t>(axis)].push_back(
                point[static_cast<std::size_t>(axis)]);
        }
    }
    return {median(std::move(coordinates[0])),
            median(std::move(coordinates[1])),
            median(std::move(coordinates[2]))};
}

float squaredDistance(const std::array<float, 3> &point,
                      const std::array<float, 3> &center)
{
    const float dx = point[0] - center[0];
    const float dy = point[1] - center[1];
    const float dz = point[2] - center[2];
    return dx * dx + dy * dy + dz * dz;
}

std::vector<std::array<float, 3>> rejectRadialOutliers(
    const std::vector<std::array<float, 3>> &points,
    const SparseOrbitalScaffoldOptions &options,
    std::size_t *rejected_count)
{
    *rejected_count = 0;
    if (points.size() < options.minimumOutlierSampleCount)
    {
        return points;
    }

    const auto center = robustCenter(points);
    std::vector<float> radii;
    radii.reserve(points.size());
    for (const auto &point : points)
    {
        radii.push_back(std::sqrt(squaredDistance(point, center)));
    }
    const float median_radius = median(radii);
    if (!std::isfinite(median_radius) || median_radius <= kLengthEpsilon)
    {
        return points;
    }

    std::vector<float> deviations;
    deviations.reserve(radii.size());
    for (const float radius : radii)
    {
        deviations.push_back(std::abs(radius - median_radius));
    }
    constexpr float kMadToSigma = 1.4826022185f;
    const float robust_sigma = kMadToSigma * median(std::move(deviations));
    const float maximum_radius = std::max(
        median_radius * options.maximumRadiusMedianFactor,
        median_radius + options.radialMadMultiplier * robust_sigma);

    std::vector<std::array<float, 3>> filtered;
    filtered.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        if (radii[index] <= maximum_radius)
        {
            filtered.push_back(points[index]);
        }
    }
    *rejected_count = points.size() - filtered.size();
    return filtered;
}

PlaCloud toPlaCloud(const std::vector<std::array<float, 3>> &points)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> matrix(
        static_cast<plamatrix::Index>(points.size()), 3);
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        for (int axis = 0; axis < 3; ++axis)
        {
            matrix.setValue(row, axis, points[index][static_cast<std::size_t>(axis)]);
        }
    }
    return PlaCloud(std::move(matrix));
}

std::vector<std::array<float, 3>> fromPlaCloud(const PlaCloud &cloud)
{
    std::vector<std::array<float, 3>> points(cloud.size());
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        const auto point = cloud[index];
        points[index] = {point.x(), point.y(), point.z()};
    }
    return points;
}

float boundingDiagonal(const std::vector<std::array<float, 3>> &points)
{
    std::array<float, 3> minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    std::array<float, 3> maximum{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    for (const auto &point : points)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[static_cast<std::size_t>(axis)] = std::min(
                minimum[static_cast<std::size_t>(axis)],
                point[static_cast<std::size_t>(axis)]);
            maximum[static_cast<std::size_t>(axis)] = std::max(
                maximum[static_cast<std::size_t>(axis)],
                point[static_cast<std::size_t>(axis)]);
        }
    }
    return std::sqrt(squaredDistance(minimum, maximum));
}

std::vector<std::array<float, 3>> downsample(
    const std::vector<std::array<float, 3>> &points,
    const SparseOrbitalScaffoldOptions &options)
{
    if (points.empty())
    {
        return {};
    }

    float leaf_size = options.voxelSize;
    if (leaf_size <= 0.0f)
    {
        const float diagonal = boundingDiagonal(points);
        if (options.voxelSizeBoundingDiagonalFraction > 0.0f)
        {
            leaf_size = diagonal *
                options.voxelSizeBoundingDiagonalFraction;
        }
        else if (points.size() > options.maximumPointCount)
        {
            leaf_size = diagonal / std::sqrt(static_cast<float>(
                std::max<std::size_t>(options.maximumPointCount, 1)));
        }
        else
        {
            return points;
        }
    }
    if (!std::isfinite(leaf_size) || leaf_size <= kLengthEpsilon)
    {
        return points;
    }

    const PlaCloud input = toPlaCloud(points);
    PlaCloud output = plapoint::voxelDownsample(input, leaf_size);
    if (options.voxelSize <= 0.0f)
    {
        for (int attempt = 0;
             output.size() > options.maximumPointCount && attempt < 12;
             ++attempt)
        {
            leaf_size *= 1.25f;
            output = plapoint::voxelDownsample(input, leaf_size);
        }
    }
    auto sampled = fromPlaCloud(output);
    if (sampled.size() > options.maximumPointCount)
    {
        std::vector<std::array<float, 3>> capped;
        capped.reserve(options.maximumPointCount);
        for (std::size_t index = 0; index < options.maximumPointCount; ++index)
        {
            capped.push_back(sampled[index * sampled.size()
                                     / options.maximumPointCount]);
        }
        return capped;
    }
    return sampled;
}

bool validateOptions(const SparseOrbitalScaffoldOptions &options,
                     std::string *error)
{
    if (options.minimumPointCount < 3
        || options.maximumPointCount < options.minimumPointCount
        || options.minimumOutlierSampleCount < 3
        || !std::isfinite(options.voxelSize) || options.voxelSize < 0.0f
        || !std::isfinite(options.voxelSizeBoundingDiagonalFraction)
        || options.voxelSizeBoundingDiagonalFraction < 0.0f
        || !std::isfinite(options.maximumRadiusMedianFactor)
        || options.maximumRadiusMedianFactor <= 1.0f
        || !std::isfinite(options.radialMadMultiplier)
        || options.radialMadMultiplier <= 0.0f
        || options.minimumTrackLength < 2
        || options.statisticalOutlierNeighborCount < 3
        || !std::isfinite(options.statisticalOutlierStdDevMultiplier)
        || options.statisticalOutlierStdDevMultiplier < 0.1f
        || !std::isfinite(options.maximumRmsReprojectionPixels)
        || options.maximumRmsReprojectionPixels <= 0.0f
        || !std::isfinite(options.minimumTriangulationAngleDegrees)
        || options.minimumTriangulationAngleDegrees <= 0.0f)
    {
        setError(error, "稀疏轨道骨架参数无效");
        return false;
    }
    return true;
}

bool coordinatesAgree(const std::array<float, 3> &left,
                      const std::array<float, 3> &right)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        const float left_value = left[static_cast<std::size_t>(axis)];
        const float right_value = right[static_cast<std::size_t>(axis)];
        const float tolerance = 1.0e-4f
            * std::max({1.0f, std::abs(left_value), std::abs(right_value)});
        if (!std::isfinite(left_value) || !std::isfinite(right_value)
            || std::abs(left_value - right_value) > tolerance)
        {
            return false;
        }
    }
    return true;
}

} // namespace

std::filesystem::path SparseOrbitalScaffoldBuilder::resolveSparsePly(
    const std::filesystem::path &mvsOutputPath,
    const std::filesystem::path &explicitSparsePly,
    std::string *error)
{
    setError(error, {});
    if (mvsOutputPath.empty())
    {
        setError(error, "MVS 输出路径为空");
        return {};
    }

    std::filesystem::path output_directory = mvsOutputPath;
    std::error_code status_error;
    if (std::filesystem::is_regular_file(output_directory, status_error))
    {
        output_directory = output_directory.parent_path();
    }
    output_directory = normalizedAbsolutePath(output_directory);
    if (output_directory.empty())
    {
        setError(error, "无法规范化 MVS 输出路径");
        return {};
    }

    if (!explicitSparsePly.empty())
    {
        std::filesystem::path candidate = explicitSparsePly;
        if (candidate.is_relative())
        {
            candidate = output_directory / candidate;
        }
        if (const auto resolved = existingPly(candidate))
        {
            return *resolved;
        }
        setError(error, "显式稀疏点云不是可读取的 PLY 文件: "
            + xjw::common::io::toUtf8Path(candidate));
        return {};
    }

    const std::filesystem::path project_root = output_directory.parent_path();
    const std::vector<std::filesystem::path> candidates{
        output_directory / "sfm_sparse.ply",
        output_directory / "sparse_cloud.ply",
        project_root / "assets" / "aerial_triangulation"
            / "sfm_sparse" / "sfm_sparse.ply",
        project_root / "assets" / "aerial_triangulation"
            / "sfm_sparse" / "sparse_cloud.ply",
        project_root / "assets" / "sfm_sparse" / "sfm_sparse.ply",
        project_root / "sparse" / "sfm_sparse.ply",
        project_root / "sparse" / "sparse_cloud.ply"};
    for (const auto &candidate : candidates)
    {
        if (const auto resolved = existingPly(candidate))
        {
            return *resolved;
        }
    }

    const std::filesystem::path report_path = project_root / "assets"
        / "reports" / "aerial_triangulation_sfm_report.json";
    if (const auto report_candidate = sparsePathFromReport(report_path, project_root))
    {
        return *report_candidate;
    }

    setError(error, "未找到 SfM 稀疏 PLY；请显式指定稀疏点云路径");
    return {};
}

std::filesystem::path SparseOrbitalScaffoldBuilder::resolveSparsePointsJson(
    const std::filesystem::path &mvsOutputPath,
    const std::filesystem::path &sparsePly,
    const std::filesystem::path &explicitPointsJson,
    std::string *error)
{
    setError(error, {});
    if (mvsOutputPath.empty() || sparsePly.empty())
    {
        setError(error, "MVS 输出路径或 SfM 稀疏 PLY 路径为空");
        return {};
    }

    std::filesystem::path output_directory = mvsOutputPath;
    std::error_code status_error;
    if (std::filesystem::is_regular_file(output_directory, status_error))
    {
        output_directory = output_directory.parent_path();
    }
    output_directory = normalizedAbsolutePath(output_directory);
    if (output_directory.empty())
    {
        setError(error, "无法规范化 MVS 输出路径");
        return {};
    }

    if (!explicitPointsJson.empty())
    {
        std::filesystem::path candidate = explicitPointsJson;
        if (candidate.is_relative())
        {
            candidate = output_directory / candidate;
        }
        if (const auto resolved = existingJson(candidate))
        {
            return *resolved;
        }
        setError(error, "显式 SfM 点质量 sidecar 不是可读取的 JSON 文件: "
            + xjw::common::io::toUtf8Path(candidate));
        return {};
    }

    const std::filesystem::path project_root = output_directory.parent_path();
    const std::filesystem::path sparse_stem_sidecar = sparsePly.parent_path()
        / (sparsePly.stem().wstring() + L"_points.json");
    const std::vector<std::filesystem::path> candidates{
        sparse_stem_sidecar,
        sparsePly.parent_path() / "sparse_cloud_points.json",
        project_root / "assets" / "aerial_triangulation"
            / "sfm_sparse" / "sfm_sparse_points.json",
        project_root / "assets" / "aerial_triangulation"
            / "sfm_sparse" / "sparse_cloud_points.json",
        output_directory / "sfm_sparse_points.json",
        output_directory / "sparse_cloud_points.json"};
    for (const auto &candidate : candidates)
    {
        if (const auto resolved = existingJson(candidate))
        {
            return *resolved;
        }
    }

    setError(error,
             "轨道自动骨架需要 SfM 点质量 sidecar（*_points.json）；"
             "仅 PLY 无法安全过滤低质量轨迹");
    return {};
}

SparseOrbitalScaffoldResult SparseOrbitalScaffoldBuilder::build(
    const std::filesystem::path &mvsOutputPath,
    const std::filesystem::path &explicitSparsePly,
    const std::filesystem::path &explicitPointsJson,
    const SparseOrbitalScaffoldOptions &options)
{
    SparseOrbitalScaffoldResult result;
    if (!validateOptions(options, &result.error))
    {
        return result;
    }
    result.sourcePath = resolveSparsePly(
        mvsOutputPath, explicitSparsePly, &result.error);
    if (result.sourcePath.empty())
    {
        return result;
    }
    result.qualitySidecarPath = resolveSparsePointsJson(
        mvsOutputPath, result.sourcePath, explicitPointsJson, &result.error);
    if (result.qualitySidecarPath.empty())
    {
        return result;
    }

    std::shared_ptr<PlaCloud> cloud;
    try
    {
        cloud = plapoint::io::readPly<float>(
            xjw::common::io::toNativeNarrowPath(result.sourcePath));
    }
    catch (const std::exception &exception)
    {
        result.error = "读取 SfM 稀疏 PLY 失败: " + std::string(exception.what());
        return result;
    }
    if (!cloud)
    {
        result.error = "读取 SfM 稀疏 PLY 失败: reader 返回空点云";
        return result;
    }

    result.statistics.inputPointCount = cloud->size();
    const auto quality_data = SparseSfmPointQualityReader::read(
        result.qualitySidecarPath);
    if (!quality_data.error.empty())
    {
        result.error = quality_data.error;
        return result;
    }
    result.statistics.sidecarPointCount = quality_data.points.size();
    if (quality_data.points.size() != cloud->size())
    {
        result.error = "SfM PLY 与点质量 sidecar 点数不一致: ply="
            + std::to_string(cloud->size()) + ", sidecar="
            + std::to_string(quality_data.points.size());
        return result;
    }

    std::vector<std::array<float, 3>> finite_points;
    finite_points.reserve(cloud->size());
    for (std::size_t index = 0; index < cloud->size(); ++index)
    {
        const auto &quality = quality_data.points[index];
        const auto ply_point = (*cloud)[index];
        const std::array<float, 3> ply_value{
            ply_point.x(), ply_point.y(), ply_point.z()};
        if (!std::isfinite(ply_value[0]) || !std::isfinite(ply_value[1])
            || !std::isfinite(ply_value[2]))
        {
            ++result.statistics.nonFiniteRejectedCount;
            continue;
        }
        if (!quality.hasPoint || !std::isfinite(quality.point[0])
            || !std::isfinite(quality.point[1]) || !std::isfinite(quality.point[2]))
        {
            ++result.statistics.nonFiniteRejectedCount;
            continue;
        }
        if (!coordinatesAgree(ply_value, quality.point))
        {
            result.error = "SfM PLY 与点质量 sidecar 坐标不一致，索引="
                + std::to_string(index);
            return result;
        }
        if (!quality.hasRequiredQuality)
        {
            ++result.statistics.missingQualityRejectedCount;
            continue;
        }
        if (quality.trackLength < options.minimumTrackLength
            || quality.rmsReprojectionPixels
                > options.maximumRmsReprojectionPixels
            || quality.triangulationAngleDegrees
                < options.minimumTriangulationAngleDegrees)
        {
            ++result.statistics.qualityRejectedCount;
            continue;
        }
        finite_points.push_back(quality.point);
    }

    auto filtered_points = rejectRadialOutliers(
        finite_points, options, &result.statistics.radialOutlierRejectedCount);
    if (filtered_points.size() < options.minimumPointCount)
    {
        result.error = "稀疏点云过滤后点数不足: "
            + std::to_string(filtered_points.size());
        return result;
    }

    if (options.enableStatisticalOutlierRemoval &&
        filtered_points.size() >= static_cast<std::size_t>(
            options.statisticalOutlierNeighborCount + 2))
    {
        const PlaCloud statistical_input = toPlaCloud(filtered_points);
        std::vector<int> removed_indices;
        try
        {
            const PlaCloud statistical_output =
                plapoint::statisticalOutlierRemoval(
                    statistical_input,
                    options.statisticalOutlierNeighborCount,
                    options.statisticalOutlierStdDevMultiplier,
                    &removed_indices);
            const auto statistical_points = fromPlaCloud(statistical_output);
            if (statistical_points.size() >= options.minimumPointCount)
            {
                filtered_points = statistical_points;
                result.statistics.statisticalOutlierRejectedCount =
                    removed_indices.size();
            }
        }
        catch (const std::exception &exception)
        {
            result.error = "稀疏点云统计离群过滤失败: " +
                std::string(exception.what());
            return result;
        }
    }

    result.points = downsample(filtered_points, options);
    result.statistics.downsampledPointCount = filtered_points.size() - result.points.size();
    result.statistics.outputPointCount = result.points.size();
    if (result.points.size() < options.minimumPointCount)
    {
        result.error = "稀疏点云下采样后点数不足: "
            + std::to_string(result.points.size());
        result.points.clear();
        return result;
    }

    result.robustCenter = robustCenter(result.points);
    result.normals.resize(result.points.size());
    for (std::size_t index = 0; index < result.points.size(); ++index)
    {
        const auto &point = result.points[index];
        std::array<float, 3> normal{
            point[0] - result.robustCenter[0],
            point[1] - result.robustCenter[1],
            point[2] - result.robustCenter[2]};
        float normal_length = std::sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (!std::isfinite(normal_length) || normal_length <= kLengthEpsilon)
        {
            normal = {0.0f, 0.0f, 1.0f};
            normal_length = 1.0f;
        }
        for (float &component : normal)
        {
            component /= normal_length;
        }
        result.normals[index] = normal;
    }
    return result;
}

} // namespace xjw::mesh
