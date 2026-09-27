#include "PointCloudInputPreparation.h"

#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <plapoint/io/ply_io.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <unordered_map>

namespace xjw::core::project
{
    namespace
    {

        QString normalizedPath(const QString& path)
        {
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
        }

        template <std::size_t Size> bool readFiniteArray(const QJsonValue& value, std::array<double, Size>* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonArray array = value.toArray();
            if (array.size() != static_cast<qsizetype>(Size))
            {
                return false;
            }
            for (std::size_t index = 0; index < Size; ++index)
            {
                const double element =
                    array.at(static_cast<qsizetype>(index)).toDouble(std::numeric_limits<double>::quiet_NaN());
                if (!std::isfinite(element))
                {
                    return false;
                }
                (*output)[index] = element;
            }
            return true;
        }

        bool readNonNegativeInteger(const QJsonValue& value, int* output)
        {
            if (!output || !value.isDouble())
            {
                return false;
            }
            const double number = value.toDouble(std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(number) || number < 0.0 || std::floor(number) != number ||
                number > static_cast<double>(std::numeric_limits<int>::max()))
            {
                return false;
            }
            *output = static_cast<int>(number);
            return true;
        }

        bool coordinatesMatchExactly(const std::array<float, 3>& left, const std::array<float, 3>& right)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const float left_value = left[static_cast<std::size_t>(axis)];
                const float right_value = right[static_cast<std::size_t>(axis)];
                if (!std::isfinite(left_value) || !std::isfinite(right_value) || left_value != right_value)
                {
                    return false;
                }
            }
            return true;
        }

        bool pointPosition(const QJsonObject& point, std::array<float, 3>* position)
        {
            if (!position)
            {
                return false;
            }
            const QJsonArray xyz = point.value(QStringLiteral("point_xyz")).toArray();
            if (xyz.size() < 3)
            {
                return false;
            }
            *position = {static_cast<float>(xyz.at(0).toDouble(std::numeric_limits<double>::quiet_NaN())),
                         static_cast<float>(xyz.at(1).toDouble(std::numeric_limits<double>::quiet_NaN())),
                         static_cast<float>(xyz.at(2).toDouble(std::numeric_limits<double>::quiet_NaN()))};
            return std::isfinite((*position)[0]) && std::isfinite((*position)[1]) && std::isfinite((*position)[2]);
        }

        bool loadSparsePlyPositions(const QString& sparseCloudPath,
                                    std::vector<std::array<float, 3>>* positions,
                                    QString* errorMessage)
        {
            if (!positions)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("内部错误：缺少稀疏 PLY 坐标输出对象");
                }
                return false;
            }

            try
            {
                const auto ply_cloud =
                    plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(sparseCloudPath));
                if (!ply_cloud || ply_cloud->size() == 0)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("正式空三稀疏 PLY 为空：%1").arg(sparseCloudPath);
                    }
                    return false;
                }

                positions->clear();
                positions->reserve(ply_cloud->size());
                const auto& points = ply_cloud->points();
                for (std::size_t index = 0; index < ply_cloud->size(); ++index)
                {
                    const auto row = static_cast<plamatrix::Index>(index);
                    const std::array<float, 3> position{
                        points.coeff(row, 0), points.coeff(row, 1), points.coeff(row, 2)};
                    if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2]))
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral("正式空三稀疏 PLY 含非有限坐标，顶点索引=%1：%2")
                                                .arg(index)
                                                .arg(sparseCloudPath);
                        }
                        return false;
                    }
                    positions->push_back(position);
                }
            }
            catch (const std::exception& exception)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法读取正式空三稀疏 PLY：%1（%2）")
                                        .arg(sparseCloudPath, QString::fromUtf8(exception.what()));
                }
                return false;
            }
            return true;
        }

        bool loadTrackedSparseCloud(const QString& sparseCloudPath,
                                    const QString& sidecarPath,
                                    const std::vector<xjw::mvs::CameraView>& views,
                                    xjw::mvs::SparseCloud* cloud,
                                    QString* errorMessage)
        {
            if (!cloud)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM track 输出指针为空");
                }
                return false;
            }

            if (sidecarPath.trimmed().isEmpty() || !QFileInfo::exists(sidecarPath))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM 点观测 sidecar 不存在：%1").arg(sidecarPath);
                }
                return false;
            }

            std::vector<std::array<float, 3>> ply_positions;
            if (!loadSparsePlyPositions(sparseCloudPath, &ply_positions, errorMessage))
            {
                return false;
            }

            QFile file(sidecarPath);
            if (!file.open(QIODevice::ReadOnly))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法读取 SfM 点观测 sidecar：%1").arg(sidecarPath);
                }
                return false;
            }
            QJsonParseError parse_error;
            const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
            if (parse_error.error != QJsonParseError::NoError || !document.isObject())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM 点观测 sidecar JSON 无效：%1（%2）")
                                        .arg(sidecarPath, parse_error.errorString());
                }
                return false;
            }

            const QJsonObject root = document.object();
            if (root.value(QStringLiteral("schema")).toString() !=
                QStringLiteral("plascan.sfm_sparse_points.v3"))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral(
                        "SfM 点观测 sidecar 必须使用 plascan.sfm_sparse_points.v3；旧 schema 不再兼容");
                }
                return false;
            }
            const QJsonArray root_images = root.value(QStringLiteral("images")).toArray();
            if (root_images.size() != static_cast<qsizetype>(views.size()) || root_images.isEmpty())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM v3 sidecar 的 images 必须完整覆盖当前 MVS 输入");
                }
                return false;
            }

            std::unordered_map<std::string, int> view_by_path;
            view_by_path.reserve(views.size());
            std::unordered_map<std::string, int> view_by_canonical_image_id;
            view_by_canonical_image_id.reserve(views.size());
            for (int view_index = 0; view_index < static_cast<int>(views.size()); ++view_index)
            {
                const std::string path =
                    normalizedPath(QString::fromStdString(views[static_cast<std::size_t>(view_index)].imagePath))
                        .toStdString();
                const std::shared_ptr<const placamera::FramePinholeModel>& camera =
                    views[static_cast<std::size_t>(view_index)].camera;
                if (path.empty() || !camera)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral(
                            "当前 MVS 输入的每个视图都必须带非空路径和 canonical 相机身份");
                    }
                    return false;
                }
                if (!view_by_path.emplace(path, view_index).second ||
                    !view_by_canonical_image_id.emplace(camera->imageId().value(), view_index).second)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("当前 MVS 输入包含重复路径或 canonical ImageId");
                    }
                    return false;
                }
            }

            std::unordered_map<int, int> view_by_image_id;
            xjw::mvs::SparseCloud tracked;
            const QJsonValue region_value = root.value(QStringLiteral("reconstruction_region"));
            if (!region_value.isUndefined() && !region_value.isNull())
            {
                const QJsonObject region = region_value.toObject();
                if (region.isEmpty() ||
                    !readFiniteArray(region.value(QStringLiteral("rotation")), &tracked.reconstructionRegionRotation) ||
                    !readFiniteArray(region.value(QStringLiteral("center")), &tracked.reconstructionRegionCenter) ||
                    !readFiniteArray(region.value(QStringLiteral("size")), &tracked.reconstructionRegionSize) ||
                    std::any_of(tracked.reconstructionRegionSize.cbegin(),
                                tracked.reconstructionRegionSize.cend(),
                                [](double value) { return value <= 0.0; }))
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("SfM 点观测 sidecar 的 reconstruction_region 无效：%1").arg(sidecarPath);
                    }
                    return false;
                }
                tracked.reconstructionRegionSpecified = true;
            }
            for (const QJsonValue& image_value : root_images)
            {
                if (!image_value.isObject())
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("SfM v3 sidecar 的 images 行必须是对象");
                    }
                    return false;
                }
                const QJsonObject image = image_value.toObject();
                int camera_index = -1;
                if (!readNonNegativeInteger(image.value(QStringLiteral("camera_index")), &camera_index))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("SfM v3 sidecar 的 images.camera_index 必须是非负整数");
                    }
                    return false;
                }
                const QJsonValue image_id_value = image.value(QStringLiteral("image_id"));
                const QJsonValue image_path_value = image.value(QStringLiteral("image_path"));
                if (!image_id_value.isString() || !image_path_value.isString())
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("SfM v3 sidecar 的 images 必须包含字符串 image_id 和 image_path");
                    }
                    return false;
                }
                const QString canonical_image_id = image_id_value.toString().trimmed();
                const QString raw_image_path = image_path_value.toString().trimmed();
                const std::string image_path = normalizedPath(raw_image_path).toStdString();
                if (canonical_image_id.isEmpty() || raw_image_path.isEmpty() || image_path.empty() ||
                    camera_index >= static_cast<int>(views.size()))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("SfM v3 sidecar 的 images 身份字段无效或超出当前 MVS 输入范围");
                    }
                    return false;
                }
                const auto found = view_by_path.find(image_path);
                const auto canonical = view_by_canonical_image_id.find(canonical_image_id.toStdString());
                if (canonical == view_by_canonical_image_id.end() || canonical->second != camera_index ||
                    (found != view_by_path.end() && found->second != camera_index))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral(
                            "SfM v3 sidecar 的 images camera_index 与 canonical ImageId 不一致，或已知路径指向另一影像");
                    }
                    return false;
                }
                const auto [existing, inserted] = view_by_image_id.emplace(camera_index, canonical->second);
                if (!inserted && existing->second != canonical->second)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("SfM v3 sidecar 的 images 表包含冲突的 camera_index 映射");
                    }
                    return false;
                }
            }
            if (view_by_image_id.size() != views.size())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM v3 sidecar 的 images 表未覆盖全部 camera_index");
                }
                return false;
            }

            std::array<float, 3> minimum{std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
            std::array<float, 3> maximum{std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest()};
            const QJsonValue sidecar_points_value = root.value(QStringLiteral("points"));
            if (!sidecar_points_value.isArray())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM v3 sidecar 的 points 必须是数组");
                }
                return false;
            }
            const QJsonArray sidecar_points = sidecar_points_value.toArray();
            if (sidecar_points.size() < static_cast<qsizetype>(ply_positions.size()))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM 点观测 sidecar 少于正式空三稀疏 PLY：ply=%1, sidecar=%2")
                                        .arg(ply_positions.size())
                                        .arg(sidecar_points.size());
                }
                return false;
            }

            std::vector<std::size_t> earliest_indices;
            earliest_indices.reserve(ply_positions.size());
            std::size_t sidecar_index = 0;
            for (std::size_t ply_index = 0; ply_index < ply_positions.size(); ++ply_index)
            {
                while (sidecar_index < static_cast<std::size_t>(sidecar_points.size()))
                {
                    const QJsonObject candidate = sidecar_points.at(static_cast<qsizetype>(sidecar_index)).toObject();
                    std::array<float, 3> candidate_position{};
                    ++sidecar_index;
                    if (pointPosition(candidate, &candidate_position) &&
                        coordinatesMatchExactly(ply_positions[ply_index], candidate_position))
                    {
                        earliest_indices.push_back(sidecar_index - 1);
                        break;
                    }
                }
                if (earliest_indices.size() != ply_index + 1)
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral(
                                "正式空三稀疏 PLY 不是 SfM 点观测 sidecar 的坐标可验证有序子集，PLY 顶点索引=%1")
                                .arg(ply_index);
                    }
                    return false;
                }
            }

            std::vector<std::size_t> latest_indices(ply_positions.size());
            sidecar_index = static_cast<std::size_t>(sidecar_points.size());
            for (std::size_t offset = 0; offset < ply_positions.size(); ++offset)
            {
                const std::size_t ply_index = ply_positions.size() - offset - 1;
                bool found = false;
                while (sidecar_index > 0)
                {
                    --sidecar_index;
                    const QJsonObject candidate = sidecar_points.at(static_cast<qsizetype>(sidecar_index)).toObject();
                    std::array<float, 3> candidate_position{};
                    if (pointPosition(candidate, &candidate_position) &&
                        coordinatesMatchExactly(ply_positions[ply_index], candidate_position))
                    {
                        latest_indices[ply_index] = sidecar_index;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("内部错误：反向验证有序 sidecar 子集失败，PLY 顶点索引=%1").arg(ply_index);
                    }
                    return false;
                }
            }
            if (earliest_indices != latest_indices)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("正式空三稀疏 PLY 对应多个合法 sidecar 有序映射，关联不唯一");
                }
                return false;
            }

            std::vector<QJsonObject> matched_points;
            matched_points.reserve(earliest_indices.size());
            for (const std::size_t index : earliest_indices)
            {
                matched_points.push_back(sidecar_points.at(static_cast<qsizetype>(index)).toObject());
            }

            tracked.points.reserve(matched_points.size());
            tracked.trackIds.reserve(matched_points.size());
            tracked.observingViewIndices.reserve(matched_points.size());
            for (const QJsonObject& point : matched_points)
            {
                std::array<float, 3> position{};
                if (!pointPosition(point, &position))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("内部错误：已匹配 sidecar 点缺少有效坐标");
                    }
                    return false;
                }

                const QJsonValue observations_value = point.value(QStringLiteral("observations"));
                if (!observations_value.isArray())
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("SfM v3 sidecar 的 points.observations 必须是数组");
                    }
                    return false;
                }
                std::vector<int> observations;
                for (const QJsonValue& observation_value : observations_value.toArray())
                {
                    if (!observation_value.isObject())
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral("SfM v3 sidecar 的 observations 必须是对象");
                        }
                        return false;
                    }
                    const QJsonObject observation = observation_value.toObject();
                    int camera_index = -1;
                    if (!readNonNegativeInteger(observation.value(QStringLiteral("camera_index")), &camera_index) ||
                        camera_index >= static_cast<int>(views.size()))
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral(
                                "SfM v3 sidecar 的 observation.camera_index 无效或超出当前 MVS 输入范围");
                        }
                        return false;
                    }
                    const QJsonValue observation_image_id = observation.value(QStringLiteral("image_id"));
                    if (!observation_image_id.isString())
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral("SfM v3 sidecar 的 observation.image_id 必须是字符串");
                        }
                        return false;
                    }
                    const QString canonical_image_id = observation_image_id.toString().trimmed();
                    const auto canonical = view_by_canonical_image_id.find(canonical_image_id.toStdString());
                    const auto mapped = view_by_image_id.find(camera_index);
                    if (canonical_image_id.isEmpty() || canonical == view_by_canonical_image_id.end() ||
                        mapped == view_by_image_id.end() || canonical->second != camera_index ||
                        mapped->second != camera_index)
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral(
                                "SfM v3 sidecar 的 observation 的 image_id 与 camera_index 身份字段不一致");
                        }
                        return false;
                    }
                    const QJsonValue observation_path_value = observation.value(QStringLiteral("image_path"));
                    if (!observation_path_value.isUndefined() && !observation_path_value.isString())
                    {
                        if (errorMessage)
                        {
                            *errorMessage = QStringLiteral("SfM v3 sidecar 的 observation.image_path 必须是字符串");
                        }
                        return false;
                    }
                    const QString observation_path = observation_path_value.toString();
                    if (!observation_path.isEmpty())
                    {
                        const auto path_match = view_by_path.find(normalizedPath(observation_path).toStdString());
                        if (path_match != view_by_path.end() && path_match->second != camera_index)
                        {
                            if (errorMessage)
                            {
                                *errorMessage = QStringLiteral("SfM 点观测的 camera_index 与相机身份不一致");
                            }
                            return false;
                        }
                    }
                    observations.push_back(camera_index);
                }
                std::sort(observations.begin(), observations.end());
                observations.erase(std::unique(observations.begin(), observations.end()), observations.end());
                if (observations.size() < 2)
                {
                    continue;
                }

                const std::size_t track_index = tracked.points.size();
                if (track_index >= std::numeric_limits<std::uint32_t>::max())
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("SfM track 数量超过 recovered depth 的 uint32 范围");
                    }
                    return false;
                }
                tracked.points.push_back(position);
                tracked.trackIds.push_back(static_cast<std::uint32_t>(track_index));
                tracked.observingViewIndices.push_back(std::move(observations));
                for (int axis = 0; axis < 3; ++axis)
                {
                    minimum[axis] = std::min(minimum[axis], position[axis]);
                    maximum[axis] = std::max(maximum[axis], position[axis]);
                }
            }

            if (tracked.points.empty())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("SfM 点观测 sidecar 没有至少被两个当前视图观测的有效 track：%1")
                                        .arg(sidecarPath);
                }
                return false;
            }
            tracked.minPt = minimum;
            tracked.maxPt = maximum;
            *cloud = std::move(tracked);
            return true;
        }

    } // namespace

    PointCloudInputPreparationResult preparePointCloudInput(const QString& sparseCloudPath,
                                                            const std::vector<xjw::mvs::CameraView>& views,
                                                            plapoint::ProcessingDevice processingDevice,
                                                            const QString& sparsePointSidecarPath)
    {
        (void)processingDevice;
        PointCloudInputPreparationResult prepared;
        if (sparseCloudPath.trimmed().isEmpty() || !QFileInfo::exists(sparseCloudPath))
        {
            prepared.errorMessage = QStringLiteral("正式空三稀疏点云不存在：%1").arg(sparseCloudPath);
            return prepared;
        }

        if (sparsePointSidecarPath.trimmed().isEmpty())
        {
            prepared.errorMessage =
                QStringLiteral("recovered 深度要求正式 SfM 点观测 sidecar（sfm_sparse_points.json）");
            return prepared;
        }
        QString tracked_error;
        if (!loadTrackedSparseCloud(sparseCloudPath, sparsePointSidecarPath, views, &prepared.cloud, &tracked_error))
        {
            prepared.errorMessage = tracked_error;
            return prepared;
        }
        prepared.ok = true;
        return prepared;
    }

} // namespace xjw::core::project
