#include "PointCloudInputPreparation.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <cmath>
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

        bool loadTrackedSparseCloud(const QString& sidecarPath,
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

            std::unordered_map<std::string, int> view_by_path;
            view_by_path.reserve(views.size());
            for (int view_index = 0; view_index < static_cast<int>(views.size()); ++view_index)
            {
                view_by_path.emplace(
                    normalizedPath(QString::fromStdString(views[static_cast<std::size_t>(view_index)].imagePath))
                        .toStdString(),
                    view_index);
            }

            std::unordered_map<int, int> view_by_image_id;
            xjw::mvs::SparseCloud tracked;
            const QJsonObject root = document.object();
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
            const QJsonArray root_images = root.value(QStringLiteral("images")).toArray();
            for (const QJsonValue& image_value : root_images)
            {
                const QJsonObject image = image_value.toObject();
                const int image_id = image.value(QStringLiteral("image_id")).toInt(-1);
                const std::string image_path =
                    normalizedPath(image.value(QStringLiteral("image_path")).toString()).toStdString();
                const auto found = view_by_path.find(image_path);
                if (image_id >= 0 && found != view_by_path.end())
                {
                    view_by_image_id[image_id] = found->second;
                }
            }

            std::array<float, 3> minimum{std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
            std::array<float, 3> maximum{std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest()};
            const QJsonArray points = root.value(QStringLiteral("points")).toArray();
            tracked.points.reserve(static_cast<std::size_t>(points.size()));
            tracked.trackIds.reserve(static_cast<std::size_t>(points.size()));
            tracked.observingViewIndices.reserve(static_cast<std::size_t>(points.size()));
            for (const QJsonValue& point_value : points)
            {
                const QJsonObject point = point_value.toObject();
                const QJsonArray xyz = point.value(QStringLiteral("point_xyz")).toArray();
                if (xyz.size() < 3)
                {
                    continue;
                }
                std::array<float, 3> position{static_cast<float>(xyz.at(0).toDouble()),
                                              static_cast<float>(xyz.at(1).toDouble()),
                                              static_cast<float>(xyz.at(2).toDouble())};
                if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2]))
                {
                    continue;
                }

                std::vector<int> observations;
                for (const QJsonValue& observation_value : point.value(QStringLiteral("observations")).toArray())
                {
                    int image_id = -1;
                    int mapped_view_index = -1;
                    if (observation_value.isArray())
                    {
                        const QJsonArray observation = observation_value.toArray();
                        if (!observation.isEmpty())
                        {
                            image_id = observation.first().toInt(-1);
                        }
                    }
                    else if (observation_value.isObject())
                    {
                        const QJsonObject observation = observation_value.toObject();
                        image_id = observation.value(QStringLiteral("image_id")).toInt(-1);
                        const QString observation_path = observation.value(QStringLiteral("image_path")).toString();
                        if (!observation_path.isEmpty())
                        {
                            const auto path_match = view_by_path.find(normalizedPath(observation_path).toStdString());
                            if (path_match != view_by_path.end())
                            {
                                mapped_view_index = path_match->second;
                            }
                        }
                    }
                    const auto mapped = view_by_image_id.find(image_id);
                    if (mapped_view_index >= 0)
                    {
                        observations.push_back(mapped_view_index);
                    }
                    else if (mapped != view_by_image_id.end())
                    {
                        observations.push_back(mapped->second);
                    }
                    else if (root_images.isEmpty() && image_id >= 0 && image_id < static_cast<int>(views.size()))
                    {
                        observations.push_back(image_id);
                    }
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
        if (!loadTrackedSparseCloud(sparsePointSidecarPath, views, &prepared.cloud, &tracked_error))
        {
            prepared.errorMessage = tracked_error;
            return prepared;
        }
        prepared.ok = true;
        return prepared;
    }

} // namespace xjw::core::project
