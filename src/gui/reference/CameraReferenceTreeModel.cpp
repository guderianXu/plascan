#include "CameraReferenceTreeModel.h"

#include "project/ProjectMetadata.h"

#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QStandardItem>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::gui::reference
{
namespace
{

using ItemRow = QList<QStandardItem *>;

struct EstimatedCamera
{
    QString path;
    std::optional<camera_reference::Vector3d> center;
    std::optional<camera_reference::Vector3d> orientationYprDegrees;
};

struct ErrorStats
{
    void add(const camera_reference::Vector3d &value)
    {
        for (std::size_t index = 0; index < squaredSum.size(); ++index)
        {
            squaredSum[index] += value[index] * value[index];
        }
        ++count;
    }

    std::optional<camera_reference::Vector3d> rms() const
    {
        if (count == 0)
        {
            return std::nullopt;
        }
        camera_reference::Vector3d value{};
        for (std::size_t index = 0; index < squaredSum.size(); ++index)
        {
            value[index] = std::sqrt(squaredSum[index] / static_cast<double>(count));
        }
        return value;
    }

    camera_reference::Vector3d squaredSum{{0.0, 0.0, 0.0}};
    int count = 0;
};

ItemRow createRow(const QString &label)
{
    ItemRow row;
    row.reserve(CameraReferenceTreeModel::ColumnCount);
    for (int column = 0; column < CameraReferenceTreeModel::ColumnCount; ++column)
    {
        auto *item = new QStandardItem();
        item->setEditable(false);
        row.append(item);
    }
    row.front()->setText(label);
    return row;
}

void setNodeData(const ItemRow &row,
                 CameraReferenceTreeModel::NodeType nodeType,
                 const QString &imageUuid = {},
                 const QString &imagePath = {})
{
    for (QStandardItem *item : row)
    {
        item->setData(static_cast<int>(nodeType), CameraReferenceTreeModel::NodeTypeRole);
        item->setData(imageUuid, CameraReferenceTreeModel::ImageUuidRole);
        item->setData(imagePath, CameraReferenceTreeModel::ImagePathRole);
    }
}

void setVector(const ItemRow &row, const camera_reference::Vector3d &value, int firstColumn)
{
    for (int index = 0; index < 3; ++index)
    {
        row.at(firstColumn + index)->setData(value[static_cast<std::size_t>(index)], Qt::DisplayRole);
    }
}

void setRawReference(const ItemRow &row,
                     const camera_reference::RawCameraReference &reference)
{
    if (reference.position)
    {
        setVector(row, *reference.position, CameraReferenceTreeModel::XColumn);
    }
    if (reference.orientationYprDegrees)
    {
        setVector(row,
                  *reference.orientationYprDegrees,
                  CameraReferenceTreeModel::YawColumn);
    }
    if (reference.positionSigma)
    {
        const auto &sigma = *reference.positionSigma;
        if (!reference.horizontalSigmaMeters)
        {
            row.at(CameraReferenceTreeModel::HorizontalAccuracyColumn)->setData(
                std::hypot(sigma[0], sigma[1]), Qt::DisplayRole);
        }
        row.at(CameraReferenceTreeModel::VerticalAccuracyColumn)->setData(
            sigma[2], Qt::DisplayRole);
    }
    if (reference.horizontalSigmaMeters)
    {
        row.at(CameraReferenceTreeModel::HorizontalAccuracyColumn)->setData(
            *reference.horizontalSigmaMeters, Qt::DisplayRole);
    }
}

std::optional<camera_reference::Vector3d> vectorFromJson(const QJsonValue &value)
{
    const QJsonArray array = value.toArray();
    if (array.size() < 3)
    {
        return std::nullopt;
    }
    camera_reference::Vector3d result{};
    for (int index = 0; index < 3; ++index)
    {
        if (!array.at(index).isDouble() || !std::isfinite(array.at(index).toDouble()))
        {
            return std::nullopt;
        }
        result[static_cast<std::size_t>(index)] = array.at(index).toDouble();
    }
    return result;
}

std::optional<camera_reference::Vector3d> yprFromMatrix(
    const camera_reference::Matrix3d &rotation)
{
    for (const double entry : rotation)
    {
        if (!std::isfinite(entry))
        {
            return std::nullopt;
        }
    }
    const double pitch = std::asin(std::clamp(-rotation[6], -1.0, 1.0));
    double yaw = 0.0;
    double roll = 0.0;
    if (std::abs(std::cos(pitch)) > 1.0e-8)
    {
        yaw = std::atan2(rotation[3], rotation[0]);
        roll = std::atan2(rotation[7], rotation[8]);
    }
    else
    {
        yaw = std::atan2(-rotation[1], rotation[4]);
    }
    constexpr double radiansToDegrees = 180.0 / 3.14159265358979323846;
    return camera_reference::Vector3d{{yaw * radiansToDegrees,
                                       pitch * radiansToDegrees,
                                       roll * radiansToDegrees}};
}

std::optional<camera_reference::Vector3d> yprFromRotation(const QJsonValue &value)
{
    const QJsonArray rotation = value.toArray();
    if (rotation.size() < 9)
    {
        return std::nullopt;
    }
    camera_reference::Matrix3d matrix{};
    for (int index = 0; index < 9; ++index)
    {
        const QJsonValue entry = rotation.at(index);
        if (!entry.isDouble() || !std::isfinite(entry.toDouble()))
        {
            return std::nullopt;
        }
        matrix[static_cast<std::size_t>(index)] = entry.toDouble();
    }
    return yprFromMatrix(matrix);
}

QHash<QString, EstimatedCamera> estimatedCameras(const QJsonObject &metadata)
{
    QHash<QString, EstimatedCamera> result;
    const QJsonObject files = xjw::common::project::projectFilesRootObject(metadata);
    for (const QJsonValue &value : files.value(QStringLiteral("images")).toArray())
    {
        const QJsonObject image = value.toObject();
        const QString imageUuid = image.value(QStringLiteral("image_uuid")).toString().trimmed();
        if (imageUuid.isEmpty())
        {
            continue;
        }
        const QJsonObject camera = image.value(QStringLiteral("camera")).toObject();
        EstimatedCamera estimated;
        estimated.path = image.value(QStringLiteral("path")).toString();
        estimated.center = vectorFromJson(camera.value(QStringLiteral("C")));
        estimated.orientationYprDegrees = yprFromRotation(camera.value(QStringLiteral("R")));
        result.insert(imageUuid, estimated);
    }
    return result;
}

double wrappedAngleDifference(double estimated, double reference)
{
    double difference = std::fmod(estimated - reference + 180.0, 360.0);
    if (difference < 0.0)
    {
        difference += 360.0;
    }
    return difference - 180.0;
}

camera_reference::Vector3d difference(const camera_reference::Vector3d &estimated,
                                      const camera_reference::Vector3d &reference)
{
    return {{estimated[0] - reference[0],
             estimated[1] - reference[1],
             estimated[2] - reference[2]}};
}

camera_reference::Vector3d angleDifference(const camera_reference::Vector3d &estimated,
                                           const camera_reference::Vector3d &reference)
{
    return {{wrappedAngleDifference(estimated[0], reference[0]),
             wrappedAngleDifference(estimated[1], reference[1]),
             wrappedAngleDifference(estimated[2], reference[2])}};
}

QString statusForRecord(const camera_reference::CameraReferenceRecord &record,
                        const EstimatedCamera &estimated,
                        ReferenceDisplayMode mode)
{
    if (mode == ReferenceDisplayMode::Source)
    {
        return record.resolved.positionUsable || record.resolved.orientationUsable
            ? QStringLiteral("参考已就绪")
            : QStringLiteral("已导入，待参考设置");
    }
    if (!estimated.center && !estimated.orientationYprDegrees)
    {
        return QStringLiteral("未解算");
    }
    if (mode == ReferenceDisplayMode::Estimated)
    {
        return QStringLiteral("已解算");
    }
    if (!record.resolved.positionUsable && !record.resolved.orientationUsable)
    {
        return QStringLiteral("待坐标/姿态转换");
    }
    return QStringLiteral("可计算误差");
}

} // namespace

CameraReferenceTreeModel::CameraReferenceTreeModel(QObject *parent)
    : QStandardItemModel(parent)
{
}

void CameraReferenceTreeModel::setReferenceData(
    const camera_reference::CameraReferenceSet &referenceSet,
    const QJsonObject &projectMetadata,
    ReferenceDisplayMode mode)
{
    clear();
    const bool geographicSource = mode == ReferenceDisplayMode::Source
        && referenceSet.source().sourceCrs.contains(QStringLiteral("4979"));
    setHorizontalHeaderLabels({QStringLiteral("相机"),
                               geographicSource ? QStringLiteral("经度 (°)") : QStringLiteral("X (m)"),
                               geographicSource ? QStringLiteral("纬度 (°)") : QStringLiteral("Y (m)"),
                               QStringLiteral("Z (m)"),
                               QStringLiteral("偏航 (°)"),
                               QStringLiteral("俯仰 (°)"),
                               QStringLiteral("滚转 (°)"),
                               QStringLiteral("平面精度 (m)"),
                               QStringLiteral("高程精度 (m)"),
                               QStringLiteral("状态"),
                               QStringLiteral("启用")});

    ItemRow totalRow = createRow(QStringLiteral("总误差"));
    setNodeData(totalRow, NodeType::TotalError);
    appendRow(totalRow);

    const QHash<QString, EstimatedCamera> estimates = estimatedCameras(projectMetadata);
    ErrorStats positionStats;
    ErrorStats orientationStats;
    for (const camera_reference::CameraReferenceRecord &record : referenceSet.records())
    {
        const EstimatedCamera estimated = estimates.value(record.imageUuid);
        const QString imagePath = estimated.path.isEmpty()
            ? record.imagePathSnapshot
            : estimated.path;
        const QString label = QFileInfo(imagePath).fileName().isEmpty()
            ? record.sourceLabel
            : QFileInfo(imagePath).fileName();
        ItemRow row = createRow(label);
        setNodeData(row, NodeType::Camera, record.imageUuid, imagePath);

        if (mode == ReferenceDisplayMode::Source)
        {
            setRawReference(row, record.raw);
        }
        else if (mode == ReferenceDisplayMode::Estimated)
        {
            if (estimated.center)
            {
                setVector(row, *estimated.center, XColumn);
            }
            if (estimated.orientationYprDegrees)
            {
                setVector(row, *estimated.orientationYprDegrees, YawColumn);
            }
        }
        else
        {
            if (record.resolved.positionUsable
                && record.resolved.cameraCenterMeters
                && estimated.center)
            {
                const camera_reference::Vector3d delta = difference(
                    *estimated.center, *record.resolved.cameraCenterMeters);
                setVector(row, delta, XColumn);
                if (record.enabled)
                {
                    positionStats.add(delta);
                }
            }
            if (record.resolved.orientationUsable
                && record.resolved.rotationCameraToWorld
                && estimated.orientationYprDegrees)
            {
                if (const auto referenceOrientation =
                        yprFromMatrix(*record.resolved.rotationCameraToWorld))
                {
                    const camera_reference::Vector3d delta = angleDifference(
                        *estimated.orientationYprDegrees, *referenceOrientation);
                    setVector(row, delta, YawColumn);
                    if (record.enabled)
                    {
                        orientationStats.add(delta);
                    }
                }
            }
        }

        row.at(StatusColumn)->setText(statusForRecord(record, estimated, mode));
        row.at(EnabledColumn)->setText(record.enabled ? QStringLiteral("是") : QStringLiteral("否"));
        invisibleRootItem()->appendRow(row);
    }

    if (mode == ReferenceDisplayMode::Error)
    {
        if (const auto value = positionStats.rms())
        {
            setVector(totalRow, *value, XColumn);
        }
        if (const auto value = orientationStats.rms())
        {
            setVector(totalRow, *value, YawColumn);
        }
    }

    if (!referenceSet.unmatchedRecords().isEmpty())
    {
        ItemRow groupRow = createRow(
            QStringLiteral("未匹配源记录 (%1)").arg(referenceSet.unmatchedRecords().size()));
        setNodeData(groupRow, NodeType::UnmatchedGroup);
        QStandardItem *groupItem = groupRow.front();
        appendRow(groupRow);
        for (const camera_reference::UnmatchedCameraReferenceRecord &record
             : referenceSet.unmatchedRecords())
        {
            ItemRow row = createRow(record.sourceLabel);
            setNodeData(row, NodeType::UnmatchedRecord);
            if (mode == ReferenceDisplayMode::Source)
            {
                setRawReference(row, record.raw);
            }
            row.at(StatusColumn)->setText(record.reason);
            groupItem->appendRow(row);
        }
    }
}

} // namespace xjw::gui::reference
