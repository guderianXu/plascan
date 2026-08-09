#include "CameraReferenceSetJson.h"

#include <QJsonArray>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace xjw::camera_reference
{
namespace
{

class DecodeError : public std::runtime_error
{
public:
    explicit DecodeError(const QString &message)
        : std::runtime_error(message.toStdString())
    {
    }
};

void require(bool condition, const QString &message)
{
    if (!condition)
    {
        throw DecodeError(message);
    }
}

QJsonObject objectField(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    require(value.isObject(), QStringLiteral("字段 %1 必须是对象").arg(key));
    return value.toObject();
}

QJsonArray arrayField(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    require(value.isArray(), QStringLiteral("字段 %1 必须是数组").arg(key));
    return value.toArray();
}

QString stringField(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    require(value.isString(), QStringLiteral("字段 %1 必须是字符串").arg(key));
    return value.toString();
}

bool boolField(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    require(value.isBool(), QStringLiteral("字段 %1 必须是布尔值").arg(key));
    return value.toBool();
}

int intField(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    require(value.isDouble(), QStringLiteral("字段 %1 必须是整数").arg(key));
    const double number = value.toDouble();
    require(std::isfinite(number) && std::floor(number) == number,
            QStringLiteral("字段 %1 必须是整数").arg(key));
    require(number >= static_cast<double>(std::numeric_limits<int>::min())
                && number <= static_cast<double>(std::numeric_limits<int>::max()),
            QStringLiteral("字段 %1 超出整数范围").arg(key));
    return static_cast<int>(number);
}

template <std::size_t Size>
std::array<double, Size> numericArray(const QJsonValue &value, const QString &field)
{
    require(value.isArray(), QStringLiteral("字段 %1 必须是数组").arg(field));
    const QJsonArray array = value.toArray();
    require(array.size() == static_cast<int>(Size),
            QStringLiteral("字段 %1 必须包含 %2 个数值").arg(field).arg(Size));
    std::array<double, Size> result{};
    for (int index = 0; index < array.size(); ++index)
    {
        require(array.at(index).isDouble(),
                QStringLiteral("字段 %1[%2] 必须是数值").arg(field).arg(index));
        const double number = array.at(index).toDouble();
        require(std::isfinite(number),
                QStringLiteral("字段 %1[%2] 必须是有限值").arg(field).arg(index));
        result[static_cast<std::size_t>(index)] = number;
    }
    return result;
}

template <std::size_t Size>
std::optional<std::array<double, Size>> optionalNumericArray(const QJsonObject &object,
                                                              const QString &key)
{
    const QJsonValue value = object.value(key);
    require(!value.isUndefined(), QStringLiteral("缺少字段 %1").arg(key));
    if (value.isNull())
    {
        return std::nullopt;
    }
    return numericArray<Size>(value, key);
}

std::optional<double> optionalPositiveDouble(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    require(!value.isUndefined(), QStringLiteral("缺少字段 %1").arg(key));
    if (value.isNull())
    {
        return std::nullopt;
    }
    require(value.isDouble() && std::isfinite(value.toDouble()) && value.toDouble() > 0.0,
            QStringLiteral("字段 %1 必须是正有限数值或 null").arg(key));
    return value.toDouble();
}

CameraReferenceSource decodeSource(const QJsonObject &object)
{
    CameraReferenceSource source;
    source.kind = stringField(object, QStringLiteral("kind"));
    source.displayName = stringField(object, QStringLiteral("display_name"));
    source.contentSha256 = stringField(object, QStringLiteral("content_sha256"));
    source.sourceCrs = stringField(object, QStringLiteral("source_crs"));
    source.axisOrder = stringField(object, QStringLiteral("axis_order"));
    source.verticalDatum = stringField(object, QStringLiteral("vertical_datum"));
    source.verticalUnit = stringField(object, QStringLiteral("vertical_unit"));
    source.orientationConvention = stringField(object, QStringLiteral("orientation_convention"));
    source.angleUnit = stringField(object, QStringLiteral("angle_unit"));
    return source;
}

CameraReferenceSolverFrame decodeSolverFrame(const QJsonObject &object)
{
    CameraReferenceSolverFrame frame;
    frame.frameId = stringField(object, QStringLiteral("frame_id"));
    frame.kind = stringField(object, QStringLiteral("kind"));
    frame.unit = stringField(object, QStringLiteral("unit"));
    frame.originEcefMeters = numericArray<3>(object.value(QStringLiteral("origin_ecef_m")),
                                              QStringLiteral("origin_ecef_m"));
    frame.rotationSolverToEcef = numericArray<9>(
        object.value(QStringLiteral("rotation_solver_to_ecef")),
        QStringLiteral("rotation_solver_to_ecef"));
    frame.targetCrs = stringField(object, QStringLiteral("target_crs"));
    frame.targetCrsWkt = stringField(object, QStringLiteral("target_crs_wkt"));
    frame.normalizationHash = stringField(object, QStringLiteral("normalization_hash"));
    return frame;
}

RawCameraReference decodeRawReference(const QJsonObject &object)
{
    RawCameraReference reference;
    reference.position = optionalNumericArray<3>(object, QStringLiteral("position"));
    reference.positionSigma = optionalNumericArray<3>(object, QStringLiteral("position_sigma"));
    reference.positionSigmaFrame = stringField(
        object, QStringLiteral("position_sigma_frame"));
    reference.positionSigmaUnit = stringField(
        object, QStringLiteral("position_sigma_unit"));
    reference.horizontalSigmaMeters = optionalPositiveDouble(
        object, QStringLiteral("horizontal_sigma_m"));
    reference.orientationYprDegrees = optionalNumericArray<3>(
        object, QStringLiteral("orientation_ypr_deg"));
    reference.orientationSigmaDegrees = optionalNumericArray<3>(
        object, QStringLiteral("orientation_sigma_deg"));
    reference.timestamp = stringField(object, QStringLiteral("timestamp"));
    return reference;
}

ResolvedCameraReference decodeResolvedReference(const QJsonObject &object)
{
    ResolvedCameraReference reference;
    reference.cameraCenterMeters = optionalNumericArray<3>(object, QStringLiteral("camera_center_m"));
    reference.rotationCameraToWorld = optionalNumericArray<9>(
        object, QStringLiteral("rotation_camera_to_world"));
    reference.positionSigmaMeters = optionalNumericArray<3>(
        object, QStringLiteral("position_sigma_m"));
    reference.rotationSigmaDegrees = optionalNumericArray<3>(
        object, QStringLiteral("rotation_sigma_deg"));
    reference.positionUsable = boolField(object, QStringLiteral("position_usable"));
    reference.orientationUsable = boolField(object, QStringLiteral("orientation_usable"));
    reference.leverArmApplied = boolField(object, QStringLiteral("lever_arm_applied"));
    reference.frameId = stringField(object, QStringLiteral("frame_id"));
    reference.normalizationHash = stringField(object, QStringLiteral("normalization_hash"));
    reference.error = stringField(object, QStringLiteral("error"));
    return reference;
}

} // namespace

bool CameraReferenceSetJson::decode(const QJsonObject &object,
                                    CameraReferenceSet *referenceSet,
                                    QString *error)
{
    if (error)
    {
        error->clear();
    }
    if (!referenceSet)
    {
        if (error)
        {
            *error = QStringLiteral("CameraReferenceSet 输出指针为空");
        }
        return false;
    }

    try
    {
        require(stringField(object, QStringLiteral("format")) == QString::fromLatin1(Format),
                QStringLiteral("相机参考 format 无效"));
        const int schemaVersion = intField(object, QStringLiteral("schema_version"));
        require(schemaVersion == CameraReferenceSet::CurrentSchemaVersion,
                QStringLiteral("不支持的相机参考 schema_version: %1").arg(schemaVersion));

        CameraReferenceSet decoded;
        decoded._schemaVersion = schemaVersion;
        decoded._createdAt = QDateTime::fromString(
            stringField(object, QStringLiteral("created_at")), Qt::ISODateWithMs);
        decoded._updatedAt = QDateTime::fromString(
            stringField(object, QStringLiteral("updated_at")), Qt::ISODateWithMs);
        decoded._imageSetFingerprint = stringField(
            object, QStringLiteral("image_set_fingerprint"));
        decoded._source = decodeSource(objectField(object, QStringLiteral("source")));
        decoded._solverFrame = decodeSolverFrame(
            objectField(object, QStringLiteral("solver_frame")));

        for (const QJsonValue &value : arrayField(object, QStringLiteral("lever_arms")))
        {
            require(value.isObject(), QStringLiteral("lever_arms 元素必须是对象"));
            const QJsonObject item = value.toObject();
            CameraReferenceLeverArm leverArm;
            leverArm.id = stringField(item, QStringLiteral("id"));
            leverArm.vectorMeters = numericArray<3>(item.value(QStringLiteral("vector_m")),
                                                     QStringLiteral("vector_m"));
            leverArm.vectorFrame = stringField(item, QStringLiteral("vector_frame"));
            leverArm.vectorDirection = stringField(item, QStringLiteral("vector_direction"));
            leverArm.source = stringField(item, QStringLiteral("source"));
            decoded._leverArms.push_back(leverArm);
        }

        for (const QJsonValue &value : arrayField(object, QStringLiteral("records")))
        {
            require(value.isObject(), QStringLiteral("records 元素必须是对象"));
            const QJsonObject item = value.toObject();
            CameraReferenceRecord record;
            record.imageUuid = stringField(item, QStringLiteral("image_uuid"));
            record.imagePathSnapshot = stringField(item, QStringLiteral("image_path_snapshot"));
            record.sourceLabel = stringField(item, QStringLiteral("source_label"));
            record.sensorKey = stringField(item, QStringLiteral("sensor_key"));
            record.leverArmId = stringField(item, QStringLiteral("lever_arm_id"));
            record.enabled = boolField(item, QStringLiteral("enabled"));
            record.raw = decodeRawReference(objectField(item, QStringLiteral("raw")));
            record.resolved = decodeResolvedReference(objectField(item, QStringLiteral("resolved")));
            decoded._records.push_back(record);
        }

        for (const QJsonValue &value : arrayField(
                 object, QStringLiteral("unmatched_source_records")))
        {
            require(value.isObject(), QStringLiteral("unmatched_source_records 元素必须是对象"));
            const QJsonObject item = value.toObject();
            UnmatchedCameraReferenceRecord record;
            record.sourceLabel = stringField(item, QStringLiteral("source_label"));
            record.sensorKey = stringField(item, QStringLiteral("sensor_key"));
            record.leverArmId = stringField(item, QStringLiteral("lever_arm_id"));
            record.raw = decodeRawReference(objectField(item, QStringLiteral("raw")));
            record.reason = stringField(item, QStringLiteral("reason"));
            decoded._unmatchedRecords.push_back(record);
        }

        decoded.validateOrThrow();
        *referenceSet = std::move(decoded);
        return true;
    }
    catch (const std::exception &exception)
    {
        if (error)
        {
            *error = QString::fromUtf8(exception.what());
        }
        return false;
    }
}

} // namespace xjw::camera_reference
