#include "CameraReferenceSet.h"

#include <QSet>

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::camera_reference
{
namespace
{

template <std::size_t Size>
bool finiteArray(const std::array<double, Size> &values)
{
    return std::all_of(values.cbegin(), values.cend(), [](double value)
    {
        return std::isfinite(value);
    });
}

bool positiveFinite(const Vector3d &values)
{
    return std::all_of(values.cbegin(), values.cend(), [](double value)
    {
        return std::isfinite(value) && value > 0.0;
    });
}

bool validRotation(const Matrix3d &rotation)
{
    if (!finiteArray(rotation))
    {
        return false;
    }
    constexpr double tolerance = 1e-6;
    for (int row = 0; row < 3; ++row)
    {
        for (int other = 0; other < 3; ++other)
        {
            double dot = 0.0;
            for (int column = 0; column < 3; ++column)
            {
                dot += rotation[static_cast<std::size_t>(row * 3 + column)]
                    * rotation[static_cast<std::size_t>(other * 3 + column)];
            }
            const double expected = row == other ? 1.0 : 0.0;
            if (std::abs(dot - expected) > tolerance)
            {
                return false;
            }
        }
    }
    const double determinant =
        rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
        - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
        + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
    return std::abs(determinant - 1.0) <= tolerance;
}

bool hasResolvedValues(const ResolvedCameraReference &reference)
{
    return reference.cameraCenterMeters.has_value()
        || reference.rotationCameraToWorld.has_value()
        || reference.positionSigmaMeters.has_value()
        || reference.rotationSigmaDegrees.has_value()
        || reference.positionUsable
        || reference.orientationUsable;
}

void require(bool condition, const QString &message)
{
    if (!condition)
    {
        throw CameraReferenceModelError(message);
    }
}

void validateRawReference(const RawCameraReference &raw,
                          const CameraReferenceSource &source,
                          const QString &label)
{
    if (raw.position)
    {
        require(finiteArray(*raw.position), QStringLiteral("原始位置无效: %1").arg(label));
    }
    if (raw.positionSigma)
    {
        require(raw.position.has_value() && positiveFinite(*raw.positionSigma),
                QStringLiteral("原始位置精度无效: %1").arg(label));
        require(!raw.positionSigmaFrame.trimmed().isEmpty()
                    && !raw.positionSigmaUnit.trimmed().isEmpty(),
                QStringLiteral("原始位置精度缺少坐标系或单位: %1").arg(label));
    }
    else
    {
        require(raw.positionSigmaFrame.trimmed().isEmpty()
                    && raw.positionSigmaUnit.trimmed().isEmpty(),
                QStringLiteral("原始位置精度元数据没有对应数值: %1").arg(label));
    }
    if (raw.horizontalSigmaMeters)
    {
        require(raw.position.has_value()
                    && std::isfinite(*raw.horizontalSigmaMeters)
                    && *raw.horizontalSigmaMeters > 0.0,
                QStringLiteral("原始平面精度无效: %1").arg(label));
    }
    if (raw.orientationYprDegrees)
    {
        require(finiteArray(*raw.orientationYprDegrees),
                QStringLiteral("原始姿态无效: %1").arg(label));
        require(!source.orientationConvention.trimmed().isEmpty()
                    && !source.angleUnit.trimmed().isEmpty(),
                QStringLiteral("原始姿态缺少约定或角度单位: %1").arg(label));
    }
    if (raw.orientationSigmaDegrees)
    {
        require(raw.orientationYprDegrees.has_value()
                    && positiveFinite(*raw.orientationSigmaDegrees),
                QStringLiteral("原始姿态精度无效: %1").arg(label));
    }
}

} // namespace

CameraReferenceSet::CameraReferenceSet()
    : _createdAt(QDateTime::currentDateTimeUtc())
    , _updatedAt(_createdAt)
{
}

int CameraReferenceSet::schemaVersion() const noexcept
{
    return _schemaVersion;
}

QDateTime CameraReferenceSet::createdAt() const
{
    return _createdAt;
}

QDateTime CameraReferenceSet::updatedAt() const
{
    return _updatedAt;
}

QString CameraReferenceSet::imageSetFingerprint() const
{
    return _imageSetFingerprint;
}

const CameraReferenceSource &CameraReferenceSet::source() const noexcept
{
    return _source;
}

const CameraReferenceSolverFrame &CameraReferenceSet::solverFrame() const noexcept
{
    return _solverFrame;
}

const QVector<CameraReferenceLeverArm> &CameraReferenceSet::leverArms() const noexcept
{
    return _leverArms;
}

const QVector<CameraReferenceRecord> &CameraReferenceSet::records() const noexcept
{
    return _records;
}

const QVector<UnmatchedCameraReferenceRecord> &CameraReferenceSet::unmatchedRecords() const noexcept
{
    return _unmatchedRecords;
}

void CameraReferenceSet::setImageSetFingerprint(const QString &fingerprint)
{
    CameraReferenceSet updated = *this;
    updated._imageSetFingerprint = fingerprint.trimmed();
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::replaceSource(const CameraReferenceSource &source)
{
    CameraReferenceSet updated = *this;
    updated._source = source;
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::replaceSolverFrame(const CameraReferenceSolverFrame &solverFrame)
{
    CameraReferenceSet updated = *this;
    updated._solverFrame = solverFrame;
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::addLeverArm(const CameraReferenceLeverArm &leverArm)
{
    CameraReferenceSet updated = *this;
    updated._leverArms.push_back(leverArm);
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::replaceLeverArm(const CameraReferenceLeverArm &leverArm)
{
    CameraReferenceSet updated = *this;
    const auto iterator = std::find_if(updated._leverArms.begin(), updated._leverArms.end(),
                                       [&leverArm](const CameraReferenceLeverArm &item)
    {
        return item.id == leverArm.id;
    });
    require(iterator != updated._leverArms.end(),
            QStringLiteral("未找到要替换的杆臂: %1").arg(leverArm.id));
    *iterator = leverArm;
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::replaceLeverArms(const QVector<CameraReferenceLeverArm> &leverArms)
{
    CameraReferenceSet updated = *this;
    updated._leverArms = leverArms;
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::addRecord(const CameraReferenceRecord &record)
{
    CameraReferenceSet updated = *this;
    updated._records.push_back(record);
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::replaceRecord(const CameraReferenceRecord &record)
{
    CameraReferenceSet updated = *this;
    const auto iterator = std::find_if(updated._records.begin(), updated._records.end(),
                                       [&record](const CameraReferenceRecord &item)
    {
        return item.imageUuid == record.imageUuid;
    });
    require(iterator != updated._records.end(),
            QStringLiteral("未找到要替换的相机参考: %1").arg(record.imageUuid));
    *iterator = record;
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::replaceRecords(const QVector<CameraReferenceRecord> &records)
{
    CameraReferenceSet updated = *this;
    updated._records = records;
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::addUnmatchedRecord(const UnmatchedCameraReferenceRecord &record)
{
    CameraReferenceSet updated = *this;
    updated._unmatchedRecords.push_back(record);
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

void CameraReferenceSet::replaceUnmatchedRecords(
    const QVector<UnmatchedCameraReferenceRecord> &records)
{
    CameraReferenceSet updated = *this;
    updated._unmatchedRecords = records;
    updated.touch();
    updated.validateOrThrow();
    *this = std::move(updated);
}

bool CameraReferenceSet::validate(QString *error) const noexcept
{
    if (error)
    {
        error->clear();
    }
    try
    {
        validateOrThrow();
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

bool CameraReferenceSet::operator==(const CameraReferenceSet &other) const
{
    return _schemaVersion == other._schemaVersion
        && _createdAt == other._createdAt
        && _updatedAt == other._updatedAt
        && _imageSetFingerprint == other._imageSetFingerprint
        && _source == other._source
        && _solverFrame == other._solverFrame
        && _leverArms == other._leverArms
        && _records == other._records
        && _unmatchedRecords == other._unmatchedRecords;
}

void CameraReferenceSet::touch()
{
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void CameraReferenceSet::validateOrThrow() const
{
    require(_schemaVersion == CurrentSchemaVersion,
            QStringLiteral("不支持的相机参考 schema_version"));
    require(_createdAt.isValid() && _updatedAt.isValid() && _updatedAt >= _createdAt,
            QStringLiteral("相机参考时间戳无效"));

    const bool hasSourceData = !_records.isEmpty() || !_leverArms.isEmpty()
        || !_unmatchedRecords.isEmpty();
    if (hasSourceData)
    {
        require(!_source.kind.trimmed().isEmpty(), QStringLiteral("相机参考缺少来源类型"));
        require(!_source.sourceCrs.trimmed().isEmpty(), QStringLiteral("相机参考缺少源 CRS"));
        require(!_source.axisOrder.trimmed().isEmpty(), QStringLiteral("相机参考缺少坐标轴顺序"));
        require(!_source.verticalUnit.trimmed().isEmpty(), QStringLiteral("相机参考缺少垂直单位"));
    }

    require(finiteArray(_solverFrame.originEcefMeters)
                && validRotation(_solverFrame.rotationSolverToEcef),
            QStringLiteral("solver frame 包含无效的原点或旋转矩阵"));

    QSet<QString> leverArmIds;
    for (const CameraReferenceLeverArm &leverArm : _leverArms)
    {
        const QString id = leverArm.id.trimmed();
        require(!id.isEmpty(), QStringLiteral("杆臂缺少 ID"));
        require(leverArm.id == id, QStringLiteral("杆臂 ID 不得包含首尾空白: %1").arg(id));
        require(!leverArmIds.contains(id), QStringLiteral("杆臂 ID 重复: %1").arg(id));
        require(finiteArray(leverArm.vectorMeters), QStringLiteral("杆臂包含非有限值: %1").arg(id));
        require(!leverArm.vectorFrame.trimmed().isEmpty(), QStringLiteral("杆臂缺少坐标系: %1").arg(id));
        require(!leverArm.vectorDirection.trimmed().isEmpty(), QStringLiteral("杆臂缺少方向定义: %1").arg(id));
        leverArmIds.insert(id);
    }

    QSet<QString> imageIds;
    for (const CameraReferenceRecord &record : _records)
    {
        const QString imageId = record.imageUuid.trimmed();
        require(!imageId.isEmpty(), QStringLiteral("相机参考缺少影像 UUID"));
        require(record.imageUuid == imageId,
                QStringLiteral("影像 UUID 不得包含首尾空白: %1").arg(imageId));
        require(!imageIds.contains(imageId), QStringLiteral("影像 UUID 重复: %1").arg(imageId));
        require(!record.sourceLabel.trimmed().isEmpty(), QStringLiteral("相机参考缺少来源标签: %1").arg(imageId));
        if (!record.leverArmId.trimmed().isEmpty())
        {
            require(record.leverArmId == record.leverArmId.trimmed(),
                    QStringLiteral("相机参考杆臂 ID 不得包含首尾空白: %1")
                        .arg(record.leverArmId));
            require(leverArmIds.contains(record.leverArmId),
                    QStringLiteral("相机参考引用未知杆臂: %1").arg(record.leverArmId));
        }
        validateRawReference(record.raw, _source, imageId);

        const ResolvedCameraReference &resolved = record.resolved;
        if (resolved.cameraCenterMeters)
        {
            require(finiteArray(*resolved.cameraCenterMeters),
                    QStringLiteral("解算相机中心无效: %1").arg(imageId));
        }
        if (resolved.rotationCameraToWorld)
        {
            require(validRotation(*resolved.rotationCameraToWorld),
                    QStringLiteral("解算相机旋转无效: %1").arg(imageId));
        }
        if (resolved.positionSigmaMeters)
        {
            require(resolved.cameraCenterMeters.has_value()
                        && positiveFinite(*resolved.positionSigmaMeters),
                    QStringLiteral("解算位置精度无效: %1").arg(imageId));
        }
        if (resolved.rotationSigmaDegrees)
        {
            require(resolved.rotationCameraToWorld.has_value()
                        && positiveFinite(*resolved.rotationSigmaDegrees),
                    QStringLiteral("解算姿态精度无效: %1").arg(imageId));
        }
        require(!resolved.positionUsable
                    || (resolved.cameraCenterMeters && resolved.positionSigmaMeters),
                QStringLiteral("可用位置参考缺少中心或精度: %1").arg(imageId));
        require(!resolved.orientationUsable
                    || (resolved.rotationCameraToWorld && resolved.rotationSigmaDegrees),
                QStringLiteral("可用姿态参考缺少旋转或精度: %1").arg(imageId));
        require(!resolved.leverArmApplied || !record.leverArmId.trimmed().isEmpty(),
                QStringLiteral("已应用杆臂但记录未声明杆臂 ID: %1").arg(imageId));

        if (hasResolvedValues(resolved))
        {
            require(!_solverFrame.frameId.trimmed().isEmpty()
                        && !_solverFrame.kind.trimmed().isEmpty()
                        && !_solverFrame.unit.trimmed().isEmpty()
                        && !_solverFrame.normalizationHash.trimmed().isEmpty(),
                    QStringLiteral("解算参考存在但 solver frame 不完整"));
            require(resolved.frameId == _solverFrame.frameId,
                    QStringLiteral("相机参考 frame_id 不匹配: %1").arg(imageId));
            require(resolved.normalizationHash == _solverFrame.normalizationHash,
                    QStringLiteral("相机参考 normalization_hash 不匹配: %1").arg(imageId));
        }
        imageIds.insert(imageId);
    }

    for (const UnmatchedCameraReferenceRecord &record : _unmatchedRecords)
    {
        require(!record.sourceLabel.trimmed().isEmpty(), QStringLiteral("未匹配记录缺少来源标签"));
        if (!record.leverArmId.trimmed().isEmpty())
        {
            require(record.leverArmId == record.leverArmId.trimmed(),
                    QStringLiteral("未匹配记录杆臂 ID 不得包含首尾空白: %1")
                        .arg(record.leverArmId));
            require(leverArmIds.contains(record.leverArmId),
                    QStringLiteral("未匹配记录引用未知杆臂: %1").arg(record.leverArmId));
        }
        validateRawReference(record.raw, _source, record.sourceLabel);
        require(!record.reason.trimmed().isEmpty(),
                QStringLiteral("未匹配记录缺少原因: %1").arg(record.sourceLabel));
    }
}

} // namespace xjw::camera_reference
