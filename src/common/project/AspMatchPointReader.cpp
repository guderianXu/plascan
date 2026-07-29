#include "AspMatchPointReader.h"

#include <QFile>
#include <QFileInfo>
#include <QtEndian>

#include <cmath>
#include <cstring>
#include <limits>

namespace xjw::common::project
{
namespace
{

constexpr quint64 kMaxPointCount = 50'000'000;
constexpr quint64 kDescriptorValueBytes = sizeof(float);
constexpr double kMaximumCoordinate = 1.0e7;

template <typename Integer>
bool readLittleEndianInteger(QFile *file, Integer *value)
{
    if (!file || !value)
    {
        return false;
    }

    const QByteArray bytes = file->read(sizeof(Integer));
    if (bytes.size() != static_cast<qsizetype>(sizeof(Integer)))
    {
        return false;
    }

    *value = qFromLittleEndian<Integer>(
        reinterpret_cast<const uchar *>(bytes.constData()));
    return true;
}

bool readLittleEndianFloat(QFile *file, float *value)
{
    quint32 bits = 0;
    if (!readLittleEndianInteger(file, &bits) || !value)
    {
        return false;
    }
    std::memcpy(value, &bits, sizeof(float));
    return true;
}

bool skipBytes(QFile *file, quint64 byteCount)
{
    if (!file)
    {
        return false;
    }
    if (byteCount > static_cast<quint64>(std::numeric_limits<qint64>::max()))
    {
        return false;
    }

    const qint64 current = file->pos();
    const qint64 count = static_cast<qint64>(byteCount);
    return count <= file->size() - current && file->seek(current + count);
}

bool readInterestPoint(QFile *file, QPointF *point)
{
    float x = 0.0f;
    float y = 0.0f;
    qint32 integerX = 0;
    qint32 integerY = 0;
    float orientation = 0.0f;
    float scale = 0.0f;
    float interest = 0.0f;
    qint8 polarity = 0;
    quint32 octave = 0;
    quint32 scaleLevel = 0;
    quint64 descriptorCount = 0;

    if (!readLittleEndianFloat(file, &x)
        || !readLittleEndianFloat(file, &y)
        || !readLittleEndianInteger(file, &integerX)
        || !readLittleEndianInteger(file, &integerY)
        || !readLittleEndianFloat(file, &orientation)
        || !readLittleEndianFloat(file, &scale)
        || !readLittleEndianFloat(file, &interest))
    {
        return false;
    }

    const QByteArray polarityByte = file->read(1);
    if (polarityByte.size() != 1)
    {
        return false;
    }
    polarity = static_cast<qint8>(polarityByte.at(0));

    if (!readLittleEndianInteger(file, &octave)
        || !readLittleEndianInteger(file, &scaleLevel)
        || !readLittleEndianInteger(file, &descriptorCount)
        || descriptorCount > std::numeric_limits<quint64>::max() / kDescriptorValueBytes
        || !skipBytes(file, descriptorCount * kDescriptorValueBytes))
    {
        return false;
    }

    Q_UNUSED(integerX);
    Q_UNUSED(integerY);
    Q_UNUSED(orientation);
    Q_UNUSED(scale);
    Q_UNUSED(interest);
    Q_UNUSED(polarity);
    Q_UNUSED(octave);
    Q_UNUSED(scaleLevel);

    if (point)
    {
        *point = QPointF(static_cast<qreal>(x), static_cast<qreal>(y));
    }
    return true;
}

bool isDisplayCoordinateValid(const QPointF &point)
{
    return std::isfinite(point.x())
        && std::isfinite(point.y())
        && std::abs(point.x()) <= kMaximumCoordinate
        && std::abs(point.y()) <= kMaximumCoordinate;
}

bool readPointSet(QFile *file,
                  quint64 count,
                  QVector<QPointF> *points,
                  QString *errorMessage)
{
    if (!file || !points || count > kMaxPointCount
        || count > static_cast<quint64>(std::numeric_limits<qsizetype>::max()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配点数量无效: %1").arg(count);
        }
        return false;
    }

    points->reserve(static_cast<qsizetype>(count));
    for (quint64 index = 0; index < count; ++index)
    {
        QPointF point;
        if (!readInterestPoint(file, &point))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("读取第 %1 个匹配点时文件提前结束")
                                    .arg(index + 1);
            }
            return false;
        }
        points->append(point);
    }
    return true;
}

} // namespace

AspMatchPointResult readAspMatchPoints(const QString &path)
{
    AspMatchPointResult result;
    if (!QFileInfo::exists(path))
    {
        result.errorMessage = QStringLiteral("匹配文件不存在: %1").arg(path);
        return result;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        result.errorMessage = QStringLiteral("无法读取匹配文件: %1").arg(path);
        return result;
    }

    quint64 leftCount = 0;
    quint64 rightCount = 0;
    if (!readLittleEndianInteger(&file, &leftCount)
        || !readLittleEndianInteger(&file, &rightCount))
    {
        result.errorMessage = QStringLiteral("匹配文件头无效: %1").arg(path);
        return result;
    }

    QVector<QPointF> rawLeft;
    QVector<QPointF> rawRight;
    if (!readPointSet(&file, leftCount, &rawLeft, &result.errorMessage)
        || !readPointSet(&file, rightCount, &rawRight, &result.errorMessage))
    {
        return result;
    }

    const qsizetype pairCount = qMin(rawLeft.size(), rawRight.size());
    result.leftPoints.reserve(pairCount);
    result.rightPoints.reserve(pairCount);
    for (qsizetype index = 0; index < pairCount; ++index)
    {
        if (!isDisplayCoordinateValid(rawLeft.at(index))
            || !isDisplayCoordinateValid(rawRight.at(index)))
        {
            continue;
        }
        result.leftPoints.append(rawLeft.at(index));
        result.rightPoints.append(rawRight.at(index));
    }

    result.success = true;
    return result;
}

} // namespace xjw::common::project
