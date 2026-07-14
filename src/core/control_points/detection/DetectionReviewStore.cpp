#include "DetectionReviewStore.h"

#include "MarkerDetectorFactory.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSaveFile>

#include <algorithm>

namespace xjw::control_points
{
namespace
{

QJsonArray encodePoint(const QPointF &point)
{
    return QJsonArray{point.x(), point.y()};
}

bool decodePoint(const QJsonValue &value, QPointF *point)
{
    if (!point || !value.isArray()) return false;
    const QJsonArray values = value.toArray();
    if (values.size() != 2 || !values.at(0).isDouble() || !values.at(1).isDouble()) return false;
    *point = QPointF(values.at(0).toDouble(), values.at(1).toDouble());
    return true;
}

QJsonObject encodeEntry(const DetectionReviewEntry &entry)
{
    QJsonArray corners;
    for (const QPointF &corner : entry.observation.detection.corners)
    {
        corners.push_back(encodePoint(corner));
    }
    const MarkerDetection &detection = entry.observation.detection;
    return QJsonObject{
        {QStringLiteral("id"), entry.id},
        {QStringLiteral("reason"), entry.reason},
        {QStringLiteral("message"), entry.message},
        {QStringLiteral("image_id"), entry.observation.imageId},
        {QStringLiteral("image_path_snapshot"), entry.observation.imagePathSnapshot},
        {QStringLiteral("image_content_signature"), entry.observation.imageContentSignature},
        {QStringLiteral("family"), markerTargetFamilyName(detection.family)},
        {QStringLiteral("target_id"), detection.targetId},
        {QStringLiteral("center"), encodePoint(detection.center)},
        {QStringLiteral("corners"), corners},
        {QStringLiteral("confidence"), detection.confidence},
        {QStringLiteral("center_sigma_px"), detection.centerSigmaPx},
        {QStringLiteral("decision_margin"), detection.decisionMargin},
        {QStringLiteral("hamming"), detection.hamming},
        {QStringLiteral("size_px"), detection.sizePx},
        {QStringLiteral("rotation_degrees"), detection.rotationDegrees},
        {QStringLiteral("source"), detection.source},
    };
}

bool decodeEntry(const QJsonObject &object, DetectionReviewEntry *entry, QString *error)
{
    if (!entry)
    {
        if (error) *error = QStringLiteral("检测复核条目输出参数为空");
        return false;
    }
    const auto family = MarkerDetectorFactory::parseFamily(
        object.value(QStringLiteral("family")).toString());
    if (!family.has_value())
    {
        if (error) *error = QStringLiteral("检测复核条目标靶 family 无效");
        return false;
    }

    DetectionReviewEntry decoded;
    decoded.id = object.value(QStringLiteral("id")).toString();
    decoded.reason = object.value(QStringLiteral("reason")).toString();
    decoded.message = object.value(QStringLiteral("message")).toString();
    decoded.observation.imageId = object.value(QStringLiteral("image_id")).toString();
    decoded.observation.imagePathSnapshot =
        object.value(QStringLiteral("image_path_snapshot")).toString();
    decoded.observation.imageContentSignature =
        object.value(QStringLiteral("image_content_signature")).toString();
    decoded.observation.detection.family = *family;
    decoded.observation.detection.targetId = object.value(QStringLiteral("target_id")).toInt(-1);
    if (decoded.id.isEmpty() || decoded.reason.isEmpty() || decoded.observation.imageId.isEmpty()
        || !decodePoint(object.value(QStringLiteral("center")),
                        &decoded.observation.detection.center))
    {
        if (error) *error = QStringLiteral("检测复核条目缺少 ID、原因、影像 UUID 或中心坐标");
        return false;
    }
    for (const QJsonValue &corner : object.value(QStringLiteral("corners")).toArray())
    {
        QPointF point;
        if (!decodePoint(corner, &point))
        {
            if (error) *error = QStringLiteral("检测复核条目的角点坐标无效");
            return false;
        }
        decoded.observation.detection.corners.push_back(point);
    }
    MarkerDetection &detection = decoded.observation.detection;
    detection.confidence = object.value(QStringLiteral("confidence")).toDouble();
    detection.centerSigmaPx = object.value(QStringLiteral("center_sigma_px")).toDouble(1.0);
    detection.decisionMargin = object.value(QStringLiteral("decision_margin")).toDouble();
    detection.hamming = object.value(QStringLiteral("hamming")).toInt();
    detection.sizePx = object.value(QStringLiteral("size_px")).toDouble();
    detection.rotationDegrees = object.value(QStringLiteral("rotation_degrees")).toDouble();
    detection.source = object.value(QStringLiteral("source")).toString();
    *entry = std::move(decoded);
    return true;
}

QJsonObject encodeQueue(const DetectionReviewQueue &queue)
{
    QJsonArray entries;
    for (const DetectionReviewEntry &entry : queue.entries)
    {
        entries.push_back(encodeEntry(entry));
    }
    return QJsonObject{
        {QStringLiteral("schema_version"), queue.schemaVersion},
        {QStringLiteral("source_revision"), QString::number(queue.sourceRevision)},
        {QStringLiteral("created_at"), queue.createdAt.toString(Qt::ISODateWithMs)},
        {QStringLiteral("updated_at"), queue.updatedAt.toString(Qt::ISODateWithMs)},
        {QStringLiteral("entries"), entries},
    };
}

bool decodeQueue(const QJsonObject &object, DetectionReviewQueue *queue, QString *error)
{
    if (!queue)
    {
        if (error) *error = QStringLiteral("检测复核队列输出参数为空");
        return false;
    }
    if (object.value(QStringLiteral("schema_version")).toInt(-1) != 1)
    {
        if (error) *error = QStringLiteral("不支持的检测复核队列 schema_version");
        return false;
    }

    DetectionReviewQueue decoded;
    decoded.sourceRevision = object.value(QStringLiteral("source_revision")).toString().toULongLong();
    decoded.createdAt = QDateTime::fromString(
        object.value(QStringLiteral("created_at")).toString(), Qt::ISODateWithMs);
    decoded.updatedAt = QDateTime::fromString(
        object.value(QStringLiteral("updated_at")).toString(), Qt::ISODateWithMs);
    if (!decoded.createdAt.isValid() || !decoded.updatedAt.isValid())
    {
        if (error) *error = QStringLiteral("检测复核队列时间戳无效");
        return false;
    }
    for (const QJsonValue &value : object.value(QStringLiteral("entries")).toArray())
    {
        if (!value.isObject())
        {
            if (error) *error = QStringLiteral("检测复核队列条目不是 JSON 对象");
            return false;
        }
        DetectionReviewEntry entry;
        if (!decodeEntry(value.toObject(), &entry, error)) return false;
        decoded.entries.push_back(std::move(entry));
    }
    *queue = std::move(decoded);
    return true;
}

} // namespace

QString detectionReviewEntryId(const MarkerDetectionObservation &observation,
                               const QString &reason)
{
    const MarkerDetection &detection = observation.detection;
    const QByteArray identity = QStringLiteral("%1\n%2\n%3\n%4\n%5\n%6\n%7\n%8")
                                    .arg(reason,
                                         observation.imageId,
                                         markerTargetFamilyName(detection.family),
                                         QString::number(detection.targetId),
                                         QString::number(detection.center.x(), 'g', 17),
                                         QString::number(detection.center.y(), 'g', 17),
                                         detection.source,
                                         observation.imageContentSignature)
                                    .toUtf8();
    return QString::fromLatin1(
        QCryptographicHash::hash(identity, QCryptographicHash::Sha256).toHex());
}

DetectionReviewStore::DetectionReviewStore(QString path)
    : _path(QDir::cleanPath(std::move(path)))
{
}

DetectionReviewIoResult DetectionReviewStore::load() const
{
    DetectionReviewIoResult result;
    if (_path.trimmed().isEmpty())
    {
        result.error = QStringLiteral("检测复核 sidecar 路径为空");
        return result;
    }
    if (!QFileInfo::exists(_path))
    {
        result.ok = true;
        return result;
    }

    QFile file(_path);
    if (!file.open(QIODevice::ReadOnly))
    {
        result.error = QStringLiteral("无法读取检测复核 sidecar: %1").arg(file.errorString());
        return result;
    }
    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject())
    {
        result.error = QStringLiteral("检测复核 sidecar JSON 损坏，文件保持只读未覆盖: %1")
                           .arg(parse_error.errorString());
        return result;
    }
    if (!decodeQueue(document.object(), &result.queue, &result.error))
    {
        result.error = QStringLiteral("检测复核 sidecar 校验失败，文件保持只读未覆盖: %1")
                           .arg(result.error);
        return result;
    }
    result.ok = true;
    return result;
}

DetectionReviewIoResult DetectionReviewStore::save(const DetectionReviewQueue &queue) const
{
    DetectionReviewIoResult result;
    result.queue = queue;
    if (_path.trimmed().isEmpty())
    {
        result.error = QStringLiteral("检测复核 sidecar 路径为空");
        return result;
    }
    result.queue.schemaVersion = 1;
    if (!result.queue.createdAt.isValid()) result.queue.createdAt = QDateTime::currentDateTimeUtc();
    result.queue.updatedAt = QDateTime::currentDateTimeUtc();

    const QFileInfo info(_path);
    if (!QDir().mkpath(info.absolutePath()))
    {
        result.error = QStringLiteral("无法创建检测复核 sidecar 目录: %1").arg(info.absolutePath());
        return result;
    }
    QSaveFile file(_path);
    if (!file.open(QIODevice::WriteOnly))
    {
        result.error = QStringLiteral("无法创建检测复核 sidecar 临时文件: %1")
                           .arg(file.errorString());
        return result;
    }
    const QByteArray bytes = QJsonDocument(encodeQueue(result.queue)).toJson(QJsonDocument::Indented);
    if (file.write(bytes) != bytes.size())
    {
        file.cancelWriting();
        result.error = QStringLiteral("写入检测复核 sidecar 失败: %1").arg(file.errorString());
        return result;
    }
    if (!file.commit())
    {
        result.error = QStringLiteral("原子替换检测复核 sidecar 失败: %1").arg(file.errorString());
        return result;
    }
    result.ok = true;
    return result;
}

QString DetectionReviewStore::path() const
{
    return _path;
}

} // namespace xjw::control_points
