#include "ImageFeaturePointFile.h"

#include "ImageMatchFile.h"

#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSaveFile>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::image_matching
{
namespace
{

constexpr char kMagic[] = {'P', 'I', 'F', 'E', 'A', 'T', '0', '1'};
constexpr std::uint32_t kMaximumFeaturePoints = 10'000'000;

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

void configureStream(QDataStream *stream)
{
    stream->setByteOrder(QDataStream::LittleEndian);
    stream->setVersion(QDataStream::Qt_6_0);
    stream->setFloatingPointPrecision(QDataStream::SinglePrecision);
}

bool validObservation(const KeypointObservation &observation)
{
    return std::isfinite(observation.x) && std::isfinite(observation.y)
        && std::isfinite(observation.scale) && observation.scale > 0.0f
        && std::isfinite(observation.orientation) && std::isfinite(observation.response);
}

bool writeIdentity(QDataStream *stream, const ImageIdentity &identity)
{
    *stream << identity.stableId
            << identity.path
            << identity.displayName
            << static_cast<quint32>(identity.width)
            << static_cast<quint32>(identity.height)
            << static_cast<quint64>(identity.fileSize)
            << static_cast<qint64>(identity.modifiedTimeMs);
    return stream->status() == QDataStream::Ok;
}

bool readIdentity(QDataStream *stream, ImageIdentity *identity)
{
    quint32 width = 0;
    quint32 height = 0;
    quint64 fileSize = 0;
    qint64 modifiedTime = 0;
    *stream >> identity->stableId
            >> identity->path
            >> identity->displayName
            >> width
            >> height
            >> fileSize
            >> modifiedTime;
    identity->width = width;
    identity->height = height;
    identity->fileSize = fileSize;
    identity->modifiedTimeMs = modifiedTime;
    return stream->status() == QDataStream::Ok && identity->isValid();
}

bool writeObservation(QDataStream *stream, const KeypointObservation &observation)
{
    *stream << static_cast<quint32>(observation.featureId)
            << observation.x
            << observation.y
            << observation.scale
            << observation.orientation
            << observation.response;
    return stream->status() == QDataStream::Ok;
}

bool readObservation(QDataStream *stream, KeypointObservation *observation)
{
    quint32 featureId = 0;
    *stream >> featureId
            >> observation->x
            >> observation->y
            >> observation->scale
            >> observation->orientation
            >> observation->response;
    observation->featureId = featureId;
    return stream->status() == QDataStream::Ok && validObservation(*observation);
}

} // namespace

QString ImageFeaturePointFile::filePathForImage(const QString &directory,
                                                const QString &imagePath)
{
    QString path = ImageMatchFile::filePathForImage(directory, imagePath);
    const QString matchSuffix = QString::fromLatin1(kImageMatchFileSuffix);
    if (path.endsWith(matchSuffix))
    {
        path.chop(matchSuffix.size());
    }
    return path + QString::fromLatin1(kImageFeaturePointFileSuffix);
}

bool ImageFeaturePointFile::write(const QString &filePath,
                                  const ImageFeaturePointCatalog &catalog,
                                  QString *errorMessage)
{
    if (!catalog.owner.isValid() || catalog.algorithmId.trimmed().isEmpty())
    {
        setError(errorMessage, QStringLiteral("特征点目录缺少有效的影像或算法身份"));
        return false;
    }
    if (catalog.observations.empty()
        || catalog.observations.size() > kMaximumFeaturePoints
        || !std::all_of(catalog.observations.begin(), catalog.observations.end(), validObservation))
    {
        setError(errorMessage, QStringLiteral("特征点目录包含无效或超限的观测"));
        return false;
    }
    if (!QDir().mkpath(QFileInfo(filePath).absolutePath()))
    {
        setError(
            errorMessage,
            QStringLiteral("无法创建特征点目录: %1").arg(QFileInfo(filePath).absolutePath()));
        return false;
    }

    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly))
    {
        setError(errorMessage, QStringLiteral("无法写入特征点文件: %1").arg(filePath));
        return false;
    }
    QDataStream stream(&file);
    configureStream(&stream);
    if (stream.writeRawData(kMagic, static_cast<int>(sizeof(kMagic))) != static_cast<int>(sizeof(kMagic)))
    {
        setError(errorMessage, QStringLiteral("写入特征点文件头失败: %1").arg(filePath));
        return false;
    }
    stream << static_cast<quint32>(kImageFeaturePointFormatVersion);
    if (!writeIdentity(&stream, catalog.owner))
    {
        setError(errorMessage, QStringLiteral("写入特征点影像身份失败: %1").arg(filePath));
        return false;
    }
    stream << catalog.algorithmId
           << static_cast<quint32>(catalog.algorithmVersion)
           << static_cast<quint32>(catalog.featureSchemaVersion)
           << static_cast<quint32>(catalog.observations.size());
    for (const KeypointObservation &observation : catalog.observations)
    {
        if (!writeObservation(&stream, observation))
        {
            setError(errorMessage, QStringLiteral("写入特征点观测失败: %1").arg(filePath));
            return false;
        }
    }
    if (stream.status() != QDataStream::Ok || !file.commit())
    {
        setError(errorMessage, QStringLiteral("提交特征点文件失败: %1").arg(filePath));
        return false;
    }
    return true;
}

bool ImageFeaturePointFile::read(const QString &filePath,
                                 ImageFeaturePointCatalog *catalog,
                                 QString *errorMessage)
{
    if (!catalog)
    {
        setError(errorMessage, QStringLiteral("特征点读取输出为空"));
        return false;
    }
    *catalog = {};
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage, QStringLiteral("无法读取特征点文件: %1").arg(filePath));
        return false;
    }
    QDataStream stream(&file);
    configureStream(&stream);
    char magic[sizeof(kMagic)]{};
    quint32 version = 0;
    if (stream.readRawData(magic, static_cast<int>(sizeof(magic))) != static_cast<int>(sizeof(magic))
        || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic)))
    {
        setError(errorMessage, QStringLiteral("特征点文件标识无效: %1").arg(filePath));
        return false;
    }
    stream >> version;
    if (version != kImageFeaturePointFormatVersion || !readIdentity(&stream, &catalog->owner))
    {
        setError(errorMessage, QStringLiteral("特征点文件版本或影像身份无效: %1").arg(filePath));
        return false;
    }
    quint32 algorithmVersion = 0;
    quint32 featureSchemaVersion = 0;
    quint32 observationCount = 0;
    stream >> catalog->algorithmId >> algorithmVersion >> featureSchemaVersion >> observationCount;
    if (stream.status() != QDataStream::Ok
        || catalog->algorithmId.trimmed().isEmpty()
        || observationCount == 0
        || observationCount > kMaximumFeaturePoints)
    {
        setError(errorMessage, QStringLiteral("特征点文件元数据无效: %1").arg(filePath));
        return false;
    }
    catalog->algorithmVersion = algorithmVersion;
    catalog->featureSchemaVersion = featureSchemaVersion;
    catalog->observations.resize(observationCount);
    for (KeypointObservation &observation : catalog->observations)
    {
        if (!readObservation(&stream, &observation))
        {
            setError(errorMessage, QStringLiteral("特征点观测数据无效: %1").arg(filePath));
            *catalog = {};
            return false;
        }
    }
    if (stream.status() != QDataStream::Ok || !file.atEnd())
    {
        setError(errorMessage, QStringLiteral("特征点文件尾部无效: %1").arg(filePath));
        *catalog = {};
        return false;
    }
    return true;
}

} // namespace xjw::image_matching
