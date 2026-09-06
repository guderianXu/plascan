#include "PlaMatchHctFeatureCacheFile.h"

#include "PlaMatchHctAlgorithm.h"
#include "PlaMatchHctFeaturePayload.h"
#include "../ImageFeaturePointFile.h"

#include <QDataStream>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSaveFile>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace xjw::image_matching
{
    namespace
    {

        constexpr std::array<char, 8> kMagic{{'P', 'I', 'H', 'C', 'A', 'C', 'H', '1'}};
        constexpr quint32 kFormatVersion = 1;
        constexpr quint64 kMaximumFeatures = 10'000'000;
        constexpr double kRadiansToDegrees = 57.2957795130823208768;

        void setReason(QString* reason, const QString& message)
        {
            if (reason)
            {
                *reason = message;
            }
        }

        void configureStream(QDataStream* stream)
        {
            stream->setByteOrder(QDataStream::LittleEndian);
            stream->setVersion(QDataStream::Qt_6_0);
            stream->setFloatingPointPrecision(QDataStream::DoublePrecision);
        }

        QString normalizedPath(const QString& path)
        {
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
        }

        bool writeRows(QDataStream* stream, const std::vector<metalign::Keypoint>& rows)
        {
            *stream << static_cast<quint64>(rows.size());
            for (const metalign::Keypoint& row : rows)
            {
                *stream << row.x << row.y << row.scale << row.orientation << row.response
                        << static_cast<qint32>(row.octave) << static_cast<qint32>(row.level)
                        << static_cast<quint64>(row.detector_id) << static_cast<quint64>(row.source_id)
                        << static_cast<qint32>(row.laplacian_sign);
                if (stream->writeRawData(reinterpret_cast<const char*>(row.descriptor.data()),
                                         static_cast<int>(row.descriptor.size())) !=
                    static_cast<int>(row.descriptor.size()))
                {
                    return false;
                }
            }
            return stream->status() == QDataStream::Ok;
        }

        bool readRows(QDataStream* stream, std::vector<metalign::Keypoint>* rows)
        {
            quint64 count = 0;
            *stream >> count;
            if (stream->status() != QDataStream::Ok || count > kMaximumFeatures)
            {
                return false;
            }
            rows->resize(static_cast<std::size_t>(count));
            for (metalign::Keypoint& row : *rows)
            {
                qint32 octave = 0;
                qint32 level = 0;
                quint64 detectorId = 0;
                quint64 sourceId = 0;
                qint32 laplacianSign = 0;
                *stream >> row.x >> row.y >> row.scale >> row.orientation >> row.response >> octave >> level >>
                    detectorId >> sourceId >> laplacianSign;
                row.octave = octave;
                row.level = level;
                row.detector_id = static_cast<std::size_t>(detectorId);
                row.source_id = static_cast<std::size_t>(sourceId);
                row.laplacian_sign = laplacianSign;
                if (stream->status() != QDataStream::Ok ||
                    stream->readRawData(reinterpret_cast<char*>(row.descriptor.data()),
                                        static_cast<int>(row.descriptor.size())) !=
                        static_cast<int>(row.descriptor.size()) ||
                    !std::isfinite(row.x) || !std::isfinite(row.y) || !std::isfinite(row.scale) || !(row.scale > 0.0) ||
                    !std::isfinite(row.orientation) || !std::isfinite(row.response))
                {
                    rows->clear();
                    return false;
                }
            }
            return true;
        }

        FeatureSet makeFeatureSet(metalign::FeatureSet vendorFeatures, const QString& computeBackend)
        {
            FeatureSet result;
            result.keypoints.reserve(vendorFeatures.keypoints.size());
            result.scores.reserve(vendorFeatures.keypoints.size());
            result.descriptors.create(
                static_cast<int>(vendorFeatures.keypoints.size()), static_cast<int>(metalign::kDescriptorSize), CV_8U);
            for (int index = 0; index < static_cast<int>(vendorFeatures.keypoints.size()); ++index)
            {
                const metalign::Keypoint& source = vendorFeatures.keypoints[static_cast<std::size_t>(index)];
                cv::KeyPoint keypoint;
                keypoint.pt = cv::Point2f(static_cast<float>(source.x), static_cast<float>(source.y));
                keypoint.size = static_cast<float>(source.scale);
                keypoint.angle = static_cast<float>(source.orientation * kRadiansToDegrees);
                keypoint.response = static_cast<float>(source.response);
                keypoint.octave = source.octave;
                result.keypoints.push_back(keypoint);
                result.scores.push_back(keypoint.response);
                std::memcpy(result.descriptors.ptr(index), source.descriptor.data(), metalign::kDescriptorSize);
            }
            result.sourceAlgorithm = kPlaMatchHctAlgorithmId;
            result.computeBackend = computeBackend.toStdString();
            result.imageWidth = static_cast<int>(vendorFeatures.image_width);
            result.imageHeight = static_cast<int>(vendorFeatures.image_height);
            result.payload = std::make_shared<PlaMatchHctFeaturePayload>(std::move(vendorFeatures));
            return result;
        }

    } // namespace

    QString PlaMatchHctFeatureCacheFile::filePathForImage(const QString& directory, const QString& imagePath)
    {
        QString path = ImageFeaturePointFile::filePathForImage(directory, imagePath);
        const QString suffix = QString::fromLatin1(kImageFeaturePointFileSuffix);
        if (path.endsWith(suffix))
        {
            path.chop(suffix.size());
        }
        return path + QStringLiteral(".pihctcache");
    }

    bool PlaMatchHctFeatureCacheFile::write(const QString& filePath,
                                            const QString& imagePath,
                                            const QString& producerSignature,
                                            const FeatureSet& features,
                                            QString* errorMessage)
    {
        const auto payload = std::dynamic_pointer_cast<const PlaMatchHctFeaturePayload>(features.payload);
        const QFileInfo sourceInfo(imagePath);
        if (!payload || !features.isConsistent() || !sourceInfo.isFile() || producerSignature.isEmpty())
        {
            setReason(errorMessage, QStringLiteral("PlaMatch-HCT 特征缓存输入无效"));
            return false;
        }
        if (!QDir().mkpath(QFileInfo(filePath).absolutePath()))
        {
            setReason(errorMessage, QStringLiteral("无法创建特征缓存目录: %1").arg(QFileInfo(filePath).absolutePath()));
            return false;
        }

        QSaveFile file(filePath);
        if (!file.open(QIODevice::WriteOnly))
        {
            setReason(errorMessage, QStringLiteral("无法写入特征缓存: %1").arg(filePath));
            return false;
        }
        QDataStream stream(&file);
        configureStream(&stream);
        if (stream.writeRawData(kMagic.data(), static_cast<int>(kMagic.size())) != static_cast<int>(kMagic.size()))
        {
            setReason(errorMessage, QStringLiteral("无法写入特征缓存文件头: %1").arg(filePath));
            return false;
        }
        const metalign::FeatureSet& full = payload->fullFeatures();
        const metalign::FeatureSet& coarse = payload->coarseFeatures();
        stream << kFormatVersion << normalizedPath(imagePath) << static_cast<quint64>(sourceInfo.size())
               << static_cast<qint64>(sourceInfo.lastModified().toMSecsSinceEpoch()) << producerSignature
               << QString::fromStdString(features.computeBackend) << static_cast<quint64>(full.image_width)
               << static_cast<quint64>(full.image_height) << static_cast<quint64>(full.source_keypoint_count);
        for (const float value : full.global_descriptor)
        {
            stream << value;
        }
        if (!writeRows(&stream, full.keypoints) || !writeRows(&stream, coarse.keypoints) ||
            stream.status() != QDataStream::Ok || !file.commit())
        {
            setReason(errorMessage, QStringLiteral("提交特征缓存失败: %1").arg(filePath));
            return false;
        }
        return true;
    }

    std::shared_ptr<FeatureSet> PlaMatchHctFeatureCacheFile::read(const QString& filePath,
                                                                  const QString& imagePath,
                                                                  const QString& producerSignature,
                                                                  QString* missReason)
    {
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly))
        {
            setReason(missReason, QStringLiteral("缓存不存在"));
            return {};
        }
        QDataStream stream(&file);
        configureStream(&stream);
        std::array<char, 8> magic{};
        quint32 version = 0;
        QString storedPath;
        quint64 storedSize = 0;
        qint64 storedModified = 0;
        QString storedSignature;
        QString storedComputeBackend;
        quint64 width = 0;
        quint64 height = 0;
        quint64 sourceCount = 0;
        if (stream.readRawData(magic.data(), static_cast<int>(magic.size())) != static_cast<int>(magic.size()) ||
            magic != kMagic)
        {
            setReason(missReason, QStringLiteral("缓存文件头无效"));
            return {};
        }
        stream >> version >> storedPath >> storedSize >> storedModified >> storedSignature >> storedComputeBackend >>
            width >> height >> sourceCount;
        const QFileInfo sourceInfo(imagePath);
        if (version != kFormatVersion || storedPath != normalizedPath(imagePath) ||
            storedSize != static_cast<quint64>(sourceInfo.size()) ||
            storedModified != sourceInfo.lastModified().toMSecsSinceEpoch() || storedSignature != producerSignature ||
            width == 0 || height == 0 || width > static_cast<quint64>(std::numeric_limits<int>::max()) ||
            height > static_cast<quint64>(std::numeric_limits<int>::max()) || sourceCount > kMaximumFeatures)
        {
            setReason(missReason, QStringLiteral("影像身份或特征参数已变化"));
            return {};
        }

        metalign::FeatureSet vendorFeatures;
        vendorFeatures.path = imagePath.toStdString();
        vendorFeatures.image_width = static_cast<std::size_t>(width);
        vendorFeatures.image_height = static_cast<std::size_t>(height);
        vendorFeatures.source_keypoint_count = static_cast<std::size_t>(sourceCount);
        for (float& value : vendorFeatures.global_descriptor)
        {
            stream >> value;
        }
        if (!readRows(&stream, &vendorFeatures.keypoints) || !readRows(&stream, &vendorFeatures.coarse_keypoints) ||
            stream.status() != QDataStream::Ok || !file.atEnd() || vendorFeatures.keypoints.empty())
        {
            setReason(missReason, QStringLiteral("缓存内容损坏"));
            return {};
        }
        auto result = std::make_shared<FeatureSet>(makeFeatureSet(std::move(vendorFeatures), storedComputeBackend));
        if (!result->isConsistent())
        {
            setReason(missReason, QStringLiteral("缓存特征不一致"));
            return {};
        }
        setReason(missReason, {});
        return result;
    }

} // namespace xjw::image_matching
