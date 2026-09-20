#include "FeatureResidualLoader.h"

#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QHash>
#include <QElapsedTimer>
#include <QMutex>
#include <QMutexLocker>
#include <QSet>
#include <QWaitCondition>

#include <cmath>
#include <limits>
#include <memory>

namespace
{

    QString normalizedPath(const QString& path)
    {
        QFileInfo info(path);
        QString normalized = info.exists() ? info.canonicalFilePath() : info.absoluteFilePath();
        normalized = QDir::cleanPath(normalized);
#ifdef Q_OS_WIN
        normalized = normalized.toLower();
#endif
        return normalized;
    }

    QString latestSidecarPath(const QJsonObject& metadata, const QString& projectPath)
    {
        const QJsonArray records = metadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
        for (int index = records.size() - 1; index >= 0; --index)
        {
            const QJsonObject files = records.at(index).toObject().value(QStringLiteral("files")).toObject();
            const QString path = xjw::common::project::ProjectIO::resolveProjectResourcePath(
                projectPath, files.value(QStringLiteral("sparse_cloud_points_json")).toString());
            if (QFileInfo::exists(path))
            {
                return path;
            }
        }
        return {};
    }

    QString observationImageName(const QJsonObject& observation)
    {
        const QString explicitName = observation.value(QStringLiteral("image_name")).toString().trimmed();
        return explicitName.isEmpty() ? QFileInfo(observation.value(QStringLiteral("image_path")).toString()).fileName()
                                      : explicitName;
    }

    bool finitePoint(const QJsonArray& value, QPointF* point)
    {
        if (!point || value.size() < 2)
        {
            return false;
        }
        const double x = value.at(0).toDouble(std::numeric_limits<double>::quiet_NaN());
        const double y = value.at(1).toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            return false;
        }
        *point = QPointF(x, y);
        return true;
    }

    struct IndexedImageDiagnostics
    {
        std::vector<cv::KeyPoint> keypoints;
        QVector<xjw::gui::views::FeatureResidualVector> residuals;
    };

    struct SidecarImageIdentity
    {
        QString path;
        QString name;
    };

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

    struct SidecarObservationIndex
    {
        QHash<QString, IndexedImageDiagnostics> diagnosticsByPath;
        QHash<QString, QSet<QString>> pathsByImageName;
    };

    struct SidecarIndexResult
    {
        std::shared_ptr<const SidecarObservationIndex> index;
        QString error;
        qint64 loadMilliseconds = 0;
        bool loadedFromCache = false;
    };

    struct SidecarIndexCache
    {
        QMutex mutex;
        QWaitCondition loadFinished;
        QString path;
        qint64 size = -1;
        qint64 modifiedMilliseconds = -1;
        std::shared_ptr<const SidecarObservationIndex> index;
        QString error;
        qint64 loadMilliseconds = 0;
        bool loading = false;
    };

    SidecarIndexCache& sidecarIndexCache()
    {
        static SidecarIndexCache cache;
        return cache;
    }

    std::shared_ptr<SidecarObservationIndex> buildSidecarObservationIndex(const QJsonDocument& document,
                                                                           QString* error)
    {
        const auto fail = [error](const QString& message) -> std::shared_ptr<SidecarObservationIndex>
        {
            if (error)
            {
                *error = message;
            }
            return {};
        };

        auto index = std::make_shared<SidecarObservationIndex>();
        index->diagnosticsByPath.reserve(512);
        index->pathsByImageName.reserve(512);
        QHash<QString, QString> normalizedPathCache;
        normalizedPathCache.reserve(512);
        const QJsonObject root = document.object();
        if (root.value(QStringLiteral("schema")).toString() !=
            QStringLiteral("plascan.sfm_sparse_points.v3"))
        {
            return fail(QStringLiteral(
                "连接点观测文件必须使用 plascan.sfm_sparse_points.v3；旧 schema 不再兼容"));
        }
        QHash<int, SidecarImageIdentity> imagesByIndex;
        QHash<QString, int> imageIndexByCanonicalId;
        const QJsonArray images = root.value(QStringLiteral("images")).toArray();
        if (images.isEmpty())
        {
            return fail(QStringLiteral("v3 连接点观测文件缺少非空 images 表"));
        }
        imagesByIndex.reserve(images.size());
        imageIndexByCanonicalId.reserve(images.size());
        for (const QJsonValue& imageValue : images)
        {
            if (!imageValue.isObject())
            {
                return fail(QStringLiteral("v3 连接点观测文件的 images 行必须是对象"));
            }
            const QJsonObject image = imageValue.toObject();
            int cameraIndex = -1;
            if (!readNonNegativeInteger(image.value(QStringLiteral("camera_index")), &cameraIndex))
            {
                return fail(QStringLiteral("v3 连接点观测文件的 images.camera_index 无效"));
            }
            const QJsonValue imageIdValue = image.value(QStringLiteral("image_id"));
            const QJsonValue imagePathValue = image.value(QStringLiteral("image_path"));
            if (!imageIdValue.isString() || !imagePathValue.isString())
            {
                return fail(QStringLiteral("v3 连接点观测文件的 images 必须包含字符串 image_id 和 image_path"));
            }
            const QString canonicalImageId = imageIdValue.toString().trimmed();
            const QString imagePath = imagePathValue.toString().trimmed();
            if (canonicalImageId.isEmpty() || imagePath.isEmpty() || imagesByIndex.contains(cameraIndex) ||
                imageIndexByCanonicalId.contains(canonicalImageId))
            {
                return fail(QStringLiteral("v3 连接点观测文件的 images 身份重复或为空"));
            }
            const SidecarImageIdentity identity{imagePath, image.value(QStringLiteral("image_name")).toString()};
            imagesByIndex.insert(cameraIndex, identity);
            imageIndexByCanonicalId.insert(canonicalImageId, cameraIndex);
        }

        if (root.value(QStringLiteral("points")).isUndefined() || !root.value(QStringLiteral("points")).isArray())
        {
            return fail(QStringLiteral("v3 连接点观测文件缺少 points 数组"));
        }
        const QJsonArray points = root.value(QStringLiteral("points")).toArray();
        for (const QJsonValue& pointValue : points)
        {
            if (!pointValue.isObject())
            {
                return fail(QStringLiteral("v3 连接点观测文件的 points 行必须是对象"));
            }
            const QJsonObject pointObject = pointValue.toObject();
            const double pointResidual = pointObject.value(QStringLiteral("rms_reproj_px")).toDouble(0.0);
            const float response =
                std::isfinite(pointResidual) ? static_cast<float>(1.0 / (1.0 + std::max(0.0, pointResidual))) : 0.0f;
            const QJsonValue observationsValue = pointObject.value(QStringLiteral("observations"));
            if (!observationsValue.isArray())
            {
                return fail(QStringLiteral("v3 连接点观测文件的 points.observations 必须是数组"));
            }
            const QJsonArray observations = observationsValue.toArray();
            for (const QJsonValue& observationValue : observations)
            {
                if (!observationValue.isObject())
                {
                    return fail(QStringLiteral("v3 连接点观测文件的 observations 行必须是对象"));
                }
                const QJsonObject observation = observationValue.toObject();
                int cameraIndex = -1;
                const QJsonValue imageIdValue = observation.value(QStringLiteral("image_id"));
                if (!readNonNegativeInteger(observation.value(QStringLiteral("camera_index")), &cameraIndex) ||
                    !imageIdValue.isString())
                {
                    return fail(QStringLiteral("v3 连接点观测文件的 observation 身份字段无效"));
                }
                const QString canonicalImageId = imageIdValue.toString().trimmed();
                const auto imageIndexIt = imageIndexByCanonicalId.constFind(canonicalImageId);
                const auto imageIt = imagesByIndex.constFind(cameraIndex);
                if (canonicalImageId.isEmpty() || imageIndexIt == imageIndexByCanonicalId.cend() ||
                    imageIt == imagesByIndex.cend() || imageIndexIt.value() != cameraIndex)
                {
                    return fail(QStringLiteral(
                        "v3 连接点观测文件的 observation 的 image_id 与 camera_index 不一致"));
                }
                const QJsonValue observationPathValue = observation.value(QStringLiteral("image_path"));
                if (!observationPathValue.isUndefined() && !observationPathValue.isString())
                {
                    return fail(QStringLiteral("v3 连接点观测文件的 observation.image_path 必须是字符串"));
                }
                if (observationPathValue.isString() && !observationPathValue.toString().trimmed().isEmpty() &&
                    normalizedPath(observationPathValue.toString()) != normalizedPath(imageIt.value().path))
                {
                    bool pointsToAnotherImage = false;
                    for (auto candidate = imagesByIndex.cbegin(); candidate != imagesByIndex.cend(); ++candidate)
                    {
                        if (normalizedPath(observationPathValue.toString()) == normalizedPath(candidate.value().path))
                        {
                            pointsToAnotherImage = candidate.key() != cameraIndex;
                            break;
                        }
                    }
                    if (pointsToAnotherImage)
                    {
                        return fail(QStringLiteral(
                            "v3 连接点观测文件的 observation.image_path 指向另一 camera_index"));
                    }
                }

                const QString rawImagePath = imageIt.value().path;
                QString imageName = observationImageName(observation);
                if (imageName.isEmpty())
                {
                    imageName = imageIt.value().name;
                }
                QPointF observed;
                QPointF projected;
                double scaleValue = 1.0;
                qint64 featureIndex = observation.value(QStringLiteral("feature_idx")).toInteger(-1);
                bool hasObserved = false;
                bool hasProjected = false;
                hasObserved = finitePoint(observation.value(QStringLiteral("xy")).toArray(), &observed);
                scaleValue = observation.value(QStringLiteral("scale")).toDouble(1.0);
                hasProjected = finitePoint(observation.value(QStringLiteral("projected_xy")).toArray(), &projected);
                auto normalizedIt = normalizedPathCache.constFind(rawImagePath);
                if (normalizedIt == normalizedPathCache.cend())
                {
                    normalizedIt = normalizedPathCache.insert(rawImagePath, normalizedPath(rawImagePath));
                }
                const QString& observationPath = normalizedIt.value();
                if (observationPath.isEmpty())
                {
                    continue;
                }

                const QString imageNameKey =
                    imageName.isEmpty() ? QFileInfo(rawImagePath).fileName().toCaseFolded() : imageName.toCaseFolded();
                if (!imageNameKey.isEmpty())
                {
                    index->pathsByImageName[imageNameKey].insert(observationPath);
                }

                if (!hasObserved)
                {
                    continue;
                }

                IndexedImageDiagnostics& diagnostics = index->diagnosticsByPath[observationPath];
                const float scale =
                    std::isfinite(scaleValue) && scaleValue > 0.0 ? static_cast<float>(scaleValue) : 1.0f;
                const int classId = featureIndex >= 0 && featureIndex <= std::numeric_limits<int>::max()
                                        ? static_cast<int>(featureIndex)
                                        : -1;
                diagnostics.keypoints.emplace_back(static_cast<float>(observed.x()),
                                                   static_cast<float>(observed.y()),
                                                   scale,
                                                   -1.0f,
                                                   response,
                                                   0,
                                                   classId);

                if (!hasProjected)
                {
                    continue;
                }
                xjw::gui::views::FeatureResidualVector vector;
                vector.observed = observed;
                vector.projected = projected;
                vector.magnitudePx = std::hypot(projected.x() - observed.x(), projected.y() - observed.y());
                if (std::isfinite(vector.magnitudePx))
                {
                    diagnostics.residuals.append(vector);
                }
            }
        }
        return index;
    }

    SidecarIndexResult loadSidecarObservationIndex(const QString& sidecarPath)
    {
        const QFileInfo info(sidecarPath);
        const QString cachePath = normalizedPath(sidecarPath);
        const qint64 sidecarSize = info.size();
        const qint64 modifiedMilliseconds = info.lastModified().toMSecsSinceEpoch();
        SidecarIndexCache& cache = sidecarIndexCache();

        for (;;)
        {
            QMutexLocker locker(&cache.mutex);
            if (cache.loading)
            {
                cache.loadFinished.wait(&cache.mutex);
                continue;
            }
            if (cache.path == cachePath && cache.size == sidecarSize &&
                cache.modifiedMilliseconds == modifiedMilliseconds && (cache.index || !cache.error.isEmpty()))
            {
                return {cache.index, cache.error, cache.loadMilliseconds, true};
            }

            cache.path = cachePath;
            cache.size = sidecarSize;
            cache.modifiedMilliseconds = modifiedMilliseconds;
            cache.index.reset();
            cache.error.clear();
            cache.loadMilliseconds = 0;
            cache.loading = true;
            break;
        }

        QElapsedTimer timer;
        timer.start();
        std::shared_ptr<const SidecarObservationIndex> index;
        QString error;
        QFile file(sidecarPath);
        if (!file.open(QIODevice::ReadOnly))
        {
            error = QStringLiteral("无法读取连接点观测文件: %1").arg(sidecarPath);
        }
        else
        {
            QJsonParseError parseError;
            const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
            if (parseError.error != QJsonParseError::NoError || !document.isObject())
            {
                error = QStringLiteral("连接点观测文件格式无效（%1）: %2").arg(parseError.errorString(), sidecarPath);
            }
            else
            {
                index = buildSidecarObservationIndex(document, &error);
            }
        }
        const qint64 loadMilliseconds = timer.elapsed();

        {
            QMutexLocker locker(&cache.mutex);
            cache.index = index;
            cache.error = error;
            cache.loadMilliseconds = loadMilliseconds;
            cache.loading = false;
            cache.loadFinished.wakeAll();
        }
        return {index, error, loadMilliseconds, false};
    }

} // namespace

namespace xjw::gui::views
{

    ValidTiePointDiagnostics loadValidTiePointDiagnosticsFromSidecar(const QString& sidecarPath,
                                                                     const QString& imagePath)
    {
        ValidTiePointDiagnostics result;
        result.sidecarPath = sidecarPath;
        if (sidecarPath.trimmed().isEmpty() || imagePath.trimmed().isEmpty())
        {
            result.message = QStringLiteral("当前项目或照片路径为空");
            return result;
        }

        const SidecarIndexResult indexResult = loadSidecarObservationIndex(sidecarPath);
        result.loadMilliseconds = indexResult.loadMilliseconds;
        result.loadedFromCache = indexResult.loadedFromCache;
        if (!indexResult.index)
        {
            result.message = indexResult.error;
            return result;
        }

        const QString targetPath = normalizedPath(imagePath);
        const QString targetNameKey = QFileInfo(imagePath).fileName().toCaseFolded();
        QString matchedPath = targetPath;
        auto diagnosticsIt = indexResult.index->diagnosticsByPath.constFind(matchedPath);
        if (diagnosticsIt == indexResult.index->diagnosticsByPath.cend())
        {
            const QSet<QString> pathsForTargetName = indexResult.index->pathsByImageName.value(targetNameKey);
            if (!targetNameKey.isEmpty() && pathsForTargetName.size() == 1)
            {
                matchedPath = *pathsForTargetName.cbegin();
                diagnosticsIt = indexResult.index->diagnosticsByPath.constFind(matchedPath);
                result.usedUniqueNameFallback = true;
            }
        }
        if (diagnosticsIt != indexResult.index->diagnosticsByPath.cend())
        {
            result.keypoints = diagnosticsIt->keypoints;
            result.residuals = diagnosticsIt->residuals;
        }

        result.available = !result.keypoints.empty();
        if (result.available)
        {
            result.message =
                QStringLiteral("有效连接点 %1 个，残差向量 %2 个%3")
                    .arg(result.keypoints.size())
                    .arg(result.residuals.size())
                    .arg(result.usedUniqueNameFallback ? QStringLiteral("（按唯一照片名称匹配）")
                         : result.loadedFromCache      ? QStringLiteral("（索引缓存命中）")
                                                  : QStringLiteral("（首次索引 %1 ms）").arg(result.loadMilliseconds));
        }
        else
        {
            result.message = QStringLiteral("当前照片没有可用的有效连接点观测");
        }
        return result;
    }

    ValidTiePointDiagnostics loadValidTiePointDiagnosticsForImage(const QString& projectPath, const QString& imagePath)
    {
        ValidTiePointDiagnostics result;
        if (projectPath.trimmed().isEmpty() || imagePath.trimmed().isEmpty())
        {
            result.message = QStringLiteral("当前项目或照片路径为空");
            return result;
        }
        const ProjectResultsSnapshot snapshot = ProjectData::loadProjectResultsSnapshot(projectPath);
        if (!snapshot.success)
        {
            result.message = snapshot.errorMessage;
            return result;
        }
        const QString sidecarPath = latestSidecarPath(snapshot.resultsMeta, projectPath);
        if (sidecarPath.isEmpty())
        {
            result.message = QStringLiteral("当前项目没有可用的稀疏连接点观测文件");
            return result;
        }
        return loadValidTiePointDiagnosticsFromSidecar(sidecarPath, imagePath);
    }

    QVector<FeatureResidualVector>
    loadFeatureResidualsForImage(const QString& projectPath, const QString& imagePath, QString* errorMessage)
    {
        ValidTiePointDiagnostics result = loadValidTiePointDiagnosticsForImage(projectPath, imagePath);
        if (errorMessage)
        {
            *errorMessage = result.available ? QString() : result.message;
        }
        return result.residuals;
    }

} // namespace xjw::gui::views
