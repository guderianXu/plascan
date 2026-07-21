#include "ProjectMatchInputReader.h"

#include "project/ProjectCommonUtils.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>

#include <cmath>
#include <limits>

namespace xjw::core::project
{
namespace
{

bool readFeatureIndex(const QJsonArray &indices, int index, xjw::FeatureIdx *featureIndex)
{
    if (!featureIndex || index < 0 || index >= indices.size())
    {
        return false;
    }

    const double raw = indices.at(index).toDouble(-1.0);
    if (!std::isfinite(raw) || raw < 0.0
        || raw > static_cast<double>(std::numeric_limits<xjw::FeatureIdx>::max()))
    {
        return false;
    }

    const double rounded = std::floor(raw);
    if (std::fabs(raw - rounded) > 1e-6)
    {
        return false;
    }

    *featureIndex = static_cast<xjw::FeatureIdx>(rounded);
    return true;
}

} // namespace

int cameraIndexForImageToken(const QString &imageToken,
                             const QMap<QString, int> &cameraIndexByPath)
{
    if (imageToken.trimmed().isEmpty())
    {
        return -1;
    }

    const QString normalizedToken = xjw::common::project::normalizePath(imageToken);
    const auto direct = cameraIndexByPath.constFind(normalizedToken);
    if (direct != cameraIndexByPath.constEnd())
    {
        return direct.value();
    }

    for (auto it = cameraIndexByPath.constBegin(); it != cameraIndexByPath.constEnd(); ++it)
    {
        if (xjw::common::project::pathTokenMatchesImage(imageToken, it.key()))
        {
            return it.value();
        }
    }
    return -1;
}

bool readProjectMatchInput(const QJsonObject &meta,
                           const QStringList &selectedImages,
                           int minMatches,
                           ProjectMatchInput *input)
{
    if (!input)
    {
        return false;
    }

    *input = {};
    QSet<QString> selectedNormalized;
    for (const QString &path : selectedImages)
    {
        selectedNormalized.insert(xjw::common::project::normalizePath(path));
    }

    const QJsonArray imageArray = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : imageArray)
    {
        const QJsonObject object = value.toObject();
        const QString normalizedPath =
            xjw::common::project::normalizePath(object.value(QStringLiteral("path")).toString());
        if (!selectedNormalized.contains(normalizedPath))
        {
            continue;
        }

        xjw::Camera camera;
        if (!xjw::common::project::cameraFromJson(
                object.value(QStringLiteral("camera")).toObject(), &camera))
        {
            continue;
        }

        input->cameraIndexByPath[normalizedPath] = static_cast<int>(input->cameras.size());
        input->cameras.push_back(camera);
        input->imagePathByIndex.append(normalizedPath);
        input->beforeCamMeta.insert(normalizedPath,
                                    object.value(QStringLiteral("camera")).toObject());
    }

    const QJsonArray matchResults = meta.value(QStringLiteral("ipmatch_results")).toArray();
    for (const QJsonValue &value : matchResults)
    {
        if (!value.isObject())
        {
            continue;
        }

        const QJsonObject record = value.toObject();
        QString rawPath0 = record.value(QStringLiteral("image0")).toString();
        QString rawPath1 = record.value(QStringLiteral("image1")).toString();
        if (rawPath0.isEmpty() || rawPath1.isEmpty())
        {
            const QJsonArray imageFiles = record.value(QStringLiteral("settings"))
                                              .toObject()
                                              .value(QStringLiteral("image_files"))
                                              .toArray();
            if (imageFiles.size() >= 2)
            {
                rawPath0 = imageFiles.at(0).toString();
                rawPath1 = imageFiles.at(1).toString();
            }
        }
        if (rawPath0.isEmpty() || rawPath1.isEmpty())
        {
            continue;
        }

        QString imageA;
        QString imageB;
        for (const QString &selected : selectedImages)
        {
            if (xjw::common::project::pathTokenMatchesImage(rawPath0, selected))
            {
                imageA = xjw::common::project::normalizePath(selected);
            }
            if (xjw::common::project::pathTokenMatchesImage(rawPath1, selected))
            {
                imageB = xjw::common::project::normalizePath(selected);
            }
        }

        if (imageA.isEmpty() || imageB.isEmpty() || imageA == imageB
            || !input->cameraIndexByPath.contains(imageA)
            || !input->cameraIndexByPath.contains(imageB))
        {
            continue;
        }

        QString sidecarPath = record.value(QStringLiteral("settings"))
                                  .toObject()
                                  .value(QStringLiteral("sidecar_json"))
                                  .toString();
        if (sidecarPath.isEmpty())
        {
            sidecarPath = record.value(QStringLiteral("output")).toString()
                        + QStringLiteral(".json");
        }

        QFile sidecarFile(sidecarPath);
        if (!sidecarFile.exists() || !sidecarFile.open(QIODevice::ReadOnly))
        {
            continue;
        }
        const QJsonDocument sidecarDocument = QJsonDocument::fromJson(sidecarFile.readAll());
        if (!sidecarDocument.isObject())
        {
            continue;
        }

        const QJsonObject sidecar = sidecarDocument.object();
        const QJsonArray matched0 = sidecar.value(QStringLiteral("matched_points0")).toArray();
        const QJsonArray matched1 = sidecar.value(QStringLiteral("matched_points1")).toArray();
        if (matched0.isEmpty() || matched0.size() != matched1.size()
            || (minMatches > 0 && matched0.size() < minMatches))
        {
            continue;
        }

        const QString sideImage0 = sidecar.value(QStringLiteral("image0_path")).toString();
        const QString sideImage1 = sidecar.value(QStringLiteral("image1_path")).toString();
        const bool direct = xjw::common::project::pathTokenMatchesImage(sideImage0, imageA)
                         && xjw::common::project::pathTokenMatchesImage(sideImage1, imageB);
        const bool reverse = xjw::common::project::pathTokenMatchesImage(sideImage0, imageB)
                          && xjw::common::project::pathTokenMatchesImage(sideImage1, imageA);
        if (!direct && !reverse)
        {
            continue;
        }

        const QJsonArray indices0 = sidecar.value(QStringLiteral("matched_indices0")).toArray();
        const QJsonArray indices1 = sidecar.value(QStringLiteral("matched_indices1")).toArray();
        const QJsonArray matchedScores = sidecar.value(QStringLiteral("matched_scores")).toArray();
        const bool hasIndexedMatches = !indices0.isEmpty()
                                    && indices0.size() == matched0.size()
                                    && indices1.size() == matched1.size();
        const bool hasMatchScores = matchedScores.size() == matched0.size();

        ProjectMatchPair pair;
        pair.cameraIndexA = input->cameraIndexByPath.value(imageA);
        pair.cameraIndexB = input->cameraIndexByPath.value(imageB);
        pair.indexed = hasIndexedMatches;
        pair.observations.reserve(static_cast<std::size_t>(matched0.size()));
        for (int pointIndex = 0; pointIndex < matched0.size(); ++pointIndex)
        {
            const QJsonArray point0 = matched0.at(pointIndex).toArray();
            const QJsonArray point1 = matched1.at(pointIndex).toArray();
            if (point0.size() < 2 || point1.size() < 2)
            {
                continue;
            }

            ProjectMatchObservationPair observation;
            observation.pixelA = direct
                ? std::array<double, 2>{point0.at(0).toDouble(), point0.at(1).toDouble()}
                : std::array<double, 2>{point1.at(0).toDouble(), point1.at(1).toDouble()};
            observation.pixelB = direct
                ? std::array<double, 2>{point1.at(0).toDouble(), point1.at(1).toDouble()}
                : std::array<double, 2>{point0.at(0).toDouble(), point0.at(1).toDouble()};
            observation.score = hasMatchScores
                ? matchedScores.at(pointIndex).toDouble(1.0)
                : 1.0;

            if (hasIndexedMatches)
            {
                xjw::FeatureIdx feature0 = xjw::kInvalidFeatureIdx;
                xjw::FeatureIdx feature1 = xjw::kInvalidFeatureIdx;
                if (!readFeatureIndex(indices0, pointIndex, &feature0)
                    || !readFeatureIndex(indices1, pointIndex, &feature1))
                {
                    continue;
                }
                observation.featureA = direct ? feature0 : feature1;
                observation.featureB = direct ? feature1 : feature0;
                ++input->sidecarV2PairCount;
            }
            pair.observations.push_back(observation);
        }

        if (!pair.observations.empty())
        {
            input->pairs.push_back(std::move(pair));
        }
    }
    return true;
}

} // namespace xjw::core::project
