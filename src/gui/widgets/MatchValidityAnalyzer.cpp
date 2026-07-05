#include "MatchValidityAnalyzer.h"

#include <QDataStream>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSet>

#include <algorithm>
#include <cstring>

namespace
{

struct MatchIndexPair
{
    int indexA = -1;
    int indexB = -1;
};

struct MatchIndexData
{
    bool ok = false;
    QVector<MatchIndexPair> pairs;
};

QString normalizedImageToken(const QString &token)
{
    QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(token.trimmed()));
    normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return normalized.toLower();
}

QString imageBaseToken(const QString &token)
{
    const QString base = QFileInfo(token.trimmed()).completeBaseName();
    return (base.isEmpty() ? token.trimmed() : base).toLower();
}

bool imageTokenMatches(const QString &filePath,
                       const QString &fileName,
                       const QString &displayPath)
{
    if (displayPath.trimmed().isEmpty())
    {
        return false;
    }

    const QString displayNorm = normalizedImageToken(displayPath);
    const QString displayBase = imageBaseToken(displayPath);

    if (!filePath.trimmed().isEmpty())
    {
        if (normalizedImageToken(filePath) == displayNorm ||
            imageBaseToken(filePath) == displayBase)
        {
            return true;
        }
    }

    return !fileName.trimmed().isEmpty() && imageBaseToken(fileName) == displayBase;
}

enum class DisplayOrder
{
    Direct,
    Reversed,
    Unknown
};

DisplayOrder displayOrderForImages(const QString &fileImage0Path,
                                   const QString &fileImage0Name,
                                   const QString &fileImage1Path,
                                   const QString &fileImage1Name,
                                   const QString &displayImageA,
                                   const QString &displayImageB)
{
    const bool direct =
        imageTokenMatches(fileImage0Path, fileImage0Name, displayImageA) &&
        imageTokenMatches(fileImage1Path, fileImage1Name, displayImageB);
    const bool reversed =
        imageTokenMatches(fileImage0Path, fileImage0Name, displayImageB) &&
        imageTokenMatches(fileImage1Path, fileImage1Name, displayImageA);

    if (direct)
    {
        return DisplayOrder::Direct;
    }
    if (reversed)
    {
        return DisplayOrder::Reversed;
    }
    return DisplayOrder::Unknown;
}

QJsonObject readJsonObject(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return QJsonObject();
    }

    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
    {
        return QJsonObject();
    }

    return doc.object();
}

QVector<int> intArrayFromJson(const QJsonArray &array)
{
    QVector<int> values;
    values.reserve(array.size());
    for (const QJsonValue &value : array)
    {
        values.append(value.toInt(-1));
    }
    return values;
}

MatchIndexData readMatchIndicesFromSidecar(const QString &matchFile,
                                           const QString &displayImageA,
                                           const QString &displayImageB)
{
    MatchIndexData data;
    const QJsonObject sidecar = readJsonObject(matchFile + QStringLiteral(".json"));
    if (sidecar.isEmpty())
    {
        return data;
    }

    const QVector<int> indices0 = intArrayFromJson(sidecar.value(QStringLiteral("matched_indices0")).toArray());
    const QVector<int> indices1 = intArrayFromJson(sidecar.value(QStringLiteral("matched_indices1")).toArray());
    if (indices0.isEmpty() || indices0.size() != indices1.size())
    {
        return data;
    }

    const DisplayOrder order = displayOrderForImages(
        sidecar.value(QStringLiteral("image0_path")).toString(),
        sidecar.value(QStringLiteral("image0_name")).toString(),
        sidecar.value(QStringLiteral("image1_path")).toString(),
        sidecar.value(QStringLiteral("image1_name")).toString(),
        displayImageA,
        displayImageB);
    if (order == DisplayOrder::Unknown)
    {
        return data;
    }
    const bool reversed = order == DisplayOrder::Reversed;

    data.ok = true;
    data.pairs.reserve(indices0.size());
    for (int i = 0; i < indices0.size(); ++i)
    {
        const int a = reversed ? indices1.at(i) : indices0.at(i);
        const int b = reversed ? indices0.at(i) : indices1.at(i);
        if (a >= 0 && b >= 0)
        {
            data.pairs.append({a, b});
        }
    }
    return data;
}

bool readSgmtHeaderAndMatches(const QString &matchFile,
                              QString *image0Name,
                              QString *image1Name,
                              QVector<int> *matches0)
{
    QFile file(matchFile);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }

    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);

    char magic[4] = {};
    if (in.readRawData(magic, 4) != 4 || std::strncmp(magic, "SGMT", 4) != 0)
    {
        return false;
    }

    quint32 version = 0;
    in >> version;
    if (version != 1)
    {
        return false;
    }

    quint32 image0Length = 0;
    quint32 image1Length = 0;
    in >> image0Length;
    QByteArray image0Bytes(static_cast<int>(image0Length), 0);
    if (in.readRawData(image0Bytes.data(), static_cast<int>(image0Length)) != static_cast<int>(image0Length))
    {
        return false;
    }
    *image0Name = QString::fromUtf8(image0Bytes);

    in >> image1Length;
    QByteArray image1Bytes(static_cast<int>(image1Length), 0);
    if (in.readRawData(image1Bytes.data(), static_cast<int>(image1Length)) != static_cast<int>(image1Length))
    {
        return false;
    }
    *image1Name = QString::fromUtf8(image1Bytes);

    qint32 numMatches = 0;
    qint32 numKeypoints0 = 0;
    qint32 numKeypoints1 = 0;
    in >> numMatches >> numKeypoints0 >> numKeypoints1;
    Q_UNUSED(numMatches);
    Q_UNUSED(numKeypoints1);
    if (numKeypoints0 < 0)
    {
        return false;
    }

    matches0->resize(numKeypoints0);
    for (int i = 0; i < numKeypoints0; ++i)
    {
        qint32 matchIndex = -1;
        float score = 0.0f;
        in >> matchIndex >> score;
        Q_UNUSED(score);
        (*matches0)[i] = static_cast<int>(matchIndex);
    }

    return in.status() == QDataStream::Ok;
}

MatchIndexData readMatchIndicesFromSgmt(const QString &matchFile,
                                        const QString &displayImageA,
                                        const QString &displayImageB)
{
    MatchIndexData data;
    QString image0Name;
    QString image1Name;
    QVector<int> matches0;
    if (!readSgmtHeaderAndMatches(matchFile, &image0Name, &image1Name, &matches0))
    {
        return data;
    }

    const DisplayOrder order = displayOrderForImages(
        QString(), image0Name, QString(), image1Name, displayImageA, displayImageB);
    if (order == DisplayOrder::Unknown)
    {
        return data;
    }
    const bool reversed = order == DisplayOrder::Reversed;

    data.ok = true;
    data.pairs.reserve(matches0.size());
    for (int index0 = 0; index0 < matches0.size(); ++index0)
    {
        const int index1 = matches0.at(index0);
        if (index1 < 0)
        {
            continue;
        }
        data.pairs.append(reversed ? MatchIndexPair{index1, index0}
                                   : MatchIndexPair{index0, index1});
    }
    return data;
}

MatchIndexData readDisplayMatchIndices(const QString &matchFile,
                                       const QString &displayImageA,
                                       const QString &displayImageB)
{
    MatchIndexData data = readMatchIndicesFromSidecar(matchFile, displayImageA, displayImageB);
    if (data.ok)
    {
        return data;
    }
    return readMatchIndicesFromSgmt(matchFile, displayImageA, displayImageB);
}

QStringList candidateSparseSidecars(const QString &matchFile)
{
    QStringList candidates;
    const QFileInfo matchInfo(matchFile);
    const QString matchDir = matchInfo.absolutePath();
    const QString assetsDir = QDir(matchDir).absoluteFilePath(QStringLiteral(".."));

    QDirIterator it(assetsDir,
                    QStringList{QStringLiteral("sfm_sparse_points.json"),
                                QStringLiteral("sparse_cloud_points.json")},
                    QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext())
    {
        candidates.append(QDir::cleanPath(it.next()));
    }

    std::sort(candidates.begin(), candidates.end(), [](const QString &a, const QString &b)
    {
        return QFileInfo(a).lastModified() > QFileInfo(b).lastModified();
    });
    return candidates;
}

QString matchPairKey(int indexA, int indexB)
{
    return QString::number(indexA) + QLatin1Char('|') + QString::number(indexB);
}

QSet<QString> validPairsFromSparseSidecar(const QString &sidecarPath,
                                          const QString &displayImageA,
                                          const QString &displayImageB,
                                          bool *hasObservationSchema)
{
    QSet<QString> validPairs;
    if (hasObservationSchema)
    {
        *hasObservationSchema = false;
    }

    const QJsonObject root = readJsonObject(sidecarPath);
    const QJsonArray points = root.value(QStringLiteral("points")).toArray();
    for (const QJsonValue &pointValue : points)
    {
        const QJsonArray observations =
            pointValue.toObject().value(QStringLiteral("observations")).toArray();
        if (observations.isEmpty())
        {
            continue;
        }
        if (hasObservationSchema)
        {
            *hasObservationSchema = true;
        }

        QVector<int> indicesA;
        QVector<int> indicesB;
        for (const QJsonValue &observationValue : observations)
        {
            const QJsonObject observation = observationValue.toObject();
            const int featureIndex = observation.value(QStringLiteral("feature_idx")).toInt(-1);
            if (featureIndex < 0)
            {
                continue;
            }

            const QString imagePath = observation.value(QStringLiteral("image_path")).toString();
            const QString imageName = observation.value(QStringLiteral("image_name")).toString();
            if (imageTokenMatches(imagePath, imageName, displayImageA))
            {
                indicesA.append(featureIndex);
            }
            if (imageTokenMatches(imagePath, imageName, displayImageB))
            {
                indicesB.append(featureIndex);
            }
        }

        for (int indexA : indicesA)
        {
            for (int indexB : indicesB)
            {
                validPairs.insert(matchPairKey(indexA, indexB));
            }
        }
    }

    return validPairs;
}

} // namespace

MatchValidityResult analyzeMatchTrackValidity(const QString &matchFile,
                                              const QString &displayImageA,
                                              const QString &displayImageB)
{
    MatchValidityResult result;
    if (matchFile.trimmed().isEmpty())
    {
        return result;
    }

    const MatchIndexData matchIndices = readDisplayMatchIndices(matchFile, displayImageA, displayImageB);
    if (!matchIndices.ok || matchIndices.pairs.isEmpty())
    {
        return result;
    }

    for (const QString &sidecarPath : candidateSparseSidecars(matchFile))
    {
        bool hasObservationSchema = false;
        const QSet<QString> validPairs = validPairsFromSparseSidecar(
            sidecarPath, displayImageA, displayImageB, &hasObservationSchema);
        if (!hasObservationSchema)
        {
            continue;
        }

        result.hasTrackValidity = true;
        result.sparseSidecarPath = sidecarPath;
        result.inlierMask.reserve(matchIndices.pairs.size());
        for (const MatchIndexPair &pair : matchIndices.pairs)
        {
            const bool valid = validPairs.contains(matchPairKey(pair.indexA, pair.indexB));
            result.inlierMask.append(valid);
            if (valid)
            {
                ++result.validCount;
            }
            else
            {
                ++result.invalidCount;
            }
        }
        return result;
    }

    return result;
}
