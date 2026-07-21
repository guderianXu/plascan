#include "ProjectMatchCatalog.h"

#include "ProjectIO.h"
#include "ProjectMetadata.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>

#include <algorithm>

namespace xjw::common::project
{

QString encodeImagePairKey(const QString &first,
                           const QString &second,
                           const QString &separator)
{
    const QString trimmed_first = first.trimmed();
    const QString trimmed_second = second.trimmed();
    if (separator.isEmpty() || trimmed_first.isEmpty() || trimmed_second.isEmpty() ||
        trimmed_first == trimmed_second)
    {
        return QString();
    }
    return trimmed_first + separator + trimmed_second;
}

QString canonicalImagePairKey(const QString &left,
                              const QString &right,
                              const QString &separator)
{
    const QString trimmed_left = left.trimmed();
    const QString trimmed_right = right.trimmed();
    if (trimmed_left <= trimmed_right)
    {
        return encodeImagePairKey(trimmed_left, trimmed_right, separator);
    }
    return encodeImagePairKey(trimmed_right, trimmed_left, separator);
}

bool decodeImagePairKey(const QString &key,
                        const QString &separator,
                        QString *first,
                        QString *second)
{
    if (first)
    {
        first->clear();
    }
    if (second)
    {
        second->clear();
    }
    if (!first || !second || separator.isEmpty())
    {
        return false;
    }

    const int separator_index = key.indexOf(separator);
    if (separator_index < 0 || separator_index != key.lastIndexOf(separator))
    {
        return false;
    }

    const QString decoded_first = key.left(separator_index).trimmed();
    const QString decoded_second = key.mid(separator_index + separator.size()).trimmed();
    if (decoded_first.isEmpty() || decoded_second.isEmpty())
    {
        return false;
    }

    *first = decoded_first;
    *second = decoded_second;
    return true;
}

namespace
{

QPair<QString, QString> canonicalPair(const QString &left, const QString &right)
{
    return left <= right ? qMakePair(left, right) : qMakePair(right, left);
}

QString pairKey(const QString &left, const QString &right)
{
    return canonicalImagePairKey(left, right, QStringLiteral("\n"));
}

void appendPair(QVector<QPair<QString, QString>> *pairs,
                QSet<QString> *seen,
                const QString &left,
                const QString &right)
{
    if (!pairs || !seen)
    {
        return;
    }
    const QString trimmed_left = left.trimmed();
    const QString trimmed_right = right.trimmed();
    if (trimmed_left.isEmpty() || trimmed_right.isEmpty() || trimmed_left == trimmed_right)
    {
        return;
    }
    const QString key = pairKey(trimmed_left, trimmed_right);
    if (seen->contains(key))
    {
        return;
    }
    seen->insert(key);
    pairs->push_back(canonicalPair(trimmed_left, trimmed_right));
}

void appendPairFromStem(QVector<QPair<QString, QString>> *pairs,
                        QSet<QString> *seen,
                        const QString &stem)
{
    const QStringList double_underscore = stem.split(QStringLiteral("__"));
    if (double_underscore.size() == 2)
    {
        appendPair(pairs, seen, double_underscore.at(0), double_underscore.at(1));
        return;
    }
    const QStringList dash = stem.split(QLatin1Char('-'));
    if (dash.size() == 2)
    {
        appendPair(pairs, seen, dash.at(0), dash.at(1));
    }
}

QString resolveDisplayName(const QString &token, const QJsonArray &image_entries)
{
    const QString trimmed = token.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }
    for (const QJsonValue &value : image_entries)
    {
        const QString image_path = value.toObject().value(QStringLiteral("path")).toString();
        if (!image_path.isEmpty() && pathTokenMatchesImage(trimmed, image_path))
        {
            const QString file_name = QFileInfo(image_path).fileName();
            return file_name.isEmpty() ? image_path : file_name;
        }
    }
    const QString file_name = QFileInfo(trimmed).fileName();
    return file_name.isEmpty() ? trimmed : file_name;
}

void sortPairs(QVector<QPair<QString, QString>> *pairs)
{
    std::sort(pairs->begin(), pairs->end(),
              [](const auto &left, const auto &right)
              {
                  return left.first == right.first
                      ? left.second < right.second
                      : left.first < right.first;
              });
}

} // namespace

QVector<QPair<QString, QString>> collectMatchedImageNamePairs(
    const QString &project_path,
    const QJsonObject &metadata)
{
    QVector<QPair<QString, QString>> pairs;
    QSet<QString> seen;
    const QJsonArray image_entries = projectImageEntries(metadata);
    const auto append_resolved = [&](const QString &left, const QString &right)
    {
        appendPair(&pairs,
                   &seen,
                   resolveDisplayName(left, image_entries),
                   resolveDisplayName(right, image_entries));
    };

    if (!project_path.isEmpty())
    {
        const QDir match_dir(ProjectIO::ipmatchOutputDir(project_path));
        const QStringList files = match_dir.entryList(
            QStringList{QStringLiteral("*.match")}, QDir::Files, QDir::Name);
        for (const QString &file_name : files)
        {
            QFile sidecar(match_dir.filePath(file_name + QStringLiteral(".json")));
            if (sidecar.open(QIODevice::ReadOnly))
            {
                const QJsonObject object = QJsonDocument::fromJson(sidecar.readAll()).object();
                const QString image0 = object.value(QStringLiteral("image0_name")).toString();
                const QString image1 = object.value(QStringLiteral("image1_name")).toString();
                append_resolved(image0, image1);
                if (!image0.isEmpty() && !image1.isEmpty())
                {
                    continue;
                }
            }
            const QString stem = QFileInfo(file_name).completeBaseName();
            QVector<QPair<QString, QString>> raw_pairs;
            QSet<QString> raw_seen;
            appendPairFromStem(&raw_pairs, &raw_seen, stem);
            for (const auto &pair : raw_pairs)
            {
                append_resolved(pair.first, pair.second);
            }
        }
    }

    const QJsonArray match_results = metadata.value(QStringLiteral("ipmatch_results")).toArray();
    for (const QJsonValue &value : match_results)
    {
        const QJsonObject object = value.toObject();
        QString image0 = object.value(QStringLiteral("image0_name")).toString();
        QString image1 = object.value(QStringLiteral("image1_name")).toString();
        if (image0.isEmpty())
        {
            image0 = QFileInfo(object.value(QStringLiteral("image0")).toString()).completeBaseName();
        }
        if (image1.isEmpty())
        {
            image1 = QFileInfo(object.value(QStringLiteral("image1")).toString()).completeBaseName();
        }
        append_resolved(image0, image1);
    }

    sortPairs(&pairs);
    return pairs;
}

QVector<QPair<QString, QString>> collectSettledNoMatchImageNamePairs(
    const QString &project_path,
    const QJsonObject &metadata)
{
    QVector<QPair<QString, QString>> pairs;
    if (project_path.isEmpty())
    {
        return pairs;
    }

    QFile file(QDir(ProjectIO::ipmatchOutputDir(project_path))
                   .filePath(QStringLiteral("no_match_pairs.json")));
    if (!file.open(QIODevice::ReadOnly))
    {
        return pairs;
    }

    QSet<QString> seen;
    const QJsonArray image_entries = projectImageEntries(metadata);
    for (const QJsonValue &value : QJsonDocument::fromJson(file.readAll()).array())
    {
        const QJsonObject object = value.toObject();
        appendPair(&pairs,
                   &seen,
                   resolveDisplayName(object.value(QStringLiteral("image0")).toString(),
                                      image_entries),
                   resolveDisplayName(object.value(QStringLiteral("image1")).toString(),
                                      image_entries));
    }
    sortPairs(&pairs);
    return pairs;
}

} // namespace xjw::common::project
