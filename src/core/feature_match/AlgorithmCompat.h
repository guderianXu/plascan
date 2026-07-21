#pragma once

#include <QString>
#include <QStringList>
#include <QMap>
#include <QFileInfo>

namespace xjw {
namespace feature_match {

inline const QMap<QString, QStringList> &algorithmFeatureMap()
{
    static const QMap<QString, QStringList> map = {
        {"superglue",       {".sp"}},
        {"lightglue",       {".sp", ".dsk", ".alk", ".sift"}},
        {"loftr",           {}},
        {"roma",            {}},
        {"dedode",          {".dedode"}},
        {"orb_bf_hamming",  {".orb", ".akz"}},
        {"sift_bf_l2",      {".sift", ".surf"}},
        {"sift_flann",      {".sift", ".surf"}},
    };
    return map;
}

inline QStringList compatibleFeatureSuffixes(const QString &algo)
{
    return algorithmFeatureMap().value(algo);
}

inline QString normalizedFeatureSuffix(const QString &pathOrSuffix)
{
    QString suffix = pathOrSuffix.trimmed().toLower();
    if (suffix.isEmpty())
    {
        return QString();
    }
    if (!suffix.startsWith(QLatin1Char('.')))
    {
        const QString fileSuffix = QFileInfo(suffix).suffix().toLower();
        if (!fileSuffix.isEmpty())
        {
            suffix = fileSuffix;
        }
        else if (suffix.contains(QLatin1Char('/')) || suffix.contains(QLatin1Char('\\')))
        {
            return QString();
        }
        suffix.prepend(QLatin1Char('.'));
    }
    return suffix;
}

inline QStringList normalizedFeatureSuffixes(const QStringList &suffixes)
{
    QStringList normalized;
    for (const QString &suffix : suffixes)
    {
        const QString value = normalizedFeatureSuffix(suffix);
        if (!value.isEmpty() && !normalized.contains(value))
        {
            normalized.append(value);
        }
    }
    return normalized;
}

inline QString featureAlgorithmForSuffix(const QString &pathOrSuffix)
{
    static const QMap<QString, QString> algorithms = {
        {QStringLiteral(".dsk"), QStringLiteral("disk")},
        {QStringLiteral(".alk"), QStringLiteral("aliked")},
        {QStringLiteral(".sp"), QStringLiteral("superpoint")},
        {QStringLiteral(".sift"), QStringLiteral("sift")},
        {QStringLiteral(".orb"), QStringLiteral("orb")},
        {QStringLiteral(".akz"), QStringLiteral("akaze")},
        {QStringLiteral(".dedode"), QStringLiteral("dedode")},
    };
    return algorithms.value(normalizedFeatureSuffix(pathOrSuffix));
}

inline QString defaultMatcherForFeatureSuffix(const QString &pathOrSuffix)
{
    const QString suffix = normalizedFeatureSuffix(pathOrSuffix);
    if (suffix == QStringLiteral(".sp"))
    {
        return QStringLiteral("superglue");
    }
    if (suffix == QStringLiteral(".dedode"))
    {
        return QStringLiteral("dedode");
    }
    if (suffix == QStringLiteral(".dsk") ||
        suffix == QStringLiteral(".alk") ||
        suffix == QStringLiteral(".sift"))
    {
        return QStringLiteral("lightglue");
    }
    if (suffix == QStringLiteral(".orb") || suffix == QStringLiteral(".akz"))
    {
        return QStringLiteral("orb_bf_hamming");
    }
    if (suffix == QStringLiteral(".surf"))
    {
        return QStringLiteral("sift_bf_l2");
    }
    return QStringLiteral("superglue");
}

inline bool isEndToEndAlgorithm(const QString &algo)
{
    const auto &map = algorithmFeatureMap();
    return map.contains(algo) && map.value(algo).isEmpty();
}

inline QString algorithmDisplayName(const QString &algo)
{
    static const QMap<QString, QString> names = {
        {"superglue",       "SuperGlue"},
        {"lightglue",       "LightGlue"},
        {"loftr",           "LoFTR"},
        {"roma",            "RoMa"},
        {"dedode",          "DeDoDe"},
        {"orb_bf_hamming",  "BF-Hamming (ORB)"},
        {"sift_bf_l2",      "BF-L2 (SIFT)"},
        {"sift_flann",      "FLANN (SIFT)"},
    };
    return names.value(algo, algo);
}

} // namespace feature_match
} // namespace xjw
