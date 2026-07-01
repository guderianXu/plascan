#pragma once

#include <QString>
#include <QStringList>
#include <QMap>

namespace xjw {
namespace feature_match {

inline const QMap<QString, QStringList> &algorithmFeatureMap()
{
    static const QMap<QString, QStringList> map = {
        {"superglue",       {".sp"}},
        {"lightglue",       {".sp", ".dsk", ".alk", ".sift"}},
        {"loftr",           {}},
        {"roma",            {}},
        {"orb_bf_hamming",  {".orb"}},
        {"sift_bf_l2",      {".sift"}},
        {"sift_flann",      {".sift"}},
    };
    return map;
}

inline QStringList compatibleFeatureSuffixes(const QString &algo)
{
    return algorithmFeatureMap().value(algo);
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
        {"orb_bf_hamming",  "BF-Hamming (ORB)"},
        {"sift_bf_l2",      "BF-L2 (SIFT)"},
        {"sift_flann",      "FLANN (SIFT)"},
    };
    return names.value(algo, algo);
}

} // namespace feature_match
} // namespace xjw
