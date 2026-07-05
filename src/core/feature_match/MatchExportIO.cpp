#include "MatchExportIO.h"

#include <QFile>
#include <QTextStream>

namespace xjw::feature_match
{

bool exportMatchCsv(const QString &path,
                    const MatchResult &result)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        return false;
    }

    QTextStream out(&file);
    out << "idx0,idx1,score\n";
    for (size_t i = 0; i < result.matches0.size(); ++i)
    {
        if (result.matches0[i] < 0)
        {
            continue;
        }

        const float score = (i < result.matchingScores0.size()) ? result.matchingScores0[i] : 0.0f;
        out << i << "," << result.matches0[i] << "," << score << "\n";
    }
    return true;
}

bool appendMatchDebugCsv(const QString &path,
                         const QString &imagePairName,
                         const MatchResult &result)
{
    const bool exists = QFile::exists(path);
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        return false;
    }

    QTextStream out(&file);
    if (!exists || file.size() == 0)
    {
        out << "image_pair,keypoint0_idx,keypoint1_idx,matching_score,distance\n";
    }

    const QString pairName = imagePairName.isEmpty() ? QStringLiteral("pair") : imagePairName;
    for (const auto &match : result.cvMatches)
    {
        out << pairName << ","
            << match.queryIdx << ","
            << match.trainIdx << ","
            << (1.0f - match.distance) << ","
            << match.distance << "\n";
    }
    return true;
}

bool exportMatchColmap(const QString &path,
                       const QString &image0Name,
                       const QString &image1Name,
                       const MatchResult &result)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        return false;
    }

    QTextStream out(&file);
    out << "# PlaScan matches export\n";
    out << "# image0: " << image0Name << "\n";
    out << "# image1: " << image1Name << "\n";
    out << "# num_matches: " << result.numMatches << "\n\n";
    out << image0Name << " " << image1Name << "\n";
    for (const auto &match : result.cvMatches)
    {
        out << match.queryIdx << " " << match.trainIdx << " ";
    }
    out << "\n";
    return true;
}

} // namespace xjw::feature_match
