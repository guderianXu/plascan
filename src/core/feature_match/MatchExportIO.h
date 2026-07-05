#pragma once

#include "match.h"

#include <QString>

namespace xjw::feature_match
{

bool exportMatchCsv(const QString &path,
                    const MatchResult &result);

bool appendMatchDebugCsv(const QString &path,
                         const QString &imagePairName,
                         const MatchResult &result);

bool exportMatchColmap(const QString &path,
                       const QString &image0Name,
                       const QString &image1Name,
                       const MatchResult &result);

} // namespace xjw::feature_match
