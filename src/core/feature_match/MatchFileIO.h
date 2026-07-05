#pragma once

#include "match.h"

#include <QByteArray>
#include <QString>

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::feature_match
{

int writeMatchFile(const QString &path,
                   const MatchResult &matchResult,
                   const std::vector<cv::KeyPoint> &keypoints0,
                   const std::vector<cv::KeyPoint> &keypoints1,
                   QByteArray *errorMessage = nullptr);

bool writeIndexedMatchFile(const QString &path,
                           const QString &image0Name,
                           const QString &image1Name,
                           const MatchResult &result);

bool readIndexedMatchFile(const QString &path,
                          QString &image0Name,
                          QString &image1Name,
                          MatchResult &result);

} // namespace xjw::feature_match
