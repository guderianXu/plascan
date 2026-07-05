#pragma once

#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "PairTypes.h"

#include <QJsonObject>
#include <QString>

namespace xjw::feature_extractors
{
struct FeatureData;
}

namespace xjw::feature_match
{
struct MatchResult;
}

namespace xjw
{
namespace matchphotos
{

struct ResolvedImagePair
{
    QString image0Path;
    QString image1Path;
    QString pairName;
    QString pairKey;
};

QString matchPhotosFeatureDirectory(const MatchPhotosContext &context);
QString matchPhotosMatchDirectory(const MatchPhotosContext &context);
QString matchPhotosFeaturePath(const MatchPhotosContext &context,
                               const QString &imagePath,
                               const MatchPhotosAlgorithmPlan &plan);
QString matchPhotosMatchPath(const MatchPhotosContext &context,
                             const QString &image0Path,
                             const QString &image1Path,
                             const MatchPhotosAlgorithmPlan &plan);

bool shouldCancelMatchPhotos(const MatchPhotosContext &context);
void advanceMatchPhotosProgress(const MatchPhotosContext &context);

bool resolveMatchPhotosPair(const MatchPhotosContext &context,
                            const PairCandidate &candidate,
                            ResolvedImagePair *resolved,
                            QString *errorMessage);

QString resolveLightGlueModelPath(const MatchPhotosAlgorithmPlan &plan,
                                  const MatchPhotosOptions &options,
                                  bool *useCuda,
                                  QString *modelName);

QJsonObject makeFeatureRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                      const MatchPhotosOptions &options);

QJsonObject makeMatchRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                    const MatchPhotosOptions &options,
                                    const ResolvedImagePair &pair,
                                    const QString &feature0Path,
                                    const QString &feature1Path,
                                    const QString &matchPath,
                                    const QString &sidecarPath,
                                    int matchCount,
                                    const QJsonObject &extraSettings = QJsonObject());

bool writeMatchPhotosSidecar(const QString &sidecarPath,
                             const ResolvedImagePair &pair,
                             const QString &feature0Path,
                             const QString &feature1Path,
                             const QString &matchPath,
                             const xjw::feature_extractors::FeatureData &feature0,
                             const xjw::feature_extractors::FeatureData &feature1,
                             const xjw::feature_match::MatchResult &matchResult,
                             const MatchPhotosAlgorithmPlan &plan,
                             const MatchPhotosOptions &options,
                             const QJsonObject &extraSettings = QJsonObject());

} // namespace matchphotos
} // namespace xjw
