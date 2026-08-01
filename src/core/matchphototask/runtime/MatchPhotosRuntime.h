#pragma once

/**
 * @file MatchPhotosRuntime.h
 * @brief 创建连接点各阶段共享的轻量运行时工具。
 *
 * 本接口不暴露任何特征文件或成对 `.match` 路径。特征只存在于任务级缓存，
 * 最终匹配统一由 ImageMatchRepository 写入每影像 `.pimatch` 分片。
 */

#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "PairTypes.h"

#include <QJsonObject>
#include <QString>

namespace xjw::matchphotos
{

struct ResolvedImagePair
{
    QString image0Path;
    QString image1Path;
    QString pairName; ///< 只用于日志显示，不参与文件命名。
    QString pairKey;
};

QString matchPhotosMatchDirectory(const MatchPhotosContext &context);

bool shouldCancelMatchPhotos(const MatchPhotosContext &context);
void advanceMatchPhotosProgress(const MatchPhotosContext &context);
void reportMatchPhotosProgress(const MatchPhotosContext &context,
                               const QString &stageId,
                               const QString &message,
                               int current,
                               int maximum);

bool resolveMatchPhotosPair(const MatchPhotosContext &context,
                            const PairCandidate &candidate,
                            ResolvedImagePair *resolved,
                            QString *errorMessage);

QString resolveLightGlueTensorRtEnginePath(const MatchPhotosOptions &options,
                                           QString *engineName);

int resolveFeatureKeypointLimit(const MatchPhotosOptions &options,
                                const MatchPhotosAlgorithmPlan &plan,
                                int imageWidth,
                                int imageHeight);

QJsonObject makeFeatureRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                      const MatchPhotosOptions &options);

QJsonObject makeMatchRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                    const MatchPhotosOptions &options,
                                    const ResolvedImagePair &pair,
                                    int matchCount,
                                    const QJsonObject &extraSettings = QJsonObject());

} // namespace xjw::matchphotos
