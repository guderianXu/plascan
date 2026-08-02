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
#include <QStringList>

namespace xjw::matchphotos
{

struct ResolvedImagePair
{
    QString image0Path;
    QString image1Path;
    QString pairName; ///< 只用于日志显示，不参与文件命名。
    QString pairKey;
};

/**
 * @brief TensorRT LightGlue 引擎的运行时解析结果。
 *
 * TensorRT engine 与 GPU 架构、TensorRT 版本以及固定关键点桶绑定，因此不能
 * 像普通权重文件一样只按一个硬编码名称查找。bucketKeypoints 优先从同名
 * `.engine.json` 读取，旧文件没有元数据时再从文件名中的 `bucketNNNN` 推断。
 */
struct ResolvedLightGlueTensorRtEngine
{
    QString path;
    QString name;
    int bucketKeypoints = 0;
    QStringList searchedDirectories;

    bool isValid() const { return !path.isEmpty(); }
};

/**
 * @brief 已校验的 LoMa-R TensorRT 模型包。
 *
 * 清单中的 engine 路径相对清单文件解析。固定输入尺寸、关键点桶和描述子维数
 * 同时进入缓存指纹，保证模型升级后不会复用旧 `.pimatch`。
 */
struct ResolvedLoMaRTensorRtPackage
{
    QString manifestPath;
    QString featureEnginePath;
    QString matcherEnginePath;
    int inputWidth = 0;
    int inputHeight = 0;
    int keypointCount = 0;
    int descriptorDimension = 0;
    QString errorMessage;
    QStringList searchedDirectories;

    bool isValid() const
    {
        return !manifestPath.isEmpty() && !featureEnginePath.isEmpty() &&
            !matcherEnginePath.isEmpty() && inputWidth > 0 && inputHeight > 0 &&
            keypointCount > 0 && descriptorDimension > 0;
    }
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

ResolvedLightGlueTensorRtEngine resolveLightGlueTensorRtEngine(
    const MatchPhotosOptions &options,
    int preferredKeypoints = 0);

// 保留轻量路径接口供既有调用者使用；新代码需要桶容量时应使用上面的完整结果。
QString resolveLightGlueTensorRtEnginePath(const MatchPhotosOptions &options,
                                           QString *engineName);

ResolvedLoMaRTensorRtPackage resolveLoMaRTensorRtPackage(
    const MatchPhotosOptions &options);

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
