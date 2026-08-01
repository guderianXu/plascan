#pragma once

/**
 * @file ImageMatchRepository.h
 * @brief 以“每影像一个分片”的方式组织、查询和批量提交像对结果。
 */

#include "ImageMatchFile.h"

#include <QString>
#include <QStringList>

#include <vector>

namespace xjw::image_matching
{

struct ImageMatchWriteResult
{
    bool success = false;
    QString errorMessage;
    QStringList writtenFiles;
    int imageCount = 0;
    int pairCount = 0;
};

class ImageMatchRepository
{
public:
    explicit ImageMatchRepository(QString directory);

    const QString &directory() const;
    QString shardPath(const QString &imagePath) const;

    bool loadShard(const QString &imagePath,
                   ImageMatchShard *shard,
                   QString *errorMessage = nullptr) const;

    /**
     * @brief 从任意一侧分片读取一个算法变体，并还原为 image0->image1 方向。
     *
     * 读取成功但没有兼容结果时返回 false，errorMessage 保持为空；文件损坏等真实
     * 错误才写入 errorMessage，调用方可据此区分“缓存缺失”和“缓存不可用”。
     */
    bool loadPair(const QString &image0Path,
                  const QString &image1Path,
                  const QString &algorithmId,
                  std::uint32_t algorithmVersion,
                  const QByteArray &configFingerprint,
                  const QByteArray &modelFingerprint,
                  PairMatchData *pair,
                  QString *errorMessage = nullptr) const;

    /**
     * @brief 将一批对称像对一次性写成所有影像分片。
     *
     * preserveOtherVariants=true 时，会保留同一影像中缓存键不同的算法变体；
     * 同一 peer+algorithm+version+config/model fingerprint 始终被本次结果替换。
     */
    ImageMatchWriteResult writePairs(const std::vector<PairMatchData> &pairs,
                                     bool preserveOtherVariants) const;

    /**
     * @brief 无复制提交任务内持有的像对对象。
     *
     * 空三任务的 PairMatchData 可能包含数百万条对应。该重载只在调用期间借用
     * 指针，避免为了持久化再复制一份完整坐标和残差数组。
     */
    ImageMatchWriteResult writePairReferences(
        const std::vector<const PairMatchData *> &pairs,
        bool preserveOtherVariants) const;

    /// 删除目录内所有 `.pimatch` 文件，不触碰其他项目资产。
    bool clear(QString *errorMessage = nullptr) const;

private:
    QString _directory;
};

} // namespace xjw::image_matching
