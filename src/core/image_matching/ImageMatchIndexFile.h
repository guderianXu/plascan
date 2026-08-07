#pragma once

/**
 * @file ImageMatchIndexFile.h
 * @brief 与 `.pimatch` payload 签名绑定的轻量邻接索引。
 *
 * 索引只保存目录浏览、空三预检和 MVS 规划需要的影像身份、算法与几何统计，
 * 不复制关键点和逐条匹配。旧工程缺少索引时会从权威 `.pimatch` 自动重建。
 */

#include "ImageMatchFile.h"

#include <QString>

#include <cstdint>
#include <vector>

namespace xjw::image_matching
{

inline constexpr const char *kImageMatchIndexFileSuffix = ".pidx";

struct ImageMatchNeighborIndex
{
    ImageIdentity peer;
    QString algorithmId;
    std::uint32_t algorithmVersion = 0;
    QByteArray configFingerprint;
    QByteArray modelFingerprint;
    std::int64_t createdTimeMs = 0;
    std::uint32_t rawMatchCount = 0;
    std::uint32_t geometryInlierCount = 0;
    std::uint32_t tiePointMatchCount = 0;
    bool geometryPassed = false;
    GeometryModel geometryModel = GeometryModel::None;
    double geometricCoverage = 0.0;
};

struct ImageMatchFileIndex
{
    ImageMatchFileSignature sourceSignature;
    ImageIdentity owner;
    std::vector<ImageMatchNeighborIndex> neighbors;
};

enum class ImageMatchIndexLoadSource
{
    MemoryCache,
    PersistentIndex,
    RebuiltFromMatchFile
};

class ImageMatchIndexFile
{
public:
    static QString pathForMatchFile(const QString &matchFilePath);

    /**
     * @brief 读取有效索引；缺失、损坏或签名过期时自动从 `.pimatch` 重建。
     */
    static bool load(const QString &matchFilePath,
                     ImageMatchFileIndex *index,
                     ImageMatchIndexLoadSource *source = nullptr,
                     QString *errorMessage = nullptr);

    /// 匹配分片成功提交后立即刷新轻量索引。
    static bool writeForShard(const QString &matchFilePath,
                              const ImageMatchShard &shard,
                              QString *errorMessage = nullptr);

    static bool removeForMatchFile(const QString &matchFilePath);
    static void clearMemoryCache();
};

} // namespace xjw::image_matching
