#pragma once

/**
 * @file ImageMatchFile.h
 * @brief `.pimatch` 影像邻接匹配分片的唯一二进制读写入口。
 *
 * 所有核心、CLI 和 GUI 调用方必须通过本类访问格式，禁止自行解析头部。这样格式
 * 版本、字节序、长度上限和校验逻辑只存在一份，不会再次出现扫描器支持一个版本、
 * 查看器却支持另一个版本的情况。
 */

#include "ImageMatchTypes.h"

#include <QString>

namespace xjw::image_matching
{

struct ImageMatchFileSummary
{
    bool valid = false;
    std::uint32_t formatVersion = 0;
    ImageIdentity owner;
    /// 各算法变体本影像观测表的数量之和；同一像点在不同变体中可分别计数。
    std::uint64_t observationCount = 0;
    std::uint32_t neighborVariantCount = 0;
    std::uint64_t payloadBytes = 0;
};

/**
 * @brief 不读取 payload 即可取得的容器签名。
 *
 * payloadSha256 同时承担持久化轻量索引的缓存失效键。索引读取只比较固定大小
 * 容器头，不会为了判断缓存是否有效再次读取全部匹配坐标。
 */
struct ImageMatchFileSignature
{
    bool valid = false;
    std::uint32_t formatVersion = 0;
    std::uint64_t payloadBytes = 0;
    std::uint64_t containerBytes = 0;
    std::int64_t modifiedTimeMs = 0;
    QByteArray payloadSha256;

    bool operator==(const ImageMatchFileSignature &) const = default;
};

class ImageMatchFile
{
public:
    /// 使用 SHA-256(path) 生成不会因同名影像发生碰撞的分片文件名。
    static QString stableImageId(const QString &imagePath);
    static ImageIdentity identityForImage(const QString &imagePath,
                                          int width = 0,
                                          int height = 0);
    static QString filePathForImage(const QString &directory, const QString &imagePath);

    /// 原子写入完整分片；成功前不会覆盖已有可用文件。
    static bool write(const QString &filePath,
                      ImageMatchShard shard,
                      QString *errorMessage = nullptr);

    /// 读取并校验完整分片。任何版本、长度或 SHA-256 错误都会显式失败。
    static bool read(const QString &filePath,
                     ImageMatchShard *shard,
                     QString *errorMessage = nullptr);

    /// 只读取并校验固定大小容器头，不加载或散列 payload。
    static bool readSignature(const QString &filePath,
                              ImageMatchFileSignature *signature,
                              QString *errorMessage = nullptr);

    /// 读取轻量摘要。当前实现仍验证完整 payload，但不会构造所有匹配对象。
    static bool readSummary(const QString &filePath,
                            ImageMatchFileSummary *summary,
                            QString *errorMessage = nullptr);
};

} // namespace xjw::image_matching
