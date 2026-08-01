#pragma once

/**
 * @file MatchPhotosFeatureCache.h
 * @brief 一次“创建连接点”任务内复用 SIFT 特征的内存缓存。
 *
 * 特征描述子体积大、格式又与具体算法紧密相关，因此不再写入项目目录。
 * FeatureStage 将每幅影像的特征写入本缓存，通用预选、LightGlue 匹配和几何
 * 验证在同一次任务内只读复用。任务结束后释放缓存即可，不会留下需要迁移的
 * 中间特征文件。
 */

#include "FeatureSet.h"

#include <QHash>
#include <QString>

#include <cstddef>
#include <memory>
#include <shared_mutex>

namespace xjw::matchphotos
{

class MatchPhotosFeatureCache
{
public:
    /// 以规范化绝对路径为键写入或替换一幅影像的特征。
    void insert(const QString &imagePath,
                std::shared_ptr<const image_matching::FeatureSet> features);

    /// 返回共享只读特征；未命中时返回空指针。
    std::shared_ptr<const image_matching::FeatureSet> find(const QString &imagePath) const;

    bool contains(const QString &imagePath) const;
    int imageCount() const;
    std::size_t approximateBytes() const;
    void clear();

private:
    static QString cacheKey(const QString &imagePath);

    mutable std::shared_mutex _mutex;
    QHash<QString, std::shared_ptr<const image_matching::FeatureSet>> _features;
};

} // namespace xjw::matchphotos
