#pragma once

// ============================================================
// 文件：CorrespondenceGraph.h
// 功能：特征匹配对应关系图。
//
// 存储多幅图像之间的特征匹配和对应关系，为增量 SfM 流水线提供：
//   - 两两图像之间的匹配查询
//   - 统计每幅图像与多少幅图像有匹配
//   - 查询某幅图像某个特征点在其他图像中的对应点
//
// 参考：COLMAP 的 CorrespondenceGraph / DatabaseCache，简化适配。
// ============================================================

#include "common/SfmTypes.h"

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

namespace xjw {

/**
 * @brief 特征匹配对应关系图。
 *
 * 以无向图形式存储所有图像对之间的已验证匹配。核心数据结构包括：
 *   - 每对图像的匹配列表（FeatureMatch 向量）
 *   - 每幅图像的「对应关系列表」：每个特征索引 → 在其他图像中的对应特征
 *
 * 典型工作流：
 *   1. addImage()          添加图像
 *   2. addMatches()        添加匹配（可多次调用）
 *   3. buildCorrespondences() 构建对应关系索引（一次性）
 *   4. 查询 API            供注册和三角化使用
 */
class CorrespondenceGraph {
public:
    /**
     * @brief 单个特征的对应关系：在哪幅图像的哪个特征。
     */
    struct Correspondence {
        ImageId    imageId    = kInvalidImageId;
        FeatureIdx featureIdx = kInvalidFeatureIdx;
    };

    CorrespondenceGraph() = default;

    // ---- 构建阶段 ----

    /**
     * @brief 注册一幅图像及其特征数量。
     * @param imageId       图像 ID
     * @param numFeatures   该图像检测到的特征数量
     */
    void addImage(ImageId imageId, size_t numFeatures);

    /**
     * @brief 添加一对图像之间的匹配。
     * @param id1      图像 1 ID
     * @param id2      图像 2 ID
     * @param matches  匹配列表
     * @note 匹配方向不影响结果，内部自动以 min/max 排序。
     */
    void addMatches(ImageId id1, ImageId id2,
                    const std::vector<FeatureMatch> &matches);

    /// 人工标记轨迹仅注入当前内存图，不改写任何特征或匹配缓存。
    bool addPriorTrack(const std::string &sourceId,
                       const std::vector<TrackElement> &observations,
                       float confidence);

    /**
     * @brief 构建对应关系索引。
     *
     * 遍历所有匹配，为每幅图像的每个特征建立对应关系列表。
     * 必须在所有 addMatches() 调用完成后、查询前调用一次。
     */
    void buildCorrespondences();

    // ---- 查询接口 ----

    /// 返回已注册图像数量
    size_t numImages() const {return imageFeatureCounts.size();}

    /// 返回图像对匹配总数
    size_t numImagePairs() const
    {
        return pairMatches.size();
    }

    /// 返回稳定排序的全部有效影像对。
    std::vector<ImagePair> imagePairs() const;

    /**
     * @brief 只保留属于指定多视轨迹的原始匹配边。
     *
     * 不会为轨迹中没有直接匹配的影像对合成新边，确保进入 SfM 的每条边
     * 都来自上游已经完成几何验证的 pairwise match。
     * @return 被移除的匹配数量。
     */
    std::size_t retainMatchesInTracks(const std::vector<Track> &tracks);

    /**
     * @brief 获取两幅图像之间的匹配数量。
     * @return 匹配数，若无匹配则返回 0。
     */
    size_t numMatchesBetween(ImageId id1, ImageId id2) const;

    /**
     * @brief 获取两幅图像之间的匹配列表。
     * @return 匹配列表的 const 引用（空引用表示无匹配）。
     */
    const std::vector<FeatureMatch> &matchesBetween(
        ImageId id1, ImageId id2) const;

    /**
     * @brief 获取某幅图像的某个特征在其他图像中的所有对应点。
     * @param imageId     图像 ID
     * @param featureIdx  特征索引
     * @return 对应关系列表
     */
    std::vector<Correspondence> findCorrespondences(
        ImageId imageId, FeatureIdx featureIdx) const;

    /**
     * @brief 获取与某幅图像有匹配的所有图像 ID。
     * @param imageId  查询图像 ID
     * @return 邻居图像 ID 集合
     */
    std::vector<ImageId> connectedImages(ImageId imageId) const;

    /**
     * @brief 获取与某幅图像匹配数最多的前 N 幅图像。
     * @param imageId  查询图像 ID
     * @param topN     返回最多前 N 幅
     * @return 按匹配数降序排列的 (imageId, matchCount) 列表
     */
    std::vector<std::pair<ImageId, size_t>> topConnectedImages(
        ImageId imageId, size_t topN) const;

    std::string priorTrackId(ImageId imageId, FeatureIdx featureIdx) const;

private:
    /// 每幅图像拥有的特征数量
    std::unordered_map<ImageId, size_t> imageFeatureCounts;

    /// 图像对 → 匹配列表
    std::unordered_map<ImagePair, std::vector<FeatureMatch>, ImagePairHash>
        pairMatches;

    /// 对应关系索引：imageId → (featureIdx → [Correspondence...])
    /// 外层 key 是图像 ID，内层 vector 按特征索引对齐（下标 = featureIdx）
    std::unordered_map<ImageId, std::vector<std::vector<Correspondence>>>
        correspondences;

    std::unordered_map<std::uint64_t, std::string> priorTrackByObservation;

    /// 查询用的空匹配列表（避免返回悬空引用）
    static const std::vector<FeatureMatch> EMPTY_MATCHES;
};

} // namespace xjw
