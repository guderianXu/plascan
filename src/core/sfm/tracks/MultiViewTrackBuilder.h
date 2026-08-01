#pragma once

/**
 * @file MultiViewTrackBuilder.h
 * @brief 把 pairwise 特征边合并为无同图像冲突的多视轨迹。
 *
 * 算法先把 `(imageId, featureIdx)` 作为观测节点，按匹配分数稳定排序边，再用并查集
 * 合并连通组件。若一条边会使组件含有同一影像的两个特征，该边被拒绝而不是任意
 * 覆盖观测。可选质量稀疏化在完整轨迹层按置信度和影像网格配额选择。
 */

#include "common/SfmTypes.h"

#include <map>
#include <utility>
#include <vector>

namespace xjw
{

struct MultiViewTrackBuildResult
{
    std::vector<Track> tracks; ///< 最终无同图像冲突且通过可选稀疏化的轨迹。
    int totalComponents = 0; ///< 并查集形成的原始连通组件数。
    int acceptedComponents = 0; ///< 产生至少两视轨迹的组件数。
    int rejectedConflictComponents = 0; ///< 曾出现同图像冲突的组件数。
    int rejectedConflictEdges = 0; ///< 为维持一图一观测而拒绝的匹配边数。
    int prunedByQualityThinning = 0; ///< 因影像/网格配额移除的轨迹数。
    int prunedStationaryTracks = 0; ///< 因跨影像像点运动过小移除的轨迹数。
    std::map<int, int> trackLengthHistogram; ///< 轨迹长度到数量。
    std::vector<double> trackConfidenceScores; ///< 与 tracks 顺序对应的聚合置信度。
    double meanTrackConfidence = 0.0; ///< 最终轨迹置信度均值。
};

/// 可重复使用的 pairwise 边收集器；build() 不修改已收集输入。
class MultiViewTrackBuilder
{
public:
    struct BuildOptions
    {
        bool enableQualityThinning = false; ///< 启用按置信度和空间配额裁剪。
        int maxTracksPerImage = 0; ///< 每幅影像最多轨迹数；<=0 不限。
        int maxTracksPerGridCell = 0; ///< 每个影像网格单元最多轨迹数；<=0 不限。
        int gridColumns = 4; ///< 网格列数。
        int gridRows = 4; ///< 网格行数。
        float imageWidth = 0.0f; ///< 网格坐标宽度，必须来自真实影像或明确估计。
        float imageHeight = 0.0f; ///< 网格坐标高度。
        bool excludeStationaryTracks = false; ///< 删除多帧位置近乎不动的传感器伪特征。
        float stationaryTrackMaxPixelMotion = 1.0f; ///< 静止判定最大像素跨度。
    };

    /// 一条 pairwise 边；first/second 分别属于 addMatchPair 的 imageA/imageB。
    struct MatchIndexPair
    {
        FeatureIdx first = kInvalidFeatureIdx;
        FeatureIdx second = kInvalidFeatureIdx;
        float score = 1.0f;

        MatchIndexPair() = default;
        MatchIndexPair(FeatureIdx featureA, FeatureIdx featureB, float matchScore = 1.0f)
            : first(featureA), second(featureB), score(matchScore)
        {
        }
    };

    /// 全局唯一观测键。
    struct ObservationKey
    {
        ImageId imageId = kInvalidImageId;
        FeatureIdx featureIdx = kInvalidFeatureIdx;

        bool operator<(const ObservationKey &other) const
        {
            if (imageId != other.imageId)
            {
                return imageId < other.imageId;
            }
            return featureIdx < other.featureIdx;
        }
    };

    /// 追加一组两两匹配；重复边会在 build 阶段按稳定顺序消解。
    void addMatchPair(ImageId imageA, ImageId imageB, const std::vector<MatchIndexPair> &matches);

    /// 提供关键点坐标，供静止轨迹检测和空间配额使用。
    void setImageKeypoints(ImageId imageId, const std::vector<FeatureKeypoint> &keypoints);

    /// 构建确定性多视轨迹和完整诊断统计。
    MultiViewTrackBuildResult build(const BuildOptions &options = BuildOptions()) const;

private:
    struct Edge
    {
        ObservationKey first; ///< 第一端全局观测键。
        ObservationKey second; ///< 第二端全局观测键。
        float score = 1.0f; ///< 边置信度，优先保留高分边。
        int insertionOrder = 0; ///< 同分时的稳定次序。
    };

    std::vector<Edge> _edges;
    std::map<ImageId, std::vector<FeatureKeypoint>> _keypointsByImage;
};

} // namespace xjw
