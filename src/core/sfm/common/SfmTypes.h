#pragma once

// ============================================================
// 文件：SfmTypes.h
// 功能：定义增量式 SfM 流水线使用的基本数据类型。
//
// 包含：
//   - ImageId / Point3DId    类型别名
//   - ImagePair              有序图像对标识
//   - FeatureKeypoint        单个特征点坐标
//   - FeatureMatch           单对匹配关系
//   - TrackElement           轨迹元素（点属于哪幅图的哪个特征）
//   - Track                  多视图轨迹
//   - ScenePoint3D           场景三维点（含轨迹和误差）
//   - ImageData              单幅图像的注册信息
//
// 参考：COLMAP 的 types.h 和 track.h，简化后适配 PlaScan。
// ============================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw 
{

// ---- 基本 ID 类型 ----

/// 图像唯一标识（在项目内递增分配）
using ImageId = uint32_t;

/// 三维点唯一标识
using Point3DId = uint64_t;

/// 特征索引（在单幅图像内部的下标）
using FeatureIdx = uint32_t;

/// 无效 ID 哨兵值
constexpr ImageId   kInvalidImageId   = std::numeric_limits<ImageId>::max();
constexpr Point3DId kInvalidPoint3DId = std::numeric_limits<Point3DId>::max();
constexpr FeatureIdx kInvalidFeatureIdx = std::numeric_limits<FeatureIdx>::max();

// ---- 图像对标识（始终 first < second） ----

/**
 * @brief 有序图像对，用作哈希表键（对应关系图的边）。
 */
struct ImagePair {
    ImageId first  = kInvalidImageId;
    ImageId second = kInvalidImageId;

    /// 构造时自动排序，保证 first <= second
    ImagePair() = default;
    ImagePair(ImageId a, ImageId b)
        : first(std::min(a, b)), second(std::max(a, b)) {}

    bool operator==(const ImagePair &o) const {
        return first == o.first && second == o.second;
    }
};

/// ImagePair 哈希器
struct ImagePairHash {
    std::size_t operator()(const ImagePair &p) const {
        // Cantor pairing
        auto h1 = std::hash<ImageId>{}(p.first);
        auto h2 = std::hash<ImageId>{}(p.second);
        return h1 ^ (h2 << 32) ^ (h2 >> 32);
    }
};

// ---- 特征与匹配 ----

/**
 * @brief 单个特征点坐标（像素）。
 */
struct FeatureKeypoint {
    float x = 0.0f;  ///< u 坐标（列）
    float y = 0.0f;  ///< v 坐标（行）
};

/**
 * @brief 单对特征匹配（索引对）。
 */
struct FeatureMatch {
    FeatureIdx idx1 = kInvalidFeatureIdx;  ///< 图像 1 中的特征索引
    FeatureIdx idx2 = kInvalidFeatureIdx;  ///< 图像 2 中的特征索引
    float score = 1.0f;                    ///< 匹配置信度，范围通常为 [0,1]
};

// ---- 轨迹 ----

/**
 * @brief 轨迹元素：表示某三维点在某幅图像中的一次观测。
 */
struct TrackElement {
    ImageId    imageId    = kInvalidImageId;      ///< 所属图像 ID
    FeatureIdx featureIdx = kInvalidFeatureIdx;    ///< 图像内特征索引
};

/**
 * @brief 多视图轨迹：同一三维点跨多幅图像的观测序列。
 *
 * 当轨迹长度 >= 2 时可以被三角化。
 */
struct Track {
    std::vector<TrackElement> elements;  ///< 观测元素列表

    /// 轨迹长度（观测数）
    std::size_t length() const { return elements.size(); }
};

// ---- 场景三维点 ----

/**
 * @brief 场景中的一个三维点，包含坐标、轨迹、误差指标。
 */
struct ScenePoint3D {
    Point3DId id = kInvalidPoint3DId;                ///< 唯一标识
    std::array<double, 3> xyz{{0, 0, 0}};            ///< 世界坐标
    Track track;                                      ///< 多视图观测轨迹
    double error = 0.0;                               ///< 平均重投影误差（像素）
    std::array<uint8_t, 3> color{{0, 0, 0}};         ///< 可选：RGB 颜色
};

// ---- 图像注册信息 ----

/**
 * @brief 单幅图像在 SfM 重建中的状态。
 *
 * 包含图像路径、特征点、内参索引、是否已注册、以及外参。
 */
struct ImageData {
    ImageId id = kInvalidImageId;          ///< 图像 ID
    std::string imagePath;                 ///< 图像文件路径
    std::string cameraPath;                ///< 相机文件路径（.tsai）

    /// 检测到的特征点坐标列表
    std::vector<FeatureKeypoint> keypoints;

    /// 每个特征点对应的三维点 ID（未关联时为 kInvalidPoint3DId）
    std::vector<Point3DId> point3DIds;

    /// 是否已成功注册到重建中
    bool registered = false;

    /// 已注册三维点的数量（keypoints 中 point3DIds != kInvalid 的计数）
    size_t numVisiblePoints3D() const {
        size_t count = 0;
        for (auto pid : point3DIds) {
            if (pid != kInvalidPoint3DId) ++count;
        }
        return count;
    }
};

} // namespace xjw
