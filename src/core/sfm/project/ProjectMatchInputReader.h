#pragma once

/**
 * @file ProjectMatchInputReader.h
 * @brief 从工程相机元数据和逐影像 `.pimatch` 分片构造统一索引化输入。
 *
 * 本层解决路径别名、对称分片去重、算法变体选择和相机顺序问题，不构建多视
 * 轨迹。输出中所有 cameraIndex 都以 selectedImages 的稳定顺序为准，后续
 * BA/SfM 不再解析文件名，也不再依赖成对 `.match` 或 JSON sidecar。
 */

#include "Camera.h"
#include "common/SfmTypes.h"

#include <QJsonObject>
#include <QMap>
#include <QStringList>

#include <array>
#include <vector>

namespace xjw::core::project
{

/// 一条 pairwise 匹配中的两端像点及可选稳定特征索引。
struct ProjectMatchObservationPair
{
    std::array<double, 2> pixelA{{0.0, 0.0}}; ///< 相机 A 中 [u,v] 像素。
    std::array<double, 2> pixelB{{0.0, 0.0}}; ///< 相机 B 中 [u,v] 像素。
    xjw::FeatureIdx featureA = xjw::kInvalidFeatureIdx; ///< A 端关键点索引。
    xjw::FeatureIdx featureB = xjw::kInvalidFeatureIdx; ///< B 端关键点索引。
    double score = 1.0; ///< 匹配置信度，后续作为观测权重来源。
};

/// 一对已解析到当前相机数组的匹配。
struct ProjectMatchPair
{
    int cameraIndexA = -1; ///< ProjectMatchInput::cameras 下标。
    int cameraIndexB = -1; ///< ProjectMatchInput::cameras 下标。
    bool indexed = false; ///< true 表示 featureA/B 可跨 pair 合并为多视轨迹。
    std::vector<ProjectMatchObservationPair> observations; ///< 通过最小匹配数门控的观测。
};

/// 工程读取阶段的完整输出；成员数组共享同一相机索引空间。
struct ProjectMatchInput
{
    std::vector<xjw::Camera> cameras; ///< 当前选中影像的相机，顺序稳定。
    QStringList imagePathByIndex; ///< 与 cameras 一一对应的规范影像路径。
    QMap<QString, QJsonObject> beforeCamMeta; ///< BA/SfM 前相机 JSON，用于事务式回写。
    QMap<QString, int> cameraIndexByPath; ///< 规范路径/受支持别名到相机下标。
    std::vector<ProjectMatchPair> pairs; ///< 当前影像集合内可用匹配。
    int indexedObservationCount = 0; ///< 成功读取的带稳定特征索引观测数。
};

/**
 * @brief 读取相机与匹配并建立稳定索引。
 * @param minMatches 每个影像对必须达到的最小观测数。
 * @return 至少成功建立输入容器时为 true；具体可用相机/轨迹数由调用方继续门控。
 */
bool readProjectMatchInput(const QJsonObject &meta,
                           const QStringList &selectedImages,
                           int minMatches,
                           ProjectMatchInput *input);

/// 将绝对路径、规范路径或受支持的工程影像 token 解析为当前相机下标。
int cameraIndexForImageToken(const QString &imageToken,
                             const QMap<QString, int> &cameraIndexByPath);

} // namespace xjw::core::project
