#pragma once

#include "MvsTypes.h"

#include <QString>
#include <QStringList>

#include <vector>

namespace xjw::core::project
{

/**
 * @brief 从项目 `.pimatch` 目录加载的 MVS 源像对质量及审计计数。
 */
struct MvsSourcePairQualityLoadResult
{
    std::vector<xjw::mvs::MvsSourcePairQuality> qualities;
    int matchFileCount = 0;
    int catalogPairCount = 0;
    int incompatibleVariantCount = 0;
    int verifiedPairCount = 0;
    int failedPairCount = 0;
    int missingStatisticsPairCount = 0;
};

/**
 * @brief 扫描权威 `.pimatch` 分片并转换为 MVS 源视图规划器输入。
 *
 * targetImagePaths 非空时只保留当前重建影像集合内的像对。目录为空或不存在时
 * 返回空结果，由调用方明确记录回退，而不是读取可能过期的 JSON 报告。
 */
MvsSourcePairQualityLoadResult loadMvsSourcePairQualities(
    const QString &matchDirectory,
    const QStringList &targetImagePaths = {});

/**
 * @brief 把加载结果应用到深度配置，并仅在存在已验证像对时启用严格源像对策略。
 */
void applyMvsSourcePairQualities(
    xjw::mvs::DepthGenConfig *config,
    MvsSourcePairQualityLoadResult result);

} // namespace xjw::core::project
