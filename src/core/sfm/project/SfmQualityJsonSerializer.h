#pragma once

/**
 * @file SfmQualityJsonSerializer.h
 * @brief 将无 Qt 依赖的 SfM 质量指标转换为工程 JSON。
 *
 * 该层只负责字段命名和类型转换，不重新计算质量，也不改变 MVS 门控结论。
 */

#include "quality/SfmQualityMetrics.h"

#include <QJsonObject>

namespace xjw
{

/// 生成可直接写入工程 report/quality_metadata 的稳定 JSON 对象。
QJsonObject serializeSfmQualityMetrics(const SfmQualityMetrics &metrics);

} // namespace xjw
