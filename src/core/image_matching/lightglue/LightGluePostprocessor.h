#pragma once

#include "../MatchResult.h"

#include <cstdint>

namespace xjw::image_matching
{

/**
 * @brief 把 TensorRT LightGlue 核心张量转换为最终互检匹配。
 *
 * TensorRT 使用带 padding 的固定关键点桶以避开动态 Softmax 编译缺陷。只有
 * keypointCount0 x keypointCount1 左上区域有效；本函数直接计算 LightGlue 的
 * 双 log-softmax 分配，确保 padding 不参与归一化，并保留原始特征索引。
 */
MatchResult postprocessLightGlueCoreOutputs(const float *similarity,
                                            std::int64_t similarityRows,
                                            std::int64_t similarityColumns,
                                            const float *matchability0,
                                            const float *matchability1,
                                            int keypointCount0,
                                            int keypointCount1,
                                            float scoreThreshold,
                                            const char *sourceAlgorithm);

} // namespace xjw::image_matching
