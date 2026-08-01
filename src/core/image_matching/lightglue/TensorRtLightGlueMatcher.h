#pragma once

#include "../FeatureSet.h"
#include "../MatchResult.h"

#include <memory>
#include <string>

namespace xjw::image_matching
{

/// TensorRT matcher 的设备、engine 和最终置信度门限。
struct TensorRtLightGlueConfig
{
    std::string enginePath;
    int cudaDevice = 0;
    float scoreThreshold = 0.2f;
};

/**
 * @brief SIFT LightGlue 评分模型的 TensorRT 运行后端。
 *
 * engine 在应用外通过 scripts/models/export_lightglue_tensorrt.py 构建，并与
 * 目标 GPU、TensorRT 版本和固定关键点桶绑定。本类只负责加载、执行和把固定桶
 * 输出还原为真实关键点范围内的互检匹配，不提供静默 CPU 或其它算法回退。
 */
class TensorRtLightGlueMatcher final
{
public:
    explicit TensorRtLightGlueMatcher(const TensorRtLightGlueConfig &config);
    ~TensorRtLightGlueMatcher();

    TensorRtLightGlueMatcher(const TensorRtLightGlueMatcher &) = delete;
    TensorRtLightGlueMatcher &operator=(const TensorRtLightGlueMatcher &) = delete;

    /// 执行一次像对推理；输入描述子必须是归一化的 128 维 SIFT。
    MatchResult match(const FeatureSet &feat0, const FeatureSet &feat1);

    /// 返回报告使用的稳定运行后端名称。
    std::string algorithmName() const;
    /// engine、执行上下文和 CUDA 缓冲区均就绪时返回 true。
    bool isLoaded() const;
    /// 返回 engine 固定关键点容量，用于任务层显存预算和重试降档。
    int bucketKeypoints() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace xjw::image_matching
