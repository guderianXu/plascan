#pragma once

/**
 * @file TensorRtEngineBuilder.h
 * @brief 将可移植 ONNX 模型编译为当前机器专用 TensorRT engine。
 *
 * TensorRT plan 默认同时绑定 TensorRT 版本和 GPU Compute Capability，不能作为
 * 跨机器模型资产发布。本构建器以 ONNX 内容和本机环境生成缓存指纹，并只复用
 * 完全相同环境下产生的 engine。
 */

#include <QString>

#include <cstdint>

namespace xjw::image_matching
{

enum class TensorRtBuildPrecision
{
    Fp32,
    Fp16
};

struct TensorRtEngineBuildRequest
{
    QString onnxPath;
    QString cacheDirectory;
    QString engineName;
    TensorRtBuildPrecision precision = TensorRtBuildPrecision::Fp32;
    int cudaDevice = 0;
    std::uint64_t workspaceBytes = 0; ///< 0 表示按目标 GPU 显存自动确定。
    int builderOptimizationLevel = 3;
    int maximumAuxiliaryStreams = 0;
    int fixedKeypointCount = 0; ///< 动态匹配器构建时固定到当前 K 桶；0 表示无需 profile。
};

struct TensorRtEngineBuildResult
{
    QString enginePath;
    QString metadataPath;
    QString cacheFingerprint;
    QString environmentSummary;
    QString errorMessage;
    bool reused = false;

    bool isValid() const { return !enginePath.isEmpty() && errorMessage.isEmpty(); }
};

/**
 * @brief 确保请求的 ONNX 在当前 TensorRT/GPU 环境中具有可加载的 engine。
 *
 * 多个任务同时请求同一模型时通过 QLockFile 串行构建。engine 和元数据均先写
 * 临时文件再原子提交，进程异常不会留下可被误判为有效的半成品。
 */
TensorRtEngineBuildResult ensureTensorRtEngine(
    const TensorRtEngineBuildRequest &request);

} // namespace xjw::image_matching
