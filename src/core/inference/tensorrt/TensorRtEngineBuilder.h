#pragma once

/**
 * @file TensorRtEngineBuilder.h
 * @brief 将可移植 ONNX 模型编译为当前机器专用 TensorRT engine。
 */

#include "TensorRtTensorInfo.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <cstdint>
#include <functional>
#include <vector>

namespace xjw::inference
{

    enum class TensorRtBuildPrecision
    {
        Fp32,
        Fp16
    };

    struct TensorRtInputShape
    {
        QString name;
        std::vector<std::int64_t> dimensions;
    };

    struct TensorRtEngineBuildProgress
    {
        QString message;
        int current = 0;
        int maximum = 0; ///< 0 表示 TensorRT 未提供可验证百分比，应显示忙碌状态。
    };

    using TensorRtEngineBuildProgressCallback =
        std::function<void(const TensorRtEngineBuildProgress&)>;

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

        /**
         * 静态 ONNX 输入会校验这里给出的形状；动态输入会以相同的 MIN/OPT/MAX
         * 形状构建固定 profile。所有动态 execution tensor 都必须在此提供形状。
         */
        std::vector<TensorRtInputShape> inputShapes;
        QStringList requiredOutputNames;
        QJsonObject fingerprintAttributes;
        TensorRtEngineBuildProgressCallback progressCallback;

        /**
         * @deprecated LoMa-R 旧调用面的兼容字段。新代码应使用 inputShapes。
         */
        int fixedKeypointCount = 0;
    };

    struct TensorRtEngineBuildResult
    {
        QString enginePath;
        QString metadataPath;
        QString cacheFingerprint;
        QString cacheDecision;
        QString environmentSummary;
        QString errorMessage;
        TensorRtBuildPrecision precision = TensorRtBuildPrecision::Fp32;
        std::vector<TensorRtTensorInfo> ioTensors;
        bool reused = false;

        bool isValid() const
        {
            return !enginePath.isEmpty() && errorMessage.isEmpty();
        }
    };

    QString tensorRtBuildPrecisionName(TensorRtBuildPrecision precision);

    /**
     * @brief 确保请求的 ONNX 在当前 TensorRT/GPU 环境中具有可加载的 engine。
     *
     * 多个任务同时请求同一模型时通过 QLockFile 串行构建。engine 和元数据均先写
     * 临时文件再原子提交，进程异常不会留下可被误判为有效的半成品。
     */
    TensorRtEngineBuildResult ensureTensorRtEngine(const TensorRtEngineBuildRequest& request);

} // namespace xjw::inference
