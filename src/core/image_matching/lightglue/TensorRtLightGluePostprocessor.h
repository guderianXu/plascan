#pragma once

#include <cuda_runtime_api.h>

namespace xjw::image_matching
{

/**
 * @brief TensorRT 后处理内核复用的设备缓冲区。
 *
 * 缓冲区由 matcher 按固定桶一次分配，在多次像对推理间复用；函数只借用指针，
 * 不拥有也不释放显存。
 */
struct TensorRtLightGluePostprocessBuffers
{
    float *rowConstants = nullptr;
    float *columnConstants = nullptr;
    int *bestColumns = nullptr;
    int *bestRows = nullptr;
    float *bestRowScores = nullptr;
    int *matches0 = nullptr;
    int *matches1 = nullptr;
    float *matchingScores0 = nullptr;
    float *matchingScores1 = nullptr;
};

/**
 * @brief 在指定 CUDA stream 上执行双向最优、互检和置信度门控。
 *
 * similarityStride 是 engine 固定桶的行跨度，keypointCount0/1 是本次有效范围；
 * padding 不参与归一化和最优值比较。返回值只表示内核启动状态，调用方仍需在
 * stream 同步后检查推理及拷贝错误。
 */
cudaError_t launchTensorRtLightGluePostprocess(
    const float *similarity,
    int similarityStride,
    const float *matchability0,
    const float *matchability1,
    int keypointCount0,
    int keypointCount1,
    float scoreThreshold,
    const TensorRtLightGluePostprocessBuffers &buffers,
    cudaStream_t stream);

} // namespace xjw::image_matching
