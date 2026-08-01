#include "TensorRtLightGluePostprocessor.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace xjw::image_matching
{
namespace
{

constexpr int kBlockSize = 256;

__device__ float reduceMaximum(float value, float *shared)
{
    const int thread = threadIdx.x;
    shared[thread] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (thread < offset)
        {
            shared[thread] = fmaxf(shared[thread], shared[thread + offset]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float reduceSum(float value, float *shared)
{
    const int thread = threadIdx.x;
    shared[thread] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (thread < offset)
        {
            shared[thread] += shared[thread + offset];
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float logSigmoid(float value)
{
    if (value >= 0.0f)
    {
        return -log1pf(expf(-value));
    }
    return value - log1pf(expf(value));
}

__device__ bool isBetter(float candidateScore,
                         int candidateIndex,
                         float currentScore,
                         int currentIndex)
{
    return candidateScore > currentScore ||
        (candidateScore == currentScore &&
         (currentIndex < 0 || candidateIndex < currentIndex));
}

__device__ void reduceBest(float &score,
                           int &index,
                           float *sharedScores,
                           int *sharedIndices)
{
    const int thread = threadIdx.x;
    sharedScores[thread] = score;
    sharedIndices[thread] = index;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (thread < offset &&
            isBetter(sharedScores[thread + offset],
                     sharedIndices[thread + offset],
                     sharedScores[thread],
                     sharedIndices[thread]))
        {
            sharedScores[thread] = sharedScores[thread + offset];
            sharedIndices[thread] = sharedIndices[thread + offset];
        }
        __syncthreads();
    }
    score = sharedScores[0];
    index = sharedIndices[0];
}

__global__ void computeRowConstants(const float *similarity,
                                    int stride,
                                    const float *matchability,
                                    int rows,
                                    int columns,
                                    float *constants)
{
    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    __shared__ float shared[kBlockSize];
    float localMaximum = -FLT_MAX;
    for (int column = threadIdx.x; column < columns; column += blockDim.x)
    {
        localMaximum = fmaxf(localMaximum, similarity[row * stride + column]);
    }
    const float maximum = reduceMaximum(localMaximum, shared);

    float localSum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x)
    {
        localSum += expf(similarity[row * stride + column] - maximum);
    }
    const float sum = reduceSum(localSum, shared);
    if (threadIdx.x == 0)
    {
        constants[row] = logSigmoid(matchability[row]) - (maximum + logf(sum));
    }
}

__global__ void computeColumnConstants(const float *similarity,
                                       int stride,
                                       const float *matchability,
                                       int rows,
                                       int columns,
                                       float *constants)
{
    const int column = blockIdx.x;
    if (column >= columns)
    {
        return;
    }

    __shared__ float shared[kBlockSize];
    float localMaximum = -FLT_MAX;
    for (int row = threadIdx.x; row < rows; row += blockDim.x)
    {
        localMaximum = fmaxf(localMaximum, similarity[row * stride + column]);
    }
    const float maximum = reduceMaximum(localMaximum, shared);

    float localSum = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x)
    {
        localSum += expf(similarity[row * stride + column] - maximum);
    }
    const float sum = reduceSum(localSum, shared);
    if (threadIdx.x == 0)
    {
        constants[column] = logSigmoid(matchability[column]) - (maximum + logf(sum));
    }
}

__global__ void findBestColumns(const float *similarity,
                                int stride,
                                const float *rowConstants,
                                const float *columnConstants,
                                int rows,
                                int columns,
                                int *bestColumns,
                                float *bestScores)
{
    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    float bestScore = -FLT_MAX;
    int bestColumn = -1;
    for (int column = threadIdx.x; column < columns; column += blockDim.x)
    {
        const float score = 2.0f * similarity[row * stride + column] +
            rowConstants[row] + columnConstants[column];
        if (isBetter(score, column, bestScore, bestColumn))
        {
            bestScore = score;
            bestColumn = column;
        }
    }

    __shared__ float sharedScores[kBlockSize];
    __shared__ int sharedIndices[kBlockSize];
    reduceBest(bestScore, bestColumn, sharedScores, sharedIndices);
    if (threadIdx.x == 0)
    {
        bestColumns[row] = bestColumn;
        bestScores[row] = bestScore;
    }
}

__global__ void findBestRows(const float *similarity,
                             int stride,
                             const float *rowConstants,
                             const float *columnConstants,
                             int rows,
                             int columns,
                             int *bestRows)
{
    const int column = blockIdx.x;
    if (column >= columns)
    {
        return;
    }

    float bestScore = -FLT_MAX;
    int bestRow = -1;
    for (int row = threadIdx.x; row < rows; row += blockDim.x)
    {
        const float score = 2.0f * similarity[row * stride + column] +
            rowConstants[row] + columnConstants[column];
        if (isBetter(score, row, bestScore, bestRow))
        {
            bestScore = score;
            bestRow = row;
        }
    }

    __shared__ float sharedScores[kBlockSize];
    __shared__ int sharedIndices[kBlockSize];
    reduceBest(bestScore, bestRow, sharedScores, sharedIndices);
    if (threadIdx.x == 0)
    {
        bestRows[column] = bestRow;
    }
}

__global__ void finalizeMatches(const int *bestColumns,
                                const int *bestRows,
                                const float *bestScores,
                                int rows,
                                float scoreThreshold,
                                int *matches0,
                                int *matches1,
                                float *matchingScores0,
                                float *matchingScores1)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    const int column = bestColumns[row];
    if (column < 0 || bestRows[column] != row)
    {
        return;
    }
    const float confidence = expf(bestScores[row]);
    if (!isfinite(confidence) || confidence <= scoreThreshold)
    {
        return;
    }

    matches0[row] = column;
    matches1[column] = row;
    matchingScores0[row] = confidence;
    matchingScores1[column] = confidence;
}

} // namespace

cudaError_t launchTensorRtLightGluePostprocess(
    const float *similarity,
    int similarityStride,
    const float *matchability0,
    const float *matchability1,
    int keypointCount0,
    int keypointCount1,
    float scoreThreshold,
    const TensorRtLightGluePostprocessBuffers &buffers,
    cudaStream_t stream)
{
    if (!similarity || !matchability0 || !matchability1 ||
        !buffers.rowConstants || !buffers.columnConstants ||
        !buffers.bestColumns || !buffers.bestRows || !buffers.bestRowScores ||
        !buffers.matches0 || !buffers.matches1 ||
        !buffers.matchingScores0 || !buffers.matchingScores1 ||
        similarityStride < keypointCount1 ||
        keypointCount0 <= 0 || keypointCount1 <= 0)
    {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaMemsetAsync(
        buffers.matches0, 0xff, static_cast<std::size_t>(keypointCount0) * sizeof(int), stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaMemsetAsync(
        buffers.matches1, 0xff, static_cast<std::size_t>(keypointCount1) * sizeof(int), stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaMemsetAsync(
        buffers.matchingScores0,
        0,
        static_cast<std::size_t>(keypointCount0) * sizeof(float),
        stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaMemsetAsync(
        buffers.matchingScores1,
        0,
        static_cast<std::size_t>(keypointCount1) * sizeof(float),
        stream);
    if (status != cudaSuccess)
    {
        return status;
    }

    computeRowConstants<<<keypointCount0, kBlockSize, 0, stream>>>(
        similarity,
        similarityStride,
        matchability0,
        keypointCount0,
        keypointCount1,
        buffers.rowConstants);
    computeColumnConstants<<<keypointCount1, kBlockSize, 0, stream>>>(
        similarity,
        similarityStride,
        matchability1,
        keypointCount0,
        keypointCount1,
        buffers.columnConstants);
    findBestColumns<<<keypointCount0, kBlockSize, 0, stream>>>(
        similarity,
        similarityStride,
        buffers.rowConstants,
        buffers.columnConstants,
        keypointCount0,
        keypointCount1,
        buffers.bestColumns,
        buffers.bestRowScores);
    findBestRows<<<keypointCount1, kBlockSize, 0, stream>>>(
        similarity,
        similarityStride,
        buffers.rowConstants,
        buffers.columnConstants,
        keypointCount0,
        keypointCount1,
        buffers.bestRows);
    const int blocks = (keypointCount0 + kBlockSize - 1) / kBlockSize;
    finalizeMatches<<<blocks, kBlockSize, 0, stream>>>(
        buffers.bestColumns,
        buffers.bestRows,
        buffers.bestRowScores,
        keypointCount0,
        scoreThreshold,
        buffers.matches0,
        buffers.matches1,
        buffers.matchingScores0,
        buffers.matchingScores1);
    return cudaGetLastError();
}

} // namespace xjw::image_matching
