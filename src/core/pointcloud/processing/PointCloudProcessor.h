#pragma once

#include "data/PointCloud.h"

#include <QString>
#include <functional>

namespace xjw::pointcloud
{

/**
 * @brief 点云统计结果。
 */
struct PointCloudStats
{
    int count = 0;
    PointCloudBounds bounds;
};

/**
 * @brief 点云处理参数。
 *
 * 该结构描述噪声过滤、降采样、预览模式、法向量整理与 CUDA 使用策略。
 */
struct PointCloudProcessParams
{
    enum class NoiseMethod
    {
        Statistical,
        Radius
    };

    enum class DownsampleMethod
    {
        Voxel,
        Uniform
    };

    enum class IntensityLevel
    {
        Light,
        Medium,
        Deep
    };

    bool useCuda = true;
    int threads = 0;

    NoiseMethod noiseMethod = NoiseMethod::Statistical;
    IntensityLevel intensityLevel = IntensityLevel::Medium;
    int statK = 20;
    double statStdMul = 1.0;
    double radiusFilter = 0.05;
    int radiusMinNeighbors = 12;

    DownsampleMethod downsampleMethod = DownsampleMethod::Voxel;
    double voxelSize = 0.02;
    int uniformStep = 4;

    int normalK = 32;
    bool smoothNormals = true;
    bool unifyNormalDirection = true;

    bool enableHoleFilling = false;
    double holeFillRadius = 0.03;

    bool previewOnly = false;
    double previewRatio = 0.2;
};

/**
 * @brief 点云处理输出摘要。
 */
struct PointCloudProcessResult
{
    int pointsBefore = 0;
    int pointsAfter = 0;
    qint64 elapsedMs = 0;
    bool cudaUsed = false;
    QString detail;
};

/**
 * @brief 点云处理入口。
 *
 * 该类对外暴露统一处理链，内部根据参数决定使用多线程 CPU 路径还是 CUDA 路径。
 */
class PointCloudProcessor
{
public:
    /** @brief 计算点数与包围盒等基础统计量。 */
    static PointCloudStats computeStats(const PointCloud &pointCloud);

    /**
     * @brief 运行默认的一键预处理流程。
     */
    static bool runOneClickPreprocess(const PointCloud &input,
                                      PointCloud *output,
                                      PointCloudProcessResult *result,
                                      std::function<void(int, const QString &)> progressCallback = {});

    /**
     * @brief 按给定参数执行处理链。
     *
     * 处理链顺序为预览裁剪、噪声过滤、降采样、法向量整理和结果汇总。
     */
    static bool runCustomProcess(const PointCloud &input,
                                 const PointCloudProcessParams &params,
                                 PointCloud *output,
                                 PointCloudProcessResult *result,
                                 std::function<void(int, const QString &)> progressCallback = {});

    /** @brief 查询当前运行环境下 CUDA 路径是否可用。 */
    static bool isCudaRuntimeAvailable();
};

namespace detail
{
/**
 * @brief 测试专用：覆盖 CUDA runtime 可用性判断。
 */
using CudaRuntimeAvailableHook = std::function<bool()>;

/**
 * @brief 测试专用：覆盖 CUDA voxel 降采样实现。
 */
using CudaVoxelDownsampleHook = std::function<bool(const PointCloud &input,
                                                   double voxelSize,
                                                   PointCloud *output,
                                                   QString *detail)>;

/** @brief 多线程统计滤波。 */
PointCloud statisticalFilterMultithread(const PointCloud &input, int k, double stdMul, int threads);

/** @brief 多线程半径滤波。 */
PointCloud radiusFilterMultithread(const PointCloud &input, double radius, int minNeighbors, int threads);

/** @brief 均匀抽样降采样。 */
PointCloud uniformDownsample(const PointCloud &input, int step);

/** @brief 多线程 voxel 降采样。 */
PointCloud voxelDownsampleMultithread(const PointCloud &input, double voxelSize, int threads);

/** @brief CUDA voxel 降采样。 */
bool voxelDownsampleCuda(const PointCloud &input, double voxelSize, PointCloud *output, QString *detail);

/** @brief 查询内部 CUDA runtime 是否可用。 */
bool isCudaRuntimeAvailableInternal();

/** @brief 为测试注入 CUDA hook。 */
void setCudaTestHooks(CudaRuntimeAvailableHook runtimeHook, CudaVoxelDownsampleHook voxelHook);

/** @brief 清理测试注入的 CUDA hook。 */
void clearCudaTestHooks();
} // namespace detail

} // namespace xjw::pointcloud