#include <gtest/gtest.h>

#include "data/PointCloud.h"
#include "processing/PointCloudProcessor.h"

#include <vector>

using namespace xjw::pointcloud;

namespace
{

class CudaHookGuard
{
public:
    CudaHookGuard(detail::CudaRuntimeAvailableHook runtimeHook,
                  detail::CudaVoxelDownsampleHook voxelHook)
    {
        detail::setCudaTestHooks(std::move(runtimeHook), std::move(voxelHook));
    }

    ~CudaHookGuard()
    {
        detail::clearCudaTestHooks();
    }
};

PointCloud makeProcessorInput()
{
    PointCloud cloud;
    cloud.addPoint(Point3f{0.00f, 0.00f, 0.00f}, Point3f{0.0f, 0.0f, -1.0f}, ColorRGBA{255, 0, 0, 255});
    cloud.addPoint(Point3f{0.01f, 0.01f, 0.00f}, Point3f{0.0f, 0.0f, -1.0f}, ColorRGBA{250, 10, 10, 255});
    cloud.addPoint(Point3f{1.00f, 1.00f, 1.00f}, Point3f{0.0f, 0.0f, -1.0f}, ColorRGBA{0, 255, 0, 255});
    cloud.addPoint(Point3f{1.02f, 1.01f, 1.00f}, Point3f{0.0f, 0.0f, -1.0f}, ColorRGBA{10, 250, 10, 255});

    PointCloudMetadata metadata;
    metadata.name = "processor_case";
    metadata.isRegistered = true;
    metadata.coordinateFrame = PointCloudCoordinateFrame::World;
    cloud.setMetadata(metadata);
    return cloud;
}

} // namespace

TEST(PointCloudProcessorTest, ComputeStatsReflectsBounds)
{
    const PointCloud cloud = makeProcessorInput();
    const PointCloudStats stats = PointCloudProcessor::computeStats(cloud);

    EXPECT_EQ(stats.count, 4);
    ASSERT_TRUE(stats.bounds.valid);
    EXPECT_NEAR(stats.bounds.minCorner.x, 0.0f, 1e-6f);
    EXPECT_NEAR(stats.bounds.minCorner.y, 0.0f, 1e-6f);
    EXPECT_NEAR(stats.bounds.minCorner.z, 0.0f, 1e-6f);
    EXPECT_NEAR(stats.bounds.maxCorner.x, 1.02f, 1e-6f);
    EXPECT_NEAR(stats.bounds.maxCorner.y, 1.01f, 1e-6f);
    EXPECT_NEAR(stats.bounds.maxCorner.z, 1.0f, 1e-6f);
}

TEST(PointCloudProcessorTest, RunCustomProcessCpuChainProducesProgressAndNormalizedNormals)
{
    const PointCloud input = makeProcessorInput();

    PointCloudProcessParams params;
    params.useCuda = false;
    params.noiseMethod = PointCloudProcessParams::NoiseMethod::Statistical;
    params.statK = 2;
    params.statStdMul = 100.0;
    params.downsampleMethod = PointCloudProcessParams::DownsampleMethod::Uniform;
    params.uniformStep = 2;
    params.unifyNormalDirection = true;

    PointCloud output;
    PointCloudProcessResult result;
    std::vector<int> progressValues;
    ASSERT_TRUE(PointCloudProcessor::runCustomProcess(
        input,
        params,
        &output,
        &result,
        [&progressValues](int value, const QString &) {
            progressValues.push_back(value);
        }));

    EXPECT_EQ(result.pointsBefore, 4);
    EXPECT_EQ(result.pointsAfter, 2);
    EXPECT_FALSE(result.cudaUsed);
    EXPECT_EQ(output.size(), 2U);
    EXPECT_EQ(output.metadata().name, "processor_case");
    ASSERT_TRUE(output.hasNormals());
    for (const Point3f &normal : output.normals())
    {
        EXPECT_GE(normal.z, 0.0f);
    }
    ASSERT_FALSE(progressValues.empty());
    EXPECT_EQ(progressValues.front(), 10);
    EXPECT_EQ(progressValues.back(), 100);
}

TEST(PointCloudProcessorTest, RunCustomProcessFallsBackToCpuWhenCudaBackendFails)
{
    const PointCloud input = makeProcessorInput();
    bool runtimeChecked = false;
    bool gpuAttempted = false;
    CudaHookGuard hookGuard(
        [&runtimeChecked]() {
            runtimeChecked = true;
            return true;
        },
        [&gpuAttempted](const PointCloud &, double, PointCloud *, QString *detail) {
            gpuAttempted = true;
            if (detail)
            {
                *detail = QStringLiteral("mock cuda failure");
            }
            return false;
        });

    PointCloudProcessParams params;
    params.useCuda = true;
    params.noiseMethod = PointCloudProcessParams::NoiseMethod::Statistical;
    params.statK = 2;
    params.statStdMul = 100.0;
    params.downsampleMethod = PointCloudProcessParams::DownsampleMethod::Voxel;
    params.voxelSize = 0.1;

    PointCloud output;
    PointCloudProcessResult result;
    ASSERT_TRUE(PointCloudProcessor::runCustomProcess(input, params, &output, &result));

    EXPECT_TRUE(runtimeChecked);
    EXPECT_TRUE(gpuAttempted);
    EXPECT_FALSE(result.cudaUsed);
    EXPECT_EQ(result.pointsBefore, 4);
    EXPECT_EQ(result.pointsAfter, 2);
    EXPECT_EQ(output.size(), 2U);
    EXPECT_TRUE(result.detail.contains(QStringLiteral("点云处理完成")));
    EXPECT_EQ(output.metadata().coordinateFrame, PointCloudCoordinateFrame::World);
}