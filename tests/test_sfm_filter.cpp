// ============================================================
// test_sfm_filter.cpp — SfM 稀疏点云过滤器单元测试
//
// 测试内容：
//   1. 重投影误差过滤
//   2. 轨迹长度过滤
//   3. 三角化角度过滤
//   4. 统计离群点过滤（KNN）
//   5. 组合过滤
//   6. 进度回调与中止
//   7. 空重建时不崩溃
// ============================================================

#include <gtest/gtest.h>
#include "filtering/SfmPointCloudFilter.h"
#include "reconstruction/SfmReconstruction.h"
#include "common/SfmTypes.h"
#include "Camera.h"

#include <array>
#include <cmath>
#include <vector>
#include <random>

using namespace xjw;

// ─── 测试 fixture ─────────────────────────────────────────────

class SfmFilterTest : public ::testing::Test {
protected:
    SfmReconstruction recon;

    // 辅助方法：添加一个注册图像（带简单相机）
    void addRegisteredImage(ImageId id, double cx, double cy, double cz)
    {
        ImageData data;
        data.id = id;
        data.imagePath = "img_" + std::to_string(id) + ".png";
        data.registered = false;
        // 添加一些特征点
        for (int i = 0; i < 100; ++i) {
            data.keypoints.push_back({static_cast<float>(i * 10),
                                       static_cast<float>(i * 5)});
        }
        data.point3DIds.resize(100, kInvalidPoint3DId);
        recon.addImage(data);

        Camera cam;
        cam.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
        std::array<double,9> R = {1,0,0, 0,1,0, 0,0,1};
        std::array<double,3> C = {cx, cy, cz};
        cam.setPose(R, C);
        recon.registerImage(id, cam);
    }

    // 辅助方法：添加一个 3D 点，带指定属性
    Point3DId addPoint(double x, double y, double z,
                       double error, const std::vector<ImageId> &imageIds)
    {
        ScenePoint3D pt;
        pt.xyz = {x, y, z};
        pt.error = error;
        for (ImageId imgId : imageIds) {
            pt.track.elements.push_back({imgId, 0u});
        }
        return recon.addPoint3D(pt);
    }
};

// ─── 测试用例 ─────────────────────────────────────────────────

// 1. 空重建不崩溃
TEST_F(SfmFilterTest, EmptyReconstruction)
{
    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    auto result = filter.run(opts);

    EXPECT_EQ(result.pointsBefore, 0);
    EXPECT_EQ(result.pointsAfter, 0);
    EXPECT_EQ(result.removedByReprojError, 0);
    EXPECT_EQ(result.removedByTrackLen, 0);
    EXPECT_EQ(result.removedByTriAngle, 0);
    EXPECT_EQ(result.removedByStatistical, 0);
}

// 2. 重投影误差过滤：大误差点被剔除
TEST_F(SfmFilterTest, FilterByReprojError)
{
    addRegisteredImage(0, 0, 0, 0);
    addRegisteredImage(1, 10, 0, 0);

    // 好点：误差 1.0
    addPoint(5, 0, 50, 1.0, {0, 1});
    addPoint(6, 0, 50, 1.5, {0, 1});
    // 坏点：误差 5.0
    addPoint(100, 200, 300, 5.0, {0, 1});
    addPoint(-50, 100, 200, 10.0, {0, 1});

    EXPECT_EQ(recon.numPoints3D(), 4u);

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    opts.filterByReprojError = true;
    opts.maxReprojError = 2.0;
    opts.filterByTrackLen = false;
    opts.filterByTriAngle = false;
    opts.filterByStatistical = false;

    auto result = filter.run(opts);

    EXPECT_EQ(result.removedByReprojError, 2);
    EXPECT_EQ(result.pointsAfter, 2);
}

// 3. 轨迹长度过滤：短轨迹点被剔除
TEST_F(SfmFilterTest, FilterByTrackLength)
{
    addRegisteredImage(0, 0, 0, 0);
    addRegisteredImage(1, 10, 0, 0);
    addRegisteredImage(2, 20, 0, 0);

    // 长轨迹（3 观测）
    addPoint(5, 0, 50, 0.5, {0, 1, 2});
    // 短轨迹（1 观测）
    addPoint(6, 0, 50, 0.5, {0});
    // 中等轨迹（2 观测）
    addPoint(7, 0, 50, 0.5, {0, 1});

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    opts.filterByReprojError = false;
    opts.filterByTrackLen = true;
    opts.minTrackLen = 3;
    opts.filterByTriAngle = false;
    opts.filterByStatistical = false;

    auto result = filter.run(opts);

    EXPECT_EQ(result.removedByTrackLen, 2); // 1观测和2观测的点
    EXPECT_EQ(result.pointsAfter, 1);       // 只剩3观测的点
}

// 4. 三角化角度过滤
TEST_F(SfmFilterTest, FilterByTriAngle)
{
    // 两个相距很远的相机 → 大三角化角
    addRegisteredImage(0,   0, 0, 0);
    addRegisteredImage(1, 100, 0, 0);
    // 两个很近的相机 → 小三角化角
    addRegisteredImage(2, 0.001, 0, 0);

    // 点用远距离相机对观测 → 大角度
    addPoint(50, 0, 10, 0.5, {0, 1});
    // 点用近距离相机对观测 → 小角度
    addPoint(0.0005, 0, 1000, 0.5, {0, 2});

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    opts.filterByReprojError = false;
    opts.filterByTrackLen = false;
    opts.filterByTriAngle = true;
    opts.minTriAngleDeg = 2.0;
    opts.filterByStatistical = false;

    auto result = filter.run(opts);

    // 近距离相机对的点应该因角度过小被移除
    EXPECT_GE(result.removedByTriAngle, 1);
    EXPECT_GE(result.pointsAfter, 1); // 远距离相机对的点应保留
}

// 5. 统计离群过滤：离群点被剔除
TEST_F(SfmFilterTest, FilterByStatistical)
{
    addRegisteredImage(0, 0, 0, 0);
    addRegisteredImage(1, 10, 0, 0);

    // 一簇正常点（紧密分布）
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < 30; ++i) {
        addPoint(50 + dist(rng), dist(rng), 50 + dist(rng),
                 0.5, {0, 1});
    }

    // 一个明显的离群点（远离簇）
    addPoint(500, 500, 500, 0.5, {0, 1});

    EXPECT_EQ(recon.numPoints3D(), 31u);

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    opts.filterByReprojError = false;
    opts.filterByTrackLen = false;
    opts.filterByTriAngle = false;
    opts.filterByStatistical = true;
    opts.statK = 5;
    opts.statStdDevMul = 2.0;

    auto result = filter.run(opts);

    // 远离的离群点应被移除
    EXPECT_GE(result.removedByStatistical, 1);
    // 大部分正常点应保留
    EXPECT_GE(result.pointsAfter, 25);
}

// 6. 组合过滤（全部启用）
TEST_F(SfmFilterTest, CombinedFilter)
{
    addRegisteredImage(0, 0, 0, 0);
    addRegisteredImage(1, 10, 0, 0);
    addRegisteredImage(2, 20, 0, 0);

    // 好点
    for (int i = 0; i < 20; ++i) {
        addPoint(10 + i*0.1, 0, 50, 0.5, {0, 1, 2});
    }
    // 高误差点
    addPoint(100, 100, 100, 10.0, {0, 1, 2});
    // 短轨迹点
    addPoint(10, 0, 50, 0.3, {0});
    // 离群点
    addPoint(1000, 1000, 1000, 0.5, {0, 1, 2});

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    opts.maxReprojError = 2.0;
    opts.minTrackLen = 2;
    opts.minTriAngleDeg = 0.1; // 宽松角度阈值
    opts.statK = 5;
    opts.statStdDevMul = 2.0;

    auto result = filter.run(opts);

    // 至少移除了高误差 + 短轨迹 + 离群
    EXPECT_GE(result.pointsBefore - result.pointsAfter, 3);
    // 好点应该大部分保留
    EXPECT_GE(result.pointsAfter, 15);
}

// 7. 进度回调被调用
TEST_F(SfmFilterTest, ProgressCallbackInvoked)
{
    addRegisteredImage(0, 0, 0, 0);
    addRegisteredImage(1, 10, 0, 0);
    addPoint(5, 0, 50, 0.5, {0, 1});

    int callCount = 0;
    auto cb = [&callCount](const std::string &step, int pct) -> bool {
        ++callCount;
        EXPECT_GE(pct, 0);
        EXPECT_LE(pct, 100);
        return true;
    };

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts; // 全部启用
    filter.run(opts, cb);

    EXPECT_GE(callCount, 1) << "Progress callback should be called at least once";
}

// 8. 进度回调中止
TEST_F(SfmFilterTest, ProgressCallbackAbort)
{
    addRegisteredImage(0, 0, 0, 0);
    addRegisteredImage(1, 10, 0, 0);

    // 添加多个点
    for (int i = 0; i < 10; ++i) {
        addPoint(i, 0, 50, 0.5, {0, 1});
    }

    int callCount = 0;
    auto cb = [&callCount](const std::string &, int) -> bool {
        ++callCount;
        return false; // 立即中止
    };

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    auto result = filter.run(opts, cb);

    // 中止后不应完成所有过滤步骤
    EXPECT_EQ(callCount, 1) << "Should stop after first callback returns false";
}

// 9. 结果 summary 文本非空
TEST_F(SfmFilterTest, ResultSummary)
{
    SfmPointCloudFilterResult result;
    result.pointsBefore = 100;
    result.pointsAfter = 80;
    result.removedByReprojError = 10;
    result.removedByTrackLen = 5;
    result.removedByTriAngle = 3;
    result.removedByStatistical = 2;

    std::string s = result.summary();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("100"), std::string::npos);
    EXPECT_NE(s.find("80"), std::string::npos);
}

// 10. 全部过滤器禁用 → 无点被移除
TEST_F(SfmFilterTest, AllFiltersDisabled)
{
    addRegisteredImage(0, 0, 0, 0);
    addRegisteredImage(1, 10, 0, 0);

    addPoint(5, 0, 50, 100.0, {0});       // 高误差、短轨迹
    addPoint(1000, 1000, 1000, 50.0, {0}); // 离群

    size_t before = recon.numPoints3D();

    SfmPointCloudFilter filter(recon);
    SfmPointCloudFilterOptions opts;
    opts.filterByReprojError = false;
    opts.filterByTrackLen = false;
    opts.filterByTriAngle = false;
    opts.filterByStatistical = false;

    auto result = filter.run(opts);

    EXPECT_EQ(result.pointsBefore, static_cast<int>(before));
    EXPECT_EQ(result.pointsAfter, static_cast<int>(before));
    EXPECT_EQ(result.removedByReprojError, 0);
    EXPECT_EQ(result.removedByTrackLen, 0);
    EXPECT_EQ(result.removedByTriAngle, 0);
    EXPECT_EQ(result.removedByStatistical, 0);
}
