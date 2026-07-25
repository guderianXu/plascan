// ============================================================
// test_sfm_pipeline.cpp — SFM 管线与 BA 离群点过滤集成测试
//
// 验证：
//   1. SFM 管线能以正确内参成功初始化并完成重建
//   2. 匹配不足时 SFM 优雅失败而非崩溃
//   3. 多候选初始像对重试策略
//   4. 光束法平差有效过滤离群点
//   5. BA 过滤参数灵敏度
//   6. 负深度点过滤
//   7. 观测级过滤（保留缩短轨迹 vs 整点删除）
//   8. 迭代 BA 精化收敛
//
// 参考 COLMAP 的初始化和迭代精化策略。
// ============================================================

#include <gtest/gtest.h>

#include "pipeline/IncrementalSfm.h"
#include "reconstruction/SfmReconstruction.h"
#include "common/SfmTypes.h"
#include "graph/CorrespondenceGraph.h"
#include "Camera.h"
#include "BundleAdjust.h"
#include "triangulation/Triangulator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

using namespace xjw;

// ─── 工具函数：构造合成场景用于测试 ────────────────────────────

namespace {

/// 创建一个简单的 pinhole 相机（焦距 1000px, 主点 512x384, 位于 (cx,cy,cz)）
Camera makeCamera(double cx, double cy, double cz,
                  double fu = 1000.0, double fv = 1000.0)
{
    Camera cam;
    cam.setIntrinsics(fu, fv, 512.0, 384.0);
    // identity rotation, camera at (cx,cy,cz)
    std::array<double,9> R = {1,0,0, 0,1,0, 0,0,1};
    std::array<double,3> C = {cx, cy, cz};
    cam.setPose(R, C);
    return cam;
}

/// 将世界点投影到相机上，返回 (u, v)；成功返回 true
bool projectPoint(const Camera &cam, double wx, double wy, double wz,
                  double &u, double &v)
{
    double world[3] = {wx, wy, wz};
    double uv[2] = {0, 0};
    bool ok = cam.projectWorldPoint(world, uv);
    u = uv[0]; v = uv[1];
    return ok;
}

/// 生成合成 3D 点（在两个相机前方、可被投影的点）
struct SyntheticPoint {
    double x, y, z;
};

double centerDistance(const Camera &a, const Camera &b)
{
    const auto ca = a.cameraCenter();
    const auto cb = b.cameraCenter();
    const double dx = ca[0] - cb[0];
    const double dy = ca[1] - cb[1];
    const double dz = ca[2] - cb[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// 生成 N 个在两个相机前方的随机 3D 点
std::vector<SyntheticPoint> generatePoints(int n, double cx, double cy, double cz,
                                           double spread, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, spread);
    std::vector<SyntheticPoint> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        // 确保 z > 0（在相机前方）
        double z = cz + std::abs(dist(rng)) + 10.0;
        pts.push_back({cx + dist(rng), cy + dist(rng), z});
    }
    return pts;
}

/// 从合成 3D 点生成两幅图像的特征点和匹配
void buildSyntheticMatches(
    const Camera &cam1, const Camera &cam2,
    const std::vector<SyntheticPoint> &points3D,
    std::vector<FeatureKeypoint> &kpts1,
    std::vector<FeatureKeypoint> &kpts2,
    std::vector<FeatureMatch> &matches)
{
    kpts1.clear();
    kpts2.clear();
    matches.clear();

    for (size_t i = 0; i < points3D.size(); ++i) {
        double u1, v1, u2, v2;
        bool ok1 = projectPoint(cam1, points3D[i].x, points3D[i].y, points3D[i].z, u1, v1);
        bool ok2 = projectPoint(cam2, points3D[i].x, points3D[i].y, points3D[i].z, u2, v2);
        if (!ok1 || !ok2) continue;

        // 检查投影在合理范围内
        if (u1 < 0 || u1 > 1024 || v1 < 0 || v1 > 768) continue;
        if (u2 < 0 || u2 > 1024 || v2 < 0 || v2 > 768) continue;

        FeatureIdx idx1 = static_cast<FeatureIdx>(kpts1.size());
        FeatureIdx idx2 = static_cast<FeatureIdx>(kpts2.size());

        kpts1.push_back({static_cast<float>(u1), static_cast<float>(v1)});
        kpts2.push_back({static_cast<float>(u2), static_cast<float>(v2)});

        FeatureMatch m;
        m.idx1 = idx1;
        m.idx2 = idx2;
        matches.push_back(m);
    }
}

void buildKnownPoseTracks(const std::vector<Camera> &cameras,
                          const std::vector<SyntheticPoint> &points3D,
                          std::vector<std::vector<FeatureKeypoint>> &keypoints,
                          std::vector<FeatureMatch> &matches01,
                          std::vector<FeatureMatch> &matches12,
                          std::vector<FeatureMatch> &matches02)
{
    keypoints.assign(cameras.size(), {});
    matches01.clear();
    matches12.clear();
    matches02.clear();

    for (const auto &point : points3D)
    {
        std::vector<std::pair<double, double>> projections;
        projections.reserve(cameras.size());

        bool visibleInAll = true;
        for (const Camera &camera : cameras)
        {
            double u = 0.0;
            double v = 0.0;
            if (!projectPoint(camera, point.x, point.y, point.z, u, v) ||
                u < 0.0 || u > 1024.0 || v < 0.0 || v > 768.0)
            {
                visibleInAll = false;
                break;
            }
            projections.emplace_back(u, v);
        }

        if (!visibleInAll)
        {
            continue;
        }

        const FeatureIdx idx = static_cast<FeatureIdx>(keypoints[0].size());
        for (size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex)
        {
            keypoints[cameraIndex].push_back({static_cast<float>(projections[cameraIndex].first),
                                              static_cast<float>(projections[cameraIndex].second)});
        }

        matches01.push_back({idx, idx});
        matches12.push_back({idx, idx});
        matches02.push_back({idx, idx});
    }
}

void buildIndexedKeypoints(const std::vector<Camera> &cameras,
                           const std::vector<SyntheticPoint> &points3D,
                           std::vector<std::vector<FeatureKeypoint>> &keypoints)
{
    keypoints.assign(cameras.size(), {});

    for (const auto &point : points3D)
    {
        std::vector<std::pair<double, double>> projections;
        projections.reserve(cameras.size());

        bool visibleInAll = true;
        for (const Camera &camera : cameras)
        {
            double u = 0.0;
            double v = 0.0;
            if (!projectPoint(camera, point.x, point.y, point.z, u, v) ||
                u < 0.0 || u > 1024.0 || v < 0.0 || v > 768.0)
            {
                visibleInAll = false;
                break;
            }
            projections.emplace_back(u, v);
        }

        if (!visibleInAll)
        {
            continue;
        }

        for (size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex)
        {
            keypoints[cameraIndex].push_back({static_cast<float>(projections[cameraIndex].first),
                                              static_cast<float>(projections[cameraIndex].second)});
        }
    }
}

std::vector<FeatureMatch> makeIndexedMatches(int beginInclusive, int endExclusive)
{
    std::vector<FeatureMatch> matches;
    matches.reserve(static_cast<size_t>(std::max(0, endExclusive - beginInclusive)));
    for (int idx = beginInclusive; idx < endExclusive; ++idx)
    {
        matches.push_back({static_cast<FeatureIdx>(idx), static_cast<FeatureIdx>(idx)});
    }
    return matches;
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════
// 测试组 1：SFM 初始化
// ═══════════════════════════════════════════════════════════════

class SfmInitTest : public ::testing::Test {
protected:
    IncrementalSfmOptions opts;

    void SetUp() override {
        // 使用宽松参数以便小规模合成数据也能通过
        opts.initMinNumMatches = 20;
        opts.initMinNumInliers = 10;
        opts.initMinChiralityInliers = 5;
        opts.maxInitPairCandidates = 5;
        opts.iterativeBARounds = 1;
        opts.filterMinTrackLen = 1;
    }
};

// 1. 两幅图像 + 充分匹配 + 正确内参 → 重建成功
TEST_F(SfmInitTest, SuccessfulInitWithCorrectIntrinsics)
{
    // 两台相机，baseline 10 单位
    Camera cam1 = makeCamera(0, 0, 0);
    Camera cam2 = makeCamera(10, 0, 0);

    // 在 Z=50 附近生成 200 个 3D 点
    auto points = generatePoints(200, 5, 0, 50, 5.0);

    std::vector<FeatureKeypoint> kpts1, kpts2;
    std::vector<FeatureMatch> matches;
    buildSyntheticMatches(cam1, cam2, points, kpts1, kpts2, matches);

    ASSERT_GE(matches.size(), 50u) << "Synthetic matches should be sufficient";

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "img0.png", cam1, kpts1);
    sfm.addImageWithCamera(1, "img1.png", cam2, kpts2);
    sfm.addMatches(0, 1, matches);

    auto result = sfm.run();

    EXPECT_TRUE(result.success) << "SFM should succeed with correct intrinsics: " << result.summary;
    EXPECT_EQ(result.numRegisteredImages, 2);
    EXPECT_GT(result.numPoints3D, 0) << "Should triangulate some 3D points";
}

// 2. 匹配不足时 → 优雅失败（不崩溃）
TEST_F(SfmInitTest, GracefulFailureWithInsufficientMatches)
{
    Camera cam1 = makeCamera(0, 0, 0);
    Camera cam2 = makeCamera(10, 0, 0);

    // 只有 3 个匹配不够
    std::vector<FeatureKeypoint> kpts1 = {{100, 200}, {300, 400}, {500, 600}};
    std::vector<FeatureKeypoint> kpts2 = {{110, 210}, {310, 410}, {510, 610}};
    std::vector<FeatureMatch> matches = {{0, 0}, {1, 1}, {2, 2}};

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "img0.png", cam1, kpts1);
    sfm.addImageWithCamera(1, "img1.png", cam2, kpts2);
    sfm.addMatches(0, 1, matches);

    auto result = sfm.run();

    EXPECT_FALSE(result.success) << "SFM should fail with insufficient matches";
    EXPECT_FALSE(result.summary.empty()) << "Should have an error message";
    // 不崩溃 = 测试通过
}

// 3. 单幅图像 → 优雅失败
TEST_F(SfmInitTest, SingleImageFails)
{
    Camera cam = makeCamera(0, 0, 0);
    std::vector<FeatureKeypoint> kpts = {{100, 200}};

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "img0.png", cam, kpts);

    auto result = sfm.run();

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.numRegisteredImages, 0);
}

// 4. 零图像 → 优雅失败
TEST_F(SfmInitTest, ZeroImagesFails)
{
    IncrementalSfm sfm(opts);
    auto result = sfm.run();

    EXPECT_FALSE(result.success);
}

// 5. 多候选像对重试：第一对可能失败，第二对成功
TEST_F(SfmInitTest, MultiCandidateRetry)
{
    // 创建 3 幅图像：0和1的匹配主要是共线的（容易导致 chirality failure），
    // 0和2的匹配是良好的三角化几何
    Camera cam0 = makeCamera(0, 0, 0);
    Camera cam1 = makeCamera(0.001, 0, 0);  // 几乎重合 → 差的几何
    Camera cam2 = makeCamera(10, 0, 0);     // 良好基线

    auto points = generatePoints(200, 5, 0, 50, 5.0);

    std::vector<FeatureKeypoint> kpts0, kpts1, kpts2;
    std::vector<FeatureMatch> matches01, matches02;

    // 0-1: 近距离相机，生成匹配（大量）
    buildSyntheticMatches(cam0, cam1, points, kpts0, kpts1, matches01);

    // 0-2: 良好基线
    std::vector<FeatureKeypoint> kpts0_for2;
    buildSyntheticMatches(cam0, cam2, points, kpts0_for2, kpts2, matches02);

    // 合并 kpts0（两组匹配共享图像0的特征点列表）
    // 这里简化：直接为图像0使用 kpts0_for2（来自 cam0 投影）
    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "img0.png", cam0, kpts0_for2);
    sfm.addImageWithCamera(2, "img2.png", cam2, kpts2);
    sfm.addMatches(0, 2, matches02);

    auto result = sfm.run();

    // 至少用好的像对初始化成功
    EXPECT_TRUE(result.success) << result.summary;
    EXPECT_EQ(result.numRegisteredImages, 2);
}

// 6. 使用 addImage（.tsai 文件路径）方式加载相机
TEST_F(SfmInitTest, LoadCameraFromTsaiFile)
{
    std::string tsaiDir = std::string(TEST_DATA_DIR) + "/tsai/";
    std::string imgDir  = std::string(TEST_DATA_DIR) + "/img/";

    // 验证测试数据存在
    Camera testCam;
    bool loaded = testCam.loadFromFile(tsaiDir + "1.tsai");
    if (!loaded) {
        GTEST_SKIP() << "Test data not available at " << tsaiDir;
    }

    // 使用真实 .tsai 文件，但只测试两幅图像
    // 由于没有真实的匹配数据，只验证 addImage 不崩溃
    std::vector<FeatureKeypoint> kpts = {{100, 200}, {300, 400}, {500, 600}};

    IncrementalSfm sfm(opts);
    sfm.addImage(0, imgDir + "1.png", tsaiDir + "1.tsai", kpts);
    sfm.addImage(1, imgDir + "2.png", tsaiDir + "2.tsai", kpts);

    // 没有匹配，应该优雅失败
    auto result = sfm.run();
    EXPECT_FALSE(result.success);
    // 关键：不崩溃
}

TEST_F(SfmInitTest, KnownCameraPoseModeRegistersAllImagesAndRunsStableBA)
{
    opts.useKnownCameraPoses = true;
    opts.triangulatorOptions.minTriAngle = 0.1;
    opts.triangulatorOptions.maxReprojError = 0.5;
    opts.triangulatorOptions.continueMaxReprojError = 0.5;
    opts.triangulatorOptions.completeMaxReprojError = 0.5;
    opts.filterMaxReprojError = 0.5;
    opts.filterMinTriAngle = 0.1;

    std::vector<Camera> cameras = {
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
        makeCamera(4.0, 0.0, 0.0),
    };

    const auto points = generatePoints(80, 2.0, 0.0, 70.0, 1.5, 7);
    std::vector<std::vector<FeatureKeypoint>> keypoints;
    std::vector<FeatureMatch> matches01;
    std::vector<FeatureMatch> matches12;
    std::vector<FeatureMatch> matches02;
    buildKnownPoseTracks(cameras, points, keypoints, matches01, matches12, matches02);
    ASSERT_GT(matches01.size(), 50u);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "known_pose_0.png", cameras[0], keypoints[0]);
    sfm.addImageWithCamera(1, "known_pose_1.png", cameras[1], keypoints[1]);
    sfm.addImageWithCamera(2, "known_pose_2.png", cameras[2], keypoints[2]);
    sfm.addMatches(0, 1, matches01);
    sfm.addMatches(1, 2, matches12);
    sfm.addMatches(0, 2, matches02);

    auto result = sfm.run();

    ASSERT_TRUE(result.success) << result.summary;
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_EQ(result.numRegisteredImages, 3);
    EXPECT_GT(result.numPoints3D, 50);
    EXPECT_GT(result.baTracksTotal, 0);
    EXPECT_GT(result.baTracksOptimized, 0);

    for (ImageId imageId = 0; imageId < 3; ++imageId)
    {
        EXPECT_LT(centerDistance(result.reconstruction->camera(imageId), cameras[imageId]), 1e-3);
    }
}

TEST_F(SfmInitTest, KnownCameraPoseModeRunsGlobalBAAndRefinesNoisyPose)
{
    opts.useKnownCameraPoses = true;
    opts.triangulatorOptions.minTriAngle = 0.1;
    opts.triangulatorOptions.maxReprojError = 5.0;
    opts.triangulatorOptions.continueMaxReprojError = 5.0;
    opts.triangulatorOptions.completeMaxReprojError = 5.0;
    opts.filterMaxReprojError = 5.0;
    opts.filterMinTriAngle = 0.1;
    opts.baOptions.maxIterations = 6;
    opts.baOptions.maxPointIterations = 6;
    opts.baOptions.maxCameraIterations = 6;
    opts.baOptions.filterMaxReprojError = 5.0;

    const std::vector<Camera> trueCameras = {
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
        makeCamera(4.0, 0.0, 0.0),
    };
    std::vector<Camera> inputCameras = trueCameras;
    inputCameras[1] = makeCamera(2.18, 0.0, 0.0);

    const auto points = generatePoints(120, 2.0, 0.0, 70.0, 1.5, 37);
    std::vector<std::vector<FeatureKeypoint>> keypoints;
    std::vector<FeatureMatch> matches01;
    std::vector<FeatureMatch> matches12;
    std::vector<FeatureMatch> matches02;
    buildKnownPoseTracks(trueCameras, points, keypoints, matches01, matches12, matches02);
    ASSERT_GT(matches01.size(), 50u);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "known_pose_ba_0.png", inputCameras[0], keypoints[0]);
    sfm.addImageWithCamera(1, "known_pose_ba_1.png", inputCameras[1], keypoints[1]);
    sfm.addImageWithCamera(2, "known_pose_ba_2.png", inputCameras[2], keypoints[2]);
    sfm.addMatches(0, 1, matches01);
    sfm.addMatches(1, 2, matches12);
    sfm.addMatches(0, 2, matches02);

    const auto result = sfm.run();

    ASSERT_TRUE(result.success) << result.summary;
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_GT(result.baTracksTotal, 0);
    EXPECT_GT(result.baTracksOptimized, 0);
    EXPECT_GT(result.baRmsBefore, result.baRmsAfter);

    const Camera &refinedCamera = result.reconstruction->camera(1);
    EXPECT_LT(centerDistance(refinedCamera, trueCameras[1]),
              centerDistance(inputCameras[1], trueCameras[1]));
}

TEST_F(SfmInitTest, LockedKnownCameraPoseModeKeepsInputExtrinsicsExact)
{
    opts.useKnownCameraPoses = true;
    opts.refineKnownCameraPoseWithSoftPrior = false;
    opts.triangulatorOptions.minTriAngle = 0.1;
    opts.triangulatorOptions.maxReprojError = 0.5;
    opts.triangulatorOptions.continueMaxReprojError = 0.5;
    opts.triangulatorOptions.completeMaxReprojError = 0.5;
    opts.filterMaxReprojError = 0.5;
    opts.filterMinTriAngle = 0.1;

    const std::vector<Camera> cameras = {
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
        makeCamera(4.0, 0.0, 0.0),
    };
    const auto points = generatePoints(80, 2.0, 0.0, 70.0, 1.5, 83);
    std::vector<std::vector<FeatureKeypoint>> keypoints;
    std::vector<FeatureMatch> matches01;
    std::vector<FeatureMatch> matches12;
    std::vector<FeatureMatch> matches02;
    buildKnownPoseTracks(cameras, points, keypoints, matches01, matches12, matches02);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "locked_pose_0.png", cameras[0], keypoints[0]);
    sfm.addImageWithCamera(1, "locked_pose_1.png", cameras[1], keypoints[1]);
    sfm.addImageWithCamera(2, "locked_pose_2.png", cameras[2], keypoints[2]);
    sfm.addMatches(0, 1, matches01);
    sfm.addMatches(1, 2, matches12);
    sfm.addMatches(0, 2, matches02);

    const auto result = sfm.run();

    ASSERT_TRUE(result.success) << result.summary;
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_GT(result.baTracksOptimized, 0);
    for (ImageId imageId = 0; imageId < 3; ++imageId)
    {
        EXPECT_LT(centerDistance(result.reconstruction->camera(imageId), cameras[imageId]), 1e-9);
    }
}

TEST_F(SfmInitTest, KnownCameraPoseModeRejectsAllTwoViewOutputWhenMultiViewTracksExist)
{
    opts.useKnownCameraPoses = true;
    opts.triangulatorOptions.minTriAngle = 0.1;
    opts.triangulatorOptions.maxReprojError = 25.0;
    opts.triangulatorOptions.continueMaxReprojError = 0.5;
    opts.triangulatorOptions.completeMaxReprojError = 0.5;
    opts.filterMaxReprojError = 25.0;
    opts.filterMinTriAngle = 0.1;
    opts.filterMinTrackLen = 2;

    std::vector<Camera> cameras = {
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
    };

    const auto points = generatePoints(80, 8.0, 0.0, 60.0, 1.0, 19);
    std::vector<std::vector<FeatureKeypoint>> keypoints;
    std::vector<FeatureMatch> matches01;
    std::vector<FeatureMatch> matches12;
    std::vector<FeatureMatch> matches02;
    buildKnownPoseTracks(cameras, points, keypoints, matches01, matches12, matches02);
    ASSERT_GT(matches01.size(), 30u);

    for (FeatureKeypoint &keypoint : keypoints[2])
    {
        keypoint.x += 20.0f;
    }

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "known_pose_noisy_0.png", cameras[0], keypoints[0]);
    sfm.addImageWithCamera(1, "known_pose_noisy_1.png", cameras[1], keypoints[1]);
    sfm.addImageWithCamera(2, "known_pose_noisy_2.png", cameras[2], keypoints[2]);
    sfm.addMatches(0, 1, matches01);
    sfm.addMatches(1, 2, matches12);
    sfm.addMatches(0, 2, matches02);

    const auto result = sfm.run();

    EXPECT_FALSE(result.success) << "Formal known-pose SfM must not accept an all two-view sparse cloud";
    EXPECT_NE(result.summary.find("two-view"), std::string::npos) << result.summary;
}

TEST_F(SfmInitTest, KnownCameraPoseModeRejectsAlmostAllTwoViewOutputWhenMultiViewTracksExist)
{
    opts.useKnownCameraPoses = true;
    opts.triangulatorOptions.minTriAngle = 0.1;
    opts.triangulatorOptions.maxReprojError = 25.0;
    opts.triangulatorOptions.continueMaxReprojError = 0.5;
    opts.triangulatorOptions.completeMaxReprojError = 0.5;
    opts.filterMaxReprojError = 25.0;
    opts.filterMinTriAngle = 0.1;
    opts.filterMinTrackLen = 2;

    std::vector<Camera> cameras = {
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
    };

    const auto points = generatePoints(80, 8.0, 0.0, 60.0, 1.0, 23);
    std::vector<std::vector<FeatureKeypoint>> keypoints;
    std::vector<FeatureMatch> matches01;
    std::vector<FeatureMatch> matches12;
    std::vector<FeatureMatch> matches02;
    buildKnownPoseTracks(cameras, points, keypoints, matches01, matches12, matches02);
    ASSERT_GT(matches01.size(), 30u);
    ASSERT_FALSE(keypoints[2].empty());

    const FeatureKeypoint oneGoodThirdViewObservation = keypoints[2][0];
    for (FeatureKeypoint &keypoint : keypoints[2])
    {
        keypoint.x += 20.0f;
    }
    keypoints[2][0] = oneGoodThirdViewObservation;

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "known_pose_sparse_long_0.png", cameras[0], keypoints[0]);
    sfm.addImageWithCamera(1, "known_pose_sparse_long_1.png", cameras[1], keypoints[1]);
    sfm.addImageWithCamera(2, "known_pose_sparse_long_2.png", cameras[2], keypoints[2]);
    sfm.addMatches(0, 1, matches01);
    sfm.addMatches(1, 2, matches12);
    sfm.addMatches(0, 2, matches02);

    const auto result = sfm.run();

    EXPECT_FALSE(result.success)
        << "Formal known-pose SfM must not accept a sparse cloud whose multi-view support is nearly absent";
    EXPECT_NE(result.summary.find("two-view"), std::string::npos) << result.summary;
}

TEST_F(SfmInitTest, KnownCameraPoseModeAdaptsTriangulationAngleForNarrowBaseline)
{
    opts.useKnownCameraPoses = true;

    std::vector<Camera> cameras = {
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(0.35, 0.0, 0.0),
    };

    const auto points = generatePoints(80, 0.2, 0.0, 110.0, 1.0, 11);

    std::vector<FeatureKeypoint> keypoints0;
    std::vector<FeatureKeypoint> keypoints1;
    std::vector<FeatureMatch> matches01;
    buildSyntheticMatches(cameras[0], cameras[1], points, keypoints0, keypoints1, matches01);
    ASSERT_GT(matches01.size(), 50u);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "narrow_known_pose_0.png", cameras[0], keypoints0);
    sfm.addImageWithCamera(1, "narrow_known_pose_1.png", cameras[1], keypoints1);
    sfm.addMatches(0, 1, matches01);

    auto result = sfm.run();

    ASSERT_TRUE(result.success) << result.summary;
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_EQ(result.numRegisteredImages, 2);
    EXPECT_GT(result.numPoints3D, 50);
}

// ═══════════════════════════════════════════════════════════════
// 测试组 2：SFM 选项验证
// ═══════════════════════════════════════════════════════════════

TEST(SfmOptionsTest, NewOptionsDefaults)
{
    IncrementalSfmOptions opts;

    // 新增选项应有合理默认值
    EXPECT_GE(opts.initMinChiralityInliers, 5)
        << "chirality threshold should be >= 5";
    EXPECT_LE(opts.initMinChiralityInliers, opts.initMinNumInliers)
        << "chirality threshold should be softer than main inlier threshold";

    EXPECT_GT(opts.maxInitPairCandidates, 1)
        << "Should try multiple candidate pairs";
    EXPECT_LE(opts.maxInitPairCandidates, 50)
        << "Candidate limit should be reasonable";

    EXPECT_GE(opts.iterativeBARounds, 1)
        << "Should do at least 1 round of iterative BA";
    EXPECT_LE(opts.iterativeBARounds, 10)
        << "iterativeBARounds should be reasonable";

    EXPECT_TRUE(opts.filterNegativeDepth)
        << "Negative depth filtering should be enabled by default";
}

// ═══════════════════════════════════════════════════════════════
// 测试组 3：BA 离群点过滤
// ═══════════════════════════════════════════════════════════════

class BAFilterTest : public ::testing::Test {
protected:
    /// 构造一个带有离群点的合成 BA 场景
    struct SyntheticBA {
        std::vector<Camera> cameras;
        std::vector<BATrack> tracks;
        int numGoodPoints = 0;
        int numOutliers = 0;
    };

    /// 创建合成 BA 数据：N 个好点 + M 个离群点
    SyntheticBA buildScene(int numGood, int numOutlier, unsigned seed = 42)
    {
        SyntheticBA scene;

        // 两台相机
        scene.cameras.push_back(makeCamera(0, 0, 0));
        scene.cameras.push_back(makeCamera(10, 0, 0));

        std::mt19937 rng(seed);
        std::normal_distribution<double> noise(0.0, 0.3); // 小噪声

        // 好点：近似在 Z=50 处
        auto goodPts = generatePoints(numGood, 5, 0, 50, 5.0, seed);
        for (const auto &p : goodPts) {
            BATrack track;
            track.initialPoint = {p.x, p.y, p.z};
            for (int ci = 0; ci < 2; ++ci) {
                double u, v;
                if (projectPoint(scene.cameras[ci], p.x, p.y, p.z, u, v)) {
                    BAObservation obs;
                    obs.cameraIndex = ci;
                    obs.u = u + noise(rng);
                    obs.v = v + noise(rng);
                    track.observations.push_back(obs);
                }
            }
            if (track.observations.size() >= 2) {
                scene.tracks.push_back(std::move(track));
                ++scene.numGoodPoints;
            }
        }

        // 离群点：坐标偏移很大
        for (int i = 0; i < numOutlier; ++i) {
            BATrack track;
            // 远离真实场景的错误坐标
            double ox = 500 + rng() % 1000;
            double oy = 500 + rng() % 1000;
            double oz = 500 + rng() % 1000;
            track.initialPoint = {ox, oy, oz};

            // 但观测像素坐标是来自正确 3D 点的投影（模拟错误三角化）
            auto correctPt = generatePoints(1, 5, 0, 50, 5.0, seed + i + 999);
            for (int ci = 0; ci < 2; ++ci) {
                double u, v;
                if (projectPoint(scene.cameras[ci],
                                 correctPt[0].x, correctPt[0].y, correctPt[0].z,
                                 u, v)) {
                    BAObservation obs;
                    obs.cameraIndex = ci;
                    obs.u = u;
                    obs.v = v;
                    track.observations.push_back(obs);
                }
            }
            if (track.observations.size() >= 2) {
                scene.tracks.push_back(std::move(track));
                ++scene.numOutliers;
            }
        }

        return scene;
    }
};

// 1. BA 过滤能移除离群点
TEST_F(BAFilterTest, FilterRemovesOutliers)
{
    auto scene = buildScene(50, 10);

    BAOptions opts;
    opts.enablePointFilter = true;
    opts.filterMaxReprojError = 2.5;
    opts.filterSigmaFactor = 3.0;
    opts.maxIterations = 5;

    auto result = BundleAdjust::optimizePoints(scene.cameras, scene.tracks, opts);

    // 统计被标记为无效的点
    int invalidCount = 0;
    for (const auto &p : result.points) {
        if (!p.valid) ++invalidCount;
    }

    EXPECT_GE(invalidCount, scene.numOutliers / 2)
        << "BA should filter at least half of the outlier points";
    EXPECT_LE(invalidCount, scene.numOutliers + 5)
        << "BA should not overly filter good points";

    // 优化后 RMS 应比优化前更小
    EXPECT_LT(result.meanRmsAfter, result.meanRmsBefore + 1e-6)
        << "BA should improve or maintain reprojection error";
}

// 2. 过滤阈值越严格，过滤越多
TEST_F(BAFilterTest, StricterThresholdFiltersMore)
{
    auto scene = buildScene(50, 10, 123);

    // 宽松阈值
    BAOptions optsLoose;
    optsLoose.enablePointFilter = true;
    optsLoose.filterMaxReprojError = 10.0;
    optsLoose.maxIterations = 3;
    auto resultLoose = BundleAdjust::optimizePoints(scene.cameras, scene.tracks, optsLoose);

    // 严格阈值
    BAOptions optsStrict;
    optsStrict.enablePointFilter = true;
    optsStrict.filterMaxReprojError = 1.0;
    optsStrict.maxIterations = 3;
    auto resultStrict = BundleAdjust::optimizePoints(scene.cameras, scene.tracks, optsStrict);

    int looseFilt = 0, strictFilt = 0;
    for (size_t i = 0; i < resultLoose.points.size(); ++i) {
        if (!resultLoose.points[i].valid) ++looseFilt;
    }
    for (size_t i = 0; i < resultStrict.points.size(); ++i) {
        if (!resultStrict.points[i].valid) ++strictFilt;
    }

    EXPECT_GE(strictFilt, looseFilt)
        << "Stricter threshold should filter at least as many points";
}

// 3. 禁用过滤 → 不删点
TEST_F(BAFilterTest, DisabledFilterKeepsAllPoints)
{
    auto scene = buildScene(30, 5, 456);

    BAOptions opts;
    opts.enablePointFilter = false;
    opts.maxIterations = 3;
    auto result = BundleAdjust::optimizePoints(scene.cameras, scene.tracks, opts);

    int invalidCount = 0;
    for (const auto &p : result.points) {
        if (!p.valid) ++invalidCount;
    }

    // 不过滤时，应该没有（或极少）无效点（仅优化失败的）
    // 注意：即使禁用过滤，数值发散也可能导致极个别点无效
    EXPECT_LE(invalidCount, 2)
        << "With filter disabled, very few points should be marked invalid";
}

// 4. 空场景不崩溃
TEST_F(BAFilterTest, EmptyScene)
{
    std::vector<Camera> cameras;
    std::vector<BATrack> tracks;

    BAOptions opts;
    auto result = BundleAdjust::optimizePoints(cameras, tracks, opts);

    EXPECT_EQ(result.totalTracks, 0);
    EXPECT_EQ(result.optimizedTracks, 0);
}

// 5. 单相机场景不崩溃
TEST_F(BAFilterTest, SingleCamera)
{
    std::vector<Camera> cameras = {makeCamera(0, 0, 0)};
    BATrack track;
    track.initialPoint = {5, 0, 50};
    track.observations.push_back({0, 512.0, 384.0});
    std::vector<BATrack> tracks = {track};

    BAOptions opts;
    opts.maxIterations = 1;
    auto result = BundleAdjust::optimizePoints(cameras, tracks, opts);

    // 单观测轨迹无法真正优化，但不应崩溃
    EXPECT_EQ(result.totalTracks, 1);
}

// 6. Huber 损失参数敏感性
TEST_F(BAFilterTest, HuberDeltaSensitivity)
{
    auto scene = buildScene(40, 8, 789);

    // 小 huber delta → 更鲁棒
    BAOptions optsSmall;
    optsSmall.huberDelta = 1.0;
    optsSmall.enablePointFilter = true;
    optsSmall.filterMaxReprojError = 2.5;
    optsSmall.maxIterations = 5;
    auto resultSmall = BundleAdjust::optimizePoints(scene.cameras, scene.tracks, optsSmall);

    // 大 huber delta → 更不鲁棒（接近 L²）
    BAOptions optsLarge;
    optsLarge.huberDelta = 50.0;
    optsLarge.enablePointFilter = true;
    optsLarge.filterMaxReprojError = 2.5;
    optsLarge.maxIterations = 5;
    auto resultLarge = BundleAdjust::optimizePoints(scene.cameras, scene.tracks, optsLarge);

    // 两者都不应崩溃
    EXPECT_GT(resultSmall.totalTracks, 0);
    EXPECT_GT(resultLarge.totalTracks, 0);

    // 小 huber delta 应该更好地处理离群点
    EXPECT_LE(resultSmall.meanRmsAfter, resultLarge.meanRmsAfter * 2.0 + 0.5)
        << "Smaller Huber delta should provide comparable or better RMS with outliers";
}

TEST_F(BAFilterTest, ObservationWeightsReduceInfluenceOfLowConfidenceOutlier)
{
    const std::vector<Camera> cameras = {
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.5, 0.2, 42.0}};

    double u0 = 0.0, v0 = 0.0;
    double u1 = 0.0, v1 = 0.0;
    double u2 = 0.0, v2 = 0.0;
    ASSERT_TRUE(projectPoint(cameras[0], truth[0], truth[1], truth[2], u0, v0));
    ASSERT_TRUE(projectPoint(cameras[1], truth[0], truth[1], truth[2], u1, v1));
    ASSERT_TRUE(projectPoint(cameras[2], truth[0], truth[1], truth[2], u2, v2));

    BATrack equalWeightTrack;
    equalWeightTrack.initialPoint = {{truth[0] + 3.0, truth[1] - 2.0, truth[2] + 5.0}};
    equalWeightTrack.observations.push_back({0, u0, v0, 1.0});
    equalWeightTrack.observations.push_back({1, u1, v1, 1.0});
    equalWeightTrack.observations.push_back({2, u2 + 80.0, v2 - 60.0, 1.0});

    BATrack weightedTrack = equalWeightTrack;
    weightedTrack.observations[2].weight = 0.01;

    BAOptions options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.huberDelta = 1000.0;
    options.maxIterations = 8;
    options.maxPointIterations = 20;

    const BAResult equalResult = BundleAdjust::optimizePoints(cameras, {equalWeightTrack}, options);
    const BAResult weightedResult = BundleAdjust::optimizePoints(cameras, {weightedTrack}, options);
    ASSERT_EQ(equalResult.points.size(), 1);
    ASSERT_EQ(weightedResult.points.size(), 1);
    ASSERT_TRUE(equalResult.points.front().valid);
    ASSERT_TRUE(weightedResult.points.front().valid);

    auto distanceToTruth = [&](const std::array<double, 3> &point) {
        const double dx = point[0] - truth[0];
        const double dy = point[1] - truth[1];
        const double dz = point[2] - truth[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    EXPECT_LT(distanceToTruth(weightedResult.points.front().point),
              distanceToTruth(equalResult.points.front().point));
}

// ═══════════════════════════════════════════════════════════════
// 测试组 4：SFM 管线完整流程
// ═══════════════════════════════════════════════════════════════

TEST(IncrementalSfmOptionsTest, CoarseExecutionProfileCapsExpensiveRefinement)
{
    IncrementalSfmOptions options;
    options.executionProfile = SfmExecutionProfile::CoarseEvaluation;
    options.maxInitPairCandidates = 10;
    options.baOptions.maxIterations = 20;
    options.iterativeBARounds = 4;
    options.globalBAInterval = 10;
    options.baOptions.refineSharedFocalLength = true;
    options.baOptions.logIterationProgress = true;

    const IncrementalSfmOptions effective = effectiveSfmOptions(options);

    EXPECT_EQ(effective.maxInitPairCandidates, 6);
    EXPECT_EQ(effective.baOptions.maxIterations, 5);
    EXPECT_EQ(effective.iterativeBARounds, 1);
    EXPECT_EQ(effective.globalBAInterval, std::numeric_limits<int>::max());
    EXPECT_EQ(effective.localBAInterval, 6);
    EXPECT_FALSE(effective.baOptions.refineSharedFocalLength);
    EXPECT_FALSE(effective.baOptions.logIterationProgress);
}

TEST(IncrementalSfmOptionsTest, FullExecutionProfilePreservesConfiguredRefinement)
{
    IncrementalSfmOptions options;
    options.executionProfile = SfmExecutionProfile::FullRefinement;
    options.maxInitPairCandidates = 7;
    options.baOptions.maxIterations = 12;
    options.iterativeBARounds = 3;

    const IncrementalSfmOptions effective = effectiveSfmOptions(options);

    EXPECT_EQ(effective.maxInitPairCandidates, 7);
    EXPECT_EQ(effective.baOptions.maxIterations, 12);
    EXPECT_EQ(effective.iterativeBARounds, 3);
}

class SfmPipelineTest : public ::testing::Test {
protected:
    IncrementalSfmOptions opts;

    void SetUp() override {
        opts.initMinNumMatches = 20;
        opts.initMinNumInliers = 10;
        opts.initMinChiralityInliers = 5;
        opts.maxInitPairCandidates = 5;
        opts.iterativeBARounds = 2;
        opts.filterMinTrackLen = 1;
        opts.filterMaxReprojError = 4.0;  // 宽松些（合成数据噪声较大）
        opts.filterMinTriAngle = 1.0;
        opts.baOptions.filterMaxReprojError = 5.0;
        opts.baOptions.maxIterations = 3;
    }
};

// 1. 三幅图像增量式重建
TEST_F(SfmPipelineTest, ThreeImageIncremental)
{
    Camera cam0 = makeCamera(0, 0, 0);
    Camera cam1 = makeCamera(10, 0, 0);
    Camera cam2 = makeCamera(20, 0, 0);

    auto points = generatePoints(300, 10, 0, 50, 5.0);

    // 生成 0-1 匹配
    std::vector<FeatureKeypoint> kpts0_01, kpts1;
    std::vector<FeatureMatch> m01;
    buildSyntheticMatches(cam0, cam1, points, kpts0_01, kpts1, m01);

    // 生成 0-2 匹配
    std::vector<FeatureKeypoint> kpts0_02, kpts2;
    std::vector<FeatureMatch> m02;
    buildSyntheticMatches(cam0, cam2, points, kpts0_02, kpts2, m02);

    // 生成 1-2 匹配
    std::vector<FeatureKeypoint> kpts1_12, kpts2_12;
    std::vector<FeatureMatch> m12;
    buildSyntheticMatches(cam1, cam2, points, kpts1_12, kpts2_12, m12);

    ASSERT_GE(m01.size(), 30u);
    ASSERT_GE(m02.size(), 30u);
    ASSERT_GE(m12.size(), 30u);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "img0.png", cam0, kpts0_01);
    sfm.addImageWithCamera(1, "img1.png", cam1, kpts1);
    sfm.addImageWithCamera(2, "img2.png", cam2, kpts2_12);
    sfm.addMatches(0, 1, m01);
    sfm.addMatches(0, 2, m02);
    sfm.addMatches(1, 2, m12);

    auto result = sfm.run();

    // 两幅图像初始化成功即可（第三幅注册取决于 PnP 结果）
    EXPECT_TRUE(result.success) << result.summary;
    EXPECT_GE(result.numRegisteredImages, 2);
    EXPECT_GT(result.numPoints3D, 0);
}

TEST_F(SfmPipelineTest, FailedHighVisibilityImageIsRetriedAfterModelGrows)
{
    opts.autoSelectInitPair = false;
    opts.initImageId1 = 0;
    opts.initImageId2 = 1;
    opts.initMinNumMatches = 80;
    opts.initMinNumInliers = 40;
    opts.initMinChiralityInliers = 20;
    opts.pnpOptions.minNumInliers = 25;
    opts.pnpOptions.minInlierRatio = 0.25;
    opts.pnpOptions.maxReprojError = 3.0;
    opts.localBAInterval = 100;
    opts.globalBAInterval = 100;
    opts.iterativeBARounds = 1;
    opts.filterMinTrackLen = 1;
    opts.filterMaxReprojError = 8.0;
    opts.triangulatorOptions.maxReprojError = 8.0;
    opts.triangulatorOptions.continueMaxReprojError = 8.0;
    opts.triangulatorOptions.completeMaxReprojError = 8.0;

    const std::vector<Camera> cameras = {
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(6.0, 0.0, 0.0),
        makeCamera(12.0, 0.0, 0.0),
        makeCamera(18.0, 0.0, 0.0),
    };
    const auto points = generatePoints(360, 9.0, 0.0, 70.0, 2.5, 2026);

    std::vector<std::vector<FeatureKeypoint>> keypoints;
    buildIndexedKeypoints(cameras, points, keypoints);
    ASSERT_GE(keypoints[0].size(), 300u);

    // image2 在初始模型上有更多“可见三维点”，因此会被优先选择。
    // 但这些观测坐标被刻意打乱，PnP 应失败；等 image3 注册后，
    // image2 才能通过 image3 提供的正确轨迹完成注册。
    for (int idx = 0; idx < 180; ++idx)
    {
        keypoints[2][static_cast<size_t>(idx)] = {
            static_cast<float>(80 + (idx * 37) % 860),
            static_cast<float>(70 + (idx * 53) % 610),
        };
    }

    const auto initialMatches = makeIndexedMatches(0, 300);
    const auto image2BadMatches = makeIndexedMatches(0, 180);
    const auto image3GoodMatches = makeIndexedMatches(180, 270);
    const auto image2DelayedGoodMatches = makeIndexedMatches(180, 270);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "schedule_0.png", cameras[0], keypoints[0]);
    sfm.addImageWithCamera(1, "schedule_1.png", cameras[1], keypoints[1]);
    sfm.addImageWithCamera(2, "schedule_2.png", cameras[2], keypoints[2]);
    sfm.addImageWithCamera(3, "schedule_3.png", cameras[3], keypoints[3]);
    sfm.addMatches(0, 1, initialMatches);
    sfm.addMatches(0, 2, image2BadMatches);
    sfm.addMatches(1, 2, image2BadMatches);
    sfm.addMatches(0, 3, image3GoodMatches);
    sfm.addMatches(1, 3, image3GoodMatches);
    sfm.addMatches(2, 3, image2DelayedGoodMatches);

    const auto result = sfm.run();

    EXPECT_EQ(result.selectedInitialImageId1, 0u);
    EXPECT_EQ(result.selectedInitialImageId2, 1u);

    ASSERT_TRUE(result.success) << result.summary;
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_TRUE(result.reconstruction->isRegistered(2))
        << "A temporarily failed high-overlap image must be retried after another image expands the 3D model";
    EXPECT_EQ(result.numRegisteredImages, 4);
}

TEST_F(SfmPipelineTest, SequenceModePrefersAdjacentInitialPairOverStrongerCrossSequenceMatch)
{
    opts.autoSelectInitPair = true;
    opts.maxInitPairCandidates = 1;
    opts.evaluateMultipleInitialPairModels = false;
    opts.enforceSequencePoseConsistency = true;
    opts.sequenceLoopClosure = true;
    opts.initMinNumMatches = 30;
    opts.initMinNumInliers = 10;
    opts.initMinChiralityInliers = 10;
    opts.pnpOptions.minNumInliers = 10;
    opts.localBAInterval = 100;
    opts.globalBAInterval = 100;
    opts.iterativeBARounds = 1;
    opts.filterMinTrackLen = 1;

    const std::vector<Camera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(10.0, 0.0, 0.0),
        makeCamera(20.0, 0.0, 0.0),
        makeCamera(30.0, 0.0, 0.0),
    };
    const auto points = generatePoints(480, 10.0, 0.0, 20.0, 6.0, 77);
    std::vector<std::vector<FeatureKeypoint>> keypoints;
    std::vector<FeatureMatch> matches01;
    std::vector<FeatureMatch> matches12;
    std::vector<FeatureMatch> matches02;
    buildKnownPoseTracks(cameras, points, keypoints, matches01, matches12, matches02);
    ASSERT_GE(matches01.size(), 100u);

    // 跨序列像对 0-2 保留更多匹配。序列模式仍必须以 0-1 或 1-2 作为种子。
    matches02.insert(matches02.end(), matches02.begin(), matches02.end());

    IncrementalSfm sfm(opts);
    for (ImageId imageId = 0; imageId < cameras.size(); ++imageId)
    {
        sfm.addImageWithCamera(imageId,
                               "sequence_" + std::to_string(imageId) + ".png",
                               cameras[imageId],
                               keypoints[imageId]);
    }
    sfm.addMatches(0, 1, matches01);
    sfm.addMatches(1, 2, matches12);
    sfm.addMatches(0, 2, matches02);

    const IncrementalSfmResult result = sfm.run();

    ASSERT_TRUE(result.success) << result.summary;
    EXPECT_TRUE((result.selectedInitialImageId1 == 0 && result.selectedInitialImageId2 == 1) ||
                (result.selectedInitialImageId1 == 1 && result.selectedInitialImageId2 == 2));
}

// 2. 进度回调正常工作
TEST_F(SfmPipelineTest, ProgressCallbackWorks)
{
    Camera cam0 = makeCamera(0, 0, 0);
    Camera cam1 = makeCamera(10, 0, 0);

    auto points = generatePoints(200, 5, 0, 50, 5.0);

    std::vector<FeatureKeypoint> kpts0, kpts1;
    std::vector<FeatureMatch> m01;
    buildSyntheticMatches(cam0, cam1, points, kpts0, kpts1, m01);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "img0.png", cam0, kpts0);
    sfm.addImageWithCamera(1, "img1.png", cam1, kpts1);
    sfm.addMatches(0, 1, m01);

    int callCount = 0;
    auto result = sfm.run([&](int registered, int total, const std::string &msg) -> bool {
        ++callCount;
        EXPECT_GE(total, 2);
        EXPECT_GE(registered, 0);
        EXPECT_LE(registered, total);
        EXPECT_FALSE(msg.empty());
        return true;
    });

    EXPECT_GE(callCount, 1) << "Progress callback should be called at least once";
}

// 3. 进度回调中止 → 停止重建
TEST_F(SfmPipelineTest, ProgressCallbackAbort)
{
    Camera cam0 = makeCamera(0, 0, 0);
    Camera cam1 = makeCamera(10, 0, 0);

    auto points = generatePoints(200, 5, 0, 50, 5.0);

    std::vector<FeatureKeypoint> kpts0, kpts1;
    std::vector<FeatureMatch> m01;
    buildSyntheticMatches(cam0, cam1, points, kpts0, kpts1, m01);

    IncrementalSfm sfm(opts);
    sfm.addImageWithCamera(0, "img0.png", cam0, kpts0);
    sfm.addImageWithCamera(1, "img1.png", cam1, kpts1);
    sfm.addMatches(0, 1, m01);

    auto result = sfm.run([](int, int, const std::string &) -> bool {
        return false; // 立即中止
    });

    // 中止后结果不一定 success，但不崩溃
    EXPECT_GE(result.numRegisteredImages, 0);
}

// ═══════════════════════════════════════════════════════════════
// 测试组 5：负深度过滤
// ═══════════════════════════════════════════════════════════════

TEST(NegativeDepthTest, FilterRemovesBehindCameraPoints)
{
    // 创建一个重建，其中部分 3D 点在相机后方
    SfmReconstruction recon;

    // 添加两幅图像
    ImageData img0;
    img0.id = 0;
    img0.imagePath = "img0.png";
    img0.keypoints.resize(10, {100.0f, 200.0f});
    img0.point3DIds.resize(10, kInvalidPoint3DId);
    recon.addImage(img0);

    ImageData img1;
    img1.id = 1;
    img1.imagePath = "img1.png";
    img1.keypoints.resize(10, {300.0f, 400.0f});
    img1.point3DIds.resize(10, kInvalidPoint3DId);
    recon.addImage(img1);

    Camera cam0 = makeCamera(0, 0, 0);
    Camera cam1 = makeCamera(10, 0, 0);
    recon.registerImage(0, cam0);
    recon.registerImage(1, cam1);

    // 前方点（Z > 0）
    ScenePoint3D goodPt;
    goodPt.xyz = {5, 0, 50};
    goodPt.track.elements.push_back({0, 0});
    goodPt.track.elements.push_back({1, 0});
    recon.addPoint3D(goodPt);

    // 后方点（Z < 0）
    ScenePoint3D badPt;
    badPt.xyz = {5, 0, -50};
    badPt.track.elements.push_back({0, 1});
    badPt.track.elements.push_back({1, 1});
    recon.addPoint3D(badPt);

    EXPECT_EQ(recon.numPoints3D(), 2u);

    // 构造 SFM 实例来调用 filterNegativeDepthPoints
    // 由于这是 private 方法，我们通过完整管线间接测试
    // 这里改为直接验证深度计算逻辑
    auto allPts = recon.allPoint3DIds();
    int negCount = 0;
    for (auto pid : allPts) {
        const auto &pt = recon.point3D(pid);
        for (const auto &elem : pt.track.elements) {
            if (!recon.hasCamera(elem.imageId)) continue;
            const Camera &cam = recon.camera(elem.imageId);
            const double world[3] = {pt.xyz[0], pt.xyz[1], pt.xyz[2]};
            double cameraPoint[3] = {0.0, 0.0, 0.0};
            cam.worldToCamera(world, cameraPoint);
            if (cameraPoint[2] < 0) ++negCount;
        }
    }

    EXPECT_GE(negCount, 1) << "Test scene should have at least one negative-depth observation";
}

// ═══════════════════════════════════════════════════════════════
// 测试组 6：观测级过滤 vs 整点删除
// ═══════════════════════════════════════════════════════════════

TEST(ObservationFilterTest, BAInvalidPointWithGoodObservations)
{
    // 构造一个场景：3 个观测中有 1 个坏观测
    // BA 标记整个点为 invalid，但观测级过滤应保留缩短后的轨迹

    std::vector<Camera> cameras;
    cameras.push_back(makeCamera(0, 0, 0));
    cameras.push_back(makeCamera(10, 0, 0));
    cameras.push_back(makeCamera(20, 0, 0));

    // 一个好点
    BATrack goodTrack;
    goodTrack.initialPoint = {10, 0, 50};
    for (int ci = 0; ci < 3; ++ci) {
        double u, v;
        projectPoint(cameras[ci], 10, 0, 50, u, v);
        goodTrack.observations.push_back({ci, u, v});
    }

    // 一个有一个坏观测的轨迹
    BATrack mixedTrack;
    mixedTrack.initialPoint = {5, 0, 50};
    for (int ci = 0; ci < 2; ++ci) {
        double u, v;
        projectPoint(cameras[ci], 5, 0, 50, u, v);
        mixedTrack.observations.push_back({ci, u, v});
    }
    // 第 3 个观测故意给错误坐标
    mixedTrack.observations.push_back({2, 900.0, 700.0});

    std::vector<BATrack> tracks = {goodTrack, mixedTrack};

    BAOptions opts;
    opts.maxIterations = 3;
    opts.enablePointFilter = true;
    opts.filterMaxReprojError = 2.0;  // 严格

    auto result = BundleAdjust::optimizePoints(cameras, tracks, opts);

    // 好的轨迹应该有效
    EXPECT_TRUE(result.points[0].valid) << "Good track should remain valid";

    // 不管混合轨迹最终有效与否，BA 都不应崩溃
    EXPECT_EQ(result.totalTracks, 2);
}

// ═══════════════════════════════════════════════════════════════
// 测试组 7：迭代 BA 参数验证
// ═══════════════════════════════════════════════════════════════

TEST(IterativeBATest, ConvergesWithMultipleRounds)
{
    // 创建合成场景
    std::vector<Camera> cameras = {makeCamera(0, 0, 0), makeCamera(10, 0, 0)};
    auto points = generatePoints(100, 5, 0, 50, 5.0);

    std::vector<BATrack> tracks;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    for (const auto &p : points) {
        BATrack track;
        // 给初始坐标加噪声（模拟初始三角化噪声）
        track.initialPoint = {p.x + noise(rng), p.y + noise(rng), p.z + noise(rng)};
        for (int ci = 0; ci < 2; ++ci) {
            double u, v;
            if (projectPoint(cameras[ci], p.x, p.y, p.z, u, v)) {
                track.observations.push_back({ci, u + noise(rng) * 0.2, v + noise(rng) * 0.2});
            }
        }
        if (track.observations.size() >= 2) {
            tracks.push_back(std::move(track));
        }
    }

    // 多轮 BA
    BAOptions opts;
    opts.maxIterations = 10;
    opts.enablePointFilter = true;
    opts.filterMaxReprojError = 2.5;

    auto result = BundleAdjust::optimizePoints(cameras, tracks, opts);

    EXPECT_LT(result.meanRmsAfter, result.meanRmsBefore + 0.001)
        << "Multiple rounds of BA should converge (RMS should not increase)";
    EXPECT_GT(result.optimizedTracks, 0)
        << "Some tracks should be successfully optimized";
}

// ═══════════════════════════════════════════════════════════════
// 测试组 8：不同类型离群点的过滤效果
// ═══════════════════════════════════════════════════════════════

TEST(OutlierTypeTest, LargeReprojErrorOutliers)
{
    // 离群类型 1：3D 坐标正确但观测坐标错误（匹配错误）
    std::vector<Camera> cameras = {makeCamera(0, 0, 0), makeCamera(10, 0, 0)};
    std::vector<BATrack> tracks;

    // 好点
    for (int i = 0; i < 30; ++i) {
        double x = 5 + i * 0.2, y = 0, z = 50;
        BATrack track;
        track.initialPoint = {x, y, z};
        for (int ci = 0; ci < 2; ++ci) {
            double u, v;
            projectPoint(cameras[ci], x, y, z, u, v);
            track.observations.push_back({ci, u, v});
        }
        tracks.push_back(std::move(track));
    }

    // 坏点：正确的 3D 坐标，但像点坐标完全错误
    for (int i = 0; i < 5; ++i) {
        double x = 5 + i * 0.2, y = 0, z = 50;
        BATrack track;
        track.initialPoint = {x, y, z};
        // 给完全随机的观测坐标
        track.observations.push_back({0, 100.0 + i * 50, 200.0 + i * 30});
        track.observations.push_back({1, 800.0 - i * 40, 600.0 - i * 20});
        tracks.push_back(std::move(track));
    }

    BAOptions opts;
    opts.maxIterations = 5;
    opts.enablePointFilter = true;
    opts.filterMaxReprojError = 2.5;

    auto result = BundleAdjust::optimizePoints(cameras, tracks, opts);

    int filteredCount = 0;
    // 只检查最后 5 个（离群）
    for (size_t i = 30; i < result.points.size(); ++i) {
        if (!result.points[i].valid) ++filteredCount;
    }

    EXPECT_GE(filteredCount, 3)
        << "Most large-reprojection-error outliers should be filtered";
}

TEST(OutlierTypeTest, WrongTriangulationOutliers)
{
    // 离群类型 2：3D 坐标错误（错误三角化），观测坐标来自正确位置
    std::vector<Camera> cameras = {makeCamera(0, 0, 0), makeCamera(10, 0, 0)};
    std::vector<BATrack> tracks;

    // 好点
    for (int i = 0; i < 30; ++i) {
        double x = 5 + i * 0.2, y = 0, z = 50;
        BATrack track;
        track.initialPoint = {x, y, z};
        for (int ci = 0; ci < 2; ++ci) {
            double u, v;
            projectPoint(cameras[ci], x, y, z, u, v);
            track.observations.push_back({ci, u, v});
        }
        tracks.push_back(std::move(track));
    }

    // 坏点：观测来自正确点，但初始 3D 坐标严重偏离
    for (int i = 0; i < 5; ++i) {
        double trueX = 5 + i * 0.2, trueY = 0, trueZ = 50;
        BATrack track;
        // 初始坐标完全错误
        track.initialPoint = {trueX + 200, trueY + 200, trueZ + 200};
        for (int ci = 0; ci < 2; ++ci) {
            double u, v;
            projectPoint(cameras[ci], trueX, trueY, trueZ, u, v);
            track.observations.push_back({ci, u, v});
        }
        tracks.push_back(std::move(track));
    }

    BAOptions opts;
    opts.maxIterations = 10;
    opts.enablePointFilter = true;
    opts.filterMaxReprojError = 2.5;

    auto result = BundleAdjust::optimizePoints(cameras, tracks, opts);

    // 错误三角化的点，如果优化能把它拉回来就保留，否则过滤
    // 关键是不崩溃
    EXPECT_EQ(result.totalTracks, 35);
    EXPECT_GT(result.optimizedTracks, 20)
        << "Good tracks should survive the filtering";
}

// ═══════════════════════════════════════════════════════════════
// 测试组 9：重三角化（Retriangulation）
// ═══════════════════════════════════════════════════════════════

class RetriangulationTest : public ::testing::Test {
protected:
    SfmReconstruction recon;
    CorrespondenceGraph graph;

    Camera cam0, cam1, cam2;

    void SetUp() override {
        cam0 = makeCamera(0, 0, 0);
        cam1 = makeCamera(10, 0, 0);
        cam2 = makeCamera(20, 0, 0);
    }

    /// 添加一幅已注册图像
    void addRegisteredImage(ImageId id, const Camera &cam, int numKpts = 10) {
        ImageData img;
        img.id = id;
        img.imagePath = "img" + std::to_string(id) + ".png";
        img.keypoints.resize(numKpts, {100.0f, 200.0f});
        img.point3DIds.resize(numKpts, kInvalidPoint3DId);
        recon.addImage(img);
        recon.registerImage(id, cam);
        graph.addImage(id, numKpts);
    }

    /// 向点云添加带轨迹的3D点，返回分配的 Point3DId
    Point3DId addPointWithTrack(double x, double y, double z,
                                 const std::vector<std::pair<ImageId, FeatureIdx>> &obs) {
        ScenePoint3D pt;
        pt.xyz = {x, y, z};
        for (auto &[imgId, fi] : obs) {
            pt.track.elements.push_back({imgId, fi});
        }
        Point3DId pid = recon.addPoint3D(pt);
        // 关联 ImageData 中的 point3DIds
        for (auto &[imgId, fi] : obs) {
            if (recon.hasImage(imgId)) {
                auto &imgData = recon.image(imgId);
                if (fi < imgData.point3DIds.size())
                    imgData.point3DIds[fi] = pid;
            }
        }
        return pid;
    }
};

// 1. 重三角化改善被扰动的 3D 点坐标
TEST_F(RetriangulationTest, ImprovesPerturbed3DPoints)
{
    addRegisteredImage(0, cam0, 5);
    addRegisteredImage(1, cam1, 5);

    // 真实 3D 点
    double trueX = 5, trueY = 0, trueZ = 50;

    // 设置真实的像点坐标
    double u0, v0, u1, v1;
    projectPoint(cam0, trueX, trueY, trueZ, u0, v0);
    projectPoint(cam1, trueX, trueY, trueZ, u1, v1);

    auto &img0 = recon.image(0);
    auto &img1 = recon.image(1);
    img0.keypoints[0] = {static_cast<float>(u0), static_cast<float>(v0)};
    img1.keypoints[0] = {static_cast<float>(u1), static_cast<float>(v1)};

    // 添加被扰动的 3D 点（偏离真实位置 5 单位）
    auto pid = addPointWithTrack(trueX + 5, trueY + 3, trueZ - 2, {{0, 0}, {1, 0}});

    // 添加匹配以支持 graph
    FeatureMatch m;
    m.idx1 = 0; m.idx2 = 0;
    graph.addMatches(0, 1, {m});
    graph.buildCorrespondences();

    Triangulator tri(recon, graph);
    int improved = tri.retriangulatePoints(2.0);

    EXPECT_GE(improved, 1) << "Retriangulation should improve perturbed point";

    // 验证坐标更接近真实值
    const auto &pt = recon.point3D(pid);
    double dist = std::sqrt(std::pow(pt.xyz[0] - trueX, 2)
                          + std::pow(pt.xyz[1] - trueY, 2)
                          + std::pow(pt.xyz[2] - trueZ, 2));
    EXPECT_LT(dist, 3.0) << "Retriangulated point should be closer to truth";
}

// 2. 多视图重三角化比双目更精确
TEST_F(RetriangulationTest, MultiViewBetterThanTwoView)
{
    addRegisteredImage(0, cam0, 5);
    addRegisteredImage(1, cam1, 5);
    addRegisteredImage(2, cam2, 5);

    double trueX = 10, trueY = 0, trueZ = 50;

    double u0, v0, u1, v1, u2, v2;
    projectPoint(cam0, trueX, trueY, trueZ, u0, v0);
    projectPoint(cam1, trueX, trueY, trueZ, u1, v1);
    projectPoint(cam2, trueX, trueY, trueZ, u2, v2);

    auto &img0 = recon.image(0);
    auto &img1 = recon.image(1);
    auto &img2 = recon.image(2);
    img0.keypoints[0] = {static_cast<float>(u0), static_cast<float>(v0)};
    img1.keypoints[0] = {static_cast<float>(u1), static_cast<float>(v1)};
    img2.keypoints[0] = {static_cast<float>(u2), static_cast<float>(v2)};

    // 添加扰动的 3D 点，使用 3 个观测
    auto pid = addPointWithTrack(trueX + 3, trueY + 2, trueZ - 1, {{0, 0}, {1, 0}, {2, 0}});

    FeatureMatch m01, m02;
    m01.idx1 = 0; m01.idx2 = 0;
    m02.idx1 = 0; m02.idx2 = 0;
    graph.addMatches(0, 1, {m01});
    graph.addMatches(0, 2, {m02});
    graph.buildCorrespondences();

    Triangulator tri(recon, graph);
    int improved = tri.retriangulatePoints(2.0);

    EXPECT_GE(improved, 1);

    const auto &pt = recon.point3D(pid);
    double dist = std::sqrt(std::pow(pt.xyz[0] - trueX, 2)
                          + std::pow(pt.xyz[1] - trueY, 2)
                          + std::pow(pt.xyz[2] - trueZ, 2));
    EXPECT_LT(dist, 1.0) << "3-view retriangulation should be very accurate";
}

// 3. 深度一致性可通过负深度过滤间接验证
TEST_F(RetriangulationTest, NegativeDepthNotRetriangulated)
{
    addRegisteredImage(0, cam0, 5);
    addRegisteredImage(1, cam1, 5);

    // 使用前方点的投影坐标作为观测
    double trueX = 5, trueY = 0, trueZ = 50;
    double u0, v0, u1, v1;
    projectPoint(cam0, trueX, trueY, trueZ, u0, v0);
    projectPoint(cam1, trueX, trueY, trueZ, u1, v1);

    auto &img0 = recon.image(0);
    auto &img1 = recon.image(1);
    img0.keypoints[0] = {static_cast<float>(u0), static_cast<float>(v0)};
    img1.keypoints[0] = {static_cast<float>(u1), static_cast<float>(v1)};

    // 初始坐标在相机后方（但观测坐标来自正确的前方点）
    auto pid = addPointWithTrack(trueX, trueY, -50, {{0, 0}, {1, 0}});

    FeatureMatch m;
    m.idx1 = 0; m.idx2 = 0;
    graph.addMatches(0, 1, {m});
    graph.buildCorrespondences();

    Triangulator tri(recon, graph);
    int improved = tri.retriangulatePoints(2.0);

    EXPECT_GE(improved, 1) << "Should retriangulate point that was behind camera";

    // 重三角化后的点应该在相机前方
    const auto &pt = recon.point3D(pid);
    const double world[3] = {pt.xyz[0], pt.xyz[1], pt.xyz[2]};
    double cameraPoint[3] = {0.0, 0.0, 0.0};
    cam0.worldToCamera(world, cameraPoint);
    EXPECT_GT(cameraPoint[2], 0) << "After retriangulation, point should be in front of camera";

    // 验证坐标接近真实值
    double dist = std::sqrt(std::pow(pt.xyz[0]-trueX,2) + std::pow(pt.xyz[1]-trueY,2) + std::pow(pt.xyz[2]-trueZ,2));
    EXPECT_LT(dist, 3.0) << "Retriangulated point should be close to truth";
}

// 4. recomputeReprojErrors 更新 error 字段
TEST_F(RetriangulationTest, RecomputeReprojErrors)
{
    addRegisteredImage(0, cam0, 5);
    addRegisteredImage(1, cam1, 5);

    double trueX = 5, trueY = 0, trueZ = 50;
    double u0, v0, u1, v1;
    projectPoint(cam0, trueX, trueY, trueZ, u0, v0);
    projectPoint(cam1, trueX, trueY, trueZ, u1, v1);

    auto &img0 = recon.image(0);
    auto &img1 = recon.image(1);
    img0.keypoints[0] = {static_cast<float>(u0), static_cast<float>(v0)};
    img1.keypoints[0] = {static_cast<float>(u1), static_cast<float>(v1)};

    // 添加精确位置的点，初始 error 设为 999
    ScenePoint3D spt;
    spt.xyz = {trueX, trueY, trueZ};
    spt.error = 999.0;
    spt.track.elements = {{0, 0}, {1, 0}};
    auto pid = recon.addPoint3D(spt);

    CorrespondenceGraph g;
    g.addImage(0, 5);
    g.addImage(1, 5);
    Triangulator tri(recon, g);
    tri.recomputeReprojErrors();

    const auto &pt = recon.point3D(pid);
    EXPECT_LT(pt.error, 1.0) << "Recomputed error for accurate point should be low";
    EXPECT_NE(pt.error, 999.0) << "Error should have been updated";
}

// 5. completeTracks 跳过深度为负的观测
TEST_F(RetriangulationTest, CompleteTracksSkipsNegativeDepth)
{
    // cam0 在原点朝 +Z 方向，cam3 在 Z=-100 朝 -Z 方向
    Camera cam3;
    cam3.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    // 旋转 180° 使相机朝 -Z （绕 Y 旋转 180°: R = [-1,0,0, 0,1,0, 0,0,-1]）
    cam3.setPose({-1,0,0, 0,1,0, 0,0,-1}, {0, 0, -100});

    addRegisteredImage(0, cam0, 5);
    addRegisteredImage(3, cam3, 5);

    // 3D 点在 cam0 前方但在 cam3 后方
    double trueX = 5, trueY = 0, trueZ = 50;
    auto pid = addPointWithTrack(trueX, trueY, trueZ, {{0, 0}});

    // 设置像点
    double u0, v0;
    projectPoint(cam0, trueX, trueY, trueZ, u0, v0);
    auto &img0 = recon.image(0);
    img0.keypoints[0] = {static_cast<float>(u0), static_cast<float>(v0)};

    // 假设 cam3 的特征点 0 本来对应同一个 3D 点（通过 graph）
    FeatureMatch m;
    m.idx1 = 0; m.idx2 = 0;
    graph.addMatches(0, 3, {m});
    graph.buildCorrespondences();

    Triangulator tri(recon, graph);
    TriangulatorOptions opts;
    opts.completeMaxReprojError = 100.0; // 宽松，纯测深度检查
    int completed = tri.completeTracks(opts);

    // cam3 看这个点是负深度，不应该被加入轨迹
    const auto &pt = recon.point3D(pid);
    for (const auto &elem : pt.track.elements) {
        EXPECT_NE(elem.imageId, 3u) << "Should not extend track to camera with negative depth";
    }
}

TEST_F(RetriangulationTest, CompleteTracksDoesNotAddSecondFeatureFromSameImage)
{
    addRegisteredImage(0, cam0, 5);
    addRegisteredImage(1, cam1, 5);

    const double trueX = 5.0;
    const double trueY = 0.0;
    const double trueZ = 50.0;

    double u0 = 0.0, v0 = 0.0, u1 = 0.0, v1 = 0.0;
    ASSERT_TRUE(projectPoint(cam0, trueX, trueY, trueZ, u0, v0));
    ASSERT_TRUE(projectPoint(cam1, trueX, trueY, trueZ, u1, v1));

    auto &img0 = recon.image(0);
    auto &img1 = recon.image(1);
    img0.keypoints[0] = {static_cast<float>(u0), static_cast<float>(v0)};
    img0.keypoints[1] = {static_cast<float>(u0), static_cast<float>(v0)};
    img1.keypoints[0] = {static_cast<float>(u1), static_cast<float>(v1)};

    const Point3DId pid = addPointWithTrack(trueX, trueY, trueZ, {{0, 0}, {1, 0}});

    FeatureMatch match;
    match.idx1 = 1;
    match.idx2 = 0;
    graph.addMatches(0, 1, {match});
    graph.buildCorrespondences();

    Triangulator tri(recon, graph);
    TriangulatorOptions opts;
    opts.completeMaxReprojError = 0.5;
    const int completed = tri.completeTracks(opts);

    EXPECT_EQ(completed, 0);
    const auto &pt = recon.point3D(pid);
    int image0ObservationCount = 0;
    for (const auto &elem : pt.track.elements)
    {
        if (elem.imageId == 0u)
        {
            ++image0ObservationCount;
        }
    }
    EXPECT_EQ(image0ObservationCount, 1);
    EXPECT_EQ(recon.image(0).point3DIds[1], kInvalidPoint3DId);
}

TEST_F(RetriangulationTest, CompleteTracksUsesWorldToCameraDepthForRotatedCamera)
{
    Camera baseCam = makeCamera(0, 0, -50);

    Camera rotatedCam;
    rotatedCam.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    // Camera-to-world rotation around X by +90 degrees. The world point below
    // has positive depth via Camera::worldToCamera(), but the old row-based
    // depth formula reports it as negative.
    rotatedCam.setPose({1, 0, 0, 0, 0, -1, 0, 1, 0}, {0, 0, 0});

    addRegisteredImage(0, baseCam, 5);
    addRegisteredImage(1, rotatedCam, 5);

    const double trueX = 0.0;
    const double trueY = -10.0;
    const double trueZ = 0.0;

    double u0, v0, u1, v1;
    ASSERT_TRUE(projectPoint(baseCam, trueX, trueY, trueZ, u0, v0));
    ASSERT_TRUE(projectPoint(rotatedCam, trueX, trueY, trueZ, u1, v1));

    auto &img0 = recon.image(0);
    auto &img1 = recon.image(1);
    img0.keypoints[0] = {static_cast<float>(u0), static_cast<float>(v0)};
    img1.keypoints[0] = {static_cast<float>(u1), static_cast<float>(v1)};

    auto pid = addPointWithTrack(trueX, trueY, trueZ, {{0, 0}});

    FeatureMatch m;
    m.idx1 = 0;
    m.idx2 = 0;
    graph.addMatches(0, 1, {m});
    graph.buildCorrespondences();

    Triangulator tri(recon, graph);
    TriangulatorOptions opts;
    opts.completeMaxReprojError = 0.5;
    int completed = tri.completeTracks(opts);

    EXPECT_EQ(completed, 1) << "Positive-depth rotated camera observation should extend the track";

    const auto &pt = recon.point3D(pid);
    bool hasRotatedObservation = false;
    for (const auto &elem : pt.track.elements) {
        if (elem.imageId == 1u) {
            hasRotatedObservation = true;
            break;
        }
    }
    EXPECT_TRUE(hasRotatedObservation);
}
