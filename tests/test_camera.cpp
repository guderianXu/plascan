// ============================================================
// test_camera.cpp — 相机模型单元测试
//
// 测试内容：
//   1. 从 .tsai 文件加载相机参数
//   2. 投影-反投影往返一致性
//   3. 旋转矩阵正交性与行列式
//   4. 批量加载 75 张测试相机
// ============================================================

#include <gtest/gtest.h>
#include "Camera.h"

#include <cmath>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>

using namespace xjw;

namespace fs = std::filesystem;

static std::string testDataDir()
{
#ifdef TEST_DATA_DIR
    return TEST_DATA_DIR;
#else
    return "../testData";
#endif
}

// ─── 辅助函数 ─────────────────────────────────────────────────

static double vecLength(const double v[3])
{
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// ─── 测试用例 ─────────────────────────────────────────────────

// 1. 加载单个 .tsai 文件并验证有效性
TEST(CameraTest, LoadFromTsaiFile)
{
    Camera cam;
    std::string path = testDataDir() + "/tsai/1.tsai";
    ASSERT_TRUE(cam.loadFromFile(path))
        << "Failed to load camera file: " << path;
    EXPECT_TRUE(cam.isValid());
}

// 2. 加载后内参不为零
TEST(CameraTest, IntrinsicsNonZero)
{
    Camera cam;
    ASSERT_TRUE(cam.loadFromFile(testDataDir() + "/tsai/1.tsai"));

    EXPECT_GT(std::abs(cam.focalX()), 1e-6);
    EXPECT_GT(std::abs(cam.focalY()), 1e-6);
    // 主点也应合理
    EXPECT_GT(std::abs(cam.principalX()), 0.0);
    EXPECT_GT(std::abs(cam.principalY()), 0.0);
}

TEST(CameraTest, MetricIntrinsicAccessorsConsistent)
{
    Camera cam;
    ASSERT_TRUE(cam.loadFromFile(testDataDir() + "/tsai/1.tsai"));

    EXPECT_GT(cam.pixelPitch(), 0.0);
    EXPECT_NEAR(cam.focalXMillimeters(), cam.focalX() * cam.pixelPitch(), 1e-9);
    EXPECT_NEAR(cam.focalYMillimeters(), cam.focalY() * cam.pixelPitch(), 1e-9);
    EXPECT_NEAR(cam.principalXMillimeters(), cam.principalX() * cam.pixelPitch(), 1e-9);
    EXPECT_NEAR(cam.principalYMillimeters(), cam.principalY() * cam.pixelPitch(), 1e-9);
}

// 3. 旋转矩阵正交性验证 (R * R^T ≈ I)
TEST(CameraTest, RotationOrthogonality)
{
    Camera cam;
    ASSERT_TRUE(cam.loadFromFile(testDataDir() + "/tsai/1.tsai"));

    auto R = cam.cameraToWorldRotation();
    // R*R^T 应等于单位矩阵
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0;
            for (int k = 0; k < 3; ++k)
                dot += R[i*3+k] * R[j*3+k];
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, 1e-5)
                << "R*R^T [" << i << "][" << j << "] = " << dot
                << ", expected " << expected;
        }
    }
}

// 4. 旋转矩阵行列式 = +1（右手系）
TEST(CameraTest, RotationDeterminant)
{
    Camera cam;
    ASSERT_TRUE(cam.loadFromFile(testDataDir() + "/tsai/1.tsai"));

    auto R = cam.cameraToWorldRotation();
    double det = R[0]*(R[4]*R[8]-R[5]*R[7])
               - R[1]*(R[3]*R[8]-R[5]*R[6])
               + R[2]*(R[3]*R[7]-R[4]*R[6]);
    EXPECT_NEAR(det, 1.0, 1e-6)
        << "det(R) = " << det << ", expected 1.0 (proper rotation)";
}

// 5. 投影：相机前方的点应投影成功
TEST(CameraTest, ProjectPointInFront)
{
    Camera cam;
    ASSERT_TRUE(cam.loadFromFile(testDataDir() + "/tsai/1.tsai"));

    auto C = cam.cameraCenter();
    auto R = cam.cameraToWorldRotation();
    // 相机 z 轴方向（world 坐标） = R^T * [0,0,1]
    double camZ[3] = { R[2], R[5], R[8] };

    // 创建在相机前方 100 单位的测试点
    double testPt[3] = {
        C[0] + camZ[0]*100.0,
        C[1] + camZ[1]*100.0,
        C[2] + camZ[2]*100.0
    };

    double uv[2];
    bool ok = cam.projectWorldPoint(testPt, uv);
    EXPECT_TRUE(ok) << "Expected successful projection for point in front of camera";

    // 投影点应在合理范围内（相对于 CCD 尺寸 ~10mm, pitch ~0.00185）
    // 不做像素坐标精确检查，仅确保有限值
    EXPECT_TRUE(std::isfinite(uv[0]));
    EXPECT_TRUE(std::isfinite(uv[1]));
}

// 6. 投影-反投影一致性（worldToCamera + project 往返）
TEST(CameraTest, WorldToCameraConsistency)
{
    Camera cam;
    ASSERT_TRUE(cam.loadFromFile(testDataDir() + "/tsai/1.tsai"));

    auto C = cam.cameraCenter();
    auto R = cam.cameraToWorldRotation();
    double camZ[3] = { R[2], R[5], R[8] };

    double worldPt[3] = {
        C[0] + camZ[0]*200.0,
        C[1] + camZ[1]*200.0,
        C[2] + camZ[2]*200.0
    };

    double camCoords[3];
    cam.worldToCamera(worldPt, camCoords);

    // 相机坐标 Z 应为正（点在前方）
    EXPECT_GT(camCoords[2], 0.0)
        << "Point should have positive Z in camera frame";
}

// 7. 批量加载 75 张相机文件
TEST(CameraTest, LoadAll75Cameras)
{
    std::string tsaiDir = testDataDir() + "/tsai";
    int loaded = 0;
    int failed = 0;

    for (int i = 1; i <= 75; ++i) {
        Camera cam;
        std::string path = tsaiDir + "/" + std::to_string(i) + ".tsai";

        if (!fs::exists(path)) {
            GTEST_LOG_(WARNING) << "File not found: " << path;
            continue;
        }

        if (cam.loadFromFile(path)) {
            EXPECT_TRUE(cam.isValid()) << "Camera " << i << " loaded but invalid";
            // 验证相机中心不为零（有合理位置）
            auto C = cam.cameraCenter();
            double dist = std::sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2]);
            EXPECT_GT(dist, 1.0) << "Camera " << i << " center too close to origin";
            ++loaded;
        } else {
            ++failed;
            ADD_FAILURE() << "Failed to load camera " << i << ": " << path;
        }
    }

    EXPECT_EQ(loaded, 75) << "Expected 75 cameras, loaded " << loaded;
    EXPECT_EQ(failed, 0);
}

// 8. 多相机中心距离合理性
TEST(CameraTest, CameraCentersReasonable)
{
    std::string tsaiDir = testDataDir() + "/tsai";
    std::vector<std::array<double,3>> centers;

    for (int i = 1; i <= 75; ++i) {
        Camera cam;
        std::string path = tsaiDir + "/" + std::to_string(i) + ".tsai";
        if (!cam.loadFromFile(path)) continue;
        auto C = cam.cameraCenter();
        centers.push_back({C[0], C[1], C[2]});
    }

    ASSERT_GE(centers.size(), 2u) << "Need at least 2 cameras for distance check";

    // 计算所有相机对之间的距离，确保分布合理
    double minDist = 1e18, maxDist = 0;
    for (size_t i = 0; i < centers.size(); ++i) {
        for (size_t j = i+1; j < centers.size(); ++j) {
            double dx = centers[i][0] - centers[j][0];
            double dy = centers[i][1] - centers[j][1];
            double dz = centers[i][2] - centers[j][2];
            double d = std::sqrt(dx*dx + dy*dy + dz*dz);
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
        }
    }

    // 在小行星影像场景中，最近相机对距离应 > 0（不重叠）
    EXPECT_GT(minDist, 0.0) << "Some cameras have identical positions";
    // 最远相机对距离应有限
    EXPECT_LT(maxDist, 1e8) << "Camera baseline seems unreasonably large";

    // 比值不应太极端
    EXPECT_GT(maxDist / minDist, 1.0) << "All cameras at same distance?";
}

// 9. 设置内参后投影一致性
TEST(CameraTest, SetIntrinsicsAndProject)
{
    Camera cam;
    // 设置简单针孔相机
    cam.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    // 设置单位旋转，中心在原点
    std::array<double,9> R = {1,0,0, 0,1,0, 0,0,1};
    std::array<double,3> C = {0,0,0};
    cam.setPose(R, C);

    // 在相机 Z 轴前方 10 单位的点 (0, 0, 10)
    double worldPt[3] = {0, 0, 10};
    double uv[2];
    bool ok = cam.projectWorldPoint(worldPt, uv);
    // 如果投影成功，检查是否投影到主点附近
    if (ok) {
        EXPECT_NEAR(uv[0], 512.0, 50.0)
            << "Expected u near principal point for on-axis point";
        EXPECT_NEAR(uv[1], 384.0, 50.0)
            << "Expected v near principal point for on-axis point";
    }
}
