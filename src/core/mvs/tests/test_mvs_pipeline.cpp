// =============================================================================
// MVS 管道集成测试
// 覆盖：CPU PatchMatch 深度估计 → DepthMapFusion 融合 → DenseCloudBuilder 反投影
// =============================================================================

#include <gtest/gtest.h>

#include "PatchMatchCUDA.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "DepthPyramidEstimator.h"
#include "DepthPyramidPropagation.h"
#include "DenseCloudBuilder.h"
#include "DepthPyramidPolicy.h"
#include "DepthFrameQualityGate.h"
#include "DepthConsistencyCache.h"
#include "DepthGeometryConsistency.h"
#include "EpipolarRectifier.h"
#include "MvsSceneClassifier.h"
#include "MvsImagePreprocessor.h"
#include "MvsQualityReport.h"
#include "SparseCloudPreprocessor.h"
#include "FramePinholeCamera.h"

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/io/ply_io.h>

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// 合成一张有纹理的灰度图
cv::Mat makeSyntheticGray(int w, int h)
{
    cv::Mat img(h, w, CV_8U);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            double v = 128.0
                + 40.0 * std::sin(0.21 * x)
                + 35.0 * std::cos(0.17 * y)
                + (((x / 5) + (y / 6)) % 2 == 0 ? 20.0 : -20.0);
            img.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
        }
    return img;
}

// 水平位移图（模拟视差）
cv::Mat makeShifted(const cv::Mat &src, int d)
{
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            if (x + d >= 0 && x + d < src.cols)
                dst.at<uint8_t>(y, x) = src.at<uint8_t>(y, x + d);
    return dst;
}

xjw::FramePinholeCamera makeMvsCamera(double fu, double fv,
                          double cu, double cv,
                          const double Rwc[9],
                          const double C[3])
{
    xjw::FramePinholeCamera cam;
    std::array<double, 9> R{Rwc[0],Rwc[1],Rwc[2],Rwc[3],Rwc[4],Rwc[5],Rwc[6],Rwc[7],Rwc[8]};
    std::array<double, 3> Cv{C[0],C[1],C[2]};
    cam.setIntrinsics(fu, fv, cu, cv);
    cam.setPose(R, Cv);
    cam.setAxisDirections(1, 1);
    cam.setDepthAxisFlipped(false);
    return cam.normalizedForPositiveDepth();
}

std::vector<xjw::mvs::CameraView> makeDownLookingGridViews(int columns, int rows)
{
    std::vector<xjw::mvs::CameraView> views;
    views.reserve(static_cast<size_t>(columns * rows));

    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            xjw::mvs::CameraView view;
            view.imageWidth = 640;
            view.imageHeight = 480;
            view.camera.setIntrinsics(500.0, 500.0, 320.0, 240.0);
            view.camera.setPose(
                std::array<double, 9>{1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0},
                std::array<double, 3>{static_cast<double>(column - columns / 2),
                                      static_cast<double>(row - rows / 2),
                                      -10.0});
            view.camera.setAxisDirections(1, 1);
            view.camera.setDepthAxisFlipped(false);
            views.push_back(std::move(view));
        }
    }
    return views;
}

xjw::mvs::SparseCloud makePlanarSparseCloud()
{
    xjw::mvs::SparseCloud sparse;
    for (int y = -5; y <= 5; ++y)
    {
        for (int x = -5; x <= 5; ++x)
        {
            sparse.points.push_back({static_cast<float>(x),
                                     static_cast<float>(y),
                                     0.01f * static_cast<float>((x + y) % 3)});
        }
    }
    return sparse;
}

xjw::mvs::SparseCloud makeVolumetricSparseCloud()
{
    xjw::mvs::SparseCloud sparse;
    for (int z = -2; z <= 2; ++z)
    {
        for (int y = -2; y <= 2; ++y)
        {
            for (int x = -2; x <= 2; ++x)
            {
                sparse.points.push_back({static_cast<float>(x),
                                         static_cast<float>(y),
                                         static_cast<float>(z)});
            }
        }
    }
    return sparse;
}

xjw::mvs::CameraView makeViewWithForward(const cv::Vec3d &center,
                                         const cv::Vec3d &forward_direction)
{
    const cv::Vec3d forward = forward_direction / cv::norm(forward_direction);
    cv::Vec3d reference_up(0.0, 0.0, 1.0);
    if (std::abs(forward.dot(reference_up)) > 0.95)
    {
        reference_up = cv::Vec3d(0.0, 1.0, 0.0);
    }
    const cv::Vec3d right = reference_up.cross(forward) /
                            cv::norm(reference_up.cross(forward));
    const cv::Vec3d camera_up = forward.cross(right);

    xjw::mvs::CameraView view;
    view.imageWidth = 640;
    view.imageHeight = 480;
    view.camera.setIntrinsics(500.0, 500.0, 320.0, 240.0);
    view.camera.setPose(
        std::array<double, 9>{right[0], camera_up[0], forward[0],
                              right[1], camera_up[1], forward[1],
                              right[2], camera_up[2], forward[2]},
        std::array<double, 3>{center[0], center[1], center[2]});
    view.camera.setAxisDirections(1, 1);
    view.camera.setDepthAxisFlipped(false);
    return view;
}

xjw::mvs::CameraView makeLookAtView(const cv::Vec3d &center,
                                    const cv::Vec3d &target)
{
    return makeViewWithForward(center, target - center);
}

std::vector<xjw::mvs::CameraView> makeOrbitalRingViews(int count)
{
    std::vector<xjw::mvs::CameraView> views;
    views.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        const double angle = 2.0 * CV_PI * static_cast<double>(index) /
                             static_cast<double>(count);
        views.push_back(makeLookAtView(
            cv::Vec3d(10.0 * std::cos(angle), 10.0 * std::sin(angle), 2.0),
            cv::Vec3d(0.0, 0.0, 0.0)));
    }
    return views;
}

class RecordingPatchMatchBackend final : public xjw::mvs::IPatchMatchBackend
{
public:
    bool estimate(const xjw::mvs::PatchMatchBackendRequest &request,
                  xjw::mvs::DepthLevelResult &result,
                  std::string *error_message) override
    {
        _downsampleCalls.push_back(request.levelConfig.patchMatch.downsampleFactor);
        _nativeOutputCalls.push_back(request.levelConfig.patchMatch.returnNativeResolution);
        _validMaskSizes.push_back(request.referenceValidMask.size());
        _validMaskValues.push_back(request.referenceValidMask.empty()
                                       ? -1
                                       : cv::countNonZero(request.referenceValidMask));
        result.level = request.levelConfig.level;
        result.downsampleFactor = request.levelConfig.patchMatch.downsampleFactor;
        const cv::Size output_size = request.levelConfig.patchMatch.returnNativeResolution
            ? xjw::mvs::depthPyramidWorkingSize(
                  request.referenceImage.cols,
                  request.referenceImage.rows,
                  request.levelConfig.patchMatch.downsampleFactor)
            : request.referenceImage.size();
        result.depth = cv::Mat(output_size, CV_32F, cv::Scalar(10.0f));
        result.confidence = cv::Mat(output_size, CV_32F, cv::Scalar(0.8f));
        result.supportCount = cv::Mat(output_size, CV_16U, cv::Scalar(4));
        result.uncertainty = cv::Mat(output_size, CV_32F, cv::Scalar(0.2f));
        result.validMask = cv::Mat(output_size, CV_8U, cv::Scalar(255));
        if (error_message)
        {
            error_message->clear();
        }
        return true;
    }

    const std::vector<int> &downsampleCalls() const
    {
        return _downsampleCalls;
    }

    const std::vector<cv::Size> &validMaskSizes() const
    {
        return _validMaskSizes;
    }

    const std::vector<bool> &nativeOutputCalls() const
    {
        return _nativeOutputCalls;
    }

    const std::vector<int> &validMaskValues() const
    {
        return _validMaskValues;
    }

private:
    std::vector<int> _downsampleCalls;
    std::vector<bool> _nativeOutputCalls;
    std::vector<cv::Size> _validMaskSizes;
    std::vector<int> _validMaskValues;
};

class CollapsingFinePatchMatchBackend final : public xjw::mvs::IPatchMatchBackend
{
public:
    bool estimate(const xjw::mvs::PatchMatchBackendRequest &request,
                  xjw::mvs::DepthLevelResult &result,
                  std::string *error_message) override
    {
        result.level = request.levelConfig.level;
        result.downsampleFactor = request.levelConfig.patchMatch.downsampleFactor;
        result.depth = cv::Mat(request.referenceImage.size(), CV_32F, cv::Scalar(10.0f));
        result.confidence = cv::Mat(request.referenceImage.size(), CV_32F, cv::Scalar(0.8f));
        result.supportCount = cv::Mat(request.referenceImage.size(), CV_16U, cv::Scalar(4));
        result.uncertainty = cv::Mat(request.referenceImage.size(), CV_32F, cv::Scalar(0.2f));
        result.validMask = cv::Mat(request.referenceImage.size(), CV_8U, cv::Scalar(255));
        if (result.downsampleFactor == 1)
        {
            const cv::Rect collapsed_region(
                request.referenceImage.cols / 4,
                0,
                request.referenceImage.cols * 3 / 4,
                request.referenceImage.rows);
            result.depth(collapsed_region).setTo(0.0f);
            result.confidence(collapsed_region).setTo(0.0f);
            result.supportCount(collapsed_region).setTo(0);
            result.uncertainty(collapsed_region).setTo(0.0f);
            result.validMask(collapsed_region).setTo(0);
        }
        if (error_message)
        {
            error_message->clear();
        }
        return true;
    }
};

xjw::mvs::DepthPyramidRequest makeSyntheticPyramidRequest()
{
    xjw::mvs::DepthPyramidRequest request;
    request.referenceImage = cv::Mat(640, 800, CV_8U, cv::Scalar(128));
    request.sourceImages = {cv::Mat(640, 800, CV_8U, cv::Scalar(128))};
    request.guideImage = request.referenceImage;
    xjw::mvs::PatchMatchConfig base;
    base.downsampleFactor = 1;
    request.pyramidConfig = xjw::mvs::makeDepthPyramidConfig(base, 800, 640);
    request.zNear = 1.0f;
    request.zFar = 20.0f;
    return request;
}

} // namespace

TEST(DepthMapBackgroundTaskTest, ExceptionBoundaryReportsFailureExactlyOnce)
{
    int task_calls = 0;
    int failure_calls = 0;
    QString failure_message;

    xjw::mvs::detail::runDepthMapBackgroundTaskWithExceptionBoundary(
        [&task_calls]()
        {
            ++task_calls;
            throw std::runtime_error("inline PlaPoint failure");
        },
        [&failure_calls, &failure_message](const QString &message)
        {
            ++failure_calls;
            failure_message = message;
        });

    EXPECT_EQ(task_calls, 1);
    EXPECT_EQ(failure_calls, 1);
    EXPECT_TRUE(failure_message.contains(QStringLiteral("inline PlaPoint failure")));
}

TEST(MvsImagePreprocessorTest,
     OrbitalPhotometricNormalizationDoesNotSplitAtBrightnessThreshold)
{
    cv::Mat texture(96, 96, CV_8UC1);
    for (int row = 0; row < texture.rows; ++row)
    {
        for (int column = 0; column < texture.cols; ++column)
        {
            texture.at<std::uint8_t>(row, column) = static_cast<std::uint8_t>(
                90 + ((row * 13 + column * 17) % 80));
        }
    }
    cv::Mat dark;
    texture.convertTo(dark, CV_8UC1, 0.40, 5.0);

    const cv::Mat normalized_bright = xjw::mvs::normalizeMvsPhotometry(
        texture, xjw::mvs::MvsSceneProfile::OrbitalObject);
    const cv::Mat normalized_dark = xjw::mvs::normalizeMvsPhotometry(
        dark, xjw::mvs::MvsSceneProfile::OrbitalObject);

    ASSERT_EQ(normalized_bright.size(), texture.size());
    ASSERT_EQ(normalized_dark.size(), texture.size());
    EXPECT_GT(cv::norm(normalized_bright, texture, cv::NORM_L1), 0.0);
    EXPECT_GT(cv::norm(normalized_dark, dark, cv::NORM_L1), 0.0);
}

TEST(DepthPyramidPolicyTest, BuildsStrictThreeLevelSchedule)
{
    xjw::mvs::PatchMatchConfig base;
    base.downsampleFactor = 1;

    const xjw::mvs::DepthPyramidConfig pyramid =
        xjw::mvs::makeDepthPyramidConfig(base, 800, 640);

    ASSERT_EQ(pyramid.levels.size(), 3u);
    EXPECT_EQ(pyramid.activeLevelCount, 3);
    EXPECT_EQ(pyramid.levels[0].level, 3);
    EXPECT_EQ(pyramid.levels[0].patchMatch.downsampleFactor, 4);
    EXPECT_EQ(pyramid.levels[1].patchMatch.downsampleFactor, 2);
    EXPECT_EQ(pyramid.levels[2].patchMatch.downsampleFactor, 1);
    EXPECT_TRUE(pyramid.degradedReason.empty());
}

TEST(DepthPyramidPolicyTest, DegradesCleanlyForTinyImages)
{
    xjw::mvs::PatchMatchConfig base;
    base.downsampleFactor = 1;

    const xjw::mvs::DepthPyramidConfig pyramid =
        xjw::mvs::makeDepthPyramidConfig(base, 200, 120);

    EXPECT_GE(pyramid.activeLevelCount, 1);
    EXPECT_LT(pyramid.activeLevelCount, 3);
    EXPECT_FALSE(pyramid.degradedReason.empty());
    for (int index = 1; index < pyramid.activeLevelCount; ++index)
    {
        EXPECT_GT(pyramid.levels[index - 1].patchMatch.downsampleFactor,
                  pyramid.levels[index].patchMatch.downsampleFactor);
    }
}

TEST(DepthPyramidPolicyTest, UsesNativeWorkingResolutionForStoredIntermediateLevels)
{
    EXPECT_EQ(xjw::mvs::depthPyramidWorkingSize(6000, 4000, 16), cv::Size(375, 250));
    EXPECT_EQ(xjw::mvs::depthPyramidWorkingSize(6000, 4000, 8), cv::Size(750, 500));
    EXPECT_EQ(xjw::mvs::depthPyramidWorkingSize(6000, 4000, 0), cv::Size(6000, 4000));
}

TEST(DepthPyramidPolicyTest, NativeFinalGridFailsClosedOutsideUnrectifiedCustomScenes)
{
    EXPECT_FALSE(xjw::mvs::shouldPreserveNativeFinalDepthGrid(
        false, xjw::mvs::MvsSceneProfile::Custom, false));
    EXPECT_TRUE(xjw::mvs::shouldPreserveNativeFinalDepthGrid(
        true, xjw::mvs::MvsSceneProfile::Custom, false));
    EXPECT_FALSE(xjw::mvs::shouldPreserveNativeFinalDepthGrid(
        true, xjw::mvs::MvsSceneProfile::Custom, true));
    EXPECT_FALSE(xjw::mvs::shouldPreserveNativeFinalDepthGrid(
        true, xjw::mvs::MvsSceneProfile::OrbitalObject, false));
    EXPECT_FALSE(xjw::mvs::shouldPreserveNativeFinalDepthGrid(
        true, xjw::mvs::MvsSceneProfile::AerialTerrain, false));
    EXPECT_FALSE(xjw::mvs::shouldPreserveNativeFinalDepthGrid(
        true, xjw::mvs::MvsSceneProfile::Auto, false));
}

TEST(DepthPyramidPolicyTest, ScalesCameraToOddNativeDepthGridWithPixelCenterConvention)
{
    const double identity[9] = {1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0};
    const double center[3] = {0.0, 0.0, 0.0};
    xjw::FramePinholeCamera raster_camera = makeMvsCamera(
        3536.75872, 3533.741148, 3217.170232, 2125.808664,
        identity, center);
    raster_camera.setImageSize(xjw::CameraImageSize{6221, 4146});

    const cv::Size raster_size(6221, 4146);
    const cv::Size grid_size = xjw::mvs::depthPyramidWorkingSize(
        raster_size.width, raster_size.height, 4);
    ASSERT_EQ(grid_size, cv::Size(1555, 1036));

    const xjw::FramePinholeCamera grid_camera =
        xjw::mvs::cameraForDepthGrid(raster_camera, raster_size, grid_size);
    const double scale_x = static_cast<double>(grid_size.width) / raster_size.width;
    const double scale_y = static_cast<double>(grid_size.height) / raster_size.height;
    EXPECT_DOUBLE_EQ(grid_camera.focalX(), raster_camera.focalX() * scale_x);
    EXPECT_DOUBLE_EQ(grid_camera.focalY(), raster_camera.focalY() * scale_y);
    EXPECT_DOUBLE_EQ(
        grid_camera.principalX(),
        (raster_camera.principalX() + 0.5) * scale_x - 0.5);
    EXPECT_DOUBLE_EQ(
        grid_camera.principalY(),
        (raster_camera.principalY() + 0.5) * scale_y - 0.5);
    ASSERT_TRUE(grid_camera.imageSize().has_value());
    EXPECT_EQ(grid_camera.imageSize()->samples, grid_size.width);
    EXPECT_EQ(grid_camera.imageSize()->lines, grid_size.height);
}

TEST(DepthPyramidPolicyTest, ScalesFullRasterPixelParametersToOddNativeGrid)
{
    const cv::Size raster_size(6221, 4146);
    const cv::Size grid_size(1555, 1036);
    const xjw::mvs::DepthPixelDomainScale scale =
        xjw::mvs::depthPixelDomainScale(raster_size, grid_size);

    EXPECT_DOUBLE_EQ(scale.scaleX, 1555.0 / 6221.0);
    EXPECT_DOUBLE_EQ(scale.scaleY, 1036.0 / 4146.0);
    EXPECT_NEAR(scale.linearScale, 0.25, 1.0e-3);
    EXPECT_NEAR(scale.areaScale, 1.0 / 16.0, 5.0e-4);
    EXPECT_EQ(xjw::mvs::scaleDepthPixelRadius(1, scale), 0);
    EXPECT_EQ(xjw::mvs::scaleDepthLocalOutlierKernel(3, scale), 1);
    EXPECT_EQ(xjw::mvs::scaleDepthPixelArea(64, scale), 4);
    EXPECT_NEAR(
        xjw::mvs::scaleDepthPixelDistance(3.0f, scale), 0.75f, 0.01f);

    const auto identity_scale = xjw::mvs::depthPixelDomainScale(
        raster_size, raster_size);
    EXPECT_EQ(xjw::mvs::scaleDepthPixelRadius(1, identity_scale), 1);
    EXPECT_EQ(xjw::mvs::scaleDepthLocalOutlierKernel(3, identity_scale), 3);
    EXPECT_EQ(xjw::mvs::scaleDepthPixelArea(64, identity_scale), 64);
}

TEST(MvsSceneClassifierTest, DetectsAerialCameraLayout)
{
    const std::vector<xjw::mvs::CameraView> views = makeDownLookingGridViews(3, 3);
    const xjw::mvs::SparseCloud sparse = makePlanarSparseCloud();

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, sparse);

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::AerialTerrain);
    EXPECT_GE(classification.downLookingConsistency, 0.75f);
    EXPECT_LE(classification.planeThicknessRatio, 0.20f);
    EXPECT_FALSE(classification.reason.empty());
}

TEST(MvsSceneClassifierTest, DetectsLongAerialStripWithoutGlobalCenterConvergence)
{
    std::vector<xjw::mvs::CameraView> views;
    for (int index = -5; index <= 5; ++index)
    {
        xjw::mvs::CameraView view;
        view.imageWidth = 640;
        view.imageHeight = 480;
        view.camera.setIntrinsics(500.0, 500.0, 320.0, 240.0);
        view.camera.setPose(
            std::array<double, 9>{1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0},
            std::array<double, 3>{static_cast<double>(index * 40), 0.0, -10.0});
        view.camera.setAxisDirections(1, 1);
        view.camera.setDepthAxisFlipped(false);
        views.push_back(std::move(view));
    }

    xjw::mvs::SparseCloud sparse;
    for (int x = -200; x <= 200; x += 10)
    {
        for (int y = -5; y <= 5; ++y)
        {
            sparse.points.push_back({static_cast<float>(x),
                                     static_cast<float>(y),
                                     0.01f * static_cast<float>((x + y) % 3)});
        }
    }

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, sparse);

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::AerialTerrain);
    EXPECT_GE(classification.downLookingConsistency, 0.75f);
    EXPECT_LT(classification.opticalAxisConvergence, 0.50f);
    EXPECT_LE(classification.planeThicknessRatio, 0.20f);
}

TEST(MvsSceneClassifierTest, KeepsTwoSidedPlanarCaptureGeneral)
{
    std::vector<xjw::mvs::CameraView> views;
    for (int index = 0; index < 8; ++index)
    {
        const bool below_plane = index < 4;
        xjw::mvs::CameraView view;
        view.imageWidth = 640;
        view.imageHeight = 480;
        view.camera.setIntrinsics(500.0, 500.0, 320.0, 240.0);
        view.camera.setPose(
            below_plane
                ? std::array<double, 9>{1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0}
                : std::array<double, 9>{1.0, 0.0, 0.0,
                                        0.0, -1.0, 0.0,
                                        0.0, 0.0, -1.0},
            std::array<double, 3>{static_cast<double>((index % 4) * 2 - 3),
                                  0.0,
                                  below_plane ? -10.0 : 10.0});
        view.camera.setAxisDirections(1, 1);
        view.camera.setDepthAxisFlipped(false);
        views.push_back(std::move(view));
    }

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, makePlanarSparseCloud());

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::Custom);
    EXPECT_FALSE(classification.orbitalGatePassed);
    EXPECT_GE(classification.downLookingConsistency, 0.75f);
    EXPECT_LE(classification.planeThicknessRatio, 0.20f);
}

TEST(MvsSceneClassifierTest, KeepsNonPlanarTerrainCaptureGeneral)
{
    const std::vector<xjw::mvs::CameraView> views = makeDownLookingGridViews(3, 3);
    xjw::mvs::SparseCloud sparse;
    for (int z = -4; z <= 4; ++z)
    {
        for (int y = -4; y <= 4; ++y)
        {
            for (int x = -4; x <= 4; ++x)
            {
                sparse.points.push_back({static_cast<float>(x),
                                         static_cast<float>(y),
                                         static_cast<float>(z)});
            }
        }
    }

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, sparse);

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::Custom);
    EXPECT_FALSE(classification.orbitalGatePassed);
    EXPECT_GT(classification.planeThicknessRatio, 0.20f);
}

TEST(MvsSceneClassifierTest, DetectsConvergentPlanarObjectRing)
{
    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(
            makeOrbitalRingViews(12), makeVolumetricSparseCloud());

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::OrbitalObject);
    EXPECT_TRUE(classification.orbitalGatePassed);
    EXPECT_EQ(classification.distinctCameraCenterCount, 12);
    EXPECT_GE(classification.cameraCenterInPlaneBalance, 0.95f);
    EXPECT_LE(classification.cameraCenterNonPlanarity, 1.0e-5f);
    EXPECT_LE(classification.orbitalProjectedRadiusMadRatio, 1.0e-5f);
    EXPECT_LE(classification.orbitalProjectedCenterOffsetRatio, 1.0e-5f);
    EXPECT_NEAR(classification.orbitalMaximumAngularGapDegrees, 30.0f, 1.0e-3f);
    EXPECT_LE(classification.orbitalOpticalAxisMedianErrorDegrees, 1.0e-4f);
    EXPECT_LE(classification.orbitalOpticalAxisP90ErrorDegrees, 1.0e-4f);
}

TEST(MvsSceneClassifierTest, RejectsOfficeStyleForwardFacingEllipse)
{
    std::vector<xjw::mvs::CameraView> views;
    for (int index = 0; index < 26; ++index)
    {
        const double angle = 2.0 * CV_PI * static_cast<double>(index) / 26.0;
        const cv::Vec3d center(
            10.0 * std::cos(angle), 6.0 * std::sin(angle), -5.0);
        const cv::Vec3d direction_to_cloud = -center / cv::norm(center);
        const cv::Vec3d tangent_raw(
            -direction_to_cloud[1], direction_to_cloud[0], 0.0);
        const cv::Vec3d tangent = tangent_raw / cv::norm(tangent_raw);
        const cv::Vec3d forward = index < 12
            ? direction_to_cloud
            : 0.30 * direction_to_cloud + std::sqrt(1.0 - 0.30 * 0.30) * tangent;
        views.push_back(makeViewWithForward(center, forward));
    }

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, makeVolumetricSparseCloud());

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::Custom);
    EXPECT_FALSE(classification.orbitalGatePassed);
    EXPECT_GE(classification.cameraCenterInPlaneBalance, 0.25f);
    EXPECT_LE(classification.cameraCenterNonPlanarity, 1.0e-5f);
    EXPECT_LE(classification.orbitalProjectedRadiusMadRatio, 0.30f);
    EXPECT_LE(classification.orbitalMaximumAngularGapDegrees, 120.0f);
    EXPECT_GT(classification.orbitalOpticalAxisMedianErrorDegrees, 30.0f);
    EXPECT_EQ(classification.convergentCameraCount, 12);
    EXPECT_NEAR(classification.opticalAxisConvergence, 0.6230769f, 1.0e-5f);
    EXPECT_NE(classification.reason.find("median-axis-not-convergent"),
              std::string::npos);
}

TEST(MvsSceneClassifierTest, RejectsCollinearCameraCenters)
{
    std::vector<xjw::mvs::CameraView> views;
    for (int index = -4; index <= 4; ++index)
    {
        views.push_back(makeLookAtView(
            cv::Vec3d(static_cast<double>(index * 2), 0.0, -8.0),
            cv::Vec3d(0.0, 0.0, 0.0)));
    }

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, makeVolumetricSparseCloud());

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::Custom);
    EXPECT_FALSE(classification.orbitalGatePassed);
    EXPECT_LT(classification.cameraCenterInPlaneBalance, 0.25f);
    EXPECT_NE(classification.reason.find("camera-centers-collinear"),
              std::string::npos);
}

TEST(MvsSceneClassifierTest, RejectsCoincidentCameraCenters)
{
    std::vector<xjw::mvs::CameraView> views;
    for (int index = 0; index < 8; ++index)
    {
        views.push_back(makeLookAtView(
            cv::Vec3d(0.0, 0.0, -8.0), cv::Vec3d(0.0, 0.0, 0.0)));
    }

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, makeVolumetricSparseCloud());

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::Custom);
    EXPECT_FALSE(classification.orbitalGatePassed);
    EXPECT_EQ(classification.distinctCameraCenterCount, 1);
    EXPECT_NE(classification.reason.find("fewer-than-3-distinct-cameras"),
              std::string::npos);
}

TEST(MvsSceneClassifierTest, RejectsNonPlanarConvergentCameraGrid)
{
    std::vector<xjw::mvs::CameraView> views;
    for (const double z : {-8.0, 8.0})
    {
        for (const double y : {-8.0, 8.0})
        {
            for (const double x : {-8.0, 8.0})
            {
                views.push_back(makeLookAtView(
                    cv::Vec3d(x, y, z), cv::Vec3d(0.0, 0.0, 0.0)));
            }
        }
    }

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, makeVolumetricSparseCloud());

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::Custom);
    EXPECT_FALSE(classification.orbitalGatePassed);
    EXPECT_GT(classification.cameraCenterNonPlanarity, 0.20f);
    EXPECT_NE(classification.reason.find("camera-centers-nonplanar"),
              std::string::npos);
}

TEST(MvsSceneClassifierTest, UsesGeneralProfileForInsufficientInputs)
{
    std::vector<xjw::mvs::CameraView> views = makeOrbitalRingViews(2);

    const xjw::mvs::MvsSceneClassification classification =
        xjw::mvs::classifyMvsScene(views, makeVolumetricSparseCloud());

    EXPECT_EQ(classification.profile, xjw::mvs::MvsSceneProfile::Custom);
    EXPECT_FALSE(classification.orbitalGatePassed);
}

TEST(MvsSceneClassifierTest, IsDeterministicAcrossInputOrdering)
{
    std::vector<xjw::mvs::CameraView> views = makeOrbitalRingViews(12);
    xjw::mvs::SparseCloud sparse = makeVolumetricSparseCloud();
    const xjw::mvs::MvsSceneClassification forward =
        xjw::mvs::classifyMvsScene(views, sparse);

    std::reverse(views.begin(), views.end());
    std::reverse(sparse.points.begin(), sparse.points.end());
    const xjw::mvs::MvsSceneClassification reversed =
        xjw::mvs::classifyMvsScene(views, sparse);

    EXPECT_EQ(reversed.profile, forward.profile);
    EXPECT_EQ(reversed.validCameraCount, forward.validCameraCount);
    EXPECT_EQ(reversed.distinctCameraCenterCount, forward.distinctCameraCenterCount);
    EXPECT_EQ(reversed.convergentCameraCount, forward.convergentCameraCount);
    EXPECT_FLOAT_EQ(reversed.cameraCenterLinearity, forward.cameraCenterLinearity);
    EXPECT_FLOAT_EQ(reversed.cameraCenterInPlaneBalance,
                    forward.cameraCenterInPlaneBalance);
    EXPECT_FLOAT_EQ(reversed.cameraCenterNonPlanarity,
                    forward.cameraCenterNonPlanarity);
    EXPECT_FLOAT_EQ(reversed.opticalAxisConvergence, forward.opticalAxisConvergence);
    EXPECT_FLOAT_EQ(reversed.orbitalOpticalAxisMedianErrorDegrees,
                    forward.orbitalOpticalAxisMedianErrorDegrees);
    EXPECT_FLOAT_EQ(reversed.orbitalOpticalAxisP90ErrorDegrees,
                    forward.orbitalOpticalAxisP90ErrorDegrees);
    EXPECT_FLOAT_EQ(reversed.orbitalProjectedRadiusMadRatio,
                    forward.orbitalProjectedRadiusMadRatio);
    EXPECT_FLOAT_EQ(reversed.orbitalProjectedCenterOffsetRatio,
                    forward.orbitalProjectedCenterOffsetRatio);
    EXPECT_FLOAT_EQ(reversed.orbitalMaximumAngularGapDegrees,
                    forward.orbitalMaximumAngularGapDegrees);
    EXPECT_FLOAT_EQ(reversed.planeThicknessRatio, forward.planeThicknessRatio);
    EXPECT_FLOAT_EQ(reversed.downLookingConsistency, forward.downLookingConsistency);
    EXPECT_EQ(reversed.orbitalGatePassed, forward.orbitalGatePassed);
    EXPECT_EQ(reversed.reason, forward.reason);
}

TEST(MvsSceneClassifierTest, RecommendsSceneAwareSourceViewPool)
{
    using xjw::mvs::MvsSceneProfile;

    EXPECT_EQ(xjw::mvs::recommendedMvsDepthFilterMode(
                  MvsSceneProfile::OrbitalObject),
              xjw::mvs::DepthFilterMode::Mild);
    EXPECT_EQ(xjw::mvs::recommendedMvsDepthFilterMode(
                  MvsSceneProfile::AerialTerrain),
              xjw::mvs::DepthFilterMode::Moderate);
    EXPECT_EQ(xjw::mvs::recommendedMvsDepthFilterMode(
                  MvsSceneProfile::Custom),
              xjw::mvs::DepthFilterMode::Moderate);
    using xjw::mvs::recommendedMvsSourceViewCount;

    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::AerialTerrain, 2, 3, 9), 8);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::AerialTerrain, 4, 3, 9), 6);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::OrbitalObject, 2, 3, 16), 6);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::OrbitalObject, 2, 3, 12), 4);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::OrbitalObject, 2, 7, 12), 4);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::OrbitalObject, 1, 8, 12), 4);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::OrbitalObject, 4, 3, 16), 4);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::OrbitalObject, 2, 6, 16), 6);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::OrbitalObject, 2, 8, 16), 6);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::AerialTerrain, 2, 3, 5), 4);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::AerialTerrain, 2, 10, 16), 10);
    EXPECT_EQ(recommendedMvsSourceViewCount(MvsSceneProfile::Custom, 2, 7, 12), 7);
}

TEST(MvsDepthConfidenceThresholdTest,
     PreservesWeakEvidenceForSparseOrbitalRingSectors)
{
    using xjw::mvs::DepthConfidenceThresholds;
    using xjw::mvs::DepthFilterMode;
    using xjw::mvs::MvsSceneProfile;
    using xjw::mvs::depthConfidenceThresholds;

    const DepthConfidenceThresholds sparse_orbital =
        depthConfidenceThresholds(
            MvsSceneProfile::OrbitalObject,
            DepthFilterMode::Mild,
            3,
            0.72f,
            0.75f);
    EXPECT_FLOAT_EQ(sparse_orbital.patchMatch, 0.50f);
    EXPECT_FLOAT_EQ(sparse_orbital.fusion, 0.60f);

    const DepthConfidenceThresholds dense_orbital =
        depthConfidenceThresholds(
            MvsSceneProfile::OrbitalObject,
            DepthFilterMode::Mild,
            4,
            0.72f,
            0.75f);
    EXPECT_FLOAT_EQ(dense_orbital.patchMatch, 0.50f);
    EXPECT_FLOAT_EQ(dense_orbital.fusion, 0.60f);

    const DepthConfidenceThresholds terrain =
        depthConfidenceThresholds(
            MvsSceneProfile::AerialTerrain,
            DepthFilterMode::Mild,
            3,
            0.72f,
            0.75f);
    EXPECT_FLOAT_EQ(terrain.patchMatch, 0.72f);
    EXPECT_FLOAT_EQ(terrain.fusion, 0.75f);
}

TEST(MvsSceneClassifierTest, AllowsWiderObjectRingBaselines)
{
    using xjw::mvs::MvsSceneProfile;
    using xjw::mvs::adaptiveMvsSourceMaximumAngleDeg;
    using xjw::mvs::recommendedMvsSourceMaximumAngleDeg;

    EXPECT_FLOAT_EQ(recommendedMvsSourceMaximumAngleDeg(MvsSceneProfile::OrbitalObject, 4), 47.0f);
    EXPECT_FLOAT_EQ(recommendedMvsSourceMaximumAngleDeg(MvsSceneProfile::OrbitalObject, 6), 70.0f);
    EXPECT_FLOAT_EQ(recommendedMvsSourceMaximumAngleDeg(MvsSceneProfile::AerialTerrain), 35.0f);
    EXPECT_FLOAT_EQ(recommendedMvsSourceMaximumAngleDeg(MvsSceneProfile::Custom), 35.0f);
    EXPECT_NEAR(
        adaptiveMvsSourceMaximumAngleDeg(
            MvsSceneProfile::OrbitalObject,
            4,
            {28.0f, 28.5f, 56.0f, 56.5f, 84.0f}),
        59.325f,
        1.0e-4f);
    EXPECT_FLOAT_EQ(
        adaptiveMvsSourceMaximumAngleDeg(
            MvsSceneProfile::OrbitalObject,
            4,
            {28.0f, 47.0f, 57.0f, 75.0f}),
        70.0f);
    EXPECT_FLOAT_EQ(
        adaptiveMvsSourceMaximumAngleDeg(
            MvsSceneProfile::AerialTerrain,
            4,
            {12.0f, 24.0f, 40.0f, 55.0f}),
        35.0f);
    EXPECT_NEAR(
        adaptiveMvsSourceMaximumAngleDeg(
            MvsSceneProfile::OrbitalObject,
            6,
            {7.8f, 8.1f, 8.4f, 15.6f, 16.0f, 16.4f,
             24.0f, 24.4f, 24.8f, 32.0f, 32.4f, 32.8f,
             40.0f, 40.4f, 40.8f, 48.0f, 48.4f, 48.8f,
             56.0f, 56.4f, 56.8f, 64.0f, 64.4f, 64.8f}),
        20.5f,
        1.0e-4f);
}

TEST(MvsSceneClassifierTest, SourceAngleExperimentCapOnlyTightensSceneMaximum)
{
    using xjw::mvs::constrainMvsSourceMaximumAngleDeg;

    EXPECT_FLOAT_EQ(constrainMvsSourceMaximumAngleDeg(35.0f, 0.0f),
                    35.0f);
    EXPECT_FLOAT_EQ(constrainMvsSourceMaximumAngleDeg(35.0f, 45.0f),
                    35.0f);
    EXPECT_FLOAT_EQ(constrainMvsSourceMaximumAngleDeg(35.0f, 25.0f),
                    25.0f);
    EXPECT_FLOAT_EQ(constrainMvsSourceMaximumAngleDeg(
                        35.0f,
                        std::numeric_limits<float>::quiet_NaN()),
                    35.0f);

    xjw::mvs::DepthGenConfig default_config;
    EXPECT_FLOAT_EQ(default_config.sourceMaximumAngleDegCap, 0.0f);
    EXPECT_FALSE(default_config.evaluateCompleteVisibilityCandidatePool);
    EXPECT_FLOAT_EQ(default_config.sourceAngleSoftRankingStrength, 0.0f);
    const QString default_hash =
        xjw::mvs::makeMvsDepthConfigHash(default_config, 26);
    xjw::mvs::DepthGenConfig explicit_default_config = default_config;
    explicit_default_config.evaluateCompleteVisibilityCandidatePool = false;
    explicit_default_config.sourceAngleSoftRankingStrength = 0.0f;
    EXPECT_EQ(default_hash,
              xjw::mvs::makeMvsDepthConfigHash(
                  explicit_default_config, 26));

    xjw::mvs::DepthGenConfig capped_config = default_config;
    capped_config.sourceMaximumAngleDegCap = 25.0f;
    EXPECT_NE(default_hash,
              xjw::mvs::makeMvsDepthConfigHash(capped_config, 26));

    xjw::mvs::DepthGenConfig complete_pool_config = default_config;
    complete_pool_config.evaluateCompleteVisibilityCandidatePool = true;
    EXPECT_NE(default_hash,
              xjw::mvs::makeMvsDepthConfigHash(
                  complete_pool_config, 26));
    xjw::mvs::DepthGenConfig soft_rank_config = complete_pool_config;
    soft_rank_config.sourceAngleSoftRankingStrength = 1.0f;
    EXPECT_NE(
        xjw::mvs::makeMvsDepthConfigHash(complete_pool_config, 26),
        xjw::mvs::makeMvsDepthConfigHash(soft_rank_config, 26));
}

TEST(DepthMapGeneratorTest,
     SourceAngleCapShortfallSafetyIsFailClosedAndIdempotent)
{
    xjw::mvs::DepthFrameResult result;
    result.success = true;
    result.sourceAngleCapEnabled = true;
    result.requestedSourceViewCount = 4;
    result.sourceViewShortfall = 1;
    result.qualityDecision.acceptance =
        xjw::mvs::DepthFrameAcceptance::Accepted;

    xjw::mvs::detail::applySourceAngleCapShortfallSafety(result);
    xjw::mvs::detail::applySourceAngleCapShortfallSafety(result);

    EXPECT_EQ(result.qualityDecision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_FALSE(result.eligibleForFusion());
    EXPECT_EQ(std::count(
                  result.qualityDecision.reasons.cbegin(),
                  result.qualityDecision.reasons.cend(),
                  "source_angle_cap_source_shortfall"),
              1);

    result.qualityDecision.acceptance =
        xjw::mvs::DepthFrameAcceptance::Rejected;
    result.qualityDecision.reasons.clear();
    xjw::mvs::detail::applySourceAngleCapShortfallSafety(result);
    EXPECT_EQ(result.qualityDecision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_TRUE(result.qualityDecision.reasons.empty());

    xjw::mvs::DepthFrameResult uncapped;
    uncapped.success = true;
    uncapped.sourceViewShortfall = 2;
    uncapped.qualityDecision.acceptance =
        xjw::mvs::DepthFrameAcceptance::Accepted;
    xjw::mvs::detail::applySourceAngleCapShortfallSafety(uncapped);
    EXPECT_EQ(uncapped.qualityDecision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Accepted);
    EXPECT_TRUE(uncapped.qualityDecision.reasons.empty());
}

TEST(DepthPyramidPropagationTest, PreservesDepthStepAndExpandsLowConfidenceRadius)
{
    xjw::mvs::DepthLevelResult parent;
    parent.level = 3;
    parent.downsampleFactor = 2;
    parent.depth = cv::Mat(6, 8, CV_32F);
    parent.confidence = cv::Mat(6, 8, CV_32F);
    parent.uncertainty = cv::Mat(6, 8, CV_32F, cv::Scalar(0.5f));
    parent.validMask = cv::Mat(6, 8, CV_8U, cv::Scalar(255));
    for (int row = 0; row < parent.depth.rows; ++row)
    {
        for (int column = 0; column < parent.depth.cols; ++column)
        {
            const bool left = column < parent.depth.cols / 2;
            parent.depth.at<float>(row, column) = left ? 10.0f : 20.0f;
            parent.confidence.at<float>(row, column) = left ? 0.9f : 0.2f;
        }
    }

    cv::Mat guide(12, 16, CV_8U);
    guide.colRange(0, 8).setTo(cv::Scalar(24));
    guide.colRange(8, 16).setTo(cv::Scalar(224));

    const xjw::mvs::DepthSearchPrior prior =
        xjw::mvs::propagateDepthPrior(parent, guide, guide.size());

    ASSERT_EQ(prior.center.size(), guide.size());
    ASSERT_EQ(prior.radius.size(), guide.size());
    EXPECT_LT(prior.center.at<float>(6, 7), 12.0f);
    EXPECT_GT(prior.center.at<float>(6, 9), 18.0f);
    EXPECT_GT(prior.radius.at<float>(6, 12), prior.radius.at<float>(6, 3));
    EXPECT_EQ(prior.validMask.at<uint8_t>(6, 7), 255);
}

TEST(DepthPyramidPropagationTest, FactorizedSpatialWeightMatchesCombinedExponential)
{
    constexpr float spatial_sigma = 1.2f;
    constexpr float denominator = 2.0f * spatial_sigma * spatial_sigma;
    constexpr int parent_extent = 7;
    constexpr int target_extent = 13;
    const float scale = static_cast<float>(parent_extent) / target_extent;

    for (int target_row = 0; target_row < target_extent; ++target_row)
    {
        const float parent_y = (target_row + 0.5f) * scale - 0.5f;
        const int base_row = static_cast<int>(std::floor(parent_y));
        for (int target_column = 0; target_column < target_extent; ++target_column)
        {
            const float parent_x = (target_column + 0.5f) * scale - 0.5f;
            const int base_column = static_cast<int>(std::floor(parent_x));
            for (int delta_row = -1; delta_row <= 2; ++delta_row)
            {
                for (int delta_column = -1; delta_column <= 2; ++delta_column)
                {
                    const float offset_x = base_column + delta_column - parent_x;
                    const float offset_y = base_row + delta_row - parent_y;
                    const float combined = std::exp(
                        -(offset_x * offset_x + offset_y * offset_y) / denominator);
                    const float factorized =
                        std::exp(-(offset_x * offset_x) / denominator) *
                        std::exp(-(offset_y * offset_y) / denominator);
                    EXPECT_NEAR(factorized, combined, 5.0e-7f);
                }
            }
        }
    }
}

TEST(DepthPyramidPropagationTest, DoesNotInventIntermediateDepthAtLowContrastStep)
{
    xjw::mvs::DepthLevelResult parent;
    parent.depth = cv::Mat(4, 4, CV_32F);
    parent.confidence = cv::Mat(4, 4, CV_32F, cv::Scalar(0.8f));
    parent.validMask = cv::Mat(4, 4, CV_8U, cv::Scalar(255));
    parent.depth.colRange(0, 2).setTo(cv::Scalar(10.0f));
    parent.depth.colRange(2, 4).setTo(cv::Scalar(20.0f));

    const cv::Mat low_contrast_guide(8, 8, CV_8U, cv::Scalar(100));
    const xjw::mvs::DepthSearchPrior prior =
        xjw::mvs::propagateDepthPrior(parent, low_contrast_guide, low_contrast_guide.size());

    const float boundary_depth = prior.center.at<float>(4, 3);
    EXPECT_TRUE(std::fabs(boundary_depth - 10.0f) < 1.0e-5f ||
                std::fabs(boundary_depth - 20.0f) < 1.0e-5f)
        << "A depth prior must select one side of a discontinuity, not average both surfaces.";
}

TEST(DepthPyramidPropagationTest, ExcludesNonFiniteParentDepthFromPreparedResources)
{
    xjw::mvs::DepthLevelResult parent;
    parent.depth = cv::Mat(3, 3, CV_32F,
                           cv::Scalar(std::numeric_limits<float>::infinity()));
    parent.depth.at<float>(1, 1) = 7.0f;
    parent.confidence = cv::Mat(3, 3, CV_32F, cv::Scalar(0.8f));
    parent.validMask = cv::Mat(3, 3, CV_8U, cv::Scalar(255));

    const cv::Mat guide(6, 6, CV_8U, cv::Scalar(100));
    const xjw::mvs::DepthSearchPrior prior =
        xjw::mvs::propagateDepthPrior(parent, guide, guide.size());

    ASSERT_EQ(cv::countNonZero(prior.validMask), guide.rows * guide.cols);
    for (int row = 0; row < prior.center.rows; ++row)
    {
        for (int column = 0; column < prior.center.cols; ++column)
        {
            EXPECT_FLOAT_EQ(prior.center.at<float>(row, column), 7.0f);
            EXPECT_TRUE(std::isfinite(prior.radius.at<float>(row, column)));
        }
    }
}

TEST(DepthPyramidEstimatorTest, RunsCoarseMiddleFineAndReturnsFinalLevel)
{
    RecordingPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(makeSyntheticPyramidRequest());

    EXPECT_EQ(backend.downsampleCalls(), (std::vector<int>{4, 2, 1}));
    EXPECT_EQ(backend.nativeOutputCalls(), (std::vector<bool>{true, true, false}));
    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.finalLevel.level, 1);
    EXPECT_EQ(result.levelSummaries.size(), 3u);
    EXPECT_TRUE(result.intermediateLevels.empty());
    EXPECT_EQ(result.levelSummaries[0].validPixelCount, 200 * 160);
    EXPECT_EQ(result.levelSummaries[1].validPixelCount, 400 * 320);
    EXPECT_EQ(result.levelSummaries[2].validPixelCount, 800 * 640);
    EXPECT_FLOAT_EQ(result.levelSummaries[0].meanSupportViews, 4.0f);
    EXPECT_FLOAT_EQ(result.levelSummaries[1].meanSupportViews, 4.0f);
    EXPECT_FLOAT_EQ(result.levelSummaries[2].meanSupportViews, 4.0f);
    EXPECT_FLOAT_EQ(result.levelSummaries[0].depthDiscontinuityRatio, 0.0f);
    EXPECT_FLOAT_EQ(result.levelSummaries[1].depthDiscontinuityRatio, 0.0f);
    EXPECT_FLOAT_EQ(result.levelSummaries[2].depthDiscontinuityRatio, 0.0f);
}

TEST(DepthPyramidEstimatorTest, ReturnsOddNativeFinalGridOnlyWhenRequested)
{
    RecordingPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);
    xjw::mvs::DepthPyramidRequest request;
    request.referenceImage = cv::Mat(667, 1001, CV_8U, cv::Scalar(128));
    request.sourceImages = {request.referenceImage};
    request.guideImage = request.referenceImage;
    xjw::mvs::PatchMatchConfig base;
    base.downsampleFactor = 4;
    request.pyramidConfig = xjw::mvs::makeDepthPyramidConfig(base, 1001, 667);
    request.pyramidConfig.returnNativeFinalResolution = true;
    request.zNear = 1.0f;
    request.zFar = 20.0f;

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(request);

    ASSERT_TRUE(result.success) << result.errorMessage;
    ASSERT_EQ(request.pyramidConfig.activeLevelCount, 2);
    EXPECT_EQ(backend.downsampleCalls(), (std::vector<int>{4, 2}));
    EXPECT_EQ(backend.nativeOutputCalls(), (std::vector<bool>{true, true}));
    EXPECT_EQ(result.finalLevel.level, 1);
    EXPECT_EQ(result.finalLevel.downsampleFactor, 2);
    EXPECT_EQ(result.finalLevel.depth.size(), cv::Size(500, 333));
    EXPECT_EQ(result.finalLevel.confidence.size(), cv::Size(500, 333));
    EXPECT_EQ(result.finalLevel.supportCount.size(), cv::Size(500, 333));
    EXPECT_EQ(result.finalLevel.uncertainty.size(), cv::Size(500, 333));
    EXPECT_EQ(result.finalLevel.validMask.size(), cv::Size(500, 333));
    ASSERT_EQ(result.levelSummaries.size(), 2U);
    EXPECT_EQ(result.levelSummaries.back().validPixelCount, 500 * 333);
}

TEST(DepthPyramidEstimatorTest, RetainsCoarseAndMiddleLevelsOnlyWhenRequested)
{
    RecordingPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);
    xjw::mvs::DepthPyramidRequest request = makeSyntheticPyramidRequest();
    request.pyramidConfig.saveIntermediateLevels = true;

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(request);

    ASSERT_TRUE(result.success) << result.errorMessage;
    ASSERT_EQ(result.intermediateLevels.size(), 2u);
    EXPECT_EQ(result.intermediateLevels[0].level, 3);
    EXPECT_EQ(result.intermediateLevels[1].level, 2);
    EXPECT_EQ(result.intermediateLevels[0].depth.size(), cv::Size(200, 160));
    EXPECT_EQ(result.intermediateLevels[1].depth.size(), cv::Size(400, 320));
    EXPECT_EQ(result.finalLevel.level, 1);
}

TEST(DepthPyramidEstimatorTest, FirstLevelGateRejectionAbortsWithoutParentFallback)
{
    RecordingPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);
    xjw::mvs::DepthPyramidRequest request = makeSyntheticPyramidRequest();
    int gate_calls = 0;
    request.firstLevelCompletionGate =
        [&gate_calls](const xjw::mvs::DepthLevelSummary &summary,
                      std::string *error_message)
    {
        ++gate_calls;
        EXPECT_EQ(summary.level, 3);
        EXPECT_TRUE(summary.success);
        if (error_message)
        {
            *error_message = "calibration_probe_unprofitable";
        }
        return false;
    };

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(request);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(gate_calls, 1);
    EXPECT_EQ(backend.downsampleCalls(), (std::vector<int>{4}));
    EXPECT_TRUE(result.finalLevel.depth.empty());
    EXPECT_TRUE(result.intermediateLevels.empty());
    ASSERT_EQ(result.levelSummaries.size(), 1U);
    EXPECT_FALSE(result.levelSummaries.front().success);
    EXPECT_EQ(result.levelSummaries.front().errorMessage,
              "calibration_probe_unprofitable");
    EXPECT_EQ(result.errorMessage, "calibration_probe_unprofitable");
}

TEST(DepthPyramidEstimatorTest, FallsBackToParentWhenFineCoverageCollapses)
{
    CollapsingFinePatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(makeSyntheticPyramidRequest());

    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.finalLevel.level, 2);
    EXPECT_EQ(result.finalLevel.depth.size(), cv::Size(800, 640));
    EXPECT_EQ(result.finalLevel.confidence.size(), cv::Size(800, 640));
    EXPECT_EQ(result.finalLevel.supportCount.size(), cv::Size(800, 640));
    EXPECT_EQ(result.finalLevel.uncertainty.size(), cv::Size(800, 640));
    EXPECT_EQ(result.finalLevel.validMask.size(), cv::Size(800, 640));
    ASSERT_EQ(result.levelSummaries.size(), 3U);
    EXPECT_FALSE(result.levelSummaries.back().success);
    EXPECT_NE(result.errorMessage.find("coverage regression"), std::string::npos);
}

TEST(DepthPyramidEstimatorTest, NativeFinalModeKeepsOddParentGridOnFineFallback)
{
    CollapsingFinePatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);
    xjw::mvs::DepthPyramidRequest request;
    request.referenceImage = cv::Mat(641, 801, CV_8U, cv::Scalar(128));
    request.sourceImages = {request.referenceImage};
    request.guideImage = request.referenceImage;
    xjw::mvs::PatchMatchConfig base;
    base.downsampleFactor = 1;
    request.pyramidConfig = xjw::mvs::makeDepthPyramidConfig(base, 801, 641);
    request.pyramidConfig.returnNativeFinalResolution = true;
    request.zNear = 1.0f;
    request.zFar = 20.0f;

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(request);

    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.finalLevel.level, 2);
    EXPECT_EQ(result.finalLevel.downsampleFactor, 2);
    EXPECT_EQ(result.finalLevel.depth.size(), cv::Size(400, 320));
    EXPECT_EQ(result.finalLevel.confidence.size(), cv::Size(400, 320));
    EXPECT_EQ(result.finalLevel.supportCount.size(), cv::Size(400, 320));
    EXPECT_EQ(result.finalLevel.uncertainty.size(), cv::Size(400, 320));
    EXPECT_EQ(result.finalLevel.validMask.size(), cv::Size(400, 320));
    ASSERT_EQ(result.levelSummaries.size(), 3U);
    EXPECT_FALSE(result.levelSummaries.back().success);
    EXPECT_NE(result.errorMessage.find("coverage regression"), std::string::npos);
}

TEST(DepthMapGeneratorMaskTest, ConvertsProjectExclusionMaskToValidRegionMask)
{
    cv::Mat project_mask(4, 4, CV_8U, cv::Scalar(255));
    project_mask(cv::Rect(1, 1, 2, 2)).setTo(cv::Scalar(0));
    project_mask.at<uint8_t>(2, 2) = 255;

    const cv::Mat valid_mask = xjw::mvs::DepthMapGenerator::projectMaskToValidMask(
        project_mask,
        cv::Size(8, 8));

    ASSERT_EQ(valid_mask.type(), CV_8U);
    ASSERT_EQ(valid_mask.size(), cv::Size(8, 8));
    EXPECT_EQ(valid_mask.at<uint8_t>(0, 0), 0);
    EXPECT_EQ(valid_mask.at<uint8_t>(2, 2), 255);
    EXPECT_EQ(valid_mask.at<uint8_t>(5, 5), 0);
    EXPECT_EQ(valid_mask.at<uint8_t>(7, 7), 0);
}

TEST(DepthPyramidEstimatorTest, AppliesReferenceValidMaskAtEveryLevelAndToEveryArtifact)
{
    RecordingPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);
    xjw::mvs::DepthPyramidRequest request = makeSyntheticPyramidRequest();
    request.referenceValidMask = cv::Mat(request.referenceImage.size(), CV_8U, cv::Scalar(0));
    request.referenceValidMask(cv::Rect(0, 0, 400, 640)).setTo(cv::Scalar(255));

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(request);

    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(backend.validMaskSizes(),
              (std::vector<cv::Size>{cv::Size(200, 160), cv::Size(400, 320), cv::Size(800, 640)}));
    EXPECT_EQ(backend.validMaskValues(), (std::vector<int>{16000, 64000, 256000}));

    const int masked_column = result.finalLevel.depth.cols - 1;
    const int row = result.finalLevel.depth.rows / 2;
    EXPECT_FLOAT_EQ(result.finalLevel.depth.at<float>(row, masked_column), 0.0f);
    EXPECT_FLOAT_EQ(result.finalLevel.confidence.at<float>(row, masked_column), 0.0f);
    EXPECT_EQ(result.finalLevel.supportCount.at<uint16_t>(row, masked_column), 0);
    EXPECT_FLOAT_EQ(result.finalLevel.uncertainty.at<float>(row, masked_column), 0.0f);
    EXPECT_EQ(result.finalLevel.validMask.at<uint8_t>(row, masked_column), 0);
}

TEST(EpipolarRectifierTest, UnrectifiesRightReferenceWithRightHomography)
{
    xjw::mvs::EpipolarRectifier::RectifiedPair pair;
    pair.H1 = cv::Mat::eye(3, 3, CV_64F);
    pair.H2 = cv::Mat::eye(3, 3, CV_64F);
    pair.H2.at<double>(0, 2) = 1.0;
    pair.H1inv = pair.H1.inv();
    pair.H2inv = pair.H2.inv();
    pair.refIsRight = true;
    xjw::FramePinholeCamera reference_camera;
    reference_camera.setIntrinsics(1.0, 1.0, 0.0, 0.0);
    reference_camera.setPose({1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0},
                             {0.0, 0.0, 0.0});
    pair.rectCamRight = reference_camera;

    cv::Mat rectified_depth(3, 4, CV_32F, cv::Scalar(0.0f));
    rectified_depth.at<float>(1, 2) = 7.0f;

    const cv::Mat depth = xjw::mvs::EpipolarRectifier::unrectifyDepth(
        rectified_depth,
        pair,
        reference_camera,
        4,
        3);

    cv::Mat rectified_support(3, 4, CV_16U, cv::Scalar(0));
    rectified_support.at<std::uint16_t>(1, 2) = 5;
    const cv::Mat support = xjw::mvs::EpipolarRectifier::unrectifyNearest(
        rectified_support,
        pair,
        4,
        3);

    EXPECT_FLOAT_EQ(depth.at<float>(1, 1), 7.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(1, 2), 0.0f);
    EXPECT_EQ(support.type(), CV_16U);
    EXPECT_EQ(support.at<std::uint16_t>(1, 1), 5);
    EXPECT_EQ(support.at<std::uint16_t>(1, 2), 0);
}

TEST(MvsPipelineTest, SparseCloudPreprocessorReadsBinaryPly)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mvs_binary_sparse_test";
    fs::create_directories(root);
    const fs::path plyPath = root / "sparse_binary.ply";

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points(0, 0) = 0.0f; points(0, 1) = 0.0f; points(0, 2) = 0.0f;
    points(1, 0) = 1.0f; points(1, 1) = 0.0f; points(1, 2) = 0.0f;
    points(2, 0) = 0.0f; points(2, 1) = 1.0f; points(2, 2) = 0.0f;
    points(3, 0) = 0.0f; points(3, 1) = 0.0f; points(3, 2) = 1.0f;
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);

    xjw::mvs::SparseCloudPreprocessor preprocessor(plapoint::ProcessingDevice::CPU);
    xjw::mvs::PreprocessResult result;
    std::string error;
    ASSERT_TRUE(preprocessor.run(plyPath.string(), {}, result, &error)) << error;
    EXPECT_EQ(result.rawCount, 4);
    EXPECT_EQ(result.filteredCount, 4);
    EXPECT_EQ(result.cloud.points.size(), 4u);
}

// ---------------------------------------------------------------------------
// Test 1: CPU PatchMatch 输出格式与有效像素
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, PatchMatchCpuOutputValid)
{
    constexpr int W = 80, H = 60;
    constexpr double FOCAL = 64.0, BASELINE = 1.0, DISP = 6;

    cv::Mat ref = makeSyntheticGray(W, H);
    cv::Mat src = makeShifted(ref, DISP);
    cv::GaussianBlur(src, src, cv::Size(3,3), 0.0);

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0}, C1[3]={BASELINE,0,0};
    auto refCam = makeMvsCamera(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);
    auto srcCam = makeMvsCamera(FOCAL, FOCAL, W*0.5, H*0.5, I, C1);

    xjw::mvs::PatchMatchConfig cfg;
    cfg.backend           = xjw::mvs::PatchMatchBackend::Cpu;
    cfg.downsampleFactor  = 2;
    cfg.patchHalf         = 2;
    cfg.confidenceThresh  = 0.10f;
    cfg.doMedianBlur      = true;
    cfg.medianKernelSize  = 3;
    cfg.doBilateralFilter = false;

    cv::Mat depth, conf;
    std::string err;
    bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
        ref, {src}, refCam, {srcCam}, 4.0f, 20.0f, cfg, depth, &conf, &err);

    ASSERT_TRUE(ok) << err;
    ASSERT_EQ(depth.type(), CV_32F);
    ASSERT_EQ(depth.rows, H);
    ASSERT_EQ(depth.cols, W);

    // 至少 30% 的中心 ROI 像素有效
    cv::Rect roi(W/4, H/4, W/2, H/2);
    int valid = 0;
    for (int y = roi.y; y < roi.y+roi.height; ++y)
        for (int x = roi.x; x < roi.x+roi.width; ++x)
            if (depth.at<float>(y,x) > 0.0f) ++valid;

    EXPECT_GT(valid, roi.area() * 0.30)
        << "CPU PatchMatch should yield > 30% valid pixels in ROI";

    cfg.returnNativeResolution = true;
    cv::Mat native_depth;
    cv::Mat native_confidence;
    ASSERT_TRUE(xjw::mvs::PatchMatchDepthEstimator::estimate(
        ref,
        {src},
        refCam,
        {srcCam},
        4.0f,
        20.0f,
        cfg,
        native_depth,
        &native_confidence,
        &err)) << err;
    EXPECT_EQ(native_depth.size(), cv::Size(W / 2, H / 2));
    EXPECT_EQ(native_confidence.size(), native_depth.size());

    cv::Mat restored_depth;
    cv::Mat restored_confidence;
    cv::resize(native_depth, restored_depth, depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(native_confidence,
               restored_confidence,
               conf.size(),
               0.0,
               0.0,
               cv::INTER_NEAREST);
    EXPECT_EQ(cv::countNonZero(restored_depth != depth), 0);
    EXPECT_EQ(cv::countNonZero(restored_confidence != conf), 0);
}

TEST(MvsPipelineTest, PatchMatchRejectsNonPositiveIterationCount)
{
    constexpr int width = 16;
    constexpr int height = 12;
    cv::Mat reference = makeSyntheticGray(width, height);
    cv::Mat source = makeShifted(reference, 1);
    const double identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double reference_center[3] = {0, 0, 0};
    const double source_center[3] = {1, 0, 0};

    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cpu;
    config.numIterations = 0;

    cv::Mat depth;
    cv::Mat confidence;
    std::string error;
    EXPECT_FALSE(xjw::mvs::PatchMatchDepthEstimator::estimate(
        reference,
        {source},
        makeMvsCamera(32.0, 32.0, width * 0.5, height * 0.5,
                      identity, reference_center),
        {makeMvsCamera(32.0, 32.0, width * 0.5, height * 0.5,
                       identity, source_center)},
        1.0f,
        10.0f,
        config,
        depth,
        &confidence,
        &error));
    EXPECT_EQ(error, "PatchMatch iteration count must be positive");
    EXPECT_TRUE(depth.empty());
    EXPECT_TRUE(confidence.empty());
}

TEST(MvsPipelineTest, PatchMatchCpuHonorsPerPixelDepthRadius)
{
    constexpr int width = 64;
    constexpr int height = 48;
    constexpr double focal = 52.0;
    constexpr double baseline = 1.0;

    cv::Mat reference = makeSyntheticGray(width, height);
    cv::Mat source = makeShifted(reference, 5);
    const double identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double center_reference[3] = {0, 0, 0};
    const double center_source[3] = {baseline, 0, 0};
    const auto reference_camera = makeMvsCamera(
        focal, focal, width * 0.5, height * 0.5, identity, center_reference);
    const auto source_camera = makeMvsCamera(
        focal, focal, width * 0.5, height * 0.5, identity, center_source);

    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cpu;
    config.downsampleFactor = 2;
    config.patchHalf = 2;
    config.numIterations = 2;
    config.confidenceThresh = 0.0f;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;

    cv::Mat hint_depth(height / 2, width / 2, CV_32F, cv::Scalar(10.0f));
    cv::Mat hint_radius(height / 2, width / 2, CV_32F, cv::Scalar(0.2f));
    cv::Mat depth;
    cv::Mat confidence;
    std::string error;
    ASSERT_TRUE(xjw::mvs::PatchMatchDepthEstimator::estimate(
        reference,
        {source},
        reference_camera,
        {source_camera},
        2.0f,
        20.0f,
        config,
        depth,
        &confidence,
        &error,
        &hint_depth,
        &hint_radius)) << error;

    ASSERT_FALSE(depth.empty());
    const cv::Mat valid_mask = depth > 0.0f;
    ASSERT_GT(cv::countNonZero(valid_mask), 0);
    double minimum_depth = 0.0;
    double maximum_depth = 0.0;
    cv::minMaxLoc(depth, &minimum_depth, &maximum_depth, nullptr, nullptr, valid_mask);
    EXPECT_GE(minimum_depth, 9.8 - 1e-4);
    EXPECT_LE(maximum_depth, 10.2 + 1e-4);
}

// ---------------------------------------------------------------------------
// Test 2: DepthMapFusion 双帧融合输出非空点云
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, DepthMapFusionTwoFrames)
{
    constexpr int W = 48, H = 36;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0, BASELINE = 1.0;

    // 合成两帧一致的恒定深度图（模拟平面场景）
    cv::Mat d0(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    cv::Mat d1(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    // 边缘置 0（无效），避免边界效应
    d0(cv::Rect(0,0,W,4)) = 0;  d0(cv::Rect(0,H-4,W,4)) = 0;
    d1(cv::Rect(0,0,W,4)) = 0;  d1(cv::Rect(0,H-4,W,4)) = 0;

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0}, C1[3]={BASELINE,0,0};

    xjw::mvs::FusionFrameInput fr0, fr1;
    fr0.depthMap    = d0;
    fr0.cameraModel = makeMvsCamera(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);
    fr0.imgW = W; fr0.imgH = H;

    fr1.depthMap    = d1;
    fr1.cameraModel = makeMvsCamera(FOCAL, FOCAL, W*0.5, H*0.5, I, C1);
    fr1.imgW = W; fr1.imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels   = 2;
    fcfg.maxReprojError = 3.0f;
    fcfg.maxDepthError  = 0.10f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    bool ok = fusion.fuse({fr0, fr1}, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(static_cast<int>(pts.size()), 0)
        << "DepthMapFusion should produce at least one fused point";
}

TEST(MvsPipelineTest, FusionReprojectionThresholdUsesEachTargetFrameGridScale)
{
    constexpr int width = 9;
    constexpr int height = 9;
    constexpr double identity[9] = {1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
    constexpr double center[3] = {0.0, 0.0, 0.0};

    std::vector<xjw::mvs::FusionFrameInput> frames(2);
    frames[0].depthMap = cv::Mat::zeros(height, width, CV_32F);
    frames[0].depthMap.at<float>(4, 4) = 8.0f;
    frames[0].cameraModel = makeMvsCamera(
        20.0, 20.0, 4.0, 4.0, identity, center);
    frames[0].sourceCamera = frames[0].cameraModel;
    frames[0].sourceCamera.setImageSize(xjw::CameraImageSize{width, height});
    frames[0].imgW = width;
    frames[0].imgH = height;
    frames[0].sourceImageIndices = {1};

    frames[1].depthMap = cv::Mat(height, width, CV_32F, cv::Scalar(8.0f));
    frames[1].cameraModel = makeMvsCamera(
        20.0, 20.0, 4.4, 4.0, identity, center);
    frames[1].sourceCamera = frames[1].cameraModel;
    frames[1].sourceCamera.setImageSize(
        xjw::CameraImageSize{width * 4, height * 4});
    frames[1].imgW = width;
    frames[1].imgH = height;

    xjw::mvs::StereoFusionConfig config;
    config.fuseOnlyFirstFrame = true;
    config.minNumPixels = 2;
    config.checkNumImages = 1;
    config.maxReprojError = 1.0f;
    config.maxDepthError = 0.01f;
    config.maxLocalDepthGradient = 0.0f;
    config.pixelParametersUsePreparedRaster = true;

    xjw::mvs::DepthMapFusion native_fusion(config);
    std::vector<xjw::mvs::FusedPoint> native_points;
    std::string error;
    ASSERT_TRUE(native_fusion.fuse(frames, native_points, nullptr, &error))
        << error;
    EXPECT_TRUE(native_points.empty())
        << "A 0.4-grid-pixel residual must exceed the target's ds4 0.25-pixel gate.";

    frames[1].sourceCamera.setImageSize(
        xjw::CameraImageSize{width, height});
    xjw::mvs::DepthMapFusion full_grid_fusion(config);
    std::vector<xjw::mvs::FusedPoint> full_grid_points;
    ASSERT_TRUE(full_grid_fusion.fuse(
        frames, full_grid_points, nullptr, &error)) << error;
    EXPECT_EQ(full_grid_points.size(), 1u);
}

TEST(MvsPipelineTest, FusionLocalGradientUsesEachFrameGridScale)
{
    constexpr int width = 9;
    constexpr int height = 9;
    constexpr double identity[9] = {1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
    constexpr double center[3] = {0.0, 0.0, 0.0};

    std::vector<xjw::mvs::FusionFrameInput> frames(2);
    frames[0].depthMap = cv::Mat::zeros(height, width, CV_32F);
    frames[0].depthMap.at<float>(4, 4) = 8.0f;
    frames[1].depthMap = cv::Mat(height, width, CV_32F, cv::Scalar(12.0f));
    frames[1].depthMap.at<float>(4, 4) = 8.0f;
    for (auto &frame : frames)
    {
        frame.cameraModel = makeMvsCamera(
            20.0, 20.0, 4.0, 4.0, identity, center);
        frame.sourceCamera = frame.cameraModel;
        frame.imgW = width;
        frame.imgH = height;
    }
    frames[0].sourceCamera.setImageSize(
        xjw::CameraImageSize{width, height});
    frames[1].sourceCamera.setImageSize(
        xjw::CameraImageSize{width * 4, height * 4});
    frames[0].sourceImageIndices = {1};

    xjw::mvs::StereoFusionConfig config;
    config.fuseOnlyFirstFrame = true;
    config.minNumPixels = 2;
    config.checkNumImages = 1;
    config.maxReprojError = 1.0f;
    config.maxDepthError = 0.01f;
    config.maxLocalDepthGradient = 0.10f;
    config.localDepthGradientRadiusPixels = 1;
    config.pixelParametersUsePreparedRaster = true;

    xjw::mvs::DepthMapFusion native_fusion(config);
    std::vector<xjw::mvs::FusedPoint> native_points;
    std::string error;
    ASSERT_TRUE(native_fusion.fuse(frames, native_points, nullptr, &error))
        << error;
    EXPECT_EQ(native_points.size(), 1u)
        << "The one-full-raster-pixel gradient radius is subpixel on the ds4 target.";

    frames[1].sourceCamera.setImageSize(
        xjw::CameraImageSize{width, height});
    xjw::mvs::DepthMapFusion full_grid_fusion(config);
    std::vector<xjw::mvs::FusedPoint> full_grid_points;
    ASSERT_TRUE(full_grid_fusion.fuse(
        frames, full_grid_points, nullptr, &error)) << error;
    EXPECT_TRUE(full_grid_points.empty());
}

TEST(MvsPipelineTest, DepthMapFusionRejectsMaskedLowSupportAndConflictingSheets)
{
    constexpr int W = 20;
    constexpr int H = 16;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(2);
    for (auto &frame : frames)
    {
        frame.depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
        frame.validMask = cv::Mat(H, W, CV_8U, cv::Scalar(255));
        frame.geometrySupportCount = cv::Mat(H, W, CV_16U, cv::Scalar(2));
        frame.cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
        frame.imgW = W;
        frame.imgH = H;

        // A doorway is explicitly outside the authoritative project mask.
        frame.validMask(cv::Rect(8, 5, 4, 6)) = 0;

        // A disconnected island has depth but insufficient multi-view support.
        frame.geometrySupportCount(cv::Rect(1, 1, 2, 2)) = 1;
    }

    // The second frame contains a conflicting rear sheet which must not survive fusion.
    frames[1].depthMap(cv::Rect(14, 5, 3, 4)) = 12.0f;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels = 2;
    fcfg.checkNumImages = 1;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;
    fcfg.requireValidMask = true;
    fcfg.minGeometryObservationCount = 2;
    fcfg.maxLocalDepthGradient = 0.20f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> points;
    std::string error;
    const bool ok = fusion.fuse(frames, points, nullptr, &error);

    ASSERT_TRUE(ok) << error;
    ASSERT_FALSE(points.empty());

    const auto &filtered_depths = fusion.filteredDepths();
    ASSERT_EQ(filtered_depths.size(), frames.size());
    EXPECT_EQ(filtered_depths[0].at<float>(7, 9), 0.0f);
    EXPECT_EQ(filtered_depths[0].at<float>(1, 1), 0.0f);
    EXPECT_EQ(filtered_depths[0].at<float>(6, 15), 0.0f);
    EXPECT_GT(filtered_depths[0].at<float>(12, 6), 0.0f);

    const xjw::mvs::FusionRejectionStats stats = fusion.rejectionStats();
    EXPECT_GT(stats.maskRejected, 0u);
    EXPECT_GT(stats.supportRejected, 0u);
    EXPECT_GT(stats.depthConsistencyRejected, 0u);
}

TEST(MvsPipelineTest, DepthFrameResultReleasesFusionQualityArtifacts)
{
    xjw::mvs::DepthFrameResult result;
    result.preparedRasterSize = cv::Size(20, 16);
    result.effectiveNativeFinalDepthGrid = true;
    result.pixelDomainDiagnostics.insert(
        QStringLiteral("linear_scale"), 0.25);
    result.depthMap = QSharedPointer<cv::Mat>::create(4, 5, CV_32F, cv::Scalar(8.0f));
    result.normalMap = QSharedPointer<cv::Mat>::create(4, 5, CV_32FC3, cv::Scalar(0.0f, 0.0f, 1.0f));
    result.supportCount = QSharedPointer<cv::Mat>::create(4, 5, CV_16U, cv::Scalar(3));
    result.adaptiveGeometrySupportWeight =
        QSharedPointer<cv::Mat>::create(4, 5, CV_32F, cv::Scalar(0.8f));
    result.adaptiveGeometryEffectiveViewCount =
        QSharedPointer<cv::Mat>::create(4, 5, CV_32F, cv::Scalar(2.0f));
    result.adaptiveGeometryConflictRatio =
        QSharedPointer<cv::Mat>::create(4, 5, CV_32F, cv::Scalar(0.1f));
    result.validMask = QSharedPointer<cv::Mat>::create(4, 5, CV_8U, cv::Scalar(255));
    result.supportRegionMask =
        QSharedPointer<cv::Mat>::create(4, 5, CV_8U, cv::Scalar(255));

    result.releasePixelStorage();

    EXPECT_TRUE(result.depthMap.isNull());
    EXPECT_TRUE(result.normalMap.isNull());
    EXPECT_TRUE(result.supportCount.isNull());
    EXPECT_TRUE(result.adaptiveGeometrySupportWeight.isNull());
    EXPECT_TRUE(result.adaptiveGeometryEffectiveViewCount.isNull());
    EXPECT_TRUE(result.adaptiveGeometryConflictRatio.isNull());
    EXPECT_TRUE(result.validMask.isNull());
    EXPECT_TRUE(result.supportRegionMask.isNull());
    EXPECT_EQ(result.preparedRasterSize, cv::Size(20, 16));
    EXPECT_TRUE(result.effectiveNativeFinalDepthGrid);
    EXPECT_DOUBLE_EQ(result.pixelDomainDiagnostics.value(
        QStringLiteral("linear_scale")).toDouble(), 0.25);
}

TEST(MvsPipelineTest, StreamingPixelReleasePreservesSupportRegionMask)
{
    xjw::mvs::DepthFrameResult result;
    result.depthMap =
        QSharedPointer<cv::Mat>::create(3, 4, CV_32F, cv::Scalar(8.0f));
    result.confidence =
        QSharedPointer<cv::Mat>::create(3, 4, CV_32F, cv::Scalar(0.8f));
    result.validMask =
        QSharedPointer<cv::Mat>::create(3, 4, CV_8U, cv::Scalar(255));
    result.supportRegionMask =
        QSharedPointer<cv::Mat>::create(3, 4, CV_8U, cv::Scalar(255));
    result.supportRegionMask->at<std::uint8_t>(1, 2) = 0;

    result.releaseStreamingPixelStorage();

    EXPECT_TRUE(result.depthMap.isNull());
    EXPECT_TRUE(result.confidence.isNull());
    EXPECT_TRUE(result.validMask.isNull());
    ASSERT_FALSE(result.supportRegionMask.isNull());
    EXPECT_EQ(result.supportRegionMask->size(), cv::Size(4, 3));
    EXPECT_EQ(result.supportRegionMask->at<std::uint8_t>(1, 2), 0);

    result.releasePixelStorage();
    EXPECT_TRUE(result.supportRegionMask.isNull());
}

TEST(MvsPipelineTest, DepthMapFusionCancelBeforeWorkClearsStaleOutput)
{
    constexpr int W = 16;
    constexpr int H = 12;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C0[3] = {0,0,0};

    xjw::mvs::FusionFrameInput frame;
    frame.depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    frame.cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C0);
    frame.imgW = W;
    frame.imgH = H;

    auto cancelFlag = std::make_shared<std::atomic_bool>(true);
    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.cancelFlag = cancelFlag;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts(1);
    pts.front().x = 123.0f;
    std::string err;

    const bool ok = fusion.fuse({frame}, pts, nullptr, &err);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(err.find("取消") != std::string::npos);
    EXPECT_TRUE(pts.empty()) << "Cancelled fusion must not leave stale points from a previous run";
}

TEST(MvsPipelineTest, DepthMapFusionTwoViewSingleObservationUsesFastParallelPath)
{
    constexpr int W = 32, H = 24;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0, BASELINE = 1.0;

    cv::Mat d0(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    cv::Mat d1(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    d0(cv::Rect(0, 0, W, 2)) = 0;
    d1(cv::Rect(0, 0, W, 2)) = 0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C0[3] = {0,0,0};
    const double C1[3] = {BASELINE,0,0};

    xjw::mvs::FusionFrameInput fr0, fr1;
    fr0.depthMap = d0;
    fr0.cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C0);
    fr0.imgW = W;
    fr0.imgH = H;

    fr1.depthMap = d1;
    fr1.cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C1);
    fr1.imgW = W;
    fr1.imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels = 1;
    fcfg.maxReprojError = 3.0f;
    fcfg.maxDepthError = 0.10f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::vector<std::string> stages;
    std::string err;
    const bool ok = fusion.fuse({fr0, fr1},
                                pts,
                                [&](const std::string &stage, float) {
                                    stages.push_back(stage);
                                },
                                &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(static_cast<int>(pts.size()), 0);
    EXPECT_TRUE(std::any_of(stages.begin(), stages.end(), [](const std::string &stage) {
        return stage.find("快速并行融合") != std::string::npos;
    }));
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionRejectsDepthsWithoutNeighborAgreement)
{
    constexpr int W = 16;
    constexpr int H = 12;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(2);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[0].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[0].imgW = W;
    frames[0].imgH = H;

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(12.0f));
    frames[1].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[1].imgW = W;
    frames[1].imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 2;
    fcfg.checkNumImages = 1;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_TRUE(pts.empty())
        << "Streaming fusion must not directly back-project first-frame depths when neighbors disagree.";
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionKeepsProductionStrictWhenFallbackIsNotEnabled)
{
    constexpr int W = 20;
    constexpr int H = 14;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[0].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[0].imgW = W;
    frames[0].imgH = H;
    frames[0].sourceImageIndices = {1, 2};

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[1].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[1].imgW = W;
    frames[1].imgH = H;

    frames[2].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(12.0f));
    frames[2].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[2].imgW = W;
    frames[2].imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_TRUE(pts.empty())
        << "Production fusion must keep strict 3-view agreement unless low-yield fallback is explicitly enabled.";
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionFallsBackToTwoViewAgreementWhenStrictYieldCollapses)
{
    constexpr int W = 20;
    constexpr int H = 14;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[0].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[0].imgW = W;
    frames[0].imgH = H;
    frames[0].sourceImageIndices = {1, 2};

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[1].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[1].imgW = W;
    frames[1].imgH = H;

    frames[2].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(12.0f));
    frames[2].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[2].imgW = W;
    frames[2].imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;
    fcfg.enableLowYieldFallback = true;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(pts.size(), 0u)
        << "When strict 3-view streaming fusion collapses, the pipeline should preserve supported "
           "2-view dense observations instead of producing a sparse-like cloud.";
}

TEST(MvsPipelineTest, DepthMapFusionFilteredDepthsIncludeAllAcceptedObservations)
{
    constexpr int W = 24;
    constexpr int H = 18;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0;

    cv::Mat depth(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    depth(cv::Rect(0, 0, W, 3)) = 0;
    depth(cv::Rect(0, H - 3, W, 3)) = 0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    for (auto &frame : frames)
    {
        frame.depthMap = depth.clone();
        frame.cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
        frame.imgW = W;
        frame.imgH = H;
    }

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 1.0f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    ASSERT_GT(static_cast<int>(pts.size()), 0);

    const auto &filteredDepths = fusion.filteredDepths();
    ASSERT_EQ(filteredDepths.size(), frames.size());

    const int validPixels = cv::countNonZero(depth > 0);
    for (size_t frameIndex = 0; frameIndex < filteredDepths.size(); ++frameIndex)
    {
        EXPECT_EQ(cv::countNonZero(filteredDepths[frameIndex] > 0), validPixels)
            << "Accepted source observations should be visible in filtered depth frame " << frameIndex;
    }
}

TEST(MvsPipelineTest, DepthMapFusionUsesPlannedSourceImagesBeforeNearestCenters)
{
    constexpr int W = 24;
    constexpr int H = 18;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C0[3] = {0,0,0};
    const double C1[3] = {0.1,0,0};
    const double C2[3] = {0.2,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    frames[0].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C0);
    frames[0].imgW = W;
    frames[0].imgH = H;
    frames[0].sourceImageIndices = {2};

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(0.0f));
    frames[1].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C1);
    frames[1].imgW = W;
    frames[1].imgH = H;

    frames[2].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    frames[2].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C2);
    frames[2].imgW = W;
    frames[2].imgH = H;
    frames[2].sourceImageIndices = {0};

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels = 2;
    fcfg.checkNumImages = 1;
    fcfg.maxReprojError = 100.0f;
    fcfg.maxDepthError = 1.0f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(static_cast<int>(pts.size()), 0)
        << "Fusion should prefer planned source image 2 over nearest invalid image 1.";
}

TEST(MvsPipelineTest, MvsQualityReportFlagsLowConfidenceFullCoverageDepthMap)
{
    cv::Mat depth(20, 30, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(20, 30, CV_32F, cv::Scalar(0.56f));
    confidence.at<float>(10, 15) = 0.82f;

    const xjw::mvs::DepthMapQualityMetrics metrics =
        xjw::mvs::analyzeDepthMapQuality(depth, confidence, 3);

    EXPECT_GT(metrics.validCoverage, 0.95f);
    EXPECT_LT(metrics.meanConfidence, 0.65f);
    EXPECT_TRUE(metrics.lowConfidenceFullCoverage);
    EXPECT_GE(metrics.recommendedFusionConfidence, 0.65f);

    const QJsonObject json = xjw::mvs::depthMapQualityMetricsToJson(metrics);
    EXPECT_TRUE(json.value(QStringLiteral("low_confidence_full_coverage")).toBool());
    EXPECT_GE(json.value(QStringLiteral("recommended_fusion_confidence")).toDouble(), 0.65);
}

TEST(MvsPipelineTest, MvsQualityReportMeasuresConnectedSupportAndBoundaryCollapse)
{
    cv::Mat depth = cv::Mat::zeros(20, 30, CV_32F);
    depth(cv::Rect(2, 3, 12, 10)).setTo(10.0f);
    depth(cv::Rect(8, 3, 6, 10)).setTo(14.0f);
    depth(cv::Rect(25, 16, 2, 2)).setTo(19.9f);
    cv::Mat confidence(20, 30, CV_32F, cv::Scalar(0.8f));

    const xjw::mvs::DepthMapQualityMetrics metrics =
        xjw::mvs::analyzeDepthMapQuality(depth, confidence, 4, 5.0f, 20.0f);

    EXPECT_GT(metrics.largestComponentRatio, 0.95f);
    EXPECT_NEAR(metrics.depthAtSearchBoundaryRatio, 4.0f / 124.0f, 1.0e-5f);

    const QJsonObject json = xjw::mvs::depthMapQualityMetricsToJson(metrics);
    EXPECT_GT(json.value(QStringLiteral("largest_component_ratio")).toDouble(), 0.95);
    EXPECT_GT(json.value(QStringLiteral("depth_at_search_boundary_ratio")).toDouble(), 0.03);
    EXPECT_GT(metrics.depthDiscontinuityRatio, 0.0f);
    EXPECT_GT(json.value(QStringLiteral("depth_discontinuity_ratio")).toDouble(), 0.0);
}

TEST(DepthFrameQualityGateTest, RejectsDepthSearchBoundaryCollapse)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    input.sourceViewCount = 4;
    input.validCoverage = 0.42f;
    input.largestComponentRatio = 0.85f;
    input.meanConfidence = 0.78f;
    input.multiViewConsistency = 0.81f;
    input.depthAtSearchBoundaryRatio = 0.72f;

    const xjw::mvs::DepthFrameQualityDecision decision =
        xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance, xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("depth_search_boundary_collapse")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest, RejectsLowConfidenceFullCoverage)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    input.sourceViewCount = 5;
    input.validCoverage = 0.99f;
    input.largestComponentRatio = 0.99f;
    input.meanConfidence = 0.31f;
    input.multiViewConsistency = 0.28f;

    const xjw::mvs::DepthFrameQualityDecision decision =
        xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance, xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("low_confidence_full_coverage")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest, RejectsDestructiveOutputFilterCollapse)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    input.sourceViewCount = 3;
    input.validCoverage = 0.20f;
    input.largestComponentRatio = 0.90f;
    input.meanConfidence = 0.80f;
    input.multiViewConsistency = 0.80f;
    input.validWithinMaskRatio = 0.92f;
    input.outputFilterRetentionRatio = 0.69f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance, xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("destructive_output_filter_collapse")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest, UsesPerFrameSourceCountForConsistencyPolicy)
{
    EXPECT_TRUE(xjw::mvs::useContradictionOnlyDepthConsistency(1));
    EXPECT_FALSE(xjw::mvs::useContradictionOnlyDepthConsistency(2));
    EXPECT_FALSE(xjw::mvs::useContradictionOnlyDepthConsistency(3));
    EXPECT_FLOAT_EQ(xjw::mvs::depthConsistencyRelativeThreshold(
                        xjw::mvs::MvsSceneProfile::OrbitalObject, 2),
                    0.06f);
    EXPECT_FLOAT_EQ(xjw::mvs::depthConsistencyRelativeThreshold(
                        xjw::mvs::MvsSceneProfile::OrbitalObject, 3),
                    0.008f);
}

TEST(DepthFrameQualityGateTest, ClassifiesOcclusionWithoutRejectingReferenceDepth)
{
    using xjw::mvs::classifyDepthConsistencyEvidence;
    using xjw::mvs::DepthConsistencyEvidence;

    EXPECT_EQ(classifyDepthConsistencyEvidence(10.0f, 10.5f, 0.10f),
              DepthConsistencyEvidence::Consistent);
    EXPECT_EQ(classifyDepthConsistencyEvidence(10.0f, 8.0f, 0.10f),
              DepthConsistencyEvidence::Occluded);
    EXPECT_EQ(classifyDepthConsistencyEvidence(10.0f, 12.0f, 0.10f),
              DepthConsistencyEvidence::Contradicted);
    EXPECT_EQ(classifyDepthConsistencyEvidence(10.0f, 0.0f, 0.10f),
              DepthConsistencyEvidence::Unverifiable);
}

TEST(DepthGeometryConsistencyTest, FindsSubpixelNeighborAndVerifiesRoundTrip)
{
    constexpr double identity[9] = {1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
    constexpr double reference_center[3] = {0.0, 0.0, 0.0};
    constexpr double source_center[3] = {0.05, 0.0, 0.0};
    const xjw::FramePinholeCamera reference_camera = makeMvsCamera(
        100.0, 100.0, 4.0, 4.0, identity, reference_center);
    const xjw::FramePinholeCamera source_camera = makeMvsCamera(
        100.0, 100.0, 4.0, 4.0, identity, source_center);

    const cv::Point2f reference_pixel(4.0f, 4.0f);
    constexpr float reference_depth = 10.0f;
    const double pixel[2] = {reference_pixel.x, reference_pixel.y};
    double world[3] = {0.0, 0.0, 0.0};
    ASSERT_TRUE(reference_camera.unprojectPixel(pixel, reference_depth, world));
    double projected[2] = {0.0, 0.0};
    double expected_depth = 0.0;
    ASSERT_TRUE(source_camera.projectWorldPointWithDepth(world, projected, expected_depth));

    const int center_column = static_cast<int>(std::lround(projected[0]));
    const int center_row = static_cast<int>(std::lround(projected[1]));
    cv::Mat source_depth(9, 9, CV_32F, cv::Scalar(0.0f));
    source_depth.at<float>(center_row, center_column) = 20.0f;
    const int consistent_column = center_column > projected[0]
        ? center_column - 1
        : center_column + 1;
    source_depth.at<float>(center_row, consistent_column) =
        static_cast<float>(expected_depth);

    const auto central_only = xjw::mvs::evaluateProjectedDepthConsistency(
        reference_camera,
        reference_pixel,
        reference_depth,
        source_camera,
        source_depth,
        0.05f,
        0,
        1.5f,
        true);
    EXPECT_EQ(central_only.evidence, xjw::mvs::DepthConsistencyEvidence::Contradicted);
    EXPECT_TRUE(central_only.continuousGeometryValid);
    EXPECT_EQ(
        xjw::mvs::adaptiveGeometryEvidenceClass(central_only),
        xjw::mvs::AdaptiveGeometryEvidenceClass::Comparable);
    EXPECT_GT(central_only.worldSurfaceResidual, 0.0f);
    EXPECT_GT(central_only.jointWorldPixelFootprint, 0.0f);

    const auto reduced_grid_subpixel =
        xjw::mvs::evaluateProjectedDepthConsistency(
            reference_camera,
            reference_pixel,
            reference_depth,
            source_camera,
            source_depth,
            0.05f,
            0,
            1.5f,
            true,
            true);
    EXPECT_EQ(reduced_grid_subpixel.evidence,
              xjw::mvs::DepthConsistencyEvidence::Consistent);
    EXPECT_EQ(reduced_grid_subpixel.sourcePixel.x, consistent_column);
    EXPECT_LE(reduced_grid_subpixel.roundTripErrorPixels, 1.5f);

    const auto edge_aware = xjw::mvs::evaluateProjectedDepthConsistency(
        reference_camera,
        reference_pixel,
        reference_depth,
        source_camera,
        source_depth,
        0.05f,
        1,
        1.5f);
    EXPECT_EQ(edge_aware.evidence, xjw::mvs::DepthConsistencyEvidence::Consistent);
    EXPECT_EQ(edge_aware.sourcePixel.x, consistent_column);
    EXPECT_LE(edge_aware.roundTripErrorPixels, 1.5f);
    EXPECT_TRUE(edge_aware.continuousGeometryValid);
    EXPECT_GT(edge_aware.jointWorldPixelFootprint, 0.0f);
    EXPECT_NEAR(edge_aware.worldSurfaceResidual, 0.05f, 1.0e-5f);
    EXPECT_NEAR(edge_aware.worldSurfaceResidual /
                    edge_aware.jointWorldPixelFootprint,
                0.5f,
                1.0e-5f);

}

TEST(DepthGeometryConsistencyTest,
     JointPixelFootprintIncludesEpipolarTriangulationUncertainty)
{
    constexpr double identity[9] = {1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
    constexpr double reference_center[3] = {0.0, 0.0, 0.0};
    constexpr double source_center[3] = {1.0, 0.0, 0.0};
    const xjw::FramePinholeCamera reference_camera = makeMvsCamera(
        100.0, 100.0, 20.0, 20.0, identity, reference_center);
    const xjw::FramePinholeCamera source_camera = makeMvsCamera(
        100.0, 100.0, 20.0, 20.0, identity, source_center);
    cv::Mat source_depth(41, 41, CV_32F, cv::Scalar(0.0f));
    source_depth.at<float>(20, 10) = 10.0f;

    const auto result = xjw::mvs::evaluateProjectedDepthConsistency(
        reference_camera,
        cv::Point2f(20.0f, 20.0f),
        10.0f,
        source_camera,
        source_depth,
        0.01f,
        0,
        1.0f,
        true);

    ASSERT_EQ(result.evidence, xjw::mvs::DepthConsistencyEvidence::Consistent);
    ASSERT_TRUE(result.continuousGeometryValid);
    const float reference_pixel_size =
        10.0f / std::sqrt(100.0f * 100.0f + 1.0f);
    const float target_epipolar_uncertainty = 10.0f / 9.0f;
    EXPECT_NEAR(
        result.jointWorldPixelFootprint,
        0.5f * (reference_pixel_size + target_epipolar_uncertainty),
        1.0e-4f);
    EXPECT_GT(result.jointWorldPixelFootprint, 5.0f * 0.1f);
}

TEST(DepthGeometryConsistencyTest,
     ReusedReferenceWorldPreservesConsistencyResultExactly)
{
    constexpr double identity[9] = {1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
    constexpr double reference_center[3] = {0.0, 0.0, 0.0};
    constexpr double source_center[3] = {0.25, 0.0, 0.0};
    const xjw::FramePinholeCamera reference_camera = makeMvsCamera(
        120.0, 120.0, 16.0, 16.0, identity, reference_center);
    const xjw::FramePinholeCamera source_camera = makeMvsCamera(
        120.0, 120.0, 16.0, 16.0, identity, source_center);
    const cv::Point2f reference_pixel(16.0f, 16.0f);
    constexpr float reference_depth = 12.0f;
    const double pixel[2] = {reference_pixel.x, reference_pixel.y};
    double world[3] = {};
    ASSERT_TRUE(reference_camera.unprojectPixel(
        pixel, reference_depth, world));
    double projected[2] = {};
    double source_depth_value = 0.0;
    ASSERT_TRUE(source_camera.projectWorldPointWithDepth(
        world, projected, source_depth_value));
    cv::Mat source_depth(33, 33, CV_32FC1, cv::Scalar(0.0f));
    source_depth.at<float>(
        static_cast<int>(std::lround(projected[1])),
        static_cast<int>(std::lround(projected[0]))) =
            static_cast<float>(source_depth_value);

    const auto direct = xjw::mvs::evaluateProjectedDepthConsistency(
        reference_camera,
        reference_pixel,
        reference_depth,
        source_camera,
        source_depth,
        0.01f,
        1,
        3.0f,
        true);
    const auto reused =
        xjw::mvs::evaluateProjectedDepthConsistencyFromReferenceWorld(
            reference_camera,
            reference_pixel,
            reference_depth,
            {world[0], world[1], world[2]},
            source_camera,
            source_depth,
            0.01f,
            1,
            3.0f,
            true);

    EXPECT_EQ(reused.evidence, direct.evidence);
    EXPECT_EQ(reused.sourcePixel, direct.sourcePixel);
    EXPECT_FLOAT_EQ(reused.relativeDepthError, direct.relativeDepthError);
    EXPECT_FLOAT_EQ(reused.roundTripErrorPixels, direct.roundTripErrorPixels);
    EXPECT_FLOAT_EQ(
        reused.consistentReferenceDepth, direct.consistentReferenceDepth);
    EXPECT_FLOAT_EQ(reused.worldSurfaceResidual, direct.worldSurfaceResidual);
    EXPECT_FLOAT_EQ(
        reused.jointWorldPixelFootprint, direct.jointWorldPixelFootprint);
    EXPECT_EQ(reused.continuousGeometryValid, direct.continuousGeometryValid);
}

TEST(DepthGeometryConsistencyTest,
     JointPixelFootprintFallsBackForDegenerateEpipolarGeometry)
{
    constexpr double identity[9] = {1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
    constexpr double reference_center[3] = {0.0, 0.0, 0.0};
    const xjw::FramePinholeCamera reference_camera = makeMvsCamera(
        100.0, 100.0, 20.0, 20.0, identity, reference_center);

    const xjw::FramePinholeCamera coincident_camera = makeMvsCamera(
        100.0, 100.0, 20.0, 20.0, identity, reference_center);
    cv::Mat coincident_depth(41, 41, CV_32F, cv::Scalar(0.0f));
    coincident_depth.at<float>(20, 20) = 10.0f;
    const auto coincident = xjw::mvs::evaluateProjectedDepthConsistency(
        reference_camera,
        cv::Point2f(20.0f, 20.0f),
        10.0f,
        coincident_camera,
        coincident_depth,
        0.01f,
        0,
        1.0f,
        true);
    ASSERT_TRUE(coincident.continuousGeometryValid);
    EXPECT_NEAR(coincident.jointWorldPixelFootprint, 0.1f, 1.0e-6f);

    constexpr double collinear_center[3] = {0.0, 0.0, 1.0};
    const xjw::FramePinholeCamera collinear_camera = makeMvsCamera(
        100.0, 100.0, 20.0, 20.0, identity, collinear_center);
    cv::Mat collinear_depth(41, 41, CV_32F, cv::Scalar(0.0f));
    collinear_depth.at<float>(20, 20) = 9.0f;
    const auto collinear = xjw::mvs::evaluateProjectedDepthConsistency(
        reference_camera,
        cv::Point2f(20.0f, 20.0f),
        10.0f,
        collinear_camera,
        collinear_depth,
        0.01f,
        0,
        1.0f,
        true);
    ASSERT_TRUE(collinear.continuousGeometryValid);
    EXPECT_NEAR(collinear.jointWorldPixelFootprint, 0.095f, 1.0e-6f);
}

TEST(DepthGeometryConsistencyTest,
     MissingSourceDepthRemainsUnverifiableWithContinuousMetrics)
{
    constexpr double identity[9] = {1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
    constexpr double reference_center[3] = {0.0, 0.0, 0.0};
    constexpr double source_center[3] = {1.0, 0.0, 0.0};
    const xjw::FramePinholeCamera reference_camera = makeMvsCamera(
        100.0, 100.0, 20.0, 20.0, identity, reference_center);
    const xjw::FramePinholeCamera source_camera = makeMvsCamera(
        100.0, 100.0, 20.0, 20.0, identity, source_center);
    const cv::Mat source_depth(41, 41, CV_32F, cv::Scalar(0.0f));

    const auto result = xjw::mvs::evaluateProjectedDepthConsistency(
        reference_camera,
        cv::Point2f(20.0f, 20.0f),
        10.0f,
        source_camera,
        source_depth,
        0.01f,
        0,
        1.0f,
        true);

    EXPECT_EQ(result.evidence, xjw::mvs::DepthConsistencyEvidence::Unverifiable);
    EXPECT_FALSE(result.continuousGeometryValid);
    EXPECT_EQ(
        xjw::mvs::adaptiveGeometryEvidenceClass(result),
        xjw::mvs::AdaptiveGeometryEvidenceClass::Unobservable);
    EXPECT_FLOAT_EQ(result.jointWorldPixelFootprint, 0.0f);
}

TEST(DepthGeometryConsistencyTest, UsesAllVerifiableSourceVotes)
{
    using xjw::mvs::shouldRetainDepthFromConsistencyVotes;

    EXPECT_TRUE(shouldRetainDepthFromConsistencyVotes(1, 0, 1, 0));
    EXPECT_FALSE(shouldRetainDepthFromConsistencyVotes(1, 0, 0, 1));
    EXPECT_TRUE(shouldRetainDepthFromConsistencyVotes(3, 1, 1, 1));
    EXPECT_TRUE(shouldRetainDepthFromConsistencyVotes(3, 1, 0, 2));
    EXPECT_TRUE(shouldRetainDepthFromConsistencyVotes(4, 1, 0, 3));
    EXPECT_FALSE(shouldRetainDepthFromConsistencyVotes(4, 0, 0, 4));
    EXPECT_TRUE(shouldRetainDepthFromConsistencyVotes(5, 3, 0, 2));
    EXPECT_FALSE(shouldRetainDepthFromConsistencyVotes(6, 1, 0, 5, 2));
    EXPECT_TRUE(shouldRetainDepthFromConsistencyVotes(6, 2, 0, 4, 2));
    EXPECT_FALSE(shouldRetainDepthFromConsistencyVotes(6, 2, 0, 4, 3));
    EXPECT_TRUE(shouldRetainDepthFromConsistencyVotes(6, 3, 0, 3, 3));
}

TEST(DepthGeometryConsistencyTest, GeometrySupportCountsReferenceAndConfirmedSources)
{
    cv::Mat depth = (cv::Mat_<float>(1, 4) << 2.0f, 0.0f, 3.0f, 4.0f);
    cv::Mat votes = (cv::Mat_<std::uint16_t>(1, 4) << 0, 7, 2, 5);

    const cv::Mat support = xjw::mvs::makeGeometrySupportCount(depth, votes);

    ASSERT_EQ(support.type(), CV_16UC1);
    EXPECT_EQ(support.at<std::uint16_t>(0, 0), 1);
    EXPECT_EQ(support.at<std::uint16_t>(0, 1), 0);
    EXPECT_EQ(support.at<std::uint16_t>(0, 2), 3);
    EXPECT_EQ(support.at<std::uint16_t>(0, 3), 6);
}

TEST(DepthGeometryConsistencyTest, BuildsSourceAndInverseDepthEvidence)
{
    const cv::Mat depth = (cv::Mat_<float>(1, 3) << 2.0f, 0.0f, 4.0f);
    const cv::Mat votes = (cv::Mat_<std::uint16_t>(1, 3) << 2, 4, 1);
    const cv::Mat source_mask =
        (cv::Mat_<std::uint16_t>(1, 3) << 0x0005, 0x000f, 0x0002);
    const float inverse_a = 1.0f / 2.02f;
    const float inverse_b = 1.0f / 1.98f;
    const cv::Mat inverse_sum =
        (cv::Mat_<float>(1, 3) << inverse_a + inverse_b, 8.0f, 0.25f);
    const cv::Mat inverse_squared_sum =
        (cv::Mat_<float>(1, 3) << inverse_a * inverse_a + inverse_b * inverse_b,
         16.0f,
         0.0625f);

    const xjw::mvs::GeometryEvidenceMaps evidence =
        xjw::mvs::makeGeometryEvidenceMaps(
            depth, votes, source_mask, inverse_sum, inverse_squared_sum);

    ASSERT_FALSE(evidence.supportCount.empty());
    EXPECT_EQ(evidence.supportCount.at<std::uint16_t>(0, 0), 3);
    EXPECT_EQ(evidence.sourceMask.at<std::uint16_t>(0, 0), 0x0005);
    EXPECT_NEAR(evidence.inverseDepthMean.at<float>(0, 0), 0.5f, 1.0e-4f);
    EXPECT_LT(evidence.inverseDepthRelativeSpread.at<float>(0, 0), 0.01f);
    EXPECT_EQ(evidence.supportCount.at<std::uint16_t>(0, 1), 0);
    EXPECT_EQ(evidence.sourceMask.at<std::uint16_t>(0, 1), 0);
    EXPECT_FLOAT_EQ(evidence.inverseDepthMean.at<float>(0, 1), 0.0f);
    EXPECT_EQ(evidence.supportCount.at<std::uint16_t>(0, 2), 2);
}

TEST(DepthGeometryConsistencyTest,
     AllowsZeroMaskWithoutOrdinalsButRejectsNonzeroMask)
{
    const cv::Mat zero_mask(2, 3, CV_16UC1, cv::Scalar(0));
    const auto zero_contract =
        xjw::mvs::validateGeometrySourceOrdinalContract(
            zero_mask, {}, 5, 6, zero_mask.size());

    EXPECT_TRUE(zero_contract.valid);
    EXPECT_FALSE(zero_contract.persistMask);

    cv::Mat nonzero_mask = zero_mask.clone();
    nonzero_mask.at<std::uint16_t>(0, 1) = 1;
    const auto invalid_contract =
        xjw::mvs::validateGeometrySourceOrdinalContract(
            nonzero_mask, {}, 5, 6, nonzero_mask.size());

    EXPECT_FALSE(invalid_contract.valid);
    EXPECT_FALSE(invalid_contract.errorMessage.isEmpty());
}

TEST(DepthGeometryConsistencyTest, RequiresEveryMaskBitToHaveAnOrdinal)
{
    const cv::Mat valid_mask(1, 2, CV_16UC1, cv::Scalar(0x0008));
    const auto valid_contract =
        xjw::mvs::validateGeometrySourceOrdinalContract(
            valid_mask, {0, 2, 3, 4}, 5, 6, valid_mask.size());
    EXPECT_TRUE(valid_contract.valid);
    EXPECT_TRUE(valid_contract.persistMask);

    const auto invalid_contract =
        xjw::mvs::validateGeometrySourceOrdinalContract(
            valid_mask, {0, 2, 3}, 5, 6, valid_mask.size());
    EXPECT_FALSE(invalid_contract.valid);
}

TEST(DepthGeometryConsistencyTest, ReportsNativeRepairAndSupportBaseline)
{
    const cv::Mat depth =
        (cv::Mat_<float>(2, 4) <<
            2.0f, 2.0f, 2.0f, 0.0f,
            3.0f, 3.0f, 3.0f, 3.0f);
    const cv::Mat support =
        (cv::Mat_<std::uint16_t>(2, 4) <<
            1, 2, 3, 0,
            4, 5, 6, 2);
    const cv::Mat spread =
        (cv::Mat_<float>(2, 4) <<
            0.001f, 0.002f, 0.003f, 0.0f,
            0.004f, 0.005f, 0.006f, 0.007f);
    const cv::Mat repaired =
        (cv::Mat_<std::uint8_t>(2, 4) <<
            0, 255, 0, 0,
            0, 255, 0, 0);
    const cv::Mat region(2, 4, CV_8UC1, cv::Scalar(255));

    const QJsonObject json =
        xjw::mvs::geometryEvidenceDiagnosticsToJson(
            depth, support, spread, repaired, region);

    ASSERT_TRUE(json.value(QStringLiteral("valid_inputs")).toBool());
    EXPECT_EQ(json.value(QStringLiteral("mask_pixel_count")).toInt(), 8);
    EXPECT_EQ(json.value(QStringLiteral("valid_pixel_count")).toInt(), 7);
    EXPECT_EQ(
        json.value(QStringLiteral("native_valid_pixel_count")).toInt(), 5);
    EXPECT_EQ(
        json.value(QStringLiteral("repaired_valid_pixel_count")).toInt(), 2);
    const QJsonObject histogram = json.value(
        QStringLiteral("geometry_support_histogram")).toObject();
    EXPECT_EQ(histogram.value(QStringLiteral("support_0")).toInt(), 1);
    EXPECT_EQ(histogram.value(QStringLiteral("support_2")).toInt(), 2);
    EXPECT_EQ(histogram.value(QStringLiteral("support_5_plus")).toInt(), 2);
    EXPECT_NEAR(
        json.value(QStringLiteral("inverse_depth_spread_p50")).toDouble(),
        0.004,
        1.0e-6);
    EXPECT_NEAR(
        json.value(QStringLiteral("inverse_depth_spread_p95")).toDouble(),
        0.0067,
        1.0e-6);
}

TEST(DepthFrameQualityGateTest, MakesOrbitalConsistencyLossAuxiliaryBeforeCollapse)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    input.sourceViewCount = 2;
    input.validCoverage = 0.20f;
    input.largestComponentRatio = 0.90f;
    input.meanConfidence = 0.80f;
    input.multiViewConsistency = 0.80f;
    input.validWithinMaskRatio = 0.90f;
    input.outputFilterRetentionRatio = 1.0f;
    input.consistencyRetentionRatio = 0.70f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_EQ(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("depth_consistency_collapse")),
              decision.reasons.end());
    EXPECT_NE(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("depth_consistency_coverage_loss")),
              decision.reasons.end());

    input.consistencyRetentionRatio = 0.08f;
    const auto collapsed = xjw::mvs::evaluateDepthFrame(input);
    EXPECT_EQ(collapsed.acceptance, xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(collapsed.reasons.begin(),
                        collapsed.reasons.end(),
                        std::string("depth_consistency_collapse")),
              collapsed.reasons.end());
}

TEST(DepthFrameQualityGateTest, KeepsConsistencyAndFusionPostprocessLossDistinct)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    input.sourceViewCount = 4;
    input.validCoverage = 0.20f;
    input.largestComponentRatio = 0.90f;
    input.meanConfidence = 0.80f;
    input.multiViewConsistency = 0.95f;
    input.outputFilterRetentionRatio = 1.0f;
    input.consistencyRetentionRatio = 0.95f;
    input.fusionPostprocessRetentionRatio = 0.85f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("fusion_postprocess_coverage_loss")),
              decision.reasons.end());
    EXPECT_EQ(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("depth_consistency_coverage_loss")),
              decision.reasons.end());
}

TEST(DepthGeometryConsistencyTest,
     FinalizesAdaptiveEvidenceMapsWithoutDiscardingPrefilterHypotheses)
{
    cv::Mat prefilter_depth(1, 3, CV_32FC1, cv::Scalar(2.0f));
    prefilter_depth.at<float>(0, 1) = 0.0f;
    auto accumulator_maps =
        xjw::mvs::makeAdaptiveGeometryEvidenceAccumulatorMaps(
            prefilter_depth.size());

    xjw::mvs::AdaptiveGeometryEvidenceAccumulator supported;
    xjw::mvs::AdaptiveGeometryEvidenceObservation comparable;
    comparable.evidenceClass =
        xjw::mvs::AdaptiveGeometryEvidenceClass::Comparable;
    comparable.worldPixelFootprint = 0.1f;
    supported.add(comparable);
    accumulator_maps.positiveSupport.at<float>(0, 0) =
        supported.positiveSupport;
    accumulator_maps.squaredPositiveSupport.at<float>(0, 0) =
        supported.squaredPositiveSupport;
    accumulator_maps.conflict.at<float>(0, 0) = supported.conflict;
    accumulator_maps.observable.at<float>(0, 0) = supported.observable;

    xjw::mvs::AdaptiveGeometryEvidenceAccumulator contradicted;
    xjw::mvs::AdaptiveGeometryEvidenceObservation contradiction;
    contradiction.evidenceClass =
        xjw::mvs::AdaptiveGeometryEvidenceClass::Contradictory;
    contradicted.add(contradiction);
    accumulator_maps.positiveSupport.at<float>(0, 2) =
        contradicted.positiveSupport;
    accumulator_maps.squaredPositiveSupport.at<float>(0, 2) =
        contradicted.squaredPositiveSupport;
    accumulator_maps.conflict.at<float>(0, 2) = contradicted.conflict;
    accumulator_maps.observable.at<float>(0, 2) = contradicted.observable;

    const auto maps = xjw::mvs::makeAdaptiveGeometryEvidenceMaps(
        prefilter_depth, accumulator_maps);
    ASSERT_FALSE(maps.supportWeight.empty());
    EXPECT_GT(maps.supportWeight.at<float>(0, 0),
              maps.supportWeight.at<float>(0, 2));
    EXPECT_FLOAT_EQ(maps.effectiveViewCount.at<float>(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(maps.effectiveViewCount.at<float>(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(maps.conflictRatio.at<float>(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(maps.supportWeight.at<float>(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(maps.effectiveViewCount.at<float>(0, 1), 0.0f);
}

TEST(DepthFrameLifecycleTest,
     ValidationOnlyFrameStillParticipatesInConsistencyBeforeFinalFusion)
{
    xjw::mvs::DepthFrameResult frame;
    frame.success = true;
    frame.qualityDecision.acceptance =
        xjw::mvs::DepthFrameAcceptance::ValidationOnly;

    EXPECT_TRUE(frame.eligibleForConsistencyCheck());
    EXPECT_TRUE(frame.eligibleAsConsistencySource());
    EXPECT_FALSE(frame.eligibleForFusion());

    frame.qualityDecision.acceptance =
        xjw::mvs::DepthFrameAcceptance::Rejected;
    EXPECT_FALSE(frame.eligibleForConsistencyCheck());
    EXPECT_FALSE(frame.eligibleAsConsistencySource());
    EXPECT_FALSE(frame.eligibleForFusion());
}

TEST(DepthFrameLifecycleTest,
     FrozenBatchPlanRetainsLateFrameSourcesAfterEarlierRejections)
{
    std::vector<xjw::mvs::DepthFrameResult> frames(5);
    for (auto &frame : frames)
    {
        frame.success = true;
        frame.qualityDecision.acceptance =
            xjw::mvs::DepthFrameAcceptance::Accepted;
    }
    frames[4].sourceViewIndices = {0, 1, 2, 3};

    const auto frozen_plans = xjw::mvs::freezeDepthConsistencySourcePlans(
        frames, 5, xjw::mvs::MvsSceneProfile::OrbitalObject, 4);
    ASSERT_EQ(frozen_plans.size(), 5U);
    EXPECT_EQ(frozen_plans[4].geometrySourceViewIndices,
              (std::vector<int>{0, 1, 2, 3}));

    for (int source_index = 3; source_index >= 0; --source_index)
    {
        frames[static_cast<std::size_t>(source_index)]
            .qualityDecision.acceptance =
            xjw::mvs::DepthFrameAcceptance::Rejected;
    }
    const auto recomputed_plans =
        xjw::mvs::freezeDepthConsistencySourcePlans(
            frames, 5, xjw::mvs::MvsSceneProfile::OrbitalObject, 4);

    EXPECT_TRUE(recomputed_plans[4].geometrySourceViewIndices.empty());
    EXPECT_EQ(frozen_plans[4].geometrySourceViewIndices,
              (std::vector<int>{0, 1, 2, 3}));
}

TEST(DepthFrameQualityGateTest, MakesMarginalProjectMaskCoverageValidationOnly)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    input.sourceViewCount = 3;
    input.validCoverage = 0.20f;
    input.largestComponentRatio = 0.90f;
    input.meanConfidence = 0.80f;
    input.multiViewConsistency = 0.80f;
    input.hasProjectSupportMask = true;
    input.validWithinMaskRatio = 0.76f;
    input.outputFilterRetentionRatio = 0.95f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("insufficient_mask_normalized_coverage")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     IgnoresProjectMaskCoverageGateForTechnicalMask)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    input.sourceViewCount = 5;
    input.validCoverage = 0.50f;
    input.largestComponentRatio = 0.90f;
    input.meanConfidence = 0.80f;
    input.multiViewConsistency = 0.70f;
    // A content/undistortion validity mask defines the measurement domain but
    // is not a semantic project support request.
    input.hasProjectSupportMask = false;
    input.validWithinMaskRatio = 0.50f;
    input.outputFilterRetentionRatio = 1.0f;
    input.consistencyRetentionRatio = 0.58f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance, xjw::mvs::DepthFrameAcceptance::Accepted);
    EXPECT_EQ(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("insufficient_mask_normalized_coverage")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest, MakesLowAerialConsistencyValidationOnlyBeforeCollapse)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    input.sourceViewCount = 5;
    input.validCoverage = 0.30f;
    input.largestComponentRatio = 0.90f;
    input.meanConfidence = 0.70f;
    input.multiViewConsistency = 0.34f;
    input.outputFilterRetentionRatio = 1.0f;
    input.consistencyRetentionRatio = 0.34f;

    auto decision = xjw::mvs::evaluateDepthFrame(input);
    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_EQ(std::find(decision.reasons.begin(),
                        decision.reasons.end(),
                        std::string("depth_consistency_collapse")),
              decision.reasons.end());

    input.consistencyRetentionRatio = 0.15f;
    decision = xjw::mvs::evaluateDepthFrame(input);
    EXPECT_EQ(decision.acceptance, xjw::mvs::DepthFrameAcceptance::Rejected);
}

TEST(DepthFrameQualityGateTest, AcceptsStableOrbitalObjectWithPartialCoverage)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    input.sourceViewCount = 4;
    input.validCoverage = 0.18f;
    input.largestComponentRatio = 0.82f;
    input.meanConfidence = 0.76f;
    input.multiViewConsistency = 0.73f;

    const xjw::mvs::DepthFrameQualityDecision decision =
        xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance, xjw::mvs::DepthFrameAcceptance::Accepted);
}

TEST(DepthFrameQualityGateTest, AcceptsAerialEdgeFrameWithModerateConsistency)
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    input.sourceViewCount = 5;
    input.validCoverage = 0.38f;
    input.largestComponentRatio = 0.94f;
    input.meanConfidence = 0.54f;
    input.multiViewConsistency = 0.516f;
    input.depthAtSearchBoundaryRatio = 0.08f;

    const xjw::mvs::DepthFrameQualityDecision decision =
        xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance, xjw::mvs::DepthFrameAcceptance::Accepted);

    input.sourceViewCount = 8;
    const xjw::mvs::DepthFrameQualityDecision interior_decision =
        xjw::mvs::evaluateDepthFrame(input);
    EXPECT_EQ(interior_decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
}

TEST(DepthFrameQualityGateTest, CalibratesConfidenceAndCapsFilterViews)
{
    xjw::mvs::DepthConfidenceComponents components;
    components.photometric = 0.90f;
    components.support = 0.80f;
    components.uniqueness = 0.75f;
    components.geometry = 0.70f;
    components.texture = 0.60f;

    EXPECT_NEAR(xjw::mvs::calibrateDepthConfidence(components), 0.755f, 2.0e-3f);

    const xjw::mvs::DepthFilterSettings settings =
        xjw::mvs::depthFilterSettings(xjw::mvs::DepthFilterMode::Aggressive, 2);
    EXPECT_EQ(settings.minComponentArea, 64);
    EXPECT_FLOAT_EQ(settings.localDepthOutlierRelThreshold, 0.15f);
    EXPECT_EQ(settings.minConsistentViews, 2);
    EXPECT_EQ(xjw::mvs::minimumDepthConsistencySourceConfirmations(
                  xjw::mvs::DepthFilterMode::Mild, 6),
              1);
    EXPECT_EQ(xjw::mvs::minimumDepthConsistencySourceConfirmations(
                  xjw::mvs::MvsSceneProfile::OrbitalObject,
                  xjw::mvs::DepthFilterMode::Mild,
                  6),
              2);
    EXPECT_EQ(xjw::mvs::minimumDepthConsistencySourceConfirmations(
                  xjw::mvs::MvsSceneProfile::AerialTerrain,
                  xjw::mvs::DepthFilterMode::Mild,
                  6),
              1);
    EXPECT_EQ(xjw::mvs::minimumDepthConsistencySourceConfirmations(
                  xjw::mvs::DepthFilterMode::Moderate, 6),
              2);
    EXPECT_EQ(xjw::mvs::minimumDepthConsistencySourceConfirmations(
                  xjw::mvs::DepthFilterMode::Aggressive, 6),
              3);
    EXPECT_EQ(xjw::mvs::minimumDepthConsistencySourceConfirmations(
                  xjw::mvs::DepthFilterMode::Moderate, 1),
              0);
}

TEST(DepthFrameQualityGateTest, ScalesMultiViewConsistencyThresholdByFilterMode)
{
    using xjw::mvs::depthConsistencyRelativeThreshold;
    using xjw::mvs::DepthFilterMode;
    using xjw::mvs::MvsSceneProfile;

    EXPECT_FLOAT_EQ(depthConsistencyRelativeThreshold(
                        MvsSceneProfile::OrbitalObject, 16, DepthFilterMode::Mild),
                    0.0125f);
    EXPECT_FLOAT_EQ(depthConsistencyRelativeThreshold(
                        MvsSceneProfile::OrbitalObject, 16, DepthFilterMode::Moderate),
                    0.008f);
    EXPECT_FLOAT_EQ(depthConsistencyRelativeThreshold(
                        MvsSceneProfile::OrbitalObject, 16, DepthFilterMode::Aggressive),
                    0.005f);
    EXPECT_FLOAT_EQ(depthConsistencyRelativeThreshold(
                        MvsSceneProfile::AerialTerrain, 9, DepthFilterMode::Moderate),
                    0.015f);
    EXPECT_FLOAT_EQ(depthConsistencyRelativeThreshold(
                        MvsSceneProfile::OrbitalObject, 2, DepthFilterMode::Mild),
                    0.10f);
    EXPECT_FLOAT_EQ(depthConsistencyRelativeThreshold(
                        MvsSceneProfile::OrbitalObject, 2, DepthFilterMode::Moderate),
                    0.06f);
}

TEST(DepthConsistencyCacheTest, EvictsUnpinnedFramesWithinByteBudget)
{
    constexpr std::size_t frame_bytes = 12 * 16 * sizeof(float);
    int load_count = 0;
    xjw::mvs::DepthConsistencyCache cache(
        [&load_count](int frame_index,
                      xjw::mvs::DepthConsistencyFrame &frame,
                      std::string *)
        {
            ++load_count;
            frame.frameIndex = frame_index;
            frame.depth = cv::Mat(12, 16, CV_32F, cv::Scalar(10.0f + frame_index));
            return true;
        },
        frame_bytes * 2);

    std::string error;
    auto reference = cache.acquire(0, &error);
    ASSERT_TRUE(reference) << error;
    {
        auto source = cache.acquire(1, &error);
        ASSERT_TRUE(source) << error;
    }
    {
        auto source = cache.acquire(2, &error);
        ASSERT_TRUE(source) << error;
    }

    EXPECT_LE(cache.currentBytes(), frame_bytes * 2);
    EXPECT_LE(cache.peakBytes(), frame_bytes * 2);
    EXPECT_EQ(cache.acquire(0, &error).get(), reference.get());
    EXPECT_EQ(load_count, 3);
}

TEST(MvsPipelineTest, SparseSupportMaskTracksProjectedSparseStructure)
{
    constexpr int W = 120;
    constexpr int H = 90;
    constexpr double FOCAL = 80.0;
    constexpr float Z = 10.0f;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    xjw::mvs::CameraView view;
    view.imageWidth = W;
    view.imageHeight = H;
    view.camera.setIntrinsics(FOCAL, FOCAL, W * 0.5, H * 0.5);
    view.camera.setPose(
        std::array<double, 9>{I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8]},
        std::array<double, 3>{C[0], C[1], C[2]});
    view.camera.setAxisDirections(1, 1);
    view.camera.setDepthAxisFlipped(false);

    xjw::mvs::SparseCloud sparse;
    for (int py = 36; py <= 54; py += 6)
    {
        for (int px = 45; px <= 75; px += 6)
        {
            const float x = static_cast<float>((px - W * 0.5) * Z / FOCAL);
            const float y = static_cast<float>((py - H * 0.5) * Z / FOCAL);
            sparse.points.push_back({x, y, Z});
        }
    }

    const cv::Mat mask = xjw::mvs::DepthMapGenerator::buildSparseSupportMask({view}, sparse, 0, W, H);

    ASSERT_FALSE(mask.empty());
    EXPECT_GT(cv::countNonZero(mask(cv::Rect(35, 25, 50, 40))), 0);
    EXPECT_EQ(mask.at<uint8_t>(5, 5), 0);

    const float coverage = static_cast<float>(cv::countNonZero(mask)) / static_cast<float>(W * H);
    EXPECT_GT(coverage, 0.05f);
    EXPECT_LT(coverage, 0.90f);
}

TEST(MvsPipelineTest, ProjectedSparseSamplesFeedHintAndSupportReuse)
{
    constexpr int W = 120;
    constexpr int H = 90;
    constexpr double FOCAL = 80.0;
    constexpr float Z = 10.0f;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    xjw::mvs::CameraView view;
    view.imageWidth = W;
    view.imageHeight = H;
    view.camera.setIntrinsics(FOCAL, FOCAL, W * 0.5, H * 0.5);
    view.camera.setPose(
        std::array<double, 9>{I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8]},
        std::array<double, 3>{C[0], C[1], C[2]});
    view.camera.setAxisDirections(1, 1);
    view.camera.setDepthAxisFlipped(false);

    xjw::mvs::SparseCloud sparse;
    std::vector<size_t> indices;
    for (int py = 36; py <= 54; py += 6)
    {
        for (int px = 45; px <= 75; px += 6)
        {
            const float x = static_cast<float>((px - W * 0.5) * Z / FOCAL);
            const float y = static_cast<float>((py - H * 0.5) * Z / FOCAL);
            indices.push_back(sparse.points.size());
            sparse.points.push_back({x, y, Z});
        }
    }
    indices.push_back(sparse.points.size());
    sparse.points.push_back({0.0f, 0.0f, Z * 8.0f});

    const std::vector<xjw::mvs::ProjectedSparseDepthSample> samples =
        xjw::mvs::DepthMapGenerator::collectProjectedSparseDepthSamples(
            sparse, view.camera.normalizedForPositiveDepth(), W, H, indices);

    EXPECT_EQ(samples.size(), indices.size() - 1)
        << "The depth outlier should be excluded once before building hint/support rasters.";

    const cv::Mat hint = xjw::mvs::DepthMapGenerator::buildHintDepthFromProjectedSamples(
        0, W / 2, H / 2, samples);
    const cv::Mat support = xjw::mvs::DepthMapGenerator::buildSparseSupportMaskFromProjectedSamples(
        0, W, H, samples);

    ASSERT_FALSE(hint.empty());
    ASSERT_FALSE(support.empty());
    EXPECT_GT(cv::countNonZero(hint > 0), 0);
    EXPECT_GT(cv::countNonZero(support(cv::Rect(35, 25, 50, 40))), 0);
    EXPECT_EQ(support.at<uint8_t>(5, 5), 0);
}

TEST(MvsPipelineTest, SparseSeedDepthOverlayDoesNotPropagateAcrossFineHint)
{
    std::vector<xjw::mvs::ProjectedSparseDepthSample> samples;
    xjw::mvs::ProjectedSparseDepthSample sample;
    sample.uNorm = 0.5f;
    sample.vNorm = 0.5f;
    sample.depth = 10.0f;
    samples.push_back(sample);

    const cv::Mat propagated = xjw::mvs::DepthMapGenerator::buildHintDepthFromProjectedSamples(
        0, 64, 64, samples);
    const cv::Mat seedOnly = xjw::mvs::DepthMapGenerator::buildSparseSeedDepthFromProjectedSamples(
        0, 64, 64, samples);

    ASSERT_FALSE(propagated.empty());
    ASSERT_FALSE(seedOnly.empty());
    EXPECT_LT(cv::countNonZero(seedOnly > 0), cv::countNonZero(propagated > 0))
        << "Fine sparse overlay should stamp local seeds only; full propagation is reserved for coarse hints.";
    EXPECT_FLOAT_EQ(seedOnly.at<float>(32, 32), 10.0f);
    EXPECT_FLOAT_EQ(seedOnly.at<float>(0, 0), 0.0f);
}

TEST(MvsPipelineTest, SparseSupportSpanStampMatchesMorphologicalDilation)
{
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr int radius = 40;
    std::vector<xjw::mvs::ProjectedSparseDepthSample> samples;
    cv::Mat seeds(height, width, CV_8U, cv::Scalar(0));
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 5; ++column)
        {
            const int x = 70 + column * 120;
            const int y = 60 + row * 100;
            xjw::mvs::ProjectedSparseDepthSample sample;
            sample.uNorm = static_cast<float>(x) / static_cast<float>(width);
            sample.vNorm = static_cast<float>(y) / static_cast<float>(height);
            sample.depth = 10.0f;
            samples.push_back(sample);
            seeds.at<uint8_t>(y, x) = 255;
        }
    }
    const std::array<cv::Point, 4> border_points = {
        cv::Point(0, 0),
        cv::Point(width - 1, 0),
        cv::Point(0, height - 1),
        cv::Point(width - 1, height - 1)};
    for (const cv::Point &point : border_points)
    {
        xjw::mvs::ProjectedSparseDepthSample sample;
        sample.uNorm = static_cast<float>(point.x) / static_cast<float>(width);
        sample.vNorm = static_cast<float>(point.y) / static_cast<float>(height);
        sample.depth = 10.0f;
        samples.push_back(sample);
        seeds.at<uint8_t>(point.y, point.x) = 255;
    }
    samples.push_back(samples.front());
    xjw::mvs::ProjectedSparseDepthSample rejected_sample = samples.front();
    rejected_sample.depth = -1.0f;
    samples.push_back(rejected_sample);

    const cv::Mat support =
        xjw::mvs::DepthMapGenerator::buildSparseSupportMaskFromProjectedSamples(
            0, width, height, samples);
    ASSERT_FALSE(support.empty());

    cv::Mat expected;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(radius * 2 + 1, radius * 2 + 1));
    cv::dilate(seeds, expected, kernel);
    EXPECT_EQ(cv::countNonZero(support != expected), 0)
        << "Span stamping must preserve the exact sparse-support mask.";
}

TEST(MvsPipelineTest, SparseSupportPriorKeepsDepthAndSoftensConfidence)
{
    cv::Mat depth(3, 3, CV_32F, cv::Scalar(12.0f));
    cv::Mat confidence(3, 3, CV_32F, cv::Scalar(0.8f));
    cv::Mat support(3, 3, CV_8U, cv::Scalar(0));
    support.at<uint8_t>(1, 1) = 255;

    const int beforeValid = cv::countNonZero(depth > 0);

    xjw::mvs::DepthMapGenerator::applySparseSupportPrior(depth, confidence, support, 0);

    EXPECT_EQ(cv::countNonZero(depth > 0), beforeValid)
        << "Sparse support must not hard-clip PatchMatch depth pixels.";
    EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 12.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(1, 1), 12.0f);

    EXPECT_FLOAT_EQ(confidence.at<float>(1, 1), 0.8f);
    EXPECT_GT(confidence.at<float>(0, 0), 0.0f);
    EXPECT_LT(confidence.at<float>(0, 0), 0.8f);
}

TEST(MvsPipelineTest, LocalDepthOutlierFilterRemovesIsolatedDepthSpike)
{
    cv::Mat depth(7, 7, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    depth.at<float>(3, 3) = 30.0f;

    const int removed = xjw::mvs::DepthMapGenerator::removeLocalDepthOutliers(
        depth, confidence, 3, 0.25f, 0.50f, 0);

    EXPECT_EQ(removed, 1);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 3), 0.0f);
    EXPECT_FLOAT_EQ(confidence.at<float>(3, 3), 0.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 2), 10.0f);
}

TEST(MvsPipelineTest, LocalDepthOutlierFilterPreservesSmoothSlope)
{
    cv::Mat depth(7, 7, CV_32F);
    cv::Mat confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    for (int y = 0; y < depth.rows; ++y)
    {
        for (int x = 0; x < depth.cols; ++x)
        {
            depth.at<float>(y, x) = 10.0f + 0.10f * static_cast<float>(x) + 0.05f * static_cast<float>(y);
        }
    }

    const int beforeValid = cv::countNonZero(depth > 0);
    const int removed = xjw::mvs::DepthMapGenerator::removeLocalDepthOutliers(
        depth, confidence, 3, 0.25f, 0.50f, 0);

    EXPECT_EQ(removed, 0);
    EXPECT_EQ(cv::countNonZero(depth > 0), beforeValid);
}

TEST(MvsPipelineTest, FusionDepthPostprocessReportsConfidenceAndLocalOutliers)
{
    cv::Mat depth(7, 7, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    depth.at<float>(3, 3) = 30.0f;
    confidence.at<float>(0, 0) = 0.10f;

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.25f;
    config.enableLocalDepthOutlierFilter = true;
    config.localDepthOutlierKernelSize = 3;
    config.localDepthOutlierRelThresh = 0.25f;
    config.maxLocalDepthOutlierRemovalRatio = 0.50f;

    const xjw::mvs::DepthPostProcessStats stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(depth, confidence, config, 0, 4);

    EXPECT_EQ(stats.validBeforePostprocess, 49);
    EXPECT_EQ(stats.validAfterConfidenceFilter, 48);
    EXPECT_EQ(stats.confidenceRemoved, 1);
    EXPECT_EQ(stats.localDepthOutlierRemoved, 1);
    EXPECT_EQ(stats.speckleRemoved, 0);
    EXPECT_EQ(stats.edgeConfidenceRemoved, 0);
    EXPECT_EQ(stats.geomConsistencyRemoved, 0);
    EXPECT_EQ(stats.validAfterPostprocess, 47);
    EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 3), 0.0f);
    EXPECT_FLOAT_EQ(confidence.at<float>(3, 3), 0.0f);
}

TEST(MvsPipelineTest, NativeGridDisablesSubpixelThreeByThreeOutlierFootprint)
{
    cv::Mat native_depth(7, 7, CV_32F, cv::Scalar(10.0f));
    cv::Mat native_confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    native_depth.at<float>(3, 3) = 30.0f;
    cv::Mat unscaled_depth = native_depth.clone();
    cv::Mat unscaled_confidence = native_confidence.clone();

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.0f;
    config.enableAdaptiveConfidenceFilter = false;
    config.enableLocalDepthOutlierFilter = true;
    config.localDepthOutlierKernelSize = 3;
    config.localDepthOutlierRelThresh = 0.25f;
    config.maxLocalDepthOutlierRemovalRatio = 0.50f;
    config.enableSpeckleFilter = false;

    const auto native_stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
            native_depth,
            native_confidence,
            config,
            0,
            4,
            nullptr,
            nullptr,
            cv::Size(28, 28));
    const auto unscaled_stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
            unscaled_depth, unscaled_confidence, config, 0, 4);

    EXPECT_EQ(native_stats.localDepthOutlierRemoved, 0)
        << "A full-raster radius of one is subpixel on a ds4 grid.";
    EXPECT_FLOAT_EQ(native_depth.at<float>(3, 3), 30.0f);
    EXPECT_EQ(unscaled_stats.localDepthOutlierRemoved, 1)
        << "This locks the quality-changing behavior that scale awareness avoids.";
}

TEST(MvsPipelineTest, NativeGridScalesSpeckleAreaByRasterToGridArea)
{
    cv::Mat depth(10, 10, CV_32F, cv::Scalar(0.0f));
    cv::Mat confidence(10, 10, CV_32F, cv::Scalar(1.0f));
    depth(cv::Rect(0, 0, 2, 2)).setTo(10.0f);  // 4 grid px == 64 raster px.
    depth(cv::Rect(7, 0, 1, 3)).setTo(10.0f);  // 3 grid px == 48 raster px.
    depth(cv::Rect(3, 5, 4, 4)).setTo(10.0f);

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.0f;
    config.enableAdaptiveConfidenceFilter = false;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = true;
    config.minSpeckleComponentArea = 64;
    config.maxSpeckleRemovalRatio = 0.50f;

    const auto stats = xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
        depth,
        confidence,
        config,
        0,
        4,
        nullptr,
        nullptr,
        cv::Size(40, 40));

    EXPECT_EQ(stats.smallComponentRemoved, 3);
    EXPECT_EQ(cv::countNonZero(depth(cv::Rect(0, 0, 2, 2)) > 0.0f), 4);
    EXPECT_EQ(cv::countNonZero(depth(cv::Rect(7, 0, 1, 3)) > 0.0f), 0);
}

TEST(MvsPipelineTest, FusionDepthPostprocessRetainsOnlyGeometrySupportedLowConfidence)
{
    cv::Mat depth(5, 5, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(5, 5, CV_32F, cv::Scalar(0.9f));
    confidence.at<float>(1, 1) = 0.50f;
    confidence.at<float>(2, 2) = 0.50f;
    confidence.at<float>(3, 3) = 0.20f;

    xjw::mvs::DepthPostProcessEvidence evidence;
    evidence.geometrySupportCount = cv::Mat(5, 5, CV_16UC1, cv::Scalar(3));
    evidence.inverseDepthRelativeSpread = cv::Mat(5, 5, CV_32FC1, cv::Scalar(0.001f));
    evidence.adaptiveSupportWeight = cv::Mat(5, 5, CV_32FC1, cv::Scalar(0.60f));
    evidence.adaptiveEffectiveViewCount = cv::Mat(5, 5, CV_32FC1, cv::Scalar(3.0f));
    evidence.adaptiveConflictRatio = cv::Mat(5, 5, CV_32FC1, cv::Scalar(0.20f));
    evidence.geometrySupportCount.at<std::uint16_t>(2, 2) = 2;

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.60f;
    config.enableAdaptiveConfidenceFilter = false;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = false;

    const xjw::mvs::DepthPostProcessStats stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
            depth, confidence, config, 0, 4, nullptr, &evidence);

    EXPECT_EQ(stats.lowConfidenceCandidateCount, 3);
    EXPECT_EQ(stats.geometrySupportedLowConfidenceRetained, 1);
    EXPECT_EQ(stats.confidenceRemoved, 2);
    EXPECT_EQ(stats.validAfterConfidenceFilter, 23);
    EXPECT_FLOAT_EQ(depth.at<float>(1, 1), 10.0f);
    EXPECT_FLOAT_EQ(confidence.at<float>(1, 1), 0.50f);
    EXPECT_FLOAT_EQ(depth.at<float>(2, 2), 0.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 3), 0.0f);
}

TEST(MvsPipelineTest, FusionDepthPostprocessProtectsGeometrySupportedSilhouette)
{
    cv::Mat depth(9, 9, CV_32F, cv::Scalar(0.0f));
    cv::Mat confidence(9, 9, CV_32F, cv::Scalar(0.0f));
    depth(cv::Rect(2, 2, 5, 5)).setTo(10.0f);
    confidence(cv::Rect(2, 2, 5, 5)).setTo(0.30f);

    xjw::mvs::DepthPostProcessEvidence evidence;
    evidence.geometrySupportCount = cv::Mat(9, 9, CV_16UC1, cv::Scalar(2));
    evidence.inverseDepthRelativeSpread = cv::Mat(9, 9, CV_32FC1, cv::Scalar(0.005f));

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.60f;
    config.enableAdaptiveConfidenceFilter = false;
    config.enableGeometrySupportedLowConfidenceRetention = true;
    config.geometrySupportedMinimumObservationCount = 3;
    config.enableBoundaryAwareRetention = true;
    config.boundaryProtectionRadiusPixels = 0;
    config.boundaryMinimumConfidence = 0.25f;
    config.boundaryMinimumObservationCount = 2;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = false;

    const xjw::mvs::DepthPostProcessStats stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
            depth, confidence, config, 0, 4, nullptr, &evidence);

    EXPECT_EQ(stats.boundaryGeometryRetained, 16);
    EXPECT_FLOAT_EQ(depth.at<float>(2, 4), 10.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(4, 4), 0.0f);
}

TEST(MvsPipelineTest, NativeGridPreservesOneAuditedBoundaryShell)
{
    cv::Mat depth(9, 9, CV_32F, cv::Scalar(0.0f));
    cv::Mat confidence(9, 9, CV_32F, cv::Scalar(0.0f));
    depth(cv::Rect(2, 2, 5, 5)).setTo(10.0f);
    confidence(cv::Rect(2, 2, 5, 5)).setTo(0.30f);
    confidence.at<float>(4, 4) = 0.90f;

    xjw::mvs::DepthPostProcessEvidence evidence;
    evidence.geometrySupportCount = cv::Mat(9, 9, CV_16UC1, cv::Scalar(2));
    evidence.inverseDepthRelativeSpread =
        cv::Mat(9, 9, CV_32FC1, cv::Scalar(0.005f));

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.60f;
    config.enableAdaptiveConfidenceFilter = false;
    config.enableGeometrySupportedLowConfidenceRetention = true;
    config.geometrySupportedMinimumObservationCount = 3;
    config.enableBoundaryAwareRetention = true;
    config.boundaryProtectionRadiusPixels = 2;
    config.boundaryMinimumConfidence = 0.25f;
    config.boundaryMinimumObservationCount = 2;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = false;

    const auto stats = xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
        depth,
        confidence,
        config,
        0,
        4,
        nullptr,
        &evidence,
        cv::Size(36, 36));

    EXPECT_EQ(stats.boundaryGeometryRetained, 16)
        << "A reduced grid still needs one explicit silhouette shell; "
           "otherwise confidence filtering erases every boundary sample.";
    EXPECT_EQ(stats.confidenceRemoved, 8);
    EXPECT_EQ(stats.validAfterPostprocess, 17);
}

TEST(MvsPipelineTest, LocalDepthOutlierFilterPreservesSupportedThinDepthLayer)
{
    cv::Mat depth(7, 7, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    depth(cv::Rect(3, 3, 2, 2)).setTo(20.0f);

    const int removed = xjw::mvs::DepthMapGenerator::removeLocalDepthOutliers(
        depth, confidence, 3, 0.25f, 0.50f, 0);

    EXPECT_EQ(removed, 0);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 3), 20.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(4, 4), 20.0f);
}

TEST(MvsPipelineTest, DepthFrameReleasePreservesSourceSelectionDiagnostics)
{
    xjw::mvs::DepthFrameResult result;
    result.sourceViewIndices = {1, 3};
    result.requestedSourceViewCount = 6;
    result.sourceViewShortfall = 4;
    result.sourceViewShortfallReason = "pair_geometry_verification_failed";
    xjw::mvs::MvsSourcePlanEntry entry;
    entry.viewIndex = 1;
    entry.verifiedPairGeometry = true;
    entry.verificationStatus =
        xjw::mvs::MvsSourceVerificationStatus::Verified;
    result.sourceViewPlan.push_back(entry);
    result.depthMap = QSharedPointer<cv::Mat>::create(
        cv::Mat(4, 4, CV_32F, cv::Scalar(1.0f)));

    result.releasePixelStorage();

    EXPECT_FALSE(result.depthMap);
    EXPECT_EQ(result.requestedSourceViewCount, 6);
    EXPECT_EQ(result.sourceViewShortfall, 4);
    EXPECT_EQ(result.sourceViewShortfallReason,
              "pair_geometry_verification_failed");
    ASSERT_EQ(result.sourceViewPlan.size(), 1);
    EXPECT_TRUE(result.sourceViewPlan.front().verifiedPairGeometry);
}

TEST(MvsPipelineTest, FusionDepthPostprocessRaisesThresholdForLowConfidenceFullCoverage)
{
    cv::Mat depth(10, 10, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(10, 10, CV_32F, cv::Scalar(0.56f));
    confidence.at<float>(4, 4) = 0.80f;
    confidence.at<float>(5, 5) = 0.78f;

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.25f;
    config.enableAdaptiveConfidenceFilter = true;
    config.adaptiveFullCoverageThreshold = 0.95f;
    config.adaptiveLowMeanConfidenceThreshold = 0.65f;
    config.adaptiveStrictConfidenceThreshold = 0.65f;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = false;

    const xjw::mvs::DepthPostProcessStats stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(depth, confidence, config, 0, 4);

    EXPECT_FLOAT_EQ(stats.effectiveConfidenceThreshold, 0.65f);
    EXPECT_EQ(stats.validAfterConfidenceFilter, 2);
    EXPECT_EQ(stats.confidenceRemoved, 98);
    EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(4, 4), 10.0f);
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionEstimatesNormalsWhenNormalMapMissing)
{
    constexpr int W = 16;
    constexpr int H = 12;
    constexpr double FOCAL = 40.0;
    constexpr float DEPTH_VAL = 8.0f;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    for (int i = 0; i < 3; ++i)
    {
        frames[static_cast<size_t>(i)].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
        frames[static_cast<size_t>(i)].cameraModel = makeMvsCamera(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
        frames[static_cast<size_t>(i)].imgW = W;
        frames[static_cast<size_t>(i)].imgH = H;
    }
    frames[0].sourceImageIndices = {1, 2};

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    ASSERT_GT(pts.size(), 0u);
    const xjw::mvs::FusedPoint &p = pts[pts.size() / 2];
    const float normalLength = std::sqrt(p.nx * p.nx + p.ny * p.ny + p.nz * p.nz);
    EXPECT_GT(normalLength, 0.90f)
        << "Production dense clouds must not write all-zero normals into PLY output";
}

TEST(MvsPipelineTest, ContentMaskSkipsNearlyFullAerialFrame)
{
    cv::Mat gray(120, 200, CV_8U, cv::Scalar(122));
    cv::Mat mask = xjw::mvs::DepthMapGenerator::buildContentMask(gray);

    EXPECT_TRUE(mask.empty())
        << "Nearly full-content aerial frames should skip the content mask instead of filtering no pixels";
}

TEST(MvsPipelineTest, ContentMaskKeepsRealBlackBorderMask)
{
    cv::Mat gray(120, 200, CV_8U, cv::Scalar(0));
    gray(cv::Rect(35, 25, 130, 70)) = cv::Scalar(125);

    cv::Mat mask = xjw::mvs::DepthMapGenerator::buildContentMask(gray);

    ASSERT_FALSE(mask.empty());
    EXPECT_EQ(mask.at<uint8_t>(5, 5), 0);
    EXPECT_GT(mask.at<uint8_t>(60, 100), 0);

    const float coverage = static_cast<float>(cv::countNonZero(mask)) / static_cast<float>(mask.rows * mask.cols);
    EXPECT_GT(coverage, 0.20f);
    EXPECT_LT(coverage, 0.80f);
}

TEST(MvsPipelineTest, OrbitalProjectMaskRefinementCarvesInteriorOpeningAndProtectsBoundary)
{
    cv::Mat gray(120, 180, CV_8U, cv::Scalar(0));
    cv::rectangle(gray, cv::Rect(25, 20, 130, 80), cv::Scalar(150), cv::FILLED);
    cv::rectangle(gray, cv::Rect(70, 42, 40, 36), cv::Scalar(0), cv::FILLED);
    cv::rectangle(gray, cv::Rect(25, 45, 5, 30), cv::Scalar(0), cv::FILLED);

    cv::Mat project_mask(120, 180, CV_8UC1, cv::Scalar(0));
    cv::rectangle(project_mask, cv::Rect(25, 20, 130, 80), cv::Scalar(255), cv::FILLED);

    bool refined = false;
    float retained_ratio = 0.0f;
    const cv::Mat result = xjw::mvs::DepthMapGenerator::refineOrbitalProjectValidMask(
        gray, project_mask, &refined, &retained_ratio);

    ASSERT_TRUE(refined);
    EXPECT_EQ(result.at<uint8_t>(60, 90), 0);
    EXPECT_EQ(result.at<uint8_t>(60, 27), 255);
    EXPECT_EQ(result.at<uint8_t>(30, 40), 255);
    EXPECT_GT(retained_ratio, 0.75f);
    EXPECT_LT(retained_ratio, 1.0f);
}

TEST(MvsPipelineTest, OrbitalProjectMaskRefinementKeepsMaskOnBrightBackground)
{
    cv::Mat gray(120, 180, CV_8U, cv::Scalar(220));
    cv::rectangle(gray, cv::Rect(25, 20, 130, 80), cv::Scalar(150), cv::FILLED);
    cv::rectangle(gray, cv::Rect(70, 42, 40, 36), cv::Scalar(0), cv::FILLED);

    cv::Mat project_mask(120, 180, CV_8UC1, cv::Scalar(0));
    cv::rectangle(project_mask, cv::Rect(25, 20, 130, 80), cv::Scalar(255), cv::FILLED);

    bool refined = true;
    const cv::Mat result = xjw::mvs::DepthMapGenerator::refineOrbitalProjectValidMask(
        gray, project_mask, &refined);

    EXPECT_FALSE(refined);
    EXPECT_EQ(cv::countNonZero(result != project_mask), 0);
}

TEST(MvsPipelineTest, CudaRetryDownsampleIncreasesUntilGpuFriendlyScale)
{
    xjw::mvs::PatchMatchConfig cfg;
    cfg.downsampleFactor = 2;

    cfg = xjw::mvs::DepthMapGenerator::nextCudaRetryPatchMatchConfig(cfg, 6000, 4000);
    EXPECT_EQ(cfg.downsampleFactor, 3);

    cfg = xjw::mvs::DepthMapGenerator::nextCudaRetryPatchMatchConfig(cfg, 6000, 4000);
    EXPECT_EQ(cfg.downsampleFactor, 4);

    cfg = xjw::mvs::DepthMapGenerator::nextCudaRetryPatchMatchConfig(cfg, 6000, 4000);
    EXPECT_EQ(cfg.downsampleFactor, 6);
}

// ---------------------------------------------------------------------------
// Test 3: DenseCloudBuilder CPU 反投影基本正确性
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, DenseCloudBuilderUnprojectBasic)
{
    constexpr int W = 32, H = 24;
    constexpr float DEPTH_VAL = 5.0f;
    constexpr double FOCAL = 30.0;

    cv::Mat depth(H, W, CV_32F, cv::Scalar(DEPTH_VAL));

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0};
    auto cam = makeMvsCamera(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);

    xjw::mvs::DenseCloudOptions opt;
    opt.minDepth = 0.1f;
    opt.maxDepth = 100.0f;

    auto pts = xjw::mvs::DenseCloudBuilder::unproject(depth, cv::Mat(), cam, cv::Mat(), opt);

    ASSERT_EQ(static_cast<int>(pts.size()), W * H);

    // 中心像素应精确落在 Z=DEPTH_VAL
    // 中心像素 (W/2, H/2) 对应 X_cam = 0, Y_cam = 0, Z_cam = DEPTH_VAL
    double cx_pix = W / 2.0 - 0.5; // 像素中心
    double cy_pix = H / 2.0 - 0.5;

    // 找最近中心点
    float minDist = 1e9f;
    xjw::mvs::DensePoint closest{};
    for (auto &p : pts)
    {
        float dx = p.x, dy = p.y;
        float d = std::sqrt(dx*dx + dy*dy);
        if (d < minDist) { minDist = d; closest = p; }
    }
    EXPECT_NEAR(closest.z, DEPTH_VAL, 0.5f)
        << "Center pixel depth should unproject to ~DEPTH_VAL";
    EXPECT_NEAR(closest.x, 0.0f, 0.5f)
        << "Center pixel X should be ~0 in camera space";
}

// ---------------------------------------------------------------------------
// Test 4: PatchMatch → DenseCloudBuilder 端到端，点云非空
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, PatchMatchToDenseCloudEndToEnd)
{
    constexpr int W = 80, H = 60;
    constexpr double FOCAL = 64.0, BASELINE = 1.0, DISP = 6;

    cv::Mat ref = makeSyntheticGray(W, H);
    cv::Mat src = makeShifted(ref, DISP);
    cv::GaussianBlur(src, src, cv::Size(3,3), 0.0);

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0}, C1[3]={BASELINE,0,0};
    auto refCam = makeMvsCamera(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);
    auto srcCam = makeMvsCamera(FOCAL, FOCAL, W*0.5, H*0.5, I, C1);

    xjw::mvs::PatchMatchConfig cfg;
    cfg.backend           = xjw::mvs::PatchMatchBackend::Cpu;
    cfg.downsampleFactor  = 2;
    cfg.patchHalf         = 2;
    cfg.confidenceThresh  = 0.10f;
    cfg.doMedianBlur      = true;
    cfg.medianKernelSize  = 3;
    cfg.doBilateralFilter = false;

    cv::Mat depth;
    bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
        ref, {src}, refCam, {srcCam}, 4.0f, 20.0f, cfg, depth);
    ASSERT_TRUE(ok);

    xjw::mvs::DenseCloudOptions opt;
    opt.minDepth = 1.0f;
    opt.maxDepth = 50.0f;
    auto pts = xjw::mvs::DenseCloudBuilder::unproject(depth, cv::Mat(), refCam, cv::Mat(), opt);

    EXPECT_GT(static_cast<int>(pts.size()), 0)
        << "End-to-end CPU MVS pipeline should produce at least one 3D point";

    // 深度应在合理范围内
    for (auto &p : pts)
    {
        EXPECT_GT(p.z, 0.0f);
        EXPECT_LT(p.z, 50.0f);
    }
}
