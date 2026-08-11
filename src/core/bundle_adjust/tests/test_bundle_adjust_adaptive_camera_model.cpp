#include <gtest/gtest.h>

#include "BundleAdjustAdaptiveCameraModel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace
{

using Vec3 = std::array<double, 3>;

Vec3 subtract(const Vec3 &lhs, const Vec3 &rhs)
{
    return {{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]}};
}

Vec3 cross(const Vec3 &lhs, const Vec3 &rhs)
{
    return {{lhs[1] * rhs[2] - lhs[2] * rhs[1],
             lhs[2] * rhs[0] - lhs[0] * rhs[2],
             lhs[0] * rhs[1] - lhs[1] * rhs[0]}};
}

Vec3 normalize(Vec3 value)
{
    const double length = std::sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    for (double &component : value)
    {
        component /= length;
    }
    return value;
}

xjw::Camera makeLookAtCamera(
    const Vec3 &center,
    const Vec3 &target,
    double focal_length = 800.0)
{
    const Vec3 forward = normalize(subtract(target, center));
    const Vec3 reference = std::abs(forward[2]) < 0.90
        ? Vec3{{0.0, 0.0, 1.0}}
        : Vec3{{0.0, 1.0, 0.0}};
    const Vec3 right = normalize(cross(reference, forward));
    const Vec3 up = normalize(cross(forward, right));

    xjw::Camera camera;
    camera.setIntrinsics(focal_length, focal_length, 512.0, 384.0);
    camera.setPose({{right[0], up[0], forward[0],
                     right[1], up[1], forward[1],
                     right[2], up[2], forward[2]}},
                   center);
    return camera;
}

std::vector<xjw::BATrack> makeTracks(
    const std::vector<xjw::Camera> &cameras,
    const std::vector<Vec3> &points,
    int imageWidth = 1024,
    int imageHeight = 768)
{
    std::vector<xjw::BATrack> tracks;
    for (const Vec3 &point : points)
    {
        xjw::BATrack track;
        track.initialPoint = point;
        for (std::size_t cameraIndex = 0;
             cameraIndex < cameras.size();
             ++cameraIndex)
        {
            const double world[3] = {point[0], point[1], point[2]};
            double pixel[2] = {0.0, 0.0};
            if (!cameras[cameraIndex].projectWorldPoint(world, pixel) ||
                pixel[0] < 16.0 || pixel[0] > imageWidth - 16.0 ||
                pixel[1] < 16.0 || pixel[1] > imageHeight - 16.0)
            {
                continue;
            }
            track.observations.push_back(xjw::BAObservation{
                static_cast<int>(cameraIndex), pixel[0], pixel[1], 1.0});
        }
        if (track.observations.size() >= 3)
        {
            tracks.push_back(std::move(track));
        }
    }
    return tracks;
}

std::vector<Vec3> aerialPoints()
{
    std::vector<Vec3> points;
    for (int row = -8; row <= 8; ++row)
    {
        for (int column = -10; column <= 10; ++column)
        {
            points.push_back({{
                static_cast<double>(column) * 0.65,
                static_cast<double>(row) * 0.60,
                0.15 * std::sin(column * 0.35) * std::cos(row * 0.30)}});
        }
    }
    return points;
}

std::vector<xjw::Camera> aerialCameras()
{
    std::vector<xjw::Camera> cameras;
    for (int row = -2; row <= 2; ++row)
    {
        for (int column = -2; column <= 2; ++column)
        {
            const Vec3 center{{column * 2.2, row * 2.2, 15.0}};
            cameras.push_back(makeLookAtCamera(
                center,
                {{center[0], center[1], 0.0}}));
        }
    }
    return cameras;
}

std::vector<xjw::Camera> weaklyParallelAerialCameras()
{
    std::vector<xjw::Camera> cameras;
    for (int row = -2; row <= 2; ++row)
    {
        for (int column = -2; column <= 2; ++column)
        {
            const Vec3 center{{column * 2.2, row * 2.2, 15.0}};
            const double offset = ((row + column) & 1) == 0 ? 5.0 : -5.0;
            cameras.push_back(makeLookAtCamera(
                center,
                {{center[0] + offset, center[1], 0.0}}));
        }
    }
    return cameras;
}

std::vector<Vec3> narrowAerialPoints()
{
    std::vector<Vec3> points;
    for (int row = -8; row <= 8; ++row)
    {
        for (int column = -10; column <= 10; ++column)
        {
            points.push_back({{
                static_cast<double>(column) * 0.035,
                static_cast<double>(row) * 0.035,
                0.01 * std::sin(column * 0.35) * std::cos(row * 0.30)}});
        }
    }
    return points;
}

std::vector<xjw::Camera> narrowAerialCameras()
{
    std::vector<xjw::Camera> cameras;
    for (int row = -2; row <= 2; ++row)
    {
        for (int column = -2; column <= 2; ++column)
        {
            const Vec3 center{{column * 0.12, row * 0.12, 15.0}};
            xjw::Camera camera = makeLookAtCamera(
                center,
                {{center[0], center[1], 0.0}},
                48000.0);
            camera.setIntrinsics(48000.0, 48000.0, 1939.0, 1444.0);
            cameras.push_back(camera);
        }
    }
    return cameras;
}

std::vector<Vec3> orbitalPoints()
{
    std::vector<Vec3> points;
    for (int depth = -2; depth <= 2; ++depth)
    {
        for (int row = -6; row <= 6; ++row)
        {
            for (int column = -6; column <= 6; ++column)
            {
                points.push_back({{
                    column * 0.78,
                    row * 0.68,
                    depth * 0.70 + 0.10 * std::sin(column + row)}});
            }
        }
    }
    return points;
}

std::vector<xjw::Camera> orbitalCameras()
{
    std::vector<xjw::Camera> cameras;
    constexpr int count = 24;
    constexpr double pi = 3.14159265358979323846;
    for (int index = 0; index < count; ++index)
    {
        const double angle = 2.0 * pi * index / count;
        const double height = 2.4 * std::sin(2.0 * angle);
        const Vec3 center{{12.0 * std::cos(angle),
                           12.0 * std::sin(angle),
                           height}};
        cameras.push_back(makeLookAtCamera(
            center,
            {{0.0, 0.0, 0.0}},
            620.0));
    }
    return cameras;
}

std::vector<Vec3> narrowOrbitalPoints()
{
    std::vector<Vec3> points;
    for (int depth = -2; depth <= 2; ++depth)
    {
        for (int row = -6; row <= 6; ++row)
        {
            for (int column = -6; column <= 6; ++column)
            {
                points.push_back({{
                    column * 0.06,
                    row * 0.055,
                    depth * 0.08 + 0.01 * std::sin(column + row)}});
            }
        }
    }
    return points;
}

std::vector<xjw::Camera> narrowOrbitalCameras()
{
    std::vector<xjw::Camera> cameras;
    constexpr int count = 24;
    constexpr double pi = 3.14159265358979323846;
    for (int index = 0; index < count; ++index)
    {
        const double angle = 2.0 * pi * index / count;
        const double height = 0.8 * std::sin(2.0 * angle);
        const Vec3 center{{12.0 * std::cos(angle),
                           12.0 * std::sin(angle),
                           height}};
        xjw::Camera camera = makeLookAtCamera(
            center,
            {{0.0, 0.0, 0.0}},
            48000.0);
        camera.setIntrinsics(48000.0, 48000.0, 1939.0, 1444.0);
        cameras.push_back(camera);
    }
    return cameras;
}

bool enabled(
    const xjw::BAAdaptiveCameraModelAssessment &assessment,
    xjw::BAIntrinsicParameter parameter)
{
    return assessment.enabled[static_cast<std::size_t>(parameter)];
}

} // namespace

TEST(BundleAdjustAdaptiveCameraModelTest,
     UnanchoredParallelAerialBlockKeepsIntrinsicsFixed)
{
    const std::vector<xjw::Camera> cameras = aerialCameras();
    const std::vector<xjw::BATrack> tracks = makeTracks(cameras, aerialPoints());
    ASSERT_GT(tracks.size(), 100u);

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks);

    ASSERT_TRUE(assessment.valid);
    EXPECT_GT(assessment.opticalAxisConcentration, 0.99);
    EXPECT_FALSE(assessment.hasAbsoluteGeometryConstraint);
    EXPECT_TRUE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::FocalLength));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK1));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::FocalAspectRatio));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::PrincipalPointX));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::PrincipalPointY));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK2));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK3));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::TangentialP1));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::TangentialP2));
    EXPECT_EQ(assessment.modelName, "fixed");
    EXPECT_EQ(
        assessment.reason,
        "unanchored_parallel_aerial_fixed_intrinsics_doming_guard");
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     WeaklyParallelUnanchoredAerialBlockKeepsIntrinsicsFixedFromFirstRound)
{
    const std::vector<xjw::Camera> cameras = weaklyParallelAerialCameras();
    const std::vector<xjw::BATrack> tracks = makeTracks(cameras, aerialPoints());
    ASSERT_GT(tracks.size(), 100u);

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks);

    ASSERT_TRUE(assessment.valid);
    EXPECT_GE(assessment.opticalAxisConcentration, 0.90);
    EXPECT_LT(assessment.opticalAxisConcentration, 0.97);
    EXPECT_FALSE(assessment.hasAbsoluteGeometryConstraint);
    EXPECT_TRUE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_EQ(xjw::enabledIntrinsicParameterCount(assessment.enabled), 0);
    EXPECT_EQ(assessment.modelName, "fixed");
    EXPECT_EQ(
        assessment.reason,
        "unanchored_parallel_aerial_fixed_intrinsics_doming_guard");

    xjw::BAOptions options;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = true;
    EXPECT_FALSE(xjw::applyAdaptiveCameraModel(assessment, &options));
    EXPECT_EQ(
        xjw::enabledIntrinsicParameterCount(
            options.sharedIntrinsicParameterMask),
        0);
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     ControlledParallelAerialBlockMayEstimateLowOrderDistortion)
{
    const std::vector<xjw::Camera> cameras = aerialCameras();
    std::vector<xjw::BATrack> tracks = makeTracks(cameras, aerialPoints());
    ASSERT_GT(tracks.size(), 100u);
    for (xjw::BATrack &track : tracks)
    {
        track.controlPointConstraints.push_back(
            {track.initialPoint, 0.02, 1.0, 0});
    }
    xjw::BAOptions options;
    options.enableControlPointConstraints = true;

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks, &options);

    ASSERT_TRUE(assessment.valid);
    EXPECT_TRUE(assessment.hasAbsoluteGeometryConstraint);
    EXPECT_FALSE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::FocalLength));
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK1));
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     NarrowFieldBlockUsesFieldNormalizedLowOrderDistortion)
{
    const std::vector<xjw::Camera> cameras = narrowAerialCameras();
    const std::vector<xjw::BATrack> tracks = makeTracks(
        cameras,
        narrowAerialPoints(),
        3878,
        2888);
    ASSERT_GT(tracks.size(), 100u);

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks);

    ASSERT_TRUE(assessment.valid);
    EXPECT_LT(assessment.normalizedRadiusP90, 0.10);
    EXPECT_GT(assessment.normalizedRadiusP90, 0.025);
    EXPECT_LT(assessment.peripheralRadiusThreshold, 0.10);
    EXPECT_GT(assessment.lowOrderDistortionScale, 1.0);
    EXPECT_GE(assessment.occupiedPeripheralSectors, 4);
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::FocalLength));
    EXPECT_TRUE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK1));
    EXPECT_EQ(assessment.modelName, "fixed");

    xjw::BAOptions options;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = true;
    EXPECT_FALSE(xjw::applyAdaptiveCameraModel(assessment, &options));
    EXPECT_DOUBLE_EQ(options.sharedLowOrderDistortionScale, 1.0);
    const double appliedScale = options.sharedLowOrderDistortionScale;
    EXPECT_FALSE(xjw::applyAdaptiveCameraModel(assessment, &options));
    EXPECT_DOUBLE_EQ(options.sharedLowOrderDistortionScale, appliedScale);
}

TEST(BundleAdjustAdaptiveCameraModelTest, InactiveObliqueCamerasDoNotChangeGeometry)
{
    std::vector<xjw::Camera> cameras = aerialCameras();
    const std::size_t activeCameraCount = cameras.size();
    const std::vector<xjw::BATrack> tracks = makeTracks(cameras, aerialPoints());
    const std::vector<xjw::Camera> inactiveObliqueCameras = orbitalCameras();
    cameras.insert(
        cameras.end(),
        inactiveObliqueCameras.begin(),
        inactiveObliqueCameras.end());

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks);

    ASSERT_TRUE(assessment.valid);
    EXPECT_EQ(
        assessment.activeCameraCount,
        static_cast<int>(activeCameraCount));
    EXPECT_GT(assessment.opticalAxisConcentration, 0.99);
    EXPECT_TRUE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_EQ(assessment.modelName, "fixed");
}

TEST(BundleAdjustAdaptiveCameraModelTest, ConvergentMultiHeightOrbitReleasesMoreParameters)
{
    const std::vector<xjw::Camera> cameras = orbitalCameras();
    const std::vector<xjw::BATrack> tracks = makeTracks(cameras, orbitalPoints());
    ASSERT_GT(tracks.size(), 200u);

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks);

    ASSERT_TRUE(assessment.valid);
    EXPECT_LT(assessment.opticalAxisConcentration, 0.20);
    EXPECT_GT(assessment.geometryStrength, 0.75);
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::FocalLength));
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK1));
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::FocalAspectRatio));
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK2))
        << "reliability="
        << assessment.reliability[static_cast<std::size_t>(
               xjw::BAIntrinsicParameter::RadialK2)]
        << " independence="
        << assessment.incrementalInformationScore[static_cast<std::size_t>(
               xjw::BAIntrinsicParameter::RadialK2)]
        << " sensitivity="
        << assessment.sensitivity[static_cast<std::size_t>(
               xjw::BAIntrinsicParameter::RadialK2)]
        << " radiusP90=" << assessment.normalizedRadiusP90
        << " sectors=" << assessment.occupiedPeripheralSectors;
    EXPECT_GT(
        xjw::enabledIntrinsicParameterCount(assessment.enabled),
        2);
    EXPECT_NE(assessment.modelName, "f+k1");
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     NarrowConvergentBlockCanReleaseTangentialDistortion)
{
    const std::vector<xjw::Camera> cameras = narrowOrbitalCameras();
    const std::vector<xjw::BATrack> tracks = makeTracks(
        cameras,
        narrowOrbitalPoints(),
        3878,
        2888);
    ASSERT_GT(tracks.size(), 200u);

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks);

    ASSERT_TRUE(assessment.valid);
    EXPECT_LT(assessment.normalizedRadiusP90, 0.10);
    EXPECT_GT(assessment.geometryStrength, 0.55);
    EXPECT_GE(assessment.occupiedPeripheralSectors, 6);
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK1));
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::TangentialP1))
        << "reliability="
        << assessment.reliability[static_cast<std::size_t>(
               xjw::BAIntrinsicParameter::TangentialP1)]
        << " sensitivity="
        << assessment.sensitivity[static_cast<std::size_t>(
               xjw::BAIntrinsicParameter::TangentialP1)]
        << " axisBalance=" << assessment.imageAxisBalance;
    EXPECT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::TangentialP2))
        << "reliability="
        << assessment.reliability[static_cast<std::size_t>(
               xjw::BAIntrinsicParameter::TangentialP2)]
        << " sensitivity="
        << assessment.sensitivity[static_cast<std::size_t>(
               xjw::BAIntrinsicParameter::TangentialP2)]
        << " axisBalance=" << assessment.imageAxisBalance;
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     NarrowFieldCeresCanEstimateLargeK1Coefficient)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    const std::vector<xjw::Camera> referenceCameras = narrowAerialCameras();
    std::vector<xjw::Camera> truthCameras = referenceCameras;
    for (xjw::Camera &camera : truthCameras)
    {
        camera.setDistortion(1.05, 0.0, 0.0, 0.0, 0.0);
    }
    std::vector<xjw::BATrack> tracks = makeTracks(
        truthCameras,
        narrowAerialPoints(),
        3878,
        2888);
    ASSERT_GT(tracks.size(), 100u);
    for (xjw::BATrack &track : tracks)
    {
        track.controlPointConstraints.push_back(
            {track.initialPoint, 1.0e-4, 1.0, 0});
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.allowBackendFallback = false;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedRadialDistortion = true;
    options.enableControlPointConstraints = true;
    options.enablePointFilter = false;
    options.stageSharedFocalRefinement = false;
    options.sharedIntrinsicReferenceCameras = referenceCameras;
    options.minSharedFocalScale = 1.0;
    options.maxSharedFocalScale = 1.0;
    options.maxIterations = 30;

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(referenceCameras, tracks, &options);
    ASSERT_TRUE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK1));
    ASSERT_TRUE(xjw::applyAdaptiveCameraModel(assessment, &options));
    ASSERT_GT(
        options.maxSharedRadialK1Abs * options.sharedLowOrderDistortionScale,
        1.05);

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(
        referenceCameras,
        tracks,
        options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_FALSE(result.refinedCameras.empty());
    EXPECT_NEAR(
        result.refinedCameras.front().distortion().radialK1,
        1.05,
        0.08);
}

TEST(BundleAdjustAdaptiveCameraModelTest, OpposingCollinearRaysRemainDegenerate)
{
    const std::vector<xjw::Camera> cameras{
        makeLookAtCamera({{-10.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}),
        makeLookAtCamera({{10.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}),
        makeLookAtCamera({{-14.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}),
    };
    std::vector<Vec3> points;
    for (int index = 0; index < 50; ++index)
    {
        points.push_back({{-0.8 + 1.6 * index / 49.0, 0.0, 0.0}});
    }

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, makeTracks(cameras, points));

    ASSERT_TRUE(assessment.valid);
    EXPECT_LT(assessment.medianTriangulationAngleDegrees, 1.0e-8);
    EXPECT_LT(assessment.geometryStrength, 0.15);
    EXPECT_FALSE(enabled(
        assessment, xjw::BAIntrinsicParameter::FocalAspectRatio));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::PrincipalPointX));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::PrincipalPointY));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK2));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK3));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::TangentialP1));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::TangentialP2));
}

TEST(BundleAdjustAdaptiveCameraModelTest, UnsupportedCalibrationGroupFreezesModel)
{
    std::vector<xjw::Camera> cameras = aerialCameras();
    std::vector<xjw::BATrack> tracks = makeTracks(cameras, aerialPoints());
    cameras.push_back(makeLookAtCamera(
        {{40.0, 0.0, 15.0}}, {{40.0, 0.0, 0.0}}));

    xjw::BAOptions options;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.cameraCalibrationGroupIds.resize(cameras.size(), 0);
    options.cameraCalibrationGroupIds.back() = 1;

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks, &options);

    ASSERT_TRUE(assessment.valid);
    EXPECT_EQ(
        assessment.reason,
        "calibration_group_conservative_intersection");
    EXPECT_EQ(xjw::enabledIntrinsicParameterCount(assessment.enabled), 0);
    EXPECT_EQ(assessment.modelName, "fixed");
}

TEST(BundleAdjustAdaptiveCameraModelTest, CalibrationGroupsUseConservativeIntersection)
{
    std::vector<xjw::Camera> cameras = aerialCameras();
    std::vector<xjw::BATrack> tracks = makeTracks(cameras, aerialPoints());
    const std::size_t aerialCameraCount = cameras.size();
    const std::vector<xjw::Camera> orbit = orbitalCameras();
    std::vector<xjw::BATrack> orbitTracks = makeTracks(orbit, orbitalPoints());
    cameras.insert(cameras.end(), orbit.begin(), orbit.end());
    for (xjw::BATrack &track : orbitTracks)
    {
        for (xjw::BAObservation &observation : track.observations)
        {
            observation.cameraIndex += static_cast<int>(aerialCameraCount);
        }
        tracks.push_back(std::move(track));
    }
    xjw::BAOptions options;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.cameraCalibrationGroupIds.resize(cameras.size(), 1);
    std::fill_n(
        options.cameraCalibrationGroupIds.begin(), aerialCameraCount, 0);

    const xjw::BAAdaptiveCameraModelAssessment assessment =
        xjw::assessAdaptiveCameraModel(cameras, tracks, &options);

    ASSERT_TRUE(assessment.valid);
    EXPECT_EQ(
        assessment.reason,
        "calibration_group_conservative_intersection");
    EXPECT_TRUE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::FocalLength));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK1));
    EXPECT_FALSE(enabled(
        assessment, xjw::BAIntrinsicParameter::FocalAspectRatio));
    EXPECT_FALSE(enabled(assessment, xjw::BAIntrinsicParameter::RadialK2));
}

TEST(BundleAdjustAdaptiveCameraModelTest, InsufficientEvidenceKeepsModelFixed)
{
    xjw::BAAdaptiveCameraModelAssessment assessment;
    xjw::BAOptions options;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;

    EXPECT_FALSE(xjw::applyAdaptiveCameraModel(assessment, &options));
    EXPECT_TRUE(options.useSharedIntrinsicParameterMask);
    EXPECT_FALSE(options.refineSharedFocalLength);
    EXPECT_FALSE(options.refineSharedFocalAspectRatio);
    EXPECT_FALSE(options.refineSharedPrincipalPoint);
    EXPECT_FALSE(options.refineSharedRadialDistortion);
    EXPECT_EQ(
        xjw::enabledIntrinsicParameterCount(options.sharedIntrinsicParameterMask),
        0);
}

TEST(BundleAdjustAdaptiveCameraModelTest, RespectsCallerIntrinsicParameterMask)
{
    xjw::BAAdaptiveCameraModelAssessment assessment;
    assessment.valid = true;
    assessment.enabled.fill(true);

    xjw::BAOptions options;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = true;
    options.useSharedIntrinsicParameterMask = true;
    options.sharedIntrinsicParameterMask.fill(false);
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)] = true;
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::RadialK1)] = true;

    ASSERT_TRUE(xjw::applyAdaptiveCameraModel(assessment, &options));
    EXPECT_EQ(
        xjw::enabledIntrinsicParameterCount(options.sharedIntrinsicParameterMask),
        2);
    EXPECT_TRUE(options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)]);
    EXPECT_TRUE(options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::RadialK1)]);
    EXPECT_FALSE(options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::RadialK2)]);

    xjw::BAOptions focalFrozen = options;
    focalFrozen.refineSharedFocalLength = true;
    focalFrozen.refineSharedRadialDistortion = true;
    focalFrozen.sharedIntrinsicParameterMask.fill(true);
    focalFrozen.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)] = false;

    EXPECT_FALSE(xjw::applyAdaptiveCameraModel(assessment, &focalFrozen));
    EXPECT_EQ(
        xjw::enabledIntrinsicParameterCount(
            focalFrozen.sharedIntrinsicParameterMask),
        0);
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     RestoresParametersDisabledAfterAnEarlierAdaptiveRound)
{
    std::vector<xjw::Camera> references = aerialCameras();
    std::vector<xjw::Camera> current = references;
    for (xjw::Camera &camera : current)
    {
        camera.setIntrinsics(840.0, 856.8, 524.0, 371.0);
        camera.setDistortion(-0.08, -0.04, 0.01, 0.002, -0.003);
    }
    xjw::BAIntrinsicParameterMask active{};
    active[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)] = true;

    ASSERT_TRUE(xjw::restoreInactiveAdaptiveIntrinsics(
        &current, references, active));
    ASSERT_EQ(current.size(), references.size());
    for (std::size_t index = 0; index < current.size(); ++index)
    {
        EXPECT_DOUBLE_EQ(current[index].focalX(), 840.0);
        EXPECT_DOUBLE_EQ(current[index].focalY(), 840.0);
        EXPECT_DOUBLE_EQ(
            current[index].principalX(), references[index].principalX());
        EXPECT_DOUBLE_EQ(
            current[index].principalY(), references[index].principalY());
        const xjw::Camera::Distortion distortion = current[index].distortion();
        const xjw::Camera::Distortion reference = references[index].distortion();
        EXPECT_DOUBLE_EQ(distortion.radialK1, reference.radialK1);
        EXPECT_DOUBLE_EQ(distortion.radialK2, reference.radialK2);
        EXPECT_DOUBLE_EQ(distortion.radialK3, reference.radialK3);
        EXPECT_DOUBLE_EQ(distortion.tangentialP1, reference.tangentialP1);
        EXPECT_DOUBLE_EQ(distortion.tangentialP2, reference.tangentialP2);
    }
}

TEST(BundleAdjustAdaptiveCameraModelTest, CeresHonorsIndividualIntrinsicMask)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    std::vector<xjw::Camera> cameras = orbitalCameras();
    for (std::size_t index = 0; index < cameras.size(); ++index)
    {
        xjw::Camera &camera = cameras[index];
        const double focal_x = camera.focalX();
        camera.setIntrinsics(
            focal_x,
            focal_x * (1.0 + 0.0005 * static_cast<double>(index)),
            camera.principalX() + 0.1 * static_cast<double>(index),
            camera.principalY() - 0.08 * static_cast<double>(index));
        camera.setDistortion(
            0.0,
            0.012 + 1.0e-5 * static_cast<double>(index),
            -0.018 - 2.0e-5 * static_cast<double>(index),
            0.001 + 1.0e-6 * static_cast<double>(index),
            -0.002 - 1.0e-6 * static_cast<double>(index));
    }
    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.allowBackendFallback = false;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedRadialDistortion = true;
    options.useSharedIntrinsicParameterMask = true;
    options.sharedIntrinsicParameterMask.fill(false);
    options.sharedIntrinsicParameterMask[
        static_cast<std::size_t>(xjw::BAIntrinsicParameter::FocalLength)] = true;
    options.sharedIntrinsicParameterMask[
        static_cast<std::size_t>(xjw::BAIntrinsicParameter::RadialK1)] = true;
    options.enablePointFilter = false;
    options.maxIterations = 8;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(
        cameras,
        makeTracks(cameras, orbitalPoints()),
        options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), cameras.size());
    EXPECT_EQ(result.selfCalibrationStagesRun, 2);
    EXPECT_LT(result.meanRmsAfter, 1.0e-5);
    for (std::size_t index = 0; index < cameras.size(); ++index)
    {
        const xjw::Camera &source = cameras[index];
        const xjw::Camera &refined = result.refinedCameras[index];
        const xjw::Camera::Distortion source_distortion = source.distortion();
        const xjw::Camera::Distortion refined_distortion = refined.distortion();
        EXPECT_DOUBLE_EQ(refined.principalX(), source.principalX());
        EXPECT_DOUBLE_EQ(refined.principalY(), source.principalY());
        EXPECT_DOUBLE_EQ(
            refined.focalY() / refined.focalX(),
            source.focalY() / source.focalX());
        EXPECT_DOUBLE_EQ(refined_distortion.radialK2, source_distortion.radialK2);
        EXPECT_DOUBLE_EQ(refined_distortion.radialK3, source_distortion.radialK3);
        EXPECT_DOUBLE_EQ(
            refined_distortion.tangentialP1,
            source_distortion.tangentialP1);
        EXPECT_DOUBLE_EQ(
            refined_distortion.tangentialP2,
            source_distortion.tangentialP2);
    }
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     FocalOnlyCeresUsesStableReferencePrior)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    std::vector<xjw::Camera> referenceCameras = aerialCameras();
    std::vector<xjw::Camera> truthCameras = referenceCameras;
    for (xjw::Camera &camera : truthCameras)
    {
        camera.setIntrinsics(1200.0, 1200.0, 512.0, 384.0);
    }
    std::vector<xjw::BATrack> tracks =
        makeTracks(truthCameras, aerialPoints());
    ASSERT_GT(tracks.size(), 100u);
    for (xjw::BATrack &track : tracks)
    {
        track.controlPointConstraints.push_back(
            {track.initialPoint, 1.0e-4, 1.0, 0});
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.allowBackendFallback = false;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.useSharedIntrinsicParameterMask = true;
    options.sharedIntrinsicParameterMask.fill(false);
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)] = true;
    options.sharedIntrinsicReferenceCameras = referenceCameras;
    options.minSharedFocalScale = 0.5;
    options.maxSharedFocalScale = 2.0;
    options.sharedFocalPriorSigma = 1.0e-8;
    options.enableControlPointConstraints = true;
    options.enablePointFilter = false;
    options.stageSharedFocalRefinement = false;
    options.maxIterations = 15;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(
        referenceCameras, tracks, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_FALSE(result.refinedCameras.empty());
    EXPECT_NEAR(result.refinedSharedFocalScale, 1.0, 1.0e-5);
    EXPECT_NEAR(result.refinedCameras.front().focalX(), 800.0, 0.01);
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     DistortionPriorRemainsAnchoredAcrossRounds)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    const std::vector<xjw::Camera> referenceCameras = aerialCameras();
    std::vector<xjw::Camera> truthCameras = referenceCameras;
    std::vector<xjw::Camera> warmCameras = referenceCameras;
    for (xjw::Camera &camera : truthCameras)
    {
        camera.setDistortion(-0.20, 0.0, 0.0, 0.0, 0.0);
    }
    for (xjw::Camera &camera : warmCameras)
    {
        camera.setDistortion(-0.10, 0.0, 0.0, 0.0, 0.0);
    }
    std::vector<xjw::BATrack> tracks =
        makeTracks(truthCameras, aerialPoints());
    ASSERT_GT(tracks.size(), 100u);
    for (xjw::BATrack &track : tracks)
    {
        track.controlPointConstraints.push_back(
            {track.initialPoint, 1.0e-4, 1.0, 0});
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.allowBackendFallback = false;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = false;
    options.useSharedIntrinsicParameterMask = true;
    options.sharedIntrinsicParameterMask.fill(false);
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)] = true;
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::RadialK1)] = true;
    options.sharedIntrinsicReferenceCameras = referenceCameras;
    options.minSharedFocalScale = 1.0;
    options.maxSharedFocalScale = 1.0;
    options.sharedRadialK1PriorSigma = 1.0e-8;
    options.enableControlPointConstraints = true;
    options.enablePointFilter = false;
    options.stageSharedFocalRefinement = false;
    options.maxIterations = 15;

    const xjw::BAResult first = xjw::BundleAdjust::optimizePoints(
        warmCameras, tracks, options);
    ASSERT_TRUE(first.solutionUsable) << first.backendMessage;
    ASSERT_FALSE(first.refinedCameras.empty());
    EXPECT_NEAR(first.refinedCameras.front().distortion().radialK1, 0.0, 1.0e-5);

    const xjw::BAResult second = xjw::BundleAdjust::optimizePoints(
        first.refinedCameras, tracks, options);
    ASSERT_TRUE(second.solutionUsable) << second.backendMessage;
    ASSERT_FALSE(second.refinedCameras.empty());
    EXPECT_NEAR(second.refinedCameras.front().distortion().radialK1, 0.0, 1.0e-5);
}

TEST(BundleAdjustAdaptiveCameraModelTest,
     StableIntrinsicReferencePreventsBoundsFromCompoundingAcrossRounds)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    std::vector<xjw::Camera> referenceCameras = orbitalCameras();
    std::vector<xjw::Camera> truthCameras = referenceCameras;
    std::vector<xjw::Camera> warmCameras = referenceCameras;
    std::vector<int> calibrationGroups(referenceCameras.size(), 0);
    for (std::size_t index = 0; index < referenceCameras.size(); ++index)
    {
        const double referenceFocal = index < referenceCameras.size() / 2
            ? 800.0
            : 1100.0;
        calibrationGroups[index] = index < referenceCameras.size() / 2 ? 0 : 1;
        referenceCameras[index].setIntrinsics(
            referenceFocal, referenceFocal, 512.0, 384.0);
        truthCameras[index] = referenceCameras[index];
        truthCameras[index].setIntrinsics(
            referenceFocal * 1.15,
            referenceFocal * 1.15 * 1.08,
            537.0,
            362.0);
        warmCameras[index] = referenceCameras[index];
        warmCameras[index].setIntrinsics(
            referenceFocal * 1.04375,
            referenceFocal * 1.04375 * 1.018,
            519.0,
            377.0);
    }

    std::vector<xjw::BATrack> tracks =
        makeTracks(truthCameras, orbitalPoints());
    ASSERT_FALSE(tracks.empty());
    for (xjw::BATrack &track : tracks)
    {
        track.controlPointConstraints.push_back(
            {track.initialPoint, 1.0e-4, 1.0, 0});
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.allowBackendFallback = false;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.useSharedIntrinsicParameterMask = true;
    options.sharedIntrinsicParameterMask.fill(false);
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)] = true;
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalAspectRatio)] = true;
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::PrincipalPointX)] = true;
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::PrincipalPointY)] = true;
    options.sharedIntrinsicReferenceCameras = referenceCameras;
    options.cameraCalibrationGroupIds = calibrationGroups;
    options.minSharedFocalScale = 0.95;
    options.maxSharedFocalScale = 1.05;
    options.minSharedFocalAspectScale = 0.98;
    options.maxSharedFocalAspectScale = 1.02;
    options.maxSharedPrincipalPointOffsetFraction = 0.01;
    options.sharedFocalPriorSigma = 10.0;
    options.sharedFocalAspectPriorSigma = 10.0;
    options.sharedPrincipalPointPriorSigmaFraction = 1.0;
    options.enableControlPointConstraints = true;
    options.enablePointFilter = false;
    options.stageSharedFocalRefinement = false;
    options.maxIterations = 20;

    const xjw::BAResult first = xjw::BundleAdjust::optimizePoints(
        warmCameras, tracks, options);
    ASSERT_TRUE(first.solutionUsable) << first.backendMessage;
    ASSERT_GT(
        first.refinedCameras.front().focalX(),
        referenceCameras.front().focalX() * 1.049);
    const xjw::BAResult second = xjw::BundleAdjust::optimizePoints(
        first.refinedCameras, tracks, options);
    ASSERT_TRUE(second.solutionUsable) << second.backendMessage;

    for (std::size_t index = 0; index < second.refinedCameras.size(); ++index)
    {
        const xjw::Camera &reference = referenceCameras[index];
        const xjw::Camera &refined = second.refinedCameras[index];
        const double referenceAspect =
            reference.focalY() / reference.focalX();
        const double refinedAspect = refined.focalY() / refined.focalX();
        const double maxPrincipalOffset = reference.focalX() * 0.01;
        EXPECT_GE(refined.focalX(), reference.focalX() * 0.95 - 1.0e-8);
        EXPECT_LE(refined.focalX(), reference.focalX() * 1.05 + 1.0e-8);
        EXPECT_GE(refinedAspect, referenceAspect * 0.98 - 1.0e-8);
        EXPECT_LE(refinedAspect, referenceAspect * 1.02 + 1.0e-8);
        EXPECT_GE(
            refined.principalX(),
            reference.principalX() - maxPrincipalOffset - 1.0e-8);
        EXPECT_LE(
            refined.principalX(),
            reference.principalX() + maxPrincipalOffset + 1.0e-8);
        EXPECT_GE(
            refined.principalY(),
            reference.principalY() - maxPrincipalOffset - 1.0e-8);
        EXPECT_LE(
            refined.principalY(),
            reference.principalY() + maxPrincipalOffset + 1.0e-8);
    }

    options.sharedIntrinsicParameterMask.fill(false);
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(
        xjw::BAIntrinsicParameter::FocalLength)] = true;
    const xjw::BAResult principalFrozen = xjw::BundleAdjust::optimizePoints(
        second.refinedCameras, tracks, options);
    ASSERT_TRUE(principalFrozen.solutionUsable)
        << principalFrozen.backendMessage;
    double expectedPrincipalOffsetX = 0.0;
    double expectedPrincipalOffsetY = 0.0;
    for (std::size_t index = 0;
         index < principalFrozen.refinedCameras.size();
         ++index)
    {
        expectedPrincipalOffsetX +=
            principalFrozen.refinedCameras[index].principalX() -
            referenceCameras[index].principalX();
        expectedPrincipalOffsetY +=
            principalFrozen.refinedCameras[index].principalY() -
            referenceCameras[index].principalY();
    }
    const double cameraCount =
        static_cast<double>(principalFrozen.refinedCameras.size());
    EXPECT_NEAR(
        principalFrozen.refinedSharedPrincipalOffsetX,
        expectedPrincipalOffsetX / cameraCount,
        1.0e-10);
    EXPECT_NEAR(
        principalFrozen.refinedSharedPrincipalOffsetY,
        expectedPrincipalOffsetY / cameraCount,
        1.0e-10);
    EXPECT_GT(
        std::abs(principalFrozen.refinedSharedPrincipalOffsetX) +
            std::abs(principalFrozen.refinedSharedPrincipalOffsetY),
        1.0);
}
