#include "OverlapAnalyzer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace
{

    std::shared_ptr<const placamera::FramePinholeModel> makeCamera(double x,
                                                                   double y,
                                                                   double z,
                                                                   placamera::RotationMatrix rotation,
                                                                   placamera::BrownConradyDistortion distortion = {},
                                                                   const std::string& frame_id = "project-world")
    {
        const placamera::FrameId frame(frame_id);
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("overlap-definition"),
                                                      {100.0, 100.0, 50.0, 50.0, 1.0, 1, 1},
                                                      distortion,
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        return std::make_shared<placamera::FramePinholeModel>(
            placamera::FramePinholeModel::create(placamera::CameraInstanceId("overlap-instance"),
                                                 placamera::ImageId("overlap-image"),
                                                 definition,
                                                 {100, 100},
                                                 placamera::Pose::create(frame, {x, y, z}, rotation)));
    }

    std::shared_ptr<const placamera::FramePinholeModel>
    makeDownLookingCamera(double x, double y, double z, const std::string& frame_id = "project-world")
    {
        return makeCamera(x, y, z, {1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0}, {}, frame_id);
    }

    xjw::OverlapImageInput makeImage(const std::string& path,
                                     std::shared_ptr<const placamera::FramePinholeModel> camera)
    {
        xjw::OverlapImageInput input;
        input.imagePath = path;
        input.camera = camera;
        input.width = 100;
        input.height = 100;
        return input;
    }

} // namespace

TEST(OverlapAnalyzerTest, ReferenceSpherePresetsExposePlanetRadii)
{
    EXPECT_NEAR(xjw::referenceBodyRadiusMeters(xjw::ReferenceBody::Earth), 6378137.0, 1e-6);
    EXPECT_NEAR(xjw::referenceBodyRadiusMeters(xjw::ReferenceBody::Moon), 1737400.0, 1e-6);
    EXPECT_NEAR(xjw::referenceBodyRadiusMeters(xjw::ReferenceBody::Mars), 3389500.0, 1e-6);
}

TEST(OverlapAnalyzerTest, ReferenceSphereHandlesLocalDownLookingCamerasWhereFixedPlaneFails)
{
    const std::vector<xjw::OverlapImageInput> images = {
        makeImage("a.jpg", makeDownLookingCamera(0.0, 0.0, 0.0)),
        makeImage("b.jpg", makeDownLookingCamera(5.0, 0.0, 0.0))
    };

    xjw::OverlapAnalysisResult fixedResult;
    std::string fixedError;
    EXPECT_FALSE(xjw::OverlapAnalyzer::analyze(images, nullptr, true, 0.0, 2.0, &fixedResult, &fixedError));
    EXPECT_NE(fixedError.find("固定高程"), std::string::npos);

    xjw::OverlapAnalysisOptions options;
    options.groundModel = xjw::OverlapGroundModel::ReferenceSphere;
    options.referenceSphere.body = xjw::ReferenceBody::Earth;
    options.referenceSphere.radiusMeters = xjw::referenceBodyRadiusMeters(xjw::ReferenceBody::Earth);
    options.referenceSphere.centerMode = xjw::ReferenceSphereCenterMode::Auto;
    options.referenceSphere.autoLocalTangentHeight = true;
    options.neighborFactor = 2.0;

    xjw::OverlapAnalysisResult sphereResult;
    std::string sphereError;
    ASSERT_TRUE(xjw::OverlapAnalyzer::analyze(images, options, &sphereResult, &sphereError)) << sphereError;
    ASSERT_EQ(sphereResult.centers.size(), 2u);
    EXPECT_LT(sphereResult.centers[0][2], images[0].camera->pose().center[2]);
    EXPECT_LT(sphereResult.centers[1][2], images[1].camera->pose().center[2]);
    ASSERT_FALSE(sphereResult.pairs.empty());
    EXPECT_NE(sphereResult.detail.find("ground=reference_sphere"), std::string::npos);
    EXPECT_NE(sphereResult.detail.find("body=earth"), std::string::npos);
}

TEST(OverlapAnalyzerTest, RejectsMissingPlaCameraModelBeforeGeometry)
{
    const std::vector<xjw::OverlapImageInput> images = {makeImage("invalid-a.jpg", nullptr),
                                                        makeImage("valid-b.jpg", makeDownLookingCamera(5.0, 0.0, 0.0))};

    xjw::OverlapAnalysisResult result;
    std::string error;
    EXPECT_FALSE(xjw::OverlapAnalyzer::analyze(images, nullptr, true, 0.0, 2.0, &result, &error));
    EXPECT_NE(error.find("PlaCamera 相机模型缺失"), std::string::npos);
}

TEST(OverlapAnalyzerTest, RejectsMixedPlaCameraWorldFrames)
{
    const std::vector<xjw::OverlapImageInput> images = {
        makeImage("a.jpg", makeDownLookingCamera(0.0, 0.0, 100.0, "local-a")),
        makeImage("b.jpg", makeDownLookingCamera(5.0, 0.0, 100.0, "local-b"))};

    xjw::OverlapAnalysisResult result;
    std::string error;
    EXPECT_FALSE(xjw::OverlapAnalyzer::analyze(images, nullptr, true, 0.0, 2.0, &result, &error));
    EXPECT_NE(error.find("世界坐标系不一致"), std::string::npos);
}

TEST(OverlapAnalyzerTest, UsesPlaCameraImagingLocusForDistortedPixels)
{
    const auto camera =
        makeCamera(0.0, 0.0, 100.0, {1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0}, {0.15, -0.02, 0.0, 0.001, -0.002});

    const std::array<double, 2> pixel{{0.0, 0.0}};
    const auto ray = camera->imageToImagingLocus({pixel[0], pixel[1]});
    ASSERT_TRUE(ray.ok()) << ray.message();
    ASSERT_LT(ray.value().direction[2], -1.0e-9);
    const double t = -ray.value().origin.position[2] / ray.value().direction[2];
    const std::array<double, 3> expected{ray.value().origin.position[0] + t * ray.value().direction[0],
                                         ray.value().origin.position[1] + t * ray.value().direction[1],
                                         0.0};

    std::array<double, 3> actual{};
    std::string error;
    ASSERT_TRUE(xjw::GroundBackProjector::backProjectToFixedZ(*camera, pixel[0], pixel[1], 0.0, &actual, &error))
        << error;
    EXPECT_NEAR(actual[0], expected[0], 1.0e-9);
    EXPECT_NEAR(actual[1], expected[1], 1.0e-9);
    EXPECT_NEAR(actual[2], expected[2], 1.0e-12);
}

TEST(OverlapAnalyzerTest, RejectsNonFiniteGroundIntersection)
{
    const auto camera = makeCamera(0.0, 0.0, -1.0e308, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    std::array<double, 3> ground{};
    std::string error;
    EXPECT_FALSE(xjw::GroundBackProjector::backProjectToFixedZ(*camera, 50.0, 50.0, 1.0e308, &ground, &error));
    EXPECT_FALSE(error.empty());
}

TEST(OverlapAnalyzerTest, RejectsMissingDemInsteadOfUsingFixedPlane)
{
    const std::vector<xjw::OverlapImageInput> images = {makeImage("a.jpg", makeDownLookingCamera(0.0, 0.0, 100.0)),
                                                        makeImage("b.jpg", makeDownLookingCamera(5.0, 0.0, 100.0))};
    xjw::OverlapAnalysisOptions options;
    options.groundModel = xjw::OverlapGroundModel::Dem;

    xjw::OverlapAnalysisResult result;
    std::string error;
    EXPECT_FALSE(xjw::OverlapAnalyzer::analyze(images, options, &result, &error));
    EXPECT_NE(error.find("DEM"), std::string::npos);
}

TEST(OverlapAnalyzerTest, RejectsNegativeReferenceSphereRadius)
{
    const std::vector<xjw::OverlapImageInput> images = {makeImage("a.jpg", makeDownLookingCamera(0.0, 0.0, 0.0)),
                                                        makeImage("b.jpg", makeDownLookingCamera(5.0, 0.0, 0.0))};
    xjw::OverlapAnalysisOptions options;
    options.groundModel = xjw::OverlapGroundModel::ReferenceSphere;
    options.referenceSphere.radiusMeters = -1.0;

    xjw::OverlapAnalysisResult result;
    std::string error;
    EXPECT_FALSE(xjw::OverlapAnalyzer::analyze(images, options, &result, &error));
    EXPECT_NE(error.find("半径"), std::string::npos);
}

TEST(OverlapAnalyzerTest, RejectsNonPositiveNeighborFactor)
{
    const std::vector<xjw::OverlapImageInput> images = {makeImage("a.jpg", makeDownLookingCamera(0.0, 0.0, 100.0)),
                                                        makeImage("b.jpg", makeDownLookingCamera(5.0, 0.0, 100.0))};

    xjw::OverlapAnalysisResult result;
    std::string error;
    EXPECT_FALSE(xjw::OverlapAnalyzer::analyze(images, nullptr, true, 0.0, 0.0, &result, &error));
    EXPECT_NE(error.find("邻域系数"), std::string::npos);
}

TEST(OverlapAnalyzerTest, RejectsPartialFootprintInsteadOfAveragingValidCorners)
{
    const auto camera =
        makeCamera(0.0, 0.0, 100.0, {1.0, 0.0, 0.0, 0.0, -0.2, -0.9797958971132712, 0.0, 0.9797958971132712, -0.2});
    const std::vector<xjw::OverlapImageInput> images = {makeImage("partial-a.jpg", camera),
                                                        makeImage("partial-b.jpg", camera)};

    xjw::OverlapAnalysisResult result;
    std::string error;
    EXPECT_FALSE(xjw::OverlapAnalyzer::analyze(images, nullptr, true, 0.0, 2.0, &result, &error));
    EXPECT_NE(error.find("角点"), std::string::npos);
}
