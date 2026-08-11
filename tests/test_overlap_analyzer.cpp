#include "OverlapAnalyzer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace
{

xjw::FramePinholeCamera makeDownLookingCamera(double x, double y, double z)
{
    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(100.0, 100.0, 50.0, 50.0);
    camera.setPose(std::array<double, 9>{{
                       1.0, 0.0, 0.0,
                       0.0, -1.0, 0.0,
                       0.0, 0.0, -1.0}},
                   std::array<double, 3>{{x, y, z}});
    return camera;
}

xjw::OverlapImageInput makeImage(const std::string &path, const xjw::FramePinholeCamera &camera)
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
    EXPECT_LT(sphereResult.centers[0][2], images[0].camera.cameraCenter()[2]);
    EXPECT_LT(sphereResult.centers[1][2], images[1].camera.cameraCenter()[2]);
    ASSERT_FALSE(sphereResult.pairs.empty());
    EXPECT_NE(sphereResult.detail.find("ground=reference_sphere"), std::string::npos);
    EXPECT_NE(sphereResult.detail.find("body=earth"), std::string::npos);
}

