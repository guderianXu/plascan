#include "placamera/reference/CameraReferenceSourceId.h"
#include "placamera/reference/CameraReferenceUncertainty.h"

#include <placamera/types.h>

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace
{

    using namespace placamera::reference;

    TEST(CameraReferenceTypesTest, RejectsEmptyStrongIds)
    {
        EXPECT_THROW(placamera::CameraDefinitionId(std::string()), placamera::CameraValidationError);
        EXPECT_THROW(placamera::ImageId(std::string(" ")), placamera::CameraValidationError);
        EXPECT_THROW(ReferenceSourceId(std::string()), placamera::CameraValidationError);
    }

    TEST(CameraReferenceTypesTest, RejectsNonRotationPose)
    {
        EXPECT_THROW(placamera::Pose::create(placoordinate::CoordinateFrameId(std::string("local")),
                                             {0.0, 0.0, 0.0},
                                             {2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
                     placamera::CameraValidationError);
    }

    TEST(CameraReferenceTypesTest, RejectsInvalidCovarianceShape)
    {
        EXPECT_THROW(PoseCovariance::diagonal({1.0, 2.0, 3.0}), std::invalid_argument);
        EXPECT_THROW(PoseCovariance::symmetric({1.0, 2.0}), std::invalid_argument);
    }

    TEST(CameraReferenceTypesTest, EnforcesSymmetryAndDefinitenessPolicies)
    {
        std::array<double, 36> asymmetric{};
        asymmetric[0] = 1.0;
        asymmetric[7] = 1.0;
        asymmetric[14] = 1.0;
        asymmetric[21] = 1.0;
        asymmetric[28] = 1.0;
        asymmetric[35] = 1.0;
        asymmetric[1] = 0.5;
        EXPECT_THROW(PoseCovariance::matrix(asymmetric), placamera::CameraValidationError);

        EXPECT_THROW(PoseCovariance::diagonal({1.0, 1.0, 1.0, 1.0, -0.1, 1.0}),
                     placamera::CameraValidationError);
        EXPECT_THROW(PoseCovariance::diagonal({1.0, 1.0, 1.0, 1.0, 0.0, 1.0},
                                              PoseCovarianceComponents::PositionAndRotation,
                                              CovarianceDefiniteness::PositiveDefinite),
                     placamera::CameraValidationError);
        const PoseCovariance semidefinite =
            PoseCovariance::diagonal({1.0, 1.0, 1.0, 1.0, 0.0, 1.0});
        EXPECT_EQ(semidefinite.definiteness(), CovarianceDefiniteness::PositiveSemidefinite);
    }

    TEST(CameraReferenceTypesTest, PropagatesRigidAndSimilarityCovarianceInCanonicalOrder)
    {
        const PoseCovariance covariance =
            PoseCovariance::diagonal({1.0, 4.0, 9.0, 0.01, 0.04, 0.09},
                                     PoseCovarianceComponents::PositionAndRotation,
                                     CovarianceDefiniteness::PositiveDefinite);
        const placamera::RotationMatrix quarter_turn{{0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        const auto rigid = propagatePoseCovarianceRigid(covariance, quarter_turn);
        ASSERT_TRUE(rigid) << rigid.message();
        EXPECT_NEAR(rigid.value().matrixValues()[0], 4.0, 1.0e-14);
        EXPECT_NEAR(rigid.value().matrixValues()[7], 1.0, 1.0e-14);
        EXPECT_NEAR(rigid.value().matrixValues()[21], 0.04, 1.0e-14);
        EXPECT_NEAR(rigid.value().matrixValues()[28], 0.01, 1.0e-14);

        const auto similarity = propagatePoseCovarianceSimilarity(covariance, 3.0, quarter_turn);
        ASSERT_TRUE(similarity) << similarity.message();
        EXPECT_NEAR(similarity.value().matrixValues()[0], 36.0, 1.0e-13);
        EXPECT_NEAR(similarity.value().matrixValues()[7], 9.0, 1.0e-13);
        EXPECT_NEAR(similarity.value().matrixValues()[21], 0.04, 1.0e-14);
    }

} // namespace
