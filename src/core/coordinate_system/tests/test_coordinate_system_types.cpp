#include "coordinate_system/context/SolverFrameDefinition.h"
#include "coordinate_system/context/CoordinateContext.h"
#include "coordinate_system/context/CoordinateHash.h"
#include "coordinate_system/context/SpatialReferenceDefinition.h"
#include "coordinate_system/types/CoordinateErrors.h"
#include "coordinate_system/types/CoordinateFrames.h"
#include "coordinate_system/types/CoordinateIds.h"
#include "coordinate_system/types/TimeReference.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace
{

    xjw::coordinate_system::SpatialReferenceDefinition makeGeocentricReference(const std::string& id)
    {
        using namespace xjw::coordinate_system;
        return SpatialReferenceDefinition::create(SpatialReferenceId(id),
                                                  CoordinateFrameId("frame-ecef"),
                                                  SpatialReferenceKind::Geocentric3d,
                                                  CoordinateAxisOrder::CanonicalXyz,
                                                  VerticalReference::NotApplicable,
                                                  "GEODCRS[\"WGS 84 geocentric\"]",
                                                  "EPSG",
                                                  "4978",
                                                  1.0);
    }

    xjw::coordinate_system::SpatialReferenceDefinition makeGeographicReference(const std::string& id)
    {
        using namespace xjw::coordinate_system;
        return SpatialReferenceDefinition::create(SpatialReferenceId(id),
                                                  CoordinateFrameId("frame-geographic"),
                                                  SpatialReferenceKind::Geographic3d,
                                                  CoordinateAxisOrder::LongitudeLatitude,
                                                  VerticalReference::Ellipsoidal,
                                                  "GEOGCRS[\"WGS 84 geographic 3D\"]",
                                                  "EPSG",
                                                  "4979",
                                                  0.0);
    }

    using namespace xjw::coordinate_system;

    CoordinateFrame localFrame(LinearUnit unit)
    {
        return CoordinateFrame::create(CoordinateFrameId("solver"),
                                       CoordinateFrameKind::LocalCartesian,
                                       unit,
                                       AngleUnit::Radian,
                                       std::nullopt,
                                       RigidTransform::identity());
    }

    CoordinateFrame ecefFrame()
    {
        return CoordinateFrame::create(CoordinateFrameId("frame-ecef"),
                                       CoordinateFrameKind::Ecef,
                                       LinearUnit::Metre,
                                       AngleUnit::Radian,
                                       std::nullopt,
                                       RigidTransform::identity());
    }

    CoordinateFrame geographicFrame()
    {
        return CoordinateFrame::create(CoordinateFrameId("frame-geographic"),
                                       CoordinateFrameKind::Geodetic,
                                       LinearUnit::Metre,
                                       AngleUnit::Degree,
                                       std::nullopt,
                                       RigidTransform::identity());
    }

    TEST(CoordinateSystemTypesTest, RejectsEmptyFrameId)
    {
        EXPECT_THROW(CoordinateFrameId(std::string()), CoordinateValidationError);
        EXPECT_THROW(CoordinateFrameId(std::string(" ")), CoordinateValidationError);
    }

    TEST(CoordinateSystemTypesTest, AcceptsValidLocalFrame)
    {
        const CoordinateFrame frame = CoordinateFrame::create(CoordinateFrameId("local-enu"),
                                                              CoordinateFrameKind::LocalEnu,
                                                              LinearUnit::Metre,
                                                              AngleUnit::Degree,
                                                              std::nullopt,
                                                              RigidTransform::identity());

        EXPECT_EQ(frame.id.value(), "local-enu");
        EXPECT_EQ(frame.kind, CoordinateFrameKind::LocalEnu);
        EXPECT_TRUE(frame.toParent.isIdentity());
    }

    TEST(CoordinateSystemTypesTest, RejectsInvalidFrameTransform)
    {
        RigidTransform transform;
        transform.rotation[0] = 2.0;
        EXPECT_THROW(CoordinateFrame::create(CoordinateFrameId("invalid"),
                                             CoordinateFrameKind::LocalCartesian,
                                             LinearUnit::Metre,
                                             AngleUnit::Radian,
                                             std::nullopt,
                                             transform),
                     CoordinateValidationError);
    }

    TEST(CoordinateSystemTypesTest, PreservesTimeScaleAndSeconds)
    {
        const TimeReference time = TimeReference::create(TimeScale::Tdb, 123.5);
        EXPECT_EQ(time.scale, TimeScale::Tdb);
        EXPECT_DOUBLE_EQ(time.seconds, 123.5);
        EXPECT_THROW(TimeReference::create(TimeScale::Utc, std::numeric_limits<double>::infinity()),
                     CoordinateValidationError);
    }

    TEST(CoordinateSystemTypesTest, KeepsMetricAndUnresolvedSolverScaleDistinct)
    {
        const SolverFrameDefinition metric =
            SolverFrameDefinition::create(CoordinateFrameId("solver"), SolverScaleStatus::Metric, "metric-hash");
        const SolverFrameDefinition relative =
            SolverFrameDefinition::create(CoordinateFrameId("solver"), SolverScaleStatus::Unresolved, "relative-hash");

        EXPECT_TRUE(metric.hasMetricScale());
        EXPECT_FALSE(relative.hasMetricScale());
        EXPECT_THROW(
            SolverFrameDefinition::create(CoordinateFrameId("solver"), SolverScaleStatus::Metric, std::string()),
            CoordinateValidationError);
    }

    TEST(CoordinateSystemTypesTest, ComputesStandardSha256Digest)
    {
        EXPECT_EQ(sha256Hash("abc"), "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        EXPECT_TRUE(isSha256Hash(sha256Hash("PlaScan")));
        EXPECT_FALSE(isSha256Hash("sha256:not-a-digest"));
    }

    TEST(CoordinateSystemTypesTest, BuildsStableMetricCoordinateContext)
    {
        const SolverFrameDefinition solver =
            SolverFrameDefinition::create(CoordinateFrameId("frame-ecef"), SolverScaleStatus::Metric, "solver-v1");
        const CoordinateContext first =
            CoordinateContext::create(CoordinateContextId("context-earth"),
                                      1,
                                      {makeGeocentricReference("crs-ecef"), makeGeographicReference("crs-geographic")},
                                      {ecefFrame(), geographicFrame()},
                                      SpatialReferenceId("crs-ecef"),
                                      solver);
        const CoordinateContext reordered =
            CoordinateContext::create(CoordinateContextId("context-earth"),
                                      1,
                                      {makeGeographicReference("crs-geographic"), makeGeocentricReference("crs-ecef")},
                                      {geographicFrame(), ecefFrame()},
                                      SpatialReferenceId("crs-ecef"),
                                      solver);
        const CoordinateContext revised =
            CoordinateContext::create(CoordinateContextId("context-earth"),
                                      2,
                                      {makeGeocentricReference("crs-ecef"), makeGeographicReference("crs-geographic")},
                                      {ecefFrame(), geographicFrame()},
                                      SpatialReferenceId("crs-ecef"),
                                      solver);

        ASSERT_NE(first.solverSpatialReference(), nullptr);
        EXPECT_EQ(first.solverSpatialReference()->id().value(), "crs-ecef");
        EXPECT_EQ(first.contextHash(), reordered.contextHash());
        EXPECT_NE(first.contextHash(), revised.contextHash());
        EXPECT_TRUE(isSha256Hash(first.contextHash()));
        EXPECT_NE(
            first.findSpatialReferenceByDefinitionHash(makeGeographicReference("temporary").canonicalDefinitionHash()),
            nullptr);
    }

    TEST(CoordinateSystemTypesTest, RejectsInvalidContextScaleClaims)
    {
        const SolverFrameDefinition metric =
            SolverFrameDefinition::create(CoordinateFrameId("frame-ecef"), SolverScaleStatus::Metric, "solver-v1");
        EXPECT_THROW(CoordinateContext::create(CoordinateContextId("context-earth"),
                                               1,
                                               {makeGeographicReference("crs-geographic")},
                                               {ecefFrame(), geographicFrame()},
                                               SpatialReferenceId("crs-geographic"),
                                               metric),
                     CoordinateValidationError);

        const SolverFrameDefinition relative =
            SolverFrameDefinition::create(CoordinateFrameId("solver"), SolverScaleStatus::Unresolved, "relative-v1");
        EXPECT_THROW(CoordinateContext::create(CoordinateContextId("context-relative"),
                                               1,
                                               {makeGeocentricReference("crs-ecef")},
                                               {localFrame(LinearUnit::ProjectUnit), ecefFrame()},
                                               SpatialReferenceId("crs-ecef"),
                                               relative),
                     CoordinateValidationError);
    }

    TEST(CoordinateSystemTypesTest, RejectsCyclicFrameRegistry)
    {
        const CoordinateFrame first = CoordinateFrame::create(CoordinateFrameId("frame-a"),
                                                              CoordinateFrameKind::LocalCartesian,
                                                              LinearUnit::ProjectUnit,
                                                              AngleUnit::Radian,
                                                              CoordinateFrameId("frame-b"),
                                                              RigidTransform::identity());
        const CoordinateFrame second = CoordinateFrame::create(CoordinateFrameId("frame-b"),
                                                               CoordinateFrameKind::LocalCartesian,
                                                               LinearUnit::ProjectUnit,
                                                               AngleUnit::Radian,
                                                               CoordinateFrameId("frame-a"),
                                                               RigidTransform::identity());
        const SolverFrameDefinition solver =
            SolverFrameDefinition::create(CoordinateFrameId("frame-a"), SolverScaleStatus::Unresolved, "relative-v1");

        EXPECT_THROW(CoordinateContext::create(
                         CoordinateContextId("cyclic-context"), 1, {}, {first, second}, std::nullopt, solver),
                     CoordinateValidationError);
    }

    TEST(CoordinateSystemTypesTest, RejectsSpatialReferenceBoundToIncompatibleFrameKind)
    {
        const SpatialReferenceDefinition invalid_binding =
            SpatialReferenceDefinition::create(SpatialReferenceId("crs-ecef"),
                                               CoordinateFrameId("frame-geographic"),
                                               SpatialReferenceKind::Geocentric3d,
                                               CoordinateAxisOrder::CanonicalXyz,
                                               VerticalReference::NotApplicable,
                                               "GEODCRS[\"WGS 84 geocentric\"]",
                                               "EPSG",
                                               "4978",
                                               1.0);
        const SolverFrameDefinition solver = SolverFrameDefinition::create(
            CoordinateFrameId("frame-geographic"), SolverScaleStatus::Metric, "invalid-v1");

        EXPECT_THROW(CoordinateContext::create(CoordinateContextId("invalid-binding"),
                                               1,
                                               {invalid_binding},
                                               {geographicFrame()},
                                               SpatialReferenceId("crs-ecef"),
                                               solver),
                     CoordinateValidationError);
    }

} // namespace
