#include "placoordinate/gdal/GdalCoordinateTransform.h"
#include "placoordinate/serialization/CoordinateContextJson.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace
{
    using namespace placoordinate;

    CoordinateFrame frame(const char* id, CoordinateFrameKind kind, AngleUnit angleUnit)
    {
        return CoordinateFrame::create(
            CoordinateFrameId(id), kind, LinearUnit::Metre, angleUnit, std::nullopt, RigidTransform::identity());
    }

    SpatialReferenceDefinition
    reference(const char* id, const char* frameId, const char* definition, VerticalReference verticalReference)
    {
        GdalSpatialReferenceResult normalized = normalizeGdalSpatialReference(
            SpatialReferenceId(id), CoordinateFrameId(frameId), definition, verticalReference);
        if (!normalized.ok())
        {
            throw std::runtime_error(normalized.error);
        }
        return std::move(*normalized.reference);
    }

    CoordinateContext earthContext()
    {
        return CoordinateContext::create(
            CoordinateContextId("coordctx-earth-ecef"),
            3,
            {reference("crs-epsg-4979", "frame-wgs84-geodetic", "EPSG:4979", VerticalReference::Ellipsoidal),
             reference("crs-epsg-4978", "frame-wgs84-ecef", "EPSG:4978", VerticalReference::NotApplicable)},
            {frame("frame-wgs84-geodetic", CoordinateFrameKind::Geodetic, AngleUnit::Degree),
             frame("frame-wgs84-ecef", CoordinateFrameKind::Ecef, AngleUnit::Radian)},
            SpatialReferenceId("crs-epsg-4978"),
            SolverFrameDefinition::create(
                CoordinateFrameId("frame-wgs84-ecef"), SolverScaleStatus::Metric, "gcp-ecef-normalization-v1"));
    }

    CoordinateContext relativeContext()
    {
        const CoordinateFrame solver = CoordinateFrame::create(CoordinateFrameId("frame-relative"),
                                                               CoordinateFrameKind::LocalCartesian,
                                                               LinearUnit::ProjectUnit,
                                                               AngleUnit::Radian,
                                                               std::nullopt,
                                                               RigidTransform::identity());
        return CoordinateContext::create(CoordinateContextId("coordctx-relative"),
                                         1,
                                         {},
                                         {solver},
                                         std::nullopt,
                                         SolverFrameDefinition::create(CoordinateFrameId("frame-relative"),
                                                                       SolverScaleStatus::Unresolved,
                                                                       "relative-normalization-v1"));
    }

    TEST(CoordinateContextJsonTest, RoundTripsCompleteCanonicalDefinitions)
    {
        const CoordinateContext source = earthContext();
        const nlohmann::json document = coordinateContextToJson(source);
        const CoordinateContextJsonResult restored = coordinateContextFromJson(document);

        ASSERT_TRUE(restored.ok()) << restored.error;
        EXPECT_EQ(restored.context->contextHash(), source.contextHash());
        EXPECT_EQ(restored.context->id(), source.id());
        ASSERT_EQ(document.at("spatial_references").size(), 2U);
        EXPECT_FALSE(document.at("spatial_references").at(0).at("canonical_definition").get<std::string>().empty());
        EXPECT_EQ(restored.context->solverCoordinateFrame()->id.value(), "frame-wgs84-ecef");
    }

    TEST(CoordinateContextJsonTest, EmbedsOnlyTheCoordinateSystemChunkMember)
    {
        nlohmann::json chunk{{"schema_version", 1}, {"name", "Chunk 1"}};
        const CoordinateContext source = earthContext();

        embedCoordinateContextInChunk(&chunk, source);
        const CoordinateContextJsonResult restored = coordinateContextFromChunk(chunk);

        ASSERT_TRUE(restored.ok()) << restored.error;
        EXPECT_EQ(chunk.at("name"), "Chunk 1");
        EXPECT_EQ(restored.context->contextHash(), source.contextHash());
    }

    TEST(CoordinateContextJsonTest, RejectsCanonicalDefinitionTampering)
    {
        nlohmann::json document = coordinateContextToJson(earthContext());
        document["spatial_references"][0]["canonical_definition"] = "GEODCRS[\"tampered\"]";

        const CoordinateContextJsonResult restored = coordinateContextFromJson(document);

        EXPECT_FALSE(restored.ok());
        EXPECT_NE(restored.error.find("definition hash mismatch"), std::string::npos);
    }

    TEST(CoordinateContextJsonTest, RejectsContextHashTamperingAndUnknownFields)
    {
        nlohmann::json tampered_hash = coordinateContextToJson(earthContext());
        tampered_hash["context_hash"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
        EXPECT_FALSE(coordinateContextFromJson(tampered_hash).ok());

        nlohmann::json unknown_field = coordinateContextToJson(earthContext());
        unknown_field["implicit_datum"] = "WGS84";
        const CoordinateContextJsonResult restored = coordinateContextFromJson(unknown_field);
        EXPECT_FALSE(restored.ok());
        EXPECT_NE(restored.error.find("unknown key"), std::string::npos);
    }

    TEST(CoordinateContextJsonTest, RejectsMissingChunkContext)
    {
        const CoordinateContextJsonResult restored = coordinateContextFromChunk(nlohmann::json::object());
        EXPECT_FALSE(restored.ok());
        EXPECT_NE(restored.error.find("chunk.coordinate_system"), std::string::npos);
    }

    TEST(CoordinateContextJsonTest, RoundTripsUnresolvedProjectUnitContext)
    {
        const nlohmann::json document = coordinateContextToJson(relativeContext());
        const CoordinateContextJsonResult restored = coordinateContextFromJson(document);

        ASSERT_TRUE(restored.ok()) << restored.error;
        EXPECT_FALSE(restored.context->solverFrame().hasMetricScale());
        EXPECT_FALSE(restored.context->solverSpatialReferenceId().has_value());
        EXPECT_TRUE(document.at("solver_spatial_reference_id").is_null());
        ASSERT_NE(restored.context->solverCoordinateFrame(), nullptr);
        EXPECT_EQ(restored.context->solverCoordinateFrame()->linearUnit, LinearUnit::ProjectUnit);
    }

} // namespace
