#include "placoordinate/serialization/CoordinateContextJson.h"

#include "placoordinate/serialization/CoordinateContextJsonStrings.h"
#include "placoordinate/types/CoordinateErrors.h"

#include <nlohmann/json.hpp>

#include <array>
#include <set>
#include <stdexcept>
#include <utility>

namespace placoordinate
{
    namespace
    {

        using Json = nlohmann::json;

        void requireKeys(const Json& value, std::initializer_list<const char*> keys, const char* path)
        {
            if (!value.is_object())
            {
                throw std::invalid_argument(std::string(path) + " must be an object");
            }
            std::set<std::string> expected;
            for (const char* key : keys)
            {
                expected.emplace(key);
            }
            for (auto iterator = value.begin(); iterator != value.end(); ++iterator)
            {
                if (!expected.contains(iterator.key()))
                {
                    throw std::invalid_argument(std::string(path) + " contains unknown key: " + iterator.key());
                }
            }
            for (const std::string& key : expected)
            {
                if (!value.contains(key))
                {
                    throw std::invalid_argument(std::string(path) + " is missing key: " + key);
                }
            }
        }

        template <typename Enum, typename Parser>
        Enum requireEnum(const Json& value, const char* key, const char* path, Parser parser)
        {
            const std::string name = value.at(key).get<std::string>();
            const std::optional<Enum> parsed = parser(name);
            if (!parsed)
            {
                throw std::invalid_argument(std::string(path) + "." + key + " has unsupported value: " + name);
            }
            return *parsed;
        }

        template <std::size_t Size>
        std::array<double, Size> requireDoubleArray(const Json& value, const char* key, const char* path)
        {
            const Json& array = value.at(key);
            if (!array.is_array() || array.size() != Size)
            {
                throw std::invalid_argument(std::string(path) + "." + key + " has an invalid length");
            }
            std::array<double, Size> result{};
            for (std::size_t index = 0; index < Size; ++index)
            {
                result[index] = array.at(index).get<double>();
            }
            return result;
        }

        Json spatialReferenceToJson(const SpatialReferenceDefinition& reference)
        {
            return Json{{"id", reference.id().value()},
                        {"frame_id", reference.frameId().value()},
                        {"authority", reference.authority()},
                        {"code", reference.authorityCode()},
                        {"kind", detail::enumName(reference.kind())},
                        {"axis_mapping", detail::enumName(reference.axisMapping())},
                        {"vertical_reference", detail::enumName(reference.verticalReference())},
                        {"linear_unit_to_metres", reference.linearUnitToMetres()},
                        {"canonical_definition", reference.canonicalDefinition()},
                        {"canonical_definition_hash", reference.canonicalDefinitionHash()}};
        }

        SpatialReferenceDefinition spatialReferenceFromJson(const Json& value)
        {
            constexpr auto Path = "coordinate_system.spatial_references[]";
            requireKeys(value,
                        {"id",
                         "frame_id",
                         "authority",
                         "code",
                         "kind",
                         "axis_mapping",
                         "vertical_reference",
                         "linear_unit_to_metres",
                         "canonical_definition",
                         "canonical_definition_hash"},
                        Path);
            SpatialReferenceDefinition result = SpatialReferenceDefinition::create(
                SpatialReferenceId(value.at("id").get<std::string>()),
                CoordinateFrameId(value.at("frame_id").get<std::string>()),
                requireEnum<SpatialReferenceKind>(value, "kind", Path, detail::spatialReferenceKindFromName),
                requireEnum<CoordinateAxisOrder>(value, "axis_mapping", Path, detail::coordinateAxisOrderFromName),
                requireEnum<VerticalReference>(value, "vertical_reference", Path, detail::verticalReferenceFromName),
                value.at("canonical_definition").get<std::string>(),
                value.at("authority").get<std::string>(),
                value.at("code").get<std::string>(),
                value.at("linear_unit_to_metres").get<double>());
            const std::string stored_hash = value.at("canonical_definition_hash").get<std::string>();
            if (stored_hash != result.canonicalDefinitionHash())
            {
                throw CoordinateValidationError(CoordinateErrorCode::CoordinateHashMismatch,
                                                "canonical spatial reference definition hash mismatch");
            }
            return result;
        }

        Json frameToJson(const CoordinateFrame& frame)
        {
            Json parent = nullptr;
            if (frame.parent)
            {
                parent = frame.parent->value();
            }
            return Json{
                {"id", frame.id.value()},
                {"kind", detail::enumName(frame.kind)},
                {"linear_unit", detail::enumName(frame.linearUnit)},
                {"angle_unit", detail::enumName(frame.angleUnit)},
                {"parent_frame_id", std::move(parent)},
                {"to_parent", {{"rotation", frame.toParent.rotation}, {"translation", frame.toParent.translation}}}};
        }

        CoordinateFrame frameFromJson(const Json& value)
        {
            constexpr auto Path = "coordinate_system.frames[]";
            requireKeys(value, {"id", "kind", "linear_unit", "angle_unit", "parent_frame_id", "to_parent"}, Path);
            const Json& transform = value.at("to_parent");
            requireKeys(transform, {"rotation", "translation"}, "coordinate_system.frames[].to_parent");
            RigidTransform rigid;
            rigid.rotation = requireDoubleArray<9>(transform, "rotation", "coordinate_system.frames[].to_parent");
            rigid.translation = requireDoubleArray<3>(transform, "translation", "coordinate_system.frames[].to_parent");
            std::optional<CoordinateFrameId> parent;
            if (!value.at("parent_frame_id").is_null())
            {
                parent.emplace(value.at("parent_frame_id").get<std::string>());
            }
            return CoordinateFrame::create(
                CoordinateFrameId(value.at("id").get<std::string>()),
                requireEnum<CoordinateFrameKind>(value, "kind", Path, detail::coordinateFrameKindFromName),
                requireEnum<LinearUnit>(value, "linear_unit", Path, detail::linearUnitFromName),
                requireEnum<AngleUnit>(value, "angle_unit", Path, detail::angleUnitFromName),
                std::move(parent),
                rigid);
        }

        CoordinateContext parseContext(const Json& document)
        {
            requireKeys(document,
                        {"schema_version",
                         "context_id",
                         "revision",
                         "spatial_references",
                         "frames",
                         "solver_frame_id",
                         "solver_spatial_reference_id",
                         "solver_scale_status",
                         "solver_normalization_hash",
                         "context_hash"},
                        "coordinate_system");
            if (document.at("schema_version").get<int>() != CoordinateContext::SchemaVersion)
            {
                throw std::invalid_argument("unsupported coordinate context schema_version");
            }
            if (!document.at("spatial_references").is_array() || !document.at("frames").is_array())
            {
                throw std::invalid_argument("coordinate context references and frames must be arrays");
            }
            std::vector<SpatialReferenceDefinition> references;
            references.reserve(document.at("spatial_references").size());
            for (const Json& value : document.at("spatial_references"))
            {
                references.push_back(spatialReferenceFromJson(value));
            }
            std::vector<CoordinateFrame> frames;
            frames.reserve(document.at("frames").size());
            for (const Json& value : document.at("frames"))
            {
                frames.push_back(frameFromJson(value));
            }
            std::optional<SpatialReferenceId> solver_reference_id;
            if (!document.at("solver_spatial_reference_id").is_null())
            {
                solver_reference_id.emplace(document.at("solver_spatial_reference_id").get<std::string>());
            }
            CoordinateContext context = CoordinateContext::create(
                CoordinateContextId(document.at("context_id").get<std::string>()),
                document.at("revision").get<int>(),
                std::move(references),
                std::move(frames),
                std::move(solver_reference_id),
                SolverFrameDefinition::create(
                    CoordinateFrameId(document.at("solver_frame_id").get<std::string>()),
                    requireEnum<SolverScaleStatus>(
                        document, "solver_scale_status", "coordinate_system", detail::solverScaleStatusFromName),
                    document.at("solver_normalization_hash").get<std::string>()));
            if (context.contextHash() != document.at("context_hash").get<std::string>())
            {
                throw CoordinateValidationError(CoordinateErrorCode::CoordinateHashMismatch,
                                                "coordinate context hash mismatch");
            }
            return context;
        }

    } // namespace

    bool CoordinateContextJsonResult::ok() const noexcept
    {
        return context.has_value();
    }

    nlohmann::json coordinateContextToJson(const CoordinateContext& context)
    {
        Json references = Json::array();
        for (const SpatialReferenceDefinition& reference : context.spatialReferences())
        {
            references.push_back(spatialReferenceToJson(reference));
        }
        Json frames = Json::array();
        for (const CoordinateFrame& frame : context.frames())
        {
            frames.push_back(frameToJson(frame));
        }
        Json solver_reference_id = nullptr;
        if (context.solverSpatialReferenceId())
        {
            solver_reference_id = context.solverSpatialReferenceId()->value();
        }
        return Json{{"schema_version", context.schemaVersion()},
                    {"context_id", context.id().value()},
                    {"revision", context.revision()},
                    {"spatial_references", std::move(references)},
                    {"frames", std::move(frames)},
                    {"solver_frame_id", context.solverFrame().frameId.value()},
                    {"solver_spatial_reference_id", std::move(solver_reference_id)},
                    {"solver_scale_status", detail::enumName(context.solverFrame().scaleStatus)},
                    {"solver_normalization_hash", context.solverFrame().normalizationHash},
                    {"context_hash", context.contextHash()}};
    }

    CoordinateContextJsonResult coordinateContextFromJson(const nlohmann::json& document)
    {
        CoordinateContextJsonResult result;
        try
        {
            result.context = parseContext(document);
        }
        catch (const std::exception& exception)
        {
            result.error = exception.what();
        }
        return result;
    }

    void embedCoordinateContextInChunk(nlohmann::json* chunk, const CoordinateContext& context)
    {
        if (!chunk || !chunk->is_object())
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinatePersistence,
                                            "chunk JSON must be a non-null object");
        }
        (*chunk)["coordinate_system"] = coordinateContextToJson(context);
    }

    CoordinateContextJsonResult coordinateContextFromChunk(const nlohmann::json& chunk)
    {
        if (!chunk.is_object() || !chunk.contains("coordinate_system"))
        {
            return CoordinateContextJsonResult{std::nullopt, "chunk.coordinate_system is missing"};
        }
        return coordinateContextFromJson(chunk.at("coordinate_system"));
    }

} // namespace placoordinate
