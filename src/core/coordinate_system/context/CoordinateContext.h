#pragma once

#include "SolverFrameDefinition.h"
#include "SpatialReferenceDefinition.h"
#include "coordinate_system/types/CoordinateFrames.h"

#include <optional>
#include <string>
#include <vector>

namespace xjw::coordinate_system
{

    class CoordinateContext
    {
    public:
        static constexpr int SchemaVersion = 1;

        static CoordinateContext create(CoordinateContextId id,
                                        int revision,
                                        std::vector<SpatialReferenceDefinition> spatialReferences,
                                        std::vector<CoordinateFrame> frames,
                                        std::optional<SpatialReferenceId> solverSpatialReferenceId,
                                        SolverFrameDefinition solverFrame);

        const CoordinateContextId& id() const noexcept;
        int schemaVersion() const noexcept;
        int revision() const noexcept;
        const std::vector<SpatialReferenceDefinition>& spatialReferences() const noexcept;
        const std::vector<CoordinateFrame>& frames() const noexcept;
        const std::optional<SpatialReferenceId>& solverSpatialReferenceId() const noexcept;
        const SolverFrameDefinition& solverFrame() const noexcept;
        const std::string& contextHash() const noexcept;

        const SpatialReferenceDefinition* findSpatialReference(const SpatialReferenceId& id) const noexcept;
        const SpatialReferenceDefinition* findSpatialReferenceByDefinitionHash(std::string_view hash) const noexcept;
        const SpatialReferenceDefinition*
        findSpatialReferenceByDefinitionHash(std::string_view hash, VerticalReference verticalReference) const noexcept;
        const SpatialReferenceDefinition* solverSpatialReference() const noexcept;
        const CoordinateFrame* findFrame(const CoordinateFrameId& id) const noexcept;
        const CoordinateFrame* solverCoordinateFrame() const noexcept;

    private:
        CoordinateContext(CoordinateContextId id,
                          int revision,
                          std::vector<SpatialReferenceDefinition> spatialReferences,
                          std::vector<CoordinateFrame> frames,
                          std::optional<SpatialReferenceId> solverSpatialReferenceId,
                          SolverFrameDefinition solverFrame,
                          std::string contextHash);

        CoordinateContextId _id;
        int _revision;
        std::vector<SpatialReferenceDefinition> _spatialReferences;
        std::vector<CoordinateFrame> _frames;
        std::optional<SpatialReferenceId> _solverSpatialReferenceId;
        SolverFrameDefinition _solverFrame;
        std::string _contextHash;
    };

} // namespace xjw::coordinate_system
