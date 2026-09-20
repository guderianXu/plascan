#include "CoordinateContext.h"

#include "CoordinateHash.h"
#include "coordinate_system/types/CoordinateErrors.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace xjw::coordinate_system
{
    namespace
    {

        template <typename Value> void appendValue(std::ostringstream& stream, const Value& value)
        {
            stream << value << '\n';
        }

        void appendString(std::ostringstream& stream, const std::string& value)
        {
            stream << value.size() << ':' << value << '\n';
        }

        bool referenceMatchesFrame(const SpatialReferenceDefinition& reference, const CoordinateFrame& frame)
        {
            switch (reference.kind())
            {
            case SpatialReferenceKind::Geographic2d:
            case SpatialReferenceKind::Geographic3d:
                return frame.kind == CoordinateFrameKind::Geodetic;
            case SpatialReferenceKind::Geocentric3d:
                return frame.kind == CoordinateFrameKind::Ecef || frame.kind == CoordinateFrameKind::BodyFixed;
            case SpatialReferenceKind::Projected2d:
            case SpatialReferenceKind::Projected3d:
                return frame.kind == CoordinateFrameKind::LocalCartesian;
            case SpatialReferenceKind::Engineering3d:
                return frame.kind == CoordinateFrameKind::LocalCartesian || frame.kind == CoordinateFrameKind::LocalEnu;
            }
            return false;
        }

        std::string contextDigest(const CoordinateContextId& id,
                                  int revision,
                                  const std::vector<SpatialReferenceDefinition>& references,
                                  const std::vector<CoordinateFrame>& frames,
                                  const std::optional<SpatialReferenceId>& solverReferenceId,
                                  const SolverFrameDefinition& solverFrame)
        {
            std::ostringstream stream;
            stream.imbue(std::locale::classic());
            stream << std::setprecision(std::numeric_limits<double>::max_digits10);
            appendValue(stream, CoordinateContext::SchemaVersion);
            appendString(stream, id.value());
            appendValue(stream, revision);
            appendValue(stream, references.size());
            for (const SpatialReferenceDefinition& reference : references)
            {
                appendString(stream, reference.id().value());
                appendString(stream, reference.frameId().value());
                appendValue(stream, static_cast<int>(reference.kind()));
                appendValue(stream, static_cast<int>(reference.axisMapping()));
                appendValue(stream, static_cast<int>(reference.verticalReference()));
                appendString(stream, reference.canonicalDefinition());
                appendString(stream, reference.authority());
                appendString(stream, reference.authorityCode());
                appendValue(stream, reference.linearUnitToMetres());
            }
            appendValue(stream, frames.size());
            for (const CoordinateFrame& frame : frames)
            {
                appendString(stream, frame.id.value());
                appendValue(stream, static_cast<int>(frame.kind));
                appendValue(stream, static_cast<int>(frame.linearUnit));
                appendValue(stream, static_cast<int>(frame.angleUnit));
                appendValue(stream, frame.parent.has_value());
                if (frame.parent)
                {
                    appendString(stream, frame.parent->value());
                }
                for (double value : frame.toParent.rotation)
                {
                    appendValue(stream, value);
                }
                for (double value : frame.toParent.translation)
                {
                    appendValue(stream, value);
                }
            }
            appendValue(stream, solverReferenceId.has_value());
            if (solverReferenceId)
            {
                appendString(stream, solverReferenceId->value());
            }
            appendString(stream, solverFrame.frameId.value());
            appendValue(stream, static_cast<int>(solverFrame.scaleStatus));
            appendString(stream, solverFrame.normalizationHash);
            return sha256Hash(stream.str());
        }

    } // namespace

    CoordinateContext CoordinateContext::create(CoordinateContextId id,
                                                int revision,
                                                std::vector<SpatialReferenceDefinition> spatialReferences,
                                                std::vector<CoordinateFrame> frames,
                                                std::optional<SpatialReferenceId> solverSpatialReferenceId,
                                                SolverFrameDefinition solverFrame)
    {
        if (revision <= 0)
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                            "coordinate context revision must be positive");
        }
        std::sort(spatialReferences.begin(),
                  spatialReferences.end(),
                  [](const auto& left, const auto& right) { return left.id().value() < right.id().value(); });
        std::sort(frames.begin(),
                  frames.end(),
                  [](const CoordinateFrame& left, const CoordinateFrame& right)
                  { return left.id.value() < right.id.value(); });
        std::unordered_set<std::string> ids;
        std::unordered_set<std::string> definitions;
        for (const SpatialReferenceDefinition& reference : spatialReferences)
        {
            if (!ids.insert(reference.id().value()).second)
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "coordinate context contains a duplicate spatial reference id");
            }
            const std::string semantic_key = reference.canonicalDefinitionHash() + "\n" +
                                             std::to_string(static_cast<int>(reference.verticalReference())) + "\n" +
                                             reference.frameId().value();
            if (!definitions.insert(semantic_key).second)
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "coordinate context contains duplicate spatial reference semantics");
            }
        }

        std::unordered_set<std::string> frame_ids;
        for (const CoordinateFrame& frame : frames)
        {
            if (!frame_ids.insert(frame.id.value()).second)
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "coordinate context contains a duplicate frame id");
            }
        }
        for (const CoordinateFrame& frame : frames)
        {
            if (frame.parent && !frame_ids.contains(frame.parent->value()))
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "coordinate frame parent is not registered in the context");
            }
        }
        for (const CoordinateFrame& frame : frames)
        {
            std::unordered_set<std::string> ancestry;
            const CoordinateFrame* current = &frame;
            while (current)
            {
                if (!ancestry.insert(current->id.value()).second)
                {
                    throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                    "coordinate context frame graph contains a cycle");
                }
                if (!current->parent)
                {
                    break;
                }
                const auto parent =
                    std::find_if(frames.begin(),
                                 frames.end(),
                                 [&](const CoordinateFrame& candidate) { return candidate.id == *current->parent; });
                current = parent == frames.end() ? nullptr : &*parent;
            }
        }
        for (const SpatialReferenceDefinition& reference : spatialReferences)
        {
            const auto frame =
                std::find_if(frames.begin(),
                             frames.end(),
                             [&](const CoordinateFrame& candidate) { return candidate.id == reference.frameId(); });
            if (frame == frames.end())
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "spatial reference frame is not registered in the context");
            }
            if (!referenceMatchesFrame(reference, *frame))
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "spatial reference kind is incompatible with its registered frame");
            }
        }
        const auto solver_frame_found =
            std::find_if(frames.begin(),
                         frames.end(),
                         [&](const CoordinateFrame& frame) { return frame.id == solverFrame.frameId; });
        if (solver_frame_found == frames.end())
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                            "solver frame is not registered in the context");
        }

        const SpatialReferenceDefinition* solver_reference = nullptr;
        if (solverSpatialReferenceId)
        {
            const auto found = std::find_if(spatialReferences.begin(),
                                            spatialReferences.end(),
                                            [&](const auto& value) { return value.id() == *solverSpatialReferenceId; });
            if (found == spatialReferences.end())
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "solver spatial reference is not registered in the context");
            }
            solver_reference = &*found;
        }
        if (solverFrame.hasMetricScale())
        {
            if (solver_frame_found->linearUnit != LinearUnit::Metre || !solver_reference ||
                !solver_reference->isMetricCartesian() || solver_reference->frameId() != solverFrame.frameId)
            {
                throw CoordinateValidationError(CoordinateErrorCode::InvalidCoordinateContext,
                                                "metric solver frame requires a matching metre Cartesian reference");
            }
        }
        else if (solverSpatialReferenceId || solver_frame_found->linearUnit != LinearUnit::ProjectUnit)
        {
            throw CoordinateValidationError(
                CoordinateErrorCode::InvalidCoordinateContext,
                "unresolved solver frame must use project units without a spatial reference");
        }

        const std::string hash =
            contextDigest(id, revision, spatialReferences, frames, solverSpatialReferenceId, solverFrame);
        return CoordinateContext(std::move(id),
                                 revision,
                                 std::move(spatialReferences),
                                 std::move(frames),
                                 std::move(solverSpatialReferenceId),
                                 std::move(solverFrame),
                                 hash);
    }

    CoordinateContext::CoordinateContext(CoordinateContextId id,
                                         int revision,
                                         std::vector<SpatialReferenceDefinition> spatialReferences,
                                         std::vector<CoordinateFrame> frames,
                                         std::optional<SpatialReferenceId> solverSpatialReferenceId,
                                         SolverFrameDefinition solverFrame,
                                         std::string contextHash)
        : _id(std::move(id)), _revision(revision), _spatialReferences(std::move(spatialReferences)),
          _frames(std::move(frames)), _solverSpatialReferenceId(std::move(solverSpatialReferenceId)),
          _solverFrame(std::move(solverFrame)), _contextHash(std::move(contextHash))
    {
    }

    const CoordinateContextId& CoordinateContext::id() const noexcept
    {
        return _id;
    }
    int CoordinateContext::schemaVersion() const noexcept
    {
        return SchemaVersion;
    }
    int CoordinateContext::revision() const noexcept
    {
        return _revision;
    }
    const std::vector<SpatialReferenceDefinition>& CoordinateContext::spatialReferences() const noexcept
    {
        return _spatialReferences;
    }
    const std::vector<CoordinateFrame>& CoordinateContext::frames() const noexcept
    {
        return _frames;
    }
    const std::optional<SpatialReferenceId>& CoordinateContext::solverSpatialReferenceId() const noexcept
    {
        return _solverSpatialReferenceId;
    }
    const SolverFrameDefinition& CoordinateContext::solverFrame() const noexcept
    {
        return _solverFrame;
    }
    const std::string& CoordinateContext::contextHash() const noexcept
    {
        return _contextHash;
    }

    const SpatialReferenceDefinition*
    CoordinateContext::findSpatialReference(const SpatialReferenceId& id) const noexcept
    {
        const auto found = std::find_if(_spatialReferences.begin(),
                                        _spatialReferences.end(),
                                        [&id](const auto& value) { return value.id() == id; });
        return found == _spatialReferences.end() ? nullptr : &*found;
    }

    const SpatialReferenceDefinition*
    CoordinateContext::findSpatialReferenceByDefinitionHash(std::string_view hash) const noexcept
    {
        const auto found = std::find_if(_spatialReferences.begin(),
                                        _spatialReferences.end(),
                                        [&](const auto& value) { return value.canonicalDefinitionHash() == hash; });
        return found == _spatialReferences.end() ? nullptr : &*found;
    }

    const SpatialReferenceDefinition*
    CoordinateContext::findSpatialReferenceByDefinitionHash(std::string_view hash,
                                                            VerticalReference verticalReference) const noexcept
    {
        const auto found = std::find_if(_spatialReferences.begin(),
                                        _spatialReferences.end(),
                                        [&](const auto& value) {
                                            return value.canonicalDefinitionHash() == hash &&
                                                   value.verticalReference() == verticalReference;
                                        });
        return found == _spatialReferences.end() ? nullptr : &*found;
    }

    const SpatialReferenceDefinition* CoordinateContext::solverSpatialReference() const noexcept
    {
        return _solverSpatialReferenceId ? findSpatialReference(*_solverSpatialReferenceId) : nullptr;
    }

    const CoordinateFrame* CoordinateContext::findFrame(const CoordinateFrameId& id) const noexcept
    {
        const auto found = std::find_if(
            _frames.begin(), _frames.end(), [&id](const CoordinateFrame& frame) { return frame.id == id; });
        return found == _frames.end() ? nullptr : &*found;
    }

    const CoordinateFrame* CoordinateContext::solverCoordinateFrame() const noexcept
    {
        return findFrame(_solverFrame.frameId);
    }

} // namespace xjw::coordinate_system
