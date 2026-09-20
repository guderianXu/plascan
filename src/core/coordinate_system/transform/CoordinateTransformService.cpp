#include "CoordinateTransformService.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace xjw::coordinate_system
{
    namespace
    {

        bool finiteVector(const std::array<double, 3>& values)
        {
            for (double value : values)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
            }
            return true;
        }

        void appendHashValue(std::ostringstream& stream, double value)
        {
            stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value << ';';
        }

    } // namespace

    void CoordinateTransformService::registerFrame(CoordinateFrame frame)
    {
        const std::string id = frame.id.value();
        if (_frames.find(id) != _frames.end())
        {
            throw CoordinateTransformError("coordinate frame is already registered: " + id);
        }

        if (frame.parent)
        {
            if (frame.parent->value() == id)
            {
                throw CoordinateTransformError("coordinate frame cannot be its own parent: " + id);
            }
            if (_frames.find(frame.parent->value()) == _frames.end())
            {
                throw CoordinateTransformError("coordinate frame parent is not registered: " + frame.parent->value());
            }
        }
        else if (!frame.toParent.isIdentity())
        {
            throw CoordinateTransformError("root coordinate frame must use the identity transform: " + id);
        }

        _frames.emplace(id, std::move(frame));
    }

    const CoordinateFrame& CoordinateTransformService::resolve(const CoordinateFrameId& id) const
    {
        const auto iterator = _frames.find(id.value());
        if (iterator == _frames.end())
        {
            throw CoordinateTransformError("unknown coordinate frame: " + id.value());
        }
        return iterator->second;
    }

    const CoordinateFrame& CoordinateTransformService::frame(const CoordinateFrameId& id) const
    {
        return resolve(id);
    }

    std::array<double, 3> CoordinateTransformService::transformPoint(const CoordinateFrameId& source,
                                                                     const CoordinateFrameId& target,
                                                                     const std::array<double, 3>& point) const
    {
        if (!finiteVector(point))
        {
            throw CoordinateTransformError("point coordinates must be finite");
        }

        const auto sourceChain = chainToRoot(source);
        const auto targetChain = chainToRoot(target);
        if (sourceChain.back() != targetChain.back())
        {
            throw CoordinateTransformError("coordinate frames do not belong to the same transform graph: " +
                                           source.value() + " -> " + target.value());
        }

        std::array<double, 3> result = toRoot(source, point);
        for (std::size_t index = targetChain.size() - 1; index > 0; --index)
        {
            const CoordinateFrame& child = resolve(targetChain[index - 1]);
            const CoordinateFrame& parent = resolve(targetChain[index]);
            std::array<double, 3> delta{result[0] - child.toParent.translation[0],
                                        result[1] - child.toParent.translation[1],
                                        result[2] - child.toParent.translation[2]};
            delta = applyRotation(transposeRotation(child.toParent.rotation), delta);
            const double scale = unitScale(parent.linearUnit, child.linearUnit);
            result = {delta[0] * scale, delta[1] * scale, delta[2] * scale};
        }
        return result;
    }

    std::array<double, 3> CoordinateTransformService::transformVector(const CoordinateFrameId& source,
                                                                      const CoordinateFrameId& target,
                                                                      const std::array<double, 3>& vector) const
    {
        if (!finiteVector(vector))
        {
            throw CoordinateTransformError("vector coordinates must be finite");
        }

        const auto sourceChain = chainToRoot(source);
        const auto targetChain = chainToRoot(target);
        if (sourceChain.back() != targetChain.back())
        {
            throw CoordinateTransformError("coordinate frames do not belong to the same transform graph: " +
                                           source.value() + " -> " + target.value());
        }

        std::array<double, 3> result = vectorToRoot(source, vector);
        for (std::size_t index = targetChain.size() - 1; index > 0; --index)
        {
            const CoordinateFrame& child = resolve(targetChain[index - 1]);
            const CoordinateFrame& parent = resolve(targetChain[index]);
            result = applyRotation(transposeRotation(child.toParent.rotation), result);
            const double scale = unitScale(parent.linearUnit, child.linearUnit);
            result = {result[0] * scale, result[1] * scale, result[2] * scale};
        }
        return result;
    }

    RotationMatrix3d CoordinateTransformService::transformRotation(const CoordinateFrameId& source,
                                                                   const CoordinateFrameId& target,
                                                                   const RotationMatrix3d& rotation) const
    {
        for (double value : rotation)
        {
            if (!std::isfinite(value))
            {
                throw CoordinateTransformError("rotation values must be finite");
            }
        }

        const auto sourceChain = chainToRoot(source);
        const auto targetChain = chainToRoot(target);
        if (sourceChain.back() != targetChain.back())
        {
            throw CoordinateTransformError("coordinate frames do not belong to the same transform graph: " +
                                           source.value() + " -> " + target.value());
        }

        const RotationMatrix3d rootFromSource = rotationToRoot(source);
        const RotationMatrix3d rootFromTarget = rotationToRoot(target);
        const RotationMatrix3d targetFromRoot = transposeRotation(rootFromTarget);
        return multiplyRotation(multiplyRotation(targetFromRoot, rootFromSource), rotation);
    }

    std::string CoordinateTransformService::transformChainHash(const CoordinateFrameId& source,
                                                               const CoordinateFrameId& target) const
    {
        const auto sourceChain = chainToRoot(source);
        const auto targetChain = chainToRoot(target);
        if (sourceChain.back() != targetChain.back())
        {
            throw CoordinateTransformError("coordinate frames do not belong to the same transform graph: " +
                                           source.value() + " -> " + target.value());
        }

        std::ostringstream canonical;
        canonical << "source=" << source.value() << ";target=" << target.value() << ';';
        const auto appendChain = [this, &canonical](const std::vector<CoordinateFrameId>& chain)
        {
            for (const CoordinateFrameId& id : chain)
            {
                const CoordinateFrame& current = resolve(id);
                canonical << id.value() << ':' << static_cast<int>(current.kind) << ':'
                          << static_cast<int>(current.linearUnit) << ':' << static_cast<int>(current.angleUnit) << ';';
                for (double value : current.toParent.rotation)
                {
                    appendHashValue(canonical, value);
                }
                for (double value : current.toParent.translation)
                {
                    appendHashValue(canonical, value);
                }
                if (current.parent)
                {
                    canonical << "parent=" << current.parent->value() << ';';
                }
            }
        };
        appendChain(sourceChain);
        appendChain(targetChain);

        std::uint64_t hash = 14695981039346656037ULL;
        for (const char value : canonical.str())
        {
            hash ^= static_cast<unsigned char>(value);
            hash *= 1099511628211ULL;
        }

        std::ostringstream result;
        result << std::hex << std::setw(16) << std::setfill('0') << hash;
        return result.str();
    }

    std::vector<CoordinateFrameId> CoordinateTransformService::chainToRoot(const CoordinateFrameId& id) const
    {
        std::vector<CoordinateFrameId> chain;
        std::unordered_set<std::string> visited;
        CoordinateFrameId current = id;
        while (true)
        {
            if (!visited.insert(current.value()).second)
            {
                throw CoordinateTransformError("coordinate frame graph contains a cycle at: " + current.value());
            }
            const CoordinateFrame& currentFrame = resolve(current);
            chain.push_back(current);
            if (!currentFrame.parent)
            {
                return chain;
            }
            current = *currentFrame.parent;
        }
    }

    std::array<double, 3> CoordinateTransformService::toRoot(const CoordinateFrameId& id,
                                                             const std::array<double, 3>& point) const
    {
        const auto chain = chainToRoot(id);
        std::array<double, 3> result = point;
        for (std::size_t index = 0; index + 1 < chain.size(); ++index)
        {
            const CoordinateFrame& child = resolve(chain[index]);
            const CoordinateFrame& parent = resolve(chain[index + 1]);
            const double scale = unitScale(child.linearUnit, parent.linearUnit);
            std::array<double, 3> scaled{result[0] * scale, result[1] * scale, result[2] * scale};
            result = applyRotation(child.toParent.rotation, scaled);
            for (std::size_t component = 0; component < result.size(); ++component)
            {
                result[component] += child.toParent.translation[component];
            }
        }
        return result;
    }

    std::array<double, 3> CoordinateTransformService::vectorToRoot(const CoordinateFrameId& id,
                                                                   const std::array<double, 3>& vector) const
    {
        const auto chain = chainToRoot(id);
        std::array<double, 3> result = vector;
        for (std::size_t index = 0; index + 1 < chain.size(); ++index)
        {
            const CoordinateFrame& child = resolve(chain[index]);
            const CoordinateFrame& parent = resolve(chain[index + 1]);
            const double scale = unitScale(child.linearUnit, parent.linearUnit);
            result = {result[0] * scale, result[1] * scale, result[2] * scale};
            result = applyRotation(child.toParent.rotation, result);
        }
        return result;
    }

    RotationMatrix3d CoordinateTransformService::rotationToRoot(const CoordinateFrameId& id) const
    {
        const auto chain = chainToRoot(id);
        RotationMatrix3d result{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        for (std::size_t index = 0; index + 1 < chain.size(); ++index)
        {
            const CoordinateFrame& child = resolve(chain[index]);
            result = multiplyRotation(child.toParent.rotation, result);
        }
        return result;
    }

    std::array<double, 3> CoordinateTransformService::applyRotation(const RotationMatrix3d& rotation,
                                                                    const std::array<double, 3>& vector)
    {
        return {rotation[0] * vector[0] + rotation[1] * vector[1] + rotation[2] * vector[2],
                rotation[3] * vector[0] + rotation[4] * vector[1] + rotation[5] * vector[2],
                rotation[6] * vector[0] + rotation[7] * vector[1] + rotation[8] * vector[2]};
    }

    RotationMatrix3d CoordinateTransformService::multiplyRotation(const RotationMatrix3d& first,
                                                                  const RotationMatrix3d& second)
    {
        RotationMatrix3d result{};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                double value = 0.0;
                for (int index = 0; index < 3; ++index)
                {
                    value += first[static_cast<std::size_t>(row * 3 + index)] *
                             second[static_cast<std::size_t>(index * 3 + column)];
                }
                result[static_cast<std::size_t>(row * 3 + column)] = value;
            }
        }
        return result;
    }

    RotationMatrix3d CoordinateTransformService::transposeRotation(const RotationMatrix3d& rotation)
    {
        return {rotation[0],
                rotation[3],
                rotation[6],
                rotation[1],
                rotation[4],
                rotation[7],
                rotation[2],
                rotation[5],
                rotation[8]};
    }

    double CoordinateTransformService::unitScale(LinearUnit from, LinearUnit to)
    {
        if (from == to)
        {
            return 1.0;
        }
        if (from == LinearUnit::ProjectUnit || to == LinearUnit::ProjectUnit)
        {
            throw CoordinateTransformError("cannot convert unresolved project units to physical units");
        }
        if (from == LinearUnit::Metre && to == LinearUnit::Kilometre)
        {
            return 0.001;
        }
        if (from == LinearUnit::Kilometre && to == LinearUnit::Metre)
        {
            return 1000.0;
        }
        throw CoordinateTransformError("unsupported coordinate frame unit conversion");
    }

} // namespace xjw::coordinate_system
