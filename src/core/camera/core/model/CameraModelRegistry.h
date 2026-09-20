#pragma once

#include "CameraDefinition.h"
#include "CameraInstance.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace xjw::camera_core
{

    class CameraRegistryError : public std::runtime_error
    {
    public:
        explicit CameraRegistryError(const std::string& message) : std::runtime_error(message)
        {
        }
    };

    struct CameraModelFactory
    {
        using DefinitionFactory = std::function<std::unique_ptr<CameraDefinition>(
            const CameraDefinitionId&, const xjw::coordinate_system::CoordinateFrameId&, int, std::string_view)>;
        using InstanceFactory = std::function<std::unique_ptr<CameraInstance>(
            const CameraInstanceId&, const ImageId&, std::shared_ptr<const CameraDefinition>, std::string_view)>;

        DefinitionFactory createDefinition;
        InstanceFactory createInstance;
    };

    class CameraModelRegistry
    {
    public:
        void registerFactory(std::string modelType, CameraModelFactory factory);

        std::unique_ptr<CameraDefinition> createDefinition(std::string_view modelType,
                                                           const CameraDefinitionId& id,
                                                           const xjw::coordinate_system::CoordinateFrameId& worldFrame,
                                                           int parameterSchemaVersion,
                                                           std::string_view serializedParameters) const;

        std::unique_ptr<CameraInstance> createInstance(std::string_view modelType,
                                                       const CameraInstanceId& id,
                                                       const ImageId& image,
                                                       std::shared_ptr<const CameraDefinition> definition,
                                                       std::string_view serializedState) const;

    private:
        const CameraModelFactory& factoryFor(std::string_view modelType) const;

        std::unordered_map<std::string, CameraModelFactory> _factories;
    };

} // namespace xjw::camera_core
