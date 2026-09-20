#include "CameraModelRegistry.h"

#include <sstream>
#include <utility>

namespace xjw::camera_core
{

    void CameraModelRegistry::registerFactory(std::string modelType, CameraModelFactory factory)
    {
        if (modelType.empty())
        {
            throw CameraRegistryError("camera model type must not be empty");
        }
        if (!factory.createDefinition || !factory.createInstance)
        {
            throw CameraRegistryError("camera model factory must provide definition and instance creators: " +
                                      modelType);
        }
        const auto [iterator, inserted] = _factories.emplace(std::move(modelType), std::move(factory));
        if (!inserted)
        {
            throw CameraRegistryError("camera model type is already registered: " + iterator->first);
        }
    }

    const CameraModelFactory& CameraModelRegistry::factoryFor(std::string_view modelType) const
    {
        const auto iterator = _factories.find(std::string(modelType));
        if (iterator == _factories.end())
        {
            throw CameraRegistryError("unknown camera model type: " + std::string(modelType));
        }
        return iterator->second;
    }

    std::unique_ptr<CameraDefinition>
    CameraModelRegistry::createDefinition(std::string_view modelType,
                                          const CameraDefinitionId& id,
                                          const xjw::coordinate_system::CoordinateFrameId& worldFrame,
                                          int parameterSchemaVersion,
                                          std::string_view serializedParameters) const
    {
        const CameraModelFactory& factory = factoryFor(modelType);
        std::unique_ptr<CameraDefinition> definition =
            factory.createDefinition(id, worldFrame, parameterSchemaVersion, serializedParameters);
        if (!definition)
        {
            throw CameraRegistryError("camera model factory returned a null definition: " + std::string(modelType));
        }
        if (definition->modelType() != modelType)
        {
            throw CameraRegistryError("camera definition model type does not match registry key: " +
                                      std::string(modelType));
        }
        return definition;
    }

    std::unique_ptr<CameraInstance>
    CameraModelRegistry::createInstance(std::string_view modelType,
                                        const CameraInstanceId& id,
                                        const ImageId& image,
                                        std::shared_ptr<const CameraDefinition> definition,
                                        std::string_view serializedState) const
    {
        if (!definition)
        {
            throw CameraRegistryError("camera instance requires a non-null definition");
        }
        if (definition->modelType() != modelType)
        {
            throw CameraRegistryError("camera instance model type does not match definition");
        }
        const CameraModelFactory& factory = factoryFor(modelType);
        std::unique_ptr<CameraInstance> instance =
            factory.createInstance(id, image, std::move(definition), serializedState);
        if (!instance)
        {
            throw CameraRegistryError("camera model factory returned a null instance: " + std::string(modelType));
        }
        return instance;
    }

} // namespace xjw::camera_core
