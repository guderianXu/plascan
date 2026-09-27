#pragma once

#include "placamera/model.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace placamera
{

    struct ModelState
    {
        std::string modelType;
        int schemaVersion = 0;
        std::string mediaType;
        std::vector<std::byte> payload;

        static ModelState
        fromText(std::string modelType, int schemaVersion, std::string mediaType, std::string_view payload);

        std::string payloadAsText() const;
    };

    struct CameraDefinitionState
    {
        CameraDefinitionId definitionId;
        FrameId groundFrame;
        ModelState parameters;
    };

    struct CameraInstanceState
    {
        CameraInstanceId instanceId;
        ImageId imageId;
        CameraDefinitionId definitionId;
        ImageSize imageSize;
        std::optional<TimeReference> captureTime;
        ModelState state;
    };

    struct CameraModelFactory
    {
        using DefinitionPointer = std::shared_ptr<const CameraDefinition>;
        using ModelPointer = RasterModelPtr;
        using DefinitionFactory = std::function<Result<DefinitionPointer>(const CameraDefinitionState&)>;
        using InstanceFactory = std::function<Result<ModelPointer>(const CameraInstanceState&, DefinitionPointer)>;

        DefinitionFactory createDefinition;
        InstanceFactory createInstance;
    };

    class ModelRegistry
    {
    public:
        using ModelPointer = RasterModelPtr;
        using Factory = std::function<Result<ModelPointer>(const ModelState&)>;

        Result<void> registerFactory(std::string modelType, Factory factory);
        Result<void> registerCameraFactory(std::string modelType, CameraModelFactory factory);
        bool contains(std::string_view modelType) const;
        std::vector<std::string> registeredModelTypes() const;

        Result<ModelPointer> create(const ModelState& state) const;
        Result<CameraModelFactory::DefinitionPointer> createDefinition(const CameraDefinitionState& state) const;
        Result<ModelPointer> createInstance(const CameraInstanceState& state,
                                            CameraModelFactory::DefinitionPointer definition) const;

    private:
        struct RegisteredFactory
        {
            Factory createModel;
            CameraModelFactory camera;
        };

        mutable std::shared_mutex _mutex;
        std::unordered_map<std::string, RegisteredFactory> _factories;
    };

} // namespace placamera
