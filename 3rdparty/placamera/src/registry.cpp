#include "placamera/registry.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <mutex>
#include <utility>

namespace placamera
{

    namespace
    {

        bool nonBlank(std::string_view value)
        {
            return !value.empty() && std::any_of(value.begin(),
                                                 value.end(),
                                                 [](unsigned char character) { return std::isspace(character) == 0; });
        }

    } // namespace

    ModelState
    ModelState::fromText(std::string modelType, int schemaVersion, std::string mediaType, std::string_view payload)
    {
        if (!nonBlank(modelType) || schemaVersion <= 0 || !nonBlank(mediaType))
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState,
                                        "model state requires a type, positive schema version, and media type");
        }

        std::vector<std::byte> bytes(payload.size());
        if (!payload.empty())
        {
            std::memcpy(bytes.data(), payload.data(), payload.size());
        }
        return ModelState{std::move(modelType), schemaVersion, std::move(mediaType), std::move(bytes)};
    }

    std::string ModelState::payloadAsText() const
    {
        if (payload.empty())
        {
            return {};
        }
        return {reinterpret_cast<const char*>(payload.data()), payload.size()};
    }

    Result<void> ModelRegistry::registerFactory(std::string modelType, Factory factory)
    {
        if (!nonBlank(modelType) || !factory)
        {
            return Result<void>::failure(CameraErrorCode::InvalidArgument,
                                         "camera model registration requires a non-blank type and factory");
        }
        std::unique_lock lock(_mutex);
        RegisteredFactory& registered = _factories[modelType];
        if (registered.createModel)
        {
            return Result<void>::failure(CameraErrorCode::InvalidModelState,
                                         "a complete-model factory is already registered for type '" + modelType + "'");
        }
        registered.createModel = std::move(factory);
        return Result<void>::success();
    }

    Result<void> ModelRegistry::registerCameraFactory(std::string modelType, CameraModelFactory factory)
    {
        if (!nonBlank(modelType) || !factory.createDefinition || !factory.createInstance)
        {
            return Result<void>::failure(
                CameraErrorCode::InvalidArgument,
                "camera model registration requires a non-blank type and definition/instance factories");
        }
        std::unique_lock lock(_mutex);
        RegisteredFactory& registered = _factories[modelType];
        if (registered.camera.createDefinition || registered.camera.createInstance)
        {
            return Result<void>::failure(CameraErrorCode::InvalidModelState,
                                         "definition/instance factories are already registered for type '" + modelType +
                                             "'");
        }
        registered.camera = std::move(factory);
        return Result<void>::success();
    }

    bool ModelRegistry::contains(std::string_view modelType) const
    {
        std::shared_lock lock(_mutex);
        return _factories.find(std::string(modelType)) != _factories.end();
    }

    std::vector<std::string> ModelRegistry::registeredModelTypes() const
    {
        std::vector<std::string> result;
        {
            std::shared_lock lock(_mutex);
            result.reserve(_factories.size());
            for (const auto& [model_type, factory] : _factories)
            {
                (void)factory;
                result.push_back(model_type);
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    Result<ModelRegistry::ModelPointer> ModelRegistry::create(const ModelState& state) const
    {
        if (!nonBlank(state.modelType) || state.schemaVersion <= 0 || !nonBlank(state.mediaType))
        {
            return Result<ModelPointer>::failure(
                CameraErrorCode::InvalidModelState,
                "model state requires a type, positive schema version, and media type");
        }

        Factory factory;
        {
            std::shared_lock lock(_mutex);
            const auto found = _factories.find(state.modelType);
            if (found == _factories.end())
            {
                return Result<ModelPointer>::failure(CameraErrorCode::UnsupportedModel,
                                                     "no camera model factory is registered for type '" +
                                                         state.modelType + "'");
            }
            factory = found->second.createModel;
            if (!factory)
            {
                return Result<ModelPointer>::failure(CameraErrorCode::UnsupportedModel,
                                                     "camera model type '" + state.modelType +
                                                         "' does not provide a complete-model factory");
            }
        }

        try
        {
            auto result = factory(state);
            if (result && !result.value())
            {
                return Result<ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                     "camera model factory returned a successful null model");
            }
            return result;
        }
        catch (const CameraValidationError& error)
        {
            return Result<ModelPointer>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                 std::string("camera model factory failed: ") + error.what());
        }
    }

    Result<CameraModelFactory::DefinitionPointer>
    ModelRegistry::createDefinition(const CameraDefinitionState& state) const
    {
        if (state.parameters.modelType.empty() || state.parameters.schemaVersion <= 0 ||
            !nonBlank(state.parameters.mediaType))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "camera definition state is malformed");
        }

        CameraModelFactory::DefinitionFactory factory;
        {
            std::shared_lock lock(_mutex);
            const auto found = _factories.find(state.parameters.modelType);
            if (found == _factories.end() || !found->second.camera.createDefinition)
            {
                return Result<CameraModelFactory::DefinitionPointer>::failure(
                    CameraErrorCode::UnsupportedModel,
                    "no camera definition factory is registered for type '" + state.parameters.modelType + "'");
            }
            factory = found->second.camera.createDefinition;
        }

        try
        {
            auto result = factory(state);
            if (result && !result.value())
            {
                return Result<CameraModelFactory::DefinitionPointer>::failure(
                    CameraErrorCode::InvalidModelState, "camera definition factory returned a successful null value");
            }
            return result;
        }
        catch (const CameraValidationError& error)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, std::string("camera definition factory failed: ") + error.what());
        }
    }

    Result<ModelRegistry::ModelPointer>
    ModelRegistry::createInstance(const CameraInstanceState& state,
                                  CameraModelFactory::DefinitionPointer definition) const
    {
        if (!definition)
        {
            return Result<ModelPointer>::failure(CameraErrorCode::InvalidArgument,
                                                 "camera instance factory requires a definition");
        }
        if (state.definitionId != definition->definitionId() || state.state.modelType != definition->modelType())
        {
            return Result<ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                 "camera instance state does not reference the supplied definition");
        }
        if (!state.imageSize.isValid() || state.state.schemaVersion <= 0 || !nonBlank(state.state.mediaType))
        {
            return Result<ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                 "camera instance state is malformed");
        }

        CameraModelFactory::InstanceFactory factory;
        {
            std::shared_lock lock(_mutex);
            const auto found = _factories.find(state.state.modelType);
            if (found == _factories.end() || !found->second.camera.createInstance)
            {
                return Result<ModelPointer>::failure(CameraErrorCode::UnsupportedModel,
                                                     "no camera instance factory is registered for type '" +
                                                         state.state.modelType + "'");
            }
            factory = found->second.camera.createInstance;
        }

        try
        {
            auto result = factory(state, std::move(definition));
            if (result && !result.value())
            {
                return Result<ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                     "camera instance factory returned a successful null model");
            }
            return result;
        }
        catch (const CameraValidationError& error)
        {
            return Result<ModelPointer>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                 std::string("camera instance factory failed: ") + error.what());
        }
    }

} // namespace placamera
