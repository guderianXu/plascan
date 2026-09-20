#pragma once

#include "CameraProjectStore.h"

#include "camera/core/capabilities/CameraOperationPlan.h"
#include "camera/core/model/CameraInstanceSet.h"
#include "camera/core/model/CameraModelRegistry.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"

#include <QStringList>

#include <string>
#include <vector>

namespace xjw::camera_project
{

    /**
     * Runtime representation of the canonical project camera collections.
     *
     * CameraProjectStore validates the document shape.  This loader performs
     * the second step: it asks the model registry to decode every definition
     * and instance, so malformed model-specific state cannot reach a solver.
     * The loader never consults images[*].camera.
     */
    struct CameraProjectRuntimeResult
    {
        camera_core::CameraInstanceSet instances;
        QStringList errors;

        /**
         * Select canonical instances and evaluate one workflow contract in
         * the same operation.  The returned plan is never based on a
         * frame-pinhole conversion, so unsupported models remain visible to
         * the caller as capability failures.
         */
        camera_core::CameraOperationPlan planOperationForImages(const std::vector<camera_core::ImageId>& images,
                                                                camera_core::CameraOperation operation) const;

        /**
         * Resolve one canonical instance into the solver-owned frame-pinhole
         * state.  The lookup is identity-based; model-specific instances that
         * do not implement frame-pinhole semantics are rejected explicitly.
         */
        bool framePinholeStateForImage(const camera_core::ImageId& image,
                                       camera_models::frame_pinhole::FramePinholeNumericState* state,
                                       std::string* error = nullptr) const;

        /**
         * Resolve an ordered, complete set of frame-pinhole states by
         * canonical ImageId.  The output is left empty when any requested
         * identity is duplicated, missing, or not convertible.
         */
        bool framePinholeStatesForImages(const std::vector<camera_core::ImageId>& images,
                                         std::vector<camera_models::frame_pinhole::FramePinholeNumericState>* states,
                                         std::string* error = nullptr) const;

        bool ok() const noexcept
        {
            return errors.isEmpty();
        }
    };

    class CameraProjectRuntime
    {
    public:
        static CameraProjectRuntimeResult load(const QJsonObject& projectFiles,
                                               const camera_core::CameraModelRegistry& registry);
    };

} // namespace xjw::camera_project
