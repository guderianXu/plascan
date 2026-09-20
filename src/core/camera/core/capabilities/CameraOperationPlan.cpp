#include "CameraOperationPlan.h"

#include <sstream>

namespace xjw::camera_core
{
    namespace
    {

        const CameraOperationRequirements kStaticRequirements{CapabilitySet{CapabilityKind::Projection,
                                                                            CapabilityKind::Ray,
                                                                            CapabilityKind::StaticPose,
                                                                            CapabilityKind::Optimization},
                                                              true};

        const CameraOperationRequirements kOrthoRequirements{
            CapabilitySet{CapabilityKind::Projection, CapabilityKind::StaticPose}, true};

        const CameraOperationRequirements kRpcRequirements{
            CapabilitySet{CapabilityKind::Projection, CapabilityKind::InverseProjection, CapabilityKind::Ray}, true};

        const CameraOperationRequirements kPushbroomRequirements{CapabilitySet{CapabilityKind::Projection,
                                                                               CapabilityKind::Ray,
                                                                               CapabilityKind::Trajectory,
                                                                               CapabilityKind::Optimization},
                                                                 true};

        const CameraOperationRequirements kExternalPoseRequirements{CapabilitySet{CapabilityKind::StaticPose}, true};

    } // namespace

    const CameraOperationRequirements& requirementsFor(CameraOperation operation) noexcept
    {
        switch (operation)
        {
        case CameraOperation::StaticSfM:
        case CameraOperation::BundleAdjustment:
        case CameraOperation::DenseMvs:
            return kStaticRequirements;
        case CameraOperation::OrthoProjection:
            return kOrthoRequirements;
        case CameraOperation::RpcAerialTriangulation:
            return kRpcRequirements;
        case CameraOperation::PushbroomAerialTriangulation:
            return kPushbroomRequirements;
        case CameraOperation::ExternalPoseReference:
            return kExternalPoseRequirements;
        }
        return kStaticRequirements;
    }

    const char* cameraOperationName(CameraOperation operation) noexcept
    {
        switch (operation)
        {
        case CameraOperation::StaticSfM:
            return "static_sfm";
        case CameraOperation::BundleAdjustment:
            return "bundle_adjustment";
        case CameraOperation::DenseMvs:
            return "dense_mvs";
        case CameraOperation::OrthoProjection:
            return "ortho_projection";
        case CameraOperation::RpcAerialTriangulation:
            return "rpc_aerial_triangulation";
        case CameraOperation::PushbroomAerialTriangulation:
            return "pushbroom_aerial_triangulation";
        case CameraOperation::ExternalPoseReference:
            return "external_pose_reference";
        }
        return "unknown";
    }

    CameraOperationPlan planCameraOperation(const CameraInstanceSet& instances, CameraOperation operation)
    {
        CameraOperationPlan plan;
        plan.operation = operation;
        plan.requirements = requirementsFor(operation);

        if (instances.empty())
        {
            plan.inputError = "camera operation requires at least one camera instance";
            return plan;
        }

        const CameraInstanceSetCapabilityResult capabilities =
            instances.requireCapabilities(plan.requirements.capabilities);
        for (const CameraInstanceSetCapabilityResult::Failure& failure : capabilities.failures)
        {
            plan.capabilityFailures.push_back({failure.image, failure.message});
            plan.failures.push_back(plan.capabilityFailures.back());
        }

        if (plan.requirements.requireCommonWorldFrame)
        {
            const CameraInstanceSetFrameResult frames = instances.requireCommonWorldFrame();
            plan.commonWorldFrame = frames.commonFrame;
            for (const CameraInstanceSetFrameResult::Failure& failure : frames.failures)
            {
                plan.frameFailures.push_back({failure.image, failure.reason});
                plan.failures.push_back(plan.frameFailures.back());
            }
        }
        return plan;
    }

    std::string CameraOperationPlan::failureMessage() const
    {
        if (inputError.empty() && failures.empty())
        {
            return {};
        }

        std::ostringstream stream;
        stream << "camera operation '" << cameraOperationName(operation) << "' is not supported";
        if (!inputError.empty())
        {
            stream << "; " << inputError;
        }
        for (const Failure& failure : failures)
        {
            stream << "; image " << failure.image.value() << ": " << failure.reason;
        }
        return stream.str();
    }

} // namespace xjw::camera_core
