#include "RecoveredDepthScene.h"
#include "RecoveredModelInput.h"
#include "io/PathIO.h"
#include <QDir>
#include <QFileInfo>
#include <QScopeGuard>
#include <QUuid>

#include "metmodel/patchmatch.hpp"
#include "metmodel/neighbor_selection.hpp"
#include "metmodel/patchmatch_store.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace xjw::mvs
{
    FramePinholeCamera recoveredPublicD4Camera(const FramePinholeCamera& source)
    {
        const auto original = source.normalizedForPositiveDepth();
        const auto intrinsics = original.intrinsics();
        auto result = original.scaledIntrinsics(0.25, 0.25);
        // Supplied depth.cpp::make_depth_camera_model: the published domain
        // removes b1/b2 and distortion; principal points use direct division.
        result.setIntrinsics(
            intrinsics.focalY / 4.0, intrinsics.focalY / 4.0, intrinsics.principalX / 4.0, intrinsics.principalY / 4.0);
        result.setDistortion(FramePinholeCamera::Distortion{});
        return result;
    }

    namespace
    {

        using metmodel::Camera;
        using metmodel::ReconstructionRegion;
        using metmodel::Scene;
        using metmodel::SparsePoint;

        void setError(std::string* errorMessage, std::string message)
        {
            if (errorMessage)
            {
                *errorMessage = std::move(message);
            }
        }

        std::filesystem::path uniquePatchMatchStoreRoot(const std::filesystem::path& workspaceRoot)
        {
            const QString workspace = xjw::common::io::fromFilesystemPath(workspaceRoot);
            for (int attempt = 0; attempt < 8; ++attempt)
            {
                const QString token = QUuid::createUuid().toString(QUuid::WithoutBraces);
                const auto candidate = xjw::common::io::toFilesystemPath(
                    QDir(workspace).filePath(QStringLiteral(".recovered_patchmatch_store-%1").arg(token)));
                std::error_code exists_error;
                if (!std::filesystem::exists(candidate, exists_error) && !exists_error)
                {
                    return candidate;
                }
            }
            throw std::runtime_error("cannot allocate a unique recovered PatchMatch store path");
        }

        bool makeScene(const std::vector<CameraView>& views,
                       const SparseCloud& sparseCloud,
                       Scene* scene,
                       std::string* errorMessage)
        {
            if (!scene || views.size() < 2 || sparseCloud.points.empty() ||
                sparseCloud.trackIds.size() != sparseCloud.points.size() ||
                sparseCloud.observingViewIndices.size() != sparseCloud.points.size())
            {
                setError(errorMessage, "recovered depth requires at least 2 cameras and a complete sparse track graph");
                return false;
            }
            Scene converted;
            converted.cameras.reserve(views.size());
            for (std::size_t index = 0; index < views.size(); ++index)
            {
                const CameraView& view = views[index];
                const FramePinholeCamera camera = view.camera.normalizedForPositiveDepth();
                if (!camera.isValid() || view.imageWidth <= 0 || view.imageHeight <= 0)
                {
                    setError(errorMessage, "recovered depth camera is invalid or has no raster dimensions");
                    return false;
                }
                const auto intrinsics = camera.intrinsics();
                const auto distortion = camera.distortion();
                Camera output;
                output.index = index;
                output.name = std::filesystem::path(view.imagePath).stem().string();
                output.path = view.preparedImagePath.empty() ? view.imagePath : view.preparedImagePath;
                output.aligned = true;
                output.model.f = intrinsics.focalY;
                output.model.b1 = intrinsics.focalX - intrinsics.focalY;
                output.model.cx = intrinsics.principalX;
                output.model.cy = intrinsics.principalY;
                output.model.k1 = distortion.radialK1;
                output.model.k2 = distortion.radialK2;
                output.model.k3 = distortion.radialK3;
                output.model.p1 = distortion.tangentialP1;
                output.model.p2 = distortion.tangentialP2;
                const auto rotation = camera.worldToCameraRotation();
                const auto translation = camera.worldToCameraTranslation();
                const auto center = camera.cameraCenter();
                output.pose.rotation.v = rotation;
                output.pose.translation = {translation[0], translation[1], translation[2]};
                output.pose.center = metalign::Vec3{center[0], center[1], center[2]};
                output.center = *output.pose.center;
                output.image.width = static_cast<std::size_t>(view.imageWidth);
                output.image.height = static_cast<std::size_t>(view.imageHeight);
                converted.cameras.push_back(std::move(output));
            }
            converted.sparse_points.reserve(sparseCloud.points.size());
            for (std::size_t index = 0; index < sparseCloud.points.size(); ++index)
            {
                const auto& point = sparseCloud.points[index];
                converted.sparse_points.push_back(
                    {{point[0], point[1], point[2]}, {255, 255, 255}, sparseCloud.trackIds[index], 1.0F});
                for (const int view_index : sparseCloud.observingViewIndices[index])
                {
                    if (view_index >= 0 && view_index < static_cast<int>(converted.cameras.size()))
                    {
                        converted.cameras[static_cast<std::size_t>(view_index)].track_ids.push_back(
                            sparseCloud.trackIds[index]);
                    }
                }
            }
            for (Camera& camera : converted.cameras)
            {
                std::sort(camera.track_ids.begin(), camera.track_ids.end());
                camera.track_ids.erase(std::unique(camera.track_ids.begin(), camera.track_ids.end()),
                                       camera.track_ids.end());
            }
            converted.region.specified = true;
            if (sparseCloud.reconstructionRegionSpecified)
            {
                converted.region.rotation = sparseCloud.reconstructionRegionRotation;
                converted.region.center = {sparseCloud.reconstructionRegionCenter[0],
                                           sparseCloud.reconstructionRegionCenter[1],
                                           sparseCloud.reconstructionRegionCenter[2]};
                converted.region.size = {sparseCloud.reconstructionRegionSize[0],
                                         sparseCloud.reconstructionRegionSize[1],
                                         sparseCloud.reconstructionRegionSize[2]};
            }
            else
            {
                converted.region.center = {(sparseCloud.minPt[0] + sparseCloud.maxPt[0]) * 0.5,
                                           (sparseCloud.minPt[1] + sparseCloud.maxPt[1]) * 0.5,
                                           (sparseCloud.minPt[2] + sparseCloud.maxPt[2]) * 0.5};
                converted.region.size = {
                    std::max(1.0e-6, static_cast<double>(sparseCloud.maxPt[0] - sparseCloud.minPt[0])),
                    std::max(1.0e-6, static_cast<double>(sparseCloud.maxPt[1] - sparseCloud.minPt[1])),
                    std::max(1.0e-6, static_cast<double>(sparseCloud.maxPt[2] - sparseCloud.minPt[2]))};
            }
            *scene = std::move(converted);
            return true;
        }

    } // namespace

    bool runRecoveredDepthScene(const std::vector<CameraView>& views,
                                const SparseCloud& sparseCloud,
                                int cudaDeviceIndex,
                                const std::string& workspaceRoot,
                                RecoveredDepthSceneResult* result,
                                std::string* errorMessage)
    {
#if defined(MVS_ENABLE_CUDA)
        if (!result)
        {
            setError(errorMessage, "recovered depth output pointer is null");
            return false;
        }
        if (workspaceRoot.empty())
        {
            setError(errorMessage, "recovered depth requires a workspace directory");
            return false;
        }
        const auto workspace_root = xjw::common::io::toFilesystemPath(workspaceRoot);
        std::filesystem::path patchmatch_store_root;
        try
        {
            patchmatch_store_root = uniquePatchMatchStoreRoot(workspace_root);
        }
        catch (const std::exception& exception)
        {
            setError(errorMessage, exception.what());
            return false;
        }
        bool patchmatch_store_removed = false;
        const auto cleanup_patchmatch_store = qScopeGuard(
            [&]()
            {
                if (!patchmatch_store_removed)
                {
                    std::error_code ignored;
                    std::filesystem::remove_all(patchmatch_store_root, ignored);
                }
            });
        Scene scene;
        if (!makeScene(views, sparseCloud, &scene, errorMessage))
        {
            return false;
        }
        std::vector<std::vector<std::size_t>> neighbors = metmodel::select_recovered_neighbors(scene, 16);
        std::vector<std::size_t> references(scene.cameras.size());
        std::iota(references.begin(), references.end(), 0U);
        for (std::size_t index = 0; index < neighbors.size(); ++index)
        {
            if (neighbors[index].empty() || neighbors[index].size() > 16)
            {
                setError(errorMessage,
                         "recovered depth camera " + std::to_string(index) + " has " +
                             std::to_string(neighbors[index].size()) +
                             " valid track-ranked neighbors; production requires 1..16");
                return false;
            }
        }
        metmodel::RecoveredPatchMatchD4SceneOutput recovered;
        std::string recovered_error;
        const std::size_t device_index = static_cast<std::size_t>(std::max(0, cudaDeviceIndex));
        if (!metmodel::run_recovered_patchmatch_d4_scene_cuda(scene,
                                                              references,
                                                              neighbors,
                                                              metmodel::FilterMode::Mild,
                                                              device_index,
                                                              recovered,
                                                              recovered_error,
                                                              false,
                                                              patchmatch_store_root))
        {
            setError(errorMessage, std::move(recovered_error));
            return false;
        }
        RecoveredDepthSceneResult converted;
        converted.patchMatchSeconds = recovered.patchmatch_seconds;
        converted.votingSeconds = recovered.voting_seconds;
        converted.frames.reserve(recovered.cameras.size());
        for (std::size_t ordinal = 0; ordinal < recovered.cameras.size(); ++ordinal)
        {
            const std::size_t camera_index = references[ordinal];
            const auto& camera_output = recovered.cameras[ordinal];
            const int width = (views[camera_index].imageWidth + 3) / 4;
            const int height = (views[camera_index].imageHeight + 3) / 4;
            if (camera_output.public_depth.size() != static_cast<std::size_t>(width * height))
            {
                setError(errorMessage, "recovered public depth dimensions do not match the d4 camera grid");
                return false;
            }
            RecoveredDepthFrame frame;
            frame.viewIndex = static_cast<int>(camera_index);
            frame.depth = cv::Mat(height, width, CV_32F, const_cast<float*>(camera_output.public_depth.data())).clone();
            frame.validMask = frame.depth > 0.0F;
            frame.confidence = cv::Mat::zeros(height, width, CV_32F);
            frame.confidence.setTo(1.0F, frame.validMask);
            frame.photometricSourceMask = cv::Mat::zeros(height, width, CV_32S);
            metmodel::RecoveredPatchMatchD4PyramidOutput stored_patchmatch;
            std::string store_error;
            if (!metmodel::read_recovered_patchmatch_store_camera(
                    patchmatch_store_root, camera_index, stored_patchmatch, store_error))
            {
                setError(errorMessage,
                         "cannot read recovered PatchMatch store for camera " + std::to_string(camera_index) + ": " +
                             store_error);
                return false;
            }
            const std::size_t pixel_count = static_cast<std::size_t>(width * height);
            for (std::size_t source_rank = 0; source_rank < neighbors[camera_index].size(); ++source_rank)
            {
                std::vector<std::uint8_t> inlier_mask;
                std::string unpack_error;
                if (!metmodel::unpack_recovered_patchmatch_inlier_mask(stored_patchmatch.packed_inlier_masks[0],
                                                                       pixel_count,
                                                                       neighbors[camera_index].size(),
                                                                       source_rank,
                                                                       inlier_mask,
                                                                       unpack_error))
                {
                    setError(errorMessage,
                             "cannot unpack recovered photometric source mask for camera " +
                                 std::to_string(camera_index) + ": " + unpack_error);
                    return false;
                }
                const std::int32_t source_bit = static_cast<std::int32_t>(1U << source_rank);
                for (int row = 0; row < height; ++row)
                {
                    std::int32_t* destination = frame.photometricSourceMask.ptr<std::int32_t>(row);
                    const std::size_t row_offset = static_cast<std::size_t>(row * width);
                    for (int column = 0; column < width; ++column)
                    {
                        if (inlier_mask[row_offset + static_cast<std::size_t>(column)] != 0U)
                        {
                            destination[column] |= source_bit;
                        }
                    }
                }
            }
            frame.photometricSourceMask.setTo(0, frame.validMask == 0);
            frame.camera = recoveredPublicD4Camera(views[camera_index].camera);
            for (const std::size_t source : neighbors[camera_index])
            {
                frame.sourceViewIndices.push_back(static_cast<int>(source));
            }
            converted.frames.push_back(std::move(frame));
        }
        try
        {
            const QString model_root = QDir(xjw::common::io::fromFilesystemPath(workspace_root))
                                           .filePath(QStringLiteral("recovered_model_input"));
            writeRecoveredModelInput(model_root, scene, recovered, true);
        }
        catch (const std::exception& exception)
        {
            setError(errorMessage, exception.what());
            return false;
        }
        std::error_code remove_error;
        const std::uintmax_t removed_entries = std::filesystem::remove_all(patchmatch_store_root, remove_error);
        if (remove_error || removed_entries == 0U)
        {
            setError(errorMessage, "cannot remove consumed recovered PatchMatch store");
            return false;
        }
        patchmatch_store_removed = true;
        *result = std::move(converted);
        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
#else
        (void)views;
        (void)sparseCloud;
        (void)cudaDeviceIndex;
        (void)workspaceRoot;
        (void)result;
        setError(errorMessage, "recovered depth requires a CUDA build");
        return false;
#endif
    }

} // namespace xjw::mvs
