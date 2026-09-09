#include "RecoveredModelBuilder.h"
#include "RecoveredModelInput.h"
#include "GpuDeviceLease.h"
#include "PatchMatchCUDA.h"
#include "metmodel/mesh.hpp"
#include "metmodel/model_pipeline.hpp"
#include "recovered_model/RecoveredModelColorizer.h"

#include <QJsonArray>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace xjw::mesh
{
    namespace
    {
        constexpr std::array<double, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

        void checkpoint(const std::function<bool()>& cancelled,
                        const std::function<void(const QString&, int)>& progress,
                        const char* stage,
                        int percent)
        {
            if (cancelled && cancelled())
            {
                throw std::runtime_error("Recovered model cancelled");
            }
            if (progress)
            {
                progress(QString::fromUtf8(stage), percent);
            }
        }

        metmodel::Mesh makeMesh(const metmodel::OocMarchingRawOutput& raw)
        {
            metmodel::Mesh mesh;
            mesh.vertices.resize(raw.vertices.size());
            for (std::size_t i = 0; i < raw.vertices.size(); ++i)
            {
                const auto& p = raw.vertices[i].position;
                mesh.vertices[i].position = {p[0], p[1], p[2]};
            }
            mesh.faces.resize(raw.triangles.size());
            for (std::size_t i = 0; i < raw.triangles.size(); ++i)
            {
                for (std::size_t corner = 0; corner < 3; ++corner)
                {
                    mesh.faces[i].vertices[corner] = raw.triangles[i].vertices[corner];
                }
            }
            metmodel::recompute_normals(mesh);
            return mesh;
        }

        std::array<double, 15> regionValues(const metmodel::ReconstructionRegion& region)
        {
            std::array<double, 15> result{};
            std::copy(region.rotation.begin(), region.rotation.end(), result.begin());
            result[9] = region.center.x;
            result[10] = region.center.y;
            result[11] = region.center.z;
            result[12] = region.size.x;
            result[13] = region.size.y;
            result[14] = region.size.z;
            return result;
        }

        void restoreWorld(metmodel::Mesh& mesh, double rootScale, const std::array<double, 15>& region)
        {
            std::array<double, 16> transform{};
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    transform[row * 4 + column] = region[row * 3 + column] * rootScale;
                }
                const double offset =
                    region[row * 3] * region[12] + region[row * 3 + 1] * region[13] + region[row * 3 + 2] * region[14];
                transform[row * 4 + 3] = region[9 + row] - offset * 0.5;
            }
            for (auto& vertex : mesh.vertices)
            {
                const auto p = vertex.position;
                std::array<double, 3> result{};
                for (std::size_t row = 0; row < 3; ++row)
                {
                    volatile double x = transform[row * 4] * static_cast<double>(static_cast<float>(p.x));
                    volatile double xy = x + transform[row * 4 + 1] * static_cast<double>(static_cast<float>(p.y));
                    volatile double xyz = xy + transform[row * 4 + 2] * static_cast<double>(static_cast<float>(p.z));
                    result[row] = xyz + transform[row * 4 + 3];
                }
                vertex.position = {result[0], result[1], result[2]};
            }
        }
    } // namespace

    std::vector<std::uint32_t> planRecoveredSupportLevels(std::uint32_t maximumLevel, const QJsonValue& requestedLevels)
    {
        if (maximumLevel == 0U || maximumLevel >= 31U)
        {
            throw std::invalid_argument("Recovered balanced-tree maximum level is outside the model domain");
        }

        std::vector<std::uint32_t> levels;
        if (requestedLevels.isUndefined() || requestedLevels.isNull())
        {
            if (maximumLevel <= 6U)
            {
                return {maximumLevel};
            }
            const std::uint32_t first_level = (maximumLevel & 1U) == 0U ? 6U : 5U;
            for (std::uint32_t level = first_level; level <= maximumLevel; level += 2U)
            {
                levels.push_back(level);
            }
            return levels;
        }
        if (!requestedLevels.isArray())
        {
            throw std::invalid_argument("recovered_support_levels must be an integer array");
        }
        for (const auto value : requestedLevels.toArray())
        {
            const auto level = value.toInteger(-1);
            if (level <= 0 || level >= 31 || value.toDouble() != static_cast<double>(level))
            {
                throw std::invalid_argument("Invalid recovered support level");
            }
            levels.push_back(static_cast<std::uint32_t>(level));
        }
        if (levels.empty())
        {
            if (maximumLevel <= 6U)
            {
                return {maximumLevel};
            }
            throw std::invalid_argument("Recovered support levels cannot be empty above level six");
        }
        for (std::size_t index = 0; index < levels.size(); ++index)
        {
            if (levels[index] > maximumLevel ||
                (index > 0U && (levels[index] <= levels[index - 1U] || levels[index] - levels[index - 1U] != 2U)))
            {
                throw std::invalid_argument("Recovered support levels must be ordered and advance by two");
            }
        }
        if (levels.back() != maximumLevel)
        {
            throw std::invalid_argument("Recovered support levels must end at the balanced-tree maximum");
        }
        return levels;
    }

    RecoveredModelResult buildRecoveredModel(const QString& inputDirectory,
                                             const QJsonObject& settings,
                                             int targetFaces,
                                             const std::function<bool()>& isCancelled,
                                             const std::function<void(const QString&, int)>& progress)
    {
#if !defined(MVS_ENABLE_CUDA)
        throw std::runtime_error("Recovered OOC model requires a CUDA build");
#else
        const auto started = std::chrono::steady_clock::now();
        const QString backend = settings.value(QStringLiteral("compute_mode")).toString(QStringLiteral("auto"));
        if (backend != QStringLiteral("auto") && backend != QStringLiteral("cuda") &&
            backend != QStringLiteral("hybrid"))
        {
            throw std::runtime_error("Recovered OOC model requires CUDA (compute_mode=auto/hybrid/cuda)");
        }
        const int device = settings.value(QStringLiteral("compute_device_index")).toInt(0);
        if (device < 0)
        {
            throw std::runtime_error("Recovered OOC CUDA device index must be nonnegative");
        }
        checkpoint(isCancelled, progress, "读取三层投票深度与模型相机…", 2);
        if (settings.value(QStringLiteral("strictVolumetricMasks")).toBool(false) ||
            settings.value(QStringLiteral("splitIntoBlocks")).toBool(false) ||
            settings.value(QStringLiteral("interpolation")).toString(QStringLiteral("enabled")) !=
                QStringLiteral("enabled"))
        {
            throw std::runtime_error(
                "Recovered OOC currently requires interpolation=enabled, no volumetric masks, and a single part");
        }
        xjw::mvs::GpuDeviceLeaseSet lease;
        QString lease_error;
        const auto gpu_identity = xjw::mvs::PatchMatchDepthEstimator::cudaDeviceIdentity(device);
        if (gpu_identity.empty() ||
            !lease.acquire({{gpu_identity, xjw::mvs::PatchMatchDepthEstimator::cudaDeviceName(device)}}, &lease_error))
        {
            throw std::runtime_error("Cannot acquire recovered OOC CUDA device: " + lease_error.toStdString());
        }
        metmodel::Scene scene;
        metmodel::RecoveredPatchMatchD4SceneOutput depth;
        xjw::mvs::readRecoveredModelInput(inputDirectory, scene, depth);
        const auto expected_cameras = settings.value(QStringLiteral("recovered_expected_camera_count")).toInteger(-1);
        if (expected_cameras >= 0 && static_cast<std::size_t>(expected_cameras) != scene.cameras.size())
        {
            throw std::runtime_error("Recovered model input camera count differs from completed depth workspace");
        }
        std::vector<std::size_t> references(scene.cameras.size());
        const auto expected_poses = settings.value(QStringLiteral("recovered_expected_camera_poses"));
        if (!expected_poses.isUndefined())
        {
            const auto poses = expected_poses.toArray();
            if (static_cast<std::size_t>(poses.size()) != scene.cameras.size())
            {
                throw std::runtime_error("Recovered model pose list differs from depth workspace");
            }
            for (std::size_t i = 0; i < scene.cameras.size(); ++i)
            {
                const auto pose = poses[static_cast<qsizetype>(i)].toArray();
                const auto& camera = scene.cameras[i];
                std::array<double, 12> actual{};
                std::copy(camera.pose.rotation.v.begin(), camera.pose.rotation.v.end(), actual.begin());
                actual[9] = camera.pose.translation.x;
                actual[10] = camera.pose.translation.y;
                actual[11] = camera.pose.translation.z;
                if (pose.size() != 12)
                    throw std::runtime_error("Incomplete recovered depth camera pose");
                for (qsizetype j = 0; j < 12; ++j)
                {
                    if (!pose[j].isDouble() || !std::isfinite(pose[j].toDouble()) ||
                        std::abs(pose[j].toDouble() - actual[static_cast<std::size_t>(j)]) >
                            1e-10 * std::max(1.0, std::abs(actual[static_cast<std::size_t>(j)])))
                    {
                        throw std::runtime_error("Recovered model camera pose differs from depth workspace");
                    }
                }
            }
        }
        std::iota(references.begin(), references.end(), 0U);
        metmodel::RecoveredD4VotingToOocMode0Input input;
        input.reference_camera_indices = references;
        const bool diagonal = settings.value(QStringLiteral("recovered_diagonal_scale")).toBool(true);
        input.diagonal_policy = diagonal ? metmodel::RecoveredOocDiagonalPolicy::DeterministicEnabled
                                         : metmodel::RecoveredOocDiagonalPolicy::DeterministicDisabled;
        checkpoint(isCancelled, progress, "构造 OOC 深度与采样尺度金字塔…", 8);
        auto pyramid = metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0_consuming(scene, depth, input);
        depth = {};
        pyramid.ooc.bundle = {};
        // Compressed cache diagnostics are not needed by the in-process solver.
        // Keep the decoded pyramids used by weighted-node and histogram stages.
        checkpoint(isCancelled, progress, "归并 Morton 节点并平衡八叉树…", 18);
        auto weighted = metmodel::build_recovered_ooc_weighted_nodes_mode0(scene, pyramid);
        if (weighted.balanced_nodes.empty())
        {
            throw std::runtime_error("Recovered OOC has no multi-camera supported nodes");
        }
        std::uint32_t maximum_level = 0;
        for (const auto& node : weighted.balanced_nodes)
        {
            maximum_level = std::max(maximum_level, static_cast<std::uint32_t>(node.level));
        }
        const std::vector<std::uint32_t> levels =
            planRecoveredSupportLevels(maximum_level, settings.value(QStringLiteral("recovered_support_levels")));
        RecoveredModelResult result;
        result.diagnostics[QStringLiteral("actual_mesh_algorithm")] = QStringLiteral("recovered_ooc");
        result.diagnostics[QStringLiteral("fusion_backend")] = QStringLiteral("cuda");
        result.diagnostics[QStringLiteral("fusion_device_index")] = device;
        result.diagnostics[QStringLiteral("recovered_ooc_camera_pyramid_execution")] =
            pyramid.camera_pyramid_execution == metmodel::RecoveredOocCameraPyramidExecution::Parallel
                ? QStringLiteral("parallel")
                : QStringLiteral("serial");
        result.diagnostics[QStringLiteral("recovered_ooc_camera_pyramid_seconds")] =
            pyramid.camera_pyramid_wall_seconds;
        result.diagnostics[QStringLiteral("recovered_ooc_bundle_serialization_seconds")] =
            pyramid.bundle_serialization_wall_seconds;
        result.diagnostics[QStringLiteral("recovered_diagonal_scale")] = diagonal;
        result.diagnostics[QStringLiteral("recovered_balanced_nodes")] =
            static_cast<double>(weighted.balanced_nodes.size());
        result.diagnostics[QStringLiteral("recovered_root_scale")] = weighted.root_scale;
        result.diagnostics[QStringLiteral("recovered_maximum_level")] = static_cast<int>(maximum_level);
        QJsonArray stage_json;
        for (const auto level : levels)
        {
            stage_json.append(static_cast<int>(level));
        }
        result.diagnostics[QStringLiteral("recovered_support_levels")] = stage_json;
        decltype(weighted.merged_nodes){}.swap(weighted.merged_nodes);
        decltype(weighted.multi_camera_nodes){}.swap(weighted.multi_camera_nodes);
        decltype(weighted.records_before_histogram){}.swap(weighted.records_before_histogram);
        checkpoint(isCancelled, progress, "CUDA 多相机直方图融合…", 30);
        auto histogram = metmodel::build_recovered_ooc_histogram_mode0_cuda_consuming(
            scene, pyramid, weighted, static_cast<std::size_t>(device));
        pyramid = {};
        decltype(histogram.histogram_voxels){}.swap(histogram.histogram_voxels);
        decltype(histogram.selected_indices){}.swap(histogram.selected_indices);
        decltype(histogram.active){}.swap(histogram.active);
        checkpoint(isCancelled, progress, "多层隐式场求解与自适应网格提取…", 45);
        metmodel::OocFusionCudaStats cuda_stats;
        metmodel::OocFusionParameters fusion_parameters;
        fusion_parameters.is_cancelled = isCancelled;
        fusion_parameters.stage_progress = [&](std::uint32_t level)
        {
            if (progress)
            {
                progress(QStringLiteral("求解 OOC 第 %1 层隐式场（200 轮）…").arg(level),
                         45 + static_cast<int>(level) * 25 / static_cast<int>(maximum_level));
            }
        };
        auto model = metmodel::run_recovered_ooc_multilevel_model_cuda(histogram.records,
                                                                       weighted.balanced_nodes,
                                                                       histogram.scalar_lut,
                                                                       levels,
                                                                       1.0,
                                                                       identity,
                                                                       static_cast<std::size_t>(device),
                                                                       cuda_stats,
                                                                       fusion_parameters);
        const double root_scale = weighted.root_scale;
        weighted = {};
        histogram = {};
        auto mesh = makeMesh(model.raw_mesh);
        if (mesh.vertices.empty() || mesh.faces.empty())
        {
            throw std::runtime_error("Recovered adaptive marching produced an empty mesh");
        }
        auto attributes = std::move(model.raw_mesh.vertex_trim_attribute);
        auto scales = std::move(model.raw_mesh.vertex_scale);
        result.diagnostics[QStringLiteral("recovered_raw_faces")] = static_cast<double>(mesh.faces.size());
        model = {};
        checkpoint(isCancelled, progress, "面积加权 QEM 简化…", 76);
        if (targetFaces > 0)
        {
            std::array<double, 3> sum{};
            for (const auto& vertex : mesh.vertices)
            {
                sum[0] += vertex.position.x;
                sum[1] += vertex.position.y;
                sum[2] += vertex.position.z;
            }
            std::array<float, 3> center{};
            for (std::size_t i = 0; i < 3; ++i)
            {
                center[i] = static_cast<float>(sum[i] / static_cast<double>(mesh.vertices.size()));
            }
            for (auto& vertex : mesh.vertices)
            {
                vertex.position.x = static_cast<float>(static_cast<float>(vertex.position.x) - center[0]);
                vertex.position.y = static_cast<float>(static_cast<float>(vertex.position.y) - center[1]);
                vertex.position.z = static_cast<float>(static_cast<float>(vertex.position.z) - center[2]);
            }
            const auto budget = metmodel::recovered_qem_part_face_budget(
                mesh.faces.size(), mesh.faces.size(), static_cast<std::size_t>(targetFaces));
            const auto qem = metmodel::decimate_mesh_qem_mode3(mesh, budget, scales, nullptr, &attributes, isCancelled);
            result.diagnostics[QStringLiteral("recovered_qem_collapses")] = static_cast<double>(qem.accepted_collapses);
            for (auto& vertex : mesh.vertices)
            {
                vertex.position.x = static_cast<float>(static_cast<float>(vertex.position.x) + center[0]);
                vertex.position.y = static_cast<float>(static_cast<float>(vertex.position.y) + center[1]);
                vertex.position.z = static_cast<float>(static_cast<float>(vertex.position.z) + center[2]);
            }
        }
        checkpoint(isCancelled, progress, "修复反向三角形并按支持度裁剪…", 88);
        const auto repair = metmodel::fix_back_triangles_recovered(mesh);
        result.diagnostics[QStringLiteral("recovered_repair_passes")] = static_cast<double>(repair.passes);
        metmodel::assign_recovered_vertex_confidence(mesh, attributes);
        const auto trim = metmodel::recover_mesh_trim_mask(mesh.faces, attributes, {1, 0.1F});
        metmodel::compact_mesh_recovered_trim(mesh, attributes, trim.final_keep);
        const auto region = regionValues(scene.region);
        restoreWorld(mesh, root_scale, region);
        const double tolerance = static_cast<double>(static_cast<float>(root_scale)) /
                                 static_cast<double>(std::uint32_t{1} << maximum_level) * 0.1;
        const auto clip = metmodel::clip_mesh_to_region_recovered(mesh, region, tolerance);
        (void)clip;
        metmodel::recompute_normals(mesh);
        const bool calculate_colors = settings.value(QStringLiteral("calculateVertexColors")).toBool(true);
        if (calculate_colors)
        {
#if defined(PLASCAN_RECOVERED_VERTEX_COLOR_VULKAN)
            const auto diagnostics = colorizeRecoveredModel(mesh, scene, settings, device, isCancelled, progress);
            for (auto it = diagnostics.begin(); it != diagnostics.end(); ++it)
                result.diagnostics[it.key()] = it.value();
#else
            throw std::runtime_error("Recovered vertex colors require Vulkan and compiled reference shaders");
#endif
        }
        else
        {
            result.diagnostics[QStringLiteral("vertex_color_algorithm")] = QStringLiteral("disabled");
        }
        checkpoint(isCancelled, progress, "发布双精度网格、颜色与顶点置信度…", 99);
        if (mesh.vertices.empty() || mesh.faces.empty() ||
            mesh.vertices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            mesh.faces.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("Recovered mesh is empty or exceeds PlaScan mesh index limits");
        }
        result.mesh.hasVertexColors = calculate_colors;
        result.mesh.hasVertexConfidence = true;
        result.mesh.vertices.reserve(mesh.vertices.size());
        result.precisePositions.reserve(mesh.vertices.size());
        for (const auto& vertex : mesh.vertices)
        {
            MeshVertex v;
            v.x = static_cast<float>(vertex.position.x);
            v.y = static_cast<float>(vertex.position.y);
            v.z = static_cast<float>(vertex.position.z);
            v.nx = static_cast<float>(vertex.normal.x);
            v.ny = static_cast<float>(vertex.normal.y);
            v.nz = static_cast<float>(vertex.normal.z);
            v.confidence = vertex.confidence;
            v.r = vertex.color[0];
            v.g = vertex.color[1];
            v.b = vertex.color[2];
            if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) || !std::isfinite(v.confidence))
            {
                throw std::runtime_error("Non-finite recovered mesh output");
            }
            result.mesh.vertices.push_back(v);
            result.precisePositions.push_back({vertex.position.x, vertex.position.y, vertex.position.z});
        }
        result.mesh.faces.resize(mesh.faces.size());
        for (std::size_t i = 0; i < mesh.faces.size(); ++i)
        {
            for (std::size_t corner = 0; corner < 3; ++corner)
            {
                result.mesh.faces[i].v[corner] = static_cast<int>(mesh.faces[i].vertices[corner]);
            }
        }
        result.diagnostics[QStringLiteral("recovered_elapsed_seconds")] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        result.diagnostics[QStringLiteral("vertex_confidence_semantics")] =
            QStringLiteral("histogram_support_over_qem_count");
        result.diagnostics[QStringLiteral("ply_schema")] = QStringLiteral("double-xyz,uchar-rgb,float-confidence");
        return result;
#endif
    }
} // namespace xjw::mesh
