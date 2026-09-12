#include "RpcPlaneSweepModelBuilder.h"

#include "GpuDeviceLease.h"
#include "PatchMatchCUDA.h"
#include "RpcCameraIO.h"
#include "RpcCameraModel.h"
#include "metmodel/gpu.hpp"
#include "metmodel/mesh.hpp"

#include <QFileInfo>
#include <QJsonArray>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw::mesh
{
namespace
{

void checkpoint(const std::function<bool()>& cancelled,
                const std::function<void(const QString&, int)>& progress,
                const char* stage,
                int percent)
{
    if (cancelled && cancelled())
    {
        throw std::runtime_error("RPC CUDA height-plane-sweep cancelled");
    }
    if (progress)
    {
        progress(QString::fromUtf8(stage), percent);
    }
}

metmodel::Rpc00bCamera packRpc(const xjw::RpcCameraModel& camera)
{
    const auto& parameters = camera.parameters();
    metmodel::Rpc00bCamera result;
    result.line_off = parameters.lineOffset;
    result.sample_off = parameters.sampleOffset;
    result.lat_off = parameters.latitudeOffset;
    result.lon_off = parameters.longitudeOffset;
    result.height_off = parameters.heightOffset;
    result.line_scale = parameters.lineScale;
    result.sample_scale = parameters.sampleScale;
    result.lat_scale = parameters.latitudeScale;
    result.lon_scale = parameters.longitudeScale;
    result.height_scale = parameters.heightScale;
    result.line_num = parameters.lineNumerator;
    result.line_den = parameters.lineDenominator;
    result.sample_num = parameters.sampleNumerator;
    result.sample_den = parameters.sampleDenominator;
    const auto& correction = camera.imageCorrection();
    // RpcCameraModel's correction is expressed in normalized image coordinates,
    // not the reference alignment sidecar's world-delta affine schema.  Accept
    // only the identity form until that exact sidecar is made part of PlaScan's
    // public RPC input; approximating it would corrupt a geographic sweep.
    if (correction.sampleOffsetPixels != 0.0 || correction.sampleSamplePixels != 0.0 ||
        correction.sampleLinePixels != 0.0 || correction.lineOffsetPixels != 0.0 ||
        correction.lineSamplePixels != 0.0 || correction.lineLinePixels != 0.0)
    {
        throw std::runtime_error("RPC normalized image correction is not representable by the RPC00B plane-sweep contract");
    }
    return result;
}

std::vector<float> readGray(const QString& path, std::size_t* width, std::size_t* height)
{
    const cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE);
    if (image.empty() || image.type() != CV_8UC1)
    {
        throw std::runtime_error("cannot decode RPC input raster: " + path.toStdString());
    }
    *width = static_cast<std::size_t>(image.cols);
    *height = static_cast<std::size_t>(image.rows);
    std::vector<float> result(*width * *height);
    for (int y = 0; y < image.rows; ++y)
    {
        const auto* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            result[static_cast<std::size_t>(y) * *width + static_cast<std::size_t>(x)] =
                static_cast<float>(row[x]) / 255.0F;
        }
    }
    return result;
}

void validateImageSize(const xjw::RpcCameraModel& camera,
                       std::size_t width,
                       std::size_t height,
                       const QString& path)
{
    const auto declared = camera.imageSize();
    if (!declared || declared->samples <= 0 || declared->lines <= 0 ||
        static_cast<std::size_t>(declared->samples) != width ||
        static_cast<std::size_t>(declared->lines) != height)
    {
        throw std::runtime_error("RPC camera raster dimensions do not match decoded input: " + path.toStdString());
    }
}

struct RpcGridKey
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const RpcGridKey&) const = default;
};

struct RpcGridHash
{
    std::size_t operator()(const RpcGridKey& key) const
    {
        std::size_t value = std::hash<std::int64_t>{}(key.x);
        value ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        value ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        return value;
    }
};

RpcGridKey quantizeRpcVertex(const metalign::Vec3& point, double cell_size)
{
    return {static_cast<std::int64_t>(std::llround(point.x / cell_size)),
            static_cast<std::int64_t>(std::llround(point.y / cell_size)),
            static_cast<std::int64_t>(std::llround(point.z / cell_size))};
}

void compactReferenceRpcMesh(metmodel::Mesh* mesh)
{
    std::vector<std::uint8_t> used(mesh->vertices.size(), 0U);
    std::vector<metmodel::Face> faces;
    faces.reserve(mesh->faces.size());
    std::set<std::array<std::size_t, 3>> unique;
    for (metmodel::Face face : mesh->faces)
    {
        if (face.vertices[0] == face.vertices[1] || face.vertices[1] == face.vertices[2] ||
            face.vertices[0] == face.vertices[2]) continue;
        std::array<std::size_t, 3> key = face.vertices;
        std::sort(key.begin(), key.end());
        if (!unique.insert(key).second) continue;
        const auto& a = mesh->vertices[face.vertices[0]].position;
        const auto& b = mesh->vertices[face.vertices[1]].position;
        const auto& c = mesh->vertices[face.vertices[2]].position;
        if (metalign::norm(metalign::cross(b - a, c - a)) < 1.0e-14) continue;
        faces.push_back(face);
        for (const std::size_t index : face.vertices) used[index] = 1U;
    }
    std::vector<std::size_t> remap(mesh->vertices.size(), 0U);
    std::vector<metmodel::Vertex> vertices;
    vertices.reserve(mesh->vertices.size());
    for (std::size_t index = 0; index < mesh->vertices.size(); ++index)
    {
        if (!used[index]) continue;
        remap[index] = vertices.size();
        vertices.push_back(mesh->vertices[index]);
    }
    for (auto& face : faces)
    {
        for (auto& index : face.vertices) index = remap[index];
    }
    mesh->vertices = std::move(vertices);
    mesh->faces = std::move(faces);
}

metmodel::Mesh clusterReferenceRpcMesh(const metmodel::Mesh& source, double cell_size)
{
    struct Accumulator
    {
        metalign::Vec3 position{};
        metalign::Vec3 normal{};
        metalign::Vec2 uv{};
        std::array<std::uint64_t, 3> color{};
        double confidence = 0.0;
        std::size_t count = 0;
    };
    metmodel::Mesh result;
    std::unordered_map<RpcGridKey, std::size_t, RpcGridHash> clusters;
    std::vector<Accumulator> accumulators;
    std::vector<std::size_t> remap(source.vertices.size());
    for (std::size_t index = 0; index < source.vertices.size(); ++index)
    {
        const auto& vertex = source.vertices[index];
        const RpcGridKey key = quantizeRpcVertex(vertex.position, cell_size);
        const auto [iterator, inserted] = clusters.emplace(key, accumulators.size());
        if (inserted) accumulators.emplace_back();
        auto& accumulator = accumulators[iterator->second];
        accumulator.position = accumulator.position + vertex.position;
        accumulator.normal = accumulator.normal + vertex.normal;
        accumulator.uv = accumulator.uv + vertex.uv;
        for (std::size_t channel = 0; channel < 3U; ++channel) accumulator.color[channel] += vertex.color[channel];
        accumulator.confidence += vertex.confidence;
        ++accumulator.count;
        remap[index] = iterator->second;
    }
    result.vertices.reserve(accumulators.size());
    for (const auto& accumulator : accumulators)
    {
        metmodel::Vertex vertex;
        vertex.position = accumulator.position / static_cast<double>(accumulator.count);
        vertex.normal = metalign::normalized(accumulator.normal);
        vertex.uv = accumulator.uv * (1.0 / static_cast<double>(accumulator.count));
        for (std::size_t channel = 0; channel < 3U; ++channel)
            vertex.color[channel] = static_cast<std::uint8_t>(accumulator.color[channel] / accumulator.count);
        vertex.confidence = static_cast<float>(accumulator.confidence / static_cast<double>(accumulator.count));
        result.vertices.push_back(vertex);
    }
    result.faces.reserve(source.faces.size());
    for (const auto& face : source.faces)
    {
        result.faces.push_back({{remap[face.vertices[0]], remap[face.vertices[1]], remap[face.vertices[2]]}});
    }
    compactReferenceRpcMesh(&result);
    return result;
}

void decimateReferenceRpcMesh(metmodel::Mesh* mesh, std::size_t target_faces)
{
    if (target_faces == 0U) throw std::runtime_error("RPC decimation target must be positive");
    if (mesh->faces.size() <= target_faces) return;
    metalign::Vec3 minimum = mesh->vertices.front().position;
    metalign::Vec3 maximum = minimum;
    for (const auto& vertex : mesh->vertices)
    {
        minimum.x = std::min(minimum.x, vertex.position.x); minimum.y = std::min(minimum.y, vertex.position.y);
        minimum.z = std::min(minimum.z, vertex.position.z); maximum.x = std::max(maximum.x, vertex.position.x);
        maximum.y = std::max(maximum.y, vertex.position.y); maximum.z = std::max(maximum.z, vertex.position.z);
    }
    const double diagonal = std::max(1.0e-9, metalign::norm(maximum - minimum));
    double low = diagonal / 100000.0;
    double high = diagonal;
    metmodel::Mesh best = *mesh;
    for (std::size_t iteration = 0; iteration < 28U; ++iteration)
    {
        const double cell = std::sqrt(low * high);
        metmodel::Mesh candidate = clusterReferenceRpcMesh(*mesh, cell);
        if (candidate.faces.size() > target_faces)
        {
            low = cell;
        }
        else
        {
            high = cell;
            if (candidate.faces.size() > best.faces.size() || best.faces.size() > target_faces) best = std::move(candidate);
        }
    }
    if (best.faces.size() > target_faces)
    {
        std::vector<metmodel::Face> sampled;
        sampled.reserve(target_faces);
        for (std::size_t index = 0; index < target_faces; ++index)
            sampled.push_back(best.faces[index * best.faces.size() / target_faces]);
        best.faces = std::move(sampled);
        compactReferenceRpcMesh(&best);
    }
    *mesh = std::move(best);
    metmodel::recompute_normals(*mesh);
}

metmodel::Mesh buildReferenceRpcMesh(const metmodel::RpcCudaHeightPlaneSweepResult& grid,
                                     const std::vector<float>& reference_gray,
                                     std::size_t reference_width,
                                     std::size_t reference_height,
                                     int downscale,
                                     float minimum_confidence)
{
    metmodel::Mesh mesh;
    std::vector<std::size_t> remap(grid.world.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t index = 0; index < grid.world.size(); ++index)
    {
        if (grid.valid[index] == 0U || grid.confidence[index] < minimum_confidence) continue;
        const std::size_t x = index % grid.width;
        const std::size_t y = index / grid.width;
        const std::size_t source_x = std::min(reference_width - 1U, x * static_cast<std::size_t>(downscale));
        const std::size_t source_y = std::min(reference_height - 1U, y * static_cast<std::size_t>(downscale));
        const std::uint8_t gray = static_cast<std::uint8_t>(std::clamp(
            reference_gray[source_y * reference_width + source_x], 0.0F, 1.0F) * 255.0F);
        metmodel::Vertex vertex;
        vertex.position = grid.world[index];
        vertex.color = {gray, gray, gray};
        vertex.confidence = grid.confidence[index];
        remap[index] = mesh.vertices.size();
        mesh.vertices.push_back(vertex);
    }
    for (std::size_t y = 0; y + 1U < grid.height; ++y) for (std::size_t x = 0; x + 1U < grid.width; ++x)
    {
        const std::size_t a = remap[y * grid.width + x];
        const std::size_t b = remap[y * grid.width + x + 1U];
        const std::size_t c = remap[(y + 1U) * grid.width + x];
        const std::size_t d = remap[(y + 1U) * grid.width + x + 1U];
        if (a != std::numeric_limits<std::size_t>::max() && b != std::numeric_limits<std::size_t>::max() && c != std::numeric_limits<std::size_t>::max())
            mesh.faces.push_back({{a, b, c}});
        if (b != std::numeric_limits<std::size_t>::max() && d != std::numeric_limits<std::size_t>::max() && c != std::numeric_limits<std::size_t>::max())
            mesh.faces.push_back({{b, d, c}});
    }
    if (mesh.faces.empty())
    {
        throw std::runtime_error("RPC CUDA height-plane-sweep produced no connected valid dense cells");
    }
    metmodel::recompute_normals(mesh);
    return mesh;
}

void applyReferenceRpcPostprocess(metmodel::Mesh* mesh, std::size_t target_faces)
{
    if (target_faces == 0U)
    {
        throw std::runtime_error("RPC face-count target must be positive");
    }
    // This is the ordinary RPC branch from the reference main.cpp: cluster
    // decimation first, then SmoothModel defaults (strength=0, fixed borders,
    // no edge preservation).  It is intentionally not PlaScan's legacy mesh
    // simplifier nor a recovered-OOC/QEM operation.
    if (mesh->faces.size() > target_faces)
    {
        decimateReferenceRpcMesh(mesh, target_faces);
    }
    metmodel::smooth_mesh(*mesh, 0.0, true, false);
}

TriMesh toTriMesh(const metmodel::Mesh& source)
{
    TriMesh result;
    result.vertices.reserve(source.vertices.size());
    result.faces.reserve(source.faces.size());
    for (const auto& vertex : source.vertices)
    {
        MeshVertex converted;
        converted.x = static_cast<float>(vertex.position.x); converted.y = static_cast<float>(vertex.position.y);
        converted.z = static_cast<float>(vertex.position.z); converted.nx = static_cast<float>(vertex.normal.x);
        converted.ny = static_cast<float>(vertex.normal.y); converted.nz = static_cast<float>(vertex.normal.z);
        converted.r = vertex.color[0]; converted.g = vertex.color[1]; converted.b = vertex.color[2];
        converted.confidence = vertex.confidence;
        result.vertices.push_back(converted);
    }
    for (const auto& face : source.faces)
    {
        result.faces.push_back({{static_cast<int>(face.vertices[0]), static_cast<int>(face.vertices[1]),
                                 static_cast<int>(face.vertices[2])}});
    }
    result.hasVertexColors = true;
    result.hasVertexConfidence = true;
    return result;
}

} // namespace

RpcPlaneSweepModelResult buildRpcPlaneSweepModel(const QJsonObject& settings,
                                                 int targetFaces,
                                                 const std::function<bool()>& isCancelled,
                                                 const std::function<void(const QString&, int)>& progress)
{
#if !defined(MVS_ENABLE_CUDA)
    throw std::runtime_error("RPC height-plane-sweep requires a CUDA build");
#else
    checkpoint(isCancelled, progress, "验证 RPC00B 高程平面扫描输入…", 2);
    const QJsonArray paths = settings.value(QStringLiteral("rpcImagePaths")).toArray();
    if (paths.size() != 2 || !paths[0].isString() || !paths[1].isString())
    {
        throw std::runtime_error("rpcImagePaths must contain exactly two RPC raster paths");
    }
    const QString reference_path = paths[0].toString().trimmed();
    const QString neighbor_path = paths[1].toString().trimmed();
    if (reference_path.isEmpty() || neighbor_path.isEmpty() || reference_path == neighbor_path ||
        !QFileInfo::exists(reference_path) || !QFileInfo::exists(neighbor_path))
    {
        throw std::runtime_error("RPC height-plane-sweep requires two distinct existing rasters");
    }
    if (!settings.value(QStringLiteral("rpcHeightMinMeters")).isDouble() ||
        !settings.value(QStringLiteral("rpcHeightMaxMeters")).isDouble())
    {
        throw std::runtime_error("RPC height-plane-sweep requires finite rpcHeightMinMeters and rpcHeightMaxMeters");
    }
    const double height_min = settings.value(QStringLiteral("rpcHeightMinMeters")).toDouble();
    const double height_max = settings.value(QStringLiteral("rpcHeightMaxMeters")).toDouble();
    if (!std::isfinite(height_min) || !std::isfinite(height_max) || !(height_min < height_max))
    {
        throw std::runtime_error("RPC height gate must be finite and satisfy rpcHeightMinMeters < rpcHeightMaxMeters");
    }
    const int device = settings.contains(QStringLiteral("compute_device_index"))
        ? settings.value(QStringLiteral("compute_device_index")).toInt(-1)
        : settings.value(QStringLiteral("cudaDevice")).toInt(-1);
    if (device < 0)
    {
        throw std::runtime_error("RPC height-plane-sweep requires a nonnegative CUDA device (compute_device_index or cudaDevice)");
    }
    if (settings.value(QStringLiteral("depthQualityProfile")).toString() != QStringLiteral("medium"))
    {
        throw std::runtime_error("RPC height-plane-sweep requires depthQualityProfile=medium (d4)");
    }
    const int downscale = 4;
    const int hypotheses = settings.value(QStringLiteral("rpcHeightHypotheses")).toInt(64);
    if (hypotheses < 8 || hypotheses > 512)
    {
        throw std::runtime_error("rpcHeightHypotheses must be in [8, 512]");
    }
    const float minimum_confidence = static_cast<float>(settings.value(QStringLiteral("rpcMinimumConfidence")).toDouble(0.0));
    if (!std::isfinite(minimum_confidence) || minimum_confidence < 0.0F || minimum_confidence > 1.0F)
    {
        throw std::runtime_error("rpcMinimumConfidence must be in [0, 1]");
    }
    xjw::RpcCameraModel reference_camera;
    xjw::RpcCameraModel neighbor_camera;
    std::string load_error;
    if (!xjw::loadRpcCameraFromRaster(reference_path.toStdString(), &reference_camera, &load_error) || !reference_camera.isValid())
    {
        throw std::runtime_error("cannot load reference RPC00B camera: " + load_error);
    }
    if (!xjw::loadRpcCameraFromRaster(neighbor_path.toStdString(), &neighbor_camera, &load_error) || !neighbor_camera.isValid())
    {
        throw std::runtime_error("cannot load neighbor RPC00B camera: " + load_error);
    }
    const metmodel::Rpc00bCamera reference_rpc = packRpc(reference_camera);
    const metmodel::Rpc00bCamera neighbor_rpc = packRpc(neighbor_camera);
    for (const auto& pair : {std::pair<const char*, const metmodel::Rpc00bCamera*>{"reference", &reference_rpc},
                             std::pair<const char*, const metmodel::Rpc00bCamera*>{"neighbor", &neighbor_rpc}})
    {
        std::string validation_error;
        if (!metmodel::validate_rpc00b(*pair.second, validation_error))
        {
            throw std::runtime_error(std::string("invalid ") + pair.first + " RPC00B camera: " + validation_error);
        }
        const double allowed_min = pair.second->height_off - pair.second->height_scale;
        const double allowed_max = pair.second->height_off + pair.second->height_scale;
        if (!std::isfinite(allowed_min) || !std::isfinite(allowed_max) || height_min < allowed_min || height_max > allowed_max)
        {
            throw std::runtime_error(std::string("RPC height gate lies outside ") + pair.first + " RPC height normalization range");
        }
    }
    std::size_t reference_width = 0, reference_height = 0, neighbor_width = 0, neighbor_height = 0;
    const auto reference_gray = readGray(reference_path, &reference_width, &reference_height);
    const auto neighbor_gray = readGray(neighbor_path, &neighbor_width, &neighbor_height);
    validateImageSize(reference_camera, reference_width, reference_height, reference_path);
    validateImageSize(neighbor_camera, neighbor_width, neighbor_height, neighbor_path);
    xjw::mvs::GpuDeviceLeaseSet lease;
    QString lease_error;
    const auto identity = xjw::mvs::PatchMatchDepthEstimator::cudaDeviceIdentity(device);
    if (identity.empty() || !lease.acquire({{identity, xjw::mvs::PatchMatchDepthEstimator::cudaDeviceName(device)}}, &lease_error))
    {
        throw std::runtime_error("cannot acquire RPC CUDA device: " + lease_error.toStdString());
    }
    checkpoint(isCancelled, progress, "执行 RPC00B CUDA 高程平面扫描…", 18);
    metmodel::RpcCudaHeightPlaneSweepInput input;
    input.reference = reference_rpc; input.neighbor = neighbor_rpc;
    input.reference_width = reference_width; input.reference_height = reference_height;
    input.neighbor_width = neighbor_width; input.neighbor_height = neighbor_height;
    input.reference_gray = &reference_gray; input.neighbor_gray = &neighbor_gray;
    input.downscale = downscale; input.hypotheses = static_cast<std::size_t>(hypotheses);
    input.height_min = height_min; input.height_max = height_max; input.device_index = static_cast<std::size_t>(device);
    metmodel::RpcCudaHeightPlaneSweepResult grid;
    std::string sweep_error;
    if (!metmodel::run_rpc_height_plane_sweep_cuda(input, grid, sweep_error))
    {
        throw std::runtime_error("RPC CUDA height-plane-sweep failed: " + sweep_error);
    }
    checkpoint(isCancelled, progress, "构建 RPC 高程平面扫描三角网格…", 84);
    RpcPlaneSweepModelResult result;
    metmodel::Mesh reference_mesh = buildReferenceRpcMesh(
        grid, reference_gray, reference_width, reference_height, downscale, minimum_confidence);
    const std::size_t raw_face_count = reference_mesh.faces.size();
    applyReferenceRpcPostprocess(&reference_mesh, static_cast<std::size_t>(targetFaces));
    result.mesh = toTriMesh(reference_mesh);
    result.diagnostics[QStringLiteral("actual_mesh_algorithm")] = QStringLiteral("rpc_height_plane_sweep");
    result.diagnostics[QStringLiteral("rpc_cuda_dispatch")] = QString::fromStdString(grid.dispatch);
    result.diagnostics[QStringLiteral("rpc_height_min_meters")] = height_min;
    result.diagnostics[QStringLiteral("rpc_height_max_meters")] = height_max;
    result.diagnostics[QStringLiteral("rpc_cuda_device")] = device;
    result.diagnostics[QStringLiteral("rpc_downscale")] = downscale;
    result.diagnostics[QStringLiteral("rpc_height_hypotheses")] = hypotheses;
    result.diagnostics[QStringLiteral("rpc_minimum_confidence")] = minimum_confidence;
    result.diagnostics[QStringLiteral("rpc_raw_face_count")] = static_cast<qint64>(raw_face_count);
    result.diagnostics[QStringLiteral("rpc_effective_face_target")] = targetFaces;
    result.diagnostics[QStringLiteral("rpc_final_face_count")] = result.mesh.faceCount();
    result.diagnostics[QStringLiteral("rpc_smooth_strength")] = 0.0;
    result.diagnostics[QStringLiteral("rpc_smooth_fix_borders")] = true;
    result.diagnostics[QStringLiteral("rpc_smooth_preserve_edges")] = false;
    result.diagnostics[QStringLiteral("rpc_input_coordinate_space")] = QStringLiteral("longitude_latitude_ellipsoidal_height");
    result.diagnostics[QStringLiteral("fallback")] = QStringLiteral("none");
    checkpoint(isCancelled, progress, "完成 RPC00B CUDA 高程平面扫描模型", 100);
    return result;
#endif
}

} // namespace xjw::mesh
