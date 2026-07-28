#include "OpenMeshSimplifier.h"

#include "MeshFaceOrientation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#ifndef PLASCAN_HAS_OPENMESH
#define PLASCAN_HAS_OPENMESH 0
#endif

#if PLASCAN_HAS_OPENMESH
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalDeviationT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalFlippingT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/Observer.hh>
#include <OpenMesh/Tools/Smoother/JacobiLaplaceSmootherT.hh>
#endif

namespace xjw::mesh
{

bool openMeshSimplifierAvailable()
{
    return PLASCAN_HAS_OPENMESH != 0;
}

#if PLASCAN_HAS_OPENMESH
namespace
{

using OpenTriMesh = OpenMesh::TriMesh_ArrayKernelT<>;
using Decimater = OpenMesh::Decimater::DecimaterT<OpenTriMesh>;

class SimplificationObserver final : public OpenMesh::Decimater::Observer
{
public:
    SimplificationObserver(
        int notificationInterval,
        int inputFaceCount,
        const OpenMeshSimplifyOptions &options)
        : Observer(static_cast<std::size_t>(std::max(1, notificationInterval))),
          _inputFaceCount(inputFaceCount),
          _options(options)
    {
    }

    void notify(std::size_t step) override
    {
        _collapsedVertexCount = static_cast<int>(step);
        if (_options.progress)
        {
            const int estimated_face_count = std::max(
                0, _inputFaceCount - _collapsedVertexCount * 2);
            _options.progress(_collapsedVertexCount, estimated_face_count);
        }
        _cancelled = _options.isCancelled && _options.isCancelled();
    }

    bool abort() const override
    {
        return _cancelled ||
            (_options.isCancelled && _options.isCancelled());
    }

    bool cancelled() const
    {
        return _cancelled ||
            (_options.isCancelled && _options.isCancelled());
    }

private:
    int _inputFaceCount = 0;
    const OpenMeshSimplifyOptions &_options;
    int _collapsedVertexCount = 0;
    bool _cancelled = false;
};

bool copyToOpenMesh(
    const TriMesh &input,
    OpenTriMesh *output,
    OpenMeshSimplifyStatistics *statistics)
{
    output->request_vertex_colors();
    output->request_face_normals();
    output->request_vertex_normals();

    std::vector<OpenTriMesh::VertexHandle> handles;
    handles.reserve(input.vertices.size());
    for (const MeshVertex &vertex : input.vertices)
    {
        const OpenTriMesh::VertexHandle handle = output->add_vertex(
            OpenTriMesh::Point(vertex.x, vertex.y, vertex.z));
        output->set_color(
            handle,
            OpenTriMesh::Color(vertex.r, vertex.g, vertex.b));
        handles.push_back(handle);
    }

    for (const Triangle &face : input.faces)
    {
        const bool valid_indices =
            face.v[0] >= 0 && face.v[1] >= 0 && face.v[2] >= 0 &&
            face.v[0] < static_cast<int>(handles.size()) &&
            face.v[1] < static_cast<int>(handles.size()) &&
            face.v[2] < static_cast<int>(handles.size());
        if (!valid_indices)
        {
            ++statistics->rejectedInputFaceCount;
            continue;
        }

        const OpenTriMesh::FaceHandle face_handle = output->add_face(
            handles[static_cast<std::size_t>(face.v[0])],
            handles[static_cast<std::size_t>(face.v[1])],
            handles[static_cast<std::size_t>(face.v[2])]);
        if (!face_handle.is_valid())
        {
            ++statistics->rejectedInputFaceCount;
        }
    }

    if (statistics->rejectedInputFaceCount > 0)
    {
        statistics->error =
            "OpenMesh rejected one or more invalid, duplicated, or non-manifold faces";
        return false;
    }

    output->update_face_normals();
    output->update_vertex_normals();
    return true;
}

TriMesh copyFromOpenMesh(OpenTriMesh *input, bool hasVertexColors)
{
    input->garbage_collection();
    input->update_face_normals();
    input->update_vertex_normals();

    TriMesh output;
    output.hasVertexColors = hasVertexColors;
    output.vertices.reserve(input->n_vertices());
    output.faces.reserve(input->n_faces());

    std::vector<int> indices(input->n_vertices(), -1);
    for (const OpenTriMesh::VertexHandle handle : input->vertices())
    {
        const OpenTriMesh::Point point = input->point(handle);
        const OpenTriMesh::Normal normal = input->normal(handle);
        const OpenTriMesh::Color color = input->color(handle);

        MeshVertex vertex;
        vertex.x = point[0];
        vertex.y = point[1];
        vertex.z = point[2];
        vertex.nx = normal[0];
        vertex.ny = normal[1];
        vertex.nz = normal[2];
        vertex.r = color[0];
        vertex.g = color[1];
        vertex.b = color[2];
        indices[static_cast<std::size_t>(handle.idx())] =
            static_cast<int>(output.vertices.size());
        output.vertices.push_back(vertex);
    }

    for (const OpenTriMesh::FaceHandle face_handle : input->faces())
    {
        Triangle face;
        int corner = 0;
        for (const OpenTriMesh::VertexHandle vertex_handle :
             input->fv_range(face_handle))
        {
            if (corner < 3)
            {
                face.v[corner] =
                    indices[static_cast<std::size_t>(vertex_handle.idx())];
            }
            ++corner;
        }
        if (corner == 3 &&
            face.v[0] >= 0 && face.v[1] >= 0 && face.v[2] >= 0)
        {
            output.faces.push_back(face);
        }
    }
    return output;
}

void markSmoothingFeatures(OpenTriMesh *mesh, float featureAngleDegrees)
{
    mesh->request_edge_status();
    mesh->request_vertex_status();
    mesh->update_face_normals();
    const float cosine_threshold = std::cos(
        std::clamp(featureAngleDegrees, 0.0f, 180.0f) *
        0.01745329251994329577f);
    for (const OpenTriMesh::EdgeHandle edge_handle : mesh->edges())
    {
        const OpenTriMesh::HalfedgeHandle first_halfedge =
            mesh->halfedge_handle(edge_handle, 0);
        const OpenTriMesh::HalfedgeHandle second_halfedge =
            mesh->halfedge_handle(edge_handle, 1);
        const OpenTriMesh::FaceHandle first_face =
            mesh->face_handle(first_halfedge);
        const OpenTriMesh::FaceHandle second_face =
            mesh->face_handle(second_halfedge);
        bool feature = !first_face.is_valid() || !second_face.is_valid();
        if (!feature)
        {
            feature =
                (mesh->normal(first_face) | mesh->normal(second_face)) <
                cosine_threshold;
        }
        if (feature)
        {
            mesh->status(edge_handle).set_feature(true);
            mesh->status(mesh->from_vertex_handle(first_halfedge))
                .set_feature(true);
            mesh->status(mesh->to_vertex_handle(first_halfedge))
                .set_feature(true);
        }
    }
}

bool smoothOpenMesh(
    OpenTriMesh *mesh,
    const OpenMeshSimplifyOptions &options)
{
    if (options.smoothingIterations <= 0 ||
        !(options.smoothingMaximumDisplacement > 0.0f))
    {
        return false;
    }

    mesh->garbage_collection();
    markSmoothingFeatures(mesh, options.smoothingFeatureAngleDegrees);
    using Smoother =
        OpenMesh::Smoother::JacobiLaplaceSmootherT<OpenTriMesh>;
    Smoother smoother(*mesh);
    smoother.initialize(
        Smoother::Tangential_and_Normal,
        Smoother::C1);
    smoother.set_absolute_local_error(
        options.smoothingMaximumDisplacement);
    smoother.skip_features(true);
    smoother.smooth(static_cast<unsigned int>(
        options.smoothingIterations));
    mesh->update_face_normals();
    mesh->update_vertex_normals();
    return true;
}

} // namespace
#endif

OpenMeshSimplifyStatistics simplifyMeshWithOpenMesh(
    TriMesh *mesh,
    const OpenMeshSimplifyOptions &options)
{
    OpenMeshSimplifyStatistics statistics;
    statistics.available = openMeshSimplifierAvailable();
    if (mesh == nullptr)
    {
        statistics.error = "mesh is null";
        return statistics;
    }

    statistics.inputVertexCount = mesh->vertexCount();
    statistics.inputFaceCount = mesh->faceCount();
    statistics.outputVertexCount = statistics.inputVertexCount;
    statistics.outputFaceCount = statistics.inputFaceCount;

#if !PLASCAN_HAS_OPENMESH
    statistics.error = "OpenMesh support is unavailable in this build";
    return statistics;
#else
    if (mesh->empty())
    {
        statistics.error = "mesh is empty";
        return statistics;
    }
    if (options.targetFaceCount <= 0 ||
        options.targetFaceCount >= mesh->faceCount())
    {
        statistics.error = "target face count must be positive and smaller than input";
        return statistics;
    }
    if (options.isCancelled && options.isCancelled())
    {
        statistics.cancelled = true;
        statistics.error = "simplification cancelled before initialization";
        return statistics;
    }

    TriMesh oriented_input = *mesh;
    const MeshFaceOrientationStatistics orientation =
        repairMeshFaceOrientation(&oriented_input);
    statistics.inconsistentSharedEdgeCountBefore =
        orientation.inconsistentSharedEdgeCountBefore;
    statistics.reorientedInputFaceCount = orientation.flippedFaceCount;
    statistics.removedContradictoryFaceCount =
        orientation.removedContradictoryFaceCount;
    statistics.orientationConflictCount =
        orientation.orientationConflictCount;
    if (!orientation.succeeded)
    {
        statistics.error =
            "mesh face orientation is non-manifold or contradictory";
        return statistics;
    }

    OpenTriMesh open_mesh;
    if (!copyToOpenMesh(oriented_input, &open_mesh, &statistics))
    {
        return statistics;
    }

    Decimater decimater(open_mesh);
    OpenMesh::Decimater::ModQuadricT<OpenTriMesh>::Handle quadric_handle;
    OpenMesh::Decimater::ModNormalDeviationT<OpenTriMesh>::Handle
        normal_deviation_handle;
    OpenMesh::Decimater::ModNormalFlippingT<OpenTriMesh>::Handle
        normal_flipping_handle;

    decimater.add(quadric_handle);
    decimater.module(quadric_handle).unset_max_err();
    decimater.add(normal_deviation_handle);
    decimater.module(normal_deviation_handle).set_normal_deviation(
        std::clamp(options.maximumNormalDeviationDegrees, 0.0f, 180.0f));
    decimater.add(normal_flipping_handle);
    decimater.module(normal_flipping_handle).set_max_normal_deviation(
        std::clamp(options.maximumNormalFlippingDegrees, 0.0f, 180.0f));

    statistics.initialized = decimater.initialize();
    if (!statistics.initialized)
    {
        statistics.error = "OpenMesh decimater initialization failed";
        return statistics;
    }

    SimplificationObserver observer(
        options.notificationInterval,
        statistics.inputFaceCount,
        options);
    decimater.set_observer(&observer);
    statistics.collapsedVertexCount = static_cast<int>(
        decimater.decimate_to_faces(
            0, static_cast<std::size_t>(options.targetFaceCount)));
    statistics.cancelled = observer.cancelled();
    if (!statistics.cancelled)
    {
        statistics.smoothingApplied =
            smoothOpenMesh(&open_mesh, options);
    }

    TriMesh candidate = copyFromOpenMesh(&open_mesh, mesh->hasVertexColors);
    statistics.outputVertexCount = candidate.vertexCount();
    statistics.outputFaceCount = candidate.faceCount();
    statistics.reachedTarget =
        statistics.outputFaceCount <= options.targetFaceCount;
    statistics.succeeded =
        !statistics.cancelled && !candidate.empty() &&
        statistics.outputFaceCount < statistics.inputFaceCount;
    if (!statistics.succeeded)
    {
        statistics.error = statistics.cancelled
            ? "OpenMesh simplification cancelled"
            : "OpenMesh could not perform a legal edge collapse";
        return statistics;
    }

    *mesh = std::move(candidate);
    return statistics;
#endif
}

} // namespace xjw::mesh
