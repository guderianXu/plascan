#include "ScreenedPoissonSurfaceBuilder.h"

#include <Reconstructors.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

namespace xjw::mesh
{
namespace
{

using Real = float;
constexpr unsigned int Dimension = 3;
using Point = PoissonRecon::Point<Real, Dimension>;
using Face = PoissonRecon::Reconstructor::Face<Dimension - 1>;

struct OrientedSample
{
    Point point;
    Point normal;
};

class OrientedSampleStream final
    : public PoissonRecon::Reconstructor::InputOrientedSampleStream<Real, Dimension>
{
public:
    explicit OrientedSampleStream(const std::vector<OrientedSample> &samples)
        : _samples(samples)
    {
    }

    void reset() override
    {
        _next = 0;
    }

    bool read(Point &point, Point &normal) override
    {
        if (_next >= _samples.size())
        {
            return false;
        }

        point = _samples[_next].point;
        normal = _samples[_next].normal;
        ++_next;
        return true;
    }

private:
    const std::vector<OrientedSample> &_samples;
    std::size_t _next = 0;
};

class VertexStream final
    : public PoissonRecon::Reconstructor::OutputLevelSetVertexStream<Real, Dimension>
{
public:
    explicit VertexStream(std::vector<MeshVertex> *vertices)
        : _vertices(vertices)
    {
    }

    std::size_t size() const override
    {
        return _vertices->size();
    }

    std::size_t write(const Point &point,
                      const Point &gradient,
                      const Real &) override
    {
        MeshVertex vertex;
        vertex.x = point[0];
        vertex.y = point[1];
        vertex.z = point[2];

        const double squared_length =
            static_cast<double>(gradient[0]) * gradient[0] +
            static_cast<double>(gradient[1]) * gradient[1] +
            static_cast<double>(gradient[2]) * gradient[2];
        if (squared_length > 1.0e-24)
        {
            const float inverse_length =
                static_cast<float>(1.0 / std::sqrt(squared_length));
            vertex.nx = gradient[0] * inverse_length;
            vertex.ny = gradient[1] * inverse_length;
            vertex.nz = gradient[2] * inverse_length;
        }

        _vertices->push_back(vertex);
        return _vertices->size() - 1;
    }

private:
    std::vector<MeshVertex> *_vertices = nullptr;
};

class FaceStream final : public PoissonRecon::Reconstructor::OutputFaceStream<Dimension - 1>
{
public:
    explicit FaceStream(std::vector<Face> *faces)
        : _faces(faces)
    {
    }

    std::size_t size() const override
    {
        return _faces->size();
    }

    std::size_t write(const Face &face) override
    {
        _faces->push_back(face);
        return _faces->size() - 1;
    }

private:
    std::vector<Face> *_faces = nullptr;
};

std::mutex &poissonMutex()
{
    static std::mutex mutex;
    return mutex;
}

bool isFinite(const std::array<float, 3> &value)
{
    return std::isfinite(value[0]) &&
        std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

bool validateOptions(const ScreenedPoissonOptions &options, std::string *error)
{
    if (options.depth < 5 || options.depth > 14)
    {
        *error = "Screened Poisson depth must be in [5, 14]";
        return false;
    }
    if (!std::isfinite(options.pointWeight) || options.pointWeight <= 0.0f)
    {
        *error = "Screened Poisson pointWeight must be finite and greater than zero";
        return false;
    }
    if (!std::isfinite(options.samplesPerNode) || options.samplesPerNode <= 0.0f)
    {
        *error = "Screened Poisson samplesPerNode must be finite and greater than zero";
        return false;
    }
    if (!std::isfinite(options.scale) || options.scale <= 1.0f)
    {
        *error = "Screened Poisson scale must be finite and greater than one";
        return false;
    }
    if (options.solverIterations < 1)
    {
        *error = "Screened Poisson solverIterations must be positive";
        return false;
    }
    if (!std::isfinite(options.cgSolverAccuracy) || options.cgSolverAccuracy <= 0.0f)
    {
        *error = "Screened Poisson cgSolverAccuracy must be finite and greater than zero";
        return false;
    }
    return true;
}

std::vector<OrientedSample> prepareSamples(
    const std::vector<std::array<float, 3>> &points,
    const std::vector<std::array<float, 3>> &normals,
    ScreenedPoissonStatistics *statistics)
{
    std::vector<OrientedSample> samples;
    samples.reserve(points.size());

    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const std::array<float, 3> &point = points[index];
        const std::array<float, 3> &normal = normals[index];
        if (!isFinite(point) || !isFinite(normal))
        {
            ++statistics->rejectedSampleCount;
            continue;
        }

        const double squared_length =
            static_cast<double>(normal[0]) * normal[0] +
            static_cast<double>(normal[1]) * normal[1] +
            static_cast<double>(normal[2]) * normal[2];
        if (squared_length <= 1.0e-24)
        {
            ++statistics->rejectedSampleCount;
            continue;
        }

        const float inverse_length =
            static_cast<float>(1.0 / std::sqrt(squared_length));
        OrientedSample sample;
        for (int coordinate = 0; coordinate < 3; ++coordinate)
        {
            sample.point[coordinate] = point[static_cast<std::size_t>(coordinate)];
            sample.normal[coordinate] =
                normal[static_cast<std::size_t>(coordinate)] * inverse_length;
        }
        samples.push_back(sample);
    }

    statistics->acceptedSampleCount = samples.size();
    return samples;
}

void appendTriangle(PoissonRecon::node_index_type first,
                    PoissonRecon::node_index_type second,
                    PoissonRecon::node_index_type third,
                    std::size_t vertexCount,
                    TriMesh *mesh,
                    bool *valid)
{
    const auto maximum_index = static_cast<PoissonRecon::node_index_type>(
        std::min<std::size_t>(vertexCount, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    if (first < 0 || second < 0 || third < 0 ||
        first >= maximum_index || second >= maximum_index || third >= maximum_index ||
        first == second || second == third || first == third)
    {
        *valid = false;
        return;
    }

    Triangle triangle;
    triangle.v[0] = static_cast<int>(first);
    triangle.v[1] = static_cast<int>(second);
    triangle.v[2] = static_cast<int>(third);
    mesh->faces.push_back(triangle);
}

void triangulateFaces(const std::vector<Face> &faces,
                      ScreenedPoissonResult *result)
{
    for (const Face &face : faces)
    {
        if (face.size() < 3)
        {
            ++result->statistics.skippedPolygonCount;
            continue;
        }

        const std::size_t face_count_before = result->mesh.faces.size();
        bool valid = true;
        for (std::size_t corner = 1; corner + 1 < face.size(); ++corner)
        {
            appendTriangle(face[0],
                           face[corner],
                           face[corner + 1],
                           result->mesh.vertices.size(),
                           &result->mesh,
                           &valid);
        }
        if (!valid)
        {
            result->mesh.faces.resize(face_count_before);
            ++result->statistics.skippedPolygonCount;
        }
    }
}

} // namespace

ScreenedPoissonResult ScreenedPoissonSurfaceBuilder::build(
    const std::vector<std::array<float, 3>> &points,
    const std::vector<std::array<float, 3>> &normals,
    const ScreenedPoissonOptions &options)
{
    ScreenedPoissonResult result;
    result.statistics.inputSampleCount = points.size();

    if (points.size() != normals.size())
    {
        result.error = "Screened Poisson points and normals must have the same size";
        return result;
    }
    if (!validateOptions(options, &result.error))
    {
        return result;
    }

    std::vector<OrientedSample> samples =
        prepareSamples(points, normals, &result.statistics);
    if (samples.size() < 16)
    {
        std::ostringstream message;
        message << "Screened Poisson requires at least 16 valid oriented points; got "
                << samples.size();
        result.error = message.str();
        return result;
    }

    constexpr unsigned int fem_signature = PoissonRecon::FEMDegreeAndBType<
        PoissonRecon::Reconstructor::Poisson::DefaultFEMDegree,
        PoissonRecon::Reconstructor::Poisson::DefaultFEMBoundary>::Signature;
    using FemSignatures = PoissonRecon::IsotropicUIntPack<Dimension, fem_signature>;
    using Implicit = PoissonRecon::Reconstructor::Implicit<
        Real, Dimension, FemSignatures>;
    using Solver = PoissonRecon::Reconstructor::Poisson::Solver<
        Real, Dimension, FemSignatures>;

    std::lock_guard<std::mutex> lock(poissonMutex());
    const auto previous_parallelization =
        PoissonRecon::ThreadPool::ParallelizationType;
    struct ParallelizationRestorer
    {
        PoissonRecon::ThreadPool::ParallelType previous;
        ~ParallelizationRestorer()
        {
            PoissonRecon::ThreadPool::ParallelizationType = previous;
        }
    } restorer{previous_parallelization};
    PoissonRecon::ThreadPool::ParallelizationType =
        PoissonRecon::ThreadPool::ParallelType::ASYNC;

    try
    {
        typename PoissonRecon::Reconstructor::Poisson::SolutionParameters<Real>
            solver_parameters;
        solver_parameters.verbose = options.verbose;
        solver_parameters.depth = static_cast<unsigned int>(options.depth);
        solver_parameters.pointWeight = options.pointWeight;
        solver_parameters.samplesPerNode = options.samplesPerNode;
        solver_parameters.scale = options.scale;
        solver_parameters.iters = static_cast<unsigned int>(options.solverIterations);
        solver_parameters.cgSolverAccuracy = options.cgSolverAccuracy;

        OrientedSampleStream sample_stream(samples);
        std::unique_ptr<Implicit> implicit(
            Solver::Solve(sample_stream, solver_parameters));
        if (!implicit)
        {
            result.error = "Screened Poisson solver returned no implicit surface";
            return result;
        }

        std::vector<Face> polygons;
        VertexStream vertex_stream(&result.mesh.vertices);
        FaceStream face_stream(&polygons);
        PoissonRecon::Reconstructor::LevelSetExtractionParameters
            extraction_parameters;
        extraction_parameters.linearFit = options.linearFit;
        extraction_parameters.outputGradients = true;
        extraction_parameters.forceManifold = options.forceManifold;
        extraction_parameters.polygonMesh = false;
        extraction_parameters.verbose = options.verbose;
        implicit->extractLevelSet(
            vertex_stream, face_stream, extraction_parameters);

        result.statistics.outputPolygonCount = polygons.size();
        triangulateFaces(polygons, &result);
        result.mesh.hasVertexColors = false;
        result.statistics.outputVertexCount = result.mesh.vertices.size();
        result.statistics.outputTriangleCount = result.mesh.faces.size();
        if (result.mesh.empty())
        {
            result.error = "Screened Poisson extraction produced an empty mesh";
            result.mesh = {};
            return result;
        }

        result.ok = true;
        return result;
    }
    catch (const std::exception &exception)
    {
        result.error = std::string("Screened Poisson failed: ") + exception.what();
    }
    catch (...)
    {
        result.error = "Screened Poisson failed with an unknown upstream error";
    }

    result.mesh = {};
    return result;
}

} // namespace xjw::mesh
