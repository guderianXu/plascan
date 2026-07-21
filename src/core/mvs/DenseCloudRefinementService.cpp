#include "DenseCloudRefinementService.h"
#include "io/PathIO.h"

#include <plapoint/io/ply_io.h>

#include <QDir>
#include <QFileInfo>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct Bounds2D
{
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    bool valid = false;
};

struct CellStats
{
    std::vector<float> zValues;
    float median = 0.0f;
    float robustSigma = 0.0f;
};

struct CellGate
{
    std::size_t count = 0;
    float median = 0.0f;
    float robustSigma = 0.0f;
};

struct CellPlaneMoments
{
    std::size_t count = 0;
    double sumX = 0.0;
    double sumY = 0.0;
    double sumZ = 0.0;
    double sumXX = 0.0;
    double sumXY = 0.0;
    double sumYY = 0.0;
    double sumXZ = 0.0;
    double sumYZ = 0.0;

    void add(const plapoint::io::PlyVertexPoint &point)
    {
        const double x = point.x;
        const double y = point.y;
        const double z = point.z;
        ++count;
        sumX += x;
        sumY += y;
        sumZ += z;
        sumXX += x * x;
        sumXY += x * y;
        sumYY += y * y;
        sumXZ += x * z;
        sumYZ += y * z;
    }

    void add(const CellPlaneMoments &other)
    {
        count += other.count;
        sumX += other.sumX;
        sumY += other.sumY;
        sumZ += other.sumZ;
        sumXX += other.sumXX;
        sumXY += other.sumXY;
        sumYY += other.sumYY;
        sumXZ += other.sumXZ;
        sumYZ += other.sumYZ;
    }
};

struct LocalPlaneGate
{
    bool valid = false;
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float threshold = 0.0f;
};

void ensureParentDirectory(const std::string &path)
{
    const std::filesystem::path file_path =
        xjw::common::io::toFilesystemPath(xjw::common::io::fromUtf8Path(path));
    const std::filesystem::path parent = file_path.parent_path();
    if (!parent.empty())
    {
        std::filesystem::create_directories(parent);
    }
}

double medianSorted(const std::vector<float> &values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const std::size_t mid = values.size() / 2;
    if ((values.size() % 2) == 1)
    {
        return values[mid];
    }
    return 0.5 * (static_cast<double>(values[mid - 1]) + static_cast<double>(values[mid]));
}

double percentileSorted(const std::vector<float> &values, double percentile)
{
    if (values.empty())
    {
        return 0.0;
    }

    const double clamped = std::clamp(percentile, 0.0, 100.0);
    const double position = (clamped / 100.0) * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = std::min(values.size() - 1, lo + 1);
    const double t = position - static_cast<double>(lo);
    return static_cast<double>(values[lo]) * (1.0 - t) + static_cast<double>(values[hi]) * t;
}

bool finitePoint(const plapoint::io::PlyVertexPoint &point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

void expandBounds(Bounds2D *bounds, const plapoint::io::PlyVertexPoint &point)
{
    if (!bounds || !finitePoint(point))
    {
        return;
    }

    bounds->minX = std::min(bounds->minX, point.x);
    bounds->minY = std::min(bounds->minY, point.y);
    bounds->maxX = std::max(bounds->maxX, point.x);
    bounds->maxY = std::max(bounds->maxY, point.y);
    bounds->valid = true;
}

int cellIndexForPoint(const plapoint::io::PlyVertexPoint &point,
                      const Bounds2D &bounds,
                      int grid_resolution)
{
    const float span_x = std::max(bounds.maxX - bounds.minX, 1.0e-6f);
    const float span_y = std::max(bounds.maxY - bounds.minY, 1.0e-6f);
    const int ix = std::clamp(static_cast<int>(std::floor((point.x - bounds.minX) / span_x * grid_resolution)),
                              0,
                              grid_resolution - 1);
    const int iy = std::clamp(static_cast<int>(std::floor((point.y - bounds.minY) / span_y * grid_resolution)),
                              0,
                              grid_resolution - 1);
    return iy * grid_resolution + ix;
}

bool solvePlaneSystem(const CellPlaneMoments &moments, float *a, float *b, float *c)
{
    if (!a || !b || !c || moments.count < 3)
    {
        return false;
    }

    double matrix[3][4] = {{moments.sumXX, moments.sumXY, moments.sumX, moments.sumXZ},
                           {moments.sumXY, moments.sumYY, moments.sumY, moments.sumYZ},
                           {moments.sumX, moments.sumY, static_cast<double>(moments.count), moments.sumZ}};

    for (int column = 0; column < 3; ++column)
    {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row)
        {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column]))
            {
                pivot = row;
            }
        }

        if (std::abs(matrix[pivot][column]) < 1.0e-12)
        {
            return false;
        }

        if (pivot != column)
        {
            for (int k = column; k < 4; ++k)
            {
                std::swap(matrix[column][k], matrix[pivot][k]);
            }
        }

        const double divisor = matrix[column][column];
        for (int k = column; k < 4; ++k)
        {
            matrix[column][k] /= divisor;
        }

        for (int row = 0; row < 3; ++row)
        {
            if (row == column)
            {
                continue;
            }
            const double factor = matrix[row][column];
            for (int k = column; k < 4; ++k)
            {
                matrix[row][k] -= factor * matrix[column][k];
            }
        }
    }

    *a = static_cast<float>(matrix[0][3]);
    *b = static_cast<float>(matrix[1][3]);
    *c = static_cast<float>(matrix[2][3]);
    return std::isfinite(*a) && std::isfinite(*b) && std::isfinite(*c);
}

CellPlaneMoments gatherPlaneNeighborhood(const std::vector<CellPlaneMoments> &moments,
                                         int cell_index,
                                         int grid_resolution)
{
    CellPlaneMoments neighborhood;
    const int cx = cell_index % grid_resolution;
    const int cy = cell_index / grid_resolution;
    for (int dy = -1; dy <= 1; ++dy)
    {
        const int y = cy + dy;
        if (y < 0 || y >= grid_resolution)
        {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int x = cx + dx;
            if (x < 0 || x >= grid_resolution)
            {
                continue;
            }
            neighborhood.add(moments[static_cast<std::size_t>(y * grid_resolution + x)]);
        }
    }
    return neighborhood;
}

float localPlaneResidual(const plapoint::io::PlyVertexPoint &point, const LocalPlaneGate &gate)
{
    return std::abs(point.z - (gate.a * point.x + gate.b * point.y + gate.c));
}

template <typename Func>
void forEachBinaryPlyRecord(const std::string &input_path,
                            const plapoint::io::PlyVertexStreamHeader &header,
                            int chunk_bytes,
                            Func &&func)
{
    std::ifstream file = xjw::common::io::openInputFile(input_path);
    if (!file)
    {
        throw std::runtime_error("Cannot open PLY file: " + input_path);
    }

    const auto chunks = plapoint::io::makePlyVertexChunks(header, chunk_bytes);
    std::vector<char> buffer;
    for (const auto &chunk : chunks)
    {
        std::string error;
        if (!plapoint::io::readPlyVertexChunk(file, header, chunk, &buffer, &error))
        {
            throw std::runtime_error("Failed to read PLY chunk: " + error);
        }

        for (std::uint64_t i = 0; i < chunk.vertexCount; ++i)
        {
            const char *record =
                buffer.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(header.vertexStride);
            const plapoint::io::PlyVertexPoint point = plapoint::io::readPlyVertexPoint(record, header);
            func(record, point);
        }
    }
}

template <typename Func>
void forEachBinaryPlyPoint(const std::string &input_path,
                           const plapoint::io::PlyVertexStreamHeader &header,
                           int chunk_bytes,
                           Func &&func)
{
    forEachBinaryPlyRecord(input_path,
                           header,
                           chunk_bytes,
                           [&](const char *, const plapoint::io::PlyVertexPoint &point) {
                               func(point);
                           });
}

void fillCellStats(std::vector<CellStats> *cells)
{
    if (!cells)
    {
        return;
    }

    for (CellStats &cell : *cells)
    {
        if (cell.zValues.empty())
        {
            continue;
        }

        std::sort(cell.zValues.begin(), cell.zValues.end());
        cell.median = static_cast<float>(medianSorted(cell.zValues));

        std::vector<float> deviations;
        deviations.reserve(cell.zValues.size());
        for (float z : cell.zValues)
        {
            deviations.push_back(std::abs(z - cell.median));
        }
        std::sort(deviations.begin(), deviations.end());
        cell.robustSigma = static_cast<float>(1.4826 * medianSorted(deviations));
    }
}

void summarizeCellRanges(const std::vector<CellStats> &cells,
                         int min_cell_points,
                         double *median_range,
                         double *p95_range)
{
    std::vector<float> ranges;
    ranges.reserve(cells.size());
    for (const CellStats &cell : cells)
    {
        if (static_cast<int>(cell.zValues.size()) < min_cell_points || cell.zValues.empty())
        {
            continue;
        }

        const auto minmax = std::minmax_element(cell.zValues.begin(), cell.zValues.end());
        ranges.push_back(*minmax.second - *minmax.first);
    }

    std::sort(ranges.begin(), ranges.end());
    if (median_range)
    {
        *median_range = medianSorted(ranges);
    }
    if (p95_range)
    {
        *p95_range = percentileSorted(ranges, 95.0);
    }
}

bool shouldKeepPoint(const plapoint::io::PlyVertexPoint &point,
                     const Bounds2D &bounds,
                     const std::vector<CellGate> &gates,
                     const xjw::mvs::TerrainHeightSpikeFilterOptions &options)
{
    if (!finitePoint(point))
    {
        return false;
    }
    if (!options.enabled || !bounds.valid)
    {
        return true;
    }

    const int cell_index = cellIndexForPoint(point, bounds, options.gridResolution);
    const CellGate &gate = gates[static_cast<std::size_t>(cell_index)];
    if (static_cast<int>(gate.count) < options.minCellPoints)
    {
        return true;
    }

    const float threshold = std::max(options.minHeightThreshold, options.madMultiplier * gate.robustSigma);
    return std::abs(point.z - gate.median) <= threshold;
}

void addResidualSample(std::vector<float> *samples,
                       std::size_t *seen_count,
                       float residual,
                       std::size_t max_samples)
{
    if (!samples || !seen_count || !std::isfinite(residual))
    {
        return;
    }

    ++(*seen_count);
    if (samples->size() < max_samples)
    {
        samples->push_back(residual);
        return;
    }

    const std::size_t slot = ((*seen_count) * 2654435761ULL) % max_samples;
    (*samples)[slot] = residual;
}

void finalizeResidualGate(LocalPlaneGate *gate,
                          std::vector<float> *samples,
                          int min_points,
                          float min_residual_threshold,
                          float mad_multiplier)
{
    if (!gate || !samples || !gate->valid || static_cast<int>(samples->size()) < min_points)
    {
        if (gate)
        {
            gate->valid = false;
        }
        return;
    }

    std::sort(samples->begin(), samples->end());
    const float median = static_cast<float>(medianSorted(*samples));

    std::vector<float> deviations;
    deviations.reserve(samples->size());
    for (float residual : *samples)
    {
        deviations.push_back(std::abs(residual - median));
    }
    std::sort(deviations.begin(), deviations.end());

    const float robust_sigma = static_cast<float>(1.4826 * medianSorted(deviations));
    gate->threshold = std::max(min_residual_threshold, median + mad_multiplier * robust_sigma);
}

std::vector<LocalPlaneGate> buildLocalPlaneGates(const std::string &input_path,
                                                 const plapoint::io::PlyVertexStreamHeader &header,
                                                 int chunk_bytes,
                                                 const Bounds2D &bounds,
                                                 const std::vector<CellGate> &height_gates,
                                                 const xjw::mvs::TerrainHeightSpikeFilterOptions &options)
{
    if (!options.enabled || !options.localPlaneFilterEnabled || !bounds.valid)
    {
        return {};
    }

    const int grid_resolution = options.gridResolution;
    const std::size_t cell_count = static_cast<std::size_t>(grid_resolution * grid_resolution);
    std::vector<CellPlaneMoments> moments(cell_count);

    forEachBinaryPlyPoint(input_path, header, chunk_bytes, [&](const plapoint::io::PlyVertexPoint &point) {
        if (!shouldKeepPoint(point, bounds, height_gates, options))
        {
            return;
        }
        const int cell_index = cellIndexForPoint(point, bounds, grid_resolution);
        moments[static_cast<std::size_t>(cell_index)].add(point);
    });

    std::vector<LocalPlaneGate> plane_gates(cell_count);
    for (std::size_t i = 0; i < cell_count; ++i)
    {
        CellPlaneMoments neighborhood = gatherPlaneNeighborhood(moments,
                                                                static_cast<int>(i),
                                                                grid_resolution);
        if (static_cast<int>(neighborhood.count) < options.localPlaneMinPoints)
        {
            continue;
        }

        LocalPlaneGate gate;
        gate.valid = solvePlaneSystem(neighborhood, &gate.a, &gate.b, &gate.c);
        plane_gates[i] = gate;
    }

    constexpr std::size_t kMaxResidualSamplesPerCell = 512;
    std::vector<std::vector<float>> residual_samples(cell_count);
    std::vector<std::size_t> residual_seen(cell_count, 0);

    forEachBinaryPlyPoint(input_path, header, chunk_bytes, [&](const plapoint::io::PlyVertexPoint &point) {
        if (!shouldKeepPoint(point, bounds, height_gates, options))
        {
            return;
        }
        const int cell_index = cellIndexForPoint(point, bounds, grid_resolution);
        LocalPlaneGate &gate = plane_gates[static_cast<std::size_t>(cell_index)];
        if (!gate.valid)
        {
            return;
        }

        const float residual = localPlaneResidual(point, gate);
        addResidualSample(&residual_samples[static_cast<std::size_t>(cell_index)],
                          &residual_seen[static_cast<std::size_t>(cell_index)],
                          residual,
                          kMaxResidualSamplesPerCell);
    });

    for (std::size_t i = 0; i < cell_count; ++i)
    {
        finalizeResidualGate(&plane_gates[i],
                             &residual_samples[i],
                             options.localPlaneMinPoints,
                             options.localPlaneMinResidualThreshold,
                             options.localPlaneMadMultiplier);
    }

    return plane_gates;
}

bool shouldKeepPointWithLocalPlane(const plapoint::io::PlyVertexPoint &point,
                                   const Bounds2D &bounds,
                                   const std::vector<CellGate> &height_gates,
                                   const std::vector<LocalPlaneGate> &plane_gates,
                                   const xjw::mvs::TerrainHeightSpikeFilterOptions &options)
{
    if (!shouldKeepPoint(point, bounds, height_gates, options))
    {
        return false;
    }
    if (plane_gates.empty() || !options.enabled || !options.localPlaneFilterEnabled || !bounds.valid)
    {
        return true;
    }

    const int cell_index = cellIndexForPoint(point, bounds, options.gridResolution);
    const LocalPlaneGate &gate = plane_gates[static_cast<std::size_t>(cell_index)];
    if (!gate.valid)
    {
        return true;
    }

    return localPlaneResidual(point, gate) <= gate.threshold;
}

const char *plyScalarTypeName(plapoint::io::PlyVertexScalarType type)
{
    switch (type)
    {
    case plapoint::io::PlyVertexScalarType::Int8:
        return "char";
    case plapoint::io::PlyVertexScalarType::UInt8:
        return "uchar";
    case plapoint::io::PlyVertexScalarType::Int16:
        return "short";
    case plapoint::io::PlyVertexScalarType::UInt16:
        return "ushort";
    case plapoint::io::PlyVertexScalarType::Int32:
        return "int";
    case plapoint::io::PlyVertexScalarType::UInt32:
        return "uint";
    case plapoint::io::PlyVertexScalarType::Float32:
        return "float";
    case plapoint::io::PlyVertexScalarType::Float64:
        return "double";
    default:
        return "float";
    }
}

void writeBinaryPlyHeader(std::ofstream &out,
                          std::size_t vertex_count,
                          const plapoint::io::PlyVertexStreamHeader &input_header)
{
    out << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "comment PlaScan dense_cloud_refine_cli streaming output; vertex scalar properties preserved\n"
        << "element vertex " << vertex_count << "\n";
    for (const plapoint::io::PlyVertexStreamProperty &property : input_header.properties)
    {
        out << "property " << plyScalarTypeName(property.type) << ' ' << property.name << "\n";
    }
    out << "end_header\n";
}

void writeBinaryPlyPointRecord(std::ofstream &out,
                               const char *record,
                               const plapoint::io::PlyVertexStreamHeader &input_header)
{
    out.write(record, input_header.vertexStride);
}

bool refineBinaryPlyStreaming(const std::string &input_path,
                              const std::string &output_path,
                              int streaming_chunk_mb,
                              const xjw::mvs::TerrainHeightSpikeFilterOptions &options,
                              xjw::mvs::TerrainHeightSpikeFilterReport *report)
{
    plapoint::io::PlyVertexStreamHeader header;
    std::string header_error;
    if (!plapoint::io::parseBinaryPlyVertexStreamHeader(
            xjw::common::io::toNativeNarrowPath(input_path), &header, &header_error))
    {
        return false;
    }

    const int chunk_bytes = std::max(1, streaming_chunk_mb) * 1024 * 1024;
    xjw::mvs::TerrainHeightSpikeFilterReport local_report;
    local_report.inputPoints = static_cast<std::size_t>(header.vertexCount);

    Bounds2D bounds;
    std::size_t finite_count = 0;
    forEachBinaryPlyPoint(input_path, header, chunk_bytes, [&](const plapoint::io::PlyVertexPoint &point) {
        if (finitePoint(point))
        {
            ++finite_count;
            expandBounds(&bounds, point);
        }
    });

    std::vector<CellGate> gates;
    if (options.enabled && bounds.valid)
    {
        std::vector<CellStats> cells(static_cast<std::size_t>(options.gridResolution * options.gridResolution));
        forEachBinaryPlyPoint(input_path, header, chunk_bytes, [&](const plapoint::io::PlyVertexPoint &point) {
            if (!finitePoint(point))
            {
                return;
            }
            const int cell_index = cellIndexForPoint(point, bounds, options.gridResolution);
            cells[static_cast<std::size_t>(cell_index)].zValues.push_back(point.z);
        });

        fillCellStats(&cells);
        summarizeCellRanges(cells,
                            options.minCellPoints,
                            &local_report.medianCellZRangeBefore,
                            &local_report.p95CellZRangeBefore);

        gates.resize(cells.size());
        for (std::size_t i = 0; i < cells.size(); ++i)
        {
            gates[i].count = cells[i].zValues.size();
            gates[i].median = cells[i].median;
            gates[i].robustSigma = cells[i].robustSigma;
        }
        cells.clear();
        cells.shrink_to_fit();
    }

    std::vector<CellStats> kept_cells;
    if (options.enabled && bounds.valid)
    {
        kept_cells.resize(static_cast<std::size_t>(options.gridResolution * options.gridResolution));
    }

    const std::vector<LocalPlaneGate> plane_gates =
        buildLocalPlaneGates(input_path, header, chunk_bytes, bounds, gates, options);

    std::size_t height_kept_count = 0;
    std::size_t kept_count = 0;
    forEachBinaryPlyPoint(input_path, header, chunk_bytes, [&](const plapoint::io::PlyVertexPoint &point) {
        if (shouldKeepPoint(point, bounds, gates, options))
        {
            ++height_kept_count;
        }
        if (shouldKeepPointWithLocalPlane(point, bounds, gates, plane_gates, options))
        {
            ++kept_count;
            if (!kept_cells.empty())
            {
                const int cell_index = cellIndexForPoint(point, bounds, options.gridResolution);
                kept_cells[static_cast<std::size_t>(cell_index)].zValues.push_back(point.z);
            }
        }
    });

    if (!kept_cells.empty())
    {
        summarizeCellRanges(kept_cells,
                            options.minCellPoints,
                            &local_report.medianCellZRangeAfter,
                            &local_report.p95CellZRangeAfter);
        kept_cells.clear();
        kept_cells.shrink_to_fit();
    }

    ensureParentDirectory(output_path);
    std::ofstream out = xjw::common::io::openOutputFile(output_path,
                                                            std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("Cannot write PLY file: " + output_path);
    }
    writeBinaryPlyHeader(out, kept_count, header);

    forEachBinaryPlyRecord(input_path,
                           header,
                           chunk_bytes,
                           [&](const char *record, const plapoint::io::PlyVertexPoint &point) {
        if (shouldKeepPointWithLocalPlane(point, bounds, gates, plane_gates, options))
        {
            writeBinaryPlyPointRecord(out, record, header);
        }
    });

    local_report.outputPoints = kept_count;
    local_report.removedPoints = local_report.inputPoints - local_report.outputPoints;
    if (height_kept_count >= kept_count)
    {
        local_report.localPlaneRemovedPoints = height_kept_count - kept_count;
    }
    if (!options.enabled || !bounds.valid)
    {
        local_report.outputPoints = finite_count;
        local_report.removedPoints = local_report.inputPoints - local_report.outputPoints;
        local_report.localPlaneRemovedPoints = 0;
    }
    if (report)
    {
        *report = local_report;
    }
    return true;
}

xjw::mvs::TerrainHeightSpikeFilterReport combinePassReports(
    const std::vector<xjw::mvs::TerrainHeightSpikeFilterReport> &pass_reports)
{
    xjw::mvs::TerrainHeightSpikeFilterReport combined;
    if (pass_reports.empty())
    {
        return combined;
    }

    combined = pass_reports.back();
    combined.inputPoints = pass_reports.front().inputPoints;
    combined.outputPoints = pass_reports.back().outputPoints;
    combined.removedPoints = combined.inputPoints - combined.outputPoints;
    combined.localPlaneRemovedPoints = 0;
    for (const xjw::mvs::TerrainHeightSpikeFilterReport &pass_report : pass_reports)
    {
        combined.localPlaneRemovedPoints += pass_report.localPlaneRemovedPoints;
    }
    combined.medianCellZRangeBefore = pass_reports.front().medianCellZRangeBefore;
    combined.p95CellZRangeBefore = pass_reports.front().p95CellZRangeBefore;
    combined.medianCellZRangeAfter = pass_reports.back().medianCellZRangeAfter;
    combined.p95CellZRangeAfter = pass_reports.back().p95CellZRangeAfter;
    return combined;
}

std::string temporaryPassPath(const std::string &output_path, int pass_index)
{
    const QFileInfo outputInfo(xjw::common::io::fromUtf8Path(output_path));
    const QString extension = outputInfo.suffix().isEmpty()
                                  ? QString()
                                  : QStringLiteral(".") + outputInfo.suffix();
    const QString passName = QStringLiteral("%1.pass%2.tmp%3")
                                 .arg(outputInfo.completeBaseName())
                                 .arg(pass_index)
                                 .arg(extension);
    return xjw::common::io::toUtf8Path(outputInfo.dir().filePath(passName));
}

void removeTemporaryPaths(const std::vector<std::string> &paths)
{
    for (const std::string &path : paths)
    {
        std::error_code ec;
        std::filesystem::remove(xjw::common::io::toFilesystemPath(xjw::common::io::fromUtf8Path(path)), ec);
    }
}

bool refineBinaryPlyStreamingPasses(const std::string &input_path,
                                    const std::string &output_path,
                                    int streaming_chunk_mb,
                                    int terrain_filter_passes,
                                    const xjw::mvs::TerrainHeightSpikeFilterOptions &options,
                                    xjw::mvs::TerrainHeightSpikeFilterReport *report,
                                    std::vector<xjw::mvs::TerrainHeightSpikeFilterReport> *pass_reports)
{
    const int pass_count = std::max(1, terrain_filter_passes);
    std::vector<xjw::mvs::TerrainHeightSpikeFilterReport> local_pass_reports;
    std::vector<std::string> temporary_paths;
    std::string current_input = input_path;

    for (int pass_index = 1; pass_index <= pass_count; ++pass_index)
    {
        const bool final_pass = pass_index == pass_count;
        const std::string current_output = final_pass ? output_path : temporaryPassPath(output_path, pass_index);
        if (!final_pass)
        {
            temporary_paths.push_back(current_output);
        }

        xjw::mvs::TerrainHeightSpikeFilterReport pass_report;
        if (!refineBinaryPlyStreaming(current_input, current_output, streaming_chunk_mb, options, &pass_report))
        {
            removeTemporaryPaths(temporary_paths);
            return false;
        }
        local_pass_reports.push_back(pass_report);

        if (current_input != input_path)
        {
            std::error_code ec;
            std::filesystem::remove(current_input, ec);
        }
        current_input = current_output;
    }

    removeTemporaryPaths(temporary_paths);
    if (report)
    {
        *report = combinePassReports(local_pass_reports);
    }
    if (pass_reports)
    {
        *pass_reports = std::move(local_pass_reports);
    }
    return true;
}

} // namespace

namespace xjw::mvs
{

bool refineDenseCloud(const DenseCloudRefinementRequest &request,
                      DenseCloudRefinementResult *result,
                      std::string *errorMessage)
{
    if (!result)
    {
        if (errorMessage)
        {
            *errorMessage = "Dense cloud refinement result is null";
        }
        return false;
    }
    if (request.inputPath.empty() || request.outputPath.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "Dense cloud refinement input or output path is empty";
        }
        return false;
    }

    try
    {
        const int passCount = std::clamp(request.filterPasses, 1, 8);
        const int streamingChunkMb = std::clamp(request.streamingChunkMb, 1, 2048);
        result->passReports.clear();
        if (refineBinaryPlyStreamingPasses(request.inputPath,
                                           request.outputPath,
                                           streamingChunkMb,
                                           passCount,
                                           request.filterOptions,
                                           &result->report,
                                           &result->passReports))
        {
            result->mode = "streaming";
            return true;
        }

        auto cloud = plapoint::io::readPly<float>(
            xjw::common::io::toNativeNarrowPath(request.inputPath));
        if (!cloud)
        {
            if (errorMessage)
            {
                *errorMessage = "无法读取输入点云: " + request.inputPath;
            }
            return false;
        }

        DensePointCloud refined = std::move(*cloud);
        for (int passIndex = 0; passIndex < passCount; ++passIndex)
        {
            TerrainHeightSpikeFilterReport passReport;
            refined = filterTerrainHeightSpikes(refined, request.filterOptions, &passReport);
            result->passReports.push_back(passReport);
        }
        result->report = combinePassReports(result->passReports);

        ensureParentDirectory(request.outputPath);
        plapoint::io::writePly<float>(xjw::common::io::toNativeNarrowPath(request.outputPath),
                                      refined,
                                      plapoint::io::PlyFormat::BinaryLE);
        result->mode = "in_memory";
        return true;
    }
    catch (const std::bad_alloc &)
    {
        if (errorMessage)
        {
            *errorMessage = "内存不足，无法细化该点云；请降低输入规模或调整流式分块大小";
        }
    }
    catch (const std::exception &exception)
    {
        if (errorMessage)
        {
            *errorMessage = exception.what();
        }
    }
    return false;
}

} // namespace xjw::mvs
