#include "SurfaceReconstructor.h"
#include "PointCloudPreprocess.h"
#include "SurfaceReconstructorHeightGrid.h"
#include "SurfaceReconstructorPostprocess.h"
#include "io/PathIO.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plamatrix/dense/dense_matrix.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <new>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xjw
{
namespace mesh
{

namespace
{

using PlaPointCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using PlyHeader = plapoint::io::PlyVertexStreamHeader;
using PlyVertexChunk = plapoint::io::PlyVertexChunk;

std::string localizePlyStreamError(const std::string &error)
{
    if (error.find("Cannot open") != std::string::npos)
    {
        return "无法打开 PLY 文件";
    }
    if (error.find("Not a PLY file") != std::string::npos ||
        error.find("missing magic header") != std::string::npos)
    {
        return "不是有效的 PLY 文件";
    }
    if (error.find("Only binary_little_endian") != std::string::npos)
    {
        return "仅支持 binary_little_endian PLY 的流式读取";
    }
    if (error.find("end_header") != std::string::npos)
    {
        return "PLY 头缺少 end_header";
    }
    if (error.find("positive vertex count") != std::string::npos)
    {
        return "PLY 头缺少有效顶点数量";
    }
    if (error.find("no scalar properties") != std::string::npos)
    {
        return "PLY 顶点属性为空";
    }
    if (error.find("x, y, and z") != std::string::npos)
    {
        return "PLY 顶点缺少 x/y/z 坐标属性";
    }
    if (error.find("list properties") != std::string::npos)
    {
        return "暂不支持顶点元素中的 list property";
    }
    if (error.find("Unsupported PLY vertex property type") != std::string::npos)
    {
        return "不支持的 PLY 顶点属性类型";
    }
    return "PLY 流式读取失败: " + error;
}

bool parseBinaryPlyHeader(const std::string &cloudPath, PlyHeader *header, std::string *errorMsg)
{
    std::string streamError;
    const bool ok = plapoint::io::parseBinaryPlyVertexStreamHeader(
        xjw::common::io::toNativeNarrowPath(cloudPath), header, &streamError);
    if (!ok && errorMsg)
    {
        *errorMsg = localizePlyStreamError(streamError);
    }
    return ok;
}

detail::PointXYZRGB toPointXYZRGB(const plapoint::io::PlyVertexPoint &source)
{
    detail::PointXYZRGB point;
    point.x = source.x;
    point.y = source.y;
    point.z = source.z;
    point.r = source.r;
    point.g = source.g;
    point.b = source.b;
    if (source.hasNormal)
    {
        point.hasNormal = true;
        point.nx = source.nx;
        point.ny = source.ny;
        point.nz = source.nz;
    }
    return point;
}

std::vector<detail::PointXYZRGB> toPointXYZRGB(const std::vector<plapoint::io::PlyVertexPoint> &source)
{
    std::vector<detail::PointXYZRGB> points;
    points.reserve(source.size());
    for (const plapoint::io::PlyVertexPoint &point : source)
    {
        points.push_back(toPointXYZRGB(point));
    }
    return points;
}

std::vector<detail::PointXYZRGB> sampleBinaryPlyForMeshing(const std::string &cloudPath,
                                                           const PlyHeader &header,
                                                           int maxPoints,
                                                           std::string *errorMsg)
{
    std::string streamError;
    const auto sampled = plapoint::io::sampleBinaryPlyVertices(cloudPath, header, maxPoints, &streamError);
    if (sampled.empty() && errorMsg)
    {
        *errorMsg = streamError.empty()
            ? "PLY 流式抽样没有读到有效顶点"
            : ("PLY 流式抽样失败: " + localizePlyStreamError(streamError));
    }
    return toPointXYZRGB(sampled);
}

struct StreamingBounds
{
    bool valid = false;
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    std::uint64_t pointCount = 0;
};

struct StreamingGridCell
{
    double zSum = 0.0;
    double zSumSq = 0.0;
    double weight = 0.0;
    double rSum = 0.0;
    double gSum = 0.0;
    double bSum = 0.0;
    std::uint32_t count = 0;
};

struct StreamingTile
{
    int x = 0;
    int y = 0;
    int nx = 1;
    int ny = 1;
    StreamingBounds bounds;
};

int resolveStreamingThreadCount(const ReconstructionConfig &config)
{
    if (config.streamingThreads > 0)
    {
        return std::clamp(config.streamingThreads, 1, 64);
    }

    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    if (hardwareThreads == 0)
    {
        return 4;
    }
    return std::clamp(static_cast<int>(hardwareThreads), 1, 16);
}

std::vector<PlyVertexChunk> makePlyVertexChunks(const PlyHeader &header,
                                                const ReconstructionConfig &config)
{
    const int chunkBytes = std::clamp(config.streamingChunkBytes, 256, 512 * 1024 * 1024);
    return plapoint::io::makePlyVertexChunks(header, chunkBytes);
}

int resolveStreamingTileSide(const PlyHeader &header,
                             const ReconstructionConfig &config)
{
    if (config.streamingTileCount > 0)
    {
        return std::clamp(config.streamingTileCount, 1, 16);
    }

    const double pointRatio = static_cast<double>(header.vertexCount) /
                              static_cast<double>(std::max(1, config.maxInputPointsForMeshing));
    const double resolutionRatio = static_cast<double>(std::clamp(config.resolution, 32, 2048)) / 128.0;
    const int side = static_cast<int>(std::ceil(std::sqrt(std::max(1.0, pointRatio * resolutionRatio))));
    return std::clamp(side, 2, 8);
}

StreamingBounds makeTileBounds(const StreamingBounds &globalBounds,
                               int tileX,
                               int tileY,
                               int tileSide,
                               int overlapCells,
                               int resolution)
{
    const float width = std::max(1.0e-6f, globalBounds.maxX - globalBounds.minX);
    const float height = std::max(1.0e-6f, globalBounds.maxY - globalBounds.minY);
    const float tileWidth = width / static_cast<float>(tileSide);
    const float tileHeight = height / static_cast<float>(tileSide);
    const float cellPaddingX = tileWidth / static_cast<float>(std::max(8, resolution));
    const float cellPaddingY = tileHeight / static_cast<float>(std::max(8, resolution));
    const float overlapX = cellPaddingX * static_cast<float>(std::max(0, overlapCells));
    const float overlapY = cellPaddingY * static_cast<float>(std::max(0, overlapCells));

    StreamingBounds bounds;
    bounds.valid = true;
    bounds.minX = globalBounds.minX + static_cast<float>(tileX) * tileWidth - overlapX;
    bounds.maxX = globalBounds.minX + static_cast<float>(tileX + 1) * tileWidth + overlapX;
    bounds.minY = globalBounds.minY + static_cast<float>(tileY) * tileHeight - overlapY;
    bounds.maxY = globalBounds.minY + static_cast<float>(tileY + 1) * tileHeight + overlapY;
    bounds.minX = std::max(bounds.minX, globalBounds.minX);
    bounds.maxX = std::min(bounds.maxX, globalBounds.maxX);
    bounds.minY = std::max(bounds.minY, globalBounds.minY);
    bounds.maxY = std::min(bounds.maxY, globalBounds.maxY);
    bounds.pointCount = 0;
    return bounds;
}

std::vector<StreamingTile> makeStreamingTiles(const StreamingBounds &globalBounds,
                                              const PlyHeader &header,
                                              const ReconstructionConfig &config)
{
    const int tileSide = resolveStreamingTileSide(header, config);
    const int resolution = std::clamp(config.resolution, 32, 2048);
    std::vector<StreamingTile> tiles;
    tiles.reserve(static_cast<std::size_t>(tileSide * tileSide));
    for (int y = 0; y < tileSide; ++y)
    {
        for (int x = 0; x < tileSide; ++x)
        {
            StreamingTile tile;
            tile.x = x;
            tile.y = y;
            tile.nx = tileSide;
            tile.ny = tileSide;
            tile.bounds = makeTileBounds(globalBounds,
                                         x,
                                         y,
                                         tileSide,
                                         config.streamingTileOverlapCells,
                                         resolution);
            tiles.push_back(tile);
        }
    }
    return tiles;
}

detail::PointXYZRGB readPointFromPlyRecord(const char *record,
                                           const PlyHeader &header)
{
    return toPointXYZRGB(plapoint::io::readPlyVertexPoint(record, header));
}

bool isFinitePoint(const detail::PointXYZRGB &point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

int activeChunkWorkerCount(int requestedThreads,
                           std::size_t chunkCount)
{
    if (chunkCount == 0)
    {
        return 1;
    }

    const int maxWorkers = static_cast<int>(std::min<std::size_t>(chunkCount, 64));
    return std::clamp(requestedThreads, 1, std::max(1, maxWorkers));
}

bool readPlyChunkFromStream(std::ifstream &file,
                            const PlyHeader &header,
                            const PlyVertexChunk &chunk,
                            std::vector<char> *buffer,
                            std::string *errorMsg)
{
    std::string streamError;
    const bool ok = plapoint::io::readPlyVertexChunk(file, header, chunk, buffer, &streamError);
    if (!ok && errorMsg)
    {
        *errorMsg = streamError.empty()
            ? "读取 PLY 顶点分块失败"
            : ("读取 PLY 顶点分块失败: " + localizePlyStreamError(streamError));
    }
    return ok;
}

void mergeStreamingBounds(const StreamingBounds &src,
                          StreamingBounds *dst)
{
    if (!dst || !src.valid)
    {
        return;
    }

    dst->valid = true;
    dst->minX = std::min(dst->minX, src.minX);
    dst->maxX = std::max(dst->maxX, src.maxX);
    dst->minY = std::min(dst->minY, src.minY);
    dst->maxY = std::max(dst->maxY, src.maxY);
    dst->pointCount += src.pointCount;
}

bool estimateStreamingBoundsParallel(const std::string &cloudPath,
                                     const PlyHeader &header,
                                     const std::vector<PlyVertexChunk> &chunks,
                                     int requestedThreads,
                                     StreamingBounds *bounds,
                                     std::string *errorMsg)
{
    if (!bounds)
    {
        return false;
    }
    *bounds = StreamingBounds{};

    if (chunks.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "PLY 顶点分块为空";
        }
        return false;
    }

    const int workerCount = activeChunkWorkerCount(requestedThreads, chunks.size());
    std::atomic<std::size_t> nextChunk{0};
    std::atomic<bool> failed{false};
    std::vector<StreamingBounds> localBounds(static_cast<std::size_t>(workerCount));
    std::vector<std::string> errors(static_cast<std::size_t>(workerCount));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(workerCount));

    for (int worker = 0; worker < workerCount; ++worker)
    {
        workers.emplace_back([&, worker]() {
            std::ifstream file = xjw::common::io::openInputFile(cloudPath);
            if (!file)
            {
                errors[static_cast<std::size_t>(worker)] = "无法打开 PLY 文件";
                failed.store(true, std::memory_order_relaxed);
                return;
            }

            std::vector<char> buffer;
            StreamingBounds &local = localBounds[static_cast<std::size_t>(worker)];
            while (!failed.load(std::memory_order_relaxed))
            {
                const std::size_t chunkIndex = nextChunk.fetch_add(1, std::memory_order_relaxed);
                if (chunkIndex >= chunks.size())
                {
                    break;
                }

                std::string chunkError;
                if (!readPlyChunkFromStream(file, header, chunks[chunkIndex], &buffer, &chunkError))
                {
                    errors[static_cast<std::size_t>(worker)] = chunkError.empty() ? "读取 PLY 顶点分块失败" : chunkError;
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }

                const char *data = buffer.data();
                const std::uint64_t stride = static_cast<std::uint64_t>(header.vertexStride);
                for (std::uint64_t i = 0; i < chunks[chunkIndex].vertexCount; ++i)
                {
                    const detail::PointXYZRGB point = readPointFromPlyRecord(data + static_cast<std::size_t>(i * stride),
                                                                             header);
                    if (!isFinitePoint(point))
                    {
                        continue;
                    }

                    local.valid = true;
                    local.minX = std::min(local.minX, point.x);
                    local.maxX = std::max(local.maxX, point.x);
                    local.minY = std::min(local.minY, point.y);
                    local.maxY = std::max(local.maxY, point.y);
                    ++local.pointCount;
                }
            }
        });
    }

    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    if (failed.load(std::memory_order_relaxed))
    {
        if (errorMsg)
        {
            *errorMsg = "并行扫描 PLY 顶点失败";
            for (const std::string &error : errors)
            {
                if (!error.empty())
                {
                    *errorMsg = error;
                    break;
                }
            }
        }
        return false;
    }

    for (const StreamingBounds &local : localBounds)
    {
        mergeStreamingBounds(local, bounds);
    }
    if (!bounds->valid || bounds->pointCount < 100)
    {
        if (errorMsg)
        {
            *errorMsg = "PLY 中有效点不足，无法构建流式地形网格";
        }
        return false;
    }
    return true;
}

detail::HeightGrid makeEmptyStreamingHeightGrid(const StreamingBounds &bounds,
                                                const ReconstructionConfig &config)
{
    detail::HeightGrid grid;

    const float padding = std::max(0.0f, config.padding);
    const float rawDx = std::max(1.0e-6f, bounds.maxX - bounds.minX);
    const float rawDy = std::max(1.0e-6f, bounds.maxY - bounds.minY);
    grid.minX = bounds.minX - rawDx * padding;
    grid.minY = bounds.minY - rawDy * padding;
    const float maxX = bounds.maxX + rawDx * padding;
    const float maxY = bounds.maxY + rawDy * padding;
    const float dx = std::max(1.0e-6f, maxX - grid.minX);
    const float dy = std::max(1.0e-6f, maxY - grid.minY);
    const float maxSpan = std::max(dx, dy);
    const int resolution = std::clamp(config.resolution, 32, 2048);

    grid.nx = std::max(8, static_cast<int>(std::round(resolution * dx / maxSpan)));
    grid.ny = std::max(8, static_cast<int>(std::round(resolution * dy / maxSpan)));
    grid.stepX = dx / static_cast<float>(std::max(1, grid.nx - 1));
    grid.stepY = dy / static_cast<float>(std::max(1, grid.ny - 1));
    const auto cells = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
    grid.heights.assign(cells, 0.0f);
    grid.sumW.assign(cells, 0.0f);
    grid.valid.assign(cells, 0);
    grid.fillPass.assign(cells, 0);
    return grid;
}

ReconstructionConfig makeStreamingTileConfig(const ReconstructionConfig &config,
                                             int tileSide)
{
    ReconstructionConfig tileConfig = config;
    const int baseResolution = std::clamp(config.resolution, 32, 2048);
    tileConfig.resolution = std::clamp(baseResolution, 32, 2048);
    tileConfig.padding = 0.0f;
    const int maxTileHoleFill = std::max(1, config.holeFillPasses / std::max(1, tileSide));
    tileConfig.holeFillPasses = std::min(config.holeFillPasses, maxTileHoleFill);
    return tileConfig;
}

std::size_t gridIndex(const detail::HeightGrid &grid, int ix, int iy)
{
    return static_cast<std::size_t>(iy * grid.nx + ix);
}

bool pointToStreamingGridIndex(const detail::PointXYZRGB &point,
                               const detail::HeightGrid &grid,
                               std::size_t *index)
{
    if (!index || !isFinitePoint(point) || grid.nx <= 0 || grid.ny <= 0 || grid.stepX <= 0.0f || grid.stepY <= 0.0f)
    {
        return false;
    }

    const int ix = std::clamp(static_cast<int>(std::lround((point.x - grid.minX) / grid.stepX)), 0, grid.nx - 1);
    const int iy = std::clamp(static_cast<int>(std::lround((point.y - grid.minY) / grid.stepY)), 0, grid.ny - 1);
    *index = gridIndex(grid, ix, iy);
    return true;
}

void accumulateStreamingCell(const detail::PointXYZRGB &point,
                             StreamingGridCell *cell)
{
    if (!cell)
    {
        return;
    }

    cell->zSum += point.z;
    cell->zSumSq += static_cast<double>(point.z) * static_cast<double>(point.z);
    cell->weight += 1.0;
    cell->rSum += point.r;
    cell->gSum += point.g;
    cell->bSum += point.b;
    ++cell->count;
}

void mergeStreamingCell(const StreamingGridCell &src,
                        StreamingGridCell *dst)
{
    if (!dst || src.count == 0)
    {
        return;
    }

    dst->zSum += src.zSum;
    dst->zSumSq += src.zSumSq;
    dst->weight += src.weight;
    dst->rSum += src.rSum;
    dst->gSum += src.gSum;
    dst->bSum += src.bSum;
    dst->count += src.count;
}

void addPointToStreamingGrid(const detail::PointXYZRGB &point,
                             const detail::HeightGrid &grid,
                             std::vector<StreamingGridCell> *cells)
{
    if (!cells || !isFinitePoint(point))
    {
        return;
    }

    std::size_t index = 0;
    if (!pointToStreamingGridIndex(point, grid, &index) || index >= cells->size())
    {
        return;
    }
    accumulateStreamingCell(point, &(*cells)[index]);
}

int activeGridWorkerCount(int requestedThreads,
                          std::size_t chunkCount,
                          std::size_t cellCount)
{
    int workerCount = activeChunkWorkerCount(requestedThreads, chunkCount);
    const std::uint64_t bytesPerWorker = static_cast<std::uint64_t>(cellCount) *
                                         static_cast<std::uint64_t>(sizeof(StreamingGridCell));
    const std::uint64_t localGridBudget = 1024ull * 1024ull * 1024ull;
    while (workerCount > 1 && bytesPerWorker * static_cast<std::uint64_t>(workerCount) > localGridBudget)
    {
        --workerCount;
    }
    return std::max(1, workerCount);
}

bool accumulateStreamingGridParallel(const std::string &cloudPath,
                                     const PlyHeader &header,
                                     const std::vector<PlyVertexChunk> &chunks,
                                     int requestedThreads,
                                     const detail::HeightGrid &grid,
                                     std::vector<StreamingGridCell> *cells,
                                     std::string *errorMsg)
{
    if (!cells)
    {
        return false;
    }
    if (chunks.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "PLY 顶点分块为空";
        }
        return false;
    }

    const std::size_t cellCount = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
    const int workerCount = activeGridWorkerCount(requestedThreads, chunks.size(), cellCount);
    std::vector<std::vector<StreamingGridCell>> localCells;
    try
    {
        localCells.resize(static_cast<std::size_t>(workerCount));
        for (std::vector<StreamingGridCell> &workerCells : localCells)
        {
            workerCells.assign(cellCount, StreamingGridCell{});
        }
    }
    catch (const std::bad_alloc &)
    {
        if (errorMsg)
        {
            *errorMsg = "并行流式地形网格缓冲区分配失败";
        }
        return false;
    }

    std::atomic<std::size_t> nextChunk{0};
    std::atomic<bool> failed{false};
    std::vector<std::string> errors(static_cast<std::size_t>(workerCount));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(workerCount));

    for (int worker = 0; worker < workerCount; ++worker)
    {
        workers.emplace_back([&, worker]() {
            std::ifstream file = xjw::common::io::openInputFile(cloudPath);
            if (!file)
            {
                errors[static_cast<std::size_t>(worker)] = "无法打开 PLY 文件";
                failed.store(true, std::memory_order_relaxed);
                return;
            }

            std::vector<char> buffer;
            std::vector<StreamingGridCell> &workerCells = localCells[static_cast<std::size_t>(worker)];
            while (!failed.load(std::memory_order_relaxed))
            {
                const std::size_t chunkIndex = nextChunk.fetch_add(1, std::memory_order_relaxed);
                if (chunkIndex >= chunks.size())
                {
                    break;
                }

                std::string chunkError;
                if (!readPlyChunkFromStream(file, header, chunks[chunkIndex], &buffer, &chunkError))
                {
                    errors[static_cast<std::size_t>(worker)] = chunkError.empty() ? "读取 PLY 顶点分块失败" : chunkError;
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }

                const char *data = buffer.data();
                const std::uint64_t stride = static_cast<std::uint64_t>(header.vertexStride);
                for (std::uint64_t i = 0; i < chunks[chunkIndex].vertexCount; ++i)
                {
                    const detail::PointXYZRGB point = readPointFromPlyRecord(data + static_cast<std::size_t>(i * stride),
                                                                             header);
                    std::size_t index = 0;
                    if (!pointToStreamingGridIndex(point, grid, &index) || index >= workerCells.size())
                    {
                        continue;
                    }
                    accumulateStreamingCell(point, &workerCells[index]);
                }
            }
        });
    }

    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    if (failed.load(std::memory_order_relaxed))
    {
        if (errorMsg)
        {
            *errorMsg = "并行构建流式地形网格失败";
            for (const std::string &error : errors)
            {
                if (!error.empty())
                {
                    *errorMsg = error;
                    break;
                }
            }
        }
        return false;
    }

    if (workerCount == 1)
    {
        *cells = std::move(localCells.front());
        return true;
    }

    cells->assign(cellCount, StreamingGridCell{});
    for (const std::vector<StreamingGridCell> &workerCells : localCells)
    {
        const std::size_t count = std::min(workerCells.size(), cells->size());
        for (std::size_t i = 0; i < count; ++i)
        {
            mergeStreamingCell(workerCells[i], &(*cells)[i]);
        }
    }
    return true;
}

float resolveStreamingCellMaxStdDev(const ReconstructionConfig &config,
                                     const detail::HeightGrid &grid)
{
    if (config.streamingCellMaxStdDev > 0.0f)
    {
        return config.streamingCellMaxStdDev;
    }

    const float xyStep = std::max(std::fabs(grid.stepX), std::fabs(grid.stepY));
    if (xyStep <= 0.0f || !std::isfinite(xyStep))
    {
        return 0.0f;
    }
    return std::max(1.0e-5f, xyStep * 1.5f);
}

void finalizeStreamingGrid(const std::vector<StreamingGridCell> &cells,
                           const ReconstructionConfig &config,
                           detail::HeightGrid *grid)
{
    if (!grid)
    {
        return;
    }
    const std::size_t count = std::min(cells.size(), grid->heights.size());
    for (std::size_t i = 0; i < count; ++i)
    {
        const StreamingGridCell &cell = cells[i];
        if (cell.weight <= 0.0)
        {
            continue;
        }
        const double mean = cell.zSum / cell.weight;
        const double variance = std::max(0.0, cell.zSumSq / cell.weight - mean * mean);
        const double stdDev = std::sqrt(variance);
        const float maxStdDev = resolveStreamingCellMaxStdDev(config, *grid);
        if (cell.count >= 4 && maxStdDev > 0.0f && stdDev > static_cast<double>(maxStdDev))
        {
            continue;
        }
        grid->heights[i] = static_cast<float>(mean);
        grid->sumW[i] = static_cast<float>(cell.weight);
        grid->valid[i] = 1;
    }
}

std::uint8_t averagedColor(double value, std::uint32_t count)
{
    if (count == 0)
    {
        return 190;
    }
    return static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(value / count)), 0, 255));
}

void streamingHeightGridToMesh(const detail::HeightGrid &grid,
                               const std::vector<StreamingGridCell> &cells,
                               TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }
    mesh->vertices.clear();
    mesh->faces.clear();
    if (grid.nx < 2 || grid.ny < 2 || grid.heights.empty())
    {
        return;
    }

    std::vector<int> vertexIndex(grid.heights.size(), -1);
    mesh->vertices.reserve(grid.heights.size());
    for (int iy = 0; iy < grid.ny; ++iy)
    {
        for (int ix = 0; ix < grid.nx; ++ix)
        {
            const std::size_t idx = gridIndex(grid, ix, iy);
            if (!grid.valid[idx])
            {
                continue;
            }
            const StreamingGridCell &cell = idx < cells.size() ? cells[idx] : StreamingGridCell{};
            MeshVertex vertex;
            vertex.x = grid.minX + static_cast<float>(ix) * grid.stepX;
            vertex.y = grid.minY + static_cast<float>(iy) * grid.stepY;
            vertex.z = grid.heights[idx];
            vertex.r = averagedColor(cell.rSum, cell.count);
            vertex.g = averagedColor(cell.gSum, cell.count);
            vertex.b = averagedColor(cell.bSum, cell.count);
            vertex.nx = 0.0f;
            vertex.ny = 0.0f;
            vertex.nz = 1.0f;
            vertexIndex[idx] = static_cast<int>(mesh->vertices.size());
            mesh->vertices.push_back(vertex);
        }
    }

    mesh->faces.reserve(static_cast<std::size_t>(std::max(0, grid.nx - 1)) *
                        static_cast<std::size_t>(std::max(0, grid.ny - 1)) * 2);
    auto addFace = [&](int a, int b, int c) {
        if (a < 0 || b < 0 || c < 0)
        {
            return;
        }
        Triangle tri;
        tri.v[0] = a;
        tri.v[1] = b;
        tri.v[2] = c;
        mesh->faces.push_back(tri);
    };

    for (int iy = 0; iy + 1 < grid.ny; ++iy)
    {
        for (int ix = 0; ix + 1 < grid.nx; ++ix)
        {
            const int v00 = vertexIndex[gridIndex(grid, ix, iy)];
            const int v10 = vertexIndex[gridIndex(grid, ix + 1, iy)];
            const int v01 = vertexIndex[gridIndex(grid, ix, iy + 1)];
            const int v11 = vertexIndex[gridIndex(grid, ix + 1, iy + 1)];
            addFace(v00, v10, v11);
            addFace(v00, v11, v01);
        }
    }
}

void appendMesh(const TriMesh &src,
                TriMesh *dst)
{
    if (!dst || src.empty())
    {
        return;
    }

    const int vertexOffset = static_cast<int>(dst->vertices.size());
    dst->vertices.insert(dst->vertices.end(), src.vertices.begin(), src.vertices.end());
    dst->faces.reserve(dst->faces.size() + src.faces.size());
    for (const Triangle &face : src.faces)
    {
        Triangle out = face;
        out.v[0] += vertexOffset;
        out.v[1] += vertexOffset;
        out.v[2] += vertexOffset;
        dst->faces.push_back(out);
    }
}

bool reconstructStreamingTileFromPly(const std::string &cloudPath,
                                     const PlyHeader &header,
                                     const std::vector<PlyVertexChunk> &chunks,
                                     int requestedThreads,
                                     const StreamingTile &tile,
                                     const ReconstructionConfig &config,
                                     TriMesh *mesh,
                                     std::string *errorMsg)
{
    if (!mesh)
    {
        return false;
    }

    detail::HeightGrid grid = makeEmptyStreamingHeightGrid(tile.bounds, config);
    std::vector<StreamingGridCell> cells;
    const bool ok = accumulateStreamingGridParallel(cloudPath,
                                                    header,
                                                    chunks,
                                                    requestedThreads,
                                                    grid,
                                                    &cells,
                                                    errorMsg);
    if (!ok)
    {
        return false;
    }

    finalizeStreamingGrid(cells, config, &grid);
    if (config.fillHoles)
    {
        detail::fillHoles(&grid, std::max(0, config.holeFillPasses));
    }

    streamingHeightGridToMesh(grid, cells, mesh);
    return !mesh->empty();
}

bool reconstructStreamingHeightGridFromPly(const std::string &cloudPath,
                                           const PlyHeader &header,
                                           const ReconstructionConfig &config,
                                           TriMesh *mesh,
                                           std::string *errorMsg,
                                           const std::function<void(const std::string &, float)> &progress)
{
    if (!mesh)
    {
        return false;
    }

    const int requestedThreads = resolveStreamingThreadCount(config);
    const std::vector<PlyVertexChunk> chunks = makePlyVertexChunks(header, config);
    const int scanThreads = activeChunkWorkerCount(requestedThreads, chunks.size());
    if (progress)
    {
        progress("正在并行分块扫描流式地形网格范围 (" + std::to_string(header.vertexCount) + " 点, " +
                     std::to_string(scanThreads) + " 线程, " + std::to_string(chunks.size()) + " 块)...",
                 0.04f);
    }
    StreamingBounds bounds;
    if (!estimateStreamingBoundsParallel(cloudPath, header, chunks, requestedThreads, &bounds, errorMsg))
    {
        return false;
    }

    const std::vector<StreamingTile> tiles = makeStreamingTiles(bounds, header, config);
    if (tiles.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "流式地形网格瓦片规划失败";
        }
        return false;
    }
    if (progress)
    {
        progress("正在规划流式地形网格瓦片 (" + std::to_string(tiles.front().nx) + "x" +
                     std::to_string(tiles.front().ny) + " 瓦片)...",
                 0.16f);
    }

    mesh->vertices.clear();
    mesh->faces.clear();
    const ReconstructionConfig tileConfig = makeStreamingTileConfig(config, tiles.front().nx);
    for (std::size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex)
    {
        const StreamingTile &tile = tiles[tileIndex];
        if (progress)
        {
            progress("正在并行分块构建流式地形网格瓦片 " + std::to_string(tileIndex + 1) + "/" +
                         std::to_string(tiles.size()) + " (" + std::to_string(tile.x + 1) + "," +
                         std::to_string(tile.y + 1) + ")...",
                     0.18f + 0.34f * static_cast<float>(tileIndex) / static_cast<float>(std::max<std::size_t>(1, tiles.size())));
        }

        TriMesh tileMesh;
        std::string tileError;
        if (!reconstructStreamingTileFromPly(cloudPath,
                                             header,
                                             chunks,
                                             requestedThreads,
                                             tile,
                                             tileConfig,
                                             &tileMesh,
                                             &tileError))
        {
            continue;
        }
        appendMesh(tileMesh, mesh);
    }

    if (mesh->empty())
    {
        if (errorMsg)
        {
            *errorMsg = "流式地形网格瓦片三角化失败（结果为空）";
        }
        return false;
    }

    if (progress)
    {
        progress("流式地形网格瓦片合并完成 (" + std::to_string(mesh->vertexCount()) + " 顶点, " +
                     std::to_string(mesh->faceCount()) + " 面)...",
                 0.56f);
    }
    return true;
}

std::vector<detail::PointXYZRGB> cloudToPointXYZRGB(const PlaPointCloud &cloud)
{
    std::vector<detail::PointXYZRGB> points;
    points.reserve(static_cast<std::size_t>(cloud.size()));
    const bool hasColors = cloud.hasColors();
    const bool hasNormals = cloud.hasNormals();
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        auto pt = cloud[i];
        detail::PointXYZRGB p;
        p.x = pt.x();
        p.y = pt.y();
        p.z = pt.z();
        if (hasNormals)
        {
            p.hasNormal = true;
            p.nx = pt.nx();
            p.ny = pt.ny();
            p.nz = pt.nz();
        }
        if (hasColors)
        {
            p.r = pt.r();
            p.g = pt.g();
            p.b = pt.b();
        }
        points.push_back(p);
    }
    return points;
}

PlaPointCloud pointXYZRGBToCloud(const std::vector<detail::PointXYZRGB> &points)
{
    const auto n = static_cast<plamatrix::Index>(points.size());
    const bool hasNormals = !points.empty() &&
                            std::all_of(points.begin(), points.end(), [](const detail::PointXYZRGB &point) {
                                return point.hasNormal;
                            });
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(n, 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(n, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(hasNormals ? n : 0, 3);
    for (plamatrix::Index i = 0; i < n; ++i)
    {
        pts(i, 0) = points[static_cast<std::size_t>(i)].x;
        pts(i, 1) = points[static_cast<std::size_t>(i)].y;
        pts(i, 2) = points[static_cast<std::size_t>(i)].z;
        colors(i, 0) = points[static_cast<std::size_t>(i)].r;
        colors(i, 1) = points[static_cast<std::size_t>(i)].g;
        colors(i, 2) = points[static_cast<std::size_t>(i)].b;
        if (hasNormals)
        {
            normals(i, 0) = points[static_cast<std::size_t>(i)].nx;
            normals(i, 1) = points[static_cast<std::size_t>(i)].ny;
            normals(i, 2) = points[static_cast<std::size_t>(i)].nz;
        }
    }
    PlaPointCloud cloud(std::move(pts));
    cloud.setColors(std::move(colors));
    if (hasNormals)
    {
        cloud.setNormals(std::move(normals));
    }
    return cloud;
}

std::vector<detail::PointXYZRGB> loadPointsForMeshing(const std::string &cloudPath,
                                                      const ReconstructionConfig &config,
                                                      std::string *errorMsg,
                                                      const std::function<void(const std::string &, float)> &progress)
{
    PlyHeader header;
    std::string headerError;
    const bool hasHeader = parseBinaryPlyHeader(cloudPath, &header, &headerError);
    const int maxInputPoints = std::max(0, config.maxInputPointsForMeshing);
    if (hasHeader && maxInputPoints > 0 &&
        header.vertexCount > static_cast<std::uint64_t>(maxInputPoints))
    {
        if (progress)
        {
            progress("点云过大，流式抽样到最多 " + std::to_string(maxInputPoints) + " 点...", 0.04f);
        }
        return sampleBinaryPlyForMeshing(cloudPath, header, maxInputPoints, errorMsg);
    }
    if (!hasHeader && maxInputPoints > 0 && header.vertexCount > static_cast<std::uint64_t>(maxInputPoints))
    {
        if (errorMsg)
        {
            *errorMsg = "点云超过网格重建输入上限，但当前 PLY 格式暂不支持流式抽样: " + headerError;
        }
        return {};
    }

    auto cloudPtr = plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(cloudPath));
    if (!cloudPtr)
    {
        if (errorMsg)
        {
            *errorMsg = hasHeader ? "无法加载点云文件" : ("无法解析 PLY 头: " + headerError);
        }
        return {};
    }
    return cloudToPointXYZRGB(*cloudPtr);
}

bool convertPoissonResultToMesh(const plamatrix::DenseMatrix<float, plamatrix::Device::CPU> &verts,
                                 const plamatrix::DenseMatrix<float, plamatrix::Device::CPU> &faces,
                                 TriMesh *mesh)
{
    if (!mesh)
    {
        return false;
    }

    mesh->vertices.clear();
    mesh->faces.clear();

    const int vertexCount = static_cast<int>(verts.rows());
    const int faceCount = static_cast<int>(faces.rows());

    mesh->vertices.reserve(static_cast<std::size_t>(vertexCount));
    for (int i = 0; i < vertexCount; ++i)
    {
        MeshVertex v;
        v.x = verts(static_cast<plamatrix::Index>(i), 0);
        v.y = verts(static_cast<plamatrix::Index>(i), 1);
        v.z = verts(static_cast<plamatrix::Index>(i), 2);
        v.nx = 0.0f;
        v.ny = 0.0f;
        v.nz = 1.0f;
        mesh->vertices.push_back(v);
    }

    mesh->faces.reserve(static_cast<std::size_t>(faceCount));
    for (int i = 0; i < faceCount; ++i)
    {
        Triangle t;
        t.v[0] = static_cast<int>(faces(static_cast<plamatrix::Index>(i), 0));
        t.v[1] = static_cast<int>(faces(static_cast<plamatrix::Index>(i), 1));
        t.v[2] = static_cast<int>(faces(static_cast<plamatrix::Index>(i), 2));
        mesh->faces.push_back(t);
    }

    return true;
}

} // namespace

bool SurfaceReconstructor::reconstructFromPointCloudFile(const std::string &cloudPath,
                                                         const ReconstructionConfig &config,
                                                         TriMesh &outMesh,
                                                         std::string *errorMsg,
                                                         std::string *algorithmUsed)
{
    if (algorithmUsed)
    {
        algorithmUsed->clear();
    }

    auto progress = [&](const std::string &stage, float p) {
        if (config.progressFn)
        {
            config.progressFn(stage, p);
        }
    };

    TriMesh mesh;
    bool usedHeightGridFallback = false;
    bool usedLateHeightGridFallback = false;
    std::string selectedAlgorithm;
    bool streamingMeshBuilt = false;

    progress("正在检查点云规模...", 0.02f);
    PlyHeader streamingHeader;
    std::string streamingHeaderError;
    const bool hasStreamingHeader = parseBinaryPlyHeader(cloudPath, &streamingHeader, &streamingHeaderError);
    const int maxInputPoints = std::max(0, config.maxInputPointsForMeshing);
    if (hasStreamingHeader && maxInputPoints > 0 &&
        streamingHeader.vertexCount > static_cast<std::uint64_t>(maxInputPoints))
    {
        progress("检测到超大密集点云(" + std::to_string(streamingHeader.vertexCount) +
                     " > " + std::to_string(maxInputPoints) +
                     ")，自动切换为流式地形网格...",
                 0.03f);
        if (!reconstructStreamingHeightGridFromPly(cloudPath,
                                                   streamingHeader,
                                                   config,
                                                   &mesh,
                                                   errorMsg,
                                                   progress))
        {
            return false;
        }
        selectedAlgorithm = "streaming_tiled_height_grid";
        usedHeightGridFallback = true;
        streamingMeshBuilt = true;
    }
    else if (!hasStreamingHeader && maxInputPoints > 0 &&
             streamingHeader.vertexCount > static_cast<std::uint64_t>(maxInputPoints))
    {
        if (errorMsg)
        {
            *errorMsg = "点云超过网格重建输入上限，但当前 PLY 格式暂不支持流式地形网格: " + streamingHeaderError;
        }
        return false;
    }

    std::vector<detail::PointXYZRGB> points;
    auto buildHeightGridMesh = [&](float buildProgress, float fillProgress, float triangulateProgress) {
        usedHeightGridFallback = true;
        selectedAlgorithm = "height_grid";
        progress("正在构建高度格网...", buildProgress);
        detail::HeightGrid hg = detail::buildHeightGrid(points, config);
        if (hg.nx < 4 || hg.ny < 4)
        {
            if (errorMsg)
            {
                *errorMsg = "高程格网构建失败（点云范围过小）";
            }
            return false;
        }

        if (config.fillHoles)
        {
            progress("正在填充空洞...", fillProgress);
            detail::fillHoles(&hg, std::max(0, config.holeFillPasses));
        }

        progress("正在三角分割...", triangulateProgress);
        detail::heightGridToMesh(hg, points, config, &mesh);
        return !mesh.empty();
    };
    auto minUsefulPoissonFaces = [&]() {
        return std::max(16, std::min(std::max(2, config.minComponentFaces), 256));
    };

    if (!streamingMeshBuilt)
    {
        progress("正在加载点云...", 0.04f);
        std::string loadError;
        points = loadPointsForMeshing(cloudPath, config, &loadError, progress);
        if (points.size() < 100)
        {
            if (errorMsg)
            {
                *errorMsg = points.empty() && !loadError.empty()
                    ? loadError
                    : "点云点数过少，无法稳定重建";
            }
            return false;
        }

        const float baseVoxel = detail::estimateBaseVoxelStep(points, config.resolution);
        if (config.enableDenoise)
        {
            progress("正在点云去噪...", 0.08f);
            points = detail::statisticalDenoisePoints(points,
                                                       std::clamp(config.denoiseK, 8, 64),
                                                       std::clamp(config.denoiseStdMul, 0.6f, 3.0f),
                                                       baseVoxel * 2.0f,
                                                       config.preprocessingDevice);
        }

        if (config.enableDownsample)
        {
            progress("正在点云下采样...", 0.12f);
            const float voxelSize = baseVoxel * std::clamp(config.downsampleVoxelScale, 0.4f, 2.5f);
            points = detail::voxelDownsamplePoints(points, voxelSize, config.preprocessingDevice);
        }

        if (points.size() < 120)
        {
            if (errorMsg)
            {
                *errorMsg = "点云预处理后点数不足，无法稳定网格化";
            }
            return false;
        }

        if (config.forcePoisson)
        {
            progress("正在执行 Poisson 重建...", 0.30f);
            try
            {
                PlaPointCloud poissonCloud = pointXYZRGBToCloud(points);
                auto poissonCloudPtr = std::make_shared<const PlaPointCloud>(std::move(poissonCloud));
                plapoint::mesh::PoissonReconstruction<float> poisson;
                poisson.setInputCloud(poissonCloudPtr);
                poisson.setDepth(std::clamp(config.poissonDepth, 1, 8));
                auto [verts, faces] = poisson.reconstruct();
                convertPoissonResultToMesh(verts, faces, &mesh);
            }
            catch (const std::exception &e)
            {
                mesh = TriMesh{};
                const std::string reason = std::string(e.what()).empty() ? "unknown error" : e.what();
                progress("Poisson 重建失败(" + reason + ")，改用高度格网...", 0.34f);
            }

            if (!mesh.empty())
            {
                selectedAlgorithm = "poisson";

                // Light simplification for Poisson output if extremely dense
                const int poissonFaceCount = mesh.faceCount();
                const int hardUpperBound = std::max(120000, std::max(1, config.simplifyTargetFaces) * 4);
                if (poissonFaceCount > hardUpperBound)
                {
                    progress("正在轻量简化 Poisson 网格...", 0.62f);
                    ReconstructionConfig liteConfig = config;
                    liteConfig.simplifyTargetFaces = std::max(config.simplifyTargetFaces * 2, 80000);
                    liteConfig.voxelSimplifyFactor = std::clamp(config.voxelSimplifyFactor * 0.85f, 1.0f, 2.4f);
                    detail::simplifyVoxelMeshAdaptive(&mesh, liteConfig, baseVoxel);
                }
            }
        }

        if (mesh.empty())
        {
            if (!buildHeightGridMesh(0.10f, 0.30f, 0.50f))
            {
                if (errorMsg && errorMsg->empty())
                {
                    *errorMsg = "高度格网重建失败";
                }
                return false;
            }
        }
    }

    if (mesh.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "网格化失败（结果为空）";
        }
        return false;
    }

    progress("正在清理退化面...", 0.64f);
    detail::removeDegenerateFaces(&mesh);

    if (config.cleanSmallComponents)
    {
        progress("正在清理碎片连通体...", 0.70f);
        detail::removeSmallConnectedComponents(&mesh, std::max(2, config.minComponentFaces));
        if (selectedAlgorithm == "poisson" && mesh.faceCount() < minUsefulPoissonFaces())
        {
            progress("Poisson 网格主体过小，改用高度格网...", 0.72f);
            mesh = TriMesh{};
            usedLateHeightGridFallback = true;
            if (!buildHeightGridMesh(0.72f, 0.74f, 0.76f))
            {
                if (errorMsg && errorMsg->empty())
                {
                    *errorMsg = "Poisson 网格主体过小，且高度格网重建失败";
                }
                return false;
            }
            detail::removeDegenerateFaces(&mesh);
            detail::removeSmallConnectedComponents(&mesh, std::max(2, config.minComponentFaces));
        }
        if (mesh.empty())
        {
            if (errorMsg)
            {
                *errorMsg = "网格清理后为空，请降低连通体阈值或提高分辨率";
            }
            return false;
        }
    }

    progress("正在平滑网格...", usedLateHeightGridFallback ? 0.80f : 0.75f);
    int smoothIters = config.smoothIterations;
    float smoothLambda = config.smoothLambda;
    if (!usedHeightGridFallback)
    {
        smoothIters = std::max(0, config.smoothIterations - 1);
        smoothLambda = std::clamp(config.smoothLambda * 0.58f, 0.08f, 0.32f);
    }
    detail::taubinSmooth(&mesh, smoothIters, smoothLambda);

    progress("正在重算法线...", 0.90f);
    detail::recomputeNormals(&mesh);

    progress("网格重建完成", 1.0f);
    if (algorithmUsed)
    {
        *algorithmUsed = selectedAlgorithm.empty() ? "unknown" : selectedAlgorithm;
    }
    outMesh = std::move(mesh);
    return true;
}

} // namespace mesh
} // namespace xjw
