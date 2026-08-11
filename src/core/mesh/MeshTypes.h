#pragma once
// =============================================================================
// 文件: MeshTypes.h
// 模块: Mesh - 三角网格基础类型
// =============================================================================

#include <functional>
#include <vector>
#include <string>
#include <cstdint>

#include <plapoint/filters/preprocessing.h>

namespace xjw 
{
namespace mesh 
{

struct MeshVertex 
{
    float x = 0, y = 0, z = 0;
    float nx = 0, ny = 0, nz = 0;
    uint8_t r = 200, g = 200, b = 200;
};

struct Triangle 
{
    int v[3] = {0, 0, 0};
};

struct TriMesh {
    std::vector<MeshVertex> vertices;
    std::vector<Triangle>   faces;
    bool hasVertexColors = true;

    bool empty() const { return vertices.empty() || faces.empty(); }
    int vertexCount() const { return static_cast<int>(vertices.size()); }
    int faceCount()   const { return static_cast<int>(faces.size()); }

    bool savePLY(const std::string &path, std::string *errorMsg = nullptr) const;
    bool saveOBJ(const std::string &path, std::string *errorMsg = nullptr) const;
    static bool loadPLY(const std::string &path,
                        TriMesh *mesh,
                        std::string *errorMsg = nullptr);
};

struct ReconstructionConfig 
{
    int   resolution       = 128;
    bool  enableDenoise    = true;
    int   denoiseK         = 20;
    float denoiseStdMul    = 1.1f;
    bool  enableDownsample = true;
    float downsampleVoxelScale = 0.8f;
    plapoint::ProcessingDevice preprocessingDevice = plapoint::ProcessingDevice::Auto;
    // Keep the Poisson CUDA/OpenCL/CPU solver selection independent from preprocessing.
    plapoint::ProcessingDevice poissonSolverDevice = plapoint::ProcessingDevice::Auto;
    int   poissonSolverIterations = 200;
    bool  forcePoisson     = true;
    bool  allowHeightGridFallback = true;
    bool  orientNormalsForClosedSurface = false;
    int   poissonDepth     = 9;
    float poissonPointWeight = 4.0f;
    float poissonTrim      = 9.5f;
    int   poissonThreads   = -1;
    int   simplifyTargetFaces = 18000;
    int   maxInputPointsForMeshing = 30000000;
    int   streamingThreads = 0;
    int   streamingChunkBytes = 64 * 1024 * 1024;
    int   streamingTileCount = 0;
    int   streamingTileOverlapCells = 2;
    float streamingCellMaxStdDev = 0.0f; // <=0 表示按网格步长自动判断竖向毛刺
    int   kNormals         = 12;
    int   kImplicit        = 10;
    float padding          = 0.05f;
    int   smoothIterations = 2;
    float smoothLambda     = 0.5f;
    bool  fillHoles        = true;
    int   holeFillPasses   = 8;
    bool  cleanSmallComponents = true;
    int   minComponentFaces = 100;
    float voxelSimplifyFactor = 1.8f;
    bool  verbose          = true;

    // 取消回调：在加载、预处理、重建和后处理阶段边界查询。
    std::function<bool()> isCancelled;
    // 进度回调：参数为 (阶段描述, 0.0~1.0 进度)
    std::function<void(const std::string &stage, float progress)> progressFn;
};

} // namespace mesh
} // namespace xjw
