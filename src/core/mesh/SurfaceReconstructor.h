#pragma once
// =============================================================================
// 文件: SurfaceReconstructor.h
// 模块: Mesh - 点云表面重建（隐式场 + Marching Tetrahedra）
// =============================================================================

#include "MeshTypes.h"

#include <string>

namespace xjw {
namespace mesh {

class SurfaceReconstructor {
public:
    // 从点云文件（.ply/.xyz/.txt）重建三角网格
    static bool reconstructFromPointCloudFile(const std::string &cloudPath,
                                              const ReconstructionConfig &config,
                                              TriMesh &outMesh,
                                              std::string *errorMsg = nullptr,
                                              std::string *algorithmUsed = nullptr);
};

} // namespace mesh
} // namespace xjw
