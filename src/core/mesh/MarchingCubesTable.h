#pragma once
// =============================================================================
// 文件: MarchingCubesTable.h
// 模块: Mesh - Marching Cubes 查找表声明
// =============================================================================

#include <cstdint>

namespace xjw {
namespace mesh {

extern const uint16_t MC_EDGE_TABLE[256];
extern const int8_t MC_TRI_TABLE[256][16];
extern const int8_t MC_EDGE_VERTEX[12][2];

} // namespace mesh
} // namespace xjw
