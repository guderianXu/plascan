#include "MarchingCubesTable.h"

namespace xjw {
namespace mesh {

const uint16_t MC_EDGE_TABLE[256] = {
    0
};

const int8_t MC_TRI_TABLE[256][16] = {{
    -1
}};

const int8_t MC_EDGE_VERTEX[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

} // namespace mesh
} // namespace xjw
