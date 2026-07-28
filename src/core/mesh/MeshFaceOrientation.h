#pragma once

#include "MeshTypes.h"

namespace xjw::mesh
{

struct MeshFaceOrientationStatistics
{
    int sharedEdgeCount = 0;
    int inconsistentSharedEdgeCountBefore = 0;
    int inconsistentSharedEdgeCountAfter = 0;
    int flippedFaceCount = 0;
    int removedContradictoryFaceCount = 0;
    int nonManifoldEdgeCount = 0;
    int orientationConflictCount = 0;
    bool succeeded = false;
};

MeshFaceOrientationStatistics repairMeshFaceOrientation(TriMesh *mesh);

} // namespace xjw::mesh
