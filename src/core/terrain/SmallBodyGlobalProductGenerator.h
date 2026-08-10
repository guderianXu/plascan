#pragma once

#include "SmallBodyGlobalProducts.h"

namespace xjw
{

class SmallBodyGlobalProductGenerator
{
public:
    static bool generate(const QString &surfacePath,
                         const QString &outputDirectory,
                         const SmallBodyGlobalOptions &options,
                         SmallBodyGlobalProducts *products,
                         QString *errorMessage = nullptr,
                         const std::atomic_bool *cancelFlag = nullptr,
                         const SmallBodyProgressCallback &progressCallback = {});

    static bool generateFromMesh(const TerrainMeshInput &surface,
                                 const QString &sourceLabel,
                                 const QString &outputDirectory,
                                 const SmallBodyGlobalOptions &options,
                                 SmallBodyGlobalProducts *products,
                                 QString *errorMessage = nullptr,
                                 const std::atomic_bool *cancelFlag = nullptr,
                                 const SmallBodyProgressCallback &progressCallback = {});
};

} // namespace xjw
