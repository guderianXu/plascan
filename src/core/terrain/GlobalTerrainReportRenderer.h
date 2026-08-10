#pragma once

#include "SmallBodyGlobalProducts.h"

#include <QString>

namespace xjw
{

class GlobalTerrainReportRenderer
{
public:
    static bool writePreview(const SmallBodyGlobalProducts &products,
                             const SmallBodyGlobalOptions &options,
                             const QString &outputPath,
                             QString *errorMessage = nullptr);
};

} // namespace xjw
