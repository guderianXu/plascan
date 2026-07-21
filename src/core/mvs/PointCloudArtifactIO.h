#pragma once

#include "DenseCloudQualityFilter.h"

#include <QString>

namespace xjw::mvs
{

bool writeDensePointCloudPly(const QString &path,
                             const DensePointCloud &pointCloud,
                             bool writeNormals,
                             QString *errorMessage);

} // namespace xjw::mvs
