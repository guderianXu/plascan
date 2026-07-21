#pragma once

#include <QString>
#include <QStringList>

namespace xjw::common::model
{

QStringList featureExtractorModelCandidates(const QString &algorithm, bool useCuda);

bool isManagedFeatureExtractorModelPath(const QString &path);

} // namespace xjw::common::model
