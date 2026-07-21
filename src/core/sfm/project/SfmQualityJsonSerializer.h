#pragma once

#include "quality/SfmQualityMetrics.h"

#include <QJsonObject>

namespace xjw
{

QJsonObject serializeSfmQualityMetrics(const SfmQualityMetrics &metrics);

} // namespace xjw
