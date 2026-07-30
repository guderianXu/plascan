#pragma once

#include <QJsonObject>

namespace xjw::common::json
{

/**
 * @brief Recursively merges object fields from @p patch into @p base.
 *
 * Object values are merged recursively. All other values, including arrays,
 * replace the corresponding value in @p base.
 */
QJsonObject deepMergeObjects(const QJsonObject &base, const QJsonObject &patch);

} // namespace xjw::common::json
