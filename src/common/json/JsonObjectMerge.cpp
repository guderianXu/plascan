#include "json/JsonObjectMerge.h"

namespace xjw::common::json
{

QJsonObject deepMergeObjects(const QJsonObject &base, const QJsonObject &patch)
{
    QJsonObject merged = base;
    for (auto it = patch.constBegin(); it != patch.constEnd(); ++it)
    {
        const QJsonValue existing = merged.value(it.key());
        if (existing.isObject() && it.value().isObject())
        {
            merged.insert(it.key(), deepMergeObjects(existing.toObject(), it.value().toObject()));
            continue;
        }

        merged.insert(it.key(), it.value());
    }

    return merged;
}

} // namespace xjw::common::json
