#include "JsonMergeUtil.h"

QJsonObject JsonMergeUtil::deepMerge(const QJsonObject &base, const QJsonObject &patch)
{
    QJsonObject merged = base;
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        if (merged.value(it.key()).isObject() && it.value().isObject()) {
            merged[it.key()] = deepMerge(merged.value(it.key()).toObject(), it.value().toObject());
        } else {
            merged[it.key()] = it.value();
        }
    }
    return merged;
}
