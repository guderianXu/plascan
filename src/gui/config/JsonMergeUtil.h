#pragma once

#include <QJsonObject>

class JsonMergeUtil
{
public:
    static QJsonObject deepMerge(const QJsonObject &base, const QJsonObject &patch);
};
