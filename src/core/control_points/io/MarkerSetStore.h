#pragma once

#include "model/MarkerSet.h"

#include <QString>

namespace xjw::control_points
{

struct MarkerSetIoResult
{
    bool ok = false;
    MarkerSet markerSet;
    QString error;
    QString backupPath;
};

class MarkerSetStore
{
public:
    explicit MarkerSetStore(QString path);

    MarkerSetIoResult load() const;
    MarkerSetIoResult save(const MarkerSet &markerSet) const;
    QString path() const;

private:
    QString _path;
};

} // namespace xjw::control_points
