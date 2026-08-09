#pragma once

#include "model/CameraReferenceSet.h"

#include <QString>

namespace xjw::camera_reference
{

struct CameraReferenceSetIoResult
{
    bool ok = false;
    CameraReferenceSet referenceSet;
    QString error;
};

class CameraReferenceSetStore
{
public:
    explicit CameraReferenceSetStore(QString path);

    CameraReferenceSetIoResult load() const;
    CameraReferenceSetIoResult save(const CameraReferenceSet &referenceSet) const;
    QString path() const;

private:
    QString _path;
};

} // namespace xjw::camera_reference
