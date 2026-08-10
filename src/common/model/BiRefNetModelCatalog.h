#pragma once

#include "model/ModelFileResolver.h"

#include <QString>
#include <QStringList>

namespace xjw::common::model
{

struct BiRefNetModelSpec
{
    QString token;
    QString displayName;
    QStringList modelFileNames;
    QString provenanceFileName;
};

struct BiRefNetModelStatus
{
    BiRefNetModelSpec spec;
    QString modelPath;
    QString provenancePath;
    QStringList missingFiles;
    QString label;
    QString detail;
    bool isInstalled = false;
};

BiRefNetModelSpec biRefNetDynamicModelSpec();
BiRefNetModelStatus biRefNetDynamicModelStatus(const ModelFileResolver& resolver);

} // namespace xjw::common::model
