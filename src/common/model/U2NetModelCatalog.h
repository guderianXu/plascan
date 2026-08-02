#pragma once

#include "model/ModelFileResolver.h"

#include <QString>
#include <QStringList>

namespace xjw::common::model
{

struct U2NetModelSpec
{
    QString token;
    QString displayName;
    QStringList modelFileNames;
};

struct U2NetModelStatus
{
    U2NetModelSpec spec;
    QString modelPath;
    QStringList missingFiles;
    QString label;
    QString detail;
    bool isInstalled = false;
};

U2NetModelSpec u2netModelSpec();
U2NetModelStatus u2netModelStatus(const ModelFileResolver &resolver);

} // namespace xjw::common::model
