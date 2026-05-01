#pragma once

#include <QString>

namespace xjw::common
{

struct OperationResult
{
    bool ok = false;
    QString errorMessage;
};

} // namespace xjw::common
