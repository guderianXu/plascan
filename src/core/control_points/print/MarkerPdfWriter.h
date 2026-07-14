#pragma once

#include "MarkerSheetRenderer.h"

namespace xjw::control_points
{

struct MarkerPdfWriteResult
{
    bool ok = false;
    QString error;
    QString outputPath;
    int pageCount = 0;
};

class MarkerPdfWriter final
{
public:
    static MarkerPdfWriteResult write(const MarkerPrintRequest &request,
                                      const QString &outputPath,
                                      int dpi = 300);
};

} // namespace xjw::control_points
