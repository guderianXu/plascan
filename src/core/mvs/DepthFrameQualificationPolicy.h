#pragma once

#include <algorithm>

#include <QString>

namespace xjw::mvs
{

inline bool isPrimaryFusionFrame(const QString &acceptance,
                                 bool fusion_eligible) noexcept
{
    return fusion_eligible &&
        acceptance.trimmed().compare(
            QStringLiteral("accepted"), Qt::CaseInsensitive) == 0;
}

inline int minimumOrbitalPrimaryDepthFrameCount(
    int discovered_frame_count) noexcept
{
    const int normalized_frame_count = std::max(0, discovered_frame_count);
    const int quarter_coverage = normalized_frame_count / 4 +
        (normalized_frame_count % 4 == 0 ? 0 : 1);
    return std::max(3, quarter_coverage);
}

} // namespace xjw::mvs
