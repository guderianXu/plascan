#pragma once

#include <QIcon>

namespace xjw::gui::widgets
{

enum class WorkspaceSection
{
    Photos,
    ObservationNetwork,
    TiePoints,
    DepthMaps,
    DenseCloud,
    Model3D,
    Dem,
    Orthomosaic,
    ReferenceData,
    Reports,
    Unknown
};

// QPixmap-backed icons must be created and consumed on the GUI thread.
QIcon workspaceSectionIcon(WorkspaceSection section);

} // namespace xjw::gui::widgets
