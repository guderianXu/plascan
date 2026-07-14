#pragma once

#include "commands/MarkerChangeSet.h"

#include <QUndoCommand>

namespace xjw::gui::markers
{

class ProjectMarkerRepository;

class MarkerUndoCommand final : public QUndoCommand
{
public:
    MarkerUndoCommand(ProjectMarkerRepository *repository,
                      control_points::MarkerChangeSet changeSet,
                      QUndoCommand *parent = nullptr);

    void redo() override;
    void undo() override;

private:
    ProjectMarkerRepository *_repository = nullptr;
    control_points::MarkerChangeSet _changeSet;
};

} // namespace xjw::gui::markers
