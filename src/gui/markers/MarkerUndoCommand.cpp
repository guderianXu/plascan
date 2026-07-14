#include "MarkerUndoCommand.h"

#include "ProjectMarkerRepository.h"

namespace xjw::gui::markers
{

MarkerUndoCommand::MarkerUndoCommand(ProjectMarkerRepository *repository,
                                     control_points::MarkerChangeSet changeSet,
                                     QUndoCommand *parent)
    : QUndoCommand(changeSet.description(), parent)
    , _repository(repository)
    , _changeSet(std::move(changeSet))
{
}

void MarkerUndoCommand::redo()
{
    if (_repository) _repository->applyChangeSet(_changeSet);
}

void MarkerUndoCommand::undo()
{
    if (_repository) _repository->revertChangeSet(_changeSet);
}

} // namespace xjw::gui::markers
