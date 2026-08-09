#pragma once

#include "model/CameraReferenceSet.h"

#include <QJsonObject>
#include <QStandardItemModel>

namespace xjw::gui::reference
{

enum class ReferenceDisplayMode
{
    Source,
    Estimated,
    Error
};

class CameraReferenceTreeModel final : public QStandardItemModel
{
public:
    enum Column
    {
        LabelColumn = 0,
        XColumn,
        YColumn,
        ZColumn,
        YawColumn,
        PitchColumn,
        RollColumn,
        HorizontalAccuracyColumn,
        VerticalAccuracyColumn,
        StatusColumn,
        EnabledColumn,
        ColumnCount
    };

    enum DataRole
    {
        NodeTypeRole = Qt::UserRole + 1,
        ImageUuidRole,
        ImagePathRole
    };

    enum class NodeType
    {
        TotalError,
        Camera,
        UnmatchedGroup,
        UnmatchedRecord
    };

    explicit CameraReferenceTreeModel(QObject *parent = nullptr);

    void setReferenceData(const camera_reference::CameraReferenceSet &referenceSet,
                          const QJsonObject &projectMetadata,
                          ReferenceDisplayMode mode);
};

} // namespace xjw::gui::reference
