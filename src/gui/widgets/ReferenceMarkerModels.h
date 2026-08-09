#pragma once

#include "model/MarkerSet.h"

#include <QStandardItemModel>

namespace xjw::gui::reference
{

class MarkerReferenceTreeModel final : public QStandardItemModel
{
public:
    enum Column
    {
        LabelColumn = 0,
        SourceXColumn,
        SourceYColumn,
        SourceZColumn,
        AccuracyXColumn,
        AccuracyYColumn,
        AccuracyZColumn,
        ResidualColumn,
        ProjectionCountColumn,
        EnabledColumn,
        ColumnCount
    };

    enum DataRole
    {
        NodeTypeRole = Qt::UserRole + 1,
        MarkerIdRole,
        MarkerRoleRole
    };

    enum class NodeType
    {
        TotalError,
        Group,
        Marker
    };

    explicit MarkerReferenceTreeModel(QObject *parent = nullptr);
    explicit MarkerReferenceTreeModel(const control_points::MarkerSet &marker_set,
                                      QObject *parent = nullptr);

    void setMarkerSet(const control_points::MarkerSet &marker_set);
};

class ScaleBarReferenceTreeModel final : public QStandardItemModel
{
public:
    enum Column
    {
        LabelColumn = 0,
        FirstMarkerColumn,
        SecondMarkerColumn,
        SourceValueColumn,
        AccuracyColumn,
        EstimatedValueColumn,
        ResidualColumn,
        EnabledColumn,
        ColumnCount
    };

    enum DataRole
    {
        NodeTypeRole = Qt::UserRole + 1,
        ScaleBarIdRole,
        ScaleBarRoleRole,
        FirstMarkerIdRole,
        SecondMarkerIdRole
    };

    enum class NodeType
    {
        TotalError,
        Group,
        ScaleBar
    };

    explicit ScaleBarReferenceTreeModel(QObject *parent = nullptr);
    explicit ScaleBarReferenceTreeModel(const control_points::MarkerSet &marker_set,
                                        QObject *parent = nullptr);

    void setMarkerSet(const control_points::MarkerSet &marker_set);
};

} // namespace xjw::gui::reference
