#include "ReferenceMarkerModels.h"

#include <QHash>
#include <QList>
#include <QStandardItem>
#include <QVariant>

#include <cmath>

namespace xjw::gui::reference
{
namespace
{

using ItemRow = QList<QStandardItem *>;

struct ResidualStats
{
    void add(double value)
    {
        if (!std::isfinite(value))
        {
            return;
        }

        squaredSum += value * value;
        ++count;
    }

    void add(const ResidualStats &other)
    {
        squaredSum += other.squaredSum;
        count += other.count;
    }

    QVariant rms() const
    {
        if (count == 0)
        {
            return {};
        }

        return std::sqrt(squaredSum / static_cast<double>(count));
    }

    double squaredSum = 0.0;
    int count = 0;
};

ItemRow createRow(int column_count, const QString &label)
{
    ItemRow row;
    row.reserve(column_count);
    for (int column = 0; column < column_count; ++column)
    {
        auto *item = new QStandardItem();
        item->setEditable(false);
        row.append(item);
    }
    row.front()->setText(label);
    return row;
}

template <typename NodeType>
void setNodeType(const ItemRow &row, int role, NodeType type)
{
    for (QStandardItem *item : row)
    {
        item->setData(static_cast<int>(type), role);
    }
}

void setDataForRow(const ItemRow &row, int role, const QVariant &value)
{
    for (QStandardItem *item : row)
    {
        item->setData(value, role);
    }
}

void setDisplayValue(const ItemRow &row, int column, const QVariant &value)
{
    row.at(column)->setData(value, Qt::DisplayRole);
}

ResidualStats markerResidualStats(const control_points::Marker &marker)
{
    ResidualStats stats;
    for (const control_points::MarkerProjection &projection : marker.projections)
    {
        stats.add(projection.residualPx);
    }
    return stats;
}

QString markerLabel(const QHash<control_points::MarkerId, QString> &labels,
                    const control_points::MarkerId &id)
{
    const QString label = labels.value(id);
    return label.isEmpty() ? id : label;
}

} // namespace

MarkerReferenceTreeModel::MarkerReferenceTreeModel(QObject *parent)
    : QStandardItemModel(parent)
{
    setHorizontalHeaderLabels({QStringLiteral("名称"),
                               QStringLiteral("源 X"),
                               QStringLiteral("源 Y"),
                               QStringLiteral("源 Z"),
                               QStringLiteral("精度 X"),
                               QStringLiteral("精度 Y"),
                               QStringLiteral("精度 Z"),
                               QStringLiteral("残差 (px)"),
                               QStringLiteral("投影数"),
                               QStringLiteral("启用")});
}

MarkerReferenceTreeModel::MarkerReferenceTreeModel(
    const control_points::MarkerSet &marker_set,
    QObject *parent)
    : MarkerReferenceTreeModel(parent)
{
    setMarkerSet(marker_set);
}

void MarkerReferenceTreeModel::setMarkerSet(const control_points::MarkerSet &marker_set)
{
    removeRows(0, rowCount());

    ItemRow total_row = createRow(ColumnCount, QStringLiteral("总误差"));
    setNodeType(total_row, NodeTypeRole, NodeType::TotalError);
    QStandardItem *total_item = total_row.at(LabelColumn);
    appendRow(total_row);

    ItemRow control_row = createRow(ColumnCount, QStringLiteral("控制点"));
    setNodeType(control_row, NodeTypeRole, NodeType::Group);
    QStandardItem *control_item = control_row.at(LabelColumn);
    total_item->appendRow(control_row);

    ItemRow check_row = createRow(ColumnCount, QStringLiteral("检查点"));
    setNodeType(check_row, NodeTypeRole, NodeType::Group);
    QStandardItem *check_item = check_row.at(LabelColumn);
    total_item->appendRow(check_row);

    ResidualStats control_stats;
    ResidualStats check_stats;

    for (const control_points::Marker &marker : marker_set.markers())
    {
        QStandardItem *group_item = nullptr;
        ResidualStats *group_stats = nullptr;
        if (marker.role == control_points::MarkerRole::ControlPoint)
        {
            group_item = control_item;
            group_stats = &control_stats;
        }
        else if (marker.role == control_points::MarkerRole::CheckPoint)
        {
            group_item = check_item;
            group_stats = &check_stats;
        }
        else
        {
            continue;
        }

        ItemRow marker_row = createRow(ColumnCount, marker.label);
        setNodeType(marker_row, NodeTypeRole, NodeType::Marker);
        setDataForRow(marker_row, MarkerIdRole, marker.id);
        setDataForRow(marker_row, MarkerRoleRole, static_cast<int>(marker.role));

        if (marker.referenceCoordinate.has_value())
        {
            const control_points::ReferenceCoordinate &reference = *marker.referenceCoordinate;
            setDisplayValue(marker_row, SourceXColumn, reference.x);
            setDisplayValue(marker_row, SourceYColumn, reference.y);
            setDisplayValue(marker_row, SourceZColumn, reference.z);
            setDisplayValue(marker_row, AccuracyXColumn, reference.sigmaX);
            setDisplayValue(marker_row, AccuracyYColumn, reference.sigmaY);
            setDisplayValue(marker_row, AccuracyZColumn, reference.sigmaZ);
        }

        const ResidualStats marker_stats = markerResidualStats(marker);
        setDisplayValue(marker_row, ResidualColumn, marker_stats.rms());
        setDisplayValue(marker_row, ProjectionCountColumn, marker.projections.size());
        setDisplayValue(marker_row, EnabledColumn, marker.enabled);
        if (marker.enabled)
        {
            group_stats->add(marker_stats);
        }
        group_item->appendRow(marker_row);
    }

    setDisplayValue(control_row, ResidualColumn, control_stats.rms());
    setDisplayValue(check_row, ResidualColumn, check_stats.rms());

    ResidualStats total_stats;
    total_stats.add(control_stats);
    total_stats.add(check_stats);
    setDisplayValue(total_row, ResidualColumn, total_stats.rms());
}

ScaleBarReferenceTreeModel::ScaleBarReferenceTreeModel(QObject *parent)
    : QStandardItemModel(parent)
{
    setHorizontalHeaderLabels({QStringLiteral("名称"),
                               QStringLiteral("起点"),
                               QStringLiteral("终点"),
                               QStringLiteral("源值"),
                               QStringLiteral("精度"),
                               QStringLiteral("估计值"),
                               QStringLiteral("残差"),
                               QStringLiteral("启用")});
}

ScaleBarReferenceTreeModel::ScaleBarReferenceTreeModel(
    const control_points::MarkerSet &marker_set,
    QObject *parent)
    : ScaleBarReferenceTreeModel(parent)
{
    setMarkerSet(marker_set);
}

void ScaleBarReferenceTreeModel::setMarkerSet(const control_points::MarkerSet &marker_set)
{
    removeRows(0, rowCount());

    ItemRow total_row = createRow(ColumnCount, QStringLiteral("总误差"));
    setNodeType(total_row, NodeTypeRole, NodeType::TotalError);
    QStandardItem *total_item = total_row.at(LabelColumn);
    appendRow(total_row);

    ItemRow control_row = createRow(ColumnCount, QStringLiteral("控制标尺"));
    setNodeType(control_row, NodeTypeRole, NodeType::Group);
    QStandardItem *control_item = control_row.at(LabelColumn);
    total_item->appendRow(control_row);

    ItemRow check_row = createRow(ColumnCount, QStringLiteral("检查标尺"));
    setNodeType(check_row, NodeTypeRole, NodeType::Group);
    QStandardItem *check_item = check_row.at(LabelColumn);
    total_item->appendRow(check_row);

    QHash<control_points::MarkerId, QString> marker_labels;
    for (const control_points::Marker &marker : marker_set.markers())
    {
        marker_labels.insert(marker.id, marker.label);
    }

    ResidualStats control_stats;
    ResidualStats check_stats;

    for (const control_points::ScaleBar &scale_bar : marker_set.scaleBars())
    {
        QStandardItem *group_item = nullptr;
        ResidualStats *group_stats = nullptr;
        if (scale_bar.role == control_points::ScaleBarRole::Control)
        {
            group_item = control_item;
            group_stats = &control_stats;
        }
        else
        {
            group_item = check_item;
            group_stats = &check_stats;
        }

        ItemRow scale_bar_row = createRow(ColumnCount, scale_bar.label);
        setNodeType(scale_bar_row, NodeTypeRole, NodeType::ScaleBar);
        setDataForRow(scale_bar_row, ScaleBarIdRole, scale_bar.id);
        setDataForRow(scale_bar_row, ScaleBarRoleRole, static_cast<int>(scale_bar.role));
        setDataForRow(scale_bar_row, FirstMarkerIdRole, scale_bar.firstMarkerId);
        setDataForRow(scale_bar_row, SecondMarkerIdRole, scale_bar.secondMarkerId);

        setDisplayValue(scale_bar_row,
                        FirstMarkerColumn,
                        markerLabel(marker_labels, scale_bar.firstMarkerId));
        setDisplayValue(scale_bar_row,
                        SecondMarkerColumn,
                        markerLabel(marker_labels, scale_bar.secondMarkerId));
        setDisplayValue(scale_bar_row, SourceValueColumn, scale_bar.measuredDistance);
        setDisplayValue(scale_bar_row, AccuracyColumn, scale_bar.sigma);
        if (std::isfinite(scale_bar.estimatedDistance))
        {
            setDisplayValue(scale_bar_row, EstimatedValueColumn, scale_bar.estimatedDistance);
        }
        if (std::isfinite(scale_bar.residual))
        {
            setDisplayValue(scale_bar_row, ResidualColumn, scale_bar.residual);
        }
        setDisplayValue(scale_bar_row, EnabledColumn, scale_bar.enabled);

        if (scale_bar.enabled)
        {
            group_stats->add(scale_bar.residual);
        }
        group_item->appendRow(scale_bar_row);
    }

    setDisplayValue(control_row, ResidualColumn, control_stats.rms());
    setDisplayValue(check_row, ResidualColumn, check_stats.rms());

    ResidualStats total_stats;
    total_stats.add(control_stats);
    total_stats.add(check_stats);
    setDisplayValue(total_row, ResidualColumn, total_stats.rms());
}

} // namespace xjw::gui::reference
