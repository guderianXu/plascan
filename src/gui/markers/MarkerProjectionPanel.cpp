#include "MarkerProjectionPanel.h"

#include <QHeaderView>
#include <QFileInfo>
#include <QTableWidget>
#include <QVBoxLayout>

#include <cmath>

namespace xjw::gui::markers
{

namespace
{

QString stateText(control_points::ProjectionState state)
{
    switch (state)
    {
    case control_points::ProjectionState::ManualPinned: return QStringLiteral("人工确认");
    case control_points::ProjectionState::AutoDetected: return QStringLiteral("自动检测");
    case control_points::ProjectionState::Predicted: return QStringLiteral("预测");
    case control_points::ProjectionState::Blocked: return QStringLiteral("已屏蔽");
    case control_points::ProjectionState::Disabled: return QStringLiteral("已禁用");
    }
    return QStringLiteral("未知");
}

} // namespace

MarkerProjectionPanel::MarkerProjectionPanel(QWidget *parent)
    : QWidget(parent)
    , _table(new QTableWidget(this))
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    _table->setObjectName(QStringLiteral("markerProjectionTable"));
    _table->setColumnCount(6);
    _table->setHorizontalHeaderLabels({QStringLiteral("影像"),
                                       QStringLiteral("状态"),
                                       QStringLiteral("X"),
                                       QStringLiteral("Y"),
                                       QStringLiteral("残差"),
                                       QStringLiteral("置信度")});
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setSelectionBehavior(QAbstractItemView::SelectRows);
    _table->verticalHeader()->setVisible(false);
    _table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int column = 1; column < _table->columnCount(); ++column)
    {
        _table->horizontalHeader()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
    }
    layout->addWidget(_table);
}

void MarkerProjectionPanel::setMarker(const control_points::Marker *marker)
{
    _table->setRowCount(marker ? marker->projections.size() : 0);
    if (!marker) return;

    for (int row = 0; row < marker->projections.size(); ++row)
    {
        const auto &projection = marker->projections.at(row);
        const QString image_label = projection.imagePathSnapshot.trimmed().isEmpty()
            ? projection.imageId
            : QFileInfo(projection.imagePathSnapshot).fileName();
        _table->setItem(row, 0, new QTableWidgetItem(image_label));
        _table->setItem(row, 1, new QTableWidgetItem(stateText(projection.state)));
        _table->setItem(row, 2, new QTableWidgetItem(QString::number(projection.xy.x(), 'f', 3)));
        _table->setItem(row, 3, new QTableWidgetItem(QString::number(projection.xy.y(), 'f', 3)));
        _table->setItem(row, 4, new QTableWidgetItem(std::isfinite(projection.residualPx)
                                                        ? QString::number(projection.residualPx, 'f', 3)
                                                        : QStringLiteral("-")));
        _table->setItem(row, 5, new QTableWidgetItem(QString::number(projection.confidence, 'f', 3)));
    }
}

} // namespace xjw::gui::markers
