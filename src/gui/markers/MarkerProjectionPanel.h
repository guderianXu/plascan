#pragma once

#include "model/MarkerTypes.h"

#include <QWidget>

class QTableWidget;

namespace xjw::gui::markers
{

class MarkerProjectionPanel final : public QWidget
{
    Q_OBJECT

public:
    explicit MarkerProjectionPanel(QWidget *parent = nullptr);
    void setMarker(const control_points::Marker *marker);

private:
    QTableWidget *_table = nullptr;
};

} // namespace xjw::gui::markers
