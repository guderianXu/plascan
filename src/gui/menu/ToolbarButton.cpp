#include "ToolbarButton.h"

#include <QAction>
#include <QMenu>
#include <QPainter>
#include <QPaintEvent>
#include <QPolygonF>
#include <QSizePolicy>
#include <QToolBar>

namespace xjw::gui::toolbar {
namespace {

QColor backgroundColor(const QToolButton *button)
{
    if (!button->isEnabled())
    {
        return Qt::transparent;
    }
    if (button->isDown())
    {
        return QColor(215, 220, 226);
    }
    if (button->isChecked())
    {
        return button->underMouse() ? QColor(222, 227, 233) : QColor(229, 233, 238);
    }
    return button->underMouse() ? QColor(232, 236, 241) : Qt::transparent;
}

void paintButtonBackground(QPainter &painter, const QToolButton *button)
{
    const QColor background = backgroundColor(button);
    if (background.alpha() == 0)
    {
        return;
    }
    painter.setPen(Qt::NoPen);
    painter.setBrush(background);
    painter.drawRect(button->rect().adjusted(1, 1, -1, -1));
}

void paintButtonIcon(QPainter &painter, const QToolButton *button, const QRect &iconArea)
{
    QRect iconRect = iconArea;
    const QSize drawSize(ToolbarMetrics::IconExtent, ToolbarMetrics::IconExtent);
    const QPoint iconTopLeft(iconRect.left() + (iconRect.width() - drawSize.width()) / 2,
                             iconRect.top() + (iconRect.height() - drawSize.height()) / 2);
    const QIcon::Mode mode = button->isEnabled() ? QIcon::Normal : QIcon::Disabled;
    const QIcon::State state = button->isChecked() ? QIcon::On : QIcon::Off;
    const QPixmap pixmap = button->icon().pixmap(drawSize, mode, state);
    painter.drawPixmap(iconTopLeft, pixmap);
}

void drawToolbarSplitButtonArrow(QPainter &painter, const QRect &arrowArea, const QColor &color)
{
    const QPointF center = arrowArea.center();
    QPolygonF arrow;
    arrow << QPointF(center.x() - 3.0, center.y() - 1.5)
          << QPointF(center.x() + 3.0, center.y() - 1.5)
          << QPointF(center.x(), center.y() + 2.0);
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawPolygon(arrow);
}

} // namespace

ToolbarButton::ToolbarButton(QWidget *parent)
    : QToolButton(parent)
{
    setAutoRaise(true);
    setToolButtonStyle(Qt::ToolButtonIconOnly);
    setFocusPolicy(Qt::NoFocus);
    setIconSize(QSize(ToolbarMetrics::IconExtent, ToolbarMetrics::IconExtent));
    setFixedSize(ToolbarMetrics::ButtonExtent, ToolbarMetrics::ButtonExtent);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void ToolbarButton::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    paintButtonBackground(painter, this);
    paintButtonIcon(painter, this, rect());
}

ToolbarSplitButton::ToolbarSplitButton(QWidget *parent)
    : QToolButton(parent)
{
    setAutoRaise(true);
    setPopupMode(QToolButton::MenuButtonPopup);
    setToolButtonStyle(Qt::ToolButtonIconOnly);
    setFocusPolicy(Qt::NoFocus);
    setIconSize(QSize(ToolbarMetrics::IconExtent, ToolbarMetrics::IconExtent));
    setFixedSize(ToolbarMetrics::SplitButtonWidth, ToolbarMetrics::ButtonExtent);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void ToolbarSplitButton::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    paintButtonBackground(painter, this);

    QRect iconArea = rect();
    iconArea.setRight(width() - ToolbarMetrics::SplitMenuWidth - 1);
    paintButtonIcon(painter, this, iconArea);

    QRect arrowArea = rect();
    arrowArea.setLeft(width() - ToolbarMetrics::SplitMenuWidth);
    drawToolbarSplitButtonArrow(painter,
                                arrowArea,
                                isEnabled() ? QColor(112, 118, 124) : QColor(178, 182, 187));
}

void configureToolbar(QToolBar *toolBar)
{
    if (!toolBar)
    {
        return;
    }
    toolBar->setMovable(false);
    toolBar->setFloatable(false);
    toolBar->setToolButtonStyle(Qt::ToolButtonIconOnly);
    toolBar->setIconSize(QSize(ToolbarMetrics::IconExtent, ToolbarMetrics::IconExtent));
    toolBar->setContentsMargins(4, 2, 4, 2);
    toolBar->setStyleSheet(QStringLiteral(
        "QToolBar { spacing: 1px; padding: 2px 4px; border: 0px; }"
        "QToolBar::separator { width: 9px; margin: 5px 4px; background: #d7dce2; }"));
}

QAction *createToolbarButton(QToolBar *toolBar,
                             QAction *action,
                             const QString &objectName,
                             QAction *before)
{
    if (!toolBar || !action)
    {
        return nullptr;
    }
    auto *button = new ToolbarButton(toolBar);
    button->setObjectName(objectName);
    button->setDefaultAction(action);
    button->setToolButtonStyle(Qt::ToolButtonIconOnly);
    button->setIconSize(QSize(ToolbarMetrics::IconExtent, ToolbarMetrics::IconExtent));
    return before && toolBar->actions().contains(before)
        ? toolBar->insertWidget(before, button)
        : toolBar->addWidget(button);
}

QAction *createToolbarSplitButton(QToolBar *toolBar,
                                  QAction *action,
                                  QMenu *menu,
                                  const QString &objectName,
                                  QAction *before)
{
    if (!toolBar || !action || !menu)
    {
        return nullptr;
    }
    auto *button = new ToolbarSplitButton(toolBar);
    button->setObjectName(objectName);
    button->setDefaultAction(action);
    button->setMenu(menu);
    button->setPopupMode(QToolButton::MenuButtonPopup);
    button->setToolButtonStyle(Qt::ToolButtonIconOnly);
    button->setIconSize(QSize(ToolbarMetrics::IconExtent, ToolbarMetrics::IconExtent));
    return before && toolBar->actions().contains(before)
        ? toolBar->insertWidget(before, button)
        : toolBar->addWidget(button);
}

} // namespace xjw::gui::toolbar
