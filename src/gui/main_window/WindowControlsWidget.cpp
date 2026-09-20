#include "WindowControlsWidget.h"

#include <QAction>
#include <QEvent>
#include <QHBoxLayout>
#include <QPainter>
#include <QShortcut>
#include <QStyle>
#include <QToolButton>

namespace
{
    QIcon fullScreenIcon(const QWidget& widget)
    {
        const qreal scale = widget.devicePixelRatioF();
        QPixmap pixmap(QSize(16, 16) * scale);
        pixmap.setDevicePixelRatio(scale);
        pixmap.fill(Qt::transparent);
        QPainter painter(&pixmap);
        painter.setPen(QPen(widget.palette().color(QPalette::ButtonText), 1.5));
        for (int x : {2, 14})
        {
            for (int y : {2, 14})
            {
                painter.drawLine(x, y, x == 2 ? 6 : 10, y);
                painter.drawLine(x, y, x, y == 2 ? 6 : 10);
            }
        }
        return QIcon(pixmap);
    }
} // namespace

WindowControlsWidget::WindowControlsWidget(QWidget& window, QAction& fullScreenAction, QWidget* parent)
    : QWidget(parent), _window(&window), _fullScreenAction(&fullScreenAction)
{
    setObjectName(QStringLiteral("windowControls"));
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(4, 0, 0, 0);
    layout->setSpacing(2);
    const auto add_button = [this, layout](const QString& name, const QString& tooltip, QStyle::StandardPixmap icon)
    {
        auto* button = new QToolButton(this);
        button->setObjectName(name);
        button->setToolTip(tooltip);
        button->setAccessibleName(tooltip);
        button->setIcon(style()->standardIcon(icon));
        button->setIconSize(QSize(16, 16));
        button->setFixedSize(32, 28);
        button->setAutoRaise(true);
        layout->addWidget(button);
        return button;
    };
    auto* minimize = add_button(QStringLiteral("windowMinimizeButton"), tr("最小化"), QStyle::SP_TitleBarMinButton);
    _maximizeButton = add_button(QStringLiteral("windowMaximizeButton"), tr("最大化"), QStyle::SP_TitleBarMaxButton);
    _fullScreenButton =
        add_button(QStringLiteral("windowFullScreenButton"), tr("全屏（F11）"), QStyle::SP_TitleBarMaxButton);
    auto* close = add_button(QStringLiteral("windowCloseButton"), tr("关闭"), QStyle::SP_TitleBarCloseButton);

    connect(minimize, &QToolButton::clicked, &window, &QWidget::showMinimized);
    connect(_maximizeButton,
            &QToolButton::clicked,
            this,
            [this]()
            {
                if (!_window)
                {
                    return;
                }
                if (_window->isMaximized())
                {
                    _window->showNormal();
                }
                else
                {
                    _window->showMaximized();
                }
            });
    // Reuse F11's action and closeEvent's save/cancel flow, rather than quitting directly.
    connect(_fullScreenButton, &QToolButton::clicked, &fullScreenAction, &QAction::trigger);
    connect(close, &QToolButton::clicked, &window, &QWidget::close);
    _escapeFullScreen = new QShortcut(QKeySequence(Qt::Key_Escape), this);
    connect(_escapeFullScreen, &QShortcut::activated, &fullScreenAction, &QAction::trigger);
    window.installEventFilter(this);
    updateWindowState();
}

bool WindowControlsWidget::isNeededForPlatform(const QString& platformName)
{
    return platformName.startsWith(QStringLiteral("wayland"));
}

bool WindowControlsWidget::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == _window && (event->type() == QEvent::WindowStateChange || event->type() == QEvent::PaletteChange))
    {
        updateWindowState();
    }
    return QWidget::eventFilter(watched, event);
}

void WindowControlsWidget::updateWindowState()
{
    if (!_window || !_fullScreenAction)
    {
        return;
    }
    const bool maximized = _window->isMaximized();
    const bool full_screen = _window->isFullScreen();
    _maximizeButton->setEnabled(!full_screen);
    _maximizeButton->setIcon(
        style()->standardIcon(maximized ? QStyle::SP_TitleBarNormalButton : QStyle::SP_TitleBarMaxButton));
    _maximizeButton->setToolTip(maximized ? tr("还原窗口") : tr("最大化"));
    _maximizeButton->setAccessibleName(_maximizeButton->toolTip());
    _fullScreenButton->setCheckable(true);
    _fullScreenButton->setChecked(full_screen);
    _fullScreenButton->setIcon(fullScreenIcon(*this));
    _fullScreenButton->setToolTip(full_screen ? tr("退出全屏（F11 / Esc）") : tr("全屏（F11）"));
    _fullScreenButton->setAccessibleName(_fullScreenButton->toolTip());
    _escapeFullScreen->setEnabled(full_screen);
}
