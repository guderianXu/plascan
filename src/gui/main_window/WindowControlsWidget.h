#pragma once

#include <QPointer>
#include <QWidget>

class QAction;
class QShortcut;
class QToolButton;

// Wayland Vulkan windows cannot rely on Qt's client-side title bar decorations.
class WindowControlsWidget : public QWidget
{
public:
    WindowControlsWidget(QWidget& window, QAction& fullScreenAction, QWidget* parent = nullptr);
    static bool isNeededForPlatform(const QString& platformName);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void updateWindowState();

    QPointer<QWidget> _window;
    QPointer<QAction> _fullScreenAction;
    QToolButton* _maximizeButton{};
    QToolButton* _fullScreenButton{};
    QShortcut* _escapeFullScreen{};
};
