#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QMainWindow>
#include <QMenuBar>
#include <QShortcut>
#include <QSignalSpy>
#include <QTest>
#include <QToolButton>

#include "WindowControlsWidget.h"

#include <algorithm>
#include <cstring>

namespace
{
    class CloseConfirmationWindow : public QMainWindow
    {
    public:
        bool allowClose{false};
        int closeRequests{0};

    protected:
        void closeEvent(QCloseEvent* event) override
        {
            ++closeRequests;
            event->setAccepted(allowClose);
        }
    };

    WindowControlsWidget* installControls(QMainWindow& window, QAction& fullScreenAction)
    {
        auto* controls = new WindowControlsWidget(window, fullScreenAction, window.menuBar());
        window.menuBar()->setCornerWidget(controls, Qt::TopRightCorner);
        window.resize(640, 480);
        window.show();
        QApplication::processEvents();
        return controls;
    }
} // namespace

TEST(WindowControlsTest, OnlyWaylandNeedsApplicationWindowControls)
{
    EXPECT_TRUE(WindowControlsWidget::isNeededForPlatform(QStringLiteral("wayland")));
    EXPECT_TRUE(WindowControlsWidget::isNeededForPlatform(QStringLiteral("wayland-egl")));
    for (const QString& platform : {QStringLiteral("xcb"), QStringLiteral("windows"), QStringLiteral("cocoa")})
    {
        EXPECT_FALSE(WindowControlsWidget::isNeededForPlatform(platform));
    }
}

TEST(WindowControlsTest, AllFourControlsAreVisibleAndAccessible)
{
    QMainWindow window;
    QAction full_screen_action(&window);
    auto* controls = installControls(window, full_screen_action);
    for (const QString& name : {QStringLiteral("windowMinimizeButton"),
                                QStringLiteral("windowMaximizeButton"),
                                QStringLiteral("windowFullScreenButton"),
                                QStringLiteral("windowCloseButton")})
    {
        auto* button = controls->findChild<QToolButton*>(name);
        ASSERT_NE(button, nullptr);
        EXPECT_TRUE(button->isVisible());
        EXPECT_FALSE(button->icon().isNull());
        EXPECT_FALSE(button->accessibleName().isEmpty());
    }
    EXPECT_FALSE(controls->findChild<QShortcut*>()->isEnabled());
}

TEST(WindowControlsTest, MinimizeAndMaximizeRestoreUseWindowState)
{
    QMainWindow window;
    QAction full_screen_action(&window);
    auto* controls = installControls(window, full_screen_action);
    auto* maximize = controls->findChild<QToolButton*>(QStringLiteral("windowMaximizeButton"));
    ASSERT_NE(maximize, nullptr);
    maximize->click();
    EXPECT_TRUE(window.isMaximized());
    EXPECT_EQ(maximize->toolTip(), QStringLiteral("还原窗口"));
    maximize->click();
    EXPECT_FALSE(window.isMaximized());
    EXPECT_EQ(maximize->toolTip(), QStringLiteral("最大化"));
    controls->findChild<QToolButton*>(QStringLiteral("windowMinimizeButton"))->click();
    EXPECT_TRUE(window.isMinimized());
    window.showNormal();
}

TEST(WindowControlsTest, FullScreenReusesMenuActionAndEscapeRestoresPreviousState)
{
    QMainWindow window;
    QAction full_screen_action(&window);
    Qt::WindowStates previous_state;
    QObject::connect(&full_screen_action,
                     &QAction::triggered,
                     &window,
                     [&]()
                     {
                         if (window.isFullScreen())
                         {
                             window.setWindowState(previous_state);
                         }
                         else
                         {
                             previous_state = window.windowState();
                             window.showFullScreen();
                         }
                     });
    auto* controls = installControls(window, full_screen_action);
    auto* full_screen = controls->findChild<QToolButton*>(QStringLiteral("windowFullScreenButton"));
    auto* maximize = controls->findChild<QToolButton*>(QStringLiteral("windowMaximizeButton"));
    ASSERT_NE(full_screen, nullptr);
    ASSERT_NE(maximize, nullptr);
    QSignalSpy action_triggered(&full_screen_action, &QAction::triggered);
    window.showMaximized();
    full_screen->click();
    EXPECT_TRUE(window.isFullScreen());
    EXPECT_TRUE(full_screen->isChecked());
    EXPECT_FALSE(maximize->isEnabled());
    EXPECT_TRUE(controls->findChild<QShortcut*>()->isEnabled());
    window.activateWindow();
    QApplication::processEvents();
    QTest::keyClick(&window, Qt::Key_Escape);
    EXPECT_FALSE(window.isFullScreen());
    EXPECT_TRUE(window.isMaximized());
    EXPECT_FALSE(full_screen->isChecked());
    EXPECT_TRUE(maximize->isEnabled());
    EXPECT_FALSE(controls->findChild<QShortcut*>()->isEnabled());
    EXPECT_EQ(action_triggered.count(), 2);
}

TEST(WindowControlsTest, CloseRespectsSaveConfirmationCancellation)
{
    CloseConfirmationWindow window;
    QAction full_screen_action(&window);
    auto* controls = installControls(window, full_screen_action);
    auto* close = controls->findChild<QToolButton*>(QStringLiteral("windowCloseButton"));
    ASSERT_NE(close, nullptr);
    close->click();
    EXPECT_EQ(window.closeRequests, 1);
    EXPECT_TRUE(window.isVisible());
    window.allowClose = true;
    close->click();
    EXPECT_EQ(window.closeRequests, 2);
    EXPECT_FALSE(window.isVisible());
}

int main(int argc, char** argv)
{
    const bool lists_tests =
        std::any_of(argv,
                    argv + argc,
                    [](const char* argument) { return argument && std::strcmp(argument, "--gtest_list_tests") == 0; });
    testing::InitGoogleTest(&argc, argv);
    if (lists_tests)
    {
        return RUN_ALL_TESTS();
    }
    QApplication app(argc, argv);
    return RUN_ALL_TESTS();
}
