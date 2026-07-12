#pragma once

#include <QToolButton>

class QAction;
class QMenu;
class QToolBar;

namespace xjw::gui::toolbar {

struct ToolbarMetrics
{
    static constexpr int ButtonExtent = 36;
    static constexpr int IconExtent = 26;
    static constexpr int SplitButtonWidth = 50;
    static constexpr int SplitMenuWidth = 14;
};

class ToolbarButton : public QToolButton
{
public:
    explicit ToolbarButton(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;
};

class ToolbarSplitButton : public QToolButton
{
public:
    explicit ToolbarSplitButton(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;
};

void configureToolbar(QToolBar *toolBar);

QAction *createToolbarButton(QToolBar *toolBar,
                             QAction *action,
                             const QString &objectName,
                             QAction *before = nullptr);

QAction *createToolbarSplitButton(QToolBar *toolBar,
                                  QAction *action,
                                  QMenu *menu,
                                  const QString &objectName,
                                  QAction *before = nullptr);

} // namespace xjw::gui::toolbar
