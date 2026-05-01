#pragma once

#include <QWidget>
#include <QList>

class QAction;

// WindowPanel: 在菜单子项中作为自定义 QWidget 显示，包含若干并列的切换按钮
// 当作为 QWidgetAction 的默认 widget 加入 QMenu 时，会在悬停展开菜单时显示
class WindowPanel : public QWidget
{
    Q_OBJECT
public:
    explicit WindowPanel(QWidget *parent = nullptr);
    ~WindowPanel() override;

    // 将一组 QAction 绑定为面板上的按钮（顺序显示）
    void setActions(const QList<QAction*> &actions);

private:
    QWidget *m_container{};
};
