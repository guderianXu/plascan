#include "WindowPanel.h"

#include "ui_WindowPanel.h"

#include <QVBoxLayout>
#include <QToolButton>
#include <QAction>

WindowPanel::WindowPanel(QWidget *parent)
    : QWidget(parent)
{
    Ui::WindowPanel ui;
    ui.setupUi(this);
    m_container = ui.m_container;
}

WindowPanel::~WindowPanel() = default;

void WindowPanel::setActions(const QList<QAction*> &actions)
{
    if (!m_container)
        return;

    auto *inner = qobject_cast<QVBoxLayout*>(m_container->layout());
    if (!inner)
        return;

    // 清理已有子控件
    QLayoutItem *child;
    while ((child = inner->takeAt(0)) != nullptr) {
        if (auto w = child->widget()) {
            w->deleteLater();
        }
        delete child;
    }

    // 为每个 QAction 创建一个 QToolButton 并绑定为默认 action
    for (auto *act : actions) {
        if (!act)
            continue;
        auto *btn = new QToolButton(m_container);
        btn->setDefaultAction(act);
        btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        btn->setCheckable(act->isCheckable());
        inner->addWidget(btn);
    }
    inner->addStretch();
}
