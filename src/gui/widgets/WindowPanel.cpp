#include "WindowPanel.h"

#include <QVBoxLayout>
#include <QToolButton>
#include <QAction>

WindowPanel::WindowPanel(QWidget *parent)
    : QWidget(parent)
{
    auto *lay = new QVBoxLayout(this);
    lay->setContentsMargins(6, 6, 6, 6);
    lay->setSpacing(6);
    m_container = new QWidget(this);
    auto *inner = new QVBoxLayout(m_container);
    inner->setContentsMargins(0, 0, 0, 0);
    inner->setSpacing(4);
    lay->addWidget(m_container);
    lay->addStretch();
}

WindowPanel::~WindowPanel() = default;

void WindowPanel::setActions(const QList<QAction*> &actions)
{
    if (!m_container)
        return;

    auto *inner = qobject_cast<QVBoxLayout*>(m_container->layout());
    if (!inner) {
        inner = new QVBoxLayout(m_container);
        inner->setContentsMargins(0, 0, 0, 0);
        inner->setSpacing(4);
    }

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
