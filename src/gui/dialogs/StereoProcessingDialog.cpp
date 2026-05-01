#include "StereoProcessingDialog.h"

#include <QVBoxLayout>
#include <QLabel>

StereoProcessingDialog::StereoProcessingDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("立体处理"));
    resize(420, 180);
    auto *lay = new QVBoxLayout(this);
    lay->addWidget(new QLabel(QStringLiteral("该对话框已保留，当前请通过“创建 DEM / 生成正射影像”流程执行处理。"), this));
}
