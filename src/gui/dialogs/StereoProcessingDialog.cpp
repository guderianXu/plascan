#include "StereoProcessingDialog.h"
#include "ui_StereoProcessingDialog.h"

StereoProcessingDialog::StereoProcessingDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::StereoProcessingDialog form;
    form.setupUi(this);
}
