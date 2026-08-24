#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QSizePolicy>
#include <QVBoxLayout>
#include <QWidget>

#include "shared/WorkflowParameterDialogStyle.h"

namespace xjw::gui::dialogs
{

void configureWorkflowParameterDialog(QDialog *dialog)
{
    if (!dialog)
    {
        return;
    }

    dialog->setProperty("workflowParameterDialog", true);
    dialog->setMinimumWidth(520);
    dialog->setSizeGripEnabled(false);
}

void configureWorkflowDialogLayout(QVBoxLayout *layout)
{
    if (!layout)
    {
        return;
    }

    layout->setContentsMargins(14, 12, 14, 12);
    layout->setSpacing(9);
    layout->setSizeConstraint(QLayout::SetMinimumSize);
    layout->addStrut(492);
}

void configureWorkflowScrollArea(QScrollArea *scrollArea)
{
    if (!scrollArea)
    {
        return;
    }

    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    scrollArea->setAutoFillBackground(false);
    scrollArea->viewport()->setAutoFillBackground(false);
}

void configureWorkflowForm(QFormLayout *form)
{
    if (!form)
    {
        return;
    }

    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setRowWrapPolicy(QFormLayout::DontWrapRows);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFormAlignment(Qt::AlignTop);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(6);
    form->setContentsMargins(12, 12, 12, 10);
}

void configureWorkflowInputWidget(QWidget *widget, int minimumWidth)
{
    if (!widget)
    {
        return;
    }

    widget->setMinimumHeight(28);
    if (minimumWidth > 0)
    {
        widget->setMinimumWidth(minimumWidth);
    }
    widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void configureWorkflowCheckBox(QCheckBox *checkBox)
{
    if (!checkBox)
    {
        return;
    }

    checkBox->setMinimumHeight(24);
    checkBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void configureWorkflowComboBox(QComboBox *comboBox, int minimumWidth)
{
    if (!comboBox)
    {
        return;
    }

    configureWorkflowInputWidget(comboBox, minimumWidth);
    comboBox->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    comboBox->setMinimumContentsLength(24);
}

void configureWorkflowButtonBox(QDialogButtonBox *buttonBox,
                                const QString &primaryActionText)
{
    if (!buttonBox)
    {
        return;
    }

    buttonBox->setCenterButtons(true);
    if (QPushButton *okButton = buttonBox->button(QDialogButtonBox::Ok))
    {
        okButton->setText(primaryActionText.isEmpty()
                              ? QCoreApplication::translate(
                                    "WorkflowParameterDialogStyle", "确定")
                              : primaryActionText);
    }
    if (QPushButton *cancelButton = buttonBox->button(QDialogButtonBox::Cancel))
    {
        cancelButton->setText(QCoreApplication::translate("WorkflowParameterDialogStyle", "取消"));
    }
}

} // namespace xjw::gui::dialogs
