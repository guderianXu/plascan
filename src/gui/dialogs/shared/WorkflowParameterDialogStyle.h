#pragma once

#include <QString>

class QCheckBox;
class QComboBox;
class QDialog;
class QDialogButtonBox;
class QFormLayout;
class QScrollArea;
class QVBoxLayout;
class QWidget;

namespace xjw::gui::dialogs
{

void configureWorkflowParameterDialog(QDialog *dialog);
void configureWorkflowDialogLayout(QVBoxLayout *layout);
void configureWorkflowScrollArea(QScrollArea *scrollArea);
void configureWorkflowForm(QFormLayout *form);
void configureWorkflowInputWidget(QWidget *widget, int minimumWidth = 0);
void configureWorkflowCheckBox(QCheckBox *checkBox);
void configureWorkflowComboBox(QComboBox *comboBox, int minimumWidth = 0);
void configureWorkflowButtonBox(QDialogButtonBox *buttonBox,
                                const QString &primaryActionText = QString());

} // namespace xjw::gui::dialogs
