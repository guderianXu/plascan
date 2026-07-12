#pragma once

class QComboBox;
class QDialog;
class QDialogButtonBox;
class QFormLayout;
class QScrollArea;
class QVBoxLayout;

namespace xjw::gui::dialogs
{

void configureWorkflowParameterDialog(QDialog *dialog);
void configureWorkflowDialogLayout(QVBoxLayout *layout);
void configureWorkflowScrollArea(QScrollArea *scrollArea);
void configureWorkflowForm(QFormLayout *form);
void configureWorkflowComboBox(QComboBox *comboBox);
void configureWorkflowButtonBox(QDialogButtonBox *buttonBox);

} // namespace xjw::gui::dialogs
