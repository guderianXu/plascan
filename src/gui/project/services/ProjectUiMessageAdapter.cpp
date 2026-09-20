#include "ProjectUiMessageAdapter.h"

#include <QFileDialog>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>

namespace
{

    QMessageBox::StandardButton toQtDefaultButton(UiAnswer answer)
    {
        switch (answer)
        {
        case UiAnswer::Yes:
            return QMessageBox::Yes;
        case UiAnswer::No:
            return QMessageBox::No;
        case UiAnswer::Cancel:
            return QMessageBox::Cancel;
        }
        return QMessageBox::Cancel;
    }

    UiAnswer fromQtAnswer(QMessageBox::StandardButton answer)
    {
        if (answer == QMessageBox::Yes)
        {
            return UiAnswer::Yes;
        }
        if (answer == QMessageBox::No)
        {
            return UiAnswer::No;
        }
        return UiAnswer::Cancel;
    }

    UiDialogResult acceptedText(const QString& value, bool accepted)
    {
        UiDialogResult result;
        result.accepted = accepted;
        if (accepted)
        {
            result.text = value;
        }
        return result;
    }

    UiDialogResult selectedFile(const QString& value)
    {
        return acceptedText(value, !value.isEmpty());
    }

    UiDialogResult executeFileDialog(QFileDialog& dialog)
    {
        if (dialog.exec() != QDialog::Accepted || dialog.selectedFiles().isEmpty())
        {
            return {};
        }
        return selectedFile(dialog.selectedFiles().constFirst());
    }

} // namespace

QtProjectUiMessageAdapter::QtProjectUiMessageAdapter(QWidget* parentWidget) : _parentWidget(parentWidget)
{
}

QWidget* QtProjectUiMessageAdapter::resolveParent(QWidget* parent) const
{
    return parent ? parent : _parentWidget;
}

void QtProjectUiMessageAdapter::information(QWidget* parent, const QString& title, const QString& text)
{
    QMessageBox::information(resolveParent(parent), title, text);
}

void QtProjectUiMessageAdapter::warning(QWidget* parent, const QString& title, const QString& text)
{
    QMessageBox::warning(resolveParent(parent), title, text);
}

void QtProjectUiMessageAdapter::critical(QWidget* parent, const QString& title, const QString& text)
{
    QMessageBox::critical(resolveParent(parent), title, text);
}

UiAnswer
QtProjectUiMessageAdapter::question(QWidget* parent, const QString& title, const QString& text, UiAnswer defaultAnswer)
{
    const auto answer = QMessageBox::question(resolveParent(parent),
                                              title,
                                              text,
                                              QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
                                              toQtDefaultButton(defaultAnswer));
    return fromQtAnswer(answer);
}

UiReviewDecision QtProjectUiMessageAdapter::review(QWidget* parent, const UiReviewDialogRequest& request)
{
    QMessageBox message_box(resolveParent(parent));
    message_box.setObjectName(request.objectName);
    message_box.setWindowTitle(request.title);
    message_box.setIcon(request.warningIcon ? QMessageBox::Warning : QMessageBox::Question);
    message_box.setText(request.text);
    message_box.setInformativeText(request.informativeText);
    if (!request.detailedText.isEmpty())
    {
        message_box.setDetailedText(request.detailedText);
    }

    QPushButton* accept_button = message_box.addButton(request.acceptText, QMessageBox::AcceptRole);
    QPushButton* discard_button = message_box.addButton(request.discardText, QMessageBox::DestructiveRole);
    accept_button->setObjectName(request.acceptObjectName);
    discard_button->setObjectName(request.discardObjectName);
    message_box.setDefaultButton(accept_button);
    message_box.setEscapeButton(discard_button);
    message_box.exec();
    return message_box.clickedButton() == accept_button ? UiReviewDecision::Accept : UiReviewDecision::Discard;
}

UiDialogResult
QtProjectUiMessageAdapter::getText(QWidget* parent, const QString& title, const QString& label, const QString& initial)
{
    bool accepted = false;
    const QString value =
        QInputDialog::getText(resolveParent(parent), title, label, QLineEdit::Normal, initial, &accepted);
    return acceptedText(value, accepted);
}

UiDialogResult QtProjectUiMessageAdapter::getDouble(QWidget* parent,
                                                    const QString& title,
                                                    const QString& label,
                                                    double initial,
                                                    double minimum,
                                                    double maximum,
                                                    int decimals)
{
    bool accepted = false;
    const double value =
        QInputDialog::getDouble(resolveParent(parent), title, label, initial, minimum, maximum, decimals, &accepted);
    UiDialogResult result;
    result.accepted = accepted;
    if (accepted)
    {
        result.number = value;
    }
    return result;
}

UiDialogResult QtProjectUiMessageAdapter::getItem(
    QWidget* parent, const QString& title, const QString& label, const QStringList& items, int current)
{
    bool accepted = false;
    const QString value = QInputDialog::getItem(resolveParent(parent), title, label, items, current, false, &accepted);
    return acceptedText(value, accepted);
}

UiDialogResult QtProjectUiMessageAdapter::selectOpenFile(QWidget* parent,
                                                         const QString& title,
                                                         const QString& directory,
                                                         const QString& filter,
                                                         QDir::Filters directoryFilters)
{
    QFileDialog dialog(resolveParent(parent), title, directory, filter);
    dialog.setFilter(directoryFilters);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    return executeFileDialog(dialog);
}

UiDialogResult QtProjectUiMessageAdapter::selectOpenFiles(QWidget* parent,
                                                          const QString& title,
                                                          const QString& directory,
                                                          const QString& filter,
                                                          QDir::Filters directoryFilters)
{
    QFileDialog dialog(resolveParent(parent), title, directory, filter);
    dialog.setFilter(directoryFilters);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFiles);
    if (dialog.exec() != QDialog::Accepted)
    {
        return {};
    }
    UiDialogResult result;
    result.texts = dialog.selectedFiles();
    result.accepted = !result.texts.isEmpty();
    return result;
}

UiDialogResult QtProjectUiMessageAdapter::selectDirectory(QWidget* parent,
                                                          const QString& title,
                                                          const QString& directory,
                                                          QDir::Filters directoryFilters)
{
    QFileDialog dialog(resolveParent(parent), title, directory);
    dialog.setFilter(directoryFilters);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::Directory);
    return executeFileDialog(dialog);
}

UiDialogResult QtProjectUiMessageAdapter::selectSaveFile(QWidget* parent,
                                                         const QString& title,
                                                         const QString& directory,
                                                         const QString& filter,
                                                         QDir::Filters directoryFilters,
                                                         const QString& defaultSuffix)
{
    QFileDialog dialog(resolveParent(parent), title, directory, filter);
    dialog.setFilter(directoryFilters);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    if (!defaultSuffix.isEmpty())
    {
        dialog.setDefaultSuffix(defaultSuffix);
    }
    return executeFileDialog(dialog);
}
