#pragma once

#include <QDialog>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTextEdit;

class CameraConvertDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CameraConvertDialog(QWidget *parent = nullptr);

private slots:
    void browseInputDirectory();
    void browseInputFile();
    void browseOutput();
    void runConversion();

private:
    void buildUi();
    void setResultText(const QString &text, bool error);

    QComboBox *_formatCombo = nullptr;
    QLineEdit *_inputEdit = nullptr;
    QLineEdit *_outputEdit = nullptr;
    QCheckBox *_overwriteCheck = nullptr;
    QLabel *_statusLabel = nullptr;
    QTextEdit *_resultEdit = nullptr;
    QPushButton *_runButton = nullptr;
};
