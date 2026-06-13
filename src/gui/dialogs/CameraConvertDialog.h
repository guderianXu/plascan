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

    QComboBox *m_formatCombo = nullptr;
    QLineEdit *m_inputEdit = nullptr;
    QLineEdit *m_outputEdit = nullptr;
    QCheckBox *m_overwriteCheck = nullptr;
    QLabel *m_statusLabel = nullptr;
    QTextEdit *m_resultEdit = nullptr;
    QPushButton *m_runButton = nullptr;
};
