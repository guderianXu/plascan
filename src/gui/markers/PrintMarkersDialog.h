#pragma once

#include "print/MarkerPdfWriter.h"

#include <QDialog>
#include <QFutureWatcher>

class QCheckBox;
class QCloseEvent;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;

namespace xjw::gui::markers
{

class PrintMarkersDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit PrintMarkersDialog(QWidget *parent = nullptr);

    void setDefaultOutputDirectory(const QString &directory);

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void setupUi();
    void populateFamilies();
    void selectOutputPath();
    void generatePdf();
    void handleFinished();
    void setRunning(bool running);
    control_points::MarkerPrintRequest printRequest() const;

    QFutureWatcher<control_points::MarkerPdfWriteResult> _watcher;
    bool _running = false;
    bool _closeAfterRun = false;
    QString _defaultOutputDirectory;

    QComboBox *_familyCombo = nullptr;
    QSpinBox *_startIdSpin = nullptr;
    QSpinBox *_countSpin = nullptr;
    QDoubleSpinBox *_diameterSpin = nullptr;
    QComboBox *_pageSizeCombo = nullptr;
    QDoubleSpinBox *_marginSpin = nullptr;
    QDoubleSpinBox *_spacingSpin = nullptr;
    QCheckBox *_showLabelsCheck = nullptr;
    QLineEdit *_outputEdit = nullptr;
    QPushButton *_browseButton = nullptr;
    QLabel *_statusLabel = nullptr;
    QPushButton *_generateButton = nullptr;
    QPushButton *_closeButton = nullptr;
};

} // namespace xjw::gui::markers
