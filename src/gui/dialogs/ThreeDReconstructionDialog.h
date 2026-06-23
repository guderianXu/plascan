#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QString>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;

class ThreeDReconstructionDialog : public QDialog
{
    Q_OBJECT

public:
    enum class Mode
    {
        ThreeDReconstruction,
        AerialTriangulation
    };

    explicit ThreeDReconstructionDialog(QWidget *parent = nullptr);

    void setMode(Mode mode);
    void setImageCount(int count);
    void setDefaultOutputDir(const QString &dir);
    void applySettings(const QJsonObject &settings);
    QJsonObject collectSettings() const;

signals:
    void settingsChanged(const QJsonObject &settings);
    void runRequested(const QJsonObject &settings);

private:
    void setupUi();
    void emitSettingsChanged();
    void browseOutputDir();
    void start();

    Mode _mode = Mode::ThreeDReconstruction;
    QLabel *_titleLabel = nullptr;
    QLabel *_statusLabel = nullptr;
    QComboBox *_qualityCombo = nullptr;
    QComboBox *_deviceCombo = nullptr;
    QSpinBox *_featureGrayMinSpin = nullptr;
    QSpinBox *_threadsSpin = nullptr;
    QLineEdit *_outputDirEdit = nullptr;
    QCheckBox *_exportObjCheck = nullptr;
    QPushButton *_browseBtn = nullptr;
    QPushButton *_startBtn = nullptr;
    QPushButton *_cancelBtn = nullptr;
};
