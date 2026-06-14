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

    Mode m_mode = Mode::ThreeDReconstruction;
    QLabel *m_titleLabel = nullptr;
    QLabel *m_statusLabel = nullptr;
    QComboBox *m_qualityCombo = nullptr;
    QComboBox *m_deviceCombo = nullptr;
    QSpinBox *m_featureGrayMinSpin = nullptr;
    QSpinBox *m_threadsSpin = nullptr;
    QLineEdit *m_outputDirEdit = nullptr;
    QCheckBox *m_exportObjCheck = nullptr;
    QPushButton *m_browseBtn = nullptr;
    QPushButton *m_startBtn = nullptr;
    QPushButton *m_cancelBtn = nullptr;
};
