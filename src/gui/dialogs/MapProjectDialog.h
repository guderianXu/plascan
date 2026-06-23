#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

class QListWidget;
class QLineEdit;
class QDoubleSpinBox;

class MapProjectDialog : public QDialog
{
    Q_OBJECT
public:
    explicit MapProjectDialog(QWidget *parent = nullptr);

    void setAvailableImages(const QStringList &images);
    void setProjectRoot(const QString &projectRoot);
    void setDefaultDemPath(const QString &demPath);

signals:
    void requestRunMapProject(const QStringList &images,
                              const QString &demPath,
                              const QString &outputPath,
                              double resolution);
    void settingsChanged(const QJsonObject &settings);

public slots:
    void applySettings(const QJsonObject &settings);

private slots:
    void onChooseDem();
    void onChooseOutput();
    void onRun();
    void onSettingsModified();

private:
    QJsonObject currentSettings() const;

    QListWidget *_imageList = nullptr;
    QLineEdit *_demEdit = nullptr;
    QLineEdit *_outputEdit = nullptr;
    QDoubleSpinBox *_resolutionSpin = nullptr;
    QString _projectRoot;
};
