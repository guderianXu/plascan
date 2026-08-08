#pragma once

#include <QObject>
#include <QJsonObject>
#include <QString>
#include <QStringList>

class QWidget;
class ProjectData;
class ProjectManager;

class ProjectCameraSetupManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectCameraSetupManager(ProjectManager *owner,
                                       ProjectData *projectData,
                                       QWidget *parentWidget,
                                       QObject *parent = nullptr);

    bool importCameraForImage(const QString &imagePath);
    bool importCamerasByFilenameBatch();
    bool initializeCamerasFromExifOrDefault(const QJsonObject &settings);
    bool initializeCamerasFromIntrinsics(const QJsonObject &settings);
    bool initializeCameraPosesWithSFM(const QJsonObject &settings);

signals:
    void atProgressChanged(const QString &stage, int percent);
    void atProgressFinished(bool success);
    void matchPairReady(const QString &img0, const QString &img1,
                        const QString &matchFilePath, int numMatches);

private:
    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
};
