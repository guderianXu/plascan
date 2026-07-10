#pragma once

#include "Camera.h"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include <vector>

namespace xjw::cli
{

struct PhotogrammetryInputItem
{
    QString imagePath;
    QString cameraPath;
    bool hasCameraPath = false;
    bool hasLoadedCamera = false;
    xjw::Camera camera;
};

struct PhotogrammetryListOptions
{
    // 连接点匹配和无先验空三都允许列表里只有影像路径。
    // 旧的一键重建流程才强制要求 image + camera.tsai。
    bool allowImageOnlyRows = true;
    bool loadCameras = false;
    bool requireExistingImages = true;
    bool requireExistingCameras = false;
};

QString cleanAbsolutePath(const QString &path);
QString resolveListToken(const QString &token, const QDir &baseDir);

bool readPhotogrammetryImageList(const QString &listPath,
                                 const PhotogrammetryListOptions &options,
                                 std::vector<PhotogrammetryInputItem> *items,
                                 QString *errorMessage);

QStringList imagePaths(const std::vector<PhotogrammetryInputItem> &items);
QStringList cameraPathsForService(const std::vector<PhotogrammetryInputItem> &items);
QMap<QString, xjw::Camera> referenceCameraMap(const std::vector<PhotogrammetryInputItem> &items);

QJsonObject cameraToJson(const xjw::Camera &camera);
QJsonArray inputItemsToJson(const std::vector<PhotogrammetryInputItem> &items);
QJsonObject projectMetaFromInputItems(const std::vector<PhotogrammetryInputItem> &items);

QMap<QString, QString> maskPathsFromDirectory(const QString &maskDirectory, const QStringList &images);

bool validateOutputDirectory(const QString &outputDir, bool force, QString *errorMessage);
bool ensureDirectory(const QString &directoryPath, QString *errorMessage);
bool writeJsonFile(const QString &path, const QJsonObject &object, QString *errorMessage);

} // namespace xjw::cli
