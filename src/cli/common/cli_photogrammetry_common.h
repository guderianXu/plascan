#pragma once

#include "CliJsonIO.h"
#include "CliOutputPolicy.h"
#include "CliPathUtils.h"
#include "placamera/reference/ReferenceCameraGeometry.h"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include <vector>
#include <array>

namespace xjw::cli
{

    struct PhotogrammetryInputItem
    {
        QString imagePath;
        QString cameraPath;
        bool hasCameraPath = false;
    };

    struct PhotogrammetryListOptions
    {
        // 连接点匹配和无先验空三都允许列表里只有影像路径。
        // 旧的一键重建流程才强制要求 image + camera.tsai。
        bool allowImageOnlyRows = true;
        bool requireExistingImages = true;
        bool requireExistingCameras = false;
    };

    bool parsePhotogrammetryListLine(const QString& line, QStringList* parts, QString* errorMessage);

    bool readPhotogrammetryImageList(const QString& listPath,
                                     const PhotogrammetryListOptions& options,
                                     std::vector<PhotogrammetryInputItem>* items,
                                     QString* errorMessage);

    QStringList imagePaths(const std::vector<PhotogrammetryInputItem>& items);
    QStringList cameraPathsForService(const std::vector<PhotogrammetryInputItem>& items);
    bool resolveProjectImageIds(const QJsonObject& projectFiles,
                                const QStringList& images,
                                std::vector<placamera::ImageId>* imageIds,
                                QString* errorMessage);
    bool buildReferenceCameraGeometries(const QJsonObject& projectFiles,
                                        const std::vector<PhotogrammetryInputItem>& items,
                                        const QStringList& images,
                                        const std::vector<placamera::ImageId>& imageIds,
                                        bool useExternalReferenceCameras,
                                        placamera::reference::ReferenceCameraGeometryMap* geometries,
                                        QString* errorMessage);
    bool readReferencePositionCsv(const QString& csvPath,
                                  const QJsonObject& projectFiles,
                                  const QStringList& images,
                                  const std::vector<placamera::ImageId>& imageIds,
                                  placamera::reference::ReferenceCameraPositionMap* positions,
                                  QString* errorMessage);

    QJsonArray inputItemsToJson(const std::vector<PhotogrammetryInputItem>& items);
    QJsonArray inputPairsToJson(const std::vector<PhotogrammetryInputItem>& items);
    QJsonObject projectMetaFromInputItems(const std::vector<PhotogrammetryInputItem>& items);

    QMap<QString, QString> maskPathsFromDirectory(const QString& maskDirectory, const QStringList& images);

} // namespace xjw::cli
