#include "ProjectCameraInitialization.h"

#include "project/ProjectSessionModel.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"

#include <QImageReader>
#include <QJsonArray>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>

#include <cmath>

namespace xjw::gui::project {

using xjw::common::project::cameraToJson;

namespace {

std::optional<double> readerExifValue(QImageReader &reader,
                                      const QStringList &preferredTokens,
                                      bool exclude35mm = false)
{
    const QStringList keys = reader.textKeys();
    for (const QString &key : keys)
    {
        const QString lowerKey = key.toLower();
        if (exclude35mm && lowerKey.contains(QStringLiteral("35mm")))
        {
            continue;
        }

        bool matched = false;
        for (const QString &token : preferredTokens)
        {
            if (lowerKey.contains(token.toLower()))
            {
                matched = true;
                break;
            }
        }
        if (!matched)
        {
            continue;
        }

        if (const auto value = parsePossiblyFractionalNumber(reader.text(key)); value.has_value() && *value > 0.0)
        {
            return value;
        }
    }
    return std::nullopt;
}

} // namespace

std::optional<double> parsePossiblyFractionalNumber(const QString &text)
{
    const QString trimmed = text.trimmed();
    if (trimmed.isEmpty())
    {
        return std::nullopt;
    }

    if (trimmed.contains('/'))
    {
        const QStringList parts = trimmed.split('/', Qt::SkipEmptyParts);
        if (parts.size() == 2)
        {
            bool ok0 = false;
            bool ok1 = false;
            const double num = parts[0].trimmed().toDouble(&ok0);
            const double den = parts[1].trimmed().toDouble(&ok1);
            if (ok0 && ok1 && std::abs(den) > 1e-12)
            {
                return num / den;
            }
        }
    }

    static const QRegularExpression re(QStringLiteral(R"(([+-]?\d+(?:\.\d+)?))"));
    const QRegularExpressionMatch match = re.match(trimmed);
    if (!match.hasMatch())
    {
        return std::nullopt;
    }

    bool ok = false;
    const double value = match.captured(1).toDouble(&ok);
    if (!ok)
    {
        return std::nullopt;
    }
    return value;
}

std::optional<double> focalPixelsFromExif(const QString &imagePath,
                                          const QSize &size,
                                          double sensorWidthMm,
                                          QString *sourceTag)
{
    QImageReader reader(imagePath);
    if (const auto focal35 = readerExifValue(reader, {QStringLiteral("focallengthin35mmfilm"), QStringLiteral("35mm")});
        focal35.has_value())
    {
        const double diagPx = std::hypot((double)size.width(), (double)size.height());
        if (sourceTag)
        {
            *sourceTag = QStringLiteral("exif_35mm");
        }
        return *focal35 / 43.27 * diagPx;
    }

    if (const auto focalMm = readerExifValue(reader, {QStringLiteral("focallength")}, true);
        focalMm.has_value() && sensorWidthMm > 1e-9)
    {
        if (sourceTag)
        {
            *sourceTag = QStringLiteral("exif_mm");
        }
        return *focalMm / sensorWidthMm * size.width();
    }

    return std::nullopt;
}

QStringList resolveInitTargets(ProjectData *projectData,
                               const QJsonObject &settings,
                               QString *errorMsg)
{
    if (!projectData)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("项目未就绪");
        }
        return {};
    }

    const QStringList allImages = projectData->getAllImages();
    if (allImages.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("当前项目没有影像");
        }
        return {};
    }

    const int applyScope = settings.value(QStringLiteral("applyScope")).toInt(0);
    if (applyScope == 0)
    {
        return allImages;
    }

    const QString targetImagePath = settings.value(QStringLiteral("applyTargetImagePath")).toString();
    if (targetImagePath.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("请先选择目标影像");
        }
        return {};
    }
    return { targetImagePath };
}

QSet<QString> existingCameraImages(const QJsonObject &meta)
{
    QSet<QString> out;
    const QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &v : images)
    {
        const QJsonObject obj = v.toObject();
        if (!obj.value(QStringLiteral("camera")).isObject())
        {
            continue;
        }
        const QString path = QDir::cleanPath(QFileInfo(obj.value(QStringLiteral("path")).toString()).absoluteFilePath());
        if (!path.isEmpty())
        {
            out.insert(path);
        }
    }
    return out;
}

QJsonObject withPreparedCameras(const QJsonObject &baseMeta,
                                const QMap<QString, QJsonObject> &preparedCameraByImage,
                                bool overwriteExisting)
{
    QJsonObject meta = baseMeta;
    QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    for (int i = 0; i < images.size(); ++i) {
        QJsonObject imgObj = images.at(i).toObject();
        const QString normPath = QDir::cleanPath(QFileInfo(imgObj.value(QStringLiteral("path")).toString()).absoluteFilePath());
        auto it = preparedCameraByImage.constFind(normPath);
        if (it == preparedCameraByImage.constEnd()) continue;
        if (!overwriteExisting && imgObj.value(QStringLiteral("camera")).isObject()) continue;
        imgObj[QStringLiteral("camera")] = it.value();
        images[i] = imgObj;
    }
    meta[QStringLiteral("images")] = images;
    return meta;
}

QJsonObject makeInitializedCameraMeta(double fx,
                                      double fy,
                                      double cx,
                                      double cy,
                                      double k1,
                                      double k2,
                                      double p1,
                                      double p2,
                                      const QString &source,
                                      const QString &distortionModel,
                                      const QSize &imageSize)
{
    xjw::FramePinholeCamera cam;
    cam.setPose(std::array<double, 9>{1,0,0,0,1,0,0,0,1}, std::array<double, 3>{0,0,0});
    cam.setIntrinsics(fx, fy, cx, cy);
    cam.setAxisDirections(1, 1);
    cam.setDistortion(k1, k2, 0.0, p1, p2);

    QJsonObject camObj = cameraToJson(cam);
    camObj[QStringLiteral("source")] = source;
    camObj[QStringLiteral("distortion_model")] = distortionModel;
    camObj[QStringLiteral("image_width")] = imageSize.width();
    camObj[QStringLiteral("image_height")] = imageSize.height();
    camObj[QStringLiteral("pose_initialized_as_identity")] = true;
    camObj[QStringLiteral("pose_note")] = QStringLiteral("R=I, C=[0,0,0]");
    return camObj;
}

} // namespace xjw::gui::project
