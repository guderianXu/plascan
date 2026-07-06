#include "ProjectSupportUtils.h"
#include "ProjectIO.h"
#include "project/ProjectCommonUtils.h"
#include "io/PathIO.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSet>
#include <QtMath>

#include <algorithm>
#include <array>

namespace xjw
{
namespace gui
{
namespace project
{

namespace
{

// 将旋转矩阵（camera->world）转换为 ZYX 欧拉角（yaw/pitch/roll，单位：度）。
// 该函数只用于导出可读元数据，不参与优化计算。
QJsonObject rotationToYprDeg(const std::array<double, 9> &R)
{
    const double r00 = R[0];
    const double r10 = R[3];
    const double r20 = R[6];
    const double r21 = R[7];
    const double r22 = R[8];

    const double pitch = std::asin(std::clamp(-r20, -1.0, 1.0));
    double yaw = 0.0;
    double roll = 0.0;

    // 常规姿态：可稳定求解 yaw/roll。
    if (std::abs(std::cos(pitch)) > 1e-8)
    {
        yaw = std::atan2(r10, r00);
        roll = std::atan2(r21, r22);
    }
    else
    {
        // 俯仰接近 ±90° 时进入万向锁，采用退化公式保证数值稳定。
        yaw = std::atan2(-R[1], R[4]);
        roll = 0.0;
    }

    QJsonObject ypr;
    ypr[QStringLiteral("yaw_deg")] = qRadiansToDegrees(yaw);
    ypr[QStringLiteral("pitch_deg")] = qRadiansToDegrees(pitch);
    ypr[QStringLiteral("roll_deg")] = qRadiansToDegrees(roll);
    return ypr;
}

QPair<QString, QString> canonicalImageNamePair(const QString &leftName, const QString &rightName)
{
    if (leftName <= rightName)
    {
        return qMakePair(leftName, rightName);
    }
    return qMakePair(rightName, leftName);
}

QString matchPairKey(const QString &leftName, const QString &rightName)
{
    const QPair<QString, QString> canonicalPair = canonicalImageNamePair(leftName, rightName);
    return canonicalPair.first + QStringLiteral("\n") + canonicalPair.second;
}

void appendMatchedPair(QVector<QPair<QString, QString>> *pairs,
                       QSet<QString> *seenKeys,
                       const QString &leftName,
                       const QString &rightName)
{
    if (!pairs || !seenKeys)
    {
        return;
    }

    const QString trimmedLeftName = leftName.trimmed();
    const QString trimmedRightName = rightName.trimmed();
    if (trimmedLeftName.isEmpty() || trimmedRightName.isEmpty() || trimmedLeftName == trimmedRightName)
    {
        return;
    }

    const QString key = matchPairKey(trimmedLeftName, trimmedRightName);
    if (seenKeys->contains(key))
    {
        return;
    }

    seenKeys->insert(key);
    pairs->push_back(canonicalImageNamePair(trimmedLeftName, trimmedRightName));
}

void appendMatchedPairFromFileStem(QVector<QPair<QString, QString>> *pairs,
                                   QSet<QString> *seenKeys,
                                   const QString &fileStem)
{
    const QStringList doubleUnderscoreParts = fileStem.split(QStringLiteral("__"));
    if (doubleUnderscoreParts.size() == 2)
    {
        appendMatchedPair(pairs, seenKeys, doubleUnderscoreParts[0], doubleUnderscoreParts[1]);
        return;
    }

    const QStringList dashParts = fileStem.split(QStringLiteral("-"));
    if (dashParts.size() == 2)
    {
        appendMatchedPair(pairs, seenKeys, dashParts[0], dashParts[1]);
    }
}

} // namespace

QString normalizePath(const QString &path)
{
    return xjw::common::project::normalizePath(path);
}

QJsonObject projectFilesRootObject(const QJsonObject &meta)
{
    if (meta.value(QStringLiteral("project_files")).isObject())
    {
        return meta.value(QStringLiteral("project_files")).toObject();
    }
    return meta;
}

QJsonArray projectImageEntries(const QJsonObject &meta)
{
    return projectFilesRootObject(meta).value(QStringLiteral("images")).toArray();
}

QStringList projectImagePaths(const QJsonObject &meta)
{
    QStringList imagePaths;
    const QJsonArray imageEntries = projectImageEntries(meta);
    imagePaths.reserve(imageEntries.size());

    for (const QJsonValue &imageValue : imageEntries)
    {
        const QString imagePath = imageValue.toObject().value(QStringLiteral("path")).toString();
        if (!imagePath.isEmpty())
        {
            imagePaths.push_back(imagePath);
        }
    }

    return imagePaths;
}

QMap<QString, QJsonObject> projectImageMetaByPath(const QJsonObject &meta, bool normalizePaths)
{
    QMap<QString, QJsonObject> imageMetaByPath;
    const QJsonArray imageEntries = projectImageEntries(meta);
    for (const QJsonValue &imageValue : imageEntries)
    {
        const QJsonObject imageObject = imageValue.toObject();
        QString imagePath = imageObject.value(QStringLiteral("path")).toString();
        if (imagePath.isEmpty())
        {
            continue;
        }

        if (normalizePaths)
        {
            imagePath = normalizePath(imagePath);
        }
        imageMetaByPath.insert(imagePath, imageObject);
    }

    return imageMetaByPath;
}

bool pathTokenMatchesImage(const QString &token, const QString &imagePath)
{
    return xjw::common::project::pathTokenMatchesImage(token, imagePath);
}

QString resolveProjectImagePathFromToken(const QString &token, const QJsonObject &meta)
{
    if (token.isEmpty())
    {
        return QString();
    }

    const QStringList imagePaths = projectImagePaths(meta);
    for (const QString &imagePath : imagePaths)
    {
        if (pathTokenMatchesImage(token, imagePath))
        {
            return imagePath;
        }
    }

    return QString();
}

QString resolveProjectFeaturePathFromToken(const QString &plascanPath,
                                          const QJsonObject &meta,
                                          const QString &token)
{
    const QString imagePath = resolveProjectImagePathFromToken(token, meta);
    if (!imagePath.isEmpty())
    {
        const QString spPath = ProjectIO::findFeatureForImage(plascanPath, imagePath);
        if (!spPath.isEmpty())
        {
            return spPath;
        }
    }

    return ProjectIO::findFeatureForImage(plascanPath, token);
}

namespace
{

QString normalizedFeatureSuffix(QString suffix)
{
    suffix = suffix.trimmed().toLower();
    if (suffix.isEmpty())
    {
        return QString();
    }
    if (!suffix.startsWith(QLatin1Char('.')))
    {
        suffix.prepend(QLatin1Char('.'));
    }
    return suffix;
}

QSet<QString> collectProjectFeatureSuffixes(const QString &plascanPath, const QJsonObject &meta)
{
    QSet<QString> availableSuffixes;
    const QStringList imagePaths = projectImagePaths(meta);
    for (const QString &imagePath : imagePaths)
    {
        for (const QString &suffix : ProjectIO::availableFeatureSuffixes(plascanPath, imagePath))
        {
            const QString normalized = normalizedFeatureSuffix(suffix);
            if (!normalized.isEmpty())
            {
                availableSuffixes.insert(normalized);
            }
        }
    }

    const QDir ipDir(ProjectIO::ipfindOutputDir(plascanPath));
    if (ipDir.exists())
    {
        static const QStringList knownSuffixes = {
            QStringLiteral(".sp"),
            QStringLiteral(".dsk"),
            QStringLiteral(".alk"),
            QStringLiteral(".sift"),
            QStringLiteral(".orb"),
            QStringLiteral(".akz"),
            QStringLiteral(".dedode")
        };
        const QFileInfoList files = ipDir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
        for (const QFileInfo &fileInfo : files)
        {
            const QString fileName = fileInfo.fileName().toLower();
            for (const QString &suffix : knownSuffixes)
            {
                if (fileName.endsWith(suffix))
                {
                    availableSuffixes.insert(suffix);
                }
            }
        }
    }

    return availableSuffixes;
}

} // namespace

QString resolveFeaturePathBySuffix(const QString &plascanPath, const QJsonObject &meta,
                                   const QString &token, const QString &suffix)
{
    const QString imagePath = resolveProjectImagePathFromToken(token, meta);
    if (imagePath.isEmpty()) return {};
    return ProjectIO::featureFileForSuffix(plascanPath, imagePath, suffix);
}

QStringList projectFeatureSuffixes(const QString &plascanPath, const QJsonObject &meta)
{
    QSet<QString> availableSuffixes = collectProjectFeatureSuffixes(plascanPath, meta);
    static const QStringList preferredOrder = {
        QStringLiteral(".dsk"),
        QStringLiteral(".alk"),
        QStringLiteral(".sp"),
        QStringLiteral(".sift"),
        QStringLiteral(".orb"),
        QStringLiteral(".akz"),
        QStringLiteral(".dedode")
    };

    QStringList ordered;
    for (const QString &suffix : preferredOrder)
    {
        if (availableSuffixes.remove(suffix))
        {
            ordered.append(suffix);
        }
    }

    QStringList extras = availableSuffixes.values();
    std::sort(extras.begin(), extras.end());
    ordered.append(extras);
    return ordered;
}

QString inferPreferredFeatureSuffix(const QString &plascanPath, const QJsonObject &meta)
{
    for (const QString &suffix : projectFeatureSuffixes(plascanPath, meta))
    {
        return suffix;
    }

    return QString();
}

bool projectHasFeatureSuffix(const QString &plascanPath, const QJsonObject &meta, const QString &suffix)
{
    const QString normalized = normalizedFeatureSuffix(suffix);
    if (normalized.isEmpty())
    {
        return false;
    }
    return collectProjectFeatureSuffixes(plascanPath, meta).contains(normalized);
}

QJsonObject cameraToJson(const xjw::Camera &camera)
{
    // 统一输出字段命名，供 ProjectData / ProjectManager 持久化使用。
    const auto intrinsics = camera.intrinsics();
    const auto distortion = camera.distortion();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonObject camObj;
    camObj[QStringLiteral("model")] = QStringLiteral("tsai");
    camObj[QStringLiteral("imported_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    camObj[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    camObj[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    camObj[QStringLiteral("pitch")] = camera.pixelPitch();
    camObj[QStringLiteral("fu")] = camera.focalXMillimeters();
    camObj[QStringLiteral("fv")] = camera.focalYMillimeters();
    camObj[QStringLiteral("cu")] = camera.principalXMillimeters();
    camObj[QStringLiteral("cv")] = camera.principalYMillimeters();
    camObj[QStringLiteral("k1")] = distortion.radialK1;
    camObj[QStringLiteral("k2")] = distortion.radialK2;
    camObj[QStringLiteral("k3")] = distortion.radialK3;
    camObj[QStringLiteral("p1")] = distortion.tangentialP1;
    camObj[QStringLiteral("p2")] = distortion.tangentialP2;
    camObj[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
    camObj[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
    camObj[QStringLiteral("depth_axis_flipped")] = camera.depthAxisFlipped();

    QJsonArray cArr;
    cArr.append(center[0]);
    cArr.append(center[1]);
    cArr.append(center[2]);
    camObj[QStringLiteral("C")] = cArr;

    QJsonArray rArr;
    for (int i = 0; i < 9; ++i)
    {
        rArr.append(rotation[i]);
    }
    camObj[QStringLiteral("R")] = rArr;

    const QJsonObject ypr = rotationToYprDeg(rotation);
    camObj[QStringLiteral("yaw_deg")] = ypr.value(QStringLiteral("yaw_deg")).toDouble();
    camObj[QStringLiteral("pitch_deg")] = ypr.value(QStringLiteral("pitch_deg")).toDouble();
    camObj[QStringLiteral("roll_deg")] = ypr.value(QStringLiteral("roll_deg")).toDouble();

    return camObj;
}

bool parseTsaiCamera(const QString &tsaiPath, QJsonObject *cameraMeta, QString *errorMsg)
{
    if (!cameraMeta)
    {
        return false;
    }
    xjw::Camera cam;
    // 文件读取或模型合法性校验任一失败都视为解析失败。
    if (!cam.loadFromFile(xjw::common::io::toUtf8Path(tsaiPath)) || !cam.isValid())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法解析相机文件: %1").arg(tsaiPath);
        }
        return false;
    }
    *cameraMeta = cameraToJson(cam);
    return true;
}

bool cameraFromJson(const QJsonObject &camObj, xjw::Camera *cam)
{
    return xjw::common::project::cameraFromJson(camObj, cam);
}

bool imageCameraFromEntry(const QJsonObject &imageObj, xjw::Camera *cam)
{
    return cameraFromJson(imageObj.value(QStringLiteral("camera")).toObject(), cam);
}

QVector<QPair<QString, QString>> collectMatchedImageNamePairs(const QString &plascanPath,
                                                              const QJsonObject &meta)
{
    QVector<QPair<QString, QString>> matchedPairs;
    QSet<QString> seenKeys;
    const QJsonArray imageEntries = projectImageEntries(meta);

    auto resolveImageDisplayName = [&imageEntries](const QString &token) -> QString
    {
        const QString trimmedToken = token.trimmed();
        if (trimmedToken.isEmpty())
        {
            return QString();
        }

        for (const QJsonValue &imageValue : imageEntries)
        {
            const QString imagePath = imageValue.toObject().value(QStringLiteral("path")).toString();
            if (imagePath.isEmpty())
            {
                continue;
            }

            if (pathTokenMatchesImage(trimmedToken, imagePath))
            {
                const QString fileName = QFileInfo(imagePath).fileName();
                return fileName.isEmpty() ? imagePath : fileName;
            }
        }

        const QString tokenFileName = QFileInfo(trimmedToken).fileName();
        return tokenFileName.isEmpty() ? trimmedToken : tokenFileName;
    };

    auto appendResolvedPair = [&](const QString &leftToken, const QString &rightToken)
    {
        appendMatchedPair(&matchedPairs,
                          &seenKeys,
                          resolveImageDisplayName(leftToken),
                          resolveImageDisplayName(rightToken));
    };

    if (!plascanPath.isEmpty())
    {
        const QString matchesDirPath = ProjectIO::ipmatchOutputDir(plascanPath);
        const QDir matchesDir(matchesDirPath);
        const QStringList matchFileNames = matchesDir.entryList(QStringList{QStringLiteral("*.match")},
                                                                QDir::Files,
                                                                QDir::Name);
        for (const QString &matchFileName : matchFileNames)
        {
            const QString sidecarPath = matchesDir.filePath(matchFileName + QStringLiteral(".json"));
            QFile sidecarFile(sidecarPath);
            if (sidecarFile.open(QIODevice::ReadOnly))
            {
                const QJsonDocument sidecarDoc = QJsonDocument::fromJson(sidecarFile.readAll());
                const QJsonObject sidecarObject = sidecarDoc.object();
                const QString image0Name = sidecarObject.value(QStringLiteral("image0_name")).toString();
                const QString image1Name = sidecarObject.value(QStringLiteral("image1_name")).toString();
                appendResolvedPair(image0Name, image1Name);
                if (!image0Name.isEmpty() && !image1Name.isEmpty())
                {
                    continue;
                }
            }

            const QString fileStem = QFileInfo(matchFileName).completeBaseName();
            const QStringList doubleUnderscoreParts = fileStem.split(QStringLiteral("__"));
            if (doubleUnderscoreParts.size() == 2)
            {
                appendResolvedPair(doubleUnderscoreParts[0], doubleUnderscoreParts[1]);
                continue;
            }

            const QStringList dashParts = fileStem.split(QStringLiteral("-"));
            if (dashParts.size() == 2)
            {
                appendResolvedPair(dashParts[0], dashParts[1]);
            }
        }
    }

    const QJsonArray ipmatchResults = meta.value(QStringLiteral("ipmatch_results")).toArray();
    for (const QJsonValue &resultValue : ipmatchResults)
    {
        const QJsonObject resultObject = resultValue.toObject();
        QString image0Name = resultObject.value(QStringLiteral("image0_name")).toString();
        QString image1Name = resultObject.value(QStringLiteral("image1_name")).toString();

        if (image0Name.isEmpty())
        {
            image0Name = QFileInfo(resultObject.value(QStringLiteral("image0")).toString()).completeBaseName();
        }
        if (image1Name.isEmpty())
        {
            image1Name = QFileInfo(resultObject.value(QStringLiteral("image1")).toString()).completeBaseName();
        }

        appendResolvedPair(image0Name, image1Name);
    }

    std::sort(matchedPairs.begin(), matchedPairs.end(),
              [](const QPair<QString, QString> &leftPair, const QPair<QString, QString> &rightPair)
              {
                  if (leftPair.first != rightPair.first)
                  {
                      return leftPair.first < rightPair.first;
                  }
                  return leftPair.second < rightPair.second;
              });
    return matchedPairs;
}

QVector<QPair<QString, QString>> collectSettledNoMatchImageNamePairs(const QString &plascanPath,
                                                                     const QJsonObject &meta)
{
    QVector<QPair<QString, QString>> noMatchPairs;
    QSet<QString> seenKeys;
    const QJsonArray imageEntries = projectImageEntries(meta);

    auto resolveImageDisplayName = [&imageEntries](const QString &token) -> QString
    {
        const QString trimmedToken = token.trimmed();
        if (trimmedToken.isEmpty())
        {
            return QString();
        }

        for (const QJsonValue &imageValue : imageEntries)
        {
            const QString imagePath = imageValue.toObject().value(QStringLiteral("path")).toString();
            if (imagePath.isEmpty())
            {
                continue;
            }

            if (pathTokenMatchesImage(trimmedToken, imagePath))
            {
                const QString fileName = QFileInfo(imagePath).fileName();
                return fileName.isEmpty() ? imagePath : fileName;
            }
        }

        const QString tokenFileName = QFileInfo(trimmedToken).fileName();
        return tokenFileName.isEmpty() ? trimmedToken : tokenFileName;
    };

    if (plascanPath.isEmpty())
    {
        return noMatchPairs;
    }

    QFile noMatchFile(QDir(ProjectIO::ipmatchOutputDir(plascanPath)).filePath(QStringLiteral("no_match_pairs.json")));
    if (!noMatchFile.open(QIODevice::ReadOnly))
    {
        return noMatchPairs;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(noMatchFile.readAll());
    const QJsonArray records = doc.array();
    for (const QJsonValue &value : records)
    {
        const QJsonObject object = value.toObject();
        appendMatchedPair(&noMatchPairs,
                          &seenKeys,
                          resolveImageDisplayName(object.value(QStringLiteral("image0")).toString()),
                          resolveImageDisplayName(object.value(QStringLiteral("image1")).toString()));
    }

    std::sort(noMatchPairs.begin(), noMatchPairs.end(),
              [](const QPair<QString, QString> &leftPair, const QPair<QString, QString> &rightPair)
              {
                  if (leftPair.first != rightPair.first)
                  {
                      return leftPair.first < rightPair.first;
                  }
                  return leftPair.second < rightPair.second;
              });
    return noMatchPairs;
}

} // namespace project
} // namespace gui
} // namespace xjw
