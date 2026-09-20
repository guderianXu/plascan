#include "cli_photogrammetry_common.h"

#include "FramePinholeTsaiIO.h"
#include "ProjectCameraIO.h"
#include "camera/project/CameraProjectRuntime.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/models/frame_pinhole/FramePinholeDefinition.h"
#include "camera/models/frame_pinhole/FramePinholeInstance.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "io/PathIO.h"
#include "project/ProjectMetadata.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileInfoList>
#include <QJsonDocument>
#include <QSaveFile>
#include <QTextStream>
#include <QRegularExpression>

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <unordered_set>

namespace xjw::cli
{
    namespace
    {

        bool hasUnquotedComma(const QString& line)
        {
            bool inQuote = false;
            QChar quoteChar;

            for (int index = 0; index < line.size(); ++index)
            {
                const QChar ch = line.at(index);
                if (ch == QLatin1Char('\\'))
                {
                    if (index + 1 < line.size())
                    {
                        const QChar next = line.at(index + 1);
                        const bool escapes_separator =
                            next == QLatin1Char(',') || (inQuote && next == quoteChar) ||
                            (!inQuote && (next == QLatin1Char('\'') || next == QLatin1Char('"')));
                        if (escapes_separator)
                        {
                            ++index;
                        }
                    }
                    continue;
                }
                if (inQuote)
                {
                    if (ch == quoteChar)
                    {
                        inQuote = false;
                    }
                    continue;
                }
                if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
                {
                    inQuote = true;
                    quoteChar = ch;
                    continue;
                }
                if (ch == QLatin1Char(','))
                {
                    return true;
                }
            }

            return false;
        }

        bool appendParsedToken(QStringList* parts, QString* token, bool* hasToken)
        {
            if (!parts || !token || !hasToken)
            {
                return false;
            }

            if (*hasToken || !token->isEmpty())
            {
                parts->append(token->trimmed());
                token->clear();
                *hasToken = false;
            }
            return true;
        }

        bool parseShellTokens(const QString& line, QStringList* parts, QString* errorMessage)
        {
            if (!parts)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("内部错误：列表行输出对象为空");
                }
                return false;
            }

            parts->clear();
            QString token;
            bool hasToken = false;
            bool inQuote = false;
            QChar quoteChar;
            bool escaped = false;

            for (int index = 0; index < line.size(); ++index)
            {
                const QChar ch = line.at(index);
                if (escaped)
                {
                    token.append(ch);
                    hasToken = true;
                    escaped = false;
                    continue;
                }
                if (ch == QLatin1Char('\\'))
                {
                    const bool has_next = index + 1 < line.size();
                    const QChar next = has_next ? line.at(index + 1) : QChar();
                    const bool escapes_next =
                        !has_next || (inQuote ? next == quoteChar
                                              : next.isSpace() || next == QLatin1Char(',') ||
                                                    next == QLatin1Char('\'') || next == QLatin1Char('"'));
                    // 普通反斜杠属于 Windows 路径；仅在分隔符或引号前承担转义含义。
                    if (escapes_next)
                    {
                        escaped = true;
                    }
                    else
                    {
                        token.append(ch);
                    }
                    hasToken = true;
                    continue;
                }
                if (inQuote)
                {
                    if (ch == quoteChar)
                    {
                        inQuote = false;
                    }
                    else
                    {
                        token.append(ch);
                    }
                    hasToken = true;
                    continue;
                }
                if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
                {
                    inQuote = true;
                    quoteChar = ch;
                    hasToken = true;
                    continue;
                }
                if (ch.isSpace())
                {
                    appendParsedToken(parts, &token, &hasToken);
                    continue;
                }

                token.append(ch);
                hasToken = true;
            }

            if (escaped)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("行尾转义字符缺少目标字符");
                }
                return false;
            }
            if (inQuote)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("引号未闭合");
                }
                return false;
            }

            appendParsedToken(parts, &token, &hasToken);
            return true;
        }

        bool parseCsvTokens(const QString& line, QStringList* parts, QString* errorMessage)
        {
            if (!parts)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("内部错误：列表行输出对象为空");
                }
                return false;
            }

            parts->clear();
            QString token;
            bool hasToken = false;
            bool inQuote = false;
            QChar quoteChar;
            bool escaped = false;

            for (int index = 0; index < line.size(); ++index)
            {
                const QChar ch = line.at(index);
                if (escaped)
                {
                    token.append(ch);
                    hasToken = true;
                    escaped = false;
                    continue;
                }
                if (ch == QLatin1Char('\\'))
                {
                    const bool has_next = index + 1 < line.size();
                    const QChar next = has_next ? line.at(index + 1) : QChar();
                    const bool escapes_next =
                        !has_next ||
                        (inQuote ? next == quoteChar
                                 : next == QLatin1Char(',') || next == QLatin1Char('\'') || next == QLatin1Char('"'));
                    if (escapes_next)
                    {
                        escaped = true;
                    }
                    else
                    {
                        token.append(ch);
                    }
                    hasToken = true;
                    continue;
                }
                if (inQuote)
                {
                    if (ch == quoteChar)
                    {
                        if (quoteChar == QLatin1Char('"') && index + 1 < line.size() &&
                            line.at(index + 1) == QLatin1Char('"'))
                        {
                            token.append(ch);
                            hasToken = true;
                            ++index;
                        }
                        else
                        {
                            inQuote = false;
                            hasToken = true;
                        }
                    }
                    else
                    {
                        token.append(ch);
                        hasToken = true;
                    }
                    continue;
                }
                if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
                {
                    inQuote = true;
                    quoteChar = ch;
                    hasToken = true;
                    continue;
                }
                if (ch == QLatin1Char(','))
                {
                    parts->append(token.trimmed());
                    token.clear();
                    hasToken = false;
                    continue;
                }

                token.append(ch);
                hasToken = true;
            }

            if (escaped)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("行尾转义字符缺少目标字符");
                }
                return false;
            }
            if (inQuote)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("引号未闭合");
                }
                return false;
            }

            if (hasToken || !token.isEmpty() || line.endsWith(QLatin1Char(',')))
            {
                parts->append(token.trimmed());
            }
            return true;
        }

    } // namespace

    bool parsePhotogrammetryListLine(const QString& line, QStringList* parts, QString* errorMessage)
    {
        if (hasUnquotedComma(line))
        {
            return parseCsvTokens(line, parts, errorMessage);
        }
        return parseShellTokens(line, parts, errorMessage);
    }

    namespace
    {

        QString withoutMaskSuffix(QString value)
        {
            static const QStringList suffixes = {QStringLiteral("_mask"),
                                                 QStringLiteral("-mask"),
                                                 QStringLiteral(".mask"),
                                                 QStringLiteral("_Mask"),
                                                 QStringLiteral("-Mask")};

            for (const QString& suffix : suffixes)
            {
                if (value.endsWith(suffix))
                {
                    value.chop(suffix.size());
                    return value;
                }
            }
            return value;
        }

        void insertMaskKey(QMap<QString, QString>* index, const QString& key, const QString& maskPath)
        {
            if (!index || key.trimmed().isEmpty() || index->contains(key))
            {
                return;
            }
            index->insert(key, maskPath);
        }

    } // namespace

    bool readPhotogrammetryImageList(const QString& listPath,
                                     const PhotogrammetryListOptions& options,
                                     std::vector<PhotogrammetryInputItem>* items,
                                     QString* errorMessage)
    {
        if (!items)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("内部错误：列表输出对象为空");
            }
            return false;
        }

        QFile file(listPath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法打开列表文件: %1").arg(listPath);
            }
            return false;
        }

        items->clear();
        const QDir listDir(QFileInfo(listPath).absolutePath());
        QTextStream stream(&file);
        int lineNumber = 0;
        while (!stream.atEnd())
        {
            ++lineNumber;
            const QString line = stream.readLine().trimmed();
            if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
            {
                continue;
            }

            QStringList parts;
            QString parseError;
            if (!parsePhotogrammetryListLine(line, &parts, &parseError))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 %3").arg(listPath).arg(lineNumber).arg(parseError);
                }
                return false;
            }

            parts.removeAll(QString());
            if (parts.isEmpty() || parts.size() > 2)
            {
                if (errorMessage)
                {
                    *errorMessage =
                        QStringLiteral("%1:%2 需要 '<image>' 或 '<image> <camera.tsai>'").arg(listPath).arg(lineNumber);
                }
                return false;
            }
            if (parts.size() == 1 && !options.allowImageOnlyRows)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 需要 '<image> <camera.tsai>'").arg(listPath).arg(lineNumber);
                }
                return false;
            }

            PhotogrammetryInputItem item;
            item.imagePath = resolveListToken(parts.at(0), listDir);
            if (parts.size() >= 2)
            {
                item.cameraPath = resolveListToken(parts.at(1), listDir);
                item.hasCameraPath = !item.cameraPath.trimmed().isEmpty();
            }

            if (options.requireExistingImages && !QFileInfo::exists(item.imagePath))
            {
                if (errorMessage)
                {
                    *errorMessage =
                        QStringLiteral("%1:%2 影像不存在: %3").arg(listPath).arg(lineNumber).arg(item.imagePath);
                }
                return false;
            }
            if (item.hasCameraPath && options.requireExistingCameras && !QFileInfo::exists(item.cameraPath))
            {
                if (errorMessage)
                {
                    *errorMessage =
                        QStringLiteral("%1:%2 相机文件不存在: %3").arg(listPath).arg(lineNumber).arg(item.cameraPath);
                }
                return false;
            }
            if (item.hasCameraPath && options.loadCameras)
            {
                std::string camera_error;
                if (!xjw::camera_io::loadFramePinholeNumericStateFromTsaiFile(
                        xjw::common::io::toUtf8Path(item.cameraPath), &item.camera, &camera_error))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("%1:%2 相机读取失败: %3 (%4)")
                                            .arg(listPath)
                                            .arg(lineNumber)
                                            .arg(item.cameraPath, QString::fromStdString(camera_error));
                    }
                    return false;
                }
                item.hasLoadedCamera = true;
            }

            items->push_back(std::move(item));
        }

        if (items->size() < 2)
        {
            if (errorMessage)
            {
                *errorMessage = options.allowImageOnlyRows ? QStringLiteral("至少需要 2 张影像输入")
                                                           : QStringLiteral("至少需要 2 组 image/camera 输入");
            }
            return false;
        }
        return true;
    }

    QStringList imagePaths(const std::vector<PhotogrammetryInputItem>& items)
    {
        QStringList paths;
        paths.reserve(static_cast<int>(items.size()));
        for (const PhotogrammetryInputItem& item : items)
        {
            paths.append(item.imagePath);
        }
        return paths;
    }

    QStringList cameraPathsForService(const std::vector<PhotogrammetryInputItem>& items)
    {
        if (items.empty())
        {
            return {};
        }

        QStringList paths;
        paths.reserve(static_cast<int>(items.size()));
        for (const PhotogrammetryInputItem& item : items)
        {
            if (!item.hasCameraPath || item.cameraPath.trimmed().isEmpty())
            {
                return {};
            }
            paths.append(item.cameraPath);
        }
        return paths;
    }

    bool resolveProjectImageIds(const QJsonObject& projectFiles,
                                const QStringList& images,
                                std::vector<xjw::camera_core::ImageId>* imageIds,
                                QString* errorMessage)
    {
        if (errorMessage)
        {
            errorMessage->clear();
        }
        if (!imageIds)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("内部错误：ImageId 输出对象为空");
            }
            return false;
        }
        const QMap<QString, QJsonObject> imageMetaByPath =
            xjw::common::project::projectImageMetaByPath(projectFiles, true);
        imageIds->clear();
        imageIds->reserve(static_cast<std::size_t>(images.size()));
        std::unordered_set<std::string> seen;
        for (const QString& image : images)
        {
            const QString id = imageMetaByPath.value(cleanAbsolutePath(image))
                                   .value(QStringLiteral("image_uuid"))
                                   .toString()
                                   .trimmed();
            if (id.isEmpty())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("工程中找不到影像身份：%1").arg(image);
                }
                return false;
            }
            try
            {
                xjw::camera_core::ImageId typedId(id.toStdString());
                if (!seen.insert(typedId.value()).second)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("输入影像重复使用 ImageId：%1").arg(id);
                    }
                    return false;
                }
                imageIds->push_back(std::move(typedId));
            }
            catch (const std::exception& exception)
            {
                if (errorMessage)
                {
                    *errorMessage =
                        QStringLiteral("工程影像 ImageId 无效：%1 (%2)").arg(id, QString::fromUtf8(exception.what()));
                }
                return false;
            }
        }
        return true;
    }

    namespace
    {

        std::optional<std::size_t> imageIndexForReferenceName(const QString& name, const QStringList& images)
        {
            const QFileInfo inputInfo(name);
            const QStringList keys = {
                name, cleanAbsolutePath(name), inputInfo.fileName(), inputInfo.completeBaseName()};
            for (const QString& key : keys)
            {
                std::optional<std::size_t> match;
                for (int index = 0; index < images.size(); ++index)
                {
                    const QFileInfo candidate(images.at(index));
                    if (key == images.at(index) || key == cleanAbsolutePath(images.at(index)) ||
                        key == candidate.fileName() || key == candidate.completeBaseName())
                    {
                        if (match)
                        {
                            return std::nullopt;
                        }
                        match = static_cast<std::size_t>(index);
                    }
                }
                if (match)
                {
                    return match;
                }
            }
            return std::nullopt;
        }

        bool makeExternalReferenceGeometry(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& source,
                                           const std::shared_ptr<const xjw::camera_core::CameraInstance>& canonical,
                                           const xjw::camera_core::ImageId& imageId,
                                           std::optional<xjw::camera_reference::ReferenceCameraGeometry>* geometry,
                                           QString* errorMessage)
        {
            if (!canonical || !geometry || !source.validateNumericalState())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("外部参考相机缺少有效的 canonical 相机身份或数值");
                }
                return false;
            }
            if (dynamic_cast<const xjw::camera_models::frame_pinhole::FramePinholeInstance*>(canonical.get()) ==
                nullptr)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("外部 Tsai 参考相机只能绑定 canonical 面阵针孔实例：%1")
                                        .arg(QString::fromStdString(imageId.value()));
                }
                return false;
            }
            xjw::camera_core::ImageSize imageSize = canonical->imageSize();
            if (!imageSize.isValid())
            {
                const auto sourceSize = source.imageSize();
                if (!sourceSize || sourceSize->samples <= 0 || sourceSize->lines <= 0)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("外部参考相机 %1 没有有效影像尺寸")
                                            .arg(QString::fromStdString(imageId.value()));
                    }
                    return false;
                }
                imageSize = {sourceSize->samples, sourceSize->lines};
            }

            try
            {
                const auto definition = xjw::camera_models::frame_pinhole::FramePinholeDefinition::create(
                    xjw::camera_core::CameraDefinitionId("external-reference-" + canonical->instanceId().value()),
                    source.intrinsics(),
                    source.distortion(),
                    xjw::camera_models::frame_pinhole::PixelConvention::PixelCenter,
                    canonical->definition().worldFrame(),
                    source.depthAxisFlipped());
                const auto pose = xjw::camera_core::Pose::create(
                    canonical->definition().worldFrame(), source.cameraCenter(), source.cameraToWorldRotation());
                const auto instance = xjw::camera_models::frame_pinhole::FramePinholeInstance::create(
                    canonical->instanceId(), imageId, definition, imageSize, pose);
                xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
                std::string conversionError;
                if (!xjw::camera_models::frame_pinhole::FramePinholeNumericState::fromInstance(
                        instance, &state, &conversionError))
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("外部参考相机转换失败：%1").arg(QString::fromStdString(conversionError));
                    }
                    return false;
                }
                const auto resolved =
                    xjw::camera_reference::ReferenceCameraGeometry::create(std::move(state), &conversionError);
                if (!resolved)
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("外部参考相机几何无效：%1").arg(QString::fromStdString(conversionError));
                    }
                    return false;
                }
                *geometry = std::move(*resolved);
                return true;
            }
            catch (const std::exception& exception)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("外部参考相机转换失败：%1").arg(QString::fromUtf8(exception.what()));
                }
                return false;
            }
        }

    } // namespace

    bool buildReferenceCameraGeometries(const QJsonObject& projectFiles,
                                        const std::vector<PhotogrammetryInputItem>& items,
                                        const QStringList& images,
                                        const std::vector<xjw::camera_core::ImageId>& imageIds,
                                        xjw::camera_reference::ReferenceCameraGeometryMap* geometries,
                                        QString* errorMessage)
    {
        if (!geometries || images.size() != static_cast<qsizetype>(imageIds.size()) ||
            images.size() != static_cast<qsizetype>(items.size()))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("参考相机输入的影像、ImageId 和相机列表长度不一致");
            }
            return false;
        }
        geometries->clear();
        const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
            projectFiles, xjw::camera_models::makeBuiltinCameraModelRegistry());
        if (!runtime.ok())
        {
            if (errorMessage)
            {
                *errorMessage =
                    QStringLiteral("canonical 相机运行时加载失败：%1").arg(runtime.errors.join(QStringLiteral("; ")));
            }
            return false;
        }
        for (std::size_t index = 0; index < imageIds.size(); ++index)
        {
            const auto lookup = runtime.instances.forImage(imageIds[index]);
            if (!lookup.ok())
            {
                if (items[index].hasLoadedCamera)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("外部参考相机 %1 没有 canonical camera instance，拒绝合成身份")
                                            .arg(images.at(static_cast<int>(index)));
                    }
                    return false;
                }
                continue;
            }
            std::optional<xjw::camera_reference::ReferenceCameraGeometry> geometry;
            if (items[index].hasLoadedCamera)
            {
                if (!makeExternalReferenceGeometry(
                        items[index].camera, lookup.instance, imageIds[index], &geometry, errorMessage))
                {
                    return false;
                }
            }
            else
            {
                xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
                std::string conversionError;
                if (!runtime.framePinholeStateForImage(imageIds[index], &state, &conversionError))
                {
                    continue;
                }
                const auto resolved =
                    xjw::camera_reference::ReferenceCameraGeometry::create(std::move(state), &conversionError);
                if (!resolved)
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("工程参考相机几何无效：%1").arg(QString::fromStdString(conversionError));
                    }
                    return false;
                }
                geometry = std::move(*resolved);
            }
            geometries->emplace(imageIds[index], std::move(*geometry));
        }
        std::string validationError;
        const auto commonFrame = xjw::camera_reference::commonReferenceWorldFrame(
            *geometries, xjw::camera_reference::ReferenceCameraPositionMap{}, &validationError);
        if (!xjw::camera_reference::validateReferenceCameraGeometryMap(*geometries, &validationError) ||
            (!geometries->empty() && !commonFrame))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("参考相机几何无效：%1").arg(QString::fromStdString(validationError));
            }
            return false;
        }
        return true;
    }

    bool readReferencePositionCsv(const QString& csvPath,
                                  const QJsonObject& projectFiles,
                                  const QStringList& images,
                                  const std::vector<xjw::camera_core::ImageId>& imageIds,
                                  xjw::camera_reference::ReferenceCameraPositionMap* positions,
                                  QString* errorMessage)
    {
        if (!positions || images.size() != static_cast<qsizetype>(imageIds.size()))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("内部错误：参考位置输出对象或 ImageId 输入无效");
            }
            return false;
        }

        QFile file(csvPath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法打开参考位置 CSV: %1").arg(csvPath);
            }
            return false;
        }
        const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
            projectFiles, xjw::camera_models::makeBuiltinCameraModelRegistry());
        if (!runtime.ok())
        {
            if (errorMessage)
            {
                *errorMessage =
                    QStringLiteral("canonical 相机运行时加载失败：%1").arg(runtime.errors.join(QStringLiteral("; ")));
            }
            return false;
        }

        positions->clear();
        bool hasRecord = false;
        QTextStream stream(&file);
        int lineNumber = 0;
        while (!stream.atEnd())
        {
            ++lineNumber;
            const QString line = stream.readLine().trimmed();
            if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
            {
                continue;
            }
            const QStringList fields =
                line.split(QRegularExpression(QStringLiteral("\\s*[,;\\t]\\s*")), Qt::KeepEmptyParts);
            if (fields.size() < 4)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 需要 name,x,y,z 四列").arg(csvPath).arg(lineNumber);
                }
                return false;
            }
            bool xOk = false;
            bool yOk = false;
            bool zOk = false;
            const double x = fields.at(1).toDouble(&xOk);
            const double y = fields.at(2).toDouble(&yOk);
            const double z = fields.at(3).toDouble(&zOk);
            const QString first = fields.at(0).trimmed().toLower();
            if (!xOk || !yOk || !zOk)
            {
                if (!hasRecord && (first == QStringLiteral("name") || first == QStringLiteral("image") ||
                                   first == QStringLiteral("label")))
                {
                    continue;
                }
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 x/y/z 必须是有限数值").arg(csvPath).arg(lineNumber);
                }
                return false;
            }
            const QString name = fields.at(0).trimmed();
            const std::array<double, 3> position{x, y, z};
            if (name.isEmpty() || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 名称为空或坐标不是有限数值").arg(csvPath).arg(lineNumber);
                }
                return false;
            }
            const auto imageIndex = imageIndexForReferenceName(name, images);
            if (!imageIndex)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 无法把参考位置绑定到唯一输入影像：%3")
                                        .arg(csvPath)
                                        .arg(lineNumber)
                                        .arg(name);
                }
                return false;
            }
            const auto lookup = runtime.instances.forImage(imageIds[*imageIndex]);
            if (!lookup.ok())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 影像没有 canonical 相机 frame，无法解释参考位置：%3")
                                        .arg(csvPath)
                                        .arg(lineNumber)
                                        .arg(name);
                }
                return false;
            }
            std::string positionError;
            const auto reference = xjw::camera_reference::ReferenceCameraPosition::create(
                imageIds[*imageIndex], lookup.instance->definition().worldFrame(), position, &positionError);
            if (!reference)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 参考位置无效：%3")
                                        .arg(csvPath)
                                        .arg(lineNumber)
                                        .arg(QString::fromStdString(positionError));
                }
                return false;
            }
            if (positions->find(imageIds[*imageIndex]) != positions->cend())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 为同一 ImageId 提供了重复参考位置：%3")
                                        .arg(csvPath)
                                        .arg(lineNumber)
                                        .arg(name);
                }
                return false;
            }
            positions->emplace(imageIds[*imageIndex], std::move(*reference));
            hasRecord = true;
        }
        if (!hasRecord)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("参考位置 CSV 没有有效记录: %1").arg(csvPath);
            }
            return false;
        }
        std::string frameError;
        if (!xjw::camera_reference::validateReferenceCameraPositionMap(*positions, &frameError) ||
            !xjw::camera_reference::commonReferenceWorldFrame(
                xjw::camera_reference::ReferenceCameraGeometryMap{}, *positions, &frameError))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("参考位置坐标系无效：%1").arg(QString::fromStdString(frameError));
            }
            return false;
        }
        return true;
    }

    QJsonObject cameraToJson(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera)
    {
        if (!camera.validateNumericalState())
        {
            return {};
        }
        return xjw::common::project::serializeFramePinholeNumericState(camera);
    }

    QJsonArray inputItemsToJson(const std::vector<PhotogrammetryInputItem>& items)
    {
        QJsonArray array;
        for (const PhotogrammetryInputItem& item : items)
        {
            QJsonObject imageObject;
            imageObject[QStringLiteral("path")] = item.imagePath;
            imageObject[QStringLiteral("name")] = QFileInfo(item.imagePath).fileName();
            if (item.hasCameraPath)
            {
                imageObject[QStringLiteral("camera_path")] = item.cameraPath;
            }
            array.append(imageObject);
        }
        return array;
    }

    QJsonArray inputPairsToJson(const std::vector<PhotogrammetryInputItem>& items)
    {
        QJsonArray array;
        for (const PhotogrammetryInputItem& item : items)
        {
            QJsonObject inputObject;
            inputObject[QStringLiteral("image")] = item.imagePath;
            if (item.hasCameraPath)
            {
                inputObject[QStringLiteral("camera")] = item.cameraPath;
            }
            array.append(inputObject);
        }
        return array;
    }

    QJsonObject projectMetaFromInputItems(const std::vector<PhotogrammetryInputItem>& items)
    {
        QJsonObject meta;
        meta[QStringLiteral("images")] = inputItemsToJson(items);
        return meta;
    }

    QMap<QString, QString> maskPathsFromDirectory(const QString& maskDirectory, const QStringList& images)
    {
        QMap<QString, QString> result;
        const QString trimmedDir = maskDirectory.trimmed();
        if (trimmedDir.isEmpty())
        {
            return result;
        }

        const QDir dir(trimmedDir);
        if (!dir.exists())
        {
            return result;
        }

        QMap<QString, QString> maskIndex;
        const QFileInfoList maskFiles = dir.entryInfoList(QDir::Files | QDir::Readable, QDir::Name);
        for (const QFileInfo& maskInfo : maskFiles)
        {
            const QString path = QDir::cleanPath(maskInfo.absoluteFilePath());
            insertMaskKey(&maskIndex, maskInfo.fileName(), path);
            insertMaskKey(&maskIndex, maskInfo.completeBaseName(), path);
            insertMaskKey(&maskIndex, maskInfo.baseName(), path);
            insertMaskKey(&maskIndex, withoutMaskSuffix(maskInfo.completeBaseName()), path);
            insertMaskKey(&maskIndex, withoutMaskSuffix(maskInfo.baseName()), path);
        }

        for (const QString& image : images)
        {
            const QFileInfo imageInfo(image);
            const QStringList keys = {imageInfo.fileName(), imageInfo.completeBaseName(), imageInfo.baseName()};
            for (const QString& key : keys)
            {
                const auto it = maskIndex.constFind(key);
                if (it != maskIndex.constEnd())
                {
                    result.insert(image, it.value());
                    result.insert(cleanAbsolutePath(image), it.value());
                    result.insert(imageInfo.fileName(), it.value());
                    result.insert(imageInfo.completeBaseName(), it.value());
                    result.insert(imageInfo.baseName(), it.value());
                    break;
                }
            }
        }
        return result;
    }

} // namespace xjw::cli
