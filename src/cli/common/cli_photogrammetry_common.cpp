#include "cli_photogrammetry_common.h"

#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileInfoList>
#include <QJsonDocument>
#include <QSaveFile>
#include <QTextStream>

#include <algorithm>

namespace xjw::cli
{
namespace
{

bool hasUnquotedComma(const QString &line)
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
                const bool escapes_separator = next == QLatin1Char(',')
                    || (inQuote && next == quoteChar)
                    || (!inQuote && (next == QLatin1Char('\'') || next == QLatin1Char('"')));
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

bool appendParsedToken(QStringList *parts, QString *token, bool *hasToken)
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

bool parseShellTokens(const QString &line, QStringList *parts, QString *errorMessage)
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
            const bool escapes_next = !has_next
                || (inQuote ? next == quoteChar
                            : next.isSpace() || next == QLatin1Char(',')
                                || next == QLatin1Char('\'') || next == QLatin1Char('"'));
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

bool parseCsvTokens(const QString &line, QStringList *parts, QString *errorMessage)
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
            const bool escapes_next = !has_next
                || (inQuote ? next == quoteChar
                            : next == QLatin1Char(',')
                                || next == QLatin1Char('\'') || next == QLatin1Char('"'));
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
                if (quoteChar == QLatin1Char('"')
                    && index + 1 < line.size()
                    && line.at(index + 1) == QLatin1Char('"'))
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

bool parsePhotogrammetryListLine(const QString &line,
                                 QStringList *parts,
                                 QString *errorMessage)
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
    static const QStringList suffixes = {
        QStringLiteral("_mask"),
        QStringLiteral("-mask"),
        QStringLiteral(".mask"),
        QStringLiteral("_Mask"),
        QStringLiteral("-Mask")
    };

    for (const QString &suffix : suffixes)
    {
        if (value.endsWith(suffix))
        {
            value.chop(suffix.size());
            return value;
        }
    }
    return value;
}

void insertMaskKey(QMap<QString, QString> *index, const QString &key, const QString &maskPath)
{
    if (!index || key.trimmed().isEmpty() || index->contains(key))
    {
        return;
    }
    index->insert(key, maskPath);
}

} // namespace

bool readPhotogrammetryImageList(const QString &listPath,
                                 const PhotogrammetryListOptions &options,
                                 std::vector<PhotogrammetryInputItem> *items,
                                 QString *errorMessage)
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
                *errorMessage = QStringLiteral("%1:%2 需要 '<image>' 或 '<image> <camera.tsai>'")
                                    .arg(listPath)
                                    .arg(lineNumber);
            }
            return false;
        }
        if (parts.size() == 1 && !options.allowImageOnlyRows)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("%1:%2 需要 '<image> <camera.tsai>'")
                                    .arg(listPath)
                                    .arg(lineNumber);
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
                *errorMessage = QStringLiteral("%1:%2 影像不存在: %3")
                                    .arg(listPath)
                                    .arg(lineNumber)
                                    .arg(item.imagePath);
            }
            return false;
        }
        if (item.hasCameraPath && options.requireExistingCameras && !QFileInfo::exists(item.cameraPath))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("%1:%2 相机文件不存在: %3")
                                    .arg(listPath)
                                    .arg(lineNumber)
                                    .arg(item.cameraPath);
            }
            return false;
        }
        if (item.hasCameraPath && options.loadCameras)
        {
            if (!item.camera.loadFromFile(xjw::common::io::toUtf8Path(item.cameraPath)) || !item.camera.isValid())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("%1:%2 相机读取失败: %3")
                                        .arg(listPath)
                                        .arg(lineNumber)
                                        .arg(item.cameraPath);
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
            *errorMessage = options.allowImageOnlyRows
                ? QStringLiteral("至少需要 2 张影像输入")
                : QStringLiteral("至少需要 2 组 image/camera 输入");
        }
        return false;
    }
    return true;
}

QStringList imagePaths(const std::vector<PhotogrammetryInputItem> &items)
{
    QStringList paths;
    paths.reserve(static_cast<int>(items.size()));
    for (const PhotogrammetryInputItem &item : items)
    {
        paths.append(item.imagePath);
    }
    return paths;
}

QStringList cameraPathsForService(const std::vector<PhotogrammetryInputItem> &items)
{
    if (items.empty())
    {
        return {};
    }

    QStringList paths;
    paths.reserve(static_cast<int>(items.size()));
    for (const PhotogrammetryInputItem &item : items)
    {
        if (!item.hasCameraPath || item.cameraPath.trimmed().isEmpty())
        {
            return {};
        }
        paths.append(item.cameraPath);
    }
    return paths;
}

QMap<QString, xjw::Camera> referenceCameraMap(const std::vector<PhotogrammetryInputItem> &items)
{
    QMap<QString, xjw::Camera> cameras;
    for (const PhotogrammetryInputItem &item : items)
    {
        if (!item.hasLoadedCamera || !item.camera.isValid())
        {
            continue;
        }
        cameras.insert(item.imagePath, item.camera);
        cameras.insert(cleanAbsolutePath(item.imagePath), item.camera);
    }
    return cameras;
}

QJsonObject cameraToJson(const xjw::Camera &camera)
{
    QJsonObject object;
    if (!camera.isValid())
    {
        return object;
    }

    const auto intrinsics = camera.intrinsics();
    const auto distortion = camera.distortion();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    object[QStringLiteral("model")] = QStringLiteral("tsai");
    object[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    object[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    object[QStringLiteral("pitch")] = camera.pixelPitch();
    object[QStringLiteral("fu")] = camera.focalXMillimeters();
    object[QStringLiteral("fv")] = camera.focalYMillimeters();
    object[QStringLiteral("cu")] = camera.principalXMillimeters();
    object[QStringLiteral("cv")] = camera.principalYMillimeters();
    object[QStringLiteral("k1")] = distortion.radialK1;
    object[QStringLiteral("k2")] = distortion.radialK2;
    object[QStringLiteral("k3")] = distortion.radialK3;
    object[QStringLiteral("p1")] = distortion.tangentialP1;
    object[QStringLiteral("p2")] = distortion.tangentialP2;
    object[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
    object[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
    object[QStringLiteral("depth_axis_flipped")] = camera.depthAxisFlipped();

    QJsonArray centerArray;
    for (const double value : center)
    {
        centerArray.append(value);
    }
    object[QStringLiteral("C")] = centerArray;

    QJsonArray rotationArray;
    for (const double value : rotation)
    {
        rotationArray.append(value);
    }
    object[QStringLiteral("R")] = rotationArray;
    return object;
}

QJsonArray inputItemsToJson(const std::vector<PhotogrammetryInputItem> &items)
{
    QJsonArray array;
    for (const PhotogrammetryInputItem &item : items)
    {
        QJsonObject imageObject;
        imageObject[QStringLiteral("path")] = item.imagePath;
        imageObject[QStringLiteral("name")] = QFileInfo(item.imagePath).fileName();
        if (item.hasCameraPath)
        {
            imageObject[QStringLiteral("camera_path")] = item.cameraPath;
        }
        if (item.hasLoadedCamera && item.camera.isValid())
        {
            imageObject[QStringLiteral("camera")] = cameraToJson(item.camera);
        }
        array.append(imageObject);
    }
    return array;
}

QJsonArray inputPairsToJson(const std::vector<PhotogrammetryInputItem> &items)
{
    QJsonArray array;
    for (const PhotogrammetryInputItem &item : items)
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

QJsonObject projectMetaFromInputItems(const std::vector<PhotogrammetryInputItem> &items)
{
    QJsonObject meta;
    meta[QStringLiteral("images")] = inputItemsToJson(items);
    return meta;
}

QMap<QString, QString> maskPathsFromDirectory(const QString &maskDirectory, const QStringList &images)
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
    for (const QFileInfo &maskInfo : maskFiles)
    {
        const QString path = QDir::cleanPath(maskInfo.absoluteFilePath());
        insertMaskKey(&maskIndex, maskInfo.fileName(), path);
        insertMaskKey(&maskIndex, maskInfo.completeBaseName(), path);
        insertMaskKey(&maskIndex, maskInfo.baseName(), path);
        insertMaskKey(&maskIndex, withoutMaskSuffix(maskInfo.completeBaseName()), path);
        insertMaskKey(&maskIndex, withoutMaskSuffix(maskInfo.baseName()), path);
    }

    for (const QString &image : images)
    {
        const QFileInfo imageInfo(image);
        const QStringList keys = {
            imageInfo.fileName(),
            imageInfo.completeBaseName(),
            imageInfo.baseName()
        };
        for (const QString &key : keys)
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
