// =============================================================================
// 文件: CameraModel3DDialog.cpp
// 功能: 相机三维模型可视化对话框实现
// 内容:
//   - CameraSceneWidget：Qt RHI/Vulkan 三维渲染控件
//       · 点云 / PLY 模型 / 相机平面卡片渲染（QRhiBuffer + .qsb shader）
//       · Arcball 自由旋转 + 单轴环旋转（X/Y/Z Gizmo）
//       · 中键平移、滚轮缩放
//       · 透明 QWidget 覆盖层（Gizmo 环、坐标轴、相机卡片、欧拉角）
//   - CameraModel3DDialog：对话框 UI + 从 ProjectManager 读取相机姿态
// =============================================================================
#include "camera/CameraModel3DDialog.h"
#include "CameraSceneViewMath.h"
#include "ObjRenderPreparation.h"
#include "ui_CameraModel3DDialog.h"

#include "ProjectManager.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "LayerImageLoader.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "io/PathIO.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPainter>
#include <QPainterPath>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QCursor>
#include <QIODevice>
#include <QPixmap>
#include <QVector2D>
#include <QJsonObject>
#include <QJsonArray>
#include <QMatrix4x4>
#include <QtMath>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QElapsedTimer>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLineF>
#include <QMetaObject>
#include <QPointer>
#include <QRegularExpression>
#include <QSet>
#include <rhi/qrhi.h>
#include <rhi/qshader.h>
#include <QSizePolicy>
#include <QStringList>
#include <QTextStream>
#include <QWidget>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <plapoint/io/xyz_io.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/obj_io.h>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace
{

constexpr quint64 kMaxDirectPlyVertices = 2'000'000;
constexpr quint64 kMinPreviewPlyVertices = 1'000'000;
constexpr quint64 kDefaultPreviewPlyVertices = 3'000'000;
constexpr quint64 kMaxPreviewPlyVertices = 5'000'000;
constexpr quint64 kPreviewMemoryReserveBytes = 2ull * 1024ull * 1024ull * 1024ull;
constexpr quint64 kEstimatedPreviewBytesPerVertex = 128;

using PlyPreviewProgressCallback = std::function<void(int, const QString &)>;

struct PointCloudSaveResult
{
    bool success = false;
    QString path;
    QString errorMessage;
    int pointCount = 0;
};

RenderCloud cloneRenderCloud(const RenderCloud &src)
{
    RenderCloud dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        for (int d = 0; d < 3; ++d)
            dst.points()(static_cast<plamatrix::Index>(i), d) = src.points()(static_cast<plamatrix::Index>(i), d);
    if (src.hasColors())
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> c(src.size(), 3);
        for (size_t i = 0; i < src.size(); ++i)
            for (int d = 0; d < 3; ++d)
                c(static_cast<plamatrix::Index>(i), d) = src.colors()->getValue(static_cast<plamatrix::Index>(i), d);
        dst.setColors(std::move(c));
    }
    if (src.hasNormals())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> n(src.size(), 3);
        for (size_t i = 0; i < src.size(); ++i)
            for (int d = 0; d < 3; ++d)
                n(static_cast<plamatrix::Index>(i), d) = src.normals()->getValue(static_cast<plamatrix::Index>(i), d);
        dst.setNormals(std::move(n));
    }
    if (src.hasTextureCoords())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> t(src.size(), 2);
        for (size_t i = 0; i < src.size(); ++i)
            for (int d = 0; d < 2; ++d)
                t(static_cast<plamatrix::Index>(i), d) = src.textureCoords()->getValue(static_cast<plamatrix::Index>(i), d);
        dst.setTextureCoords(std::move(t));
    }
    if (src.hasFaces())
    {
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> f(src.faces()->rows(), 3);
        for (int i = 0; i < src.faces()->rows(); ++i)
            for (int d = 0; d < 3; ++d)
                f(i, d) = src.faces()->getValue(i, d);
        dst.setFaces(std::move(f));
    }
    if (src.hasFaceTextureIndices())
    {
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> ft(
            src.faceTextureIndices()->rows(), 3);
        for (int i = 0; i < src.faceTextureIndices()->rows(); ++i)
        {
            for (int d = 0; d < 3; ++d)
            {
                ft(i, d) = src.faceTextureIndices()->getValue(i, d);
            }
        }
        dst.setFaceTextureIndices(std::move(ft));
    }
    dst.setMaterialLibraryFile(src.materialLibraryFile());
    dst.setTextureImageFile(src.textureImageFile());
    return dst;
}

struct ObjLoadResult
{
    std::shared_ptr<RenderCloud> cloud;
    ObjRenderPreparation renderPreparation;
    QImage textureImage;
    QString texturePath;
    QString textureWarning;
    qint64 parseElapsedMs = 0;
    qint64 prepareElapsedMs = 0;
};

QString firstDiffuseTexturePath(const QString &obj_path, const RenderCloud &cloud)
{
    QString material_name = xjw::common::io::fromUtf8Path(cloud.materialLibraryFile()).trimmed();
    if (material_name.isEmpty())
    {
        return QString();
    }

    QFileInfo material_info(material_name);
    const QString material_path = material_info.isAbsolute()
        ? material_info.absoluteFilePath()
        : QDir(QFileInfo(obj_path).absolutePath()).filePath(material_name);
    QFile material_file(material_path);
    if (!material_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return QString();
    }

    QTextStream stream(&material_file);
    while (!stream.atEnd())
    {
        const QString line = stream.readLine().trimmed();
        if (!line.startsWith(QStringLiteral("map_Kd")))
        {
            continue;
        }
        const QString texture_reference = line.mid(6).trimmed();
        if (texture_reference.isEmpty())
        {
            return QString();
        }
        QFileInfo texture_info(texture_reference);
        return QDir::cleanPath(texture_info.isAbsolute()
            ? texture_info.absoluteFilePath()
            : QDir(QFileInfo(material_path).absolutePath()).filePath(texture_reference));
    }
    return QString();
}

ObjLoadResult loadObjWithMaterialTexture(const QString &obj_path,
                                         const std::function<void(int, const QString &)> &progress)
{
    ObjLoadResult result;
    QElapsedTimer timer;
    timer.start();
    result.cloud = plapoint::io::readObj<float>(
        xjw::common::io::toNativeNarrowPath(obj_path));
    result.parseElapsedMs = timer.elapsed();
    if (!result.cloud || result.cloud->size() == 0)
    {
        return result;
    }
    if (!result.cloud->hasTextureCoords() || !result.cloud->hasFaceTextureIndices())
    {
        result.textureWarning = QStringLiteral("OBJ 未包含完整的面级 UV，使用顶点颜色显示");
    }
    else
    {
        if (progress)
        {
            progress(82, QStringLiteral("正在读取 OBJ 材质纹理..."));
        }
        result.texturePath = firstDiffuseTexturePath(obj_path, *result.cloud);
        if (result.texturePath.isEmpty())
        {
            result.textureWarning = QStringLiteral("OBJ 的 MTL 未指定漫反射纹理，使用顶点颜色显示");
        }
        else if (!QFileInfo::exists(result.texturePath)
                 || !result.textureImage.load(result.texturePath))
        {
            result.textureWarning = QStringLiteral("无法读取 OBJ 纹理图像: %1")
                                        .arg(result.texturePath);
            result.textureImage = QImage();
        }
    }

    if (!result.textureImage.isNull())
    {
        result.cloud->setTextureImageFile(
            xjw::common::io::toUtf8Path(result.texturePath));
    }
    if (progress)
    {
        progress(88, QStringLiteral("正在准备 OBJ 渲染数据..."));
    }
    timer.restart();
    result.renderPreparation = prepareObjRenderData(
        *result.cloud, !result.textureImage.isNull());
    result.prepareElapsedMs = timer.elapsed();
    return result;
}

PointCloudSaveResult savePointCloudSnapshot(const QString &path,
                                            const RenderCloud &cloud)
{
    PointCloudSaveResult result;
    result.path = path;
    result.pointCount = static_cast<int>(cloud.size());
    if (path.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("当前点云来源未知，无法覆盖保存。");
        return result;
    }

    const std::string nativePath = xjw::common::io::toNativeNarrowPath(path);
    const bool isPly =
        nativePath.size() >= 4
        && (nativePath.substr(nativePath.size() - 4) == ".ply"
            || nativePath.substr(nativePath.size() - 4) == ".PLY");
    try
    {
        if (isPly)
        {
            plapoint::io::writePly<float>(nativePath, cloud);
        }
        else
        {
            plapoint::io::writeXyz<float>(nativePath, cloud);
        }
        result.success = true;
    }
    catch (const std::exception &error)
    {
        result.errorMessage = QString::fromStdString(error.what());
    }
    return result;
}

enum class PlyScalarType
{
    Invalid,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64
};

struct PlyPreviewProperty
{
    QString name;
    PlyScalarType type = PlyScalarType::Invalid;
    int offset = 0;
    int size = 0;
};

struct PlyPreviewHeader
{
    bool valid = false;
    bool binaryLittleEndian = false;
    quint64 vertexCount = 0;
    quint64 faceCount = 0;
    qint64 dataStartOffset = 0;
    int vertexStride = 0;
    int xProperty = -1;
    int yProperty = -1;
    int zProperty = -1;
    int redProperty = -1;
    int greenProperty = -1;
    int blueProperty = -1;
    QVector<PlyPreviewProperty> properties;
};

struct PlyPreviewResult
{
    PlyPreviewHeader header;
    std::shared_ptr<RenderCloud> cloud;
    QString error;
};

PlyScalarType plyScalarTypeFromName(const QString &name)
{
    const QString lower = name.toLower();
    if (lower == QLatin1String("char") || lower == QLatin1String("int8"))
        return PlyScalarType::Int8;
    if (lower == QLatin1String("uchar")
        || lower == QLatin1String("uint8")
        || lower == QLatin1String("unsigned_char"))
        return PlyScalarType::UInt8;
    if (lower == QLatin1String("short") || lower == QLatin1String("int16"))
        return PlyScalarType::Int16;
    if (lower == QLatin1String("ushort")
        || lower == QLatin1String("uint16")
        || lower == QLatin1String("unsigned_short"))
        return PlyScalarType::UInt16;
    if (lower == QLatin1String("int") || lower == QLatin1String("int32"))
        return PlyScalarType::Int32;
    if (lower == QLatin1String("uint")
        || lower == QLatin1String("uint32")
        || lower == QLatin1String("unsigned_int"))
        return PlyScalarType::UInt32;
    if (lower == QLatin1String("float") || lower == QLatin1String("float32"))
        return PlyScalarType::Float32;
    if (lower == QLatin1String("double") || lower == QLatin1String("float64"))
        return PlyScalarType::Float64;
    return PlyScalarType::Invalid;
}

int plyScalarTypeSize(PlyScalarType type)
{
    switch (type)
    {
    case PlyScalarType::Int8:
    case PlyScalarType::UInt8:
        return 1;
    case PlyScalarType::Int16:
    case PlyScalarType::UInt16:
        return 2;
    case PlyScalarType::Int32:
    case PlyScalarType::UInt32:
    case PlyScalarType::Float32:
        return 4;
    case PlyScalarType::Float64:
        return 8;
    default:
        return 0;
    }
}

quint64 availableSystemMemoryBytes()
{
#ifdef Q_OS_WIN
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
    {
        return static_cast<quint64>(status.ullAvailPhys);
    }
#elif defined(Q_OS_LINUX)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    quint64 valueKb = 0;
    std::string unit;
    quint64 availableKb = 0;
    while (meminfo >> key >> valueKb >> unit)
    {
        if (key == "MemAvailable:")
        {
            availableKb = valueKb;
            break;
        }
    }
    if (availableKb > 0)
    {
        return availableKb * 1024ull;
    }
#endif
    return 0;
}

quint64 choosePreviewPlyVertexLimit()
{
    const quint64 available = availableSystemMemoryBytes();
    if (available == 0)
    {
        return kDefaultPreviewPlyVertices;
    }

    if (available >= kPreviewMemoryReserveBytes + kMaxPreviewPlyVertices * kEstimatedPreviewBytesPerVertex)
    {
        return kMaxPreviewPlyVertices;
    }
    if (available >= kPreviewMemoryReserveBytes + kDefaultPreviewPlyVertices * kEstimatedPreviewBytesPerVertex)
    {
        return kDefaultPreviewPlyVertices;
    }
    if (available > kPreviewMemoryReserveBytes)
    {
        const quint64 budget = (available - kPreviewMemoryReserveBytes) / kEstimatedPreviewBytesPerVertex;
        return qBound(kMinPreviewPlyVertices, budget, kDefaultPreviewPlyVertices);
    }

    return kMinPreviewPlyVertices;
}

bool parsePlyPreviewHeader(const QString &plyPath, PlyPreviewHeader *header, QString *error)
{
    if (!header)
    {
        return false;
    }

    *header = PlyPreviewHeader();
    QFile file(plyPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (error) *error = QStringLiteral("无法打开 PLY 文件: %1").arg(plyPath);
        return false;
    }

    const QByteArray magic = file.readLine();
    if (magic.trimmed() != QByteArrayLiteral("ply"))
    {
        if (error) *error = QStringLiteral("不是有效的 PLY 文件: %1").arg(plyPath);
        return false;
    }

    bool inVertexElement = false;
    while (!file.atEnd())
    {
        const QByteArray rawLine = file.readLine();
        const QString line = QString::fromLatin1(rawLine).trimmed();
        if (line == QLatin1String("end_header"))
        {
            header->dataStartOffset = file.pos();
            break;
        }

        const QStringList parts = line.split(QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts);
        if (parts.isEmpty())
        {
            continue;
        }

        if (parts.size() >= 2 && parts[0] == QLatin1String("format"))
        {
            header->binaryLittleEndian = parts[1] == QLatin1String("binary_little_endian");
            continue;
        }

        if (parts.size() >= 3 && parts[0] == QLatin1String("element"))
        {
            inVertexElement = parts[1] == QLatin1String("vertex");
            if (inVertexElement)
            {
                bool ok = false;
                header->vertexCount = parts[2].toULongLong(&ok);
                if (!ok)
                {
                    if (error) *error = QStringLiteral("PLY 顶点数量无效: %1").arg(parts[2]);
                    return false;
                }
            }
            else if (parts[1] == QLatin1String("face"))
            {
                bool ok = false;
                header->faceCount = parts[2].toULongLong(&ok);
                if (!ok)
                {
                    if (error) *error = QStringLiteral("PLY 面片数量无效: %1").arg(parts[2]);
                    return false;
                }
            }
            continue;
        }

        if (inVertexElement && parts.size() >= 3 && parts[0] == QLatin1String("property"))
        {
            if (parts[1] == QLatin1String("list"))
            {
                if (error) *error = QStringLiteral("暂不支持顶点元素中的 list property");
                return false;
            }

            PlyPreviewProperty property;
            property.type = plyScalarTypeFromName(parts[1]);
            property.size = plyScalarTypeSize(property.type);
            property.name = parts[2].toLower();
            property.offset = header->vertexStride;
            if (property.size <= 0)
            {
                if (error) *error = QStringLiteral("不支持的 PLY 属性类型: %1").arg(parts[1]);
                return false;
            }

            const int index = header->properties.size();
            if (property.name == QLatin1String("x")) header->xProperty = index;
            else if (property.name == QLatin1String("y")) header->yProperty = index;
            else if (property.name == QLatin1String("z")) header->zProperty = index;
            else if (property.name == QLatin1String("red") || property.name == QLatin1String("r"))
                header->redProperty = index;
            else if (property.name == QLatin1String("green") || property.name == QLatin1String("g"))
                header->greenProperty = index;
            else if (property.name == QLatin1String("blue") || property.name == QLatin1String("b"))
                header->blueProperty = index;

            header->properties.push_back(property);
            header->vertexStride += property.size;
        }
    }

    header->valid = header->dataStartOffset > 0
        && header->binaryLittleEndian
        && header->vertexCount > 0
        && header->vertexStride > 0
        && header->xProperty >= 0
        && header->yProperty >= 0
        && header->zProperty >= 0;

    if (!header->valid && error)
    {
        *error = QStringLiteral("PLY 头缺少二进制小端顶点 XYZ 信息");
    }
    return header->valid;
}

template <typename T>
T readUnalignedValue(const QByteArray &record, int offset)
{
    T value{};
    std::memcpy(&value, record.constData() + offset, sizeof(T));
    return value;
}

double readPlyScalarAsDouble(const QByteArray &record, const PlyPreviewProperty &property)
{
    switch (property.type)
    {
    case PlyScalarType::Int8:    return readUnalignedValue<qint8>(record, property.offset);
    case PlyScalarType::UInt8:   return readUnalignedValue<quint8>(record, property.offset);
    case PlyScalarType::Int16:   return readUnalignedValue<qint16>(record, property.offset);
    case PlyScalarType::UInt16:  return readUnalignedValue<quint16>(record, property.offset);
    case PlyScalarType::Int32:   return readUnalignedValue<qint32>(record, property.offset);
    case PlyScalarType::UInt32:  return readUnalignedValue<quint32>(record, property.offset);
    case PlyScalarType::Float32: return readUnalignedValue<float>(record, property.offset);
    case PlyScalarType::Float64: return readUnalignedValue<double>(record, property.offset);
    default:                     return 0.0;
    }
}

quint8 readPlyColorAsByte(const QByteArray &record, const PlyPreviewProperty &property)
{
    if (property.type == PlyScalarType::UInt8)
    {
        return readUnalignedValue<quint8>(record, property.offset);
    }

    const double raw = readPlyScalarAsDouble(record, property);
    const double scaled = raw <= 1.0 ? raw * 255.0 : raw;
    return static_cast<quint8>(qBound(0, static_cast<int>(std::lround(scaled)), 255));
}

PlyPreviewResult readBinaryPlyPreview(const QString &plyPath,
                                      const PlyPreviewProgressCallback &progress = PlyPreviewProgressCallback())
{
    PlyPreviewResult preview;
    auto reportProgress = [&](int percent, const QString &statusText)
    {
        if (progress)
        {
            progress(qBound(0, percent, 100), statusText);
        }
    };

    reportProgress(1, QStringLiteral("正在解析 PLY 头..."));
    if (!parsePlyPreviewHeader(plyPath, &preview.header, &preview.error))
    {
        return preview;
    }

    if (preview.header.vertexCount <= kMaxDirectPlyVertices || preview.header.faceCount > 0)
    {
        return preview;
    }

    const quint64 previewVertexLimit = choosePreviewPlyVertexLimit();
    QFile file(plyPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        preview.error = QStringLiteral("无法打开 PLY 文件: %1").arg(plyPath);
        return preview;
    }

    const quint64 sampleStride = qMax<quint64>(
        1,
        (preview.header.vertexCount + previewVertexLimit - 1) / previewVertexLimit);
    const quint64 sampleCount = qMin<quint64>(
        previewVertexLimit,
        (preview.header.vertexCount + sampleStride - 1) / sampleStride);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(sampleCount), 3);
    const bool hasColors = preview.header.redProperty >= 0
        && preview.header.greenProperty >= 0
        && preview.header.blueProperty >= 0;
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(
        static_cast<plamatrix::Index>(sampleCount), 3);

    const qint64 recordOffset = preview.header.dataStartOffset;
    quint64 outIndex = 0;
    int lastProgressPercent = -1;
    reportProgress(3, QStringLiteral("正在抽样密集点云预览..."));
    for (quint64 i = 0; i < preview.header.vertexCount && outIndex < sampleCount; i += sampleStride)
    {
        if (!file.seek(recordOffset + static_cast<qint64>(i) * preview.header.vertexStride))
        {
            break;
        }

        const QByteArray record = file.read(preview.header.vertexStride);
        if (record.size() != preview.header.vertexStride)
        {
            break;
        }

        const auto row = static_cast<plamatrix::Index>(outIndex);
        points(row, 0) = static_cast<float>(
            readPlyScalarAsDouble(record, preview.header.properties[preview.header.xProperty]));
        points(row, 1) = static_cast<float>(
            readPlyScalarAsDouble(record, preview.header.properties[preview.header.yProperty]));
        points(row, 2) = static_cast<float>(
            readPlyScalarAsDouble(record, preview.header.properties[preview.header.zProperty]));

        if (hasColors)
        {
            colors(row, 0) = readPlyColorAsByte(record, preview.header.properties[preview.header.redProperty]);
            colors(row, 1) = readPlyColorAsByte(record, preview.header.properties[preview.header.greenProperty]);
            colors(row, 2) = readPlyColorAsByte(record, preview.header.properties[preview.header.blueProperty]);
        }
        ++outIndex;

        if ((outIndex & 0x3ffu) == 0 || outIndex == sampleCount)
        {
            const int percent = 3 + static_cast<int>((outIndex * 92) / qMax<quint64>(1, sampleCount));
            if (percent != lastProgressPercent)
            {
                lastProgressPercent = percent;
                reportProgress(percent,
                               QStringLiteral("正在加载密集点云预览 %1/%2 点")
                                   .arg(outIndex)
                                   .arg(sampleCount));
            }
        }
    }

    if (outIndex == 0)
    {
        preview.error = QStringLiteral("PLY 预览抽样没有读到有效顶点");
        return preview;
    }

    if (outIndex < sampleCount)
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> trimmedPoints(
            static_cast<plamatrix::Index>(outIndex), 3);
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> trimmedColors(
            static_cast<plamatrix::Index>(outIndex), 3);
        for (quint64 i = 0; i < outIndex; ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            for (int c = 0; c < 3; ++c)
            {
                trimmedPoints(row, c) = points(row, c);
                if (hasColors)
                {
                    trimmedColors(row, c) = colors(row, c);
                }
            }
        }
        points = std::move(trimmedPoints);
        if (hasColors)
        {
            colors = std::move(trimmedColors);
        }
    }

    preview.cloud = std::make_shared<RenderCloud>(std::move(points));
    if (hasColors)
    {
        preview.cloud->setColors(std::move(colors));
    }
    reportProgress(96, QStringLiteral("正在上传密集点云预览..."));
    return preview;
}

QShader loadSceneShader(const QString &resourcePath, QString *errorMessage)
{
    QFile file(resourcePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Vulkan 渲染着色器资源缺失：%1").arg(resourcePath);
        }
        return {};
    }

    QShader shader = QShader::fromSerialized(file.readAll());
    if (!shader.isValid() && errorMessage)
    {
        *errorMessage = QStringLiteral("Vulkan 渲染着色器加载失败：%1").arg(resourcePath);
    }
    return shader;
}

} // namespace

class CameraSceneOverlayWidget : public QWidget
{
public:
    explicit CameraSceneOverlayWidget(CameraSceneWidget *scene)
        : QWidget(scene)
        , _scene(scene)
    {
        setAttribute(Qt::WA_TransparentForMouseEvents, true);
        setAttribute(Qt::WA_NoSystemBackground, true);
        setAttribute(Qt::WA_TranslucentBackground, true);
        setAutoFillBackground(false);
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        if (!_scene)
        {
            return;
        }

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);
        _scene->paintOverlay(painter);
    }

private:
    CameraSceneWidget *_scene = nullptr;
};

// 构造函数：设置基础可用尺寸，启用鼠标追踪（悬停检测需要），
// 设置默认视角为俯仰 -25°、偏航 35°（斜上方看向场景）
CameraSceneWidget::CameraSceneWidget(QWidget *parent)
    : QRhiWidget(parent)
{
    setApi(QRhiWidget::Api::Vulkan);
    setSampleCount(4);

    setMinimumSize(240, 160);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setMouseTracking(true); // 启用鼠标追踪，以便在无按键时检测悬停轴
    _viewRot = QQuaternion::fromEulerAngles(-25.0f, 35.0f, 0.0f); // 默认斜视角
    setFocusPolicy(Qt::StrongFocus);
    updateCursor();

    _overlayWidget = new CameraSceneOverlayWidget(this);
    _overlayWidget->setGeometry(rect());
    _overlayWidget->show();
    _overlayWidget->raise();

    connect(this, &CameraSceneWidget::plyLoadProgressChanged,
            this, [this](int generation, int percent, const QString &statusText)
    {
        if (generation != _loadGen)
        {
            return;
        }
        if (!_loading && percent < 100)
        {
            return;
        }
        _loading = percent < 100;
        _plyLoadProgressPercent = qBound(0, percent, 100);
        _plyLoadProgressText = statusText;
        update();
    }, Qt::QueuedConnection);
}

CameraSceneWidget::~CameraSceneWidget() = default;

// 设置要渲染的相机姿态列表。
// 调用后触发重绘，场景中每个姿态点将绘制相机平面卡片和名称标注。
void CameraSceneWidget::setCameraPoses(const QVector<CameraPose> &poses)
{
    _poses.clear();
    _poses.reserve(poses.size());
    QSet<QString> cameraKeys;
    for (const CameraPose &pose : poses)
    {
        QString cameraKey = normalizedCameraPath(pose.imagePath);
        if (cameraKey.isEmpty())
        {
            cameraKey = pose.name.trimmed();
        }
        if (!cameraKey.isEmpty())
        {
#ifdef Q_OS_WIN
            cameraKey = cameraKey.toCaseFolded();
#endif
            if (cameraKeys.contains(cameraKey))
            {
                continue;
            }
            cameraKeys.insert(cameraKey);
        }
        _poses.push_back(pose);
    }
    _activeCameraImagePoseIndex = -1;
    _cacheDirty = true; // 相机位置变更，缓存失效
    ++_cameraImageLoadGeneration;
    _cameraImageCache.clear();
    _cameraImageLoadsInFlight.clear();
    _cameraImageLoadFailures.clear();
    _thumbnailPipeline.resourcesDirty = true;
    if (_showCameraImage)
    {
        updateActiveCameraForView();
    }
    update(); // 触发 Vulkan 帧重绘
}

void CameraSceneWidget::setShowGizmo(bool show)
{
    if (_showGizmo != show)
    {
        _showGizmo = show;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameras(bool show)
{
    if (_showCameras != show)
    {
        _showCameras = show;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameraThumbnails(bool show)
{
    if (_showCameraThumbnails != show)
    {
        _showCameraThumbnails = show;
        _thumbnailPipeline.resourcesDirty = true;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameraLocalAxes(bool show)
{
    if (_showCameraLocalAxes != show)
    {
        _showCameraLocalAxes = show;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameraImage(bool show)
{
    if (_showCameraImage != show)
    {
        _showCameraImage = show;
        if (_showCameraImage)
        {
            updateActiveCameraForView();
            if (_cameraImageLocked)
            {
                refreshLockedCameraImage();
            }
        }
        else
        {
            _activeCameraImagePoseIndex = -1;
        }
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setCameraImagePlaneMode(CameraImagePlaneMode mode)
{
    if (_cameraImagePlaneMode != mode)
    {
        _cameraImagePlaneMode = mode;
        _showCameraThumbnails = mode == CameraImagePlaneMode::Thumbnail;
        setShowCameraImage(mode == CameraImagePlaneMode::Image);
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setCameraImageDisplayLayer(CameraImageDisplayLayer layer)
{
    if (_cameraImageDisplayLayer != layer)
    {
        _cameraImageDisplayLayer = layer;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setCameraImageLocked(bool locked)
{
    if (_cameraImageLocked == locked)
    {
        return;
    }

    _cameraImageLocked = locked;
    if (_cameraImageLocked)
    {
        refreshLockedCameraImage();
    }
    else
    {
        _lockedCameraImagePath.clear();
        _lockedCameraImageName.clear();
        updateActiveCameraForView();
    }
    updateCameraOverlay();
}

void CameraSceneWidget::setHighlightedCameraPath(const QString &imagePath)
{
    const QString normalizedPath = normalizedCameraPath(imagePath);
    if (_highlightedCameraPath == normalizedPath && _highlightedCameraName.isEmpty())
    {
        return;
    }

    _highlightedCameraPath = normalizedPath;
    _highlightedCameraName.clear();
    if (!_showCameraThumbnails)
    {
        _thumbnailPipeline.resourcesDirty = true;
    }
    if (_showCameraImage && !_cameraImageLocked)
    {
        updateActiveCameraForView();
    }
    updateCameraOverlay();
}

void CameraSceneWidget::setHighlightedCameraName(const QString &imageName)
{
    if (_highlightedCameraName == imageName && _highlightedCameraPath.isEmpty())
    {
        return;
    }

    _highlightedCameraName = imageName;
    _highlightedCameraPath.clear();
    if (!_showCameraThumbnails)
    {
        _thumbnailPipeline.resourcesDirty = true;
    }
    if (_showCameraImage && !_cameraImageLocked)
    {
        updateActiveCameraForView();
    }
    updateCameraOverlay();
}

void CameraSceneWidget::clearHighlightedCamera()
{
    if (_highlightedCameraPath.isEmpty() && _highlightedCameraName.isEmpty())
    {
        return;
    }

    _highlightedCameraPath.clear();
    _highlightedCameraName.clear();
    if (!_showCameraThumbnails)
    {
        _thumbnailPipeline.resourcesDirty = true;
    }
    updateCameraOverlay();
}

void CameraSceneWidget::clearProjectScene()
{
    cancelPendingLoad();
    ++_cameraImageLoadGeneration;
    _poses.clear();
    _cloud = RenderCloud();
    _isTiePointCloud = false;
    _tiePointImageCounts.clear();
    _tiePointMetadataLoading = false;
    _tiePointMetadataError.clear();
    _meshTextureImage = QImage();
    _meshTexturePath.clear();
    _meshHasTexture = false;
    _preparedObjMeshBuffer = false;
    _preparedObjMeshHasTexture = false;
    _preparedObjVertexData.clear();
    _preparedObjVertexCount = 0;
    _preparedObjStrideBytes = 0;
    _currentCloudPath.clear();
    _cameraImageCache.clear();
    _cameraImageLoadsInFlight.clear();
    _cameraImageLoadFailures.clear();
    _activeCameraImagePoseIndex = -1;
    _highlightedCameraPath.clear();
    _highlightedCameraName.clear();
    _hasFocusedGeometryBounds = false;
    _fitViewAfterLoad = false;
    _cacheDirty = true;
    _gpuDirty = true;
    _thumbnailPipeline.resourcesDirty = true;
    resetView();
}

// 取消未完成的加载（递增 generation 令旧回调自行失效）
void CameraSceneWidget::cancelPendingLoad()
{
    ++_loadGen;
    _loading = false;
    _plyLoadProgressPercent = -1;
    _plyLoadProgressText.clear();
    _fitViewAfterLoad = false;
}

// 标记缓存脏 + 重算（在加载完成后或场景数据变更后调用）
void CameraSceneWidget::invalidateCache() const
{
    bool has = false;
    bool has_cloud = false;
    QVector3D acc(0, 0, 0);
    int count = 0;
    QVector3D mn(0, 0, 0), mx(0, 0, 0);
    QVector3D cloud_mn(0, 0, 0), cloud_mx(0, 0, 0);

    auto accum = [&](const QVector3D &p)
    {
        acc += p; ++count;
        if (!has) { mn = p; mx = p; has = true; return; }
        mn.setX(qMin(mn.x(), p.x())); mn.setY(qMin(mn.y(), p.y())); mn.setZ(qMin(mn.z(), p.z()));
        mx.setX(qMax(mx.x(), p.x())); mx.setY(qMax(mx.y(), p.y())); mx.setZ(qMax(mx.z(), p.z()));
    };

    auto accum_cloud = [&](const QVector3D &p)
    {
        accum(p);
        if (!has_cloud)
        {
            cloud_mn = p;
            cloud_mx = p;
            has_cloud = true;
            return;
        }
        cloud_mn.setX(qMin(cloud_mn.x(), p.x()));
        cloud_mn.setY(qMin(cloud_mn.y(), p.y()));
        cloud_mn.setZ(qMin(cloud_mn.z(), p.z()));
        cloud_mx.setX(qMax(cloud_mx.x(), p.x()));
        cloud_mx.setY(qMax(cloud_mx.y(), p.y()));
        cloud_mx.setZ(qMax(cloud_mx.z(), p.z()));
    };

    for (const auto &p : _poses)             accum(p.center);
    for (size_t i = 0; i < _cloud.size(); ++i)
    {
        accum_cloud(QVector3D(_cloud.points()(static_cast<plamatrix::Index>(i), 0),
                              _cloud.points()(static_cast<plamatrix::Index>(i), 1),
                              _cloud.points()(static_cast<plamatrix::Index>(i), 2)));
    }
    _hasCloudBounds = has_cloud;
    if (has_cloud)
    {
        _cachedCloudAABBMin = cloud_mn;
        _cachedCloudAABBMax = cloud_mx;
    }

    if (count <= 0)
    {
        _cachedCenter  = QVector3D(0, 0, 0);
        _cachedRadius  = 10.0f;
        _cachedAABBMin = QVector3D(-10, -10, -10);
        _cachedAABBMax = QVector3D( 10,  10,  10);
    }
    else
    {
        _cachedCenter  = acc / float(count);
        _cachedAABBMin = mn;
        _cachedAABBMax = mx;
        // 使用 95th 百分位距离作为场景半径，避免离群点把相机推太远
        std::vector<float> dists;
        dists.reserve(_poses.size() + _cloud.size());
        for (const auto &p : _poses)             dists.push_back((p.center - _cachedCenter).length());
        for (size_t i = 0; i < _cloud.size(); ++i)
        {
            const QVector3D qp(_cloud.points()(static_cast<plamatrix::Index>(i), 0),
                               _cloud.points()(static_cast<plamatrix::Index>(i), 1),
                               _cloud.points()(static_cast<plamatrix::Index>(i), 2));
            dists.push_back((qp - _cachedCenter).length());
        }
        if (!dists.empty())
        {
            std::sort(dists.begin(), dists.end());
            const int p95 = std::min((int)dists.size() - 1, (int)(dists.size() * 0.95));
            // Keep the framing scale proportional to the loaded scene.  A fixed
            // one-world-unit floor makes compact photogrammetry models (Temple
            // is well below one unit across) appear as a tiny speck even though
            // their geometry is valid.
            _cachedRadius = qMax(1.0e-4f, dists[p95] * 1.15f);
        }
        else
        {
            _cachedRadius = 1.0f;
        }

    }
    _cacheDirty = false;
}

// 直接设置点云或网格（cloud.hasFaces() 决定渲染模式）
void CameraSceneWidget::setPointCloud(const RenderCloud &cloud)
{
    cancelPendingLoad();
    _cloud = cloneRenderCloud(cloud);
    _isTiePointCloud = false;
    _tiePointImageCounts.clear();
    _tiePointMetadataLoading = false;
    _tiePointMetadataError.clear();
    _meshTextureImage = QImage();
    _meshTexturePath.clear();
    _meshHasTexture = false;
    _preparedObjMeshBuffer = false;
    _preparedObjMeshHasTexture = false;
    _preparedObjVertexData.clear();
    _preparedObjVertexCount = 0;
    _preparedObjStrideBytes = 0;
    _texturedMeshPipeline.uploadedTexturePath.clear();
    _currentCloudPath.clear();
    _hasFocusedGeometryBounds = false;
    _fitViewAfterLoad = false;
    _preferModelPointRender = false;
    _cacheDirty = true;
    _gpuDirty   = true;
    update();
}

void CameraSceneWidget::setMesh(const RenderCloud &mesh)
{
    setPointCloud(mesh); // 实现相同，语义区分（mesh 应含面片）
}

// 从 XYZ 格式文本文件异步加载点云数据，使用 plapoint IO 解析。
void CameraSceneWidget::loadPointCloudFromXyz(const QString &xyzPath)
{
    loadPointCloudFromXyzInternal(xyzPath, false, true);
}

void CameraSceneWidget::loadPointCloudFromXyzInternal(const QString &xyzPath,
                                                       bool tiePointCloud,
                                                       bool fitAfterLoad)
{
    cancelPendingLoad();
    _currentCloudPath = xyzPath;
    _cloud = RenderCloud();
    _isTiePointCloud = tiePointCloud;
    _tiePointImageCounts.clear();
    _tiePointMetadataLoading = false;
    _tiePointMetadataError.clear();
    _meshTextureImage = QImage();
    _meshTexturePath.clear();
    _meshHasTexture = false;
    _preparedObjMeshBuffer = false;
    _preparedObjMeshHasTexture = false;
    _preparedObjVertexData.clear();
    _preparedObjVertexCount = 0;
    _preparedObjStrideBytes = 0;
    _texturedMeshPipeline.uploadedTexturePath.clear();
    _hasFocusedGeometryBounds = false;
    _fitViewAfterLoad = fitAfterLoad;
    _preferModelPointRender = false;
    _cacheDirty = true;
    _gpuDirty   = true;
    LOG_INFO(QStringLiteral("[3D] 正在加载点云: %1").arg(xyzPath));

    const int gen = _loadGen;
    QPointer<CameraSceneWidget> self(this);
    auto *watcher = new QFutureWatcher<std::shared_ptr<RenderCloud>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<RenderCloud>>::finished,
            watcher, [self, watcher, gen]()
    {
        if (!self)
        {
            watcher->deleteLater();
            return;
        }
        if (gen == self->_loadGen)
        {
            auto result = watcher->result();
            if (result)
            {
                self->_cloud = std::move(*result);
                LOG_INFO(QStringLiteral("[3D] 点云加载完成，共 %1 点")
                    .arg(self->_cloud.size()));
                self->invalidateCache();
                if (self->_fitViewAfterLoad)
                {
                    self->fitViewToLoadedGeometry();
                    self->_fitViewAfterLoad = false;
                }
                self->_gpuDirty = true;
                self->update();
            }
            else
            {
                self->_fitViewAfterLoad = false;
            }
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([xyzPath]() -> std::shared_ptr<RenderCloud>
    {
        try
        {
            return plapoint::io::readXyz<float>(xjw::common::io::toNativeNarrowPath(xyzPath));
        }
        catch (const std::exception &e)
        {
            LOG_ERROR(QStringLiteral("[3D] XYZ 加载失败: %1").arg(QString::fromStdString(e.what())));
        }
        return nullptr;
    }));
}

// 从 PLY 文件异步加载网格模型或点云。
void CameraSceneWidget::loadModelFromPly(const QString &plyPath)
{
    loadModelFromPlyInternal(plyPath, false, false, false);
}

void CameraSceneWidget::loadPointCloudFromPly(const QString &plyPath)
{
    loadModelFromPlyInternal(plyPath, false, true, true);
}

void CameraSceneWidget::loadModelFromPlyInternal(const QString &plyPath,
                                                 bool tiePointCloud,
                                                 bool fitAfterLoad,
                                                 bool pointCloudResource)
{
    cancelPendingLoad();
    _currentCloudPath = plyPath;
    _cloud = RenderCloud();
    _isTiePointCloud = tiePointCloud;
    _tiePointImageCounts.clear();
    _tiePointMetadataLoading = false;
    _tiePointMetadataError.clear();
    _meshTextureImage = QImage();
    _meshTexturePath.clear();
    _meshHasTexture = false;
    _preparedObjMeshBuffer = false;
    _preparedObjMeshHasTexture = false;
    _preparedObjVertexData.clear();
    _preparedObjVertexCount = 0;
    _preparedObjStrideBytes = 0;
    _texturedMeshPipeline.uploadedTexturePath.clear();
    _hasFocusedGeometryBounds = false;
    _fitViewAfterLoad = fitAfterLoad;
    _preferModelPointRender = !tiePointCloud;
    _cacheDirty = true;
    _gpuDirty   = true;
    _loading = true;
    _plyLoadProgressPercent = 0;
    _plyLoadProgressText = pointCloudResource
        ? QStringLiteral("正在加载 PLY 点云...")
        : QStringLiteral("正在加载 PLY 模型...");
    update();
    LOG_INFO(QStringLiteral("[3D] 正在加载 PLY %1: %2")
                 .arg(pointCloudResource ? QStringLiteral("点云")
                                         : QStringLiteral("模型"),
                      plyPath));

    const int gen = _loadGen;
    emit plyLoadProgressChanged(gen, 0, _plyLoadProgressText);
    QPointer<CameraSceneWidget> self(this);
    auto *watcher = new QFutureWatcher<std::shared_ptr<RenderCloud>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<RenderCloud>>::finished,
            watcher, [self, watcher, gen, pointCloudResource]()
    {
        if (!self)
        {
            watcher->deleteLater();
            return;
        }
        if (gen == self->_loadGen)
        {
            auto result = watcher->result();
            if (result) self->_cloud = std::move(*result);
            if (!self->_isTiePointCloud && self->_cloud.hasColors())
            {
                self->setModelColorMode(ModelColorMode::Texture);
            }
            else if (!self->_isTiePointCloud && self->_cloud.hasFaces())
            {
                // A mesh without vertex colors should expose its actual face
                // tessellation.  Smooth shading is still available explicitly,
                // but it must not masquerade as vertex colour information.
                self->setModelColorMode(ModelColorMode::Solid);
            }
            self->_preferModelPointRender = !self->_isTiePointCloud && !self->_cloud.hasFaces();
            if (!self->_cloud.hasFaces())
            {
                if (self->_cloud.size() >= 3'000'000)
                {
                    self->_modelPointSize = 1.1f;
                }
                else if (self->_cloud.size() >= 1'000'000)
                {
                    self->_modelPointSize = 1.4f;
                }
                else
                {
                    self->_modelPointSize = 2.4f;
                }
            }
            self->_loading = false;
            self->_plyLoadProgressPercent = -1;
            self->_plyLoadProgressText.clear();
            LOG_INFO(QStringLiteral("[3D] PLY %1加载完成，共 %2 顶点 / %3 面%4")
                         .arg(pointCloudResource ? QStringLiteral("点云")
                                                 : QStringLiteral("模型"))
                         .arg(self->_cloud.size())
                         .arg(self->_cloud.hasFaces()
                                  ? static_cast<int>(self->_cloud.faces()->rows())
                                  : 0)
                         .arg(self->_cloud.hasColors()
                                  ? QStringLiteral("（含RGB颜色）")
                                  : QStringLiteral("（无颜色）")));
            self->invalidateCache();
            if (self->_fitViewAfterLoad)
            {
                self->fitViewToLoadedGeometry();
                self->_fitViewAfterLoad = false;
            }
            self->_gpuDirty = true;
            self->update();
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([plyPath, self, gen]() -> std::shared_ptr<RenderCloud>
    {
        auto reportProgress = [self, gen](int percent, const QString &statusText)
        {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, gen, percent, statusText]()
            {
                if (!self)
                {
                    return;
                }
                emit self->plyLoadProgressChanged(gen, percent, statusText);
            }, Qt::QueuedConnection);
        };

        try
        {
            const PlyPreviewResult preview = readBinaryPlyPreview(plyPath, reportProgress);
            if (preview.header.vertexCount > kMaxDirectPlyVertices && preview.header.faceCount == 0)
            {
                if (preview.header.valid && preview.cloud)
                {
                    LOG_INFO(QStringLiteral("[3D] PLY 过大，使用预览抽样: 原始 %1 点，显示 %2 点，抽样步长 %3")
                                 .arg(preview.header.vertexCount)
                                 .arg(preview.cloud->size())
                                 .arg(qMax<quint64>(
                                     1,
                                     (preview.header.vertexCount + preview.cloud->size() - 1)
                                         / qMax<quint64>(1, preview.cloud->size()))));
                    return preview.cloud;
                }
                LOG_ERROR(QStringLiteral("[3D] PLY 文件过大，无法安全完整加载；预览加载失败: %1")
                              .arg(preview.error));
                return nullptr;
            }
            if (preview.header.vertexCount > kMaxDirectPlyVertices && preview.header.faceCount > 0)
            {
                reportProgress(5,
                               QStringLiteral("正在完整加载 PLY 网格 (%1 顶点 / %2 面)...")
                                   .arg(preview.header.vertexCount)
                                   .arg(preview.header.faceCount));
                LOG_INFO(QStringLiteral("[3D] PLY 网格较大，保留面片完整加载: %1 顶点 / %2 面")
                             .arg(preview.header.vertexCount)
                             .arg(preview.header.faceCount));
            }
            else if (preview.header.valid)
            {
                reportProgress(5,
                               QStringLiteral("正在完整加载 PLY 点云 (%1 顶点 / %2 面)...")
                                   .arg(preview.header.vertexCount)
                                   .arg(preview.header.faceCount));
            }
            else
            {
                reportProgress(5, QStringLiteral("正在完整加载 PLY 点云..."));
            }
            return plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(plyPath));
        }
        catch (const std::exception &e)
        {
            LOG_ERROR(QStringLiteral("[3D] PLY 加载失败: %1").arg(QString::fromStdString(e.what())));
        }
        reportProgress(100, QStringLiteral("密集点云加载失败"));
        return nullptr;
    }));
}

void CameraSceneWidget::loadModelFromObj(const QString &objPath)
{
    loadModelFromObjInternal(objPath, false, false, false);
}

void CameraSceneWidget::loadPointCloudFromObj(const QString &objPath)
{
    loadModelFromObjInternal(objPath, false, true, true);
}

void CameraSceneWidget::loadModelFromObjInternal(const QString &objPath,
                                                 bool tiePointCloud,
                                                 bool fitAfterLoad,
                                                 bool pointCloudResource)
{
    cancelPendingLoad();
    _currentCloudPath = objPath;
    _cloud = RenderCloud();
    _isTiePointCloud = tiePointCloud;
    _tiePointImageCounts.clear();
    _tiePointMetadataLoading = false;
    _tiePointMetadataError.clear();
    _meshTextureImage = QImage();
    _meshTexturePath.clear();
    _meshHasTexture = false;
    _preparedObjMeshBuffer = false;
    _preparedObjMeshHasTexture = false;
    _preparedObjVertexData.clear();
    _preparedObjVertexCount = 0;
    _preparedObjStrideBytes = 0;
    _texturedMeshPipeline.uploadedTexturePath.clear();
    _hasFocusedGeometryBounds = false;
    _fitViewAfterLoad = fitAfterLoad;
    _preferModelPointRender = false;
    _cacheDirty = true;
    _gpuDirty = true;
    _loading = true;
    _plyLoadProgressPercent = 0;
    _plyLoadProgressText = pointCloudResource
        ? QStringLiteral("正在加载 OBJ 点云...")
        : QStringLiteral("正在加载 OBJ 模型...");
    update();
    LOG_INFO(QStringLiteral("[3D] 正在加载 OBJ %1: %2")
                 .arg(pointCloudResource ? QStringLiteral("点云")
                                         : QStringLiteral("模型"),
                      objPath));

    const int gen = _loadGen;
    emit plyLoadProgressChanged(gen, 0, _plyLoadProgressText);
    QPointer<CameraSceneWidget> self(this);
    auto *watcher = new QFutureWatcher<ObjLoadResult>(this);
    connect(watcher, &QFutureWatcher<ObjLoadResult>::finished,
            watcher, [self, watcher, gen, pointCloudResource]()
    {
        if (!self)
        {
            watcher->deleteLater();
            return;
        }
        if (gen == self->_loadGen)
        {
            ObjLoadResult result = watcher->result();
            if (result.cloud)
            {
                self->_cloud = std::move(*result.cloud);
                self->_meshTextureImage = result.textureImage;
                self->_meshTexturePath = result.texturePath;
                self->_texturedMeshPipeline.uploadedTexturePath.clear();
                if (result.renderPreparation.isValid())
                {
                    self->_preparedObjVertexData = std::move(
                        result.renderPreparation.vertexData);
                    self->_preparedObjVertexCount = result.renderPreparation.vertexCount;
                    self->_preparedObjStrideBytes = result.renderPreparation.strideBytes;
                    self->_meshBuffer.vertexData = self->_preparedObjVertexData;
                    self->_meshBuffer.vertexCount = self->_preparedObjVertexCount;
                    self->_meshBuffer.strideBytes = self->_preparedObjStrideBytes;
                    self->_meshBuffer.dirty = true;
                    self->_preparedObjMeshBuffer = true;
                    self->_preparedObjMeshHasTexture = result.renderPreparation.hasTexture;
                }
                if (!self->_isTiePointCloud &&
                    !self->_meshTextureImage.isNull() &&
                    self->_preparedObjMeshHasTexture)
                {
                    self->setModelColorMode(ModelColorMode::Texture);
                }
                else if (!self->_isTiePointCloud && self->_cloud.hasColors())
                {
                    self->setModelColorMode(ModelColorMode::Texture);
                }
                else if (!self->_isTiePointCloud && self->_cloud.hasFaces())
                {
                    self->setModelColorMode(ModelColorMode::Solid);
                }
            }
            self->_preferModelPointRender = !self->_isTiePointCloud && !self->_cloud.hasFaces();
            self->_loading = false;
            self->_plyLoadProgressPercent = -1;
            self->_plyLoadProgressText.clear();
            if (self->_cloud.size() == 0)
            {
                LOG_ERROR(QStringLiteral("[3D] OBJ %1加载失败或为空")
                              .arg(pointCloudResource ? QStringLiteral("点云")
                                                      : QStringLiteral("模型")));
            }
            else
            {
                LOG_INFO(QStringLiteral("[3D] OBJ %1加载完成，共 %2 顶点 / %3 面%4")
                             .arg(pointCloudResource ? QStringLiteral("点云")
                                                     : QStringLiteral("模型"))
                             .arg(self->_cloud.size())
                             .arg(self->_cloud.hasFaces() ? static_cast<int>(self->_cloud.faces()->rows()) : 0)
                             .arg(self->_meshTextureImage.isNull()
                                      ? QStringLiteral("（顶点颜色）")
                                      : QStringLiteral("（MTL 纹理）")));
                LOG_INFO(QStringLiteral("[3D] OBJ %1耗时: 解析 %2 ms，渲染数据准备 %3 ms")
                             .arg(pointCloudResource ? QStringLiteral("点云")
                                                     : QStringLiteral("模型"))
                             .arg(result.parseElapsedMs)
                             .arg(result.prepareElapsedMs));
                if (!pointCloudResource && !result.textureWarning.isEmpty())
                {
                    LOG_WARN(QStringLiteral("[3D] %1").arg(result.textureWarning));
                }
            }
            self->invalidateCache();
            if (self->_fitViewAfterLoad)
            {
                self->fitViewToLoadedGeometry();
                self->_fitViewAfterLoad = false;
            }
            self->_gpuDirty = true;
            self->update();
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run(
        [objPath, self, gen, pointCloudResource]() -> ObjLoadResult
    {
        auto reportProgress = [self, gen](int percent, const QString &statusText)
        {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, gen, percent, statusText]()
            {
                if (!self)
                {
                    return;
                }
                emit self->plyLoadProgressChanged(gen, percent, statusText);
            }, Qt::QueuedConnection);
        };

        try
        {
            reportProgress(5,
                           pointCloudResource
                               ? QStringLiteral("正在解析 OBJ 点云...")
                               : QStringLiteral("正在解析 OBJ 模型..."));
            ObjLoadResult result;
            if (pointCloudResource)
            {
                QElapsedTimer timer;
                timer.start();
                result.cloud = plapoint::io::readObj<float>(
                    xjw::common::io::toNativeNarrowPath(objPath));
                result.parseElapsedMs = timer.elapsed();
            }
            else
            {
                result = loadObjWithMaterialTexture(objPath, reportProgress);
            }
            if (!result.cloud || result.cloud->size() == 0)
            {
                reportProgress(100,
                               pointCloudResource
                                   ? QStringLiteral("OBJ 点云为空")
                                   : QStringLiteral("OBJ 模型为空"));
                return result;
            }
            reportProgress(96,
                           QStringLiteral("正在上传 OBJ %1 (%2 顶点 / %3 面)...")
                               .arg(pointCloudResource ? QStringLiteral("点云")
                                                       : QStringLiteral("模型"))
                               .arg(result.cloud->size())
                               .arg(result.cloud->hasFaces()
                                        ? static_cast<int>(result.cloud->faces()->rows())
                                        : 0));
            return result;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR(QStringLiteral("[3D] OBJ 加载失败: %1").arg(QString::fromStdString(e.what())));
        }
        reportProgress(100, QStringLiteral("OBJ 加载失败"));
        return ObjLoadResult();
    }));
}

void CameraSceneWidget::loadTiePointCloudFromFile(const QString &pointCloudPath,
                                                  const QString &sidecarPath)
{
    const QString extension = QFileInfo(pointCloudPath).suffix().toLower();
    if (extension == QLatin1String("ply"))
    {
        loadModelFromPlyInternal(pointCloudPath, true, true, true);
    }
    else if (extension == QLatin1String("obj"))
    {
        loadModelFromObjInternal(pointCloudPath, true, true, true);
    }
    else
    {
        loadPointCloudFromXyzInternal(pointCloudPath, true, true);
    }

    if (_loading)
    {
        _plyLoadProgressText = tr("正在加载连接点...");
    }

    QString metadataPath = sidecarPath.trimmed();
    if (metadataPath.isEmpty())
    {
        metadataPath = xjw::gui::tie_points::inferSidecarPath(pointCloudPath);
    }
    startTiePointMetadataLoad(metadataPath, _loadGen);
    _gpuDirty = true;
    update();
}

void CameraSceneWidget::setTiePointColorMode(TiePointColorMode mode)
{
    if (_tiePointColorMode == mode)
    {
        return;
    }
    _tiePointColorMode = mode;
    _gpuDirty = true;
    update();
    requestOverlayUpdate();
}

void CameraSceneWidget::setModelColorMode(ModelColorMode mode)
{
    if (mode == ModelColorMode::Confidence ||
        mode == ModelColorMode::AssignedImage)
    {
        return;
    }
    if (_modelColorMode == mode)
    {
        return;
    }
    _modelColorMode = mode;
    _modelVisualization.setMode(mode);
    _gpuDirty = true;
    _pipelinesDirty = true;
    update();
    requestOverlayUpdate();
}

void CameraSceneWidget::startTiePointMetadataLoad(const QString &sidecarPath, int generation)
{
    if (sidecarPath.trimmed().isEmpty())
    {
        _tiePointMetadataLoading = false;
        _tiePointMetadataError = tr("无观测数据");
        requestOverlayUpdate();
        return;
    }

    _tiePointMetadataLoading = true;
    QPointer<CameraSceneWidget> self(this);
    auto *watcher =
        new QFutureWatcher<xjw::gui::tie_points::ImageCountMetadata>(this);
    connect(watcher,
            &QFutureWatcher<xjw::gui::tie_points::ImageCountMetadata>::finished,
            watcher,
            [self, watcher, generation]()
    {
        if (!self)
        {
            watcher->deleteLater();
            return;
        }
        if (generation == self->_loadGen && self->_isTiePointCloud)
        {
            const xjw::gui::tie_points::ImageCountMetadata metadata = watcher->result();
            self->_tiePointImageCounts = metadata.counts;
            self->_tiePointMetadataError = metadata.errorMessage;
            self->_tiePointMetadataLoading = false;
            self->_gpuDirty = true;
            self->update();
            self->requestOverlayUpdate();
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([sidecarPath]()
    {
        return xjw::gui::tie_points::loadImageCountMetadata(sidecarPath);
    }));
}

void CameraSceneWidget::fitViewToLoadedGeometry()
{
    if (_cloud.size() == 0)
    {
        _hasFocusedGeometryBounds = false;
        return;
    }

    QVector3D center;
    for (size_t index = 0; index < _cloud.size(); ++index)
    {
        center += QVector3D(
            _cloud.points()(static_cast<plamatrix::Index>(index), 0),
            _cloud.points()(static_cast<plamatrix::Index>(index), 1),
            _cloud.points()(static_cast<plamatrix::Index>(index), 2));
    }
    center /= static_cast<float>(_cloud.size());

    std::vector<float> distances;
    distances.reserve(_cloud.size());
    for (size_t index = 0; index < _cloud.size(); ++index)
    {
        const QVector3D point(
            _cloud.points()(static_cast<plamatrix::Index>(index), 0),
            _cloud.points()(static_cast<plamatrix::Index>(index), 1),
            _cloud.points()(static_cast<plamatrix::Index>(index), 2));
        distances.push_back((point - center).length());
    }
    std::sort(distances.begin(), distances.end());
    const size_t percentileIndex = qMin(
        distances.size() - 1,
        static_cast<size_t>(distances.size() * 0.95));

    _focusedGeometryCenter = center;
    _focusedGeometryRadius = qMax(
        1.0e-4f,
        distances.at(percentileIndex) * 1.15f);
    _hasFocusedGeometryBounds = true;
    _zoomScale = 1.0;
    _sceneOffsetPx = QPointF();
    _hoverAxis = HoverAxis::None;
    _dragAxis = HoverAxis::None;
    LOG_INFO(QStringLiteral("[3D] 已聚焦加载几何：中心 (%1, %2, %3)，半径 %4")
                 .arg(center.x(), 0, 'g', 6)
                 .arg(center.y(), 0, 'g', 6)
                 .arg(center.z(), 0, 'g', 6)
                 .arg(_focusedGeometryRadius, 0, 'g', 6));
    updateCameraOverlay();
}

// 计算场景中所有点（相机光心、点云、模型顶点）的质心作为场景中心。
// 使用缓存，仅在数据变更后重新计算。
QVector3D CameraSceneWidget::sceneCenter() const
{
    if (_hasFocusedGeometryBounds)
    {
        return _focusedGeometryCenter;
    }
    if (_cacheDirty) invalidateCache();
    return _cachedCenter;
}

// 计算场景中所有点到质心的最大距离，用于自适应相机距离、投影远裁平面等。
// 使用缓存，仅在数据变更后重新计算。
float CameraSceneWidget::sceneRadius() const
{
    if (_hasFocusedGeometryBounds)
    {
        return _focusedGeometryRadius;
    }
    if (_cacheDirty) invalidateCache();
    return _cachedRadius;
}

CameraSceneWidget::SceneMatrices CameraSceneWidget::sceneMatrices() const
{
    const QVector3D center = sceneCenter();
    const float radius = sceneRadius();
    const float distance = static_cast<float>(
        static_cast<double>(radius) * 3.2 / _zoomScale);
    const float nearPlane = distance * 0.001f;
    const float farPlane  = qMax(1000.0f, distance * 100.0f + radius * 50.0f);

    SceneMatrices matrices;
    // 从 +Z 方向看向原点：世界X→屏幕右，世界Y→屏幕上，与overlay/arcball坐标系一致
    const QVector3D eye = center + QVector3D(0.0f, 0.0f, distance);
    QMatrix4x4 view;
    view.lookAt(eye, center, QVector3D(0.0f, 1.0f, 0.0f));
    QMatrix4x4 model;
    model.setToIdentity();
    model.translate(center);
    model.rotate(_viewRot);
    model.translate(-center);
    matrices.modelView = view * model;

    const float aspect = qMax(1.0f, float(width()) / qMax(1, height()));
    matrices.projection.perspective(45.0f, aspect, nearPlane, farPlane);
    return matrices;
}

QPointF CameraSceneWidget::projectToScreen(const QVector3D &p, bool *ok) const
{
    const SceneMatrices matrices = sceneMatrices();
    QVector4D clip = matrices.projection * matrices.modelView * QVector4D(p, 1.0f);
    if (clip.w() <= 1e-6f) {
        if (ok) *ok = false;
        return QPointF();
    }
    QVector3D ndc(clip.x() / clip.w(), clip.y() / clip.w(), clip.z() / clip.w());
    if (ok) *ok = true;
    const float sx = (ndc.x() * 0.5f + 0.5f) * width() + float(_sceneOffsetPx.x());
    const float sy = (1.0f - (ndc.y() * 0.5f + 0.5f)) * height() + float(_sceneOffsetPx.y());
    return QPointF(sx, sy);
}

// 将向量从副本局部空间旋转到当前视图空间（应用 _viewRot）
QVector3D CameraSceneWidget::applyViewRotation(const QVector3D &v) const
{
    return _viewRot.rotatedVector(v);
}

// 返回当前视图四元数对应的欧拉角（度， x=pitch, y=yaw, z=roll）
QVector3D CameraSceneWidget::eulerAnglesDeg() const
{
    return _viewRot.toEulerAngles();
}

// 根据窗口尺寸自适应计算 Gizmo 操控球屏幕半径（px）
// 范围：[30, min(w,h)*0.24]，基准为 min(w,h)*0.11
qreal CameraSceneWidget::manipRadiusPx() const
{
    const qreal base = qMin(width(), height()) * 0.11;
    return qBound<qreal>(30.0, base, qMin(width(), height()) * 0.24);
}

int CameraSceneWidget::maxVisibleCameraLabels() const
{
    const int viewportBudget = qBound(8, width() / 140, 28);
    if (_poses.size() > 300)
    {
        return qMin(viewportBudget, 12);
    }
    if (_poses.size() > 120)
    {
        return qMin(viewportBudget, 18);
    }
    if (_poses.size() > 60)
    {
        return qMin(viewportBudget, 24);
    }
    return qMin(viewportBudget, 40);
}

float CameraSceneWidget::cameraImagePlaneHalfExtent(
    const CameraPose &pose,
    const QMatrix4x4 &worldToView) const
{
    if (_cacheDirty) invalidateCache();
    const float screenScaledExtent =
        xjw::gui::camera_scene::cameraPlaneHalfExtentForScreenSize(
            pose.center,
            worldToView,
            height(),
            _zoomScale,
            45.0f,
            34.0);
    if (screenScaledExtent > 0.0f)
    {
        return screenScaledExtent;
    }

    const float radius = sceneRadius();
    return qMax(1.0e-5f, radius * 0.065f);
}

bool CameraSceneWidget::isCameraHighlighted(const CameraPose &pose) const
{
    if (!_highlightedCameraPath.isEmpty())
    {
        if (normalizedCameraPath(pose.imagePath) == _highlightedCameraPath)
        {
            return true;
        }

        const QString highlightedFileName = QFileInfo(_highlightedCameraPath).fileName();
        if (!highlightedFileName.isEmpty())
        {
            return pose.name == highlightedFileName
                || QFileInfo(pose.imagePath).fileName() == highlightedFileName;
        }
    }

    if (!_highlightedCameraName.isEmpty())
    {
        return pose.name == _highlightedCameraName
            || QFileInfo(pose.imagePath).fileName() == _highlightedCameraName;
    }

    return false;
}

QString CameraSceneWidget::normalizedCameraPath(const QString &imagePath) const
{
    if (imagePath.isEmpty())
    {
        return QString();
    }

    return QDir::cleanPath(QFileInfo(imagePath).absoluteFilePath());
}

QString CameraSceneWidget::cameraPlaneImageKey(const QString &imagePath, CameraImagePlaneMode mode) const
{
    const QString normalizedPath = normalizedCameraPath(imagePath);
    if (normalizedPath.isEmpty())
    {
        return QString();
    }

    QString modeKey;
    switch (mode)
    {
    case CameraImagePlaneMode::Image:
        modeKey = QStringLiteral("image");
        break;
    case CameraImagePlaneMode::Thumbnail:
        modeKey = QStringLiteral("thumb");
        break;
    case CameraImagePlaneMode::Solid:
        modeKey = QStringLiteral("solid");
        break;
    }
    return modeKey + QLatin1Char('|') + normalizedPath;
}

QImage CameraSceneWidget::cachedCameraPlaneImage(const QString &imagePath, CameraImagePlaneMode mode) const
{
    if (mode == CameraImagePlaneMode::Solid)
    {
        return QImage();
    }

    return _cameraImageCache.value(cameraPlaneImageKey(imagePath, mode));
}

void CameraSceneWidget::requestCameraPlaneImage(const QString &imagePath, CameraImagePlaneMode mode)
{
    if (mode == CameraImagePlaneMode::Solid)
    {
        return;
    }

    const QString key = cameraPlaneImageKey(imagePath, mode);
    if (key.isEmpty() || _cameraImageCache.contains(key) || _cameraImageLoadsInFlight.contains(key)
        || _cameraImageLoadFailures.contains(key))
    {
        return;
    }
    if (_cameraImageLoadsInFlight.size() >= 6)
    {
        return;
    }

    _cameraImageLoadsInFlight.insert(key);
    auto *watcher = new QFutureWatcher<CameraPlaneImageResult>(this);
    const int generation = _cameraImageLoadGeneration;
    connect(watcher, &QFutureWatcher<CameraPlaneImageResult>::finished, this, [this, watcher, key]()
    {
        applyCameraPlaneImage(watcher->result());
        _cameraImageLoadsInFlight.remove(key);
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run(&CameraSceneWidget::loadCameraPlaneImage,
                                         imagePath,
                                         mode,
                                         generation));
}

void CameraSceneWidget::applyCameraPlaneImage(const CameraPlaneImageResult &result)
{
    if (result.generation != _cameraImageLoadGeneration)
    {
        return;
    }

    const QString key = cameraPlaneImageKey(result.path, result.mode);
    if (!result.loaded || result.image.isNull())
    {
        if (!key.isEmpty())
        {
            _cameraImageLoadFailures.insert(key);
        }
        if (!result.errorMessage.isEmpty())
        {
            LOG_WARN("%s", qUtf8Printable(result.errorMessage));
        }
        return;
    }

    if (key.isEmpty())
    {
        return;
    }

    if (_cameraImageCache.size() > 512)
    {
        _cameraImageCache.clear();
    }
    _cameraImageCache.insert(key, result.image);
    if (result.originalSize.isValid())
    {
        const QString result_path = normalizedCameraPath(result.path);
        for (CameraPose &pose : _poses)
        {
            if (normalizedCameraPath(pose.imagePath) == result_path)
            {
                if (pose.imageWidth <= 0)
                {
                    pose.imageWidth = result.originalSize.width();
                }
                if (pose.imageHeight <= 0)
                {
                    pose.imageHeight = result.originalSize.height();
                }
                break;
            }
        }
    }
    update();
}

CameraSceneWidget::CameraPlaneImageResult CameraSceneWidget::loadCameraPlaneImage(const QString &imagePath,
                                                                                  CameraImagePlaneMode mode,
                                                                                  int generation)
{
    CameraPlaneImageResult result;
    result.path = imagePath;
    result.mode = mode;
    result.generation = generation;

    QImage image = xjw::gui::views::loadImageForDisplay(imagePath, QString());
    if (image.isNull())
    {
        result.errorMessage = QStringLiteral("三维视图无法读取照片：%1").arg(imagePath);
        return result;
    }

    result.originalSize = image.size();

    const QSize targetSize = mode == CameraImagePlaneMode::Image
        ? QSize(2048, 2048)
        : QSize(220, 160);
    result.image = image.scaled(targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    result.loaded = !result.image.isNull();
    return result;
}

void CameraSceneWidget::updateActiveCameraForView()
{
    if (_cameraImageLocked || _poses.isEmpty())
    {
        return;
    }

    QVector<xjw::gui::camera_scene::CameraViewCandidate> candidates;
    candidates.reserve(_poses.size());
    for (qsizetype index = 0; index < _poses.size(); ++index)
    {
        const CameraPose &pose = _poses.at(index);
        candidates.push_back({
            static_cast<int>(index),
            xjw::gui::camera_scene::cameraForwardDirection(pose.rotation, pose.depthAxisFlipped),
            pose.center,
            !pose.imagePath.isEmpty(),
        });
    }

    _activeCameraImagePoseIndex = xjw::gui::camera_scene::selectCameraForView(
        candidates,
        xjw::gui::camera_scene::currentWorldViewDirection(_viewRot),
        sceneCenter());
}

int CameraSceneWidget::displayedCameraImagePoseIndex() const
{
    if (_poses.isEmpty())
    {
        return -1;
    }

    if (_cameraImageLocked)
    {
        const QString lockedPath = normalizedCameraPath(_lockedCameraImagePath);
        for (qsizetype i = 0; i < _poses.size(); ++i)
        {
            const CameraPose &pose = _poses.at(i);
            if (!lockedPath.isEmpty() && normalizedCameraPath(pose.imagePath) == lockedPath)
            {
                return static_cast<int>(i);
            }
            if (!_lockedCameraImageName.isEmpty()
                && (pose.name == _lockedCameraImageName
                    || QFileInfo(pose.imagePath).fileName() == _lockedCameraImageName))
            {
                return static_cast<int>(i);
            }
        }
    }

    if (_activeCameraImagePoseIndex >= 0 && _activeCameraImagePoseIndex < _poses.size())
    {
        return _activeCameraImagePoseIndex;
    }
    return -1;
}

void CameraSceneWidget::refreshLockedCameraImage()
{
    const int poseIndex = displayedCameraImagePoseIndex();
    if (poseIndex < 0 || poseIndex >= _poses.size())
    {
        _lockedCameraImagePath.clear();
        _lockedCameraImageName.clear();
        return;
    }

    const CameraPose &pose = _poses.at(poseIndex);
    _lockedCameraImagePath = pose.imagePath;
    _lockedCameraImageName = pose.name;
}

void CameraSceneWidget::drawFloorPivotCross(QPainter &painter) const
{
    if (_cacheDirty) invalidateCache();

    const QVector3D minimum = _hasCloudBounds ? _cachedCloudAABBMin : _cachedAABBMin;
    const QVector3D maximum = _hasCloudBounds ? _cachedCloudAABBMax : _cachedAABBMax;
    const QVector3D floorPivot((minimum.x() + maximum.x()) * 0.5f,
                               (minimum.y() + maximum.y()) * 0.5f,
                               minimum.z());

    bool okCenter = false;
    const QPointF c = projectToScreen(floorPivot, &okCenter);
    if (!okCenter)
    {
        return;
    }

    constexpr qreal half_size_pixels = 7.0;
    painter.setBrush(Qt::NoBrush);
    painter.setPen(QPen(QColor(105, 108, 112, 170), 1.2));
    painter.drawLine(c + QPointF(-half_size_pixels, 0.0),
                     c + QPointF(half_size_pixels, 0.0));
    painter.drawLine(c + QPointF(0.0, -half_size_pixels),
                     c + QPointF(0.0, half_size_pixels));
}

QLineF CameraSceneWidget::cameraDirectionLeaderLine(const CameraPose &pose,
                                                    float planeHalfExtent) const
{
    const QVector3D forward = xjw::gui::camera_scene::cameraForwardDirection(
        pose.rotation, pose.depthAxisFlipped);
    if (forward.isNull() || planeHalfExtent <= 0.0f)
    {
        return {};
    }

    bool centerOk = false;
    bool probeOk = false;
    const QPointF center = projectToScreen(pose.center, &centerOk);
    // Metashape 风格的黑色引线从照片平面向相机后方延伸；相机本身
    // 沿 forward 朝向被摄物，因此这里使用反向光轴而不是绘制箭头。
    const QPointF probe = projectToScreen(
        pose.center - forward * planeHalfExtent,
        &probeOk);
    if (!centerOk || !probeOk)
    {
        return {};
    }

    QPointF screenDirection = probe - center;
    qreal directionLength = QLineF(QPointF(), screenDirection).length();
    if (directionLength < 1.0)
    {
        bool sceneCenterOk = false;
        const QPointF sceneCenterScreen = projectToScreen(sceneCenter(), &sceneCenterOk);
        if (sceneCenterOk)
        {
            screenDirection = center - sceneCenterScreen;
            directionLength = QLineF(QPointF(), screenDirection).length();
        }
    }
    if (directionLength < 1.0)
    {
        return {};
    }

    const qreal screenHalfExtent =
        xjw::gui::camera_scene::cameraPlaneScreenHalfExtentPixels(
            _zoomScale,
            34.0);
    const qreal lineStartOffset = screenHalfExtent + 3.0;
    const qreal leaderLength = qBound<qreal>(26.0, screenHalfExtent * 0.5, 44.0);
    return xjw::gui::camera_scene::cameraPlaneLeaderLine(
        center,
        center + screenDirection,
        lineStartOffset,
        leaderLength);
}

// Gizmo 操控球的世界中心点（等于场景质心）
QVector3D CameraSceneWidget::manipCenterWorld() const
{
    return sceneCenter();
}

// Gizmo 操控球的屏幕中心点（固定为窗口中心）
QPointF CameraSceneWidget::manipCenterScreen() const
{
    return QPointF(width() * 0.5, height() * 0.5);
}

// 根据鼠标位置检测鼠标当前悬停的 Gizmo 展向轴环。
// 原理：遍历 X/Y/Z 三个轴环的散列点，计算鼠标到环上最近线段的距离，
// 距离小于阈値 12px 时判定为悬停在该轴环上。
CameraSceneWidget::HoverAxis CameraSceneWidget::pickHoverAxis(const QPoint &mousePos) const
{
    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // 计算鼠标到指定轴环的最近距离（只考虑正面可见的弧段）
    auto minDistToCircle = [&](HoverAxis axis) {
        qreal best = 1e9;
        QPointF prev;
        bool hasPrev = false;
        bool prevVisible = false;
        for (int i = 0; i <= 96; ++i) {
            const qreal t = (2.0 * M_PI * i) / 96.0;
            QVector3D pLocal;
            // 根据轴选择环面上的局部点
            if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
            else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
            else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
            QVector3D pView = applyViewRotation(pLocal);
            const bool currVisible = (pView.z() > 0.0f); // z>0 表示正面对观察者
            QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
            if (hasPrev && prevVisible && currVisible) {
                // 计算鼠标到线段 [prev, curr] 的最近距离
                const QPointF ab = curr - prev;
                const qreal ab2 = ab.x() * ab.x() + ab.y() * ab.y();
                if (ab2 > 1e-6) {
                    qreal u = ((mousePos.x() - prev.x()) * ab.x() + (mousePos.y() - prev.y()) * ab.y()) / ab2;
                    u = qBound<qreal>(0.0, u, 1.0);
                    QPointF proj = prev + ab * u;
                    best = qMin(best, QLineF(QPointF(mousePos), proj).length());
                }
            }
            prev = curr;
            prevVisible = currVisible;
            hasPrev = true;
        }
        return best;
    };

    const qreal th = 12.0; // 距离阈値（像素）
    const qreal dx = minDistToCircle(HoverAxis::X);
    const qreal dy = minDistToCircle(HoverAxis::Y);
    const qreal dz = minDistToCircle(HoverAxis::Z);
    const qreal dmin = qMin(dx, qMin(dy, dz));
    if (dmin > th) return HoverAxis::None; // 距离过远，无悬停
    // 返回距离最小的轴
    if (dx <= dy && dx <= dz) return HoverAxis::X;
    if (dy <= dx && dy <= dz) return HoverAxis::Y;
    return HoverAxis::Z;
}

// 计算鼠标附近指定轴环的切线方向（屏幕空间单位向量）。
// 用于将鼠标拖拽距离投影到切线方向以计算旋转角度。
QVector2D CameraSceneWidget::pickAxisTangent(const QPoint &mousePos, HoverAxis axis) const
{
    if (axis == HoverAxis::None) return QVector2D(1.0f, 0.0f);
    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // 遍历环面散列点，找到鼠标最近的线段 [bestA, bestB]
    QPointF bestA;
    QPointF bestB;
    qreal bestDist = 1e12;
    QPointF prev;
    bool hasPrev = false;
    bool prevVisible = false;
    for (int i = 0; i <= 128; ++i) 
    {
        const qreal t = (2.0 * M_PI * i) / 128.0;
        QVector3D pLocal;
        if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
        else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
        else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
        const QVector3D pView = applyViewRotation(pLocal);
        const bool currVisible = (pView.z() > 0.0f);
        const QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
        if (hasPrev && prevVisible && currVisible) 
        {
            const QPointF ab = curr - prev;
            const qreal ab2 = ab.x() * ab.x() + ab.y() * ab.y();
            if (ab2 > 1e-9) 
            {
                qreal u = ((mousePos.x() - prev.x()) * ab.x() + (mousePos.y() - prev.y()) * ab.y()) / ab2;
                u = qBound<qreal>(0.0, u, 1.0);
                const QPointF proj = prev + ab * u;
                const qreal d = QLineF(QPointF(mousePos), proj).length();
                if (d < bestDist) 
                {
                    bestDist = d;
                    bestA = prev;  // 最近线段起点
                    bestB = curr;  // 最近线段终点
                }
            }
        }
        prev = curr;
        prevVisible = currVisible;
        hasPrev = true;
    }

    // 计算并归一化切线方向向量
    QVector2D tanDir(float(bestB.x() - bestA.x()), float(bestB.y() - bestA.y()));
    if (tanDir.lengthSquared() < 1e-8f) tanDir = QVector2D(1.0f, 0.0f); // 防止除以零
    return tanDir.normalized();
}

// ---------------------------------------------------------------------------
// Arcball 球面投影
//   屏幕投影规则： pView.x → 屏幕右， pView.y → 屏幕上（y已翻转）， pView.z → 朝向观察者
//   注：与 drawGreatCircle 里 cursor2d + (pView.x*r, -pView.y*r) 保持一致
// ---------------------------------------------------------------------------
QVector3D CameraSceneWidget::arcballVector(const QPoint &mousePos) const
{
    const QPointF c = manipCenterScreen();
    const float r = float(manipRadiusPx());
    if (r < 1.0f) return QVector3D(0.0f, 0.0f, 1.0f);

    const float x =  float(mousePos.x() - c.x()) / r;
    const float y = -float(mousePos.y() - c.y()) / r;  // 屏幕 Y 选转为那数学 Y
    const float len2 = x * x + y * y;
    if (len2 <= 1.0f) {
        return QVector3D(x, y, std::sqrt(1.0f - len2));   // 在球面上
    }
    // 在球外：投影到赤道圆
    const float len = std::sqrt(len2);
    return QVector3D(x / len, y / len, 0.0f);
}

// 平移偏移量无限制（允许将模型拖出视口外）
void CameraSceneWidget::clampSceneOffset()
{
    // 不做任何限制
}

// 根据当前的悬停轴/拖拽状态更新鼠标光标样式：
//   - 中键拖拽中        → 四向移动光标
//   - 左键自由旋转中    → 闭合手型光标
//   - 悬停/拖拽 X/Y/Z 环 → 带颜色和字母的自定义圆形光标
//   - 默认              → 开放手型光标
void CameraSceneWidget::updateCursor()
{
    if (_manualPruneMode && _manualSelecting)
    {
        setCursor(Qt::CrossCursor);
        return;
    }

    // 创建带颜色和轴标签的自定义光标（24x24 像素，圆心热点）
    auto axisCursor = [](const QColor &color, const QString &label) {
        QPixmap pm(24, 24);
        pm.fill(Qt::transparent);
        QPainter p(&pm);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(color, 2));
        p.drawEllipse(QPointF(12, 12), 8, 8);
        p.drawText(QRect(0, 0, 24, 24), Qt::AlignCenter, label);
        return QCursor(pm, 12, 12);
    };

    if (_middleDragging) {
        setCursor(Qt::SizeAllCursor); // 中键平移中
        return;
    }
    if (_leftDragging && _dragAxis == HoverAxis::None) {
        setCursor(Qt::ClosedHandCursor); // Arcball 自由旋转中
        return;
    }

    // 优先显示当前拖拽轴（若无则显示悬停轴）
    HoverAxis axis = (_leftDragging ? _dragAxis : _hoverAxis);
    if (axis == HoverAxis::X) {
        setCursor(axisCursor(QColor(255, 120, 120), QStringLiteral("X")));
    } else if (axis == HoverAxis::Y) {
        setCursor(axisCursor(QColor(120, 255, 120), QStringLiteral("Y")));
    } else if (axis == HoverAxis::Z) {
        setCursor(axisCursor(QColor(120, 180, 255), QStringLiteral("Z")));
    } else {
        setCursor(Qt::OpenHandCursor); // 空闲时显示开放手型
    }
}

void CameraSceneWidget::initialize(QRhiCommandBuffer *cb)
{
    Q_UNUSED(cb);

    _renderError.clear();
    _rhiReady = rhi() && api() == QRhiWidget::Api::Vulkan;
    if (!_rhiReady)
    {
        _renderError = QStringLiteral("Vulkan 渲染初始化失败，请检查显卡驱动和 Qt Vulkan 支持。");
        LOG_ERROR("%s", qPrintable(_renderError));
        return;
    }

    _colorPointPipeline.vertexShaderPath = QStringLiteral(":/shaders/camera_scene_color.vert.qsb");
    _colorPointPipeline.fragmentShaderPath = QStringLiteral(":/shaders/camera_scene_color.frag.qsb");
    _colorLinePipeline.vertexShaderPath = _colorPointPipeline.vertexShaderPath;
    _colorLinePipeline.fragmentShaderPath = _colorPointPipeline.fragmentShaderPath;
    _modelPointPipeline.vertexShaderPath = _colorPointPipeline.vertexShaderPath;
    _modelPointPipeline.fragmentShaderPath = _colorPointPipeline.fragmentShaderPath;
    _meshTrianglePipeline.vertexShaderPath = QStringLiteral(":/shaders/camera_scene_mesh.vert.qsb");
    _meshTrianglePipeline.fragmentShaderPath = QStringLiteral(":/shaders/camera_scene_mesh.frag.qsb");
    _meshPointPipeline.vertexShaderPath = _meshTrianglePipeline.vertexShaderPath;
    _meshPointPipeline.fragmentShaderPath = _meshTrianglePipeline.fragmentShaderPath;
    _texturedMeshPipeline.vertexShaderPath =
        QStringLiteral(":/shaders/camera_scene_textured_mesh.vert.qsb");
    _texturedMeshPipeline.fragmentShaderPath =
        QStringLiteral(":/shaders/camera_scene_textured_mesh.frag.qsb");

    _gpuDirty = true;
    _pipelinesDirty = true;
}

void CameraSceneWidget::releaseResources()
{
    _pointBuffer.vertexBuffer.reset();
    _meshBuffer.vertexBuffer.reset();
    _modelWireframeBuffer.vertexBuffer.reset();
    _modelPointBuffer.vertexBuffer.reset();
    _lineBuffer.vertexBuffer.reset();
    _colorPointPipeline.uniformBuffer.reset();
    _colorPointPipeline.bindings.reset();
    _colorPointPipeline.pipeline.reset();
    _colorLinePipeline.uniformBuffer.reset();
    _colorLinePipeline.bindings.reset();
    _colorLinePipeline.pipeline.reset();
    _modelPointPipeline.uniformBuffer.reset();
    _modelPointPipeline.bindings.reset();
    _modelPointPipeline.pipeline.reset();
    _meshTrianglePipeline.uniformBuffer.reset();
    _meshTrianglePipeline.bindings.reset();
    _meshTrianglePipeline.pipeline.reset();
    _meshPointPipeline.uniformBuffer.reset();
    _meshPointPipeline.bindings.reset();
    _meshPointPipeline.pipeline.reset();
    _texturedMeshPipeline.uniformBuffer.reset();
    _texturedMeshPipeline.texture.reset();
    _texturedMeshPipeline.sampler.reset();
    _texturedMeshPipeline.bindings.reset();
    _texturedMeshPipeline.pipeline.reset();
    _texturedMeshPipeline.textureSize = QSize();
    _texturedMeshPipeline.uploadedTexturePath.clear();
    _imagePipeline.vertexBuffer.reset();
    _imagePipeline.uniformBuffer.reset();
    _imagePipeline.texture.reset();
    _imagePipeline.sampler.reset();
    _imagePipeline.bindings.reset();
    _imagePipeline.pipeline.reset();
    _imagePipeline.textureSize = QSize();
    _imagePipeline.uploadedImageKey.clear();
    _thumbnailPipeline.resources.clear();
    _thumbnailPipeline.uniformBuffer.reset();
    _thumbnailPipeline.sampler.reset();
    _thumbnailPipeline.pipeline.reset();
    _thumbnailPipeline.resourcesDirty = true;
    _rhiReady = false;
    _gpuDirty = true;
    _pipelinesDirty = true;
}

void CameraSceneWidget::resizeEvent(QResizeEvent *event)
{
    QRhiWidget::resizeEvent(event);
    if (_overlayWidget)
    {
        _overlayWidget->setGeometry(rect());
        _overlayWidget->raise();
    }
    _pipelinesDirty = true;
    _gpuDirty = true;
}

void CameraSceneWidget::uploadGpuData()
{
    const bool use_prepared_obj_mesh = _preparedObjMeshBuffer
        && _modelColorMode == ModelColorMode::Texture
        && _cloud.hasFaces()
        && _preparedObjVertexCount > 0
        && !_preparedObjVertexData.isEmpty();
    auto assignBuffer = [](RhiBufferSet &buffer,
                           const QVector<float> &data,
                           int vertexCount,
                           int strideFloats)
    {
        buffer.vertexData = QByteArray(reinterpret_cast<const char *>(data.constData()),
                                       int(data.size() * sizeof(float)));
        buffer.vertexCount = vertexCount;
        buffer.strideBytes = strideFloats * int(sizeof(float));
        buffer.dirty = true;
    };

    _pointBuffer.vertexCount = 0;
    _modelPointBuffer.vertexCount = 0;
    _modelWireframeBuffer.vertexCount = 0;
    _lineBuffer.vertexCount = 0;
    _pointBuffer.vertexData.clear();
    _modelPointBuffer.vertexData.clear();
    _modelWireframeBuffer.vertexData.clear();
    _lineBuffer.vertexData.clear();
    if (use_prepared_obj_mesh)
    {
        _meshBuffer.vertexData = _preparedObjVertexData;
        _meshBuffer.vertexCount = _preparedObjVertexCount;
        _meshBuffer.strideBytes = _preparedObjStrideBytes;
        _meshBuffer.dirty = true;
    }
    else
    {
        _meshBuffer.vertexCount = 0;
        _meshBuffer.vertexData.clear();
    }

    // ── 1. 点云（_cloud，无面片；法向量可选，颜色直通）──────────────────────
    _pointCount = 0;
    _modelPtCount = 0;
    _modelWireframeVertCount = 0;
    if (!(_cloud.size() == 0) && !_cloud.hasFaces()) {
        const bool hasColors = _cloud.hasColors();
        _pointColorScale = 1.0f;
        if (hasColors)
        {
            float maximumColorComponent = 0.0f;
            for (std::size_t index = 0; index < _cloud.size(); ++index)
            {
                const plamatrix::Index pointIndex = static_cast<plamatrix::Index>(index);
                for (int component = 0; component < 3; ++component)
                {
                    const float value = _cloud.colors()->getValue(pointIndex, component);
                    if (std::isfinite(value))
                    {
                        maximumColorComponent = qMax(maximumColorComponent, value);
                    }
                }
            }
            // Metashape OBJ uses normalized RGB floats, while PLY commonly stores bytes.
            _pointColorScale = maximumColorComponent <= 1.0f ? 1.0f : (1.0f / 255.0f);
        }
        if (_isTiePointCloud)
        {
            double minimumElevation = std::numeric_limits<double>::infinity();
            double maximumElevation = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < _cloud.size(); ++i)
            {
                const double elevation =
                    _cloud.points()(static_cast<plamatrix::Index>(i), 2);
                if (std::isfinite(elevation))
                {
                    minimumElevation = std::min(minimumElevation, elevation);
                    maximumElevation = std::max(maximumElevation, elevation);
                }
            }
            _tiePointElevationRange = {minimumElevation, maximumElevation};

            if (_tiePointImageCounts.size() == static_cast<qsizetype>(_cloud.size()))
            {
                const auto [minimumCount, maximumCount] =
                    std::minmax_element(_tiePointImageCounts.cbegin(),
                                        _tiePointImageCounts.cend());
                _tiePointImageCountRange = {
                    static_cast<double>(*minimumCount),
                    static_cast<double>(*maximumCount)
                };
            }
            else
            {
                _tiePointImageCountRange = {};
                if (!_tiePointImageCounts.isEmpty())
                {
                    _tiePointMetadataError = tr("观测数据与连接点数量不一致");
                }
            }
        }

        QVector<float> data;
        data.reserve(static_cast<int>(_cloud.size()) * 6);
        for (std::size_t i = 0; i < _cloud.size(); ++i) {
            const plamatrix::Index pointIndex = static_cast<plamatrix::Index>(i);
            const float x = _cloud.points()(pointIndex, 0);
            const float y = _cloud.points()(pointIndex, 1);
            const float z = _cloud.points()(pointIndex, 2);
            float red = 0.45f;
            float green = 0.45f;
            float blue = 0.50f;
            if (_isTiePointCloud && _tiePointColorMode == TiePointColorMode::Elevation)
            {
                const QColor color = xjw::gui::tie_points::elevationColor(
                    z,
                    _tiePointElevationRange);
                red = color.redF();
                green = color.greenF();
                blue = color.blueF();
            }
            else if (_isTiePointCloud &&
                     _tiePointColorMode == TiePointColorMode::ImageCount &&
                     _tiePointImageCounts.size() == static_cast<qsizetype>(_cloud.size()))
            {
                const QColor color = xjw::gui::tie_points::imageCountColor(
                    _tiePointImageCounts.at(static_cast<qsizetype>(i)),
                    _tiePointImageCountRange);
                red = color.redF();
                green = color.greenF();
                blue = color.blueF();
            }
            else if (hasColors) {
                red = qBound(0.0f,
                             _cloud.colors()->getValue(pointIndex, 0) * _pointColorScale,
                             1.0f);
                green = qBound(0.0f,
                               _cloud.colors()->getValue(pointIndex, 1) * _pointColorScale,
                               1.0f);
                blue = qBound(0.0f,
                              _cloud.colors()->getValue(pointIndex, 2) * _pointColorScale,
                              1.0f);
            }

            data << x << y << z << red << green << blue;
        }
        if (_preferModelPointRender && !_isTiePointCloud)
        {
            assignBuffer(_modelPointBuffer, data, int(_cloud.size()), 6);
            _modelPtCount = (int)_cloud.size();
        }
        else
        {
            assignBuffer(_pointBuffer, data, int(_cloud.size()), 6);
            _pointCount = (int)_cloud.size();
        }
    }

    // ── 2. 网格（hasFaces）──────────────────────────────────────────────────
    _meshVertCount = 0;
    _meshHasFaces = false;
    _meshHasTexture = false;
    if (!(_cloud.size() == 0) && _cloud.hasFaces()) {
        _meshHasFaces = true;
        if (use_prepared_obj_mesh)
        {
            _meshVertCount = _meshBuffer.vertexCount;
            _meshHasTexture = _preparedObjMeshHasTexture;
        }
        else
        {
        const bool hasVertCol = _cloud.hasColors();
        const bool hasNrm = _cloud.hasNormals();
        const std::size_t Nv = _cloud.size();
        bool hasTexture = !_meshTextureImage.isNull()
            && _cloud.hasTextureCoords()
            && _cloud.hasFaceTextureIndices()
            && _cloud.faceTextureIndices()->rows() == _cloud.faces()->rows();
        if (hasTexture)
        {
            const int textureCoordinateCount = _cloud.textureCoords()->rows();
            for (int faceIndex = 0; faceIndex < _cloud.faceTextureIndices()->rows() && hasTexture;
                 ++faceIndex)
            {
                for (int corner = 0; corner < 3; ++corner)
                {
                    const int textureIndex = _cloud.faceTextureIndices()->getValue(faceIndex, corner);
                    hasTexture = textureIndex >= 0 && textureIndex < textureCoordinateCount;
                    if (!hasTexture)
                    {
                        break;
                    }
                }
            }
        }
        const bool renderTexture =
            _modelColorMode == ModelColorMode::Texture && hasTexture;
        _meshHasTexture = renderTexture;

        double minimumElevation = std::numeric_limits<double>::infinity();
        double maximumElevation = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < Nv; ++index)
        {
            const double elevation = _cloud.points()(
                static_cast<plamatrix::Index>(index), 2);
            if (std::isfinite(elevation))
            {
                minimumElevation = std::min(minimumElevation, elevation);
                maximumElevation = std::max(maximumElevation, elevation);
            }
        }
        _modelElevationRange = {minimumElevation, maximumElevation};

        // 纹理模式保留 OBJ 的 UV 展开；其他模型视图统一交给
        // ModelVisualizationManager 生成颜色和法线，避免两套逻辑漂移。
        const int meshStrideFloats = renderTexture ? 11 : 9;
        QVector<float> data;
        data.reserve(static_cast<int>(_cloud.faces()->rows()) * 3 * meshStrideFloats);
        QVector<float> wireframeData;
        if (renderTexture)
        {
            // 计算（或复用）逐顶点法向量
            std::vector<QVector3D> vNormals(Nv);
            if (hasNrm)
            {
                for (std::size_t i = 0; i < Nv; ++i)
                {
                    vNormals[i] = QVector3D(_cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0),
                                            _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1),
                                            _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2));
                }
            }
            else
            {
                const auto nF = static_cast<std::size_t>(_cloud.faces()->rows());
                for (std::size_t fi = 0; fi < nF; ++fi)
                {
                    const std::size_t i0 =
                        static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 0));
                    const std::size_t i1 =
                        static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 1));
                    const std::size_t i2 =
                        static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 2));
                    if (i0 >= Nv || i1 >= Nv || i2 >= Nv)
                        continue;
                    float p0x = _cloud.points()(static_cast<plamatrix::Index>(i0), 0);
                    float p0y = _cloud.points()(static_cast<plamatrix::Index>(i0), 1);
                    float p0z = _cloud.points()(static_cast<plamatrix::Index>(i0), 2);
                    float p1x = _cloud.points()(static_cast<plamatrix::Index>(i1), 0);
                    float p1y = _cloud.points()(static_cast<plamatrix::Index>(i1), 1);
                    float p1z = _cloud.points()(static_cast<plamatrix::Index>(i1), 2);
                    float p2x = _cloud.points()(static_cast<plamatrix::Index>(i2), 0);
                    float p2y = _cloud.points()(static_cast<plamatrix::Index>(i2), 1);
                    float p2z = _cloud.points()(static_cast<plamatrix::Index>(i2), 2);
                    const QVector3D fn = QVector3D::crossProduct(QVector3D(p1x - p0x, p1y - p0y, p1z - p0z),
                                                                 QVector3D(p2x - p0x, p2y - p0y, p2z - p0z));
                    vNormals[i0] += fn;
                    vNormals[i1] += fn;
                    vNormals[i2] += fn;
                }
                for (auto& n : vNormals)
                    n.normalize();
            }

            // 展开面片并携带面角 UV，避免共享顶点跨相机接缝时丢失纹理坐标。
            const auto nFaces = static_cast<std::size_t>(_cloud.faces()->rows());
            for (std::size_t fi = 0; fi < nFaces; ++fi)
            {
                const std::size_t i0 =
                    static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 0));
                const std::size_t i1 =
                    static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 1));
                const std::size_t i2 =
                    static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 2));
                if (i0 >= Nv || i1 >= Nv || i2 >= Nv)
                    continue;
                const std::size_t indices[3] = {i0, i1, i2};
                const QVector3D positions[3] = {QVector3D(_cloud.points()(static_cast<plamatrix::Index>(i0), 0),
                                                          _cloud.points()(static_cast<plamatrix::Index>(i0), 1),
                                                          _cloud.points()(static_cast<plamatrix::Index>(i0), 2)),
                                                QVector3D(_cloud.points()(static_cast<plamatrix::Index>(i1), 0),
                                                          _cloud.points()(static_cast<plamatrix::Index>(i1), 1),
                                                          _cloud.points()(static_cast<plamatrix::Index>(i1), 2)),
                                                QVector3D(_cloud.points()(static_cast<plamatrix::Index>(i2), 0),
                                                          _cloud.points()(static_cast<plamatrix::Index>(i2), 1),
                                                          _cloud.points()(static_cast<plamatrix::Index>(i2), 2))};
                QVector3D faceNormal =
                    QVector3D::crossProduct(positions[1] - positions[0], positions[2] - positions[0]);
                faceNormal.normalize();

                if (_modelColorMode == ModelColorMode::Wireframe)
                {
                    const QColor wireColor = xjw::gui::model_views::surfaceColor(ModelColorMode::Wireframe);
                    const int edgeCorners[6] = {0, 1, 1, 2, 2, 0};
                    for (int edgeVertex : edgeCorners)
                    {
                        const QVector3D& position = positions[edgeVertex];
                        wireframeData << position.x() << position.y() << position.z() << wireColor.redF()
                                      << wireColor.greenF() << wireColor.blueF();
                    }
                }

                for (int vi = 0; vi < 3; ++vi)
                {
                    const std::size_t idx = indices[vi];
                    const QVector3D& position = positions[vi];
                    const QVector3D normal = _modelColorMode == ModelColorMode::Solid ? faceNormal : vNormals[idx];
                    data << position.x() << position.y() << position.z();
                    data << normal.x() << normal.y() << normal.z();

                    if (_modelColorMode == ModelColorMode::Shaded || _modelColorMode == ModelColorMode::Solid)
                    {
                        const QColor color = xjw::gui::model_views::surfaceColor(_modelColorMode);
                        data << color.redF() << color.greenF() << color.blueF();
                    }
                    else if (_modelColorMode == ModelColorMode::Elevation)
                    {
                        const QColor color = xjw::gui::model_views::elevationColor(position.z(), _modelElevationRange);
                        data << color.redF() << color.greenF() << color.blueF();
                    }
                    else if (_modelColorMode == ModelColorMode::Wireframe)
                    {
                        const QColor color = xjw::gui::model_views::surfaceColor(ModelColorMode::Wireframe);
                        data << color.redF() << color.greenF() << color.blueF();
                    }
                    else if (hasVertCol)
                    {
                        data << _cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 0) / 255.f
                             << _cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 1) / 255.f
                             << _cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 2) / 255.f;
                    }
                    else
                    {
                        if (renderTexture)
                        {
                            data << -1.0f << -1.0f << -1.0f;
                        }
                        else
                        {
                            data << 0.55f << 0.55f << 0.58f;
                        }
                    }
                    if (renderTexture)
                    {
                        const int textureIndex =
                            _cloud.faceTextureIndices()->getValue(static_cast<plamatrix::Index>(fi), vi);
                        data << _cloud.textureCoords()->getValue(textureIndex, 0)
                             << _cloud.textureCoords()->getValue(textureIndex, 1);
                    }
                }
            }
        }
        else
        {
            xjw::gui::model_views::GeometryInput geometryInput;
            geometryInput.positions.reserve(static_cast<qsizetype>(Nv));
            if (hasNrm)
            {
                geometryInput.vertexNormals.reserve(static_cast<qsizetype>(Nv));
            }
            if (hasVertCol)
            {
                geometryInput.vertexColors.reserve(static_cast<qsizetype>(Nv));
            }
            for (std::size_t vertexIndex = 0; vertexIndex < Nv; ++vertexIndex)
            {
                const auto index = static_cast<plamatrix::Index>(vertexIndex);
                geometryInput.positions.push_back(QVector3D(
                    _cloud.points()(index, 0),
                    _cloud.points()(index, 1),
                    _cloud.points()(index, 2)));
                if (hasNrm)
                {
                    geometryInput.vertexNormals.push_back(QVector3D(
                        _cloud.normals()->getValue(index, 0),
                        _cloud.normals()->getValue(index, 1),
                        _cloud.normals()->getValue(index, 2)));
                }
                if (hasVertCol)
                {
                    geometryInput.vertexColors.push_back(QColor(
                        _cloud.colors()->getValue(index, 0),
                        _cloud.colors()->getValue(index, 1),
                        _cloud.colors()->getValue(index, 2)));
                }
            }

            geometryInput.faces.reserve(_cloud.faces()->rows());
            for (int faceIndex = 0; faceIndex < _cloud.faces()->rows(); ++faceIndex)
            {
                xjw::gui::model_views::Triangle triangle;
                for (int corner = 0; corner < 3; ++corner)
                {
                    triangle.vertexIndices[corner] =
                        _cloud.faces()->getValue(faceIndex, corner);
                }
                geometryInput.faces.push_back(triangle);
            }
            _modelVisualization.setMode(_modelColorMode);
            const xjw::gui::model_views::GeometryResult geometry =
                _modelVisualization.buildGeometry(geometryInput);
            data = geometry.filledVertices;
            wireframeData = geometry.wireframeVertices;
            _modelElevationRange = geometry.elevationRange;
        }

        _meshVertCount = data.size() / meshStrideFloats;
        assignBuffer(_meshBuffer, data, _meshVertCount, meshStrideFloats);
        if (!wireframeData.isEmpty())
        {
            _modelWireframeVertCount = wireframeData.size() / 6;
            assignBuffer(_modelWireframeBuffer,
                         wireframeData,
                         _modelWireframeVertCount,
                         6);
        }
        }
    }

    // ── 3. 点云包围盒 ────────────────────────────────────────────────────
    // 只使用点云自身的边界，避免相机轨迹把包围盒扩展到整个场景。
    if (_cacheDirty)
    {
        invalidateCache();
    }
    if (_hasCloudBounds && !_cloud.hasFaces())
    {
        const QVector<QVector3D> vertices =
            xjw::gui::camera_scene::axisAlignedBoundingBoxLineVertices(
                _cachedCloudAABBMin, _cachedCloudAABBMax);
        QVector<float> line_data;
        line_data.reserve(vertices.size() * 6);
        for (const QVector3D &vertex : vertices)
        {
            line_data << vertex.x() << vertex.y() << vertex.z()
                      << 0.52f << 0.58f << 0.66f;
        }
        _lineCount = static_cast<int>(vertices.size());
        assignBuffer(_lineBuffer, line_data, _lineCount, 6);
    }
    else
    {
        _lineCount = 0;
    }

    if (_cloud.size() > 0)
    {
        LOG_INFO(QStringLiteral(
                     "[3D] GPU 几何缓存已准备: 点=%1，普通点缓冲=%2，模型点缓冲=%3，网格顶点=%4，"
                     "法向量=%5，颜色=%6，面=%7")
                     .arg(_cloud.size())
                     .arg(_pointBuffer.vertexCount)
                     .arg(_modelPointBuffer.vertexCount)
                     .arg(_meshBuffer.vertexCount)
                     .arg(_cloud.hasNormals() ? QStringLiteral("有") : QStringLiteral("无"))
                     .arg(_cloud.hasColors() ? QStringLiteral("有") : QStringLiteral("无"))
                     .arg(_cloud.hasFaces()
                              ? static_cast<int>(_cloud.faces()->rows())
                              : 0));
    }

    _thumbnailPipeline.resourcesDirty = true;
    _pipelinesDirty = true;
    _gpuDirty = false;
}



bool CameraSceneWidget::ensureRhiBuffer(RhiBufferSet *buffer, QRhiResourceUpdateBatch *updates)
{
    if (!buffer || !updates || buffer->vertexData.isEmpty() || buffer->vertexCount <= 0)
    {
        return true;
    }

    const quint32 byteCount = quint32(buffer->vertexData.size());
    if (!buffer->vertexBuffer || buffer->vertexBuffer->size() != byteCount)
    {
        buffer->vertexBuffer.reset(rhi()->newBuffer(QRhiBuffer::Static, QRhiBuffer::VertexBuffer, byteCount));
        if (!buffer->vertexBuffer->create())
        {
            _renderError = QStringLiteral("Vulkan 顶点缓冲创建失败。");
            return false;
        }
        buffer->dirty = true;
    }

    if (buffer->dirty)
    {
        updates->uploadStaticBuffer(buffer->vertexBuffer.data(), buffer->vertexData);
        buffer->dirty = false;
    }
    return true;
}

bool CameraSceneWidget::ensurePipeline(RhiPipelineSet *pipeline,
                                       int topology,
                                       int strideBytes,
                                       bool hasNormals)
{
    if (!pipeline)
    {
        return false;
    }
    if (pipeline->pipeline && !_pipelinesDirty)
    {
        return true;
    }

    QString error;
    const QShader vertexShader = loadSceneShader(pipeline->vertexShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragmentShader = loadSceneShader(pipeline->fragmentShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    pipeline->pipeline.reset();
    pipeline->bindings.reset();
    pipeline->uniformBuffer.reset();

    pipeline->uniformBuffer.reset(rhi()->newBuffer(QRhiBuffer::Dynamic,
                                                   QRhiBuffer::UniformBuffer,
                                                   quint32(sizeof(SceneUniforms))));
    if (!pipeline->uniformBuffer->create())
    {
        _renderError = QStringLiteral("Vulkan uniform 缓冲创建失败。");
        return false;
    }

    pipeline->bindings.reset(rhi()->newShaderResourceBindings());
    pipeline->bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            pipeline->uniformBuffer.data())
    });
    if (!pipeline->bindings->create())
    {
        _renderError = QStringLiteral("Vulkan shader 资源绑定创建失败。");
        return false;
    }

    QRhiVertexInputLayout inputLayout;
    inputLayout.setBindings({ QRhiVertexInputBinding(quint32(strideBytes)) });
    if (hasNormals)
    {
        inputLayout.setAttributes({
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
            QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3, 6 * sizeof(float)),
        });
    }
    else
    {
        inputLayout.setAttributes({
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
        });
    }

    pipeline->pipeline.reset(rhi()->newGraphicsPipeline());
    pipeline->pipeline->setTopology(static_cast<QRhiGraphicsPipeline::Topology>(topology));
    pipeline->pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertexShader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragmentShader),
    });
    pipeline->pipeline->setVertexInputLayout(inputLayout);
    pipeline->pipeline->setShaderResourceBindings(pipeline->bindings.data());
    pipeline->pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    pipeline->pipeline->setSampleCount(sampleCount());
    pipeline->pipeline->setDepthTest(true);
    pipeline->pipeline->setDepthWrite(true);
    pipeline->pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    pipeline->pipeline->setCullMode(QRhiGraphicsPipeline::None);

    QRhiGraphicsPipeline::TargetBlend blend;
    blend.enable = true;
    blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
    blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    blend.srcAlpha = QRhiGraphicsPipeline::One;
    blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    pipeline->pipeline->setTargetBlends({ blend });

    if (!pipeline->pipeline->create())
    {
        _renderError = QStringLiteral("Vulkan 图形管线创建失败。");
        return false;
    }
    return true;
}

bool CameraSceneWidget::ensureTexturedMeshPipeline(QRhiResourceUpdateBatch *updates)
{
    if (!_meshHasTexture || _meshTextureImage.isNull() || !updates)
    {
        return true;
    }

    const bool recreateTexture = !_texturedMeshPipeline.texture
        || _texturedMeshPipeline.textureSize != _meshTextureImage.size();
    if (recreateTexture)
    {
        _texturedMeshPipeline.texture.reset(
            rhi()->newTexture(QRhiTexture::RGBA8, _meshTextureImage.size()));
        if (!_texturedMeshPipeline.texture->create())
        {
            _renderError = QStringLiteral("Vulkan 模型纹理创建失败：%1").arg(_meshTexturePath);
            return false;
        }
        _texturedMeshPipeline.textureSize = _meshTextureImage.size();
        _texturedMeshPipeline.uploadedTexturePath.clear();
        _texturedMeshPipeline.pipeline.reset();
        _texturedMeshPipeline.bindings.reset();
    }
    if (_texturedMeshPipeline.uploadedTexturePath != _meshTexturePath)
    {
        const QImage uploadImage = _meshTextureImage.convertToFormat(QImage::Format_RGBA8888);
        updates->uploadTexture(_texturedMeshPipeline.texture.data(), uploadImage);
        _texturedMeshPipeline.uploadedTexturePath = _meshTexturePath;
    }

    if (!_texturedMeshPipeline.sampler)
    {
        _texturedMeshPipeline.sampler.reset(rhi()->newSampler(QRhiSampler::Linear,
                                                              QRhiSampler::Linear,
                                                              QRhiSampler::None,
                                                              QRhiSampler::ClampToEdge,
                                                              QRhiSampler::ClampToEdge));
        if (!_texturedMeshPipeline.sampler->create())
        {
            _renderError = QStringLiteral("Vulkan 模型纹理采样器创建失败。");
            return false;
        }
    }
    if (_texturedMeshPipeline.pipeline && !_pipelinesDirty)
    {
        return true;
    }

    QString error;
    const QShader vertexShader = loadSceneShader(_texturedMeshPipeline.vertexShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragmentShader = loadSceneShader(_texturedMeshPipeline.fragmentShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    _texturedMeshPipeline.pipeline.reset();
    _texturedMeshPipeline.bindings.reset();
    _texturedMeshPipeline.uniformBuffer.reset(
        rhi()->newBuffer(QRhiBuffer::Dynamic,
                         QRhiBuffer::UniformBuffer,
                         quint32(sizeof(SceneUniforms))));
    if (!_texturedMeshPipeline.uniformBuffer->create())
    {
        _renderError = QStringLiteral("Vulkan 纹理模型 uniform 缓冲创建失败。");
        return false;
    }

    _texturedMeshPipeline.bindings.reset(rhi()->newShaderResourceBindings());
    _texturedMeshPipeline.bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            _texturedMeshPipeline.uniformBuffer.data()),
        QRhiShaderResourceBinding::sampledTexture(
            1,
            QRhiShaderResourceBinding::FragmentStage,
            _texturedMeshPipeline.texture.data(),
            _texturedMeshPipeline.sampler.data())
    });
    if (!_texturedMeshPipeline.bindings->create())
    {
        _renderError = QStringLiteral("Vulkan 纹理模型 shader 资源绑定创建失败。");
        return false;
    }

    QRhiVertexInputLayout inputLayout;
    inputLayout.setBindings({QRhiVertexInputBinding(quint32(_meshBuffer.strideBytes))});
    inputLayout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
        QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3, 6 * sizeof(float)),
        QRhiVertexInputAttribute(0, 3, QRhiVertexInputAttribute::Float2, 9 * sizeof(float)),
    });

    _texturedMeshPipeline.pipeline.reset(rhi()->newGraphicsPipeline());
    _texturedMeshPipeline.pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    _texturedMeshPipeline.pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertexShader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragmentShader),
    });
    _texturedMeshPipeline.pipeline->setVertexInputLayout(inputLayout);
    _texturedMeshPipeline.pipeline->setShaderResourceBindings(_texturedMeshPipeline.bindings.data());
    _texturedMeshPipeline.pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    _texturedMeshPipeline.pipeline->setSampleCount(sampleCount());
    _texturedMeshPipeline.pipeline->setDepthTest(true);
    _texturedMeshPipeline.pipeline->setDepthWrite(true);
    _texturedMeshPipeline.pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    _texturedMeshPipeline.pipeline->setCullMode(QRhiGraphicsPipeline::None);
    if (!_texturedMeshPipeline.pipeline->create())
    {
        _renderError = QStringLiteral("Vulkan 纹理模型图形管线创建失败。");
        return false;
    }
    return true;
}

void CameraSceneWidget::drawRhiBuffer(QRhiCommandBuffer *cb,
                                      RhiBufferSet *buffer,
                                      RhiPipelineSet *pipeline,
                                      const SceneUniforms &uniforms)
{
    if (!cb || !buffer || !pipeline || !buffer->vertexBuffer || buffer->vertexCount <= 0 || !pipeline->pipeline)
    {
        return;
    }

    pipeline->uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(&uniforms, sizeof(SceneUniforms));
    cb->setGraphicsPipeline(pipeline->pipeline.data());
    cb->setShaderResources(pipeline->bindings.data());
    const QRhiCommandBuffer::VertexInput vertexInput(buffer->vertexBuffer.data(), 0);
    cb->setVertexInput(0, 1, &vertexInput);
    cb->draw(quint32(buffer->vertexCount));
}

void CameraSceneWidget::drawTexturedMesh(QRhiCommandBuffer *cb, const SceneUniforms &uniforms)
{
    if (!cb || !_meshBuffer.vertexBuffer || _meshBuffer.vertexCount <= 0
        || !_texturedMeshPipeline.pipeline || !_texturedMeshPipeline.uniformBuffer)
    {
        return;
    }

    _texturedMeshPipeline.uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(
        &uniforms, sizeof(SceneUniforms));
    cb->setGraphicsPipeline(_texturedMeshPipeline.pipeline.data());
    cb->setShaderResources(_texturedMeshPipeline.bindings.data());
    const QRhiCommandBuffer::VertexInput vertexInput(_meshBuffer.vertexBuffer.data(), 0);
    cb->setVertexInput(0, 1, &vertexInput);
    cb->draw(quint32(_meshBuffer.vertexCount));
}

bool CameraSceneWidget::ensureImagePipeline(QRhiResourceUpdateBatch *updates)
{
    if (!_showCameraImage || !updates)
    {
        return true;
    }

    const int pose_index = displayedCameraImagePoseIndex();
    if (pose_index < 0 || pose_index >= _poses.size())
    {
        return true;
    }

    const CameraPose &pose = _poses.at(pose_index);
    requestCameraPlaneImage(pose.imagePath, CameraImagePlaneMode::Image);
    const QImage image = cachedCameraPlaneImage(pose.imagePath, CameraImagePlaneMode::Image);
    if (image.isNull())
    {
        return true;
    }

    if (!_imagePipeline.vertexBuffer)
    {
        _imagePipeline.vertexBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic, QRhiBuffer::VertexBuffer, 6 * 5 * sizeof(float)));
        if (!_imagePipeline.vertexBuffer->create())
        {
            _renderError = QStringLiteral("Vulkan 照片顶点缓冲创建失败。");
            return false;
        }
    }
    if (!_imagePipeline.uniformBuffer)
    {
        _imagePipeline.uniformBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(ImagePlaneUniforms)));
        if (!_imagePipeline.uniformBuffer->create())
        {
            _renderError = QStringLiteral("Vulkan 照片矩阵缓冲创建失败。");
            return false;
        }
    }

    const QString image_key = cameraPlaneImageKey(pose.imagePath, CameraImagePlaneMode::Image);
    const bool recreate_texture = !_imagePipeline.texture || _imagePipeline.textureSize != image.size();
    if (recreate_texture)
    {
        _imagePipeline.texture.reset(rhi()->newTexture(QRhiTexture::RGBA8, image.size()));
        if (!_imagePipeline.texture->create())
        {
            _renderError = QStringLiteral("Vulkan 照片纹理创建失败：%1").arg(pose.imagePath);
            return false;
        }
        _imagePipeline.textureSize = image.size();
        _imagePipeline.uploadedImageKey.clear();
        _imagePipeline.pipeline.reset();
        _imagePipeline.bindings.reset();
    }
    if (_imagePipeline.uploadedImageKey != image_key)
    {
        // 三维相机卡片是实体遮挡面；忽略源图可能携带的 alpha，
        // 否则连接点会从照片透明通道中穿出。
        QImage upload_image = image.convertToFormat(QImage::Format_RGBX8888);
        if (rhi()->isYUpInNDC())
        {
            upload_image = upload_image.mirrored();
        }
        updates->uploadTexture(_imagePipeline.texture.data(), upload_image);
        _imagePipeline.uploadedImageKey = image_key;
    }

    if (!_imagePipeline.sampler)
    {
        _imagePipeline.sampler.reset(rhi()->newSampler(QRhiSampler::Linear,
                                                       QRhiSampler::Linear,
                                                       QRhiSampler::None,
                                                       QRhiSampler::ClampToEdge,
                                                       QRhiSampler::ClampToEdge));
        if (!_imagePipeline.sampler->create())
        {
            _renderError = QStringLiteral("Vulkan 照片采样器创建失败。");
            return false;
        }
    }

    if (_imagePipeline.pipeline && !_pipelinesDirty)
    {
        return true;
    }

    QString error;
    const QShader vertex_shader = loadSceneShader(
        QStringLiteral(":/shaders/camera_scene_image.vert.qsb"), &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragment_shader = loadSceneShader(
        QStringLiteral(":/shaders/camera_scene_image.frag.qsb"), &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    _imagePipeline.pipeline.reset();
    _imagePipeline.bindings.reset(rhi()->newShaderResourceBindings());
    _imagePipeline.bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(0,
                                                  QRhiShaderResourceBinding::VertexStage,
                                                  _imagePipeline.uniformBuffer.data()),
        QRhiShaderResourceBinding::sampledTexture(1,
                                                  QRhiShaderResourceBinding::FragmentStage,
                                                  _imagePipeline.texture.data(),
                                                  _imagePipeline.sampler.data())
    });
    if (!_imagePipeline.bindings->create())
    {
        _renderError = QStringLiteral("Vulkan 照片 shader 资源绑定创建失败。");
        return false;
    }

    QRhiVertexInputLayout input_layout;
    input_layout.setBindings({QRhiVertexInputBinding(5 * sizeof(float))});
    input_layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float2, 3 * sizeof(float)),
    });

    _imagePipeline.pipeline.reset(rhi()->newGraphicsPipeline());
    _imagePipeline.pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    _imagePipeline.pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertex_shader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragment_shader),
    });
    _imagePipeline.pipeline->setVertexInputLayout(input_layout);
    _imagePipeline.pipeline->setShaderResourceBindings(_imagePipeline.bindings.data());
    _imagePipeline.pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    _imagePipeline.pipeline->setSampleCount(sampleCount());
    _imagePipeline.pipeline->setDepthTest(false);
    _imagePipeline.pipeline->setDepthWrite(false);
    _imagePipeline.pipeline->setCullMode(QRhiGraphicsPipeline::None);
    if (!_imagePipeline.pipeline->create())
    {
        _renderError = QStringLiteral("Vulkan 照片合成管线创建失败。");
        return false;
    }
    return true;
}

bool CameraSceneWidget::ensureCameraThumbnailPipeline(QRhiResourceUpdateBatch *updates)
{
    if (!_showCameras || !updates)
    {
        return true;
    }

    if (_thumbnailPipeline.resourcesDirty)
    {
        _thumbnailPipeline.resources.clear();
        _thumbnailPipeline.pipeline.reset();
        _thumbnailPipeline.resourcesDirty = false;
    }

    if (!_thumbnailPipeline.uniformBuffer)
    {
        _thumbnailPipeline.uniformBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(ImagePlaneUniforms)));
        if (!_thumbnailPipeline.uniformBuffer->create())
        {
            _renderError = QStringLiteral("Vulkan 相机缩略图矩阵缓冲创建失败。");
            return false;
        }
    }
    if (!_thumbnailPipeline.sampler)
    {
        _thumbnailPipeline.sampler.reset(rhi()->newSampler(QRhiSampler::Linear,
                                                           QRhiSampler::Linear,
                                                           QRhiSampler::None,
                                                           QRhiSampler::ClampToEdge,
                                                           QRhiSampler::ClampToEdge));
        if (!_thumbnailPipeline.sampler->create())
        {
            _renderError = QStringLiteral("Vulkan 相机缩略图采样器创建失败。");
            return false;
        }
    }

    for (qsizetype pose_index = 0; pose_index < _poses.size(); ++pose_index)
    {
        const CameraPose &pose = _poses.at(pose_index);
        const CameraImagePlaneMode planeMode = _showCameraThumbnails
            ? CameraImagePlaneMode::Thumbnail
            : CameraImagePlaneMode::Solid;
        QImage image;
        if (planeMode == CameraImagePlaneMode::Thumbnail)
        {
            if (pose.imagePath.isEmpty())
            {
                continue;
            }
            requestCameraPlaneImage(pose.imagePath, planeMode);
            image = cachedCameraPlaneImage(pose.imagePath, planeMode);
        }
        else
        {
            const bool highlighted = isCameraHighlighted(pose);
            image = QImage(QSize(1, 1), QImage::Format_RGBA8888);
            image.fill(highlighted ? QColor(205, 60, 70, 230)
                                   : QColor(57, 112, 173, 220));
        }
        if (image.isNull())
        {
            continue;
        }

        QString planeKey = cameraPlaneImageKey(pose.imagePath, planeMode);
        if (planeKey.isEmpty())
        {
            planeKey = QStringLiteral("solid|") + pose.name;
        }
        const QString resource_key = planeKey + QLatin1Char('#') + QString::number(pose_index);
        if (_thumbnailPipeline.resources.contains(resource_key))
        {
            continue;
        }

        auto resource = QSharedPointer<RhiCameraThumbnailResource>::create();
        resource->vertexBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic,
            QRhiBuffer::VertexBuffer,
            6 * 5 * int(sizeof(float))));
        if (!resource->vertexBuffer->create())
        {
            _renderError = QStringLiteral("Vulkan 相机缩略图顶点缓冲创建失败。");
            return false;
        }

        resource->texture.reset(rhi()->newTexture(QRhiTexture::RGBA8, image.size()));
        if (!resource->texture->create())
        {
            _renderError = QStringLiteral("Vulkan 相机缩略图纹理创建失败：%1").arg(pose.imagePath);
            return false;
        }
        // 相机缩略图参与场景遮挡，统一上传为不透明纹理。
        QImage upload_image = image.convertToFormat(QImage::Format_RGBX8888);
        if (rhi()->isYUpInNDC())
        {
            upload_image = upload_image.mirrored();
        }
        updates->uploadTexture(resource->texture.data(), upload_image);

        resource->bindings.reset(rhi()->newShaderResourceBindings());
        resource->bindings->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0,
                                                      QRhiShaderResourceBinding::VertexStage,
                                                      _thumbnailPipeline.uniformBuffer.data()),
            QRhiShaderResourceBinding::sampledTexture(1,
                                                      QRhiShaderResourceBinding::FragmentStage,
                                                      resource->texture.data(),
                                                      _thumbnailPipeline.sampler.data())
        });
        if (!resource->bindings->create())
        {
            _renderError = QStringLiteral("Vulkan 相机缩略图资源绑定创建失败。");
            return false;
        }
        _thumbnailPipeline.resources.insert(resource_key, resource);
    }

    if (_thumbnailPipeline.resources.isEmpty())
    {
        return true;
    }
    if (_thumbnailPipeline.pipeline && !_pipelinesDirty)
    {
        return true;
    }

    QString error;
    const QShader vertex_shader = loadSceneShader(
        QStringLiteral(":/shaders/camera_scene_image.vert.qsb"), &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragment_shader = loadSceneShader(
        QStringLiteral(":/shaders/camera_scene_image.frag.qsb"), &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    QRhiVertexInputLayout input_layout;
    input_layout.setBindings({QRhiVertexInputBinding(5 * sizeof(float))});
    input_layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float2, 3 * sizeof(float)),
    });

    _thumbnailPipeline.pipeline.reset(rhi()->newGraphicsPipeline());
    _thumbnailPipeline.pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    _thumbnailPipeline.pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertex_shader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragment_shader),
    });
    _thumbnailPipeline.pipeline->setVertexInputLayout(input_layout);
    _thumbnailPipeline.pipeline->setShaderResourceBindings(
        _thumbnailPipeline.resources.constBegin().value()->bindings.data());
    _thumbnailPipeline.pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    _thumbnailPipeline.pipeline->setSampleCount(sampleCount());
    _thumbnailPipeline.pipeline->setDepthTest(true);
    _thumbnailPipeline.pipeline->setDepthWrite(true);
    _thumbnailPipeline.pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    _thumbnailPipeline.pipeline->setCullMode(QRhiGraphicsPipeline::None);
    if (!_thumbnailPipeline.pipeline->create())
    {
        _renderError = QStringLiteral("Vulkan 相机缩略图图形管线创建失败。");
        return false;
    }
    return true;
}

void CameraSceneWidget::drawCameraThumbnails(QRhiCommandBuffer *cb, const QMatrix4x4 &mvp)
{
    if (!cb || !_showCameras || !_thumbnailPipeline.pipeline
        || !_thumbnailPipeline.uniformBuffer)
    {
        return;
    }

    ImagePlaneUniforms uniforms;
    uniforms.mvp = mvp;
    _thumbnailPipeline.uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(&uniforms, sizeof(uniforms));
    cb->setGraphicsPipeline(_thumbnailPipeline.pipeline.data());

    QVector<QVector3D> camera_centers;
    camera_centers.reserve(_poses.size());
    for (const CameraPose &pose : _poses)
    {
        camera_centers.push_back(pose.center);
    }
    const SceneMatrices matrices = sceneMatrices();
    const QVector<int> camera_draw_order = xjw::gui::camera_scene::farToNearCameraIndices(
        camera_centers, matrices.modelView);
    for (const int pose_index : camera_draw_order)
    {
        const CameraPose &pose = _poses.at(pose_index);
        const CameraImagePlaneMode planeMode = _showCameraThumbnails
            ? CameraImagePlaneMode::Thumbnail
            : CameraImagePlaneMode::Solid;
        QString planeKey = cameraPlaneImageKey(pose.imagePath, planeMode);
        if (planeKey.isEmpty())
        {
            planeKey = QStringLiteral("solid|") + pose.name;
        }
        const QString resource_key = planeKey + QLatin1Char('#') + QString::number(pose_index);
        const auto resource = _thumbnailPipeline.resources.value(resource_key);
        if (!resource || !resource->vertexBuffer || !resource->bindings)
        {
            continue;
        }

        const float plane_half_extent = cameraImagePlaneHalfExtent(
            pose, matrices.modelView);
        const float plane_half_height = plane_half_extent * 0.68f;
        const xjw::gui::camera_scene::CameraImagePlaneAxes axes =
            xjw::gui::camera_scene::cameraImagePlaneAxes(
                pose.rotation, pose.uAxisSign, pose.vAxisSign);
        const QVector<QVector3D> corners = xjw::gui::camera_scene::cameraImagePlaneCorners(
            pose.center, axes.right, axes.up, plane_half_extent, plane_half_height);
        if (corners.size() != 4)
        {
            continue;
        }

        const QVector3D &p1 = corners.at(0);
        const QVector3D &p2 = corners.at(1);
        const QVector3D &p3 = corners.at(2);
        const QVector3D &p4 = corners.at(3);
        const float vertices[] = {
            p1.x(), p1.y(), p1.z(), 1.0f, 0.0f,
            p2.x(), p2.y(), p2.z(), 0.0f, 0.0f,
            p3.x(), p3.y(), p3.z(), 0.0f, 1.0f,
            p1.x(), p1.y(), p1.z(), 1.0f, 0.0f,
            p3.x(), p3.y(), p3.z(), 0.0f, 1.0f,
            p4.x(), p4.y(), p4.z(), 1.0f, 1.0f,
        };
        resource->vertexBuffer->fullDynamicBufferUpdateForCurrentFrame(
            vertices, sizeof(vertices));
        cb->setShaderResources(resource->bindings.data());
        const QRhiCommandBuffer::VertexInput vertex_input(resource->vertexBuffer.data(), 0);
        cb->setVertexInput(0, 1, &vertex_input);
        cb->draw(6);
    }
}

void CameraSceneWidget::drawActiveCameraImage(QRhiCommandBuffer *cb, const QMatrix4x4 &mvp)
{
    if (!cb || !_showCameraImage || !_imagePipeline.pipeline || !_imagePipeline.vertexBuffer
        || !_imagePipeline.uniformBuffer)
    {
        return;
    }

    const int pose_index = displayedCameraImagePoseIndex();
    if (pose_index < 0 || pose_index >= _poses.size())
    {
        return;
    }
    const QString active_key = cameraPlaneImageKey(
        _poses.at(pose_index).imagePath, CameraImagePlaneMode::Image);
    if (_imagePipeline.uploadedImageKey != active_key)
    {
        return;
    }

    const QVector<QVector3D> corners = displayedCameraImagePlaneCorners();
    if (corners.size() != 4)
    {
        return;
    }

    const QVector3D &p1 = corners.at(0);
    const QVector3D &p2 = corners.at(1);
    const QVector3D &p3 = corners.at(2);
    const QVector3D &p4 = corners.at(3);
    const float vertices[] = {
        p1.x(), p1.y(), p1.z(), 1.0f, 0.0f,
        p2.x(), p2.y(), p2.z(), 0.0f, 0.0f,
        p3.x(), p3.y(), p3.z(), 0.0f, 1.0f,
        p1.x(), p1.y(), p1.z(), 1.0f, 0.0f,
        p3.x(), p3.y(), p3.z(), 0.0f, 1.0f,
        p4.x(), p4.y(), p4.z(), 1.0f, 1.0f,
    };
    ImagePlaneUniforms uniforms;
    uniforms.mvp = mvp;
    _imagePipeline.vertexBuffer->fullDynamicBufferUpdateForCurrentFrame(vertices, sizeof(vertices));
    _imagePipeline.uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(&uniforms, sizeof(uniforms));
    cb->setGraphicsPipeline(_imagePipeline.pipeline.data());
    cb->setShaderResources(_imagePipeline.bindings.data());
    const QRhiCommandBuffer::VertexInput vertex_input(_imagePipeline.vertexBuffer.data(), 0);
    cb->setVertexInput(0, 1, &vertex_input);
    cb->draw(6);
}

QVector<QVector3D> CameraSceneWidget::displayedCameraImagePlaneCorners() const
{
    const int poseIndex = displayedCameraImagePoseIndex();
    if (poseIndex < 0 || poseIndex >= _poses.size())
    {
        return {};
    }

    const CameraPose &pose = _poses.at(poseIndex);
    const QImage image = cachedCameraPlaneImage(pose.imagePath, CameraImagePlaneMode::Image);
    if (image.isNull())
    {
        return {};
    }

    const int imageWidth = pose.imageWidth > 0 ? pose.imageWidth : image.width();
    const int imageHeight = pose.imageHeight > 0 ? pose.imageHeight : image.height();
    const QVector3D forward = xjw::gui::camera_scene::cameraForwardDirection(
        pose.rotation, pose.depthAxisFlipped);
    const xjw::gui::camera_scene::CameraImagePlaneAxes axes =
        xjw::gui::camera_scene::cameraImagePlaneAxes(
            pose.rotation, pose.uAxisSign, pose.vAxisSign);
    return xjw::gui::camera_scene::calibratedImagePlaneCorners(
        pose.center,
        forward,
        axes.right,
        axes.up,
        sceneCenter(),
        pose.focalX,
        pose.focalY,
        pose.principalX,
        pose.principalY,
        imageWidth,
        imageHeight);
}

QPainterPath CameraSceneWidget::foregroundCameraImageOcclusionPath() const
{
    QPainterPath occlusionPath;
    if (!_showCameraImage
        || _cameraImageDisplayLayer != CameraImageDisplayLayer::Foreground)
    {
        return occlusionPath;
    }

    const QVector<QVector3D> corners = displayedCameraImagePlaneCorners();
    if (corners.size() != 4)
    {
        return occlusionPath;
    }

    QPolygonF projectedPlane;
    projectedPlane.reserve(corners.size());
    for (const QVector3D &corner : corners)
    {
        bool projected = false;
        const QPointF screenPoint = projectToScreen(corner, &projected);
        if (!projected)
        {
            return {};
        }
        projectedPlane.push_back(screenPoint);
    }

    occlusionPath.addPolygon(projectedPlane);
    occlusionPath.closeSubpath();
    return occlusionPath;
}

void CameraSceneWidget::drawSceneGeometry(QRhiCommandBuffer *cb, SceneUniforms &uniforms)
{
    const float pointDiameter = _isTiePointCloud
        ? qMax(2.4f, xjw::gui::tie_points::pointSizeForMode(_tiePointColorMode))
        : 1.8f;
    uniforms.lightDirPointSize = QVector4D(
        -0.45f,
        0.70f,
        0.70f,
        pointDiameter * float(devicePixelRatioF()));
    drawRhiBuffer(cb, &_pointBuffer, &_colorPointPipeline, uniforms);

    uniforms.lightDirPointSize.setW(_meshHasFaces ? 1.0f : 1.5f);
    if (_modelColorMode == ModelColorMode::Wireframe && _meshHasFaces)
    {
        drawRhiBuffer(cb,
                      &_modelWireframeBuffer,
                      &_colorLinePipeline,
                      uniforms);
    }
    else if (_modelColorMode == ModelColorMode::Texture && _meshHasTexture)
    {
        drawTexturedMesh(cb, uniforms);
    }
    else
    {
        drawRhiBuffer(cb,
                      &_meshBuffer,
                      _meshHasFaces ? &_meshTrianglePipeline : &_meshPointPipeline,
                      uniforms);
    }

    uniforms.lightDirPointSize.setW(_modelPointSize);
    drawRhiBuffer(cb, &_modelPointBuffer, &_modelPointPipeline, uniforms);

    uniforms.lightDirPointSize.setW(1.0f);
    drawRhiBuffer(cb, &_lineBuffer, &_colorLinePipeline, uniforms);
}

void CameraSceneWidget::render(QRhiCommandBuffer *cb)
{
    if (!_rhiReady || !rhi() || !renderTarget())
    {
        if (_renderError.isEmpty())
        {
            _renderError = QStringLiteral("Vulkan 渲染初始化失败，请检查显卡驱动和 Qt Vulkan 支持。");
        }
        requestOverlayUpdate();
        return;
    }

    if (_gpuDirty)
    {
        uploadGpuData();
    }

    if (!ensurePipeline(&_colorPointPipeline,
                        int(QRhiGraphicsPipeline::Points),
                        6 * int(sizeof(float)),
                        false) ||
        !ensurePipeline(&_colorLinePipeline,
                        int(QRhiGraphicsPipeline::Lines),
                        6 * int(sizeof(float)),
                        false) ||
        !ensurePipeline(&_modelPointPipeline,
                        int(QRhiGraphicsPipeline::Points),
                        6 * int(sizeof(float)),
                        false) ||
        !ensurePipeline(&_meshTrianglePipeline,
                        int(QRhiGraphicsPipeline::Triangles),
                        _meshBuffer.strideBytes > 0
                            ? _meshBuffer.strideBytes
                            : 9 * int(sizeof(float)),
                        true) ||
        !ensurePipeline(&_meshPointPipeline,
            int(QRhiGraphicsPipeline::Points),
            _meshBuffer.strideBytes > 0
                ? _meshBuffer.strideBytes
                : 9 * int(sizeof(float)),
            true))
    {
        requestOverlayUpdate();
        return;
    }
    QRhiResourceUpdateBatch *updates = rhi()->nextResourceUpdateBatch();
    if (!ensureRhiBuffer(&_pointBuffer, updates) ||
        !ensureRhiBuffer(&_meshBuffer, updates) ||
        !ensureRhiBuffer(&_modelWireframeBuffer, updates) ||
        !ensureRhiBuffer(&_modelPointBuffer, updates) ||
        !ensureRhiBuffer(&_lineBuffer, updates))
    {
        requestOverlayUpdate();
        return;
    }
    if (!ensureTexturedMeshPipeline(updates)
        || !ensureCameraThumbnailPipeline(updates)
        || !ensureImagePipeline(updates))
    {
        requestOverlayUpdate();
        return;
    }
    _pipelinesDirty = false;
    _renderError.clear();

    const SceneMatrices matrices = sceneMatrices();
    QMatrix4x4 shift;
    shift.setToIdentity();
    shift.translate(float(2.0 * _sceneOffsetPx.x() / qMax(1, width())),
                    float(-2.0 * _sceneOffsetPx.y() / qMax(1, height())),
                    0.0f);
    const QMatrix4x4 mv = matrices.modelView;
    const QMatrix4x4 mvp = rhi()->clipSpaceCorrMatrix() * shift * matrices.projection * mv;

    cb->beginPass(renderTarget(),
                  QColor::fromRgbF(1.0f, 1.0f, 1.0f, 1.0f),
                  QRhiDepthStencilClearValue(1.0f, 0),
                  updates);
    const QSize pixelSize = renderTarget()->pixelSize();
    cb->setViewport(QRhiViewport(0.0f, 0.0f, float(pixelSize.width()), float(pixelSize.height())));

    SceneUniforms uniforms;
    uniforms.mvp = mvp;
    uniforms.modelView = mv;
    uniforms.normalMatrix.setToIdentity();
    const QMatrix3x3 normal3x3 = mv.normalMatrix();
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            uniforms.normalMatrix(row, col) = normal3x3(row, col);
        }
    }
    if (_cameraImageDisplayLayer == CameraImageDisplayLayer::Background)
    {
        drawActiveCameraImage(cb, mvp);
    }
    drawSceneGeometry(cb, uniforms);
    // 相机平面最后参与同一个深度缓冲。它只覆盖实际位于其后的点，
    // 并在共面时优先保留照片，避免连接点从照片表面穿透出来。
    drawCameraThumbnails(cb, mvp);
    if (_cameraImageDisplayLayer == CameraImageDisplayLayer::Foreground)
    {
        drawActiveCameraImage(cb, mvp);
    }

    cb->endPass();

    requestOverlayUpdate();
}

void CameraSceneWidget::updateCameraOverlay()
{
    update();
    requestOverlayUpdate();
}

void CameraSceneWidget::requestOverlayUpdate()
{
    if (!_overlayWidget)
    {
        return;
    }
    _overlayWidget->setGeometry(rect());
    _overlayWidget->raise();
    _overlayWidget->update();
}

void CameraSceneWidget::drawPointCloudOverlay(QPainter &painter) const
{
    if (_cloud.size() == 0 || _cloud.hasFaces())
    {
        return;
    }

    const SceneMatrices matrices = sceneMatrices();
    const QMatrix4x4 clipMatrix = matrices.projection * matrices.modelView;
    QVector<QVector<QVector3D>> cameraOccluders;
    if (_isTiePointCloud && _showCameras)
    {
        cameraOccluders.reserve(_poses.size());
        for (const CameraPose &pose : _poses)
        {
            const float halfExtent = cameraImagePlaneHalfExtent(
                pose, matrices.modelView);
            const xjw::gui::camera_scene::CameraImagePlaneAxes axes =
                xjw::gui::camera_scene::cameraImagePlaneAxes(
                    pose.rotation, pose.uAxisSign, pose.vAxisSign);
            const QVector<QVector3D> corners =
                xjw::gui::camera_scene::cameraImagePlaneCorners(
                    pose.center,
                    axes.right,
                    axes.up,
                    halfExtent,
                    halfExtent * 0.68f);

            QVector<QVector3D> quadNdc;
            quadNdc.reserve(corners.size());
            for (const QVector3D &corner : corners)
            {
                const QVector4D clip = clipMatrix * QVector4D(corner, 1.0f);
                if (clip.w() <= 1.0e-6f)
                {
                    quadNdc.clear();
                    break;
                }
                quadNdc.push_back(QVector3D(
                    clip.x() / clip.w(),
                    clip.y() / clip.w(),
                    clip.z() / clip.w()));
            }
            if (quadNdc.size() == 4)
            {
                cameraOccluders.push_back(quadNdc);
            }
        }
    }

    constexpr std::size_t maximumOverlayPointCount = 150'000;
    const std::size_t pointStride = qMax<std::size_t>(
        1,
        (_cloud.size() + maximumOverlayPointCount - 1) / maximumOverlayPointCount);
    const bool hasColors = _cloud.hasColors();
    const bool hasImageCounts =
        _tiePointImageCounts.size() == static_cast<qsizetype>(_cloud.size());

    QHash<QRgb, QVector<QPointF>> pointsByColor;
    pointsByColor.reserve(512);
    for (std::size_t index = 0; index < _cloud.size(); index += pointStride)
    {
        const plamatrix::Index cloudIndex = static_cast<plamatrix::Index>(index);
        const QVector3D point(
            _cloud.points()(cloudIndex, 0),
            _cloud.points()(cloudIndex, 1),
            _cloud.points()(cloudIndex, 2));
        const QVector4D clip = clipMatrix * QVector4D(point, 1.0f);
        if (clip.w() <= 1.0e-6f)
        {
            continue;
        }

        const QVector3D ndc(
            clip.x() / clip.w(),
            clip.y() / clip.w(),
            clip.z() / clip.w());
        if (ndc.x() < -1.02f || ndc.x() > 1.02f
            || ndc.y() < -1.02f || ndc.y() > 1.02f
            || ndc.z() < -1.02f || ndc.z() > 1.02f)
        {
            continue;
        }

        bool occludedByCamera = false;
        for (const QVector<QVector3D> &quadNdc : cameraOccluders)
        {
            if (xjw::gui::camera_scene::pointIsBehindProjectedQuad(ndc, quadNdc))
            {
                occludedByCamera = true;
                break;
            }
        }
        if (occludedByCamera)
        {
            continue;
        }

        QColor color(115, 115, 128);
        if (_isTiePointCloud && _tiePointColorMode == TiePointColorMode::Elevation)
        {
            color = xjw::gui::tie_points::elevationColor(
                point.z(), _tiePointElevationRange);
        }
        else if (_isTiePointCloud
                 && _tiePointColorMode == TiePointColorMode::ImageCount
                 && hasImageCounts)
        {
            color = xjw::gui::tie_points::imageCountColor(
                _tiePointImageCounts.at(static_cast<qsizetype>(index)),
                _tiePointImageCountRange);
        }
        else if (hasColors)
        {
            color = QColor::fromRgbF(
                qBound(0.0f,
                       _cloud.colors()->getValue(cloudIndex, 0) * _pointColorScale,
                       1.0f),
                qBound(0.0f,
                       _cloud.colors()->getValue(cloudIndex, 1) * _pointColorScale,
                       1.0f),
                qBound(0.0f,
                       _cloud.colors()->getValue(cloudIndex, 2) * _pointColorScale,
                       1.0f));
        }

        const auto quantizeChannel = [](int channel)
        {
            return qMin(255, ((channel >> 5) << 5) + 16);
        };
        const QRgb colorKey = qRgb(
            quantizeChannel(color.red()),
            quantizeChannel(color.green()),
            quantizeChannel(color.blue()));
        pointsByColor[colorKey].append(QPointF(
            (ndc.x() * 0.5f + 0.5f) * width() + _sceneOffsetPx.x(),
            (1.0f - (ndc.y() * 0.5f + 0.5f)) * height() + _sceneOffsetPx.y()));
    }

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, false);
    const qreal pointSize = _isTiePointCloud
        ? qMax<qreal>(2.2, xjw::gui::tie_points::pointSizeForMode(_tiePointColorMode))
        : qMax<qreal>(2.2, _modelPointSize);
    for (auto iterator = pointsByColor.cbegin(); iterator != pointsByColor.cend(); ++iterator)
    {
        painter.setPen(QPen(
            QColor::fromRgb(iterator.key()),
            pointSize,
            Qt::SolidLine,
            Qt::RoundCap));
        painter.drawPoints(
            iterator.value().constData(),
            static_cast<int>(iterator.value().size()));
    }
    painter.restore();
}

void CameraSceneWidget::paintOverlay(QPainter &painter)
{
    if (!painter.isActive())
    {
        return;
    }

    if (!_renderError.isEmpty())
    {
        painter.fillRect(rect(), Qt::white);
        painter.setPen(QColor(180, 42, 42));
        painter.drawText(rect().adjusted(12, 12, -12, -12),
                         Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap,
                         _renderError);
        return;
    }

    // 前景照片是最终遮挡层。Vulkan 场景先被不透明照片覆盖，随后绘制的
    // QWidget 叠加内容也必须使用同一照片投影区域裁剪，不能再次穿透照片。
    painter.save();
    const QPainterPath foregroundImageOcclusion = foregroundCameraImageOcclusionPath();
    if (!foregroundImageOcclusion.isEmpty())
    {
        QPainterPath visibleScene;
        visibleScene.addRect(QRectF(rect()));
        visibleScene = visibleScene.subtracted(foregroundImageOcclusion);
        painter.setClipPath(visibleScene, Qt::IntersectClip);
    }

    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // ── 操控球（Gizmo）：仅当 _showGizmo 为 true 时绘制 ───────────────
    if (_showGizmo) {
    QRadialGradient grad(center2d - QPointF(radiusPx * 0.18, radiusPx * 0.18), radiusPx * 1.25);
    grad.setColorAt(0.0, QColor(245, 245, 248, 40));
    grad.setColorAt(1.0, QColor(175, 178, 186, 28));
    painter.setPen(QPen(QColor(210, 210, 216, 44), 1.0));
    painter.setBrush(grad);
    painter.drawEllipse(center2d, radiusPx, radiusPx);

    auto axisPen = [&](HoverAxis axis, const QColor &base) {
        const bool hl = (_hoverAxis == axis) || (_dragAxis == axis && _leftDragging);
        QColor cc = base;
        if (hl) cc = cc.lighter(150);
        return QPen(cc, hl ? 4.0 : 2.0);
    };
    auto drawGreatCircle = [&](HoverAxis axis, const QColor &color) {
        painter.setPen(axisPen(axis, color));
        QPointF prev;
        QPointF first;
        bool hasPrev = false;
        bool prevVisible = false;
        bool firstVisible = false;
        for (int i = 0; i <= 128; ++i) {
            const qreal t = (2.0 * M_PI * i) / 128.0;
            QVector3D pLocal;
            if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
            else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
            else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
            QVector3D pView = applyViewRotation(pLocal);
            const bool currVisible = (pView.z() > 0.0f);
            QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
            if (!hasPrev) {
                first = curr;
                firstVisible = currVisible;
            } else {
                if (prevVisible && currVisible) painter.drawLine(prev, curr);
            }
            prev = curr;
            prevVisible = currVisible;
            hasPrev = true;
        }
        if (hasPrev && firstVisible && prevVisible) painter.drawLine(prev, first);
    };
    drawGreatCircle(HoverAxis::X, QColor(255, 110, 110, 52));
    drawGreatCircle(HoverAxis::Y, QColor(110, 255, 150, 52));
    drawGreatCircle(HoverAxis::Z, QColor(110, 170, 255, 52));
    } // end if (_showGizmo)

    // 无面点云使用稳定的 QPainter 点样式；连接点还会按三维照片平面的
    // 投影深度剔除被遮挡点，避免点云穿透相机缩略图。
    drawPointCloudOverlay(painter);

    if (_showCameras)
    {
        const int labelBudget = maxVisibleCameraLabels();
        const int cameraCount = static_cast<int>(_poses.size());
        const bool drawAllCameraLabels = _poses.size() <= maxVisibleCameraLabels();
        const int cameraLabelStride = drawAllCameraLabels
            ? 1
            : qMax(1, static_cast<int>(std::ceil(double(cameraCount) / double(qMax(1, labelBudget)))));

        if (_poses.isEmpty())
        {
            painter.setPen(QColor(120, 120, 120));
            painter.drawText(rect(), Qt::AlignCenter, tr("暂无相机参数，显示默认模型球"));
        }

        auto drawWorldSegment = [this, &painter](const QVector3D &start,
                                                 const QVector3D &end,
                                                 const QColor &color,
                                                 qreal width)
        {
            bool startOk = false;
            bool endOk = false;
            const QPointF startScreen = projectToScreen(start, &startOk);
            const QPointF endScreen = projectToScreen(end, &endOk);
            if (startOk && endOk)
            {
                painter.setPen(QPen(color, width));
                painter.drawLine(startScreen, endScreen);
            }
        };
        const SceneMatrices cameraMatrices = sceneMatrices();

        for (const CameraPose &pose : _poses)
        {
            const float planeHalfExtent = cameraImagePlaneHalfExtent(
                pose, cameraMatrices.modelView);

            if (_showCameraLocalAxes)
            {
                const float localAxisLength = planeHalfExtent * 0.75f;
                const xjw::gui::camera_scene::CameraLocalAxes localAxes =
                    xjw::gui::camera_scene::cameraLocalAxes(
                        pose.rotation, pose.depthAxisFlipped);
                drawWorldSegment(pose.center,
                                 pose.center + localAxes.x * localAxisLength,
                                 QColor(220, 55, 55),
                                 1.65);
                drawWorldSegment(pose.center,
                                 pose.center + localAxes.y * localAxisLength,
                                 QColor(35, 165, 70),
                                 1.65);
                drawWorldSegment(pose.center,
                                 pose.center + localAxes.z * localAxisLength,
                                 QColor(45, 105, 225),
                                 1.65);
            }
        }

        QVector<int> camera_label_order;
        camera_label_order.reserve(_poses.size());
        for (qsizetype poseIndex = 0; poseIndex < _poses.size(); ++poseIndex)
        {
            camera_label_order.push_back(static_cast<int>(poseIndex));
        }

        // 相机方位线属于二维标注层，无法直接使用 Vulkan 深度缓冲。
        // 将所有已投影照片平面从其绘制区域中扣除，确保任何相机的
        // 方位线都不会透过自身或其他照片平面。
        QPainterPath cameraPlaneOcclusionPath;
        cameraPlaneOcclusionPath.setFillRule(Qt::WindingFill);
        for (const CameraPose &pose : _poses)
        {
            const float halfExtent = cameraImagePlaneHalfExtent(
                pose, cameraMatrices.modelView);
            const xjw::gui::camera_scene::CameraImagePlaneAxes axes =
                xjw::gui::camera_scene::cameraImagePlaneAxes(
                    pose.rotation, pose.uAxisSign, pose.vAxisSign);
            const QVector<QVector3D> corners =
                xjw::gui::camera_scene::cameraImagePlaneCorners(
                    pose.center,
                    axes.right,
                    axes.up,
                    halfExtent,
                    halfExtent * 0.68f);

            QPolygonF projectedPlane;
            bool planeVisible = corners.size() == 4;
            for (const QVector3D &corner : corners)
            {
                bool cornerOk = false;
                const QPointF projectedCorner = projectToScreen(corner, &cornerOk);
                if (!cornerOk)
                {
                    planeVisible = false;
                    break;
                }
                projectedPlane.push_back(projectedCorner);
            }
            if (planeVisible)
            {
                QPainterPath planePath;
                planePath.addPolygon(projectedPlane);
                planePath.closeSubpath();
                cameraPlaneOcclusionPath = cameraPlaneOcclusionPath.united(planePath);
            }
        }
        QPainterPath cameraLeaderClip;
        cameraLeaderClip.addRect(QRectF(rect()));
        cameraLeaderClip = cameraLeaderClip.subtracted(cameraPlaneOcclusionPath);

        for (const int poseIndex : camera_label_order)
        {
            const CameraPose &pose = _poses.at(poseIndex);
            const bool highlighted = isCameraHighlighted(pose);
            const float thumbnailHalfExtent = cameraImagePlaneHalfExtent(
                pose, cameraMatrices.modelView);
            const QLineF directionLeader = cameraDirectionLeaderLine(
                pose, thumbnailHalfExtent);
            if (!directionLeader.isNull())
            {
                painter.save();
                painter.setClipPath(cameraLeaderClip, Qt::IntersectClip);
                painter.setPen(QPen(QColor(25, 25, 25, cameraCount > 120 ? 155 : 215),
                                    highlighted ? 1.25 : 1.0,
                                    Qt::SolidLine,
                                    Qt::RoundCap));
                painter.drawLine(directionLeader);
                painter.restore();
            }
            const bool drawCameraLabel = highlighted
                || drawAllCameraLabels
                || (poseIndex == 0)
                || (poseIndex == _poses.size() - 1)
                || (poseIndex % cameraLabelStride == 0);
            if (drawCameraLabel)
            {
                bool centerOk = false;
                const QPointF center = projectToScreen(pose.center, &centerOk);
                if (!centerOk)
                {
                    continue;
                }
                const QString labelSource = pose.imagePath.isEmpty() ? pose.name : pose.imagePath;
                const QString label = QFileInfo(labelSource).fileName().isEmpty()
                    ? pose.name
                    : QFileInfo(labelSource).fileName();
                const QPointF labelAnchor = directionLeader.isNull()
                    ? center
                    : directionLeader.p2();
                const bool placeLabelLeft = !directionLeader.isNull()
                    && directionLeader.dx() < 0.0;
                const qreal labelWidth = painter.fontMetrics().horizontalAdvance(label);
                const QPointF textOffset = placeLabelLeft
                    ? QPointF(-labelWidth - 5.0, -2.0)
                    : QPointF(5.0, -2.0);
                painter.setPen(highlighted
                    ? QColor(210, 45, 65, 230)
                    : (drawAllCameraLabels ? QColor(60, 60, 60) : QColor(45, 45, 45, 170)));
                painter.drawText(labelAnchor + textOffset, label);
            }
        }

    }

    drawFloorPivotCross(painter);
    painter.restore();

    drawTiePointLegend(painter);
    drawModelLegend(painter);

    const QPoint origin(width() - 64, height() - 64);
    const QVector3D ex = applyViewRotation(QVector3D(1, 0, 0)).normalized();
    const QVector3D ey = applyViewRotation(QVector3D(0, 1, 0)).normalized();
    const QVector3D ez = applyViewRotation(QVector3D(0, 0, 1)).normalized();
    auto drawMiniAxis = [&](const QVector3D &dir, const QColor &color, const QString &label) {
        const QPoint end(origin.x() + int(dir.x() * 28.0f), origin.y() - int(dir.y() * 28.0f));
        painter.setPen(QPen(color, 2));
        painter.drawLine(origin, end);
        painter.setPen(color);
        painter.drawText(end + QPoint(4, -2), label);
    };
    painter.setPen(QPen(QColor(80, 80, 80), 1));
    painter.setBrush(QColor(80, 80, 80));
    painter.drawEllipse(QPointF(origin), 2.5, 2.5);
    drawMiniAxis(ex, QColor(210, 50, 50), QStringLiteral("X"));
    drawMiniAxis(ey, QColor(30, 160, 60), QStringLiteral("Y"));
    drawMiniAxis(ez, QColor(40, 100, 220), QStringLiteral("Z"));
    painter.setPen(QColor(100, 100, 110));
    const QVector3D euler = eulerAnglesDeg();
    painter.drawText(origin + QPoint(-84, 26),
                     QStringLiteral("Yaw %1°  Pitch %2°  Roll %3°")
                         .arg(QString::number(euler.y(), 'f', 1))
                         .arg(QString::number(euler.x(), 'f', 1))
                         .arg(QString::number(euler.z(), 'f', 1)));

    if (_manualPruneMode)
    {
        painter.setPen(QPen(QColor(255, 90, 90, 220), 1.5, Qt::DashLine));
        painter.setBrush(QColor(255, 90, 90, 40));
        if (!_manualSelectRect.isNull())
        {
            painter.drawRect(_manualSelectRect.normalized());
        }

        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 230, 90, 170));
        const int highlightCap = 12000;
        const int drawCount = std::min<int>(static_cast<int>(_manualPreviewIndices.size()), highlightCap);
        for (int i = 0; i < drawCount; ++i)
        {
            const std::size_t pointIndex = _manualPreviewIndices[static_cast<std::size_t>(i)];
            if (pointIndex >= _cloud.size())
            {
                continue;
            }
            bool ok = false;
            const QPointF screenPoint = projectToScreen(QVector3D(
                _cloud.points()(static_cast<plamatrix::Index>(pointIndex), 0),
                _cloud.points()(static_cast<plamatrix::Index>(pointIndex), 1),
                _cloud.points()(static_cast<plamatrix::Index>(pointIndex), 2)), &ok);
            if (ok)
            {
                painter.drawEllipse(screenPoint, 1.5, 1.5);
            }
        }

        painter.setPen(QColor(235, 80, 80));
        painter.drawText(QPointF(14.0, 24.0),
                         tr("手动剔除模式：右键框选高亮，前进侧键删除，Ctrl+Z 撤销（已选 %1）")
                             .arg(static_cast<int>(_manualPreviewIndices.size())));
    }

    drawPlyLoadProgressOverlay(painter);
}

void CameraSceneWidget::drawTiePointLegend(QPainter &painter) const
{
    if (!_isTiePointCloud || _tiePointColorMode == TiePointColorMode::Color ||
        _cloud.size() == 0)
    {
        return;
    }

    const bool elevationMode = _tiePointColorMode == TiePointColorMode::Elevation;
    const bool imageCountReady =
        _tiePointImageCounts.size() == static_cast<qsizetype>(_cloud.size());
    if (!elevationMode && !imageCountReady)
    {
        const QString status = _tiePointMetadataLoading
            ? tr("影像数：正在读取观测数据...")
            : tr("影像数：%1").arg(
                  _tiePointMetadataError.isEmpty() ? tr("无观测数据")
                                                   : _tiePointMetadataError);
        const QRectF statusRect(24.0, height() - 58.0, 310.0, 30.0);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 255, 255, 220));
        painter.drawRoundedRect(statusRect, 4.0, 4.0);
        painter.setPen(QColor(85, 85, 90));
        painter.drawText(statusRect.adjusted(10.0, 0.0, -8.0, 0.0),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         status);
        return;
    }

    const xjw::gui::tie_points::ScalarRange range =
        elevationMode ? _tiePointElevationRange : _tiePointImageCountRange;
    if (!range.isValid())
    {
        return;
    }

    const qreal legendHeight = qBound<qreal>(112.0, height() * 0.22, 188.0);
    const QRectF panel(22.0, height() - legendHeight - 70.0, 174.0, legendHeight + 48.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.top() + 28.0,
                     20.0,
                     legendHeight);

    painter.setPen(QPen(QColor(205, 205, 210, 190), 1.0));
    painter.setBrush(QColor(255, 255, 255, 220));
    painter.drawRoundedRect(panel, 5.0, 5.0);

    QLinearGradient gradient(bar.topLeft(), bar.bottomLeft());
    constexpr int colorStopCount = 5;
    for (int stopIndex = 0; stopIndex < colorStopCount; ++stopIndex)
    {
        const double position =
            static_cast<double>(stopIndex) / static_cast<double>(colorStopCount - 1);
        const double rampValue = elevationMode ? 1.0 - position : position;
        gradient.setColorAt(position,
                            xjw::gui::tie_points::scalarRampColor(rampValue));
    }
    painter.fillRect(bar, gradient);
    painter.setPen(QColor(100, 100, 105));
    painter.drawRect(bar);

    painter.setPen(QColor(50, 50, 55));
    painter.drawText(QRectF(panel.left() + 12.0,
                            panel.top() + 4.0,
                            panel.width() - 24.0,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     elevationMode ? tr("连接点 — 高程 (Z)")
                                   : tr("连接点 — 影像数"));

    auto formatValue = [elevationMode](double value)
    {
        if (!elevationMode)
        {
            return QString::number(qRound(value)) + QStringLiteral(" 张");
        }
        return QString::number(value, 'g', 7);
    };
    const double middle = (range.minimum + range.maximum) * 0.5;
    const qreal labelLeft = bar.right() + 10.0;
    const qreal labelWidth = panel.right() - labelLeft - 6.0;
    painter.drawText(QRectF(labelLeft, bar.top() - 9.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.maximum));
    painter.drawText(QRectF(labelLeft,
                            bar.center().y() - 10.0,
                            labelWidth,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(middle));
    painter.drawText(QRectF(labelLeft, bar.bottom() - 11.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.minimum));
}

void CameraSceneWidget::drawModelLegend(QPainter &painter) const
{
    const bool elevationMode = _modelColorMode == ModelColorMode::Elevation;
    const bool confidenceMode = _modelColorMode == ModelColorMode::Confidence;
    if (_isTiePointCloud || !_cloud.hasFaces() || (!elevationMode && !confidenceMode))
    {
        return;
    }

    const xjw::gui::tie_points::ScalarRange range =
        elevationMode
        ? _modelElevationRange
        : xjw::gui::tie_points::ScalarRange{1.0, 100.0};
    if (!range.isValid())
    {
        return;
    }

    const qreal legendHeight = qBound<qreal>(112.0, height() * 0.22, 188.0);
    const QRectF panel(22.0, height() - legendHeight - 70.0, 174.0, legendHeight + 48.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.top() + 28.0,
                     20.0,
                     legendHeight);

    painter.setPen(QPen(QColor(205, 205, 210, 190), 1.0));
    painter.setBrush(QColor(255, 255, 255, 220));
    painter.drawRoundedRect(panel, 5.0, 5.0);

    QLinearGradient gradient(bar.topLeft(), bar.bottomLeft());
    constexpr int colorStopCount = 5;
    for (int stopIndex = 0; stopIndex < colorStopCount; ++stopIndex)
    {
        const double position =
            static_cast<double>(stopIndex) / static_cast<double>(colorStopCount - 1);
        const QColor color = elevationMode
            ? xjw::gui::tie_points::scalarRampColor(1.0 - position)
            : xjw::gui::model_views::confidenceColor(
                  qRound(100.0 - position * 99.0));
        gradient.setColorAt(position, color);
    }
    painter.fillRect(bar, gradient);
    painter.setPen(QColor(100, 100, 105));
    painter.drawRect(bar);

    painter.setPen(QColor(50, 50, 55));
    painter.drawText(QRectF(panel.left() + 12.0,
                            panel.top() + 4.0,
                            panel.width() - 24.0,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     elevationMode ? tr("模型 — 高程 (Z)")
                                   : tr("模型 — 可信度"));

    const auto formatValue = [elevationMode](double value)
    {
        return elevationMode
            ? QString::number(value, 'g', 7)
            : QString::number(qRound(value));
    };
    const double middle = (range.minimum + range.maximum) * 0.5;
    const qreal labelLeft = bar.right() + 10.0;
    const qreal labelWidth = panel.right() - labelLeft - 6.0;
    painter.drawText(QRectF(labelLeft, bar.top() - 9.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.maximum));
    painter.drawText(QRectF(labelLeft,
                            bar.center().y() - 10.0,
                            labelWidth,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(middle));
    painter.drawText(QRectF(labelLeft, bar.bottom() - 11.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.minimum));
}

void CameraSceneWidget::drawPlyLoadProgressOverlay(QPainter &painter)
{
    if (!_loading || _plyLoadProgressPercent < 0)
    {
        return;
    }

    const int panelWidth = qMin(width() - 48, 520);
    if (panelWidth <= 160 || height() <= 100)
    {
        return;
    }

    const QRectF panel(24.0, height() - 72.0, panelWidth, 48.0);
    const QRectF bar(panel.left() + 16.0, panel.bottom() - 16.0, panel.width() - 32.0, 6.0);
    const qreal fillWidth = bar.width() * qBound(0, _plyLoadProgressPercent, 100) / 100.0;

    painter.save();
    painter.setPen(QPen(QColor(70, 82, 96, 160), 1.0));
    painter.setBrush(QColor(250, 252, 255, 235));
    painter.drawRoundedRect(panel, 6.0, 6.0);

    painter.setPen(QColor(34, 48, 68));
    const QString title = _plyLoadProgressText.isEmpty()
        ? tr("正在加载密集点云...")
        : _plyLoadProgressText;
    painter.drawText(QRectF(panel.left() + 16.0,
                            panel.top() + 8.0,
                            panel.width() - 96.0,
                            20.0),
                     Qt::AlignVCenter | Qt::AlignLeft,
                     title);
    painter.drawText(QRectF(panel.right() - 70.0,
                            panel.top() + 8.0,
                            54.0,
                            20.0),
                     Qt::AlignVCenter | Qt::AlignRight,
                     QStringLiteral("%1%").arg(qBound(0, _plyLoadProgressPercent, 100)));

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(218, 226, 238));
    painter.drawRoundedRect(bar, 3.0, 3.0);
    painter.setBrush(QColor(36, 115, 218));
    painter.drawRoundedRect(QRectF(bar.left(), bar.top(), fillWidth, bar.height()), 3.0, 3.0);
    painter.restore();
}

bool CameraSceneWidget::setManualPruneModeEnabled(bool enabled, QString *errorMessage)
{
    if (enabled)
    {
        if ((_cloud.size() == 0))
        {
            if (errorMessage)
            {
                *errorMessage = tr("当前未加载点云数据。");
            }
            return false;
        }
        if (_cloud.hasFaces())
        {
            if (errorMessage)
            {
                *errorMessage = tr("当前为网格模型，手动剔除仅支持点云。");
            }
            return false;
        }
    }

    _manualPruneMode = enabled;
    _manualSelecting = false;
    _manualSelectRect = QRect();
    _manualPreviewIndices.clear();
    _manualPreviewValid = false;
    if (!enabled)
    {
        _manualUndoStack.clear();
    }
    emit manualPruneModeChanged(_manualPruneMode);
    updateCursor();
    update();
    return true;
}

void CameraSceneWidget::pushManualUndoSnapshot(RenderCloud snapshot)
{
    if (!_manualPruneMode)
    {
        return;
    }
    _manualUndoStack.push_back(std::move(snapshot));
    if (static_cast<int>(_manualUndoStack.size()) > _manualUndoLimit)
    {
        _manualUndoStack.erase(_manualUndoStack.begin());
    }
}

bool CameraSceneWidget::undoLastManualPrune(QString *errorMessage)
{
    if (!_manualPruneMode)
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前未处于手动剔除模式。");
        }
        return false;
    }
    if (_manualUndoStack.empty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("没有可撤销的删除操作。");
        }
        return false;
    }

    _cloud = std::move(_manualUndoStack.back());
    _manualUndoStack.pop_back();
    _cacheDirty = true;
    _gpuDirty = true;
    update();

    emit manualPruneUndone(static_cast<int>(_cloud.size()));
    queueCurrentPointCloudSave();
    return true;
}

int CameraSceneWidget::removePointsInScreenRect(const QRect &screenRect)
{
    const QRect rect = screenRect.normalized();
    if (rect.width() < 3 || rect.height() < 3 || (_cloud.size() == 0))
    {
        return 0;
    }

    std::vector<std::size_t> selectedIndices;
    if (_manualPreviewValid && rect == _manualSelectRect.normalized())
    {
        selectedIndices = _manualPreviewIndices;
    }
    else
    {
        collectPointIndicesInScreenRect(rect, &selectedIndices);
    }
    const std::size_t pointCount = _cloud.size();
    std::vector<bool> removeMask(pointCount, false);
    for (const std::size_t index : selectedIndices)
    {
        if (index < pointCount)
        {
            removeMask[index] = true;
        }
    }
    const int removedCount = static_cast<int>(selectedIndices.size());

    if (removedCount <= 0)
    {
        return 0;
    }

    // Build kept indices list
    const std::size_t keepCount = pointCount - static_cast<std::size_t>(removedCount);
    std::vector<plamatrix::Index> kept;
    kept.reserve(keepCount);
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        if (!removeMask[index])
            kept.push_back(static_cast<plamatrix::Index>(index));
    }

    // Build new point cloud from kept indices
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> newPts(keepCount, 3);
    for (std::size_t i = 0; i < keepCount; ++i)
    {
        plamatrix::Index src = kept[i];
        for (int d = 0; d < 3; ++d)
            newPts(static_cast<plamatrix::Index>(i), d) = _cloud.points()(src, d);
    }

    RenderCloud filtered(std::move(newPts));
    filtered.setMaterialLibraryFile(_cloud.materialLibraryFile());
    filtered.setTextureImageFile(_cloud.textureImageFile());

    // Copy colors if present
    if (_cloud.hasColors())
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> newCol(keepCount, 3);
        for (std::size_t i = 0; i < keepCount; ++i)
        {
            plamatrix::Index src = kept[i];
            for (int d = 0; d < 3; ++d)
                newCol(static_cast<plamatrix::Index>(i), d) = _cloud.colors()->getValue(src, d);
        }
        filtered.setColors(std::move(newCol));
    }

    // Copy normals if present
    if (_cloud.hasNormals())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> newNrm(keepCount, 3);
        for (std::size_t i = 0; i < keepCount; ++i)
        {
            plamatrix::Index src = kept[i];
            for (int d = 0; d < 3; ++d)
                newNrm(static_cast<plamatrix::Index>(i), d) = _cloud.normals()->getValue(src, d);
        }
        filtered.setNormals(std::move(newNrm));
    }

    // Copy texture coords if present
    if (_cloud.hasTextureCoords())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> newTex(keepCount, 2);
        for (std::size_t i = 0; i < keepCount; ++i)
        {
            plamatrix::Index src = kept[i];
            for (int d = 0; d < 2; ++d)
                newTex(static_cast<plamatrix::Index>(i), d) = _cloud.textureCoords()->getValue(src, d);
        }
        filtered.setTextureCoords(std::move(newTex));
    }

    _cloud = std::move(filtered);
    _manualPreviewIndices.clear();
    _manualPreviewValid = false;
    _cacheDirty = true;
    _gpuDirty = true;
    update();
    return removedCount;
}

void CameraSceneWidget::collectPointIndicesInScreenRect(const QRect &screenRect,
                                                        std::vector<std::size_t> *indices) const
{
    if (!indices)
    {
        return;
    }
    indices->clear();

    const QRect rect = screenRect.normalized();
    if (rect.width() < 3 || rect.height() < 3 || (_cloud.size() == 0))
    {
        return;
    }

    const std::size_t pointCount = _cloud.size();
    indices->reserve(pointCount / 8);
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        bool projected = false;
        const QPointF screenPoint = projectToScreen(QVector3D(
            _cloud.points()(static_cast<plamatrix::Index>(index), 0),
            _cloud.points()(static_cast<plamatrix::Index>(index), 1),
            _cloud.points()(static_cast<plamatrix::Index>(index), 2)), &projected);
        if (projected && rect.contains(screenPoint.toPoint()))
        {
            indices->push_back(index);
        }
    }
}

bool CameraSceneWidget::saveCurrentPointCloudToSource(QString *errorMessage)
{
    const PointCloudSaveResult result =
        savePointCloudSnapshot(_currentCloudPath, _cloud);
    if (!result.success && errorMessage)
    {
        *errorMessage = result.errorMessage;
    }
    return result.success;
}

void CameraSceneWidget::queueCurrentPointCloudSave()
{
    _manualSavePending = true;
    startCurrentPointCloudSave();
}

void CameraSceneWidget::startCurrentPointCloudSave()
{
    if (_manualSaveRunning || !_manualSavePending)
    {
        return;
    }
    if (_currentCloudPath.trimmed().isEmpty())
    {
        _manualSavePending = false;
        emit manualPruneSaveFailed(tr("当前点云来源未知，无法覆盖保存。"));
        return;
    }

    _manualSavePending = false;
    _manualSaveRunning = true;
    const QString path = _currentCloudPath;
    RenderCloud snapshot = cloneRenderCloud(_cloud);
    xjw::gui::tasks::runGuarded(
        this,
        [path, snapshot = std::move(snapshot)]() mutable
        {
            return savePointCloudSnapshot(path, snapshot);
        },
        [](CameraSceneWidget *self, PointCloudSaveResult result)
        {
            self->_manualSaveRunning = false;
            if (result.success)
            {
                emit self->manualPruneSaved(result.path, result.pointCount);
            }
            else
            {
                emit self->manualPruneSaveFailed(result.errorMessage);
            }
            self->startCurrentPointCloudSave();
        });
}

void CameraSceneWidget::mousePressEvent(QMouseEvent *event)
{
    setFocus();

    if (_manualPruneMode && event->button() == Qt::RightButton)
    {
        _manualSelecting = true;
        _manualSelectStart = event->pos();
        _manualSelectRect = QRect(_manualSelectStart, _manualSelectStart);
        _manualPreviewIndices.clear();
        _manualPreviewValid = false;
        updateCursor();
        event->accept();
        update();
        return;
    }

    if (_manualPruneMode && event->button() == Qt::ForwardButton)
    {
        const QRect selectionRect = _manualSelectRect.normalized();
        RenderCloud snapshot = cloneRenderCloud(_cloud);
        const int removedCount = removePointsInScreenRect(selectionRect);
        if (removedCount > 0)
        {
            pushManualUndoSnapshot(std::move(snapshot));
            emit manualPruneApplied(removedCount, static_cast<int>(_cloud.size()));
            queueCurrentPointCloudSave();
        }
        event->accept();
        update();
        return;
    }

    _lastMousePos = event->pos();
    if (event->button() == Qt::LeftButton) {
        _leftDragging = true;
        _dragAxis = _hoverAxis;
        // 无论单轴还是自由旋转，均记录按下时的旋转状态
        _viewRotAtPress = _viewRot;
        if (_dragAxis != HoverAxis::None) {
            _dragAxisDir = pickAxisTangent(event->pos(), _dragAxis);
            // 注意：Y 轴环的屏幕切线方向在参数化时与鼠标拖拽方向有符号差，
            // 在多数情况下需要翻转切线方向以使鼠标向右/上时视图按直觉旋转。
            // 仅对 Y 轴做翻转修正，避免对 X/Z 轴产生副作用。
            if (_dragAxis == HoverAxis::Y) _dragAxisDir = -_dragAxisDir;
        } else {
            // Arcball 自由旋转：记录按下那一刻球面坐标
            _arcballPressVector = arcballVector(event->pos());
        }
        updateCursor();
        event->accept();
        return;
    }
    if (event->button() == Qt::MiddleButton) {
        _middleDragging = true;
        updateCursor();
        event->accept();
        return;
    }
    QRhiWidget::mousePressEvent(event);
}

void CameraSceneWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (_manualPruneMode && _manualSelecting && (event->buttons() & Qt::RightButton))
    {
        _manualSelectRect = QRect(_manualSelectStart, event->pos()).normalized();
        _manualPreviewIndices.clear();
        _manualPreviewValid = false;
        event->accept();
        update();
        return;
    }

    const QPoint delta = event->pos() - _lastMousePos;

    if (_leftDragging && (event->buttons() & Qt::LeftButton)) {
        if (_dragAxis == HoverAxis::None) {
            // ── Arcball 自由旋转 ──────────────────────────────────────────────
            // 将当前鼠标投影到球面，计算从按下点到当前点的旋转，
            // 再䈛到按下时保存的初始视图旋转上——
            // 这样球面上最始点击处就会一直跟随鼠标移动。
            const QVector3D v2 = arcballVector(event->pos());
            const QVector3D axis = QVector3D::crossProduct(_arcballPressVector, v2);
            if (axis.lengthSquared() > 1e-10f) {
                const float dot = qBound(-1.0f,
                    QVector3D::dotProduct(_arcballPressVector, v2), 1.0f);
                const float angleDeg = qRadiansToDegrees(std::acos(dot));
                const QQuaternion delta_q =
                    QQuaternion::fromAxisAndAngle(axis.normalized(), angleDeg);
                // 应用到按下时的初始旋转（非增量式，避免浮点漂移）
                _viewRot = (delta_q * _viewRotAtPress).normalized();
            }
        } else {
            // ── 单轴环旋转 ─────────────────────────────────────────────────────
            // 目标：环面的法向方向（即 X/Y/Z 轴在当前世界空间中的指向）固定不动，
            //       环只在自身平面内"自旋"，看起来像环面始终保持水平/垂直。
            // 实现：将本地轴转换到世界空间 axisView，绕 axisView 前乘旋转。
            //       前乘（世界空间旋转）效果：环法向不变，环平面姿态不变，
            //       场景内容（相机等）绕该轴旋转。
            const QVector2D d(float(delta.x()), float(delta.y()));
            const float scalar = QVector2D::dotProduct(d, _dragAxisDir);
            const float ang = scalar * 0.35f;
            // 取该环的本地法向轴，转换为当前视图下的世界方向
            QVector3D localAxis;
            if (_dragAxis == HoverAxis::X)      localAxis = QVector3D(1.0f, 0.0f, 0.0f);
            else if (_dragAxis == HoverAxis::Y) localAxis = QVector3D(0.0f, 1.0f, 0.0f);
            else                                  localAxis = QVector3D(0.0f, 0.0f, 1.0f);
            const QVector3D axisWorld = applyViewRotation(localAxis).normalized();
            // 绕世界空间轴前乘：new_rot = qa_world * old_rot
            // 这样环的法向量方向(axisWorld)在此次旋转后保持恒定
            const QQuaternion qa = QQuaternion::fromAxisAndAngle(axisWorld, ang);
            _viewRot = (qa * _viewRot).normalized();
        }
        if (_showCameraImage && !_cameraImageLocked)
        {
            updateActiveCameraForView();
        }
        update();
    } else if (_middleDragging && (event->buttons() & Qt::MiddleButton)) {
        // 中键平移：1:1 映射鼠标像素，无论缩放倍率如何，拖拽同量始终移动同距离
        _sceneOffsetPx += QPointF(delta.x(), delta.y());
        clampSceneOffset();
        update();
    } else {
        const HoverAxis newHover = pickHoverAxis(event->pos());
        if (newHover != _hoverAxis) {
            _hoverAxis = newHover;
            updateCursor();
            update();
        }
    }

    _lastMousePos = event->pos();
    QRhiWidget::mouseMoveEvent(event);
}

void CameraSceneWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (_manualPruneMode && event->button() == Qt::RightButton)
    {
        _manualSelecting = false;
        _manualSelectRect = _manualSelectRect.normalized();
        collectPointIndicesInScreenRect(_manualSelectRect, &_manualPreviewIndices);
        _manualPreviewValid = true;
        updateCursor();
        update();
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        _leftDragging = false;
        _dragAxis = HoverAxis::None;
    }
    if (event->button() == Qt::MiddleButton) {
        _middleDragging = false;
    }
    updateCursor();
    QRhiWidget::mouseReleaseEvent(event);
}

void CameraSceneWidget::wheelEvent(QWheelEvent *event)
{
    if (_leftDragging || _middleDragging) {
        event->ignore();
        return;
    }

    const QPoint angle = event->angleDelta();
    if (!angle.isNull())
    {
        const double wheel_steps = static_cast<double>(angle.y()) / 120.0;
        const double factor = std::pow(1.10, wheel_steps);
        applyZoomFactor(factor);
    }
    event->accept();
}

void CameraSceneWidget::zoomIn()
{
    applyZoomFactor(1.10);
}

void CameraSceneWidget::zoomOut()
{
    applyZoomFactor(1.0 / 1.10);
}

void CameraSceneWidget::resetView()
{
    _viewRot = QQuaternion();
    _zoomScale = 1.0;
    _sceneOffsetPx = QPointF();
    _hoverAxis = HoverAxis::None;
    _dragAxis = HoverAxis::None;
    updateCameraOverlay();
}

void CameraSceneWidget::applyZoomFactor(double factor)
{
    if (!std::isfinite(factor) || factor <= 0.0)
    {
        return;
    }

    const double next_zoom_scale = _zoomScale * factor;
    if (!std::isfinite(next_zoom_scale) || next_zoom_scale <= 0.0)
    {
        return;
    }

    _zoomScale = next_zoom_scale;
    clampSceneOffset();
    update();
}

void CameraSceneWidget::keyPressEvent(QKeyEvent *event)
{
    if (_manualPruneMode && event->matches(QKeySequence::Undo))
    {
        QString errorMessage;
        undoLastManualPrune(&errorMessage);
        event->accept();
        return;
    }
    QRhiWidget::keyPressEvent(event);
}

CameraModel3DDialog::CameraModel3DDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , _projectManager(projectManager)
{
    Ui::CameraModel3DDialog form;
    form.setupUi(this);

    _scene = form.m_scene;
    _summaryLabel = form.m_summaryLabel;

    connect(form.reloadButton, &QPushButton::clicked, this, &CameraModel3DDialog::reloadFromProject);
    connect(form.closeButton, &QPushButton::clicked, this, &QDialog::accept);

    reloadFromProject();
}

QVector<CameraSceneWidget::CameraPose> CameraModel3DDialog::readCamerasFromMeta() const
{
    QVector<CameraSceneWidget::CameraPose> poses;
    if (!_projectManager)
    {
        return poses;
    }

    const QJsonArray images = xjw::common::project::projectImageEntries(_projectManager->currentMeta());
    for (const QJsonValue &imageValue : images)
    {
        const QJsonObject imageObject = imageValue.toObject();
        xjw::Camera camera;
        if (!xjw::common::project::imageCameraFromEntry(imageObject, &camera))
        {
            continue;
        }

        const std::array<double, 3> cameraCenter = camera.cameraCenter();
        const std::array<double, 9> cameraToWorldRotation = camera.cameraToWorldRotation();
        const xjw::Camera::Intrinsics intrinsics = camera.intrinsics();
        const QJsonObject camera_object = imageObject.value(QStringLiteral("camera")).toObject();

        CameraSceneWidget::CameraPose pose;
        pose.name = imageObject.value(QStringLiteral("name")).toString();
        pose.imagePath = imageObject.value(QStringLiteral("path")).toString();
        if (pose.imagePath.isEmpty())
        {
            pose.imagePath = imageObject.value(QStringLiteral("image_path")).toString();
        }
        if (pose.name.isEmpty())
        {
            pose.name = pose.imagePath;
        }
        pose.center = QVector3D(
            float(cameraCenter[0]),
            float(cameraCenter[1]),
            float(cameraCenter[2]));
        pose.focalX = static_cast<float>(intrinsics.focalX);
        pose.focalY = static_cast<float>(intrinsics.focalY);
        pose.principalX = static_cast<float>(intrinsics.principalX);
        pose.principalY = static_cast<float>(intrinsics.principalY);
        pose.imageWidth = camera_object.value(QStringLiteral("image_width")).toInt();
        pose.imageHeight = camera_object.value(QStringLiteral("image_height")).toInt();
        pose.uAxisSign = intrinsics.uAxisSign;
        pose.vAxisSign = intrinsics.vAxisSign;
        pose.depthAxisFlipped = camera.depthAxisFlipped();

        QMatrix3x3 rot;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                rot(row, col) = float(cameraToWorldRotation[row * 3 + col]);
            }
        }
        pose.rotation = rot;
        poses.push_back(pose);
    }

    return poses;
}

void CameraModel3DDialog::reloadFromProject()
{
    const QVector<CameraSceneWidget::CameraPose> poses = readCamerasFromMeta();
    _scene->setCameraPoses(poses);
    const QString labelHint = poses.size() > 40
        ? tr("，相机名称已抽样显示")
        : QString();
    _summaryLabel->setText(tr("相机数量: %1%2（左键旋转，滚轮缩放）").arg(poses.size()).arg(labelHint));
}
