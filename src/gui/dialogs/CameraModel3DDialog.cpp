// =============================================================================
// 文件: CameraModel3DDialog.cpp
// 功能: 相机三维模型可视化对话框实现
// 内容:
//   - CameraSceneWidget：OpenGL 4.x Core Profile 可编程管线渲染控件
//       · 点云 / PLY 模型 / 相机视锥体渲染（VAO/VBO + GLSL shader）
//       · Arcball 自由旋转 + 单轴环旋转（X/Y/Z Gizmo）
//       · 中键平移、滚轮缩放
//       · QPainter 覆盖层（Gizmo 环、坐标轴、相机视锥体、欧拉角）
//   - CameraModel3DDialog：对话框 UI + 从 ProjectManager 读取相机姿态
// =============================================================================
#include "CameraModel3DDialog.h"
#include "ui_CameraModel3DDialog.h"

#include "ProjectManager.h"
#include "ProjectSupportUtils.h"
#include "Logger.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QCursor>
#include <QPixmap>
#include <QVector2D>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLVersionFunctionsFactory>
#include <QSurfaceFormat>
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
#include <QKeyEvent>
#include <QKeySequence>
#include <QMetaObject>
#include <QPointer>
#include <QRegularExpression>
#include <QSizePolicy>
#include <QStringList>
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
    dst.setMaterialLibraryFile(src.materialLibraryFile());
    dst.setTextureImageFile(src.textureImageFile());
    return dst;
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

} // namespace

// 构造函数：设置基础可用尺寸，启用鼠标追踪（悬停检测需要），
// 设置默认视角为俯仰 -25°、偏航 35°（斜上方看向场景）
CameraSceneWidget::CameraSceneWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setSamples(4);
    setFormat(format);

    setMinimumSize(240, 160);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setMouseTracking(true); // 启用鼠标追踪，以便在无按键时检测悬停轴
    _viewRot = QQuaternion::fromEulerAngles(-25.0f, 35.0f, 0.0f); // 默认斜视角
    setFocusPolicy(Qt::StrongFocus);
    updateCursor();

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

// 设置要渲染的相机姿态列表。
// 调用后触发重绘，场景中每个姿态点将绘制光心标记和视锥体框线。
void CameraSceneWidget::setCameraPoses(const QVector<CameraPose> &poses)
{
    _poses = poses;
    _cacheDirty = true; // 相机位置变更，缓存失效
    update(); // 触发 paintGL 重绘
}

void CameraSceneWidget::setShowGizmo(bool show)
{
    if (_showGizmo != show)
    {
        _showGizmo = show;
        update(); // 触发重绘
    }
}

void CameraSceneWidget::setShowCameras(bool show)
{
    if (_showCameras != show)
    {
        _showCameras = show;
        update();
    }
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
    update();
}

void CameraSceneWidget::setHighlightedCameraName(const QString &imageName)
{
    if (_highlightedCameraName == imageName && _highlightedCameraPath.isEmpty())
    {
        return;
    }

    _highlightedCameraName = imageName;
    _highlightedCameraPath.clear();
    update();
}

void CameraSceneWidget::clearHighlightedCamera()
{
    if (_highlightedCameraPath.isEmpty() && _highlightedCameraName.isEmpty())
    {
        return;
    }

    _highlightedCameraPath.clear();
    _highlightedCameraName.clear();
    update();
}

// 取消未完成的加载（递增 generation 令旧回调自行失效）
void CameraSceneWidget::cancelPendingLoad()
{
    ++_loadGen;
    _loading = false;
    _plyLoadProgressPercent = -1;
    _plyLoadProgressText.clear();
}

// 标记缓存脏 + 重算（在加载完成后或场景数据变更后调用）
void CameraSceneWidget::invalidateCache() const
{
    bool has = false;
    QVector3D acc(0, 0, 0);
    int count = 0;
    QVector3D mn(0, 0, 0), mx(0, 0, 0);

    auto accum = [&](const QVector3D &p)
    {
        acc += p; ++count;
        if (!has) { mn = p; mx = p; has = true; return; }
        mn.setX(qMin(mn.x(), p.x())); mn.setY(qMin(mn.y(), p.y())); mn.setZ(qMin(mn.z(), p.z()));
        mx.setX(qMax(mx.x(), p.x())); mx.setY(qMax(mx.y(), p.y())); mx.setZ(qMax(mx.z(), p.z()));
    };

    for (const auto &p : _poses)             accum(p.center);
    for (size_t i = 0; i < _cloud.size(); ++i)
    {
        accum(QVector3D(_cloud.points()(static_cast<plamatrix::Index>(i), 0),
                        _cloud.points()(static_cast<plamatrix::Index>(i), 1),
                        _cloud.points()(static_cast<plamatrix::Index>(i), 2)));
    }

    if (count <= 0)
    {
        _cachedCenter  = QVector3D(0, 0, 0);
        _cachedRadius  = 10.0f;
        _cachedCameraFrustumBase = 0.6f;
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
            _cachedRadius = qMax(1.0f, dists[p95] * 1.15f);
        }
        else
        {
            _cachedRadius = 1.0f;
        }

        _cachedCameraFrustumBase = qMax(0.1f, _cachedRadius * 0.02f);
        if (_poses.size() > 1)
        {
            std::vector<float> nearestDistances;
            nearestDistances.reserve(static_cast<size_t>(_poses.size()));
            for (qsizetype i = 0; i < _poses.size(); ++i)
            {
                float nearest = std::numeric_limits<float>::max();
                for (qsizetype j = 0; j < _poses.size(); ++j)
                {
                    if (i == j)
                    {
                        continue;
                    }
                    nearest = qMin(nearest, (_poses[i].center - _poses[j].center).length());
                }
                if (std::isfinite(nearest))
                {
                    nearestDistances.push_back(nearest);
                }
            }
            if (!nearestDistances.empty())
            {
                const size_t medianIndex = nearestDistances.size() / 2;
                std::nth_element(nearestDistances.begin(),
                                 nearestDistances.begin() + static_cast<std::ptrdiff_t>(medianIndex),
                                 nearestDistances.end());
                const float spacingBase = nearestDistances[medianIndex] * 0.25f;
                _cachedCameraFrustumBase = qMax(0.1f, qMin(_cachedCameraFrustumBase, spacingBase));
            }
        }
    }
    _cacheDirty = false;
}

// 直接设置点云或网格（cloud.hasFaces() 决定渲染模式）
void CameraSceneWidget::setPointCloud(const RenderCloud &cloud)
{
    cancelPendingLoad();
    _cloud = cloneRenderCloud(cloud);
    _currentCloudPath.clear();
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
    cancelPendingLoad();
    _currentCloudPath = xyzPath;
    _cloud = RenderCloud();
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
                self->_gpuDirty = true;
                self->update();
            }
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([xyzPath]() -> std::shared_ptr<RenderCloud>
    {
        try
        {
            return plapoint::io::readXyz<float>(xyzPath.toStdString());
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
    cancelPendingLoad();
    _currentCloudPath = plyPath;
    _cloud = RenderCloud();
    _preferModelPointRender = true;
    _cacheDirty = true;
    _gpuDirty   = true;
    _loading = true;
    _plyLoadProgressPercent = 0;
    _plyLoadProgressText = QStringLiteral("正在加载密集点云...");
    update();
    LOG_INFO(QStringLiteral("[3D] 正在加载模型: %1").arg(plyPath));

    const int gen = _loadGen;
    emit plyLoadProgressChanged(gen, 0, _plyLoadProgressText);
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
            if (result) self->_cloud = std::move(*result);
            self->_preferModelPointRender = !self->_cloud.hasFaces();
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
            LOG_INFO(QStringLiteral("[3D] 模型加载完成，共 %1 顶点 / %2 面%3")
                     .arg(self->_cloud.size())
                     .arg(self->_cloud.hasFaces() ? static_cast<int>(self->_cloud.faces()->rows()) : 0)
                     .arg(self->_cloud.hasColors() ? QStringLiteral("（含RGB颜色）")
                                                    : QStringLiteral("（无颜色）")));
            self->invalidateCache();
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
            return plapoint::io::readPly<float>(plyPath.toStdString());
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
    cancelPendingLoad();
    _currentCloudPath = objPath;
    _cloud = RenderCloud();
    _preferModelPointRender = false;
    _cacheDirty = true;
    _gpuDirty = true;
    _loading = true;
    _plyLoadProgressPercent = 0;
    _plyLoadProgressText = QStringLiteral("正在加载 OBJ 模型...");
    update();
    LOG_INFO(QStringLiteral("[3D] 正在加载 OBJ 模型: %1").arg(objPath));

    const int gen = _loadGen;
    emit plyLoadProgressChanged(gen, 0, _plyLoadProgressText);
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
            }
            self->_preferModelPointRender = !self->_cloud.hasFaces();
            self->_loading = false;
            self->_plyLoadProgressPercent = -1;
            self->_plyLoadProgressText.clear();
            if (self->_cloud.size() == 0)
            {
                LOG_ERROR(QStringLiteral("[3D] OBJ 模型加载失败或为空"));
            }
            else
            {
                LOG_INFO(QStringLiteral("[3D] OBJ 模型加载完成，共 %1 顶点 / %2 面")
                             .arg(self->_cloud.size())
                             .arg(self->_cloud.hasFaces() ? static_cast<int>(self->_cloud.faces()->rows()) : 0));
            }
            self->invalidateCache();
            self->_gpuDirty = true;
            self->update();
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([objPath, self, gen]() -> std::shared_ptr<RenderCloud>
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
            reportProgress(5, QStringLiteral("正在解析 OBJ 模型..."));
            auto cloud = plapoint::io::readObj<float>(objPath.toStdString());
            if (!cloud || cloud->size() == 0)
            {
                reportProgress(100, QStringLiteral("OBJ 模型为空"));
                return nullptr;
            }
            reportProgress(96,
                           QStringLiteral("正在上传 OBJ 模型 (%1 顶点 / %2 面)...")
                               .arg(cloud->size())
                               .arg(cloud->hasFaces() ? static_cast<int>(cloud->faces()->rows()) : 0));
            return cloud;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR(QStringLiteral("[3D] OBJ 加载失败: %1").arg(QString::fromStdString(e.what())));
        }
        reportProgress(100, QStringLiteral("OBJ 加载失败"));
        return nullptr;
    }));
}

// 计算场景中所有点（相机光心、点云、模型顶点）的质心作为场景中心。
// 使用缓存，仅在数据变更后重新计算。
QVector3D CameraSceneWidget::sceneCenter() const
{
    if (_cacheDirty) invalidateCache();
    return _cachedCenter;
}

// 计算场景中所有点到质心的最大距离，用于自适应相机距离、投影远裁平面等。
// 使用缓存，仅在数据变更后重新计算。
float CameraSceneWidget::sceneRadius() const
{
    if (_cacheDirty) invalidateCache();
    return _cachedRadius;
}

QPointF CameraSceneWidget::projectToScreen(const QVector3D &p, bool *ok) const
{
    const QVector3D center = sceneCenter();
    const float radius = sceneRadius();
    const float distance = qMax(radius * 0.001f, radius * (3.2f / qMax(0.1f, _zoomScale)));

    // 从 +Z 方向看向原点：世界X→屏幕右，世界Y→屏幕上，与overlay/arcball坐标系一致
    const QVector3D eye = center + QVector3D(0.0f, 0.0f, distance);

    QMatrix4x4 view;
    view.lookAt(eye, center, QVector3D(0.0f, 1.0f, 0.0f));

    QMatrix4x4 model;
    model.setToIdentity();
    model.translate(center);
    model.rotate(_viewRot);
    model.translate(-center);

    const float nearPlane = qMax(1e-4f, distance * 0.001f);
    const float farPlane  = qMax(1000.0f, distance * 100.0f + radius * 50.0f);
    QMatrix4x4 proj;
    const float aspect = qMax(1.0f, float(width()) / qMax(1, height()));
    proj.perspective(45.0f, aspect, nearPlane, farPlane);

    QVector4D clip = proj * view * model * QVector4D(p, 1.0f);
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

float CameraSceneWidget::cameraFrustumBase() const
{
    if (_cacheDirty) invalidateCache();
    return _cachedCameraFrustumBase;
}

float CameraSceneWidget::cameraImagePlaneHalfExtent() const
{
    if (_cacheDirty) invalidateCache();

    float halfExtent = qMax(_cachedCameraFrustumBase * 3.8f, _cachedRadius * 0.055f);
    if (_poses.size() > 1)
    {
        std::vector<float> nearestDistances;
        nearestDistances.reserve(static_cast<size_t>(_poses.size()));
        for (qsizetype i = 0; i < _poses.size(); ++i)
        {
            float nearest = std::numeric_limits<float>::max();
            for (qsizetype j = 0; j < _poses.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                nearest = qMin(nearest, (_poses[i].center - _poses[j].center).length());
            }
            if (std::isfinite(nearest))
            {
                nearestDistances.push_back(nearest);
            }
        }
        if (!nearestDistances.empty())
        {
            const size_t medianIndex = nearestDistances.size() / 2;
            std::nth_element(nearestDistances.begin(),
                             nearestDistances.begin() + static_cast<std::ptrdiff_t>(medianIndex),
                             nearestDistances.end());
            halfExtent = qMin(halfExtent, nearestDistances[medianIndex] * 0.48f);
        }
    }

    return qBound(0.2f, halfExtent, qMax(0.5f, _cachedRadius * 0.12f));
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

void CameraSceneWidget::drawFloorPivotCross(QPainter &painter) const
{
    if (_cacheDirty) invalidateCache();

    const QVector3D floorPivot((_cachedAABBMin.x() + _cachedAABBMax.x()) * 0.5f,
                               (_cachedAABBMin.y() + _cachedAABBMax.y()) * 0.5f,
                               _cachedAABBMin.z());
    const float halfSize = qMax(0.25f, _cachedRadius * 0.045f);

    bool okCenter = false;
    bool okX1 = false;
    bool okX2 = false;
    bool okY1 = false;
    bool okY2 = false;
    const QPointF c = projectToScreen(floorPivot, &okCenter);
    const QPointF x1 = projectToScreen(floorPivot - QVector3D(halfSize, 0.0f, 0.0f), &okX1);
    const QPointF x2 = projectToScreen(floorPivot + QVector3D(halfSize, 0.0f, 0.0f), &okX2);
    const QPointF y1 = projectToScreen(floorPivot - QVector3D(0.0f, halfSize, 0.0f), &okY1);
    const QPointF y2 = projectToScreen(floorPivot + QVector3D(0.0f, halfSize, 0.0f), &okY2);
    if (!okCenter)
    {
        return;
    }

    painter.setBrush(Qt::NoBrush);
    painter.setPen(QPen(QColor(125, 128, 132, 150), 1.4));
    if (okX1 && okX2)
    {
        painter.drawLine(x1, x2);
    }
    if (okY1 && okY2)
    {
        painter.drawLine(y1, y2);
    }
    painter.setPen(QPen(QColor(105, 108, 112, 170), 1.0));
    painter.drawLine(c + QPointF(-4.0, 0.0), c + QPointF(4.0, 0.0));
    painter.drawLine(c + QPointF(0.0, -4.0), c + QPointF(0.0, 4.0));
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

// OpenGL 初始化：获取 GL 4.3 Core Profile 函数对象，编译 shader，创建 VAO/VBO
void CameraSceneWidget::initializeGL()
{
    _gl = context() ? QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_3_Core>(context()) : nullptr;
    if (!_gl) return;
    _gl->initializeOpenGLFunctions();
    _gl->glEnable(GL_DEPTH_TEST);
    _gl->glEnable(GL_BLEND);
    _gl->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    _gl->glEnable(GL_PROGRAM_POINT_SIZE); // 允许 vertex shader 通过 gl_PointSize 控制点大小

    // ── 顶点色直通 shader（点云 + 线框）────────────────────────────────────
    static const char *colorVert =
        "#version 430 core\n"
        "layout(location=0) in vec3 aPos;\n"
        "layout(location=1) in vec3 aColor;\n"
        "uniform mat4 uMVP;\n"
        "uniform float uPointSize;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "    gl_Position  = uMVP * vec4(aPos, 1.0);\n"
        "    gl_PointSize = uPointSize;\n"
        "    vColor = aColor;\n"
        "}\n";
    static const char *colorFrag =
        "#version 430 core\n"
        "in vec3 vColor;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "    fragColor = vec4(vColor, 1.0);\n"
        "}\n";

    _colorProgram = new QOpenGLShaderProgram(this);
    _colorProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,   colorVert);
    _colorProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, colorFrag);
    _colorProgram->link();

    // ── Phong 光照 shader（三角网格）────────────────────────────────────────
    static const char *meshVert =
        "#version 430 core\n"
        "layout(location=0) in vec3 aPos;\n"
        "layout(location=1) in vec3 aNormal;\n"
        "layout(location=2) in vec3 aColor;\n"
        "uniform mat4 uMVP;\n"
        "uniform mat3 uNormalMat;\n"
        "uniform float uPointSize;\n"
        "out vec3 vNormal;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "    gl_Position = uMVP * vec4(aPos, 1.0);\n"
        "    gl_PointSize = uPointSize;\n"
        "    vNormal = uNormalMat * aNormal;\n"
        "    vColor  = aColor;\n"
        "}\n";
    static const char *meshFrag =
        "#version 430 core\n"
        "in vec3 vNormal;\n"
        "in vec3 vColor;\n"
        "uniform vec3 uLightDir;\n"
        "out vec4 fragColor;\n"
        "vec3 srgbToLinear(vec3 c) {\n"
        "    return pow(max(c, vec3(0.0)), vec3(2.2));\n"
        "}\n"
        "vec3 linearToSrgb(vec3 c) {\n"
        "    return pow(clamp(c, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));\n"
        "}\n"
        "void main() {\n"
        "    vec3 n = normalize(vNormal);\n"
        "    if (!gl_FrontFacing) n = -n;\n"
        "    vec3 L = normalize(uLightDir);\n"
        "    float diff = max(dot(n, L), 0.0);\n"
        "    vec3 R = reflect(-L, n);\n"
        "    float spec = pow(max(R.z, 0.0), 32.0) * 0.25;\n"
        "    vec3 baseLinear = srgbToLinear(vColor);\n"
        "    vec3 litLinear = baseLinear * (0.55 + 0.75 * diff) + vec3(spec);\n"
        "    fragColor = vec4(linearToSrgb(litLinear), 1.0);\n"
        "}\n";

    _meshProgram = new QOpenGLShaderProgram(this);
    _meshProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,   meshVert);
    _meshProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, meshFrag);
    _meshProgram->link();

    // 创建 VAO/VBO（GL 资源，需在 context 活跃时创建）
    _pointVao.create();   _pointVbo.create();
    _meshVao.create();    _meshVbo.create();
    _modelPtVao.create(); _modelPtVbo.create();
    _lineVao.create();    _lineVbo.create();

    _gpuDirty = true;
}

// 视口尺寸变化时更新 OpenGL 的视口矩形
void CameraSceneWidget::resizeGL(int w, int h)
{
    if (!_gl) return;
    _gl->glViewport(0, 0, w, h);
    _gpuDirty = true; // 包围盒线框依赖 AABB，重建一次以防万一
}

// 将点云/模型/包围盒数据上传到 GPU（VBO/VAO）。
// 在 paintGL 检测到 _gpuDirty 时调用，避免每帧重复上传。
void CameraSceneWidget::uploadGpuData()
{
    if (!_gl) return;

    // ── 辅助：建立颜色直通 VAO（stride=6 floats: xyz rgb）────────────────
    auto setupColorVao = [this](QOpenGLVertexArrayObject &vao,
                                QOpenGLBuffer &vbo,
                                const QVector<float> &data)
    {
        vao.bind();
        vbo.bind();
        vbo.allocate(data.constData(), data.size() * sizeof(float));
        const int stride = 6 * sizeof(float);
        _gl->glEnableVertexAttribArray(0);
        _gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
        _gl->glEnableVertexAttribArray(1);
        _gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(3 * sizeof(float)));
        vbo.release();
        vao.release();
    };

    // ── 1. 点云（_cloud，无面片且无法向量，颜色直通）──────────────────────────
    _pointCount = 0;
    _modelPtCount = 0;
    if (!(_cloud.size() == 0) && !_cloud.hasFaces() && !_cloud.hasNormals()) {
        const bool hasColors = _cloud.hasColors();
        QVector<float> data;
        data.reserve((int)_cloud.size() * 6);
        for (std::size_t i = 0; i < _cloud.size(); ++i) {
            data << _cloud.points()(static_cast<plamatrix::Index>(i), 0)
                 << _cloud.points()(static_cast<plamatrix::Index>(i), 1)
                 << _cloud.points()(static_cast<plamatrix::Index>(i), 2);
            if (hasColors) {
                data << _cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 0) / 255.f
                     << _cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 1) / 255.f
                     << _cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 2) / 255.f;
            } else {
                data << 0.45f << 0.45f << 0.50f;
            }
        }
        if (_preferModelPointRender) {
            setupColorVao(_modelPtVao, _modelPtVbo, data);
            _modelPtCount = (int)_cloud.size();
        } else {
            setupColorVao(_pointVao, _pointVbo, data);
            _pointCount = (int)_cloud.size();
        }
    }

    // ── 2. 网格（hasFaces）或含法向量点云（!hasFaces && hasNormals）──────────
    _meshVertCount = 0;
    _meshHasFaces = false;
    if (!(_cloud.size() == 0) && _cloud.hasFaces()) {
        _meshHasFaces = true;
        const bool hasVertCol = _cloud.hasColors();
        const bool hasNrm     = _cloud.hasNormals();
        const std::size_t Nv  = _cloud.size();

        // 计算（或复用）逐顶点法向量
        std::vector<QVector3D> vNormals(Nv);
        if (hasNrm) {
            for (std::size_t i = 0; i < Nv; ++i) {
                vNormals[i] = QVector3D(
                    _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0),
                    _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1),
                    _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2));
            }
        } else {
            const auto nF = static_cast<std::size_t>(_cloud.faces()->rows());
            for (std::size_t fi = 0; fi < nF; ++fi) {
                const std::size_t i0 = static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 0));
                const std::size_t i1 = static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 1));
                const std::size_t i2 = static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 2));
                if (i0 >= Nv || i1 >= Nv || i2 >= Nv) continue;
                float p0x = _cloud.points()(static_cast<plamatrix::Index>(i0), 0);
                float p0y = _cloud.points()(static_cast<plamatrix::Index>(i0), 1);
                float p0z = _cloud.points()(static_cast<plamatrix::Index>(i0), 2);
                float p1x = _cloud.points()(static_cast<plamatrix::Index>(i1), 0);
                float p1y = _cloud.points()(static_cast<plamatrix::Index>(i1), 1);
                float p1z = _cloud.points()(static_cast<plamatrix::Index>(i1), 2);
                float p2x = _cloud.points()(static_cast<plamatrix::Index>(i2), 0);
                float p2y = _cloud.points()(static_cast<plamatrix::Index>(i2), 1);
                float p2z = _cloud.points()(static_cast<plamatrix::Index>(i2), 2);
                const QVector3D fn = QVector3D::crossProduct(
                    QVector3D(p1x - p0x, p1y - p0y, p1z - p0z),
                    QVector3D(p2x - p0x, p2y - p0y, p2z - p0z));
                vNormals[i0] += fn;
                vNormals[i1] += fn;
                vNormals[i2] += fn;
            }
            for (auto &n : vNormals) n.normalize();
        }

        // 展开面片为平坦顶点数组（stride=9 floats: xyz nxnynz rgb）
        QVector<float> data;
        data.reserve(static_cast<int>(_cloud.faces()->rows()) * 3 * 9);
        const auto nFaces = static_cast<std::size_t>(_cloud.faces()->rows());
        for (std::size_t fi = 0; fi < nFaces; ++fi) {
            const std::size_t i0 = static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 0));
            const std::size_t i1 = static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 1));
            const std::size_t i2 = static_cast<std::size_t>(_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 2));
            if (i0 >= Nv || i1 >= Nv || i2 >= Nv) continue;
            const std::size_t indices[3] = {i0, i1, i2};
            for (int vi = 0; vi < 3; ++vi) {
                const std::size_t idx = indices[vi];
                data << _cloud.points()(static_cast<plamatrix::Index>(idx), 0)
                     << _cloud.points()(static_cast<plamatrix::Index>(idx), 1)
                     << _cloud.points()(static_cast<plamatrix::Index>(idx), 2);
                data << vNormals[idx].x() << vNormals[idx].y() << vNormals[idx].z();
                if (hasVertCol) {
                    data << _cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 0) / 255.f
                         << _cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 1) / 255.f
                         << _cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 2) / 255.f;
                } else {
                    data << 0.55f << 0.55f << 0.58f;
                }
            }
        }
        _meshVao.bind();
        _meshVbo.bind();
        _meshVbo.allocate(data.constData(), data.size() * sizeof(float));
        const int stride = 9 * sizeof(float);
        _gl->glEnableVertexAttribArray(0);
        _gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
        _gl->glEnableVertexAttribArray(1);
        _gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(3 * sizeof(float)));
        _gl->glEnableVertexAttribArray(2);
        _gl->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(6 * sizeof(float)));
        _meshVbo.release();
        _meshVao.release();
        _meshVertCount = data.size() / 9;
    } else if (!(_cloud.size() == 0) && !_cloud.hasFaces() && _cloud.hasNormals()) {
        // 含法向量但无面片 → GL_POINTS + Phong 光照（稠密点云等）
        const bool hasColors = _cloud.hasColors();
        const std::size_t Nv = _cloud.size();
        QVector<float> data;
        data.reserve((int)Nv * 9);
        for (std::size_t i = 0; i < Nv; ++i) {
            data << _cloud.points()(static_cast<plamatrix::Index>(i), 0)
                 << _cloud.points()(static_cast<plamatrix::Index>(i), 1)
                 << _cloud.points()(static_cast<plamatrix::Index>(i), 2);
            data << _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0)
                 << _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1)
                 << _cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2);
            if (hasColors) {
                data << _cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 0) / 255.f
                     << _cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 1) / 255.f
                     << _cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 2) / 255.f;
            } else {
                data << 0.55f << 0.55f << 0.58f;
            }
        }
        _meshVao.bind();
        _meshVbo.bind();
        _meshVbo.allocate(data.constData(), data.size() * sizeof(float));
        const int stride = 9 * sizeof(float);
        _gl->glEnableVertexAttribArray(0);
        _gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
        _gl->glEnableVertexAttribArray(1);
        _gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(3 * sizeof(float)));
        _gl->glEnableVertexAttribArray(2);
        _gl->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(6 * sizeof(float)));
        _meshVbo.release();
        _meshVao.release();
        _meshVertCount = (int)Nv;
    }

    // ── 3. 包围盒线框（12 条边 × 2 顶点）────────────────────────────────
    _lineCount = 0;
    {
        if (_cacheDirty) invalidateCache();
        const bool empty = _poses.isEmpty() && (_cloud.size() == 0);
        if (!empty) {
            const QVector3D &mn = _cachedAABBMin;
            const QVector3D &mx = _cachedAABBMax;
            const QVector3D v000(mn.x(), mn.y(), mn.z());
            const QVector3D v001(mn.x(), mn.y(), mx.z());
            const QVector3D v010(mn.x(), mx.y(), mn.z());
            const QVector3D v011(mn.x(), mx.y(), mx.z());
            const QVector3D v100(mx.x(), mn.y(), mn.z());
            const QVector3D v101(mx.x(), mn.y(), mx.z());
            const QVector3D v110(mx.x(), mx.y(), mn.z());
            const QVector3D v111(mx.x(), mx.y(), mx.z());

            constexpr float r = 0.72f, g = 0.72f, b = 0.76f;
            auto addEdge = [&](QVector<float> &d, const QVector3D &a, const QVector3D &b2) {
                d << a.x() << a.y() << a.z() << r << g << b;
                d << b2.x() << b2.y() << b2.z() << r << g << b;
            };
            QVector<float> data;
            data.reserve(24 * 6);
            addEdge(data, v000, v001); addEdge(data, v000, v010); addEdge(data, v000, v100);
            addEdge(data, v111, v110); addEdge(data, v111, v101); addEdge(data, v111, v011);
            addEdge(data, v001, v011); addEdge(data, v001, v101);
            addEdge(data, v010, v011); addEdge(data, v010, v110);
            addEdge(data, v100, v101); addEdge(data, v100, v110);
            setupColorVao(_lineVao, _lineVbo, data);
            _lineCount = 24; // 12 条边 × 2 顶点
        }
    }

    _gpuDirty = false;
}



void CameraSceneWidget::paintGL()
{
    if (!_gl || !_colorProgram || !_meshProgram) {
        QPainter p(this);
        p.fillRect(rect(), Qt::white);
        return;
    }

    // 按需上传 GPU 数据
    if (_gpuDirty) uploadGpuData();

    _gl->glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    _gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ── 构建 MVP 矩阵（与旧代码逻辑保持一致）───────────────────────────────
    const QVector3D center = sceneCenter();
    const float radius = sceneRadius();
    const float distance = qMax(radius * 0.001f, radius * (3.2f / qMax(0.1f, _zoomScale)));
    const QVector3D eye = center + QVector3D(0.0f, 0.0f, distance);

    QMatrix4x4 view;
    view.lookAt(eye, center, QVector3D(0.0f, 1.0f, 0.0f));

    QMatrix4x4 model;
    model.setToIdentity();
    model.translate(center);
    model.rotate(_viewRot);
    model.translate(-center);

    const float nearPlane = qMax(1e-4f, distance * 0.001f);
    const float farPlane  = qMax(1000.0f, distance * 100.0f + radius * 50.0f);
    QMatrix4x4 proj;
    const float aspect = qMax(1.0f, float(width()) / qMax(1, height()));
    proj.perspective(45.0f, aspect, nearPlane, farPlane);

    QMatrix4x4 shift;
    shift.setToIdentity();
    shift.translate(float(2.0 * _sceneOffsetPx.x() / qMax(1, width())),
                    float(-2.0 * _sceneOffsetPx.y() / qMax(1, height())),
                    0.0f);
    const QMatrix4x4 mv  = view * model;
    const QMatrix4x4 mvp = shift * proj * mv;

    auto drawOpaquePointSet = [this, &mvp](QOpenGLVertexArrayObject &vao,
                                           int pointCount,
                                           float pointSize)
    {
        if (pointCount <= 0) {
            return;
        }

        const float depthOnlyPointSize = qMax(pointSize + 1.25f, pointSize * 1.35f);

        _gl->glEnable(GL_DEPTH_TEST);
        _gl->glDepthMask(GL_TRUE);
        _gl->glDepthFunc(GL_LEQUAL);
        _gl->glDisable(GL_BLEND);

        // 先写入稍大一点的深度轮廓，减少后层点从前层点之间漏出来。
        _gl->glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        _colorProgram->bind();
        _colorProgram->setUniformValue("uMVP", mvp);
        _colorProgram->setUniformValue("uPointSize", depthOnlyPointSize);
        vao.bind();
        _gl->glDrawArrays(GL_POINTS, 0, pointCount);
        vao.release();
        _colorProgram->release();

        _gl->glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        _colorProgram->bind();
        _colorProgram->setUniformValue("uMVP", mvp);
        _colorProgram->setUniformValue("uPointSize", pointSize);
        vao.bind();
        _gl->glDrawArrays(GL_POINTS, 0, pointCount);
        vao.release();
        _colorProgram->release();

        _gl->glEnable(GL_BLEND);
        _gl->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    };

    // ── 点云（颜色直通）──────────────────────────────────────────────────
    if (_pointCount > 0) {
        drawOpaquePointSet(_pointVao, _pointCount, 1.8f);
    }

    // ── 三角网格（Phong 双面光照）────────────────────────────────────────
    if (_meshVertCount > 0) {
        _gl->glDisable(GL_BLEND);
        _gl->glEnable(GL_DEPTH_TEST);
        _gl->glDepthMask(GL_TRUE);
        _gl->glDepthFunc(GL_LEQUAL);
        _meshProgram->bind();
        _meshProgram->setUniformValue("uMVP",       mvp);
        _meshProgram->setUniformValue("uNormalMat", mv.normalMatrix());
        // 固定方向光：视角坐标系左上前方（与旧 GL_LIGHT0 位置一致）
        _meshProgram->setUniformValue("uLightDir",  QVector3D(0.5f, 0.8f, 0.6f));
        _meshVao.bind();
        if (_meshHasFaces) {
            _gl->glDrawArrays(GL_TRIANGLES, 0, _meshVertCount);
        } else {
            _meshProgram->setUniformValue("uPointSize", 1.5f);
            _gl->glDrawArrays(GL_POINTS, 0, _meshVertCount);
        }
        _meshVao.release();
        _meshProgram->release();
        _gl->glEnable(GL_BLEND);
        _gl->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // ── 模型顶点（无面片时作为点云）──────────────────────────────────────
    if (_modelPtCount > 0) {
        drawOpaquePointSet(_modelPtVao, _modelPtCount, _modelPointSize);
    }

    // ── 包围盒线框 ────────────────────────────────────────────────────────
    if (_lineCount > 0) {
        _colorProgram->bind();
        _colorProgram->setUniformValue("uMVP",       mvp);
        _colorProgram->setUniformValue("uPointSize", 1.0f);
        _lineVao.bind();
        _gl->glLineWidth(1.0f);
        _gl->glDrawArrays(GL_LINES, 0, _lineCount);
        _lineVao.release();
        _colorProgram->release();
    }

    drawOverlay();
}

void CameraSceneWidget::drawOverlay()
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

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

    if (_showCameras)
    {
        const float planeHalfExtent = cameraImagePlaneHalfExtent();
        const float planeHalfHeight = planeHalfExtent * 0.68f;
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

        for (qsizetype poseIndex = 0; poseIndex < _poses.size(); ++poseIndex)
        {
            const CameraPose &pose = _poses.at(poseIndex);
            const bool highlighted = isCameraHighlighted(pose);
            const QColor selectedCameraFill(255, 86, 104, 185);
            const QColor normalCameraFill(40, 118, 190, cameraCount > 200 ? 88 : 150);
            const QColor planeOutlineColor = highlighted ? QColor(210, 42, 62, 235)
                                                         : QColor(32, 92, 160, cameraCount > 200 ? 115 : 170);
            const QColor labelStemColor = highlighted ? QColor(95, 24, 34, 220)
                                                      : QColor(45, 45, 45, cameraCount > 120 ? 150 : 190);
            bool ok = false;
            const QPointF pc = projectToScreen(pose.center, &ok);
            if (!ok) continue;

            const QVector3D right(pose.rotation(0, 0), pose.rotation(1, 0), pose.rotation(2, 0));
            const QVector3D up(pose.rotation(0, 1), pose.rotation(1, 1), pose.rotation(2, 1));
            const QVector3D p1 = pose.center + right * planeHalfExtent + up * planeHalfHeight;
            const QVector3D p2 = pose.center - right * planeHalfExtent + up * planeHalfHeight;
            const QVector3D p3 = pose.center - right * planeHalfExtent - up * planeHalfHeight;
            const QVector3D p4 = pose.center + right * planeHalfExtent - up * planeHalfHeight;

            bool ok1 = false;
            bool ok2 = false;
            bool ok3 = false;
            bool ok4 = false;
            const QPointF pp1 = projectToScreen(p1, &ok1);
            const QPointF pp2 = projectToScreen(p2, &ok2);
            const QPointF pp3 = projectToScreen(p3, &ok3);
            const QPointF pp4 = projectToScreen(p4, &ok4);
            if (ok1 && ok2 && ok3 && ok4)
            {
                QPolygonF imagePlane;
                imagePlane << pp1 << pp2 << pp3 << pp4;
                painter.setPen(Qt::NoPen);
                painter.setBrush(highlighted ? selectedCameraFill : normalCameraFill);
                painter.drawPolygon(imagePlane);
                painter.setBrush(Qt::NoBrush);
                const QPen planeOutlinePen(planeOutlineColor, highlighted ? 2.2 : 1.1);
                painter.setPen(planeOutlinePen);
                painter.drawPolygon(imagePlane);
                if (highlighted)
                {
                    painter.setPen(QPen(QColor(255, 245, 248, 210), 1.0));
                    painter.drawLine((pp1 + pp3) * 0.5, (pp2 + pp4) * 0.5);
                }
            }

            const bool drawCameraLabel = highlighted
                || drawAllCameraLabels
                || (poseIndex == 0)
                || (poseIndex == _poses.size() - 1)
                || (static_cast<int>(poseIndex) % cameraLabelStride == 0);
            if (drawCameraLabel)
            {
                const QString labelSource = pose.imagePath.isEmpty() ? pose.name : pose.imagePath;
                const QString label = QFileInfo(labelSource).fileName().isEmpty()
                    ? pose.name
                    : QFileInfo(labelSource).fileName();
                const QPointF planeTop = ok1 && ok2 ? (pp1 + pp2) * 0.5 : pc;
                const qreal labelLift = highlighted ? 30.0 : 24.0;
                const QPointF labelAnchor = planeTop + QPointF(0.0, -labelLift);
                painter.setPen(QPen(labelStemColor, highlighted ? 1.35 : 1.0));
                painter.drawLine(planeTop, labelAnchor);
                painter.setPen(highlighted
                    ? QColor(210, 45, 65, 230)
                    : (drawAllCameraLabels ? QColor(60, 60, 60) : QColor(45, 45, 45, 170)));
                painter.drawText(labelAnchor + QPointF(5.0, -2.0), label);
            }
        }
    }

    drawFloorPivotCross(painter);

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

    QString saveError;
    if (!saveCurrentPointCloudToSource(&saveError))
    {
        if (errorMessage)
        {
            *errorMessage = saveError;
        }
        emit manualPruneSaveFailed(saveError);
        return false;
    }

    emit manualPruneUndone(static_cast<int>(_cloud.size()));
    emit manualPruneSaved(_currentCloudPath, static_cast<int>(_cloud.size()));
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
    collectPointIndicesInScreenRect(rect, &selectedIndices);
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
    if (_currentCloudPath.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前点云来源未知，无法覆盖保存。");
        }
        return false;
    }

    // Determine file format from extension
    const std::string stdPath = _currentCloudPath.toStdString();
    const bool isPly = (stdPath.size() >= 4 &&
        (stdPath.substr(stdPath.size() - 4) == ".ply" || stdPath.substr(stdPath.size() - 4) == ".PLY"));

    try
    {
        if (isPly)
        {
            plapoint::io::writePly<float>(stdPath, _cloud);
        }
        else
        {
            plapoint::io::writeXyz<float>(stdPath, _cloud);
        }
        return true;
    }
    catch (const std::exception &e)
    {
        if (errorMessage)
        {
            *errorMessage = QString::fromStdString(e.what());
        }
        return false;
    }
}

void CameraSceneWidget::mousePressEvent(QMouseEvent *event)
{
    setFocus();

    if (_manualPruneMode && event->button() == Qt::RightButton)
    {
        _manualSelecting = true;
        _manualSelectStart = event->pos();
        _manualSelectRect = QRect(_manualSelectStart, _manualSelectStart);
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
            QString saveError;
            if (saveCurrentPointCloudToSource(&saveError))
            {
                emit manualPruneSaved(_currentCloudPath, static_cast<int>(_cloud.size()));
            }
            else
            {
                emit manualPruneSaveFailed(saveError);
            }
            collectPointIndicesInScreenRect(_manualSelectRect, &_manualPreviewIndices);
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
    QOpenGLWidget::mousePressEvent(event);
}

void CameraSceneWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (_manualPruneMode && _manualSelecting && (event->buttons() & Qt::RightButton))
    {
        _manualSelectRect = QRect(_manualSelectStart, event->pos()).normalized();
        collectPointIndicesInScreenRect(_manualSelectRect, &_manualPreviewIndices);
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
    QOpenGLWidget::mouseMoveEvent(event);
}

void CameraSceneWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (_manualPruneMode && event->button() == Qt::RightButton)
    {
        _manualSelecting = false;
        _manualSelectRect = _manualSelectRect.normalized();
        collectPointIndicesInScreenRect(_manualSelectRect, &_manualPreviewIndices);
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
    QOpenGLWidget::mouseReleaseEvent(event);
}

void CameraSceneWidget::wheelEvent(QWheelEvent *event)
{
    if (_leftDragging || _middleDragging) {
        event->ignore();
        return;
    }

    const QPoint angle = event->angleDelta();
    if (!angle.isNull()) {
        const float factor = (angle.y() > 0) ? 1.10f : 0.90f;
        _zoomScale = _zoomScale * factor;   // 无缩放上下限
        clampSceneOffset();
        update();
    }
    event->accept();
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
    QOpenGLWidget::keyPressEvent(event);
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

    const QJsonArray images = xjw::gui::project::projectImageEntries(_projectManager->currentMeta());
    for (const QJsonValue &imageValue : images)
    {
        const QJsonObject imageObject = imageValue.toObject();
        xjw::Camera camera;
        if (!xjw::gui::project::imageCameraFromEntry(imageObject, &camera))
        {
            continue;
        }

        const std::array<double, 3> cameraCenter = camera.cameraCenter();
        const std::array<double, 9> cameraToWorldRotation = camera.cameraToWorldRotation();

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
