#include "ProjectDenseReconstructionManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "ProjectDenseWorkflowConfig.h"
#include "ProjectDepthFrameUtils.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "Logger.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "SparseCloudPreprocessor.h"
#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/features/normal_estimation.h>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>
#include <QPushButton>
#include <QRegularExpression>
#include <QSet>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

using xjw::gui::project::buildDepthGenConfig;
using xjw::gui::project::denseGenerationSettingsFromJson;
using xjw::gui::project::denseRefineSettingsFromJson;
using xjw::gui::project::makeDenseResultRecord;
using xjw::gui::project::makeDepthResultRecord;
using xjw::gui::project::buildStoredFusionFrame;
using xjw::gui::project::collectLatestStoredDepthFrames;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::rawConfidenceStoragePath;
using xjw::gui::project::rawDepthStoragePath;
using xjw::gui::project::resolveLatestDenseCloudPath;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::gui::project::StoredDepthFrameRecord;
using xjw::gui::project::upsertMetaArrayRecordByPath;
using xjw::gui::project::upsertProjectRecordByPath;

namespace
{

QString utcNowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}

enum class ExistingDepthAction
{
    Cancel,
    Overwrite,
    ContinueMissing
};

QSet<int> collectExistingDepthFrameIndices(const QString &outputDir, int expectedCount)
{
    QSet<int> indices;
    for (int index = 0; index < expectedCount; ++index)
    {
        const QString pngPath = QDir(outputDir).filePath(QStringLiteral("depth_%1.png").arg(index));
        if (QFileInfo::exists(pngPath))
        {
            indices.insert(index);
        }
    }
    return indices;
}

ExistingDepthAction askExistingDepthAction(QWidget *parent,
                                           int existingCount,
                                           int totalCount,
                                           const QString &outputDir)
{
    QMessageBox box(parent);
    box.setIcon(QMessageBox::Question);
    box.setWindowTitle(QStringLiteral("深度图估计"));
    box.setText(QStringLiteral("检测到已有深度图结果（%1/%2）。").arg(existingCount).arg(totalCount));
    box.setInformativeText(QStringLiteral("输出目录：%1\n请选择覆盖重算，或继续生成未完成的帧。")
                               .arg(QDir::toNativeSeparators(outputDir)));

    QPushButton *overwriteButton = box.addButton(QStringLiteral("覆盖重算"), QMessageBox::DestructiveRole);
    QPushButton *continueButton = box.addButton(QStringLiteral("继续生成未完成帧"), QMessageBox::AcceptRole);
    QPushButton *cancelButton = box.addButton(QStringLiteral("取消"), QMessageBox::RejectRole);
    box.setDefaultButton(continueButton);

    box.exec();
    if (box.clickedButton() == overwriteButton)
    {
        return ExistingDepthAction::Overwrite;
    }
    if (box.clickedButton() == continueButton)
    {
        return ExistingDepthAction::ContinueMissing;
    }
    if (box.clickedButton() == cancelButton)
    {
        return ExistingDepthAction::Cancel;
    }
    return ExistingDepthAction::Cancel;
}

void removeDepthArtifactsForIndices(const QString &outputDir, const QSet<int> &indices)
{
    for (const int index : indices)
    {
        const QString pngPath = QDir(outputDir).filePath(QStringLiteral("depth_%1.png").arg(index));
        const QString rawPath = rawDepthStoragePath(pngPath);
        const QString confPath = rawConfidenceStoragePath(pngPath);
        QFile::remove(pngPath);
        QFile::remove(rawPath);
        QFile::remove(confPath);
    }
}

void upsertExistingDepthRecords(ProjectData *projectData,
                                const QStringList &selectedImages,
                                const QString &sparseXyz,
                                const QString &outputDir,
                                const QSet<int> &indices)
{
    if (!projectData || indices.isEmpty())
    {
        return;
    }

    QJsonObject meta = projectData->metadata();
    bool changed = false;

    for (const int index : indices)
    {
        if (index < 0 || index >= selectedImages.size())
        {
            continue;
        }

        const QString pngPath = QDir(outputDir).filePath(QStringLiteral("depth_%1.png").arg(index));
        if (!QFileInfo::exists(pngPath))
        {
            continue;
        }

        QJsonObject depthResult = makeDepthResultRecord(utcNowIso(),
                                                        pngPath,
                                                        0,
                                                        0,
                                                        sparseXyz,
                                                        selectedImages.at(index));
        depthResult[QStringLiteral("raw_depth_path")] = rawDepthStoragePath(pngPath);
        depthResult[QStringLiteral("raw_confidence_path")] = rawConfidenceStoragePath(pngPath);
        depthResult[QStringLiteral("mvs_output_dir")] = outputDir;
        upsertMetaArrayRecordByPath(&meta,
                                    QStringLiteral("depth_map_results"),
                                    QStringLiteral("depth_png"),
                                    depthResult);
        changed = true;
    }

    if (changed)
    {
        persistProjectMeta(projectData, meta, true);
    }
}

using PlaPC = plapoint::PointCloud<float, plamatrix::Device::CPU>;

PlaPC densePointsToPointCloud(const std::vector<xjw::mvs::DensePoint> &cloud)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(cloud.size(), 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        pts(i, 0) = cloud[i].x;
        pts(i, 1) = cloud[i].y;
        pts(i, 2) = cloud[i].z;
        colors(i, 0) = cloud[i].r;
        colors(i, 1) = cloud[i].g;
        colors(i, 2) = cloud[i].b;
    }
    PlaPC pc(std::move(pts));
    pc.setColors(std::move(colors));
    return pc;
}

PlaPC fusedPointsToPointCloud(const std::vector<xjw::mvs::FusedPoint> &cloud,
                              bool keepColor,
                              bool keepNormals)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        pts(i, 0) = cloud[i].x;
        pts(i, 1) = cloud[i].y;
        pts(i, 2) = cloud[i].z;
    }
    PlaPC pc(std::move(pts));

    if (keepColor)
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            colors(i, 0) = cloud[i].r;
            colors(i, 1) = cloud[i].g;
            colors(i, 2) = cloud[i].b;
        }
        pc.setColors(std::move(colors));
    }

    if (keepNormals)
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> nrm(cloud.size(), 3);
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            nrm(i, 0) = cloud[i].nx;
            nrm(i, 1) = cloud[i].ny;
            nrm(i, 2) = cloud[i].nz;
        }
        pc.setNormals(std::move(nrm));
    }

    return pc;
}

bool readPointCloudPly(const QString &path, PlaPC *cloud, QString *errorMessage)
{
    std::ifstream f(path.toStdString(), std::ios::binary);
    if (!f)
    {
        if (errorMessage) *errorMessage = QStringLiteral("无法打开文件: %1").arg(path);
        return false;
    }

    std::string line;
    std::getline(f, line);
    if (line != "ply") { if (errorMessage) *errorMessage = QStringLiteral("不是PLY文件"); return false; }

    std::getline(f, line); // format line
    bool isBinary = (line.find("binary_little_endian") != std::string::npos);

    int nVerts = 0;
    bool hasColor = false;
    std::vector<std::string> props;
    while (std::getline(f, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") break;
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element") { std::string e; iss >> e >> nVerts; }
        else if (token == "property")
        {
            std::string type, name;
            iss >> type >> name;
            props.push_back(name);
            if (name == "red") hasColor = true;
        }
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(nVerts, 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(nVerts, 3);
    if (isBinary)
    {
        for (int i = 0; i < nVerts; ++i)
        {
            for (const auto &p : props)
            {
                if (p == "x" || p == "y" || p == "z")
                {
                    float v; f.read(reinterpret_cast<char*>(&v), 4);
                    int col = (p == "x") ? 0 : (p == "y") ? 1 : 2;
                    pts(i, col) = v;
                }
                else if (p == "red")   { uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); colors(i, 0) = v; }
                else if (p == "green") { uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); colors(i, 1) = v; }
                else if (p == "blue")  { uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); colors(i, 2) = v; }
                else { float dummy; f.read(reinterpret_cast<char*>(&dummy), 4); } // skip unknown float props
            }
        }
    }
    else
    {
        for (int i = 0; i < nVerts; ++i)
        {
            std::getline(f, line);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream iss(line);
            for (const auto &p : props)
            {
                if (p == "x") { float v; iss >> v; pts(i, 0) = v; }
                else if (p == "y") { float v; iss >> v; pts(i, 1) = v; }
                else if (p == "z") { float v; iss >> v; pts(i, 2) = v; }
                else if (p == "red")   { int v; iss >> v; colors(i, 0) = static_cast<uint8_t>(v); }
                else if (p == "green") { int v; iss >> v; colors(i, 1) = static_cast<uint8_t>(v); }
                else if (p == "blue")  { int v; iss >> v; colors(i, 2) = static_cast<uint8_t>(v); }
                else { float dummy; iss >> dummy; } // skip unknown props
            }
        }
    }

    *cloud = PlaPC(std::move(pts));
    if (hasColor) cloud->setColors(std::move(colors));
    return true;
}

bool writePointCloudPly(const QString &path,
                        const PlaPC &pointCloud,
                        bool writeNormals,
                        QString *errorMessage)
{
    std::ofstream ofs(path.toStdString(), std::ios::binary);
    if (!ofs)
    {
        if (errorMessage) *errorMessage = QStringLiteral("无法创建文件: %1").arg(path);
        return false;
    }

    const bool hasNormalsOut = writeNormals && pointCloud.hasNormals();
    const bool hasColorsOut = pointCloud.hasColors();

    ofs << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << pointCloud.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n";
    if (hasNormalsOut)
        ofs << "property float nx\nproperty float ny\nproperty float nz\n";
    if (hasColorsOut)
        ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "end_header\n";

    for (size_t i = 0; i < pointCloud.size(); ++i)
    {
        float v[3];
        for (int d = 0; d < 3; ++d) v[d] = static_cast<float>(pointCloud.points()(static_cast<plamatrix::Index>(i), d));
        ofs.write(reinterpret_cast<const char*>(v), sizeof(float) * 3);

        if (hasNormalsOut)
        {
            float n[3];
            auto *nrm = pointCloud.normals();
            for (int d = 0; d < 3; ++d) n[d] = static_cast<float>(nrm->getValue(static_cast<plamatrix::Index>(i), d));
            ofs.write(reinterpret_cast<const char*>(n), sizeof(float) * 3);
        }

        if (hasColorsOut)
        {
            uint8_t c[3];
            auto *col = pointCloud.colors();
            for (int d = 0; d < 3; ++d) c[d] = col->getValue(static_cast<plamatrix::Index>(i), d);
            ofs.write(reinterpret_cast<const char*>(c), sizeof(uint8_t) * 3);
        }
    }

    return ofs.good();
}

// Helper: deep-copy PointCloud to shared_ptr
static std::shared_ptr<PlaPC> cloneCloud(const PlaPC &cloud)
{
    auto copy = std::make_shared<PlaPC>(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i)
        for (int d = 0; d < 3; ++d)
            copy->points()(static_cast<plamatrix::Index>(i), d) = cloud.points()(static_cast<plamatrix::Index>(i), d);
    if (cloud.hasColors())
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> c(cloud.size(), 3);
        for (size_t i = 0; i < cloud.size(); ++i)
            for (int d = 0; d < 3; ++d)
                c(static_cast<plamatrix::Index>(i), d) = cloud.colors()->getValue(static_cast<plamatrix::Index>(i), d);
        copy->setColors(std::move(c));
    }
    if (cloud.hasNormals())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> n(cloud.size(), 3);
        for (size_t i = 0; i < cloud.size(); ++i)
            for (int d = 0; d < 3; ++d)
                n(static_cast<plamatrix::Index>(i), d) = cloud.normals()->getValue(static_cast<plamatrix::Index>(i), d);
        copy->setNormals(std::move(n));
    }
    return copy;
}

// Color-preserving SOR filter
PlaPC sorFilter(const PlaPC &cloud, int k, float stdRatio)
{
    if (cloud.size() < static_cast<size_t>(k + 1))
    {
        return std::move(*cloneCloud(cloud));
    }

    auto pcCopy = cloneCloud(cloud);
    plapoint::search::KdTree<float, plamatrix::Device::CPU> tree;
    tree.setInputCloud(pcCopy);
    tree.build();

    const int N = static_cast<int>(cloud.size());
    std::vector<float> meanDists(N);
    for (int i = 0; i < N; ++i)
    {
        plamatrix::Vec3<float> q{cloud.points()(i, 0), cloud.points()(i, 1), cloud.points()(i, 2)};
        auto neighbors = tree.nearestKSearch(q, k + 1);
        float sum = 0; int cnt = 0;
        for (int nb : neighbors)
        {
            if (nb == i) continue;
            float dx = cloud.points()(i, 0) - cloud.points()(nb, 0);
            float dy = cloud.points()(i, 1) - cloud.points()(nb, 1);
            float dz = cloud.points()(i, 2) - cloud.points()(nb, 2);
            sum += std::sqrt(dx*dx + dy*dy + dz*dz);
            ++cnt;
        }
        meanDists[i] = cnt > 0 ? sum / static_cast<float>(cnt) : 1e9f;
    }

    double s = 0, s2 = 0;
    for (float d : meanDists) { s += d; s2 += static_cast<double>(d) * d; }
    float mean = static_cast<float>(s / N);
    float std = static_cast<float>(std::sqrt(s2 / N - static_cast<double>(mean) * mean));
    float thresh = mean + stdRatio * std;

    std::vector<int> keep;
    for (int i = 0; i < N; ++i)
        if (meanDists[i] <= thresh) keep.push_back(i);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> outPts(keep.size(), 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> outCol(keep.size(), 3);
    for (size_t i = 0; i < keep.size(); ++i)
    {
        int src = keep[i];
        for (int d = 0; d < 3; ++d) outPts(static_cast<plamatrix::Index>(i), d) = cloud.points()(src, d);
        if (cloud.hasColors())
        {
            auto *col = cloud.colors();
            for (int d = 0; d < 3; ++d) outCol(static_cast<plamatrix::Index>(i), d) = col->getValue(src, d);
        }
    }
    PlaPC out(std::move(outPts));
    if (cloud.hasColors()) out.setColors(std::move(outCol));
    return out;
}

// Color-preserving radius filter
PlaPC radiusFilter(const PlaPC &cloud, float radius, int minNeighbors)
{
    if (cloud.size() == 0)
    {
        PlaPC emptyOut(0);
        return emptyOut;
    }

    auto pcCopy = cloneCloud(cloud);
    plapoint::search::KdTree<float, plamatrix::Device::CPU> tree;
    tree.setInputCloud(pcCopy);
    tree.build();

    const int N = static_cast<int>(cloud.size());
    std::vector<int> keep;
    for (int i = 0; i < N; ++i)
    {
        plamatrix::Vec3<float> q{cloud.points()(i, 0), cloud.points()(i, 1), cloud.points()(i, 2)};
        int cnt = static_cast<int>(tree.radiusSearch(q, radius).size());
        if (cnt >= minNeighbors) keep.push_back(i);
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> outPts(keep.size(), 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> outCol(keep.size(), 3);
    for (size_t i = 0; i < keep.size(); ++i)
    {
        int src = keep[i];
        for (int d = 0; d < 3; ++d) outPts(static_cast<plamatrix::Index>(i), d) = cloud.points()(src, d);
        if (cloud.hasColors())
        {
            auto *col = cloud.colors();
            for (int d = 0; d < 3; ++d) outCol(static_cast<plamatrix::Index>(i), d) = col->getValue(src, d);
        }
    }
    PlaPC out(std::move(outPts));
    if (cloud.hasColors()) out.setColors(std::move(outCol));
    return out;
}

// Color-preserving voxel downsample
PlaPC voxelDownsample(const PlaPC &cloud, float leafSize)
{
    if (cloud.size() == 0 || leafSize <= 0) return std::move(*cloneCloud(cloud));

    using Key = std::tuple<int, int, int>;
    struct Accum { float sx = 0, sy = 0, sz = 0; uint32_t cr = 0, cg = 0, cb = 0; int count = 0; };
    std::map<Key, Accum> voxels;

    const bool hasCol = cloud.hasColors();
    auto *col = hasCol ? cloud.colors() : nullptr;

    for (size_t i = 0; i < cloud.size(); ++i)
    {
        Key key{
            static_cast<int>(std::floor(cloud.points()(static_cast<plamatrix::Index>(i), 0) / leafSize)),
            static_cast<int>(std::floor(cloud.points()(static_cast<plamatrix::Index>(i), 1) / leafSize)),
            static_cast<int>(std::floor(cloud.points()(static_cast<plamatrix::Index>(i), 2) / leafSize))
        };
        auto &acc = voxels[key];
        acc.sx += cloud.points()(static_cast<plamatrix::Index>(i), 0);
        acc.sy += cloud.points()(static_cast<plamatrix::Index>(i), 1);
        acc.sz += cloud.points()(static_cast<plamatrix::Index>(i), 2);
        if (hasCol)
        {
            acc.cr += col->getValue(static_cast<plamatrix::Index>(i), 0);
            acc.cg += col->getValue(static_cast<plamatrix::Index>(i), 1);
            acc.cb += col->getValue(static_cast<plamatrix::Index>(i), 2);
        }
        acc.count += 1;
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(voxels.size(), 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> outCol(voxels.size(), 3);
    int outIdx = 0;
    for (const auto &kv : voxels)
    {
        const auto &acc = kv.second;
        float inv = 1.0f / static_cast<float>(acc.count);
        pts(outIdx, 0) = acc.sx * inv;
        pts(outIdx, 1) = acc.sy * inv;
        pts(outIdx, 2) = acc.sz * inv;
        if (hasCol)
        {
            outCol(outIdx, 0) = static_cast<uint8_t>(static_cast<float>(acc.cr) * inv + 0.5f);
            outCol(outIdx, 1) = static_cast<uint8_t>(static_cast<float>(acc.cg) * inv + 0.5f);
            outCol(outIdx, 2) = static_cast<uint8_t>(static_cast<float>(acc.cb) * inv + 0.5f);
        }
        ++outIdx;
    }

    PlaPC out(std::move(pts));
    if (hasCol) out.setColors(std::move(outCol));
    return out;
}

} // namespace

ProjectDenseReconstructionManager::ProjectDenseReconstructionManager(ProjectManager *owner,
                                                                     ProjectData *projectData,
                                                                     QWidget *parentWidget,
                                                                     QObject *parent)
    : QObject(parent)
    , m_owner(owner)
    , m_projectData(projectData)
    , m_parentWidget(parentWidget)
{
}

bool ProjectDenseReconstructionManager::ensureProjectOpen(const QString &message,
                                                          const QString &title) const
{
    if (m_projectData && m_projectData->hasProject())
    {
        return true;
    }
    QMessageBox::warning(m_parentWidget, title, message);
    return false;
}

void ProjectDenseReconstructionManager::startEstimateDepthMapsAsync(const QJsonObject &settings)
{
    using namespace xjw::mvs;

    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("深度图估计")))
    {
        return;
    }

    const xjw::gui::project::DenseGenerationSettings request = denseGenerationSettingsFromJson(settings);
    const QJsonObject meta = m_projectData->metadata();
    const QJsonArray atArr = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (atArr.isEmpty())
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("深度图估计"),
                             QStringLiteral("未找到空三结果，请先执行空中三角测量。"));
        return;
    }

    const int realIdx = (request.atIndex >= 0 && request.atIndex < atArr.size()) ? request.atIndex : atArr.size() - 1;
    const QJsonObject atResult = atArr[realIdx].toObject();
    const QJsonArray selImgArr = atResult.value(QStringLiteral("selected_images")).toArray();
    const QJsonObject files = atResult.value(QStringLiteral("files")).toObject();
    const QString sparseXyz = files.value(QStringLiteral("sparse_cloud_xyz")).toString();

    QStringList selectedImages;
    for (const auto &value : selImgArr)
    {
        selectedImages.append(value.toString());
    }

    if (selectedImages.size() < 2)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("深度图估计"),
                             QStringLiteral("空三结果中影像数量不足（至少需要2张）。"));
        return;
    }

    const QString mvsOutDir = resolveProjectOutputDir(m_owner->currentProjectPath(), request.outputDir, QStringLiteral("mvs_output"));
    const QSet<int> existingIndices = collectExistingDepthFrameIndices(mvsOutDir, selectedImages.size());
    QSet<int> skipIndices;
    if (!existingIndices.isEmpty())
    {
        const ExistingDepthAction action = askExistingDepthAction(m_parentWidget,
                                                                  existingIndices.size(),
                                                                  selectedImages.size(),
                                                                  mvsOutDir);
        if (action == ExistingDepthAction::Cancel)
        {
            return;
        }

        if (action == ExistingDepthAction::Overwrite)
        {
            removeDepthArtifactsForIndices(mvsOutDir, existingIndices);
            LOG_INFO(QStringLiteral("[MVS] 用户选择覆盖重算，已清理已有深度图 %1 帧").arg(existingIndices.size()));
        }
        else if (action == ExistingDepthAction::ContinueMissing)
        {
            skipIndices = existingIndices;
            upsertExistingDepthRecords(m_projectData, selectedImages, sparseXyz, mvsOutDir, skipIndices);

            if (skipIndices.size() >= selectedImages.size())
            {
                QMessageBox::information(m_parentWidget,
                                         QStringLiteral("深度图估计"),
                                         QStringLiteral("当前影像的深度图已全部存在，无需继续生成。"));
                return;
            }

            LOG_INFO(QStringLiteral("[MVS] 用户选择续跑：已存在 %1 帧，将继续生成剩余 %2 帧")
                         .arg(skipIndices.size())
                         .arg(selectedImages.size() - skipIndices.size()));
        }
    }

    bool allCams = false;
    const QMap<QString, xjw::Camera> camMap = m_owner->getCamerasForImages(selectedImages, &allCams);
    if (!allCams)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("深度图估计"),
                             QStringLiteral("部分影像缺少相机参数，无法执行深度图估计。"));
        return;
    }

    std::vector<CameraView> views;
    views.reserve(selectedImages.size());
    for (const QString &imgPath : selectedImages)
    {
        CameraView view;
        view.imagePath = imgPath.toStdString();
        view.camera = camMap.value(imgPath);
        views.push_back(std::move(view));
    }

    DepthGenConfig genCfg = buildDepthGenConfig(request, static_cast<int>(views.size()));
    genCfg.runFusion = false;
    genCfg.saveIntermediateDepthMaps = true;
    genCfg.intermediateDir = mvsOutDir.toStdString();

    auto *gen = new DepthMapGenerator(this);
    gen->setViews(views);
    if (!skipIndices.isEmpty())
    {
        std::vector<int> skipVector;
        skipVector.reserve(static_cast<size_t>(skipIndices.size()));
        for (const int index : skipIndices)
        {
            skipVector.push_back(index);
        }
        std::sort(skipVector.begin(), skipVector.end());
        gen->setSkippedFrameIndices(skipVector);
    }
    gen->setConfig(genCfg);
    gen->setOutputDir(mvsOutDir.toStdString());

    QObject::connect(gen, &DepthMapGenerator::progressChanged, this,
                     [this](const QString &stage, float ratio) {
        emit mvsProgressChanged(stage, static_cast<int>(ratio * 100));
    });
    connect(gen, &DepthMapGenerator::errorOccurred, this, [](const QString &msg) {
        qWarning() << "[MVS] 错误:" << msg;
    });
    connect(gen, &DepthMapGenerator::depthMapSaved, this,
            [this, sparseXyz, mvsOutDir](const QString &pngPath, int width, int height, const QString &refImage) {
        QJsonObject depthResult = makeDepthResultRecord(utcNowIso(), pngPath, width, height, sparseXyz, refImage);
        depthResult[QStringLiteral("raw_depth_path")] = rawDepthStoragePath(pngPath);
        depthResult[QStringLiteral("raw_confidence_path")] = rawConfidenceStoragePath(pngPath);
        depthResult[QStringLiteral("mvs_output_dir")] = mvsOutDir;
        upsertProjectRecordByPath(m_projectData,
                                  QStringLiteral("depth_map_results"),
                                  QStringLiteral("depth_png"),
                                  depthResult);
    });
    connect(gen, &DepthMapGenerator::finished, this, [this](bool success) {
        emit mvsProgressFinished(success);
        QMessageBox::information(m_parentWidget,
                                 QStringLiteral("深度图估计"),
                                 success ? QStringLiteral("深度图估计完成。")
                                         : QStringLiteral("深度图估计失败或被取消。"));
        if (m_activeMvsGenerator)
        {
            m_activeMvsGenerator->deleteLater();
        }
    });

    m_activeMvsGenerator = gen;
    emit mvsProgressChanged(QStringLiteral("正在加载稀疏点云..."), 0);
    (void)QtConcurrent::run([gen, sparseXyz, views]() {
        SparseCloud sparse;
        if (!sparseXyz.isEmpty() && QFile::exists(sparseXyz))
        {
            SparseCloudPreprocessor pp;
            PreprocessResult ppRes;
            std::string ppErr;
            if (pp.run(sparseXyz.toStdString(), views, ppRes, &ppErr))
            {
                sparse = ppRes.cloud;
            }
        }
        gen->setSparseCloud(sparse);
        QMetaObject::invokeMethod(gen, "start", Qt::QueuedConnection);
    });
}

void ProjectDenseReconstructionManager::startFuseDepthMapsAsync(const QJsonObject &settings)
{
    using namespace xjw::mvs;

    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("深度图融合")))
    {
        return;
    }

    const xjw::gui::project::DenseGenerationSettings request = denseGenerationSettingsFromJson(settings);
    const bool keepColor = settings.value(QStringLiteral("keepColor")).toBool(true);
    const bool keepNormals = settings.value(QStringLiteral("keepNormals")).toBool(true);

    const auto storedFramesResult = collectLatestStoredDepthFrames(m_projectData->metadata());
    const std::vector<StoredDepthFrameRecord> &storedFrames = storedFramesResult.frames;
    if (storedFrames.size() < 2)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("深度图融合"),
                             storedFrames.empty()
                                 ? storedFramesResult.status.errorMessage
                                 : QStringLiteral("可用深度图数量不足（至少需要2张）。"));
        return;
    }

    QStringList imagePaths;
    for (const auto &frame : storedFrames)
    {
        imagePaths.append(frame.refImage);
    }

    bool allCams = false;
    const QMap<QString, xjw::Camera> camMap = m_owner->getCamerasForImages(imagePaths, &allCams);
    if (!allCams)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("深度图融合"),
                             QStringLiteral("部分深度图对应影像缺少相机参数，无法执行融合。"));
        return;
    }

    const QString outputDir = request.outputDir.trimmed().isEmpty()
        ? storedFramesResult.batchDir
        : resolveProjectOutputDir(m_owner->currentProjectPath(), request.outputDir, QStringLiteral("mvs_output"));
    const QString outputPly = QDir(outputDir).filePath(QStringLiteral("dense_cloud.ply"));

    emit mvsProgressChanged(QStringLiteral("正在加载深度图批次..."), 0);

    const bool pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
    (void)QtConcurrent::run([this, storedFrames, camMap, request, keepColor, keepNormals, outputPly, pipelineMode]() {
        std::vector<FusionFrameInput> frames;
        frames.reserve(storedFrames.size());
        for (const auto &stored : storedFrames)
        {
            const xjw::Camera camera = camMap.value(stored.refImage);
            auto frameResult = buildStoredFusionFrame(stored,
                                                      camera,
                                                      request.fusionMinConfidence,
                                                      static_cast<int>(storedFrames.size()));
            if (!frameResult.status.ok)
            {
                const QString loadError = frameResult.status.errorMessage;
                QMetaObject::invokeMethod(this, [this, loadError]() {
                    emit mvsProgressFinished(false);
                    QMessageBox::warning(m_parentWidget, QStringLiteral("深度图融合"), loadError);
                }, Qt::QueuedConnection);
                return;
            }
            frames.push_back(std::move(frameResult.frame));
        }

        StereoFusionConfig fusionCfg;
        fusionCfg.minNumPixels = std::max(1, request.minConsistentViews);
        fusionCfg.maxReprojError = request.depthConsistency;
        fusionCfg.maxDepthError = 0.05f;
        fusionCfg.checkNumImages = std::min(50, static_cast<int>(frames.size()));
        if (static_cast<int>(frames.size()) <= 2)
        {
            fusionCfg.minNumPixels = 1;
            fusionCfg.maxDepthError = std::max(fusionCfg.maxDepthError, 0.08f);
            fusionCfg.maxReprojError = std::max(fusionCfg.maxReprojError, 3.0f);
        }

        DepthMapFusion fusion(fusionCfg);
        std::vector<FusedPoint> fusedPoints;
        std::string fuseErr;
        const bool fuseOk = fusion.fuse(frames,
                                        fusedPoints,
                                        [this](const std::string &stage, float ratio) {
                                            QMetaObject::invokeMethod(this, [this, stage, ratio]() {
                                                emit mvsProgressChanged(QString::fromStdString(stage),
                                                                        static_cast<int>(ratio * 100.0f));
                                            }, Qt::QueuedConnection);
                                        },
                                        &fuseErr);
        if (!fuseOk)
        {
            const QString err = QString::fromStdString(fuseErr);
            QMetaObject::invokeMethod(this, [this, err]() {
                emit mvsProgressFinished(false);
                QMessageBox::warning(m_parentWidget,
                                     QStringLiteral("深度图融合"),
                                     QStringLiteral("深度图融合失败：%1").arg(err));
            }, Qt::QueuedConnection);
            return;
        }

        const PlaPC pointCloud = fusedPointsToPointCloud(fusedPoints, keepColor, keepNormals);
        QString saveError;
        if (!writePointCloudPly(outputPly, pointCloud, keepNormals, &saveError))
        {
            QMetaObject::invokeMethod(this, [this, saveError]() {
                emit mvsProgressFinished(false);
                QMessageBox::warning(m_parentWidget,
                                     QStringLiteral("深度图融合"),
                                     QStringLiteral("保存密集点云失败：%1").arg(saveError));
            }, Qt::QueuedConnection);
            return;
        }

        const int pointCount = static_cast<int>(pointCloud.size());
        QMetaObject::invokeMethod(this, [this, outputPly, pointCount, pipelineMode]() {
            upsertProjectRecordByPath(m_projectData,
                                      QStringLiteral("dense_cloud_results"),
                                      QStringLiteral("dense_cloud_xyz"),
                                      makeDenseResultRecord(utcNowIso(), outputPly, pointCount));
            emit mvsProgressFinished(true);
            if (!pipelineMode)
            {
                QMessageBox::information(m_parentWidget,
                                         QStringLiteral("深度图融合"),
                                         QStringLiteral("密集点云生成完成，共 %1 个点。").arg(pointCount));
            }
        }, Qt::QueuedConnection);
    });
}

void ProjectDenseReconstructionManager::startGenerateDenseCloudAsync(const QJsonObject &settings)
{
    using namespace xjw::mvs;

    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("稠密重建")))
    {
        return;
    }

    const xjw::gui::project::DenseGenerationSettings request = denseGenerationSettingsFromJson(settings);
    const QJsonObject meta = m_projectData->metadata();
    const QJsonArray atArr = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (atArr.isEmpty())
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("稠密重建"),
                             QStringLiteral("未找到空三结果，请先执行空中三角测量。"));
        return;
    }

    const int realIdx = (request.atIndex >= 0 && request.atIndex < atArr.size()) ? request.atIndex : atArr.size() - 1;
    const QJsonObject atResult = atArr[realIdx].toObject();
    const QJsonArray selImgArr = atResult.value(QStringLiteral("selected_images")).toArray();
    const QJsonObject files = atResult.value(QStringLiteral("files")).toObject();
    const QString sparseXyz = files.value(QStringLiteral("sparse_cloud_xyz")).toString();

    QStringList selectedImages;
    for (const auto &value : selImgArr)
    {
        selectedImages.append(value.toString());
    }
    if (selectedImages.size() < 2)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("稠密重建"),
                             QStringLiteral("空三结果中影像数量不足（至少需要2张）。"));
        return;
    }

    bool allCams = false;
    const QMap<QString, xjw::Camera> camMap = m_owner->getCamerasForImages(selectedImages, &allCams);
    if (!allCams)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("稠密重建"),
                             QStringLiteral("部分影像缺少相机参数，无法执行稠密重建。"));
        return;
    }

    std::vector<CameraView> views;
    for (const QString &imgPath : selectedImages)
    {
        CameraView view;
        view.imagePath = imgPath.toStdString();
        view.camera = camMap.value(imgPath);
        view.imageWidth = 0;
        view.imageHeight = 0;
        views.push_back(std::move(view));
    }

    DepthGenConfig genCfg = buildDepthGenConfig(request, static_cast<int>(views.size()));
    genCfg.saveIntermediateDepthMaps = true;
    const QString mvsOutDir = resolveProjectOutputDir(m_owner->currentProjectPath(), request.outputDir, QStringLiteral("mvs_output"));
    genCfg.intermediateDir = mvsOutDir.toStdString();

    const QSet<int> existingIndices = collectExistingDepthFrameIndices(mvsOutDir, selectedImages.size());
    QSet<int> skipIndices;
    bool continueMissingMode = false;
    if (!existingIndices.isEmpty())
    {
        ExistingDepthAction action;
        if (request.pipelineMode)
        {
            // 流水线模式：静默续跑，不弹对话框
            action = ExistingDepthAction::ContinueMissing;
            LOG_INFO(QStringLiteral("[MVS][pipeline] 检测到已有深度图 %1 帧，自动续跑").arg(existingIndices.size()));
        }
        else
        {
            action = askExistingDepthAction(m_parentWidget,
                                            existingIndices.size(),
                                            selectedImages.size(),
                                            mvsOutDir);
        }
        if (action == ExistingDepthAction::Cancel)
        {
            return;
        }

        if (action == ExistingDepthAction::Overwrite)
        {
            removeDepthArtifactsForIndices(mvsOutDir, existingIndices);
            LOG_INFO(QStringLiteral("[MVS] 用户选择覆盖重算，已清理已有深度图 %1 帧").arg(existingIndices.size()));
        }
        else if (action == ExistingDepthAction::ContinueMissing)
        {
            continueMissingMode = true;
            skipIndices = existingIndices;
            upsertExistingDepthRecords(m_projectData, selectedImages, sparseXyz, mvsOutDir, skipIndices);

            if (skipIndices.size() >= selectedImages.size())
            {
                LOG_INFO(QStringLiteral("[MVS] 深度图已全部存在，直接进入融合阶段"));
                startFuseDepthMapsAsync(settings);
                return;
            }

            genCfg.runFusion = false;
            LOG_INFO(QStringLiteral("[MVS] 用户选择续跑：已存在 %1 帧，将继续生成剩余 %2 帧；完成后自动执行融合")
                         .arg(skipIndices.size())
                         .arg(selectedImages.size() - skipIndices.size()));
        }
    }

    auto *gen = new DepthMapGenerator(this);
    gen->setViews(views);
    if (!skipIndices.isEmpty())
    {
        std::vector<int> skipVector;
        skipVector.reserve(static_cast<size_t>(skipIndices.size()));
        for (const int index : skipIndices)
        {
            skipVector.push_back(index);
        }
        std::sort(skipVector.begin(), skipVector.end());
        gen->setSkippedFrameIndices(skipVector);
    }
    gen->setConfig(genCfg);
    gen->setOutputDir(mvsOutDir.toStdString());

    QObject::connect(gen, &DepthMapGenerator::progressChanged, this,
                     [this](const QString &stage, float ratio) {
        emit mvsProgressChanged(stage, static_cast<int>(ratio * 100));
    });
    connect(gen, &DepthMapGenerator::errorOccurred, this, [](const QString &msg) {
        qWarning() << "[MVS] 错误:" << msg;
    });
    connect(gen, &DepthMapGenerator::depthMapSaved, this,
            [this, sparseXyz, mvsOutDir](const QString &pngPath, int width, int height, const QString &refImage) {
        QJsonObject depthResult = makeDepthResultRecord(utcNowIso(), pngPath, width, height, sparseXyz, refImage);
        depthResult[QStringLiteral("raw_depth_path")] = rawDepthStoragePath(pngPath);
        depthResult[QStringLiteral("raw_confidence_path")] = rawConfidenceStoragePath(pngPath);
        depthResult[QStringLiteral("mvs_output_dir")] = mvsOutDir;
        upsertProjectRecordByPath(m_projectData,
                                  QStringLiteral("depth_map_results"),
                                  QStringLiteral("depth_png"),
                                  depthResult);
    });
    connect(gen, &DepthMapGenerator::pointCloudReady, this,
            [this, mvsOutDir](const std::vector<DensePoint> &cloud) {
        if ((cloud.size() == 0))
        {
            return;
        }
        const QString plyPath = mvsOutDir + QStringLiteral("/dense_cloud.ply");
        const PlaPC pointCloud = densePointsToPointCloud(cloud);
        QString saveErr;
        if (writePointCloudPly(plyPath, pointCloud, false, &saveErr))
        {
            upsertProjectRecordByPath(m_projectData,
                                      QStringLiteral("dense_cloud_results"),
                                      QStringLiteral("dense_cloud_xyz"),
                                      makeDenseResultRecord(utcNowIso(), plyPath, static_cast<int>(cloud.size())));
        }
        else
        {
            qWarning() << "[MVS] 保存失败:" << saveErr;
        }
    });
    connect(gen, &DepthMapGenerator::finished, this, [this, settings, continueMissingMode](bool success) {
        emit mvsProgressFinished(success);

        if (m_activeMvsGenerator)
        {
            m_activeMvsGenerator->deleteLater();
            m_activeMvsGenerator = nullptr;
        }

        const bool pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);

        if (success && continueMissingMode)
        {
            if (!pipelineMode)
            {
                QMessageBox::information(m_parentWidget,
                                         QStringLiteral("稠密重建"),
                                         QStringLiteral("缺失深度图补齐完成，正在开始融合。"));
            }
            QMetaObject::invokeMethod(this,
                                      [this, settings]() {
                                          startFuseDepthMapsAsync(settings);
                                      },
                                      Qt::QueuedConnection);
        }
        else if (!pipelineMode)
        {
            QMessageBox::information(m_parentWidget,
                                     QStringLiteral("稠密重建"),
                                     success ? QStringLiteral("稠密点云生成完成。")
                                             : QStringLiteral("稠密点云生成失败或被取消。"));
        }
    });

    m_activeMvsGenerator = gen;
    emit mvsProgressChanged(QStringLiteral("正在加载稀疏点云..."), 0);
    (void)QtConcurrent::run([gen, sparseXyz, views]() {
        SparseCloud sparse;
        if (!sparseXyz.isEmpty() && QFile::exists(sparseXyz))
        {
            SparseCloudPreprocessor pp;
            PreprocessResult ppRes;
            std::string ppErr;
            if (pp.run(sparseXyz.toStdString(), views, ppRes, &ppErr))
            {
                sparse = ppRes.cloud;
            }
        }
        gen->setSparseCloud(sparse);
        QMetaObject::invokeMethod(gen, "start", Qt::QueuedConnection);
    });
}

void ProjectDenseReconstructionManager::startDenseCloudRefineAsync(const QJsonObject &settings)
{
    if (!ensureProjectOpen(QStringLiteral("请先打开一个项目。"), QStringLiteral("密集点云后处理")))
    {
        return;
    }

    QString inputPly;
    QString denseError;
    if (!resolveLatestDenseCloudPath(m_projectData, &inputPly, &denseError))
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("密集点云后处理"),
                             QStringLiteral("%1，请先执行深度图融合生成密集点云。").arg(denseError));
        return;
    }

    const xjw::gui::project::DenseRefineSettings request = denseRefineSettingsFromJson(settings);
    const QString outDir = QFileInfo(inputPly).absolutePath();
    const QString ts = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const QString outputPly = outDir + QStringLiteral("/dense_cloud_refined_%1.ply").arg(ts);

    emit mvsProgressChanged(QStringLiteral("正在加载密集点云..."), 0);
    (void)QtConcurrent::run([this, inputPly, outputPly, request]() {
        PlaPC cloud;
        QString loadErr;
        if (!readPointCloudPly(inputPly, &cloud, &loadErr))
        {
            QMetaObject::invokeMethod(this, [this, loadErr]() {
                emit mvsProgressFinished(false);
                QMessageBox::warning(m_parentWidget,
                                     QStringLiteral("密集点云后处理"),
                                     QStringLiteral("加载点云失败：%1").arg(loadErr));
            }, Qt::QueuedConnection);
            return;
        }

        if (request.sorEnabled)
        {
            QMetaObject::invokeMethod(this, [this]() {
                emit mvsProgressChanged(QStringLiteral("统计离群点移除 (SOR)..."), 20);
            }, Qt::QueuedConnection);

            const auto beforeSor = cloud.size();
            cloud = sorFilter(cloud, request.sorK, static_cast<float>(request.sorStdDev));

            if (cloud.size() > 64)
            {
                // Compute bounds
                float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
                float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
                for (size_t i = 0; i < cloud.size(); ++i)
                {
                    float x = cloud.points()(static_cast<plamatrix::Index>(i), 0);
                    float y = cloud.points()(static_cast<plamatrix::Index>(i), 1);
                    float z = cloud.points()(static_cast<plamatrix::Index>(i), 2);
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (z < minZ) minZ = z;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                    if (z > maxZ) maxZ = z;
                }

                const double dx = static_cast<double>(maxX - minX);
                const double dy = static_cast<double>(maxY - minY);
                const double dz = static_cast<double>(maxZ - minZ);
                const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double volume = std::max(dx * dy * dz, 1e-12);
                const double density = static_cast<double>(cloud.size()) / volume;
                const int radiusMinNeighbors = std::clamp(request.sorK / 2, 6, 32);
                double adaptiveRadius = std::cbrt(std::max(1.0, static_cast<double>(radiusMinNeighbors))
                                                  / std::max(density, 1e-12));
                adaptiveRadius *= 1.2;
                if (diag > 1e-9)
                {
                    const double radiusMin = std::max(diag * 0.001, 1e-4);
                    const double radiusMax = std::max(radiusMin * 2.0, diag * 0.08);
                    adaptiveRadius = std::clamp(adaptiveRadius, radiusMin, radiusMax);
                }

                QMetaObject::invokeMethod(this, [this]() {
                    emit mvsProgressChanged(QStringLiteral("半径离群点移除..."), 35);
                }, Qt::QueuedConnection);
                cloud = radiusFilter(cloud, static_cast<float>(adaptiveRadius), radiusMinNeighbors);

                const auto afterRadius = cloud.size();
                const bool largeCloud = beforeSor > 200000;
                const bool weakRemoval = (beforeSor > 0)
                    && (static_cast<double>(beforeSor - afterRadius) / static_cast<double>(beforeSor) < 0.02);
                if (largeCloud && weakRemoval)
                {
                    const double stricterStdDev = std::clamp(request.sorStdDev - 0.3, 0.8, request.sorStdDev);
                    const int stricterK = std::clamp(request.sorK + 6, request.sorK, 96);
                    QMetaObject::invokeMethod(this, [this]() {
                        emit mvsProgressChanged(QStringLiteral("离群点二次清理..."), 42);
                    }, Qt::QueuedConnection);
                    cloud = sorFilter(cloud, stricterK, static_cast<float>(stricterStdDev));
                }
            }
        }
        if (request.voxelEnabled && request.voxelSize > 0.0)
        {
            QMetaObject::invokeMethod(this, [this]() {
                emit mvsProgressChanged(QStringLiteral("体素下采样..."), 50);
            }, Qt::QueuedConnection);
            cloud = voxelDownsample(cloud, static_cast<float>(request.voxelSize));
        }
        if (request.normalsEnabled)
        {
            QMetaObject::invokeMethod(this, [this]() {
                emit mvsProgressChanged(QStringLiteral("估计法向量..."), 70);
            }, Qt::QueuedConnection);

            auto pcPtr = cloneCloud(cloud);
            plapoint::search::KdTree<float, plamatrix::Device::CPU> normTree;
            normTree.setInputCloud(pcPtr);
            normTree.build();

            plapoint::NormalEstimation<float, plamatrix::Device::CPU> ne;
            ne.setInputCloud(pcPtr);
            auto treePtr = std::make_shared<plapoint::search::KdTree<float, plamatrix::Device::CPU>>(std::move(normTree));
            ne.setSearchMethod(treePtr);
            ne.setKSearch(request.normalK);
            auto normals = ne.compute();
            cloud.setNormals(std::move(normals));
        }

        QMetaObject::invokeMethod(this, [this]() {
            emit mvsProgressChanged(QStringLiteral("保存后处理结果..."), 90);
        }, Qt::QueuedConnection);

        const bool writeNormalsOut = request.normalsEnabled && cloud.hasNormals();
        QString saveErr;
        if (!writePointCloudPly(outputPly, cloud, writeNormalsOut, &saveErr))
        {
            QMetaObject::invokeMethod(this, [this, saveErr]() {
                emit mvsProgressFinished(false);
                QMessageBox::warning(m_parentWidget,
                                     QStringLiteral("密集点云后处理"),
                                     QStringLiteral("保存点云失败：%1").arg(saveErr));
            }, Qt::QueuedConnection);
            return;
        }

        const int pointCount = static_cast<int>(cloud.size());
        QMetaObject::invokeMethod(this, [this, outputPly, pointCount]() {
            upsertProjectRecordByPath(m_projectData,
                                      QStringLiteral("dense_cloud_results"),
                                      QStringLiteral("dense_cloud_xyz"),
                                      makeDenseResultRecord(utcNowIso(), outputPly, pointCount));
            emit mvsProgressFinished(true);
            QMessageBox::information(m_parentWidget,
                                     QStringLiteral("密集点云后处理"),
                                     QStringLiteral("后处理完成，共 %1 个点。").arg(pointCount));
        }, Qt::QueuedConnection);
    });
}

void ProjectDenseReconstructionManager::cancelMvs()
{
    if (!m_activeMvsGenerator)
    {
        return;
    }
    auto *gen = qobject_cast<xjw::mvs::DepthMapGenerator *>(m_activeMvsGenerator.data());
    if (gen)
    {
        gen->requestCancel();
        qDebug() << "[MVS] 已请求取消";
    }
}
