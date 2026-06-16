#include "ProjectResultRecords.h"

#include <QFileInfo>
#include <QJsonArray>

namespace xjw::gui::project {

QJsonObject makeDepthResultRecord(const QString &createdAt,
                                  const QString &depthPng,
                                  int gridWidth,
                                  int gridHeight,
                                  const QString &sourceSparseCloud,
                                  const QString &refImage)
{
    QJsonObject rec;
    rec[QStringLiteral("created_at")] = createdAt;
    if (!sourceSparseCloud.isEmpty()) {
        rec[QStringLiteral("source_sparse_cloud")] = sourceSparseCloud;
    }
    rec[QStringLiteral("depth_png")] = depthPng;
    rec[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
    rec[QStringLiteral("grid_width")] = gridWidth;
    rec[QStringLiteral("grid_height")] = gridHeight;
    if (!refImage.isEmpty()) {
        rec[QStringLiteral("ref_image")] = refImage;
    }
    return rec;
}

QJsonObject makeDenseResultRecord(const QString &createdAt,
                                  const QString &densePath,
                                  int pointCount,
                                  const QString &sourceSparseCloud)
{
    QJsonObject rec;
    rec[QStringLiteral("created_at")] = createdAt;
    if (!sourceSparseCloud.isEmpty()) {
        rec[QStringLiteral("source_sparse_cloud")] = sourceSparseCloud;
    }
    rec[QStringLiteral("dense_cloud_xyz")] = densePath;
    rec[QStringLiteral("point_count")] = pointCount;
    return rec;
}

QJsonObject makeModelResultRecord(const QString &createdAt,
                                  const QString &sourceTag,
                                  const QString &modelPly,
                                  int vertexCount,
                                  int faceCount,
                                  const QString &sourceSparseCloud,
                                  const QString &sourceDenseCloud,
                                  const QString &modelDenseCloud)
{
    QJsonObject rec;
    rec[QStringLiteral("created_at")] = createdAt;
    rec[QStringLiteral("source")] = sourceTag;
    if (!sourceSparseCloud.isEmpty()) {
        rec[QStringLiteral("source_sparse_cloud")] = sourceSparseCloud;
    }
    if (!sourceDenseCloud.isEmpty()) {
        rec[QStringLiteral("source_dense_cloud")] = sourceDenseCloud;
    }
    if (!modelDenseCloud.isEmpty()) {
        rec[QStringLiteral("model_dense_cloud")] = modelDenseCloud;
    }
    rec[QStringLiteral("vertex_count")] = vertexCount;
    rec[QStringLiteral("face_count")] = faceCount;
    rec[QStringLiteral("model_ply")] = modelPly;
    return rec;
}

QJsonObject makeDemResultRecord(const QString &createdAt,
                                const QString &outputDir,
                                const QString &sourceSparseCloud,
                                const QString &demTif,
                                const QString &demType,
                                double demResolution,
                                const QString &tSrs,
                                const QStringList &images)
{
    QJsonObject rec;
    rec[QStringLiteral("created_at")] = createdAt;
    rec[QStringLiteral("output_dir")] = outputDir;
    rec[QStringLiteral("source_sparse_cloud")] = sourceSparseCloud;
    rec[QStringLiteral("dem_tif")] = demTif;
    rec[QStringLiteral("dem_type")] = demType;
    rec[QStringLiteral("dem_resolution")] = demResolution;
    if (!tSrs.isEmpty()) {
        rec[QStringLiteral("t_srs")] = tSrs;
    }
    if (!images.isEmpty()) {
        rec[QStringLiteral("images")] = QJsonArray::fromStringList(images);
    }
    return rec;
}

QJsonObject makeOrthoResultRecord(const QString &createdAt,
                                  const QString &demPath,
                                  const QString &outputPath,
                                  int sourceImageCount,
                                  const QStringList &images,
                                  bool includeResolution,
                                  double resolution)
{
    QJsonObject rec;
    rec[QStringLiteral("created_at")] = createdAt;
    rec[QStringLiteral("dem_path")] = demPath;
    rec[QStringLiteral("output_path")] = outputPath;
    rec[QStringLiteral("source_image_count")] = sourceImageCount;
    rec[QStringLiteral("images")] = QJsonArray::fromStringList(images);
    if (includeResolution) {
        rec[QStringLiteral("resolution")] = resolution;
    }
    return rec;
}

QJsonObject makeAtResultRecord(const QString &createdAt,
                               const QString &sparseCloudPath,
                               int sparsePointCount,
                               const QStringList &selectedImages,
                               const QString &outputDir,
                               const QJsonObject &extraRecord)
{
    QJsonObject files;
    files[QStringLiteral("sparse_cloud_xyz")] = sparseCloudPath;
    const QJsonObject extraFiles = extraRecord.value(QStringLiteral("files")).toObject();
    for (auto it = extraFiles.begin(); it != extraFiles.end(); ++it)
    {
        files[it.key()] = it.value();
    }

    QJsonObject rec;
    rec[QStringLiteral("created_at")] = createdAt;
    rec[QStringLiteral("output_dir")] = outputDir;
    rec[QStringLiteral("sparse_point_count")] = sparsePointCount;
    rec[QStringLiteral("selected_images")] = QJsonArray::fromStringList(selectedImages);
    rec[QStringLiteral("files")] = files;
    for (auto it = extraRecord.begin(); it != extraRecord.end(); ++it)
    {
        if (it.key() == QLatin1String("files"))
        {
            continue;
        }
        rec[it.key()] = it.value();
    }
    return rec;
}

void upsertMetaArrayRecordByPath(QJsonObject *meta,
                                 const QString &arrayKey,
                                 const QString &pathKey,
                                 const QJsonObject &record)
{
    if (!meta) return;
    const QString targetPath = record.value(pathKey).toString();
    const QString targetBaseName = QFileInfo(targetPath).fileName();
    const QJsonArray src = meta->value(arrayKey).toArray();
    QJsonArray deduped;
    for (const QJsonValue &value : src)
    {
        const QString existingPath = value.toObject().value(pathKey).toString();
        if (existingPath == targetPath)
        {
            continue;
        }
        if (!targetBaseName.isEmpty() && QFileInfo(existingPath).fileName() == targetBaseName)
        {
            continue;
        }
        deduped.append(value);
    }
    deduped.append(record);
    (*meta)[arrayKey] = deduped;
}

void replaceMetaArrayWithLatest(QJsonObject *meta,
                                const QString &arrayKey,
                                const QJsonObject &record)
{
    if (!meta)
    {
        return;
    }
    QJsonArray arr;
    arr.append(record);
    (*meta)[arrayKey] = arr;
}

void upsertMetaArrayRecordByIndex(QJsonObject *meta,
                                  const QString &arrayKey,
                                  const QJsonObject &record,
                                  int replaceIndex)
{
    if (!meta)
    {
        return;
    }

    QJsonArray array = meta->value(arrayKey).toArray();
    if (replaceIndex >= 0 && replaceIndex < array.size())
    {
        array[replaceIndex] = record;
    }
    else
    {
        array.append(record);
    }
    (*meta)[arrayKey] = array;
}

} // namespace xjw::gui::project
