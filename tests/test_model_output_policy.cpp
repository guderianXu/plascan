#include <gtest/gtest.h>

#include "ModelOutputPolicy.h"
#include "ModelWorkflowService.h"
#include "ProjectModelResultPolicy.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QTextStream>

#include <cmath>
#include <atomic>

namespace
{

void writeFile(const QString &path)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_GT(file.write("artifact"), 0);
}

void writeJsonFile(const QString &path, const QJsonObject &object)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    const QByteArray bytes = QJsonDocument(object).toJson(
        QJsonDocument::Compact);
    ASSERT_EQ(file.write(bytes), bytes.size());
}

QString writeDenseGridPointCloud(const QString &root)
{
    const QString path = QDir(root).filePath(QStringLiteral("dense_grid.ply"));
    if (!QDir().mkpath(QFileInfo(path).absolutePath()))
    {
        ADD_FAILURE() << "Unable to create point-cloud directory";
        return {};
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        ADD_FAILURE() << file.errorString().toStdString();
        return {};
    }

    constexpr int gridSize = 24;
    QTextStream stream(&file);
    stream << "ply\n"
           << "format ascii 1.0\n"
           << "element vertex " << gridSize * gridSize << "\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "end_header\n";
    for (int y = 0; y < gridSize; ++y)
    {
        for (int x = 0; x < gridSize; ++x)
        {
            const double px = static_cast<double>(x) / (gridSize - 1);
            const double py = static_cast<double>(y) / (gridSize - 1);
            const double pz = 0.04 * std::sin(px * 8.0)
                + 0.03 * std::cos(py * 6.0);
            stream << px << ' ' << py << ' ' << pz << '\n';
        }
    }
    stream.flush();
    file.close();
    EXPECT_GT(QFileInfo(path).size(), 0);
    return path;
}

QString writeColoredTriangleMesh(const QString &root)
{
    const QString path = QDir(root).filePath(
        QStringLiteral("colored_triangle.ply"));
    if (!QDir().mkpath(QFileInfo(path).absolutePath()))
    {
        ADD_FAILURE() << "Unable to create mesh directory";
        return {};
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        ADD_FAILURE() << file.errorString().toStdString();
        return {};
    }
    QTextStream stream(&file);
    stream << "ply\n"
           << "format ascii 1.0\n"
           << "element vertex 3\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property uchar red\n"
           << "property uchar green\n"
           << "property uchar blue\n"
           << "element face 1\n"
           << "property list uchar int vertex_indices\n"
           << "end_header\n"
           << "-0.5 -0.5 0 255 0 0\n"
           << "0.5 -0.5 0 0 255 0\n"
           << "0 0.5 0 0 0 255\n"
           << "3 0 1 2\n";
    stream.flush();
    file.close();
    return path;
}

QJsonObject completedRecord(const QString &runId,
                            const QString &meshPath,
                            const QString &diagnosticsPath)
{
    const QString runDirectory = QFileInfo(diagnosticsPath).absolutePath();
    QJsonObject record{
        {QStringLiteral("model_run_id"), runId},
        {QStringLiteral("model_run_directory"), runDirectory},
        {QStringLiteral("model_artifact_directory"),
         QFileInfo(meshPath).absolutePath()},
        {QStringLiteral("model_ply"), meshPath},
        {QStringLiteral("mesh_ply"), meshPath},
        {QStringLiteral("final_model_path"), meshPath},
        {QStringLiteral("model_diagnostics_path"), diagnosticsPath}
    };
    QJsonObject diagnostics = record;
    diagnostics[QStringLiteral("schema_version")] = 1;
    diagnostics[QStringLiteral("diagnostics_type")] =
        QStringLiteral("model");
    diagnostics[QStringLiteral("ok")] = true;
    writeJsonFile(diagnosticsPath, diagnostics);
    return record;
}

QJsonObject createCompletedModelRun(const QString &baseOutputRoot,
                                    const QString &requestedRunId,
                                    bool coloredMesh = false)
{
    QString runId;
    QString runDirectory;
    QString error;
    if (!xjw::mesh::workflow::createModelRunOutputDirectory(
            baseOutputRoot,
            requestedRunId,
            &runId,
            &runDirectory,
            &error))
    {
        ADD_FAILURE() << error.toStdString();
        return {};
    }

    const QString artifactDirectory = QDir(runDirectory).filePath(
        QStringLiteral("products"));
    QString meshPath;
    if (coloredMesh)
    {
        meshPath = writeColoredTriangleMesh(artifactDirectory);
    }
    else
    {
        meshPath = QDir(artifactDirectory).filePath(
            QStringLiteral("model.ply"));
        writeFile(meshPath);
    }
    return completedRecord(
        runId,
        meshPath,
        QDir(runDirectory).filePath(QStringLiteral("model_result.json")));
}

QJsonObject withCompletedTextureRun(const QJsonObject &modelRecord,
                                    const QString &requestedRunId)
{
    QString textureRunId;
    QString textureRunDirectory;
    QString error;
    if (!xjw::mesh::workflow::createTextureRunOutputDirectory(
            modelRecord.value(QStringLiteral("model_run_directory")).toString(),
            requestedRunId,
            &textureRunId,
            &textureRunDirectory,
            &error))
    {
        ADD_FAILURE() << error.toStdString();
        return {};
    }

    QJsonObject record = modelRecord;
    const QString objPath = QDir(textureRunDirectory).filePath(
        QStringLiteral("model.obj"));
    const QString mtlPath = QDir(textureRunDirectory).filePath(
        QStringLiteral("model.mtl"));
    const QString texturePath = QDir(textureRunDirectory).filePath(
        QStringLiteral("texture.png"));
    const QString diagnosticsPath = QDir(textureRunDirectory).filePath(
        QStringLiteral("texture_result.json"));
    for (const QString &path : {objPath, mtlPath, texturePath})
    {
        writeFile(path);
    }
    record[QStringLiteral("textured")] = true;
    record[QStringLiteral("final_model_path")] = objPath;
    record[QStringLiteral("model_obj")] = objPath;
    record[QStringLiteral("model_mtl")] = mtlPath;
    record[QStringLiteral("texture_png")] = texturePath;
    record[QStringLiteral("texture_image")] = texturePath;
    record[QStringLiteral("texture_run_id")] = textureRunId;
    record[QStringLiteral("texture_run_directory")] = textureRunDirectory;
    record[QStringLiteral("texture_diagnostics_path")] = diagnosticsPath;

    QJsonObject diagnostics{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("diagnostics_type"), QStringLiteral("texture")},
        {QStringLiteral("ok"), true},
        {QStringLiteral("texture_run_id"), textureRunId},
        {QStringLiteral("texture_run_directory"), textureRunDirectory},
        {QStringLiteral("model_obj"), objPath},
        {QStringLiteral("model_mtl"), mtlPath},
        {QStringLiteral("texture_png"), texturePath},
        {QStringLiteral("texture_image"), texturePath},
        {QStringLiteral("texture_diagnostics_path"), diagnosticsPath}
    };
    writeJsonFile(diagnosticsPath, diagnostics);
    return record;
}

} // namespace

TEST(ModelOutputPolicyTest, ParsesDialogSettingIntoExplicitPolicy)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    EXPECT_EQ(xjw::mesh::workflow::modelOutputPolicyFromSettings({}),
              ModelOutputPolicy::CreateVersionedResult);
    EXPECT_EQ(xjw::mesh::workflow::modelOutputPolicyFromSettings(
                  QJsonObject{{QStringLiteral("replaceDefaultModel"), true}}),
              ModelOutputPolicy::ReplaceDefault);
    EXPECT_EQ(xjw::mesh::workflow::modelOutputPolicyFromSettings(
                  QJsonObject{{QStringLiteral("model_output_policy"),
                               QStringLiteral("create_versioned_result")},
                              {QStringLiteral("replaceDefaultModel"), true}}),
              ModelOutputPolicy::CreateVersionedResult);
}

TEST(ModelOutputPolicyTest, CreatesUniqueRunDirectoryAndRejectsReuse)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    QString runId;
    QString runRoot;
    QString error;
    ASSERT_TRUE(xjw::mesh::workflow::createModelRunOutputDirectory(
        temporary.path(),
        QStringLiteral("run-001"),
        &runId,
        &runRoot,
        &error)) << error.toStdString();
    EXPECT_EQ(runId, QStringLiteral("run-001"));
    EXPECT_TRUE(QFileInfo(runRoot).isDir());
    EXPECT_EQ(QFileInfo(runRoot).fileName(), QStringLiteral("run-001"));
    EXPECT_EQ(QFileInfo(runRoot).dir().dirName(), QStringLiteral("model_runs"));
    EXPECT_TRUE(QFileInfo(QDir(runRoot).filePath(
        QStringLiteral(".plascan_task_run.json"))).isFile());

    QString duplicateId;
    QString duplicateRoot;
    EXPECT_FALSE(xjw::mesh::workflow::createModelRunOutputDirectory(
        temporary.path(),
        QStringLiteral("run-001"),
        &duplicateId,
        &duplicateRoot,
        &error));
    EXPECT_FALSE(error.isEmpty());
}

TEST(ModelOutputPolicyTest, RefusesToDeleteRunWithoutOwnershipMarker)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString runId = QStringLiteral("unowned-run");
    const QString runRoot = QDir(temporary.path()).filePath(
        QStringLiteral("model_runs/%1").arg(runId));
    writeFile(QDir(runRoot).filePath(QStringLiteral("keep.txt")));

    QString error;
    EXPECT_FALSE(xjw::mesh::workflow::removeUnpublishedModelRunDirectory(
        temporary.path(), runId, runRoot, &error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(runRoot));
}

TEST(ModelOutputPolicyTest, BuildModelWritesCompleteIndependentRuns)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString pointCloudPath = writeDenseGridPointCloud(
        temporary.filePath(QStringLiteral("input")));
    xjw::mesh::workflow::ModelBuildRequest request;
    request.sourceData = QStringLiteral("point_cloud");
    request.requestedSourcePath = pointCloudPath;
    request.sourcePointCloudPath = pointCloudPath;
    request.outputRoot = temporary.filePath(QStringLiteral("model"));
    request.settings = QJsonObject{
        {QStringLiteral("surface_type"), QStringLiteral("height_field")},
        {QStringLiteral("method"), QStringLiteral("Height Grid")},
        {QStringLiteral("meshResolution"), 64},
        {QStringLiteral("cleanSmall"), false},
        {QStringLiteral("smoothIter"), 0},
        {QStringLiteral("depthFiltering"), QStringLiteral("disabled")}
    };
    request.runId = QStringLiteral("run-versioned");

    const auto first = xjw::mesh::workflow::buildModel(request);
    ASSERT_TRUE(first.ok) << first.errorMessage.toStdString();

    request.outputPolicy = ModelOutputPolicy::ReplaceDefault;
    request.runId = QStringLiteral("run-replacement");
    const auto second = xjw::mesh::workflow::buildModel(request);
    ASSERT_TRUE(second.ok) << second.errorMessage.toStdString();

    const QString firstModel = first.payload.value(
        QStringLiteral("model_ply")).toString();
    const QString secondModel = second.payload.value(
        QStringLiteral("model_ply")).toString();
    EXPECT_NE(firstModel, secondModel);
    EXPECT_TRUE(QFileInfo(firstModel).isFile());
    EXPECT_TRUE(QFileInfo(secondModel).isFile());
    EXPECT_TRUE(firstModel.contains(QStringLiteral("run-versioned")));
    EXPECT_TRUE(secondModel.contains(QStringLiteral("run-replacement")));
    EXPECT_EQ(first.payload.value(QStringLiteral("model_output_policy")).toString(),
              QStringLiteral("create_versioned_result"));
    EXPECT_EQ(second.payload.value(QStringLiteral("model_output_policy")).toString(),
              QStringLiteral("replace_default"));

    for (const QJsonObject &payload : {first.payload, second.payload})
    {
        const QString diagnosticsPath = payload.value(
            QStringLiteral("model_diagnostics_path")).toString();
        QFile diagnosticsFile(diagnosticsPath);
        ASSERT_TRUE(diagnosticsFile.open(QIODevice::ReadOnly));
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(
            diagnosticsFile.readAll(), &parseError);
        EXPECT_EQ(parseError.error, QJsonParseError::NoError);
        EXPECT_TRUE(document.object().value(QStringLiteral("ok")).toBool());
        EXPECT_EQ(document.object().value(
                      QStringLiteral("model_run_id")).toString(),
                  payload.value(QStringLiteral("model_run_id")).toString());
    }
}

TEST(ModelOutputPolicyTest,
     PointCloudCancellationDoesNotFinalizeOrPublishRun)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString pointCloudPath = writeDenseGridPointCloud(
        temporary.filePath(QStringLiteral("cancel-input")));
    const QString outputRoot = temporary.filePath(
        QStringLiteral("cancel-output"));
    const QString runId = QStringLiteral("cancelled-run");
    std::atomic_bool cancelRequested{false};

    xjw::mesh::workflow::ModelBuildRequest request;
    request.sourceData = QStringLiteral("point_cloud");
    request.requestedSourcePath = pointCloudPath;
    request.sourcePointCloudPath = pointCloudPath;
    request.outputRoot = outputRoot;
    request.runId = runId;
    request.settings = QJsonObject{
        {QStringLiteral("surface_type"), QStringLiteral("height_field")},
        {QStringLiteral("method"), QStringLiteral("Height Grid")},
        {QStringLiteral("meshResolution"), 64}
    };
    request.isCancelled = [&cancelRequested]()
    {
        return cancelRequested.load(std::memory_order_relaxed);
    };
    request.progress = [&cancelRequested](const QString &, int)
    {
        cancelRequested.store(true, std::memory_order_relaxed);
    };

    const auto result = xjw::mesh::workflow::buildModel(request);
    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.payload.value(QStringLiteral("cancelled")).toBool());
    EXPECT_FALSE(result.payload.contains(
        QStringLiteral("model_diagnostics_path")));

    const QString runDirectory = QDir(outputRoot).filePath(
        QStringLiteral("model_runs/%1").arg(runId));
    EXPECT_FALSE(QFileInfo::exists(runDirectory));
}

TEST(ModelOutputPolicyTest,
     FailedRunIsCleanedWithoutTouchingCommittedHistory)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString pointCloudPath = writeDenseGridPointCloud(
        temporary.filePath(QStringLiteral("history-input")));
    const QString outputRoot = temporary.filePath(
        QStringLiteral("history-output"));
    xjw::mesh::workflow::ModelBuildRequest request;
    request.sourceData = QStringLiteral("point_cloud");
    request.requestedSourcePath = pointCloudPath;
    request.sourcePointCloudPath = pointCloudPath;
    request.outputRoot = outputRoot;
    request.runId = QStringLiteral("committed-run");
    request.settings = QJsonObject{
        {QStringLiteral("surface_type"), QStringLiteral("height_field")},
        {QStringLiteral("method"), QStringLiteral("Height Grid")},
        {QStringLiteral("meshResolution"), 64},
        {QStringLiteral("cleanSmall"), false},
        {QStringLiteral("smoothIter"), 0}
    };

    const auto committed = xjw::mesh::workflow::buildModel(request);
    ASSERT_TRUE(committed.ok) << committed.errorMessage.toStdString();
    const QString committedDirectory = committed.payload.value(
        QStringLiteral("model_run_directory")).toString();
    const QString committedModel = committed.payload.value(
        QStringLiteral("model_ply")).toString();
    const QString committedDiagnostics = committed.payload.value(
        QStringLiteral("model_diagnostics_path")).toString();

    const QString missingInput = temporary.filePath(
        QStringLiteral("missing-input.ply"));
    request.requestedSourcePath = missingInput;
    request.sourcePointCloudPath = missingInput;
    request.runId = QStringLiteral("failed-run");
    const auto failed = xjw::mesh::workflow::buildModel(request);

    EXPECT_FALSE(failed.ok);
    EXPECT_FALSE(QFileInfo::exists(QDir(outputRoot).filePath(
        QStringLiteral("model_runs/failed-run"))));
    EXPECT_TRUE(QFileInfo(committedDirectory).isDir());
    EXPECT_GT(QFileInfo(committedModel).size(), 0);
    EXPECT_GT(QFileInfo(committedDiagnostics).size(), 0);
}

TEST(ModelOutputPolicyTest, TextureBuildWritesCompleteIndependentRevisions)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    QJsonObject baseRecord = createCompletedModelRun(
        temporary.filePath(QStringLiteral("model")),
        QStringLiteral("model-run"),
        true);
    ASSERT_FALSE(baseRecord.isEmpty());
    const QString meshPath = baseRecord.value(
        QStringLiteral("model_ply")).toString();
    const QString modelRunDirectory = baseRecord.value(
        QStringLiteral("model_run_directory")).toString();
    xjw::mesh::workflow::TextureBuildRequest request;
    request.meshPath = meshPath;
    request.allowVertexColorFallback = true;
    request.texture = xjw::mesh::workflow::defaultTextureConfig();
    request.texture.textureSize = 64;
    request.texture.padding = 2;

    QString firstRunId;
    QString firstRunRoot;
    QString error;
    ASSERT_TRUE(xjw::mesh::workflow::createTextureRunOutputDirectory(
        modelRunDirectory,
        QStringLiteral("texture-first"),
        &firstRunId,
        &firstRunRoot,
        &error)) << error.toStdString();
    request.outputDir = firstRunRoot;
    request.textureRunId = firstRunId;
    const auto first = xjw::mesh::workflow::buildTextureOnly(request);
    ASSERT_TRUE(first.ok) << first.errorMessage.toStdString();

    QString secondRunId;
    QString secondRunRoot;
    ASSERT_TRUE(xjw::mesh::workflow::createTextureRunOutputDirectory(
        modelRunDirectory,
        QStringLiteral("texture-second"),
        &secondRunId,
        &secondRunRoot,
        &error)) << error.toStdString();
    request.outputDir = secondRunRoot;
    request.textureRunId = secondRunId;
    const auto second = xjw::mesh::workflow::buildTextureOnly(request);
    ASSERT_TRUE(second.ok) << second.errorMessage.toStdString();

    for (const QString &key : {
             QStringLiteral("model_obj"),
             QStringLiteral("model_mtl"),
             QStringLiteral("texture_png")})
    {
        const QString firstPath = first.payload.value(key).toString();
        const QString secondPath = second.payload.value(key).toString();
        EXPECT_NE(firstPath, secondPath);
        EXPECT_GT(QFileInfo(firstPath).size(), 0);
        EXPECT_GT(QFileInfo(secondPath).size(), 0);
    }
    EXPECT_EQ(first.payload.value(QStringLiteral("texture_run_id")).toString(),
              firstRunId);
    EXPECT_EQ(second.payload.value(QStringLiteral("texture_run_id")).toString(),
              secondRunId);
    EXPECT_GT(QFileInfo(first.payload.value(
                  QStringLiteral("texture_diagnostics_path")).toString()).size(),
              0);
    EXPECT_GT(QFileInfo(second.payload.value(
                  QStringLiteral("texture_diagnostics_path")).toString()).size(),
              0);
    EXPECT_TRUE(first.payload.value(
        QStringLiteral("model_diagnostics_path")).toString().isEmpty());

    const QString modelDiagnostics = baseRecord.value(
        QStringLiteral("model_diagnostics_path")).toString();
    baseRecord[QStringLiteral("is_default_model")] = true;
    QJsonObject metadata{
        {QStringLiteral("model_results"), QJsonArray{baseRecord}}
    };
    const auto texturedRecord = [&baseRecord](const QJsonObject &payload)
    {
        QJsonObject record = baseRecord;
        for (auto it = payload.begin(); it != payload.end(); ++it)
        {
            record.insert(it.key(), it.value());
        }
        record[QStringLiteral("textured")] = true;
        record[QStringLiteral("final_model_path")] = payload.value(
            QStringLiteral("model_obj"));
        return record;
    };
    ASSERT_TRUE(xjw::gui::project::updateCompletedModelRun(
        &metadata, texturedRecord(first.payload), &error))
        << error.toStdString();
    ASSERT_TRUE(xjw::gui::project::updateCompletedModelRun(
        &metadata, texturedRecord(second.payload), &error))
        << error.toStdString();
    const QJsonObject stored = metadata.value(
        QStringLiteral("model_results")).toArray().at(0).toObject();
    EXPECT_EQ(stored.value(QStringLiteral("model_diagnostics_path")).toString(),
              modelDiagnostics);
    EXPECT_EQ(stored.value(QStringLiteral("texture_diagnostics_path")).toString(),
              second.payload.value(
                  QStringLiteral("texture_diagnostics_path")).toString());
    EXPECT_EQ(stored.value(QStringLiteral("texture_run_id")).toString(),
              secondRunId);
    EXPECT_GT(QFileInfo(first.payload.value(
                  QStringLiteral("model_obj")).toString()).size(),
              0);
}

TEST(ModelOutputPolicyTest, VersionedRunPreservesLegacyDefaultAndReplaceSwitchesIt)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString legacyMesh = temporary.filePath(QStringLiteral("legacy.ply"));
    writeFile(legacyMesh);
    const QJsonObject versionRecord = createCompletedModelRun(
        temporary.path(), QStringLiteral("run-version"));
    const QJsonObject replacementRecord = createCompletedModelRun(
        temporary.path(), QStringLiteral("run-replacement"));
    ASSERT_FALSE(versionRecord.isEmpty());
    ASSERT_FALSE(replacementRecord.isEmpty());
    const QString versionMesh = versionRecord.value(
        QStringLiteral("model_ply")).toString();
    const QString replacementMesh = replacementRecord.value(
        QStringLiteral("model_ply")).toString();

    QJsonObject metadata{
        {QStringLiteral("model_results"),
         QJsonArray{QJsonObject{{QStringLiteral("model_ply"), legacyMesh}}}}
    };
    QString error;
    ASSERT_TRUE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        versionRecord,
        ModelOutputPolicy::CreateVersionedResult,
        &error)) << error.toStdString();

    QJsonArray records = metadata.value(QStringLiteral("model_results")).toArray();
    ASSERT_EQ(records.size(), 2);
    EXPECT_TRUE(records.at(0).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_FALSE(records.at(1).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_EQ(xjw::gui::project::resolveDefaultModelResult(metadata).meshPath,
              legacyMesh);

    ASSERT_TRUE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        replacementRecord,
        ModelOutputPolicy::ReplaceDefault,
        &error)) << error.toStdString();
    records = metadata.value(QStringLiteral("model_results")).toArray();
    ASSERT_EQ(records.size(), 3);
    EXPECT_FALSE(records.at(0).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_FALSE(records.at(1).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_TRUE(records.at(2).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_EQ(xjw::gui::project::resolveDefaultModelResult(metadata).meshPath,
              replacementMesh);
    EXPECT_TRUE(QFileInfo::exists(legacyMesh));
    EXPECT_TRUE(QFileInfo::exists(versionMesh));
}

TEST(ModelOutputPolicyTest, IncompleteRunCannotChangeDefault)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString legacyMesh = temporary.filePath(QStringLiteral("legacy.ply"));
    writeFile(legacyMesh);
    QJsonObject candidateRecord = createCompletedModelRun(
        temporary.path(), QStringLiteral("incomplete"));
    ASSERT_FALSE(candidateRecord.isEmpty());
    ASSERT_TRUE(QFile::remove(candidateRecord.value(
        QStringLiteral("model_diagnostics_path")).toString()));
    QJsonObject metadata{
        {QStringLiteral("model_results"),
         QJsonArray{QJsonObject{{QStringLiteral("model_ply"), legacyMesh}}}}
    };
    const QJsonObject original = metadata;

    QString error;
    EXPECT_FALSE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        candidateRecord,
        ModelOutputPolicy::ReplaceDefault,
        &error));
    EXPECT_EQ(metadata, original);
    EXPECT_FALSE(error.isEmpty());
}

TEST(ModelOutputPolicyTest, UpdatingRunPreservesDefaultPositionAndMetadata)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    QJsonObject record = createCompletedModelRun(
        temporary.path(), QStringLiteral("run-default"));
    ASSERT_FALSE(record.isEmpty());
    record[QStringLiteral("is_default_model")] = true;
    QJsonObject metadata{
        {QStringLiteral("model_results"), QJsonArray{record}}
    };

    QJsonObject updated = record;
    updated.remove(QStringLiteral("is_default_model"));
    updated[QStringLiteral("quality_note")] = QStringLiteral("updated");
    QString error;
    ASSERT_TRUE(xjw::gui::project::updateCompletedModelRun(
        &metadata, updated, &error)) << error.toStdString();
    const QJsonArray records = metadata.value(
        QStringLiteral("model_results")).toArray();
    ASSERT_EQ(records.size(), 1);
    EXPECT_TRUE(records.at(0).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_EQ(records.at(0).toObject().value(
                  QStringLiteral("quality_note")).toString(),
              QStringLiteral("updated"));
}

TEST(ModelOutputPolicyTest, RegistrationNormalizesDefaultsAndRunId)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString firstMesh = temporary.filePath(QStringLiteral("first.ply"));
    const QString secondMesh = temporary.filePath(QStringLiteral("second.ply"));
    for (const QString &path : {firstMesh, secondMesh})
    {
        writeFile(path);
    }
    QJsonObject newRecord = createCompletedModelRun(
        temporary.path(), QStringLiteral("normalized-run"));
    ASSERT_FALSE(newRecord.isEmpty());
    newRecord[QStringLiteral("model_run_id")] =
        QStringLiteral("  normalized-run  ");
    QJsonObject metadata{
        {QStringLiteral("model_results"),
         QJsonArray{
             QJsonObject{{QStringLiteral("model_ply"), firstMesh},
                         {QStringLiteral("is_default_model"), true}},
             QJsonObject{{QStringLiteral("model_ply"), secondMesh},
                         {QStringLiteral("is_default_model"), true}}}}
    };
    QString error;
    ASSERT_TRUE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        newRecord,
        ModelOutputPolicy::CreateVersionedResult,
        &error)) << error.toStdString();

    const QJsonArray records = metadata.value(
        QStringLiteral("model_results")).toArray();
    ASSERT_EQ(records.size(), 3);
    EXPECT_FALSE(records.at(0).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_TRUE(records.at(1).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_FALSE(records.at(2).toObject().value(
        QStringLiteral("is_default_model")).toBool());
    EXPECT_EQ(records.at(2).toObject().value(
                  QStringLiteral("model_run_id")).toString(),
              QStringLiteral("normalized-run"));
}

TEST(ModelOutputPolicyTest, TexturedUpdateRequiresObjMtlAndTexture)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    QJsonObject record = createCompletedModelRun(
        temporary.path(), QStringLiteral("run-default"));
    ASSERT_FALSE(record.isEmpty());
    record[QStringLiteral("is_default_model")] = true;
    QJsonObject metadata{
        {QStringLiteral("model_results"), QJsonArray{record}}
    };
    const QJsonObject original = metadata;

    const QJsonObject textured = withCompletedTextureRun(
        record, QStringLiteral("texture-run"));
    ASSERT_FALSE(textured.isEmpty());
    QString error;
    for (const QString &requiredKey : {
             QStringLiteral("model_obj"),
             QStringLiteral("model_mtl"),
             QStringLiteral("texture_png"),
             QStringLiteral("texture_diagnostics_path")})
    {
        QJsonObject incomplete = textured;
        incomplete.remove(requiredKey);
        EXPECT_FALSE(xjw::gui::project::updateCompletedModelRun(
            &metadata, incomplete, &error)) << qPrintable(requiredKey);
        EXPECT_EQ(metadata, original);
    }

    ASSERT_TRUE(xjw::gui::project::updateCompletedModelRun(
        &metadata, textured, &error)) << error.toStdString();
    EXPECT_TRUE(metadata.value(QStringLiteral("model_results")).toArray()
                    .at(0).toObject().value(QStringLiteral("textured")).toBool());
}

TEST(ModelOutputPolicyTest, RejectsLiteralDiagnosticsBeforePublication)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    QJsonObject record = createCompletedModelRun(
        temporary.path(), QStringLiteral("literal-diagnostics"));
    ASSERT_FALSE(record.isEmpty());
    writeFile(record.value(
        QStringLiteral("model_diagnostics_path")).toString());

    QJsonObject metadata;
    const QJsonObject original = metadata;
    QString error;
    EXPECT_FALSE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        record,
        ModelOutputPolicy::ReplaceDefault,
        &error));
    EXPECT_EQ(metadata, original);
    EXPECT_FALSE(error.isEmpty());
}

TEST(ModelOutputPolicyTest, RejectsArtifactFromAnotherRun)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    QJsonObject first = createCompletedModelRun(
        temporary.path(), QStringLiteral("first-run"));
    const QJsonObject second = createCompletedModelRun(
        temporary.path(), QStringLiteral("second-run"));
    ASSERT_FALSE(first.isEmpty());
    ASSERT_FALSE(second.isEmpty());
    const QString foreignMesh = second.value(
        QStringLiteral("model_ply")).toString();
    first[QStringLiteral("model_ply")] = foreignMesh;
    first[QStringLiteral("mesh_ply")] = foreignMesh;
    first[QStringLiteral("final_model_path")] = foreignMesh;

    QJsonObject metadata;
    QString error;
    EXPECT_FALSE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        first,
        ModelOutputPolicy::ReplaceDefault,
        &error));
    EXPECT_TRUE(metadata.isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(ModelOutputPolicyTest, RejectsDiagnosticsWithWrongRunIdentity)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QJsonObject record = createCompletedModelRun(
        temporary.path(), QStringLiteral("identity-run"));
    ASSERT_FALSE(record.isEmpty());
    const QString diagnosticsPath = record.value(
        QStringLiteral("model_diagnostics_path")).toString();
    QJsonObject diagnostics = record;
    diagnostics[QStringLiteral("schema_version")] = 1;
    diagnostics[QStringLiteral("diagnostics_type")] =
        QStringLiteral("model");
    diagnostics[QStringLiteral("ok")] = true;
    diagnostics[QStringLiteral("model_run_id")] =
        QStringLiteral("another-run");
    writeJsonFile(diagnosticsPath, diagnostics);

    QJsonObject metadata;
    QString error;
    EXPECT_FALSE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        record,
        ModelOutputPolicy::ReplaceDefault,
        &error));
    EXPECT_TRUE(metadata.isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(ModelOutputPolicyTest, RejectsModelDiagnosticsWithTextureType)
{
    using xjw::mesh::workflow::ModelOutputPolicy;
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QJsonObject record = createCompletedModelRun(
        temporary.path(), QStringLiteral("wrong-type-run"));
    ASSERT_FALSE(record.isEmpty());
    const QString diagnosticsPath = record.value(
        QStringLiteral("model_diagnostics_path")).toString();
    QJsonObject diagnostics = record;
    diagnostics[QStringLiteral("schema_version")] = 1;
    diagnostics[QStringLiteral("diagnostics_type")] =
        QStringLiteral("texture");
    diagnostics[QStringLiteral("ok")] = true;
    writeJsonFile(diagnosticsPath, diagnostics);

    QJsonObject metadata;
    QString error;
    EXPECT_FALSE(xjw::gui::project::registerCompletedModelRun(
        &metadata,
        record,
        ModelOutputPolicy::ReplaceDefault,
        &error));
    EXPECT_TRUE(metadata.isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(ModelOutputPolicyTest, ResolvesLegacyRecordWithoutRunDiagnostics)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString legacyMesh = temporary.filePath(
        QStringLiteral("legacy-model.ply"));
    writeFile(legacyMesh);
    const QJsonObject legacyRecord{
        {QStringLiteral("model_ply"), legacyMesh},
        {QStringLiteral("is_default_model"), true}
    };
    const QJsonObject metadata{
        {QStringLiteral("model_results"), QJsonArray{legacyRecord}}
    };

    const auto resolved =
        xjw::gui::project::resolveDefaultModelResult(metadata);
    EXPECT_TRUE(resolved.ok);
    EXPECT_EQ(resolved.index, 0);
    EXPECT_EQ(resolved.meshPath, legacyMesh);
    EXPECT_EQ(resolved.modelRecord, legacyRecord);
}
