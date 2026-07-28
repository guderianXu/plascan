#include "ProcessingBaselineManager.h"

#include "MeshTopologyQuality.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace xjw::qc
{
namespace
{

constexpr auto kSchema = "plascan.processing_baseline.v1";

QJsonObject metricsToJson(const ProcessingBaselineMeshMetrics &metrics)
{
    QJsonObject object;
    object.insert(QStringLiteral("vertex_count"), metrics.vertexCount);
    object.insert(QStringLiteral("face_count"), metrics.faceCount);
    object.insert(QStringLiteral("boundary_edge_count"), metrics.boundaryEdgeCount);
    object.insert(QStringLiteral("non_manifold_edge_count"), metrics.nonManifoldEdgeCount);
    object.insert(QStringLiteral("component_count"), metrics.componentCount);
    object.insert(QStringLiteral("largest_component_face_ratio"),
                  metrics.largestComponentFaceRatio);
    object.insert(QStringLiteral("high_aspect_face_ratio"),
                  metrics.highAspectFaceRatio);
    object.insert(QStringLiteral("extreme_aspect_face_ratio"),
                  metrics.extremeAspectFaceRatio);
    object.insert(QStringLiteral("adjacent_normal_angle_median_degrees"),
                  metrics.adjacentNormalAngleMedianDegrees);
    object.insert(QStringLiteral("adjacent_normal_angle_p90_degrees"),
                  metrics.adjacentNormalAngleP90Degrees);
    object.insert(QStringLiteral("adjacent_normal_angle_over_30_ratio"),
                  metrics.adjacentNormalAngleOver30Ratio);
    object.insert(QStringLiteral("surface_area"), metrics.surfaceArea);
    object.insert(QStringLiteral("bounding_box_diagonal"),
                  metrics.boundingBoxDiagonal);
    object.insert(QStringLiteral("normalized_surface_area"),
                  metrics.normalizedSurfaceArea);
    return object;
}

ProcessingBaselineMeshMetrics metricsFromJson(const QJsonObject &object)
{
    ProcessingBaselineMeshMetrics metrics;
    metrics.vertexCount = object.value(QStringLiteral("vertex_count")).toInt();
    metrics.faceCount = object.value(QStringLiteral("face_count")).toInt();
    metrics.boundaryEdgeCount =
        object.value(QStringLiteral("boundary_edge_count")).toInt();
    metrics.nonManifoldEdgeCount =
        object.value(QStringLiteral("non_manifold_edge_count")).toInt();
    metrics.componentCount = object.value(QStringLiteral("component_count")).toInt();
    metrics.largestComponentFaceRatio =
        object.value(QStringLiteral("largest_component_face_ratio")).toDouble();
    metrics.highAspectFaceRatio =
        object.value(QStringLiteral("high_aspect_face_ratio")).toDouble();
    metrics.extremeAspectFaceRatio =
        object.value(QStringLiteral("extreme_aspect_face_ratio")).toDouble();
    metrics.adjacentNormalAngleMedianDegrees =
        object.value(QStringLiteral("adjacent_normal_angle_median_degrees")).toDouble();
    metrics.adjacentNormalAngleP90Degrees =
        object.value(QStringLiteral("adjacent_normal_angle_p90_degrees")).toDouble();
    metrics.adjacentNormalAngleOver30Ratio =
        object.value(QStringLiteral("adjacent_normal_angle_over_30_ratio")).toDouble();
    metrics.surfaceArea = object.value(QStringLiteral("surface_area")).toDouble();
    metrics.boundingBoxDiagonal =
        object.value(QStringLiteral("bounding_box_diagonal")).toDouble();
    metrics.normalizedSurfaceArea =
        object.value(QStringLiteral("normalized_surface_area")).toDouble();
    return metrics;
}

QJsonObject thresholdsToJson(const ProcessingBaselineThresholds &thresholds)
{
    QJsonObject object;
    object.insert(QStringLiteral("maximum_face_count_ratio"),
                  thresholds.maximumFaceCountRatio);
    object.insert(QStringLiteral("maximum_boundary_edge_count_ratio"),
                  thresholds.maximumBoundaryEdgeCountRatio);
    object.insert(QStringLiteral("maximum_normalized_surface_area_ratio"),
                  thresholds.maximumNormalizedSurfaceAreaRatio);
    object.insert(QStringLiteral("maximum_high_aspect_face_ratio"),
                  thresholds.maximumHighAspectFaceRatio);
    object.insert(QStringLiteral("maximum_extreme_aspect_face_ratio"),
                  thresholds.maximumExtremeAspectFaceRatio);
    object.insert(QStringLiteral("maximum_adjacent_normal_angle_median_degrees"),
                  thresholds.maximumAdjacentNormalAngleMedianDegrees);
    object.insert(QStringLiteral("maximum_adjacent_normal_angle_p90_degrees"),
                  thresholds.maximumAdjacentNormalAngleP90Degrees);
    object.insert(QStringLiteral("maximum_adjacent_normal_angle_over_30_ratio"),
                  thresholds.maximumAdjacentNormalAngleOver30Ratio);
    object.insert(QStringLiteral("maximum_component_count"),
                  thresholds.maximumComponentCount);
    object.insert(QStringLiteral("minimum_largest_component_face_ratio"),
                  thresholds.minimumLargestComponentFaceRatio);
    return object;
}

ProcessingBaselineThresholds thresholdsFromJson(const QJsonObject &object)
{
    ProcessingBaselineThresholds thresholds;
    const auto valueOr = [&object](const char *name, double fallback)
    {
        const QJsonValue value = object.value(QString::fromLatin1(name));
        return value.isDouble() ? value.toDouble() : fallback;
    };
    thresholds.maximumFaceCountRatio =
        valueOr("maximum_face_count_ratio", thresholds.maximumFaceCountRatio);
    thresholds.maximumBoundaryEdgeCountRatio =
        valueOr("maximum_boundary_edge_count_ratio",
                thresholds.maximumBoundaryEdgeCountRatio);
    thresholds.maximumNormalizedSurfaceAreaRatio =
        valueOr("maximum_normalized_surface_area_ratio",
                thresholds.maximumNormalizedSurfaceAreaRatio);
    thresholds.maximumHighAspectFaceRatio =
        valueOr("maximum_high_aspect_face_ratio",
                thresholds.maximumHighAspectFaceRatio);
    thresholds.maximumExtremeAspectFaceRatio =
        valueOr("maximum_extreme_aspect_face_ratio",
                thresholds.maximumExtremeAspectFaceRatio);
    thresholds.maximumAdjacentNormalAngleMedianDegrees =
        valueOr("maximum_adjacent_normal_angle_median_degrees",
                thresholds.maximumAdjacentNormalAngleMedianDegrees);
    thresholds.maximumAdjacentNormalAngleP90Degrees =
        valueOr("maximum_adjacent_normal_angle_p90_degrees",
                thresholds.maximumAdjacentNormalAngleP90Degrees);
    thresholds.maximumAdjacentNormalAngleOver30Ratio =
        valueOr("maximum_adjacent_normal_angle_over_30_ratio",
                thresholds.maximumAdjacentNormalAngleOver30Ratio);
    thresholds.maximumComponentCount = static_cast<int>(
        valueOr("maximum_component_count", thresholds.maximumComponentCount));
    thresholds.minimumLargestComponentFaceRatio =
        valueOr("minimum_largest_component_face_ratio",
                thresholds.minimumLargestComponentFaceRatio);
    return thresholds;
}

double ratio(double candidate, double reference)
{
    return reference > 0.0
        ? candidate / reference
        : (candidate <= 0.0 ? 1.0 : std::numeric_limits<double>::infinity());
}

void appendRatioFailure(QStringList *failures,
                        const QString &label,
                        double candidate,
                        double reference,
                        double maximum)
{
    const double actual = ratio(candidate, reference);
    if (actual > maximum)
    {
        failures->push_back(
            QStringLiteral("%1相对基线上升 %2 倍，超过上限 %3 倍")
                .arg(label)
                .arg(actual, 0, 'f', 3)
                .arg(maximum, 0, 'f', 3));
    }
}

void appendMaximumFailure(QStringList *failures,
                          const QString &label,
                          double actual,
                          double maximum)
{
    if (actual > maximum)
    {
        failures->push_back(
            QStringLiteral("%1为 %2，超过上限 %3")
                .arg(label)
                .arg(actual, 0, 'f', 4)
                .arg(maximum, 0, 'f', 4));
    }
}

} // namespace

QString ProcessingBaselineManager::inputFingerprint(
    const QJsonObject &inputSnapshot)
{
    const QByteArray canonical =
        QJsonDocument(inputSnapshot).toJson(QJsonDocument::Compact);
    return QString::fromLatin1(
        QCryptographicHash::hash(canonical, QCryptographicHash::Sha256).toHex());
}

ProcessingBaselineMeshMetrics ProcessingBaselineManager::analyzeMesh(
    const xjw::mesh::TriMesh &mesh)
{
    ProcessingBaselineMeshMetrics metrics;
    const xjw::mesh::MeshTopologyQualityStatistics topology =
        xjw::mesh::evaluateMeshTopologyQuality(mesh);
    metrics.vertexCount = mesh.vertexCount();
    metrics.faceCount = topology.validFaceCount;
    metrics.boundaryEdgeCount = topology.boundaryEdgeCount;
    metrics.nonManifoldEdgeCount = topology.nonManifoldEdgeCount;
    metrics.componentCount = topology.componentCount;
    metrics.largestComponentFaceRatio = topology.largestComponentFaceRatio;
    metrics.highAspectFaceRatio = topology.highAspectFaceRatio;
    metrics.extremeAspectFaceRatio = topology.extremeAspectFaceRatio;
    metrics.adjacentNormalAngleMedianDegrees =
        topology.adjacentNormalAngleMedianDegrees;
    metrics.adjacentNormalAngleP90Degrees =
        topology.adjacentNormalAngleP90Degrees;
    metrics.adjacentNormalAngleOver30Ratio =
        topology.adjacentNormalAngleOver30Ratio;

    std::array<double, 3> minimum{{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()}};
    std::array<double, 3> maximum{{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}};
    for (const xjw::mesh::MeshVertex &vertex : mesh.vertices)
    {
        const std::array<double, 3> point{{vertex.x, vertex.y, vertex.z}};
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[static_cast<std::size_t>(axis)] =
                std::min(minimum[static_cast<std::size_t>(axis)],
                         point[static_cast<std::size_t>(axis)]);
            maximum[static_cast<std::size_t>(axis)] =
                std::max(maximum[static_cast<std::size_t>(axis)],
                         point[static_cast<std::size_t>(axis)]);
        }
    }
    for (const xjw::mesh::Triangle &face : mesh.faces)
    {
        if (face.v[0] < 0 || face.v[1] < 0 || face.v[2] < 0 ||
            static_cast<std::size_t>(face.v[0]) >= mesh.vertices.size() ||
            static_cast<std::size_t>(face.v[1]) >= mesh.vertices.size() ||
            static_cast<std::size_t>(face.v[2]) >= mesh.vertices.size())
        {
            continue;
        }
        const xjw::mesh::MeshVertex &a =
            mesh.vertices[static_cast<std::size_t>(face.v[0])];
        const xjw::mesh::MeshVertex &b =
            mesh.vertices[static_cast<std::size_t>(face.v[1])];
        const xjw::mesh::MeshVertex &c =
            mesh.vertices[static_cast<std::size_t>(face.v[2])];
        const double ab_x = b.x - a.x;
        const double ab_y = b.y - a.y;
        const double ab_z = b.z - a.z;
        const double ac_x = c.x - a.x;
        const double ac_y = c.y - a.y;
        const double ac_z = c.z - a.z;
        const double cross_x = ab_y * ac_z - ab_z * ac_y;
        const double cross_y = ab_z * ac_x - ab_x * ac_z;
        const double cross_z = ab_x * ac_y - ab_y * ac_x;
        metrics.surfaceArea += 0.5 * std::sqrt(
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
    }
    if (!mesh.vertices.empty())
    {
        double squared_diagonal = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double extent =
                maximum[static_cast<std::size_t>(axis)] -
                minimum[static_cast<std::size_t>(axis)];
            squared_diagonal += extent * extent;
        }
        metrics.boundingBoxDiagonal = std::sqrt(squared_diagonal);
        if (squared_diagonal > 0.0)
        {
            metrics.normalizedSurfaceArea =
                metrics.surfaceArea / squared_diagonal;
        }
    }
    return metrics;
}

ProcessingBaselineDefinition ProcessingBaselineManager::create(
    const QString &name,
    const QString &sceneType,
    const QJsonObject &inputSnapshot,
    const xjw::mesh::TriMesh &referenceMesh,
    const ProcessingBaselineThresholds &thresholds)
{
    ProcessingBaselineDefinition baseline;
    baseline.name = name;
    baseline.sceneType = sceneType;
    baseline.createdAtUtc =
        QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    baseline.inputSnapshot = inputSnapshot;
    baseline.inputFingerprintSha256 = inputFingerprint(inputSnapshot);
    baseline.referenceMesh = analyzeMesh(referenceMesh);
    baseline.thresholds = thresholds;
    return baseline;
}

ProcessingBaselineComparison ProcessingBaselineManager::compare(
    const ProcessingBaselineDefinition &baseline,
    const QJsonObject &candidateInputSnapshot,
    const xjw::mesh::TriMesh &candidateMesh)
{
    ProcessingBaselineComparison comparison;
    comparison.candidateMesh = analyzeMesh(candidateMesh);
    comparison.inputMatches =
        inputFingerprint(candidateInputSnapshot) ==
        baseline.inputFingerprintSha256;
    if (!comparison.inputMatches)
    {
        comparison.failures.push_back(
            QStringLiteral("候选处理的输入快照与基线不一致"));
    }

    const auto &reference = baseline.referenceMesh;
    const auto &candidate = comparison.candidateMesh;
    const auto &thresholds = baseline.thresholds;
    appendRatioFailure(&comparison.failures, QStringLiteral("面数"),
                       candidate.faceCount, reference.faceCount,
                       thresholds.maximumFaceCountRatio);
    appendRatioFailure(&comparison.failures, QStringLiteral("开放边数量"),
                       candidate.boundaryEdgeCount, reference.boundaryEdgeCount,
                       thresholds.maximumBoundaryEdgeCountRatio);
    appendRatioFailure(&comparison.failures, QStringLiteral("归一化表面积"),
                       candidate.normalizedSurfaceArea,
                       reference.normalizedSurfaceArea,
                       thresholds.maximumNormalizedSurfaceAreaRatio);
    appendMaximumFailure(&comparison.failures, QStringLiteral("高宽高比三角面比例"),
                         candidate.highAspectFaceRatio,
                         thresholds.maximumHighAspectFaceRatio);
    appendMaximumFailure(&comparison.failures, QStringLiteral("极端宽高比三角面比例"),
                         candidate.extremeAspectFaceRatio,
                         thresholds.maximumExtremeAspectFaceRatio);
    appendMaximumFailure(&comparison.failures, QStringLiteral("相邻面法线夹角中位数"),
                         candidate.adjacentNormalAngleMedianDegrees,
                         thresholds.maximumAdjacentNormalAngleMedianDegrees);
    appendMaximumFailure(&comparison.failures, QStringLiteral("相邻面法线夹角 P90"),
                         candidate.adjacentNormalAngleP90Degrees,
                         thresholds.maximumAdjacentNormalAngleP90Degrees);
    appendMaximumFailure(&comparison.failures,
                         QStringLiteral("法线夹角超过 30 度的相邻面比例"),
                         candidate.adjacentNormalAngleOver30Ratio,
                         thresholds.maximumAdjacentNormalAngleOver30Ratio);
    if (candidate.componentCount > thresholds.maximumComponentCount)
    {
        comparison.failures.push_back(
            QStringLiteral("连通分量为 %1，超过上限 %2")
                .arg(candidate.componentCount)
                .arg(thresholds.maximumComponentCount));
    }
    if (candidate.largestComponentFaceRatio <
        thresholds.minimumLargestComponentFaceRatio)
    {
        comparison.failures.push_back(
            QStringLiteral("最大连通分量面占比为 %1，低于下限 %2")
                .arg(candidate.largestComponentFaceRatio, 0, 'f', 4)
                .arg(thresholds.minimumLargestComponentFaceRatio, 0, 'f', 4));
    }
    comparison.passed = comparison.inputMatches &&
        comparison.failures.isEmpty();
    comparison.report.insert(QStringLiteral("schema"),
                             QStringLiteral("plascan.processing_baseline_comparison.v1"));
    comparison.report.insert(QStringLiteral("baseline_name"), baseline.name);
    comparison.report.insert(QStringLiteral("input_matches"),
                             comparison.inputMatches);
    comparison.report.insert(QStringLiteral("passed"), comparison.passed);
    comparison.report.insert(QStringLiteral("reference_mesh"),
                             metricsToJson(reference));
    comparison.report.insert(QStringLiteral("candidate_mesh"),
                             metricsToJson(candidate));
    QJsonArray failures;
    for (const QString &failure : comparison.failures)
    {
        failures.push_back(failure);
    }
    comparison.report.insert(QStringLiteral("failures"), failures);
    return comparison;
}

QJsonObject ProcessingBaselineManager::toJson(
    const ProcessingBaselineDefinition &baseline)
{
    QJsonObject object;
    object.insert(QStringLiteral("schema"), QString::fromLatin1(kSchema));
    object.insert(QStringLiteral("name"), baseline.name);
    object.insert(QStringLiteral("scene_type"), baseline.sceneType);
    object.insert(QStringLiteral("created_at_utc"), baseline.createdAtUtc);
    object.insert(QStringLiteral("input_snapshot"), baseline.inputSnapshot);
    object.insert(QStringLiteral("input_fingerprint_sha256"),
                  baseline.inputFingerprintSha256);
    object.insert(QStringLiteral("reference_mesh"),
                  metricsToJson(baseline.referenceMesh));
    object.insert(QStringLiteral("thresholds"),
                  thresholdsToJson(baseline.thresholds));
    return object;
}

bool ProcessingBaselineManager::fromJson(
    const QJsonObject &object,
    ProcessingBaselineDefinition *baseline,
    QString *errorMessage)
{
    if (!baseline)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("处理基线输出对象不能为空");
        }
        return false;
    }
    if (object.value(QStringLiteral("schema")).toString() !=
        QString::fromLatin1(kSchema))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("不支持的处理基线格式");
        }
        return false;
    }
    ProcessingBaselineDefinition decoded;
    decoded.name = object.value(QStringLiteral("name")).toString();
    decoded.sceneType = object.value(QStringLiteral("scene_type")).toString();
    decoded.createdAtUtc =
        object.value(QStringLiteral("created_at_utc")).toString();
    decoded.inputSnapshot =
        object.value(QStringLiteral("input_snapshot")).toObject();
    decoded.inputFingerprintSha256 =
        object.value(QStringLiteral("input_fingerprint_sha256")).toString();
    decoded.referenceMesh =
        metricsFromJson(object.value(QStringLiteral("reference_mesh")).toObject());
    decoded.thresholds =
        thresholdsFromJson(object.value(QStringLiteral("thresholds")).toObject());
    if (decoded.name.trimmed().isEmpty() ||
        decoded.referenceMesh.faceCount <= 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("处理基线缺少名称或有效参考网格");
        }
        return false;
    }
    if (decoded.inputFingerprintSha256 !=
        inputFingerprint(decoded.inputSnapshot))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("处理基线输入指纹校验失败");
        }
        return false;
    }
    *baseline = decoded;
    return true;
}

bool ProcessingBaselineManager::save(
    const QString &path,
    const ProcessingBaselineDefinition &baseline,
    QString *errorMessage)
{
    const QFileInfo file_info(path);
    if (!QDir().mkpath(file_info.absolutePath()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建处理基线目录: %1")
                .arg(file_info.absolutePath());
        }
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入处理基线: %1")
                .arg(file.errorString());
        }
        return false;
    }
    file.write(QJsonDocument(toJson(baseline)).toJson(QJsonDocument::Indented));
    if (!file.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("提交处理基线失败: %1")
                .arg(file.errorString());
        }
        return false;
    }
    return true;
}

bool ProcessingBaselineManager::load(
    const QString &path,
    ProcessingBaselineDefinition *baseline,
    QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取处理基线: %1")
                .arg(file.errorString());
        }
        return false;
    }
    QJsonParseError parse_error;
    const QJsonDocument document =
        QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (parse_error.error != QJsonParseError::NoError ||
        !document.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("处理基线 JSON 无效: %1")
                .arg(parse_error.errorString());
        }
        return false;
    }
    return fromJson(document.object(), baseline, errorMessage);
}

} // namespace xjw::qc
