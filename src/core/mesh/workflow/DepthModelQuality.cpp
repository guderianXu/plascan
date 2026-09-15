#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    QString orbitalRoleForDepthFrame(const QJsonArray& roles, int refIndex)
    {
        for (const QJsonValue& value : roles)
        {
            const QJsonObject role = value.toObject();
            if (role.value(QStringLiteral("ref_index")).toInt(-1) == refIndex)
            {
                return role.value(QStringLiteral("role")).toString();
            }
        }
        return {};
    }

    QStringList worstDepthCompletenessLabels(const DepthMeshCompletenessStatistics& completeness,
                                             const QJsonArray& orbitalRoles,
                                             int maximumCount)
    {
        std::vector<DepthMeshFrameCompleteness> frames(completeness.frames.cbegin(), completeness.frames.cend());
        std::sort(frames.begin(),
                  frames.end(),
                  [](const DepthMeshFrameCompleteness& lhs, const DepthMeshFrameCompleteness& rhs)
                  { return lhs.recall < rhs.recall; });
        QStringList labels;
        const int count = std::min(maximumCount, static_cast<int>(frames.size()));
        for (int index = 0; index < count; ++index)
        {
            const DepthMeshFrameCompleteness& frame = frames[static_cast<std::size_t>(index)];
            const QString role = orbitalRoleForDepthFrame(orbitalRoles, frame.refIndex);
            labels.push_back(
                role.isEmpty()
                    ? QStringLiteral("%1=%2%").arg(frame.refIndex).arg(100.0 * frame.recall, 0, 'f', 1)
                    : QStringLiteral("%1=%2%[%3]").arg(frame.refIndex).arg(100.0 * frame.recall, 0, 'f', 1).arg(role));
        }
        return labels;
    }

    void addDepthCompletenessPayload(const DepthMeshCompletenessStatistics& completeness,
                                     const QString& prefix,
                                     QJsonObject* payload)
    {
        if (payload == nullptr)
        {
            return;
        }
        (*payload)[prefix + QStringLiteral("available")] = completeness.available;
        (*payload)[prefix + QStringLiteral("gate_passed")] = completeness.gatePassed;
        (*payload)[prefix + QStringLiteral("distance_method")] = QStringLiteral("exact_point_to_triangle_bvh");
        (*payload)[prefix + QStringLiteral("tolerance")] = completeness.tolerance;
        (*payload)[prefix + QStringLiteral("sampled_point_count")] =
            static_cast<double>(completeness.sampledDepthPointCount);
        (*payload)[prefix + QStringLiteral("explained_point_count")] =
            static_cast<double>(completeness.explainedDepthPointCount);
        (*payload)[prefix + QStringLiteral("aggregate_recall")] = completeness.aggregateRecall;
        (*payload)[prefix + QStringLiteral("minimum_frame_recall")] = completeness.minimumFrameRecall;
        (*payload)[prefix + QStringLiteral("p10_frame_recall")] = completeness.p10FrameRecall;
        (*payload)[prefix + QStringLiteral("median_frame_recall")] = completeness.medianFrameRecall;
        QJsonArray frames;
        for (const DepthMeshFrameCompleteness& frame : completeness.frames)
        {
            frames.push_back(QJsonObject{
                {QStringLiteral("ref_index"), frame.refIndex},
                {QStringLiteral("auxiliary_surface_only"), frame.auxiliarySurfaceOnly},
                {QStringLiteral("sampled_point_count"), static_cast<double>(frame.sampledDepthPointCount)},
                {QStringLiteral("explained_point_count"), static_cast<double>(frame.explainedDepthPointCount)},
                {QStringLiteral("recall"), frame.recall}});
        }
        (*payload)[prefix + QStringLiteral("frames")] = frames;
    }

    DepthMeshCompletenessStatistics evaluateDepthCompleteness(const TriMesh& mesh,
                                                              const QVector<DepthTsdfFrame>& frames,
                                                              const DepthTsdfOptions& options,
                                                              const DepthTsdfLayout& layout,
                                                              const QVector<int>& excludedRefIndices)
    {
        DepthMeshCompletenessOptions completeness_options;
        completeness_options.maximumDepthSamplesPerFrame = options.depthCompletenessMaximumSamplesPerFrame;
        completeness_options.tolerance = std::max({layout.voxelSize[0], layout.voxelSize[1], layout.voxelSize[2]}) *
                                         std::max(1.0f, options.depthCompletenessToleranceVoxels);
        completeness_options.minimumP10FrameRecall = options.minimumDepthCompletenessP10Recall;
        completeness_options.minimumMedianFrameRecall = options.minimumDepthCompletenessMedianRecall;
        completeness_options.excludeAuxiliaryFrames = true;
        completeness_options.excludedRefIndices.assign(excludedRefIndices.cbegin(), excludedRefIndices.cend());
        return DepthMeshCompleteness::evaluate(mesh, frames, completeness_options);
    }

    PointCloudQualityReport evaluatePointCloudQuality(const QString& pointCloudPath, qint64 recommendedMinimum)
    {
        PointCloudQualityReport report;

        const QString extension = QFileInfo(pointCloudPath).suffix().toLower();
        if (extension == QStringLiteral("ply"))
        {
            QFile file(pointCloudPath);
            if (file.open(QIODevice::ReadOnly | QIODevice::Text))
            {
                QTextStream stream(&file);
                while (!stream.atEnd())
                {
                    const QString line = stream.readLine().trimmed();
                    if (line.startsWith(QStringLiteral("element vertex")))
                    {
                        report.pointCount = line.split(QLatin1Char(' ')).last().toLongLong();
                        report.hasCount = true;
                    }
                    if (line == QStringLiteral("end_header"))
                    {
                        break;
                    }
                }
                file.close();
            }
        }
        else
        {
            QFile file(pointCloudPath);
            if (file.open(QIODevice::ReadOnly | QIODevice::Text))
            {
                report.hasCount = true;
                while (!file.atEnd())
                {
                    file.readLine();
                    ++report.pointCount;
                }
                file.close();
            }
        }

        report.belowRecommended = report.hasCount && report.pointCount < recommendedMinimum;
        return report;
    }
} // namespace xjw::mesh::workflow
