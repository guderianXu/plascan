#include "TrackBuildStage.h"

#include "ImageMatchRepository.h"
#include "MatchPhotosRuntime.h"
#include "TiePointTrackManager.h"

#include <QHash>
#include <QJsonArray>
#include <QSet>

#include <vector>

namespace xjw::matchphotos
{
    namespace
    {

        MatchPhotosStageReport makeTrackReport(MatchPhotosStageStatus status, const QString& message, int itemCount = 0)
        {
            MatchPhotosStageReport report;
            report.stageId = QStringLiteral("track_build");
            report.displayName = QStringLiteral("连接点轨迹与匹配提交");
            report.status = status;
            report.message = message;
            report.itemCount = itemCount;
            return report;
        }

        image_matching::ImageMatchWriteResult persistMatches(const MatchPhotosContext& context,
                                                             const std::vector<MatchPhotosMatchRecord>& records)
        {
            std::vector<const image_matching::PairMatchData*> pairs;
            pairs.reserve(records.size());
            for (const MatchPhotosMatchRecord& record : records)
            {
                if (record.pairData)
                {
                    pairs.push_back(record.pairData.get());
                }
            }
            image_matching::ImageMatchRepository repository(matchPhotosMatchDirectory(context));
            return repository.writePairReferences(pairs, true);
        }

        void populateImageMatchRecords(const MatchPhotosContext& context,
                                       const std::vector<MatchPhotosMatchRecord>& pairRecords,
                                       MatchPhotosResult* result)
        {
            if (!result)
            {
                return;
            }

            struct PendingRecord
            {
                MatchPhotosImageMatchRecord record;
                QSet<QString> neighbors;
            };

            const image_matching::ImageMatchRepository repository(matchPhotosMatchDirectory(context));
            QHash<QString, int> indexByImageId;
            std::vector<PendingRecord> pending;
            const auto appendPairSide =
                [&](const QString& ownerPath, const QString& peerPath, const QJsonObject& pairSettings)
            {
                const QString stableId = image_matching::ImageMatchFile::stableImageId(ownerPath);
                int index = indexByImageId.value(stableId, -1);
                if (index < 0)
                {
                    PendingRecord item;
                    item.record.imagePath = ownerPath;
                    item.record.matchFilePath = repository.shardPath(ownerPath);
                    item.record.settings = pairSettings;
                    item.record.settings.remove(QStringLiteral("image_files"));
                    item.record.settings.remove(QStringLiteral("image0_match_file"));
                    item.record.settings.remove(QStringLiteral("image1_match_file"));
                    item.record.settings[QStringLiteral("owner_image")] = ownerPath;
                    index = static_cast<int>(pending.size());
                    indexByImageId.insert(stableId, index);
                    pending.push_back(std::move(item));
                }
                pending[static_cast<std::size_t>(index)].neighbors.insert(peerPath);
            };

            for (const MatchPhotosMatchRecord& pair : pairRecords)
            {
                appendPairSide(pair.image0Path, pair.image1Path, pair.settings);
                appendPairSide(pair.image1Path, pair.image0Path, pair.settings);
            }

            result->imageMatchFiles.clear();
            result->imageMatchFiles.reserve(pending.size());
            for (PendingRecord& item : pending)
            {
                item.record.neighborImagePaths = item.neighbors.values();
                item.record.neighborImagePaths.sort(Qt::CaseInsensitive);
                item.record.neighborVariantCount = item.record.neighborImagePaths.size();
                item.record.settings[QStringLiteral("neighbor_images")] =
                    QJsonArray::fromStringList(item.record.neighborImagePaths);
                item.record.settings[QStringLiteral("neighbor_count")] = item.record.neighborVariantCount;
                result->imageMatchFiles.push_back(std::move(item.record));
            }
        }

    } // namespace

    MatchPhotosStageReport TrackBuildStage::run(const MatchPhotosContext& context,
                                                const MatchPhotosOptions& options,
                                                std::vector<MatchPhotosMatchRecord>* matchRecords,
                                                MatchPhotosResult* result) const
    {
        if (options.planOnly)
        {
            return makeTrackReport(MatchPhotosStageStatus::Skipped,
                                   QStringLiteral("plan-only 模式，跳过轨迹构建和匹配提交"));
        }
        if (!matchRecords || matchRecords->empty())
        {
            return makeTrackReport(MatchPhotosStageStatus::Failed, QStringLiteral("没有可提交的匹配结果"));
        }

        TiePointTrackBuildResult buildResult;
        if (options.enableTrackBuild)
        {
            const TiePointTrackManager manager;
            buildResult = manager.build(context, options, matchRecords);
        }
        else
        {
            buildResult.success = true;
        }

        // 无论轨迹质量门是否通过，都先持久化已完成的原始匹配、几何内点和残差。
        // 这样用户可以在匹配查看器中诊断失败像对，而不会因 SfM 前置质量门丢数据。
        const image_matching::ImageMatchWriteResult writeResult = persistMatches(context, *matchRecords);
        if (!writeResult.success)
        {
            return makeTrackReport(MatchPhotosStageStatus::Failed,
                                   QStringLiteral("写入每影像匹配分片失败：%1").arg(writeResult.errorMessage));
        }

        // 项目层只登记逐影像分片。即使轨迹质量门失败，也保留这些记录供用户检查
        // 原始匹配、几何内点和残差，避免再次扫描目录或解析文件名。
        populateImageMatchRecords(context, *matchRecords, result);

        if (!options.enableTrackBuild)
        {
            return makeTrackReport(MatchPhotosStageStatus::Skipped,
                                   QStringLiteral("轨迹构建已禁用；已提交 %1 对匹配到 %2 个影像分片")
                                       .arg(writeResult.pairCount)
                                       .arg(writeResult.imageCount),
                                   writeResult.pairCount);
        }
        if (!buildResult.success)
        {
            return makeTrackReport(MatchPhotosStageStatus::Failed,
                                   QStringLiteral("%1；诊断匹配已写入 %2 个影像分片")
                                       .arg(buildResult.errorMessage)
                                       .arg(writeResult.imageCount),
                                   buildResult.consumedPairCount);
        }

        if (result)
        {
            result->trackCount = static_cast<int>(buildResult.tracks.size());
            result->acceptedTrackComponents = buildResult.acceptedComponents;
            result->rejectedTrackConflictComponents = 0;
            result->trackSummary = buildResult.trackSummary;
            result->tiePointPath = buildResult.tiePointPath;
        }

        return makeTrackReport(MatchPhotosStageStatus::Completed,
                               QStringLiteral("连接点轨迹完成：track %1，消费匹配对 %2，跳过 %3；"
                                              "已提交 %4 对到 %5 个影像分片")
                                   .arg(static_cast<int>(buildResult.tracks.size()))
                                   .arg(buildResult.consumedPairCount)
                                   .arg(buildResult.skippedPairCount)
                                   .arg(writeResult.pairCount)
                                   .arg(writeResult.imageCount),
                               static_cast<int>(buildResult.tracks.size()));
    }

} // namespace xjw::matchphotos
