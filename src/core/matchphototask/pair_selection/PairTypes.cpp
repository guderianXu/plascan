#include "PairTypes.h"

#include <QDir>
#include <QFileInfo>

#include <algorithm>

namespace xjw
{
    namespace matchphotos
    {

        bool ImagePair::isValid(int imageCount) const
        {
            return indexA >= 0 && indexB >= 0 && indexA < imageCount && indexB < imageCount && indexA != indexB;
        }

        ImagePair ImagePair::normalized() const
        {
            if (indexA <= indexB)
            {
                return *this;
            }
            return {indexB, indexA};
        }

        QString pairSourceId(PairSource source)
        {
            switch (source)
            {
            case PairSource::Manual:
                return QStringLiteral("manual");
            case PairSource::Exhaustive:
                return QStringLiteral("exhaustive");
            case PairSource::SequenceWindow:
                return QStringLiteral("sequence_window");
            case PairSource::PlaMatchGeneric:
                return QStringLiteral("plamatch_generic");
            case PairSource::PlaMatchReferenceSource:
                return QStringLiteral("plamatch_reference_source");
            case PairSource::PlaMatchReferenceEstimated:
                return QStringLiteral("plamatch_reference_estimated");
            case PairSource::PlaMatchReferenceSequential:
                return QStringLiteral("plamatch_reference_sequential");
            case PairSource::GuidedRematch:
                return QStringLiteral("guided_rematch");
            }
            return QStringLiteral("unknown");
        }

        QString pairSourceDisplayName(PairSource source)
        {
            switch (source)
            {
            case PairSource::Manual:
                return QStringLiteral("手动指定");
            case PairSource::Exhaustive:
                return QStringLiteral("全量匹配");
            case PairSource::SequenceWindow:
                return QStringLiteral("序列窗口");
            case PairSource::PlaMatchGeneric:
                return QStringLiteral("PlaMatch 通用预选");
            case PairSource::PlaMatchReferenceSource:
                return QStringLiteral("PlaMatch 源参考预选");
            case PairSource::PlaMatchReferenceEstimated:
                return QStringLiteral("PlaMatch 估计参考预选");
            case PairSource::PlaMatchReferenceSequential:
                return QStringLiteral("PlaMatch 序列参考预选");
            case PairSource::GuidedRematch:
                return QStringLiteral("引导重匹配");
            }
            return QStringLiteral("未知来源");
        }

        QString canonicalImagePath(const QString& path)
        {
            const QString trimmed = path.trimmed();
            if (trimmed.isEmpty())
            {
                return QString();
            }
            return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
        }

        QString makePairKey(const QString& pathA, const QString& pathB)
        {
            const QString normA = canonicalImagePath(pathA);
            const QString normB = canonicalImagePath(pathB);
            if (normA.isEmpty() || normB.isEmpty() || normA == normB)
            {
                return QString();
            }
            return normA < normB ? normA + QStringLiteral("\n") + normB : normB + QStringLiteral("\n") + normA;
        }

        QString makePairKey(const QStringList& images, int indexA, int indexB)
        {
            if (indexA < 0 || indexB < 0 || indexA >= images.size() || indexB >= images.size() || indexA == indexB)
            {
                return QString();
            }
            return makePairKey(images.at(indexA), images.at(indexB));
        }

        void appendPairSource(PairCandidate* candidate, PairSource source)
        {
            if (!candidate)
            {
                return;
            }

            const auto found = std::find(candidate->sources.begin(), candidate->sources.end(), source);
            if (found == candidate->sources.end())
            {
                candidate->sources.push_back(source);
            }
        }

    } // namespace matchphotos
} // namespace xjw
