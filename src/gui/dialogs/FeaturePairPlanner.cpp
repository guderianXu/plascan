#include "FeaturePairPlanner.h"

#include <algorithm>

namespace xjw::gui
{

QStringList planFeatureMatchPairs(const QStringList &imageBaseNames,
                                  const FeaturePairPlannerOptions &options)
{
    QStringList names;
    names.reserve(imageBaseNames.size());
    for (const QString &name : imageBaseNames)
    {
        const QString trimmed = name.trimmed();
        if (!trimmed.isEmpty())
        {
            names.append(trimmed);
        }
    }

    const int count = names.size();
    if (count < 2)
    {
        return {};
    }

    const bool exhaustive = count <= std::max(2, options.exhaustiveMaxImages);
    const int window = exhaustive ? (count - 1) : std::max(1, options.sequentialWindow);

    QStringList pairs;
    for (int i = 0; i < count; ++i)
    {
        const int end = std::min(count, i + window + 1);
        for (int j = i + 1; j < end; ++j)
        {
            pairs.append(QStringLiteral("%1__%2").arg(names.at(i), names.at(j)));
        }
    }
    return pairs;
}

} // namespace xjw::gui
