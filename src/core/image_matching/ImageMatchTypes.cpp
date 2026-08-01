#include "ImageMatchTypes.h"

#include <algorithm>

namespace xjw::image_matching
{

bool ImageIdentity::isValid() const
{
    return !stableId.trimmed().isEmpty() && !path.trimmed().isEmpty();
}

bool NeighborMatchBlock::isCompatible(
    const QString &requestedAlgorithmId,
    std::uint32_t requestedAlgorithmVersion,
    const QByteArray &requestedConfigFingerprint,
    const QByteArray &requestedModelFingerprint) const
{
    return algorithmId.compare(requestedAlgorithmId, Qt::CaseInsensitive) == 0 &&
        algorithmVersion == requestedAlgorithmVersion &&
        configFingerprint == requestedConfigFingerprint &&
        (requestedModelFingerprint.isEmpty() ||
         modelFingerprint == requestedModelFingerprint);
}

const KeypointObservation *NeighborMatchBlock::findOwnerObservation(
    std::uint32_t featureId) const
{
    const auto it = std::lower_bound(
        ownerObservations.begin(), ownerObservations.end(), featureId,
        [](const KeypointObservation &observation, std::uint32_t id)
        {
            return observation.featureId < id;
        });
    return it != ownerObservations.end() && it->featureId == featureId ? &(*it) : nullptr;
}

void NeighborMatchBlock::normalize()
{
    std::sort(ownerObservations.begin(), ownerObservations.end(),
              [](const KeypointObservation &left, const KeypointObservation &right)
              {
                  return left.featureId < right.featureId;
              });
    ownerObservations.erase(
        std::unique(ownerObservations.begin(), ownerObservations.end(),
                    [](const KeypointObservation &left, const KeypointObservation &right)
                    {
                        return left.featureId == right.featureId;
                    }),
        ownerObservations.end());

    std::sort(matches.begin(), matches.end(),
              [](const MatchRecord &left, const MatchRecord &right)
              {
                  return left.ownerFeatureId == right.ownerFeatureId
                      ? left.peerFeatureId < right.peerFeatureId
                      : left.ownerFeatureId < right.ownerFeatureId;
              });
    matches.erase(
        std::unique(matches.begin(), matches.end(),
                    [](const MatchRecord &left, const MatchRecord &right)
                    {
                        return left.ownerFeatureId == right.ownerFeatureId &&
                            left.peerFeatureId == right.peerFeatureId;
                    }),
        matches.end());
}

const NeighborMatchBlock *ImageMatchShard::findNeighbor(
    const QString &peerStableId,
    const QString &algorithmId,
    std::uint32_t algorithmVersion,
    const QByteArray &configFingerprint,
    const QByteArray &modelFingerprint) const
{
    const auto it = std::find_if(
        neighbors.begin(), neighbors.end(),
        [&](const NeighborMatchBlock &block)
        {
            return block.peer.stableId == peerStableId &&
                block.isCompatible(
                    algorithmId, algorithmVersion, configFingerprint, modelFingerprint);
        });
    return it == neighbors.end() ? nullptr : &(*it);
}

void ImageMatchShard::normalize()
{
    for (NeighborMatchBlock &block : neighbors)
    {
        block.normalize();
    }

    std::sort(neighbors.begin(), neighbors.end(),
              [](const NeighborMatchBlock &left, const NeighborMatchBlock &right)
              {
                  if (left.peer.stableId != right.peer.stableId)
                  {
                      return left.peer.stableId < right.peer.stableId;
                  }
                   const int algorithmOrder = left.algorithmId.compare(
                       right.algorithmId, Qt::CaseInsensitive);
                   if (algorithmOrder != 0)
                   {
                       return algorithmOrder < 0;
                   }
                  if (left.algorithmVersion != right.algorithmVersion)
                  {
                      return left.algorithmVersion < right.algorithmVersion;
                  }
                   if (left.configFingerprint != right.configFingerprint)
                   {
                       return left.configFingerprint < right.configFingerprint;
                   }
                   return left.modelFingerprint < right.modelFingerprint;
               });
    neighbors.erase(
        std::unique(neighbors.begin(), neighbors.end(),
                    [](const NeighborMatchBlock &left, const NeighborMatchBlock &right)
                    {
                        return left.peer.stableId == right.peer.stableId &&
                            left.algorithmId.compare(
                                right.algorithmId, Qt::CaseInsensitive) == 0 &&
                            left.algorithmVersion == right.algorithmVersion &&
                            left.configFingerprint == right.configFingerprint &&
                            left.modelFingerprint == right.modelFingerprint;
                    }),
        neighbors.end());
}

} // namespace xjw::image_matching
