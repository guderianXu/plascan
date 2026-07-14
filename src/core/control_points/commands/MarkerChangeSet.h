#pragma once

#include "model/MarkerSet.h"

#include <QString>
#include <QVector>

namespace xjw::control_points
{

class MarkerChangeSet
{
public:
    MarkerChangeSet() = default;

    static MarkerChangeSet replaceProjection(const MarkerSet &set,
                                              const MarkerId &markerId,
                                              const QString &imageId,
                                              const MarkerProjection &projection);
    static MarkerChangeSet replaceProjections(const MarkerSet &set,
                                              const MarkerId &markerId,
                                              const QVector<MarkerProjection> &projections,
                                              const QString &description);
    static MarkerChangeSet createMarkerWithProjection(const MarkerSet &set,
                                                      const QString &label,
                                                      MarkerRole role,
                                                      const MarkerProjection &projection);
    static MarkerChangeSet removeProjection(const MarkerSet &set,
                                            const MarkerId &markerId,
                                            const QString &imageId);
    static MarkerChangeSet setProjectionState(const MarkerSet &set,
                                              const MarkerId &markerId,
                                              const QString &imageId,
                                              ProjectionState state,
                                              const QString &description);
    static MarkerChangeSet updateMarker(const MarkerSet &set,
                                        const MarkerId &markerId,
                                        const QString &label,
                                        MarkerRole role,
                                        bool enabled,
                                        const std::optional<ReferenceCoordinate> &referenceCoordinate);
    static MarkerChangeSet replaceMarkerSet(const MarkerSet &before,
                                            const MarkerSet &after,
                                            const QString &description,
                                            const QVector<MarkerId> &affectedMarkerIds);

    void apply(MarkerSet *set) const;
    void revert(MarkerSet *set) const;

    QString description() const;
    QVector<MarkerId> affectedMarkerIds() const;
    bool isValid() const noexcept;

private:
    static MarkerChangeSet fromSnapshots(const MarkerSet &before,
                                         MarkerSet after,
                                         const QString &description,
                                         const QVector<MarkerId> &affectedMarkerIds);

    MarkerSet _before;
    MarkerSet _after;
    QString _description;
    QVector<MarkerId> _affectedMarkerIds;
    bool _valid = false;
};

} // namespace xjw::control_points
