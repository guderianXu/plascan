#pragma once

#include "MarkerTypes.h"

#include <QDateTime>
#include <QVector>

namespace xjw::control_points
{

class MarkerSet
{
public:
    MarkerSet();

    MarkerId addMarker(const QString &label, MarkerRole role);
    void removeMarker(const MarkerId &id);
    void renameMarker(const MarkerId &id, const QString &label);
    void setMarkerRole(const MarkerId &id, MarkerRole role);
    void setMarkerEnabled(const MarkerId &id, bool enabled);
    void setReferenceCoordinate(const MarkerId &id, const ReferenceCoordinate &coordinate);
    void clearReferenceCoordinate(const MarkerId &id);
    void setTargetIdentity(const MarkerId &id, const TargetIdentity &identity);
    void clearTargetIdentity(const MarkerId &id);

    void upsertProjection(const MarkerId &id, const MarkerProjection &projection);
    void removeProjection(const MarkerId &id, const QString &imageId);

    ScaleBarId addScaleBar(const QString &label,
                           const MarkerId &firstMarkerId,
                           const MarkerId &secondMarkerId,
                           double measuredDistance,
                           double sigma,
                           ScaleBarRole role = ScaleBarRole::Control);

    const Marker &marker(const MarkerId &id) const;

    const QVector<Marker> &markers() const noexcept;
    const QVector<ScaleBar> &scaleBars() const noexcept;
    int schemaVersion() const noexcept;
    QString projectImageRevision() const;
    QDateTime createdAt() const;
    QDateTime updatedAt() const;

    bool operator==(const MarkerSet &other) const;

private:
    friend class MarkerSetJson;

    Marker &mutableMarker(const MarkerId &id);
    void validateUniqueLabel(const QString &label, const MarkerId &ignoredId = {}) const;

    int _schemaVersion = 1;
    QString _projectImageRevision;
    QVector<Marker> _markers;
    QVector<ScaleBar> _scaleBars;
    QDateTime _createdAt;
    QDateTime _updatedAt;
};

} // namespace xjw::control_points
