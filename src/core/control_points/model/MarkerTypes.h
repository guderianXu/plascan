#pragma once

#include "model/MarkerRoles.h"

#include <QPointF>
#include <QString>
#include <QVector>

#include <limits>
#include <optional>
#include <stdexcept>

namespace xjw::control_points
{

using MarkerId = QString;
using ScaleBarId = QString;

class MarkerModelError final : public std::runtime_error
{
public:
    explicit MarkerModelError(const QString &message)
        : std::runtime_error(message.toStdString())
    {
    }
};

struct MarkerProjection
{
    QString imageId;
    QString imagePathSnapshot;
    QPointF xy;
    ProjectionState state = ProjectionState::Predicted;
    double sigmaPx = 1.0;
    double confidence = 0.0;
    double residualPx = std::numeric_limits<double>::quiet_NaN();
    QString source;
    QString imageContentSignature;
};

bool operator==(const MarkerProjection &left, const MarkerProjection &right);

struct ReferenceCoordinate
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double sigmaX = 1.0;
    double sigmaY = 1.0;
    double sigmaZ = 1.0;
    QString sourceCrs;
    QString axisOrder = QStringLiteral("traditional_gis");
    QString verticalDatum;
    QString verticalUnit;
    bool referenceUsable = false;
    QString referenceError;
};

bool operator==(const ReferenceCoordinate &left, const ReferenceCoordinate &right);

struct TargetIdentity
{
    QString family;
    int encodedId = -1;
    double rotationDegrees = 0.0;
    QString generationSource;
};

bool operator==(const TargetIdentity &left, const TargetIdentity &right);

struct Marker
{
    MarkerId id;
    QString label;
    MarkerRole role = MarkerRole::TieMarker;
    bool enabled = true;
    std::optional<ReferenceCoordinate> referenceCoordinate;
    std::optional<TargetIdentity> targetIdentity;
    QVector<MarkerProjection> projections;

    const MarkerProjection &projection(const QString &imageId) const;
    MarkerProjection &projection(const QString &imageId);
};

bool operator==(const Marker &left, const Marker &right);

struct ScaleBar
{
    ScaleBarId id;
    QString label;
    MarkerId firstMarkerId;
    MarkerId secondMarkerId;
    ScaleBarRole role = ScaleBarRole::Control;
    bool enabled = true;
    double measuredDistance = 0.0;
    double sigma = 1.0;
    double estimatedDistance = std::numeric_limits<double>::quiet_NaN();
    double residual = std::numeric_limits<double>::quiet_NaN();
};

bool operator==(const ScaleBar &left, const ScaleBar &right);

inline bool markerRoleUsesReferenceConstraint(MarkerRole role)
{
    return role == MarkerRole::ControlPoint;
}

inline bool projectionParticipatesInAdjustment(ProjectionState state)
{
    return state == ProjectionState::ManualPinned || state == ProjectionState::AutoDetected;
}

} // namespace xjw::control_points
