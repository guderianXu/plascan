#pragma once

#include <QString>

#include <array>
#include <optional>

namespace xjw::camera_reference
{

using Vector3d = std::array<double, 3>;
using Matrix3d = std::array<double, 9>;

struct CameraReferenceSource
{
    QString kind;
    QString displayName;
    QString contentSha256;
    QString sourceCrs;
    QString axisOrder = QStringLiteral("traditional_gis");
    QString verticalDatum;
    QString verticalUnit;
    QString orientationConvention;
    QString angleUnit = QStringLiteral("deg");

    bool operator==(const CameraReferenceSource &) const = default;
};

struct CameraReferenceSolverFrame
{
    QString frameId;
    QString kind;
    QString unit = QStringLiteral("m");
    Vector3d originEcefMeters{{0.0, 0.0, 0.0}};
    Matrix3d rotationSolverToEcef{{1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0}};
    QString targetCrs;
    QString targetCrsWkt;
    QString normalizationHash;

    bool operator==(const CameraReferenceSolverFrame &) const = default;
};

struct CameraReferenceLeverArm
{
    QString id;
    Vector3d vectorMeters{{0.0, 0.0, 0.0}};
    QString vectorFrame;
    QString vectorDirection;
    QString source;

    bool operator==(const CameraReferenceLeverArm &) const = default;
};

struct RawCameraReference
{
    std::optional<Vector3d> position;
    std::optional<Vector3d> positionSigma;
    QString positionSigmaFrame;
    QString positionSigmaUnit;
    std::optional<double> horizontalSigmaMeters;
    std::optional<Vector3d> orientationYprDegrees;
    std::optional<Vector3d> orientationSigmaDegrees;
    QString timestamp;

    bool operator==(const RawCameraReference &) const = default;
};

struct ResolvedCameraReference
{
    std::optional<Vector3d> cameraCenterMeters;
    std::optional<Matrix3d> rotationCameraToWorld;
    std::optional<Vector3d> positionSigmaMeters;
    std::optional<Vector3d> rotationSigmaDegrees;
    bool positionUsable = false;
    bool orientationUsable = false;
    bool leverArmApplied = false;
    QString frameId;
    QString normalizationHash;
    QString error;

    bool operator==(const ResolvedCameraReference &) const = default;
};

struct CameraReferenceRecord
{
    QString imageUuid;
    QString imagePathSnapshot;
    QString sourceLabel;
    QString sensorKey;
    QString leverArmId;
    bool enabled = true;
    RawCameraReference raw;
    ResolvedCameraReference resolved;

    bool operator==(const CameraReferenceRecord &) const = default;
};

struct UnmatchedCameraReferenceRecord
{
    QString sourceLabel;
    QString sensorKey;
    QString leverArmId;
    RawCameraReference raw;
    QString reason;

    bool operator==(const UnmatchedCameraReferenceRecord &) const = default;
};

} // namespace xjw::camera_reference
