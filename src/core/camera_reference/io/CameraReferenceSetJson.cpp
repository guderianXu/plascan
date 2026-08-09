#include "CameraReferenceSetJson.h"

#include <QJsonArray>

namespace xjw::camera_reference
{
namespace
{

template <std::size_t Size>
QJsonArray encodeArray(const std::array<double, Size> &values)
{
    QJsonArray array;
    for (double value : values)
    {
        array.append(value);
    }
    return array;
}

template <std::size_t Size>
QJsonValue encodeOptionalArray(const std::optional<std::array<double, Size>> &values)
{
    return values ? QJsonValue(encodeArray(*values)) : QJsonValue(QJsonValue::Null);
}

QJsonObject encodeSource(const CameraReferenceSource &source)
{
    return {
        {QStringLiteral("kind"), source.kind},
        {QStringLiteral("display_name"), source.displayName},
        {QStringLiteral("content_sha256"), source.contentSha256},
        {QStringLiteral("source_crs"), source.sourceCrs},
        {QStringLiteral("axis_order"), source.axisOrder},
        {QStringLiteral("vertical_datum"), source.verticalDatum},
        {QStringLiteral("vertical_unit"), source.verticalUnit},
        {QStringLiteral("orientation_convention"), source.orientationConvention},
        {QStringLiteral("angle_unit"), source.angleUnit}
    };
}

QJsonObject encodeSolverFrame(const CameraReferenceSolverFrame &frame)
{
    return {
        {QStringLiteral("frame_id"), frame.frameId},
        {QStringLiteral("kind"), frame.kind},
        {QStringLiteral("unit"), frame.unit},
        {QStringLiteral("origin_ecef_m"), encodeArray(frame.originEcefMeters)},
        {QStringLiteral("rotation_solver_to_ecef"), encodeArray(frame.rotationSolverToEcef)},
        {QStringLiteral("target_crs"), frame.targetCrs},
        {QStringLiteral("target_crs_wkt"), frame.targetCrsWkt},
        {QStringLiteral("normalization_hash"), frame.normalizationHash}
    };
}

QJsonObject encodeRawReference(const RawCameraReference &reference)
{
    return {
        {QStringLiteral("position"), encodeOptionalArray(reference.position)},
        {QStringLiteral("position_sigma"), encodeOptionalArray(reference.positionSigma)},
        {QStringLiteral("position_sigma_frame"), reference.positionSigmaFrame},
        {QStringLiteral("position_sigma_unit"), reference.positionSigmaUnit},
        {QStringLiteral("horizontal_sigma_m"),
         reference.horizontalSigmaMeters
             ? QJsonValue(*reference.horizontalSigmaMeters)
             : QJsonValue(QJsonValue::Null)},
        {QStringLiteral("orientation_ypr_deg"), encodeOptionalArray(reference.orientationYprDegrees)},
        {QStringLiteral("orientation_sigma_deg"), encodeOptionalArray(reference.orientationSigmaDegrees)},
        {QStringLiteral("timestamp"), reference.timestamp}
    };
}

QJsonObject encodeResolvedReference(const ResolvedCameraReference &reference)
{
    return {
        {QStringLiteral("camera_center_m"), encodeOptionalArray(reference.cameraCenterMeters)},
        {QStringLiteral("rotation_camera_to_world"), encodeOptionalArray(reference.rotationCameraToWorld)},
        {QStringLiteral("position_sigma_m"), encodeOptionalArray(reference.positionSigmaMeters)},
        {QStringLiteral("rotation_sigma_deg"), encodeOptionalArray(reference.rotationSigmaDegrees)},
        {QStringLiteral("position_usable"), reference.positionUsable},
        {QStringLiteral("orientation_usable"), reference.orientationUsable},
        {QStringLiteral("lever_arm_applied"), reference.leverArmApplied},
        {QStringLiteral("frame_id"), reference.frameId},
        {QStringLiteral("normalization_hash"), reference.normalizationHash},
        {QStringLiteral("error"), reference.error}
    };
}

} // namespace

QJsonObject CameraReferenceSetJson::encode(const CameraReferenceSet &referenceSet)
{
    QJsonArray leverArms;
    for (const CameraReferenceLeverArm &leverArm : referenceSet._leverArms)
    {
        leverArms.append(QJsonObject{
            {QStringLiteral("id"), leverArm.id},
            {QStringLiteral("vector_m"), encodeArray(leverArm.vectorMeters)},
            {QStringLiteral("vector_frame"), leverArm.vectorFrame},
            {QStringLiteral("vector_direction"), leverArm.vectorDirection},
            {QStringLiteral("source"), leverArm.source}
        });
    }

    QJsonArray records;
    for (const CameraReferenceRecord &record : referenceSet._records)
    {
        records.append(QJsonObject{
            {QStringLiteral("image_uuid"), record.imageUuid},
            {QStringLiteral("image_path_snapshot"), record.imagePathSnapshot},
            {QStringLiteral("source_label"), record.sourceLabel},
            {QStringLiteral("sensor_key"), record.sensorKey},
            {QStringLiteral("lever_arm_id"), record.leverArmId},
            {QStringLiteral("enabled"), record.enabled},
            {QStringLiteral("raw"), encodeRawReference(record.raw)},
            {QStringLiteral("resolved"), encodeResolvedReference(record.resolved)}
        });
    }

    QJsonArray unmatchedRecords;
    for (const UnmatchedCameraReferenceRecord &record : referenceSet._unmatchedRecords)
    {
        unmatchedRecords.append(QJsonObject{
            {QStringLiteral("source_label"), record.sourceLabel},
            {QStringLiteral("sensor_key"), record.sensorKey},
            {QStringLiteral("lever_arm_id"), record.leverArmId},
            {QStringLiteral("raw"), encodeRawReference(record.raw)},
            {QStringLiteral("reason"), record.reason}
        });
    }

    return {
        {QStringLiteral("format"), QString::fromLatin1(Format)},
        {QStringLiteral("schema_version"), referenceSet._schemaVersion},
        {QStringLiteral("created_at"), referenceSet._createdAt.toString(Qt::ISODateWithMs)},
        {QStringLiteral("updated_at"), referenceSet._updatedAt.toString(Qt::ISODateWithMs)},
        {QStringLiteral("image_set_fingerprint"), referenceSet._imageSetFingerprint},
        {QStringLiteral("source"), encodeSource(referenceSet._source)},
        {QStringLiteral("solver_frame"), encodeSolverFrame(referenceSet._solverFrame)},
        {QStringLiteral("lever_arms"), leverArms},
        {QStringLiteral("records"), records},
        {QStringLiteral("unmatched_source_records"), unmatchedRecords}
    };
}

} // namespace xjw::camera_reference
