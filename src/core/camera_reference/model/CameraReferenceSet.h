#pragma once

#include "CameraReferenceTypes.h"

#include <QDateTime>
#include <QVector>

#include <stdexcept>

namespace xjw::camera_reference
{

class CameraReferenceModelError : public std::runtime_error
{
public:
    explicit CameraReferenceModelError(const QString &message)
        : std::runtime_error(message.toStdString())
    {
    }
};

class CameraReferenceSet
{
public:
    static constexpr int CurrentSchemaVersion = 1;

    CameraReferenceSet();

    int schemaVersion() const noexcept;
    QDateTime createdAt() const;
    QDateTime updatedAt() const;
    QString imageSetFingerprint() const;
    const CameraReferenceSource &source() const noexcept;
    const CameraReferenceSolverFrame &solverFrame() const noexcept;
    const QVector<CameraReferenceLeverArm> &leverArms() const noexcept;
    const QVector<CameraReferenceRecord> &records() const noexcept;
    const QVector<UnmatchedCameraReferenceRecord> &unmatchedRecords() const noexcept;

    void setImageSetFingerprint(const QString &fingerprint);
    void replaceSource(const CameraReferenceSource &source);
    void replaceSolverFrame(const CameraReferenceSolverFrame &solverFrame);
    void addLeverArm(const CameraReferenceLeverArm &leverArm);
    void replaceLeverArm(const CameraReferenceLeverArm &leverArm);
    void replaceLeverArms(const QVector<CameraReferenceLeverArm> &leverArms);
    void addRecord(const CameraReferenceRecord &record);
    void replaceRecord(const CameraReferenceRecord &record);
    void replaceRecords(const QVector<CameraReferenceRecord> &records);
    void addUnmatchedRecord(const UnmatchedCameraReferenceRecord &record);
    void replaceUnmatchedRecords(const QVector<UnmatchedCameraReferenceRecord> &records);

    bool validate(QString *error = nullptr) const noexcept;
    bool operator==(const CameraReferenceSet &other) const;

private:
    friend class CameraReferenceSetJson;

    void touch();
    void validateOrThrow() const;

    int _schemaVersion = CurrentSchemaVersion;
    QDateTime _createdAt;
    QDateTime _updatedAt;
    QString _imageSetFingerprint;
    CameraReferenceSource _source;
    CameraReferenceSolverFrame _solverFrame;
    QVector<CameraReferenceLeverArm> _leverArms;
    QVector<CameraReferenceRecord> _records;
    QVector<UnmatchedCameraReferenceRecord> _unmatchedRecords;
};

} // namespace xjw::camera_reference
