#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

#include <optional>

namespace xjw::gui::reference_import
{

// Metashape 相机参考文件中的原始观测。这里仅保存文件语义，
// 不进行坐标系转换、杆臂改正或姿态约定转换。
struct RawCameraReferenceRecord
{
    QString fileName;
    double wgs84LatitudeDegrees = 0.0;
    double wgs84LongitudeDegrees = 0.0;
    double wgs84EllipsoidalHeightMeters = 0.0;
    double rollDegrees = 0.0;
    double pitchDegrees = 0.0;
    double yawDegrees = 0.0;
    QString timeText;
    std::optional<double> stdDevNorthMeters;
    std::optional<double> stdDevEastMeters;
    std::optional<double> stdDevUpMeters;
    std::optional<double> stdDevHorizontalMeters;
};

// GNSS 天线相对于相机的原始 X/Y/Z 偏移。轴向约定由后续配置解释。
struct LeverArm
{
    double xMeters = 0.0;
    double yMeters = 0.0;
    double zMeters = 0.0;
};

struct MetashapeCameraReferenceImportResult
{
    QVector<RawCameraReferenceRecord> records;
    std::optional<LeverArm> leverArm;
    QStringList warnings;
};

/**
 * 读取 Metashape/Agisoft 制表符分隔的相机参考 TXT。
 *
 * cameraTxtPath 必须指向相机参考文件。gnssOffsetTxtPath 为空时不读取杆臂；
 * 非空时，文件必须包含且仅定义一次 X、Y、Z。函数成功后才会更新 result。
 * 失败时 errorMessage 包含文件和物理行号（文件级错误除外）。
 */
bool importMetashapeCameraReferenceTxt(
    const QString &cameraTxtPath,
    const QString &gnssOffsetTxtPath,
    MetashapeCameraReferenceImportResult *result,
    QString *errorMessage);

} // namespace xjw::gui::reference_import
