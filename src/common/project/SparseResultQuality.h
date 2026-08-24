#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QString>

namespace xjw::common::project
{

inline const QString kSparseResultKindPairwisePreview =
    QStringLiteral("pairwise_triangulation_preview");
inline const QString kSparseResultKindSfmSparseReconstruction =
    QStringLiteral("sfm_sparse_reconstruction");
inline const QString kSparseResultKindSparsePostprocess =
    QStringLiteral("sparse_postprocess");
inline const QString kSparseResultKindUnknown =
    QStringLiteral("unknown_sparse_result");

QJsonObject buildSparseQualityMetadata(const QJsonArray &points,
                                       int cameraCount,
                                       bool baApplied,
                                       const QString &resultKind,
                                       const QString &sourceResultKind = QString(),
                                       const QString &sourceResultRef = QString());

QJsonObject buildSparseQualityMetadata(const QJsonArray &points,
                                       int cameraCount,
                                       bool baApplied,
                                       const QString &resultKind,
                                       const QString &sourceResultKind,
                                       const QString &sourceResultRef,
                                       int inputImageCount);

QJsonObject mergeSparseQualityIntoRecord(const QJsonObject &record,
                                         const QJsonObject &quality);

QString sparseResultKind(const QJsonObject &record);
QString sparseResultKindDisplayName(const QString &resultKind);
bool isPairwisePreviewSparseResult(const QJsonObject &record);
bool isProductionSparseResult(const QJsonObject &record);
bool isStandardMvsCompatibleSparseResult(const QJsonObject &record);
QString sparseResultBlockingReason(const QJsonObject &record);
QString standardMvsBlockingReason(const QJsonObject &record);
QString sparseResultWarningText(const QJsonObject &record);

} // namespace xjw::common::project

namespace xjw::gui::project
{
using xjw::common::project::buildSparseQualityMetadata;
using xjw::common::project::isPairwisePreviewSparseResult;
using xjw::common::project::isProductionSparseResult;
using xjw::common::project::isStandardMvsCompatibleSparseResult;
using xjw::common::project::kSparseResultKindPairwisePreview;
using xjw::common::project::kSparseResultKindSfmSparseReconstruction;
using xjw::common::project::kSparseResultKindSparsePostprocess;
using xjw::common::project::kSparseResultKindUnknown;
using xjw::common::project::mergeSparseQualityIntoRecord;
using xjw::common::project::sparseResultBlockingReason;
using xjw::common::project::sparseResultKind;
using xjw::common::project::sparseResultKindDisplayName;
using xjw::common::project::sparseResultWarningText;
using xjw::common::project::standardMvsBlockingReason;
} // namespace xjw::gui::project
