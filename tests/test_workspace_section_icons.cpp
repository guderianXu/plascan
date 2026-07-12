#include <array>

#include <gtest/gtest.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QSet>
#include <QStandardItemModel>
#include <QTreeView>

#include "DataTreeWidget.h"
#include "WorkspaceSectionIcons.h"

namespace xjw::gui::project
{

QString sparseOperationDisplayName(const QString &operation)
{
    return operation;
}

} // namespace xjw::gui::project

namespace
{

QByteArray iconPixelSignature(const QIcon &icon, const QSize &size)
{
    const QImage image = icon.pixmap(size).toImage().convertToFormat(QImage::Format_ARGB32);
    if (image.isNull())
    {
        return {};
    }

    const QByteArray pixels(reinterpret_cast<const char *>(image.constBits()),
                            static_cast<qsizetype>(image.sizeInBytes()));
    return QCryptographicHash::hash(pixels, QCryptographicHash::Sha256);
}

bool iconHasVisiblePixel(const QIcon &icon, const QSize &size)
{
    const QImage image = icon.pixmap(size).toImage().convertToFormat(QImage::Format_ARGB32);
    for (int y = 0; y < image.height(); ++y)
    {
        const QRgb *row = reinterpret_cast<const QRgb *>(image.constScanLine(y));
        for (int x = 0; x < image.width(); ++x)
        {
            if (qAlpha(row[x]) > 0)
            {
                return true;
            }
        }
    }
    return false;
}

xjw::gui::widgets::WorkspaceSection sectionForTitle(const QString &title)
{
    using xjw::gui::widgets::WorkspaceSection;
    if (title.startsWith(QStringLiteral("照片"))) return WorkspaceSection::Photos;
    if (title.startsWith(QStringLiteral("观测网络"))) return WorkspaceSection::ObservationNetwork;
    if (title.startsWith(QStringLiteral("连接点"))) return WorkspaceSection::TiePoints;
    if (title.startsWith(QStringLiteral("深度图"))) return WorkspaceSection::DepthMaps;
    if (title.startsWith(QStringLiteral("稠密点云"))) return WorkspaceSection::DenseCloud;
    if (title.startsWith(QStringLiteral("3D模型"))) return WorkspaceSection::Model3D;
    if (title.startsWith(QStringLiteral("DEM"))) return WorkspaceSection::Dem;
    if (title.startsWith(QStringLiteral("正射影像"))) return WorkspaceSection::Orthomosaic;
    if (title.startsWith(QStringLiteral("参考数据"))) return WorkspaceSection::ReferenceData;
    if (title.startsWith(QStringLiteral("报告"))) return WorkspaceSection::Reports;
    return WorkspaceSection::Unknown;
}

} // namespace

TEST(WorkspaceSectionIconsTest, FactoryRendersEverySectionAtSupportedSizes)
{
    using xjw::gui::widgets::WorkspaceSection;
    const std::array sections = {
        WorkspaceSection::Photos,
        WorkspaceSection::ObservationNetwork,
        WorkspaceSection::TiePoints,
        WorkspaceSection::DepthMaps,
        WorkspaceSection::DenseCloud,
        WorkspaceSection::Model3D,
        WorkspaceSection::Dem,
        WorkspaceSection::Orthomosaic,
        WorkspaceSection::ReferenceData,
        WorkspaceSection::Reports,
        WorkspaceSection::Unknown
    };

    for (const WorkspaceSection section : sections)
    {
        const QIcon icon = xjw::gui::widgets::workspaceSectionIcon(section);
        ASSERT_FALSE(icon.isNull());
        for (const int size : {16, 18, 20, 24, 32, 36, 48, 64})
        {
            EXPECT_TRUE(iconHasVisiblePixel(icon, QSize(size, size)))
                << "section=" << static_cast<int>(section) << ", size=" << size;
        }
    }
}

TEST(WorkspaceSectionIconsTest, VisibleSectionsUseDistinctSemanticIcons)
{
    DataTreeWidget tree;
    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{QStringLiteral("/tmp/image.tif")};
    meta[QStringLiteral("observation_network_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("algorithm"), QStringLiteral("overlap")}}
    };
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
        QJsonObject{
            {QStringLiteral("sparse_point_count"), 2314},
            {QStringLiteral("files"), QJsonObject{
                {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("/tmp/sparse.xyz")}
            }}
        }
    };
    meta[QStringLiteral("depth_map_results")] = QJsonArray{
        QJsonObject{
            {QStringLiteral("result_type"), QStringLiteral("mvs_depth")},
            {QStringLiteral("depth_png"), QStringLiteral("/tmp/depth.png")}
        }
    };
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("dense_cloud_xyz"), QStringLiteral("/tmp/dense.ply")}}
    };
    meta[QStringLiteral("model_results")] = QJsonArray{
        QJsonObject{
            {QStringLiteral("model_ply"), QStringLiteral("/tmp/model.ply")},
            {QStringLiteral("vertex_count"), 8},
            {QStringLiteral("face_count"), 12}
        }
    };
    meta[QStringLiteral("dem_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("dem_tif"), QStringLiteral("/tmp/dem.tif")}}
    };
    meta[QStringLiteral("ortho_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("output_path"), QStringLiteral("/tmp/ortho.tif")}}
    };
    meta[QStringLiteral("reference_datasets")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/reference.tif")}}
    };
    meta[QStringLiteral("report_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/report.json")}}
    };
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->iconSize(), QSize(18, 18));
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->rowCount(), 10);

    QSet<QByteArray> signatures;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *section = model->item(row, 0);
        ASSERT_NE(section, nullptr);
        ASSERT_FALSE(section->icon().isNull()) << section->text().toStdString();
        const QByteArray signature = iconPixelSignature(section->icon(), QSize(18, 18));
        ASSERT_FALSE(signature.isEmpty()) << section->text().toStdString();
        EXPECT_TRUE(iconHasVisiblePixel(section->icon(), QSize(18, 18)))
            << section->text().toStdString();

        const auto expectedSection = sectionForTitle(section->text());
        ASSERT_NE(expectedSection, xjw::gui::widgets::WorkspaceSection::Unknown)
            << section->text().toStdString();
        EXPECT_EQ(signature,
                  iconPixelSignature(xjw::gui::widgets::workspaceSectionIcon(expectedSection),
                                     QSize(18, 18)))
            << section->text().toStdString();
        signatures.insert(signature);
    }
    EXPECT_EQ(signatures.size(), model->rowCount());
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
