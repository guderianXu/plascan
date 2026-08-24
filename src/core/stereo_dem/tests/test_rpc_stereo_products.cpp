#include "RpcDomGenerator.h"
#include "RpcStereoDemGenerator.h"

#include "DemDomIO.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonObject>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

    QString testImage(const QString& name)
    {
        return QDir(QString::fromUtf8(TEST_DATA_DIR)).filePath(QStringLiteral("rpc_stereo_pair/Images/") + name);
    }

} // namespace

TEST(RpcStereoProducts, RejectsMissingInputs)
{
    QTemporaryDir output;
    ASSERT_TRUE(output.isValid());
    QJsonObject result;
    QString error;
    EXPECT_FALSE(xjw::RpcStereoDemGenerator::generate(QStringLiteral("missing-left.tif"),
                                                      QStringLiteral("missing-right.tif"),
                                                      output.path(),
                                                      xjw::RpcStereoDemOptions{},
                                                      &result,
                                                      &error));
    EXPECT_FALSE(error.isEmpty());
}

TEST(RpcStereoProducts, GeneratesGeoreferencedDemAndDomFromRepositoryPair)
{
    const QString left = testImage(QStringLiteral("img_01.tif"));
    const QString right = testImage(QStringLiteral("img_02.tif"));
    ASSERT_TRUE(QFileInfo::exists(left));
    ASSERT_TRUE(QFileInfo::exists(right));
    QTemporaryDir output;
    ASSERT_TRUE(output.isValid());

    xjw::RpcStereoDemOptions demOptions;
    demOptions.maximumFeatures = 10000;
    demOptions.minimumAcceptedPoints = 50;
    demOptions.gridResolutionMeters = 5.0;
    QJsonObject demResult;
    QString error;
    ASSERT_TRUE(xjw::RpcStereoDemGenerator::generate(left, right, output.path(), demOptions, &demResult, &error))
        << error.toStdString();

    const QString demPath = demResult.value(QStringLiteral("dem_path")).toString();
    EXPECT_TRUE(QFileInfo::exists(demPath));
    EXPECT_GT(demResult.value(QStringLiteral("accepted_stereo_points")).toInteger(), 100);
    EXPECT_TRUE(demResult.value(QStringLiteral("coordinate_system")).toString().startsWith("EPSG:327"));

    xjw::DemGridData dem;
    ASSERT_TRUE(xjw::DemDomIO::readDemRaster(demPath, &dem, &error)) << error.toStdString();
    EXPECT_TRUE(dem.isValid());
    EXPECT_FALSE(dem.projection.projectionWkt.isEmpty());

    const QString domPath = QDir(output.path()).filePath(QStringLiteral("dom.tif"));
    QJsonObject domResult;
    ASSERT_TRUE(
        xjw::RpcDomGenerator::generate({left, right}, demPath, domPath, xjw::RpcDomOptions{}, &domResult, &error))
        << error.toStdString();
    EXPECT_TRUE(QFileInfo::exists(domPath));
    EXPECT_GT(domResult.value(QStringLiteral("valid_pixels")).toInteger(), 0);
    EXPECT_GT(domResult.value(QStringLiteral("coverage_fraction")).toDouble(), 0.25);
}
