#include "project/AspMatchPointReader.h"

#include <QFile>
#include <QTemporaryDir>
#include <QtEndian>

#include <gtest/gtest.h>

#include <cstring>

namespace
{

template <typename Integer>
void appendLittleEndian(QByteArray *bytes, Integer value)
{
    const Integer encoded = qToLittleEndian(value);
    bytes->append(reinterpret_cast<const char *>(&encoded), sizeof(Integer));
}

void appendFloat(QByteArray *bytes, float value)
{
    quint32 bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    appendLittleEndian(bytes, bits);
}

void appendPoint(QByteArray *bytes, float x, float y)
{
    appendFloat(bytes, x);
    appendFloat(bytes, y);
    appendLittleEndian<qint32>(bytes, static_cast<qint32>(x));
    appendLittleEndian<qint32>(bytes, static_cast<qint32>(y));
    appendFloat(bytes, 0.0f);
    appendFloat(bytes, 1.0f);
    appendFloat(bytes, 1.0f);
    bytes->append('\0');
    appendLittleEndian<quint32>(bytes, 0);
    appendLittleEndian<quint32>(bytes, 0);
    appendLittleEndian<quint64>(bytes, 2);
    appendFloat(bytes, 0.25f);
    appendFloat(bytes, 0.75f);
}

} // namespace

TEST(AspMatchPointReaderTest, ReadsCoordinatesAndSkipsDescriptors)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());

    QByteArray bytes;
    appendLittleEndian<quint64>(&bytes, 2);
    appendLittleEndian<quint64>(&bytes, 2);
    appendPoint(&bytes, 10.0f, 20.0f);
    appendPoint(&bytes, 30.0f, 40.0f);
    appendPoint(&bytes, 11.0f, 21.0f);
    appendPoint(&bytes, 31.0f, 41.0f);

    const QString path = temporaryDirectory.filePath(QStringLiteral("pair.match"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_EQ(file.write(bytes), bytes.size());
    file.close();

    const auto result = xjw::common::project::readAspMatchPoints(path);
    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    ASSERT_EQ(result.leftPoints.size(), 2);
    ASSERT_EQ(result.rightPoints.size(), 2);
    EXPECT_EQ(result.leftPoints.at(0), QPointF(10.0, 20.0));
    EXPECT_EQ(result.rightPoints.at(1), QPointF(31.0, 41.0));
}

TEST(AspMatchPointReaderTest, RejectsTruncatedPointData)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());

    QByteArray bytes;
    appendLittleEndian<quint64>(&bytes, 1);
    appendLittleEndian<quint64>(&bytes, 1);
    appendFloat(&bytes, 10.0f);

    const QString path = temporaryDirectory.filePath(QStringLiteral("truncated.match"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_EQ(file.write(bytes), bytes.size());
    file.close();

    const auto result = xjw::common::project::readAspMatchPoints(path);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.isEmpty());
}
