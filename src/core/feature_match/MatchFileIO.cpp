#include "MatchFileIO.h"

#include <QDataStream>
#include <QFile>

#include <cstring>

namespace xjw::feature_match
{

namespace
{

void configureIndexedStream(QDataStream &stream)
{
    stream.setVersion(QDataStream::Qt_5_15);
    stream.setByteOrder(QDataStream::BigEndian);
    stream.setFloatingPointPrecision(QDataStream::DoublePrecision);
}

float scoreAt(const std::vector<float> &scores, size_t index)
{
    return index < scores.size() ? scores[index] : 0.0f;
}

} // namespace

int writeMatchFile(const QString &path,
                   const MatchResult &matchResult,
                   const std::vector<cv::KeyPoint> &keypoints0,
                   const std::vector<cv::KeyPoint> &keypoints1,
                   QByteArray *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        if (errorMessage)
        {
            *errorMessage = file.errorString().toUtf8();
        }
        return -1;
    }

    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::BigEndian);

    qint32 count = 0;
    for (size_t index = 0; index < matchResult.matches0.size(); ++index)
    {
        if (matchResult.matches0[index] >= 0 &&
            index < keypoints0.size() &&
            matchResult.matches0[index] < static_cast<int>(keypoints1.size()))
        {
            ++count;
        }
    }

    stream << count;
    for (size_t index = 0; index < matchResult.matches0.size(); ++index)
    {
        const int matchIndex = matchResult.matches0[index];
        if (matchIndex < 0 ||
            index >= keypoints0.size() ||
            matchIndex >= static_cast<int>(keypoints1.size()))
        {
            continue;
        }

        stream << keypoints0[index].pt.x << keypoints0[index].pt.y
               << keypoints1[static_cast<size_t>(matchIndex)].pt.x
               << keypoints1[static_cast<size_t>(matchIndex)].pt.y;
    }
    return count;
}

bool writeIndexedMatchFile(const QString &path,
                           const QString &image0Name,
                           const QString &image1Name,
                           const MatchResult &result)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        return false;
    }

    QDataStream out(&file);
    configureIndexedStream(out);

    out.writeRawData("SGMT", 4);
    out << quint32(1);

    const QByteArray image0Bytes = image0Name.toUtf8();
    const QByteArray image1Bytes = image1Name.toUtf8();
    out << quint32(image0Bytes.size());
    out.writeRawData(image0Bytes.constData(), image0Bytes.size());
    out << quint32(image1Bytes.size());
    out.writeRawData(image1Bytes.constData(), image1Bytes.size());

    out << qint32(result.numMatches);
    out << qint32(result.matches0.size());
    out << qint32(result.matches1.size());

    for (size_t i = 0; i < result.matches0.size(); ++i)
    {
        out << qint32(result.matches0[i]);
        out << scoreAt(result.matchingScores0, i);
    }

    for (size_t i = 0; i < result.matches1.size(); ++i)
    {
        out << qint32(result.matches1[i]);
        out << scoreAt(result.matchingScores1, i);
    }

    return out.status() == QDataStream::Ok;
}

bool readIndexedMatchFile(const QString &path,
                          QString &image0Name,
                          QString &image1Name,
                          MatchResult &result)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }

    QDataStream in(&file);
    configureIndexedStream(in);

    char magic[4] = {};
    if (in.readRawData(magic, 4) != 4 || std::strncmp(magic, "SGMT", 4) != 0)
    {
        return false;
    }

    quint32 version = 0;
    in >> version;
    if (version != 1)
    {
        return false;
    }

    quint32 image0Length = 0;
    quint32 image1Length = 0;
    in >> image0Length;
    QByteArray image0Bytes(static_cast<int>(image0Length), 0);
    if (in.readRawData(image0Bytes.data(), static_cast<int>(image0Length)) != static_cast<int>(image0Length))
    {
        return false;
    }
    image0Name = QString::fromUtf8(image0Bytes);

    in >> image1Length;
    QByteArray image1Bytes(static_cast<int>(image1Length), 0);
    if (in.readRawData(image1Bytes.data(), static_cast<int>(image1Length)) != static_cast<int>(image1Length))
    {
        return false;
    }
    image1Name = QString::fromUtf8(image1Bytes);

    qint32 numMatches = 0;
    qint32 numKeypoints0 = 0;
    qint32 numKeypoints1 = 0;
    in >> numMatches >> numKeypoints0 >> numKeypoints1;
    if (numMatches < 0 || numKeypoints0 < 0 || numKeypoints1 < 0)
    {
        return false;
    }

    result = MatchResult();
    result.numMatches = numMatches;
    result.matches0.resize(static_cast<size_t>(numKeypoints0));
    result.matchingScores0.resize(static_cast<size_t>(numKeypoints0));
    result.matches1.resize(static_cast<size_t>(numKeypoints1));
    result.matchingScores1.resize(static_cast<size_t>(numKeypoints1));

    for (int i = 0; i < numKeypoints0; ++i)
    {
        qint32 matchIndex = -1;
        float score = 0.0f;
        in >> matchIndex >> score;
        result.matches0[static_cast<size_t>(i)] = matchIndex;
        result.matchingScores0[static_cast<size_t>(i)] = score;
    }

    for (int i = 0; i < numKeypoints1; ++i)
    {
        qint32 matchIndex = -1;
        float score = 0.0f;
        in >> matchIndex >> score;
        result.matches1[static_cast<size_t>(i)] = matchIndex;
        result.matchingScores1[static_cast<size_t>(i)] = score;
    }

    if (in.status() != QDataStream::Ok)
    {
        return false;
    }

    result.cvMatches.clear();
    for (int i = 0; i < numKeypoints0; ++i)
    {
        const int matchIndex = result.matches0[static_cast<size_t>(i)];
        if (matchIndex < 0)
        {
            continue;
        }

        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = matchIndex;
        match.distance = 1.0f - result.matchingScores0[static_cast<size_t>(i)];
        result.cvMatches.push_back(match);
    }
    result.numMatches = static_cast<int>(result.cvMatches.size());
    return true;
}

} // namespace xjw::feature_match
