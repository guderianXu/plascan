// =============================================================================
// 文件: FeatureFileIO.cpp
// 功能: 多提取器二进制特征文件 I/O (支持 SPBT/DSKB/ALKB/SFTB/ORBB/AKZB)
// =============================================================================
#include "FeatureOutput.h"
#include <torch/torch.h>

#include "FeatureFileIO.h"
#include <QDebug>
#include <cstring>

namespace
{

struct MagicEntry
{
    char bytes[4];
    const char *algoName;
};

const MagicEntry MAGIC_TABLE[] = {
    {{'S','P','B','T'}, "superpoint"},
    {{'D','S','K','B'}, "disk"},
    {{'A','L','K','B'}, "aliked"},
    {{'S','F','T','B'}, "sift"},
    {{'O','R','B','B'}, "orb"},
    {{'A','K','Z','B'}, "akaze"},
};

const char magicForAlgo(const std::string &algo)
{
    for (const auto &e : MAGIC_TABLE)
        if (e.algoName == algo) return e.bytes[0];
    return 'S'; // default superpoint
}

const char *ALGO_FOR_MAGIC(const char magic[4])
{
    for (const auto &e : MAGIC_TABLE)
        if (std::memcmp(e.bytes, magic, 4) == 0) return e.algoName;
    return "superpoint"; // fallback
}

void writeMagic(QDataStream &out, const std::string &algoName)
{
    for (const auto &e : MAGIC_TABLE)
    {
        if (e.algoName == algoName)
        {
            out.writeRawData(e.bytes, 4);
            return;
        }
    }
    out.writeRawData("SPBT", 4); // default
}

} // anonymous

bool FeatureFileIO::write(const QString& path, const QString& image_name,
                          const FeatureOutput& output, const std::string &algoName)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        qWarning() << "Cannot open for writing:" << path;
        return false;
    }
    QDataStream out(&file);
    out.setByteOrder(QDataStream::LittleEndian);
    out.setFloatingPointPrecision(QDataStream::SinglePrecision);

    writeMagic(out, algoName);
    out << quint32(1);  // version

    QByteArray nameBytes = image_name.toUtf8();
    out << quint32(nameBytes.size());
    out.writeRawData(nameBytes.constData(), nameBytes.size());

    quint32 N = static_cast<quint32>(output.keypoints.size());
    out << N;
    for (quint32 i = 0; i < N; ++i)
    {
        out << float(output.keypoints[i].pt.x);
        out << float(output.keypoints[i].pt.y);
        out << float(output.scores[i]);
    }

    if (!output.descriptors.defined() || output.descriptors.numel() == 0)
    {
        out << quint32(0);
    }
    else
    {
        auto d = output.descriptors.to(torch::kCPU).contiguous();
        int rows = d.size(0), cols = d.size(1);
        out << quint32(cols);
        auto acc = d.accessor<float, 2>();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                out << acc[i][j];
    }

    file.close();
    return true;
}

bool FeatureFileIO::read(const QString& path, QString& image_name, FeatureOutput& output)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        qWarning() << "Cannot open for reading:" << path;
        return false;
    }
    QDataStream in(&file);
    in.setByteOrder(QDataStream::LittleEndian);
    in.setFloatingPointPrecision(QDataStream::SinglePrecision);

    char magic[4];
    if (in.readRawData(magic, 4) != 4) { file.close(); return false; }

    // Accept all known magic bytes
    const char *algo = ALGO_FOR_MAGIC(magic);
    if (!algo) { file.close(); return false; }

    quint32 version; in >> version;
    quint32 nameLen; in >> nameLen;
    QByteArray nameBytes(nameLen, 0);
    if (in.readRawData(nameBytes.data(), nameLen) != int(nameLen)) { file.close(); return false; }
    image_name = QString::fromUtf8(nameBytes);

    quint32 N; in >> N;
    output.keypoints.clear(); output.scores.clear();
    for (quint32 i = 0; i < N; ++i)
    {
        float x, y, s; in >> x >> y >> s;
        cv::KeyPoint kp;
        kp.pt.x = x; kp.pt.y = y; kp.response = s; kp.size = 8.0f;
        output.keypoints.push_back(kp);
        output.scores.push_back(s);
    }

    quint32 descDim; in >> descDim;
    if (descDim == 0)
    {
        output.descriptors = torch::Tensor();
    }
    else
    {
        int rows = int(N), cols = int(descDim);
        output.descriptors = torch::empty({rows, cols}, torch::kFloat32);
        auto acc = output.descriptors.accessor<float, 2>();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            { float v; in >> v; acc[i][j] = v; }
    }

    file.close();
    return true;
}

std::string FeatureFileIO::peekAlgorithm(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return "";
    char magic[4];
    if (file.read(magic, 4) != 4) { file.close(); return ""; }
    file.close();
    return ALGO_FOR_MAGIC(magic);
}

int FeatureFileIO::peekKeypointCount(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return -1;
    QDataStream in(&file);
    in.setByteOrder(QDataStream::LittleEndian);
    char magic[4];
    if (in.readRawData(magic, 4) != 4) { file.close(); return -1; }
    quint32 ver; in >> ver;
    quint32 nameLen; in >> nameLen;
    file.skip(nameLen);
    quint32 N; in >> N;
    file.close();
    return int(N);
}
