// =============================================================================
// 文件: FeatureFileIO.cpp
// 功能: 多提取器二进制特征文件 I/O (支持 SPBT/DSKB/ALKB/SFTB/ORBB/AKZB/DEDE/SURF)
// =============================================================================
#include "FeatureData.h"
#include "FeatureOutput.h"
#include <torch/torch.h>

#include "FeatureFileIO.h"
#include <QFile>
#include <QDataStream>
#include <QDebug>
#include <cmath>
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
    {{'D','E','D','E'}, "dedode"},
    {{'S','U','R','F'}, "surf"},
};

constexpr quint32 kCurrentFeatureFileVersion = 3;

const char *ALGO_FOR_MAGIC(const char magic[4])
{
    for (const auto &e : MAGIC_TABLE)
        if (std::memcmp(e.bytes, magic, 4) == 0) return e.algoName;
    return nullptr;
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
    out << kCurrentFeatureFileVersion;  // v2 stores scale/orientation; v3 stores image size.

    QByteArray nameBytes = image_name.toUtf8();
    out << quint32(nameBytes.size());
    out.writeRawData(nameBytes.constData(), nameBytes.size());
    out << qint32(output.imageWidth);
    out << qint32(output.imageHeight);

    quint32 N = static_cast<quint32>(output.keypoints.size());
    out << N;
    for (quint32 i = 0; i < N; ++i)
    {
        const cv::KeyPoint &keypoint = output.keypoints[i];
        const float score = i < static_cast<quint32>(output.scores.size())
            ? output.scores[i]
            : keypoint.response;
        out << float(keypoint.pt.x);
        out << float(keypoint.pt.y);
        out << float(score);
        out << float(keypoint.size);
        out << float(keypoint.angle);
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

bool FeatureFileIO::read(const QString& path,
                         QString& image_name,
                         FeatureOutput& output,
                         std::string *algorithmName)
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
    if (algorithmName)
    {
        *algorithmName = algo;
    }

    quint32 version; in >> version;
    if (version < 1 || version > kCurrentFeatureFileVersion)
    {
        file.close();
        return false;
    }
    quint32 nameLen; in >> nameLen;
    QByteArray nameBytes(nameLen, 0);
    if (in.readRawData(nameBytes.data(), nameLen) != int(nameLen)) { file.close(); return false; }
    image_name = QString::fromUtf8(nameBytes);
    output.imageWidth = 0;
    output.imageHeight = 0;
    if (version >= 3)
    {
        qint32 imageWidth = 0;
        qint32 imageHeight = 0;
        in >> imageWidth >> imageHeight;
        output.imageWidth = std::max(0, static_cast<int>(imageWidth));
        output.imageHeight = std::max(0, static_cast<int>(imageHeight));
    }

    quint32 N; in >> N;
    output.keypoints.clear(); output.scores.clear();
    for (quint32 i = 0; i < N; ++i)
    {
        float x = 0.0f;
        float y = 0.0f;
        float s = 0.0f;
        float size = 8.0f;
        float angle = -1.0f;
        in >> x >> y >> s;
        if (version >= 2)
        {
            in >> size >> angle;
        }
        cv::KeyPoint kp;
        kp.pt.x = x;
        kp.pt.y = y;
        kp.response = s;
        kp.size = (std::isfinite(size) && size > 0.0f) ? size : 8.0f;
        kp.angle = std::isfinite(angle) ? angle : -1.0f;
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

bool FeatureFileIO::readData(const QString& path,
                             QString& imageName,
                             xjw::feature_extractors::FeatureData& output)
{
    FeatureOutput featureOutput;
    std::string algorithmName;
    if (!read(path, imageName, featureOutput, &algorithmName))
    {
        return false;
    }
    if (algorithmName.empty())
    {
        algorithmName = "superpoint";
    }
    output = xjw::feature_extractors::FeatureData::fromFeatureOutput(featureOutput, algorithmName);
    return true;
}

std::string FeatureFileIO::peekAlgorithm(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return "";
    char magic[4];
    if (file.read(magic, 4) != 4) { file.close(); return ""; }
    file.close();
    const char *algo = ALGO_FOR_MAGIC(magic);
    return algo ? std::string(algo) : std::string();
}

int FeatureFileIO::peekCount(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return -1;
    QDataStream in(&file);
    in.setByteOrder(QDataStream::LittleEndian);
    char magic[4];
    if (in.readRawData(magic, 4) != 4) { file.close(); return -1; }
    if (!ALGO_FOR_MAGIC(magic)) { file.close(); return -1; }
    quint32 ver; in >> ver;
    if (ver < 1 || ver > kCurrentFeatureFileVersion) { file.close(); return -1; }
    quint32 nameLen; in >> nameLen;
    if (file.skip(static_cast<qint64>(nameLen)) != static_cast<qint64>(nameLen))
    {
        file.close();
        return -1;
    }
    if (ver >= 3)
    {
        qint32 imageWidth = 0;
        qint32 imageHeight = 0;
        in >> imageWidth >> imageHeight;
    }
    quint32 N; in >> N;
    file.close();
    return int(N);
}
