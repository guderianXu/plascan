#include "BaBenchmarkRealDataset.h"

#include "FramePinholeCamera.h"
#include "io/PathIO.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QString>

#include <cmath>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xjw::ba_benchmark
{
    namespace
    {

        QString toQString(const std::filesystem::path& path)
        {
#ifdef _WIN32
            return QString::fromStdWString(path.wstring());
#else
            const std::u8string value = path.u8string();
            return QString::fromUtf8(reinterpret_cast<const char*>(value.data()), static_cast<qsizetype>(value.size()));
#endif
        }

        std::vector<std::string> tokenizeListLine(const std::string& line)
        {
            std::vector<std::string> tokens;
            std::string token;
            char quote = '\0';
            bool active = false;
            for (char ch : line)
            {
                if (quote != '\0')
                {
                    if (ch == quote)
                    {
                        quote = '\0';
                    }
                    else
                    {
                        token.push_back(ch);
                    }
                    active = true;
                }
                else if (ch == '\'' || ch == '"')
                {
                    quote = ch;
                    active = true;
                }
                else if (std::isspace(static_cast<unsigned char>(ch)) != 0)
                {
                    if (active)
                    {
                        tokens.push_back(std::move(token));
                        token.clear();
                        active = false;
                    }
                }
                else
                {
                    token.push_back(ch);
                    active = true;
                }
            }
            if (quote != '\0')
            {
                throw std::runtime_error("image_camera.lis 中存在未闭合引号");
            }
            if (active)
            {
                tokens.push_back(std::move(token));
            }
            return tokens;
        }

        std::vector<FramePinholeCamera> loadCameras(const std::filesystem::path& listPath)
        {
            QFile file(toQString(listPath));
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            {
                throw std::runtime_error("无法读取相机列表: " + file.errorString().toStdString());
            }

            std::vector<FramePinholeCamera> cameras;
            const std::filesystem::path base = std::filesystem::absolute(listPath).parent_path();
            int lineNumber = 0;
            while (!file.atEnd())
            {
                ++lineNumber;
                const std::string line = file.readLine().toStdString();
                const std::vector<std::string> tokens = tokenizeListLine(line);
                if (tokens.empty() || (!tokens[0].empty() && tokens[0][0] == '#'))
                {
                    continue;
                }
                if (tokens.size() != 2)
                {
                    throw std::runtime_error("相机列表第 " + std::to_string(lineNumber) +
                                             " 行必须包含影像和相机两个路径");
                }
                std::filesystem::path cameraPath = xjw::common::io::toFilesystemPath(tokens[1]);
                if (cameraPath.is_relative())
                {
                    cameraPath = base / cameraPath;
                }
                cameraPath = cameraPath.lexically_normal();
                FramePinholeCamera camera;
                if (!camera.loadFromFile(toQString(cameraPath).toUtf8().toStdString()))
                {
                    throw std::runtime_error("无法加载 TSAI 相机: " + toQString(cameraPath).toStdString());
                }
                cameras.push_back(std::move(camera));
            }
            if (cameras.size() < 2)
            {
                throw std::runtime_error("相机列表至少需要两个有效 TSAI 相机");
            }
            return cameras;
        }

        double requiredNumber(const QJsonArray& values, qsizetype index, const std::string& label)
        {
            if (index >= values.size() || !values[index].isDouble())
            {
                throw std::runtime_error(label + " 缺少有限数值");
            }
            const double value = values[index].toDouble();
            if (!std::isfinite(value))
            {
                throw std::runtime_error(label + " 不是有限数值");
            }
            return value;
        }

    } // namespace

    BenchmarkDataset loadRealDataset(const std::filesystem::path& datasetJson, const std::filesystem::path& cameraList)
    {
        BenchmarkDataset dataset;
        dataset.cameras = loadCameras(cameraList);

        QFile file(toQString(datasetJson));
        if (!file.open(QIODevice::ReadOnly))
        {
            throw std::runtime_error("无法读取真实 BA JSON: " + file.errorString().toStdString());
        }
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject())
        {
            throw std::runtime_error("真实 BA JSON 解析失败: " + parseError.errorString().toStdString());
        }
        const QJsonArray points = document.object().value(QStringLiteral("points")).toArray();
        if (points.isEmpty())
        {
            throw std::runtime_error("真实 BA JSON 不包含 points 数组");
        }

        dataset.tracks.reserve(static_cast<std::size_t>(points.size()));
        for (qsizetype pointIndex = 0; pointIndex < points.size(); ++pointIndex)
        {
            const QJsonObject point = points[pointIndex].toObject();
            const QJsonArray xyz = point.value(QStringLiteral("point_xyz")).toArray();
            BATrack track;
            const std::string prefix = "points[" + std::to_string(pointIndex) + "]";
            track.initialPoint = {{requiredNumber(xyz, 0, prefix + ".point_xyz[0]"),
                                   requiredNumber(xyz, 1, prefix + ".point_xyz[1]"),
                                   requiredNumber(xyz, 2, prefix + ".point_xyz[2]")}};

            const QJsonArray observations = point.value(QStringLiteral("observations")).toArray();
            track.observations.reserve(static_cast<std::size_t>(observations.size()));
            for (qsizetype observationIndex = 0; observationIndex < observations.size(); ++observationIndex)
            {
                const QJsonValue observationValue = observations[observationIndex];
                const QJsonObject observation = observationValue.toObject();
                const QJsonArray compactObservation = observationValue.toArray();
                if (observationValue.isArray() && compactObservation.size() < 5)
                {
                    throw std::runtime_error(prefix + ".observations[" + std::to_string(observationIndex) +
                                             "] compact row is incomplete");
                }
                const int imageId = observationValue.isArray()
                                        ? compactObservation.at(0).toInt(-1)
                                        : observation.value(QStringLiteral("image_id")).toInt(-1);
                if (imageId < 0 || imageId >= static_cast<int>(dataset.cameras.size()))
                {
                    throw std::runtime_error(prefix + ".observations[" + std::to_string(observationIndex) +
                                             "].image_id 超出相机列表范围");
                }
                const QJsonArray xy = observationValue.isArray()
                                          ? QJsonArray{compactObservation.at(2), compactObservation.at(3)}
                                          : observation.value(QStringLiteral("xy")).toArray();
                const std::string observationPrefix =
                    prefix + ".observations[" + std::to_string(observationIndex) + "].xy";
                track.observations.push_back(BAObservation{imageId,
                                                           requiredNumber(xy, 0, observationPrefix + "[0]"),
                                                           requiredNumber(xy, 1, observationPrefix + "[1]"),
                                                           1.0});
            }
            dataset.observations += track.observations.size();
            dataset.tracks.push_back(std::move(track));
        }
        return dataset;
    }

} // namespace xjw::ba_benchmark
