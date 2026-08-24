#include "reconstruction/RpcAerialTriangulationRunner.h"

#include "reconstruction/SfmAttemptRunner.h"

#include "ProjectCameraIO.h"
#include "RpcStereoIntersection.h"
#include "io/PathIO.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"
#include "project/SparseResultQuality.h"

#include <QColor>
#include <QDataStream>
#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QSet>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <vector>

namespace xjw::aerial_triangulation
{
    namespace
    {

        struct Observation
        {
            ImageId imageId = kInvalidImageId;
            FeatureIdx featureIdx = kInvalidFeatureIdx;

            bool operator<(const Observation& other) const
            {
                return imageId < other.imageId || (imageId == other.imageId && featureIdx < other.featureIdx);
            }
        };

        struct RpcPoint
        {
            RpcCameraModel::EcefCoordinate ecef{};
            RpcCameraModel::GeodeticCoordinate geodetic{};
            std::array<float, 3> localEnu{};
            std::array<quint8, 3> color{{128, 128, 128}};
            std::vector<Observation> observations;
            double rmsPixels = 0.0;
            double maximumResidualPixels = 0.0;
        };

        class DisjointSet
        {
        public:
            std::size_t add()
            {
                const std::size_t index = _parent.size();
                _parent.push_back(index);
                _rank.push_back(0);
                return index;
            }

            std::size_t find(std::size_t index)
            {
                if (_parent[index] != index)
                {
                    _parent[index] = find(_parent[index]);
                }
                return _parent[index];
            }

            void unite(std::size_t first, std::size_t second)
            {
                first = find(first);
                second = find(second);
                if (first == second)
                {
                    return;
                }
                if (_rank[first] < _rank[second])
                {
                    std::swap(first, second);
                }
                _parent[second] = first;
                if (_rank[first] == _rank[second])
                {
                    ++_rank[first];
                }
            }

        private:
            std::vector<std::size_t> _parent;
            std::vector<unsigned char> _rank;
        };

        struct CameraResidualAccumulator
        {
            int observationCount = 0;
            double squaredErrorSum = 0.0;
            double maximumResidual = 0.0;
        };

        bool canceled(const PreparedAerialTriangulationInput& input)
        {
            return input.cancelFlag && input.cancelFlag->load(std::memory_order_relaxed);
        }

        void reportProgress(const PreparedAerialTriangulationInput& input, const QString& stage, int percent)
        {
            if (input.progressFn)
            {
                input.progressFn(stage, std::clamp(percent, 0, 100));
            }
        }

        double median(std::vector<double> values)
        {
            if (values.empty())
            {
                return 0.0;
            }
            std::sort(values.begin(), values.end());
            const std::size_t middle = values.size() / 2;
            return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
        }

        std::vector<std::vector<Observation>> buildTracks(const PreparedTiePointGraph& graph)
        {
            std::map<Observation, std::size_t> indexByObservation;
            std::vector<Observation> observations;
            DisjointSet components;

            const auto observationIndex = [&](const Observation& observation)
            {
                const auto existing = indexByObservation.find(observation);
                if (existing != indexByObservation.end())
                {
                    return existing->second;
                }
                const std::size_t index = components.add();
                indexByObservation.emplace(observation, index);
                observations.push_back(observation);
                return index;
            };

            for (const PreparedTiePointMatchPair& pair : graph.matchPairs)
            {
                const auto keypointsA = graph.keypointsByImage.constFind(pair.imageA);
                const auto keypointsB = graph.keypointsByImage.constFind(pair.imageB);
                if (keypointsA == graph.keypointsByImage.cend() || keypointsB == graph.keypointsByImage.cend())
                {
                    continue;
                }
                for (const FeatureMatch& match : pair.matches)
                {
                    if (match.idx1 >= keypointsA->size() || match.idx2 >= keypointsB->size())
                    {
                        continue;
                    }
                    components.unite(observationIndex({pair.imageA, match.idx1}),
                                     observationIndex({pair.imageB, match.idx2}));
                }
            }

            std::map<std::size_t, std::vector<Observation>> grouped;
            for (std::size_t index = 0; index < observations.size(); ++index)
            {
                grouped[components.find(index)].push_back(observations[index]);
            }

            std::vector<std::vector<Observation>> tracks;
            tracks.reserve(grouped.size());
            for (auto& [root, track] : grouped)
            {
                (void)root;
                std::sort(track.begin(), track.end());
                bool duplicateImage = false;
                for (std::size_t index = 1; index < track.size(); ++index)
                {
                    duplicateImage |= track[index - 1].imageId == track[index].imageId;
                }
                if (!duplicateImage && track.size() >= 2)
                {
                    tracks.push_back(std::move(track));
                }
            }
            return tracks;
        }

        CameraImageCoordinate imageCoordinate(const PreparedTiePointGraph& graph, const Observation& observation)
        {
            const FeatureKeypoint& keypoint =
                graph.keypointsByImage.value(observation.imageId).at(observation.featureIdx);
            return {keypoint.x, keypoint.y};
        }

        bool solve3x3(double matrix[3][3], double values[3], std::array<double, 3>* solution)
        {
            double augmented[3][4]{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    augmented[row][column] = matrix[row][column];
                }
                augmented[row][3] = values[row];
            }
            for (int column = 0; column < 3; ++column)
            {
                int pivot = column;
                for (int row = column + 1; row < 3; ++row)
                {
                    if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
                    {
                        pivot = row;
                    }
                }
                if (!std::isfinite(augmented[pivot][column]) || std::abs(augmented[pivot][column]) < 1.0e-20)
                {
                    return false;
                }
                for (int entry = column; entry < 4; ++entry)
                {
                    std::swap(augmented[column][entry], augmented[pivot][entry]);
                }
                for (int row = column + 1; row < 3; ++row)
                {
                    const double factor = augmented[row][column] / augmented[column][column];
                    for (int entry = column; entry < 4; ++entry)
                    {
                        augmented[row][entry] -= factor * augmented[column][entry];
                    }
                }
            }
            for (int row = 2; row >= 0; --row)
            {
                double value = augmented[row][3];
                for (int column = row + 1; column < 3; ++column)
                {
                    value -= augmented[row][column] * (*solution)[column];
                }
                (*solution)[row] = value / augmented[row][row];
            }
            return std::isfinite((*solution)[0]) && std::isfinite((*solution)[1]) && std::isfinite((*solution)[2]);
        }

        bool rpcResiduals(const PreparedTiePointGraph& graph,
                          const QMap<ImageId, RpcCameraModel>& cameras,
                          const std::vector<Observation>& observations,
                          const RpcCameraModel::EcefCoordinate& ecef,
                          std::vector<double>* residuals)
        {
            residuals->clear();
            residuals->reserve(observations.size() * 2);
            for (const Observation& observation : observations)
            {
                const auto camera = cameras.constFind(observation.imageId);
                CameraGroundProjection projection;
                if (camera == cameras.cend() || !camera->groundToImage(ecef, &projection))
                {
                    return false;
                }
                const CameraImageCoordinate measured = imageCoordinate(graph, observation);
                residuals->push_back(projection.image.sample - measured.sample);
                residuals->push_back(projection.image.line - measured.line);
            }
            return true;
        }

        double residualRms(const std::vector<double>& residuals)
        {
            if (residuals.empty())
            {
                return std::numeric_limits<double>::infinity();
            }
            double squaredSum = 0.0;
            for (double residual : residuals)
            {
                squaredSum += residual * residual;
            }
            return std::sqrt(squaredSum / residuals.size());
        }

        bool refineRpcPoint(const PreparedTiePointGraph& graph,
                            const QMap<ImageId, RpcCameraModel>& cameras,
                            const std::vector<Observation>& observations,
                            RpcCameraModel::EcefCoordinate* ecef)
        {
            std::vector<double> currentResiduals;
            if (!rpcResiduals(graph, cameras, observations, *ecef, &currentResiduals))
            {
                return false;
            }
            double currentRms = residualRms(currentResiduals);
            constexpr double derivativeStepMeters = 0.5;
            for (int iteration = 0; iteration < 15; ++iteration)
            {
                std::vector<std::array<double, 3>> jacobian(currentResiduals.size());
                for (int axis = 0; axis < 3; ++axis)
                {
                    RpcCameraModel::EcefCoordinate plus = *ecef;
                    RpcCameraModel::EcefCoordinate minus = *ecef;
                    plus[axis] += derivativeStepMeters;
                    minus[axis] -= derivativeStepMeters;
                    std::vector<double> plusResiduals;
                    std::vector<double> minusResiduals;
                    if (!rpcResiduals(graph, cameras, observations, plus, &plusResiduals) ||
                        !rpcResiduals(graph, cameras, observations, minus, &minusResiduals))
                    {
                        return false;
                    }
                    for (std::size_t row = 0; row < currentResiduals.size(); ++row)
                    {
                        jacobian[row][axis] = (plusResiduals[row] - minusResiduals[row]) / (2.0 * derivativeStepMeters);
                    }
                }

                double normal[3][3]{};
                double rightHandSide[3]{};
                for (std::size_t row = 0; row < currentResiduals.size(); ++row)
                {
                    const double residualMagnitude = std::abs(currentResiduals[row]);
                    const double weight = residualMagnitude <= 2.0 ? 1.0 : 2.0 / residualMagnitude;
                    for (int firstAxis = 0; firstAxis < 3; ++firstAxis)
                    {
                        rightHandSide[firstAxis] -= weight * jacobian[row][firstAxis] * currentResiduals[row];
                        for (int secondAxis = 0; secondAxis < 3; ++secondAxis)
                        {
                            normal[firstAxis][secondAxis] +=
                                weight * jacobian[row][firstAxis] * jacobian[row][secondAxis];
                        }
                    }
                }
                const double diagonalScale = std::max({normal[0][0], normal[1][1], normal[2][2], 1.0e-20});
                for (int axis = 0; axis < 3; ++axis)
                {
                    normal[axis][axis] += diagonalScale * 1.0e-8;
                }
                std::array<double, 3> update{};
                if (!solve3x3(normal, rightHandSide, &update))
                {
                    break;
                }
                const double updateLength = std::hypot(update[0], std::hypot(update[1], update[2]));
                const double updateScale = updateLength > 500.0 ? 500.0 / updateLength : 1.0;
                RpcCameraModel::EcefCoordinate candidate{(*ecef)[0] + updateScale * update[0],
                                                         (*ecef)[1] + updateScale * update[1],
                                                         (*ecef)[2] + updateScale * update[2]};
                std::vector<double> candidateResiduals;
                if (!rpcResiduals(graph, cameras, observations, candidate, &candidateResiduals))
                {
                    break;
                }
                const double candidateRms = residualRms(candidateResiduals);
                if (candidateRms > currentRms + 1.0e-12)
                {
                    break;
                }
                *ecef = candidate;
                currentResiduals = std::move(candidateResiduals);
                currentRms = candidateRms;
                if (updateLength * updateScale <= 1.0e-3)
                {
                    break;
                }
            }
            return true;
        }

        bool intersectTrack(const PreparedTiePointGraph& graph,
                            const QMap<ImageId, RpcCameraModel>& cameras,
                            const std::vector<Observation>& observations,
                            double maximumRmsPixels,
                            RpcPoint* point,
                            QMap<ImageId, CameraResidualAccumulator>* cameraResiduals)
        {
            RpcStereoIntersectionOptions options;
            options.pixelTolerance = 1.0e-4;
            options.positionToleranceMeters = 1.0e-3;
            options.maximumIterations = 40;

            RpcStereoIntersectionResult best;
            bool hasBest = false;
            for (std::size_t first = 0; first + 1 < observations.size(); ++first)
            {
                for (std::size_t second = first + 1; second < observations.size(); ++second)
                {
                    const Observation& firstObservation = observations[first];
                    const Observation& secondObservation = observations[second];
                    const auto firstCamera = cameras.constFind(firstObservation.imageId);
                    const auto secondCamera = cameras.constFind(secondObservation.imageId);
                    if (firstCamera == cameras.cend() || secondCamera == cameras.cend())
                    {
                        continue;
                    }
                    RpcStereoIntersectionResult candidate;
                    const bool converged = intersectRpcObservations(firstCamera.value(),
                                                                    imageCoordinate(graph, firstObservation),
                                                                    secondCamera.value(),
                                                                    imageCoordinate(graph, secondObservation),
                                                                    &candidate,
                                                                    options);
                    if ((!converged && candidate.iterations <= 0) || !std::isfinite(candidate.reprojectionRmsPixels))
                    {
                        continue;
                    }
                    if (!hasBest || candidate.reprojectionRmsPixels < best.reprojectionRmsPixels)
                    {
                        best = candidate;
                        hasBest = true;
                    }
                }
            }
            if (!hasBest)
            {
                return false;
            }

            if (!refineRpcPoint(graph, cameras, observations, &best.ecefMeters) ||
                !RpcCameraModel::ecefToGeodetic(best.ecefMeters, &best.geodetic))
            {
                return false;
            }

            double squaredErrorSum = 0.0;
            double maximumResidual = 0.0;
            for (const Observation& observation : observations)
            {
                const auto camera = cameras.constFind(observation.imageId);
                CameraGroundProjection projection;
                if (camera == cameras.cend() || !camera->groundToImage(best.ecefMeters, &projection))
                {
                    return false;
                }
                const CameraImageCoordinate measured = imageCoordinate(graph, observation);
                const double residual =
                    std::hypot(projection.image.sample - measured.sample, projection.image.line - measured.line);
                if (!std::isfinite(residual))
                {
                    return false;
                }
                squaredErrorSum += residual * residual;
                maximumResidual = std::max(maximumResidual, residual);
            }
            const double rms = std::sqrt(squaredErrorSum / observations.size());
            if (!std::isfinite(rms) || rms > maximumRmsPixels || maximumResidual > maximumRmsPixels * 2.0 ||
                !std::isfinite(best.geodetic[0]) || !std::isfinite(best.geodetic[1]) ||
                !std::isfinite(best.geodetic[2]) || std::abs(best.geodetic[0]) > 180.0 ||
                std::abs(best.geodetic[1]) > 90.0)
            {
                return false;
            }

            point->ecef = best.ecefMeters;
            point->geodetic = best.geodetic;
            point->observations = observations;
            point->rmsPixels = rms;
            point->maximumResidualPixels = maximumResidual;

            if (cameraResiduals)
            {
                for (const Observation& observation : observations)
                {
                    CameraGroundProjection projection;
                    cameras.value(observation.imageId).groundToImage(best.ecefMeters, &projection);
                    const CameraImageCoordinate measured = imageCoordinate(graph, observation);
                    const double residual =
                        std::hypot(projection.image.sample - measured.sample, projection.image.line - measured.line);
                    CameraResidualAccumulator& accumulator = (*cameraResiduals)[observation.imageId];
                    ++accumulator.observationCount;
                    accumulator.squaredErrorSum += residual * residual;
                    accumulator.maximumResidual = std::max(accumulator.maximumResidual, residual);
                }
            }
            return true;
        }

        RpcCameraModel::GeodeticCoordinate assignLocalEnu(std::vector<RpcPoint>* points)
        {
            std::vector<double> longitudes;
            std::vector<double> latitudes;
            std::vector<double> heights;
            longitudes.reserve(points->size());
            latitudes.reserve(points->size());
            heights.reserve(points->size());
            for (const RpcPoint& point : *points)
            {
                longitudes.push_back(point.geodetic[0]);
                latitudes.push_back(point.geodetic[1]);
                heights.push_back(point.geodetic[2]);
            }
            const RpcCameraModel::GeodeticCoordinate origin{
                median(std::move(longitudes)), median(std::move(latitudes)), median(std::move(heights))};
            RpcCameraModel::EcefCoordinate originEcef{};
            RpcCameraModel::geodeticToEcef(origin, &originEcef);

            constexpr double degreesToRadians = 3.14159265358979323846 / 180.0;
            const double longitude = origin[0] * degreesToRadians;
            const double latitude = origin[1] * degreesToRadians;
            const std::array<double, 3> east{-std::sin(longitude), std::cos(longitude), 0.0};
            const std::array<double, 3> north{-std::sin(latitude) * std::cos(longitude),
                                              -std::sin(latitude) * std::sin(longitude),
                                              std::cos(latitude)};
            const std::array<double, 3> up{
                std::cos(latitude) * std::cos(longitude), std::cos(latitude) * std::sin(longitude), std::sin(latitude)};
            for (RpcPoint& point : *points)
            {
                const std::array<double, 3> delta{
                    point.ecef[0] - originEcef[0], point.ecef[1] - originEcef[1], point.ecef[2] - originEcef[2]};
                const auto dot = [&delta](const std::array<double, 3>& axis)
                { return delta[0] * axis[0] + delta[1] * axis[1] + delta[2] * axis[2]; };
                point.localEnu = {
                    static_cast<float>(dot(east)), static_cast<float>(dot(north)), static_cast<float>(dot(up))};
            }
            return origin;
        }

        void sampleColors(const PreparedAerialTriangulationInput& input,
                          const PreparedTiePointGraph& graph,
                          std::vector<RpcPoint>* points)
        {
            QMap<ImageId, QImage> imageCache;
            for (RpcPoint& point : *points)
            {
                if (point.observations.empty())
                {
                    continue;
                }
                const Observation& observation = point.observations.front();
                if (!imageCache.contains(observation.imageId))
                {
                    imageCache.insert(observation.imageId,
                                      QImage(input.images.value(static_cast<int>(observation.imageId))));
                }
                const QImage& image = imageCache[observation.imageId];
                if (image.isNull())
                {
                    continue;
                }
                const CameraImageCoordinate coordinate = imageCoordinate(graph, observation);
                const QColor color = image.pixelColor(std::clamp(qRound(coordinate.sample), 0, image.width() - 1),
                                                      std::clamp(qRound(coordinate.line), 0, image.height() - 1));
                point.color = {static_cast<quint8>(color.red()),
                               static_cast<quint8>(color.green()),
                               static_cast<quint8>(color.blue())};
            }
        }

        bool writePly(const QString& path, const std::vector<RpcPoint>& points, QString* errorMessage)
        {
            QSaveFile file(path);
            if (!file.open(QIODevice::WriteOnly))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法写入 RPC 空三稀疏点云: %1").arg(path);
                }
                return false;
            }
            const QByteArray header =
                QByteArrayLiteral("ply\nformat binary_little_endian 1.0\n"
                                  "comment PlaScan RPC aerial triangulation local ENU\n") +
                QByteArray("element vertex ") + QByteArray::number(points.size()) +
                QByteArrayLiteral("\nproperty float x\nproperty float y\nproperty float z\n"
                                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
            if (file.write(header) != header.size())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("写入 RPC 空三点云头失败: %1").arg(path);
                }
                return false;
            }

            QDataStream stream(&file);
            stream.setByteOrder(QDataStream::LittleEndian);
            stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
            for (const RpcPoint& point : points)
            {
                stream << point.localEnu[0] << point.localEnu[1] << point.localEnu[2];
                const char colors[3]{static_cast<char>(point.color[0]),
                                     static_cast<char>(point.color[1]),
                                     static_cast<char>(point.color[2])};
                if (stream.writeRawData(colors, 3) != 3)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("写入 RPC 空三点云数据失败: %1").arg(path);
                    }
                    return false;
                }
            }
            if (stream.status() != QDataStream::Ok || !file.commit())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("提交 RPC 空三点云失败: %1").arg(path);
                }
                return false;
            }
            return true;
        }

        QJsonObject pointJson(const RpcPoint& point)
        {
            QJsonArray observations;
            for (const Observation& observation : point.observations)
            {
                observations.append(
                    QJsonObject{{QStringLiteral("image_id"), static_cast<qint64>(observation.imageId)},
                                {QStringLiteral("feature_idx"), static_cast<qint64>(observation.featureIdx)}});
            }
            return QJsonObject{
                {QStringLiteral("xyz"), QJsonArray{point.localEnu[0], point.localEnu[1], point.localEnu[2]}},
                {QStringLiteral("geodetic_wgs84"), QJsonArray{point.geodetic[0], point.geodetic[1], point.geodetic[2]}},
                {QStringLiteral("rms_reproj_px"), point.rmsPixels},
                {QStringLiteral("maximum_reproj_px"), point.maximumResidualPixels},
                {QStringLiteral("track_len"), static_cast<int>(point.observations.size())},
                {QStringLiteral("observations"), observations}};
        }

    } // namespace

    RpcCameraInput RpcAerialTriangulationRunner::inspectInput(const PreparedAerialTriangulationInput& input)
    {
        RpcCameraInput result;
        const QMap<QString, QJsonObject> metadataByPath =
            xjw::common::project::projectImageMetaByPath(input.projectMeta, true);
        QStringList nonRpcImages;

        for (int index = 0; index < input.images.size(); ++index)
        {
            const QString& imagePath = input.images.at(index);
            const QString normalizedPath = xjw::common::project::normalizePath(imagePath);
            const QJsonObject imageMetadata = metadataByPath.value(normalizedPath);
            const QJsonObject declaredCamera = imageMetadata.value(QStringLiteral("camera")).toObject();
            const QString declaredModel = declaredCamera.value(QStringLiteral("model")).toString().trimmed();

            RpcCameraModel camera;
            bool loaded = false;
        if (declaredModel.compare(QStringLiteral("rpc"), Qt::CaseInsensitive) == 0)
        {
            loaded = xjw::common::project::cameraFromJson(declaredCamera, &camera);
        }
        if (!loaded)
        {
            QJsonObject rasterCamera;
            loaded = xjw::common::project::parseRpcCameraRaster(imagePath, &rasterCamera, nullptr) &&
                     xjw::common::project::cameraFromJson(rasterCamera, &camera);
            }

            if (loaded && camera.isValid())
            {
                result.cameras.insert(static_cast<ImageId>(index), camera);
            }
            else
            {
                nonRpcImages.append(QFileInfo(imagePath).fileName());
            }
        }

        if (result.cameras.isEmpty())
        {
            result.status = RpcCameraInputStatus::None;
            return result;
        }
        if (result.cameras.size() == input.images.size())
        {
            result.status = RpcCameraInputStatus::Complete;
            return result;
        }

        result.status = RpcCameraInputStatus::Mixed;
        result.errorMessage = QStringLiteral("RPC 空三要求本次选择的全部影像都具有有效 RPC00B；缺失或非 RPC 影像: %1")
                                  .arg(nonRpcImages.join(QStringLiteral("、")));
        return result;
    }

    AerialTriangulationReconstructionResult
    RpcAerialTriangulationRunner::run(const PreparedAerialTriangulationInput& input,
                                      const QMap<ImageId, RpcCameraModel>& cameras) const
    {
        AerialTriangulationReconstructionResult result;
        if (input.images.size() < 2 || cameras.size() != input.images.size())
        {
            result.errorMessage = QStringLiteral("RPC 空三至少需要两张且每张都具有有效 RPC00B 的影像");
            result.summary = result.errorMessage;
            return result;
        }
        if (input.outputDir.trimmed().isEmpty() || !QDir().mkpath(input.outputDir))
        {
            result.errorMessage = QStringLiteral("无法创建 RPC 空三输出目录: %1").arg(input.outputDir);
            result.summary = result.errorMessage;
            return result;
        }

        reportProgress(input, QStringLiteral("读取 RPC 连接点图"), 0);
        std::shared_ptr<const PreparedTiePointGraph> graph = input.preparedTiePointGraph;
        if (!graph)
        {
            auto loadedGraph = std::make_shared<PreparedTiePointGraph>();
            if (!SfmAttemptRunner::readTiePointGraph(
                    input.tiePointPath, input.images, loadedGraph.get(), &result.errorMessage))
            {
                result.summary = result.errorMessage;
                return result;
            }
            graph = std::move(loadedGraph);
        }
        if (canceled(input))
        {
            result.errorMessage = QStringLiteral("用户取消");
            result.summary = result.errorMessage;
            return result;
        }

        const std::vector<std::vector<Observation>> tracks = buildTracks(*graph);
        if (tracks.empty())
        {
            result.errorMessage = QStringLiteral("RPC 连接点图中没有可交会的多视轨迹");
            result.summary = result.errorMessage;
            return result;
        }

        const double maximumRmsPixels = input.quality >= 3 ? 1.5 : (input.quality >= 2 ? 2.0 : 3.0);
        QMap<ImageId, CameraResidualAccumulator> cameraResiduals;
        std::vector<RpcPoint> points;
        points.reserve(tracks.size());
        reportProgress(input, QStringLiteral("执行 RPC 多视前方交会"), 10);
        for (std::size_t index = 0; index < tracks.size(); ++index)
        {
            if (index % 64 == 0)
            {
                if (canceled(input))
                {
                    result.errorMessage = QStringLiteral("用户取消");
                    result.summary = result.errorMessage;
                    return result;
                }
                reportProgress(input,
                               QStringLiteral("执行 RPC 多视前方交会 %1/%2").arg(index).arg(tracks.size()),
                               10 + static_cast<int>(70 * index / std::max<std::size_t>(1, tracks.size())));
            }
            RpcPoint point;
            if (intersectTrack(*graph, cameras, tracks[index], maximumRmsPixels, &point, &cameraResiduals))
            {
                points.push_back(std::move(point));
            }
        }

        constexpr int minimumAcceptedPoints = 3;
        if (points.size() < minimumAcceptedPoints)
        {
            result.errorMessage = QStringLiteral("RPC 前方交会有效点不足: %1/%2（至少需要 %3 个）")
                                      .arg(points.size())
                                      .arg(tracks.size())
                                      .arg(minimumAcceptedPoints);
            result.summary = result.errorMessage;
            return result;
        }

        reportProgress(input, QStringLiteral("建立局部 ENU 坐标并写出稀疏云"), 85);
        const RpcCameraModel::GeodeticCoordinate origin = assignLocalEnu(&points);
        sampleColors(input, *graph, &points);
        const QString plyPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse.ply"));
        if (!writePly(plyPath, points, &result.errorMessage))
        {
            result.summary = result.errorMessage;
            return result;
        }

        QJsonArray pointArray;
        double squaredErrorSum = 0.0;
        QSet<ImageId> registeredImages;
        for (const RpcPoint& point : points)
        {
            pointArray.append(pointJson(point));
            squaredErrorSum += point.rmsPixels * point.rmsPixels;
            for (const Observation& observation : point.observations)
            {
                registeredImages.insert(observation.imageId);
            }
        }
        const double meanRms = std::sqrt(squaredErrorSum / points.size());
        QJsonObject quality = xjw::common::project::buildSparseQualityMetadata(
            pointArray,
            registeredImages.size(),
            true,
            xjw::common::project::kSparseResultKindSfmSparseReconstruction,
            QString(),
            QString(),
            input.images.size());
        quality.insert(QStringLiteral("camera_model"), QStringLiteral("rpc"));
        quality.insert(QStringLiteral("absolute_sensor_model"), true);
        quality.insert(QStringLiteral("coordinate_frame"), QStringLiteral("local_enu_wgs84"));
        quality.insert(QStringLiteral("rpc_point_adjustment"), true);
        quality.insert(QStringLiteral("rpc_bias_adjustment"), false);
        quality.insert(
            QStringLiteral("quality_gate"),
            QJsonObject{{QStringLiteral("acceptable_for_mvs"), false},
                        {QStringLiteral("warnings"), QJsonArray{QStringLiteral("rpc_requires_rpc_dense_workflow")}},
                        {QStringLiteral("maximum_rpc_rms_px"), maximumRmsPixels}});

        QJsonArray perCamera;
        for (ImageId imageId : registeredImages)
        {
            const CameraResidualAccumulator accumulator = cameraResiduals.value(imageId);
            const double rms = accumulator.observationCount > 0
                                   ? std::sqrt(accumulator.squaredErrorSum / accumulator.observationCount)
                                   : 0.0;
            perCamera.append(QJsonObject{{QStringLiteral("image_id"), static_cast<qint64>(imageId)},
                                         {QStringLiteral("image_path"), input.images.value(static_cast<int>(imageId))},
                                         {QStringLiteral("camera_model"), QStringLiteral("rpc")},
                                         {QStringLiteral("observation_count"), accumulator.observationCount},
                                         {QStringLiteral("rms_reproj_px"), rms},
                                         {QStringLiteral("maximum_reproj_px"), accumulator.maximumResidual}});

            QJsonObject cameraObject = xjw::common::project::cameraToJson(cameras.value(imageId));
            cameraObject.insert(QStringLiteral("intrinsic_source"), QStringLiteral("embedded_rpc00b"));
            cameraObject.insert(QStringLiteral("pose_source"), QStringLiteral("rpc00b"));
            cameraObject.insert(QStringLiteral("adjustment_status"), QStringLiteral("rpc_fixed_model"));
            cameraObject.insert(QStringLiteral("rpc_adjustment_mode"), QStringLiteral("fixed_sensor_point_only"));
            result.pendingCamUpdates.insert(
                xjw::common::project::normalizePath(input.images.value(static_cast<int>(imageId))), cameraObject);
        }

        const QJsonObject originJson{{QStringLiteral("longitude_deg"), origin[0]},
                                     {QStringLiteral("latitude_deg"), origin[1]},
                                     {QStringLiteral("ellipsoidal_height_m"), origin[2]}};
        const QString sidecarPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse_points.json"));
        QJsonObject diagnostics{
            {QStringLiteral("camera_model"), QStringLiteral("rpc")},
            {QStringLiteral("rpc_model_fixed"), true},
            {QStringLiteral("rpc_bias_adjustment_applied"), false},
            {QStringLiteral("rpc_point_adjustment_applied"), true},
            {QStringLiteral("input_track_count"), static_cast<qint64>(tracks.size())},
            {QStringLiteral("accepted_track_count"), static_cast<qint64>(points.size())},
            {QStringLiteral("rejected_track_count"), static_cast<qint64>(tracks.size() - points.size())},
            {QStringLiteral("maximum_reprojection_error_px"), maximumRmsPixels},
            {QStringLiteral("local_enu_origin_wgs84"), originJson}};
        QJsonObject sidecar = xjw::common::project::mergeSparseQualityIntoRecord(
            QJsonObject{{QStringLiteral("schema"), QStringLiteral("plascan.rpc_aerial_triangulation.v1")},
                        {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")},
                        {QStringLiteral("camera_model"), QStringLiteral("rpc")},
                        {QStringLiteral("coordinate_frame"), QStringLiteral("local_enu_wgs84")},
                        {QStringLiteral("local_enu_origin_wgs84"), originJson},
                        {QStringLiteral("points"), pointArray},
                        {QStringLiteral("per_camera"), perCamera},
                        {QStringLiteral("sfm_diagnostics"), diagnostics}},
            quality);
        QString writeError;
        if (!xjw::common::io::writeFileBytesAtomic(
                sidecarPath, QJsonDocument(sidecar).toJson(QJsonDocument::Compact), &writeError))
        {
            result.errorMessage =
                writeError.isEmpty() ? QStringLiteral("无法写入 RPC 空三质量文件: %1").arg(sidecarPath) : writeError;
            result.summary = result.errorMessage;
            return result;
        }

        result.success = true;
        result.numRegisteredImages = registeredImages.size();
        result.numPoints3D = static_cast<int>(points.size());
        result.meanReprojError = meanRms;
        result.baRmsBefore = meanRms;
        result.baRmsAfter = meanRms;
        result.baTracksTotal = static_cast<int>(tracks.size());
        result.baTracksOptimized = static_cast<int>(points.size());
        result.baTracksFiltered = static_cast<int>(tracks.size() - points.size());
        result.sparseCloudPath = plyPath;
        result.qualityMetadata = quality;
        result.sfmDiagnostics = diagnostics;
        result.perCameraResiduals = perCamera;
        result.resultRecordExtra = xjw::common::project::mergeSparseQualityIntoRecord(
            QJsonObject{
                {QStringLiteral("source"), QStringLiteral("aerial_triangulation")},
                {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")},
                {QStringLiteral("camera_model"), QStringLiteral("rpc")},
                {QStringLiteral("absolute_sensor_model"), true},
                {QStringLiteral("coordinate_frame"), QStringLiteral("local_enu_wgs84")},
                {QStringLiteral("local_enu_origin_wgs84"), originJson},
                {QStringLiteral("files"), QJsonObject{{QStringLiteral("sparse_cloud_points_json"), sidecarPath}}},
                {QStringLiteral("sfm_diagnostics"), diagnostics}},
            quality);
        result.summary = QStringLiteral("RPC 空三成功：注册 %1/%2 张影像，%3 个地面点，RMS %4 px")
                             .arg(result.numRegisteredImages)
                             .arg(input.images.size())
                             .arg(result.numPoints3D)
                             .arg(result.meanReprojError, 0, 'f', 3);
        reportProgress(input, QStringLiteral("RPC 空三完成"), 100);
        return result;
    }

} // namespace xjw::aerial_triangulation
