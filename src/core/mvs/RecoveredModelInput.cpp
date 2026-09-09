#include "RecoveredModelInput.h"
#include "io/PathIO.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QUuid>
#include <QtEndian>

#include <bit>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xjw::mvs
{
    namespace
    {
        void require(bool valid, const QString& message)
        {
            if (!valid)
            {
                throw std::runtime_error(message.toStdString());
            }
        }

        QJsonArray values(std::initializer_list<double> data)
        {
            QJsonArray result;
            for (double value : data)
            {
                require(std::isfinite(value), QStringLiteral("Non-finite recovered camera/region"));
                result.append(value);
            }
            return result;
        }

        template <std::size_t N> QJsonArray values(const std::array<double, N>& data)
        {
            QJsonArray result;
            for (double value : data)
            {
                require(std::isfinite(value), QStringLiteral("Non-finite recovered matrix"));
                result.append(value);
            }
            return result;
        }

        std::vector<double> numbers(const QJsonValue& value, qsizetype size)
        {
            const auto array = value.toArray();
            require(array.size() == size, QStringLiteral("Invalid recovered camera/region array"));
            std::vector<double> result;
            for (const auto item : array)
            {
                require(item.isDouble() && std::isfinite(item.toDouble()), QStringLiteral("Invalid recovered scalar"));
                result.push_back(item.toDouble());
            }
            return result;
        }

        void save(const QString& path, const QByteArray& bytes)
        {
            QSaveFile file(path);
            require(file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit(),
                    QStringLiteral("Cannot write recovered model input: %1").arg(path));
        }

        QByteArray read(const QString& path, qint64 expected_size)
        {
            QFile file(path);
            require(file.open(QIODevice::ReadOnly) && file.size() == expected_size,
                    QStringLiteral("Missing or wrong-size recovered model input: %1").arg(path));
            auto bytes = file.readAll();
            require(bytes.size() == expected_size, QStringLiteral("Truncated recovered input: %1").arg(path));
            return bytes;
        }

        std::size_t pixelCount(const metmodel::Camera& camera, std::size_t level)
        {
            const auto scale = std::size_t{4} << level;
            require(camera.image.width > 0 && camera.image.height > 0 && camera.image.width <= 100000 &&
                        camera.image.height <= 100000 && camera.image.width % 16 == 0 && camera.image.height % 16 == 0,
                    QStringLiteral("Recovered OOC requires positive image dimensions divisible by 16"));
            const auto count = (camera.image.width / scale) * (camera.image.height / scale);
            require(count <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 4),
                    QStringLiteral("Recovered depth plane exceeds supported size"));
            return count;
        }

        void publishStagedDirectory(const QString& staging, const QString& destination, bool replaceExisting)
        {
            const QFileInfo destination_info(destination);
            if (!destination_info.exists())
            {
                require(QDir().rename(staging, destination),
                        QStringLiteral("Cannot publish recovered model input: %1").arg(destination));
                return;
            }
            require(replaceExisting, QStringLiteral("Recovered model input already exists: %1").arg(destination));
            require(destination_info.isDir() && !destination_info.isSymLink(),
                    QStringLiteral("Recovered model input destination is not a replaceable directory: %1")
                        .arg(destination));

            const QString backup =
                destination + QStringLiteral(".backup-%1").arg(QUuid::createUuid().toString(QUuid::WithoutBraces));
            require(QDir().rename(destination, backup),
                    QStringLiteral("Cannot stage existing recovered model input for replacement: %1").arg(destination));
            bool published = false;
            const auto finish_transaction = qScopeGuard(
                [&]()
                {
                    if (published)
                    {
                        QDir(backup).removeRecursively();
                    }
                    else if (!QFileInfo::exists(destination))
                    {
                        QDir().rename(backup, destination);
                    }
                });
            require(QDir().rename(staging, destination),
                    QStringLiteral("Cannot publish replacement recovered model input: %1").arg(destination));
            published = true;
        }
    } // namespace

    void writeRecoveredModelInput(const QString& directory,
                                  const metmodel::Scene& scene,
                                  const metmodel::RecoveredPatchMatchD4SceneOutput& depth,
                                  bool replaceExisting)
    {
        require(!scene.cameras.empty() && scene.cameras.size() == depth.cameras.size() && scene.region.specified,
                QStringLiteral("Incomplete recovered model scene"));
        require(replaceExisting || !QFileInfo::exists(directory),
                QStringLiteral("Recovered model input already exists: %1").arg(directory));
        QTemporaryDir staging(directory + QStringLiteral(".staging-XXXXXX"));
        require(staging.isValid(), QStringLiteral("Cannot create recovered model input staging directory"));
        QJsonArray cameras;
        for (std::size_t i = 0; i < scene.cameras.size(); ++i)
        {
            const auto& camera = scene.cameras[i];
            const auto& output = depth.cameras[i];
            require(camera.index == i && camera.aligned && output.patchmatch.camera_index == i,
                    QStringLiteral("Recovered camera identity/order mismatch"));
            const auto& m = camera.model;
            QJsonObject object{
                {QStringLiteral("index"), static_cast<double>(i)},
                {QStringLiteral("width"), static_cast<double>(camera.image.width)},
                {QStringLiteral("height"), static_cast<double>(camera.image.height)},
                {QStringLiteral("calibration"),
                 values({m.f,
                         m.cx,
                         m.cy,
                         m.b1,
                         m.b2,
                         m.k1,
                         m.k2,
                         m.k3,
                         m.k4,
                         m.p1,
                         m.p2,
                         m.p3,
                         m.p4,
                         m.cx_offset,
                         m.cy_offset})},
                {QStringLiteral("rotation"), values(camera.pose.rotation.v)},
                {QStringLiteral("translation"),
                 values({camera.pose.translation.x, camera.pose.translation.y, camera.pose.translation.z})},
                {QStringLiteral("center"), values({camera.center.x, camera.center.y, camera.center.z})}};
            const auto image_path = camera.path.u8string();
            object[QStringLiteral("image_path")] = QString::fromUtf8(reinterpret_cast<const char*>(image_path.data()),
                                                                     static_cast<qsizetype>(image_path.size()));
            QJsonArray hashes;
            for (std::size_t level = 0; level < 3; ++level)
            {
                const auto& plane = output.voting.depth_after_components[level];
                require(plane.size() == pixelCount(camera, level), QStringLiteral("Missing voted depth pyramid level"));
                QByteArray bytes(static_cast<qsizetype>(plane.size() * sizeof(float)), Qt::Uninitialized);
                for (std::size_t p = 0; p < plane.size(); ++p)
                {
                    require(std::isfinite(plane[p]) && plane[p] >= 0, QStringLiteral("Invalid voted depth value"));
                    qToLittleEndian(std::bit_cast<quint32>(plane[p]), bytes.data() + p * 4);
                }
                save(QDir(staging.path()).filePath(QStringLiteral("camera_%1_d%2.bin").arg(i).arg(4U << level)), bytes);
                hashes.append(QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex()));
            }
            object[QStringLiteral("sha256")] = hashes;
            cameras.append(object);
        }
        const auto& region = scene.region;
        QJsonObject manifest{
            {QStringLiteral("schema"), QStringLiteral("plascan.recovered-model-input.v1")},
            {QStringLiteral("depth_semantics"), QStringLiteral("d4-d8-d16-voting-after-components")},
            {QStringLiteral("cameras"), cameras},
            {QStringLiteral("region_rotation"), values(region.rotation)},
            {QStringLiteral("region_center"), values({region.center.x, region.center.y, region.center.z})},
            {QStringLiteral("region_size"), values({region.size.x, region.size.y, region.size.z})}};
        save(QDir(staging.path()).filePath(QStringLiteral("manifest.json")), QJsonDocument(manifest).toJson());
        publishStagedDirectory(staging.path(), directory, replaceExisting);
        staging.setAutoRemove(false);
    }

    void readRecoveredModelInput(const QString& directory,
                                 metmodel::Scene& scene,
                                 metmodel::RecoveredPatchMatchD4SceneOutput& depth)
    {
        const auto path = QDir(directory).filePath(QStringLiteral("manifest.json"));
        const auto manifest_size = QFileInfo(path).size();
        require(manifest_size > 0 && manifest_size <= 64 * 1024 * 1024,
                QStringLiteral("Invalid model input manifest size"));
        QJsonParseError error;
        const auto document = QJsonDocument::fromJson(read(path, manifest_size), &error);
        const auto manifest = document.object();
        require(error.error == QJsonParseError::NoError && document.isObject() &&
                    manifest.value(QStringLiteral("schema")).toString() ==
                        QStringLiteral("plascan.recovered-model-input.v1") &&
                    manifest.value(QStringLiteral("depth_semantics")).toString() ==
                        QStringLiteral("d4-d8-d16-voting-after-components"),
                QStringLiteral("Unsupported recovered model input manifest"));
        metmodel::Scene loaded;
        metmodel::RecoveredPatchMatchD4SceneOutput loaded_depth;
        const auto rotation = numbers(manifest.value(QStringLiteral("region_rotation")), 9);
        std::copy(rotation.begin(), rotation.end(), loaded.region.rotation.begin());
        const auto center = numbers(manifest.value(QStringLiteral("region_center")), 3);
        const auto size = numbers(manifest.value(QStringLiteral("region_size")), 3);
        require(size[0] > 0 && size[1] > 0 && size[2] > 0, QStringLiteral("Invalid recovered region size"));
        loaded.region.center = {center[0], center[1], center[2]};
        loaded.region.size = {size[0], size[1], size[2]};
        loaded.region.specified = true;
        const auto cameras = manifest.value(QStringLiteral("cameras")).toArray();
        require(!cameras.empty(), QStringLiteral("Empty recovered model camera list"));
        for (qsizetype i = 0; i < cameras.size(); ++i)
        {
            const auto object = cameras[i].toObject();
            metmodel::Camera camera;
            camera.index = static_cast<std::size_t>(i);
            require(object.value(QStringLiteral("index")).toInteger(-1) == i,
                    QStringLiteral("Model camera order mismatch"));
            camera.aligned = true;
            camera.path = xjw::common::io::toFilesystemPath(object.value(QStringLiteral("image_path")).toString());
            camera.image.width = static_cast<std::size_t>(object.value(QStringLiteral("width")).toInteger());
            camera.image.height = static_cast<std::size_t>(object.value(QStringLiteral("height")).toInteger());
            const auto c = numbers(object.value(QStringLiteral("calibration")), 15);
            auto& m = camera.model;
            m.f = c[0];
            m.cx = c[1];
            m.cy = c[2];
            m.b1 = c[3];
            m.b2 = c[4];
            m.k1 = c[5];
            m.k2 = c[6];
            m.k3 = c[7];
            m.k4 = c[8];
            m.p1 = c[9];
            m.p2 = c[10];
            m.p3 = c[11];
            m.p4 = c[12];
            m.cx_offset = c[13];
            m.cy_offset = c[14];
            require(m.f > 0 && m.f + m.b1 > 0, QStringLiteral("Invalid model camera focal length"));
            const auto r = numbers(object.value(QStringLiteral("rotation")), 9);
            std::copy(r.begin(), r.end(), camera.pose.rotation.v.begin());
            const auto t = numbers(object.value(QStringLiteral("translation")), 3);
            const auto p = numbers(object.value(QStringLiteral("center")), 3);
            camera.pose.translation = {t[0], t[1], t[2]};
            camera.center = {p[0], p[1], p[2]};
            camera.pose.center = camera.center;
            metmodel::RecoveredPatchMatchD4CameraOutput output;
            output.patchmatch.camera_index = camera.index;
            const auto hashes = object.value(QStringLiteral("sha256")).toArray();
            require(hashes.size() == 3, QStringLiteral("Missing model depth checksums"));
            for (std::size_t level = 0; level < 3; ++level)
            {
                const auto count = pixelCount(camera, level);
                const auto bytes =
                    read(QDir(directory).filePath(QStringLiteral("camera_%1_d%2.bin").arg(i).arg(4U << level)),
                         static_cast<qint64>(count * 4));
                require(QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex()) ==
                            hashes[static_cast<qsizetype>(level)].toString(),
                        QStringLiteral("Model depth checksum mismatch"));
                auto& plane = output.voting.depth_after_components[level];
                plane.resize(count);
                for (std::size_t j = 0; j < count; ++j)
                {
                    plane[j] = std::bit_cast<float>(qFromLittleEndian<quint32>(bytes.constData() + j * 4));
                    require(std::isfinite(plane[j]) && plane[j] >= 0, QStringLiteral("Invalid model depth sample"));
                }
            }
            loaded.cameras.push_back(std::move(camera));
            loaded_depth.cameras.push_back(std::move(output));
        }
        scene = std::move(loaded);
        depth = std::move(loaded_depth);
    }
} // namespace xjw::mvs
