#include "PlaMatchHctPairPreselector.h"

#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosParallelism.h"
#include "io/PathIO.h"
#include "plamatch_hct/PlaMatchHctAlgorithm.h"
#include "plamatch_hct/PlaMatchHctFeaturePayload.h"

#include "metalign/gpu.hpp"
#include "metalign/matching.hpp"

#include <QFileInfo>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <numeric>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace xjw::matchphotos
{
    namespace
    {

        int allPairCount(int imageCount)
        {
            return imageCount > 1 ? imageCount * (imageCount - 1) / 2 : 0;
        }

        int coarseCandidateCount(int imageCount, bool referencePreselection)
        {
            if (!referencePreselection)
            {
                return allPairCount(imageCount);
            }

            std::set<metalign::ImagePair> candidates;
            const std::size_t count = static_cast<std::size_t>(std::max(0, imageCount));
            for (std::size_t camera = 0; camera < count; ++camera)
            {
                for (std::size_t delta = 1; delta <= 24 && delta < count; ++delta)
                {
                    const std::size_t other = (camera + delta) % count;
                    if (camera != other)
                    {
                        candidates.insert(metalign::ImagePair{camera, other}.ordered());
                    }
                }
            }
            return static_cast<int>(candidates.size());
        }

        std::unique_ptr<metalign::DescriptorAccelerator> createAccelerator(image_matching::SiftComputeBackend backend,
                                                                           int deviceIndex)
        {
            if (backend == image_matching::SiftComputeBackend::Cpu)
            {
                return {};
            }

            const char* backendName = nullptr;
            if (backend == image_matching::SiftComputeBackend::Cuda)
            {
                backendName = "cuda";
            }
            else if (backend == image_matching::SiftComputeBackend::OpenCl)
            {
                backendName = "opencl";
            }
            else
            {
                throw std::invalid_argument("PlaMatch-HCT coarse preselection received an unresolved backend");
            }
            return metalign::create_descriptor_accelerator(
                backendName, deviceIndex, std::numeric_limits<std::uint64_t>::max(), false);
        }

        bool cameraForImage(const QMap<QString, FramePinholeCamera>& cameras,
                            const QString& imagePath,
                            FramePinholeCamera* camera)
        {
            if (!camera)
            {
                return false;
            }

            const QFileInfo imageInfo(imagePath);
            const QStringList keys = {
                imagePath, imageInfo.absoluteFilePath(), imageInfo.fileName(), imageInfo.completeBaseName()};
            for (const QString& key : keys)
            {
                const auto found = cameras.constFind(key);
                if (found != cameras.constEnd() && found->isValid())
                {
                    *camera = *found;
                    return true;
                }
            }
            return false;
        }

        std::optional<std::array<double, 3>> positionForImage(const QMap<QString, std::array<double, 3>>& positions,
                                                              const QMap<QString, FramePinholeCamera>& cameras,
                                                              const QString& imagePath)
        {
            const QFileInfo imageInfo(imagePath);
            const QStringList keys = {
                imagePath, imageInfo.absoluteFilePath(), imageInfo.fileName(), imageInfo.completeBaseName()};
            for (const QString& key : keys)
            {
                const auto found = positions.constFind(key);
                if (found != positions.constEnd() && std::isfinite((*found)[0]) && std::isfinite((*found)[1]) &&
                    std::isfinite((*found)[2]))
                {
                    return *found;
                }
            }

            FramePinholeCamera camera;
            if (cameraForImage(cameras, imagePath, &camera))
            {
                const std::array<double, 3> center = camera.cameraCenter();
                if (std::isfinite(center[0]) && std::isfinite(center[1]) && std::isfinite(center[2]))
                {
                    return center;
                }
            }
            return std::nullopt;
        }

        double squaredDistance(const std::array<double, 3>& left, const std::array<double, 3>& right)
        {
            const double dx = left[0] - right[0];
            const double dy = left[1] - right[1];
            const double dz = left[2] - right[2];
            return dx * dx + dy * dy + dz * dz;
        }

        std::set<metalign::ImagePair> referencePairs(const QStringList& images,
                                                     const QMap<QString, FramePinholeCamera>& referenceCameras,
                                                     const QMap<QString, std::array<double, 3>>& referencePositions,
                                                     ReferencePreselectionMode mode,
                                                     int neighborCount,
                                                     bool* usedIndexFallback)
        {
            if (usedIndexFallback)
            {
                *usedIndexFallback = false;
            }

            std::set<metalign::ImagePair> pairs;
            const std::size_t count = static_cast<std::size_t>(images.size());
            if (mode == ReferencePreselectionMode::Sequential)
            {
                // 与参考的普通影像目录适配器保持一致：没有序列 group 元数据时，
                // 每张影像属于独立 singleton group，因此不添加猜测的线性邻接边。
                return pairs;
            }

            std::vector<std::optional<std::array<double, 3>>> positions(count);
            bool hasReferencePosition = false;
            for (std::size_t index = 0; index < count; ++index)
            {
                const auto position =
                    positionForImage(referencePositions, referenceCameras, images.at(static_cast<int>(index)));
                if (!position)
                {
                    continue;
                }
                positions[index] = *position;
                hasReferencePosition = true;
            }

            const std::size_t neighbors = static_cast<std::size_t>(std::max(1, neighborCount));
            if (!hasReferencePosition)
            {
                if (usedIndexFallback)
                {
                    *usedIndexFallback = true;
                }
                for (std::size_t first = 0; first < count; ++first)
                {
                    for (std::size_t delta = 1; delta <= neighbors && first + delta < count; ++delta)
                    {
                        pairs.insert({first, first + delta});
                    }
                }
                return pairs;
            }

            for (std::size_t first = 0; first < count; ++first)
            {
                if (!positions[first])
                {
                    continue;
                }
                std::vector<std::pair<double, std::size_t>> scored;
                for (std::size_t second = 0; second < count; ++second)
                {
                    if (first != second && positions[second])
                    {
                        scored.emplace_back(squaredDistance(*positions[first], *positions[second]), second);
                    }
                }

                const std::size_t keep = std::min(neighbors, scored.size());
                std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(keep), scored.end());
                for (std::size_t index = 0; index < keep; ++index)
                {
                    pairs.insert(metalign::ImagePair{first, scored[index].second}.ordered());
                }
            }
            return pairs;
        }

        PairSource referencePairSource(ReferencePreselectionMode mode)
        {
            switch (mode)
            {
            case ReferencePreselectionMode::Source:
                return PairSource::PlaMatchReferenceSource;
            case ReferencePreselectionMode::Estimated:
                return PairSource::PlaMatchReferenceEstimated;
            case ReferencePreselectionMode::Sequential:
                return PairSource::PlaMatchReferenceSequential;
            }
            return PairSource::PlaMatchReferenceSource;
        }

        std::vector<int> selectCoarseRows(const image_matching::FeatureSet& features)
        {
            constexpr int maximumRows = 2048;
            std::vector<int> ranked(static_cast<std::size_t>(features.size()));
            std::iota(ranked.begin(), ranked.end(), 0);
            std::stable_sort(ranked.begin(),
                             ranked.end(),
                             [&](int left, int right)
                             {
                                 const float leftScore = features.scores[static_cast<std::size_t>(left)];
                                 const float rightScore = features.scores[static_cast<std::size_t>(right)];
                                 return leftScore == rightScore ? left < right : leftScore > rightScore;
                             });
            if (features.size() <= maximumRows)
            {
                return ranked;
            }

            const double aspect =
                std::clamp(static_cast<double>(features.imageWidth) / std::max(1, features.imageHeight), 0.25, 4.0);
            const int columns = std::max(1, static_cast<int>(std::round(std::sqrt(maximumRows * aspect))));
            const int rows = std::max(1, static_cast<int>(std::ceil(static_cast<double>(maximumRows) / columns)));
            std::vector<unsigned char> occupied(static_cast<std::size_t>(columns * rows), 0U);
            std::vector<unsigned char> selected(static_cast<std::size_t>(features.size()), 0U);
            std::vector<int> result;
            result.reserve(maximumRows);
            for (const int index : ranked)
            {
                const cv::Point2f point = features.keypoints[static_cast<std::size_t>(index)].pt;
                const int column =
                    std::clamp(static_cast<int>(point.x * columns / std::max(1, features.imageWidth)), 0, columns - 1);
                const int row =
                    std::clamp(static_cast<int>(point.y * rows / std::max(1, features.imageHeight)), 0, rows - 1);
                const std::size_t cell = static_cast<std::size_t>(row * columns + column);
                if (occupied[cell] != 0U)
                {
                    continue;
                }
                occupied[cell] = 1U;
                selected[static_cast<std::size_t>(index)] = 1U;
                result.push_back(index);
                if (static_cast<int>(result.size()) == maximumRows)
                {
                    return result;
                }
            }
            for (const int index : ranked)
            {
                if (selected[static_cast<std::size_t>(index)] == 0U)
                {
                    result.push_back(index);
                    if (static_cast<int>(result.size()) == maximumRows)
                    {
                        break;
                    }
                }
            }
            return result;
        }

        metalign::Descriptor binaryRankSignature(const cv::Mat& descriptor)
        {
            metalign::Descriptor result{};
            const float* values = descriptor.ptr<float>();
            const int dimension = descriptor.cols;
            for (std::size_t bit = 0; bit < metalign::kDescriptorSize * 8U; ++bit)
            {
                const int left = static_cast<int>((bit * 37U + 17U) % static_cast<std::size_t>(dimension));
                int right = static_cast<int>((bit * 101U + 53U) % static_cast<std::size_t>(dimension));
                if (right == left)
                {
                    right = (right + 1) % dimension;
                }
                if (std::isfinite(values[left]) && std::isfinite(values[right]) && values[left] > values[right])
                {
                    result[bit / 8U] |= static_cast<std::uint8_t>(1U << (bit % 8U));
                }
            }
            return result;
        }

        metalign::FeatureSet adaptCoarseFeatures(const QString& imagePath, const image_matching::FeatureSet& features)
        {
            if (!features.isConsistent() || features.descriptors.type() != CV_32F || features.descriptors.cols < 2)
            {
                throw std::invalid_argument("coarse preselection requires consistent floating-point descriptors");
            }

            const std::vector<int> selectedRows = selectCoarseRows(features);
            metalign::FeatureSet result;
            result.path = common::io::toUtf8Path(imagePath);
            result.image_width = static_cast<std::size_t>(features.imageWidth);
            result.image_height = static_cast<std::size_t>(features.imageHeight);
            result.keypoints.reserve(selectedRows.size());
            for (std::size_t row = 0; row < selectedRows.size(); ++row)
            {
                const int sourceRow = selectedRows[row];
                const cv::KeyPoint& source = features.keypoints[static_cast<std::size_t>(sourceRow)];
                metalign::Keypoint keypoint;
                keypoint.x = source.pt.x;
                keypoint.y = source.pt.y;
                keypoint.scale = source.size;
                keypoint.orientation = source.angle;
                keypoint.response = source.response;
                keypoint.octave = source.octave;
                keypoint.detector_id = row;
                keypoint.source_id = row;
                keypoint.descriptor = binaryRankSignature(features.descriptors.row(sourceRow));
                keypoint.laplacian_sign = 1;
                result.keypoints.push_back(std::move(keypoint));
            }
            result.source_keypoint_count = result.keypoints.size();
            return result;
        }

    } // namespace

    bool PlaMatchHctPairPreselector::select(const QStringList& images,
                                            const MatchPhotosFeatureCache& featureCache,
                                            const MatchPhotosOptions& options,
                                            const QMap<QString, FramePinholeCamera>& referenceCameras,
                                            image_matching::SiftComputeBackend backend,
                                            int deviceIndex,
                                            PairSelectionResult* output,
                                            PlaMatchHctPairPreselectionStats* stats,
                                            std::atomic_bool* cancelFlag,
                                            QString* errorMessage)
    {
        return selectWithPositions(images,
                                   featureCache,
                                   options,
                                   referenceCameras,
                                   {},
                                   backend,
                                   deviceIndex,
                                   output,
                                   stats,
                                   cancelFlag,
                                   errorMessage);
    }

    bool PlaMatchHctPairPreselector::selectWithPositions(const QStringList& images,
                                                         const MatchPhotosFeatureCache& featureCache,
                                                         const MatchPhotosOptions& options,
                                                         const QMap<QString, FramePinholeCamera>& referenceCameras,
                                                         const QMap<QString, std::array<double, 3>>& referencePositions,
                                                         image_matching::SiftComputeBackend backend,
                                                         int deviceIndex,
                                                         PairSelectionResult* output,
                                                         PlaMatchHctPairPreselectionStats* stats,
                                                         std::atomic_bool* cancelFlag,
                                                         QString* errorMessage)
    {
        if (errorMessage)
        {
            errorMessage->clear();
        }
        if (!output)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("内部错误：PlaMatch-HCT 预选输出为空");
            }
            return false;
        }
        if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("PlaMatch-HCT 预选已取消");
            }
            return false;
        }

        PlaMatchHctPairPreselectionStats localStats;
        std::vector<metalign::FeatureSet> coarseFeatures;
        std::vector<std::unique_ptr<metalign::CpuDescriptorIndex>> adaptedIndices;
        std::vector<const metalign::CpuDescriptorIndex*> coarseIndices;
        coarseFeatures.reserve(static_cast<std::size_t>(images.size()));
        adaptedIndices.reserve(static_cast<std::size_t>(images.size()));
        coarseIndices.reserve(static_cast<std::size_t>(images.size()));
        if (options.useGenericPreselection)
        {
            for (const QString& imagePath : images)
            {
                const std::shared_ptr<const image_matching::FeatureSet> features = featureCache.find(imagePath);
                const auto payload =
                    features
                        ? std::dynamic_pointer_cast<const image_matching::PlaMatchHctFeaturePayload>(features->payload)
                        : nullptr;
                if (!features)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("统一 coarse 预选缺少有效特征：%1").arg(imagePath);
                    }
                    return false;
                }
                if (features->sourceAlgorithm == QLatin1String(image_matching::kPlaMatchHctAlgorithmId) && payload &&
                    payload->isConsistent(features->keypoints.size()))
                {
                    coarseFeatures.push_back(payload->coarseFeatures());
                    coarseIndices.push_back(&payload->coarseIndex());
                    continue;
                }
                try
                {
                    coarseFeatures.push_back(adaptCoarseFeatures(imagePath, *features));
                    adaptedIndices.push_back(std::make_unique<metalign::CpuDescriptorIndex>(coarseFeatures.back()));
                    coarseIndices.push_back(adaptedIndices.back().get());
                    localStats.usedDescriptorAdapter = true;
                }
                catch (const std::exception& error)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("无法从 %1 特征构造 coarse 预选：%2")
                                            .arg(QString::fromStdString(features->sourceAlgorithm),
                                                 QString::fromUtf8(error.what()));
                    }
                    return false;
                }
            }
        }

        try
        {
            std::set<metalign::ImagePair> genericPairs;
            std::unique_ptr<metalign::DescriptorAccelerator> accelerator;
            if (options.useGenericPreselection)
            {
                accelerator = createAccelerator(backend, deviceIndex);
                metalign::MatchPhotosOptions vendorOptions;
                vendorOptions.generic_preselection = true;
                vendorOptions.reference_preselection = options.useReferencePreselection;
                const int imageCount = images.size();
                localStats.coarseCandidateCount = coarseCandidateCount(imageCount, options.useReferencePreselection);
                const std::size_t workerCount = static_cast<std::size_t>(resolveGeometryVerificationWorkers(
                    std::max(1, localStats.coarseCandidateCount), std::thread::hardware_concurrency()));
                genericPairs = metalign::select_generic_image_pairs_from_coarse(
                    coarseFeatures, vendorOptions, workerCount, accelerator.get(), &coarseIndices);
            }
            localStats.genericSelectedCount = static_cast<int>(genericPairs.size());

            std::set<metalign::ImagePair> selected = genericPairs;
            std::set<metalign::ImagePair> selectedByReference;
            if (options.useReferencePreselection)
            {
                selectedByReference = referencePairs(images,
                                                     referenceCameras,
                                                     referencePositions,
                                                     options.referencePreselectionMode,
                                                     options.referencePreselectionNeighbors,
                                                     &localStats.usedReferenceIndexFallback);
                selected.insert(selectedByReference.begin(), selectedByReference.end());
            }
            localStats.referenceSelectedCount = static_cast<int>(selectedByReference.size());

            if (selected.empty())
            {
                localStats.usedAllPairsFallback = true;
                for (std::size_t first = 0; first < static_cast<std::size_t>(images.size()); ++first)
                {
                    for (std::size_t second = first + 1; second < static_cast<std::size_t>(images.size()); ++second)
                    {
                        selected.insert({first, second});
                    }
                }
            }

            PairSelectionResult result;
            result.imageCount = images.size();
            result.allPairCount = allPairCount(result.imageCount);
            result.restrictPairs = static_cast<int>(selected.size()) != result.allPairCount;
            result.candidates.reserve(selected.size());
            const PairSource referenceSource = referencePairSource(options.referencePreselectionMode);
            for (const metalign::ImagePair pair : selected)
            {
                PairCandidate candidate;
                candidate.pair = {static_cast<int>(pair.first), static_cast<int>(pair.second)};
                candidate.pairKey = makePairKey(images, candidate.pair.indexA, candidate.pair.indexB);
                if (genericPairs.contains(pair))
                {
                    appendPairSource(&candidate, PairSource::PlaMatchGeneric);
                    candidate.priorityScore += 80.0;
                }
                if (selectedByReference.contains(pair))
                {
                    appendPairSource(&candidate, referenceSource);
                    candidate.priorityScore += 100.0;
                }
                if (localStats.usedAllPairsFallback)
                {
                    appendPairSource(&candidate, PairSource::Exhaustive);
                    candidate.priorityScore += 10.0;
                }
                result.candidates.push_back(std::move(candidate));
            }

            if (options.pairPolicy.maxPairs > 0 &&
                static_cast<int>(result.candidates.size()) > options.pairPolicy.maxPairs)
            {
                std::sort(result.candidates.begin(),
                          result.candidates.end(),
                          [](const PairCandidate& left, const PairCandidate& right)
                          {
                              if (left.priorityScore != right.priorityScore)
                              {
                                  return left.priorityScore > right.priorityScore;
                              }
                              return left.pairKey < right.pairKey;
                          });
                result.candidates.resize(static_cast<std::size_t>(options.pairPolicy.maxPairs));
                result.restrictPairs = true;
            }
            for (const PairCandidate& candidate : result.candidates)
            {
                result.allowedPairKeys.append(candidate.pairKey);
            }

            localStats.finalSelectedCount = static_cast<int>(result.candidates.size());
            const QString backendDetail =
                accelerator ? QStringLiteral("%1 / %2").arg(QString::fromStdString(accelerator->backend_name()),
                                                            QString::fromStdString(accelerator->device_name()))
                            : QStringLiteral("cpu");
            localStats.detail =
                QStringLiteral("PlaMatch-HCT 参考兼容预选：coarse 候选 %1，"
                               "通用保留 %2，参考加入 %3，最终 %4/%5，模式 %6，后端 %7")
                    .arg(localStats.coarseCandidateCount)
                    .arg(localStats.genericSelectedCount)
                    .arg(localStats.referenceSelectedCount)
                    .arg(localStats.finalSelectedCount)
                    .arg(result.allPairCount)
                    .arg(referencePreselectionModeName(options.referencePreselectionMode), backendDetail);
            if (localStats.usedReferenceIndexFallback)
            {
                localStats.detail += QStringLiteral("；参考坐标为空，使用索引邻域回退");
            }
            if (options.useReferencePreselection &&
                options.referencePreselectionMode == ReferencePreselectionMode::Sequential)
            {
                localStats.detail += QStringLiteral("；普通影像目录无序列组元数据，序列增补为 0 对");
            }
            if (localStats.usedAllPairsFallback)
            {
                localStats.detail += QStringLiteral("；空候选回退全量像对");
            }
            if (localStats.usedDescriptorAdapter)
            {
                localStats.detail += QStringLiteral("；复用正式算法描述子生成 2048 点 coarse 视图");
            }
            result.detail = localStats.detail;
            *output = std::move(result);
            if (stats)
            {
                *stats = std::move(localStats);
            }
        }
        catch (const std::exception& error)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("PlaMatch-HCT 预选失败：%1").arg(QString::fromUtf8(error.what()));
            }
            return false;
        }

        if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("PlaMatch-HCT 预选已取消");
            }
            return false;
        }
        return true;
    }

} // namespace xjw::matchphotos
