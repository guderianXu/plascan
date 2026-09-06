#include "PlaMatchHctAlgorithm.h"

#include "PlaMatchHctFeaturePayload.h"
#include "PlaMatchHctImage.h"
#include "../ImageMatchingRegistry.h"
#include "io/PathIO.h"

#include "metalign/features.hpp"
#include "metalign/gpu.hpp"
#include "metalign/matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <vector>

namespace xjw::image_matching
{
    namespace
    {

        std::filesystem::path fileSystemPath(const QString& path)
        {
            return common::io::toFilesystemPath(path);
        }

        ImageMatchingAlgorithmDescriptor plaMatchHctDescriptor()
        {
            ImageMatchingAlgorithmDescriptor value;
            value.id = QString::fromLatin1(kPlaMatchHctAlgorithmId);
            value.displayName = QStringLiteral("PlaMatch-HCT（空间一致性二进制匹配）");
            value.version = kPlaMatchHctAlgorithmVersion;
            value.inputModel = AlgorithmInputModel::ReusableFeatures;
            value.requiresCuda = false;
            value.suppliesStableFeatureIds = true;
            value.requiresColorInput = true;
            value.suppliesCoarsePairPreselection = true;
            value.supportsBatchFeatureMatching = true;
            return value;
        }

        void transformFeatureRows(std::vector<metalign::Keypoint>& rows, double scale, double offsetX, double offsetY)
        {
            if (!(scale > 0.0))
            {
                return;
            }
            for (metalign::Keypoint& keypoint : rows)
            {
                keypoint.x = keypoint.x * scale + offsetX;
                keypoint.y = keypoint.y * scale + offsetY;
                keypoint.scale *= scale;
            }
        }

        float orientationDegrees(double radians)
        {
            constexpr double radiansToDegrees = 180.0 / 3.14159265358979323846;
            double degrees = std::fmod(radians * radiansToDegrees, 360.0);
            if (degrees < 0.0)
            {
                degrees += 360.0;
            }
            return static_cast<float>(degrees);
        }

        const char* acceleratorBackendName(SiftComputeBackend backend)
        {
            switch (backend)
            {
            case SiftComputeBackend::Cuda:
                return "cuda";
            case SiftComputeBackend::OpenCl:
                return "opencl";
            case SiftComputeBackend::Cpu:
                return "cpu";
            case SiftComputeBackend::Automatic:
                return "auto";
            case SiftComputeBackend::Metal:
                break;
            }
            throw std::invalid_argument("PlaMatch-HCT does not provide a Metal backend");
        }

        std::unique_ptr<metalign::DescriptorAccelerator> createAccelerator(SiftComputeBackend backend, int deviceIndex)
        {
            if (backend == SiftComputeBackend::Cpu)
            {
                return {};
            }
            auto accelerator = metalign::create_descriptor_accelerator(
                acceleratorBackendName(backend), deviceIndex, std::numeric_limits<std::uint64_t>::max(), false);
            if (!accelerator)
            {
                throw std::runtime_error("PlaMatch-HCT accelerator creation returned no device");
            }
            if (!accelerator->supports_feature_extraction())
            {
                throw std::runtime_error("PlaMatch-HCT accelerator does not support feature extraction");
            }
            return accelerator;
        }

        FeatureSet makePlaScanFeatures(metalign::FeatureSet features,
                                       const ImageFeatureInput& input,
                                       const std::string& computeBackend)
        {
            const double coordinateScale = input.coordinateScale > 0.0 ? input.coordinateScale : 1.0;
            transformFeatureRows(features.keypoints, coordinateScale, input.coordinateOffsetX, input.coordinateOffsetY);
            transformFeatureRows(
                features.coarse_keypoints, coordinateScale, input.coordinateOffsetX, input.coordinateOffsetY);
            features.image_width = static_cast<std::size_t>(
                input.originalWidth > 0 ? input.originalWidth : static_cast<int>(features.image_width));
            features.image_height = static_cast<std::size_t>(
                input.originalHeight > 0 ? input.originalHeight : static_cast<int>(features.image_height));

            FeatureSet result;
            result.keypoints.reserve(features.keypoints.size());
            result.scores.reserve(features.keypoints.size());
            result.descriptors.create(
                static_cast<int>(features.keypoints.size()), static_cast<int>(metalign::kDescriptorSize), CV_8U);
            for (int row = 0; row < static_cast<int>(features.keypoints.size()); ++row)
            {
                const metalign::Keypoint& source = features.keypoints[static_cast<std::size_t>(row)];
                cv::KeyPoint keypoint;
                keypoint.pt = cv::Point2f(static_cast<float>(source.x), static_cast<float>(source.y));
                keypoint.size = static_cast<float>(source.scale);
                keypoint.angle = orientationDegrees(source.orientation);
                keypoint.response = static_cast<float>(source.response);
                keypoint.octave = source.octave;
                result.keypoints.push_back(keypoint);
                result.scores.push_back(keypoint.response);
                std::memcpy(result.descriptors.ptr(row), source.descriptor.data(), metalign::kDescriptorSize);
            }
            result.descriptorsL2Normalized = false;
            result.sourceAlgorithm = kPlaMatchHctAlgorithmId;
            result.computeBackend = computeBackend;
            result.imageWidth = static_cast<int>(features.image_width);
            result.imageHeight = static_cast<int>(features.image_height);
            result.payload = std::make_shared<PlaMatchHctFeaturePayload>(std::move(features));
            return result;
        }

        const PlaMatchHctFeaturePayload& payloadFor(const FeatureSet& features)
        {
            if (!features.isConsistent() || features.sourceAlgorithm != kPlaMatchHctAlgorithmId ||
                features.descriptors.type() != CV_8U ||
                features.descriptors.cols != static_cast<int>(metalign::kDescriptorSize))
            {
                throw std::invalid_argument("PlaMatch-HCT requires consistent 64-byte MLDB features");
            }
            const auto payload = std::dynamic_pointer_cast<const PlaMatchHctFeaturePayload>(features.payload);
            if (!payload)
            {
                throw std::invalid_argument("PlaMatch-HCT feature payload is missing");
            }
            return *payload;
        }

        struct AcceptedMatch
        {
            int query = -1;
            int train = -1;
            float distance = 0.0F;
        };

        std::vector<AcceptedMatch> uniqueMatches(const metalign::FeatureSet& first,
                                                 const metalign::FeatureSet& second,
                                                 const std::vector<metalign::FeatureMatch>& matches,
                                                 const std::vector<std::size_t>& acceptedRows)
        {
            std::vector<AcceptedMatch> candidates;
            candidates.reserve(acceptedRows.size());
            for (const std::size_t row : acceptedRows)
            {
                if (row >= matches.size())
                {
                    continue;
                }
                const metalign::FeatureMatch& match = matches[row];
                if (match.first >= first.keypoints.size() || match.second >= second.keypoints.size())
                {
                    continue;
                }
                const float distance = static_cast<float>(metalign::descriptor_hamming_distance(
                    first.keypoints[match.first].descriptor, second.keypoints[match.second].descriptor));
                candidates.push_back({static_cast<int>(match.first), static_cast<int>(match.second), distance});
            }
            std::stable_sort(candidates.begin(),
                             candidates.end(),
                             [](const AcceptedMatch& left, const AcceptedMatch& right)
                             {
                                 if (left.distance != right.distance)
                                 {
                                     return left.distance < right.distance;
                                 }
                                 if (left.query != right.query)
                                 {
                                     return left.query < right.query;
                                 }
                                 return left.train < right.train;
                             });

            std::vector<bool> usedFirst(first.keypoints.size(), false);
            std::vector<bool> usedSecond(second.keypoints.size(), false);
            std::vector<AcceptedMatch> result;
            result.reserve(candidates.size());
            for (const AcceptedMatch& candidate : candidates)
            {
                if (usedFirst[static_cast<std::size_t>(candidate.query)] ||
                    usedSecond[static_cast<std::size_t>(candidate.train)])
                {
                    continue;
                }
                usedFirst[static_cast<std::size_t>(candidate.query)] = true;
                usedSecond[static_cast<std::size_t>(candidate.train)] = true;
                result.push_back(candidate);
            }
            std::sort(result.begin(),
                      result.end(),
                      [](const AcceptedMatch& left, const AcceptedMatch& right)
                      {
                          if (left.query != right.query)
                          {
                              return left.query < right.query;
                          }
                          return left.train < right.train;
                      });
            return result;
        }

        MatchResult makeMatchResult(const FeatureSet& features0,
                                    const FeatureSet& features1,
                                    const metalign::FeatureSet& vendorFeatures0,
                                    const metalign::FeatureSet& vendorFeatures1,
                                    const std::vector<metalign::FeatureMatch>& raw)
        {
            const std::vector<std::size_t> locallyConsistent =
                metalign::local_consistency_inliers(vendorFeatures0, vendorFeatures1, raw);
            const std::vector<AcceptedMatch> accepted =
                uniqueMatches(vendorFeatures0, vendorFeatures1, raw, locallyConsistent);

            MatchResult result;
            result.sourceAlgorithm = kPlaMatchHctAlgorithmId;
            result.matches0.assign(features0.keypoints.size(), -1);
            result.matches1.assign(features1.keypoints.size(), -1);
            result.matchingScores0.assign(features0.keypoints.size(), 0.0F);
            result.matchingScores1.assign(features1.keypoints.size(), 0.0F);
            result.cvMatches.reserve(accepted.size());
            constexpr float descriptorBits = static_cast<float>(metalign::kDescriptorSize * 8U);
            for (const AcceptedMatch& match : accepted)
            {
                const float confidence = std::clamp(1.0F - match.distance / descriptorBits, 0.0F, 1.0F);
                result.matches0[static_cast<std::size_t>(match.query)] = match.train;
                result.matches1[static_cast<std::size_t>(match.train)] = match.query;
                result.matchingScores0[static_cast<std::size_t>(match.query)] = confidence;
                result.matchingScores1[static_cast<std::size_t>(match.train)] = confidence;
                result.cvMatches.emplace_back(match.query, match.train, match.distance);
            }
            result.numMatches = static_cast<int>(result.cvMatches.size());
            return result;
        }

    } // namespace

    PlaMatchHctBackendResolution resolvePlaMatchHctBackend(SiftComputeBackend requestedBackend, int deviceIndex)
    {
        if (requestedBackend == SiftComputeBackend::Automatic)
        {
            PlaMatchHctBackendResolution cuda = resolvePlaMatchHctBackend(SiftComputeBackend::Cuda, deviceIndex);
            if (cuda.valid)
            {
                return cuda;
            }
            PlaMatchHctBackendResolution openCl = resolvePlaMatchHctBackend(SiftComputeBackend::OpenCl, deviceIndex);
            if (openCl.valid)
            {
                return openCl;
            }
            return resolvePlaMatchHctBackend(SiftComputeBackend::Cpu, deviceIndex);
        }

        PlaMatchHctBackendResolution result;
        result.backend = requestedBackend;
        result.deviceIndex = deviceIndex;
        if (requestedBackend == SiftComputeBackend::Metal)
        {
            result.errorMessage = QStringLiteral("PlaMatch-HCT 不提供 Metal 后端");
            return result;
        }
        if (requestedBackend == SiftComputeBackend::Cpu)
        {
            result.valid = true;
            result.deviceName = QStringLiteral("CPU");
            result.displayName = QStringLiteral("CPU HCTree");
            return result;
        }

        try
        {
            auto accelerator = createAccelerator(requestedBackend, deviceIndex);
            result.deviceName = QString::fromStdString(accelerator->device_name());
            result.displayName = QStringLiteral("%1 / %2").arg(
                requestedBackend == SiftComputeBackend::Cuda ? QStringLiteral("CUDA") : QStringLiteral("OpenCL"),
                result.deviceName);
            result.valid = true;
        }
        catch (const std::exception& error)
        {
            result.errorMessage = QStringLiteral("PlaMatch-HCT %1 设备 %2 不可用：%3")
                                      .arg(requestedBackend == SiftComputeBackend::Cuda ? QStringLiteral("CUDA")
                                                                                        : QStringLiteral("OpenCL"))
                                      .arg(deviceIndex)
                                      .arg(QString::fromUtf8(error.what()));
        }
        return result;
    }

    PlaMatchHctAlgorithm::PlaMatchHctAlgorithm(ImageMatchingRuntimeConfig config) : _config(std::move(config))
    {
        _resolvedBackend = _config.siftBackend;
        if (_resolvedBackend == SiftComputeBackend::Automatic)
        {
            const PlaMatchHctBackendResolution resolution =
                resolvePlaMatchHctBackend(_resolvedBackend, _config.cudaDevice);
            if (!resolution.valid)
            {
                throw std::runtime_error(resolution.errorMessage.toStdString());
            }
            _resolvedBackend = resolution.backend;
        }
        _accelerator = createAccelerator(_resolvedBackend, _config.cudaDevice);
        _computeBackend =
            _accelerator ? QStringLiteral("plamatch_hct_%1").arg(QString::fromStdString(_accelerator->backend_name()))
                         : QStringLiteral("plamatch_hct_cpu");
    }

    PlaMatchHctAlgorithm::~PlaMatchHctAlgorithm() = default;

    ImageMatchingAlgorithmDescriptor PlaMatchHctAlgorithm::descriptor() const
    {
        return plaMatchHctDescriptor();
    }

    FeatureSet PlaMatchHctAlgorithm::extract(const ImageFeatureInput& input) const
    {
        const metalign::Image image = makePlaMatchHctImage(input.grayImage, input.colorImage);
        const metalign::Image mask = makePlaMatchHctMask(input.validMask);

        metalign::MatchPhotosOptions options;
        options.downscale = _config.alignmentDownscale;
        options.keypoint_limit = _config.maxKeypoints > 0 ? static_cast<std::size_t>(_config.maxKeypoints) : 0;
        options.keypoint_limit_per_mpx = 0;
        options.filter_mask = !mask.empty();
        options.mask_tiepoints = !mask.empty();
        metalign::FeatureExtractor extractor(std::move(options), _accelerator.get());
        metalign::FeatureSet features =
            extractor.extract(image, fileSystemPath(input.imagePath), mask.empty() ? nullptr : &mask);
        return makePlaScanFeatures(std::move(features), input, _computeBackend.toStdString());
    }

    MatchResult PlaMatchHctAlgorithm::matchFeatures(const FeatureSet& features0, const FeatureSet& features1)
    {
        const PlaMatchHctFeaturePayload& payload0 = payloadFor(features0);
        const PlaMatchHctFeaturePayload& payload1 = payloadFor(features1);
        metalign::MatchPhotosOptions options;
        std::vector<metalign::FeatureMatch> raw =
            metalign::match_feature_sets(payload0.fullFeatures(),
                                         payload1.fullFeatures(),
                                         options,
                                         _accelerator.get(),
                                         _accelerator ? nullptr : &payload0.fullIndex(),
                                         _accelerator ? nullptr : &payload1.fullIndex());
        return makeMatchResult(features0, features1, payload0.fullFeatures(), payload1.fullFeatures(), raw);
    }

    std::vector<MatchResult> PlaMatchHctAlgorithm::matchFeatureBatch(std::span<const FeaturePairInput> pairs,
                                                                     const std::function<bool()>& shouldCancel,
                                                                     const BatchMatchProgressCallback& progressCallback)
    {
        if (!_accelerator)
        {
            throw std::logic_error("PlaMatch-HCT batch feature matching requires CUDA or OpenCL");
        }

        std::unordered_map<const FeatureSet*, std::size_t> indices;
        std::vector<const metalign::FeatureSet*> vendorFeatures;
        std::vector<metalign::ImagePair> vendorPairs;
        vendorPairs.reserve(pairs.size());
        const auto featureIndex = [&](const FeatureSet* feature)
        {
            if (!feature)
            {
                throw std::invalid_argument("PlaMatch-HCT batch contains an empty feature set");
            }
            const auto found = indices.find(feature);
            if (found != indices.cend())
            {
                return found->second;
            }
            const PlaMatchHctFeaturePayload& payload = payloadFor(*feature);
            const std::size_t index = vendorFeatures.size();
            indices.emplace(feature, index);
            vendorFeatures.push_back(&payload.fullFeatures());
            return index;
        };
        for (const FeaturePairInput& pair : pairs)
        {
            vendorPairs.push_back({featureIndex(pair.features0), featureIndex(pair.features1)});
        }

        metalign::MatchPhotosOptions options;
        const std::vector<metalign::PairMatches> matched = metalign::match_feature_pairs_accelerated_batches(
            std::span<const metalign::FeatureSet* const>(vendorFeatures),
            vendorPairs,
            options,
            *_accelerator,
            shouldCancel,
            progressCallback);
        if (matched.size() != pairs.size())
        {
            throw std::runtime_error("PlaMatch-HCT accelerated batch result count mismatch");
        }

        std::vector<MatchResult> result;
        result.reserve(pairs.size());
        for (std::size_t index = 0; index < pairs.size(); ++index)
        {
            const FeaturePairInput& pair = pairs[index];
            const PlaMatchHctFeaturePayload& payload0 = payloadFor(*pair.features0);
            const PlaMatchHctFeaturePayload& payload1 = payloadFor(*pair.features1);
            result.push_back(makeMatchResult(*pair.features0,
                                             *pair.features1,
                                             payload0.fullFeatures(),
                                             payload1.fullFeatures(),
                                             matched[index].matches));
        }
        return result;
    }

    void registerPlaMatchHctAlgorithm()
    {
        const ImageMatchingAlgorithmDescriptor descriptor = plaMatchHctDescriptor();
        QString ignoredError;
        ImageMatchingRegistry::registerAlgorithm(
            descriptor,
            [](const ImageMatchingRuntimeConfig& config) { return std::make_unique<PlaMatchHctAlgorithm>(config); },
            &ignoredError);
    }

} // namespace xjw::image_matching
