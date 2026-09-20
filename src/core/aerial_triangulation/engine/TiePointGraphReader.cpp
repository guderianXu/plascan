#include "engine/TiePointGraphReader.h"
#include "file/JsonFile.h"
#include "file/FileIO.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace xjw::aerial_triangulation::engine
{
    namespace
    {
        using Json = nlohmann::json;
        const Json& field(const Json& object, const char* key)
        {
            static const Json empty;
            if (!object.is_object())
            {
                return empty;
            }
            const auto found = object.find(key);
            return found == object.end() ? empty : *found;
        }
        const Json& arrayField(const Json& object, const char* key)
        {
            static const Json empty = Json::array();
            const auto& value = field(object, key);
            return value.is_array() ? value : empty;
        }
        std::int64_t integer(const Json& value, std::int64_t fallback)
        {
            if (!value.is_number())
            {
                return fallback;
            }
            if (value.is_number_unsigned())
            {
                const auto result = value.get<std::uint64_t>();
                return result <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                           ? static_cast<std::int64_t>(result)
                           : fallback;
            }
            if (value.is_number_integer())
            {
                return value.get<std::int64_t>();
            }
            const auto number = value.get<double>();
            // Floating inputs must be integral and representable (2^63 is already outside int64).
            return std::isfinite(number) && number == std::trunc(number) && number >= -0x1p63 && number < 0x1p63
                       ? static_cast<std::int64_t>(number)
                       : fallback;
        }
        double number(const Json& value, double fallback = std::numeric_limits<double>::quiet_NaN())
        {
            return value.is_number() ? value.get<double>() : fallback;
        }
        struct ParsedObservation
        {
            ImageId imageId;
            FeatureIdx featureIndex;
        };
        std::uint64_t pairKey(ImageId first, ImageId second)
        {
            return (static_cast<std::uint64_t>(first) << 32U) | second;
        }
    } // namespace

    bool readTiePointGraph(const std::filesystem::path& path,
                           const std::vector<std::filesystem::path>& selectedImages,
                           TiePointGraph* graph,
                           std::string* error,
                           const ImageTokenResolver& resolver)
    {
        if (error)
        {
            error->clear();
        }
        const auto fail = [&](const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
            return false;
        };
        if (!graph)
        {
            return fail("连接点图输出参数为空");
        }
        *graph = {};
        Json root;
        if (!common::file::readJson(path, &root, error))
        {
            return fail("无法读取连接点 JSON: " + common::file::pathToUtf8(path) + (error ? ": " + *error : ""));
        }
        if (!root.is_object())
        {
            return fail("连接点文件不是有效 JSON 对象: " + common::file::pathToUtf8(path));
        }
        const auto formatVersion = integer(field(root, "format_version"), 0);
        if (field(root, "format") != "plascan_tie_points" || formatVersion < 1 || formatVersion > 3)
        {
            return fail("不支持的连接点文件格式或版本");
        }
        if (selectedImages.size() < 2 ||
            selectedImages.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return fail("SfM 影像数量必须在有效 ID 范围内且至少为两张");
        }
        std::map<std::int64_t, ImageId> selectedIdByPersistedId;
        std::set<int> coveredSelectedIds;
        for (const Json& image : arrayField(root, "images"))
        {
            const auto persistedId = integer(field(image, "image_id"), -1);
            const auto& token = field(image, "path");
            if (persistedId < 0 || !token.is_string())
            {
                continue;
            }
            const auto text = token.get<std::string>();
            int selectedId = -1;
            if (resolver)
            {
                selectedId = resolver(text);
            }
            else
            {
                const auto candidate = common::file::absoluteNormalizedPath(common::file::pathFromUtf8(text));
                for (std::size_t i = 0; i < selectedImages.size(); ++i)
                {
                    if (!candidate.empty() && candidate == common::file::absoluteNormalizedPath(selectedImages[i]))
                    {
                        selectedId = static_cast<int>(i);
                        break;
                    }
                }
            }
            if (selectedId >= 0 && static_cast<std::size_t>(selectedId) < selectedImages.size())
            {
                selectedIdByPersistedId[persistedId] = static_cast<ImageId>(selectedId);
                coveredSelectedIds.insert(selectedId);
            }
        }
        if (coveredSelectedIds.size() != selectedImages.size())
        {
            return fail("连接点文件的影像集合与当前空三影像集合不一致");
        }
        graph->imagePaths = selectedImages;
        // 连接点文件保留原始特征索引；SfM 只需要轨迹实际引用的稀疏子集。
        // 每张影像独立压缩索引可显著降低关键点内存，同时保持同一原始索引一致。
        std::map<ImageId, std::map<std::uint64_t, FeatureIdx>> compactIndexByOriginal;
        std::map<std::uint64_t, std::size_t> pairPosition;
        std::map<std::uint64_t, std::set<std::pair<FeatureIdx, FeatureIdx>>> pairObservations;
        std::size_t persistedObservationCount = 0;
        std::size_t acceptedObservationCount = 0;
        std::size_t persistedDirectEdgeCount = 0;
        std::size_t acceptedDirectEdgeCount = 0;
        graph->usesRawDirectEdges = formatVersion >= 2;

        const auto addMatch = [&](ParsedObservation observationA,
                                  ParsedObservation observationB,
                                  float confidence,
                                  bool synthesizedClosure)
        {
            if (observationA.imageId == observationB.imageId)
            {
                return false;
            }
            if (observationA.imageId > observationB.imageId)
            {
                std::swap(observationA, observationB);
            }

            const std::uint64_t key = pairKey(observationA.imageId, observationB.imageId);
            const std::pair<FeatureIdx, FeatureIdx> featurePair{observationA.featureIndex, observationB.featureIndex};
            if (!pairObservations[key].insert(featurePair).second)
            {
                return false;
            }

            auto position = pairPosition.find(key);
            if (position == pairPosition.end())
            {
                TiePointPair pair;
                pair.imageA = observationA.imageId;
                pair.imageB = observationB.imageId;
                graph->matchPairs.push_back(std::move(pair));
                position = pairPosition.emplace(key, graph->matchPairs.size() - 1).first;
            }
            graph->matchPairs[position->second].matches.push_back(
                {observationA.featureIndex, observationB.featureIndex, confidence});
            if (synthesizedClosure)
            {
                ++graph->synthesizedClosureEdgeCount;
            }
            else
            {
                ++graph->directEdgeCount;
            }
            return true;
        };

        for (const Json& trackValue : arrayField(root, "tracks"))
        {
            const Json& trackObject = trackValue;
            const float confidence =
                static_cast<float>(std::clamp(number(field(trackObject, "confidence"), 1.0), 0.0, 1.0));
            const Json& persistedObservationArray = arrayField(trackObject, "observations");
            persistedObservationCount += persistedObservationArray.size();
            std::vector<std::optional<ParsedObservation>> persistedObservations(
                static_cast<std::size_t>(persistedObservationArray.size()));
            std::vector<ParsedObservation> observations;

            for (std::size_t persistedIndex = 0; persistedIndex < persistedObservationArray.size(); ++persistedIndex)
            {
                const Json& observationValue = persistedObservationArray.at(persistedIndex);
                const Json& observationObject = observationValue;
                const Json& compactObservation = observationValue;
                const bool compact = observationValue.is_array();
                if (compact && compactObservation.size() < 4)
                {
                    continue;
                }
                const std::int64_t persistedImageId =
                    compact ? integer(compactObservation.at(0), -1) : integer(field(observationObject, "image_id"), -1);
                const std::int64_t originalFeatureIndex = compact
                                                              ? integer(compactObservation.at(1), -1)
                                                              : integer(field(observationObject, "feature_idx"), -1);
                const Json& xy = arrayField(observationObject, "xy");
                if (!selectedIdByPersistedId.contains(persistedImageId) || originalFeatureIndex < 0 ||
                    (!compact && xy.size() < 2))
                {
                    continue;
                }

                const double x = number(compact ? compactObservation.at(2) : xy.at(0));
                const double y = number(compact ? compactObservation.at(3) : xy.at(1));
                const double persistedScale =
                    compact && compactObservation.size() >= 5 ? number(compactObservation.at(4))
                    : observationObject.contains("scale")     ? number(field(observationObject, "scale"))
                                                              : std::numeric_limits<double>::quiet_NaN();
                if (!std::isfinite(x) || !std::isfinite(y))
                {
                    continue;
                }

                const ImageId imageId = selectedIdByPersistedId.at(persistedImageId);
                std::map<std::uint64_t, FeatureIdx>& indexMap = compactIndexByOriginal[imageId];
                const std::uint64_t originalKey = static_cast<std::uint64_t>(originalFeatureIndex);
                const auto existing = indexMap.find(originalKey);
                FeatureIdx compactIndex = existing == indexMap.end() ? kInvalidFeatureIdx : existing->second;
                if (compactIndex == kInvalidFeatureIdx)
                {
                    compactIndex = static_cast<FeatureIdx>(graph->keypointsByImage[imageId].size());
                    indexMap.emplace(originalKey, compactIndex);
                    const double scale = std::isfinite(persistedScale) && persistedScale > 0.0 ? persistedScale : 1.0;
                    graph->keypointsByImage[imageId].push_back(
                        {static_cast<float>(x), static_cast<float>(y), static_cast<float>(scale)});
                }
                const ParsedObservation parsed{imageId, compactIndex};
                persistedObservations[static_cast<std::size_t>(persistedIndex)] = parsed;
                observations.push_back(parsed);
                ++acceptedObservationCount;
            }

            // 一条多视轨迹在同一影像最多保留一个观测，避免生成自相矛盾 pair。
            std::sort(observations.begin(),
                      observations.end(),
                      [](const ParsedObservation& left, const ParsedObservation& right)
                      { return left.imageId < right.imageId; });
            observations.erase(std::unique(observations.begin(),
                                           observations.end(),
                                           [](const ParsedObservation& left, const ParsedObservation& right)
                                           { return left.imageId == right.imageId; }),
                               observations.end());
            if (observations.size() < 2)
            {
                continue;
            }

            Track track;
            track.confidence = confidence;
            track.elements.reserve(observations.size());
            for (const ParsedObservation& observation : observations)
            {
                track.elements.push_back({observation.imageId, observation.featureIndex});
            }
            graph->tracks.push_back(std::move(track));

            bool addedTrackEdge = false;
            if (formatVersion >= 2)
            {
                // v2 明确保留前端几何验证的原始边。轨迹连通性与直接匹配语义不能
                // 混为一谈，否则 A-B-C 的传递闭包会伪造并不存在的 A-C 匹配。
                for (const Json& edgeValue : arrayField(trackObject, "direct_edges"))
                {
                    ++persistedDirectEdgeCount;
                    const Json& edge = edgeValue;
                    if (!edge.is_array() || edge.size() < 2)
                    {
                        continue;
                    }
                    const std::int64_t first = integer(edge.at(0), -1);
                    const std::int64_t second = integer(edge.at(1), -1);
                    if (first < 0 || second < 0 || first == second ||
                        first >= static_cast<std::int64_t>(persistedObservations.size()) ||
                        second >= static_cast<std::int64_t>(persistedObservations.size()))
                    {
                        continue;
                    }
                    const auto& firstObservation = persistedObservations[static_cast<std::size_t>(first)];
                    const auto& secondObservation = persistedObservations[static_cast<std::size_t>(second)];
                    if (!firstObservation || !secondObservation)
                    {
                        continue;
                    }
                    if (addMatch(*firstObservation, *secondObservation, confidence, false))
                    {
                        addedTrackEdge = true;
                        ++acceptedDirectEdgeCount;
                    }
                }
            }
            else
            {
                // v1 没有保存边拓扑，只能保留历史兼容行为。诊断字段会明确记录这些
                // 是合成闭包边，新生成的 v2 文件不再走此路径。
                for (std::size_t first = 0; first + 1 < observations.size(); ++first)
                {
                    for (std::size_t second = first + 1; second < observations.size(); ++second)
                    {
                        addedTrackEdge |= addMatch(observations[first], observations[second], confidence, true);
                    }
                }
            }
            (void)addedTrackEdge;
        }

        graph->trackCount = static_cast<int>(graph->tracks.size());
        if (graph->tracks.empty() || graph->matchPairs.empty())
        {
            std::ostringstream message;
            message << "连接点文件中没有可用于 SfM 的多视图轨迹（轨迹 " << graph->tracks.size() << "，匹配对 "
                    << graph->matchPairs.size() << "，接受观测 " << acceptedObservationCount << "/"
                    << persistedObservationCount << "，接受直接边 " << acceptedDirectEdgeCount << "/"
                    << persistedDirectEdgeCount << "）";
            return fail(message.str());
        }
        return true;
    }
} // namespace xjw::aerial_triangulation::engine
