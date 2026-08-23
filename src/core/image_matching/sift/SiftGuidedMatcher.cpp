#include "SiftGuidedMatcher.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace xjw::image_matching
{
    namespace
    {

        double
        sampsonDistance(const std::array<double, 9>& fundamental, const cv::Point2f& point0, const cv::Point2f& point1)
        {
            const double fx0x = fundamental[0] * point0.x + fundamental[1] * point0.y + fundamental[2];
            const double fx0y = fundamental[3] * point0.x + fundamental[4] * point0.y + fundamental[5];
            const double fx0z = fundamental[6] * point0.x + fundamental[7] * point0.y + fundamental[8];
            const double ftx1x = fundamental[0] * point1.x + fundamental[3] * point1.y + fundamental[6];
            const double ftx1y = fundamental[1] * point1.x + fundamental[4] * point1.y + fundamental[7];
            const double numerator = point1.x * fx0x + point1.y * fx0y + fx0z;
            const double denominator = fx0x * fx0x + fx0y * fx0y + ftx1x * ftx1x + ftx1y * ftx1y;
            if (!std::isfinite(denominator) || denominator <= 1e-15)
            {
                return std::numeric_limits<double>::infinity();
            }
            return std::sqrt((numerator * numerator) / denominator);
        }

        std::vector<int> unmatchedIndices(int count, const std::vector<int>& used)
        {
            std::unordered_set<int> usedSet;
            usedSet.reserve(used.size());
            for (const int index : used)
            {
                if (index >= 0 && index < count)
                {
                    usedSet.insert(index);
                }
            }
            std::vector<int> result;
            result.reserve(static_cast<std::size_t>(std::max(0, count - static_cast<int>(usedSet.size()))));
            for (int index = 0; index < count; ++index)
            {
                if (!usedSet.contains(index))
                {
                    result.push_back(index);
                }
            }
            return result;
        }

        bool cancellationRequested(const SiftGuidedMatchOptions& options)
        {
            return options.shouldCancel && options.shouldCancel();
        }

        struct EpipolarLine
        {
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
        };

        EpipolarLine targetEpipolarLine(const std::array<double, 9>& fundamental,
                                        const cv::Point2f& queryPoint,
                                        bool reverse)
        {
            if (reverse)
            {
                return {fundamental[0] * queryPoint.x + fundamental[3] * queryPoint.y + fundamental[6],
                        fundamental[1] * queryPoint.x + fundamental[4] * queryPoint.y + fundamental[7],
                        fundamental[2] * queryPoint.x + fundamental[5] * queryPoint.y + fundamental[8]};
            }
            return {fundamental[0] * queryPoint.x + fundamental[1] * queryPoint.y + fundamental[2],
                    fundamental[3] * queryPoint.x + fundamental[4] * queryPoint.y + fundamental[5],
                    fundamental[6] * queryPoint.x + fundamental[7] * queryPoint.y + fundamental[8]};
        }

        double oppositeLineNormSquared(const std::array<double, 9>& fundamental,
                                       const cv::Point2f& targetPoint,
                                       bool reverse)
        {
            double x = 0.0;
            double y = 0.0;
            if (reverse)
            {
                x = fundamental[0] * targetPoint.x + fundamental[1] * targetPoint.y + fundamental[2];
                y = fundamental[3] * targetPoint.x + fundamental[4] * targetPoint.y + fundamental[5];
            }
            else
            {
                x = fundamental[0] * targetPoint.x + fundamental[3] * targetPoint.y + fundamental[6];
                y = fundamental[1] * targetPoint.x + fundamental[4] * targetPoint.y + fundamental[7];
            }
            return x * x + y * y;
        }

        class SpatialFeatureGrid
        {
        public:
            SpatialFeatureGrid(const FeatureSet& features, const std::vector<int>& indices, int cellSize)
                : _indices(indices),
                  _cellSize(std::max(8, cellSize)),
                  _width(std::max(1, features.imageWidth)),
                  _height(std::max(1, features.imageHeight)),
                  _columns(std::max(1, (_width + _cellSize - 1) / _cellSize)),
                  _rows(std::max(1, (_height + _cellSize - 1) / _cellSize)),
                  _cells(static_cast<std::size_t>(_columns * _rows))
            {
                for (const int index : _indices)
                {
                    const cv::Point2f point = features.keypoints[static_cast<std::size_t>(index)].pt;
                    if (point.x < 0.0f || point.y < 0.0f || point.x >= _width || point.y >= _height)
                    {
                        _outsideIndices.push_back(index);
                        continue;
                    }
                    const int column = std::min(_columns - 1, static_cast<int>(point.x) / _cellSize);
                    const int row = std::min(_rows - 1, static_cast<int>(point.y) / _cellSize);
                    _cells[static_cast<std::size_t>(row * _columns + column)].push_back(index);
                }
            }

            void findCandidates(const std::array<double, 9>& fundamental,
                                const cv::Point2f& queryPoint,
                                bool reverse,
                                double thresholdPixels,
                                std::vector<int>* result) const
            {
                if (!result)
                {
                    return;
                }
                result->clear();
                const EpipolarLine line = targetEpipolarLine(fundamental, queryPoint, reverse);
                const double line_norm_squared = line.a * line.a + line.b * line.b;
                if (!std::isfinite(line_norm_squared) || line_norm_squared <= 1e-15)
                {
                    result->assign(_indices.cbegin(), _indices.cend());
                    return;
                }

                const double maximum_x = static_cast<double>(std::max(0, _width - 1));
                const double maximum_y = static_cast<double>(std::max(0, _height - 1));
                const std::array<cv::Point2f, 4> corners{{{0.0f, 0.0f},
                                                          {static_cast<float>(maximum_x), 0.0f},
                                                          {0.0f, static_cast<float>(maximum_y)},
                                                          {static_cast<float>(maximum_x),
                                                           static_cast<float>(maximum_y)}}};
                double maximum_opposite_norm_squared = 0.0;
                for (const cv::Point2f& corner : corners)
                {
                    maximum_opposite_norm_squared =
                        std::max(maximum_opposite_norm_squared,
                                 oppositeLineNormSquared(fundamental, corner, reverse));
                }
                const double perpendicular_band = std::max(0.0, thresholdPixels) *
                    std::sqrt(1.0 + maximum_opposite_norm_squared / line_norm_squared);

                result->insert(result->end(), _outsideIndices.cbegin(), _outsideIndices.cend());
                if (std::abs(line.b) >= std::abs(line.a) && std::abs(line.b) > 1e-15)
                {
                    const double vertical_padding =
                        perpendicular_band * std::sqrt(line_norm_squared) / std::abs(line.b);
                    for (int column = 0; column < _columns; ++column)
                    {
                        const double x0 = static_cast<double>(column * _cellSize);
                        const double x1 = static_cast<double>(std::min(_width, (column + 1) * _cellSize));
                        const double y0 = -(line.a * x0 + line.c) / line.b;
                        const double y1 = -(line.a * x1 + line.c) / line.b;
                        appendColumnCells(column,
                                          std::min(y0, y1) - vertical_padding,
                                          std::max(y0, y1) + vertical_padding,
                                          result);
                    }
                }
                else if (std::abs(line.a) > 1e-15)
                {
                    const double horizontal_padding =
                        perpendicular_band * std::sqrt(line_norm_squared) / std::abs(line.a);
                    for (int row = 0; row < _rows; ++row)
                    {
                        const double y0 = static_cast<double>(row * _cellSize);
                        const double y1 = static_cast<double>(std::min(_height, (row + 1) * _cellSize));
                        const double x0 = -(line.b * y0 + line.c) / line.a;
                        const double x1 = -(line.b * y1 + line.c) / line.a;
                        appendRowCells(row,
                                       std::min(x0, x1) - horizontal_padding,
                                       std::max(x0, x1) + horizontal_padding,
                                       result);
                    }
                }
            }

        private:
            void appendColumnCells(int column, double minimumY, double maximumY, std::vector<int>* result) const
            {
                if (!result || maximumY < 0.0 || minimumY >= _height)
                {
                    return;
                }
                const int first_row = std::clamp(static_cast<int>(std::floor(minimumY / _cellSize)), 0, _rows - 1);
                const int last_row = std::clamp(static_cast<int>(std::floor(maximumY / _cellSize)), 0, _rows - 1);
                for (int row = first_row; row <= last_row; ++row)
                {
                    const auto& cell = _cells[static_cast<std::size_t>(row * _columns + column)];
                    result->insert(result->end(), cell.cbegin(), cell.cend());
                }
            }

            void appendRowCells(int row, double minimumX, double maximumX, std::vector<int>* result) const
            {
                if (!result || maximumX < 0.0 || minimumX >= _width)
                {
                    return;
                }
                const int first_column =
                    std::clamp(static_cast<int>(std::floor(minimumX / _cellSize)), 0, _columns - 1);
                const int last_column =
                    std::clamp(static_cast<int>(std::floor(maximumX / _cellSize)), 0, _columns - 1);
                for (int column = first_column; column <= last_column; ++column)
                {
                    const auto& cell = _cells[static_cast<std::size_t>(row * _columns + column)];
                    result->insert(result->end(), cell.cbegin(), cell.cend());
                }
            }

            const std::vector<int>& _indices;
            int _cellSize = 64;
            int _width = 1;
            int _height = 1;
            int _columns = 1;
            int _rows = 1;
            std::vector<std::vector<int>> _cells;
            std::vector<int> _outsideIndices;
        };

        float squaredDescriptorDistance(const cv::Mat& queryDescriptors,
                                        int queryIndex,
                                        const cv::Mat& trainDescriptors,
                                        int trainIndex)
        {
            const float* query = queryDescriptors.ptr<float>(queryIndex);
            const float* train = trainDescriptors.ptr<float>(trainIndex);
            float distance = 0.0f;
            for (int column = 0; column < queryDescriptors.cols; ++column)
            {
                const float difference = query[column] - train[column];
                distance += difference * difference;
            }
            return distance;
        }

        struct GuidedCandidate
        {
            int peerIndex = -1;
            float distance = std::numeric_limits<float>::infinity();
            float ratio = 1.0f;
        };

        struct GuidedCandidateSearchResult
        {
            std::vector<GuidedCandidate> candidates;
            SiftGuidedMatchDiagnostics diagnostics;
            bool canceled = false;
        };

        GuidedCandidateSearchResult guidedCandidates(const FeatureSet& queryFeatures,
                                                     const FeatureSet& trainFeatures,
                                                     const std::vector<int>& queryIndices,
                                                     const std::vector<int>& trainIndices,
                                                     const std::array<double, 9>& fundamental,
                                                     bool reverse,
                                                     const SiftGuidedMatchOptions& options)
        {
            GuidedCandidateSearchResult search_result;
            search_result.candidates.resize(queryIndices.size());
            if (queryIndices.empty() || trainIndices.empty())
            {
                return search_result;
            }
            if (cancellationRequested(options))
            {
                search_result.canceled = true;
                return search_result;
            }

            const SpatialFeatureGrid train_grid(trainFeatures, trainIndices, options.spatialCellSizePixels);
            const std::uint64_t check_interval =
                static_cast<std::uint64_t>(std::max(1, options.cancellationCheckInterval));
            std::vector<int> train_candidates;
            train_candidates.reserve(std::min<std::size_t>(trainIndices.size(), 1024U));
            for (int query_row = 0; query_row < static_cast<int>(queryIndices.size()); ++query_row)
            {
                if (cancellationRequested(options))
                {
                    search_result.canceled = true;
                    return search_result;
                }
                const int query_index = queryIndices[static_cast<std::size_t>(query_row)];
                const cv::Point2f query_point =
                    queryFeatures.keypoints[static_cast<std::size_t>(query_index)].pt;
                train_grid.findCandidates(
                    fundamental, query_point, reverse, options.epipolarThresholdPixels, &train_candidates);

                float best_squared_distance = std::numeric_limits<float>::infinity();
                float second_squared_distance = std::numeric_limits<float>::infinity();
                GuidedCandidate best;
                for (const int train_index : train_candidates)
                {
                    ++search_result.diagnostics.spatialCandidates;
                    if (search_result.diagnostics.spatialCandidates % check_interval == 0 &&
                        cancellationRequested(options))
                    {
                        search_result.canceled = true;
                        return search_result;
                    }
                    const cv::Point2f point0 = reverse
                                                   ? trainFeatures.keypoints[static_cast<std::size_t>(train_index)].pt
                                                   : query_point;
                    const cv::Point2f point1 = reverse
                                                   ? query_point
                                                   : trainFeatures.keypoints[static_cast<std::size_t>(train_index)].pt;
                    if (sampsonDistance(fundamental, point0, point1) > options.epipolarThresholdPixels)
                    {
                        continue;
                    }
                    ++search_result.diagnostics.descriptorComparisons;
                    const float squared_distance = squaredDescriptorDistance(
                        queryFeatures.descriptors, query_index, trainFeatures.descriptors, train_index);
                    if (squared_distance < best_squared_distance ||
                        (squared_distance == best_squared_distance && train_index < best.peerIndex))
                    {
                        second_squared_distance = best_squared_distance;
                        best_squared_distance = squared_distance;
                        best.peerIndex = train_index;
                    }
                    else if (squared_distance < second_squared_distance)
                    {
                        second_squared_distance = squared_distance;
                    }
                }
                if (best.peerIndex >= 0)
                {
                    best.distance = std::sqrt(best_squared_distance);
                    best.ratio = std::isfinite(second_squared_distance) && second_squared_distance > 1e-12f
                                     ? std::sqrt(best_squared_distance / second_squared_distance)
                                     : 0.0f;
                    search_result.candidates[static_cast<std::size_t>(query_row)] = best;
                }
            }
            return search_result;
        }

    } // namespace

    SiftGuidedMatchResult findGuidedSiftMatchesDetailed(const FeatureSet& features0,
                                                        const FeatureSet& features1,
                                                        const std::array<double, 9>& fundamental,
                                                        const std::vector<int>& existingFeatureIds0,
                                                        const std::vector<int>& existingFeatureIds1,
                                                        const SiftGuidedMatchOptions& options)
    {
        SiftGuidedMatchResult result;
        if (!features0.isConsistent() || !features1.isConsistent() || features0.descriptors.type() != CV_32F ||
            features1.descriptors.type() != CV_32F ||
            features0.descriptors.cols != features1.descriptors.cols)
        {
            return result;
        }
        if (cancellationRequested(options))
        {
            result.canceled = true;
            return result;
        }
        const std::vector<int> unmatched0 = unmatchedIndices(features0.size(), existingFeatureIds0);
        const std::vector<int> unmatched1 = unmatchedIndices(features1.size(), existingFeatureIds1);
        if (unmatched0.empty() || unmatched1.empty())
        {
            return result;
        }

        const GuidedCandidateSearchResult forward =
            guidedCandidates(features0, features1, unmatched0, unmatched1, fundamental, false, options);
        result.diagnostics.spatialCandidates += forward.diagnostics.spatialCandidates;
        result.diagnostics.descriptorComparisons += forward.diagnostics.descriptorComparisons;
        if (forward.canceled || cancellationRequested(options))
        {
            result.canceled = true;
            return result;
        }
        const GuidedCandidateSearchResult reverse =
            guidedCandidates(features1, features0, unmatched1, unmatched0, fundamental, true, options);
        result.diagnostics.spatialCandidates += reverse.diagnostics.spatialCandidates;
        result.diagnostics.descriptorComparisons += reverse.diagnostics.descriptorComparisons;
        if (reverse.canceled || cancellationRequested(options))
        {
            result.canceled = true;
            return result;
        }
        std::vector<int> reverseRowByFeature(static_cast<std::size_t>(features1.size()), -1);
        for (int row = 0; row < static_cast<int>(unmatched1.size()); ++row)
        {
            reverseRowByFeature[static_cast<std::size_t>(unmatched1[static_cast<std::size_t>(row)])] = row;
        }

        for (int row = 0; row < static_cast<int>(forward.candidates.size()); ++row)
        {
            const GuidedCandidate& candidate = forward.candidates[static_cast<std::size_t>(row)];
            if (candidate.peerIndex < 0 || candidate.ratio > options.maximumDescriptorRatio)
            {
                continue;
            }
            const int reverseRow = reverseRowByFeature[static_cast<std::size_t>(candidate.peerIndex)];
            if (reverseRow < 0 || reverseRow >= static_cast<int>(reverse.candidates.size()))
            {
                continue;
            }
            const GuidedCandidate& reverseCandidate = reverse.candidates[static_cast<std::size_t>(reverseRow)];
            const int index0 = unmatched0[static_cast<std::size_t>(row)];
            if (reverseCandidate.peerIndex != index0 || reverseCandidate.ratio > options.maximumDescriptorRatio)
            {
                continue;
            }
            const float distance = std::max(candidate.distance, reverseCandidate.distance);
            const double residual = sampsonDistance(
                fundamental,
                features0.keypoints[static_cast<std::size_t>(index0)].pt,
                features1.keypoints[static_cast<std::size_t>(candidate.peerIndex)].pt);
            if (!std::isfinite(residual) || residual > options.epipolarThresholdPixels)
            {
                continue;
            }
            result.matches.push_back({index0,
                                      candidate.peerIndex,
                                      std::clamp(1.0f - 0.5f * distance * distance, 0.0f, 1.0f),
                                      candidate.ratio,
                                      reverseCandidate.ratio,
                                      static_cast<float>(residual)});
        }
        std::stable_sort(result.matches.begin(),
                         result.matches.end(),
                         [](const SiftGuidedMatch& left, const SiftGuidedMatch& right)
                         { return left.confidence > right.confidence; });
        if (options.maximumAdditionalMatches > 0 &&
            static_cast<int>(result.matches.size()) > options.maximumAdditionalMatches)
        {
            result.matches.resize(static_cast<std::size_t>(options.maximumAdditionalMatches));
        }
        return result;
    }

    std::vector<SiftGuidedMatch> findGuidedSiftMatches(const FeatureSet& features0,
                                                       const FeatureSet& features1,
                                                       const std::array<double, 9>& fundamental,
                                                       const std::vector<int>& existingFeatureIds0,
                                                       const std::vector<int>& existingFeatureIds1,
                                                       const SiftGuidedMatchOptions& options)
    {
        SiftGuidedMatchResult result = findGuidedSiftMatchesDetailed(
            features0, features1, fundamental, existingFeatureIds0, existingFeatureIds1, options);
        return result.canceled ? std::vector<SiftGuidedMatch>() : std::move(result.matches);
    }

} // namespace xjw::image_matching
