#include "placamera/rpc_adjustment.h"

#include "internal/rpc_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace placamera
{
    namespace
    {

        struct AdjustmentSample
        {
            std::array<double, 3> design{};
            double sampleResidual = 0.0;
            double lineResidual = 0.0;
            double baseWeight = 1.0;
            double effectiveWeight = 1.0;
        };

        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        bool solveSystem(std::array<std::array<double, 3>, 3> matrix,
                         std::array<double, 3> values,
                         int dimension,
                         std::array<double, 3>* solution) noexcept
        {
            if (!solution)
            {
                return false;
            }
            for (int column = 0; column < dimension; ++column)
            {
                int pivot = column;
                for (int row = column + 1; row < dimension; ++row)
                {
                    if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column]))
                    {
                        pivot = row;
                    }
                }
                if (!finite(matrix[pivot][column]) || std::abs(matrix[pivot][column]) < 1.0e-12)
                {
                    return false;
                }
                std::swap(matrix[column], matrix[pivot]);
                std::swap(values[column], values[pivot]);
                for (int row = column + 1; row < dimension; ++row)
                {
                    const double factor = matrix[row][column] / matrix[column][column];
                    for (int index = column; index < dimension; ++index)
                    {
                        matrix[row][index] -= factor * matrix[column][index];
                    }
                    values[row] -= factor * values[column];
                }
            }

            solution->fill(0.0);
            for (int row = dimension - 1; row >= 0; --row)
            {
                double value = values[row];
                for (int column = row + 1; column < dimension; ++column)
                {
                    value -= matrix[row][column] * (*solution)[column];
                }
                (*solution)[row] = value / matrix[row][row];
                if (!finite((*solution)[row]))
                {
                    return false;
                }
            }
            return true;
        }

        bool solveCorrection(const std::vector<AdjustmentSample>& samples,
                             int dimension,
                             std::array<double, 3>* sampleCoefficients,
                             std::array<double, 3>* lineCoefficients) noexcept
        {
            std::array<std::array<double, 3>, 3> normal{};
            std::array<double, 3> sample_rhs{};
            std::array<double, 3> line_rhs{};
            for (const AdjustmentSample& sample : samples)
            {
                for (int row = 0; row < dimension; ++row)
                {
                    sample_rhs[row] += sample.effectiveWeight * sample.design[row] * sample.sampleResidual;
                    line_rhs[row] += sample.effectiveWeight * sample.design[row] * sample.lineResidual;
                    for (int column = 0; column < dimension; ++column)
                    {
                        normal[row][column] += sample.effectiveWeight * sample.design[row] * sample.design[column];
                    }
                }
            }
            return solveSystem(normal, sample_rhs, dimension, sampleCoefficients) &&
                   solveSystem(normal, line_rhs, dimension, lineCoefficients);
        }

        double predicted(const std::array<double, 3>& coefficients,
                         const std::array<double, 3>& design,
                         int dimension) noexcept
        {
            double value = 0.0;
            for (int index = 0; index < dimension; ++index)
            {
                value += coefficients[index] * design[index];
            }
            return value;
        }

    } // namespace

    EvaluationResult<RpcBiasAdjustmentResult>
    estimateRpcImageCorrection(const RpcModel& camera,
                               const std::vector<RpcControlPointObservation>& observations,
                               const RpcBiasAdjustmentOptions& options)
    {
        const int dimension = options.model == RpcImageCorrectionModel::Translation ? 1 : 3;
        if (observations.size() < static_cast<std::size_t>(dimension))
        {
            return EvaluationResult<RpcBiasAdjustmentResult>::failure(
                CameraErrorCode::InvalidArgument, "RPC image correction has too few control-point observations");
        }
        if (options.maximumIterations <= 0 || !finite(options.huberThresholdPixels) ||
            options.huberThresholdPixels <= 0.0)
        {
            return EvaluationResult<RpcBiasAdjustmentResult>::failure(
                CameraErrorCode::InvalidArgument,
                "RPC bias-adjustment iteration count and Huber threshold must be positive");
        }

        const RpcParameters& parameters = camera.rpcDefinition().parameters();
        std::vector<AdjustmentSample> samples;
        samples.reserve(observations.size());
        double total_weight = 0.0;
        double before_sum = 0.0;
        for (const RpcControlPointObservation& observation : observations)
        {
            if (!finite(observation.weight) || observation.weight <= 0.0 || !finite(observation.observedImage.sample) ||
                !finite(observation.observedImage.line))
            {
                return EvaluationResult<RpcBiasAdjustmentResult>::failure(
                    CameraErrorCode::InvalidArgument,
                    "RPC control-point image coordinates and weights must be finite and positive");
            }
            const auto raw_image = internal::projectRpc(
                camera.rpcDefinition(), RpcCorrection::normalizedImage(), observation.ground, false);
            if (!raw_image)
            {
                return EvaluationResult<RpcBiasAdjustmentResult>::failure(raw_image.errorCode(), raw_image.message());
            }
            AdjustmentSample sample;
            sample.design = {{1.0,
                              (raw_image.value().sample - parameters.sampleOffset) / parameters.sampleScale,
                              (raw_image.value().line - parameters.lineOffset) / parameters.lineScale}};
            sample.sampleResidual = observation.observedImage.sample - raw_image.value().sample;
            sample.lineResidual = observation.observedImage.line - raw_image.value().line;
            sample.baseWeight = observation.weight;
            sample.effectiveWeight = observation.weight;
            samples.push_back(sample);
            before_sum += observation.weight *
                          (sample.sampleResidual * sample.sampleResidual + sample.lineResidual * sample.lineResidual);
            total_weight += observation.weight;
        }

        std::array<double, 3> sample_coefficients{};
        std::array<double, 3> line_coefficients{};
        int iterations = 0;
        const int iteration_limit = options.robust ? options.maximumIterations : 1;
        for (; iterations < iteration_limit; ++iterations)
        {
            if (!solveCorrection(samples, dimension, &sample_coefficients, &line_coefficients))
            {
                return EvaluationResult<RpcBiasAdjustmentResult>::failure(
                    CameraErrorCode::NonConvergence,
                    "RPC control-point distribution is singular for the requested correction model");
            }
            if (!options.robust)
            {
                ++iterations;
                break;
            }

            double maximum_weight_change = 0.0;
            for (AdjustmentSample& sample : samples)
            {
                const double sample_error =
                    sample.sampleResidual - predicted(sample_coefficients, sample.design, dimension);
                const double line_error = sample.lineResidual - predicted(line_coefficients, sample.design, dimension);
                const double residual = std::hypot(sample_error, line_error);
                const double robust_weight =
                    residual <= options.huberThresholdPixels ? 1.0 : options.huberThresholdPixels / residual;
                const double updated_weight = sample.baseWeight * robust_weight;
                maximum_weight_change =
                    std::max(maximum_weight_change, std::abs(updated_weight - sample.effectiveWeight));
                sample.effectiveWeight = updated_weight;
            }
            if (maximum_weight_change <= 1.0e-10)
            {
                ++iterations;
                break;
            }
        }

        RpcBiasAdjustmentResult adjusted;
        adjusted.correction.sampleOffsetPixels = sample_coefficients[0];
        adjusted.correction.lineOffsetPixels = line_coefficients[0];
        if (dimension == 3)
        {
            adjusted.correction.sampleSamplePixels = sample_coefficients[1];
            adjusted.correction.sampleLinePixels = sample_coefficients[2];
            adjusted.correction.lineSamplePixels = line_coefficients[1];
            adjusted.correction.lineLinePixels = line_coefficients[2];
        }
        double after_sum = 0.0;
        for (const AdjustmentSample& sample : samples)
        {
            const double sample_error =
                sample.sampleResidual - predicted(sample_coefficients, sample.design, dimension);
            const double line_error = sample.lineResidual - predicted(line_coefficients, sample.design, dimension);
            const double residual = std::hypot(sample_error, line_error);
            after_sum += sample.baseWeight * residual * residual;
            adjusted.maximumResidualPixels = std::max(adjusted.maximumResidualPixels, residual);
        }
        adjusted.rmsBeforePixels = std::sqrt(before_sum / (2.0 * total_weight));
        adjusted.rmsAfterPixels = std::sqrt(after_sum / (2.0 * total_weight));
        adjusted.observationCount = samples.size();
        adjusted.iterations = iterations;
        return EvaluationResult<RpcBiasAdjustmentResult>::success(adjusted, adjusted.rmsAfterPixels);
    }

} // namespace placamera
