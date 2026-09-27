#include "placamera/reference/CameraReferenceUncertainty.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

namespace placamera::reference
{
    namespace
    {

        constexpr std::size_t kDimension = 6;

        [[noreturn]] void invalidCovariance(const char* message)
        {
            throw CameraValidationError(CameraErrorCode::InvalidArgument, message);
        }

        std::array<double, 36> diagonalMatrix(const std::vector<double>& values)
        {
            if (values.size() != kDimension)
            {
                invalidCovariance("pose covariance diagonal must contain 6 values");
            }
            std::array<double, 36> matrix{};
            for (std::size_t index = 0; index < kDimension; ++index)
            {
                matrix[index * kDimension + index] = values[index];
            }
            return matrix;
        }

        std::array<double, 36> symmetricMatrix(const std::vector<double>& values)
        {
            if (values.size() != 21)
            {
                invalidCovariance("pose covariance symmetric packing must contain 21 values");
            }
            std::array<double, 36> matrix{};
            std::size_t packed_index = 0;
            for (std::size_t row = 0; row < kDimension; ++row)
            {
                for (std::size_t column = row; column < kDimension; ++column)
                {
                    matrix[row * kDimension + column] = values[packed_index];
                    matrix[column * kDimension + row] = values[packed_index];
                    ++packed_index;
                }
            }
            return matrix;
        }

        std::vector<double> packSymmetric(const std::array<double, 36>& matrix)
        {
            std::vector<double> values;
            values.reserve(21);
            for (std::size_t row = 0; row < kDimension; ++row)
            {
                for (std::size_t column = row; column < kDimension; ++column)
                {
                    values.push_back(matrix[row * kDimension + column]);
                }
            }
            return values;
        }

        bool componentActive(PoseCovarianceComponents components, std::size_t index) noexcept
        {
            if (components == PoseCovarianceComponents::PositionAndRotation)
            {
                return true;
            }
            return components == PoseCovarianceComponents::PositionOnly ? index < 3 : index >= 3;
        }

        void validateCovariance(std::array<double, 36>* matrix,
                                PoseCovarianceComponents components,
                                CovarianceDefiniteness definiteness)
        {
            if (!matrix)
            {
                invalidCovariance("pose covariance matrix is missing");
            }
            double scale = 1.0;
            for (double value : *matrix)
            {
                if (!std::isfinite(value))
                {
                    invalidCovariance("pose covariance must contain finite values");
                }
                scale = std::max(scale, std::abs(value));
            }
            const double symmetry_tolerance = 1.0e-10 * scale;
            for (std::size_t row = 0; row < kDimension; ++row)
            {
                for (std::size_t column = row + 1; column < kDimension; ++column)
                {
                    const double first = (*matrix)[row * kDimension + column];
                    const double second = (*matrix)[column * kDimension + row];
                    if (std::abs(first - second) > symmetry_tolerance)
                    {
                        invalidCovariance("pose covariance matrix must be symmetric");
                    }
                    const double average = 0.5 * (first + second);
                    (*matrix)[row * kDimension + column] = average;
                    (*matrix)[column * kDimension + row] = average;
                }
            }

            for (std::size_t row = 0; row < kDimension; ++row)
            {
                for (std::size_t column = 0; column < kDimension; ++column)
                {
                    if ((!componentActive(components, row) || !componentActive(components, column)) &&
                        std::abs((*matrix)[row * kDimension + column]) > symmetry_tolerance)
                    {
                        invalidCovariance("inactive pose covariance components must have zero rows and columns");
                    }
                }
            }

            std::array<std::size_t, 6> active{};
            std::size_t active_count = 0;
            for (std::size_t index = 0; index < kDimension; ++index)
            {
                if (componentActive(components, index))
                {
                    active[active_count++] = index;
                }
            }
            std::array<double, 36> eigen_work{};
            for (std::size_t row = 0; row < active_count; ++row)
            {
                for (std::size_t column = 0; column < active_count; ++column)
                {
                    eigen_work[row * kDimension + column] = (*matrix)[active[row] * kDimension + active[column]];
                }
            }
            for (int iteration = 0; iteration < 128; ++iteration)
            {
                std::size_t pivot_row = 0;
                std::size_t pivot_column = 0;
                double maximum = 0.0;
                for (std::size_t row = 0; row < active_count; ++row)
                {
                    for (std::size_t column = row + 1; column < active_count; ++column)
                    {
                        const double value = std::abs(eigen_work[row * kDimension + column]);
                        if (value > maximum)
                        {
                            maximum = value;
                            pivot_row = row;
                            pivot_column = column;
                        }
                    }
                }
                if (maximum <= 1.0e-14 * scale)
                {
                    break;
                }
                const double app = eigen_work[pivot_row * kDimension + pivot_row];
                const double aqq = eigen_work[pivot_column * kDimension + pivot_column];
                const double apq = eigen_work[pivot_row * kDimension + pivot_column];
                const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                for (std::size_t index = 0; index < active_count; ++index)
                {
                    if (index == pivot_row || index == pivot_column)
                    {
                        continue;
                    }
                    const double aip = eigen_work[index * kDimension + pivot_row];
                    const double aiq = eigen_work[index * kDimension + pivot_column];
                    const double rotated_p = cosine * aip - sine * aiq;
                    const double rotated_q = sine * aip + cosine * aiq;
                    eigen_work[index * kDimension + pivot_row] = rotated_p;
                    eigen_work[pivot_row * kDimension + index] = rotated_p;
                    eigen_work[index * kDimension + pivot_column] = rotated_q;
                    eigen_work[pivot_column * kDimension + index] = rotated_q;
                }
                eigen_work[pivot_row * kDimension + pivot_row] =
                    cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
                eigen_work[pivot_column * kDimension + pivot_column] =
                    sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
                eigen_work[pivot_row * kDimension + pivot_column] = 0.0;
                eigen_work[pivot_column * kDimension + pivot_row] = 0.0;
            }

            const double eigen_tolerance = 1.0e-12 * scale;
            for (std::size_t index = 0; index < active_count; ++index)
            {
                const double eigenvalue = eigen_work[index * kDimension + index];
                if (definiteness == CovarianceDefiniteness::PositiveDefinite ? eigenvalue <= eigen_tolerance
                                                                             : eigenvalue < -eigen_tolerance)
                {
                    invalidCovariance(definiteness == CovarianceDefiniteness::PositiveDefinite
                                          ? "pose covariance must be positive definite on its active components"
                                          : "pose covariance must be positive semidefinite");
                }
            }
        }

        bool validRotation(const RotationMatrix& rotation) noexcept
        {
            for (double value : rotation)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
            }
            const auto dot_row = [&rotation](std::size_t first, std::size_t second)
            {
                return rotation[first] * rotation[second] + rotation[first + 1] * rotation[second + 1] +
                       rotation[first + 2] * rotation[second + 2];
            };
            const double determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                                       rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                                       rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
            constexpr double tolerance = 1.0e-8;
            return std::abs(dot_row(0, 0) - 1.0) <= tolerance && std::abs(dot_row(3, 3) - 1.0) <= tolerance &&
                   std::abs(dot_row(6, 6) - 1.0) <= tolerance && std::abs(dot_row(0, 3)) <= tolerance &&
                   std::abs(dot_row(0, 6)) <= tolerance && std::abs(dot_row(3, 6)) <= tolerance &&
                   std::abs(determinant - 1.0) <= tolerance;
        }

    } // namespace

    PoseCovariance PoseCovariance::diagonal(std::vector<double> values,
                                            PoseCovarianceComponents components,
                                            CovarianceDefiniteness definiteness)
    {
        std::array<double, 36> matrix = diagonalMatrix(values);
        validateCovariance(&matrix, components, definiteness);
        return PoseCovariance(CovarianceLayout::Diagonal6, std::move(values), matrix, components, definiteness);
    }

    PoseCovariance PoseCovariance::symmetric(std::vector<double> values,
                                             PoseCovarianceComponents components,
                                             CovarianceDefiniteness definiteness)
    {
        std::array<double, 36> matrix = symmetricMatrix(values);
        validateCovariance(&matrix, components, definiteness);
        return PoseCovariance(CovarianceLayout::Symmetric6x6, std::move(values), matrix, components, definiteness);
    }

    PoseCovariance PoseCovariance::matrix(std::array<double, 36> rowMajor,
                                          PoseCovarianceComponents components,
                                          CovarianceDefiniteness definiteness)
    {
        validateCovariance(&rowMajor, components, definiteness);
        return PoseCovariance(
            CovarianceLayout::Symmetric6x6, packSymmetric(rowMajor), rowMajor, components, definiteness);
    }

    PoseCovariance PoseCovariance::positionDiagonal(Vector3 variancesMetersSquared, CovarianceDefiniteness definiteness)
    {
        return diagonal(
            {variancesMetersSquared[0], variancesMetersSquared[1], variancesMetersSquared[2], 0.0, 0.0, 0.0},
            PoseCovarianceComponents::PositionOnly,
            definiteness);
    }

    PoseCovariance PoseCovariance::rotationDiagonal(Vector3 variancesRadiansSquared,
                                                    CovarianceDefiniteness definiteness)
    {
        return diagonal(
            {0.0, 0.0, 0.0, variancesRadiansSquared[0], variancesRadiansSquared[1], variancesRadiansSquared[2]},
            PoseCovarianceComponents::RotationOnly,
            definiteness);
    }

    CovarianceLayout PoseCovariance::layout() const noexcept
    {
        return _layout;
    }

    PoseCovarianceComponents PoseCovariance::components() const noexcept
    {
        return _components;
    }

    CovarianceDefiniteness PoseCovariance::definiteness() const noexcept
    {
        return _definiteness;
    }

    bool PoseCovariance::hasPosition() const noexcept
    {
        return _components != PoseCovarianceComponents::RotationOnly;
    }

    bool PoseCovariance::hasRotation() const noexcept
    {
        return _components != PoseCovarianceComponents::PositionOnly;
    }

    const std::vector<double>& PoseCovariance::values() const noexcept
    {
        return _values;
    }

    const std::array<double, 36>& PoseCovariance::matrixValues() const noexcept
    {
        return _matrix;
    }

    PoseCovariance::PoseCovariance(CovarianceLayout layout,
                                   std::vector<double> values,
                                   std::array<double, 36> matrix,
                                   PoseCovarianceComponents components,
                                   CovarianceDefiniteness definiteness)
        : _layout(layout), _values(std::move(values)), _matrix(matrix), _components(components),
          _definiteness(definiteness)
    {
    }

    Result<PoseCovariance> propagatePoseCovarianceSimilarity(const PoseCovariance& covariance,
                                                             double scale,
                                                             const RotationMatrix& targetFromSource)
    {
        if (!std::isfinite(scale) || !(scale > 0.0) || !validRotation(targetFromSource))
        {
            return Result<PoseCovariance>::failure(
                CameraErrorCode::InvalidArgument,
                "pose covariance propagation requires a positive scale and proper rotation");
        }
        std::array<double, 36> jacobian{};
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                jacobian[row * kDimension + column] = scale * targetFromSource[row * 3 + column];
                jacobian[(row + 3) * kDimension + column + 3] = targetFromSource[row * 3 + column];
            }
        }
        std::array<double, 36> intermediate{};
        std::array<double, 36> propagated{};
        for (std::size_t row = 0; row < kDimension; ++row)
        {
            for (std::size_t column = 0; column < kDimension; ++column)
            {
                for (std::size_t inner = 0; inner < kDimension; ++inner)
                {
                    intermediate[row * kDimension + column] +=
                        jacobian[row * kDimension + inner] * covariance.matrixValues()[inner * kDimension + column];
                }
            }
        }
        for (std::size_t row = 0; row < kDimension; ++row)
        {
            for (std::size_t column = 0; column < kDimension; ++column)
            {
                for (std::size_t inner = 0; inner < kDimension; ++inner)
                {
                    propagated[row * kDimension + column] +=
                        intermediate[row * kDimension + inner] * jacobian[column * kDimension + inner];
                }
            }
        }
        try
        {
            return Result<PoseCovariance>::success(
                PoseCovariance::matrix(propagated, covariance.components(), covariance.definiteness()));
        }
        catch (const CameraValidationError& error)
        {
            return Result<PoseCovariance>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<PoseCovariance>::failure(CameraErrorCode::InvalidModelState, error.what());
        }
    }

    Result<PoseCovariance> propagatePoseCovarianceRigid(const PoseCovariance& covariance,
                                                        const RotationMatrix& targetFromSource)
    {
        return propagatePoseCovarianceSimilarity(covariance, 1.0, targetFromSource);
    }

} // namespace placamera::reference
