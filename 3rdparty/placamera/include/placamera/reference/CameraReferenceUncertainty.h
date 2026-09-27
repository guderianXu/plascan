#pragma once

#include <placamera/result.h>
#include <placamera/types.h>

#include <array>
#include <cstddef>
#include <vector>

namespace placamera::reference
{

    enum class CovarianceLayout
    {
        Diagonal6,
        Symmetric6x6,
    };

    /** Canonical tangent order: position XYZ in metres, then rotation XYZ in radians. */
    enum class PoseTangentComponent : std::size_t
    {
        PositionX = 0,
        PositionY = 1,
        PositionZ = 2,
        RotationX = 3,
        RotationY = 4,
        RotationZ = 5,
    };

    enum class PoseCovarianceComponents
    {
        PositionOnly,
        RotationOnly,
        PositionAndRotation,
    };

    enum class CovarianceDefiniteness
    {
        PositiveSemidefinite,
        PositiveDefinite,
    };

    class PoseCovariance
    {
    public:
        static PoseCovariance
        diagonal(std::vector<double> values,
                 PoseCovarianceComponents components = PoseCovarianceComponents::PositionAndRotation,
                 CovarianceDefiniteness definiteness = CovarianceDefiniteness::PositiveSemidefinite);

        /** Upper-triangular row-major packing: (0,0),(0,1)..(0,5),(1,1)..(5,5). */
        static PoseCovariance
        symmetric(std::vector<double> values,
                  PoseCovarianceComponents components = PoseCovarianceComponents::PositionAndRotation,
                  CovarianceDefiniteness definiteness = CovarianceDefiniteness::PositiveSemidefinite);

        static PoseCovariance
        matrix(std::array<double, 36> rowMajor,
               PoseCovarianceComponents components = PoseCovarianceComponents::PositionAndRotation,
               CovarianceDefiniteness definiteness = CovarianceDefiniteness::PositiveSemidefinite);

        static PoseCovariance
        positionDiagonal(placamera::Vector3 variancesMetersSquared,
                         CovarianceDefiniteness definiteness = CovarianceDefiniteness::PositiveSemidefinite);
        static PoseCovariance
        rotationDiagonal(placamera::Vector3 variancesRadiansSquared,
                         CovarianceDefiniteness definiteness = CovarianceDefiniteness::PositiveSemidefinite);

        CovarianceLayout layout() const noexcept;
        PoseCovarianceComponents components() const noexcept;
        CovarianceDefiniteness definiteness() const noexcept;
        bool hasPosition() const noexcept;
        bool hasRotation() const noexcept;
        const std::vector<double>& values() const noexcept;
        const std::array<double, 36>& matrixValues() const noexcept;

    private:
        PoseCovariance(CovarianceLayout layout,
                       std::vector<double> values,
                       std::array<double, 36> matrix,
                       PoseCovarianceComponents components,
                       CovarianceDefiniteness definiteness);

        CovarianceLayout _layout;
        std::vector<double> _values;
        std::array<double, 36> _matrix{};
        PoseCovarianceComponents _components = PoseCovarianceComponents::PositionAndRotation;
        CovarianceDefiniteness _definiteness = CovarianceDefiniteness::PositiveSemidefinite;
    };

    /** Propagate [position, rotation] covariance with J=diag(R,R). */
    placamera::Result<PoseCovariance> propagatePoseCovarianceRigid(const PoseCovariance& covariance,
                                                                   const placamera::RotationMatrix& targetFromSource);

    /** Propagate [position, rotation] covariance with J=diag(scale*R,R). */
    placamera::Result<PoseCovariance> propagatePoseCovarianceSimilarity(
        const PoseCovariance& covariance, double scale, const placamera::RotationMatrix& targetFromSource);

} // namespace placamera::reference
