#pragma once

#include "CameraErrors.h"

#include <cmath>
#include <utility>
#include <vector>

namespace xjw::camera_core
{

    enum class CovarianceLayout
    {
        Diagonal6,
        Symmetric6x6,
    };

    class PoseCovariance
    {
    public:
        static PoseCovariance diagonal(std::vector<double> values)
        {
            return create(CovarianceLayout::Diagonal6, std::move(values), 6);
        }

        static PoseCovariance symmetric(std::vector<double> values)
        {
            return create(CovarianceLayout::Symmetric6x6, std::move(values), 21);
        }

        CovarianceLayout layout() const noexcept
        {
            return _layout;
        }

        const std::vector<double>& values() const noexcept
        {
            return _values;
        }

    private:
        static PoseCovariance create(CovarianceLayout layout, std::vector<double> values, std::size_t expectedSize)
        {
            if (values.size() != expectedSize)
            {
                throw CameraValidationError(CameraErrorCode::InvalidCovariance,
                                            "pose covariance has an invalid number of values");
            }
            for (double value : values)
            {
                if (!std::isfinite(value))
                {
                    throw CameraValidationError(CameraErrorCode::InvalidCovariance,
                                                "pose covariance must contain finite values");
                }
            }
            return PoseCovariance(layout, std::move(values));
        }

        PoseCovariance(CovarianceLayout layout, std::vector<double> values)
            : _layout(layout), _values(std::move(values))
        {
        }

        CovarianceLayout _layout;
        std::vector<double> _values;
    };

} // namespace xjw::camera_core
