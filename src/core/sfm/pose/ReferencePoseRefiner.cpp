#include "ReferencePoseRefiner.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
    namespace
    {

        struct RodriguesEvaluation
        {
            cv::Matx33d rotation = cv::Matx33d::eye();
            std::array<cv::Matx33d, 3> derivatives{{cv::Matx33d::zeros(), cv::Matx33d::zeros(), cv::Matx33d::zeros()}};
        };

        cv::Matx33d skew(const cv::Vec3d& vector)
        {
            return {0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0], -vector[1], vector[0], 0.0};
        }

        RodriguesEvaluation evaluateRodrigues(const cv::Vec3d& vector)
        {
            RodriguesEvaluation result;
            const double squared_length = vector.dot(vector);
            const double length = std::sqrt(squared_length);
            if (length < std::numeric_limits<double>::epsilon())
            {
                result.derivatives[0] = skew({1.0, 0.0, 0.0});
                result.derivatives[1] = skew({0.0, 1.0, 0.0});
                result.derivatives[2] = skew({0.0, 0.0, 1.0});
                return result;
            }

            const double sine = std::sin(length);
            const double cosine = std::cos(length);
            const double one_minus_cosine = 1.0 - cosine;
            const double inverse_length = 1.0 / length;
            const cv::Vec3d axis = vector * inverse_length;
            const cv::Matx33d axis_skew = skew(axis);
            cv::Matx33d axis_outer;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    axis_outer(row, column) = axis[row] * axis[column];
                    const double identity = row == column ? 1.0 : 0.0;
                    result.rotation(row, column) =
                        identity * cosine + axis_outer(row, column) * one_minus_cosine + axis_skew(row, column) * sine;
                }
            }

            const double twice_one_minus_cosine = one_minus_cosine + one_minus_cosine;
            const double outer_scale = one_minus_cosine * inverse_length;
            const double outer_axis_scale = sine - twice_one_minus_cosine * inverse_length;
            const double skew_basis_scale = inverse_length * sine;
            const double skew_axis_scale = cosine - inverse_length * sine;
            for (int derivative_index = 0; derivative_index < 3; ++derivative_index)
            {
                cv::Vec3d basis{};
                basis[derivative_index] = 1.0;
                const cv::Matx33d basis_skew = skew(basis);
                const double axis_component = axis[derivative_index];
                const double first_scale = outer_axis_scale * axis_component;
                const double identity_scale = -sine * axis_component;
                const double fourth_scale = axis_component * skew_axis_scale;
                cv::Matx33d& derivative = result.derivatives[derivative_index];
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        const double identity = row == column ? 1.0 : 0.0;
                        const double outer_numerator = (row == derivative_index ? axis[column] : 0.0) +
                                                       (column == derivative_index ? axis[row] : 0.0);
                        derivative(row, column) = axis_outer(row, column) * first_scale + identity * identity_scale +
                                                  outer_numerator * outer_scale +
                                                  axis_skew(row, column) * fourth_scale +
                                                  basis_skew(row, column) * skew_basis_scale;
                    }
                }
            }
            return result;
        }

        cv::Matx33d matrixFromArray(const std::array<double, 9>& values)
        {
            return {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]};
        }

        std::array<double, 9> arrayFromMatrix(const cv::Matx33d& matrix)
        {
            return {{matrix(0, 0),
                     matrix(0, 1),
                     matrix(0, 2),
                     matrix(1, 0),
                     matrix(1, 1),
                     matrix(1, 2),
                     matrix(2, 0),
                     matrix(2, 1),
                     matrix(2, 2)}};
        }

        cv::Vec3d rotationVectorFromMatrix(const cv::Matx33d& input)
        {
            cv::Mat singular_values;
            cv::Mat u;
            cv::Mat vt;
            cv::SVD::compute(cv::Mat(input), singular_values, u, vt, cv::SVD::FULL_UV);
            cv::Mat orthogonal_matrix = u * vt;
            if (cv::determinant(orthogonal_matrix) < 0.0)
            {
                u.col(2) *= -1.0;
                orthogonal_matrix = u * vt;
            }
            cv::Matx33d orthogonal;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    orthogonal(row, column) = orthogonal_matrix.at<double>(row, column);
                }
            }
            const double x = orthogonal(2, 1) - orthogonal(1, 2);
            const double y = orthogonal(0, 2) - orthogonal(2, 0);
            const double z = orthogonal(1, 0) - orthogonal(0, 1);
            const double sine = std::sqrt((x * x + y * y + z * z) * 0.25);
            const double cosine =
                std::clamp((orthogonal(0, 0) + orthogonal(1, 1) + orthogonal(2, 2) - 1.0) * 0.5, -1.0, 1.0);
            if (sine < 1e-5)
            {
                return {x * 0.5, y * 0.5, z * 0.5};
            }
            const double scale = (1.0 / (sine + sine)) * std::acos(cosine);
            return {x * scale, y * scale, z * scale};
        }

        bool projectLocal(const FramePinholeCamera& camera, const cv::Vec3d& local, cv::Vec2d* pixel)
        {
            if (!pixel || !(local[2] > 1e-9))
            {
                return false;
            }
            const double inverse_z = 1.0 / local[2];
            const double x = local[0] * inverse_z;
            const double y = local[1] * inverse_z;
            const double r2 = x * x + y * y;
            const FramePinholeCamera::Distortion distortion = camera.distortion();
            const double radial =
                1.0 + distortion.radialK1 * r2 + distortion.radialK2 * r2 * r2 + distortion.radialK3 * r2 * r2 * r2;
            const double distorted_x =
                x * radial + 2.0 * distortion.tangentialP1 * x * y + distortion.tangentialP2 * (r2 + 2.0 * x * x);
            const double distorted_y =
                y * radial + distortion.tangentialP1 * (r2 + 2.0 * y * y) + 2.0 * distortion.tangentialP2 * x * y;
            (*pixel)[0] = camera.focalX() * distorted_x + camera.principalX();
            (*pixel)[1] = camera.focalY() * distorted_y + camera.principalY();
            return std::isfinite((*pixel)[0]) && std::isfinite((*pixel)[1]);
        }

        std::array<double, 6> projectionLocalJacobian(const FramePinholeCamera& camera, const cv::Vec3d& local)
        {
            std::array<double, 6> result{};
            if (std::abs(local[2]) < 1e-15)
            {
                return result;
            }
            const double inverse_z = 1.0 / local[2];
            const double x = local[0] * inverse_z;
            const double y = local[1] * inverse_z;
            const double r2 = x * x + y * y;
            const double r4 = r2 * r2;
            const FramePinholeCamera::Distortion distortion = camera.distortion();
            const double radial =
                1.0 + distortion.radialK1 * r2 + distortion.radialK2 * r4 + distortion.radialK3 * r4 * r2;
            const double radial_derivative =
                distortion.radialK1 + 2.0 * distortion.radialK2 * r2 + 3.0 * distortion.radialK3 * r4;
            const double radial_x = 2.0 * x * radial_derivative;
            const double radial_y = 2.0 * y * radial_derivative;
            const double dxd_x =
                radial + x * radial_x + 2.0 * distortion.tangentialP1 * y + 6.0 * distortion.tangentialP2 * x;
            const double dxd_y = x * radial_y + 2.0 * distortion.tangentialP1 * x + 2.0 * distortion.tangentialP2 * y;
            const double dyd_x = y * radial_x + 2.0 * distortion.tangentialP1 * x + 2.0 * distortion.tangentialP2 * y;
            const double dyd_y =
                radial + y * radial_y + 6.0 * distortion.tangentialP1 * y + 2.0 * distortion.tangentialP2 * x;
            const double z_x = -x * inverse_z;
            const double z_y = -y * inverse_z;
            result = {{camera.focalX() * dxd_x * inverse_z,
                       camera.focalX() * dxd_y * inverse_z,
                       camera.focalX() * (dxd_x * z_x + dxd_y * z_y),
                       camera.focalY() * dyd_x * inverse_z,
                       camera.focalY() * dyd_y * inverse_z,
                       camera.focalY() * (dyd_x * z_x + dyd_y * z_y)}};
            return result;
        }

    } // namespace

    void refineReferencePose(const FramePinholeCamera& camera,
                             const std::vector<std::array<double, 3>>& worldPoints,
                             const std::vector<std::array<double, 2>>& imagePoints,
                             const std::vector<std::size_t>& inlierIndices,
                             ReferenceWorldToCameraPose* pose,
                             std::size_t iterations)
    {
        if (!pose || inlierIndices.size() < 3 || worldPoints.size() != imagePoints.size())
        {
            return;
        }
        const cv::Matx33d initial_rotation = matrixFromArray(pose->rotation);
        cv::Vec3d rotation_vector = rotationVectorFromMatrix(initial_rotation);
        const cv::Vec3d initial_translation(pose->translation[0], pose->translation[1], pose->translation[2]);
        cv::Vec3d center = -(initial_rotation.t() * initial_translation);
        double damping = 0.001;

        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            const RodriguesEvaluation evaluation = evaluateRodrigues(rotation_vector);
            cv::Matx<double, 6, 6> hessian = cv::Matx<double, 6, 6>::zeros();
            cv::Vec<double, 6> gradient{};
            std::size_t usable = 0;
            for (const std::size_t index : inlierIndices)
            {
                if (index >= worldPoints.size())
                {
                    continue;
                }
                const cv::Vec3d world(worldPoints[index][0], worldPoints[index][1], worldPoints[index][2]);
                const cv::Vec3d centered = world - center;
                const cv::Vec3d local = evaluation.rotation * centered;
                cv::Vec2d prediction;
                if (!projectLocal(camera, local, &prediction))
                {
                    continue;
                }
                const cv::Vec2d residual(prediction[0] - imagePoints[index][0], prediction[1] - imagePoints[index][1]);
                const std::array<double, 6> projection = projectionLocalJacobian(camera, local);
                double jacobian[2][6]{};
                for (int parameter = 0; parameter < 3; ++parameter)
                {
                    const cv::Vec3d rotation_derivative = evaluation.derivatives[parameter] * centered;
                    for (int row = 0; row < 2; ++row)
                    {
                        jacobian[row][parameter] = projection[row * 3] * rotation_derivative[0] +
                                                   projection[row * 3 + 1] * rotation_derivative[1] +
                                                   projection[row * 3 + 2] * rotation_derivative[2];
                        jacobian[row][parameter + 3] = -(projection[row * 3] * evaluation.rotation(0, parameter) +
                                                         projection[row * 3 + 1] * evaluation.rotation(1, parameter) +
                                                         projection[row * 3 + 2] * evaluation.rotation(2, parameter));
                    }
                }
                for (int row = 0; row < 2; ++row)
                {
                    for (int first = 0; first < 6; ++first)
                    {
                        gradient[first] += jacobian[row][first] * residual[row];
                        for (int second = 0; second < 6; ++second)
                        {
                            hessian(first, second) += jacobian[row][first] * jacobian[row][second];
                        }
                    }
                }
                ++usable;
            }
            if (usable < 3)
            {
                break;
            }
            for (int diagonal = 0; diagonal < 6; ++diagonal)
            {
                hessian(diagonal, diagonal) += damping;
                gradient[diagonal] = -gradient[diagonal];
            }
            cv::Vec<double, 6> update{};
            if (!cv::solve(hessian, gradient, update, cv::DECOMP_CHOLESKY))
            {
                break;
            }
            const cv::Vec<double, 6> previous(
                rotation_vector[0], rotation_vector[1], rotation_vector[2], center[0], center[1], center[2]);
            rotation_vector += cv::Vec3d(update[0], update[1], update[2]);
            center += cv::Vec3d(update[3], update[4], update[5]);
            const cv::Vec<double, 6> next(
                rotation_vector[0], rotation_vector[1], rotation_vector[2], center[0], center[1], center[2]);
            double delta_squared = 0.0;
            double previous_squared = 0.0;
            for (int parameter = 0; parameter < 6; ++parameter)
            {
                const double delta = next[parameter] - previous[parameter];
                delta_squared += delta * delta;
                previous_squared += previous[parameter] * previous[parameter];
            }
            damping *= 0.1;
            if (previous_squared > 0.0 && std::sqrt(delta_squared / previous_squared) <= 0.0005)
            {
                break;
            }
        }

        const cv::Matx33d rotation = evaluateRodrigues(rotation_vector).rotation;
        const cv::Vec3d translation = -(rotation * center);
        pose->rotation = arrayFromMatrix(rotation);
        pose->translation = {{translation[0], translation[1], translation[2]}};
    }

} // namespace xjw
