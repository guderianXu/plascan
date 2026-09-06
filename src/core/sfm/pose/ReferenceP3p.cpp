#include "ReferenceP3p.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
    namespace
    {

        std::vector<double> monicQuadraticRealRoots(double linear, double constant)
        {
            const double half_linear = 0.5 * linear;
            double discriminant = half_linear * half_linear;
            discriminant -= constant;
            if (discriminant < 0.0)
            {
                return {};
            }
            if (discriminant == 0.0)
            {
                return {-half_linear};
            }
            const double square_root = std::sqrt(discriminant);
            return {square_root - half_linear, -half_linear - square_root};
        }

        std::vector<double> cubicRealRoots(double constant, double linear, double quadratic, double cubic)
        {
            if (cubic == 0.0)
            {
                if (quadratic == 0.0)
                {
                    if (linear == 0.0)
                    {
                        return {};
                    }
                    return {-constant / linear};
                }
                const double inverse = 1.0 / quadratic;
                return monicQuadraticRealRoots(linear * inverse, constant * inverse);
            }

            quadratic /= cubic;
            linear /= cubic;
            constant /= cubic;

            double cardano_q = quadratic * quadratic;
            cardano_q -= 3.0 * linear;
            cardano_q *= 0.1111111111111111;

            double numerator = quadratic + quadratic;
            numerator *= quadratic;
            numerator *= quadratic;
            numerator -= linear * (9.0 * quadratic);
            double cardano_r = 27.0 * constant;
            cardano_r += numerator;
            cardano_r *= 0.018518518518518517;

            double q_cubed = cardano_q * cardano_q;
            q_cubed *= cardano_q;
            const double discriminant = q_cubed - cardano_r * cardano_r;
            const double shift = quadratic * 0.33333333333333331;
            if (discriminant >= 0.0 && q_cubed > 0.0)
            {
                const double ratio = std::clamp(cardano_r / std::sqrt(q_cubed), -1.0, 1.0);
                double angle = std::acos(ratio);
                const double factor = std::sqrt(std::max(0.0, cardano_q)) * -2.0;
                angle *= 0.33333333333333331;
                return {factor * std::cos(angle) - shift,
                        factor * std::cos(angle + 2.0943951023931953) - shift,
                        factor * std::cos(angle + 4.1887902047863905) - shift};
            }
            if (discriminant >= 0.0)
            {
                return {-shift};
            }

            const double square_root = std::sqrt(-discriminant);
            double cube_root = std::pow(std::abs(cardano_r) + square_root, 0.33333333333300003);
            if (cardano_r > 0.0)
            {
                cube_root = -cube_root;
            }
            if (cube_root == 0.0)
            {
                return {-shift};
            }
            return {cardano_q / cube_root + cube_root - shift};
        }

        std::vector<double> realPolynomialRoots(std::vector<double> coefficients)
        {
            while (coefficients.size() > 1 && coefficients.back() == 0.0)
            {
                coefficients.pop_back();
            }
            if (coefficients.size() == 4)
            {
                return cubicRealRoots(coefficients[0], coefficients[1], coefficients[2], coefficients[3]);
            }
            if (coefficients.size() != 5)
            {
                return {};
            }

            const double leading = coefficients[4];
            double quartic = coefficients[3] / leading;
            double quadratic = coefficients[2] / leading;
            double linear = coefficients[1] / leading;
            double constant = coefficients[0] / leading;

            const double shift = quartic * 0.25;
            double six_shift_squared = 6.0 * shift;
            six_shift_squared *= shift;
            const double depressed_quadratic = quadratic - six_shift_squared;

            double three_shift_squared = 3.0 * shift;
            three_shift_squared *= shift;
            double eight_shift_squared = 8.0 * shift;
            eight_shift_squared *= shift;
            eight_shift_squared -= quadratic + quadratic;
            double depressed_linear = eight_shift_squared * shift;
            depressed_linear += linear;

            double depressed_constant = quadratic - three_shift_squared;
            depressed_constant *= shift;
            depressed_constant -= linear;
            depressed_constant *= shift;
            depressed_constant += constant;

            if (depressed_linear == 0.0)
            {
                double discriminant = depressed_quadratic * depressed_quadratic;
                discriminant -= 4.0 * depressed_constant;
                if (discriminant < 0.0)
                {
                    return {};
                }
                const double discriminant_root = std::sqrt(discriminant);
                double first_square = (discriminant_root - depressed_quadratic) * 0.5;
                if (first_square < 0.0)
                {
                    return {};
                }
                const double first_root = std::sqrt(first_square);
                std::vector<double> roots{first_root - shift, -first_root - shift};
                if (discriminant_root > 0.0)
                {
                    const double second_square = (-depressed_quadratic - discriminant_root) * 0.5;
                    if (second_square >= 0.0)
                    {
                        const double second_root = std::sqrt(second_square);
                        roots.push_back(second_root - shift);
                        roots.push_back(-second_root - shift);
                    }
                }
                return roots;
            }

            double resolvent_constant = 0.5 * depressed_constant;
            resolvent_constant *= depressed_quadratic;
            double linear_square_over_eight = 0.125 * depressed_linear;
            linear_square_over_eight *= depressed_linear;
            resolvent_constant -= linear_square_over_eight;
            const std::vector<double> resolvent_roots =
                cubicRealRoots(resolvent_constant, -depressed_constant, depressed_quadratic * -0.5, 1.0);
            if (resolvent_roots.empty())
            {
                return {};
            }
            const double z = resolvent_roots.front();
            const double z_squared_minus_constant = z * z - depressed_constant;
            const double twice_z_minus_quadratic = z + z - depressed_quadratic;
            if (z_squared_minus_constant < 0.0 || twice_z_minus_quadratic < 0.0)
            {
                return {};
            }
            const double first_square_root = std::sqrt(z_squared_minus_constant);
            double signed_root = std::sqrt(twice_z_minus_quadratic);
            double opposite_root = -signed_root;
            if (depressed_linear < 0.0)
            {
                std::swap(signed_root, opposite_root);
            }
            std::vector<double> roots = monicQuadraticRealRoots(signed_root, z - first_square_root);
            std::vector<double> second_roots = monicQuadraticRealRoots(opposite_root, z + first_square_root);
            roots.insert(roots.end(), second_roots.begin(), second_roots.end());
            for (double& root : roots)
            {
                root -= shift;
            }
            return roots;
        }

        cv::Vec3d toVector(const std::array<double, 3>& value)
        {
            return {value[0], value[1], value[2]};
        }

        std::array<double, 9> toArray(const cv::Matx33d& matrix)
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

    } // namespace

    std::vector<ReferenceWorldToCameraPose>
    solveReferenceP3p(const std::array<std::array<double, 3>, 3>& worldPoints,
                      const std::array<std::array<double, 3>, 3>& bearingVectors)
    {
        std::array<cv::Vec3d, 3> world{};
        std::array<cv::Vec3d, 3> input_rays{};
        for (std::size_t index = 0; index < 3; ++index)
        {
            world[index] = toVector(worldPoints[index]);
            input_rays[index] = toVector(bearingVectors[index]);
        }

        const double d01 = (world[0] - world[1]).dot(world[0] - world[1]);
        const double d12 = (world[1] - world[2]).dot(world[1] - world[2]);
        const double d02 = (world[0] - world[2]).dot(world[0] - world[2]);
        if (d01 < 1e-8 || d12 < 1e-8 || d02 < 1e-8)
        {
            return {};
        }

        std::array<double, 3> inverse_norms{};
        for (std::size_t index = 0; index < 3; ++index)
        {
            const double length_squared = input_rays[index].dot(input_rays[index]);
            if (!(length_squared > 0.0) || !std::isfinite(length_squared))
            {
                return {};
            }
            inverse_norms[index] = 1.0 / std::sqrt(length_squared);
        }
        const auto normalizedDot = [&](std::size_t first, std::size_t second)
        { return input_rays[first].dot(input_rays[second]) * inverse_norms[first] * inverse_norms[second]; };
        const double c01 = normalizedDot(0, 1);
        const double c02 = normalizedDot(0, 2);
        const double c12 = normalizedDot(1, 2);
        if (c01 > 0.9999999899999999 || c02 > 0.9999999899999999 || c12 > 0.9999999899999999)
        {
            return {};
        }

        const double a = d12 / d02;
        const double b = d12 / d01;
        const double one_minus_a = 1.0 - a;
        const double one_minus_b = 1.0 - b;
        const double m = a * b - a;
        const double plus = a * b + a - b;
        const double middle = b + a * b - a;
        const double c12_squared = c12 * c12;
        const double c02_squared = c02 * c02;
        std::vector<double> polynomial(5);
        double p4_subtrahend = a * 4.0;
        p4_subtrahend *= b;
        p4_subtrahend *= c12_squared;
        polynomial[4] = (m - b) * (m - b) - p4_subtrahend;
        polynomial[3] = 4.0 * a * c12 * (2.0 * b * c01 * c12 + middle * c02) + 4.0 * (m - b) * b * one_minus_a * c01;
        polynomial[2] =
            2.0 * (m - b) * plus + std::pow(2.0 * b * one_minus_a * c01, 2.0) +
            4.0 * a * (one_minus_b * a * c02_squared + (a - b) * c12_squared - 2.0 * b * (a + 1.0) * c01 * c02 * c12);
        double p1_first = a + a;
        p1_first *= b;
        p1_first *= c01;
        p1_first *= c02_squared;
        double p1_middle = m;
        p1_middle += b;
        p1_middle *= c02;
        p1_middle *= c12;
        p1_first += p1_middle;
        double four_a = a;
        four_a *= 4.0;
        p1_first *= four_a;
        double p1_second = plus;
        p1_second *= 4.0;
        p1_second *= b;
        p1_second *= one_minus_a;
        p1_second *= c01;
        polynomial[1] = p1_first + p1_second;
        double p0_subtrahend = a * a;
        p0_subtrahend *= 4.0;
        p0_subtrahend *= b;
        p0_subtrahend *= c02_squared;
        polynomial[0] = plus * plus - p0_subtrahend;

        std::vector<ReferenceWorldToCameraPose> poses;
        for (const double x : realPolynomialRoots(std::move(polynomial)))
        {
            if (!(x > 0.0))
            {
                continue;
            }
            const double denominator = x * x - 2.0 * c01 * x + 1.0;
            const double depth0_squared = d01 / denominator;
            if (!(depth0_squared > 0.0) || !std::isfinite(depth0_squared))
            {
                continue;
            }
            const double depth0 = std::sqrt(depth0_squared);
            const double x2_minus_a = x * x - a;
            const double equation = 2.0 * x * b * c01 + x * x * one_minus_b - b;
            const double divisor = one_minus_a * equation - x2_minus_a;
            if (std::abs(divisor) < 1e-14)
            {
                continue;
            }
            const double depth1 = x * depth0;
            const double depth2 =
                ((-2.0 * x * c12) * x2_minus_a - equation * (2.0 * a * c02 - 2.0 * x * c12)) * depth0 / divisor;
            if (!std::isfinite(depth2))
            {
                continue;
            }
            const std::array<cv::Vec3d, 3> local{{input_rays[0] * depth0 * inverse_norms[0],
                                                  input_rays[1] * depth1 * inverse_norms[1],
                                                  input_rays[2] * depth2 * inverse_norms[2]}};
            cv::Vec3d world_center{};
            cv::Vec3d local_center{};
            for (std::size_t index = 0; index < 3; ++index)
            {
                world_center += world[index];
                local_center += local[index];
            }
            world_center *= 1.0 / 3.0;
            local_center *= 1.0 / 3.0;
            cv::Matx33d covariance = cv::Matx33d::zeros();
            for (std::size_t index = 0; index < 3; ++index)
            {
                const cv::Vec3d p = world[index] - world_center;
                const cv::Vec3d q = local[index] - local_center;
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        covariance(row, column) += p[row] * q[column];
                    }
                }
            }
            cv::Mat singular_values;
            cv::Mat u;
            cv::Mat vt;
            cv::SVD::compute(cv::Mat(covariance), singular_values, u, vt, cv::SVD::FULL_UV);
            cv::Mat rotation_matrix = vt.t() * u.t();
            if (cv::determinant(rotation_matrix) < 0.0)
            {
                cv::Mat v = vt.t();
                v.col(2) *= -1.0;
                rotation_matrix = v * u.t();
            }
            cv::Matx33d rotation;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    rotation(row, column) = rotation_matrix.at<double>(row, column);
                }
            }
            const cv::Vec3d translation_basis = rotation.t() * local_center - world_center;
            const cv::Vec3d translation = rotation * translation_basis;
            if (!std::isfinite(translation[0]) || !std::isfinite(translation[1]) || !std::isfinite(translation[2]))
            {
                continue;
            }
            poses.push_back({toArray(rotation), {{translation[0], translation[1], translation[2]}}});
        }
        return poses;
    }

} // namespace xjw
