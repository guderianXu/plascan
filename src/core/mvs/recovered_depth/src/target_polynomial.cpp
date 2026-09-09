#include "metalign/math.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metalign
{
    std::vector<double> polynomial_real_roots_target(const std::vector<double>& coefficients_ascending)
    {
        constexpr double kCoefficientCutoff = 1e-12;
        constexpr double kRootTolerance = 1e-14;
        constexpr int kMaximumIterations = 800;

        using Poly = std::vector<double>;
        auto trim_exact = [](Poly polynomial)
        {
            while (polynomial.size() > 1 && polynomial.back() == 0.0)
                polynomial.pop_back();
            return polynomial;
        };
        Poly source = trim_exact(coefficients_ascending);
        if (source.size() <= 1)
            return {};

        auto evaluate = [](const Poly& polynomial, double x)
        {
            double value = polynomial.back();
            for (std::size_t i = polynomial.size() - 1; i-- > 0;)
                value = value * x + polynomial[i];
            return value;
        };
        auto variations_at = [&](const std::vector<Poly>& chain, double x)
        {
            int variations = 0;
            double previous = evaluate(chain.front(), x);
            for (std::size_t i = 1; i < chain.size(); ++i)
            {
                const double current = evaluate(chain[i], x);
                // sub_413B570 advances `previous` even when the current value is
                // zero.  A zero previous value therefore counts as a variation
                // against the following record; it is not skipped as in the
                // textbook Sturm convention.
                if (previous == 0.0 || previous * current < 0.0)
                    ++variations;
                previous = current;
            }
            return variations;
        };
        auto variations_at_infinity = [](const std::vector<Poly>& chain, bool negative)
        {
            int variations = 0;
            double previous = 0.0;
            for (const Poly& polynomial : chain)
            {
                double sign = polynomial.back();
                if (negative && ((polynomial.size() - 1) & 1U))
                    sign = -sign;
                if (previous != 0.0 && previous * sign < 0.0)
                    ++variations;
                previous = sign;
            }
            return variations;
        };

        // sub_413B160 stores fixed-size polynomial records.  Record zero is the
        // source polynomial; record one is the derivative divided by the absolute
        // value of its leading coefficient.  Each subsequent record is the
        // normalized negative Euclidean remainder, with 1e-12 leading terms
        // discarded exactly as in the target.
        std::vector<Poly> chain;
        chain.push_back(source);
        const std::size_t degree = source.size() - 1;
        const double derivative_scale = std::abs(static_cast<double>(degree) * source.back());
        Poly derivative(degree);
        for (std::size_t i = 1; i <= degree; ++i)
            derivative[i - 1] = static_cast<double>(i) * source[i] / derivative_scale;
        chain.push_back(std::move(derivative));
        while (chain.back().size() > 1)
        {
            const Poly& dividend = chain[chain.size() - 2];
            const Poly& divisor = chain.back();
            Poly remainder = dividend;
            const std::size_t divisor_degree = divisor.size() - 1;
            for (std::size_t k = remainder.size() - divisor.size() + 1; k-- > 0;)
            {
                const double quotient = remainder[divisor_degree + k] / divisor.back();
                for (std::size_t j = 0; j <= divisor_degree; ++j)
                    remainder[j + k] -= quotient * divisor[j];
            }
            std::size_t remainder_degree = divisor_degree;
            while (remainder_degree > 0 && std::abs(remainder[remainder_degree]) < kCoefficientCutoff)
            {
                remainder[remainder_degree] = 0.0;
                --remainder_degree;
            }
            remainder.resize(remainder_degree + 1);
            const double leading_magnitude = std::abs(remainder.back());
            if (leading_magnitude == 0.0)
                break;
            for (double& coefficient : remainder)
                coefficient = -coefficient / leading_magnitude;
            chain.push_back(std::move(remainder));
        }

        const int negative_infinity = variations_at_infinity(chain, true);
        const int positive_infinity = variations_at_infinity(chain, false);
        if (negative_infinity == positive_infinity)
            return {};

        double left = -1.0;
        for (int i = 0; i < 32 && variations_at(chain, left) != negative_infinity; ++i)
            left *= 10.0;
        double right = 1.0;
        for (int i = 0; i < 32 && variations_at(chain, right) != positive_infinity; ++i)
            right *= 10.0;

        auto refine = [&](double bracket_left, double bracket_right, double& root)
        {
            double value_left = evaluate(source, bracket_left);
            double value_right = evaluate(source, bracket_right);
            if (value_left * value_right > 0.0)
                return false;
            if (std::abs(value_left) < kRootTolerance)
            {
                root = bracket_left;
                return true;
            }
            if (std::abs(value_right) < kRootTolerance)
            {
                root = bracket_right;
                return true;
            }
            double previous_value = value_left;
            for (int iteration = 0; iteration < kMaximumIterations; ++iteration)
            {
                const double candidate =
                    (bracket_left * value_right - bracket_right * value_left) / (value_right - value_left);
                const double value = evaluate(source, candidate);
                if ((std::abs(candidate) > kRootTolerance && std::abs(value / candidate) < kRootTolerance) ||
                    (std::abs(candidate) <= kRootTolerance && std::abs(value) < kRootTolerance))
                {
                    root = candidate;
                    return true;
                }
                previous_value *= value;
                if (value_left * value < 0.0)
                {
                    if (previous_value > 0.0)
                        value_left *= 0.5;
                    bracket_right = candidate;
                    value_right = value;
                }
                else
                {
                    if (previous_value > 0.0)
                        value_right *= 0.5;
                    bracket_left = candidate;
                    value_left = value;
                }
                previous_value = value;
            }
            return false;
        };

        std::vector<double> roots;
        roots.reserve(static_cast<std::size_t>(negative_infinity - positive_infinity));
        auto isolate = [&](auto&& self,
                           double bracket_left,
                           int left_variations,
                           double bracket_right,
                           int right_variations) -> void
        {
            const int count = left_variations - right_variations;
            if (count <= 0)
                return;
            if (count == 1)
            {
                double root = 0.0;
                if (refine(bracket_left, bracket_right, root))
                {
                    roots.push_back(root);
                    return;
                }
                for (int iteration = 0; iteration < kMaximumIterations; ++iteration)
                {
                    const double midpoint = (bracket_left + bracket_right) * 0.5;
                    const int midpoint_variations = variations_at(chain, midpoint);
                    const double width = bracket_right - bracket_left;
                    if ((std::abs(midpoint) > kRootTolerance && std::abs(width / midpoint) < kRootTolerance) ||
                        (std::abs(midpoint) <= kRootTolerance && std::abs(width) < kRootTolerance))
                    {
                        roots.push_back(midpoint);
                        return;
                    }
                    if (midpoint_variations == right_variations)
                        bracket_right = midpoint;
                    else
                    {
                        bracket_left = midpoint;
                        left_variations = midpoint_variations;
                    }
                }
                roots.push_back((bracket_left + bracket_right) * 0.5);
                return;
            }
            for (int iteration = 0; iteration < kMaximumIterations; ++iteration)
            {
                const double midpoint = (bracket_left + bracket_right) * 0.5;
                const int midpoint_variations = variations_at(chain, midpoint);
                if (midpoint_variations == left_variations)
                {
                    bracket_left = midpoint;
                    continue;
                }
                if (midpoint_variations == right_variations)
                {
                    bracket_right = midpoint;
                    continue;
                }
                self(self, bracket_left, left_variations, midpoint, midpoint_variations);
                self(self, midpoint, midpoint_variations, bracket_right, right_variations);
                return;
            }
            for (int i = 0; i < count; ++i)
                roots.push_back((bracket_left + bracket_right) * 0.5);
        };
        isolate(isolate, left, negative_infinity, right, positive_infinity);
        return roots;
    }

} // namespace metalign
