#include "metalign/math.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace metalign
{
    namespace
    {

        double target_signed_magnitude(double magnitude, double sign_source)
        {
            const double absolute = std::abs(magnitude);
            return sign_source >= 0.0 ? absolute : -absolute;
        }

        double target_stable_hypot(double first, double second)
        {
            const double absolute_first = std::abs(first);
            const double absolute_second = std::abs(second);
            if (absolute_first > absolute_second)
            {
                const double ratio = absolute_second / absolute_first;
                return absolute_first * std::sqrt(1.0 + ratio * ratio);
            }
            if (absolute_second == 0.0)
                return 0.0;
            const double ratio = absolute_first / absolute_second;
            return absolute_second * std::sqrt(1.0 + ratio * ratio);
        }

        double target_grouped_dot(const double* first,
                                  std::ptrdiff_t first_stride,
                                  const double* second,
                                  std::ptrdiff_t second_stride,
                                  std::size_t count)
        {
            // sub_4108C40 accumulates four products into a temporary subtotal and only
            // then adds that subtotal to the running sum.  Its scalar tail is linear.
            // This ordering is observable in the smallest singular vectors of the
            // 9x5 transpose used by the five-point solver.
            double sum = 0.0;
            std::size_t index = 0;
            for (; index + 4 <= count; index += 4)
            {
                double group = first[index * first_stride] * second[index * second_stride];
                group += first[(index + 1) * first_stride] * second[(index + 1) * second_stride];
                group += first[(index + 2) * first_stride] * second[(index + 2) * second_stride];
                group += first[(index + 3) * first_stride] * second[(index + 3) * second_stride];
                sum += group;
            }
            for (; index < count; ++index)
                sum += first[index * first_stride] * second[index * second_stride];
            return sum;
        }

    } // namespace

    SvdResult svd_target_golub_reinsch(const std::vector<double>& row_major,
                                       std::size_t rows,
                                       std::size_t cols,
                                       std::vector<double>* full_left_vectors,
                                       bool singular_values_only)
    {
        if (rows == 0 || cols == 0 || row_major.size() != rows * cols || rows < cols)
            throw std::invalid_argument("target SVD requires a nonempty rows >= cols matrix");

        // sub_4109450 is the classic Golub-Reinsch path: Householder reduction to
        // bidiagonal form, accumulation of both orthogonal factors, then at most
        // 30 implicit-shift QR steps for each singular value.  Keep the scalar
        // loop nesting explicit because its rounding is part of the recovered ABI.
        // The target can request the complete left factor.  This matters for its
        // 5x9 five-point design: the wide-matrix entry transposes to 9x5 and uses
        // the last four columns of the complete 9x9 left factor as the nullspace.
        std::vector<double> u(rows * rows, 0.0);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < cols; ++column)
                u[row * rows + column] = row_major[row * cols + column];
        // The target keeps the left Householder vectors in the caller-provided
        // factor buffer while reducing a separate working matrix.  In particular,
        // these vectors remain normalized; the working columns are scaled back.
        std::vector<double> left_householders;
        std::vector<double> right_householders;
        if (!singular_values_only)
        {
            left_householders.assign(rows * rows, 0.0);
            right_householders.assign(cols * cols, 0.0);
        }
        std::vector<double> singular(cols, 0.0);
        std::vector<double> right;
        if (!singular_values_only)
            right.assign(cols * cols, 0.0);
        std::vector<double> superdiagonal(cols, 0.0);
        // sub_4109450 saves each left Householder coefficient inside the requested
        // full-left buffer and reuses it verbatim during factor accumulation.
        // Reconstructing it later from the singular value is algebraically equal
        // but not floating-point equivalent.
        std::vector<double> left_reflector_factor(cols, 0.0);
        std::vector<double> right_reflector_factor(cols, 0.0);
        auto U = [&](std::size_t row, std::size_t column) -> double& { return u[row * rows + column]; };
        auto V = [&](std::size_t row, std::size_t column) -> double& { return right[row * cols + column]; };

        double scale = 0.0;
        double g = 0.0;
        double norm_bound = 0.0;
        for (std::size_t column = 0; column < cols; ++column)
        {
            const std::size_t next = column + 1;
            superdiagonal[column] = scale * g;
            double sum = 0.0;
            g = 0.0;
            scale = 0.0;
            for (std::size_t row = column; row < rows; ++row)
                scale += std::abs(U(row, column));
            if (scale != 0.0)
            {
                const double inverse_scale = 1.0 / scale;
                for (std::size_t row = column; row < rows; ++row)
                {
                    U(row, column) *= inverse_scale;
                    sum += U(row, column) * U(row, column);
                }
                const double first = U(column, column);
                g = -target_signed_magnitude(std::sqrt(sum), first);
                const double factor = first * g - sum;
                const double inverse_factor = 1.0 / factor;
                if (!singular_values_only)
                    left_reflector_factor[column] = inverse_factor;
                U(column, column) = first - g;
                if (!singular_values_only)
                {
                    for (std::size_t row = column; row < rows; ++row)
                        left_householders[row * rows + column] = U(row, column);
                }
                for (std::size_t other = next; other < cols; ++other)
                {
                    sum = 0.0;
                    for (std::size_t row = column; row < rows; ++row)
                        sum += U(row, column) * U(row, other);
                    const double multiplier = sum * inverse_factor;
                    for (std::size_t row = column; row < rows; ++row)
                        U(row, other) += multiplier * U(row, column);
                }
                for (std::size_t row = column; row < rows; ++row)
                    U(row, column) *= scale;
            }
            singular[column] = scale * g;

            sum = 0.0;
            g = 0.0;
            scale = 0.0;
            if (column < rows && next < cols)
            {
                for (std::size_t other = next; other < cols; ++other)
                    scale += std::abs(U(column, other));
                if (scale != 0.0)
                {
                    const double inverse_scale = 1.0 / scale;
                    for (std::size_t other = next; other < cols; ++other)
                    {
                        U(column, other) *= inverse_scale;
                        sum += U(column, other) * U(column, other);
                    }
                    const double first = U(column, next);
                    g = -target_signed_magnitude(std::sqrt(sum), first);
                    const double factor = first * g - sum;
                    const double inverse_factor = 1.0 / factor;
                    U(column, next) = first - g;
                    if (!singular_values_only)
                    {
                        right_reflector_factor[next] = inverse_factor;
                        for (std::size_t other = next; other < cols; ++other)
                            right_householders[other * cols + next] = U(column, other);
                    }
                    for (std::size_t other = next; other < cols; ++other)
                        superdiagonal[other] = U(column, other) * inverse_factor;
                    for (std::size_t row = next; row < rows; ++row)
                    {
                        sum = target_grouped_dot(&U(row, next), 1, &U(column, next), 1, cols - next);
                        const double multiplier = sum * inverse_factor;
                        for (std::size_t other = next; other < cols; ++other)
                            U(row, other) += multiplier * U(column, other);
                    }
                    for (std::size_t other = next; other < cols; ++other)
                        U(column, other) *= scale;
                }
            }
            norm_bound = std::max(norm_bound, std::abs(singular[column]) + std::abs(superdiagonal[column]));
        }

        const double qr_tolerance = norm_bound * std::numeric_limits<double>::epsilon();
        // As with the complete left factor, target accumulates the right factor
        // from a separately retained set of normalized Householder vectors.  The
        // textbook reconstruction from the scaled working row is algebraically
        // equivalent but differs by one or two ulps in the 3x3 pose SVD.
        if (!singular_values_only)
        {
            right = std::move(right_householders);
            for (std::size_t reverse = cols; reverse-- > 0;)
            {
                const std::size_t column = reverse;
                const std::size_t following = column + 1;
                for (std::size_t other = following; other < cols; ++other)
                    V(column, other) = 0.0;
                const double inverse_factor = right_reflector_factor[column];
                if (inverse_factor != 0.0)
                {
                    for (std::size_t other = following; other < cols; ++other)
                    {
                        const double sum = target_grouped_dot(&V(following, column),
                                                              static_cast<std::ptrdiff_t>(cols),
                                                              &V(following, other),
                                                              static_cast<std::ptrdiff_t>(cols),
                                                              cols - following);
                        const double multiplier = sum * inverse_factor;
                        for (std::size_t row = column; row < cols; ++row)
                            V(row, other) += multiplier * V(row, column);
                    }
                    const double column_scale = inverse_factor * V(column, column);
                    for (std::size_t row = column; row < cols; ++row)
                        V(row, column) *= column_scale;
                }
                else
                {
                    for (std::size_t row = column; row < cols; ++row)
                        V(row, column) = 0.0;
                }
                V(column, column) += 1.0;
            }

            // Right-factor accumulation above consumes the reduced working matrix.
            // Left-factor accumulation below operates on the separately retained,
            // normalized reflectors (v181 in sub_4109450).
            u = std::move(left_householders);
            for (std::size_t column = cols; column < rows; ++column)
                U(column, column) = 1.0;
            for (std::size_t reverse = std::min(rows, cols); reverse-- > 0;)
            {
                const std::size_t column = reverse;
                const std::size_t following = column + 1;
                g = singular[column];
                for (std::size_t other = following; other < rows; ++other)
                    U(column, other) = 0.0;
                if (g != 0.0)
                {
                    const double inverse_factor = left_reflector_factor[column];
                    for (std::size_t other = following; other < rows; ++other)
                    {
                        const double sum = target_grouped_dot(&U(following, column),
                                                              static_cast<std::ptrdiff_t>(rows),
                                                              &U(following, other),
                                                              static_cast<std::ptrdiff_t>(rows),
                                                              rows - following);
                        const double multiplier = sum * inverse_factor;
                        for (std::size_t row = column; row < rows; ++row)
                            U(row, other) += multiplier * U(row, column);
                    }
                    const double column_scale = inverse_factor * U(column, column);
                    for (std::size_t row = column; row < rows; ++row)
                        U(row, column) *= column_scale;
                }
                else
                {
                    for (std::size_t row = column; row < rows; ++row)
                        U(row, column) = 0.0;
                }
                U(column, column) += 1.0;
            }
        }

        if (std::getenv("METALIGN_TRACE_TARGET_SVD_QR"))
        {
            std::cerr << std::setprecision(17) << "target_svd_qr_entry rows=" << rows << " cols=" << cols
                      << " singular=[";
            for (std::size_t i = 0; i < cols; ++i)
                std::cerr << (i ? "," : "") << singular[i];
            std::cerr << "] superdiagonal=[";
            for (std::size_t i = 0; i < cols; ++i)
                std::cerr << (i ? "," : "") << superdiagonal[i];
            std::cerr << "] tolerance=" << qr_tolerance << " left=[";
            for (std::size_t i = 0; i < u.size(); ++i)
                std::cerr << (i ? "," : "") << u[i];
            std::cerr << "] right=[";
            for (std::size_t i = 0; i < right.size(); ++i)
                std::cerr << (i ? "," : "") << right[i];
            std::cerr << "]\n";
        }

        for (std::size_t reverse = cols; reverse-- > 0;)
        {
            const std::size_t value_index = reverse;
            bool converged = false;
            for (std::size_t iteration = 0; iteration < 30; ++iteration)
            {
                bool cancel = true;
                std::size_t split = 0;
                std::size_t previous = 0;
                for (std::size_t scan = value_index + 1; scan-- > 0;)
                {
                    split = scan;
                    if (std::abs(superdiagonal[split]) <= qr_tolerance)
                    {
                        cancel = false;
                        break;
                    }
                    if (split == 0)
                        break;
                    previous = split - 1;
                    if (std::abs(singular[previous]) <= qr_tolerance)
                        break;
                }
                if (cancel)
                {
                    double cosine = 0.0;
                    double sine = 1.0;
                    for (std::size_t row = split; row <= value_index; ++row)
                    {
                        const double first = sine * superdiagonal[row];
                        superdiagonal[row] = cosine * superdiagonal[row];
                        if (std::abs(first) + qr_tolerance == qr_tolerance)
                            break;
                        const double second = singular[row];
                        const double hypotenuse = target_stable_hypot(first, second);
                        singular[row] = hypotenuse;
                        // sub_4109450 emits two independent divsd operations here;
                        // multiplying by a shared reciprocal changes the last bit.
                        cosine = second / hypotenuse;
                        sine = -first / hypotenuse;
                        if (!singular_values_only)
                        {
                            for (std::size_t matrix_row = 0; matrix_row < rows; ++matrix_row)
                            {
                                const double left = U(matrix_row, previous);
                                const double right_value = U(matrix_row, row);
                                U(matrix_row, previous) = left * cosine + right_value * sine;
                                U(matrix_row, row) = right_value * cosine - left * sine;
                            }
                        }
                    }
                }
                double final_value = singular[value_index];
                if (split == value_index)
                {
                    if (final_value < 0.0)
                    {
                        singular[value_index] = -final_value;
                        if (!singular_values_only)
                        {
                            for (std::size_t row = 0; row < cols; ++row)
                                V(row, value_index) = -V(row, value_index);
                        }
                    }
                    converged = true;
                    break;
                }
                if (iteration == 29)
                    break;

                const std::size_t penultimate = value_index - 1;
                double first_value = singular[split];
                double middle_value = singular[penultimate];
                double first_super = superdiagonal[penultimate];
                double final_super = superdiagonal[value_index];
                // sub_4109450 evaluates the classic Golub-Reinsch shift in this
                // divided form.  The algebra is equivalent to the textbook
                // expression, but the operation order is not interchangeable for
                // the nearly rank-deficient five-point design.
                double shift =
                    (((first_super + final_value) / final_super) * ((first_super - final_value) / middle_value) +
                     middle_value / final_super - final_super / middle_value) *
                    0.5;
                first_super = target_stable_hypot(shift, 1.0);
                if (shift < 0.0)
                    first_super = -first_super;
                shift = first_value - final_value * (final_value / first_value) +
                        (middle_value / (shift + first_super) - final_super) * (final_super / first_value);
                double cosine = 1.0;
                double sine = 1.0;
                for (std::size_t column = split; column <= penultimate; ++column)
                {
                    const std::size_t following = column + 1;
                    first_super = superdiagonal[following];
                    middle_value = singular[following];
                    final_super = sine * first_super;
                    first_super = cosine * first_super;
                    double hypotenuse = target_stable_hypot(shift, final_super);
                    superdiagonal[column] = hypotenuse;
                    cosine = shift / hypotenuse;
                    sine = final_super / hypotenuse;
                    shift = first_value * cosine + first_super * sine;
                    first_super = first_super * cosine - first_value * sine;
                    final_super = middle_value * sine;
                    middle_value *= cosine;
                    if (!singular_values_only)
                    {
                        for (std::size_t row = 0; row < cols; ++row)
                        {
                            first_value = V(row, column);
                            final_value = V(row, following);
                            V(row, column) = first_value * cosine + final_value * sine;
                            V(row, following) = final_value * cosine - first_value * sine;
                        }
                    }
                    hypotenuse = target_stable_hypot(shift, final_super);
                    singular[column] = hypotenuse;
                    if (hypotenuse != 0.0)
                    {
                        // Same direct-divide ordering as v99/v103 in the target
                        // QR loop (0x410A2xx), rather than one reciprocal.
                        cosine = shift / hypotenuse;
                        sine = final_super / hypotenuse;
                    }
                    shift = cosine * first_super + sine * middle_value;
                    first_value = cosine * middle_value - sine * first_super;
                    if (!singular_values_only)
                    {
                        for (std::size_t row = 0; row < rows; ++row)
                        {
                            middle_value = U(row, column);
                            final_value = U(row, following);
                            U(row, column) = middle_value * cosine + final_value * sine;
                            U(row, following) = final_value * cosine - middle_value * sine;
                        }
                    }
                }
                superdiagonal[split] = 0.0;
                superdiagonal[value_index] = shift;
                singular[value_index] = first_value;
            }
            if (!converged)
                throw std::runtime_error("target SVD QR iteration did not converge");
        }

        // sub_4109450 returns descending singular values.  Its final permutation
        // moves the corresponding columns of both orthogonal factors together.
        for (std::size_t destination = 0; destination < cols; ++destination)
        {
            std::size_t selected = destination;
            for (std::size_t candidate = destination + 1; candidate < cols; ++candidate)
                if (singular[candidate] > singular[selected])
                    selected = candidate;
            if (selected == destination)
                continue;
            std::swap(singular[destination], singular[selected]);
            if (!singular_values_only)
            {
                for (std::size_t row = 0; row < rows; ++row)
                    std::swap(U(row, destination), U(row, selected));
                for (std::size_t row = 0; row < cols; ++row)
                    std::swap(V(row, destination), V(row, selected));
            }
        }

        SvdResult result;
        result.rows = rows;
        result.cols = cols;
        if (!singular_values_only)
        {
            result.u.resize(rows * cols);
            for (std::size_t row = 0; row < rows; ++row)
                for (std::size_t column = 0; column < cols; ++column)
                    result.u[row * cols + column] = U(row, column);
            if (full_left_vectors)
                *full_left_vectors = std::move(u);
        }
        result.s = std::move(singular);
        if (!singular_values_only)
        {
            result.vt.resize(cols * cols);
            for (std::size_t row = 0; row < cols; ++row)
                for (std::size_t column = 0; column < cols; ++column)
                    result.vt[row * cols + column] = V(column, row);
        }
        return result;
    }

} // namespace metalign
