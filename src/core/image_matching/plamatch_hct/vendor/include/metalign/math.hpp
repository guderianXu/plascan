#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace metalign {

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Mat3 {
    std::array<double, 9> v{};
    double& operator()(std::size_t row, std::size_t col) { return v[row * 3 + col]; }
    double operator()(std::size_t row, std::size_t col) const { return v[row * 3 + col]; }
    static Mat3 identity();
};

struct Pose {
    Mat3 rotation = Mat3::identity();
    Vec3 translation{};
    // Metashape's camera transform record retains C independently from the
    // rounded t=-R*C projection column.  Keep that state when it is available
    // so a later BA/triangulation boundary does not recover C through R^T(-t)
    // and lose one or two ulps.
    std::optional<Vec3> center;
};

Vec2 operator+(Vec2 a, Vec2 b);
Vec2 operator-(Vec2 a, Vec2 b);
Vec2 operator*(Vec2 a, double s);
Vec3 operator+(Vec3 a, Vec3 b);
Vec3 operator-(Vec3 a, Vec3 b);
Vec3 operator*(Vec3 a, double s);
Vec3 operator/(Vec3 a, double s);
double dot(Vec2 a, Vec2 b);
double dot(Vec3 a, Vec3 b);
Vec3 cross(Vec3 a, Vec3 b);
double norm(Vec2 value);
double norm(Vec3 value);
Vec3 normalized(Vec3 value);
Mat3 transpose(const Mat3& matrix);
Mat3 operator*(const Mat3& left, const Mat3& right);
Vec3 operator*(const Mat3& matrix, Vec3 vector);
Mat3 operator*(const Mat3& matrix, double scalar);
Mat3 operator+(const Mat3& left, const Mat3& right);
double determinant(const Mat3& matrix);
Mat3 inverse(const Mat3& matrix);
Mat3 skew(Vec3 vector);
Mat3 rodrigues(Vec3 axis_angle);
Vec3 camera_center(const Pose& pose);

struct SvdResult {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> u;   // row-major rows x min(rows, cols)
    std::vector<double> s;
    std::vector<double> vt;  // row-major min(rows, cols) x cols
};

SvdResult svd(const std::vector<double>& row_major, std::size_t rows, std::size_t cols);
// Householder bidiagonalization followed by implicit-shift QR.  This mirrors
// Metashape's in-binary sub_4109450 instead of delegating to LAPACK, whose
// valid but different basis/rounding is observable in essential decomposition.
SvdResult svd_target_golub_reinsch(
    const std::vector<double>& row_major, std::size_t rows, std::size_t cols,
    std::vector<double>* full_left_vectors = nullptr,
    bool singular_values_only = false);

// Real roots of an ascending-order polynomial.  This mirrors Metashape's
// sub_413B8C0 Sturm-chain isolation and Illinois regula-falsi refinement.
std::vector<double> polynomial_real_roots_target(
    const std::vector<double>& coefficients_ascending);
std::vector<double> smallest_right_singular_vector(
    const std::vector<double>& row_major, std::size_t rows, std::size_t cols);
bool solve_linear_system(std::vector<double>& matrix, std::vector<double>& rhs, std::size_t n);
struct SparseSymmetricMatrix {
    std::size_t dimension = 0;
    // Lower-triangular, sorted, packed CSC. Stored scalar zeroes are allowed so
    // a dense parameter block can retain its target CHOLMOD structure.
    std::vector<std::size_t> column_offsets;
    std::vector<std::size_t> row_indices;
    std::vector<double> values;
};
// Packed row factor used by the recovered 0x42E7C00 / 0x42E9680 scalar
// forward/backward path.  For every row, starts[row] identifies its diagonal
// coefficient; the remaining lengths[row]-1 entries update the column given by
// indices.  This is intentionally a raw replay primitive: it records the
// observed target factor ABI without assigning it a CHOLMOD or Cholesky label.
struct TargetSparseRowFactor {
    std::vector<std::size_t> starts;
    std::vector<std::size_t> indices;
    std::vector<double> coefficients;
    std::vector<std::size_t> lengths;
};
bool target_sparse_factor_forward_substitute(
    const TargetSparseRowFactor& factor,
    std::vector<double>& right_hand_side);
bool target_sparse_factor_backward_substitute(
    const TargetSparseRowFactor& factor,
    std::vector<double>& right_hand_side);
bool solve_target_sparse_row_factor(
    const TargetSparseRowFactor& factor,
    std::vector<double>& right_hand_side);
// Factor a dense lower-packed symmetric CSC fixture into the row-packed form
// consumed above.  The target first transposes that input to upper CSC, then
// uses the scalar LDL' row-factorization path recovered at
// 0x42DF100..0x42DF130: diagonal slots carry D and row i's later slots carry
// L(j,i). This is a raw native replay primitive, not the general production
// BA solver.
bool target_dense_ldlt_factorize_lower_csc(
    const SparseSymmetricMatrix& matrix,
    TargetSparseRowFactor& factor);
// Native fixture primitive for the target's observed single-supernode LL'
// factor shape. It materializes a lower-packed CSC block, applies LAPACK
// Cholesky, then performs two triangular solves. It is deliberately not wired
// into the general sparse fallback until the target's super-solve data layout
// and arithmetic are closed by a raw differential fixture.
bool target_dense_supernodal_llt_solve_lower_csc(
    const SparseSymmetricMatrix& matrix,
    std::vector<double>& right_hand_side);
// Raw native replay of the first observed real/double one-supernode solve.
// The panel is already factorized, column-major lower LL', with exactly n*n
// values and one RHS.  It retains the target's four BLAS calls, including the
// two m=0 GEMV calls and its one-double workspace.  Fixture-only until the
// general multi-supernode and multi-RHS contracts are independently captured.
bool target_single_supernode_llt_solve_one_rhs(
    const std::vector<double>& lower_panel,
    std::vector<double>& right_hand_side,
    double& workspace);
// Raw native replay of the observed real/double multi-supernode, one-RHS
// forward/backward workers. `super`, `pi`, `px`, and `indices` preserve the
// target CHOLMOD factor ABI; `values` is its supernodal panel storage.
// Fixture-only: multi-RHS and general production ownership are still open.
struct TargetSupernodalFactor {
    std::vector<std::size_t> super;
    std::vector<std::size_t> pi;
    std::vector<std::size_t> px;
    std::vector<std::size_t> indices;
    std::vector<double> values;
};

// One contribution segment consumed by the numeric supernodal scheduler. The
// offsets are relative to `pi[source_supernode]` and identify the source `s`
// suffix before and after processing `target_supernode`.
struct TargetSupernodalContribution {
    std::size_t source_supernode = 0;
    std::size_t target_supernode = 0;
    std::size_t source_offset = 0;
    std::size_t next_source_offset = 0;
};

// Native fixture replay of the Head/Next supernode scheduler observed in the
// target's 0x43591A0 numeric worker. It builds the per-column supernode map,
// advances each source's sorted `s` suffix with lower_bound(super[target+1]),
// and preserves the target's LIFO Head/Next relink order. Numerical BLAS and
// panel writes are intentionally outside this structural routine.
bool target_supernodal_numeric_schedule(
    const TargetSupernodalFactor& factor,
    std::vector<TargetSupernodalContribution>& contributions);

// Fixture-only conversion at the 0x42D2050 -> 0x43591A0 boundary. `permutation`
// has CHOLMOD's target meaning `permutation[new_index] = old_index`; this
// builds the sorted, packed lower CSC representation of P*A*P'. No duplicate
// coalescing is inferred or performed because the observed packed source has
// one lower entry per symmetric position.
bool target_permute_packed_lower_symmetric(
    const SparseSymmetricMatrix& source,
    const std::vector<std::size_t>& permutation,
    SparseSymmetricMatrix& result);

// Fixture-only packed/lower sparse-to-panel writer recovered from the target
// 0x4355DF0 stype=-1, sorted, packed branch. The caller controls panel
// zeroing, just as the target's separate preceding worker does.
bool target_supernodal_assemble_packed_lower_panel(
    const SparseSymmetricMatrix& sparse,
    TargetSupernodalFactor& factor,
    std::size_t supernode);

// Fixture-only transcription of one scheduled numeric contribution. It
// performs the observed dsyrk, conditional dgemm, compact row mapping and
// lower-panel subtraction in target order. `workspace` is the target's
// caller-provided C buffer (ldc * contribution width); if non-null,
// `compact_map` receives the full remaining source-suffix map.
bool target_supernodal_apply_contribution(
    TargetSupernodalFactor& factor,
    const TargetSupernodalContribution& contribution,
    std::vector<double>& workspace,
    std::vector<std::size_t>* compact_map = nullptr);

// Fixture-only target panel transition: dpotrf(L, nscol, lda=nsrow), followed
// by dtrsm(R,L,C,N) for a non-full-height panel. It does not provide general
// production factor ownership, symbolic construction or BA routing.
bool target_supernodal_factorize_panel(
    TargetSupernodalFactor& factor,
    std::size_t supernode);

bool target_supernodal_llt_solve_one_rhs(
    const TargetSupernodalFactor& factor,
    std::vector<double>& right_hand_side,
    std::vector<double>& workspace);
// Solve a symmetric positive-definite system using Metashape's recovered
// CHOLMOD layout: sorted/packed long-index CSC with only the lower triangle.
// Builds without CHOLMOD retain the dense LAPACK fallback.
bool solve_sparse_symmetric_system(
    const SparseSymmetricMatrix& matrix,
    std::vector<double>& rhs);
bool solve_sparse_symmetric_system(
    const std::vector<double>& row_major,
    std::vector<double>& rhs,
    std::size_t dimension);
bool solve_least_squares(
    const std::vector<double>& jacobian,
    const std::vector<double>& residual,
    std::size_t rows,
    std::size_t cols,
    double damping,
    std::vector<double>& update);
// Dense Gauss-Newton path recovered from Metashape sub_40B8CA0 ->
// sub_4107F90. Unlike the general LAPACK helper, this retains the target's
// row-major partial-pivot elimination and back-substitution order.
bool solve_least_squares_target_dense(
    const std::vector<double>& jacobian,
    const std::vector<double>& residual,
    std::size_t rows,
    std::size_t cols,
    double damping,
    std::vector<double>& update);

}  // namespace metalign
