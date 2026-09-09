#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "metashape_texture/image.hpp"

namespace metashape_texture::recovered {

// CPU translations of exact formulas recovered from Metashape 2.3.1 build
// 22416 embedded SPIR-V modules. Coordinates outside an image are treated the
// same way as the corresponding GPU module documented beside each function.

Image<float> heat_diffusion_step(const Image<float> &input,
                                 const Image<std::uint8_t> &constraint_mask);

Image<float> chamfer_distance_step(const Image<float> &distance,
                                   const Image<std::uint8_t> &mask,
                                   float maximum_distance);

Image<float> downscale_distance_2x(const Image<float> &input,
                                   const Image<std::uint8_t> &mask,
                                   Image<std::uint8_t> *output_mask);

Image<float> minimum_filter_square(const Image<float> &input, int radius);

Image<float> downscale_pyramid_6tap(const Image<float> &input);

Image<float> sharpen_cross(const Image<float> &input, float sharpening);

// Recovered host call at metashape.exe 0x141e89f10, used between the
// 2000-byte pack shader and the 1740-byte unpack shader.  Pixels unequal to
// background form 8-neighbour connected components; components whose size is
// strictly less than minimum_size are replaced by background.
Image<std::uint8_t> remove_small_components_8(
    const Image<std::uint8_t> &input,
    std::size_t minimum_size,
    std::uint8_t background = 0);

// Exact CPU translation of the 3168-byte module133 expression.
Image<float> initialize_camera_weights(
    const Image<std::int32_t> &face_ids,
    const std::vector<std::uint32_t> &face_labels,
    const Image<std::uint8_t> &occlusion_mask,
    std::uint32_t camera_label);

// Exact CPU translations of modules 144 and 236 (the embedded
// distances_to_weights shader).  Metashape dispatches module144
// floor(maximum_distance)+1 times per mode and stores every intermediate in
// R16F, so build_distance_weights includes binary16 rounding after each pass.
Image<float> distance_via_bfs(
    const Image<std::int32_t> &constraint,
    const Image<float> &camera_value,
    int mode,
    float maximum_distance);

Image<float> distances_to_weights(
    const Image<float> &boundary_distance,
    const Image<float> &winner_distance,
    const Image<float> &nonwinner_distance,
    const Image<float> &previous_value,
    float maximum_distance);

Image<float> build_distance_weights(
    const Image<std::int32_t> &constraint,
    const Image<float> &camera_value,
    float maximum_distance);

// Exact host-readable translation of
// update_faces_resolution_with_proj_area.comp.  Inputs equal to 0 or -1 are
// skipped; all other values update the per-face greatest and second-greatest
// arrays with the shader's ordered >= comparisons.
void update_face_resolution_max2(
    const std::vector<float> &input,
    std::vector<float> &maximum,
    std::vector<float> &second_greatest);

std::vector<float> second_greatest_face_resolution(
    const std::vector<std::vector<float>> &camera_resolution);

// Exact default-Frame Record20 membership predicate recovered by exhaustive
// comparison and a causal replacement of the 4072-byte gather shader.  Both
// positive resolution and the -1 partial marker are candidates; either signed
// representation of zero is absent.
bool candidate_exists_from_scale4_resolution(float resolution) noexcept;

std::uint16_t quantize_weight(float weight);

}  // namespace metashape_texture::recovered
