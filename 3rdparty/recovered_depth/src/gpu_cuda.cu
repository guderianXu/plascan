#include "metmodel/gpu.hpp"
#include "metmodel/octree_prepare.hpp"
#include "metmodel/patchmatch.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace metmodel {
namespace {

__global__ void cost_kernel(const float* reference, const float* samples, float* costs,
                            std::size_t count, std::size_t hypotheses,
                            std::size_t neighbors) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t pixel = index / hypotheses;
    float sum = 0.0F;
    unsigned int valid = 0;
    for (std::size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
        const float sample = samples[index * neighbors + neighbor];
        if (isfinite(sample)) { sum += fabsf(reference[pixel] - sample); ++valid; }
    }
    costs[index] = valid == 0 ? __int_as_float(0x7f800000) : sum / static_cast<float>(valid);
}

class CUDAContext final : public ComputeContext {
public:
    explicit CUDAContext(std::size_t index, const cudaDeviceProp& properties,
                         std::size_t memory_guard)
        : memory_guard_(memory_guard) {
        device_.backend = ComputeBackend::CUDA;
        device_.index = index;
        device_.name = properties.name;
        device_.memory_bytes = properties.totalGlobalMem;
        cudaSetDevice(static_cast<int>(index));
    }
    const ComputeDeviceInfo& device() const override { return device_; }
    bool compute_cost_volume(const CostVolumeInput& input, std::vector<float>& costs,
                             std::string& error) override {
        const std::size_t count = input.pixels * input.hypotheses;
        const std::size_t reference_bytes = input.pixels * sizeof(float);
        const std::size_t sample_bytes = count * input.neighbors * sizeof(float);
        const std::size_t cost_bytes = count * sizeof(float);
        std::size_t free_bytes = 0, total_bytes = 0;
        const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (memory_status != cudaSuccess) {
            error = std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(memory_status);
            return false;
        }
        const std::size_t required = reference_bytes + sample_bytes + cost_bytes + memory_guard_;
        if (required > free_bytes) {
            error = "insufficient device memory after guard (required " +
                std::to_string(required / (1024ULL * 1024ULL)) + " MiB, free " +
                std::to_string(free_bytes / (1024ULL * 1024ULL)) + " MiB)";
            return false;
        }
        float* reference = nullptr; float* samples = nullptr; float* output = nullptr;
        auto release = [&]() { cudaFree(reference); cudaFree(samples); cudaFree(output); };
        cudaError_t status = cudaMalloc(&reference, reference_bytes);
        if (status == cudaSuccess) status = cudaMalloc(&samples, sample_bytes);
        if (status == cudaSuccess) status = cudaMalloc(&output, cost_bytes);
        if (status == cudaSuccess) status = cudaMemcpy(reference, input.reference->data(), reference_bytes, cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(samples, input.samples->data(), sample_bytes, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) { error = cudaGetErrorString(status); release(); return false; }
        constexpr std::size_t threads = 256;
        cost_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads>>>(
            reference, samples, output, count, input.hypotheses, input.neighbors);
        status = cudaGetLastError();
        costs.resize(count);
        if (status == cudaSuccess) status = cudaMemcpy(costs.data(), output, cost_bytes, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) { error = cudaGetErrorString(status); release(); return false; }
        release();
        return true;
    }
private:
    ComputeDeviceInfo device_;
    std::size_t memory_guard_ = 0;
};

struct RecoveredCudaCachedModule {
    CUmodule module = nullptr;
    std::unordered_map<std::string, CUfunction> functions;
};

struct RecoveredCudaCachedAllocation {
    void* pointer = nullptr;
    std::size_t capacity = 0;
};

struct RecoveredCudaCostWorkspace {
    // 13 PatchMatch pipeline roles plus the filter chain's estimated-normal,
    // mask and ten physical counters.  The two clear-depth stages reuse the
    // same counter slot, matching the captured target lifetime.
    static constexpr std::size_t fixed_allocation_count = 25U;
    bool in_use = false;
    std::array<RecoveredCudaCachedAllocation, fixed_allocation_count> fixed;
    std::vector<RecoveredCudaCachedAllocation> offset_groups;
    std::vector<RecoveredCudaCachedAllocation> mask_groups;
    // Immutable neighbour mip sources live in linear device memory for the
    // whole reference-camera generation.  The target copies five rectangles
    // device-to-device from these grouped sources into its writable atlas.
    std::vector<RecoveredCudaCachedAllocation> texture_sources;
    std::vector<std::uint64_t> texture_source_generations;
    CUarray texture_array = nullptr;
    CUtexObject texture = 0;
    std::uint32_t texture_width = 0;
    std::uint32_t texture_height = 0;
    std::uint64_t handoff_generation = 0;
    std::uint32_t handoff_stage = 0;
    bool handoff_winner_valid = false;
    bool handoff_inlier_masks_valid = false;
    // The offset tables and packed rejection masks are immutable for one
    // reference-camera generation.  They are uploaded by its first cost
    // batch and then retained across every level/binding of that camera.
    std::uint64_t resource_generation = 0;
};

struct RecoveredCudaVotingWorkspace {
    // reference depth[3], radius[3], votes[3], neighbor depth/radius,
    // inlier mask[3], direct counters[7], occlusion counters[2], final[4].
    static constexpr std::size_t fixed_allocation_count = 27U;
    bool in_use = false;
    std::array<RecoveredCudaCachedAllocation, fixed_allocation_count> fixed;
};

struct RecoveredCudaVotingDepthCacheEntry {
    void* device_pointer = nullptr;
    std::size_t bytes = 0U;
};

struct RecoveredCudaModuleSessionState {
    bool active = false;
    std::size_t nesting = 0;
    CUcontext context = nullptr;
    std::map<std::string, RecoveredCudaCachedModule> modules;
    // Metashape schedules two camera workers.  A worker owns its writable
    // atlas, scratch and counters for the whole camera; module/function
    // objects remain shared by the CUDA context.
    std::map<std::thread::id, RecoveredCudaCostWorkspace> cost_workspaces;
    std::map<std::thread::id, CUstream> worker_streams;
    std::uint64_t next_handoff_generation = 0U;
    std::map<std::thread::id, RecoveredCudaVotingWorkspace> voting_workspaces;
    std::map<const float*, RecoveredCudaVotingDepthCacheEntry>
        voting_depth_cache;
    RecoveredCudaModuleSessionStats stats;
};

std::mutex recovered_cuda_module_session_mutex;
RecoveredCudaModuleSessionState recovered_cuda_module_session;

RecoveredCudaCostWorkspace& current_recovered_cuda_cost_workspace_locked(
    RecoveredCudaModuleSessionState& session) {
    return session.cost_workspaces[std::this_thread::get_id()];
}

const RecoveredCudaCostWorkspace*
find_current_recovered_cuda_cost_workspace_locked(
    const RecoveredCudaModuleSessionState& session) {
    const auto found = session.cost_workspaces.find(std::this_thread::get_id());
    return found == session.cost_workspaces.end() ? nullptr : &found->second;
}

RecoveredCudaCostWorkspace* find_current_recovered_cuda_cost_workspace_locked(
    RecoveredCudaModuleSessionState& session) {
    const auto found = session.cost_workspaces.find(std::this_thread::get_id());
    return found == session.cost_workspaces.end() ? nullptr : &found->second;
}

enum class RecoveredCudaTransferPhase {
    uncategorized,
    undistort,
    producer,
    cost,
    wta,
    inlier,
    coarse_to_precise,
    bilateral,
    filter,
    voting,
};

thread_local RecoveredCudaTransferPhase recovered_cuda_transfer_phase =
    RecoveredCudaTransferPhase::uncategorized;

class RecoveredCudaTransferPhaseScope {
public:
    explicit RecoveredCudaTransferPhaseScope(RecoveredCudaTransferPhase phase)
        : previous_(recovered_cuda_transfer_phase) {
        recovered_cuda_transfer_phase = phase;
    }
    ~RecoveredCudaTransferPhaseScope() {
        recovered_cuda_transfer_phase = previous_;
    }
    RecoveredCudaTransferPhaseScope(const RecoveredCudaTransferPhaseScope&) = delete;
    RecoveredCudaTransferPhaseScope& operator=(
        const RecoveredCudaTransferPhaseScope&) = delete;

private:
    RecoveredCudaTransferPhase previous_;
};

void record_recovered_cuda_h2d_phase(
    RecoveredCudaModuleSessionStats& stats,
    std::uint64_t bytes) {
    std::uint64_t* calls = nullptr;
    std::uint64_t* phase_bytes = nullptr;
    switch (recovered_cuda_transfer_phase) {
        case RecoveredCudaTransferPhase::undistort:
            calls = &stats.undistort_h2d_calls;
            phase_bytes = &stats.undistort_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::producer:
            calls = &stats.producer_h2d_calls;
            phase_bytes = &stats.producer_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::cost:
            calls = &stats.cost_h2d_calls;
            phase_bytes = &stats.cost_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::wta:
            calls = &stats.wta_h2d_calls;
            phase_bytes = &stats.wta_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::inlier:
            calls = &stats.inlier_h2d_calls;
            phase_bytes = &stats.inlier_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::coarse_to_precise:
            calls = &stats.coarse_to_precise_h2d_calls;
            phase_bytes = &stats.coarse_to_precise_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::bilateral:
            calls = &stats.bilateral_h2d_calls;
            phase_bytes = &stats.bilateral_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::filter:
            calls = &stats.filter_h2d_calls;
            phase_bytes = &stats.filter_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::voting:
            calls = &stats.voting_h2d_calls;
            phase_bytes = &stats.voting_h2d_bytes;
            break;
        case RecoveredCudaTransferPhase::uncategorized:
            calls = &stats.uncategorized_h2d_calls;
            phase_bytes = &stats.uncategorized_h2d_bytes;
            break;
    }
    ++*calls;
    *phase_bytes += bytes;
}

void record_recovered_cuda_d2h_phase(
    RecoveredCudaModuleSessionStats& stats,
    std::uint64_t bytes) {
    std::uint64_t* calls = nullptr;
    std::uint64_t* phase_bytes = nullptr;
    switch (recovered_cuda_transfer_phase) {
        case RecoveredCudaTransferPhase::undistort:
            calls = &stats.undistort_d2h_calls;
            phase_bytes = &stats.undistort_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::producer:
            calls = &stats.producer_d2h_calls;
            phase_bytes = &stats.producer_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::cost:
            calls = &stats.cost_d2h_calls;
            phase_bytes = &stats.cost_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::wta:
            calls = &stats.wta_d2h_calls;
            phase_bytes = &stats.wta_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::inlier:
            calls = &stats.inlier_d2h_calls;
            phase_bytes = &stats.inlier_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::coarse_to_precise:
            calls = &stats.coarse_to_precise_d2h_calls;
            phase_bytes = &stats.coarse_to_precise_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::bilateral:
            calls = &stats.bilateral_d2h_calls;
            phase_bytes = &stats.bilateral_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::filter:
            calls = &stats.filter_d2h_calls;
            phase_bytes = &stats.filter_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::voting:
            calls = &stats.voting_d2h_calls;
            phase_bytes = &stats.voting_d2h_bytes;
            break;
        case RecoveredCudaTransferPhase::uncategorized:
            calls = &stats.uncategorized_d2h_calls;
            phase_bytes = &stats.uncategorized_d2h_bytes;
            break;
    }
    ++*calls;
    *phase_bytes += bytes;
}

struct RecoveredCudaCostWorkspaceHandles {
    std::array<void*, RecoveredCudaCostWorkspace::fixed_allocation_count> fixed{};
    std::vector<void*> offset_groups;
    std::vector<void*> mask_groups;
    std::vector<void*> texture_sources;
    RecoveredCudaCostWorkspace* workspace = nullptr;
    CUarray texture_array = nullptr;
    CUtexObject texture = 0;
    bool grouped_resources_resident = false;
};

enum RecoveredCudaCostWorkspaceSlot : std::size_t {
    cost_depth_slot = 0,
    cost_normal_slot,
    cost_main_cost_slot,
    cost_coarse_depth_slot,
    cost_coarse_radius_slot,
    cost_reference_slot,
    cost_candidate_depth_slot,
    cost_candidate_normal_slot,
    cost_neighbor_cost_slot,
    cost_average_slot,
    cost_auxiliary_slot,
    pipeline_winner_slot,
    pipeline_inlier_masks_slot,
    filter_estimated_normal_slot,
    filter_mask_slot,
    filter_counter_no_cost_slot,
    filter_counter_big_cost_slot,
    filter_counter_no_neighbours_slot,
    filter_counter_no_close_neighbours_slot,
    filter_counter_clear_slot,
    filter_counter_inconsistent_normal_slot,
    filter_counter_bad_estimated_normal_slot,
    filter_counter_bad_found_normal_slot,
    filter_counter_cos_sum_slot,
    filter_counter_ncos_sum_slot,
};

enum RecoveredCudaVotingWorkspaceSlot : std::size_t {
    voting_reference_depth_0_slot = 0,
    voting_reference_depth_1_slot,
    voting_reference_depth_2_slot,
    voting_reference_radius_0_slot,
    voting_reference_radius_1_slot,
    voting_reference_radius_2_slot,
    voting_reference_votes_0_slot,
    voting_reference_votes_1_slot,
    voting_reference_votes_2_slot,
    voting_neighbor_depth_slot,
    voting_neighbor_radius_slot,
    voting_neighbor_mask_0_slot,
    voting_neighbor_mask_1_slot,
    voting_neighbor_mask_2_slot,
    voting_direct_counter_0_slot,
    voting_direct_counter_1_slot,
    voting_direct_counter_2_slot,
    voting_direct_counter_3_slot,
    voting_direct_counter_4_slot,
    voting_direct_counter_5_slot,
    voting_direct_counter_6_slot,
    voting_occlusion_counter_0_slot,
    voting_occlusion_counter_1_slot,
    voting_final_counter_0_slot,
    voting_final_counter_1_slot,
    voting_final_counter_2_slot,
    voting_final_counter_3_slot,
};

struct RecoveredCudaVotingWorkspaceHandles {
    std::array<void*, RecoveredCudaVotingWorkspace::fixed_allocation_count> fixed{};
};

std::uint64_t recovered_cuda_cost_workspace_bytes(
    const RecoveredCudaCostWorkspace& workspace) {
    std::uint64_t bytes = 0;
    for (const auto& allocation : workspace.fixed) bytes += allocation.capacity;
    for (const auto& allocation : workspace.offset_groups) bytes += allocation.capacity;
    for (const auto& allocation : workspace.mask_groups) bytes += allocation.capacity;
    for (const auto& allocation : workspace.texture_sources)
        bytes += allocation.capacity;
    if (workspace.texture_array != nullptr)
        bytes += static_cast<std::uint64_t>(workspace.texture_width) *
                 workspace.texture_height;
    return bytes;
}

bool recovered_cuda_ensure_cost_allocation_locked(
    RecoveredCudaCachedAllocation& allocation,
    std::size_t bytes,
    RecoveredCudaModuleSessionStats& stats,
    std::string& error) {
    if (bytes == 0U) return true;
    if (allocation.pointer != nullptr && allocation.capacity >= bytes) {
        ++stats.cost_workspace_reused_buffers;
        return true;
    }
    void* replacement = nullptr;
    const cudaError_t allocate_status = ::cudaMalloc(&replacement, bytes);
    ++stats.cuda_malloc_calls;
    stats.cuda_malloc_bytes += bytes;
    ++stats.cost_workspace_physical_allocations;
    if (allocate_status != cudaSuccess) {
        error = std::string("growing recovered CUDA cost workspace failed: ") +
                cudaGetErrorString(allocate_status);
        return false;
    }
    if (allocation.pointer != nullptr) {
        const cudaError_t free_status = ::cudaFree(allocation.pointer);
        ++stats.cuda_free_calls;
        if (free_status != cudaSuccess) {
            ::cudaFree(replacement);
            ++stats.cuda_free_calls;
            error = std::string("releasing superseded recovered CUDA cost workspace buffer failed: ") +
                    cudaGetErrorString(free_status);
            return false;
        }
    }
    allocation.pointer = replacement;
    allocation.capacity = bytes;
    return true;
}

bool acquire_recovered_cuda_cost_workspace(
    const std::array<std::size_t, RecoveredCudaCostWorkspace::fixed_allocation_count>&
        fixed_bytes,
    const std::vector<std::size_t>& offset_group_bytes,
    const std::vector<std::size_t>& mask_group_bytes,
    std::uint32_t texture_width,
    std::uint32_t texture_height,
    RecoveredCudaCostWorkspaceHandles& handles,
    bool& acquired,
    std::string& error,
    std::uint64_t required_handoff_generation = 0U,
    std::uint32_t required_handoff_stage = 0U,
    std::uint64_t required_resource_generation = 0U) {
    acquired = false;
    CUcontext context = nullptr;
    const CUresult context_status = ::cuCtxGetCurrent(&context);
    if (context_status != CUDA_SUCCESS) return true;

    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active) return true;
    if (context == nullptr || context != session.context) {
        error = "recovered CUDA cost workspace context does not match its session";
        return false;
    }
    auto& workspace = current_recovered_cuda_cost_workspace_locked(session);
    if (workspace.in_use) {
        error = "recovered CUDA cost workspace does not permit concurrent use";
        return false;
    }
    if (required_handoff_generation != 0U) {
        if (workspace.handoff_generation != required_handoff_generation ||
            workspace.handoff_stage != required_handoff_stage) {
            error = "recovered CUDA producer-to-cost handoff is stale or out of order: expected generation=" +
                    std::to_string(required_handoff_generation) +
                    " stage=" + std::to_string(required_handoff_stage) +
                    ", actual generation=" +
                    std::to_string(workspace.handoff_generation) +
                    " stage=" + std::to_string(workspace.handoff_stage);
            return false;
        }
        constexpr std::array<std::size_t, 6> resident_slots = {
            cost_depth_slot,
            cost_normal_slot,
            cost_main_cost_slot,
            cost_reference_slot,
            cost_candidate_depth_slot,
            cost_candidate_normal_slot,
        };
        for (const std::size_t slot : resident_slots) {
            if (workspace.fixed[slot].pointer == nullptr ||
                workspace.fixed[slot].capacity < fixed_bytes[slot]) {
                error = "recovered CUDA producer-to-cost resident buffer is undersized";
                return false;
            }
        }
    }
    workspace.in_use = true;
    acquired = true;
    ++session.stats.cost_workspace_acquires;

    auto fail = [&]() {
        workspace.in_use = false;
        acquired = false;
        return false;
    };
    for (std::size_t index = 0; index < fixed_bytes.size(); ++index) {
        if (!recovered_cuda_ensure_cost_allocation_locked(
                workspace.fixed[index], fixed_bytes[index], session.stats,
                error))
            return fail();
    }
    workspace.offset_groups.resize(offset_group_bytes.size());
    workspace.mask_groups.resize(mask_group_bytes.size());
    for (std::size_t group = 0; group < offset_group_bytes.size(); ++group) {
        if (!recovered_cuda_ensure_cost_allocation_locked(
                workspace.offset_groups[group], offset_group_bytes[group],
                session.stats, error) ||
            !recovered_cuda_ensure_cost_allocation_locked(
                workspace.mask_groups[group], mask_group_bytes[group],
                session.stats, error))
            return fail();
    }

    if (workspace.texture_array == nullptr ||
        workspace.texture_width != texture_width ||
        workspace.texture_height != texture_height) {
        if (workspace.texture != 0) {
            const CUresult status = ::cuTexObjectDestroy(workspace.texture);
            ++session.stats.texture_destroy_calls;
            workspace.texture = 0;
            if (status != CUDA_SUCCESS) {
                error = "destroying resized recovered CUDA cost texture failed";
                return fail();
            }
        }
        if (workspace.texture_array != nullptr) {
            const CUresult status = ::cuArrayDestroy(workspace.texture_array);
            ++session.stats.array_destroy_calls;
            workspace.texture_array = nullptr;
            if (status != CUDA_SUCCESS) {
                error = "destroying resized recovered CUDA cost array failed";
                return fail();
            }
        }
        CUDA_ARRAY3D_DESCRIPTOR array_description{};
        array_description.Width = texture_width;
        array_description.Height = texture_height;
        array_description.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        array_description.NumChannels = 1;
        array_description.Flags = CUDA_ARRAY3D_SURFACE_LDST;
        CUresult status = ::cuArray3DCreate(
            &workspace.texture_array, &array_description);
        ++session.stats.array_create_calls;
        if (status != CUDA_SUCCESS) {
            error = "creating recovered CUDA cost workspace array failed";
            return fail();
        }
        CUDA_RESOURCE_DESC resource{};
        resource.resType = CU_RESOURCE_TYPE_ARRAY;
        resource.res.array.hArray = workspace.texture_array;
        CUDA_TEXTURE_DESC texture_description{};
        texture_description.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_description.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_description.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
        texture_description.filterMode = CU_TR_FILTER_MODE_LINEAR;
        status = ::cuTexObjectCreate(
            &workspace.texture, &resource, &texture_description, nullptr);
        ++session.stats.texture_create_calls;
        if (status != CUDA_SUCCESS) {
            error = "creating recovered CUDA cost workspace texture failed";
            return fail();
        }
        workspace.texture_width = texture_width;
        workspace.texture_height = texture_height;
    }

    for (std::size_t index = 0; index < workspace.fixed.size(); ++index)
        handles.fixed[index] = workspace.fixed[index].pointer;
    handles.offset_groups.resize(offset_group_bytes.size());
    handles.mask_groups.resize(mask_group_bytes.size());
    for (std::size_t group = 0; group < offset_group_bytes.size(); ++group) {
        handles.offset_groups[group] = workspace.offset_groups[group].pointer;
        handles.mask_groups[group] = workspace.mask_groups[group].pointer;
    }
    handles.texture_array = workspace.texture_array;
    handles.texture = workspace.texture;
    handles.workspace = &workspace;
    handles.grouped_resources_resident =
        required_resource_generation != 0U &&
        workspace.resource_generation == required_resource_generation;
    std::uint64_t aggregate_workspace_bytes = 0U;
    for (const auto& [thread, current] : session.cost_workspaces) {
        (void)thread;
        aggregate_workspace_bytes += recovered_cuda_cost_workspace_bytes(current);
    }
    session.stats.cost_workspace_peak_bytes = std::max(
        session.stats.cost_workspace_peak_bytes, aggregate_workspace_bytes);
    error.clear();
    return true;
}

bool ensure_recovered_cuda_cost_texture_source(
    std::size_t source_count,
    std::size_t source_index,
    std::size_t source_bytes,
    RecoveredCudaCostWorkspaceHandles& handles,
    std::string& error) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    auto* workspace = session.active
        ? find_current_recovered_cuda_cost_workspace_locked(session)
        : nullptr;
    if (workspace == nullptr || !workspace->in_use ||
        handles.workspace != workspace || source_count == 0U ||
        source_index >= source_count || source_bytes == 0U) {
        error = "recovered CUDA texture-source workspace is unavailable";
        return false;
    }
    if (workspace->texture_sources.size() < source_count) {
        workspace->texture_sources.resize(source_count);
        workspace->texture_source_generations.resize(source_count, 0U);
    }
    handles.texture_sources.resize(source_count, nullptr);
    void* previous = workspace->texture_sources[source_index].pointer;
    if (!recovered_cuda_ensure_cost_allocation_locked(
            workspace->texture_sources[source_index], source_bytes,
            session.stats, error))
        return false;
    if (workspace->texture_sources[source_index].pointer != previous)
        workspace->texture_source_generations[source_index] = 0U;
    handles.texture_sources[source_index] =
        workspace->texture_sources[source_index].pointer;
    std::uint64_t aggregate_workspace_bytes = 0U;
    for (const auto& [thread, current] : session.cost_workspaces) {
        (void)thread;
        aggregate_workspace_bytes += recovered_cuda_cost_workspace_bytes(current);
    }
    session.stats.cost_workspace_peak_bytes = std::max(
        session.stats.cost_workspace_peak_bytes, aggregate_workspace_bytes);
    error.clear();
    return true;
}

CUresult acquire_recovered_cuda_worker_stream(CUstream& stream) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active) {
        stream = nullptr;
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUstream& worker_stream =
        session.worker_streams[std::this_thread::get_id()];
    if (worker_stream == nullptr) {
        const CUresult status =
            ::cuStreamCreate(&worker_stream, CU_STREAM_DEFAULT);
        if (status != CUDA_SUCCESS) return status;
    }
    stream = worker_stream;
    return CUDA_SUCCESS;
}

bool mark_recovered_cuda_cost_resources_resident(
    std::uint64_t resource_generation,
    std::uint64_t handoff_generation) {
    if (resource_generation == 0U || handoff_generation == 0U) return false;
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    auto* workspace = session.active
        ? find_current_recovered_cuda_cost_workspace_locked(session)
        : nullptr;
    if (workspace == nullptr ||
        workspace->handoff_generation != handoff_generation)
        return false;
    workspace->resource_generation = resource_generation;
    return true;
}

void release_recovered_cuda_cost_workspace_lease() {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    if (!recovered_cuda_module_session.active) return;
    auto* workspace = find_current_recovered_cuda_cost_workspace_locked(
        recovered_cuda_module_session);
    if (workspace != nullptr) workspace->in_use = false;
}

std::uint64_t publish_recovered_cuda_workspace_handoff(
    std::uint32_t required_stage,
    std::uint64_t required_generation,
    std::uint32_t next_stage) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active) return 0U;
    auto& workspace = current_recovered_cuda_cost_workspace_locked(session);
    if (required_stage != 0U &&
        (workspace.handoff_stage != required_stage ||
         workspace.handoff_generation != required_generation))
        return 0U;
    if (required_stage == 0U) {
        ++session.next_handoff_generation;
        if (session.next_handoff_generation == 0U)
            ++session.next_handoff_generation;
        workspace.handoff_generation = session.next_handoff_generation;
        workspace.handoff_winner_valid = false;
        workspace.handoff_inlier_masks_valid = false;
    }
    workspace.handoff_stage = next_stage;
    if (next_stage == 2U) workspace.handoff_winner_valid = true;
    if (next_stage == 4U) workspace.handoff_inlier_masks_valid = true;
    return workspace.handoff_generation;
}

bool recovered_cuda_workspace_handoff_winner_is_valid(
    std::uint64_t generation,
    std::uint32_t required_stage) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    const auto& session = recovered_cuda_module_session;
    const auto* workspace = session.active
        ? find_current_recovered_cuda_cost_workspace_locked(session)
        : nullptr;
    return generation != 0U && workspace != nullptr &&
           workspace->handoff_generation == generation &&
           workspace->handoff_stage == required_stage &&
           workspace->handoff_winner_valid;
}

bool recovered_cuda_workspace_handoff_inlier_masks_are_valid(
    std::uint64_t generation,
    std::uint32_t required_stage) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    const auto& session = recovered_cuda_module_session;
    const auto* workspace = session.active
        ? find_current_recovered_cuda_cost_workspace_locked(session)
        : nullptr;
    return generation != 0U && workspace != nullptr &&
           workspace->handoff_generation == generation &&
           workspace->handoff_stage == required_stage &&
           workspace->handoff_inlier_masks_valid;
}

bool validate_recovered_cuda_workspace_handoff(
    std::uint64_t generation,
    std::uint32_t required_stage,
    std::string& error) {
    if (generation == 0U) return false;
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    const auto& session = recovered_cuda_module_session;
    const auto* workspace = session.active
        ? find_current_recovered_cuda_cost_workspace_locked(session)
        : nullptr;
    if (workspace == nullptr ||
        workspace->handoff_generation != generation ||
        workspace->handoff_stage != required_stage) {
        error = "recovered CUDA workspace handoff token is stale or out of order";
        return false;
    }
    return true;
}

std::uint32_t recovered_cuda_workspace_handoff_stage(
    std::uint64_t generation) {
    if (generation == 0U) return 0U;
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    const auto& session = recovered_cuda_module_session;
    const auto* workspace = session.active
        ? find_current_recovered_cuda_cost_workspace_locked(session)
        : nullptr;
    if (workspace == nullptr || workspace->handoff_generation != generation)
        return 0U;
    return workspace->handoff_stage;
}

void invalidate_recovered_cuda_workspace_handoff(std::uint64_t generation) {
    if (generation == 0U) return;
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    auto* workspace = session.active
        ? find_current_recovered_cuda_cost_workspace_locked(session)
        : nullptr;
    if (workspace != nullptr && workspace->handoff_generation == generation) {
        workspace->handoff_stage = 0U;
        workspace->handoff_winner_valid = false;
        workspace->handoff_inlier_masks_valid = false;
    }
}

void record_recovered_cuda_workspace_skipped_h2d(
    std::uint64_t calls,
    std::uint64_t bytes) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    if (!recovered_cuda_module_session.active) return;
    auto& stats = recovered_cuda_module_session.stats;
    stats.workspace_handoff_skipped_h2d_calls += calls;
    stats.workspace_handoff_skipped_h2d_bytes += bytes;
}

void record_recovered_cuda_workspace_skipped_d2h(
    std::uint64_t calls,
    std::uint64_t bytes) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    if (!recovered_cuda_module_session.active) return;
    auto& stats = recovered_cuda_module_session.stats;
    stats.workspace_handoff_skipped_d2h_calls += calls;
    stats.workspace_handoff_skipped_d2h_bytes += bytes;
}

enum class RecoveredCudaCostH2dKind {
    main_state,
    group_resources,
    scratch,
    texture_source,
};

void record_recovered_cuda_cost_h2d_detail(
    RecoveredCudaCostH2dKind kind,
    std::uint64_t bytes) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    if (!recovered_cuda_module_session.active) return;
    auto& stats = recovered_cuda_module_session.stats;
    switch (kind) {
        case RecoveredCudaCostH2dKind::main_state:
            ++stats.cost_main_state_h2d_calls;
            stats.cost_main_state_h2d_bytes += bytes;
            break;
        case RecoveredCudaCostH2dKind::group_resources:
            ++stats.cost_group_resources_h2d_calls;
            stats.cost_group_resources_h2d_bytes += bytes;
            break;
        case RecoveredCudaCostH2dKind::scratch:
            ++stats.cost_scratch_h2d_calls;
            stats.cost_scratch_h2d_bytes += bytes;
            break;
        case RecoveredCudaCostH2dKind::texture_source:
            ++stats.cost_texture_sources_h2d_calls;
            stats.cost_texture_sources_h2d_bytes += bytes;
            break;
    }
}

enum class RecoveredCudaWorkspaceClient {
    producer,
    wta,
    inlier,
    coarse_to_precise,
    filter,
};

bool acquire_recovered_cuda_pipeline_fixed_workspace(
    const std::array<std::size_t, RecoveredCudaCostWorkspace::fixed_allocation_count>&
        fixed_bytes,
    RecoveredCudaWorkspaceClient client,
    RecoveredCudaCostWorkspaceHandles& handles,
    bool& acquired,
    std::string& error,
    std::uint64_t required_handoff_generation = 0U,
    std::uint32_t required_handoff_stage = 0U) {
    acquired = false;
    CUcontext context = nullptr;
    const CUresult context_status = ::cuCtxGetCurrent(&context);
    if (context_status != CUDA_SUCCESS) return true;

    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active) return true;
    if (context == nullptr || context != session.context) {
        error = "recovered CUDA pipeline workspace context does not match its session";
        return false;
    }
    auto& workspace = current_recovered_cuda_cost_workspace_locked(session);
    if (workspace.in_use) {
        error = "recovered CUDA pipeline workspace does not permit concurrent use";
        return false;
    }
    if (required_handoff_generation != 0U) {
        if (workspace.handoff_generation != required_handoff_generation ||
            workspace.handoff_stage != required_handoff_stage) {
            error = "recovered CUDA pipeline workspace handoff is stale or out of order";
            return false;
        }
        for (std::size_t index = 0; index < fixed_bytes.size(); ++index) {
            if (fixed_bytes[index] != 0U &&
                (workspace.fixed[index].pointer == nullptr ||
                 workspace.fixed[index].capacity < fixed_bytes[index])) {
                error = "recovered CUDA pipeline resident buffer is undersized";
                return false;
            }
        }
    }
    workspace.in_use = true;
    acquired = true;
    switch (client) {
        case RecoveredCudaWorkspaceClient::producer:
            ++session.stats.producer_workspace_acquires;
            break;
        case RecoveredCudaWorkspaceClient::wta:
            ++session.stats.wta_workspace_acquires;
            break;
        case RecoveredCudaWorkspaceClient::inlier:
            ++session.stats.inlier_workspace_acquires;
            break;
        case RecoveredCudaWorkspaceClient::coarse_to_precise:
            ++session.stats.coarse_to_precise_workspace_acquires;
            break;
        case RecoveredCudaWorkspaceClient::filter:
            ++session.stats.filter_workspace_acquires;
            break;
    }
    auto fail = [&]() {
        workspace.in_use = false;
        acquired = false;
        return false;
    };
    for (std::size_t index = 0; index < fixed_bytes.size(); ++index) {
        if (!recovered_cuda_ensure_cost_allocation_locked(
                workspace.fixed[index], fixed_bytes[index], session.stats,
                error))
            return fail();
        handles.fixed[index] = workspace.fixed[index].pointer;
    }
    std::uint64_t aggregate_workspace_bytes = 0U;
    for (const auto& [thread, current] : session.cost_workspaces) {
        (void)thread;
        aggregate_workspace_bytes += recovered_cuda_cost_workspace_bytes(current);
    }
    session.stats.cost_workspace_peak_bytes = std::max(
        session.stats.cost_workspace_peak_bytes, aggregate_workspace_bytes);
    error.clear();
    return true;
}

CUresult destroy_recovered_cuda_cost_workspace_locked(
    RecoveredCudaModuleSessionState& session) {
    CUresult first_failure = CUDA_SUCCESS;
    for (auto& [thread, workspace] : session.cost_workspaces) {
        (void)thread;
        if (workspace.texture != 0) {
            const CUresult status = ::cuTexObjectDestroy(workspace.texture);
            ++session.stats.texture_destroy_calls;
            if (status != CUDA_SUCCESS && first_failure == CUDA_SUCCESS)
                first_failure = status;
        }
        if (workspace.texture_array != nullptr) {
            const CUresult status = ::cuArrayDestroy(workspace.texture_array);
            ++session.stats.array_destroy_calls;
            if (status != CUDA_SUCCESS && first_failure == CUDA_SUCCESS)
                first_failure = status;
        }
        auto free_allocation = [&](RecoveredCudaCachedAllocation& allocation) {
            if (allocation.pointer == nullptr) return;
            const cudaError_t status = ::cudaFree(allocation.pointer);
            ++session.stats.cuda_free_calls;
            if (status != cudaSuccess && first_failure == CUDA_SUCCESS)
                first_failure = CUDA_ERROR_UNKNOWN;
            allocation = {};
        };
        for (auto& allocation : workspace.fixed) free_allocation(allocation);
        for (auto& allocation : workspace.offset_groups) free_allocation(allocation);
        for (auto& allocation : workspace.mask_groups) free_allocation(allocation);
        for (auto& allocation : workspace.texture_sources)
            free_allocation(allocation);
        workspace = {};
    }
    session.cost_workspaces.clear();
    return first_failure;
}

std::uint64_t recovered_cuda_voting_workspace_bytes(
    const RecoveredCudaVotingWorkspace& workspace) {
    std::uint64_t bytes = 0;
    for (const auto& allocation : workspace.fixed) bytes += allocation.capacity;
    return bytes;
}

bool acquire_recovered_cuda_voting_workspace(
    const std::array<std::size_t,
                     RecoveredCudaVotingWorkspace::fixed_allocation_count>& bytes,
    RecoveredCudaVotingWorkspaceHandles& handles,
    bool& acquired,
    std::string& error) {
    acquired = false;
    CUcontext context = nullptr;
    const CUresult context_status = ::cuCtxGetCurrent(&context);
    if (context_status != CUDA_SUCCESS) return true;

    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active) return true;
    if (context == nullptr || context != session.context) {
        error = "recovered CUDA voting workspace context does not match its session";
        return false;
    }
    auto& workspace = session.voting_workspaces[std::this_thread::get_id()];
    if (workspace.in_use) {
        error = "recovered CUDA voting workspace does not permit concurrent use";
        return false;
    }
    workspace.in_use = true;
    acquired = true;
    ++session.stats.voting_workspace_acquires;
    auto fail = [&]() {
        workspace.in_use = false;
        acquired = false;
        return false;
    };
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        auto& allocation = workspace.fixed[index];
        if (allocation.pointer != nullptr && allocation.capacity >= bytes[index]) {
            ++session.stats.voting_workspace_reused_buffers;
        } else {
            void* replacement = nullptr;
            const cudaError_t allocation_status =
                ::cudaMalloc(&replacement, bytes[index]);
            ++session.stats.cuda_malloc_calls;
            session.stats.cuda_malloc_bytes += bytes[index];
            ++session.stats.voting_workspace_physical_allocations;
            if (allocation_status != cudaSuccess) {
                error = std::string("growing recovered CUDA voting workspace failed: ") +
                        cudaGetErrorString(allocation_status);
                return fail();
            }
            if (allocation.pointer != nullptr) {
                const cudaError_t free_status = ::cudaFree(allocation.pointer);
                ++session.stats.cuda_free_calls;
                if (free_status != cudaSuccess) {
                    ::cudaFree(replacement);
                    ++session.stats.cuda_free_calls;
                    error = std::string(
                        "releasing superseded recovered CUDA voting workspace buffer failed: ") +
                        cudaGetErrorString(free_status);
                    return fail();
                }
            }
            allocation.pointer = replacement;
            allocation.capacity = bytes[index];
        }
        handles.fixed[index] = allocation.pointer;
    }
    std::uint64_t total_workspace_bytes = 0U;
    for (const auto& [thread, candidate] : session.voting_workspaces) {
        (void)thread;
        total_workspace_bytes +=
            recovered_cuda_voting_workspace_bytes(candidate);
    }
    session.stats.voting_workspace_peak_bytes = std::max(
        session.stats.voting_workspace_peak_bytes, total_workspace_bytes);
    error.clear();
    return true;
}

void release_recovered_cuda_voting_workspace_lease() {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    if (!recovered_cuda_module_session.active) return;
    const auto workspace =
        recovered_cuda_module_session.voting_workspaces.find(
            std::this_thread::get_id());
    if (workspace != recovered_cuda_module_session.voting_workspaces.end())
        workspace->second.in_use = false;
}

CUresult destroy_recovered_cuda_voting_workspace_locked(
    RecoveredCudaModuleSessionState& session) {
    CUresult first_failure = CUDA_SUCCESS;
    for (auto& [thread, workspace] : session.voting_workspaces) {
        (void)thread;
        for (auto& allocation : workspace.fixed) {
            if (allocation.pointer == nullptr) continue;
            const cudaError_t status = ::cudaFree(allocation.pointer);
            ++session.stats.cuda_free_calls;
            if (status != cudaSuccess && first_failure == CUDA_SUCCESS)
                first_failure = CUDA_ERROR_UNKNOWN;
            allocation = {};
        }
        workspace = {};
    }
    session.voting_workspaces.clear();
    return first_failure;
}

CUresult destroy_recovered_cuda_voting_depth_cache_locked(
    RecoveredCudaModuleSessionState& session) {
    CUresult first_failure = CUDA_SUCCESS;
    for (auto& [host_pointer, entry] : session.voting_depth_cache) {
        (void)host_pointer;
        if (entry.device_pointer == nullptr) continue;
        const CUresult status = ::cuMemFree(
            reinterpret_cast<CUdeviceptr>(entry.device_pointer));
        ++session.stats.cuda_free_calls;
        if (status != CUDA_SUCCESS && first_failure == CUDA_SUCCESS)
            first_failure = status;
    }
    session.voting_depth_cache.clear();
    return first_failure;
}

template <class Update>
void record_recovered_cuda_session(Update&& update) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    if (recovered_cuda_module_session.active)
        update(recovered_cuda_module_session.stats);
}

CUresult recovered_cuda_module_load(CUmodule* module,
                                    const char* path) {
    if (module == nullptr || path == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    CUcontext context = nullptr;
    CUresult status = ::cuCtxGetCurrent(&context);
    if (status != CUDA_SUCCESS) return status;

    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active) return ::cuModuleLoad(module, path);
    if (context == nullptr || context != session.context)
        return CUDA_ERROR_INVALID_CONTEXT;

    ++session.stats.module_load_requests;
    const auto existing = session.modules.find(path);
    if (existing != session.modules.end()) {
        ++session.stats.module_cache_hits;
        *module = existing->second.module;
        return CUDA_SUCCESS;
    }

    CUmodule loaded = nullptr;
    status = ::cuModuleLoad(&loaded, path);
    if (status != CUDA_SUCCESS) return status;
    RecoveredCudaCachedModule cached;
    cached.module = loaded;
    session.modules.emplace(path, std::move(cached));
    ++session.stats.physical_module_loads;
    *module = loaded;
    return CUDA_SUCCESS;
}

CUresult recovered_cuda_module_get_function(CUfunction* function,
                                            CUmodule module,
                                            const char* name) {
    if (function == nullptr || module == nullptr || name == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active)
        return ::cuModuleGetFunction(function, module, name);

    auto cached_module = std::find_if(
        session.modules.begin(), session.modules.end(),
        [module](const auto& entry) {
            return entry.second.module == module;
        });
    if (cached_module == session.modules.end())
        return ::cuModuleGetFunction(function, module, name);

    ++session.stats.function_requests;
    const auto existing = cached_module->second.functions.find(name);
    if (existing != cached_module->second.functions.end()) {
        ++session.stats.function_cache_hits;
        *function = existing->second;
        return CUDA_SUCCESS;
    }

    CUfunction loaded = nullptr;
    const CUresult status = ::cuModuleGetFunction(&loaded, module, name);
    if (status != CUDA_SUCCESS) return status;
    cached_module->second.functions.emplace(name, loaded);
    ++session.stats.physical_function_lookups;
    *function = loaded;
    return CUDA_SUCCESS;
}

CUresult recovered_cuda_module_unload(CUmodule module) {
    if (module == nullptr) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (session.active) {
        const auto cached_module = std::find_if(
            session.modules.begin(), session.modules.end(),
            [module](const auto& entry) {
                return entry.second.module == module;
            });
        if (cached_module != session.modules.end()) {
            ++session.stats.module_release_requests;
            return CUDA_SUCCESS;
        }
    }
    return ::cuModuleUnload(module);
}

cudaError_t recovered_cuda_set_device(int device) {
    const cudaError_t status = ::cudaSetDevice(device);
    record_recovered_cuda_session(
        [](RecoveredCudaModuleSessionStats& stats) {
            ++stats.set_device_calls;
        });
    return status;
}

cudaError_t recovered_cuda_malloc(void** pointer, std::size_t bytes) {
    const cudaError_t status = ::cudaMalloc(pointer, bytes);
    record_recovered_cuda_session(
        [bytes](RecoveredCudaModuleSessionStats& stats) {
            ++stats.cuda_malloc_calls;
            stats.cuda_malloc_bytes += bytes;
        });
    return status;
}

template <class T>
cudaError_t recovered_cuda_malloc(T** pointer, std::size_t bytes) {
    return recovered_cuda_malloc(reinterpret_cast<void**>(pointer), bytes);
}

cudaError_t recovered_cuda_free(void* pointer) {
    const cudaError_t status = ::cudaFree(pointer);
    record_recovered_cuda_session(
        [pointer](RecoveredCudaModuleSessionStats& stats) {
            ++stats.cuda_free_calls;
            if (pointer == nullptr) ++stats.cuda_free_null_calls;
        });
    return status;
}

cudaError_t recovered_cuda_memcpy(void* destination,
                                  const void* source,
                                  std::size_t bytes,
                                  enum cudaMemcpyKind kind) {
    const auto started = std::chrono::steady_clock::now();
    const cudaError_t status =
        ::cudaMemcpy(destination, source, bytes, kind);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    record_recovered_cuda_session(
        [bytes, kind, elapsed](RecoveredCudaModuleSessionStats& stats) {
            switch (kind) {
                case cudaMemcpyHostToDevice:
                    ++stats.host_to_device_copy_calls;
                    stats.host_to_device_copy_bytes += bytes;
                    stats.host_to_device_copy_nanoseconds += elapsed;
                    record_recovered_cuda_h2d_phase(stats, bytes);
                    break;
                case cudaMemcpyDeviceToHost:
                    ++stats.device_to_host_copy_calls;
                    stats.device_to_host_copy_bytes += bytes;
                    stats.device_to_host_copy_nanoseconds += elapsed;
                    record_recovered_cuda_d2h_phase(stats, bytes);
                    break;
                case cudaMemcpyDeviceToDevice:
                    ++stats.device_to_device_copy_calls;
                    stats.device_to_device_copy_bytes += bytes;
                    break;
                default:
                    break;
            }
        });
    return status;
}

cudaError_t recovered_cuda_memcpy_2d_runtime(
    void* destination, std::size_t destination_pitch,
    const void* source, std::size_t source_pitch,
    std::size_t width_bytes, std::size_t height,
    enum cudaMemcpyKind kind) {
    const auto started = std::chrono::steady_clock::now();
    const cudaError_t status = ::cudaMemcpy2D(
        destination, destination_pitch, source, source_pitch,
        width_bytes, height, kind);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(width_bytes) * height;
    record_recovered_cuda_session(
        [bytes, kind, elapsed](RecoveredCudaModuleSessionStats& stats) {
            switch (kind) {
                case cudaMemcpyHostToDevice:
                    ++stats.host_to_device_copy_calls;
                    stats.host_to_device_copy_bytes += bytes;
                    stats.host_to_device_copy_nanoseconds += elapsed;
                    record_recovered_cuda_h2d_phase(stats, bytes);
                    break;
                case cudaMemcpyDeviceToHost:
                    ++stats.device_to_host_copy_calls;
                    stats.device_to_host_copy_bytes += bytes;
                    stats.device_to_host_copy_nanoseconds += elapsed;
                    record_recovered_cuda_d2h_phase(stats, bytes);
                    break;
                case cudaMemcpyDeviceToDevice:
                    ++stats.device_to_device_copy_calls;
                    stats.device_to_device_copy_bytes += bytes;
                    break;
                default:
                    break;
            }
        });
    return status;
}

cudaError_t recovered_cuda_memset(void* pointer,
                                  int value,
                                  std::size_t bytes) {
    const cudaError_t status = ::cudaMemset(pointer, value, bytes);
    record_recovered_cuda_session(
        [bytes](RecoveredCudaModuleSessionStats& stats) {
            ++stats.memset_calls;
            stats.memset_bytes += bytes;
        });
    return status;
}

CUresult recovered_cuda_array_create(
    CUarray* array,
    const CUDA_ARRAY3D_DESCRIPTOR* description) {
    const CUresult status = ::cuArray3DCreate(array, description);
    record_recovered_cuda_session(
        [](RecoveredCudaModuleSessionStats& stats) {
            ++stats.array_create_calls;
        });
    return status;
}

CUresult recovered_cuda_array_destroy(CUarray array) {
    const CUresult status = ::cuArrayDestroy(array);
    record_recovered_cuda_session(
        [](RecoveredCudaModuleSessionStats& stats) {
            ++stats.array_destroy_calls;
        });
    return status;
}

CUresult recovered_cuda_memcpy_2d(const CUDA_MEMCPY2D* copy) {
    const auto started = std::chrono::steady_clock::now();
    const CUresult status = ::cuMemcpy2D(copy);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    const std::uint64_t bytes = copy == nullptr
        ? 0U
        : static_cast<std::uint64_t>(copy->WidthInBytes) * copy->Height;
    record_recovered_cuda_session(
        [bytes, elapsed](RecoveredCudaModuleSessionStats& stats) {
            ++stats.array_copy_calls;
            stats.array_copy_bytes += bytes;
            stats.array_copy_nanoseconds += elapsed;
        });
    return status;
}

CUresult recovered_cuda_memcpy_2d_async(
    const CUDA_MEMCPY2D* copy,
    CUstream stream) {
    const auto started = std::chrono::steady_clock::now();
    const CUresult status = ::cuMemcpy2DAsync(copy, stream);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    const std::uint64_t bytes = copy == nullptr
        ? 0U
        : static_cast<std::uint64_t>(copy->WidthInBytes) * copy->Height;
    record_recovered_cuda_session(
        [bytes, elapsed](RecoveredCudaModuleSessionStats& stats) {
            ++stats.array_copy_calls;
            stats.array_copy_bytes += bytes;
            stats.array_copy_nanoseconds += elapsed;
        });
    return status;
}

CUresult recovered_cuda_texture_create(
    CUtexObject* texture,
    const CUDA_RESOURCE_DESC* resource,
    const CUDA_TEXTURE_DESC* description,
    const CUDA_RESOURCE_VIEW_DESC* view) {
    const CUresult status =
        ::cuTexObjectCreate(texture, resource, description, view);
    record_recovered_cuda_session(
        [](RecoveredCudaModuleSessionStats& stats) {
            ++stats.texture_create_calls;
        });
    return status;
}

CUresult recovered_cuda_texture_destroy(CUtexObject texture) {
    const CUresult status = ::cuTexObjectDestroy(texture);
    record_recovered_cuda_session(
        [](RecoveredCudaModuleSessionStats& stats) {
            ++stats.texture_destroy_calls;
        });
    return status;
}

CUresult recovered_cuda_launch_kernel(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    unsigned int shared_memory_bytes,
    CUstream stream,
    void** kernel_parameters,
    void** extra) {
    CUstream effective_stream = stream;
    if (effective_stream == nullptr) {
        std::lock_guard lock(recovered_cuda_module_session_mutex);
        auto& session = recovered_cuda_module_session;
        if (session.active) {
            CUstream& worker_stream =
                session.worker_streams[std::this_thread::get_id()];
            if (worker_stream == nullptr) {
                const CUresult create_status =
                    ::cuStreamCreate(&worker_stream, CU_STREAM_DEFAULT);
                if (create_status != CUDA_SUCCESS) return create_status;
            }
            effective_stream = worker_stream;
        }
    }
    const CUresult status = ::cuLaunchKernel(
        function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
        shared_memory_bytes, effective_stream, kernel_parameters, extra);
    record_recovered_cuda_session(
        [](RecoveredCudaModuleSessionStats& stats) {
            ++stats.kernel_launches;
        });
    return status;
}

CUresult recovered_cuda_context_synchronize() {
    CUstream worker_stream = nullptr;
    {
        std::lock_guard lock(recovered_cuda_module_session_mutex);
        const auto& session = recovered_cuda_module_session;
        if (session.active) {
            const auto found = session.worker_streams.find(
                std::this_thread::get_id());
            if (found != session.worker_streams.end())
                worker_stream = found->second;
        }
    }
    const auto started = std::chrono::steady_clock::now();
    const CUresult status = worker_stream != nullptr
        ? ::cuStreamSynchronize(worker_stream)
        : ::cuCtxSynchronize();
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    record_recovered_cuda_session([worker_stream, elapsed](
                                      RecoveredCudaModuleSessionStats& stats) {
        if (worker_stream != nullptr)
            ++stats.stream_synchronizations;
        else
            ++stats.context_synchronizations;
        if (worker_stream != nullptr)
            stats.stream_synchronization_nanoseconds += elapsed;
    });
    return status;
}

CUresult recovered_cuda_stream_synchronize(CUstream stream) {
    const auto started = std::chrono::steady_clock::now();
    const CUresult status = ::cuStreamSynchronize(stream);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    record_recovered_cuda_session(
        [elapsed](RecoveredCudaModuleSessionStats& stats) {
            ++stats.stream_synchronizations;
            stats.stream_synchronization_nanoseconds += elapsed;
        });
    return status;
}

}  // namespace

RecoveredCudaPinnedNeighborTextures::~RecoveredCudaPinnedNeighborTextures() {
    reset();
}

RecoveredCudaPinnedNeighborTextures::RecoveredCudaPinnedNeighborTextures(
    RecoveredCudaPinnedNeighborTextures&& other) noexcept
    : device_index_(other.device_index_), pointers_(std::move(other.pointers_)) {
    other.pointers_.clear();
}

RecoveredCudaPinnedNeighborTextures&
RecoveredCudaPinnedNeighborTextures::operator=(
    RecoveredCudaPinnedNeighborTextures&& other) noexcept {
    if (this == &other) return *this;
    reset();
    device_index_ = other.device_index_;
    pointers_ = std::move(other.pointers_);
    other.pointers_.clear();
    return *this;
}

void RecoveredCudaPinnedNeighborTextures::reset() noexcept {
    if (pointers_.empty()) return;
    (void)::cudaSetDevice(static_cast<int>(device_index_));
    for (void* pointer : pointers_) {
        const auto started = std::chrono::steady_clock::now();
        (void)::cudaHostUnregister(pointer);
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        record_recovered_cuda_session(
            [elapsed](RecoveredCudaModuleSessionStats& stats) {
                ++stats.host_unregister_calls;
                stats.host_unregister_nanoseconds += elapsed;
            });
    }
    pointers_.clear();
}

bool pin_recovered_patchmatch_neighbor_textures_cuda(
    const RecoveredPatchMatchNeighborResources& resources,
    std::size_t device_index,
    RecoveredCudaPinnedNeighborTextures& registration,
    std::string& error) {
    if (device_index >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = "recovered CUDA pinned-texture device index exceeds int range";
        return false;
    }
    registration.reset();
    registration.device_index_ = device_index;
    cudaError_t status = ::cudaSetDevice(static_cast<int>(device_index));
    const auto register_bytes = [&](const std::vector<std::uint8_t>& bytes) {
        if (status != cudaSuccess) return;
        if (bytes.empty()) {
            status = cudaErrorInvalidValue;
            return;
        }
        void* pointer = const_cast<std::uint8_t*>(bytes.data());
        const auto started = std::chrono::steady_clock::now();
        status = ::cudaHostRegister(
            pointer, bytes.size(), cudaHostRegisterDefault);
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        record_recovered_cuda_session(
            [byte_count = bytes.size(), elapsed](
                RecoveredCudaModuleSessionStats& stats) {
                ++stats.host_register_calls;
                stats.host_register_bytes += byte_count;
                stats.host_register_nanoseconds += elapsed;
            });
        if (status == cudaSuccess)
            registration.pointers_.push_back(pointer);
    };
    for (const PatchMatchCostResourceGroup& group : resources.resource_groups) {
        if (!group.image.empty()) register_bytes(group.image);
        if (status != cudaSuccess) break;
    }
    for (const PatchMatchCostNeighbor& neighbor : resources.ranked_neighbors) {
        if (!neighbor.texture_source.empty())
            register_bytes(neighbor.texture_source);
        if (status != cudaSuccess) break;
    }
    if (status != cudaSuccess) {
        error = std::string("pinning recovered PatchMatch neighbor texture failed: ") +
                ::cudaGetErrorString(status);
        registration.reset();
        return false;
    }
    error.clear();
    return true;
}

bool begin_recovered_cuda_module_session_impl(
    std::size_t device_index,
    std::string& error) {
    if (device_index >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = "recovered CUDA module session device index exceeds int range";
        return false;
    }
    const int physical_device = static_cast<int>(device_index);
    cudaError_t runtime_status = cudaSetDevice(physical_device);
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("initializing recovered CUDA module session failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUcontext context = nullptr;
    const CUresult context_status = ::cuCtxGetCurrent(&context);
    if (context_status != CUDA_SUCCESS || context == nullptr) {
        const char* message = nullptr;
        ::cuGetErrorString(context_status, &message);
        error = std::string("querying recovered CUDA module session context failed: ") +
                (message ? message : "no current CUDA context");
        return false;
    }

    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (session.active) {
        if (session.context != context ||
            session.stats.device_index != device_index) {
            error = "nested recovered CUDA module session changed device or context";
            return false;
        }
        ++session.nesting;
        ++session.stats.scope_entries;
        error.clear();
        return true;
    }

    cudaDeviceProp properties{};
    runtime_status = cudaGetDeviceProperties(&properties, physical_device);
    int driver_version = 0;
    int runtime_version = 0;
    if (runtime_status == cudaSuccess)
        runtime_status = cudaDriverGetVersion(&driver_version);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaRuntimeGetVersion(&runtime_version);
    if (runtime_status != cudaSuccess) {
        error = std::string("querying recovered CUDA runtime identity failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    session = {};
    session.active = true;
    session.nesting = 1U;
    session.context = context;
    session.stats.device_index = device_index;
    session.stats.device_name = properties.name;
    session.stats.total_global_memory = properties.totalGlobalMem;
    session.stats.compute_capability_major = properties.major;
    session.stats.compute_capability_minor = properties.minor;
    session.stats.driver_version = driver_version;
    session.stats.runtime_version = runtime_version;
    session.stats.scope_entries = 1U;
    error.clear();
    return true;
}

bool end_recovered_cuda_module_session_impl(
    RecoveredCudaModuleSessionStats& stats,
    std::string& error) {
    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active || session.nesting == 0U) {
        error = "recovered CUDA module session is not active";
        return false;
    }
    --session.nesting;
    if (session.nesting != 0U) {
        stats = session.stats;
        error.clear();
        return true;
    }

    const bool cost_workspace_in_use = std::any_of(
        session.cost_workspaces.begin(), session.cost_workspaces.end(),
        [](const auto& entry) { return entry.second.in_use; });
    const bool voting_workspace_in_use = std::any_of(
        session.voting_workspaces.begin(), session.voting_workspaces.end(),
        [](const auto& entry) { return entry.second.in_use; });
    if (cost_workspace_in_use || voting_workspace_in_use) {
        ++session.nesting;
        error = "recovered CUDA module session ended while a workspace was in use";
        return false;
    }

    CUresult first_failure =
        destroy_recovered_cuda_cost_workspace_locked(session);
    const CUresult voting_workspace_status =
        destroy_recovered_cuda_voting_workspace_locked(session);
    if (first_failure == CUDA_SUCCESS)
        first_failure = voting_workspace_status;
    const CUresult voting_depth_cache_status =
        destroy_recovered_cuda_voting_depth_cache_locked(session);
    if (first_failure == CUDA_SUCCESS)
        first_failure = voting_depth_cache_status;
    for (const auto& [thread, stream] : session.worker_streams) {
        (void)thread;
        if (stream == nullptr) continue;
        const CUresult status = ::cuStreamDestroy(stream);
        if (status != CUDA_SUCCESS && first_failure == CUDA_SUCCESS)
            first_failure = status;
    }
    session.worker_streams.clear();
    for (auto& [path, cached] : session.modules) {
        (void)path;
        const CUresult status = ::cuModuleUnload(cached.module);
        if (status == CUDA_SUCCESS) {
            ++session.stats.physical_module_unloads;
        } else if (first_failure == CUDA_SUCCESS) {
            first_failure = status;
        }
    }
    stats = session.stats;
    session = {};
    if (first_failure != CUDA_SUCCESS) {
        const char* message = nullptr;
        ::cuGetErrorString(first_failure, &message);
        error = std::string("unloading recovered CUDA module session failed: ") +
                (message ? message : "unknown CUDA driver error");
        return false;
    }
    error.clear();
    return true;
}

// Route every existing recovered wrapper through the explicit session cache
// without changing the wrapper control flow. Outside a session these helpers
// delegate directly to the CUDA driver and preserve the old behavior.
#define cuModuleLoad recovered_cuda_module_load
#define cuModuleGetFunction recovered_cuda_module_get_function
#define cuModuleUnload recovered_cuda_module_unload
#define cudaSetDevice recovered_cuda_set_device
#define cudaMalloc recovered_cuda_malloc
#define cudaFree recovered_cuda_free
#define cudaMemcpy recovered_cuda_memcpy
#define cudaMemcpy2D recovered_cuda_memcpy_2d_runtime
#define cudaMemset recovered_cuda_memset
#undef cuArray3DCreate
#define cuArray3DCreate recovered_cuda_array_create
#define cuArrayDestroy recovered_cuda_array_destroy
#undef cuMemcpy2D
#define cuMemcpy2D recovered_cuda_memcpy_2d
#undef cuMemcpy2DAsync
#define cuMemcpy2DAsync recovered_cuda_memcpy_2d_async
#define cuTexObjectCreate recovered_cuda_texture_create
#define cuTexObjectDestroy recovered_cuda_texture_destroy
#define cuLaunchKernel recovered_cuda_launch_kernel
#define cuCtxSynchronize recovered_cuda_context_synchronize
#define cuStreamSynchronize recovered_cuda_stream_synchronize

std::vector<ComputeDeviceInfo> enumerate_cuda_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return {};
    std::vector<ComputeDeviceInfo> result;
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, index) == cudaSuccess)
            result.push_back({ComputeBackend::CUDA, static_cast<std::size_t>(index),
                              properties.name, properties.totalGlobalMem});
    }
    return result;
}

std::unique_ptr<ComputeContext> create_cuda_context(const GPUOptions& options,
                                                    std::string& diagnostic) {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) { diagnostic = cudaGetErrorString(status); return {}; }
    for (int index = 0; index < count; ++index) {
        if ((options.gpu_mask & (std::uint64_t{1} << static_cast<unsigned int>(index))) == 0) continue;
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, index) != cudaSuccess) continue;
        diagnostic = "using device " + std::to_string(index) + " (" + properties.name + ")";
        return std::make_unique<CUDAContext>(static_cast<std::size_t>(index), properties,
                                              options.memory_guard_mb * 1024ULL * 1024ULL);
    }
    diagnostic = "no enabled CUDA device found with gpu mask";
    return {};
}

bool run_recovered_ooc_histogram_cuda_chain_impl(
    const OocHistogramCudaChainInput& input,
    OocHistogramCudaChainOutput& output,
    std::string& error) {
    output = {};
    if (input.cameras.empty()) {
        error = "recovered OOC CUDA histogram chain requires at least one camera";
        return false;
    }
    if (input.device_index >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = "recovered OOC CUDA histogram device index exceeds int range";
        return false;
    }
    std::size_t pyramid_float_count = 0U;
    for (std::size_t camera = 0; camera != input.cameras.size(); ++camera) {
        const auto& current = input.cameras[camera];
        if (current.concatenated_float2_pyramid.empty() ||
            current.pyramid_levels == 0U || current.pyramid_levels > 31U ||
            !std::isfinite(current.threshold_a) ||
            !std::isfinite(current.threshold_b)) {
            error = "recovered OOC CUDA histogram camera input is outside the proven finite mode-0 domain";
            return false;
        }
        if (camera == 0U) {
            pyramid_float_count = current.concatenated_float2_pyramid.size();
        } else if (current.concatenated_float2_pyramid.size() !=
                   pyramid_float_count) {
            error = "recovered OOC CUDA histogram cameras require one common pyramid allocation size";
            return false;
        }
    }
    std::array<std::size_t, 4> partition_bytes{};
    for (std::size_t partition = 0; partition != partition_bytes.size();
         ++partition) {
        const std::size_t records = input.initial_partitions[partition].size();
        if (records == 0U ||
            records > std::numeric_limits<std::uint32_t>::max() ||
            records > std::numeric_limits<std::size_t>::max() /
                          sizeof(OocHistogramVoxel)) {
            error = "recovered OOC CUDA histogram partition is empty or exceeds the uint32 kernel domain";
            return false;
        }
        partition_bytes[partition] = records * sizeof(OocHistogramVoxel);
    }
    if (pyramid_float_count > std::numeric_limits<std::size_t>::max() /
                                  sizeof(float)) {
        error = "recovered OOC CUDA histogram pyramid size overflows size_t";
        return false;
    }
    const std::size_t pyramid_bytes = pyramid_float_count * sizeof(float);

    cudaError_t runtime_status =
        cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA OOC histogram context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    constexpr const char* kernel_name =
        "_ZN4cuda18ooc_histogram_initEP9HistVoxelPKfPKhjjNS_13CalibrationCuE"
        "NS_16RollingShutterCuENS_10Matrix3x3fES8_NS_25CameraExteriorTransfor"
        "mCuEffjj";
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUstream stream = nullptr;
    CUresult driver_status =
        cuModuleLoad(&module, METMODEL_OOC_HISTOGRAM_CUDA_PTX);
    if (driver_status == CUDA_SUCCESS) {
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    }
    if (driver_status == CUDA_SUCCESS) {
        driver_status = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    }
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered OOC histogram PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (stream) cuStreamDestroy(stream);
        if (module) cuModuleUnload(module);
        return false;
    }
    std::array<void*, 4> device_partitions{};
    void* device_pyramid = nullptr;
    void* unused_auxiliary = nullptr;
    const auto release = [&]() {
        for (void* pointer : device_partitions) cudaFree(pointer);
        cudaFree(device_pyramid);
        cudaFree(unused_auxiliary);
        if (stream) cuStreamDestroy(stream);
        if (module) cuModuleUnload(module);
    };
    for (std::size_t partition = 0;
         partition != device_partitions.size() &&
         runtime_status == cudaSuccess;
         ++partition) {
        runtime_status =
            cudaMalloc(&device_partitions[partition], partition_bytes[partition]);
    }
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMalloc(&device_pyramid, pyramid_bytes);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMalloc(&unused_auxiliary, 1U);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemset(unused_auxiliary, 0, 1U);
    for (std::size_t partition = 0;
         partition != device_partitions.size() &&
         runtime_status == cudaSuccess;
         ++partition) {
        runtime_status = cudaMemcpy(
            device_partitions[partition],
            input.initial_partitions[partition].data(), partition_bytes[partition],
            cudaMemcpyHostToDevice);
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating/uploading recovered OOC histogram workspace failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    constexpr std::uint32_t block_size = 128U;
    std::uint64_t launches = 0U;
    for (const auto& camera : input.cameras) {
        runtime_status = cudaMemcpy(
            device_pyramid, camera.concatenated_float2_pyramid.data(),
            pyramid_bytes, cudaMemcpyHostToDevice);
        if (runtime_status != cudaSuccess) {
            error = std::string("uploading recovered OOC histogram pyramid failed: ") +
                    cudaGetErrorString(runtime_status);
            release();
            return false;
        }
        for (std::size_t partition = 0;
             partition != device_partitions.size(); ++partition) {
            const std::uint32_t begin = 0U;
            const std::uint32_t end = static_cast<std::uint32_t>(
                input.initial_partitions[partition].size());
            const std::uint32_t grid =
                (end - begin + block_size - 1U) / block_size;
            void* parameters[] = {
                &device_partitions[partition],
                &device_pyramid,
                &unused_auxiliary,
                const_cast<std::uint32_t*>(&camera.pyramid_levels),
                const_cast<std::uint32_t*>(&camera.mode_b),
                const_cast<OocHistogramCalibrationCu*>(
                    &camera.parameters.calibration),
                const_cast<OocHistogramRollingShutterCu*>(
                    &camera.parameters.rolling_shutter),
                const_cast<OocHistogramMatrix3x3f*>(
                    &camera.parameters.before_rotation),
                const_cast<OocHistogramMatrix3x3f*>(
                    &camera.parameters.after_rotation),
                const_cast<OocHistogramCameraExteriorTransformCu*>(
                    &camera.parameters.exterior_transform),
                const_cast<float*>(&camera.threshold_a),
                const_cast<float*>(&camera.threshold_b),
                const_cast<std::uint32_t*>(&begin),
                const_cast<std::uint32_t*>(&end),
            };
            driver_status = cuLaunchKernel(
                function, grid, 1U, 1U, block_size, 1U, 1U, 0U, stream,
                parameters, nullptr);
            if (driver_status == CUDA_SUCCESS)
                driver_status = cuStreamSynchronize(stream);
            if (driver_status != CUDA_SUCCESS) {
                const char* message = nullptr;
                cuGetErrorString(driver_status, &message);
                error = std::string("recovered OOC histogram kernel failed: ") +
                        (message ? message : "unknown CUDA driver error");
                release();
                return false;
            }
            ++launches;
        }
    }

    OocHistogramCudaChainOutput result;
    result.kernel_launches = launches;
    for (std::size_t partition = 0; partition != result.partitions.size();
         ++partition) {
        result.partitions[partition].resize(
            input.initial_partitions[partition].size());
        runtime_status = cudaMemcpy(
            result.partitions[partition].data(), device_partitions[partition],
            partition_bytes[partition], cudaMemcpyDeviceToHost);
        if (runtime_status != cudaSuccess) {
            error = std::string("downloading recovered OOC histogram result failed: ") +
                    cudaGetErrorString(runtime_status);
            release();
            return false;
        }
    }
    release();
    output = std::move(result);
    return true;
}

bool run_ooc_fusion_cuda_impl(OocFusionState& state,
                              const OocFusionParameters& parameters,
                              std::size_t device_index,
                              OocFusionCudaStats& stats,
                              std::string& error) {
    stats = {};
    try {
        state.validate();
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    if (state.size() == 0U) {
        error = "recovered OOC CUDA fusion requires at least one voxel";
        return false;
    }
    if (state.size() > std::numeric_limits<std::uint32_t>::max()) {
        error = "recovered OOC CUDA fusion count exceeds uint32 kernel domain";
        return false;
    }
    if (device_index >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = "recovered OOC CUDA fusion device index exceeds int range";
        return false;
    }
    if (!(parameters.alpha > 0.0F) || !(parameters.beta > 0.0F) ||
        !(parameters.data_weight >= 0.0F) ||
        !std::isfinite(parameters.alpha) ||
        !std::isfinite(parameters.beta) ||
        !std::isfinite(parameters.data_weight)) {
        error = "invalid recovered OOC CUDA fusion parameters";
        return false;
    }
    if (parameters.iterations >
        std::numeric_limits<std::uint64_t>::max() / 2U) {
        error = "recovered OOC CUDA fusion iteration count overflows statistics";
        return false;
    }

    cudaError_t runtime_status =
        cudaSetDevice(static_cast<int>(device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("initializing recovered OOC CUDA fusion failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    constexpr const char* update_u_name =
        "_ZN4cuda12ooc_update_uEPKhS1_PKjS1_S1_S1_PfS4_PK6__halfPS5_S8_S7_jfffffij";
    constexpr const char* update_p_name =
        "_ZN4cuda12ooc_update_pEPKjPKhS3_S3_PKfS5_P6__halfPKS6_S9_S7_jffffij";
    CUmodule update_u_module = nullptr;
    CUmodule update_p_module = nullptr;
    CUfunction update_u = nullptr;
    CUfunction update_p = nullptr;
    CUstream stream = nullptr;
    std::array<void*, 12> device{};

    const std::array<std::size_t, 12> bytes{{
        state.weights.size() * sizeof(state.weights[0]),
        state.histogram.size() * sizeof(state.histogram[0]),
        state.neighbors.size() * sizeof(state.neighbors[0]),
        state.connectivity.size() * sizeof(state.connectivity[0]),
        state.refinement.size() * sizeof(state.refinement[0]),
        state.flags.size() * sizeof(state.flags[0]),
        state.u.size() * sizeof(state.u[0]),
        state.u_old.size() * sizeof(state.u_old[0]),
        state.p.size() * sizeof(state.p[0]),
        state.v.size() * sizeof(state.v[0]),
        state.v_old.size() * sizeof(state.v_old[0]),
        state.q.size() * sizeof(state.q[0]),
    }};
    const std::array<const void*, 12> host_input{{
        state.weights.data(), state.histogram.data(), state.neighbors.data(),
        state.connectivity.data(), state.refinement.data(), state.flags.data(),
        state.u.data(), state.u_old.data(), state.p.data(), state.v.data(),
        state.v_old.data(), state.q.data(),
    }};
    const std::array<void*, 12> host_output{{
        state.weights.data(), state.histogram.data(), state.neighbors.data(),
        state.connectivity.data(), state.refinement.data(), state.flags.data(),
        state.u.data(), state.u_old.data(), state.p.data(), state.v.data(),
        state.v_old.data(), state.q.data(),
    }};

    const auto release = [&]() {
        for (void* pointer : device) cudaFree(pointer);
        if (stream != nullptr) ::cuStreamDestroy(stream);
        if (update_p_module != nullptr) cuModuleUnload(update_p_module);
        if (update_u_module != nullptr) cuModuleUnload(update_u_module);
    };
    const auto driver_failure = [&](const char* operation,
                                    CUresult status) {
        const char* message = nullptr;
        ::cuGetErrorString(status, &message);
        error = std::string(operation) + ": " +
                (message ? message : "unknown CUDA driver error");
    };

    CUresult driver_status =
        cuModuleLoad(&update_u_module, METMODEL_OOC_UPDATE_U_CUDA_PTX);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(
            &update_u, update_u_module, update_u_name);
    if (driver_status == CUDA_SUCCESS)
        driver_status =
            cuModuleLoad(&update_p_module, METMODEL_OOC_UPDATE_P_CUDA_PTX);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(
            &update_p, update_p_module, update_p_name);
    if (driver_status == CUDA_SUCCESS)
        driver_status = ::cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    if (driver_status != CUDA_SUCCESS) {
        driver_failure("loading recovered OOC fusion PTX failed", driver_status);
        release();
        return false;
    }

    for (std::size_t index = 0U;
         index != device.size() && runtime_status == cudaSuccess; ++index) {
        runtime_status = cudaMalloc(&device[index], bytes[index]);
        if (runtime_status == cudaSuccess) {
            runtime_status = cudaMemcpy(
                device[index], host_input[index], bytes[index],
                cudaMemcpyHostToDevice);
        }
        if (runtime_status == cudaSuccess)
            stats.host_to_device_bytes += bytes[index];
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("uploading recovered OOC fusion state failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    // sub_4168770(count, 128, 1000000): split into ceil(count/1e6)
    // pieces, then round the per-piece capacity up to a full 128-thread block.
    const std::uint32_t count = static_cast<std::uint32_t>(state.size());
    constexpr std::uint32_t block_size = 128U;
    constexpr std::uint32_t requested_maximum = 1000000U;
    const std::uint32_t piece_count =
        (count + requested_maximum - 1U) / requested_maximum;
    const std::uint32_t average_piece =
        (count + piece_count - 1U) / piece_count;
    const std::uint32_t piece_capacity =
        block_size * ((average_piece + block_size - 1U) / block_size);
    const float spatial_scale = 8.0F;
    const float update_u_threshold = 0.02F;
    const std::uint32_t functional_mode = 1U;

    const auto launch_and_sync = [&](CUfunction function,
                                     std::uint32_t active,
                                     void** arguments) {
        const std::uint32_t blocks =
            (active + block_size - 1U) / block_size;
        CUresult status = cuLaunchKernel(
            function, blocks, 1U, 1U, block_size, 1U, 1U,
            0U, stream, arguments, nullptr);
        if (status != CUDA_SUCCESS) return status;
        ++stats.kernel_launches;
        const cudaError_t launch_status = ::cudaGetLastError();
        if (launch_status != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;
        status = cuStreamSynchronize(stream);
        if (status == CUDA_SUCCESS) ++stats.stream_synchronizations;
        return status;
    };

    for (std::size_t iteration = 0U;
         iteration != parameters.iterations; ++iteration) {
        for (std::uint32_t offset = 0U; offset < count;) {
            const std::uint32_t active =
                std::min(piece_capacity, count - offset);
            void* arguments[] = {
                &device[0], &device[1], &device[2], &device[3],
                &device[4], &device[5], &device[6], &device[7],
                &device[8], &device[9], &device[10], &device[11],
                const_cast<std::uint32_t*>(&count),
                const_cast<float*>(&spatial_scale),
                const_cast<float*>(&update_u_threshold),
                const_cast<float*>(&parameters.alpha),
                const_cast<float*>(&parameters.beta),
                const_cast<float*>(&parameters.data_weight),
                const_cast<std::uint32_t*>(&functional_mode), &offset,
            };
            driver_status = launch_and_sync(update_u, active, arguments);
            if (driver_status != CUDA_SUCCESS) break;
            offset += piece_capacity;
        }
        if (driver_status != CUDA_SUCCESS) break;
        for (std::uint32_t offset = 0U; offset < count;) {
            const std::uint32_t active =
                std::min(piece_capacity, count - offset);
            void* arguments[] = {
                &device[2], &device[3], &device[4], &device[5],
                &device[6], &device[7], &device[8], &device[9],
                &device[10], &device[11],
                const_cast<std::uint32_t*>(&count),
                const_cast<float*>(&spatial_scale),
                const_cast<float*>(&parameters.alpha),
                const_cast<float*>(&parameters.beta),
                const_cast<float*>(&parameters.data_weight),
                const_cast<std::uint32_t*>(&functional_mode), &offset,
            };
            driver_status = launch_and_sync(update_p, active, arguments);
            if (driver_status != CUDA_SUCCESS) break;
            offset += piece_capacity;
        }
        if (driver_status != CUDA_SUCCESS) break;
    }
    if (driver_status != CUDA_SUCCESS) {
        driver_failure("running recovered OOC fusion kernel failed",
                       driver_status);
        release();
        return false;
    }

    for (std::size_t index = 0U;
         index != device.size() && runtime_status == cudaSuccess; ++index) {
        runtime_status = cudaMemcpy(
            host_output[index], device[index], bytes[index],
            cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            stats.device_to_host_bytes += bytes[index];
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("downloading recovered OOC fusion state failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    error.clear();
    return true;
}

bool run_recovered_patchmatch_undistort_u8_cuda_impl(
    const PatchMatchUndistortU8Input& input,
    RecoveredPatchMatchImageU8& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::undistort);
    const std::size_t pixels =
        static_cast<std::size_t>(input.width) * input.height;
    const auto calibration_u32 = [](const DepthVotingCalibrationCu& calibration,
                                    std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, calibration.bytes.data() + offset, sizeof(value));
        return value;
    };
    if (input.width == 0 || input.height == 0 ||
        pixels > std::numeric_limits<std::uint32_t>::max() ||
        input.source_image.size() != pixels ||
        (!input.source_mask.empty() && input.source_mask.size() != pixels) ||
        calibration_u32(input.source_calibration, 84U) != input.width ||
        calibration_u32(input.source_calibration, 88U) != input.height ||
        calibration_u32(input.target_calibration, 84U) != input.width ||
        calibration_u32(input.target_calibration, 88U) != input.height) {
        error = "PatchMatch uint8 undistort dimensions exceed the recovered uint32 launch domain or do not match buffers/calibrations";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA undistort context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status =
        cuModuleLoad(&module, METMODEL_PM_UNDISTORT_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda18pm_undistort_imageIhhEEvPKT_PKhPT0_PhPKfNS_13CalibrationCuESA_NS_16RollingShutterCuENS_10Matrix3x3fESB_jjjjjjhhj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch undistort PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    void* device_source = nullptr;
    void* device_source_mask = nullptr;
    void* device_result = nullptr;
    void* device_result_mask = nullptr;
    auto release = [&]() {
        cudaFree(device_source);
        cudaFree(device_source_mask);
        cudaFree(device_result);
        cudaFree(device_result_mask);
        cuModuleUnload(module);
    };
    runtime_status = cudaMalloc(&device_source, pixels);
    if (runtime_status == cudaSuccess && !input.source_mask.empty())
        runtime_status = cudaMalloc(&device_source_mask, pixels);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMalloc(&device_result, pixels);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMalloc(&device_result_mask, pixels);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(device_source, input.source_image.data(), pixels,
                                    cudaMemcpyHostToDevice);
    if (runtime_status == cudaSuccess && !input.source_mask.empty())
        runtime_status = cudaMemcpy(device_source_mask, input.source_mask.data(), pixels,
                                    cudaMemcpyHostToDevice);
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered PatchMatch undistort buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    alignas(4) std::array<std::byte, 24> rolling_shutter{};
    alignas(16) std::array<float, 12> film{
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F};
    void* depth_map = nullptr;
    void* fourier_coefficients = nullptr;
    std::uint32_t channels = 1U;
    std::uint32_t source_width = input.width;
    std::uint32_t source_height = input.height;
    std::uint32_t result_width = input.width;
    std::uint32_t result_height = input.height;
    std::uint32_t depth_map_downscale = 2U;
    std::uint8_t rolling_shutter_translation = 0U;
    std::uint8_t with_mask = input.source_mask.empty() ? 0U : 1U;

    // 0x25A6480 partitions this preprocessing kernel at approximately one
    // million pixels using the same balanced/aligned rule.  South's 7,077,888
    // pixels therefore use eight 884,736-item launches.
    const std::uint32_t pixel_count = static_cast<std::uint32_t>(pixels);
    const std::uint32_t span =
        patchmatch_balanced_batch_span(pixel_count, 128U, 1000000U);
    for (std::uint32_t pixel_offset = 0; pixel_offset < pixel_count;
         pixel_offset += span) {
        const std::uint32_t items = std::min(span, pixel_count - pixel_offset);
        void* arguments[] = {
            &device_source, &device_source_mask, &device_result,
            &device_result_mask, &depth_map,
            const_cast<DepthVotingCalibrationCu*>(&input.source_calibration),
            &fourier_coefficients, rolling_shutter.data(), film.data(),
            const_cast<DepthVotingCalibrationCu*>(&input.target_calibration),
            &channels, &source_width, &source_height, &result_width,
            &result_height, &depth_map_downscale, &rolling_shutter_translation,
            &with_mask, &pixel_offset};
        const unsigned int blocks = (items + 127U) / 128U;
        driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                       nullptr, arguments, nullptr);
        if (driver_status != CUDA_SUCCESS) break;
    }
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch undistort launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    output.width = input.width;
    output.height = input.height;
    output.image.resize(pixels);
    output.rejection_mask.resize(pixels);
    runtime_status = cudaMemcpy(output.image.data(), device_result, pixels,
                                cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(output.rejection_mask.data(), device_result_mask,
                                    pixels, cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered PatchMatch undistort output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_refinement_cuda_impl(
    PatchMatchRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::producer);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch depth downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    const std::span<const std::uint8_t> reference_image =
        input.reference_image_view.empty()
            ? std::span<const std::uint8_t>(input.reference_image)
            : input.reference_image_view;
    const std::span<const float> depth = input.depth_view.empty()
        ? std::span<const float>(input.depth) : input.depth_view;
    const std::span<const std::uint8_t> normal = input.normal_view.empty()
        ? std::span<const std::uint8_t>(input.normal) : input.normal_view;
    const std::span<const float> main_cost = input.cost_view.empty()
        ? std::span<const float>(input.cost) : input.cost_view;
    const std::span<const float> coarse_depth =
        input.coarse_depth_view.empty()
            ? std::span<const float>(input.coarse_depth)
            : input.coarse_depth_view;
    const std::span<const float> coarse_radius =
        input.coarse_depth_radius_view.empty()
            ? std::span<const float>(input.coarse_depth_radius)
            : input.coarse_depth_radius_view;
    if (pixels == 0 || work_items == 0 ||
        work_items > PatchMatchCandidateOutput::capacity ||
        input.pixel_offset + work_items > pixels) {
        error = "PatchMatch dimensions or launch range are invalid";
        return false;
    }
    if (depth.size() != pixels || normal.size() != pixels * 3 ||
        main_cost.size() != pixels || coarse_depth.size() != pixels ||
        coarse_radius.size() != pixels || reference_image.empty() ||
        input.initial_candidate_depth.empty() !=
            input.initial_candidate_normal.empty() ||
        (!input.initial_candidate_depth.empty() &&
         (input.initial_candidate_depth.size() !=
              PatchMatchCandidateOutput::hypotheses *
                  PatchMatchCandidateOutput::capacity ||
          input.initial_candidate_normal.size() !=
              PatchMatchCandidateOutput::hypotheses *
                  PatchMatchCandidateOutput::capacity * 3))) {
        error = "PatchMatch input buffer sizes do not match the camera grid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda13pm_refinementIhEEvPfP6uchar3S1_S1_S1_13PinholeCamerajffPKT_jfS1_P6float3jjj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.depth = input.initial_candidate_depth.empty()
        ? std::vector<float>(PatchMatchCandidateOutput::hypotheses *
                                 PatchMatchCandidateOutput::capacity,
                             std::numeric_limits<float>::quiet_NaN())
        : std::move(input.initial_candidate_depth);
    output.normal = input.initial_candidate_normal.empty()
        ? std::vector<float>(PatchMatchCandidateOutput::hypotheses *
                                 PatchMatchCandidateOutput::capacity * 3,
                             std::numeric_limits<float>::quiet_NaN())
        : std::move(input.initial_candidate_normal);
    output.cuda_workspace_handoff = 0U;
    output.cuda_resident_cost_scratch = false;
    output.cuda_host_output_materialized = true;
    const std::array<std::size_t,
                     RecoveredCudaCostWorkspace::fixed_allocation_count>
        workspace_bytes = {
            depth.size() * sizeof(float),
            normal.size(),
            main_cost.size() * sizeof(float),
            coarse_depth.size() * sizeof(float),
            coarse_radius.size() * sizeof(float),
            reference_image.size(),
            output.depth.size() * sizeof(float),
            output.normal.size() * sizeof(float),
        };
    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_pipeline_fixed_workspace(
            workspace_bytes, RecoveredCudaWorkspaceClient::producer,
            workspace_handles, using_cached_workspace, error)) {
        cuModuleUnload(module);
        return false;
    }

    bool completed_handoff = false;
    bool consume_pipeline_handoff = false;
    bool consume_cross_level_handoff = false;
    std::uint32_t consumed_handoff_stage = 0U;
    if (input.cuda_workspace_handoff != 0U) {
        consumed_handoff_stage = recovered_cuda_workspace_handoff_stage(
            input.cuda_workspace_handoff);
        if (!using_cached_workspace ||
            (consumed_handoff_stage != 4U &&
             consumed_handoff_stage != 6U)) {
            if (error.empty())
                error = "recovered CUDA refinement handoff requires stage 4 or cross-level stage 6";
            release_recovered_cuda_cost_workspace_lease();
            cuModuleUnload(module);
            return false;
        }
        consume_pipeline_handoff = consumed_handoff_stage == 4U;
        consume_cross_level_handoff = consumed_handoff_stage == 6U;
    }
    const bool defer_host_output =
        input.defer_cuda_host_output && using_cached_workspace;

    void* device_depth = nullptr;
    void* device_normal = nullptr;
    void* device_cost = nullptr;
    void* device_coarse_depth = nullptr;
    void* device_coarse_radius = nullptr;
    void* device_reference = nullptr;
    void* device_tmp_depth = nullptr;
    void* device_tmp_normal = nullptr;
    if (using_cached_workspace) {
        device_depth = workspace_handles.fixed[cost_depth_slot];
        device_normal = workspace_handles.fixed[cost_normal_slot];
        device_cost = workspace_handles.fixed[cost_main_cost_slot];
        device_coarse_depth = workspace_handles.fixed[cost_coarse_depth_slot];
        device_coarse_radius = workspace_handles.fixed[cost_coarse_radius_slot];
        device_reference = workspace_handles.fixed[cost_reference_slot];
        device_tmp_depth = workspace_handles.fixed[cost_candidate_depth_slot];
        device_tmp_normal = workspace_handles.fixed[cost_candidate_normal_slot];
    }
    auto release = [&]() {
        if (!completed_handoff)
            invalidate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff);
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            cudaFree(device_depth);
            cudaFree(device_normal);
            cudaFree(device_cost);
            cudaFree(device_coarse_depth);
            cudaFree(device_coarse_radius);
            cudaFree(device_reference);
            cudaFree(device_tmp_depth);
            cudaFree(device_tmp_normal);
        }
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = *destination == nullptr
            ? cudaMalloc(destination, bytes)
            : cudaSuccess;
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice);
        return status;
    };
    const auto bytes_equal = [](const void* data, std::size_t bytes,
                                std::uint8_t expected) {
        const auto* values = static_cast<const std::uint8_t*>(data);
        return std::all_of(values, values + bytes,
                           [expected](std::uint8_t value) {
                               return value == expected;
                           });
    };
    if (input.initial_cuda_uniform_state &&
        (consume_pipeline_handoff || consume_cross_level_handoff ||
         !bytes_equal(depth.data(), depth.size() * sizeof(float), 0U) ||
         !bytes_equal(normal.data(), normal.size(), 0x80U) ||
         !bytes_equal(coarse_depth.data(),
                      coarse_depth.size() * sizeof(float), 0U) ||
         !bytes_equal(coarse_radius.data(),
                      coarse_radius.size() * sizeof(float), 0U) ||
         !bytes_equal(output.depth.data(),
                      output.depth.size() * sizeof(float), 0U) ||
         !bytes_equal(output.normal.data(),
                      output.normal.size() * sizeof(float), 0U))) {
        error = "uniform recovered CUDA refinement state assertion failed";
        release();
        return false;
    }
    if (!consume_pipeline_handoff && !consume_cross_level_handoff) {
        runtime_status = allocate_and_copy(&device_depth, depth.data(),
                                           depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, normal.data(), normal.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, main_cost.data(),
                main_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_depth, coarse_depth.data(),
                coarse_depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_radius, coarse_radius.data(),
                coarse_radius.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_reference, reference_image.data(),
                reference_image.size());
        if (runtime_status == cudaSuccess)
            runtime_status = input.initial_cuda_uniform_state
                ? cudaMemset(device_tmp_depth, 0,
                             output.depth.size() * sizeof(float))
                : allocate_and_copy(
                      &device_tmp_depth, output.depth.data(),
                      output.depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = input.initial_cuda_uniform_state
                ? cudaMemset(device_tmp_normal, 0,
                             output.normal.size() * sizeof(float))
                : allocate_and_copy(
                      &device_tmp_normal, output.normal.data(),
                      output.normal.size() * sizeof(float));
        if (runtime_status == cudaSuccess &&
            input.initial_cuda_uniform_state) {
            record_recovered_cuda_workspace_skipped_h2d(
                2U, output.depth.size() * sizeof(float) +
                        output.normal.size() * sizeof(float));
        }
    } else if (consume_cross_level_handoff) {
        runtime_status = allocate_and_copy(&device_depth, depth.data(),
                                           depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, normal.data(), normal.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, main_cost.data(),
                main_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_depth, coarse_depth.data(),
                coarse_depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_radius, coarse_radius.data(),
                coarse_radius.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_reference, reference_image.data(),
                reference_image.size());
        record_recovered_cuda_workspace_skipped_h2d(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            8U,
            depth.size() * sizeof(float) + normal.size() +
                main_cost.size() * sizeof(float) +
                coarse_depth.size() * sizeof(float) +
                coarse_radius.size() * sizeof(float) +
                reference_image.size() +
                output.depth.size() * sizeof(float) +
                output.normal.size() * sizeof(float));
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered PatchMatch buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    float depth_min = input.depth_min;
    float depth_max = input.depth_max;
    std::uint32_t detailed = input.image_one_step_more_detailed;
    float deviation = input.deviation_threshold_multiplier;
    std::uint32_t iteration = input.iteration;
    std::uint32_t fourth = input.only_each_fourth_pixel;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth, &device_normal, &device_cost, &device_coarse_depth,
        &device_coarse_radius, const_cast<PatchMatchCamera*>(&input.camera),
        &downscale32, &depth_min, &depth_max, &device_reference, &detailed,
        &deviation, &device_tmp_depth, &device_tmp_normal, &iteration, &fourth,
        &offset,
    };
    const unsigned int blocks = static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }
    if (!defer_host_output) {
        runtime_status = cudaMemcpy(output.depth.data(), device_tmp_depth,
                                    output.depth.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(output.normal.data(), device_tmp_normal,
                                        output.normal.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost);
    } else {
        output.cuda_host_output_materialized = false;
        record_recovered_cuda_workspace_skipped_d2h(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered PatchMatch output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (using_cached_workspace) {
        output.cuda_workspace_handoff =
            input.cuda_workspace_handoff != 0U
                ? publish_recovered_cuda_workspace_handoff(
                      consumed_handoff_stage,
                      input.cuda_workspace_handoff, 5U)
                : publish_recovered_cuda_workspace_handoff(0U, 0U, 3U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing recovered CUDA refinement workspace handoff failed";
            release();
            return false;
        }
        output.cuda_resident_cost_scratch =
            consume_pipeline_handoff || consume_cross_level_handoff;
        completed_handoff = true;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_final_refinement_cuda_impl(
    PatchMatchFinalRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::vector<float>* updated_cost,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::producer);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch final-refinement depth downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    const std::span<const std::uint8_t> reference_image =
        input.reference_image_view.empty()
            ? std::span<const std::uint8_t>(input.reference_image)
            : input.reference_image_view;
    const std::span<const float> depth = input.depth_view.empty()
        ? std::span<const float>(input.depth) : input.depth_view;
    const std::span<const std::uint8_t> normal = input.normal_view.empty()
        ? std::span<const std::uint8_t>(input.normal) : input.normal_view;
    const std::span<const float> main_cost = input.cost_view.empty()
        ? std::span<const float>(input.cost) : input.cost_view;
    if (pixels == 0 || work_items == 0 ||
        work_items > PatchMatchCandidateOutput::capacity ||
        input.pixel_offset + work_items > pixels ||
        depth.size() != pixels || normal.size() != pixels * 3 ||
        main_cost.size() != pixels || reference_image.empty()) {
        error = "PatchMatch final-refinement launch or buffers are invalid";
        return false;
    }
    if (input.initial_candidate_depth.empty() !=
            input.initial_candidate_normal.empty() ||
        (!input.initial_candidate_depth.empty() &&
         (input.initial_candidate_depth.size() !=
              PatchMatchCandidateOutput::hypotheses *
                  PatchMatchCandidateOutput::capacity ||
          input.initial_candidate_normal.size() !=
              PatchMatchCandidateOutput::hypotheses *
                  PatchMatchCandidateOutput::capacity * 3))) {
        error = "PatchMatch final-refinement initial candidates are invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA final-refinement context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status =
        cuModuleLoad(&module, METMODEL_PM_FINAL_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda19pm_refinement_finalIhEEvPfP6uchar3S1_13PinholeCamerajPKT_jfS1_P6float3j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch final-refinement PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.depth = input.initial_candidate_depth.empty()
        ? std::vector<float>(PatchMatchCandidateOutput::hypotheses *
                                 PatchMatchCandidateOutput::capacity,
                             std::numeric_limits<float>::quiet_NaN())
        : std::move(input.initial_candidate_depth);
    output.normal = input.initial_candidate_normal.empty()
        ? std::vector<float>(PatchMatchCandidateOutput::hypotheses *
                                 PatchMatchCandidateOutput::capacity * 3,
                             std::numeric_limits<float>::quiet_NaN())
        : std::move(input.initial_candidate_normal);
    output.cuda_workspace_handoff = 0U;
    output.cuda_resident_cost_scratch = false;
    output.cuda_host_output_materialized = true;
    const std::array<std::size_t,
                     RecoveredCudaCostWorkspace::fixed_allocation_count>
        workspace_bytes = {
            depth.size() * sizeof(float),
            normal.size(),
            main_cost.size() * sizeof(float),
            0U,
            0U,
            reference_image.size(),
            output.depth.size() * sizeof(float),
            output.normal.size() * sizeof(float),
        };
    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_pipeline_fixed_workspace(
            workspace_bytes, RecoveredCudaWorkspaceClient::producer,
            workspace_handles, using_cached_workspace, error)) {
        cuModuleUnload(module);
        return false;
    }

    bool completed_handoff = false;
    bool consume_pipeline_handoff = false;
    bool consume_cross_level_handoff = false;
    std::uint32_t consumed_handoff_stage = 0U;
    if (input.cuda_workspace_handoff != 0U) {
        consumed_handoff_stage = recovered_cuda_workspace_handoff_stage(
            input.cuda_workspace_handoff);
        if (!using_cached_workspace ||
            (consumed_handoff_stage != 4U &&
             consumed_handoff_stage != 6U)) {
            if (error.empty())
                error = "recovered CUDA final-refinement handoff requires stage 4 or cross-level stage 6";
            release_recovered_cuda_cost_workspace_lease();
            cuModuleUnload(module);
            return false;
        }
        consume_pipeline_handoff = consumed_handoff_stage == 4U;
        consume_cross_level_handoff = consumed_handoff_stage == 6U;
    }
    const bool defer_host_output =
        input.defer_cuda_host_output && using_cached_workspace;

    void* device_depth = nullptr;
    void* device_normal = nullptr;
    void* device_cost = nullptr;
    void* device_reference = nullptr;
    void* device_tmp_depth = nullptr;
    void* device_tmp_normal = nullptr;
    if (using_cached_workspace) {
        device_depth = workspace_handles.fixed[cost_depth_slot];
        device_normal = workspace_handles.fixed[cost_normal_slot];
        device_cost = workspace_handles.fixed[cost_main_cost_slot];
        device_reference = workspace_handles.fixed[cost_reference_slot];
        device_tmp_depth = workspace_handles.fixed[cost_candidate_depth_slot];
        device_tmp_normal = workspace_handles.fixed[cost_candidate_normal_slot];
    }
    auto release = [&]() {
        if (!completed_handoff)
            invalidate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff);
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            cudaFree(device_depth);
            cudaFree(device_normal);
            cudaFree(device_cost);
            cudaFree(device_reference);
            cudaFree(device_tmp_depth);
            cudaFree(device_tmp_normal);
        }
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = *destination == nullptr
            ? cudaMalloc(destination, bytes)
            : cudaSuccess;
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    if (!consume_pipeline_handoff && !consume_cross_level_handoff) {
        runtime_status = allocate_and_copy(
            &device_depth, depth.data(), depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, normal.data(), normal.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, main_cost.data(),
                main_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_reference, reference_image.data(),
                reference_image.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_depth, output.depth.data(),
                output.depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_normal, output.normal.data(),
                output.normal.size() * sizeof(float));
    } else if (consume_cross_level_handoff) {
        runtime_status = allocate_and_copy(
            &device_depth, depth.data(), depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, normal.data(), normal.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, main_cost.data(),
                main_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_reference, reference_image.data(),
                reference_image.size());
        record_recovered_cuda_workspace_skipped_h2d(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            6U,
            depth.size() * sizeof(float) + normal.size() +
                main_cost.size() * sizeof(float) +
                reference_image.size() +
                output.depth.size() * sizeof(float) +
                output.normal.size() * sizeof(float));
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered final-refinement buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t detailed = input.image_one_step_more_detailed;
    float deviation = input.deviation_threshold_multiplier;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth, &device_normal, &device_cost,
        const_cast<PatchMatchCamera*>(&input.camera), &downscale32,
        &device_reference, &detailed, &deviation, &device_tmp_depth,
        &device_tmp_normal, &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch final-refinement launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }
    if (!defer_host_output) {
        runtime_status = cudaMemcpy(output.depth.data(), device_tmp_depth,
                                    output.depth.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(output.normal.data(), device_tmp_normal,
                                        output.normal.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost);
    } else {
        output.cuda_host_output_materialized = false;
        record_recovered_cuda_workspace_skipped_d2h(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    }
    if (runtime_status == cudaSuccess && updated_cost != nullptr) {
        if (input.cost_view.empty()) {
            *updated_cost = std::move(input.cost);
        } else {
            updated_cost->assign(main_cost.begin(), main_cost.end());
        }
        if (!defer_host_output) {
            runtime_status = cudaMemcpy(
                updated_cost->data(), device_cost,
                updated_cost->size() * sizeof(float),
                cudaMemcpyDeviceToHost);
        } else {
            record_recovered_cuda_workspace_skipped_d2h(
                1U, updated_cost->size() * sizeof(float));
        }
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered final-refinement output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (using_cached_workspace) {
        output.cuda_workspace_handoff =
            input.cuda_workspace_handoff != 0U
                ? publish_recovered_cuda_workspace_handoff(
                      consumed_handoff_stage,
                      input.cuda_workspace_handoff, 5U)
                : publish_recovered_cuda_workspace_handoff(0U, 0U, 3U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing recovered CUDA final-refinement workspace handoff failed";
            release();
            return false;
        }
        output.cuda_resident_cost_scratch =
            consume_pipeline_handoff || consume_cross_level_handoff;
        completed_handoff = true;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_wta_cuda_impl(
    PatchMatchWtaInput& input,
    PatchMatchWtaOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::wta);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch WTA depth downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    const std::size_t capacity = PatchMatchCandidateOutput::capacity;
    if (pixels == 0 || work_items == 0 || work_items > capacity ||
        input.hypotheses_per_pixel == 0 ||
        input.hypotheses_per_pixel > PatchMatchCandidateOutput::hypotheses ||
        input.depth.size() != pixels || input.normal.size() != pixels * 3 ||
        input.cost.size() != pixels ||
        input.candidates.depth.size() !=
            PatchMatchCandidateOutput::hypotheses * capacity ||
        input.candidates.normal.size() !=
            PatchMatchCandidateOutput::hypotheses * capacity * 3 ||
        input.average_cost.size() !=
            PatchMatchCandidateOutput::hypotheses * capacity ||
        (!input.winner.empty() && input.winner.size() != pixels &&
         input.winner.size() != capacity)) {
        error = "PatchMatch WTA launch or buffer dimensions are invalid";
        return false;
    }
    if (input.cuda_workspace_handoff == 0U &&
        (!input.cuda_cost_output_materialized ||
         !input.candidates.cuda_host_output_materialized)) {
        error = "unmaterialized recovered CUDA cost output requires a cost-to-WTA handoff";
        return false;
    }
    if (input.defer_cuda_host_output &&
        input.cuda_workspace_handoff == 0U) {
        error = "deferred recovered CUDA WTA output requires a cost-to-WTA handoff";
        return false;
    }
    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA WTA context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_WTA_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda23pm_choose_best_hypo_wtaEPfP6uchar3S0_13PinholeCamerajjjPKfPK6float3S5_Phjjjj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch WTA PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.depth = std::move(input.depth);
    output.normal = std::move(input.normal);
    output.cost = std::move(input.cost);
    output.winner = std::move(input.winner);
    output.winner.resize(capacity, 0U);
    output.cuda_host_output_materialized = true;
    void* device_depth = nullptr;
    void* device_normal = nullptr;
    void* device_cost = nullptr;
    void* device_tmp_depth = nullptr;
    void* device_tmp_normal = nullptr;
    void* device_avg_cost = nullptr;
    void* device_winner = nullptr;
    std::array<std::size_t,
               RecoveredCudaCostWorkspace::fixed_allocation_count>
        workspace_bytes{};
    workspace_bytes[cost_depth_slot] = output.depth.size() * sizeof(float);
    workspace_bytes[cost_normal_slot] = output.normal.size();
    workspace_bytes[cost_main_cost_slot] =
        output.cost.size() * sizeof(float);
    workspace_bytes[cost_candidate_depth_slot] =
        input.candidates.depth.size() * sizeof(float);
    workspace_bytes[cost_candidate_normal_slot] =
        input.candidates.normal.size() * sizeof(float);
    workspace_bytes[cost_average_slot] =
        input.average_cost.size() * sizeof(float);
    workspace_bytes[pipeline_winner_slot] = output.winner.size();
    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_pipeline_fixed_workspace(
            workspace_bytes, RecoveredCudaWorkspaceClient::wta,
            workspace_handles,
            using_cached_workspace, error)) {
        cuModuleUnload(module);
        return false;
    }
    if (using_cached_workspace) {
        device_depth = workspace_handles.fixed[cost_depth_slot];
        device_normal = workspace_handles.fixed[cost_normal_slot];
        device_cost = workspace_handles.fixed[cost_main_cost_slot];
        device_tmp_depth =
            workspace_handles.fixed[cost_candidate_depth_slot];
        device_tmp_normal =
            workspace_handles.fixed[cost_candidate_normal_slot];
        device_avg_cost = workspace_handles.fixed[cost_average_slot];
        device_winner = workspace_handles.fixed[pipeline_winner_slot];
    }
    bool completed_handoff = false;
    bool consume_cost_handoff = false;
    bool consume_resident_winner = false;
    if (input.cuda_workspace_handoff != 0U) {
        if (!using_cached_workspace ||
            !validate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff, 1U, error)) {
            if (error.empty())
                error = "recovered CUDA cost-to-WTA handoff requires an active workspace";
            release_recovered_cuda_cost_workspace_lease();
            cuModuleUnload(module);
            return false;
        }
        consume_cost_handoff = true;
        consume_resident_winner =
            recovered_cuda_workspace_handoff_winner_is_valid(
                input.cuda_workspace_handoff, 1U);
    }
    const bool defer_host_output =
        input.defer_cuda_host_output && consume_cost_handoff &&
        using_cached_workspace;
    auto release = [&]() {
        if (!completed_handoff)
            invalidate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff);
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            cudaFree(device_depth);
            cudaFree(device_normal);
            cudaFree(device_cost);
            cudaFree(device_tmp_depth);
            cudaFree(device_tmp_normal);
            cudaFree(device_avg_cost);
            cudaFree(device_winner);
        }
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = *destination == nullptr
            ? cudaMalloc(destination, bytes)
            : cudaSuccess;
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice);
        return status;
    };
    if (!consume_cost_handoff) {
        runtime_status = allocate_and_copy(&device_depth, output.depth.data(),
                                           output.depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, output.normal.data(), output.normal.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, output.cost.data(),
                output.cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_depth, input.candidates.depth.data(),
                input.candidates.depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_normal, input.candidates.normal.data(),
                input.candidates.normal.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_avg_cost, input.average_cost.data(),
                input.average_cost.size() * sizeof(float));
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            6U,
            output.depth.size() * sizeof(float) + output.normal.size() +
                output.cost.size() * sizeof(float) +
                input.candidates.depth.size() * sizeof(float) +
                input.candidates.normal.size() * sizeof(float) +
                input.average_cost.size() * sizeof(float));
    }
    if (runtime_status == cudaSuccess && !consume_resident_winner)
        runtime_status = allocate_and_copy(&device_winner, output.winner.data(),
                                           output.winner.size());
    if (runtime_status == cudaSuccess && consume_resident_winner)
        record_recovered_cuda_workspace_skipped_h2d(
            1U, output.winner.size());
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered PatchMatch WTA buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t detailed = input.image_one_step_more_detailed;
    std::uint32_t hypotheses = input.hypotheses_per_pixel;
    std::uint32_t checkboard = input.is_checkboard;
    std::uint32_t checkboard_step = input.checkboard_step;
    std::uint32_t fourth = input.only_each_fourth_pixel;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth, &device_normal, &device_cost,
        const_cast<PatchMatchCamera*>(&input.camera), &downscale32, &detailed,
        &hypotheses, &device_tmp_depth, &device_tmp_normal, &device_avg_cost,
        &device_winner, &checkboard, &checkboard_step, &fourth, &offset,
    };
    const unsigned int blocks = static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch WTA launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }
    if (!defer_host_output) {
        runtime_status = cudaMemcpy(output.depth.data(), device_depth,
                                    output.depth.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.normal.data(), device_normal,
                output.normal.size(), cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.cost.data(), device_cost,
                output.cost.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.winner.data(), device_winner,
                output.winner.size(), cudaMemcpyDeviceToHost);
    } else {
        output.cuda_host_output_materialized = false;
        record_recovered_cuda_workspace_skipped_d2h(
            4U, output.depth.size() * sizeof(float) + output.normal.size() +
                    output.cost.size() * sizeof(float) +
                    output.winner.size());
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered PatchMatch WTA output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (consume_cost_handoff) {
        output.cuda_workspace_handoff =
            publish_recovered_cuda_workspace_handoff(
                1U, input.cuda_workspace_handoff, 2U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing recovered CUDA WTA workspace handoff failed";
            release();
            return false;
        }
        completed_handoff = true;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_copy_inlier_masks_cuda_impl(
    PatchMatchCopyInlierMasksInput& input,
    PatchMatchCopyInlierMasksOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::inlier);
    constexpr std::size_t capacity = PatchMatchCandidateOutput::capacity;
    if (input.width == 0U || input.height == 0U ||
        (input.hypotheses_per_pixel != 2U &&
         input.hypotheses_per_pixel != 8U) ||
        input.neighbor_count == 0U || input.neighbor_count > 16U ||
        input.is_checkboard > 1U || input.checkboard_step > 1U ||
        input.only_each_fourth_pixel > 1U ||
        input.global_work_items == 0U ||
        input.global_work_items > capacity ||
        input.device_index >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        static_cast<std::uint64_t>(input.pixel_offset) +
                input.global_work_items + 127U >
            std::numeric_limits<std::uint32_t>::max()) {
        error = "PatchMatch inlier-mask scalar parameters are invalid";
        return false;
    }
    const std::size_t pixels =
        static_cast<std::size_t>(input.width) * input.height;
    const std::size_t groups =
        (static_cast<std::size_t>(input.neighbor_count) + 7U) / 8U;
    if (pixels > std::numeric_limits<std::size_t>::max() / groups) {
        error = "PatchMatch inlier-mask output dimensions overflow";
        return false;
    }
    const std::size_t temporary_bytes =
        groups * input.hypotheses_per_pixel * capacity;
    const std::size_t active_output_bytes = groups * pixels;
    if (active_output_bytes > std::numeric_limits<std::uint32_t>::max() ||
        (input.only_each_fourth_pixel != 0U &&
         input.width > std::numeric_limits<std::uint32_t>::max() -
                           (input.is_checkboard != 0U ? 3U : 1U)) ||
        (input.is_checkboard != 0U &&
         input.only_each_fourth_pixel == 0U &&
         input.width == std::numeric_limits<std::uint32_t>::max())) {
        error = "PatchMatch inlier-mask device index arithmetic would overflow";
        return false;
    }
    if (input.temporary_inlier_masks.size() < temporary_bytes ||
        input.winner.size() != capacity ||
        input.initial_neighbor_inlier_masks.size() < active_output_bytes) {
        error = "PatchMatch inlier-mask buffer dimensions are invalid";
        return false;
    }
    if (!input.cuda_temporary_inlier_masks_materialized &&
        input.cuda_workspace_handoff == 0U) {
        error = "unmaterialized recovered CUDA inlier scratch requires a WTA handoff";
        return false;
    }
    if (!input.cuda_winner_materialized &&
        input.cuda_workspace_handoff == 0U) {
        error = "unmaterialized recovered CUDA winner requires a WTA handoff";
        return false;
    }
    if (!input.cuda_initial_neighbor_inlier_masks_materialized &&
        input.cuda_workspace_handoff == 0U) {
        error = "unmaterialized recovered CUDA inlier-mask backing requires a WTA handoff";
        return false;
    }
    if (input.defer_cuda_host_output &&
        input.cuda_workspace_handoff == 0U) {
        error = "deferred recovered CUDA inlier-mask output requires a WTA handoff";
        return false;
    }

    // The device kernel intentionally has no winner-range guard.  Reject an
    // invalid active winner on the host so malformed input cannot address past
    // the evidence-backed scratch allocation.  Coordinates rejected by the
    // target kernel are not inspected here.
    const std::size_t rounded_work_items =
        ((input.global_work_items + 127U) / 128U) * 128U;
    for (std::size_t temporary_index = 0;
         input.cuda_winner_materialized &&
         temporary_index < rounded_work_items; ++temporary_index) {
        const std::uint64_t global_index =
            static_cast<std::uint64_t>(input.pixel_offset) + temporary_index;
        std::uint64_t x = 0;
        std::uint64_t y = 0;
        if (input.is_checkboard != 0U) {
            if (input.only_each_fourth_pixel == 0U) {
                const std::uint64_t width_half = (input.width + 1U) / 2U;
                y = global_index / width_half;
                x = ((y % 2U) + input.checkboard_step) % 2U +
                    2U * (global_index % width_half);
            } else {
                if (global_index >
                    std::numeric_limits<std::uint32_t>::max() / 2U) {
                    error = "PatchMatch inlier-mask coordinate arithmetic would overflow";
                    return false;
                }
                const std::uint64_t width_part = (input.width + 3U) / 4U;
                y = 1U + 2U * global_index / width_part;
                x = 1U + (((y / 2U) % 2U + input.checkboard_step) % 2U) * 2U +
                    4U * (global_index % width_part);
            }
        } else if (input.only_each_fourth_pixel == 0U) {
            y = global_index / input.width;
            x = global_index % input.width;
        } else {
            const std::uint64_t width_half = (input.width + 1U) / 2U;
            const std::uint64_t row = global_index / width_half;
            if (row >
                (std::numeric_limits<std::uint32_t>::max() - 1U) / 2U) {
                error = "PatchMatch inlier-mask coordinate arithmetic would overflow";
                return false;
            }
            y = 1U + 2U * row;
            x = 1U + 2U * (global_index % width_half);
        }
        if (x >= input.width || y >= input.height) continue;
        if (temporary_index >= input.global_work_items) {
            error = "PatchMatch inlier-mask launch padding reaches an active pixel";
            return false;
        }
        const std::uint8_t winner = input.winner[temporary_index];
        if (winner != 255U && winner >= input.hypotheses_per_pixel) {
            error = "PatchMatch inlier-mask active winner is out of range";
            return false;
        }
    }

    cudaError_t runtime_status =
        cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA inlier-mask context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_WTA_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda29pm_copy_inliers_masks_wrt_wtaEPhjjjjPKhS2_jjjj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch inlier-mask PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.neighbor_inlier_masks =
        std::move(input.initial_neighbor_inlier_masks);
    output.cuda_workspace_handoff = 0U;
    output.cuda_host_output_materialized = true;
    void* device_output = nullptr;
    void* device_temporary = nullptr;
    void* device_winner = nullptr;
    std::array<std::size_t,
               RecoveredCudaCostWorkspace::fixed_allocation_count>
        workspace_bytes{};
    workspace_bytes[pipeline_inlier_masks_slot] =
        output.neighbor_inlier_masks.size();
    workspace_bytes[cost_auxiliary_slot] = temporary_bytes;
    workspace_bytes[pipeline_winner_slot] = input.winner.size();
    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_pipeline_fixed_workspace(
            workspace_bytes, RecoveredCudaWorkspaceClient::inlier,
            workspace_handles,
            using_cached_workspace, error)) {
        cuModuleUnload(module);
        return false;
    }
    if (using_cached_workspace) {
        device_output =
            workspace_handles.fixed[pipeline_inlier_masks_slot];
        device_temporary = workspace_handles.fixed[cost_auxiliary_slot];
        device_winner = workspace_handles.fixed[pipeline_winner_slot];
    }
    bool completed_handoff = false;
    bool consume_wta_handoff = false;
    bool consume_resident_inlier_masks = false;
    if (input.cuda_workspace_handoff != 0U) {
        if (!using_cached_workspace ||
            !validate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff, 2U, error)) {
            if (error.empty())
                error = "recovered CUDA WTA-to-inlier handoff requires an active workspace";
            release_recovered_cuda_cost_workspace_lease();
            cuModuleUnload(module);
            return false;
        }
        consume_wta_handoff = true;
        consume_resident_inlier_masks =
            recovered_cuda_workspace_handoff_inlier_masks_are_valid(
                input.cuda_workspace_handoff, 2U);
    }
    if (!input.cuda_initial_neighbor_inlier_masks_materialized &&
        !consume_resident_inlier_masks) {
        error = "unmaterialized recovered CUDA inlier-mask backing is not resident";
        release_recovered_cuda_cost_workspace_lease();
        cuModuleUnload(module);
        return false;
    }
    const bool defer_host_output =
        input.defer_cuda_host_output && consume_wta_handoff &&
        using_cached_workspace;
    auto release = [&]() {
        if (!completed_handoff)
            invalidate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff);
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            cudaFree(device_output);
            cudaFree(device_temporary);
            cudaFree(device_winner);
        }
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = *destination == nullptr
            ? cudaMalloc(destination, bytes)
            : cudaSuccess;
        if (status == cudaSuccess)
            status = cudaMemcpy(
                *destination, source, bytes, cudaMemcpyHostToDevice);
        return status;
    };
    if (!consume_resident_inlier_masks)
        runtime_status = allocate_and_copy(
            &device_output, output.neighbor_inlier_masks.data(),
            output.neighbor_inlier_masks.size());
    else
        record_recovered_cuda_workspace_skipped_h2d(
            1U, output.neighbor_inlier_masks.size());
    if (!consume_wta_handoff) {
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_temporary, input.temporary_inlier_masks.data(),
                temporary_bytes);
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_winner, input.winner.data(), input.winner.size());
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            2U, temporary_bytes + input.winner.size());
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered PatchMatch inlier-mask buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t width = input.width;
    std::uint32_t height = input.height;
    std::uint32_t hypotheses = input.hypotheses_per_pixel;
    std::uint32_t neighbors = input.neighbor_count;
    std::uint32_t checkboard = input.is_checkboard;
    std::uint32_t checkboard_step = input.checkboard_step;
    std::uint32_t fourth = input.only_each_fourth_pixel;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_output, &width, &height, &hypotheses, &neighbors,
        &device_temporary, &device_winner, &checkboard, &checkboard_step,
        &fourth, &offset,
    };
    const unsigned int blocks = static_cast<unsigned int>(
        (input.global_work_items + 127U) / 128U);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch inlier-mask launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }
    if (!defer_host_output)
        runtime_status = cudaMemcpy(
            output.neighbor_inlier_masks.data(), device_output,
            output.neighbor_inlier_masks.size(), cudaMemcpyDeviceToHost);
    else {
        output.cuda_host_output_materialized = false;
        record_recovered_cuda_workspace_skipped_d2h(
            1U, output.neighbor_inlier_masks.size());
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered PatchMatch inlier-mask output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (consume_wta_handoff) {
        output.cuda_workspace_handoff =
            publish_recovered_cuda_workspace_handoff(
                2U, input.cuda_workspace_handoff, 4U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing recovered CUDA inlier workspace handoff failed";
            release();
            return false;
        }
        completed_handoff = true;
    }
    release();
    error.clear();
    return true;
}

bool run_recovered_patchmatch_propagation_cuda_impl(
    PatchMatchPropagationInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::producer);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch propagation depth downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    const std::span<const std::uint8_t> reference_image =
        input.reference_image_view.empty()
            ? std::span<const std::uint8_t>(input.reference_image)
            : input.reference_image_view;
    const std::span<const float> depth = input.depth_view.empty()
        ? std::span<const float>(input.depth) : input.depth_view;
    const std::span<const std::uint8_t> normal = input.normal_view.empty()
        ? std::span<const std::uint8_t>(input.normal) : input.normal_view;
    const std::span<const float> main_cost = input.cost_view.empty()
        ? std::span<const float>(input.cost) : input.cost_view;
    const std::span<const float> coarse_depth =
        input.coarse_depth_view.empty()
            ? std::span<const float>(input.coarse_depth)
            : input.coarse_depth_view;
    const std::span<const float> coarse_radius =
        input.coarse_depth_radius_view.empty()
            ? std::span<const float>(input.coarse_depth_radius)
            : input.coarse_depth_radius_view;
    if (pixels == 0 || work_items == 0 ||
        work_items > PatchMatchCandidateOutput::capacity ||
        depth.size() != pixels || normal.size() != pixels * 3 ||
        main_cost.size() != pixels || coarse_depth.size() != pixels ||
        coarse_radius.size() != pixels || reference_image.empty() ||
        input.initial_candidate_depth.empty() !=
            input.initial_candidate_normal.empty() ||
        (!input.initial_candidate_depth.empty() &&
         (input.initial_candidate_depth.size() !=
              PatchMatchCandidateOutput::hypotheses *
                  PatchMatchCandidateOutput::capacity ||
          input.initial_candidate_normal.size() !=
              PatchMatchCandidateOutput::hypotheses *
                  PatchMatchCandidateOutput::capacity * 3))) {
        error = "PatchMatch propagation launch or buffer dimensions are invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA propagation context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status =
        cuModuleLoad(&module, METMODEL_PM_PROPAGATION_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda14pm_propagationIhEEvPfP6uchar3S1_S1_S1_13PinholeCameraNS_10Matrix3x3fEjPKT_jfS1_P6float3jjj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch propagation PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.depth = input.initial_candidate_depth.empty()
        ? std::vector<float>(PatchMatchCandidateOutput::hypotheses *
                                 PatchMatchCandidateOutput::capacity,
                             std::numeric_limits<float>::quiet_NaN())
        : std::move(input.initial_candidate_depth);
    output.normal = input.initial_candidate_normal.empty()
        ? std::vector<float>(PatchMatchCandidateOutput::hypotheses *
                                 PatchMatchCandidateOutput::capacity * 3,
                             std::numeric_limits<float>::quiet_NaN())
        : std::move(input.initial_candidate_normal);
    output.cuda_workspace_handoff = 0U;
    output.cuda_resident_cost_scratch = false;
    output.cuda_host_output_materialized = true;
    const std::array<std::size_t,
                     RecoveredCudaCostWorkspace::fixed_allocation_count>
        workspace_bytes = {
            depth.size() * sizeof(float),
            normal.size(),
            main_cost.size() * sizeof(float),
            coarse_depth.size() * sizeof(float),
            coarse_radius.size() * sizeof(float),
            reference_image.size(),
            output.depth.size() * sizeof(float),
            output.normal.size() * sizeof(float),
        };
    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_pipeline_fixed_workspace(
            workspace_bytes, RecoveredCudaWorkspaceClient::producer,
            workspace_handles, using_cached_workspace, error)) {
        cuModuleUnload(module);
        return false;
    }

    bool completed_handoff = false;
    bool consume_pipeline_handoff = false;
    bool consume_cross_level_handoff = false;
    std::uint32_t consumed_handoff_stage = 0U;
    if (input.cuda_workspace_handoff != 0U) {
        consumed_handoff_stage = recovered_cuda_workspace_handoff_stage(
            input.cuda_workspace_handoff);
        if (!using_cached_workspace ||
            (consumed_handoff_stage != 4U &&
             consumed_handoff_stage != 6U)) {
            if (error.empty())
                error = "recovered CUDA propagation handoff requires stage 4 or cross-level stage 6";
            release_recovered_cuda_cost_workspace_lease();
            cuModuleUnload(module);
            return false;
        }
        consume_pipeline_handoff = consumed_handoff_stage == 4U;
        consume_cross_level_handoff = consumed_handoff_stage == 6U;
    }
    const bool defer_host_output =
        input.defer_cuda_host_output && using_cached_workspace;

    void* device_depth = nullptr;
    void* device_normal = nullptr;
    void* device_cost = nullptr;
    void* device_coarse_depth = nullptr;
    void* device_coarse_radius = nullptr;
    void* device_reference = nullptr;
    void* device_tmp_depth = nullptr;
    void* device_tmp_normal = nullptr;
    if (using_cached_workspace) {
        device_depth = workspace_handles.fixed[cost_depth_slot];
        device_normal = workspace_handles.fixed[cost_normal_slot];
        device_cost = workspace_handles.fixed[cost_main_cost_slot];
        device_coarse_depth = workspace_handles.fixed[cost_coarse_depth_slot];
        device_coarse_radius = workspace_handles.fixed[cost_coarse_radius_slot];
        device_reference = workspace_handles.fixed[cost_reference_slot];
        device_tmp_depth = workspace_handles.fixed[cost_candidate_depth_slot];
        device_tmp_normal = workspace_handles.fixed[cost_candidate_normal_slot];
    }
    auto release = [&]() {
        if (!completed_handoff)
            invalidate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff);
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            cudaFree(device_depth);
            cudaFree(device_normal);
            cudaFree(device_cost);
            cudaFree(device_coarse_depth);
            cudaFree(device_coarse_radius);
            cudaFree(device_reference);
            cudaFree(device_tmp_depth);
            cudaFree(device_tmp_normal);
        }
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = *destination == nullptr
            ? cudaMalloc(destination, bytes)
            : cudaSuccess;
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    if (!consume_pipeline_handoff && !consume_cross_level_handoff) {
        runtime_status = allocate_and_copy(
            &device_depth, depth.data(), depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, normal.data(), normal.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, main_cost.data(),
                main_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_depth, coarse_depth.data(),
                coarse_depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_radius, coarse_radius.data(),
                coarse_radius.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_reference, reference_image.data(),
                reference_image.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_depth, output.depth.data(),
                output.depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_normal, output.normal.data(),
                output.normal.size() * sizeof(float));
    } else if (consume_cross_level_handoff) {
        runtime_status = allocate_and_copy(
            &device_depth, depth.data(), depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, normal.data(), normal.size());
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, main_cost.data(),
                main_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_depth, coarse_depth.data(),
                coarse_depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_coarse_radius, coarse_radius.data(),
                coarse_radius.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_reference, reference_image.data(),
                reference_image.size());
        record_recovered_cuda_workspace_skipped_h2d(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            8U,
            depth.size() * sizeof(float) + normal.size() +
                main_cost.size() * sizeof(float) +
                coarse_depth.size() * sizeof(float) +
                coarse_radius.size() * sizeof(float) +
                reference_image.size() +
                output.depth.size() * sizeof(float) +
                output.normal.size() * sizeof(float));
    }
    if (runtime_status != cudaSuccess) {
        error =
            std::string("allocating recovered PatchMatch propagation buffers failed: ") +
            cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t detailed = input.image_one_step_more_detailed;
    float deviation = input.deviation_threshold_multiplier;
    std::uint32_t checkboard_step = input.checkboard_step;
    std::uint32_t fourth = input.only_each_fourth_pixel;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth,
        &device_normal,
        &device_cost,
        &device_coarse_depth,
        &device_coarse_radius,
        const_cast<PatchMatchCamera*>(&input.camera),
        const_cast<float*>(input.reference_to_neighbor_rotation.data()),
        &downscale32,
        &device_reference,
        &detailed,
        &deviation,
        &device_tmp_depth,
        &device_tmp_normal,
        &checkboard_step,
        &fourth,
        &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch propagation launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }
    if (!defer_host_output) {
        runtime_status = cudaMemcpy(output.depth.data(), device_tmp_depth,
                                    output.depth.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(output.normal.data(), device_tmp_normal,
                                        output.normal.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost);
    } else {
        output.cuda_host_output_materialized = false;
        record_recovered_cuda_workspace_skipped_d2h(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered PatchMatch propagation output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (using_cached_workspace) {
        output.cuda_workspace_handoff =
            input.cuda_workspace_handoff != 0U
                ? publish_recovered_cuda_workspace_handoff(
                      consumed_handoff_stage,
                      input.cuda_workspace_handoff, 5U)
                : publish_recovered_cuda_workspace_handoff(0U, 0U, 3U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing recovered CUDA propagation workspace handoff failed";
            release();
            return false;
        }
        output.cuda_resident_cost_scratch =
            consume_pipeline_handoff || consume_cross_level_handoff;
        completed_handoff = true;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_coarse_to_precise_cuda_impl(
    PatchMatchCoarseToPreciseInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::coarse_to_precise);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch coarse-to-precise depth downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (pixels == 0 || work_items == 0 ||
        work_items > PatchMatchCandidateOutput::capacity ||
        input.pixel_offset >= pixels || input.depth.size() != pixels ||
        input.normal.size() != pixels * 3 || input.cost.size() != pixels) {
        error = "PatchMatch coarse-to-precise launch or buffers are invalid";
        return false;
    }
    const bool has_initial_candidates =
        !input.initial_candidates.depth.empty() ||
        !input.initial_candidates.normal.empty();
    const bool consume_pipeline_handoff =
        input.cuda_workspace_handoff != 0U;
    if (has_initial_candidates &&
        !input.initial_candidates.cuda_host_output_materialized &&
        !consume_pipeline_handoff) {
        error = "coarse-to-precise requires materialized host candidates or a resident handoff";
        return false;
    }
    if (!input.cuda_main_state_materialized && !consume_pipeline_handoff) {
        error = "coarse-to-precise requires materialized main state or a resident handoff";
        return false;
    }
    if (input.defer_cuda_host_output && !consume_pipeline_handoff) {
        error = "deferred coarse-to-precise output requires a resident handoff";
        return false;
    }
    if (has_initial_candidates &&
        (input.initial_candidates.depth.size() !=
             PatchMatchCandidateOutput::hypotheses *
                 PatchMatchCandidateOutput::capacity ||
         input.initial_candidates.normal.size() !=
             PatchMatchCandidateOutput::hypotheses *
                 PatchMatchCandidateOutput::capacity * 3)) {
        error = "PatchMatch coarse-to-precise initial scratch size is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA coarse-to-precise context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_COARSE_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda20pm_coarse_to_preciseEPfP6uchar3S0_13PinholeCamerajS0_P6float3j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch coarse-to-precise PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    std::array<std::size_t, RecoveredCudaCostWorkspace::fixed_allocation_count>
        fixed_bytes{};
    fixed_bytes[cost_depth_slot] = input.depth.size() * sizeof(float);
    fixed_bytes[cost_normal_slot] = input.normal.size();
    fixed_bytes[cost_main_cost_slot] = input.cost.size() * sizeof(float);
    fixed_bytes[cost_candidate_depth_slot] =
        PatchMatchCandidateOutput::hypotheses *
        PatchMatchCandidateOutput::capacity * sizeof(float);
    fixed_bytes[cost_candidate_normal_slot] =
        PatchMatchCandidateOutput::hypotheses *
        PatchMatchCandidateOutput::capacity * 3U * sizeof(float);
    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_pipeline_fixed_workspace(
            fixed_bytes, RecoveredCudaWorkspaceClient::coarse_to_precise,
            workspace_handles, using_cached_workspace, error,
            input.cuda_workspace_handoff,
            consume_pipeline_handoff ? 4U : 0U)) {
        cuModuleUnload(module);
        return false;
    }
    if (consume_pipeline_handoff && !using_cached_workspace) {
        error = "coarse-to-precise resident handoff requires an active CUDA workspace session";
        cuModuleUnload(module);
        return false;
    }
    void* device_depth = using_cached_workspace
        ? workspace_handles.fixed[cost_depth_slot] : nullptr;
    void* device_normal = using_cached_workspace
        ? workspace_handles.fixed[cost_normal_slot] : nullptr;
    void* device_cost = using_cached_workspace
        ? workspace_handles.fixed[cost_main_cost_slot] : nullptr;
    void* device_tmp_depth = using_cached_workspace
        ? workspace_handles.fixed[cost_candidate_depth_slot] : nullptr;
    void* device_tmp_normal = using_cached_workspace
        ? workspace_handles.fixed[cost_candidate_normal_slot] : nullptr;
    bool completed_handoff = false;
    auto release = [&]() {
        if (consume_pipeline_handoff && !completed_handoff)
            invalidate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff);
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            cudaFree(device_depth);
            cudaFree(device_normal);
            cudaFree(device_cost);
            cudaFree(device_tmp_depth);
            cudaFree(device_tmp_normal);
        }
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = *destination == nullptr
            ? cudaMalloc(destination, bytes)
            : cudaSuccess;
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    if (!consume_pipeline_handoff) {
        runtime_status = allocate_and_copy(
            &device_depth, input.depth.data(), fixed_bytes[cost_depth_slot]);
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, input.normal.data(),
                fixed_bytes[cost_normal_slot]);
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, input.cost.data(),
                fixed_bytes[cost_main_cost_slot]);
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            3U, fixed_bytes[cost_depth_slot] + fixed_bytes[cost_normal_slot] +
                    fixed_bytes[cost_main_cost_slot]);
    }

    if (has_initial_candidates) {
        output = std::move(input.initial_candidates);
        output.cuda_workspace_handoff = 0U;
        output.cuda_resident_cost_scratch = false;
        output.cuda_host_output_materialized =
            input.initial_candidates.cuda_host_output_materialized;
    } else {
        output.depth.assign(PatchMatchCandidateOutput::hypotheses *
                                PatchMatchCandidateOutput::capacity,
                            std::numeric_limits<float>::quiet_NaN());
        output.normal.assign(PatchMatchCandidateOutput::hypotheses *
                                 PatchMatchCandidateOutput::capacity * 3,
                             std::numeric_limits<float>::quiet_NaN());
    }
    if (!consume_pipeline_handoff) {
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_depth, output.depth.data(),
                output.depth.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_tmp_normal, output.normal.data(),
                output.normal.size() * sizeof(float));
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered coarse-to-precise buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth,
        &device_normal,
        &device_cost,
        const_cast<PatchMatchCamera*>(&input.camera),
        &downscale32,
        &device_tmp_depth,
        &device_tmp_normal,
        &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered coarse-to-precise launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }
    if (!input.defer_cuda_host_output) {
        runtime_status = cudaMemcpy(output.depth.data(), device_tmp_depth,
                                    output.depth.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(output.normal.data(), device_tmp_normal,
                                        output.normal.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost);
        if (runtime_status != cudaSuccess) {
            error = std::string("copying recovered coarse-to-precise output failed: ") +
                    cudaGetErrorString(runtime_status);
            release();
            return false;
        }
        output.cuda_host_output_materialized = true;
    } else {
        output.cuda_host_output_materialized = false;
        record_recovered_cuda_workspace_skipped_d2h(
            2U, output.depth.size() * sizeof(float) +
                    output.normal.size() * sizeof(float));
    }
    if (consume_pipeline_handoff) {
        output.cuda_workspace_handoff =
            publish_recovered_cuda_workspace_handoff(
                4U, input.cuda_workspace_handoff, 5U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing coarse-to-precise CUDA handoff failed";
            release();
            return false;
        }
        output.cuda_resident_cost_scratch = true;
        completed_handoff = true;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_cost_cuda_impl(
    PatchMatchCostInput& input,
    PatchMatchCostOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::cost);
    const std::size_t capacity = PatchMatchCandidateOutput::capacity;
    const std::size_t hypotheses = PatchMatchCandidateOutput::hypotheses;
    const bool batched = !input.neighbor_batch.empty();
    const std::span<const PatchMatchCostResourceGroup> resource_groups =
        input.resource_groups_view.empty()
            ? std::span<const PatchMatchCostResourceGroup>(
                  input.resource_groups)
            : input.resource_groups_view;
    const std::span<const std::uint8_t> reference_image =
        input.reference_image_view.empty()
            ? std::span<const std::uint8_t>(
                  input.reference_image_allocation)
            : input.reference_image_view;
    const std::span<const float> depth = input.depth_view.empty()
        ? std::span<const float>(input.depth)
        : input.depth_view;
    const std::span<const std::uint8_t> normal = input.normal_view.empty()
        ? std::span<const std::uint8_t>(input.normal)
        : input.normal_view;
    const std::span<const float> main_cost = input.cost_view.empty()
        ? std::span<const float>(input.cost)
        : input.cost_view;
    const std::span<const float> coarse_depth =
        input.coarse_depth_view.empty()
            ? std::span<const float>(input.coarse_depth)
            : input.coarse_depth_view;
    const std::span<const float> coarse_radius =
        input.coarse_depth_radius_view.empty()
            ? std::span<const float>(input.coarse_depth_radius)
            : input.coarse_depth_radius_view;
    const std::size_t effective_neighbor_count = batched
        ? input.neighbor_batch.size()
        : static_cast<std::size_t>(input.neighbor_count);
    if (input.depth_downscale == 0 ||
        (input.reference_patch_radius != 2 &&
         input.reference_patch_radius != 3) ||
        (input.hypotheses_per_pixel != 2 &&
         input.hypotheses_per_pixel != 8) ||
        effective_neighbor_count == 0 ||
        input.neighbor_count != effective_neighbor_count ||
        input.neighbor_cost_capacity < effective_neighbor_count) {
        error = "PatchMatch cost scalar parameters are invalid";
        return false;
    }
    if (!batched && (input.neighbor_count != 1 ||
                     input.neighbor_output_index != 0 ||
                     input.neighbor_cost_capacity != 1)) {
        error = "legacy PatchMatch cost input must describe one neighbor";
        return false;
    }
    if (batched) {
        if (resource_groups.empty()) {
            error = "PatchMatch cost batch has no resource groups";
            return false;
        }
        for (const PatchMatchCostNeighbor& neighbor : input.neighbor_batch) {
            if (neighbor.resource_group >= resource_groups.size() ||
                neighbor.output_index >= input.neighbor_cost_capacity ||
                neighbor.resource_index >=
                    resource_groups[neighbor.resource_group]
                        .level_offsets.size()) {
                error = "PatchMatch cost batch neighbor selector is invalid";
                return false;
            }
            const std::span<const std::uint8_t> texture =
                neighbor.texture_view.empty()
                    ? std::span<const std::uint8_t>(neighbor.texture)
                    : neighbor.texture_view;
            if (neighbor.texture_copy_regions.empty() &&
                texture.size() !=
                    static_cast<std::size_t>(input.neighbor_texture_width) *
                        input.neighbor_texture_height) {
                error = "PatchMatch cost batch neighbor texture is invalid";
                return false;
            }
            for (const PatchMatchTextureCopyRegion& region :
                 neighbor.texture_copy_regions) {
                if (region.width == 0U || region.height == 0U ||
                    region.source_pitch < region.width ||
                    region.x > input.neighbor_texture_width ||
                    region.y > input.neighbor_texture_height ||
                    region.width > input.neighbor_texture_width - region.x ||
                    region.height > input.neighbor_texture_height - region.y) {
                    error = "PatchMatch cost batch texture-copy region is invalid";
                    return false;
                }
                const std::uint64_t last_row = region.source_offset +
                    static_cast<std::uint64_t>(region.height - 1U) *
                        region.source_pitch;
                if (last_row > texture.size() ||
                    region.width > texture.size() - last_row) {
                    error = "PatchMatch cost batch texture-copy source is invalid";
                    return false;
                }
            }
        }
    }
    const std::size_t width =
        (input.reference_camera.width_original + input.depth_downscale - 1) /
        input.depth_downscale;
    const std::size_t height =
        (input.reference_camera.height_original + input.depth_downscale - 1) /
        input.depth_downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (pixels == 0 || work_items == 0 || work_items > capacity ||
        depth.size() != pixels || normal.size() != pixels * 3 ||
        main_cost.size() != pixels || coarse_depth.size() != pixels ||
        coarse_radius.size() != pixels ||
        reference_image.empty() ||
        input.neighbor_texture_width == 0 ||
        input.neighbor_texture_height == 0 ||
        (!batched && input.neighbor_texture.size() !=
            static_cast<std::size_t>(input.neighbor_texture_width) *
                input.neighbor_texture_height) ||
        (!batched && (input.neighbor_level_offsets.empty() ||
                      input.neighbor_mask.empty())) ||
        input.candidates.depth.size() != hypotheses * capacity ||
        input.candidates.normal.size() != hypotheses * capacity * 3 ||
        (!input.initial_neighbor_cost.empty() &&
         input.initial_neighbor_cost.size() !=
             static_cast<std::size_t>(input.neighbor_cost_capacity) *
                 hypotheses * capacity) ||
        (!input.initial_average_cost.empty() &&
         input.initial_average_cost.size() !=
             hypotheses * capacity) ||
        (!input.initial_auxiliary.empty() &&
         input.initial_auxiliary.size() < capacity)) {
        error = "PatchMatch cost launch or buffer dimensions are invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA cost context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction rotate = nullptr;
    CUfunction estimate = nullptr;
    CUfunction average = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_COST_CUDA_PTX);
    constexpr const char* rotate_name =
        "_ZN4cuda17pm_rotate_normalsEP6float313PinholeCamera";
    constexpr const char* estimate_name_8 =
        "_ZN4cuda27pm_estimate_cost_per_neighbIhLj3ELj8EEEvPfP6uchar3S1_PKfS5_13PinholeCamerajPKT_jfS6_jjyPKmPKhS5_PK6float3S1_jjjj";
    constexpr const char* estimate_name_radius_2_8 =
        "_ZN4cuda27pm_estimate_cost_per_neighbIhLj2ELj8EEEvPfP6uchar3S1_PKfS5_13PinholeCamerajPKT_jfS6_jjyPKmPKhS5_PK6float3S1_jjjj";
    constexpr const char* estimate_name_2 =
        "_ZN4cuda27pm_estimate_cost_per_neighbIhLj3ELj2EEEvPfP6uchar3S1_PKfS5_13PinholeCamerajPKT_jfS6_jjyPKmPKhS5_PK6float3S1_jjjj";
    constexpr const char* average_name =
        "_ZN4cuda24pm_avg_all_neighbs_costsE13PinholeCamerajjjjPKfPfPhjjjj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&rotate, module, rotate_name);
    if (driver_status == CUDA_SUCCESS) {
        const char* estimate_name = nullptr;
        if (input.hypotheses_per_pixel == 2) {
            if (input.reference_patch_radius != 3) {
                error = "recovered two-hypothesis cost path requires patch radius 3";
                cuModuleUnload(module);
                return false;
            }
            estimate_name = estimate_name_2;
        } else {
            estimate_name = input.reference_patch_radius == 2
                ? estimate_name_radius_2_8
                : estimate_name_8;
        }
        driver_status = cuModuleGetFunction(&estimate, module, estimate_name);
    }
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&average, module, average_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch cost PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }
    const bool consume_producer_handoff =
        input.cuda_workspace_handoff != 0U;
    const bool consume_resident_cost_scratch =
        input.candidates.cuda_resident_cost_scratch;
    const auto byte_zero = [](const void* data, std::size_t bytes) {
        const auto* values = static_cast<const std::uint8_t*>(data);
        return std::all_of(values, values + bytes,
                           [](std::uint8_t value) { return value == 0U; });
    };
    if (input.initial_cuda_cost_scratch_all_zero &&
        (!input.initial_cuda_cost_scratch_materialized ||
         consume_resident_cost_scratch ||
         !byte_zero(input.initial_neighbor_cost.data(),
                    input.initial_neighbor_cost.size() * sizeof(float)) ||
         !byte_zero(input.initial_average_cost.data(),
                    input.initial_average_cost.size() * sizeof(float)) ||
         !byte_zero(input.initial_auxiliary.data(),
                    input.initial_auxiliary.size()))) {
        error = "zero-initialized recovered CUDA cost scratch assertion failed";
        cuModuleUnload(module);
        return false;
    }
    if (!consume_producer_handoff &&
        !input.candidates.cuda_host_output_materialized) {
        error = "unmaterialized recovered CUDA candidates require a producer handoff token";
        cuModuleUnload(module);
        return false;
    }
    if (consume_resident_cost_scratch && !consume_producer_handoff) {
        error = "resident recovered CUDA cost scratch requires a producer handoff token";
        cuModuleUnload(module);
        return false;
    }
    if (!consume_resident_cost_scratch &&
        !input.initial_cuda_cost_scratch_materialized) {
        error = "unmaterialized recovered CUDA cost scratch requires a resident producer handoff";
        cuModuleUnload(module);
        return false;
    }

    void* device_depth = nullptr;
    void* device_normal = nullptr;
    void* device_cost = nullptr;
    void* device_coarse_depth = nullptr;
    void* device_coarse_radius = nullptr;
    void* device_reference = nullptr;
    std::vector<void*> device_offset_groups;
    std::vector<void*> device_mask_groups;
    void* device_candidate_depth = nullptr;
    void* device_candidate_normal = nullptr;
    void* device_neighbor_cost = nullptr;
    void* device_average_cost = nullptr;
    void* device_auxiliary = nullptr;
    CUstream texture_copy_stream = nullptr;
    CUarray texture_array = nullptr;
    CUtexObject texture = 0;
    const std::size_t resource_group_count =
        batched ? resource_groups.size() : 1;
    std::vector<std::size_t> offset_group_bytes(resource_group_count, 0U);
    std::vector<std::size_t> mask_group_bytes(resource_group_count, 0U);
    for (std::size_t group = 0; group < resource_group_count; ++group) {
        const std::vector<std::uint64_t>& offsets = batched
            ? resource_groups[group].level_offsets
            : input.neighbor_level_offsets;
        const std::vector<std::uint8_t>& mask = batched
            ? resource_groups[group].mask
            : input.neighbor_mask;
        if (offsets.empty() || mask.empty()) {
            error = "PatchMatch cost resource group is empty";
            cuModuleUnload(module);
            return false;
        }
        offset_group_bytes[group] = offsets.size() * sizeof(std::uint64_t);
        mask_group_bytes[group] = mask.size();
    }
    const std::size_t neighbor_cost_bytes =
        static_cast<std::size_t>(input.neighbor_cost_capacity) * hypotheses *
        capacity * sizeof(float);
    const std::size_t average_cost_bytes =
        hypotheses * capacity * sizeof(float);
    const std::size_t auxiliary_bytes = input.initial_auxiliary.empty()
        ? capacity
        : input.initial_auxiliary.size();
    const std::array<std::size_t,
                     RecoveredCudaCostWorkspace::fixed_allocation_count>
        fixed_workspace_bytes = {
            depth.size() * sizeof(float),
            normal.size(),
            main_cost.size() * sizeof(float),
            coarse_depth.size() * sizeof(float),
            coarse_radius.size() * sizeof(float),
            reference_image.size(),
            input.candidates.depth.size() * sizeof(float),
            input.candidates.normal.size() * sizeof(float),
            neighbor_cost_bytes,
            average_cost_bytes,
            auxiliary_bytes,
        };
    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_cost_workspace(
            fixed_workspace_bytes, offset_group_bytes, mask_group_bytes,
            input.neighbor_texture_width, input.neighbor_texture_height,
            workspace_handles, using_cached_workspace, error,
            input.cuda_workspace_handoff,
            consume_producer_handoff
                ? (consume_resident_cost_scratch ? 5U : 3U)
                : 0U,
            input.camera_resource_generation)) {
        cuModuleUnload(module);
        return false;
    }
    if (consume_producer_handoff && !using_cached_workspace) {
        error = "producer-to-cost CUDA handoff requires the active cached workspace";
        cuModuleUnload(module);
        return false;
    }
    if (using_cached_workspace) {
        device_depth = workspace_handles.fixed[cost_depth_slot];
        device_normal = workspace_handles.fixed[cost_normal_slot];
        device_cost = workspace_handles.fixed[cost_main_cost_slot];
        device_coarse_depth = workspace_handles.fixed[cost_coarse_depth_slot];
        device_coarse_radius = workspace_handles.fixed[cost_coarse_radius_slot];
        device_reference = workspace_handles.fixed[cost_reference_slot];
        device_candidate_depth =
            workspace_handles.fixed[cost_candidate_depth_slot];
        device_candidate_normal =
            workspace_handles.fixed[cost_candidate_normal_slot];
        device_neighbor_cost =
            workspace_handles.fixed[cost_neighbor_cost_slot];
        device_average_cost = workspace_handles.fixed[cost_average_slot];
        device_auxiliary = workspace_handles.fixed[cost_auxiliary_slot];
        device_offset_groups = workspace_handles.offset_groups;
        device_mask_groups = workspace_handles.mask_groups;
        texture_array = workspace_handles.texture_array;
        texture = workspace_handles.texture;
    }
    auto release = [&]() {
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            if (texture) cuTexObjectDestroy(texture);
            if (texture_array) cuArrayDestroy(texture_array);
            cudaFree(device_depth);
            cudaFree(device_normal);
            cudaFree(device_cost);
            cudaFree(device_coarse_depth);
            cudaFree(device_coarse_radius);
            cudaFree(device_reference);
            for (void* pointer : device_offset_groups) cudaFree(pointer);
            for (void* pointer : device_mask_groups) cudaFree(pointer);
            cudaFree(device_candidate_depth);
            cudaFree(device_candidate_normal);
            cudaFree(device_neighbor_cost);
            cudaFree(device_average_cost);
            cudaFree(device_auxiliary);
        }
        cuModuleUnload(module);
    };
    const std::size_t texture_bytes =
        static_cast<std::size_t>(input.neighbor_texture_width) *
        input.neighbor_texture_height;
    const bool use_resident_texture_sources = using_cached_workspace &&
        batched && input.cuda_workspace_handoff != 0U &&
        input.camera_resource_generation != 0U;
    if (use_resident_texture_sources) {
        driver_status = acquire_recovered_cuda_worker_stream(
            texture_copy_stream);
        if (driver_status != CUDA_SUCCESS) {
            error = "acquiring recovered CUDA texture-copy stream failed";
            release();
            return false;
        }
    }
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes,
                                 RecoveredCudaCostH2dKind kind) {
        cudaError_t status = *destination == nullptr
            ? cudaMalloc(destination, bytes)
            : cudaSuccess;
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        if (status == cudaSuccess)
            record_recovered_cuda_cost_h2d_detail(kind, bytes);
        return status;
    };
    if (!consume_producer_handoff) {
        runtime_status = allocate_and_copy(
            &device_depth, depth.data(), depth.size() * sizeof(float),
            RecoveredCudaCostH2dKind::main_state);
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_normal, normal.data(), normal.size(),
                RecoveredCudaCostH2dKind::main_state);
        if (runtime_status == cudaSuccess)
            runtime_status = allocate_and_copy(
                &device_cost, main_cost.data(),
                main_cost.size() * sizeof(float),
                RecoveredCudaCostH2dKind::main_state);
    }
    if (runtime_status == cudaSuccess && !consume_producer_handoff)
        runtime_status = allocate_and_copy(
            &device_coarse_depth, coarse_depth.data(),
            coarse_depth.size() * sizeof(float),
            RecoveredCudaCostH2dKind::main_state);
    if (runtime_status == cudaSuccess && !consume_producer_handoff)
        runtime_status = allocate_and_copy(
            &device_coarse_radius, coarse_radius.data(),
            coarse_radius.size() * sizeof(float),
            RecoveredCudaCostH2dKind::main_state);
    if (runtime_status == cudaSuccess && !consume_producer_handoff)
        runtime_status = allocate_and_copy(
            &device_reference, reference_image.data(),
            reference_image.size(), RecoveredCudaCostH2dKind::main_state);
    if (!using_cached_workspace) {
        device_offset_groups.assign(resource_group_count, nullptr);
        device_mask_groups.assign(resource_group_count, nullptr);
    }
    const bool grouped_resources_resident =
        using_cached_workspace &&
        workspace_handles.grouped_resources_resident;
    if (!grouped_resources_resident) {
        for (std::size_t group = 0;
             runtime_status == cudaSuccess && group < resource_group_count;
             ++group) {
            const std::vector<std::uint64_t>& offsets = batched
                ? resource_groups[group].level_offsets
                : input.neighbor_level_offsets;
            const std::vector<std::uint8_t>& mask = batched
                ? resource_groups[group].mask
                : input.neighbor_mask;
            runtime_status = allocate_and_copy(
                &device_offset_groups[group], offsets.data(),
                offsets.size() * sizeof(std::uint64_t),
                RecoveredCudaCostH2dKind::group_resources);
            if (runtime_status == cudaSuccess)
                runtime_status = allocate_and_copy(
                    &device_mask_groups[group], mask.data(), mask.size(),
                    RecoveredCudaCostH2dKind::group_resources);
        }
        if (runtime_status == cudaSuccess && using_cached_workspace &&
            !mark_recovered_cuda_cost_resources_resident(
                input.camera_resource_generation,
                input.cuda_workspace_handoff)) {
            error = "publishing recovered CUDA cost resource residency failed";
            release();
            return false;
        }
    } else {
        std::uint64_t resource_bytes = 0U;
        for (std::size_t group = 0; group < resource_group_count; ++group)
            resource_bytes += offset_group_bytes[group] +
                              mask_group_bytes[group];
        record_recovered_cuda_workspace_skipped_h2d(
            2U * resource_group_count, resource_bytes);
    }

    output.candidates = std::move(input.candidates);
    output.candidates.cuda_workspace_handoff = 0U;
    output.candidates.cuda_resident_cost_scratch = false;
    output.cuda_host_output_materialized = true;
    if (input.initial_neighbor_cost.empty()) {
        output.per_neighbor_cost.assign(
            static_cast<std::size_t>(input.neighbor_cost_capacity) *
                hypotheses * capacity,
            std::numeric_limits<float>::quiet_NaN());
    } else {
        output.per_neighbor_cost = std::move(input.initial_neighbor_cost);
    }
    if (input.initial_average_cost.empty()) {
        output.average_cost.assign(
            hypotheses * capacity,
            std::numeric_limits<float>::quiet_NaN());
    } else {
        output.average_cost = std::move(input.initial_average_cost);
    }
    output.auxiliary = input.initial_auxiliary.empty()
        ? std::vector<std::uint8_t>(capacity, 0)
        : std::move(input.initial_auxiliary);
    if (runtime_status == cudaSuccess && !consume_producer_handoff)
        runtime_status = allocate_and_copy(
            &device_candidate_depth, output.candidates.depth.data(),
            output.candidates.depth.size() * sizeof(float),
            RecoveredCudaCostH2dKind::main_state);
    if (runtime_status == cudaSuccess && !consume_producer_handoff)
        runtime_status = allocate_and_copy(
            &device_candidate_normal, output.candidates.normal.data(),
            output.candidates.normal.size() * sizeof(float),
            RecoveredCudaCostH2dKind::main_state);
    if (runtime_status == cudaSuccess && !consume_resident_cost_scratch &&
        input.initial_cuda_cost_scratch_all_zero) {
        runtime_status = cudaMemset(
            device_neighbor_cost, 0,
            output.per_neighbor_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemset(
                device_average_cost, 0,
                output.average_cost.size() * sizeof(float));
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemset(
                device_auxiliary, 0, output.auxiliary.size());
        if (runtime_status == cudaSuccess)
            record_recovered_cuda_workspace_skipped_h2d(
                3U, output.per_neighbor_cost.size() * sizeof(float) +
                        output.average_cost.size() * sizeof(float) +
                        output.auxiliary.size());
    } else {
        if (runtime_status == cudaSuccess && !consume_resident_cost_scratch)
            runtime_status = allocate_and_copy(
                &device_neighbor_cost, output.per_neighbor_cost.data(),
                output.per_neighbor_cost.size() * sizeof(float),
                RecoveredCudaCostH2dKind::scratch);
        if (runtime_status == cudaSuccess && !consume_resident_cost_scratch)
            runtime_status = allocate_and_copy(
                &device_average_cost, output.average_cost.data(),
                output.average_cost.size() * sizeof(float),
                RecoveredCudaCostH2dKind::scratch);
        if (runtime_status == cudaSuccess && !consume_resident_cost_scratch)
            runtime_status = allocate_and_copy(
                &device_auxiliary, output.auxiliary.data(),
                output.auxiliary.size(),
                RecoveredCudaCostH2dKind::scratch);
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered PatchMatch cost buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (consume_producer_handoff) {
        record_recovered_cuda_workspace_skipped_h2d(
            8U,
            depth.size() * sizeof(float) + normal.size() +
                main_cost.size() * sizeof(float) +
                coarse_depth.size() * sizeof(float) +
                coarse_radius.size() * sizeof(float) +
                reference_image.size() +
                output.candidates.depth.size() * sizeof(float) +
                output.candidates.normal.size() * sizeof(float));
    }
    if (consume_resident_cost_scratch) {
        record_recovered_cuda_workspace_skipped_h2d(
            3U,
            output.per_neighbor_cost.size() * sizeof(float) +
                output.average_cost.size() * sizeof(float) +
                output.auxiliary.size());
    }

    CUDA_ARRAY3D_DESCRIPTOR array_description{};
    array_description.Width = input.neighbor_texture_width;
    array_description.Height = input.neighbor_texture_height;
    array_description.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    array_description.NumChannels = 1;
    array_description.Flags = CUDA_ARRAY3D_SURFACE_LDST;
    driver_status = using_cached_workspace
        ? CUDA_SUCCESS
        : cuArray3DCreate(&texture_array, &array_description);
    auto upload_texture = [&](const PatchMatchCostNeighbor* neighbor,
                              bool force_full) {
        const std::span<const std::uint8_t> initial_texture =
            input.neighbor_texture_view.empty()
                ? std::span<const std::uint8_t>(input.neighbor_texture)
                : input.neighbor_texture_view;
        if (force_full &&
            (neighbor == nullptr || neighbor->texture.size() != texture_bytes) &&
            initial_texture.size() != texture_bytes) {
            error = "initial recovered CUDA atlas source is not a full texture";
            return CUDA_ERROR_INVALID_VALUE;
        }
        const std::uint8_t* source = neighbor == nullptr || force_full
            ? initial_texture.data()
            : (neighbor->texture_view.empty()
                   ? neighbor->texture.data()
                   : neighbor->texture_view.data());
        if (neighbor != nullptr && force_full &&
            neighbor->texture.size() == texture_bytes)
            source = neighbor->texture.data();
        const bool use_regions = neighbor != nullptr && !force_full &&
            !neighbor->texture_copy_regions.empty();
        const bool use_device_source = use_resident_texture_sources &&
            !force_full;
        std::size_t source_slot = 0U;
        CUdeviceptr device_source = 0U;
        if (use_device_source) {
            source_slot = neighbor->texture_source_grouped
                ? static_cast<std::size_t>(neighbor->resource_group)
                : static_cast<std::size_t>(neighbor->resource_group) * 10U +
                      neighbor->resource_index;
            const std::size_t source_count = neighbor->texture_source_grouped
                ? resource_group_count
                : static_cast<std::size_t>(input.neighbor_cost_capacity);
            if (workspace_handles.workspace == nullptr ||
                source_slot >= source_count)
                return CUDA_ERROR_INVALID_VALUE;
            const std::size_t source_size = neighbor->texture_view.empty()
                ? neighbor->texture.size()
                : neighbor->texture_view.size();
            if (source_slot >= workspace_handles.texture_sources.size() ||
                workspace_handles.texture_sources[source_slot] == nullptr) {
                if (!ensure_recovered_cuda_cost_texture_source(
                        source_count, source_slot, source_size,
                        workspace_handles, error))
                    return CUDA_ERROR_OUT_OF_MEMORY;
            }
            if (source_slot >= workspace_handles.workspace
                                   ->texture_source_generations.size())
                return CUDA_ERROR_INVALID_VALUE;
            if (workspace_handles.workspace
                    ->texture_source_generations[source_slot] !=
                input.camera_resource_generation) {
                const cudaError_t upload_status = cudaMemcpy(
                    workspace_handles.texture_sources[source_slot], source,
                    source_size, cudaMemcpyHostToDevice);
                if (upload_status != cudaSuccess)
                    return CUDA_ERROR_UNKNOWN;
                record_recovered_cuda_cost_h2d_detail(
                    RecoveredCudaCostH2dKind::texture_source, source_size);
                workspace_handles.workspace
                    ->texture_source_generations[source_slot] =
                    input.camera_resource_generation;
            }
            device_source = reinterpret_cast<CUdeviceptr>(
                workspace_handles.texture_sources[source_slot]);
        }
        auto upload_region = [&](std::uint64_t source_offset,
                                 std::uint32_t source_pitch,
                                 std::uint32_t x, std::uint32_t y,
                                 std::uint32_t copy_width,
                                 std::uint32_t copy_height) {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = use_device_source
                ? CU_MEMORYTYPE_DEVICE
                : CU_MEMORYTYPE_HOST;
            if (use_device_source) {
                copy.srcDevice = device_source + source_offset;
            } else {
                copy.srcHost = source + source_offset;
            }
            copy.srcPitch = source_pitch;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = texture_array;
            copy.dstXInBytes = x;
            copy.dstY = y;
            copy.WidthInBytes = copy_width;
            copy.Height = copy_height;
            return use_device_source
                ? cuMemcpy2DAsync(&copy, texture_copy_stream)
                : cuMemcpy2D(&copy);
        };
        if (!use_regions)
            return upload_region(0U, input.neighbor_texture_width,
                                 0U, 0U, input.neighbor_texture_width,
                                 input.neighbor_texture_height);
        CUresult status = CUDA_SUCCESS;
        for (const PatchMatchTextureCopyRegion& region :
             neighbor->texture_copy_regions) {
            status = upload_region(region.source_offset,
                                   region.source_pitch, region.x, region.y,
                                   region.width, region.height);
            if (status != CUDA_SUCCESS) break;
        }
        return status;
    };
    // A cached workspace is reused by its worker across reference cameras.
    // The first upload of a new resource generation must therefore
    // materialize the complete zero-backed host atlas.  Every later prepare
    // copies only the five proven-written mip rectangles, matching the
    // target's persistent-array lifetime.
    if (driver_status == CUDA_SUCCESS) {
        const PatchMatchCostNeighbor* first_neighbor =
            batched ? &input.neighbor_batch.front() : nullptr;
        const bool initialize_generation = !grouped_resources_resident;
        driver_status = upload_texture(first_neighbor, initialize_generation);
        // Capture/replay resources carry a full zero-backed atlas with the
        // first neighbour already written. Compact production resources do
        // not: initialize the fresh camera array to the independently proven
        // all-zero state, then issue that neighbour's five target rectangles.
        if (driver_status == CUDA_SUCCESS && initialize_generation &&
            first_neighbor != nullptr && first_neighbor->texture.empty())
            driver_status = upload_texture(first_neighbor, false);
    }
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = texture_array;
    CUDA_TEXTURE_DESC texture_description{};
    texture_description.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texture_description.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texture_description.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
    texture_description.filterMode = CU_TR_FILTER_MODE_LINEAR;
    if (driver_status == CUDA_SUCCESS && !using_cached_workspace)
        driver_status = cuTexObjectCreate(
            &texture, &resource, &texture_description, nullptr);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        if (error.empty())
            error = std::string("creating recovered PatchMatch texture failed: ") +
                    (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    void* rotate_before_arguments[] = {
        &device_candidate_normal,
        const_cast<PatchMatchCamera*>(&input.rotate_before_camera),
    };
    // The target rotates exactly the active template hypothesis planes.  The
    // two-hypothesis C2P specialization leaves persistent planes 2..7 bitwise
    // untouched; rotating the full eight-plane allocation introduces a
    // non-invertible float round-trip in those inactive planes.
    const unsigned int rotate_blocks = static_cast<unsigned int>(
        (capacity * input.hypotheses_per_pixel + 127U) / 128U);
    driver_status = cuLaunchKernel(rotate, rotate_blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, rotate_before_arguments, nullptr);

    std::uint32_t downscale = input.depth_downscale;
    std::uint32_t detailed = input.image_one_step_more_detailed;
    float deviation = input.deviation_threshold_multiplier;
    std::uint32_t checkboard = input.is_checkboard;
    std::uint32_t checkboard_step = input.checkboard_step;
    std::uint32_t fourth = input.only_each_fourth_pixel;
    std::uint32_t offset = input.pixel_offset;
    const unsigned int blocks = static_cast<unsigned int>(
        (work_items * input.hypotheses_per_pixel + 127) / 128);
    for (std::size_t neighbor_offset = 0;
        driver_status == CUDA_SUCCESS &&
         neighbor_offset < effective_neighbor_count;
         ++neighbor_offset) {
        if (batched && neighbor_offset != 0) {
            driver_status = upload_texture(
                &input.neighbor_batch[neighbor_offset], false);
            if (driver_status != CUDA_SUCCESS) break;
        }
        PatchMatchCamera* neighbor_camera = const_cast<PatchMatchCamera*>(
            batched ? &input.neighbor_batch[neighbor_offset].camera
                    : &input.neighbor_camera);
        std::uint32_t neighbor_level = batched
            ? input.neighbor_batch[neighbor_offset].resource_index
            : input.neighbor_camera_level;
        std::uint32_t neighbor_index = batched
            ? input.neighbor_batch[neighbor_offset].output_index
            : input.neighbor_output_index;
        const std::size_t resource_group = batched
            ? input.neighbor_batch[neighbor_offset].resource_group
            : 0;
        void* device_offsets = device_offset_groups[resource_group];
        void* device_mask = device_mask_groups[resource_group];
        void* estimate_arguments[] = {
            &device_depth, &device_normal, &device_cost, &device_coarse_depth,
            &device_coarse_radius,
            const_cast<PatchMatchCamera*>(&input.reference_camera), &downscale,
            &device_reference, &detailed, &deviation, neighbor_camera,
            &neighbor_level, &neighbor_index, &texture, &device_offsets,
            &device_mask, &device_candidate_depth, &device_candidate_normal,
            &device_neighbor_cost, &checkboard, &checkboard_step, &fourth,
            &offset,
        };
        driver_status = cuLaunchKernel(estimate, blocks, 1, 1, 128, 1, 1, 0,
                                       nullptr, estimate_arguments, nullptr);
    }

    void* rotate_after_arguments[] = {
        &device_candidate_normal,
        const_cast<PatchMatchCamera*>(&input.rotate_after_camera),
    };
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuLaunchKernel(rotate, rotate_blocks, 1, 1,
                                       128, 1, 1, 0,
                                       nullptr, rotate_after_arguments, nullptr);

    std::uint32_t hypotheses32 = input.hypotheses_per_pixel;
    std::uint32_t neighbors32 =
        static_cast<std::uint32_t>(effective_neighbor_count);
    void* average_arguments[] = {
        const_cast<PatchMatchCamera*>(&input.reference_camera), &downscale,
        &detailed, &hypotheses32, &neighbors32, &device_neighbor_cost,
        &device_average_cost, &device_auxiliary, &checkboard,
        &checkboard_step, &fourth, &offset,
    };
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuLaunchKernel(average, blocks, 1, 1, 128, 1, 1, 0,
                                       nullptr, average_arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch cost launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    const std::size_t active_hypotheses = input.hypotheses_per_pixel;
    const std::size_t active_auxiliary_groups =
        (effective_neighbor_count + 7U) / 8U;
    bool contiguous_neighbor_outputs = true;
    if (batched) {
        for (std::size_t neighbor = 0;
             neighbor < input.neighbor_batch.size(); ++neighbor) {
            if (input.neighbor_batch[neighbor].output_index != neighbor) {
                contiguous_neighbor_outputs = false;
                break;
            }
        }
    }
    // Every recovered production launch has an exact 128-thread extent.  In
    // that domain the estimate/average kernels modify only the tmp-index
    // prefix of each strided plane.  Preserve the remaining host snapshot and
    // transfer only those proven-written ranges.  Standalone or padded public
    // calls retain the conservative full-allocation copy.
    const bool copy_written_ranges_only =
        using_cached_workspace &&
        contiguous_neighbor_outputs &&
        (work_items * active_hypotheses) % 128U == 0U &&
        active_auxiliary_groups * active_hypotheses * capacity <=
            output.auxiliary.size();
    const bool defer_host_output =
        input.defer_cuda_host_output && using_cached_workspace;
    if (defer_host_output) {
        const std::uint64_t candidate_normal_bytes = copy_written_ranges_only
            ? active_hypotheses * capacity * 3U * sizeof(float)
            : output.candidates.normal.size() * sizeof(float);
        const std::uint64_t neighbor_bytes = copy_written_ranges_only
            ? work_items * active_hypotheses * effective_neighbor_count *
                  sizeof(float)
            : output.per_neighbor_cost.size() * sizeof(float);
        const std::uint64_t average_bytes = copy_written_ranges_only
            ? work_items * active_hypotheses * sizeof(float)
            : output.average_cost.size() * sizeof(float);
        const std::uint64_t auxiliary_copy_bytes = copy_written_ranges_only
            ? work_items * active_auxiliary_groups * active_hypotheses
            : output.auxiliary.size();
        output.candidates.cuda_host_output_materialized = false;
        output.cuda_host_output_materialized = false;
        record_recovered_cuda_workspace_skipped_d2h(
            4U, candidate_normal_bytes + neighbor_bytes + average_bytes +
                    auxiliary_copy_bytes);
    } else if (copy_written_ranges_only) {
        if (!input.candidates.cuda_host_output_materialized)
            runtime_status = cudaMemcpy(
                output.candidates.depth.data(), device_candidate_depth,
                output.candidates.depth.size() * sizeof(float),
                cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.candidates.normal.data(), device_candidate_normal,
                active_hypotheses * capacity * 3U * sizeof(float),
                cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy2D(
                output.per_neighbor_cost.data(),
                capacity * active_hypotheses * sizeof(float),
                device_neighbor_cost,
                capacity * active_hypotheses * sizeof(float),
                work_items * active_hypotheses * sizeof(float),
                effective_neighbor_count, cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy2D(
                output.average_cost.data(), capacity * sizeof(float),
                device_average_cost,
                capacity * sizeof(float), work_items * sizeof(float),
                active_hypotheses, cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy2D(
                output.auxiliary.data(), capacity, device_auxiliary, capacity,
                work_items, active_auxiliary_groups * active_hypotheses,
                cudaMemcpyDeviceToHost);
    } else {
        if (!input.candidates.cuda_host_output_materialized)
            runtime_status = cudaMemcpy(
                output.candidates.depth.data(), device_candidate_depth,
                output.candidates.depth.size() * sizeof(float),
                cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.candidates.normal.data(), device_candidate_normal,
                output.candidates.normal.size() * sizeof(float),
                cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.per_neighbor_cost.data(), device_neighbor_cost,
                output.per_neighbor_cost.size() * sizeof(float),
                cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.average_cost.data(), device_average_cost,
                output.average_cost.size() * sizeof(float),
                cudaMemcpyDeviceToHost);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                output.auxiliary.data(), device_auxiliary,
                output.auxiliary.size(), cudaMemcpyDeviceToHost);
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered PatchMatch cost output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (!defer_host_output) {
        output.candidates.cuda_host_output_materialized = true;
        output.cuda_host_output_materialized = true;
    }
    if (using_cached_workspace) {
        output.cuda_workspace_handoff =
            consume_producer_handoff
                ? publish_recovered_cuda_workspace_handoff(
                      consume_resident_cost_scratch ? 5U : 3U,
                      input.cuda_workspace_handoff, 1U)
                : publish_recovered_cuda_workspace_handoff(0U, 0U, 1U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing recovered CUDA cost workspace handoff failed";
            release();
            return false;
        }
    }
    release();
    return true;
}

bool run_recovered_patchmatch_bilateral_u8_cuda_impl(
    const PatchMatchBilateralU8Input& input,
    PatchMatchBilateralU8Output& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::bilateral);
    const std::size_t pixels =
        static_cast<std::size_t>(input.width) * input.height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (input.width == 0U || input.height == 0U || pixels == 0U ||
        input.pixel_offset != 0U || work_items != pixels ||
        input.depth.size() != pixels || input.normal.size() != pixels * 3U ||
        input.image.size() != pixels || !(input.sigma_d > 0.0F) ||
        !(input.sigma_r > 0.0F) || !std::isfinite(input.sigma_d) ||
        !std::isfinite(input.sigma_r)) {
        error = "PatchMatch bilateral uchar full-frame launch is invalid";
        return false;
    }

    cudaError_t runtime_status =
        cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA bilateral context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status =
        cuModuleLoad(&module, METMODEL_PM_BILATERAL_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda35pm_bilateral_depth_map_filtering_r3IhEEvPKfPK6uchar3PKT_PfPS3_jjffj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch bilateral PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    float* device_input_depth = nullptr;
    std::uint8_t* device_input_normal = nullptr;
    std::uint8_t* device_image = nullptr;
    float* device_output_depth = nullptr;
    std::uint8_t* device_output_normal = nullptr;
    auto release = [&]() {
        cudaFree(device_input_depth);
        cudaFree(device_input_normal);
        cudaFree(device_image);
        cudaFree(device_output_depth);
        cudaFree(device_output_normal);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](auto** destination, const auto* source,
                                 std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_input_depth, input.depth.data(), pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_input_normal, input.normal.data(), pixels * 3U);
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_image, input.image.data(), pixels);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMalloc(
            &device_output_depth, pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMalloc(&device_output_normal, pixels * 3U);
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered PatchMatch bilateral buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t width = input.width;
    std::uint32_t height = input.height;
    float sigma_d = input.sigma_d;
    float sigma_r = input.sigma_r;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_input_depth, &device_input_normal, &device_image,
        &device_output_depth, &device_output_normal, &width, &height,
        &sigma_d, &sigma_r, &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127U) / 128U);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch bilateral launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    output.depth.resize(pixels);
    output.normal.resize(pixels * 3U);
    runtime_status = cudaMemcpy(
        output.depth.data(), device_output_depth, pixels * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            output.normal.data(), device_output_normal, pixels * 3U,
            cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered PatchMatch bilateral output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_filter_chain_cuda_impl(
    const PatchMatchFilterChainInput& input,
    PatchMatchFilterChainOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::filter);
    if (input.device_index >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        input.depth_downscale == 0U || input.camera.width_original == 0U ||
        input.camera.height_original == 0U) {
        error = "PatchMatch resident filter scalar parameters are invalid";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + input.depth_downscale - 1U) /
        input.depth_downscale;
    const std::size_t height =
        (input.camera.height_original + input.depth_downscale - 1U) /
        input.depth_downscale;
    if (width == 0U || height == 0U ||
        width > std::numeric_limits<std::size_t>::max() / height) {
        error = "PatchMatch resident filter active grid is invalid";
        return false;
    }
    const std::size_t pixels = width * height;
    if (pixels > std::numeric_limits<std::uint32_t>::max() ||
        input.depth_allocation.size() < pixels ||
        input.depth_allocation.size() != input.cost_allocation.size() ||
        input.normal_allocation.size() < pixels * 3U ||
        input.estimated_normal_allocation.empty() ||
        (input.estimate_normal_map &&
         input.estimated_normal_allocation.size() < pixels * 3U) ||
        input.filtered_mask_allocation.size() < pixels) {
        error = "PatchMatch resident filter allocations are invalid";
        return false;
    }
    const bool consume_pipeline_handoff =
        input.cuda_workspace_handoff != 0U;
    if ((!input.cuda_main_state_materialized ||
         !input.cuda_inlier_masks_materialized) &&
        !consume_pipeline_handoff) {
        error = "unmaterialized recovered filter input requires a workspace handoff";
        return false;
    }
    if (consume_pipeline_handoff &&
        recovered_cuda_workspace_handoff_stage(
            input.cuda_workspace_handoff) != 4U) {
        error = "recovered resident filter requires a stage-4 inlier handoff";
        return false;
    }

    cudaError_t runtime_status =
        cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA resident filter context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    std::array<CUfunction, 5> functions{};
    constexpr std::array<const char*, 5> kernel_names = {
        "_ZN4cuda23pm_filtering_check_costEPfPKf13PinholeCamerajPiS4_j",
        "_ZN4cuda29pm_filtering_check_neighboursEPKfPh13PinholeCamerajffPiS4_j",
        "_ZN4cuda46pm_filtering_clear_depth_map_wrt_filtered_maskEPfPKh13PinholeCamerajPij",
        "_ZN4cuda20pm_filtering_normalsEPKfPK6uchar3PS2_hPh13PinholeCamerajPiS8_S8_PfS8_j",
        "_ZN4cuda36pm_filtering_estimate_speckles_edgesEPKfPh13PinholeCamerajj",
    };
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_FILTER_CUDA_PTX);
    for (std::size_t index = 0;
         driver_status == CUDA_SUCCESS && index < functions.size(); ++index)
        driver_status = cuModuleGetFunction(
            &functions[index], module, kernel_names[index]);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered resident filter PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    std::array<std::size_t,
               RecoveredCudaCostWorkspace::fixed_allocation_count>
        workspace_bytes{};
    workspace_bytes[cost_depth_slot] =
        input.depth_allocation.size() * sizeof(float);
    workspace_bytes[cost_main_cost_slot] =
        input.cost_allocation.size() * sizeof(float);
    workspace_bytes[cost_normal_slot] = input.normal_allocation.size();
    workspace_bytes[filter_estimated_normal_slot] =
        input.estimated_normal_allocation.size();
    workspace_bytes[filter_mask_slot] =
        input.filtered_mask_allocation.size();
    workspace_bytes[pipeline_inlier_masks_slot] =
        input.neighbor_inlier_masks_allocation.size();
    for (std::size_t slot = filter_counter_no_cost_slot;
         slot <= filter_counter_ncos_sum_slot; ++slot)
        workspace_bytes[slot] = sizeof(std::uint32_t);

    RecoveredCudaCostWorkspaceHandles workspace_handles;
    bool using_cached_workspace = false;
    if (!acquire_recovered_cuda_pipeline_fixed_workspace(
            workspace_bytes, RecoveredCudaWorkspaceClient::filter,
            workspace_handles, using_cached_workspace, error)) {
        cuModuleUnload(module);
        return false;
    }
    bool completed_handoff = false;
    auto release = [&]() {
        if (!completed_handoff)
            invalidate_recovered_cuda_workspace_handoff(
                input.cuda_workspace_handoff);
        if (using_cached_workspace) {
            release_recovered_cuda_cost_workspace_lease();
        } else {
            for (std::size_t slot = 0; slot < workspace_bytes.size(); ++slot) {
                if (workspace_bytes[slot] != 0U)
                    cudaFree(workspace_handles.fixed[slot]);
            }
        }
        cuModuleUnload(module);
    };
    if (!using_cached_workspace) {
        for (std::size_t slot = 0;
             runtime_status == cudaSuccess && slot < workspace_bytes.size();
             ++slot) {
            if (workspace_bytes[slot] != 0U)
                runtime_status = cudaMalloc(
                    &workspace_handles.fixed[slot], workspace_bytes[slot]);
        }
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered resident filter buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    output = {};
    output.depth_allocation = input.depth_allocation;
    output.estimated_normal_allocation = input.estimated_normal_allocation;
    output.filtered_mask_allocation = input.filtered_mask_allocation;
    output.counter_no_cost_samples = input.counter_no_cost_samples;
    output.counter_big_cost_samples = input.counter_big_cost_samples;
    output.counter_no_neighbours = input.counter_no_neighbours;
    output.counter_no_close_neighbours = input.counter_no_close_neighbours;
    output.counter_inconsistent_normal = input.counter_inconsistent_normal;
    output.counter_bad_view_angle_estimated_normal =
        input.counter_bad_view_angle_estimated_normal;
    output.counter_bad_view_angle_found_normal =
        input.counter_bad_view_angle_found_normal;
    output.counter_cos_sum = input.counter_cos_sum;
    output.counter_ncos_sum = input.counter_ncos_sum;
    output.neighbor_inlier_masks_allocation =
        input.neighbor_inlier_masks_allocation;

    auto upload = [&](std::size_t slot, const void* source) {
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                workspace_handles.fixed[slot], source,
                workspace_bytes[slot], cudaMemcpyHostToDevice);
    };
    if (!consume_pipeline_handoff) {
        upload(cost_depth_slot, input.depth_allocation.data());
        upload(cost_main_cost_slot, input.cost_allocation.data());
        upload(cost_normal_slot, input.normal_allocation.data());
        if (!input.neighbor_inlier_masks_allocation.empty())
            upload(pipeline_inlier_masks_slot,
                   input.neighbor_inlier_masks_allocation.data());
    } else {
        record_recovered_cuda_workspace_skipped_h2d(
            3U +
                (input.neighbor_inlier_masks_allocation.empty() ? 0U : 1U),
            workspace_bytes[cost_depth_slot] +
                workspace_bytes[cost_main_cost_slot] +
                workspace_bytes[cost_normal_slot] +
                workspace_bytes[pipeline_inlier_masks_slot]);
    }
    upload(filter_estimated_normal_slot,
           input.estimated_normal_allocation.data());
    upload(filter_mask_slot, input.filtered_mask_allocation.data());
    upload(filter_counter_no_cost_slot, &input.counter_no_cost_samples);
    upload(filter_counter_big_cost_slot, &input.counter_big_cost_samples);
    upload(filter_counter_no_neighbours_slot, &input.counter_no_neighbours);
    upload(filter_counter_no_close_neighbours_slot,
           &input.counter_no_close_neighbours);
    upload(filter_counter_clear_slot, &input.first_counter_not_empty);
    upload(filter_counter_inconsistent_normal_slot,
           &input.counter_inconsistent_normal);
    upload(filter_counter_bad_estimated_normal_slot,
           &input.counter_bad_view_angle_estimated_normal);
    upload(filter_counter_bad_found_normal_slot,
           &input.counter_bad_view_angle_found_normal);
    upload(filter_counter_cos_sum_slot, &input.counter_cos_sum);
    upload(filter_counter_ncos_sum_slot, &input.counter_ncos_sum);
    if (runtime_status != cudaSuccess) {
        error = std::string("uploading recovered resident filter state failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    const auto launch = [&](CUfunction function, std::size_t work_items,
                            void** arguments, const char* label) {
        const unsigned int blocks = static_cast<unsigned int>(
            (work_items + 127U) / 128U);
        CUresult status = cuLaunchKernel(
            function, blocks, 1U, 1U, 128U, 1U, 1U, 0U, nullptr,
            arguments, nullptr);
        if (status == CUDA_SUCCESS) status = cuCtxSynchronize();
        if (status != CUDA_SUCCESS) {
            const char* message = nullptr;
            cuGetErrorString(status, &message);
            error = std::string(label) + " failed: " +
                    (message ? message : "unknown CUDA driver error");
            return false;
        }
        return true;
    };
    const std::uint32_t span = patchmatch_balanced_batch_span(
        static_cast<std::uint32_t>(pixels));
    const auto for_each_batch = [&](auto&& callback) {
        for (std::uint32_t offset = 0U; offset < pixels;) {
            const std::uint32_t work_items = std::min(
                span, static_cast<std::uint32_t>(pixels) - offset);
            if (!callback(offset, work_items)) return false;
            offset += work_items;
        }
        return true;
    };
    std::uint32_t downscale = input.depth_downscale;
    float depth_min = input.depth_min;
    float depth_max = input.depth_max;
    std::uint8_t estimate_normal = input.estimate_normal_map ? 1U : 0U;
    void* device_depth = workspace_handles.fixed[cost_depth_slot];
    void* device_cost = workspace_handles.fixed[cost_main_cost_slot];
    void* device_normal = workspace_handles.fixed[cost_normal_slot];
    void* device_estimated =
        workspace_handles.fixed[filter_estimated_normal_slot];
    void* device_mask = workspace_handles.fixed[filter_mask_slot];

    if (!for_each_batch([&](std::uint32_t offset,
                             std::uint32_t work_items) {
            void* arguments[] = {
                &device_depth, &device_cost,
                const_cast<PatchMatchCamera*>(&input.camera), &downscale,
                &workspace_handles.fixed[filter_counter_no_cost_slot],
                &workspace_handles.fixed[filter_counter_big_cost_slot],
                &offset,
            };
            return launch(functions[0], work_items, arguments,
                          "recovered resident check-cost filter");
        }) ||
        !for_each_batch([&](std::uint32_t offset,
                             std::uint32_t work_items) {
            void* arguments[] = {
                &device_depth, &device_mask,
                const_cast<PatchMatchCamera*>(&input.camera), &downscale,
                &depth_min, &depth_max,
                &workspace_handles.fixed[filter_counter_no_neighbours_slot],
                &workspace_handles.fixed[
                    filter_counter_no_close_neighbours_slot],
                &offset,
            };
            return launch(functions[1], work_items, arguments,
                          "recovered resident neighbour filter");
        }) ||
        !for_each_batch([&](std::uint32_t offset,
                             std::uint32_t work_items) {
            void* arguments[] = {
                &device_depth, &device_mask,
                const_cast<PatchMatchCamera*>(&input.camera), &downscale,
                &workspace_handles.fixed[filter_counter_clear_slot], &offset,
            };
            return launch(functions[2], work_items, arguments,
                          "recovered resident first clear-depth filter");
        })) {
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        &output.first_counter_not_empty,
        workspace_handles.fixed[filter_counter_clear_slot],
        sizeof(output.first_counter_not_empty), cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            workspace_handles.fixed[filter_counter_clear_slot],
            &input.second_counter_not_empty,
            sizeof(input.second_counter_not_empty), cudaMemcpyHostToDevice);
    if (runtime_status != cudaSuccess) {
        error = std::string("resetting recovered resident clear counter failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    if (!for_each_batch([&](std::uint32_t offset,
                             std::uint32_t work_items) {
            void* arguments[] = {
                &device_depth, &device_normal, &device_estimated,
                &estimate_normal, &device_mask,
                const_cast<PatchMatchCamera*>(&input.camera), &downscale,
                &workspace_handles.fixed[
                    filter_counter_inconsistent_normal_slot],
                &workspace_handles.fixed[
                    filter_counter_bad_estimated_normal_slot],
                &workspace_handles.fixed[filter_counter_bad_found_normal_slot],
                &workspace_handles.fixed[filter_counter_cos_sum_slot],
                &workspace_handles.fixed[filter_counter_ncos_sum_slot],
                &offset,
            };
            return launch(functions[3], work_items, arguments,
                          "recovered resident normal filter");
        }) ||
        !for_each_batch([&](std::uint32_t offset,
                             std::uint32_t work_items) {
            void* arguments[] = {
                &device_depth, &device_mask,
                const_cast<PatchMatchCamera*>(&input.camera), &downscale,
                &workspace_handles.fixed[filter_counter_clear_slot], &offset,
            };
            return launch(functions[2], work_items, arguments,
                          "recovered resident second clear-depth filter");
        }) ||
        !for_each_batch([&](std::uint32_t offset,
                             std::uint32_t work_items) {
            void* arguments[] = {
                &device_depth, &device_mask,
                const_cast<PatchMatchCamera*>(&input.camera), &downscale,
                &offset,
            };
            return launch(functions[4], work_items, arguments,
                          "recovered resident speckles-edge filter");
        })) {
        release();
        return false;
    }

    auto download = [&](void* destination, std::size_t slot) {
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                destination, workspace_handles.fixed[slot],
                workspace_bytes[slot], cudaMemcpyDeviceToHost);
    };
    download(output.depth_allocation.data(), cost_depth_slot);
    download(output.estimated_normal_allocation.data(),
             filter_estimated_normal_slot);
    download(output.filtered_mask_allocation.data(), filter_mask_slot);
    if (!output.neighbor_inlier_masks_allocation.empty())
        download(output.neighbor_inlier_masks_allocation.data(),
                 pipeline_inlier_masks_slot);
    download(&output.counter_no_cost_samples, filter_counter_no_cost_slot);
    download(&output.counter_big_cost_samples, filter_counter_big_cost_slot);
    download(&output.counter_no_neighbours,
             filter_counter_no_neighbours_slot);
    download(&output.counter_no_close_neighbours,
             filter_counter_no_close_neighbours_slot);
    download(&output.second_counter_not_empty, filter_counter_clear_slot);
    download(&output.counter_inconsistent_normal,
             filter_counter_inconsistent_normal_slot);
    download(&output.counter_bad_view_angle_estimated_normal,
             filter_counter_bad_estimated_normal_slot);
    download(&output.counter_bad_view_angle_found_normal,
             filter_counter_bad_found_normal_slot);
    download(&output.counter_cos_sum, filter_counter_cos_sum_slot);
    download(&output.counter_ncos_sum, filter_counter_ncos_sum_slot);
    if (runtime_status != cudaSuccess) {
        error = std::string("downloading recovered resident filter output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    if (consume_pipeline_handoff) {
        output.cuda_workspace_handoff =
            publish_recovered_cuda_workspace_handoff(
                4U, input.cuda_workspace_handoff, 6U);
        if (output.cuda_workspace_handoff == 0U) {
            error = "publishing recovered cross-level filter handoff failed";
            release();
            return false;
        }
        completed_handoff = true;
    }
    release();
    error.clear();
    return true;
}

bool run_recovered_patchmatch_filter_check_cost_cuda_impl(
    const PatchMatchFilterCheckCostInput& input,
    PatchMatchFilterCheckCostOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::filter);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch filter depth downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (pixels == 0 || work_items == 0 ||
        input.pixel_offset + work_items > pixels ||
        input.depth_allocation.size() < pixels ||
        input.cost_allocation.size() < pixels ||
        input.depth_allocation.size() != input.cost_allocation.size()) {
        error = "PatchMatch filter check-cost launch or allocation is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA filter context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_FILTER_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda23pm_filtering_check_costEPfPKf13PinholeCamerajPiS4_j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered PatchMatch filter PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.depth_allocation = input.depth_allocation;
    output.counter_no_cost_samples = input.counter_no_cost_samples;
    output.counter_big_cost_samples = input.counter_big_cost_samples;
    void* device_depth = nullptr;
    void* device_cost = nullptr;
    void* device_counter_no_cost = nullptr;
    void* device_counter_big_cost = nullptr;
    auto release = [&]() {
        cudaFree(device_depth);
        cudaFree(device_cost);
        cudaFree(device_counter_no_cost);
        cudaFree(device_counter_big_cost);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    const std::size_t allocation_bytes =
        input.depth_allocation.size() * sizeof(float);
    runtime_status = allocate_and_copy(
        &device_depth, input.depth_allocation.data(), allocation_bytes);
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_cost, input.cost_allocation.data(), allocation_bytes);
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_counter_no_cost, &output.counter_no_cost_samples,
            sizeof(output.counter_no_cost_samples));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_counter_big_cost, &output.counter_big_cost_samples,
            sizeof(output.counter_big_cost_samples));
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered filter buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth,
        &device_cost,
        const_cast<PatchMatchCamera*>(&input.camera),
        &downscale32,
        &device_counter_no_cost,
        &device_counter_big_cost,
        &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered PatchMatch filter launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.depth_allocation.data(), device_depth, allocation_bytes,
        cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            &output.counter_no_cost_samples, device_counter_no_cost,
            sizeof(output.counter_no_cost_samples), cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            &output.counter_big_cost_samples, device_counter_big_cost,
            sizeof(output.counter_big_cost_samples), cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered filter output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_filter_check_neighbours_cuda_impl(
    const PatchMatchFilterCheckNeighboursInput& input,
    PatchMatchFilterCheckNeighboursOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::filter);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch neighbour filter downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (pixels == 0 || work_items == 0 ||
        input.pixel_offset + work_items > pixels ||
        input.depth_allocation.size() < pixels ||
        input.filtered_mask_allocation.size() < pixels) {
        error = "PatchMatch neighbour filter launch or allocation is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA neighbour-filter context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_FILTER_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda29pm_filtering_check_neighboursEPKfPh13PinholeCamerajffPiS4_j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered neighbour-filter PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.filtered_mask_allocation = input.filtered_mask_allocation;
    output.counter_no_neighbours = input.counter_no_neighbours;
    output.counter_no_close_neighbours = input.counter_no_close_neighbours;
    void* device_depth = nullptr;
    void* device_mask = nullptr;
    void* device_counter_no_neighbours = nullptr;
    void* device_counter_no_close = nullptr;
    auto release = [&]() {
        cudaFree(device_depth);
        cudaFree(device_mask);
        cudaFree(device_counter_no_neighbours);
        cudaFree(device_counter_no_close);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_depth, input.depth_allocation.data(),
        input.depth_allocation.size() * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_mask, input.filtered_mask_allocation.data(),
            input.filtered_mask_allocation.size());
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_counter_no_neighbours, &output.counter_no_neighbours,
            sizeof(output.counter_no_neighbours));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_counter_no_close, &output.counter_no_close_neighbours,
            sizeof(output.counter_no_close_neighbours));
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered neighbour-filter buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    float depth_min = input.depth_min;
    float depth_max = input.depth_max;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth,
        &device_mask,
        const_cast<PatchMatchCamera*>(&input.camera),
        &downscale32,
        &depth_min,
        &depth_max,
        &device_counter_no_neighbours,
        &device_counter_no_close,
        &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered neighbour-filter launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.filtered_mask_allocation.data(), device_mask,
        output.filtered_mask_allocation.size(), cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            &output.counter_no_neighbours, device_counter_no_neighbours,
            sizeof(output.counter_no_neighbours), cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            &output.counter_no_close_neighbours, device_counter_no_close,
            sizeof(output.counter_no_close_neighbours), cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered neighbour-filter output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_filter_clear_depth_cuda_impl(
    const PatchMatchFilterClearDepthInput& input,
    PatchMatchFilterClearDepthOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::filter);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch clear-depth filter downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (pixels == 0 || work_items == 0 ||
        input.pixel_offset + work_items > pixels ||
        input.depth_allocation.size() < pixels ||
        input.filtered_mask_allocation.size() < pixels) {
        error = "PatchMatch clear-depth launch or allocation is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA clear-depth context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_FILTER_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda46pm_filtering_clear_depth_map_wrt_filtered_maskEPfPKh13PinholeCamerajPij";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered clear-depth PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.depth_allocation = input.depth_allocation;
    output.counter_not_empty = input.counter_not_empty;
    void* device_depth = nullptr;
    void* device_mask = nullptr;
    void* device_counter = nullptr;
    auto release = [&]() {
        cudaFree(device_depth);
        cudaFree(device_mask);
        cudaFree(device_counter);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_depth, input.depth_allocation.data(),
        input.depth_allocation.size() * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_mask, input.filtered_mask_allocation.data(),
            input.filtered_mask_allocation.size());
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_counter, &output.counter_not_empty,
            sizeof(output.counter_not_empty));
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered clear-depth buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth,
        &device_mask,
        const_cast<PatchMatchCamera*>(&input.camera),
        &downscale32,
        &device_counter,
        &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered clear-depth launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.depth_allocation.data(), device_depth,
        output.depth_allocation.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            &output.counter_not_empty, device_counter,
            sizeof(output.counter_not_empty), cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered clear-depth output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_filter_normals_cuda_impl(
    const PatchMatchFilterNormalsInput& input,
    PatchMatchFilterNormalsOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::filter);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch normal-filter downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (pixels == 0 || work_items == 0 ||
        input.pixel_offset + work_items > pixels ||
        input.depth_allocation.size() < pixels ||
        input.normal_allocation.size() < pixels * 3 ||
        (input.estimate_normal_map &&
         input.estimated_normal_allocation.size() < pixels * 3) ||
        input.estimated_normal_allocation.empty() ||
        input.filtered_mask_allocation.size() < pixels) {
        error = "PatchMatch normal-filter launch or allocation is invalid";
        return false;
    }
    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA normal-filter context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_FILTER_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda20pm_filtering_normalsEPKfPK6uchar3PS2_hPh13PinholeCamerajPiS8_S8_PfS8_j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered normal-filter PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.estimated_normal_allocation = input.estimated_normal_allocation;
    output.filtered_mask_allocation = input.filtered_mask_allocation;
    output.counter_inconsistent_normal = input.counter_inconsistent_normal;
    output.counter_bad_view_angle_estimated_normal =
        input.counter_bad_view_angle_estimated_normal;
    output.counter_bad_view_angle_found_normal =
        input.counter_bad_view_angle_found_normal;
    output.counter_cos_sum = input.counter_cos_sum;
    output.counter_ncos_sum = input.counter_ncos_sum;
    void* device_depth = nullptr;
    void* device_normal = nullptr;
    void* device_estimated_normal = nullptr;
    void* device_mask = nullptr;
    void* device_counters[5]{};
    auto release = [&]() {
        cudaFree(device_depth);
        cudaFree(device_normal);
        cudaFree(device_estimated_normal);
        cudaFree(device_mask);
        for (void* pointer : device_counters) cudaFree(pointer);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_depth, input.depth_allocation.data(),
        input.depth_allocation.size() * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_normal, input.normal_allocation.data(),
            input.normal_allocation.size());
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_estimated_normal, output.estimated_normal_allocation.data(),
            output.estimated_normal_allocation.size());
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_mask, output.filtered_mask_allocation.data(),
            output.filtered_mask_allocation.size());
    void* host_counters[5] = {
        &output.counter_inconsistent_normal,
        &output.counter_bad_view_angle_estimated_normal,
        &output.counter_bad_view_angle_found_normal,
        &output.counter_cos_sum,
        &output.counter_ncos_sum,
    };
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 5; ++i)
        runtime_status = allocate_and_copy(
            &device_counters[i], host_counters[i], sizeof(std::uint32_t));
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered normal-filter buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint8_t estimate_normal = input.estimate_normal_map ? 1 : 0;
    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth,
        &device_normal,
        &device_estimated_normal,
        &estimate_normal,
        &device_mask,
        const_cast<PatchMatchCamera*>(&input.camera),
        &downscale32,
        &device_counters[0],
        &device_counters[1],
        &device_counters[2],
        &device_counters[3],
        &device_counters[4],
        &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered normal-filter launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.estimated_normal_allocation.data(), device_estimated_normal,
        output.estimated_normal_allocation.size(), cudaMemcpyDeviceToHost);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            output.filtered_mask_allocation.data(), device_mask,
            output.filtered_mask_allocation.size(), cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 5; ++i)
        runtime_status = cudaMemcpy(
            host_counters[i], device_counters[i], sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered normal-filter output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_patchmatch_filter_speckles_edges_cuda_impl(
    const PatchMatchFilterSpecklesEdgesInput& input,
    PatchMatchFilterSpecklesEdgesOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::filter);
    const std::size_t downscale = input.depth_downscale;
    if (downscale == 0) {
        error = "PatchMatch speckles-edge filter downscale must be nonzero";
        return false;
    }
    const std::size_t width =
        (input.camera.width_original + downscale - 1) / downscale;
    const std::size_t height =
        (input.camera.height_original + downscale - 1) / downscale;
    const std::size_t pixels = width * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (pixels == 0 || work_items == 0 ||
        input.pixel_offset + work_items > pixels ||
        input.depth_allocation.size() < pixels ||
        input.filtered_mask_allocation.size() < pixels) {
        error = "PatchMatch speckles-edge launch or allocation is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA speckles-edge context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_PM_FILTER_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda36pm_filtering_estimate_speckles_edgesEPKfPh13PinholeCamerajj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered speckles-edge PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.filtered_mask_allocation = input.filtered_mask_allocation;
    void* device_depth = nullptr;
    void* device_mask = nullptr;
    auto release = [&]() {
        cudaFree(device_depth);
        cudaFree(device_mask);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes,
                                cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_depth, input.depth_allocation.data(),
        input.depth_allocation.size() * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_mask, output.filtered_mask_allocation.data(),
            output.filtered_mask_allocation.size());
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered speckles-edge buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t downscale32 = input.depth_downscale;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth,
        &device_mask,
        const_cast<PatchMatchCamera*>(&input.camera),
        &downscale32,
        &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered speckles-edge launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.filtered_mask_allocation.data(), device_mask,
        output.filtered_mask_allocation.size(), cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered speckles-edge output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_depth_radius_estimate_cuda_impl(
    const DepthRadiusEstimateInput& input,
    std::vector<float>& radius_output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::voting);
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    static_assert(sizeof(width) == 4);
    std::memcpy(&width, input.calibration.bytes.data() + 84, sizeof(width));
    std::memcpy(&height, input.calibration.bytes.data() + 88, sizeof(height));
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (width == 0 || height == 0 || pixels == 0 || work_items == 0 ||
        static_cast<std::size_t>(input.level_offset) + pixels >
            input.depth_allocation.size() ||
        static_cast<std::size_t>(input.level_offset) + pixels >
            input.radius_allocation.size() ||
        static_cast<std::size_t>(input.kernel_offset) + work_items > pixels ||
        input.depth_allocation.size() != input.radius_allocation.size()) {
        error = "depth-radius estimate launch or persistent allocation is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA depth-radius context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_DEPTH_RADIUS_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda28estimateDepthMapSampleRadiusEPKfPfjNS_13CalibrationCuENS_10Matrix4x4fEj";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered depth-radius PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    radius_output = input.radius_allocation;
    void* device_depth = nullptr;
    void* device_radius = nullptr;
    auto release = [&]() {
        cudaFree(device_depth);
        cudaFree(device_radius);
        cuModuleUnload(module);
    };
    const std::size_t allocation_bytes =
        input.depth_allocation.size() * sizeof(float);
    runtime_status = cudaMalloc(&device_depth, allocation_bytes);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMalloc(&device_radius, allocation_bytes);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            device_depth, input.depth_allocation.data(), allocation_bytes,
            cudaMemcpyHostToDevice);
    if (runtime_status == cudaSuccess)
        runtime_status = cudaMemcpy(
            device_radius, radius_output.data(), allocation_bytes,
            cudaMemcpyHostToDevice);
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered depth-radius buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t level_offset = input.level_offset;
    std::uint32_t kernel_offset = input.kernel_offset;
    void* arguments[] = {
        &device_depth,
        &device_radius,
        &level_offset,
        const_cast<DepthVotingCalibrationCu*>(&input.calibration),
        const_cast<DepthVotingMatrix4x4f*>(&input.transform),
        &kernel_offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(
        function, blocks, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered depth-radius launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        radius_output.data(), device_radius, allocation_bytes,
        cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered depth-radius output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_depth_neighbor_votes_cuda_impl(
    const DepthNeighborVotesInput& input,
    DepthNeighborVotesOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::voting);
    auto calibration_u32 = [](const DepthVotingCalibrationCu& calibration,
                              std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, calibration.bytes.data() + offset, sizeof(value));
        return value;
    };
    const std::uint32_t width =
        calibration_u32(input.reference_calibration, 84);
    const std::uint32_t height =
        calibration_u32(input.reference_calibration, 88);
    const std::uint32_t neighbor_width =
        calibration_u32(input.neighbor_calibration, 84);
    const std::uint32_t neighbor_height =
        calibration_u32(input.neighbor_calibration, 88);
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::size_t neighbor_pyramid_pixels = 0;
    std::size_t level_width = neighbor_width;
    std::size_t level_height = neighbor_height;
    for (std::uint32_t level = 0; level < input.neighbor_levels; ++level) {
        neighbor_pyramid_pixels += level_width * level_height;
        level_width = (level_width + 1) / 2;
        level_height = (level_height + 1) / 2;
    }
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (width == 0 || height == 0 || neighbor_width == 0 ||
        neighbor_height == 0 || input.neighbor_levels == 0 ||
        work_items == 0 ||
        static_cast<std::size_t>(input.kernel_offset) + work_items > pixels ||
        input.votes.size() != pixels ||
        input.reference_depth.size() != pixels ||
        input.reference_radius.size() != pixels ||
        input.neighbor_inlier_mask.size() != pixels ||
        input.neighbor_depth_levels.size() != neighbor_pyramid_pixels ||
        input.neighbor_radius_levels.size() != neighbor_pyramid_pixels) {
        error = "direct depth-voting launch or resource size is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA direct depth-voting context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status =
        cuModuleLoad(&module, METMODEL_DEPTH_NEIGHBOR_VOTES_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda16addNeighborVotesEPiPKfS2_PKhS2_S2_iNS_13CalibrationCuENS_10Matrix4x4fEiS5_S0_S0_S0_S0_S0_S0_S0_j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered direct depth-voting PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.votes = input.votes;
    output.counters = input.counters;
    void* device_votes = nullptr;
    void* device_reference_depth = nullptr;
    void* device_reference_radius = nullptr;
    void* device_neighbor_inlier_mask = nullptr;
    void* device_neighbor_depth = nullptr;
    void* device_neighbor_radius = nullptr;
    std::array<void*, 7> device_counters{};
    auto release = [&]() {
        cudaFree(device_votes);
        cudaFree(device_reference_depth);
        cudaFree(device_reference_radius);
        cudaFree(device_neighbor_inlier_mask);
        cudaFree(device_neighbor_depth);
        cudaFree(device_neighbor_radius);
        for (void* counter : device_counters) cudaFree(counter);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [](void** destination, const void* source,
                                std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(
                *destination, source, bytes, cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_votes, output.votes.data(), pixels * sizeof(std::int32_t));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_reference_depth, input.reference_depth.data(),
            pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_reference_radius, input.reference_radius.data(),
            pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_neighbor_inlier_mask, input.neighbor_inlier_mask.data(), pixels);
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_neighbor_depth, input.neighbor_depth_levels.data(),
            neighbor_pyramid_pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_neighbor_radius, input.neighbor_radius_levels.data(),
            neighbor_pyramid_pixels * sizeof(float));
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 7; ++i)
        runtime_status = allocate_and_copy(
            &device_counters[i], &output.counters[i], sizeof(std::int32_t));
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered direct voting buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t reference_level = input.reference_level;
    std::uint32_t neighbor_levels = input.neighbor_levels;
    std::uint32_t kernel_offset = input.kernel_offset;
    void* arguments[] = {
        &device_votes,
        &device_reference_depth,
        &device_reference_radius,
        &device_neighbor_inlier_mask,
        &device_neighbor_depth,
        &device_neighbor_radius,
        &reference_level,
        const_cast<DepthVotingCalibrationCu*>(&input.reference_calibration),
        const_cast<DepthVotingMatrix4x4f*>(&input.reference_to_neighbor),
        &neighbor_levels,
        const_cast<DepthVotingCalibrationCu*>(&input.neighbor_calibration),
        &device_counters[0], &device_counters[1], &device_counters[2],
        &device_counters[3], &device_counters[4], &device_counters[5],
        &device_counters[6], &kernel_offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(
        function, blocks, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered direct depth-voting launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.votes.data(), device_votes, pixels * sizeof(std::int32_t),
        cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 7; ++i)
        runtime_status = cudaMemcpy(
            &output.counters[i], device_counters[i], sizeof(std::int32_t),
            cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered direct voting output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_depth_neighbor_occlusion_votes_cuda_impl(
    const DepthNeighborOcclusionVotesInput& input,
    DepthNeighborOcclusionVotesOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::voting);
    auto calibration_u32 = [](const DepthVotingCalibrationCu& calibration,
                              std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, calibration.bytes.data() + offset, sizeof(value));
        return value;
    };
    const std::uint32_t width = calibration_u32(input.reference_calibration, 84);
    const std::uint32_t height = calibration_u32(input.reference_calibration, 88);
    const std::uint32_t neighbor_width =
        calibration_u32(input.neighbor_calibration, 84);
    const std::uint32_t neighbor_height =
        calibration_u32(input.neighbor_calibration, 88);
    const std::size_t reference_pixels =
        static_cast<std::size_t>(width) * height;
    std::size_t neighbor_pyramid_pixels = 0;
    std::size_t selected_level_pixels = 0;
    if (input.neighbor_level < 31) {
        for (std::uint32_t level = 0; level <= input.neighbor_level; ++level) {
            const std::uint32_t downscale = std::uint32_t{1} << level;
            const std::size_t level_width =
                (static_cast<std::size_t>(neighbor_width) + downscale - 1) /
                downscale;
            const std::size_t level_height =
                (static_cast<std::size_t>(neighbor_height) + downscale - 1) /
                downscale;
            selected_level_pixels = level_width * level_height;
            neighbor_pyramid_pixels += selected_level_pixels;
        }
    }
    const std::size_t work_items = input.global_work_items == 0
        ? selected_level_pixels : input.global_work_items;
    if (width == 0 || height == 0 || neighbor_width == 0 ||
        neighbor_height == 0 || input.neighbor_level >= 31 ||
        work_items == 0 ||
        static_cast<std::size_t>(input.kernel_offset) + work_items >
            selected_level_pixels ||
        input.votes.size() != reference_pixels ||
        input.reference_depth.size() != reference_pixels ||
        input.reference_radius.size() != reference_pixels ||
        input.neighbor_inlier_mask.size() != reference_pixels ||
        input.neighbor_depth_levels.size() != neighbor_pyramid_pixels ||
        input.neighbor_radius_levels.size() != neighbor_pyramid_pixels) {
        error = "occlusion depth-voting launch or resource size is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA occlusion voting context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status =
        cuModuleLoad(&module, METMODEL_DEPTH_NEIGHBOR_VOTES_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda25addNeighborOcclusionVotesEPiPKfS2_PKhiS2_S2_iNS_13CalibrationCuENS_10Matrix4x4fES5_S0_S0_j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered occlusion voting PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.votes = input.votes;
    output.counters = input.counters;
    void* device_votes = nullptr;
    void* device_reference_depth = nullptr;
    void* device_reference_radius = nullptr;
    void* device_neighbor_mask = nullptr;
    void* device_neighbor_depth = nullptr;
    void* device_neighbor_radius = nullptr;
    std::array<void*, 2> device_counters{};
    auto release = [&]() {
        cudaFree(device_votes);
        cudaFree(device_reference_depth);
        cudaFree(device_reference_radius);
        cudaFree(device_neighbor_mask);
        cudaFree(device_neighbor_depth);
        cudaFree(device_neighbor_radius);
        for (void* counter : device_counters) cudaFree(counter);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [](void** destination, const void* source,
                                std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(
                *destination, source, bytes, cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_votes, output.votes.data(), reference_pixels * sizeof(std::int32_t));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_reference_depth, input.reference_depth.data(),
            reference_pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_reference_radius, input.reference_radius.data(),
            reference_pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_neighbor_mask, input.neighbor_inlier_mask.data(),
            reference_pixels);
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_neighbor_depth, input.neighbor_depth_levels.data(),
            neighbor_pyramid_pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_neighbor_radius, input.neighbor_radius_levels.data(),
            neighbor_pyramid_pixels * sizeof(float));
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 2; ++i)
        runtime_status = allocate_and_copy(
            &device_counters[i], &output.counters[i], sizeof(std::int32_t));
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered occlusion voting buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t neighbor_level = input.neighbor_level;
    std::uint32_t reference_level = input.reference_level;
    std::uint32_t kernel_offset = input.kernel_offset;
    void* arguments[] = {
        &device_votes,
        &device_reference_depth,
        &device_reference_radius,
        &device_neighbor_mask,
        &neighbor_level,
        &device_neighbor_depth,
        &device_neighbor_radius,
        &reference_level,
        const_cast<DepthVotingCalibrationCu*>(&input.reference_calibration),
        const_cast<DepthVotingMatrix4x4f*>(&input.neighbor_to_reference),
        const_cast<DepthVotingCalibrationCu*>(&input.neighbor_calibration),
        &device_counters[0],
        &device_counters[1],
        &kernel_offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(
        function, blocks, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered occlusion voting launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.votes.data(), device_votes,
        reference_pixels * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 2; ++i)
        runtime_status = cudaMemcpy(
            &output.counters[i], device_counters[i], sizeof(std::int32_t),
            cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered occlusion voting output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool run_recovered_depth_voting_finalize_cuda_impl(
    const DepthVotingFinalizeInput& input,
    DepthVotingFinalizeOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::voting);
    const std::size_t pixels =
        static_cast<std::size_t>(input.width) * input.height;
    const std::size_t work_items =
        input.global_work_items == 0 ? pixels : input.global_work_items;
    if (input.width == 0 || input.height == 0 || work_items == 0 ||
        input.pixel_offset + work_items > pixels ||
        input.depth_allocation.size() != pixels ||
        input.votes_allocation.size() != pixels) {
        error = "depth-voting finalization launch or allocation is invalid";
        return false;
    }

    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA depth-voting context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult driver_status = cuModuleLoad(&module, METMODEL_DEPTH_VOTING_CUDA_PTX);
    constexpr const char* kernel_name =
        "_ZN4cuda19filterBasedOnVotingEPfPKijjPiS3_S3_S3_j";
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(&function, module, kernel_name);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered depth-voting PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (module) cuModuleUnload(module);
        return false;
    }

    output.depth_allocation = input.depth_allocation;
    output.counter_empty = input.counter_empty;
    output.counter_bad = input.counter_bad;
    output.counter_normal = input.counter_normal;
    output.counter_good = input.counter_good;
    void* device_depth = nullptr;
    void* device_votes = nullptr;
    void* device_counters[4]{};
    auto release = [&]() {
        cudaFree(device_depth);
        cudaFree(device_votes);
        for (void* counter : device_counters) cudaFree(counter);
        cuModuleUnload(module);
    };
    auto allocate_and_copy = [&](void** destination, const void* source,
                                 std::size_t bytes) {
        cudaError_t status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice);
        return status;
    };
    runtime_status = allocate_and_copy(
        &device_depth, output.depth_allocation.data(), pixels * sizeof(float));
    if (runtime_status == cudaSuccess)
        runtime_status = allocate_and_copy(
            &device_votes, input.votes_allocation.data(),
            pixels * sizeof(std::int32_t));
    std::int32_t* host_counters[4] = {
        &output.counter_empty, &output.counter_bad,
        &output.counter_normal, &output.counter_good,
    };
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 4; ++i)
        runtime_status = allocate_and_copy(
            &device_counters[i], host_counters[i], sizeof(std::int32_t));
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered depth-voting buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    std::uint32_t width = input.width;
    std::uint32_t height = input.height;
    std::uint32_t offset = input.pixel_offset;
    void* arguments[] = {
        &device_depth, &device_votes, &width, &height,
        &device_counters[0], &device_counters[1],
        &device_counters[2], &device_counters[3], &offset,
    };
    const unsigned int blocks =
        static_cast<unsigned int>((work_items + 127) / 128);
    driver_status = cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                                   nullptr, arguments, nullptr);
    if (driver_status == CUDA_SUCCESS) driver_status = cuCtxSynchronize();
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("recovered depth-voting launch failed: ") +
                (message ? message : "unknown CUDA driver error");
        release();
        return false;
    }

    runtime_status = cudaMemcpy(
        output.depth_allocation.data(), device_depth, pixels * sizeof(float),
        cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; runtime_status == cudaSuccess && i < 4; ++i)
        runtime_status = cudaMemcpy(
            host_counters[i], device_counters[i], sizeof(std::int32_t),
            cudaMemcpyDeviceToHost);
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered depth-voting output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }
    release();
    return true;
}

bool prime_recovered_cuda_voting_depth_cache_impl(
    std::span<const std::span<const float>> depth_levels,
    std::size_t device_index,
    std::string& error) {
    if (depth_levels.empty() ||
        device_index >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = "recovered CUDA voting depth cache input is invalid";
        return false;
    }
    cudaError_t runtime_status = cudaSetDevice(static_cast<int>(device_index));
    if (runtime_status != cudaSuccess) {
        error = std::string("selecting recovered CUDA voting cache device failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUcontext context = nullptr;
    const CUresult context_status = ::cuCtxGetCurrent(&context);
    if (context_status != CUDA_SUCCESS || context == nullptr) {
        error = "recovered CUDA voting depth cache lacks a current context";
        return false;
    }

    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active || session.context != context ||
        session.stats.device_index != device_index) {
        error = "recovered CUDA voting depth cache requires its active scene session";
        return false;
    }
    std::vector<const float*> created;
    auto rollback = [&]() {
        for (const float* host_pointer : created) {
            const auto found = session.voting_depth_cache.find(host_pointer);
            if (found == session.voting_depth_cache.end()) continue;
            if (found->second.device_pointer != nullptr) {
                ::cuMemFree(reinterpret_cast<CUdeviceptr>(
                    found->second.device_pointer));
                ++session.stats.cuda_free_calls;
            }
            session.voting_depth_cache.erase(found);
        }
    };
    for (const std::span<const float> level : depth_levels) {
        if (level.empty() || level.data() == nullptr ||
            level.size() > std::numeric_limits<std::size_t>::max() /
                               sizeof(float)) {
            rollback();
            error = "recovered CUDA voting depth cache contains an empty or oversized level";
            return false;
        }
        const std::size_t bytes = level.size() * sizeof(float);
        const auto existing = session.voting_depth_cache.find(level.data());
        if (existing != session.voting_depth_cache.end()) {
            if (existing->second.bytes != bytes) {
                rollback();
                error = "recovered CUDA voting depth cache pointer changed size";
                return false;
            }
            continue;
        }
        CUdeviceptr device_pointer = 0U;
        const CUresult allocation_status =
            ::cuMemAlloc(&device_pointer, bytes);
        ++session.stats.cuda_malloc_calls;
        session.stats.cuda_malloc_bytes += bytes;
        if (allocation_status != CUDA_SUCCESS) {
            rollback();
            const char* message = nullptr;
            ::cuGetErrorString(allocation_status, &message);
            error = std::string("allocating recovered CUDA voting depth cache failed: ") +
                    (message ? message : "unknown CUDA driver error");
            return false;
        }
        const auto started = std::chrono::steady_clock::now();
        const CUresult upload_status =
            ::cuMemcpyHtoD(device_pointer, level.data(), bytes);
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        ++session.stats.host_to_device_copy_calls;
        session.stats.host_to_device_copy_bytes += bytes;
        session.stats.host_to_device_copy_nanoseconds += elapsed;
        ++session.stats.voting_h2d_calls;
        session.stats.voting_h2d_bytes += bytes;
        if (upload_status != CUDA_SUCCESS) {
            ::cuMemFree(device_pointer);
            ++session.stats.cuda_free_calls;
            rollback();
            const char* message = nullptr;
            ::cuGetErrorString(upload_status, &message);
            error = std::string("uploading recovered CUDA voting depth cache failed: ") +
                    (message ? message : "unknown CUDA driver error");
            return false;
        }
        session.voting_depth_cache.emplace(
            level.data(),
            RecoveredCudaVotingDepthCacheEntry{
                reinterpret_cast<void*>(device_pointer), bytes});
        created.push_back(level.data());
        ++session.stats.voting_depth_cache_entries;
        session.stats.voting_depth_cache_bytes += bytes;
    }
    error.clear();
    return true;
}

bool clear_recovered_cuda_voting_depth_cache_impl(
    std::size_t device_index,
    std::string& error) {
    if (device_index > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = "recovered CUDA voting depth cache device index is out of range";
        return false;
    }
    const cudaError_t runtime_status = cudaSetDevice(
        static_cast<int>(device_index));
    if (runtime_status != cudaSuccess) {
        error = std::string("selecting recovered CUDA voting cache device failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }
    CUcontext context = nullptr;
    const CUresult context_status = ::cuCtxGetCurrent(&context);
    if (context_status != CUDA_SUCCESS || context == nullptr) {
        error = "recovered CUDA voting depth cache lacks a current context";
        return false;
    }

    std::lock_guard lock(recovered_cuda_module_session_mutex);
    auto& session = recovered_cuda_module_session;
    if (!session.active || session.context != context ||
        session.stats.device_index != device_index) {
        error = "recovered CUDA voting depth cache requires its active scene session";
        return false;
    }
    const bool workspace_in_use = std::any_of(
        session.voting_workspaces.begin(), session.voting_workspaces.end(),
        [](const auto& entry) { return entry.second.in_use; });
    if (workspace_in_use) {
        error = "recovered CUDA voting depth cache cannot clear while a voting workspace is in use";
        return false;
    }
    const CUresult status =
        destroy_recovered_cuda_voting_depth_cache_locked(session);
    if (status != CUDA_SUCCESS) {
        const char* message = nullptr;
        ::cuGetErrorString(status, &message);
        error = std::string("clearing recovered CUDA voting depth cache failed: ") +
                (message ? message : "unknown CUDA driver error");
        return false;
    }
    error.clear();
    return true;
}

bool run_recovered_depth_voting_chain_cuda_impl(
    const DepthVotingChainInput& input,
    DepthVotingChainOutput& output,
    std::string& error) {
    const RecoveredCudaTransferPhaseScope transfer_phase(
        RecoveredCudaTransferPhase::voting);
    constexpr std::size_t level_count = 3U;
    auto calibration_u32 = [](const DepthVotingCalibrationCu& calibration,
                              std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, calibration.bytes.data() + offset, sizeof(value));
        return value;
    };
    auto level_pixels = [](std::uint32_t width, std::uint32_t height,
                           std::uint32_t level) {
        const std::size_t divisor = std::size_t{1} << level;
        return ((static_cast<std::size_t>(width) + divisor - 1U) / divisor) *
               ((static_cast<std::size_t>(height) + divisor - 1U) / divisor);
    };
    auto pyramid_pixels = [&](std::uint32_t width, std::uint32_t height,
                              std::uint32_t levels) {
        std::size_t pixels = 0;
        for (std::uint32_t level = 0; level < levels; ++level)
            pixels += level_pixels(width, height, level);
        return pixels;
    };
    auto checked_work_items = [](std::size_t requested,
                                 std::size_t default_items,
                                 std::uint32_t offset,
                                 std::string_view label,
                                 std::string& validation_error) {
        const std::size_t work_items = requested == 0U ? default_items : requested;
        if (work_items == 0U ||
            static_cast<std::size_t>(offset) + work_items > default_items ||
            work_items > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) *
                             128U) {
            validation_error = std::string(label) + " launch range is invalid";
            return std::size_t{0};
        }
        return work_items;
    };

    if (input.device_index >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = "depth-voting chain device index exceeds int range";
        return false;
    }
    if (input.neighbors.empty()) {
        error = "depth-voting chain requires at least one neighbor";
        return false;
    }

    std::array<std::size_t, level_count> reference_pixels{};
    std::size_t largest_neighbor_pixels = 0U;
    for (std::size_t level = 0; level < level_count; ++level) {
        const auto& reference = input.reference[level];
        const std::span<const float> reference_depth =
            reference.depth_view.empty()
                ? std::span<const float>(reference.depth)
                : reference.depth_view;
        reference_pixels[level] =
            static_cast<std::size_t>(reference.width) * reference.height;
        const std::uint32_t calibration_width =
            calibration_u32(reference.radius.calibration, 84U);
        const std::uint32_t calibration_height =
            calibration_u32(reference.radius.calibration, 88U);
        if (reference.width == 0U || reference.height == 0U ||
            reference_depth.size() != reference_pixels[level] ||
            (!reference.initial_votes.empty() &&
             reference.initial_votes.size() != reference_pixels[level]) ||
            calibration_width != reference.width ||
            calibration_height != reference.height ||
            static_cast<std::size_t>(reference.radius.level_offset) +
                    reference_pixels[level] >
                reference_depth.size() ||
            checked_work_items(reference.radius.global_work_items,
                               reference_pixels[level],
                               reference.radius.kernel_offset,
                               "reference radius", error) == 0U) {
            if (error.empty())
                error = "depth-voting chain reference resource size is invalid";
            return false;
        }
    }
    for (const auto& neighbor : input.neighbors) {
        const bool use_depth_views = neighbor.depth_levels.empty() &&
            std::all_of(neighbor.depth_level_views.begin(),
                        neighbor.depth_level_views.end(),
                        [](std::span<const float> level) {
                            return !level.empty();
                        });
        std::size_t neighbor_pyramid_size = neighbor.depth_levels.size();
        if (use_depth_views) {
            neighbor_pyramid_size = 0U;
            for (const std::span<const float> level :
                 neighbor.depth_level_views)
                neighbor_pyramid_size += level.size();
        }
        if (neighbor_pyramid_size == 0U) {
            error = "depth-voting chain neighbor depth pyramid is empty";
            return false;
        }
        largest_neighbor_pixels =
            std::max(largest_neighbor_pixels, neighbor_pyramid_size);
        const std::uint32_t base_width =
            calibration_u32(neighbor.radius[0].calibration, 84U);
        const std::uint32_t base_height =
            calibration_u32(neighbor.radius[0].calibration, 88U);
        if (base_width == 0U || base_height == 0U ||
            neighbor_pyramid_size !=
                pyramid_pixels(base_width, base_height,
                               static_cast<std::uint32_t>(level_count))) {
            error = "depth-voting chain neighbor pyramid size is invalid";
            return false;
        }
        for (std::size_t level = 0; level < level_count; ++level) {
            const std::size_t pixels = level_pixels(
                base_width, base_height, static_cast<std::uint32_t>(level));
            const std::span<const std::uint8_t> mask =
                neighbor.inlier_mask_views[level].empty()
                    ? std::span<const std::uint8_t>(
                          neighbor.inlier_masks[level])
                    : neighbor.inlier_mask_views[level];
            if (mask.size() !=
                    reference_pixels[level] ||
                static_cast<std::size_t>(neighbor.radius[level].level_offset) +
                        pixels >
                    neighbor_pyramid_size ||
                checked_work_items(neighbor.radius[level].global_work_items,
                                   pixels,
                                   neighbor.radius[level].kernel_offset,
                                   "neighbor radius", error) == 0U ||
                checked_work_items(neighbor.direct[level].global_work_items,
                                   reference_pixels[level],
                                   neighbor.direct[level].kernel_offset,
                                   "direct voting", error) == 0U) {
                if (error.empty())
                    error = "depth-voting chain neighbor resource size is invalid";
                return false;
            }
            for (std::size_t neighbor_level = 0;
                 neighbor_level < level_count; ++neighbor_level) {
                const std::size_t selected_pixels = level_pixels(
                    base_width, base_height,
                    static_cast<std::uint32_t>(neighbor_level));
                const auto& launch =
                    neighbor.occlusion[level][neighbor_level];
                if (checked_work_items(launch.global_work_items,
                                       selected_pixels,
                                       launch.kernel_offset,
                                       "occlusion voting", error) == 0U)
                    return false;
            }
        }
    }

    cudaError_t runtime_status =
        cudaSetDevice(static_cast<int>(input.device_index));
    if (runtime_status == cudaSuccess) runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        error = std::string("CUDA depth-voting chain context initialization failed: ") +
                cudaGetErrorString(runtime_status);
        return false;
    }

    CUmodule radius_module = nullptr;
    CUmodule neighbor_module = nullptr;
    CUmodule final_module = nullptr;
    CUfunction radius_function = nullptr;
    CUfunction direct_function = nullptr;
    CUfunction occlusion_function = nullptr;
    CUfunction final_function = nullptr;
    constexpr const char* radius_kernel =
        "_ZN4cuda28estimateDepthMapSampleRadiusEPKfPfjNS_13CalibrationCuENS_10Matrix4x4fEj";
    constexpr const char* direct_kernel =
        "_ZN4cuda16addNeighborVotesEPiPKfS2_PKhS2_S2_iNS_13CalibrationCuENS_10Matrix4x4fEiS5_S0_S0_S0_S0_S0_S0_S0_j";
    constexpr const char* occlusion_kernel =
        "_ZN4cuda25addNeighborOcclusionVotesEPiPKfS2_PKhiS2_S2_iNS_13CalibrationCuENS_10Matrix4x4fES5_S0_S0_j";
    constexpr const char* final_kernel =
        "_ZN4cuda19filterBasedOnVotingEPfPKijjPiS3_S3_S3_j";
    CUresult driver_status =
        cuModuleLoad(&radius_module, METMODEL_DEPTH_RADIUS_CUDA_PTX);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(
            &radius_function, radius_module, radius_kernel);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleLoad(
            &neighbor_module, METMODEL_DEPTH_NEIGHBOR_VOTES_CUDA_PTX);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(
            &direct_function, neighbor_module, direct_kernel);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(
            &occlusion_function, neighbor_module, occlusion_kernel);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleLoad(&final_module, METMODEL_DEPTH_VOTING_CUDA_PTX);
    if (driver_status == CUDA_SUCCESS)
        driver_status = cuModuleGetFunction(
            &final_function, final_module, final_kernel);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        error = std::string("loading recovered depth-voting chain PTX failed: ") +
                (message ? message : "unknown CUDA driver error");
        if (radius_module) cuModuleUnload(radius_module);
        if (neighbor_module) cuModuleUnload(neighbor_module);
        if (final_module) cuModuleUnload(final_module);
        return false;
    }

    std::array<std::size_t,
               RecoveredCudaVotingWorkspace::fixed_allocation_count>
        allocation_bytes{};
    for (std::size_t level = 0; level < level_count; ++level) {
        allocation_bytes[voting_reference_depth_0_slot + level] =
            reference_pixels[level] * sizeof(float);
        allocation_bytes[voting_reference_radius_0_slot + level] =
            reference_pixels[level] * sizeof(float);
        allocation_bytes[voting_reference_votes_0_slot + level] =
            reference_pixels[level] * sizeof(std::int32_t);
        allocation_bytes[voting_neighbor_mask_0_slot + level] =
            reference_pixels[level];
    }
    allocation_bytes[voting_neighbor_depth_slot] =
        largest_neighbor_pixels * sizeof(float);
    allocation_bytes[voting_neighbor_radius_slot] =
        largest_neighbor_pixels * sizeof(float);
    for (std::size_t index = voting_direct_counter_0_slot;
         index < allocation_bytes.size(); ++index)
        allocation_bytes[index] = sizeof(std::int32_t);

    RecoveredCudaVotingWorkspaceHandles workspace_handles;
    bool workspace_acquired = false;
    if (!acquire_recovered_cuda_voting_workspace(
            allocation_bytes, workspace_handles, workspace_acquired, error)) {
        cuModuleUnload(radius_module);
        cuModuleUnload(neighbor_module);
        cuModuleUnload(final_module);
        return false;
    }
    std::array<void*, RecoveredCudaVotingWorkspace::fixed_allocation_count>
        device = workspace_handles.fixed;
    auto release = [&]() {
        if (workspace_acquired) {
            release_recovered_cuda_voting_workspace_lease();
        } else {
            for (void* pointer : device) cudaFree(pointer);
        }
        cuModuleUnload(radius_module);
        cuModuleUnload(neighbor_module);
        cuModuleUnload(final_module);
    };
    if (!workspace_acquired) {
        for (std::size_t index = 0;
             runtime_status == cudaSuccess && index < device.size(); ++index)
            runtime_status = cudaMalloc(&device[index], allocation_bytes[index]);
    }
    if (runtime_status != cudaSuccess) {
        error = std::string("allocating recovered depth-voting chain buffers failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    output = {};
    output.direct_counters = input.direct_counters;
    output.occlusion_counters = input.occlusion_counters;
    output.final_counters = input.final_counters;
    auto copy_to_device = [&](void* destination, const void* source,
                              std::size_t bytes) {
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                destination, source, bytes, cudaMemcpyHostToDevice);
    };
    auto copy_depth_to_device = [&](void* destination,
                                    std::span<const float> source) {
        if (runtime_status != cudaSuccess) return;
        void* cached_device_pointer = nullptr;
        const std::size_t bytes = source.size() * sizeof(float);
        {
            std::lock_guard lock(recovered_cuda_module_session_mutex);
            auto& session = recovered_cuda_module_session;
            if (session.active) {
                const auto cached =
                    session.voting_depth_cache.find(source.data());
                if (cached != session.voting_depth_cache.end() &&
                    cached->second.bytes == bytes) {
                    cached_device_pointer = cached->second.device_pointer;
                    ++session.stats.voting_depth_cache_hits;
                    session.stats.voting_depth_cache_hit_bytes += bytes;
                }
            }
        }
        runtime_status = cached_device_pointer == nullptr
            ? cudaMemcpy(destination, source.data(), bytes,
                         cudaMemcpyHostToDevice)
            : cudaMemcpy(destination, cached_device_pointer, bytes,
                         cudaMemcpyDeviceToDevice);
    };
    for (std::size_t level = 0; level < level_count; ++level) {
        output.votes[level] = input.reference[level].initial_votes;
        if (output.votes[level].empty())
            output.votes[level].assign(reference_pixels[level], 0);
        output.depth_before_components[level].resize(reference_pixels[level]);
        const std::span<const float> reference_depth =
            input.reference[level].depth_view.empty()
                ? std::span<const float>(input.reference[level].depth)
                : input.reference[level].depth_view;
        copy_depth_to_device(
            device[voting_reference_depth_0_slot + level], reference_depth);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemset(
                device[voting_reference_radius_0_slot + level], 0,
                allocation_bytes[voting_reference_radius_0_slot + level]);
        copy_to_device(device[voting_reference_votes_0_slot + level],
                       output.votes[level].data(),
                       allocation_bytes[voting_reference_votes_0_slot + level]);
    }
    for (std::size_t index = 0; index < input.direct_counters.size(); ++index)
        copy_to_device(device[voting_direct_counter_0_slot + index],
                       &input.direct_counters[index], sizeof(std::int32_t));
    for (std::size_t index = 0; index < input.occlusion_counters.size(); ++index)
        copy_to_device(device[voting_occlusion_counter_0_slot + index],
                       &input.occlusion_counters[index], sizeof(std::int32_t));
    for (std::size_t index = 0; index < input.final_counters.size(); ++index)
        copy_to_device(device[voting_final_counter_0_slot + index],
                       &input.final_counters[index], sizeof(std::int32_t));
    if (runtime_status != cudaSuccess) {
        error = std::string("uploading recovered depth-voting chain state failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    auto launch_and_synchronize = [&](CUfunction function,
                                      std::size_t work_items,
                                      void** arguments,
                                      std::string_view label) {
        const unsigned int blocks =
            static_cast<unsigned int>((work_items + 127U) / 128U);
        CUresult status = cuLaunchKernel(
            function, blocks, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr);
        if (status == CUDA_SUCCESS) status = cuCtxSynchronize();
        if (status != CUDA_SUCCESS) {
            const char* message = nullptr;
            cuGetErrorString(status, &message);
            error = std::string(label) + " failed: " +
                    (message ? message : "unknown CUDA driver error");
            return false;
        }
        return true;
    };
    auto launch_radius = [&](const DepthVotingRadiusLaunch& launch,
                             void* depth, void* radius,
                             std::size_t default_items) {
        std::uint32_t level_offset = launch.level_offset;
        std::uint32_t kernel_offset = launch.kernel_offset;
        void* arguments[] = {
            &depth, &radius, &level_offset,
            const_cast<DepthVotingCalibrationCu*>(&launch.calibration),
            const_cast<DepthVotingMatrix4x4f*>(&launch.transform),
            &kernel_offset,
        };
        const std::size_t work_items = launch.global_work_items == 0U
            ? default_items : launch.global_work_items;
        return launch_and_synchronize(
            radius_function, work_items, arguments,
            "recovered depth-voting radius launch");
    };

    for (std::size_t level = 0; level < level_count; ++level) {
        if (!launch_radius(
                input.reference[level].radius,
                device[voting_reference_depth_0_slot + level],
                device[voting_reference_radius_0_slot + level],
                reference_pixels[level])) {
            release();
            return false;
        }
    }

    for (std::size_t neighbor_index = 0;
         neighbor_index < input.neighbors.size(); ++neighbor_index) {
        const auto& neighbor = input.neighbors[neighbor_index];
        std::size_t neighbor_elements = neighbor.depth_levels.size();
        if (neighbor.depth_levels.empty()) {
            neighbor_elements = 0U;
            for (std::size_t level = 0; level < level_count; ++level) {
                const std::span<const float> depth =
                    neighbor.depth_level_views[level];
                copy_depth_to_device(
                    static_cast<std::uint8_t*>(
                        device[voting_neighbor_depth_slot]) +
                        neighbor.radius[level].level_offset * sizeof(float),
                    depth);
                neighbor_elements += depth.size();
            }
        } else {
            copy_depth_to_device(
                device[voting_neighbor_depth_slot], neighbor.depth_levels);
        }
        const std::size_t neighbor_bytes =
            neighbor_elements * sizeof(float);
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemset(
                device[voting_neighbor_radius_slot], 0, neighbor_bytes);
        for (std::size_t level = 0; level < level_count; ++level) {
            const std::span<const std::uint8_t> mask =
                neighbor.inlier_mask_views[level].empty()
                    ? std::span<const std::uint8_t>(
                          neighbor.inlier_masks[level])
                    : neighbor.inlier_mask_views[level];
            copy_to_device(device[voting_neighbor_mask_0_slot + level],
                           mask.data(),
                           reference_pixels[level]);
        }
        if (runtime_status != cudaSuccess) {
            error = "uploading recovered depth-voting neighbor state failed: " +
                    std::string(cudaGetErrorString(runtime_status));
            release();
            return false;
        }

        for (std::size_t level = 0; level < level_count; ++level) {
            const std::uint32_t level_width =
                calibration_u32(neighbor.radius[level].calibration, 84U);
            const std::uint32_t level_height =
                calibration_u32(neighbor.radius[level].calibration, 88U);
            const std::size_t items =
                static_cast<std::size_t>(level_width) * level_height;
            if (!launch_radius(neighbor.radius[level],
                               device[voting_neighbor_depth_slot],
                               device[voting_neighbor_radius_slot], items)) {
                error = "neighbor " + std::to_string(neighbor_index) +
                        " " + error;
                release();
                return false;
            }
        }

        for (std::size_t reference_level = 0;
             reference_level < level_count; ++reference_level) {
            const auto& direct = neighbor.direct[reference_level];
            std::uint32_t reference_level_u32 =
                static_cast<std::uint32_t>(reference_level);
            std::uint32_t neighbor_levels =
                static_cast<std::uint32_t>(level_count);
            std::uint32_t kernel_offset = direct.kernel_offset;
            void* device_votes =
                device[voting_reference_votes_0_slot + reference_level];
            void* device_reference_depth =
                device[voting_reference_depth_0_slot + reference_level];
            void* device_reference_radius =
                device[voting_reference_radius_0_slot + reference_level];
            void* device_neighbor_mask =
                device[voting_neighbor_mask_0_slot + reference_level];
            void* device_neighbor_depth = device[voting_neighbor_depth_slot];
            void* device_neighbor_radius = device[voting_neighbor_radius_slot];
            void* direct_arguments[] = {
                &device_votes,
                &device_reference_depth,
                &device_reference_radius,
                &device_neighbor_mask,
                &device_neighbor_depth,
                &device_neighbor_radius,
                &reference_level_u32,
                const_cast<DepthVotingCalibrationCu*>(
                    &direct.reference_calibration),
                const_cast<DepthVotingMatrix4x4f*>(
                    &direct.reference_to_neighbor),
                &neighbor_levels,
                const_cast<DepthVotingCalibrationCu*>(
                    &direct.neighbor_calibration),
                &device[voting_direct_counter_0_slot + 0U],
                &device[voting_direct_counter_0_slot + 1U],
                &device[voting_direct_counter_0_slot + 2U],
                &device[voting_direct_counter_0_slot + 3U],
                &device[voting_direct_counter_0_slot + 4U],
                &device[voting_direct_counter_0_slot + 5U],
                &device[voting_direct_counter_0_slot + 6U],
                &kernel_offset,
            };
            const std::size_t direct_items = direct.global_work_items == 0U
                ? reference_pixels[reference_level]
                : direct.global_work_items;
            if (!launch_and_synchronize(
                    direct_function, direct_items, direct_arguments,
                    "recovered direct depth-voting launch")) {
                error = "neighbor " + std::to_string(neighbor_index) +
                        " direct level " +
                        std::to_string(reference_level) + ": " + error;
                release();
                return false;
            }

            for (std::size_t neighbor_level = 0;
                 neighbor_level < level_count; ++neighbor_level) {
                const auto& occlusion =
                    neighbor.occlusion[reference_level][neighbor_level];
                std::uint32_t neighbor_level_u32 =
                    static_cast<std::uint32_t>(neighbor_level);
                std::uint32_t occlusion_kernel_offset =
                    occlusion.kernel_offset;
                void* occlusion_arguments[] = {
                    &device_votes,
                    &device_reference_depth,
                    &device_reference_radius,
                    &device_neighbor_mask,
                    &neighbor_level_u32,
                    &device_neighbor_depth,
                    &device_neighbor_radius,
                    &reference_level_u32,
                    const_cast<DepthVotingCalibrationCu*>(
                        &occlusion.reference_calibration),
                    const_cast<DepthVotingMatrix4x4f*>(
                        &occlusion.neighbor_to_reference),
                    const_cast<DepthVotingCalibrationCu*>(
                        &occlusion.neighbor_calibration),
                    &device[voting_occlusion_counter_0_slot + 0U],
                    &device[voting_occlusion_counter_0_slot + 1U],
                    &occlusion_kernel_offset,
                };
                const std::uint32_t base_width = calibration_u32(
                    occlusion.neighbor_calibration, 84U);
                const std::uint32_t base_height = calibration_u32(
                    occlusion.neighbor_calibration, 88U);
                const std::size_t default_items = level_pixels(
                    base_width, base_height,
                    static_cast<std::uint32_t>(neighbor_level));
                const std::size_t occlusion_items =
                    occlusion.global_work_items == 0U
                        ? default_items : occlusion.global_work_items;
                if (!launch_and_synchronize(
                        occlusion_function, occlusion_items,
                        occlusion_arguments,
                        "recovered occlusion depth-voting launch")) {
                    error = "neighbor " + std::to_string(neighbor_index) +
                            " occlusion levels " +
                            std::to_string(reference_level) + "/" +
                            std::to_string(neighbor_level) + ": " + error;
                    release();
                    return false;
                }
            }
        }
    }

    for (std::size_t level = 0; level < level_count; ++level) {
        std::uint32_t width = input.reference[level].width;
        std::uint32_t height = input.reference[level].height;
        std::uint32_t offset = 0U;
        void* device_depth = device[voting_reference_depth_0_slot + level];
        void* device_votes = device[voting_reference_votes_0_slot + level];
        void* arguments[] = {
            &device_depth, &device_votes, &width, &height,
            &device[voting_final_counter_0_slot + 0U],
            &device[voting_final_counter_0_slot + 1U],
            &device[voting_final_counter_0_slot + 2U],
            &device[voting_final_counter_0_slot + 3U],
            &offset,
        };
        if (!launch_and_synchronize(
                final_function, reference_pixels[level], arguments,
                "recovered final depth-voting launch")) {
            error = "final voting level " + std::to_string(level) +
                    ": " + error;
            release();
            return false;
        }
    }

    auto copy_from_device = [&](void* destination, const void* source,
                                std::size_t bytes) {
        if (runtime_status == cudaSuccess)
            runtime_status = cudaMemcpy(
                destination, source, bytes, cudaMemcpyDeviceToHost);
    };
    for (std::size_t level = 0; level < level_count; ++level) {
        copy_from_device(output.depth_before_components[level].data(),
                         device[voting_reference_depth_0_slot + level],
                         allocation_bytes[voting_reference_depth_0_slot + level]);
        copy_from_device(output.votes[level].data(),
                         device[voting_reference_votes_0_slot + level],
                         allocation_bytes[voting_reference_votes_0_slot + level]);
    }
    for (std::size_t index = 0; index < output.direct_counters.size(); ++index)
        copy_from_device(&output.direct_counters[index],
                         device[voting_direct_counter_0_slot + index],
                         sizeof(std::int32_t));
    for (std::size_t index = 0; index < output.occlusion_counters.size(); ++index)
        copy_from_device(&output.occlusion_counters[index],
                         device[voting_occlusion_counter_0_slot + index],
                         sizeof(std::int32_t));
    for (std::size_t index = 0; index < output.final_counters.size(); ++index)
        copy_from_device(&output.final_counters[index],
                         device[voting_final_counter_0_slot + index],
                         sizeof(std::int32_t));
    if (runtime_status != cudaSuccess) {
        error = std::string("copying recovered depth-voting chain output failed: ") +
                cudaGetErrorString(runtime_status);
        release();
        return false;
    }

    release();
    error.clear();
    return true;
}

}  // namespace metmodel
