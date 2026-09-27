#include <placamera/linescan_numeric_state.h>

#include <atomic>
#include <cstdlib>
#include <new>

namespace
{
    std::atomic<std::size_t> allocationCount{0};
    thread_local bool countAllocations = false;
} // namespace

void* operator new(std::size_t size)
{
    if (countAllocations)
    {
        allocationCount.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* memory = std::malloc(size))
    {
        return memory;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size)
{
    return ::operator new(size);
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete[](void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept
{
    std::free(memory);
}

int main()
{
    using namespace placamera;

    LineScanOptics optics;
    optics.focalLengthMillimeters = 10.0;
    optics.samplePitchMillimeters = 0.01;
    optics.principalSample = 5.0;
    const auto definition = LineScanDefinition::create(CameraDefinitionId("definition"), FrameId("body-fixed"), optics);
    TrajectoryKnotConstraints free;
    free.positionFixed = false;
    free.rotationFixed = false;
    const RotationMatrix identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    const auto trajectory = LineScanTrajectory::create(
        {TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0), {0.0, -1.0, 0.0}, identity, free},
         TrajectorySample{TimeReference::create(TimeScale::Tdb, 1.0), {0.0, 1.0, 0.0}, identity, free}});
    const auto model = LineScanModel::create(CameraInstanceId("instance"),
                                             ImageId("image"),
                                             definition,
                                             ImageSize{1000, 11},
                                             trajectory,
                                             LineTiming{0.5, 0.0, 0.1, TimeScale::Tdb, {}});
    LineScanOptics complete_optics;
    complete_optics.focalLengthMillimeters = 35.0;
    MetashapeCalibration calibration;
    calibration.f = 980.0;
    calibration.cx = 512.25;
    calibration.cy = 384.75;
    calibration.b1 = 3.0;
    calibration.b2 = -0.4;
    calibration.k1 = 0.01;
    calibration.k2 = -0.001;
    calibration.k3 = 0.0002;
    calibration.k4 = -0.00003;
    calibration.p1 = 0.0005;
    calibration.p2 = -0.0004;
    calibration.p3 = 0.00007;
    calibration.p4 = -0.000005;
    calibration.principalPointDecomposition = PrincipalPointDecomposition{400.0, 300.0, 112.25, 84.75};
    complete_optics.completeCalibration = calibration;
    const auto complete_definition =
        LineScanDefinition::create(CameraDefinitionId("complete-definition"), FrameId("body-fixed"), complete_optics);
    const auto complete_model = LineScanModel::create(CameraInstanceId("complete-instance"),
                                                      ImageId("complete-image"),
                                                      complete_definition,
                                                      ImageSize{1200, 11},
                                                      trajectory,
                                                      LineTiming{0.5, 0.0, 0.1, TimeScale::Tdb, {}});
    auto state_result = LineScanNumericState::fromModel(model);
    auto complete_state_result = LineScanNumericState::fromModel(complete_model);
    if (!state_result || !complete_state_result)
    {
        return 1;
    }
    auto state = state_result.takeValue();
    auto complete_state = complete_state_result.takeValue();
    const GroundCoordinate ground{FrameId("body-fixed"), {2.0, 0.0, 10.0}};
    if (!state.projectAtLine(ground, 5.5) || !complete_state.projectAtLine(ground, 5.5))
    {
        return 2;
    }

    allocationCount.store(0, std::memory_order_relaxed);
    countAllocations = true;
    bool all_valid = true;
    for (int index = 0; index < 1000; ++index)
    {
        const auto projection = state.projectAtLine(ground, 5.5);
        const auto complete_projection = complete_state.projectAtLine(ground, 5.5);
        all_valid = all_valid && projection.ok() && projection.value().projection.image.sample > 0.0 &&
                    complete_projection.ok() && complete_projection.value().projection.image.sample > 0.0;
    }
    countAllocations = false;
    return all_valid && allocationCount.load(std::memory_order_relaxed) == 0 ? 0 : 3;
}
