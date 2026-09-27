#include <placamera/linescan_camera.h>
#include <placamera/linescan_numeric_state.h>

#include <array>
#include <iostream>
#include <optional>
#include <span>
#include <utility>
#include <vector>

int main()
{
    using namespace placamera;

    LineScanOptics optics;
    optics.focalLengthMillimeters = 10.0;
    MetashapeCalibration calibration;
    calibration.f = 1000.0;
    calibration.cx = 500.0;
    calibration.cy = 0.0;
    calibration.k1 = 0.001;
    calibration.principalPointDecomposition = PrincipalPointDecomposition{500.0, 0.0, 0.0, 0.0};
    optics.completeCalibration = calibration;
    const auto definition =
        LineScanDefinition::create(CameraDefinitionId("line-definition"), FrameId("body-fixed"), optics);

    const RotationMatrix identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    TrajectoryKnotConstraints free_knot;
    free_knot.positionFixed = false;
    free_knot.rotationFixed = false;
    free_knot.positionSigmaMeters = Vector3{{2.0, 2.0, 5.0}};
    free_knot.rotationSigmaRadians = Vector3{{0.01, 0.01, 0.02}};

    auto trajectory = LineScanTrajectory::create(
        {TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0), {0.0, -1.0, 0.0}, identity},
         TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.5), {0.0, 0.0, 0.0}, identity, free_knot},
         TrajectorySample{TimeReference::create(TimeScale::Tdb, 1.0), {0.0, 1.0, 0.0}, identity}});

    const LineScanModel model = LineScanModel::create(CameraInstanceId("line-instance"),
                                                      ImageId("line-image"),
                                                      definition,
                                                      ImageSize{1000, 11},
                                                      std::move(trajectory),
                                                      LineTiming{0.5, 0.0, 0.1, TimeScale::Tdb, {}},
                                                      {},
                                                      std::nullopt,
                                                      LineScanTimeOffsetPrior{0.0, 0.02});

    LineScanOptimizationSelection selection;
    selection.positionSecondDifferenceWeight = 4.0;
    selection.rotationSecondDifferenceWeight = 9.0;
    selection.detector.focalLength = true;
    selection.calibration.f = true;
    selection.calibration.cx = true;
    selection.calibration.k1 = true;

    auto state_result = LineScanNumericState::fromModel(model, selection);
    if (!state_result)
    {
        std::cerr << state_result.message() << '\n';
        return 1;
    }
    LineScanNumericState state = state_result.takeValue();

    std::vector<double> delta(state.optimizationLayout().parameterCount(), 0.0);
    if (const auto applied = state.applyOptimizationDelta(std::span<const double>(delta)); !applied)
    {
        std::cerr << applied.message() << '\n';
        return 2;
    }

    std::vector<double> regularization(state.regularizationResidualCount());
    if (const auto written = state.writeRegularizationResiduals(std::span<double>(regularization)); !written)
    {
        std::cerr << written.message() << '\n';
        return 3;
    }

    const auto projection = state.projectAtLine(GroundCoordinate{FrameId("body-fixed"), {1.0, 0.0, 10.0}}, 5.5);
    if (!projection)
    {
        std::cerr << projection.message() << '\n';
        return 4;
    }

    const auto refined =
        state.toModel(CameraInstanceId("refined-line-instance"), CameraDefinitionId("refined-line-definition"));
    if (!refined)
    {
        std::cerr << refined.message() << '\n';
        return 5;
    }

    std::cout << "parameters=" << state.optimizationLayout().parameterCount()
              << ", regularization=" << regularization.size()
              << ", sample=" << projection.value().projection.image.sample << '\n';
    return 0;
}
