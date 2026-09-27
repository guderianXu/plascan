#include <placamera/reference/CameraReferenceUncertainty.h>
#include <placamera/reference/MetashapeCameraReferenceAdapter.h>

#include <array>
#include <iostream>

int main()
{
    using namespace placamera;
    using namespace placamera::reference;

    MetashapeCameraReference source;
    source.positionMeters = Vector3{{1.0, 2.0, 3.0}};
    source.positionCovarianceMetersSquared = std::array<double, 9>{{4.0, 0.5, 0.0, 0.5, 9.0, 0.0, 0.0, 0.0, 16.0}};
    source.yawPitchRollDegrees = Vector3{{10.0, 2.0, -3.0}};
    source.yawPitchRollCovarianceDegreesSquared = std::array<double, 9>{{1.0, 0.1, 0.0, 0.1, 4.0, 0.2, 0.0, 0.2, 9.0}};

    const auto observation =
        adaptMetashapeCameraReference(ImageId("image"), ReferenceSourceId("metashape"), FrameId("reference"), source);
    if (!observation || !observation.value().covariance)
    {
        std::cerr << observation.message() << '\n';
        return 1;
    }

    const RotationMatrix target_from_source{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    const auto propagated = propagatePoseCovarianceSimilarity(*observation.value().covariance, 2.0, target_from_source);
    if (!propagated)
    {
        std::cerr << propagated.message() << '\n';
        return 2;
    }

    std::cout << "position-x variance=" << propagated.value().matrixValues()[0]
              << ", rotation-x variance=" << propagated.value().matrixValues()[21] << '\n';
    return 0;
}
