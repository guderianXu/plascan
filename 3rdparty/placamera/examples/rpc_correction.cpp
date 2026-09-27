#include <placamera/rpc_camera.h>

#include <iostream>

int main()
{
    using namespace placamera;

    RpcParameters parameters;
    parameters.lineScale = 1.0;
    parameters.sampleScale = 1.0;
    parameters.latitudeScale = 1.0;
    parameters.longitudeScale = 1.0;
    parameters.heightScale = 1.0;
    parameters.lineNumerator[2] = 1.0;
    parameters.sampleNumerator[1] = 1.0;
    parameters.lineDenominator[0] = 1.0;
    parameters.sampleDenominator[0] = 1.0;

    const auto definition =
        RpcDefinition::create(CameraDefinitionId("rpc-definition"), FrameId("EPSG:4978"), parameters);

    RpcGroundCorrection correction;
    correction.sampleOffsetPixels = 0.25;
    correction.lineOffsetPixels = -0.5;
    correction.sampleLongitudePixelsPerDegree = 0.01;
    correction.lineLatitudePixelsPerDegree = -0.02;

    const RpcModel model = RpcModel::createWithCorrection(CameraInstanceId("rpc-instance"),
                                                          ImageId("rpc-image"),
                                                          definition,
                                                          ImageSize{2048, 2048},
                                                          RpcCorrection::groundCoordinates(correction));
    if (model.correctionDomain() != RpcCorrectionDomain::GroundCoordinates || !model.groundCorrection() ||
        model.normalizedImageCorrection())
    {
        return 1;
    }

    const auto projection = model.groundToImageGeodetic({0.5, 0.25, 0.0});
    if (!projection)
    {
        std::cerr << projection.message() << '\n';
        return 2;
    }

    std::cout << "sample=" << projection.value().image.sample << ", line=" << projection.value().image.line << '\n';
    return 0;
}
