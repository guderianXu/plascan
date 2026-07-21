#include "CliTestSupport.h"

TEST(ModelQualityCliGTest, UsesSharedQualityEvaluator)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/quality/CMakeLists.txt"));
    const QString source = readSourceFile(QStringLiteral("src/cli/quality/cli_model_quality.cpp"));

    expectContainsAll(cmake, {
        "model_quality_cli",
        "cli_model_quality.cpp",
        "qc",
    });
    expectContainsAll(source, {
        "--mesh",
        "--image-camera-list",
        "--mvs-workspace",
        "--scene-type",
        "--validation-split",
        "--reference-cloud",
        "--reference-camera-list",
        "--output-dir",
        "ModelImageQualityEvaluator",
    });
}
