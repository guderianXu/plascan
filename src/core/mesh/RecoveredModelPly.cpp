#include "RecoveredModelBuilder.h"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>

#include <QSaveFile>

namespace xjw::mesh
{
    bool writeRecoveredModelPly(const RecoveredModelResult& model, const QString& path, std::string* error)
    {
        // Reference run_recovered_ooc_model.cpp::export_ply schema and decimal precision.
        // QSaveFile only supplies Unicode-safe, transactional publication.
        const auto fail = [&](const std::string& reason)
        {
            if (error)
                *error = reason;
            return false;
        };
        if (model.mesh.empty() || model.precisePositions.size() != model.mesh.vertices.size())
        {
            return fail("Incomplete recovered double-precision mesh");
        }
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly))
            return fail(file.errorString().toStdString());
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << "ply\nformat ascii 1.0\n"
               << "element vertex " << model.mesh.vertices.size() << '\n'
               << "property double x\nproperty double y\nproperty double z\n"
               << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
               << "property float confidence\n"
               << "element face " << model.mesh.faces.size() << '\n'
               << "property list uchar uint vertex_indices\nend_header\n"
               << std::setprecision(17);
        const auto flush = [&]()
        {
            const auto bytes = output.str();
            const bool ok =
                file.write(bytes.data(), static_cast<qint64>(bytes.size())) == static_cast<qint64>(bytes.size());
            output.str({});
            return ok;
        };
        for (std::size_t i = 0; i < model.mesh.vertices.size(); ++i)
        {
            const auto& p = model.precisePositions[i];
            const auto& v = model.mesh.vertices[i];
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || !std::isfinite(v.confidence))
                return fail("Non-finite recovered PLY vertex");
            output << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << static_cast<unsigned>(v.r) << ' '
                   << static_cast<unsigned>(v.g) << ' ' << static_cast<unsigned>(v.b) << ' ' << std::setprecision(9)
                   << v.confidence << '\n'
                   << std::setprecision(17);
            if (i % 4096 == 4095 && !flush())
                return fail(file.errorString().toStdString());
        }
        for (const auto& face : model.mesh.faces)
        {
            for (const int index : face.v)
                if (index < 0 || static_cast<std::size_t>(index) >= model.mesh.vertices.size())
                    return fail("Invalid recovered PLY face index");
            output << "3 " << face.v[0] << ' ' << face.v[1] << ' ' << face.v[2] << '\n';
            if (output.tellp() >= 256 * 1024 && !flush())
                return fail(file.errorString().toStdString());
        }
        if (!flush() || !file.commit())
            return fail(file.errorString().toStdString());
        return true;
    }
} // namespace xjw::mesh
