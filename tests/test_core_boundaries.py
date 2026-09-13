from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validation" / "check_core_boundaries.py"
BASELINE_PATH = ROOT / "scripts" / "validation" / "core_boundary_baseline.json"

SPEC = importlib.util.spec_from_file_location("check_core_boundaries", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import boundary scanner: {SCRIPT_PATH}")
BOUNDARIES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOUNDARIES
SPEC.loader.exec_module(BOUNDARIES)


class CoreBoundaryTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        (self.root / "src" / "core" / "sample").mkdir(parents=True)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def write(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_committed_repository_matches_exact_baseline(self):
        result = BOUNDARIES.scan_repository(ROOT)
        baseline = BOUNDARIES.load_baseline(BASELINE_PATH)
        drift = BOUNDARIES.compare_with_baseline(result, baseline)

        self.assertEqual((), drift.forbidden)
        self.assertEqual((), drift.unexpected)
        self.assertEqual((), drift.missing)

    def test_forbidden_gui_boundaries_are_zero_tolerance(self):
        self.write(
            "src/core/sample/CMakeLists.txt",
            """
add_library(sample STATIC sample.cpp)
target_link_libraries(sample PRIVATE Qt6::Widgets)
target_include_directories(sample PUBLIC "${CMAKE_SOURCE_DIR}/src/gui")
""",
        )
        self.write(
            "src/core/sample/sample.h",
            """
#include <QtWidgets/QWidget>
#include <QDialog>
#include "../../gui/GuiService.h"

class SampleDialog : public QDialog
{
    void report(QMessageBox *message_box);
};
""",
        )

        result = BOUNDARIES.scan_repository(self.root)
        rules = {finding.rule for finding in result.forbidden}

        self.assertEqual(
            {
                "forbidden_qt_widgets",
                "forbidden_gui_type",
                "forbidden_gui_include",
                "forbidden_gui_include_root",
            },
            rules,
        )

    def test_directory_wide_gui_include_root_is_forbidden(self):
        self.write(
            "src/core/sample/CMakeLists.txt",
            """
add_library(sample STATIC sample.cpp)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../../gui)
target_include_directories(sample PRIVATE ${PLASCAN_GUI_INCLUDE_DIRECTORIES})
""",
        )

        result = BOUNDARIES.scan_repository(self.root)

        self.assertEqual(2, len(result.forbidden))
        self.assertEqual(
            {"forbidden_gui_include_root"},
            {finding.rule for finding in result.forbidden},
        )
        self.assertTrue(
            any(
                "PLASCAN_GUI_INCLUDE_DIRECTORIES" in finding.evidence
                for finding in result.forbidden
            )
        )

    def test_moving_existing_debt_cannot_cancel_drift(self):
        original = self.write(
            "src/core/sample/original.h",
            "#include <QString>\nstruct Value { QString text; };\n",
        )
        baseline = BOUNDARIES.scan_repository(self.root).frozen

        original.unlink()
        self.write(
            "src/core/sample/moved.h",
            "#include <QString>\nstruct Value { QString text; };\n",
        )
        drift = BOUNDARIES.compare_with_baseline(
            BOUNDARIES.scan_repository(self.root), baseline
        )

        self.assertTrue(drift.unexpected)
        self.assertTrue(drift.missing)
        self.assertEqual(
            {"src/core/sample/moved.h"},
            {finding.path for finding in drift.unexpected},
        )
        self.assertEqual(
            {"src/core/sample/original.h"},
            {finding.path for finding in drift.missing},
        )

    def test_qt_link_visibility_change_is_drift(self):
        cmake = self.write(
            "src/core/sample/CMakeLists.txt",
            """
add_library(sample STATIC sample.cpp)
target_link_libraries(sample PUBLIC Qt6::Core)
""",
        )
        baseline = BOUNDARIES.scan_repository(self.root).frozen

        cmake.write_text(
            """
add_library(sample STATIC sample.cpp)
target_link_libraries(sample PRIVATE Qt6::Core)
""",
            encoding="utf-8",
        )
        drift = BOUNDARIES.compare_with_baseline(
            BOUNDARIES.scan_repository(self.root), baseline
        )

        self.assertEqual(1, len(drift.unexpected))
        self.assertEqual(1, len(drift.missing))
        self.assertIn("|PRIVATE|", drift.unexpected[0].evidence)
        self.assertIn("|PUBLIC|", drift.missing[0].evidence)

    def test_comments_and_literals_do_not_create_findings(self):
        self.write(
            "src/core/sample/CMakeLists.txt",
            """
add_library(sample STATIC sample.cpp)
# target_link_libraries(sample PRIVATE Qt6::Widgets)
#[[
target_include_directories(sample PRIVATE ${CMAKE_SOURCE_DIR}/src/gui)
]]
""",
        )
        self.write(
            "src/core/sample/sample.h",
            r'''
// #include <QDialog>
/* QMessageBox *dialog; QtConcurrent::run(); display_name */
struct Sample
{
    const char *description = "QFuture QObject ../gui/X.h";
};
''',
        )

        result = BOUNDARIES.scan_repository(self.root)

        self.assertEqual((), result.forbidden)
        self.assertEqual((), result.frozen)

    def test_scan_and_baseline_serialization_are_deterministic(self):
        self.write(
            "src/core/zeta/zeta.h",
            "#include <QJsonObject>\nstruct Zeta { QJsonObject value; };\n",
        )
        self.write(
            "src/core/alpha/CMakeLists.txt",
            """
add_library(alpha STATIC alpha.cpp)
target_link_libraries(alpha PRIVATE Qt6::Core PUBLIC Qt6::Gui)
""",
        )

        first = BOUNDARIES.scan_repository(self.root)
        second = BOUNDARIES.scan_repository(self.root)

        self.assertEqual(first, second)
        self.assertEqual(
            BOUNDARIES.format_baseline(first.frozen),
            BOUNDARIES.format_baseline(second.frozen),
        )
        self.assertNotIn(str(self.root), BOUNDARIES.format_baseline(first.frozen))

    def test_occurrence_count_change_requires_baseline_update(self):
        header = self.write(
            "src/core/sample/sample.h",
            "#include <QString>\nstruct Sample { QString first; };\n",
        )
        baseline = BOUNDARIES.scan_repository(self.root).frozen
        header.write_text(
            "#include <QString>\nstruct Sample { QString first; QString second; };\n",
            encoding="utf-8",
        )

        drift = BOUNDARIES.compare_with_baseline(
            BOUNDARIES.scan_repository(self.root), baseline
        )

        symbol_addition = next(
            finding
            for finding in drift.unexpected
            if finding.rule == "public_header_qt_symbol" and finding.evidence == "QString"
        )
        self.assertEqual(2, symbol_addition.count)
        self.assertTrue(
            any(
                finding.rule == "public_header_qt_symbol"
                and finding.evidence == "QString"
                and finding.count == 1
                for finding in drift.missing
            )
        )

    def test_runtime_and_presentation_debt_is_unexpected_without_baseline(self):
        self.write(
            "src/core/sample/worker.h",
            """
#include <QFuture>
#include <QObject>

class Worker : public QObject
{
    Q_OBJECT
    QFuture<void> _future;
};
""",
        )
        self.write(
            "src/core/sample/worker.cpp",
            """
#include <QtConcurrent/QtConcurrent>
void run(QObject *owner)
{
    QtConcurrent::run([] {});
    const char *operation_key = "operation_display_name";
    const char *display_key = "display_name";
}
""",
        )

        drift = BOUNDARIES.compare_with_baseline(
            BOUNDARIES.scan_repository(self.root), ()
        )
        runtime_evidence = {
            finding.evidence
            for finding in drift.unexpected
            if finding.rule == "runtime_qt_dependency"
        }
        display_evidence = {
            finding.evidence
            for finding in drift.unexpected
            if finding.rule == "display_name"
        }

        self.assertTrue(
            {
                "include:QFuture",
                "include:QObject",
                "include:QtConcurrent/QtConcurrent",
                "symbol:QFuture",
                "symbol:QObject",
                "symbol:Q_OBJECT",
                "symbol:QtConcurrent",
            }.issubset(runtime_evidence)
        )
        self.assertEqual({"operation_display_name", "display_name"}, display_evidence)


if __name__ == "__main__":
    unittest.main()
