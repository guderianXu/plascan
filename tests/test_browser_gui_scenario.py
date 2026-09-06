import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "dev" / "browser_gui_scenario.py"
SPEC = importlib.util.spec_from_file_location("plascan_browser_gui_scenario", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCENARIO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENARIO)


def snapshot(modal="", visible=True):
    return {
        "application": {"modal_window": modal},
        "project": {"open": False, "dirty": False, "image_count": 0},
        "tasks": [],
        "recent_error": "",
        "windows": [
            {
                "object_name": "mainWindow",
                "children": [],
                "actions": [
                    {
                        "object_name": "actionAbout",
                        "text": "About",
                        "visible": visible,
                        "enabled": True,
                    }
                ],
            }
        ],
    }


class BrowserGuiScenarioTest(unittest.TestCase):
    def test_condition_matches_named_action_and_modal(self):
        matched, _ = SCENARIO.condition_matches(
            snapshot("PlaScan About"),
            {
                "object_name": "actionAbout",
                "exists": True,
                "visible": True,
                "modal_present": True,
                "modal_title_contains": "PlaScan",
            },
        )
        self.assertTrue(matched)

    def test_unknown_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            SCENARIO.condition_matches(snapshot(), {"arbitrary_code": "no"})

    def test_condition_matches_project_image_count(self):
        current = snapshot()
        current["project"] = {"open": True, "dirty": False, "image_count": 123}
        matched, _ = SCENARIO.condition_matches(
            current, {"project_open": True, "image_count": 123}
        )
        self.assertTrue(matched)

    def test_condition_supports_nested_path_expressions(self):
        current = snapshot()
        current["project"] = {"open": True, "dirty": False, "image_count": 123}
        matched, _ = SCENARIO.condition_matches(
            current,
            {
                "all": [
                    {"path": "project.image_count", "gte": 100},
                    {"not": {"project_dirty": True}},
                ]
            },
        )
        self.assertTrue(matched)

    def test_successful_scenario_records_each_step(self):
        current = snapshot()

        def bridge_call(method, parameters, _timeout):
            nonlocal current
            if method == "snapshot":
                return current
            if method == "interact":
                self.assertEqual("actionAbout", parameters["object_name"])
                current = snapshot("PlaScan About")
                return {"accepted": True}
            if method == "close_dialog":
                current = snapshot()
                return {"closed": True}
            if method == "screenshot":
                return {"available": False}
            self.fail(f"unexpected method {method}")

        scenario = {
            "name": "about",
            "steps": [
                {"wait": {"object_name": "actionAbout"}},
                {"action": {"object_name": "actionAbout", "operation": "activate"}},
                {"assert": {"modal_present": True}},
                {"close_dialog": {}},
                {"assert": {"modal_present": False}},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = SCENARIO.run_scenario(
                scenario, {"run_directory": temporary_directory}, bridge_call,
                Path(temporary_directory) / "diagnostics"
            )
            self.assertTrue(result["ok"])
            self.assertEqual(5, len(result["completed"]))

    def test_failed_assertion_writes_diagnostics(self):
        def bridge_call(method, _parameters, _timeout):
            if method == "snapshot":
                return snapshot()
            if method == "screenshot":
                return {"available": False}
            return {}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "failure"
            result = SCENARIO.run_scenario(
                {"name": "failure", "steps": [{"assert": {"modal_present": True}}]},
                {
                    "token": "secret",
                    "url": "http://127.0.0.1:6080/?token=secret",
                    "run_directory": temporary_directory,
                },
                bridge_call,
                output,
            )
            self.assertFalse(result["ok"])
            self.assertTrue((output / "result.json").is_file())
            runtime = (output / "runtime.json").read_text(encoding="utf-8")
            self.assertNotIn("secret", runtime)
            self.assertNotIn("?token=", runtime)


if __name__ == "__main__":
    unittest.main()
