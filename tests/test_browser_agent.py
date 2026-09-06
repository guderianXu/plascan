import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = load_module("plascan_browser_agent_protocol", "scripts/dev/browser_agent_protocol.py")
AGENT = load_module("plascan_browser_agent", "scripts/dev/browser_agent.py")


def sample_snapshot(timestamp="2026-08-30T00:00:00Z"):
    return {
        "timestamp": timestamp,
        "application": {
            "name": "PlaScan",
            "active_window": "PlaScan - demo",
            "modal_window": "",
        },
        "project": {
            "open": True,
            "path": "/tmp/demo.plascan",
            "dirty": False,
            "image_count": 3,
            "artifacts": [{"key": "mesh", "path": "/tmp/mesh.ply", "exists": True}],
        },
        "tasks": [],
        "recent_error": "",
        "logs": [{"level": "info", "message": "ready"}],
        "windows": [
            {
                "class": "QMainWindow",
                "object_name": "MainWindow",
                "window_title": "PlaScan - demo",
                "visible": True,
                "enabled": True,
                "children": [
                    {
                        "class": "QLineEdit",
                        "object_name": "resourcePath",
                        "text": "old",
                        "visible": True,
                        "enabled": True,
                        "children": [],
                    },
                    {
                        "class": "QCheckBox",
                        "object_name": "reuseMatches",
                        "text": "Reuse",
                        "checkable": True,
                        "checked": False,
                        "visible": True,
                        "enabled": True,
                        "children": [],
                    },
                ],
                "actions": [
                    {
                        "class": "QAction",
                        "object_name": "actionAbout",
                        "text": "About",
                        "visible": True,
                        "enabled": True,
                    },
                    {
                        "class": "QAction",
                        "object_name": "actionAddPhoto",
                        "text": "Add photos",
                        "visible": True,
                        "enabled": True,
                    },
                ],
            }
        ],
    }


class BrowserAgentProtocolTest(unittest.TestCase):
    def test_summary_omits_heavy_arrays_and_reports_counts(self):
        result = PROTOCOL.summary(sample_snapshot())
        self.assertNotIn("artifacts", result["project"])
        self.assertNotIn("windows", result)
        self.assertEqual({"controls": 5, "logs": 1, "artifacts": 1}, result["counts"])

    def test_revision_ignores_timestamp_but_tracks_control_state(self):
        first = sample_snapshot("first")
        second = sample_snapshot("second")
        self.assertEqual(PROTOCOL.revision_for(first), PROTOCOL.revision_for(second))
        second["windows"][0]["children"][1]["checked"] = True
        self.assertNotEqual(PROTOCOL.revision_for(first), PROTOCOL.revision_for(second))

    def test_query_and_pagination_are_bounded(self):
        controls = PROTOCOL.filter_items(PROTOCOL.named_objects(sample_snapshot()), "about")
        page = PROTOCOL.paginate(controls, 0, 10)
        self.assertEqual(1, page["total"])
        self.assertEqual("actionAbout", page["items"][0]["object_name"])
        with self.assertRaises(ValueError):
            PROTOCOL.paginate(controls, 0, 201)

    def test_nested_and_path_conditions_share_scenario_semantics(self):
        condition = {
            "all": [
                {"path": "project.image_count", "gte": 3},
                {"any": [{"modal_present": True}, {"project_dirty": False}]},
                {"not": {"recent_error_contains": "fatal"}},
            ]
        }
        matched, _ = PROTOCOL.condition_matches(sample_snapshot(), condition)
        self.assertTrue(matched)

    def test_wait_timeout_is_machine_classifiable(self):
        with self.assertRaises(TimeoutError):
            PROTOCOL.wait_for_condition(
                lambda: sample_snapshot(), {"path": "project.image_count", "equals": 99}, 0.0
            )

    def test_diff_reports_field_paths(self):
        before = {"project": {"dirty": False}}
        after = {"project": {"dirty": True}}
        self.assertEqual("project.dirty", PROTOCOL.diff_states(before, after)[0]["path"])

    def test_read_only_fixture_blocks_known_write_actions(self):
        target = AGENT.find_target(sample_snapshot(), "actionAddPhoto")
        with self.assertRaises(AGENT.AgentProtocolError) as context:
            AGENT.validate_operation(
                {"project_read_only": True}, target, "activate", allow_project_write=True
            )
        self.assertEqual("project_write_blocked", context.exception.code)

    def test_writable_project_still_requires_explicit_write_confirmation(self):
        target = AGENT.find_target(sample_snapshot(), "actionAddPhoto")
        with self.assertRaises(AGENT.AgentProtocolError) as context:
            AGENT.validate_operation({}, target, "activate", allow_project_write=False)
        self.assertEqual("project_write_confirmation_required", context.exception.code)
        AGENT.validate_operation({}, target, "activate", allow_project_write=True)

    def test_form_validates_then_rolls_back_on_failure(self):
        calls = []

        def bridge(method, parameters):
            calls.append((method, dict(parameters)))
            if parameters["object_name"] == "reuseMatches":
                raise RuntimeError("simulated failure")
            return {}

        with self.assertRaises(RuntimeError):
            AGENT.apply_form(
                {},
                sample_snapshot(),
                bridge,
                {"resourcePath": "new", "reuseMatches": True},
                allow_project_write=False,
            )
        self.assertEqual("new", calls[0][1]["value"])
        self.assertEqual("old", calls[-1][1]["value"])

    def test_direct_project_form_requires_confirmation(self):
        with self.assertRaises(AGENT.AgentProtocolError) as context:
            AGENT.validate_form_scope({"project_is_copy": False}, False)
        self.assertEqual("project_write_confirmation_required", context.exception.code)
        AGENT.validate_form_scope({"project_is_copy": True}, False)
        AGENT.validate_form_scope({"project_is_copy": False}, True)

    def test_history_is_bounded_and_delta_ready(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = {"run_directory": temporary_directory}
            for index in range(12):
                current = sample_snapshot()
                current["project"]["image_count"] = index
                AGENT.record_history(state, current)
            history = AGENT.read_history(state)
            self.assertEqual(AGENT.MAXIMUM_HISTORY, len(history))
            self.assertIn("state", history[-1])

    def test_capabilities_advertise_agent_safety_boundary(self):
        result = AGENT.capabilities(ROOT)
        self.assertFalse(result["safety"]["true_copy_on_write_sandbox"])
        self.assertTrue(result["safety"]["direct_project_form_requires_flag"])
        self.assertIn("south_building", result["fixtures"])
        self.assertIn("diagnose", result["commands"])
        self.assertIn("task-command", result["commands"])

    def test_task_command_parser_exposes_revision_guarded_queue_actions(self):
        parser = AGENT.build_parser(ROOT)
        args = parser.parse_args(
            [
                "task-command",
                "--action",
                "move_before",
                "--run-id",
                "depth-run-2",
                "--reference-run-id",
                "depth-run-1",
                "--revision",
                "9",
            ]
        )
        self.assertEqual("move_before", args.action)
        self.assertEqual("depth-run-2", args.run_id)
        self.assertEqual(9, args.revision)


if __name__ == "__main__":
    unittest.main()
