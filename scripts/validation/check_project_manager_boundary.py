#!/usr/bin/env python3
"""Audit the ProjectManager GUI boundary during the staged rewrite.

The audit deliberately works on source text instead of a compiler database so it
is deterministic in a clean checkout and can run before the GUI target builds.
The default mode is an informational migration report (exit status 0); ``--strict``
turns every finding into a failure.  Tasks 2--7 may annotate an explicitly listed
ProjectManager line with ``PROJECT_MANAGER_MIGRATION_SHIM`` while the replacement
service is being landed.  The allow-list is intentionally narrow and removed by
the final migration task.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PM_HEADER = Path("src/gui/project/manager/ProjectManager.h")
PM_SOURCE = Path("src/gui/project/manager/ProjectManager.cpp")
TRANSITIONAL_FILES = {PM_HEADER.as_posix(), PM_SOURCE.as_posix()}
SHIM_TOKEN = "PROJECT_MANAGER_MIGRATION_SHIM"

FORBIDDEN_HEADER_INCLUDES = ("QFuture",)
FORBIDDEN_SOURCE_INCLUDES = (
    "QMessageBox",
    "QInputDialog",
    "QFileDialog",
    "QtConcurrent",
    "ProjectSparseReconstructionManager.h",
    "ProjectPointCloudWorkflowController.h",
    "ProjectModelManager.h",
    "ProjectTerrainProductsManager.h",
    "ProjectCameraSetupManager.h",
    "ProjectLifecycleController.h",
    "ProjectMaskWorkflowController.h",
    "ProjectUiCommands.h",
)
FORBIDDEN_HEADER_MEMBERS = (
    "_projectSessionGeneration",
    "_automaticModelDepthPreparationActive",
    "_imageImportActive",
    "_portableExportActive",
    "_resourceCleanupRunning",
    "_resourceCleanupFuture",
    "_atCancelFlag",
    "_pendingBaCameraMeta",
    "_pendingBaBeforeCameraMeta",
    "_pendingBaResult",
    "_hasPendingBaPreview",
    "_sessionFacade",
    "_sparseReconstructionManager",
    "_pointCloudWorkflowController",
    "_modelManager",
    "_terrainProductsManager",
    "_cameraSetupManager",
    "_uiCommands",
    "_lifecycleController",
    "_maskWorkflowController",
    "_pendingAutomaticModelSettings",
    "_resourceCleanupShutdownFinalize",
)
FORBIDDEN_SOURCE_CONSTRUCTORS = (
    "new ProjectSparseReconstructionManager",
    "new ProjectPointCloudWorkflowController",
    "new ProjectModelManager",
    "new ProjectTerrainProductsManager",
    "new ProjectCameraSetupManager",
    "new ProjectLifecycleController",
    "new ProjectMaskWorkflowController",
    "new ProjectUiCommands",
)
REQUIRED_SERVICE_HEADERS = (
    "src/gui/project/services/ProjectServiceContainer.h",
    "src/gui/project/services/ProjectSession.h",
    "src/gui/project/services/ProjectResourceService.h",
    "src/gui/project/services/ProjectResourceCleanupCoordinator.h",
    "src/gui/project/services/ProjectUiMessageAdapter.h",
    "src/gui/project/tasks/ProjectTaskOrchestrator.h",
)
OWNER_RE = re.compile(
    r"\b(?:const\s+)?ProjectManager\s*(?:const\s*)?(?:\*|&)"
    r"|\b(?:QPointer|std::(?:unique_ptr|shared_ptr|weak_ptr))\s*<\s*(?:const\s+)?ProjectManager\s*>"
)

TASK_IMPLEMENTATION_FILES = (
    "src/gui/project/manager/ProjectSparseReconstructionManager.h",
    "src/gui/project/manager/ProjectSparseReconstructionManager.cpp",
    "src/gui/project/manager/ProjectMaskWorkflowController.h",
    "src/gui/project/manager/ProjectMaskWorkflowController.cpp",
    "src/gui/project/manager/ProjectPointCloudWorkflowController.h",
    "src/gui/project/manager/ProjectPointCloudWorkflowController.cpp",
    "src/gui/project/manager/ProjectModelManager.h",
    "src/gui/project/manager/ProjectModelManager.cpp",
    "src/gui/project/manager/ProjectTerrainProductsManager.h",
    "src/gui/project/manager/ProjectTerrainProductsManager.cpp",
    "src/gui/project/manager/ProjectTerrainRpcProducts.cpp",
    "src/gui/project/manager/ProjectCameraSetupManager.h",
    "src/gui/project/manager/ProjectCameraSetupManager.cpp",
    "src/gui/project/tasks/ProjectTaskOrchestrator.h",
    "src/gui/project/tasks/ProjectTaskOrchestrator.cpp",
)
TASK_DIALOG_DEPENDENCIES = ("QMessageBox", "QFileDialog", "QInputDialog", "ProjectOpenGuard")
PROJECT_MANAGER_CALL_METHODS = frozenset(
    {
        "appendimagematchresults",
        "cancelat",
        "coreprojectmeta",
        "currentmeta",
        "currentprojectpath",
        "currentsessioncontext",
        "getallimages",
        "getimageidsforimages",
        "getlastuseddir",
        "iscurrentsession",
        "ownsatcancelflag",
        "refreshreconstructionqualityreport",
        "savelastuseddir",
        "setatcancelflag",
    }
)
PROJECT_DATA_MUTATION_PREFIXES = (
    "add",
    "append",
    "clear",
    "create",
    "delete",
    "mark",
    "remove",
    "replace",
    "save",
    "schedule",
    "set",
    "update",
    "upsert",
    "write",
)
RAW_STRING_START_RE = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
INCLUDE_OPERAND_RE = re.compile(
    r'^[ \t\f\v]*#[ \t\f\v]*include\b[ \t\f\v]*'
    r'(?P<operand>"(?:\\.|[^"\\\r\n])*"|<[^>\r\n]*>|[A-Za-z_]\w*)'
)
CPP_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|->|::|[().;,{}\[\]*&]")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    detail: str


@dataclass(frozen=True)
class SplicedCppSource:
    text: str
    physical_lines: tuple[str, ...]
    physical_line_by_offset: tuple[int, ...]


@dataclass(frozen=True)
class CppLexicalViews:
    spliced: SplicedCppSource
    code: str
    preprocessor: str


@dataclass(frozen=True)
class IncludeDirective:
    line: int
    operand: str


@dataclass(frozen=True)
class CppToken:
    text: str
    offset: int
    line: int


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _allowed(root: Path, path: Path, line: str) -> bool:
    return _rel(root, path) in TRANSITIONAL_FILES and SHIM_TOKEN in line


def _splice_cpp_source(source: str) -> SplicedCppSource:
    """Apply translation-phase-2 line splicing and retain physical source locations."""
    compacted: list[str] = []
    physical_line_by_offset: list[int] = []
    physical_line = 1
    index = 0
    while index < len(source):
        if source[index] == "\\":
            if index + 1 < len(source) and source[index + 1] == "\n":
                index += 2
                physical_line += 1
                continue
            if index + 2 < len(source) and source[index + 1:index + 3] == "\r\n":
                index += 3
                physical_line += 1
                continue
        compacted.append(source[index])
        physical_line_by_offset.append(physical_line)
        if source[index] == "\n":
            physical_line += 1
        index += 1
    return SplicedCppSource(
        "".join(compacted), tuple(source.splitlines()), tuple(physical_line_by_offset)
    )


def _is_cpp_digit_separator(source: str, index: int) -> bool:
    """Return whether an apostrophe separates digits in the current numeric token."""
    if source[index] != "'" or index == 0 or index + 1 >= len(source):
        return False

    token_start = index
    while token_start > 0:
        preceding = source[token_start - 1]
        if not (preceding.isascii() and (preceding.isalnum() or preceding in "_.'")):
            break
        token_start -= 1

    numeric_prefix = source[token_start:index].replace("'", "")
    if not numeric_prefix:
        return False
    if numeric_prefix.startswith("."):
        if len(numeric_prefix) == 1 or numeric_prefix[1] not in "0123456789":
            return False
    elif numeric_prefix[0] not in "0123456789":
        return False

    normalized_prefix = numeric_prefix.lower()
    if normalized_prefix.startswith("0x"):
        separator_digits = "0123456789abcdefABCDEF"
    elif normalized_prefix.startswith("0b"):
        separator_digits = "01"
    else:
        separator_digits = "0123456789"
    return source[index - 1] in separator_digits and source[index + 1] in separator_digits


def _cpp_lexical_views(source: str) -> CppLexicalViews:
    """Build spliced code and preprocessor views with stable physical-line mapping."""
    spliced = _splice_cpp_source(source)
    code = list(spliced.text)
    preprocessor = list(spliced.text)

    def hide(view: list[str], start: int, length: int) -> None:
        for offset in range(start, min(start + length, len(view))):
            if view[offset] not in {"\r", "\n"}:
                view[offset] = " "

    def hide_code(start: int, length: int) -> None:
        hide(code, start, length)

    def hide_both(start: int, length: int) -> None:
        hide(code, start, length)
        hide(preprocessor, start, length)

    index = 0
    state = "code"
    raw_terminator = ""
    while index < len(spliced.text):
        if state == "line_comment":
            if spliced.text[index] in "\r\n":
                state = "code"
                index += 1
            else:
                hide_both(index, 1)
                index += 1
            continue
        if state == "block_comment":
            if spliced.text.startswith("*/", index):
                hide_both(index, 2)
                index += 2
                state = "code"
            else:
                hide_both(index, 1)
                index += 1
            continue
        if state == "raw_string":
            if spliced.text.startswith(raw_terminator, index):
                hide_code(index, len(raw_terminator))
                index += len(raw_terminator)
                state = "code"
                raw_terminator = ""
            else:
                hide_code(index, 1)
                index += 1
            continue
        if state in {"string", "character"}:
            delimiter = '"' if state == "string" else "'"
            hide_code(index, 1)
            if spliced.text[index] == "\\" and index + 1 < len(spliced.text):
                hide_code(index + 1, 1)
                index += 2
            elif spliced.text[index] == delimiter:
                index += 1
                state = "code"
            elif spliced.text[index] in "\r\n":
                index += 1
                state = "code"
            else:
                index += 1
            continue

        if spliced.text.startswith("//", index):
            hide_both(index, 2)
            index += 2
            state = "line_comment"
            continue
        if spliced.text.startswith("/*", index):
            hide_both(index, 2)
            index += 2
            state = "block_comment"
            continue
        raw_match = RAW_STRING_START_RE.match(spliced.text, index)
        if raw_match:
            hide_code(index, raw_match.end() - index)
            index = raw_match.end()
            raw_terminator = f'){raw_match.group(1)}"'
            state = "raw_string"
            continue
        if spliced.text[index] == '"':
            hide_code(index, 1)
            index += 1
            state = "string"
            continue
        if spliced.text[index] == "'" and _is_cpp_digit_separator(spliced.text, index):
            index += 1
            continue
        if spliced.text[index] == "'":
            hide_code(index, 1)
            index += 1
            state = "character"
            continue
        index += 1

    return CppLexicalViews(spliced, "".join(code), "".join(preprocessor))


def _physical_line(spliced: SplicedCppSource, offset: int) -> int:
    if not spliced.physical_line_by_offset:
        return 1
    bounded_offset = min(max(offset, 0), len(spliced.physical_line_by_offset) - 1)
    return spliced.physical_line_by_offset[bounded_offset]


def _line_detail(spliced: SplicedCppSource, line: int) -> str:
    if 1 <= line <= len(spliced.physical_lines):
        return spliced.physical_lines[line - 1].strip()
    return ""


def _preprocessor_analysis(views: CppLexicalViews) -> tuple[str, tuple[IncludeDirective, ...]]:
    """Mask directives from code rules and extract active include operands."""
    code = list(views.code)
    includes: list[IncludeDirective] = []
    start = 0
    while start <= len(views.preprocessor):
        newline = views.preprocessor.find("\n", start)
        end = len(views.preprocessor) if newline < 0 else newline
        logical_line = views.preprocessor[start:end]
        directive = re.match(r"[ \t\f\v]*#", logical_line)
        if directive:
            for offset in range(start, end):
                if code[offset] not in {"\r", "\n"}:
                    code[offset] = " "
            include_match = INCLUDE_OPERAND_RE.match(logical_line)
            if include_match:
                operand = include_match.group("operand")
                if operand.startswith(('"', "<")):
                    operand = operand[1:-1]
                hash_offset = start + directive.end() - 1
                includes.append(IncludeDirective(_physical_line(views.spliced, hash_offset), operand))
        if newline < 0:
            break
        start = newline + 1
    return "".join(code), tuple(includes)


def _tokenize_cpp(code: str, spliced: SplicedCppSource) -> tuple[CppToken, ...]:
    return tuple(
        CppToken(match.group(), match.start(), _physical_line(spliced, match.start()))
        for match in CPP_TOKEN_RE.finditer(code)
    )


def _receiver_identifier(tokens: tuple[CppToken, ...], operator_index: int) -> CppToken | None:
    index = operator_index - 1
    if index < 0:
        return None
    if tokens[index].text.isidentifier():
        return tokens[index]
    if tokens[index].text != ")":
        return None

    depth = 1
    close_index = index
    index -= 1
    while index >= 0:
        if tokens[index].text == ")":
            depth += 1
        elif tokens[index].text == "(":
            depth -= 1
            if depth == 0:
                inner = tokens[index + 1:close_index]
                while len(inner) >= 3 and inner[0].text == "(" and inner[-1].text == ")":
                    inner = inner[1:-1]
                if len(inner) == 1 and inner[0].text.isidentifier():
                    return inner[0]
                return None
        index -= 1
    return None


def _looks_like_project_manager_receiver(identifier: str) -> bool:
    lowered = identifier.lower()
    return (
        lowered in {"manager", "mgr", "owner"}
        or lowered.endswith(("manager", "mgr", "owner"))
        or lowered.startswith("pm")
    )


def _add_line_finding(findings: list[Finding], root: Path, path: Path, number: int, rule: str, detail: str) -> None:
    line = path.read_text(encoding="utf-8").splitlines()[number - 1]
    if not _allowed(root, path, line):
        findings.append(Finding(_rel(root, path), number, rule, detail))


def _scan_project_manager(root: Path, findings: list[Finding]) -> None:
    header = root / PM_HEADER
    source = root / PM_SOURCE
    for path, include_needles, member_needles, constructor_needles in (
        (header, FORBIDDEN_HEADER_INCLUDES, FORBIDDEN_HEADER_MEMBERS, ()),
        (source, FORBIDDEN_SOURCE_INCLUDES, (), FORBIDDEN_SOURCE_CONSTRUCTORS),
    ):
        if not path.is_file():
            findings.append(
                Finding(_rel(root, path), 0, "missing_project_manager_file", "required source file is missing")
            )
            continue
        file_text = path.read_text(encoding="utf-8")
        lines = file_text.splitlines()
        views = _cpp_lexical_views(file_text)
        _, includes = _preprocessor_analysis(views)
        for include in includes:
            for needle in include_needles:
                if needle in include.operand:
                    _add_line_finding(findings, root, path, include.line, "forbidden_include", needle)
        for number, line in enumerate(lines, start=1):
            for needle in member_needles:
                if needle in line:
                    _add_line_finding(findings, root, path, number, "forbidden_member", needle)
            for needle in constructor_needles:
                if needle in line:
                    _add_line_finding(findings, root, path, number, "forbidden_constructor", needle)

    source_text = source.read_text(encoding="utf-8") if source.is_file() else ""
    if source.is_file() and "ProjectServiceContainer" not in source_text:
        findings.append(
            Finding(_rel(root, source), 0, "missing_service_container_dependency", "ProjectServiceContainer")
        )

    for relative in REQUIRED_SERVICE_HEADERS:
        if not (root / relative).is_file():
            findings.append(Finding(relative, 0, "missing_required_service_header", relative))


def _scan_owner_dependencies(root: Path, findings: list[Finding]) -> None:
    gui_root = root / "src/gui"
    if not gui_root.is_dir():
        return
    for path in sorted(gui_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".h", ".hh", ".hpp", ".cpp", ".cc", ".cxx"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, start=1):
            if OWNER_RE.search(line):
                if _allowed(root, path, line):
                    continue
                findings.append(
                    Finding(_rel(root, path), number, "project_manager_owner_dependency", line.strip())
                )


def _scan_task_implementations(root: Path, findings: list[Finding]) -> None:
    for relative in TASK_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            findings.append(Finding(relative, 0, "missing_task_implementation_file", relative))
            continue

        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(Finding(relative, 0, "unreadable_task_implementation_file", relative))
            continue

        views = _cpp_lexical_views(source)
        code, includes = _preprocessor_analysis(views)
        tokens = _tokenize_cpp(code, views.spliced)
        file_findings: dict[tuple[int, str], Finding] = {}

        def add(line: int, rule: str, detail: str) -> None:
            file_findings.setdefault((line, rule), Finding(relative, line, rule, detail))

        for include in includes:
            if "ProjectManager" in include.operand:
                add(include.line,
                    "project_manager_include",
                    _line_detail(views.spliced, include.line))
            for dependency in TASK_DIALOG_DEPENDENCIES:
                if dependency in include.operand:
                    add(include.line, "direct_generic_dialog_dependency", dependency)

        for index, token in enumerate(tokens):
            if token.text == "ProjectManager":
                add(token.line,
                    "project_manager_owner_dependency",
                    _line_detail(views.spliced, token.line))
                if index + 1 < len(tokens) and tokens[index + 1].text == "::":
                    add(token.line,
                        "project_manager_owner_call",
                        _line_detail(views.spliced, token.line))
            elif token.text == "ProjectData":
                add(token.line,
                    "retained_project_data_dependency",
                    _line_detail(views.spliced, token.line))
            if token.text in TASK_DIALOG_DEPENDENCIES:
                add(token.line, "direct_generic_dialog_dependency", token.text)

            if index == 0 or index + 1 >= len(tokens) or token.text not in {"->", "."}:
                continue
            receiver = _receiver_identifier(tokens, index)
            method = tokens[index + 1]
            has_call = index + 2 < len(tokens) and tokens[index + 2].text == "("
            if not receiver or not method.text.isidentifier() or not has_call:
                continue

            receiver_lower = receiver.text.lower()
            method_lower = method.text.lower()
            if (_looks_like_project_manager_receiver(receiver.text) and
                    method_lower in PROJECT_MANAGER_CALL_METHODS):
                add(receiver.line,
                    "project_manager_owner_call",
                    _line_detail(views.spliced, receiver.line))
            if "session" in receiver_lower and method_lower == "data":
                add(receiver.line,
                    "project_session_data_escape",
                    _line_detail(views.spliced, receiver.line))
            if receiver.text in {"_projectData", "projectData", "project_data"} and method_lower.startswith(
                PROJECT_DATA_MUTATION_PREFIXES
            ):
                add(receiver.line,
                    "direct_project_data_mutation",
                    _line_detail(views.spliced, receiver.line))

        findings.extend(file_findings.values())


def scan(root: Path, scope: str = "global") -> list[Finding]:
    findings: list[Finding] = []
    if scope == "task-implementations":
        _scan_task_implementations(root, findings)
    else:
        _scan_project_manager(root, findings)
        _scan_owner_dependencies(root, findings)
    return sorted(findings, key=lambda item: (item.path, item.line, item.rule, item.detail))


def _run_task_implementation_self_test() -> bool:
    cases = (
        ("#include \"ProjectManager.h\"\n", ((1, "project_manager_include"),)),
        ("# include \"ProjectManager.h\"\n", ((1, "project_manager_include"),)),
        ('#include "ProjectManager.h" // "QMessageBox"\n', ((1, "project_manager_include"),)),
        ('#include "SafeHeader.h" // ProjectManager "QMessageBox" ProjectOpenGuard\n', ()),
        ('# include "ProjectOpenGuard.h" // ProjectManager "QMessageBox"\n',
         ((1, "direct_generic_dialog_dependency"),)),
        ("# include <QMessageBox> // ProjectManager\n", ((1, "direct_generic_dialog_dependency"),)),
        ("class ProjectManager;\nProjectManager* _owner = nullptr;\n_owner->cancelAt();\n",
         ((1, "project_manager_owner_dependency"),
          (2, "project_manager_owner_dependency"),
          (3, "project_manager_owner_call"))),
        ("class ProjectData;\nProjectData* _projectData = nullptr;\n",
         ((1, "retained_project_data_dependency"), (2, "retained_project_data_dependency"))),
        ("#include <QMessageBox>\n#include \"ProjectOpenGuard.h\"\n",
         ((1, "direct_generic_dialog_dependency"), (2, "direct_generic_dialog_dependency"))),
        ("_projectData->updateMetadata(meta, true);\n", ((1, "direct_project_data_mutation"),)),
        ("_session->data()->updateMetadata(meta, true);\n", ((1, "project_session_data_escape"),)),
        ("auto* project = _session->data();\nproject->updateMetadata(meta, true);\n",
         ((1, "project_session_data_escape"),)),
        ("auto* project = _sessionPtr->data();\n", ((1, "project_session_data_escape"),)),
        ("auto* project = session_ptr.data();\n", ((1, "project_session_data_escape"),)),
        ("auto* manager = ownerProvider();\nmanager->cancelAt();\n", ((2, "project_manager_owner_call"),)),
        ("auto* mgr = ownerProvider();\nmgr->appendImageMatchResults(records);\n",
         ((2, "project_manager_owner_call"),)),
        ("ProjectManager manager;\n", ((1, "project_manager_owner_dependency"),)),
        ("ProjectManager::currentMeta();\n",
         ((1, "project_manager_owner_call"), (1, "project_manager_owner_dependency"))),
        ("scheduler->cancelAt();\n", ()),
        ("// ProjectManager manager; mgr->cancelAt();\n", ()),
        ("/* ProjectManager manager;\nmanager->cancelAt(); */\n", ()),
        ('const char* text = "ProjectManager mgr->cancelAt()";\n', ()),
        ('const char* text = "mgr->cancelAt(); \\\nProjectManager";\n', ()),
        ('const char* text = R"tag(ProjectManager\nmgr->cancelAt())tag";\n', ()),
        ("const char marker = 'P';\n", ()),
        ("auto n = 1'000; mgr->appendImageMatchResults(records);\n",
         ((1, "project_manager_owner_call"),)),
        ("auto h = 0xDEAD'BEEF; _owner->cancelAt();\n", ((1, "project_manager_owner_call"),)),
        ("auto marker = 'ProjectManager mgr->cancelAt()';\n", ()),
        ("auto marker = L'ProjectManager mgr->cancelAt()';\n", ()),
        ("auto marker = u'ProjectManager mgr->cancelAt()';\n", ()),
        ("auto marker = U'ProjectManager mgr->cancelAt()';\n", ()),
        ("auto marker = u8'ProjectManager mgr->cancelAt()';\n", ()),
        ('# include /* comment */ "ProjectManager.h"\n', ((1, "project_manager_include"),)),
        ('# include \\\n  /* comment */ "ProjectManager.h"\n', ((1, "project_manager_include"),)),
        ("# /* comment */ include /* comment */ <QMessageBox>\n",
         ((1, "direct_generic_dialog_dependency"),)),
        ('# include "Safe.h" // ProjectManager QMessageBox ProjectOpenGuard\n', ()),
        ("_session\n  ->data()\n", ((1, "project_session_data_escape"),)),
        ("(_session)\n  -> /* comment */ data()\n", ((1, "project_session_data_escape"),)),
        ("_session\\\n  ->data()\n", ((1, "project_session_data_escape"),)),
        ("_session\\\r\n  ->data()\r\n", ((1, "project_session_data_escape"),)),
        ("mgr\n  ->appendImageMatchResults(records);\n", ((1, "project_manager_owner_call"),)),
        ("mgr\\\n  ->appendImageMatchResults(records);\n", ((1, "project_manager_owner_call"),)),
        ("scheduler\n  ->cancelAt();\n", ()),
        ("_projectData\n  ->updateMetadata(meta, true);\n", ((1, "direct_project_data_mutation"),)),
        ("ProjectMan\\\nager* owner = nullptr;\n", ((1, "project_manager_owner_dependency"),)),
        ("ProjectDa\\\nta* data = nullptr;\n", ((1, "retained_project_data_dependency"),)),
        ("// mgr->cancelAt(); \\\nmgr->appendImageMatchResults(records);\n", ()),
    )
    with tempfile.TemporaryDirectory(prefix="plascan-project-manager-audit-") as directory:
        root = Path(directory)
        for relative in TASK_IMPLEMENTATION_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

        target = root / TASK_IMPLEMENTATION_FILES[0]
        for source, expected_occurrences in cases:
            target.write_text(source, encoding="utf-8")
            actual_occurrences = tuple(
                (finding.line, finding.rule) for finding in scan(root, "task-implementations")
            )
            if actual_occurrences != expected_occurrences:
                print(
                    "ProjectManager task-implementation scanner self-test failed: "
                    f"expected {expected_occurrences}, got {actual_occurrences}",
                    file=sys.stderr,
                )
                return False
        target.write_text("", encoding="utf-8")
        if scan(root, "task-implementations"):
            print("ProjectManager task-implementation scanner self-test failed: clean fixture was rejected",
                  file=sys.stderr)
            return False
    print("ProjectManager task-implementation scanner self-test passed")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--scope", choices=("global", "task-implementations"), default="global")
    parser.add_argument("--strict", action="store_true", help="return 1 when any boundary finding remains")
    parser.add_argument("--self-test-task-implementations", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test_task_implementations:
        return 0 if _run_task_implementation_self_test() else 1

    root = args.source_root.resolve()
    if not root.is_dir():
        print(f"ProjectManager boundary: source root does not exist: {root}", file=sys.stderr)
        return 2

    findings = scan(root, args.scope)
    if findings:
        mode = "strict" if args.strict else "transitional"
        scope = "" if args.scope == "global" else f", scope={args.scope}"
        print(f"ProjectManager boundary audit ({mode}{scope}): {len(findings)} finding(s)")
        for finding in findings:
            location = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"- {location} [{finding.rule}] {finding.detail}")
    else:
        print("ProjectManager boundary audit passed: no findings")
    return 1 if args.strict and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
