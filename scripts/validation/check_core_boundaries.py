#!/usr/bin/env python3
"""Reject new GUI coupling and unreviewed Qt dependency drift in ``src/core``.

The committed baseline records existing debt by rule, repository-relative path,
evidence, and occurrence count.  Both additions and removals are reported so an
intentional cleanup remains visible in review and requires an explicit baseline
update.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


SCHEMA_VERSION = 1
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hxx"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}

FROZEN_RULES = (
    "core_library_qt_link",
    "public_header_qt_include",
    "public_header_qt_symbol",
    "runtime_qt_dependency",
    "display_name",
)

FORBIDDEN_RULES = (
    "forbidden_qt_widgets",
    "forbidden_gui_type",
    "forbidden_gui_include",
    "forbidden_gui_include_root",
)

QT_TARGET_RE = re.compile(r"\b(Qt(?:[0-9]+|\$\{[^}]+\})?::[A-Za-z][A-Za-z0-9_]*)\b")
QT_LINK_VARIABLE_RE = re.compile(r"\$\{[^}]*qt[^}]*\}", re.IGNORECASE)
CPP_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*(?:<(?P<angle>[^>\n]+)>|"(?P<quote>[^"\n]+)")',
    re.MULTILINE,
)
QT_SYMBOL_RE = re.compile(
    r"\b(?:Q[A-Z_][A-Za-z0-9_]*|q(?:int|uint)(?:8|16|32|64)|qreal|qsizetype|"
    r"qptrdiff|qintptr|quintptr|qlonglong|qulonglong)\b|\bQt(?=::)"
)
RUNTIME_QT_SYMBOL_RE = re.compile(r"\b(?:QObject|Q_OBJECT|QFuture|QtConcurrent)\b")
DISPLAY_NAME_RE = re.compile(r"\b(?:operation_display_name|display_name)\b")
FORBIDDEN_GUI_TYPE_RE = re.compile(r"\b(?:QDialog|QMessageBox)\b")
COMMAND_START_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
RAW_STRING_START_RE = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')

# Direct Qt includes do not encode their owning module.  These headers are a
# conservative set of Qt Widgets entry points that must never appear in core.
QT_WIDGET_HEADERS = {
    "QAbstractButton",
    "QAbstractItemDelegate",
    "QAbstractItemView",
    "QApplication",
    "QBoxLayout",
    "QButtonGroup",
    "QCheckBox",
    "QColorDialog",
    "QComboBox",
    "QCommandLinkButton",
    "QCompleter",
    "QDataWidgetMapper",
    "QDateEdit",
    "QDateTimeEdit",
    "QDial",
    "QDialog",
    "QDialogButtonBox",
    "QDockWidget",
    "QDoubleSpinBox",
    "QErrorMessage",
    "QFileDialog",
    "QFocusFrame",
    "QFontComboBox",
    "QFontDialog",
    "QFormLayout",
    "QFrame",
    "QGraphicsItem",
    "QGraphicsScene",
    "QGraphicsView",
    "QGridLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QInputDialog",
    "QItemDelegate",
    "QLabel",
    "QLayout",
    "QLineEdit",
    "QListView",
    "QListWidget",
    "QMainWindow",
    "QMdiArea",
    "QMenu",
    "QMenuBar",
    "QMessageBox",
    "QPlainTextEdit",
    "QProgressBar",
    "QProgressDialog",
    "QPushButton",
    "QRadioButton",
    "QScrollArea",
    "QScrollBar",
    "QSlider",
    "QSpinBox",
    "QSplitter",
    "QStackedLayout",
    "QStackedWidget",
    "QStatusBar",
    "QStyle",
    "QStyledItemDelegate",
    "QTabBar",
    "QTableView",
    "QTableWidget",
    "QTabWidget",
    "QTextBrowser",
    "QTextEdit",
    "QToolBar",
    "QToolButton",
    "QTreeView",
    "QTreeWidget",
    "QUndoView",
    "QVBoxLayout",
    "QWidget",
    "QWizard",
}


class BoundaryScanError(RuntimeError):
    """Raised when the repository or baseline cannot be scanned safely."""


@dataclass(frozen=True, order=True)
class Finding:
    rule: str
    path: str
    evidence: str
    count: int = 1


@dataclass(frozen=True)
class ScanResult:
    frozen: tuple[Finding, ...]
    forbidden: tuple[Finding, ...]


@dataclass(frozen=True)
class BaselineDrift:
    forbidden: tuple[Finding, ...]
    unexpected: tuple[Finding, ...]
    missing: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return not self.forbidden and not self.unexpected and not self.missing


class _FindingCounter:
    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str, str]] = Counter()

    def add(self, rule: str, path: str, evidence: str, count: int = 1) -> None:
        if count > 0:
            self._counts[(rule, path, evidence)] += count

    def findings(self) -> tuple[Finding, ...]:
        return tuple(
            Finding(rule=rule, path=path, evidence=evidence, count=count)
            for (rule, path, evidence), count in sorted(self._counts.items())
        )


def _repository_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_test_source(path: Path, core_root: Path) -> bool:
    relative = path.relative_to(core_root)
    lower_parts = {part.lower() for part in relative.parts[:-1]}
    stem = path.stem.lower()
    return bool({"test", "tests"} & lower_parts) or stem.startswith("test_") or stem.endswith("_test")


def _blank(text: str) -> str:
    return "".join("\n" if character == "\n" else " " for character in text)


def _sanitize_cpp(text: str, *, remove_literals: bool) -> str:
    """Remove comments and optionally literals while preserving line structure."""

    output: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            if end < 0:
                end = length
            output.append(_blank(text[index:end]))
            index = end
            continue

        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            output.append(_blank(text[index:end]))
            index = end
            continue

        raw_match = RAW_STRING_START_RE.match(text, index)
        if raw_match is not None:
            terminator = ")" + raw_match.group(1) + '"'
            end = text.find(terminator, raw_match.end())
            end = length if end < 0 else end + len(terminator)
            literal = text[index:end]
            output.append(_blank(literal) if remove_literals else literal)
            index = end
            continue

        if text[index] in {'"', "'"}:
            quote = text[index]
            end = index + 1
            while end < length:
                if text[end] == "\\":
                    end = min(length, end + 2)
                    continue
                end += 1
                if text[end - 1] == quote:
                    break
            literal = text[index:end]
            output.append(_blank(literal) if remove_literals else literal)
            index = end
            continue

        output.append(text[index])
        index += 1

    return "".join(output)


def _without_include_directives(text: str) -> str:
    output: list[str] = []
    continuing = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        is_include = continuing or bool(re.match(r"#\s*include\b", stripped))
        if is_include:
            output.append(_blank(line))
            continuing = line.rstrip("\r\n").rstrip().endswith("\\")
        else:
            output.append(line)
            continuing = False
    return "".join(output)


def _cpp_includes(comment_free_text: str) -> list[str]:
    includes: list[str] = []
    for match in CPP_INCLUDE_RE.finditer(comment_free_text):
        includes.append((match.group("angle") or match.group("quote")).strip())
    return includes


def _is_qt_include(include: str) -> bool:
    normalized = include.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    if parts[0].startswith("Qt"):
        return True
    leaf = parts[-1]
    return bool(re.fullmatch(r"Q[A-Z_][A-Za-z0-9_]*", leaf)) or bool(
        re.fullmatch(r"q[a-z0-9_]+\.h", leaf)
    )


def _is_qt_widgets_include(include: str) -> bool:
    normalized = include.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    return parts[0] == "QtWidgets" or parts[-1] in QT_WIDGET_HEADERS


def _is_gui_include(include: str) -> bool:
    normalized = include.replace("\\", "/")
    return "gui" in [
        part.lower()
        for part in normalized.split("/")
        if part not in {"", ".", ".."}
    ]


def _matching_cmake_bracket(text: str, start: int) -> tuple[str, int] | None:
    match = re.match(r"\[(=*)\[", text[start:])
    if match is None:
        return None
    terminator = "]" + match.group(1) + "]"
    end = text.find(terminator, start + match.end())
    return terminator, len(text) if end < 0 else end + len(terminator)


def _strip_cmake_comments(text: str) -> str:
    output: list[str] = []
    index = 0
    length = len(text)
    in_quote = False
    while index < length:
        character = text[index]
        if in_quote:
            output.append(character)
            if character == "\\" and index + 1 < length:
                index += 1
                output.append(text[index])
            elif character == '"':
                in_quote = False
            index += 1
            continue

        if character == '"':
            in_quote = True
            output.append(character)
            index += 1
            continue

        if character == "#":
            bracket = _matching_cmake_bracket(text, index + 1)
            if bracket is not None:
                _, end = bracket
            else:
                end = text.find("\n", index + 1)
                if end < 0:
                    end = length
            output.append(_blank(text[index:end]))
            index = end
            continue

        bracket = _matching_cmake_bracket(text, index)
        if bracket is not None:
            _, end = bracket
            output.append(text[index:end])
            index = end
            continue

        output.append(character)
        index += 1

    return "".join(output)


def _iter_cmake_commands(text: str) -> Iterator[tuple[str, str]]:
    clean = _strip_cmake_comments(text)
    position = 0
    while True:
        match = COMMAND_START_RE.search(clean, position)
        if match is None:
            return
        depth = 1
        index = match.end()
        body_start = index
        in_quote = False
        while index < len(clean) and depth > 0:
            character = clean[index]
            if in_quote:
                if character == "\\":
                    index += 2
                    continue
                if character == '"':
                    in_quote = False
                index += 1
                continue
            if character == '"':
                in_quote = True
                index += 1
                continue
            bracket = _matching_cmake_bracket(clean, index)
            if bracket is not None:
                _, index = bracket
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            index += 1
        if depth != 0:
            return
        yield match.group(1).lower(), clean[body_start : index - 1]
        position = index


def _cmake_tokens(body: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(body):
        while index < len(body) and (body[index].isspace() or body[index] == ";"):
            index += 1
        if index >= len(body):
            break
        if body[index] == '"':
            index += 1
            token: list[str] = []
            while index < len(body):
                if body[index] == "\\" and index + 1 < len(body):
                    token.extend(body[index : index + 2])
                    index += 2
                    continue
                if body[index] == '"':
                    index += 1
                    break
                token.append(body[index])
                index += 1
            tokens.append("".join(token))
            continue
        bracket = _matching_cmake_bracket(body, index)
        if bracket is not None:
            terminator, end = bracket
            opener = re.match(r"\[(=*)\[", body[index:])
            assert opener is not None
            content_start = index + opener.end()
            tokens.append(body[content_start : end - len(terminator)])
            index = end
            continue
        end = index
        while end < len(body) and not body[end].isspace() and body[end] != ";":
            end += 1
        tokens.append(body[index:end])
        index = end
    return tokens


def _target_scoped_values(tokens: Sequence[str]) -> Iterator[tuple[str, str]]:
    visibility = "PLAIN"
    visibility_keywords = {
        "PUBLIC": "PUBLIC",
        "PRIVATE": "PRIVATE",
        "INTERFACE": "INTERFACE",
        "LINK_PUBLIC": "PUBLIC",
        "LINK_PRIVATE": "PRIVATE",
    }
    for token in tokens:
        normalized = token.upper()
        if normalized in visibility_keywords:
            visibility = visibility_keywords[normalized]
            continue
        yield visibility, token


def _contains_gui_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if re.search(r"(?:^|/)(?:src/)?gui(?:/|$)", normalized, re.IGNORECASE):
        return True
    for match in re.finditer(r"\$(?:ENV)?\{([^}]*)\}", value, re.IGNORECASE):
        variable_name = match.group(1).upper()
        if "GUI" in variable_name and any(
            marker in variable_name for marker in ("INCLUDE", "DIR", "PATH")
        ):
            return True
    return False


def _qt_link_references(value: str) -> list[str]:
    target_matches = list(QT_TARGET_RE.finditer(value))
    references = [match.group(1) for match in target_matches]
    for variable_match in QT_LINK_VARIABLE_RE.finditer(value):
        if any(
            target_match.start() <= variable_match.start()
            and variable_match.end() <= target_match.end()
            for target_match in target_matches
        ):
            continue
        references.append(variable_match.group(0))
    return sorted(references)


def _scan_cmake(root: Path, core_root: Path, frozen: _FindingCounter, forbidden: _FindingCounter) -> None:
    parsed: list[tuple[Path, list[tuple[str, list[str]]]]] = []
    library_targets: set[str] = set()

    for path in sorted(core_root.rglob("CMakeLists.txt")):
        commands: list[tuple[str, list[str]]] = []
        for name, body in _iter_cmake_commands(path.read_text(encoding="utf-8")):
            tokens = _cmake_tokens(body)
            commands.append((name, tokens))
            if name == "add_library" and tokens:
                library_targets.add(tokens[0])
        parsed.append((path, commands))

    for path, commands in parsed:
        relative = _repository_path(root, path)
        for name, tokens in commands:
            if name == "include_directories":
                for value in tokens:
                    if _contains_gui_path(value):
                        forbidden.add(
                            "forbidden_gui_include_root",
                            relative,
                            f"directory|PLAIN|{value}",
                        )
                continue
            if len(tokens) < 2 or tokens[0] not in library_targets:
                continue
            target = tokens[0]
            if name == "target_link_libraries":
                for visibility, value in _target_scoped_values(tokens[1:]):
                    for qt_reference in _qt_link_references(value):
                        evidence = f"{target}|{visibility}|{qt_reference}"
                        frozen.add("core_library_qt_link", relative, evidence)
                        if "widgets" in qt_reference.lower():
                            forbidden.add("forbidden_qt_widgets", relative, evidence)
            elif name == "target_include_directories":
                for visibility, value in _target_scoped_values(tokens[1:]):
                    if _contains_gui_path(value):
                        forbidden.add(
                            "forbidden_gui_include_root",
                            relative,
                            f"{target}|{visibility}|{value}",
                        )


def _scan_sources(root: Path, core_root: Path, frozen: _FindingCounter, forbidden: _FindingCounter) -> None:
    paths = sorted(
        path
        for path in core_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES and not _is_test_source(path, core_root)
    )
    for path in paths:
        relative = _repository_path(root, path)
        text = path.read_text(encoding="utf-8")
        comment_free = _sanitize_cpp(text, remove_literals=False)
        code = _without_include_directives(_sanitize_cpp(text, remove_literals=True))
        includes = _cpp_includes(comment_free)

        include_counts = Counter(includes)
        for include, count in sorted(include_counts.items()):
            if _is_gui_include(include):
                forbidden.add("forbidden_gui_include", relative, include, count)
            if _is_qt_widgets_include(include):
                forbidden.add("forbidden_qt_widgets", relative, f"include:{include}", count)

        for symbol, count in sorted(Counter(FORBIDDEN_GUI_TYPE_RE.findall(code)).items()):
            forbidden.add("forbidden_gui_type", relative, symbol, count)

        for token, count in sorted(Counter(DISPLAY_NAME_RE.findall(comment_free)).items()):
            frozen.add("display_name", relative, token, count)

        runtime_counts = Counter(RUNTIME_QT_SYMBOL_RE.findall(code))
        for symbol, count in sorted(runtime_counts.items()):
            frozen.add("runtime_qt_dependency", relative, f"symbol:{symbol}", count)
        for include, count in sorted(include_counts.items()):
            if RUNTIME_QT_SYMBOL_RE.search(include):
                frozen.add("runtime_qt_dependency", relative, f"include:{include}", count)

        if path.suffix.lower() not in HEADER_SUFFIXES:
            continue
        for include, count in sorted(include_counts.items()):
            if _is_qt_include(include):
                frozen.add("public_header_qt_include", relative, include, count)
        for symbol, count in sorted(Counter(QT_SYMBOL_RE.findall(code)).items()):
            frozen.add("public_header_qt_symbol", relative, symbol, count)


def scan_repository(root: Path) -> ScanResult:
    root = root.resolve()
    core_root = root / "src" / "core"
    if not core_root.is_dir():
        raise BoundaryScanError(f"Core source directory does not exist: {core_root}")

    frozen = _FindingCounter()
    forbidden = _FindingCounter()
    _scan_cmake(root, core_root, frozen, forbidden)
    _scan_sources(root, core_root, frozen, forbidden)
    return ScanResult(frozen=frozen.findings(), forbidden=forbidden.findings())


def baseline_document(findings: Iterable[Finding]) -> dict[str, object]:
    grouped: dict[str, dict[str, dict[str, int]]] = {rule: {} for rule in FROZEN_RULES}
    for finding in sorted(findings):
        if finding.rule not in grouped:
            raise BoundaryScanError(f"Unknown frozen rule: {finding.rule}")
        path_entries = grouped[finding.rule].setdefault(finding.path, {})
        if finding.evidence in path_entries:
            raise BoundaryScanError(
                f"Duplicate baseline entry: {finding.rule} {finding.path} {finding.evidence}"
            )
        path_entries[finding.evidence] = finding.count
    return {"schema_version": SCHEMA_VERSION, "frozen_debt": grouped}


def format_baseline(findings: Iterable[Finding]) -> str:
    return json.dumps(baseline_document(findings), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_baseline(path: Path) -> tuple[Finding, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BoundaryScanError(f"Boundary baseline does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise BoundaryScanError(f"Boundary baseline is invalid JSON: {path}: {error}") from error

    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise BoundaryScanError(
            f"Boundary baseline must use schema_version {SCHEMA_VERSION}: {path}"
        )
    grouped = document.get("frozen_debt")
    if not isinstance(grouped, dict) or set(grouped) != set(FROZEN_RULES):
        raise BoundaryScanError(
            "Boundary baseline frozen_debt must contain exactly: " + ", ".join(FROZEN_RULES)
        )

    findings: list[Finding] = []
    for rule in FROZEN_RULES:
        paths = grouped[rule]
        if not isinstance(paths, dict):
            raise BoundaryScanError(f"Boundary baseline rule must be an object: {rule}")
        for relative_path, evidence_counts in paths.items():
            if not isinstance(relative_path, str) or not isinstance(evidence_counts, dict):
                raise BoundaryScanError(f"Invalid path mapping in baseline rule: {rule}")
            if Path(relative_path).is_absolute() or "\\" in relative_path:
                raise BoundaryScanError(f"Baseline path must be relative POSIX form: {relative_path}")
            for evidence, count in evidence_counts.items():
                if not isinstance(evidence, str) or not isinstance(count, int) or count <= 0:
                    raise BoundaryScanError(
                        f"Invalid evidence count in baseline: {rule} {relative_path} {evidence}"
                    )
                findings.append(Finding(rule, relative_path, evidence, count))
    return tuple(sorted(findings))


def compare_with_baseline(result: ScanResult, baseline: Iterable[Finding]) -> BaselineDrift:
    actual = set(result.frozen)
    expected = set(baseline)
    return BaselineDrift(
        forbidden=result.forbidden,
        unexpected=tuple(sorted(actual - expected)),
        missing=tuple(sorted(expected - actual)),
    )


def write_baseline(path: Path, findings: Iterable[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_baseline(findings), encoding="utf-8")


def _format_finding(finding: Finding) -> str:
    count_suffix = f" (count={finding.count})" if finding.count != 1 else ""
    return f"  {finding.rule}: {finding.path}: {finding.evidence}{count_suffix}"


def _print_group(title: str, findings: Sequence[Finding]) -> None:
    if not findings:
        return
    print(title, file=sys.stderr)
    for finding in findings:
        print(_format_finding(finding), file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root, help="Repository root")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=script_path.with_name("core_boundary_baseline.json"),
        help="Committed frozen-debt baseline",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Explicitly replace the baseline with the current frozen debt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = scan_repository(args.root)
        if args.write_baseline:
            if result.forbidden:
                _print_group("Forbidden core boundary violations prevent baseline update:", result.forbidden)
                return 1
            write_baseline(args.baseline, result.frozen)
            print(f"Wrote {len(result.frozen)} frozen boundary entries to {args.baseline}")
            return 0

        baseline = load_baseline(args.baseline)
        drift = compare_with_baseline(result, baseline)
    except (BoundaryScanError, OSError) as error:
        print(f"Core boundary check failed: {error}", file=sys.stderr)
        return 2

    if drift.ok:
        print(
            f"Core boundary check passed: {len(result.frozen)} frozen entries, "
            "0 forbidden dependencies."
        )
        return 0

    _print_group("Forbidden core boundary violations:", drift.forbidden)
    _print_group("Unexpected frozen debt:", drift.unexpected)
    _print_group("Baseline debt no longer present (update explicitly after review):", drift.missing)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
