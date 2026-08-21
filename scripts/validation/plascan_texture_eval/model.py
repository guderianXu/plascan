"""Minimal strict OBJ/MTL loader for textured-model quality evaluation."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class Material:
    name: str
    texture_path: Path
    texture_linear_rgb: np.ndarray


@dataclass(frozen=True)
class TexturedMesh:
    source_path: Path
    material_library_paths: tuple[Path, ...]
    vertices: np.ndarray
    texcoords: np.ndarray
    faces: np.ndarray
    face_texcoords: np.ndarray
    face_materials: np.ndarray
    materials: tuple[Material, ...]


def _resolve_index(token: str, count: int, description: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"Invalid {description} index: {token}") from exc
    index = value - 1 if value > 0 else count + value
    if value == 0 or index < 0 or index >= count:
        raise ValueError(f"{description} index is out of range: {token}")
    return index


def _map_texture_token(tokens: list[str]) -> str:
    if len(tokens) < 2:
        raise ValueError("map_Kd is missing its texture path")
    option_widths = {
        "-blendu": 1,
        "-blendv": 1,
        "-boost": 1,
        "-bm": 1,
        "-clamp": 1,
        "-imfchan": 1,
        "-mm": 2,
        "-o": 3,
        "-s": 3,
        "-t": 3,
        "-texres": 1,
        "-type": 1,
    }
    index = 1
    while index < len(tokens) and tokens[index].startswith("-"):
        width = option_widths.get(tokens[index])
        if width is None:
            raise ValueError(f"Unsupported map_Kd option: {tokens[index]}")
        index += width + 1
    if index >= len(tokens):
        raise ValueError("map_Kd is missing its texture path")
    return " ".join(tokens[index:])


def _load_material_library(path: Path) -> dict[str, Material]:
    if not path.is_file():
        raise FileNotFoundError(f"MTL file does not exist: {path}")
    texture_by_name: dict[str, Path] = {}
    current_name = ""
    with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = shlex.split(line, comments=True, posix=True)
            if not tokens:
                continue
            if tokens[0] == "newmtl":
                current_name = " ".join(tokens[1:]).strip()
                if not current_name:
                    raise ValueError(f"Empty newmtl at {path}:{line_number}")
            elif tokens[0] == "map_Kd":
                if not current_name:
                    raise ValueError(f"map_Kd before newmtl at {path}:{line_number}")
                texture_by_name[current_name] = (path.parent / _map_texture_token(tokens)).resolve()

    materials: dict[str, Material] = {}
    for name, texture_path in texture_by_name.items():
        if not texture_path.is_file():
            raise FileNotFoundError(f"Texture for material '{name}' does not exist: {texture_path}")
        with Image.open(texture_path) as image:
            srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        materials[name] = Material(name, texture_path, srgb_to_linear(srgb))
    return materials


def load_textured_obj(path: Path) -> TexturedMesh:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OBJ file does not exist: {path}")

    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_texcoords: list[list[int]] = []
    face_material_names: list[str] = []
    materials_by_name: dict[str, Material] = {}
    material_library_paths: list[Path] = []
    current_material = ""

    with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = shlex.split(line, comments=True, posix=True)
            if not tokens:
                continue
            tag = tokens[0]
            if tag == "v":
                if len(tokens) < 4:
                    raise ValueError(f"Short vertex at {path}:{line_number}")
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tag == "vt":
                if len(tokens) < 3:
                    raise ValueError(f"Short texture coordinate at {path}:{line_number}")
                texcoords.append([float(tokens[1]), float(tokens[2])])
            elif tag == "mtllib":
                for library_name in tokens[1:]:
                    library_path = (path.parent / library_name).resolve()
                    library = _load_material_library(library_path)
                    duplicate = set(materials_by_name).intersection(library)
                    if duplicate:
                        raise ValueError(f"Duplicate material names: {sorted(duplicate)}")
                    materials_by_name.update(library)
                    material_library_paths.append(library_path)
            elif tag == "usemtl":
                current_material = " ".join(tokens[1:]).strip()
            elif tag == "f":
                if len(tokens) < 4:
                    raise ValueError(f"Face has fewer than three corners at {path}:{line_number}")
                corner_vertices: list[int] = []
                corner_texcoords: list[int] = []
                for corner in tokens[1:]:
                    fields = corner.split("/")
                    if len(fields) < 2 or not fields[1]:
                        raise ValueError(f"Face has no texture coordinate at {path}:{line_number}")
                    corner_vertices.append(_resolve_index(fields[0], len(vertices), "vertex"))
                    corner_texcoords.append(_resolve_index(fields[1], len(texcoords), "texture"))
                for offset in range(1, len(corner_vertices) - 1):
                    faces.append([corner_vertices[0], corner_vertices[offset], corner_vertices[offset + 1]])
                    face_texcoords.append(
                        [corner_texcoords[0], corner_texcoords[offset], corner_texcoords[offset + 1]]
                    )
                    face_material_names.append(current_material)

    if not vertices or not texcoords or not faces:
        raise ValueError(f"OBJ must contain vertices, texture coordinates, and faces: {path}")
    if not materials_by_name:
        raise ValueError(f"OBJ has no MTL material with map_Kd: {path}")

    material_names = list(materials_by_name)
    material_index = {name: index for index, name in enumerate(material_names)}
    if len(material_names) == 1:
        face_material_names = [name or material_names[0] for name in face_material_names]
    unknown = sorted({name for name in face_material_names if name not in material_index})
    if unknown:
        raise ValueError(f"Faces reference materials without map_Kd: {unknown}")

    vertex_array = np.asarray(vertices, dtype=np.float64)
    texcoord_array = np.asarray(texcoords, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    face_texcoord_array = np.asarray(face_texcoords, dtype=np.int64)
    if not np.all(np.isfinite(vertex_array)) or not np.all(np.isfinite(texcoord_array)):
        raise ValueError(f"OBJ contains non-finite geometry or UV values: {path}")
    if (
        np.any(face_array[:, 0] == face_array[:, 1])
        or np.any(face_array[:, 1] == face_array[:, 2])
        or np.any(face_array[:, 2] == face_array[:, 0])
    ):
        raise ValueError(f"OBJ contains a face with repeated vertex indices: {path}")

    return TexturedMesh(
        source_path=path,
        material_library_paths=tuple(material_library_paths),
        vertices=vertex_array,
        texcoords=texcoord_array,
        faces=face_array,
        face_texcoords=face_texcoord_array,
        face_materials=np.asarray([material_index[name] for name in face_material_names], dtype=np.int32),
        materials=tuple(materials_by_name[name] for name in material_names),
    )
