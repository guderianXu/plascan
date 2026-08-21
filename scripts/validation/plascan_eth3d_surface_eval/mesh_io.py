"""Minimal deterministic PLY/OBJ readers and triangle-surface sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


_PLY_TYPES = {
    "char": ("i1", 1), "uchar": ("u1", 1),
    "short": ("<i2", 2), "ushort": ("<u2", 2),
    "int": ("<i4", 4), "uint": ("<u4", 4),
    "float": ("<f4", 4), "double": ("<f8", 8),
}


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray


def _ply_header(stream) -> tuple[str, list[tuple[str, int, list]], int]:
    if stream.readline().rstrip() != b"ply":
        raise ValueError("not a PLY file")
    encoding = ""
    elements: list[tuple[str, int, list]] = []
    name = ""
    count = 0
    properties: list = []
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("unterminated PLY header")
        line = raw.decode("ascii").strip()
        parts = line.split()
        if not parts or parts[0] == "comment" or parts[0] == "obj_info":
            continue
        if parts[0] == "format":
            encoding = parts[1]
        elif parts[0] == "element":
            if name:
                elements.append((name, count, properties))
            name, count, properties = parts[1], int(parts[2]), []
        elif parts[0] == "property":
            properties.append(tuple(parts[1:]))
        elif parts[0] == "end_header":
            if name:
                elements.append((name, count, properties))
            return encoding, elements, stream.tell()


def _binary_vertex_dtype(properties: list[tuple[str, ...]]) -> np.dtype:
    fields = []
    for prop in properties:
        if len(prop) != 2 or prop[0] not in _PLY_TYPES:
            raise ValueError("PLY vertex properties must be fixed scalar values")
        fields.append((prop[1], _PLY_TYPES[prop[0]][0]))
    return np.dtype(fields)


def load_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        encoding, elements, _ = _ply_header(stream)
        vertex = next((item for item in elements if item[0] == "vertex"), None)
        if vertex is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        if elements[0][0] != "vertex":
            raise ValueError("vertex must be the first PLY element")
        _, count, properties = vertex
        if encoding == "binary_little_endian":
            values = np.fromfile(stream, dtype=_binary_vertex_dtype(properties), count=count)
            points = np.column_stack((values["x"], values["y"], values["z"]))
        elif encoding == "ascii":
            names = [prop[-1] for prop in properties]
            xyz = [names.index(axis) for axis in ("x", "y", "z")]
            rows = [stream.readline().decode("ascii").split() for _ in range(count)]
            points = np.asarray([[float(row[index]) for index in xyz] for row in rows])
        else:
            raise ValueError(f"unsupported PLY encoding: {encoding}")
    return np.asarray(points, dtype=np.float64)


def load_ply_mesh(path: Path) -> TriangleMesh:
    with path.open("rb") as stream:
        encoding, elements, _ = _ply_header(stream)
        if not elements or elements[0][0] != "vertex":
            raise ValueError("vertex must be the first PLY element")
        _, vertex_count, vertex_properties = elements[0]
        if encoding == "binary_little_endian":
            values = np.fromfile(
                stream, dtype=_binary_vertex_dtype(vertex_properties), count=vertex_count
            )
            vertices = np.column_stack((values["x"], values["y"], values["z"]))
        elif encoding == "ascii":
            names = [prop[-1] for prop in vertex_properties]
            xyz = [names.index(axis) for axis in ("x", "y", "z")]
            rows = [stream.readline().decode("ascii").split() for _ in range(vertex_count)]
            vertices = np.asarray(
                [[float(row[index]) for index in xyz] for row in rows], dtype=np.float64
            )
        else:
            raise ValueError(f"unsupported PLY encoding: {encoding}")
        face = next((item for item in elements if item[0] == "face"), None)
        if face is None:
            raise ValueError(f"PLY has no face element: {path}")
        _, face_count, face_properties = face
        if len(face_properties) != 1 or len(face_properties[0]) != 4:
            raise ValueError("PLY faces must have one list property")
        count_type, index_type = face_properties[0][1], face_properties[0][2]
        if count_type not in _PLY_TYPES or index_type not in _PLY_TYPES:
            raise ValueError("unsupported PLY face index type")
        faces = np.empty((face_count, 3), dtype=np.int64)
        if encoding == "binary_little_endian":
            count_format = {1: "B", 2: "<H", 4: "<I"}[_PLY_TYPES[count_type][1]]
            index_format = {1: "b", 2: "<h", 4: "<i"}[_PLY_TYPES[index_type][1]]
            for index in range(face_count):
                size = struct.unpack(count_format, stream.read(_PLY_TYPES[count_type][1]))[0]
                if size != 3:
                    raise ValueError("only triangular PLY faces are supported")
                faces[index] = [
                    struct.unpack(index_format, stream.read(_PLY_TYPES[index_type][1]))[0]
                    for _ in range(3)
                ]
        else:
            for index in range(face_count):
                row = stream.readline().decode("ascii").split()
                if not row or int(row[0]) != 3:
                    raise ValueError("only triangular PLY faces are supported")
                faces[index] = [int(value) for value in row[1:4]]
    return TriangleMesh(np.asarray(vertices, dtype=np.float64), faces)


def load_obj_mesh(path: Path) -> TriangleMesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) != 3:
                    raise ValueError("only triangular OBJ faces are supported")
                face = []
                for token in tokens:
                    value = int(token.split("/", 1)[0])
                    face.append(value - 1 if value > 0 else len(vertices) + value)
                faces.append(face)
    return TriangleMesh(
        np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    )


def load_triangle_mesh(path: Path) -> TriangleMesh:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        mesh = load_ply_mesh(path)
    elif suffix == ".obj":
        mesh = load_obj_mesh(path)
    else:
        raise ValueError(f"unsupported mesh extension: {suffix}")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3 or not mesh.faces.size:
        raise ValueError(f"expected a non-empty triangle mesh: {path}")
    if np.min(mesh.faces) < 0 or np.max(mesh.faces) >= len(mesh.vertices):
        raise ValueError("mesh face index is outside the vertex array")
    return mesh


def sample_surface(mesh: TriangleMesh, count: int, seed: int) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 0.0)
    if not np.any(valid):
        raise ValueError("mesh has no finite positive-area faces")
    triangles = triangles[valid]
    probabilities = areas[valid] / np.sum(areas[valid])
    generator = np.random.default_rng(seed)
    chosen = generator.choice(len(triangles), size=count, p=probabilities)
    random = generator.random((count, 2))
    root = np.sqrt(random[:, 0])
    weights = np.column_stack((1.0 - root, root * (1.0 - random[:, 1]), root * random[:, 1]))
    return np.einsum("ni,nij->nj", weights, triangles[chosen])
