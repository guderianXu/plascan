# Recovered multi-view depth core

This directory vendors the depth-only production snapshot from
`henu-tw2/metashape_code/生成模型` as observed on 2026-09-06. The upstream working
tree contained uncommitted recovered implementation work, so this is a source
snapshot rather than a Git submodule revision.

Imported unchanged algorithm units:

- `patchmatch.cpp`
- `patchmatch_orchestrator.cpp`
- `patchmatch_store.cpp`
- `gpu_cuda.cu`
- the twelve PTX modules used by camera preparation, PatchMatch, filtering and
  three-level depth voting

`metalign_compat.cpp` is a PlaScan integration shim. It preserves the recovered
camera math and grayscale formula while using PlaScan's OpenCV image decoder,
so it does not claim byte-identical libjpeg-turbo 3.1.2 decoding.

The imported path is intentionally CUDA/d4/Mild/fail-closed. It does not expose
the reference OOC fusion or mesh pipeline.
