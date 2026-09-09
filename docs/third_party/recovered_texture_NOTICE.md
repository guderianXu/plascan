# Recovered Natural texture boundary notice

`3rdparty/recovered_texture` contains a clean-room subset derived from the local
authorized compatibility study at `E:/code/metashape_reverse/生成纹理`.  The
subset was originally synchronized from revision
`a6a7f6ae82e603621d6a52f41c5e8807f44c12c8`; its public
`PipelineInput::precise_output_uv` contract was updated through reference
revision `6310ce5ff20f47fd6bdbf0b1400dc671a81959ad`.  It is not a complete mirror
of that revision.  The study's top-level README states that it is for authorized
local compatibility research and contains no vendor binary.  It does **not**
provide a top-level LICENSE file; therefore the clean-room source's
redistribution/license status remains subject to project-owner review and is
not represented here as a third-party open-source license grant.

The vendored `xatlas` subtree is separately MIT licensed.  Its complete license
text is retained at `3rdparty/recovered_texture/xatlas/LICENSE` and must remain
with any distribution that includes that subtree.

PlaScan currently builds only `plascan_recovered_texture_core`: recovered CPU
kernels and Natural UV geometry routines.  The reference Vulkan path, five
RGBA32F atlas attachments, project float32 ABI, and pinned TIFF/JPEG page codec
are not shipped or silently substituted by this target.
