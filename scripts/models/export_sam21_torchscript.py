#!/usr/bin/env python3
"""Export SAM 2.1 encoder/decoder TorchScript modules for PlaScan.

The C++ GUI loads two modules:
  - sam21_hiera_<variant>_encoder_<device>.pt
  - sam21_hiera_<variant>_decoder_<device>.pt

The exported decoder is optimized for PlaScan's automatic mask workflow:
full-image box prompt, optional point prompts, optional low-res mask input, and
CPU/CUDA execution through LibTorch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    torch = None
    F = None
    TORCH_IMPORT_ERROR = exc

    class _MissingTorchModule:
        pass

    class _MissingNN:
        Module = _MissingTorchModule

    nn = _MissingNN()
else:
    TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class Sam21Variant:
    token: str
    config: str
    checkpoint_name: str


VARIANTS: dict[str, Sam21Variant] = {
    "tiny": Sam21Variant(
        token="tiny",
        config="configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint_name="sam2.1_hiera_tiny.pt",
    ),
    "small": Sam21Variant(
        token="small",
        config="configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint_name="sam2.1_hiera_small.pt",
    ),
    "base_plus": Sam21Variant(
        token="base_plus",
        config="configs/sam2.1/sam2.1_hiera_b+.yaml",
        checkpoint_name="sam2.1_hiera_base_plus.pt",
    ),
    "large": Sam21Variant(
        token="large",
        config="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_name="sam2.1_hiera_large.pt",
    ),
}


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "Missing PyTorch. Use the PlaScan Python runtime or install torch before exporting SAM2.1 models."
        ) from TORCH_IMPORT_ERROR


class Sam21EncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_size: int) -> None:
        super().__init__()
        self.model = model
        scale = input_size // 1024
        if scale <= 0 or input_size % 1024 != 0:
            self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        else:
            self._bb_feat_sizes = [(256 * scale, 256 * scale),
                                   (128 * scale, 128 * scale),
                                   (64 * scale, 64 * scale)]

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        batch_size = image.shape[0]
        feats = [
            feat.permute(1, 2, 0).reshape(batch_size, -1, feat_size[0], feat_size[1])
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return feats[-1], feats[0], feats[1]


class Sam21DecoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_size: int) -> None:
        super().__init__()
        self.model = model
        self.input_size = int(input_size)

    @staticmethod
    def _flag(value: torch.Tensor) -> bool:
        return bool(int(value.reshape(-1)[0].item()) != 0)

    def _predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.model.sam_mask_decoder
        token_offset = 0
        if decoder.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    decoder.obj_score_token.weight,
                    decoder.iou_token.weight,
                    decoder.mask_tokens.weight,
                ],
                dim=0,
            )
            token_offset = 1
        else:
            output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)

        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = image_embeddings.expand(tokens.shape[0], -1, -1, -1)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(tokens.shape[0], -1, -1, -1)
        batch_size, channel_count, height, width = src.shape

        hidden_states, src = decoder.transformer(src, pos_src, tokens)
        iou_token_out = hidden_states[:, token_offset, :]
        mask_tokens_out = hidden_states[
            :, token_offset + 1 : (token_offset + 1 + decoder.num_mask_tokens), :
        ]

        src = src.transpose(1, 2).reshape(batch_size, channel_count, height, width)
        if not decoder.use_high_res_features:
            upscaled_embedding = decoder.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = decoder.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: list[torch.Tensor] = []
        for i in range(decoder.num_mask_tokens):
            hyper_in_list.append(decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        batch_size, channel_count, height, width = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.reshape(batch_size, channel_count, height * width)).reshape(
            batch_size, -1, height, width
        )

        iou_predictions = decoder.iou_prediction_head(iou_token_out)
        if decoder.pred_obj_scores:
            object_score_logits = decoder.pred_obj_score_head(hidden_states[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_predictions.new_ones(iou_predictions.shape[0], 1)
        return masks, iou_predictions, mask_tokens_out, object_score_logits

    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_0: torch.Tensor,
        high_res_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box: torch.Tensor,
        has_box: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        multimask_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concat_points = None
        if point_coords.numel() > 0:
            concat_points = (point_coords, point_labels.to(torch.int32))

        if self._flag(has_box):
            box_coords = box.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=box.device)
            box_labels = box_labels.repeat(box_coords.shape[0], 1)
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        mask_arg = mask_input if self._flag(has_mask_input) else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_arg,
        )

        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        low_res_masks, iou_predictions, _, _ = self._predict_masks(
            image_embed,
            self.model.sam_prompt_encoder.get_dense_pe().to(image_embed.device),
            sparse_embeddings,
            dense_embeddings,
            batched_mode,
            [high_res_0, high_res_1],
        )
        if self._flag(multimask_output):
            low_res_masks = low_res_masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        else:
            low_res_masks = low_res_masks[:, 0:1, :, :]
            iou_predictions = iou_predictions[:, 0:1]

        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return high_res_masks, iou_predictions, torch.clamp(low_res_masks, -32.0, 32.0)


def parse_devices(raw: str) -> list[str]:
    require_torch()
    if raw == "auto":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices
    devices = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(devices) - {"cpu", "cuda"})
    if invalid:
        raise ValueError(f"invalid device(s): {', '.join(invalid)}")
    return devices


def load_sam2_model(config: str, checkpoint: Path, device: torch.device) -> nn.Module:
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError(
            "Missing SAM2 package. Install facebookresearch/sam2 in this Python environment first."
        ) from exc

    model = build_sam2(config, str(checkpoint), device=device, mode="eval", apply_postprocessing=False)
    model.eval()
    return model


def export_for_device(args: argparse.Namespace,
                      variant: Sam21Variant,
                      checkpoint: Path,
                      device_name: str) -> tuple[Path, Path]:
    require_torch()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() is false")

    device = torch.device(device_name)
    model = load_sam2_model(args.config or variant.config, checkpoint, device)
    encoder = Sam21EncoderWrapper(model, args.input_size).to(device).eval()
    decoder = Sam21DecoderWrapper(model, args.input_size).to(device).eval()

    example_image = torch.zeros(1, 3, args.input_size, args.input_size, dtype=torch.float32, device=device)
    with torch.no_grad():
        example_embed, example_high_0, example_high_1 = encoder(example_image)

    point_coords = torch.empty(1, 0, 2, dtype=torch.float32, device=device)
    point_labels = torch.empty(1, 0, dtype=torch.int32, device=device)
    box = torch.tensor([[0.0, 0.0, float(args.input_size), float(args.input_size)]],
                       dtype=torch.float32,
                       device=device)
    has_box = torch.tensor([1], dtype=torch.int64, device=device)
    mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float32, device=device)
    has_mask_input = torch.tensor([0], dtype=torch.int64, device=device)
    multimask_output = torch.tensor([1], dtype=torch.int64, device=device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / f"sam21_hiera_{variant.token}_encoder_{device_name}.pt"
    decoder_path = output_dir / f"sam21_hiera_{variant.token}_decoder_{device_name}.pt"

    with torch.no_grad():
        traced_encoder = torch.jit.trace(encoder, example_image, strict=False)
        traced_decoder = torch.jit.trace(
            decoder,
            (example_embed,
             example_high_0,
             example_high_1,
             point_coords,
             point_labels,
             box,
             has_box,
             mask_input,
             has_mask_input,
             multimask_output),
            strict=False,
        )

    traced_encoder.save(str(encoder_path))
    traced_decoder.save(str(decoder_path))
    return encoder_path, decoder_path


def resolve_checkpoint(raw: str | None, checkpoint_dir: Path, variant: Sam21Variant) -> Path:
    if raw:
        path = Path(raw).expanduser().resolve()
    else:
        path = (checkpoint_dir / variant.checkpoint_name).resolve()
    if not path.exists():
        raise FileNotFoundError(f"SAM2.1 checkpoint not found: {path}")
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SAM2.1 TorchScript models for PlaScan")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="tiny")
    parser.add_argument(
        "--checkpoint",
        help="Path to the SAM2.1 checkpoint. Defaults to --checkpoint-dir variant file.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("resources/models"))
    parser.add_argument("--config", help="SAM2.1 config path; defaults to the official config for the variant.")
    parser.add_argument("--devices", default="auto", help="auto, cpu, cuda, or cpu,cuda")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("resources/models"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")

    variant = VARIANTS[args.variant]
    checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_dir, variant)
    devices = parse_devices(args.devices)

    for device_name in devices:
        encoder_path, decoder_path = export_for_device(args, variant, checkpoint, device_name)
        print(f"exported {device_name}: {encoder_path}")
        print(f"exported {device_name}: {decoder_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
